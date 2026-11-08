// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the attention hand-off against RFC-0022 §5.3 and RFC-0040 §4.2: the header, its
//! generated Rust and Metal versions, the layer layout against the header's macros as a C and a
//! C++ compiler see them, and the empty side's encoding.

use std::path::{Path, PathBuf};
use std::process::Command;

use proptest::prelude::*;

use super::layer::{Flag, HandoffError, HandoffLayout, HandoffShape, Partial};
use super::mirror::{self, ATTN_HANDOFF_ABI_VERSION, ATTN_LINE, AttnDoorbell, AttnFlag};
use super::msl;
use crate::step::cgen::{self, ATTN_HANDOFF};

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn header() -> cgen::Header {
    let text = std::fs::read_to_string(crate_path(ATTN_HANDOFF.header)).unwrap();
    cgen::parse_named(ATTN_HANDOFF.header, &text).unwrap()
}

/// `generated` is what is on disk at `rel`, ignoring CRLF line ends (a Git for Windows
/// checkout). `SYNTROPY_BLESS_GENERATED=1` rewrites it first.
fn assert_current(rel: &str, generated: &str) {
    let path = crate_path(rel);
    if std::env::var_os("SYNTROPY_BLESS_GENERATED").is_some() {
        std::fs::write(&path, generated).unwrap();
    }
    let on_disk = std::fs::read_to_string(&path)
        .unwrap()
        .replace("\r\n", "\n");
    assert!(
        on_disk == generated,
        "{rel} is stale: run `SYNTROPY_BLESS_GENERATED=1 cargo test -p syntropy-kernels \
         attn_handoff::`"
    );
}

// ---- The header and its generated versions ------------------------------------------------

/// The Rust mirror is what the generator makes of the header today.
#[test]
fn mirror_is_generated_from_the_header() {
    assert_current(
        ATTN_HANDOFF.mirror,
        &cgen::generate_for(&header(), &ATTN_HANDOFF),
    );
}

/// The Metal version is what the printer makes of the header today, with each struct's layout
/// as `static_assert`s and the flag's cache-line alignment.
#[test]
fn msl_version_is_generated_from_the_header() {
    let header = header();
    let msl = msl::generate(&header);
    assert_current(msl::MSL_PATH, &msl);
    for s in &header.structs {
        assert!(msl.contains(&format!("struct {} {{", s.c_name)));
        assert!(msl.contains(&format!("static_assert(sizeof({}) == {}", s.c_name, s.size)));
    }
    assert!(msl.contains("    alignas(64) uint value;\n"));
    assert!(msl.contains("constant uint SYN_ATTN_LINE = 64u;"));
}

/// The header's structs: the doorbell is 16 bytes, and a flag fills one cache line.
#[test]
fn the_header_lays_out_the_doorbell_and_a_flag_on_its_own_line() {
    let header = header();
    let tags: Vec<(&str, usize, usize)> = header
        .structs
        .iter()
        .map(|s| (s.c_name.as_str(), s.size, s.align))
        .collect();
    assert_eq!(
        tags,
        [("syn_attn_doorbell", 16, 4), ("syn_attn_flag", 64, 64)]
    );
    assert_eq!(ATTN_HANDOFF_ABI_VERSION, 1);
    assert_eq!(ATTN_LINE, 64);
    assert_eq!(size_of::<AttnDoorbell>(), 16);
    assert_eq!(size_of::<AttnFlag>(), ATTN_LINE as usize);
    assert_eq!(align_of::<AttnFlag>(), ATTN_LINE as usize);
}

/// The Rust compiler's layout of each mirror equals the C layout the generator computes from
/// the header.
#[test]
fn rust_layout_matches_the_header() {
    let header = header();
    assert_eq!(header.structs.len(), mirror::LAYOUTS.len());
    for (c, rust) in header.structs.iter().zip(mirror::LAYOUTS) {
        let tag = rust.tag;
        assert_eq!(c.c_name, tag);
        assert_eq!(c.size, rust.size, "{tag} size");
        assert_eq!(c.align, rust.align, "{tag} align");
        let c_fields: Vec<(&str, usize)> = c
            .fields
            .iter()
            .map(|f| (f.name.as_str(), f.offset))
            .collect();
        assert_eq!(c_fields, rust.fields.to_vec(), "{tag} members");
    }
}

// ---- The C compiler check --------------------------------------------------------------

/// A C or C++ compiler: `$CC`/`$CXX`, else `cc`/`c++`. In CI (`CI` set) a missing one fails
/// the calling test instead of skipping it.
fn compiler(env: &str, default: &str) -> Option<String> {
    let cmd = std::env::var(env).unwrap_or_else(|_| default.to_owned());
    let found = Command::new(&cmd)
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success());
    if found.is_none() {
        assert!(
            std::env::var_os("CI").is_none(),
            "no {default} compiler found in CI (set {env} to one)"
        );
        eprintln!("skipped: no {default} compiler found (set {env} to use one)");
    }
    found.map(|_| cmd)
}

/// Compiles `source` with `-fsyntax-only` as C11 and C++11, with warnings as errors. Returns
/// how many compilers ran.
fn compile_syntax(name: &str, source: &str) -> usize {
    let dir = std::env::temp_dir().join(format!(
        "syntropy-attn-handoff-{name}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let mut ran = 0;
    for (env, default, file, std) in [
        ("CC", "cc", "check.c", "-std=c11"),
        ("CXX", "c++", "check.cpp", "-std=c++11"),
    ] {
        let Some(cc) = compiler(env, default) else {
            continue;
        };
        let src = dir.join(file);
        std::fs::write(&src, source).unwrap();
        let out = Command::new(&cc)
            .args([std, "-fsyntax-only", "-Wall", "-Wextra", "-Werror", "-I"])
            .arg(crate_path("include"))
            .arg(&src)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{cc} rejected {name}:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        ran += 1;
    }
    let _ = std::fs::remove_dir_all(&dir);
    ran
}

/// The shapes the macro check covers: decode, a verify window, a sparse kind, several domains,
/// and widths that are not multiples of a cache line.
const SHAPES: [HandoffShape; 5] = [
    // Llama-3.3-70B decode on one NUMA domain: 64 query heads of 128.
    HandoffShape {
        rows: 1,
        n_q_heads: 64,
        d_q: 128,
        d_v: 128,
        selected: 0,
        domains: 1,
    },
    // A verify window of 5 rows on two domains.
    HandoffShape {
        rows: 5,
        n_q_heads: 64,
        d_q: 128,
        d_v: 128,
        selected: 0,
        domains: 2,
    },
    // A sparse kind with 2,048 selected positions per row.
    HandoffShape {
        rows: 4,
        n_q_heads: 16,
        d_q: 192,
        d_v: 128,
        selected: 2048,
        domains: 2,
    },
    // Odd widths, so no part is already a multiple of a cache line.
    HandoffShape {
        rows: 3,
        n_q_heads: 7,
        d_q: 5,
        d_v: 3,
        selected: 1,
        domains: 3,
    },
    HandoffShape {
        rows: 1,
        n_q_heads: 1,
        d_q: 1,
        d_v: 1,
        selected: 0,
        domains: 1,
    },
];

/// `_Static_assert`s of the header's layer macros against [`HandoffLayout`] for each shape.
fn c_macro_check(shapes: &[HandoffShape]) -> String {
    let mut out = String::from(
        "#include \"syntropy/attn_handoff.h\"\n#if defined(__cplusplus)\n\
         #define SYN_ASSERT static_assert\n#else\n\
         #define SYN_ASSERT _Static_assert\n#endif\n",
    );
    for (i, &shape) in shapes.iter().enumerate() {
        let layout = HandoffLayout::new(shape).unwrap();
        let HandoffShape {
            rows,
            n_q_heads: h,
            d_q,
            d_v,
            selected: sel,
            domains,
        } = shape;
        out.push_str(&format!(
            "SYN_ASSERT(SYN_ATTN_SELECTED_OFFSET() == {}u, \"selected {i}\");\n\
             SYN_ASSERT(SYN_ATTN_QUERY_OFFSET({rows}u, {sel}u) == {}u, \"query {i}\");\n\
             SYN_ASSERT(SYN_ATTN_LAYER_SIZE({rows}u, {sel}u, {h}u, {d_q}u, {d_v}u, {domains}u) == \
             {}u, \"size {i}\");\n",
            HandoffLayout::selected_offset(),
            layout.query_offset(),
            layout.size(),
        ));
        for d in 0..domains {
            out.push_str(&format!(
                "SYN_ASSERT(SYN_ATTN_DOMAIN_OFFSET({rows}u, {sel}u, {h}u, {d_q}u, {d_v}u, {d}u) == \
                 {}u, \"ra {i} {d}\");\n\
                 SYN_ASSERT(SYN_ATTN_DOMAIN_OFFSET({rows}u, {sel}u, {h}u, {d_q}u, {d_v}u, {d}u) + \
                 sizeof(struct syn_attn_flag) == {}u, \"qa {i} {d}\");\n\
                 SYN_ASSERT(SYN_ATTN_DOMAIN_OFFSET({rows}u, {sel}u, {h}u, {d_q}u, {d_v}u, {d}u) + \
                 2u * sizeof(struct syn_attn_flag) == {}u, \"partials {i} {d}\");\n",
                layout.flag_offset(d, Flag::Ra).unwrap(),
                layout.flag_offset(d, Flag::Qa).unwrap(),
                layout.partial_offset(d, 0, 0).unwrap(),
            ));
            // Single elements at the corners and in the middle, so a transposed order or a wrong
            // stride differs from the header's.
            for (row, head) in corners(shape) {
                out.push_str(&format!(
                    "SYN_ASSERT(SYN_ATTN_PARTIAL_AT({rows}u, {sel}u, {h}u, {d_q}u, {d_v}u, {d}u, {row}u, {head}u) == {}u, \"partial {i} {d} {row} {head}\");\n",
                    layout.partial_offset(d, row, head).unwrap(),
                ));
            }
        }
        for (row, head) in corners(shape) {
            out.push_str(&format!(
                "SYN_ASSERT(SYN_ATTN_QUERY_AT({rows}u, {sel}u, {h}u, {d_q}u, {row}u, {head}u) == {}u, \"query {i} {row} {head}\");\n",
                layout.query_at(row, head).unwrap(),
            ));
            if sel > 0 {
                for k in [0, sel / 2, sel - 1] {
                    out.push_str(&format!(
                        "SYN_ASSERT(SYN_ATTN_SELECTED_AT({sel}u, {row}u, {k}u) == {}u, \"selected {i} {row} {k}\");\n",
                        layout.selected_at(row, k).unwrap(),
                    ));
                }
            }
        }
    }
    out
}

/// Rows and heads at the corners and in the middle of `shape`.
fn corners(shape: HandoffShape) -> Vec<(u32, u32)> {
    let (r, h) = (shape.rows - 1, shape.n_q_heads - 1);
    let mut out = vec![(0, 0), (r, h), (r, 0), (0, h), (r / 2, h / 2)];
    out.dedup();
    out
}

/// The structs' layouts, and every offset and size the header's macros compute, as a C and a
/// C++ compiler see them: the layout the CPU side and the C++ host pass of a CUDA build compile.
#[test]
fn c_compiler_agrees_when_present() {
    let layout = cgen::c_layout_check_for(ATTN_HANDOFF.include, mirror::LAYOUTS);
    let mut ran = compile_syntax("layout", &layout);
    ran += compile_syntax("macros", &c_macro_check(&SHAPES));
    eprintln!("C layout checks run: {ran}");
}

// ---- The layer layout ------------------------------------------------------------------

/// RFC-0022 §5.3's sizing [estimate for Llama-3.3-70B at T = 1]: the query is
/// 64 × 128 × 4 = 32 KiB, and each domain's partials `rows × n_q_heads × (d_v + 2) × 4`.
#[test]
fn the_parts_have_rfc_0022_s_sizes() {
    let layout = HandoffLayout::new(SHAPES[0]).unwrap();
    assert_eq!(layout.query_bytes(), 32 * 1024);
    assert_eq!(HandoffLayout::selected_offset(), 64);
    // No sparse positions: the query follows the doorbell's line.
    assert_eq!(layout.query_offset(), 64);
    let partials: usize = 64 * (128 + 2) * 4;
    assert_eq!(layout.flag_offset(0, Flag::Ra).unwrap(), 64 + 32 * 1024);
    assert_eq!(
        layout.flag_offset(0, Flag::Qa).unwrap(),
        layout.flag_offset(0, Flag::Ra).unwrap() + 64
    );
    assert_eq!(
        layout.partial_offset(0, 0, 0).unwrap(),
        layout.flag_offset(0, Flag::Ra).unwrap() + 128
    );
    assert_eq!(
        layout.size(),
        64 + 32 * 1024 + 128 + partials.next_multiple_of(64)
    );
    assert_eq!(layout.arena_size(), 2 * layout.size());
    assert_eq!(layout.half_offset(0), 0);
    assert_eq!(layout.half_offset(1), layout.size());
    assert_eq!(layout.half_offset(80), 0);
}

/// Literal byte offsets for a small shape with several rows, heads and domains, worked by hand
/// from the header's order: rows, then heads, then the value; a row stride of the shape's
/// `selected`. A head-major order, or a stride of the doorbell's `selected`, fails here.
#[test]
fn offsets_follow_the_row_major_order_by_hand() {
    let shape = HandoffShape {
        rows: 2,
        n_q_heads: 3,
        d_q: 4,
        d_v: 2,
        selected: 8,
        domains: 2,
    };
    let layout = HandoffLayout::new(shape).unwrap();
    // Doorbell 0..16, line 64; selected 2 × 8 × 4 = 64 bytes at 64; query at 128,
    // 2 × 3 × 4 × 4 = 96 bytes, rounded to 128; domains from 256, each 2 flags (128) and
    // 2 × 3 × (2 + 2) × 4 = 96 bytes of partials rounded to 128: 256 bytes per domain.
    assert_eq!(layout.selected_at(1, 0), Ok(64 + 8 * 4));
    assert_eq!(layout.selected_at(1, 7), Ok(64 + 15 * 4));
    assert_eq!(layout.query_offset(), 128);
    assert_eq!(layout.query_at(1, 2), Ok(128 + 5 * 16));
    assert_eq!(layout.flag_offset(1, Flag::Ra), Ok(512));
    assert_eq!(layout.flag_offset(1, Flag::Qa), Ok(576));
    assert_eq!(layout.partial_offset(0, 0, 1), Ok(384 + 16));
    assert_eq!(layout.partial_offset(1, 1, 0), Ok(640 + 3 * 16));
    assert_eq!(layout.size(), 768);
    // A doorbell of 2 rows with 1 position each: row 1's position sits a full stride of 8 on.
    let mut buf = vec![0u8; layout.size()];
    let doorbell = AttnDoorbell {
        step: 9,
        rows: 2,
        _pad: 0,
        host_pages: 1,
        selected: 1,
    };
    layout.write_doorbell(&doorbell, &[7, 9], &mut buf).unwrap();
    let word = |at: usize| u32::from_le_bytes(buf[at..at + 4].try_into().unwrap());
    assert_eq!((word(64), word(68), word(96)), (7, 0, 9));
    assert_eq!(layout.read_doorbell(&buf).unwrap(), (doorbell, vec![7, 9]));
}

/// Every part starts on a cache line and the parts do not overlap: the doorbell, the positions,
/// the query, and each domain's two flags and its partials, in that order, within the layer.
#[test]
fn every_part_starts_on_a_line_and_none_overlap() {
    for shape in SHAPES {
        let layout = HandoffLayout::new(shape).unwrap();
        let line = ATTN_LINE as usize;
        let mut parts = vec![
            (0, size_of::<AttnDoorbell>()),
            (
                HandoffLayout::selected_offset(),
                shape.rows as usize * shape.selected as usize * 4,
            ),
            (layout.query_offset(), layout.query_bytes()),
        ];
        for d in 0..shape.domains {
            parts.push((
                layout.flag_offset(d, Flag::Ra).unwrap(),
                size_of::<AttnFlag>(),
            ));
            parts.push((
                layout.flag_offset(d, Flag::Qa).unwrap(),
                size_of::<AttnFlag>(),
            ));
            let per = (shape.d_v as usize + 2) * 4;
            parts.push((
                layout.partial_offset(d, 0, 0).unwrap(),
                shape.rows as usize * shape.n_q_heads as usize * per,
            ));
            let last = layout
                .partial_offset(d, shape.rows - 1, shape.n_q_heads - 1)
                .unwrap();
            assert_eq!(
                last + per,
                layout.partial_offset(d, 0, 0).unwrap() + parts.last().unwrap().1
            );
        }
        let mut end = 0;
        for &(at, len) in &parts {
            assert_eq!(at % line, 0, "{shape:?}: a part at {at}");
            assert!(
                at >= end,
                "{shape:?}: a part at {at} overlaps the one before, to {end}"
            );
            end = at + len;
        }
        assert!(end <= layout.size(), "{shape:?}");
        assert_eq!(layout.size() % line, 0);
    }
}

/// RFC-0022 §5.3's empty side: a domain with no pages writes `o = 0`, `m = −∞`, `l = 0` for
/// every row and head, and raises its `Qa` flag. It round-trips through the bytes, bit for bit;
/// the GPU's `Ra` flag is left alone; the consumer's reset is read back.
#[test]
fn the_empty_side_round_trips() {
    for shape in SHAPES {
        let layout = HandoffLayout::new(shape).unwrap();
        let mut buf = vec![0xA5u8; layout.size()];
        for d in 0..shape.domains {
            layout.set_flag(d, Flag::Ra, 0, &mut buf).unwrap();
            layout.set_flag(d, Flag::Qa, 0, &mut buf).unwrap();
        }
        let d = shape.domains - 1;
        layout.write_empty_side(d, &mut buf).unwrap();
        for row in 0..shape.rows {
            for head in 0..shape.n_q_heads {
                let partial = layout.read_partial(d, row, head, &buf).unwrap();
                assert!(
                    partial.is_empty_side(),
                    "{shape:?} {row} {head}: {partial:?}"
                );
                assert_eq!(partial, Partial::empty(shape.d_v));
            }
        }
        assert_eq!(layout.flag(d, Flag::Qa, &buf), Ok(1), "the flag is raised");
        assert_eq!(layout.flag(d, Flag::Ra, &buf), Ok(0));
        // The consumer resets the flag after its `EQ 1` wait.
        layout.set_flag(d, Flag::Qa, 0, &mut buf).unwrap();
        assert_eq!(layout.flag(d, Flag::Qa, &buf), Ok(0));
        // The other domains' partials were not touched.
        if shape.domains > 1 {
            let other = layout.read_partial(0, 0, 0, &buf).unwrap();
            assert!(!other.is_empty_side());
        }
    }
}

proptest! {
    /// A partial written at any domain, row and head reads back bit for bit, and touches only
    /// its own block.
    #[test]
    fn a_partial_round_trips(
        shape_ix in 0usize..SHAPES.len(),
        seed in any::<u64>(),
        values in proptest::collection::vec(any::<u32>(), 130),
    ) {
        let shape = SHAPES[shape_ix];
        let layout = HandoffLayout::new(shape).unwrap();
        let d = (seed % u64::from(shape.domains)) as u32;
        let row = ((seed >> 8) % u64::from(shape.rows)) as u32;
        let head = ((seed >> 16) % u64::from(shape.n_q_heads)) as u32;
        let partial = Partial {
            o: values[..shape.d_v as usize].iter().map(|&b| f32::from_bits(b)).collect(),
            m: f32::from_bits(values[128]),
            l: f32::from_bits(values[129]),
        };
        let mut buf = vec![0u8; layout.size()];
        layout.write_partial(d, row, head, &partial, &mut buf).unwrap();
        let back = layout.read_partial(d, row, head, &buf).unwrap();
        let bits = |p: &Partial| -> Vec<u32> {
            p.o.iter().chain([&p.m, &p.l]).map(|v| v.to_bits()).collect()
        };
        prop_assert_eq!(bits(&back), bits(&partial));
        let at = layout.partial_offset(d, row, head).unwrap();
        let len = (shape.d_v as usize + 2) * 4;
        prop_assert!(buf[..at].iter().chain(&buf[at + len..]).all(|&b| b == 0));
    }

    /// A doorbell with its positions round-trips, whatever the step and page counts.
    #[test]
    fn a_doorbell_round_trips(step in any::<u32>(), host_pages in any::<u32>(), rows in 1u16..=4, sel in 0u32..=8) {
        let shape = HandoffShape { rows: 4, n_q_heads: 2, d_q: 4, d_v: 4, selected: 8, domains: 1 };
        let layout = HandoffLayout::new(shape).unwrap();
        let doorbell = AttnDoorbell { step, rows, _pad: 0, host_pages, selected: sel };
        let positions: Vec<u32> = (0..u32::from(rows) * sel).map(|i| i.wrapping_mul(2_654_435_761)).collect();
        let mut buf = vec![0u8; layout.size()];
        layout.write_doorbell(&doorbell, &positions, &mut buf).unwrap();
        prop_assert_eq!(layout.read_doorbell(&buf).unwrap(), (doorbell, positions));
    }
}

/// Shapes and arguments outside the layer are errors, never a panic or a write elsewhere.
#[test]
fn bad_shapes_and_arguments_are_errors() {
    let zero = HandoffShape {
        d_v: 0,
        ..SHAPES[0]
    };
    assert_eq!(HandoffLayout::new(zero), Err(HandoffError::BadShape(zero)));
    let huge = HandoffShape {
        rows: u32::MAX,
        n_q_heads: u32::MAX,
        d_q: u32::MAX,
        d_v: u32::MAX,
        selected: u32::MAX,
        domains: u32::MAX,
    };
    assert_eq!(HandoffLayout::new(huge), Err(HandoffError::BadShape(huge)));
    let too_many_rows = HandoffShape {
        rows: 65_536,
        ..SHAPES[4]
    };
    assert_eq!(
        HandoffLayout::new(too_many_rows),
        Err(HandoffError::BadShape(too_many_rows))
    );
    // A layer that fits in a `usize` but whose two-layer arena does not: the query alone is
    // 2^15 × 2^20 × 2^26 × 4 = 2^63 bytes.
    let half = HandoffShape {
        rows: 1 << 15,
        n_q_heads: 1 << 20,
        d_q: 1 << 26,
        d_v: 1,
        selected: 0,
        domains: 1,
    };
    if usize::BITS == 64 {
        assert_eq!(HandoffLayout::new(half), Err(HandoffError::BadShape(half)));
    }
    assert!(layout_flag_out_of_range());
    let layout = HandoffLayout::new(SHAPES[1]).unwrap();
    let mut buf = vec![0u8; layout.size()];
    let partial = Partial::empty(128);
    assert_eq!(
        layout.write_partial(2, 0, 0, &partial, &mut buf),
        Err(HandoffError::OutOfRange {
            domain: 2,
            row: 0,
            head: 0
        })
    );
    assert!(layout.read_partial(0, 5, 0, &buf).is_err());
    assert!(layout.read_partial(0, 0, 64, &buf).is_err());
    assert_eq!(
        layout.write_partial(0, 0, 0, &Partial::empty(3), &mut buf),
        Err(HandoffError::WrongWidth { d_v: 128, have: 3 })
    );
    assert_eq!(
        layout.write_empty_side(0, &mut buf[..10]),
        Err(HandoffError::BufferTooSmall {
            need: layout.size(),
            have: 10
        })
    );
    // A dense layer takes no positions; too many rows is refused.
    let doorbell = AttnDoorbell {
        step: 1,
        rows: 6,
        _pad: 0,
        host_pages: 3,
        selected: 0,
    };
    assert!(matches!(
        layout.write_doorbell(&doorbell, &[], &mut buf),
        Err(HandoffError::BadDoorbell { rows: 6, .. })
    ));
    let doorbell = AttnDoorbell {
        rows: 1,
        selected: 1,
        ..doorbell
    };
    assert!(layout.write_doorbell(&doorbell, &[7], &mut buf).is_err());
}

fn layout_flag_out_of_range() -> bool {
    let layout = HandoffLayout::new(SHAPES[0]).unwrap();
    let mut buf = vec![0u8; layout.size()];
    layout.set_flag(1, Flag::Qa, 1, &mut buf)
        == Err(HandoffError::OutOfRange {
            domain: 1,
            row: 0,
            head: 0,
        })
        && layout.selected_at(0, 0).is_err()
}
