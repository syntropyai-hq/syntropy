// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the step descriptor against RFC-0040 §3.5, §3.8 and §4.2 rule 5: the C header and
//! its generated Rust mirror, descriptors written and read back, the KV write rule and the
//! per-parity handshake against a fake device.

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

use proptest::prelude::*;

use super::cgen::{self, CType};
use super::mirror::{self, STEP_DESC_ABI_VERSION, STEP_DESC_ALIGN};
use super::*;

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn header() -> cgen::Header {
    let text = std::fs::read_to_string(crate_path(cgen::HEADER_PATH)).unwrap();
    cgen::parse(&text).unwrap()
}

// ---- The header and its mirror ------------------------------------------------------------

/// The mirror is what the generator makes of the header today.
/// `SYNTROPY_BLESS_GENERATED=1` rewrites it.
#[test]
fn mirror_is_generated_from_the_header() {
    let generated = cgen::generate(&header());
    let path = crate_path(cgen::MIRROR_PATH);
    if std::env::var_os("SYNTROPY_BLESS_GENERATED").is_some() {
        std::fs::write(&path, &generated).unwrap();
    }
    // A CRLF checkout (Git for Windows) must not make an unchanged mirror stale.
    let on_disk = std::fs::read_to_string(&path)
        .unwrap()
        .replace("\r\n", "\n");
    assert!(
        on_disk == generated,
        "{} is stale: run `SYNTROPY_BLESS_GENERATED=1 cargo test -p syntropy-kernels step::`",
        cgen::MIRROR_PATH
    );
}

/// The Rust compiler's layout of each mirror equals the C layout the generator computes from
/// the header: size, alignment, and every member's name and offset in order.
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

/// Writing and reading a descriptor moves every member of both mirrors.
#[test]
fn every_member_is_serialized() {
    for (names, rust) in desc::serialized_fields().iter().zip(mirror::LAYOUTS) {
        let members: Vec<&str> = rust.fields.iter().map(|f| f.0).collect();
        assert_eq!(*names, members, "{}", rust.tag);
    }
}

/// The layouts transcribed by hand from RFC-0040 §3.5's struct: members in its order, at C's
/// natural alignment, the header 64-byte aligned. Independent of the generator, so a
/// generator bug that moves the header's layout and the mirror's together still fails here.
#[test]
fn layouts_match_rfc_0040() {
    const STEP_DESC: [(&str, usize); 25] = [
        ("step", 0),
        ("rows_active", 8),
        ("seqs_active", 12),
        ("moe_epoch", 16),
        ("flags", 20),
        ("off_row_seq", 24),
        ("off_row_pos", 28),
        ("off_row_token", 32),
        ("off_tree_parent", 36),
        ("off_seq_kv_len", 40),
        ("off_seq_page_table", 44),
        ("off_seq_state_slot", 48),
        ("off_seq_sampler", 52),
        ("off_moe_balance", 56),
        ("off_row_kv_write_from", 60),
        ("delta_head", 64),
        ("in_payload", 72),
        ("out_payload", 80),
        ("rows_tap", 88),
        ("tap_k", 92),
        ("off_tap_rows", 96),
        ("tap_flag", 100),
        ("tap_ids", 104),
        ("tap_out", 112),
        ("tap_full", 120),
    ];
    const SAMPLER: [(&str, usize); 4] = [
        ("seed", 0),
        ("stream_hi", 8),
        ("stream_lo", 16),
        ("rng_counter", 24),
    ];
    let layout = |m: &cgen::MirrorLayout| (m.tag, m.size, m.align, m.fields);
    assert_eq!(
        layout(&mirror::LAYOUTS[0]),
        ("syn_step_desc", 128, 64, &STEP_DESC[..])
    );
    assert_eq!(
        layout(&mirror::LAYOUTS[1]),
        ("syn_sampler_state", 32, 8, &SAMPLER[..])
    );
    assert_eq!(size_of::<StepDesc>() % STEP_DESC_ALIGN as usize, 0);
}

#[test]
fn header_constants() {
    assert_eq!(STEP_DESC_ABI_VERSION, 1);
    assert_eq!(STEP_DESC_ALIGN, 64);
    assert_eq!(StepFlags::PREFILL_LAST.bits(), 1);
    assert_eq!(StepFlags::TREE.bits(), 2);
    let header = header();
    let names: Vec<&str> = header.defines.iter().map(|d| d.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "SYN_STEP_DESC_ABI_VERSION",
            "SYN_STEP_DESC_ALIGN",
            "SYN_STEP_FLAG_PREFILL_LAST",
            "SYN_STEP_FLAG_TREE",
        ]
    );
}

/// Finds a C or C++ compiler: `$CC`/`$CXX`, else `cc`/`c++`.
fn compiler(env: &str, default: &str) -> Option<String> {
    let cmd = std::env::var(env).unwrap_or_else(|_| default.to_owned());
    let found = Command::new(&cmd).arg("--version").output().ok()?;
    found.status.success().then_some(cmd)
}

/// When a C or C++ compiler is present, it compiles the header with `_Static_assert`s of the
/// Rust compiler's layout. The other tests do not need one, but CI (`CI` set) must have both:
/// there a missing compiler fails the test instead of skipping it.
#[test]
fn c_compiler_agrees_when_present() {
    let dir = std::env::temp_dir().join(format!("syntropy-step-desc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let check = cgen::c_layout_check(mirror::LAYOUTS);
    let include = crate_path("include");
    let mut ran = 0;
    for (env, default, file, std) in [
        ("CC", "cc", "check.c", "-std=c11"),
        ("CXX", "c++", "check.cpp", "-std=c++11"),
    ] {
        let Some(cc) = compiler(env, default) else {
            assert!(
                std::env::var_os("CI").is_none(),
                "no {default} compiler found in CI (set {env} to one)"
            );
            eprintln!("skipped: no {default} compiler found (set {env} to use one)");
            continue;
        };
        let src = dir.join(file);
        std::fs::write(&src, &check).unwrap();
        let out = Command::new(&cc)
            .arg(std)
            .arg("-fsyntax-only")
            .arg("-Wall")
            .arg("-Werror")
            .arg("-I")
            .arg(&include)
            .arg(&src)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{cc} rejected the layout:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        ran += 1;
    }
    let _ = std::fs::remove_dir_all(&dir);
    eprintln!("C layout checks run: {ran}");
}

#[test]
fn generator_lays_out_by_the_c_rules() {
    let h = cgen::parse(
        "/// A.\n#define SYN_A 3u\n#define SYN_B (1u << 4)\n#define OTHER 9\n\
         #define SYN_F(x) x\n/* block\n comment */\n#include <stdint.h>\n\
         /// Doc.\nstruct syn_pad_test {\n    /// First.\n    uint32_t a;\n    uint64_t b;\n\
         int32_t c;\n};\nstruct syn_wide {\n    SYN_ALIGNAS(32) uint32_t x;\n};\n",
    )
    .unwrap();
    assert_eq!(h.defines.len(), 2);
    assert_eq!((h.defines[0].value, h.defines[0].expr.as_str()), (3, "3"));
    assert_eq!(h.defines[0].docs, ["A."]);
    assert_eq!(
        (h.defines[1].value, h.defines[1].expr.as_str()),
        (16, "1 << 4")
    );
    let s = &h.structs[0];
    assert_eq!(s.rust_name(), "PadTest");
    assert_eq!(s.docs, ["Doc."]);
    assert_eq!(s.fields[0].docs, ["First."]);
    let offs: Vec<_> = s.fields.iter().map(|f| (f.ty, f.offset)).collect();
    assert_eq!(offs, [(CType::U32, 0), (CType::U64, 8), (CType::I32, 16)]);
    assert_eq!((s.size, s.align), (24, 8));
    assert_eq!((h.structs[1].size, h.structs[1].align), (32, 32));
}

#[test]
fn generator_rejects_what_it_cannot_lay_out() {
    for (text, why) in [
        (
            "struct syn_x {\n    uint16_t a;\n};\n",
            "unsupported member type",
        ),
        ("struct syn_x {\n    uint32_t a[4];\n};\n", "no arrays"),
        ("struct syn_x {\n    uint32_t a, b;\n};\n", "no arrays"),
        ("struct syn_x {\n    uint32_t a\n};\n", "expected a member"),
        (
            "struct syn_x {\n    uint32_t a;\n    SYN_ALIGNAS(8) uint32_t b;\n};\n",
            "only on the first member",
        ),
        (
            "struct syn_x {\n    SYN_ALIGNAS(12) uint32_t a;\n};\n",
            "power of two",
        ),
        ("struct syn_x {\n    uint32_t a;\n", "unterminated"),
        ("#define SYN_X banana\n", "unreadable value"),
        ("typedef int foo;\n", "outside the subset"),
        // Review of #1622: a member after an inline block comment was dropped, and
        // `#pragma pack` was skipped.
        (
            "struct syn_x {\n    uint32_t a;\n    /* reserved */ uint64_t pad0;\n};\n",
            "code after a block comment",
        ),
        (
            "struct syn_x {\n    uint32_t a; /* note */\n};\n",
            "a block comment inside a line",
        ),
        (
            "struct syn_x {\n    uint32_t a;\n    /* note\n     */ uint32_t b;\n};\n",
            "code after a block comment",
        ),
        ("/* open\nstruct syn_x {\n", "unterminated block comment"),
        (
            "#pragma pack(push, 4)\nstruct syn_x {\n    uint32_t a;\n};\n",
            "does not read",
        ),
        (
            "struct syn_x {\n#pragma pack(1)\n    uint32_t a;\n};\n",
            "directive inside a struct",
        ),
        (
            "struct syn_x {\n#if 0\n    uint32_t a;\n#endif\n};\n",
            "directive inside a struct",
        ),
        ("#undef SYN_X\n", "does not read"),
        ("#elif 1\n", "does not read"),
        (
            "#ifndef G\n#if 0\nstruct syn_x {\n};\n#endif\n#endif\n",
            "under a condition",
        ),
        (
            "#ifndef G\n#if 0\n#define SYN_X 1\n#endif\n#endif\n",
            "under a condition",
        ),
        ("#endif\n", "unbalanced #endif"),
        ("#ifndef G\n", "unbalanced #if"),
        (
            "struct syn_x {\n    uint32_t a; // note\n};\n",
            "expected a member",
        ),
    ] {
        let err = cgen::parse(text).unwrap_err();
        assert!(err.contains(why), "{text:?}: {err}");
    }
}

// ---- Layout -------------------------------------------------------------------------------

fn limits(max_rows: u32, max_seqs: u32, moe_layers: u32, max_tap_rows: u32) -> StepDescLimits {
    StepDescLimits {
        max_rows,
        max_seqs,
        moe_layers,
        max_tap_rows,
    }
}

#[test]
fn empty_layout_is_the_header() {
    let l = StepDescLayout::new(StepDescLimits::default()).unwrap();
    assert_eq!(l.size(), 128);
    let o = l.offsets();
    for off in [o.off_row_seq, o.off_seq_sampler, o.off_tap_rows] {
        assert_eq!(off, 128);
    }
}

/// R = 3, S = 2, L_moe = 1, R_tap = 1, by hand: five row arrays of 12 bytes from 128, three
/// sequence arrays of 8 from 188, the sampler array at 216 (212 rounded up to 8) for 64, then
/// `moe_balance` at 280 and `tap_rows` at 284, ending at 288, rounded up to 320.
#[test]
fn small_layout_by_hand() {
    let l = StepDescLayout::new(limits(3, 2, 1, 1)).unwrap();
    let o = l.offsets();
    let got = [
        o.off_row_seq,
        o.off_row_pos,
        o.off_row_token,
        o.off_tree_parent,
        o.off_row_kv_write_from,
        o.off_seq_kv_len,
        o.off_seq_page_table,
        o.off_seq_state_slot,
        o.off_seq_sampler,
        o.off_moe_balance,
        o.off_tap_rows,
    ];
    assert_eq!(got, [128, 140, 152, 164, 176, 188, 196, 204, 216, 280, 284]);
    assert_eq!(l.size(), 320);
    assert_eq!(o.step, 0);
    assert_eq!(o.rows_active, 0);
}

#[test]
fn layout_past_4_gib_is_refused() {
    assert_eq!(
        StepDescLayout::new(limits(u32::MAX, 0, 0, 0)),
        Err(StepDescError::TooLarge)
    );
    // Offsets fit, but the size does not.
    assert_eq!(
        StepDescLayout::new(limits(0, 0, 0, u32::MAX / 4)),
        Err(StepDescError::TooLarge)
    );
    assert_eq!(
        StepDescLayout::new(limits(0, u32::MAX / 32, 0, 0)),
        Err(StepDescError::TooLarge)
    );
}

proptest! {
    /// Arrays sit after the header, in order, aligned, without overlap and inside the size.
    #[test]
    fn layout_arrays_are_disjoint_and_aligned(
        r in 0u32..5000, s in 0u32..5000, m in 0u32..300, t in 0u32..5000,
    ) {
        let l = StepDescLayout::new(limits(r, s, m, t)).unwrap();
        let o = l.offsets();
        let arrays = [
            (o.off_row_seq, r, 4),
            (o.off_row_pos, r, 4),
            (o.off_row_token, r, 4),
            (o.off_tree_parent, r, 4),
            (o.off_row_kv_write_from, r, 4),
            (o.off_seq_kv_len, s, 4),
            (o.off_seq_page_table, s, 4),
            (o.off_seq_state_slot, s, 4),
            (o.off_seq_sampler, s, 32),
            (o.off_moe_balance, m, 4),
            (o.off_tap_rows, t, 4),
        ];
        let mut end = 128u64;
        for (off, n, elem) in arrays {
            prop_assert!(u64::from(off) >= end);
            prop_assert_eq!(off % elem.min(8), 0);
            end = u64::from(off) + u64::from(n) * u64::from(elem);
        }
        prop_assert!(end <= u64::from(l.size()));
        prop_assert!(u64::from(l.size()) < end + 64);
        prop_assert_eq!(l.size() % 64, 0);
    }
}

// ---- Descriptors --------------------------------------------------------------------------

/// The largest descriptor the property test writes: R = 4096 rows, the largest prefill bucket
/// of RFC-0012 §2 (`PREFILL_LADDER`); S = 4096, since every sequence in a step has a row;
/// every row tapped; and L_moe = 256 [target: a chosen test bound above the layer count of
/// any planned stage].
fn largest() -> StepDescLimits {
    let rows = *crate::PREFILL_LADDER.last().unwrap();
    limits(rows, rows, 256, rows)
}

/// A descriptor with exactly the given counts and random contents, every index valid.
fn descriptor(
    rows: usize,
    seqs: usize,
    taps: usize,
    moe: usize,
) -> impl Strategy<Value = StepDescriptor> {
    let seqs = if rows > 0 { seqs.max(1) } else { seqs };
    let taps = if rows > 0 { taps } else { 0 };
    (
        any::<(u64, u32, u64, u64, u64, bool, bool)>(),
        prop::collection::vec(any::<[u32; 5]>(), rows),
        prop::collection::vec(any::<([u32; 3], [u64; 4])>(), seqs),
        prop::collection::vec(any::<u32>(), moe),
        prop::collection::vec(any::<u32>(), taps),
        any::<(u32, u32, u64, u64, u64)>(),
    )
        .prop_map(move |(h, rs, ss, moe_balance, ts, t)| {
            let (n_rows, n_seqs) = (rows as u32, seqs.max(1) as u32);
            let mut flags = StepFlags::empty();
            flags.set(StepFlags::PREFILL_LAST, h.5);
            flags.set(StepFlags::TREE, h.6);
            StepDescriptor {
                step: h.0,
                moe_epoch: h.1,
                flags,
                delta_head: h.2,
                in_payload: h.3,
                out_payload: h.4,
                rows: rs
                    .into_iter()
                    .map(|v| RowEntry {
                        seq: v[0] % n_seqs,
                        pos: v[1],
                        token: v[2],
                        tree_parent: (v[3] % (n_rows + 1)) as i32 - 1,
                        kv_write_from: v[4],
                    })
                    .collect(),
                seqs: ss
                    .into_iter()
                    .map(|(v, s)| SeqEntry {
                        kv_len: v[0],
                        page_table: v[1],
                        state_slot: v[2],
                        sampler: SamplerState {
                            seed: s[0],
                            stream_hi: s[1],
                            stream_lo: s[2],
                            rng_counter: s[3],
                        },
                    })
                    .collect(),
                moe_balance,
                tap_rows: ts.into_iter().map(|v| v % n_rows.max(1)).collect(),
                tap: TapFields {
                    k: t.0,
                    flag: t.1,
                    ids: t.2,
                    out: t.3,
                    full: t.4,
                },
            }
        })
}

/// Random limits, then a descriptor with random counts inside them.
fn layout_and_descriptor() -> impl Strategy<Value = (StepDescLayout, StepDescriptor)> {
    (1u32..64, 1u32..64, 0u32..8, 0u32..64)
        .prop_flat_map(|(r, s, m, t)| {
            let l = StepDescLayout::new(limits(r, s, m, t)).unwrap();
            (Just(l), 0..=r as usize, 1..=s as usize, 0..=t as usize)
        })
        .prop_flat_map(|(l, r, s, t)| {
            let m = l.limits().moe_layers as usize;
            (Just(l), descriptor(r, s, t.min(r), m))
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    /// RFC-0040 §3.5: descriptors at the largest sizes, every array full, written into a
    /// mirror and read back identically.
    #[test]
    fn largest_descriptors_round_trip(d in descriptor(4096, 4096, 4096, 256)) {
        let l = StepDescLayout::new(largest()).unwrap();
        let mut mirror = vec![0u8; l.size() as usize];
        d.write(&l, &mut mirror).unwrap();
        prop_assert_eq!(StepDescriptor::read(&l, &mirror).unwrap(), d);
    }
}

proptest! {
    #[test]
    fn descriptors_round_trip((l, d) in layout_and_descriptor(), fill in any::<u8>()) {
        // Whatever the mirror held before, reading gives back what was written.
        let mut mirror = vec![fill; l.size() as usize];
        d.write(&l, &mut mirror).unwrap();
        let back = StepDescriptor::read(&l, &mirror).unwrap();
        prop_assert_eq!(&back, &d);
        // Rewriting a smaller descriptor over a larger one reads back the smaller.
        let mut smaller = d.clone();
        smaller.rows.clear();
        smaller.tap_rows.clear();
        smaller.write(&l, &mut mirror).unwrap();
        prop_assert_eq!(StepDescriptor::read(&l, &mirror).unwrap(), smaller);
    }

    /// A row's attention writes no KV below its `row_kv_write_from` (RFC-0040 §3.5), and
    /// writes its position when it is at or above it. In a replay step, KV that survived is
    /// never overwritten; in a normal step every row writes.
    #[test]
    fn kv_is_never_written_below_kv_write_from(
        (_, d) in layout_and_descriptor(),
        replay in prop::collection::vec(0u32..4, 64),
    ) {
        let mut d = d;
        for (row, back) in d.rows.iter_mut().zip(&replay) {
            row.pos %= 1 << 20;
            row.kv_write_from = row.pos + 2 - back.min(&2);
            if *back == 3 {
                row.kv_write_from = row.pos.saturating_sub(1);
            }
        }
        // KV that survived: every (seq, pos) below each row's write-from, marked as old.
        let mut kv: BTreeMap<(u32, u32), &str> = BTreeMap::new();
        for row in &d.rows {
            for pos in row.kv_write_from.saturating_sub(3)..row.kv_write_from {
                kv.insert((row.seq, pos), "survived");
            }
        }
        let survived = kv.clone();
        let writes: Vec<_> = kv_writes(&d).collect();
        for &(r, seq, pos) in &writes {
            let row = d.rows[r];
            prop_assert_eq!((seq, pos), (row.seq, row.pos));
            prop_assert!(pos >= row.kv_write_from);
            kv.insert((seq, pos), "written");
        }
        for (r, row) in d.rows.iter().enumerate() {
            let wrote = writes.iter().any(|w| w.0 == r);
            prop_assert_eq!(wrote, row.pos >= row.kv_write_from);
        }
        for (key, _) in survived {
            let below_every_writer = d
                .rows
                .iter()
                .filter(|row| (row.seq, row.pos) == key)
                .all(|row| row.pos < row.kv_write_from);
            if below_every_writer {
                prop_assert_eq!(kv[&key], "survived");
            }
        }
        // In a normal step every row writes.
        let mut normal = d.clone();
        for row in &mut normal.rows {
            row.kv_write_from = row.pos;
        }
        prop_assert_eq!(kv_writes(&normal).count(), normal.rows.len());
    }
}

fn sample(l: &StepDescLayout) -> StepDescriptor {
    let rows = (0..l.limits().max_rows)
        .map(|i| RowEntry {
            seq: i % l.limits().max_seqs,
            pos: 10 + i,
            token: 100 + i,
            tree_parent: i as i32 - 1,
            kv_write_from: 10 + i,
        })
        .collect();
    let seqs = (0..l.limits().max_seqs)
        .map(|i| SeqEntry {
            kv_len: 7 + i,
            page_table: 3 * i,
            state_slot: i,
            sampler: SamplerState {
                seed: 0x5eed,
                stream_hi: 1,
                stream_lo: 2 + u64::from(i),
                rng_counter: 9,
            },
        })
        .collect();
    StepDescriptor {
        step: 0x0102_0304_0506_0708,
        moe_epoch: 4,
        flags: StepFlags::TREE,
        delta_head: 0xdead_beef,
        in_payload: 0x1000,
        out_payload: 0x2000,
        rows,
        seqs,
        moe_balance: vec![4096; l.limits().moe_layers as usize],
        tap_rows: vec![0; l.limits().max_tap_rows as usize],
        tap: TapFields {
            k: 8,
            flag: 17,
            ids: 0x3000,
            out: 0x4000,
            full: 0,
        },
    }
}

/// Bytes at known offsets: little-endian, each value where the header and the layout put it.
#[test]
fn bytes_land_where_the_header_says() {
    let l = StepDescLayout::new(limits(3, 2, 1, 1)).unwrap();
    let d = sample(&l);
    let mut m = vec![0u8; l.size() as usize];
    d.write(&l, &mut m).unwrap();
    let u32_at = |at: usize| u32::from_le_bytes(m[at..at + 4].try_into().unwrap());
    let u64_at = |at: usize| u64::from_le_bytes(m[at..at + 8].try_into().unwrap());
    assert_eq!(m[0..8], [8, 7, 6, 5, 4, 3, 2, 1]);
    assert_eq!((u32_at(8), u32_at(12)), (3, 2));
    assert_eq!((u32_at(16), u32_at(20)), (4, StepFlags::TREE.bits()));
    assert_eq!(u32_at(24), 128);
    assert_eq!(u64_at(64), 0xdead_beef);
    assert_eq!((u64_at(72), u64_at(80)), (0x1000, 0x2000));
    assert_eq!(
        (u32_at(88), u32_at(92), u32_at(96), u32_at(100)),
        (1, 8, 284, 17)
    );
    assert_eq!((u64_at(104), u64_at(112), u64_at(120)), (0x3000, 0x4000, 0));
    // row_seq, row_pos, row_token, tree_parent (-1 for row 0), row_kv_write_from.
    assert_eq!([u32_at(128), u32_at(132), u32_at(136)], [0, 1, 0]);
    assert_eq!(
        [u32_at(140), u32_at(152), u32_at(164), u32_at(176)],
        [10, 100, u32::MAX, 10]
    );
    assert_eq!(u32_at(168), 0);
    // seq_kv_len, seq_page_table, seq_state_slot of sequence 1, then its sampler state.
    assert_eq!([u32_at(192), u32_at(200), u32_at(208)], [8, 3, 1]);
    let s1 = 216 + 32;
    assert_eq!(
        [u64_at(s1), u64_at(s1 + 8), u64_at(s1 + 16), u64_at(s1 + 24)],
        [0x5eed, 1, 3, 9]
    );
    assert_eq!((u32_at(280), u32_at(284)), (4096, 0));
}

#[test]
fn write_refuses_and_leaves_the_mirror_alone() {
    let l = StepDescLayout::new(limits(3, 2, 1, 1)).unwrap();
    let good = sample(&l);
    let refused = |d: &StepDescriptor, want: StepDescError| {
        let mut m = vec![0xa5u8; l.size() as usize];
        assert_eq!(d.write(&l, &mut m), Err(want));
        assert!(m.iter().all(|&b| b == 0xa5));
    };

    let mut d = good.clone();
    d.rows.push(RowEntry::default());
    refused(
        &d,
        StepDescError::OverLimit {
            what: "rows_active",
            count: 4,
            limit: 3,
        },
    );
    let mut d = good.clone();
    d.seqs.push(SeqEntry::default());
    refused(
        &d,
        StepDescError::OverLimit {
            what: "seqs_active",
            count: 3,
            limit: 2,
        },
    );
    let mut d = good.clone();
    d.tap_rows.push(0);
    refused(
        &d,
        StepDescError::OverLimit {
            what: "rows_tap",
            count: 2,
            limit: 1,
        },
    );
    let mut d = good.clone();
    d.moe_balance.clear();
    refused(&d, StepDescError::MoeBalanceLength { have: 0, need: 1 });
    let mut d = good.clone();
    d.rows[1].seq = 2;
    refused(
        &d,
        StepDescError::BadIndex {
            what: "row_seq",
            at: 1,
            value: 2,
        },
    );
    let mut d = good.clone();
    d.rows[2].tree_parent = 3;
    refused(
        &d,
        StepDescError::BadIndex {
            what: "tree_parent",
            at: 2,
            value: 3,
        },
    );
    let mut d = good.clone();
    d.rows[0].tree_parent = -2;
    refused(
        &d,
        StepDescError::BadIndex {
            what: "tree_parent",
            at: 0,
            value: -2,
        },
    );
    let mut d = good.clone();
    d.tap_rows[0] = 3;
    refused(
        &d,
        StepDescError::BadIndex {
            what: "tap_rows",
            at: 0,
            value: 3,
        },
    );

    let mut short = vec![0u8; l.size() as usize - 1];
    assert_eq!(
        good.write(&l, &mut short),
        Err(StepDescError::MirrorTooSmall {
            need: 320,
            have: 319
        })
    );
    // Boundaries: exactly at each limit, and a larger mirror, are accepted.
    let mut big = vec![0u8; l.size() as usize + 64];
    good.write(&l, &mut big).unwrap();
    assert_eq!(StepDescriptor::read(&l, &big).unwrap(), good);
}

#[test]
fn read_refuses_foreign_or_corrupt_mirrors() {
    let l = StepDescLayout::new(limits(3, 2, 1, 1)).unwrap();
    let mut m = vec![0u8; l.size() as usize];
    sample(&l).write(&l, &mut m).unwrap();

    // Written for other limits.
    let other = StepDescLayout::new(limits(3, 2, 2, 1)).unwrap();
    let mut m2 = vec![0u8; other.size() as usize];
    m2[..m.len()].copy_from_slice(&m);
    assert_eq!(
        StepDescriptor::read(&other, &m2),
        Err(StepDescError::LayoutMismatch)
    );
    // A mirror of zeros has no offsets.
    assert_eq!(
        StepDescriptor::read(&l, &vec![0u8; l.size() as usize]),
        Err(StepDescError::LayoutMismatch)
    );
    // Undefined flag bits.
    let mut bad = m.clone();
    bad[20] |= 0x80;
    assert_eq!(
        StepDescriptor::read(&l, &bad),
        Err(StepDescError::UnknownFlags(0x82))
    );
    // A corrupt count is refused before any array is read.
    for (at, what, limit) in [
        (8, "rows_active", 3),
        (12, "seqs_active", 2),
        (88, "rows_tap", 1),
    ] {
        let mut bad = m.clone();
        bad[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            StepDescriptor::read(&l, &bad),
            Err(StepDescError::OverLimit {
                what,
                count: u32::MAX as usize,
                limit
            })
        );
    }
    // A corrupt index.
    let mut bad = m.clone();
    bad[128..132].copy_from_slice(&5u32.to_le_bytes());
    assert_eq!(
        StepDescriptor::read(&l, &bad),
        Err(StepDescError::BadIndex {
            what: "row_seq",
            at: 0,
            value: 5
        })
    );
    assert!(matches!(
        StepDescriptor::read(&l, &m[..100]),
        Err(StepDescError::MirrorTooSmall {
            need: 320,
            have: 100
        })
    ));
}

#[test]
fn errors_display() {
    let e = StepDescError::BadIndex {
        what: "row_seq",
        at: 1,
        value: 2,
    };
    assert_eq!(e.to_string(), "row_seq[1] = 2 names no entry of this step");
    assert_eq!(
        HandshakeError::StepNotDone(Parity::Odd).to_string(),
        "O[1] of the previous use is not consumed"
    );
    assert_eq!(
        HandshakeError::from(StepDescError::TooLarge).to_string(),
        "step descriptor: step descriptor limits need 4 GiB or more"
    );
}

// ---- The parity handshake against a fake device -------------------------------------------

/// A device that runs captured step graphs the way RFC-0040 §3.5 and §4.2 describe, on one
/// compute stream in order: each replay of parity `p` copies mirror `p`, raises `DC[p]`,
/// computes, and raises `O[p]`. Each of those is a separate [`tick`](Self::tick), so the host
/// can act between them.
///
/// It records a violation whenever the host writes a mirror or resubmits a parity whose
/// previous use is not finished and fully consumed (`DC[p]` and `O[p]` raised and reset to 0),
/// whenever the bytes it copies are not the ones submitted, and whenever it raises a flag the
/// host has not reset (a stale 1).
struct FakeDevice<'a> {
    layout: StepDescLayout,
    mirrors: [Vec<u8>; 2],
    bells: &'a Doorbells,
    /// Queued replays: the parity and its mirror's bytes at submission.
    queue: VecDeque<(Parity, Vec<u8>)>,
    /// The replay that has copied its descriptor and not yet raised `O`.
    running: Option<Parity>,
    /// Parities queued or running.
    in_use: [bool; 2],
    /// Step numbers of the descriptors copied, in stream order.
    copied: Vec<u64>,
    done: usize,
    violations: Vec<String>,
}

/// The doorbell slots of `DC[0]`, `DC[1]`, `O[0]` and `O[1]`: what a provider maps from its
/// pinned doorbell page. Owned apart from the device so the handshake can borrow them.
struct Doorbells {
    dc: [AtomicU32; 2],
    o: [AtomicU32; 2],
}

impl Doorbells {
    /// Fresh slots at 0: the handshake arms them at 1 when it binds them.
    fn new() -> Self {
        Self {
            dc: [AtomicU32::new(0), AtomicU32::new(0)],
            o: [AtomicU32::new(0), AtomicU32::new(0)],
        }
    }

    fn flags(&self) -> ParityFlags<'_> {
        ParityFlags {
            dc: [&self.dc[0], &self.dc[1]],
            o: [&self.o[0], &self.o[1]],
        }
    }

    fn handshake(&self) -> ParityHandshake<'_> {
        ParityHandshake::new(self.flags()).unwrap()
    }
}

impl<'a> FakeDevice<'a> {
    fn new(layout: StepDescLayout, bells: &'a Doorbells) -> Self {
        let size = layout.size() as usize;
        Self {
            layout,
            mirrors: [vec![0; size], vec![0; size]],
            bells,
            queue: VecDeque::new(),
            running: None,
            in_use: [false; 2],
            copied: Vec::new(),
            done: 0,
            violations: Vec::new(),
        }
    }

    /// Checks that parity `p`'s previous use is over and both its flags are back at 0.
    fn check_free(&mut self, p: Parity, action: &str) {
        let i = p.index();
        if self.in_use[i] {
            self.violations
                .push(format!("{action} {i} while its previous use runs"));
        }
        for (name, flag) in [("DC", &self.bells.dc[i]), ("O", &self.bells.o[i])] {
            if flag.load(Ordering::Acquire) != 0 {
                self.violations
                    .push(format!("{action} {i} before {name}[{i}] was consumed"));
            }
        }
    }

    /// The host wrote mirror `p`.
    fn wrote(&mut self, p: Parity) {
        self.check_free(p, "rewrite");
    }

    /// The host replayed parity `p`'s graph instance.
    fn submit(&mut self, p: Parity) {
        self.check_free(p, "resubmit");
        self.in_use[p.index()] = true;
        let snapshot = self.mirrors[p.index()].clone();
        self.queue.push_back((p, snapshot));
    }

    fn raise(flag: &AtomicU32, name: &str, p: Parity, violations: &mut Vec<String>) {
        if flag.load(Ordering::Acquire) != 0 {
            violations.push(format!("{name}[{}] raised while still 1", p.index()));
        }
        flag.store(1, Ordering::Release);
    }

    /// Advances the stream by one event. Returns whether anything happened.
    fn tick(&mut self) -> bool {
        if let Some(p) = self.running.take() {
            Self::raise(&self.bells.o[p.index()], "O", p, &mut self.violations);
            self.in_use[p.index()] = false;
            self.done += 1;
            return true;
        }
        let Some((p, snapshot)) = self.queue.pop_front() else {
            return false;
        };
        let mirror = &self.mirrors[p.index()];
        if *mirror != snapshot {
            self.violations
                .push(format!("mirror {} rewritten before its copy", p.index()));
        }
        match StepDescriptor::read(&self.layout, mirror) {
            Ok(d) => self.copied.push(d.step),
            Err(e) => self.violations.push(format!("unreadable descriptor: {e}")),
        }
        Self::raise(&self.bells.dc[p.index()], "DC", p, &mut self.violations);
        self.running = Some(p);
        true
    }
}

/// A host step: what the dispatcher might try next.
#[derive(Clone, Copy, Debug)]
enum HostAction {
    Write(Parity),
    Submit(Parity),
    ConsumeDc(Parity),
    ConsumeO(Parity),
    Tick,
}

fn host_action() -> impl Strategy<Value = HostAction> {
    let parity = prop_oneof![Just(Parity::Even), Just(Parity::Odd)];
    (0u8..5, parity).prop_map(|(a, p)| match a {
        0 => HostAction::Write(p),
        1 => HostAction::Submit(p),
        2 => HostAction::ConsumeDc(p),
        3 => HostAction::ConsumeO(p),
        _ => HostAction::Tick,
    })
}

/// Runs `actions` through the handshake against the fake device. Every action the handshake
/// refuses is skipped, as a dispatcher would retry it later. Returns the device and the step
/// numbers of the descriptors submitted, in order.
fn drive<'a>(actions: &[HostAction], bells: &'a Doorbells) -> (FakeDevice<'a>, Vec<u64>) {
    let layout = StepDescLayout::new(limits(4, 2, 1, 1)).unwrap();
    let mut hs = bells.handshake();
    let mut dev = FakeDevice::new(layout, bells);
    let mut written: [Option<u64>; 2] = [None, None];
    let mut submitted = Vec::new();
    let mut next_step = 0u64;
    for &a in actions {
        match a {
            HostAction::Write(p) => {
                let mut d = sample(&layout);
                d.step = next_step;
                let i = p.index();
                if hs.write(p, &d, &layout, &mut dev.mirrors[i]).is_ok() {
                    dev.wrote(p);
                    written[i] = Some(next_step);
                    next_step += 1;
                }
            }
            HostAction::Submit(p) => {
                if hs.submit(p).is_ok() {
                    dev.submit(p);
                    submitted.push(written[p.index()].take().unwrap());
                }
            }
            HostAction::ConsumeDc(p) => {
                let _ = hs.consume_dc(p);
            }
            HostAction::ConsumeO(p) => {
                let _ = hs.consume_o(p);
            }
            HostAction::Tick => {
                dev.tick();
            }
        }
    }
    (dev, submitted)
}

proptest! {
    /// RFC-0040 §4.2 rule 5, with the fake device: under any interleaving of host actions
    /// and device progress, a parity is never rewritten or resubmitted before the host has
    /// consumed and reset its `DC[p]` and `O[p]` from the previous use, no flag is raised
    /// while still 1, and the device copies every submitted descriptor unchanged, in order.
    #[test]
    fn parity_is_never_reused_before_both_flags_are_consumed(
        actions in prop::collection::vec(host_action(), 0..400),
    ) {
        let bells = Doorbells::new();
        let (mut dev, submitted) = drive(&actions, &bells);
        while dev.tick() {}
        prop_assert!(dev.violations.is_empty(), "{:?}", dev.violations);
        prop_assert_eq!(&dev.copied, &submitted);
        prop_assert_eq!(dev.done, submitted.len());
    }
}

/// A dispatcher that keeps two steps queued (throughput mode) completes every step, with no
/// violation: the handshake never blocks progress that the protocol allows.
#[test]
fn two_steps_in_flight_make_progress() {
    let mut actions = Vec::new();
    for _ in 0..1000 {
        for p in Parity::BOTH {
            actions.extend([
                HostAction::ConsumeDc(p),
                HostAction::ConsumeO(p),
                HostAction::Write(p),
                HostAction::Submit(p),
            ]);
        }
        actions.extend([HostAction::Tick; 3]);
    }
    let bells = Doorbells::new();
    let (mut dev, submitted) = drive(&actions, &bells);
    while dev.tick() {}
    assert!(dev.violations.is_empty(), "{:?}", dev.violations);
    assert!(submitted.len() >= 1000, "{} steps", submitted.len());
    assert_eq!(dev.copied, submitted);
    assert!(submitted.windows(2).all(|w| w[0] < w[1]));
}

/// The fake catches a host that skips the handshake, so the property test above is not
/// vacuous.
#[test]
fn the_fake_device_catches_a_host_that_skips_the_handshake() {
    let layout = StepDescLayout::new(limits(4, 2, 1, 1)).unwrap();
    let d = sample(&layout);

    // Resubmitting without consuming the flags at 1 from the start.
    let bells = Doorbells::new();
    let _armed = bells.handshake();
    let mut dev = FakeDevice::new(layout, &bells);
    d.write(&layout, &mut dev.mirrors[0]).unwrap();
    dev.submit(Parity::Even);
    while dev.tick() {}
    assert!(dev.violations.iter().any(|v| v.contains("before DC[0]")));
    assert!(
        dev.violations
            .iter()
            .any(|v| v.contains("DC[0] raised while still 1"))
    );

    // Rewriting a queued mirror before the device copied it.
    let bells = Doorbells::new();
    let mut dev = FakeDevice::new(layout, &bells);
    d.write(&layout, &mut dev.mirrors[1]).unwrap();
    dev.submit(Parity::Odd);
    let mut d2 = d.clone();
    d2.step += 1;
    d2.write(&layout, &mut dev.mirrors[1]).unwrap();
    dev.wrote(Parity::Odd);
    dev.tick();
    assert!(dev.violations.iter().any(|v| v.contains("rewrite 1 while")));
    assert!(
        dev.violations
            .iter()
            .any(|v| v.contains("rewritten before its copy"))
    );
}

#[test]
fn handshake_refusals() {
    let layout = StepDescLayout::new(limits(4, 2, 1, 1)).unwrap();
    let d = sample(&layout);
    let dev = Doorbells::new();
    let mut m = vec![0u8; layout.size() as usize];
    let mut hs = dev.handshake();
    let p = Parity::Even;

    // Both flags start at 1 and are owed: nothing may be written or submitted yet.
    assert!(!hs.is_free(p));
    assert_eq!(
        hs.write(p, &d, &layout, &mut m),
        Err(HandshakeError::DescriptorNotConsumed(p))
    );
    assert!(m.iter().all(|&b| b == 0));
    assert_eq!(hs.submit(p), Err(HandshakeError::DescriptorNotConsumed(p)));
    assert_eq!(hs.consume_dc(p), Ok(true));
    assert_eq!(dev.dc[0].load(Ordering::Acquire), 0);
    assert_eq!(hs.consume_dc(p), Err(HandshakeError::NothingPending(p)));
    assert_eq!(
        hs.write(p, &d, &layout, &mut m),
        Err(HandshakeError::StepNotDone(p))
    );
    assert_eq!(hs.submit(p), Err(HandshakeError::StepNotDone(p)));
    assert_eq!(hs.consume_o(p), Ok(true));
    assert!(hs.is_free(p));
    // The other parity is untouched.
    assert!(!hs.is_free(p.other()));

    assert_eq!(hs.submit(p), Err(HandshakeError::NotWritten(p)));
    let mut bad = d.clone();
    bad.moe_balance.clear();
    assert_eq!(
        hs.write(p, &bad, &layout, &mut m),
        Err(HandshakeError::Descriptor(
            StepDescError::MoeBalanceLength { have: 0, need: 1 }
        ))
    );
    assert_eq!(hs.submit(p), Err(HandshakeError::NotWritten(p)));
    hs.write(p, &d, &layout, &mut m).unwrap();
    hs.submit(p).unwrap();
    // Submitted: owed again; a flag still at 0 is not consumed.
    assert_eq!(hs.consume_dc(p), Ok(false));
    assert_eq!(hs.consume_o(p), Ok(false));
    assert_eq!(hs.submit(p), Err(HandshakeError::DescriptorNotConsumed(p)));
    // Submitted once, so a second submit needs a second write too.
    dev.dc[0].store(1, Ordering::Release);
    dev.o[0].store(1, Ordering::Release);
    assert_eq!(hs.consume_o(p), Ok(true));
    assert_eq!(hs.consume_dc(p), Ok(true));
    assert_eq!(hs.submit(p), Err(HandshakeError::NotWritten(p)));
}

/// After a capture or an abandoned step every flag is re-armed at 1 and owed again.
#[test]
fn rearm_restores_the_start_state() {
    let bells = Doorbells::new();
    let mut hs = bells.handshake();
    // Binding arms all four at 1.
    for f in bells.dc.iter().chain(&bells.o) {
        assert_eq!(f.load(Ordering::Acquire), 1);
    }
    for p in Parity::BOTH {
        hs.consume_dc(p).unwrap();
        hs.consume_o(p).unwrap();
    }
    assert!(Parity::BOTH.iter().all(|&p| hs.is_free(p)));
    hs.rearm();
    for f in bells.dc.iter().chain(&bells.o) {
        assert_eq!(f.load(Ordering::Acquire), 1);
    }
    for p in Parity::BOTH {
        assert!(!hs.is_free(p));
        assert_eq!(hs.consume_dc(p), Ok(true));
        assert_eq!(hs.consume_o(p), Ok(true));
    }
    assert_eq!(Parity::Even.other(), Parity::Odd);
    assert_eq!(Parity::Odd.other().index(), 0);
}

/// The flags are bound once, by name. Binding refuses two names for one slot, and each consume
/// reads only its own flag, so `O[p]` raised alone never frees `DC[p]`.
#[test]
fn flags_cannot_be_swapped_or_shared() {
    let bells = Doorbells::new();
    for flags in [
        ParityFlags {
            dc: [&bells.o[0], &bells.dc[1]],
            o: [&bells.o[0], &bells.o[1]],
        },
        ParityFlags {
            dc: [&bells.dc[0], &bells.dc[0]],
            o: [&bells.o[0], &bells.o[1]],
        },
        ParityFlags {
            dc: [&bells.dc[0], &bells.dc[1]],
            o: [&bells.o[1], &bells.o[1]],
        },
    ] {
        assert_eq!(
            ParityHandshake::new(flags).err(),
            Some(HandshakeError::AliasedFlags)
        );
    }
    // A refused binding stores nothing.
    for f in bells.dc.iter().chain(&bells.o) {
        assert_eq!(f.load(Ordering::Acquire), 0);
    }

    let mut hs = bells.handshake();
    let p = Parity::Odd;
    hs.consume_dc(p).unwrap();
    hs.consume_o(p).unwrap();
    let layout = StepDescLayout::new(limits(4, 2, 1, 1)).unwrap();
    let mut m = vec![0u8; layout.size() as usize];
    hs.write(p, &sample(&layout), &layout, &mut m).unwrap();
    hs.submit(p).unwrap();
    // Only O[1] rises: DC[1] stays owed, and neither the other parity's flags nor O[1] free it.
    bells.o[1].store(1, Ordering::Release);
    assert_eq!(hs.consume_dc(p), Ok(false));
    assert_eq!(hs.consume_o(p), Ok(true));
    assert_eq!(bells.dc[1].load(Ordering::Acquire), 0);
    assert_eq!(
        hs.write(p, &sample(&layout), &layout, &mut m),
        Err(HandshakeError::DescriptorNotConsumed(p))
    );
    bells.dc[1].store(1, Ordering::Release);
    assert_eq!(hs.consume_dc(p), Ok(true));
    assert!(hs.is_free(p));
    assert_eq!(
        HandshakeError::AliasedFlags.to_string(),
        "two step flags share one slot"
    );
}
