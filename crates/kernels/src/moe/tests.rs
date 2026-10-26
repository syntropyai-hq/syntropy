// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the MoE contract against RFC-0040 §3.4, RFC-0023 §2, §5.4 and §6, and ADR-0404
//! items 4 and 5: the doorbell header and its generated mirror, the split rule against a
//! direct transcription of RFC-0023 §6 and between its Rust and C versions, the tables, and
//! the host mirror of `copy_eligible` under injected version skew.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use proptest::prelude::*;

use super::mirror::{self, MOE_CPU, MOE_DOORBELL_ABI_VERSION, MOE_HIT, MOE_MISS};
use super::spec;
use super::*;
use crate::step::cgen::{self, MOE_DOORBELL};
use Assignment::{Copy, Cpu, Hit};

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn header() -> cgen::Header {
    let text = std::fs::read_to_string(crate_path(MOE_DOORBELL.header)).unwrap();
    cgen::parse_named(MOE_DOORBELL.header, &text).unwrap()
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
        "{rel} is stale: run `SYNTROPY_BLESS_GENERATED=1 cargo test -p syntropy-kernels moe::`"
    );
}

// ---- The header, its mirror and the generated split rule ----------------------------------

/// The mirror is what the generator makes of the header today.
#[test]
fn mirror_is_generated_from_the_header() {
    assert_current(
        MOE_DOORBELL.mirror,
        &cgen::generate_for(&header(), &MOE_DOORBELL),
    );
}

/// The Rust and C versions of the split rule are what the printers make of the one
/// specification today.
#[test]
fn split_rule_is_generated_from_the_specification() {
    let f = spec::assign();
    assert_current(spec::RUST_PATH, &spec::rust(&f));
    assert_current(spec::C_PATH, &spec::c(&f));
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

/// Writing and reading a doorbell layer moves every member of the three mirrors.
#[test]
fn every_member_is_serialized() {
    for (names, rust) in doorbell::serialized_fields().iter().zip(mirror::LAYOUTS) {
        let members: Vec<&str> = rust.fields.iter().map(|f| f.0).collect();
        assert_eq!(*names, members, "{}", rust.tag);
    }
}

/// The layouts transcribed by hand from RFC-0023 §2's structs, at C's natural alignment.
/// Independent of the generator.
#[test]
fn layouts_match_rfc_0023() {
    let layout = |m: &cgen::MirrorLayout| (m.tag, m.size, m.align, m.fields);
    assert_eq!(
        layout(&mirror::LAYOUTS[0]),
        (
            "syn_doorbell_header",
            8,
            4,
            &[("seq", 0), ("n_route", 4), ("n_pred", 6)][..]
        )
    );
    assert_eq!(
        layout(&mirror::LAYOUTS[1]),
        (
            "syn_route_slot",
            8,
            4,
            &[("expert", 0), ("tokens", 2), ("cache_slot", 4)][..]
        )
    );
    assert_eq!(
        layout(&mirror::LAYOUTS[2]),
        (
            "syn_pred_slot",
            8,
            4,
            &[("ahead", 0), ("_pad", 1), ("expert", 2), ("score", 4)][..]
        )
    );
}

#[test]
fn header_constants() {
    assert_eq!(MOE_DOORBELL_ABI_VERSION, 1);
    assert_eq!(MOE_MISS, u32::MAX);
    assert_eq!(MOE_HIT, u32::MAX);
    assert_eq!(MOE_CPU, u32::MAX - 1);
    let names: Vec<String> = header().defines.into_iter().map(|d| d.name).collect();
    assert_eq!(
        names,
        [
            "SYN_MOE_DOORBELL_ABI_VERSION",
            "SYN_MOE_MISS",
            "SYN_MOE_HIT",
            "SYN_MOE_CPU",
        ]
    );
    // The offset macros are skipped, and so is their documentation.
    assert!(header().structs[0].docs[0].starts_with("The fixed header"));
}

/// The generator lays `uint8_t`, `uint16_t` and `float` members out by the C rules, and
/// derives no total equality for a struct with a float.
#[test]
fn generator_lays_out_narrow_and_float_members() {
    let h = cgen::parse_named(
        "t.h",
        "struct syn_n {\n    uint8_t a;\n    uint16_t b;\n    uint8_t c;\n    float d;\n    \
         uint8_t e;\n};\n",
    )
    .unwrap();
    let s = &h.structs[0];
    let offs: Vec<_> = s.fields.iter().map(|f| f.offset).collect();
    assert_eq!(offs, [0, 2, 4, 8, 12]);
    assert_eq!((s.size, s.align), (16, 4));
    let rust = cgen::generate_for(&h, &MOE_DOORBELL);
    assert!(rust.contains("#[derive(Clone, Copy, Debug, Default, PartialEq)]\npub struct N {"));
    assert!(rust.contains("    pub d: f32,\n"));
}

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

/// A scratch directory for one test's C files.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("syntropy-moe-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Compiles `source` with `-fsyntax-only` as C11 and C++11, with warnings as errors. Returns
/// how many compilers ran.
fn compile_syntax(name: &str, source: &str) -> usize {
    let dir = scratch(name);
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

/// `_Static_assert`s of the doorbell's route and pred offsets and its size, as the header's
/// macros compute them, against [`DoorbellLayout`] for each `(MAX_UNION, MAX_PRED)`.
fn c_offset_check(grid: &[(u16, u16)]) -> String {
    let mut out = String::from(
        "#include \"syntropy/moe_doorbell.h\"\n#include \"syntropy/moe_assign.h\"\n\
         #if defined(__cplusplus)\n#define SYN_ASSERT static_assert\n#else\n\
         #define SYN_ASSERT _Static_assert\n#endif\n",
    );
    for &(u, p) in grid {
        let l = DoorbellLayout::new(u, p);
        out.push_str(&format!(
            "SYN_ASSERT(SYN_DOORBELL_ROUTE_OFFSET() == {}u, \"route {u} {p}\");\n\
             SYN_ASSERT(SYN_DOORBELL_PRED_OFFSET({u}u) == {}u, \"pred {u} {p}\");\n\
             SYN_ASSERT(SYN_DOORBELL_LAYER_SIZE({u}u, {p}u) == {}u, \"size {u} {p}\");\n",
            l.route_offset(),
            l.pred_offset(),
            l.size(),
        ));
    }
    out
}

/// The mirrors' sizes, alignments and member offsets as a C and a C++ compiler lay out the
/// header, at the edges of `MAX_UNION` and `MAX_PRED` as well; the generated C split rule
/// compiles cleanly as both.
#[test]
fn c_compiler_agrees_when_present() {
    let layout = cgen::c_layout_check_for(MOE_DOORBELL.include, mirror::LAYOUTS);
    let mut ran = compile_syntax("layout", &layout);
    let edges = [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (u16::MAX, 0),
        (0, u16::MAX),
        (u16::MAX, u16::MAX),
    ];
    ran += compile_syntax("edges", &c_offset_check(&edges));
    eprintln!("C layout checks run: {ran}");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]
    /// The route and pred offsets and the layer's size, as functions of `MAX_UNION` and
    /// `MAX_PRED`, agree between the header's macros and [`DoorbellLayout`] over random grids
    /// of 64 points per case, each grid one C and one C++ compilation.
    #[test]
    fn doorbell_offsets_agree_with_c_over_a_grid(
        grid in proptest::collection::vec((any::<u16>(), any::<u16>()), 64),
    ) {
        let l = grid.iter().map(|&(u, p)| DoorbellLayout::new(u, p));
        for (layout, &(u, p)) in l.zip(&grid) {
            // Header, then 8-byte route entries, then 8-byte pred entries.
            prop_assert_eq!(layout.route_offset(), 8);
            prop_assert_eq!(layout.pred_offset(), 8 + 8 * usize::from(u));
            prop_assert_eq!(layout.size(), 8 + 8 * usize::from(u) + 8 * usize::from(p));
        }
        compile_syntax("grid", &c_offset_check(&grid));
    }
}

// ---- The split rule -----------------------------------------------------------------------

/// splitmix64: the input generator the Rust and C sides of the cross-implementation test
/// share, transcribed in `C_DRIVER`.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// One input of the split rule.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Case {
    route: Vec<RouteSlot>,
    words: Vec<u32>,
    experts: u32,
    split: LayerSplit,
}

impl Case {
    fn bits(&self) -> EligibleBits<'_> {
        EligibleBits::new(&self.words, self.experts)
    }

    /// Drawn the way `C_DRIVER`'s `gen_case` draws it. The regimes cover what the criteria
    /// name: no experts, experts outside the bitmap, repeated experts, token ties (most
    /// counts are 0 to 3), hits and misses, empty and full bitmaps, zero and maximal budgets,
    /// zero and maximal expert sizes, no copy slots and `u32::MAX` of them.
    fn draw(rng: &mut Rng) -> Self {
        let experts = match rng.below(4) {
            0 => 0,
            1 => 1 + rng.below(8),
            2 => 1 + rng.below(300),
            _ => 1 + rng.below(65536),
        } as u32;
        let n = if rng.below(32) == 0 {
            rng.below(300)
        } else {
            rng.below(40)
        } as usize;
        let mut route: Vec<RouteSlot> = Vec::with_capacity(n);
        for i in 0..n {
            let mut expert = if rng.below(8) == 0 || experts == 0 {
                rng.below(65536)
            } else {
                rng.below(u64::from(experts))
            } as u16;
            if i > 0 && rng.below(16) == 0 {
                expert = route[rng.below(i as u64) as usize].expert;
            }
            let tokens = if rng.below(4) == 0 {
                rng.below(65536)
            } else {
                rng.below(4)
            } as u16;
            let cache_slot = if rng.below(2) == 0 {
                MOE_MISS
            } else if rng.below(16) == 0 {
                MOE_MISS - 1
            } else {
                rng.below(1 << 20) as u32
            };
            route.push(RouteSlot {
                expert,
                tokens,
                cache_slot,
            });
        }
        let density = rng.below(4);
        let words = (0..experts.div_ceil(32))
            .map(|_| match density {
                0 => 0,
                1 => u32::MAX,
                2 => rng.next() as u32,
                _ => (rng.next() & rng.next()) as u32,
            })
            .collect();
        let b_copy = match rng.below(4) {
            0 => 0,
            1 => u32::MAX,
            2 => rng.below(1 << 20) as u32,
            _ => rng.next() as u32,
        };
        let expert_bytes = match rng.below(5) {
            0 => 0,
            1 => 1 + rng.below(1 << 16),
            2 => 1 + rng.below(1 << 24),
            3 => rng.next(),
            _ => u64::from(b_copy),
        };
        let copy_slots = match rng.below(4) {
            0 => 0,
            1 => rng.below(8) as u32,
            2 => u32::MAX,
            _ => rng.below(n as u64 + 1) as u32,
        };
        Self {
            route,
            words,
            experts,
            split: LayerSplit {
                b_copy,
                expert_bytes,
                copy_slots,
            },
        }
    }

    /// The digest `C_DRIVER`'s `digest_input` computes, so a drift between the two
    /// generators shows as a different input, not as a different rule.
    fn digest(&self) -> u64 {
        let mut h = Fnv::new();
        h.add(u64::from(self.experts));
        h.add(self.route.len() as u64);
        for r in &self.route {
            h.add(u64::from(r.expert));
            h.add(u64::from(r.tokens));
            h.add(u64::from(r.cache_slot));
        }
        for &w in &self.words {
            h.add(u64::from(w));
        }
        h.add(u64::from(self.split.b_copy));
        h.add(self.split.expert_bytes);
        h.add(u64::from(self.split.copy_slots));
        h.0
    }
}

/// FNV-1a over 64-bit values.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(0xCBF2_9CE4_8422_2325)
    }

    fn add(&mut self, v: u64) {
        self.0 = (self.0 ^ v).wrapping_mul(0x0000_0100_0000_01B3);
    }
}

/// RFC-0023 §6's rule, transcribed directly: "order misses by `tokens` descending, then
/// expert index; walk in that order and assign a miss to the copied set if it is marked in
/// `copy_eligible[l]` at the step's table version, the copied bytes stay at or below the
/// step's `B_copy(l)`, and layer l's half of the copy slots is not yet full; everything else
/// is in-place. The i-th expert of the copied set goes to copy slot i." Repeated experts, which
/// a well-formed doorbell never has, keep their entry order (a stable sort).
fn transcription(c: &Case) -> Vec<Assignment> {
    let mut misses: Vec<usize> = (0..c.route.len())
        .filter(|&i| c.route[i].cache_slot == MOE_MISS)
        .collect();
    misses.sort_by(|&a, &b| {
        let (ra, rb) = (c.route[a], c.route[b]);
        rb.tokens.cmp(&ra.tokens).then(ra.expert.cmp(&rb.expert))
    });
    let mut out: Vec<Assignment> = c
        .route
        .iter()
        .map(|r| {
            if r.cache_slot == MOE_MISS {
                Assignment::Cpu
            } else {
                Assignment::Hit
            }
        })
        .collect();
    let mut copied: u32 = 0;
    let mut bytes: u128 = 0;
    for i in misses {
        let e = u32::from(c.route[i].expert);
        let marked = e < c.experts && (c.words[(e / 32) as usize] >> (e % 32)) & 1 == 1;
        let fits = bytes + u128::from(c.split.expert_bytes) <= u128::from(c.split.b_copy);
        if marked && fits && copied < c.split.copy_slots {
            out[i] = Assignment::Copy(copied);
            copied += 1;
            bytes += u128::from(c.split.expert_bytes);
        }
    }
    out
}

/// The criteria every output must meet on its own, whatever the transcription says.
fn check_invariants(c: &Case, got: &[Assignment]) -> Result<(), String> {
    let mut copies = Vec::new();
    for (r, a) in c.route.iter().zip(got) {
        let hit = r.cache_slot != MOE_MISS;
        match *a {
            Assignment::Hit if !hit => return Err(format!("a miss is HIT: {r:?}")),
            Assignment::Copy(i) => {
                if hit {
                    return Err(format!("a hit is COPY: {r:?}"));
                }
                if !c.bits().get(u32::from(r.expert)) {
                    return Err(format!("a non-eligible miss is COPY: {r:?}"));
                }
                copies.push(i);
            }
            Assignment::Cpu if hit => return Err(format!("a hit is CPU: {r:?}")),
            _ => {}
        }
    }
    let bytes = copies.len() as u128 * u128::from(c.split.expert_bytes);
    if bytes > u128::from(c.split.b_copy) {
        return Err(format!(
            "copied {bytes} bytes over B_copy {}",
            c.split.b_copy
        ));
    }
    copies.sort_unstable();
    if copies.iter().enumerate().any(|(i, &s)| s as usize != i) {
        return Err(format!("copy slots are not 0..k: {copies:?}"));
    }
    if copies.len() as u64 > u64::from(c.split.copy_slots) {
        return Err("more copies than copy slots".into());
    }
    Ok(())
}

/// The input count RFC-0040 §3.4's property test names: 10⁶ [target: COMP-E1-T06's criterion].
const CASES: u64 = 1_000_000;
/// The seed both property runs and the C driver start from.
const SEED: u64 = 0x05EE_D0E1_7006;

/// The generated Rust version equals the direct transcription of RFC-0023 §6 on 10⁶ random
/// inputs, and every output keeps the copied bytes at or below `B_copy` and never copies a
/// non-eligible miss.
#[test]
fn assign_matches_rfc_0023_on_a_million_inputs() {
    let mut rng = Rng(SEED);
    let mut copied = 0u64;
    for case in 0..CASES {
        let c = Case::draw(&mut rng);
        let got = assign(&c.route, c.bits(), c.split).unwrap();
        assert_eq!(got, transcription(&c), "case {case}: {c:?}");
        if let Err(why) = check_invariants(&c, &got) {
            panic!("case {case}: {why}: {c:?}");
        }
        copied += got
            .iter()
            .filter(|a| matches!(a, Assignment::Copy(_)))
            .count() as u64;
    }
    // The inputs reach the copy path, not only the hit and CPU paths.
    eprintln!("{copied} copies in {CASES} cases");
    assert!(copied > CASES / 10, "only {copied} copies in {CASES} cases");
}

fn case_strategy() -> impl Strategy<Value = Case> {
    let route = proptest::collection::vec(
        (0u16..48, 0u16..4, prop_oneof![Just(MOE_MISS), 0u32..64]).prop_map(
            |(expert, tokens, cache_slot)| RouteSlot {
                expert,
                tokens,
                cache_slot,
            },
        ),
        0..64,
    );
    (
        route,
        0u32..48,
        any::<[u32; 2]>(),
        0u32..40,
        0u64..6,
        0u32..12,
    )
        .prop_map(
            |(route, experts, words, b_copy, expert_bytes, copy_slots)| Case {
                route,
                words: words[..experts.div_ceil(32) as usize].to_vec(),
                experts,
                split: LayerSplit {
                    b_copy,
                    expert_bytes,
                    copy_slots,
                },
            },
        )
}

proptest! {
    /// The same property with shrinking, over small inputs dense in ties and repeats.
    #[test]
    fn assign_matches_rfc_0023_shrinking(c in case_strategy()) {
        let got = assign(&c.route, c.bits(), c.split).unwrap();
        prop_assert_eq!(&got, &transcription(&c));
        prop_assert_eq!(check_invariants(&c, &got), Ok(()));
    }

    /// For distinct experts, the decision for each expert does not depend on where the
    /// router wrote its entry: permuting the doorbell permutes the output the same way.
    #[test]
    fn assign_ignores_entry_order(c in case_strategy(), seed in any::<u64>()) {
        let mut seen = std::collections::BTreeSet::new();
        let route: Vec<RouteSlot> =
            c.route.iter().copied().filter(|r| seen.insert(r.expert)).collect();
        let mut perm: Vec<usize> = (0..route.len()).collect();
        let mut rng = Rng(seed);
        for i in (1..perm.len()).rev() {
            perm.swap(i, rng.below(i as u64 + 1) as usize);
        }
        let shuffled: Vec<RouteSlot> = perm.iter().map(|&i| route[i]).collect();
        let a = assign(&route, c.bits(), c.split).unwrap();
        let b = assign(&shuffled, c.bits(), c.split).unwrap();
        for (j, &i) in perm.iter().enumerate() {
            prop_assert_eq!(b[j], a[i]);
        }
    }
}

/// `C_DRIVER` reads the seed, the case count and an output path from its arguments, draws the
/// same cases as [`Case::draw`], runs the generated `syn_moe_assign` and writes, per case, the
/// input digest and an FNV-1a digest of the codes, both little-endian. It writes to a file
/// opened with `"wb"`, not to `stdout`: on Windows `stdout` is a text stream, and the C runtime
/// turns every 0x0A byte of the digests into 0x0D 0x0A.
const C_DRIVER: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include "syntropy/moe_assign.h"

static uint64_t state;
static uint64_t next(void) {
    state += 0x9E3779B97F4A7C15ull;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static uint64_t below(uint64_t n) { return next() % n; }
static uint64_t fnv(uint64_t h, uint64_t v) { return (h ^ v) * 0x00000100000001B3ull; }
static FILE *sink;
static void put64(uint64_t v) {
    unsigned char b[8];
    for (int i = 0; i < 8; i++) b[i] = (unsigned char)(v >> (8 * i));
    if (fwrite(b, 1, 8, sink) != 8) exit(3);
}

static struct syn_route_slot route[300];
static uint32_t words[2048];
static uint32_t out[300];

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    sink = fopen(argv[3], "wb");
    if (sink == NULL) return 3;
    state = strtoull(argv[1], NULL, 10);
    uint64_t cases = strtoull(argv[2], NULL, 10);
    for (uint64_t k = 0; k < cases; k++) {
        uint64_t sel = below(4);
        uint32_t experts = (uint32_t)(sel == 0 ? 0 : sel == 1 ? 1 + below(8)
            : sel == 2 ? 1 + below(300) : 1 + below(65536));
        uint32_t n = (uint32_t)(below(32) == 0 ? below(300) : below(40));
        for (uint32_t i = 0; i < n; i++) {
            uint16_t expert = (uint16_t)((below(8) == 0 || experts == 0)
                ? below(65536) : below(experts));
            if (i > 0 && below(16) == 0) expert = route[below(i)].expert;
            uint16_t tokens = (uint16_t)(below(4) == 0 ? below(65536) : below(4));
            uint32_t slot;
            if (below(2) == 0) slot = SYN_MOE_MISS;
            else if (below(16) == 0) slot = SYN_MOE_MISS - 1u;
            else slot = (uint32_t)below(1u << 20);
            route[i].expert = expert;
            route[i].tokens = tokens;
            route[i].cache_slot = slot;
        }
        uint64_t density = below(4);
        uint32_t n_words = (experts + 31u) / 32u;
        for (uint32_t w = 0; w < n_words; w++) {
            if (density == 0) words[w] = 0;
            else if (density == 1) words[w] = 0xFFFFFFFFu;
            else if (density == 2) words[w] = (uint32_t)next();
            else { uint64_t a = next(); words[w] = (uint32_t)(a & next()); }
        }
        uint64_t bs = below(4);
        uint32_t b_copy = bs == 0 ? 0 : bs == 1 ? 0xFFFFFFFFu
            : bs == 2 ? (uint32_t)below(1u << 20) : (uint32_t)next();
        uint64_t es = below(5);
        uint64_t expert_bytes = es == 0 ? 0 : es == 1 ? 1 + below(1u << 16)
            : es == 2 ? 1 + below(1u << 24) : es == 3 ? next() : (uint64_t)b_copy;
        uint64_t cs = below(4);
        uint32_t copy_slots = cs == 0 ? 0 : cs == 1 ? (uint32_t)below(8)
            : cs == 2 ? 0xFFFFFFFFu : (uint32_t)below((uint64_t)n + 1);

        uint64_t h = 0xCBF29CE484222325ull;
        h = fnv(h, experts);
        h = fnv(h, n);
        for (uint32_t i = 0; i < n; i++) {
            h = fnv(h, route[i].expert);
            h = fnv(h, route[i].tokens);
            h = fnv(h, route[i].cache_slot);
        }
        for (uint32_t w = 0; w < n_words; w++) h = fnv(h, words[w]);
        h = fnv(h, b_copy);
        h = fnv(h, expert_bytes);
        h = fnv(h, copy_slots);
        put64(h);

        syn_moe_assign(route, n, words, experts, b_copy, expert_bytes, copy_slots, out);
        uint64_t o = 0xCBF29CE484222325ull;
        for (uint32_t i = 0; i < n; i++) o = fnv(o, out[i]);
        put64(o);
    }
    return fclose(sink) == 0 ? 0 : 3;
}
"#;

/// The generated C version gives the same codes as the generated Rust version on the 10⁶
/// inputs of [`assign_matches_rfc_0023_on_a_million_inputs`]. Needs a C compiler that links,
/// which CI has.
#[test]
fn c_version_matches_rust_on_a_million_inputs() {
    let Some(cc) = compiler("CC", "cc") else {
        return;
    };
    let dir = scratch("driver");
    let src = dir.join("driver.c");
    let exe = dir.join(format!("driver{}", std::env::consts::EXE_SUFFIX));
    std::fs::write(&src, C_DRIVER).unwrap();
    let built = Command::new(&cc)
        .args(["-std=c11", "-O2", "-Wall", "-Werror", "-I"])
        .arg(crate_path("include"))
        .arg(&src)
        .arg("-o")
        .arg(&exe)
        .output()
        .unwrap();
    assert!(
        built.status.success(),
        "{cc} rejected the driver:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let digests = dir.join("digests.bin");
    let run = Command::new(&exe)
        .arg(SEED.to_string())
        .arg(CASES.to_string())
        .arg(&digests)
        .output()
        .unwrap();
    assert!(run.status.success(), "the driver failed: {:?}", run.status);
    let out = std::fs::read(&digests).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(out.len() as u64, CASES * 16);
    let mut rng = Rng(SEED);
    let mut codes = Vec::new();
    for (case, rec) in out.as_chunks::<16>().0.iter().enumerate() {
        let c = Case::draw(&mut rng);
        let word = |i: usize| u64::from_le_bytes(rec[i..i + 8].try_into().unwrap());
        assert_eq!(word(0), c.digest(), "case {case}: the generators drifted");
        codes.clear();
        codes.resize(c.route.len(), 0);
        assign_into(&c.route, c.bits(), c.split, &mut codes).unwrap();
        let mut h = Fnv::new();
        for &code in &codes {
            h.add(u64::from(code));
        }
        assert_eq!(
            word(8),
            h.0,
            "case {case}: C and Rust differ on {c:?}: Rust {codes:?}"
        );
    }
}

// ---- Edge cases ---------------------------------------------------------------------------

fn slot(expert: u16, tokens: u16, cache_slot: u32) -> RouteSlot {
    RouteSlot {
        expert,
        tokens,
        cache_slot,
    }
}

fn split(b_copy: u32, expert_bytes: u64, copy_slots: u32) -> LayerSplit {
    LayerSplit {
        b_copy,
        expert_bytes,
        copy_slots,
    }
}

/// By hand: misses 7 (3 tokens), 2 (1 token), 5 (3 tokens), 9 (1 token, not eligible) and a
/// hit on 4. The walk is 5, 7, 2, 9; a budget of two experts copies 5 and 7.
#[test]
fn worked_example() {
    let route = [
        slot(7, 3, MOE_MISS),
        slot(4, 2, 11),
        slot(2, 1, MOE_MISS),
        slot(5, 3, MOE_MISS),
        slot(9, 1, MOE_MISS),
    ];
    let words = [1u32 << 2 | 1 << 5 | 1 << 7];
    let bits = EligibleBits::new(&words, 16);
    let got = assign(&route, bits, split(200, 100, 8)).unwrap();
    assert_eq!(got, [Copy(1), Hit, Cpu, Copy(0), Cpu]);
    // A third expert's bytes fit, so 2 is copied too; 9 never is.
    let got = assign(&route, bits, split(300, 100, 8)).unwrap();
    assert_eq!(got, [Copy(1), Hit, Copy(2), Copy(0), Cpu]);
    // The copy slots bound the copied set before the bytes do.
    let got = assign(&route, bits, split(300, 100, 1)).unwrap();
    assert_eq!(got, [Cpu, Hit, Cpu, Copy(0), Cpu]);
}

#[test]
fn ties_go_to_the_lower_expert_then_the_earlier_entry() {
    let all = [u32::MAX];
    let bits = EligibleBits::new(&all, 32);
    let route = [
        slot(9, 2, MOE_MISS),
        slot(3, 2, MOE_MISS),
        slot(6, 2, MOE_MISS),
    ];
    let got = assign(&route, bits, split(u32::MAX, 1, 2)).unwrap();
    assert_eq!(got, [Cpu, Copy(0), Copy(1)]);
    // A repeated expert, which a well-formed doorbell never has, is still decided the same
    // way at every site: the earlier entry first.
    let route = [
        slot(4, 1, MOE_MISS),
        slot(4, 1, MOE_MISS),
        slot(4, 1, MOE_MISS),
    ];
    let got = assign(&route, bits, split(u32::MAX, 1, 2)).unwrap();
    assert_eq!(got, [Copy(0), Copy(1), Cpu]);
}

#[test]
fn zero_experts_and_empty_doorbells() {
    // No route entries.
    assert_eq!(
        assign(&[], EligibleBits::none(), split(100, 1, 4)),
        Ok(vec![])
    );
    // A layer with no eligibility bitmap: every miss is CPU, every hit stays HIT.
    let route = [slot(0, 1, MOE_MISS), slot(1, 1, 3)];
    let got = assign(&route, EligibleBits::none(), split(u32::MAX, 1, 4)).unwrap();
    assert_eq!(got, [Cpu, Hit]);
}

/// More entries than the layer has experts (a top-k above E, or a corrupt doorbell): experts
/// outside the bitmap are never eligible, and the bitmap is never read past its words.
#[test]
fn experts_outside_the_bitmap_are_cpu() {
    let words = [u32::MAX];
    let bits = EligibleBits::new(&words, 4);
    let route: Vec<RouteSlot> = (0..10).map(|e| slot(e, 1, MOE_MISS)).collect();
    let got = assign(&route, bits, split(u32::MAX, 1, u32::MAX)).unwrap();
    let want: Vec<Assignment> = (0..10).map(|e| if e < 4 { Copy(e) } else { Cpu }).collect();
    assert_eq!(got, want);
    // Expert 65535 with a bitmap of one word.
    let got = assign(&[slot(u16::MAX, 1, MOE_MISS)], bits, split(u32::MAX, 1, 1)).unwrap();
    assert_eq!(got, [Cpu]);
}

#[test]
fn budgets_at_their_limits() {
    let all = [u32::MAX; 2];
    let bits = EligibleBits::new(&all, 64);
    let route: Vec<RouteSlot> = (0..40).map(|e| slot(e, 1, MOE_MISS)).collect();
    let copies = |got: Vec<Assignment>| got.iter().filter(|a| matches!(a, Copy(_))).count();
    // An expert larger than the whole budget is never copied, at the largest sizes.
    assert_eq!(
        copies(assign(&route, bits, split(u32::MAX, u64::MAX, u32::MAX)).unwrap()),
        0
    );
    assert_eq!(
        copies(assign(&route, bits, split(u32::MAX, u64::from(u32::MAX) + 1, 9)).unwrap()),
        0
    );
    // Exactly one fits.
    assert_eq!(
        copies(assign(&route, bits, split(u32::MAX, u64::from(u32::MAX), 9)).unwrap()),
        1
    );
    // Zero-byte experts are bounded by the copy slots alone; zero slots copy nothing.
    assert_eq!(copies(assign(&route, bits, split(0, 0, 7)).unwrap()), 7);
    assert_eq!(
        copies(assign(&route, bits, split(u32::MAX, 1, 0)).unwrap()),
        0
    );
    // A zero budget copies nothing of a non-zero size.
    assert_eq!(copies(assign(&route, bits, split(0, 1, 9)).unwrap()), 0);
}

/// The largest doorbell layer, 65535 entries, on its linear paths: all hits, and all misses
/// with none eligible. One more entry is refused.
#[test]
fn largest_doorbell() {
    let n = usize::from(u16::MAX);
    let hits: Vec<RouteSlot> = (0..n).map(|i| slot(i as u16, 1, i as u32)).collect();
    let got = assign(&hits, EligibleBits::none(), split(u32::MAX, 1, u32::MAX)).unwrap();
    assert!(got.iter().all(|a| *a == Hit));
    let misses: Vec<RouteSlot> = (0..n).map(|i| slot(i as u16, 1, MOE_MISS)).collect();
    let none = vec![0u32; 2048];
    let bits = EligibleBits::new(&none, 65536);
    let got = assign(&misses, bits, split(u32::MAX, 1, u32::MAX)).unwrap();
    assert!(got.iter().all(|a| *a == Cpu));
    let mut over = misses;
    over.push(slot(0, 1, MOE_MISS));
    assert_eq!(
        assign(&over, bits, split(0, 0, 0)),
        Err(AssignError::TooManyRoutes(n + 1))
    );
}

#[test]
fn assign_refuses_short_buffers() {
    let route = [slot(0, 1, MOE_MISS), slot(1, 1, MOE_MISS)];
    let mut out = [7u32; 1];
    assert_eq!(
        assign_into(&route, EligibleBits::none(), split(0, 0, 0), &mut out),
        Err(AssignError::OutputTooShort { need: 2, have: 1 })
    );
    assert_eq!(out, [7]);
    let words = [0u32; 1];
    assert_eq!(
        assign(&route, EligibleBits::new(&words, 33), split(0, 0, 0)),
        Err(AssignError::BitmapTooShort { need: 2, have: 1 })
    );
    // A longer output keeps its tail.
    let mut out = [7u32; 3];
    assign_into(&route, EligibleBits::none(), split(0, 0, 0), &mut out).unwrap();
    assert_eq!(out, [MOE_CPU, MOE_CPU, 7]);
}

#[test]
fn codes_round_trip() {
    for a in [Hit, Cpu, Copy(0), Copy(17), Copy(MOE_CPU - 1)] {
        assert_eq!(Assignment::from_code(a.code()), a);
    }
    assert_eq!(Hit.code(), MOE_MISS);
}

// ---- The doorbell layer -------------------------------------------------------------------

/// MAX_UNION = 8 and MAX_PRED = 16, by hand: the header's 8 bytes, 64 bytes of route entries
/// from 8, 128 of predictions from 72, 200 in all.
#[test]
fn doorbell_layout_by_hand() {
    let l = DoorbellLayout::new(8, 16);
    assert_eq!((l.route_offset(), l.pred_offset(), l.size()), (8, 72, 200));
    assert_eq!(DoorbellLayout::new(0, 0).size(), 8);
    let max = DoorbellLayout::new(u16::MAX, u16::MAX);
    assert_eq!(max.size(), 8 + 16 * usize::from(u16::MAX));
}

fn pred(ahead: u8, expert: u16, score: f32) -> PredSlot {
    PredSlot {
        ahead,
        _pad: 0,
        expert,
        score,
    }
}

#[test]
fn doorbell_round_trips_with_nan_scores() {
    let l = DoorbellLayout::new(4, 3);
    let layer = DoorbellLayer {
        seq: 0xDEAD_BEEF,
        route: vec![slot(3, 2, MOE_MISS), slot(65535, 65535, 0)],
        pred: vec![
            pred(1, 9, f32::from_bits(0x7FC0_1234)),
            pred(2, 65535, -0.0),
            pred(255, 0, f32::INFINITY),
        ],
    };
    let mut buf = vec![0xAAu8; l.size()];
    layer.write(&l, &mut buf).unwrap();
    // The header, then the first route entry, little-endian.
    assert_eq!(
        &buf[..16],
        &[
            0xEF, 0xBE, 0xAD, 0xDE, 2, 0, 3, 0, 3, 0, 2, 0, 255, 255, 255, 255
        ]
    );
    // Unused route entries are left alone.
    assert_eq!(&buf[24..40], &[0xAA; 16]);
    let back = DoorbellLayer::read(&l, &buf).unwrap();
    assert_eq!((back.seq, &back.route), (layer.seq, &layer.route));
    let bits = |p: &[PredSlot]| -> Vec<(u8, u16, u32)> {
        p.iter()
            .map(|p| (p.ahead, p.expert, p.score.to_bits()))
            .collect()
    };
    assert_eq!(bits(&back.pred), bits(&layer.pred));
    // The split rule reads no score: the doorbell's NaN predictions cannot change it.
    let got = assign(&back.route, EligibleBits::none(), split(9, 1, 9)).unwrap();
    assert_eq!(got, [Cpu, Hit]);
}

#[test]
fn doorbell_refuses_what_does_not_fit() {
    let l = DoorbellLayout::new(1, 1);
    let two = DoorbellLayer {
        seq: 1,
        route: vec![slot(0, 1, 0), slot(1, 1, 1)],
        pred: vec![],
    };
    let mut buf = vec![0u8; l.size()];
    assert_eq!(
        two.write(&l, &mut buf),
        Err(DoorbellError::TooManyRoutes { count: 2, max: 1 })
    );
    assert_eq!(buf, vec![0u8; l.size()]);
    let preds = DoorbellLayer {
        pred: vec![pred(1, 0, 0.0); 2],
        ..DoorbellLayer::default()
    };
    assert_eq!(
        preds.write(&l, &mut buf),
        Err(DoorbellError::TooManyPreds { count: 2, max: 1 })
    );
    assert_eq!(
        DoorbellLayer::default().write(&l, &mut buf[..l.size() - 1]),
        Err(DoorbellError::BufferTooSmall { need: 24, have: 23 })
    );
    // A corrupt count is refused before any entry is read.
    buf[4] = 2;
    assert_eq!(
        DoorbellLayer::read(&l, &buf),
        Err(DoorbellError::TooManyRoutes { count: 2, max: 1 })
    );
    buf[4] = 0;
    buf[6] = 9;
    assert_eq!(
        DoorbellLayer::read(&l, &buf),
        Err(DoorbellError::TooManyPreds { count: 9, max: 1 })
    );
}

proptest! {
    #[test]
    fn doorbell_round_trips(
        max_union in 0u16..64, max_pred in 0u16..64, seq in any::<u32>(),
        route in proptest::collection::vec(any::<(u16, u16, u32)>(), 0..64),
        pred in proptest::collection::vec(any::<(u8, u16, u32)>(), 0..64),
    ) {
        let l = DoorbellLayout::new(max_union, max_pred);
        let layer = DoorbellLayer {
            seq,
            route: route.into_iter().take(max_union.into()).map(|(e, t, s)| slot(e, t, s)).collect(),
            pred: pred
                .into_iter()
                .take(max_pred.into())
                .map(|(a, e, s)| pred_slot(a, e, s))
                .collect(),
        };
        let mut buf = vec![0u8; l.size()];
        layer.write(&l, &mut buf).unwrap();
        let back = DoorbellLayer::read(&l, &buf).unwrap();
        prop_assert_eq!(back.route, layer.route);
        let bits = |p: &[PredSlot]| -> Vec<(u8, u16, u32)> {
            p.iter().map(|p| (p.ahead, p.expert, p.score.to_bits())).collect()
        };
        prop_assert_eq!(bits(&back.pred), bits(&layer.pred));
    }
}

fn pred_slot(ahead: u8, expert: u16, score_bits: u32) -> PredSlot {
    pred(ahead, expert, f32::from_bits(score_bits))
}

// ---- Tables -------------------------------------------------------------------------------

#[test]
fn slot_table() {
    let mut t = SlotOf::new(3, 10).unwrap();
    assert_eq!(t.index(2, 9), Ok(29));
    assert_eq!(t.get(1, 4), Ok(None));
    t.set(1, 4, Some(77)).unwrap();
    assert_eq!(t.get(1, 4), Ok(Some(77)));
    assert_eq!(t.as_words()[14], 77);
    assert_eq!(t.route_slot(1, 4, 3), Ok(slot(4, 3, 77)));
    assert_eq!(t.route_slot(1, 5, 3), Ok(slot(5, 3, MOE_MISS)));
    t.set(1, 4, None).unwrap();
    assert_eq!(t.get(1, 4), Ok(None));
    assert!(matches!(
        t.set(1, 4, Some(MOE_MISS)),
        Err(TableError::OutOfRange { what: "slot", .. })
    ));
    assert!(matches!(
        t.get(3, 0),
        Err(TableError::OutOfRange { what: "layer", .. })
    ));
    assert!(matches!(
        t.get(0, 10),
        Err(TableError::OutOfRange { what: "expert", .. })
    ));
    assert_eq!(SlotOf::new(1 << 16, 1 << 16), Err(TableError::TooLarge));
}

#[test]
fn slot_addresses_and_copy_sources() {
    let a = SlotAddr::new(vec![0x1000, 0x2000]).unwrap();
    assert_eq!((a.addr(1), a.addr(2)), (Some(0x2000), None));
    assert_eq!(a.as_words(), &[0x1000, 0x2000]);
    let mut s = CopySrc::new(2, 4).unwrap();
    assert_eq!(s.get(1, 3), Ok(None));
    s.set(1, 3, Some(0xABC0)).unwrap();
    assert_eq!(s.get(1, 3), Ok(Some(0xABC0)));
    s.set(1, 3, None).unwrap();
    assert_eq!(s.get(1, 3), Ok(None));
    assert!(s.get(2, 0).is_err());
}

#[test]
fn eligibility_bitmap() {
    let mut t = CopyEligible::new(2, 40).unwrap();
    for (layer, expert) in [(0, 0), (0, 33), (1, 39)] {
        t.apply(EligibilityDelta {
            layer,
            expert,
            eligible: true,
        })
        .unwrap();
    }
    assert_eq!(t.as_words(), &[1, 2, 0, 1 << 7]);
    let l1 = t.layer(1).unwrap();
    assert_eq!((l1.experts(), l1.words()), (40, &[0, 1 << 7][..]));
    assert!(l1.get(39) && !l1.get(38) && !l1.get(40));
    t.apply(EligibilityDelta {
        layer: 0,
        expert: 33,
        eligible: false,
    })
    .unwrap();
    assert_eq!(t.get(0, 33), Ok(false));
    assert!(
        t.apply(EligibilityDelta {
            layer: 0,
            expert: 40,
            eligible: true
        })
        .is_err()
    );
}

#[test]
fn balance_block_from_the_step_descriptor() {
    let d = crate::StepDescriptor {
        moe_balance: vec![4096, 0, 7],
        ..Default::default()
    };
    let b = BalanceBlock::of(&d);
    assert_eq!(
        (b.b_copy(0), b.b_copy(2), b.b_copy(3)),
        (Some(4096), Some(7), None)
    );
    assert_eq!(b.split(0, 1024, 2), Some(split(4096, 1024, 2)));
    assert_eq!(BalanceBlock::new(&[5]).split(1, 1, 1), None);
}

// ---- Eligibility at the step's delta_head --------------------------------------------------

#[test]
fn mirror_holds_exactly_the_step_version() {
    let mut m = EligibilityMirror::new(CopyEligible::new(1, 8).unwrap(), 10);
    let on = |expert| EligibilityDelta {
        layer: 0,
        expert,
        eligible: true,
    };
    m.post(10, on(1)).unwrap();
    m.post(14, on(2)).unwrap();
    assert!(matches!(
        m.post(14, on(3)),
        Err(TableError::PositionOutOfOrder { .. })
    ));
    // The step at delta_head 14 sees position 10, not 14.
    assert_eq!(m.advance_to(14).unwrap().as_words(), &[0b10]);
    assert_eq!(m.at(14).unwrap().as_words(), &[0b10]);
    // Evaluating another version is refused, ahead or behind.
    assert_eq!(
        m.at(15),
        Err(TableError::VersionSkew {
            version: 14,
            delta_head: 15
        })
    );
    assert!(matches!(m.at(13), Err(TableError::VersionSkew { .. })));
    assert!(matches!(
        m.advance_to(13),
        Err(TableError::DeltaHeadRegressed { .. })
    ));
    assert!(matches!(
        m.post(13, on(3)),
        Err(TableError::PositionOutOfOrder { .. })
    ));
    assert_eq!(m.advance_to(15).unwrap().as_words(), &[0b110]);
    assert_eq!(m.version(), 15);
}

#[test]
fn retire_gate_by_hand() {
    let mut g = RetireGate::new();
    // A clear posted at position 5; steps 1 and 2 launched before it.
    g.launch(1, 3).unwrap();
    g.launch(2, 5).unwrap();
    assert!(!g.may_retire(5), "no step has seen the clear yet");
    assert_eq!(g.waits_on(5), Some(2));
    g.launch(3, 6).unwrap();
    assert!(!g.may_retire(5));
    g.complete(2).unwrap();
    assert!(!g.may_retire(5), "step 1 may still read it");
    assert_eq!(g.waits_on(5), Some(1));
    g.complete(1).unwrap();
    assert!(g.may_retire(5));
    assert_eq!(g.waits_on(5), None);
    assert!(!g.may_retire(6));
    assert!(matches!(
        g.launch(3, 9),
        Err(TableError::StepOutOfOrder { .. })
    ));
    assert!(matches!(
        g.launch(4, 5),
        Err(TableError::StepOutOfOrder { .. })
    ));
    assert_eq!(g.complete(2), Err(TableError::UnknownStep(2)));
    assert_eq!(g.complete(9), Err(TableError::UnknownStep(9)));
}

/// One action of the skew simulation.
#[derive(Clone, Debug)]
enum Op {
    /// The dispatcher posts an eligibility change, `gap` ring positions after the last entry
    /// (the ring carries other tables' deltas in between).
    Post {
        layer: u32,
        expert: u32,
        eligible: bool,
        gap: u64,
    },
    /// The dispatcher launches a step at a `delta_head` `lag` positions below its latest
    /// posted position (0: everything posted, never below the last step's), and the host
    /// evaluates it with the mirror in skew `skew`: 0 in step, 1 ahead (one more delta posted
    /// after the launch), 2 behind (checked before the mirror is advanced).
    Launch { lag: u64, skew: u8, seed: u64 },
    /// The oldest incomplete step's fence passes.
    Complete,
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        4 => (0u32..2, 0u32..40, any::<bool>(), 0u64..3).prop_map(
            |(layer, expert, eligible, gap)| Op::Post { layer, expert, eligible, gap }
        ),
        3 => (0u64..3, 0u8..3, any::<u64>()).prop_map(|(lag, skew, seed)| Op::Launch { lag, skew, seed }),
        2 => Just(Op::Complete),
    ]
}

const LAYERS: u32 = 2;
const EXPERTS: u32 = 40;

/// The device's view, modelled without the types under test: every posted delta, and a
/// bitmap rebuilt by applying the ones below a `delta_head`, as `ApplyTableDeltas` does.
fn device_bits(ring: &[(u64, EligibilityDelta)], delta_head: u64) -> Vec<Vec<u32>> {
    let mut bits = vec![vec![0u32; EXPERTS.div_ceil(32) as usize]; LAYERS as usize];
    for &(_, d) in ring.iter().filter(|e| e.0 < delta_head) {
        let w = &mut bits[d.layer as usize][(d.expert / 32) as usize];
        if d.eligible {
            *w |= 1 << (d.expert % 32);
        } else {
            *w &= !(1 << (d.expert % 32));
        }
    }
    bits
}

/// The source of `(layer, expert)` at `delta_head`: the position of the delta that last made
/// it eligible, if it is eligible there.
fn source_at(
    ring: &[(u64, EligibilityDelta)],
    delta_head: u64,
    layer: u32,
    expert: u32,
) -> Option<u64> {
    let mut src = None;
    for &(pos, d) in ring.iter().filter(|e| e.0 < delta_head) {
        if (d.layer, d.expert) == (layer, expert) {
            src = d.eligible.then_some(pos);
        }
    }
    src
}

fn doorbell(seed: u64) -> Vec<RouteSlot> {
    let mut rng = Rng(seed);
    let n = rng.below(16) as usize;
    let mut seen = std::collections::BTreeSet::new();
    (0..n)
        .filter_map(|_| {
            let e = rng.below(u64::from(EXPERTS)) as u16;
            let r = slot(
                e,
                rng.below(3) as u16,
                if rng.below(3) == 0 { 5 } else { MOE_MISS },
            );
            seen.insert(e).then_some(r)
        })
        .collect()
}

/// The dispatcher's side of the skew simulation: the ring of eligibility deltas as posted,
/// the host mirror fed from it, and the sources waiting for retirement.
struct Sim {
    ring: Vec<(u64, EligibilityDelta)>,
    next_pos: u64,
    mirror: EligibilityMirror,
    /// (layer, expert, source position, clearing position).
    waiting: Vec<(u32, u32, u64, u64)>,
}

impl Sim {
    fn post(&mut self, d: EligibilityDelta, gap: u64) {
        let pos = self.next_pos + gap;
        self.next_pos = pos + 1;
        // A clear of an eligible expert puts its current source up for retirement.
        if !d.eligible
            && let Some(src) = source_at(&self.ring, pos, d.layer, d.expert)
        {
            self.waiting.push((d.layer, d.expert, src, pos));
        }
        self.ring.push((pos, d));
        self.mirror.post(pos, d).unwrap();
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]
    /// With the host mirror one delta ahead of the step's `delta_head` or behind it, the host
    /// sites and the device still produce the same split, because both evaluate the step at its
    /// `delta_head`; and a copy source is never retired while a step that saw it eligible is
    /// incomplete, nor seen eligible by a step launched after its retirement.
    #[test]
    fn host_and_device_agree_under_skew(ops in proptest::collection::vec(op_strategy(), 1..120)) {
        let mut sim = Sim {
            ring: Vec::new(),
            next_pos: 0,
            mirror: EligibilityMirror::new(CopyEligible::new(LAYERS, EXPERTS).unwrap(), 0),
            waiting: Vec::new(),
        };
        let mut gate = RetireGate::new();
        let (mut step, mut head) = (0u64, 0u64);
        // Launched steps: (step, delta_head, complete).
        let mut steps: Vec<(u64, u64, bool)> = Vec::new();
        let mut retired: BTreeSet<(u32, u32, u64)> = BTreeSet::new();
        let split = split(u32::MAX, 1, 4);
        for op in ops {
            match op {
                Op::Post { layer, expert, eligible, gap } => {
                    sim.post(EligibilityDelta { layer, expert, eligible }, gap);
                }
                Op::Launch { lag, skew, seed } => {
                    step += 1;
                    head = sim.next_pos.saturating_sub(lag).max(head);
                    gate.launch(step, head).unwrap();
                    steps.push((step, head, false));
                    // No retired source is eligible at a step launched after its retirement.
                    for &(l, e, src) in &retired {
                        prop_assert_ne!(source_at(&sim.ring, head, l, e), Some(src));
                    }
                    let device = device_bits(&sim.ring, head);
                    match skew {
                        // Ahead: a delta posted after the launch reaches the mirror before the
                        // host evaluates the step.
                        1 => sim.post(
                            EligibilityDelta {
                                layer: (seed % 2) as u32,
                                expert: ((seed >> 1) % u64::from(EXPERTS)) as u32,
                                eligible: (seed >> 8) % 2 == 0,
                            },
                            0,
                        ),
                        // Behind: the mirror is still at an earlier step's version, and refuses
                        // to evaluate this one until it is advanced.
                        2 if sim.mirror.version() < head => {
                            prop_assert!(sim.mirror.at(head).is_err());
                        }
                        _ => {}
                    }
                    let table = sim.mirror.advance_to(head).unwrap();
                    for layer in 0..LAYERS {
                        let route = doorbell(seed ^ u64::from(layer));
                        let bits = EligibleBits::new(&device[layer as usize], EXPERTS);
                        let dev = assign(&route, bits, split).unwrap();
                        let host = assign(&route, table.layer(layer).unwrap(), split).unwrap();
                        prop_assert_eq!(&dev, &host, "step {} layer {}", step, layer);
                    }
                }
                Op::Complete => {
                    if let Some(s) = steps.iter_mut().find(|s| !s.2) {
                        s.2 = true;
                        gate.complete(s.0).unwrap();
                    }
                }
            }
            // Retire every source the gate allows, checking that no incomplete step saw it.
            let mut i = 0;
            while i < sim.waiting.len() {
                let (l, e, src, clear) = sim.waiting[i];
                if gate.may_retire(clear) {
                    for &(s, h, done) in &steps {
                        if source_at(&sim.ring, h, l, e) == Some(src) {
                            prop_assert!(done, "source {} of ({}, {}) retired before step {} completed", src, l, e, s);
                        }
                    }
                    retired.insert((l, e, src));
                    sim.waiting.swap_remove(i);
                } else {
                    i += 1;
                }
            }
        }
    }
}

/// The negative control of the skew test: a host that evaluated at the dispatcher's latest
/// posted state instead of the step's `delta_head` would disagree with the device.
#[test]
fn latest_state_would_disagree() {
    let mut m = EligibilityMirror::new(CopyEligible::new(1, 8).unwrap(), 0);
    let d = |expert, eligible| EligibilityDelta {
        layer: 0,
        expert,
        eligible,
    };
    m.post(0, d(3, true)).unwrap();
    // The step launches at delta_head 1; the dispatcher then posts a clear at 1.
    m.post(1, d(3, false)).unwrap();
    let route = [slot(3, 1, MOE_MISS)];
    let at_step = assign(
        &route,
        m.advance_to(1).unwrap().layer(0).unwrap(),
        split(9, 1, 1),
    )
    .unwrap();
    assert_eq!(at_step, [Copy(0)]);
    let mut latest = m.clone();
    let latest = assign(
        &route,
        latest.advance_to(2).unwrap().layer(0).unwrap(),
        split(9, 1, 1),
    )
    .unwrap();
    assert_eq!(latest, [Cpu]);
}
