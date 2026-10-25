// SPDX-License-Identifier: MIT OR Apache-2.0

//! COMP-E3-T07's tests: the dispatch test (no token, and no routine call, without SME2), the
//! tiles' conformance against scalar references, and the callee-saved register harness.
//!
//! The conformance and harness tests need an SME2 host. Elsewhere they print that they were
//! skipped and pass; the dispatch test runs everywhere.

use std::cell::Cell;

use syntropy_kernels::CpuIsa;

use super::*;
use crate::dispatch::DispatchTable;
use crate::isa::{ArmInputs, IsaInputs, KillSwitches, detect, select};

thread_local! {
    /// The tile routines this thread called; `ffi::count` adds one per call.
    pub(super) static CALLS: Cell<u64> = const { Cell::new(0) };
}

fn calls() -> u64 {
    CALLS.with(Cell::get)
}

const LEVELS: [CpuIsa; 8] = [
    CpuIsa::X86V2,
    CpuIsa::X86V3,
    CpuIsa::X86V4,
    CpuIsa::X86V4Amx,
    CpuIsa::Arm80,
    CpuIsa::Arm82Dot,
    CpuIsa::Arm86I8mm,
    CpuIsa::Arm92Sme2,
];

/// The live host's token; `None`, with a note, when the host has no SME2.
fn host_token(test: &str) -> Option<Sme2> {
    let t = Sme2::for_level(CpuIsa::Arm92Sme2);
    if t.is_none() {
        eprintln!("{test}: skipped: this host has no SME2 (or the build has no SME2 unit)");
    }
    t
}

/// What a kernel holding `table` does: it reaches the SME2 routines only through the table's
/// token, and does nothing without one.
fn kernel_through(table: &DispatchTable) {
    if let Some(t) = table.sme2() {
        let mut c = [0.0_f32; 4];
        t.gemm_f32(&[1.0; 4], &[1.0; 4], &mut c, 2, 2, 2)
            .expect("a 2 × 2 product");
    }
}

// ---- Dispatch -----------------------------------------------------------------------------

#[test]
fn a_token_needs_the_sme2_level_and_an_sme2_host() {
    for isa in LEVELS {
        let t = token(isa, Some(16));
        assert_eq!(t.is_some(), isa == CpuIsa::Arm92Sme2, "{isa:?}");
        // A host that reports no SME2 gives no token, whatever the level says.
        assert_eq!(token(isa, None), None, "{isa:?}");
        assert_eq!(token(isa, Some(0)), None, "{isa:?}");
    }
    assert_eq!(
        token(CpuIsa::Arm92Sme2, Some(16)).map(Sme2::tile_max),
        Some(32)
    );
}

/// ADR-0402 item 3: on a machine without SME2, or with the `sme2` kill switch, the routines are
/// never called. Every table below the SME2 level has no token, so a kernel that goes through
/// its table cannot reach a routine, and the call counter stays at zero.
#[test]
fn without_sme2_the_routines_are_never_called() {
    let before = calls();
    for isa in LEVELS.into_iter().filter(|l| *l != CpuIsa::Arm92Sme2) {
        let table = DispatchTable::for_level(isa);
        assert_eq!(table.sme2(), None, "{isa:?}");
        kernel_through(&table);
    }

    // Recorded inputs of an Arm host without SME2, and of one with SME2 under the kill switch.
    let full = ArmInputs {
        dotprod: true,
        fp16: true,
        i8mm: true,
        bf16: true,
        sve2: false,
        sme2: false,
    };
    let kill = KillSwitches::from_disable(["sme2"]);
    for (inputs, k) in [
        (full, KillSwitches::NONE),
        (ArmInputs { sme2: true, ..full }, kill),
    ] {
        let d = select(&IsaInputs::Arm(inputs), &k).expect("an Arm level");
        assert_eq!(d.isa, CpuIsa::Arm86I8mm);
        let table = d.table();
        assert_eq!(table.sme2(), None);
        kernel_through(&table);
    }

    // The live host with SME2 forced off: whatever it has, detection gives no SME2 table.
    if let Some(d) = detect(&kill) {
        assert_ne!(d.isa, CpuIsa::Arm92Sme2);
        let table = d.table();
        assert_eq!(table.sme2(), None);
        kernel_through(&table);
    }
    assert_eq!(calls(), before, "a routine was called without SME2");
}

/// The positive control for the dispatch test: on an SME2 host the SME2 table does reach the
/// routines, so the counter above would have seen a call.
#[test]
fn the_sme2_table_reaches_the_routines_on_an_sme2_host() {
    let Some(_) = host_token("the_sme2_table_reaches_the_routines_on_an_sme2_host") else {
        return;
    };
    let d = detect(&KillSwitches::NONE).expect("a level");
    assert_eq!(d.isa, CpuIsa::Arm92Sme2);
    let before = calls();
    kernel_through(&d.table());
    assert_eq!(calls(), before + 1);
}

// ---- Conformance --------------------------------------------------------------------------

/// A xorshift64* stream, so the cases are the same on every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform in [-1, 1).
    fn f32(&mut self) -> f32 {
        let u = (self.next() >> 40) as f32 / (1_u64 << 24) as f32;
        2.0 * u - 1.0
    }

    fn i8(&mut self) -> i8 {
        (self.next() >> 56) as u8 as i8
    }
}

/// The F32 reference: per output, a chain of fused multiply-adds over k in increasing order.
fn ref_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc = a[i * k + p].mul_add(b[p * n + j], acc);
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// The F64 reference RFC-0042 §6.4 measures `rel_l2` against.
fn ref_f64(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = (0..k)
                .map(|p| f64::from(a[i * k + p]) * f64::from(b[p * n + j]))
                .sum();
        }
    }
    c
}

fn rel_l2(y: &[f32], r: &[f64]) -> f64 {
    let num: f64 = y
        .iter()
        .zip(r)
        .map(|(y, r)| (f64::from(*y) - r).powi(2))
        .sum();
    let den: f64 = r.iter().map(|r| r * r).sum();
    if den == 0.0 {
        num.sqrt()
    } else {
        (num / den).sqrt()
    }
}

fn ref_i8(a: &[i8], b: &[i8], m: usize, n: usize, k: usize) -> Vec<i32> {
    let mut c = vec![0_i32; m * n];
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = (0..k)
                .map(|p| i32::from(a[i * k + p]) * i32::from(b[p * n + j]))
                .sum();
        }
    }
    c
}

/// Shapes around the tile edges: one lane, one vector, one tile, past a tile, several tiles.
fn shapes(t: &Sme2) -> Vec<(usize, usize, usize)> {
    let v = t.svl_words();
    let mut out = Vec::new();
    for m in [1, 3, v - 1, v, v + 1, 2 * v, 2 * v + 3, 5 * v + 7] {
        for n in [1, v, v + 5, 2 * v, 3 * v + 1] {
            for k in [0, 1, 4, 7, 64, 259] {
                out.push((m, n, k));
            }
        }
    }
    out
}

/// Conformance of the F32 tiles: equal bit for bit to the fused-multiply-add reference on
/// every shape, with subnormal products kept. `rel_l2` against F64 is the reference's own,
/// RFC-0042 §6.4's floor `ε_op`.
#[test]
fn f32_gemm_equals_the_fused_reference_bit_for_bit() {
    let Some(t) = host_token("f32_gemm_equals_the_fused_reference_bit_for_bit") else {
        return;
    };
    let mut rng = Rng(0x5eed_0007);
    let mut worst = 0.0_f64;
    for (m, n, k) in shapes(&t) {
        for scale in [1.0_f32, 1.0e-20] {
            let a: Vec<f32> = (0..m * k).map(|_| rng.f32() * scale).collect();
            let b: Vec<f32> = (0..k * n).map(|_| rng.f32() * scale).collect();
            let mut c = vec![f32::NAN; m * n];
            t.gemm_f32(&a, &b, &mut c, m, n, k).expect("gemm_f32");
            let r = ref_f32(&a, &b, m, n, k);
            for (idx, (got, want)) in c.iter().zip(&r).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "m {m} n {n} k {k} scale {scale}: element {idx}: {got:e} != {want:e}"
                );
            }
            if scale == 1.0 {
                worst = worst.max(rel_l2(&c, &ref_f64(&a, &b, m, n, k)));
            }
        }
    }
    eprintln!(
        "SME2 F32 GEMM, SVL_s = {}: bit-identical to the fused F32 reference; worst rel_l2 \
         against F64 {worst:e}",
        t.svl_words()
    );
}

/// Subnormal products are kept, not flushed (RFC-0042 §4.1 rule 7).
#[test]
fn f32_tile_keeps_subnormals() {
    let Some(t) = host_token("f32_tile_keeps_subnormals") else {
        return;
    };
    // 2^-70 · 2^-70 = 2^-140, an F32 subnormal.
    let x = f32::from_bits(0x1c80_0000);
    assert_eq!(x, 2.0_f32.powi(-70));
    let mut c = [0.0_f32];
    t.f32_tile(&[x], &[x], &mut c, 1, 1, 1, 1, 1).expect("tile");
    assert!(c[0].is_subnormal(), "{:e}", c[0]);
    assert_eq!(c[0], x * x);
}

/// A tile with B and C strides wider than the tile writes only its own elements.
#[test]
fn f32_tile_writes_only_its_elements() {
    let Some(t) = host_token("f32_tile_writes_only_its_elements") else {
        return;
    };
    let (k, ldb, ldc) = (9, 3 * t.tile_max(), 2 * t.tile_max() + 1);
    let mut rng = Rng(42);
    for (m, n) in [(1, 1), (t.svl_words() + 2, 3), (t.tile_max(), t.tile_max())] {
        let a: Vec<f32> = (0..k * m).map(|_| rng.f32()).collect();
        let b: Vec<f32> = (0..k * ldb).map(|_| rng.f32()).collect();
        let sentinel = -7.5_f32;
        let mut c = vec![sentinel; m * ldc];
        t.f32_tile(&a, &b, &mut c, k, m, n, ldb, ldc).expect("tile");
        for i in 0..m {
            for j in 0..ldc {
                let got = c[i * ldc + j];
                if j < n {
                    let mut want = 0.0_f32;
                    for p in 0..k {
                        want = a[p * m + i].mul_add(b[p * ldb + j], want);
                    }
                    assert_eq!(got.to_bits(), want.to_bits(), "({i}, {j})");
                } else {
                    assert_eq!(got, sentinel, "({i}, {j}) is outside the tile");
                }
            }
        }
    }
}

/// Conformance of the i8 tiles: exact, so equal to the integer reference on every shape,
/// including the extremes of the range at a long k.
#[test]
fn i8_gemm_equals_the_integer_reference_exactly() {
    let Some(t) = host_token("i8_gemm_equals_the_integer_reference_exactly") else {
        return;
    };
    let mut rng = Rng(0x5eed_0008);
    for (m, n, k) in shapes(&t) {
        let a: Vec<i8> = (0..m * k).map(|_| rng.i8()).collect();
        let b: Vec<i8> = (0..k * n).map(|_| rng.i8()).collect();
        let mut c = vec![i32::MIN; m * n];
        t.gemm_i8(&a, &b, &mut c, m, n, k).expect("gemm_i8");
        assert_eq!(c, ref_i8(&a, &b, m, n, k), "m {m} n {n} k {k}");
    }
    // Every product at -128 · -128, over the longest k the API accepts.
    let (m, n, k) = (t.svl_words() + 1, 3, I8_MAX_K);
    let a = vec![i8::MIN; m * k];
    let b = vec![i8::MIN; k * n];
    let mut c = vec![0_i32; m * n];
    t.gemm_i8(&a, &b, &mut c, m, n, k).expect("gemm_i8");
    let want = i32::try_from(k * 128 * 128).expect("fits");
    assert!(c.iter().all(|v| *v == want), "{want}");
}

#[test]
fn bad_shapes_are_refused_before_any_call() {
    let Some(t) = host_token("bad_shapes_are_refused_before_any_call") else {
        return;
    };
    let before = calls();
    let big = t.tile_max() + 1;
    let a = vec![0.0_f32; 4 * big];
    let b = vec![0.0_f32; 4 * big];
    let mut c = vec![0.0_f32; big * big];
    // Too many rows or columns for one tile.
    assert!(t.f32_tile(&a, &b, &mut c, 1, big, 1, 1, 1).is_err());
    assert!(t.f32_tile(&a, &b, &mut c, 1, 1, big, big, big).is_err());
    // A stride below the width; slices too short for the shape.
    assert!(t.f32_tile(&a, &b, &mut c, 1, 1, 2, 1, 2).is_err());
    assert!(t.f32_tile(&a, &b, &mut c, 1, 2, 2, 2, 1).is_err());
    assert!(t.f32_tile(&a[..1], &b, &mut c, 2, 1, 1, 1, 1).is_err());
    assert!(t.f32_tile(&a, &b[..1], &mut c, 2, 1, 1, 1, 1).is_err());
    assert!(t.f32_tile(&a, &b, &mut c[..1], 1, 2, 1, 1, 1).is_err());
    // An i8 tile whose k could overflow, and a product whose slices do not match its shape.
    let mut ci = [0_i32; 1];
    assert!(
        t.i8_tile(&[0; 4], &[0; 4], &mut ci, I8_MAX_K, 1, 1, 4, 1)
            .is_err()
    );
    assert!(t.i8_tile(&[0; 4], &[0; 4], &mut ci, 1, 1, 1, 3, 1).is_err());
    assert!(t.gemm_f32(&a, &b, &mut c, 2, 2, 2).is_err());
    assert_eq!(calls(), before);
}

// ---- The callee-saved register harness -----------------------------------------------------

#[cfg(syntropy_sme2)]
mod harness {
    use super::*;

    /// The pattern `harness.S`'s `PAT` puts in a register: `0xA5A5_0000_nnnn_nnnn`.
    fn pattern(n: u64) -> u64 {
        0xa5a5_0000_0000_0000 | (n << 16) | n
    }

    /// What the harness recorded after one call.
    struct Report([u64; 23]);

    impl Report {
        fn x(&self, r: usize) -> u64 {
            self.0[r - 19]
        }
        fn d(&self, r: usize) -> u64 {
            self.0[11 + r - 8]
        }
        fn ret(&self) -> u64 {
            self.0[19]
        }
        fn svcr(&self) -> u64 {
            self.0[20]
        }
        fn sp_moved(&self) -> bool {
            self.0[21] != self.0[22]
        }
    }

    fn harness(routine: *const core::ffi::c_void, args: [u64; 8]) -> Report {
        let mut report = [0_u64; 23];
        // SAFETY: the host has SME2 (the caller holds a token). The harness saves and restores
        // its own callee-saved registers and calls `routine` with `args`; the callers pass
        // routines from this unit with arguments that satisfy them (valid buffers and shapes).
        unsafe { ffi::syn_sme2_harness_call(report.as_mut_ptr(), routine, args.as_ptr()) };
        Report(report)
    }

    /// Register numbers with the pattern each should hold.
    type Regs = Vec<(usize, u64)>;

    /// The register numbers and patterns `harness.S` uses: x19–x29 get 0x19–0x29, and d8–d15 get
    /// 0x108–0x115 written as hexadecimal digits of the register number.
    fn expected() -> (Regs, Regs) {
        let x = (19..=29)
            .map(|r| {
                (
                    r,
                    pattern(u64::from_str_radix(&r.to_string(), 16).expect("hex")),
                )
            })
            .collect();
        let d = (8..=15)
            .map(|r| {
                let n = u64::from_str_radix(&format!("1{r:02}"), 16).expect("hex");
                (r, pattern(n))
            })
            .collect();
        (x, d)
    }

    /// The harness test: every routine preserves x19–x29 and d8–d15 and SP, and returns with
    /// streaming mode and ZA off (SVCR = 0).
    #[test]
    fn every_routine_preserves_callee_saved_registers() {
        let Some(t) = host_token("every_routine_preserves_callee_saved_registers") else {
            return;
        };
        let (m, n, k) = (t.tile_max(), t.tile_max(), 5);
        let a = vec![1.0_f32; k * m];
        let b = vec![2.0_f32; k * n];
        let mut c = vec![0.0_f32; m * n];
        let ai = vec![1_i8; 4 * k * m];
        let bi = vec![2_i8; 4 * k * n];
        let mut ci = vec![0_i32; m * n];
        let (xs, ds) = expected();
        for (name, routine) in ffi::routines() {
            let (args, want_ret) = match name {
                "syn_sme2_svl_bytes" => ([0; 8], 4 * t.svl_words() as u64),
                "syn_sme2_f32_tile" => (
                    [
                        a.as_ptr() as u64,
                        b.as_ptr() as u64,
                        c.as_mut_ptr() as u64,
                        k as u64,
                        m as u64,
                        n as u64,
                        n as u64,
                        n as u64,
                    ],
                    0,
                ),
                "syn_sme2_i8_tile" => (
                    [
                        ai.as_ptr() as u64,
                        bi.as_ptr() as u64,
                        ci.as_mut_ptr() as u64,
                        k as u64,
                        m as u64,
                        n as u64,
                        4 * n as u64,
                        n as u64,
                    ],
                    0,
                ),
                other => panic!("no arguments for {other}"),
            };
            let r = harness(routine, args);
            for (reg, want) in &xs {
                assert_eq!(r.x(*reg), *want, "{name}: x{reg}");
            }
            for (reg, want) in &ds {
                assert_eq!(r.d(*reg), *want, "{name}: d{reg}");
            }
            assert_eq!(
                r.svcr(),
                0,
                "{name}: returned with PSTATE.SM or PSTATE.ZA set"
            );
            assert!(!r.sp_moved(), "{name}: SP moved");
            assert_eq!(r.ret(), want_ret, "{name}: return value");
        }
        // The tiles really ran under the harness.
        assert!(c.iter().all(|v| *v == 2.0 * k as f32));
        assert!(ci.iter().all(|v| *v == 2 * 4 * k as i32));
    }

    /// The harness's negative control: a routine that enters and leaves streaming mode without
    /// saving d8–d15 loses them, and the harness sees it.
    #[test]
    fn the_harness_detects_a_routine_that_loses_d8_to_d15() {
        let Some(_) = host_token("the_harness_detects_a_routine_that_loses_d8_to_d15") else {
            return;
        };
        let r = harness(ffi::syn_sme2_harness_clobber as *const _, [0; 8]);
        let (xs, ds) = expected();
        for (reg, want) in &xs {
            assert_eq!(r.x(*reg), *want, "x{reg}");
        }
        for (reg, want) in &ds {
            assert_ne!(r.d(*reg), *want, "d{reg} survived streaming mode unsaved");
            assert_eq!(r.d(*reg), 0, "d{reg}");
        }
        assert_eq!(r.svcr(), 0);
    }
}
