// SPDX-License-Identifier: MIT OR Apache-2.0

//! The SME2 assembly unit: outer-product GEMM tiles for `Arm92Sme2` (RFC-0040 §7.3, the SME
//! row and "Compute-bound shapes"; ADR-0402 item 3; ADR-0001).
//!
//! Rust has no stable SME intrinsics, and streaming mode changes the vector length and the
//! register state compiled Rust code assumes. The routines are therefore complete leaf
//! functions in `src/sme2/*.S`, assembled by `build.rs` through the `cc` crate and called
//! through `extern "C"`. Each enters streaming mode and enables ZA with `smstart`, runs its
//! outer-product loop on the ZA tiles, leaves both with `smstop`, and preserves the AAPCS64
//! callee-saved registers (it saves d8–d15, which `smstart` and `smstop` zero). No Rust code
//! runs in streaming mode.
//!
//! **Who may call them.** Only a holder of an [`Sme2`] token, and [`Sme2::for_level`] gives one
//! only when the level is `Arm92Sme2` *and* the live host reports SME2. A kill switch
//! (`features.disable = ["sme2"]`) or a host without SME2 lowers the level, so no token exists
//! and no routine is called. A build for any target but AArch64 macOS, Linux or Android has no
//! assembly at all (`cfg(syntropy_sme2)` unset), and its hosts never report SME2.
//!
//! **What they compute.** A tile of up to 2·SVL_s × 2·SVL_s outputs, where SVL_s is the
//! streaming vector length in 32-bit lanes (16 on a 512-bit implementation), in ZA tiles 0–3:
//!
//! - [`Sme2::f32_tile`]: F32, one `FMOPA` per k step, so each output is a chain of fused
//!   multiply-adds in increasing k, rounded once per step. It equals a scalar loop of
//!   `f32::mul_add` over k bit for bit.
//! - [`Sme2::i8_tile`]: signed 8-bit inputs into exact 32-bit sums, one four-way widening
//!   `SMOPA` per group of four k values. Exact, so equal to any other exact integer kernel on
//!   the same quantised inputs (ADR-0402, Consequences).
//!
//! [`Sme2::gemm_f32`] and [`Sme2::gemm_i8`] tile a whole row-major product over them. They
//! pack each tile's A panel into a buffer they allocate; the `Gemm` op kernel that uses these
//! tiles will pack into the op's workspace instead (RFC-0040 §5.3), so steady-state steps
//! allocate nothing.

use std::sync::OnceLock;

use syntropy_kernels::{CpuIsa, Error};

use crate::device::internal;

#[cfg(test)]
mod tests;

/// The largest k [`Sme2::gemm_i8`] and [`Sme2::i8_tile`] accept: the largest multiple of 4
/// (one `SMOPA` group) with `k · 128 · 128` within `i32`, so no 32-bit sum overflows.
pub const I8_MAX_K: usize = (i32::MAX as usize) / (128 * 128) / 4 * 4;

/// Proof that the SME2 routines may run in this process: the level is `Arm92Sme2` and the host
/// reports SME2. It carries the streaming vector length the routines run at.
///
/// The field is private, so [`Sme2::for_level`] is the only way to get one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sme2 {
    svl_words: usize,
}

impl Sme2 {
    /// The token for `isa`: `Some` only when `isa` is `Arm92Sme2` and the live host reports
    /// SME2 (RFC-0040 §7.3's detection, with no kill switch, read once per process).
    #[must_use]
    pub fn for_level(isa: CpuIsa) -> Option<Self> {
        token(isa, host_svl_words())
    }

    /// The streaming vector length in 32-bit lanes, SVL_s.
    #[must_use]
    pub fn svl_words(self) -> usize {
        self.svl_words
    }

    /// The most rows, and the most columns, one tile call computes: 2 · SVL_s.
    #[must_use]
    pub fn tile_max(self) -> usize {
        2 * self.svl_words
    }

    /// One F32 tile: `c[i·ldc + j] = Σ_{p<k} a[p·m + i] · b[p·ldb + j]` for `i < m`, `j < n`,
    /// accumulated in increasing p with one fused multiply-add per step. `a` is A transposed
    /// and packed, k rows of m values; `b` holds k rows of B with row stride `ldb`. `c` is
    /// overwritten; its elements outside the tile are untouched.
    ///
    /// # Errors
    ///
    /// Class `Internal` when `m` or `n` exceeds [`tile_max`](Self::tile_max), a stride is
    /// below the tile width, or a slice is too short for the shape; nothing is computed.
    #[allow(clippy::too_many_arguments)]
    pub fn f32_tile(
        self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        k: usize,
        m: usize,
        n: usize,
        ldb: usize,
        ldc: usize,
    ) -> Result<(), Error> {
        let shape = TileShape { k, m, n, ldb, ldc };
        self.check(&shape, a.len(), k * m, b.len(), c.len())?;
        if m == 0 || n == 0 {
            return Ok(());
        }
        ffi::f32_tile(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), &shape)
    }

    /// One signed 8-bit tile with exact 32-bit sums:
    /// `c[i·ldc + j] = Σ_{g<groups} Σ_{q<4} a[g·4m + 4i + q] · b[g·ldb + 4j + q]`.
    /// `a` and `b` hold four consecutive k values per row or column for each group, so k is
    /// `4 · groups` (pad with zeros). `ldb` counts bytes and is at least `4n`; `ldc` counts
    /// `i32`s. `c` is overwritten; its elements outside the tile are untouched.
    ///
    /// # Errors
    ///
    /// As [`f32_tile`](Self::f32_tile), and when `4 · groups` exceeds [`I8_MAX_K`].
    #[allow(clippy::too_many_arguments)]
    pub fn i8_tile(
        self,
        a: &[i8],
        b: &[i8],
        c: &mut [i32],
        groups: usize,
        m: usize,
        n: usize,
        ldb: usize,
        ldc: usize,
    ) -> Result<(), Error> {
        if groups.saturating_mul(4) > I8_MAX_K {
            return Err(internal(format!(
                "an SME2 i8 tile of {groups} groups exceeds k = {I8_MAX_K}, past which an i32 \
                 sum can overflow"
            )));
        }
        let shape = TileShape {
            k: groups,
            m,
            n,
            ldb,
            ldc,
        };
        // In bytes, B's rows are 4n wide; check the stride against that.
        let bytes = TileShape { n: 4 * n, ..shape };
        self.check_bounds(&bytes, a.len(), groups * 4 * m, b.len())?;
        self.check_tile(&shape)?;
        check_c(&shape, c.len())?;
        if m == 0 || n == 0 {
            return Ok(());
        }
        ffi::i8_tile(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), &shape)
    }

    /// `c = a · b` for row-major `a` (m × k), `b` (k × n) and `c` (m × n), in F32, tile by
    /// tile through [`f32_tile`](Self::f32_tile). Each output is the fused multiply-add chain
    /// over k in increasing order.
    ///
    /// # Errors
    ///
    /// Class `Internal` when a slice's length is not its shape's product.
    pub fn gemm_f32(
        self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), Error> {
        check_gemm(a.len(), b.len(), c.len(), m, n, k)?;
        let t = self.tile_max();
        let mut panel = vec![0.0_f32; k * t.min(m)];
        for i0 in (0..m).step_by(t) {
            let mt = t.min(m - i0);
            for p in 0..k {
                for i in 0..mt {
                    panel[p * mt + i] = a[(i0 + i) * k + p];
                }
            }
            for j0 in (0..n).step_by(t) {
                let nt = t.min(n - j0);
                self.f32_tile(
                    &panel[..k * mt],
                    b.get(j0..).unwrap_or(&[]),
                    &mut c[i0 * n + j0..],
                    k,
                    mt,
                    nt,
                    n,
                    n,
                )?;
            }
        }
        Ok(())
    }

    /// `c = a · b` for row-major `a` (m × k) and `b` (k × n) in `i8`, into row-major `c`
    /// (m × n) in exact `i32`, tile by tile through [`i8_tile`](Self::i8_tile).
    ///
    /// # Errors
    ///
    /// Class `Internal` when a slice's length is not its shape's product, or k exceeds
    /// [`I8_MAX_K`].
    pub fn gemm_i8(
        self,
        a: &[i8],
        b: &[i8],
        c: &mut [i32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), Error> {
        check_gemm(a.len(), b.len(), c.len(), m, n, k)?;
        if k > I8_MAX_K {
            return Err(internal(format!(
                "an SME2 i8 product with k = {k} exceeds {I8_MAX_K}, past which an i32 sum \
                 can overflow"
            )));
        }
        let groups = k.div_ceil(4);
        // B, interleaved by groups of four k values, zero-padded: bp[g·4n + 4j + q].
        let mut bp = vec![0_i8; groups * 4 * n];
        for p in 0..k {
            for j in 0..n {
                bp[(p / 4) * 4 * n + 4 * j + p % 4] = b[p * n + j];
            }
        }
        let t = self.tile_max();
        let mut panel = vec![0_i8; groups * 4 * t.min(m)];
        for i0 in (0..m).step_by(t) {
            let mt = t.min(m - i0);
            let ap = &mut panel[..groups * 4 * mt];
            ap.fill(0);
            for i in 0..mt {
                for p in 0..k {
                    ap[(p / 4) * 4 * mt + 4 * i + p % 4] = a[(i0 + i) * k + p];
                }
            }
            for j0 in (0..n).step_by(t) {
                let nt = t.min(n - j0);
                self.i8_tile(
                    ap,
                    bp.get(4 * j0..).unwrap_or(&[]),
                    &mut c[i0 * n + j0..],
                    groups,
                    mt,
                    nt,
                    4 * n,
                    n,
                )?;
            }
        }
        Ok(())
    }

    /// The tile-size, stride and slice checks every routine's memory safety rests on.
    fn check(
        self,
        s: &TileShape,
        a_len: usize,
        a_need: usize,
        b_len: usize,
        c_len: usize,
    ) -> Result<(), Error> {
        self.check_tile(s)?;
        self.check_bounds(s, a_len, a_need, b_len)?;
        check_c(s, c_len)
    }

    fn check_tile(self, s: &TileShape) -> Result<(), Error> {
        let t = self.tile_max();
        if s.m > t || s.n > t {
            return Err(internal(format!(
                "an SME2 tile of {} × {} exceeds {t} × {t} (twice the streaming vector length)",
                s.m, s.n
            )));
        }
        Ok(())
    }

    /// `a` holds `a_need` elements; `b` holds k rows of `s.n` with stride `s.ldb`.
    fn check_bounds(
        self,
        s: &TileShape,
        a_len: usize,
        a_need: usize,
        b_len: usize,
    ) -> Result<(), Error> {
        if s.ldb < s.n {
            return Err(internal(format!(
                "an SME2 tile's B stride {} is below its width {}",
                s.ldb, s.n
            )));
        }
        let b_need = match s.k {
            0 => 0,
            k => (k - 1) * s.ldb + s.n,
        };
        if a_len < a_need || b_len < b_need {
            return Err(internal(format!(
                "an SME2 tile needs {a_need} A and {b_need} B elements but has {a_len} and \
                 {b_len}"
            )));
        }
        Ok(())
    }
}

/// One tile call's dimensions, in the routine's own units.
#[derive(Clone, Copy, Debug)]
struct TileShape {
    /// k steps for F32; groups of four k values for i8.
    k: usize,
    m: usize,
    n: usize,
    ldb: usize,
    ldc: usize,
}

/// `c` holds m rows of `s.n` with stride `s.ldc`.
fn check_c(s: &TileShape, c_len: usize) -> Result<(), Error> {
    if s.ldc < s.n {
        return Err(internal(format!(
            "an SME2 tile's C stride {} is below its width {}",
            s.ldc, s.n
        )));
    }
    let c_need = match s.m {
        0 => 0,
        m => (m - 1) * s.ldc + s.n,
    };
    if c_len < c_need {
        return Err(internal(format!(
            "an SME2 tile needs {c_need} C elements but has {c_len}"
        )));
    }
    Ok(())
}

fn check_gemm(a: usize, b: usize, c: usize, m: usize, n: usize, k: usize) -> Result<(), Error> {
    if a != m * k || b != k * n || c != m * n {
        return Err(internal(format!(
            "an SME2 product of {m} × {k} by {k} × {n} got slices of {a}, {b} and {c} elements"
        )));
    }
    Ok(())
}

/// The token rule, apart from the host query so tests can drive it: a token exists only for
/// `Arm92Sme2` on a host that reported a streaming vector length.
fn token(isa: CpuIsa, host_svl_words: Option<usize>) -> Option<Sme2> {
    match (isa, host_svl_words) {
        (CpuIsa::Arm92Sme2, Some(svl_words)) if svl_words > 0 => Some(Sme2 { svl_words }),
        _ => None,
    }
}

/// SVL_s on this host when it reports SME2 and this build has the routines; `None` otherwise.
/// Read once per process.
fn host_svl_words() -> Option<usize> {
    static SVL: OnceLock<Option<usize>> = OnceLock::new();
    *SVL.get_or_init(ffi::host_svl_words)
}

#[cfg(syntropy_sme2)]
mod ffi {
    //! The routines' declarations and their only call sites.

    use syntropy_kernels::Error;

    use super::TileShape;
    use crate::device::internal;
    use crate::isa::{IsaInputs, KillSwitches, inputs};

    unsafe extern "C" {
        fn syn_sme2_svl_bytes() -> u64;
        fn syn_sme2_f32_tile(
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            k: u64,
            m: u64,
            n: u64,
            ldb: u64,
            ldc: u64,
        ) -> u64;
        fn syn_sme2_i8_tile(
            a: *const i8,
            b: *const i8,
            c: *mut i32,
            groups: u64,
            m: u64,
            n: u64,
            ldb: u64,
            ldc: u64,
        ) -> u64;
    }

    #[cfg(test)]
    unsafe extern "C" {
        /// The callee-saved register harness, `src/sme2/harness.S`.
        pub(super) fn syn_sme2_harness_call(
            report: *mut u64,
            routine: *const core::ffi::c_void,
            args: *const u64,
        );
        /// The deliberately wrong routine the harness's negative test calls.
        pub(super) fn syn_sme2_harness_clobber() -> u64;
    }

    /// The routines' addresses, for the harness.
    #[cfg(test)]
    pub(super) fn routines() -> [(&'static str, *const core::ffi::c_void); 3] {
        [
            ("syn_sme2_svl_bytes", syn_sme2_svl_bytes as *const _),
            ("syn_sme2_f32_tile", syn_sme2_f32_tile as *const _),
            ("syn_sme2_i8_tile", syn_sme2_i8_tile as *const _),
        ]
    }

    pub(super) fn host_svl_words() -> Option<usize> {
        match inputs(&KillSwitches::NONE) {
            Some(IsaInputs::Arm(a)) if a.sme2 => {
                // SAFETY: the host reports SME2 (`FEAT_SME2` sysctl or `HWCAP2_SME2`), so
                // RDSVL exists; it is legal outside streaming mode, reads no memory and
                // changes no state.
                let bytes = unsafe { syn_sme2_svl_bytes() };
                usize::try_from(bytes / 4).ok()
            }
            _ => None,
        }
    }

    /// Counts the tile routines this thread called, for the dispatch test.
    #[cfg(test)]
    pub(super) fn count() {
        super::tests::CALLS.with(|c| c.set(c.get() + 1));
    }

    #[cfg(not(test))]
    fn count() {}

    fn done(rc: u64) -> Result<(), Error> {
        match rc {
            0 => Ok(()),
            _ => Err(internal(
                "an SME2 routine found a ZA lazy save pending (TPIDR2_EL0 is set) and did \
                 nothing; no Rust caller holds ZA state, so another component left it set"
                    .to_owned(),
            )),
        }
    }

    pub(super) fn f32_tile(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        s: &TileShape,
    ) -> Result<(), Error> {
        count();
        // SAFETY: the caller holds an `Sme2` token, so the host has SME2 and SVL_s is the
        // token's. `Sme2::check` bounded m and n by 2 · SVL_s, and checked that `a` holds
        // k · m floats, `b` holds k rows of n with stride ldb ≥ n, and `c` holds m rows of n
        // with stride ldc ≥ n. The routine's predicated loads and stores touch no other
        // element, it saves and restores d8–d15 around streaming mode, uses no other
        // callee-saved register, and returns with PSTATE.SM and PSTATE.ZA clear.
        let rc = unsafe {
            syn_sme2_f32_tile(
                a,
                b,
                c,
                s.k as u64,
                s.m as u64,
                s.n as u64,
                s.ldb as u64,
                s.ldc as u64,
            )
        };
        done(rc)
    }

    pub(super) fn i8_tile(
        a: *const i8,
        b: *const i8,
        c: *mut i32,
        s: &TileShape,
    ) -> Result<(), Error> {
        count();
        // SAFETY: as for `f32_tile`, with `a` holding groups · 4m bytes, `b` holding `groups`
        // rows of 4n bytes with stride ldb ≥ 4n, and `c` m rows of n `i32`s with stride
        // ldc ≥ n, all checked by `Sme2::i8_tile`.
        let rc = unsafe {
            syn_sme2_i8_tile(
                a,
                b,
                c,
                s.k as u64,
                s.m as u64,
                s.n as u64,
                s.ldb as u64,
                s.ldc as u64,
            )
        };
        done(rc)
    }
}

#[cfg(not(syntropy_sme2))]
mod ffi {
    //! This build has no SME2 unit: no host reports SME2, so no token exists and nothing below
    //! the token is reachable.

    use syntropy_kernels::Error;

    use super::TileShape;

    pub(super) fn host_svl_words() -> Option<usize> {
        None
    }

    pub(super) fn f32_tile(
        _: *const f32,
        _: *const f32,
        _: *mut f32,
        _: &TileShape,
    ) -> Result<(), Error> {
        unreachable!("no `Sme2` token exists in a build without the SME2 unit")
    }

    pub(super) fn i8_tile(
        _: *const i8,
        _: *const i8,
        _: *mut i32,
        _: &TileShape,
    ) -> Result<(), Error> {
        unreachable!("no `Sme2` token exists in a build without the SME2 unit")
    }
}
