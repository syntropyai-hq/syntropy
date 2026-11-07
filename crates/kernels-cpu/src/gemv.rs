// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decode GEMV over GGUF weights: `Q4_K`, `Q8_0`, `Q4_0`, `BF16` and `F16`, for rows 1–8, at
//! `X86V3`, `X86V4`, `Arm82Dot` and `Arm86I8mm` (RFC-0040 §7.3 "GEMV over pinned weights",
//! §9 rows 4 and 5; ADR-0421 item 4; COMP-E3-T03).
//!
//! `y[r][o] = Σ_k W[o][k] · x[r][k]` for every active row `r`, the value
//! [`syntropy_kernels::reference::gemv`] defines on the dequantised weights. The weights stay
//! in GGUF's own block layout, one output row after another, as the pack stores them. The
//! repacked `cpu.rows4`/`rows8` layouts are COMP-E3-T04's.
//!
//! # What a kernel computes
//!
//! Every level computes the same value, bit for bit, for every input that holds no NaN. The
//! portable kernels in `portable.rs` define it; the SIMD kernels differ only in how they reach
//! it. The tests check each level against the portable kernels.
//!
//! - **Integer-coded weights** (`Q8_0`, `Q4_0`, `Q4_K`). The activation row is quantised
//!   once per call to the 8-bit block ggml's CPU path uses for the weight type
//!   ([`act`]: `Q8Block32` for `Q8_0` and `Q4_0`, `Q8K256` for `Q4_K`; RFC-0042 §4.3 rule b,
//!   ADR-0421 item 4). Each weight block's dot product with the activation block is an exact
//!   `i32` sum (RFC-0042 §4.1 rule 2), and each block contributes one F32 term:
//!   - `Q8_0`, `Q4_0`: `t = fl(fl(d_w · d_x) · s)`, with `s` the exact sum of `q_w · q_x`
//!     (`Q4_0`'s codes offset by 8);
//!   - `Q4_K`: per 256-element super-block, `t = fl(fl(fl(d_w · d_x) · S₁) −
//!     fl(fl(dmin_w · d_x) · S₂))`, where `S₁ = Σⱼ scⱼ · sⱼ` over the eight 32-element
//!     sub-blocks and `S₂ = Σⱼ mⱼ · Σ q_x` over the same sub-blocks, both exact in `i32` and
//!     rounded once to F32.
//!
//!   The terms are added in block order into an F32 accumulator that starts at `+0.0`.
//! - **Float weights** (`F16`, `BF16`). Each weight is widened exactly to F32 and the
//!   activation stays F32 (ADR-0421 item 4: no rounding of the activation). The products are
//!   fused multiply-adds into [`LANES`] = 32 F32 accumulators, element `k` into lane
//!   `k mod 32`, in increasing k. The lanes are then added pairwise: lane `j` and lane
//!   `j + h` for `h` = 16, 8, 4, 2, 1, keeping the lower lane, so the output is lane 0.
//!
//! A NaN anywhere in a weight row or an activation row makes that output NaN, as do `∞ − ∞`
//! and `0 · ∞`; NaN is never clamped (RFC-0042 §4.1 rule 6). Which NaN payload comes out may
//! differ between levels, so the tests compare NaN outputs by kind only. With
//! `rows_active = 0` nothing is written. With `k_in = 0` every output is `+0.0`.
//!
//! # Rows and partitioning
//!
//! The activation rows are processed in groups of up to [`ROW_GROUP`] = 8, with one
//! accumulator per row, so each weight block is read once per group (RFC-0040 §9 row 5). The
//! output rows a call computes are a range, [`partition`] splits a matrix's output rows into
//! contiguous runs of whole cache lines of weight bytes, one per thread of the NUMA-local
//! pool, and each thread then streams a contiguous run of its own node's weight memory
//! (RFC-0040 §7.3). The value of an output never depends on the range it was computed in.
//!
//! # Levels
//!
//! [`Level::for_isa`] picks the kernels for a `CpuIsa`, never above what the live host
//! supports: a level whose instructions the host lacks falls back to the next level down.
//! `X86V4` uses AVX-512 for the float types, and the 256-bit integer kernels compiled with
//! the AVX-512 features: one 32-element block is one 256-bit register. `Arm86I8mm` uses
//! `smmla` for two activation rows at a time, and `Arm82Dot`'s kernels for an odd last row.

use std::fmt;
use std::ops::Range;

use syntropy_core::GgmlType;
use syntropy_kernels::CpuIsa;

pub mod act;
pub(crate) mod entry;
mod portable;

#[cfg(target_arch = "aarch64")]
mod arm;
#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(test)]
mod tests;

pub use act::{ActView, Activations, quantize_into, workspace_bytes};

/// F32 accumulators of the float kernels (module docs).
pub const LANES: usize = 32;

/// The most activation rows one pass over a weight block serves (RFC-0040 §7.3, §9 row 5).
pub const ROW_GROUP: usize = 8;

/// Elements in a `Q8_0` and `Q4_0` block.
pub(crate) const QK32: usize = 32;
/// Elements in a `Q4_K` super-block.
pub(crate) const QK_K: usize = 256;

/// A weight type this module computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WeightType {
    /// F16.
    F16,
    /// BF16.
    Bf16,
    /// `Q8_0`: 32 weights in 34 bytes.
    Q8_0,
    /// `Q4_0`: 32 weights in 18 bytes.
    Q4_0,
    /// `Q4_K`: 256 weights in 144 bytes.
    Q4K,
}

impl WeightType {
    /// The type for a GGUF type, or `None` when this module has no kernel for it.
    #[must_use]
    pub const fn from_ggml(ty: GgmlType) -> Option<Self> {
        match ty {
            GgmlType::F16 => Some(Self::F16),
            GgmlType::BF16 => Some(Self::Bf16),
            GgmlType::Q8_0 => Some(Self::Q8_0),
            GgmlType::Q4_0 => Some(Self::Q4_0),
            GgmlType::Q4_K => Some(Self::Q4K),
            _ => None,
        }
    }

    /// The GGUF type.
    #[must_use]
    pub const fn ggml(self) -> GgmlType {
        match self {
            Self::F16 => GgmlType::F16,
            Self::Bf16 => GgmlType::BF16,
            Self::Q8_0 => GgmlType::Q8_0,
            Self::Q4_0 => GgmlType::Q4_0,
            Self::Q4K => GgmlType::Q4_K,
        }
    }

    /// Weights per block.
    #[must_use]
    pub const fn block_elems(self) -> usize {
        match self {
            Self::F16 | Self::Bf16 => 1,
            Self::Q8_0 | Self::Q4_0 => QK32,
            Self::Q4K => QK_K,
        }
    }

    /// Bytes per block.
    #[must_use]
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::F16 | Self::Bf16 => 2,
            Self::Q8_0 => 34,
            Self::Q4_0 => 18,
            Self::Q4K => 144,
        }
    }

    /// The activation quantisation a product over this type applies (module docs).
    #[must_use]
    pub const fn act(self) -> act::ActKind {
        match self {
            Self::F16 | Self::Bf16 => act::ActKind::F32,
            Self::Q8_0 | Self::Q4_0 => act::ActKind::Q8_0,
            Self::Q4K => act::ActKind::Q8K,
        }
    }
}

/// A weight matrix in GGUF block layout: `n_out` rows of `k_in` weights, each row whole blocks.
#[derive(Clone, Copy, Debug)]
pub struct Weights<'a> {
    ty: WeightType,
    bytes: &'a [u8],
    n_out: usize,
    k_in: usize,
    row_bytes: usize,
}

impl<'a> Weights<'a> {
    /// `bytes` as `n_out` rows of `k_in` weights of type `ty`.
    ///
    /// # Errors
    ///
    /// [`GemvError::RowLength`] when `k_in` is not whole blocks of `ty`;
    /// [`GemvError::Len`] when `bytes` does not hold exactly `n_out` rows.
    pub fn new(
        ty: WeightType,
        bytes: &'a [u8],
        n_out: usize,
        k_in: usize,
    ) -> Result<Self, GemvError> {
        if !k_in.is_multiple_of(ty.block_elems()) {
            return Err(GemvError::RowLength {
                ty,
                k_in,
                block: ty.block_elems(),
            });
        }
        let row_bytes = k_in / ty.block_elems() * ty.block_bytes();
        let need = row_bytes.checked_mul(n_out).ok_or(GemvError::Len {
            arg: "weights",
            len: bytes.len(),
            need: usize::MAX,
        })?;
        if bytes.len() != need {
            return Err(GemvError::Len {
                arg: "weights",
                len: bytes.len(),
                need,
            });
        }
        Ok(Self {
            ty,
            bytes,
            n_out,
            k_in,
            row_bytes,
        })
    }

    /// The weight type.
    #[must_use]
    pub fn ty(&self) -> WeightType {
        self.ty
    }

    /// Output rows.
    #[must_use]
    pub fn n_out(&self) -> usize {
        self.n_out
    }

    /// Weights per row.
    #[must_use]
    pub fn k_in(&self) -> usize {
        self.k_in
    }

    /// Bytes per row.
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// The bytes of output row `o`.
    pub(crate) fn row(&self, o: usize) -> &'a [u8] {
        &self.bytes[o * self.row_bytes..(o + 1) * self.row_bytes]
    }
}

/// Why a GEMV call refused its inputs. A refused call writes nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GemvError {
    /// A row is not whole blocks of the weight type.
    RowLength {
        /// The weight type.
        ty: WeightType,
        /// Weights per row.
        k_in: usize,
        /// The type's block.
        block: usize,
    },
    /// An argument holds the wrong number of values.
    Len {
        /// The argument.
        arg: &'static str,
        /// The values given.
        len: usize,
        /// The values needed: at least this many for `x` and `y`, exactly this many for the
        /// weights.
        need: usize,
    },
    /// The activations were quantised for another weight type, width or row count.
    Activations,
    /// The output range is past the matrix.
    Range {
        /// The range's end.
        end: usize,
        /// The matrix's output rows.
        n_out: usize,
    },
}

impl fmt::Display for GemvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RowLength { ty, k_in, block } => write!(
                f,
                "Gemv: a row of {k_in} weights is not whole {block}-weight {ty:?} blocks"
            ),
            Self::Len { arg, len, need } => {
                write!(f, "Gemv: `{arg}` holds {len} values, the op needs {need}")
            }
            Self::Activations => write!(
                f,
                "Gemv: the activations were quantised for another weight type, width or row count"
            ),
            Self::Range { end, n_out } => {
                write!(
                    f,
                    "Gemv: output rows up to {end} are past the matrix's {n_out}"
                )
            }
        }
    }
}

impl std::error::Error for GemvError {}

/// The kernels one call runs: a level the live host supports (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Level {
    /// Scalar Rust: the definition every other level equals.
    Portable,
    /// AVX2, FMA and F16C (`X86V3`).
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// AVX-512 F, BW, VL and DQ (`X86V4` and `X86V4Amx`).
    #[cfg(target_arch = "x86_64")]
    Avx512,
    /// NEON with SDOT (`Arm82Dot`).
    #[cfg(target_arch = "aarch64")]
    Dot,
    /// NEON with SDOT and SMMLA (`Arm86I8mm` and `Arm92Sme2`).
    #[cfg(target_arch = "aarch64")]
    I8mm,
}

impl Level {
    /// The kernels for `isa`, lowered to what the live host supports: a table built for a
    /// level the host cannot run never reaches an instruction the host lacks.
    #[must_use]
    pub fn for_isa(isa: CpuIsa) -> Self {
        Self::lowered(isa, host())
    }

    /// The kernels for `isa` on a host that supports up to `host`.
    fn lowered(isa: CpuIsa, host: Self) -> Self {
        let wanted = Self::wanted(isa);
        if wanted.rank() <= host.rank() {
            wanted
        } else {
            host
        }
    }

    /// The kernels `isa` names, before the host check.
    fn wanted(isa: CpuIsa) -> Self {
        match isa {
            #[cfg(target_arch = "x86_64")]
            CpuIsa::X86V3 => Self::Avx2,
            #[cfg(target_arch = "x86_64")]
            CpuIsa::X86V4 | CpuIsa::X86V4Amx => Self::Avx512,
            #[cfg(target_arch = "aarch64")]
            CpuIsa::Arm82Dot => Self::Dot,
            #[cfg(target_arch = "aarch64")]
            CpuIsa::Arm86I8mm | CpuIsa::Arm92Sme2 => Self::I8mm,
            _ => Self::Portable,
        }
    }

    /// The level's place in its chain: a host that supports one level supports every lower one.
    fn rank(self) -> u8 {
        match self {
            Self::Portable => 0,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => 1,
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => 2,
            #[cfg(target_arch = "aarch64")]
            Self::Dot => 1,
            #[cfg(target_arch = "aarch64")]
            Self::I8mm => 2,
        }
    }
}

/// The highest level the live host supports, from the standard library's detection, which
/// caches its result.
fn host() -> Level {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::is_x86_feature_detected as has;
        if has!("avx512f") && has!("avx512bw") && has!("avx512vl") && has!("avx512dq") {
            return Level::Avx512;
        }
        if has!("avx2") && has!("fma") && has!("f16c") {
            return Level::Avx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::is_aarch64_feature_detected as has;
        if has!("dotprod") && has!("i8mm") {
            return Level::I8mm;
        }
        if has!("dotprod") {
            return Level::Dot;
        }
    }
    Level::Portable
}

/// The output rows of `n_out`, split into `parts` contiguous runs for the threads of one
/// NUMA-local pool (module docs). Each run but the last starts on a multiple of `align` rows,
/// chosen so a run's weight bytes start on a cache line where the row size allows; the runs
/// are as even as that alignment lets them be. Empty runs come last. `parts = 0` is taken as 1.
pub fn partition(w: &Weights<'_>, parts: usize) -> impl Iterator<Item = Range<usize>> {
    let n = w.n_out();
    let parts = parts.max(1);
    let align = row_align(w.row_bytes());
    // Whole aligned units, shared as evenly as possible; the first `extra` runs take one more.
    let units = n.div_ceil(align);
    let (base, extra) = (units / parts, units % parts);
    (0..parts).map(move |p| {
        let start_unit = p * base + p.min(extra);
        let end_unit = start_unit + base + usize::from(p < extra);
        (start_unit * align).min(n)..(end_unit * align).min(n)
    })
}

/// The fewest rows whose bytes are a whole number of 64-byte cache lines, capped at 8 so a
/// small matrix still spreads across threads.
fn row_align(row_bytes: usize) -> usize {
    const LINE: usize = 64;
    (1..=8)
        .find(|r| (r * row_bytes).is_multiple_of(LINE))
        .unwrap_or(1)
}

/// `y[r][o] = Σ_k W[o][k] · x[r][k]` for every row `r` that `x` holds and every `o` in
/// `out`, with the kernels of `level` (module docs). `x` was quantised for `w`'s type and
/// width ([`quantize_into`], [`Activations::quantize`]). `y` holds `x.rows()` rows of
/// `w.n_out()` values; only the outputs in `out` are written.
///
/// # Errors
///
/// [`GemvError::Activations`] when `x` was quantised for another type or width;
/// [`GemvError::Range`] when `out` ends past `w.n_out()`; [`GemvError::Len`] when `y` is too
/// short.
pub fn gemv(
    level: Level,
    w: &Weights<'_>,
    x: &ActView<'_>,
    out: Range<usize>,
    y: &mut [f32],
) -> Result<(), GemvError> {
    if x.kind() != w.ty().act() || x.k() != w.k_in() {
        return Err(GemvError::Activations);
    }
    if out.end > w.n_out() {
        return Err(GemvError::Range {
            end: out.end,
            n_out: w.n_out(),
        });
    }
    let need = x.rows() * w.n_out();
    if y.len() < need {
        return Err(GemvError::Len {
            arg: "y",
            len: y.len(),
            need,
        });
    }
    if x.rows() == 0 || out.is_empty() {
        return Ok(());
    }
    let job = Job {
        w,
        x,
        out,
        n_out: w.n_out(),
    };
    match level {
        Level::Portable => portable::run(&job, y),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: a `Level` other than `Portable` comes only from `Level::for_isa`, which never
        // returns a level above what the live host supports (`host`), so the host has AVX2,
        // FMA and F16C.
        Level::Avx2 => unsafe { x86::run_avx2(&job, y) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: as above; `Avx512` means the host has AVX-512 F, BW, VL and DQ, and every
        // AVX-512 host has AVX2, FMA and F16C.
        Level::Avx512 => unsafe { x86::run_avx512(&job, y) },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: as above; `Dot` means the host has the dot-product instructions.
        Level::Dot => unsafe { arm::run_dot(&job, y) },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: as above; `I8mm` means the host has the dot-product and I8MM instructions.
        Level::I8mm => unsafe { arm::run_i8mm(&job, y) },
    }
    Ok(())
}

/// One checked call: what every level's kernels read.
pub(crate) struct Job<'w, 'a> {
    pub(crate) w: &'w Weights<'a>,
    pub(crate) x: &'w ActView<'a>,
    pub(crate) out: Range<usize>,
    pub(crate) n_out: usize,
}

/// The row groups of a job: `(first row, rows)` with at most [`ROW_GROUP`] rows each.
pub(crate) fn row_groups(rows: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..rows)
        .step_by(ROW_GROUP)
        .map(move |r0| (r0, (rows - r0).min(ROW_GROUP)))
}

/// Adds the [`LANES`] float accumulators pairwise into lane 0 (module docs).
#[inline(always)]
pub(crate) fn fold_lanes(mut lanes: [f32; LANES]) -> f32 {
    let mut h = LANES / 2;
    while h > 0 {
        for j in 0..h {
            lanes[j] += lanes[j + h];
        }
        h /= 2;
    }
    lanes[0]
}

/// The F32 value of the F16 scale in `bytes[0..2]`.
#[inline(always)]
pub(crate) fn half(bytes: &[u8]) -> f32 {
    syntropy_kernels::reference::act_quant::f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]))
}

/// ggml's `get_scale_min_k4`: sub-block `j`'s 6-bit scale and minimum from a `Q4_K` block's 12
/// packed bytes.
#[inline(always)]
pub(crate) fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// One `Q4_K` super-block's F32 term from its exact sums (module docs).
#[inline(always)]
pub(crate) fn q4k_term(d: f32, dmin: f32, dx: f32, s1: i32, s2: i32) -> f32 {
    (d * dx) * (s1 as f32) - (dmin * dx) * (s2 as f32)
}

/// One 32-element block's F32 term from its exact sum (module docs).
#[inline(always)]
pub(crate) fn q32_term(dw: f32, dx: f32, s: i32) -> f32 {
    (dw * dx) * (s as f32)
}
