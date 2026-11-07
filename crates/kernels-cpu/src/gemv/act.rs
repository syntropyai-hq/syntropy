// SPDX-License-Identifier: MIT OR Apache-2.0

//! The activation rows of one GEMV call, quantised once to the block the weight type's product
//! uses (RFC-0042 §4.3 rule b; ADR-0421 item 4).
//!
//! - [`ActKind::Q8_0`] (`ActQuant::Q8Block32`), for `Q8_0` and `Q4_0` weights: per 32
//!   elements, `d = amax / 127`, `q = round_half_even(x · (1/d))`, and the scale the product
//!   uses is `d` rounded to F16. This is ggml's arm64 quantizer, the `Q8Path::Arm64Neon` of
//!   [`quantize_q8_0`](syntropy_kernels::reference::act_quant::quantize_q8_0), on every level
//!   and every architecture, so a level never changes the quantised values. ggml's x86 path
//!   differs from it only where `x · id` falls next to a half integer.
//! - [`ActKind::Q8K`] (`ActQuant::Q8K256`), for `Q4_K` weights: ggml's one `Q8_K` quantizer,
//!   [`quantize_q8_k`](syntropy_kernels::reference::act_quant::quantize_q8_k).
//! - [`ActKind::F32`], for `F16` and `BF16` weights: the row as it is.
//!
//! A block holding a NaN or an infinity has a NaN scale and zero codes, as the references
//! define, so the NaN reaches every output the row feeds.
//!
//! [`quantize_into`] writes into a launch's workspace and allocates nothing; it is what the
//! `Gemv` kernel uses. [`Activations`] owns its buffers, which grow to the largest call and are
//! reused after that.

use syntropy_kernels::reference::act_quant::{f16_bits, f16_to_f32};

use super::{GemvError, QK_K, QK32};

/// Which block the activations were quantised to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ActKind {
    /// Not quantised: F32.
    F32,
    /// `Q8Block32`: ggml's `Q8_0`.
    Q8_0,
    /// `Q8K256`: ggml's `Q8_K`.
    Q8K,
}

/// The values one block of `kind` covers.
const fn block(kind: ActKind) -> usize {
    match kind {
        ActKind::F32 => 1,
        ActKind::Q8_0 => QK32,
        ActKind::Q8K => QK_K,
    }
}

/// The counts of codes, scales and sums for `n` values of `kind`.
const fn counts(kind: ActKind, n: usize) -> (usize, usize, usize) {
    match kind {
        ActKind::F32 => (0, 0, 0),
        ActKind::Q8_0 => (n, n / QK32, 0),
        ActKind::Q8K => (n, n / QK_K, n / QK32),
    }
}

/// Checks a call's shape: `k` whole blocks of `kind`, and `x` at least `rows · k` values.
/// Returns `rows · k`.
fn check(kind: ActKind, x: &[f32], rows: usize, k: usize) -> Result<usize, GemvError> {
    if !k.is_multiple_of(block(kind)) {
        return Err(GemvError::Activations);
    }
    let n = rows.checked_mul(k).ok_or(GemvError::Activations)?;
    if x.len() < n {
        return Err(GemvError::Len {
            arg: "x",
            len: x.len(),
            need: n,
        });
    }
    Ok(n)
}

/// Quantises `x` (whole blocks of `kind`) into `qs`, `d` and `sums`, sized by [`counts`].
fn fill(kind: ActKind, x: &[f32], qs: &mut [i8], d: &mut [f32], sums: &mut [i32]) {
    match kind {
        ActKind::F32 => {}
        ActKind::Q8_0 => {
            for ((x, qs), d) in x
                .as_chunks::<QK32>()
                .0
                .iter()
                .zip(qs.as_chunks_mut::<QK32>().0)
                .zip(d)
            {
                *d = q8_0_block(x, qs);
            }
        }
        ActKind::Q8K => {
            for ((x, qs), d) in x
                .as_chunks::<QK_K>()
                .0
                .iter()
                .zip(qs.as_chunks_mut::<QK_K>().0)
                .zip(d)
            {
                *d = q8_k_block(x, qs);
            }
            for (s, qs) in sums.iter_mut().zip(qs.as_chunks::<QK32>().0) {
                *s = qs.iter().map(|&q| i32::from(q)).sum();
            }
        }
    }
}

/// The scratch bytes [`quantize_into`] needs for `rows` rows of `k` values of `kind`: the
/// scales and sums (4 bytes each), then the codes. 0 for `F32`, which is read in place.
#[must_use]
pub const fn workspace_bytes(kind: ActKind, rows: usize, k: usize) -> usize {
    let (codes, scales, sums) = counts(kind, rows * k);
    4 * (scales + sums) + codes
}

/// Quantises the first `rows` rows of `x`, `k` values each, to `kind`'s blocks in `ws`, and
/// returns the view a GEMV reads. `F32` rows are read from `x` in place. `ws` must hold
/// [`workspace_bytes`] and, when that is not 0, start on a 4-byte boundary. Allocation-free: this is how the `Gemv`
/// kernel quantises into its launch's workspace (RFC-0040 §5.3).
///
/// # Errors
///
/// As [`Activations::quantize`], and [`GemvError::Len`] for a workspace too short or not
/// 4-byte aligned.
pub fn quantize_into<'a>(
    kind: ActKind,
    x: &'a [f32],
    rows: usize,
    k: usize,
    ws: &'a mut [u8],
) -> Result<ActView<'a>, GemvError> {
    let n = check(kind, x, rows, k)?;
    let need = workspace_bytes(kind, rows, k);
    let (codes, scales, sums) = counts(kind, n);
    if ws.len() < need || (need > 0 && !(ws.as_ptr() as usize).is_multiple_of(4)) {
        return Err(GemvError::Len {
            arg: "workspace",
            len: ws.len(),
            need,
        });
    }
    let (words, rest) = ws.split_at_mut(4 * (scales + sums));
    // SAFETY: `words` starts on a 4-byte boundary (checked) and holds `scales + sums` 4-byte
    // values; every bit pattern is a valid `f32` and `i32`. `align_to_mut` then gives an empty
    // prefix and suffix.
    let (pre, words, post) = unsafe { words.align_to_mut::<u32>() };
    debug_assert!(pre.is_empty() && post.is_empty());
    let (d, s) = words.split_at_mut(scales);
    // SAFETY: `u32`, `f32` and `i32` have one size and alignment, and every bit pattern is
    // valid for each; the two slices are disjoint parts of `ws`, borrowed for `'a`.
    let (d, s) = unsafe {
        (
            std::slice::from_raw_parts_mut(d.as_mut_ptr().cast::<f32>(), d.len()),
            std::slice::from_raw_parts_mut(s.as_mut_ptr().cast::<i32>(), s.len()),
        )
    };
    let qs = &mut rest[..codes];
    // SAFETY: `u8` and `i8` have one size and alignment, and every bit pattern is valid.
    let qs = unsafe { std::slice::from_raw_parts_mut(qs.as_mut_ptr().cast::<i8>(), qs.len()) };
    fill(kind, &x[..n], qs, d, s);
    Ok(ActView {
        kind,
        rows,
        k,
        f: if kind == ActKind::F32 { &x[..n] } else { &[] },
        qs,
        d,
        sums: s,
    })
}

/// One call's activation rows as a GEMV reads them: quantised to `kind`'s blocks, or F32.
#[derive(Clone, Copy, Debug)]
pub struct ActView<'a> {
    kind: ActKind,
    rows: usize,
    k: usize,
    /// `F32`: the rows.
    f: &'a [f32],
    /// `Q8_0`, `Q8K`: the codes, `rows · k`.
    qs: &'a [i8],
    /// `Q8_0`, `Q8K`: each block's scale, `rows · k / block`.
    d: &'a [f32],
    /// `Q8K`: the sum of each 32 consecutive codes, `rows · k / 32`. (ggml keeps sums of 16;
    /// the products only ever add them in pairs.)
    sums: &'a [i32],
}

impl ActView<'_> {
    /// The block the activations were quantised to.
    #[must_use]
    pub fn kind(&self) -> ActKind {
        self.kind
    }

    /// Rows held.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Values per row.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// `F32`: row `r`.
    pub(crate) fn f32_row(&self, r: usize) -> &[f32] {
        &self.f[r * self.k..(r + 1) * self.k]
    }

    /// `Q8_0`, `Q8K`: row `r`'s codes.
    pub(crate) fn qs_row(&self, r: usize) -> &[i8] {
        &self.qs[r * self.k..(r + 1) * self.k]
    }

    /// `Q8_0`: row `r`'s block scales; `Q8K`: its super-block scales.
    pub(crate) fn d_row(&self, r: usize) -> &[f32] {
        let per = self.k / block(self.kind);
        &self.d[r * per..(r + 1) * per]
    }

    /// `Q8K`: row `r`'s sums of each 32 codes.
    pub(crate) fn sums_row(&self, r: usize) -> &[i32] {
        let per = self.k / QK32;
        &self.sums[r * per..(r + 1) * per]
    }
}

/// One call's quantised activation rows, in buffers this value owns (module docs).
#[derive(Clone, Debug)]
pub struct Activations {
    kind: ActKind,
    rows: usize,
    k: usize,
    f: Vec<f32>,
    qs: Vec<i8>,
    d: Vec<f32>,
    sums: Vec<i32>,
}

impl Default for Activations {
    fn default() -> Self {
        Self::new()
    }
}

impl Activations {
    /// Empty activations; the first [`quantize`](Self::quantize) sizes them.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            kind: ActKind::F32,
            rows: 0,
            k: 0,
            f: Vec::new(),
            qs: Vec::new(),
            d: Vec::new(),
            sums: Vec::new(),
        }
    }

    /// Activations with room for `rows` rows of `k` values of any kind, so calls up to that
    /// size never allocate.
    #[must_use]
    pub fn with_capacity(rows: usize, k: usize) -> Self {
        let n = rows * k;
        Self {
            f: Vec::with_capacity(n),
            qs: Vec::with_capacity(n),
            d: Vec::with_capacity(n / QK32),
            sums: Vec::with_capacity(n / QK32),
            ..Self::new()
        }
    }

    /// Quantises the first `rows` rows of `x`, `k` values each, to `kind`'s blocks, replacing
    /// what was held.
    ///
    /// # Errors
    ///
    /// [`GemvError::Len`] when `x` holds fewer than `rows · k` values;
    /// [`GemvError::Activations`] when `k` is not whole blocks of `kind`. Nothing changes on
    /// an error.
    pub fn quantize(
        &mut self,
        kind: ActKind,
        x: &[f32],
        rows: usize,
        k: usize,
    ) -> Result<(), GemvError> {
        let n = check(kind, x, rows, k)?;
        let (codes, scales, sums) = counts(kind, n);
        (self.kind, self.rows, self.k) = (kind, rows, k);
        self.f.clear();
        if kind == ActKind::F32 {
            self.f.extend_from_slice(&x[..n]);
        }
        self.qs.clear();
        self.qs.resize(codes, 0);
        self.d.clear();
        self.d.resize(scales, 0.0);
        self.sums.clear();
        self.sums.resize(sums, 0);
        fill(kind, &x[..n], &mut self.qs, &mut self.d, &mut self.sums);
        Ok(())
    }

    /// The view a GEMV reads.
    #[must_use]
    pub fn view(&self) -> ActView<'_> {
        ActView {
            kind: self.kind,
            rows: self.rows,
            k: self.k,
            f: &self.f,
            qs: &self.qs,
            d: &self.d,
            sums: &self.sums,
        }
    }
}

/// One `Q8_0` block as ggml's arm64 quantizer gives it; returns the scale the product uses.
fn q8_0_block(x: &[f32], qs: &mut [i8]) -> f32 {
    if x.iter().any(|v| !v.is_finite()) {
        qs.fill(0);
        return f32::NAN;
    }
    let amax = x
        .iter()
        .fold(0.0f32, |m, v| if m > v.abs() { m } else { v.abs() });
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    if id.is_finite() {
        for (q, &v) in qs.iter_mut().zip(x) {
            // |v · id| ≤ 127 plus rounding, so the conversion is exact.
            *q = (v * id).round_ties_even() as i8;
        }
    } else {
        qs.fill(0);
    }
    f16_to_f32(f16_bits(d))
}

/// ggml's `nearest_int`: rounds half to even through the F32 sum `f + 1.5 · 2^23`.
fn nearest_int(f: f32) -> i32 {
    let v = f + 12_582_912.0;
    (v.to_bits() & 0x007F_FFFF) as i32 - 0x0040_0000
}

/// One `Q8_K` block as ggml's `quantize_row_q8_K_ref` gives it; returns its scale.
fn q8_k_block(x: &[f32], qs: &mut [i8]) -> f32 {
    qs.fill(0);
    if x.iter().any(|v| !v.is_finite()) {
        return f32::NAN;
    }
    let (mut amax, mut max) = (0.0f32, 0.0f32);
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    if amax == 0.0 {
        return 0.0;
    }
    let iscale = -127.0 / max;
    if iscale.is_finite() {
        for (q, &v) in qs.iter_mut().zip(x) {
            *q = nearest_int(iscale * v).min(127) as i8;
        }
    }
    1.0 / iscale
}
