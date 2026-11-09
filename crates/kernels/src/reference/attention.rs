// SPDX-License-Identifier: MIT OR Apache-2.0

//! `OpKind::KvWrite(Gqa)`, `OpKind::AttnDecode(Gqa)` and `OpKind::AttnPrefill(Gqa)`'s reference
//! semantics over paged KV (RFC-0040 §3.1, §3.3, §3.5, §5.5; RFC-0042 §4.2; RFC-0060 §4.1;
//! COMP-E1-T17).
//!
//! **The KV pool.** A [`KvLayout`] is one layer's page geometry: `n_kv_heads`, `d_head`,
//! `page_tokens` positions per page (`P`), and the model's KV dtype (`KvFormula::dtype`,
//! RFC-0042 §4.2), one of F32, F16 and BF16. The pool is whole pages of bytes, each in the
//! `gqa.hpd` order of RFC-0040 §5.5: a K plane then a V plane, each `[kv_head][slot][d_head]`,
//! elements little-endian. Position `j` of a sequence lives in slot `j mod P` of page
//! `pages[j / P]`, where `pages` is the sequence's page table ([`Seq`]). The step's rows and
//! sequences are the step descriptor's arrays (RFC-0040 §3.5): [`Row`] holds `row_seq`,
//! `row_pos` and `row_kv_write_from`; [`Seq`] holds `seq_page_table` and `seq_kv_len`.
//!
//! **`KvWrite(Gqa)`** ([`kv_write`]) stores each active row's K and V at the row's position, for
//! every row whose position is at or above its `kv_write_from` (RFC-0040 §3.5: a replay row
//! below it writes nothing). K is rotated by the layer's RoPE in F32 ([`rope`](mod@super::rope)),
//! then K and V are rounded once to the KV dtype, to nearest with ties to even, as
//! [`f16_bits`] and [`round_bf16`] define it (RFC-0042 §4.2's `KvWrite` row). Nothing else in
//! the pool changes.
//!
//! **`AttnDecode(Gqa)` and `AttnPrefill(Gqa)`** ([`attn_decode`], [`attn_prefill`]) are one
//! definition; the kinds differ in the row bucket a provider prepares (up to 16 query rows per
//! sequence for decode, a causal chunk for prefill), never in the value. For each active row at
//! position `p` and each query head `h`, reading KV head `h / (n_q_heads / n_kv_heads)`:
//!
//! ```text
//! q      = the row's head h, rotated by the layer's RoPE at p in F32
//! s[j]   = (Σ_d q[d] · K_j[d]) · scale       for j = 0 … p    (dot in the F32Fixed order)
//! m      = max_j s[j]                         (NaN if any s[j] is NaN)
//! e[j]   = exp(s[j] − m)                      (the software exp, math::exp)
//! l      = Σ_j e[j]                           (F32Fixed order)
//! out[d] = (Σ_j e[j] · V_j[d]) / l            (F32Fixed order, one division)
//! ```
//!
//! with `K_j` and `V_j` the stored values widened exactly to F32, and `scale` the layer's
//! attention scale, `1 / sqrt(d_head)` by default (an F32 `sqrt`, then an F32 division). The
//! score row is materialised in full and the row max subtracted, as RFC-0060 §4.1's `F32Fixed`
//! specifies; there is no running rescaling. This is RFC-0042 §4.2's `AttnDecode` row: scores,
//! max, sum, `exp` and the `p·v` product all in F32.
//!
//! **Masked positions.** A row at `p` reads positions `0 … p` and nothing else: positions after
//! it (the causal mask) and slots past its sequence's KV length are never read, so whatever they
//! hold, NaN included, cannot change the output. Every active row's position must be below its
//! sequence's KV length, which counts this step's rows: `KvWrite` stores a row's own key before
//! attention reads it, so every row reads at least one position and `l ≥ 1`.
//!
//! **Edge cases.** With `rows_active = 0` nothing is read or written. Rows past `rows_active`, and
//! the output's padded rows, are neither read nor written. NaN propagates and is never clamped
//! (RFC-0042 §4.1 rule 6): a NaN in `q` or in a read key makes every score of that head NaN, so
//! `m` and the head's whole output are NaN; a NaN in a read value makes that one output dim NaN
//! for every head reading it. An infinite score makes `m` infinite and `s[j] − m` NaN somewhere
//! (`∞ − ∞`), so the head's output is NaN. A score of `−∞` below a finite max gives `e[j] = 0`.
//!
//! Every shape, sequence index, position and page is checked before anything is written, so a
//! refused call writes nothing.

use syntropy_core::DType;

use super::act_quant::{bf16_to_f32, f16_bits, f16_to_f32, round_bf16};
use super::math;
use super::reduce;
use super::rope::RopeTable;
use super::shape::{OpError, at_least};

/// One layer's KV page geometry and dtype (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KvLayout {
    n_kv_heads: usize,
    d_head: usize,
    page_tokens: usize,
    dtype: DType,
    page_bytes: usize,
}

/// The K plane or the V plane of a page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Plane {
    K = 0,
    V = 1,
}

impl KvLayout {
    /// The layout of a layer with `n_kv_heads` KV heads of `d_head` values, `page_tokens`
    /// positions per page, stored as `dtype`.
    ///
    /// # Errors
    ///
    /// [`OpError::Param`] when a dimension is 0 or a page's size overflows;
    /// [`OpError::Unsupported`] for a dtype other than F32, F16 and BF16.
    pub fn new(
        n_kv_heads: usize,
        d_head: usize,
        page_tokens: usize,
        dtype: DType,
    ) -> Result<Self, OpError> {
        let elem = match dtype {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            _ => {
                return Err(OpError::Unsupported {
                    op: "KvWrite",
                    what: "a KV dtype other than F32, F16 and BF16",
                });
            }
        };
        if n_kv_heads == 0 || d_head == 0 || page_tokens == 0 {
            return Err(OpError::Param {
                op: "KvWrite",
                what: "a KV dimension is 0",
            });
        }
        let page_bytes = [2, n_kv_heads, page_tokens, d_head, elem]
            .into_iter()
            .try_fold(1usize, usize::checked_mul)
            .ok_or(OpError::Param {
                op: "KvWrite",
                what: "a page's size overflows",
            })?;
        Ok(Self {
            n_kv_heads,
            d_head,
            page_tokens,
            dtype,
            page_bytes,
        })
    }

    /// KV heads.
    #[must_use]
    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    /// Values per head.
    #[must_use]
    pub fn d_head(&self) -> usize {
        self.d_head
    }

    /// Positions per page, `P`.
    #[must_use]
    pub fn page_tokens(&self) -> usize {
        self.page_tokens
    }

    /// The KV dtype.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Bytes per page: `2 · n_kv_heads · P · d_head` elements.
    #[must_use]
    pub fn page_bytes(&self) -> usize {
        self.page_bytes
    }

    fn elem_bytes(&self) -> usize {
        match self.dtype {
            DType::F32 => 4,
            _ => 2,
        }
    }

    /// The byte offset of the `d_head` values of (`plane`, `head`) at position `pos`, in a
    /// sequence whose page table is `pages`. The caller has checked the page.
    fn offset(&self, pages: &[u32], plane: Plane, head: usize, pos: usize) -> usize {
        let page = pages[pos / self.page_tokens] as usize;
        let slot = pos % self.page_tokens;
        let vector =
            ((plane as usize * self.n_kv_heads + head) * self.page_tokens + slot) * self.d_head;
        page * self.page_bytes + vector * self.elem_bytes()
    }

    /// Reads `out.len() = d_head` values, widened exactly to F32.
    fn read(&self, pool: &[u8], at: usize, out: &mut [f32]) {
        let eb = self.elem_bytes();
        for (i, v) in out.iter_mut().enumerate() {
            let b = &pool[at + i * eb..at + (i + 1) * eb];
            *v = match self.dtype {
                DType::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                DType::F16 => f16_to_f32(u16::from_le_bytes([b[0], b[1]])),
                _ => bf16_to_f32(u16::from_le_bytes([b[0], b[1]])),
            };
        }
    }

    /// Writes `xs`, each rounded once to the KV dtype.
    fn write(&self, pool: &mut [u8], at: usize, xs: &[f32]) {
        let eb = self.elem_bytes();
        for (i, &x) in xs.iter().enumerate() {
            let b = &mut pool[at + i * eb..at + (i + 1) * eb];
            match self.dtype {
                DType::F32 => b.copy_from_slice(&x.to_le_bytes()),
                DType::F16 => b.copy_from_slice(&f16_bits(x).to_le_bytes()),
                _ => b.copy_from_slice(&round_bf16(x).to_le_bytes()),
            }
        }
    }

    fn pool_pages(&self, op: &'static str, pool: usize) -> Result<usize, OpError> {
        if !pool.is_multiple_of(self.page_bytes) {
            return Err(OpError::Param {
                op,
                what: "the pool is not a whole number of pages",
            });
        }
        Ok(pool / self.page_bytes)
    }
}

/// One active row of the step: the step descriptor's `row_seq`, `row_pos` and
/// `row_kv_write_from` (RFC-0040 §3.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Row {
    /// The row's sequence, an index into the step's sequences.
    pub seq: u32,
    /// The row's position in its sequence.
    pub pos: u32,
    /// The first position whose KV this step writes for the row: `pos` in a normal step,
    /// above it in a replay step (RFC-0033).
    pub kv_write_from: u32,
}

/// One sequence of the step: the step descriptor's `seq_page_table` and `seq_kv_len`
/// (RFC-0040 §3.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Seq<'a> {
    /// The pool page holding each run of `P` positions, in position order.
    pub pages: &'a [u32],
    /// The sequence's KV positions after this step's writes: positions `0 … kv_len − 1`.
    pub kv_len: u32,
}

/// A `Gqa` layer's attention parameters beyond its [`KvLayout`] (module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gqa {
    /// Query heads, a whole multiple of the layout's KV heads.
    pub n_q_heads: usize,
    /// The attention scale; `None` for `1 / sqrt(d_head)`.
    pub scale: Option<f32>,
}

/// Checks that the row's sequence exists and that positions `0 … last` have pages in the pool.
fn check_pages(
    op: &'static str,
    layout: &KvLayout,
    r: usize,
    row: Row,
    last: u32,
    seqs: &[Seq<'_>],
    pool_pages: usize,
) -> Result<(), OpError> {
    let position = |what| OpError::Position {
        op,
        row: r,
        pos: row.pos,
        what,
    };
    let seq = seqs
        .get(row.seq as usize)
        .ok_or(position("its sequence is past the step's sequences"))?;
    let need = last as usize / layout.page_tokens + 1;
    let Some(pages) = seq.pages.get(..need) else {
        return Err(position("past its sequence's page table"));
    };
    if pages.iter().any(|&p| p as usize >= pool_pages) {
        return Err(position("a page it uses is past the pool"));
    }
    Ok(())
}

fn check_rope(op: &'static str, rope: Option<&RopeTable>, d_head: usize) -> Result<(), OpError> {
    if rope.is_some_and(|t| t.d_rot() > d_head) {
        return Err(OpError::Param {
            op,
            what: "d_rot exceeds d_head",
        });
    }
    Ok(())
}

/// `KvWrite(Gqa)`: stores K and V of the first `rows_active` rows, each `n_kv_heads · d_head`
/// values, at their positions (module docs).
///
/// # Errors
///
/// [`OpError::Len`] when `k` or `v` holds fewer than `rows_active` rows, or `rows` fewer than
/// `rows_active` entries; [`OpError::Param`] when the pool is not whole pages or the RoPE's
/// `d_rot` exceeds `d_head`; [`OpError::Position`] when a writing row's sequence is past `seqs`,
/// its position is past its page table, or its page is past the pool.
#[allow(
    clippy::too_many_arguments,
    reason = "the op's inputs as RFC-0040 §3.5 lists them"
)]
pub fn kv_write(
    layout: &KvLayout,
    rope: Option<&RopeTable>,
    k: &[f32],
    v: &[f32],
    rows: &[Row],
    rows_active: usize,
    seqs: &[Seq<'_>],
    pool: &mut [u8],
) -> Result<(), OpError> {
    const OP: &str = "KvWrite";
    let (n_kv, d) = (layout.n_kv_heads, layout.d_head);
    let width = n_kv * d;
    check_rope(OP, rope, d)?;
    at_least(OP, "k", k.len(), rows_active, width)?;
    at_least(OP, "v", v.len(), rows_active, width)?;
    at_least(OP, "rows", rows.len(), rows_active, 1)?;
    let pool_pages = layout.pool_pages(OP, pool.len())?;
    let writes = |row: &Row| row.pos >= row.kv_write_from;
    for (r, row) in rows[..rows_active].iter().enumerate() {
        if writes(row) {
            check_pages(OP, layout, r, *row, row.pos, seqs, pool_pages)?;
        }
    }
    let mut head = vec![0.0f32; d];
    for (r, row) in rows[..rows_active].iter().enumerate() {
        if !writes(row) {
            continue;
        }
        let pages = seqs[row.seq as usize].pages;
        let pos = row.pos as usize;
        for h in 0..n_kv {
            let at = r * width + h * d;
            head.copy_from_slice(&k[at..at + d]);
            if let Some(t) = rope {
                t.rotate(row.pos, &mut head);
            }
            layout.write(pool, layout.offset(pages, Plane::K, h, pos), &head);
            layout.write(pool, layout.offset(pages, Plane::V, h, pos), &v[at..at + d]);
        }
    }
    Ok(())
}

/// The max of `xs`, NaN if any is NaN; `xs` is not empty.
fn max_or_nan(xs: &[f32]) -> f32 {
    let mut m = xs[0];
    for &x in xs {
        if x.is_nan() {
            return x;
        }
        if x > m {
            m = x;
        }
    }
    m
}

#[allow(
    clippy::too_many_arguments,
    reason = "the op's inputs as RFC-0040 §3.5 lists them"
)]
fn attend(
    op: &'static str,
    gqa: &Gqa,
    layout: &KvLayout,
    rope: Option<&RopeTable>,
    q: &[f32],
    rows: &[Row],
    rows_active: usize,
    seqs: &[Seq<'_>],
    pool: &[u8],
    out: &mut [f32],
) -> Result<(), OpError> {
    let (n_kv, d) = (layout.n_kv_heads, layout.d_head);
    let n_q = gqa.n_q_heads;
    if n_q == 0 || !n_q.is_multiple_of(n_kv) {
        return Err(OpError::Param {
            op,
            what: "n_q_heads is not a positive multiple of n_kv_heads",
        });
    }
    check_rope(op, rope, d)?;
    let width = n_q.checked_mul(d).ok_or(OpError::Param {
        op,
        what: "n_q_heads · d_head overflows",
    })?;
    at_least(op, "q", q.len(), rows_active, width)?;
    at_least(op, "out", out.len(), rows_active, width)?;
    at_least(op, "rows", rows.len(), rows_active, 1)?;
    let pool_pages = layout.pool_pages(op, pool.len())?;
    for (r, row) in rows[..rows_active].iter().enumerate() {
        check_pages(op, layout, r, *row, row.pos, seqs, pool_pages)?;
        if row.pos >= seqs[row.seq as usize].kv_len {
            return Err(OpError::Position {
                op,
                row: r,
                pos: row.pos,
                what: "not below its sequence's KV length",
            });
        }
    }

    let group = n_q / n_kv;
    let scale = gqa.scale.unwrap_or_else(|| 1.0 / (d as f32).sqrt());
    let mut qh = vec![0.0f32; d];
    let mut kj = vec![0.0f32; d];
    let mut scores = Vec::new();
    let mut values = Vec::new();
    for (r, row) in rows[..rows_active].iter().enumerate() {
        let pages = seqs[row.seq as usize].pages;
        let n = row.pos as usize + 1;
        for h in 0..n_q {
            let kvh = h / group;
            let at = r * width + h * d;
            qh.copy_from_slice(&q[at..at + d]);
            if let Some(t) = rope {
                t.rotate(row.pos, &mut qh);
            }
            scores.clear();
            for j in 0..n {
                layout.read(pool, layout.offset(pages, Plane::K, kvh, j), &mut kj);
                scores.push(reduce::dot(&qh, &kj) * scale);
            }
            let m = max_or_nan(&scores);
            for s in &mut scores {
                *s = math::exp(*s - m);
            }
            let l = reduce::sum(&scores);
            values.resize(n * d, 0.0);
            for j in 0..n {
                let at = layout.offset(pages, Plane::V, kvh, j);
                layout.read(pool, at, &mut values[j * d..(j + 1) * d]);
            }
            for (dd, o) in out[at..at + d].iter_mut().enumerate() {
                *o = reduce::sum_by(n, |j| scores[j] * values[j * d + dd]) / l;
            }
        }
    }
    Ok(())
}

/// `AttnDecode(Gqa)`: attention of the first `rows_active` rows of `q`, each `n_q_heads ·
/// d_head` values, over the paged KV, into `out` (module docs).
///
/// # Errors
///
/// [`OpError::Param`] when `n_q_heads` is not a positive multiple of the layout's KV heads, the
/// RoPE's `d_rot` exceeds `d_head`, or the pool is not whole pages; [`OpError::Len`] when `q` or
/// `out` holds fewer than `rows_active` rows, or `rows` fewer than `rows_active` entries;
/// [`OpError::Position`] when a row's sequence is past `seqs`, its position is not below its
/// sequence's KV length or is past its page table, or a page it reads is past the pool.
#[allow(
    clippy::too_many_arguments,
    reason = "the op's inputs as RFC-0040 §3.5 lists them"
)]
pub fn attn_decode(
    gqa: &Gqa,
    layout: &KvLayout,
    rope: Option<&RopeTable>,
    q: &[f32],
    rows: &[Row],
    rows_active: usize,
    seqs: &[Seq<'_>],
    pool: &[u8],
    out: &mut [f32],
) -> Result<(), OpError> {
    attend(
        "AttnDecode",
        gqa,
        layout,
        rope,
        q,
        rows,
        rows_active,
        seqs,
        pool,
        out,
    )
}

/// `AttnPrefill(Gqa)`: [`attn_decode`]'s definition, for a causal prefill chunk (module docs).
///
/// # Errors
///
/// As [`attn_decode`].
#[allow(
    clippy::too_many_arguments,
    reason = "the op's inputs as RFC-0040 §3.5 lists them"
)]
pub fn attn_prefill(
    gqa: &Gqa,
    layout: &KvLayout,
    rope: Option<&RopeTable>,
    q: &[f32],
    rows: &[Row],
    rows_active: usize,
    seqs: &[Seq<'_>],
    pool: &[u8],
    out: &mut [f32],
) -> Result<(), OpError> {
    attend(
        "AttnPrefill",
        gqa,
        layout,
        rope,
        q,
        rows,
        rows_active,
        seqs,
        pool,
        out,
    )
}

#[cfg(test)]
mod tests;
