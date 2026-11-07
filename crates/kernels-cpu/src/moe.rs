// SPDX-License-Identifier: MIT OR Apache-2.0

//! Expert misses on the CPU: `MoeGrouped` at the `CPU` site for GGUF `Q4_K` and `Q8_0` experts
//! (RFC-0040 §7.3 "Expert misses", §3.4; ADR-0420 item 4; COMP-E3-T05).
//!
//! # What it computes
//!
//! For every active row `r` and every rank `j < top_k`, with `e = ids[r][j]` an expert this
//! site computes, the expert's gated FFN of the row:
//!
//! ```text
//! h[i]          = silu(gate_e[i] · x[r]) · (up_e[i] · x[r])        for i < inter
//! out[r][j][o]  = down_e[o] · h                                     for o < hidden
//! ```
//!
//! That is `kernels::reference`'s `GateUpAct` with SiLU gating followed by its `Gemv`, on the
//! dequantised weights (RFC-0040 §3.3): the activation is applied to the rounded gate, the
//! product is rounded once, and `silu` is [`reference::math::silu`](syntropy_kernels::reference::math::silu).
//! Each product is the [`gemv`](crate::gemv) module's, so it quantises its activation rows to
//! the weight type's 8-bit blocks (`Q8K256` for `Q4_K`, `Q8Block32` for `Q8_0`; RFC-0042 §4.3
//! rule b) and sums each block exactly in `i32`: `x` before the gate and up products, `h`
//! before the down product. The output is F32, written per (row, rank) to
//! `out[(r · top_k + j) · hidden + o]`, the per-(row, expert) block `MoeCombine` reads
//! (§3.4 step 6).
//!
//! A slot whose expert this site does not compute is left as it was: on a stage with a GPU,
//! the hit and copy kernels fill those slots, and on a CPU-only stage every expert is `CPU`
//! (§3.4). The caller says which experts are this site's through the expert source it passes.
//!
//! # Grouping, and why the value does not depend on it
//!
//! The slots are grouped by expert, in ascending expert index, and within an expert in
//! ascending (row, rank) order. Each expert's rows are gathered, quantised once, and computed
//! in one pass over its weights. Every output of the `gemv` module's kernels depends only on
//! its own weight row and activation row, never on the other rows in the call, the row
//! partition or the level (the `gemv` module docs). So each `out[r][j]` equals the expert's
//! FFN of row `r` alone, bit for bit, whichever rows chose the same expert and in whatever
//! order the rows come. That is the property the row-order test checks.
//!
//! # Tasks
//!
//! The work is tasks per (expert, output-row block) ([`Task`]): first the gate and up blocks
//! of `inter` rows, which write that block of `h`, then the down blocks of `hidden` rows,
//! which write that block of every slot's output. A block holds at most [`BLOCK_BYTES`] of one
//! matrix's weights (whole rows). A task streams its block once, and its `gemv` call serves
//! all the expert's rows against it: in groups of up to [`ROW_GROUP`](crate::gemv::ROW_GROUP)
//! rows, so with more than eight rows the later groups re-read a block that was just read and
//! is at most [`BLOCK_BYTES`] \[target\]. [`moe_grouped`] runs the tasks in order on the
//! calling thread. Spreading them over a NUMA domain's pool is the pool's (PLC-E7-T07): the
//! gate and up tasks of one expert are independent, and so are its down tasks once every gate
//! and up task of that expert has run.
//!
//! [`WeightReads`] counts, per expert, the weight bytes the tasks stream. One pass reads each
//! routed expert's gate, up and down bytes exactly once, whatever the number of rows that
//! chose it, and an expert no row chose not at all. It counts the schedule's reads in
//! software; the memory traffic a hardware counter sees is the benchmark's (COMP-E2-T09).
//!
//! # Edge cases
//!
//! With `rows = 0`, or no slot routed to an expert of this site, nothing is written. A row
//! that lists one expert at two ranks, which top-k routing never does, is computed once and
//! its output written to both slots. A NaN in a
//! weight or an activation propagates to the outputs it reaches, as in the `gemv` module. An
//! expert index outside `0..n_experts` refuses the call, and a refused call writes nothing.

use std::fmt;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

use syntropy_kernels::reference::math::silu;

use crate::gemv::act::ActKind;
use crate::gemv::{GemvError, Level, WeightType, Weights, gemv, quantize_into};

pub(crate) mod entry;

#[cfg(test)]
mod tests;

/// The most weight bytes of one matrix a task streams: 16 KiB \[target\], a chosen budget
/// that fits a block in the level-1 data cache of the Tier 1 CPUs, so the row groups after the
/// first re-read it from there (module docs).
pub const BLOCK_BYTES: usize = 16 * 1024;

/// The layer's MoE shape: `MoeGrouped`'s dims `[n_experts, top_k, hidden, inter]`
/// (RFC-0040 §3.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MoeShape {
    /// Routed experts in the layer.
    pub n_experts: usize,
    /// Experts each row selects.
    pub top_k: usize,
    /// The model width: the gate and up products' input, the down product's output.
    pub hidden: usize,
    /// Each expert's intermediate width.
    pub inter: usize,
}

/// One expert's weights in GGUF block layout: `gate` and `up` are `[inter, hidden]` and of
/// one type, `down` is `[hidden, inter]`.
#[derive(Clone, Copy, Debug)]
pub struct Expert<'a> {
    gate: Weights<'a>,
    up: Weights<'a>,
    down: Weights<'a>,
}

impl<'a> Expert<'a> {
    /// The expert of `gate`, `up` and `down`.
    ///
    /// # Errors
    ///
    /// [`MoeError::Expert`] when the shapes do not form one expert FFN, or `gate` and `up`
    /// differ in type.
    pub fn new(gate: Weights<'a>, up: Weights<'a>, down: Weights<'a>) -> Result<Self, MoeError> {
        let (inter, hidden) = (gate.n_out(), gate.k_in());
        if up.ty() != gate.ty()
            || (up.n_out(), up.k_in()) != (inter, hidden)
            || (down.n_out(), down.k_in()) != (hidden, inter)
        {
            return Err(MoeError::Expert);
        }
        Ok(Self { gate, up, down })
    }

    /// The gate and up products' weight type.
    #[must_use]
    pub fn gate_type(&self) -> WeightType {
        self.gate.ty()
    }

    /// The down product's weight type.
    #[must_use]
    pub fn down_type(&self) -> WeightType {
        self.down.ty()
    }

    /// The expert's weight bytes: gate, up and down.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.gate.row_bytes() * self.gate.n_out()
            + self.up.row_bytes() * self.up.n_out()
            + self.down.row_bytes() * self.down.n_out()
    }

    /// Whether this expert fits `shape`.
    fn fits(&self, shape: &MoeShape) -> bool {
        (self.gate.n_out(), self.gate.k_in()) == (shape.inter, shape.hidden)
    }
}

/// Why [`moe_grouped`] refused its inputs. A refused call writes nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MoeError {
    /// An expert's matrices do not form one expert FFN of the layer's shape, or its gate and
    /// up types differ.
    Expert,
    /// An argument holds too few values.
    Len {
        /// The argument.
        arg: &'static str,
        /// The values given.
        len: usize,
        /// The values needed.
        need: usize,
    },
    /// A route entry names no expert of the layer.
    ExpertId {
        /// The row.
        row: usize,
        /// The rank within the row.
        rank: usize,
        /// The id it holds.
        id: i32,
    },
    /// A product refused its operands.
    Gemv(GemvError),
}

impl fmt::Display for MoeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Expert => write!(
                f,
                "MoeGrouped: an expert's gate, up and down matrices do not form the layer's FFN"
            ),
            Self::Len { arg, len, need } => {
                write!(
                    f,
                    "MoeGrouped: `{arg}` holds {len} values, the op needs {need}"
                )
            }
            Self::ExpertId { row, rank, id } => write!(
                f,
                "MoeGrouped: row {row} rank {rank} routes to expert {id}, which the layer lacks"
            ),
            Self::Gemv(e) => write!(f, "MoeGrouped: {e}"),
        }
    }
}

impl std::error::Error for MoeError {}

impl From<GemvError> for MoeError {
    fn from(e: GemvError) -> Self {
        Self::Gemv(e)
    }
}

/// Per-expert counters of the weight bytes the tasks stream (module docs). Adding to one is a
/// relaxed atomic add, so tasks on several threads may share the counters.
#[derive(Debug)]
pub struct WeightReads(Vec<AtomicU64>);

impl WeightReads {
    /// Zeroed counters for `n_experts` experts.
    #[must_use]
    pub fn new(n_experts: usize) -> Self {
        Self((0..n_experts).map(|_| AtomicU64::new(0)).collect())
    }

    /// The bytes counted for `expert` so far; 0 for an expert past the counters.
    #[must_use]
    pub fn get(&self, expert: usize) -> u64 {
        self.0.get(expert).map_or(0, |c| c.load(Ordering::Relaxed))
    }

    /// Sets every counter to 0.
    pub fn reset(&self) {
        for c in &self.0 {
            c.store(0, Ordering::Relaxed);
        }
    }

    fn add(&self, expert: usize, bytes: usize) {
        if let Some(c) = self.0.get(expert) {
            c.fetch_add(bytes as u64, Ordering::Relaxed);
        }
    }
}

/// Which product a [`Task`] computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Phase {
    /// A block of the gate and up products, and that block of `h`.
    GateUp,
    /// A block of the down product, scattered into each slot's output.
    Down,
}

/// One task: a block of output rows of one expert's product (module docs).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Task {
    /// The expert.
    pub expert: u32,
    /// The product.
    pub phase: Phase,
    /// The product's output rows this task computes.
    pub out: Range<usize>,
}

/// Rows per task block of `w`: as many whole rows as fit [`BLOCK_BYTES`], and at least one.
#[must_use]
pub fn block_rows(w: &Weights<'_>) -> usize {
    (BLOCK_BYTES / w.row_bytes().max(1)).max(1)
}

/// The blocks of `w`'s output rows, in order.
fn blocks(w: &Weights<'_>) -> impl Iterator<Item = Range<usize>> {
    let (n, step) = (w.n_out(), block_rows(w));
    (0..n).step_by(step).map(move |s| s..(s + step).min(n))
}

/// The tasks one expert's pass runs, in the order [`moe_grouped`] runs them: its gate and up
/// blocks, then its down blocks.
pub fn expert_tasks(expert: u32, e: &Expert<'_>) -> impl Iterator<Item = Task> {
    let gate_up = blocks(&e.gate).map(move |out| Task {
        expert,
        phase: Phase::GateUp,
        out,
    });
    let down = blocks(&e.down).map(move |out| Task {
        expert,
        phase: Phase::Down,
        out,
    });
    gate_up.chain(down)
}

/// The quantised-activation bytes the larger of the two 8-bit block kinds needs.
fn act_bytes(rows: usize, k: usize) -> usize {
    crate::gemv::workspace_bytes(ActKind::Q8_0, rows, k).max(crate::gemv::workspace_bytes(
        ActKind::Q8K,
        rows,
        k,
    ))
}

/// The scratch bytes [`moe_grouped`] needs for `rows` rows of `shape`, for any expert type the
/// `gemv` module computes, including 4 bytes of slack so the scratch may start anywhere.
#[must_use]
pub fn workspace_bytes(shape: &MoeShape, rows: usize) -> usize {
    let words = (shape.n_experts + 1) + rows * shape.top_k;
    let f32s = rows * shape.hidden + 2 * rows * shape.inter + rows * shape.hidden;
    let acts = act_bytes(rows, shape.hidden).next_multiple_of(4)
        + act_bytes(rows, shape.inter).next_multiple_of(4);
    4 + 4 * (words + f32s) + acts
}

/// The scratch, carved from the caller's bytes.
struct Scratch<'a> {
    /// `offsets[e]..offsets[e + 1]` are expert `e`'s entries of `slots`.
    offsets: &'a mut [u32],
    /// Slot indices `r · top_k + j`, grouped by expert.
    slots: &'a mut [u32],
    /// The expert's gathered input rows, `[rows, hidden]`.
    xg: &'a mut [f32],
    /// The gate product, then `h`, `[rows, inter]`.
    g: &'a mut [f32],
    /// The up product, `[rows, inter]`.
    u: &'a mut [f32],
    /// The down product, `[rows, hidden]`.
    yd: &'a mut [f32],
    /// Quantised `xg`.
    xq: &'a mut [u8],
    /// Quantised `h`.
    hq: &'a mut [u8],
}

/// Takes `n` 4-byte words from the front of `ws`, which starts on a 4-byte boundary.
fn take_words<'a>(ws: &mut &'a mut [u8], n: usize) -> &'a mut [u32] {
    let (head, tail) = std::mem::take(ws).split_at_mut(4 * n);
    *ws = tail;
    // SAFETY: `head` starts on a 4-byte boundary (the caller's invariant, kept because every
    // piece taken is a multiple of 4 bytes) and holds `4 · n` bytes; every bit pattern is a
    // valid `u32`. `align_to_mut` then gives an empty prefix and suffix.
    let (pre, words, post) = unsafe { head.align_to_mut::<u32>() };
    debug_assert!(pre.is_empty() && post.is_empty());
    words
}

/// [`take_words`] as `f32`s.
fn take_f32s<'a>(ws: &mut &'a mut [u8], n: usize) -> &'a mut [f32] {
    let w = take_words(ws, n);
    // SAFETY: `u32` and `f32` have one size and alignment, and every bit pattern is valid for
    // each; `w` is the only view of its bytes.
    unsafe { std::slice::from_raw_parts_mut(w.as_mut_ptr().cast::<f32>(), w.len()) }
}

impl<'a> Scratch<'a> {
    /// Carves the scratch for `rows` rows of `shape` from `ws`, which holds
    /// [`workspace_bytes`].
    fn carve(ws: &'a mut [u8], shape: &MoeShape, rows: usize) -> Result<Self, MoeError> {
        let need = workspace_bytes(shape, rows);
        if ws.len() < need {
            return Err(MoeError::Len {
                arg: "workspace",
                len: ws.len(),
                need,
            });
        }
        let skip = ws.as_ptr().align_offset(4).min(4);
        let mut ws = &mut ws[skip..];
        let offsets = take_words(&mut ws, shape.n_experts + 1);
        let slots = take_words(&mut ws, rows * shape.top_k);
        let xg = take_f32s(&mut ws, rows * shape.hidden);
        let g = take_f32s(&mut ws, rows * shape.inter);
        let u = take_f32s(&mut ws, rows * shape.inter);
        let yd = take_f32s(&mut ws, rows * shape.hidden);
        let (xq, rest) = ws.split_at_mut(act_bytes(rows, shape.hidden).next_multiple_of(4));
        let hq = &mut rest[..act_bytes(rows, shape.inter).next_multiple_of(4)];
        Ok(Self {
            offsets,
            slots,
            xg,
            g,
            u,
            yd,
            xq,
            hq,
        })
    }
}

/// `MoeGrouped` at the `CPU` site (module docs): for the first `rows` rows of `x`
/// (`[rows, hidden]`) and `ids` (`[rows, top_k]`), every slot whose expert `experts` returns
/// gets that expert's FFN of its row in `out` (`[rows, top_k, hidden]`). `experts(e)` gives
/// expert `e`'s weights when this site computes it, and `None` otherwise; it is asked once per
/// routed expert before anything is written, and again when the expert's pass runs, and must
/// answer the same both times. `ws` holds [`workspace_bytes`] and is overwritten. With
/// `reads`, each expert's streamed weight bytes are added to its counter.
///
/// # Errors
///
/// [`MoeError::Len`] for an argument too short; [`MoeError::ExpertId`] for a route entry
/// outside `0..n_experts`; [`MoeError::Expert`] for an expert of another shape. A refused call
/// writes nothing to `out`.
#[allow(
    clippy::too_many_arguments,
    reason = "the op's operands, each a separate buffer or view (RFC-0040 §3.5)"
)]
pub fn moe_grouped<'w>(
    level: Level,
    shape: &MoeShape,
    experts: impl Fn(u32) -> Option<Expert<'w>>,
    x: &[f32],
    ids: &[i32],
    rows: usize,
    out: &mut [f32],
    ws: &mut [u8],
    reads: Option<&WeightReads>,
) -> Result<(), MoeError> {
    let MoeShape {
        n_experts,
        top_k,
        hidden,
        inter,
    } = *shape;
    let len = |arg, len, need| {
        if len < need {
            Err(MoeError::Len { arg, len, need })
        } else {
            Ok(())
        }
    };
    len("x", x.len(), rows * hidden)?;
    len("ids", ids.len(), rows * top_k)?;
    len("out", out.len(), rows * top_k * hidden)?;
    let s = Scratch::carve(ws, shape, rows)?;

    // Group the slots by expert: a counting sort, stable in (row, rank) order.
    s.offsets.fill(0);
    for (slot, &id) in ids[..rows * top_k].iter().enumerate() {
        let e = usize::try_from(id)
            .ok()
            .filter(|&e| e < n_experts)
            .ok_or(MoeError::ExpertId {
                row: slot / top_k,
                rank: slot % top_k,
                id,
            })?;
        s.offsets[e + 1] += 1;
    }
    for e in 0..n_experts {
        s.offsets[e + 1] += s.offsets[e];
    }
    // `offsets[e]` is used as expert e's cursor, which leaves it at expert e + 1's start; the
    // starts are restored by shifting down once every slot is placed.
    for (slot, &id) in ids[..rows * top_k].iter().enumerate() {
        let e = id as usize;
        s.slots[s.offsets[e] as usize] = slot as u32;
        s.offsets[e] += 1;
    }
    for e in (1..=n_experts).rev() {
        s.offsets[e] = s.offsets[e - 1];
    }
    s.offsets[0] = 0;

    // Check every routed expert of this site before writing anything.
    for e in 0..n_experts {
        if s.offsets[e] == s.offsets[e + 1] {
            continue;
        }
        if let Some(ex) = experts(e as u32)
            && !ex.fits(shape)
        {
            return Err(MoeError::Expert);
        }
    }

    for e in 0..n_experts {
        let group = &s.slots[s.offsets[e] as usize..s.offsets[e + 1] as usize];
        if group.is_empty() {
            continue;
        }
        let Some(ex) = experts(e as u32) else {
            continue;
        };
        // The group's distinct rows, gathered in order. A row's slots are adjacent in the
        // group, so a row that lists the expert twice is gathered once.
        let mut n = 0;
        for (_, r) in distinct_rows(group, top_k) {
            s.xg[n * hidden..(n + 1) * hidden].copy_from_slice(&x[r * hidden..(r + 1) * hidden]);
            n += 1;
        }
        let xq = quantize_into(
            ex.gate.ty().act(),
            &s.xg[..n * hidden],
            n,
            hidden,
            &mut s.xq[..],
        )?;
        let (g, u) = (&mut s.g[..n * inter], &mut s.u[..n * inter]);
        for task in expert_tasks(e as u32, &ex).filter(|t| t.phase == Phase::GateUp) {
            gate_up_block(level, &ex, &xq, task.out.clone(), n, g, u)?;
            if let Some(c) = reads {
                c.add(e, task.out.len() * 2 * ex.gate.row_bytes());
            }
        }
        let hq = quantize_into(ex.down.ty().act(), g, n, inter, &mut s.hq[..])?;
        let yd = &mut s.yd[..n * hidden];
        for task in expert_tasks(e as u32, &ex).filter(|t| t.phase == Phase::Down) {
            gemv(level, &ex.down, &hq, task.out.clone(), yd)?;
            let cols = task.out.clone();
            for (i, (slots, _)) in distinct_rows(group, top_k).enumerate() {
                let src = &yd[i * hidden + cols.start..i * hidden + cols.end];
                for &slot in slots {
                    let dst = slot as usize * hidden;
                    out[dst + cols.start..dst + cols.end].copy_from_slice(src);
                }
            }
            if let Some(c) = reads {
                c.add(e, task.out.len() * ex.down.row_bytes());
            }
        }
    }
    Ok(())
}

/// The runs of `group` (slot indices in ascending order) that share a row, with that row.
fn distinct_rows(group: &[u32], top_k: usize) -> impl Iterator<Item = (&[u32], usize)> {
    group
        .chunk_by(move |&a, &b| a as usize / top_k == b as usize / top_k)
        .map(move |run| (run, run[0] as usize / top_k))
}

/// One gate-and-up task: the gate and up products of the `n` gathered rows for the `inter`
/// rows in `out`, then `h = silu(gate) · up` over the same block, written over the gate.
fn gate_up_block(
    level: Level,
    ex: &Expert<'_>,
    xq: &crate::gemv::ActView<'_>,
    out: Range<usize>,
    n: usize,
    g: &mut [f32],
    u: &mut [f32],
) -> Result<(), GemvError> {
    let inter = ex.gate.n_out();
    gemv(level, &ex.gate, xq, out.clone(), g)?;
    gemv(level, &ex.up, xq, out.clone(), u)?;
    for i in 0..n {
        let (gr, ur) = (
            &mut g[i * inter + out.start..i * inter + out.end],
            &u[i * inter + out.start..i * inter + out.end],
        );
        for (gv, &uv) in gr.iter_mut().zip(ur) {
            *gv = silu(*gv) * uv;
        }
    }
    Ok(())
}
