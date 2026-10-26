// SPDX-License-Identifier: MIT OR Apache-2.0

//! A whole step descriptor: the [`StepDesc`] header and the arrays after it, sized at
//! `Warming` and written into a mirror's bytes (RFC-0040 §3.5, §3.8).

use std::fmt;
use std::mem::{offset_of, size_of};

use bitflags::bitflags;
use serde::{Deserialize, Serialize};

use super::mirror::{
    STEP_DESC_ALIGN, STEP_FLAG_PREFILL_LAST, STEP_FLAG_TREE, SamplerState, StepDesc,
};

bitflags! {
    /// [`StepDesc::flags`] (RFC-0040 §3.5). The bits are the header's `SYN_STEP_FLAG_*`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(transparent)]
    pub struct StepFlags: u32 {
        /// This step carries the last chunk of a prefill.
        const PREFILL_LAST = STEP_FLAG_PREFILL_LAST;
        /// The rows form a token tree, and [`RowEntry::tree_parent`] holds its edges.
        const TREE = STEP_FLAG_TREE;
    }
}

/// The most a descriptor holds, from which [`StepDescLayout`] sizes it at `Warming` (RFC-0040
/// §3.5: "sized at Warming from `BatchingLimits`"). The runner derives the row and sequence
/// counts from the stage's largest bucket.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StepDescLimits {
    /// Rows of the largest bucket: `R`.
    pub max_rows: u32,
    /// Sequences one step can hold: `S`.
    pub max_seqs: u32,
    /// MoE layers of the stage: `L_moe`, the exact length of `moe_balance`.
    pub moe_layers: u32,
    /// Rows one step can tap (RFC-0040 §3.8); 0 when the stage has no tap ops.
    pub max_tap_rows: u32,
}

/// Where each array of a descriptor sits, and the descriptor's size, for one
/// [`StepDescLimits`].
///
/// The arrays follow the header in the order RFC-0040 §3.5's struct lists them, then
/// `tap_rows` (RFC-0040 §3.8), each at the next multiple of its element's alignment. The size
/// is rounded up to `SYN_STEP_DESC_ALIGN`, 64 bytes. Offsets are `u32` in the header, so a
/// layout must fit in 4 GiB.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StepDescLayout {
    limits: StepDescLimits,
    /// The header with every offset set and every other field 0.
    offsets: StepDesc,
    size: u32,
}

/// One row of a step (RFC-0040 §3.5).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RowEntry {
    /// Index of the row's sequence in [`StepDescriptor::seqs`].
    pub seq: u32,
    /// The row's position in its sequence.
    pub pos: u32,
    /// The row's input token.
    pub token: u32,
    /// The row's parent row in a token tree, or -1.
    pub tree_parent: i32,
    /// The first position whose KV this step writes for this row: `pos` in normal steps,
    /// higher in a replay step (RFC-0033 §4.2). See [`kv_writes`].
    pub kv_write_from: u32,
}

/// One sequence of a step (RFC-0040 §3.5).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SeqEntry {
    /// The sequence's KV length before this step.
    pub kv_len: u32,
    /// Its offset into the KV store's page table (RFC-0022).
    pub page_table: u32,
    /// Its recurrent-state slot.
    pub state_slot: u32,
    /// Its random-number key and counter.
    pub sampler: SamplerState,
}

/// The parity-tap fields of the header other than the counts and offset (RFC-0040 §3.8).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TapFields {
    /// Reference ids per tapped row, K.
    pub k: u32,
    /// The doorbell slot the reference ids' arrival raises.
    pub flag: u32,
    /// Pinned buffer of reference token ids, `rows_tap × k`.
    pub ids: u64,
    /// The result block.
    pub out: u64,
    /// The `FullLocal` logits region, or 0.
    pub full: u64,
}

/// A step descriptor's contents: what the host writes into a mirror each step.
///
/// The counts in the header come from the vectors: `rows_active` is `rows.len()`,
/// `seqs_active` is `seqs.len()` and `rows_tap` is `tap_rows.len()`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct StepDescriptor {
    /// `ActivationFrame.step`.
    pub step: u64,
    /// The MoE table epoch the step reads.
    pub moe_epoch: u32,
    /// The step's flags.
    pub flags: StepFlags,
    /// The table-delta ring position `ApplyTableDeltas` applies up to (RFC-0040 §3.4).
    pub delta_head: u64,
    /// Device-visible address of the inbound frame payload.
    pub in_payload: u64,
    /// Device-visible address of the outbound frame's send buffer.
    pub out_payload: u64,
    /// The active rows.
    pub rows: Vec<RowEntry>,
    /// The active sequences.
    pub seqs: Vec<SeqEntry>,
    /// Per MoE layer of the stage, B_copy(l) in bytes; exactly `moe_layers` entries.
    pub moe_balance: Vec<u32>,
    /// The tapped rows' indices in [`rows`](Self::rows); empty in service steps.
    pub tap_rows: Vec<u32>,
    /// The other tap fields.
    pub tap: TapFields,
}

/// Why a descriptor could not be laid out, written or read.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StepDescError {
    /// The limits need a descriptor of 4 GiB or more.
    TooLarge,
    /// The mirror is smaller than the layout.
    MirrorTooSmall {
        /// The layout's size.
        need: u32,
        /// The mirror's length.
        have: usize,
    },
    /// A count exceeds its limit.
    OverLimit {
        /// Which count.
        what: &'static str,
        /// Its value.
        count: usize,
        /// Its limit.
        limit: u32,
    },
    /// `moe_balance` does not hold exactly one entry per MoE layer.
    MoeBalanceLength {
        /// Entries given.
        have: usize,
        /// MoE layers of the layout.
        need: u32,
    },
    /// An index names a row or sequence the step does not have.
    BadIndex {
        /// Which array.
        what: &'static str,
        /// The entry's index in its array.
        at: usize,
        /// The value.
        value: i64,
    },
    /// The mirror's offsets are not the layout's: it was written for other limits.
    LayoutMismatch,
    /// The flags set a bit the header does not define.
    UnknownFlags(u32),
}

impl fmt::Display for StepDescError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge => write!(f, "step descriptor limits need 4 GiB or more"),
            Self::MirrorTooSmall { need, have } => {
                write!(f, "step mirror holds {have} bytes, the layout needs {need}")
            }
            Self::OverLimit { what, count, limit } => {
                write!(f, "{what} is {count}, above its limit {limit}")
            }
            Self::MoeBalanceLength { have, need } => {
                write!(f, "moe_balance has {have} entries for {need} MoE layers")
            }
            Self::BadIndex { what, at, value } => {
                write!(f, "{what}[{at}] = {value} names no entry of this step")
            }
            Self::LayoutMismatch => write!(f, "step mirror offsets differ from the layout"),
            Self::UnknownFlags(bits) => write!(f, "step flags set undefined bits {bits:#x}"),
        }
    }
}

impl std::error::Error for StepDescError {}

/// The size of [`SamplerState`] in the descriptor's `seq_sampler` array.
const SAMPLER_SIZE: u64 = size_of::<SamplerState>() as u64;
/// The alignment of [`SamplerState`], the largest of any array element.
const SAMPLER_ALIGN: u64 = align_of::<SamplerState>() as u64;

impl StepDescLayout {
    /// Lays out a descriptor for `limits`.
    ///
    /// # Errors
    ///
    /// [`StepDescError::TooLarge`] when an offset or the size does not fit in a `u32`.
    pub fn new(limits: StepDescLimits) -> Result<Self, StepDescError> {
        let (r, s) = (u64::from(limits.max_rows), u64::from(limits.max_seqs));
        let mut at = size_of::<StepDesc>() as u64;
        // Each array starts at the next multiple of its element's alignment: the element's
        // size, which is 4 for every array but `seq_sampler`, whose alignment is 8.
        let mut place = |count: u64, elem: u64| -> Result<u32, StepDescError> {
            let start = at.next_multiple_of(elem.min(SAMPLER_ALIGN));
            at = start + count * elem;
            u32::try_from(start).map_err(|_| StepDescError::TooLarge)
        };
        let offsets = StepDesc {
            off_row_seq: place(r, 4)?,
            off_row_pos: place(r, 4)?,
            off_row_token: place(r, 4)?,
            off_tree_parent: place(r, 4)?,
            off_row_kv_write_from: place(r, 4)?,
            off_seq_kv_len: place(s, 4)?,
            off_seq_page_table: place(s, 4)?,
            off_seq_state_slot: place(s, 4)?,
            off_seq_sampler: place(s, SAMPLER_SIZE)?,
            off_moe_balance: place(u64::from(limits.moe_layers), 4)?,
            off_tap_rows: place(u64::from(limits.max_tap_rows), 4)?,
            ..StepDesc::default()
        };
        let size = at.next_multiple_of(u64::from(STEP_DESC_ALIGN));
        let size = u32::try_from(size).map_err(|_| StepDescError::TooLarge)?;
        Ok(Self {
            limits,
            offsets,
            size,
        })
    }

    /// The limits the layout was made for.
    #[must_use]
    pub fn limits(&self) -> StepDescLimits {
        self.limits
    }

    /// The descriptor's size in bytes: a multiple of 64.
    #[must_use]
    pub fn size(&self) -> u32 {
        self.size
    }

    /// A header with the layout's offsets and every other field 0.
    #[must_use]
    pub fn offsets(&self) -> StepDesc {
        self.offsets
    }

    fn same_offsets(&self, h: &StepDesc) -> bool {
        let o = &self.offsets;
        [
            (o.off_row_seq, h.off_row_seq),
            (o.off_row_pos, h.off_row_pos),
            (o.off_row_token, h.off_row_token),
            (o.off_tree_parent, h.off_tree_parent),
            (o.off_row_kv_write_from, h.off_row_kv_write_from),
            (o.off_seq_kv_len, h.off_seq_kv_len),
            (o.off_seq_page_table, h.off_seq_page_table),
            (o.off_seq_state_slot, h.off_seq_state_slot),
            (o.off_seq_sampler, h.off_seq_sampler),
            (o.off_moe_balance, h.off_moe_balance),
            (o.off_tap_rows, h.off_tap_rows),
        ]
        .iter()
        .all(|(a, b)| a == b)
    }

    fn check_mirror(&self, len: usize) -> Result<(), StepDescError> {
        if (len as u64) < u64::from(self.size) {
            return Err(StepDescError::MirrorTooSmall {
                need: self.size,
                have: len,
            });
        }
        Ok(())
    }
}

fn put32(buf: &mut [u8], at: usize, v: u32) {
    buf[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn put64(buf: &mut [u8], at: usize, v: u64) {
    buf[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

fn get32(buf: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(buf[at..at + 4].try_into().expect("4 bytes"))
}

fn get64(buf: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(buf[at..at + 8].try_into().expect("8 bytes"))
}

/// Element `i` of the array at byte offset `base` with `elem`-byte elements.
fn at(base: u32, i: usize, elem: usize) -> usize {
    base as usize + i * elem
}

macro_rules! header_fields {
    ($mac:ident) => {
        $mac!(step, u64);
        $mac!(rows_active, u32);
        $mac!(seqs_active, u32);
        $mac!(moe_epoch, u32);
        $mac!(flags, u32);
        $mac!(off_row_seq, u32);
        $mac!(off_row_pos, u32);
        $mac!(off_row_token, u32);
        $mac!(off_tree_parent, u32);
        $mac!(off_seq_kv_len, u32);
        $mac!(off_seq_page_table, u32);
        $mac!(off_seq_state_slot, u32);
        $mac!(off_seq_sampler, u32);
        $mac!(off_moe_balance, u32);
        $mac!(off_row_kv_write_from, u32);
        $mac!(delta_head, u64);
        $mac!(in_payload, u64);
        $mac!(out_payload, u64);
        $mac!(rows_tap, u32);
        $mac!(tap_k, u32);
        $mac!(off_tap_rows, u32);
        $mac!(tap_flag, u32);
        $mac!(tap_ids, u64);
        $mac!(tap_out, u64);
        $mac!(tap_full, u64);
    };
}

impl StepDesc {
    /// Writes the header's bytes, little-endian, at the offsets the mirror's layout gives each
    /// field. `buf` holds at least `size_of::<StepDesc>()` bytes.
    fn write_to(&self, buf: &mut [u8]) {
        macro_rules! put {
            ($f:ident, u32) => {
                put32(buf, offset_of!(StepDesc, $f), self.$f)
            };
            ($f:ident, u64) => {
                put64(buf, offset_of!(StepDesc, $f), self.$f)
            };
        }
        header_fields!(put);
    }

    /// Reads a header that [`write_to`](Self::write_to) wrote.
    fn read_from(buf: &[u8]) -> Self {
        let mut h = Self::default();
        macro_rules! get {
            ($f:ident, u32) => {
                h.$f = get32(buf, offset_of!(StepDesc, $f))
            };
            ($f:ident, u64) => {
                h.$f = get64(buf, offset_of!(StepDesc, $f))
            };
        }
        header_fields!(get);
        h
    }
}

macro_rules! sampler_fields {
    ($mac:ident) => {
        $mac!(seed);
        $mac!(stream_hi);
        $mac!(stream_lo);
        $mac!(rng_counter);
    };
}

/// The members `write_to` and `read_from` move, in order, for the test that checks they are
/// every member of the generated mirrors: a member added to the header and missed here would
/// otherwise never reach the device.
#[cfg(test)]
pub(super) fn serialized_fields() -> [Vec<&'static str>; 2] {
    let mut header = Vec::new();
    let mut sampler = Vec::new();
    macro_rules! name {
        ($f:ident, $t:ident) => {
            header.push(stringify!($f))
        };
        ($f:ident) => {
            sampler.push(stringify!($f))
        };
    }
    header_fields!(name);
    sampler_fields!(name);
    [header, sampler]
}

impl SamplerState {
    fn write_to(&self, buf: &mut [u8], base: usize) {
        macro_rules! put {
            ($f:ident) => {
                put64(buf, base + offset_of!(SamplerState, $f), self.$f)
            };
        }
        sampler_fields!(put);
    }

    fn read_from(buf: &[u8], base: usize) -> Self {
        let mut s = Self::default();
        macro_rules! get {
            ($f:ident) => {
                s.$f = get64(buf, base + offset_of!(SamplerState, $f))
            };
        }
        sampler_fields!(get);
        s
    }
}

/// Checks counts against limits and every index against the step it is in.
fn check_counts(
    limits: &StepDescLimits,
    rows: usize,
    seqs: usize,
    taps: usize,
    moe: usize,
) -> Result<(), StepDescError> {
    for (what, count, limit) in [
        ("rows_active", rows, limits.max_rows),
        ("seqs_active", seqs, limits.max_seqs),
        ("rows_tap", taps, limits.max_tap_rows),
    ] {
        if count as u64 > u64::from(limit) {
            return Err(StepDescError::OverLimit { what, count, limit });
        }
    }
    if moe as u64 != u64::from(limits.moe_layers) {
        return Err(StepDescError::MoeBalanceLength {
            have: moe,
            need: limits.moe_layers,
        });
    }
    Ok(())
}

impl StepDescriptor {
    /// Checks the descriptor against `layout`: counts within the limits, one balance entry per
    /// MoE layer, every row's sequence among `seqs`, every tree parent -1 or a row of the
    /// step, every tapped row a row of the step.
    ///
    /// # Errors
    ///
    /// The first [`StepDescError`] found.
    pub fn validate(&self, layout: &StepDescLayout) -> Result<(), StepDescError> {
        check_counts(
            &layout.limits,
            self.rows.len(),
            self.seqs.len(),
            self.tap_rows.len(),
            self.moe_balance.len(),
        )?;
        let (rows, seqs) = (self.rows.len() as i64, self.seqs.len() as i64);
        for (i, row) in self.rows.iter().enumerate() {
            if i64::from(row.seq) >= seqs {
                return Err(StepDescError::BadIndex {
                    what: "row_seq",
                    at: i,
                    value: row.seq.into(),
                });
            }
            if row.tree_parent < -1 || i64::from(row.tree_parent) >= rows {
                return Err(StepDescError::BadIndex {
                    what: "tree_parent",
                    at: i,
                    value: row.tree_parent.into(),
                });
            }
        }
        for (i, &r) in self.tap_rows.iter().enumerate() {
            if i64::from(r) >= rows {
                return Err(StepDescError::BadIndex {
                    what: "tap_rows",
                    at: i,
                    value: r.into(),
                });
            }
        }
        Ok(())
    }

    /// The header this descriptor writes under `layout`.
    #[must_use]
    pub fn header(&self, layout: &StepDescLayout) -> StepDesc {
        // Counts are below the u32 limits once `validate` passed.
        let count = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
        StepDesc {
            step: self.step,
            rows_active: count(self.rows.len()),
            seqs_active: count(self.seqs.len()),
            moe_epoch: self.moe_epoch,
            flags: self.flags.bits(),
            delta_head: self.delta_head,
            in_payload: self.in_payload,
            out_payload: self.out_payload,
            rows_tap: count(self.tap_rows.len()),
            tap_k: self.tap.k,
            tap_flag: self.tap.flag,
            tap_ids: self.tap.ids,
            tap_out: self.tap.out,
            tap_full: self.tap.full,
            ..layout.offsets
        }
    }

    /// Writes the descriptor into `mirror`, a pinned mirror of at least `layout.size()` bytes.
    /// Entries past the active counts are left as they were: kernels never read them.
    ///
    /// # Errors
    ///
    /// What [`validate`](Self::validate) finds, and [`StepDescError::MirrorTooSmall`]. The
    /// mirror is unchanged on error.
    pub fn write(&self, layout: &StepDescLayout, mirror: &mut [u8]) -> Result<(), StepDescError> {
        layout.check_mirror(mirror.len())?;
        self.validate(layout)?;
        let h = self.header(layout);
        h.write_to(mirror);
        for (i, row) in self.rows.iter().enumerate() {
            put32(mirror, at(h.off_row_seq, i, 4), row.seq);
            put32(mirror, at(h.off_row_pos, i, 4), row.pos);
            put32(mirror, at(h.off_row_token, i, 4), row.token);
            put32(mirror, at(h.off_tree_parent, i, 4), row.tree_parent as u32);
            put32(mirror, at(h.off_row_kv_write_from, i, 4), row.kv_write_from);
        }
        for (i, seq) in self.seqs.iter().enumerate() {
            put32(mirror, at(h.off_seq_kv_len, i, 4), seq.kv_len);
            put32(mirror, at(h.off_seq_page_table, i, 4), seq.page_table);
            put32(mirror, at(h.off_seq_state_slot, i, 4), seq.state_slot);
            let base = at(h.off_seq_sampler, i, SAMPLER_SIZE as usize);
            seq.sampler.write_to(mirror, base);
        }
        for (i, &b) in self.moe_balance.iter().enumerate() {
            put32(mirror, at(h.off_moe_balance, i, 4), b);
        }
        for (i, &r) in self.tap_rows.iter().enumerate() {
            put32(mirror, at(h.off_tap_rows, i, 4), r);
        }
        Ok(())
    }

    /// Reads a descriptor that [`write`](Self::write) wrote under the same `layout`.
    ///
    /// # Errors
    ///
    /// [`StepDescError::MirrorTooSmall`]; [`StepDescError::LayoutMismatch`] when the header's
    /// offsets are not `layout`'s; [`StepDescError::UnknownFlags`]; and what
    /// [`validate`](Self::validate) finds in the contents.
    pub fn read(layout: &StepDescLayout, mirror: &[u8]) -> Result<Self, StepDescError> {
        layout.check_mirror(mirror.len())?;
        let h = StepDesc::read_from(mirror);
        if !layout.same_offsets(&h) {
            return Err(StepDescError::LayoutMismatch);
        }
        let flags = StepFlags::from_bits(h.flags).ok_or(StepDescError::UnknownFlags(h.flags))?;
        // Counts are checked before any array is read, so a corrupt count never reads past
        // the layout.
        check_counts(
            &layout.limits,
            h.rows_active as usize,
            h.seqs_active as usize,
            h.rows_tap as usize,
            layout.limits.moe_layers as usize,
        )?;
        let rows = (0..h.rows_active as usize)
            .map(|i| RowEntry {
                seq: get32(mirror, at(h.off_row_seq, i, 4)),
                pos: get32(mirror, at(h.off_row_pos, i, 4)),
                token: get32(mirror, at(h.off_row_token, i, 4)),
                tree_parent: get32(mirror, at(h.off_tree_parent, i, 4)) as i32,
                kv_write_from: get32(mirror, at(h.off_row_kv_write_from, i, 4)),
            })
            .collect();
        let seqs = (0..h.seqs_active as usize)
            .map(|i| SeqEntry {
                kv_len: get32(mirror, at(h.off_seq_kv_len, i, 4)),
                page_table: get32(mirror, at(h.off_seq_page_table, i, 4)),
                state_slot: get32(mirror, at(h.off_seq_state_slot, i, 4)),
                sampler: SamplerState::read_from(
                    mirror,
                    at(h.off_seq_sampler, i, SAMPLER_SIZE as usize),
                ),
            })
            .collect();
        let moe_balance = (0..layout.limits.moe_layers as usize)
            .map(|i| get32(mirror, at(h.off_moe_balance, i, 4)))
            .collect();
        let tap_rows = (0..h.rows_tap as usize)
            .map(|i| get32(mirror, at(h.off_tap_rows, i, 4)))
            .collect();
        let d = Self {
            step: h.step,
            moe_epoch: h.moe_epoch,
            flags,
            delta_head: h.delta_head,
            in_payload: h.in_payload,
            out_payload: h.out_payload,
            rows,
            seqs,
            moe_balance,
            tap_rows,
            tap: TapFields {
                k: h.tap_k,
                flag: h.tap_flag,
                ids: h.tap_ids,
                out: h.tap_out,
                full: h.tap_full,
            },
        };
        d.validate(layout)?;
        Ok(d)
    }
}

/// The KV positions a step writes, as `(row, seq, pos)`, in row order (RFC-0040 §3.5).
///
/// Row `r` computes the token at `rows[r].pos`, and its attention and cache-append kernels
/// write that position's KV only when it is at or above `rows[r].kv_write_from`. A replay step
/// (RFC-0033 §4.2) recomputes positions below it without overwriting KV that survived. In
/// normal steps the two are equal, so every row writes. `kernels::reference` and every
/// provider's KV-writing kernel apply this rule.
pub fn kv_writes(d: &StepDescriptor) -> impl Iterator<Item = (usize, u32, u32)> + '_ {
    d.rows
        .iter()
        .enumerate()
        .filter(|(_, row)| row.pos >= row.kv_write_from)
        .map(|(r, row)| (r, row.seq, row.pos))
}
