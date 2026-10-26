// SPDX-License-Identifier: MIT OR Apache-2.0

//! The host side of one device's table-delta ring: where its parts sit in the bytes a plan
//! maps ([`DeltaRingLayout`]), how the dispatcher posts entries into them, and how the host
//! reads entries back and takes the device's echo (RFC-0023 §5.4; RFC-0040 §3.4).
//!
//! A ring is the [`DeltaRingHeader`] followed by `capacity` [`TableDelta`] entries with no
//! padding between them: the header's size, 32, is a multiple of an entry's alignment, 4.
//! Positions are monotonic `u64`s, and the entry at position `p` sits in slot
//! `p & (capacity - 1)`, so posting wraps. All values are little-endian. The dispatcher is the
//! ring's only producer: `head` is the position after the last entry it posted, `applied` the
//! position after the last entry a step applied, and `applied_step` the step number that
//! produced `applied`. A step applies up to the `delta_head` its descriptor names, never the
//! live head, so `head - applied` is how many deltas are still waiting.

use std::fmt;
use std::mem::{offset_of, size_of};

use super::mirror::{DeltaRingHeader, TABLE_DELTA_RING_MIN_CAPACITY, TableDelta};

/// Why a ring could not be laid out, written or read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RingError {
    /// The buffer is smaller than the layout.
    BufferTooSmall {
        /// The layout's size.
        need: usize,
        /// The buffer's length.
        have: usize,
    },
    /// A capacity that is not a power of two.
    BadCapacity(u32),
    /// More deltas pending than the ring holds: `EXPERT_DELTA_RING_FULL`, class `Internal`
    /// (RFC-0023 §5.4).
    Full {
        /// The ring's head at the attempt.
        head: u64,
        /// The ring's applied position at the attempt.
        applied: u64,
        /// The ring's capacity in entries.
        capacity: u32,
        /// How many entries the attempt would have posted.
        posting: usize,
    },
    /// A range [`DeltaRingLayout::read`] was asked for that the ring does not hold.
    BadRange {
        /// The range's first position.
        from: u64,
        /// The range's end, exclusive.
        to: u64,
        /// The ring's head.
        head: u64,
        /// The ring's capacity in entries.
        capacity: u32,
    },
}

impl fmt::Display for RingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::BufferTooSmall { need, have } => {
                write!(f, "ring buffer holds {have} bytes; the layout needs {need}")
            }
            Self::BadCapacity(c) => write!(f, "ring capacity {c} is not a power of two"),
            Self::Full {
                head,
                applied,
                capacity,
                posting,
            } => write!(
                f,
                "EXPERT_DELTA_RING_FULL: {posting} deltas would put head {head} and applied \
                 {applied} past the capacity {capacity} of the ring"
            ),
            Self::BadRange {
                from,
                to,
                head,
                capacity,
            } => write!(
                f,
                "ring positions {from}..{to} are outside head {head} and capacity {capacity}"
            ),
        }
    }
}

impl std::error::Error for RingError {}

/// Where a ring's parts sit, and its size, for one capacity (RFC-0023 §5.4).
///
/// `capacity` is a power of two. A plan takes it from the ring's bound for the epoch, never
/// below [`TABLE_DELTA_RING_MIN_CAPACITY`]. It is
/// the capacity [`post`](Self::post) and [`read`](Self::read) index entries by; [`init`](Self::init)
/// writes the same value into the header for the device to read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DeltaRingLayout {
    capacity: u32,
}

impl DeltaRingLayout {
    /// The layout for a ring of `capacity` entries.
    ///
    /// # Errors
    ///
    /// [`RingError::BadCapacity`] when `capacity` is not a power of two.
    pub fn new(capacity: u32) -> Result<Self, RingError> {
        if capacity == 0 || capacity & (capacity - 1) != 0 {
            return Err(RingError::BadCapacity(capacity));
        }
        Ok(Self { capacity })
    }

    /// `capacity`.
    #[must_use]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Byte offset of the entries: the header's size. That size, 32, is a multiple of an
    /// entry's alignment, 4, so the entries start immediately after the header.
    #[must_use]
    pub fn entries_offset(&self) -> usize {
        size_of::<DeltaRingHeader>()
    }

    /// The ring's size in bytes: the header plus `capacity` entries.
    #[must_use]
    pub fn size(&self) -> usize {
        self.entries_offset() + (self.capacity as usize) * size_of::<TableDelta>()
    }

    fn check(&self, len: usize) -> Result<(), RingError> {
        if len < self.size() {
            return Err(RingError::BufferTooSmall {
                need: self.size(),
                have: len,
            });
        }
        Ok(())
    }

    /// The byte offset of the entry at position `p`: slot `p & (capacity - 1)`, which wraps.
    fn entry_at(&self, p: u64) -> usize {
        let slot = (p & (u64::from(self.capacity) - 1)) as usize;
        self.entries_offset() + slot * size_of::<TableDelta>()
    }

    /// Writes a fresh ring header into `buf`: `head`, `applied` and `applied_step` 0, and this
    /// layout's `capacity`. Entries are left as they were; nothing is posted yet.
    ///
    /// # Errors
    ///
    /// [`RingError::BufferTooSmall`], and `buf` is unchanged then.
    pub fn init(&self, buf: &mut [u8]) -> Result<(), RingError> {
        self.check(buf.len())?;
        DeltaRingHeader {
            head: 0,
            applied: 0,
            applied_step: 0,
            capacity: self.capacity,
            _pad: 0,
        }
        .write_to(buf);
        Ok(())
    }

    /// Posts `deltas`, in order, and returns the ring's new head. Entry `i` of the batch goes to
    /// position `head + i`; the head moves last, so a reader that takes the head alone sees
    /// either the whole batch or none of it.
    ///
    /// # Errors
    ///
    /// [`RingError::BufferTooSmall`], or [`RingError::Full`] (the code name
    /// `EXPERT_DELTA_RING_FULL`) when `head + deltas.len() - applied` exceeds `capacity`: the
    /// dispatcher posts nothing until the echo shows the device applied more. `buf` is
    /// unchanged on error.
    pub fn post(&self, buf: &mut [u8], deltas: &[TableDelta]) -> Result<u64, RingError> {
        self.check(buf.len())?;
        let h = DeltaRingHeader::read_from(buf);
        let capacity = u64::from(self.capacity);
        let full = RingError::Full {
            head: h.head,
            applied: h.applied,
            capacity: self.capacity,
            posting: deltas.len(),
        };
        let posted = match u64::try_from(deltas.len()) {
            Ok(n) => n,
            Err(_) => return Err(full),
        };
        // Deltas still waiting: the head is ahead of the applied position by at most
        // `capacity`, and the batch must fit in the room left in it.
        let pending = match h.head.checked_sub(h.applied) {
            Some(p) => p,
            None => return Err(full),
        };
        if pending
            .checked_add(posted)
            .is_none_or(|total| total > capacity)
        {
            return Err(full);
        }
        let new_head = match h.head.checked_add(posted) {
            Some(head) => head,
            None => return Err(full),
        };
        for (i, delta) in deltas.iter().enumerate() {
            delta.write_to(buf, self.entry_at(h.head + (i as u64)));
        }
        put64(buf, offset_of!(DeltaRingHeader, head), new_head);
        Ok(new_head)
    }

    /// The entries at positions `from .. to`, in the order they were posted. A step reads the
    /// range up to the `delta_head` its descriptor names.
    ///
    /// # Errors
    ///
    /// [`RingError::BufferTooSmall`], or [`RingError::BadRange`] when `from` is past `to`, when
    /// `to` is past the head, when the range is longer than the capacity, or when `from` is
    /// older than the `capacity` entries the ring still holds: the producer has already
    /// overwritten those positions.
    pub fn read(&self, buf: &[u8], from: u64, to: u64) -> Result<Vec<TableDelta>, RingError> {
        self.check(buf.len())?;
        let head = DeltaRingHeader::read_from(buf).head;
        // Before the first wrap every posted position is still in the ring; after it the
        // producer has overwritten everything older than the last `capacity` positions.
        let overwritten = head
            .checked_sub(u64::from(self.capacity))
            .is_some_and(|oldest| from < oldest);
        if from > to || to > head || to - from > u64::from(self.capacity) || overwritten {
            return Err(RingError::BadRange {
                from,
                to,
                head,
                capacity: self.capacity,
            });
        }
        Ok((from..to)
            .map(|p| TableDelta::read_from(buf, self.entry_at(p)))
            .collect())
    }

    /// Writes the device's echo: the position it applied up to, and the number of the step whose
    /// `delta_head` produced it (RFC-0023 §5.4; RFC-0040 §3.4).
    ///
    /// # Errors
    ///
    /// [`RingError::BufferTooSmall`], and `buf` is unchanged then.
    pub fn echo(&self, buf: &mut [u8], applied: u64, step: u64) -> Result<(), RingError> {
        self.check(buf.len())?;
        put64(buf, offset_of!(DeltaRingHeader, applied), applied);
        put64(buf, offset_of!(DeltaRingHeader, applied_step), step);
        Ok(())
    }
}

/// The capacity a plan gives a device's ring for an epoch (RFC-0023 §5.4 "Capacity and the KV
/// block tables"): the ring carries only incremental changes of running sequences and
/// expert-cache replacements, so its capacity is fixed per epoch at the next power of two ≥
/// `2 · (max_live_sequences · page_classes · (⌈T_max / P⌉ + 1) + max_replace · caches)`,
/// minimum 4,096 ([`TABLE_DELTA_RING_MIN_CAPACITY`]).
///
/// Block tables of a sequence that has not yet run (a new sequence, an attached prefix, a
/// restored session) are written with a direct copy on the copy stream before the sequence's
/// first step, ordered by that step's fence, not through the ring.
///
/// With that sizing an overflow is a bug, not a policy outcome: posting then reports
/// [`RingError::Full`], the code name `EXPERT_DELTA_RING_FULL`, class `Internal`.
///
/// # Errors
///
/// `None` when `page_size` is 0, or when the bound or its next power of two does not fit a `u32`.
/// All arithmetic is done in `u64`.
#[must_use]
pub fn epoch_capacity(
    max_live_sequences: u32,
    page_classes: u32,
    t_max: u32,
    page_size: u32,
    max_replace: u32,
    caches: u32,
) -> Option<u32> {
    if page_size == 0 {
        return None;
    }
    // ceil(t_max / page_size) as a `u64`; `page_size` is not 0 here.
    let pages = (u64::from(t_max) + u64::from(page_size) - 1) / u64::from(page_size);
    let per_sequence = u64::from(page_classes).checked_mul(pages + 1)?;
    let kv = u64::from(max_live_sequences).checked_mul(per_sequence)?;
    let replacements = u64::from(max_replace).checked_mul(u64::from(caches))?;
    let doubled = kv.checked_add(replacements)?.checked_mul(2)?;
    // `want` is at most `u32::MAX`, so its next power of two is at most 2^32 and still fits a
    // `u64`; when that is 2^32 the ring has no `u32` capacity, so the bound is too big.
    u32::try_from(
        doubled
            .max(u64::from(TABLE_DELTA_RING_MIN_CAPACITY))
            .next_power_of_two(),
    )
    .ok()
}

/// The ring's `head`: the position after the last delta posted. A step never reads it; the host
/// uses it to bound what it reads back.
///
/// # Errors
///
/// [`RingError::BufferTooSmall`] when `buf` is shorter than the header.
pub fn head(buf: &[u8]) -> Result<u64, RingError> {
    if buf.len() < size_of::<DeltaRingHeader>() {
        return Err(RingError::BufferTooSmall {
            need: size_of::<DeltaRingHeader>(),
            have: buf.len(),
        });
    }
    Ok(DeltaRingHeader::read_from(buf).head)
}

/// The ring's echo, `(applied, applied_step)`.
///
/// # Errors
///
/// [`RingError::BufferTooSmall`] when `buf` is shorter than the header.
pub fn applied(buf: &[u8]) -> Result<(u64, u64), RingError> {
    if buf.len() < size_of::<DeltaRingHeader>() {
        return Err(RingError::BufferTooSmall {
            need: size_of::<DeltaRingHeader>(),
            have: buf.len(),
        });
    }
    let h = DeltaRingHeader::read_from(buf);
    Ok((h.applied, h.applied_step))
}

fn put16(buf: &mut [u8], at: usize, v: u16) {
    buf[at..at + 2].copy_from_slice(&v.to_le_bytes());
}

fn put32(buf: &mut [u8], at: usize, v: u32) {
    buf[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn put64(buf: &mut [u8], at: usize, v: u64) {
    buf[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

fn get16(buf: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(buf[at..at + 2].try_into().expect("2 bytes"))
}

fn get32(buf: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(buf[at..at + 4].try_into().expect("4 bytes"))
}

fn get64(buf: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(buf[at..at + 8].try_into().expect("8 bytes"))
}

macro_rules! header_fields {
    ($mac:ident) => {
        $mac!(DeltaRingHeader, head, 64);
        $mac!(DeltaRingHeader, applied, 64);
        $mac!(DeltaRingHeader, applied_step, 64);
        $mac!(DeltaRingHeader, capacity, 32);
        $mac!(DeltaRingHeader, _pad, 32);
    };
}

macro_rules! delta_fields {
    ($mac:ident) => {
        $mac!(TableDelta, table, 16);
        $mac!(TableDelta, _pad, 16);
        $mac!(TableDelta, index, 32);
        $mac!(TableDelta, value, 32);
    };
}

impl DeltaRingHeader {
    /// Writes the members at the start of `buf`, little-endian, at the offsets the mirror gives
    /// them. `buf` holds at least `size_of::<DeltaRingHeader>()` bytes.
    fn write_to(&self, buf: &mut [u8]) {
        macro_rules! put {
            ($s:ident, $f:ident, 64) => {
                put64(buf, offset_of!($s, $f), self.$f)
            };
            ($s:ident, $f:ident, 32) => {
                put32(buf, offset_of!($s, $f), self.$f)
            };
        }
        header_fields!(put);
    }

    /// Reads what [`write_to`](Self::write_to) wrote.
    fn read_from(buf: &[u8]) -> Self {
        let mut h = Self::default();
        macro_rules! get {
            ($s:ident, $f:ident, 64) => {
                h.$f = get64(buf, offset_of!($s, $f))
            };
            ($s:ident, $f:ident, 32) => {
                h.$f = get32(buf, offset_of!($s, $f))
            };
        }
        header_fields!(get);
        h
    }
}

impl TableDelta {
    /// Writes the members at `base`, little-endian, at the offsets the mirror gives them.
    fn write_to(&self, buf: &mut [u8], base: usize) {
        macro_rules! put {
            ($s:ident, $f:ident, 16) => {
                put16(buf, base + offset_of!($s, $f), self.$f)
            };
            ($s:ident, $f:ident, 32) => {
                put32(buf, base + offset_of!($s, $f), self.$f)
            };
        }
        delta_fields!(put);
    }

    /// Reads what [`write_to`](Self::write_to) wrote.
    fn read_from(buf: &[u8], base: usize) -> Self {
        let mut d = Self::default();
        macro_rules! get {
            ($s:ident, $f:ident, 16) => {
                d.$f = get16(buf, base + offset_of!($s, $f))
            };
            ($s:ident, $f:ident, 32) => {
                d.$f = get32(buf, base + offset_of!($s, $f))
            };
        }
        delta_fields!(get);
        d
    }
}
