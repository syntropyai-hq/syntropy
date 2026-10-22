// SPDX-License-Identifier: MIT OR Apache-2.0

//! A ring of fixed-size byte slots in memory the caller provides: the form of RFC-0001 §3.5's
//! rings, which live in a region shared between the daemon and a runner process.
//!
//! The index protocol is the heap ring's (see [`ring`](crate::ring)). The producer owns `tail` and
//! the consumer owns `head`; both count up forever, and each end publishes its index with a release
//! store and reads the other's with an acquire load. What differs is the storage:
//! - **The memory** is [`SharedLayout::bytes`] bytes at an address aligned to
//!   [`SharedLayout::ALIGN`]: a header holding the two indices and the close flags, each on its own
//!   cache line, then the slots. A slot is a little-endian `u32` length, padded to 8 bytes, then
//!   up to `slot_bytes` bytes of record.
//! - **A record is bytes.** The record types and their encodings belong to the rings' owners
//!   (RFC-0053 for telemetry, steps and logs).
//! - **Neither end trusts the other.** The peer may be another process, buggy or hostile. Each
//!   index and each length is read once, atomically, into a local copy, and only the copy is
//!   checked and used, so the peer cannot change a value between its check and its use. A reader
//!   that finds the indices more than `slots` apart, or a length above `slot_bytes`, returns
//!   [`SharedPopError::Corrupt`] or [`SharedPushError::Corrupt`] and reads nothing out of range.
//!   A record's bytes are copied out before the caller reads them, and every access to a slot,
//!   its length word and its bytes alike, is atomic (8-byte words, then single bytes), so a peer
//!   writing at the same time causes no data race in this process (RFC-0060 §10). Copying through
//!   relaxed atomics compiles to plain loads and stores on x86-64 and AArch64, so it costs about
//!   what a `memcpy` of the same words would, without its vector instructions.
//! - **No waking.** Both ends poll. A consumer that needs waking across processes needs a handle
//!   passed to it, which the process that spawns the runner provides.
//!
//! [`SharedLayout::init`] writes an empty ring's header. Each process then takes its end with
//! [`SharedLayout::producer`] or [`SharedLayout::consumer`], which pick up from the indices in the
//! header, and clear their own side's close flag. A process that reattaches, such as a daemon
//! readopting a runner (RFC-0052 §6.5), therefore continues where the last end of its side
//! stopped.

use std::fmt;
use std::marker::PhantomData;
use std::ptr::NonNull;

use crate::pad::CachePadded;
use crate::sync::{AtomicU32, AtomicU64, Ordering};

/// The bit the producer sets in `closed` when it goes.
const PRODUCER_CLOSED: u32 = 1;
/// The bit the consumer sets in `closed` when it goes.
const CONSUMER_CLOSED: u32 = 2;
/// The length word ahead of each record: a `u32`, padded so the record starts 8-byte aligned.
const LENGTH_BYTES: usize = 8;

/// The ring's header, at the start of its memory.
#[repr(C)]
struct Header {
    /// The consumer's index: the next slot to read.
    head: CachePadded<AtomicU64>,
    /// The producer's index: the next slot to write.
    tail: CachePadded<AtomicU64>,
    /// [`PRODUCER_CLOSED`] and [`CONSUMER_CLOSED`].
    closed: CachePadded<AtomicU32>,
    /// Under loom only: one cell per slot, which `push` marks as written and `pop` as read, so
    /// loom checks that the index protocol orders a slot's write before its read, and its read
    /// before the next write. The slots' own bytes are plain memory, which loom cannot see.
    #[cfg(loom)]
    shadow: Box<[crate::sync::UnsafeCell<()>]>,
    /// Under loom only: whether `push` publishes with a relaxed store, the ordering bug the loom
    /// models must find (QUAL-E3-T09). Only the tests' planted initialiser sets it.
    #[cfg(loom)]
    planted_relaxed_publish: bool,
}

impl Header {
    fn new(_slots: u32) -> Header {
        Header {
            head: CachePadded(AtomicU64::new(0)),
            tail: CachePadded(AtomicU64::new(0)),
            closed: CachePadded(AtomicU32::new(0)),
            #[cfg(loom)]
            shadow: (0.._slots)
                .map(|_| crate::sync::UnsafeCell::new(()))
                .collect(),
            #[cfg(loom)]
            planted_relaxed_publish: false,
        }
    }
}

/// Copies `from` into shared memory at `to`: 8-byte words with relaxed atomic stores, then the
/// last bytes with atomic byte stores. Every access to slot memory is atomic, on both ends, so a
/// peer that writes the same bytes at the same time makes no data race, only garbled bytes.
///
/// # Safety
///
/// `to` is 8-aligned and valid for `from.len()` bytes, inside a live ring.
unsafe fn copy_in(from: &[u8], to: NonNull<u8>) {
    use std::sync::atomic::{AtomicU8, AtomicU64 as SlotWord, Ordering::Relaxed};
    let (words, rest) = from.as_chunks::<8>();
    for (index, word) in words.iter().enumerate() {
        // SAFETY: `index * 8` is a multiple of 8 below `from.len()`, so the word is 8-aligned and
        // inside the region the caller promised.
        let slot_word = unsafe { SlotWord::from_ptr(to.add(index * 8).cast::<u64>().as_ptr()) };
        slot_word.store(u64::from_ne_bytes(*word), Relaxed);
    }
    let start = words.len() * 8;
    for (offset, byte) in rest.iter().enumerate() {
        // SAFETY: `start + offset` is below `from.len()`, inside the region the caller promised.
        unsafe { AtomicU8::from_ptr(to.add(start + offset).as_ptr()) }.store(*byte, Relaxed);
    }
}

/// Copies shared memory at `from` into `to`, as [`copy_in`] writes it: 8-byte words, then bytes,
/// all with relaxed atomic loads.
///
/// # Safety
///
/// `from` is 8-aligned and valid for `to.len()` bytes, inside a live ring.
unsafe fn copy_out(from: NonNull<u8>, to: &mut [u8]) {
    use std::sync::atomic::{AtomicU8, AtomicU64 as SlotWord, Ordering::Relaxed};
    let (words, rest) = to.as_chunks_mut::<8>();
    let start = words.len() * 8;
    for (index, word) in words.iter_mut().enumerate() {
        // SAFETY: as in `copy_in`.
        let slot_word = unsafe { SlotWord::from_ptr(from.add(index * 8).cast::<u64>().as_ptr()) };
        *word = slot_word.load(Relaxed).to_ne_bytes();
    }
    for (offset, byte) in rest.iter_mut().enumerate() {
        // SAFETY: as in `copy_in`.
        *byte = unsafe { AtomicU8::from_ptr(from.add(start + offset).as_ptr()) }.load(Relaxed);
    }
}

/// The bytes the header takes, rounded up so the slots start [`SharedLayout::ALIGN`]-aligned.
fn header_bytes() -> usize {
    std::mem::size_of::<Header>().next_multiple_of(SharedLayout::ALIGN)
}

/// Why a [`SharedLayout`] was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutError {
    /// The slot count is not a power of two from 1 to [`SharedLayout::MAX_SLOTS`].
    Slots(u32),
    /// The slot size is not from 1 to [`SharedLayout::MAX_SLOT_BYTES`].
    SlotBytes(u32),
    /// The ring would take more than `isize::MAX` bytes, which only a 32-bit target can reach.
    TooLarge,
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayoutError::Slots(slots) => write!(
                f,
                "a shared ring's slot count is a power of two from 1 to {}, not {slots}",
                SharedLayout::MAX_SLOTS
            ),
            LayoutError::SlotBytes(bytes) => write!(
                f,
                "a shared ring's slot holds 1 to {} bytes, not {bytes}",
                SharedLayout::MAX_SLOT_BYTES
            ),
            LayoutError::TooLarge => f.write_str("the shared ring would not fit the address space"),
        }
    }
}

impl std::error::Error for LayoutError {}

/// A shared ring's shape: how many slots, and how many bytes a record may take.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SharedLayout {
    slots: u32,
    slot_bytes: u32,
    /// [`Self::bytes`], computed once with checked arithmetic.
    bytes: usize,
}

impl SharedLayout {
    /// The alignment the ring's memory needs: a cache line pair, as for the heap ring.
    pub const ALIGN: usize = 128;
    /// The most slots a ring may have: 2^24.
    pub const MAX_SLOTS: u32 = 1 << 24;
    /// The largest record a slot may hold: 1 MiB.
    pub const MAX_SLOT_BYTES: u32 = 1 << 20;

    /// A layout of `slots` slots of up to `slot_bytes` bytes each.
    ///
    /// # Errors
    ///
    /// [`LayoutError::Slots`] when `slots` is not a power of two from 1 to [`Self::MAX_SLOTS`],
    /// [`LayoutError::SlotBytes`] when `slot_bytes` is 0 or above [`Self::MAX_SLOT_BYTES`], and
    /// [`LayoutError::TooLarge`] when the ring would take more than `isize::MAX` bytes. At most
    /// 2^24 slots of 1 MiB fit a 64-bit address space, so only a 32-bit target refuses a ring
    /// that way.
    pub fn new(slots: u32, slot_bytes: u32) -> Result<SharedLayout, LayoutError> {
        if !slots.is_power_of_two() || slots > Self::MAX_SLOTS {
            return Err(LayoutError::Slots(slots));
        }
        if slot_bytes == 0 || slot_bytes > Self::MAX_SLOT_BYTES {
            return Err(LayoutError::SlotBytes(slot_bytes));
        }
        let stride = LENGTH_BYTES + (slot_bytes as usize).next_multiple_of(8);
        let bytes = (slots as usize)
            .checked_mul(stride)
            .and_then(|slots_bytes| slots_bytes.checked_add(header_bytes()))
            .and_then(|total| total.checked_next_multiple_of(Self::ALIGN))
            .filter(|total| *total <= isize::MAX as usize)
            .ok_or(LayoutError::TooLarge)?;
        Ok(SharedLayout {
            slots,
            slot_bytes,
            bytes,
        })
    }

    /// The number of slots.
    #[must_use]
    pub fn slots(&self) -> u32 {
        self.slots
    }

    /// The largest record a slot holds, in bytes.
    #[must_use]
    pub fn slot_bytes(&self) -> u32 {
        self.slot_bytes
    }

    /// The bytes one slot takes: its length word and its record, rounded up to 8.
    fn stride(&self) -> usize {
        LENGTH_BYTES + (self.slot_bytes as usize).next_multiple_of(8)
    }

    /// The bytes the ring takes: its header, then its slots. A multiple of [`Self::ALIGN`], so
    /// rings laid end to end stay aligned, and never more than `isize::MAX`.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Writes an empty ring's header at `memory`.
    ///
    /// # Safety
    ///
    /// `memory` is valid for writes of [`Self::bytes`] bytes and aligned to [`Self::ALIGN`], and
    /// no end of a ring at `memory` exists, in this process or any other, while this runs.
    pub unsafe fn init(&self, memory: NonNull<u8>) {
        // SAFETY: the caller promises `memory` is writable for the whole ring, so for the header
        // at its start, and aligned to 128 bytes, which covers `Header`'s alignment. No end reads
        // the header while it is written.
        unsafe { memory.cast::<Header>().write(Header::new(self.slots)) };
    }

    /// Under loom, for the models only: [`Self::init`], with the bug the models must find
    /// planted: `push` publishes with a relaxed store (QUAL-E3-T09).
    ///
    /// # Safety
    ///
    /// As for [`Self::init`].
    #[cfg(all(test, loom))]
    pub(crate) unsafe fn init_planted_relaxed_publish(&self, memory: NonNull<u8>) {
        let mut header = Header::new(self.slots);
        header.planted_relaxed_publish = true;
        // SAFETY: as in `init`.
        unsafe { memory.cast::<Header>().write(header) };
    }

    /// The producing end of the ring at `memory`, continuing from the indices in its header. It
    /// clears the producer's close flag, so a producer can reattach after an earlier one went.
    ///
    /// # Safety
    ///
    /// `memory` holds a ring that [`Self::init`] wrote with this layout, in this process or
    /// another; it stays mapped and valid for reads and writes for `'a`; and no other producer of
    /// this ring exists, in any process, while this one does.
    #[must_use]
    pub unsafe fn producer<'a>(&self, memory: NonNull<u8>) -> SharedProducer<'a> {
        // SAFETY: the caller promises the header was initialised and stays valid for `'a`. It is
        // all atomics, so a shared reference to it allows the other end's writes.
        let header = unsafe { memory.cast::<Header>().as_ref() };
        header.closed.fetch_and(!PRODUCER_CLOSED, Ordering::AcqRel);
        SharedProducer {
            end: End::new(header, memory, *self),
            tail: header.tail.load(Ordering::Acquire),
            head: header.head.load(Ordering::Acquire),
        }
    }

    /// The consuming end of the ring at `memory`, continuing from the indices in its header. It
    /// clears the consumer's close flag, so a consumer can reattach after an earlier one went.
    ///
    /// # Safety
    ///
    /// As for [`Self::producer`], with no other consumer in place of no other producer.
    #[must_use]
    pub unsafe fn consumer<'a>(&self, memory: NonNull<u8>) -> SharedConsumer<'a> {
        // SAFETY: as in `producer`.
        let header = unsafe { memory.cast::<Header>().as_ref() };
        header.closed.fetch_and(!CONSUMER_CLOSED, Ordering::AcqRel);
        SharedConsumer {
            end: End::new(header, memory, *self),
            head: header.head.load(Ordering::Acquire),
            tail: header.tail.load(Ordering::Acquire),
        }
    }
}

/// The length word at the start of `slot`, as an atomic. Every access to it is atomic, on both
/// ends, so the reader's single load cannot be split or repeated.
///
/// # Safety
///
/// `slot` is a slot of a live ring: 8-aligned, with its length word inside the ring's memory, which
/// stays mapped for `'b`.
unsafe fn length_word<'b>(slot: NonNull<u8>) -> &'b std::sync::atomic::AtomicU32 {
    // SAFETY: the caller promises the word is aligned, in bounds and live for `'b`, and every
    // access to it goes through this atomic.
    unsafe { std::sync::atomic::AtomicU32::from_ptr(slot.cast::<u32>().as_ptr()) }
}

/// What both ends hold: the header, the first slot and the layout.
struct End<'a> {
    header: &'a Header,
    /// The first slot, [`header_bytes`] past the start of the ring's memory.
    slots: NonNull<u8>,
    layout: SharedLayout,
    _memory: PhantomData<&'a [u8]>,
}

impl<'a> End<'a> {
    fn new(header: &'a Header, memory: NonNull<u8>, layout: SharedLayout) -> End<'a> {
        // SAFETY: the ring's memory is `layout.bytes()` long, which is more than `header_bytes()`.
        let slots = unsafe { memory.add(header_bytes()) };
        End {
            header,
            slots,
            layout,
            _memory: PhantomData,
        }
    }

    /// The position, from 0, of the slot that index `at` names.
    fn position(&self, at: u64) -> usize {
        (at & u64::from(self.layout.slots - 1)) as usize
    }

    /// The ordering of the store that publishes a push: `Release`, or `Relaxed` in a ring with the
    /// planted bug, under loom.
    fn publish_ordering(&self) -> Ordering {
        #[cfg(loom)]
        if self.header.planted_relaxed_publish {
            return Ordering::Relaxed;
        }
        Ordering::Release
    }

    /// Under loom, records a write to the slot `at`; elsewhere, nothing.
    fn shadow_write(&self, _at: u64) {
        #[cfg(loom)]
        self.header.shadow[self.position(_at)].with_mut(|_| ());
    }

    /// Under loom, records a read of the slot `at`; elsewhere, nothing.
    fn shadow_read(&self, _at: u64) {
        #[cfg(loom)]
        self.header.shadow[self.position(_at)].with(|_| ());
    }

    /// The slot that index `at` names.
    fn slot(&self, at: u64) -> NonNull<u8> {
        let position = self.position(at);
        // SAFETY: `position` is below `slots`, so the slot's `stride` bytes lie inside the ring's
        // memory, which `layout.bytes()` sized for `slots` strides after the header.
        unsafe { self.slots.add(position * self.layout.stride()) }
    }

    fn closed(&self, bit: u32) -> bool {
        self.header.closed.load(Ordering::Acquire) & bit != 0
    }
}

/// Why a push to a shared ring failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SharedPushError {
    /// Every slot holds a record the consumer has not read yet.
    Full,
    /// The consumer is gone.
    Closed,
    /// The record is longer than a slot.
    TooLong {
        /// The record's length.
        len: usize,
        /// The longest record a slot holds.
        max: usize,
    },
    /// The consumer's index is out of range: the peer broke the protocol.
    Corrupt,
}

impl fmt::Display for SharedPushError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SharedPushError::Full => f.write_str("the shared ring is full"),
            SharedPushError::Closed => f.write_str("the shared ring's consumer is gone"),
            SharedPushError::TooLong { len, max } => write!(
                f,
                "a {len}-byte record does not fit the shared ring's {max}-byte slots"
            ),
            SharedPushError::Corrupt => {
                f.write_str("the shared ring's consumer index is out of range")
            }
        }
    }
}

impl std::error::Error for SharedPushError {}

/// Why a pop from a shared ring returned no record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SharedPopError {
    /// The ring is empty, and the producer may push more.
    Empty,
    /// The ring is empty and the producer is gone.
    Closed,
    /// The next record is `len` bytes, more than the buffer holds. It stays in the ring.
    BufferTooSmall {
        /// The record's length.
        len: usize,
    },
    /// The producer's index or a record's length is out of range: the peer broke the protocol.
    Corrupt,
}

impl fmt::Display for SharedPopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SharedPopError::Empty => f.write_str("the shared ring is empty"),
            SharedPopError::Closed => {
                f.write_str("the shared ring is empty and its producer is gone")
            }
            SharedPopError::BufferTooSmall { len } => {
                write!(
                    f,
                    "the next record is {len} bytes, more than the buffer holds"
                )
            }
            SharedPopError::Corrupt => {
                f.write_str("the shared ring's producer index or a record length is out of range")
            }
        }
    }
}

impl std::error::Error for SharedPopError {}

/// The writing end of a shared ring.
pub struct SharedProducer<'a> {
    end: End<'a>,
    /// This end's copy of `tail`.
    tail: u64,
    /// The consumer's `head` as last read.
    head: u64,
}

// SAFETY: an end is a view of memory that stays valid for `'a` and that only this end, of its
// side, writes, by the caller's promise to `SharedLayout::producer`. Nothing in it is tied to the
// thread that made it, so it may move to another thread.
unsafe impl Send for SharedProducer<'_> {}

impl SharedProducer<'_> {
    /// Copies `record` into the next slot. Never allocates or blocks.
    ///
    /// # Errors
    ///
    /// [`SharedPushError::TooLong`] for a record longer than a slot, [`SharedPushError::Closed`]
    /// when the consumer is gone, [`SharedPushError::Full`] when every slot is unread, and
    /// [`SharedPushError::Corrupt`] when the consumer's index is out of range.
    pub fn push(&mut self, record: &[u8]) -> Result<(), SharedPushError> {
        let end = &self.end;
        let max = end.layout.slot_bytes as usize;
        if record.len() > max {
            return Err(SharedPushError::TooLong {
                len: record.len(),
                max,
            });
        }
        if end.closed(CONSUMER_CLOSED) {
            return Err(SharedPushError::Closed);
        }
        let slots = u64::from(end.layout.slots);
        if self.tail.wrapping_sub(self.head) >= slots {
            self.head = end.header.head.load(Ordering::Acquire);
            let used = self.tail.wrapping_sub(self.head);
            if used > slots {
                return Err(SharedPushError::Corrupt);
            }
            if used == slots {
                return Err(SharedPushError::Full);
            }
        }
        end.shadow_write(self.tail);
        let slot = end.slot(self.tail);
        // `record.len()` is at most `slot_bytes`, at most 1 MiB, so it fits a `u32`.
        let length = (record.len() as u32).to_le();
        // SAFETY: the slot at `tail` lies outside `head..tail`: the acquire load of `head` above, or
        // an earlier one, saw the consumer finish reading it, so a peer that keeps the protocol
        // does not touch it now. The slot is 8-aligned (the memory is 128-aligned and every stride
        // is a multiple of 8), holds the length word, and has `slot_bytes` bytes after it, which
        // `record` fits. Every write is atomic, so even a peer that breaks the protocol makes no
        // data race.
        unsafe {
            length_word(slot).store(length, Ordering::Relaxed);
            copy_in(record, slot.add(LENGTH_BYTES));
        }
        self.tail = self.tail.wrapping_add(1);
        end.header.tail.store(self.tail, end.publish_ordering());
        Ok(())
    }

    /// Whether the consumer is gone.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.end.closed(CONSUMER_CLOSED)
    }

    /// The ring's layout.
    #[must_use]
    pub fn layout(&self) -> SharedLayout {
        self.end.layout
    }
}

impl Drop for SharedProducer<'_> {
    fn drop(&mut self) {
        self.end
            .header
            .closed
            .fetch_or(PRODUCER_CLOSED, Ordering::Release);
    }
}

/// The reading end of a shared ring.
pub struct SharedConsumer<'a> {
    end: End<'a>,
    /// This end's copy of `head`.
    head: u64,
    /// The producer's `tail` as last read.
    tail: u64,
}

// SAFETY: as for `SharedProducer`, for the consumer's side.
unsafe impl Send for SharedConsumer<'_> {}

impl SharedConsumer<'_> {
    /// Copies the oldest record into `out` and returns its length. Never allocates or blocks.
    ///
    /// # Errors
    ///
    /// [`SharedPopError::Empty`] when the ring is empty, [`SharedPopError::Closed`] when it is empty
    /// and the producer is gone, [`SharedPopError::BufferTooSmall`] when the record is longer than
    /// `out` (it stays in the ring), and [`SharedPopError::Corrupt`] when the producer's index or
    /// the record's length is out of range.
    pub fn pop_into(&mut self, out: &mut [u8]) -> Result<usize, SharedPopError> {
        if self.head == self.tail && !self.reload()? {
            if !self.end.closed(PRODUCER_CLOSED) {
                return Err(SharedPopError::Empty);
            }
            // The producer's last push happens before its close, so a reload now finds it.
            if !self.reload()? {
                return Err(SharedPopError::Closed);
            }
        }
        let end = &self.end;
        end.shadow_read(self.head);
        let slot = end.slot(self.head);
        // SAFETY: the slot at `head` lies inside `head..tail`: the acquire load of `tail` saw the
        // producer's writes to it, and the producer will not write it again until the release
        // store of `head` below. The slot is 8-aligned and starts with its length word.
        //
        // The length is read once, with an atomic load, into `len`, and only that copy is checked
        // and used: a hostile peer that rewrites the word after the check cannot change the bound
        // the copy below uses.
        let len = u32::from_le(unsafe { length_word(slot) }.load(Ordering::Relaxed)) as usize;
        if len > end.layout.slot_bytes as usize {
            return Err(SharedPopError::Corrupt);
        }
        if len > out.len() {
            return Err(SharedPopError::BufferTooSmall { len });
        }
        // SAFETY: as above; the record's `len` bytes follow the length word, inside the slot,
        // since `len` is at most `slot_bytes`, and `out` holds at least `len` bytes. Every read is
        // atomic, so a peer that writes the bytes during the copy makes no data race: it can
        // garble the record, which the caller parses from `out`, but reach nothing outside it.
        unsafe {
            copy_out(slot.add(LENGTH_BYTES), &mut out[..len]);
        }
        self.head = self.head.wrapping_add(1);
        end.header.head.store(self.head, Ordering::Release);
        Ok(len)
    }

    /// Reloads the producer's index; whether a record is now ready.
    fn reload(&mut self) -> Result<bool, SharedPopError> {
        self.tail = self.end.header.tail.load(Ordering::Acquire);
        let ready = self.tail.wrapping_sub(self.head);
        if ready > u64::from(self.end.layout.slots) {
            return Err(SharedPopError::Corrupt);
        }
        Ok(ready > 0)
    }

    /// Whether the producer is gone. Records it pushed first may still be ready to pop.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.end.closed(PRODUCER_CLOSED)
    }

    /// The ring's layout.
    #[must_use]
    pub fn layout(&self) -> SharedLayout {
        self.end.layout
    }
}

impl Drop for SharedConsumer<'_> {
    fn drop(&mut self) {
        self.end
            .header
            .closed
            .fetch_or(CONSUMER_CLOSED, Ordering::Release);
    }
}

impl fmt::Debug for SharedProducer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedProducer")
            .field("layout", &self.end.layout)
            .field("tail", &self.tail)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for SharedConsumer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedConsumer")
            .field("layout", &self.end.layout)
            .field("head", &self.head)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;
