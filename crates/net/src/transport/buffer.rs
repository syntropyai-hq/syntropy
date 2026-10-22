// SPDX-License-Identifier: MIT OR Apache-2.0

//! Registered buffer pools, the descriptors a frame carries, and the leases a received frame
//! holds (RFC-0001 §5.8, RFC-0031 §2, §8). Owner: RFC-0031.
//!
//! A [`BufferPool`] divides memory into equal slots, 64-byte aligned. The memory comes from the
//! tier manager's pinned network sub-arena (RFC-0031 §8), reached through [`PoolMemory`]. `tier`
//! cannot depend on `net` (RFC-0001 §4.2 rule 3), so its `NetPool` hands out owned leases that meet
//! the contract (TIER-E2-T21), and the runner implements [`PoolMemory`] over a lease, in the one
//! module RFC-0001 §4.2 rule 8 lets it hold `unsafe` in, and registers it with the transports
//! (TIER-E2-T25). [`HeapMemory`] is a plain heap implementation for tests and for pools that need
//! no pinning.
//!
//! **Each slot has one state**, kept in an atomic: free, held by one [`SlotWriter`], or held by
//! one or more [`SlotReader`]s. A writer is the only way to get `&mut` access to a slot's bytes,
//! and readers give `&` access, so the pool upholds Rust's aliasing rule slot by slot. Slots never
//! overlap. Freezing a writer turns it into a reader; the slot is free again when the last reader
//! is dropped. No allocation and no lock is taken per frame: claiming, freezing, cloning a reader
//! and dropping one are single atomic operations, or a compare-and-swap loop on one word.
//!
//! **Handing out slots** (RFC-0031 §8). [`BufferPool::take`] claims any free slot without a lock:
//! each taker starts one slot further round than the last and claims the first free slot it
//! finds. [`BufferPool::claim`] claims a given slot.
//!
//! **A slot's generation** (RFC-0031 §8, note of 2026-10-22). The atomic word holds a generation
//! beside the state, which advances each time the slot becomes free. A descriptor carries the
//! generation of the freeze it was made from, and [`BufferPool::retain_payload`] and
//! [`BufferPool::retain_ext`] lease a slot only at that generation. A descriptor whose slot was
//! freed and frozen again since is refused, and the channel's `try_send` returns
//! `SendError::Invalid`, rather than send the new bytes under the old name.
//!
//! A frame names its extension and payload with [`ExtBuf`] and [`PayloadSlice`]: `Copy`
//! descriptors of a range inside a frozen slot, with no pointer. The channel that sends the frame
//! takes its own reader for each ([`BufferPool::retain_payload`]), and keeps it until the peer
//! acknowledges the frame (RFC-0031 §7.3), so the caller may drop its reader as soon as
//! `try_send` returns. A received frame holds [`ExtRef`] and [`PayloadRef`]: leases on its
//! receive slot, which return the slot when the frame is dropped.
//!
//! DIST-E7-T01 built the types; DIST-E7-T12 added the hand-out and the generation. The rest of
//! RFC-0031 §8 sits beside this module: the registration tables
//! ([`registration`](super::registration)), the slot sizes from the plan
//! ([`sizing`](super::sizing)), and multi-part prefill landing ([`landing`](super::landing)).

use std::alloc::{self, Layout};
use std::fmt;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use syntropy_core::resource::TransportKind;

/// The alignment of every pool's memory and of every slot, and the unit of a slot's size
/// (RFC-0031 §8).
pub const SLOT_ALIGN: usize = 64;

/// Identifies a pool among a runner's pools. The pool's creator assigns it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PoolId(pub u32);

/// What [`Transport::register`](super::Transport::register) returns: a pool registered with one
/// transport (RFC-0001 §5.8). A [`ChannelSpec`](super::ChannelSpec) names the registration its
/// receive slots come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PoolRegistration {
    /// The pool registered.
    pub pool: PoolId,
    /// The transport it is registered with.
    pub transport: TransportKind,
    /// The backend's own handle for the registration, such as an RDMA memory region's key; 0
    /// for a transport that needs no registration (`TransportCaps::needs_registration`).
    pub handle: u64,
}

/// Memory a [`BufferPool`] divides into slots.
///
/// # Safety
///
/// An implementation promises, for as long as the value lives:
/// - **Extent.** [`base`](Self::base) points to [`size`](Self::size) bytes that are allocated,
///   readable and writable, and both methods return the same values on every call.
/// - **Alignment.** `base` is a multiple of [`SLOT_ALIGN`] (64 bytes).
/// - **Lifetime.** The bytes stay valid, at that address, until the value is dropped. Moving the
///   value does not move them.
/// - **No other users.** Nothing but the [`BufferPool`] that owns the value reads or writes the
///   bytes, through references or pointers. The pool's slot states are the only access control.
///   Hardware that reaches the bytes follows the same states: an RDMA NIC a transport registered
///   the pool with, and a GPU's copy engine when the pool is registered for DMA. It writes a slot
///   only while the slot's [`SlotWriter`] is held, and reads a slot only while a [`SlotReader`] of
///   it exists, by whoever issued the access: a transport backend, or a dispatcher whose device
///   copy targets the slot. That holder keeps the writer or reader until the access completes,
///   not only until it is issued.
pub unsafe trait PoolMemory: Send + Sync {
    /// The first byte.
    fn base(&self) -> NonNull<u8>;
    /// The number of bytes.
    fn size(&self) -> usize;
}

/// Zeroed heap memory, aligned to [`SLOT_ALIGN`]: a [`PoolMemory`] for tests and for pools that
/// need no pinning.
pub struct HeapMemory {
    base: NonNull<u8>,
    layout: Layout,
}

impl HeapMemory {
    /// `size` zeroed bytes.
    ///
    /// # Panics
    ///
    /// If `size` is 0, or larger than an allocation can be.
    pub fn new(size: usize) -> HeapMemory {
        assert!(size > 0, "a pool's memory is at least one byte");
        let layout = Layout::from_size_align(size, SLOT_ALIGN).expect("a valid layout");
        // SAFETY: `layout` has a non-zero size, checked above.
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        let base = NonNull::new(ptr).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        HeapMemory { base, layout }
    }
}

impl Drop for HeapMemory {
    fn drop(&mut self) {
        // SAFETY: `base` was allocated in `new` with `layout`, and is freed only here, once.
        unsafe { alloc::dealloc(self.base.as_ptr(), self.layout) }
    }
}

// SAFETY: `HeapMemory` owns its allocation outright; the pointer is not shared with anything
// that could be left behind on another thread, and the bytes are reached only through the
// owning pool's slot states.
unsafe impl Send for HeapMemory {}
// SAFETY: `&HeapMemory` gives out only the address and the size, never the bytes themselves.
unsafe impl Sync for HeapMemory {}

// SAFETY: `base` is the start of a live allocation of `layout.size()` bytes, aligned to
// `SLOT_ALIGN` by `layout`, freed only when the value is dropped and reachable through nothing
// but this value, which hands it to one pool.
unsafe impl PoolMemory for HeapMemory {
    fn base(&self) -> NonNull<u8> {
        self.base
    }

    fn size(&self) -> usize {
        self.layout.size()
    }
}

/// Why a [`BufferPool`] could not be made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolError {
    /// The slot size is 0 or not a multiple of [`SLOT_ALIGN`].
    SlotSize(usize),
    /// A pool has at least one slot.
    NoSlots,
    /// The memory is smaller than `slots × slot_size`.
    TooSmall { needed: usize, available: usize },
    /// `slots × slot_size` is more bytes than one allocation can have: past `isize::MAX`.
    TooLarge { slot_size: usize, slots: u32 },
    /// The memory's base address is not a multiple of [`SLOT_ALIGN`].
    BaseUnaligned,
}

impl fmt::Display for PoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PoolError::SlotSize(size) => {
                write!(
                    f,
                    "slot size {size} is not a positive multiple of {SLOT_ALIGN}"
                )
            }
            PoolError::NoSlots => write!(f, "a pool has at least one slot"),
            PoolError::TooSmall { needed, available } => {
                write!(
                    f,
                    "the slots need {needed} bytes; the memory has {available}"
                )
            }
            PoolError::BaseUnaligned => {
                write!(f, "the memory is not aligned to {SLOT_ALIGN} bytes")
            }
            PoolError::TooLarge { slot_size, slots } => write!(
                f,
                "{slots} slots of {slot_size} bytes are more than one allocation can hold"
            ),
        }
    }
}

impl std::error::Error for PoolError {}

/// The bytes `slots` slots of `slot_size` bytes take, after checking the shape: `slot_size` a
/// positive multiple of [`SLOT_ALIGN`], at least one slot, and a product computed without
/// overflow that one allocation can hold (at most `isize::MAX`). Every slot's bounds rest on
/// this check.
fn pool_bytes(slot_size: usize, slots: u32) -> Result<usize, PoolError> {
    if slot_size == 0 || !slot_size.is_multiple_of(SLOT_ALIGN) {
        return Err(PoolError::SlotSize(slot_size));
    }
    if slots == 0 {
        return Err(PoolError::NoSlots);
    }
    // A `usize` times a `u32` always fits in 128 bits.
    let bytes = slot_size as u128 * u128::from(slots);
    if bytes > isize::MAX as u128 {
        return Err(PoolError::TooLarge { slot_size, slots });
    }
    Ok(bytes as usize)
}

/// A slot's state: free, held by one writer, or held by this many readers.
const FREE: u32 = 0;
const WRITING: u32 = u32::MAX;
/// The most readers a slot may have; one more aborts, as an `Arc` count overflow does.
const MAX_READERS: u32 = WRITING - 1;

/// A slot's word: its generation in the high 32 bits, its state in the low 32 (RFC-0031 §8,
/// note of 2026-10-22). The generation advances each time the slot becomes free, so a descriptor
/// made from an earlier freeze names an older generation.
fn word(generation: u32, state: u32) -> u64 {
    (u64::from(generation) << 32) | u64::from(state)
}

fn state_of(word: u64) -> u32 {
    word as u32
}

fn generation_of(word: u64) -> u32 {
    (word >> 32) as u32
}

struct PoolInner {
    id: PoolId,
    memory: Box<dyn PoolMemory>,
    slot_size: usize,
    states: Box<[AtomicU64]>,
    /// Where the next [`BufferPool::take`] starts looking, so takers spread over the slots.
    cursor: AtomicU32,
}

/// Memory divided into equal, 64-byte-aligned slots, each held by at most one writer or by
/// readers (RFC-0031 §8). Cloning the pool clones a handle to the same slots.
#[derive(Clone)]
pub struct BufferPool {
    inner: Arc<PoolInner>,
}

impl fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferPool")
            .field("id", &self.inner.id)
            .field("slot_size", &self.inner.slot_size)
            .field("slots", &self.inner.states.len())
            .finish()
    }
}

impl BufferPool {
    /// `slots` slots of `slot_size` bytes at the start of `memory`, all free.
    pub fn new(
        id: PoolId,
        memory: Box<dyn PoolMemory>,
        slot_size: usize,
        slots: u32,
    ) -> Result<BufferPool, PoolError> {
        let needed = pool_bytes(slot_size, slots)?;
        if needed > memory.size() {
            return Err(PoolError::TooSmall {
                needed,
                available: memory.size(),
            });
        }
        if !(memory.base().as_ptr() as usize).is_multiple_of(SLOT_ALIGN) {
            return Err(PoolError::BaseUnaligned);
        }
        let states = (0..slots).map(|_| AtomicU64::new(word(0, FREE))).collect();
        Ok(BufferPool {
            inner: Arc::new(PoolInner {
                id,
                memory,
                slot_size,
                states,
                cursor: AtomicU32::new(0),
            }),
        })
    }

    /// A pool of `slots` slots of `slot_size` bytes in zeroed [`HeapMemory`].
    pub fn heap(id: PoolId, slot_size: usize, slots: u32) -> Result<BufferPool, PoolError> {
        let size = pool_bytes(slot_size, slots)?;
        BufferPool::new(id, Box::new(HeapMemory::new(size)), slot_size, slots)
    }

    pub fn id(&self) -> PoolId {
        self.inner.id
    }

    /// Each slot's size in bytes, a multiple of [`SLOT_ALIGN`].
    pub fn slot_size(&self) -> usize {
        self.inner.slot_size
    }

    pub fn slots(&self) -> u32 {
        self.inner.states.len() as u32
    }

    /// Whether `other` is a handle to this pool.
    pub fn same_pool(&self, other: &BufferPool) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Whether slot `slot` is free: no writer and no reader holds it. A channel core that hands
    /// out a pool's slots uses it to see which of its delivered frames' slots have come back.
    pub fn is_free(&self, slot: u32) -> bool {
        self.inner
            .states
            .get(slot as usize)
            .is_some_and(|state| state_of(state.load(Ordering::Acquire)) == FREE)
    }

    /// Slot `slot`'s generation: how many times it has become free. A descriptor made while the
    /// slot is frozen carries it; `None` past the last slot.
    pub fn generation(&self, slot: u32) -> Option<u32> {
        let state = self.inner.states.get(slot as usize)?;
        Some(generation_of(state.load(Ordering::Acquire)))
    }

    /// Takes slot `slot` for writing, if it is free.
    pub fn claim(&self, slot: u32) -> Option<SlotWriter> {
        let state = self.inner.states.get(slot as usize)?;
        let mut current = state.load(Ordering::Relaxed);
        loop {
            if state_of(current) != FREE {
                return None;
            }
            let generation = generation_of(current);
            match state.compare_exchange_weak(
                current,
                word(generation, WRITING),
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return Some(SlotWriter {
                        pool: self.clone(),
                        slot,
                        generation,
                        frozen: false,
                    });
                }
                Err(seen) => current = seen,
            }
        }
    }

    /// Takes any free slot for writing, without a lock (RFC-0031 §8): each taker starts at the
    /// next slot after the last taker's start and claims the first free one it finds, with one
    /// compare-and-swap per slot tried. `None` when every slot is held.
    pub fn take(&self) -> Option<SlotWriter> {
        let slots = self.slots();
        let start = self.inner.cursor.fetch_add(1, Ordering::Relaxed) % slots;
        (0..slots).find_map(|i| self.claim((start + i) % slots))
    }

    /// Another reader of slot `slot`, if it is frozen: held by readers, not free and not being
    /// written.
    pub fn reader(&self, slot: u32) -> Option<SlotReader> {
        self.reader_where(slot, |_| true)
    }

    /// [`BufferPool::reader`], only while the slot is still at `generation`: a descriptor's slot
    /// that was freed and frozen again since gives `None` (RFC-0031 §8, note of 2026-10-22).
    pub fn reader_at(&self, slot: u32, generation: u32) -> Option<SlotReader> {
        self.reader_where(slot, |current| current == generation)
    }

    fn reader_where(&self, slot: u32, generation_ok: impl Fn(u32) -> bool) -> Option<SlotReader> {
        let state = self.inner.states.get(slot as usize)?;
        let mut current = state.load(Ordering::Relaxed);
        loop {
            let readers = state_of(current);
            if readers == FREE || readers == WRITING || !generation_ok(generation_of(current)) {
                return None;
            }
            if readers >= MAX_READERS {
                std::process::abort();
            }
            match state.compare_exchange_weak(
                current,
                current + 1,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return Some(SlotReader {
                        pool: self.clone(),
                        slot,
                        generation: generation_of(current),
                    });
                }
                Err(seen) => current = seen,
            }
        }
    }

    /// A lease on the range `slice` names, for a channel to keep until the frame is
    /// acknowledged; `None` if `slice` names another pool, a slot that is not frozen, a slot
    /// freed and frozen again since the descriptor was made (an older generation), or a range
    /// outside the slot. An empty slice gives an empty lease.
    pub fn retain_payload(&self, slice: PayloadSlice) -> Option<PayloadRef> {
        if slice.len == 0 {
            return Some(PayloadRef::empty());
        }
        if slice.pool != self.id() {
            return None;
        }
        let reader = self.reader_at(slice.slot, slice.generation)?;
        PayloadRef::new(reader, slice.offset, slice.len)
    }

    /// [`retain_payload`](Self::retain_payload) for an extension.
    pub fn retain_ext(&self, ext: ExtBuf) -> Option<ExtRef> {
        if ext.len == 0 {
            return Some(ExtRef::empty());
        }
        if ext.pool != self.id() {
            return None;
        }
        ExtRef::new(
            self.reader_at(ext.slot, ext.generation)?,
            ext.offset,
            ext.len,
        )
    }

    fn state(&self, slot: u32) -> &AtomicU64 {
        &self.inner.states[slot as usize]
    }

    /// The first byte of slot `slot`, which is in bounds.
    fn slot_ptr(&self, slot: u32) -> *mut u8 {
        debug_assert!(slot < self.slots());
        // SAFETY: `slot < slots`, and `new` checked, without overflow (`pool_bytes`), that
        // `slots × slot_size` is at most `isize::MAX` and at most the memory's size. So
        // `slot × slot_size` cannot overflow, and the offset stays inside the allocation.
        unsafe {
            self.inner
                .memory
                .base()
                .as_ptr()
                .add(slot as usize * self.inner.slot_size)
        }
    }
}

/// Slot `slot` of a pool, held for writing: no reader and no other writer exists until it is
/// frozen or dropped. Dropping it frees the slot.
pub struct SlotWriter {
    pool: BufferPool,
    slot: u32,
    generation: u32,
    frozen: bool,
}

impl SlotWriter {
    pub fn pool(&self) -> &BufferPool {
        &self.pool
    }

    pub fn slot(&self) -> u32 {
        self.slot
    }

    /// The slot's bytes.
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: the slot's state is `WRITING`, set by this writer's claim, so no reader or
        // other writer of these bytes exists until this writer is frozen or dropped, which needs
        // the `&mut self` this borrow holds. The range is `slot_size` bytes from `slot_ptr`, so it
        // ends at `(slot + 1) × slot_size ≤ slots × slot_size`, which `new` checked is at most the
        // memory's size (`pool_bytes`). It is aligned, and lives while the pool does;
        // `self.pool` keeps it alive.
        unsafe {
            std::slice::from_raw_parts_mut(self.pool.slot_ptr(self.slot), self.pool.slot_size())
        }
    }

    /// The slot's generation while this writer holds it.
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Ends the writing and makes the slot readable: the returned reader is its first.
    pub fn freeze(mut self) -> SlotReader {
        self.frozen = true;
        self.pool
            .state(self.slot)
            .store(word(self.generation, 1), Ordering::Release);
        SlotReader {
            pool: self.pool.clone(),
            slot: self.slot,
            generation: self.generation,
        }
    }
}

impl Drop for SlotWriter {
    fn drop(&mut self) {
        if !self.frozen {
            // Free again, at the next generation.
            self.pool.state(self.slot).store(
                word(self.generation.wrapping_add(1), FREE),
                Ordering::Release,
            );
        }
    }
}

impl fmt::Debug for SlotWriter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SlotWriter")
            .field("pool", &self.pool.id())
            .field("slot", &self.slot)
            .finish()
    }
}

/// Slot `slot` of a pool, held for reading. The slot cannot be written while any reader of it
/// exists; it is free again when the last one is dropped.
pub struct SlotReader {
    pool: BufferPool,
    slot: u32,
    generation: u32,
}

impl SlotReader {
    pub fn pool(&self) -> &BufferPool {
        &self.pool
    }

    pub fn slot(&self) -> u32 {
        self.slot
    }

    /// The slot's generation, which the descriptors made from this reader carry.
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// The slot's bytes.
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: this reader is counted in the slot's state, so no writer exists until it is
        // dropped: a claim needs the state to be `FREE`. The range is `slot_size` bytes from
        // `slot_ptr`, so it ends at `(slot + 1) × slot_size ≤ slots × slot_size`, which `new`
        // checked is at most the memory's size (`pool_bytes`). It is aligned, and lives while
        // the pool does; `self.pool` keeps it alive.
        unsafe { std::slice::from_raw_parts(self.pool.slot_ptr(self.slot), self.pool.slot_size()) }
    }

    /// The descriptor of `len` bytes at `offset` in this slot; `None` past the slot's end.
    pub fn payload(&self, offset: u32, len: u32) -> Option<PayloadSlice> {
        self.in_bounds(offset, len).then_some(PayloadSlice {
            pool: self.pool.id(),
            slot: self.slot,
            generation: self.generation,
            offset,
            len,
        })
    }

    /// The descriptor of an extension of `len` bytes at `offset` in this slot; `None` past the
    /// slot's end.
    pub fn ext(&self, offset: u32, len: u32) -> Option<ExtBuf> {
        self.in_bounds(offset, len).then_some(ExtBuf {
            pool: self.pool.id(),
            slot: self.slot,
            generation: self.generation,
            offset,
            len,
        })
    }

    fn in_bounds(&self, offset: u32, len: u32) -> bool {
        u64::from(offset) + u64::from(len) <= self.pool.slot_size() as u64
    }
}

impl Clone for SlotReader {
    fn clone(&self) -> SlotReader {
        // The count is in the low half and stays below `MAX_READERS`, so the add never carries
        // into the generation.
        let previous = self.pool.state(self.slot).fetch_add(1, Ordering::Relaxed);
        if state_of(previous) >= MAX_READERS {
            std::process::abort();
        }
        SlotReader {
            pool: self.pool.clone(),
            slot: self.slot,
            generation: self.generation,
        }
    }
}

impl Drop for SlotReader {
    fn drop(&mut self) {
        // The last reader leaves the slot free at the next generation; any other just counts
        // down. `Release` orders this reader's reads before a later claim's `Acquire`.
        let state = self.pool.state(self.slot);
        let mut current = state.load(Ordering::Relaxed);
        loop {
            let next = if state_of(current) == 1 {
                word(generation_of(current).wrapping_add(1), FREE)
            } else {
                current - 1
            };
            match state.compare_exchange_weak(current, next, Ordering::Release, Ordering::Relaxed) {
                Ok(_) => return,
                Err(seen) => current = seen,
            }
        }
    }
}

impl fmt::Debug for SlotReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SlotReader")
            .field("pool", &self.pool.id())
            .field("slot", &self.slot)
            .finish()
    }
}

/// A frame's payload: `len` bytes at `offset` in a frozen slot of pool `pool` (RFC-0031 §2). A
/// descriptor only; the transport reads the bytes from the registered slot, and on RDMA and
/// shared memory never copies them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PayloadSlice {
    pub pool: PoolId,
    pub slot: u32,
    /// The slot's generation when the descriptor was made: a lease through it is refused once
    /// the slot has been freed since (RFC-0031 §8, note of 2026-10-22).
    pub generation: u32,
    pub offset: u32,
    pub len: u32,
}

impl PayloadSlice {
    /// No payload.
    pub const EMPTY: PayloadSlice = PayloadSlice {
        pool: PoolId(0),
        slot: 0,
        generation: 0,
        offset: 0,
        len: 0,
    };

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A frame's extension, its type-specific header: `len` bytes at `offset` in a frozen slot of
/// pool `pool`, padding included, so `len` is a multiple of 64 (RFC-0031 §2, §6.1). Where the
/// base header goes, and whether it shares the extension's registered area, is the header-area
/// design of RFC-0031 §8 (DIST-E7-T12).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExtBuf {
    pub pool: PoolId,
    pub slot: u32,
    /// The slot's generation when the descriptor was made, as [`PayloadSlice::generation`].
    pub generation: u32,
    pub offset: u32,
    pub len: u32,
}

impl ExtBuf {
    /// No extension.
    pub const EMPTY: ExtBuf = ExtBuf {
        pool: PoolId(0),
        slot: 0,
        generation: 0,
        offset: 0,
        len: 0,
    };

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A lease on `len` bytes at `offset` of a frozen slot, or on nothing.
struct Lease {
    reader: Option<SlotReader>,
    offset: u32,
    len: u32,
}

impl Lease {
    fn new(reader: SlotReader, offset: u32, len: u32) -> Option<Lease> {
        reader.in_bounds(offset, len).then_some(Lease {
            reader: Some(reader),
            offset,
            len,
        })
    }

    const EMPTY: Lease = Lease {
        reader: None,
        offset: 0,
        len: 0,
    };

    fn bytes(&self) -> &[u8] {
        match &self.reader {
            Some(reader) => {
                let start = self.offset as usize;
                &reader.bytes()[start..start + self.len as usize]
            }
            None => &[],
        }
    }

    fn slot(&self) -> Option<(PoolId, u32)> {
        self.reader
            .as_ref()
            .map(|reader| (reader.pool().id(), reader.slot()))
    }
}

/// A received frame's payload, borrowed from its receive slot (RFC-0031 §2). Dropping it, with
/// the rest of the [`InFrame`](super::InFrame), returns the slot; the channel then re-posts it and
/// may grant a credit for it (§7.1). On the sending side, a channel holds one of these per
/// unacknowledged frame, which keeps its payload slot from being reused (§7.3).
pub struct PayloadRef(Lease);

impl PayloadRef {
    /// A lease on `len` bytes at `offset` of `reader`'s slot; `None` past the slot's end.
    pub fn new(reader: SlotReader, offset: u32, len: u32) -> Option<PayloadRef> {
        Lease::new(reader, offset, len).map(PayloadRef)
    }

    /// No payload.
    pub fn empty() -> PayloadRef {
        PayloadRef(Lease::EMPTY)
    }

    pub fn bytes(&self) -> &[u8] {
        self.0.bytes()
    }

    pub fn len(&self) -> usize {
        self.0.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.0.len == 0
    }

    /// The pool and slot the bytes sit in; `None` for no payload.
    pub fn slot(&self) -> Option<(PoolId, u32)> {
        self.0.slot()
    }
}

impl fmt::Debug for PayloadRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PayloadRef")
            .field("slot", &self.slot())
            .field("offset", &self.0.offset)
            .field("len", &self.0.len)
            .finish()
    }
}

/// A received frame's extension, borrowed from its receive slot (RFC-0031 §2): the same lease as
/// [`PayloadRef`], for the extension's bytes, padding included.
pub struct ExtRef(Lease);

impl ExtRef {
    /// A lease on `len` bytes at `offset` of `reader`'s slot; `None` past the slot's end.
    pub fn new(reader: SlotReader, offset: u32, len: u32) -> Option<ExtRef> {
        Lease::new(reader, offset, len).map(ExtRef)
    }

    /// No extension.
    pub fn empty() -> ExtRef {
        ExtRef(Lease::EMPTY)
    }

    pub fn bytes(&self) -> &[u8] {
        self.0.bytes()
    }

    pub fn len(&self) -> usize {
        self.0.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.0.len == 0
    }

    /// The pool and slot the bytes sit in; `None` for no extension.
    pub fn slot(&self) -> Option<(PoolId, u32)> {
        self.0.slot()
    }
}

impl fmt::Debug for ExtRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtRef")
            .field("slot", &self.slot())
            .field("offset", &self.0.offset)
            .field("len", &self.0.len)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(slots: u32) -> BufferPool {
        BufferPool::heap(PoolId(7), 128, slots).expect("a valid pool")
    }

    #[test]
    fn a_pool_checks_its_shape() {
        assert_eq!(
            BufferPool::heap(PoolId(0), 100, 2).unwrap_err(),
            PoolError::SlotSize(100)
        );
        assert_eq!(
            BufferPool::heap(PoolId(0), 0, 2).unwrap_err(),
            PoolError::SlotSize(0)
        );
        assert_eq!(
            BufferPool::heap(PoolId(0), 64, 0).unwrap_err(),
            PoolError::NoSlots
        );
        assert_eq!(
            BufferPool::new(PoolId(0), Box::new(HeapMemory::new(256)), 128, 3).unwrap_err(),
            PoolError::TooSmall {
                needed: 384,
                available: 256
            }
        );
        let pool = pool(3);
        assert_eq!(
            (pool.id(), pool.slot_size(), pool.slots()),
            (PoolId(7), 128, 3)
        );
        assert!(pool.claim(3).is_none(), "slot 3 is past the end");
    }

    /// `slots × slot_size` cannot wrap past the size check (review of #1737, B1). In a build
    /// without overflow checks, as `[profile.release]` ships, an unchecked product of 2^24 slots
    /// of 2^40 bytes wraps to 0 and passes any size check.
    #[test]
    #[cfg(target_pointer_width = "64")]
    fn a_size_that_overflows_is_refused() {
        let small = || -> Box<dyn PoolMemory> { Box::new(HeapMemory::new(64)) };
        let too_large = |slot_size, slots| PoolError::TooLarge { slot_size, slots };
        // 2^24 × 2^40 = 2^64: the probe's input.
        assert_eq!(
            BufferPool::new(PoolId(0), small(), 1 << 40, 1 << 24).unwrap_err(),
            too_large(1 << 40, 1 << 24)
        );
        assert_eq!(
            BufferPool::heap(PoolId(0), 1 << 40, 1 << 24).unwrap_err(),
            too_large(1 << 40, 1 << 24),
            "refused before it allocates"
        );
        // The boundary: 2^23 × 2^40 = 2^63 is one past isize::MAX; one slot fewer is within it,
        // and then simply more than the memory holds.
        assert_eq!(
            BufferPool::new(PoolId(0), small(), 1 << 40, 1 << 23).unwrap_err(),
            too_large(1 << 40, 1 << 23)
        );
        assert_eq!(
            BufferPool::new(PoolId(0), small(), 1 << 40, (1 << 23) - 1).unwrap_err(),
            PoolError::TooSmall {
                needed: (1 << 63) - (1 << 40),
                available: 64
            }
        );
        // The most slots at the smallest size is large, but no overflow.
        assert_eq!(
            BufferPool::new(PoolId(0), small(), 64, u32::MAX).unwrap_err(),
            PoolError::TooSmall {
                needed: 64 * u32::MAX as usize,
                available: 64
            }
        );
        assert!(
            too_large(1 << 40, 1 << 24)
                .to_string()
                .contains("16777216 slots")
        );
    }

    #[test]
    fn slots_are_aligned_and_do_not_overlap() {
        let pool = pool(3);
        let mut writers: Vec<SlotWriter> = (0..3).map(|i| pool.claim(i).unwrap()).collect();
        for (i, writer) in writers.iter_mut().enumerate() {
            let bytes = writer.bytes_mut();
            assert_eq!(bytes.len(), 128);
            assert_eq!(bytes.as_ptr() as usize % SLOT_ALIGN, 0);
            bytes.fill(i as u8 + 1);
        }
        let readers: Vec<SlotReader> = writers.into_iter().map(SlotWriter::freeze).collect();
        for (i, reader) in readers.iter().enumerate() {
            assert!(reader.bytes().iter().all(|&b| b == i as u8 + 1), "slot {i}");
        }
    }

    #[test]
    fn a_slot_has_one_writer_or_readers() {
        let pool = pool(1);
        let mut writer = pool.claim(0).expect("a free slot");
        assert!(pool.claim(0).is_none(), "a second writer");
        assert!(pool.reader(0).is_none(), "a reader while writing");
        writer.bytes_mut()[..3].copy_from_slice(b"abc");
        let first = writer.freeze();
        assert!(pool.claim(0).is_none(), "a writer while frozen");
        let second = pool.reader(0).expect("another reader of a frozen slot");
        let third = second.clone();
        drop(first);
        drop(second);
        assert!(pool.claim(0).is_none(), "a writer while one reader remains");
        assert_eq!(&third.bytes()[..3], b"abc");
        drop(third);
        let mut again = pool.claim(0).expect("free after the last reader");
        assert_eq!(&again.bytes_mut()[..3], b"abc", "slots are not cleared");
        drop(again);
        assert!(pool.claim(0).is_some(), "a dropped writer frees the slot");
    }

    #[test]
    fn descriptors_and_leases_stay_inside_the_slot() {
        let pool = pool(2);
        let mut writer = pool.claim(1).unwrap();
        writer.bytes_mut()[64..68].copy_from_slice(b"wxyz");
        let reader = writer.freeze();
        assert!(reader.payload(100, 29).is_none());
        let slice = reader.payload(64, 4).unwrap();
        assert_eq!(
            slice,
            PayloadSlice {
                pool: PoolId(7),
                slot: 1,
                generation: 0,
                offset: 64,
                len: 4
            }
        );
        let ext = reader.ext(0, 64).unwrap();
        let lease = pool.retain_payload(slice).expect("a frozen slot");
        let ext_lease = pool.retain_ext(ext).expect("a frozen slot");
        assert_eq!(lease.bytes(), b"wxyz");
        assert_eq!(lease.slot(), Some((PoolId(7), 1)));
        assert_eq!(ext_lease.len(), 64);
        drop(reader);
        assert!(pool.claim(1).is_none(), "the leases hold the slot");
        drop((lease, ext_lease));
        assert!(pool.claim(1).is_some());
    }

    #[test]
    fn a_slots_generation_advances_each_time_it_becomes_free() {
        let pool = pool(2);
        assert_eq!(pool.generation(0), Some(0));
        // A writer dropped unfrozen frees the slot.
        drop(pool.claim(0).unwrap());
        assert_eq!(pool.generation(0), Some(1));
        // Freezing keeps the generation; readers keep it while any is left.
        let reader = pool.claim(0).unwrap().freeze();
        assert_eq!(reader.generation(), 1);
        let second = reader.clone();
        drop(reader);
        assert_eq!(pool.generation(0), Some(1), "a reader is left");
        drop(second);
        assert_eq!(pool.generation(0), Some(2), "the last reader freed it");
        assert!(pool.is_free(0));
        assert_eq!(pool.generation(1), Some(0), "slots count apart");
        assert_eq!(pool.generation(2), None);
    }

    /// RFC-0031 §8's note of 2026-10-22: a descriptor whose slot was freed and handed out again
    /// is refused, never leased.
    #[test]
    fn a_descriptor_of_a_freed_and_refrozen_slot_is_refused() {
        let pool = pool(1);
        let mut writer = pool.claim(0).unwrap();
        writer.bytes_mut()[..4].copy_from_slice(b"old!");
        let reader = writer.freeze();
        let stale_payload = reader.payload(0, 4).unwrap();
        let stale_ext = reader.ext(64, 64).unwrap();
        assert!(
            pool.retain_payload(stale_payload).is_some(),
            "fresh: leased"
        );
        // The holder drops the slot's last reader; the slot is claimed and frozen again.
        drop(reader);
        let mut writer = pool.claim(0).unwrap();
        writer.bytes_mut()[..4].copy_from_slice(b"new!");
        let fresh = writer.freeze();
        assert!(
            pool.retain_payload(stale_payload).is_none(),
            "a stale PayloadSlice"
        );
        assert!(pool.retain_ext(stale_ext).is_none(), "a stale ExtBuf");
        // A descriptor of the new freeze leases the new bytes.
        let lease = pool.retain_payload(fresh.payload(0, 4).unwrap()).unwrap();
        assert_eq!(lease.bytes(), b"new!");
    }

    #[test]
    fn take_hands_out_every_free_slot_once() {
        let pool = pool(5);
        let mut held: Vec<SlotWriter> = (0..5).map(|_| pool.take().expect("a free slot")).collect();
        let mut slots: Vec<u32> = held.iter().map(SlotWriter::slot).collect();
        slots.sort_unstable();
        assert_eq!(slots, [0, 1, 2, 3, 4]);
        assert!(pool.take().is_none(), "every slot is held");
        let freed = held.remove(2).slot();
        assert_eq!(pool.take().map(|w| w.slot()), Some(freed));
    }

    /// Takers on several threads never hold one slot at once: each writes its thread's mark into
    /// the slot, checks it is still there, and frees it.
    #[test]
    fn concurrent_takers_never_share_a_slot() {
        let pool = BufferPool::heap(PoolId(1), 64, 8).unwrap();
        let threads: Vec<_> = (0u8..6)
            .map(|mark| {
                let pool = pool.clone();
                std::thread::spawn(move || {
                    let mut taken = 0;
                    for _ in 0..20_000 {
                        let Some(mut writer) = pool.take() else {
                            continue;
                        };
                        writer.bytes_mut().fill(mark);
                        std::hint::spin_loop();
                        assert!(writer.bytes_mut().iter().all(|b| *b == mark));
                        taken += 1;
                        if taken % 2 == 0 {
                            drop(writer.freeze());
                        }
                    }
                    taken
                })
            })
            .collect();
        let taken: u32 = threads
            .into_iter()
            .map(|t| t.join().expect("no panic"))
            .sum();
        assert!(taken > 0);
        assert!((0..8).all(|slot| pool.is_free(slot)));
    }

    #[test]
    fn retaining_needs_the_right_pool_and_a_frozen_slot() {
        let pool = pool(2);
        let other = BufferPool::heap(PoolId(8), 128, 2).unwrap();
        let reader = pool.claim(0).unwrap().freeze();
        let slice = reader.payload(0, 8).unwrap();
        assert!(other.retain_payload(slice).is_none(), "another pool's id");
        let free = PayloadSlice { slot: 1, ..slice };
        assert!(pool.retain_payload(free).is_none(), "a free slot");
        let _writer = pool.claim(1).unwrap();
        assert!(pool.retain_payload(free).is_none(), "a slot being written");
        let past = PayloadSlice {
            offset: 124,
            ..slice
        };
        assert!(pool.retain_payload(past).is_none(), "past the slot's end");
        assert!(pool.retain_payload(PayloadSlice::EMPTY).unwrap().is_empty());
        assert!(pool.retain_ext(ExtBuf::EMPTY).unwrap().is_empty());
        assert!(PayloadRef::empty().bytes().is_empty());
        assert_eq!(ExtRef::empty().slot(), None);
    }

    #[test]
    fn readers_on_other_threads_hold_the_slot() {
        let pool = pool(1);
        let mut writer = pool.claim(0).unwrap();
        writer.bytes_mut().fill(9);
        let reader = writer.freeze();
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let reader = reader.clone();
                std::thread::spawn(move || {
                    reader.bytes().iter().map(|&b| u32::from(b)).sum::<u32>()
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), 9 * 128);
        }
        drop(reader);
        assert!(pool.claim(0).is_some());
    }
}
