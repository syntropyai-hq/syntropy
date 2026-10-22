// SPDX-License-Identifier: MIT OR Apache-2.0

//! One open CPU device: the state `open` puts in the `DeviceCtx` (RFC-0040 §7.3, §2.4).
//!
//! The context is `!Send`, so every provider call on it runs on one thread, and the state uses
//! `Cell` and `RefCell` rather than locks. What a stream's runner needs on another thread (the
//! ring's consumer end, the doorbells, the events, the device's health) is shared through
//! `Arc`s and atomics.

use std::alloc::Layout;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use syntropy_core::{Bytes, ErrorClass};
use syntropy_kernels::{Error, ErrorCode, StreamPriority};
use syntropy_spsc::{Producer, PushError};

use crate::dispatch::{DispatchTable, KernelFn};
use crate::doorbell::Doorbells;
use crate::stream::{EventState, Health, Item, STREAM_CAPACITY, StreamRunner, StreamShared};
use crate::wake::CpuWaker;

/// How long a dropped device waits for its taken runners to drain before it leaks their
/// memory rather than free it under them: 30 s \[target\], a chosen budget.
const DROP_WAIT: Duration = Duration::from_secs(30);

/// A host buffer the device knows: allocated by it, or imported.
#[derive(Debug)]
pub(crate) struct HostBuf {
    pub(crate) addr: u64,
    pub(crate) bytes: u64,
    /// The layout it was allocated with, if the device owns it; `None` for an import.
    pub(crate) owned: Option<Layout>,
}

/// One stream: the ring's producer end, its runner until a thread takes it, and the op list
/// being captured, if any.
#[derive(Debug)]
pub(crate) struct StreamState {
    pub(crate) prio: StreamPriority,
    tx: Producer<Item>,
    pub(crate) runner: Option<StreamRunner>,
    pub(crate) capture: Option<Vec<Item>>,
    pub(crate) shared: Arc<StreamShared>,
    /// Items pushed onto the ring so far.
    pub(crate) pushed: u64,
    /// Whether `stream` has handed this stream out.
    pub(crate) handed_out: bool,
}

/// An open CPU device.
#[derive(Debug)]
pub(crate) struct CpuDevice {
    /// The NUMA node the device is.
    pub(crate) node: u16,
    pub(crate) table: DispatchTable,
    pub(crate) doorbells: Doorbells,
    pub(crate) health: Arc<Health>,
    pub(crate) streams: RefCell<Vec<StreamState>>,
    pub(crate) events: RefCell<Vec<Arc<EventState>>>,
    pub(crate) buffers: RefCell<BTreeMap<u64, HostBuf>>,
    pub(crate) next_handle: Cell<u64>,
    pub(crate) prepared: RefCell<Vec<KernelFn>>,
    pub(crate) graphs: RefCell<Vec<Vec<Item>>>,
    pub(crate) wakers: RefCell<Vec<Arc<CpuWaker>>>,
    /// Memory `free` released while a stream still had items queued, which may hold its
    /// address. It is deallocated once every stream has drained.
    pub(crate) deferred: RefCell<Vec<(u64, Layout)>>,
}

impl CpuDevice {
    pub(crate) fn new(node: u16, table: DispatchTable, doorbells: Doorbells) -> CpuDevice {
        CpuDevice {
            node,
            table,
            doorbells,
            health: Arc::default(),
            streams: RefCell::new(Vec::new()),
            events: RefCell::new(Vec::new()),
            buffers: RefCell::new(BTreeMap::new()),
            next_handle: Cell::new(1),
            prepared: RefCell::new(Vec::new()),
            graphs: RefCell::new(Vec::new()),
            wakers: RefCell::new(Vec::new()),
            deferred: RefCell::new(Vec::new()),
        }
    }

    /// Whether every stream has run everything pushed onto it, or lost the runner that would
    /// (its thread dropped it), so no queued item can still touch memory.
    pub(crate) fn quiet(&self) -> bool {
        self.streams.borrow().iter().all(|s| {
            s.shared.completed.load(Ordering::Acquire) >= s.pushed
                || (s.runner.is_none() && Arc::strong_count(&s.shared) == 1)
        })
    }

    /// Deallocates `addr` now if no queued item can touch it, else once every stream has
    /// drained.
    pub(crate) fn release(&self, addr: u64, layout: Layout) {
        self.deferred.borrow_mut().push((addr, layout));
        self.release_deferred();
    }

    /// Deallocates the deferred memory if every stream has drained. Called from `free`,
    /// `synchronize` and `poll`; with nothing deferred it reads no stream.
    pub(crate) fn release_deferred(&self) {
        if self.deferred.borrow().is_empty() || !self.quiet() {
            return;
        }
        for (addr, layout) in self.deferred.borrow_mut().drain(..) {
            // SAFETY: each entry is memory `alloc` allocated at `addr` with `layout`, which
            // `free` took out of the buffer table, so it is deallocated once. Every stream has
            // drained, so no queued copy still holds the address.
            unsafe { std::alloc::dealloc(addr as *mut u8, layout) };
        }
    }

    /// A fresh handle for a buffer or registration.
    pub(crate) fn handle(&self) -> u64 {
        let handle = self.next_handle.get();
        self.next_handle.set(handle + 1);
        handle
    }

    /// Creates a stream of `prio`, with its ring and runner. Called at `open` and between
    /// steps, never inside one.
    pub(crate) fn add_stream(&self, prio: StreamPriority) -> Result<usize, Error> {
        let (tx, rx) = syntropy_spsc::ring(STREAM_CAPACITY)
            .map_err(|error| internal(format!("a stream's ring: {error}")))?;
        let shared = Arc::new(StreamShared {
            health: Arc::clone(&self.health),
            ..StreamShared::default()
        });
        let runner = StreamRunner::new(rx, self.doorbells.clone(), Arc::clone(&shared));
        let mut streams = self.streams.borrow_mut();
        streams.push(StreamState {
            prio,
            tx,
            runner: Some(runner),
            capture: None,
            shared,
            pushed: 0,
            handed_out: false,
        });
        Ok(streams.len() - 1)
    }

    /// Pushes `item` onto stream `index`, or onto the op list it is capturing. Allocation-free
    /// and lock-free outside a capture.
    ///
    /// # Errors
    ///
    /// No such stream, or a full ring: a step program longer than [`STREAM_CAPACITY`] items.
    pub(crate) fn push(&self, index: u64, item: Item) -> Result<(), Error> {
        let mut streams = self.streams.borrow_mut();
        let stream = usize::try_from(index)
            .ok()
            .and_then(|i| streams.get_mut(i))
            .ok_or_else(|| internal(format!("no stream {index} on this device")))?;
        if let Some(list) = &mut stream.capture {
            list.push(item);
            return Ok(());
        }
        let item = self.resolve(item)?;
        match stream.tx.push(item) {
            Ok(()) => {
                stream.pushed += 1;
                Ok(())
            }
            Err(PushError::Full(_)) => Err(internal(format!(
                "stream {index}'s ring is full: {STREAM_CAPACITY} items"
            ))),
            Err(PushError::Closed(_)) => Err(internal(format!("stream {index}'s runner is gone"))),
        }
    }

    /// The item to put on a ring now: a copy's buffers resolved to addresses, which fails for a
    /// buffer freed since the item was captured, and an event's generation taken.
    fn resolve(&self, item: Item) -> Result<Item, Error> {
        match item {
            Item::CopyRef { src, dst } => Ok(Item::Copy {
                src: self.range(src.buffer, src.offset, src.bytes)?,
                dst: self.range(dst.buffer, dst.offset, dst.bytes)?,
                len: usize::try_from(src.bytes.0)
                    .map_err(|_| internal(format!("a copy of {} bytes", src.bytes.0)))?,
            }),
            other => Ok(other.resolved()),
        }
    }

    /// Whether every item pushed onto stream `index` has run.
    pub(crate) fn drained(&self, index: usize) -> bool {
        self.streams
            .borrow()
            .get(index)
            .is_none_or(|s| s.shared.completed.load(Ordering::Acquire) >= s.pushed)
    }

    /// The host range `[addr, addr + bytes)` of buffer `handle` from `offset`, if it is in
    /// bounds.
    pub(crate) fn range(&self, handle: u64, offset: Bytes, bytes: Bytes) -> Result<u64, Error> {
        let buffers = self.buffers.borrow();
        let buf = buffers
            .get(&handle)
            .ok_or_else(|| internal(format!("no buffer {handle} on this device")))?;
        let end = offset.0.checked_add(bytes.0);
        if end.is_none_or(|end| end > buf.bytes) {
            return Err(internal(format!(
                "{} bytes at offset {} run past buffer {handle}'s {} bytes",
                bytes.0, offset.0, buf.bytes
            )));
        }
        Ok(buf.addr + offset.0)
    }
}

impl Drop for CpuDevice {
    fn drop(&mut self) {
        // Items still in a runner kept here never run, so only taken runners can touch memory:
        // wait until each has drained or its thread has dropped it.
        for stream in self.streams.get_mut().iter_mut() {
            stream.runner = None;
        }
        // Bounded, as the I/O engines' drop is: a runner that never drains or drops costs the
        // memory, not a hang.
        let deadline = Instant::now() + DROP_WAIT;
        while !self.quiet() {
            if Instant::now() >= deadline {
                return;
            }
            std::hint::spin_loop();
        }
        for (addr, layout) in self.deferred.get_mut().drain(..) {
            // SAFETY: as in `release_deferred`: memory `free` deferred, deallocated once, with
            // every stream drained.
            unsafe { std::alloc::dealloc(addr as *mut u8, layout) };
        }
        for buf in self.buffers.get_mut().values() {
            if let Some(layout) = buf.owned {
                // SAFETY: `owned` is set only for memory `alloc` allocated with exactly this
                // layout, and `free` removes the entry when it deallocates, so each owned
                // buffer still in the table is deallocated here once.
                unsafe { std::alloc::dealloc(buf.addr as *mut u8, layout) };
            }
        }
    }
}

/// A caller's error the RFC-0040 §13 codes do not name, such as a handle from another device
/// or a full ring: class `Internal`, code 0, which no RFC-0040 code uses.
pub(crate) fn internal(message: String) -> Error {
    Error {
        class: ErrorClass::Internal,
        code: ErrorCode(0),
        message,
    }
}
