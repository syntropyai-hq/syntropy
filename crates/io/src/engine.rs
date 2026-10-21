// SPDX-License-Identifier: MIT OR Apache-2.0

//! The engine-facing interface: the [`IoEngine`] trait and the types it takes and returns
//! (RFC-0021 §2; ADR-0013 item 1).
//!
//! One engine instance belongs to one I/O thread, which drives it in the loop of RFC-0021 §3.1.
//! The trait has no async methods and the crate links no async runtime (ADR-0012 item 6).
//!
//! [`EngineKind`], [`DioAlign`], [`FileIx`] and [`HostRegion`] are `syntropy-core` types, because
//! `core`'s `StorageCaps` and `syntropy-kernels` use them too; `syntropy-io` re-exports them.

use std::path::Path;

use crate::batch::{CompletionBuf, SubmitBatch};
use crate::error::IoError;
use crate::request::IoTicket;
use syntropy_core::resource::{DioAlign, EngineKind};
use syntropy_core::{Bytes, FileIx, HostRegion, Nanos};

/// What an engine supports on this host and volume (RFC-0021 §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EngineCaps {
    /// Registered (fixed) buffers are in use, so [`IoDestKind::Fixed`](crate::IoDestKind::Fixed)
    /// destinations avoid per-request page pinning.
    pub fixed_buffers: bool,
    /// Registered file handles are in use.
    pub fixed_files: bool,
    /// The engine accepts [`IoOp::Write`](crate::IoOp::Write).
    pub writes: bool,
    /// [`IoEngine::cancel`] can stop a submitted operation.
    pub cancel: bool,
    /// The most operations the engine holds in flight at once.
    pub max_depth: u32,
    /// `true` for direct I/O; `false` for the flagged buffered mode (RFC-0021 §4.2).
    pub direct: bool,
    /// Alignment every request must meet (RFC-0021 §6), read from the OS per open file. Zero
    /// means the OS reported nothing; §6's 4096-byte floor still applies
    /// (`A = max(4096, offset)`, `M = max(4096, mem, OS page size)`).
    pub dio: DioAlign,
}

/// A registered buffer, as [`IoEngine::register_buffers`] returned it.
///
/// 16 bits, the narrower of the native indexes (io_uring's `buf_index`; Windows I/O rings take
/// 32 bits), so every value is valid on every engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufIndex(pub u16);

/// How [`IoEngine::open`] opens a file (RFC-0021 §4.1, §13).
///
/// The engine chooses direct or buffered access itself and reports it in
/// [`EngineCaps::direct`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpenMode {
    /// Read an existing file: pack and source files.
    Read,
    /// Read and write an existing file: the KV spill file.
    ReadWrite,
    /// Create a file for reading and writing, replacing any file of that name: a pack's
    /// `.tmp` file, a new KV spill file, a session blob's temporary name.
    Create,
}

/// How long [`IoEngine::reap`] may block (RFC-0021 §3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Wait {
    /// Never block: return the completions that are ready.
    None,
    /// Block in the kernel until a completion arrives or this much time passes. The I/O thread
    /// passes the time to its next ring check, at most 100 µs \[target\] (RFC-0021 §3.1).
    Until(Nanos),
}

/// The result of [`IoEngine::cancel`] (RFC-0021 §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CancelOutcome {
    /// No operation with that ticket is queued or in flight.
    NotFound,
    /// The operation was stopped. It still completes, with
    /// [`IoErrorCode::Cancelled`](crate::IoErrorCode::Cancelled).
    Cancelled,
    /// The operation was already in the device; it completes normally and the I/O thread
    /// discards the completion as stale (RFC-0021 §4.1).
    TooLate,
}

/// An I/O engine: batched submission of reads and writes into tier-manager memory, plus a
/// completion poll (RFC-0021 §2; ADR-0013 item 1).
///
/// One instance per I/O thread. Not `Sync`: it is driven only by its thread. `Send`, so the
/// runner can build it and move it onto the thread.
///
/// The memory an engine reads and writes is named only through `unsafe` calls:
/// [`IoEngine::register_buffers`] and the [`IoDest`](crate::IoDest) constructors. Its other
/// methods are safe.
pub trait IoEngine: Send {
    /// Which engine this is.
    fn kind(&self) -> EngineKind;

    /// What it supports on this host and volume.
    fn caps(&self) -> EngineCaps;

    /// Opens a file and, where the engine supports it, registers the handle.
    fn open(&mut self, path: &Path, mode: OpenMode) -> Result<FileIx, IoError>;

    /// Registers host ranges as engine buffers, returning one index per region in order.
    ///
    /// RFC-0021 §2 returns a `SmallVec<[BufIndex; 16]>`; this crate takes no `smallvec`
    /// dependency yet, and registration happens at `Loading`, outside the I/O loop, so a `Vec`
    /// costs nothing on the hot path.
    ///
    /// Code without `unsafe` cannot call it:
    ///
    /// ```compile_fail,E0133
    /// use syntropy_io::{Bytes, HostRegion, IoEngine};
    /// fn register(engine: &mut dyn IoEngine) {
    ///     let any = [HostRegion { addr: 0x1000, len: Bytes(1 << 30) }];
    ///     let _ = engine.register_buffers(&any);
    /// }
    /// ```
    ///
    /// # Safety
    ///
    /// Each region is memory the caller owns, valid for reads and writes over its whole length,
    /// and stays so (not freed, unmapped or given to anything else) until the engine is dropped.
    /// The engine touches a registered region only through [`IoDest::fixed`](crate::IoDest::fixed)
    /// destinations, whose own contracts reserve each range it touches.
    unsafe fn register_buffers(&mut self, regions: &[HostRegion])
    -> Result<Vec<BufIndex>, IoError>;

    /// Never blocks. Returns how many were accepted; the rest stay with the caller.
    ///
    /// The engine accepts operations from the front of the batch, in order, and removes each
    /// one it accepts with [`SubmitBatch::consume`]. Whatever remains in the batch afterwards,
    /// on `Ok` or `Err`, was not submitted.
    ///
    /// An engine transfers at most an operation's extent length, which
    /// [`SubmitBatch::push`] has checked against the destination's
    /// [`len`](crate::IoDest::len). For a [`Fixed`](crate::IoDestKind::Fixed) destination it
    /// also checks that it registered `buf` and that the range lies inside that buffer; an
    /// operation that fails the check completes with an error, and its memory is never touched.
    fn submit(&mut self, batch: &mut SubmitBatch) -> Result<u32, IoError>;

    /// Non-blocking unless `wait` allows it; appends completions.
    ///
    /// Appends at most [`CompletionBuf::remaining`] completions and returns how many it
    /// appended. Completions that do not fit stay with the engine for the next call.
    fn reap(&mut self, out: &mut CompletionBuf, wait: Wait) -> u32;

    /// Cancels a queued or in-flight operation.
    fn cancel(&mut self, ticket: IoTicket) -> CancelOutcome;

    /// Operations in flight and their total length.
    fn in_flight(&self) -> (u32, Bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_engine_is_object_safe_and_send() {
        fn assert_send<T: Send + ?Sized>() {}
        assert_send::<dyn IoEngine>();
        // An engine moves onto its I/O thread behind a box.
        let _ = std::mem::size_of::<Option<Box<dyn IoEngine>>>();
    }
}
