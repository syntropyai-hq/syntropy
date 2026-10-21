// SPDX-License-Identifier: MIT OR Apache-2.0

//! Producer-facing requests and completions (RFC-0021 §2; RFC-0001 §8.3).
//!
//! Dispatchers, housekeeping and the loader post [`ReadRequest`]s and [`WriteRequest`]s on
//! their SPSC rings to a volume's I/O thread, which returns one [`ReadCompletion`] per request
//! on the producer's own ring (RFC-0021 §1). The rings move these values without allocating.
//! A request is neither `Copy` nor `Clone`, because its [`IoDest`] is not: once submitted, it
//! cannot be submitted again.
//!
//! A request carries an [`ExtentRef`] and, for a prefetch, a [`StepLayer`]: RFC-0020 §3's
//! supporting types, defined in `syntropy-core` so that `tier`, a layer above this crate, can use
//! and serialise them too (RFC-0001 §4.1, §4.2 rule 2).

use syntropy_core::{Bytes, ExtentRef, Nanos, StepLayer};

use crate::engine::BufIndex;
use crate::error::IoError;

/// A producer-unique request identifier, echoed in the request's completion (RFC-0021 §2).
///
/// RFC-0001 §5.5 and RFC-0020 §3 record it in residency (`CopyState::Inflight`,
/// `HomeState::Loading`) while a read is in flight.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IoTicket(pub u64);

/// What an [`IoDest`] names: the two destination kinds of RFC-0021 §2.
///
/// Plain data, returned by [`IoDest::kind`] for engines to read. Turning it into an `IoDest`
/// takes one of the `unsafe` constructors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IoDestKind {
    /// A slot in an `IO_FIXED` arena (RFC-0020 §2): engine buffer index plus offset.
    Fixed {
        /// The registered buffer, as [`IoEngine::register_buffers`](crate::IoEngine::register_buffers)
        /// returned it.
        buf: BufIndex,
        /// Byte offset inside that buffer.
        offset: u64,
    },
    /// Any tier-manager memory, aligned per RFC-0021 §6; used for bulk `Loading` reads.
    Raw {
        /// Address of the first byte.
        addr: u64,
    },
}

/// Where a read puts its bytes, or where a write takes them from (RFC-0021 §2), and how many
/// bytes that memory holds.
///
/// An engine reads and writes the memory an `IoDest` names, so whoever names it must vouch for
/// that memory. The fields are private and the constructors, [`IoDest::fixed`] and
/// [`IoDest::raw`], are `unsafe`. The obligation therefore sits with the tier manager, which owns
/// every destination buffer (RFC-0021 §1) and may use `unsafe` (RFC-0001 §4.2 rule 8; ADR-0001
/// item 4). Other crates, including those under `forbid(unsafe_code)`, only forward the values
/// it hands them.
///
/// The length is part of the destination, so the caller vouches for a range it chose. A request
/// can name a longer extent, but [`SubmitBatch::push`](crate::SubmitBatch::push), the only way
/// to reach an engine, refuses an operation whose extent is longer than its destination. An
/// `IoDest` is neither `Copy` nor `Clone`, so a request that has been submitted cannot be
/// submitted again from a stale copy.
///
/// Code without `unsafe` cannot make one:
///
/// ```compile_fail,E0133
/// use syntropy_io::{Bytes, IoDest};
/// let dest = IoDest::raw(0x1000, Bytes(4096));
/// ```
///
/// ```compile_fail,E0451
/// use syntropy_io::{Bytes, IoDest, IoDestKind};
/// let dest = IoDest { kind: IoDestKind::Raw { addr: 0x1000 }, len: Bytes(4096) };
/// ```
///
/// ```compile_fail,E0277
/// use syntropy_io::IoDest;
/// fn twice(dest: &IoDest) -> IoDest {
///     IoDest::clone(dest)
/// }
/// ```
///
/// With `unsafe`, and a contract the caller keeps, it compiles:
///
/// ```
/// use syntropy_io::{Bytes, IoDest, IoDestKind};
/// let mut page = vec![0u8; 4096];
/// let addr = page.as_mut_ptr() as u64;
/// // SAFETY: `page` is one allocation of 4096 bytes that this code owns. It outlives `dest`,
/// // which is never submitted, and nothing reads or writes it while `dest` exists.
/// let dest = unsafe { IoDest::raw(addr, Bytes(4096)) };
/// assert_eq!(dest.kind(), IoDestKind::Raw { addr });
/// assert_eq!(dest.len(), Bytes(4096));
/// ```
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct IoDest {
    kind: IoDestKind,
    len: Bytes,
}

impl IoDest {
    /// `len` bytes of a registered buffer, starting `offset` bytes into buffer `buf`.
    ///
    /// The offset should also be a multiple of the volume's
    /// [`DioAlign::mem`](crate::DioAlign::mem); the I/O thread rejects a misaligned request
    /// before submission (RFC-0021 §6). That is a correctness check, not part of the safety
    /// contract.
    ///
    /// # Safety
    ///
    /// From this call until the value is dropped without being submitted, or, once an engine
    /// accepts a request that carries it, until that request's completion has been reaped:
    /// - on every engine in this process that has a buffer registered at index `buf`,
    ///   `offset .. offset + len` lies inside that buffer. Code without `unsafe` chooses which
    ///   engine a request reaches, so this has to hold on each of them; an engine never
    ///   touches memory for a `buf` it has not registered (see [`IoEngine::submit`](crate::IoEngine::submit));
    /// - nothing else reads or writes the memory that range names on those engines: no other
    ///   destination, thread or device.
    ///
    /// The caller cannot see when a forwarding crate submits or drops the value, so in practice
    /// it keeps the range reserved until the request's completion reaches it. A range whose
    /// destination never comes back stays reserved: leaking a slot is safe, reusing it early is
    /// not.
    #[must_use]
    pub const unsafe fn fixed(buf: BufIndex, offset: u64, len: Bytes) -> Self {
        Self {
            kind: IoDestKind::Fixed { buf, offset },
            len,
        }
    }

    /// `len` bytes of memory the tier manager owns, starting at `addr`.
    ///
    /// The address should also be aligned to the volume's
    /// [`DioAlign::mem`](crate::DioAlign::mem), and to at least 4096 bytes; the I/O thread rejects
    /// a misaligned request before submission (RFC-0021 §6). That is a correctness check, not part
    /// of the safety contract.
    ///
    /// # Safety
    ///
    /// From this call until the value is dropped without being submitted, or, once an engine
    /// accepts a request that carries it, until that request's completion has been reaped:
    /// - `addr .. addr + len` lies inside one allocation the caller owns, valid for reads and
    ///   writes;
    /// - nothing else reads or writes that range: no other destination, thread or device.
    ///
    /// As for [`IoDest::fixed`], the caller in practice keeps the range reserved until the
    /// request's completion reaches it, and a range whose destination never comes back stays
    /// reserved.
    #[must_use]
    pub const unsafe fn raw(addr: u64, len: Bytes) -> Self {
        Self {
            kind: IoDestKind::Raw { addr },
            len,
        }
    }

    /// What this destination names.
    #[must_use]
    pub const fn kind(&self) -> IoDestKind {
        self.kind
    }

    /// How many bytes this destination holds: the most a request carrying it may transfer.
    #[must_use]
    pub const fn len(&self) -> Bytes {
        self.len
    }
}

/// The class of a request, which decides its place in the I/O thread's queues (RFC-0021 §3.2).
///
/// The derived order is the submission order: `Demand` sorts first and is always submitted
/// first, and `Background` sorts last.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IoPriority {
    /// The current step waits for it (glossary: demand read).
    Demand,
    /// Predicted for the next layer or the next step.
    PrefetchNear,
    /// `Loading` and pack building.
    Load,
    /// Predicted further ahead; first to be dropped.
    PrefetchFar,
    /// KV spill, session save, scrub.
    Background,
}

impl IoPriority {
    /// Every class, in submission order: one queue each in the I/O thread (RFC-0021 §3.1).
    pub const ALL: [IoPriority; 5] = [
        IoPriority::Demand,
        IoPriority::PrefetchNear,
        IoPriority::Load,
        IoPriority::PrefetchFar,
        IoPriority::Background,
    ];
}

/// A read from a volume into tier-manager memory (RFC-0021 §2; RFC-0001 §8.3).
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ReadRequest {
    /// Producer-unique; echoed in the completion.
    pub ticket: IoTicket,
    /// The bytes to read, aligned per RFC-0021 §6.
    pub extent: ExtentRef,
    /// Where the bytes go.
    pub dest: IoDest,
    /// The request's class.
    pub priority: IoPriority,
    /// Prefetch only: drop the request if it has not been submitted when the consumer
    /// reaches this point (RFC-0023 sets it to the target layer of the target step).
    pub useful_until: Option<StepLayer>,
}

/// How a write is made durable (RFC-0021 §2, §13).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SyncMode {
    /// No sync: KV spill pages, which are discarded on restart.
    None,
    /// Data sync (`fdatasync`): pack units and session blobs before their rename.
    DataSync,
    /// Full sync (`fsync`, `FlushFileBuffers`).
    FullSync,
}

/// A write from tier-manager memory to a volume (RFC-0021 §2, §13).
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct WriteRequest {
    /// Producer-unique; echoed in the completion.
    pub ticket: IoTicket,
    /// The bytes to write, aligned per RFC-0021 §6.
    pub extent: ExtentRef,
    /// Where the bytes come from.
    pub src: IoDest,
    /// The request's class: `Load` for pack building, `Background` otherwise (RFC-0021 §13).
    pub priority: IoPriority,
    /// How the write is made durable.
    pub sync: SyncMode,
}

/// The result of one request, returned on the producer's ring (RFC-0021 §2; RFC-0001 §8.3).
///
/// RFC-0021 §2 defines no separate write completion, so a write also completes with a
/// `ReadCompletion`. `result` is the number of bytes transferred. From an engine, fewer bytes
/// than the extent's `len` is a short read or write, which the I/O thread resubmits
/// (RFC-0021 §16); the I/O thread posts a completion only once the whole extent is done or has
/// failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReadCompletion {
    /// The request's ticket.
    pub ticket: IoTicket,
    /// Bytes transferred, or why the request failed.
    pub result: Result<Bytes, IoError>,
    /// Time from submission to completion.
    pub latency: Nanos,
}

/// One operation in a [`SubmitBatch`](crate::SubmitBatch): a read or a write.
///
/// The I/O thread splits each request into chunks (RFC-0021 §3.3) and hands each chunk to the
/// engine as one `IoOp`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub enum IoOp {
    /// A read.
    Read(ReadRequest),
    /// A write.
    Write(WriteRequest),
}

impl IoOp {
    /// The operation's ticket.
    #[must_use]
    pub const fn ticket(&self) -> IoTicket {
        match self {
            IoOp::Read(r) => r.ticket,
            IoOp::Write(w) => w.ticket,
        }
    }

    /// The operation's extent.
    #[must_use]
    pub const fn extent(&self) -> ExtentRef {
        match self {
            IoOp::Read(r) => r.extent,
            IoOp::Write(w) => w.extent,
        }
    }

    /// The operation's class.
    #[must_use]
    pub const fn priority(&self) -> IoPriority {
        match self {
            IoOp::Read(r) => r.priority,
            IoOp::Write(w) => w.priority,
        }
    }

    /// The memory the operation touches: a read's destination or a write's source.
    #[must_use]
    pub const fn dest(&self) -> &IoDest {
        match self {
            IoOp::Read(r) => &r.dest,
            IoOp::Write(w) => &w.src,
        }
    }
}

#[cfg(test)]
mod tests {
    use syntropy_core::FileIx;

    use super::*;

    #[test]
    fn priority_order_is_submission_order() {
        let mut sorted = IoPriority::ALL;
        sorted.sort();
        assert_eq!(sorted, IoPriority::ALL);
        assert_eq!(IoPriority::ALL[0], IoPriority::Demand);
        assert_eq!(IoPriority::ALL[4], IoPriority::Background);
        // The cast gives each class its queue index.
        for (i, p) in IoPriority::ALL.iter().enumerate() {
            assert_eq!(*p as usize, i);
        }
    }

    #[test]
    fn io_op_accessors() {
        let extent = ExtentRef {
            file: FileIx(2),
            offset: 4096,
            len: Bytes(8192),
        };
        let mut page = [0u8; 8192];
        let addr = page.as_mut_ptr() as u64;
        // SAFETY: no engine in this test process registers buffers, so the `fixed` destination
        // names no memory. `page` is one 8192-byte allocation this test owns; it outlives `raw`,
        // which is never submitted, and nothing reads or writes it while `raw` exists.
        let (fixed, raw) = unsafe {
            (
                IoDest::fixed(BufIndex(1), 0, Bytes(8192)),
                IoDest::raw(addr, Bytes(8192)),
            )
        };
        assert_eq!(
            fixed.kind(),
            IoDestKind::Fixed {
                buf: BufIndex(1),
                offset: 0
            }
        );
        assert_eq!(raw.kind(), IoDestKind::Raw { addr });
        assert_eq!(fixed.len(), Bytes(8192));
        let read = IoOp::Read(ReadRequest {
            ticket: IoTicket(11),
            extent,
            dest: fixed,
            priority: IoPriority::PrefetchNear,
            useful_until: Some(StepLayer { step: 5, layer: 2 }),
        });
        let write = IoOp::Write(WriteRequest {
            ticket: IoTicket(12),
            extent,
            src: raw,
            priority: IoPriority::Background,
            sync: SyncMode::DataSync,
        });
        assert_eq!(read.ticket(), IoTicket(11));
        assert_eq!(write.ticket(), IoTicket(12));
        assert_eq!(read.extent(), extent);
        assert_eq!(write.extent(), extent);
        assert_eq!(read.priority(), IoPriority::PrefetchNear);
        assert_eq!(write.priority(), IoPriority::Background);
        assert_eq!(
            read.dest().kind(),
            IoDestKind::Fixed {
                buf: BufIndex(1),
                offset: 0
            }
        );
        assert_eq!(write.dest().kind(), IoDestKind::Raw { addr });
    }
}
