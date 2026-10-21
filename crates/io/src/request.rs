// SPDX-License-Identifier: MIT OR Apache-2.0

//! Producer-facing requests and completions (RFC-0021 §2; RFC-0001 §8.3).
//!
//! Dispatchers, housekeeping and the loader post [`ReadRequest`]s and [`WriteRequest`]s on
//! their SPSC rings to a volume's I/O thread, which returns one [`ReadCompletion`] per request
//! on the producer's own ring (RFC-0021 §1). The rings move these values without allocating.
//! A request is neither `Copy` nor `Clone`, because its [`IoDest`] is not: once submitted, it
//! cannot be submitted again.
//!
//! [`ExtentRef`] and [`StepLayer`] are the supporting types RFC-0020 §3 defines. They live here
//! rather than in `syntropy-tier`, because a [`ReadRequest`] carries them and this crate sits a
//! layer below `tier` (RFC-0001 §4.1, §4.2 rule 2).

use crate::engine::{BufIndex, FileIx};
use crate::error::IoError;
use crate::units::{Bytes, Nanos};

/// A producer-unique request identifier, echoed in the request's completion (RFC-0021 §2).
///
/// RFC-0001 §5.5 and RFC-0020 §3 record it in residency (`CopyState::Inflight`,
/// `HomeState::Loading`) while a read is in flight.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IoTicket(pub u64);

/// A byte range of one file on a volume (RFC-0020 §3).
///
/// `offset` and `len` are multiples of the volume's [`DioAlign`](crate::DioAlign), as RFC-0021
/// §6 requires of every request. RFC-0020 §3 writes `file` and `len` as bare integers; here they
/// are [`FileIx`] and [`Bytes`], because interfaces never take bare integers for sizes (RFC-0001
/// §5.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExtentRef {
    /// The file, as the I/O thread opened it.
    pub file: FileIx,
    /// Offset of the first byte in the file.
    pub offset: u64,
    /// Length of the range.
    pub len: Bytes,
}

/// A point in a device's execution: the step number and the layer within it (RFC-0020 §3).
///
/// Consumers publish their progress as one `AtomicU64` per device, `(step << 16) | layer`
/// ([`StepLayer::encode`]). A `StepLayer` has passed when the published word is greater than
/// or equal to its encoding ([`StepLayer::has_passed`]). The derived order, step first and then
/// layer, is the order of the encodings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StepLayer {
    /// The step number. The encoding keeps its low 48 bits.
    pub step: u64,
    /// The layer within the step.
    pub layer: u16,
}

impl StepLayer {
    /// Steps at or above this value do not fit the 48 bits the encoding gives them.
    pub const STEP_LIMIT: u64 = 1 << 48;

    /// The progress-word encoding, `(step << 16) | layer` (RFC-0020 §3).
    ///
    /// `step` must be below [`StepLayer::STEP_LIMIT`]; debug builds assert it.
    #[must_use]
    pub const fn encode(self) -> u64 {
        debug_assert!(self.step < Self::STEP_LIMIT);
        (self.step << 16) | self.layer as u64
    }

    /// Whether a consumer whose published progress word is `progress` has reached this point.
    #[must_use]
    pub const fn has_passed(self, progress: u64) -> bool {
        progress >= self.encode()
    }
}

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

/// Where a read puts its bytes, or where a write takes them from (RFC-0021 §2).
///
/// An engine reads and writes the memory an `IoDest` names, so whoever names it must vouch for
/// that memory. The field is private and the constructors, [`IoDest::fixed`] and
/// [`IoDest::raw`], are `unsafe`. The obligation therefore sits with the tier manager, which owns
/// every destination buffer (RFC-0021 §1) and may use `unsafe` (RFC-0001 §4.2 rule 8; ADR-0001
/// item 4). Other crates, including those under `forbid(unsafe_code)`, only forward the values
/// it hands them. An `IoDest` is neither `Copy` nor `Clone`, so a request that has been
/// submitted cannot be submitted again from a stale copy.
///
/// Code without `unsafe` cannot make one:
///
/// ```compile_fail,E0133
/// let dest = syntropy_io::IoDest::raw(0x1000);
/// ```
///
/// ```compile_fail,E0451
/// use syntropy_io::{IoDest, IoDestKind};
/// let dest = IoDest { kind: IoDestKind::Raw { addr: 0x1000 } };
/// ```
///
/// ```compile_fail,E0599
/// fn twice(dest: &syntropy_io::IoDest) -> syntropy_io::IoDest {
///     dest.clone()
/// }
/// ```
///
/// With `unsafe`, and a contract the caller keeps, it compiles:
///
/// ```
/// use syntropy_io::{IoDest, IoDestKind};
/// // SAFETY: this destination is never put in a request, so no engine accesses the address.
/// let dest = unsafe { IoDest::raw(0x1000) };
/// assert_eq!(dest.kind(), IoDestKind::Raw { addr: 0x1000 });
/// ```
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct IoDest {
    kind: IoDestKind,
}

impl IoDest {
    /// A slot in a registered buffer: `offset` bytes into buffer `buf`.
    ///
    /// The offset should also be a multiple of the volume's
    /// [`DioAlign::mem`](crate::DioAlign::mem); the I/O thread rejects a misaligned request
    /// before submission (RFC-0021 §6). That is a correctness check, not part of the safety
    /// contract.
    ///
    /// # Safety
    ///
    /// From the submission of the request that carries this destination until its completion
    /// has been reaped, where `len` is that request's extent length:
    /// - `buf` was returned by [`IoEngine::register_buffers`](crate::IoEngine::register_buffers)
    ///   on the engine the request is submitted to, and `offset .. offset + len` lies inside
    ///   that buffer;
    /// - nothing else reads or writes that range: no other request, thread or device.
    #[must_use]
    pub const unsafe fn fixed(buf: BufIndex, offset: u64) -> Self {
        Self {
            kind: IoDestKind::Fixed { buf, offset },
        }
    }

    /// Any memory the tier manager owns, starting at `addr`.
    ///
    /// The address should also be aligned to the volume's
    /// [`DioAlign::mem`](crate::DioAlign::mem), and to at least 4096 bytes; the I/O thread rejects
    /// a misaligned request before submission (RFC-0021 §6). That is a correctness check, not part
    /// of the safety contract.
    ///
    /// # Safety
    ///
    /// From the submission of the request that carries this destination until its completion
    /// has been reaped, where `len` is that request's extent length:
    /// - `addr .. addr + len` is one allocation the caller owns, valid for writes when the request
    ///   is a read and for reads when it is a write;
    /// - nothing else reads or writes that range: no other request, thread or device.
    #[must_use]
    pub const unsafe fn raw(addr: u64) -> Self {
        Self {
            kind: IoDestKind::Raw { addr },
        }
    }

    /// What this destination names.
    #[must_use]
    pub const fn kind(&self) -> IoDestKind {
        self.kind
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
}

#[cfg(test)]
mod tests {
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
    fn step_layer_encoding() {
        let at = StepLayer { step: 3, layer: 7 };
        assert_eq!(at.encode(), (3 << 16) | 7);
        assert!(!at.has_passed(0));
        assert!(!at.has_passed(StepLayer { step: 3, layer: 6 }.encode()));
        assert!(at.has_passed(at.encode()));
        assert!(at.has_passed(StepLayer { step: 3, layer: 8 }.encode()));
        assert!(at.has_passed(StepLayer { step: 4, layer: 0 }.encode()));

        let last = StepLayer {
            step: StepLayer::STEP_LIMIT - 1,
            layer: u16::MAX,
        };
        assert_eq!(last.encode(), u64::MAX);
    }

    #[test]
    fn step_layer_order_matches_encoding_order() {
        let points = [
            StepLayer { step: 0, layer: 0 },
            StepLayer {
                step: 0,
                layer: u16::MAX,
            },
            StepLayer { step: 1, layer: 0 },
            StepLayer { step: 1, layer: 1 },
            StepLayer {
                step: 1 << 40,
                layer: 2,
            },
            StepLayer {
                step: StepLayer::STEP_LIMIT - 1,
                layer: 0,
            },
        ];
        for a in points {
            for b in points {
                assert_eq!(a.cmp(&b), a.encode().cmp(&b.encode()), "{a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn io_op_accessors() {
        let extent = ExtentRef {
            file: FileIx(2),
            offset: 4096,
            len: Bytes(8192),
        };
        // SAFETY: neither request is submitted to an engine, so no memory is accessed through
        // these destinations.
        let (fixed, raw) = unsafe { (IoDest::fixed(BufIndex(1), 0), IoDest::raw(0x1000)) };
        assert_eq!(
            fixed.kind(),
            IoDestKind::Fixed {
                buf: BufIndex(1),
                offset: 0
            }
        );
        assert_eq!(raw.kind(), IoDestKind::Raw { addr: 0x1000 });
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
    }
}
