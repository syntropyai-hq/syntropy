// SPDX-License-Identifier: MIT OR Apache-2.0

//! Producer-facing requests and completions (RFC-0021 §2; RFC-0001 §8.3).
//!
//! Dispatchers, housekeeping and the loader post [`ReadRequest`]s and [`WriteRequest`]s on
//! their SPSC rings to a volume's I/O thread, which returns one [`ReadCompletion`] per request
//! on the producer's own ring (RFC-0021 §1). Every type here is `Copy`, so the rings move them
//! without allocating.
//!
//! [`ExtentRef`] and [`StepLayer`] are the supporting types RFC-0020 §3 defines. They live here
//! rather than in `syntropy-tier`, because a [`ReadRequest`] carries them and this crate sits a
//! layer below `tier` (RFC-0001 §4.1, §4.2 rule 2).

use crate::engine::BufIndex;
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
/// §6 requires of every request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExtentRef {
    /// Index of the file among the files the I/O thread opened.
    pub file: u16,
    /// Offset of the first byte in the file.
    pub offset: u64,
    /// Length in bytes.
    pub len: u64,
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

/// Where a read puts its bytes, or where a write takes them from (RFC-0021 §2).
///
/// The tier manager owns every buffer an `IoDest` names (RFC-0021 §1). It guarantees that the
/// memory is valid, aligned to the volume's [`DioAlign`](crate::DioAlign) (RFC-0021 §6), and
/// neither read nor written by anyone else from submission until the request's completion is
/// reaped. Engines write into the memory on the strength of that guarantee.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IoDest {
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
            file: 2,
            offset: 4096,
            len: 8192,
        };
        let read = IoOp::Read(ReadRequest {
            ticket: IoTicket(11),
            extent,
            dest: IoDest::Fixed {
                buf: BufIndex(1),
                offset: 0,
            },
            priority: IoPriority::PrefetchNear,
            useful_until: Some(StepLayer { step: 5, layer: 2 }),
        });
        let write = IoOp::Write(WriteRequest {
            ticket: IoTicket(12),
            extent,
            src: IoDest::Raw { addr: 0x1000 },
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
