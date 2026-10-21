// SPDX-License-Identifier: MIT OR Apache-2.0

//! I/O engines (io_uring, IoRing, IOCP, a `preadv` pool, mmap fallback) with per-core submission
//! rings and registered buffers.
//!
//! Layer 1 in RFC-0001 §4.1, designed in RFC-0021. This crate holds the interface of RFC-0021
//! §2 and ADR-0013 item 1: the producer-facing requests and completions ([`ReadRequest`],
//! [`WriteRequest`], [`ReadCompletion`]), the [`IoEngine`] trait that each engine implements,
//! and the preallocated [`SubmitBatch`] and [`CompletionBuf`] the I/O thread passes to it. No
//! engine and no I/O thread loop exists yet.
//!
//! The crate depends on no async runtime: engines run on dedicated I/O threads (ADR-0012
//! items 2 and 6). Its only dependency is `syntropy-core` (RFC-0021 §1), which defines the
//! types the interface shares with other crates; they are re-exported here.
//!
//! `unsafe` is allowed in this crate (RFC-0001 §4.2 rule 8). Every `unsafe` block and impl carries
//! a `// SAFETY:` comment that states its invariant (ADR-0001 item 4).

#![deny(clippy::undocumented_unsafe_blocks)]
#![warn(missing_docs)]

pub mod batch;
pub mod engine;
pub mod error;
pub mod request;

pub use batch::{CompletionBuf, Full, PushError, SubmitBatch};
pub use engine::{BufIndex, CancelOutcome, EngineCaps, IoEngine, OpenMode, Wait};
pub use error::{IoError, IoErrorCode};
pub use request::{
    IoDest, IoDestKind, IoOp, IoPriority, IoTicket, ReadCompletion, ReadRequest, SyncMode,
    WriteRequest,
};
pub use syntropy_core::resource::{DioAlign, EngineKind};
pub use syntropy_core::{Bytes, ExtentRef, FileIx, HostRegion, Nanos, StepLayer};
