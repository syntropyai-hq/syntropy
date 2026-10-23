// SPDX-License-Identifier: MIT OR Apache-2.0

//! I/O engines (io_uring, IoRing, IOCP, a `preadv` pool, mmap fallback) with per-core submission
//! rings and registered buffers.
//!
//! Layer 1 in RFC-0001 §4.1, designed in RFC-0021. This crate holds the interface of RFC-0021
//! §2 and ADR-0013 item 1: the producer-facing requests and completions ([`ReadRequest`],
//! [`WriteRequest`], [`ReadCompletion`]), the [`IoEngine`] trait that each engine implements,
//! and the preallocated [`SubmitBatch`] and [`CompletionBuf`] the I/O thread passes to it; and
//! the I/O thread loop of RFC-0021 §3.1 ([`IoLoop`], with an [`IoClient`] per producer). The
//! Windows engines are in the `windows` module, built on Windows only, which holds IoRing and
//! IOCP (RFC-0021 §7). The thread-pool engines of RFC-0021 §4.2 and §5 are in `pool`:
//! `PreadPool` on Linux and Windows, `MacNoCache` on macOS.
//!
//! The [`conformance`] module is the harness every engine's tests run (RFC-0021 Testing, "Every
//! engine returns the right bytes"): it checks any [`IoEngine`] against `std::fs` reads of the
//! same file.
//!
//! The [`conformance`] module is the harness every engine's tests run (RFC-0021 Testing, "Every
//! engine returns the right bytes"): it checks any [`IoEngine`] against `std::fs` reads of the
//! same file.
//!
//! The crate depends on no async runtime: engines run on dedicated I/O threads (ADR-0012
//! items 2 and 6). Its workspace dependencies are `syntropy-core` (RFC-0021 §1), which defines
//! the types the interface shares with other crates, re-exported here, and `syntropy-spsc` for
//! the rings between producers and the I/O thread (ADR-0012 item 4; RFC-0001 §4.2 rule 2).
//!
//! The [`os`] module holds the production clock and entropy sources, `SystemClock` and
//! `OsEntropy` (ADR-0022). They are why this crate depends on `getrandom`, which RFC-0001 §4.2
//! rule 6 allows here only, and on `libc` (Linux and Android) and `windows-sys` (Windows) for the
//! clock calls.
//!
//! `unsafe` is allowed in this crate (RFC-0001 §4.2 rule 8). Every `unsafe` block and impl carries
//! a `// SAFETY:` comment that states its invariant (ADR-0001 item 4).

#![deny(clippy::undocumented_unsafe_blocks)]
#![warn(missing_docs)]

pub mod align;
pub mod batch;
pub mod conformance;
pub mod engine;
pub mod error;
pub mod io_thread;
pub mod os;
#[cfg(any(
    all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ),
    target_os = "macos",
    windows
))]
pub mod pool;
pub mod request;
#[cfg(windows)]
pub mod windows;

pub use batch::{CompletionBuf, Full, PushError, SubmitBatch};
pub use engine::{BufIndex, CancelOutcome, EngineCaps, IoEngine, OpenMode, Wait};
pub use error::{IoError, IoErrorCode};
pub use io_thread::{
    CHUNK_DEFAULT, ConfigError, IoClient, IoLoop, LoopConfig, MAX_COALESCE, Poll, REAP_WAIT_MAX,
};
pub use request::{
    IoDest, IoDestKind, IoOp, IoPriority, IoTicket, ReadCompletion, ReadRequest, SyncMode,
    WriteRequest,
};
pub use syntropy_core::resource::{DioAlign, EngineKind};
pub use syntropy_core::{Bytes, ExtentRef, FileIx, HostRegion, Nanos, StepLayer};
