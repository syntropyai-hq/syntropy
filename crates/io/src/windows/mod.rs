// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Windows engines (RFC-0021 §7; ADR-0013 item 4): [`IoRingEngine`] on Windows 11 and
//! later, and [`IocpEngine`] wherever IoRing is missing or lacks an operation.
//!
//! Both open files with `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`. A file system that
//! refuses unbuffered opens (`ERROR_INVALID_PARAMETER`) gets a buffered handle, and the engine
//! reports [`EngineCaps::direct`](crate::EngineCaps::direct) `false`. Each engine reads the
//! file's alignment with [`align::query`](crate::align::query) when it opens it.
//!
//! IoRing's functions are resolved at run time from `api-ms-win-core-ioring-l1-1-0.dll`, and its
//! write and flush from `kernel32.dll`, so a binary that links this crate still starts on a
//! Windows without them: there [`IoRingEngine::new`] returns
//! [`IoErrorCode::EngineUnavailable`](crate::IoErrorCode::EngineUnavailable) and the IOCP engine
//! is used.
//!
//! `cancel` on either engine asks the OS to stop the operation and returns
//! [`CancelOutcome::TooLate`](crate::CancelOutcome::TooLate): the operation is already in the
//! kernel, so it completes either normally or with `IO_CANCELLED`, and the I/O thread discards
//! that completion as stale either way (RFC-0021 §4.1).

mod common;
mod iocp;
mod ioring;

pub use iocp::{IOCP_THREADS_DEFAULT, IocpEngine, IocpOptions};
pub use ioring::{IoRingEngine, IoRingOptions, IoRingSupport, ioring_support};
