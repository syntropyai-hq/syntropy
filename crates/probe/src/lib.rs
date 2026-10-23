// SPDX-License-Identifier: MIT OR Apache-2.0

//! Device enumeration, startup microbenchmarks, live-sample aggregation and drift detection.
//!
//! Layer 2 in RFC-0001 §4.1, designed in RFC-0010. Built so far: E1 host enumeration
//! (RFC-0010 §2.1, §2.2) on Linux in [`linux`], over the filesystem view in [`fs`], and on
//! macOS and Windows in [`macos`] and [`windows`], over the OS calls that
//! `syntropy_io::os` wraps (OQ-PRC-26). [`host`] holds the records all three emit.

#![forbid(unsafe_code)]

pub mod error;
pub mod fs;
pub mod host;
pub mod linux;
pub mod macos;
pub mod windows;

pub use error::EnumError;
pub use fs::{DirFs, HostFs};
