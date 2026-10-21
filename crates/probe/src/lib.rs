// SPDX-License-Identifier: MIT OR Apache-2.0

//! Device enumeration, startup microbenchmarks, live-sample aggregation and drift detection.
//!
//! Layer 2 in RFC-0001 §4.1, designed in RFC-0010. Built so far: E1 host enumeration on
//! Linux (RFC-0010 §2.1, §2.2), in [`linux`], over the filesystem view in [`fs`].

#![forbid(unsafe_code)]

pub mod error;
pub mod fs;
pub mod linux;

pub use error::EnumError;
pub use fs::{DirFs, HostFs};
