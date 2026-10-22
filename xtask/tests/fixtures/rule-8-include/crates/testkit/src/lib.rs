// SPDX-License-Identifier: MIT OR Apache-2.0

//! A file that is not `.rs`, compiled through `include!`.

#![deny(unsafe_code)]

pub mod guards;

include!("probe.inc");
