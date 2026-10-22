// SPDX-License-Identifier: MIT OR Apache-2.0

//! Code a build script generates, compiled through `include!`.

#![deny(unsafe_code)]

pub mod guards;

include!(concat!(env!("OUT_DIR"), "/gen.rs"));
