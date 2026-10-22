// SPDX-License-Identifier: MIT OR Apache-2.0

//! A leading allow after the deny turns the crate-wide level back.

#![deny(unsafe_code)]
#![allow(unsafe_code)]

pub mod asm;
pub mod guards;
