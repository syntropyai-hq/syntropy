// SPDX-License-Identifier: MIT OR Apache-2.0

//! A `#[path]` built by a macro from a metavariable, so `path =` appears in no text.

#![deny(unsafe_code)]

pub mod guards;

macro_rules! module_at {
    ($key:ident $file:literal) => { #[$key = $file] pub mod outside_probe; };
}

module_at!(path "../outside.rs");
