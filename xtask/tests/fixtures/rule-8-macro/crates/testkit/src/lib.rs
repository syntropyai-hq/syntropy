// SPDX-License-Identifier: MIT OR Apache-2.0

//! `#[path]` assembled by a local macro from `#` in another attribute and `path` in `std::path`.

#![deny(unsafe_code)]

pub mod guards;

macro_rules! at {
    ($h:tt $d:tt std :: $k:ident $v:literal ; $($rest:tt)*) => { $h [$k = $v] $($rest)* };
}

at!(#[doc = "a probe"] std::path "../outside.rs" ; pub mod outside_probe;);
