// SPDX-License-Identifier: MIT OR Apache-2.0

//! The core binary of this fixture. It calls an always-compiled item of `obs::faults`, which rule
//! 10's symbol scan must not report.

#![forbid(unsafe_code)]

fn main() {
    std::process::exit(i32::from(syntropy_obs::faults::effect()));
}
