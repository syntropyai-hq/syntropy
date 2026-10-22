// SPDX-License-Identifier: MIT OR Apache-2.0

//! The core binary of this fixture. Its release build enables `obs/fault-injection`, and it calls
//! a fault point, so its symbol table holds `syntropy_obs::faults::inject::check` (rule 10).

#![forbid(unsafe_code)]

fn main() {
    let hit = syntropy_obs::faults::inject::check(std::hint::black_box("cli.main.start"));
    std::process::exit(i32::from(hit == 0));
}
