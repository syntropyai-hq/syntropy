// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `syntropy` binary (RFC-0001 §4.1 layer 6; RFC-0051). Everything is in the library, so the
//! tests reach it too.

#![forbid(unsafe_code)]

fn main() -> std::process::ExitCode {
    syntropy_cli::run(std::env::args_os())
}
