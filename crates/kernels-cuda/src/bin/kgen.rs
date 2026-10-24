// SPDX-License-Identifier: MIT OR Apache-2.0

//! `syntropy-kgen [--check] [--root <dir>]`: writes the CUDA provider's generated files, or with
//! `--check` fails when any is stale (RFC-0040 §10.2 rule 2). `cargo xtask kgen` runs it.
//!
//! `--root` is the `syntropy-kernels-cuda` crate directory; by default, the one this binary was
//! built from.

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut check = false;
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--check" => check = true,
            "--root" => match args.next() {
                Some(dir) => root = PathBuf::from(dir),
                None => return usage("--root needs a directory"),
            },
            "-h" | "--help" => {
                println!("usage: syntropy-kgen [--check] [--root <dir>]");
                return ExitCode::SUCCESS;
            }
            other => return usage(&format!("unknown argument `{other}`")),
        }
    }
    match syntropy_kernels_cuda::kgen::run(&root, check) {
        Ok((ok, report)) => {
            eprint!("{report}");
            if ok {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(error) => {
            eprintln!("kgen: FAIL: {error}");
            ExitCode::FAILURE
        }
    }
}

fn usage(message: &str) -> ExitCode {
    eprintln!("kgen: {message}\nusage: syntropy-kgen [--check] [--root <dir>]");
    ExitCode::from(2)
}
