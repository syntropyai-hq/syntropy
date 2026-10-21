// SPDX-License-Identifier: MIT OR Apache-2.0

//! Opens store roots, reads local aliases through each read-only root, and exits.
//!
//! This is the program the read-only syscall trace runs (RFC-0041 §4.1):
//!
//! ```text
//! cargo build -p syntropy-store --example open_roots
//! strace -f -e trace=%file,write,openat target/debug/examples/open_roots \
//!     --writable RW_DIR --readonly RO_DIR --alias demo
//! ```
//!
//! In the trace, every path under `RO_DIR` must be opened read-only, and no create, write,
//! rename or delete may name one.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use syntropy_store::{Name, StoreRoots};

const USAGE: &str = "usage: open_roots [--writable DIR]... [--readonly DIR]... [--alias NAME]...";

fn main() -> ExitCode {
    let mut writable = Vec::new();
    let mut readonly = Vec::new();
    let mut aliases = Vec::new();
    let mut args = env::args_os().skip(1);
    while let Some(flag) = args.next() {
        let Some(value) = args.next() else {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        };
        match flag.to_str() {
            Some("--writable") => writable.push(PathBuf::from(value)),
            Some("--readonly") => readonly.push(PathBuf::from(value)),
            Some("--alias") => match value.into_string().map(Name::new) {
                Ok(Ok(name)) => aliases.push(name),
                _ => {
                    eprintln!("--alias takes a store path name");
                    return ExitCode::from(2);
                }
            },
            _ => {
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            }
        }
    }

    let roots = match StoreRoots::open(&writable, &readonly) {
        Ok(roots) => roots,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    for root in roots.writable() {
        println!("writable {}", root.path().display());
    }
    for root in roots.readonly() {
        println!("read-only {}", root.path().display());
        for alias in &aliases {
            let path = root.layout().local_ref(alias);
            match fs::read(&path) {
                Ok(bytes) => println!("  read {} ({} bytes)", path.display(), bytes.len()),
                Err(e) => {
                    eprintln!("error: cannot read {}: {e}", path.display());
                    return ExitCode::FAILURE;
                }
            }
        }
    }
    ExitCode::SUCCESS
}
