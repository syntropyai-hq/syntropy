// SPDX-License-Identifier: MIT OR Apache-2.0

//! Builds a stand-in `libcufft.so` that exports `cufftPlan1d`, and links this crate against it, so
//! the core binary of this fixture needs a vendor GPU library by name. Only rustc is needed.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    let rustc = env::var_os("RUSTC").expect("cargo sets RUSTC");
    let status = Command::new(rustc)
        .args(["--edition", "2024", "--crate-type", "cdylib", "--crate-name", "cufft", "-o"])
        .arg(out.join("libcufft.so"))
        .arg("stub/cufft.rs")
        .status()
        .expect("rustc runs");
    assert!(status.success(), "the stand-in libcufft.so did not build");
    println!("cargo::rerun-if-changed=stub/cufft.rs");
    println!("cargo::rustc-link-search=native={}", out.display());
    println!("cargo::rustc-link-lib=dylib=cufft");
}
