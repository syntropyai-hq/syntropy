// SPDX-License-Identifier: MIT OR Apache-2.0

//! Assembles the SME2 unit, `src/sme2/*.S`, with the platform assembler (ADR-0402 item 3;
//! RFC-0040 §7.3).
//!
//! The unit is built only for AArch64 macOS, Linux and Android, whose assemblers take the same
//! GNU-style syntax. There the build sets `cfg(syntropy_sme2)`, under which `src/sme2.rs`
//! declares and calls the routines. Every other target (x86-64, Windows on Arm, ...) builds no
//! assembly and has no routine to call; detection reports SME2 absent there anyway.

use std::env;

const SOURCES: [&str; 2] = ["src/sme2/tiles.S", "src/sme2/harness.S"];

fn main() {
    println!("cargo::rustc-check-cfg=cfg(syntropy_sme2)");
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/sme2/common.h");
    for s in SOURCES {
        println!("cargo::rerun-if-changed={s}");
    }

    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if arch != "aarch64" || !matches!(os.as_str(), "macos" | "linux" | "android") {
        return;
    }
    cc::Build::new()
        .files(SOURCES)
        .compile("syntropy_kernels_cpu_sme2");
    println!("cargo::rustc-cfg=syntropy_sme2");
}
