// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Metal version of the attention hand-off's layouts, `shaders/attn_handoff.metal`, compiles
//! with the Metal compiler, so its `static_assert`s hold: Metal lays the doorbell and the flag
//! out as `include/syntropy/attn_handoff.h` does in C (TIER-E6-T20; RFC-0022 §5.3).
//!
//! The `syntropy-kernels` tests check that the file is what the header prints today. This test
//! builds `msl_compile_harness.swift` with `swiftc` and compiles the file with the default
//! device's Metal compiler. It skips, saying so, only when the machine has no Metal device; a
//! missing `swiftc` skips it outside CI and fails it in CI.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::process::Command;

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

#[test]
fn the_metal_compiler_accepts_the_hand_off_layout() {
    let swiftc = std::env::var("SWIFTC").unwrap_or_else(|_| "swiftc".to_owned());
    let found = Command::new(&swiftc)
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success());
    if found.is_none() {
        assert!(
            std::env::var_os("CI").is_none(),
            "no swiftc found in CI (set SWIFTC to one)"
        );
        eprintln!("skipped: no swiftc found (set SWIFTC to use one)");
        return;
    }
    let dir = std::env::temp_dir().join(format!("syntropy-metal-attn-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let exe = dir.join("msl_compile_harness");
    let built = Command::new(&swiftc)
        .arg("-module-cache-path")
        .arg(dir.join("module-cache"))
        .arg(crate_path("tests/msl_compile_harness.swift"))
        .arg("-o")
        .arg(&exe)
        .output()
        .unwrap();
    assert!(
        built.status.success(),
        "{swiftc} rejected the harness:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let compile = |path: &Path| Command::new(&exe).arg(path).output().unwrap();
    let run = compile(&crate_path("shaders/attn_handoff.metal"));
    if run.status.code() == Some(77) {
        let _ = std::fs::remove_dir_all(&dir);
        eprintln!("skipped: this machine has no Metal device");
        return;
    }
    assert!(
        run.status.success(),
        "the Metal compiler rejected shaders/attn_handoff.metal:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );
    // A negative control: the same file with one layout assertion made false must be rejected,
    // so a compiler that ignored `static_assert` could not pass the check above.
    let wrong = syntropy_kernels_metal::ATTN_HANDOFF_MSL.replace(
        "static_assert(sizeof(syn_attn_flag) == 64,",
        "static_assert(sizeof(syn_attn_flag) == 32,",
    );
    assert_ne!(
        wrong,
        syntropy_kernels_metal::ATTN_HANDOFF_MSL,
        "the control edit applies"
    );
    let wrong_path = dir.join("wrong.metal");
    std::fs::write(&wrong_path, wrong).unwrap();
    let control = compile(&wrong_path);
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(
        control.status.code(),
        Some(1),
        "a false layout assertion must fail to compile:\n{}",
        String::from_utf8_lossy(&control.stderr)
    );
}
