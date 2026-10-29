// SPDX-License-Identifier: MIT OR Apache-2.0

//! Builds a stand-in `libvulkan.1.dylib` that exports `vkCreateInstance`, with the install name
//! `@rpath/libvulkan.1.dylib` the Vulkan loader has, and links this crate against it, so the core
//! binary of this fixture loads a vendor GPU library by name. Only rustc is needed.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    let rustc = env::var_os("RUSTC").expect("cargo sets RUSTC");
    let status = Command::new(rustc)
        .args(["--edition", "2024", "--crate-type", "cdylib", "--crate-name", "vulkan"])
        .args(["-C", "link-arg=-Wl,-install_name,@rpath/libvulkan.1.dylib", "-o"])
        .arg(out.join("libvulkan.dylib"))
        .arg("stub/vulkan.rs")
        .status()
        .expect("rustc runs");
    assert!(status.success(), "the stand-in libvulkan.1.dylib did not build");
    println!("cargo::rerun-if-changed=stub/vulkan.rs");
    println!("cargo::rustc-link-search=native={}", out.display());
    println!("cargo::rustc-link-lib=dylib=vulkan");
}
