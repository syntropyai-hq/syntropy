// SPDX-License-Identifier: MIT OR Apache-2.0

//! Makes the MSVC linker delay-load `vulkan-1.dll`, so the core binary of this fixture names it
//! in its delay-load directory instead of its import directory. `delayimp.lib` holds the helper
//! that loads a delayed DLL on first call.

fn main() {
    println!("cargo::rustc-link-arg-bins=/DELAYLOAD:vulkan-1.dll");
    println!("cargo::rustc-link-arg-bins=delayimp.lib");
}
