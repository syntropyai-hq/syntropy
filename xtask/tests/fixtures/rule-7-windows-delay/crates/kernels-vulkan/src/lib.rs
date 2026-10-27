// SPDX-License-Identifier: MIT OR Apache-2.0

//! A provider crate that imports a stand-in for the Vulkan loader, `vulkan-1.dll`. The
//! `raw-dylib` kind makes rustc write the import itself, and the core binary's build script tells
//! the linker to delay-load it.

#[link(name = "vulkan-1", kind = "raw-dylib")]
unsafe extern "C" {
    fn vkCreateInstance(info: *const u8, allocator: *const u8, instance: *mut usize) -> i32;
}

/// Calls `vkCreateInstance`. The fixture binary is built and read, never run.
pub fn init() -> i32 {
    let mut instance = 0;
    // SAFETY: never executed; the fixture only needs the import of `vkCreateInstance`.
    unsafe { vkCreateInstance(std::ptr::null(), std::ptr::null(), &mut instance) }
}
