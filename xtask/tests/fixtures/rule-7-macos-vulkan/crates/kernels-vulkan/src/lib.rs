// SPDX-License-Identifier: MIT OR Apache-2.0

//! A provider crate that links a stand-in for the Vulkan loader.

unsafe extern "C" {
    fn vkCreateInstance(info: *const u8, allocator: *const u8, instance: *mut usize) -> i32;
}

/// Calls `vkCreateInstance`. The fixture binary is built and read, never run.
pub fn init() -> i32 {
    let mut instance = 0;
    // SAFETY: never executed; the fixture only needs the reference to `vkCreateInstance`.
    unsafe { vkCreateInstance(std::ptr::null(), std::ptr::null(), &mut instance) }
}
