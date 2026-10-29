// SPDX-License-Identifier: MIT OR Apache-2.0

//! A stand-in for the Vulkan loader: one entry point, never called.

#[unsafe(no_mangle)]
pub extern "C" fn vkCreateInstance(_info: *const u8, _allocator: *const u8, _instance: *mut usize) -> i32 {
    0
}
