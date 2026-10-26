// SPDX-License-Identifier: MIT OR Apache-2.0

//! A provider crate that links what the Metal provider links: the Metal framework, the frameworks
//! beneath its binding and the Objective-C runtime. Each is named by one symbol it exports, so the
//! linker keeps every one in the core binary's load commands.

use std::ffi::{c_char, c_void};

#[link(name = "Metal", kind = "framework")]
unsafe extern "C" {
    fn MTLCreateSystemDefaultDevice() -> *mut c_void;
}

#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {
    static NSFoundationVersionNumber: f64;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFRelease(object: *const c_void);
}

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    fn CGMainDisplayID() -> u32;
}

#[link(name = "objc")]
unsafe extern "C" {
    fn objc_getClass(name: *const c_char) -> *mut c_void;
}

/// Calls one function of each library. The fixture binary is built and read, never run.
pub fn init() -> i32 {
    // SAFETY: never executed; the fixture only needs the references.
    unsafe {
        let device = MTLCreateSystemDefaultDevice();
        CFRelease(device);
        let class = objc_getClass(c"NSObject".as_ptr());
        (CGMainDisplayID() as f64 + NSFoundationVersionNumber) as i32 + i32::from(class.is_null())
    }
}
