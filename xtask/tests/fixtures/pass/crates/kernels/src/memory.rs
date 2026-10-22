// SPDX-License-Identifier: MIT OR Apache-2.0

//! The host registrar, with the one `unsafe` method the list allows in this file.

pub trait HostRegistrar {
    /// # Safety
    /// As for `KernelProvider::register_host`.
    #[allow(unsafe_code)]
    unsafe fn register(&self, region: &[u8]) -> u64;
}
