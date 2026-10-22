// SPDX-License-Identifier: MIT OR Apache-2.0

//! The listed declaration passes; the call below it is refused.

pub trait HostRegistrar {
    /// # Safety
    /// As for `KernelProvider::register_host`.
    #[allow(unsafe_code)]
    unsafe fn register(&self, region: &[u8]) -> u64;
}

pub fn register_all(registrar: &dyn HostRegistrar, region: &[u8]) -> u64 {
    unsafe { registrar.register(region) }
}
