// SPDX-License-Identifier: MIT OR Apache-2.0

//! The provider trait, with the two `unsafe` methods the list allows in this file.

pub trait KernelProvider {
    fn name(&self) -> &str;

    /// # Safety
    /// The caller keeps `region` valid until the registration is released.
    #[allow(unsafe_code)]
    unsafe fn register_host(&self, region: &[u8]) -> u64;

    /// # Safety
    /// The caller keeps `region` valid until the buffer is freed.
    #[allow(unsafe_code)]
    unsafe fn import(&self, region: &[u8]) -> u64;
}
