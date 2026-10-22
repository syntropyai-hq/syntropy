// SPDX-License-Identifier: MIT OR Apache-2.0

//! The two declarations the list allows here pass. `register` is listed for `src/memory.rs`
//! only, so its declaration here is refused.

pub trait KernelProvider {
    /// # Safety
    /// The caller keeps `region` valid until the registration is released.
    #[allow(unsafe_code)]
    unsafe fn register_host(&self, region: &[u8]) -> u64;

    /// # Safety
    /// The caller keeps `region` valid until the buffer is freed.
    #[allow(unsafe_code)]
    unsafe fn import(&self, region: &[u8]) -> u64;

    #[allow(unsafe_code)]
    unsafe fn register(&self, region: &[u8]) -> u64;
}
