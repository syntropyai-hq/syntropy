// SPDX-License-Identifier: MIT OR Apache-2.0

//! WSL2 (RFC-0020 §4.2, WSL2 row; §5, WSL2; §16).
//!
//! WSL2 is Linux in a VM, so reserve, NUMA binding, huge pages and lock are the Linux
//! adapter's. Two things differ, and both are decided here:
//!
//! - **Detection.** WSL2 exposes the GPU through `/dev/dxg`. [`is_wsl2`] tests for it under a
//!   root directory, as RFC-0010's probe does for `HostMemInfo::wsl2`.
//! - **The register-only DMA path.** The pinned allocation call (`cudaHostAlloc`, which a
//!   provider serves as `alloc(MemKind::HostMapped)`) is capped at about 1 GiB under WSL2, so a
//!   buffer a GPU reads by DMA is always anonymous memory from [`OsMemory::reserve`]
//!   registered with `register_host` (`cudaHostRegister`). [`dma_route`] returns
//!   [`DmaRoute::Register`] whenever [`PinLimits::host_alloc_cap`] is set, and
//!   [`host_alloc_allowed`] refuses any pinned allocation above that cap, which is
//!   [`WSL2_HOST_ALLOC_CAP`] (512 MiB) under WSL2.
//!
//! This crate links no GPU runtime: the routes are the decision the pinning loop
//! (TIER-E2-T07) passes to the provider, not provider calls.
//!
//! On Linux (x86-64 and AArch64), [`Wsl2Memory`] is the Linux adapter with the cap recorded in
//! its [`PinLimits`]. Everything else here is portable and tested on every OS.

#![forbid(unsafe_code)]

use std::path::Path;

use syntropy_core::Bytes;

use super::PinLimits;

/// The device node that marks WSL2, relative to the filesystem root.
pub const DXG_DEVICE: &str = "dev/dxg";

/// The largest pinned allocation (`cudaHostAlloc`) anything may make under WSL2: 512 MiB
/// [target], RFC-0020 §5's "nothing uses it for more than 512 MiB", half the ≈1 GiB cap of
/// RFC-0020 §4.2.
pub const WSL2_HOST_ALLOC_CAP: Bytes = Bytes(512 << 20);

/// Whether the system under `root` (normally `/`) is WSL2: `<root>/dev/dxg` exists.
pub fn is_wsl2(root: &Path) -> bool {
    root.join(DXG_DEVICE).exists()
}

/// `limits` with the WSL2 pinned-allocation cap recorded in [`PinLimits::host_alloc_cap`].
/// A lower cap already there is kept.
pub fn wsl2_limits(mut limits: PinLimits) -> PinLimits {
    limits.host_alloc_cap = Some(match limits.host_alloc_cap {
        Some(c) if c.0 < WSL2_HOST_ALLOC_CAP.0 => c,
        _ => WSL2_HOST_ALLOC_CAP,
    });
    limits
}

/// How host memory a GPU reads by DMA is made pinned.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DmaRoute {
    /// Anonymous memory from [`OsMemory::reserve`](super::OsMemory::reserve), registered with
    /// the provider's `register_host` (`cudaHostRegister`, `hipHostRegister`).
    Register,
    /// A pinned allocation by the provider (`cudaHostAlloc`; `alloc(MemKind::HostMapped)`).
    HostAlloc,
}

/// The route for a DMA source: `wanted` where the OS has no pinned-allocation cap, and
/// [`DmaRoute::Register`] wherever it has one (WSL2), whatever was wanted and whatever the
/// size (RFC-0020 §4.2, WSL2 row; §16, "`cudaHostRegister` only").
pub fn dma_route(limits: &PinLimits, wanted: DmaRoute) -> DmaRoute {
    match limits.host_alloc_cap {
        Some(_) => DmaRoute::Register,
        None => wanted,
    }
}

/// Whether a pinned allocation of `bytes` is allowed: always without a cap, and only up to
/// [`PinLimits::host_alloc_cap`] with one (RFC-0020 §5, WSL2). A caller outside the arenas,
/// such as a provider's own small pinned buffer, checks this before allocating.
pub fn host_alloc_allowed(limits: &PinLimits, bytes: Bytes) -> bool {
    limits.host_alloc_cap.is_none_or(|cap| bytes.0 <= cap.0)
}

#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
pub use adapter::Wsl2Memory;

#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
mod adapter {
    use std::ops::Range;
    use std::path::Path;

    use syntropy_core::Bytes;

    use super::{is_wsl2, wsl2_limits};
    use crate::os::linux::LinuxMemory;
    use crate::os::{HostRange, HugePages, MemSnapshot, OsMemError, OsMemory, PinLimits};

    /// The WSL2 host-memory adapter: the Linux adapter, run inside the VM (so `MemTotal` is
    /// the VM's, set by `.wslconfig`), with the pinned-allocation cap in its limits.
    #[derive(Debug)]
    pub struct Wsl2Memory {
        inner: LinuxMemory,
    }

    impl Wsl2Memory {
        /// The adapter when `/dev/dxg` exists, else `None`.
        pub fn detect(numa_reserve: Bytes) -> Option<Self> {
            is_wsl2(Path::new("/")).then(|| Self::new(numa_reserve))
        }

        /// The adapter, without testing for `/dev/dxg`.
        pub fn new(numa_reserve: Bytes) -> Self {
            Self {
                inner: LinuxMemory::new(numa_reserve),
            }
        }

        /// The base page size.
        pub fn page_size(&self) -> Bytes {
            self.inner.page_size()
        }
    }

    impl OsMemory for Wsl2Memory {
        fn snapshot(&self) -> MemSnapshot {
            self.inner.snapshot()
        }

        fn limits(&self) -> PinLimits {
            wsl2_limits(self.inner.limits())
        }

        fn reserve(
            &self,
            bytes: Bytes,
            numa: Option<u16>,
            huge: HugePages,
        ) -> Result<HostRange, OsMemError> {
            self.inner.reserve(bytes, numa, huge)
        }

        fn commit(&self, r: &HostRange, sub: Range<u64>) -> Result<(), OsMemError> {
            self.inner.commit(r, sub)
        }

        fn lock(&self, r: &HostRange, sub: Range<u64>) -> Result<(), OsMemError> {
            self.inner.lock(r, sub)
        }

        fn unlock(&self, r: &HostRange, sub: Range<u64>) -> Result<(), OsMemError> {
            self.inner.unlock(r, sub)
        }

        fn release(&self, r: HostRange) {
            self.inner.release(r)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    #[test]
    fn the_cap_forces_the_register_route() {
        let native = PinLimits::default();
        let wsl = wsl2_limits(native);
        assert_eq!(wsl.host_alloc_cap, Some(Bytes(512 * MIB)));
        assert_eq!(dma_route(&native, DmaRoute::HostAlloc), DmaRoute::HostAlloc);
        assert_eq!(dma_route(&native, DmaRoute::Register), DmaRoute::Register);
        assert_eq!(dma_route(&wsl, DmaRoute::HostAlloc), DmaRoute::Register);
        assert_eq!(dma_route(&wsl, DmaRoute::Register), DmaRoute::Register);
    }

    #[test]
    fn host_alloc_is_never_above_512_mib_under_wsl2() {
        let wsl = wsl2_limits(PinLimits::default());
        assert!(host_alloc_allowed(&wsl, Bytes(512 * MIB)));
        assert!(!host_alloc_allowed(&wsl, Bytes(512 * MIB + 1)));
        assert!(!host_alloc_allowed(&wsl, Bytes(1024 * MIB)));
        assert!(host_alloc_allowed(&PinLimits::default(), Bytes(1 << 40)));
    }

    #[test]
    fn a_lower_cap_is_kept() {
        let low = PinLimits {
            host_alloc_cap: Some(Bytes(64 * MIB)),
            ..PinLimits::default()
        };
        assert_eq!(wsl2_limits(low).host_alloc_cap, Some(Bytes(64 * MIB)));
    }
}
