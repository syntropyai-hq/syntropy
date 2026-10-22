// SPDX-License-Identifier: MIT OR Apache-2.0

//! WSL2's register-only DMA path (TIER-E2-T04; RFC-0020 §4.2, WSL2 row; §5, WSL2; §16).
//!
//! No CI job runs on WSL2 with a GPU, so the provider is a shim that records which call each
//! pin step would make: `register_host` (`cudaHostRegister`) or a pinned allocation
//! (`cudaHostAlloc`). The steps are routed by `dma_route` over the WSL2 limits.

use std::sync::atomic::{AtomicU64, Ordering};

use syntropy_core::Bytes;
use syntropy_tier::PinLimits;
use syntropy_tier::os::wsl::{
    DmaRoute, WSL2_HOST_ALLOC_CAP, dma_route, host_alloc_allowed, is_wsl2, wsl2_limits,
};

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

/// The provider calls a pinning loop would make, counted.
#[derive(Default)]
struct ShimProvider {
    registered: AtomicU64,
    registered_bytes: AtomicU64,
    host_allocs: AtomicU64,
    largest_host_alloc: AtomicU64,
}

impl ShimProvider {
    /// Pins one step of `bytes` by the route `dma_route` picks for a caller that would rather
    /// have a pinned allocation; a pinned allocation is checked against the cap first.
    fn pin_step(&self, limits: &PinLimits, bytes: Bytes) {
        match dma_route(limits, DmaRoute::HostAlloc) {
            DmaRoute::Register => {
                self.registered.fetch_add(1, Ordering::SeqCst);
                self.registered_bytes.fetch_add(bytes.0, Ordering::SeqCst);
            }
            DmaRoute::HostAlloc => {
                assert!(host_alloc_allowed(limits, bytes));
                self.host_allocs.fetch_add(1, Ordering::SeqCst);
                self.largest_host_alloc.fetch_max(bytes.0, Ordering::SeqCst);
            }
            _ => unreachable!(),
        }
    }
}

/// Criterion 1 with a shim: a 6 GiB arena pinned in steps from 64 MiB (below the cap) to
/// 2 GiB (above it) goes through `register_host` only, never a pinned allocation.
#[test]
fn wsl2_pins_through_register_only() {
    let limits = wsl2_limits(PinLimits::default());
    let shim = ShimProvider::default();
    let steps = [64 * MIB, 256 * MIB, 512 * MIB, 512 * MIB + 1, GIB, 2 * GIB];
    let mut total = 0;
    for s in steps {
        shim.pin_step(&limits, Bytes(s));
        total += s;
    }
    assert_eq!(shim.host_allocs.load(Ordering::SeqCst), 0);
    assert_eq!(shim.registered.load(Ordering::SeqCst), steps.len() as u64);
    assert_eq!(shim.registered_bytes.load(Ordering::SeqCst), total);
}

/// Off WSL2 the caller's wish stands, so the shim does see pinned allocations: the WSL2 result
/// above is the cap's doing, not the shim's.
#[test]
fn without_the_cap_the_wanted_route_stands() {
    let shim = ShimProvider::default();
    shim.pin_step(&PinLimits::default(), Bytes(2 * GIB));
    assert_eq!(shim.host_allocs.load(Ordering::SeqCst), 1);
    assert_eq!(shim.largest_host_alloc.load(Ordering::SeqCst), 2 * GIB);
}

#[test]
fn no_pinned_allocation_above_512_mib() {
    let limits = wsl2_limits(PinLimits::default());
    assert_eq!(WSL2_HOST_ALLOC_CAP, Bytes(512 * MIB));
    assert!(host_alloc_allowed(&limits, WSL2_HOST_ALLOC_CAP));
    assert!(!host_alloc_allowed(
        &limits,
        Bytes(WSL2_HOST_ALLOC_CAP.0 + 1)
    ));
}

#[test]
fn dev_dxg_marks_wsl2() {
    let root = std::env::temp_dir().join(format!("syntropy-tier-wsl2-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("dev")).unwrap();
    assert!(!is_wsl2(&root));
    std::fs::write(root.join("dev/dxg"), b"").unwrap();
    assert!(is_wsl2(&root));
    std::fs::remove_dir_all(&root).unwrap();
}

/// The Linux build of the adapter records the cap in its limits, and is otherwise the Linux
/// adapter.
#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
#[test]
fn the_wsl2_adapter_records_the_cap() {
    use syntropy_tier::os::linux::LinuxMemory;
    use syntropy_tier::os::wsl::Wsl2Memory;
    use syntropy_tier::{HugePages, OsMemory};

    let os = Wsl2Memory::new(Bytes(0));
    let linux = LinuxMemory::new(Bytes(0));
    assert_eq!(os.limits().host_alloc_cap, Some(WSL2_HOST_ALLOC_CAP));
    assert_eq!(os.limits().memlock, linux.limits().memlock);
    assert_eq!(
        dma_route(&os.limits(), DmaRoute::HostAlloc),
        DmaRoute::Register
    );
    let r = os
        .reserve(Bytes(os.page_size().0), None, HugePages::Off)
        .unwrap();
    os.commit(&r, 0..1).unwrap();
    os.release(r);
    assert_eq!(
        Wsl2Memory::detect(Bytes(0)).is_some(),
        is_wsl2(std::path::Path::new("/"))
    );
}
