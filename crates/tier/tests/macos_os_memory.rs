// SPDX-License-Identifier: MIT OR Apache-2.0

//! The macOS `OsMemory` on the live host (TIER-E2-T04; RFC-0020 §4.2, macOS row; §5, macOS).
//! Runs as an unprivileged user.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::process::Command;

use syntropy_core::Bytes;
use syntropy_tier::os::lock_or_pageable;
use syntropy_tier::os::macos::{MAC_PAGE, MacMemory};
use syntropy_tier::{HostRange, HugePages, LockOutcome, OsMemError, OsMemory, PageableReason};

const MIB: u64 = 1 << 20;

/// `sysctl -n <name>` from the command-line tool, to cross-check the adapter's reads.
fn sysctl_cli(name: &str) -> u64 {
    let out = Command::new("/usr/sbin/sysctl")
        .args(["-n", name])
        .output()
        .expect("run sysctl");
    String::from_utf8(out.stdout)
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}

#[test]
fn reads_the_wire_limits_and_the_gpu_limit() {
    let os = MacMemory::new();
    let w = os.wire_limits();
    assert_eq!(w.user, Some(Bytes(sysctl_cli("vm.user_wire_limit"))));
    assert_eq!(
        w.global,
        Some(Bytes(sysctl_cli("vm.global_user_wire_limit")))
    );
    let iogpu_mb = sysctl_cli("iogpu.wired_limit_mb");
    assert_eq!(
        w.iogpu_wired_limit,
        (iogpu_mb > 0).then(|| Bytes(iogpu_mb * MIB))
    );
    assert_eq!(os.limits().user_wire_limit, w.cap());
    assert!(w.cap().is_some());
}

#[test]
fn snapshot_reads_ram_and_an_available_estimate() {
    let os = MacMemory::new();
    let s = os.snapshot();
    assert_eq!(s.total, Bytes(sysctl_cli("hw.memsize")));
    assert!(s.available.0 > 0 && s.available.0 <= s.total.0, "{s:?}");
    assert_eq!(s.commit_headroom, None);
    assert!(s.cgroup.is_none() && s.psi.is_none() && s.numa_available.is_empty());
}

#[test]
fn reserve_is_16_kib_aligned_and_rounded() {
    let os = MacMemory::new();
    assert_eq!(os.page_size(), MAC_PAGE);
    let r = os
        .reserve(Bytes(MAC_PAGE.0 + 1), Some(3), HugePages::Auto)
        .unwrap();
    assert_eq!(r.base as usize % MAC_PAGE.0 as usize, 0);
    assert_eq!(r.len, Bytes(2 * MAC_PAGE.0));
    // One memory domain: a NUMA node asked for is not bound.
    assert_eq!(r.numa, None);
    os.release(r);
    assert!(matches!(
        os.reserve(Bytes(0), None, HugePages::Off),
        Err(OsMemError::OutOfRange { .. })
    ));
}

#[test]
fn commit_keeps_the_bytes() {
    let os = MacMemory::new();
    let r = os
        .reserve(Bytes(4 * MAC_PAGE.0), None, HugePages::Off)
        .unwrap();
    // SAFETY: the range is a live read-write mapping of 4 pages that only this test uses.
    unsafe {
        r.base.write(7);
        r.base.add(MAC_PAGE.0 as usize).write(9);
    }
    os.commit(&r, 0..4 * MAC_PAGE.0).unwrap();
    // SAFETY: as above.
    unsafe {
        assert_eq!(r.base.read(), 7);
        assert_eq!(r.base.add(MAC_PAGE.0 as usize).read(), 9);
    }
    os.release(r);
}

/// Criterion 3, live: wiring 32 MiB through `mlock` succeeds within the real user wire
/// limit and the adapter's wired bytes stay within it; a request past the limit is refused
/// before any `mlock`, and the range stays pageable.
#[test]
fn wired_bytes_stay_within_the_user_wire_limit() {
    let os = MacMemory::new();
    let limit = os.limits().user_wire_limit.expect("wire limit readable");
    let r = os.reserve(Bytes(32 * MIB), None, HugePages::Off).unwrap();
    os.commit(&r, 0..32 * MIB).unwrap();
    assert_eq!(
        lock_or_pageable(&os, &r, 0..32 * MIB).unwrap(),
        LockOutcome::Locked
    );
    assert_eq!(os.wired(), Bytes(32 * MIB));
    assert!(os.wired().0 <= limit.0);
    os.unlock(&r, 0..32 * MIB).unwrap();
    assert_eq!(os.wired(), Bytes(0));
    os.release(r);

    // A reservation one page larger than the limit: never touched, so it costs no RAM.
    let big = os
        .reserve(Bytes(limit.0 + MAC_PAGE.0), None, HugePages::Off)
        .unwrap();
    let out = lock_or_pageable(&os, &big, 0..big.len.0).unwrap();
    assert!(
        matches!(
            out,
            LockOutcome::Pageable(PageableReason::WireLimit { limit: l, wired: Bytes(0), .. })
                if l == limit
        ),
        "{out:?}"
    );
    assert_eq!(os.wired(), Bytes(0));
    os.release(big);
}

#[test]
fn a_forged_or_released_range_is_refused() {
    let os = MacMemory::new();
    let r = os.reserve(Bytes(MAC_PAGE.0), None, HugePages::Off).unwrap();
    let forged = HostRange {
        base: r.base,
        len: r.len,
        numa: None,
        os_handle: r.os_handle + 1_000_000,
    };
    assert_eq!(os.lock(&forged, 0..1), Err(OsMemError::UnknownRange));
    let stale = HostRange {
        base: r.base,
        len: r.len,
        numa: None,
        os_handle: r.os_handle,
    };
    os.release(r);
    assert_eq!(os.commit(&stale, 0..1), Err(OsMemError::UnknownRange));
    os.release(stale);
}
