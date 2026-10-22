// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Linux `OsMemory` on the live host (TIER-E2-T02; RFC-0020 §4.2, Linux row). Runs as an
//! unprivileged user in hosted CI.

#![cfg(target_os = "linux")]

use syntropy_core::Bytes;
use syntropy_tier::os::linux::LinuxMemory;
use syntropy_tier::os::lock_or_pageable;
use syntropy_tier::os::procfs::{self, ThpMode};
use syntropy_tier::{HugePages, LockOutcome, OsMemError, OsMemory, PageableReason, PinClass};

const MIB: u64 = 1 << 20;

/// The node the arena asks for. Hosted runners have one node, node 0.
const NODE: u16 = 0;

fn proc_text(rel: &str) -> String {
    std::fs::read_to_string(format!("/proc/self/{rel}")).unwrap()
}

/// The `VmFlags` of the `/proc/self/smaps` mapping that holds `addr`.
fn vm_flags(addr: u64) -> Vec<String> {
    let smaps = proc_text("smaps");
    let mut in_ours = false;
    for line in smaps.lines() {
        if let Some((range, _)) = line.split_once(' ')
            && let Some((s, e)) = range.split_once('-')
            && let (Ok(s), Ok(e)) = (u64::from_str_radix(s, 16), u64::from_str_radix(e, 16))
        {
            in_ours = s <= addr && addr < e;
            continue;
        }
        if in_ours && let Some(flags) = line.strip_prefix("VmFlags:") {
            return flags.split_whitespace().map(str::to_owned).collect();
        }
    }
    panic!("no smaps entry holds {addr:#x}");
}

#[test]
fn arena_pages_land_on_the_requested_node() {
    let os = LinuxMemory::new(Bytes(0));
    let len = 8 * MIB;
    let r = os.reserve(Bytes(len), Some(NODE), HugePages::Off).unwrap();
    assert_eq!(r.numa, Some(NODE), "mbind to node {NODE} failed");
    os.commit(&r, 0..len).unwrap();

    let base = r.base as u64;
    let entry = procfs::numa_maps_entry(&proc_text("numa_maps"), base).unwrap();
    // Neighbouring mappings with the same flags and policy may have merged with ours, so
    // check that the entry found covers the whole arena.
    let end = procfs::maps_end(&proc_text("maps"), entry.start).unwrap();
    assert!(
        entry.start <= base && base + len <= end,
        "{entry:?} ends at {end:#x}"
    );
    assert_eq!(entry.policy, format!("bind:{NODE}"), "{entry:?}");
    let on_node: u64 = entry
        .pages
        .iter()
        .filter(|(n, _)| *n == NODE)
        .map(|(_, p)| p)
        .sum();
    let elsewhere: Vec<_> = entry.pages.iter().filter(|(n, _)| *n != NODE).collect();
    assert!(elsewhere.is_empty(), "pages off node {NODE}: {elsewhere:?}");
    assert!(
        on_node >= len / os.page_size().0,
        "{on_node} pages on node {NODE}, expected at least {}",
        len / os.page_size().0
    );
    os.release(r);
}

#[test]
fn a_node_that_does_not_exist_leaves_the_range_unbound() {
    let os = LinuxMemory::new(Bytes(0));
    let r = os.reserve(Bytes(MIB), Some(1023), HugePages::Off).unwrap();
    assert_eq!(r.numa, None);
    os.commit(&r, 0..MIB).unwrap();
    os.release(r);
}

/// RLIMIT_MEMLOCK at its default: a lock beyond it is refused before `mlock` is called, the
/// range stays `Pageable`, and the reason names the limit.
#[test]
fn default_memlock_leaves_the_arena_pageable_with_a_reason() {
    let os = LinuxMemory::new(Bytes(0));
    let limits = os.limits();
    assert!(
        !limits.cap_ipc_lock,
        "this test must run without CAP_IPC_LOCK"
    );
    let limit = limits
        .memlock
        .expect("this test needs a finite RLIMIT_MEMLOCK, as on an unprivileged account");
    let len = limit.0 + os.page_size().0;
    let r = os.reserve(Bytes(len), None, HugePages::Off).unwrap();
    os.commit(&r, 0..MIB.min(len)).unwrap();
    let outcome = lock_or_pageable(&os, &r, 0..len).unwrap();
    let LockOutcome::Pageable(reason) = outcome else {
        panic!("locked {len} bytes under a {limit:?} limit");
    };
    assert_eq!(outcome.pin_class(), PinClass::Pageable);
    assert_eq!(reason.id(), "memlock_limit");
    let PageableReason::MemlockLimit {
        limit: l,
        requested,
        ..
    } = reason
    else {
        panic!("{reason:?}");
    };
    assert_eq!(l, limit);
    assert!(requested.0 >= len);
    assert!(reason.to_string().contains("RLIMIT_MEMLOCK"));
    os.release(r);
}

/// A lock the limit covers succeeds, shows in `VmLck`, and is undone by `unlock`.
#[test]
fn a_lock_within_memlock_is_taken_and_released() {
    let os = LinuxMemory::new(Bytes(0));
    let page = os.page_size().0;
    let r = os.reserve(Bytes(page), None, HugePages::Off).unwrap();
    os.commit(&r, 0..page).unwrap();
    let before = os.snapshot().own_locked;
    assert_eq!(lock_or_pageable(&os, &r, 0..page), Ok(LockOutcome::Locked));
    assert!(os.snapshot().own_locked.0 >= before.0 + page);
    os.unlock(&r, 0..page).unwrap();
    assert_eq!(os.snapshot().own_locked, before);
    os.release(r);
}

#[test]
fn reserve_sets_dontfork_and_huge_page_advice() {
    let os = LinuxMemory::new(Bytes(0));
    let r = os.reserve(Bytes(4 * MIB), None, HugePages::Auto).unwrap();
    let flags = vm_flags(r.base as u64);
    assert!(
        flags.iter().any(|f| f == "dc"),
        "no MADV_DONTFORK: {flags:?}"
    );
    let thp = procfs::thp(std::path::Path::new("/"));
    assert_eq!(
        flags.iter().any(|f| f == "hg"),
        thp != ThpMode::Never,
        "THP {thp:?}, flags {flags:?}"
    );
    os.release(r);

    let r = os.reserve(Bytes(4 * MIB), None, HugePages::Off).unwrap();
    assert!(!vm_flags(r.base as u64).iter().any(|f| f == "hg"));
    os.release(r);
}

#[test]
fn commit_keeps_bytes_already_written() {
    let os = LinuxMemory::new(Bytes(0));
    let r = os.reserve(Bytes(2 * MIB), None, HugePages::Off).unwrap();
    // SAFETY: the range is a live read-write mapping of 2 MiB that only this test uses.
    unsafe { r.base.add(MIB as usize).write(42) };
    os.commit(&r, 0..2 * MIB).unwrap();
    // SAFETY: as above.
    assert_eq!(unsafe { r.base.add(MIB as usize).read() }, 42);
    os.release(r);
}

#[test]
fn ranges_the_adapter_did_not_reserve_are_refused() {
    let os = LinuxMemory::new(Bytes(0));
    let r = os.reserve(Bytes(MIB), None, HugePages::Off).unwrap();
    assert!(matches!(
        os.commit(&r, 0..MIB + 1),
        Err(OsMemError::OutOfRange { .. })
    ));
    let forged = syntropy_tier::HostRange {
        base: r.base.wrapping_add(4096),
        len: r.len,
        numa: None,
        os_handle: 0,
    };
    assert_eq!(os.commit(&forged, 0..4096), Err(OsMemError::UnknownRange));
    os.release(forged);
    os.release(r);
}

#[test]
fn live_snapshot_reads_this_host() {
    let os = LinuxMemory::new(Bytes(0));
    let s = os.snapshot();
    assert!(s.total.0 > 0 && s.available.0 > 0 && s.available <= s.total);
    assert!(
        s.numa_free.iter().any(|(n, _)| *n == NODE),
        "{:?}",
        s.numa_free
    );
    assert_eq!(s.commit_headroom, None);
}
