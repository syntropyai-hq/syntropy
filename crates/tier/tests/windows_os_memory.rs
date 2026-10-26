// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Windows `OsMemory` on the live host (TIER-E2-T03; RFC-0020 §4.2, Windows row; §5,
//! Windows). Runs in hosted CI's `hosted-test-windows`.

#![cfg(all(windows, any(target_arch = "x86_64", target_arch = "aarch64")))]

use std::ffi::c_void;
use std::io::{BufRead, Write};
use std::process::{Command, Stdio};

use syntropy_core::Bytes;
use syntropy_core::budget::{HostBudgetInputs, HostBudgetPolicy, HostOs, host_usable};
use syntropy_core::resource::HostMemInfo;
use syntropy_tier::arena::{SlotClass, SlotGeometry};
use syntropy_tier::os::windows::{WindowsMemory, job_memory, process_commit, working_set_size};
use syntropy_tier::os::{commit_if, commit_term, lock_or_pageable};
use syntropy_tier::{
    CommitOutcome, HostRange, HugePages, LockOutcome, OsMemError, OsMemory, lock_ranges,
    slot_layout,
};

const MIB: u64 = 1 << 20;

/// Held by every test: the commit charge and the locked total they measure are per process,
/// so the tests run one at a time.
static LOCKS: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn locks_guard() -> std::sync::MutexGuard<'static, ()> {
    LOCKS.lock().unwrap_or_else(|p| p.into_inner())
}

fn commit_now() -> u64 {
    process_commit().expect("K32GetProcessMemoryInfo").0
}

/// Reserving takes no commit charge; committing takes it.
#[test]
fn reserve_takes_no_commit_charge_and_commit_does() {
    let _process = locks_guard();
    let os = WindowsMemory::new(Bytes(0));
    let len = 256 * MIB;
    let before = commit_now();
    let r = os.reserve(Bytes(len), None, HugePages::Auto).unwrap();
    let reserved = commit_now();
    assert!(
        reserved < before + 4 * MIB,
        "reserve charged {} bytes",
        reserved.saturating_sub(before)
    );
    os.commit(&r, 0..64 * MIB).unwrap();
    let committed = commit_now();
    assert!(
        committed >= reserved + 64 * MIB,
        "commit of 64 MiB charged {} bytes",
        committed.saturating_sub(reserved)
    );
    // SAFETY: the first 64 MiB are committed read-write and only this test uses them.
    unsafe {
        r.base.write(5);
        assert_eq!(r.base.read(), 5);
    }
    os.release(r);
}

/// With a gate that refuses, `commit_if` takes no charge; with the real commit term and
/// ample headroom, it commits.
#[test]
fn commit_if_takes_no_charge_until_the_gate_allows_it() {
    let _process = locks_guard();
    let os = WindowsMemory::new(Bytes(0));
    let r = os.reserve(Bytes(128 * MIB), None, HugePages::Off).unwrap();
    let before = commit_now();
    let out = commit_if(&os, &r, 0..128 * MIB, |s, _| {
        assert!(
            s.commit_headroom.is_some(),
            "no commit accounting in the snapshot"
        );
        false
    })
    .unwrap();
    assert_eq!(out, CommitOutcome::Deferred);
    assert!(commit_now() < before + 4 * MIB);

    let out = commit_if(&os, &r, 0..128 * MIB, |s, len| {
        commit_term(s, len, Bytes(0))
    })
    .unwrap();
    assert_eq!(out, CommitOutcome::Committed);
    assert!(commit_now() >= before + 128 * MIB);
    // A gate whose headroom is the whole commit limit can never allow a step.
    let limit_headroom = Bytes(u64::MAX / 2);
    let out = commit_if(&os, &r, 0..MIB, |s, len| {
        commit_term(s, len, limit_headroom)
    })
    .unwrap();
    assert_eq!(out, CommitOutcome::Deferred);
    os.release(r);
}

/// Lock ranges never exceed `tier.pin_step` on this host: an expert-cache layout is cut into
/// lock ranges, each reserved on its own, committed and locked, and no slot spans two.
#[test]
fn lock_ranges_stay_within_pin_step_on_this_host() {
    let _process = locks_guard();
    let pin_step = Bytes(8 * MIB);
    let geometry = SlotGeometry::Classes(
        [(3 * MIB, 5), (MIB, 9), (2 * MIB, 3)]
            .into_iter()
            .map(|(size, count)| SlotClass {
                size: Bytes(size),
                count,
                layer: None,
            })
            .collect(),
    );
    let slots = slot_layout(&geometry, &[]);
    let ranges = lock_ranges(&slots, pin_step);
    let os = WindowsMemory::new(Bytes(0));
    let mut held = Vec::new();
    for lr in &ranges {
        assert!(lr.len() <= pin_step, "{lr:?}");
        let r = os.reserve(lr.len(), None, HugePages::Off).unwrap();
        assert!(r.len <= pin_step, "reservation {:?} > {pin_step:?}", r.len);
        os.commit(&r, 0..r.len.0).unwrap();
        let outcome = lock_or_pageable(&os, &r, 0..r.len.0).unwrap();
        assert_eq!(outcome, LockOutcome::Locked, "lock range {lr:?}");
        // Each slot of the range lies inside this reservation.
        let mut offset = 0;
        for s in &slots[lr.slots.clone()] {
            assert!(offset + s.len.0 <= r.len.0);
            offset += s.len.0;
        }
        held.push(r);
    }
    assert!(os.locked().0 >= ranges.iter().map(|r| r.len().0).sum::<u64>());
    assert!(os.limits().ws_hard_min_ok);
    for r in held {
        os.unlock(&r, 0..r.len.0).unwrap();
        os.release(r);
    }
    assert_eq!(os.locked(), Bytes(0));
}

#[test]
fn snapshot_reads_this_host() {
    let _process = locks_guard();
    let os = WindowsMemory::new(Bytes(0));
    let s = os.snapshot();
    assert!(s.total.0 > 0 && s.available.0 > 0 && s.available <= s.total);
    assert!(s.commit_headroom.is_some());
    assert!(s.os_level.is_some(), "no low-memory notification handle");
    assert!(
        s.numa_free.iter().any(|(n, _)| *n == 0),
        "{:?}",
        s.numa_free
    );
    assert_eq!(s.psi, None);
    let limits = os.limits();
    assert_eq!(limits.memlock, None);
    assert_eq!(limits.io_fixed_max, Some(Bytes(0)));
    assert_eq!(limits.helper_gpu_cap, Some(Bytes(8 << 30)));
}

#[test]
fn a_node_the_host_lacks_leaves_the_range_unbound() {
    let _process = locks_guard();
    let os = WindowsMemory::new(Bytes(0));
    let r = os.reserve(Bytes(MIB), Some(1023), HugePages::Off).unwrap();
    assert_eq!(r.numa, None);
    os.release(r);
    let r = os.reserve(Bytes(MIB), Some(0), HugePages::Off).unwrap();
    assert_eq!(r.numa, Some(0));
    os.release(r);
}

/// A copy of a range's identity that can cross threads.
#[derive(Clone, Copy)]
struct Handle {
    base: usize,
    len: u64,
    id: u64,
}

impl Handle {
    fn of(r: &HostRange) -> Self {
        Handle {
            base: r.base as usize,
            len: r.len.0,
            id: r.os_handle,
        }
    }

    fn range(self) -> HostRange {
        HostRange {
            base: self.base as *mut u8,
            len: Bytes(self.len),
            numa: None,
            os_handle: self.id,
        }
    }
}

/// The registry hardening of the Linux adapter holds here too: `release` racing `commit` on
/// other threads never faults, and a stale handle is refused, also by another adapter.
#[test]
fn release_racing_commit_is_clean_and_stale_handles_are_refused() {
    let _process = locks_guard();
    use std::sync::atomic::{AtomicBool, Ordering};

    let os = WindowsMemory::new(Bytes(0));
    let page = os.page_size().0;
    for _round in 0..20 {
        let len = 64 * page;
        let h = Handle::of(&os.reserve(Bytes(len), None, HugePages::Off).unwrap());
        let released = AtomicBool::new(false);
        std::thread::scope(|s| {
            for _ in 0..3 {
                let (os, released) = (&os, &released);
                s.spawn(move || {
                    for i in 0..200u64 {
                        let after = released.load(Ordering::Acquire);
                        let sub = (i % 64) * page..(i % 64 + 1) * page;
                        match os.commit(&h.range(), sub) {
                            Ok(()) => assert!(!after, "a commit succeeded after release"),
                            Err(OsMemError::UnknownRange) => {}
                            Err(e) => panic!("unexpected {e:?}"),
                        }
                    }
                });
            }
            s.spawn(|| {
                std::thread::yield_now();
                os.release(h.range());
                released.store(true, Ordering::Release);
                for _ in 0..8 {
                    let r = os.reserve(Bytes(len), None, HugePages::Off).unwrap();
                    assert_ne!(r.os_handle, h.id);
                    os.release(r);
                }
            });
        });
        assert_eq!(
            os.commit(&h.range(), 0..page),
            Err(OsMemError::UnknownRange)
        );
    }

    let other = WindowsMemory::new(Bytes(0));
    let r = os.reserve(Bytes(MIB), None, HugePages::Off).unwrap();
    let stale = Handle::of(&r);
    os.release(r);
    let fresh = other.reserve(Bytes(MIB), None, HugePages::Off).unwrap();
    assert_eq!(
        other.commit(&stale.range(), 0..MIB),
        Err(OsMemError::UnknownRange)
    );
    other.release(stale.range());
    assert_eq!(other.commit(&fresh, 0..MIB), Ok(()));
    other.release(fresh);
}

#[test]
fn unlock_keeps_a_page_a_neighbour_shares() {
    let _process = locks_guard();
    let os = WindowsMemory::new(Bytes(0));
    let page = os.page_size().0;
    let r = os.reserve(Bytes(2 * page), None, HugePages::Off).unwrap();
    os.commit(&r, 0..2 * page).unwrap();
    os.lock(&r, 0..2 * page).unwrap();
    let locked = os.locked();
    os.unlock(&r, page / 2..page + page / 2).unwrap();
    assert_eq!(os.locked(), locked);
    os.unlock(&r, 0..2 * page).unwrap();
    assert_eq!(os.locked(), Bytes(0));
    os.release(r);
}

/// `QUOTA_LIMITS_HARDWS_MIN_ENABLE`.
const HARDWS_MIN_ENABLE: u32 = 0x1;

/// The hard working-set minimum covers locked bytes only while they are locked: after an
/// unlock, and after a release of a locked range, the minimum and flags fall back to what the
/// process had before.
#[test]
fn working_set_minimum_falls_back_after_unlock_and_release() {
    let _process = locks_guard();
    let (min0, _, flags0) = working_set_size().unwrap();
    let os = WindowsMemory::new(Bytes(0));
    let len = 16 * MIB;
    let r = os.reserve(Bytes(len), None, HugePages::Off).unwrap();
    os.commit(&r, 0..len).unwrap();

    os.lock(&r, 0..len).unwrap();
    let (min, _, flags) = working_set_size().unwrap();
    assert!(
        min as u64 >= min0 as u64 + len,
        "minimum {min} after locking {len}"
    );
    assert_ne!(
        flags & HARDWS_MIN_ENABLE,
        0,
        "no hard minimum: flags {flags:#x}"
    );

    // Half unlocked: the minimum falls by that half and stays hard.
    os.unlock(&r, 0..len / 2).unwrap();
    let (half, _, flags) = working_set_size().unwrap();
    assert!(
        half < min && half as u64 >= min0 as u64 + len / 2,
        "minimum {half}"
    );
    assert_ne!(flags & HARDWS_MIN_ENABLE, 0);

    os.unlock(&r, len / 2..len).unwrap();
    let (min, _, flags) = working_set_size().unwrap();
    assert_eq!((min, flags), (min0, flags0), "not restored after unlock");

    os.lock(&r, 0..len).unwrap();
    assert!(working_set_size().unwrap().0 > min0);
    os.release(r);
    let (min, _, flags) = working_set_size().unwrap();
    assert_eq!((min, flags), (min0, flags0), "not restored after release");
    assert_eq!(os.locked(), Bytes(0));
}

// ---- Job-object limits ---------------------------------------------------------------------

mod job {
    use std::ffi::c_void;

    pub type Handle = *mut c_void;

    pub const JOB_OBJECT_EXTENDED_LIMIT_INFORMATION: i32 = 9;
    pub const JOB_OBJECT_LIMIT_PROCESS_MEMORY: u32 = 0x0000_0100;

    #[repr(C)]
    #[derive(Default)]
    pub struct BasicLimit {
        pub per_process_user_time_limit: i64,
        pub per_job_user_time_limit: i64,
        pub limit_flags: u32,
        pub minimum_working_set_size: usize,
        pub maximum_working_set_size: usize,
        pub active_process_limit: u32,
        pub affinity: usize,
        pub priority_class: u32,
        pub scheduling_class: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct ExtendedLimit {
        pub basic: BasicLimit,
        pub io: [u64; 6],
        pub process_memory_limit: usize,
        pub job_memory_limit: usize,
        pub peak_process_memory_used: usize,
        pub peak_job_memory_used: usize,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn CreateJobObjectW(attributes: *mut c_void, name: *const u16) -> Handle;
        pub fn SetInformationJobObject(
            job: Handle,
            class: i32,
            info: *mut c_void,
            length: u32,
        ) -> i32;
        pub fn AssignProcessToJobObject(job: Handle, process: Handle) -> i32;
        pub fn CloseHandle(h: Handle) -> i32;
    }
}

const CHILD_ENV: &str = "SYNTROPY_TIER_JOB_CHILD_LIMIT";
const JOB_LIMIT: u64 = 256 * MIB;

/// A job object with a 256 MiB process memory limit: the child placed in it reads the limit,
/// the snapshot's cgroup term and the budget rule follow it, a commit inside the headroom
/// succeeds and one past it fails. The parent creates the job and assigns the child, so the
/// test process itself is never constrained.
#[test]
fn job_object_limits_are_read_and_honoured() {
    let _process = locks_guard();
    if let Ok(limit) = std::env::var(CHILD_ENV) {
        return job_child(limit.parse().unwrap());
    }

    let exe = std::env::current_exe().unwrap();
    let mut child = Command::new(exe)
        .args([
            "--exact",
            "job_object_limits_are_read_and_honoured",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ENV, JOB_LIMIT.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    // SAFETY: an anonymous job with default security; the handle is closed below.
    let job = unsafe { job::CreateJobObjectW(std::ptr::null_mut(), std::ptr::null()) };
    assert!(!job.is_null(), "CreateJobObjectW failed");
    let mut info = job::ExtendedLimit::default();
    info.basic.limit_flags = job::JOB_OBJECT_LIMIT_PROCESS_MEMORY;
    info.process_memory_limit = JOB_LIMIT as usize;
    // SAFETY: `info` is a JOBOBJECT_EXTENDED_LIMIT_INFORMATION of the size passed; the job
    // handle is valid.
    let ok = unsafe {
        job::SetInformationJobObject(
            job,
            job::JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            (&mut info as *mut job::ExtendedLimit).cast::<c_void>(),
            size_of::<job::ExtendedLimit>() as u32,
        )
    };
    assert_ne!(ok, 0, "SetInformationJobObject failed");
    let process = std::os::windows::io::AsRawHandle::as_raw_handle(&child);
    // SAFETY: both handles are valid; the child is a process this test spawned.
    let ok = unsafe { job::AssignProcessToJobObject(job, process.cast()) };
    assert_ne!(ok, 0, "AssignProcessToJobObject failed");

    // Release the child, which waits on stdin until it is in the job.
    child.stdin.take().unwrap().write_all(b"go\n").unwrap();
    let out = child.wait_with_output().unwrap();
    // SAFETY: the job handle this test created, closed once.
    unsafe { job::CloseHandle(job) };
    let text = format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.status.success(), "child failed: {text}");
    assert!(text.contains("job child: ok"), "child did not run: {text}");
}

fn job_child(limit: u64) {
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line).unwrap();

    let job = job_memory().expect("the child is in a job");
    assert_eq!(job.process_limit, Some(Bytes(limit)), "{job:?}");
    let headroom = job.headroom().unwrap();
    assert!(headroom.0 > 0 && headroom.0 < limit, "{job:?}");

    let os = WindowsMemory::new(Bytes(0));
    let s = os.snapshot();
    let cgroup = s.cgroup.expect("the snapshot carries the job limit");
    assert_eq!(cgroup.max, Some(Bytes(limit)));

    // RFC-0020 §5's Windows rule: host_usable ≤ job limit − usage + own.
    let mem = HostMemInfo {
        total: s.total,
        available: s.available,
        cgroup_limit: Some(headroom),
        ..HostMemInfo::default()
    };
    let usable = host_usable(
        &HostBudgetInputs {
            os: HostOs::Windows,
            mem: &mem,
            own: Bytes(0),
            own_locked: Bytes(0),
            working_set: None,
            user_locked_vm: Bytes(0),
        },
        &HostBudgetPolicy::default(),
    );
    assert!(usable <= headroom, "{usable:?} > {headroom:?}");

    // The OS enforces it: a commit inside the headroom succeeds, one past it fails.
    let r = os.reserve(Bytes(2 * limit), None, HugePages::Off).unwrap();
    let inside = (headroom.0 / 2) & !(os.page_size().0 - 1);
    os.commit(&r, 0..inside).unwrap();
    let past = os.commit(&r, inside..inside + limit);
    assert!(
        matches!(past, Err(OsMemError::HostReserve { .. })),
        "a commit past the job limit gave {past:?}"
    );
    os.release(r);
    println!("job child: ok");
}
