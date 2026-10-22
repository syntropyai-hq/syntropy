// SPDX-License-Identifier: MIT OR Apache-2.0

//! The region across two processes (DIST-E1-T02): this test binary plays the daemon, and runs
//! itself again as the runner, which opens the region from the handle it is given.
//!
//! - 10⁶ records go each way at once: runner to daemon on `local_tokens`, and daemon to runner on
//!   `local_sequences`. Each side checks every record arrives once and in order.
//! - Both sides read the segment's permissions back as owner-only.
//! - After the runner exits, the daemon unlinks the region, and no process can open it by name.

use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use syntropy_core::Entropy;
use syntropy_ipc::shm::{IpcProtocol, Region, RegionHandle, RingRole, RingSpec, ShmLayout, Side};
use syntropy_spsc::shared::{SharedConsumer, SharedPopError, SharedProducer, SharedPushError};

/// The runner's handle, set only in the child.
const RUNNER: &str = "SYNTROPY_IPC_TEST_RUNNER";
const PROTOCOL: IpcProtocol = IpcProtocol { major: 1, minor: 0 };
const BUILD: [u8; 32] = [0x5a; 32];
const RECORDS: u64 = 1_000_000;
/// The `RunnerState` discriminant the runner records before it exits, as a test value.
const FINAL_STATE: u32 = 7;

/// The segment name's random part, for the test: the process's ID and a counter, mixed. The
/// daemon passes the OS entropy source instead.
struct TestEntropy(AtomicU64);

impl Entropy for TestEntropy {
    fn fill(&self, buf: &mut [u8]) {
        let n = self.0.fetch_add(1, Ordering::Relaxed);
        let mut x = (u64::from(std::process::id()) << 32 | n).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        for byte in buf {
            x ^= x >> 29;
            x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            *byte = (x >> 56) as u8;
        }
    }
}

static ENTROPY: TestEntropy = TestEntropy(AtomicU64::new(0));

fn layout() -> ShmLayout {
    let ring = |role, slots, slot_bytes| RingSpec {
        role,
        slots,
        slot_bytes,
    };
    ShmLayout {
        rings: vec![
            ring(RingRole::Telemetry, 256, 256),
            ring(RingRole::Step { dispatcher: 0 }, 256, 64),
            ring(RingRole::LocalTokens, 1024, 64),
            ring(RingRole::LocalSequences, 1024, 64),
            ring(RingRole::Log, 256, 256),
        ],
    }
}

/// A record: its sequence number, then a few bytes derived from it, so a torn copy shows.
fn record(n: u64) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&n.to_le_bytes());
    bytes[8..].copy_from_slice(&(!n).to_le_bytes());
    bytes
}

fn send(mut producer: SharedProducer<'_>, count: u64) {
    for n in 0..count {
        let bytes = record(n);
        loop {
            match producer.push(&bytes) {
                Ok(()) => break,
                Err(SharedPushError::Full) => thread::yield_now(),
                Err(other) => panic!("push {n}: {other}"),
            }
        }
    }
}

/// Receives `count` records in order; returns once the last arrives.
fn receive(mut consumer: SharedConsumer<'_>, count: u64) {
    let mut buffer = [0u8; 64];
    let mut expected = 0;
    while expected < count {
        match consumer.pop_into(&mut buffer) {
            Ok(len) => {
                assert_eq!(&buffer[..len], record(expected), "record {expected}");
                expected += 1;
            }
            Err(SharedPopError::Empty) => thread::yield_now(),
            Err(other) => panic!("pop after {expected} records: {other}"),
        }
    }
}

/// The runner's side, in the child process.
fn run_runner(handle: &str) {
    let handle: RegionHandle = handle.parse().unwrap();
    let region = Region::open(&handle, Side::Runner, &layout(), PROTOCOL).unwrap();
    assert_eq!(
        region.build_id(),
        BUILD,
        "a spawned runner is the daemon's build"
    );
    assert!(
        region.is_owner_only().unwrap(),
        "the runner sees an owner-only segment"
    );
    let tokens = region.producer(RingRole::LocalTokens).unwrap();
    let sequences = region.consumer(RingRole::LocalSequences).unwrap();
    thread::scope(|scope| {
        scope.spawn(|| send(tokens, RECORDS));
        receive(sequences, RECORDS);
    });
    for _ in 0..3 {
        region.beat();
        region.step();
    }
    region.set_runner_state(FINAL_STATE);
}

#[test]
fn a_runner_process_exchanges_a_million_records_each_way() {
    if let Ok(handle) = std::env::var(RUNNER) {
        run_runner(&handle);
        return;
    }
    let region = Region::create(&layout(), BUILD, PROTOCOL, &ENTROPY).unwrap();
    assert!(
        region.is_owner_only().unwrap(),
        "the daemon made an owner-only segment"
    );
    let spawn = region.spawn_handle().unwrap();
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "a_runner_process_exchanges_a_million_records_each_way",
            "--test-threads",
            "1",
        ])
        .env(RUNNER, spawn.handle().to_string())
        .spawn()
        .unwrap();
    drop(spawn);
    let tokens = region.consumer(RingRole::LocalTokens).unwrap();
    let sequences = region.producer(RingRole::LocalSequences).unwrap();
    thread::scope(|scope| {
        scope.spawn(|| send(sequences, RECORDS));
        receive(tokens, RECORDS);
    });
    let status = child.wait().unwrap();
    assert!(status.success(), "the runner process failed: {status}");
    assert_eq!(region.peer_heartbeat(), 3);
    assert_eq!(region.steps(), 3);
    assert_eq!(region.runner_state(), FINAL_STATE);

    // The runner has exited: the daemon removes the segment's name.
    let handle = region.handle();
    region.unlink().unwrap();
    region.unlink().unwrap();
    drop(region);
    assert!(
        !name_exists(&handle),
        "the segment can still be opened by name after unlink: {handle}"
    );
}

/// Whether a new process could still open the segment `handle` names.
fn name_exists(handle: &RegionHandle) -> bool {
    match handle {
        // A memfd has no name, so nothing could ever open it except through a descriptor.
        RegionHandle::Fd(_) => false,
        RegionHandle::Name(name) => opens_by_name(name),
    }
}

#[cfg(target_vendor = "apple")]
fn opens_by_name(name: &str) -> bool {
    let name = std::ffi::CString::new(name).unwrap();
    // SAFETY: a NUL-terminated name and read-only flags; a descriptor or -1 comes back.
    let fd = unsafe { libc::shm_open(name.as_ptr(), libc::O_RDONLY, 0 as libc::c_uint) };
    if fd >= 0 {
        // SAFETY: `fd` was just opened, and nothing else owns it.
        unsafe { libc::close(fd) };
        return true;
    }
    let error = std::io::Error::last_os_error();
    assert_eq!(error.raw_os_error(), Some(libc::ENOENT), "{error}");
    false
}

#[cfg(windows)]
fn opens_by_name(name: &str) -> bool {
    use windows_sys::Win32::Foundation::{CloseHandle, ERROR_FILE_NOT_FOUND, GetLastError};
    use windows_sys::Win32::System::Memory::{FILE_MAP_READ, OpenFileMappingW};
    let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
    // SAFETY: read access, not inheritable, and a NUL-terminated name; a handle or null comes back.
    let handle = unsafe { OpenFileMappingW(FILE_MAP_READ, 0, wide.as_ptr()) };
    if !handle.is_null() {
        // SAFETY: `handle` was just opened, and nothing else owns it.
        unsafe { CloseHandle(handle) };
        return true;
    }
    // SAFETY: reads this thread's last error, which the call just set.
    assert_eq!(unsafe { GetLastError() }, ERROR_FILE_NOT_FOUND);
    false
}

#[cfg(target_os = "linux")]
fn opens_by_name(_name: &str) -> bool {
    unreachable!("a Linux region has no name")
}

/// A Linux memfd is anonymous: the descriptor's link names a memfd, not a path.
#[cfg(target_os = "linux")]
#[test]
fn a_linux_region_has_no_name_in_the_filesystem() {
    let region = Region::create(&layout(), BUILD, PROTOCOL, &ENTROPY).unwrap();
    let RegionHandle::Fd(fd) = region.handle() else {
        panic!("a Linux region is a descriptor");
    };
    let link = std::fs::read_link(format!("/proc/self/fd/{fd}")).unwrap();
    let link = link.to_string_lossy();
    assert!(link.starts_with("/memfd:syntropy-runner"), "{link}");
}
