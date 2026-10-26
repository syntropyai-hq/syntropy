// SPDX-License-Identifier: MIT OR Apache-2.0

//! PLC-E1-T14 criterion 2: a child process whose seccomp filter blocks `io_uring_setup` reports
//! io_uring not allowed although `kernel.io_uring_disabled` is 0.
//!
//! The test re-runs this test binary as a child with [`CHILD`] set. The child installs a seccomp
//! filter that answers `io_uring_setup` with `EACCES` and allows every other call, then runs the
//! live check and prints one report line, which the parent reads. A filter cannot be removed
//! once installed, so it must not be installed in the test runner's own process.
//!
//! The filter's errno is `EACCES`, not the `EPERM` of the sysctl and of container runtimes'
//! default profiles, so the child seeing exactly `EACCES` shows the test's own filter refused
//! the call. Under CI (the `CI` environment variable set) the test refuses to skip: a host whose
//! sysctl is not 0, or whose test process is already filtered so the control cannot run, fails
//! it instead of passing vacuously.
//!
//! Linux on x86-64 and AArch64 only: the filter matches the system call number of those two
//! architectures.

#![cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]

use std::process::Command;

use syntropy_io::os::checks::{self, LiveSyscalls, SysCheck};

/// Set in the child's environment.
const CHILD: &str = "SYNTROPY_IO_SECCOMP_CHILD";
/// The prefix of the child's report line.
const REPORT: &str = "seccomp-report:";

/// The seccomp filter, declared here because the workspace has no C bindings crate.
mod filter {
    use std::ffi::{c_int, c_ulong};

    const PR_SET_NO_NEW_PRIVS: c_int = 38;
    const PR_SET_SECCOMP: c_int = 22;
    const SECCOMP_MODE_FILTER: c_ulong = 2;

    // Classic BPF opcodes (`linux/bpf_common.h`).
    /// `BPF_LD | BPF_W | BPF_ABS`.
    const BPF_LD_W_ABS: u16 = 0x20;
    /// `BPF_JMP | BPF_JEQ | BPF_K`.
    const BPF_JMP_JEQ_K: u16 = 0x15;
    /// `BPF_RET | BPF_K`.
    const BPF_RET_K: u16 = 0x06;

    /// `offsetof(struct seccomp_data, nr)` and `arch`.
    const DATA_NR: u32 = 0;
    const DATA_ARCH: u32 = 4;

    #[cfg(target_arch = "x86_64")]
    const AUDIT_ARCH: u32 = 0xC000_003E;
    #[cfg(target_arch = "aarch64")]
    const AUDIT_ARCH: u32 = 0xC000_00B7;
    const NR_IO_URING_SETUP: u32 = 425;

    const SECCOMP_RET_ALLOW: u32 = 0x7fff_0000;
    const SECCOMP_RET_ERRNO: u32 = 0x0005_0000;
    /// The filter's errno: `EACCES`, distinct from the sysctl's and runtimes' `EPERM`.
    pub const EACCES: u32 = 13;

    #[repr(C)]
    struct SockFilter {
        code: u16,
        jt: u8,
        jf: u8,
        k: u32,
    }

    #[repr(C)]
    struct SockFprog {
        len: u16,
        filter: *const SockFilter,
    }

    unsafe extern "C" {
        fn prctl(option: c_int, ...) -> c_int;
    }

    const fn op(code: u16, jt: u8, jf: u8, k: u32) -> SockFilter {
        SockFilter { code, jt, jf, k }
    }

    /// Installs, on the calling thread, a filter that answers `io_uring_setup` with `EACCES` and
    /// allows everything else, including every call of a foreign architecture.
    pub fn deny_io_uring_setup() -> std::io::Result<()> {
        let program = [
            op(BPF_LD_W_ABS, 0, 0, DATA_ARCH),
            op(BPF_JMP_JEQ_K, 0, 3, AUDIT_ARCH),
            op(BPF_LD_W_ABS, 0, 0, DATA_NR),
            op(BPF_JMP_JEQ_K, 0, 1, NR_IO_URING_SETUP),
            op(BPF_RET_K, 0, 0, SECCOMP_RET_ERRNO | EACCES),
            op(BPF_RET_K, 0, 0, SECCOMP_RET_ALLOW),
        ];
        let fprog = SockFprog {
            len: program.len() as u16,
            filter: program.as_ptr(),
        };
        // SAFETY: `PR_SET_NO_NEW_PRIVS` takes integer arguments only.
        if unsafe {
            prctl(
                PR_SET_NO_NEW_PRIVS,
                1 as c_ulong,
                0 as c_ulong,
                0 as c_ulong,
                0 as c_ulong,
            )
        } != 0
        {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: `fprog` and the `program` it points to are live for the whole call; the kernel
        // copies the filter before `prctl` returns.
        if unsafe {
            prctl(
                PR_SET_SECCOMP,
                SECCOMP_MODE_FILTER,
                &raw const fprog,
                0 as c_ulong,
                0 as c_ulong,
            )
        } != 0
        {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
}

/// `/proc/sys/kernel/io_uring_disabled`, trimmed, or `None` on a kernel before the knob.
fn sysctl() -> Option<String> {
    std::fs::read_to_string("/proc/sys/kernel/io_uring_disabled")
        .ok()
        .map(|s| s.trim().to_owned())
}

/// The sysctl lets every process create rings: 0, or absent (a kernel before 6.6).
fn sysctl_is_zero() -> bool {
    sysctl().is_none_or(|v| v == "0")
}

/// Under CI a skip is a failure; elsewhere it is a printed notice.
fn cannot_run(why: &str) {
    assert!(
        std::env::var_os("CI").is_none(),
        "criterion 2 cannot run on this CI host: {why}"
    );
    eprintln!("NOT RUN: {why}");
}

/// This process has no seccomp filter (`Seccomp: 0`).
fn unfiltered() -> bool {
    std::fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find_map(|l| l.strip_prefix("Seccomp:"))
        .is_some_and(|v| v.trim() == "0")
}

/// The child's half. Ignored in a normal run; the parent runs it with `--ignored`.
#[test]
#[ignore = "run only as the child of `seccomp_blocked_io_uring_is_not_allowed`"]
fn seccomp_child() {
    if std::env::var_os(CHILD).is_none() {
        return;
    }
    filter::deny_io_uring_setup().expect("install the seccomp filter");
    let setup = checks::io_uring_setup(&LiveSyscalls);
    let allowed = checks::io_uring_allowed(sysctl_is_zero(), setup);
    println!(
        "{REPORT} sysctl={:?} setup={setup:?} allowed={allowed}",
        sysctl()
    );
}

#[test]
fn seccomp_blocked_io_uring_is_not_allowed() {
    let sysctl = sysctl();
    if !sysctl_is_zero() {
        cannot_run(&format!(
            "kernel.io_uring_disabled is {sysctl:?}, so this host cannot show a filter blocking \
             io_uring while the sysctl is 0"
        ));
        return;
    }
    let exe = std::env::current_exe().expect("current test binary");
    let out = Command::new(exe)
        .args(["--exact", "seccomp_child", "--ignored", "--nocapture"])
        .args(["--test-threads", "1"])
        .env(CHILD, "1")
        .output()
        .expect("spawn the child");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "child failed: {}\nstdout:\n{stdout}\nstderr:\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let report = stdout
        .lines()
        // The test harness prints `test seccomp_child ... ` first, on the same line.
        .find_map(|l| l.split_once(REPORT).map(|(_, rest)| rest))
        .unwrap_or_else(|| panic!("no report line in the child's output:\n{stdout}"));
    eprintln!("child report:{report}");
    assert_eq!(
        report.trim(),
        format!(
            "sysctl={sysctl:?} setup={:?} allowed=false",
            SysCheck::Denied {
                errno: filter::EACCES as i32
            }
        )
    );

    // The control: this process has no filter of the test's. Where nothing else filters it
    // either, the same call succeeds, so the child's denial came from its filter.
    let control = checks::io_uring_setup(&LiveSyscalls);
    let unfiltered = unfiltered();
    eprintln!("control: io_uring_setup {control:?}, process unfiltered {unfiltered}");
    if unfiltered {
        assert_eq!(control, SysCheck::Granted);
        assert!(checks::io_uring_allowed(true, control));
    } else {
        cannot_run(&format!(
            "the test process already has a seccomp filter, so the unfiltered control cannot \
             run (it gave {control:?})"
        ));
    }
}
