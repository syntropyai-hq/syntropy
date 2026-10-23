// SPDX-License-Identifier: MIT OR Apache-2.0

//! E1's Linux system-call checks (RFC-0010 §2.2, ISA and Environment rows; PLC-E1-T14).
//!
//! Two host facts cannot be read from a file:
//!
//! - **AMX tile-data permission.** Linux hands out the AMX tile state only to a process that asks
//!   for it with `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)`. A grant fills
//!   `CpuTopology::amx_permitted`. The permission belongs to the calling process, and asking has
//!   no effect beyond letting that process use the tiles.
//! - **io_uring allowed.** RFC-0010 §2.2's test is "`io_uring_setup` succeeds and
//!   `/proc/sys/kernel/io_uring_disabled` is 0". `probe` reads the sysctl; this module makes the
//!   call. A seccomp filter, such as a container runtime's default profile, can refuse the call
//!   while the sysctl is 0 (ADR-0020).
//!
//! `probe` may not use `unsafe` (RFC-0001 §4.2 rule 8), so the calls live here behind safe
//! functions (rule 8, note of 2026-10-22; OQ-PRC-26). Each check goes through the [`Syscalls`]
//! trait, so the code that classifies a result runs, and is tested, on every OS with an injected
//! shim; [`LiveSyscalls`] makes the real calls. Neither call needs a capability or root.
//!
//! The workspace has no C bindings crate, so `syscall(2)` and the constants are declared here.
//! The errno values are Linux's generic ones, the same on x86-64 and AArch64.

/// Linux errno values the classifiers read.
mod errno {
    pub const EPERM: i32 = 1;
    pub const EACCES: i32 = 13;
    pub const EINVAL: i32 = 22;
    pub const ENOSYS: i32 = 38;
    pub const EOPNOTSUPP: i32 = 95;
}

/// What one raw system call returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RawCall {
    /// The call does not exist on this target, so it was not made: any OS but Linux, and
    /// `arch_prctl` on any architecture but x86-64.
    Absent,
    /// The call succeeded.
    Ok,
    /// The call failed with this Linux errno.
    Err(i32),
}

/// The raw system calls behind the checks. [`LiveSyscalls`] makes them; tests inject a shim.
pub trait Syscalls {
    /// `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)`.
    fn request_xtiledata_permission(&self) -> RawCall;

    /// `io_uring_setup(1, &params)` with zeroed parameters. A ring it creates is closed before
    /// this returns.
    fn io_uring_setup(&self) -> RawCall;
}

/// The outcome of one check.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SysCheck {
    /// The call succeeded: the permission is granted, or a ring could be created.
    Granted,
    /// The kernel or a filter refused the call (`EPERM`, `EACCES`).
    Denied {
        /// The errno.
        errno: i32,
    },
    /// The kernel, the CPU or the target lacks the feature. `errno` is `None` when the call was
    /// not made because it does not exist on this target.
    ///
    /// For `io_uring_setup`, `ENOSYS` is a kernel built without io_uring or a filter that answers
    /// with `ENOSYS`; the two cannot be told apart, and either way rings are not allowed.
    Unsupported {
        /// The errno, when the call was made.
        errno: Option<i32>,
    },
    /// The call failed for another reason, such as a resource limit.
    Failed {
        /// The errno.
        errno: i32,
    },
}

impl SysCheck {
    /// The call succeeded.
    #[must_use]
    pub fn granted(self) -> bool {
        self == SysCheck::Granted
    }
}

/// Requests the AMX tile-data permission for this process.
///
/// `EINVAL` (a kernel before 5.16, which does not know the request), `EOPNOTSUPP` (a CPU or
/// kernel without AMX) and `ENOSYS` are [`SysCheck::Unsupported`]; `EPERM` and `EACCES` (a
/// filter) are [`SysCheck::Denied`]; anything else, such as `ENOSPC` for an alternate signal
/// stack too small for the tile state, is [`SysCheck::Failed`].
pub fn amx_tile_permission(sys: &dyn Syscalls) -> SysCheck {
    match sys.request_xtiledata_permission() {
        RawCall::Ok => SysCheck::Granted,
        RawCall::Absent => SysCheck::Unsupported { errno: None },
        RawCall::Err(e @ (errno::EINVAL | errno::EOPNOTSUPP | errno::ENOSYS)) => {
            SysCheck::Unsupported { errno: Some(e) }
        }
        RawCall::Err(e @ (errno::EPERM | errno::EACCES)) => SysCheck::Denied { errno: e },
        RawCall::Err(e) => SysCheck::Failed { errno: e },
    }
}

/// Tries to create, and closes, a one-entry io_uring.
///
/// `ENOSYS` is [`SysCheck::Unsupported`]; `EPERM` (the sysctl, a filter) and `EACCES` are
/// [`SysCheck::Denied`]; anything else, such as `ENOMEM` under a small memory-lock limit on an
/// older kernel or `EMFILE`, is [`SysCheck::Failed`].
pub fn io_uring_setup(sys: &dyn Syscalls) -> SysCheck {
    match sys.io_uring_setup() {
        RawCall::Ok => SysCheck::Granted,
        RawCall::Absent => SysCheck::Unsupported { errno: None },
        RawCall::Err(e @ errno::ENOSYS) => SysCheck::Unsupported { errno: Some(e) },
        RawCall::Err(e @ (errno::EPERM | errno::EACCES)) => SysCheck::Denied { errno: e },
        RawCall::Err(e) => SysCheck::Failed { errno: e },
    }
}

/// RFC-0010 §2.2's io_uring-allowed test: the sysctl permits this process to create rings and
/// `io_uring_setup` succeeded.
#[must_use]
pub fn io_uring_allowed(sysctl_permits: bool, setup: SysCheck) -> bool {
    sysctl_permits && setup.granted()
}

/// The real system calls.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LiveSyscalls;

impl Syscalls for LiveSyscalls {
    fn request_xtiledata_permission(&self) -> RawCall {
        live::request_xtiledata_permission()
    }

    fn io_uring_setup(&self) -> RawCall {
        live::io_uring_setup()
    }
}

#[cfg(target_os = "linux")]
mod live {
    use std::ffi::c_long;
    use std::os::fd::{FromRawFd, OwnedFd, RawFd};

    use super::RawCall;

    /// `__NR_arch_prctl`, which exists on x86-64 only.
    #[cfg(target_arch = "x86_64")]
    const SYS_ARCH_PRCTL: c_long = 158;
    /// `ARCH_REQ_XCOMP_PERM` (`asm/prctl.h`).
    #[cfg(target_arch = "x86_64")]
    const ARCH_REQ_XCOMP_PERM: c_long = 0x1023;
    /// `XFEATURE_XTILEDATA`: the AMX tile-data state component.
    #[cfg(target_arch = "x86_64")]
    const XFEATURE_XTILEDATA: c_long = 18;
    /// `__NR_io_uring_setup`: 425 in the unified numbering every Linux architecture shares for
    /// calls added since 5.1.
    const SYS_IO_URING_SETUP: c_long = 425;

    unsafe extern "C" {
        fn syscall(number: c_long, ...) -> c_long;
    }

    fn last_errno() -> i32 {
        std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
    }

    #[cfg(target_arch = "x86_64")]
    pub fn request_xtiledata_permission() -> RawCall {
        // SAFETY: `ARCH_REQ_XCOMP_PERM` takes the feature number by value and reads and writes no
        // user memory.
        let r = unsafe { syscall(SYS_ARCH_PRCTL, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) };
        if r == 0 {
            RawCall::Ok
        } else {
            RawCall::Err(last_errno())
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn request_xtiledata_permission() -> RawCall {
        RawCall::Absent
    }

    pub fn io_uring_setup() -> RawCall {
        // `struct io_uring_params` is 120 bytes: seven `u32` fields, `resv[3]`, and the two
        // 40-byte ring offset records. The kernel reads it, requires the reserved words to be
        // zero, and writes the ring geometry back. `u64` words give it the struct's alignment.
        let mut params = [0u64; 15];
        // SAFETY: `params` is a live, writable, suitably aligned 120-byte buffer for the whole
        // call, which is all `io_uring_setup` reads and writes besides its integer arguments.
        let r = unsafe { syscall(SYS_IO_URING_SETUP, 1 as c_long, params.as_mut_ptr()) };
        if r < 0 {
            return RawCall::Err(last_errno());
        }
        match RawFd::try_from(r) {
            Ok(fd) => {
                // SAFETY: a successful `io_uring_setup` returns a new file descriptor that
                // nothing else owns; the `OwnedFd` closes it when dropped here.
                drop(unsafe { OwnedFd::from_raw_fd(fd) });
                RawCall::Ok
            }
            // A descriptor outside `RawFd` cannot happen; report it as a failure, not success.
            Err(_) => RawCall::Err(super::errno::EINVAL),
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod live {
    use super::RawCall;

    pub fn request_xtiledata_permission() -> RawCall {
        RawCall::Absent
    }

    pub fn io_uring_setup() -> RawCall {
        RawCall::Absent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A shim that answers both calls with one fixed result.
    struct Shim(RawCall);

    impl Syscalls for Shim {
        fn request_xtiledata_permission(&self) -> RawCall {
            self.0
        }

        fn io_uring_setup(&self) -> RawCall {
            self.0
        }
    }

    #[test]
    fn amx_granted_denied_unsupported() {
        let check = |raw| amx_tile_permission(&Shim(raw));
        assert_eq!(check(RawCall::Ok), SysCheck::Granted);
        assert_eq!(
            check(RawCall::Err(errno::EPERM)),
            SysCheck::Denied { errno: 1 }
        );
        assert_eq!(
            check(RawCall::Err(errno::EACCES)),
            SysCheck::Denied { errno: 13 }
        );
        for e in [errno::EINVAL, errno::EOPNOTSUPP, errno::ENOSYS] {
            assert_eq!(
                check(RawCall::Err(e)),
                SysCheck::Unsupported { errno: Some(e) }
            );
        }
        assert_eq!(
            check(RawCall::Absent),
            SysCheck::Unsupported { errno: None }
        );
        // ENOSPC: the alternate signal stack is too small for the tile state.
        assert_eq!(check(RawCall::Err(28)), SysCheck::Failed { errno: 28 });
    }

    #[test]
    fn io_uring_granted_denied_unsupported() {
        let check = |raw| io_uring_setup(&Shim(raw));
        assert_eq!(check(RawCall::Ok), SysCheck::Granted);
        assert_eq!(
            check(RawCall::Err(errno::EPERM)),
            SysCheck::Denied { errno: 1 }
        );
        assert_eq!(
            check(RawCall::Err(errno::EACCES)),
            SysCheck::Denied { errno: 13 }
        );
        assert_eq!(
            check(RawCall::Err(errno::ENOSYS)),
            SysCheck::Unsupported { errno: Some(38) }
        );
        assert_eq!(
            check(RawCall::Absent),
            SysCheck::Unsupported { errno: None }
        );
        // EINVAL here is a bad parameter, not a missing feature; ENOMEM is a resource limit.
        assert_eq!(check(RawCall::Err(22)), SysCheck::Failed { errno: 22 });
        assert_eq!(check(RawCall::Err(12)), SysCheck::Failed { errno: 12 });
    }

    #[test]
    fn io_uring_allowed_needs_both_halves() {
        assert!(io_uring_allowed(true, SysCheck::Granted));
        assert!(!io_uring_allowed(false, SysCheck::Granted));
        assert!(!io_uring_allowed(true, SysCheck::Denied { errno: 1 }));
        assert!(!io_uring_allowed(
            true,
            SysCheck::Unsupported { errno: Some(38) }
        ));
        assert!(!io_uring_allowed(true, SysCheck::Failed { errno: 12 }));
    }

    /// The live calls never panic and give a classified result on every OS. Off Linux both are
    /// absent.
    #[test]
    fn live_calls_classify() {
        let amx = amx_tile_permission(&LiveSyscalls);
        let uring = io_uring_setup(&LiveSyscalls);
        if !cfg!(target_os = "linux") {
            assert_eq!(amx, SysCheck::Unsupported { errno: None });
            assert_eq!(uring, SysCheck::Unsupported { errno: None });
        }
        if cfg!(all(target_os = "linux", not(target_arch = "x86_64"))) {
            assert_eq!(amx, SysCheck::Unsupported { errno: None });
        }
    }
}
