// SPDX-License-Identifier: MIT OR Apache-2.0

//! Confinement of a short-lived child that reads hostile input: the header parse of a newly
//! added model (RFC-0034 §10.4, open question 3; DIST-E17-T16).
//!
//! [`confine_self`] is called by the child itself, first thing, before it opens the file:
//! - **Resources.** On Linux, `setrlimit` caps the address space, CPU seconds, open files and core
//!   dumps, and is tested here. On macOS it sets the data segment (macOS does not enforce the
//!   address-space limit), CPU seconds, open files and core dumps; on Windows the process puts
//!   itself in a job with a per-process memory and user-time limit, and open files have no
//!   limit. The macOS and Windows limits are set but not confirmed enforced on those systems.
//! - **Inherited descriptors.** On Linux, every descriptor from 3 up is closed, so a socket or
//!   channel end the child inherited cannot be used.
//! - **System calls.** On Linux, a seccomp filter, installed with `no_new_privs`, makes `socket()`
//!   fail with `EACCES`, and fails with `EPERM`: `io_uring_setup()` (io_uring can open sockets
//!   itself), `ptrace`, `process_vm_readv`, `process_vm_writev`, `pidfd_open` and `pidfd_getfd`
//!   (each could reach the daemon's sockets). x32 system calls are refused and a foreign
//!   architecture kills the process. It is a deny list, not the allow list and landlock of the M4
//!   sandbox (DIST-E17-T08). It needs no privilege, unlike a network namespace, which Ubuntu 24.04
//!   denies unprivileged processes by default. The process is also made non-dumpable. On macOS and
//!   Windows there is no network ban yet: a macOS sandbox profile and an AppContainer are
//!   DIST-E17-T08's.

use std::io;

/// The limits a confined child runs under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Address space (Linux), data segment (macOS) or process memory (Windows), in bytes.
    pub memory: u64,
    /// CPU time, in seconds.
    pub cpu_secs: u64,
    /// Open file descriptors (Linux and macOS).
    pub open_files: u64,
}

/// What [`confine_self`] put in place, for the caller to report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Confined {
    /// The resource limits are set.
    pub limits: bool,
    /// New sockets are refused (Linux only, for now).
    pub network: bool,
}

/// Confines the calling process (module docs). Call it once, in a child, before reading any
/// hostile input: it cannot be undone.
///
/// # Errors
///
/// The OS error of the first step that fails; the caller must then refuse to read the input.
pub fn confine_self(limits: &Limits) -> io::Result<Confined> {
    imp::limit(limits)?;
    let network = imp::deny_network()?;
    Ok(Confined {
        limits: true,
        network,
    })
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
mod imp {
    use std::io;

    use super::Limits;

    /// Sets `resource`'s soft and hard limits to `value`. A failure names the resource, so a
    /// platform that refuses one says which.
    fn set(name: &str, resource: libc::c_int, value: u64) -> io::Result<()> {
        let limit = libc::rlimit {
            rlim_cur: value as libc::rlim_t,
            rlim_max: value as libc::rlim_t,
        };
        // SAFETY: `setrlimit` reads the struct it is given, which lives for the call. The
        // resource's type differs by platform, hence the cast.
        let rc = unsafe { libc::setrlimit(resource as _, &limit) };
        if rc == 0 {
            Ok(())
        } else {
            let e = io::Error::last_os_error();
            Err(io::Error::new(
                e.kind(),
                format!("setrlimit({name}, {value}): {e}"),
            ))
        }
    }

    pub(super) fn limit(limits: &Limits) -> io::Result<()> {
        // No core dump of hostile content.
        set("RLIMIT_CORE", libc::RLIMIT_CORE as libc::c_int, 0)?;
        #[cfg(target_os = "linux")]
        set("RLIMIT_AS", libc::RLIMIT_AS as libc::c_int, limits.memory)?;
        // macOS does not enforce RLIMIT_AS; the data segment is what it does enforce.
        #[cfg(target_os = "macos")]
        set("RLIMIT_DATA", libc::RLIMIT_DATA, limits.memory)?;
        set(
            "RLIMIT_CPU",
            libc::RLIMIT_CPU as libc::c_int,
            limits.cpu_secs,
        )?;
        set(
            "RLIMIT_NOFILE",
            libc::RLIMIT_NOFILE as libc::c_int,
            limits.open_files,
        )?;
        Ok(())
    }

    #[cfg(target_os = "macos")]
    pub(super) fn deny_network() -> io::Result<bool> {
        Ok(false)
    }

    #[cfg(target_os = "linux")]
    pub(super) fn deny_network() -> io::Result<bool> {
        // SAFETY: closing descriptors this process holds; nothing in it uses them afterwards, as
        // the child opens its file only after confinement.
        if unsafe { libc::close_range(3, libc::c_uint::MAX, 0) } != 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `prctl` with these arguments reads nothing from memory.
        if unsafe { libc::prctl(libc::PR_SET_DUMPABLE, 0 as libc::c_ulong, 0, 0, 0) } != 0 {
            return Err(io::Error::last_os_error());
        }
        seccomp::install()?;
        Ok(true)
    }

    #[cfg(target_os = "linux")]
    mod seccomp {
        use std::io;

        // Classic BPF opcodes and `struct seccomp_data` offsets (linux/filter.h, linux/seccomp.h).
        const LD_W_ABS: u16 = 0x20;
        const JEQ_K: u16 = 0x15;
        const JGE_K: u16 = 0x35;
        const RET_K: u16 = 0x06;
        const NR: u32 = 0;
        const ARCH: u32 = 4;

        #[cfg(target_arch = "x86_64")]
        const AUDIT_ARCH: u32 = 0xc000_003e;
        #[cfg(target_arch = "aarch64")]
        const AUDIT_ARCH: u32 = 0xc000_00b7;
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        compile_error!("the header child's seccomp filter knows x86_64 and aarch64 only");
        /// x32 system calls carry this bit; refusing them closes the x32 path to `socket`.
        const X32_BIT: u32 = 0x4000_0000;

        const fn op(code: u16, jt: u8, jf: u8, k: u32) -> libc::sock_filter {
            libc::sock_filter { code, jt, jf, k }
        }

        pub(super) fn install() -> io::Result<()> {
            let errno = |e: i32| libc::SECCOMP_RET_ERRNO | (e as u32 & libc::SECCOMP_RET_DATA);
            let filter = [
                // A foreign architecture: kill, since its numbers mean other calls.
                op(LD_W_ABS, 0, 0, ARCH),
                op(JEQ_K, 1, 0, AUDIT_ARCH),
                op(RET_K, 0, 0, libc::SECCOMP_RET_KILL_PROCESS),
                op(LD_W_ABS, 0, 0, NR),
                op(JGE_K, 0, 1, X32_BIT),
                op(RET_K, 0, 0, errno(libc::EPERM)),
                op(JEQ_K, 0, 1, libc::SYS_socket as u32),
                op(RET_K, 0, 0, errno(libc::EACCES)),
                op(JEQ_K, 5, 0, libc::SYS_io_uring_setup as u32),
                op(JEQ_K, 4, 0, libc::SYS_ptrace as u32),
                op(JEQ_K, 3, 0, libc::SYS_process_vm_readv as u32),
                op(JEQ_K, 2, 0, libc::SYS_process_vm_writev as u32),
                op(JEQ_K, 1, 0, libc::SYS_pidfd_open as u32),
                op(JEQ_K, 0, 1, libc::SYS_pidfd_getfd as u32),
                op(RET_K, 0, 0, errno(libc::EPERM)),
                op(RET_K, 0, 0, libc::SECCOMP_RET_ALLOW),
            ];
            let program = libc::sock_fprog {
                len: filter.len() as u16,
                filter: filter.as_ptr().cast_mut(),
            };
            // SAFETY: `prctl` with these arguments reads nothing from memory.
            if unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1 as libc::c_ulong, 0, 0, 0) } != 0 {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: `program` and the filter it points to live until the call returns; the kernel
            // copies the filter.
            let rc = unsafe {
                libc::prctl(
                    libc::PR_SET_SECCOMP,
                    libc::SECCOMP_MODE_FILTER as libc::c_ulong,
                    &program as *const libc::sock_fprog,
                    0,
                    0,
                )
            };
            if rc == 0 {
                Ok(())
            } else {
                Err(io::Error::last_os_error())
            }
        }
    }
}

#[cfg(windows)]
mod imp {
    use std::io;

    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_PROCESS_MEMORY,
        JOB_OBJECT_LIMIT_PROCESS_TIME, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JobObjectExtendedLimitInformation, SetInformationJobObject,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    use super::Limits;

    pub(super) fn limit(limits: &Limits) -> io::Result<()> {
        // SAFETY: an unnamed job with default security; the handle is checked.
        let job = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if job.is_null() {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: all-zero is a valid `JOBOBJECT_EXTENDED_LIMIT_INFORMATION`.
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        info.BasicLimitInformation.LimitFlags =
            JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_PROCESS_TIME;
        // Process time is in 100 ns units.
        info.BasicLimitInformation.PerProcessUserTimeLimit =
            i64::try_from(limits.cpu_secs.saturating_mul(10_000_000)).unwrap_or(i64::MAX);
        info.ProcessMemoryLimit = usize::try_from(limits.memory).unwrap_or(usize::MAX);
        // SAFETY: `info` is the struct the class names, and lives for the call.
        let set = unsafe {
            SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                (&raw const info).cast(),
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        };
        if set == 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: the job handle is valid, and the pseudo-handle names this process. The job's
        // handle stays open for the process's life, which keeps the job.
        if unsafe { AssignProcessToJobObject(job, GetCurrentProcess()) } == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    pub(super) fn deny_network() -> io::Result<bool> {
        Ok(false)
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
mod imp {
    use std::io;

    use super::Limits;

    pub(super) fn limit(_: &Limits) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "no confinement on this platform",
        ))
    }

    pub(super) fn deny_network() -> io::Result<bool> {
        Ok(false)
    }
}
