// SPDX-License-Identifier: MIT OR Apache-2.0

//! Confinement of a short-lived child that reads hostile input: the header parse of a newly
//! added model (RFC-0034 §10.4, open question 3; DIST-E17-T16).
//!
//! [`confine_and_open`] is called by the child itself, first thing. It sets the limits, closes
//! what the child inherited, opens the one file the child may read, and then applies the OS
//! sandbox profile, so the child can open nothing more (DIST-E17-T08):
//! - **Resources.** On Linux, `setrlimit` caps the address space, CPU seconds, open files and core
//!   dumps, and is tested here. On macOS it caps CPU seconds, open files and core dumps, but not
//!   memory: macOS does not enforce the address-space limit, and it refuses the data-segment
//!   limit with `EINVAL` (hosted macOS CI, 2026-10-30), so the macOS sandbox profile of
//!   DIST-E17-T08 owns that limit. On Windows the process puts itself in a job with a
//!   per-process memory and user-time limit, and open files have no limit. The macOS and Windows
//!   limits are set but not confirmed enforced on those systems.
//! - **Inherited descriptors.** On Linux, every descriptor from 3 up is closed, so a socket or
//!   channel end the child inherited cannot be used.
//! - **System calls.** On Linux, a seccomp filter, installed with `no_new_privs`, makes `socket()`
//!   fail with `EACCES`, and fails with `EPERM`: `io_uring_setup()` (io_uring can open sockets
//!   itself), `ptrace`, `process_vm_readv`, `process_vm_writev`, `pidfd_open` and `pidfd_getfd`
//!   (each could reach the daemon's sockets). x32 system calls are refused and a foreign
//!   architecture kills the process. After the file is open the filter also fails every call that
//!   opens or creates a file (`open`, `openat`, `openat2`, `creat`) with `EACCES`, and `execve` and
//!   `execveat` with `EPERM`, every call that changes the file system (unlink, rename, mkdir,
//!   mknod, link, symlink, chmod, chown, truncate) with `EACCES`, and `pidfd_send_signal`; and
//!   `kill`, `tkill`, `tgkill`, `rt_sigqueueinfo` and `rt_tgsigqueueinfo` may name only the
//!   child's own pid, so it cannot signal the daemon. It needs no privilege, unlike a network namespace, which Ubuntu
//!   24.04 denies unprivileged processes by default. The process is also made non-dumpable.
//! - **Landlock** (Linux, DIST-E17-T08). Where the kernel has it, a ruleset that handles every
//!   file-system right up to ABI 6, the newest this code knows, and grants none, plus TCP bind and connect from ABI 4
//!   and the abstract-socket and signal scopes from ABI 6, so the child cannot reach the file
//!   system by any call the seccomp filter leaves, nor another process. A kernel without Landlock
//!   leaves the seccomp filter as the file ban, and [`Confined::landlock`] says so.
//! - **macOS sandbox profile** (DIST-E17-T08). Every inherited descriptor from 3 up is closed, then
//!   `sandbox_init` applies [`MACOS_PROFILE`]: file reads and writes, sockets and network, process
//!   execution and forks, Mach service lookups, IOKit, POSIX shared memory and semaphores, and
//!   signals to any process but itself are denied. The descriptors already open,
//!   the file and stdio, still work. macOS still enforces no memory limit, so
//!   [`Confined::memory`] is false there and the child sizes what it reads to the limit itself.

use std::fs::File;
use std::io;
use std::path::Path;

/// The macOS sandbox profile (SBPL) the child runs under: everything a parse of an already open
/// file needs is allowed, and everything that reaches outside it is denied.
pub const MACOS_PROFILE: &str = "(version 1)
(allow default)
(deny file-read* file-write* file-ioctl)
(deny network* system-socket)
(deny process-exec* process-fork)
(deny mach-lookup iokit-open)
(deny ipc-posix-shm* ipc-posix-sem*)
(deny signal)
(allow signal (target self))
";

/// The limits a confined child runs under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Address space (Linux) or process memory (Windows), in bytes. Not set on macOS (module
    /// docs).
    pub memory: u64,
    /// CPU time, in seconds.
    pub cpu_secs: u64,
    /// Open file descriptors (Linux and macOS).
    pub open_files: u64,
}

/// What [`confine_and_open`] put in place, for the caller to report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Confined {
    /// The resource limits are set: on macOS, all but memory (module docs).
    pub limits: bool,
    /// The OS enforces [`Limits::memory`]: Linux and Windows. On macOS the caller must keep its
    /// own use under the limit.
    pub memory: bool,
    /// New sockets are refused (Linux and macOS).
    pub network: bool,
    /// No file but the one opened can be opened (Linux and macOS).
    pub files: bool,
    /// The Landlock ABI version the ruleset was built for; 0 where there is none (module docs).
    pub landlock: u32,
}

/// Confines the calling process and opens `path`, the one file it may read, for reading (module
/// docs). Call it once, in a child, before reading any hostile input: it cannot be undone.
///
/// # Errors
///
/// The OS error of the first step that fails, or the file's open error; the caller must then
/// refuse to read the input.
pub fn confine_and_open(limits: &Limits, path: &Path) -> io::Result<(Confined, File)> {
    imp::limit(limits)?;
    imp::sandbox(path)
}

/// Sends `SIGTERM` to process `pid`: for the header child's `--check-signal`, which must be
/// refused under the profile (Unix only).
///
/// # Errors
///
/// The OS error, `EPERM` under the profile.
#[cfg(unix)]
pub fn terminate(pid: u32) -> io::Result<()> {
    let pid = libc::pid_t::try_from(pid)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "pid out of range"))?;
    // SAFETY: `kill` reads nothing from memory.
    if unsafe { libc::kill(pid, libc::SIGTERM) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Creates and removes a POSIX shared-memory object: for the header child's `--check-shm`, which
/// must be refused under the profile (Unix only).
///
/// # Errors
///
/// The OS error of `shm_open`, `EPERM` or `EACCES` under the profile.
#[cfg(unix)]
pub fn probe_shm() -> io::Result<()> {
    let name = std::ffi::CString::new(format!("/syntropy-probe-{}", std::process::id()))
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    // SAFETY: `name` is NUL-terminated and outlives the call.
    let fd = unsafe {
        libc::shm_open(
            name.as_ptr(),
            libc::O_CREAT | libc::O_EXCL | libc::O_RDWR,
            0o600,
        )
    };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: closing the descriptor just opened and removing the name just created.
    unsafe {
        libc::close(fd);
        libc::shm_unlink(name.as_ptr());
    }
    Ok(())
}

/// The Landlock ABI version this kernel offers; 0 where it has none (Linux only).
#[cfg(target_os = "linux")]
#[must_use]
pub fn landlock_abi() -> u32 {
    imp::landlock::abi()
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
mod imp {
    use std::fs::File;
    use std::io;
    use std::path::Path;

    use super::{Confined, Limits};

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
    pub(super) fn sandbox(path: &Path) -> io::Result<(Confined, File)> {
        // The descriptors this process inherited, listed before any is closed. `read_dir` holds
        // one of its own while it lists, which is closed again by the time the loop runs.
        let inherited: Vec<libc::c_int> = std::fs::read_dir("/dev/fd")?
            .filter_map(|entry| entry.ok()?.file_name().to_str()?.parse().ok())
            .filter(|fd| *fd > 2)
            .collect();
        for fd in inherited {
            // SAFETY: closing a descriptor this process holds and nothing in it uses: the child
            // opens its own file only below. A descriptor already closed fails with `EBADF`,
            // which is ignored.
            unsafe { libc::close(fd) };
        }
        let file = File::open(path)?;
        let profile = std::ffi::CString::new(super::MACOS_PROFILE)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        let mut error: *mut libc::c_char = std::ptr::null_mut();
        // SAFETY: `profile` is NUL-terminated and outlives the call; flags 0 reads it as SBPL text;
        // `error` is written only on failure.
        let rc = unsafe { sandbox_init(profile.as_ptr(), 0, &raw mut error) };
        if rc != 0 {
            let reason = if error.is_null() {
                "no reason given".to_owned()
            } else {
                // SAFETY: on failure `error` is a NUL-terminated string the call allocated, read
                // once and freed once here.
                let text = unsafe { std::ffi::CStr::from_ptr(error) }
                    .to_string_lossy()
                    .into_owned();
                // SAFETY: as above.
                unsafe { sandbox_free_error(error) };
                text
            };
            return Err(io::Error::other(format!("sandbox_init: {reason}")));
        }
        Ok((
            Confined {
                limits: true,
                memory: false,
                network: true,
                files: true,
                landlock: 0,
            },
            file,
        ))
    }

    // SAFETY: the declarations of `<sandbox.h>`, which libSystem exports.
    #[cfg(target_os = "macos")]
    unsafe extern "C" {
        fn sandbox_init(
            profile: *const libc::c_char,
            flags: u64,
            errorbuf: *mut *mut libc::c_char,
        ) -> libc::c_int;
        fn sandbox_free_error(errorbuf: *mut libc::c_char);
    }

    #[cfg(target_os = "linux")]
    pub(super) fn sandbox(path: &Path) -> io::Result<(Confined, File)> {
        // SAFETY: closing descriptors this process holds; nothing in it uses them afterwards, as
        // the child opens its own file only below.
        if unsafe { libc::close_range(3, libc::c_uint::MAX, 0) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let file = File::open(path)?;
        // SAFETY: `prctl` with these arguments reads nothing from memory.
        if unsafe { libc::prctl(libc::PR_SET_DUMPABLE, 0 as libc::c_ulong, 0, 0, 0) } != 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: as above. Landlock and seccomp both need `no_new_privs`.
        if unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1 as libc::c_ulong, 0, 0, 0) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let landlock = landlock::restrict()?;
        seccomp::install()?;
        Ok((
            Confined {
                limits: true,
                memory: true,
                network: true,
                files: true,
                landlock,
            },
            file,
        ))
    }

    #[cfg(target_os = "linux")]
    pub(super) mod landlock {
        //! Landlock through its three system calls (`linux/landlock.h`), with no crate.

        use std::io;

        /// `landlock_create_ruleset`'s flag that asks for the ABI version.
        const CREATE_RULESET_VERSION: libc::c_uint = 1;

        /// The newest ABI whose rights this code knows. A newer kernel is asked for this ABI's
        /// rights and no others: what a later ABI adds is not handled, so it stays allowed, and
        /// [`Confined::landlock`](crate::os::confine::Confined::landlock) reports this version.
        pub(crate) const KNOWN_ABI: u32 = 6;

        /// The file-system rights each ABI version handles, cumulatively: ABI 1 bits 0–12
        /// (execute … make symlink), ABI 2 `REFER`, ABI 3 `TRUNCATE`, ABI 5 `IOCTL_DEV`.
        #[must_use]
        pub(crate) fn fs_rights(abi: u32) -> u64 {
            let mut rights = (1u64 << 13) - 1;
            if abi >= 2 {
                rights |= 1 << 13;
            }
            if abi >= 3 {
                rights |= 1 << 14;
            }
            if abi >= 5 {
                rights |= 1 << 15;
            }
            rights
        }

        /// The network rights from ABI 4: TCP bind and connect.
        #[must_use]
        pub(crate) fn net_rights(abi: u32) -> u64 {
            if abi >= 4 { 0b11 } else { 0 }
        }

        /// The scopes from ABI 6: abstract Unix sockets and signals to processes outside the
        /// domain.
        #[must_use]
        pub(crate) fn scopes(abi: u32) -> u64 {
            if abi >= 6 { 0b11 } else { 0 }
        }

        /// `struct landlock_ruleset_attr`, ABI 6's layout.
        #[repr(C)]
        struct RulesetAttr {
            handled_access_fs: u64,
            handled_access_net: u64,
            scoped: u64,
        }

        /// The kernel's Landlock ABI version; 0 when it has none (not built, or disabled).
        #[must_use]
        pub(crate) fn abi() -> u32 {
            // SAFETY: a null attribute with size 0 and the version flag reads no memory.
            let rc = unsafe {
                libc::syscall(
                    libc::SYS_landlock_create_ruleset,
                    std::ptr::null::<RulesetAttr>(),
                    0usize,
                    CREATE_RULESET_VERSION,
                )
            };
            u32::try_from(rc).unwrap_or(0)
        }

        /// Restricts this process to a ruleset that handles every right of the kernel's ABI and
        /// grants none. Returns the ABI it used, or 0 when the kernel has no Landlock. Needs
        /// `no_new_privs`.
        /// The ABI the ruleset is built for: the kernel's, clamped to [`KNOWN_ABI`]. A newer
        /// kernel accepts an older ABI's ruleset, so the clamp is a deliberate choice: the rights
        /// this code knows are handled, and whatever a newer ABI adds stays as the kernel allows
        /// it until this code learns it.
        pub(crate) fn effective_abi(kernel: u32) -> u32 {
            kernel.min(KNOWN_ABI)
        }

        pub(super) fn restrict() -> io::Result<u32> {
            let abi = effective_abi(abi());
            if abi == 0 {
                return Ok(0);
            }
            let attr = RulesetAttr {
                handled_access_fs: fs_rights(abi),
                handled_access_net: net_rights(abi),
                scoped: scopes(abi),
            };
            // The struct as long as this ABI knows it, so an older kernel sees no field it lacks.
            let size = match abi {
                1..=3 => 8,
                4 | 5 => 16,
                _ => size_of::<RulesetAttr>(),
            };
            // SAFETY: `attr` lives for the call and is at least `size` bytes.
            let fd = unsafe {
                libc::syscall(
                    libc::SYS_landlock_create_ruleset,
                    &raw const attr,
                    size,
                    0 as libc::c_uint,
                )
            };
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            let fd = fd as libc::c_int;
            // SAFETY: `fd` is the ruleset just created; flags must be 0.
            let rc =
                unsafe { libc::syscall(libc::SYS_landlock_restrict_self, fd, 0 as libc::c_uint) };
            let restrict = io::Error::last_os_error();
            // SAFETY: closing the ruleset's descriptor, which nothing else holds.
            unsafe { libc::close(fd) };
            if rc != 0 {
                return Err(restrict);
            }
            Ok(abi)
        }
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
        /// The low 32 bits of the first argument, `seccomp_data.args[0]`, on a little-endian
        /// machine. A pid is a 32-bit `pid_t`, which is all the kernel reads of it.
        const ARG0_LOW: u32 = 16;

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

        /// The calls the filter fails, with the errno each gets.
        fn denied() -> Vec<(u32, i32)> {
            let calls = vec![
                // No network: io_uring can open sockets itself.
                (libc::SYS_socket, libc::EACCES),
                (libc::SYS_io_uring_setup, libc::EPERM),
                // No reaching another process's memory or descriptors.
                (libc::SYS_ptrace, libc::EPERM),
                (libc::SYS_process_vm_readv, libc::EPERM),
                (libc::SYS_process_vm_writev, libc::EPERM),
                (libc::SYS_pidfd_open, libc::EPERM),
                (libc::SYS_pidfd_getfd, libc::EPERM),
                // No file but the one already open (DIST-E17-T08).
                (libc::SYS_openat, libc::EACCES),
                (libc::SYS_openat2, libc::EACCES),
                (libc::SYS_execve, libc::EPERM),
                (libc::SYS_execveat, libc::EPERM),
                // No changing the file system, for a kernel without Landlock.
                (libc::SYS_unlinkat, libc::EACCES),
                (libc::SYS_renameat2, libc::EACCES),
                (libc::SYS_mkdirat, libc::EACCES),
                (libc::SYS_mknodat, libc::EACCES),
                (libc::SYS_symlinkat, libc::EACCES),
                (libc::SYS_linkat, libc::EACCES),
                (libc::SYS_fchmodat, libc::EACCES),
                (libc::SYS_fchownat, libc::EACCES),
                (libc::SYS_fchmod, libc::EACCES),
                (libc::SYS_fchown, libc::EACCES),
                (libc::SYS_truncate, libc::EACCES),
                // The child only reads its file and writes stdout and stderr, so it truncates
                // nothing.
                (libc::SYS_ftruncate, libc::EACCES),
                // No signal by pidfd; the other signal calls are checked below.
                (libc::SYS_pidfd_send_signal, libc::EPERM),
            ];
            // aarch64 has only the `at` (and `renameat2`) forms of these.
            #[cfg(target_arch = "x86_64")]
            let calls = [
                calls,
                vec![
                    (libc::SYS_open, libc::EACCES),
                    (libc::SYS_creat, libc::EACCES),
                    (libc::SYS_unlink, libc::EACCES),
                    (libc::SYS_rmdir, libc::EACCES),
                    (libc::SYS_rename, libc::EACCES),
                    (libc::SYS_renameat, libc::EACCES),
                    (libc::SYS_mkdir, libc::EACCES),
                    (libc::SYS_mknod, libc::EACCES),
                    (libc::SYS_symlink, libc::EACCES),
                    (libc::SYS_link, libc::EACCES),
                    (libc::SYS_chmod, libc::EACCES),
                    (libc::SYS_chown, libc::EACCES),
                    (libc::SYS_lchown, libc::EACCES),
                ],
            ]
            .concat();
            calls
                .into_iter()
                .map(|(nr, errno)| (nr as u32, errno))
                .collect()
        }

        pub(super) fn install() -> io::Result<()> {
            let errno = |e: i32| libc::SECCOMP_RET_ERRNO | (e as u32 & libc::SECCOMP_RET_DATA);
            let mut filter = vec![
                // A foreign architecture: kill, since its numbers mean other calls.
                op(LD_W_ABS, 0, 0, ARCH),
                op(JEQ_K, 1, 0, AUDIT_ARCH),
                op(RET_K, 0, 0, libc::SECCOMP_RET_KILL_PROCESS),
                op(LD_W_ABS, 0, 0, NR),
                op(JGE_K, 0, 1, X32_BIT),
                op(RET_K, 0, 0, errno(libc::EPERM)),
            ];
            for (nr, e) in denied() {
                filter.push(op(JEQ_K, 0, 1, nr));
                filter.push(op(RET_K, 0, 0, errno(e)));
            }
            // Signals only to this process: the first argument, the target pid or thread-group
            // id, must be its own pid (the child has one thread, so its tid is its pid). That
            // refuses the daemon and every other process, and pid 0 and negative pids, which
            // name process groups. `abort` signals itself with `tgkill` and still works.
            // SAFETY: `getpid` reads nothing from memory.
            let own = unsafe { libc::getpid() } as u32;
            for nr in [
                libc::SYS_kill,
                libc::SYS_tkill,
                libc::SYS_tgkill,
                libc::SYS_rt_sigqueueinfo,
                libc::SYS_rt_tgsigqueueinfo,
            ] {
                filter.push(op(JEQ_K, 0, 4, nr as u32));
                filter.push(op(LD_W_ABS, 0, 0, ARG0_LOW));
                filter.push(op(JEQ_K, 1, 0, own));
                filter.push(op(RET_K, 0, 0, errno(libc::EPERM)));
                filter.push(op(RET_K, 0, 0, libc::SECCOMP_RET_ALLOW));
            }
            filter.push(op(RET_K, 0, 0, libc::SECCOMP_RET_ALLOW));
            let program = libc::sock_fprog {
                len: filter.len() as u16,
                filter: filter.as_ptr().cast_mut(),
            };
            // `no_new_privs` is set by the caller. SAFETY: `program` and the filter it points to live until the call returns; the kernel
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

    use std::fs::File;
    use std::path::Path;

    use super::{Confined, Limits};

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

    /// Windows has no profile yet: the AppContainer the child must be started in is not built
    /// (module docs). The file is opened, and the limits stand.
    pub(super) fn sandbox(path: &Path) -> io::Result<(Confined, File)> {
        Ok((
            Confined {
                limits: true,
                memory: true,
                network: false,
                files: false,
                landlock: 0,
            },
            File::open(path)?,
        ))
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

    pub(super) fn sandbox(_: &Path) -> io::Result<(super::Confined, std::fs::File)> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "no confinement on this platform",
        ))
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::imp::landlock::{KNOWN_ABI, effective_abi, fs_rights, net_rights, scopes};

    /// A kernel newer than the code is asked for the newest ABI the code knows, never more; an
    /// older one for its own.
    #[test]
    fn landlock_abi_is_clamped_to_the_known_one() {
        assert_eq!(KNOWN_ABI, 6);
        assert_eq!(effective_abi(0), 0);
        assert_eq!(effective_abi(3), 3);
        assert_eq!(effective_abi(6), 6);
        assert_eq!(effective_abi(8), 6);
        assert_eq!(effective_abi(u32::MAX), 6);
    }

    /// Each ABI handles every right it defines, and no right it does not, which the kernel
    /// would refuse with `EINVAL`.
    #[test]
    fn landlock_rights_follow_the_abi() {
        assert_eq!(fs_rights(1), 0x1FFF);
        assert_eq!(fs_rights(2), 0x3FFF);
        assert_eq!(fs_rights(3), 0x7FFF);
        assert_eq!(fs_rights(4), 0x7FFF);
        assert_eq!(fs_rights(5), 0xFFFF);
        assert_eq!(fs_rights(6), 0xFFFF);
        assert_eq!((net_rights(3), net_rights(4)), (0, 0b11));
        assert_eq!((scopes(5), scopes(6)), (0, 0b11));
    }

    /// The kernel's ABI, printed for the evidence. On a kernel that lists Landlock among its
    /// security modules the version is at least 1.
    #[test]
    fn the_kernel_reports_its_landlock_abi() {
        let abi = super::landlock_abi();
        println!("landlock ABI {abi}");
        let lsm = std::fs::read_to_string("/sys/kernel/security/lsm").unwrap_or_default();
        if lsm.split(',').any(|m| m.trim() == "landlock") {
            assert!(abi >= 1, "{lsm}");
        }
    }
}
