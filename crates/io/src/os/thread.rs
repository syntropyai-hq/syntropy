// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hot-thread placement: the cores a process may run on with their NUMA nodes, pinning the
//! calling thread to one, and raising its priority as far as the OS allows without elevated
//! privileges (ADR-0012 item 2; RFC-0012 §1.1; PLC-E7-T01).
//!
//! `exec` places its dispatcher threads with these, and the runner its I/O threads. They take OS
//! calls that need `unsafe`, which RFC-0001 §4.2 rule 8 keeps here.
//!
//! | | Linux, Android | Windows | macOS, iOS |
//! |---|---|---|---|
//! | [`cpu_topology`] | the process's affinity mask (`sched_getaffinity`), NUMA nodes from `/sys/devices/system/node` | the process's affinity mask in processor group 0, NUMA nodes from `GetNumaProcessorNode` | every core, all on node 0 (unified memory) |
//! | [`pin_current_thread`] | `sched_setaffinity` | `SetThreadAffinityMask` | [`PinError::Unsupported`]: the OS has no thread-to-core affinity |
//! | [`raise_current_thread_priority`] | the lowest nice value `RLIMIT_NICE` allows, if below the current one | `THREAD_PRIORITY_TIME_CRITICAL` | QoS class `USER_INTERACTIVE` |
//! | [`current_core`] | `sched_getcpu` | `GetCurrentProcessorNumber` | `None` |

use std::io;

/// A logical CPU, numbered as the OS numbers it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CoreId(pub u32);

/// A core this process may run on, and the NUMA node it belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Core {
    /// The core.
    pub id: CoreId,
    /// Its NUMA node: 0 on a host with one, or where the OS reports none.
    pub numa_node: u16,
}

/// Why a thread was not pinned.
#[derive(Debug)]
pub enum PinError {
    /// The OS has no way to bind a thread to a core: macOS and iOS.
    Unsupported,
    /// The core cannot be named in the call, such as a core outside processor group 0 on Windows.
    OutOfRange(CoreId),
    /// The OS refused, for example because the core is outside the process's affinity mask.
    Os(io::Error),
}

impl std::fmt::Display for PinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PinError::Unsupported => f.write_str("this OS cannot bind a thread to a core"),
            PinError::OutOfRange(core) => write!(f, "core {} cannot be named here", core.0),
            PinError::Os(error) => write!(f, "the OS refused to pin the thread: {error}"),
        }
    }
}

impl std::error::Error for PinError {}

/// The scheduling priority the calling thread ended up with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Priority {
    /// A Linux or Android nice value: lower runs first. `RLIMIT_NICE` bounds how low an
    /// unprivileged thread may set it, and its usual value keeps it at 0.
    Nice(i32),
    /// A Windows thread priority level, such as `THREAD_PRIORITY_TIME_CRITICAL` (15).
    Level(i32),
    /// The Apple QoS class `USER_INTERACTIVE`, the highest an unprivileged thread may ask for.
    UserInteractive,
    /// The OS refused to raise it, so the thread keeps the priority it started with.
    Unchanged,
}

/// The cores this process may run on, in core order, each with its NUMA node.
///
/// # Errors
///
/// If the OS cannot report the process's affinity mask.
pub fn cpu_topology() -> io::Result<Vec<Core>> {
    platform::cpu_topology()
}

/// Binds the calling thread to `core`.
///
/// # Errors
///
/// [`PinError::Unsupported`] where the OS cannot, and [`PinError::Os`] when it refuses.
pub fn pin_current_thread(core: CoreId) -> Result<(), PinError> {
    platform::pin_current_thread(core)
}

/// The core the calling thread is running on, where the OS can say.
#[must_use]
pub fn current_core() -> Option<CoreId> {
    platform::current_core()
}

/// Raises the calling thread's priority to the highest the OS grants without elevated
/// privileges (ADR-0012 item 2), and returns the priority it ended up with.
#[must_use]
pub fn raise_current_thread_priority() -> Priority {
    platform::raise_current_thread_priority()
}

/// The CPUs of a Linux `cpulist`, such as `0-3,8,10-11`. A malformed entry is skipped.
#[cfg_attr(not(any(target_os = "linux", target_os = "android")), allow(dead_code))]
fn parse_cpulist(text: &str) -> Vec<u32> {
    let mut cpus = Vec::new();
    for part in text.trim().split(',').filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((first, last)) => {
                if let (Ok(first), Ok(last)) = (first.parse::<u32>(), last.parse::<u32>()) {
                    cpus.extend(first..=last);
                }
            }
            None => {
                if let Ok(cpu) = part.parse::<u32>() {
                    cpus.push(cpu);
                }
            }
        }
    }
    cpus
}

#[cfg(any(target_os = "linux", target_os = "android"))]
mod platform {
    use std::collections::BTreeMap;
    use std::io;
    use std::mem::MaybeUninit;
    use std::path::Path;

    use super::{Core, CoreId, PinError, Priority, parse_cpulist};

    /// The process's affinity mask: its main thread's, read by process id. Pid 0 would read the
    /// calling thread's, which is one core once that thread is pinned.
    fn affinity() -> io::Result<Vec<u32>> {
        let pid = libc::pid_t::try_from(std::process::id()).map_err(io::Error::other)?;
        let mut set = MaybeUninit::<libc::cpu_set_t>::zeroed();
        // SAFETY: `set` is a zeroed, writable `cpu_set_t` of the size passed, valid for the call,
        // and `sched_getaffinity` writes only through it.
        let rc =
            unsafe { libc::sched_getaffinity(pid, size_of::<libc::cpu_set_t>(), set.as_mut_ptr()) };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `sched_getaffinity` returned 0, and the set was zeroed before the call, so
        // every bit of it is initialised.
        let set = unsafe { set.assume_init() };
        let bits = u32::try_from(8 * size_of::<libc::cpu_set_t>()).unwrap_or(u32::MAX);
        Ok((0..bits)
            .filter(|&cpu| {
                // SAFETY: `cpu` is below the set's size in bits, which `CPU_ISSET` requires.
                unsafe { libc::CPU_ISSET(cpu as usize, &set) }
            })
            .collect())
    }

    /// Each CPU's NUMA node, from `node<N>/cpulist` under `root` (`/sys/devices/system/node`).
    pub(super) fn numa_of(root: &Path) -> BTreeMap<u32, u16> {
        let mut nodes = BTreeMap::new();
        let Ok(entries) = std::fs::read_dir(root) else {
            return nodes;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(node) = name
                .to_str()
                .and_then(|n| n.strip_prefix("node"))
                .and_then(|n| n.parse::<u16>().ok())
            else {
                continue;
            };
            if let Ok(list) = std::fs::read_to_string(entry.path().join("cpulist")) {
                for cpu in parse_cpulist(&list) {
                    nodes.insert(cpu, node);
                }
            }
        }
        nodes
    }

    pub(super) fn cpu_topology() -> io::Result<Vec<Core>> {
        let nodes = numa_of(Path::new("/sys/devices/system/node"));
        Ok(affinity()?
            .into_iter()
            .map(|cpu| Core {
                id: CoreId(cpu),
                numa_node: nodes.get(&cpu).copied().unwrap_or(0),
            })
            .collect())
    }

    pub(super) fn pin_current_thread(core: CoreId) -> Result<(), PinError> {
        let bits = 8 * size_of::<libc::cpu_set_t>();
        let cpu = core.0 as usize;
        if cpu >= bits {
            return Err(PinError::OutOfRange(core));
        }
        let set = MaybeUninit::<libc::cpu_set_t>::zeroed();
        // SAFETY: the set is zeroed, so every bit is initialised; `cpu` is below its size in
        // bits, which `CPU_SET` requires.
        let set = unsafe {
            let mut set = set.assume_init();
            libc::CPU_SET(cpu, &mut set);
            set
        };
        // SAFETY: `set` is a valid `cpu_set_t` of the size passed, read only for the call. Pid 0
        // is the calling thread.
        let rc = unsafe { libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set) };
        if rc == 0 {
            Ok(())
        } else {
            Err(PinError::Os(io::Error::last_os_error()))
        }
    }

    pub(super) fn current_core() -> Option<CoreId> {
        // SAFETY: `sched_getcpu` takes no arguments and touches no memory of the caller.
        let cpu = unsafe { libc::sched_getcpu() };
        u32::try_from(cpu).ok().map(CoreId)
    }

    /// The calling thread's id, which `setpriority` takes on Linux to act on one thread.
    fn tid() -> libc::id_t {
        // SAFETY: `SYS_gettid` takes no arguments and cannot fail.
        let tid = unsafe { libc::syscall(libc::SYS_gettid) };
        libc::id_t::try_from(tid).unwrap_or(0)
    }

    /// The calling thread's nice value.
    fn nice_of(tid: libc::id_t) -> Option<i32> {
        // `getpriority` may return -1 as a value, so `errno` tells success from failure.
        // SAFETY: `errno_location` returns the calling thread's `errno`, valid while it runs.
        unsafe { *errno_location() = 0 };
        // SAFETY: `PRIO_PROCESS` with a thread id reads that thread's nice value, and touches no
        // memory of the caller.
        let nice = unsafe { libc::getpriority(libc::PRIO_PROCESS, tid) };
        // SAFETY: as above.
        let errno = unsafe { *errno_location() };
        (errno == 0).then_some(nice)
    }

    #[cfg(target_os = "linux")]
    fn errno_location() -> *mut libc::c_int {
        // SAFETY: `__errno_location` takes no arguments and returns the thread's `errno`.
        unsafe { libc::__errno_location() }
    }

    #[cfg(target_os = "android")]
    fn errno_location() -> *mut libc::c_int {
        // SAFETY: `__errno` takes no arguments and returns the thread's `errno`.
        unsafe { libc::__errno() }
    }

    pub(super) fn raise_current_thread_priority() -> Priority {
        let tid = tid();
        let Some(current) = nice_of(tid) else {
            return Priority::Unchanged;
        };
        let mut limit = MaybeUninit::<libc::rlimit>::uninit();
        // SAFETY: `limit` is writable for the call, and `getrlimit` writes only through it.
        let rc = unsafe { libc::getrlimit(libc::RLIMIT_NICE, limit.as_mut_ptr()) };
        if rc != 0 {
            return Priority::Nice(current);
        }
        // SAFETY: `getrlimit` returned 0, so it initialised `limit`.
        let soft = unsafe { limit.assume_init() }.rlim_cur;
        // `RLIMIT_NICE`'s soft limit `r` lets a thread lower its nice value to `20 - r`.
        let lowest = 20 - i32::try_from(soft.min(40)).unwrap_or(40);
        let target = lowest.clamp(-20, 19);
        if target >= current {
            return Priority::Nice(current);
        }
        // SAFETY: `PRIO_PROCESS` with a thread id sets that thread's nice value, and touches no
        // memory of the caller.
        let rc = unsafe { libc::setpriority(libc::PRIO_PROCESS, tid, target) };
        if rc == 0 {
            Priority::Nice(target)
        } else {
            Priority::Nice(current)
        }
    }
}

#[cfg(windows)]
mod platform {
    use std::io;

    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, GetCurrentProcessorNumber, GetCurrentThread, GetNumaProcessorNode,
        GetProcessAffinityMask, SetThreadAffinityMask, SetThreadPriority,
        THREAD_PRIORITY_TIME_CRITICAL,
    };

    use super::{Core, CoreId, PinError, Priority};

    pub(super) fn cpu_topology() -> io::Result<Vec<Core>> {
        let (mut process, mut system) = (0usize, 0usize);
        // SAFETY: both pointers are to writable `usize`s valid for the call, and the call writes
        // only through them. `GetCurrentProcess` returns a pseudo-handle that needs no closing.
        let ok = unsafe { GetProcessAffinityMask(GetCurrentProcess(), &mut process, &mut system) };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok((0..usize::BITS)
            .filter(|&cpu| process & (1usize << cpu) != 0)
            .map(|cpu| {
                let mut node = 0u8;
                // SAFETY: `node` is a writable byte for the call; `cpu` is below 64, so it fits
                // the call's `u8`.
                let ok = unsafe { GetNumaProcessorNode(cpu as u8, &mut node) };
                Core {
                    id: CoreId(cpu),
                    numa_node: if ok != 0 { u16::from(node) } else { 0 },
                }
            })
            .collect())
    }

    pub(super) fn pin_current_thread(core: CoreId) -> Result<(), PinError> {
        if core.0 >= usize::BITS {
            return Err(PinError::OutOfRange(core));
        }
        // SAFETY: `GetCurrentThread` returns the calling thread's pseudo-handle, valid for the
        // call; the mask names one core of processor group 0.
        let previous = unsafe { SetThreadAffinityMask(GetCurrentThread(), 1usize << core.0) };
        if previous != 0 {
            Ok(())
        } else {
            Err(PinError::Os(io::Error::last_os_error()))
        }
    }

    pub(super) fn current_core() -> Option<CoreId> {
        // SAFETY: `GetCurrentProcessorNumber` takes no arguments and cannot fail.
        Some(CoreId(unsafe { GetCurrentProcessorNumber() }))
    }

    pub(super) fn raise_current_thread_priority() -> Priority {
        // SAFETY: `GetCurrentThread` returns the calling thread's pseudo-handle, valid for the
        // call. A process of normal priority class may set any thread level up to time-critical.
        let ok = unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL) };
        if ok != 0 {
            Priority::Level(THREAD_PRIORITY_TIME_CRITICAL)
        } else {
            Priority::Unchanged
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod platform {
    use std::io;

    use super::{Core, CoreId, PinError, Priority};

    /// `QOS_CLASS_USER_INTERACTIVE` from `<sys/qos.h>`.
    const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;

    // SAFETY: the declaration of `<pthread/qos.h>` in libSystem, which every macOS and iOS
    // process links.
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }

    pub(super) fn cpu_topology() -> io::Result<Vec<Core>> {
        let cores = std::thread::available_parallelism()?.get();
        let cores = u32::try_from(cores).unwrap_or(u32::MAX);
        Ok((0..cores)
            .map(|cpu| Core {
                id: CoreId(cpu),
                numa_node: 0,
            })
            .collect())
    }

    pub(super) fn pin_current_thread(_core: CoreId) -> Result<(), PinError> {
        Err(PinError::Unsupported)
    }

    pub(super) fn current_core() -> Option<CoreId> {
        None
    }

    pub(super) fn raise_current_thread_priority() -> Priority {
        // SAFETY: the call acts on the calling thread only and touches no memory of the caller.
        let rc = unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) };
        if rc == 0 {
            Priority::UserInteractive
        } else {
            Priority::Unchanged
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cpulist_reads_ranges_and_single_cpus() {
        assert_eq!(parse_cpulist("0-3,8,10-11\n"), [0, 1, 2, 3, 8, 10, 11]);
        assert_eq!(parse_cpulist(""), Vec::<u32>::new());
        assert_eq!(parse_cpulist("2,x,4-y,6"), [2, 6]);
    }

    #[test]
    fn the_topology_lists_at_least_one_core_in_order() {
        let cores = cpu_topology().expect("the affinity mask");
        assert!(!cores.is_empty());
        assert!(cores.windows(2).all(|w| w[0].id < w[1].id), "{cores:?}");
    }

    /// A thread pinned to a core of the process's mask runs on it, where the OS can pin.
    #[test]
    fn a_pinned_thread_runs_on_its_core() {
        let cores = cpu_topology().unwrap();
        let target = cores.last().unwrap().id;
        let seen = std::thread::spawn(move || match pin_current_thread(target) {
            Ok(()) => Some((0..1_000).all(|_| current_core() == Some(target))),
            Err(PinError::Unsupported) => None,
            Err(error) => panic!("{error}"),
        })
        .join()
        .unwrap();
        if cfg!(any(target_os = "macos", target_os = "ios")) {
            assert_eq!(seen, None);
        } else {
            assert_eq!(seen, Some(true));
        }
    }

    /// The topology is the process's, so a pinned thread asking for it still sees every core.
    #[test]
    fn a_pinned_thread_still_sees_the_whole_topology() {
        let outside = cpu_topology().unwrap();
        let target = outside[0].id;
        let (pinned, inside) = std::thread::spawn(move || {
            let pinned = pin_current_thread(target).is_ok();
            (pinned, cpu_topology().unwrap())
        })
        .join()
        .unwrap();
        assert_eq!(inside, outside, "pinned: {pinned}");
    }

    #[test]
    fn raising_the_priority_never_lowers_it() {
        let granted = std::thread::spawn(raise_current_thread_priority)
            .join()
            .unwrap();
        match granted {
            Priority::Nice(nice) => assert!(nice <= 0, "{nice}"),
            Priority::Level(level) => assert_eq!(level, 15),
            Priority::UserInteractive | Priority::Unchanged => {}
        }
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn numa_nodes_come_from_each_nodes_cpulist() {
        let root = std::env::temp_dir().join(format!("syntropy-io-numa-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for (node, list) in [
            ("node0", "0-1\n"),
            ("node1", "2-3\n"),
            ("possible", "0-3\n"),
        ] {
            std::fs::create_dir_all(root.join(node)).unwrap();
            std::fs::write(root.join(node).join("cpulist"), list).unwrap();
        }
        let nodes = platform::numa_of(&root);
        std::fs::remove_dir_all(&root).unwrap();
        assert_eq!(
            nodes.into_iter().collect::<Vec<_>>(),
            [(0, 0), (1, 0), (2, 1), (3, 1)]
        );
    }
}
