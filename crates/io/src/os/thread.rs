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
//! | [`cpu_topology`] | the process's affinity mask (`sched_getaffinity`), NUMA nodes from `/sys/devices/system/node` | the process's affinity mask over its primary group (`GetProcessAffinityMask`), NUMA nodes from `GetNumaProcessorNodeEx` in that group | every core, all on node 0 (unified memory) |
//! | [`pin_current_thread`] | `sched_setaffinity` | `SetThreadAffinityMask`, within the thread's primary group | [`PinError::Unsupported`]: the OS has no thread-to-core affinity |
//! | [`raise_current_thread_priority`] | the lowest nice value `RLIMIT_NICE` allows, if below the current one | `THREAD_PRIORITY_TIME_CRITICAL` | QoS class `USER_INTERACTIVE` |
//! | [`current_core`] | `sched_getcpu` | `GetCurrentProcessorNumberEx`, `None` outside the primary group | `None` |
//!
//! **Windows processor groups.** Windows numbers a logical processor within its processor group,
//! a set of up to 64. A process gets one primary group when it starts, and by default its
//! threads share it. `GetProcessAffinityMask` and `SetThreadAffinityMask` act within that group,
//! so on Windows a [`CoreId`] is a number within the process's primary group. On a host with more
//! than 64 logical processors, hot threads use only that group's cores. Before Windows 11, a
//! process with threads in several groups gets no affinity mask. There, [`cpu_topology`] fails
//! with an error that says so, rather than returning no cores.

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
    /// The core cannot be named in the call, such as a number of 64 or more on Windows, which
    /// numbers cores within the process's primary group.
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
/// On Linux the process's mask is its main thread's, so call this before pinning the main
/// thread, if it is ever pinned.
///
/// # Errors
///
/// If the OS cannot report the process's affinity mask: on Windows before Windows 11, that
/// includes a process with threads in more than one processor group.
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

/// Why a Windows process has no affinity mask before Windows 11.
#[cfg(any(windows, test))]
const MULTI_GROUP: &str = "this process has threads in more than one Windows processor group, and \
     before Windows 11 GetProcessAffinityMask reports no mask for such a process, so its cores \
     cannot be listed and its hot threads cannot be pinned";

/// The cores of a Windows process, from what its calls report: `mask`, the process's affinity
/// mask over its primary group `group`, and `node_of(group, number)`, the NUMA node
/// `GetNumaProcessorNodeEx` gives a processor of a group, or `None` if the call fails. A node of
/// `0xFFFF` means the processor does not exist. Either way, the core's node is unknown and
/// reported as 0, as [`Core::numa_node`] says.
///
/// Kept apart from the calls so a test can run it on any OS.
///
/// # Errors
///
/// A mask of 0, which `GetProcessAffinityMask` returns before Windows 11 for a process with
/// threads in more than one group ([`MULTI_GROUP`]).
#[cfg(any(windows, test))]
fn group_topology(
    group: u16,
    mask: usize,
    mut node_of: impl FnMut(u16, u8) -> Option<u16>,
) -> io::Result<Vec<Core>> {
    if mask == 0 {
        return Err(io::Error::other(MULTI_GROUP));
    }
    Ok((0..usize::BITS)
        .filter(|&number| mask & (1usize << number) != 0)
        .map(|number| {
            // `number` is below `usize::BITS`, at most 64, so it fits the call's `u8`.
            let node = node_of(group, number as u8).filter(|&node| node != u16::MAX);
            Core {
                id: CoreId(number),
                numa_node: node.unwrap_or(0),
            }
        })
        .collect())
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
    pub(super) fn tid() -> libc::id_t {
        // SAFETY: `SYS_gettid` takes no arguments and cannot fail.
        let tid = unsafe { libc::syscall(libc::SYS_gettid) };
        libc::id_t::try_from(tid).unwrap_or(0)
    }

    /// The calling thread's nice value.
    pub(super) fn nice_of(tid: libc::id_t) -> Option<i32> {
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

    use windows_sys::Win32::System::Kernel::PROCESSOR_NUMBER;
    use windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY;
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, GetCurrentProcessorNumberEx, GetCurrentThread, GetNumaProcessorNodeEx,
        GetProcessAffinityMask, GetThreadGroupAffinity, SetThreadAffinityMask, SetThreadPriority,
        THREAD_PRIORITY_TIME_CRITICAL,
    };

    use super::{Core, CoreId, PinError, Priority, group_topology};

    /// The calling thread's primary group: the group `GetProcessAffinityMask` and
    /// `SetThreadAffinityMask` number its cores in. Its threads share the process's by default.
    fn primary_group() -> io::Result<u16> {
        let mut affinity = GROUP_AFFINITY::default();
        // SAFETY: `affinity` is a writable `GROUP_AFFINITY` valid for the call, which writes only
        // through it. `GetCurrentThread`'s pseudo-handle carries the query right the call needs.
        let ok = unsafe { GetThreadGroupAffinity(GetCurrentThread(), &mut affinity) };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(affinity.Group)
    }

    pub(super) fn cpu_topology() -> io::Result<Vec<Core>> {
        let group = primary_group()?;
        let (mut process, mut system) = (0usize, 0usize);
        // SAFETY: both pointers are to writable `usize`s valid for the call, and the call writes
        // only through them. `GetCurrentProcess` returns a pseudo-handle that needs no closing.
        let ok = unsafe { GetProcessAffinityMask(GetCurrentProcess(), &mut process, &mut system) };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        group_topology(group, process, |group, number| {
            let processor = PROCESSOR_NUMBER {
                Group: group,
                Number: number,
                Reserved: 0,
            };
            let mut node = 0u16;
            // SAFETY: `processor` is a readable `PROCESSOR_NUMBER` and `node` a writable `u16`,
            // both valid for the call, which writes only through `node`.
            let ok = unsafe { GetNumaProcessorNodeEx(&processor, &mut node) };
            (ok != 0).then_some(node)
        })
    }

    pub(super) fn pin_current_thread(core: CoreId) -> Result<(), PinError> {
        if core.0 >= usize::BITS {
            return Err(PinError::OutOfRange(core));
        }
        // SAFETY: `GetCurrentThread` returns the calling thread's pseudo-handle, valid for the
        // call; the mask names one core of the thread's primary group, and the call reads no
        // memory of the caller.
        let previous = unsafe { SetThreadAffinityMask(GetCurrentThread(), 1usize << core.0) };
        if previous != 0 {
            Ok(())
        } else {
            Err(PinError::Os(io::Error::last_os_error()))
        }
    }

    /// The core the thread runs on, if it is in the primary group: since Windows 11 an unpinned
    /// thread may run in another group, where its number would name a different core.
    pub(super) fn current_core() -> Option<CoreId> {
        let group = primary_group().ok()?;
        let mut processor = PROCESSOR_NUMBER::default();
        // SAFETY: `processor` is a writable `PROCESSOR_NUMBER` valid for the call, which writes
        // only through it and cannot fail.
        unsafe { GetCurrentProcessorNumberEx(&mut processor) };
        (processor.Group == group).then_some(CoreId(u32::from(processor.Number)))
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
        let (before, granted) =
            std::thread::spawn(|| (current_nice(), raise_current_thread_priority()))
                .join()
                .unwrap();
        match granted {
            // Lower runs first, so a raised priority is a nice value no higher than before, which
            // need not be 0: under `nice cargo test` it starts above.
            Priority::Nice(nice) => {
                let before = before.expect("the nice value before the call");
                assert!(nice <= before, "{before} -> {nice}");
            }
            Priority::Level(level) => assert_eq!(level, 15),
            Priority::UserInteractive | Priority::Unchanged => {}
        }
    }

    /// The calling thread's nice value, where the OS has one.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn current_nice() -> Option<i32> {
        platform::nice_of(platform::tid())
    }

    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    fn current_nice() -> Option<i32> {
        None
    }

    /// Windows' cores come from the process's mask over its primary group, and each core's NUMA
    /// node is looked up in that group, not in group 0 or the group the caller runs in.
    #[test]
    fn windows_cores_are_numbered_and_noded_within_the_primary_group() {
        let mut asked = Vec::new();
        // Group 1 of a two-group host: its processor 0 on node 2, processor 1 on node 3,
        // processor 3 failing the lookup and the top processor reported as not existing.
        let top = usize::BITS - 1;
        let cores = group_topology(1, 0b1011 | 1 << top, |group, number| {
            asked.push((group, number));
            match number {
                0 => Some(2),
                1 => Some(3),
                _ if u32::from(number) == top => Some(u16::MAX),
                _ => None,
            }
        })
        .unwrap();
        assert_eq!(asked, [(1, 0), (1, 1), (1, 3), (1, top as u8)]);
        let node = |id, numa_node| Core {
            id: CoreId(id),
            numa_node,
        };
        assert_eq!(cores, [node(0, 2), node(1, 3), node(3, 0), node(top, 0)]);
    }

    /// Before Windows 11, a process with threads in several groups gets a mask of 0. That is an
    /// error that says why, not a topology with no cores.
    #[test]
    fn a_multi_group_process_without_a_mask_is_an_error() {
        let error = group_topology(0, 0, |_, _| Some(0)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("more than one Windows processor group"),
            "{error}"
        );
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
