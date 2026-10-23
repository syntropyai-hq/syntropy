// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Windows system calls of E1 host enumeration (RFC-0010 §2.2; PLC-E1-T03), behind safe
//! functions that `probe` calls (RFC-0001 §4.2 rule 8, note of 2026-10-22; OQ-PRC-26).
//!
//! | Function | Call |
//! |---|---|
//! | [`logical_processor_information`] | `GetLogicalProcessorInformationEx(RelationAll)`: cores with `EfficiencyClass`, caches, NUMA nodes and their group masks, as the raw buffer |
//! | [`memory_status`] | `GlobalMemoryStatusEx` |
//! | [`numa_available_memory`] | `GetNumaAvailableMemoryNodeEx` |
//! | [`numa_proximity_node`] | `GetNumaProximityNodeEx` |
//! | [`firmware_table`] | `GetSystemFirmwareTable`, for the ACPI SRAT |
//! | [`power_status`] | `GetSystemPowerStatus` |
//! | [`process_working_set`], [`set_process_working_set`] | `GetProcessWorkingSetSizeEx`, `SetProcessWorkingSetSizeEx` on this process, for the working-set ceiling probe |
//! | [`os_version`] | `RtlGetVersion`, which reports the real version whatever the manifest says |
//!
//! The processor buffer is returned as bytes and parsed in `probe`, so the parser has no
//! `unsafe` and runs on recorded buffers.
//!
//! Every `unsafe` block carries a `// SAFETY:` comment with its invariant (ADR-0001 item 4).

use std::io;
use std::ptr;

use windows_sys::Wdk::System::SystemServices::RtlGetVersion;
use windows_sys::Win32::Foundation::ERROR_INSUFFICIENT_BUFFER;
use windows_sys::Win32::System::Memory::{GetProcessWorkingSetSizeEx, SetProcessWorkingSetSizeEx};
use windows_sys::Win32::System::Power::{GetSystemPowerStatus, SYSTEM_POWER_STATUS};
use windows_sys::Win32::System::SystemInformation::{
    GetLogicalProcessorInformationEx, GetSystemFirmwareTable, GlobalMemoryStatusEx, MEMORYSTATUSEX,
    OSVERSIONINFOW, RelationAll, SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
};
use windows_sys::Win32::System::Threading::{
    GetCurrentProcess, GetNumaAvailableMemoryNodeEx, GetNumaProximityNodeEx,
};

/// The `GetLogicalProcessorInformationEx(RelationAll)` buffer: a sequence of
/// `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX` records, each `Size` bytes long.
///
/// # Errors
///
/// When the call fails.
pub fn logical_processor_information() -> io::Result<Vec<u8>> {
    let mut len: u32 = 0;
    // The set of processors can change between the size query and the read (hot add), so
    // retry while the buffer is too small.
    for _ in 0..4 {
        // Records hold `usize` masks, so the buffer is 8-byte aligned.
        let mut buf = vec![0u64; (len as usize).div_ceil(8)];
        let ptr = if buf.is_empty() {
            ptr::null_mut()
        } else {
            buf.as_mut_ptr()
                .cast::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>()
        };
        // SAFETY: `ptr` is null with `len` 0 (a size query), or points to `buf`, which is
        // writable for at least `len` bytes and 8-byte aligned. The call writes at most `len`
        // bytes and the length it needs or wrote to `len`.
        let ok = unsafe { GetLogicalProcessorInformationEx(RelationAll, ptr, &mut len) };
        if ok != 0 {
            let bytes: Vec<u8> = buf.iter().flat_map(|w| w.to_ne_bytes()).collect();
            return Ok(bytes[..len as usize].to_vec());
        }
        let err = io::Error::last_os_error();
        if err.raw_os_error() != Some(ERROR_INSUFFICIENT_BUFFER as i32) {
            return Err(err);
        }
    }
    Err(io::Error::other(
        "GetLogicalProcessorInformationEx kept growing",
    ))
}

/// The fields of `MEMORYSTATUSEX` E1 reads, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryStatus {
    /// `ullTotalPhys`.
    pub total_phys: u64,
    /// `ullAvailPhys`.
    pub avail_phys: u64,
    /// `ullTotalPageFile`: the commit limit.
    pub total_page_file: u64,
    /// `ullAvailPageFile`.
    pub avail_page_file: u64,
}

/// `GlobalMemoryStatusEx`.
///
/// # Errors
///
/// When the call fails.
pub fn memory_status() -> io::Result<MemoryStatus> {
    let mut m = MEMORYSTATUSEX {
        dwLength: size_of::<MEMORYSTATUSEX>() as u32,
        dwMemoryLoad: 0,
        ullTotalPhys: 0,
        ullAvailPhys: 0,
        ullTotalPageFile: 0,
        ullAvailPageFile: 0,
        ullTotalVirtual: 0,
        ullAvailVirtual: 0,
        ullAvailExtendedVirtual: 0,
    };
    // SAFETY: `m` is a writable `MEMORYSTATUSEX` with `dwLength` set to its size, as the call
    // requires, and the call writes only through it.
    if unsafe { GlobalMemoryStatusEx(&mut m) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(MemoryStatus {
        total_phys: m.ullTotalPhys,
        avail_phys: m.ullAvailPhys,
        total_page_file: m.ullTotalPageFile,
        avail_page_file: m.ullAvailPageFile,
    })
}

/// Free memory on NUMA node `node`, in bytes (`GetNumaAvailableMemoryNodeEx`).
///
/// # Errors
///
/// When the call fails, for example for a node that does not exist.
pub fn numa_available_memory(node: u16) -> io::Result<u64> {
    let mut bytes: u64 = 0;
    // SAFETY: `bytes` is a writable `u64` for the call, which writes only through it.
    if unsafe { GetNumaAvailableMemoryNodeEx(node, &mut bytes) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(bytes)
}

/// The NUMA node number of ACPI proximity domain `proximity` (`GetNumaProximityNodeEx`);
/// `None` when the OS maps it to no node.
#[must_use]
pub fn numa_proximity_node(proximity: u32) -> Option<u16> {
    let mut node: u16 = 0;
    // SAFETY: `node` is a writable `u16` for the call, which writes only through it.
    let ok = unsafe { GetNumaProximityNodeEx(proximity, &mut node) };
    (ok != 0).then_some(node)
}

/// A firmware table (`GetSystemFirmwareTable`), such as the ACPI SRAT: `provider` and `id`
/// are the signatures as the call takes them. `Ok(None)` when the firmware has no such table.
///
/// # Errors
///
/// When the call fails for another reason.
pub fn firmware_table(provider: u32, id: u32) -> io::Result<Option<Vec<u8>>> {
    let mut size = 0u32;
    for _ in 0..4 {
        let mut buf = vec![0u8; size as usize];
        let ptr = if buf.is_empty() {
            ptr::null_mut()
        } else {
            buf.as_mut_ptr()
        };
        // SAFETY: `ptr` is null with size 0 (a size query) or points to `buf`, writable for
        // `size` bytes, which is the size passed. The call writes at most that many bytes.
        let need = unsafe { GetSystemFirmwareTable(provider, id, ptr, size) };
        if need == 0 {
            let err = io::Error::last_os_error();
            // ERROR_NOT_FOUND (1168): no such table.
            return match err.raw_os_error() {
                Some(1168) | Some(0) => Ok(None),
                _ => Err(err),
            };
        }
        if need <= size {
            buf.truncate(need as usize);
            return Ok(Some(buf));
        }
        size = need;
    }
    Err(io::Error::other("GetSystemFirmwareTable kept growing"))
}

/// `SYSTEM_POWER_STATUS`'s fields E1 reads, as the call reports them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PowerStatus {
    /// `ACLineStatus`: 0 offline, 1 online, 255 unknown.
    pub ac_line: u8,
    /// `BatteryFlag`: 128 no system battery, 255 unknown, otherwise charge flags.
    pub battery_flag: u8,
    /// `BatteryLifePercent`: 0–100, or 255 unknown.
    pub battery_percent: u8,
}

/// `GetSystemPowerStatus`.
///
/// # Errors
///
/// When the call fails.
pub fn power_status() -> io::Result<PowerStatus> {
    let mut s = SYSTEM_POWER_STATUS {
        ACLineStatus: 0,
        BatteryFlag: 0,
        BatteryLifePercent: 0,
        SystemStatusFlag: 0,
        BatteryLifeTime: 0,
        BatteryFullLifeTime: 0,
    };
    // SAFETY: `s` is a writable `SYSTEM_POWER_STATUS` for the call, which writes only through
    // it.
    if unsafe { GetSystemPowerStatus(&mut s) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(PowerStatus {
        ac_line: s.ACLineStatus,
        battery_flag: s.BatteryFlag,
        battery_percent: s.BatteryLifePercent,
    })
}

/// This process's working-set limits (`GetProcessWorkingSetSizeEx`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkingSet {
    /// Minimum working set, in bytes.
    pub min: u64,
    /// Maximum working set, in bytes.
    pub max: u64,
    /// The `QUOTA_LIMITS_HARDWS_*` flags.
    pub flags: u32,
}

/// This process's working-set limits.
///
/// # Errors
///
/// When the call fails.
pub fn process_working_set() -> io::Result<WorkingSet> {
    let (mut min, mut max, mut flags) = (0usize, 0usize, 0u32);
    // SAFETY: `GetCurrentProcess` returns a pseudo-handle that needs no closing. The three
    // outputs are writable for the call, which writes only through them.
    let ok =
        unsafe { GetProcessWorkingSetSizeEx(GetCurrentProcess(), &mut min, &mut max, &mut flags) };
    if ok == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(WorkingSet {
        min: min as u64,
        max: max as u64,
        flags,
    })
}

/// Sets this process's working-set minimum and maximum, leaving the hard-limit flags as they
/// are (`SetProcessWorkingSetSizeEx` with flags 0). Raising the minimum takes no memory; it
/// asks the OS to guarantee that much resident memory, and fails when the OS cannot.
///
/// # Errors
///
/// When the OS refuses, for example `ERROR_NO_SYSTEM_RESOURCES` for a minimum it cannot
/// guarantee.
pub fn set_process_working_set(min: u64, max: u64) -> io::Result<()> {
    let min = usize::try_from(min).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    let max = usize::try_from(max).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    // SAFETY: the pseudo-handle of this process and plain integers; the call reads no caller
    // memory.
    let ok = unsafe { SetProcessWorkingSetSizeEx(GetCurrentProcess(), min, max, 0) };
    if ok == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// The OS version (`RtlGetVersion`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OsVersion {
    /// `dwMajorVersion`.
    pub major: u32,
    /// `dwMinorVersion`.
    pub minor: u32,
    /// `dwBuildNumber`: 22000 and later is Windows 11.
    pub build: u32,
}

/// The running OS version.
///
/// # Errors
///
/// When the call returns a failure status.
pub fn os_version() -> io::Result<OsVersion> {
    let mut v = OSVERSIONINFOW {
        dwOSVersionInfoSize: size_of::<OSVERSIONINFOW>() as u32,
        dwMajorVersion: 0,
        dwMinorVersion: 0,
        dwBuildNumber: 0,
        dwPlatformId: 0,
        szCSDVersion: [0; 128],
    };
    // SAFETY: `v` is a writable `OSVERSIONINFOW` with its size field set, as the call
    // requires, and the call writes only through it.
    let status = unsafe { RtlGetVersion(&mut v) };
    if status != 0 {
        return Err(io::Error::other(format!(
            "RtlGetVersion returned {status:#x}"
        )));
    }
    Ok(OsVersion {
        major: v.dwMajorVersion,
        minor: v.dwMinorVersion,
        build: v.dwBuildNumber,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn processor_buffer_is_a_sequence_of_records() {
        let buf = logical_processor_information().unwrap();
        let mut at = 0;
        let mut records = 0;
        while at + 8 <= buf.len() {
            let size = u32::from_ne_bytes(buf[at + 4..at + 8].try_into().unwrap()) as usize;
            assert!(size >= 8);
            at += size;
            records += 1;
        }
        assert_eq!(at, buf.len());
        assert!(records > 0);
    }

    #[test]
    fn memory_and_version_read() {
        let m = memory_status().unwrap();
        assert!(m.total_phys > 0 && m.avail_phys <= m.total_phys);
        assert!(numa_available_memory(0).unwrap() > 0);
        assert!(os_version().unwrap().major >= 10);
        power_status().unwrap();
    }

    #[test]
    fn working_set_round_trips() {
        let ws = process_working_set().unwrap();
        set_process_working_set(ws.min, ws.max).unwrap();
        assert_eq!(process_working_set().unwrap(), ws);
    }
}
