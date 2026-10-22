// SPDX-License-Identifier: MIT OR Apache-2.0

//! The OS shim: records the live host's [`IsaInputs`]. It makes no decision; [`select`] does.
//!
//! - x86-64, every OS: `is_x86_feature_detected!`, CPUID leaf 7 for the AMX bits, and XCR0
//!   through XGETBV. On Linux it also asks for the AMX tile permission with
//!   `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)`, once per call, when the CPU and
//!   the OS state have the tiles and no kill switch rules AMX out.
//! - AArch64 on macOS: the `hw.optional.arm.FEAT_*` sysctls, read with `sysctlbyname`.
//! - AArch64 elsewhere: `is_aarch64_feature_detected!`, plus `AT_HWCAP2` for SME2 on Linux
//!   and Android. Elsewhere (Windows) SME2 is reported absent.
//!
//! The few C functions it calls are declared here, behind `cfg`, because the workspace has no
//! C bindings crate.

use super::{Detection, IsaInputs, KillSwitches, select};

/// Detects the level of the CPU this process runs on, with `kill` applied. `None` on an
/// architecture this provider has no kernels for, or an x86-64 CPU below `X86V2`.
///
/// `enumerate` and `open` each call it in their own process (RFC-0040 §7.3).
#[must_use]
pub fn detect(kill: &KillSwitches) -> Option<Detection> {
    select(&inputs(kill)?, kill)
}

/// The live host's detection inputs. `kill` decides only whether the AMX permission is
/// requested.
#[must_use]
pub fn inputs(kill: &KillSwitches) -> Option<IsaInputs> {
    imp::inputs(kill)
}

#[cfg(target_arch = "x86_64")]
mod imp {
    use std::arch::is_x86_feature_detected as detected;
    use std::arch::x86_64::{__cpuid, __cpuid_count, _xgetbv};

    use crate::isa::{AmxPermission, IsaInputs, KillSwitches, X86Feature, X86Features, X86Inputs};

    /// CPUID leaf 1, ECX bit 27: the OS has enabled XGETBV (OSXSAVE).
    const CPUID1_ECX_OSXSAVE: u32 = 1 << 27;

    pub(super) fn inputs(kill: &KillSwitches) -> Option<IsaInputs> {
        let detected = X86Feature::ALL
            .into_iter()
            .filter(|f| std_detected(*f))
            .collect::<X86Features>();
        let max_leaf = __cpuid(0).eax;
        let cpuid7_edx = if max_leaf >= 7 {
            __cpuid_count(7, 0).edx
        } else {
            0
        };
        let xcr0 = if __cpuid(1).ecx & CPUID1_ECX_OSXSAVE != 0 {
            // SAFETY: XGETBV needs XSAVE support, and CPUID's OSXSAVE bit, just checked,
            // says both that the CPU has it and that the OS enabled it. XCR0 (index 0)
            // always exists when XGETBV does.
            unsafe { _xgetbv(0) }
        } else {
            0
        };
        let mut x = X86Inputs {
            detected,
            cpuid7_edx,
            xcr0,
            amx: AmxPermission::NotRequested,
        };
        if x.amx_hardware() && !kill.amx && !kill.avx512 {
            x.amx = request_amx();
        }
        Some(IsaInputs::X86(x))
    }

    fn std_detected(f: X86Feature) -> bool {
        match f {
            X86Feature::Sse42 => detected!("sse4.2"),
            X86Feature::Popcnt => detected!("popcnt"),
            X86Feature::Avx2 => detected!("avx2"),
            X86Feature::Fma => detected!("fma"),
            X86Feature::F16c => detected!("f16c"),
            X86Feature::Avx512f => detected!("avx512f"),
            X86Feature::Avx512bw => detected!("avx512bw"),
            X86Feature::Avx512vl => detected!("avx512vl"),
            X86Feature::Avx512dq => detected!("avx512dq"),
            X86Feature::Avx512vnni => detected!("avx512vnni"),
            X86Feature::Avx512bf16 => detected!("avx512bf16"),
            X86Feature::Avx512fp16 => detected!("avx512fp16"),
            X86Feature::Avxvnni => detected!("avxvnni"),
        }
    }

    /// Asks Linux for the AMX tile-data permission for this process. The grant is
    /// per process and lasts for its life, so asking again is harmless.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn request_amx() -> AmxPermission {
        use std::ffi::{c_long, c_ulong};

        /// `__NR_arch_prctl` on x86-64.
        const SYS_ARCH_PRCTL: c_long = 158;
        /// `ARCH_REQ_XCOMP_PERM` from `asm/prctl.h`.
        const ARCH_REQ_XCOMP_PERM: c_ulong = 0x1023;
        /// `XFEATURE_XTILEDATA`: the XSAVE component number of the tile data.
        const XFEATURE_XTILEDATA: c_ulong = 18;

        unsafe extern "C" {
            fn syscall(number: c_long, ...) -> c_long;
        }

        // SAFETY: `arch_prctl(ARCH_REQ_XCOMP_PERM, n)` takes two integer arguments, reads
        // and writes no user memory, and only changes which XSAVE components this process
        // may use. A kernel that does not know the request returns -1 with EINVAL.
        let rc = unsafe { syscall(SYS_ARCH_PRCTL, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) };
        if rc == 0 {
            AmxPermission::Granted
        } else {
            AmxPermission::Denied {
                errno: std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
            }
        }
    }

    /// Outside Linux there is no request: the XCR0 tile bits are the OS's enablement.
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    fn request_amx() -> AmxPermission {
        AmxPermission::NotNeeded
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
mod imp {
    use std::ffi::{CStr, c_char, c_int, c_void};

    use crate::isa::{ArmInputs, IsaInputs, KillSwitches, MacSysctls};

    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }

    pub(super) fn inputs(_: &KillSwitches) -> Option<IsaInputs> {
        let s = MacSysctls {
            dot_prod: sysctl_i32(c"hw.optional.arm.FEAT_DotProd"),
            fp16: sysctl_i32(c"hw.optional.arm.FEAT_FP16"),
            i8mm: sysctl_i32(c"hw.optional.arm.FEAT_I8MM"),
            bf16: sysctl_i32(c"hw.optional.arm.FEAT_BF16"),
            sme2: sysctl_i32(c"hw.optional.arm.FEAT_SME2"),
        };
        Some(IsaInputs::Arm(ArmInputs::from_mac_sysctls(&s)))
    }

    /// An integer sysctl's value; `None` when it does not exist or is not 4 bytes.
    fn sysctl_i32(name: &CStr) -> Option<i32> {
        let mut value: i32 = 0;
        let mut len = size_of::<i32>();
        // SAFETY: `name` is NUL-terminated; `oldp` points to `len` writable bytes owned by
        // this frame, and the call writes at most `len` bytes there and the written length
        // to `len`; `newp` is null with `newlen` 0, so nothing is set.
        let rc = unsafe {
            sysctlbyname(
                name.as_ptr(),
                (&raw mut value).cast::<c_void>(),
                &raw mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (rc == 0 && len == size_of::<i32>()).then_some(value)
    }
}

#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
mod imp {
    use std::arch::is_aarch64_feature_detected as detected;

    use crate::isa::{ArmInputs, IsaInputs, KillSwitches};

    pub(super) fn inputs(_: &KillSwitches) -> Option<IsaInputs> {
        let a = ArmInputs {
            dotprod: detected!("dotprod"),
            fp16: detected!("fp16"),
            i8mm: detected!("i8mm"),
            bf16: detected!("bf16"),
            sve2: detected!("sve2"),
            sme2: false,
        };
        Some(IsaInputs::Arm(a.with_hwcap2(hwcap2())))
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn hwcap2() -> u64 {
        use std::ffi::c_ulong;

        /// `AT_HWCAP2` from `elf.h`.
        const AT_HWCAP2: c_ulong = 26;

        unsafe extern "C" {
            fn getauxval(kind: c_ulong) -> c_ulong;
        }

        // SAFETY: `getauxval` reads the process's auxiliary vector, which the C library set
        // up before `main`; it takes an integer and returns 0 for an absent entry.
        unsafe { getauxval(AT_HWCAP2) }
    }

    /// No `AT_HWCAP2` outside Linux: SME2 is reported absent.
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    fn hwcap2() -> u64 {
        0
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod imp {
    use crate::isa::{IsaInputs, KillSwitches};

    /// No level exists for this architecture.
    pub(super) fn inputs(_: &KillSwitches) -> Option<IsaInputs> {
        None
    }
}
