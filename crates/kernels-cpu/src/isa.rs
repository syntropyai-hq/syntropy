// SPDX-License-Identifier: MIT OR Apache-2.0

//! The ISA level of a CPU device (RFC-0040 §7.3; ADR-0402 items 1–3; ADR-0405 decision 3).
//!
//! Detection has two halves. The OS shim ([`detect`], in `host.rs`) records the facts of the
//! live host in an [`IsaInputs`]: what the standard library's detection macros report, the
//! AMX CPUID bits, XCR0 and the Linux `arch_prctl` result on x86-64, and the macro results,
//! `AT_HWCAP2` or the `hw.optional.arm.FEAT_*` sysctls on AArch64. [`select`] is a pure
//! function of those inputs and the [`KillSwitches`], so recorded inputs give the same level
//! on every machine. `enumerate` reports the selected level as the device's class; `open`
//! detects again and [`table_for_class`] checks the class before it hands out a table.
//!
//! Each level's requirements are RFC-0040 §7.3's table. Every feature bit this provider
//! reports belongs to one level ([`FEATURE_LEVELS`]), and a bit is set only when that level
//! or one above it in the same chain is selected. No bit then advertises a feature above the
//! class, whose packs and certificates are the only ones that apply (RFC-0040 §2.1: `open`
//! never loads a variant set or table the class does not name). A kill switch that lowers
//! the level drops the bits with it. A bit that belongs to no level, such as `SVE2` (no SVE2
//! kernels in v1, ADR-0402 item 4), is never reported.

use syntropy_core::ErrorClass;
use syntropy_kernels::{CpuIsa, DeviceFeatures, Error, ErrorCode};

use crate::dispatch::DispatchTable;

mod host;
#[cfg(test)]
mod tests;

pub use host::{detect, inputs};

/// A feature the standard library's `is_x86_feature_detected!` reports, which the x86 levels
/// need. The macro also checks that the OS saves the YMM and ZMM/opmask state, through
/// XGETBV.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum X86Feature {
    /// SSE4.2.
    Sse42,
    /// POPCNT.
    Popcnt,
    /// AVX2.
    Avx2,
    /// FMA.
    Fma,
    /// F16C.
    F16c,
    /// AVX-512 foundation.
    Avx512f,
    /// AVX-512 byte and word.
    Avx512bw,
    /// AVX-512 vector length.
    Avx512vl,
    /// AVX-512 doubleword and quadword.
    Avx512dq,
    /// AVX-512 VNNI.
    Avx512vnni,
    /// AVX-512 BF16.
    Avx512bf16,
    /// AVX-512 FP16.
    Avx512fp16,
    /// AVX-VNNI.
    Avxvnni,
}

impl X86Feature {
    /// Every feature, in declaration order.
    pub const ALL: [Self; 13] = [
        Self::Sse42,
        Self::Popcnt,
        Self::Avx2,
        Self::Fma,
        Self::F16c,
        Self::Avx512f,
        Self::Avx512bw,
        Self::Avx512vl,
        Self::Avx512dq,
        Self::Avx512vnni,
        Self::Avx512bf16,
        Self::Avx512fp16,
        Self::Avxvnni,
    ];

    /// The name the detection macro takes, e.g. `"sse4.2"`.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Sse42 => "sse4.2",
            Self::Popcnt => "popcnt",
            Self::Avx2 => "avx2",
            Self::Fma => "fma",
            Self::F16c => "f16c",
            Self::Avx512f => "avx512f",
            Self::Avx512bw => "avx512bw",
            Self::Avx512vl => "avx512vl",
            Self::Avx512dq => "avx512dq",
            Self::Avx512vnni => "avx512vnni",
            Self::Avx512bf16 => "avx512bf16",
            Self::Avx512fp16 => "avx512fp16",
            Self::Avxvnni => "avxvnni",
        }
    }

    fn bit(self) -> u32 {
        // The declaration index; `ALL` has 13 entries, so the shift fits a `u32`.
        1 << (self as u32)
    }
}

/// A set of [`X86Feature`]s: the ones the detection macro reported present.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct X86Features(u32);

impl X86Features {
    /// No feature.
    pub const NONE: Self = Self(0);

    /// This set plus `f`.
    #[must_use]
    pub fn with(self, f: X86Feature) -> Self {
        Self(self.0 | f.bit())
    }

    /// Whether `f` is in the set.
    #[must_use]
    pub fn has(self, f: X86Feature) -> bool {
        self.0 & f.bit() != 0
    }

    fn all(self, fs: &[X86Feature]) -> bool {
        fs.iter().all(|f| self.has(*f))
    }
}

impl FromIterator<X86Feature> for X86Features {
    fn from_iter<I: IntoIterator<Item = X86Feature>>(iter: I) -> Self {
        iter.into_iter().fold(Self::NONE, Self::with)
    }
}

/// CPUID leaf 7 sub-leaf 0, EDX bit 22: AMX-BF16.
pub const CPUID7_EDX_AMX_BF16: u32 = 1 << 22;
/// CPUID leaf 7 sub-leaf 0, EDX bit 24: AMX-TILE.
pub const CPUID7_EDX_AMX_TILE: u32 = 1 << 24;
/// CPUID leaf 7 sub-leaf 0, EDX bit 25: AMX-INT8.
pub const CPUID7_EDX_AMX_INT8: u32 = 1 << 25;
/// XCR0 bits 17 and 18: the OS saves the tile configuration and the tile data.
pub const XCR0_AMX_TILE: u64 = (1 << 17) | (1 << 18);

/// The outcome of asking the OS for AMX tile permission (RFC-0040 §7.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AmxPermission {
    /// No request was made: the CPU or the OS state lacks the tiles, or a kill switch rules
    /// AMX out.
    NotRequested,
    /// The OS has no request to make. Outside Linux, the XCR0 tile bits are the OS's
    /// enablement.
    NotNeeded,
    /// Linux granted `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` to this process.
    Granted,
    /// Linux refused the request.
    Denied {
        /// The `errno` the call set.
        errno: i32,
    },
}

/// The recorded x86-64 detection inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct X86Inputs {
    /// What `is_x86_feature_detected!` reported.
    pub detected: X86Features,
    /// CPUID leaf 7 sub-leaf 0, EDX; 0 when the CPU's highest leaf is below 7.
    pub cpuid7_edx: u32,
    /// XCR0 through XGETBV; 0 when CPUID reports OSXSAVE clear.
    pub xcr0: u64,
    /// The AMX permission request's outcome.
    pub amx: AmxPermission,
}

impl X86Inputs {
    /// Whether the CPU has AMX-TILE and AMX-INT8 and the OS saves their state. The shim asks
    /// for the Linux permission only when this holds.
    #[must_use]
    pub fn amx_hardware(&self) -> bool {
        let tiles = CPUID7_EDX_AMX_TILE | CPUID7_EDX_AMX_INT8;
        self.cpuid7_edx & tiles == tiles && self.xcr0 & XCR0_AMX_TILE == XCR0_AMX_TILE
    }
}

/// The recorded AArch64 detection inputs. NEON is always present on AArch64.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ArmInputs {
    /// SDOT/UDOT (`FEAT_DotProd`).
    pub dotprod: bool,
    /// FP16 arithmetic (`FEAT_FP16`).
    pub fp16: bool,
    /// SMMLA (`FEAT_I8MM`).
    pub i8mm: bool,
    /// BF16 (`FEAT_BF16`).
    pub bf16: bool,
    /// SVE2. No level needs it (ADR-0402 item 4), so it is recorded but never reported.
    pub sve2: bool,
    /// SME2 (`FEAT_SME2`).
    pub sme2: bool,
}

/// `AT_HWCAP2` bit 37 on Linux AArch64: `HWCAP2_SME2`.
pub const HWCAP2_SME2: u64 = 1 << 37;

impl ArmInputs {
    /// These inputs with SME2 taken from a Linux `AT_HWCAP2` value. The detection macro's
    /// `sme2` name is not stable on the pinned toolchain, so the shim reads `AT_HWCAP2`
    /// itself.
    #[must_use]
    pub fn with_hwcap2(mut self, hwcap2: u64) -> Self {
        self.sme2 = hwcap2 & HWCAP2_SME2 != 0;
        self
    }

    /// The inputs the macOS sysctls give. A sysctl that is absent or not 1 means the feature
    /// is absent. Apple silicon has no non-streaming SVE, so `sve2` is false.
    #[must_use]
    pub fn from_mac_sysctls(s: &MacSysctls) -> Self {
        let on = |v: Option<i32>| v == Some(1);
        Self {
            dotprod: on(s.dot_prod),
            fp16: on(s.fp16),
            i8mm: on(s.i8mm),
            bf16: on(s.bf16),
            sve2: false,
            sme2: on(s.sme2),
        }
    }
}

/// The `hw.optional.arm.FEAT_*` values macOS reports; `None` when the sysctl does not exist.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MacSysctls {
    /// `hw.optional.arm.FEAT_DotProd`.
    pub dot_prod: Option<i32>,
    /// `hw.optional.arm.FEAT_FP16`.
    pub fp16: Option<i32>,
    /// `hw.optional.arm.FEAT_I8MM`.
    pub i8mm: Option<i32>,
    /// `hw.optional.arm.FEAT_BF16`.
    pub bf16: Option<i32>,
    /// `hw.optional.arm.FEAT_SME2`.
    pub sme2: Option<i32>,
}

/// The recorded detection inputs of one host.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IsaInputs {
    /// An x86-64 host.
    X86(X86Inputs),
    /// An AArch64 host.
    Arm(ArmInputs),
}

/// The CPU entries of `features.disable` (RFC-0040 §12; ADR-0014 rule 4). Each one forces the
/// documented fallback: `avx512` to `X86V3`, `amx` to `X86V4`, `sme2` to `Arm86I8mm`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct KillSwitches {
    /// `avx512`: no AVX-512 and so no AMX.
    pub avx512: bool,
    /// `amx`: no AMX tiles.
    pub amx: bool,
    /// `sme2`: no SME2.
    pub sme2: bool,
}

impl KillSwitches {
    /// No CPU feature disabled.
    pub const NONE: Self = Self {
        avx512: false,
        amx: false,
        sme2: false,
    };

    /// The switches a `features.disable` list names. Entries for other parts of the system
    /// (`graph_capture`, `metal4`, ...) are not this provider's and are ignored.
    #[must_use]
    pub fn from_disable<'a>(entries: impl IntoIterator<Item = &'a str>) -> Self {
        let mut k = Self::NONE;
        for e in entries {
            match e {
                "avx512" => k.avx512 = true,
                "amx" => k.amx = true,
                "sme2" => k.sme2 = true,
                _ => {}
            }
        }
        k
    }
}

/// What detection decided: the level, the feature bits, and the AMX permission outcome,
/// which `enumerate` records in the profile when it was a refusal (RFC-0040 §13 code 13).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Detection {
    /// The level the provider dispatches to, reported as `DeviceClass::Cpu { isa }`.
    pub isa: CpuIsa,
    /// The capability bits, for `ProviderDevice::features`.
    pub features: DeviceFeatures,
    /// The AMX permission outcome; [`AmxPermission::NotRequested`] on AArch64.
    pub amx: AmxPermission,
}

impl Detection {
    /// The table for the detected level: what `enumerate` reports alongside the class.
    #[must_use]
    pub fn table(&self) -> DispatchTable {
        DispatchTable::for_level(self.isa)
    }
}

/// RFC-0040 §7.3's `X86V2` requirement.
const X86V2: [X86Feature; 2] = [X86Feature::Sse42, X86Feature::Popcnt];
/// RFC-0040 §7.3's `X86V3` requirement, on top of `X86V2`.
const X86V3: [X86Feature; 3] = [X86Feature::Avx2, X86Feature::Fma, X86Feature::F16c];
/// RFC-0040 §7.3's `X86V4` requirement, on top of `X86V3`.
const X86V4: [X86Feature; 4] = [
    X86Feature::Avx512f,
    X86Feature::Avx512bw,
    X86Feature::Avx512vl,
    X86Feature::Avx512dq,
];

/// The level and feature bits for `inputs` with `kill` applied. `None` when an x86-64 CPU
/// lacks `X86V2`'s SSE4.2 and POPCNT, below which the provider has no kernels.
#[must_use]
pub fn select(inputs: &IsaInputs, kill: &KillSwitches) -> Option<Detection> {
    match inputs {
        IsaInputs::X86(x) => select_x86(x, kill),
        IsaInputs::Arm(a) => Some(select_arm(a, kill)),
    }
}

fn select_x86(x: &X86Inputs, kill: &KillSwitches) -> Option<Detection> {
    let d = x.detected;
    if !d.all(&X86V2) {
        return None;
    }
    let v3 = d.all(&X86V3);
    let v4 = v3 && d.all(&X86V4) && !kill.avx512;
    let permitted = matches!(x.amx, AmxPermission::Granted | AmxPermission::NotNeeded);
    let amx = v4 && x.amx_hardware() && permitted && !kill.amx;
    let isa = if amx {
        CpuIsa::X86V4Amx
    } else if v4 {
        CpuIsa::X86V4
    } else if v3 {
        CpuIsa::X86V3
    } else {
        CpuIsa::X86V2
    };

    // What the CPU has, before the level gate.
    let mut present = DeviceFeatures::empty();
    let mut set = |cond: bool, f: DeviceFeatures| {
        if cond {
            present |= f;
        }
    };
    set(d.has(X86Feature::Avx2), DeviceFeatures::AVX2);
    set(d.has(X86Feature::Avxvnni), DeviceFeatures::AVX_VNNI);
    set(d.all(&X86V4), DeviceFeatures::AVX512);
    set(d.has(X86Feature::Avx512vnni), DeviceFeatures::AVX512_VNNI);
    set(d.has(X86Feature::Avx512bf16), DeviceFeatures::AVX512_BF16);
    set(d.has(X86Feature::Avx512fp16), DeviceFeatures::AVX512_FP16);
    set(
        x.cpuid7_edx & CPUID7_EDX_AMX_INT8 != 0,
        DeviceFeatures::AMX_INT8,
    );
    set(
        x.cpuid7_edx & CPUID7_EDX_AMX_BF16 != 0,
        DeviceFeatures::AMX_BF16,
    );
    let features = gate(present, isa);
    Some(Detection {
        isa,
        features,
        amx: x.amx,
    })
}

fn select_arm(a: &ArmInputs, kill: &KillSwitches) -> Detection {
    // Each level adds to the one below (RFC-0040 §7.3).
    let dot = a.dotprod && a.fp16;
    let i8mm = dot && a.i8mm && a.bf16;
    let sme2 = i8mm && a.sme2 && !kill.sme2;
    let isa = if sme2 {
        CpuIsa::Arm92Sme2
    } else if i8mm {
        CpuIsa::Arm86I8mm
    } else if dot {
        CpuIsa::Arm82Dot
    } else {
        CpuIsa::Arm80
    };

    let mut present = DeviceFeatures::empty();
    let mut set = |cond: bool, f: DeviceFeatures| {
        if cond {
            present |= f;
        }
    };
    set(a.dotprod, DeviceFeatures::NEON_DOTPROD);
    set(a.fp16, DeviceFeatures::NEON_FP16);
    set(a.i8mm, DeviceFeatures::NEON_I8MM);
    set(a.bf16, DeviceFeatures::NEON_BF16);
    // The SME2 routines are called only under `Arm92Sme2` (ADR-0402 item 3).
    set(a.sme2, DeviceFeatures::SME2);
    let features = gate(present, isa);
    Detection {
        isa,
        features,
        amx: AmxPermission::NotRequested,
    }
}

/// The level each reported feature bit belongs to: the lowest level whose kernels may use
/// it (RFC-0040 §7.3's table; "used when present" features belong to the level that lists
/// them). A bit not listed is never reported.
pub const FEATURE_LEVELS: [(DeviceFeatures, CpuIsa); 13] = [
    (DeviceFeatures::AVX2, CpuIsa::X86V3),
    (DeviceFeatures::AVX_VNNI, CpuIsa::X86V3),
    (DeviceFeatures::AVX512, CpuIsa::X86V4),
    (DeviceFeatures::AVX512_VNNI, CpuIsa::X86V4),
    (DeviceFeatures::AVX512_BF16, CpuIsa::X86V4),
    (DeviceFeatures::AVX512_FP16, CpuIsa::X86V4),
    (DeviceFeatures::AMX_INT8, CpuIsa::X86V4Amx),
    (DeviceFeatures::AMX_BF16, CpuIsa::X86V4Amx),
    (DeviceFeatures::NEON_DOTPROD, CpuIsa::Arm82Dot),
    (DeviceFeatures::NEON_FP16, CpuIsa::Arm82Dot),
    (DeviceFeatures::NEON_I8MM, CpuIsa::Arm86I8mm),
    (DeviceFeatures::NEON_BF16, CpuIsa::Arm86I8mm),
    (DeviceFeatures::SME2, CpuIsa::Arm92Sme2),
];

/// Whether `isa` is `level` or above it in the same chain.
fn covers(isa: CpuIsa, level: CpuIsa) -> bool {
    matches!(
        (rank(isa), rank(level)),
        (Some((a, i)), Some((b, l))) if a == b && l <= i
    )
}

/// The bits of `present` whose level `isa` covers.
fn gate(present: DeviceFeatures, isa: CpuIsa) -> DeviceFeatures {
    FEATURE_LEVELS
        .iter()
        .filter(|(f, level)| present.contains(*f) && covers(isa, *level))
        .fold(DeviceFeatures::empty(), |acc, (f, _)| acc | *f)
}

/// A level's architecture and its place in that architecture's chain.
fn rank(isa: CpuIsa) -> Option<(bool, u8)> {
    Some(match isa {
        CpuIsa::X86V2 => (true, 0),
        CpuIsa::X86V3 => (true, 1),
        CpuIsa::X86V4 => (true, 2),
        CpuIsa::X86V4Amx => (true, 3),
        CpuIsa::Arm80 => (false, 0),
        CpuIsa::Arm82Dot => (false, 1),
        CpuIsa::Arm86I8mm => (false, 2),
        CpuIsa::Arm92Sme2 => (false, 3),
        // `CpuIsa` is non-exhaustive; a level this crate does not know has no kernels here.
        _ => return None,
    })
}

/// The table `open` installs for a device whose class carries `class`, given what detection
/// found in the opening process (RFC-0040 §2.1, §7.3; ADR-0405 decision 3).
///
/// The table is always the class's own: `open` never loads a table the class does not name,
/// because the pack was built for the class. The host may run a level above the class, for
/// example when `enumerate` was refused AMX and `open` was granted it; the class's table
/// still runs correctly there.
///
/// # Errors
///
/// - `AmxDenied` (RFC-0040 §13 code 13, class `Unsupported`) when the class is `X86V4Amx`
///   and the OS refused this process AMX tile permission. The pack holds the `cpu.amx16`
///   layout, so the provider never falls back to `X86V4` kernels; the node re-probes.
/// - `OpUnsupported` (code 1, class `Unsupported`) for any other class this host cannot run
///   with the active kill switches, naming both levels.
pub fn table_for_class(class: CpuIsa, detected: &Detection) -> Result<DispatchTable, Error> {
    if class == CpuIsa::X86V4Amx
        && detected.isa != CpuIsa::X86V4Amx
        && let AmxPermission::Denied { errno } = detected.amx
    {
        return Err(Error {
            class: ErrorClass::Unsupported,
            code: ErrorCode::AMX_DENIED,
            message: format!(
                "the device class is X86V4Amx, but the OS refused this process AMX tile \
                 permission (errno {errno}), so detection here gives {:?}; re-probe the node",
                detected.isa
            ),
        });
    }
    if covers(detected.isa, class) {
        Ok(DispatchTable::for_level(class))
    } else {
        Err(Error {
            class: ErrorClass::Unsupported,
            code: ErrorCode::OP_UNSUPPORTED,
            message: format!(
                "the device class is {class:?}, but detection in this process gives {:?}; \
                 re-probe the node",
                detected.isa
            ),
        })
    }
}
