// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU ISA flags and microarchitecture level.
//!
//! Two sources give the same answers: the `flags` (x86-64) or `Features` (AArch64) line of
//! `/proc/cpuinfo`, which the kernel derives from CPUID and `AT_HWCAP`/`AT_HWCAP2`, and, on
//! the live host, the standard library's runtime detection macros, which read CPUID and
//! `getauxval` themselves. Fixtures use the first; [`IsaSource::Runtime`] prefers the second
//! for every feature a macro covers.

use std::collections::BTreeSet;

use syntropy_core::resource::{CpuArch, DeviceFeatures, IsaFlags, IsaLevel};

use crate::error::EnumError;

/// Where ISA facts come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IsaSource {
    /// The `/proc/cpuinfo` flags in the filesystem view. For fixtures and for features no
    /// runtime macro covers.
    CpuInfo,
    /// The standard library's runtime detection on the CPU this process runs on, with
    /// `/proc/cpuinfo` for the rest. Only meaningful for the live `/`.
    Runtime,
}

/// A feature E1 checks. `cpuinfo` is its name in `/proc/cpuinfo`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Feature {
    Cx16,
    LahfLm,
    Popcnt,
    Sse41,
    Sse42,
    Ssse3,
    Avx,
    Avx2,
    Bmi1,
    Bmi2,
    F16c,
    Fma,
    Lzcnt,
    Movbe,
    Xsave,
    Avx512f,
    Avx512bw,
    Avx512cd,
    Avx512dq,
    Avx512vl,
    Avx512Vnni,
    Avx512Bf16,
    Avx512Fp16,
    AvxVnni,
    AmxBf16,
    AmxInt8,
    DotProd,
    I8mm,
    Bf16,
    Sve2,
    Sme2,
    Fp16,
}

impl Feature {
    fn cpuinfo(self) -> &'static str {
        match self {
            Self::Cx16 => "cx16",
            Self::LahfLm => "lahf_lm",
            Self::Popcnt => "popcnt",
            Self::Sse41 => "sse4_1",
            Self::Sse42 => "sse4_2",
            Self::Ssse3 => "ssse3",
            Self::Avx => "avx",
            Self::Avx2 => "avx2",
            Self::Bmi1 => "bmi1",
            Self::Bmi2 => "bmi2",
            Self::F16c => "f16c",
            Self::Fma => "fma",
            Self::Lzcnt => "abm",
            Self::Movbe => "movbe",
            Self::Xsave => "xsave",
            Self::Avx512f => "avx512f",
            Self::Avx512bw => "avx512bw",
            Self::Avx512cd => "avx512cd",
            Self::Avx512dq => "avx512dq",
            Self::Avx512vl => "avx512vl",
            Self::Avx512Vnni => "avx512_vnni",
            Self::Avx512Bf16 => "avx512_bf16",
            Self::Avx512Fp16 => "avx512_fp16",
            Self::AvxVnni => "avx_vnni",
            Self::AmxBf16 => "amx_bf16",
            Self::AmxInt8 => "amx_int8",
            Self::DotProd => "asimddp",
            Self::I8mm => "i8mm",
            Self::Bf16 => "bf16",
            Self::Sve2 => "sve2",
            Self::Sme2 => "sme2",
            Self::Fp16 => "asimdhp",
        }
    }
}

const X86_V2: [Feature; 6] = [
    Feature::Cx16,
    Feature::LahfLm,
    Feature::Popcnt,
    Feature::Sse41,
    Feature::Sse42,
    Feature::Ssse3,
];
const X86_V3: [Feature; 9] = [
    Feature::Avx,
    Feature::Avx2,
    Feature::Bmi1,
    Feature::Bmi2,
    Feature::F16c,
    Feature::Fma,
    Feature::Lzcnt,
    Feature::Movbe,
    Feature::Xsave,
];
const X86_V4: [Feature; 5] = [
    Feature::Avx512f,
    Feature::Avx512bw,
    Feature::Avx512cd,
    Feature::Avx512dq,
    Feature::Avx512vl,
];

/// The ISA facts E1 records for a CPU device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Isa {
    pub level: IsaLevel,
    pub flags: IsaFlags,
    pub features: DeviceFeatures,
}

/// Answers "does the CPU have this feature?" from the chosen source.
pub(crate) struct Detector<'a> {
    pub source: IsaSource,
    /// Flags present on every processor entry of `/proc/cpuinfo`.
    pub cpuinfo: &'a BTreeSet<String>,
}

impl Detector<'_> {
    fn has(&self, f: Feature) -> bool {
        if self.source == IsaSource::Runtime
            && let Some(v) = runtime(f)
        {
            return v;
        }
        self.cpuinfo.contains(f.cpuinfo())
    }

    pub fn isa(&self, arch: CpuArch) -> Result<Isa, EnumError> {
        let mut flags = IsaFlags::empty();
        let mut features = DeviceFeatures::empty();
        let mut set = |cond: bool, i: IsaFlags, d: DeviceFeatures| {
            if cond {
                flags |= i;
                features |= d;
            }
        };
        let level = match arch {
            CpuArch::X86_64 => {
                let missing: Vec<&'static str> = X86_V2
                    .iter()
                    .filter(|f| !self.has(**f))
                    .map(|f| f.cpuinfo())
                    .collect();
                if !missing.is_empty() {
                    return Err(EnumError::BelowBaseline { missing });
                }
                set(
                    self.has(Feature::Avx2),
                    IsaFlags::AVX2,
                    DeviceFeatures::AVX2,
                );
                set(
                    self.has(Feature::Avx512f),
                    IsaFlags::AVX512F,
                    DeviceFeatures::AVX512,
                );
                set(
                    self.has(Feature::Avx512bw),
                    IsaFlags::AVX512BW,
                    DeviceFeatures::empty(),
                );
                set(
                    self.has(Feature::Avx512vl),
                    IsaFlags::AVX512VL,
                    DeviceFeatures::empty(),
                );
                set(
                    self.has(Feature::Avx512Vnni),
                    IsaFlags::AVX512VNNI,
                    DeviceFeatures::AVX512_VNNI,
                );
                set(
                    self.has(Feature::Avx512Bf16),
                    IsaFlags::AVX512BF16,
                    DeviceFeatures::AVX512_BF16,
                );
                set(
                    self.has(Feature::Avx512Fp16),
                    IsaFlags::AVX512FP16,
                    DeviceFeatures::AVX512_FP16,
                );
                set(
                    self.has(Feature::AvxVnni),
                    IsaFlags::empty(),
                    DeviceFeatures::AVX_VNNI,
                );
                // AMX: the hardware has the tiles. `DeviceFeatures` stays clear of AMX until
                // the kernel grants the tile-data permission, which E1 does not request.
                set(
                    self.has(Feature::AmxBf16),
                    IsaFlags::AMX_BF16,
                    DeviceFeatures::empty(),
                );
                set(
                    self.has(Feature::AmxInt8),
                    IsaFlags::AMX_INT8,
                    DeviceFeatures::empty(),
                );
                if X86_V4.iter().all(|f| self.has(*f)) && X86_V3.iter().all(|f| self.has(*f)) {
                    IsaLevel::X86_64V4
                } else if X86_V3.iter().all(|f| self.has(*f)) {
                    IsaLevel::X86_64V3
                } else {
                    IsaLevel::X86_64V2
                }
            }
            CpuArch::Aarch64 => {
                set(
                    self.has(Feature::DotProd),
                    IsaFlags::NEON_DOTPROD,
                    DeviceFeatures::NEON_DOTPROD,
                );
                set(
                    self.has(Feature::I8mm),
                    IsaFlags::NEON_I8MM,
                    DeviceFeatures::NEON_I8MM,
                );
                set(
                    self.has(Feature::Bf16),
                    IsaFlags::empty(),
                    DeviceFeatures::NEON_BF16,
                );
                set(
                    self.has(Feature::Sve2),
                    IsaFlags::SVE2,
                    DeviceFeatures::SVE2,
                );
                set(
                    self.has(Feature::Sme2),
                    IsaFlags::SME2,
                    DeviceFeatures::SME2,
                );
                set(
                    self.has(Feature::Fp16),
                    IsaFlags::empty(),
                    DeviceFeatures::NEON_FP16,
                );
                if self.has(Feature::Sve2) {
                    IsaLevel::Aarch64V9Sve2
                } else if self.has(Feature::DotProd) {
                    IsaLevel::Aarch64V8_2Dotprod
                } else {
                    IsaLevel::Aarch64V8_0
                }
            }
            // `CpuArch` is non-exhaustive; a new architecture needs its own rules.
            #[allow(unreachable_patterns)]
            other => {
                return Err(EnumError::UnsupportedArch {
                    found: format!("{other:?}"),
                });
            }
        };
        Ok(Isa {
            level,
            flags,
            features,
        })
    }
}

/// The standard library's answer for `f` on the running CPU, where a stable macro covers
/// it; `None` sends the question to `/proc/cpuinfo`.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn runtime(f: Feature) -> Option<bool> {
    use std::arch::is_x86_feature_detected as d;
    Some(match f {
        Feature::Cx16 => d!("cmpxchg16b"),
        Feature::Popcnt => d!("popcnt"),
        Feature::Sse41 => d!("sse4.1"),
        Feature::Sse42 => d!("sse4.2"),
        Feature::Ssse3 => d!("ssse3"),
        Feature::Avx => d!("avx"),
        Feature::Avx2 => d!("avx2"),
        Feature::Bmi1 => d!("bmi1"),
        Feature::Bmi2 => d!("bmi2"),
        Feature::F16c => d!("f16c"),
        Feature::Fma => d!("fma"),
        Feature::Lzcnt => d!("lzcnt"),
        Feature::Movbe => d!("movbe"),
        Feature::Xsave => d!("xsave"),
        Feature::Avx512f => d!("avx512f"),
        Feature::Avx512bw => d!("avx512bw"),
        Feature::Avx512cd => d!("avx512cd"),
        Feature::Avx512dq => d!("avx512dq"),
        Feature::Avx512vl => d!("avx512vl"),
        Feature::Avx512Vnni => d!("avx512vnni"),
        Feature::Avx512Bf16 => d!("avx512bf16"),
        Feature::Avx512Fp16 => d!("avx512fp16"),
        Feature::AvxVnni => d!("avxvnni"),
        _ => return None,
    })
}

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
fn runtime(f: Feature) -> Option<bool> {
    use std::arch::is_aarch64_feature_detected as d;
    Some(match f {
        Feature::DotProd => d!("dotprod"),
        Feature::I8mm => d!("i8mm"),
        Feature::Bf16 => d!("bf16"),
        Feature::Sve2 => d!("sve2"),
        Feature::Fp16 => d!("fp16"),
        _ => return None,
    })
}

#[cfg(not(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
)))]
fn runtime(_: Feature) -> Option<bool> {
    None
}

/// The flags present on every processor entry: the `flags` line on x86-64, `Features` on
/// AArch64.
pub(crate) fn common_flags(entries: &[Vec<&str>]) -> BTreeSet<String> {
    let mut iter = entries.iter();
    let Some(first) = iter.next() else {
        return BTreeSet::new();
    };
    let mut common: BTreeSet<String> = first.iter().map(|s| (*s).to_owned()).collect();
    for e in iter {
        common.retain(|f| e.contains(&f.as_str()));
    }
    common
}

#[cfg(test)]
mod tests {
    use super::*;

    fn isa(arch: CpuArch, flags: &str) -> Result<Isa, EnumError> {
        let set = flags.split_whitespace().map(str::to_owned).collect();
        Detector {
            source: IsaSource::CpuInfo,
            cpuinfo: &set,
        }
        .isa(arch)
    }

    const V2: &str = "cx16 lahf_lm popcnt sse4_1 sse4_2 ssse3";
    const V3: &str = "avx avx2 bmi1 bmi2 f16c fma abm movbe xsave";
    const V4: &str = "avx512f avx512bw avx512cd avx512dq avx512vl";

    #[test]
    fn x86_levels() {
        assert_eq!(isa(CpuArch::X86_64, V2).unwrap().level, IsaLevel::X86_64V2);
        let v3 = format!("{V2} {V3}");
        assert_eq!(isa(CpuArch::X86_64, &v3).unwrap().level, IsaLevel::X86_64V3);
        let v4 = format!("{V2} {V3} {V4}");
        assert_eq!(isa(CpuArch::X86_64, &v4).unwrap().level, IsaLevel::X86_64V4);
        // AVX-512 without the v3 set is still v2.
        let odd = format!("{V2} {V4}");
        assert_eq!(
            isa(CpuArch::X86_64, &odd).unwrap().level,
            IsaLevel::X86_64V2
        );
        // One v3 feature missing drops to v2.
        let short = format!("{V2} {}", V3.replace("movbe", ""));
        assert_eq!(
            isa(CpuArch::X86_64, &short).unwrap().level,
            IsaLevel::X86_64V2
        );
    }

    #[test]
    fn below_v2_is_refused() {
        assert_eq!(
            isa(CpuArch::X86_64, "cx16 popcnt sse4_1 sse4_2 ssse3"),
            Err(EnumError::BelowBaseline {
                missing: vec!["lahf_lm"]
            })
        );
    }

    #[test]
    fn amx_sets_isa_but_not_features() {
        let f = format!("{V2} amx_bf16 amx_int8 amx_tile");
        let i = isa(CpuArch::X86_64, &f).unwrap();
        assert_eq!(i.flags, IsaFlags::AMX_BF16 | IsaFlags::AMX_INT8);
        assert_eq!(i.features, DeviceFeatures::empty());
    }

    #[test]
    fn arm_levels() {
        let base = "fp asimd";
        assert_eq!(
            isa(CpuArch::Aarch64, base).unwrap().level,
            IsaLevel::Aarch64V8_0
        );
        let dp = isa(CpuArch::Aarch64, "fp asimd asimddp asimdhp").unwrap();
        assert_eq!(dp.level, IsaLevel::Aarch64V8_2Dotprod);
        assert_eq!(dp.flags, IsaFlags::NEON_DOTPROD);
        assert_eq!(
            dp.features,
            DeviceFeatures::NEON_DOTPROD | DeviceFeatures::NEON_FP16
        );
        let v9 = isa(CpuArch::Aarch64, "asimddp sve2 i8mm bf16").unwrap();
        assert_eq!(v9.level, IsaLevel::Aarch64V9Sve2);
    }

    #[test]
    fn common_flags_is_the_intersection() {
        let a = vec!["avx2", "avx512f", "sse4_2"];
        let b = vec!["avx2", "sse4_2"];
        let c = common_flags(&[a, b]);
        assert_eq!(c.into_iter().collect::<Vec<_>>(), ["avx2", "sse4_2"]);
        assert!(common_flags(&[]).is_empty());
    }
}
