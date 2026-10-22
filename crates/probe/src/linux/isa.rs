// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU ISA flags and the provisional `CpuIsa` level.
//!
//! Two sources give the same answers: the `flags` (x86-64) or `Features` (AArch64) line of
//! `/proc/cpuinfo`, which the kernel derives from CPUID and `AT_HWCAP`/`AT_HWCAP2`, and, on
//! the live host, the standard library's runtime detection macros, which read CPUID and
//! `getauxval` themselves. Fixtures use the first; [`IsaSource::Runtime`] prefers the second
//! for every feature a macro covers.
//!
//! The level is E1's provisional value (ADR-0405 decision 3; RFC-0010 §2.1): RFC-0040
//! §7.3's requirements per [`CpuIsa`] level, checked on these flags. The CPU provider decides
//! the class at E5, and its value replaces this one.

use std::collections::BTreeSet;

use syntropy_core::resource::{CpuArch, CpuIsa, DeviceFeatures, IsaFlags};

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
    Avx2,
    F16c,
    Fma,
    Avx512f,
    Avx512bw,
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
            Self::Avx2 => "avx2",
            Self::F16c => "f16c",
            Self::Fma => "fma",
            Self::Avx512f => "avx512f",
            Self::Avx512bw => "avx512bw",
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

/// The x86-64 floor below which E1 refuses the CPU (`BelowBaseline`): the psABI x86-64-v2
/// set, not only RFC-0040 §7.3's "SSE4.2, POPCNT" for `X86V2`. ADR-0014 and ADR-0020 name
/// x86-64-v2 as the platform floor, and the core's ggml CPU backend is built for that baseline
/// (RFC-0040 §8.1), so a CPU with SSE4.2 and POPCNT but without CX16, LAHF-SAHF, SSE4.1 or
/// SSSE3 cannot run it. The psABI set also names SSE3; it is not listed, because every CPU with
/// SSSE3 has it.
const X86_V2: [Feature; 6] = [
    Feature::Cx16,
    Feature::LahfLm,
    Feature::Popcnt,
    Feature::Sse41,
    Feature::Sse42,
    Feature::Ssse3,
];
/// RFC-0040 §7.3's `X86V3` requirement: AVX2, FMA and F16C.
const X86V3: [Feature; 3] = [Feature::Avx2, Feature::Fma, Feature::F16c];

/// RFC-0040 §7.3's `X86V4` requirement, on top of `X86V3`: what `DeviceFeatures::AVX512`
/// means.
const AVX512_KERNELS: [Feature; 4] = [
    Feature::Avx512f,
    Feature::Avx512bw,
    Feature::Avx512vl,
    Feature::Avx512dq,
];

/// The ISA facts E1 records for a CPU device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Isa {
    /// The provisional class level.
    pub level: CpuIsa,
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
                // `IsaFlags` records each AVX-512 extension the CPU has. `DeviceFeatures::AVX512`
                // is the kernel gate, so it is set exactly when the level is `X86V4`: RFC-0040
                // §7.3's `X86V4` requirement (AVX-512 F, BW, VL and DQ together) on top of
                // `X86V3`. VNNI, BF16 and FP16 are "used when present" on top of `X86V4`, so
                // their feature bits need the gate too. A CPU with AVX-512 but not the `X86V3`
                // set is `X86V2` with no AVX-512 feature bit.
                let v3 = X86V3.iter().all(|f| self.has(*f));
                let avx512 = v3 && AVX512_KERNELS.iter().all(|f| self.has(*f));
                let gated = |f: Feature| avx512 && self.has(f);
                set(avx512, IsaFlags::empty(), DeviceFeatures::AVX512);
                set(
                    self.has(Feature::Avx512f),
                    IsaFlags::AVX512F,
                    DeviceFeatures::empty(),
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
                    DeviceFeatures::empty(),
                );
                set(
                    gated(Feature::Avx512Vnni),
                    IsaFlags::empty(),
                    DeviceFeatures::AVX512_VNNI,
                );
                set(
                    self.has(Feature::Avx512Bf16),
                    IsaFlags::AVX512BF16,
                    DeviceFeatures::empty(),
                );
                set(
                    gated(Feature::Avx512Bf16),
                    IsaFlags::empty(),
                    DeviceFeatures::AVX512_BF16,
                );
                set(
                    self.has(Feature::Avx512Fp16),
                    IsaFlags::AVX512FP16,
                    DeviceFeatures::empty(),
                );
                set(
                    gated(Feature::Avx512Fp16),
                    IsaFlags::empty(),
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
                // Never `X86V4Amx`: that level needs the tile-data permission granted, which
                // E1 does not request; the CPU provider requests it at E5 (ADR-0405 decision 3).
                if avx512 {
                    CpuIsa::X86V4
                } else if v3 {
                    CpuIsa::X86V3
                } else {
                    CpuIsa::X86V2
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
                // Each level adds to the one below (RFC-0040 §7.3). SVE2 is not in any
                // level's requirement, so SVE2 without SME2 stays at `Arm86I8mm`.
                let dot = self.has(Feature::DotProd) && self.has(Feature::Fp16);
                let i8mm = dot && self.has(Feature::I8mm) && self.has(Feature::Bf16);
                if i8mm && self.has(Feature::Sme2) {
                    CpuIsa::Arm92Sme2
                } else if i8mm {
                    CpuIsa::Arm86I8mm
                } else if dot {
                    CpuIsa::Arm82Dot
                } else {
                    CpuIsa::Arm80
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
        Feature::Avx2 => d!("avx2"),
        Feature::F16c => d!("f16c"),
        Feature::Fma => d!("fma"),
        Feature::Avx512f => d!("avx512f"),
        Feature::Avx512bw => d!("avx512bw"),
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
        assert_eq!(isa(CpuArch::X86_64, V2).unwrap().level, CpuIsa::X86V2);
        let v3 = format!("{V2} {V3}");
        assert_eq!(isa(CpuArch::X86_64, &v3).unwrap().level, CpuIsa::X86V3);
        let v4 = format!("{V2} {V3} {V4}");
        assert_eq!(isa(CpuArch::X86_64, &v4).unwrap().level, CpuIsa::X86V4);
        // RFC-0040 §7.3's lists: AVX2, FMA and F16C for `X86V3`; F, BW, VL and DQ for `X86V4`.
        let minimal = format!("{V2} avx2 fma f16c avx512f avx512bw avx512vl avx512dq");
        assert_eq!(isa(CpuArch::X86_64, &minimal).unwrap().level, CpuIsa::X86V4);
        // AVX-512 without the v3 set is still v2.
        let odd = format!("{V2} {V4}");
        assert_eq!(isa(CpuArch::X86_64, &odd).unwrap().level, CpuIsa::X86V2);
        // Each of AVX2, FMA and F16C missing drops to v2.
        for drop in ["avx2 ", "fma ", "f16c "] {
            let short = format!("{V2} {} ", V3).replace(drop, "");
            assert_eq!(
                isa(CpuArch::X86_64, &short).unwrap().level,
                CpuIsa::X86V2,
                "without {drop}"
            );
        }
        // Each of F, BW, VL and DQ missing drops to v3.
        for drop in ["avx512f ", "avx512bw ", "avx512vl ", "avx512dq "] {
            let short = format!("{V2} {V3} {V4} ").replace(drop, "");
            assert_eq!(
                isa(CpuArch::X86_64, &short).unwrap().level,
                CpuIsa::X86V3,
                "without {drop}"
            );
        }
    }

    /// The AVX-512 feature bits follow the level: AVX-512 without the `X86V3` set is `X86V2`,
    /// with every AVX-512 extension in `IsaFlags` and none in `DeviceFeatures`.
    #[test]
    fn avx512_features_need_the_x86v4_level() {
        let ext = "avx512_vnni avx512_bf16 avx512_fp16";
        let i = isa(CpuArch::X86_64, &format!("{V2} {V4} {ext}")).unwrap();
        assert_eq!(i.level, CpuIsa::X86V2);
        assert_eq!(i.features, DeviceFeatures::empty());
        assert!(i.flags.contains(
            IsaFlags::AVX512F
                | IsaFlags::AVX512BW
                | IsaFlags::AVX512VL
                | IsaFlags::AVX512VNNI
                | IsaFlags::AVX512BF16
                | IsaFlags::AVX512FP16
        ));
        let v4 = isa(CpuArch::X86_64, &format!("{V2} {V3} {V4} {ext}")).unwrap();
        assert_eq!(v4.level, CpuIsa::X86V4);
        assert!(v4.features.contains(DeviceFeatures::AVX512));
    }

    #[test]
    fn amx_hardware_is_never_x86v4amx_at_e1() {
        let f = format!("{V2} {V3} {V4} amx_bf16 amx_int8 amx_tile");
        assert_eq!(isa(CpuArch::X86_64, &f).unwrap().level, CpuIsa::X86V4);
    }

    #[test]
    fn avx512_feature_needs_f_bw_vl_dq() {
        let ext = "avx512_vnni avx512_bf16 avx512_fp16";
        let all = format!("{V2} {V3} avx512f avx512bw avx512vl avx512dq {ext}");
        let i = isa(CpuArch::X86_64, &all).unwrap();
        assert_eq!(
            i.features,
            DeviceFeatures::AVX2
                | DeviceFeatures::AVX512
                | DeviceFeatures::AVX512_VNNI
                | DeviceFeatures::AVX512_BF16
                | DeviceFeatures::AVX512_FP16
        );
        // Each of F, BW, VL and DQ missing in turn: no AVX-512 feature bits at all (AVX2
        // stays), while `IsaFlags` still records what the CPU has.
        for drop in ["avx512f ", "avx512bw ", "avx512vl ", "avx512dq "] {
            let i = isa(CpuArch::X86_64, &all.replace(drop, "")).unwrap();
            assert_eq!(i.features, DeviceFeatures::AVX2, "without {drop}");
            assert!(
                i.flags
                    .contains(IsaFlags::AVX512VNNI | IsaFlags::AVX512BF16 | IsaFlags::AVX512FP16),
                "IsaFlags keeps each extension without {drop}"
            );
        }
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
        assert_eq!(isa(CpuArch::Aarch64, base).unwrap().level, CpuIsa::Arm80);
        let dp = isa(CpuArch::Aarch64, "fp asimd asimddp asimdhp").unwrap();
        assert_eq!(dp.level, CpuIsa::Arm82Dot);
        assert_eq!(dp.flags, IsaFlags::NEON_DOTPROD);
        assert_eq!(
            dp.features,
            DeviceFeatures::NEON_DOTPROD | DeviceFeatures::NEON_FP16
        );
        // Dot product without FP16 arithmetic is not `Arm82Dot`.
        let no_fp16 = isa(CpuArch::Aarch64, "fp asimd asimddp").unwrap();
        assert_eq!(no_fp16.level, CpuIsa::Arm80);
        let i8mm = isa(CpuArch::Aarch64, "asimddp asimdhp i8mm bf16").unwrap();
        assert_eq!(i8mm.level, CpuIsa::Arm86I8mm);
        // I8MM without BF16 stays at `Arm82Dot`.
        let no_bf16 = isa(CpuArch::Aarch64, "asimddp asimdhp i8mm").unwrap();
        assert_eq!(no_bf16.level, CpuIsa::Arm82Dot);
        let sme2 = isa(CpuArch::Aarch64, "asimddp asimdhp i8mm bf16 sme2").unwrap();
        assert_eq!(sme2.level, CpuIsa::Arm92Sme2);
        // I8MM and BF16 without the `Arm82Dot` base are not `Arm86I8mm`.
        for flags in ["fp asimd i8mm bf16", "fp asimd asimddp i8mm bf16"] {
            assert_eq!(
                isa(CpuArch::Aarch64, flags).unwrap().level,
                CpuIsa::Arm80,
                "{flags}"
            );
        }
        // SME2 without the `Arm86I8mm` base is not `Arm92Sme2`.
        let bare_sme2 = isa(CpuArch::Aarch64, "asimddp asimdhp sme2").unwrap();
        assert_eq!(bare_sme2.level, CpuIsa::Arm82Dot);
    }

    /// A CPU with SVE2 but no SME2 never gives `Arm92Sme2`.
    #[test]
    fn sve2_without_sme2_is_never_arm92sme2() {
        for flags in [
            "asimddp asimdhp i8mm bf16 sve2",
            "asimddp asimdhp sve sve2",
            "fp asimd sve2",
        ] {
            let i = isa(CpuArch::Aarch64, flags).unwrap();
            assert_ne!(i.level, CpuIsa::Arm92Sme2, "{flags}");
            assert!(i.flags.contains(IsaFlags::SVE2), "{flags}");
        }
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
