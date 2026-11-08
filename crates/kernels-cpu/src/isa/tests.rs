// SPDX-License-Identifier: MIT OR Apache-2.0

//! The level table test over recorded detection inputs, the kill switches, the `open` check,
//! and the live host's own level.

use syntropy_core::ErrorClass;
use syntropy_kernels::{CpuIsa, DeviceFeatures as F, ErrorCode};

use super::*;

use X86Feature as X;

/// `EPERM`, the `errno` a seccomp profile or a kernel policy gives a refused request.
const EPERM: i32 = 1;

const V2: &[X86Feature] = &[X::Sse42, X::Popcnt];
const V3: &[X86Feature] = &[X::Sse42, X::Popcnt, X::Avx2, X::Fma, X::F16c];
const V4: &[X86Feature] = &[
    X::Sse42,
    X::Popcnt,
    X::Avx2,
    X::Fma,
    X::F16c,
    X::Avx512f,
    X::Avx512bw,
    X::Avx512vl,
    X::Avx512dq,
];
/// XCR0 of an OS that saves x87, SSE, AVX and the AVX-512 state: bits 0–2 and 5–7.
const XCR0_AVX512: u64 = 0b1110_0111;
/// The same plus the tile configuration and data: bits 17 and 18.
const XCR0_AMX: u64 = XCR0_AVX512 | (1 << 17) | (1 << 18);
/// CPUID leaf 7 EDX of a CPU with AMX-BF16, AMX-TILE and AMX-INT8.
const EDX_AMX: u32 = (1 << 22) | (1 << 24) | (1 << 25);

fn feats(base: &[X86Feature], extra: &[X86Feature]) -> X86Features {
    base.iter().chain(extra).copied().collect()
}

fn x86(detected: X86Features, cpuid7_edx: u32, xcr0: u64, amx: AmxPermission) -> IsaInputs {
    IsaInputs::X86(X86Inputs {
        detected,
        cpuid7_edx,
        xcr0,
        amx,
    })
}

/// A Sapphire Rapids-class CPU: `X86V4` with VNNI, BF16 and FP16, AVX-VNNI and the AMX bits.
fn spr(amx: AmxPermission) -> IsaInputs {
    let d = feats(
        V4,
        &[X::Avx512vnni, X::Avx512bf16, X::Avx512fp16, X::Avxvnni],
    );
    x86(d, EDX_AMX, XCR0_AMX, amx)
}

fn linux_arm(dotprod: bool, fp16: bool, i8mm: bool, bf16: bool, hwcap2: u64) -> IsaInputs {
    let a = ArmInputs {
        dotprod,
        fp16,
        i8mm,
        bf16,
        sve2: false,
        sme2: false,
    };
    IsaInputs::Arm(a.with_hwcap2(hwcap2))
}

fn mac(
    dot_prod: Option<i32>,
    fp16: Option<i32>,
    i8mm: Option<i32>,
    bf16: Option<i32>,
    sme2: Option<i32>,
) -> IsaInputs {
    IsaInputs::Arm(ArmInputs::from_mac_sysctls(&MacSysctls {
        dot_prod,
        fp16,
        i8mm,
        bf16,
        sme2,
    }))
}

/// One row: what the inputs are, the kill switches, the level expected, and the feature bits
/// expected (exactly).
struct Row {
    name: &'static str,
    inputs: IsaInputs,
    kill: KillSwitches,
    isa: CpuIsa,
    features: F,
}

const NONE: KillSwitches = KillSwitches::NONE;
const NO_AVX512: KillSwitches = KillSwitches {
    avx512: true,
    amx: false,
    sme2: false,
};
const NO_AMX: KillSwitches = KillSwitches {
    avx512: false,
    amx: true,
    sme2: false,
};
const NO_SME2: KillSwitches = KillSwitches {
    avx512: false,
    amx: false,
    sme2: true,
};

#[allow(clippy::too_many_lines)]
fn rows() -> Vec<Row> {
    let v4_feats = F::AVX2 | F::AVX512;
    let spr_v4 =
        F::AVX2 | F::AVX_VNNI | F::AVX512 | F::AVX512_VNNI | F::AVX512_BF16 | F::AVX512_FP16;
    let y = Some(1);
    let n = Some(0);
    vec![
        // ---- x86-64 ----
        Row {
            name: "SSE4.2 and POPCNT only",
            inputs: x86(feats(V2, &[]), 0, 0b111, AmxPermission::NotRequested),
            kill: NONE,
            isa: CpuIsa::X86V2,
            features: F::empty(),
        },
        Row {
            name: "AVX2, FMA, F16C",
            inputs: x86(feats(V3, &[]), 0, 0b111, AmxPermission::NotRequested),
            kill: NONE,
            isa: CpuIsa::X86V3,
            features: F::AVX2,
        },
        Row {
            name: "X86V3 with AVX-VNNI",
            inputs: x86(
                feats(V3, &[X::Avxvnni]),
                0,
                0b111,
                AmxPermission::NotRequested,
            ),
            kill: NONE,
            isa: CpuIsa::X86V3,
            features: F::AVX2 | F::AVX_VNNI,
        },
        Row {
            name: "AVX2 and FMA without F16C",
            inputs: x86(
                feats(V2, &[X::Avx2, X::Fma]),
                0,
                0b111,
                AmxPermission::NotRequested,
            ),
            kill: NONE,
            isa: CpuIsa::X86V2,
            // AVX2 belongs to `X86V3`, which this CPU does not reach.
            features: F::empty(),
        },
        Row {
            name: "AVX-512 without the X86V3 set",
            inputs: x86(
                feats(V2, &[X::Avx512f, X::Avx512bw, X::Avx512vl, X::Avx512dq]),
                0,
                XCR0_AVX512,
                AmxPermission::NotRequested,
            ),
            kill: NONE,
            isa: CpuIsa::X86V2,
            features: F::empty(),
        },
        Row {
            name: "AVX-512 F, BW, VL without DQ",
            inputs: x86(
                feats(V3, &[X::Avx512f, X::Avx512bw, X::Avx512vl, X::Avx512vnni]),
                0,
                XCR0_AVX512,
                AmxPermission::NotRequested,
            ),
            kill: NONE,
            isa: CpuIsa::X86V3,
            features: F::AVX2,
        },
        Row {
            name: "X86V4 with VNNI and BF16",
            inputs: x86(
                feats(V4, &[X::Avx512vnni, X::Avx512bf16]),
                0,
                XCR0_AVX512,
                AmxPermission::NotRequested,
            ),
            kill: NONE,
            isa: CpuIsa::X86V4,
            features: v4_feats | F::AVX512_VNNI | F::AVX512_BF16,
        },
        Row {
            name: "AMX, permission granted",
            inputs: spr(AmxPermission::Granted),
            kill: NONE,
            isa: CpuIsa::X86V4Amx,
            features: spr_v4 | F::AMX_INT8 | F::AMX_BF16,
        },
        Row {
            name: "AMX, permission denied",
            inputs: spr(AmxPermission::Denied { errno: EPERM }),
            kill: NONE,
            isa: CpuIsa::X86V4,
            features: spr_v4,
        },
        Row {
            name: "AMX, no request needed (outside Linux)",
            inputs: spr(AmxPermission::NotNeeded),
            kill: NONE,
            isa: CpuIsa::X86V4Amx,
            features: spr_v4 | F::AMX_INT8 | F::AMX_BF16,
        },
        Row {
            name: "AMX, not requested",
            inputs: spr(AmxPermission::NotRequested),
            kill: NONE,
            isa: CpuIsa::X86V4,
            features: spr_v4,
        },
        Row {
            name: "AMX without AMX-BF16",
            inputs: x86(
                feats(V4, &[]),
                (1 << 24) | (1 << 25),
                XCR0_AMX,
                AmxPermission::Granted,
            ),
            kill: NONE,
            isa: CpuIsa::X86V4Amx,
            features: v4_feats | F::AMX_INT8,
        },
        Row {
            name: "AMX-TILE without AMX-INT8",
            inputs: x86(feats(V4, &[]), 1 << 24, XCR0_AMX, AmxPermission::Granted),
            kill: NONE,
            isa: CpuIsa::X86V4,
            features: v4_feats,
        },
        Row {
            name: "AMX, OS does not save the tile data (XCR0 bit 18 clear)",
            inputs: x86(
                feats(V4, &[]),
                EDX_AMX,
                XCR0_AMX & !(1 << 18),
                AmxPermission::NotNeeded,
            ),
            kill: NONE,
            isa: CpuIsa::X86V4,
            features: v4_feats,
        },
        Row {
            name: "AMX bits on a CPU without AVX-512",
            inputs: x86(feats(V3, &[]), EDX_AMX, XCR0_AMX, AmxPermission::Granted),
            kill: NONE,
            isa: CpuIsa::X86V3,
            features: F::AVX2,
        },
        Row {
            name: "kill switch amx",
            inputs: spr(AmxPermission::Granted),
            kill: NO_AMX,
            isa: CpuIsa::X86V4,
            features: spr_v4,
        },
        Row {
            name: "kill switch avx512 on an AMX CPU",
            inputs: spr(AmxPermission::Granted),
            kill: NO_AVX512,
            isa: CpuIsa::X86V3,
            features: F::AVX2 | F::AVX_VNNI,
        },
        Row {
            name: "kill switch avx512 on X86V4",
            inputs: x86(
                feats(V4, &[X::Avx512vnni]),
                0,
                XCR0_AVX512,
                AmxPermission::NotRequested,
            ),
            kill: NO_AVX512,
            isa: CpuIsa::X86V3,
            features: F::AVX2,
        },
        Row {
            name: "kill switch sme2 on x86 changes nothing",
            inputs: spr(AmxPermission::Granted),
            kill: NO_SME2,
            isa: CpuIsa::X86V4Amx,
            features: spr_v4 | F::AMX_INT8 | F::AMX_BF16,
        },
        // ---- AArch64, Linux (detection macro plus AT_HWCAP2) ----
        Row {
            name: "Linux, NEON only",
            inputs: linux_arm(false, false, false, false, 0),
            kill: NONE,
            isa: CpuIsa::Arm80,
            features: F::empty(),
        },
        Row {
            name: "Linux, dotprod without FP16",
            inputs: linux_arm(true, false, false, false, 0),
            kill: NONE,
            isa: CpuIsa::Arm80,
            features: F::empty(),
        },
        Row {
            name: "Linux, dotprod and FP16",
            inputs: linux_arm(true, true, false, false, 0),
            kill: NONE,
            isa: CpuIsa::Arm82Dot,
            features: F::NEON_DOTPROD | F::NEON_FP16,
        },
        Row {
            name: "Linux, I8MM without BF16",
            inputs: linux_arm(true, true, true, false, 0),
            kill: NONE,
            isa: CpuIsa::Arm82Dot,
            features: F::NEON_DOTPROD | F::NEON_FP16,
        },
        Row {
            name: "Linux, I8MM and BF16, SVE2",
            inputs: IsaInputs::Arm(ArmInputs {
                dotprod: true,
                fp16: true,
                i8mm: true,
                bf16: true,
                sve2: true,
                sme2: false,
            }),
            kill: NONE,
            isa: CpuIsa::Arm86I8mm,
            // SVE2 belongs to no level, so it is never reported.
            features: F::NEON_DOTPROD | F::NEON_FP16 | F::NEON_I8MM | F::NEON_BF16,
        },
        Row {
            name: "Linux, HWCAP2_SME2",
            inputs: linux_arm(true, true, true, true, HWCAP2_SME2),
            kill: NONE,
            isa: CpuIsa::Arm92Sme2,
            features: F::NEON_DOTPROD | F::NEON_FP16 | F::NEON_I8MM | F::NEON_BF16 | F::SME2,
        },
        Row {
            name: "Linux, other AT_HWCAP2 bits only",
            inputs: linux_arm(true, true, true, true, !HWCAP2_SME2),
            kill: NONE,
            isa: CpuIsa::Arm86I8mm,
            features: F::NEON_DOTPROD | F::NEON_FP16 | F::NEON_I8MM | F::NEON_BF16,
        },
        Row {
            name: "Linux, SME2 without I8MM",
            inputs: linux_arm(true, true, false, false, HWCAP2_SME2),
            kill: NONE,
            isa: CpuIsa::Arm82Dot,
            features: F::NEON_DOTPROD | F::NEON_FP16,
        },
        // ---- AArch64, macOS (hw.optional.arm.FEAT_* sysctls) ----
        Row {
            name: "macOS, M1-class sysctls",
            inputs: mac(y, y, n, n, None),
            kill: NONE,
            isa: CpuIsa::Arm82Dot,
            features: F::NEON_DOTPROD | F::NEON_FP16,
        },
        Row {
            name: "macOS, M2-class sysctls",
            inputs: mac(y, y, y, y, n),
            kill: NONE,
            isa: CpuIsa::Arm86I8mm,
            features: F::NEON_DOTPROD | F::NEON_FP16 | F::NEON_I8MM | F::NEON_BF16,
        },
        Row {
            name: "macOS, M4-class sysctls",
            inputs: mac(y, y, y, y, y),
            kill: NONE,
            isa: CpuIsa::Arm92Sme2,
            features: F::NEON_DOTPROD | F::NEON_FP16 | F::NEON_I8MM | F::NEON_BF16 | F::SME2,
        },
        Row {
            name: "macOS, kill switch sme2",
            inputs: mac(y, y, y, y, y),
            kill: NO_SME2,
            isa: CpuIsa::Arm86I8mm,
            features: F::NEON_DOTPROD | F::NEON_FP16 | F::NEON_I8MM | F::NEON_BF16,
        },
        Row {
            name: "macOS, kill switches avx512 and amx change nothing",
            inputs: mac(y, y, y, y, y),
            kill: KillSwitches {
                avx512: true,
                amx: true,
                sme2: false,
            },
            isa: CpuIsa::Arm92Sme2,
            features: F::NEON_DOTPROD | F::NEON_FP16 | F::NEON_I8MM | F::NEON_BF16 | F::SME2,
        },
        Row {
            name: "macOS, no sysctl exists",
            inputs: mac(None, None, None, None, None),
            kill: NONE,
            isa: CpuIsa::Arm80,
            features: F::empty(),
        },
        Row {
            name: "macOS, a sysctl reads 0",
            inputs: mac(y, n, y, y, y),
            kill: NONE,
            isa: CpuIsa::Arm80,
            features: F::empty(),
        },
    ]
}

#[test]
fn recorded_inputs_select_the_expected_level() {
    let mut failures = Vec::new();
    for row in rows() {
        let Some(d) = select(&row.inputs, &row.kill) else {
            failures.push(format!("{}: no level", row.name));
            continue;
        };
        if d.isa != row.isa || d.features != row.features {
            failures.push(format!(
                "{}: got {:?} {:?}, expected {:?} {:?}",
                row.name, d.isa, d.features, row.isa, row.features
            ));
        }
        // The table enumerate reports is the selected level's.
        if d.table().isa() != row.isa {
            failures.push(format!("{}: table for {:?}", row.name, d.table().isa()));
        }
        // The AMX outcome is carried through for the profile.
        if let IsaInputs::X86(x) = row.inputs
            && d.amx != x.amx
        {
            failures.push(format!("{}: amx outcome {:?}", row.name, d.amx));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The level each bit belongs to, written out here apart from `FEATURE_LEVELS`.
fn bit_level(f: F) -> Option<CpuIsa> {
    let levels = [
        (F::AVX2 | F::AVX_VNNI, CpuIsa::X86V3),
        (
            F::AVX512 | F::AVX512_VNNI | F::AVX512_BF16 | F::AVX512_FP16,
            CpuIsa::X86V4,
        ),
        (F::AMX_INT8 | F::AMX_BF16, CpuIsa::X86V4Amx),
        (F::NEON_DOTPROD | F::NEON_FP16, CpuIsa::Arm82Dot),
        (F::NEON_I8MM | F::NEON_BF16, CpuIsa::Arm86I8mm),
        (F::SME2, CpuIsa::Arm92Sme2),
    ];
    levels.iter().find(|(s, _)| s.contains(f)).map(|(_, l)| *l)
}

/// The levels the selected level covers: itself and those below it in its chain.
fn covered(isa: CpuIsa) -> &'static [CpuIsa] {
    use CpuIsa::*;
    const X86: [CpuIsa; 4] = [X86V2, X86V3, X86V4, X86V4Amx];
    const ARM: [CpuIsa; 4] = [Arm80, Arm82Dot, Arm86I8mm, Arm92Sme2];
    for chain in [&X86, &ARM] {
        if let Some(i) = chain.iter().position(|l| *l == isa) {
            return &chain[..=i];
        }
    }
    &[]
}

#[test]
fn no_bit_belongs_to_a_level_above_the_selected_one() {
    let kills = [NONE, NO_AVX512, NO_AMX, NO_SME2];
    for row in rows() {
        for kill in kills {
            let d = select(&row.inputs, &kill).unwrap();
            for f in d.features.iter() {
                let level = bit_level(f);
                assert!(
                    level.is_some_and(|l| covered(d.isa).contains(&l)),
                    "{} with {kill:?}: {f:?} ({level:?}) above {:?}",
                    row.name,
                    d.isa
                );
            }
            // A kill switch only ever removes bits.
            let all = select(&row.inputs, &NONE).unwrap();
            assert!(all.features.contains(d.features), "{}: {kill:?}", row.name);
        }
    }
    // `FEATURE_LEVELS` agrees with the list above.
    for (f, level) in FEATURE_LEVELS {
        assert_eq!(bit_level(f), Some(level), "{f:?}");
    }
}

#[test]
fn below_x86v2_has_no_level() {
    for missing in [X::Sse42, X::Popcnt] {
        let d: X86Features = V3.iter().copied().filter(|f| *f != missing).collect();
        let inputs = x86(d, 0, 0b111, AmxPermission::NotRequested);
        assert_eq!(select(&inputs, &NONE), None, "without {missing:?}");
    }
}

#[test]
fn each_kill_switch_lowers_the_level_and_its_table() {
    let cases = [
        (
            spr(AmxPermission::Granted),
            "amx",
            CpuIsa::X86V4Amx,
            CpuIsa::X86V4,
        ),
        (
            spr(AmxPermission::Granted),
            "avx512",
            CpuIsa::X86V4Amx,
            CpuIsa::X86V3,
        ),
        (
            mac(Some(1), Some(1), Some(1), Some(1), Some(1)),
            "sme2",
            CpuIsa::Arm92Sme2,
            CpuIsa::Arm86I8mm,
        ),
    ];
    for (inputs, switch, before, after) in cases {
        let on = select(&inputs, &NONE).unwrap();
        let off = select(&inputs, &KillSwitches::from_disable([switch])).unwrap();
        assert_eq!((on.isa, on.table().isa()), (before, before), "{switch}");
        assert_eq!((off.isa, off.table().isa()), (after, after), "{switch}");
    }
}

#[test]
fn kill_switches_parse_from_features_disable() {
    let k = KillSwitches::from_disable(["graph_capture", "amx", "metal4", "sme2"]);
    assert_eq!(
        k,
        KillSwitches {
            avx512: false,
            amx: true,
            sme2: true
        }
    );
    assert_eq!(KillSwitches::from_disable([]), KillSwitches::NONE);
    assert_eq!(KillSwitches::from_disable(["avx512"]), NO_AVX512);
    // Names are exact: RFC-0040 §12's entries, lower case.
    assert_eq!(
        KillSwitches::from_disable(["AMX", "avx-512", "sme"]),
        KillSwitches::NONE
    );
}

#[test]
fn x86_feature_names_and_bits_are_distinct() {
    let mut names: Vec<&str> = X86Feature::ALL.iter().map(|f| f.name()).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), X86Feature::ALL.len());
    let all: X86Features = X86Feature::ALL.into_iter().collect();
    for f in X86Feature::ALL {
        assert!(all.has(f));
        assert!(!X86Features::NONE.has(f));
        assert_eq!(X86Features::NONE.with(f).0.count_ones(), 1);
    }
}

// ---- `open` against the class `enumerate` reported (ADR-0405 decision 3) ----

#[test]
fn amx_denied_at_enumerate_reports_x86v4_and_open_agrees() {
    // The probe runner's request is refused: enumerate reports `X86V4` and its table.
    let denied = spr(AmxPermission::Denied { errno: EPERM });
    let at_enumerate = select(&denied, &NONE).unwrap();
    assert_eq!(at_enumerate.isa, CpuIsa::X86V4);
    assert_eq!(at_enumerate.table().isa(), CpuIsa::X86V4);
    // `open` detects again; refused again, it installs the `X86V4` table.
    let at_open = select(&denied, &NONE).unwrap();
    assert_eq!(
        table_for_class(at_enumerate.isa, &at_open).unwrap().isa(),
        CpuIsa::X86V4
    );
    // Granted in the serving runner, `open` still installs the class's table, not AMX's.
    let granted = select(&spr(AmxPermission::Granted), &NONE).unwrap();
    assert_eq!(granted.isa, CpuIsa::X86V4Amx);
    assert_eq!(
        table_for_class(at_enumerate.isa, &granted).unwrap().isa(),
        CpuIsa::X86V4
    );
}

#[test]
fn open_under_x86v4amx_fails_with_amx_denied_when_refused() {
    let at_enumerate = select(&spr(AmxPermission::Granted), &NONE).unwrap();
    assert_eq!(at_enumerate.isa, CpuIsa::X86V4Amx);
    assert_eq!(
        table_for_class(at_enumerate.isa, &at_enumerate)
            .unwrap()
            .isa(),
        CpuIsa::X86V4Amx
    );

    // The serving runner is refused: `open` fails and installs no table.
    let at_open = select(&spr(AmxPermission::Denied { errno: EPERM }), &NONE).unwrap();
    let err = table_for_class(at_enumerate.isa, &at_open).unwrap_err();
    assert_eq!(err.code, syntropy_kernels::error::codes::AMX_DENIED);
    assert_eq!(err.code, ErrorCode::new(syntropy_core::Domain::Kernels, 13));
    assert_eq!(err.class, ErrorClass::Unsupported);
    // The message names the mismatch: the class, what detection gives, and the errno.
    for part in ["X86V4Amx", "X86V4;", "errno 1", "re-probe"] {
        assert!(err.message.contains(part), "{part:?} in {:?}", err.message);
    }
}

#[test]
fn open_fails_on_any_other_mismatch() {
    let amx_class = CpuIsa::X86V4Amx;
    // The `amx` kill switch at `open` under an `X86V4Amx` class: no request is made, so this
    // is not a refusal, but the provider still never falls back to `X86V4` kernels.
    let killed = select(&spr(AmxPermission::NotRequested), &NO_AMX).unwrap();
    let err = table_for_class(amx_class, &killed).unwrap_err();
    assert_eq!(
        (err.class, err.code),
        (
            ErrorClass::Unsupported,
            syntropy_kernels::error::codes::OP_UNSUPPORTED
        )
    );
    assert!(err.message.contains("X86V4Amx") && err.message.contains("X86V4;"));

    // A class above what the host can run.
    let v3 = select(&spr(AmxPermission::Granted), &NO_AVX512).unwrap();
    assert_eq!(
        table_for_class(CpuIsa::X86V4, &v3).unwrap_err().code,
        syntropy_kernels::error::codes::OP_UNSUPPORTED
    );
    // A class of the other architecture.
    let m4 = select(&mac(Some(1), Some(1), Some(1), Some(1), Some(1)), &NONE).unwrap();
    assert!(table_for_class(CpuIsa::X86V2, &m4).is_err());
    assert!(table_for_class(CpuIsa::Arm80, &v3).is_err());
    // Equal or lower in the same chain is the class's table.
    assert_eq!(
        table_for_class(CpuIsa::Arm86I8mm, &m4).unwrap().isa(),
        CpuIsa::Arm86I8mm
    );
}

// ---- The live host ----

/// The level the standard library's macros alone give on this host, without SME2 or AMX,
/// which the macros on the pinned toolchain cannot report.
#[cfg(target_arch = "x86_64")]
fn macro_level() -> CpuIsa {
    use std::arch::is_x86_feature_detected as d;
    let v3 = d!("avx2") && d!("fma") && d!("f16c");
    let v4 = v3 && d!("avx512f") && d!("avx512bw") && d!("avx512vl") && d!("avx512dq");
    if v4 {
        CpuIsa::X86V4
    } else if v3 {
        CpuIsa::X86V3
    } else {
        CpuIsa::X86V2
    }
}

#[cfg(target_arch = "aarch64")]
fn macro_level() -> CpuIsa {
    use std::arch::is_aarch64_feature_detected as d;
    let dot = d!("dotprod") && d!("fp16");
    if dot && d!("i8mm") && d!("bf16") {
        CpuIsa::Arm86I8mm
    } else if dot {
        CpuIsa::Arm82Dot
    } else {
        CpuIsa::Arm80
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn hosted_runner_selects_its_own_level() {
    let inputs = inputs(&NONE).expect("x86-64 and AArch64 hosts have inputs");
    let d = detect(&NONE).expect("the host is at least X86V2 or Arm80");
    eprintln!("host inputs: {inputs:?}");
    eprintln!("host level: {:?}, features {:?}", d.isa, d.features);
    let expected = macro_level();
    match d.isa {
        // AMX and SME2 sit one level above what the macros can tell; each needs its own
        // evidence.
        CpuIsa::X86V4Amx => {
            assert_eq!(expected, CpuIsa::X86V4);
            assert!(matches!(
                d.amx,
                AmxPermission::Granted | AmxPermission::NotNeeded
            ));
        }
        CpuIsa::Arm92Sme2 => {
            assert_eq!(expected, CpuIsa::Arm86I8mm);
            assert!(d.features.contains(F::SME2));
        }
        isa => assert_eq!(isa, expected),
    }
    // `open` in the same process agrees with `enumerate`.
    let again = detect(&NONE).unwrap();
    assert_eq!(table_for_class(d.isa, &again).unwrap().isa(), d.isa);
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn host_kill_switches_cap_the_level() {
    let all = KillSwitches::from_disable(["avx512", "amx", "sme2"]);
    let d = detect(&all).unwrap();
    eprintln!("host level with every kill switch: {:?}", d.isa);
    assert!(matches!(
        d.isa,
        CpuIsa::X86V2 | CpuIsa::X86V3 | CpuIsa::Arm80 | CpuIsa::Arm82Dot | CpuIsa::Arm86I8mm
    ));
    assert!(
        !d.features
            .intersects(F::AVX512 | F::AMX_INT8 | F::AMX_BF16 | F::SME2)
    );
    assert_eq!(d.table().isa(), d.isa);
    // With AMX killed, no permission request is made.
    assert_eq!(d.amx, AmxPermission::NotRequested);
}
