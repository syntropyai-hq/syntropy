// SPDX-License-Identifier: MIT OR Apache-2.0

//! COMP-E13-T07: the deterministic quantizers fixtures are made with (RFC-0042 §2).
//! [`syntropy_kernels_ggml::quantize`] calls ggml's reference quantizers (`from_float_ref`).
//!
//! The criterion is that the same seed gives byte-identical files on every Tier 1 OS. A seed here
//! makes the F32 input with integer arithmetic only: a SplitMix64 stream, its top 24 bits as a
//! value in [-1, 1), times a power-of-two block scale. Every step is exact, so the input is the
//! same on every platform, with no `libm` call whose last bit could differ. The quantised bytes
//! are hashed with FNV-1a 64 and compared with digests recorded on Linux aarch64. Each platform
//! that runs this file and passes has produced the recorded bytes; the PR that added it lists the
//! platforms that have (while ADR-0626 holds, the macOS and Windows jobs are advisory or not run).
//!
//! The digests cover the bytes. The round-trip tests below catch a wrong type or layout, and
//! are tightest where the block scales are narrow: on the wide-spread input the Q4_K bound is
//! loose for the smallest sub-blocks, which only the digests then cover.

use syntropy_kernels_ggml::{self as ggml, GgmlError, GgmlType, QUANTIZE_MAX_ABS};

/// Elements per case: 64 Q4_K superblocks, 512 Q8_0 and Q4_0 blocks.
const N: usize = 256 * 64;

/// The seeds the digests are recorded for.
const SEEDS: [u64; 2] = [1, 0x5EED_F1C5_7E57_0007];

/// The quantised types and F16 and BF16: every type with a reference quantizer here.
const TYPES: [GgmlType; 5] = [
    GgmlType::Q4K,
    GgmlType::Q8_0,
    GgmlType::Q4_0,
    GgmlType::Bf16,
    GgmlType::F16,
];

/// SplitMix64, the input stream: integer-only, so the same on every platform.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in [-1, 1): the top 24 bits as a signed integer below 2^23, times 2^-23. Exact.
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as i32 - (1 << 23)) as f32 * pow2(-23)
    }
}

/// 2^e as an F32, exactly, for e in the normal range.
fn pow2(e: i32) -> f32 {
    f32::from_bits(((127 + e) as u32) << 23)
}

/// A block whose Q8_0 scale is exactly 1 (its largest magnitude is 127) and whose other values
/// are exact halves, so each one is a rounding tie: `roundf` rounds them away from zero, and a
/// round-to-even routine would not. The block also holds a negative extreme and a zero.
const TIES: [f32; 32] = [
    127.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.5, -3.5, 100.5, -100.5, 126.5, -126.5, 63.5, -63.5,
    31.5, -31.5, 15.5, -15.5, 7.5, -7.5, 0.0, -127.0, 64.5, -64.5, 32.5, -32.5, 16.5, -16.5, 8.5,
    -8.5, 4.5,
];

/// `n` F32 values from `seed`, exact on every platform: [`TIES`] first, then 32-element blocks
/// that each draw a scale of 2^-12 to 2^3, with a few outliers (64 times the block's scale),
/// all-zero blocks and constant blocks, so Q4_K's scale and minimum search sees more than one
/// shape.
fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut rng = SplitMix(seed);
    let mut out = TIES.to_vec();
    while out.len() < n {
        let pick = rng.next();
        let scale = pow2((pick % 16) as i32 - 12);
        let shape = (pick >> 8) % 16;
        for _ in 0..32 {
            let unit = rng.unit();
            let v = match shape {
                0 => 0.0,
                1 => scale,
                2 if rng.next().is_multiple_of(8) => unit * scale * 64.0,
                _ => unit * scale,
            };
            out.push(v);
        }
    }
    out.truncate(n);
    out
}

/// `n` values from `seed` with one scale per 256-element superblock, so each Q4_K superblock's
/// sub-blocks have similar ranges and a per-sub-block bound is meaningful.
fn narrow(seed: u64, n: usize) -> Vec<f32> {
    let mut rng = SplitMix(seed);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let scale = pow2((rng.next() % 8) as i32 - 4);
        for _ in 0..256 {
            out.push(rng.unit() * scale);
        }
    }
    out.truncate(n);
    out
}

/// FNV-1a 64 of `bytes`.
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xCBF2_9CE4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3)
    })
}

/// The digest of each type's quantised bytes for each seed, recorded on Linux aarch64 at the
/// pinned ggml commit ([`ggml::COMMIT`]), in a debug and a release build. A change here means the
/// bytes changed: a new ggml commit, a build flag, or a fault.
const GOLDEN: [(GgmlType, [u64; 2]); 5] = [
    (
        GgmlType::Q4K,
        [0x6618_701a_aa73_59e9, 0x9e84_fc73_2a4f_55de],
    ),
    (
        GgmlType::Q8_0,
        [0x4c10_3906_ff52_cfac, 0x259b_cef0_ea1b_cf85],
    ),
    (
        GgmlType::Q4_0,
        [0x7e81_10b4_c98a_9de1, 0x61e6_2fe7_44ad_866a],
    ),
    (
        GgmlType::Bf16,
        [0xa4ac_eaf1_1d9b_1e15, 0x1e32_d4a9_cbdd_bca1],
    ),
    (
        GgmlType::F16,
        [0x7b60_2116_30c5_0897, 0x8b8f_6283_2a16_dc77],
    ),
];

#[test]
fn a_seed_gives_the_recorded_bytes_for_each_type() {
    let mut wrong = Vec::new();
    for (ty, digests) in GOLDEN {
        for (seed, want) in SEEDS.iter().zip(digests) {
            let bytes = ggml::quantize(ty, &values(*seed, N)).unwrap();
            assert_eq!(
                bytes.len(),
                N / ty.block_elems() * ty.block_bytes(),
                "{ty:?}"
            );
            let got = fnv1a(&bytes);
            if got != want {
                wrong.push(format!(
                    "{ty:?} seed {seed:#x}: got {got:#018x}, recorded {want:#018x}"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "digests differ:\n{}", wrong.join("\n"));
}

#[test]
fn the_same_seed_gives_the_same_bytes_and_another_seed_does_not() {
    for ty in TYPES {
        let a = ggml::quantize(ty, &values(SEEDS[0], N)).unwrap();
        let again = ggml::quantize(ty, &values(SEEDS[0], N)).unwrap();
        let other = ggml::quantize(ty, &values(SEEDS[1], N)).unwrap();
        assert_eq!(a, again, "{ty:?}");
        assert_ne!(a, other, "{ty:?}");
    }
}

/// The rounding error of an F16 scale in its subnormal range, 2^-25, times the most levels a
/// type's integer codes span: where the block scale is that small, its own rounding outgrows one
/// step.
fn scale_floor(ty: GgmlType) -> f32 {
    match ty {
        GgmlType::Q8_0 => 127.0 * pow2(-25),
        GgmlType::Q4_0 => 8.0 * pow2(-25),
        // Q4_K's scales and minimums are 6-bit multiples of two F16 super-scales.
        GgmlType::Q4K => 15.0 * 63.0 * pow2(-25),
        // F16 output: half a subnormal step. BF16 has no scale and F32's exponent range.
        GgmlType::F16 => pow2(-25),
        _ => 0.0,
    }
}

/// Asserts `ys` is `xs` within `step` element by element.
fn near(ty: GgmlType, what: &str, xs: &[f32], ys: &[f32], step: f32) {
    for (j, (a, b)) in xs.iter().zip(ys).enumerate() {
        assert!(
            (a - b).abs() <= step,
            "{ty:?} {what} element {j}: {a} came back as {b}, bound {step}"
        );
    }
}

/// The quantised values come back close to the input through ggml's own dequantiser: within one
/// step of each type's grid, from the block's largest magnitude, plus [`scale_floor`].
#[test]
fn quantised_values_dequantise_near_the_input() {
    let x = values(SEEDS[0], N);
    for ty in TYPES {
        let back = ggml::dequantize(ty, &ggml::quantize(ty, &x).unwrap()).unwrap();
        let block = ty.block_elems().max(32);
        for (i, (xs, ys)) in x.chunks(block).zip(back.chunks(block)).enumerate() {
            let amax = xs.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let step = match ty {
                GgmlType::Q8_0 => amax / 127.0 * 1.01,
                // Q4_0 maps the signed extreme to -8, so the far end of the range is clamped at
                // 7 steps: within amax / 7 still.
                GgmlType::Q4_0 => amax / 7.0 * 1.01,
                // Loose on purpose here: the superblock's scales spread over 2^15, so a
                // sub-block far below the largest can quantise to its minimum. The narrow-spread
                // test bounds Q4_K per sub-block.
                GgmlType::Q4K => 2.0 * amax / 15.0 * 1.05,
                // BF16 keeps 8 significant bits: half a step is 2^-8 of the value.
                GgmlType::Bf16 => amax * pow2(-8),
                // F16 keeps 11: half a step is 2^-11 of the value.
                _ => amax * pow2(-11),
            } + scale_floor(ty);
            near(ty, &format!("block {i}"), xs, ys, step);
        }
    }
}

/// Q4_K on superblocks with one scale each: every 32-element sub-block comes back within one of
/// its own 15 steps, plus the error of the 6-bit sub-block scale and minimum, each a share of
/// 1/63 of the superblock's range.
#[test]
fn q4_k_bounds_each_sub_block_on_narrow_scales() {
    let ty = GgmlType::Q4K;
    let range = |v: &[f32]| {
        let (lo, hi) = v
            .iter()
            .fold((f32::MAX, f32::MIN), |(l, h), &a| (l.min(a), h.max(a)));
        hi - lo.min(0.0)
    };
    for seed in SEEDS {
        let x = narrow(seed, N);
        let back = ggml::dequantize(ty, &ggml::quantize(ty, &x).unwrap()).unwrap();
        for (s, (xs, ys)) in x.chunks(256).zip(back.chunks(256)).enumerate() {
            let super_range = range(xs);
            for (b, (sub, got)) in xs.chunks(32).zip(ys.chunks(32)).enumerate() {
                let step = range(sub) / 15.0 + super_range / 63.0 + scale_floor(ty);
                let what = format!("seed {seed:#x} superblock {s} sub-block {b}");
                near(ty, &what, sub, got, step);
            }
        }
    }
}

#[test]
fn a_partial_block_and_f32_are_refused() {
    let n = GgmlType::Q4K.block_elems();
    assert_eq!(
        ggml::quantize(GgmlType::Q4K, &vec![0.0; n + 1]),
        Err(GgmlError::RowLength {
            elems: n + 1,
            block: n
        })
    );
    assert_eq!(
        ggml::quantize(GgmlType::F32, &[0.0; 32]),
        Err(GgmlError::NoQuantizer(GgmlType::F32))
    );
}

/// A NaN, an infinity, or for every type but BF16 a value past F16's range, is refused with its
/// index, before ggml is called.
#[test]
fn a_value_ggml_cannot_quantise_is_refused() {
    let past = QUANTIZE_MAX_ABS * 2.0;
    for ty in TYPES {
        let len = ty.block_elems().max(32);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut x = vec![1.0; len];
            x[3] = bad;
            assert_eq!(
                ggml::quantize(ty, &x),
                Err(GgmlError::OutOfRange {
                    index: 3,
                    bits: bad.to_bits()
                }),
                "{ty:?} {bad}"
            );
        }
        let mut x = vec![1.0; len];
        x[5] = -past;
        let got = ggml::quantize(ty, &x);
        if ty == GgmlType::Bf16 {
            assert!(got.is_ok(), "{got:?}");
        } else {
            assert_eq!(
                got,
                Err(GgmlError::OutOfRange {
                    index: 5,
                    bits: (-past).to_bits()
                }),
                "{ty:?}"
            );
        }
        x[5] = QUANTIZE_MAX_ABS;
        assert!(ggml::quantize(ty, &x).is_ok(), "{ty:?} at the limit");
    }
}
