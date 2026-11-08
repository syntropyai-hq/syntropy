// SPDX-License-Identifier: MIT OR Apache-2.0

//! COMP-E13-T07: the deterministic quantizers fixtures are made with (RFC-0042 §2).
//! [`syntropy_kernels_ggml::quantize`] calls ggml's reference quantizers (`from_float_ref`).
//!
//! The criterion is that the same seed gives byte-identical files on every Tier 1 OS. A seed here
//! makes the F32 input with integer arithmetic only: a SplitMix64 stream, its top 24 bits as a
//! value in [-1, 1), times a power-of-two block scale. Every step is exact, so the input is the
//! same on every OS, with no `libm` call whose last bit could differ. The quantised bytes are
//! hashed with FNV-1a 64 and compared with the digests below, which were recorded on Linux
//! aarch64. The hosted test jobs run this file on Linux x86-64, macOS and Windows, so the same
//! digests passing there is the cross-OS check.

use syntropy_kernels_ggml::{self as ggml, GgmlError, GgmlType};

/// Elements per case: 64 Q4_K superblocks, 512 Q8_0 and Q4_0 blocks.
const N: usize = 256 * 64;

/// The seeds the digests are recorded for.
const SEEDS: [u64; 2] = [1, 0x5EED_F1C5_7E57_0007];

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
}

/// `n` F32 values from `seed`, exact on every platform. Each 32-element block draws a scale of
/// 2^-12 to 2^3, and the stream also places a few outliers (64 times the block's scale), all-zero
/// blocks and constant blocks, so Q4_K's scale and minimum search sees more than one shape.
fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut rng = SplitMix(seed);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let pick = rng.next();
        // A power of two in [2^-12, 2^3]: exact as an F32.
        let scale = f32::from_bits(((127 - 12 + (pick % 16) as u32) & 0xFF) << 23);
        let shape = (pick >> 8) % 16;
        for _ in 0..32 {
            let r = rng.next();
            // Top 24 bits as a signed value in [-2^23, 2^23), times 2^-23: exact.
            let unit = ((r >> 40) as i32 - (1 << 23)) as f32 * f32::from_bits(104 << 23);
            let v = match shape {
                0 => 0.0,
                1 => scale,
                2 if r.is_multiple_of(8) => unit * scale * 64.0,
                _ => unit * scale,
            };
            out.push(v);
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
/// pinned ggml commit ([`ggml::COMMIT`]). A change here means the bytes changed: a new ggml
/// commit, a build flag, or a fault.
const GOLDEN: [(GgmlType, [u64; 2]); 5] = [
    (
        GgmlType::Q4K,
        [0xf08b_c378_9088_15c1, 0xfb05_3ee1_430c_d043],
    ),
    (
        GgmlType::Q8_0,
        [0xbe79_73c2_5cc2_0e1e, 0xccb9_a239_cdf7_49ad],
    ),
    (
        GgmlType::Q4_0,
        [0x2c84_7a36_d0ea_3b5d, 0x25a8_fbc4_0a1c_64ba],
    ),
    (
        GgmlType::Bf16,
        [0xcf27_eba4_6e3d_7495, 0x506e_81e0_0d77_bd3c],
    ),
    (
        GgmlType::F16,
        [0x8d0d_fb7a_b741_0e81, 0xe3fd_6fc0_fdd6_cb4e],
    ),
];

#[test]
fn a_seed_gives_the_recorded_bytes_for_each_type() {
    let mut found = Vec::new();
    for (ty, digests) in GOLDEN {
        for (seed, want) in SEEDS.iter().zip(digests) {
            let bytes = ggml::quantize(ty, &values(*seed, N)).unwrap();
            assert_eq!(
                bytes.len(),
                N / ty.block_elems() * ty.block_bytes(),
                "{ty:?}"
            );
            found.push((ty, *seed, fnv1a(&bytes), want));
        }
    }
    let wrong: Vec<_> = found
        .iter()
        .filter(|(_, _, got, want)| got != want)
        .collect();
    assert!(
        wrong.is_empty(),
        "digests differ from the recorded ones (type, seed, got, recorded): {wrong:#x?}"
    );
}

#[test]
fn the_same_seed_gives_the_same_bytes_and_another_seed_does_not() {
    for (ty, _) in GOLDEN {
        let a = ggml::quantize(ty, &values(SEEDS[0], N)).unwrap();
        let again = ggml::quantize(ty, &values(SEEDS[0], N)).unwrap();
        let other = ggml::quantize(ty, &values(SEEDS[1], N)).unwrap();
        assert_eq!(a, again, "{ty:?}");
        assert_ne!(a, other, "{ty:?}");
    }
}

/// The quantised values come back close to the input through ggml's own dequantiser: within one
/// step of each type's grid, with the step bounded from the block's largest magnitude.
#[test]
fn quantised_values_dequantise_near_the_input() {
    let x = values(SEEDS[0], N);
    for (ty, _) in GOLDEN {
        let back = ggml::dequantize(ty, &ggml::quantize(ty, &x).unwrap()).unwrap();
        let block = ty.block_elems().max(32);
        for (i, (xs, ys)) in x.chunks(block).zip(back.chunks(block)).enumerate() {
            let amax = xs.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            // One quantisation step of the block's range, with room for its F16 scales.
            let step = match ty {
                GgmlType::Q8_0 => amax / 127.0 * 1.01,
                GgmlType::Q4_0 => amax / 7.0 * 1.01,
                // Q4_K: 15 levels over each 32-element sub-block's range, through 6-bit scales
                // and minimums; bounded by a step of the superblock's full range.
                GgmlType::Q4K => 2.0 * amax / 15.0 * 1.05,
                // BF16 keeps 8 significant bits, F16 11: a relative step of the largest value.
                GgmlType::Bf16 => amax * f32::EPSILON * 65536.0,
                _ => amax * 2.0f32.powi(-10),
            };
            // An absolute floor for the blocks whose F16 scale is subnormal (block scales down to
            // 2^-12 here), where the scale's own rounding outgrows one step.
            let step = step + f32::from_bits((127 - 16) << 23);
            for (j, (a, b)) in xs.iter().zip(ys).enumerate() {
                assert!(
                    (a - b).abs() <= step,
                    "{ty:?} block {i} element {j}: {a} came back as {b}, step {step}"
                );
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
