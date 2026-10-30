// SPDX-License-Identifier: MIT OR Apache-2.0

//! The bindings run ggml's CPU dequantisation and `get_rows` on fixed blocks (COMP-E7-T01).
//!
//! Each block is built by hand, so its expected F32 values are known exactly: small integer
//! codes times power-of-two scales, which every conversion order gives the same result for. The
//! bit-exact comparison with `kernels::reference` on general blocks is COMP-E13-T02's (the
//! dequantisers) and COMP-E1-T03's (`RowGather` against `get_rows`).

use syntropy_kernels_ggml::{
    BUILD_FLAGS, COMMIT, GgmlError, GgmlType, commit, dequantize, get_rows, version,
};

/// IEEE half bits: 1.0, -2.0, 0.5, 65504 (the largest finite half), 2^-24 (the smallest
/// subnormal), -0.0.
const F16_BITS: [(u16, f32); 6] = [
    (0x3C00, 1.0),
    (0xC000, -2.0),
    (0x3800, 0.5),
    (0x7BFF, 65504.0),
    (0x0001, 5.960_464_5e-8),
    (0x8000, -0.0),
];

fn le16(v: u16) -> [u8; 2] {
    v.to_le_bytes()
}

#[test]
fn the_build_reports_the_locked_commit() {
    assert_eq!(commit(), COMMIT);
    assert_eq!(version(), "0.27.0");
}

#[test]
fn the_build_used_the_baseline_isa_without_contraction() {
    if cfg!(target_env = "msvc") {
        assert!(BUILD_FLAGS.contains("/fp:precise"), "{BUILD_FLAGS}");
        if cfg!(target_arch = "x86_64") {
            assert!(BUILD_FLAGS.contains("/arch:SSE4.2"), "{BUILD_FLAGS}");
        }
    } else {
        assert!(BUILD_FLAGS.contains("-ffp-contract=off"), "{BUILD_FLAGS}");
        assert!(BUILD_FLAGS.contains("-fvisibility=hidden"), "{BUILD_FLAGS}");
        if cfg!(target_arch = "x86_64") {
            assert!(BUILD_FLAGS.contains("-march=x86-64-v2"), "{BUILD_FLAGS}");
        } else if cfg!(target_arch = "aarch64") {
            assert!(BUILD_FLAGS.contains("-march=armv8-a"), "{BUILD_FLAGS}");
        }
    }
}

#[test]
fn block_sizes_are_ggml_s() {
    for (ty, elems, bytes) in [
        (GgmlType::F32, 1, 4),
        (GgmlType::F16, 1, 2),
        (GgmlType::Bf16, 1, 2),
        (GgmlType::Q4_0, 32, 18),
        (GgmlType::Q8_0, 32, 34),
        (GgmlType::Q4K, 256, 144),
    ] {
        assert_eq!(
            (ty.block_elems(), ty.block_bytes()),
            (elems, bytes),
            "{ty:?}"
        );
    }
}

#[test]
fn f16_and_bf16_dequantise_exactly() {
    let bytes: Vec<u8> = F16_BITS.iter().flat_map(|&(b, _)| le16(b)).collect();
    let got = dequantize(GgmlType::F16, &bytes).unwrap();
    for (g, &(bits, want)) in got.iter().zip(&F16_BITS) {
        assert_eq!(g.to_bits(), want.to_bits(), "half {bits:#06x}");
    }
    // bfloat16 is the top half of an F32.
    let floats = [1.0f32, -3.5, 1.0e-38, f32::MAX, -0.0];
    let bytes: Vec<u8> = floats
        .iter()
        .flat_map(|f| le16((f.to_bits() >> 16) as u16))
        .collect();
    let got = dequantize(GgmlType::Bf16, &bytes).unwrap();
    for (g, f) in got.iter().zip(floats) {
        assert_eq!(g.to_bits(), f.to_bits() & 0xFFFF_0000, "bf16 of {f}");
    }
}

/// A Q8_0 block: scale 0.5 and the codes -128, -127, … , -97.
fn q8_0_block(first: i8) -> Vec<u8> {
    let mut b = le16(0x3800).to_vec();
    b.extend((0..32).map(|i| first.wrapping_add(i) as u8));
    b
}

#[test]
fn q8_0_dequantises_codes_times_scale() {
    let got = dequantize(GgmlType::Q8_0, &q8_0_block(-128)).unwrap();
    let want: Vec<f32> = (0..32).map(|i| f32::from(-128 + i as i16) * 0.5).collect();
    assert_eq!(got, want);
}

/// A Q4_0 block: scale 2.0; element i of the first 16 is the low nibble of byte i, element
/// 16 + i its high nibble, each offset by 8.
#[test]
fn q4_0_dequantises_nibbles_offset_by_eight() {
    let mut b = le16(0x4000).to_vec();
    b.extend((0..16u8).map(|i| (15 - i) << 4 | i));
    let got = dequantize(GgmlType::Q4_0, &b).unwrap();
    let want: Vec<f32> = (0..16)
        .map(|i| f32::from(i as i16 - 8) * 2.0)
        .chain((0..16).map(|i| f32::from(15 - i as i16 - 8) * 2.0))
        .collect();
    assert_eq!(got, want);
}

/// A Q4_K super-block: `d` = 1, `dmin` = 4, every sub-block scale 1 and minimum 0, so each
/// element is its 4-bit code. In each 64-element chunk the low nibbles of 32 bytes give the
/// first 32 elements and the high nibbles the next 32.
#[test]
fn q4_k_dequantises_its_sub_block_layout() {
    let mut b = le16(0x3C00).to_vec(); // d
    b.extend(le16(0x4400)); // dmin, unused with every minimum 0
    // scales[0..4] = 1: sub-blocks 0–3 scale 1; scales[4..8] = 0: their minimums 0;
    // scales[8..12] = 0x01: sub-blocks 4–7 scale 1, minimum 0.
    b.extend([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]);
    b.extend((0..128u32).map(|i| ((i % 16) as u8) | (((15 - i % 16) as u8) << 4)));
    assert_eq!(b.len(), GgmlType::Q4K.block_bytes());
    let got = dequantize(GgmlType::Q4K, &b).unwrap();
    let mut want = Vec::new();
    for chunk in 0..4 {
        let qs = &b[16 + chunk * 32..16 + chunk * 32 + 32];
        want.extend(qs.iter().map(|q| f32::from(q & 0xF)));
        want.extend(qs.iter().map(|q| f32::from(q >> 4)));
    }
    assert_eq!(got, want);
}

#[test]
fn get_rows_picks_f32_rows_in_index_order() {
    let rows = 4;
    let elems = 3;
    let matrix: Vec<f32> = (0..rows * elems).map(|i| i as f32 * 1.5 - 4.0).collect();
    let bytes: Vec<u8> = matrix.iter().flat_map(|f| f.to_le_bytes()).collect();
    let got = get_rows(GgmlType::F32, &bytes, elems, &[2, 0, 2, 3]).unwrap();
    let want: Vec<f32> = [2, 0, 2, 3]
        .iter()
        .flat_map(|&r| matrix[r * elems..(r + 1) * elems].iter().copied())
        .collect();
    assert_eq!(got, want);
}

#[test]
fn get_rows_dequantises_quantised_rows() {
    // Three Q8_0 rows of one block each: scale 0.5 and the codes first, first + 1, …, wrapping
    // as i8. The expected values are computed here, not with ggml's own dequantisation.
    let firsts = [-128i8, -10, 90];
    let matrix: Vec<u8> = firsts.iter().flat_map(|&f| q8_0_block(f)).collect();
    let got = get_rows(GgmlType::Q8_0, &matrix, 32, &[1, 2, 1]).unwrap();
    let want: Vec<f32> = [1usize, 2, 1]
        .iter()
        .flat_map(|&r| (0..32).map(move |i| f32::from(firsts[r].wrapping_add(i)) * 0.5))
        .collect();
    assert_eq!(got, want);
}

#[test]
fn bad_shapes_and_indices_are_refused_before_ggml_runs() {
    assert_eq!(
        dequantize(GgmlType::Q8_0, &[0; 33]),
        Err(GgmlError::Length {
            bytes: 33,
            expected: 34
        })
    );
    assert_eq!(
        dequantize(GgmlType::F32, &[0; 4]),
        Err(GgmlError::NoConversion(GgmlType::F32))
    );
    let matrix = q8_0_block(0);
    assert_eq!(
        get_rows(GgmlType::Q8_0, &matrix, 16, &[0]),
        Err(GgmlError::RowLength {
            elems: 16,
            block: 32
        })
    );
    assert_eq!(
        get_rows(GgmlType::Q8_0, &matrix, 32, &[1]),
        Err(GgmlError::Row { index: 1, rows: 1 })
    );
    assert_eq!(
        get_rows(GgmlType::Q8_0, &matrix, 32, &[-1]),
        Err(GgmlError::Row { index: -1, rows: 1 })
    );
    assert_eq!(get_rows(GgmlType::Q8_0, &matrix, 32, &[]), Ok(Vec::new()));
}
