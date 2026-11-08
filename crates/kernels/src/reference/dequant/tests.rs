// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hand-computed cases for the reference dequantisers. The bit-for-bit comparison with ggml is
//! in `kernels-ggml`'s tests, which link it.

use proptest::prelude::*;

use super::*;

const ONE: [u8; 2] = 0x3C00u16.to_le_bytes();
const HALF_F16: [u8; 2] = 0x3800u16.to_le_bytes();

fn q8_0_block(d: [u8; 2], codes: impl Fn(usize) -> i8) -> Vec<u8> {
    let mut b = d.to_vec();
    b.extend((0..32).map(|j| codes(j) as u8));
    b
}

#[test]
fn q8_0_is_code_times_scale() {
    let block = q8_0_block(HALF_F16, |j| j as i8 - 16);
    let y = dequantize(GgmlType::Q8_0, &block).unwrap();
    let want: Vec<f32> = (0..32).map(|j| (j as f32 - 16.0) * 0.5).collect();
    assert_eq!(y, want);
    // The extremes of the code.
    let block = q8_0_block(ONE, |j| if j == 0 { i8::MIN } else { i8::MAX });
    let y = dequantize(GgmlType::Q8_0, &block).unwrap();
    assert_eq!((y[0], y[1]), (-128.0, 127.0));
}

#[test]
fn q4_0_puts_low_nibbles_first_and_offsets_by_eight() {
    let mut block = ONE.to_vec();
    // Byte j holds element j in its low nibble and element j + 16 in its high one.
    block.extend((0..16u8).map(|j| j | ((15 - j) << 4)));
    let y = dequantize(GgmlType::Q4_0, &block).unwrap();
    for j in 0..16 {
        assert_eq!(y[j], j as f32 - 8.0, "element {j}");
        assert_eq!(y[j + 16], (15 - j) as f32 - 8.0, "element {}", j + 16);
    }
}

/// A `Q4_K` block whose eight sub-blocks have scales `sc[i]` and minimums `m[i]`, all below 64,
/// packed as ggml's `get_scale_min_k4` reads them, and codes `q(element)`.
fn q4_k_block(d: u16, dmin: u16, sc: [u8; 8], m: [u8; 8], q: impl Fn(usize) -> u8) -> Vec<u8> {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = sc[j] | ((sc[j + 4] >> 4) << 6);
        scales[j + 4] = m[j] | ((m[j + 4] >> 4) << 6);
        scales[j + 8] = (sc[j + 4] & 0xF) | ((m[j + 4] & 0xF) << 4);
    }
    let mut b = d.to_le_bytes().to_vec();
    b.extend(dmin.to_le_bytes());
    b.extend(scales);
    // Bytes 32g..32g+32 hold elements 64g..64g+32 low and 64g+32..64g+64 high.
    for g in 0..4 {
        for l in 0..32 {
            b.push((q(64 * g + l) & 0xF) | ((q(64 * g + 32 + l) & 0xF) << 4));
        }
    }
    b
}

#[test]
fn q4_k_is_scaled_code_minus_scaled_minimum() {
    let sc = [1, 2, 3, 4, 33, 47, 60, 63];
    let m = [0, 1, 2, 3, 40, 50, 61, 63];
    let block = q4_k_block(0x3C00, 0x3800, sc, m, |e| (e % 16) as u8);
    let y = dequantize(GgmlType::Q4_K, &block).unwrap();
    assert_eq!(y.len(), 256);
    for (e, &v) in y.iter().enumerate() {
        let sub = e / 32;
        let want = f32::from(sc[sub]) * (e % 16) as f32 - 0.5 * f32::from(m[sub]);
        assert_eq!(v, want, "element {e}");
    }
}

/// The scale and minimum packing of sub-blocks 4–7, which take their top two bits from the
/// first eight bytes.
#[test]
fn q4_k_unpacks_the_high_sub_block_scales() {
    for j in 0..8 {
        let mut sc = [0u8; 8];
        let mut m = [0u8; 8];
        sc[j] = 0b11_0101;
        m[j] = 0b10_1010;
        let block = q4_k_block(0x3C00, 0x3C00, sc, m, |_| 1);
        let scales = &block[4..16];
        assert_eq!(scale_min_k4(j, scales), (sc[j], m[j]), "sub-block {j}");
    }
}

#[test]
fn f16_and_bf16_widen_exactly() {
    let f16 = [0x3C00u16, 0x0001, 0x03FF, 0x7BFF, 0xFC00, 0x8000];
    let bytes: Vec<u8> = f16.iter().flat_map(|h| h.to_le_bytes()).collect();
    let y = dequantize(GgmlType::F16, &bytes).unwrap();
    let want = [
        1.0,
        2f32.powi(-24),
        1023.0 * 2f32.powi(-24),
        65504.0,
        f32::NEG_INFINITY,
        -0.0,
    ];
    for (got, want) in y.iter().zip(want) {
        assert_eq!(got.to_bits(), want.to_bits());
    }
    let bf16 = [0x3F80u16, 0x0001, 0x7F80, 0xFF7F];
    let bytes: Vec<u8> = bf16.iter().flat_map(|h| h.to_le_bytes()).collect();
    let y = dequantize(GgmlType::BF16, &bytes).unwrap();
    for (got, h) in y.iter().zip(bf16) {
        assert_eq!(got.to_bits(), u32::from(h) << 16);
    }
}

#[test]
fn a_nan_scale_gives_nan_and_a_zero_scale_gives_signed_zeros() {
    let nan = q8_0_block(0x7E00u16.to_le_bytes(), |_| 3);
    assert!(
        dequantize(GgmlType::Q8_0, &nan)
            .unwrap()
            .iter()
            .all(|v| v.is_nan())
    );
    let zero = q8_0_block([0, 0], |j| j as i8 - 16);
    let y = dequantize(GgmlType::Q8_0, &zero).unwrap();
    assert!(y.iter().all(|v| *v == 0.0));
    assert!(y[0].is_sign_negative(), "-16 · +0 is −0");
}

#[test]
fn partial_blocks_and_other_types_are_refused() {
    assert_eq!(
        dequantize(GgmlType::Q4_K, &[0; 143]),
        Err(DequantError::Length {
            ty: GgmlType::Q4_K,
            bytes: 143
        })
    );
    assert_eq!(
        dequantize(GgmlType::F16, &[0; 3]),
        Err(DequantError::Length {
            ty: GgmlType::F16,
            bytes: 3
        })
    );
    for ty in [
        GgmlType::F32,
        GgmlType::Q6_K,
        GgmlType::MXFP4,
        GgmlType::IQ4_NL,
    ] {
        assert_eq!(dequantize(ty, &[]), Err(DequantError::Unsupported(ty)));
    }
    assert_eq!(dequantize(GgmlType::Q8_0, &[]), Ok(Vec::new()));
}

/// The five decoded layouts' geometry, and no other type's.
#[test]
fn the_decoded_layouts_have_rfc_0042_s_geometry() {
    let g = |t| block_geometry(t).map(|g| (g.elems, g.bytes));
    assert_eq!(g(GgmlType::F16), Some((1, 2)));
    assert_eq!(g(GgmlType::BF16), Some((1, 2)));
    assert_eq!(g(GgmlType::Q8_0), Some((32, 34)));
    assert_eq!(g(GgmlType::Q4_0), Some((32, 18)));
    assert_eq!(g(GgmlType::Q4_K), Some((256, 144)));
    for t in [
        GgmlType::F32,
        GgmlType::Q4_1,
        GgmlType::Q6_K,
        GgmlType::MXFP4,
    ] {
        assert_eq!(g(t), None, "{t:?}");
    }
}

proptest! {
    /// Any whole number of blocks dequantises to that many blocks' elements, and an F16 or BF16
    /// row is its elements widened one by one.
    #[test]
    fn whole_blocks_give_their_elements(blocks in 0usize..4, seed in any::<u64>(), ty in 0usize..5) {
        let ty = [GgmlType::F16, GgmlType::BF16, GgmlType::Q8_0, GgmlType::Q4_0, GgmlType::Q4_K][ty];
        let g = block_geometry(ty).unwrap();
        let mut state = seed;
        let bytes: Vec<u8> = (0..blocks * g.bytes)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 56) as u8
            })
            .collect();
        let y = dequantize(ty, &bytes).unwrap();
        prop_assert_eq!(y.len(), blocks * g.elems);
        if ty == GgmlType::F16 || ty == GgmlType::BF16 {
            for (v, b) in y.iter().zip(bytes.as_chunks::<2>().0) {
                let h = u16::from_le_bytes(*b);
                let want = if ty == GgmlType::F16 { f16_to_f32(h) } else { bf16_to_f32(h) };
                prop_assert_eq!(v.to_bits(), want.to_bits());
            }
        }
    }
}
