// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hand-computed cases for `RowGather` and `Convert`. The bit-for-bit comparison of `RowGather`
//! with ggml's `get_rows` is in `kernels-ggml`'s tests, which link ggml.

use proptest::prelude::*;

use super::*;

const SENTINEL: f32 = -12345.0;

fn f32_bytes(xs: &[f32]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn q8_0_block(d: u16, first: i8) -> Vec<u8> {
    let mut b = d.to_le_bytes().to_vec();
    b.extend((0..32).map(|i| first.wrapping_add(i) as u8));
    b
}

#[test]
fn row_gather_copies_f32_rows_bit_for_bit() {
    let nan = f32::from_bits(0x7FA0_0001);
    let table = [1.0, -0.0, nan, 4.0, 5.0, 6.0];
    let mut out = [SENTINEL; 8];
    row_gather(GgmlType::F32, &f32_bytes(&table), 2, &[2, 1, 2], &mut out).unwrap();
    let want = [5.0, 6.0, nan, 4.0, 5.0, 6.0, SENTINEL, SENTINEL];
    let bits = |xs: &[f32]| xs.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&out), bits(&want));
    let mut out = [SENTINEL; 2];
    row_gather(GgmlType::F32, &f32_bytes(&table), 2, &[0], &mut out).unwrap();
    assert_eq!(bits(&out), bits(&[1.0, -0.0]));
}

#[test]
fn row_gather_dequantises_q8_0_rows() {
    // Two rows of one block each; scale 0.5 (F16 0x3800).
    let mut table = q8_0_block(0x3800, -16);
    table.extend(q8_0_block(0x3800, 100));
    let mut out = [SENTINEL; 64];
    row_gather(GgmlType::Q8_0, &table, 32, &[1, 0], &mut out).unwrap();
    for i in 0..32 {
        assert_eq!(out[i], f32::from(100i8.wrapping_add(i as i8)) * 0.5);
        assert_eq!(out[32 + i], f32::from(i as i8 - 16) * 0.5);
    }
}

#[test]
fn row_gather_with_no_ids_writes_nothing() {
    let mut out = [SENTINEL; 2];
    row_gather(GgmlType::F32, &f32_bytes(&[1.0, 2.0]), 2, &[], &mut out).unwrap();
    assert_eq!(out, [SENTINEL; 2]);
    row_gather(GgmlType::F32, &[], 2, &[], &mut []).unwrap();
}

#[test]
fn row_gather_refusals_write_nothing() {
    let table = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let mut out = [SENTINEL; 4];
    assert_eq!(
        row_gather(GgmlType::F32, &table, 2, &[0, 2], &mut out),
        Err(OpError::RowOutOfRange { id: 2, rows: 2 })
    );
    assert_eq!(out, [SENTINEL; 4]);
    assert_eq!(
        row_gather(GgmlType::Q8_0, &q8_0_block(0, 0), 16, &[0], &mut out),
        Err(OpError::RowLength {
            ty: GgmlType::Q8_0,
            elems: 16,
            block: 32
        })
    );
    assert_eq!(
        row_gather(GgmlType::F32, &table, 0, &[0], &mut out),
        Err(OpError::RowLength {
            ty: GgmlType::F32,
            elems: 0,
            block: 1
        })
    );
    assert_eq!(
        row_gather(GgmlType::F32, &table[..12], 2, &[0], &mut out),
        Err(OpError::Len {
            op: "RowGather",
            arg: "table",
            len: 12,
            need: 16
        })
    );
    assert_eq!(
        row_gather(GgmlType::F32, &table, 2, &[0, 1, 0], &mut out),
        Err(OpError::Len {
            op: "RowGather",
            arg: "out",
            len: 4,
            need: 6
        })
    );
    assert!(matches!(
        row_gather(GgmlType::Q5_K, &[0; 176], 256, &[0], &mut [0.0; 256]),
        Err(OpError::Dequant(DequantError::Unsupported(GgmlType::Q5_K)))
    ));
    assert_eq!(out, [SENTINEL; 4]);
}

#[test]
fn convert_by_hand() {
    // F32 → F16: 1, -2, 65504, 65520 (to infinity), 2^-25 (a tie, to +0), 3 · 2^-25 (a tie, up
    // to 2^-23, the even neighbour), NaN.
    let src = f32_bytes(&[
        1.0,
        -2.0,
        65504.0,
        65520.0,
        2f32.powi(-25),
        3.0 * 2f32.powi(-25),
        f32::NAN,
    ]);
    let mut dst = [0u8; 14];
    convert(GgmlType::F32, &src, GgmlType::F16, &mut dst).unwrap();
    let got: Vec<u16> = dst
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect();
    assert_eq!(
        got,
        [0x3C00, 0xC000, 0x7BFF, 0x7C00, 0x0000, 0x0002, 0x7E00]
    );
    // F32 → BF16: 1 + 2^-8 is a tie between 1 and 1 + 2^-7, to even (1); 1 + 3 · 2^-8 rounds up.
    let src = f32_bytes(&[1.0 + 2f32.powi(-8), 1.0 + 3.0 * 2f32.powi(-8)]);
    let mut dst = [0u8; 4];
    convert(GgmlType::F32, &src, GgmlType::BF16, &mut dst).unwrap();
    assert_eq!(dst, [0x80, 0x3F, 0x82, 0x3F]);
    // Q8_0 → F32 is the dequantiser.
    let block = q8_0_block(0x3C00, -3);
    let mut dst = [0u8; 128];
    convert(GgmlType::Q8_0, &block, GgmlType::F32, &mut dst).unwrap();
    assert_eq!(&dst[..4], &(-3.0f32).to_le_bytes());
    assert_eq!(&dst[124..], &28.0f32.to_le_bytes());
}

#[test]
fn convert_refusals() {
    let mut dst = [0u8; 4];
    assert!(matches!(
        convert(GgmlType::F32, &[0; 4], GgmlType::Q8_0, &mut dst),
        Err(OpError::Unsupported { op: "Convert", .. })
    ));
    assert_eq!(
        convert(GgmlType::F32, &[0; 4], GgmlType::F16, &mut dst),
        Err(OpError::Len {
            op: "Convert",
            arg: "dst",
            len: 4,
            need: 2
        })
    );
    assert_eq!(
        convert(GgmlType::F16, &[0; 3], GgmlType::F32, &mut dst),
        Err(OpError::Len {
            op: "Convert",
            arg: "src",
            len: 3,
            need: 4
        })
    );
}

proptest! {
    #[test]
    fn widening_then_narrowing_is_the_identity(halves in prop::collection::vec(any::<u16>(), 0..64)) {
        // Every F16 value widens exactly and narrows back to its bits, except that a NaN comes
        // back as ggml's quiet NaN with its sign.
        let src: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
        let mut wide = vec![0u8; src.len() * 2];
        convert(GgmlType::F16, &src, GgmlType::F32, &mut wide).unwrap();
        let mut back = vec![0u8; src.len()];
        convert(GgmlType::F32, &wide, GgmlType::F16, &mut back).unwrap();
        for (h, b) in halves.iter().zip(back.as_chunks::<2>().0) {
            let b = u16::from_le_bytes(*b);
            if h & 0x7C00 == 0x7C00 && h & 0x03FF != 0 {
                prop_assert_eq!(b, (h & 0x8000) | 0x7E00);
            } else {
                prop_assert_eq!(b, *h);
            }
        }
        // The same for BF16, whose NaNs keep their upper bits and are made quiet.
        let mut wide = vec![0u8; src.len() * 2];
        convert(GgmlType::BF16, &src, GgmlType::F32, &mut wide).unwrap();
        convert(GgmlType::F32, &wide, GgmlType::BF16, &mut back).unwrap();
        for (h, b) in halves.iter().zip(back.as_chunks::<2>().0) {
            let b = u16::from_le_bytes(*b);
            if h & 0x7F80 == 0x7F80 && h & 0x007F != 0 {
                prop_assert_eq!(b, h | 0x0040);
            } else {
                prop_assert_eq!(b, *h);
            }
        }
    }

    #[test]
    fn row_gather_is_the_dequantised_row(
        rows in 1usize..5,
        ids in prop::collection::vec(0u32..5, 0..8),
        firsts in prop::collection::vec(any::<i8>(), 5),
    ) {
        let ids: Vec<u32> = ids.into_iter().map(|i| i % rows as u32).collect();
        let table: Vec<u8> =
            firsts[..rows].iter().flat_map(|&f| q8_0_block(0x3400, f)).collect();
        let whole = dequant::dequantize(GgmlType::Q8_0, &table).unwrap();
        let mut out = vec![SENTINEL; ids.len() * 32];
        row_gather(GgmlType::Q8_0, &table, 32, &ids, &mut out).unwrap();
        for (j, &id) in ids.iter().enumerate() {
            let want = &whole[id as usize * 32..(id as usize + 1) * 32];
            prop_assert_eq!(&out[j * 32..(j + 1) * 32], want);
        }
    }
}
