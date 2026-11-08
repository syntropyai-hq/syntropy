// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hand-computed cases for `Add` and `Mul`, and property tests for `rows_active = 0`, padded rows
//! and NaN propagation.

use proptest::prelude::*;

use super::*;

const SENTINEL: f32 = -12345.0;

#[test]
fn add_and_mul_by_hand() {
    let a = [1.0, 2.0, 3.0, 4.0, 99.0, 99.0];
    let mut y = [SENTINEL; 6];
    add(&a, Operand::Rows(&[0.5, -2.0, 1.0, 1.0]), 2, 2, &mut y).unwrap();
    assert_eq!(y, [1.5, 0.0, 4.0, 5.0, SENTINEL, SENTINEL]);
    mul(&a, Operand::Row(&[10.0, -1.0]), 2, 3, &mut y).unwrap();
    assert_eq!(y, [10.0, -2.0, 30.0, -4.0, 990.0, -99.0]);
}

#[test]
fn ieee_special_values() {
    let mut y = [SENTINEL; 4];
    add(
        &[f32::INFINITY, 0.0, -0.0, f32::NAN],
        Operand::Rows(&[f32::NEG_INFINITY, -0.0, -0.0, 1.0]),
        4,
        1,
        &mut y,
    )
    .unwrap();
    assert!(y[0].is_nan());
    assert_eq!(y[1].to_bits(), 0);
    assert_eq!(y[2].to_bits(), 0x8000_0000);
    assert!(y[3].is_nan());
    mul(
        &[0.0, f32::INFINITY, -2.0, 1.0],
        Operand::Rows(&[f32::INFINITY, -1.0, 0.0, f32::NAN]),
        4,
        1,
        &mut y,
    )
    .unwrap();
    assert!(y[0].is_nan());
    assert_eq!(y[1], f32::NEG_INFINITY);
    assert_eq!(y[2].to_bits(), 0x8000_0000);
    assert!(y[3].is_nan());
}

#[test]
fn a_broadcast_row_must_be_one_row() {
    let mut y = [SENTINEL; 4];
    assert_eq!(
        add(&[1.0; 4], Operand::Row(&[1.0; 4]), 2, 2, &mut y),
        Err(OpError::Len {
            op: "Add",
            arg: "b",
            len: 4,
            need: 2
        })
    );
    assert_eq!(y, [SENTINEL; 4]);
}

proptest! {
    #[test]
    fn only_active_rows_are_written(
        width in 0usize..40,
        active in 0usize..4,
        pad in 0usize..3,
        broadcast in any::<bool>(),
        is_mul in any::<bool>(),
        poison in any::<prop::sample::Index>(),
    ) {
        let rows = active + pad;
        let mut a: Vec<f32> = (0..rows * width).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..rows.max(1) * width).map(|i| 3.0 - i as f32).collect();
        // A NaN in a padded row of `a` must not reach anything.
        if pad > 0 && width > 0 {
            let i = active * width + poison.index(pad * width);
            a[i] = f32::NAN;
        }
        let operand = if broadcast { Operand::Row(&b[..width]) } else { Operand::Rows(&b) };
        let f = if is_mul { mul } else { add };
        let mut y = vec![SENTINEL; rows * width];
        f(&a, operand, width, active, &mut y).unwrap();
        for (i, v) in y.iter().enumerate() {
            if i < active * width {
                let bv = if broadcast { b[i % width] } else { b[i] };
                let want = if is_mul { a[i] * bv } else { a[i] + bv };
                prop_assert_eq!(v.to_bits(), want.to_bits());
            } else {
                prop_assert_eq!(v.to_bits(), SENTINEL.to_bits());
            }
        }
    }

    #[test]
    fn a_nan_propagates_to_its_element_only(
        width in 1usize..40,
        rows in 1usize..4,
        at in any::<prop::sample::Index>(),
    ) {
        let n = rows * width;
        let mut a = vec![1.0f32; n];
        let i = at.index(n);
        a[i] = f32::NAN;
        let mut y = vec![SENTINEL; n];
        add(&a, Operand::Rows(&vec![2.0; n]), width, rows, &mut y).unwrap();
        for (j, v) in y.iter().enumerate() {
            prop_assert_eq!(v.is_nan(), j == i);
        }
    }
}
