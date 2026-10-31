// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hand-computed cases for `Gemv`, `Gemm`, `GateUpAct`, `Logits` and `RmsNorm`, and property
//! tests for `rows_active = 0`, padded rows and NaN propagation.

use proptest::prelude::*;

use super::*;
use crate::reference::norm::rms_norm;

const SENTINEL: f32 = -12345.0;

fn bits(xs: &[f32]) -> Vec<u32> {
    xs.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn gemv_by_hand() {
    // W = [[1, 2, 3], [4, 5, 6]], x = [[1, 0, -1], [0.5, 0.25, 2]].
    let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w = Matrix::new(&w, 2, 3).unwrap();
    let x = [1.0, 0.0, -1.0, 0.5, 0.25, 2.0];
    let mut y = [SENTINEL; 6];
    gemv(w, &x, 2, &mut y).unwrap();
    // Row 0: 1 - 3 = -2, 4 - 6 = -2. Row 1: 0.5 + 0.5 + 6 = 7, 2 + 1.25 + 12 = 15.25.
    assert_eq!(y, [-2.0, -2.0, 7.0, 15.25, SENTINEL, SENTINEL]);
    let mut z = [SENTINEL; 6];
    gemm(w, &x, 2, &mut z).unwrap();
    assert_eq!(bits(&y), bits(&z));
}

#[test]
fn gemv_uses_the_fixed_order() {
    // One row of 33 inputs: 2^24, then 31 ones in block 0, then 1 in block 1. Block 0 sums to
    // 2^24 (each 1 rounds away), block 1 to 1, and 2^24 + 1 rounds to 2^24 again.
    let mut w = vec![1.0f32; 33];
    w[0] = 16_777_216.0;
    let x = vec![1.0f32; 33];
    let w = Matrix::new(&w, 1, 33).unwrap();
    let mut y = [0.0];
    gemv(w, &x, 1, &mut y).unwrap();
    assert_eq!(y[0], 16_777_216.0);
}

#[test]
fn an_empty_input_dimension_gives_positive_zeros() {
    let w = Matrix::new(&[], 3, 0).unwrap();
    let mut y = [SENTINEL; 3];
    gemv(w, &[], 1, &mut y).unwrap();
    assert_eq!(bits(&y), [0, 0, 0]);
}

#[test]
fn gate_up_act_by_hand() {
    // gate = [[1, 0], [0, 1]], up = [[2, 0], [1, 1]], x = [ln 3, 4].
    let ln3 = 3f32.ln();
    let gate = [1.0, 0.0, 0.0, 1.0];
    let up = [2.0, 0.0, 1.0, 1.0];
    let x = [ln3, 4.0];
    let mut y = [SENTINEL; 2];
    gate_up_act(
        GatedAct::Silu,
        Matrix::new(&gate, 2, 2).unwrap(),
        Matrix::new(&up, 2, 2).unwrap(),
        &x,
        1,
        &mut y,
    )
    .unwrap();
    // Unit 0: silu(ln 3) · 2 ln 3; unit 1: silu(4) · (ln 3 + 4).
    assert_eq!(y[0], math::silu(ln3) * (2.0 * ln3));
    assert_eq!(y[1], math::silu(4.0) * (ln3 + 4.0));
    // silu(ln 3) = 0.75 ln 3, so unit 0 is 1.5 (ln 3)^2 ≈ 1.810_43.
    assert!((y[0] - 1.810_43).abs() < 1e-4);
}

#[test]
fn gate_up_act_refuses_what_m1_defines() {
    let m = Matrix::new(&[1.0], 1, 1).unwrap();
    let mut y = [SENTINEL];
    for act in [GatedAct::Gelu, GatedAct::ClampedSilu] {
        let err = gate_up_act(act, m, m, &[1.0], 1, &mut y).unwrap_err();
        assert!(
            matches!(
                err,
                OpError::Unsupported {
                    op: "GateUpAct",
                    ..
                }
            ),
            "{err}"
        );
    }
    assert_eq!(y, [SENTINEL]);
    let other = Matrix::new(&[1.0, 2.0], 2, 1).unwrap();
    assert_eq!(
        gate_up_act(GatedAct::Silu, m, other, &[1.0], 1, &mut y),
        Err(OpError::ShapeMismatch { op: "GateUpAct" })
    );
}

#[test]
fn rms_norm_by_hand() {
    // x = [3, 4]: ss = 25, mean = 12.5; with eps = 3.5, scale = 1 / sqrt(16) = 0.25.
    let g = [1.0, 2.0];
    let norm = Norm {
        weight: &g,
        eps: 3.5,
    };
    let mut y = [SENTINEL; 4];
    rms_norm(norm, &[3.0, 4.0, 9.0, 9.0], 1, &mut y).unwrap();
    assert_eq!(y, [0.75, 2.0, SENTINEL, SENTINEL]);
}

#[test]
fn rms_norm_edge_rows() {
    let g = [1.0, 1.0];
    let mut y = [SENTINEL; 2];
    rms_norm(
        Norm {
            weight: &g,
            eps: 1e-6,
        },
        &[0.0, 0.0],
        1,
        &mut y,
    )
    .unwrap();
    assert_eq!(y, [0.0, 0.0]);
    rms_norm(
        Norm {
            weight: &g,
            eps: 0.0,
        },
        &[0.0, 0.0],
        1,
        &mut y,
    )
    .unwrap();
    assert!(y.iter().all(|v| v.is_nan()));
    rms_norm(
        Norm {
            weight: &g,
            eps: 1e-6,
        },
        &[f32::INFINITY, 1.0],
        1,
        &mut y,
    )
    .unwrap();
    assert!(y[0].is_nan());
    assert_eq!(y[1], 0.0);
}

#[test]
fn logits_is_the_norm_then_the_head() {
    let g = [1.0, 2.0];
    let norm = Norm {
        weight: &g,
        eps: 3.5,
    };
    // Normalised row [0.75, 2.0]; head [[1, 1], [2, -1], [0, 4]].
    let head = [1.0, 1.0, 2.0, -1.0, 0.0, 4.0];
    let head = Matrix::new(&head, 3, 2).unwrap();
    let mut scratch = [0.0; 2];
    let mut y = [SENTINEL; 6];
    logits(norm, head, &[3.0, 4.0, 1.0, 1.0], 1, &mut scratch, &mut y).unwrap();
    assert_eq!(y, [2.75, -0.5, 8.0, SENTINEL, SENTINEL, SENTINEL]);
    assert_eq!(
        logits(
            Norm {
                weight: &[1.0],
                eps: 0.0
            },
            head,
            &[0.0; 2],
            1,
            &mut scratch,
            &mut y
        ),
        Err(OpError::Len {
            op: "Logits",
            arg: "norm.weight",
            len: 1,
            need: 2
        })
    );
}

#[test]
fn short_buffers_are_refused_before_anything_is_written() {
    let w = Matrix::new(&[1.0; 6], 2, 3).unwrap();
    let mut y = [SENTINEL; 4];
    assert_eq!(
        gemv(w, &[1.0; 5], 2, &mut y),
        Err(OpError::Len {
            op: "Gemv",
            arg: "x",
            len: 5,
            need: 6
        })
    );
    assert_eq!(y, [SENTINEL; 4]);
    assert_eq!(
        gemv(w, &[1.0; 6], 3, &mut y),
        Err(OpError::Len {
            op: "Gemv",
            arg: "x",
            len: 6,
            need: 9
        })
    );
    assert_eq!(
        Matrix::new(&[1.0; 5], 2, 3),
        Err(OpError::MatrixShape {
            len: 5,
            rows: 2,
            cols: 3
        })
    );
}

/// `n` values small enough that no sum overflows.
fn values(n: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-8.0f32..8.0, n)
}

/// One of the ops under test, by index (`Gemv`, `Gemm`, `GateUpAct`, `Logits`), run on `x` into
/// `y`.
fn run(op: usize, w: &[f32], n_out: usize, k_in: usize, x: &[f32], rows: usize, y: &mut [f32]) {
    let m = Matrix::new(w, n_out, k_in).unwrap();
    match op {
        0 => gemv(m, x, rows, y).unwrap(),
        1 => gemm(m, x, rows, y).unwrap(),
        2 => gate_up_act(GatedAct::Silu, m, m, x, rows, y).unwrap(),
        _ => {
            let g = vec![1.5f32; k_in];
            let mut scratch = vec![0.0; k_in];
            logits(
                Norm {
                    weight: &g,
                    eps: 1e-5,
                },
                m,
                x,
                rows,
                &mut scratch,
                y,
            )
            .unwrap();
        }
    }
}

proptest! {
    #[test]
    fn rows_active_zero_writes_nothing(
        op in 0usize..4,
        n_out in 0usize..6,
        k_in in 1usize..40,
        bucket in 0usize..4,
        seed in values(256),
    ) {
        let w: Vec<f32> = seed.iter().cycle().take(n_out * k_in).copied().collect();
        let x: Vec<f32> = seed.iter().rev().cycle().take(bucket * k_in).copied().collect();
        let mut y = vec![SENTINEL; bucket * n_out];
        run(op, &w, n_out, k_in, &x, 0, &mut y);
        prop_assert!(y.iter().all(|v| v.to_bits() == SENTINEL.to_bits()));
    }

    #[test]
    fn padded_rows_are_neither_read_nor_written(
        op in 0usize..4,
        n_out in 1usize..6,
        k_in in 1usize..70,
        active in 0usize..4,
        pad in 1usize..3,
        seed in values(256),
    ) {
        let w: Vec<f32> = seed.iter().cycle().take(n_out * k_in).copied().collect();
        let rows = active + pad;
        let mut x: Vec<f32> = seed.iter().rev().cycle().take(rows * k_in).copied().collect();
        let mut y = vec![SENTINEL; rows * n_out];
        run(op, &w, n_out, k_in, &x, active, &mut y);
        // Padded rows of y keep the sentinel.
        prop_assert!(y[active * n_out..].iter().all(|v| v.to_bits() == SENTINEL.to_bits()));
        // Poisoning the padded rows of x changes nothing.
        x[active * k_in..].fill(f32::NAN);
        let mut z = vec![SENTINEL; rows * n_out];
        run(op, &w, n_out, k_in, &x, active, &mut z);
        prop_assert_eq!(bits(&y), bits(&z));
    }

    #[test]
    fn a_nan_input_poisons_exactly_its_row(
        op in 0usize..3,
        n_out in 1usize..6,
        k_in in 1usize..70,
        rows in 2usize..4,
        at in any::<prop::sample::Index>(),
        seed in values(256),
    ) {
        let w: Vec<f32> = seed.iter().cycle().take(n_out * k_in).copied().collect();
        let mut x: Vec<f32> = seed.iter().rev().cycle().take(rows * k_in).copied().collect();
        let i = at.index(rows * k_in);
        x[i] = f32::NAN;
        let mut y = vec![SENTINEL; rows * n_out];
        run(op, &w, n_out, k_in, &x, rows, &mut y);
        let bad = i / k_in;
        for (r, row) in y.chunks(n_out).enumerate() {
            if r == bad {
                prop_assert!(row.iter().all(|v| v.is_nan()), "row {r}: {row:?}");
            } else {
                prop_assert!(row.iter().all(|v| !v.is_nan()), "row {r}: {row:?}");
            }
        }
    }

    #[test]
    fn a_nan_weight_poisons_exactly_its_column(
        op in 0usize..2,
        n_out in 1usize..6,
        k_in in 1usize..70,
        rows in 1usize..4,
        at in any::<prop::sample::Index>(),
        seed in values(256),
    ) {
        let mut w: Vec<f32> = seed.iter().cycle().take(n_out * k_in).copied().collect();
        let x: Vec<f32> = seed.iter().rev().cycle().take(rows * k_in).copied().collect();
        let i = at.index(n_out * k_in);
        w[i] = f32::NAN;
        let mut y = vec![SENTINEL; rows * n_out];
        run(op, &w, n_out, k_in, &x, rows, &mut y);
        for (j, v) in y.iter().enumerate() {
            prop_assert_eq!(v.is_nan(), j % n_out == i / k_in, "output {}", j);
        }
    }

    #[test]
    fn rms_norm_is_the_documented_formula(
        width in 1usize..80,
        rows in 1usize..3,
        eps in 1e-6f32..1e-2,
        seed in values(256),
    ) {
        let x: Vec<f32> = seed.iter().cycle().take(rows * width).copied().collect();
        let g: Vec<f32> = seed.iter().rev().cycle().take(width).map(|v| v * 0.37).collect();
        let mut y = vec![SENTINEL; rows * width];
        rms_norm(Norm { weight: &g, eps }, &x, rows, &mut y).unwrap();
        for (x, y) in x.chunks(width).zip(y.chunks(width)) {
            let ss = reduce::sum_by(width, |i| x[i] * x[i]);
            let scale = 1.0 / (ss / width as f32 + eps).sqrt();
            for i in 0..width {
                prop_assert_eq!(y[i].to_bits(), ((x[i] * scale) * g[i]).to_bits());
            }
        }
    }

    #[test]
    fn each_output_is_the_fixed_order_dot(
        n_out in 1usize..5,
        k_in in 1usize..100,
        rows in 1usize..4,
        seed in values(512),
    ) {
        let w: Vec<f32> = seed.iter().cycle().take(n_out * k_in).copied().collect();
        let x: Vec<f32> = seed.iter().rev().cycle().take(rows * k_in).copied().collect();
        let mut y = vec![0.0; rows * n_out];
        gemv(Matrix::new(&w, n_out, k_in).unwrap(), &x, rows, &mut y).unwrap();
        for r in 0..rows {
            for o in 0..n_out {
                let want = reduce::dot(&w[o * k_in..(o + 1) * k_in], &x[r * k_in..(r + 1) * k_in]);
                prop_assert_eq!(y[r * n_out + o].to_bits(), want.to_bits());
            }
        }
    }
}
