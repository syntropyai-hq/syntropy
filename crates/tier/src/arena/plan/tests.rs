// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use syntropy_core::NodeId;

use super::*;

const T: TierId = TierId {
    node: NodeId([0; 32]),
    ordinal: 0,
};

#[test]
fn align_up_rounds_to_the_next_multiple() {
    let a = Bytes(4096);
    assert_eq!(align_up(Bytes(0), a), Ok(Bytes(0)));
    assert_eq!(align_up(Bytes(1), a), Ok(Bytes(4096)));
    assert_eq!(align_up(Bytes(4096), a), Ok(Bytes(4096)));
    assert_eq!(align_up(Bytes(4097), a), Ok(Bytes(8192)));
    assert_eq!(align_up(Bytes(7), Bytes(1)), Ok(Bytes(7)));
    assert_eq!(align_up(Bytes(7), Bytes(0)), Ok(Bytes(7)));
    assert_eq!(align_up(Bytes(u64::MAX), a), Err(ArenaPlanError::Overflow));
}

#[test]
fn uniform_slot_counts_spend_the_remainder_in_layer_order() {
    let layers = [
        (LayerIx(0), Bytes(10)),
        (LayerIx(1), Bytes(20)),
        (LayerIx(2), Bytes(5)),
    ];
    // 2 × 35 = 70; 25 left: layer 0 takes 10, layer 1 cannot take 20 from 15, layer 2 takes 5.
    assert_eq!(slot_counts(T, Bytes(95), None, &layers), Ok(vec![3, 2, 3]));
    assert_eq!(slot_counts(T, Bytes(70), None, &layers), Ok(vec![2, 2, 2]));
    assert_eq!(slot_counts(T, Bytes(34), None, &layers), Ok(vec![1, 1, 0]));
    assert_eq!(slot_counts(T, Bytes(35), None, &layers), Ok(vec![1, 1, 1]));
    // A remainder of exactly one slot buys it.
    assert_eq!(slot_counts(T, Bytes(45), None, &layers), Ok(vec![2, 1, 1]));
    assert_eq!(
        slot_counts(T, Bytes(1), None, &[(LayerIx(0), Bytes(0))]),
        Err(ArenaPlanError::NoExperts(T))
    );
}

#[test]
fn per_layer_slot_counts_divide_each_layers_budget() {
    let layers = [(LayerIx(0), Bytes(10)), (LayerIx(1), Bytes(0))];
    let per = [Bytes(25), Bytes(5)];
    assert_eq!(
        slot_counts(T, Bytes(30), Some(&per), &layers),
        Ok(vec![2, 0])
    );
    assert!(matches!(
        slot_counts(T, Bytes(29), Some(&per), &layers),
        Err(ArenaPlanError::PerLayer { .. })
    ));
}

#[test]
fn checked_arithmetic_reports_overflow() {
    assert_eq!(
        add(Bytes(u64::MAX), Bytes(1)),
        Err(ArenaPlanError::Overflow)
    );
    assert_eq!(mul(Bytes(u64::MAX), 2), Err(ArenaPlanError::Overflow));
    assert_eq!(mul(Bytes(3), 2), Ok(Bytes(6)));
}
