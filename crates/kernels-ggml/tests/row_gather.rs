// SPDX-License-Identifier: MIT OR Apache-2.0

//! COMP-E1-T03: `kernels::reference`'s `RowGather` against ggml's CPU `get_rows` at the locked
//! commit, bit for bit (RFC-0040 §3.3; RFC-0042 §6.4's bit-exact scope).
//!
//! ggml here is the core's build: the baseline ISA with contraction off. Every comparison is on
//! the F32 bit patterns, so a NaN's payload and a zero's sign count. Tables are any bytes at
//! all, for `F32`, `F16`, `BF16`, `Q8_0`, `Q4_0` and `Q4_K`: every scale, every code, NaN,
//! infinite and subnormal values included. ggml's own CPU paths differ by architecture (its F16
//! conversion has F16C and NEON forms), so this test is evidence only for the architecture it
//! runs on.

use proptest::prelude::*;
use syntropy_core::GgmlType;
use syntropy_kernels::reference::{self, dequant::block_geometry};
use syntropy_kernels_ggml::{self as ggml, GgmlType as Raw};

/// The six table types and ggml's names for them.
const TYPES: [(GgmlType, Raw); 6] = [
    (GgmlType::F32, Raw::F32),
    (GgmlType::F16, Raw::F16),
    (GgmlType::BF16, Raw::Bf16),
    (GgmlType::Q8_0, Raw::Q8_0),
    (GgmlType::Q4_0, Raw::Q4_0),
    (GgmlType::Q4_K, Raw::Q4K),
];

/// A block's elements and bytes.
fn geometry(ty: GgmlType) -> (usize, usize) {
    if ty == GgmlType::F32 {
        return (1, 4);
    }
    let g = block_geometry(ty).unwrap();
    (g.elems, g.bytes)
}

/// Asserts the reference and ggml gather the same bits.
fn same_bits(ty: GgmlType, raw: Raw, table: &[u8], row_elems: usize, ids: &[u32]) {
    let mut reference = vec![0.0f32; ids.len() * row_elems];
    reference::row_gather(ty, table, row_elems, ids, &mut reference).unwrap();
    let ids_i32: Vec<i32> = ids.iter().map(|&i| i32::try_from(i).unwrap()).collect();
    let ggml = ggml::get_rows(raw, table, row_elems, &ids_i32).unwrap();
    assert_eq!(reference.len(), ggml.len(), "{ty:?}");
    for (i, (r, g)) in reference.iter().zip(&ggml).enumerate() {
        assert_eq!(
            r.to_bits(),
            g.to_bits(),
            "{ty:?} element {i} (row {}): reference {r:e} ({:#010x}), ggml {g:e} ({:#010x})",
            ids[i / row_elems],
            r.to_bits(),
            g.to_bits(),
        );
    }
}

/// Every F16 and every BF16 bit pattern as one table row, gathered in reverse row order: ggml's
/// half conversion against the reference's exact widening, on all 65,536 values.
#[test]
fn every_half_value_is_gathered_as_ggml_gathers_it() {
    let table: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
    // 256 rows of 256 elements.
    let ids: Vec<u32> = (0..256).rev().collect();
    same_bits(GgmlType::F16, Raw::F16, &table, 256, &ids);
    same_bits(GgmlType::BF16, Raw::Bf16, &table, 256, &ids);
}

/// F32 rows are copied: signalling NaNs, payloads and negative zeros keep their bits.
#[test]
fn f32_special_values_are_copied_bit_for_bit() {
    let values: [u32; 8] = [
        0x7F80_0001, // signalling NaN
        0xFFC0_1234, // negative quiet NaN with a payload
        0x8000_0000, // −0
        0x0000_0001, // smallest subnormal
        0x7F80_0000, // +∞
        0xFF7F_FFFF, // most negative finite
        0x3F80_0000, // 1
        0x807F_FFFF, // largest negative subnormal
    ];
    let table: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    same_bits(GgmlType::F32, Raw::F32, &table, 2, &[3, 0, 2, 1, 0]);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// Random tables of each type, one to four blocks per row and one to six rows, any bytes,
    /// gathered by random IDs with repeats.
    #[test]
    fn random_tables_are_gathered_as_ggml_gathers_them(
        which in 0usize..6,
        bytes in proptest::collection::vec(any::<u8>(), 6 * 4 * 144),
        blocks_per_row in 1usize..=4,
        rows in 1usize..=6,
        ids in proptest::collection::vec(0u32..6, 0..12),
    ) {
        let (ty, raw) = TYPES[which];
        let (elems, size) = geometry(ty);
        let row_elems = blocks_per_row * elems;
        let table = &bytes[..rows * blocks_per_row * size];
        let ids: Vec<u32> = ids.into_iter().map(|i| i % rows as u32).collect();
        same_bits(ty, raw, table, row_elems, &ids);
    }
}
