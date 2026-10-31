// SPDX-License-Identifier: MIT OR Apache-2.0

//! COMP-E13-T02: `kernels::reference`'s dequantisers against ggml's own, bit for bit
//! (RFC-0042 §2.1, §6.4's bit-exact scope), and the GGUF block geometry against ggml's type
//! traits.
//!
//! ggml here is the core's build: the locked commit at the baseline ISA with contraction off
//! (RFC-0042 §4.1 rule 7), the "ggml CPU (core build)" row of RFC-0042 §6.4's bit-exact table.
//! [`syntropy_kernels_ggml::dequantize`] calls ggml's `to_float` type trait, which is its
//! `dequantize_row_*` for the quantised types and its F16 and BF16 row conversions. Every
//! comparison is on the F32 bit patterns, so a NaN's payload and a zero's sign count.

use proptest::prelude::*;
use syntropy_core::GgmlType;
use syntropy_kernels::reference::dequant::{self, block_geometry};
use syntropy_kernels_ggml::{self as ggml, GgmlType as Raw};
use syntropy_manifest::gguf;

/// The five types and ggml's names for them.
const TYPES: [(GgmlType, Raw); 5] = [
    (GgmlType::F16, Raw::F16),
    (GgmlType::BF16, Raw::Bf16),
    (GgmlType::Q8_0, Raw::Q8_0),
    (GgmlType::Q4_0, Raw::Q4_0),
    (GgmlType::Q4_K, Raw::Q4K),
];

/// Asserts the reference and ggml give the same bits for `src`.
fn same_bits(ty: GgmlType, raw: Raw, src: &[u8]) {
    let reference = dequant::dequantize(ty, src).unwrap();
    let ggml = ggml::dequantize(raw, src).unwrap();
    assert_eq!(reference.len(), ggml.len(), "{ty:?}");
    for (i, (r, g)) in reference.iter().zip(&ggml).enumerate() {
        assert_eq!(
            r.to_bits(),
            g.to_bits(),
            "{ty:?} element {i}: reference {r:e} ({:#010x}), ggml {g:e} ({:#010x}), block {:02x?}",
            r.to_bits(),
            g.to_bits(),
            src
        );
    }
}

/// F16 and BF16 scale values at the edges: ±0, the smallest and largest subnormals, the
/// smallest normal, the largest finite, ±infinity, a quiet and a signalling NaN, and one.
const EDGE_HALVES: [u16; 13] = [
    0x0000, 0x8000, 0x0001, 0x03FF, 0x8001, 0x0400, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0x7C01,
    0x3C00,
];

/// Every F16 and every BF16 bit pattern, exhaustively.
#[test]
fn every_f16_and_bf16_value_widens_as_ggml_widens_it() {
    let all: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
    same_bits(GgmlType::F16, Raw::F16, &all);
    same_bits(GgmlType::BF16, Raw::Bf16, &all);
}

/// A block of `ty` with scale bytes `head` and the rest from `fill`.
fn block(ty: GgmlType, head: &[u8], fill: impl Fn(usize) -> u8) -> Vec<u8> {
    let bytes = block_geometry(ty).unwrap().bytes;
    let mut b: Vec<u8> = (0..bytes).map(&fill).collect();
    b[..head.len()].copy_from_slice(head);
    b
}

/// Every edge scale, with codes covering every value: Q8_0's whole signed range, Q4_0's and
/// Q4_K's sixteen nibbles. For Q4_K every pairing of edge `d` and `dmin`, with sub-block scales
/// and minimums of 0, 63 and values between (the k-quant min term `d·sc·q − dmin·m`).
#[test]
fn edge_scales_dequantise_as_ggml_does() {
    for &d in &EDGE_HALVES {
        let h = d.to_le_bytes();
        let codes = |j: usize| (j * 37 + 11) as u8;
        same_bits(GgmlType::Q8_0, Raw::Q8_0, &block(GgmlType::Q8_0, &h, codes));
        same_bits(
            GgmlType::Q8_0,
            Raw::Q8_0,
            &block(GgmlType::Q8_0, &h, |j| (j * 8) as u8),
        );
        same_bits(GgmlType::Q4_0, Raw::Q4_0, &block(GgmlType::Q4_0, &h, codes));
        for &dmin in &EDGE_HALVES {
            let mut head = h.to_vec();
            head.extend(dmin.to_le_bytes());
            for scales in [[0u8; 12], [0xFF; 12], [0x3F; 12], [0x81; 12]] {
                let mut head = head.clone();
                head.extend(scales);
                same_bits(
                    GgmlType::Q4_K,
                    Raw::Q4K,
                    &block(GgmlType::Q4_K, &head, codes),
                );
            }
        }
    }
}

/// Values whose product or difference rounds: two-rounding results, which a fused multiply-add
/// would change, and subnormal products, which a flush to zero would change.
#[test]
fn rounded_and_subnormal_results_match_ggml() {
    // d = 1 + 2^-10 times codes near 255: d·q needs more than 24 bits only after the subtraction.
    let d = 0x3C01u16.to_le_bytes();
    let dmin = 0x3555u16.to_le_bytes(); // ≈ 0.3333
    let mut head = d.to_vec();
    head.extend(dmin);
    head.extend([0x3F; 12]);
    same_bits(
        GgmlType::Q4_K,
        Raw::Q4K,
        &block(GgmlType::Q4_K, &head, |j| j as u8),
    );
    // The smallest subnormal scale times small codes: subnormal F32 results.
    let tiny = 0x0001u16.to_le_bytes();
    same_bits(
        GgmlType::Q8_0,
        Raw::Q8_0,
        &block(GgmlType::Q8_0, &tiny, |j| j as u8),
    );
    let mut head = tiny.to_vec();
    head.extend(tiny);
    head.extend([0x01; 12]);
    same_bits(
        GgmlType::Q4_K,
        Raw::Q4K,
        &block(GgmlType::Q4_K, &head, |j| j as u8),
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// Random blocks of each type, one to four of them, any bytes at all: every scale, every
    /// code, NaN and infinite scales included.
    #[test]
    fn random_blocks_dequantise_as_ggml_does(
        which in 0usize..5,
        bytes in proptest::collection::vec(any::<u8>(), 4 * 144),
        blocks in 1usize..=4,
    ) {
        let (ty, raw) = TYPES[which];
        let size = block_geometry(ty).unwrap().bytes;
        same_bits(ty, raw, &bytes[..blocks * size]);
    }
}

/// RFC-0042 §2.1's table (`manifest`'s, for every type) and the reference's decoded layouts
/// hold ggml's own geometry: `ggml_blck_size` and `ggml_type_size` for each type's ID.
#[test]
fn the_geometry_tables_are_ggml_s() {
    for t in GgmlType::ALL {
        let id = gguf::ggml_type_id(t).unwrap_or_else(|| panic!("{t:?} has no ggml ID"));
        let ggml = ggml::geometry_of_id(id).unwrap_or_else(|| panic!("ggml has no type {id}"));
        let (elems, bytes) = gguf::block_geometry(t).unwrap();
        assert_eq!(
            (
                usize::try_from(elems).unwrap(),
                usize::try_from(bytes).unwrap()
            ),
            ggml,
            "{t:?} (ID {id}): RFC-0042 §2.1's table against ggml's type traits"
        );
        if let Some(g) = block_geometry(t) {
            assert_eq!((g.elems, g.bytes), ggml, "{t:?}: the reference layout");
        }
    }
    for (t, raw) in TYPES {
        let g = block_geometry(t).unwrap();
        assert_eq!(
            (g.elems, g.bytes),
            (raw.block_elems(), raw.block_bytes()),
            "{t:?}"
        );
    }
    assert_eq!(ggml::geometry_of_id(u32::MAX), None);
}
