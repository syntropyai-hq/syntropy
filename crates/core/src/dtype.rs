// SPDX-License-Identifier: MIT OR Apache-2.0

//! Element data types and GGUF block types.
//!
//! RFC-0001 §5.1 declares [`DType`], and RFC-0042 §1.1 owns it and adds the MX, FP4, F6,
//! index and diagnostic types. [`GgmlType`] lists the GGUF block types of RFC-0042 §2.1 that
//! [`QuantScheme::Ggml`](crate::QuantScheme::Ggml) carries.

use serde::{Deserialize, Serialize};

/// The element type of a tensor, a scale or a kernel argument (RFC-0042 §1.1).
///
/// Variants are declared in the order RFC-0042 §1.1 lists them, and [`DType::ALL`] holds
/// them in that order.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DType {
    /// IEEE 754 binary32.
    F32,
    /// IEEE 754 binary16.
    F16,
    /// bfloat16: binary32 with the low 16 significand bits dropped.
    BF16,
    /// OCP FP8 E4M3, the "FN" variant: no infinities, maximum 448, NaN with every exponent
    /// and mantissa bit set (RFC-0042 §2.3).
    F8E4M3,
    /// OCP FP8 E5M2, IEEE-style, with infinities (RFC-0042 §2.3).
    F8E5M2,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 32-bit integer.
    I32,
    /// MX power-of-two scale, `2^(e − 127)`; `e = 255` is NaN (RFC-0042 §2.2).
    F8E8M0,
    /// FP4 element: sign, 2-bit exponent with bias 1, 1-bit mantissa (RFC-0042 §2.2).
    F4E2M1,
    /// FP6 element with a 2-bit exponent and 3-bit mantissa. Read only (safetensors F6);
    /// packs hold a widened copy (RFC-0042 §3.3).
    F6E2M3,
    /// FP6 element with a 3-bit exponent and 2-bit mantissa. Read only (safetensors F6);
    /// packs hold a widened copy (RFC-0042 §3.3).
    F6E3M2,
    /// Signed 16-bit integer, for index tensors.
    I16,
    /// Signed 64-bit integer, for index tensors.
    I64,
    /// IEEE 754 binary64, for index tensors and the reference executor's diagnostic mode.
    F64,
}

impl DType {
    /// Every variant, in declaration order.
    pub const ALL: [DType; 15] = [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::F8E4M3,
        DType::F8E5M2,
        DType::I8,
        DType::U8,
        DType::I32,
        DType::F8E8M0,
        DType::F4E2M1,
        DType::F6E2M3,
        DType::F6E3M2,
        DType::I16,
        DType::I64,
        DType::F64,
    ];

    /// The storage width of one element in bits.
    pub const fn bits(self) -> u32 {
        match self {
            DType::F4E2M1 => 4,
            DType::F6E2M3 | DType::F6E3M2 => 6,
            DType::F8E4M3 | DType::F8E5M2 | DType::F8E8M0 | DType::I8 | DType::U8 => 8,
            DType::F16 | DType::BF16 | DType::I16 => 16,
            DType::F32 | DType::I32 => 32,
            DType::F64 | DType::I64 => 64,
        }
    }
}

/// A GGUF block type that weights can carry (RFC-0042 §2.1), named as ggml names it.
///
/// The value of each element is ggml's own F32 dequantisation at the pinned ggml version
/// (RFC-0042 §2.1). Block geometry comes from ggml's type traits at build time; the
/// geometry in each variant's description is RFC-0042 §2.1's informative table.
///
/// `Q8_1` and `Q8_K` are activation-only types in ggml and are deliberately absent:
/// Syntropy uses their layouts for activation blocks ([`ActQuant`](crate::ActQuant)), never
/// as weight schemes. A GGUF type ID with no variant here is unsupported (RFC-0041 §2.1).
///
/// Variants are declared in the order of RFC-0042 §2.1's table, and [`GgmlType::ALL`] holds
/// them in that order.
#[non_exhaustive]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GgmlType {
    /// Plain F32: 1 weight in 4 bytes (32 bits per weight).
    F32,
    /// Plain F16: 1 weight in 2 bytes (16 bits per weight).
    F16,
    /// Plain BF16: 1 weight in 2 bytes (16 bits per weight).
    BF16,
    /// 32 weights in 34 bytes (8.5 bits per weight).
    Q8_0,
    /// 32 weights in 18 bytes (4.5 bits per weight).
    Q4_0,
    /// 32 weights in 20 bytes (5.0 bits per weight).
    Q4_1,
    /// 32 weights in 22 bytes (5.5 bits per weight).
    Q5_0,
    /// 32 weights in 24 bytes (6.0 bits per weight).
    Q5_1,
    /// 256 weights in 84 bytes (2.625 bits per weight).
    Q2_K,
    /// 256 weights in 110 bytes (3.4375 bits per weight).
    Q3_K,
    /// 256 weights in 144 bytes (4.5 bits per weight).
    Q4_K,
    /// 256 weights in 176 bytes (5.5 bits per weight).
    Q5_K,
    /// 256 weights in 210 bytes (6.5625 bits per weight).
    Q6_K,
    /// 32 weights in 18 bytes (4.5 bits per weight).
    IQ4_NL,
    /// 256 weights in 136 bytes (4.25 bits per weight).
    IQ4_XS,
    /// 256 weights in 98 bytes (3.0625 bits per weight).
    IQ3_XXS,
    /// 256 weights in 110 bytes (3.4375 bits per weight).
    IQ3_S,
    /// 256 weights in 66 bytes (2.0625 bits per weight).
    IQ2_XXS,
    /// 256 weights in 74 bytes (2.3125 bits per weight).
    IQ2_XS,
    /// 256 weights in 82 bytes (2.5625 bits per weight).
    IQ2_S,
    /// 256 weights in 50 bytes (1.5625 bits per weight).
    IQ1_S,
    /// 256 weights in 56 bytes (1.75 bits per weight).
    IQ1_M,
    /// 256 weights in 54 bytes (1.6875 bits per weight).
    TQ1_0,
    /// 256 weights in 66 bytes (2.0625 bits per weight).
    TQ2_0,
    /// 64 weights in 18 bytes (2.25 bits per weight).
    Q2_0,
    /// 128 weights in 18 bytes (1.125 bits per weight).
    Q1_0,
    /// 32 E2M1 elements and one E8M0 scale in 17 bytes (4.25 bits per weight; RFC-0042 §2.2).
    MXFP4,
    /// 64 weights in 16-weight sub-blocks with E4M3 scales, in 36 bytes (4.5 bits per weight;
    /// RFC-0042 §2.2).
    NVFP4,
}

impl GgmlType {
    /// Every variant, in declaration order.
    pub const ALL: [GgmlType; 28] = [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::BF16,
        GgmlType::Q8_0,
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::Q5_0,
        GgmlType::Q5_1,
        GgmlType::Q2_K,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
        GgmlType::IQ4_NL,
        GgmlType::IQ4_XS,
        GgmlType::IQ3_XXS,
        GgmlType::IQ3_S,
        GgmlType::IQ2_XXS,
        GgmlType::IQ2_XS,
        GgmlType::IQ2_S,
        GgmlType::IQ1_S,
        GgmlType::IQ1_M,
        GgmlType::TQ1_0,
        GgmlType::TQ2_0,
        GgmlType::Q2_0,
        GgmlType::Q1_0,
        GgmlType::MXFP4,
        GgmlType::NVFP4,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Position of each variant in `ALL`. The match has no wildcard, so a new variant fails
    /// to compile here until it is given a position, added to `ALL` and counted.
    fn dtype_position(d: DType) -> usize {
        match d {
            DType::F32 => 0,
            DType::F16 => 1,
            DType::BF16 => 2,
            DType::F8E4M3 => 3,
            DType::F8E5M2 => 4,
            DType::I8 => 5,
            DType::U8 => 6,
            DType::I32 => 7,
            DType::F8E8M0 => 8,
            DType::F4E2M1 => 9,
            DType::F6E2M3 => 10,
            DType::F6E3M2 => 11,
            DType::I16 => 12,
            DType::I64 => 13,
            DType::F64 => 14,
        }
    }
    const DTYPE_VARIANTS: usize = 15;

    fn ggml_position(t: GgmlType) -> usize {
        match t {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::BF16 => 2,
            GgmlType::Q8_0 => 3,
            GgmlType::Q4_0 => 4,
            GgmlType::Q4_1 => 5,
            GgmlType::Q5_0 => 6,
            GgmlType::Q5_1 => 7,
            GgmlType::Q2_K => 8,
            GgmlType::Q3_K => 9,
            GgmlType::Q4_K => 10,
            GgmlType::Q5_K => 11,
            GgmlType::Q6_K => 12,
            GgmlType::IQ4_NL => 13,
            GgmlType::IQ4_XS => 14,
            GgmlType::IQ3_XXS => 15,
            GgmlType::IQ3_S => 16,
            GgmlType::IQ2_XXS => 17,
            GgmlType::IQ2_XS => 18,
            GgmlType::IQ2_S => 19,
            GgmlType::IQ1_S => 20,
            GgmlType::IQ1_M => 21,
            GgmlType::TQ1_0 => 22,
            GgmlType::TQ2_0 => 23,
            GgmlType::Q2_0 => 24,
            GgmlType::Q1_0 => 25,
            GgmlType::MXFP4 => 26,
            GgmlType::NVFP4 => 27,
        }
    }
    const GGML_VARIANTS: usize = 28;

    #[test]
    fn dtype_all_lists_every_variant_once_in_order() {
        assert_eq!(DType::ALL.len(), DTYPE_VARIANTS);
        for (i, d) in DType::ALL.iter().enumerate() {
            assert_eq!(dtype_position(*d), i, "{d:?} is out of place in DType::ALL");
        }
    }

    #[test]
    fn ggml_all_lists_every_variant_once_in_order() {
        assert_eq!(GgmlType::ALL.len(), GGML_VARIANTS);
        for (i, t) in GgmlType::ALL.iter().enumerate() {
            assert_eq!(
                ggml_position(*t),
                i,
                "{t:?} is out of place in GgmlType::ALL"
            );
        }
    }

    #[test]
    fn dtype_bits_match_the_formats() {
        let expected = [
            (DType::F32, 32),
            (DType::F16, 16),
            (DType::BF16, 16),
            (DType::F8E4M3, 8),
            (DType::F8E5M2, 8),
            (DType::I8, 8),
            (DType::U8, 8),
            (DType::I32, 32),
            (DType::F8E8M0, 8),
            (DType::F4E2M1, 4),
            (DType::F6E2M3, 6),
            (DType::F6E3M2, 6),
            (DType::I16, 16),
            (DType::I64, 64),
            (DType::F64, 64),
        ];
        assert_eq!(expected.len(), DTYPE_VARIANTS);
        for (d, bits) in expected {
            assert_eq!(d.bits(), bits, "{d:?}");
        }
    }
}
