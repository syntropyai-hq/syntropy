// SPDX-License-Identifier: MIT OR Apache-2.0

//! The safetensors dtypes and their byte lengths (RFC-0041 §2.2, "dtypes" and "Container").
//!
//! [`SafetensorsDtype`] has one variant per dtype RFC-0041 §2.2 lists, named as
//! `syntropy_core::DType` names the same format (`F8_E4M3` is [`SafetensorsDtype::F8E4M3`],
//! `F4` is [`SafetensorsDtype::F4E2M1`]). Every variant but `BOOL` maps to a [`DType`] through
//! [`SafetensorsDtype::dtype`]. `BOOL` has no `DType`: no kernel takes it, and RFC-0042 §1.1
//! lists no boolean element type, so it stays a container dtype here.
//!
//! **Lengths.** A tensor's data holds its elements packed in order, with no padding: `numel ×
//! bits` bits, as the format's reference reader computes it. That is `numel × bits / 8` bytes
//! for the whole-byte dtypes. For the sub-byte dtypes, `F4` (two elements per byte) and
//! `F6_E2M3`/`F6_E3M2` (four elements per three bytes), the whole tensor is packed, not each
//! row, and a tensor whose bit count is not a whole number of bytes is refused
//! ([`ByteLenError::PartialByte`]). `numel` is the product of the shape, 1 for a scalar; a
//! product of the non-zero dimensions that overflows `u64` is refused even when another
//! dimension is zero ([`ByteLenError::Overflow`]).

use syntropy_core::DType;

/// A tensor dtype a safetensors header may name (RFC-0041 §2.2).
///
/// Variants are declared in RFC-0041 §2.2's order, and [`SafetensorsDtype::ALL`] holds them in
/// that order.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SafetensorsDtype {
    /// `F32`: IEEE 754 binary32.
    F32,
    /// `F16`: IEEE 754 binary16.
    F16,
    /// `BF16`: bfloat16.
    BF16,
    /// `F8_E4M3`: OCP FP8 E4M3, the "FN" variant.
    F8E4M3,
    /// `F8_E5M2`: OCP FP8 E5M2.
    F8E5M2,
    /// `F8_E8M0`: the MX power-of-two scale.
    F8E8M0,
    /// `F4`: FP4 E2M1, two elements per byte.
    F4E2M1,
    /// `F6_E2M3`: FP6 with a 2-bit exponent, four elements per three bytes.
    F6E2M3,
    /// `F6_E3M2`: FP6 with a 3-bit exponent, four elements per three bytes.
    F6E3M2,
    /// `I8`: signed 8-bit integer.
    I8,
    /// `U8`: unsigned 8-bit integer.
    U8,
    /// `I16`: signed 16-bit integer.
    I16,
    /// `I32`: signed 32-bit integer.
    I32,
    /// `I64`: signed 64-bit integer.
    I64,
    /// `BOOL`: one byte per element.
    Bool,
}

impl SafetensorsDtype {
    /// Every variant, in RFC-0041 §2.2's order.
    pub const ALL: [SafetensorsDtype; 15] = [
        SafetensorsDtype::F32,
        SafetensorsDtype::F16,
        SafetensorsDtype::BF16,
        SafetensorsDtype::F8E4M3,
        SafetensorsDtype::F8E5M2,
        SafetensorsDtype::F8E8M0,
        SafetensorsDtype::F4E2M1,
        SafetensorsDtype::F6E2M3,
        SafetensorsDtype::F6E3M2,
        SafetensorsDtype::I8,
        SafetensorsDtype::U8,
        SafetensorsDtype::I16,
        SafetensorsDtype::I32,
        SafetensorsDtype::I64,
        SafetensorsDtype::Bool,
    ];

    /// The dtype a header spells `name`, matched exactly (case matters).
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|d| d.name() == name)
    }

    /// The name as a safetensors header spells it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F8E4M3 => "F8_E4M3",
            Self::F8E5M2 => "F8_E5M2",
            Self::F8E8M0 => "F8_E8M0",
            Self::F4E2M1 => "F4",
            Self::F6E2M3 => "F6_E2M3",
            Self::F6E3M2 => "F6_E3M2",
            Self::I8 => "I8",
            Self::U8 => "U8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::Bool => "BOOL",
        }
    }

    /// The element type, or `None` for `BOOL`, which has none.
    #[must_use]
    pub const fn dtype(self) -> Option<DType> {
        Some(match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::BF16 => DType::BF16,
            Self::F8E4M3 => DType::F8E4M3,
            Self::F8E5M2 => DType::F8E5M2,
            Self::F8E8M0 => DType::F8E8M0,
            Self::F4E2M1 => DType::F4E2M1,
            Self::F6E2M3 => DType::F6E2M3,
            Self::F6E3M2 => DType::F6E3M2,
            Self::I8 => DType::I8,
            Self::U8 => DType::U8,
            Self::I16 => DType::I16,
            Self::I32 => DType::I32,
            Self::I64 => DType::I64,
            Self::Bool => return None,
        })
    }

    /// The storage width of one element in bits: [`DType::bits`] for every variant with a
    /// `DType`, and 8 for `BOOL`.
    #[must_use]
    pub const fn bits(self) -> u32 {
        match self.dtype() {
            Some(d) => d.bits(),
            None => 8,
        }
    }

    /// Whether elements are packed below one byte each (`F4`, `F6_*`).
    #[must_use]
    pub const fn is_sub_byte(self) -> bool {
        self.bits() < 8
    }

    /// The bytes a tensor of `shape` takes, per the rules in the [module](self) documentation.
    ///
    /// # Errors
    ///
    /// [`ByteLenError::Overflow`] if the element or bit count overflows `u64`;
    /// [`ByteLenError::PartialByte`] if the bit count is not a multiple of 8.
    pub fn byte_len(self, shape: &[u64]) -> Result<u64, ByteLenError> {
        // Multiply the non-zero dimensions first, so a zero cannot hide an overflow.
        let nonzero = shape
            .iter()
            .filter(|d| **d != 0)
            .try_fold(1u64, |acc, d| acc.checked_mul(*d))
            .ok_or(ByteLenError::Overflow)?;
        let numel = if shape.contains(&0) { 0 } else { nonzero };
        let bits = numel
            .checked_mul(u64::from(self.bits()))
            .ok_or(ByteLenError::Overflow)?;
        if !bits.is_multiple_of(8) {
            return Err(ByteLenError::PartialByte { bits });
        }
        Ok(bits / 8)
    }
}

/// Why [`SafetensorsDtype::byte_len`] has no length for a shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteLenError {
    /// The element or bit count overflows `u64`.
    Overflow,
    /// The packed elements end inside a byte: `bits` is not a multiple of 8.
    PartialByte {
        /// The tensor's bit count.
        bits: u64,
    },
}

impl std::fmt::Display for ByteLenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Overflow => f.write_str("the byte length overflows"),
            Self::PartialByte { bits } => {
                write!(
                    f,
                    "its {bits} bits of packed elements do not fill whole bytes"
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip_and_are_exact() {
        for d in SafetensorsDtype::ALL {
            assert_eq!(SafetensorsDtype::from_name(d.name()), Some(d));
            assert_eq!(SafetensorsDtype::from_name(&d.name().to_lowercase()), None);
        }
        for other in ["F64", "Q4_0", "F8_E4M3FN", "F4_E2M1", "F6", "bool", ""] {
            assert_eq!(SafetensorsDtype::from_name(other), None, "{other}");
        }
    }

    #[test]
    fn widths_match_core_and_bool_has_no_core_type() {
        let expect = [32, 16, 16, 8, 8, 8, 4, 6, 6, 8, 8, 16, 32, 64, 8];
        for (d, bits) in SafetensorsDtype::ALL.into_iter().zip(expect) {
            assert_eq!(d.bits(), bits, "{d:?}");
            if let Some(core) = d.dtype() {
                assert_eq!(core.bits(), bits, "{d:?}");
                assert_eq!(
                    format!("{core:?}"),
                    format!("{d:?}"),
                    "named as core names it"
                );
            }
        }
        assert_eq!(SafetensorsDtype::Bool.dtype(), None);
    }

    #[test]
    fn whole_byte_lengths_are_numel_times_size() {
        use SafetensorsDtype as D;
        // [3, 5] holds 15 elements; the expected lengths are written out, not computed.
        let expect = [
            (D::F32, 60),
            (D::F16, 30),
            (D::BF16, 30),
            (D::F8E4M3, 15),
            (D::F8E5M2, 15),
            (D::F8E8M0, 15),
            (D::I8, 15),
            (D::U8, 15),
            (D::I16, 30),
            (D::I32, 60),
            (D::I64, 120),
            (D::Bool, 15),
        ];
        for (d, len) in expect {
            assert_eq!(d.byte_len(&[3, 5]), Ok(len), "{d:?}");
        }
        let scalar = [
            (D::F32, 4),
            (D::BF16, 2),
            (D::F8E8M0, 1),
            (D::I64, 8),
            (D::Bool, 1),
        ];
        for (d, len) in scalar {
            assert_eq!(d.byte_len(&[]), Ok(len), "{d:?} scalar");
        }
    }

    #[test]
    fn sub_byte_dtypes_pack_the_whole_tensor() {
        use SafetensorsDtype::{F4E2M1, F6E2M3, F6E3M2};
        // F4: two per byte, across rows.
        assert_eq!(F4E2M1.byte_len(&[4, 8]), Ok(16));
        assert_eq!(F4E2M1.byte_len(&[2, 3]), Ok(3), "rows of 3 share a byte");
        assert_eq!(
            F4E2M1.byte_len(&[3]),
            Err(ByteLenError::PartialByte { bits: 12 })
        );
        assert_eq!(
            F4E2M1.byte_len(&[]),
            Err(ByteLenError::PartialByte { bits: 4 })
        );
        // F6: four per three bytes, across rows.
        for d in [F6E2M3, F6E3M2] {
            assert_eq!(d.byte_len(&[4, 2]), Ok(6), "{d:?}: 48 bits");
            assert_eq!(d.byte_len(&[2, 2]), Ok(3), "{d:?}: 24 bits");
            assert_eq!(d.byte_len(&[32]), Ok(24), "{d:?}");
            assert_eq!(
                d.byte_len(&[2, 3]),
                Err(ByteLenError::PartialByte { bits: 36 }),
                "{d:?}"
            );
            assert_eq!(d.byte_len(&[1]), Err(ByteLenError::PartialByte { bits: 6 }));
        }
    }

    #[test]
    fn zero_dimensions_take_no_bytes_but_still_check_overflow() {
        for d in SafetensorsDtype::ALL {
            assert_eq!(d.byte_len(&[0, 7]), Ok(0), "{d:?}");
            assert_eq!(
                d.byte_len(&[u64::MAX, 0]),
                Ok(0),
                "{d:?}: one big dimension fits"
            );
            assert_eq!(
                d.byte_len(&[0, u64::MAX, 3]),
                Err(ByteLenError::Overflow),
                "{d:?}: the non-zero dimensions overflow"
            );
            assert_eq!(
                d.byte_len(&[u64::MAX, 3]),
                Err(ByteLenError::Overflow),
                "{d:?}"
            );
        }
        assert_eq!(
            SafetensorsDtype::F4E2M1.byte_len(&[u64::MAX]),
            Err(ByteLenError::Overflow)
        );
        assert_eq!(
            SafetensorsDtype::U8.byte_len(&[u64::MAX]),
            Err(ByteLenError::Overflow)
        );
    }
}
