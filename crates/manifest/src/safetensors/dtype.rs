// SPDX-License-Identifier: MIT OR Apache-2.0

//! The safetensors dtypes and their byte lengths (RFC-0041 §2.2, "dtypes" and "Container").
//!
//! [`SafetensorsDtype`] has one variant per dtype RFC-0041 §2.2 lists, named as
//! `syntropy_core::DType` names the same format (`F8_E4M3` is [`SafetensorsDtype::F8E4M3`],
//! `F4` is [`SafetensorsDtype::F4E2M1`]). Every variant but `BOOL` maps to a [`DType`] through
//! [`SafetensorsDtype::dtype`]. `BOOL` has no `DType`: no kernel takes it, and RFC-0042 §1.1
//! lists no boolean element type, so it stays a container dtype here.
//!
//! **Lengths.** A whole-byte dtype takes `numel × bits / 8` bytes. The sub-byte dtypes, `F4`
//! (two elements per byte) and `F6_E2M3`/`F6_E3M2` (four elements per three bytes), are packed
//! along each row, the innermost dimension, and each row is rounded up to whole bytes
//! (RFC-0041 §2.2, "sub-byte types round up per row"): `rows × ⌈row_len × bits / 8⌉`, where
//! `row_len` is the last dimension and `rows` the product of the others. A scalar is one row of
//! one element. A tensor with a zero dimension takes no bytes.

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

    /// The bytes a tensor of `shape` takes, per the rules in the [module](self) documentation,
    /// or `None` if the count overflows `u64`.
    #[must_use]
    pub fn byte_len(self, shape: &[u64]) -> Option<u64> {
        let bits = u64::from(self.bits());
        if shape.contains(&0) {
            return Some(0);
        }
        if !self.is_sub_byte() {
            let numel = shape.iter().try_fold(1u64, |acc, d| acc.checked_mul(*d))?;
            return numel.checked_mul(bits / 8);
        }
        let (row_len, outer) = match shape.split_last() {
            Some((last, outer)) => (*last, outer),
            None => (1, &[][..]),
        };
        let rows = outer.iter().try_fold(1u64, |acc, d| acc.checked_mul(*d))?;
        let row_bytes = row_len.checked_mul(bits)?.div_ceil(8);
        rows.checked_mul(row_bytes)
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
        let shape = [3, 5];
        // F32 F16 BF16 F8E4M3 F8E5M2 F8E8M0 I8 U8 I16 I32 I64 BOOL.
        let expect = [60, 30, 30, 15, 15, 15, 15, 15, 30, 60, 120, 15];
        let whole = SafetensorsDtype::ALL
            .into_iter()
            .filter(|d| !d.is_sub_byte());
        for (d, len) in whole.zip(expect) {
            assert_eq!(d.byte_len(&shape), Some(len), "{d:?}");
            assert_eq!(d.byte_len(&[]), Some(len / 15), "{d:?} scalar");
        }
    }

    #[test]
    fn sub_byte_rows_round_up() {
        use SafetensorsDtype::{F4E2M1, F6E2M3, F6E3M2};
        // F4: two per byte.
        assert_eq!(F4E2M1.byte_len(&[4, 8]), Some(16));
        assert_eq!(
            F4E2M1.byte_len(&[4, 7]),
            Some(16),
            "each 7-element row takes 4 bytes"
        );
        assert_eq!(F4E2M1.byte_len(&[1]), Some(1));
        assert_eq!(F4E2M1.byte_len(&[]), Some(1), "a scalar is one row of one");
        // F6: four per three bytes.
        for d in [F6E2M3, F6E3M2] {
            assert_eq!(d.byte_len(&[2, 4]), Some(6), "{d:?}");
            assert_eq!(d.byte_len(&[2, 5]), Some(8), "{d:?}: 30 bits round up to 4");
            assert_eq!(
                d.byte_len(&[3, 2, 1]),
                Some(6),
                "{d:?}: 6 bits round up to 1"
            );
            assert_eq!(d.byte_len(&[32]), Some(24), "{d:?}");
        }
    }

    #[test]
    fn zero_dimensions_take_no_bytes_and_overflow_is_none() {
        for d in SafetensorsDtype::ALL {
            assert_eq!(d.byte_len(&[0, u64::MAX]), Some(0), "{d:?}");
            assert_eq!(d.byte_len(&[u64::MAX, 3]), None, "{d:?}");
        }
        assert_eq!(SafetensorsDtype::F4E2M1.byte_len(&[u64::MAX]), None);
        assert_eq!(SafetensorsDtype::U8.byte_len(&[u64::MAX]), Some(u64::MAX));
    }
}
