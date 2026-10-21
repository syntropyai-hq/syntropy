// SPDX-License-Identifier: MIT OR Apache-2.0

//! Quantisation schemes, dtype classes and the model-class rule.
//!
//! RFC-0001 §5.1 declares [`QuantScheme`]; RFC-0042 §1.1 owns it and defines
//! [`ScaleGranularity`] (including `Group`) and the [`CtScheme`] variants. RFC-0042 §1.2
//! assigns every scheme a [`DTypeClass`] and defines a model's class
//! ([`DTypeClass::of_model`]).

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::dtype::{DType, GgmlType};

/// How finely a scheme's scales vary over a weight tensor (RFC-0042 §1.1, §2.3, §2.4).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ScaleGranularity {
    /// One scale for the whole tensor.
    Tensor,
    /// One scale per output channel.
    Channel,
    /// One scale per `rows` × `cols` tile of the weight. `Block { rows: 128, cols: 128 }` is
    /// the FP8 `weight_scale_inv` convention (RFC-0042 §2.3).
    Block {
        /// Tile height, in rows of the weight.
        rows: u16,
        /// Tile width, in elements along K.
        cols: u16,
    },
    /// One scale per `k` consecutive elements along K.
    Group {
        /// Group length along K.
        k: u16,
    },
}

/// A compressed-tensors scheme (RFC-0042 §1.1, §2.4; RFC-0041 §2.2).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CtScheme {
    /// `pack-quantized` integers: `value = scale_g · (q − z_g)`, in groups of `group`
    /// consecutive elements along K. RFC-0042 §2.4 defines `bits` 4 and 8 only.
    PackedInt {
        /// Bits per integer element.
        bits: u8,
        /// Elements per scale group along K.
        group: u16,
        /// `z_g = 0` for every group.
        symmetric: bool,
        /// The checkpoint stores a zero-point tensor.
        zero_point: bool,
    },
    /// `float-quantized` E4M3 weights with scales at `strategy` granularity.
    Fp8 {
        /// Where the scales vary.
        strategy: ScaleGranularity,
    },
}

/// How a weight tensor's stored elements encode its values (RFC-0042 §1.1, §2).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QuantScheme {
    /// Elements stored unquantised in the given type and used as they are (RFC-0042 §2.5).
    Plain(DType),
    /// A GGUF block type, read unchanged (RFC-0042 §2.1).
    Ggml(GgmlType),
    /// OCP MX FP4: `block` E2M1 elements (32) share one E8M0 scale (RFC-0042 §2.2).
    Mxfp4 {
        /// Elements per scale.
        block: u16,
    },
    /// NVFP4: `block` E2M1 elements (16) share one `scale` (E4M3); the per-tensor F32 global
    /// scale is an auxiliary scalar (RFC-0042 §2.2).
    Nvfp4 {
        /// Elements per block scale.
        block: u16,
        /// The block scale's type.
        scale: DType,
    },
    /// FP8 elements of type `format` (E4M3 or E5M2) with F32 scales at `scale` granularity;
    /// `value = element × scale` (RFC-0042 §2.3).
    Fp8 {
        /// The element type.
        format: DType,
        /// Where the scales vary.
        scale: ScaleGranularity,
    },
    /// A compressed-tensors scheme (RFC-0042 §2.4).
    CompressedTensors(CtScheme),
}

/// A dtype class: the grouping ADR-0019 sets tolerances by (RFC-0042 §1.2).
///
/// Variants are declared from the lowest precision to the highest, and the derived `Ord`
/// follows that order: `Sub4 < Bits4 < Int8 < Fp8 < Float16`. It is RFC-0042 §1.2's order,
/// which follows ADR-0019 decision 3's listing. A comparison such as
/// `class >= DTypeClass::Bits4` therefore reads "`Bits4` or more precise".
///
/// [`QuantScheme::dtype_class`] gives each scheme's class.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DTypeClass {
    /// Below 4 bits per weight: the 1- to 3-bit GGUF types.
    Sub4,
    /// "4-bit integer and FP4 families", extended to 4–6 bits: the 4- to 6-bit GGUF types,
    /// MXFP4, NVFP4 and 4-bit packed integers.
    Bits4,
    /// 8-bit integers: `Q8_0` and 8-bit packed integers.
    Int8,
    /// FP8 elements.
    Fp8,
    /// Unquantised: F32, F16 and BF16.
    Float16,
}

impl DTypeClass {
    /// Every variant, from the lowest precision to the highest.
    pub const ALL: [DTypeClass; 5] = [
        DTypeClass::Sub4,
        DTypeClass::Bits4,
        DTypeClass::Int8,
        DTypeClass::Fp8,
        DTypeClass::Float16,
    ];

    /// A model's dtype class (RFC-0042 §1.2): the class of the lowest-precision scheme that
    /// carries at least 10 % of the model's batch-1 `read_per_token` bytes (RFC-0041 §3.1).
    ///
    /// `read_per_token` yields `(scheme, bytes)` pairs, typically one per tensor group of the
    /// census. Each distinct `QuantScheme` value is its own scheme, block and group sizes
    /// included. Pairs with the same scheme are summed first: the 10 % threshold applies to
    /// each scheme's total, not to each pair, and not to each class. The comparison is exact
    /// integer arithmetic, so a scheme carrying exactly 10 % qualifies.
    ///
    /// # Errors
    ///
    /// - [`ModelClassError::Unclassified`] for the first scheme that has no class
    ///   ([`QuantScheme::dtype_class`] returns `None`), whatever its share.
    /// - [`ModelClassError::NoBytes`] when the pairs carry no bytes in total.
    /// - [`ModelClassError::NoSchemeAtThreshold`] when no scheme reaches 10 %, which needs at
    ///   least eleven distinct schemes. RFC-0042 §1.2 defines no class for that case (its
    ///   open question 8).
    pub fn of_model<I>(read_per_token: I) -> Result<DTypeClass, ModelClassError>
    where
        I: IntoIterator<Item = (QuantScheme, u64)>,
    {
        let mut per_scheme: BTreeMap<QuantScheme, u128> = BTreeMap::new();
        let mut total: u128 = 0;
        for (scheme, bytes) in read_per_token {
            if scheme.dtype_class().is_none() {
                return Err(ModelClassError::Unclassified { scheme });
            }
            // u128 sums of u64 values cannot overflow below 2^64 pairs.
            *per_scheme.entry(scheme).or_insert(0) += u128::from(bytes);
            total += u128::from(bytes);
        }
        if total == 0 {
            return Err(ModelClassError::NoBytes);
        }
        // bytes ≥ total / 10 holds for an integer `bytes` exactly when bytes ≥ ⌈total / 10⌉.
        let min_bytes = total.div_ceil(10);
        per_scheme
            .iter()
            .filter(|&(_, &bytes)| bytes >= min_bytes)
            .filter_map(|(scheme, _)| scheme.dtype_class())
            .min()
            .ok_or(ModelClassError::NoSchemeAtThreshold)
    }
}

impl QuantScheme {
    /// The scheme's dtype class (RFC-0042 §1.2), or `None` for a value RFC-0042 does not
    /// define as a weight scheme.
    ///
    /// RFC-0042 §1.2's table, with the GGUF classes of §2.1, gives:
    ///
    /// | Class | Schemes |
    /// |---|---|
    /// | `Float16` | `Plain(F32 \| F16 \| BF16)`, `Ggml(F32 \| F16 \| BF16)` |
    /// | `Fp8` | `Fp8 { format: F8E4M3 \| F8E5M2, .. }`, `CompressedTensors(Fp8 { .. })` |
    /// | `Int8` | `Ggml(Q8_0)`, `CompressedTensors(PackedInt { bits: 8, .. })` |
    /// | `Bits4` | `Ggml` `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_NL`, `IQ4_XS`, `MXFP4`, `NVFP4`; `Mxfp4 { .. }`, `Nvfp4 { .. }`, `CompressedTensors(PackedInt { bits: 4, .. })` |
    /// | `Sub4` | `Ggml` `Q2_K`, `Q3_K`, `IQ3_XXS`, `IQ3_S`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ1_S`, `IQ1_M`, `TQ1_0`, `TQ2_0`, `Q2_0`, `Q1_0` |
    ///
    /// RFC-0041 §2.2 reads any safetensors tensor without a quantisation config as
    /// `Plain(dtype)`, so §1.2 gives the plain types the table does not list a class by
    /// element format and width: `F64`, `I16`, `I32` and `I64` are `Float16`; `F8E4M3` and
    /// `F8E5M2` are `Fp8`; `I8` and `U8` are `Int8`; `F4E2M1`, `F6E2M3` and `F6E3M2` are
    /// `Bits4`.
    ///
    /// These values have no class (§1.2):
    /// - `Plain(F8E8M0)`: E8M0 is a scale format, bound to its weight as an auxiliary
    ///   extent (RFC-0041 §2.2), so a plain E8M0 tensor is a mis-bound scale.
    /// - `Fp8 { format }` other than `F8E4M3` or `F8E5M2` (§2.3).
    /// - `PackedInt { bits }` other than 4 or 8 (§2.4).
    ///
    /// Block, group and scale-type fields do not change the class.
    pub const fn dtype_class(&self) -> Option<DTypeClass> {
        match *self {
            QuantScheme::Plain(dtype) => plain_class(dtype),
            QuantScheme::Ggml(ty) => Some(ggml_class(ty)),
            QuantScheme::Mxfp4 { .. } | QuantScheme::Nvfp4 { .. } => Some(DTypeClass::Bits4),
            QuantScheme::Fp8 {
                format: DType::F8E4M3 | DType::F8E5M2,
                ..
            } => Some(DTypeClass::Fp8),
            QuantScheme::Fp8 { .. } => None,
            QuantScheme::CompressedTensors(ct) => match ct {
                CtScheme::Fp8 { .. } => Some(DTypeClass::Fp8),
                CtScheme::PackedInt { bits: 4, .. } => Some(DTypeClass::Bits4),
                CtScheme::PackedInt { bits: 8, .. } => Some(DTypeClass::Int8),
                CtScheme::PackedInt { .. } => None,
            },
        }
    }
}

/// The class of an unquantised element type, by element format and width.
const fn plain_class(dtype: DType) -> Option<DTypeClass> {
    match dtype {
        DType::F32
        | DType::F16
        | DType::BF16
        | DType::F64
        | DType::I16
        | DType::I32
        | DType::I64 => Some(DTypeClass::Float16),
        DType::F8E4M3 | DType::F8E5M2 => Some(DTypeClass::Fp8),
        DType::I8 | DType::U8 => Some(DTypeClass::Int8),
        DType::F4E2M1 | DType::F6E2M3 | DType::F6E3M2 => Some(DTypeClass::Bits4),
        DType::F8E8M0 => None,
    }
}

/// The class column of RFC-0042 §2.1's table.
const fn ggml_class(ty: GgmlType) -> DTypeClass {
    match ty {
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => DTypeClass::Float16,
        GgmlType::Q8_0 => DTypeClass::Int8,
        GgmlType::Q4_0
        | GgmlType::Q4_1
        | GgmlType::Q5_0
        | GgmlType::Q5_1
        | GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q6_K
        | GgmlType::IQ4_NL
        | GgmlType::IQ4_XS
        | GgmlType::MXFP4
        | GgmlType::NVFP4 => DTypeClass::Bits4,
        GgmlType::Q2_K
        | GgmlType::Q3_K
        | GgmlType::IQ3_XXS
        | GgmlType::IQ3_S
        | GgmlType::IQ2_XXS
        | GgmlType::IQ2_XS
        | GgmlType::IQ2_S
        | GgmlType::IQ1_S
        | GgmlType::IQ1_M
        | GgmlType::TQ1_0
        | GgmlType::TQ2_0
        | GgmlType::Q2_0
        | GgmlType::Q1_0 => DTypeClass::Sub4,
    }
}

/// Why [`DTypeClass::of_model`] could not give a model a class.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelClassError {
    /// The pairs carry no bytes in total, so no scheme has a share.
    NoBytes,
    /// A scheme has no dtype class.
    Unclassified {
        /// The first such scheme.
        scheme: QuantScheme,
    },
    /// No scheme carries at least 10 % of the bytes; RFC-0042 §1.2 defines no class.
    NoSchemeAtThreshold,
}

impl fmt::Display for ModelClassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelClassError::NoBytes => f.write_str("the model reads no bytes per token"),
            ModelClassError::Unclassified { scheme } => {
                write!(f, "scheme {scheme:?} has no dtype class")
            }
            ModelClassError::NoSchemeAtThreshold => {
                f.write_str("no scheme carries at least 10 % of the bytes read per token")
            }
        }
    }
}

impl std::error::Error for ModelClassError {}

#[cfg(test)]
mod tests {
    use super::*;

    use DTypeClass::{Bits4, Float16, Fp8, Int8, Sub4};

    const GRANULARITIES: [ScaleGranularity; 6] = [
        ScaleGranularity::Tensor,
        ScaleGranularity::Channel,
        ScaleGranularity::Block {
            rows: 128,
            cols: 128,
        },
        ScaleGranularity::Block { rows: 1, cols: 32 },
        ScaleGranularity::Group { k: 32 },
        ScaleGranularity::Group { k: 128 },
    ];

    /// Index of a scheme's variant. No wildcard: a new variant of `QuantScheme` or `CtScheme`
    /// fails to compile here until it is indexed, counted and enumerated in `all_schemes`.
    fn scheme_variant(s: &QuantScheme) -> usize {
        match s {
            QuantScheme::Plain(_) => 0,
            QuantScheme::Ggml(_) => 1,
            QuantScheme::Mxfp4 { .. } => 2,
            QuantScheme::Nvfp4 { .. } => 3,
            QuantScheme::Fp8 { .. } => 4,
            QuantScheme::CompressedTensors(CtScheme::PackedInt { .. }) => 5,
            QuantScheme::CompressedTensors(CtScheme::Fp8 { .. }) => 6,
        }
    }
    const SCHEME_VARIANTS: usize = 7;

    /// As `scheme_variant`, for `ScaleGranularity`.
    fn granularity_variant(g: &ScaleGranularity) -> usize {
        match g {
            ScaleGranularity::Tensor => 0,
            ScaleGranularity::Channel => 1,
            ScaleGranularity::Block { .. } => 2,
            ScaleGranularity::Group { .. } => 3,
        }
    }
    const GRANULARITY_VARIANTS: usize = 4;

    /// Every scheme value the exhaustive test checks: every `DType` and `GgmlType` where a
    /// field holds one, every `u8` packed width, and sample values of the other fields.
    fn all_schemes() -> Vec<QuantScheme> {
        let mut out: Vec<QuantScheme> = Vec::new();
        out.extend(DType::ALL.map(QuantScheme::Plain));
        out.extend(GgmlType::ALL.map(QuantScheme::Ggml));
        out.extend([0, 16, 32, 64].map(|block| QuantScheme::Mxfp4 { block }));
        for scale in DType::ALL {
            out.extend([0, 16, 32].map(|block| QuantScheme::Nvfp4 { block, scale }));
        }
        for format in DType::ALL {
            out.extend(GRANULARITIES.map(|scale| QuantScheme::Fp8 { format, scale }));
        }
        for bits in 0..=u8::MAX {
            for group in [0, 32, 128] {
                for (symmetric, zero_point) in
                    [(false, false), (false, true), (true, false), (true, true)]
                {
                    out.push(QuantScheme::CompressedTensors(CtScheme::PackedInt {
                        bits,
                        group,
                        symmetric,
                        zero_point,
                    }));
                }
            }
        }
        out.extend(
            GRANULARITIES
                .map(|strategy| QuantScheme::CompressedTensors(CtScheme::Fp8 { strategy })),
        );
        out
    }

    /// The rows that contain `s`, written independently of `dtype_class` as one `matches!`
    /// per row. RFC-0042 §1.2's table (with §2.1's GGUF plain types) comes first, then the
    /// plain types §1.2's text classes by element format and width.
    fn rows_containing(s: &QuantScheme) -> Vec<DTypeClass> {
        use CtScheme as C;
        use DType as D;
        use GgmlType as G;
        use QuantScheme as Q;
        let rows: [(DTypeClass, bool); 9] = [
            // RFC-0042 §1.2 and §2.1.
            (
                Float16,
                matches!(s, Q::Plain(D::F32 | D::F16 | D::BF16))
                    || matches!(s, Q::Ggml(G::F32 | G::F16 | G::BF16)),
            ),
            (
                Fp8,
                matches!(
                    s,
                    Q::Fp8 {
                        format: D::F8E4M3 | D::F8E5M2,
                        ..
                    }
                ) || matches!(s, Q::CompressedTensors(C::Fp8 { .. })),
            ),
            (
                Int8,
                matches!(s, Q::Ggml(G::Q8_0))
                    || matches!(s, Q::CompressedTensors(C::PackedInt { bits: 8, .. })),
            ),
            (
                Bits4,
                matches!(
                    s,
                    Q::Ggml(
                        G::Q4_0
                            | G::Q4_1
                            | G::Q5_0
                            | G::Q5_1
                            | G::Q4_K
                            | G::Q5_K
                            | G::Q6_K
                            | G::IQ4_NL
                            | G::IQ4_XS
                            | G::MXFP4
                            | G::NVFP4
                    )
                ) || matches!(s, Q::Mxfp4 { .. } | Q::Nvfp4 { .. })
                    || matches!(s, Q::CompressedTensors(C::PackedInt { bits: 4, .. })),
            ),
            (
                Sub4,
                matches!(
                    s,
                    Q::Ggml(
                        G::Q2_K
                            | G::Q3_K
                            | G::IQ3_XXS
                            | G::IQ3_S
                            | G::IQ2_XXS
                            | G::IQ2_XS
                            | G::IQ2_S
                            | G::IQ1_S
                            | G::IQ1_M
                            | G::TQ1_0
                            | G::TQ2_0
                            | G::Q2_0
                            | G::Q1_0
                    )
                ),
            ),
            // §1.2's text: the plain types the table does not list, except E8M0.
            (
                Float16,
                matches!(s, Q::Plain(D::F64 | D::I16 | D::I32 | D::I64)),
            ),
            (Fp8, matches!(s, Q::Plain(D::F8E4M3 | D::F8E5M2))),
            (Int8, matches!(s, Q::Plain(D::I8 | D::U8))),
            (
                Bits4,
                matches!(s, Q::Plain(D::F4E2M1 | D::F6E2M3 | D::F6E3M2)),
            ),
        ];
        rows.iter()
            .filter(|(_, hit)| *hit)
            .map(|(class, _)| *class)
            .collect()
    }

    /// Values RFC-0042 §1.2 gives no class: plain E8M0, an `Fp8` format that is not FP8, and
    /// a packed integer width other than 4 or 8.
    fn is_undefined(s: &QuantScheme) -> bool {
        match s {
            QuantScheme::Plain(DType::F8E8M0) => true,
            QuantScheme::Fp8 { format, .. } => !matches!(format, DType::F8E4M3 | DType::F8E5M2),
            QuantScheme::CompressedTensors(CtScheme::PackedInt { bits, .. }) => {
                !matches!(bits, 4 | 8)
            }
            _ => false,
        }
    }

    #[test]
    fn every_scheme_maps_to_exactly_one_class() {
        let schemes = all_schemes();
        // The enumeration reaches every variant.
        let mut variants: Vec<usize> = schemes.iter().map(scheme_variant).collect();
        variants.sort_unstable();
        variants.dedup();
        assert_eq!(variants, (0..SCHEME_VARIANTS).collect::<Vec<_>>());
        let mut granularities: Vec<usize> = GRANULARITIES.iter().map(granularity_variant).collect();
        granularities.sort_unstable();
        granularities.dedup();
        assert_eq!(granularities, (0..GRANULARITY_VARIANTS).collect::<Vec<_>>());
        // 15 plain + 28 GGUF + 4 MXFP4 + 45 NVFP4 + 90 FP8 + 3072 packed-int + 6 CT FP8.
        assert_eq!(
            schemes.len(),
            15 + 28 + 4 + 15 * 3 + 15 * 6 + 256 * 3 * 4 + 6
        );
        let mut classified = 0;
        for s in &schemes {
            let rows = rows_containing(s);
            if is_undefined(s) {
                assert!(rows.is_empty(), "{s:?} is undefined but sits in {rows:?}");
                assert_eq!(s.dtype_class(), None, "{s:?} is undefined but has a class");
                continue;
            }
            assert_eq!(
                rows.len(),
                1,
                "{s:?} sits in rows {rows:?}, not exactly one"
            );
            assert_eq!(s.dtype_class(), Some(rows[0]), "{s:?}");
            classified += 1;
        }
        // Every class has at least one scheme.
        for class in DTypeClass::ALL {
            assert!(
                schemes.iter().any(|s| s.dtype_class() == Some(class)),
                "{class:?} has no scheme"
            );
        }
        // The undefined values are exactly plain E8M0, the 13 non-FP8 formats × 6
        // granularities, and the 254 packed widths other than 4 and 8 × 12 field combinations.
        assert_eq!(schemes.len() - classified, 1 + 13 * 6 + 254 * 12);
    }

    #[test]
    fn gguf_classes_match_rfc_0042_table_2_1() {
        // RFC-0042 §2.1, class column, row by row.
        let table: [(&[GgmlType], DTypeClass); 13] = [
            (&[GgmlType::F32, GgmlType::F16, GgmlType::BF16], Float16),
            (&[GgmlType::Q8_0], Int8),
            (&[GgmlType::Q4_0, GgmlType::Q4_1], Bits4),
            (&[GgmlType::Q5_0, GgmlType::Q5_1], Bits4),
            (&[GgmlType::Q2_K, GgmlType::Q3_K], Sub4),
            (&[GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K], Bits4),
            (&[GgmlType::IQ4_NL, GgmlType::IQ4_XS], Bits4),
            (&[GgmlType::IQ3_XXS, GgmlType::IQ3_S], Sub4),
            (
                &[GgmlType::IQ2_XXS, GgmlType::IQ2_XS, GgmlType::IQ2_S],
                Sub4,
            ),
            (&[GgmlType::IQ1_S, GgmlType::IQ1_M], Sub4),
            (&[GgmlType::TQ1_0, GgmlType::TQ2_0], Sub4),
            (&[GgmlType::Q2_0, GgmlType::Q1_0], Sub4),
            (&[GgmlType::MXFP4, GgmlType::NVFP4], Bits4),
        ];
        let mut seen = Vec::new();
        for (types, class) in table {
            for &t in types {
                assert_eq!(QuantScheme::Ggml(t).dtype_class(), Some(class), "{t:?}");
                assert!(!seen.contains(&t), "{t:?} appears twice in the table");
                seen.push(t);
            }
        }
        seen.sort();
        let mut all = GgmlType::ALL.to_vec();
        all.sort();
        assert_eq!(
            seen, all,
            "the table and GgmlType::ALL list different types"
        );
    }

    #[test]
    fn classes_are_ordered_by_precision() {
        // RFC-0042 §1.2 lists the classes from the highest precision to the lowest.
        let rfc_listing = [Float16, Fp8, Int8, Bits4, Sub4];
        assert!(rfc_listing.windows(2).all(|w| w[0] > w[1]));
        assert!(Sub4 < Bits4 && Bits4 < Int8 && Int8 < Fp8 && Fp8 < Float16);
        let mut sorted = DTypeClass::ALL;
        sorted.sort();
        assert_eq!(sorted, DTypeClass::ALL);
    }

    fn q(t: GgmlType) -> QuantScheme {
        QuantScheme::Ggml(t)
    }

    #[test]
    fn threshold_applies_per_scheme_not_per_class() {
        // Two Bits4 schemes at 6 % each make 12 % of Bits4, but neither reaches 10 %.
        let census = [
            (q(GgmlType::Q4_K), 6),
            (q(GgmlType::Q5_K), 6),
            (q(GgmlType::Q8_0), 88),
        ];
        assert_eq!(DTypeClass::of_model(census), Ok(Int8));
    }

    #[test]
    fn pairs_of_one_scheme_are_summed() {
        // Ten 1 % entries of one scheme are one 10 % scheme.
        let mut census = vec![(q(GgmlType::Q8_0), 90)];
        census.extend(std::iter::repeat_n((q(GgmlType::Q2_K), 1), 10));
        assert_eq!(DTypeClass::of_model(census), Ok(Sub4));
    }

    #[test]
    fn exactly_ten_percent_qualifies() {
        let at = [(q(GgmlType::Q4_0), 10), (q(GgmlType::F16), 90)];
        assert_eq!(DTypeClass::of_model(at), Ok(Bits4));
        let below = [(q(GgmlType::Q4_0), 99), (q(GgmlType::F16), 901)];
        assert_eq!(DTypeClass::of_model(below), Ok(Float16));
    }

    #[test]
    fn lowest_precision_qualifying_scheme_wins() {
        let fp8 = QuantScheme::Fp8 {
            format: DType::F8E4M3,
            scale: ScaleGranularity::Block {
                rows: 128,
                cols: 128,
            },
        };
        let census = [(fp8, 50), (q(GgmlType::Q8_0), 50)];
        assert_eq!(DTypeClass::of_model(census), Ok(Int8));
        let census = [(fp8, 50), (q(GgmlType::Q4_K), 50)];
        assert_eq!(DTypeClass::of_model(census), Ok(Bits4));
    }

    #[test]
    fn errors() {
        assert_eq!(
            DTypeClass::of_model(std::iter::empty()),
            Err(ModelClassError::NoBytes)
        );
        assert_eq!(
            DTypeClass::of_model([(q(GgmlType::Q4_K), 0)]),
            Err(ModelClassError::NoBytes)
        );
        let odd = QuantScheme::CompressedTensors(CtScheme::PackedInt {
            bits: 3,
            group: 32,
            symmetric: true,
            zero_point: false,
        });
        assert_eq!(
            DTypeClass::of_model([(q(GgmlType::Q4_K), 1000), (odd, 1)]),
            Err(ModelClassError::Unclassified { scheme: odd })
        );
        // A plain E8M0 tensor is a scale that lost its weight.
        let e8m0 = QuantScheme::Plain(DType::F8E8M0);
        assert_eq!(
            DTypeClass::of_model([(q(GgmlType::BF16), 99), (e8m0, 1)]),
            Err(ModelClassError::Unclassified { scheme: e8m0 })
        );
        // Eleven equal schemes: each carries 1/11 < 10 %.
        let eleven: Vec<_> = GgmlType::ALL[..11].iter().map(|&t| (q(t), 1)).collect();
        assert_eq!(
            DTypeClass::of_model(eleven),
            Err(ModelClassError::NoSchemeAtThreshold)
        );
        // Ten equal schemes: each carries exactly 10 %, the lowest class wins.
        let ten: Vec<_> = GgmlType::ALL[..10].iter().map(|&t| (q(t), 1)).collect();
        assert_eq!(DTypeClass::of_model(ten), Ok(Sub4));
    }

    #[test]
    fn large_totals_do_not_overflow() {
        let census = [(q(GgmlType::Q4_K), u64::MAX), (q(GgmlType::BF16), u64::MAX)];
        assert_eq!(DTypeClass::of_model(census), Ok(Bits4));
    }
}
