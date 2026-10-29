// SPDX-License-Identifier: MIT OR Apache-2.0

//! The ggml type mapping (RFC-0041 §2.1 "Types"; RFC-0042 §2.1).
//!
//! A GGUF tensor names its type by ggml's numeric type ID. Each ID Syntropy reads maps to one
//! [`GgmlType`], and so to `QuantScheme::Ggml`. The IDs are ggml's `enum ggml_type` for the
//! pinned ggml version; the block geometry is RFC-0042 §2.1's table. Every other ID is refused
//! with `UnknownQuant` (RFC-0041 §10 code 4), never guessed. That includes IDs ggml defines
//! but Syntropy never reads as weights: the activation-only `Q8_1` and `Q8_K`, the plain
//! integer and F64 types, and the IDs ggml has retired.

use syntropy_core::GgmlType;

/// ggml's type ID and RFC-0042 §2.1's geometry for each [`GgmlType`]:
/// (type, ID, weights per block, bytes per block).
const TABLE: [(GgmlType, u32, u64, u64); 28] = [
    (GgmlType::F32, 0, 1, 4),
    (GgmlType::F16, 1, 1, 2),
    (GgmlType::BF16, 30, 1, 2),
    (GgmlType::Q8_0, 8, 32, 34),
    (GgmlType::Q4_0, 2, 32, 18),
    (GgmlType::Q4_1, 3, 32, 20),
    (GgmlType::Q5_0, 6, 32, 22),
    (GgmlType::Q5_1, 7, 32, 24),
    (GgmlType::Q2_K, 10, 256, 84),
    (GgmlType::Q3_K, 11, 256, 110),
    (GgmlType::Q4_K, 12, 256, 144),
    (GgmlType::Q5_K, 13, 256, 176),
    (GgmlType::Q6_K, 14, 256, 210),
    (GgmlType::IQ4_NL, 20, 32, 18),
    (GgmlType::IQ4_XS, 23, 256, 136),
    (GgmlType::IQ3_XXS, 18, 256, 98),
    (GgmlType::IQ3_S, 21, 256, 110),
    (GgmlType::IQ2_XXS, 16, 256, 66),
    (GgmlType::IQ2_XS, 17, 256, 74),
    (GgmlType::IQ2_S, 22, 256, 82),
    (GgmlType::IQ1_S, 19, 256, 50),
    (GgmlType::IQ1_M, 29, 256, 56),
    (GgmlType::TQ1_0, 34, 256, 54),
    (GgmlType::TQ2_0, 35, 256, 66),
    (GgmlType::Q2_0, 42, 64, 18),
    (GgmlType::Q1_0, 41, 128, 18),
    (GgmlType::MXFP4, 39, 32, 17),
    (GgmlType::NVFP4, 40, 64, 36),
];

/// ggml's names for the type IDs it defines that Syntropy does not read as weights, so a
/// refusal can name the type.
const REFUSED: [(u32, &str); 7] = [
    (9, "Q8_1"),
    (15, "Q8_K"),
    (24, "I8"),
    (25, "I16"),
    (26, "I32"),
    (27, "I64"),
    (28, "F64"),
];

/// The [`GgmlType`] for ggml type ID `id`, or `None` when Syntropy does not read it.
#[must_use]
pub fn ggml_type_from_id(id: u32) -> Option<GgmlType> {
    TABLE.iter().find(|row| row.1 == id).map(|row| row.0)
}

/// ggml's type ID for `t`, or `None` for a variant this table does not know yet.
#[must_use]
pub fn ggml_type_id(t: GgmlType) -> Option<u32> {
    TABLE.iter().find(|row| row.0 == t).map(|row| row.1)
}

/// `t`'s block geometry from RFC-0042 §2.1: (weights per block, bytes per block).
#[must_use]
pub fn block_geometry(t: GgmlType) -> Option<(u64, u64)> {
    TABLE
        .iter()
        .find(|row| row.0 == t)
        .map(|row| (row.2, row.3))
}

/// ggml's name for type ID `id`, when ggml defines it, whether or not Syntropy reads it.
#[must_use]
pub fn ggml_type_name(id: u32) -> Option<String> {
    if let Some(t) = ggml_type_from_id(id) {
        return Some(format!("{t:?}"));
    }
    REFUSED
        .iter()
        .find(|row| row.0 == id)
        .map(|row| row.1.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_ggml_type_has_one_id_and_its_geometry() {
        assert_eq!(TABLE.len(), GgmlType::ALL.len());
        for t in GgmlType::ALL {
            let id = ggml_type_id(t).unwrap_or_else(|| panic!("{t:?} has no type ID"));
            assert_eq!(ggml_type_from_id(id), Some(t));
            let (block, bytes) = block_geometry(t).unwrap();
            assert!(block > 0 && bytes > 0);
        }
        let mut ids: Vec<u32> = TABLE.iter().map(|r| r.1).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), TABLE.len(), "a type ID is used twice");
    }

    #[test]
    fn geometry_matches_rfc_0042_bits_per_weight() {
        // RFC-0042 §2.1's bits-per-weight column, times 10 000 to stay in integers.
        let bpw: [(GgmlType, u64); 28] = [
            (GgmlType::F32, 320_000),
            (GgmlType::F16, 160_000),
            (GgmlType::BF16, 160_000),
            (GgmlType::Q8_0, 85_000),
            (GgmlType::Q4_0, 45_000),
            (GgmlType::Q4_1, 50_000),
            (GgmlType::Q5_0, 55_000),
            (GgmlType::Q5_1, 60_000),
            (GgmlType::Q2_K, 26_250),
            (GgmlType::Q3_K, 34_375),
            (GgmlType::Q4_K, 45_000),
            (GgmlType::Q5_K, 55_000),
            (GgmlType::Q6_K, 65_625),
            (GgmlType::IQ4_NL, 45_000),
            (GgmlType::IQ4_XS, 42_500),
            (GgmlType::IQ3_XXS, 30_625),
            (GgmlType::IQ3_S, 34_375),
            (GgmlType::IQ2_XXS, 20_625),
            (GgmlType::IQ2_XS, 23_125),
            (GgmlType::IQ2_S, 25_625),
            (GgmlType::IQ1_S, 15_625),
            (GgmlType::IQ1_M, 17_500),
            (GgmlType::TQ1_0, 16_875),
            (GgmlType::TQ2_0, 20_625),
            (GgmlType::Q2_0, 22_500),
            (GgmlType::Q1_0, 11_250),
            (GgmlType::MXFP4, 42_500),
            (GgmlType::NVFP4, 45_000),
        ];
        for (t, want) in bpw {
            let (block, bytes) = block_geometry(t).unwrap();
            assert_eq!(bytes * 8 * 10_000 / block, want, "{t:?}");
        }
    }

    #[test]
    fn refused_and_unknown_ids_map_to_nothing() {
        for (id, name) in REFUSED {
            assert_eq!(ggml_type_from_id(id), None);
            assert_eq!(ggml_type_name(id).as_deref(), Some(name));
        }
        // Retired IDs and IDs past the pinned ggml's count.
        for id in [4, 5, 31, 32, 33, 36, 37, 38, 43, 1000, u32::MAX] {
            assert_eq!(ggml_type_from_id(id), None, "{id}");
            assert_eq!(ggml_type_name(id), None, "{id}");
        }
        assert_eq!(ggml_type_name(12).as_deref(), Some("Q4_K"));
    }
}
