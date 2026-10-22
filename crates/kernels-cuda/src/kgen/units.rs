// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CUDA translation units: one per (op family, quant-scheme group) (RFC-0040 §10.2 rule 1).
//!
//! A kernel change recompiles the units that change, times the targets, never the whole
//! provider. Each unit is `csrc/<name>.cu`, and its explicit instantiation list is
//! `csrc/instantiate/<name>.inc` (rule 2). A unit holding bit-exact-class kernels is compiled
//! with contraction off (rule 7; RFC-0042 §4.1 rule 7, §6.4).

use syntropy_core::{DType, GgmlType, QuantScheme};
use syntropy_kernels::OpKind;

/// The ops one unit's kernels implement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Family {
    /// `Gemv`, `GateUpAct` and `Logits`: the quantized GEMV kernels (RFC-0040 §7.1).
    Gemv,
    /// `Gemm`: prefill GEMM.
    Gemm,
    /// `MoeGrouped`: the residency-aware grouped expert FFN (RFC-0040 §3.4).
    MoeGrouped,
    /// `MoeRoute` and `MoeCombine`.
    MoeRoute,
    /// `AttnDecode`, `AttnPartial` and `AttnMerge`: split-KV decode attention.
    AttnDecode,
    /// `AttnPrefill`.
    AttnPrefill,
    /// `KvWrite`.
    KvWrite,
    /// `RmsNorm` and `LayerNorm`.
    Norm,
    /// `Rope`, `Act`, `Add`, `Mul` and `Scale`.
    Elementwise,
    /// `LogitsProcess`, `SampleSeeded` and `VerifyAccept`.
    SampleTail,
    /// `Argmax` and `TopKTopP`: index selection, bit-exact on identical scores.
    SampleSelect,
    /// `RowGather`, `GatherRows` and `ScatterRows`: row copies.
    Rows,
    /// `Convert`: dtype conversion, including standalone dequantisation.
    Convert,
    /// `KvPack` and `KvUnpack`: KV page moves between tiers.
    KvPages,
}

impl Family {
    /// The family whose units implement `op`, or `None` when no CUDA unit implements it yet.
    #[must_use]
    pub fn of(op: &OpKind) -> Option<Family> {
        Some(match op {
            OpKind::Gemv | OpKind::GateUpAct { .. } | OpKind::Logits => Family::Gemv,
            OpKind::Gemm => Family::Gemm,
            OpKind::MoeGrouped => Family::MoeGrouped,
            OpKind::MoeRoute { .. } | OpKind::MoeCombine => Family::MoeRoute,
            OpKind::AttnDecode(_) | OpKind::AttnPartial(_) | OpKind::AttnMerge => {
                Family::AttnDecode
            }
            OpKind::AttnPrefill(_) => Family::AttnPrefill,
            OpKind::KvWrite(_) => Family::KvWrite,
            OpKind::RmsNorm | OpKind::LayerNorm => Family::Norm,
            OpKind::Rope(_) | OpKind::Act(_) | OpKind::Add | OpKind::Mul | OpKind::Scale => {
                Family::Elementwise
            }
            OpKind::LogitsProcess | OpKind::SampleSeeded | OpKind::VerifyAccept => {
                Family::SampleTail
            }
            OpKind::Argmax | OpKind::TopKTopP => Family::SampleSelect,
            OpKind::RowGather | OpKind::GatherRows | OpKind::ScatterRows => Family::Rows,
            OpKind::Convert => Family::Convert,
            OpKind::KvPack | OpKind::KvUnpack => Family::KvPages,
            _ => return None,
        })
    }
}

/// A group of weight schemes one unit's kernels share code for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SchemeGroup {
    /// The unit's kernels do not depend on a weight scheme, or take any.
    Any,
    /// Unquantised F32, F16 and BF16.
    Float,
    /// The legacy GGUF block types: `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`.
    Legacy,
    /// The GGUF k-quants: `Q2_K` to `Q6_K`.
    KQuant,
    /// The other GGUF types: i-quants and ternary types.
    IQuant,
    /// FP8 elements.
    Fp8,
    /// MXFP4 and NVFP4.
    Fp4,
}

impl SchemeGroup {
    /// The group of a weight scheme, or `None` for a scheme no CUDA unit reads yet
    /// (compressed-tensors integers).
    #[must_use]
    pub fn of(scheme: &QuantScheme) -> Option<SchemeGroup> {
        Some(match scheme {
            QuantScheme::Plain(DType::F32 | DType::F16 | DType::BF16)
            | QuantScheme::Ggml(GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) => {
                SchemeGroup::Float
            }
            QuantScheme::Plain(DType::F8E4M3 | DType::F8E5M2) | QuantScheme::Fp8 { .. } => {
                SchemeGroup::Fp8
            }
            QuantScheme::Ggml(
                GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q8_0,
            ) => SchemeGroup::Legacy,
            QuantScheme::Ggml(
                GgmlType::Q2_K | GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K,
            ) => SchemeGroup::KQuant,
            QuantScheme::Ggml(_) => SchemeGroup::IQuant,
            QuantScheme::Mxfp4 { .. } | QuantScheme::Nvfp4 { .. } => SchemeGroup::Fp4,
            _ => return None,
        })
    }
}

/// One translation unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unit {
    /// The unit's path under `csrc/`, without `.cu`: `gemv/kquant`.
    pub name: &'static str,
    /// The ops its kernels implement.
    pub family: Family,
    /// The weight schemes it reads.
    pub group: SchemeGroup,
    /// Whether it holds bit-exact-class kernels: dequantisation, row gather, copies, KV page
    /// moves, index selection (RFC-0042 §4.1 rule 7, §6.4; RFC-0060 §4.2 rule 4). Such a unit
    /// is compiled with contraction off (RFC-0040 §10.2 rule 7).
    pub bit_exact: bool,
}

impl Unit {
    /// The unit's source, relative to the crate root.
    #[must_use]
    pub fn source_path(&self) -> String {
        format!("csrc/{}.cu", self.name)
    }

    /// The unit's generated instantiation list, relative to the crate root.
    #[must_use]
    pub fn inc_path(&self) -> String {
        format!("csrc/instantiate/{}.inc", self.name)
    }
}

const fn unit(name: &'static str, family: Family, group: SchemeGroup, bit_exact: bool) -> Unit {
    Unit {
        name,
        family,
        group,
        bit_exact,
    }
}

/// Every CUDA translation unit, in name order.
pub const UNITS: &[Unit] = &[
    unit("attn/decode", Family::AttnDecode, SchemeGroup::Any, false),
    unit("attn/kv_write", Family::KvWrite, SchemeGroup::Any, false),
    unit("attn/prefill", Family::AttnPrefill, SchemeGroup::Any, false),
    unit("convert/dequant", Family::Convert, SchemeGroup::Any, true),
    unit("copy/kv_pages", Family::KvPages, SchemeGroup::Any, true),
    unit("copy/rows", Family::Rows, SchemeGroup::Any, true),
    unit(
        "elementwise/basic",
        Family::Elementwise,
        SchemeGroup::Any,
        false,
    ),
    unit("gemm/float", Family::Gemm, SchemeGroup::Float, false),
    unit("gemm/kquant", Family::Gemm, SchemeGroup::KQuant, false),
    unit("gemv/float", Family::Gemv, SchemeGroup::Float, false),
    unit("gemv/fp4", Family::Gemv, SchemeGroup::Fp4, false),
    unit("gemv/fp8", Family::Gemv, SchemeGroup::Fp8, false),
    unit("gemv/iquant", Family::Gemv, SchemeGroup::IQuant, false),
    unit("gemv/kquant", Family::Gemv, SchemeGroup::KQuant, false),
    unit("gemv/legacy", Family::Gemv, SchemeGroup::Legacy, false),
    unit(
        "moe/grouped_kquant",
        Family::MoeGrouped,
        SchemeGroup::KQuant,
        false,
    ),
    unit("moe/route", Family::MoeRoute, SchemeGroup::Any, false),
    unit("norm/norm", Family::Norm, SchemeGroup::Any, false),
    unit(
        "sample/select",
        Family::SampleSelect,
        SchemeGroup::Any,
        true,
    ),
    unit("sample/tail", Family::SampleTail, SchemeGroup::Any, false),
];

/// The unit that implements `op` with weight scheme `weight` (the first of an `OpKey`'s
/// schemes), or `None` when no unit does.
#[must_use]
pub fn unit_for(op: &OpKind, weight: Option<&QuantScheme>) -> Option<&'static Unit> {
    let family = Family::of(op)?;
    let group = weight.and_then(SchemeGroup::of);
    UNITS
        .iter()
        .find(|u| u.family == family && (u.group == SchemeGroup::Any || Some(u.group) == group))
}
