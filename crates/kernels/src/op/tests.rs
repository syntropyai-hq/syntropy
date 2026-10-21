// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the op registry and op types against RFC-0040 §3.1, §3.2, §5.3 and RFC-0012 §2.

use std::collections::BTreeSet;

use proptest::prelude::*;
use smallvec::smallvec;
use syntropy_core::resource::KernelClass;
use syntropy_core::{AccumKind, ActQuant, Blake3, Bytes, DType, ErrorClass, Numerics, QuantScheme};

use super::*;
use crate::error::ErrorCode;

// ---- Golden data, transcribed from the RFC ------------------------------------------------

/// Every variant of `OpKind`, in the order of the `pub enum OpKind` block of RFC-0040 §3.1
/// (`plan/rfcs/RFC-0040-kernel-providers-and-backends.md`, the code block that starts at
/// `pub enum OpKind {`). `RmsNorm, LayerNorm, Rope(RopeKind), Act(ActKind), Add, Mul, Scale,
/// HyperMix`, `TopKTopP, SampleSeeded, Argmax`, `GatherRows, ScatterRows` and
/// `KvPack, KvUnpack` share a line there.
const GOLDEN_KINDS: [&str; 42] = [
    // Weight-bound linear algebra
    "Gemv",
    "Gemm",
    "GateUpAct",
    "MoeRoute",
    "MoeGrouped",
    "MoeCombine",
    // Attention and sequence mixers
    "AttnDecode",
    "AttnPrefill",
    "AttnPartial",
    "AttnMerge",
    "AttnIndex",
    "KvWrite",
    "ScanDecode",
    "ScanPrefill",
    "CausalConv1d",
    // Normalisation and elementwise
    "RmsNorm",
    "LayerNorm",
    "Rope",
    "Act",
    "Add",
    "Mul",
    "Scale",
    "HyperMix",
    // Tables, head and sampling
    "RowGather",
    "EmbedInject",
    "Pool",
    "Logits",
    "LogitsProcess",
    "TopKTopP",
    "SampleSeeded",
    "Argmax",
    "VerifyAccept",
    // Data movement
    "Convert",
    "GatherRows",
    "ScatterRows",
    "KvPack",
    "KvUnpack",
    "PartialReduce",
    "ApplyTableDeltas",
    "Timestamp",
    "TapGather",
    "HiddenTap",
];

/// The `KernelClass` mapping of RFC-0040 §3.1's last paragraph ("`OpKind` lines up with
/// RFC-0001's `KernelClass` for profiling: ..."), by variant name. Every other variant maps
/// to `None`. `AttnDecode` maps to `Attention(kind)`, checked separately.
const GOLDEN_CLASSES: [(&str, KernelClass); 8] = [
    ("Gemv", KernelClass::DenseGemv),
    ("GateUpAct", KernelClass::DenseGemv),
    ("Logits", KernelClass::DenseGemv),
    ("MoeGrouped", KernelClass::ExpertGemv),
    ("RowGather", KernelClass::RowGather),
    ("Convert", KernelClass::Copy),
    ("GatherRows", KernelClass::Copy),
    ("ScatterRows", KernelClass::Copy),
];

// ---- Enumerating every kind --------------------------------------------------------------

/// The variant name. The match is exhaustive, so a new kind fails to compile here until it
/// is added, and then fails `kind_list_matches_rfc` until the golden list has it.
fn variant_name(kind: &OpKind) -> &'static str {
    match kind {
        OpKind::Gemv => "Gemv",
        OpKind::Gemm => "Gemm",
        OpKind::GateUpAct { .. } => "GateUpAct",
        OpKind::MoeRoute { .. } => "MoeRoute",
        OpKind::MoeGrouped => "MoeGrouped",
        OpKind::MoeCombine => "MoeCombine",
        OpKind::AttnDecode(_) => "AttnDecode",
        OpKind::AttnPrefill(_) => "AttnPrefill",
        OpKind::AttnPartial(_) => "AttnPartial",
        OpKind::AttnMerge => "AttnMerge",
        OpKind::AttnIndex(_) => "AttnIndex",
        OpKind::KvWrite(_) => "KvWrite",
        OpKind::ScanDecode(_) => "ScanDecode",
        OpKind::ScanPrefill(_) => "ScanPrefill",
        OpKind::CausalConv1d => "CausalConv1d",
        OpKind::RmsNorm => "RmsNorm",
        OpKind::LayerNorm => "LayerNorm",
        OpKind::Rope(_) => "Rope",
        OpKind::Act(_) => "Act",
        OpKind::Add => "Add",
        OpKind::Mul => "Mul",
        OpKind::Scale => "Scale",
        OpKind::HyperMix => "HyperMix",
        OpKind::RowGather => "RowGather",
        OpKind::EmbedInject => "EmbedInject",
        OpKind::Pool(_) => "Pool",
        OpKind::Logits => "Logits",
        OpKind::LogitsProcess => "LogitsProcess",
        OpKind::TopKTopP => "TopKTopP",
        OpKind::SampleSeeded => "SampleSeeded",
        OpKind::Argmax => "Argmax",
        OpKind::VerifyAccept => "VerifyAccept",
        OpKind::Convert => "Convert",
        OpKind::GatherRows => "GatherRows",
        OpKind::ScatterRows => "ScatterRows",
        OpKind::KvPack => "KvPack",
        OpKind::KvUnpack => "KvUnpack",
        OpKind::PartialReduce => "PartialReduce",
        OpKind::ApplyTableDeltas => "ApplyTableDeltas",
        OpKind::Timestamp => "Timestamp",
        OpKind::TapGather => "TapGather",
        OpKind::HiddenTap => "HiddenTap",
    }
}

const ATTN_KINDS: [AttnKind; 7] = [
    AttnKind::Gqa,
    AttnKind::Mla,
    AttnKind::SlidingGqa { sinks: false },
    AttnKind::BlockSparse,
    AttnKind::TopKSparse,
    AttnKind::Compressed,
    AttnKind::Bidirectional,
];
const GATED: [GatedAct; 3] = [GatedAct::Silu, GatedAct::Gelu, GatedAct::ClampedSilu];
const ROUTE: [RouteNorm; 2] = [RouteNorm::Softmax, RouteNorm::Sigmoid];
const ROPE: [RopeKind; 4] = [
    RopeKind::Plain,
    RopeKind::Linear,
    RopeKind::Yarn,
    RopeKind::Llama3,
];
const ACT: [ActKind; 2] = [ActKind::Silu, ActKind::Gelu];
const POOL: [PoolKind; 3] = [PoolKind::Mean, PoolKind::Cls, PoolKind::LastToken];
const INDEX: [IndexKind; 3] = [
    IndexKind::BlockTopK,
    IndexKind::TokenTopK,
    IndexKind::CompressedTopK,
];
const SCAN: [ScanKind; 3] = [ScanKind::GatedDeltaNet, ScanKind::Kda, ScanKind::Mamba2];

/// Every kind with every parameter value (`SlidingGqa` with and without sinks).
pub(crate) fn all_kinds() -> Vec<OpKind> {
    let mut attn: Vec<AttnKind> = ATTN_KINDS.to_vec();
    attn.push(AttnKind::SlidingGqa { sinks: true });
    let mut kinds = vec![
        OpKind::Gemv,
        OpKind::Gemm,
        OpKind::MoeGrouped,
        OpKind::MoeCombine,
        OpKind::AttnMerge,
        OpKind::CausalConv1d,
        OpKind::RmsNorm,
        OpKind::LayerNorm,
        OpKind::Add,
        OpKind::Mul,
        OpKind::Scale,
        OpKind::HyperMix,
        OpKind::RowGather,
        OpKind::EmbedInject,
        OpKind::Logits,
        OpKind::LogitsProcess,
        OpKind::TopKTopP,
        OpKind::SampleSeeded,
        OpKind::Argmax,
        OpKind::VerifyAccept,
        OpKind::Convert,
        OpKind::GatherRows,
        OpKind::ScatterRows,
        OpKind::KvPack,
        OpKind::KvUnpack,
        OpKind::PartialReduce,
        OpKind::ApplyTableDeltas,
        OpKind::Timestamp,
        OpKind::TapGather,
        OpKind::HiddenTap,
    ];
    kinds.extend(GATED.map(|act| OpKind::GateUpAct { act }));
    kinds.extend(ROUTE.map(|norm| OpKind::MoeRoute { norm }));
    for &k in &attn {
        kinds.extend([
            OpKind::AttnDecode(k),
            OpKind::AttnPrefill(k),
            OpKind::AttnPartial(k),
            OpKind::KvWrite(k),
        ]);
    }
    kinds.extend(INDEX.map(OpKind::AttnIndex));
    kinds.extend(SCAN.map(OpKind::ScanDecode));
    kinds.extend(SCAN.map(OpKind::ScanPrefill));
    kinds.extend(ROPE.map(OpKind::Rope));
    kinds.extend(ACT.map(OpKind::Act));
    kinds.extend(POOL.map(OpKind::Pool));
    kinds
}

// ---- OpKind ------------------------------------------------------------------------------

#[test]
fn kind_list_matches_rfc() {
    let golden: BTreeSet<&str> = GOLDEN_KINDS.into_iter().collect();
    assert_eq!(
        golden.len(),
        GOLDEN_KINDS.len(),
        "golden list has duplicates"
    );
    let built: BTreeSet<&str> = all_kinds().iter().map(variant_name).collect();
    assert_eq!(built, golden);
}

#[test]
fn every_kind_has_a_dims_signature() {
    for kind in all_kinds() {
        let dims = kind.dims();
        assert!(
            (1..=6).contains(&dims.len()),
            "{kind:?} has {} dims; ShapeSig holds up to 6 inline",
            dims.len()
        );
        let distinct: BTreeSet<&str> = dims.iter().copied().collect();
        assert_eq!(distinct.len(), dims.len(), "{kind:?} repeats a dim name");
    }
}

/// The five signatures RFC-0040 §3.2 writes out ("`dims` per kind, for example: ...").
#[test]
fn dims_match_rfc_examples() {
    assert_eq!(OpKind::Gemv.dims(), ["n_out", "k_in"]);
    assert_eq!(OpKind::Gemm.dims(), ["n_out", "k_in"]);
    assert_eq!(
        OpKind::MoeGrouped.dims(),
        ["n_experts", "top_k", "hidden", "inter"]
    );
    assert_eq!(
        OpKind::AttnDecode(AttnKind::Gqa).dims(),
        ["n_q_heads", "n_kv_heads", "d_head", "page_tokens"]
    );
    assert_eq!(
        OpKind::AttnDecode(AttnKind::Mla).dims(),
        ["n_heads", "d_latent", "d_rope", "page_tokens"]
    );
    for scan in SCAN {
        assert_eq!(OpKind::ScanDecode(scan).dims(), ["n_heads", "d_k", "d_v"]);
    }
}

#[test]
fn kernel_class_table_matches_rfc() {
    for kind in all_kinds() {
        let name = variant_name(&kind);
        let expected = match kind {
            OpKind::AttnDecode(k) => Some(KernelClass::Attention(k)),
            _ => GOLDEN_CLASSES
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, class)| *class),
        };
        assert_eq!(kind.kernel_class(), expected, "{kind:?}");
    }
    // Every golden row names a real variant.
    let names: BTreeSet<&str> = GOLDEN_KINDS.into_iter().collect();
    for (name, _) in GOLDEN_CLASSES {
        assert!(names.contains(name), "{name}");
    }
}

// ---- OpKey, fusions and LayoutVersion ----------------------------------------------------

fn key(op: OpKind, variant: u16) -> OpKey {
    OpKey {
        op,
        schemes: smallvec![
            QuantScheme::Plain(DType::F32),
            QuantScheme::Plain(DType::F32),
            QuantScheme::Plain(DType::F32)
        ],
        layout: LayoutVersion::SOURCE,
        variant,
    }
}

/// RFC-0040 §3.1 lists six fusions; each owns one bit, in the listed order.
#[test]
fn fusion_bits_are_single_distinct_and_ordered() {
    let listed = [
        Fusions::FUSE_BIAS,
        Fusions::FUSE_RESIDUAL,
        Fusions::FUSE_ACT,
        Fusions::FUSE_ROPE,
        Fusions::FUSE_NORM_IN,
        Fusions::FUSE_SCALE_OUT,
    ];
    for (i, f) in listed.iter().enumerate() {
        assert_eq!(f.bits(), 1 << i, "{f:?}");
    }
    assert_eq!(Fusions::all().bits(), 0b11_1111);
    assert_eq!(Fusions::all().iter().count(), listed.len());
}

#[test]
fn key_fusions_reject_unknown_bits() {
    let fused = Fusions::FUSE_BIAS | Fusions::FUSE_RESIDUAL;
    assert_eq!(key(OpKind::Gemv, fused.bits()).fusions(), Some(fused));
    assert_eq!(key(OpKind::Gemv, 0).fusions(), Some(Fusions::empty()));
    assert_eq!(key(OpKind::Gemv, 1 << 6).fusions(), None);
    assert_eq!(key(OpKind::Gemv, 1 << 15).fusions(), None);
}

/// RFC-0040 §3.1 and §5.4: `SOURCE` is layout 0, and it is the default.
#[test]
fn layout_source_is_zero() {
    assert_eq!(LayoutVersion::SOURCE, LayoutVersion(0));
    assert_eq!(LayoutVersion::default(), LayoutVersion::SOURCE);
    assert_ne!(LayoutVersion(1), LayoutVersion::SOURCE);
}

/// Keys that differ only in layout or only in variant are different ops.
#[test]
fn layout_and_variant_are_part_of_the_key() {
    let base = key(OpKind::Gemv, 0);
    let repacked = OpKey {
        layout: LayoutVersion(1),
        ..base.clone()
    };
    let fused = key(OpKind::Gemv, Fusions::FUSE_BIAS.bits());
    assert_ne!(base, repacked);
    assert_ne!(base, fused);
    let mut reg = OpRegistry::new();
    for k in [base.clone(), repacked.clone(), fused.clone()] {
        reg.register(k).expect("distinct keys register");
    }
    assert_eq!(reg.len(), 3);
}

// ---- OpRegistry --------------------------------------------------------------------------

#[test]
fn registry_rejects_duplicate_keys() {
    let mut reg = OpRegistry::new();
    assert!(reg.is_empty());
    reg.register(key(OpKind::Gemv, 0)).expect("first registers");
    let before = reg.clone();
    assert_eq!(
        reg.register(key(OpKind::Gemv, 0)),
        Err(RegistryError::DuplicateKey(key(OpKind::Gemv, 0)))
    );
    assert_eq!(reg, before, "a refused key leaves the registry unchanged");
    assert_eq!(reg.len(), 1);
    assert!(reg.contains(&key(OpKind::Gemv, 0)));
    assert!(!reg.contains(&key(OpKind::Gemm, 0)));
}

#[test]
fn registry_rejects_unknown_fusion_bits() {
    let mut reg = OpRegistry::new();
    let bad = key(OpKind::Gemv, 1 << 6);
    assert_eq!(
        reg.register(bad.clone()),
        Err(RegistryError::UnknownFusion(bad))
    );
    assert!(reg.is_empty());
}

#[test]
fn registry_iterates_in_key_order() {
    let mut a = OpRegistry::new();
    let mut b = OpRegistry::new();
    let keys = [
        key(OpKind::Logits, 0),
        key(OpKind::Gemv, 0),
        key(OpKind::RmsNorm, 1),
    ];
    for k in keys.iter().cloned() {
        a.register(k).expect("registers");
    }
    for k in keys.iter().rev().cloned() {
        b.register(k).expect("registers");
    }
    let order_a: Vec<_> = a.iter().cloned().collect();
    let order_b: Vec<_> = b.iter().cloned().collect();
    assert_eq!(order_a, order_b);
    let mut sorted = keys.to_vec();
    sorted.sort();
    assert_eq!(order_a, sorted);
}

#[test]
fn registry_error_display_names_the_problem() {
    let e = RegistryError::DuplicateKey(key(OpKind::Gemv, 0));
    assert!(e.to_string().starts_with("op key registered twice: "));
    let e = RegistryError::UnknownFusion(key(OpKind::Gemv, 1 << 9));
    assert!(
        e.to_string()
            .starts_with("op key sets a variant bit no fusion owns: ")
    );
}

// ---- ShapeSig and buckets ----------------------------------------------------------------

#[test]
fn class_rows_follow_the_bucket() {
    assert_eq!(ShapeClass::Decode.rows(), 1);
    assert_eq!(ShapeClass::Verify { window: 9 }.rows(), 9);
    assert_eq!(ShapeClass::Batch { rows: 4 }.rows(), 4);
    assert_eq!(ShapeClass::Prefill { chunk: 256 }.rows(), 256);
    assert_eq!(ShapeClass::PrefillStream { chunk: 4096 }.rows(), 4096);
    let sig = ShapeSig::new(ShapeClass::Prefill { chunk: 64 }, &[4096, 4096]);
    assert_eq!(sig.rows, 64);
    assert_eq!(sig.dims.as_slice(), [4096, 4096]);
}

/// RFC-0012 §2's example: a plan chunk of 4096 in latency mode with speculation and
/// streaming gives `Decode`, `Verify`, the ladder 64, 256, 1024, then 4096, and
/// `PrefillStream`.
#[test]
fn bucket_set_matches_rfc_0012_example() {
    let set = bucket_set(4096, Some(5), &[], true);
    assert_eq!(
        set,
        [
            ShapeClass::Decode,
            ShapeClass::Verify { window: 5 },
            ShapeClass::Prefill { chunk: 64 },
            ShapeClass::Prefill { chunk: 256 },
            ShapeClass::Prefill { chunk: 1024 },
            ShapeClass::Prefill { chunk: 4096 },
            ShapeClass::PrefillStream { chunk: 4096 },
        ]
    );
    // 1 + 1 + 4 + 1 buckets, as RFC-0012 §2's table counts per parity.
    assert_eq!(set.len(), 7);
}

#[test]
fn bucket_set_ladder_is_strictly_below_the_chunk() {
    assert_eq!(
        bucket_set(1024, None, &[4, 2, 4], false),
        [
            ShapeClass::Decode,
            ShapeClass::Batch { rows: 2 },
            ShapeClass::Batch { rows: 4 },
            ShapeClass::Prefill { chunk: 64 },
            ShapeClass::Prefill { chunk: 256 },
            ShapeClass::Prefill { chunk: 1024 },
        ]
    );
    assert_eq!(
        bucket_set(8192, None, &[], false)
            .iter()
            .filter(|c| matches!(c, ShapeClass::Prefill { .. }))
            .count(),
        5
    );
    assert_eq!(
        bucket_set(32, None, &[], false),
        [ShapeClass::Decode, ShapeClass::Prefill { chunk: 32 }]
    );
}

fn shape_class() -> impl Strategy<Value = ShapeClass> {
    prop_oneof![
        Just(ShapeClass::Decode),
        any::<u16>().prop_map(|window| ShapeClass::Verify { window }),
        any::<u32>().prop_map(|rows| ShapeClass::Batch { rows }),
        any::<u32>().prop_map(|chunk| ShapeClass::Prefill { chunk }),
        any::<u32>().prop_map(|chunk| ShapeClass::PrefillStream { chunk }),
    ]
}

proptest! {
    /// RFC-0040 §3.2: a `ShapeSig` admits every `rows_active` up to its bucket's rows and
    /// rejects every larger count.
    #[test]
    fn shape_sig_admits_exactly_up_to_its_bucket(
        class in shape_class(),
        dims in prop::collection::vec(any::<u32>(), 0..=6),
        probe in any::<u32>(),
    ) {
        let sig = ShapeSig::new(class, &dims);
        let rows = class.rows();
        prop_assert_eq!(sig.rows, rows);
        prop_assert!(sig.admits(0));
        prop_assert!(sig.admits(rows));
        prop_assert!(sig.admits(probe % rows.saturating_add(1)));
        if let Some(above) = rows.checked_add(1) {
            prop_assert!(!sig.admits(above));
            prop_assert!(!sig.admits(above.saturating_add(probe % (u32::MAX - rows))));
        }
        prop_assert_eq!(sig.admits(probe), probe <= rows);
    }

    /// Every prompt length up to the plan's chunk fits some `Prefill` bucket of the set, and
    /// no `Prefill` bucket is larger than the chunk.
    #[test]
    fn bucket_set_covers_every_prefill_length(
        prefill_chunk in 1u32..=16_384,
        len in 0u32..=16_384,
    ) {
        let len = len % (prefill_chunk + 1);
        let set = bucket_set(prefill_chunk, None, &[], false);
        let fits: Vec<u32> = set
            .iter()
            .filter(|c| matches!(c, ShapeClass::Prefill { .. }))
            .map(|c| c.rows())
            .filter(|&rows| ShapeSig::new(ShapeClass::Prefill { chunk: rows }, &[]).admits(len))
            .collect();
        prop_assert!(!fits.is_empty());
        prop_assert!(fits.iter().all(|&rows| rows <= prefill_chunk));
    }
}

// ---- OpCaps, PreparedOp and the workspace check ------------------------------------------

fn caps(workspace: u64) -> OpCaps {
    OpCaps {
        workspace: Bytes(workspace),
        est: None,
        layout: LayoutVersion::SOURCE,
        capture_safe: true,
        deterministic: true,
        numerics: Numerics {
            accumulate: AccumKind::F32,
            act_in: ActQuant::None,
            out: DType::F32,
            softmax_f32: false,
            state_f32: false,
        },
        kernel_hash: Blake3::default(),
        source_key: Blake3::default(),
    }
}

/// RFC-0040 §5.3 and §13 code 14: a prepared op that asks for more workspace than `query`
/// declared fails with `WorkspaceExceeded`, class `Internal`.
#[test]
fn workspace_check_fails_an_op_asking_for_more() {
    let declared = caps(4096);
    let err = PreparedOp::new(Bytes(4097))
        .check_workspace(&declared)
        .expect_err("one byte over fails");
    assert_eq!(err.code, ErrorCode::WORKSPACE_EXCEEDED);
    assert_eq!(err.code, ErrorCode(14));
    assert_eq!(err.class, ErrorClass::Internal);
    assert_eq!(
        err.to_string(),
        "prepared op asks for 4097 bytes of workspace; query declared 4096 (Internal, Kernels 14)"
    );
}

#[test]
fn workspace_check_passes_at_or_below_the_declaration() {
    let declared = caps(4096);
    assert_eq!(
        PreparedOp::new(Bytes(4096)).check_workspace(&declared),
        Ok(())
    );
    assert_eq!(PreparedOp::new(Bytes(0)).check_workspace(&declared), Ok(()));
    assert_eq!(PreparedOp::new(Bytes(0)).check_workspace(&caps(0)), Ok(()));
    assert!(PreparedOp::new(Bytes(1)).check_workspace(&caps(0)).is_err());
}

// ---- OpArgs, TensorArg and Scalar --------------------------------------------------------

fn slice(bytes: u64) -> crate::BufferSlice {
    crate::BufferSlice {
        buffer: 1,
        offset: Bytes(0),
        bytes: Bytes(bytes),
    }
}

#[test]
fn packed_tensors_have_row_major_element_strides() {
    let t = TensorArg::packed(slice(4 * 2 * 3 * 5 * 7), DType::F32, [2, 3, 5, 7]).unwrap();
    assert_eq!(t.strides, [105, 35, 7, 1]);
    assert_eq!(t.span_bytes(), 4 * 210);
    assert!(t.fits());
    let short = TensorArg {
        buf: slice(4 * 210 - 1),
        ..t
    };
    assert!(!short.fits());
    // A stride that does not fit in a u32.
    assert_eq!(
        TensorArg::packed(slice(0), DType::U8, [2, 1 << 16, 1 << 16, 2]),
        None
    );
}

#[test]
fn span_bytes_edge_cases() {
    // Sub-byte elements round up to a whole byte: 3 FP4 elements span 2 bytes.
    let t = TensorArg::packed(slice(2), DType::F4E2M1, [1, 1, 1, 3]).unwrap();
    assert_eq!(t.span_bytes(), 2);
    // An empty dimension spans nothing, whatever the strides.
    let empty = TensorArg {
        dims: [4, 0, 4, 4],
        ..t
    };
    assert_eq!(empty.span_bytes(), 0);
    assert!(empty.fits());
    // Padded rows: 2 rows of 3 F16 elements, 8 elements apart, span 8 + 3 elements.
    let padded = TensorArg {
        buf: slice(22),
        dtype: DType::F16,
        dims: [1, 1, 2, 3],
        strides: [0, 0, 8, 1],
    };
    assert_eq!(padded.span_bytes(), 22);
    assert!(padded.fits());
    // A span past u64 saturates rather than wrapping.
    let huge = TensorArg {
        buf: slice(u64::MAX),
        dtype: DType::F64,
        dims: [u32::MAX; 4],
        strides: [u32::MAX; 4],
    };
    assert_eq!(huge.span_bytes(), u64::MAX);
}

#[test]
fn op_args_bind_weights_step_and_statics() {
    use syntropy_core::{BufferView, DeviceId, ViewKind};
    let x = TensorArg::packed(slice(64), DType::F32, [1, 1, 1, 16]).unwrap();
    let w = BufferView {
        kind: ViewKind::Device(DeviceId {
            node: syntropy_core::NodeId([0; 32]),
            ordinal: 0,
        }),
        addr: 0x1000,
        len: Bytes(4096),
    };
    let statics = [Scalar::F32(1e-6), Scalar::U32(3), Scalar::I32(-1)];
    let args = OpArgs {
        inputs: &[x],
        outputs: &[x],
        weights: &[w],
        step: slice(320),
        workspace: None,
        statics: &statics,
    };
    let copy = args;
    assert_eq!(copy, args);
    assert_eq!(args.weights[0].len, Bytes(4096));
    assert_eq!(args.statics[0], Scalar::F32(1e-6));
}
