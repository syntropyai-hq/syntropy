// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the M0 blocks against RFC-0043 §3.1–§3.4.
//!
//! - **Rows.** Each block's op sequence is checked against its table row, transcribed below.
//! - **Ops.** Each block, built alone in each form and configuration, emits only ops in its
//!   own `ops()`, and its two forms together emit all of them.
//! - **Snapshots.** Each block's full lowering (ops, classes, value types, weights, fusion
//!   hints and attributes) is compared with checked-in text in `snapshots/`. To regenerate
//!   after an intended change, run the tests with `SYNTROPY_UPDATE_SNAPSHOTS=1` and review the
//!   diff against the rows. The variable is refused when `CI` is set.
//! - **Nano graphs.** `llama` and `qwen_moe` graphs at RFC-0060 §12.1's `Nano` size are
//!   composed from the blocks. The families themselves, their `FamilySpec` and their
//!   `required_ops` are COMP-E16-T03 and COMP-E16-T04's, so the composition and the nano
//!   hyperparameters here are the minimum the acceptance criteria need.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use smallvec::smallvec;
use syntropy_core::{DType, LayerIx, TensorGroupId};
use syntropy_kernels::{AttnKind, Fusions, GatedAct, OpKind, RopeKind, RouteNorm};

use super::*;
use crate::ir::{
    Block, BoundarySet, CensusGroup, GroupClass, LayerKind, LayerProgram, ModelGraph, ResidualSpec,
    SideData, validate,
};

// ---- Nano hyperparameters ----------------------------------------------------------------

/// RFC-0060 §12.1 `Nano`: at most 8 layers, hidden 64–256, at most 16 experts, a byte-level
/// vocabulary of 256 plus specials.
const LAYERS: u16 = 4;
const HIDDEN: u32 = 64;
const VOCAB: u32 = 256 + 4;
const EPS: f32 = 1e-6;

fn rope() -> RopeParams {
    RopeParams {
        scaling: RopeScaling::Plain,
        theta: 10_000.0,
        dims: 16,
    }
}

/// GQA with 4 query heads and 2 KV heads of dim 16.
fn gqa_params(qkv: QkvLayout) -> GqaParams {
    GqaParams {
        heads: 4,
        kv_heads: 2,
        head_dim: 16,
        qkv,
        bias: false,
        rope: rope(),
        scale: None,
        softcap: None,
        eps: EPS,
    }
}

fn dense_params() -> DenseFfnParams {
    DenseFfnParams {
        inter: 4 * HIDDEN,
        act: GatedAct::Silu,
        eps: EPS,
    }
}

fn moe_params() -> MoeParams {
    MoeParams {
        experts: 8,
        top_k: 2,
        expert_inter: HIDDEN,
        renormalize: true,
        eps: EPS,
    }
}

/// The two M0 families (ADR-0003 decision 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    /// Llama-class dense: separate Q, K and V, Llama-3 RoPE scaling, SiLU-gated dense FFN.
    Llama,
    /// Qwen MoE: GQA with Q/K norms and softmax top-k MoE with no shared expert in every layer.
    QwenMoe,
}

/// A nano graph with its census, its required ops and each layer's `RoutingKnown`.
struct Nano {
    graph: ModelGraph,
    census: Vec<CensusGroup>,
    required: OpSet,
    routing: Vec<(LayerIx, RoutingKnown)>,
}

/// Composes a nano graph from the blocks, as a family module will.
fn nano(family: Family, form: Form) -> Nano {
    let mut b = Builder::new();
    let mut census = Vec::new();
    let mut group = |class, layer: Option<u16>| {
        let id = TensorGroupId(u32::try_from(census.len()).unwrap());
        census.push(CensusGroup {
            id,
            class,
            layer: layer.map(LayerIx),
        });
        id
    };
    let mut gqa_p = match family {
        Family::Llama => gqa_params(QkvLayout::Separate { qk_norm: false }),
        Family::QwenMoe => gqa_params(QkvLayout::Separate { qk_norm: true }),
    };
    if family == Family::Llama {
        // Llama-3-style scaling has no parameters in the plan yet (see `RopeScaling`); the
        // nano graph exercises linear scaling instead.
        gqa_p.rope.scaling = RopeScaling::Linear { factor: 8.0 };
    }
    let embed_p = EmbedParams {
        hidden: HIDDEN,
        vocab: VOCAB,
        scale: None,
    };
    let head_p = LmHeadParams {
        vocab: VOCAB,
        softcap: None,
        eps: EPS,
    };
    let mut required = OpSet::new();
    required.extend(embed_p.ops());
    required.extend(gqa_p.ops());
    required.extend(head_p.ops());
    match family {
        Family::Llama => required.extend(dense_params().ops()),
        Family::QwenMoe => required.extend(moe_params().ops()),
    }

    let tokens = b.value(DType::I32, &[Dim::Rows], ValueKind::TokenIds);
    let mut prologue = Program {
        inputs: smallvec![tokens],
        ..Program::default()
    };
    let table = group(GroupClass::RowLookup, None);
    let mut residual = embed(&mut b, &mut prologue, &embed_p, table, tokens);
    b.set_kind(residual, ValueKind::Residual);
    prologue.outputs.push(residual);

    let mut layers = Vec::new();
    let mut routing = Vec::new();
    for i in 0..LAYERS {
        let ix = LayerIx(i);
        let mut body = Program {
            inputs: smallvec![residual],
            ..Program::default()
        };
        let mixer = group(GroupClass::DenseEveryToken, Some(i));
        let h = gqa(&mut b, &mut body, &gqa_p, form, mixer, residual);
        let (out, ffn) = match family {
            Family::Llama => {
                let ffn = group(GroupClass::DenseEveryToken, Some(i));
                (
                    dense_ffn(&mut b, &mut body, &dense_params(), form, ffn, h),
                    Block::DenseFfn,
                )
            }
            Family::QwenMoe => {
                let router = group(GroupClass::Router, Some(i));
                for _ in 0..moe_params().experts {
                    group(GroupClass::Expert, Some(i));
                }
                let (out, known) = moe(&mut b, &mut body, &moe_params(), form, router, ix, h);
                routing.push((ix, known));
                (out, Block::Moe)
            }
        };
        b.set_kind(out, ValueKind::Residual);
        body.outputs.push(out);
        residual = out;
        layers.push(LayerProgram {
            layer: ix,
            kind: LayerKind {
                mixer: Block::Gqa,
                ffn,
            },
            blocks: smallvec![Block::Gqa, ffn],
            body,
        });
    }

    let mut epilogue = Program {
        inputs: smallvec![residual],
        ..Program::default()
    };
    let head = group(GroupClass::Head, None);
    let logits = lm_head(&mut b, &mut epilogue, &head_p, head, residual);
    epilogue.outputs.push(logits);

    let n = usize::from(LAYERS);
    let graph = ModelGraph {
        family: match family {
            Family::Llama => "llama",
            Family::QwenMoe => "qwen_moe",
        },
        values: b.values,
        prologue,
        layers,
        epilogue,
        mtp: Vec::new(),
        encoders: Vec::new(),
        residual: ResidualSpec {
            streams: 1,
            dtype: DType::F32,
        },
        // Every boundary is legal: only the residual stream crosses.
        boundaries: BoundarySet {
            legal: vec![true; n + 1],
            side: vec![SideData::default(); n + 1],
        },
    };
    Nano {
        graph,
        census,
        required,
        routing,
    }
}

/// Every op the graph uses.
fn used_ops(g: &ModelGraph) -> OpSet {
    let programs = std::iter::once(&g.prologue)
        .chain(g.layers.iter().map(|l| &l.body))
        .chain(std::iter::once(&g.epilogue));
    programs
        .flat_map(|p| p.nodes.iter().map(|n| n.op))
        .collect()
}

// ---- Nano graphs ---------------------------------------------------------------------------

#[test]
fn nano_graphs_validate() {
    for family in [Family::Llama, Family::QwenMoe] {
        for form in Form::ALL {
            let built = nano(family, form);
            assert_eq!(
                validate(&built.graph, &built.census, &built.required),
                Ok(()),
                "{family:?} {form:?}"
            );
        }
    }
}

/// The ops each graph uses are a subset of its `required_ops`, and the two forms together
/// use every required op, so no block's `ops` lists an op it never emits.
#[test]
fn used_ops_are_a_subset_of_required_ops() {
    for family in [Family::Llama, Family::QwenMoe] {
        let mut both = OpSet::new();
        for form in Form::ALL {
            let built = nano(family, form);
            let used = used_ops(&built.graph);
            assert!(
                used.is_subset(&built.required),
                "{family:?} {form:?}: {used:?}"
            );
            both.extend(used);
        }
        assert_eq!(both, nano(family, Form::Decode).required, "{family:?}");
    }
}

#[test]
fn required_ops_per_family() {
    let common = [
        OpKind::RowGather,
        OpKind::RmsNorm,
        OpKind::Gemv,
        OpKind::Gemm,
        OpKind::KvWrite(AttnKind::Gqa),
        OpKind::AttnDecode(AttnKind::Gqa),
        OpKind::AttnPrefill(AttnKind::Gqa),
        OpKind::Add,
        OpKind::Logits,
    ];
    let mut llama: OpSet = common.into_iter().collect();
    llama.insert(OpKind::GateUpAct {
        act: GatedAct::Silu,
    });
    assert_eq!(nano(Family::Llama, Form::Decode).required, llama);
    let mut qwen: OpSet = common.into_iter().collect();
    qwen.extend([
        OpKind::MoeRoute {
            norm: RouteNorm::Softmax,
        },
        OpKind::MoeGrouped,
        OpKind::MoeCombine,
    ]);
    assert_eq!(nano(Family::QwenMoe, Form::Decode).required, qwen);
}

/// RFC-0043 §3.4: every M0 MoE layer's routing is known at its router.
#[test]
fn moe_layers_are_routing_known_at_router() {
    let built = nano(Family::QwenMoe, Form::Decode);
    let layers: Vec<LayerIx> = (0..LAYERS).map(LayerIx).collect();
    assert_eq!(
        built.routing,
        layers
            .iter()
            .map(|&l| (l, RoutingKnown::AtRouter))
            .collect::<Vec<_>>()
    );
    assert!(nano(Family::Llama, Form::Decode).routing.is_empty());
}

/// The nano graphs sit inside RFC-0060 §12.1's `Nano` size class.
#[test]
fn nano_graphs_are_nano_sized() {
    for family in [Family::Llama, Family::QwenMoe] {
        let g = nano(family, Form::Decode);
        assert!(g.graph.layers.len() <= 8);
        assert!((64..=256).contains(&HIDDEN));
        let experts = g
            .census
            .iter()
            .filter(|c| c.class == GroupClass::Expert && c.layer == Some(LayerIx(0)))
            .count();
        assert!(experts <= 16);
        // The MoE pre-FFN norm reads the router group, as RFC-0041 §3.1 places it.
        if family == Family::QwenMoe {
            let router = g
                .census
                .iter()
                .find(|c| c.class == GroupClass::Router && c.layer == Some(LayerIx(0)))
                .unwrap()
                .id;
            let norm = g.graph.layers[0]
                .body
                .nodes
                .iter()
                .rev()
                .find(|n| n.op == OpKind::RmsNorm)
                .unwrap();
            assert_eq!(norm.weights[0].group, GroupSel::Fixed(router));
        }
    }
}

// ---- Rows ----------------------------------------------------------------------------------

/// Builds one block into an empty program, after one input value of `[Rows, HIDDEN]` (or
/// token IDs for `Embed`).
fn one_block(build: impl FnOnce(&mut Builder, &mut Program, ValueId)) -> (Builder, Program) {
    let mut b = Builder::new();
    let x = b.rows(HIDDEN);
    let mut p = Program {
        inputs: smallvec![x],
        ..Program::default()
    };
    build(&mut b, &mut p, x);
    (b, p)
}

fn ops(p: &Program) -> Vec<OpKind> {
    p.nodes.iter().map(|n| n.op).collect()
}

const G: TensorGroupId = TensorGroupId(7);

/// RFC-0043 §3.1 `Gqa`: `RmsNorm` → `Gemv/Gemm` (QKV, often fused) → `Rope` folded into
/// `KvWrite(Gqa)` → `AttnDecode/AttnPrefill(Gqa)` → `Gemv/Gemm` (O) → `Add`.
#[test]
fn gqa_lowers_to_its_row() {
    for form in Form::ALL {
        let mm = form.matmul();
        let attn = match form {
            Form::Decode => OpKind::AttnDecode(AttnKind::Gqa),
            Form::Prefill => OpKind::AttnPrefill(AttnKind::Gqa),
        };
        let kv = OpKind::KvWrite(AttnKind::Gqa);
        let (_, fused) = one_block(|b, p, x| {
            gqa(b, p, &gqa_params(QkvLayout::Fused), form, G, x);
        });
        assert_eq!(
            ops(&fused),
            [OpKind::RmsNorm, mm, kv, attn, mm, OpKind::Add]
        );
        let (_, separate) = one_block(|b, p, x| {
            gqa(
                b,
                p,
                &gqa_params(QkvLayout::Separate { qk_norm: false }),
                form,
                G,
                x,
            );
        });
        assert_eq!(
            ops(&separate),
            [OpKind::RmsNorm, mm, mm, mm, kv, attn, mm, OpKind::Add]
        );
        let (_, normed) = one_block(|b, p, x| {
            gqa(
                b,
                p,
                &gqa_params(QkvLayout::Separate { qk_norm: true }),
                form,
                G,
                x,
            );
        });
        assert_eq!(
            ops(&normed),
            [
                OpKind::RmsNorm,
                mm,
                mm,
                mm,
                OpKind::RmsNorm,
                OpKind::RmsNorm,
                kv,
                attn,
                mm,
                OpKind::Add
            ]
        );
        // No `Rope` node: it is folded into `KvWrite` and the attention op.
        for p in [&fused, &separate, &normed] {
            assert!(!ops(p).iter().any(|op| matches!(op, OpKind::Rope(_))));
            for n in p.nodes.iter().filter(|n| n.op == kv || n.op == attn) {
                assert!(n.fuse.allowed.contains(Fusions::FUSE_ROPE));
            }
        }
    }
}

/// RFC-0043 §3.2 `DenseFfn`: `RmsNorm` → `GateUpAct` → `Gemv/Gemm` (down) → `Add`.
#[test]
fn dense_ffn_lowers_to_its_row() {
    for form in Form::ALL {
        let (_, p) = one_block(|b, p, x| {
            dense_ffn(b, p, &dense_params(), form, G, x);
        });
        assert_eq!(
            ops(&p),
            [
                OpKind::RmsNorm,
                OpKind::GateUpAct {
                    act: GatedAct::Silu
                },
                form.matmul(),
                OpKind::Add
            ]
        );
    }
}

/// RFC-0043 §3.2 `Moe` without a shared expert: `RmsNorm` → `MoeRoute` → `MoeGrouped` over
/// `GroupSel::Experts` → `MoeCombine` → `Add`.
#[test]
fn moe_lowers_to_its_row() {
    for form in Form::ALL {
        let (_, p) = one_block(|b, p, x| {
            let (_, known) = moe(b, p, &moe_params(), form, G, LayerIx(3), x);
            assert_eq!(known, RoutingKnown::AtRouter);
        });
        assert_eq!(
            ops(&p),
            [
                OpKind::RmsNorm,
                OpKind::MoeRoute {
                    norm: RouteNorm::Softmax
                },
                OpKind::MoeGrouped,
                OpKind::MoeCombine,
                OpKind::Add
            ]
        );
        let grouped = &p.nodes[2];
        assert_eq!(
            grouped.weights[0].group,
            GroupSel::Experts { layer: LayerIx(3) }
        );
        assert_eq!(grouped.class, NodeClass::Expert);
        assert_eq!(p.nodes[1].class, NodeClass::Router);
    }
}

/// RFC-0043 §3.3 `Embed`: `RowGather` → optional scale. `LmHead`: final norm → `Logits` →
/// optional logit soft-cap.
#[test]
fn embed_and_lm_head_lower_to_their_rows() {
    let mut params = EmbedParams {
        hidden: HIDDEN,
        vocab: VOCAB,
        scale: None,
    };
    let tokens = |b: &mut Builder| b.value(DType::I32, &[Dim::Rows], ValueKind::TokenIds);
    let (_, plain) = one_block(|b, p, _| {
        let t = tokens(b);
        embed(b, p, &params, G, t);
    });
    assert_eq!(ops(&plain), [OpKind::RowGather]);
    params.scale = Some(8.0);
    let (_, scaled) = one_block(|b, p, _| {
        let t = tokens(b);
        embed(b, p, &params, G, t);
    });
    assert_eq!(ops(&scaled), [OpKind::RowGather, OpKind::Scale]);
    assert_eq!(
        params.ops(),
        OpSet::from([OpKind::RowGather, OpKind::Scale])
    );

    let head = LmHeadParams {
        vocab: VOCAB,
        softcap: Some(30.0),
        eps: EPS,
    };
    let (_, p) = one_block(|b, p, x| {
        lm_head(b, p, &head, G, x);
    });
    assert_eq!(ops(&p), [OpKind::RmsNorm, OpKind::Logits]);
    assert_eq!(p.nodes[1].attrs.get("softcap"), Some(AttrValue::F32(30.0)));
}

/// Builds one block alone in each form and checks that every op it emits is in `declared`, its
/// own `ops()`, and that the two forms together emit all of them.
fn assert_emits_its_ops(
    what: &str,
    declared: &OpSet,
    build: impl Fn(&mut Builder, &mut Program, Form, ValueId),
) {
    let mut both = OpSet::new();
    for form in Form::ALL {
        let (_, p) = one_block(|b, p, x| build(b, p, form, x));
        let emitted: OpSet = ops(&p).into_iter().collect();
        assert!(
            emitted.is_subset(declared),
            "{what} {form:?} emits {:?}, which its ops() leaves out",
            emitted.difference(declared).collect::<Vec<_>>()
        );
        both.extend(emitted);
    }
    assert_eq!(
        &both, declared,
        "{what}: ops() lists an op neither form emits"
    );
}

/// Every op a block emits is in that block's own `ops()`, for each block, each form and each
/// configuration that changes the lowering.
#[test]
fn each_block_emits_only_its_own_ops() {
    let layouts = [
        QkvLayout::Fused,
        QkvLayout::Separate { qk_norm: false },
        QkvLayout::Separate { qk_norm: true },
    ];
    for qkv in layouts {
        for bias in [false, true] {
            for (scale, softcap) in [(None, None), (Some(0.125), Some(50.0))] {
                let params = GqaParams {
                    bias,
                    scale,
                    softcap,
                    ..gqa_params(qkv)
                };
                assert_emits_its_ops(
                    &format!("Gqa {qkv:?} bias={bias} scale={scale:?} softcap={softcap:?}"),
                    &params.ops(),
                    |b, p, form, x| {
                        gqa(b, p, &params, form, G, x);
                    },
                );
            }
        }
    }

    for act in [GatedAct::Silu, GatedAct::Gelu, GatedAct::ClampedSilu] {
        let params = DenseFfnParams {
            act,
            ..dense_params()
        };
        assert_emits_its_ops(
            &format!("DenseFfn {act:?}"),
            &params.ops(),
            |b, p, form, x| {
                dense_ffn(b, p, &params, form, G, x);
            },
        );
    }

    for renormalize in [true, false] {
        let params = MoeParams {
            renormalize,
            ..moe_params()
        };
        assert_emits_its_ops(
            &format!("Moe renormalize={renormalize}"),
            &params.ops(),
            |b, p, form, x| {
                moe(b, p, &params, form, G, LayerIx(0), x);
            },
        );
    }

    for scale in [None, Some(8.0)] {
        let params = EmbedParams {
            hidden: HIDDEN,
            vocab: VOCAB,
            scale,
        };
        assert_emits_its_ops(
            &format!("Embed scale={scale:?}"),
            &params.ops(),
            |b, p, _, _| {
                let t = b.value(DType::I32, &[Dim::Rows], ValueKind::TokenIds);
                embed(b, p, &params, G, t);
            },
        );
    }

    for softcap in [None, Some(30.0)] {
        let params = LmHeadParams {
            vocab: VOCAB,
            softcap,
            eps: EPS,
        };
        assert_emits_its_ops(
            &format!("LmHead softcap={softcap:?}"),
            &params.ops(),
            |b, p, _, x| {
                lm_head(b, p, &params, G, x);
            },
        );
    }
}

#[test]
fn rope_rules_have_distinct_codes_and_kinds() {
    let rules = [RopeScaling::Plain, RopeScaling::Linear { factor: 2.0 }];
    let codes: BTreeSet<u32> = rules.iter().map(rope_kind_code).collect();
    assert_eq!(codes, BTreeSet::from([0, 1]));
    assert_eq!(rules.map(|r| r.kind()), [RopeKind::Plain, RopeKind::Linear]);
}

/// Every input of the RoPE table is an attribute: the rule, the base, the rotated dims and,
/// for linear scaling, its factor.
#[test]
fn rope_attributes_carry_every_table_input() {
    let mut params = gqa_params(QkvLayout::Fused);
    params.rope = RopeParams {
        scaling: RopeScaling::Linear { factor: 4.0 },
        theta: 500_000.0,
        dims: 8,
    };
    let (_, p) = one_block(|b, p, x| {
        gqa(b, p, &params, Form::Decode, G, x);
    });
    for node in p
        .nodes
        .iter()
        .filter(|n| n.fuse.allowed.contains(Fusions::FUSE_ROPE))
    {
        assert_eq!(node.attrs.get("rope_kind"), Some(AttrValue::U32(1)));
        assert_eq!(
            node.attrs.get("rope_theta"),
            Some(AttrValue::F32(500_000.0))
        );
        assert_eq!(node.attrs.get("rope_dims"), Some(AttrValue::U32(8)));
        assert_eq!(node.attrs.get("rope_factor"), Some(AttrValue::F32(4.0)));
    }
    params.rope.scaling = RopeScaling::Plain;
    let (_, p) = one_block(|b, p, x| {
        gqa(b, p, &params, Form::Decode, G, x);
    });
    assert!(p.nodes.iter().all(|n| n.attrs.get("rope_factor").is_none()));
}

/// The Q/K norms are per head: each normalises a `[Rows, heads, head_dim]` value over its last
/// dim, `head_dim`, never over all heads at once.
#[test]
fn qk_norms_are_per_head() {
    let params = gqa_params(QkvLayout::Separate { qk_norm: true });
    let (b, p) = one_block(|b, p, x| {
        gqa(b, p, &params, Form::Decode, G, x);
    });
    let norms: Vec<&crate::ir::Node> = p
        .nodes
        .iter()
        .filter(|n| n.op == OpKind::RmsNorm)
        .skip(1)
        .collect();
    assert_eq!(norms.len(), 2);
    let dims = |v: ValueId| b.values[v.0 as usize].ty.dims.to_vec();
    let d = Dim::Const(u32::from(params.head_dim));
    for (norm, heads) in norms.iter().zip([params.heads, params.kv_heads]) {
        let expected = vec![Dim::Rows, Dim::Const(u32::from(heads)), d];
        assert_eq!(dims(norm.inputs[0]), expected);
        assert_eq!(dims(norm.outputs[0]), expected);
        // The normalised dim is the last one: one head's width.
        assert_eq!(dims(norm.inputs[0]).last(), Some(&d));
    }
    // V, which is not normalised, has the per-head shape too.
    let v = p
        .nodes
        .iter()
        .find(|n| n.weights.iter().any(|w| w.role == TensorRole::V))
        .unwrap();
    assert_eq!(
        dims(v.outputs[0]),
        vec![Dim::Rows, Dim::Const(u32::from(params.kv_heads)), d]
    );
}

#[test]
fn set_kind_ignores_foreign_values() {
    let mut b = Builder::new();
    let v = b.rows(HIDDEN);
    b.set_kind(ValueId(99), ValueKind::Residual);
    b.set_kind(v, ValueKind::Residual);
    assert_eq!(b.values[0].kind, ValueKind::Residual);
    assert_eq!(b.values.len(), 1);
}

// ---- Snapshots -----------------------------------------------------------------------------

/// One line per node: kind, class, inputs and outputs with their types, weights, fusion
/// hints and attributes. Values are written as `v<id>:<dtype>[dims]`.
fn render(b: &Builder, p: &Program) -> String {
    let value = |v: &ValueId| {
        let ty = &b.values[v.0 as usize].ty;
        let dims: Vec<String> = ty
            .dims
            .iter()
            .map(|d| match d {
                Dim::Const(n) => n.to_string(),
                other => format!("{other:?}"),
            })
            .collect();
        format!("v{}:{:?}[{}]", v.0, ty.dtype, dims.join(","))
    };
    let mut out = String::new();
    for n in &p.nodes {
        let inputs: Vec<String> = n.inputs.iter().map(value).collect();
        let outputs: Vec<String> = n.outputs.iter().map(value).collect();
        let weights: Vec<String> = n
            .weights
            .iter()
            .map(|w| match w.group {
                GroupSel::Fixed(g) => format!("{:?}@g{}", w.role, g.0),
                GroupSel::Experts { layer } => format!("{:?}@experts(l{})", w.role, layer.0),
            })
            .collect();
        let attrs: Vec<String> = n
            .attrs
            .entries
            .iter()
            .map(|(k, v)| match v {
                AttrValue::U32(x) => format!("{k}={x}"),
                AttrValue::F32(x) => format!("{k}={x:?}"),
                AttrValue::Bool(x) => format!("{k}={x}"),
            })
            .collect();
        let fuse = if n.fuse.allowed.is_empty() {
            String::from("-")
        } else {
            format!("{:?}", n.fuse.allowed)
                .trim_start_matches("Fusions(")
                .trim_end_matches(')')
                .to_owned()
        };
        writeln!(
            out,
            "{:?} [{:?}] in({}) out({}) w({}) fuse({}) attrs({})",
            n.op,
            n.class,
            inputs.join(" "),
            outputs.join(" "),
            weights.join(" "),
            fuse,
            attrs.join(" ")
        )
        .unwrap();
    }
    out
}

/// Every block configuration the snapshots cover, by file name: both forms of each.
fn snapshot(name: &str) -> String {
    let mut text = String::new();
    for form in Form::ALL {
        let (b, p) = one_block(|b, p, x| match name {
            "gqa_fused" => {
                let mut params = gqa_params(QkvLayout::Fused);
                params.bias = true;
                params.scale = Some(0.25);
                params.softcap = Some(50.0);
                gqa(b, p, &params, form, G, x);
            }
            "gqa_separate" => {
                let mut params = gqa_params(QkvLayout::Separate { qk_norm: true });
                params.rope = RopeParams {
                    scaling: RopeScaling::Linear { factor: 4.0 },
                    theta: 1_000_000.0,
                    dims: 8,
                };
                gqa(b, p, &params, form, G, x);
            }
            "gqa_no_rope" => {
                let mut params = gqa_params(QkvLayout::Fused);
                params.rope.dims = 0;
                gqa(b, p, &params, form, G, x);
            }
            "dense_ffn" => {
                dense_ffn(b, p, &dense_params(), form, G, x);
            }
            "moe" => {
                moe(b, p, &moe_params(), form, G, LayerIx(3), x);
            }
            "embed" => {
                let params = EmbedParams {
                    hidden: HIDDEN,
                    vocab: VOCAB,
                    scale: Some(8.0),
                };
                let _ = x;
                let tokens = b.value(DType::I32, &[Dim::Rows], ValueKind::TokenIds);
                embed(b, p, &params, G, tokens);
            }
            "lm_head" => {
                let params = LmHeadParams {
                    vocab: VOCAB,
                    softcap: Some(30.0),
                    eps: EPS,
                };
                lm_head(b, p, &params, G, x);
            }
            other => panic!("no snapshot named {other}"),
        });
        writeln!(text, "## {form:?}").unwrap();
        text.push_str(&render(&b, &p));
    }
    text
}

/// Whether this run rewrites the snapshots: only when `SYNTROPY_UPDATE_SNAPSHOTS` is set, and
/// never under CI, where a set variable would let any lowering change pass.
///
/// # Errors
///
/// When both `SYNTROPY_UPDATE_SNAPSHOTS` and `CI` are set.
fn update_snapshots(
    update: Option<std::ffi::OsString>,
    ci: Option<std::ffi::OsString>,
) -> Result<bool, &'static str> {
    match (update, ci) {
        (Some(_), Some(_)) => {
            Err("SYNTROPY_UPDATE_SNAPSHOTS is set under CI; snapshots are only rewritten locally")
        }
        (update, _) => Ok(update.is_some()),
    }
}

#[test]
fn snapshot_updates_are_refused_under_ci() {
    let set = || Some(std::ffi::OsString::from("1"));
    assert_eq!(update_snapshots(None, None), Ok(false));
    assert_eq!(update_snapshots(None, set()), Ok(false));
    assert_eq!(update_snapshots(set(), None), Ok(true));
    assert!(update_snapshots(set(), set()).is_err());
}

fn check_snapshot(name: &str, expected: &str) {
    let actual = snapshot(name);
    let update = update_snapshots(
        std::env::var_os("SYNTROPY_UPDATE_SNAPSHOTS"),
        std::env::var_os("CI"),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    if update {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/blocks/snapshots")
            .join(format!("{name}.txt"));
        std::fs::write(path, &actual).unwrap();
        return;
    }
    // A CRLF checkout (Git for Windows) must not fail an unchanged lowering.
    let expected = expected.replace("\r\n", "\n");
    assert_eq!(
        actual, expected,
        "snapshot {name} differs; the actual lowering is:\n{actual}"
    );
}

macro_rules! snapshot_tests {
    ($($test:ident: $name:literal;)*) => {
        $(
            #[test]
            fn $test() {
                check_snapshot($name, include_str!(concat!("snapshots/", $name, ".txt")));
            }
        )*
    };
}

snapshot_tests! {
    snapshot_gqa_fused: "gqa_fused";
    snapshot_gqa_separate: "gqa_separate";
    snapshot_gqa_no_rope: "gqa_no_rope";
    snapshot_dense_ffn: "dense_ffn";
    snapshot_moe: "moe";
    snapshot_embed: "embed";
    snapshot_lm_head: "lm_head";
}
