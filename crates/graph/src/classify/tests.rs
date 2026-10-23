// SPDX-License-Identifier: MIT OR Apache-2.0

use proptest::prelude::*;
use syntropy_core::{ExpertIx, LayerIx};
use syntropy_manifest::census::{TensorClass, TowerIx};
use syntropy_manifest::hparams::{HParamValue, HyperParams, N_EXPERTS};

use super::*;
use crate::ir::{Block, TensorRole};

const fn rule(
    pattern: &'static str,
    block: BlockSel,
    role: TensorRole,
    class: ClassSel,
    shape: &'static [ShapeDim],
) -> ClassRule {
    ClassRule {
        pattern,
        block,
        role,
        class,
        shape,
        aux_of: None,
    }
}

use ShapeDim::{Any, Hp, Mul};

const TOWER: ClassSel = ClassSel::Encoder { tower: TowerIx(0) };

/// GGUF names, shapes innermost first.
const GGUF: &[ClassRule] = &[
    rule(
        "token_embd.weight",
        BlockSel::Prologue,
        TensorRole::Embed,
        ClassSel::RowLookup,
        &[Hp("hidden"), Hp("vocab")],
    ),
    rule(
        "blk.{L}.attn_norm.weight",
        BlockSel::Layer(Block::Gqa),
        TensorRole::Norm,
        ClassSel::DenseEveryToken,
        &[Hp("hidden")],
    ),
    rule(
        "blk.{L}.attn_q.weight",
        BlockSel::Layer(Block::Gqa),
        TensorRole::Q,
        ClassSel::DenseEveryToken,
        &[Hp("hidden"), Mul("n_heads", "d_head")],
    ),
    rule(
        "blk.{L}.ffn_gate_inp.weight",
        BlockSel::Layer(Block::Moe),
        TensorRole::Router,
        ClassSel::Router,
        &[Hp("hidden"), Hp(N_EXPERTS)],
    ),
    rule(
        "blk.{L}.ffn_gate_exps.weight",
        BlockSel::Layer(Block::Moe),
        TensorRole::Gate,
        ClassSel::ExpertsStacked,
        &[Hp("hidden"), Hp("d_ffn"), Hp(N_EXPERTS)],
    ),
    rule(
        "output.weight",
        BlockSel::Epilogue,
        TensorRole::Head,
        ClassSel::Head,
        &[Hp("hidden"), Hp("vocab")],
    ),
    rule(
        "blk.{L}.nextn.eh_proj.weight",
        BlockSel::Mtp(0),
        TensorRole::Proj,
        ClassSel::DenseEveryToken,
        &[Any, Hp("hidden")],
    ),
    rule(
        "v.blk.{N}.attn_q.weight",
        BlockSel::Encoder,
        TensorRole::Q,
        TOWER,
        &[Hp("v_hidden"), Hp("v_hidden")],
    ),
];

/// safetensors names, shapes outermost first.
const ST: &[ClassRule] = &[
    rule(
        "model.layers.{L}.self_attn.q_proj.weight",
        BlockSel::Layer(Block::Gqa),
        TensorRole::Q,
        ClassSel::DenseEveryToken,
        &[Mul("n_heads", "d_head"), Hp("hidden")],
    ),
    rule(
        "model.layers.{L}.mlp.experts.{E}.gate_proj.weight",
        BlockSel::Layer(Block::Moe),
        TensorRole::Gate,
        ClassSel::Expert,
        &[Hp("d_ffn"), Hp("hidden")],
    ),
    rule(
        "model.layers.{L}.mlp.experts.gate_up_proj",
        BlockSel::Layer(Block::Moe),
        TensorRole::GateUp,
        ClassSel::ExpertsStacked,
        &[Hp(N_EXPERTS), Hp("hidden"), Any],
    ),
    rule(
        "model.layers.{L}.self_attn.rotary_emb.inv_freq",
        BlockSel::Layer(Block::Gqa),
        TensorRole::Other(0),
        ClassSel::Ignored {
            reason: "a recomputed buffer, not a weight",
        },
        &[Any],
    ),
    rule(
        "model.visual.blocks.{N}.attn.qkv.weight",
        BlockSel::Encoder,
        TensorRole::Qkv,
        TOWER,
        &[Any, Hp("v_hidden")],
    ),
    rule(
        "model.multi_modal_projector.linear_{N}.weight",
        BlockSel::Encoder,
        TensorRole::Proj,
        TOWER,
        &[Hp("hidden"), Any],
    ),
];

const TABLES: &[RuleTable] = &[
    RuleTable {
        order: DimOrder::InnermostFirst,
        rules: GGUF,
    },
    RuleTable {
        order: DimOrder::OutermostFirst,
        rules: ST,
    },
];

fn hp() -> HyperParams {
    HyperParams::new(2, 64, 256)
        .with("n_heads", HParamValue::U64(4))
        .with("d_head", HParamValue::U64(16))
        .with("d_ffn", HParamValue::U64(96))
        .with(N_EXPERTS, HParamValue::U64(8))
        .with("v_hidden", HParamValue::U64(32))
}

fn matcher() -> Matcher {
    Matcher::new(TABLES).unwrap()
}

fn dense(layer: Option<u16>, role: TensorRole, block: BlockSel) -> Classification {
    Classification::Classified {
        class: TensorClass::DenseEveryToken,
        layer: layer.map(LayerIx),
        role,
        stacked_experts: None,
        block,
    }
}

#[test]
fn every_rule_classifies_its_tensor() {
    let m = matcher();
    let hp = hp();
    let gqa = BlockSel::Layer(Block::Gqa);
    let moe = BlockSel::Layer(Block::Moe);
    let cases: &[(&str, &[u64], Classification)] = &[
        (
            "token_embd.weight",
            &[64, 256],
            Classification::Classified {
                class: TensorClass::RowLookup,
                layer: None,
                role: TensorRole::Embed,
                stacked_experts: None,
                block: BlockSel::Prologue,
            },
        ),
        (
            "blk.1.attn_norm.weight",
            &[64],
            dense(Some(1), TensorRole::Norm, gqa),
        ),
        (
            "blk.0.attn_q.weight",
            &[64, 64],
            dense(Some(0), TensorRole::Q, gqa),
        ),
        (
            "blk.1.ffn_gate_inp.weight",
            &[64, 8],
            Classification::Classified {
                class: TensorClass::Router { layer: LayerIx(1) },
                layer: Some(LayerIx(1)),
                role: TensorRole::Router,
                stacked_experts: None,
                block: moe,
            },
        ),
        (
            "blk.1.ffn_gate_exps.weight",
            &[64, 96, 8],
            Classification::Classified {
                class: TensorClass::Expert {
                    layer: LayerIx(1),
                    id: ExpertIx(0),
                },
                layer: Some(LayerIx(1)),
                role: TensorRole::Gate,
                stacked_experts: Some(8),
                block: moe,
            },
        ),
        (
            "output.weight",
            &[64, 256],
            Classification::Classified {
                class: TensorClass::Head,
                layer: None,
                role: TensorRole::Head,
                stacked_experts: None,
                block: BlockSel::Epilogue,
            },
        ),
        (
            "blk.2.nextn.eh_proj.weight",
            &[128, 64],
            dense(Some(2), TensorRole::Proj, BlockSel::Mtp(0)),
        ),
        (
            "model.layers.1.self_attn.q_proj.weight",
            &[64, 64],
            dense(Some(1), TensorRole::Q, gqa),
        ),
        (
            "model.layers.0.mlp.experts.7.gate_proj.weight",
            &[96, 64],
            Classification::Classified {
                class: TensorClass::Expert {
                    layer: LayerIx(0),
                    id: ExpertIx(7),
                },
                layer: Some(LayerIx(0)),
                role: TensorRole::Gate,
                stacked_experts: None,
                block: moe,
            },
        ),
        (
            "model.layers.0.mlp.experts.gate_up_proj",
            &[8, 64, 192],
            Classification::Classified {
                class: TensorClass::Expert {
                    layer: LayerIx(0),
                    id: ExpertIx(0),
                },
                layer: Some(LayerIx(0)),
                role: TensorRole::GateUp,
                stacked_experts: Some(8),
                block: moe,
            },
        ),
        (
            "model.layers.0.self_attn.rotary_emb.inv_freq",
            &[8],
            Classification::Ignored {
                reason: "a recomputed buffer, not a weight",
            },
        ),
    ];
    for (name, shape, want) in cases {
        assert_eq!(m.classify(name, shape, &hp), *want, "{name}");
    }
}

#[test]
fn tower_tensors_are_ignored_with_the_post_1_0_reason() {
    let m = matcher();
    let hp = hp();
    let ignored = Classification::Ignored {
        reason: MULTIMODAL_REASON,
    };
    assert_eq!(m.classify("v.blk.0.attn_q.weight", &[32, 32], &hp), ignored);
    assert_eq!(
        m.classify("v.blk.31.attn_q.weight", &[32, 32], &hp),
        ignored
    );
    assert_eq!(
        m.classify("model.visual.blocks.5.attn.qkv.weight", &[96, 32], &hp),
        ignored
    );
    assert_eq!(
        m.classify(
            "model.multi_modal_projector.linear_2.weight",
            &[64, 32],
            &hp
        ),
        ignored
    );
    // Tower tensors are shape-checked like any other, and one that matches no rule is
    // `Unknown`.
    assert_eq!(
        m.classify("v.blk.0.attn_q.weight", &[32, 31], &hp),
        Classification::Unknown
    );
    assert_eq!(
        m.classify("model.visual.blocks.5.attn.proj.weight", &[32, 32], &hp),
        Classification::Unknown
    );
}

#[test]
fn a_matching_name_with_the_wrong_shape_is_unknown() {
    let m = matcher();
    let hp = hp();
    let cases: &[(&str, &[u64])] = &[
        // Dimensions swapped: the safetensors order on a GGUF name.
        ("blk.0.ffn_gate_inp.weight", &[8, 64]),
        // A wrong size.
        ("blk.0.attn_q.weight", &[64, 63]),
        // A wrong rank.
        ("blk.0.attn_norm.weight", &[64, 1]),
        ("blk.0.attn_q.weight", &[4096]),
        ("blk.0.attn_q.weight", &[]),
        // A stacked expert tensor without `n_experts` outermost.
        ("blk.0.ffn_gate_exps.weight", &[64, 96, 7]),
        ("model.layers.0.mlp.experts.gate_up_proj", &[64, 8, 192]),
        ("token_embd.weight", &[256, 64]),
    ];
    for (name, shape) in cases {
        assert_eq!(
            m.classify(name, shape, &hp),
            Classification::Unknown,
            "{name} {shape:?}"
        );
    }

    // Two rules whose patterns both match one name: the first decides, and when its checks
    // fail the tensor is `Unknown`. The later rule, which the shape (or the layer) would fit,
    // is never tried (RFC-0043 §4.1, "`Unknown`, not a guess").
    const FIRST_THEN_LOOSE: &[ClassRule] = &[
        rule(
            "blk.{L}.attn_q.weight",
            BlockSel::Layer(Block::Gqa),
            TensorRole::Q,
            ClassSel::DenseEveryToken,
            &[Hp("hidden"), Mul("n_heads", "d_head")],
        ),
        rule(
            "blk.{N}.attn_q.weight",
            BlockSel::Encoder,
            TensorRole::Q,
            TOWER,
            &[Any, Any],
        ),
    ];
    let both = Matcher::new(&[RuleTable {
        order: DimOrder::InnermostFirst,
        rules: FIRST_THEN_LOOSE,
    }])
    .unwrap();
    assert!(
        matches!(
            both.classify("blk.0.attn_q.weight", &[64, 64], &hp),
            Classification::Classified { .. }
        ),
        "the first rule classifies a tensor that fits it"
    );
    for (name, shape) in [
        ("blk.0.attn_q.weight", &[64u64, 63][..]),
        ("blk.0.attn_q.weight", &[64, 64, 1][..]),
        ("blk.7.attn_q.weight", &[64, 64][..]),
    ] {
        assert_eq!(
            both.classify(name, shape, &hp),
            Classification::Unknown,
            "{name} {shape:?} fell through to the later rule"
        );
    }
}

#[test]
fn indices_out_of_range_or_malformed_are_unknown() {
    let m = matcher();
    let hp = hp();
    let cases: &[(&str, &[u64])] = &[
        ("blk.2.attn_norm.weight", &[64]),
        ("blk.65536.attn_norm.weight", &[64]),
        ("blk.99999999999.attn_norm.weight", &[64]),
        ("blk.01.attn_norm.weight", &[64]),
        ("blk.-1.attn_norm.weight", &[64]),
        ("blk..attn_norm.weight", &[64]),
        ("blk.1.attn_norm.weight.extra", &[64]),
        ("xblk.1.attn_norm.weight", &[64]),
        ("model.layers.0.mlp.experts.8.gate_proj.weight", &[96, 64]),
        // The MTP rule's layer is `n_layers + 0`, and nothing else.
        ("blk.1.nextn.eh_proj.weight", &[128, 64]),
        ("blk.3.nextn.eh_proj.weight", &[128, 64]),
        ("unrelated", &[1]),
        ("", &[]),
    ];
    for (name, shape) in cases {
        assert_eq!(
            m.classify(name, shape, &hp),
            Classification::Unknown,
            "{name}"
        );
    }
}

#[test]
fn a_shape_naming_a_missing_hyperparameter_is_unknown() {
    let m = matcher();
    let hp = HyperParams::new(2, 64, 256);
    assert_eq!(
        m.classify("blk.0.attn_q.weight", &[64, 64], &hp),
        Classification::Unknown
    );
    assert_eq!(
        m.classify(
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            &[96, 64],
            &hp
        ),
        Classification::Unknown
    );
}

fn refused(r: ClassRule, order: DimOrder) -> String {
    let rules: &'static [ClassRule] = Box::leak(Box::new([r]));
    Matcher::new(&[RuleTable { order, rules }])
        .unwrap_err()
        .reason
}

#[test]
fn malformed_rules_are_refused() {
    let gqa = BlockSel::Layer(Block::Gqa);
    let moe = BlockSel::Layer(Block::Moe);
    let dense = ClassSel::DenseEveryToken;
    let q = TensorRole::Q;
    let inner = DimOrder::InnermostFirst;
    let cases: &[(ClassRule, &str)] = &[
        (rule("blk.{X}.q", gqa, q, dense, &[]), "unknown placeholder"),
        (rule("blk.{L", gqa, q, dense, &[]), "not closed"),
        (rule("blk.L}.q", gqa, q, dense, &[]), "has no `{`"),
        (rule("blk.{L}{N}.q", gqa, q, dense, &[]), "no text between"),
        (
            rule("blk.{L}0.q", gqa, q, dense, &[]),
            "followed by a digit",
        ),
        (rule("blk.{L}.{L}", gqa, q, dense, &[]), "appears twice"),
        (rule("blk.q", gqa, q, dense, &[]), "needs `{L}`"),
        (
            rule("blk.{L}.r", BlockSel::Prologue, q, ClassSel::Router, &[]),
            "has no `{L}`",
        ),
        (
            rule("embd.{L}", BlockSel::Prologue, q, dense, &[]),
            "has no `{L}`",
        ),
        (
            rule("blk.{L}.e", moe, q, ClassSel::Expert, &[]),
            "`{E}` is used exactly",
        ),
        (
            rule("blk.{L}.e.{E}", moe, q, dense, &[]),
            "`{E}` is used exactly",
        ),
        (
            rule("v.{N}", BlockSel::Encoder, q, dense, &[]),
            "goes with `BlockSel::Encoder`",
        ),
        (
            rule("v.{N}", BlockSel::Prologue, q, TOWER, &[]),
            "goes with `BlockSel::Encoder`",
        ),
        (
            rule("v.{L}", BlockSel::Encoder, q, TOWER, &[]),
            "is `{N}`, not `{L}`",
        ),
        (
            rule(
                "blk.{L}.x",
                moe,
                q,
                ClassSel::ExpertsStacked,
                &[Hp(N_EXPERTS), Hp("hidden")],
            ),
            "outermost dimension",
        ),
    ];
    for (r, want) in cases {
        let got = refused(*r, inner);
        assert!(got.contains(want), "{}: {got}", r.pattern);
    }
    // The same stacked shape is right in outermost-first order.
    let stacked = rule(
        "blk.{L}.x",
        moe,
        q,
        ClassSel::ExpertsStacked,
        &[Hp(N_EXPERTS), Hp("hidden")],
    );
    let rules: &'static [ClassRule] = Box::leak(Box::new([stacked]));
    assert!(
        Matcher::new(&[RuleTable {
            order: DimOrder::OutermostFirst,
            rules,
        }])
        .is_ok()
    );
    // A pattern listed twice, here across the two tables.
    let twice = [
        RuleTable {
            order: DimOrder::InnermostFirst,
            rules: &GGUF[..1],
        },
        RuleTable {
            order: DimOrder::OutermostFirst,
            rules: &GGUF[..1],
        },
    ];
    assert!(
        Matcher::new(&twice)
            .unwrap_err()
            .reason
            .contains("listed twice")
    );
}

#[test]
fn the_plan_lists_ignored_and_unknown_tensors_apart_from_the_fetch() {
    struct Rules(Matcher);
    impl crate::families::Family for Rules {
        fn name(&self) -> &'static str {
            "test"
        }
        fn accepts(&self) -> &'static [&'static str] {
            &[]
        }
        fn version(&self) -> u32 {
            1
        }
        fn parse_hparams(
            &self,
            _: &crate::families::MetaView<'_>,
        ) -> Result<HyperParams, crate::families::FamilyError> {
            Ok(hp())
        }
        fn classify(&self, name: &str, shape: &[u64], hp: &HyperParams) -> Classification {
            self.0.classify(name, shape, hp)
        }
        fn required_ops(&self, _: &HyperParams) -> crate::ir::OpSet {
            crate::ir::OpSet::new()
        }
    }
    let family = Rules(matcher());
    let tensors: &[(&str, &[u64])] = &[
        ("token_embd.weight", &[64, 256]),
        ("v.blk.0.attn_q.weight", &[32, 32]),
        ("blk.0.attn_q.weight", &[64, 64]),
        ("blk.0.attn_q.weight.bad", &[64, 64]),
        ("output.weight", &[64, 256]),
    ];
    let plan = classify_tensors(&family, &hp(), tensors.iter().copied());
    let fetch: Vec<&str> = plan.fetch.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        fetch,
        ["token_embd.weight", "blk.0.attn_q.weight", "output.weight"]
    );
    assert_eq!(
        plan.ignored,
        [("v.blk.0.attn_q.weight".to_owned(), MULTIMODAL_REASON)]
    );
    assert_eq!(plan.unknown, ["blk.0.attn_q.weight.bad"]);
}

proptest! {
    /// Any shape other than the rule's, at any matching name, is `Unknown`.
    #[test]
    fn any_other_shape_is_unknown(
        layer in 0u16..2,
        shape in proptest::collection::vec(0u64..300, 0..4),
    ) {
        let m = matcher();
        let hp = hp();
        let name = format!("blk.{layer}.attn_q.weight");
        let got = m.classify(&name, &shape, &hp);
        if shape == [64, 64] {
            let classified = matches!(got, Classification::Classified { .. });
            prop_assert!(classified);
        } else {
            prop_assert_eq!(got, Classification::Unknown);
        }
    }

    /// Any decoder layer at or past `n_layers` is `Unknown`.
    #[test]
    fn any_layer_past_the_last_is_unknown(layer in 2u64..1_000_000) {
        let name = format!("blk.{layer}.attn_norm.weight");
        prop_assert_eq!(matcher().classify(&name, &[64], &hp()), Classification::Unknown);
    }
}
