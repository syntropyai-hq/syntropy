// SPDX-License-Identifier: MIT OR Apache-2.0

//! The M0 blocks against the real families (COMP-E15-T13; RFC-0043 §3.1–§3.4).
//!
//! `graph`'s own block tests compose `nano` graphs from hand-written hyperparameters, with a
//! required op set built in the test (COMP-E15-T02). Here every input is the real one:
//!
//! - the checkpoints are the `Nano` fixtures QUAL-E7-T01's generator writes for every variant of
//!   the `llama` and `qwen_moe` `FamilySpec`s (COMP-E16-T03, COMP-E16-T04);
//! - the hyperparameters are the ones the family parsed from each fixture's `config.json`, and
//!   the census groups the ones `graph::census::build` gave its tensors (`Fixture::manifest`);
//! - each block's parameters come from the family's own functions (`llama::gqa_params`,
//!   `ffn_params`, `embed_params`, `head_params`, `qwen_moe::moe_params`);
//! - the op set checked is the family's `Family::required_ops` for those hyperparameters.
//!
//! The graph is composed the way a family's `build` will (RFC-0043 §5.1; `build` joins the trait
//! with a later task): `Embed`, then per layer the mixer and FFN blocks over that layer's census
//! groups, then `LmHead`. Each graph is built in both forms, since a family's required op set
//! covers decode and prefill (RFC-0040 §11.1).

use std::path::PathBuf;

use smallvec::smallvec;
use syntropy_core::{DType, LayerIx, TensorGroupId};
use syntropy_graph::blocks::{Builder, Form, RoutingKnown, dense_ffn, embed, gqa, lm_head, moe};
use syntropy_graph::families::{Family, llama, qwen_moe};
use syntropy_graph::ir::{
    Block, BoundarySet, CensusGroup, Dim, GroupClass, LayerKind, LayerProgram, ModelGraph, OpSet,
    Program, ResidualSpec, Rule, SideData, ValueKind, validate,
};
use syntropy_manifest::census::{ModelManifest, TensorClass};
use syntropy_testkit::fixtures::{Fixture, QuantClass, SizeClass, generate};

/// The generator's seed. Any seed gives the same structure; only the weights differ.
const SEED: u64 = 0x5EED;

/// A fresh directory under the system temp directory, removed when dropped.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!(
            "syntropy-family-blocks-{}-{tag}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Every `Nano` fixture of both M0 families, with the family module that reads it.
fn fixtures(dir: &TempDir) -> Vec<(&'static dyn Family, Fixture)> {
    let families: [(&'static dyn Family, _); 2] = [
        (&llama::LLAMA, llama::fixture_spec()),
        (&qwen_moe::QWEN_MOE, qwen_moe::fixture_spec()),
    ];
    let mut out = Vec::new();
    for (family, spec) in families {
        let made = generate(spec, SizeClass::Nano, SEED, &[QuantClass::F32], &dir.0)
            .unwrap_or_else(|e| panic!("{}: {e}", spec.family));
        assert!(!made.is_empty(), "{}: no fixture", spec.family);
        out.extend(made.into_iter().map(|f| (family, f)));
    }
    out
}

/// The census as the validator reads it: each group's class and layer.
fn census(m: &ModelManifest) -> Vec<CensusGroup> {
    m.census
        .groups
        .iter()
        .map(|g| CensusGroup {
            id: g.id,
            class: match g.class {
                TensorClass::DenseEveryToken => GroupClass::DenseEveryToken,
                TensorClass::Expert { .. } => GroupClass::Expert,
                TensorClass::RowLookup => GroupClass::RowLookup,
                TensorClass::Head => GroupClass::Head,
                TensorClass::Router { .. } => GroupClass::Router,
                TensorClass::Encoder { .. } => GroupClass::Encoder,
                other => panic!("no validator class for {other:?}"),
            },
            layer: g.layer,
        })
        .collect()
}

/// The one group of class `class` (at `layer`, if given) whose tensors' names contain `part`;
/// `""` matches any name. Panics unless exactly one group matches.
fn group(
    m: &ModelManifest,
    class: GroupClass,
    layer: Option<LayerIx>,
    part: &str,
) -> TensorGroupId {
    let groups = census(m);
    let hits: Vec<TensorGroupId> = m
        .census
        .groups
        .iter()
        .zip(&groups)
        .filter(|(g, c)| {
            c.class == class
                && c.layer == layer
                && g.tensors
                    .iter()
                    .any(|t| m.census.tensors[t.0 as usize].name.contains(part))
        })
        .map(|(g, _)| g.id)
        .collect();
    assert_eq!(hits.len(), 1, "{class:?} {layer:?} {part:?}: {hits:?}");
    hits[0]
}

/// A composed graph with the census it was composed over and each MoE layer's routing.
struct Composed {
    graph: ModelGraph,
    census: Vec<CensusGroup>,
    routing: Vec<(LayerIx, RoutingKnown)>,
}

/// Composes the fixture's graph in `form` from the family's block parameters and the fixture's
/// census groups.
fn compose(family: &dyn Family, f: &Fixture, form: Form) -> Composed {
    let m = &f.manifest;
    let hp = &m.hparams;
    let gqa_p = llama::gqa_params(hp).unwrap_or_else(|| panic!("{}: no Gqa params", f.variant));
    let moe_p = qwen_moe::moe_params(hp);
    let is_moe = family.name() == qwen_moe::TAG;
    assert_eq!(is_moe, moe_p.is_some(), "{}: MoE parameters", f.variant);

    let mut b = Builder::new();
    let tokens = b.value(DType::I32, &[Dim::Rows], ValueKind::TokenIds);
    let mut prologue = Program {
        inputs: smallvec![tokens],
        ..Program::default()
    };
    let table = group(m, GroupClass::RowLookup, None, "");
    let mut residual = embed(
        &mut b,
        &mut prologue,
        &llama::embed_params(hp),
        table,
        tokens,
    );
    b.set_kind(residual, ValueKind::Residual);
    prologue.outputs.push(residual);

    let mut layers = Vec::new();
    let mut routing = Vec::new();
    for i in 0..hp.n_layers {
        let ix = LayerIx(i);
        let mut body = Program {
            inputs: smallvec![residual],
            ..Program::default()
        };
        let mixer = group(m, GroupClass::DenseEveryToken, Some(ix), "self_attn");
        let h = gqa(&mut b, &mut body, &gqa_p, form, mixer, residual);
        let (out, ffn) = match &moe_p {
            None => {
                let ffn = group(m, GroupClass::DenseEveryToken, Some(ix), "mlp");
                let out = dense_ffn(&mut b, &mut body, &llama::ffn_params(hp), form, ffn, h);
                (out, Block::DenseFfn)
            }
            Some(p) => {
                let router = group(m, GroupClass::Router, Some(ix), "");
                let (out, known) = moe(&mut b, &mut body, p, form, router, ix, h);
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
    let head = group(m, GroupClass::Head, None, "");
    let logits = lm_head(
        &mut b,
        &mut epilogue,
        &llama::head_params(hp),
        head,
        residual,
    );
    epilogue.outputs.push(logits);

    let n = layers.len();
    Composed {
        graph: ModelGraph {
            family: family.name(),
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
            // Only the residual stream crosses a layer boundary, so every boundary is legal.
            boundaries: BoundarySet {
                legal: vec![true; n + 1],
                side: vec![SideData::default(); n + 1],
            },
        },
        census: census(m),
        routing,
    }
}

/// Every op the graph uses.
fn used_ops(g: &ModelGraph) -> OpSet {
    std::iter::once(&g.prologue)
        .chain(g.layers.iter().map(|l| &l.body))
        .chain(std::iter::once(&g.epilogue))
        .flat_map(|p| p.nodes.iter().map(|n| n.op))
        .collect()
}

/// Criterion 1: the `llama` and `qwen_moe` `nano` graphs validate against the generated
/// checkpoints' census and their family's `required_ops`, in both forms, for every variant.
#[test]
fn the_nano_graphs_of_the_real_families_validate() {
    let dir = TempDir::new("validate");
    let all = fixtures(&dir);
    let mut seen: Vec<(&str, &str)> = Vec::new();
    for (family, f) in &all {
        let required = family.required_ops(&f.manifest.hparams);
        for form in Form::ALL {
            let c = compose(*family, f, form);
            assert_eq!(
                validate(&c.graph, &c.census, &required),
                Ok(()),
                "{} {} {form:?}",
                f.family,
                f.variant
            );
            assert_eq!(
                c.graph.layers.len() as u64,
                u64::from(f.manifest.hparams.n_layers)
            );
        }
        seen.push((f.family, f.variant));
    }
    // Every variant of both specs was composed: `llama`'s four RoPE and attention forms and
    // `qwen_moe`'s one.
    let specs = [llama::fixture_spec(), qwen_moe::fixture_spec()];
    let want: Vec<(&str, &str)> = specs
        .iter()
        .flat_map(|s| s.variants.iter().map(move |v| (s.family, v.name)))
        .collect();
    assert_eq!(seen, want);
}

/// Criterion 2: the ops each graph uses are a subset of its family's `required_ops`. The two
/// forms together use every required op, so `required_ops` lists no op the graphs never emit.
#[test]
fn the_ops_used_are_a_subset_of_the_familys_required_ops() {
    let dir = TempDir::new("ops");
    for (family, f) in fixtures(&dir) {
        let required = family.required_ops(&f.manifest.hparams);
        let mut both = OpSet::new();
        for form in Form::ALL {
            let used = used_ops(&compose(family, &f, form).graph);
            let extra: Vec<_> = used.difference(&required).collect();
            assert!(
                extra.is_empty(),
                "{} {} {form:?} uses ops outside required_ops: {extra:?}",
                f.family,
                f.variant
            );
            both.extend(used);
        }
        assert_eq!(both, required, "{} {}", f.family, f.variant);
    }
}

/// RFC-0043 §3.4: every `qwen_moe` layer's routing is known at its router, and `llama` has no
/// MoE layer.
#[test]
fn moe_layers_of_the_real_family_are_routing_known_at_router() {
    let dir = TempDir::new("routing");
    for (family, f) in fixtures(&dir) {
        let c = compose(family, &f, Form::Decode);
        if family.name() == qwen_moe::TAG {
            let want: Vec<_> = (0..f.manifest.hparams.n_layers)
                .map(|l| (LayerIx(l), RoutingKnown::AtRouter))
                .collect();
            assert_eq!(c.routing, want, "{}", f.variant);
        } else {
            assert!(c.routing.is_empty(), "{}", f.variant);
        }
    }
}

/// The checks above fail when they should: a graph composed over another family's op set, or
/// with a layer's weights pointed at the wrong census group, does not validate.
#[test]
fn a_wrong_op_set_or_group_fails_validation() {
    let dir = TempDir::new("negative");
    let all = fixtures(&dir);
    let (dense, d) = all
        .iter()
        .find(|(fam, _)| fam.name() == llama::TAG)
        .unwrap();
    let (moe_family, q) = all
        .iter()
        .find(|(fam, _)| fam.name() == qwen_moe::TAG)
        .unwrap();
    // The MoE graph against the dense family's op set: rule 5 (coverage) fails.
    let c = compose(*moe_family, q, Form::Decode);
    let dense_ops = dense.required_ops(&d.manifest.hparams);
    let e = validate(&c.graph, &c.census, &dense_ops).unwrap_err();
    assert_eq!(e.rule(), Rule::Coverage, "{e}");
    // The dense graph with layer 0's mixer pointed at the head group: rule 3 (weights) fails.
    let mut c = compose(*dense, d, Form::Decode);
    let head = group(&d.manifest, GroupClass::Head, None, "");
    for node in &mut c.graph.layers[0].body.nodes {
        for w in &mut node.weights {
            if let syntropy_graph::ir::GroupSel::Fixed(g) = &mut w.group {
                *g = head;
            }
        }
    }
    let required = dense.required_ops(&d.manifest.hparams);
    let e = validate(&c.graph, &c.census, &required).unwrap_err();
    assert_eq!(e.rule(), Rule::Weights, "{e}");
}
