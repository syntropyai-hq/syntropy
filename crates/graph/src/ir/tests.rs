// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the IR types and `ir::validate` against RFC-0043 §2.1, §2.2 and §6.1.
//!
//! A generator builds valid graphs from random shapes: 1–5 layers, each with a GQA mixer and a
//! dense or MoE FFN, one or two residual streams, optional n-gram and per-layer side data,
//! up to two MTP modules sharing the LM head, and random interior boundaries. Each mutator
//! then breaks exactly one rule, and the tests check that `validate` names that rule and
//! that every other rule still passes.

use proptest::prelude::*;
use proptest::sample::Index;
use smallvec::{SmallVec, smallvec};
use syntropy_core::{DType, LayerIx, TensorGroupId};
use syntropy_kernels::{AttnKind, GatedAct, OpKind, RouteNorm};

use super::*;

const HIDDEN: u32 = 64;
const VOCAB: u32 = 128;
const TOP_K: u32 = 2;

// ---- The generator ------------------------------------------------------------------------

/// The shape of one generated graph.
#[derive(Clone, Debug)]
pub(crate) struct Params {
    /// Per layer: MoE FFN (else dense), and whether it adds n-gram rows by token ID.
    pub(crate) layers: Vec<(bool, bool)>,
    /// One or two residual streams.
    pub(crate) streams: u8,
    /// How many MTP modules.
    pub(crate) mtp: u8,
    /// The layer that reads a per-layer input from the prologue, if any.
    pub(crate) per_layer: Option<usize>,
    /// Bit `b − 1` marks interior boundary `b` legal.
    pub(crate) legal: u8,
}

pub(crate) fn params() -> impl Strategy<Value = Params> {
    (
        prop::collection::vec((any::<bool>(), any::<bool>()), 1..=5),
        1u8..=2,
        0u8..=2,
        any::<Option<Index>>(),
        any::<u8>(),
    )
        .prop_map(|(layers, streams, mtp, per_layer, legal)| {
            let per_layer = per_layer.map(|i| i.index(layers.len()));
            Params {
                layers,
                streams,
                mtp,
                per_layer,
                legal,
            }
        })
}

/// What the generator records about one layer, for the mutators.
#[derive(Clone, Debug)]
pub(crate) struct LayerInfo {
    pub(crate) mixer: TensorGroupId,
    pub(crate) qkv_node: usize,
    pub(crate) o_node: usize,
    pub(crate) o: ValueId,
    pub(crate) att: ValueId,
    pub(crate) moe_node: Option<usize>,
    /// The router group of a MoE layer.
    pub(crate) router: Option<TensorGroupId>,
    /// The pre-FFN norm.
    pub(crate) ffn_norm_node: usize,
    /// The `KvWrite` node.
    pub(crate) kv_node: usize,
    /// The final `Add`, whose output is residual stream 0.
    pub(crate) last_node: usize,
    pub(crate) out: ValueId,
}

/// A generated graph with its census and op set, and handles for the mutators.
#[derive(Clone, Debug)]
pub(crate) struct Fixture {
    pub(crate) graph: ModelGraph,
    pub(crate) census: Vec<CensusGroup>,
    pub(crate) ops: OpSet,
    pub(crate) tok: ValueId,
    pub(crate) embed: TensorGroupId,
    pub(crate) head: TensorGroupId,
    pub(crate) layers: Vec<LayerInfo>,
}

impl Fixture {
    pub(crate) fn validate(&self) -> Result<(), Violation> {
        validate(&self.graph, &self.census, &self.ops)
    }

    fn check(&self, rule: Rule) -> Result<(), Violation> {
        check(rule, &self.graph, &self.census, &self.ops)
    }

    fn layer(&mut self, i: usize) -> &mut Program {
        &mut self.graph.layers[i].body
    }
}

/// Builds values, groups and nodes, and collects the ops it uses.
#[derive(Default)]
struct Builder {
    values: Vec<Value>,
    census: Vec<CensusGroup>,
    ops: OpSet,
}

fn hidden() -> SmallVec<[Dim; 4]> {
    smallvec![Dim::Rows, Dim::Const(HIDDEN)]
}

impl Builder {
    fn value(&mut self, dtype: DType, dims: SmallVec<[Dim; 4]>, kind: ValueKind) -> ValueId {
        self.values.push(Value {
            ty: ValueType { dtype, dims },
            kind,
        });
        ValueId(u32::try_from(self.values.len() - 1).unwrap())
    }

    fn act(&mut self, kind: ValueKind) -> ValueId {
        self.value(DType::F32, hidden(), kind)
    }

    fn group(&mut self, class: GroupClass, layer: Option<u16>) -> TensorGroupId {
        let id = TensorGroupId(u32::try_from(self.census.len()).unwrap() + 100);
        self.census.push(CensusGroup {
            id,
            class,
            layer: layer.map(LayerIx),
        });
        id
    }

    /// Appends a node to `p` and returns its position.
    fn node(
        &mut self,
        p: &mut Program,
        op: OpKind,
        inputs: &[ValueId],
        outputs: &[ValueId],
        weights: &[(GroupSel, TensorRole)],
        class: NodeClass,
    ) -> usize {
        self.ops.insert(op);
        p.nodes.push(Node {
            op,
            inputs: SmallVec::from_slice(inputs),
            outputs: SmallVec::from_slice(outputs),
            weights: weights
                .iter()
                .map(|&(group, role)| WeightRef { group, role })
                .collect(),
            attrs: OpAttrs::default(),
            class,
            fuse: FuseHints::default(),
        });
        p.nodes.len() - 1
    }
}

fn fixed(id: TensorGroupId, role: TensorRole) -> (GroupSel, TensorRole) {
    (GroupSel::Fixed(id), role)
}

pub(crate) fn build(p: &Params) -> Fixture {
    let n = p.layers.len();
    let mut b = Builder::default();
    let tok = b.value(DType::I32, smallvec![Dim::Rows], ValueKind::TokenIds);

    // Prologue: token embedding, the residual streams and the optional per-layer input.
    let mut prologue = Program {
        inputs: smallvec![tok],
        ..Program::default()
    };
    let embed = b.group(GroupClass::RowLookup, None);
    let one = p.streams == 1;
    let e = b.act(if one {
        ValueKind::Residual
    } else {
        ValueKind::Local
    });
    b.node(
        &mut prologue,
        OpKind::RowGather,
        &[tok],
        &[e],
        &[fixed(embed, TensorRole::Embed)],
        NodeClass::Lookup,
    );
    let mut streams: SmallVec<[ValueId; 2]> = smallvec![e];
    if !one {
        let s0 = b.act(ValueKind::Residual);
        let s1 = b.act(ValueKind::Residual);
        b.node(
            &mut prologue,
            OpKind::HyperMix,
            &[e],
            &[s0, s1],
            &[],
            NodeClass::Elementwise,
        );
        streams = smallvec![s0, s1];
    }
    prologue.outputs.extend(streams.iter().copied());
    let per_layer = p.per_layer.map(|_| {
        let table = b.group(GroupClass::RowLookup, None);
        let pl = b.act(ValueKind::LayerInput);
        b.node(
            &mut prologue,
            OpKind::RowGather,
            &[tok],
            &[pl],
            &[fixed(table, TensorRole::Embed)],
            NodeClass::Lookup,
        );
        prologue.outputs.push(pl);
        pl
    });

    // Layers.
    let mut layers = Vec::with_capacity(n);
    let mut infos = Vec::with_capacity(n);
    for (i, &(moe, ngram)) in p.layers.iter().enumerate() {
        let ix = u16::try_from(i).unwrap();
        let mut body = Program {
            inputs: streams.iter().copied().collect(),
            ..Program::default()
        };
        let mixer = b.group(GroupClass::DenseEveryToken, Some(ix));
        let mut blocks: SmallVec<[Block; 3]> = SmallVec::new();
        let (x, carry) = if one {
            (streams[0], None)
        } else {
            blocks.push(Block::HyperConn);
            let h0 = b.act(ValueKind::Local);
            let h1 = b.act(ValueKind::Residual);
            b.node(
                &mut body,
                OpKind::HyperMix,
                &[streams[0], streams[1]],
                &[h0, h1],
                &[fixed(mixer, TensorRole::Proj)],
                NodeClass::Elementwise,
            );
            (h0, Some(h1))
        };
        blocks.push(Block::Gqa);
        let n1 = b.act(ValueKind::Local);
        b.node(
            &mut body,
            OpKind::RmsNorm,
            &[x],
            &[n1],
            &[fixed(mixer, TensorRole::Norm)],
            NodeClass::Elementwise,
        );
        let qkv = b.value(
            DType::F32,
            smallvec![Dim::Rows, Dim::Const(3 * HIDDEN)],
            ValueKind::Local,
        );
        let qkv_node = b.node(
            &mut body,
            OpKind::Gemv,
            &[n1],
            &[qkv],
            &[fixed(mixer, TensorRole::Qkv)],
            NodeClass::Dense,
        );
        let kv_node = b.node(
            &mut body,
            OpKind::KvWrite(AttnKind::Gqa),
            &[qkv],
            &[],
            &[],
            NodeClass::State,
        );
        let att = b.act(ValueKind::Local);
        b.node(
            &mut body,
            OpKind::AttnDecode(AttnKind::Gqa),
            &[qkv],
            &[att],
            &[],
            NodeClass::State,
        );
        let o = b.act(ValueKind::Local);
        let o_node = b.node(
            &mut body,
            OpKind::Gemv,
            &[att],
            &[o],
            &[fixed(mixer, TensorRole::O)],
            NodeClass::Dense,
        );
        let mut a = b.act(ValueKind::Local);
        b.node(
            &mut body,
            OpKind::Add,
            &[x, o],
            &[a],
            &[],
            NodeClass::Elementwise,
        );
        if ngram {
            blocks.push(Block::NgramEmbed);
            body.inputs.push(tok);
            let table = b.group(GroupClass::RowLookup, Some(ix));
            let g = b.act(ValueKind::Local);
            b.node(
                &mut body,
                OpKind::RowGather,
                &[tok],
                &[g],
                &[fixed(table, TensorRole::Embed)],
                NodeClass::Lookup,
            );
            let sum = b.act(ValueKind::Local);
            b.node(
                &mut body,
                OpKind::Add,
                &[a, g],
                &[sum],
                &[],
                NodeClass::Elementwise,
            );
            a = sum;
        }
        if p.per_layer == Some(i) {
            let pl = per_layer.unwrap();
            body.inputs.push(pl);
            let sum = b.act(ValueKind::Local);
            b.node(
                &mut body,
                OpKind::Add,
                &[a, pl],
                &[sum],
                &[],
                NodeClass::Elementwise,
            );
            a = sum;
        }
        let n2 = b.act(ValueKind::Local);
        let mut router_group = None;
        let ffn_norm_node;
        let (ffn_out, moe_node, ffn) = if moe {
            // RFC-0041 §3.1: in a MoE layer the pre-FFN norm joins the router group.
            let router = b.group(GroupClass::Router, Some(ix));
            router_group = Some(router);
            ffn_norm_node = b.node(
                &mut body,
                OpKind::RmsNorm,
                &[a],
                &[n2],
                &[fixed(router, TensorRole::Norm)],
                NodeClass::Elementwise,
            );
            for _ in 0..3 {
                b.group(GroupClass::Expert, Some(ix));
            }
            let ids = b.value(
                DType::I32,
                smallvec![Dim::Rows, Dim::Const(TOP_K)],
                ValueKind::Local,
            );
            let w = b.value(
                DType::F32,
                smallvec![Dim::Rows, Dim::Const(TOP_K)],
                ValueKind::Local,
            );
            b.node(
                &mut body,
                OpKind::MoeRoute {
                    norm: RouteNorm::Softmax,
                },
                &[n2],
                &[ids, w],
                &[fixed(router, TensorRole::Router)],
                NodeClass::Router,
            );
            let eo = b.value(
                DType::F32,
                smallvec![Dim::Rows, Dim::Const(TOP_K), Dim::Const(HIDDEN)],
                ValueKind::Local,
            );
            let moe_node = b.node(
                &mut body,
                OpKind::MoeGrouped,
                &[n2, ids],
                &[eo],
                &[(GroupSel::Experts { layer: LayerIx(ix) }, TensorRole::GateUp)],
                NodeClass::Expert,
            );
            let c = b.act(ValueKind::Local);
            b.node(
                &mut body,
                OpKind::MoeCombine,
                &[eo, w],
                &[c],
                &[],
                NodeClass::Elementwise,
            );
            (c, Some(moe_node), Block::Moe)
        } else {
            let dense = b.group(GroupClass::DenseEveryToken, Some(ix));
            ffn_norm_node = b.node(
                &mut body,
                OpKind::RmsNorm,
                &[a],
                &[n2],
                &[fixed(dense, TensorRole::Norm)],
                NodeClass::Elementwise,
            );
            let gu = b.value(
                DType::F32,
                smallvec![Dim::Rows, Dim::Const(4 * HIDDEN)],
                ValueKind::Local,
            );
            b.node(
                &mut body,
                OpKind::GateUpAct {
                    act: GatedAct::Silu,
                },
                &[n2],
                &[gu],
                &[fixed(dense, TensorRole::GateUp)],
                NodeClass::Dense,
            );
            let d = b.act(ValueKind::Local);
            b.node(
                &mut body,
                OpKind::Gemv,
                &[gu],
                &[d],
                &[fixed(dense, TensorRole::Down)],
                NodeClass::Dense,
            );
            (d, None, Block::DenseFfn)
        };
        blocks.push(ffn);
        let out = b.act(ValueKind::Residual);
        let last_node = b.node(
            &mut body,
            OpKind::Add,
            &[a, ffn_out],
            &[out],
            &[],
            NodeClass::Elementwise,
        );
        streams = match carry {
            None => smallvec![out],
            Some(h1) => smallvec![out, h1],
        };
        body.outputs.extend(streams.iter().copied());
        layers.push(LayerProgram {
            layer: LayerIx(ix),
            kind: LayerKind {
                mixer: Block::Gqa,
                ffn,
            },
            blocks,
            body,
        });
        infos.push(LayerInfo {
            mixer,
            qkv_node,
            o_node,
            o,
            att,
            moe_node,
            router: router_group,
            ffn_norm_node,
            kv_node,
            last_node,
            out,
        });
    }

    // Epilogue: the final norm, which joins the head group (RFC-0041 §3.1), the LM head,
    // and greedy next-token IDs for the MTP modules.
    let head = b.group(GroupClass::Head, None);
    let mut epilogue = Program {
        inputs: smallvec![streams[0]],
        ..Program::default()
    };
    let normed = b.act(ValueKind::Local);
    b.node(
        &mut epilogue,
        OpKind::RmsNorm,
        &[streams[0]],
        &[normed],
        &[fixed(head, TensorRole::Norm)],
        NodeClass::Elementwise,
    );
    let logits_ty: SmallVec<[Dim; 4]> = smallvec![Dim::Rows, Dim::Const(VOCAB)];
    let lg = b.value(DType::F32, logits_ty.clone(), ValueKind::Local);
    b.node(
        &mut epilogue,
        OpKind::Logits,
        &[normed],
        &[lg],
        &[fixed(head, TensorRole::Head)],
        NodeClass::Head,
    );
    epilogue.outputs.push(lg);

    // MTP modules: own embedding and dense groups, the shared head.
    let mut mtp = Vec::new();
    if p.mtp > 0 {
        let next = b.value(DType::I32, smallvec![Dim::Rows], ValueKind::Local);
        b.node(
            &mut epilogue,
            OpKind::Argmax,
            &[lg],
            &[next],
            &[],
            NodeClass::Elementwise,
        );
        epilogue.outputs.push(next);
        for j in 0..p.mtp {
            let ix = u16::try_from(n).unwrap() + u16::from(j);
            let own_embed = b.group(GroupClass::RowLookup, Some(ix));
            let dense = b.group(GroupClass::DenseEveryToken, Some(ix));
            let mut body = Program {
                inputs: smallvec![streams[0], next],
                ..Program::default()
            };
            let e = b.act(ValueKind::Local);
            b.node(
                &mut body,
                OpKind::RowGather,
                &[next],
                &[e],
                &[fixed(own_embed, TensorRole::Embed)],
                NodeClass::Lookup,
            );
            let m = b.act(ValueKind::Local);
            b.node(
                &mut body,
                OpKind::Add,
                &[streams[0], e],
                &[m],
                &[],
                NodeClass::Elementwise,
            );
            let mn = b.act(ValueKind::Local);
            b.node(
                &mut body,
                OpKind::RmsNorm,
                &[m],
                &[mn],
                &[fixed(dense, TensorRole::Norm)],
                NodeClass::Elementwise,
            );
            let x = b.act(ValueKind::Local);
            b.node(
                &mut body,
                OpKind::Gemv,
                &[mn],
                &[x],
                &[fixed(dense, TensorRole::Proj)],
                NodeClass::Dense,
            );
            let lg2 = b.value(DType::F32, logits_ty.clone(), ValueKind::Local);
            b.node(
                &mut body,
                OpKind::Logits,
                &[x],
                &[lg2],
                &[fixed(head, TensorRole::Head)],
                NodeClass::Head,
            );
            body.outputs.push(lg2);
            mtp.push(LayerProgram {
                layer: LayerIx(ix),
                kind: LayerKind {
                    mixer: Block::Gqa,
                    ffn: Block::DenseFfn,
                },
                blocks: smallvec![Block::Mtp],
                body,
            });
        }
    }

    // Boundaries: token IDs cross every boundary up to the last n-gram layer, and the
    // per-layer input every boundary up to its layer.
    let last_ngram = p.layers.iter().rposition(|&(_, ngram)| ngram);
    let legal = (0..=n)
        .map(|bd| bd == 0 || bd == n || p.legal & (1 << ((bd - 1) % 8)) != 0)
        .collect();
    let side = (0..=n)
        .map(|bd| {
            let token_ids = last_ngram.is_some_and(|k| bd <= k);
            let layer_inputs = if p.per_layer.is_some_and(|k| bd <= k) {
                HIDDEN
            } else {
                0
            };
            SideData {
                token_ids,
                layer_inputs,
                bytes_per_token: u32::from(token_ids) * 4 + layer_inputs * 4,
            }
        })
        .collect();

    Fixture {
        graph: ModelGraph {
            family: "test",
            values: b.values,
            prologue,
            layers,
            epilogue,
            mtp,
            encoders: Vec::new(),
            residual: ResidualSpec {
                streams: p.streams,
                dtype: DType::F32,
            },
            boundaries: BoundarySet { legal, side },
        },
        census: b.census,
        ops: b.ops,
        tok,
        embed,
        head,
        layers: infos,
    }
}

// ---- The mutators -------------------------------------------------------------------------

/// A mutator breaks exactly one rule: it returns the rule and the name of the violation it
/// expects, or `None` when the graph lacks what it needs (no MoE layer, say).
type Mutator = fn(&mut Fixture, usize) -> Option<(Rule, &'static str)>;

const MUTATORS: &[(&str, Mutator)] = &[
    // Rule 1
    ("read a later node's output", |f, l| {
        let out = f.layers[l].out;
        f.layer(l).nodes[0].inputs[0] = out;
        Some((Rule::Ssa, "UseBeforeDef"))
    }),
    ("define a value twice", |f, l| {
        let (o_node, att) = (f.layers[l].o_node, f.layers[l].att);
        f.layer(l).nodes[o_node].outputs[0] = att;
        Some((Rule::Ssa, "DefinedTwice"))
    }),
    ("read a value past the table", |f, l| {
        let past = ValueId(u32::try_from(f.graph.values.len()).unwrap() + 7);
        f.layer(l).nodes[0].inputs[0] = past;
        Some((Rule::Ssa, "UnknownValue"))
    }),
    ("take an input the program defines later", |f, l| {
        let out = f.layers[l].out;
        f.layer(l).inputs.push(out);
        Some((Rule::Ssa, "InputNotDefined"))
    }),
    ("output a value never defined in the program", |f, l| {
        let att = f.layers[l].att;
        f.graph.epilogue.outputs.push(att);
        Some((Rule::Ssa, "UseBeforeDef"))
    }),
    // Rule 2
    ("drop an input of Add", |f, l| {
        let last = f.layers[l].last_node;
        f.layer(l).nodes[last].inputs.pop();
        Some((Rule::Types, "Arity"))
    }),
    ("give KvWrite an output", |f, l| {
        let v = push_value(f, DType::F32, hidden(), ValueKind::Local);
        let kv = f.layers[l].kv_node;
        f.layer(l).nodes[kv].outputs.push(v);
        Some((Rule::Types, "Arity"))
    }),
    ("make token IDs float", |f, _| {
        f.graph.values[f.tok.0 as usize].ty.dtype = DType::F32;
        Some((Rule::Types, "Dtype"))
    }),
    ("change the dims of an Add input", |f, l| {
        let o = f.layers[l].o;
        f.graph.values[o.0 as usize].ty.dims = smallvec![Dim::Rows, Dim::Const(HIDDEN + 1)];
        Some((Rule::Types, "Dims"))
    }),
    // Rule 3
    ("read the head from a dense node", |f, l| {
        let (node, head) = (f.layers[l].qkv_node, f.head);
        f.layer(l).nodes[node].weights[0].group = GroupSel::Fixed(head);
        Some((Rule::Weights, "ClassMismatch"))
    }),
    ("read a group the census lacks", |f, l| {
        let node = f.layers[l].qkv_node;
        f.layer(l).nodes[node].weights[0].group = GroupSel::Fixed(TensorGroupId(9999));
        Some((Rule::Weights, "UnknownGroup"))
    }),
    ("add a census group nobody reads", |f, l| {
        f.census.push(CensusGroup {
            id: TensorGroupId(5000),
            class: GroupClass::DenseEveryToken,
            layer: Some(LayerIx(u16::try_from(l).unwrap())),
        });
        Some((Rule::Weights, "GroupUnread"))
    }),
    ("list a census group twice", |f, _| {
        let first = f.census[0];
        f.census.push(first);
        Some((Rule::Weights, "DuplicateGroup"))
    }),
    ("read another layer's mixer group", |f, l| {
        let n = f.layers.len();
        if n < 2 {
            return None;
        }
        let other = f.layers[(l + 1) % n].mixer;
        let node = f.layers[l].qkv_node;
        f.layer(l).nodes[node].weights[0].group = GroupSel::Fixed(other);
        Some((Rule::Weights, "GroupOfOtherLayer"))
    }),
    ("move a router group to another layer", |f, l| {
        let router = f.layers[l].router?;
        let entry = f.census.iter_mut().find(|c| c.id == router).unwrap();
        entry.layer = Some(LayerIx(u16::try_from(l).unwrap() + 7));
        Some((Rule::Weights, "GroupOfOtherLayer"))
    }),
    ("read the embedding table from a norm", |f, l| {
        let (node, embed) = (f.layers[l].ffn_norm_node, f.embed);
        f.layer(l).nodes[node].weights[0].group = GroupSel::Fixed(embed);
        Some((Rule::Weights, "ClassMismatch"))
    }),
    ("read another layer's experts", |f, l| {
        let node = f.layers[l].moe_node?;
        f.layer(l).nodes[node].weights[0].group = GroupSel::Experts {
            layer: LayerIx(999),
        };
        Some((Rule::Weights, "ExpertsOfOtherLayer"))
    }),
    (
        "read the prologue's embedding from an MTP module",
        |f, _| {
            let embed = f.embed;
            // The module keeps reading its own group, so only the cross-program read is wrong.
            let mtp = f.graph.mtp.first_mut()?;
            mtp.body.nodes[0].weights.push(WeightRef {
                group: GroupSel::Fixed(embed),
                role: TensorRole::Embed,
            });
            Some((Rule::Weights, "GroupReadAcrossLayers"))
        },
    ),
    // Rule 4
    ("make boundary 0 illegal", |f, _| {
        f.graph.boundaries.legal[0] = false;
        Some((Rule::Boundaries, "EndBoundaryIllegal"))
    }),
    ("make boundary n_layers illegal", |f, _| {
        let n = f.layers.len();
        f.graph.boundaries.legal[n] = false;
        Some((Rule::Boundaries, "EndBoundaryIllegal"))
    }),
    ("drop a side entry", |f, _| {
        f.graph.boundaries.side.pop();
        Some((Rule::Boundaries, "BoundarySetSize"))
    }),
    ("misnumber a layer", |f, l| {
        f.graph.layers[l].layer = LayerIx(u16::try_from(l).unwrap() + 100);
        Some((Rule::Boundaries, "LayerIndex"))
    }),
    ("misnumber an MTP module", |f, _| {
        f.graph.mtp.first_mut()?.layer = LayerIx(0);
        Some((Rule::Boundaries, "LayerIndex"))
    }),
    ("carry a local value into the epilogue", |f, l| {
        let o = f.layers[l].o;
        f.graph.epilogue.inputs.push(o);
        Some((Rule::Boundaries, "ValueCrossesBoundary"))
    }),
    ("make the last residual F16", |f, _| {
        let out = f.layers.last().unwrap().out;
        f.graph.values[out.0 as usize].ty.dtype = DType::F16;
        Some((Rule::Boundaries, "ValueCrossesBoundary"))
    }),
    ("make the last residual local", |f, _| {
        let out = f.layers.last().unwrap().out;
        f.graph.values[out.0 as usize].kind = ValueKind::Local;
        Some((Rule::Boundaries, "ValueCrossesBoundary"))
    }),
    ("undeclare token IDs at boundary 0", |f, _| {
        let side = &mut f.graph.boundaries.side[0];
        if !side.token_ids {
            return None;
        }
        side.token_ids = false;
        Some((Rule::Boundaries, "SideDataUndeclared"))
    }),
    ("undeclare per-layer inputs at boundary 0", |f, _| {
        let side = &mut f.graph.boundaries.side[0];
        if side.layer_inputs == 0 {
            return None;
        }
        side.layer_inputs = 0;
        Some((Rule::Boundaries, "SideDataUndeclared"))
    }),
    // Rule 5
    ("drop a used op from required_ops", |f, l| {
        let op = f.graph.layers[l].body.nodes[0].op;
        f.ops.remove(&op);
        Some((Rule::Coverage, "OpNotRequired"))
    }),
    // Rule 6
    ("give the graph an encoder program", |f, _| {
        f.graph.encoders.push(crate::ir::EncoderProgram {
            modality: crate::ir::Modality::Image,
            body: Program {
                inputs: SmallVec::new(),
                nodes: Vec::new(),
                outputs: SmallVec::new(),
            },
            output: ValueType {
                dtype: DType::F32,
                dims: SmallVec::new(),
            },
        });
        Some((Rule::ReservedHooks, "EncoderProgram"))
    }),
    // EmbedInject's signature is two floats in, one out, with no shape link between them, so
    // it admits every Add node's values and rule 2 still passes.
    ("turn an Add into EmbedInject", |f, l| {
        let node = f.graph.layers[l]
            .body
            .nodes
            .iter_mut()
            .find(|n| n.op == OpKind::Add)?;
        node.op = OpKind::EmbedInject;
        f.ops.insert(OpKind::EmbedInject);
        Some((Rule::ReservedHooks, "EmbedInject"))
    }),
    (
        "mark a node that reads no weight as an encoder node",
        |f, l| {
            let node = f.graph.layers[l]
                .body
                .nodes
                .iter_mut()
                .find(|n| n.weights.is_empty())?;
            node.class = NodeClass::Encoder;
            Some((Rule::ReservedHooks, "EncoderNode"))
        },
    ),
];

fn push_value(f: &mut Fixture, dtype: DType, dims: SmallVec<[Dim; 4]>, kind: ValueKind) -> ValueId {
    f.graph.values.push(Value {
        ty: ValueType { dtype, dims },
        kind,
    });
    ValueId(u32::try_from(f.graph.values.len() - 1).unwrap())
}

/// The variant name of a violation, from its `Debug` form.
fn variant(v: &Violation) -> String {
    let debug = format!("{v:?}");
    debug
        .split(|c: char| !c.is_alphanumeric())
        .next()
        .unwrap_or_default()
        .to_owned()
}

/// Every rule other than `rule` passes.
fn only_rule_broken(f: &Fixture, rule: Rule) -> Result<(), TestCaseError> {
    for other in Rule::ALL {
        if other != rule {
            prop_assert_eq!(f.check(other), Ok(()), "rule {:?} broke too", other);
        }
    }
    Ok(())
}

/// `f` breaks `rule` with an `expected` violation, and no other rule.
fn assert_breaks(f: &Fixture, name: &str, rule: Rule, expected: &str) -> Result<(), TestCaseError> {
    let err = f.validate().expect_err(name);
    prop_assert_eq!(err.rule(), rule, "{}: {:?}", name, err);
    prop_assert_eq!(variant(&err), expected, "{}: {:?}", name, err);
    prop_assert_eq!(f.check(rule), Err(err));
    only_rule_broken(f, rule)
}

// ---- Property tests -----------------------------------------------------------------------

proptest! {
    /// Every generated graph is valid, rule by rule and as a whole.
    #[test]
    fn generated_graphs_validate(p in params()) {
        let f = build(&p);
        for rule in Rule::ALL {
            prop_assert_eq!(f.check(rule), Ok(()));
        }
        prop_assert_eq!(f.validate(), Ok(()));
    }

    /// Extra ops in `required_ops` are allowed: rule 5 asks for a superset of what is used.
    #[test]
    fn extra_required_ops_are_allowed(p in params()) {
        let mut f = build(&p);
        f.ops.insert(OpKind::AttnDecode(AttnKind::Mla));
        f.ops.insert(OpKind::Pool(syntropy_kernels::PoolKind::Mean));
        prop_assert_eq!(f.validate(), Ok(()));
    }

    /// Each mutator breaks exactly its rule, and `validate` names it.
    #[test]
    fn every_invalid_graph_names_the_rule_it_broke(p in params(), layer in any::<Index>()) {
        let f = build(&p);
        let l = layer.index(f.layers.len());
        for &(name, mutate) in MUTATORS {
            let mut m = f.clone();
            if let Some((rule, expected)) = mutate(&mut m, l) {
                assert_breaks(&m, name, rule, expected)?;
            }
        }
    }

    /// A local value may cross an illegal boundary (cross-layer KV sharing, RFC-0043 §6.1),
    /// and the same graph is rejected under rule 4 once that boundary is marked legal.
    #[test]
    fn local_values_may_cross_only_illegal_boundaries(p in params(), at in any::<Index>()) {
        let mut f = build(&p);
        let n = f.layers.len();
        prop_assume!(n >= 2);
        // Layer `l + 1` reads layer `l`'s attention output, crossing boundary `l + 1` only.
        let l = at.index(n - 1);
        let att = f.layers[l].att;
        f.layer(l + 1).inputs.push(att);
        f.graph.boundaries.legal[l + 1] = false;
        prop_assert_eq!(f.validate(), Ok(()));
        f.graph.boundaries.legal[l + 1] = true;
        let err = f.validate().unwrap_err();
        prop_assert_eq!(err, Violation::ValueCrossesBoundary { boundary: l + 1, value: att });
    }
}

// ---- Unit tests ---------------------------------------------------------------------------

/// The smallest generated graph: one dense layer, one stream, no side data, no MTP.
fn small() -> Fixture {
    build(&Params {
        layers: vec![(false, false)],
        streams: 1,
        mtp: 0,
        per_layer: None,
        legal: 0,
    })
}

/// Every mutator applies to a graph with every feature, at every layer, so none of them is
/// skipped by the property test for want of what it needs.
#[test]
fn every_mutator_applies_to_a_full_graph() {
    for streams in [1, 2] {
        let f = build(&Params {
            layers: vec![(true, true), (false, false), (true, false)],
            streams,
            mtp: 2,
            per_layer: Some(1),
            legal: 0b01,
        });
        assert_eq!(f.validate(), Ok(()));
        for l in 0..f.layers.len() {
            for &(name, mutate) in MUTATORS {
                let mut m = f.clone();
                let Some((rule, expected)) = mutate(&mut m, l) else {
                    // Only the MoE mutators need a MoE layer; layer 1 is dense.
                    assert!(
                        l == 1
                            && matches!(
                                name,
                                "read another layer's experts"
                                    | "move a router group to another layer"
                            ),
                        "{name} at {l}"
                    );
                    continue;
                };
                assert_breaks(&m, name, rule, expected).unwrap();
            }
        }
    }
}

/// RFC-0041 §3.1: the pre-FFN norm of a MoE layer joins the router group, and the final
/// norm joins the head group. Both placements validate.
#[test]
fn norms_may_read_the_group_of_the_block_they_feed() {
    let f = build(&Params {
        layers: vec![(true, false)],
        streams: 1,
        mtp: 0,
        per_layer: None,
        legal: 0,
    });
    let norm = &f.graph.layers[0].body.nodes[f.layers[0].ffn_norm_node];
    assert_eq!(norm.op, OpKind::RmsNorm);
    assert_eq!(
        norm.weights[0].group,
        GroupSel::Fixed(f.layers[0].router.unwrap())
    );
    let final_norm = &f.graph.epilogue.nodes[0];
    assert_eq!(final_norm.op, OpKind::RmsNorm);
    assert_eq!(final_norm.weights[0].group, GroupSel::Fixed(f.head));
    assert_eq!(f.validate(), Ok(()));
}

/// A norm reading a group of a block it does not feed still fails rule 3: a lookup table,
/// an expert group, another layer's router. A state node may not read a router group.
#[test]
fn norms_may_not_read_unrelated_groups() {
    let base = build(&Params {
        layers: vec![(true, false), (true, false)],
        streams: 1,
        mtp: 0,
        per_layer: None,
        legal: 0,
    });
    let node = base.layers[0].ffn_norm_node;
    let expert = base
        .census
        .iter()
        .find(|c| c.class == GroupClass::Expert)
        .unwrap()
        .id;
    let cases = [
        (GroupSel::Fixed(base.embed), "ClassMismatch"),
        (GroupSel::Fixed(expert), "ClassMismatch"),
        (
            GroupSel::Fixed(base.layers[1].router.unwrap()),
            "GroupOfOtherLayer",
        ),
    ];
    for (group, expected) in cases {
        let mut f = base.clone();
        f.graph.layers[0].body.nodes[node].weights[0].group = group;
        let err = f.validate().unwrap_err();
        assert_eq!(err.rule(), Rule::Weights, "{group:?}");
        assert_eq!(err.variant_name(), expected, "{group:?}");
    }
    let mut f = base.clone();
    let kv = f.layers[0].kv_node;
    let router = f.layers[0].router.unwrap();
    f.graph.layers[0].body.nodes[kv].weights.push(WeightRef {
        group: GroupSel::Fixed(router),
        role: TensorRole::Other(0),
    });
    assert_eq!(f.validate().unwrap_err().variant_name(), "ClassMismatch");
}

/// A router group of layer 7 read only by layer 0 fails rule 3.
#[test]
fn a_router_group_of_another_layer_is_rejected() {
    let mut f = build(&Params {
        layers: vec![(true, false)],
        streams: 1,
        mtp: 0,
        per_layer: None,
        legal: 0,
    });
    let router = f.layers[0].router.unwrap();
    f.census.iter_mut().find(|c| c.id == router).unwrap().layer = Some(LayerIx(7));
    assert!(matches!(
        f.validate(),
        Err(Violation::GroupOfOtherLayer {
            program: ProgramId::Layer(0),
            group,
            layer: LayerIx(7),
            ..
        }) if group == router
    ));
}

#[test]
fn rules_are_numbered_as_in_the_rfc() {
    let numbers: Vec<u8> = Rule::ALL.iter().map(|r| r.number()).collect();
    assert_eq!(numbers, [1, 2, 3, 4, 5, 6]);
}

#[test]
fn display_names_the_rule() {
    let mut f = small();
    f.ops.remove(&OpKind::Add);
    let err = f.validate().unwrap_err();
    assert_eq!(
        err.to_string(),
        "graph rule 5 broken: Layer(0) node 5 uses Add, which required_ops does not list"
    );
    let boxed: Box<dyn std::error::Error> = Box::new(err);
    assert!(boxed.source().is_none());
}

#[test]
fn validate_reports_the_lowest_rule_first() {
    let mut f = small();
    f.ops.clear();
    f.graph.boundaries.legal[0] = false;
    f.graph.layers[0].body.nodes[0].inputs.clear();
    assert_eq!(f.validate().unwrap_err().rule(), Rule::Types);
}

#[test]
fn experts_outside_a_layer_are_rejected() {
    let mut f = build(&Params {
        layers: vec![(true, false)],
        streams: 1,
        mtp: 0,
        per_layer: None,
        legal: 0,
    });
    let node = &mut f.graph.prologue.nodes[0];
    node.class = NodeClass::Expert;
    node.weights[0].group = GroupSel::Experts { layer: LayerIx(0) };
    assert_eq!(
        f.validate(),
        Err(Violation::ExpertsOfOtherLayer {
            program: ProgramId::Prologue,
            node: 0,
            layer: LayerIx(0),
        })
    );
}

#[test]
fn experts_need_expert_groups_and_an_expert_node() {
    let moe = Params {
        layers: vec![(true, false), (false, false)],
        streams: 1,
        mtp: 0,
        per_layer: None,
        legal: 0,
    };
    let mut f = build(&moe);
    // Layer 1 is dense, so the census has no expert groups for it.
    f.graph.layers[1].body.nodes[1].class = NodeClass::Expert;
    f.graph.layers[1].body.nodes[1].weights[0].group = GroupSel::Experts { layer: LayerIx(1) };
    assert_eq!(
        f.validate(),
        Err(Violation::NoExpertGroups {
            program: ProgramId::Layer(1),
            node: 1,
            layer: LayerIx(1),
        })
    );

    let mut f = build(&moe);
    let node = f.layers[0].moe_node.unwrap();
    f.graph.layers[0].body.nodes[node].class = NodeClass::Dense;
    assert_eq!(f.validate().unwrap_err().variant_name(), "ClassMismatch");
}

#[test]
fn a_fixed_read_of_an_expert_group_is_a_class_mismatch() {
    let mut f = build(&Params {
        layers: vec![(true, false)],
        streams: 1,
        mtp: 0,
        per_layer: None,
        legal: 0,
    });
    let expert = f
        .census
        .iter()
        .find(|c| c.class == GroupClass::Expert)
        .unwrap()
        .id;
    let node = f.layers[0].moe_node.unwrap();
    f.graph.layers[0].body.nodes[node].weights[0].group = GroupSel::Fixed(expert);
    assert_eq!(f.validate().unwrap_err().variant_name(), "ClassMismatch");
}

#[test]
fn mtp_modules_share_the_head_but_nothing_else() {
    let f = build(&Params {
        layers: vec![(false, false)],
        streams: 1,
        mtp: 2,
        per_layer: None,
        legal: 0,
    });
    assert_eq!(f.validate(), Ok(()));

    // The head read only by MTP modules and a layer is no exception.
    let mut g = f.clone();
    let head = g.head;
    let node = g.layers[0].o_node;
    g.graph.layers[0].body.nodes[node].class = NodeClass::Head;
    g.graph.layers[0].body.nodes[node].weights[0].group = GroupSel::Fixed(head);
    assert_eq!(
        g.validate(),
        Err(Violation::GroupReadAcrossLayers {
            group: head,
            first: ProgramId::Layer(0),
            second: ProgramId::Epilogue,
        })
    );
}

#[test]
fn every_listed_kind_has_a_signature() {
    let kinds = [
        OpKind::Gemm,
        OpKind::AttnPrefill(AttnKind::Bidirectional),
        OpKind::AttnPartial(AttnKind::Gqa),
        OpKind::AttnMerge,
        OpKind::ScanPrefill(syntropy_kernels::ScanKind::Mamba2),
        OpKind::CausalConv1d,
        OpKind::LayerNorm,
        OpKind::Rope(syntropy_kernels::RopeKind::Yarn),
        OpKind::EmbedInject,
        OpKind::VerifyAccept,
        OpKind::GatherRows,
        OpKind::KvPack,
        OpKind::TapGather,
        OpKind::HiddenTap,
    ];
    for op in kinds {
        assert!(signature(&op).is_some(), "{op:?}");
    }
    let partial = signature(&OpKind::AttnPartial(AttnKind::Gqa)).unwrap();
    assert!(partial.outputs.admits_count(3) && !partial.outputs.admits_count(2));
    let merge = signature(&OpKind::AttnMerge).unwrap();
    assert!(merge.inputs.admits_count(9) && !merge.inputs.admits_count(2));
    assert_eq!(merge.inputs.port(8), Some(Port::Float));
}

#[test]
fn op_attrs_look_up_by_name() {
    let attrs = OpAttrs {
        entries: smallvec![
            ("eps", AttrValue::F32(1e-6)),
            ("window", AttrValue::U32(4096))
        ],
    };
    assert_eq!(attrs.get("window"), Some(AttrValue::U32(4096)));
    assert_eq!(attrs.get("top_k"), None);
}

impl Violation {
    fn variant_name(&self) -> String {
        variant(self)
    }
}

/// Rule 6's mutators apply to the small graph, and each refusal names rule 6.
#[test]
fn rule_6_refuses_each_reserved_hook() {
    let mut applied = 0;
    for (name, mutate) in MUTATORS {
        let mut f = small();
        let Some((rule, expected)) = mutate(&mut f, 0) else {
            continue;
        };
        if rule != Rule::ReservedHooks {
            continue;
        }
        applied += 1;
        let err = f.validate().expect_err(name);
        assert_eq!(variant(&err), expected, "{name}");
        assert!(
            err.to_string().starts_with("graph rule 6 broken: ")
                && err.to_string().contains("reserved multimodal hook"),
            "{err}"
        );
    }
    assert_eq!(applied, 3, "each rule-6 mutator applies to the small graph");
}

/// Rule 5 comes before rule 6: an `EmbedInject` node that `required_ops` does not list is
/// reported as rule 5; once listed, as rule 6.
#[test]
fn rule_5_is_reported_before_rule_6() {
    let mut f = small();
    let node = f.graph.layers[0]
        .body
        .nodes
        .iter_mut()
        .find(|n| n.op == OpKind::Add)
        .expect("the small graph has an Add");
    node.op = OpKind::EmbedInject;
    assert_eq!(f.validate().unwrap_err().rule(), Rule::Coverage);
    f.ops.insert(OpKind::EmbedInject);
    assert_eq!(f.validate().unwrap_err().rule(), Rule::ReservedHooks);
}
