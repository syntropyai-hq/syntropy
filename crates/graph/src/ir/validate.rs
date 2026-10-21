// SPDX-License-Identifier: MIT OR Apache-2.0

//! `ir::validate`: rules 1–5 of RFC-0043 §2.2, which run after every family build and in
//! tests (ADR-0430 decision 4).
//!
//! 1. **SSA**: every value is defined once, before use, and programs are acyclic.
//! 2. **Types**: every node's values match its kind's [`signature`].
//! 3. **Weights**: every [`WeightRef`] names a census group of the matching class, every
//!    census group is read by at least one node, and no group is read by two programs, except
//!    the epilogue's `Head` group, which MTP modules that share the LM head also read.
//! 4. **Boundaries**: every legal boundary in [`BoundarySet`] satisfies RFC-0043 §6.1: every
//!    value live across it is a residual stream or declared side data.
//! 5. **Coverage**: every [`OpKind`] used is in the family's `required_ops` (RFC-0040 §11.1).
//!
//! Rules 6 (reserved hooks) and 7 (attention direction) of §2.2 come with the tasks that add
//! encoder-only and multimodal checks. Encoder programs are checked here like any other
//! program, ordered before the prologue, since they feed it.
//!
//! [`validate`] checks the rules in order and returns the first violation; [`check`] checks
//! one. Every check reads the graph defensively, so a graph that breaks one rule never makes
//! another check panic. A violation surfaces to callers as `GraphInvalid`, code
//! `(Graph, 4)` (RFC-0043 §10), which the family-build task raises.
//!
//! [`signature`]: super::signature()
//! [`WeightRef`]: super::WeightRef

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use syntropy_core::{LayerIx, TensorGroupId};
use syntropy_kernels::OpKind;

use super::signature::{Ports, signature};
use super::{GroupSel, ModelGraph, NodeClass, Program, ValueId, ValueKind};

/// A family's declared op set (RFC-0040 §11.1; RFC-0043 §5.1 `required_ops`). Kinds are
/// compared with their parameters, so `AttnDecode(Gqa)` does not cover `AttnDecode(Mla)`.
pub type OpSet = BTreeSet<OpKind>;

/// What rule 3 needs to know about one census group (RFC-0001 §5.3 `TensorGroup`).
///
/// The census types live in `manifest` (RFC-0041 §3.1), which is still a stub, so the caller
/// passes this view of each group; it is built from `TensorCensus::groups` once that exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CensusGroup {
    /// The group.
    pub id: TensorGroupId,
    /// Its class.
    pub class: GroupClass,
    /// Its layer, if it has one. An expert group's layer is the one `GroupSel::Experts`
    /// names.
    pub layer: Option<LayerIx>,
}

/// A census group's class: RFC-0001 §5.3's `TensorClass` without its payloads, whose layer
/// is [`CensusGroup::layer`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GroupClass {
    /// `DenseEveryToken`.
    DenseEveryToken,
    /// `Expert { layer, id }`.
    Expert,
    /// `RowLookup`.
    RowLookup,
    /// `Head`.
    Head,
    /// `Router { layer }`.
    Router,
    /// `Encoder { tower }` (reserved, RFC-0043 §2.4).
    Encoder,
}

/// The rules of RFC-0043 §2.2 that [`validate`] checks.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Rule {
    /// Rule 1: SSA.
    Ssa = 1,
    /// Rule 2: types.
    Types = 2,
    /// Rule 3: weights.
    Weights = 3,
    /// Rule 4: boundaries.
    Boundaries = 4,
    /// Rule 5: coverage.
    Coverage = 5,
}

impl Rule {
    /// Every rule, in the order [`validate`] checks them.
    pub const ALL: [Rule; 5] = [
        Rule::Ssa,
        Rule::Types,
        Rule::Weights,
        Rule::Boundaries,
        Rule::Coverage,
    ];

    /// Its number in RFC-0043 §2.2.
    #[must_use]
    pub fn number(self) -> u8 {
        self as u8
    }
}

/// One program of a [`ModelGraph`], by position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ProgramId {
    /// `encoders[i].body`.
    Encoder(u16),
    /// The prologue.
    Prologue,
    /// `layers[i].body`.
    Layer(u16),
    /// The epilogue.
    Epilogue,
    /// `mtp[i].body`.
    Mtp(u16),
}

/// Whether a value list is a node's inputs or its outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    /// The node's inputs.
    Inputs,
    /// The node's outputs.
    Outputs,
}

/// The first violation [`validate`] found. [`Violation::rule`] names the rule it breaks.
///
/// `node` is a position in its program's `nodes`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Violation {
    // ---- Rule 1: SSA ----
    /// A value ID is not an index of `ModelGraph::values`.
    UnknownValue {
        /// The program.
        program: ProgramId,
        /// The value.
        value: ValueId,
    },
    /// A program input is defined neither by the step (token IDs) nor by an earlier program.
    InputNotDefined {
        /// The program.
        program: ProgramId,
        /// The value.
        value: ValueId,
    },
    /// A node, or with `node: None` the program's outputs, reads a value that is neither a
    /// program input nor defined by an earlier node of the program.
    UseBeforeDef {
        /// The program.
        program: ProgramId,
        /// The node, or `None` for the program's outputs.
        node: Option<usize>,
        /// The value.
        value: ValueId,
    },
    /// A node defines a value that is already defined, by a node or, for token IDs, by the
    /// step.
    DefinedTwice {
        /// The program.
        program: ProgramId,
        /// The node.
        node: usize,
        /// The value.
        value: ValueId,
    },
    // ---- Rule 2: types ----
    /// The node's kind has no value signature.
    NoSignature {
        /// The program.
        program: ProgramId,
        /// The node.
        node: usize,
        /// Its kind.
        op: OpKind,
    },
    /// The node reads or defines a number of values its signature does not allow.
    Arity {
        /// The program.
        program: ProgramId,
        /// The node.
        node: usize,
        /// Inputs or outputs.
        side: Side,
        /// How many it has.
        count: usize,
    },
    /// A value's dtype does not fit its position in the node's signature.
    Dtype {
        /// The program.
        program: ProgramId,
        /// The node.
        node: usize,
        /// The value.
        value: ValueId,
    },
    /// A value of a node whose signature keeps dims has dims other than the first input's.
    Dims {
        /// The program.
        program: ProgramId,
        /// The node.
        node: usize,
        /// The value.
        value: ValueId,
    },
    // ---- Rule 3: weights ----
    /// Two census entries have the same group ID.
    DuplicateGroup {
        /// The group.
        group: TensorGroupId,
    },
    /// A `GroupSel::Fixed` names a group the census does not have.
    UnknownGroup {
        /// The program.
        program: ProgramId,
        /// The node.
        node: usize,
        /// The group.
        group: TensorGroupId,
    },
    /// A weight's group class does not match the node's class.
    ClassMismatch {
        /// The program.
        program: ProgramId,
        /// The node.
        node: usize,
        /// The weight's group.
        group: GroupSel,
    },
    /// A `GroupSel::Experts` names a layer other than the program's own, or appears outside
    /// a layer or MTP module.
    ExpertsOfOtherLayer {
        /// The program.
        program: ProgramId,
        /// The node.
        node: usize,
        /// The layer it names.
        layer: LayerIx,
    },
    /// A `GroupSel::Experts` names a layer with no expert groups in the census.
    NoExpertGroups {
        /// The program.
        program: ProgramId,
        /// The node.
        node: usize,
        /// The layer it names.
        layer: LayerIx,
    },
    /// No node reads a census group.
    GroupUnread {
        /// The group.
        group: TensorGroupId,
    },
    /// Nodes in two programs read one group, outside the MTP head exception.
    GroupReadAcrossLayers {
        /// The group.
        group: TensorGroupId,
        /// The first program that reads it.
        first: ProgramId,
        /// The next program that reads it.
        second: ProgramId,
    },
    // ---- Rule 4: boundaries ----
    /// `BoundarySet::legal` or `side` does not have one entry per boundary, `0..=n_layers`.
    BoundarySetSize {
        /// `legal.len()`.
        legal: usize,
        /// `side.len()`.
        side: usize,
        /// `n_layers + 1`.
        expected: usize,
    },
    /// Boundary 0 or `n_layers`, which always exist, is not legal.
    EndBoundaryIllegal {
        /// The boundary.
        boundary: usize,
    },
    /// A layer's or MTP module's `LayerIx` is not its position, which boundaries are counted
    /// by.
    LayerIndex {
        /// The program.
        program: ProgramId,
        /// The `LayerIx` it has.
        layer: LayerIx,
    },
    /// A value that is neither a residual stream in the residual dtype nor side data is live
    /// across a legal boundary.
    ValueCrossesBoundary {
        /// The boundary.
        boundary: usize,
        /// The value.
        value: ValueId,
    },
    /// Side data is live across a legal boundary whose `SideData` does not declare it.
    SideDataUndeclared {
        /// The boundary.
        boundary: usize,
        /// The value.
        value: ValueId,
    },
    // ---- Rule 5: coverage ----
    /// A node's kind is not in the family's `required_ops`.
    OpNotRequired {
        /// The program.
        program: ProgramId,
        /// The node.
        node: usize,
        /// Its kind.
        op: OpKind,
    },
}

impl Violation {
    /// The rule this violation breaks.
    #[must_use]
    pub fn rule(&self) -> Rule {
        match self {
            Self::UnknownValue { .. }
            | Self::InputNotDefined { .. }
            | Self::UseBeforeDef { .. }
            | Self::DefinedTwice { .. } => Rule::Ssa,
            Self::NoSignature { .. }
            | Self::Arity { .. }
            | Self::Dtype { .. }
            | Self::Dims { .. } => Rule::Types,
            Self::DuplicateGroup { .. }
            | Self::UnknownGroup { .. }
            | Self::ClassMismatch { .. }
            | Self::ExpertsOfOtherLayer { .. }
            | Self::NoExpertGroups { .. }
            | Self::GroupUnread { .. }
            | Self::GroupReadAcrossLayers { .. } => Rule::Weights,
            Self::BoundarySetSize { .. }
            | Self::EndBoundaryIllegal { .. }
            | Self::LayerIndex { .. }
            | Self::ValueCrossesBoundary { .. }
            | Self::SideDataUndeclared { .. } => Rule::Boundaries,
            Self::OpNotRequired { .. } => Rule::Coverage,
        }
    }
}

/// Shows the rule's number, then the details, e.g.
/// `graph rule 5 broken: Layer(0) node 3 uses Add, which required_ops does not list`.
impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "graph rule {} broken: ", self.rule().number())?;
        match self {
            Self::UnknownValue { program, value } => {
                write!(f, "{program:?} names value {} past the table", value.0)
            }
            Self::InputNotDefined { program, value } => write!(
                f,
                "{program:?} input {} is not defined by the step or an earlier program",
                value.0
            ),
            Self::UseBeforeDef {
                program,
                node: Some(node),
                value,
            } => write!(
                f,
                "{program:?} node {node} reads {} before it is defined",
                value.0
            ),
            Self::UseBeforeDef {
                program,
                node: None,
                value,
            } => write!(f, "{program:?} outputs {}, which it never has", value.0),
            Self::DefinedTwice {
                program,
                node,
                value,
            } => write!(f, "{program:?} node {node} defines {} again", value.0),
            Self::NoSignature { program, node, op } => {
                write!(
                    f,
                    "{program:?} node {node} uses {op:?}, which has no signature"
                )
            }
            Self::Arity {
                program,
                node,
                side,
                count,
            } => write!(
                f,
                "{program:?} node {node} has {count} {side:?}, which its signature does not allow"
            ),
            Self::Dtype {
                program,
                node,
                value,
            } => write!(
                f,
                "{program:?} node {node}: the dtype of {} does not fit",
                value.0
            ),
            Self::Dims {
                program,
                node,
                value,
            } => write!(
                f,
                "{program:?} node {node}: {} has other dims than its first input",
                value.0
            ),
            Self::DuplicateGroup { group } => {
                write!(f, "the census lists group {} twice", group.0)
            }
            Self::UnknownGroup {
                program,
                node,
                group,
            } => write!(
                f,
                "{program:?} node {node} reads group {}, which the census does not have",
                group.0
            ),
            Self::ClassMismatch {
                program,
                node,
                group,
            } => write!(
                f,
                "{program:?} node {node} reads {group:?} of another class"
            ),
            Self::ExpertsOfOtherLayer {
                program,
                node,
                layer,
            } => write!(
                f,
                "{program:?} node {node} reads the experts of layer {}",
                layer.0
            ),
            Self::NoExpertGroups {
                program,
                node,
                layer,
            } => write!(
                f,
                "{program:?} node {node} reads the experts of layer {}, which has none",
                layer.0
            ),
            Self::GroupUnread { group } => write!(f, "no node reads group {}", group.0),
            Self::GroupReadAcrossLayers {
                group,
                first,
                second,
            } => write!(f, "group {} is read by {first:?} and {second:?}", group.0),
            Self::BoundarySetSize {
                legal,
                side,
                expected,
            } => write!(
                f,
                "the boundary set has {legal} legal and {side} side entries, not {expected}"
            ),
            Self::EndBoundaryIllegal { boundary } => {
                write!(f, "boundary {boundary} always exists but is not legal")
            }
            Self::LayerIndex { program, layer } => {
                write!(f, "{program:?} has LayerIx {}", layer.0)
            }
            Self::ValueCrossesBoundary { boundary, value } => write!(
                f,
                "value {} is live across legal boundary {boundary} but is no residual stream or side data",
                value.0
            ),
            Self::SideDataUndeclared { boundary, value } => write!(
                f,
                "side data {} is live across boundary {boundary}, which does not declare it",
                value.0
            ),
            Self::OpNotRequired { program, node, op } => write!(
                f,
                "{program:?} node {node} uses {op:?}, which required_ops does not list"
            ),
        }
    }
}

impl std::error::Error for Violation {}

/// Checks rules 1–5 of RFC-0043 §2.2 in order on `graph`, against its census and its
/// family's `required_ops`.
///
/// # Errors
///
/// The first violation of the first rule `graph` breaks.
pub fn validate(
    graph: &ModelGraph,
    census: &[CensusGroup],
    required_ops: &OpSet,
) -> Result<(), Violation> {
    Rule::ALL
        .into_iter()
        .try_for_each(|rule| check(rule, graph, census, required_ops))
}

/// Checks one rule of RFC-0043 §2.2.
///
/// # Errors
///
/// The first violation of `rule`.
pub fn check(
    rule: Rule,
    graph: &ModelGraph,
    census: &[CensusGroup],
    required_ops: &OpSet,
) -> Result<(), Violation> {
    let programs = programs(graph);
    match rule {
        Rule::Ssa => check_ssa(graph, &programs),
        Rule::Types => check_types(graph, &programs),
        Rule::Weights => check_weights(graph, &programs, census),
        Rule::Boundaries => check_boundaries(graph, &programs),
        Rule::Coverage => check_coverage(&programs, required_ops),
    }
}

/// Every program in execution order: encoders, prologue, layers, epilogue, MTP modules.
fn programs(g: &ModelGraph) -> Vec<(ProgramId, &Program)> {
    let mut all = Vec::with_capacity(g.encoders.len() + g.layers.len() + g.mtp.len() + 2);
    all.extend(
        g.encoders
            .iter()
            .enumerate()
            .map(|(i, e)| (ProgramId::Encoder(ix(i)), &e.body)),
    );
    all.push((ProgramId::Prologue, &g.prologue));
    all.extend(
        g.layers
            .iter()
            .enumerate()
            .map(|(i, l)| (ProgramId::Layer(ix(i)), &l.body)),
    );
    all.push((ProgramId::Epilogue, &g.epilogue));
    all.extend(
        g.mtp
            .iter()
            .enumerate()
            .map(|(i, l)| (ProgramId::Mtp(ix(i)), &l.body)),
    );
    all
}

/// A position as a `u16`, saturating; rule 4 rejects a graph with more layers than
/// `LayerIx` holds.
fn ix(i: usize) -> u16 {
    u16::try_from(i).unwrap_or(u16::MAX)
}

/// The `LayerIx` a program has by position: `i` for layer `i`, `n_layers + i` for MTP module
/// `i`, none for the others.
fn positional_layer(g: &ModelGraph, id: ProgramId) -> Option<usize> {
    match id {
        ProgramId::Layer(i) => Some(usize::from(i)),
        ProgramId::Mtp(i) => Some(g.layers.len() + usize::from(i)),
        ProgramId::Encoder(_) | ProgramId::Prologue | ProgramId::Epilogue => None,
    }
}

/// The index of `v` in the value table, if it has one.
fn index(g: &ModelGraph, v: ValueId) -> Option<usize> {
    usize::try_from(v.0).ok().filter(|&i| i < g.values.len())
}

fn check_ssa(g: &ModelGraph, programs: &[(ProgramId, &Program)]) -> Result<(), Violation> {
    // The step defines the token IDs before the first program.
    let mut defined: Vec<bool> = g
        .values
        .iter()
        .map(|v| v.kind == ValueKind::TokenIds)
        .collect();
    // `avail[v] == stamp` when the current program may read `v`.
    let mut avail = vec![0usize; g.values.len()];
    for (pos, &(program, body)) in programs.iter().enumerate() {
        let stamp = pos + 1;
        let known = |value| index(g, value).ok_or(Violation::UnknownValue { program, value });
        for &value in &body.inputs {
            let i = known(value)?;
            if !defined[i] {
                return Err(Violation::InputNotDefined { program, value });
            }
            avail[i] = stamp;
        }
        for (node, n) in body.nodes.iter().enumerate() {
            for &value in &n.inputs {
                if avail[known(value)?] != stamp {
                    return Err(Violation::UseBeforeDef {
                        program,
                        node: Some(node),
                        value,
                    });
                }
            }
            for &value in &n.outputs {
                let i = known(value)?;
                if defined[i] {
                    return Err(Violation::DefinedTwice {
                        program,
                        node,
                        value,
                    });
                }
                defined[i] = true;
                avail[i] = stamp;
            }
        }
        for &value in &body.outputs {
            if avail[known(value)?] != stamp {
                return Err(Violation::UseBeforeDef {
                    program,
                    node: None,
                    value,
                });
            }
        }
    }
    Ok(())
}

fn check_types(g: &ModelGraph, programs: &[(ProgramId, &Program)]) -> Result<(), Violation> {
    for &(program, body) in programs {
        for (node, n) in body.nodes.iter().enumerate() {
            let Some(sig) = signature(&n.op) else {
                return Err(Violation::NoSignature {
                    program,
                    node,
                    op: n.op,
                });
            };
            let sides: [(Side, &[ValueId], &Ports); 2] = [
                (Side::Inputs, &n.inputs, &sig.inputs),
                (Side::Outputs, &n.outputs, &sig.outputs),
            ];
            for (side, values, ports) in sides {
                if !ports.admits_count(values.len()) {
                    return Err(Violation::Arity {
                        program,
                        node,
                        side,
                        count: values.len(),
                    });
                }
                for (i, &value) in values.iter().enumerate() {
                    // An unknown value is rule 1's to report.
                    let Some(v) = index(g, value).map(|i| &g.values[i]) else {
                        continue;
                    };
                    if !ports.port(i).is_some_and(|port| port.admits(v.ty.dtype)) {
                        return Err(Violation::Dtype {
                            program,
                            node,
                            value,
                        });
                    }
                }
            }
            if !sig.same_dims {
                continue;
            }
            let first = n
                .inputs
                .first()
                .and_then(|&v| index(g, v))
                .map(|i| &g.values[i].ty.dims);
            let Some(dims) = first else { continue };
            for &value in n.inputs.iter().chain(&n.outputs) {
                if let Some(i) = index(g, value)
                    && g.values[i].ty.dims != *dims
                {
                    return Err(Violation::Dims {
                        program,
                        node,
                        value,
                    });
                }
            }
        }
    }
    Ok(())
}

/// Whether a node of `node` class may read a `GroupSel::Fixed` group of `group` class.
/// Elementwise and state nodes read the dense group of their block: norm weights, sink
/// vectors, convolution and scan parameters.
fn class_matches(node: NodeClass, group: GroupClass) -> bool {
    matches!(
        (node, group),
        (
            NodeClass::Dense | NodeClass::Elementwise | NodeClass::State,
            GroupClass::DenseEveryToken
        ) | (NodeClass::Router, GroupClass::Router)
            | (NodeClass::Head, GroupClass::Head)
            | (NodeClass::Lookup, GroupClass::RowLookup)
            | (NodeClass::Encoder, GroupClass::Encoder)
    )
}

fn check_weights(
    g: &ModelGraph,
    programs: &[(ProgramId, &Program)],
    census: &[CensusGroup],
) -> Result<(), Violation> {
    let mut groups: BTreeMap<TensorGroupId, &CensusGroup> = BTreeMap::new();
    for group in census {
        if groups.insert(group.id, group).is_some() {
            return Err(Violation::DuplicateGroup { group: group.id });
        }
    }
    let expert_layers: BTreeSet<LayerIx> = census
        .iter()
        .filter(|c| c.class == GroupClass::Expert)
        .filter_map(|c| c.layer)
        .collect();
    // The programs that read each group, in program order and without repeats.
    let mut readers: BTreeMap<TensorGroupId, Vec<ProgramId>> = BTreeMap::new();
    let mut experts_read: BTreeSet<LayerIx> = BTreeSet::new();
    for &(program, body) in programs {
        let own = positional_layer(g, program);
        for (node, n) in body.nodes.iter().enumerate() {
            for w in &n.weights {
                match w.group {
                    GroupSel::Fixed(id) => {
                        let Some(group) = groups.get(&id) else {
                            return Err(Violation::UnknownGroup {
                                program,
                                node,
                                group: id,
                            });
                        };
                        if !class_matches(n.class, group.class) {
                            return Err(Violation::ClassMismatch {
                                program,
                                node,
                                group: w.group,
                            });
                        }
                        let list = readers.entry(id).or_default();
                        if list.last() != Some(&program) {
                            list.push(program);
                        }
                    }
                    GroupSel::Experts { layer } => {
                        if n.class != NodeClass::Expert {
                            return Err(Violation::ClassMismatch {
                                program,
                                node,
                                group: w.group,
                            });
                        }
                        if own != Some(usize::from(layer.0)) {
                            return Err(Violation::ExpertsOfOtherLayer {
                                program,
                                node,
                                layer,
                            });
                        }
                        if !expert_layers.contains(&layer) {
                            return Err(Violation::NoExpertGroups {
                                program,
                                node,
                                layer,
                            });
                        }
                        experts_read.insert(layer);
                    }
                }
            }
        }
    }
    for group in census {
        let read = match group.class {
            GroupClass::Expert => group.layer.is_some_and(|l| experts_read.contains(&l)),
            _ => readers.contains_key(&group.id),
        };
        if !read {
            return Err(Violation::GroupUnread { group: group.id });
        }
    }
    for (&id, list) in &readers {
        let [first, second, ..] = list[..] else {
            continue;
        };
        // An MTP module that shares the LM head reads the epilogue's `Head` group (§2.2
        // rule 3); the epilogue comes first in program order.
        let shared_head = groups[&id].class == GroupClass::Head
            && first == ProgramId::Epilogue
            && list[1..].iter().all(|p| matches!(p, ProgramId::Mtp(_)));
        if !shared_head {
            return Err(Violation::GroupReadAcrossLayers {
                group: id,
                first,
                second,
            });
        }
    }
    Ok(())
}

fn check_boundaries(g: &ModelGraph, programs: &[(ProgramId, &Program)]) -> Result<(), Violation> {
    let n = g.layers.len();
    let b = &g.boundaries;
    if b.legal.len() != n + 1 || b.side.len() != n + 1 {
        return Err(Violation::BoundarySetSize {
            legal: b.legal.len(),
            side: b.side.len(),
            expected: n + 1,
        });
    }
    for boundary in [0, n] {
        if !b.legal[boundary] {
            return Err(Violation::EndBoundaryIllegal { boundary });
        }
    }
    let indexed = g
        .layers
        .iter()
        .enumerate()
        .map(|(i, l)| (ProgramId::Layer(ix(i)), i, l))
        .chain(
            g.mtp
                .iter()
                .enumerate()
                .map(|(i, l)| (ProgramId::Mtp(ix(i)), n + i, l)),
        );
    for (program, pos, l) in indexed {
        if usize::from(l.layer.0) != pos {
            return Err(Violation::LayerIndex {
                program,
                layer: l.layer,
            });
        }
    }

    // Liveness by program position: a value crosses the cut after position `c` when it is
    // defined at or before `c` and read after it. The step defines token IDs at -1.
    let mut def: Vec<Option<i64>> = g
        .values
        .iter()
        .map(|v| (v.kind == ValueKind::TokenIds).then_some(-1))
        .collect();
    let mut last_use: Vec<Option<i64>> = vec![None; g.values.len()];
    for (pos, &(_, body)) in programs.iter().enumerate() {
        let pos = i64::try_from(pos).unwrap_or(i64::MAX);
        let reads = body
            .inputs
            .iter()
            .chain(body.nodes.iter().flat_map(|n| &n.inputs));
        for &v in reads {
            if let Some(i) = index(g, v) {
                last_use[i] = Some(last_use[i].map_or(pos, |u| u.max(pos)));
            }
        }
        for &v in body.nodes.iter().flat_map(|n| &n.outputs) {
            if let Some(i) = index(g, v) {
                def[i].get_or_insert(pos);
            }
        }
    }
    let prologue = i64::try_from(g.encoders.len()).unwrap_or(i64::MAX);
    for (boundary, side) in b.side.iter().enumerate() {
        if !b.legal[boundary] {
            continue;
        }
        // Boundary `b` is the cut after layer `b − 1`, or after the prologue for `b = 0`.
        let cut = prologue + i64::try_from(boundary).unwrap_or(i64::MAX);
        for (i, v) in g.values.iter().enumerate() {
            let (Some(d), Some(u)) = (def[i], last_use[i]) else {
                continue;
            };
            if !(d <= cut && cut < u) {
                continue;
            }
            let value = ValueId(u32::try_from(i).unwrap_or(u32::MAX));
            let ok = match v.kind {
                ValueKind::Residual if v.ty.dtype == g.residual.dtype => true,
                ValueKind::TokenIds if side.token_ids => true,
                ValueKind::LayerInput if side.layer_inputs > 0 => true,
                ValueKind::TokenIds | ValueKind::LayerInput => {
                    return Err(Violation::SideDataUndeclared { boundary, value });
                }
                _ => false,
            };
            if !ok {
                return Err(Violation::ValueCrossesBoundary { boundary, value });
            }
        }
    }
    Ok(())
}

fn check_coverage(
    programs: &[(ProgramId, &Program)],
    required_ops: &OpSet,
) -> Result<(), Violation> {
    for &(program, body) in programs {
        for (node, n) in body.nodes.iter().enumerate() {
            if !required_ops.contains(&n.op) {
                return Err(Violation::OpNotRequired {
                    program,
                    node,
                    op: n.op,
                });
            }
        }
    }
    Ok(())
}
