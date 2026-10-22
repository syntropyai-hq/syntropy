// SPDX-License-Identifier: MIT OR Apache-2.0

//! Legal stage boundaries and side data, computed from value liveness on the IR (RFC-0043
//! §6.1; ADR-0006; ADR-0003 decision 3).
//!
//! Boundary `b` sits between layer `b − 1` and layer `b`, over `0..=n_layers` decoder layers.
//! [`compute`] decides each boundary:
//!
//! - **Liveness.** It is legal only when every value live across it is a residual stream or
//!   declared side data: token IDs ([`ValueKind::TokenIds`]) or a per-layer input
//!   ([`ValueKind::LayerInput`]). Any other value makes it illegal.
//! - **Cross-layer KV sharing.** A layer that reads an earlier layer's KV ([`KvShare`]) makes
//!   every boundary between the two illegal. KV lives in the KV store, not in the IR (RFC-0043
//!   §2.3), so the family declares the sharing in [`BoundaryFacts`].
//! - **Atomic pairs.** A pair of adjacent layers the family marks atomic makes the boundary
//!   between them illegal.
//!
//! Boundaries 0 and `n_layers` always exist. A value other than a residual stream or side
//! data live across one of them is a family bug, which [`compute`] reports as
//! [`BoundaryError::EndBoundaryCrossed`].
//!
//! For every boundary, legal or not, the side data live across it is recorded with its bytes
//! per token: the solver adds them to the hop size `S_hop` (cost model §3.7), and RFC-0031
//! carries them in `ActivationFrame::side`. `layer_inputs` is RFC-0043 §6.1's "elements per
//! token per remaining layer": the width of one per-layer input. Every per-layer input live
//! across a boundary must have that one width ([`BoundaryError::LayerInputWidths`]), and
//! `bytes_per_token` counts all of them, so it grows with the number of remaining layers that
//! read one.
//!
//! **Plans.** [`check_cuts`] rejects a plan whose stage ranges cut at an illegal boundary,
//! or between the decoder layers' end and the MTP modules' end, which separates an MTP module
//! from the epilogue (both run on the tail, RFC-0043 §6.2). A cut at boundary `n_layers` is
//! accepted when the graph has MTP modules: the last stage then holds the epilogue and the MTP
//! modules only. [`partition`] applies the same rule through the same function. That is `BoundaryIllegal`, code
//! `(Graph, 5)`, class `Internal`: a solver bug, and the plan is rejected (RFC-0043 §10).
//!
//! **Parts.** [`partition`] splits the graph at legal cuts into the programs each stage runs
//! and the values each stage receives and sends, which is what RFC-0060 §7.2 check 8 and the
//! segment builders (COMP-E15-T07, COMP-E15-T10) rely on.

use std::fmt;
use std::ops::Range;

use syntropy_core::{ErrorClass, LayerIx};

use crate::ir::validate::{Liveness, programs};
use crate::ir::{BoundarySet, Dim, ModelGraph, ProgramId, SideData, ValueId, ValueKind};

/// A layer that reads an earlier layer's KV (RFC-0043 §6.1: a cross-layer KV-sharing group).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KvShare {
    /// The layer that reads the KV.
    pub reader: LayerIx,
    /// The earlier layer whose KV it reads.
    pub source: LayerIx,
}

/// What a family declares about its graph that the IR does not show (RFC-0043 §6.1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BoundaryFacts<'a> {
    /// The cross-layer KV-sharing pairs.
    pub kv_shared: &'a [KvShare],
    /// The atomic pairs, each named by its first layer: `l` makes layers `l` and `l + 1` one
    /// unit, so boundary `l + 1` is illegal.
    pub atomic: &'a [LayerIx],
}

/// Why a boundary is illegal (RFC-0043 §6.1).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Illegal {
    /// This value is live across it and is neither a residual stream nor side data. Of
    /// several such values, the lowest ID is named.
    Crossing(ValueId),
    /// A layer after it reads the KV of a layer before it.
    KvShared(KvShare),
    /// It splits the atomic pair that starts at this layer.
    Atomic(LayerIx),
}

impl fmt::Display for Illegal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Crossing(v) => write!(
                f,
                "value {} is live across it and is no residual stream or side data",
                v.0
            ),
            Self::KvShared(s) => write!(
                f,
                "layer {} reads the KV of layer {}",
                s.reader.0, s.source.0
            ),
            Self::Atomic(l) => write!(
                f,
                "it splits the atomic pair of layers {} and {}",
                l.0,
                l.0 + 1
            ),
        }
    }
}

/// The boundary set of a graph, with the reason for each illegal boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundaryAnalysis {
    /// What [`ModelGraph::boundaries`] carries.
    pub set: BoundarySet,
    /// Per boundary, `0..=n_layers`: why it is illegal, or `None` when it is legal.
    pub illegal: Vec<Option<Illegal>>,
}

/// Why [`compute`] could not build the boundary set: a family bug.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BoundaryError {
    /// A value other than a residual stream or side data is live across boundary 0 or
    /// `n_layers`, which always exist.
    EndBoundaryCrossed {
        /// The boundary.
        boundary: usize,
        /// The value.
        value: ValueId,
    },
    /// A [`KvShare`] whose reader is not after its source, or names a layer the graph does
    /// not have.
    KvShareInvalid(KvShare),
    /// An atomic pair whose second layer the graph does not have.
    AtomicInvalid(LayerIx),
    /// A side-data value has a `Ctx` dim, so its bytes per token are not fixed.
    SideDataShape(ValueId),
    /// Two per-layer inputs live across one boundary differ in width, so the boundary has no
    /// single "elements per token per remaining layer" (RFC-0043 §6.1).
    LayerInputWidths {
        /// The boundary.
        boundary: usize,
        /// The per-layer input whose width differs from an earlier one's.
        value: ValueId,
    },
    /// The graph has more layers than a `LayerIx` holds.
    TooManyLayers(usize),
}

impl fmt::Display for BoundaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EndBoundaryCrossed { boundary, value } => write!(
                f,
                "value {} is live across boundary {boundary}, which always exists, and is no residual stream or side data",
                value.0
            ),
            Self::KvShareInvalid(s) => write!(
                f,
                "KV sharing of layer {} by layer {} is not from an earlier layer of the graph",
                s.source.0, s.reader.0
            ),
            Self::AtomicInvalid(l) => {
                write!(
                    f,
                    "atomic pair at layer {} has no layer {}",
                    l.0,
                    u32::from(l.0) + 1
                )
            }
            Self::SideDataShape(v) => write!(f, "side data {} has a context-length dim", v.0),
            Self::LayerInputWidths { boundary, value } => write!(
                f,
                "per-layer input {} across boundary {boundary} differs in width from another",
                value.0
            ),
            Self::TooManyLayers(n) => write!(f, "{n} layers do not fit a LayerIx"),
        }
    }
}

impl std::error::Error for BoundaryError {}

/// The values live across boundary `boundary` of `g`, in ID order. Boundaries past
/// `n_layers` have none.
#[must_use]
pub fn crossing(g: &ModelGraph, boundary: usize) -> Vec<ValueId> {
    if boundary > g.layers.len() {
        return Vec::new();
    }
    let live = Liveness::of(g, &programs(g));
    crossing_with(g, &live, boundary)
}

fn crossing_with(g: &ModelGraph, live: &Liveness, boundary: usize) -> Vec<ValueId> {
    let cut = live.cut(boundary);
    (0..g.values.len())
        .filter(|&i| live.crosses(i, cut))
        .map(|i| ValueId(u32::try_from(i).unwrap_or(u32::MAX)))
        .collect()
}

/// The elements and bytes per token of a side-data value: the product of its `Const` dims,
/// with `Rows` and `Seqs` as the token dims.
fn per_token(g: &ModelGraph, v: ValueId) -> Result<(u64, u64), BoundaryError> {
    let ty = &g.values[usize::try_from(v.0).unwrap_or(usize::MAX)].ty;
    let mut elems: u64 = 1;
    for d in &ty.dims {
        match d {
            Dim::Const(n) => elems = elems.saturating_mul(u64::from(*n)),
            Dim::Rows | Dim::Seqs => {}
            Dim::Ctx => return Err(BoundaryError::SideDataShape(v)),
        }
    }
    let bits = elems.saturating_mul(u64::from(ty.dtype.bits()));
    Ok((elems, bits.div_ceil(8)))
}

fn sat_u32(x: u64) -> u32 {
    u32::try_from(x).unwrap_or(u32::MAX)
}

/// Computes the boundary set of `g` from value liveness and the family's `facts` (RFC-0043
/// §6.1). Store `set` in [`ModelGraph::boundaries`]; `ir::validate` rule 4 then holds.
///
/// # Errors
///
/// A [`BoundaryError`]: the graph or the facts break §6.1 in a way no boundary choice fixes.
pub fn compute(
    g: &ModelGraph,
    facts: &BoundaryFacts<'_>,
) -> Result<BoundaryAnalysis, BoundaryError> {
    let n = g.layers.len();
    let n16 = u16::try_from(n).map_err(|_| BoundaryError::TooManyLayers(n))?;
    for &s in facts.kv_shared {
        if s.source >= s.reader || s.reader.0 >= n16 {
            return Err(BoundaryError::KvShareInvalid(s));
        }
    }
    for &l in facts.atomic {
        if u32::from(l.0) + 1 >= u32::from(n16) {
            return Err(BoundaryError::AtomicInvalid(l));
        }
    }

    let live = Liveness::of(g, &programs(g));
    let mut legal = Vec::with_capacity(n + 1);
    let mut side = Vec::with_capacity(n + 1);
    let mut illegal = Vec::with_capacity(n + 1);
    for b in 0..=n {
        let mut data = SideData::default();
        let mut tokens = 0u64;
        let mut inputs = 0u64;
        let mut cause = None;
        for v in crossing_with(g, &live, b) {
            let value = &g.values[usize::try_from(v.0).unwrap_or(usize::MAX)];
            match value.kind {
                ValueKind::Residual if value.ty.dtype == g.residual.dtype => {}
                ValueKind::TokenIds => {
                    data.token_ids = true;
                    tokens = tokens.saturating_add(per_token(g, v)?.1);
                }
                ValueKind::LayerInput => {
                    let (elems, bytes) = per_token(g, v)?;
                    let elems = sat_u32(elems);
                    if data.layer_inputs != 0 && data.layer_inputs != elems {
                        return Err(BoundaryError::LayerInputWidths {
                            boundary: b,
                            value: v,
                        });
                    }
                    data.layer_inputs = elems;
                    inputs = inputs.saturating_add(bytes);
                }
                _ => {
                    if b == 0 || b == n {
                        return Err(BoundaryError::EndBoundaryCrossed {
                            boundary: b,
                            value: v,
                        });
                    }
                    cause.get_or_insert(Illegal::Crossing(v));
                }
            }
        }
        data.bytes_per_token = sat_u32(tokens.saturating_add(inputs));
        let b16 = u16::try_from(b).unwrap_or(u16::MAX);
        if cause.is_none() {
            cause = facts
                .kv_shared
                .iter()
                .filter(|s| s.source.0 < b16 && b16 <= s.reader.0)
                .min()
                .map(|&s| Illegal::KvShared(s));
        }
        if cause.is_none() {
            cause = facts
                .atomic
                .iter()
                .filter(|l| u32::from(l.0) + 1 == u32::from(b16))
                .min()
                .map(|&l| Illegal::Atomic(l));
        }
        legal.push(cause.is_none());
        side.push(data);
        illegal.push(cause);
    }
    Ok(BoundaryAnalysis {
        set: BoundarySet { legal, side },
        illegal,
    })
}

/// Why a plan's cut is rejected.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CutCause {
    /// The boundary is not legal in the graph's [`BoundarySet`].
    NotLegal,
    /// The cut falls after the decoder layers and inside the MTP modules, separating an MTP
    /// module from the epilogue.
    MtpSeparated,
    /// The cut is past the last decoder layer and MTP module, or the cuts are not in
    /// increasing order.
    OutOfRange,
}

/// A plan cuts at an illegal boundary: `BoundaryIllegal`, code `(Graph, 5)`, class
/// `Internal` (RFC-0043 §10). The plan is rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BoundaryIllegal {
    /// The cut: the first layer of the stage that starts there.
    pub boundary: LayerIx,
    /// Why it is rejected.
    pub cause: CutCause,
}

impl BoundaryIllegal {
    /// Its number in the `Graph` domain (RFC-0043 §10).
    pub const CODE: u16 = 5;
    /// Its class: a solver bug.
    pub const CLASS: ErrorClass = ErrorClass::Internal;
}

impl fmt::Display for BoundaryIllegal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let why = match self.cause {
            CutCause::NotLegal => "it is not a legal boundary",
            CutCause::MtpSeparated => "it separates an MTP module from the epilogue",
            CutCause::OutOfRange => "it is out of range or out of order",
        };
        write!(
            f,
            "BoundaryIllegal (Graph 5): the plan cuts at boundary {}: {why}",
            self.boundary.0
        )
    }
}

impl std::error::Error for BoundaryIllegal {}

/// The one cut rule [`check_cuts`] and [`partition`] share: `b` is the next cut after `prev`
/// over a graph whose boundaries are `legal` (`0..=n_layers`) and which has `n_mtp` MTP
/// modules.
fn cut_cause(legal: &[bool], n_mtp: usize, prev: usize, b: usize) -> Option<CutCause> {
    let n = legal.len().saturating_sub(1);
    if b <= prev || b >= n.saturating_add(n_mtp) {
        Some(CutCause::OutOfRange)
    } else if b > n {
        Some(CutCause::MtpSeparated)
    } else if !legal[b] {
        Some(CutCause::NotLegal)
    } else {
        None
    }
}

/// Checks a plan's cuts against `set` (RFC-0043 §6.1, §10). `stages` are the stages' layer
/// ranges in ring order (`StageAssignment::layers`), covering the decoder layers and then the
/// `n_mtp` MTP modules; every stage but the first starts at a cut. Contiguity and coverage are
/// `PlacementPlan::validate`'s checks.
///
/// # Errors
///
/// [`BoundaryIllegal`] for the first cut that is not legal, separates an MTP module from the
/// epilogue, or is out of range.
pub fn check_cuts(
    set: &BoundarySet,
    n_mtp: u16,
    stages: &[Range<LayerIx>],
) -> Result<(), BoundaryIllegal> {
    let cuts: Vec<LayerIx> = stages.iter().skip(1).map(|s| s.start).collect();
    check_cut_list(&set.legal, usize::from(n_mtp), &cuts)
}

fn check_cut_list(legal: &[bool], n_mtp: usize, cuts: &[LayerIx]) -> Result<(), BoundaryIllegal> {
    let mut prev = 0usize;
    for &c in cuts {
        let b = usize::from(c.0);
        if let Some(cause) = cut_cause(legal, n_mtp, prev, b) {
            return Err(BoundaryIllegal { boundary: c, cause });
        }
        prev = b;
    }
    Ok(())
}

/// One stage's share of the graph after [`partition`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StagePart {
    /// Its programs, in graph order. The first stage holds any encoders and the prologue; the
    /// last holds the epilogue and the MTP modules.
    pub programs: Vec<ProgramId>,
    /// What its inbound frame carries, in ID order: every value live across its inbound
    /// boundary, which is the previous stage's `sends`. That includes values it only passes
    /// on, such as a per-layer input for a later stage (RFC-0043 §6.1). Empty for the first
    /// stage, which reads token IDs from the step descriptor.
    pub receives: Vec<ValueId>,
    /// The values live across its outbound boundary, in ID order: what its outbound frame
    /// carries. Empty for the last stage.
    pub sends: Vec<ValueId>,
}

/// Splits `g` at the boundaries `cuts`, in increasing order and each accepted by the rule
/// [`check_cuts`] applies against [`ModelGraph::boundaries`]: interior and legal, or
/// `n_layers` when the graph has MTP modules.
///
/// # Errors
///
/// [`BoundaryIllegal`] for the first cut that is not.
pub fn partition(g: &ModelGraph, cuts: &[LayerIx]) -> Result<Vec<StagePart>, BoundaryIllegal> {
    // A boundary set of the wrong size is rule 4's to report; treat its missing entries as
    // illegal.
    let n = g.layers.len();
    let legal: Vec<bool> = (0..=n)
        .map(|b| g.boundaries.legal.get(b).copied().unwrap_or(false))
        .collect();
    check_cut_list(&legal, g.mtp.len(), cuts)?;

    let all = programs(g);
    let live = Liveness::of(g, &all);
    // The stage of each program: encoders and the prologue go to the first stage, layer `l`
    // to the stage whose range holds it, the epilogue and the MTP modules to the last.
    let stage_of = |id: ProgramId| match id {
        ProgramId::Encoder(_) | ProgramId::Prologue => 0,
        ProgramId::Layer(l) => cuts.iter().filter(|c| c.0 <= l).count(),
        ProgramId::Epilogue | ProgramId::Mtp(_) => cuts.len(),
    };
    let mut parts: Vec<StagePart> = (0..=cuts.len())
        .map(|_| StagePart {
            programs: Vec::new(),
            receives: Vec::new(),
            sends: Vec::new(),
        })
        .collect();
    for &(id, _) in &all {
        parts[stage_of(id)].programs.push(id);
    }
    for (k, &c) in cuts.iter().enumerate() {
        let frame = crossing_with(g, &live, usize::from(c.0));
        parts[k + 1].receives.clone_from(&frame);
        parts[k].sends = frame;
    }
    Ok(parts)
}

#[cfg(test)]
mod tests;
