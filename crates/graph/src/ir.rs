// SPDX-License-Identifier: MIT OR Apache-2.0

//! The graph IR of RFC-0043 §2 and ADR-0430: a model is a prologue, a list of layers, an
//! epilogue and optional MTP modules, each a small SSA program of [`OpKind`] nodes over typed
//! values.
//!
//! - This module holds the types of RFC-0043 §2.1, and the `BoundarySet` of §6.1 that
//!   [`ModelGraph`] carries.
//! - [`signature`] holds each kind's value signature, which rule 2 checks.
//! - [`validate`](mod@validate) holds [`validate()`] and rules 1–5 of RFC-0043 §2.2.
//!
//! There are no control-flow nodes and no dynamic shapes beyond [`Dim::Rows`], [`Dim::Seqs`]
//! and [`Dim::Ctx`]. KV and recurrent state live in the KV store, not in the IR (RFC-0043
//! §2.3; ADR-0430 decision 5): a `KvWrite` node has no output value, and an attention or scan
//! node reads its layer's state through the step descriptor.
//!
//! **Values.** [`ValueId`]s are global to one [`ModelGraph`] and index
//! [`ModelGraph::values`], which holds each value's [`ValueType`] and [`ValueKind`].
//! RFC-0043 §2.1 types values but gives the graph no table for those types; liveness across a
//! layer boundary (§6.1) needs one namespace for the whole graph, so the table sits on
//! `ModelGraph`. A program reads values of earlier programs only through its `inputs`.

use smallvec::SmallVec;
use syntropy_core::{DType, LayerIx, TensorGroupId};
use syntropy_kernels::{Fusions, OpKind};

pub mod signature;
pub mod validate;

pub use signature::{Port, Ports, Signature, signature};
pub use validate::{
    CensusGroup, GroupClass, OpSet, ProgramId, Rule, Side, Violation, check, validate,
};

/// One model as a layer program (RFC-0043 §2.1; ADR-0430 decision 1).
#[derive(Clone, Debug, PartialEq)]
pub struct ModelGraph {
    /// The family's frozen tag (RFC-0043 §5.1).
    pub family: &'static str,
    /// Every value of the graph, indexed by [`ValueId`].
    pub values: Vec<Value>,
    /// Embeddings, n-gram and per-layer tables, input scaling.
    pub prologue: Program,
    /// The decoder layers; the index is the [`LayerIx`].
    pub layers: Vec<LayerProgram>,
    /// Final norm, LM head, logit soft-capping.
    pub epilogue: Program,
    /// MTP modules, with `LayerIx` values after the last decoder layer.
    pub mtp: Vec<LayerProgram>,
    /// Reserved (RFC-0043 §2.4): empty for every v1 family.
    pub encoders: Vec<EncoderProgram>,
    /// The residual streams.
    pub residual: ResidualSpec,
    /// The legal stage boundaries and their side data (RFC-0043 §6.1).
    pub boundaries: BoundarySet,
}

/// A value's index in [`ModelGraph::values`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId(pub u32);

/// One entry of [`ModelGraph::values`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Value {
    /// Its type.
    pub ty: ValueType,
    /// What it is to the boundary rule (RFC-0043 §6.1).
    pub kind: ValueKind,
}

/// What a value is to the boundary rule of RFC-0043 §6.1, under which a boundary is legal when
/// every value live across it is a residual stream or declared side data.
///
/// RFC-0043 names the two kinds of side data but not where the IR records them; the family
/// declares each value's kind here when it builds the graph.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValueKind {
    /// Anything else: it must not cross a legal boundary.
    Local,
    /// A residual stream, in [`ResidualSpec::dtype`].
    Residual,
    /// The step's token IDs. No node defines them: the step does, before the prologue, and a
    /// program that needs them lists them in its inputs. Crossing a boundary needs
    /// [`SideData::token_ids`].
    TokenIds,
    /// A per-layer input computed in the prologue for a later layer (`PerLayerEmbed`).
    /// Crossing a boundary needs [`SideData::layer_inputs`] above zero.
    LayerInput,
}

/// Reserved for multimodal input after 1.0 (RFC-0043 §2.4). An encoder turns one non-text
/// input into rows of the text model's hidden size, which the prologue's `EmbedInject` node
/// places.
#[derive(Clone, Debug, PartialEq)]
pub struct EncoderProgram {
    /// The input it encodes.
    pub modality: Modality,
    /// Tower and projector; every weight node has [`NodeClass::Encoder`].
    pub body: Program,
    /// `[Rows, hidden]` in F32.
    pub output: ValueType,
}

/// The input an [`EncoderProgram`] encodes (RFC-0043 §2.4).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Modality {
    /// Images.
    Image,
    /// Audio.
    Audio,
    /// Video.
    Video,
}

/// One decoder layer or MTP module (RFC-0043 §2.1).
#[derive(Clone, Debug, PartialEq)]
pub struct LayerProgram {
    /// Its index: its position in [`ModelGraph::layers`], or for an MTP module the number of
    /// decoder layers plus its position in [`ModelGraph::mtp`].
    pub layer: LayerIx,
    /// Its mixer and FFN.
    pub kind: LayerKind,
    /// The blocks it was built from, typically `[mixer, ffn]`; hyper-connection mixing adds
    /// one.
    pub blocks: SmallVec<[Block; 3]>,
    /// Its program.
    pub body: Program,
}

/// A layer's mixer and FFN (RFC-0043 §2.1, after RFC-0001 §5.3 `HyperParams::layers`).
///
/// RFC-0043 §5.2 has `LayerSpec { mixer: MixerKind, ffn: FfnKind, mtp }` carry the block
/// parameters, and those types come with `HyperParams` in `manifest`, which is still a stub.
/// Until then each side is the [`Block`] that built it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LayerKind {
    /// The mixer block (§3.1).
    pub mixer: Block,
    /// The FFN block (§3.2).
    pub ffn: Block,
}

/// The blocks of RFC-0043 §3, by name. Each block's builder comes with the task that adds it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Block {
    // ---- Mixers (§3.1) ----
    /// Grouped-query attention, covering MHA and MQA.
    Gqa,
    /// Sliding-window GQA, with optional sinks.
    SlidingGqa,
    /// Multi-head latent attention.
    Mla,
    /// Token top-k sparse attention with an indexer.
    SparseIndexed,
    /// Block-sparse attention.
    BlockSparse,
    /// Compressed attention.
    Compressed,
    /// Gated DeltaNet scan.
    GatedDeltaNet,
    /// KDA scan.
    Kda,
    /// Mamba-2 scan.
    Mamba2,
    /// Bidirectional attention of encoder-only embedding models.
    EncoderAttn,
    // ---- Feed-forward (§3.2) ----
    /// Dense gated FFN.
    DenseFfn,
    /// Routed experts, with optional shared experts.
    Moe,
    // ---- Embeddings, head, MTP and residual structure (§3.3) ----
    /// Token embedding.
    Embed,
    /// N-gram embedding.
    NgramEmbed,
    /// Per-layer embedding.
    PerLayerEmbed,
    /// Hyper-connection mixing.
    HyperConn,
    /// Final norm and LM head.
    LmHead,
    /// Pooling epilogue of embedding models.
    Pool,
    /// Segment-type and absolute-position embeddings of encoder models.
    TokenTypeEmbed,
    /// An MTP module.
    Mtp,
}

/// A small SSA program (RFC-0043 §2.1).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Program {
    /// Values of earlier programs it reads: residual streams in, and token IDs where the
    /// program needs them.
    pub inputs: SmallVec<[ValueId; 4]>,
    /// Its nodes, in topological order.
    pub nodes: Vec<Node>,
    /// Values it hands on: for a layer, its residual streams out.
    pub outputs: SmallVec<[ValueId; 4]>,
}

/// One op (RFC-0043 §2.1).
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    /// The op kind (RFC-0040 §3.1).
    pub op: OpKind,
    /// The values it reads.
    pub inputs: SmallVec<[ValueId; 4]>,
    /// The values it defines.
    pub outputs: SmallVec<[ValueId; 2]>,
    /// Which tensor group it reads, and which role inside it.
    pub weights: SmallVec<[WeightRef; 4]>,
    /// Static parameters: eps, RoPE config, window, top-k, scales.
    pub attrs: OpAttrs,
    /// How the step-program lowering places it (ADR-0430 decision 3).
    pub class: NodeClass,
    /// Which neighbours the lowering may fuse into it.
    pub fuse: FuseHints,
}

/// A weight a node reads: a census group and the tensor's role inside it (RFC-0043 §2.1;
/// ADR-0430 decision 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WeightRef {
    /// The group.
    pub group: GroupSel,
    /// The tensor's role in it.
    pub role: TensorRole,
}

/// Which census group a [`WeightRef`] reads (RFC-0043 §2.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GroupSel {
    /// One group: dense groups, heads, routers, tables.
    Fixed(TensorGroupId),
    /// The expert groups of this layer, chosen per step (`MoeGrouped`), so they are never
    /// enumerated in the graph.
    Experts {
        /// The layer.
        layer: LayerIx,
    },
}

/// A tensor's role inside its group (RFC-0043 §2.1).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorRole {
    /// Query projection.
    Q,
    /// Key projection.
    K,
    /// Value projection.
    V,
    /// Fused query, key and value projection.
    Qkv,
    /// Output projection.
    O,
    /// MLA query down projection.
    QA,
    /// MLA query up projection.
    QB,
    /// MLA key-value down projection.
    KvA,
    /// MLA key-value up projection.
    KvB,
    /// FFN gate projection.
    Gate,
    /// FFN up projection.
    Up,
    /// Fused gate and up projection.
    GateUp,
    /// FFN down projection.
    Down,
    /// MoE router.
    Router,
    /// MoE router per-expert bias.
    RouterBias,
    /// A norm's weight.
    Norm,
    /// An embedding table.
    Embed,
    /// The LM head.
    Head,
    /// Sparse-attention indexer weights.
    Indexer,
    /// Scan input projection.
    ScanIn,
    /// Scan gate projection.
    ScanGate,
    /// Scan output projection.
    ScanOut,
    /// Convolution weights.
    Conv,
    /// Any other projection.
    Proj,
    /// A family-specific role.
    Other(u8),
}

/// How the step-program lowering places a node without knowing the architecture (RFC-0043
/// §2.1; ADR-0430 decision 3). For weight-reading nodes it mirrors the census class of the
/// weights.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NodeClass {
    /// Reads `DenseEveryToken` weights.
    Dense,
    /// Reads routed experts (`GroupSel::Experts`).
    Expert,
    /// A MoE router.
    Router,
    /// The LM head or an MTP head.
    Head,
    /// Rows of a table by token ID.
    Lookup,
    /// Norms, residual adds, activations: runs on the device of the neighbouring weight node.
    Elementwise,
    /// KV writes and scans: runs where the layer's KV lives (RFC-0022).
    State,
    /// Reserved (RFC-0043 §2.4): weight-reading nodes of an encoder program.
    Encoder,
}

/// A value's type: a dtype and symbolic dims (RFC-0043 §2.1).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ValueType {
    /// The element type.
    pub dtype: DType,
    /// The dims, outermost first.
    pub dims: SmallVec<[Dim; 4]>,
}

/// One dim of a [`ValueType`] (RFC-0043 §2.1). Lowering a segment to a bucket substitutes
/// `Rows` and `Seqs` with the bucket size; `Ctx` stays dynamic and kernels read it from the
/// step descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dim {
    /// A fixed size.
    Const(u32),
    /// The step's token rows.
    Rows,
    /// The context length.
    Ctx,
    /// The step's sequences.
    Seqs,
}

/// A node's static parameters (RFC-0043 §2.1): eps, RoPE config, window, top-k, scales.
///
/// RFC-0043 does not list the parameters, and the blocks that set them come with later tasks,
/// so this is a list of named values, named the way [`OpKind::dims`] names its dims. Rules 1–5
/// read none of them. Whether RoPE variants and activations are attributes or part of the
/// op key is issue #1610; this type follows the current code, where they are in [`OpKind`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OpAttrs {
    /// The parameters, by name.
    pub entries: SmallVec<[(&'static str, AttrValue); 4]>,
}

impl OpAttrs {
    /// The value named `name`, if set.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<AttrValue> {
        self.entries
            .iter()
            .find(|(key, _)| *key == name)
            .map(|&(_, value)| value)
    }
}

/// One static parameter of a node.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AttrValue {
    /// A count or size.
    U32(u32),
    /// A scale, eps or base.
    F32(f32),
    /// A flag.
    Bool(bool),
}

/// Which neighbours the lowering may fuse into a node (RFC-0043 §2.1), as the fusion bits of
/// RFC-0040's `OpKey::variant`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FuseHints {
    /// The fusions the lowering may choose.
    pub allowed: Fusions,
}

/// The residual streams (RFC-0043 §2.1; RFC-0001 §5.3 `HyperParams::residual_streams`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResidualSpec {
    /// How many there are; above 1 for hyper-connection models.
    pub streams: u8,
    /// Their dtype: F32 (RFC-0042 §4.1).
    pub dtype: DType,
}

/// The legal stage boundaries and the side data that must cross each one (RFC-0043 §6.1).
///
/// Boundary `b` sits between layer `b − 1` and layer `b`; boundaries 0 and `n_layers` always
/// exist. Both vectors have one entry per boundary, `0..=n_layers`. RFC-0043 makes `legal` a
/// bit vector; no bit-vector crate is a dependency, so it is a `Vec<bool>`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BoundarySet {
    /// Whether each boundary is legal.
    pub legal: Vec<bool>,
    /// The side data that crosses each boundary.
    pub side: Vec<SideData>,
}

/// The side data that crosses one boundary (RFC-0043 §6.1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SideData {
    /// Token IDs, for later layers that route by token hash or add n-gram rows.
    pub token_ids: bool,
    /// Per-layer input elements per token, for later layers (`PerLayerEmbed`).
    pub layer_inputs: u32,
    /// The bytes per token this side data adds to the solver's hop size.
    pub bytes_per_token: u32,
}

#[cfg(test)]
mod tests;
