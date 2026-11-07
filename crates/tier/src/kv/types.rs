// SPDX-License-Identifier: MIT OR Apache-2.0

//! RFC-0022 §4's types and RFC-0001 §5.9's `KvHandle` (TIER-E6-T02).

#![forbid(unsafe_code)]

use std::ops::Range;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use syntropy_core::{Blake3, LayerIx, NodeId, PrefixKey, SequenceId, StageIx};

/// Positions per page, `P` (ADR-0205 item 1). `kv.page_tokens` may set 16 to 256, a power of
/// two, per instance; the key arrives with TIER-E6-T12, so [`KvStoreConfig`](super::KvStoreConfig)
/// carries it.
pub const PAGE_TOKENS: u32 = 64;

/// One page of a pool: the tier's node-local ordinal and the page's index in its class's pool
/// on that tier (RFC-0022 §4). 8 bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PageId {
    /// The tier's node-local ordinal.
    pub tier: u8,
    /// The page's index in its pool.
    pub index: u32,
}

/// Layers whose pages advance in lockstep and share one block table (RFC-0022 §3; ADR-0205
/// item 2). `Full`, `Windowed` and `State` are built here; `Compressed` and `Aux` come with
/// TIER-E6-T16 and TIER-E6-T08.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PageClass {
    /// `PerToken { window: None }` and `Sparse` layers: indexed by token page `⌊pos / P⌋`.
    Full,
    /// `PerToken { window: Some(w) }` layers, one class per distinct `w`: a ring of
    /// `⌈w / P⌉ + 1` pages.
    Windowed {
        /// The window, in tokens.
        window: u32,
    },
    /// `Compressed { tokens_per_entry: r }` layers, one class per distinct `r`.
    Compressed {
        /// Tokens per entry.
        ratio: u16,
    },
    /// `Fixed` layers: one page per sequence holds each layer's whole state.
    State,
    /// `AuxKv` caches.
    Aux,
}

/// A page in a block table. `shared` marks copy-on-write: the page is held by more than one
/// table, so the next write to it copies it first (RFC-0022 §4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PageRef {
    /// The page.
    pub page: PageId,
    /// Held by more than one table.
    pub shared: bool,
}

/// One class's block table of one sequence: its pages in token order, or for `Windowed` the
/// ring, slot `⌊pos / P⌋ mod ring` (RFC-0022 §3, §4). Its capacity is fixed when the sequence
/// slot is made, so appends never allocate.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassTable {
    /// The class.
    pub class: PageClass,
    /// The pages. A `Windowed` ring keeps slots that left the window until they are reused.
    pub pages: Vec<PageRef>,
    /// Positions appended but not committed (speculative windows, RFC-0022 §6; TIER-E6-T09).
    /// Always 0 here.
    pub provisional: u16,
}

/// Per sequence and stage: one block table per class (RFC-0022 §4).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageTable {
    /// One table per class of the store, in the store's class order.
    pub classes: SmallVec<[ClassTable; 5]>,
}

/// `KvHandle::pages`: the index of the sequence slot's [`PageTable`] in the stage's store.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PageTableId(pub u32);

/// The prefix index's tail-pool index: `core`'s type (DIST-E1-T14), re-exported.
pub use syntropy_core::TailId;

/// A run of a sequence's pages in one class, in token order (RFC-0022 §4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PageSpan {
    /// The class.
    pub class: PageClass,
    /// The first page's index in the class's block table.
    pub first_page: u32,
    /// Pages in the run.
    pub pages: u32,
}

/// A tier move in flight; its completion arrives on the dispatcher's ring (RFC-0022 §4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MoveTicket(pub u64);

/// Where a restore reads a session blob from (RFC-0022 §4, §8.4).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionSource {
    /// A blob on a local volume.
    Local {
        /// The blob's manifest.
        manifest: PathBuf,
    },
    /// A blob held by other nodes.
    Remote {
        /// The holders.
        holders: SmallVec<[NodeId; 4]>,
        /// The manifest's hash.
        manifest: Blake3,
    },
}

/// One chunk of an export: a layer's token range in one class (RFC-0022 §4, §10).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvChunkDesc {
    /// The layer.
    pub layer: LayerIx,
    /// The class.
    pub class: PageClass,
    /// The first token.
    pub first_token: u32,
    /// Tokens in the chunk.
    pub n_tokens: u32,
    /// The pages that hold them.
    pub pages: PageSpan,
}

/// Sealed pages ready to stream as `KvChunk`s (RFC-0022 §4, §10). RFC-0022 holds RFC-0020 §7.4's
/// counted leases here; until those exist (TIER-E1-T05), `leases` names the pages whose
/// reference the export holds.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvExport {
    /// The sequence.
    pub seq: SequenceId,
    /// The chunks, in order.
    pub chunks: Vec<KvChunkDesc>,
    /// The token the export stops at.
    pub cut_at: u32,
    /// The pages the export keeps alive.
    pub leases: Vec<PageId>,
}

/// `KvHandle::residency`: pages per tier (RFC-0022 §4).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KvResidency {
    /// Pages in `Vram` (the hot window).
    pub vram: u32,
    /// Pages in `HostPinned`.
    pub host: u32,
    /// Pages on `Nvme`.
    pub nvme: u32,
    /// Pages moving between tiers.
    pub moving: u32,
}

/// Free capacity of a stage's KV store in tokens, per tier `[Vram, HostPinned, Nvme]`: the
/// minimum over the token-indexed classes of free pages × P (RFC-0022 §11). `Windowed`, `State`
/// and `Aux` never limit it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KvHeadroom {
    /// Tokens per tier: `[Vram, HostPinned, Nvme]`.
    pub tokens: [u32; 3],
}

/// One sequence's KV and recurrent state on one stage (RFC-0001 §5.9). Not `Clone`: the store
/// takes it back by value in [`release`](super::KvStore::release).
#[derive(Debug, PartialEq, Eq)]
pub struct KvHandle {
    /// The sequence.
    pub seq: SequenceId,
    /// The stage.
    pub stage: StageIx,
    /// The stage's layers.
    pub layers: Range<LayerIx>,
    /// Committed positions.
    pub tokens: u32,
    /// The sequence slot's block tables.
    pub pages: PageTableId,
    /// Pages per tier.
    pub residency: KvResidency,
    /// The shared prefix this sequence extends (RFC-0022 §7.1).
    pub prefix: Option<PrefixKey>,
}

/// A copy the caller must issue before writing: the store gave a sequence its own page in
/// place of a shared one (copy-on-write), and the first `tokens` positions of `from`, or the
/// whole page for `State`, belong in `to` (RFC-0022 §3, §4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageCopy {
    /// The class.
    pub class: PageClass,
    /// The shared page.
    pub from: PageId,
    /// The sequence's new page.
    pub to: PageId,
    /// Positions to copy from the page's start; for `State`, 0 means the whole state.
    pub tokens: u32,
}

/// What [`append`](super::KvStore::append) returns (RFC-0001 §5.9's `KvWriteSlots`): the
/// positions to write, the block tables that place them, and the copies to issue first. Holds
/// at most one copy per class inline, so it never allocates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvWriteSlots {
    /// The positions appended.
    pub positions: Range<u32>,
    /// The sequence slot whose block tables give each position's page.
    pub table: PageTableId,
    /// Copies to issue, in order, before the step writes.
    pub copies: SmallVec<[PageCopy; 5]>,
}
