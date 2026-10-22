// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Activation` frame: its extension, spans, tree area and side-channel TLVs, its F32 and
//! BF16 payloads, and the parts a chunk larger than `max_frame` travels in (RFC-0031 §6.3, §6.4).
//!
//! The extension is a 64-byte fixed part followed by three areas, then padding:
//!
//! ```text
//! | fixed (64 B) | spans (n_spans × 32 B) | tree area (tree_len B) | side TLVs (side_len B) | pad |
//! ```
//!
//! The payload is `n_rows` rows, each `streams × hidden` elements of the frame's dtype laid out
//! `[stream][hidden]`, with the row stride rounded up to 64 bytes. This codec carries F32 and
//! BF16, the dtypes exact mode may send (ADR-0010 item 5, ADR-0019 item 5). The approximate
//! containers `F16`, `Q8B32` and `F8E4M3B32` are refused as `Unsupported` until DIST-E8-T11
//! adds them.
//!
//! [`Frame::activation`] reads a frame against its channel's wire dtype and refuses one in
//! another dtype with [`FrameError::WireDTypeMismatch`], code 0x0206. [`Activation::encode`]
//! writes one frame, and [`Activation::parts`] with [`Activation::encode_part`] splits a chunk
//! into parts at row boundaries (RFC-0031 §6.3, revised 2026-10-22). None of them allocates: a
//! decoded frame borrows its spans, tree and items from the frame's bytes, and an encoded one is
//! written straight into the caller's buffer.
//!
//! Side items 1–7 are typed here. Items 8–15 (`KV_HEADROOM` and the prefix-cache items) are
//! DIST-E8-T04's; until then they pass through as [`SideItem::Other`], as does any type a later
//! minor adds (RFC-0031 §10). Reserved fields are written as zero and ignored on receive, and so
//! are extension bytes past the side area.

use std::fmt;
use std::ops::Range;

use syntropy_core::plan::WireDType;
use syntropy_core::{InstanceId, PlanEpoch, SequenceId, SessionId, StageIx};

use super::error::FrameError;
use super::header::{self, ALIGN, Flags, Frame, FrameSpec, FrameType, HEADER_LEN};

/// The extension's fixed part, before the spans.
pub const ACTIVATION_FIXED_LEN: usize = 64;

/// One span on the wire.
pub const SPAN_LEN: usize = 32;

/// One tree node on the wire.
pub const TREE_NODE_LEN: usize = 4;

/// A side TLV's header: `type: u16`, `span: u16`, `len: u32`. Each TLV starts 8-byte aligned in
/// the side area, its value padded to 8.
pub const TLV_HEADER_LEN: usize = 8;

/// The side-channel TLV types of RFC-0031 §6.3.
pub mod tlv {
    pub const TOKEN_IDS: u16 = 1;
    pub const ROUTER_HINT: u16 = 2;
    pub const SEQ_META: u16 = 3;
    pub const SEQ_CTRL: u16 = 4;
    pub const TRACE: u16 = 5;
    pub const TRUNCATE: u16 = 6;
    pub const DRAFT_CONF: u16 = 7;
    pub const KV_HEADROOM: u16 = 8;
    pub const PAGE_KEYS: u16 = 9;
    pub const PREFIX_EVICT: u16 = 10;
    pub const PREFIX_DROPPED: u16 = 11;
    pub const PREFIX_PRESSURE: u16 = 12;
    pub const PREFIX_MISS: u16 = 13;
    pub const RESTART: u16 = 14;
    pub const SESSION_SAVED: u16 = 15;
}

/// `span` of an item for every span, `tree_offset` of a span that is not a tree, and `parent` of
/// a node whose parent is the root.
pub(super) const ABSENT: u16 = 0xFFFF;

/// The `TRACE` value: W3C trace context.
const TRACE_LEN: usize = 26;

/// What an `Activation` frame carries: `kind` with its `kind_arg` and `kind_flags` (RFC-0031
/// §6.3; RFC-0001 §5.8's `FrameKind`). The other `kind_flags` bits are reserved, and `kind_arg`
/// is 0 for `Decode` and `Canary`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ActivationKind {
    /// A prefill chunk: `kind_arg` is its index; flag bit0 `LAST_CHUNK`.
    Prefill { chunk: u32, last: bool },
    /// Decode rows.
    Decode,
    /// A verify window: `kind_arg` is the window, γ + 1 positions or the tree's nodes; flag bit0
    /// `TREE`, set when the frame has a tree area.
    Verify { window: u32, tree: bool },
    /// Activation-history replay: `kind_arg` is the replay id; flag bit0 `LAST`, bit1 `FORWARD`
    /// (RFC-0033 §4.2).
    Replay { id: u32, last: bool, forward: bool },
    /// A warm-up check.
    Canary,
}

impl ActivationKind {
    fn code(self) -> u8 {
        match self {
            ActivationKind::Prefill { .. } => 1,
            ActivationKind::Decode => 2,
            ActivationKind::Verify { .. } => 3,
            ActivationKind::Replay { .. } => 4,
            ActivationKind::Canary => 5,
        }
    }

    fn arg(self) -> u32 {
        match self {
            ActivationKind::Prefill { chunk, .. } => chunk,
            ActivationKind::Verify { window, .. } => window,
            ActivationKind::Replay { id, .. } => id,
            ActivationKind::Decode | ActivationKind::Canary => 0,
        }
    }

    fn flags(self) -> u32 {
        match self {
            ActivationKind::Prefill { last, .. } => u32::from(last),
            ActivationKind::Verify { tree, .. } => u32::from(tree),
            ActivationKind::Replay { last, forward, .. } => {
                u32::from(last) | u32::from(forward) << 1
            }
            ActivationKind::Decode | ActivationKind::Canary => 0,
        }
    }

    fn from_wire(code: u8, arg: u32, flags: u32) -> Option<ActivationKind> {
        Some(match code {
            1 => ActivationKind::Prefill {
                chunk: arg,
                last: flags & 1 != 0,
            },
            2 => ActivationKind::Decode,
            3 => ActivationKind::Verify {
                window: arg,
                tree: flags & 1 != 0,
            },
            4 => ActivationKind::Replay {
                id: arg,
                last: flags & 1 != 0,
                forward: flags & 2 != 0,
            },
            5 => ActivationKind::Canary,
            _ => return None,
        })
    }

    /// Whether the frame has a tree area: a verify window with `TREE`.
    pub fn is_tree(self) -> bool {
        matches!(self, ActivationKind::Verify { tree: true, .. })
    }
}

/// The wire code of a dtype (RFC-0031 §6.4).
fn dtype_code(dtype: WireDType) -> Option<u8> {
    match dtype {
        WireDType::F32 => Some(1),
        WireDType::BF16 => Some(2),
        WireDType::F16 => Some(3),
        WireDType::Q8B32 => Some(4),
        WireDType::F8E4M3B32 => Some(5),
        _ => None,
    }
}

fn dtype_from_code(code: u8) -> Option<WireDType> {
    match code {
        1 => Some(WireDType::F32),
        2 => Some(WireDType::BF16),
        3 => Some(WireDType::F16),
        4 => Some(WireDType::Q8B32),
        5 => Some(WireDType::F8E4M3B32),
        _ => None,
    }
}

/// How the payload's rows are laid out: the dtype, and `streams × hidden` elements per row
/// (RFC-0031 §6.3, §6.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RowLayout {
    pub dtype: WireDType,
    /// Residual streams: 1, or more for hyper-connection models.
    pub streams: u8,
    pub hidden: u32,
}

impl RowLayout {
    /// Bytes per element of the dtypes this codec carries: 4 for F32, 2 for BF16.
    pub fn element_len(&self) -> Option<u64> {
        match self.dtype {
            WireDType::F32 => Some(4),
            WireDType::BF16 => Some(2),
            _ => None,
        }
    }

    /// Bytes of one row's elements.
    pub fn row_len(&self) -> Option<u64> {
        Some(u64::from(self.streams) * u64::from(self.hidden) * self.element_len()?)
    }

    /// The row stride: a row's bytes rounded up to 64.
    pub fn stride(&self) -> Option<u64> {
        Some(self.row_len()?.next_multiple_of(ALIGN as u64))
    }
}

/// The fixed fields of the extension. `n_spans`, `tree_len` and `side_len` follow from the
/// areas when the frame is encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ActivationHeader {
    pub from_stage: StageIx,
    pub to_stage: StageIx,
    pub kind: ActivationKind,
    pub layout: RowLayout,
    /// Token positions carried in the payload.
    pub n_rows: u32,
    /// 0-based; a chunk larger than `max_frame` travels in parts.
    pub part_index: u16,
    /// At least 1.
    pub part_count: u16,
}

/// One sequence segment of the frame's rows (RFC-0031 §6.3). The spans are in row order and
/// cover the rows without gaps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Span {
    pub seq: SequenceId,
    /// The position of the span's first row.
    pub first_position: u32,
    /// Rows belonging to the span. A span may hold none, to carry a sequence's side items on a
    /// frame without its rows, such as a `SEQ_CTRL` on an empty frame (RFC-0032 §3.1).
    pub n_positions: u32,
    /// The index of the span's first row in the payload.
    pub row_offset: u32,
    /// The span's draft tree, in a verify frame with `TREE`.
    pub tree: Option<SpanTree>,
}

/// Where a span's tree sits in the tree area: one node per row of the span, in row order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpanTree {
    /// The index of the span's first node in the tree area.
    pub offset: u16,
    pub nodes: u16,
}

/// One node of a draft tree (RFC-0031 §6.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TreeNode {
    /// The parent's index among the span's nodes, or `None` (0xFFFF) when the parent is the
    /// root, the last accepted token. A parent precedes its children.
    pub parent: Option<u16>,
    pub depth: u8,
    /// bit0 [`TreeNode::PROMPT_LOOKUP`]; the other bits are reserved.
    pub flags: u8,
}

impl TreeNode {
    /// The node comes from the prompt-lookup drafter.
    pub const PROMPT_LOOKUP: u8 = 1;

    pub fn prompt_lookup(&self) -> bool {
        self.flags & Self::PROMPT_LOOKUP != 0
    }
}

/// A side-channel item and the span it applies to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Side<'a> {
    /// The span's index in the frame; `None` (0xFFFF) for every span.
    pub span: Option<u16>,
    pub item: SideItem<'a>,
}

/// A side-channel item (RFC-0031 §6.3). Arrays are little-endian bytes, as on the wire, so a
/// decoded item borrows them from the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SideItem<'a> {
    /// `TOKEN_IDS`: a u32 per row of the span, or of the frame for every span.
    TokenIds(&'a [u8]),
    /// `ROUTER_HINT`: for `layer`, `k` predicted experts per row as u16s, row by row.
    RouterHint {
        layer: u16,
        k: u8,
        experts: &'a [u8],
    },
    /// `SEQ_META`: the CBOR `SequenceMeta` of RFC-0032 §3.6, which DIST-E8-T05 decodes.
    SeqMeta(&'a [u8]),
    /// `SEQ_CTRL`: sequence control riding the ring.
    SeqCtrl(SeqCtrl),
    /// `TRACE`: the W3C trace context of a sampled frame.
    Trace(TraceContext),
    /// `TRUNCATE`: drop the span's KV from position `to` on (RFC-0032 §3.5).
    Truncate { to: u32 },
    /// `DRAFT_CONF`: an f16 per tree node of the span, or of the frame for every span.
    DraftConf(&'a [u8]),
    /// Any other type: items 8–15, which DIST-E8-T04 decodes, and types a later minor adds,
    /// which a receiver skips (RFC-0031 §10).
    Other { ty: u16, value: &'a [u8] },
}

/// A `SEQ_CTRL` code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SeqCtrl {
    /// 1.
    Cancel,
    /// 2.
    Release,
    /// 3: release and persist the KV as `session`.
    ReleasePersist { session: SessionId },
    /// 4 or 5, reserved for Pause and Resume, which stop at the head and are never sent.
    /// Receivers ignore them.
    Reserved(u8),
}

/// W3C trace context: version, trace-id, span-id and flags, 26 bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TraceContext {
    pub version: u8,
    pub trace_id: [u8; 16],
    pub span_id: [u8; 8],
    pub flags: u8,
}

pub(super) fn get16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

pub(super) fn get32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

pub(super) fn get64(b: &[u8], at: usize) -> u64 {
    let mut word = [0; 8];
    word.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(word)
}

fn bad(field: &'static str, value: impl Into<u64>) -> FrameError {
    FrameError::BadField {
        ftype: FrameType::Activation,
        field,
        value: value.into(),
    }
}

fn inconsistent(
    field: &'static str,
    value: impl Into<u64>,
    expected: impl Into<u64>,
) -> FrameError {
    FrameError::Inconsistent {
        ftype: FrameType::Activation,
        field,
        value: value.into(),
        expected: expected.into(),
    }
}

fn too_short(len: usize, needed: usize) -> FrameError {
    FrameError::ExtTooShort {
        ftype: FrameType::Activation,
        len,
        needed,
    }
}

fn read_span(b: &[u8]) -> Span {
    let offset = get16(b, 28);
    Span {
        seq: SequenceId {
            ingress: get64(b, 0),
            boot: get32(b, 8),
            n: get32(b, 12),
        },
        first_position: get32(b, 16),
        n_positions: get32(b, 20),
        row_offset: get32(b, 24),
        tree: (offset != ABSENT).then(|| SpanTree {
            offset,
            nodes: get16(b, 30),
        }),
    }
}

fn read_node(b: &[u8]) -> TreeNode {
    let parent = get16(b, 0);
    TreeNode {
        parent: (parent != ABSENT).then_some(parent),
        depth: b[2],
        flags: b[3],
    }
}

/// Reads the TLV at `at` in the side area and returns it with the offset of the next one. The
/// `Token` frame's TLV area uses the same encoding (RFC-0031 §6.5).
pub(super) fn read_tlv(side: &[u8], at: usize) -> Result<(Side<'_>, usize), FrameError> {
    let remaining = side.len().saturating_sub(at);
    if remaining < TLV_HEADER_LEN {
        return Err(inconsistent(
            "side_len",
            side.len() as u64,
            (at + TLV_HEADER_LEN) as u64,
        ));
    }
    let ty = get16(side, at);
    let span = get16(side, at + 2);
    let len = get32(side, at + 4) as usize;
    let start = at + TLV_HEADER_LEN;
    let padded = len.checked_next_multiple_of(8).unwrap_or(usize::MAX);
    if padded > side.len() - start {
        return Err(inconsistent(
            "side_len",
            side.len() as u64,
            (start as u64).saturating_add(padded as u64),
        ));
    }
    let value = &side[start..start + len];
    let item = match ty {
        tlv::TOKEN_IDS => SideItem::TokenIds(value),
        tlv::ROUTER_HINT => {
            if len < 4 {
                return Err(bad("ROUTER_HINT len", len as u64));
            }
            SideItem::RouterHint {
                layer: get16(value, 0),
                k: value[2],
                experts: &value[4..],
            }
        }
        tlv::SEQ_META => SideItem::SeqMeta(value),
        tlv::SEQ_CTRL => SideItem::SeqCtrl(read_seq_ctrl(value)?),
        tlv::TRACE => {
            if len != TRACE_LEN {
                return Err(bad("TRACE len", len as u64));
            }
            let mut trace_id = [0; 16];
            trace_id.copy_from_slice(&value[1..17]);
            let mut span_id = [0; 8];
            span_id.copy_from_slice(&value[17..25]);
            SideItem::Trace(TraceContext {
                version: value[0],
                trace_id,
                span_id,
                flags: value[25],
            })
        }
        tlv::TRUNCATE => {
            if len != 4 {
                return Err(bad("TRUNCATE len", len as u64));
            }
            SideItem::Truncate {
                to: get32(value, 0),
            }
        }
        tlv::DRAFT_CONF => SideItem::DraftConf(value),
        ty => SideItem::Other { ty, value },
    };
    let span = (span != ABSENT).then_some(span);
    Ok((Side { span, item }, start + padded))
}

fn read_seq_ctrl(value: &[u8]) -> Result<SeqCtrl, FrameError> {
    let code = *value.first().ok_or(bad("SEQ_CTRL len", 0u64))?;
    let len = value.len();
    match code {
        1 | 2 if len != 8 => Err(bad("SEQ_CTRL len", len as u64)),
        1 => Ok(SeqCtrl::Cancel),
        2 => Ok(SeqCtrl::Release),
        3 if len != 24 => Err(bad("SEQ_CTRL len", len as u64)),
        3 => {
            let mut session = [0; 16];
            session.copy_from_slice(&value[8..24]);
            Ok(SeqCtrl::ReleasePersist {
                session: SessionId(session),
            })
        }
        // Reserved, and ignored whatever follows the code.
        4 | 5 => Ok(SeqCtrl::Reserved(code)),
        other => Err(bad("SEQ_CTRL code", other)),
    }
}

/// The spans of a frame: read from its bytes, or given by a sender.
#[derive(Clone, Copy)]
pub struct Spans<'a>(Repr<'a, Span>);

/// The tree area of a frame: read from its bytes, or given by a sender.
#[derive(Clone, Copy)]
pub struct TreeArea<'a>(Repr<'a, TreeNode>);

/// The side-channel items of a frame, in order: read from its bytes, or given by a sender.
#[derive(Clone, Copy)]
pub struct SideItems<'a>(Repr<'a, Side<'a>>);

/// Wire bytes, already checked when the frame was parsed, or a sender's values.
#[derive(Clone, Copy)]
enum Repr<'a, T> {
    Wire(&'a [u8]),
    Typed(&'a [T]),
}

impl<'a> Spans<'a> {
    pub fn new(spans: &'a [Span]) -> Spans<'a> {
        Spans(Repr::Typed(spans))
    }

    pub fn len(&self) -> usize {
        match self.0 {
            Repr::Wire(bytes) => bytes.len() / SPAN_LEN,
            Repr::Typed(spans) => spans.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> Option<Span> {
        match self.0 {
            Repr::Wire(bytes) => {
                let at = index.checked_mul(SPAN_LEN)?;
                bytes.get(at..at.checked_add(SPAN_LEN)?).map(read_span)
            }
            Repr::Typed(spans) => spans.get(index).copied(),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = Span> + 'a {
        let spans = *self;
        (0..spans.len()).filter_map(move |index| spans.get(index))
    }
}

impl<'a> TreeArea<'a> {
    pub fn new(nodes: &'a [TreeNode]) -> TreeArea<'a> {
        TreeArea(Repr::Typed(nodes))
    }

    /// Nodes on the wire, 4 bytes each, already checked.
    pub(super) fn wire(bytes: &'a [u8]) -> TreeArea<'a> {
        TreeArea(Repr::Wire(bytes))
    }

    pub fn len(&self) -> usize {
        match self.0 {
            Repr::Wire(bytes) => bytes.len() / TREE_NODE_LEN,
            Repr::Typed(nodes) => nodes.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> Option<TreeNode> {
        match self.0 {
            Repr::Wire(bytes) => {
                let at = index.checked_mul(TREE_NODE_LEN)?;
                bytes.get(at..at.checked_add(TREE_NODE_LEN)?).map(read_node)
            }
            Repr::Typed(nodes) => nodes.get(index).copied(),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = TreeNode> + 'a {
        let tree = *self;
        (0..tree.len()).filter_map(move |index| tree.get(index))
    }
}

impl<'a> SideItems<'a> {
    pub fn new(items: &'a [Side<'a>]) -> SideItems<'a> {
        SideItems(Repr::Typed(items))
    }

    /// TLVs on the wire, already walked with `read_tlv`.
    pub(super) fn wire(bytes: &'a [u8]) -> SideItems<'a> {
        SideItems(Repr::Wire(bytes))
    }

    pub fn iter(&self) -> impl Iterator<Item = Side<'a>> + 'a {
        let repr = self.0;
        let mut at = 0;
        std::iter::from_fn(move || match repr {
            Repr::Wire(bytes) if at < bytes.len() => {
                let (side, next) = read_tlv(bytes, at).ok()?;
                at = next;
                Some(side)
            }
            Repr::Wire(_) => None,
            Repr::Typed(items) => {
                let side = items.get(at).copied();
                at += 1;
                side
            }
        })
    }
}

impl PartialEq for Spans<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl PartialEq for TreeArea<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl PartialEq for SideItems<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl fmt::Debug for Spans<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl fmt::Debug for TreeArea<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl fmt::Debug for SideItems<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// The header fields of an unsealed `Activation` frame. `flags` bit0 `HAS_TRACE` is set from the
/// side items, as the lengths and CRCs follow from the frame; `ENCRYPTED` is the AEAD layer's,
/// which seals the extension that [`Activation::write_ext`] writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ActivationSpec {
    pub wire_minor: u8,
    pub flags: Flags,
    pub instance: InstanceId,
    pub epoch: PlanEpoch,
    pub step: u64,
    pub chan_seq: u64,
}

/// One part of a chunk (RFC-0031 §6.3): its index, the part count and the chunk's rows it holds.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Part {
    pub index: u16,
    pub count: u16,
    /// The chunk's rows this part carries.
    pub rows: Range<u32>,
    /// The part's whole frame, header and padding included.
    pub frame_len: usize,
}

/// The parts of a chunk, in order: see [`Activation::parts`]. Its length is the part count.
#[derive(Clone, Debug)]
pub struct Parts<'c, 'a> {
    chunk: &'c Activation<'a>,
    max_frame: usize,
    next: u32,
    index: u16,
    count: u16,
}

impl ExactSizeIterator for Parts<'_, '_> {}

impl Iterator for Parts<'_, '_> {
    type Item = Part;

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = usize::from(self.count - self.index);
        (left, Some(left))
    }

    fn next(&mut self) -> Option<Part> {
        if self.index == self.count {
            return None;
        }
        let (end, frame_len) = self.chunk.fit(self.next, self.max_frame).ok()?;
        let part = Part {
            index: self.index,
            count: self.count,
            rows: self.next..end,
            frame_len,
        };
        self.next = end;
        self.index += 1;
        Some(part)
    }
}

/// A run of a chunk's rows, encoded as one frame.
#[derive(Clone, Copy, Debug)]
struct Window {
    start: u32,
    end: u32,
    part_index: u16,
    part_count: u16,
}

/// The areas of one frame's extension, before padding.
#[derive(Clone, Copy, Debug)]
struct Areas {
    n_spans: usize,
    tree_len: usize,
    side_len: usize,
    has_trace: bool,
}

impl Areas {
    fn ext_len(&self) -> usize {
        ACTIVATION_FIXED_LEN + self.n_spans * SPAN_LEN + self.tree_len + self.side_len
    }
}

/// Writes into an extension buffer, or only counts when it has none.
struct Sink<'o>(Option<&'o mut [u8]>);

impl Sink<'_> {
    fn put(&mut self, at: usize, bytes: &[u8]) {
        if let Some(buf) = self.0.as_deref_mut() {
            buf[at..at + bytes.len()].copy_from_slice(bytes);
        }
    }
}

/// A frame's `Activation`: the fixed fields, the three areas and the payload.
///
/// A sender builds one from slices with [`Spans::new`], [`TreeArea::new`] and
/// [`SideItems::new`]; [`Frame::activation`] reads one that borrows the frame's bytes. Two
/// compare equal when their fields and contents do, however each was made.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Activation<'a> {
    pub header: ActivationHeader,
    pub spans: Spans<'a>,
    pub tree: TreeArea<'a>,
    pub side: SideItems<'a>,
    /// `n_rows` rows at the layout's stride.
    pub payload: &'a [u8],
}

impl<'a> Activation<'a> {
    /// Reads an unsealed `Activation` extension and payload, on a channel whose wire dtype is
    /// `wire` (`ChannelSpec::wire`). A frame in another dtype is refused with
    /// [`FrameError::WireDTypeMismatch`], whatever else it holds.
    pub fn parse(
        ext: &'a [u8],
        payload: &'a [u8],
        wire: WireDType,
    ) -> Result<Activation<'a>, FrameError> {
        if ext.len() < ACTIVATION_FIXED_LEN {
            return Err(too_short(ext.len(), ACTIVATION_FIXED_LEN));
        }
        let kind = ActivationKind::from_wire(ext[4], get32(ext, 24), get32(ext, 28))
            .ok_or(bad("kind", ext[4]))?;
        let dtype = dtype_from_code(ext[5]).ok_or(bad("dtype", ext[5]))?;
        if dtype != wire {
            return Err(FrameError::WireDTypeMismatch {
                expected: wire,
                found: dtype,
            });
        }
        let header = ActivationHeader {
            from_stage: StageIx(get16(ext, 0)),
            to_stage: StageIx(get16(ext, 2)),
            kind,
            layout: RowLayout {
                dtype,
                streams: ext[6],
                hidden: get32(ext, 8),
            },
            n_rows: get32(ext, 12),
            part_index: get16(ext, 18),
            part_count: get16(ext, 20),
        };
        let n_spans = usize::from(get16(ext, 16));
        let tree_len = get32(ext, 32) as usize;
        let side_len = get32(ext, 36) as usize;
        if !tree_len.is_multiple_of(TREE_NODE_LEN) {
            return Err(bad("tree_len", tree_len as u64));
        }
        if !side_len.is_multiple_of(8) {
            return Err(bad("side_len", side_len as u64));
        }
        let spans_end = ACTIVATION_FIXED_LEN + n_spans * SPAN_LEN;
        let tree_end = spans_end.saturating_add(tree_len);
        let side_end = tree_end.saturating_add(side_len);
        if ext.len() < side_end {
            return Err(too_short(ext.len(), side_end));
        }
        let spans = &ext[ACTIVATION_FIXED_LEN..spans_end];
        for span in spans.as_chunks::<SPAN_LEN>().0 {
            if get16(span, 28) == ABSENT && get16(span, 30) != 0 {
                return Err(bad("tree_nodes", get16(span, 30)));
            }
        }
        let side = &ext[tree_end..side_end];
        let mut at = 0;
        while at < side.len() {
            at = read_tlv(side, at)?.1;
        }
        let activation = Activation {
            header,
            spans: Spans(Repr::Wire(spans)),
            tree: TreeArea(Repr::Wire(&ext[spans_end..tree_end])),
            side: SideItems(Repr::Wire(side)),
            payload,
        };
        activation.validate()?;
        Ok(activation)
    }

    /// Checks the rules the bytes cannot express:
    /// - the dtype is F32 or BF16, `streams` and `hidden` are not 0, and the payload is `n_rows`
    ///   strides long;
    /// - `part_index < part_count`;
    /// - the spans cover the rows in order without gaps, and their trees cover the tree area the
    ///   same way, one node per row, in a verify frame with `TREE` only;
    /// - each node's parent precedes it in its span's tree;
    /// - each side item names a span of the frame, and an item sized by rows or nodes has the
    ///   rows or nodes of its span: `TOKEN_IDS`, `ROUTER_HINT` and `DRAFT_CONF`, which needs a
    ///   tree.
    pub fn validate(&self) -> Result<(), FrameError> {
        let header = &self.header;
        let layout = header.layout;
        if layout.streams == 0 {
            return Err(bad("streams", 0u64));
        }
        if layout.hidden == 0 {
            return Err(bad("hidden", 0u64));
        }
        let stride = layout
            .stride()
            .ok_or(FrameError::UnsupportedDType(layout.dtype))?;
        if header.part_count == 0 {
            return Err(bad("part_count", 0u64));
        }
        if header.part_index >= header.part_count {
            return Err(bad("part_index", header.part_index));
        }
        if self.spans.len() > usize::from(u16::MAX) {
            return Err(FrameError::TooLong {
                what: "spans",
                len: self.spans.len(),
            });
        }
        let tree_frame = header.kind.is_tree();
        let mut rows = 0u64;
        let mut nodes = 0u64;
        for span in self.spans.iter() {
            if u64::from(span.row_offset) != rows {
                return Err(inconsistent("row_offset", span.row_offset, rows));
            }
            rows += u64::from(span.n_positions);
            let Some(tree) = span.tree else { continue };
            if !tree_frame || tree.offset == ABSENT {
                return Err(bad("tree_offset", tree.offset));
            }
            if u64::from(tree.offset) != nodes {
                return Err(inconsistent("tree_offset", tree.offset, nodes));
            }
            if u32::from(tree.nodes) != span.n_positions {
                return Err(inconsistent("tree_nodes", tree.nodes, span.n_positions));
            }
            for index in 0..tree.nodes {
                let node = self
                    .tree
                    .get(usize::from(tree.offset) + usize::from(index))
                    .ok_or(inconsistent(
                        "tree_len",
                        (self.tree.len() * TREE_NODE_LEN) as u64,
                        (nodes + u64::from(tree.nodes)) * TREE_NODE_LEN as u64,
                    ))?;
                if node.parent.is_some_and(|parent| parent >= index) {
                    return Err(bad("parent", node.parent.unwrap_or(ABSENT)));
                }
            }
            nodes += u64::from(tree.nodes);
        }
        if rows != u64::from(header.n_rows) {
            return Err(inconsistent("n_rows", header.n_rows, rows));
        }
        if nodes != self.tree.len() as u64 {
            return Err(inconsistent(
                "tree_len",
                (self.tree.len() * TREE_NODE_LEN) as u64,
                nodes * TREE_NODE_LEN as u64,
            ));
        }
        for side in self.side.iter() {
            self.check_side(&side)?;
        }
        let expected = u64::from(header.n_rows).saturating_mul(stride);
        if self.payload.len() as u64 != expected {
            return Err(inconsistent(
                "payload_len",
                self.payload.len() as u64,
                expected,
            ));
        }
        Ok(())
    }

    fn check_side(&self, side: &Side<'_>) -> Result<(), FrameError> {
        let span = match side.span {
            None => None,
            Some(index) => Some(
                self.spans
                    .get(usize::from(index))
                    .filter(|_| index != ABSENT)
                    .ok_or(bad("TLV span", index))?,
            ),
        };
        let rows = span.map_or(self.header.n_rows, |span| span.n_positions);
        let rows = u64::from(rows);
        let expect = |what: &'static str, len: usize, per_row: u64, rows: u64| {
            if len as u64 == per_row * rows {
                Ok(())
            } else {
                Err(inconsistent(what, len as u64, per_row * rows))
            }
        };
        match side.item {
            SideItem::TokenIds(ids) => expect("TOKEN_IDS len", ids.len(), 4, rows),
            SideItem::RouterHint { k, experts, .. } => {
                expect("ROUTER_HINT len", experts.len(), 2 * u64::from(k), rows)
            }
            SideItem::DraftConf(conf) => {
                let nodes = match span {
                    None if self.header.kind.is_tree() => self.tree.len() as u64,
                    Some(Span {
                        tree: Some(tree), ..
                    }) => u64::from(tree.nodes),
                    _ => return Err(bad("DRAFT_CONF span", side.span.unwrap_or(ABSENT))),
                };
                expect("DRAFT_CONF len", conf.len(), 2, nodes)
            }
            SideItem::SeqCtrl(SeqCtrl::Reserved(code)) if !matches!(code, 4 | 5) => {
                Err(bad("SEQ_CTRL code", code))
            }
            SideItem::Other { ty, .. } if (tlv::TOKEN_IDS..=tlv::DRAFT_CONF).contains(&ty) => {
                Err(bad("TLV type", ty))
            }
            SideItem::SeqMeta(_)
            | SideItem::SeqCtrl(_)
            | SideItem::Trace(_)
            | SideItem::Truncate { .. }
            | SideItem::Other { .. } => Ok(()),
        }
    }

    /// Whether the frame carries a `TRACE` item, which flags bit0 `HAS_TRACE` announces.
    pub fn has_trace(&self) -> bool {
        self.side
            .iter()
            .any(|side| matches!(side.item, SideItem::Trace(_)))
    }

    /// The payload's row stride; 0 for a dtype this codec does not carry.
    fn stride(&self) -> usize {
        self.header.layout.stride().unwrap_or(0) as usize
    }

    /// Row `index`'s elements, `streams × hidden` of the dtype, without the stride's padding.
    pub fn row(&self, index: u32) -> Option<&'a [u8]> {
        let row_len = self.header.layout.row_len()? as usize;
        let at = (index as usize).checked_mul(self.stride())?;
        self.payload.get(at..at.checked_add(row_len)?)
    }

    /// Every row's elements, in order.
    pub fn rows(&self) -> impl Iterator<Item = &'a [u8]> + 'a {
        let activation = *self;
        (0..activation.header.n_rows).filter_map(move |index| activation.row(index))
    }

    fn whole(&self) -> Window {
        Window {
            start: 0,
            end: self.header.n_rows,
            part_index: self.header.part_index,
            part_count: self.header.part_count,
        }
    }

    /// Whether `window` carries `span`: a span with rows when they overlap, and a span without
    /// rows when its offset falls inside, or at the end of the last part.
    fn carries(&self, span: &Span, window: &Window) -> bool {
        let (start, end) = (span.row_offset, span.row_offset + span.n_positions);
        if span.n_positions > 0 {
            start < window.end && end > window.start
        } else {
            (window.start..window.end).contains(&start)
                || (start == window.end && window.end == self.header.n_rows)
        }
    }

    /// The spans `window` carries, a contiguous run since spans are in row order.
    fn window_spans(&self, window: &Window) -> Range<usize> {
        let mut first = None;
        let mut last = 0;
        for (index, span) in self.spans.iter().enumerate() {
            if self.carries(&span, window) {
                first.get_or_insert(index);
                last = index + 1;
            }
        }
        first.map_or(0..0, |first| first..last)
    }

    /// Where `side` goes in `window`: its span index there and, for an item sized by rows, the
    /// rows of its span the window holds. Every other item travels once, in the first part that
    /// carries its span, or in the first part for every span.
    fn place(
        &self,
        side: &Side<'_>,
        window: &Window,
        spans: &Range<usize>,
    ) -> Option<(Option<u16>, Range<u32>)> {
        let per_row = matches!(
            side.item,
            SideItem::TokenIds(_) | SideItem::RouterHint { .. }
        );
        match side.span {
            None if per_row => Some((None, window.start..window.end)),
            None => (window.start == 0).then_some((None, 0..0)),
            Some(index) => {
                let index = usize::from(index);
                if !spans.contains(&index) {
                    return None;
                }
                let span = self.spans.get(index)?;
                let local = Some((index - spans.start) as u16);
                if per_row {
                    let from = span.row_offset.max(window.start) - span.row_offset;
                    let to = (span.row_offset + span.n_positions).min(window.end) - span.row_offset;
                    Some((local, from..to))
                } else {
                    (span.row_offset >= window.start).then_some((local, 0..0))
                }
            }
        }
    }

    /// Writes `window`'s extension through `sink`, unpadded, and returns its areas. Called once
    /// to count, then again to write into a zeroed buffer of the counted length.
    fn emit(&self, window: &Window, mut sink: Sink<'_>) -> Result<Areas, FrameError> {
        let whole = window.start == 0 && window.end == self.header.n_rows;
        let spans = self.window_spans(window);
        let mut at = ACTIVATION_FIXED_LEN;
        for index in spans.clone() {
            let Some(span) = self.spans.get(index) else {
                continue;
            };
            let first = span.row_offset.max(window.start);
            let last = (span.row_offset + span.n_positions)
                .min(window.end)
                .max(first);
            let mut b = [0u8; SPAN_LEN];
            b[0..8].copy_from_slice(&span.seq.ingress.to_le_bytes());
            b[8..12].copy_from_slice(&span.seq.boot.to_le_bytes());
            b[12..16].copy_from_slice(&span.seq.n.to_le_bytes());
            let moved = first - span.row_offset;
            b[16..20].copy_from_slice(&span.first_position.wrapping_add(moved).to_le_bytes());
            b[20..24].copy_from_slice(&(last - first).to_le_bytes());
            b[24..28].copy_from_slice(&(first - window.start).to_le_bytes());
            let (offset, nodes) = match span.tree {
                Some(tree) if whole => (tree.offset, tree.nodes),
                _ => (ABSENT, 0),
            };
            b[28..30].copy_from_slice(&offset.to_le_bytes());
            b[30..32].copy_from_slice(&nodes.to_le_bytes());
            sink.put(at, &b);
            at += SPAN_LEN;
        }
        let tree_start = at;
        if whole {
            for node in self.tree.iter() {
                let mut b = [0u8; TREE_NODE_LEN];
                b[0..2].copy_from_slice(&node.parent.unwrap_or(ABSENT).to_le_bytes());
                b[2] = node.depth;
                b[3] = node.flags;
                sink.put(at, &b);
                at += TREE_NODE_LEN;
            }
        }
        let tree_len = at - tree_start;
        let side_start = at;
        let mut has_trace = false;
        for side in self.side.iter() {
            let Some((span, rows)) = self.place(&side, window, &spans) else {
                continue;
            };
            let rows = rows.start as usize..rows.end as usize;
            let mut head = [0u8; TRACE_LEN];
            let (ty, head_len, body): (u16, usize, &[u8]) = match side.item {
                SideItem::TokenIds(ids) => (
                    tlv::TOKEN_IDS,
                    0,
                    ids.get(rows.start * 4..rows.end * 4).unwrap_or(&[]),
                ),
                SideItem::RouterHint { layer, k, experts } => {
                    head[0..2].copy_from_slice(&layer.to_le_bytes());
                    head[2] = k;
                    let per_row = 2 * usize::from(k);
                    let experts = experts
                        .get(rows.start * per_row..rows.end * per_row)
                        .unwrap_or(&[]);
                    (tlv::ROUTER_HINT, 4, experts)
                }
                SideItem::SeqMeta(cbor) => (tlv::SEQ_META, 0, cbor),
                SideItem::SeqCtrl(ctrl) => {
                    let (code, len) = match ctrl {
                        SeqCtrl::Cancel => (1, 8),
                        SeqCtrl::Release => (2, 8),
                        SeqCtrl::ReleasePersist { session } => {
                            head[8..24].copy_from_slice(&session.0);
                            (3, 24)
                        }
                        SeqCtrl::Reserved(code) => (code, 8),
                    };
                    head[0] = code;
                    (tlv::SEQ_CTRL, len, &[][..])
                }
                SideItem::Trace(trace) => {
                    head[0] = trace.version;
                    head[1..17].copy_from_slice(&trace.trace_id);
                    head[17..25].copy_from_slice(&trace.span_id);
                    head[25] = trace.flags;
                    has_trace = true;
                    (tlv::TRACE, TRACE_LEN, &[][..])
                }
                SideItem::Truncate { to } => {
                    head[0..4].copy_from_slice(&to.to_le_bytes());
                    (tlv::TRUNCATE, 4, &[][..])
                }
                SideItem::DraftConf(conf) => (tlv::DRAFT_CONF, 0, conf),
                SideItem::Other { ty, value } => (ty, 0, value),
            };
            let len = head_len + body.len();
            let len32 = u32::try_from(len).map_err(|_| FrameError::TooLong {
                what: "side item",
                len,
            })?;
            let mut tlv_header = [0u8; TLV_HEADER_LEN];
            tlv_header[0..2].copy_from_slice(&ty.to_le_bytes());
            tlv_header[2..4].copy_from_slice(&span.unwrap_or(ABSENT).to_le_bytes());
            tlv_header[4..8].copy_from_slice(&len32.to_le_bytes());
            sink.put(at, &tlv_header);
            sink.put(at + TLV_HEADER_LEN, &head[..head_len]);
            sink.put(at + TLV_HEADER_LEN + head_len, body);
            at += TLV_HEADER_LEN + len.next_multiple_of(8);
        }
        let side_len = at - side_start;
        let n_spans = spans.len();
        let areas = Areas {
            n_spans,
            tree_len,
            side_len,
            has_trace,
        };
        let too_long = |what| FrameError::TooLong {
            what,
            len: areas.ext_len(),
        };
        let tree_len32 = u32::try_from(tree_len).map_err(|_| too_long("tree area"))?;
        let side_len32 = u32::try_from(side_len).map_err(|_| too_long("side area"))?;
        let mut b = [0u8; ACTIVATION_FIXED_LEN];
        let header = &self.header;
        b[0..2].copy_from_slice(&header.from_stage.0.to_le_bytes());
        b[2..4].copy_from_slice(&header.to_stage.0.to_le_bytes());
        b[4] = header.kind.code();
        b[5] = dtype_code(header.layout.dtype)
            .ok_or(FrameError::UnsupportedDType(header.layout.dtype))?;
        b[6] = header.layout.streams;
        b[8..12].copy_from_slice(&header.layout.hidden.to_le_bytes());
        b[12..16].copy_from_slice(&(window.end - window.start).to_le_bytes());
        b[16..18].copy_from_slice(&(n_spans as u16).to_le_bytes());
        b[18..20].copy_from_slice(&window.part_index.to_le_bytes());
        b[20..22].copy_from_slice(&window.part_count.to_le_bytes());
        b[24..28].copy_from_slice(&header.kind.arg().to_le_bytes());
        b[28..32].copy_from_slice(&header.kind.flags().to_le_bytes());
        b[32..36].copy_from_slice(&tree_len32.to_le_bytes());
        b[36..40].copy_from_slice(&side_len32.to_le_bytes());
        sink.put(0, &b);
        Ok(areas)
    }

    /// The payload bytes of `window`'s rows.
    fn window_payload(&self, window: &Window) -> &'a [u8] {
        let stride = self.stride();
        self.payload
            .get(window.start as usize * stride..window.end as usize * stride)
            .unwrap_or(&[])
    }

    /// `window`'s frame length, header and padding included.
    fn window_len(&self, window: &Window) -> Result<usize, FrameError> {
        let areas = self.emit(window, Sink(None))?;
        Ok(header::encoded_len(
            areas.ext_len(),
            self.window_payload(window).len(),
        ))
    }

    /// The extension's length, padded to a multiple of 64.
    pub fn ext_len(&self) -> Result<usize, FrameError> {
        self.validate()?;
        let areas = self.emit(&self.whole(), Sink(None))?;
        Ok(areas.ext_len().next_multiple_of(ALIGN))
    }

    /// Writes the extension into `out`, zero-padded, and returns its length, a multiple of 64.
    /// The AEAD layer seals what this writes; [`Activation::encode`] writes a whole unsealed
    /// frame.
    pub fn write_ext(&self, out: &mut [u8]) -> Result<usize, FrameError> {
        self.validate()?;
        self.write_window_ext(&self.whole(), out)
            .map(|(len, _)| len)
    }

    fn write_window_ext(
        &self,
        window: &Window,
        out: &mut [u8],
    ) -> Result<(usize, Areas), FrameError> {
        let areas = self.emit(window, Sink(None))?;
        let len = areas.ext_len().next_multiple_of(ALIGN);
        let available = out.len();
        let ext = out.get_mut(..len).ok_or(FrameError::BufferTooSmall {
            needed: len,
            available,
        })?;
        ext.fill(0);
        self.emit(window, Sink(Some(ext)))?;
        Ok((len, areas))
    }

    fn encode_window(
        &self,
        window: &Window,
        spec: &ActivationSpec,
        out: &mut [u8],
    ) -> Result<usize, FrameError> {
        if spec.flags.encrypted() {
            return Err(bad("flags", spec.flags.0));
        }
        let payload = self.window_payload(window);
        let areas = self.emit(window, Sink(None))?;
        let total = header::encoded_len(areas.ext_len(), payload.len());
        if out.len() < total {
            return Err(FrameError::BufferTooSmall {
                needed: total,
                available: out.len(),
            });
        }
        let (ext_len, areas) = self.write_window_ext(window, &mut out[HEADER_LEN..])?;
        let mut flags = spec.flags.0 & !Flags::HAS_TRACE;
        if areas.has_trace {
            flags |= Flags::HAS_TRACE;
        }
        let frame = FrameSpec {
            wire_minor: spec.wire_minor,
            ftype: FrameType::Activation,
            flags: Flags(flags),
            instance: spec.instance,
            epoch: spec.epoch,
            step: spec.step,
            chan_seq: spec.chan_seq,
        };
        header::encode_in_place(&frame, ext_len, payload, out)
    }

    /// Encodes this frame, unsealed, into `out` and returns its length. It is written as one
    /// frame, whatever its size; [`Activation::parts`] splits a chunk that exceeds `max_frame`.
    pub fn encode(&self, spec: &ActivationSpec, out: &mut [u8]) -> Result<usize, FrameError> {
        self.validate()?;
        self.encode_window(&self.whole(), spec, out)
    }

    /// [`Activation::encode`] into a new vector.
    pub fn encode_to_vec(&self, spec: &ActivationSpec) -> Result<Vec<u8>, FrameError> {
        self.validate()?;
        let mut out = vec![0; self.window_len(&self.whole())?];
        let len = self.encode_window(&self.whole(), spec, &mut out)?;
        out.truncate(len);
        Ok(out)
    }

    /// The rows from `start` that the next part holds under `max_frame`, and that part's frame
    /// length: as many as fit with the part's extension, which counts against its row limit
    /// (RFC-0031 §8).
    fn fit(&self, start: u32, max_frame: usize) -> Result<(u32, usize), FrameError> {
        let n_rows = self.header.n_rows;
        let len = |end| {
            self.window_len(&Window {
                start,
                end,
                part_index: 0,
                part_count: 1,
            })
        };
        let mut lo = (start + 1).min(n_rows);
        let first = len(lo)?;
        if first > max_frame {
            return Err(FrameError::TooLong {
                what: "activation part",
                len: first,
            });
        }
        let mut hi = n_rows;
        // The frame grows with every row it takes, so the last row that fits is found by halving.
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if len(mid)? <= max_frame {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        Ok((lo, len(lo)?))
    }

    /// The parts this chunk travels in when frames are at most `max_frame` bytes, header
    /// included: the fewest, each holding as many rows as fit (RFC-0031 §6.3, revised
    /// 2026-10-22). A chunk that fits is one part. Each part is a whole frame for its rows: its
    /// spans are cut at the part's edges, `TOKEN_IDS` and `ROUTER_HINT` are cut with them, and
    /// every other item travels once, in the first part that carries its span. A verify frame
    /// with `TREE` is never split, and the chunk itself must be whole, `part_count = 1`.
    pub fn parts(&self, max_frame: usize) -> Result<Parts<'_, 'a>, FrameError> {
        self.validate()?;
        if self.header.part_count != 1 {
            return Err(bad("part_count", self.header.part_count));
        }
        if self.header.kind.is_tree() {
            let len = self.window_len(&self.whole())?;
            if len > max_frame {
                return Err(FrameError::TooLong {
                    what: "verify frame with a tree",
                    len,
                });
            }
        }
        let mut count = 0usize;
        let mut start = 0;
        loop {
            let (end, _) = self.fit(start, max_frame)?;
            count += 1;
            if end >= self.header.n_rows {
                break;
            }
            start = end;
        }
        let count = u16::try_from(count).map_err(|_| FrameError::TooLong {
            what: "part count",
            len: count,
        })?;
        Ok(Parts {
            chunk: self,
            max_frame,
            next: 0,
            index: 0,
            count,
        })
    }

    /// Encodes one of this chunk's [`Activation::parts`], unsealed, into `out` and returns its
    /// length.
    pub fn encode_part(
        &self,
        part: &Part,
        spec: &ActivationSpec,
        out: &mut [u8],
    ) -> Result<usize, FrameError> {
        self.validate()?;
        let whole = part.rows.start == 0 && part.rows.end == self.header.n_rows;
        if self.header.part_count != 1
            || part.index >= part.count
            || part.rows.start > part.rows.end
            || part.rows.end > self.header.n_rows
            || (self.header.kind.is_tree() && !whole)
        {
            return Err(bad("part", part.index));
        }
        let window = Window {
            start: part.rows.start,
            end: part.rows.end,
            part_index: part.index,
            part_count: part.count,
        };
        self.encode_window(&window, spec, out)
    }
}

impl<'a> Frame<'a> {
    /// The `Activation` of an unsealed frame, read on a channel whose wire dtype is `wire`, its
    /// `ChannelSpec::wire`. A frame in another dtype is refused with
    /// [`FrameError::WireDTypeMismatch`] (RFC-0031 §6.4), and `HAS_TRACE` must match the items.
    pub fn activation(&self, wire: WireDType) -> Result<Activation<'a>, FrameError> {
        let ftype = self.header.spec.ftype;
        if ftype != FrameType::Activation {
            return Err(FrameError::NotActivation(ftype));
        }
        let flags = self.header.spec.flags;
        if flags.encrypted() {
            return Err(FrameError::Sealed(ftype));
        }
        let activation = Activation::parse(self.ext, self.payload, wire)?;
        if flags.has_trace() != activation.has_trace() {
            return Err(inconsistent(
                "HAS_TRACE",
                flags.has_trace(),
                activation.has_trace(),
            ));
        }
        Ok(activation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEQ: SequenceId = SequenceId {
        ingress: 0xA1,
        boot: 2,
        n: 3,
    };

    fn layout(dtype: WireDType, hidden: u32) -> RowLayout {
        RowLayout {
            dtype,
            streams: 1,
            hidden,
        }
    }

    fn header(kind: ActivationKind, layout: RowLayout, n_rows: u32) -> ActivationHeader {
        ActivationHeader {
            from_stage: StageIx(0),
            to_stage: StageIx(1),
            kind,
            layout,
            n_rows,
            part_index: 0,
            part_count: 1,
        }
    }

    fn span(row_offset: u32, n_positions: u32) -> Span {
        Span {
            seq: SEQ,
            first_position: 10 + row_offset,
            n_positions,
            row_offset,
            tree: None,
        }
    }

    fn spec() -> ActivationSpec {
        ActivationSpec {
            wire_minor: header::WIRE_MINOR,
            flags: Flags::default(),
            instance: InstanceId(1),
            epoch: PlanEpoch { term: 1, seq: 1 },
            step: 1,
            chan_seq: 1,
        }
    }

    #[test]
    fn strides_round_rows_up_to_64_bytes() {
        assert_eq!(layout(WireDType::F32, 24).row_len(), Some(96));
        assert_eq!(layout(WireDType::F32, 24).stride(), Some(128));
        assert_eq!(layout(WireDType::BF16, 32).stride(), Some(64));
        assert_eq!(layout(WireDType::BF16, 33).stride(), Some(128));
        let two = RowLayout {
            dtype: WireDType::F32,
            streams: 2,
            hidden: 16,
        };
        assert_eq!(two.row_len(), Some(128));
        assert_eq!(layout(WireDType::F16, 32).stride(), None);
    }

    #[test]
    fn kinds_carry_their_argument_and_flags() {
        let kinds = [
            ActivationKind::Prefill {
                chunk: 7,
                last: true,
            },
            ActivationKind::Decode,
            ActivationKind::Verify {
                window: 5,
                tree: true,
            },
            ActivationKind::Replay {
                id: 9,
                last: false,
                forward: true,
            },
            ActivationKind::Canary,
        ];
        for (index, kind) in kinds.into_iter().enumerate() {
            assert_eq!(kind.code() as usize, index + 1);
            assert_eq!(
                ActivationKind::from_wire(kind.code(), kind.arg(), kind.flags()),
                Some(kind)
            );
        }
        // Reserved flag bits and a `Decode` argument are ignored.
        assert_eq!(
            ActivationKind::from_wire(2, 77, 0xFFFF_FFFF),
            Some(ActivationKind::Decode)
        );
        assert_eq!(ActivationKind::from_wire(0, 0, 0), None);
        assert_eq!(ActivationKind::from_wire(6, 0, 0), None);
    }

    #[test]
    fn spans_must_cover_the_rows_in_order() {
        let payload = [0u8; 3 * 64];
        let gap = [span(0, 1), span(2, 1)];
        let activation = Activation {
            header: header(ActivationKind::Decode, layout(WireDType::F32, 16), 3),
            spans: Spans::new(&gap),
            tree: TreeArea::new(&[]),
            side: SideItems::new(&[]),
            payload: &payload,
        };
        assert_eq!(
            activation.validate(),
            Err(inconsistent("row_offset", 2u32, 1u64))
        );
        let short = [span(0, 1), span(1, 1)];
        let activation = Activation {
            spans: Spans::new(&short),
            ..activation
        };
        assert_eq!(
            activation.validate(),
            Err(inconsistent("n_rows", 3u32, 2u64))
        );
    }

    #[test]
    fn an_empty_frame_carries_a_span_without_rows() {
        let spans = [span(0, 0)];
        let items = [Side {
            span: Some(0),
            item: SideItem::SeqCtrl(SeqCtrl::Cancel),
        }];
        let activation = Activation {
            header: header(ActivationKind::Decode, layout(WireDType::F32, 16), 0),
            spans: Spans::new(&spans),
            tree: TreeArea::new(&[]),
            side: SideItems::new(&items),
            payload: &[],
        };
        let bytes = activation.encode_to_vec(&spec()).unwrap();
        let frame = Frame::decode(&bytes).unwrap();
        assert!(frame.payload.is_empty());
        assert_eq!(frame.activation(WireDType::F32), Ok(activation));
        let parts: Vec<Part> = activation.parts(4096).unwrap().collect();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].rows, 0..0);
    }

    #[test]
    fn a_sealed_frame_or_an_encrypted_spec_is_refused() {
        let payload = [0u8; 64];
        let spans = [span(0, 1)];
        let activation = Activation {
            header: header(ActivationKind::Decode, layout(WireDType::F32, 16), 1),
            spans: Spans::new(&spans),
            tree: TreeArea::new(&[]),
            side: SideItems::new(&[]),
            payload: &payload,
        };
        let sealed = ActivationSpec {
            flags: Flags(Flags::ENCRYPTED),
            ..spec()
        };
        assert_eq!(
            activation.encode_to_vec(&sealed),
            Err(bad("flags", Flags::ENCRYPTED))
        );
        let mut ext = [0u8; 256];
        let ext_len = activation.write_ext(&mut ext).unwrap();
        assert_eq!(ext_len, 128);
        let mut tagged = payload.to_vec();
        tagged.extend_from_slice(&[0; 16]);
        let frame_spec = FrameSpec {
            wire_minor: header::WIRE_MINOR,
            ftype: FrameType::Activation,
            flags: Flags(Flags::ENCRYPTED),
            instance: InstanceId(1),
            epoch: PlanEpoch { term: 1, seq: 1 },
            step: 1,
            chan_seq: 1,
        };
        let bytes = header::encode_to_vec(&frame_spec, &ext[..ext_len], &tagged).unwrap();
        let frame = Frame::decode(&bytes).unwrap();
        assert_eq!(
            frame.activation(WireDType::F32),
            Err(FrameError::Sealed(FrameType::Activation))
        );
    }
}
