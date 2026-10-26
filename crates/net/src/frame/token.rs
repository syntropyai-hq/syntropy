// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Token` frame (RFC-0031 §6.5): the tail's tokens, to the head on the ring return for all
//! of a step's sequences, and to each ingress node for its own.
//!
//! ```text
//! extension | n_entries u16 | flags u16 | side_len u32 | part_index u16 | part_count u16 |
//!           | reserved u32 | side TLVs at offset 16 (side_len B, §6.3's encoding) | padding |
//! payload   n_entries entries: a 48-byte header, then tokens, draft, draft tree, logprobs and
//!           the tap block, each entry padded to 8
//! ```
//!
//! The two copies of a step carry different things. The ring-return copy carries each entry's
//! draft for the next window and the items for the head, and no tap block. The ingress copy
//! carries the tap blocks, and no draft, and of the items only its entries' `TRACE`. Both carry
//! a sampled step's `TRACE` items, one per traced entry and naming it, so the return path appears
//! as a hop span (RFC-0053 §3.2).
//! [`StepTokens::encode`] writes either copy from the tail's whole step, leaving out what that
//! copy does not carry. [`Frame::token`] refuses a frame that carries one.
//!
//! An entry's header cannot say whether its draft is a chain or a tree, since `draft_nodes`
//! equals `n_draft` either way. The extension's flags bit2 `DRAFT_TREE` says it for the frame:
//! when set, every entry with a draft carries its tree (RFC-0031 §6.5, revised 2026-10-22).
//!
//! What this codec leaves to other tasks:
//! - the tap block's contents, which travel as bytes until DIST-E8-T06 reads them. Until then a
//!   [`TokenEntryView`] converts to and from `core`'s `TokenEntry` only without a tap, and a
//!   tap block is refused with [`FrameError::TapNotDecoded`], and a `core` tap with
//!   [`FrameError::TapNotEncoded`];
//! - the ring-return items (`KV_HEADROOM`, `PREFIX_DROPPED`, …), carried as
//!   [`SideItem::Other`], and the narrower set of types a ring-return frame may carry, both
//!   DIST-E8-T04's. Of the `Activation` items, types 1–7, only `TRACE` is accepted;
//! - splitting a step into parts: [`Token`] reads and writes `part_index` and `part_count`.
//!
//! Neither direction allocates: a decoded frame borrows its entries and items from the frame's
//! bytes, and an encoded one is written straight into the caller's buffer.
//! [`TokenEntryView::to_entry`] and [`TokenEntryBytes`] build `core`'s owned types, so they
//! allocate; the codec itself does not.

use std::fmt;

use syntropy_core::{
    ApproxFlags, FinishReason, InstanceId, LogprobBlock, PlanEpoch, SequenceId, TokenEntry,
};

use super::activation::{
    ABSENT, Side, SideItem, SideItems, TraceContext, TreeArea, TreeNode, get16, get32, get64,
    read_tlv,
};
use super::error::FrameError;
use super::header::{self, ALIGN, Flags, Frame, FrameSpec, FrameType, HEADER_LEN};

/// The extension's fixed part, before the TLVs.
pub const TOKEN_FIXED_LEN: usize = 16;

/// An entry's header, before its arrays.
pub const ENTRY_HEADER_LEN: usize = 48;

/// The extension's `flags`.
pub mod token_flags {
    /// bit0: the ring-return copy, tail → head.
    pub const RING_RETURN: u16 = 1 << 0;
    /// bit1: the copy to an ingress node.
    pub const TO_INGRESS: u16 = 1 << 1;
    /// bit2: every entry with a draft carries its draft tree (RFC-0031 §6.5, revised
    /// 2026-10-22).
    pub const DRAFT_TREE: u16 = 1 << 2;
}

/// Which copy of a step a frame is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Route {
    /// Tail → head, with every sequence of the step.
    RingReturn,
    /// Tail → an ingress node, with only its sequences.
    ToIngress,
}

/// Which copy of a step [`StepTokens::encode`] writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TokenCopy {
    /// The ring-return copy: every entry, drafts and items, no tap.
    RingReturn,
    /// The copy to the ingress node whose short ID is `ingress`: the entries whose
    /// `SequenceId::ingress` it is, taps, no draft and no items.
    Ingress { ingress: u64 },
}

/// `finish`'s wire code (RFC-0031 §6.5): 0 none, 1 to 5 [`FinishReason`]'s variants in order.
fn finish_code(finish: Option<FinishReason>) -> u8 {
    match finish {
        None => 0,
        Some(FinishReason::Stop) => 1,
        Some(FinishReason::Length) => 2,
        Some(FinishReason::TokenStop) => 3,
        Some(FinishReason::Cancelled) => 4,
        Some(FinishReason::Error { .. }) => 5,
    }
}

/// The [`FinishReason`] of `finish`'s wire code. `error_domain` and `error_code` are read only for
/// `Error`, and written as zero otherwise.
fn finish_from_wire(code: u8, domain: u16, error: u16) -> Result<Option<FinishReason>, FrameError> {
    Ok(match code {
        0 => None,
        1 => Some(FinishReason::Stop),
        2 => Some(FinishReason::Length),
        3 => Some(FinishReason::TokenStop),
        4 => Some(FinishReason::Cancelled),
        5 => Some(FinishReason::Error {
            domain,
            code: error,
        }),
        other => return Err(bad("finish", other)),
    })
}

/// One sequence's tokens from a step, as a zero-copy view of a `Token` frame's entry (RFC-0031
/// §6.5): `core`'s [`TokenEntry`], whose definition this codec does not repeat (RFC-0001 §5.13).
/// Arrays are little-endian bytes, as on the wire, so a decoded entry borrows them from the
/// frame. [`TokenEntryView::to_entry`] converts it to the owned type, and [`TokenEntryBytes`]
/// holds an owned entry's arrays as bytes, for a view to be encoded from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenEntryView<'a> {
    pub seq: SequenceId,
    /// The position of the first accepted token.
    pub first_position: u32,
    /// The accepted tokens, a u32 each: one for plain decode, 1..=γ+1 after a verify window.
    pub tokens: &'a [u8],
    pub finish: Option<FinishReason>,
    /// Top-k alternatives per token; 0 for no logprobs.
    pub logprob_k: u8,
    /// Per token: the chosen token's logprob as an f32, then `logprob_k` × (id u32, logprob
    /// f32). Empty when `logprob_k` is 0.
    pub logprobs: &'a [u8],
    /// The approximate knobs that touched these tokens; empty in exact mode (RFC-0001 §11.3).
    pub approx: ApproxFlags,
    /// The tail-side drafter's draft for the next window, a u32 each (RFC-0032 §3.5).
    pub draft: &'a [u8],
    /// The draft's tree, one node per drafted token; `None` for a chain.
    pub draft_tree: Option<TreeArea<'a>>,
    /// The parity tap block (RFC-0060 §5.5), which DIST-E8-T06 reads; empty outside parity
    /// sequences.
    pub tap: &'a [u8],
    /// The sequence's sampler draw counter after these tokens (RFC-0032 §3.7).
    pub rng_counter: u64,
}

impl TokenEntryView<'_> {
    /// Accepted tokens.
    pub fn n_tokens(&self) -> usize {
        self.tokens.len() / 4
    }

    /// Token `index`.
    pub fn token(&self, index: usize) -> Option<u32> {
        word(self.tokens, index)
    }

    /// Drafted tokens.
    pub fn n_draft(&self) -> usize {
        self.draft.len() / 4
    }

    /// Drafted token `index`.
    pub fn draft_token(&self, index: usize) -> Option<u32> {
        word(self.draft, index)
    }

    /// The logprob bytes per token: the chosen token's, then `logprob_k` alternatives.
    fn logprob_stride(&self) -> usize {
        4 + 8 * usize::from(self.logprob_k)
    }

    /// The bytes this entry takes in a frame, padding included, with or without its draft and
    /// tap.
    fn wire_len(&self, draft: bool, tap: bool) -> usize {
        let mut len = ENTRY_HEADER_LEN + self.tokens.len() + self.logprobs.len();
        if draft {
            len += self.draft.len();
            if let Some(tree) = self.draft_tree {
                len += tree.len() * 4;
            }
        }
        if tap {
            len += self.tap.len();
        }
        len.next_multiple_of(8)
    }

    /// The rules on one entry that hold on either copy.
    fn check(&self) -> Result<(), FrameError> {
        for (field, bytes) in [("tokens", self.tokens), ("draft", self.draft)] {
            if !bytes.len().is_multiple_of(4) || bytes.len() / 4 > usize::from(u16::MAX) {
                return Err(FrameError::TooLong {
                    what: field,
                    len: bytes.len(),
                });
            }
        }
        let logprobs = if self.logprob_k == 0 {
            0
        } else {
            self.n_tokens() * self.logprob_stride()
        };
        if self.logprobs.len() != logprobs {
            return Err(inconsistent(
                "logprobs len",
                self.logprobs.len() as u64,
                logprobs as u64,
            ));
        }
        if u32::try_from(self.tap.len()).is_err() {
            return Err(FrameError::TooLong {
                what: "tap",
                len: self.tap.len(),
            });
        }
        if let Some(tree) = self.draft_tree {
            if tree.len() != self.n_draft() || tree.is_empty() {
                return Err(inconsistent(
                    "draft_nodes",
                    tree.len() as u64,
                    self.n_draft() as u64,
                ));
            }
            for (index, node) in tree.iter().enumerate() {
                if node
                    .parent
                    .is_some_and(|parent| usize::from(parent) >= index)
                {
                    return Err(bad("parent", node.parent.unwrap_or(ABSENT)));
                }
                // 1 under the last accepted token, else one below the parent (review-net N2).
                let depth = match node.parent {
                    None => Some(1),
                    Some(parent) => tree
                        .get(usize::from(parent))
                        .and_then(|p| p.depth.checked_add(1)),
                };
                if depth != Some(node.depth) {
                    return Err(bad("depth", node.depth));
                }
            }
        }
        Ok(())
    }
}

impl TokenEntryView<'_> {
    /// The owned [`TokenEntry`] this view reads (RFC-0001 §5.8). A tap block is refused with
    /// [`FrameError::TapNotDecoded`] until its codec exists (DIST-E8-T06): it travels here as
    /// bytes, and `core`'s `TokenEntry::tap` is a decoded `LogitTap`.
    pub fn to_entry(&self) -> Result<TokenEntry, FrameError> {
        if !self.tap.is_empty() {
            return Err(FrameError::TapNotDecoded {
                len: self.tap.len(),
            });
        }
        let logprobs = (self.logprob_k > 0).then(|| {
            let k = usize::from(self.logprob_k);
            let n = self.n_tokens();
            let mut block = LogprobBlock {
                k: self.logprob_k,
                chosen: Vec::with_capacity(n),
                top_ids: Vec::with_capacity(n * k),
                top_logprobs: Vec::with_capacity(n * k),
            };
            for token in self.logprobs.chunks_exact(self.logprob_stride()) {
                block
                    .chosen
                    .push(f32::from_le_bytes([token[0], token[1], token[2], token[3]]));
                for alt in token[4..].as_chunks::<8>().0 {
                    let [i0, i1, i2, i3, l0, l1, l2, l3] = *alt;
                    block.top_ids.push(u32::from_le_bytes([i0, i1, i2, i3]));
                    block
                        .top_logprobs
                        .push(f32::from_le_bytes([l0, l1, l2, l3]));
                }
            }
            block
        });
        Ok(TokenEntry {
            seq: self.seq,
            tokens: words(self.tokens).collect(),
            first_position: self.first_position,
            logprobs,
            finish: self.finish,
            approx: self.approx,
            draft: words(self.draft).collect(),
            draft_tree: self
                .draft_tree
                .map(|tree| (0..tree.len()).filter_map(|i| tree.get(i)).collect()),
            rng_counter: self.rng_counter,
            tap: None,
        })
    }
}

fn words(bytes: &[u8]) -> impl Iterator<Item = u32> + '_ {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
}

/// An owned [`TokenEntry`]'s arrays as wire bytes, so that a [`TokenEntryView`] borrowing them
/// can be encoded: build one with [`TokenEntryBytes::new`], then encode
/// [`TokenEntryBytes::view`]. A tap is refused with [`FrameError::TapNotEncoded`] until its
/// codec exists (DIST-E8-T06), a [`LogprobBlock`] with `k = 0` with
/// [`FrameError::ChosenLogprobsOnly`] until RFC-0031 §6.5 has a form for it (OQ-NET-35), and one
/// whose lengths disagree with its `k` and the entry's tokens as inconsistent.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenEntryBytes {
    seq: SequenceId,
    first_position: u32,
    tokens: Vec<u8>,
    finish: Option<FinishReason>,
    logprob_k: u8,
    logprobs: Vec<u8>,
    approx: ApproxFlags,
    draft: Vec<u8>,
    draft_tree: Option<Vec<TreeNode>>,
    rng_counter: u64,
}

impl TokenEntryBytes {
    /// `entry`'s arrays as little-endian bytes, in RFC-0031 §6.5's layout.
    pub fn new(entry: &TokenEntry) -> Result<TokenEntryBytes, FrameError> {
        if let Some(tap) = &entry.tap {
            return Err(FrameError::TapNotEncoded {
                rows: tap.rows.len(),
            });
        }
        let bytes = |values: &[u32]| {
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>()
        };
        let (logprob_k, logprobs) = match &entry.logprobs {
            None => (0, Vec::new()),
            Some(block) if block.k == 0 => {
                return Err(FrameError::ChosenLogprobsOnly {
                    n_tokens: block.chosen.len(),
                });
            }
            Some(block) => {
                let k = usize::from(block.k);
                let expected = entry.tokens.len() * k;
                if block.chosen.len() != entry.tokens.len() || !block.is_consistent() {
                    return Err(inconsistent(
                        "logprobs len",
                        block.top_ids.len() as u64,
                        expected as u64,
                    ));
                }
                let mut out = Vec::with_capacity(entry.tokens.len() * (4 + 8 * k));
                for (i, chosen) in block.chosen.iter().enumerate() {
                    out.extend(chosen.to_le_bytes());
                    for j in i * k..(i + 1) * k {
                        out.extend(block.top_ids[j].to_le_bytes());
                        out.extend(block.top_logprobs[j].to_le_bytes());
                    }
                }
                (block.k, out)
            }
        };
        Ok(TokenEntryBytes {
            seq: entry.seq,
            first_position: entry.first_position,
            tokens: bytes(&entry.tokens),
            finish: entry.finish,
            logprob_k,
            logprobs,
            approx: entry.approx,
            draft: bytes(&entry.draft),
            draft_tree: entry.draft_tree.as_ref().map(|tree| tree.to_vec()),
            rng_counter: entry.rng_counter,
        })
    }

    /// A view of the entry, for [`StepTokens`] to encode.
    pub fn view(&self) -> TokenEntryView<'_> {
        TokenEntryView {
            seq: self.seq,
            first_position: self.first_position,
            tokens: &self.tokens,
            finish: self.finish,
            logprob_k: self.logprob_k,
            logprobs: &self.logprobs,
            approx: self.approx,
            draft: &self.draft,
            draft_tree: self.draft_tree.as_deref().map(TreeArea::new),
            tap: &[],
            rng_counter: self.rng_counter,
        }
    }
}

fn word(bytes: &[u8], index: usize) -> Option<u32> {
    let at = index.checked_mul(4)?;
    bytes
        .get(at..at.checked_add(4)?)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn bad(field: &'static str, value: impl Into<u64>) -> FrameError {
    FrameError::BadField {
        ftype: FrameType::Token,
        field,
        value: value.into(),
    }
}

fn inconsistent(field: &'static str, value: u64, expected: u64) -> FrameError {
    FrameError::Inconsistent {
        ftype: FrameType::Token,
        field,
        value,
        expected,
    }
}

/// Reads the entry at `at` in the payload and returns it with the offset of the next one.
fn read_entry(
    payload: &[u8],
    at: usize,
    tree: bool,
) -> Result<(TokenEntryView<'_>, usize), FrameError> {
    let header_end = at + ENTRY_HEADER_LEN;
    if header_end > payload.len() {
        return Err(inconsistent(
            "payload_len",
            payload.len() as u64,
            header_end as u64,
        ));
    }
    let h = &payload[at..header_end];
    let n_tokens = usize::from(get16(h, 20));
    let logprob_k = h[23];
    let n_draft = usize::from(get16(h, 32));
    let draft_nodes = usize::from(get16(h, 34));
    let tap_len = get32(h, 36) as usize;
    // `draft_nodes = n_draft` for a chain, and a tree has a node per drafted token.
    if draft_nodes != n_draft {
        return Err(inconsistent(
            "draft_nodes",
            draft_nodes as u64,
            n_draft as u64,
        ));
    }
    let nodes = if tree && n_draft > 0 { n_draft } else { 0 };
    let logprobs = if logprob_k == 0 {
        0
    } else {
        n_tokens * (4 + 8 * usize::from(logprob_k))
    };
    let lens = [4 * n_tokens, 4 * n_draft, 4 * nodes, logprobs, tap_len];
    let body = lens
        .iter()
        .try_fold(0usize, |sum, len| sum.checked_add(*len))
        .unwrap_or(usize::MAX);
    let end = header_end
        .checked_add(body)
        .and_then(|end| end.checked_next_multiple_of(8))
        .filter(|end| *end <= payload.len())
        .ok_or(inconsistent(
            "payload_len",
            payload.len() as u64,
            (header_end as u64).saturating_add(body as u64),
        ))?;
    let mut parts = [&payload[..0]; 5];
    let mut cursor = header_end;
    for (part, len) in parts.iter_mut().zip(lens) {
        *part = &payload[cursor..cursor + len];
        cursor += len;
    }
    let [tokens, draft, tree_nodes, logprobs, tap] = parts;
    let entry = TokenEntryView {
        seq: SequenceId {
            ingress: get64(h, 0),
            boot: get32(h, 8),
            n: get32(h, 12),
        },
        first_position: get32(h, 16),
        tokens,
        finish: finish_from_wire(h[22], get16(h, 28), get16(h, 30))?,
        logprob_k,
        logprobs,
        approx: ApproxFlags::from_bits_retain(get32(h, 24)),
        draft,
        draft_tree: (nodes > 0).then(|| TreeArea::wire(tree_nodes)),
        tap,
        rng_counter: get64(h, 40),
    };
    Ok((entry, end))
}

/// Writes `entry` at `out[0..]`, with its draft and tap only when asked. `out` holds
/// [`TokenEntryView::wire_len`] zeroed bytes.
fn write_entry(entry: &TokenEntryView<'_>, draft: bool, tap: bool, out: &mut [u8]) {
    let (n_draft, tree) = if draft {
        (entry.n_draft(), entry.draft_tree)
    } else {
        (0, None)
    };
    let tap = if tap { entry.tap } else { &[] };
    let h = &mut out[..ENTRY_HEADER_LEN];
    h[0..8].copy_from_slice(&entry.seq.ingress.to_le_bytes());
    h[8..12].copy_from_slice(&entry.seq.boot.to_le_bytes());
    h[12..16].copy_from_slice(&entry.seq.n.to_le_bytes());
    h[16..20].copy_from_slice(&entry.first_position.to_le_bytes());
    h[20..22].copy_from_slice(&(entry.n_tokens() as u16).to_le_bytes());
    h[22] = finish_code(entry.finish);
    h[23] = entry.logprob_k;
    h[24..28].copy_from_slice(&entry.approx.bits().to_le_bytes());
    if let Some(FinishReason::Error { domain, code }) = entry.finish {
        h[28..30].copy_from_slice(&domain.to_le_bytes());
        h[30..32].copy_from_slice(&code.to_le_bytes());
    }
    h[32..34].copy_from_slice(&(n_draft as u16).to_le_bytes());
    h[34..36].copy_from_slice(&(n_draft as u16).to_le_bytes());
    h[36..40].copy_from_slice(&(tap.len() as u32).to_le_bytes());
    h[40..48].copy_from_slice(&entry.rng_counter.to_le_bytes());
    let mut at = ENTRY_HEADER_LEN;
    let mut put = |bytes: &[u8]| {
        out[at..at + bytes.len()].copy_from_slice(bytes);
        at += bytes.len();
    };
    put(entry.tokens);
    if draft {
        put(entry.draft);
    }
    for node in tree.iter().flat_map(|tree| tree.iter()) {
        let parent = node.parent.unwrap_or(ABSENT).to_le_bytes();
        put(&[parent[0], parent[1], node.depth, node.flags]);
    }
    put(entry.logprobs);
    put(tap);
}

/// The entries of a frame: read from its bytes, or given by a sender.
#[derive(Clone, Copy)]
pub struct Entries<'a>(EntryRepr<'a>);

#[derive(Clone, Copy)]
enum EntryRepr<'a> {
    /// `count` entries, already checked, and whether they carry draft trees.
    Wire {
        bytes: &'a [u8],
        count: u16,
        tree: bool,
    },
    Typed(&'a [TokenEntryView<'a>]),
}

impl<'a> Entries<'a> {
    pub fn new(entries: &'a [TokenEntryView<'a>]) -> Entries<'a> {
        Entries(EntryRepr::Typed(entries))
    }

    pub fn len(&self) -> usize {
        match self.0 {
            EntryRepr::Wire { count, .. } => usize::from(count),
            EntryRepr::Typed(entries) => entries.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = TokenEntryView<'a>> + 'a {
        let repr = self.0;
        let mut at = 0;
        let mut index = 0;
        std::iter::from_fn(move || match repr {
            EntryRepr::Wire { bytes, count, tree } if index < usize::from(count) => {
                let (entry, next) = read_entry(bytes, at, tree).ok()?;
                at = next;
                index += 1;
                Some(entry)
            }
            EntryRepr::Wire { .. } => None,
            EntryRepr::Typed(entries) => {
                let entry = entries.get(index).copied();
                index += 1;
                entry
            }
        })
    }
}

impl PartialEq for Entries<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl fmt::Debug for Entries<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// The header fields of an unsealed `Token` frame. `flags` bit0 `HAS_TRACE` is set from the
/// `TRACE` items the copy carries, as the lengths and CRCs follow from the frame; `ENCRYPTED`
/// is the AEAD layer's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TokenSpec {
    pub wire_minor: u8,
    pub flags: Flags,
    pub instance: InstanceId,
    pub epoch: PlanEpoch,
    pub step: u64,
    pub chan_seq: u64,
}

/// A `Token` frame as it travels: one copy of a step, or one part of it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Token<'a> {
    pub route: Route,
    /// 0-based; a step larger than the channel's slot travels in parts.
    pub part_index: u16,
    /// At least 1.
    pub part_count: u16,
    /// The items: the items for the head on the ring-return copy, and `TRACE` for a sampled
    /// step's entries on either copy, its `span` the entry.
    pub side: SideItems<'a>,
    pub entries: Entries<'a>,
}

/// Checks the rules on side items and parts that hold on either copy.
fn check_frame_fields(
    side: &SideItems<'_>,
    n_entries: usize,
    part_index: u16,
    part_count: u16,
) -> Result<(), FrameError> {
    if part_count == 0 {
        return Err(bad("part_count", 0u64));
    }
    if part_index >= part_count {
        return Err(bad("part_index", part_index));
    }
    if n_entries > usize::from(u16::MAX) {
        return Err(FrameError::TooLong {
            what: "entries",
            len: n_entries,
        });
    }
    // One bit per entry index, on the stack: which entries already have a `TRACE`.
    let mut traced = [0u64; (u16::MAX as usize).div_ceil(64)];
    for item in side.iter() {
        match item.item {
            // One sequence's trace context, never every entry's: an unspanned `TRACE` would go to
            // every ingress copy, carrying one sequence's trace to other nodes.
            SideItem::Trace(_) => {
                let span = item.span.ok_or(bad("TRACE span", ABSENT))?;
                if usize::from(span) < n_entries {
                    let (word, bit) = (usize::from(span) / 64, 1u64 << (span % 64));
                    if traced[word] & bit != 0 {
                        return Err(bad("second TRACE for entry", span));
                    }
                    traced[word] |= bit;
                }
            }
            SideItem::Other { ty, value } if !(1..=7).contains(&ty) => {
                if u32::try_from(value.len()).is_err() {
                    return Err(FrameError::TooLong {
                        what: "side item",
                        len: value.len(),
                    });
                }
            }
            SideItem::Other { ty, .. } => return Err(bad("TLV type", ty)),
            _ => return Err(bad("TLV type", tlv_type(&item.item))),
        }
        if let Some(span) = item.span
            && usize::from(span) >= n_entries
        {
            return Err(bad("TLV span", span));
        }
    }
    Ok(())
}

/// The type code of an `Activation` item.
fn tlv_type(item: &SideItem<'_>) -> u16 {
    match item {
        SideItem::TokenIds(_) => 1,
        SideItem::RouterHint { .. } => 2,
        SideItem::SeqMeta(_) => 3,
        SideItem::SeqCtrl(_) => 4,
        SideItem::Trace(_) => 5,
        SideItem::Truncate { .. } => 6,
        SideItem::DraftConf(_) => 7,
        SideItem::Other { ty, .. } => *ty,
    }
}

/// `TRACE`'s value: W3C trace context, 26 bytes (RFC-0031 §6.3).
const TRACE_LEN: usize = 26;

fn trace_bytes(trace: &TraceContext) -> [u8; TRACE_LEN] {
    let mut b = [0u8; TRACE_LEN];
    b[0] = trace.version;
    b[1..17].copy_from_slice(&trace.trace_id);
    b[17..25].copy_from_slice(&trace.span_id);
    b[25] = trace.flags;
    b
}

/// The value bytes of an item a `Token` frame may carry.
fn item_len(item: &SideItem<'_>) -> usize {
    match item {
        SideItem::Trace(_) => TRACE_LEN,
        SideItem::Other { value, .. } => value.len(),
        _ => 0,
    }
}

/// What one frame holds once its copy is chosen.
struct Contents<'s, 'a> {
    route: Route,
    part_index: u16,
    part_count: u16,
    side: &'s SideItems<'a>,
    entries: &'s Entries<'a>,
    /// Only the entries of this ingress node, on an ingress copy built from a step.
    ingress: Option<u64>,
}

impl Contents<'_, '_> {
    fn keeps(&self, entry: &TokenEntryView<'_>) -> bool {
        self.ingress
            .is_none_or(|ingress| entry.seq.ingress == ingress)
    }

    fn draft(&self) -> bool {
        self.route == Route::RingReturn
    }

    fn tap(&self) -> bool {
        self.route == Route::ToIngress
    }

    /// Where `item` goes in this copy, with its span there: every item on the ring-return copy;
    /// on an ingress copy, only the `TRACE` of an entry it keeps, the span counted among the kept
    /// entries. A `TRACE` names one entry; [`check_frame_fields`] refuses one that names none.
    fn place(&self, item: &Side<'_>) -> Option<Option<u16>> {
        if self.route == Route::RingReturn {
            return Some(item.span);
        }
        if !matches!(item.item, SideItem::Trace(_)) {
            return None;
        }
        let span = item.span?;
        let mut kept = 0u16;
        for (index, entry) in self.entries.iter().enumerate() {
            let keeps = self.keeps(&entry);
            if index == usize::from(span) {
                return keeps.then_some(Some(kept));
            }
            kept += u16::from(keeps);
        }
        None
    }

    /// Whether the frame's drafts are trees: `DRAFT_TREE`.
    fn tree(&self) -> bool {
        self.draft() && self.entries.iter().any(|entry| entry.draft_tree.is_some())
    }

    /// The unpadded extension length, and the payload length.
    fn lens(&self) -> (usize, usize) {
        let mut ext = TOKEN_FIXED_LEN;
        for item in self.side.iter() {
            if self.place(&item).is_some() {
                ext += 8 + item_len(&item.item).next_multiple_of(8);
            }
        }
        let payload = self
            .entries
            .iter()
            .filter(|entry| self.keeps(entry))
            .map(|entry| entry.wire_len(self.draft(), self.tap()))
            .sum();
        (ext, payload)
    }

    fn encode(&self, spec: &TokenSpec, out: &mut [u8]) -> Result<usize, FrameError> {
        if spec.flags.encrypted() {
            return Err(bad("flags", spec.flags.0));
        }
        let tree = self.tree();
        // On a frame with trees, every draft is one.
        if tree
            && self
                .entries
                .iter()
                .any(|entry| !entry.draft.is_empty() && entry.draft_tree.is_none())
        {
            return Err(bad("draft_tree", 0u64));
        }
        let (ext, payload) = self.lens();
        let ext_len = ext.next_multiple_of(ALIGN);
        let total = header::encoded_len(ext_len, payload);
        if out.len() < total {
            return Err(FrameError::BufferTooSmall {
                needed: total,
                available: out.len(),
            });
        }
        let kept = self
            .entries
            .iter()
            .filter(|entry| self.keeps(entry))
            .count();
        let frame = &mut out[..total];
        frame[HEADER_LEN..].fill(0);
        let e = &mut frame[HEADER_LEN..HEADER_LEN + ext_len];
        let mut flags = match self.route {
            Route::RingReturn => token_flags::RING_RETURN,
            Route::ToIngress => token_flags::TO_INGRESS,
        };
        if tree {
            flags |= token_flags::DRAFT_TREE;
        }
        e[0..2].copy_from_slice(&(kept as u16).to_le_bytes());
        e[2..4].copy_from_slice(&flags.to_le_bytes());
        e[4..8].copy_from_slice(&((ext - TOKEN_FIXED_LEN) as u32).to_le_bytes());
        e[8..10].copy_from_slice(&self.part_index.to_le_bytes());
        e[10..12].copy_from_slice(&self.part_count.to_le_bytes());
        let mut at = TOKEN_FIXED_LEN;
        let mut has_trace = false;
        for item in self.side.iter() {
            let Some(span) = self.place(&item) else {
                continue;
            };
            let trace;
            let (ty, value): (u16, &[u8]) = match item.item {
                SideItem::Trace(t) => {
                    trace = trace_bytes(&t);
                    has_trace = true;
                    (5, &trace[..])
                }
                SideItem::Other { ty, value } => (ty, value),
                // Refused by `check_frame_fields` before any copy is written.
                _ => continue,
            };
            e[at..at + 2].copy_from_slice(&ty.to_le_bytes());
            e[at + 2..at + 4].copy_from_slice(&span.unwrap_or(ABSENT).to_le_bytes());
            e[at + 4..at + 8].copy_from_slice(&(value.len() as u32).to_le_bytes());
            e[at + 8..at + 8 + value.len()].copy_from_slice(value);
            at += 8 + value.len().next_multiple_of(8);
        }
        let mut at = HEADER_LEN + ext_len;
        for entry in self.entries.iter().filter(|entry| self.keeps(entry)) {
            let len = entry.wire_len(self.draft(), self.tap());
            write_entry(&entry, self.draft(), self.tap(), &mut frame[at..at + len]);
            at += len;
        }
        let mut frame_flags = spec.flags.0 & !Flags::HAS_TRACE;
        if has_trace {
            frame_flags |= Flags::HAS_TRACE;
        }
        let spec = FrameSpec {
            wire_minor: spec.wire_minor,
            ftype: FrameType::Token,
            flags: Flags(frame_flags),
            instance: spec.instance,
            epoch: spec.epoch,
            step: spec.step,
            chan_seq: spec.chan_seq,
        };
        header::encode_assembled(&spec, ext_len, payload, frame)
    }
}

impl<'a> Token<'a> {
    /// Reads an unsealed `Token` extension and payload.
    pub fn parse(ext: &'a [u8], payload: &'a [u8]) -> Result<Token<'a>, FrameError> {
        if ext.len() < TOKEN_FIXED_LEN {
            return Err(FrameError::ExtTooShort {
                ftype: FrameType::Token,
                len: ext.len(),
                needed: TOKEN_FIXED_LEN,
            });
        }
        let n_entries = get16(ext, 0);
        let flags = get16(ext, 2);
        let route = match flags & (token_flags::RING_RETURN | token_flags::TO_INGRESS) {
            token_flags::RING_RETURN => Route::RingReturn,
            token_flags::TO_INGRESS => Route::ToIngress,
            _ => return Err(bad("flags", flags)),
        };
        let side_len = get32(ext, 4) as usize;
        if !side_len.is_multiple_of(8) {
            return Err(bad("side_len", side_len as u64));
        }
        let side_end = TOKEN_FIXED_LEN.saturating_add(side_len);
        if ext.len() < side_end {
            return Err(FrameError::ExtTooShort {
                ftype: FrameType::Token,
                len: ext.len(),
                needed: side_end,
            });
        }
        let side = &ext[TOKEN_FIXED_LEN..side_end];
        let mut at = 0;
        while at < side.len() {
            at = read_tlv(side, at, FrameType::Token)?.1;
        }
        let tree = flags & token_flags::DRAFT_TREE != 0;
        let mut at = 0;
        for _ in 0..n_entries {
            at = read_entry(payload, at, tree)?.1;
        }
        if at != payload.len() {
            return Err(inconsistent("payload_len", payload.len() as u64, at as u64));
        }
        let token = Token {
            route,
            part_index: get16(ext, 8),
            part_count: get16(ext, 10),
            side: SideItems::wire(side),
            entries: Entries(EntryRepr::Wire {
                bytes: payload,
                count: n_entries,
                tree,
            }),
        };
        token.validate()?;
        Ok(token)
    }

    /// Checks the rules the bytes cannot express:
    /// - the ring-return copy carries no tap block, and the ingress copy no draft and no item
    ///   but `TRACE`;
    /// - each entry's logprobs are `n_tokens` strides of the chosen logprob and `logprob_k`
    ///   alternatives, and a draft tree has a node per drafted token, each node's parent before
    ///   it and its depth one below its parent's, or 1 under the last accepted token;
    /// - a frame whose drafts are trees has a tree for every draft;
    /// - side items are `TRACE` or not `Activation` items (types 1–7), and name an entry of the
    ///   frame;
    /// - `part_index < part_count`.
    pub fn validate(&self) -> Result<(), FrameError> {
        check_frame_fields(
            &self.side,
            self.entries.len(),
            self.part_index,
            self.part_count,
        )?;
        let mut trees = 0;
        let mut drafts = 0;
        for entry in self.entries.iter() {
            entry.check()?;
            match self.route {
                Route::RingReturn if !entry.tap.is_empty() => {
                    return Err(bad("tap_len", entry.tap.len() as u64));
                }
                Route::ToIngress if !entry.draft.is_empty() => {
                    return Err(bad("n_draft", entry.n_draft() as u64));
                }
                _ => {}
            }
            drafts += usize::from(!entry.draft.is_empty());
            trees += usize::from(entry.draft_tree.is_some());
        }
        if trees > 0 && trees != drafts {
            return Err(inconsistent("draft_tree", trees as u64, drafts as u64));
        }
        if self.route == Route::ToIngress
            && let Some(item) = self
                .side
                .iter()
                .find(|item| !matches!(item.item, SideItem::Trace(_)))
        {
            return Err(bad("ingress TLV type", tlv_type(&item.item)));
        }
        Ok(())
    }

    /// Whether the frame carries a `TRACE` item, which flags bit0 `HAS_TRACE` announces.
    pub fn has_trace(&self) -> bool {
        self.side
            .iter()
            .any(|item| matches!(item.item, SideItem::Trace(_)))
    }

    fn contents(&self) -> Contents<'_, 'a> {
        Contents {
            route: self.route,
            part_index: self.part_index,
            part_count: self.part_count,
            side: &self.side,
            entries: &self.entries,
            ingress: None,
        }
    }

    /// Encodes this frame, unsealed, into `out` and returns its length.
    pub fn encode(&self, spec: &TokenSpec, out: &mut [u8]) -> Result<usize, FrameError> {
        self.validate()?;
        self.contents().encode(spec, out)
    }

    /// [`Token::encode`] into a new vector.
    pub fn encode_to_vec(&self, spec: &TokenSpec) -> Result<Vec<u8>, FrameError> {
        self.validate()?;
        let contents = self.contents();
        let (ext, payload) = contents.lens();
        let mut out = vec![0; header::encoded_len(ext, payload)];
        let len = contents.encode(spec, &mut out)?;
        out.truncate(len);
        Ok(out)
    }
}

/// One step's tokens as the tail holds them: every entry with its draft and tap, and the items
/// for the head. [`StepTokens::encode`] writes the ring-return copy or an ingress copy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepTokens<'a> {
    pub entries: Entries<'a>,
    pub side: SideItems<'a>,
    pub part_index: u16,
    pub part_count: u16,
}

impl<'a> StepTokens<'a> {
    fn validate(&self) -> Result<(), FrameError> {
        check_frame_fields(
            &self.side,
            self.entries.len(),
            self.part_index,
            self.part_count,
        )?;
        for entry in self.entries.iter() {
            entry.check()?;
        }
        Ok(())
    }

    fn contents(&self, copy: TokenCopy) -> Contents<'_, 'a> {
        let (route, ingress) = match copy {
            TokenCopy::RingReturn => (Route::RingReturn, None),
            TokenCopy::Ingress { ingress } => (Route::ToIngress, Some(ingress)),
        };
        Contents {
            route,
            part_index: self.part_index,
            part_count: self.part_count,
            side: &self.side,
            entries: &self.entries,
            ingress,
        }
    }

    /// Encodes `copy` of this step, unsealed, into `out` and returns its length. The
    /// ring-return copy leaves out every tap block. An ingress copy keeps only that node's
    /// entries and leaves out their drafts and the step's items.
    pub fn encode(
        &self,
        copy: TokenCopy,
        spec: &TokenSpec,
        out: &mut [u8],
    ) -> Result<usize, FrameError> {
        self.validate()?;
        self.contents(copy).encode(spec, out)
    }

    /// [`StepTokens::encode`] into a new vector.
    pub fn encode_to_vec(&self, copy: TokenCopy, spec: &TokenSpec) -> Result<Vec<u8>, FrameError> {
        self.validate()?;
        let contents = self.contents(copy);
        let (ext, payload) = contents.lens();
        let mut out = vec![0; header::encoded_len(ext, payload)];
        let len = contents.encode(spec, &mut out)?;
        out.truncate(len);
        Ok(out)
    }
}

impl<'a> Frame<'a> {
    /// The `Token` of an unsealed frame. `HAS_TRACE` must match the `TRACE` items.
    pub fn token(&self) -> Result<Token<'a>, FrameError> {
        let ftype = self.header.spec.ftype;
        if ftype != FrameType::Token {
            return Err(FrameError::WrongType {
                expected: FrameType::Token,
                found: ftype,
            });
        }
        let flags = self.header.spec.flags;
        if flags.encrypted() {
            return Err(FrameError::Sealed(ftype));
        }
        let token = Token::parse(self.ext, self.payload)?;
        if flags.has_trace() != token.has_trace() {
            return Err(inconsistent(
                "HAS_TRACE",
                u64::from(flags.has_trace()),
                u64::from(token.has_trace()),
            ));
        }
        Ok(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEQ: SequenceId = SequenceId {
        ingress: 0xA1,
        boot: 1,
        n: 2,
    };

    fn entry<'a>(tokens: &'a [u8], draft: &'a [u8], tap: &'a [u8]) -> TokenEntryView<'a> {
        TokenEntryView {
            seq: SEQ,
            first_position: 9,
            tokens,
            finish: None,
            logprob_k: 0,
            logprobs: &[],
            approx: ApproxFlags::empty(),
            draft,
            draft_tree: None,
            tap,
            rng_counter: 1 << 40,
        }
    }

    fn spec() -> TokenSpec {
        TokenSpec {
            wire_minor: header::WIRE_MINOR,
            flags: Flags::default(),
            instance: InstanceId(1),
            epoch: PlanEpoch { term: 1, seq: 1 },
            step: 3,
            chan_seq: 1,
        }
    }

    #[test]
    fn finish_codes_round_trip_and_7_is_refused() {
        let all = [
            None,
            Some(FinishReason::Stop),
            Some(FinishReason::Length),
            Some(FinishReason::TokenStop),
            Some(FinishReason::Cancelled),
            Some(FinishReason::Error {
                domain: 3,
                code: 0x030C,
            }),
        ];
        for (code, finish) in all.into_iter().enumerate() {
            assert_eq!(finish_code(finish) as usize, code);
            assert_eq!(finish_from_wire(code as u8, 3, 0x030C), Ok(finish));
        }
        // The error fields are read only for `Error`.
        assert_eq!(finish_from_wire(1, 9, 9), Ok(Some(FinishReason::Stop)));
        assert_eq!(finish_from_wire(6, 0, 0), Err(bad("finish", 6u8)));
    }

    #[test]
    fn an_entry_takes_its_header_and_arrays_padded_to_8() {
        let tokens = [1, 0, 0, 0];
        let draft = [2, 0, 0, 0, 3, 0, 0, 0];
        let tap = [7; 5];
        let e = entry(&tokens, &draft, &tap);
        // 48 + 4 = 52, padded to 56; with the draft 60, padded to 64; with the tap 57, to 64.
        assert_eq!(e.wire_len(false, false), 56);
        assert_eq!(e.wire_len(true, false), 64);
        assert_eq!(e.wire_len(false, true), 64);
        assert_eq!(e.token(0), Some(1));
        assert_eq!(e.draft_token(1), Some(3));
        assert_eq!(e.draft_token(2), None);
    }

    #[test]
    fn each_copy_leaves_out_what_it_does_not_carry() {
        let tokens = [1, 0, 0, 0];
        let draft = [2, 0, 0, 0];
        let tap = [7; 8];
        let entries = [entry(&tokens, &draft, &tap)];
        let step = StepTokens {
            entries: Entries::new(&entries),
            side: SideItems::new(&[]),
            part_index: 0,
            part_count: 1,
        };
        let ring = step.encode_to_vec(TokenCopy::RingReturn, &spec()).unwrap();
        let frame = Frame::decode(&ring).unwrap();
        let token = frame.token().unwrap();
        assert_eq!(token.route, Route::RingReturn);
        let decoded: Vec<_> = token.entries.iter().collect();
        assert_eq!(decoded[0].draft, &draft);
        assert!(decoded[0].tap.is_empty());
        let ingress = step
            .encode_to_vec(TokenCopy::Ingress { ingress: 0xA1 }, &spec())
            .unwrap();
        let frame = Frame::decode(&ingress).unwrap();
        let token = frame.token().unwrap();
        let decoded: Vec<_> = token.entries.iter().collect();
        assert!(decoded[0].draft.is_empty());
        assert_eq!(decoded[0].tap, &tap);
        // Another node's ingress copy has none of this step's entries.
        let other = step
            .encode_to_vec(TokenCopy::Ingress { ingress: 0xB2 }, &spec())
            .unwrap();
        let frame = Frame::decode(&other).unwrap();
        assert!(frame.token().unwrap().entries.is_empty());
    }
}
