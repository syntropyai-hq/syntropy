// SPDX-License-Identifier: MIT OR Apache-2.0

//! Slot sizes from the plan (RFC-0031 §8's pool table): header areas, token slots, and the slots
//! a channel's pool holds.
//!
//! - **Header areas.** A frame's base header and extension are built in place in a slot of their
//!   own, which RDMA sends as the first scatter-gather entry. [`header_area`] sizes it from the
//!   plan's limits: the 64-byte base header plus the `Activation` extension at the plan's
//!   maximum rows per frame, spans, tree nodes and per-row items, rounded up to 64. The
//!   extension counts against each part's row limit (§6.3).
//! - **Token slots.** [`token_slot`] is `max_batch × (48 B + 4 B × T_max + logprob bytes + draft
//!   bytes)` plus the frame's own header, rounded up to 4 KiB, and on a parity sequence's ingress
//!   channel the tap block for one prefill chunk at the spec's k on top. A larger step travels in
//!   parts (§6.5).
//! - **A channel's slots.** [`channel_slots`] is what one side of a channel holds: the receive
//!   slots its credit allows, the four the channel core keeps for control frames (§7.1), and the
//!   frames it retains for retransmission (§7.3).
//!
//! Every figure here is computed from the frame layouts of RFC-0031 §6, through the codec's own
//! constants, so a layout change moves the sizes with it.

use crate::frame::{
    ACTIVATION_FIXED_LEN, ALIGN, ENTRY_HEADER_LEN, HEADER_LEN, SPAN_LEN, TLV_HEADER_LEN,
    TOKEN_FIXED_LEN, TREE_NODE_LEN,
};
use crate::transport::core::CONTROL_RECEIVES;

/// The rounding of a token or activation slot (RFC-0031 §8): 4 KiB.
pub const PAGE: usize = 4096;

/// The head a `ROUTER_HINT` item's value starts with: `layer u16, k u8`, padded to 4.
const ROUTER_HINT_HEAD: usize = 4;

/// The head of a tap block (RFC-0031 §6.5): `mode, reserved, n_pos, k, tail`.
const TAP_HEAD: usize = 8;

/// A tapped position's fixed fields: position, argmax, margin, reserved, `lse`, `lse_complement`.
const TAP_POSITION_FIXED: usize = 32;

fn round(value: usize, to: usize) -> usize {
    value.next_multiple_of(to)
}

/// What the plan allows in one `Activation` frame of a hop, for its header area.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HeaderLimits {
    /// The most rows a frame carries: `batching.prefill_chunk` on the prefill lane, the batch
    /// times the verify window on the decode lane.
    pub max_rows: u32,
    /// The most spans, one per sequence, a frame carries.
    pub max_spans: u32,
    /// The most tree nodes of a verify frame's draft trees; 0 where the hop sends no tree.
    pub max_tree_nodes: u32,
    /// The frames carry `TOKEN_IDS`: a u32 per row.
    pub token_ids: bool,
    /// The frames carry `ROUTER_HINT` at this k: k u16 expert ids per row.
    pub router_k: Option<u8>,
    /// Further side items per span, such as `SEQ_META`, `SEQ_CTRL`, `TRUNCATE`, `TRACE` or EP
    /// row metadata: how many. Each takes a TLV header of its own and pads its value to 8.
    pub items_per_span: u32,
    /// The bytes of those items' values, per span, all together.
    pub item_bytes_per_span: u32,
    /// Side items of the whole frame (`span` 0xFFFF), such as a frame's `TRACE`: how many.
    pub frame_items: u32,
    /// The bytes of those items' values, all together.
    pub frame_item_bytes: u32,
}

/// The header area of RFC-0031 §8 for frames within `limits`: 64 B of base header plus the
/// extension at the limits, rounded up to 64. Every side item is counted with its own TLV header
/// and up to 7 bytes of padding, and a per-row item once per span, so a frame whose rows spread
/// over its spans still fits.
pub fn header_area(limits: &HeaderLimits) -> usize {
    let rows = limits.max_rows as usize;
    let spans = (limits.max_spans as usize).max(1);
    let mut ext =
        ACTIVATION_FIXED_LEN + spans * SPAN_LEN + limits.max_tree_nodes as usize * TREE_NODE_LEN;
    // An item cut per span has a header and up to 7 bytes of padding per span.
    let per_row_item = |value: usize| value + spans * (TLV_HEADER_LEN + 7);
    if limits.token_ids {
        ext += per_row_item(4 * rows);
    }
    if let Some(k) = limits.router_k {
        ext += per_row_item(ROUTER_HINT_HEAD + 2 * usize::from(k) * rows);
    }
    // Each item, per span or of the frame, has its header and up to 7 bytes of padding.
    let items = |count: u32, bytes: u32| count as usize * (TLV_HEADER_LEN + 7) + bytes as usize;
    ext += spans * items(limits.items_per_span, limits.item_bytes_per_span);
    ext += items(limits.frame_items, limits.frame_item_bytes);
    HEADER_LEN + round(ext, ALIGN)
}

/// A parity sequence's tap block, as the ingress copy of its tokens carries it (RFC-0031 §6.5,
/// RFC-0060 §5.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TapLimits {
    /// `TapMode::Gather { k, tail }` over `positions` positions: the reference top-k ids and
    /// `tail` tail samples per position.
    Gather { positions: u32, k: u16, tail: u16 },
    /// `TapMode::FullLocal` over `positions` positions: a shared-memory offset per position.
    FullLocal { positions: u32 },
}

impl TapLimits {
    /// The tap block's bytes.
    pub fn bytes(&self) -> usize {
        match *self {
            TapLimits::Gather { positions, k, tail } => {
                let per = round(
                    TAP_POSITION_FIXED + 4 * (usize::from(k) + usize::from(tail)),
                    8,
                );
                TAP_HEAD + positions as usize * per
            }
            TapLimits::FullLocal { positions } => {
                TAP_HEAD + positions as usize * (TAP_POSITION_FIXED + 8)
            }
        }
    }
}

/// What the plan allows in one step's `Token` frame on a channel, for its slot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TokenLimits {
    /// The most sequences in a step, the `max_batch` of RFC-0031 §8.
    pub max_batch: u32,
    /// `T_max`: the most tokens one entry accepts, `γ + 1` after a verify window, 1 for plain
    /// decode.
    pub max_tokens: u32,
    /// The most logprob alternatives per token, `logprob_k`; 0 for none. Each token with
    /// logprobs carries its chosen f32 and `k × (id u32, logprob f32)`.
    pub logprob_k: u8,
    /// The most drafted tokens per entry, on the ring-return copy; 0 where no tail-side drafter
    /// runs.
    pub max_draft: u32,
    /// The drafts are trees, with a node record per drafted token.
    pub draft_tree: bool,
    /// On a parity sequence's ingress channel: the tap block for one prefill chunk at the spec's
    /// k (RFC-0031 §8).
    pub tap: Option<TapLimits>,
}

/// A token slot of RFC-0031 §8 for frames within `limits`: the frame's header and extension, then
/// `max_batch` entries of `48 B + 4 B × T_max + logprob bytes + draft bytes`, each 8-aligned,
/// rounded up to 4 KiB, and the tap block on top on a parity channel.
pub fn token_slot(limits: &TokenLimits) -> usize {
    let tokens = limits.max_tokens as usize;
    let logprobs = if limits.logprob_k == 0 {
        0
    } else {
        tokens * (4 + 8 * usize::from(limits.logprob_k))
    };
    let draft = limits.max_draft as usize * if limits.draft_tree { 8 } else { 4 };
    let entry = round(ENTRY_HEADER_LEN + 4 * tokens + logprobs + draft, 8);
    let frame = HEADER_LEN + round(TOKEN_FIXED_LEN, ALIGN) + limits.max_batch as usize * entry;
    let tap = limits.tap.map_or(0, |tap| round(tap.bytes(), 8));
    round(frame + tap, PAGE)
}

/// The slots one side of a channel holds (RFC-0031 §8): `recv_slots` for the data the peer's
/// credit may send, four more for control frames, which credit never counts (§7.1), and
/// `retention` for the frames this side keeps until the peer acknowledges them (§7.3).
pub fn channel_slots(recv_slots: u16, retention: u16) -> u32 {
    u32::from(recv_slots) + u32::from(CONTROL_RECEIVES) + u32::from(retention)
}

/// The bytes of one side's slots: [`channel_slots`] × `slot_size`.
pub fn channel_bytes(recv_slots: u16, retention: u16, slot_size: usize) -> usize {
    channel_slots(recv_slots, retention) as usize * slot_size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_512_row_prefill_header_with_token_ids_and_router_hint_at_k_8_is_over_10_kib() {
        let limits = HeaderLimits {
            max_rows: 512,
            max_spans: 1,
            token_ids: true,
            router_k: Some(8),
            ..HeaderLimits::default()
        };
        // 64 + round64(64 + 32 + (2048 + 8 + 7) + (4 + 8192 + 8 + 7)) = 64 + 10,432. The codec's
        // own extension for such a frame is 10,368 bytes (`tests/pools`); the area allows each
        // per-row item its padding in every span.
        assert_eq!(header_area(&limits), 10_496);
        assert!(header_area(&limits) > 10 * 1024);
        assert_eq!(header_area(&limits) % 64, 0);
    }

    /// Review of #1833, B1: a span's items each take a TLV header. `SEQ_META` (10 B) and
    /// `TRUNCATE` (4 B) in one span need 256 B with the codec (`tests/pools`), not 192.
    #[test]
    fn each_side_item_of_a_span_takes_its_own_header() {
        let limits = HeaderLimits {
            max_rows: 1,
            max_spans: 1,
            items_per_span: 2,
            item_bytes_per_span: 14,
            ..HeaderLimits::default()
        };
        // 64 + round64(64 + 32 + 2 × 15 + 14) = 64 + 192.
        assert_eq!(header_area(&limits), 256);
        let with_trace = HeaderLimits {
            frame_items: 1,
            frame_item_bytes: 26,
            ..limits
        };
        // The frame's `TRACE` adds 8 + 7 + 26 bytes: 64 + round64(140 + 41).
        assert_eq!(
            header_area(&with_trace),
            64 + (140usize + 41).next_multiple_of(64)
        );
        let two_traces = HeaderLimits {
            frame_items: 4,
            frame_item_bytes: 4 * 26,
            ..limits
        };
        assert_eq!(
            header_area(&two_traces),
            64 + (140usize + 4 * 41).next_multiple_of(64)
        );
    }

    #[test]
    fn a_bare_header_is_the_base_header_and_the_fixed_fields() {
        assert_eq!(
            header_area(&HeaderLimits::default()),
            HEADER_LEN + ACTIVATION_FIXED_LEN + SPAN_LEN.next_multiple_of(64)
        );
    }

    #[test]
    fn a_token_slot_follows_the_batch_and_logprob_limits() {
        // Plain decode, no logprobs: 64 + 64 + 16 × 56 = 1,024, one page.
        let plain = TokenLimits {
            max_batch: 16,
            max_tokens: 1,
            ..TokenLimits::default()
        };
        assert_eq!(token_slot(&plain), PAGE);
        // γ = 8 and top-20 logprobs: each entry 48 + 36 + 9 × 164 = 1,560 bytes.
        let logprobs = TokenLimits {
            max_batch: 16,
            max_tokens: 9,
            logprob_k: 20,
            ..TokenLimits::default()
        };
        assert_eq!(
            token_slot(&logprobs),
            (128usize + 16 * 1_560).next_multiple_of(PAGE)
        );
        assert_eq!(token_slot(&logprobs) % PAGE, 0);
    }

    #[test]
    fn a_parity_channel_adds_the_tap_block_for_one_prefill_chunk() {
        // RFC-0031 §6.5: 32 + 4 × (256 + 64) = 1,312 B per position at K = 256, M = 64.
        let tap = TapLimits::Gather {
            positions: 512,
            k: 256,
            tail: 64,
        };
        assert_eq!(tap.bytes(), 8 + 512 * 1_312);
        let ingress = TokenLimits {
            max_batch: 1,
            max_tokens: 1,
            tap: Some(tap),
            ..TokenLimits::default()
        };
        // 64 + 64 + one 56-byte entry, then the 671,752-byte tap block, rounded up to 4 KiB.
        assert_eq!(
            token_slot(&ingress),
            (184usize + 671_752).next_multiple_of(PAGE)
        );
        let without = token_slot(&TokenLimits {
            tap: None,
            ..ingress
        });
        assert!(without < token_slot(&ingress));
        assert_eq!(TapLimits::FullLocal { positions: 2 }.bytes(), 8 + 2 * 40);
    }

    #[test]
    fn a_channel_holds_its_receives_four_control_receives_and_its_retention() {
        assert_eq!(channel_slots(4, 4), 12);
        assert_eq!(channel_bytes(4, 4, 4096), 12 * 4096);
        // A side that only receives keeps no retention, and still its control receives.
        assert_eq!(channel_slots(8, 0), 12);
    }
}
