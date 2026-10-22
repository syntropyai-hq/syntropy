// SPDX-License-Identifier: MIT OR Apache-2.0

//! What the tail sends back: [`TokenFrame`] and its [`TokenEntry`]s (RFC-0001 §5.8; RFC-0031
//! §6.5 owns the layout, RFC-0012 the semantics).

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::parity::{LogitTap, tap_float};
use crate::{ApproxFlags, InstanceId, PlanEpoch, SequenceId};

/// One step's tokens from the tail: to the head on the ring return, for every sequence of the
/// step, and to each ingress node for its own (RFC-0001 §5.8). `net`'s `frame::Token<'a>` is a
/// `Token` frame's zero-copy view of one. Owner: RFC-0031 (layout), RFC-0012 (semantics).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TokenFrame {
    /// The instance the step ran on.
    pub instance: InstanceId,
    /// The plan epoch of the step.
    pub epoch: PlanEpoch,
    /// The head's step number.
    pub step: u64,
    /// One entry per sequence of the step that this copy carries.
    pub entries: SmallVec<[TokenEntry; 8]>,
}

/// One sequence's tokens from one step (RFC-0001 §5.8; RFC-0031 §6.5's entry). `net`'s
/// `frame::TokenEntryView<'a>` is one entry's zero-copy view; it converts to this type and is
/// encoded from it. Owner: RFC-0031 (layout), RFC-0012 (semantics).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TokenEntry {
    /// The sequence.
    pub seq: SequenceId,
    /// Accepted tokens in order: one for plain decode, 1..=γ+1 after a verify window.
    pub tokens: SmallVec<[u32; 8]>,
    /// The position of the first accepted token.
    pub first_position: u32,
    /// The accepted tokens' log-probabilities, when the sequence asked for them
    /// (`SamplingParams::logprobs`).
    #[serde(default)]
    pub logprobs: Option<LogprobBlock>,
    /// Set when the sequence finished at this step.
    #[serde(default)]
    pub finish: Option<FinishReason>,
    /// The approximate knobs that touched these tokens; empty in exact mode (D3; RFC-0001 §11.3).
    #[serde(default)]
    pub approx: ApproxFlags,
    /// The tail-side drafter's draft for the next window (RFC-0032 §3.5), on the ring-return copy
    /// only: one token per drafted position, or per tree node.
    #[serde(default)]
    pub draft: SmallVec<[u32; 8]>,
    /// The draft's tree, one node per drafted token, when the draft is a tree; `None` for a chain
    /// (RFC-0031 §6.5's `DRAFT_TREE`).
    #[serde(default)]
    pub draft_tree: Option<SmallVec<[DraftNode; 8]>>,
    /// The sequence's sampler draw counter after these tokens (RFC-0001 §5.9; RFC-0032 §3.7),
    /// stored by ingress in the transcript so recovery and migration continue the same random
    /// stream.
    pub rng_counter: u64,
    /// Teacher-forced logit tap for parity runs (RFC-0060 §5.5), on the ingress copy only; `None`
    /// outside them.
    #[serde(default)]
    pub tap: Option<LogitTap>,
}

/// Why a sequence finished (RFC-0001 §5.8; RFC-0031 §6.5's `finish` codes 1 to 5). `net`'s own
/// `Finish` moved here (DIST-E1-T14).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FinishReason {
    /// The model emitted its end-of-sequence token, or a token of the template's end-of-turn set
    /// (RFC-0012 §8.4).
    Stop,
    /// `SequenceLimits::max_tokens` was reached.
    Length,
    /// A token of the request's stop set (`SequenceLimits::stop_tokens`) was emitted.
    TokenStop,
    /// The sequence was cancelled, by `SequenceControl::Cancel` or because its ingress was lost
    /// (RFC-0033 §9.2).
    Cancelled,
    /// A sequence-scoped error at a stage (RFC-0001 §10.4): RFC-0001 §10's `ErrorCode`, in the
    /// wire's numeric form, a domain and a code within it.
    Error {
        /// The error's domain.
        domain: u16,
        /// The code within the domain.
        code: u16,
    },
}

/// The accepted tokens' log-probabilities (RFC-0031 §6.5's layout: per token, the chosen token's
/// logprob, then `k` alternatives as (id, logprob)). The floats keep non-finite values through
/// serde, as a parity tap's do.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LogprobBlock {
    /// Top-k alternatives per token, at least 1: RFC-0031 §6.5's `logprob_k`, where 0 means the
    /// entry carries no logprobs.
    pub k: u8,
    /// Per accepted token, the chosen token's logprob.
    #[serde(with = "tap_float::seq")]
    pub chosen: Vec<f32>,
    /// Per accepted token, its `k` alternatives' token IDs, best first: `chosen.len() × k`
    /// values.
    pub top_ids: Vec<u32>,
    /// The alternatives' logprobs, in the same order as `top_ids`.
    #[serde(with = "tap_float::seq")]
    pub top_logprobs: Vec<f32>,
}

impl LogprobBlock {
    /// Whether the block can be carried: `k` is at least 1, and `top_ids` and `top_logprobs` hold
    /// `k` values per chosen token.
    pub fn is_consistent(&self) -> bool {
        let n = self.chosen.len() * usize::from(self.k);
        self.k > 0 && self.top_ids.len() == n && self.top_logprobs.len() == n
    }
}

/// One node of a draft tree (RFC-0031 §6.3's tree area; RFC-0032 §3.5): `net`'s `TreeNode`,
/// which re-exports this type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DraftNode {
    /// The parent's index among the draft's nodes, or `None` when the parent is the root, the
    /// last accepted token. A parent precedes its children.
    pub parent: Option<u16>,
    /// 1 under the root, else the parent's depth plus one.
    pub depth: u8,
    /// Bit 0 [`DraftNode::PROMPT_LOOKUP`]; the other bits are reserved.
    pub flags: u8,
}

impl DraftNode {
    /// The node comes from the prompt-lookup drafter.
    pub const PROMPT_LOOKUP: u8 = 1 << 0;

    /// Whether the node comes from the prompt-lookup drafter.
    pub fn prompt_lookup(&self) -> bool {
        self.flags & DraftNode::PROMPT_LOOKUP != 0
    }
}
