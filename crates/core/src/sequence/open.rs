// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sequence control: [`SequenceControl`], [`SequenceOpen`] and what an open carries (RFC-0001
//! §5.8; RFC-0032 §3.6, which owns `SequenceControl`).

use std::fmt;
use std::ops::Deref;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use smallvec::SmallVec;

use super::{Priority, TraceContext};
use crate::{
    Blake3, ExactnessPolicy, Nanos, NodeId, ParityTapSpec, PlanEpoch, SequenceId, SessionId,
};

/// Sequence control on the data plane (RFC-0001 §5.8, and RFC-0033 §11's `Reconcile`). Owner:
/// RFC-0032.
// `Open` is far larger than the other variants. Control messages are sent once or twice per
// sequence, not per step, so the size costs nothing, and the variant keeps RFC-0001 §5.8's
// `Open(SequenceOpen)` shape.
#[allow(clippy::large_enum_variant)]
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SequenceControl {
    /// Ingress → head: a new sequence, or a resumed one (`SequenceOpen::resume`).
    Open(SequenceOpen),
    /// Ingress → every stage: stop at once. The tail stops emitting, and ingress discards any
    /// tokens still in flight (RFC-0032 §3.6).
    Cancel {
        /// The sequence.
        seq: SequenceId,
        /// Why.
        reason: CancelReason,
    },
    /// Ingress → head: stop issuing steps for the sequence, keeping its state at every stage
    /// (a slow client, RFC-0001 §8.4; the detach grace, RFC-0033 §9). Only the head composes
    /// steps (ADR-0106).
    Pause {
        /// The sequence.
        seq: SequenceId,
    },
    /// Ingress → head: issue steps again after a `Pause`.
    Resume {
        /// The sequence.
        seq: SequenceId,
    },
    /// Ingress → every stage: free the sequence's KV and activation history, or persist the KV
    /// as `persist` (RFC-0022).
    Release {
        /// The sequence.
        seq: SequenceId,
        /// The session to persist the KV as; `None` frees it.
        persist: Option<SessionId>,
    },
    /// Ingress → head, in recovery: what the client has seen of the sequence, from the ingress
    /// transcript (RFC-0033 §4.2, §11). Ingress sends it once the repair `PlanCommitted` is
    /// applied and the new epoch's head reports `Ready`. Every stage then holds positions up to
    /// `prompt_len + emitted − 1`, and the head resumes there. A new head, which has no
    /// sequences, gets an `Open` with `resume` set instead (RFC-0033 §4.4).
    Reconcile {
        /// The sequence.
        seq: SequenceId,
        /// Tokens the client has received.
        emitted: u32,
        /// The sequence's draw counter after the last emitted token, from the last `TokenEntry`
        /// ingress received (RFC-0032 §3.7).
        rng_counter: u64,
        /// The last emitted tokens, up to 8, in the order they were emitted.
        recent: SmallVec<[u32; 8]>,
    },
}

/// Why ingress cancels a sequence (RFC-0001 §5.8, `SequenceControl::Cancel`; RFC-0001 §5.8's
/// note of 2026-10-22 derives the set). A sequence that ends because its ingress was lost, or
/// because a detach grace ran out, ends with `FinishReason::Cancelled` and a `Release` instead
/// (RFC-0033 §9.2).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CancelReason {
    /// The client's connection closed (RFC-0050 §2.2).
    ClientGone,
    /// The client asked, for example with `POST /v1/responses/{id}/cancel` (RFC-0050 §3).
    ClientRequest,
    /// Ingress matched a stop string, which can span tokens (RFC-0012 §8.4).
    StopString,
    /// The request's deadline passed (`SequenceLimits::deadline`).
    Deadline,
    /// A sequence-scoped error at a stage (RFC-0001 §10.4), which the tail reports as
    /// `FinishReason::Error` with the same domain and code.
    Error {
        /// RFC-0001 §10's `ErrorCode` domain, in its numeric form.
        domain: u16,
        /// The code within the domain.
        code: u16,
    },
}

/// The prompt's token IDs, tokenized at ingress (`SequenceOpen::prompt`). Owned, so `core` stays
/// plain data; a zero-copy view of a ring slot belongs to `ipc` or `net`.
///
/// **Serde form.** A human-readable format, such as JSON, gets an array of the IDs. A binary
/// format gets a byte string of the IDs as little-endian `u32`s, 4 bytes each, so a prompt's size
/// depends only on its length: a 128K-token prompt is 512 KiB (RFC-0031 §6.6). That covers CBOR
/// on the wire, the IPC's postcard and [`canon`](crate::canon). A byte string whose length is not
/// a multiple of 4 is refused.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TokenSlice(pub Vec<u32>);

impl Serialize for TokenSlice {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if serializer.is_human_readable() {
            self.0.serialize(serializer)
        } else {
            let bytes: Vec<u8> = self.0.iter().flat_map(|id| id.to_le_bytes()).collect();
            serializer.serialize_bytes(&bytes)
        }
    }
}

impl<'de> Deserialize<'de> for TokenSlice {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<TokenSlice, D::Error> {
        if deserializer.is_human_readable() {
            Vec::deserialize(deserializer).map(TokenSlice)
        } else {
            deserializer.deserialize_byte_buf(TokenBytes)
        }
    }
}

/// Reads a binary format's byte string of little-endian `u32` token IDs.
struct TokenBytes;

impl Visitor<'_> for TokenBytes {
    type Value = TokenSlice;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a byte string of little-endian u32 token IDs, 4 bytes each")
    }

    fn visit_bytes<E: de::Error>(self, bytes: &[u8]) -> Result<TokenSlice, E> {
        let (ids, rest) = bytes.as_chunks::<4>();
        if !rest.is_empty() {
            return Err(E::invalid_length(bytes.len(), &self));
        }
        Ok(TokenSlice(
            ids.iter().map(|id| u32::from_le_bytes(*id)).collect(),
        ))
    }

    fn visit_byte_buf<E: de::Error>(self, bytes: Vec<u8>) -> Result<TokenSlice, E> {
        self.visit_bytes(&bytes)
    }
}

impl TokenSlice {
    /// The token IDs.
    pub fn as_slice(&self) -> &[u32] {
        &self.0
    }
}

impl Deref for TokenSlice {
    type Target = [u32];

    fn deref(&self) -> &[u32] {
        &self.0
    }
}

impl From<Vec<u32>> for TokenSlice {
    fn from(tokens: Vec<u32>) -> TokenSlice {
        TokenSlice(tokens)
    }
}

/// A page key of the prefix index: `k_b`, the BLAKE3 chain over a prefix's pages (RFC-0022
/// §7.1), named in `PrefixRef::Cached` for the last full page a sequence shares. TIER-E7-T01
/// computes it. Owner: RFC-0022.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PrefixKey(pub Blake3);

/// A partial page hanging off a prefix-index node: an index into the prefix index's tail pool
/// (RFC-0022 §7.2). Owner: RFC-0022.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TailId(pub u32);

/// What the head found in its prefix index (RFC-0022 §7) or session index (§8). Every stage
/// attaches it to the sequence's new KV handle. Owner: RFC-0022.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrefixRef {
    /// A cached prefix: its last full page's key, the tokens it covers, the partial page after
    /// it if any, and the plan epoch it was cached under.
    Cached {
        /// The last full page's key.
        last_full: PrefixKey,
        /// Tokens covered.
        tokens: u32,
        /// The partial page after the last full one.
        #[serde(default)]
        tail: Option<TailId>,
        /// The epoch the pages were cached under.
        epoch: PlanEpoch,
    },
    /// A persisted session: its tokens and the blob that holds them.
    Session {
        /// The session.
        session: SessionId,
        /// Tokens covered.
        tokens: u32,
        /// The session blob's digest.
        blob: Blake3,
    },
}

/// The principal whose prefix-cache pages a sequence may share, and how far its prefix is
/// shared (RFC-0050 amendment A7; RFC-0022 §7.1). The head derives the prefix key's
/// `scope_salt` from it, so one principal's cached tokens reveal nothing about another's
/// prompts. Default: the local principal, nothing shared beyond it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PrefixScope {
    /// The principal, an API key or local user, hashed; `None` is the local principal.
    #[serde(default)]
    pub principal: Option<[u8; 16]>,
    /// The token index up to which a prefix the client marked with `cache_control` is shared:
    /// pages wholly inside it are keyed with the empty salt.
    #[serde(default)]
    pub shared_until: Option<u32>,
}

/// A compiled grammar, named by the BLAKE3 of its compiled form (RFC-0050 §7, amendment A3). The
/// first `SequenceOpen` that uses a hash carries the grammar inline
/// (`SamplingParams::grammar_inline`), so a tail compiles each once (PROD-E3-T05). Owner:
/// RFC-0050.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GrammarRef(pub Blake3);

/// How an embedding sequence pools its final hidden states (RFC-0043 §5.4's
/// `EmbeddingSpec::pooling`; RFC-0050 amendment A5). COMP-E16-T10's `EmbeddingSpec` uses it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PoolingMode {
    /// The last token's state.
    LastToken,
    /// The mean over the sequence's tokens.
    Mean,
    /// The first (`[CLS]`) token's state.
    Cls,
}

/// The sampler's parameters (RFC-0001 §5.8; RFC-0012 §8.1–§8.3; RFC-0050 §2.1 and amendment
/// A3). Defaults are RFC-0054 §5's fallbacks when the model gives none: temperature 1, top-p 1,
/// top-k and min-p off, no penalties. Owner: RFC-0012.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SamplingParams {
    /// The key's first component: the client's seed, or 64 random bits ingress picked and
    /// recorded (RFC-0012 §8.2).
    pub seed: u64,
    /// Temperature; 0 is greedy (`Argmax`, no draws).
    pub temperature: f32,
    /// Nucleus sampling's mass; 1 is off.
    pub top_p: f32,
    /// Top-k; 0 is off.
    pub top_k: u32,
    /// Min-p; 0 is off.
    pub min_p: f32,
    /// Presence penalty, over emitted tokens (RFC-0050 §2.2); 0 is off.
    pub presence_penalty: f32,
    /// Frequency penalty, over emitted tokens; 0 is off.
    pub frequency_penalty: f32,
    /// Repetition penalty, over the prompt and the emitted tokens; 1 is off.
    pub repetition_penalty: f32,
    /// Logit bias per token ID.
    #[serde(default)]
    pub logit_bias: SmallVec<[(u32, f32); 4]>,
    /// Top-k alternatives per token to return in `TokenEntry::logprobs`; `None` returns none, and
    /// `Some(0)` the chosen tokens' logprobs alone (OpenAI's `top_logprobs = 0`). The wire cannot
    /// carry `Some(0)` until OQ-NET-35 settles (`LogprobBlock::k`).
    #[serde(default)]
    pub logprobs: Option<u8>,
    /// The output constraint, by reference.
    #[serde(default)]
    pub constraint: Option<GrammarRef>,
    /// The compiled grammar, on the first `SequenceOpen` that uses `constraint`'s hash.
    #[serde(default)]
    pub grammar_inline: Option<Vec<u8>>,
    /// Reasoning tokens allowed before the tail forces the family's end-of-reasoning marker
    /// (RFC-0050 §8.3).
    #[serde(default)]
    pub reasoning_budget: Option<u32>,
}

impl Default for SamplingParams {
    fn default() -> SamplingParams {
        SamplingParams {
            seed: 0,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            logit_bias: SmallVec::new(),
            logprobs: None,
            constraint: None,
            grammar_inline: None,
            reasoning_budget: None,
        }
    }
}

/// A sequence's limits (RFC-0001 §5.8; RFC-0012 §8.4; RFC-0050 amendment A5).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SequenceLimits {
    /// Tokens to emit at most, a resumed sequence's emitted ones included (`ResumePoint`).
    pub max_tokens: u32,
    /// The request's stop token IDs, which the tail checks beside the model's end-of-sequence
    /// token and the template's end-of-turn set (RFC-0012 §8.4).
    #[serde(default)]
    pub stop_tokens: SmallVec<[u32; 4]>,
    /// Time from `SequenceOpen` to the request's deadline; ingress cancels at it
    /// (`CancelReason::Deadline`).
    #[serde(default)]
    pub deadline: Option<Nanos>,
    /// A prefill-only embedding sequence: the tail pools its final hidden states this way and
    /// returns an `EmbeddingResult` instead of tokens.
    #[serde(default)]
    pub embed: Option<PoolingMode>,
}

/// Where a resumed sequence continues (RFC-0032 §3.6; RFC-0033 §5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResumePoint {
    /// Tokens already delivered: counted against `max_tokens` and never emitted again.
    pub emitted: u32,
    /// The sequence's draw counter after the last emitted token (RFC-0032 §3.7).
    pub rng_counter: u64,
    /// Why the sequence moved.
    pub cause: ResumeCause,
}

/// Why a sequence was re-opened (RFC-0032 §3.6).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResumeCause {
    /// Moved to another instance or tail (RFC-0033 §8).
    Migration,
    /// Re-opened after a failure (RFC-0033 §4, §5).
    Recovery,
    /// Moved to balance load.
    Rebalance,
}

/// Opens a sequence at the head (RFC-0001 §5.8, with RFC-0032 §3.6's additions). Every field
/// after `priority` was added later and has a default, so an open encoded without it still
/// decodes. Owner: RFC-0032.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SequenceOpen {
    /// The sequence.
    pub seq: SequenceId,
    /// The session whose KV this sequence extends, if any.
    #[serde(default)]
    pub session: Option<SessionId>,
    /// The ingress node: where the tokens go.
    pub ingress: NodeId,
    /// The prompt's token IDs. For a resumed sequence, the original prompt followed by every
    /// token already emitted (RFC-0032 §3.6).
    pub prompt: TokenSlice,
    /// The head's prefix-cache or session hit (RFC-0022).
    #[serde(default)]
    pub prefix: Option<PrefixRef>,
    /// The sampler's parameters.
    pub sampling: SamplingParams,
    /// Max tokens, stop tokens, deadline and embedding.
    pub limits: SequenceLimits,
    /// Per-sequence approximate knobs only; instance-wide knobs come from the plan.
    #[serde(default)]
    pub exactness: ExactnessPolicy,
    /// Scheduling priority.
    #[serde(default)]
    pub priority: Priority,
    /// `Some(i)` when the client supplied the seed: the sampler key is (seed, i), so a repeated
    /// request reproduces. `None`: (seed, `SequenceId`) (RFC-0032 §3.7).
    #[serde(default)]
    pub choice_index: Option<u16>,
    /// The router's prefill chunk size, applied by the head (RFC-0011 §13); 0 takes the plan's.
    #[serde(default)]
    pub prefill_chunk: u32,
    /// The router's first verify window; `None` takes the plan's default.
    #[serde(default)]
    pub initial_window: Option<u16>,
    /// A continuation after migration or recovery.
    #[serde(default)]
    pub resume: Option<ResumePoint>,
    /// Whose prefix-cache pages the sequence may share (RFC-0050 amendment A7).
    #[serde(default)]
    pub prefix_scope: PrefixScope,
    /// From M2, the session's X25519 public key: a stage that saves its part of the session
    /// seals the blob to it (RFC-0034 §13).
    #[serde(default)]
    pub session_pk: Option<[u8; 32]>,
    /// The trace context of the request's scheduling spans (RFC-0053 §3.2), with flags from
    /// Syntropy's policy alone.
    #[serde(default)]
    pub trace: Option<TraceContext>,
    /// A teacher-forced parity sequence (RFC-0060 §5.5); accepted only with `debug.parity_api`.
    #[serde(default)]
    pub parity: Option<ParityTapSpec>,
}

impl SequenceOpen {
    /// The metadata the head forwards on the sequence's first frame: every field but the prompt
    /// tokens, and their count (RFC-0032 §3.6).
    pub fn meta(&self) -> SequenceMeta {
        SequenceMeta {
            seq: self.seq,
            session: self.session,
            ingress: self.ingress,
            // A prompt is far shorter than 2^32 tokens; ingress refuses anything above the
            // model's context length long before.
            prompt_len: u32::try_from(self.prompt.len()).unwrap_or(u32::MAX),
            prefix: self.prefix,
            sampling: self.sampling.clone(),
            limits: self.limits.clone(),
            exactness: self.exactness,
            priority: self.priority,
            choice_index: self.choice_index,
            prefill_chunk: self.prefill_chunk,
            initial_window: self.initial_window,
            resume: self.resume,
            prefix_scope: self.prefix_scope,
            session_pk: self.session_pk,
            trace: self.trace,
            parity: self.parity.clone(),
        }
    }
}

/// What travels downstream in the `SEQ_META` item on a sequence's first frame (RFC-0031 §6.3):
/// every field of the [`SequenceOpen`] except the prompt tokens, plus the prompt's length
/// (RFC-0032 §3.6). Every stage allocates the sequence's KV when it arrives, and the tail learns
/// its sampler and ingress node from it. Build one with [`SequenceOpen::meta`]. Owner:
/// RFC-0032.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SequenceMeta {
    /// As [`SequenceOpen::seq`].
    pub seq: SequenceId,
    /// As [`SequenceOpen::session`].
    #[serde(default)]
    pub session: Option<SessionId>,
    /// As [`SequenceOpen::ingress`].
    pub ingress: NodeId,
    /// The prompt's length in tokens.
    pub prompt_len: u32,
    /// As [`SequenceOpen::prefix`].
    #[serde(default)]
    pub prefix: Option<PrefixRef>,
    /// As [`SequenceOpen::sampling`].
    pub sampling: SamplingParams,
    /// As [`SequenceOpen::limits`].
    pub limits: SequenceLimits,
    /// As [`SequenceOpen::exactness`].
    #[serde(default)]
    pub exactness: ExactnessPolicy,
    /// As [`SequenceOpen::priority`].
    #[serde(default)]
    pub priority: Priority,
    /// As [`SequenceOpen::choice_index`].
    #[serde(default)]
    pub choice_index: Option<u16>,
    /// As [`SequenceOpen::prefill_chunk`].
    #[serde(default)]
    pub prefill_chunk: u32,
    /// As [`SequenceOpen::initial_window`].
    #[serde(default)]
    pub initial_window: Option<u16>,
    /// As [`SequenceOpen::resume`].
    #[serde(default)]
    pub resume: Option<ResumePoint>,
    /// As [`SequenceOpen::prefix_scope`].
    #[serde(default)]
    pub prefix_scope: PrefixScope,
    /// As [`SequenceOpen::session_pk`].
    #[serde(default)]
    pub session_pk: Option<[u8; 32]>,
    /// As [`SequenceOpen::trace`].
    #[serde(default)]
    pub trace: Option<TraceContext>,
    /// As [`SequenceOpen::parity`].
    #[serde(default)]
    pub parity: Option<ParityTapSpec>,
}
