// SPDX-License-Identifier: MIT OR Apache-2.0

//! The seams where other tasks' stages plug into the pipeline (RFC-0050 §2.2).
//!
//! The pipeline owns the order of the stages, the context check, the `SequenceOpen` it builds,
//! the rings, the transcript and the stream loop. What each stage computes belongs to other
//! tasks, which implement these traits:
//!
//! | Stage | Seam | Built by |
//! |---|---|---|
//! | 3 model resolution | [`Stages::resolve_model`] | PROD-E2-T08 |
//! | 5 template and tokenization | [`Stages::tokenize`] | PROD-E3-T10, COMP-E17 |
//! | 7 routing | [`Stages::route`] | PLC-E6-T19, PROD-E2-T08 |
//! | 8 admission | [`Stages::admit`] | RFC-0001 §8.4 rule 5 |
//! | 10 detokenizer and stop matcher | [`Detokenize`], [`StopMatch`] | PROD-E2-T03 |
//! | 10 encoder | [`EventSink`] | each shape's encoder, PROD-E2-T04 first |
//!
//! Stages 1 and 2, authentication and the front end, run before the pipeline: they produce the
//! [`GenerateRequest`](super::GenerateRequest). Stage 4's overrides (PROD-E1-T10) and stage 6's
//! constraint compile (PROD-E3-T05) are not in M0's pipeline yet; their tasks add them between
//! the stages they name. §2.3's token-ID range check runs after stage 5, with the
//! [`ModelInfo::vocab_limit`] stage 3 returns ([`check_request`](super::check_request)).

use syntropy_core::{InstanceId, PlanEpoch};

use super::logprobs::TailChannels;
use super::types::{ApiError, GenEvent, Input, ModelRef};

/// What the pipeline needs to know of a resolved model.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModelInfo {
    /// The model's name, as the API reports it.
    pub name: String,
    /// The largest prompt plus `max_tokens` the model takes.
    pub context_length: u32,
    /// `min(embedding rows, LM-head rows, sampler table size)`, from the model's manifest
    /// ([`manifest_vocab_limit`](super::manifest_vocab_limit); [`ModelInfo::from_manifest`]):
    /// every token ID a request carries must be below it (RFC-0050 §2.3).
    pub vocab_limit: u32,
}

/// Where a request goes (stage 7): the instance, and the router's choices that travel in
/// `SequenceOpen` (RFC-0001 §6.1 step 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Route {
    /// The instance.
    pub instance: InstanceId,
    /// Its plan epoch.
    pub epoch: PlanEpoch,
    /// The prefill chunk size; 0 takes the plan's.
    pub prefill_chunk: u32,
    /// The first verify window; `None` takes the plan's.
    pub initial_window: Option<u16>,
    /// The tail's channels to this ingress and to the head, which decide whether the tail is
    /// asked for the chosen tokens' logprobs alone (RFC-0050 §3.1). The default is a tail in
    /// this ingress's process.
    pub tail: TailChannels,
}

/// Stages 3, 5, 7 and 8, and the per-choice text state of stage 10.
pub trait Stages: Send + Sync {
    /// Stage 3: the model a reference names.
    ///
    /// # Errors
    ///
    /// The request's error, such as [`ApiError::ModelNotFound`].
    fn resolve_model(&self, model: &ModelRef) -> Result<ModelInfo, ApiError>;

    /// Stage 5: renders the input with the model's template and tokenizes it, content spans
    /// literally unless `raw_special_tokens` (RFC-0050 §6.4).
    ///
    /// # Errors
    ///
    /// The request's error, such as [`ApiError::InvalidRequest`].
    fn tokenize(
        &self,
        model: &ModelInfo,
        input: &Input,
        raw_special_tokens: bool,
    ) -> Result<Vec<u32>, ApiError>;

    /// Stage 7: the instance for `prompt`, which needs `need` tokens of context.
    ///
    /// # Errors
    ///
    /// The request's error, such as [`ApiError::Capacity`] when no instance can take it.
    fn route(&self, model: &ModelInfo, prompt: &[u32], need: u64) -> Result<Route, ApiError>;

    /// Stage 8: KV admission on the routed instance (RFC-0001 §8.4 rule 5).
    ///
    /// # Errors
    ///
    /// [`ApiError::Capacity`] when the instance cannot take the sequence now.
    fn admit(&self, route: &Route, prompt_tokens: u32, max_tokens: u32) -> Result<(), ApiError>;

    /// Stage 10: a fresh incremental detokenizer for one choice.
    fn detokenizer(&self, model: &ModelInfo) -> Box<dyn Detokenize>;

    /// Stage 10: a fresh stop matcher for one choice. Called only when `stops` is not empty.
    fn stop_matcher(&self, stops: &[String]) -> Box<dyn StopMatch>;
}

/// An incremental detokenizer for one choice (RFC-0050 §5.2).
pub trait Detokenize: Send {
    /// Adds one token and returns the text it releases, which never ends in an incomplete
    /// UTF-8 sequence.
    fn push(&mut self, token: u32) -> String;

    /// Releases whatever is held back, at the end of the choice.
    fn finish(&mut self) -> String;
}

/// What a stop matcher released for one piece of text.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct StopStep {
    /// Text that is safe to deliver: it cannot be part of a stop string.
    pub release: String,
    /// The stop string that matched, if one did. `release` then holds the text before it, and
    /// nothing after it is ever released.
    pub matched: Option<String>,
}

/// A stop-string matcher with hold-back, for one choice (RFC-0050 §5.2).
pub trait StopMatch: Send {
    /// Adds released text.
    fn push(&mut self, text: &str) -> StopStep;

    /// Releases the held-back text at the end of a choice that matched no stop string.
    fn finish(&mut self) -> String;
}

/// The client went away: the connection closed, or a write to it failed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ClientGone;

/// Where a request's events go: the shape's encoder, writing to the client (stage 10).
pub trait EventSink {
    /// Encodes and writes one event.
    ///
    /// # Errors
    ///
    /// [`ClientGone`] when the client cannot be written to; the pipeline then cancels the
    /// sequence and writes nothing more.
    fn emit(&mut self, event: GenEvent) -> Result<(), ClientGone>;
}

/// The matcher for a choice without stop strings: it releases everything at once.
#[derive(Debug, Default)]
pub(super) struct NoStops;

impl StopMatch for NoStops {
    fn push(&mut self, text: &str) -> StopStep {
        StopStep {
            release: text.to_owned(),
            matched: None,
        }
    }

    fn finish(&mut self) -> String {
        String::new()
    }
}
