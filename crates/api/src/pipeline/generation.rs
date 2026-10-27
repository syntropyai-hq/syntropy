// SPDX-License-Identifier: MIT OR Apache-2.0

//! One request through stages 3 to 11 of RFC-0050 §2.2.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use syntropy_core::{
    CancelReason, FinishReason, MonoTime, NodeId, PrefixScope, SamplingParams, SequenceControl,
    SequenceId, SequenceLimits, SequenceOpen, TokenEntry,
};

use super::ingress::{Ingress, SendError};
use super::stages::{ClientGone, Detokenize, EventSink, NoStops, Route, StopMatch};
use super::types::{ApiError, GenEvent, GenerateRequest, StopReason, Timings, Usage};

/// Set by the HTTP layer when the client's connection closes (RFC-0050 §5.3): on the socket's
/// error event, or an HTTP/2 `RST_STREAM`. Cloned freely; every clone sets the same flag.
#[derive(Clone, Debug, Default)]
pub struct ClientWatch(Arc<AtomicBool>);

impl ClientWatch {
    /// Records that the client is gone.
    pub fn client_gone(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// Whether the client is gone.
    #[must_use]
    pub fn is_gone(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// How a generation ended.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Outcome {
    /// The choice finished and the client was told.
    Finished(StopReason),
    /// The request failed after it started, and the client was told.
    Failed(ApiError),
    /// The client went away; the sequence was cancelled and released.
    ClientGone,
}

/// What [`Generation::poll`] did.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Progress {
    /// Everything that has arrived is delivered; poll again when more may have.
    Pending,
    /// The generation is over, and its sequence released.
    Ended(Outcome),
}

/// One request's sequence, from `SequenceOpen` to `Release` (stages 9 to 11).
///
/// [`Ingress::start`] runs stages 3 to 9 and returns it. [`Generation::poll`] runs stage 10 over
/// whatever has arrived on `local_tokens`, and stage 11 when the sequence ends. It never blocks:
/// the connection's task polls it when the ring may have moved. Dropping a generation that has
/// not ended cancels and releases its sequence, as a disconnect does.
pub struct Generation<'i, 'r> {
    ingress: &'i Ingress<'r>,
    seq: SequenceId,
    route: Route,
    prompt_tokens: u32,
    seed: u64,
    detok: Box<dyn Detokenize>,
    stops: Box<dyn StopMatch>,
    watch: ClientWatch,
    begun: MonoTime,
    first_token: Option<MonoTime>,
    completion_tokens: u32,
    announced: bool,
    ended: Option<Outcome>,
}

impl std::fmt::Debug for Generation<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Generation")
            .field("seq", &self.seq)
            .field("completion_tokens", &self.completion_tokens)
            .field("ended", &self.ended)
            .finish_non_exhaustive()
    }
}

impl<'r> Ingress<'r> {
    /// Stages 3 to 9 for `request`: resolves the model, tokenizes the prompt, checks the context,
    /// routes, admits, and sends `SequenceOpen` on `local_sequences`. `watch` is the flag the
    /// HTTP layer sets when the client goes away.
    ///
    /// # Errors
    ///
    /// The first stage's error, before any `SequenceOpen` is sent:
    /// - [`ApiError::InvalidRequest`] for `n = 0` or `max_tokens = 0`;
    /// - [`ApiError::Unsupported`] for `n > 1` (PROD-E2-T06), or a prompt larger than a ring
    ///   slot;
    /// - [`ApiError::ContextLengthExceeded`] when the prompt plus `max_tokens` exceeds the model's
    ///   context;
    /// - [`ApiError::Capacity`] when `local_sequences` is full;
    /// - [`ApiError::RunnerUnavailable`] when the runner is gone;
    /// - [`ApiError::SequenceIdsExhausted`] when the node has minted every ID of this boot;
    /// - any error a stage returns.
    pub fn start(
        &self,
        request: &GenerateRequest,
        watch: ClientWatch,
    ) -> Result<Generation<'_, 'r>, ApiError> {
        let begun = self.clock.now();
        match request.n {
            0 => return Err(ApiError::InvalidRequest("n must be at least 1".to_owned())),
            1 => {}
            n => {
                return Err(ApiError::Unsupported(format!(
                    "n = {n}: one choice per request until PROD-E2-T06"
                )));
            }
        }
        let stages = self.stages;
        // Stage 3.
        let model = stages.resolve_model(&request.model)?;
        // Stage 5.
        let prompt = stages.tokenize(&model, &request.input, request.raw_special_tokens)?;
        let prompt_len = u32::try_from(prompt.len()).unwrap_or(u32::MAX);
        let limit = model.context_length;
        let max_tokens = match request.sampling.max_tokens {
            Some(0) => {
                return Err(ApiError::InvalidRequest(
                    "max_tokens must be at least 1".to_owned(),
                ));
            }
            Some(max) => max,
            None => limit.saturating_sub(prompt_len),
        };
        let need = u64::from(prompt_len) + u64::from(max_tokens.max(1));
        if need > u64::from(limit) {
            return Err(ApiError::ContextLengthExceeded { need, limit });
        }
        // Stages 7 and 8.
        let route = stages.route(&model, &prompt, need)?;
        stages.admit(&route, prompt_len, max_tokens)?;
        // Stage 9.
        let seed = request.sampling.seed.unwrap_or_else(|| self.entropy.u64());
        let seq = self.minter.mint()?;
        let open = open(
            request,
            seq,
            self.minter.node(),
            prompt.clone(),
            seed,
            max_tokens,
            &route,
        );
        let detok = stages.detokenizer(&model);
        let stops: Box<dyn StopMatch> = if request.sampling.stop.is_empty() {
            Box::new(NoStops)
        } else {
            stages.stop_matcher(&request.sampling.stop)
        };
        // Live before the open is sent, so no entry for it can be taken for a stray.
        self.register(seq, prompt);
        if let Err(error) = self.send(&SequenceControl::Open(open)) {
            self.unregister(seq);
            return Err(match error {
                SendError::Full => {
                    ApiError::Capacity("the runner's sequence ring is full".to_owned())
                }
                SendError::TooLong => ApiError::Unsupported(format!(
                    "a prompt of {prompt_len} tokens does not fit the runner's sequence ring"
                )),
                SendError::Broken(why) => ApiError::RunnerUnavailable(why),
            });
        }
        Ok(Generation {
            ingress: self,
            seq,
            route,
            prompt_tokens: prompt_len,
            seed,
            detok,
            stops,
            watch,
            begun,
            first_token: None,
            completion_tokens: 0,
            announced: false,
            ended: None,
        })
    }
}

/// Stage 9's `SequenceOpen`.
fn open(
    request: &GenerateRequest,
    seq: SequenceId,
    node: NodeId,
    prompt: Vec<u32>,
    seed: u64,
    max_tokens: u32,
    route: &Route,
) -> SequenceOpen {
    let asked = &request.sampling;
    let fallback = SamplingParams::default();
    SequenceOpen {
        seq,
        session: None,
        ingress: node,
        prompt: prompt.into(),
        prefix: None,
        sampling: SamplingParams {
            seed,
            temperature: asked.temperature.unwrap_or(fallback.temperature),
            top_p: asked.top_p.unwrap_or(fallback.top_p),
            top_k: asked.top_k.unwrap_or(fallback.top_k),
            min_p: asked.min_p.unwrap_or(fallback.min_p),
            presence_penalty: asked.presence_penalty.unwrap_or(fallback.presence_penalty),
            frequency_penalty: asked
                .frequency_penalty
                .unwrap_or(fallback.frequency_penalty),
            repetition_penalty: asked
                .repetition_penalty
                .unwrap_or(fallback.repetition_penalty),
            logit_bias: asked.logit_bias.iter().copied().collect(),
            logprobs: request.logprobs,
            ..fallback
        },
        limits: SequenceLimits {
            max_tokens,
            deadline: request.deadline,
            ..SequenceLimits::default()
        },
        exactness: Default::default(),
        priority: Default::default(),
        // A client seed keys the sampler by (seed, choice), so a repeated request reproduces
        // (RFC-0032 §3.7).
        choice_index: asked.seed.map(|_| 0),
        prefill_chunk: route.prefill_chunk,
        initial_window: route.initial_window,
        resume: None,
        prefix_scope: PrefixScope {
            principal: request.client.principal,
            shared_until: None,
        },
        session_pk: None,
        trace: None,
        parity: None,
    }
}

impl Generation<'_, '_> {
    /// The sequence.
    #[must_use]
    pub fn seq(&self) -> SequenceId {
        self.seq
    }

    /// The sampler seed: the client's, or the 64 bits ingress drew and recorded.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Tokens emitted and counted so far.
    #[must_use]
    pub fn completion_tokens(&self) -> u32 {
        self.completion_tokens
    }

    /// How the generation ended, once it has.
    #[must_use]
    pub fn outcome(&self) -> Option<&Outcome> {
        self.ended.as_ref()
    }

    /// Stage 10 over every entry that has arrived for the sequence, and stage 11 when it ends.
    ///
    /// Before each event it writes, it checks the client: a client gone, by its [`ClientWatch`]
    /// or a write that failed, gets its sequence cancelled and released, and nothing more is
    /// written. A matched stop string sends `Cancel`, and the tokens after the one that completed
    /// it are discarded and not counted (RFC-0050 §5.2).
    pub fn poll(&mut self, sink: &mut dyn EventSink) -> Progress {
        if let Some(outcome) = &self.ended {
            return Progress::Ended(outcome.clone());
        }
        if let Err(error) = self.ingress.flush() {
            return self.fail(error, sink);
        }
        if self.watch.is_gone() {
            return self.abandon();
        }
        if !self.announced {
            self.announced = true;
            let started = GenEvent::Started {
                seq: self.seq,
                instance: self.route.instance,
                epoch: self.route.epoch,
                prompt_tokens: self.prompt_tokens,
                cached_tokens: 0,
            };
            if self.emit(sink, started).is_err() {
                return self.abandon();
            }
        }
        let entries = match self.ingress.take(self.seq) {
            Ok(entries) => entries,
            Err(error) => return self.fail(error, sink),
        };
        for entry in entries {
            if let Some(progress) = self.entry(entry, sink) {
                return progress;
            }
        }
        Progress::Pending
    }

    /// Delivers one entry; `Some` when the generation ended in it.
    fn entry(&mut self, entry: TokenEntry, sink: &mut dyn EventSink) -> Option<Progress> {
        for &token in &entry.tokens {
            if self.first_token.is_none() {
                self.first_token = Some(self.ingress.clock.now());
            }
            self.completion_tokens += 1;
            self.ingress.record(self.seq, token);
            let text = self.detok.push(token);
            let step = self.stops.push(&text);
            if !step.release.is_empty() && self.text(sink, step.release).is_err() {
                return Some(self.abandon());
            }
            if let Some(stop) = step.matched {
                // Later tokens of this entry, and of any frame still in flight, are discarded.
                self.control(SequenceControl::Cancel {
                    seq: self.seq,
                    reason: CancelReason::StopString,
                });
                return Some(self.finish(StopReason::StopSequence(stop), sink));
            }
        }
        self.ingress.record_counter(self.seq, entry.rng_counter);
        let reason = match entry.finish? {
            FinishReason::Stop | FinishReason::TokenStop => StopReason::EndOfTurn,
            FinishReason::Length => StopReason::Length,
            FinishReason::Cancelled => StopReason::Cancelled,
            FinishReason::Error { domain, code } => {
                return Some(self.fail(ApiError::Sequence { domain, code }, sink));
            }
        };
        // Release what the detokenizer and the stop matcher hold back; a stop string can still
        // complete in the detokenizer's last bytes.
        let tail = self.detok.finish();
        let step = self.stops.push(&tail);
        if !step.release.is_empty() && self.text(sink, step.release).is_err() {
            return Some(self.abandon());
        }
        if let Some(stop) = step.matched {
            return Some(self.finish(StopReason::StopSequence(stop), sink));
        }
        let rest = self.stops.finish();
        if !rest.is_empty() && self.text(sink, rest).is_err() {
            return Some(self.abandon());
        }
        Some(self.finish(reason, sink))
    }

    fn text(&mut self, sink: &mut dyn EventSink, text: String) -> Result<(), ClientGone> {
        self.emit(sink, GenEvent::TextDelta { choice: 0, text })
    }

    /// Writes `event`, unless the client is gone.
    fn emit(&mut self, sink: &mut dyn EventSink, event: GenEvent) -> Result<(), ClientGone> {
        if self.watch.is_gone() {
            return Err(ClientGone);
        }
        sink.emit(event)
    }

    /// Stage 11: the final event, then `Release`.
    fn finish(&mut self, reason: StopReason, sink: &mut dyn EventSink) -> Progress {
        let now = self.ingress.clock.now();
        let finished = GenEvent::Finished {
            choice: 0,
            reason: reason.clone(),
            usage: Usage {
                prompt_tokens: self.prompt_tokens,
                completion_tokens: self.completion_tokens,
                cached_tokens: 0,
            },
            timings: Timings {
                ttft: self
                    .first_token
                    .map(|at| at.saturating_duration_since(self.begun)),
                total: now.saturating_duration_since(self.begun),
            },
        };
        // The sequence is over whether or not this last write reaches the client.
        let _ = self.emit(sink, finished);
        self.end(Outcome::Finished(reason))
    }

    /// The request failed after it started: tell the client if it is there, and release. The
    /// tail finished the sequence itself, or the runner's rings are gone, so there is nothing to
    /// cancel.
    fn fail(&mut self, error: ApiError, sink: &mut dyn EventSink) -> Progress {
        let _ = self.emit(
            sink,
            GenEvent::Failed {
                error: error.clone(),
            },
        );
        self.end(Outcome::Failed(error))
    }

    /// The client is gone: cancel and release, write nothing.
    fn abandon(&mut self) -> Progress {
        self.control(SequenceControl::Cancel {
            seq: self.seq,
            reason: CancelReason::ClientGone,
        });
        self.end(Outcome::ClientGone)
    }

    /// Sends `Release`, drops the transcript, and records the outcome.
    fn end(&mut self, outcome: Outcome) -> Progress {
        self.control(SequenceControl::Release {
            seq: self.seq,
            persist: None,
        });
        self.ingress.unregister(self.seq);
        self.ended = Some(outcome.clone());
        Progress::Ended(outcome)
    }

    /// Sends a `Cancel` or `Release`. Neither is an `Open`, so a full ring queues it; a broken
    /// ring has no runner left to tell.
    fn control(&self, control: SequenceControl) {
        let _ = self.ingress.send(&control);
    }
}

impl Drop for Generation<'_, '_> {
    fn drop(&mut self) {
        if self.ended.is_none() {
            self.abandon();
        }
    }
}
