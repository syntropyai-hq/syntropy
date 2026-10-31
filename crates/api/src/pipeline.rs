// SPDX-License-Identifier: MIT OR Apache-2.0

//! The shape-neutral ingress pipeline (RFC-0050 §2; PROD-E2-T02).
//!
//! A library layer below the HTTP server: a route parses its body with the shape's front end
//! into a [`GenerateRequest`] (stages 1 and 2), then hands it to [`Ingress::start`] together with
//! the [`ClientWatch`] it sets when the connection closes. It polls the returned [`Generation`]
//! with the shape's encoder as the [`EventSink`] whenever `local_tokens` may have moved. The HTTP
//! server (PROD-E2-T01) and the first shape (PROD-E2-T04) wire it to routes.
//!
//! - [`types`]: [`GenerateRequest`], [`Input`], [`GenEvent`], [`StopReason`] and the types they
//!   hold (RFC-0050 §2.1).
//! - [`Stages`]: the seams for stages 3, 5, 7 and 8 and for the detokenizer and stop matcher of
//!   stage 10, which other tasks fill in.
//! - [`SequenceIdMinter`]: the node's one source of `SequenceId`s for this boot, which every
//!   [`Ingress`] on the node shares, so no two mint the same ID.
//! - [`Ingress`]: one runner's `local_sequences` producer and `local_tokens` consumer, the live
//!   sequences and their [`Transcript`]s (RFC-0001 §6.1 step 5).
//! - [`TailChannels`]: whether ingress asks the tail for the chosen tokens' logprobs alone, or
//!   for one alternative that [`answer_logprobs`] drops (RFC-0050 §3.1).
//! - [`Generation`]: one request, from `SequenceOpen` (stage 9) through the stream (stage 10) to
//!   `Release` (stage 11).
//!
//! **The rings.** At M0 the head and tail are on the ingress node, so `SequenceOpen` and every
//! `SequenceControl` go to the head on `local_sequences`, and each `TokenFrame` arrives on
//! `local_tokens`, in `syntropy_ipc::shm::records`' encoding (RFC-0001 §3.5). A remote head or
//! tail uses a sequence channel instead (DIST-E11-T03, M2).
//!
//! **Ending a sequence.** Every sequence ends with exactly one `Release`, after which ingress
//! drops its transcript and treats later entries for it as strays:
//! - the tail finishes it: the final event, then `Release`;
//! - a stop string matches: `Cancel`, the text before the stop, the final event, then `Release`.
//!   The tokens after the one that completed the match, in its entry or in frames still in
//!   flight, are discarded and not counted (RFC-0050 §5.2);
//! - the client goes away: `Cancel`, then `Release`, and nothing more is written. The pipeline
//!   checks the [`ClientWatch`] before every write, so the `Cancel` goes before the next write
//!   would have; a write that fails counts the same (RFC-0050 §5.3);
//! - the generation is dropped before it ends: as for a client gone.
//!
//! **Controls behind a full ring.** A `Cancel` or `Release` that finds `local_sequences` full
//! waits in the [`Ingress`] and goes out, in order, on the next call that reaches the ring: any
//! generation's [`Generation::poll`], an [`Ingress::start`], or [`Ingress::flush`]. `poll` never
//! waits for room, so a generation can end with its `Release` still waiting. The owner of the
//! `Ingress`, the daemon's ring task, calls [`Ingress::flush`] whenever the runner may have read
//! from the ring, until [`Ingress::waiting_controls`] is 0.
//!
//! **Determinism.** The pipeline reads time only through the `Clock` and randomness only through
//! the `Entropy` the [`Ingress`] is given (RFC-0060 §2.1 rule T1).

mod generation;
mod ingress;
mod logprobs;
mod stages;
pub mod types;

pub use generation::{ClientWatch, Generation, Outcome, Progress};
pub use ingress::{Ingress, SequenceIdMinter, Transcript};
pub use logprobs::{TailChannels, answer_logprobs};
pub use stages::{
    ClientGone, Detokenize, EventSink, ModelInfo, Route, Stages, StopMatch, StopStep,
};
pub use types::{
    ApiError, ApiShape, ChatMessage, ClientMeta, ContentPart, GenEvent, GenerateRequest, Input,
    ModelRef, SamplingRequest, StopReason, StreamMode, Timings, ToolCall, Usage,
};
