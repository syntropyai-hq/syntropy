// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-request routing's request, view and decision (RFC-0011 §13). RFC-0011 §13 owns them, and
//! RFC-0032 §8.2 supplies the view's inputs.
//!
//! Each type holds the fields of RFC-0011 §13 that `core` can already name. PLC-E6-T19 adds the
//! rest with the local routing core: the request's model, prefix keys and deadline, the view's
//! routable instances, and the decision's admission. PLC-E6-T05 adds the cluster inputs: load
//! digests, ingress α and prefix hints.

use serde::{Deserialize, Serialize};
use syntropy_core::plan::PlanEpoch;
use syntropy_core::policy::ExactnessPolicy;
use syntropy_core::sequence::Priority;
use syntropy_core::{InstanceId, MonoTime, Nanos, SessionId};

/// One request to route (RFC-0011 §13).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RouteRequest {
    /// Prompt tokens.
    pub prompt_tokens: u32,
    /// The session, if the request belongs to one (RFC-0022).
    pub session: Option<SessionId>,
    /// The request's exactness, resolved per RFC-0001 §11.3.
    pub exactness: ExactnessPolicy,
    /// Greedy sampling accepts more drafts than sampled.
    pub greedy: bool,
    /// The request's priority.
    pub priority: Priority,
}

/// What [`route`](crate::route) reads on the ingress node (RFC-0011 §13).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RoutingView<'a> {
    /// Session pins (RFC-0022).
    pub sessions: &'a [(SessionId, InstanceId)],
    /// The ingress node's clock reading.
    pub now: MonoTime,
}

/// Where a request goes (RFC-0011 §13).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteDecision {
    /// The instance.
    pub instance: InstanceId,
    /// The plan epoch the decision was made against.
    pub epoch: PlanEpoch,
    /// One of the instance's captured ladder sizes.
    pub prefill_chunk: u32,
    /// The initial verify window γ for RFC-0012's controller, at most `max_window`.
    pub initial_window: u16,
    /// The predicted time to first token.
    pub predicted_ttft: Nanos,
}
