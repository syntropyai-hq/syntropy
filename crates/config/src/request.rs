// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-request overrides and their resolution at ingress (RFC-0054 §7, RFC-0001 §11.3).
//! Owner: RFC-0054.
//!
//! Each API front end extracts a request's policy into [`RequestOverrides`] (RFC-0050 §3).
//! [`resolve_request`] decides, against one candidate instance's own policy, what the request
//! gets. This is where D3 is enforced for requests:
//! - A request is approximate only when it asks for approximate, the instance policy is
//!   `Approximate`, and the cluster's `allow_approximate` is true. Its knobs are the
//!   intersection of what it asks for and what the instance allows.
//! - An exact request is never served by a plan that applies instance-wide knobs: it fails with
//!   `fidelity_conflict`, so routing tries another instance.
//! - A request that asks for approximate where it is not available is served exactly, with the
//!   `approximate_not_available` warning.
//! - A knob bit this version cannot name, such as one a later version adds, is never served:
//!   it counts as instance-wide, and no request may run with it, since no label could list it.
//!
//! `config` sits below the API crate, so the errors carry their RFC-0050 §12.3 codes for the API
//! to report, rather than the API's own error type.

use std::fmt;

use syntropy_core::{
    ApproxKnobs, ApproxParams, ErrorClass, ExactnessPolicy, Fidelity, InstancePolicy, Priority,
    SessionId,
};

use crate::debug::{DebugApi, DebugSection};

/// A request's policy, the same whatever API shape carried it (RFC-0054 §7). Owner: RFC-0054
/// (fields and rules); RFC-0050 says how each API shape carries them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RequestOverrides {
    /// `Exact` or `Approximate`. `None` takes the instance's `default_fidelity`, which is
    /// `Exact` unless the operator set it (D3, amendment A6).
    pub fidelity: Option<Fidelity>,
    /// The knobs the request asks for. Only with an approximate fidelity; `None` with one asks
    /// for every knob the instance allows.
    pub knobs: Option<ApproxKnobs>,
    /// `Interactive`, `Normal` or `Background`; `None` is `Normal`.
    pub priority: Option<Priority>,
    /// Switches speculative decoding off. A request cannot force it on.
    pub speculative_off: bool,
    /// Asks for deterministic generation, which needs an instance whose policy is deterministic
    /// (ADR-0019 decision 8).
    pub deterministic: Option<bool>,
    /// The session the request continues (RFC-0050 §9).
    pub session: Option<SessionBinding>,
    /// Overrides `kv.autosave` for this session (RFC-0022).
    pub persist: Option<bool>,
    /// Records every step span (RFC-0053 §3.3).
    pub trace: bool,
    /// Benchmark controls, only with `debug.bench_api` (RFC-0054 §7.1).
    pub bench: Option<BenchControls>,
}

/// The benchmark controls on `POST /v1/syntropy/generate` (RFC-0054 §7.1, RFC-0061 §3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchControls {
    /// Generation does not stop at end-of-sequence, stop sets or stop strings.
    pub ignore_eos: bool,
    /// Exactly this many tokens are generated.
    pub gen_tokens: u32,
}

/// The session a request continues (RFC-0050 §9). This holds §9.3's native session; §9.2's
/// `previous_response_id` comes with the task that stores responses.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionBinding {
    /// A native session, `ses_…` (RFC-0050 §9.3).
    Native(SessionId),
}

/// What a request gets on one candidate instance: its exactness, with the knobs in effect, and
/// the rest of its policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EffectiveRequestPolicy {
    /// `Exact`, or `Approximate` with the knobs in effect for this request and the instance's
    /// parameters for them.
    pub exactness: ExactnessPolicy,
    /// The scheduling priority. Whether an `Interactive` request keeps it is decided at
    /// admission, since an instance holds at most one at a time (RFC-0054 §7).
    pub priority: Priority,
    /// Speculative decoding is off for this request.
    pub speculative_off: bool,
    /// The request asks for deterministic generation. Routing serves it only from an instance
    /// whose policy is deterministic (PROD-E2-T08).
    pub deterministic: bool,
    /// The session the request continues.
    pub session: Option<SessionBinding>,
    /// Overrides `kv.autosave` for this session.
    pub persist: Option<bool>,
    /// Records every step span.
    pub trace: bool,
    /// Benchmark controls, which `debug.bench_api` allowed.
    pub bench: Option<BenchControls>,
    /// What the response's `syntropy.warnings` must say about how the request was resolved.
    pub warnings: Vec<RequestWarning>,
}

impl EffectiveRequestPolicy {
    /// `Exact` or `Approximate`.
    pub fn fidelity(&self) -> Fidelity {
        if self.knobs().is_empty() {
            Fidelity::Exact
        } else {
            Fidelity::Approximate
        }
    }

    /// The knobs in effect for this request: what the response's fidelity label lists as
    /// enabled (RFC-0050 §10). Empty when exact.
    pub fn knobs(&self) -> ApproxKnobs {
        match self.exactness {
            ExactnessPolicy::Approximate { knobs, .. } => knobs,
            _ => ApproxKnobs::empty(),
        }
    }

    /// The per-sequence part of the knobs, which goes in `SequenceOpen` (RFC-0001 §5.8). The
    /// instance-wide knobs come from the plan.
    pub fn sequence_knobs(&self) -> ApproxKnobs {
        self.knobs() & ApproxKnobs::PER_SEQUENCE
    }
}

/// A note on the response about how its request was resolved (`syntropy.warnings`,
/// RFC-0050 §10, §12).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestWarning {
    /// The request asked for approximate where it is not available, so it is served exactly.
    ApproximateNotAvailable,
    /// Generation ignores end-of-sequence (`BenchControls::ignore_eos`, RFC-0054 §7.1). The
    /// fidelity label is unchanged.
    Bench,
}

impl RequestWarning {
    /// The warning's name in `syntropy.warnings`.
    pub const fn name(self) -> &'static str {
        match self {
            RequestWarning::ApproximateNotAvailable => "approximate_not_available",
            RequestWarning::Bench => "bench",
        }
    }
}

impl fmt::Display for RequestWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Why a request cannot be served by a candidate instance. Each is `InvalidRequest`, with its
/// code in RFC-0050 §12.3's `Api` domain, which the API reports.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestError {
    /// The candidate's plan applies instance-wide knobs this request cannot be served with: the
    /// request is exact, or it named knobs and left out one the plan applies to every sequence
    /// (RFC-0054 §7 note of 2026-10-22), or a knob in effect is one this version cannot name,
    /// so no label could list it. It names those knobs. Routing tries another instance before
    /// reporting it.
    FidelityConflict {
        /// The instance-wide knobs the request cannot be served with.
        knobs: ApproxKnobs,
    },
    /// The request named knobs without an approximate fidelity.
    KnobsNeedApproximate,
    /// The request carries `BenchControls`, and `debug.bench_api` is off.
    DebugApiDisabled,
}

impl RequestError {
    /// The error class: every one is a client error (RFC-0001 §5.12).
    pub const fn class(self) -> ErrorClass {
        ErrorClass::InvalidRequest
    }

    /// The code in RFC-0050 §12.3's `Api` domain.
    pub const fn api_code(self) -> u16 {
        match self {
            RequestError::FidelityConflict { .. } => 7,
            RequestError::KnobsNeedApproximate => 8,
            RequestError::DebugApiDisabled => 31,
        }
    }

    /// The code's name.
    pub const fn name(self) -> &'static str {
        match self {
            RequestError::FidelityConflict { .. } => "fidelity_conflict",
            RequestError::KnobsNeedApproximate => "knobs_need_approximate",
            RequestError::DebugApiDisabled => "debug_api_disabled",
        }
    }

    /// The HTTP status in the native API (RFC-0050 §12): 409 for `fidelity_conflict`, 400 for
    /// the others. The vendor shapes use 400 for all three.
    pub const fn native_status(self) -> u16 {
        match self {
            RequestError::FidelityConflict { .. } => 409,
            RequestError::KnobsNeedApproximate | RequestError::DebugApiDisabled => 400,
        }
    }
}

impl fmt::Display for RequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RequestError::FidelityConflict { knobs } => {
                let unnamed = knobs.difference(ApproxKnobs::all());
                if unnamed.is_empty() {
                    write!(
                        f,
                        "fidelity_conflict: the instance's plan applies {} to every sequence, \
                         which this request did not accept",
                        knob_names(*knobs)
                    )
                } else {
                    write!(
                        f,
                        "fidelity_conflict: the instance's plan applies {} to every sequence, \
                         and this version cannot name {}, so it cannot label a response that \
                         uses it",
                        knob_names(*knobs),
                        knob_names(unnamed)
                    )
                }
            }
            RequestError::KnobsNeedApproximate => f.write_str(
                "knobs_need_approximate: approximate knobs need `fidelity: \"approximate\"`",
            ),
            RequestError::DebugApiDisabled => f.write_str(
                "debug_api_disabled: benchmark controls need `debug.bench_api` on this node",
            ),
        }
    }
}

impl std::error::Error for RequestError {}

/// The knobs' names, lowercase and comma-separated, as the error and label name them. A bit
/// this version cannot name is written as its value in hex, such as `0x100`, one per bit.
pub fn knob_names(knobs: ApproxKnobs) -> String {
    let mut names: Vec<String> = knobs
        .iter_names()
        .map(|(name, _)| name.to_ascii_lowercase())
        .collect();
    let mut unnamed = knobs.difference(ApproxKnobs::all()).bits();
    while unnamed != 0 {
        let bit = unnamed.isolate_lowest_one();
        names.push(format!("{bit:#x}"));
        unnamed &= !bit;
    }
    names.join(", ")
}

/// The priority of a request that carries OpenAI's `service_tier`: `flex` is `Background`, and
/// any other value, or none, is `Normal` (RFC-0054 §7, RFC-0050 §3). Only the native API can
/// ask for `Interactive`.
pub fn priority_for_service_tier(service_tier: Option<&str>) -> Priority {
    match service_tier {
        Some("flex") => Priority::Background,
        _ => Priority::Normal,
    }
}

/// Resolves `req` against one candidate instance's own policy `inst` (RFC-0054 §7,
/// RFC-0001 §11.3). `allow_approximate` is the cluster's key, and `debug` this node's debug
/// gates (RFC-0054 §7.1).
///
/// Routing calls this once per candidate, against that candidate's own policy (RFC-0050 §2.2
/// step 7, §4.2; PROD-E2-T08). When none passes, it calls this once more with the policy the
/// tables would build for a new instance, and fails at once with its error rather than wait on
/// an `EnsureInstance` that could not serve the request. Either way the rule is the same.
///
/// It does not check `req.deterministic` against `inst.deterministic`: routing picks a
/// deterministic instance for such a request (RFC-0054 §7, PROD-E2-T08).
///
/// The effective fidelity is `req.fidelity.unwrap_or(inst.default_fidelity)`, for both the knob
/// check and the approximate decision. Then:
/// 1. Knobs without an approximate fidelity fail with `knobs_need_approximate`, and benchmark
///    controls without `debug.bench_api` with `debug_api_disabled`.
/// 2. The request is approximate only when its fidelity is approximate, `inst.exactness` is
///    `Approximate` and `allow_approximate` holds. Its knobs are `req.knobs` (all the instance
///    allows when `None`) intersected with what the instance allows.
/// 3. Every knob the instance allows outside `PER_SEQUENCE` is in its plan, so it applies to
///    every sequence on it: the `PER_INSTANCE` knobs, and any bit this version cannot name,
///    which fails safe as instance-wide. An approximate request whose knobs leave one out
///    fails with `fidelity_conflict` naming it, so it never runs with a knob it did not accept
///    and its label never misses one (RFC-0054 §7 note of 2026-10-22). So does one whose knobs
///    hold a bit this version cannot name, since no label could list it.
/// 4. Anything else, an empty intersection included, is exact. A request that asked for
///    approximate gets `approximate_not_available`. An exact request on a plan with
///    instance-wide knobs fails with `fidelity_conflict` naming them.
pub fn resolve_request(
    req: &RequestOverrides,
    inst: &InstancePolicy,
    allow_approximate: bool,
    debug: DebugSection,
) -> Result<EffectiveRequestPolicy, RequestError> {
    let fidelity = req.fidelity.unwrap_or(inst.default_fidelity);
    let wants_approximate = fidelity == Fidelity::Approximate;
    if req.knobs.is_some() && !wants_approximate {
        return Err(RequestError::KnobsNeedApproximate);
    }
    if req.bench.is_some() && !debug.is_on(DebugApi::Bench) {
        return Err(RequestError::DebugApiDisabled);
    }
    let (instance_approximate, allowed, params) = match inst.exactness {
        ExactnessPolicy::Approximate { knobs, params } => (true, knobs, params),
        _ => (false, ApproxKnobs::empty(), ApproxParams::default()),
    };
    // The knobs the instance's plan applies to every sequence on it: every allowed knob that
    // is not per-sequence, so a bit this version cannot name counts as instance-wide. Both
    // `difference` and `&` keep unnamed bits; `!` would drop them.
    let plan_knobs = allowed.difference(ApproxKnobs::PER_SEQUENCE);

    let mut exactness = ExactnessPolicy::Exact;
    if wants_approximate && instance_approximate && allow_approximate {
        let knobs = req.knobs.unwrap_or(allowed) & allowed;
        if !knobs.is_empty() {
            // The plan's knobs the request left out, and the knobs in effect that no label could
            // list.
            let refused = plan_knobs
                .difference(knobs)
                .union(knobs.difference(ApproxKnobs::all()));
            if !refused.is_empty() {
                return Err(RequestError::FidelityConflict { knobs: refused });
            }
            exactness = ExactnessPolicy::Approximate { knobs, params };
        }
    }

    let mut warnings = Vec::new();
    if exactness == ExactnessPolicy::Exact {
        if !plan_knobs.is_empty() {
            return Err(RequestError::FidelityConflict { knobs: plan_knobs });
        }
        if wants_approximate {
            warnings.push(RequestWarning::ApproximateNotAvailable);
        }
    }
    if req.bench.is_some_and(|bench| bench.ignore_eos) {
        warnings.push(RequestWarning::Bench);
    }
    Ok(EffectiveRequestPolicy {
        exactness,
        priority: req.priority.unwrap_or_default(),
        speculative_off: req.speculative_off,
        deterministic: req.deterministic.unwrap_or(false),
        session: req.session,
        persist: req.persist,
        trace: req.trace,
        bench: req.bench,
        warnings,
    })
}
