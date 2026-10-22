// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-request overrides and their resolution (PROD-E1-T10, RFC-0054 §7, RFC-0001 §11.3).
//!
//! The `p_config_2_` tests are property P-config-2 of RFC-0060 §3.2, registered in
//! `ci/properties.toml`. Over arbitrary (cluster, instance, request) triples they check that
//! exactness resolution yields `Approximate` only when all three opt in, with the knob set the
//! intersection, against the rule written out as a truth table in [`expected`].

use proptest::prelude::*;
use syntropy_config::{
    BenchControls, DebugSection, RequestError, RequestOverrides, RequestWarning, SessionBinding,
    knob_names, priority_for_service_tier, resolve_request,
};
use syntropy_core::{ApproxKnobs, ExactnessPolicy, Fidelity, InstancePolicy, Priority, SessionId};

fn instance(exactness: ExactnessPolicy, default_fidelity: Fidelity) -> InstancePolicy {
    InstancePolicy {
        exactness,
        default_fidelity,
        ..InstancePolicy::default()
    }
}

fn approximate(knobs: ApproxKnobs) -> ExactnessPolicy {
    ExactnessPolicy::Approximate {
        knobs,
        params: Default::default(),
    }
}

fn asks(fidelity: Option<Fidelity>, knobs: Option<ApproxKnobs>) -> RequestOverrides {
    RequestOverrides {
        fidelity,
        knobs,
        ..RequestOverrides::default()
    }
}

const OFF: DebugSection = DebugSection::OFF;

// ---- The rule as a truth table ----------------------------------------------------------

/// What RFC-0054 §7 says a request gets, written out case by case rather than as the code
/// computes it: `Ok(knobs)` for the knobs in effect (empty for exact), or the error.
fn expected(
    req: &RequestOverrides,
    inst: &InstancePolicy,
    allow_approximate: bool,
    debug: DebugSection,
) -> Result<ApproxKnobs, RequestError> {
    let fidelity = req.fidelity.unwrap_or(inst.default_fidelity);
    let asks_approximate = matches!(fidelity, Fidelity::Approximate);
    // Contradictory input, then the debug gate.
    if req.knobs.is_some() && !asks_approximate {
        return Err(RequestError::KnobsNeedApproximate);
    }
    if req.bench.is_some() && !debug.bench_api {
        return Err(RequestError::DebugApiDisabled);
    }
    let (instance_opts_in, allowed) = match inst.exactness {
        ExactnessPolicy::Approximate { knobs, .. } => (true, knobs),
        _ => (false, ApproxKnobs::empty()),
    };
    let plan = allowed & ApproxKnobs::PER_INSTANCE;
    let asked = req.knobs.unwrap_or(allowed);
    let both = asked & allowed;
    match (
        asks_approximate,
        instance_opts_in,
        allow_approximate,
        both.is_empty(),
    ) {
        // All three opt in and the sides share a knob: approximate, unless the request left out
        // a knob the plan applies to every sequence.
        (true, true, true, false) => {
            if plan.difference(both).is_empty() {
                Ok(both)
            } else {
                Err(RequestError::FidelityConflict {
                    knobs: plan.difference(both),
                })
            }
        }
        // Every other row is exact, which a plan with instance-wide knobs cannot serve.
        _ if !plan.is_empty() => Err(RequestError::FidelityConflict { knobs: plan }),
        _ => Ok(ApproxKnobs::empty()),
    }
}

// ---- Strategies ---------------------------------------------------------------------------

fn knobs() -> impl Strategy<Value = ApproxKnobs> {
    any::<u32>().prop_map(ApproxKnobs::from_bits_truncate)
}

fn fidelity() -> impl Strategy<Value = Fidelity> {
    prop_oneof![Just(Fidelity::Exact), Just(Fidelity::Approximate)]
}

fn exactness() -> impl Strategy<Value = ExactnessPolicy> {
    prop_oneof![Just(ExactnessPolicy::Exact), knobs().prop_map(approximate)]
}

fn request() -> impl Strategy<Value = RequestOverrides> {
    (
        proptest::option::of(fidelity()),
        proptest::option::of(knobs()),
        proptest::option::of((any::<bool>(), any::<u32>())),
    )
        .prop_map(|(fidelity, knobs, bench)| RequestOverrides {
            fidelity,
            knobs,
            bench: bench.map(|(ignore_eos, gen_tokens)| BenchControls {
                ignore_eos,
                gen_tokens,
            }),
            ..RequestOverrides::default()
        })
}

fn triple() -> impl Strategy<Value = (RequestOverrides, InstancePolicy, bool, DebugSection)> {
    (
        request(),
        exactness(),
        fidelity(),
        any::<bool>(),
        any::<bool>(),
    )
        .prop_map(|(req, exactness, default_fidelity, allow, bench_api)| {
            let debug = DebugSection {
                bench_api,
                ..DebugSection::OFF
            };
            (req, instance(exactness, default_fidelity), allow, debug)
        })
}

proptest! {
    /// The resolution matches the rule's truth table for every triple.
    #[test]
    fn p_config_2_resolution_follows_the_truth_table((req, inst, allow, debug) in triple()) {
        let got = resolve_request(&req, &inst, allow, debug).map(|policy| policy.knobs());
        prop_assert_eq!(got, expected(&req, &inst, allow, debug));
    }

    /// Approximate only when the request (or the instance default it takes), the instance
    /// policy and the cluster all opt in.
    #[test]
    fn p_config_2_approximate_only_when_all_three_opt_in((req, inst, allow, debug) in triple()) {
        if let Ok(policy) = resolve_request(&req, &inst, allow, debug)
            && policy.fidelity() == Fidelity::Approximate
        {
            prop_assert_eq!(req.fidelity.unwrap_or(inst.default_fidelity), Fidelity::Approximate);
            let instance_opts_in = matches!(inst.exactness, ExactnessPolicy::Approximate { .. });
            prop_assert!(instance_opts_in);
            prop_assert!(allow);
        }
    }

    /// The knobs in effect are a subset of both what the instance allows and what the request
    /// named, and cover every knob the plan applies to every sequence, so the label lists them
    /// all.
    #[test]
    fn p_config_2_knobs_are_a_subset_of_both_sides((req, inst, allow, debug) in triple()) {
        if let Ok(policy) = resolve_request(&req, &inst, allow, debug) {
            let allowed = match inst.exactness {
                ExactnessPolicy::Approximate { knobs, .. } => knobs,
                _ => ApproxKnobs::empty(),
            };
            prop_assert!(allowed.contains(policy.knobs()));
            if let Some(asked) = req.knobs {
                prop_assert!(asked.contains(policy.knobs()));
            }
            if policy.fidelity() == Fidelity::Approximate {
                prop_assert!(policy.knobs().contains(allowed & ApproxKnobs::PER_INSTANCE));
            }
            prop_assert_eq!(policy.sequence_knobs(), policy.knobs() & ApproxKnobs::PER_SEQUENCE);
        }
    }

    /// An exact request on a plan with `KV_LOSSY` always fails with `fidelity_conflict`, 409
    /// in the native API, naming the plan's instance-wide knobs.
    #[test]
    fn p_config_2_exact_on_a_kv_lossy_plan_is_a_conflict(
        extra in knobs(),
        default_fidelity in fidelity(),
        allow in any::<bool>(),
    ) {
        let allowed = extra | ApproxKnobs::KV_LOSSY;
        let inst = instance(approximate(allowed), default_fidelity);
        let err = resolve_request(&asks(Some(Fidelity::Exact), None), &inst, allow, OFF).unwrap_err();
        prop_assert_eq!(err, RequestError::FidelityConflict { knobs: allowed & ApproxKnobs::PER_INSTANCE });
        prop_assert_eq!(err.native_status(), 409);
        prop_assert_eq!(err.api_code(), 7);
    }

    /// An approximate request that names its knobs is never given an instance-wide knob it
    /// left out. When the plan applies one to every sequence, the candidate refuses the request
    /// with `fidelity_conflict`, naming exactly the knobs it left out when the cluster allows
    /// approximate, and every instance-wide knob when it does not (RFC-0054 §7 note of
    /// 2026-10-22).
    #[test]
    fn p_config_2_a_request_never_gets_an_instance_knob_it_did_not_name(
        allowed in knobs(),
        asked in knobs(),
        default_fidelity in fidelity(),
        allow in any::<bool>(),
    ) {
        let inst = instance(approximate(allowed), default_fidelity);
        let req = asks(Some(Fidelity::Approximate), Some(asked));
        let plan = allowed & ApproxKnobs::PER_INSTANCE;
        match resolve_request(&req, &inst, allow, OFF) {
            Ok(policy) => {
                prop_assert!(asked.contains(plan));
                prop_assert!(asked.contains(policy.knobs()));
            }
            // Where the cluster forbids approximate, the request takes the exact path, which a
            // plan with instance-wide knobs cannot serve: the conflict names them all.
            Err(RequestError::FidelityConflict { knobs }) => {
                prop_assert!(!knobs.is_empty());
                prop_assert_eq!(knobs, if allow { plan.difference(asked) } else { plan });
            }
            Err(other) => prop_assert!(false, "unexpected {}", other),
        }
    }

    /// The effective fidelity is `req.fidelity.unwrap_or(inst.default_fidelity)` for the knob
    /// check: knobs with no stated fidelity pass on an instance whose default is approximate,
    /// and fail when the request states exact.
    #[test]
    fn p_config_2_the_knob_check_uses_the_effective_fidelity(asked in knobs(), allow in any::<bool>()) {
        let inst = instance(approximate(ApproxKnobs::PER_SEQUENCE), Fidelity::Approximate);
        let no_fidelity = resolve_request(&asks(None, Some(asked)), &inst, allow, OFF);
        prop_assert_ne!(no_fidelity, Err(RequestError::KnobsNeedApproximate));
        let stated_exact = resolve_request(&asks(Some(Fidelity::Exact), Some(asked)), &inst, allow, OFF);
        prop_assert_eq!(stated_exact, Err(RequestError::KnobsNeedApproximate));
    }
}

// ---- Cases --------------------------------------------------------------------------------

#[test]
fn an_exact_request_is_exact_by_default() {
    let policy = resolve_request(
        &RequestOverrides::default(),
        &InstancePolicy::default(),
        true,
        OFF,
    )
    .unwrap();
    assert_eq!(policy.exactness, ExactnessPolicy::Exact);
    assert_eq!(policy.priority, Priority::Normal);
    assert!(policy.warnings.is_empty());
}

/// An exact request may use an instance that allows only per-sequence knobs; they are simply
/// not used for it.
#[test]
fn an_exact_request_is_served_where_only_per_sequence_knobs_are_allowed() {
    let inst = instance(approximate(ApproxKnobs::PER_SEQUENCE), Fidelity::Exact);
    let policy = resolve_request(&asks(Some(Fidelity::Exact), None), &inst, true, OFF).unwrap();
    assert_eq!(policy.exactness, ExactnessPolicy::Exact);
}

/// Approximate where every side opts in: the knobs are the intersection, and the per-sequence
/// part is what goes in `SequenceOpen`.
#[test]
fn all_three_opting_in_gives_the_intersection() {
    let allowed = ApproxKnobs::EXPERT_SKIP | ApproxKnobs::TOPK_REDUCE | ApproxKnobs::KV_LOSSY;
    let inst = instance(approximate(allowed), Fidelity::Exact);
    let asked = ApproxKnobs::EXPERT_SKIP | ApproxKnobs::KV_LOSSY | ApproxKnobs::SPEC_LENIENT;
    let policy = resolve_request(
        &asks(Some(Fidelity::Approximate), Some(asked)),
        &inst,
        true,
        OFF,
    )
    .unwrap();
    assert_eq!(
        policy.knobs(),
        ApproxKnobs::EXPERT_SKIP | ApproxKnobs::KV_LOSSY
    );
    assert_eq!(policy.sequence_knobs(), ApproxKnobs::EXPERT_SKIP);
    assert!(policy.warnings.is_empty());
    // No knobs named: everything the instance allows.
    let all = resolve_request(&asks(Some(Fidelity::Approximate), None), &inst, true, OFF).unwrap();
    assert_eq!(all.knobs(), allowed);
}

/// An approximate request whose named knobs leave out one the plan applies to every sequence
/// cannot be served there: it would run with a knob it did not accept.
#[test]
fn naming_knobs_that_leave_out_the_plans_is_a_conflict() {
    let inst = instance(
        approximate(ApproxKnobs::KV_LOSSY | ApproxKnobs::EXPERT_SKIP),
        Fidelity::Exact,
    );
    let err = resolve_request(
        &asks(Some(Fidelity::Approximate), Some(ApproxKnobs::EXPERT_SKIP)),
        &inst,
        true,
        OFF,
    )
    .unwrap_err();
    assert_eq!(
        err,
        RequestError::FidelityConflict {
            knobs: ApproxKnobs::KV_LOSSY
        }
    );
    assert!(err.to_string().contains("kv_lossy"), "{err}");
}

/// Asking for approximate where it is not available is served exactly, with the warning,
/// unless the plan itself is approximate.
#[test]
fn approximate_where_unavailable_is_exact_with_a_warning() {
    let exact = InstancePolicy::default();
    let policy =
        resolve_request(&asks(Some(Fidelity::Approximate), None), &exact, true, OFF).unwrap();
    assert_eq!(policy.exactness, ExactnessPolicy::Exact);
    assert_eq!(policy.warnings, [RequestWarning::ApproximateNotAvailable]);
    assert_eq!(policy.warnings[0].name(), "approximate_not_available");
    // The cluster forbids approximate.
    let per_sequence = instance(approximate(ApproxKnobs::EXPERT_SKIP), Fidelity::Exact);
    let policy = resolve_request(
        &asks(Some(Fidelity::Approximate), None),
        &per_sequence,
        false,
        OFF,
    )
    .unwrap();
    assert_eq!(policy.exactness, ExactnessPolicy::Exact);
    assert_eq!(policy.warnings, [RequestWarning::ApproximateNotAvailable]);
    // The intersection is empty.
    let policy = resolve_request(
        &asks(Some(Fidelity::Approximate), Some(ApproxKnobs::TOPK_REDUCE)),
        &per_sequence,
        true,
        OFF,
    )
    .unwrap();
    assert_eq!(policy.exactness, ExactnessPolicy::Exact);
    // An empty intersection on an approximate plan is still a conflict, never an exact label
    // on a lossy plan.
    let lossy = instance(approximate(ApproxKnobs::KV_LOSSY), Fidelity::Exact);
    let err =
        resolve_request(&asks(Some(Fidelity::Approximate), None), &lossy, false, OFF).unwrap_err();
    assert_eq!(
        err,
        RequestError::FidelityConflict {
            knobs: ApproxKnobs::KV_LOSSY
        }
    );
}

/// A per-model default of approximate is the operator's opt-in: a request that states no
/// fidelity is approximate, still subject to the cluster, and an explicit exact is honoured or
/// fails (amendment A6).
#[test]
fn the_instance_default_fidelity_applies_when_the_request_states_none() {
    let inst = instance(approximate(ApproxKnobs::EXPERT_SKIP), Fidelity::Approximate);
    let policy = resolve_request(&asks(None, None), &inst, true, OFF).unwrap();
    assert_eq!(policy.knobs(), ApproxKnobs::EXPERT_SKIP);
    let policy = resolve_request(&asks(None, None), &inst, false, OFF).unwrap();
    assert_eq!(policy.exactness, ExactnessPolicy::Exact);
    let policy = resolve_request(&asks(Some(Fidelity::Exact), None), &inst, true, OFF).unwrap();
    assert_eq!(policy.exactness, ExactnessPolicy::Exact);
    assert!(policy.warnings.is_empty());
}

/// Each candidate is resolved against its own policy, so one request can be exact on one
/// instance and a conflict on another.
#[test]
fn each_candidate_is_resolved_against_its_own_policy() {
    let req = asks(Some(Fidelity::Exact), None);
    let exact = InstancePolicy::default();
    let lossy = instance(approximate(ApproxKnobs::KV_LOSSY), Fidelity::Exact);
    assert!(resolve_request(&req, &exact, true, OFF).is_ok());
    assert!(resolve_request(&req, &lossy, true, OFF).is_err());
}

#[test]
fn knobs_without_approximate_are_refused() {
    let err = resolve_request(
        &asks(None, Some(ApproxKnobs::EXPERT_SKIP)),
        &InstancePolicy::default(),
        true,
        OFF,
    )
    .unwrap_err();
    assert_eq!(err, RequestError::KnobsNeedApproximate);
    assert_eq!((err.api_code(), err.native_status()), (8, 400));
    assert_eq!(err.name(), "knobs_need_approximate");
}

/// Benchmark controls need `debug.bench_api`, are never silently ignored, and label the
/// response `bench` without changing its fidelity.
#[test]
fn benchmark_controls_need_the_bench_api() {
    let req = RequestOverrides {
        bench: Some(BenchControls {
            ignore_eos: true,
            gen_tokens: 128,
        }),
        ..RequestOverrides::default()
    };
    let err = resolve_request(&req, &InstancePolicy::default(), true, OFF).unwrap_err();
    assert_eq!(err, RequestError::DebugApiDisabled);
    assert_eq!((err.api_code(), err.native_status()), (31, 400));
    let on = DebugSection {
        bench_api: true,
        ..DebugSection::OFF
    };
    let policy = resolve_request(&req, &InstancePolicy::default(), true, on).unwrap();
    assert_eq!(policy.exactness, ExactnessPolicy::Exact);
    assert_eq!(policy.warnings, [RequestWarning::Bench]);
    assert_eq!(policy.bench.map(|b| b.gen_tokens), Some(128));
    // Another debug key does not open benchmark controls.
    let parity = DebugSection {
        parity_api: true,
        ..DebugSection::OFF
    };
    assert!(resolve_request(&req, &InstancePolicy::default(), true, parity).is_err());
}

#[test]
fn priorities_map_from_service_tier() {
    assert_eq!(
        priority_for_service_tier(Some("flex")),
        Priority::Background
    );
    for other in [
        Some("auto"),
        Some("default"),
        Some("priority"),
        Some("scale"),
        None,
    ] {
        assert_eq!(
            priority_for_service_tier(other),
            Priority::Normal,
            "{other:?}"
        );
    }
    let req = RequestOverrides {
        priority: Some(Priority::Interactive),
        ..RequestOverrides::default()
    };
    let policy = resolve_request(&req, &InstancePolicy::default(), true, OFF).unwrap();
    assert_eq!(policy.priority, Priority::Interactive);
}

/// The rest of the request's policy is carried through unchanged.
#[test]
fn the_request_flags_are_carried_through() {
    let session = SessionBinding::Native(SessionId([7; 16]));
    let req = RequestOverrides {
        speculative_off: true,
        deterministic: Some(true),
        session: Some(session),
        persist: Some(false),
        trace: true,
        ..RequestOverrides::default()
    };
    let policy = resolve_request(&req, &InstancePolicy::default(), true, OFF).unwrap();
    assert!(policy.speculative_off && policy.deterministic && policy.trace);
    assert_eq!(policy.session, Some(session));
    assert_eq!(policy.persist, Some(false));
}

#[test]
fn knob_names_are_lowercase() {
    assert_eq!(
        knob_names(ApproxKnobs::KV_LOSSY | ApproxKnobs::WIRE_COMPRESS),
        "kv_lossy, wire_compress"
    );
    assert_eq!(knob_names(ApproxKnobs::empty()), "");
}
