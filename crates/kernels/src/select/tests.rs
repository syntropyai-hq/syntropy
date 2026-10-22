// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the selection engine against RFC-0040 §6.2 and §6.4 and ADR-0401 items 1 and 4
//! (COMP-E2-T01): recorded-table replay, the 3 % and 5 % margins, ties to native, the filters,
//! the failure paths and determinism.

use proptest::prelude::*;
use serde::Deserialize;
use smallvec::smallvec;
use syntropy_core::resource::{AppleGpuFamily, CpuIsa};
use syntropy_core::{Blake3, Bytes, DType, QuantScheme};

use super::*;
use crate::numerics::rule;
use crate::op::{OpKind, ShapeClass};

// ---- Builders ------------------------------------------------------------------------------

fn key(op: OpKind) -> OpKey {
    OpKey {
        op,
        schemes: smallvec![QuantScheme::Plain(DType::F32); 3],
        layout: LayoutVersion::SOURCE,
        variant: 0,
    }
}

fn shape() -> ShapeSig {
    ShapeSig::new(ShapeClass::Decode, &[4096, 4096])
}

fn caps(op: &OpKind) -> OpCaps {
    OpCaps {
        workspace: Bytes(0),
        est: None,
        layout: LayoutVersion::SOURCE,
        capture_safe: true,
        deterministic: true,
        numerics: rule(op).minimal(),
        kernel_hash: Blake3([0; 32]),
        source_key: Blake3([0; 32]),
    }
}

fn cand(provider: &'static str, backend: Backend, domain: MemoryDomain, caps: OpCaps) -> Candidate {
    Candidate {
        provider: ProviderId(provider),
        backend,
        domain,
        variant: 0,
        caps: Some(caps),
    }
}

fn measured(ns: u64) -> Option<Measurement> {
    Some(Measurement {
        measured: Nanos(ns),
        breaks: 0,
        fences: 0,
    })
}

fn slot(op: OpKind, offers: Vec<Offer>) -> OpSlot {
    OpSlot {
        key: key(op),
        shape: shape(),
        predicted: Nanos(1),
        layout: Some(LayoutVersion::SOURCE),
        cached: None,
        offers,
    }
}

/// A native CUDA candidate and a ggml one on the same device, with the given times.
fn native_vs_ggml(native_ns: u64, ggml_ns: u64) -> OpSlot {
    let c = caps(&OpKind::Gemv);
    slot(
        OpKind::Gemv,
        vec![
            Offer {
                candidate: cand("cuda", Backend::Cuda, MemoryDomain::Cuda, c),
                measurement: measured(native_ns),
            },
            Offer {
                candidate: cand(
                    "ggml-cuda",
                    Backend::Ggml(GgmlBackend::Cuda),
                    MemoryDomain::Cuda,
                    c,
                ),
                measurement: measured(ggml_ns),
            },
        ],
    )
}

fn decide(op: &OpSlot) -> Result<OpChoice, Box<Unsupported>> {
    decide_op(op, 0, MemoryDomain::Cuda, false, &CostTerms::default())
}

// ---- Recorded-table replay -----------------------------------------------------------------

#[derive(Deserialize)]
struct Table {
    name: String,
    vendor: String,
    deterministic: bool,
    terms: TableTerms,
    programs: Vec<TableProgram>,
    opened: Vec<String>,
    expect_domain: String,
}

#[derive(Deserialize)]
struct TableTerms {
    t_submit: u64,
    t_fence: u64,
}

#[derive(Deserialize)]
struct TableProgram {
    domain: String,
    ops: Vec<TableOp>,
    expect_step: Option<u64>,
}

#[derive(Deserialize)]
struct TableOp {
    op: String,
    predicted: u64,
    offers: Vec<TableOffer>,
    expect: String,
}

fn yes() -> bool {
    true
}

#[derive(Deserialize)]
struct TableOffer {
    provider: String,
    backend: String,
    domain: String,
    ns: Option<u64>,
    est: Option<u64>,
    #[serde(default)]
    breaks: u32,
    #[serde(default)]
    fences: u32,
    #[serde(default = "yes")]
    capture_safe: bool,
    #[serde(default = "yes")]
    deterministic: bool,
    numerics: Option<String>,
}

fn op_kind(name: &str) -> OpKind {
    match name {
        "Gemv" => OpKind::Gemv,
        "Gemm" => OpKind::Gemm,
        "Logits" => OpKind::Logits,
        "RmsNorm" => OpKind::RmsNorm,
        "Add" => OpKind::Add,
        "Argmax" => OpKind::Argmax,
        other => panic!("table op {other} is not mapped"),
    }
}

fn backend(name: &str) -> Backend {
    match name {
        "Cuda" => Backend::Cuda,
        "Hip" => Backend::Hip,
        "Metal" => Backend::Metal,
        "Vulkan" => Backend::Vulkan,
        "Cpu" => Backend::Cpu,
        "GgmlCuda" => Backend::Ggml(GgmlBackend::Cuda),
        "GgmlHip" => Backend::Ggml(GgmlBackend::Hip),
        "GgmlVulkan" => Backend::Ggml(GgmlBackend::Vulkan),
        "GgmlCpu" => Backend::Ggml(GgmlBackend::Cpu),
        "GgmlSycl" => Backend::Ggml(GgmlBackend::Sycl),
        other => panic!("table backend {other} is not mapped"),
    }
}

fn domain(name: &str) -> MemoryDomain {
    match name {
        "Cuda" => MemoryDomain::Cuda,
        "Hip" => MemoryDomain::Hip,
        "Metal" => MemoryDomain::Metal,
        "Vulkan" => MemoryDomain::Vulkan,
        "Host" => MemoryDomain::Host,
        "GgmlSycl" => MemoryDomain::Ggml(GgmlBackend::Sycl),
        other => panic!("table domain {other} is not mapped"),
    }
}

fn vendor(name: &str) -> Vendor {
    match name {
        "Nvidia" => Vendor::Nvidia,
        "Amd" => Vendor::Amd,
        "Apple" => Vendor::Apple,
        "IntelGpu" => Vendor::IntelGpu,
        "Cpu" => Vendor::Cpu,
        other => panic!("table vendor {other} is not mapped"),
    }
}

fn leak(s: &str) -> &'static str {
    Box::leak(s.to_owned().into_boxed_str())
}

fn table_slot(op: &TableOp) -> OpSlot {
    let kind = op_kind(&op.op);
    let offers = op
        .offers
        .iter()
        .map(|o| {
            let mut c = caps(&kind);
            c.est = o.est.map(Nanos);
            c.capture_safe = o.capture_safe;
            c.deterministic = o.deterministic;
            match o.numerics.as_deref() {
                None => {}
                Some("bad_output") => c.numerics.out = DType::F16,
                Some(other) => panic!("table numerics {other} is not mapped"),
            }
            Offer {
                candidate: cand(leak(&o.provider), backend(&o.backend), domain(&o.domain), c),
                measurement: o.ns.map(|ns| Measurement {
                    measured: Nanos(ns),
                    breaks: o.breaks,
                    fences: o.fences,
                }),
            }
        })
        .collect();
    OpSlot {
        predicted: Nanos(op.predicted),
        ..slot(kind, offers)
    }
}

/// Replays `json`: every program's per-op choices and predicted step time, then the domain.
fn replay(json: &str) {
    let table: Table = serde_json::from_str(json).expect("a valid table");
    let terms = CostTerms {
        t_submit: Nanos(table.terms.t_submit),
        t_fence: Nanos(table.terms.t_fence),
    };
    let mut steps = Vec::new();
    for program in &table.programs {
        let d = domain(&program.domain);
        let slots: Vec<_> = program.ops.iter().map(table_slot).collect();
        let got = select_program(&slots, d, table.deterministic, &terms)
            .unwrap_or_else(|e| panic!("{}: {e}", table.name));
        for (op, choice) in program.ops.iter().zip(&got.choices) {
            assert_eq!(
                format!("{}#{}", choice.provider.0, choice.variant),
                op.expect,
                "{} {:?} {}: {choice:?}",
                table.name,
                d,
                op.op
            );
        }
        assert_eq!(
            got.predicted_step,
            program.expect_step.map(Nanos),
            "{} {d:?}",
            table.name
        );
        steps.push((d, got.predicted_step));
    }
    let opened: Vec<_> = table
        .opened
        .iter()
        .map(|name| {
            let d = domain(name);
            DomainOption {
                domain: d,
                predicted: steps.iter().find(|(s, _)| *s == d).and_then(|&(_, t)| t),
            }
        })
        .collect();
    let chosen = choose_domain(vendor(&table.vendor), &opened).expect("a domain");
    assert_eq!(
        chosen.domain,
        domain(&table.expect_domain),
        "{}",
        table.name
    );
}

#[test]
fn recorded_tables_reproduce_the_expected_choices() {
    for json in [
        include_str!("tables/nvidia_decode.json"),
        include_str!("tables/amd_hip_stays.json"),
        include_str!("tables/amd_vulkan_wins.json"),
        include_str!("tables/cpu_budget.json"),
    ] {
        replay(json);
    }
}

#[test]
fn a_wrong_expectation_in_a_table_fails_the_replay() {
    let json = include_str!("tables/nvidia_decode.json").replacen(
        "\"expect\": \"cuda#0\"",
        "\"expect\": \"ggml-cuda#0\"",
        1,
    );
    assert!(std::panic::catch_unwind(|| replay(&json)).is_err());
}

// ---- The 3 % margin and ties to native -----------------------------------------------------

#[test]
fn a_challenger_2_9_percent_faster_loses() {
    let got = decide(&native_vs_ggml(100_000, 97_100)).unwrap();
    assert_eq!(got.provider, ProviderId("cuda"));
    assert_eq!(got.basis, Basis::Incumbent(Incumbent::Native));
    assert_eq!(got.cost, Some(Nanos(100_000)));
}

#[test]
fn a_challenger_3_1_percent_faster_wins() {
    let got = decide(&native_vs_ggml(100_000, 96_900)).unwrap();
    assert_eq!(got.provider, ProviderId("ggml-cuda"));
    assert_eq!(
        got.basis,
        Basis::Challenger {
            replaced: (ProviderId("cuda"), 0)
        }
    );
    assert_eq!(got.cost, Some(Nanos(96_900)));
}

#[test]
fn a_challenger_exactly_3_percent_faster_loses() {
    // RFC-0040 §6.4: the challenger must cost strictly less than 0.97 of the incumbent.
    let got = decide(&native_vs_ggml(100_000, 97_000)).unwrap();
    assert_eq!(got.provider, ProviderId("cuda"));
}

#[test]
fn equal_costs_go_to_native() {
    let got = decide(&native_vs_ggml(1000, 1000)).unwrap();
    assert_eq!(got.provider, ProviderId("cuda"));
    assert!(got.native);
}

#[test]
fn a_slower_native_is_still_the_incumbent() {
    // The incumbent is the native candidate even when ggml is the cheapest.
    let got = decide(&native_vs_ggml(1000, 990)).unwrap();
    assert_eq!(got.basis, Basis::Incumbent(Incumbent::Native));
}

fn two_ggml(a_ns: u64, b_ns: u64, cached: Option<&'static str>) -> OpSlot {
    let c = caps(&OpKind::Gemv);
    let g = |name, ns| Offer {
        candidate: cand(
            name,
            Backend::Ggml(GgmlBackend::Vulkan),
            MemoryDomain::Vulkan,
            c,
        ),
        measurement: measured(ns),
    };
    OpSlot {
        cached: cached.map(|p| (ProviderId(p), 0)),
        ..slot(OpKind::Gemv, vec![g("ggml-a", a_ns), g("ggml-b", b_ns)])
    }
}

#[test]
fn with_no_native_the_cached_choice_is_the_incumbent() {
    let terms = CostTerms::default();
    let d = MemoryDomain::Vulkan;
    let kept = decide_op(&two_ggml(980, 1000, Some("ggml-b")), 0, d, false, &terms).unwrap();
    assert_eq!(kept.provider, ProviderId("ggml-b"));
    assert_eq!(kept.basis, Basis::Incumbent(Incumbent::Cached));
    let replaced = decide_op(&two_ggml(960, 1000, Some("ggml-b")), 0, d, false, &terms).unwrap();
    assert_eq!(replaced.provider, ProviderId("ggml-a"));
    // A cached choice that is no longer offered is ignored.
    let cheapest = decide_op(&two_ggml(990, 1000, Some("gone")), 0, d, false, &terms).unwrap();
    assert_eq!(cheapest.provider, ProviderId("ggml-a"));
    assert_eq!(cheapest.basis, Basis::Incumbent(Incumbent::Cheapest));
}

#[test]
fn a_cached_choice_does_not_override_native() {
    let mut op = native_vs_ggml(1000, 980);
    op.cached = Some((ProviderId("ggml-cuda"), 0));
    let got = decide(&op).unwrap();
    assert_eq!(got.basis, Basis::Incumbent(Incumbent::Native));
}

// ---- Segment-break and fence costs ---------------------------------------------------------

#[test]
fn segment_breaks_and_fences_are_charged() {
    let terms = CostTerms {
        t_submit: Nanos(2000),
        t_fence: Nanos(500),
    };
    let m = Measurement {
        measured: Nanos(1000),
        breaks: 2,
        fences: 3,
    };
    assert_eq!(m.cost(false, &terms), Nanos(1000 + 2 * 2000 + 3 * 500));
    // A capture-safe kernel forces no break.
    assert_eq!(m.cost(true, &terms), Nanos(1000 + 3 * 500));
    let huge = Measurement {
        measured: Nanos(u64::MAX - 1),
        breaks: u32::MAX,
        fences: u32::MAX,
    };
    assert_eq!(huge.cost(false, &terms), Nanos(u64::MAX));
}

#[test]
fn a_forced_break_can_cost_native_its_place() {
    let terms = CostTerms {
        t_submit: Nanos(5000),
        t_fence: Nanos(0),
    };
    let mut op = native_vs_ggml(10_000, 12_000);
    op.offers[0].candidate.caps.as_mut().unwrap().capture_safe = false;
    op.offers[0].measurement.as_mut().unwrap().breaks = 1;
    let got = decide_op(&op, 0, MemoryDomain::Cuda, false, &terms).unwrap();
    assert_eq!(got.provider, ProviderId("ggml-cuda"));
    assert_eq!(got.cost, Some(Nanos(12_000)));
}

// ---- The candidate filter ------------------------------------------------------------------

fn req(layout: Option<LayoutVersion>, deterministic: bool) -> Requirements {
    Requirements {
        domain: MemoryDomain::Cuda,
        layout,
        deterministic,
    }
}

#[test]
fn the_filter_applies_each_condition() {
    let k = key(OpKind::RmsNorm);
    let good = cand("cuda", Backend::Cuda, MemoryDomain::Cuda, caps(&k.op));
    let r = req(Some(LayoutVersion::SOURCE), true);
    assert_eq!(admit(&k, &r, &good), Ok(caps(&k.op)));

    let other = Candidate {
        domain: MemoryDomain::Vulkan,
        ..good
    };
    assert_eq!(
        admit(&k, &r, &other),
        Err(Rejection::OtherDomain(MemoryDomain::Vulkan))
    );

    let none = Candidate { caps: None, ..good };
    assert_eq!(admit(&k, &r, &none), Err(Rejection::NoQuery));

    let mut bad = caps(&k.op);
    bad.numerics.out = DType::F16;
    let bad = Candidate {
        caps: Some(bad),
        ..good
    };
    assert_eq!(
        admit(&k, &r, &bad),
        Err(Rejection::Numerics(NumericsViolation::Output(DType::F16)))
    );

    let mut repacked = caps(&k.op);
    repacked.layout = LayoutVersion(7);
    let repacked = Candidate {
        caps: Some(repacked),
        ..good
    };
    assert_eq!(
        admit(&k, &r, &repacked),
        Err(Rejection::Layout {
            wanted: LayoutVersion::SOURCE,
            offered: LayoutVersion(7)
        })
    );
    // The layout pass admits every layout.
    assert!(admit(&k, &req(None, true), &repacked).is_ok());

    let mut nondet = caps(&k.op);
    nondet.deterministic = false;
    let nondet = Candidate {
        caps: Some(nondet),
        ..good
    };
    assert_eq!(admit(&k, &r, &nondet), Err(Rejection::NotDeterministic));
    assert!(admit(&k, &req(Some(LayoutVersion::SOURCE), false), &nondet).is_ok());
}

#[test]
fn the_numerics_filter_is_the_static_check() {
    // Every kind's minimal declaration passes; a softmax op without F32 softmax does not.
    let k = key(OpKind::TopKTopP);
    let mut c = caps(&k.op);
    assert!(
        admit(
            &k,
            &req(None, false),
            &cand("cuda", Backend::Cuda, MemoryDomain::Cuda, c)
        )
        .is_ok()
    );
    c.numerics.softmax_f32 = false;
    assert_eq!(
        admit(
            &k,
            &req(None, false),
            &cand("cuda", Backend::Cuda, MemoryDomain::Cuda, c)
        ),
        Err(Rejection::Numerics(NumericsViolation::SoftmaxNotF32))
    );
}

// ---- Failure paths -------------------------------------------------------------------------

#[test]
fn an_op_with_no_candidate_is_unsupported() {
    let k = key(OpKind::Gemv);
    let mut c = caps(&k.op);
    c.deterministic = false;
    let op = slot(
        OpKind::Gemv,
        vec![
            Offer {
                candidate: cand("zeta", Backend::Cuda, MemoryDomain::Cuda, c),
                measurement: measured(1),
            },
            Offer {
                candidate: Candidate {
                    caps: None,
                    ..cand("alpha", Backend::Cuda, MemoryDomain::Cuda, c)
                },
                measurement: None,
            },
        ],
    );
    let err = decide_op(&op, 4, MemoryDomain::Cuda, true, &CostTerms::default()).unwrap_err();
    assert_eq!(err.slot, 4);
    assert_eq!(
        err.rejections,
        vec![
            (ProviderId("alpha"), 0, Rejection::NoQuery),
            (ProviderId("zeta"), 0, Rejection::NotDeterministic),
        ]
    );
    let msg = err.to_string();
    assert!(msg.contains("alpha#0: query returned None"), "{msg}");
    let e = Error::from(err);
    assert_eq!(e.code, ErrorCode::OP_UNSUPPORTED);
    assert_eq!(e.class, ErrorClass::Unsupported);

    let empty = slot(OpKind::Gemv, Vec::new());
    assert!(decide(&empty).unwrap_err().rejections.is_empty());
}

#[test]
fn the_program_stops_at_the_costliest_unsupported_op() {
    let unsupported = |predicted| OpSlot {
        predicted: Nanos(predicted),
        ..slot(OpKind::Gemm, Vec::new())
    };
    let ok = OpSlot {
        predicted: Nanos(10),
        ..native_vs_ggml(1, 2)
    };
    let program = vec![ok.clone(), unsupported(5), unsupported(50)];
    let err =
        select_program(&program, MemoryDomain::Cuda, false, &CostTerms::default()).unwrap_err();
    assert_eq!(err.slot, 2);

    let fine = select_program(
        &[ok.clone(), ok],
        MemoryDomain::Cuda,
        false,
        &CostTerms::default(),
    )
    .unwrap();
    assert_eq!(fine.predicted_step, Some(Nanos(2)));
    assert_eq!(fine.estimated, 0);

    let none = select_program(&[], MemoryDomain::Cuda, false, &CostTerms::default()).unwrap();
    assert!(none.choices.is_empty());
    assert_eq!(none.predicted_step, Some(Nanos(0)));
}

#[test]
fn without_a_full_benchmark_the_estimates_decide() {
    let mut op = native_vs_ggml(0, 0);
    op.offers[0].measurement = None;
    op.offers[0].candidate.caps.as_mut().unwrap().est = Some(Nanos(900));
    op.offers[1].candidate.caps.as_mut().unwrap().est = Some(Nanos(800));
    let got = decide(&op).unwrap();
    assert_eq!(got.provider, ProviderId("ggml-cuda"));
    assert_eq!((got.basis, got.cost), (Basis::Estimate, None));

    // A missing estimate sorts last; equal estimates go to native.
    op.offers[1].candidate.caps.as_mut().unwrap().est = None;
    assert_eq!(decide(&op).unwrap().provider, ProviderId("cuda"));
    op.offers[1].candidate.caps.as_mut().unwrap().est = Some(Nanos(900));
    assert_eq!(decide(&op).unwrap().provider, ProviderId("cuda"));
}

/// A program of `n` Gemv slots in `domain`, each offered by one native provider: the first
/// `measured` slots benchmarked at `ns`, the rest left to the provider's estimate of `ns`.
fn program(
    domain: MemoryDomain,
    backend: Backend,
    n: usize,
    measured_ops: usize,
    ns: u64,
) -> Vec<OpSlot> {
    (0..n)
        .map(|i| {
            let mut c = caps(&OpKind::Gemv);
            c.est = Some(Nanos(ns));
            slot(
                OpKind::Gemv,
                vec![Offer {
                    candidate: cand("native", backend, domain, c),
                    measurement: if i < measured_ops { measured(ns) } else { None },
                }],
            )
        })
        .collect()
}

#[test]
fn a_partly_benchmarked_domain_cannot_win_on_a_partial_sum() {
    // Review of #1718: HIP measured all 10 ops at 100 µs (1000 µs); the budget ran out after
    // 5 of Vulkan's 10 ops at 110 µs. A partial sum of 550 µs would beat 1000 µs by 45 %.
    let terms = CostTerms::default();
    let hip = select_program(
        &program(MemoryDomain::Hip, Backend::Hip, 10, 10, 100_000),
        MemoryDomain::Hip,
        false,
        &terms,
    )
    .unwrap();
    let vk = select_program(
        &program(MemoryDomain::Vulkan, Backend::Vulkan, 10, 5, 110_000),
        MemoryDomain::Vulkan,
        false,
        &terms,
    )
    .unwrap();
    assert_eq!(hip.predicted_step, Some(Nanos(1_000_000)));
    assert_eq!((vk.predicted_step, vk.estimated), (None, 5));
    let opts = [
        DomainOption {
            domain: MemoryDomain::Hip,
            predicted: hip.predicted_step,
        },
        DomainOption {
            domain: MemoryDomain::Vulkan,
            predicted: vk.predicted_step,
        },
    ];
    assert_eq!(
        choose_domain(Vendor::Amd, &opts).unwrap(),
        DomainChoice {
            domain: MemoryDomain::Hip,
            reason: DomainReason::Unmeasured(MemoryDomain::Vulkan)
        }
    );
    // Fully benchmarked, the same Vulkan times lose on the like-for-like sum (1100 µs).
    let vk_full = select_program(
        &program(MemoryDomain::Vulkan, Backend::Vulkan, 10, 10, 110_000),
        MemoryDomain::Vulkan,
        false,
        &terms,
    )
    .unwrap();
    assert_eq!(vk_full.predicted_step, Some(Nanos(1_100_000)));
    let full = [
        opts[0],
        DomainOption {
            domain: MemoryDomain::Vulkan,
            predicted: vk_full.predicted_step,
        },
    ];
    assert_eq!(
        choose_domain(Vendor::Amd, &full).unwrap().domain,
        MemoryDomain::Hip
    );
}

// ---- Domain choice -------------------------------------------------------------------------

fn opened(domains: &[(MemoryDomain, Option<u64>)]) -> Vec<DomainOption> {
    domains
        .iter()
        .map(|&(domain, t)| DomainOption {
            domain,
            predicted: t.map(Nanos),
        })
        .collect()
}

#[test]
fn nvidia_takes_cuda_whenever_it_opens() {
    use MemoryDomain::*;
    let sycl = Ggml(GgmlBackend::Sycl);
    let pick = |d: &[_]| choose_domain(Vendor::Nvidia, &opened(d));
    // Even when Vulkan predicts a far faster step.
    let got = pick(&[(Vulkan, Some(1)), (Cuda, Some(1000))]).unwrap();
    assert_eq!((got.domain, got.reason), (Cuda, DomainReason::Rule));
    assert_eq!(
        pick(&[(sycl, None), (Vulkan, None)]).unwrap().domain,
        Vulkan
    );
    let ggml = pick(&[(Ggml(GgmlBackend::Vulkan), None), (sycl, None)]).unwrap();
    assert_eq!(
        (ggml.domain, ggml.reason),
        (Ggml(GgmlBackend::Vulkan), DomainReason::FirstThatOpens)
    );
    assert_eq!(pick(&[]), Err(DomainError::NoneOpens(Vendor::Nvidia)));
}

#[test]
fn amd_and_intel_need_a_5_percent_margin() {
    use MemoryDomain::*;
    let amd = |hip, vk| {
        choose_domain(
            Vendor::Amd,
            &opened(&[(Vulkan, Some(vk)), (Hip, Some(hip))]),
        )
    };
    assert_eq!(amd(100_000, 95_100).unwrap().domain, Hip); // 4.9 % faster
    assert_eq!(amd(100_000, 95_000).unwrap().domain, Hip); // exactly 5 %
    assert_eq!(amd(100_000, 94_900).unwrap().domain, Vulkan); // 5.1 % faster
    assert_eq!(amd(90_000, 100_000).unwrap().domain, Hip);
    assert_eq!(
        amd(100_000, 94_900).unwrap().reason,
        DomainReason::Measured {
            default: Nanos(100_000),
            challenger: Nanos(94_900)
        }
    );

    let sycl = Ggml(GgmlBackend::Sycl);
    let intel = |vk, sy| {
        choose_domain(
            Vendor::IntelGpu,
            &opened(&[(sycl, Some(sy)), (Vulkan, Some(vk))]),
        )
    };
    assert_eq!(intel(100_000, 95_100).unwrap().domain, Vulkan);
    assert_eq!(intel(100_000, 94_900).unwrap().domain, sycl);
}

#[test]
fn a_measured_domain_choice_needs_both_measurements() {
    use MemoryDomain::*;
    let only_vk = choose_domain(Vendor::Amd, &opened(&[(Vulkan, None)])).unwrap();
    assert_eq!(
        (only_vk.domain, only_vk.reason),
        (Vulkan, DomainReason::OnlyOneOpens)
    );
    let only_hip = choose_domain(Vendor::Amd, &opened(&[(Hip, None), (Cuda, Some(1))])).unwrap();
    assert_eq!(only_hip.domain, Hip);
    // Both open but one prediction is missing: the default stays, whichever is missing.
    let vk_missing = choose_domain(Vendor::Amd, &opened(&[(Hip, Some(1000)), (Vulkan, None)]));
    assert_eq!(
        vk_missing.unwrap(),
        DomainChoice {
            domain: Hip,
            reason: DomainReason::Unmeasured(Vulkan)
        }
    );
    let hip_missing = choose_domain(Vendor::Amd, &opened(&[(Hip, None), (Vulkan, Some(1))]));
    assert_eq!(
        hip_missing.unwrap(),
        DomainChoice {
            domain: Hip,
            reason: DomainReason::Unmeasured(Hip)
        }
    );
    assert_eq!(
        choose_domain(Vendor::Amd, &opened(&[(Cuda, Some(1))])),
        Err(DomainError::NoneOpens(Vendor::Amd))
    );
}

#[test]
fn fixed_vendors_take_their_one_domain() {
    use MemoryDomain::*;
    let metal = choose_domain(Vendor::Apple, &opened(&[(Host, Some(1)), (Metal, None)])).unwrap();
    assert_eq!((metal.domain, metal.reason), (Metal, DomainReason::Rule));
    assert_eq!(
        choose_domain(Vendor::Apple, &opened(&[(Host, None)])),
        Err(DomainError::NoneOpens(Vendor::Apple))
    );
    assert_eq!(
        choose_domain(Vendor::Cpu, &opened(&[(Host, None)]))
            .unwrap()
            .domain,
        Host
    );
    let b = GgmlBackend::Sycl;
    assert_eq!(
        choose_domain(Vendor::GgmlOnly(b), &opened(&[(Ggml(b), None)]))
            .unwrap()
            .domain,
        Ggml(b)
    );
    assert!(
        DomainError::NoneOpens(Vendor::Cpu)
            .to_string()
            .contains("Cpu")
    );
}

#[test]
fn device_classes_map_to_their_row() {
    let vk = |vendor| DeviceClass::Vulkan {
        vendor,
        coopmat: false,
        subgroup: 32,
    };
    assert_eq!(
        Vendor::of_class(&DeviceClass::Cuda { sm: 89 }),
        Some(Vendor::Nvidia)
    );
    assert_eq!(
        Vendor::of_class(&DeviceClass::Hip { gfx: 0x1100 }),
        Some(Vendor::Amd)
    );
    assert_eq!(
        Vendor::of_class(&DeviceClass::Metal {
            family: AppleGpuFamily::Apple7,
            metal4: false
        }),
        Some(Vendor::Apple)
    );
    assert_eq!(
        Vendor::of_class(&DeviceClass::Cpu { isa: CpuIsa::X86V2 }),
        Some(Vendor::Cpu)
    );
    assert_eq!(Vendor::of_class(&vk(0x10DE)), Some(Vendor::Nvidia));
    assert_eq!(Vendor::of_class(&vk(0x1002)), Some(Vendor::Amd));
    assert_eq!(Vendor::of_class(&vk(0x8086)), Some(Vendor::IntelGpu));
    assert_eq!(Vendor::of_class(&vk(0x5143)), None);
    assert_eq!(
        Vendor::of_class(&DeviceClass::Npu {
            vendor: 1,
            model: 2
        }),
        None
    );
}

// ---- Determinism ---------------------------------------------------------------------------

const PROVIDERS: [(&str, Backend); 4] = [
    ("cuda", Backend::Cuda),
    ("cuda-alt", Backend::Cuda),
    ("ggml-cuda", Backend::Ggml(GgmlBackend::Cuda)),
    ("ggml-x", Backend::Ggml(GgmlBackend::Cuda)),
];

fn arb_offer() -> impl Strategy<Value = Offer> {
    (
        0..PROVIDERS.len(),
        0u32..3,
        prop::option::weighted(0.9, 1u64..2000),
        prop::option::of(1u64..2000),
        any::<bool>(),
        any::<bool>(),
        0u32..3,
        0u32..3,
    )
        .prop_map(|(p, variant, ns, est, capture_safe, det, breaks, fences)| {
            let mut c = caps(&OpKind::Gemv);
            c.est = est.map(Nanos);
            c.capture_safe = capture_safe;
            c.deterministic = det;
            let (name, backend) = PROVIDERS[p];
            Offer {
                candidate: Candidate {
                    variant,
                    ..cand(name, backend, MemoryDomain::Cuda, c)
                },
                measurement: ns.map(|ns| Measurement {
                    measured: Nanos(ns),
                    breaks,
                    fences,
                }),
            }
        })
}

proptest! {
    #[test]
    fn identical_inputs_give_identical_choices(
        offers in prop::collection::vec(arb_offer(), 0..8),
        cached in prop::option::of((0..PROVIDERS.len(), 0u32..3)),
        deterministic in any::<bool>(),
        rotate in 0usize..8,
    ) {
        // Distinct (provider, variant) pairs, as a provider offers each variant once.
        let mut seen = std::collections::BTreeSet::new();
        let offers: Vec<_> = offers
            .into_iter()
            .filter(|o| seen.insert((o.candidate.provider, o.candidate.variant)))
            .collect();
        let terms = CostTerms { t_submit: Nanos(300), t_fence: Nanos(70) };
        let op = OpSlot {
            cached: cached.map(|(p, v)| (ProviderId(PROVIDERS[p].0), v)),
            ..slot(OpKind::Gemv, offers)
        };
        let first = decide_op(&op, 0, MemoryDomain::Cuda, deterministic, &terms);
        let again = decide_op(&op.clone(), 0, MemoryDomain::Cuda, deterministic, &terms);
        prop_assert_eq!(&first, &again);

        // The order the candidates are listed in does not matter either.
        let mut reordered = op.clone();
        reordered.offers.reverse();
        if !reordered.offers.is_empty() {
            let n = rotate % reordered.offers.len();
            reordered.offers.rotate_left(n);
        }
        let shuffled = decide_op(&reordered, 0, MemoryDomain::Cuda, deterministic, &terms);
        prop_assert_eq!(&first, &shuffled);

        // A chosen candidate is always one that passes the filter, and the challenger rule
        // holds: a replacement is more than 3 % cheaper than the native incumbent.
        if let Ok(choice) = first {
            let offer = op
                .offers
                .iter()
                .find(|o| o.candidate.provider == choice.provider && o.candidate.variant == choice.variant)
                .expect("chosen from the offers");
            let req = Requirements { domain: MemoryDomain::Cuda, layout: op.layout, deterministic };
            prop_assert!(admit(&op.key, &req, &offer.candidate).is_ok());
            if let Basis::Challenger { replaced } = choice.basis {
                let inc = op.offers.iter().find(|o| (o.candidate.provider, o.candidate.variant) == replaced).unwrap();
                let inc_cost = inc.measurement.unwrap().cost(inc.candidate.caps.unwrap().capture_safe, &terms);
                prop_assert!(u128::from(choice.cost.unwrap().0) * 100 < u128::from(inc_cost.0) * 97);
            }
        }
    }
}
