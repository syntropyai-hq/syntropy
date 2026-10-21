// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared fixtures and generators for the schema tests.
//!
//! Every value here is synthetic test data, not a measurement.

use std::collections::{BTreeMap, BTreeSet};

use proptest::collection::{btree_map, btree_set, vec};
use proptest::prelude::*;
use syntropy_bench::schema::*;

/// The committed JSON schema.
pub const SCHEMA_FILE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/schema/bench-result.v1.schema.json"
);

/// The committed golden record, `sample_result()` in JSON.
pub const GOLDEN_FILE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/schema/data/result-v1.json"
);

fn digest32(fill: u8) -> Blake3 {
    Blake3([fill; 32])
}

/// A build with synthetic identity.
pub fn sample_build() -> BuildId {
    BuildId {
        version: "0.1.0".into(),
        commit: GitSha([0x11; 20]),
        profile: BuildProfile::Release,
        target: "x86_64-unknown-linux-gnu".into(),
        toolchain: digest32(0x22),
    }
}

/// A key with every field set.
pub fn sample_key() -> BenchKey {
    BenchKey {
        scenario: ScenarioId("tiered.single.fixture.q4km.sustained".into()),
        scenario_version: 3,
        model: ModelId(digest32(0x33)),
        quant: QuantClass("q4km".into()),
        prompt_tokens: 2048,
        gen_tokens: 128,
        concurrency: 1,
        objective: Objective::Latency,
        fidelity: ExactnessPolicy::Exact,
        spec: SpecSetting::Off,
        prefix: PrefixSetting::Cold,
        topology: TopologyId("single".into()),
        interconnect: InterconnectId("none".into()),
        rig: RigFingerprint(digest32(0x44)),
        thermal: ThermalProtocol::Sustained { preload_s: 600 },
        expert_cache: CacheStart::Learned,
        build: sample_build(),
    }
}

/// A key that differs from [`sample_key`] in every field.
pub fn other_key() -> BenchKey {
    BenchKey {
        scenario: ScenarioId("decode.single.fixture.q8".into()),
        scenario_version: 4,
        model: ModelId(digest32(0x55)),
        quant: QuantClass("q8".into()),
        prompt_tokens: 4096,
        gen_tokens: 256,
        concurrency: 4,
        objective: Objective::Throughput {
            per_sequence_floor_tps: 5.0,
            target_concurrency: Some(8),
            ttft_ceiling: Some(Nanos(2_000_000_000)),
        },
        fidelity: ExactnessPolicy::Approximate {
            knobs: BTreeSet::from([ApproxKnob::KvLossy, ApproxKnob::TopkReduce]),
            params: Box::new(ApproxParams {
                kv_lossy: Some(KnobParams::Name("q8_0".into())),
                topk_reduce: Some(KnobParams::Table(BTreeMap::from([(
                    "experts".into(),
                    ParamValue::Integer(6),
                )]))),
                ..ApproxParams::default()
            }),
        },
        spec: SpecSetting::Force(SpecConfig {
            source: DraftSource::Mtp,
            max_window: 3,
        }),
        prefix: PrefixSetting::Warm,
        topology: TopologyId("pp2".into()),
        interconnect: InterconnectId("2.5GbE-switch-A".into()),
        rig: RigFingerprint(digest32(0x66)),
        thermal: ThermalProtocol::Cold,
        expert_cache: CacheStart::Cold,
        build: BuildId {
            commit: GitSha([0x77; 20]),
            ..sample_build()
        },
    }
}

/// A result that uses every record type and both arms of every `Option`.
pub fn sample_result() -> BenchResult {
    let repetition = RepResult {
        ttft_s: 0.25,
        tpot_s: 0.0625,
        itl_s: vec![0.0625, 0.0625, 0.125],
        prefill_tps: 1024.0,
        aggregate_tps: None,
        floor_attainment: None,
        steps: StepTraceRef::Stored {
            object: "steps/0.trace".into(),
            digest: digest32(0x01),
        },
        energy_j_per_token: Some(2.5),
    };
    let ci = MedianCi {
        median: 0.0625,
        ci95_low: 0.0600,
        ci95_high: 0.0650,
    };
    let pct = Percentiles {
        p50: 0.0625,
        p90: 0.07,
        p99: 0.08,
        max: 0.125,
    };
    BenchResult {
        schema: SCHEMA_VERSION,
        key: sample_key(),
        started_unix_ms: 1_700_000_000_000,
        repetitions: vec![
            repetition.clone(),
            RepResult {
                steps: StepTraceRef::NotCollected,
                aggregate_tps: Some(16.0),
                floor_attainment: Some(1.0),
                energy_j_per_token: None,
                ..repetition
            },
        ],
        summary: Summary {
            ttft_s: ci,
            tpot_s: ci,
            prefill_tps: MedianCi {
                median: 1024.0,
                ci95_low: 1000.0,
                ci95_high: 1050.0,
            },
            aggregate_tps: Some(MedianCi {
                median: 16.0,
                ci95_low: 15.5,
                ci95_high: 16.5,
            }),
            floor_attainment: Some(1.0),
            itl_s: pct,
            itl_event_s: pct,
            cv: Some(0.01),
        },
        terms: TermBreakdown {
            rows: vec![
                TermRow {
                    term: Term::Kernel {
                        stage: 0,
                        tier: TierKind::Vram,
                        kernel_class: "DenseGemv".into(),
                    },
                    bytes: Some(1 << 30),
                    w_probe_bps: Some(9.0e11),
                    w_ach_bps: Some(8.5e11),
                    predicted_s: Some(0.0012),
                    model_s: Some(0.0011),
                    measured_s: 0.0013,
                    flagged: true,
                },
                TermRow {
                    term: Term::Miss { stage: 0 },
                    bytes: None,
                    w_probe_bps: None,
                    w_ach_bps: None,
                    predicted_s: None,
                    model_s: None,
                    measured_s: 0.0001,
                    flagged: false,
                },
                TermRow {
                    term: Term::Fixed { stage: 0 },
                    bytes: None,
                    w_probe_bps: None,
                    w_ach_bps: None,
                    predicted_s: Some(0.0002),
                    model_s: Some(0.0002),
                    measured_s: 0.0002,
                    flagged: false,
                },
                TermRow {
                    term: Term::Hop { from: 0, to: 1 },
                    bytes: Some(16384),
                    w_probe_bps: Some(3.0e8),
                    w_ach_bps: None,
                    predicted_s: Some(0.0001),
                    model_s: Some(0.0001),
                    measured_s: 0.0001,
                    flagged: false,
                },
                TermRow {
                    term: Term::Sample,
                    bytes: None,
                    w_probe_bps: None,
                    w_ach_bps: None,
                    predicted_s: None,
                    model_s: None,
                    measured_s: 0.00005,
                    flagged: false,
                },
            ],
            remainder_s: Some(0.00001),
        },
        bounds: Bounds {
            t_model_s: Some(0.06),
            t_peak_s: Some(0.05),
            overlap: Some(OverlapBound {
                t_overlap_s: 0.055,
                hit_rate: 0.9,
            }),
        },
        plan: PlanDigest {
            epoch: PlanEpoch { term: 1, seq: 2 },
            stages: vec![StageDigest {
                node: "node-a".into(),
                layer_start: 0,
                layer_end: 32,
                tier_bytes: BTreeMap::from([
                    (TierKind::Vram, 1 << 34),
                    (TierKind::HostPinned, 1 << 35),
                ]),
            }],
            predicted_t_tok_s: Some(0.06),
        },
        parity: ParityRef {
            report: digest32(0x02),
            path: "expert-tiered".into(),
            verdict: ParityVerdict::Pass,
        },
        env: EnvCapture {
            rig_id: "R-FIXTURE".into(),
            checks: vec![
                EnvCheck {
                    check: EnvCheckKind::PowerSource,
                    passed: true,
                    observed: "mains".into(),
                },
                EnvCheck {
                    check: EnvCheckKind::Temperatures,
                    passed: true,
                    observed: "within 2 °C".into(),
                },
            ],
            dirty: None,
            driver: Some("fixture-driver 1.0".into()),
            firmware: BTreeMap::from([("vbios".into(), "1.2.3".into())]),
            kernel: "6.6.0".into(),
            microcode: None,
            gpu_clock_mhz: Some(ClockRange {
                min_mhz: 2400,
                max_mhz: 2500,
            }),
        },
        config_hash: digest32(0x03),
        baseline_engine: Some(BaselineRun {
            engine: "fixture-engine".into(),
            version: "1.0.0".into(),
            commit: "abc123".into(),
            command: vec![
                "fixture-engine".into(),
                "--model".into(),
                "model.gguf".into(),
            ],
            config_files: vec![FileDigest {
                path: "engine.toml".into(),
                blake3: digest32(0x04),
            }],
            sweep: vec![SweepPoint {
                options: BTreeMap::from([("threads".into(), "8".into())]),
                decode_tps: 15.0,
            }],
            model: BaselineModel::FormatAdjusted {
                command: vec!["convert".into(), "model.gguf".into()],
                file: digest32(0x05),
            },
            fidelity_settings: BaselineFidelity::Default {
                approximations: vec!["expert_skip".into()],
            },
            measured_fidelity: Some(MeasuredFidelity {
                top_n: 20,
                top1_agreement: 0.99,
                kl_top_n: 0.001,
            }),
            repetitions: vec![BaselineRep {
                ttft_s: 0.3,
                tpot_s: 0.07,
                aggregate_tps: None,
            }],
        }),
        verdict: Some(GateVerdict {
            gate: GateKind::Pr,
            against: Some(other_key().build),
            metrics: vec![MetricVerdict {
                metric: "decode_rate".into(),
                r_hat: -0.01,
                band: 0.02,
                p_value: Some(0.25),
                outcome: GateOutcome::Pass,
            }],
            outcome: GateOutcome::Pass,
            override_reason: None,
        }),
        trust: Trust::Trusted,
    }
}

// ---- generators ----

/// Finite `f64`s of every class and sign; JSON carries no NaN or infinity.
pub fn finite_f64() -> impl Strategy<Value = f64> {
    use proptest::num::f64 as f;
    f::POSITIVE | f::NEGATIVE | f::NORMAL | f::SUBNORMAL | f::ZERO
}

/// Finite `f32`s of every class and sign.
pub fn finite_f32() -> impl Strategy<Value = f32> {
    use proptest::num::f32 as f;
    f::POSITIVE | f::NEGATIVE | f::NORMAL | f::SUBNORMAL | f::ZERO
}

fn text() -> impl Strategy<Value = String> {
    any::<String>()
}

fn blake3() -> impl Strategy<Value = Blake3> {
    any::<[u8; 32]>().prop_map(Blake3)
}

fn opt<S: Strategy>(s: S) -> impl Strategy<Value = Option<S::Value>> {
    proptest::option::of(s)
}

pub fn build_id() -> BoxedStrategy<BuildId> {
    let profile = prop_oneof![
        Just(BuildProfile::Dev),
        Just(BuildProfile::TestRelease),
        Just(BuildProfile::Chaos),
        Just(BuildProfile::Pr),
        Just(BuildProfile::Release),
    ];
    (text(), any::<[u8; 20]>(), profile, text(), blake3())
        .prop_map(|(version, commit, profile, target, toolchain)| BuildId {
            version,
            commit: GitSha(commit),
            profile,
            target,
            toolchain,
        })
        .boxed()
}

fn approx_knob() -> impl Strategy<Value = ApproxKnob> {
    prop_oneof![
        Just(ApproxKnob::ExpertSkip),
        Just(ApproxKnob::ExpertDefer),
        Just(ApproxKnob::MissLowPrecision),
        Just(ApproxKnob::TopkReduce),
        Just(ApproxKnob::KvLossy),
        Just(ApproxKnob::WireCompress),
        Just(ApproxKnob::ApproxAttention),
        Just(ApproxKnob::SpecLenient),
    ]
}

fn param_value() -> impl Strategy<Value = ParamValue> {
    prop_oneof![
        any::<bool>().prop_map(ParamValue::Bool),
        any::<i64>().prop_map(ParamValue::Integer),
        finite_f64().prop_map(ParamValue::Number),
        text().prop_map(ParamValue::Text),
    ]
}

fn approx_params() -> BoxedStrategy<ApproxParams> {
    let knob = || {
        opt(prop_oneof![
            text().prop_map(KnobParams::Name),
            btree_map(text(), param_value(), 0..3).prop_map(KnobParams::Table),
        ])
    };
    (
        knob(),
        knob(),
        knob(),
        knob(),
        knob(),
        knob(),
        knob(),
        knob(),
    )
        .prop_map(
            |(
                expert_skip,
                expert_defer,
                miss_low_precision,
                topk_reduce,
                kv_lossy,
                wire_compress,
                approx_attention,
                spec_lenient,
            )| ApproxParams {
                expert_skip,
                expert_defer,
                miss_low_precision,
                topk_reduce,
                kv_lossy,
                wire_compress,
                approx_attention,
                spec_lenient,
            },
        )
        .boxed()
}

pub fn bench_key() -> BoxedStrategy<BenchKey> {
    let objective = prop_oneof![
        Just(Objective::Latency),
        (
            finite_f32(),
            opt(any::<u16>()),
            opt(any::<u64>().prop_map(Nanos))
        )
            .prop_map(
                |(per_sequence_floor_tps, target_concurrency, ttft_ceiling)| {
                    Objective::Throughput {
                        per_sequence_floor_tps,
                        target_concurrency,
                        ttft_ceiling,
                    }
                }
            ),
    ];
    let fidelity = prop_oneof![
        Just(ExactnessPolicy::Exact),
        (btree_set(approx_knob(), 1..=8), approx_params()).prop_map(|(knobs, params)| {
            ExactnessPolicy::Approximate {
                knobs,
                params: Box::new(params),
            }
        }),
    ];
    let source = prop_oneof![
        Just(DraftSource::Mtp),
        Just(DraftSource::DraftHead),
        blake3().prop_map(|d| DraftSource::DraftModel { model: ModelId(d) }),
    ];
    let spec = prop_oneof![
        Just(SpecSetting::Off),
        Just(SpecSetting::Auto),
        (source, any::<u16>())
            .prop_map(|(source, max_window)| SpecSetting::Force(SpecConfig { source, max_window })),
    ];
    let prefix = prop_oneof![Just(PrefixSetting::Cold), Just(PrefixSetting::Warm)];
    let thermal = prop_oneof![
        Just(ThermalProtocol::Cold),
        any::<u32>().prop_map(|preload_s| ThermalProtocol::Sustained { preload_s }),
    ];
    let expert_cache = prop_oneof![Just(CacheStart::Learned), Just(CacheStart::Cold)];
    (
        (
            text(),
            any::<u16>(),
            blake3(),
            text(),
            any::<u32>(),
            any::<u32>(),
            any::<u16>(),
            objective,
        ),
        (
            fidelity,
            spec,
            prefix,
            text(),
            text(),
            blake3(),
            thermal,
            expert_cache,
        ),
        build_id(),
    )
        .prop_map(
            |(
                (
                    scenario,
                    scenario_version,
                    model,
                    quant,
                    prompt_tokens,
                    gen_tokens,
                    concurrency,
                    objective,
                ),
                (fidelity, spec, prefix, topology, interconnect, rig, thermal, expert_cache),
                build,
            )| BenchKey {
                scenario: ScenarioId(scenario),
                scenario_version,
                model: ModelId(model),
                quant: QuantClass(quant),
                prompt_tokens,
                gen_tokens,
                concurrency,
                objective,
                fidelity,
                spec,
                prefix,
                topology: TopologyId(topology),
                interconnect: InterconnectId(interconnect),
                rig: RigFingerprint(rig),
                thermal,
                expert_cache,
                build,
            },
        )
        .boxed()
}

/// Copies `field` from `from` into `into`.
pub fn copy_field(into: &mut BenchKey, from: &BenchKey, field: KeyField) {
    let from = from.clone();
    match field {
        KeyField::Scenario => into.scenario = from.scenario,
        KeyField::ScenarioVersion => into.scenario_version = from.scenario_version,
        KeyField::Model => into.model = from.model,
        KeyField::Quant => into.quant = from.quant,
        KeyField::PromptTokens => into.prompt_tokens = from.prompt_tokens,
        KeyField::GenTokens => into.gen_tokens = from.gen_tokens,
        KeyField::Concurrency => into.concurrency = from.concurrency,
        KeyField::Objective => into.objective = from.objective,
        KeyField::Fidelity => into.fidelity = from.fidelity,
        KeyField::Spec => into.spec = from.spec,
        KeyField::Prefix => into.prefix = from.prefix,
        KeyField::Topology => into.topology = from.topology,
        KeyField::Interconnect => into.interconnect = from.interconnect,
        KeyField::Rig => into.rig = from.rig,
        KeyField::Thermal => into.thermal = from.thermal,
        KeyField::ExpertCache => into.expert_cache = from.expert_cache,
        KeyField::Build => into.build = from.build,
    }
}

fn step_trace() -> impl Strategy<Value = StepTraceRef> {
    prop_oneof![
        Just(StepTraceRef::NotCollected),
        (text(), blake3()).prop_map(|(object, digest)| StepTraceRef::Stored { object, digest }),
    ]
}

fn rep_result() -> BoxedStrategy<RepResult> {
    (
        finite_f64(),
        finite_f64(),
        vec(finite_f32(), 0..16),
        finite_f64(),
        opt(finite_f64()),
        opt(finite_f32()),
        step_trace(),
        opt(finite_f64()),
    )
        .prop_map(
            |(
                ttft_s,
                tpot_s,
                itl_s,
                prefill_tps,
                aggregate_tps,
                floor_attainment,
                steps,
                energy_j_per_token,
            )| {
                RepResult {
                    ttft_s,
                    tpot_s,
                    itl_s,
                    prefill_tps,
                    aggregate_tps,
                    floor_attainment,
                    steps,
                    energy_j_per_token,
                }
            },
        )
        .boxed()
}

fn median_ci() -> impl Strategy<Value = MedianCi> {
    (finite_f64(), finite_f64(), finite_f64()).prop_map(|(median, ci95_low, ci95_high)| MedianCi {
        median,
        ci95_low,
        ci95_high,
    })
}

fn percentiles() -> impl Strategy<Value = Percentiles> {
    (finite_f64(), finite_f64(), finite_f64(), finite_f64())
        .prop_map(|(p50, p90, p99, max)| Percentiles { p50, p90, p99, max })
}

fn summary() -> BoxedStrategy<Summary> {
    (
        median_ci(),
        median_ci(),
        median_ci(),
        opt(median_ci()),
        opt(finite_f32()),
        percentiles(),
        percentiles(),
        opt(finite_f64()),
    )
        .prop_map(
            |(
                ttft_s,
                tpot_s,
                prefill_tps,
                aggregate_tps,
                floor_attainment,
                itl_s,
                itl_event_s,
                cv,
            )| Summary {
                ttft_s,
                tpot_s,
                prefill_tps,
                aggregate_tps,
                floor_attainment,
                itl_s,
                itl_event_s,
                cv,
            },
        )
        .boxed()
}

fn tier_kind() -> impl Strategy<Value = TierKind> {
    prop_oneof![
        Just(TierKind::Vram),
        Just(TierKind::Unified),
        Just(TierKind::HostPinned),
        Just(TierKind::HostPageable),
        Just(TierKind::Nvme),
    ]
}

fn term() -> impl Strategy<Value = Term> {
    prop_oneof![
        (any::<u16>(), tier_kind(), text()).prop_map(|(stage, tier, kernel_class)| Term::Kernel {
            stage,
            tier,
            kernel_class
        }),
        any::<u16>().prop_map(|stage| Term::Miss { stage }),
        any::<u16>().prop_map(|stage| Term::Fixed { stage }),
        (any::<u16>(), any::<u16>()).prop_map(|(from, to)| Term::Hop { from, to }),
        Just(Term::Sample),
    ]
}

fn term_row() -> impl Strategy<Value = TermRow> {
    (
        term(),
        opt(any::<u64>()),
        opt(finite_f64()),
        opt(finite_f64()),
        opt(finite_f64()),
        opt(finite_f64()),
        finite_f64(),
        any::<bool>(),
    )
        .prop_map(
            |(term, bytes, w_probe_bps, w_ach_bps, predicted_s, model_s, measured_s, flagged)| {
                TermRow {
                    term,
                    bytes,
                    w_probe_bps,
                    w_ach_bps,
                    predicted_s,
                    model_s,
                    measured_s,
                    flagged,
                }
            },
        )
}

fn terms() -> BoxedStrategy<TermBreakdown> {
    (vec(term_row(), 0..4), opt(finite_f64()))
        .prop_map(|(rows, remainder_s)| TermBreakdown { rows, remainder_s })
        .boxed()
}

fn bounds() -> BoxedStrategy<Bounds> {
    let overlap = (finite_f64(), finite_f64()).prop_map(|(t_overlap_s, hit_rate)| OverlapBound {
        t_overlap_s,
        hit_rate,
    });
    (opt(finite_f64()), opt(finite_f64()), opt(overlap))
        .prop_map(|(t_model_s, t_peak_s, overlap)| Bounds {
            t_model_s,
            t_peak_s,
            overlap,
        })
        .boxed()
}

fn plan() -> BoxedStrategy<PlanDigest> {
    let stage = (
        text(),
        any::<u16>(),
        any::<u16>(),
        btree_map(tier_kind(), any::<u64>(), 0..5),
    )
        .prop_map(|(node, layer_start, layer_end, tier_bytes)| StageDigest {
            node,
            layer_start,
            layer_end,
            tier_bytes,
        });
    (
        any::<u64>(),
        any::<u64>(),
        vec(stage, 0..3),
        opt(finite_f64()),
    )
        .prop_map(|(term, seq, stages, predicted_t_tok_s)| PlanDigest {
            epoch: PlanEpoch { term, seq },
            stages,
            predicted_t_tok_s,
        })
        .boxed()
}

fn parity() -> BoxedStrategy<ParityRef> {
    let verdict = prop_oneof![
        Just(ParityVerdict::Pass),
        Just(ParityVerdict::Fail),
        Just(ParityVerdict::ReferenceDefect)
    ];
    (blake3(), text(), verdict)
        .prop_map(|(report, path, verdict)| ParityRef {
            report,
            path,
            verdict,
        })
        .boxed()
}

fn env_capture() -> BoxedStrategy<EnvCapture> {
    let kind = prop_oneof![
        Just(EnvCheckKind::PowerSource),
        Just(EnvCheckKind::PowerPlan),
        Just(EnvCheckKind::OtherProcesses),
        Just(EnvCheckKind::MemoryPagedOut),
        Just(EnvCheckKind::Temperatures),
        Just(EnvCheckKind::PageCache),
        Just(EnvCheckKind::Network),
        Just(EnvCheckKind::OsBackgroundWork),
    ];
    let check = (kind, any::<bool>(), text()).prop_map(|(check, passed, observed)| EnvCheck {
        check,
        passed,
        observed,
    });
    let clock =
        (any::<u32>(), any::<u32>()).prop_map(|(min_mhz, max_mhz)| ClockRange { min_mhz, max_mhz });
    (
        text(),
        vec(check, 0..4),
        opt(text()),
        opt(text()),
        btree_map(text(), text(), 0..3),
        text(),
        opt(text()),
        opt(clock),
    )
        .prop_map(
            |(rig_id, checks, dirty, driver, firmware, kernel, microcode, gpu_clock_mhz)| {
                EnvCapture {
                    rig_id,
                    checks,
                    dirty,
                    driver,
                    firmware,
                    kernel,
                    microcode,
                    gpu_clock_mhz,
                }
            },
        )
        .boxed()
}

fn baseline_run() -> BoxedStrategy<BaselineRun> {
    let file = (text(), blake3()).prop_map(|(path, blake3)| FileDigest { path, blake3 });
    let point =
        (btree_map(text(), text(), 0..3), finite_f64()).prop_map(|(options, decode_tps)| {
            SweepPoint {
                options,
                decode_tps,
            }
        });
    let model = prop_oneof![
        blake3().prop_map(|file| BaselineModel::Unchanged { file }),
        (vec(text(), 0..3), blake3())
            .prop_map(|(command, file)| BaselineModel::FormatAdjusted { command, file }),
    ];
    let settings = prop_oneof![
        Just(BaselineFidelity::Exact),
        vec(text(), 0..3).prop_map(|approximations| BaselineFidelity::Default { approximations }),
    ];
    let measured =
        (any::<u16>(), finite_f64(), finite_f64()).prop_map(|(top_n, top1_agreement, kl_top_n)| {
            MeasuredFidelity {
                top_n,
                top1_agreement,
                kl_top_n,
            }
        });
    let rep = (finite_f64(), finite_f64(), opt(finite_f64())).prop_map(
        |(ttft_s, tpot_s, aggregate_tps)| BaselineRep {
            ttft_s,
            tpot_s,
            aggregate_tps,
        },
    );
    (
        (text(), text(), text(), vec(text(), 0..3), vec(file, 0..2)),
        (
            vec(point, 0..3),
            model,
            settings,
            opt(measured),
            vec(rep, 0..3),
        ),
    )
        .prop_map(
            |(
                (engine, version, commit, command, config_files),
                (sweep, model, fidelity_settings, measured_fidelity, repetitions),
            )| BaselineRun {
                engine,
                version,
                commit,
                command,
                config_files,
                sweep,
                model,
                fidelity_settings,
                measured_fidelity,
                repetitions,
            },
        )
        .boxed()
}

fn gate_verdict() -> BoxedStrategy<GateVerdict> {
    let outcome = || {
        prop_oneof![
            Just(GateOutcome::Pass),
            Just(GateOutcome::Fail),
            Just(GateOutcome::ReportOnly)
        ]
    };
    let kind = prop_oneof![
        Just(GateKind::Pr),
        Just(GateKind::Trend),
        Just(GateKind::Bisection),
        Just(GateKind::Release)
    ];
    let metric = (
        text(),
        finite_f64(),
        finite_f64(),
        opt(finite_f64()),
        outcome(),
    )
        .prop_map(|(metric, r_hat, band, p_value, outcome)| MetricVerdict {
            metric,
            r_hat,
            band,
            p_value,
            outcome,
        });
    (
        kind,
        opt(build_id()),
        vec(metric, 0..3),
        outcome(),
        opt(text()),
    )
        .prop_map(
            |(gate, against, metrics, outcome, override_reason)| GateVerdict {
                gate,
                against,
                metrics,
                outcome,
                override_reason,
            },
        )
        .boxed()
}

pub fn bench_result() -> BoxedStrategy<BenchResult> {
    let trust = prop_oneof![Just(Trust::Trusted), Just(Trust::Untrusted)];
    (
        (
            bench_key(),
            any::<u64>(),
            vec(rep_result(), 0..4),
            summary(),
            terms(),
            bounds(),
        ),
        (
            plan(),
            parity(),
            env_capture(),
            blake3(),
            opt(baseline_run()),
            opt(gate_verdict()),
            trust,
        ),
    )
        .prop_map(
            |(
                (key, started_unix_ms, repetitions, summary, terms, bounds),
                (plan, parity, env, config_hash, baseline_engine, verdict, trust),
            )| BenchResult {
                schema: SCHEMA_VERSION,
                key,
                started_unix_ms,
                repetitions,
                summary,
                terms,
                bounds,
                plan,
                parity,
                env,
                config_hash,
                baseline_engine,
                verdict,
                trust,
            },
        )
        .boxed()
}
