// SPDX-License-Identifier: MIT OR Apache-2.0

//! The benchmark result record of RFC-0061 §2, stored as JSON.
//!
//! [`BenchResult::from_json`] is the validation entry point. It rejects unknown fields, a
//! missing field (including a field whose value may be `null`: it must be written), a digest
//! that is not lowercase hex, and a `schema` other than [`SCHEMA_VERSION`].
//! `crates/bench/schema/bench-result.v1.schema.json` describes the same format for tools
//! outside Rust, and the crate's tests check one against the other.
//!
//! RFC-0061 §2.1 and §2.2 fix the fields of [`BenchKey`], [`BenchResult`] and [`RepResult`].
//! The fields of the other records follow the prose of the sections each one cites.

mod digest;
mod key;

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Deserializer, Serialize};

pub use digest::{Blake3, GitSha, HexError};
pub use key::{
    ApproxKnob, ApproxParams, BenchKey, BuildId, BuildProfile, CacheStart, DraftSource,
    ExactnessPolicy, InterconnectId, KeyField, KnobParams, ModelId, Nanos, Objective, ParamValue,
    PrefixSetting, QuantClass, RigFingerprint, ScenarioId, SpecConfig, SpecSetting,
    ThermalProtocol, TopologyId,
};

/// The `schema` value this crate writes and the only one it reads.
pub const SCHEMA_VERSION: u16 = 1;

/// Makes an `Option` field required: it must be written, as `null` when absent.
fn present<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

/// One benchmark run of one build: the key, every repetition and what was derived from them
/// (RFC-0061 §2.2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchResult {
    /// The record format version, [`SCHEMA_VERSION`].
    pub schema: u16,
    /// What was measured, where and on which build.
    pub key: BenchKey,
    /// Wall-clock start of the run, Unix milliseconds.
    pub started_unix_ms: u64,
    /// Every repetition, raw; none is dropped as an outlier (RFC-0061 §3.3).
    pub repetitions: Vec<RepResult>,
    /// Medians, bootstrap 95% CIs and ITL percentiles.
    pub summary: Summary,
    /// Predicted against measured time per term (RFC-0061 §1.3, §5.2).
    pub terms: TermBreakdown,
    /// `t_model` (probe) and `t_peak` (spec) (RFC-0061 §5.1).
    pub bounds: Bounds,
    /// The plan the run executed.
    pub plan: PlanDigest,
    /// The RFC-0060 `ParityReport` for this build, rig and path (RFC-0061 §3.6).
    pub parity: ParityRef,
    /// The rig's state before measurement (RFC-0061 §3.5).
    pub env: EnvCapture,
    /// BLAKE3 of the resolved configuration (RFC-0054), secrets excluded.
    pub config_hash: Blake3,
    /// The baseline engine's run; comparison scenarios only (RFC-0061 §7).
    #[serde(deserialize_with = "present")]
    pub baseline_engine: Option<BaselineRun>,
    /// Set when a gate evaluated this result (RFC-0061 §8).
    #[serde(deserialize_with = "present")]
    pub verdict: Option<GateVerdict>,
    /// Whether the result may gate, become a baseline or be published (RFC-0061 §10.3).
    pub trust: Trust,
}

impl BenchResult {
    /// RFC-0061 §2.1: comparable only when the keys are equal in every field except `build`.
    pub fn comparable_with(&self, other: &BenchResult) -> bool {
        self.key.comparable_with(&other.key)
    }

    /// Parses and validates a stored result.
    pub fn from_json(text: &str) -> Result<BenchResult, SchemaError> {
        let result: BenchResult = serde_json::from_str(text).map_err(SchemaError::Json)?;
        if result.schema != SCHEMA_VERSION {
            return Err(SchemaError::Version(result.schema));
        }
        Ok(result)
    }

    /// Writes the result as compact JSON.
    ///
    /// Fails when the record would not read back as itself: a `schema` other than
    /// [`SCHEMA_VERSION`], or a NaN or infinite float, which JSON cannot carry.
    pub fn to_json(&self) -> Result<String, SchemaError> {
        let text = serde_json::to_string(self).map_err(SchemaError::Json)?;
        match BenchResult::from_json(&text) {
            Ok(back) if back == *self => Ok(text),
            Ok(_) | Err(SchemaError::Json(_)) => Err(SchemaError::NotRepresentable),
            Err(other) => Err(other),
        }
    }
}

/// Why a result failed validation or could not be written.
#[derive(Debug)]
pub enum SchemaError {
    /// Malformed JSON, a missing or unknown field, or a value of the wrong type or form.
    Json(serde_json::Error),
    /// A `schema` value this crate does not read.
    Version(u16),
    /// A value that does not survive a JSON round trip, such as a NaN or infinite float.
    NotRepresentable,
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchemaError::Json(e) => write!(f, "invalid benchmark result: {e}"),
            SchemaError::Version(found) => {
                write!(
                    f,
                    "benchmark result schema {found}, expected {SCHEMA_VERSION}"
                )
            }
            SchemaError::NotRepresentable => {
                f.write_str("benchmark result holds a value JSON cannot carry (NaN or infinity)")
            }
        }
    }
}

impl std::error::Error for SchemaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SchemaError::Json(e) => Some(e),
            SchemaError::Version(_) | SchemaError::NotRepresentable => None,
        }
    }
}

/// Who wrote a result, which decides what it may be used for (RFC-0061 §10.3, RFC-0062 §3.3).
///
/// The writer of record sets it, never the run itself: the trusted ingestion job writes
/// `untrusted` on every fork-derived result. Every producer other than the two writers of record
/// (`lab.yml` jobs on untainted rigs and that ingestion job), such as a user's `syntropy bench`
/// run, writes `untrusted`. An `untrusted` result is shown on its PR and never gates `main`,
/// becomes a baseline or is published.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Trust {
    /// Written by a `lab.yml` job on an untainted rig or by the trusted ingestion job.
    Trusted,
    /// Derived from a fork's run.
    Untrusted,
}

/// One repetition's raw metrics (RFC-0061 §1.2, §2.2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepResult {
    /// Time to first token, `c_first − c_send`, seconds.
    pub ttft_s: f64,
    /// Time per output token, `(c_last − c_first) / (n_gen − 1)`, seconds.
    pub tpot_s: f64,
    /// Inter-token latency per token, `c_k − c_{k−1}` for k ≥ 2, seconds.
    pub itl_s: Vec<f32>,
    /// Prefill rate, tokens per second.
    pub prefill_tps: f64,
    /// Aggregate rate over the steady window, tokens per second; throughput mode only.
    #[serde(deserialize_with = "present")]
    pub aggregate_tps: Option<f64>,
    /// Fraction of sequences that met the per-sequence floor; throughput mode only.
    #[serde(deserialize_with = "present")]
    pub floor_attainment: Option<f32>,
    /// Per-step telemetry, stored beside the record.
    pub steps: StepTraceRef,
    /// Joules per generated token, where a source exists (RFC-0061 §1.5).
    #[serde(deserialize_with = "present")]
    pub energy_j_per_token: Option<f64>,
}

/// Where a repetition's per-step telemetry is (RFC-0061 §2.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum StepTraceRef {
    /// No per-step telemetry was collected for this repetition.
    NotCollected,
    /// A compressed columnar file beside the record in the results store (RFC-0061 §10.3).
    Stored {
        /// The file's object name.
        object: String,
        /// BLAKE3 of the file's bytes.
        digest: Blake3,
    },
}

/// Descriptive statistics over the repetitions (RFC-0061 §3.3).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Summary {
    /// TTFT, seconds.
    pub ttft_s: MedianCi,
    /// TPOT, seconds.
    pub tpot_s: MedianCi,
    /// Prefill rate, tokens per second.
    pub prefill_tps: MedianCi,
    /// Aggregate rate, tokens per second; throughput mode only.
    #[serde(deserialize_with = "present")]
    pub aggregate_tps: Option<MedianCi>,
    /// Floor attainment over all steady windows; throughput mode only.
    #[serde(deserialize_with = "present")]
    pub floor_attainment: Option<f32>,
    /// ITL per token, seconds.
    pub itl_s: Percentiles,
    /// ITL per token event (a speculative burst is one event), seconds.
    pub itl_event_s: Percentiles,
    /// Coefficient of variation of the scenario class's statistic across repetitions; `null`
    /// with fewer than two repetitions.
    #[serde(deserialize_with = "present")]
    pub cv: Option<f64>,
}

/// A median with its bootstrap 95% confidence interval.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MedianCi {
    /// The median.
    pub median: f64,
    /// Lower end of the 95% CI of the median.
    pub ci95_low: f64,
    /// Upper end of the 95% CI of the median.
    pub ci95_high: f64,
}

/// Latency percentiles (RFC-0061 §1.2).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Percentiles {
    /// 50th percentile.
    pub p50: f64,
    /// 90th percentile.
    pub p90: f64,
    /// 99th percentile.
    pub p99: f64,
    /// Maximum.
    pub max: f64,
}

/// The per-term decomposition of a decode step, per token (RFC-0061 §1.3, §5.2).
///
/// η_kernel (`w_ach_bps / w_probe_bps`) and Δ (`measured_s − model_s`) are not stored; they
/// follow from the row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TermBreakdown {
    /// One row per term; empty when no per-step telemetry was collected.
    pub rows: Vec<TermRow>,
    /// The unexplained remainder: measured `t_tok` minus the sum of the terms, seconds.
    #[serde(deserialize_with = "present")]
    pub remainder_s: Option<f64>,
}

/// One row of the RFC-0061 §5.2 table.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TermRow {
    /// Which term.
    pub term: Term,
    /// `B_meas`: bytes read per token, or the hop's message size.
    #[serde(deserialize_with = "present")]
    pub bytes: Option<u64>,
    /// `W_probe` (or `β_probe` for a hop), bytes per second.
    #[serde(deserialize_with = "present")]
    pub w_probe_bps: Option<f64>,
    /// `W_ach = B / t_kernel`, bytes per second.
    #[serde(deserialize_with = "present")]
    pub w_ach_bps: Option<f64>,
    /// Time predicted by the plan's `PredictedCost`, seconds.
    #[serde(deserialize_with = "present")]
    pub predicted_s: Option<f64>,
    /// Time from the probe-based model, seconds.
    #[serde(deserialize_with = "present")]
    pub model_s: Option<f64>,
    /// Measured time, seconds.
    pub measured_s: f64,
    /// Measured exceeds model by more than 10% (RFC-0001 §14 rule 2).
    pub flagged: bool,
}

/// A term of the per-step decomposition (RFC-0061 §1.3).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Term {
    /// Kernel time on one tier for one kernel class, including CPU execute-in-place.
    Kernel {
        /// Stage index; 0 is the head.
        stage: u16,
        /// The tier the kernel reads.
        tier: TierKind,
        /// The kernel class, for example `DenseGemv`.
        kernel_class: String,
    },
    /// Expert-miss time, `t_miss`.
    Miss {
        /// Stage index.
        stage: u16,
    },
    /// Device hand-offs and fixed per-step time, `t_switch_total` and `t_fixed`.
    Fixed {
        /// Stage index.
        stage: u16,
    },
    /// One ring hop, `t_hop`.
    Hop {
        /// Sending stage.
        from: u16,
        /// Receiving stage.
        to: u16,
    },
    /// The sampler at the tail, `t_sample`.
    Sample,
}

/// A memory tier kind (RFC-0001 §5.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TierKind {
    /// Device-local memory of a discrete GPU.
    Vram,
    /// Memory shared by CPU and GPU.
    Unified,
    /// Host RAM registered for DMA.
    HostPinned,
    /// Host RAM that only the CPU reads.
    HostPageable,
    /// A storage volume.
    Nvme,
}

/// Per-token time bounds from measured bytes (RFC-0061 §5.1); each is `null` until computed.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bounds {
    /// `t_model`: the probe-measured bound (SC-3), seconds.
    #[serde(deserialize_with = "present")]
    pub t_model_s: Option<f64>,
    /// `t_peak`: the spec-sheet bound (SC-4), seconds.
    #[serde(deserialize_with = "present")]
    pub t_peak_s: Option<f64>,
    /// `t_overlap(h)` at the measured hit rate (SC-5); tiered MoE scenarios only.
    #[serde(deserialize_with = "present")]
    pub overlap: Option<OverlapBound>,
}

/// SC-5's overlap bound and the hit rate it was computed at.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlapBound {
    /// `t_overlap(h)`, seconds.
    pub t_overlap_s: f64,
    /// The measured expert-cache hit rate `h`.
    pub hit_rate: f64,
}

/// A summary of the plan the run executed (RFC-0061 §2.2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanDigest {
    /// The plan's epoch.
    pub epoch: PlanEpoch,
    /// The stages, head first.
    pub stages: Vec<StageDigest>,
    /// The plan's predicted `t_tok` from `PredictedCost`, seconds (SC-30).
    #[serde(deserialize_with = "present")]
    pub predicted_t_tok_s: Option<f64>,
}

/// A plan epoch (RFC-0001 §5.4): a higher `term` dominates any `seq`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanEpoch {
    /// Coordinator term.
    pub term: u64,
    /// Sequence within the term.
    pub seq: u64,
}

/// One stage of the plan and where its weights sit.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageDigest {
    /// The node hosting the stage.
    pub node: String,
    /// First layer of the stage.
    pub layer_start: u16,
    /// One past the stage's last layer.
    pub layer_end: u16,
    /// Bytes assigned to each tier kind.
    pub tier_bytes: BTreeMap<TierKind, u64>,
}

/// The parity report this result is paired with (RFC-0061 §3.6, RFC-0060 §5.9).
///
/// It joins to the report by (commit, rig, path): the commit is `key.build.commit` and the rig
/// is `key.rig`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParityRef {
    /// BLAKE3 of the stored `ParityReport`.
    pub report: Blake3,
    /// The execution-path class the report covers (RFC-0060 §6.1).
    pub path: String,
    /// The report's verdict. With `fail` the result is `invalid: fidelity`: it never gates,
    /// publishes or becomes a baseline.
    pub verdict: ParityVerdict,
}

/// RFC-0060 §5.9's verdict, without the failure's detail (the report holds it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParityVerdict {
    /// Within tolerance.
    Pass,
    /// Outside tolerance on some metric and category.
    Fail,
    /// The reference itself was found defective.
    ReferenceDefect,
}

/// The rig's state, recorded by the lab agent before measurement (RFC-0061 §3.5).
///
/// The resolved config hash §3.5 also lists is [`BenchResult::config_hash`], stored once.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvCapture {
    /// The rig's inventory ID, for example `R-ADA` (RFC-0061 §6.2).
    pub rig_id: String,
    /// The §3.5 checks that apply to this scenario, with what each observed.
    pub checks: Vec<EnvCheck>,
    /// `null` when clean; otherwise why the run is `dirty`, for example `inventory`,
    /// `misconfigured` or a failed check. A dirty result never gates or publishes.
    #[serde(deserialize_with = "present")]
    pub dirty: Option<String>,
    /// GPU driver version; `null` on a rig without a discrete GPU driver.
    #[serde(deserialize_with = "present")]
    pub driver: Option<String>,
    /// Firmware versions by component, for example `vbios` or `nvme0`.
    pub firmware: BTreeMap<String, String>,
    /// OS kernel version.
    pub kernel: String,
    /// CPU microcode revision; `null` where the platform does not expose it.
    #[serde(deserialize_with = "present")]
    pub microcode: Option<String>,
    /// The GPU clock range observed during measurement; `null` without a GPU.
    #[serde(deserialize_with = "present")]
    pub gpu_clock_mhz: Option<ClockRange>,
}

/// One cleanliness check and its outcome.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvCheck {
    /// Which check.
    pub check: EnvCheckKind,
    /// Whether the required state held.
    pub passed: bool,
    /// What the agent observed, for the record.
    pub observed: String,
}

/// The checks of RFC-0061 §3.5, in its table's order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnvCheckKind {
    /// Mains; laptop rigs at ≥ 80% charge.
    PowerSource,
    /// The rig's recorded power plan or CPU governor.
    PowerPlan,
    /// No foreign process above 2% of one core over 10 s.
    OtherProcesses,
    /// No memory paged out; macOS memory pressure normal.
    MemoryPagedOut,
    /// GPU, CPU and NVMe temperatures at block start.
    Temperatures,
    /// Page cache dropped for cold-load scenarios.
    PageCache,
    /// No foreign traffic on the lab data-plane VLAN.
    Network,
    /// OS updates paused and indexing off.
    OsBackgroundWork,
}

/// An observed clock range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClockRange {
    /// Lowest observed clock, MHz.
    pub min_mhz: u32,
    /// Highest observed clock, MHz.
    pub max_mhz: u32,
}

/// The baseline engine's side of a comparison (RFC-0061 §7.6).
///
/// Records name the engine precisely so a third party can reproduce the run; plan documents
/// never do (D5).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaselineRun {
    /// The engine's name.
    pub engine: String,
    /// Its tagged release version.
    pub version: String,
    /// The release's commit, as the engine's repository spells it.
    pub commit: String,
    /// The exact command line, one argument per element.
    pub command: Vec<String>,
    /// The configuration files used; their contents go in the raw-data bundle (RFC-0061 §11.2).
    pub config_files: Vec<FileDigest>,
    /// The configuration sweep of RFC-0061 §7.3, at most 20 points; the best one was run.
    pub sweep: Vec<SweepPoint>,
    /// The model bytes the engine loaded.
    pub model: BaselineModel,
    /// The fidelity settings the engine ran with (RFC-0061 §7.4).
    pub fidelity_settings: BaselineFidelity,
    /// Measured fidelity, where the engine exposes per-position top-n log-probabilities.
    #[serde(deserialize_with = "present")]
    pub measured_fidelity: Option<MeasuredFidelity>,
    /// Raw timings, one per repetition.
    pub repetitions: Vec<BaselineRep>,
}

/// A file named in a record, with the digest of its bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileDigest {
    /// The file's path as the engine was given it.
    pub path: String,
    /// BLAKE3 of its bytes.
    pub blake3: Blake3,
}

/// One configuration of the baseline's sweep and the rate it reached.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SweepPoint {
    /// Option name to value, as the engine's documentation names them.
    pub options: BTreeMap<String, String>,
    /// Decode rate, tokens per second.
    pub decode_tps: f64,
}

/// The model the baseline loaded (RFC-0061 §7.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum BaselineModel {
    /// The scenario's own model file.
    Unchanged {
        /// BLAKE3 of the file.
        file: Blake3,
    },
    /// A conversion within 2% bytes per token, labelled `format-adjusted`.
    FormatAdjusted {
        /// The conversion command line, one argument per element.
        command: Vec<String>,
        /// BLAKE3 of the converted file.
        file: Blake3,
    },
}

/// The baseline's fidelity settings (RFC-0061 §7.4).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum BaselineFidelity {
    /// Its closest exact settings: `baseline-exact`.
    Exact,
    /// Its defaults: `baseline-default (approximate: <list>)`.
    Default {
        /// The approximations its defaults include; empty when they include none.
        approximations: Vec<String>,
    },
}

/// The baseline's fidelity on the RFC-0060 `smoke` corpus, against the same reference blocks.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredFidelity {
    /// The n of top-n; with n < K the KL is a looser bound.
    pub top_n: u16,
    /// Top-1 agreement, a fraction.
    pub top1_agreement: f64,
    /// Top-n coarse-grained KL.
    pub kl_top_n: f64,
}

/// One repetition of the baseline engine.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaselineRep {
    /// Steady-state TTFT, seconds.
    pub ttft_s: f64,
    /// TPOT, seconds.
    pub tpot_s: f64,
    /// Aggregate rate, tokens per second; throughput mode only.
    #[serde(deserialize_with = "present")]
    pub aggregate_tps: Option<f64>,
}

/// A gate's evaluation of this result (RFC-0061 §8).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateVerdict {
    /// Which gate.
    pub gate: GateKind,
    /// The build compared against; `null` for nightly trend detection, which compares series.
    #[serde(deserialize_with = "present")]
    pub against: Option<BuildId>,
    /// One entry per gated metric (RFC-0061 §8.1).
    pub metrics: Vec<MetricVerdict>,
    /// The overall outcome.
    pub outcome: GateOutcome,
    /// A maintainer's written reason for an override (`perf-regression-accepted`, §8.3) or a
    /// release waiver (§8.5).
    #[serde(deserialize_with = "present")]
    pub override_reason: Option<String>,
}

/// The gates of RFC-0061 §8.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateKind {
    /// The PR† gate (§8.3).
    Pr,
    /// Nightly trend detection (§8.4).
    Trend,
    /// A bisection step (§8.4).
    Bisection,
    /// The release gate (§8.5).
    Release,
}

/// One gated metric's evaluation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricVerdict {
    /// The metric's name, as RFC-0061 §8.1 lists it.
    pub metric: String,
    /// r̂: the mean block-pair log ratio, as a relative change (§3.4); for trend detection,
    /// the level shift.
    pub r_hat: f64,
    /// The band applied: `b_gate`, `b_trend` or `b_step` (§8.2, §8.4).
    pub band: f64,
    /// The sign-flip test's p; `null` where the gate runs no test.
    #[serde(deserialize_with = "present")]
    pub p_value: Option<f64>,
    /// This metric's outcome.
    pub outcome: GateOutcome,
}

/// A gate outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateOutcome {
    /// Within the band, or not significant.
    Pass,
    /// Beyond the band in the bad direction, and significant.
    Fail,
    /// Evaluated without the power to block, for example before 4 A/A runs exist (§8.2) or
    /// with the CV flag set (§8.1).
    ReportOnly,
}
