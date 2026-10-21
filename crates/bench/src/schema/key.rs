// SPDX-License-Identifier: MIT OR Apache-2.0

//! The benchmark key (RFC-0061 §2.1) and the comparability rule.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Deserializer, Serialize};

use super::digest::{Blake3, GitSha};

/// Everything that must match before two results may be compared (RFC-0061 §2.1).
///
/// Two results are comparable only when their keys are equal in every field except `build`
/// ([`BenchKey::comparable_with`]). The regression gates compare across builds, so `build` is
/// the one field allowed to differ.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchKey {
    /// For example `decode.single.dense70b.q4km` (RFC-0061 §4.1).
    pub scenario: ScenarioId,
    /// Bumped on any change to the scenario definition.
    pub scenario_version: u16,
    /// The model variant under test.
    pub model: ModelId,
    /// The quantization class, as the scenario ID spells it.
    pub quant: QuantClass,
    /// Prompt length in tokens; every prompt is cut to exactly this many (RFC-0061 §3.1).
    pub prompt_tokens: u32,
    /// Tokens generated per repetition, with EOS ignored (RFC-0061 §3.1).
    pub gen_tokens: u32,
    /// Sequences in flight.
    pub concurrency: u16,
    /// What the solver and scheduler optimise.
    pub objective: Objective,
    /// `Exact` unless the scenario measures a knob.
    pub fidelity: ExactnessPolicy,
    /// Speculative decoding setting.
    pub spec: SpecSetting,
    /// Whether a prefix-cache or session hit is expected.
    pub prefix: PrefixSetting,
    /// The nodes and their roles in the plan.
    pub topology: TopologyId,
    /// For example `2.5GbE-switch-A`, `TB5-RDMA-direct` or `none`.
    pub interconnect: InterconnectId,
    /// Changes on any hardware, firmware, BIOS-setting or OS-image change (RFC-0061 §6.3).
    pub rig: RigFingerprint,
    /// Thermal protocol (RFC-0061 §3.2).
    pub thermal: ThermalProtocol,
    /// The MoE expert cache's starting state (RFC-0061 §3.2).
    pub expert_cache: CacheStart,
    /// The build under test. The only field allowed to differ between comparable results.
    pub build: BuildId,
}

/// A field of [`BenchKey`], for reporting why two keys are not comparable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KeyField {
    /// [`BenchKey::scenario`]
    Scenario,
    /// [`BenchKey::scenario_version`]
    ScenarioVersion,
    /// [`BenchKey::model`]
    Model,
    /// [`BenchKey::quant`]
    Quant,
    /// [`BenchKey::prompt_tokens`]
    PromptTokens,
    /// [`BenchKey::gen_tokens`]
    GenTokens,
    /// [`BenchKey::concurrency`]
    Concurrency,
    /// [`BenchKey::objective`]
    Objective,
    /// [`BenchKey::fidelity`]
    Fidelity,
    /// [`BenchKey::spec`]
    Spec,
    /// [`BenchKey::prefix`]
    Prefix,
    /// [`BenchKey::topology`]
    Topology,
    /// [`BenchKey::interconnect`]
    Interconnect,
    /// [`BenchKey::rig`]
    Rig,
    /// [`BenchKey::thermal`]
    Thermal,
    /// [`BenchKey::expert_cache`]
    ExpertCache,
    /// [`BenchKey::build`]
    Build,
}

impl KeyField {
    /// Every field, in declaration order.
    pub const ALL: [KeyField; 17] = [
        KeyField::Scenario,
        KeyField::ScenarioVersion,
        KeyField::Model,
        KeyField::Quant,
        KeyField::PromptTokens,
        KeyField::GenTokens,
        KeyField::Concurrency,
        KeyField::Objective,
        KeyField::Fidelity,
        KeyField::Spec,
        KeyField::Prefix,
        KeyField::Topology,
        KeyField::Interconnect,
        KeyField::Rig,
        KeyField::Thermal,
        KeyField::ExpertCache,
        KeyField::Build,
    ];

    /// The field's name in JSON.
    pub fn name(self) -> &'static str {
        match self {
            KeyField::Scenario => "scenario",
            KeyField::ScenarioVersion => "scenario_version",
            KeyField::Model => "model",
            KeyField::Quant => "quant",
            KeyField::PromptTokens => "prompt_tokens",
            KeyField::GenTokens => "gen_tokens",
            KeyField::Concurrency => "concurrency",
            KeyField::Objective => "objective",
            KeyField::Fidelity => "fidelity",
            KeyField::Spec => "spec",
            KeyField::Prefix => "prefix",
            KeyField::Topology => "topology",
            KeyField::Interconnect => "interconnect",
            KeyField::Rig => "rig",
            KeyField::Thermal => "thermal",
            KeyField::ExpertCache => "expert_cache",
            KeyField::Build => "build",
        }
    }
}

impl fmt::Display for KeyField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl BenchKey {
    /// The fields in which `self` and `other` differ, in declaration order.
    pub fn differences(&self, other: &BenchKey) -> Vec<KeyField> {
        // Destructured so that a new key field fails to compile until it is listed here.
        let BenchKey {
            scenario,
            scenario_version,
            model,
            quant,
            prompt_tokens,
            gen_tokens,
            concurrency,
            objective,
            fidelity,
            spec,
            prefix,
            topology,
            interconnect,
            rig,
            thermal,
            expert_cache,
            build,
        } = self;
        let pairs = [
            (KeyField::Scenario, *scenario != other.scenario),
            (
                KeyField::ScenarioVersion,
                *scenario_version != other.scenario_version,
            ),
            (KeyField::Model, *model != other.model),
            (KeyField::Quant, *quant != other.quant),
            (
                KeyField::PromptTokens,
                *prompt_tokens != other.prompt_tokens,
            ),
            (KeyField::GenTokens, *gen_tokens != other.gen_tokens),
            (KeyField::Concurrency, *concurrency != other.concurrency),
            (KeyField::Objective, *objective != other.objective),
            (KeyField::Fidelity, *fidelity != other.fidelity),
            (KeyField::Spec, *spec != other.spec),
            (KeyField::Prefix, *prefix != other.prefix),
            (KeyField::Topology, *topology != other.topology),
            (KeyField::Interconnect, *interconnect != other.interconnect),
            (KeyField::Rig, *rig != other.rig),
            (KeyField::Thermal, *thermal != other.thermal),
            (KeyField::ExpertCache, *expert_cache != other.expert_cache),
            (KeyField::Build, *build != other.build),
        ];
        pairs
            .into_iter()
            .filter(|(_, differs)| *differs)
            .map(|(field, _)| field)
            .collect()
    }

    /// RFC-0061 §2.1: comparable only when the keys are equal in every field except `build`.
    pub fn comparable_with(&self, other: &BenchKey) -> bool {
        self.differences(other)
            .iter()
            .all(|field| *field == KeyField::Build)
    }
}

/// A scenario ID, `<class>.<mode>.<model>.<quant>[.<variant>]` (RFC-0061 §4.1).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ScenarioId(pub String);

/// The content identity of a model variant (RFC-0001 §5.1): a BLAKE3 digest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelId(pub Blake3);

/// A quantization class as scenario IDs spell it, for example `q4km` or `4bit`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct QuantClass(pub String);

/// Names the nodes and their roles in the plan, for example `single` or `pp2`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TopologyId(pub String);

/// Names the links between nodes, for example `2.5GbE-switch-A` or `none`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct InterconnectId(pub String);

/// A BLAKE3 digest of the rig's inventory record (RFC-0061 §2.1, §6.3).
///
/// It covers the whole record of `bench/lab/rigs.toml`: CPU model and microcode, memory modules,
/// GPU model, VBIOS and driver, NVMe model, firmware and slot, NICs, the BIOS settings that move
/// performance, each negotiated PCIe link, OS image and kernel, the power plan and the room
/// sensor. A new fingerprint starts a new baseline epoch for the rig, and no gate compares
/// results across epochs. [`crate::rigs::Rig::fingerprint`] computes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RigFingerprint(pub Blake3);

/// A duration in nanoseconds (RFC-0001 §5.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Nanos(pub u64);

/// What the solver and scheduler optimise, with the fields of its owner, RFC-0054 §4.1.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Objective {
    /// Minimise time per token, then TTFT.
    Latency,
    /// Maximise aggregate tokens per second subject to a per-sequence floor.
    Throughput {
        /// The floor each sequence's own decode rate must meet, in tokens per second.
        per_sequence_floor_tps: f32,
        /// Sequences the plan is sized for; `null` leaves it to the solver.
        #[serde(deserialize_with = "super::present")]
        target_concurrency: Option<u16>,
        /// Admission rejects a request above this predicted TTFT; `null` sets no ceiling.
        #[serde(deserialize_with = "super::present")]
        ttft_ceiling: Option<Nanos>,
    },
}

/// Exact by default (D3, ADR-0019); `Approximate` lists the knobs in use. The fields follow
/// the type's owner, RFC-0054 §4.2.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ExactnessPolicy {
    /// Output matches the reference within ADR-0019's tolerances.
    Exact,
    /// The named knobs may be used.
    Approximate {
        /// At least one knob, written in bit order (RFC-0054 §4.2). An empty set is not a second spelling of
        /// `Exact`: RFC-0054 §7 resolves it to `Exact`. A repeated knob fails validation too.
        #[serde(deserialize_with = "unique_knobs")]
        knobs: BTreeSet<ApproxKnob>,
        /// Each knob's parameters (boxed to keep the enum small; the JSON is unchanged).
        params: Box<ApproxParams>,
    },
}

/// Per-knob parameters (RFC-0054 §4.2). `null` means RFC-0042's default for that knob.
///
/// RFC-0042 has not defined the parameter types yet, so each value is stored in the form the
/// knob's `approximate.<knob>` configuration key takes (RFC-0054 §5.4).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApproxParams {
    /// `EXPERT_SKIP` parameters.
    #[serde(deserialize_with = "super::present")]
    pub expert_skip: Option<KnobParams>,
    /// `EXPERT_DEFER` parameters.
    #[serde(deserialize_with = "super::present")]
    pub expert_defer: Option<KnobParams>,
    /// The quant scheme of the lower-precision expert copies.
    #[serde(deserialize_with = "super::present")]
    pub miss_low_precision: Option<KnobParams>,
    /// `TOPK_REDUCE` parameters, for example `{"experts": 6}`.
    #[serde(deserialize_with = "super::present")]
    pub topk_reduce: Option<KnobParams>,
    /// The KV dtype.
    #[serde(deserialize_with = "super::present")]
    pub kv_lossy: Option<KnobParams>,
    /// The wire dtype.
    #[serde(deserialize_with = "super::present")]
    pub wire_compress: Option<KnobParams>,
    /// `APPROX_ATTENTION` parameters.
    #[serde(deserialize_with = "super::present")]
    pub approx_attention: Option<KnobParams>,
    /// `SPEC_LENIENT` parameters.
    #[serde(deserialize_with = "super::present")]
    pub spec_lenient: Option<KnobParams>,
}

/// A knob's parameters as its `approximate.<knob>` key gives them: a name (a dtype or quant
/// scheme) or a table of named values.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum KnobParams {
    /// A dtype or quant scheme, by name.
    Name(String),
    /// A parameter table.
    Table(BTreeMap<String, ParamValue>),
}

/// One value in a knob's parameter table.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParamValue {
    /// A boolean.
    Bool(bool),
    /// An integer literal.
    Integer(i64),
    /// A number with a fraction or exponent.
    Number(f64),
    /// A string.
    Text(String),
}

/// An approximation knob of ADR-0019 §6, named as in RFC-0054's `approximate.*` keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApproxKnob {
    /// `approximate.expert_skip`
    ExpertSkip,
    /// `approximate.expert_defer`
    ExpertDefer,
    /// `approximate.miss_low_precision`
    MissLowPrecision,
    /// `approximate.topk_reduce`
    TopkReduce,
    /// `approximate.kv_lossy`
    KvLossy,
    /// `approximate.wire_compress`
    WireCompress,
    /// `approximate.approx_attention`
    ApproxAttention,
    /// `approximate.spec_lenient`
    SpecLenient,
}

/// Reads a knob list. Refuses an empty list, and a repeated knob, which a set would otherwise
/// drop silently.
fn unique_knobs<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<BTreeSet<ApproxKnob>, D::Error> {
    let listed = Vec::<ApproxKnob>::deserialize(deserializer)?;
    if listed.is_empty() {
        return Err(serde::de::Error::custom(
            "an approximate policy names at least one knob",
        ));
    }
    let mut knobs = BTreeSet::new();
    for knob in listed {
        if !knobs.insert(knob) {
            return Err(serde::de::Error::custom(format_args!(
                "knob {knob:?} listed twice"
            )));
        }
    }
    Ok(knobs)
}

/// Speculative decoding in the scenario, as RFC-0054's `speculative.mode` sets it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum SpecSetting {
    /// No speculation.
    Off,
    /// The scheduler decides (the default).
    Auto,
    /// Speculation with a fixed configuration.
    Force(SpecConfig),
}

/// The `speculative.*` keys a forced setting fixes (RFC-0054 §5.4, RFC-0012).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecConfig {
    /// `speculative.source`, with `speculative.draft_model` for a draft model.
    pub source: DraftSource,
    /// `speculative.max_window`: γ_max.
    pub max_window: u16,
}

/// Where draft tokens come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum DraftSource {
    /// The model's own multi-token-prediction heads.
    Mtp,
    /// A trained draft head.
    DraftHead,
    /// A separate draft model.
    DraftModel {
        /// The draft model variant.
        model: ModelId,
    },
}

/// Prefix-cache state at the start of each repetition (RFC-0061 §3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixSetting {
    /// A per-repetition nonce keeps every repetition off the prefix cache and earlier sessions.
    Cold,
    /// A cache or session hit is expected (the agent pattern).
    Warm,
}

/// Thermal protocol (RFC-0061 §3.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ThermalProtocol {
    /// Measurement starts from the rig's idle temperatures.
    Cold,
    /// The workload runs for `preload_s` before measurement starts.
    Sustained {
        /// Preload length in seconds; RFC-0061 §3.2's default is 600.
        preload_s: u32,
    },
}

/// The MoE expert cache's starting state (RFC-0061 §3.2).
///
/// It applies to MoE scenarios. A dense scenario records `learned`, the default, and the field
/// then has no effect on the run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheStart {
    /// From the frequency profile of the prompt set's training split, as in production.
    Learned,
    /// From the uniform prior: what a user sees on first load.
    Cold,
}

/// Build identity: the fields of RFC-0001 §5.2's `BuildId` (RFC-0062 §8.1).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildId {
    /// Product version from `version.txt`, `MAJOR.MINOR.PATCH`.
    pub version: String,
    /// The commit built.
    pub commit: GitSha,
    /// The build profile (RFC-0062 §4.4).
    pub profile: BuildProfile,
    /// The target triple, for example `x86_64-unknown-linux-gnu`.
    pub target: String,
    /// BLAKE3 of the toolchain.
    pub toolchain: Blake3,
}

/// A build profile of RFC-0062 §4.4.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BuildProfile {
    /// `dev`
    Dev,
    /// `test-release`
    TestRelease,
    /// `chaos`
    Chaos,
    /// `pr`: PR builds and PR† jobs.
    Pr,
    /// `release`: everything shipped, nightly and RC benchmarks.
    Release,
}
