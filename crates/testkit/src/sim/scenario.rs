// SPDX-License-Identifier: MIT OR Apache-2.0

//! The scenario file (RFC-0060 §8.5): a TOML description of one simulated cluster, its workload,
//! its faults and what the run must show.
//!
//! [`Scenario::from_toml`] parses a file and checks what can be checked on its own, so a mistake
//! fails when the file loads, not partway through a run:
//! - every checker in `[expect]` is one RFC-0060 §8.6 names, or the `safety` group;
//! - the seed counts are within the caps of RFC-0060's "Simulator throughput": 64 per PR and
//!   4,096 nightly;
//! - node IDs are unique, and every link joins nodes the file declares;
//! - a workload length range runs low to high, and `resume_within` is a formula or a duration.
//!
//! What other tasks interpret stays as text here: a fault's `at`, `action` and `target`
//! (QUAL-E9-T03), the model reference, and a link's preset name, which the fabric's presets
//! resolve when the cluster model builds the links (QUAL-E8-T02). Device and drive names are
//! checked against [`presets`](super::presets).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::Deserialize;
use syntropy_config::Duration;

use super::presets::{self, PresetKind};

/// The most seeds a PR run takes per scenario (RFC-0060 "Simulator throughput").
pub const PR_SEED_CAP: u32 = 64;

/// The most seeds a nightly run takes per scenario (RFC-0060 "Simulator throughput").
pub const NIGHTLY_SEED_CAP: u32 = 4_096;

/// The checkers of RFC-0060 §8.6, by the names a scenario's `[expect]` uses.
pub const CHECKERS: [&str; 15] = [
    "single_coordinator",
    "log_prefix",
    "epoch_monotonic",
    "capacity",
    "single_owner",
    "planes",
    "exactly_once_tokens",
    "no_approx_in_exact",
    "numeric_oracle",
    "sc25_join_detect",
    "sc27_coord_change",
    "sc28_resume",
    "predicted_vs_simulated",
    "no_flap",
    "feature_level_gate",
];

/// The checkers `safety` stands for: ADR-0600 decision 6's safety checkers, run after every
/// event.
pub const SAFETY: [&str; 8] = [
    "single_coordinator",
    "log_prefix",
    "epoch_monotonic",
    "capacity",
    "single_owner",
    "planes",
    "exactly_once_tokens",
    "no_approx_in_exact",
];

/// One scenario file (RFC-0060 §8.5).
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scenario {
    pub name: String,
    /// `"digest"`, or `"numeric"` with a nano or micro fixture (RFC-0060 §8.4).
    pub numeric: NumericMode,
    /// The model, such as `fixture:deepseek-v3-class/full-shape`.
    pub model: String,
    #[serde(default)]
    pub seeds: Seeds,
    pub virtual_duration: Duration,
    pub nodes: Vec<NodeSpec>,
    #[serde(default)]
    pub links: Vec<LinkSpec>,
    pub workload: Workload,
    #[serde(default)]
    pub faults: Vec<FaultSpec>,
    pub expect: Expect,
}

/// How the fake kernel provider computes (RFC-0060 §8.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NumericMode {
    Digest,
    Numeric,
}

/// Seeds per cadence (RFC-0060 "Simulator throughput"): derived from the measured cost per seed,
/// recorded in the file by CI, and capped at [`PR_SEED_CAP`] and [`NIGHTLY_SEED_CAP`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Seeds {
    pub pr_cap: u32,
    pub nightly_cap: u32,
}

/// The caps themselves, until CI records derived counts.
impl Default for Seeds {
    fn default() -> Seeds {
        Seeds {
            pr_cap: PR_SEED_CAP,
            nightly_cap: NIGHTLY_SEED_CAP,
        }
    }
}

/// One simulated node: its devices and its drive, by preset name.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeSpec {
    pub id: String,
    /// Device presets, such as `gpu:ada-24g`.
    pub devices: Vec<String>,
    /// The drive preset, such as `gen4-990pro-class`.
    #[serde(default)]
    pub nvme: Option<String>,
}

/// Links between every pair of the named nodes, both ways, from one fabric preset.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinkSpec {
    pub between: Vec<String>,
    /// A fabric preset name (RFC-0060 §8.3), such as `lan_2_5g`.
    pub preset: String,
}

/// The client workload (RFC-0060 §8.4, `FakeClient`).
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Workload {
    pub kind: WorkloadKind,
    /// Concurrent request streams.
    pub streams: u32,
    /// Prompt lengths, in tokens.
    pub prompt: Lengths,
    /// Generation lengths, in tokens.
    pub r#gen: Lengths,
}

/// How requests arrive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkloadKind {
    /// Each stream sends its next request when the last one ends.
    ClosedLoop,
    /// Requests arrive as a Poisson process.
    Poisson,
}

/// A token count: fixed, or drawn from a distribution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum Lengths {
    Fixed(u32),
    Drawn(LengthDist),
}

/// A distribution of token counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(tag = "dist", rename_all = "snake_case", deny_unknown_fields)]
pub enum LengthDist {
    /// Uniform over `lo..=hi`.
    Uniform { lo: u32, hi: u32 },
}

/// One scripted fault (RFC-0060 §8.5, §9.1). The fault actions interpret `at`, `action`,
/// `target` and any other key (QUAL-E9-T03).
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct FaultSpec {
    /// When: `t+30s`, or a condition such as `after:decode_tokens>=64`.
    pub at: String,
    pub action: String,
    #[serde(default)]
    pub target: Option<String>,
    /// The action's other parameters, such as a link flap's `down`.
    #[serde(flatten)]
    pub params: BTreeMap<String, toml::Value>,
}

/// What the run must show (RFC-0060 §8.5, §8.6).
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Expect {
    /// Checker names from [`CHECKERS`], or `safety` for the [`SAFETY`] group.
    pub checkers: Vec<String>,
    /// The bound on a resumed sequence's delay: `formula:<name>`, such as `formula:sc28`, or a
    /// duration.
    #[serde(default)]
    pub resume_within: Option<String>,
}

impl Expect {
    /// The checkers to run, `safety` expanded, each once, in [`CHECKERS`] order.
    pub fn resolved_checkers(&self) -> Vec<&'static str> {
        let named: BTreeSet<&str> = self
            .checkers
            .iter()
            .flat_map(|name| {
                if name == "safety" {
                    SAFETY.to_vec()
                } else {
                    vec![name.as_str()]
                }
            })
            .collect();
        CHECKERS
            .into_iter()
            .filter(|checker| named.contains(checker))
            .collect()
    }
}

/// Why a scenario file did not load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScenarioError {
    /// Not TOML, or not a scenario's shape: the parser's message, with the line.
    Parse(String),
    /// A checker RFC-0060 §8.6 does not name.
    UnknownChecker(String),
    /// A seed count past its cap.
    TooManySeeds {
        cadence: &'static str,
        count: u32,
        cap: u32,
    },
    /// Two nodes with one ID.
    DuplicateNode(String),
    /// A link names a node the file does not declare.
    UnknownNode(String),
    /// A link joins fewer than two nodes.
    LinkTooShort,
    /// A length range whose low end is above its high end.
    EmptyRange { lo: u32, hi: u32 },
    /// No request streams.
    NoStreams,
    /// `resume_within` is neither `formula:<name>` nor a duration.
    BadResumeWithin(String),
    /// A device or drive name with no preset.
    UnknownPreset(String),
    /// A drive preset named as a device, or a device preset as a drive.
    WrongPresetKind {
        name: String,
        expected: &'static str,
    },
}

impl fmt::Display for ScenarioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScenarioError::Parse(message) => write!(f, "not a scenario file: {message}"),
            ScenarioError::UnknownChecker(name) => write!(
                f,
                "`{name}` is not a checker; RFC-0060 §8.6 names {}, and `safety`",
                CHECKERS.join(", ")
            ),
            ScenarioError::TooManySeeds {
                cadence,
                count,
                cap,
            } => write!(f, "{count} {cadence} seeds is over the cap of {cap}"),
            ScenarioError::DuplicateNode(id) => write!(f, "two nodes are called `{id}`"),
            ScenarioError::UnknownNode(id) => write!(f, "a link names `{id}`, which is no node"),
            ScenarioError::LinkTooShort => write!(f, "a link joins fewer than two nodes"),
            ScenarioError::EmptyRange { lo, hi } => {
                write!(f, "a length range runs from {lo} down to {hi}")
            }
            ScenarioError::NoStreams => write!(f, "the workload has no streams"),
            ScenarioError::BadResumeWithin(text) => write!(
                f,
                "`resume_within = \"{text}\"` is neither `formula:<name>` nor a duration"
            ),
            ScenarioError::UnknownPreset(name) => {
                write!(f, "`{name}` is no device or drive preset")
            }
            ScenarioError::WrongPresetKind { name, expected } => {
                write!(f, "`{name}` is a preset, but not {expected}")
            }
        }
    }
}

impl std::error::Error for ScenarioError {}

impl Scenario {
    /// Parses a scenario file and checks it (see the [module](self) docs).
    pub fn from_toml(text: &str) -> Result<Scenario, ScenarioError> {
        let scenario: Scenario =
            toml::from_str(text).map_err(|error| ScenarioError::Parse(error.to_string()))?;
        scenario.check()?;
        Ok(scenario)
    }

    fn check(&self) -> Result<(), ScenarioError> {
        for name in &self.expect.checkers {
            if name != "safety" && !CHECKERS.contains(&name.as_str()) {
                return Err(ScenarioError::UnknownChecker(name.clone()));
            }
        }
        for (cadence, count, cap) in [
            ("PR", self.seeds.pr_cap, PR_SEED_CAP),
            ("nightly", self.seeds.nightly_cap, NIGHTLY_SEED_CAP),
        ] {
            if count > cap {
                return Err(ScenarioError::TooManySeeds {
                    cadence,
                    count,
                    cap,
                });
            }
        }
        let mut ids = BTreeSet::new();
        for node in &self.nodes {
            if !ids.insert(node.id.as_str()) {
                return Err(ScenarioError::DuplicateNode(node.id.clone()));
            }
        }
        let presets = presets::by_name();
        for node in &self.nodes {
            let named = node.devices.iter().map(|name| (name, false));
            for (name, drive) in named.chain(node.nvme.iter().map(|name| (name, true))) {
                match presets.get(name) {
                    None => return Err(ScenarioError::UnknownPreset(name.clone())),
                    Some(preset) if (preset.kind == PresetKind::Nvme) != drive => {
                        return Err(ScenarioError::WrongPresetKind {
                            name: name.clone(),
                            expected: if drive { "a drive" } else { "a device" },
                        });
                    }
                    Some(_) => {}
                }
            }
        }
        for link in &self.links {
            if link.between.len() < 2 {
                return Err(ScenarioError::LinkTooShort);
            }
            if let Some(unknown) = link.between.iter().find(|id| !ids.contains(id.as_str())) {
                return Err(ScenarioError::UnknownNode(unknown.clone()));
            }
        }
        if self.workload.streams == 0 {
            return Err(ScenarioError::NoStreams);
        }
        for lengths in [self.workload.prompt, self.workload.r#gen] {
            if let Lengths::Drawn(LengthDist::Uniform { lo, hi }) = lengths
                && lo > hi
            {
                return Err(ScenarioError::EmptyRange { lo, hi });
            }
        }
        if let Some(bound) = &self.expect.resume_within {
            let formula = bound
                .strip_prefix("formula:")
                .is_some_and(|name| !name.is_empty());
            if !formula && bound.parse::<Duration>().is_err() {
                return Err(ScenarioError::BadResumeWithin(bound.clone()));
            }
        }
        Ok(())
    }
}
