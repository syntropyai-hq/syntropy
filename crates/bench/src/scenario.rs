// SPDX-License-Identifier: MIT OR Apache-2.0

//! Scenario files: `bench/scenarios/<id>.toml` (RFC-0061 §4.1).
//!
//! A scenario file fixes every field of [`BenchKey`] except `rig` and `build`, lists the rigs it
//! runs on, and names a grid. The grid expands the scenario into [`Point`]s, one per combination
//! of prompt length, generated tokens, concurrency and speculative setting, each with the
//! cadences that run it. [`Scenario::bench_key`] completes a key at run time.
//!
//! IDs take one of two forms (RFC-0061 §4.1, revised 2026-10-22):
//! - **[`IdForm::Model`]**, a scenario that names a model: `<class>.<mode>.<model>.<quant>`
//!   with an optional fifth `<variant>` segment, whose fourth segment equals its `quant`;
//! - **[`IdForm::Other`]**, a scenario that runs no model, and RFC-0061 §9's microbenchmarks:
//!   2 to 6 segments.
//!
//! Every segment is lowercase ASCII letters, digits and `_`, and the first is one of
//! [`CLASSES`].
//!
//! A scenario names its model by its `fixtures.toml` checkpoint ID (RFC-0060 §12.3). The runner
//! resolves that ID to a [`ModelId`] when it loads the checkpoint, because a `ModelId` is a
//! content digest that only the census can compute.
//!
//! **The version rule.** Any change to a scenario file must raise its `scenario_version`
//! (RFC-0061 §4.1). [`check_versions`] compares each file with its copy at a base commit, and
//! fails closed when it can't find that commit.

use std::collections::BTreeSet;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;

use crate::schema::{
    BenchKey, BuildId, CacheStart, ExactnessPolicy, InterconnectId, ModelId, Nanos, Objective,
    PrefixSetting, QuantClass, RigFingerprint, ScenarioId, SpecSetting, ThermalProtocol,
    TopologyId,
};

/// Where scenario files live, relative to the repository root.
pub const SCENARIO_DIR: &str = "bench/scenarios";

/// The class prefixes a scenario ID may start with: RFC-0061 §4.2's scenario classes, §9's
/// microbenchmark prefixes, and the `feat` and `compare` benchmarks of §4.3.
pub const CLASSES: &[&str] = &[
    "agent",
    "api",
    "cache",
    "cluster",
    "compare",
    "decode",
    "exec",
    "feat",
    "io",
    "kern",
    "kv",
    "life",
    "net",
    "pipeline",
    "prefill",
    "probe",
    "sched",
    "solver",
    "spec",
    "store",
    "text",
    "throughput",
    "tier",
    "tiered",
];

/// The two forms of a scenario ID (RFC-0061 §4.1, revised 2026-10-22).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdForm {
    /// `<class>.<mode>.<model>.<quant>[.<variant>]`: a scenario that names a model.
    Model,
    /// 2 to 6 segments: a scenario that runs no model, or a microbenchmark (RFC-0061 §9).
    Other,
}

/// Checks `id` against `form`. The error says what is wrong.
pub fn check_id(id: &str, form: IdForm) -> Result<(), String> {
    let segments: Vec<&str> = id.split('.').collect();
    for segment in &segments {
        if let Err(why) = check_segment(segment) {
            return Err(format!("`{id}`: {why}"));
        }
    }
    if !CLASSES.contains(&segments[0]) {
        return Err(format!(
            "`{id}`: `{}` is not a scenario class; the classes are {}",
            segments[0],
            CLASSES.join(", ")
        ));
    }
    let (low, high, shape) = match form {
        IdForm::Model => (4, 5, "`<class>.<mode>.<model>.<quant>[.<variant>]`"),
        IdForm::Other => (2, 6, "2 to 6 segments"),
    };
    if segments.len() < low || segments.len() > high {
        return Err(format!(
            "`{id}` has {} segments; this form takes {shape}",
            segments.len()
        ));
    }
    Ok(())
}

fn check_segment(segment: &str) -> Result<(), String> {
    let mut chars = segment.chars();
    match chars.next() {
        None => Err(String::from("an empty segment")),
        Some(first) if !(first.is_ascii_lowercase() || first.is_ascii_digit()) => Err(format!(
            "segment `{segment}` must start with a lowercase letter or a digit"
        )),
        Some(_) => {
            if chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') {
                Ok(())
            } else {
                Err(format!(
                    "segment `{segment}` may hold only lowercase letters, digits and `_`"
                ))
            }
        }
    }
}

/// Checks a scenario-ID pattern, as `bench/sc-map.toml` uses them. A `*` segment matches
/// exactly one segment, or one or more segments when it is the last.
pub fn check_pattern(pattern: &str) -> Result<(), String> {
    let segments: Vec<&str> = pattern.split('.').collect();
    if segments.len() < 2 || segments.len() > 6 {
        return Err(format!("pattern `{pattern}` must have 2 to 6 segments"));
    }
    if !CLASSES.contains(&segments[0]) {
        return Err(format!(
            "pattern `{pattern}`: `{}` is not a scenario class",
            segments[0]
        ));
    }
    for segment in &segments[1..] {
        if *segment != "*"
            && let Err(why) = check_segment(segment)
        {
            return Err(format!("pattern `{pattern}`: {why}"));
        }
    }
    Ok(())
}

/// Whether `id` matches `pattern` ([`check_pattern`]'s rules).
pub fn pattern_matches(pattern: &str, id: &str) -> bool {
    let pattern: Vec<&str> = pattern.split('.').collect();
    let id: Vec<&str> = id.split('.').collect();
    for (i, segment) in pattern.iter().enumerate() {
        let last = i + 1 == pattern.len();
        if *segment == "*" && last {
            return id.len() > i;
        }
        match id.get(i) {
            Some(part) if *segment == "*" || segment == part => {}
            _ => return false,
        }
    }
    id.len() == pattern.len()
}

/// When a point of the grid runs (RFC-0061 §4.1's Cadence column).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cadence {
    /// The PR† timed blocks.
    Pr,
    /// The nightly run.
    Nightly,
    /// The release-candidate run.
    Rc,
}

/// One run of a scenario: the [`BenchKey`] fields its grid varies, and when it runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Point {
    pub prompt_tokens: u32,
    pub gen_tokens: u32,
    pub concurrency: u16,
    pub spec: SpecSetting,
    pub cadences: BTreeSet<Cadence>,
}

/// The standard grid of RFC-0061 §4.1.
///
/// - PR†: prompts of 128 and 2,048 tokens, 128 generated, one sequence, and the plan's default
///   speculative setting (`Auto`).
/// - Nightly and RC: prompts of 128, 2,048, 8,192 and 32,768 tokens, 512 generated, speculation
///   `Off` and `Auto`, and in throughput mode concurrency 1, 2, 4, 8, 16 and 32. §4.1 names RC
///   only for generated tokens and the sustained protocol, so RC runs the nightly points.
/// - A `Sustained` scenario runs nightly and at RC only, never on PR† (§4.1's Thermal row).
pub fn standard_grid(throughput: bool, sustained: bool) -> Vec<Point> {
    let mut points = Vec::new();
    if !sustained {
        for prompt_tokens in [128, 2_048] {
            points.push(Point {
                prompt_tokens,
                gen_tokens: 128,
                concurrency: 1,
                spec: SpecSetting::Auto,
                cadences: BTreeSet::from([Cadence::Pr]),
            });
        }
    }
    let concurrency: &[u16] = if throughput {
        &[1, 2, 4, 8, 16, 32]
    } else {
        &[1]
    };
    for prompt_tokens in [128, 2_048, 8_192, 32_768] {
        for &concurrency in concurrency {
            for spec in [SpecSetting::Off, SpecSetting::Auto] {
                points.push(Point {
                    prompt_tokens,
                    gen_tokens: 512,
                    concurrency,
                    spec,
                    cadences: BTreeSet::from([Cadence::Nightly, Cadence::Rc]),
                });
            }
        }
    }
    points
}

/// The model a scenario runs, by its `fixtures.toml` checkpoint ID (RFC-0060 §12.3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelRef {
    /// The checkpoint ID in `fixtures.toml`.
    pub fixture: String,
    /// The quantization class, as the scenario ID spells it.
    pub quant: QuantClass,
}

/// A checked scenario file.
#[derive(Clone, Debug, PartialEq)]
pub struct Scenario {
    pub id: ScenarioId,
    pub version: u16,
    /// `None` for a scenario that runs no model ([`IdForm::Other`]).
    pub model: Option<ModelRef>,
    /// Rig IDs from RFC-0061 §6.2.
    pub rigs: Vec<String>,
    pub topology: TopologyId,
    pub interconnect: InterconnectId,
    pub objective: Objective,
    pub fidelity: ExactnessPolicy,
    pub prefix: PrefixSetting,
    pub expert_cache: CacheStart,
    pub thermal: ThermalProtocol,
    /// The grid's points, in grid order.
    pub points: Vec<Point>,
}

impl Scenario {
    /// Parses and checks a scenario file. Returns every problem found, not only the first.
    pub fn parse(text: &str) -> Result<Scenario, Vec<String>> {
        let raw: RawScenario = toml::from_str(text).map_err(|e| vec![e.to_string()])?;
        let mut problems = Vec::new();

        let model = match (raw.model, raw.quant) {
            (Some(fixture), Some(quant)) => Some(ModelRef {
                fixture,
                quant: QuantClass(quant),
            }),
            (None, None) => None,
            (Some(_), None) => {
                problems.push(String::from("`model` is set but `quant` is not"));
                None
            }
            (None, Some(_)) => {
                problems.push(String::from("`quant` is set but `model` is not"));
                None
            }
        };
        let form = if model.is_some() {
            IdForm::Model
        } else {
            IdForm::Other
        };
        if let Err(why) = check_id(&raw.scenario, form) {
            problems.push(why);
        } else if let Some(model) = &model {
            let fourth = raw.scenario.split('.').nth(3).unwrap_or_default();
            if fourth != model.quant.0 {
                problems.push(format!(
                    "the ID's quant segment is `{fourth}` but `quant` is `{}`",
                    model.quant.0
                ));
            }
        }
        if let Some(model) = &model
            && (model.fixture.is_empty() || model.fixture.chars().any(char::is_whitespace))
        {
            problems.push(format!(
                "`model` must be a fixtures.toml checkpoint ID, not `{}`",
                model.fixture
            ));
        }
        if raw.scenario_version == 0 {
            problems.push(String::from("`scenario_version` starts at 1"));
        }
        if raw.rigs.is_empty() {
            problems.push(String::from("`rigs` lists no rig"));
        }
        let mut seen = BTreeSet::new();
        for rig in &raw.rigs {
            if !seen.insert(rig) {
                problems.push(format!("rig `{rig}` is listed twice"));
            }
        }
        for (field, value) in [
            ("topology", &raw.topology),
            ("interconnect", &raw.interconnect),
        ] {
            if value.is_empty() {
                problems.push(format!("`{field}` is empty"));
            }
        }

        let objective = match raw.objective {
            RawObjective::Latency => Objective::Latency,
            RawObjective::Throughput {
                per_sequence_floor_tps,
                target_concurrency,
                ttft_ceiling_ns,
            } => {
                if !(per_sequence_floor_tps.is_finite() && per_sequence_floor_tps > 0.0) {
                    problems.push(String::from(
                        "`per_sequence_floor_tps` must be a positive number",
                    ));
                }
                Objective::Throughput {
                    per_sequence_floor_tps,
                    target_concurrency,
                    ttft_ceiling: ttft_ceiling_ns.map(Nanos),
                }
            }
        };
        let thermal = raw.thermal.unwrap_or(ThermalProtocol::Cold);
        let sustained = matches!(thermal, ThermalProtocol::Sustained { .. });

        let points = match raw.grid {
            RawGrid::Named(name) if name == "standard" => {
                standard_grid(matches!(objective, Objective::Throughput { .. }), sustained)
            }
            RawGrid::Named(name) => {
                problems.push(format!(
                    "`grid = \"{name}\"`: the only named grid is \"standard\"; give axes as a \
                     [grid] table instead"
                ));
                Vec::new()
            }
            RawGrid::Axes(axes) => axes.points(&mut problems),
        };
        if points.is_empty() && problems.is_empty() {
            problems.push(String::from("the grid has no points"));
        }

        if !problems.is_empty() {
            return Err(problems);
        }
        Ok(Scenario {
            id: ScenarioId(raw.scenario),
            version: raw.scenario_version,
            model,
            rigs: raw.rigs,
            topology: TopologyId(raw.topology),
            interconnect: InterconnectId(raw.interconnect),
            objective,
            fidelity: ExactnessPolicy::Exact,
            prefix: raw.prefix.unwrap_or(PrefixSetting::Cold),
            expert_cache: raw.expert_cache.unwrap_or(CacheStart::Learned),
            thermal,
            points,
        })
    }

    /// The full [`BenchKey`] of one point, once the runner knows the model's [`ModelId`], the
    /// rig's fingerprint and the build.
    ///
    /// A scenario that runs no model has no `BenchKey` yet, because `BenchKey::model` and
    /// `BenchKey::quant` are required fields (tracked as a gap; RFC-0061 §4.1).
    pub fn bench_key(
        &self,
        point: &Point,
        model: ModelId,
        rig: RigFingerprint,
        build: BuildId,
    ) -> Result<BenchKey, String> {
        let Some(model_ref) = &self.model else {
            return Err(format!(
                "`{}` runs no model, and BenchKey needs a model and a quant",
                self.id.0
            ));
        };
        Ok(BenchKey {
            scenario: self.id.clone(),
            scenario_version: self.version,
            model,
            quant: model_ref.quant.clone(),
            prompt_tokens: point.prompt_tokens,
            gen_tokens: point.gen_tokens,
            concurrency: point.concurrency,
            objective: self.objective,
            fidelity: self.fidelity.clone(),
            spec: point.spec,
            prefix: self.prefix,
            topology: self.topology.clone(),
            interconnect: self.interconnect.clone(),
            rig,
            thermal: self.thermal,
            expert_cache: self.expert_cache,
            build,
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawScenario {
    scenario: String,
    scenario_version: u16,
    model: Option<String>,
    quant: Option<String>,
    rigs: Vec<String>,
    topology: String,
    interconnect: String,
    objective: RawObjective,
    // Only `exact` today: `ExactnessPolicy::Approximate` needs explicit nulls, which TOML can't
    // write.
    #[allow(dead_code)]
    fidelity: Option<RawFidelity>,
    prefix: Option<PrefixSetting>,
    expert_cache: Option<CacheStart>,
    thermal: Option<ThermalProtocol>,
    grid: RawGrid,
}

/// [`Objective`] as a scenario file spells it: an absent key is `None`, since TOML has no null.
#[derive(Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum RawObjective {
    Latency,
    Throughput {
        per_sequence_floor_tps: f32,
        target_concurrency: Option<u16>,
        ttft_ceiling_ns: Option<u64>,
    },
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum RawFidelity {
    Exact,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawGrid {
    Named(String),
    Axes(RawAxes),
}

/// An explicit grid: every combination of the listed values, at the listed cadences.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAxes {
    prompt_tokens: Vec<u32>,
    gen_tokens: Vec<u32>,
    concurrency: Option<Vec<u16>>,
    spec: Option<Vec<SpecSetting>>,
    cadence: Vec<Cadence>,
}

impl RawAxes {
    fn points(self, problems: &mut Vec<String>) -> Vec<Point> {
        let concurrency = self.concurrency.unwrap_or_else(|| vec![1]);
        let spec = self.spec.unwrap_or_else(|| vec![SpecSetting::Auto]);
        let cadences: BTreeSet<Cadence> = self.cadence.iter().copied().collect();
        for (axis, empty) in [
            ("prompt_tokens", self.prompt_tokens.is_empty()),
            ("gen_tokens", self.gen_tokens.is_empty()),
            ("concurrency", concurrency.is_empty()),
            ("spec", spec.is_empty()),
            ("cadence", cadences.is_empty()),
        ] {
            if empty {
                problems.push(format!("grid axis `{axis}` is empty"));
            }
        }
        if self.prompt_tokens.contains(&0)
            || self.gen_tokens.contains(&0)
            || concurrency.contains(&0)
        {
            problems.push(String::from(
                "grid values for tokens and concurrency must be positive",
            ));
        }
        let mut points: Vec<Point> = Vec::new();
        for &prompt_tokens in &self.prompt_tokens {
            for &gen_tokens in &self.gen_tokens {
                for &concurrency in &concurrency {
                    for &spec in &spec {
                        let point = Point {
                            prompt_tokens,
                            gen_tokens,
                            concurrency,
                            spec,
                            cadences: cadences.clone(),
                        };
                        if points.contains(&point) {
                            problems.push(format!(
                                "grid point (prompt {prompt_tokens}, generated {gen_tokens}, \
                                 concurrency {concurrency}) appears twice"
                            ));
                        } else {
                            points.push(point);
                        }
                    }
                }
            }
        }
        points
    }
}

/// A problem found in one file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Problem {
    pub path: PathBuf,
    pub message: String,
}

impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path.display(), self.message)
    }
}

/// The rig IDs RFC-0061 §6.2 defines: its bold names that start with `R-`, and `S2` for the
/// S2 cluster.
pub fn rig_ids(rfc_0061: &str) -> BTreeSet<String> {
    let section = rfc_0061
        .split_once("#### 6.2")
        .map(|(_, rest)| rest)
        .and_then(|rest| rest.split_once("#### 6.3").map(|(body, _)| body))
        .unwrap_or_default();
    let mut ids = BTreeSet::new();
    let mut parts = section.split("**");
    parts.next();
    while let Some(bold) = parts.next() {
        let first = bold.split_whitespace().next().unwrap_or_default();
        let is_rig = first.starts_with("R-")
            || (first.len() > 1
                && first.starts_with('S')
                && first[1..].chars().all(|c| c.is_ascii_digit()));
        if is_rig {
            ids.insert(first.to_string());
        }
        parts.next();
    }
    ids
}

/// Loads and checks every scenario file under `root/bench/scenarios/`. A missing directory
/// holds no scenarios.
pub fn load(root: &Path, known_rigs: &BTreeSet<String>) -> (Vec<Scenario>, Vec<Problem>) {
    let dir = root.join(SCENARIO_DIR);
    let mut scenarios: Vec<Scenario> = Vec::new();
    let mut problems = Vec::new();
    let mut paths = match fs::read_dir(&dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .collect::<Vec<_>>(),
        Err(_) => return (scenarios, problems),
    };
    paths.sort();
    for path in paths {
        let shown = Path::new(SCENARIO_DIR).join(path.file_name().unwrap_or_default());
        let problem = |message: String| Problem {
            path: shown.clone(),
            message,
        };
        if !path.is_file() || path.extension().and_then(|e| e.to_str()) != Some("toml") {
            problems.push(problem(String::from(
                "only `<scenario id>.toml` files belong in bench/scenarios/",
            )));
            continue;
        }
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) => {
                problems.push(problem(format!("cannot read: {e}")));
                continue;
            }
        };
        let scenario = match Scenario::parse(&text) {
            Ok(scenario) => scenario,
            Err(found) => {
                problems.extend(found.into_iter().map(&problem));
                continue;
            }
        };
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        if stem != scenario.id.0 {
            problems.push(problem(format!(
                "the file name must be `{}.toml`, the scenario's ID",
                scenario.id.0
            )));
        }
        for rig in &scenario.rigs {
            if !known_rigs.contains(rig) {
                problems.push(problem(format!(
                    "rig `{rig}` is not a rig of RFC-0061 §6.2"
                )));
            }
        }
        if scenarios.iter().any(|s| s.id == scenario.id) {
            problems.push(problem(format!(
                "scenario `{}` is defined twice",
                scenario.id.0
            )));
        }
        scenarios.push(scenario);
    }
    (scenarios, problems)
}

/// Checks that every scenario file that differs from its copy at the base commit has a higher
/// `scenario_version` there (RFC-0061 §4.1). A new file needs no bump.
///
/// `base` names the base commit. Without it, the base is `git merge-base HEAD origin/main`. The
/// check fails closed: an `Err` means the base could not be found, and the message says how to
/// fix that.
pub fn check_versions(root: &Path, base: Option<&str>) -> Result<Vec<Problem>, String> {
    let sha = resolve_base(root, base)?;
    let listing = git(
        root,
        &[
            "ls-tree",
            "-r",
            "-z",
            "--name-only",
            &sha,
            "--",
            SCENARIO_DIR,
        ],
    )?;
    let at_base: BTreeSet<String> = String::from_utf8_lossy(&listing)
        .split('\0')
        .filter(|p| !p.is_empty())
        .map(str::to_string)
        .collect();
    let mut problems = Vec::new();
    let dir = root.join(SCENARIO_DIR);
    let Ok(entries) = fs::read_dir(&dir) else {
        return Ok(problems);
    };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();
    for path in paths {
        if path.extension().and_then(|e| e.to_str()) != Some("toml") {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        let rel = format!("{SCENARIO_DIR}/{name}");
        if !at_base.contains(&rel) {
            continue;
        }
        let Ok(now) = fs::read(&path) else {
            continue;
        };
        let then = git(root, &["show", &format!("{sha}:{rel}")])?;
        if now == then {
            continue;
        }
        let problem = |message: String| Problem {
            path: PathBuf::from(&rel),
            message,
        };
        match (version_of(&then), version_of(&now)) {
            (Some(old), Some(new)) if new > old => {}
            (Some(old), Some(new)) => problems.push(problem(format!(
                "changed since the base commit {} without a `scenario_version` bump: it was {old} \
                 there and is {new} now; raise it to {}",
                short(&sha),
                old + 1
            ))),
            (None, _) => problems.push(problem(format!(
                "has no readable `scenario_version` at the base commit {}",
                short(&sha)
            ))),
            (_, None) => problems.push(problem(String::from("has no readable `scenario_version`"))),
        }
    }
    Ok(problems)
}

fn version_of(bytes: &[u8]) -> Option<i64> {
    let text = std::str::from_utf8(bytes).ok()?;
    let table: toml::Table = toml::from_str(text).ok()?;
    table.get("scenario_version")?.as_integer()
}

fn resolve_base(root: &Path, base: Option<&str>) -> Result<String, String> {
    let fix = "Fetch the base with `git fetch origin main`, or pass `--base <ref>`. A shallow CI \
               checkout must fetch the base ref too (fetch-depth 0, or an explicit fetch).";
    let found = match base {
        Some(reference) => git(
            root,
            &[
                "rev-parse",
                "--verify",
                "--quiet",
                &format!("{reference}^{{commit}}"),
            ],
        )
        .map_err(|_| format!("the base `{reference}` is not a commit here")),
        None => git(root, &["merge-base", "HEAD", "origin/main"])
            .map_err(|_| String::from("`git merge-base HEAD origin/main` found no base")),
    };
    match found {
        Ok(out) => {
            let sha = String::from_utf8_lossy(&out).trim().to_string();
            if sha.is_empty() {
                Err(format!(
                    "the scenario_version check has no base commit. {fix}"
                ))
            } else {
                Ok(sha)
            }
        }
        Err(why) => Err(format!(
            "the scenario_version check has no base commit: {why}. {fix}"
        )),
    }
}

fn git(root: &Path, args: &[&str]) -> Result<Vec<u8>, String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|e| format!("could not run git: {e}"))?;
    if output.status.success() {
        Ok(output.stdout)
    } else {
        Err(format!(
            "`git {}` failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

fn short(sha: &str) -> &str {
    sha.get(..12).unwrap_or(sha)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL: &str = r#"
scenario = "decode.single.llama8b.q4km"
scenario_version = 1
model = "llama-3.1-8b"
quant = "q4km"
rigs = ["R-ADA"]
topology = "single"
interconnect = "none"
objective = "latency"
grid = "standard"
"#;

    fn with(line: &str, replacement: &str) -> String {
        assert!(MODEL.contains(line), "test fixture lacks `{line}`");
        MODEL.replacen(line, replacement, 1)
    }

    #[test]
    fn a_model_scenario_parses_with_its_defaults() {
        let s = Scenario::parse(MODEL).expect("valid");
        assert_eq!(s.id.0, "decode.single.llama8b.q4km");
        assert_eq!(s.version, 1);
        let model = s.model.expect("names a model");
        assert_eq!(model.fixture, "llama-3.1-8b");
        assert_eq!(model.quant.0, "q4km");
        assert_eq!(s.fidelity, ExactnessPolicy::Exact);
        assert_eq!(s.prefix, PrefixSetting::Cold);
        assert_eq!(s.expert_cache, CacheStart::Learned);
        assert_eq!(s.thermal, ThermalProtocol::Cold);
        assert_eq!(s.points, standard_grid(false, false));
    }

    #[test]
    fn ids_follow_the_two_forms() {
        for ok in [
            "decode.single.llama70b.q4km",
            "decode.single.llama70b.q4km.cap64g",
            "tiered.single.dsv3.4bit",
        ] {
            assert_eq!(check_id(ok, IdForm::Model), Ok(()), "{ok}");
        }
        for ok in [
            "life.probe",
            "cluster.rejoin",
            "kern.gemv.dense.cuda.q4k",
            "net.hop.p99_under_bulk",
        ] {
            assert_eq!(check_id(ok, IdForm::Other), Ok(()), "{ok}");
        }
        for (bad, form) in [
            ("decode.single.llama70b", IdForm::Model),
            ("decode.single.llama70b.q4km.a.b", IdForm::Model),
            ("life", IdForm::Other),
            ("a.b.c.d.e.f.g", IdForm::Other),
            ("Decode.single.x.q4", IdForm::Model),
            ("bogus.single.x.q4", IdForm::Model),
            ("decode..x.q4", IdForm::Model),
            ("decode.single.x-y.q4", IdForm::Model),
            ("decode.single._x.q4", IdForm::Model),
        ] {
            assert!(check_id(bad, form).is_err(), "{bad} passed");
        }
    }

    #[test]
    fn the_quant_segment_must_match_quant() {
        let text = with("quant = \"q4km\"", "quant = \"q8\"");
        let err = Scenario::parse(&text).unwrap_err();
        assert!(err.iter().any(|e| e.contains("quant segment")), "{err:?}");
    }

    #[test]
    fn model_and_quant_come_together() {
        let text = with("quant = \"q4km\"\n", "");
        let err = Scenario::parse(&text).unwrap_err();
        assert!(err.iter().any(|e| e.contains("`quant` is not")), "{err:?}");
    }

    #[test]
    fn a_scenario_without_a_model_takes_the_other_form() {
        let text = r#"
scenario = "life.probe"
scenario_version = 2
rigs = ["R-ADA", "R-BW"]
topology = "single"
interconnect = "none"
objective = "latency"
[grid]
prompt_tokens = [1]
gen_tokens = [1]
cadence = ["nightly"]
"#;
        let s = Scenario::parse(text).expect("valid");
        assert!(s.model.is_none());
        assert_eq!(s.points.len(), 1);
        assert_eq!(s.points[0].cadences, BTreeSet::from([Cadence::Nightly]));
        let key = s.bench_key(
            &s.points[0],
            ModelId(crate::schema::Blake3([0; 32])),
            RigFingerprint(crate::schema::Blake3([0; 32])),
            test_build(),
        );
        assert!(key.is_err(), "a model-less scenario has no BenchKey yet");
    }

    #[test]
    fn version_zero_unknown_fields_and_empty_rigs_fail() {
        assert!(Scenario::parse(&with("scenario_version = 1", "scenario_version = 0")).is_err());
        assert!(Scenario::parse(&with("rigs = [\"R-ADA\"]", "rigs = []")).is_err());
        assert!(
            Scenario::parse(&with("rigs = [\"R-ADA\"]", "rigs = [\"R-ADA\", \"R-ADA\"]")).is_err()
        );
        let err = Scenario::parse(&format!("{MODEL}colour = \"blue\"\n")).unwrap_err();
        assert!(err[0].contains("colour"), "{err:?}");
        assert!(Scenario::parse(&with("grid = \"standard\"", "grid = \"wide\"")).is_err());
    }

    #[test]
    fn the_standard_grid_follows_rfc_0061_section_4_1() {
        let latency = standard_grid(false, false);
        let pr: Vec<&Point> = latency
            .iter()
            .filter(|p| p.cadences.contains(&Cadence::Pr))
            .collect();
        assert_eq!(pr.len(), 2);
        assert!(
            pr.iter()
                .all(|p| p.gen_tokens == 128 && p.spec == SpecSetting::Auto)
        );
        assert_eq!(
            pr.iter().map(|p| p.prompt_tokens).collect::<Vec<_>>(),
            [128, 2_048]
        );
        let nightly: Vec<&Point> = latency
            .iter()
            .filter(|p| p.cadences.contains(&Cadence::Nightly))
            .collect();
        assert_eq!(nightly.len(), 4 * 2);
        assert!(nightly.iter().all(|p| p.gen_tokens == 512
            && p.concurrency == 1
            && p.cadences.contains(&Cadence::Rc)));

        let throughput = standard_grid(true, false);
        let levels: BTreeSet<u16> = throughput.iter().map(|p| p.concurrency).collect();
        assert_eq!(levels, BTreeSet::from([1, 2, 4, 8, 16, 32]));
        assert_eq!(throughput.len(), 2 + 4 * 6 * 2);

        let sustained = standard_grid(false, true);
        assert!(sustained.iter().all(|p| !p.cadences.contains(&Cadence::Pr)));
        assert_eq!(sustained.len(), 8);
    }

    #[test]
    fn a_sustained_scenario_drops_the_pr_points() {
        let text = format!("{MODEL}thermal = {{ sustained = {{ preload_s = 600 }} }}\n");
        let s = Scenario::parse(&text).expect("valid");
        assert_eq!(s.thermal, ThermalProtocol::Sustained { preload_s: 600 });
        assert_eq!(s.points, standard_grid(false, true));
    }

    #[test]
    fn a_throughput_objective_reads_without_nulls() {
        let text = with(
            "objective = \"latency\"",
            "[objective.throughput]\nper_sequence_floor_tps = 5.0",
        );
        // `grid` must come before the table that `[objective.throughput]` opens.
        let text = text.replace("grid = \"standard\"\n", "");
        let text = text.replace("topology", "grid = \"standard\"\ntopology");
        let s = Scenario::parse(&text).expect("valid");
        assert_eq!(
            s.objective,
            Objective::Throughput {
                per_sequence_floor_tps: 5.0,
                target_concurrency: None,
                ttft_ceiling: None,
            }
        );
        assert_eq!(s.points, standard_grid(true, false));
    }

    #[test]
    fn an_explicit_grid_rejects_duplicates_and_zeros() {
        let base = with(
            "grid = \"standard\"",
            "[grid]\nprompt_tokens = [128, 128]\ngen_tokens = [128]\ncadence = [\"pr\"]",
        );
        let err = Scenario::parse(&base).unwrap_err();
        assert!(err.iter().any(|e| e.contains("appears twice")), "{err:?}");
        let zero = with(
            "grid = \"standard\"",
            "[grid]\nprompt_tokens = [0]\ngen_tokens = [128]\ncadence = [\"pr\"]",
        );
        assert!(Scenario::parse(&zero).is_err());
    }

    #[test]
    fn bench_key_fills_every_field_but_rig_model_and_build_from_the_file() {
        let s = Scenario::parse(MODEL).expect("valid");
        let model = ModelId(crate::schema::Blake3([1; 32]));
        let rig = RigFingerprint(crate::schema::Blake3([2; 32]));
        let key = s
            .bench_key(&s.points[0], model, rig, test_build())
            .expect("model scenario");
        assert_eq!(key.scenario, s.id);
        assert_eq!(key.scenario_version, 1);
        assert_eq!(key.quant.0, "q4km");
        assert_eq!(key.prompt_tokens, 128);
        assert_eq!(key.gen_tokens, 128);
        assert_eq!(key.model, model);
        assert_eq!(key.rig, rig);
        assert_eq!(key.topology.0, "single");
    }

    #[test]
    fn patterns_match_whole_segments() {
        assert!(pattern_matches(
            "decode.single.*",
            "decode.single.llama8b.q4km"
        ));
        assert!(pattern_matches(
            "kern.gemv.dense.*",
            "kern.gemv.dense.cuda.q4k"
        ));
        assert!(!pattern_matches("decode.single.*", "decode.single"));
        assert!(pattern_matches("net.hop.*", "net.hop.quic"));
        assert!(pattern_matches("life.probe", "life.probe"));
        assert!(!pattern_matches("life.probe", "life.probe.x"));
        assert!(pattern_matches("spec.*.dense70b", "spec.off.dense70b"));
        assert!(!pattern_matches("spec.*.dense70b", "spec.off.other"));
        assert_eq!(check_pattern("decode.single.*"), Ok(()));
        assert!(check_pattern("*.single").is_err());
        assert!(check_pattern("decode").is_err());
        assert!(check_pattern("decode.Single").is_err());
    }

    #[test]
    fn rig_ids_come_from_section_6_2() {
        let rfc = "## x\n#### 6.2 Rigs\n| **R-ADA** (M0) | a |\n| **R-ULTRA-A** (M1), \
                   **R-ULTRA-B** (M2) | b |\n- **S2 cluster**: R-ADA + R-MBP\n- **R-TB5**: two\n\
                   Not **bold-word** here.\n#### 6.3 Inventory\n| **R-NOPE** |\n";
        let ids = rig_ids(rfc);
        assert_eq!(
            ids,
            ["R-ADA", "R-TB5", "R-ULTRA-A", "R-ULTRA-B", "S2"]
                .into_iter()
                .map(String::from)
                .collect()
        );
    }

    fn test_build() -> BuildId {
        let json = r#"{"version":"0.0.0","commit":"0000000000000000000000000000000000000000",
            "profile":"release","target":"x86_64-unknown-linux-gnu",
            "toolchain":"0000000000000000000000000000000000000000000000000000000000000000"}"#;
        serde_json::from_str(json).expect("test build id")
    }
}
