// SPDX-License-Identifier: MIT OR Apache-2.0

//! Device and drive presets for scenarios (RFC-0060 §8.5): the files in `testkit/sim/presets/`.
//!
//! A preset names a device or a drive and the figures the research notes give for it. Each figure
//! carries a label and its source:
//!
//! | Label | Meaning | The research note's tag | AGENTS.md's label |
//! |---|---|---|---|
//! | `published` | a primary-source spec, such as a datasheet; never Syntropy's own claim | `[verified: Sxx]` | none of the three: a vendor's figure, neither our target, our estimate nor a measurement |
//! | `measured` | a third-party measurement | `[measured: …]`, `[verified (user reports): …]` | `[measured]`, with its source |
//! | `estimate` | a derivation, given in the source | `[estimate]` | `[estimate]` |
//! | `target` | a value a scenario chooses | — | `[target]` |
//!
//! A figure the research does not give is `unknown`, with where the research leaves it open. A
//! preset never fills one in.
//!
//! **Each kind has a fixed set of figure keys** ([`Preset::required_keys`]), and every preset of
//! the kind has each of them, known or `unknown`, so a figure is never absent by accident:
//! - a device (`gpu`, `cpu`): `memory`, `f_eff`, `t_fixed`, `sigma_dev`, and RFC-0060 §8.4's
//!   `W_eff(device, tier, kernel class)` as `w_eff.<tier>.<kernel class>`. The tiers are the ones
//!   the device's kernels read (RFC-0010's M1 probe): a discrete GPU's `vram` and `host_pinned`,
//!   a unified GPU's `unified`, a CPU's `host_pinned` and `host_pageable`. The kernel classes are
//!   the two M1 measures W_eff for, `dense_gemv` and `expert_gemv`. A discrete GPU also has
//!   `host_link_pinned_h2d`;
//! - a drive (`nvme`): `seq_read`, `sustained_read`, `throttled_read`, `heating_time`,
//!   `qd1_4k_read_iops`, `qd32_4k_read_iops` and `read_latency`.
//!
//! **A figure is read through [`Preset::figure`]**, which refuses one that is `unknown` or
//! absent, so no fake can take a figure the research leaves open for a zero or a default. A
//! scenario supplies such a figure itself, labelled, in its node's `set`
//! ([`NodeSpec::figure`](super::scenario::NodeSpec::figure)).
//!
//! Other figures, such as peak rates, may be added beside them. A `w_eff.*` key outside the set
//! is refused, so a misspelt tier or class cannot pass for a figure. `cargo xtask ci lint`'s
//! `sim-presets` step checks every file the same way: its sources, its keys, and each figure's
//! label and source.
//!
//! The files are compiled in ([`ALL`]), so a scenario's device and drive names resolve without
//! reading the disk, and a test checks [`ALL`] against the directory.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::OnceLock;

use serde::Deserialize;

/// The preset files, by file name, as compiled in.
pub const ALL: [(&str, &str); 7] = [
    (
        "cpu.ddr5-2ch-128g.toml",
        include_str!("../../sim/presets/cpu.ddr5-2ch-128g.toml"),
    ),
    (
        "gpu.ada-24g.toml",
        include_str!("../../sim/presets/gpu.ada-24g.toml"),
    ),
    (
        "gpu.apple-max-128g-unified.toml",
        include_str!("../../sim/presets/gpu.apple-max-128g-unified.toml"),
    ),
    (
        "gpu.strix-halo-128g-unified.toml",
        include_str!("../../sim/presets/gpu.strix-halo-128g-unified.toml"),
    ),
    (
        "nvme.apple-internal.toml",
        include_str!("../../sim/presets/nvme.apple-internal.toml"),
    ),
    (
        "nvme.gen4-990pro-class.toml",
        include_str!("../../sim/presets/nvme.gen4-990pro-class.toml"),
    ),
    (
        "nvme.gen4-generic.toml",
        include_str!("../../sim/presets/nvme.gen4-generic.toml"),
    ),
];

/// What a preset describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PresetKind {
    Gpu,
    Cpu,
    Nvme,
}

impl PresetKind {
    fn file_prefix(self) -> &'static str {
        match self {
            PresetKind::Gpu => "gpu",
            PresetKind::Cpu => "cpu",
            PresetKind::Nvme => "nvme",
        }
    }
}

/// The memory a GPU's kernels compute from: its own VRAM, or memory it shares with the CPU
/// (RFC-0001 §5's tier kinds `Vram` and `Unified`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTier {
    Vram,
    Unified,
}

/// The figure keys of every device preset, beside its `w_eff.<tier>.<kernel class>` keys.
pub const DEVICE_KEYS: [&str; 4] = ["memory", "f_eff", "t_fixed", "sigma_dev"];

/// The figure key a discrete GPU has besides: its host link's pinned host-to-device copy rate.
pub const DISCRETE_GPU_KEY: &str = "host_link_pinned_h2d";

/// The figure keys of every drive preset: `FakeIo`'s model (RFC-0060 §8.4).
pub const DRIVE_KEYS: [&str; 7] = [
    "seq_read",
    "sustained_read",
    "throttled_read",
    "heating_time",
    "qd1_4k_read_iops",
    "qd32_4k_read_iops",
    "read_latency",
];

/// The kernel classes W_eff is given for: the two RFC-0010's M1 probe measures, `DenseGemv` and
/// `ExpertGemv` (`syntropy_core::resource::KernelClass`).
pub const KERNEL_CLASSES: [&str; 2] = ["dense_gemv", "expert_gemv"];

/// The prefix of the W_eff keys, `w_eff.<tier>.<kernel class>`.
pub const W_EFF: &str = "w_eff";

/// One device or drive preset.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Preset {
    /// The name a scenario uses: `gpu:ada-24g`, or a drive's bare name, `gen4-990pro-class`.
    pub name: String,
    pub kind: PresetKind,
    /// A GPU's own memory, which sets the tiers of its W_eff keys. Only a `gpu` preset has it.
    #[serde(default)]
    pub memory_tier: Option<MemoryTier>,
    /// The part, in words.
    pub describes: String,
    /// Where the figures come from.
    pub sources: Vec<String>,
    pub figures: BTreeMap<String, Figure>,
}

/// One figure of a preset.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum Figure {
    Known(Known),
    /// The research does not give it; the text says where it is left open.
    Unknown {
        unknown: String,
    },
}

/// A figure the research gives.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Known {
    /// The value, where the source gives one.
    #[serde(default)]
    pub value: Option<f64>,
    /// The range, where the source gives one.
    #[serde(default)]
    pub range: Option<[f64; 2]>,
    pub unit: String,
    pub label: Label,
    pub source: String,
}

/// How a figure is known (see the [module](self) table).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Label {
    Published,
    Measured,
    Estimate,
    Target,
}

/// Why a preset file did not load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PresetError {
    Parse {
        file: String,
        message: String,
    },
    /// The file's name is not `<kind>.<name>.toml`.
    FileName {
        file: String,
        expected: String,
    },
    NoSources(String),
    NoFigures(String),
    /// A `gpu` preset with no `memory_tier`, or another kind with one.
    MemoryTier {
        preset: String,
        kind: PresetKind,
    },
    /// A key of the kind's set that the preset leaves out: write it as `unknown` if the research
    /// gives no figure.
    MissingFigure {
        preset: String,
        figure: String,
    },
    /// A `w_eff.*` key outside the kind's set: a tier the device does not read, or a misspelling.
    NotAKey {
        preset: String,
        figure: String,
    },
    /// A figure with neither a value nor a range, a range that runs backwards, or no unit.
    NoValue {
        preset: String,
        figure: String,
    },
    /// A figure with an empty source, or an unknown with no pointer to where it is left open.
    NoSource {
        preset: String,
        figure: String,
    },
}

impl fmt::Display for PresetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PresetError::Parse { file, message } => write!(f, "{file}: not a preset: {message}"),
            PresetError::FileName { file, expected } => {
                write!(f, "{file}: the preset's file must be named {expected}")
            }
            PresetError::NoSources(name) => write!(f, "preset `{name}` cites no source"),
            PresetError::NoFigures(name) => write!(f, "preset `{name}` has no figures"),
            PresetError::MemoryTier { preset, kind } => {
                if *kind == PresetKind::Gpu {
                    write!(
                        f,
                        "preset `{preset}` needs `memory_tier`, `vram` or `unified`"
                    )
                } else {
                    write!(
                        f,
                        "preset `{preset}` has `memory_tier`, which only a gpu has"
                    )
                }
            }
            PresetError::MissingFigure { preset, figure } => write!(
                f,
                "preset `{preset}` has no `{figure}`; write it as `unknown`, with where the \
                 research leaves it open, if the research gives none"
            ),
            PresetError::NotAKey { preset, figure } => write!(
                f,
                "preset `{preset}`: `{figure}` is not one of its W_eff keys, \
                 `w_eff.<tier>.<kernel class>`"
            ),
            PresetError::NoValue { preset, figure } => {
                write!(
                    f,
                    "preset `{preset}`: `{figure}` has no value, range or unit"
                )
            }
            PresetError::NoSource { preset, figure } => {
                write!(f, "preset `{preset}`: `{figure}` cites no source")
            }
        }
    }
}

impl std::error::Error for PresetError {}

/// Why a figure cannot be used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FigureError {
    /// The preset has no figure of that key.
    Absent { preset: String, key: String },
    /// The research leaves the figure open; `open` says where.
    Unknown {
        preset: String,
        key: String,
        open: String,
    },
}

impl fmt::Display for FigureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FigureError::Absent { preset, key } => {
                write!(f, "preset `{preset}` has no figure `{key}`")
            }
            FigureError::Unknown { preset, key, open } => write!(
                f,
                "preset `{preset}`'s `{key}` is unknown ({open}); a scenario must supply it"
            ),
        }
    }
}

impl std::error::Error for FigureError {}

impl Preset {
    /// Parses the preset file `file` and checks it: its name, its sources, and each figure's
    /// value and source.
    pub fn from_toml(file: &str, text: &str) -> Result<Preset, PresetError> {
        let preset: Preset = toml::from_str(text).map_err(|error| PresetError::Parse {
            file: file.to_owned(),
            message: error.to_string(),
        })?;
        let expected = preset.file_name();
        if file != expected {
            return Err(PresetError::FileName {
                file: file.to_owned(),
                expected,
            });
        }
        if preset.sources.iter().all(|source| source.trim().is_empty()) {
            return Err(PresetError::NoSources(preset.name));
        }
        if preset.figures.is_empty() {
            return Err(PresetError::NoFigures(preset.name));
        }
        if preset.memory_tier.is_some() != (preset.kind == PresetKind::Gpu) {
            return Err(PresetError::MemoryTier {
                preset: preset.name,
                kind: preset.kind,
            });
        }
        let required = preset.required_keys();
        if let Some(missing) = required
            .iter()
            .find(|key| !preset.figures.contains_key(*key))
        {
            return Err(PresetError::MissingFigure {
                preset: preset.name.clone(),
                figure: missing.clone(),
            });
        }
        for (name, figure) in &preset.figures {
            let at = || (preset.name.clone(), name.clone());
            if is_w_eff(name) && !required.contains(name) {
                let (preset, figure) = at();
                return Err(PresetError::NotAKey { preset, figure });
            }
            match figure {
                Figure::Known(known) => match known.problem() {
                    Some(KnownProblem::NoValue) => {
                        let (preset, figure) = at();
                        return Err(PresetError::NoValue { preset, figure });
                    }
                    Some(KnownProblem::NoSource) => {
                        let (preset, figure) = at();
                        return Err(PresetError::NoSource { preset, figure });
                    }
                    None => {}
                },
                Figure::Unknown { unknown } if unknown.trim().is_empty() => {
                    let (preset, figure) = at();
                    return Err(PresetError::NoSource { preset, figure });
                }
                Figure::Unknown { .. } => {}
            }
        }
        Ok(preset)
    }

    /// The figure `key`, if the research gives it: an `unknown` or absent figure is an error,
    /// never a default.
    pub fn figure(&self, key: &str) -> Result<&Known, FigureError> {
        match self.figures.get(key) {
            Some(Figure::Known(known)) => Ok(known),
            Some(Figure::Unknown { unknown }) => Err(FigureError::Unknown {
                preset: self.name.clone(),
                key: key.to_owned(),
                open: unknown.clone(),
            }),
            None => Err(FigureError::Absent {
                preset: self.name.clone(),
                key: key.to_owned(),
            }),
        }
    }

    /// The memory tiers the device's kernels read, which its W_eff keys name (RFC-0010's M1
    /// probe): none for a drive.
    pub fn tiers(&self) -> &'static [&'static str] {
        match (self.kind, self.memory_tier) {
            (PresetKind::Gpu, Some(MemoryTier::Vram)) => &["vram", "host_pinned"],
            (PresetKind::Gpu, Some(MemoryTier::Unified)) => &["unified"],
            (PresetKind::Cpu, _) => &["host_pinned", "host_pageable"],
            (PresetKind::Gpu, None) | (PresetKind::Nvme, _) => &[],
        }
    }

    /// Every figure key the preset's kind requires (see the [module](self) docs).
    pub fn required_keys(&self) -> Vec<String> {
        if self.kind == PresetKind::Nvme {
            return DRIVE_KEYS.iter().map(|key| (*key).to_owned()).collect();
        }
        let mut keys: Vec<String> = DEVICE_KEYS.iter().map(|key| (*key).to_owned()).collect();
        for tier in self.tiers() {
            for class in KERNEL_CLASSES {
                keys.push(w_eff_key(tier, class));
            }
        }
        if self.memory_tier == Some(MemoryTier::Vram) {
            keys.push(DISCRETE_GPU_KEY.to_owned());
        }
        keys
    }

    /// `<kind>.<name>.toml`, the name's own `<kind>:` prefix left out.
    pub fn file_name(&self) -> String {
        let prefix = self.kind.file_prefix();
        let bare = self
            .name
            .strip_prefix(&format!("{prefix}:"))
            .unwrap_or(&self.name);
        format!("{prefix}.{bare}.toml")
    }
}

/// The W_eff key for `tier` and kernel `class`: `w_eff.<tier>.<class>`.
pub fn w_eff_key(tier: &str, class: &str) -> String {
    format!("{W_EFF}.{tier}.{class}")
}

fn is_w_eff(key: &str) -> bool {
    key == W_EFF || key.starts_with("w_eff.")
}

/// What is wrong with a known figure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KnownProblem {
    /// Neither a value nor a range, a range that runs backwards, or no unit.
    NoValue,
    /// No source.
    NoSource,
}

impl Known {
    pub(crate) fn problem(&self) -> Option<KnownProblem> {
        let backwards = self.range.is_some_and(|[lo, hi]| lo > hi);
        if (self.value.is_none() && self.range.is_none())
            || backwards
            || self.unit.trim().is_empty()
        {
            Some(KnownProblem::NoValue)
        } else if self.source.trim().is_empty() {
            Some(KnownProblem::NoSource)
        } else {
            None
        }
    }
}

/// Every compiled-in preset, by name.
///
/// # Panics
///
/// If a compiled-in file does not load; `presets_load` tests that none can.
pub fn by_name() -> BTreeMap<String, Preset> {
    ALL.iter()
        .map(|(file, text)| {
            let preset = Preset::from_toml(file, text)
                .unwrap_or_else(|error| panic!("a compiled-in preset does not load: {error}"));
            (preset.name.clone(), preset)
        })
        .collect()
}

/// The compiled-in preset `name`, loaded once.
pub fn get(name: &str) -> Option<&'static Preset> {
    static ALL_PRESETS: OnceLock<BTreeMap<String, Preset>> = OnceLock::new();
    ALL_PRESETS.get_or_init(by_name).get(name)
}

/// Every figure key some kind requires: the device and drive keys, the discrete GPU's host link,
/// and every tier's W_eff for each kernel class.
pub fn all_keys() -> Vec<String> {
    let mut keys: Vec<String> = DEVICE_KEYS
        .iter()
        .chain(DRIVE_KEYS.iter())
        .chain([DISCRETE_GPU_KEY].iter())
        .map(|key| (*key).to_owned())
        .collect();
    for tier in ["vram", "unified", "host_pinned", "host_pageable"] {
        for class in KERNEL_CLASSES {
            keys.push(w_eff_key(tier, class));
        }
    }
    keys
}
