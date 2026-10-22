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
//! preset never fills one in. The fakes that use a figure (QUAL-E8-T05, T06, T07) decide what an
//! unknown means for them, and a scenario can supply a value. `cargo xtask ci lint`'s
//! `sim-presets` step checks every file: its sources, and each figure's label and source.
//!
//! The files are compiled in ([`ALL`]), so a scenario's device and drive names resolve without
//! reading the disk, and a test checks [`ALL`] against the directory.

use std::collections::BTreeMap;
use std::fmt;

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

/// One device or drive preset.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Preset {
    /// The name a scenario uses: `gpu:ada-24g`, or a drive's bare name, `gen4-990pro-class`.
    pub name: String,
    pub kind: PresetKind,
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
    /// A figure with neither a value nor a range, or a range that runs backwards.
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
            PresetError::NoValue { preset, figure } => {
                write!(f, "preset `{preset}`: `{figure}` has no value or range")
            }
            PresetError::NoSource { preset, figure } => {
                write!(f, "preset `{preset}`: `{figure}` cites no source")
            }
        }
    }
}

impl std::error::Error for PresetError {}

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
        for (name, figure) in &preset.figures {
            let at = || (preset.name.clone(), name.clone());
            match figure {
                Figure::Known(known) => {
                    let backwards = known.range.is_some_and(|[lo, hi]| lo > hi);
                    if (known.value.is_none() && known.range.is_none()) || backwards {
                        let (preset, figure) = at();
                        return Err(PresetError::NoValue { preset, figure });
                    }
                    if known.source.trim().is_empty() {
                        let (preset, figure) = at();
                        return Err(PresetError::NoSource { preset, figure });
                    }
                }
                Figure::Unknown { unknown } if unknown.trim().is_empty() => {
                    let (preset, figure) = at();
                    return Err(PresetError::NoSource { preset, figure });
                }
                Figure::Unknown { .. } => {}
            }
        }
        Ok(preset)
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
