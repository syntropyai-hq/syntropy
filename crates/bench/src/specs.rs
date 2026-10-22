// SPDX-License-Identifier: MIT OR Apache-2.0

//! The spec database, `bench/specs/` (RFC-0061 §5.3): the peak figures `t_peak` needs, one
//! TOML record per device or memory configuration, each figure with its source.
//!
//! A record names the exact `models` it covers, as the rig inventory (`bench/lab/rigs.toml`,
//! [`crate::rigs`]) spells them, and its kind decides which figures it holds:
//!
//! - `gpu`: `memory_gb_per_s`, for the `Vram` tier.
//! - `cpu`: `memory_type`, `memory_channels` and an optional `fabric_read_gb_per_s`, for the
//!   DRAM formula's channel count and a fabric limit narrower than DRAM.
//! - `dram`: `bytes_per_transfer`, the DRAM formula's width per channel.
//! - `pcie`: `generation`, `transfer_gt_per_s` and `encoding`, a PCIe link's rate per lane.
//! - `nvme`: `sequential_read_gb_per_s`, for the `Nvme` tier.
//!
//! Every figure is a table `{ value, source, where, checked }`: the value; the vendor or
//! standards document it is read from, an `https://` URL; where in that document; and the date
//! someone checked the value against it. A spec figure never comes from a measurement.
//!
//! [`peaks`] turns a rig's inventory record into `W_peak` per path. Each path's peak is its
//! narrowest element (RFC-0061 §5.3):
//! - DRAM is derived, `channels × bytes_per_transfer × MT/s`. The channel count is the CPU's,
//!   capped by the DIMMs fitted. The MT/s is the slowest DIMM's `speed_mts`, which the inventory
//!   defines as the speed the module runs at, not its rating, so a board that runs below the
//!   module's profile, or two DIMMs per channel, gives the speed actually used. A fabric limit
//!   then caps it.
//! - A PCIe link is derived, `GT/s × payload/total ÷ 8 × lanes`: 16 GT/s × 16 × 128/130 ÷ 8 =
//!   31.5 GB/s per direction for a 4.0 x16 link.
//! - A GPU reading host memory is capped by its PCIe link, and a drive by its link.
//!
//! Every other figure is a cited vendor or standards figure, never a measurement.
//!
//! [`sanity`] compares probe figures with those peaks: one below 50% of its peak marks the run
//! `dirty: misconfigured` (`00-vision-and-scope.md` §9, principle 2), and a probe of a path
//! with no peak is reported as unchecked.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::rigs::{Attachment, PcieLink, Rig};
use crate::schema::{EnvCapture, TierKind};

/// The spec database, relative to the repository root.
pub const DIR: &str = "bench/specs";

/// The `dirty` reason a probe below half its spec gives (`EnvCapture::dirty`).
pub const MISCONFIGURED: &str = "misconfigured";

/// The fraction of its spec below which a probe figure marks the rig misconfigured.
pub const SANITY_FRACTION: f64 = 0.5;

/// One figure, with where it comes from.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fact<T> {
    /// The figure.
    pub value: T,
    /// The vendor or standards document, as an `https://` URL.
    pub source: String,
    /// Where in the document the figure is: a page, table or row.
    #[serde(rename = "where")]
    pub location: String,
    /// When someone last checked the figure against the document, `YYYY-MM-DD`.
    pub checked: String,
}

/// What a record describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// A GPU's device memory.
    Gpu,
    /// A CPU's memory controller and fabric.
    Cpu,
    /// A DRAM standard's channel width.
    Dram,
    /// A PCIe generation's signalling.
    Pcie,
    /// An NVMe drive.
    Nvme,
}

impl Kind {
    fn as_str(self) -> &'static str {
        match self {
            Kind::Gpu => "gpu",
            Kind::Cpu => "cpu",
            Kind::Dram => "dram",
            Kind::Pcie => "pcie",
            Kind::Nvme => "nvme",
        }
    }
}

/// A record as written. [`Record::parse`] checks that it holds exactly its kind's figures.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRecord {
    kind: Kind,
    models: Vec<String>,
    generation: Option<u8>,
    memory_gb_per_s: Option<Fact<f64>>,
    memory_type: Option<Fact<String>>,
    memory_channels: Option<Fact<u32>>,
    fabric_read_gb_per_s: Option<Fact<f64>>,
    bytes_per_transfer: Option<Fact<u32>>,
    transfer_gt_per_s: Option<Fact<f64>>,
    encoding: Option<Fact<String>>,
    sequential_read_gb_per_s: Option<Fact<f64>>,
}

/// One spec record.
#[derive(Clone, Debug, PartialEq)]
pub enum Record {
    /// A GPU's device memory.
    Gpu {
        /// The models it covers.
        models: Vec<String>,
        /// Peak device-memory bandwidth, GB/s.
        memory_gb_per_s: Fact<f64>,
    },
    /// A CPU's memory controller and fabric.
    Cpu {
        /// The models it covers.
        models: Vec<String>,
        /// The DRAM standard, such as `DDR5`, which names the `dram` record to use.
        memory_type: Fact<String>,
        /// The memory channels the CPU drives.
        memory_channels: Fact<u32>,
        /// A vendor-documented fabric read limit for the whole package, GB/s, where it is
        /// narrower than DRAM (such as AM5's per-CCD links).
        fabric_read_gb_per_s: Option<Fact<f64>>,
    },
    /// A DRAM standard's channel width.
    Dram {
        /// The standards it covers, such as `DDR5`.
        models: Vec<String>,
        /// Bytes moved per transfer on one channel.
        bytes_per_transfer: Fact<u32>,
    },
    /// A PCIe generation's signalling.
    Pcie {
        /// The names it covers, such as `PCIe 4.0`.
        models: Vec<String>,
        /// The generation, as the inventory's `gen<N>` writes it.
        generation: u8,
        /// The bit rate per lane, GT/s.
        transfer_gt_per_s: Fact<f64>,
        /// The line encoding, such as `128b/130b`.
        encoding: Fact<String>,
    },
    /// An NVMe drive.
    Nvme {
        /// The models it covers.
        models: Vec<String>,
        /// Peak sequential read, GB/s.
        sequential_read_gb_per_s: Fact<f64>,
    },
}

impl Record {
    /// Parses one record and checks that it holds exactly its kind's figures, each with an
    /// `https://` source, a location and a check date (the source lint).
    pub fn parse(text: &str) -> Result<Record, Vec<String>> {
        let raw: RawRecord = toml::from_str(text).map_err(|e| vec![e.message().to_owned()])?;
        let mut problems = Vec::new();
        if raw.models.is_empty() || raw.models.iter().any(|m| m.trim().is_empty()) {
            problems.push(String::from(
                "`models` must name at least one model, none empty",
            ));
        }
        let mut lint = |name: &str, source: &str, location: &str, checked: &str| {
            if !source.starts_with("https://") || source.len() <= "https://".len() {
                problems.push(format!(
                    "`{name}`'s `source` must be the vendor or standards document's `https://` \
                     URL"
                ));
            }
            if location.trim().is_empty() {
                problems.push(format!(
                    "`{name}`'s `where` must say where in the document it is"
                ));
            }
            if !is_date(checked) {
                problems.push(format!("`{name}`'s `checked` must be a date, `YYYY-MM-DD`"));
            }
        };
        macro_rules! lint {
            ($($field:ident),*) => {
                $(if let Some(fact) = &raw.$field {
                    lint(stringify!($field), &fact.source, &fact.location, &fact.checked);
                })*
            };
        }
        lint!(
            memory_gb_per_s,
            memory_type,
            memory_channels,
            fabric_read_gb_per_s,
            bytes_per_transfer,
            transfer_gt_per_s,
            encoding,
            sequential_read_gb_per_s
        );
        let present: BTreeSet<&str> = [
            ("generation", raw.generation.is_some()),
            ("memory_gb_per_s", raw.memory_gb_per_s.is_some()),
            ("memory_type", raw.memory_type.is_some()),
            ("memory_channels", raw.memory_channels.is_some()),
            ("fabric_read_gb_per_s", raw.fabric_read_gb_per_s.is_some()),
            ("bytes_per_transfer", raw.bytes_per_transfer.is_some()),
            ("transfer_gt_per_s", raw.transfer_gt_per_s.is_some()),
            ("encoding", raw.encoding.is_some()),
            (
                "sequential_read_gb_per_s",
                raw.sequential_read_gb_per_s.is_some(),
            ),
        ]
        .into_iter()
        .filter_map(|(name, set)| set.then_some(name))
        .collect();
        let (required, optional): (&[&str], &[&str]) = match raw.kind {
            Kind::Gpu => (&["memory_gb_per_s"], &[]),
            Kind::Cpu => (
                &["memory_type", "memory_channels"],
                &["fabric_read_gb_per_s"],
            ),
            Kind::Dram => (&["bytes_per_transfer"], &[]),
            Kind::Pcie => (&["generation", "transfer_gt_per_s", "encoding"], &[]),
            Kind::Nvme => (&["sequential_read_gb_per_s"], &[]),
        };
        let kind = raw.kind.as_str();
        for name in required {
            if !present.contains(name) {
                problems.push(format!("a `{kind}` record needs `{name}`"));
            }
        }
        for name in &present {
            if !required.contains(name) && !optional.contains(name) {
                problems.push(format!("a `{kind}` record has no `{name}`"));
            }
        }
        let positive = |name: &str, value: f64, problems: &mut Vec<String>| {
            if !(value.is_finite() && value > 0.0) {
                problems.push(format!("`{name}` must be a positive number"));
            }
        };
        for (name, fact) in [
            ("memory_gb_per_s", &raw.memory_gb_per_s),
            ("fabric_read_gb_per_s", &raw.fabric_read_gb_per_s),
            ("transfer_gt_per_s", &raw.transfer_gt_per_s),
            ("sequential_read_gb_per_s", &raw.sequential_read_gb_per_s),
        ] {
            if let Some(fact) = fact {
                positive(name, fact.value, &mut problems);
            }
        }
        // A zero here would give a zero `W_peak`, which no probe can fall below.
        for (name, fact) in [
            ("memory_channels", &raw.memory_channels),
            ("bytes_per_transfer", &raw.bytes_per_transfer),
        ] {
            if fact.as_ref().is_some_and(|f| f.value == 0) {
                problems.push(format!("`{name}` must be a positive integer"));
            }
        }
        if let Some(fact) = &raw.encoding
            && encoding_ratio(&fact.value).is_none()
        {
            problems.push(String::from(
                "`encoding` must be `<payload>b/<total>b`, such as `128b/130b`",
            ));
        }
        if !problems.is_empty() {
            return Err(problems);
        }
        // Every required figure is present: checked above.
        let missing = || vec![String::from("internal: a required figure is missing")];
        let RawRecord { models, .. } = raw.clone();
        Ok(match raw.kind {
            Kind::Gpu => Record::Gpu {
                models,
                memory_gb_per_s: raw.memory_gb_per_s.ok_or_else(missing)?,
            },
            Kind::Cpu => Record::Cpu {
                models,
                memory_type: raw.memory_type.ok_or_else(missing)?,
                memory_channels: raw.memory_channels.ok_or_else(missing)?,
                fabric_read_gb_per_s: raw.fabric_read_gb_per_s,
            },
            Kind::Dram => Record::Dram {
                models,
                bytes_per_transfer: raw.bytes_per_transfer.ok_or_else(missing)?,
            },
            Kind::Pcie => Record::Pcie {
                models,
                generation: raw.generation.ok_or_else(missing)?,
                transfer_gt_per_s: raw.transfer_gt_per_s.ok_or_else(missing)?,
                encoding: raw.encoding.ok_or_else(missing)?,
            },
            Kind::Nvme => Record::Nvme {
                models,
                sequential_read_gb_per_s: raw.sequential_read_gb_per_s.ok_or_else(missing)?,
            },
        })
    }

    /// What the record describes.
    pub fn kind(&self) -> Kind {
        match self {
            Record::Gpu { .. } => Kind::Gpu,
            Record::Cpu { .. } => Kind::Cpu,
            Record::Dram { .. } => Kind::Dram,
            Record::Pcie { .. } => Kind::Pcie,
            Record::Nvme { .. } => Kind::Nvme,
        }
    }

    /// The models it covers.
    pub fn models(&self) -> &[String] {
        match self {
            Record::Gpu { models, .. }
            | Record::Cpu { models, .. }
            | Record::Dram { models, .. }
            | Record::Pcie { models, .. }
            | Record::Nvme { models, .. } => models,
        }
    }
}

/// `YYYY-MM-DD`, with a month 01–12 and a day 01–31.
fn is_date(text: &str) -> bool {
    let parts: Vec<&str> = text.split('-').collect();
    let [year, month, day] = parts.as_slice() else {
        return false;
    };
    let number = |s: &str, len: usize| {
        (s.len() == len && s.bytes().all(|b| b.is_ascii_digit())).then(|| s.parse::<u32>().ok())?
    };
    matches!(
        (number(year, 4), number(month, 2), number(day, 2)),
        (Some(_), Some(1..=12), Some(1..=31))
    )
}

/// `128b/130b` as the fraction of line bits that carry data.
fn encoding_ratio(encoding: &str) -> Option<f64> {
    let (payload, total) = encoding.split_once('/')?;
    let bits = |s: &str| s.strip_suffix('b')?.parse::<u32>().ok().filter(|&n| n > 0);
    let (payload, total) = (bits(payload)?, bits(total)?);
    (payload <= total).then(|| f64::from(payload) / f64::from(total))
}

/// A problem with one file of the database.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Problem {
    /// The file, as read: under `--dir` when one is given, else under `bench/specs/`.
    pub path: String,
    /// What is wrong.
    pub message: String,
}

impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

/// Every record of the database, with the file each came from.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpecDb {
    /// The records, by file name.
    pub records: BTreeMap<String, Record>,
}

impl SpecDb {
    /// Reads and checks every `*.toml` file of `dir`. A file is named `<kind>-<name>.toml`, and
    /// no two records of a kind cover one model.
    pub fn load(dir: &Path) -> (SpecDb, Vec<Problem>) {
        let mut db = SpecDb::default();
        let mut problems = Vec::new();
        let Ok(entries) = fs::read_dir(dir) else {
            problems.push(Problem {
                path: dir.display().to_string(),
                message: String::from("the spec database directory cannot be read"),
            });
            return (db, problems);
        };
        let mut paths: Vec<_> = entries.flatten().map(|e| e.path()).collect();
        paths.sort();
        let mut covered: BTreeMap<(Kind, String), String> = BTreeMap::new();
        for path in paths {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let shown = format!("{}/{name}", dir.display());
            let mut problem = |message: String| {
                problems.push(Problem {
                    path: shown.clone(),
                    message,
                });
            };
            if !path.is_file() || !name.ends_with(".toml") {
                problem(String::from(
                    "only `<kind>-<name>.toml` records belong here",
                ));
                continue;
            }
            let text = match fs::read_to_string(&path) {
                Ok(text) => text.replace("\r\n", "\n"),
                Err(e) => {
                    problem(format!("cannot read: {e}"));
                    continue;
                }
            };
            let record = match Record::parse(&text) {
                Ok(record) => record,
                Err(found) => {
                    for message in found {
                        problem(message);
                    }
                    continue;
                }
            };
            let prefix = format!("{}-", record.kind().as_str());
            if !name.starts_with(&prefix) || name.len() <= prefix.len() + ".toml".len() {
                problem(format!(
                    "a `{}` record's file is named `{prefix}<name>.toml`",
                    record.kind().as_str()
                ));
            }
            for model in record.models() {
                let key = (record.kind(), model.clone());
                if let Some(other) = covered.get(&key) {
                    problem(format!(
                        "`{model}` is covered by {other} too; one record per model"
                    ));
                } else {
                    covered.insert(key, name.clone());
                }
            }
            db.records.insert(name, record);
        }
        (db, problems)
    }

    /// The record of `kind` that covers `model`.
    pub fn find(&self, kind: Kind, model: &str) -> Option<&Record> {
        self.records
            .values()
            .find(|r| r.kind() == kind && r.models().iter().any(|m| m == model))
    }

    fn pcie(&self, link: PcieLink) -> Option<(f64, &Fact<f64>, &Fact<String>)> {
        self.records.values().find_map(|r| match r {
            Record::Pcie {
                generation,
                transfer_gt_per_s,
                encoding,
                ..
            } if *generation == link.gen_ => {
                let ratio = encoding_ratio(&encoding.value)?;
                let gb_per_s = transfer_gt_per_s.value * ratio / 8.0 * f64::from(link.width);
                Some((gb_per_s, transfer_gt_per_s, encoding))
            }
            _ => None,
        })
    }
}

/// A data path whose peak `t_peak` uses.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DataPath {
    /// A GPU reading its own device memory, `Vram`. The inventory's GPU slot.
    Vram(String),
    /// The CPU reading host memory in place, `HostPinned` or `HostPageable`.
    HostByCpu,
    /// A GPU reading host memory over its PCIe link, `HostPinned`. The GPU's slot.
    HostByGpu(String),
    /// Reading an NVMe drive, `Nvme`. The drive's slot.
    Nvme(String),
}

impl DataPath {
    /// The tiers this path reads.
    pub fn tiers(&self) -> &'static [TierKind] {
        match self {
            DataPath::Vram(_) => &[TierKind::Vram],
            DataPath::HostByCpu => &[TierKind::HostPinned, TierKind::HostPageable],
            DataPath::HostByGpu(_) => &[TierKind::HostPinned],
            DataPath::Nvme(_) => &[TierKind::Nvme],
        }
    }
}

impl fmt::Display for DataPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataPath::Vram(slot) => write!(f, "vram {slot}"),
            DataPath::HostByCpu => f.write_str("host memory, CPU in place"),
            DataPath::HostByGpu(slot) => write!(f, "host memory, read by the GPU in {slot}"),
            DataPath::Nvme(slot) => write!(f, "nvme {slot}"),
        }
    }
}

/// One element along a path, with its rate and how it was found.
#[derive(Clone, Debug, PartialEq)]
pub struct Element {
    /// What the element is, such as `DDR5, 2 channels × 8 B × 6000 MT/s`.
    pub name: String,
    /// Its rate, GB/s.
    pub gb_per_s: f64,
    /// The sources the rate rests on.
    pub sources: Vec<String>,
}

/// A path's `W_peak`: the narrowest of its elements.
#[derive(Clone, Debug, PartialEq)]
pub struct Peak {
    /// The path.
    pub path: DataPath,
    /// Its elements, in the order the data crosses them.
    pub chain: Vec<Element>,
}

impl Peak {
    /// `W_peak`, GB/s: the narrowest element's rate.
    pub fn gb_per_s(&self) -> f64 {
        self.chain
            .iter()
            .map(|e| e.gb_per_s)
            .fold(f64::INFINITY, f64::min)
    }

    /// The narrowest element.
    pub fn narrowest(&self) -> Option<&Element> {
        self.chain
            .iter()
            .min_by(|a, b| a.gb_per_s.total_cmp(&b.gb_per_s))
    }
}

/// `W_peak` for every path of `rig`, from its inventory record and the database. Returns every
/// device or configuration the database cannot cover. A path the database cannot price yet,
/// such as a chipset-attached drive, has no peak and is no error: [`without_peak`] lists those.
pub fn peaks(rig: &Rig, db: &SpecDb) -> Result<Vec<Peak>, Vec<String>> {
    let mut problems = Vec::new();
    let mut peaks = Vec::new();

    let dram = dram(rig, db).map_err(|p| problems.push(p)).ok();
    if let Some(dram) = &dram {
        peaks.push(Peak {
            path: DataPath::HostByCpu,
            chain: dram.clone(),
        });
    }
    for gpu in &rig.gpu {
        match db.find(Kind::Gpu, &gpu.model) {
            Some(Record::Gpu {
                memory_gb_per_s, ..
            }) => peaks.push(Peak {
                path: DataPath::Vram(gpu.slot.clone()),
                chain: vec![Element {
                    name: format!("{} device memory", gpu.model),
                    gb_per_s: memory_gb_per_s.value,
                    sources: vec![memory_gb_per_s.source.clone()],
                }],
            }),
            _ => problems.push(format!(
                "GPU `{}` in {} has no `gpu` record in {DIR}",
                gpu.model, gpu.slot
            )),
        }
        if gpu.slot == "integrated" {
            continue;
        }
        let Some(link) = gpu.pcie else {
            problems.push(format!(
                "GPU `{}` in {}: the inventory records no PCIe link, which caps its reads of \
                 host memory",
                gpu.model, gpu.slot
            ));
            continue;
        };
        match (db.pcie(link), &dram) {
            (Some(element), Some(dram)) => {
                let mut chain = dram.clone();
                chain.push(link_element(link, element));
                peaks.push(Peak {
                    path: DataPath::HostByGpu(gpu.slot.clone()),
                    chain,
                });
            }
            (None, _) => problems.push(format!("PCIe {link} has no `pcie` record in {DIR}")),
            (Some(_), None) => {}
        }
    }
    for drive in &rig.nvme {
        let mut chain = Vec::new();
        match db.find(Kind::Nvme, &drive.model) {
            Some(Record::Nvme {
                sequential_read_gb_per_s,
                ..
            }) => chain.push(Element {
                name: format!("{} sequential read", drive.model),
                gb_per_s: sequential_read_gb_per_s.value,
                sources: vec![sequential_read_gb_per_s.source.clone()],
            }),
            _ => {
                problems.push(format!(
                    "drive `{}` in {} has no `nvme` record in {DIR}",
                    drive.model, drive.slot
                ));
                continue;
            }
        }
        if let Some(link) = drive.pcie {
            match db.pcie(link) {
                Some(element) => chain.push(link_element(link, element)),
                None => {
                    problems.push(format!("PCIe {link} has no `pcie` record in {DIR}"));
                    continue;
                }
            }
        }
        if drive.attachment == Attachment::Chipset {
            // No peak for this drive only: see [`without_peak`].
            continue;
        }
        peaks.push(Peak {
            path: DataPath::Nvme(drive.slot.clone()),
            chain,
        });
    }
    if problems.is_empty() {
        Ok(peaks)
    } else {
        Err(problems)
    }
}

/// The paths of `rig` that have no `W_peak` by design, with the reason: a chipset-attached
/// drive, whose chipset uplink (shared by every chipset device) has no record yet.
pub fn without_peak(rig: &Rig) -> Vec<(DataPath, String)> {
    rig.nvme
        .iter()
        .filter(|d| d.attachment == Attachment::Chipset)
        .map(|d| {
            (
                DataPath::Nvme(d.slot.clone()),
                format!(
                    "drive `{}` is chipset-attached, and its chipset uplink, which every chipset \
                     device shares, has no record in {DIR} yet",
                    d.model
                ),
            )
        })
        .collect()
}

/// The CPU's path to DRAM: `channels × bytes_per_transfer × MT/s`, then any fabric limit.
fn dram(rig: &Rig, db: &SpecDb) -> Result<Vec<Element>, String> {
    let Some(Record::Cpu {
        memory_type,
        memory_channels,
        fabric_read_gb_per_s,
        ..
    }) = db.find(Kind::Cpu, &rig.cpu.model)
    else {
        return Err(format!(
            "CPU `{}` has no `cpu` record in {DIR}",
            rig.cpu.model
        ));
    };
    let Some(Record::Dram {
        bytes_per_transfer, ..
    }) = db.find(Kind::Dram, &memory_type.value)
    else {
        return Err(format!(
            "`{}`, the memory of CPU `{}`, has no `dram` record in {DIR}",
            memory_type.value, rig.cpu.model
        ));
    };
    let dimms = u32::try_from(rig.memory.len()).unwrap_or(u32::MAX);
    let Some(mts) = rig.memory.iter().map(|m| m.speed_mts).min() else {
        return Err(format!("{} lists no memory", rig.id));
    };
    // A channel carries data only with a module in it, and every channel runs at the speed of
    // the slowest module.
    let channels = memory_channels.value.min(dimms);
    let gb_per_s =
        f64::from(channels) * f64::from(bytes_per_transfer.value) * f64::from(mts) / 1000.0;
    let mut chain = vec![Element {
        name: format!(
            "{}, {channels} channels × {} B × {mts} MT/s",
            memory_type.value, bytes_per_transfer.value
        ),
        gb_per_s,
        sources: vec![
            memory_channels.source.clone(),
            bytes_per_transfer.source.clone(),
        ],
    }];
    if let Some(fabric) = fabric_read_gb_per_s {
        chain.push(Element {
            name: format!("{} fabric read limit", rig.cpu.model),
            gb_per_s: fabric.value,
            sources: vec![fabric.source.clone()],
        });
    }
    Ok(chain)
}

fn link_element(
    link: PcieLink,
    (gb_per_s, rate, encoding): (f64, &Fact<f64>, &Fact<String>),
) -> Element {
    Element {
        name: format!(
            "PCIe {link}, {} GT/s × {} × {} lanes ÷ 8",
            rate.value, encoding.value, link.width
        ),
        gb_per_s,
        sources: vec![rate.source.clone(), encoding.source.clone()],
    }
}

/// A probe figure to compare with its path's spec.
#[derive(Clone, Debug, PartialEq)]
pub struct Probe {
    /// The path probed.
    pub path: DataPath,
    /// What the probe measured, GB/s.
    pub gb_per_s: f64,
}

/// What [`sanity`] found: probes below half their spec, and probes it could not check.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Sanity {
    /// Each probe figure below [`SANITY_FRACTION`] of its path's `W_peak`, as a sentence naming
    /// the path, the probe and the spec. Any of these marks the rig misconfigured.
    pub misconfigured: Vec<String>,
    /// Each probe on a path the database gives no peak, such as a chipset-attached drive. That
    /// is a gap in the database, not in the rig, so it never marks the run dirty.
    pub unchecked: Vec<String>,
}

/// Compares each probe figure with its path's `W_peak`.
pub fn sanity(peaks: &[Peak], probes: &[Probe]) -> Sanity {
    let mut found = Sanity::default();
    for probe in probes {
        let Some(peak) = peaks.iter().find(|p| p.path == probe.path) else {
            found.unchecked.push(format!(
                "{}: probed {:.2} GB/s, but the spec database gives this path no peak",
                probe.path, probe.gb_per_s
            ));
            continue;
        };
        let spec = peak.gb_per_s();
        if probe.gb_per_s < SANITY_FRACTION * spec {
            found.misconfigured.push(format!(
                "{}: probed {:.2} GB/s, {:.0}% of its spec {:.2} GB/s, below {:.0}%",
                probe.path,
                probe.gb_per_s,
                100.0 * probe.gb_per_s / spec,
                spec,
                100.0 * SANITY_FRACTION
            ));
        }
    }
    found
}

/// Runs [`sanity`] and, when any probe falls below half its spec, marks the run
/// `dirty: misconfigured` (RFC-0061 §5.3). An unchecked probe leaves the run as it is. A run
/// already dirty for another reason keeps that reason. Returns the findings for the run's log.
pub fn mark_misconfigured(env: &mut EnvCapture, peaks: &[Peak], probes: &[Probe]) -> Sanity {
    let found = sanity(peaks, probes);
    if !found.misconfigured.is_empty() && env.dirty.is_none() {
        env.dirty = Some(String::from(MISCONFIGURED));
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    const GPU: &str = r#"
kind = "gpu"
models = ["Example GPU"]

[memory_gb_per_s]
value = 1000.0
source = "https://example.com/gpu.pdf"
where = "table 1"
checked = "2026-10-22"
"#;

    #[test]
    fn a_record_parses_and_reports_its_kind() {
        let record = Record::parse(GPU).expect("parses");
        assert_eq!(record.kind(), Kind::Gpu);
        assert_eq!(record.models(), ["Example GPU"]);
    }

    #[test]
    fn the_source_lint_fails_a_figure_without_a_url() {
        let no_source = GPU.replace("source = \"https://example.com/gpu.pdf\"\n", "");
        let problems = Record::parse(&no_source).unwrap_err();
        assert!(
            problems.iter().any(|p| p.contains("source")),
            "{problems:?}"
        );
        for (bad, want) in [
            (
                GPU.replace("https://example.com", "http://example.com"),
                "`https://` URL",
            ),
            (
                GPU.replace("https://example.com/gpu.pdf", "https://"),
                "`https://` URL",
            ),
            (
                GPU.replace("where = \"table 1\"", "where = \" \""),
                "`where`",
            ),
            (GPU.replace("2026-10-22", "22/10/2026"), "`checked`"),
            (GPU.replace("2026-10-22", "2026-13-01"), "`checked`"),
        ] {
            let problems = Record::parse(&bad).unwrap_err();
            assert!(
                problems.iter().any(|p| p.contains(want)),
                "{want}: {problems:?}"
            );
        }
    }

    #[test]
    fn a_record_holds_exactly_its_kinds_figures() {
        let problems =
            Record::parse(&GPU.replace("kind = \"gpu\"", "kind = \"nvme\"")).unwrap_err();
        assert!(problems.contains(&String::from(
            "a `nvme` record needs `sequential_read_gb_per_s`"
        )));
        assert!(problems.contains(&String::from("a `nvme` record has no `memory_gb_per_s`")));
        let problems = Record::parse(&GPU.replace("1000.0", "0.0")).unwrap_err();
        assert!(problems.iter().any(|p| p.contains("positive")));
        // A zero channel count or width would give a zero peak that no probe falls below.
        let fact = "source = \"https://e.com\"\nwhere = \"p\"\nchecked = \"2026-10-22\"\n";
        let cpu = format!(
            "kind = \"cpu\"\nmodels = [\"C\"]\n[memory_type]\nvalue = \"DDR5\"\n{fact}\
             [memory_channels]\nvalue = 0\n{fact}"
        );
        assert!(Record::parse(&cpu).unwrap_err().contains(&String::from(
            "`memory_channels` must be a positive integer"
        )));
        let dram = format!(
            "kind = \"dram\"\nmodels = [\"DDR5\"]\n[bytes_per_transfer]\nvalue = 0\n{fact}"
        );
        assert!(Record::parse(&dram).unwrap_err().contains(&String::from(
            "`bytes_per_transfer` must be a positive integer"
        )));
        let problems = Record::parse(&GPU.replace("[\"Example GPU\"]", "[]")).unwrap_err();
        assert!(problems.iter().any(|p| p.contains("`models`")));
        assert!(Record::parse(&format!("{GPU}\nextra = 1\n")).is_err());
    }

    #[test]
    fn encodings_and_dates_parse() {
        assert_eq!(encoding_ratio("128b/130b"), Some(128.0 / 130.0));
        assert_eq!(encoding_ratio("8b/10b"), Some(0.8));
        for bad in ["128/130", "130b/128b", "b/130b", "0b/1b", "x"] {
            assert_eq!(encoding_ratio(bad), None, "{bad}");
        }
        assert!(is_date("2026-10-22"));
        for bad in [
            "2026-1-22",
            "2026-00-10",
            "2026-10-32",
            "26-10-22",
            "2026-10-22x",
        ] {
            assert!(!is_date(bad), "{bad}");
        }
    }
}
