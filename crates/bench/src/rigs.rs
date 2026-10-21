// SPDX-License-Identifier: MIT OR Apache-2.0

//! The rig inventory of RFC-0061 §6.3, `bench/lab/rigs.toml`, and the [`RigFingerprint`] of §2.1
//! (QUAL-E15-T01).
//!
//! The inventory holds one `[[rig]]` record per lab rig: CPU and microcode, memory modules, GPUs
//! with VBIOS and driver, NVMe drives with firmware and slot attachment, NICs with firmware, the
//! BIOS settings that move performance, the negotiated PCIe link of each device, the OS image and
//! kernel, the power plan and the room-temperature sensor. `bench/lab/rigs.toml` documents the
//! format, and [`Inventory::parse`] is the check `lint` runs on it.
//!
//! - **Fingerprint.** [`Rig::fingerprint`] is BLAKE3 over a canonical encoding of the whole
//!   record, so a change to any field starts a new baseline epoch. The encoding sorts keys and
//!   lists, and holds no file path, line ending or map iteration order, so the same record gives
//!   the same fingerprint on every operating system, and reordering a list or reformatting the
//!   file starts no epoch. Each list element is identified by its `slot`.
//! - **Comparison.** [`diff`] compares a rig as observed with its record and names every field
//!   that differs. The lab agent runs it before every block (QUAL-E12-T04).
//! - **Epochs.** [`epochs`] lists the rigs whose fingerprint differs between two revisions of the
//!   inventory, with the fields that changed, for the nightly report. A change to a record lands
//!   only through a PR to `bench/lab/rigs.toml`; the agent reads the record on `main`, so a rig
//!   that changed without one is caught as a mismatch.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer};

use crate::schema::{Blake3, RigFingerprint};

/// Where the inventory lives, relative to the workspace root.
pub const PATH: &str = "bench/lab/rigs.toml";

/// The inventory format version this crate reads, the file's `version`.
pub const VERSION: u16 = 1;

/// Prefixed to the canonical encoding before hashing, so a fingerprint can never equal another
/// BLAKE3 digest of the same bytes, and a new format version changes every fingerprint.
const DOMAIN: &[u8] = b"syntropy rig inventory fingerprint v1\0";

/// `bench/lab/rigs.toml`.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Inventory {
    /// The format version, [`VERSION`].
    pub version: u16,
    /// One record per rig, `[[rig]]` in the file.
    #[serde(default, rename = "rig")]
    pub rigs: Vec<Rig>,
}

/// One rig's record (RFC-0061 §6.3).
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rig {
    /// The rig's name in RFC-0061 §6.2, such as `R-ADA`.
    pub id: String,
    /// The OS image ID the rig boots (RFC-0061 §6.4).
    pub os_image: String,
    /// The kernel version string.
    pub kernel: String,
    /// The power plan or CPU governor, normally `performance`.
    pub power_plan: String,
    /// The room-temperature sensor the rig's ambient is read from.
    pub room_sensor: String,
    pub cpu: Cpu,
    /// One entry per memory module, or one `unified` entry on a system-on-chip.
    pub memory: Vec<Memory>,
    #[serde(default)]
    pub gpu: Vec<Gpu>,
    #[serde(default)]
    pub nvme: Vec<Nvme>,
    #[serde(default)]
    pub nic: Vec<Nic>,
    /// The firmware settings that move performance; absent where the firmware has none a user
    /// sets, as on a Mac.
    pub bios: Option<Bios>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cpu {
    pub model: String,
    /// Absent on CPUs without a loadable microcode revision.
    pub microcode: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Memory {
    /// The slot label, such as `DIMM_A2`, or `unified`.
    pub slot: String,
    pub size_gib: u32,
    /// The speed it runs at, in MT/s.
    pub speed_mts: u32,
    /// The module's part number.
    pub part: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gpu {
    /// The slot label, such as `PCIEX16_1`, or `integrated`.
    pub slot: String,
    pub model: String,
    /// Absent on an integrated GPU.
    pub vbios: Option<String>,
    pub driver: String,
    /// The negotiated link; absent on an integrated GPU.
    pub pcie: Option<PcieLink>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Nvme {
    /// The slot label, such as `M2_1`.
    pub slot: String,
    pub model: String,
    pub firmware: String,
    /// Whether the slot hangs off the CPU or the chipset.
    pub attachment: Attachment,
    /// The negotiated link; absent where the drive is not on PCIe.
    pub pcie: Option<PcieLink>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Nic {
    /// The slot or port label, such as `PCIEX4_1` or `onboard`.
    pub slot: String,
    pub model: String,
    pub firmware: String,
    /// The negotiated link; absent where the NIC is not on PCIe.
    pub pcie: Option<PcieLink>,
}

/// The firmware settings that move performance (RFC-0061 §6.3).
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bios {
    pub version: String,
    /// Such as `XMP-6000` or `JEDEC`.
    pub memory_profile: String,
    /// Such as `PL1=125W,PL2=250W`.
    pub power_limits: String,
    pub smt: bool,
    pub resizable_bar: bool,
    /// Any other setting recorded because it moves performance, by name.
    #[serde(default)]
    pub other: BTreeMap<String, String>,
}

/// Where an NVMe slot is attached.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Attachment {
    Cpu,
    Chipset,
}

impl Attachment {
    fn as_str(self) -> &'static str {
        match self {
            Attachment::Cpu => "cpu",
            Attachment::Chipset => "chipset",
        }
    }
}

/// A negotiated PCIe link, written `gen4x16`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PcieLink {
    /// The generation, 1 to 7.
    pub gen_: u8,
    /// The lane count: 1, 2, 4, 8 or 16.
    pub width: u8,
}

impl FromStr for PcieLink {
    type Err = String;

    fn from_str(s: &str) -> Result<PcieLink, String> {
        let err = || format!("`{s}` is not a PCIe link such as `gen4x16`");
        let (gen_, width) = s
            .strip_prefix("gen")
            .and_then(|rest| rest.split_once('x'))
            .ok_or_else(err)?;
        let digits = |t: &str| !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit());
        if !digits(gen_) || !digits(width) {
            return Err(err());
        }
        let gen_: u8 = gen_.parse().map_err(|_| err())?;
        let width: u8 = width.parse().map_err(|_| err())?;
        if !(1..=7).contains(&gen_) || ![1, 2, 4, 8, 16].contains(&width) {
            return Err(err());
        }
        Ok(PcieLink { gen_, width })
    }
}

impl fmt::Display for PcieLink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "gen{}x{}", self.gen_, self.width)
    }
}

impl<'de> Deserialize<'de> for PcieLink {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<PcieLink, D::Error> {
        let text = String::deserialize(deserializer)?;
        text.parse().map_err(serde::de::Error::custom)
    }
}

/// Why an inventory was rejected: each problem, with the rig it is in where there is one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Invalid(pub Vec<String>);

impl fmt::Display for Invalid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.join("\n"))
    }
}

impl std::error::Error for Invalid {}

impl Inventory {
    /// Parses and validates an inventory: TOML with no unknown key, `version` equal to
    /// [`VERSION`], and every rig valid ([`Rig::problems`]) with an ID no other rig has.
    pub fn parse(text: &str) -> Result<Inventory, Invalid> {
        let inventory: Inventory =
            toml::from_str(text).map_err(|error| Invalid(vec![error.to_string()]))?;
        let mut problems = Vec::new();
        if inventory.version != VERSION {
            problems.push(format!(
                "`version` is {}; this check reads version {VERSION}",
                inventory.version
            ));
        }
        let mut seen = BTreeSet::new();
        for rig in &inventory.rigs {
            if !seen.insert(rig.id.as_str()) {
                problems.push(format!("rig `{}`: the ID is used twice", rig.id));
            }
            problems.extend(
                rig.problems()
                    .into_iter()
                    .map(|problem| format!("rig `{}`: {problem}", rig.id)),
            );
        }
        if problems.is_empty() {
            Ok(inventory)
        } else {
            Err(Invalid(problems))
        }
    }

    /// The rig called `id`.
    pub fn rig(&self, id: &str) -> Option<&Rig> {
        self.rigs.iter().find(|rig| rig.id == id)
    }
}

impl Rig {
    /// What makes the record invalid, if anything: an ID not of the form `R-NAME`, an empty
    /// string or one with surrounding whitespace, a zero size or speed, no memory, or two
    /// elements of one list with the same `slot`. A record must be valid before it is
    /// fingerprinted or compared, because elements are identified by their slot.
    pub fn problems(&self) -> Vec<String> {
        let mut problems = Vec::new();
        if !is_rig_id(&self.id) {
            problems.push(format!(
                "`id` is `{}`; a rig ID is `R-` and upper-case words joined by `-`, such as \
                 `R-ADA`",
                self.id
            ));
        }
        if self.memory.is_empty() {
            problems.push("`memory` lists no module".to_owned());
        }
        for module in &self.memory {
            if module.size_gib == 0 || module.speed_mts == 0 {
                problems.push(format!(
                    "`memory[{}]`: `size_gib` and `speed_mts` must be above zero",
                    module.slot
                ));
            }
        }
        let lists: [(&str, Vec<&str>); 4] = [
            (
                "memory",
                self.memory.iter().map(|m| m.slot.as_str()).collect(),
            ),
            ("gpu", self.gpu.iter().map(|g| g.slot.as_str()).collect()),
            ("nvme", self.nvme.iter().map(|n| n.slot.as_str()).collect()),
            ("nic", self.nic.iter().map(|n| n.slot.as_str()).collect()),
        ];
        for (list, slots) in lists {
            let mut seen = BTreeSet::new();
            for slot in slots {
                if !seen.insert(slot) {
                    problems.push(format!("`{list}`: the slot `{slot}` appears twice"));
                }
            }
        }
        for (field, value) in leaves(&self.node()) {
            if let Leaf::Str(text) = value
                && (text.is_empty() || text.trim() != text)
            {
                problems.push(format!(
                    "`{field}` is `{text}`; a value is non-empty, with no surrounding whitespace"
                ));
            }
        }
        problems
    }

    /// The fingerprint of RFC-0061 §2.1: BLAKE3 over [`DOMAIN`] and [`Rig::canonical`].
    pub fn fingerprint(&self) -> RigFingerprint {
        let mut hasher = blake3::Hasher::new();
        hasher.update(DOMAIN);
        hasher.update(&self.canonical());
        RigFingerprint(Blake3(*hasher.finalize().as_bytes()))
    }

    /// The canonical encoding the fingerprint hashes. Every value is tagged and length-prefixed,
    /// so no two records share an encoding:
    /// - a string is `s<byte length>:<UTF-8 bytes>`;
    /// - a number is `i<decimal>;`, and a flag `t` or `f`;
    /// - a table is `m<entries>:`, then each key, as a string, and its value, in key order;
    /// - a list is `k<elements>:`, then each element's slot, as a string, and the element, in
    ///   slot order.
    ///
    /// An absent optional field is left out of its table.
    pub fn canonical(&self) -> Vec<u8> {
        let mut out = Vec::new();
        encode(&self.node(), &mut out);
        out
    }

    /// The record as a tree of named values, in the form [`Rig::canonical`] encodes.
    fn node(&self) -> Node {
        let mut rig = Table::new();
        rig.str("id", &self.id);
        rig.str("os_image", &self.os_image);
        rig.str("kernel", &self.kernel);
        rig.str("power_plan", &self.power_plan);
        rig.str("room_sensor", &self.room_sensor);
        let mut cpu = Table::new();
        cpu.str("model", &self.cpu.model);
        cpu.opt("microcode", self.cpu.microcode.as_deref());
        rig.node("cpu", cpu.done());
        rig.list(
            "memory",
            self.memory.iter().map(|m| {
                let mut t = Table::new();
                t.str("slot", &m.slot);
                t.int("size_gib", m.size_gib.into());
                t.int("speed_mts", m.speed_mts.into());
                t.str("part", &m.part);
                (m.slot.clone(), t.done())
            }),
        );
        rig.list(
            "gpu",
            self.gpu.iter().map(|g| {
                let mut t = Table::new();
                t.str("slot", &g.slot);
                t.str("model", &g.model);
                t.opt("vbios", g.vbios.as_deref());
                t.str("driver", &g.driver);
                t.opt("pcie", g.pcie.map(|l| l.to_string()).as_deref());
                (g.slot.clone(), t.done())
            }),
        );
        rig.list(
            "nvme",
            self.nvme.iter().map(|n| {
                let mut t = Table::new();
                t.str("slot", &n.slot);
                t.str("model", &n.model);
                t.str("firmware", &n.firmware);
                t.str("attachment", n.attachment.as_str());
                t.opt("pcie", n.pcie.map(|l| l.to_string()).as_deref());
                (n.slot.clone(), t.done())
            }),
        );
        rig.list(
            "nic",
            self.nic.iter().map(|n| {
                let mut t = Table::new();
                t.str("slot", &n.slot);
                t.str("model", &n.model);
                t.str("firmware", &n.firmware);
                t.opt("pcie", n.pcie.map(|l| l.to_string()).as_deref());
                (n.slot.clone(), t.done())
            }),
        );
        if let Some(bios) = &self.bios {
            let mut t = Table::new();
            t.str("version", &bios.version);
            t.str("memory_profile", &bios.memory_profile);
            t.str("power_limits", &bios.power_limits);
            t.flag("smt", bios.smt);
            t.flag("resizable_bar", bios.resizable_bar);
            t.list(
                "other",
                bios.other
                    .iter()
                    .map(|(name, value)| (name.clone(), Node::Leaf(Leaf::Str(value.clone())))),
            );
            rig.node("bios", t.done());
        }
        rig.done()
    }
}

/// `R-` and upper-case words of letters and digits joined by `-`, such as `R-ADA` or
/// `R-ULTRA-A`.
fn is_rig_id(id: &str) -> bool {
    id.strip_prefix("R-").is_some_and(|rest| {
        rest.split('-').all(|word| {
            !word.is_empty()
                && word
                    .bytes()
                    .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
        })
    })
}

/// A value of the record tree.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Node {
    Leaf(Leaf),
    /// A table of named fields.
    Table(BTreeMap<String, Node>),
    /// A list, each element under its identity: its `slot`, or a setting's name.
    List(BTreeMap<String, Node>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Leaf {
    Str(String),
    Int(u64),
    Flag(bool),
}

impl fmt::Display for Leaf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Leaf::Str(text) => write!(f, "{text}"),
            Leaf::Int(n) => write!(f, "{n}"),
            Leaf::Flag(b) => write!(f, "{b}"),
        }
    }
}

/// Builds a [`Node::Table`].
struct Table(BTreeMap<String, Node>);

impl Table {
    fn new() -> Table {
        Table(BTreeMap::new())
    }

    fn node(&mut self, key: &str, node: Node) {
        self.0.insert(key.to_owned(), node);
    }

    fn str(&mut self, key: &str, value: &str) {
        self.node(key, Node::Leaf(Leaf::Str(value.to_owned())));
    }

    fn opt(&mut self, key: &str, value: Option<&str>) {
        if let Some(value) = value {
            self.str(key, value);
        }
    }

    fn int(&mut self, key: &str, value: u64) {
        self.node(key, Node::Leaf(Leaf::Int(value)));
    }

    fn flag(&mut self, key: &str, value: bool) {
        self.node(key, Node::Leaf(Leaf::Flag(value)));
    }

    /// A list keyed by each element's identity. A repeated identity keeps one element; the
    /// validation rejects such a record before it is encoded.
    fn list(&mut self, key: &str, elements: impl Iterator<Item = (String, Node)>) {
        self.node(key, Node::List(elements.collect()));
    }

    fn done(self) -> Node {
        Node::Table(self.0)
    }
}

fn encode(node: &Node, out: &mut Vec<u8>) {
    fn string(text: &str, out: &mut Vec<u8>) {
        out.push(b's');
        out.extend_from_slice(text.len().to_string().as_bytes());
        out.push(b':');
        out.extend_from_slice(text.as_bytes());
    }
    match node {
        Node::Leaf(Leaf::Str(text)) => string(text, out),
        Node::Leaf(Leaf::Int(n)) => {
            out.push(b'i');
            out.extend_from_slice(n.to_string().as_bytes());
            out.push(b';');
        }
        Node::Leaf(Leaf::Flag(b)) => out.push(if *b { b't' } else { b'f' }),
        Node::Table(entries) | Node::List(entries) => {
            out.push(if matches!(node, Node::Table(_)) {
                b'm'
            } else {
                b'k'
            });
            out.extend_from_slice(entries.len().to_string().as_bytes());
            out.push(b':');
            // A `BTreeMap<String, _>` iterates in byte order of its keys on every platform.
            for (key, value) in entries {
                string(key, out);
                encode(value, out);
            }
        }
    }
}

/// Every leaf of `node`, by its field path: `cpu.microcode`, `memory[DIMM_A2].speed_mts`,
/// `bios.other[c_states]`.
fn leaves(node: &Node) -> BTreeMap<String, Leaf> {
    fn walk(node: &Node, path: &str, out: &mut BTreeMap<String, Leaf>) {
        match node {
            Node::Leaf(leaf) => {
                out.insert(path.to_owned(), leaf.clone());
            }
            Node::Table(entries) => {
                for (key, value) in entries {
                    let path = if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{path}.{key}")
                    };
                    walk(value, &path, out);
                }
            }
            Node::List(entries) => {
                for (key, value) in entries {
                    walk(value, &format!("{path}[{key}]"), out);
                }
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(node, "", &mut out);
    out
}

/// One field where a rig as observed differs from its record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mismatch {
    /// The field's path, such as `memory[DIMM_A2].speed_mts` or `gpu[PCIEX16_1].driver`.
    pub field: String,
    /// The record's value; `None` when the record has no such field.
    pub recorded: Option<String>,
    /// The observed value; `None` when the rig has no such field.
    pub observed: Option<String>,
}

impl fmt::Display for Mismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shown = |value: &Option<String>| match value {
            Some(value) => format!("`{value}`"),
            None => "nothing".to_owned(),
        };
        write!(
            f,
            "`{}`: the record has {}, the rig has {}",
            self.field,
            shown(&self.recorded),
            shown(&self.observed)
        )
    }
}

/// Every field where `observed` differs from `record`, in field order; empty when they match.
/// A list element missing on one side shows as its fields with no value there. Both rigs must be
/// valid ([`Rig::problems`]).
pub fn diff(record: &Rig, observed: &Rig) -> Vec<Mismatch> {
    let recorded = leaves(&record.node());
    let seen = leaves(&observed.node());
    let fields: BTreeSet<&String> = recorded.keys().chain(seen.keys()).collect();
    fields
        .into_iter()
        .filter_map(|field| {
            let a = recorded.get(field);
            let b = seen.get(field);
            (a != b).then(|| Mismatch {
                field: field.clone(),
                recorded: a.map(Leaf::to_string),
                observed: b.map(Leaf::to_string),
            })
        })
        .collect()
}

/// How a rig's baseline epoch changed between two revisions of the inventory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EpochChange {
    /// A new record: the rig's first epoch.
    Added {
        rig: String,
        fingerprint: RigFingerprint,
    },
    /// The record was removed.
    Removed {
        rig: String,
        fingerprint: RigFingerprint,
    },
    /// The record changed, so a new epoch starts. `fields` names every field that changed.
    Changed {
        rig: String,
        from: RigFingerprint,
        to: RigFingerprint,
        fields: Vec<String>,
    },
}

impl fmt::Display for EpochChange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EpochChange::Added { rig, fingerprint } => {
                write!(f, "{rig}: added; first epoch {}", fingerprint.0)
            }
            EpochChange::Removed { rig, fingerprint } => {
                write!(f, "{rig}: removed; last epoch {}", fingerprint.0)
            }
            EpochChange::Changed {
                rig,
                from,
                to,
                fields,
            } => write!(
                f,
                "{rig}: new epoch {} (was {}); changed: {}",
                to.0,
                from.0,
                fields.join(", ")
            ),
        }
    }
}

/// The epoch changes from `old` to `new`, by rig ID. A rig whose record changed only in
/// formatting or list order keeps its fingerprint and is not listed.
pub fn epochs(old: &Inventory, new: &Inventory) -> Vec<EpochChange> {
    let ids: BTreeSet<&str> = old
        .rigs
        .iter()
        .chain(&new.rigs)
        .map(|rig| rig.id.as_str())
        .collect();
    let mut changes = Vec::new();
    for id in ids {
        match (old.rig(id), new.rig(id)) {
            (None, Some(rig)) => changes.push(EpochChange::Added {
                rig: id.to_owned(),
                fingerprint: rig.fingerprint(),
            }),
            (Some(rig), None) => changes.push(EpochChange::Removed {
                rig: id.to_owned(),
                fingerprint: rig.fingerprint(),
            }),
            (Some(before), Some(after)) => {
                let (from, to) = (before.fingerprint(), after.fingerprint());
                if from != to {
                    changes.push(EpochChange::Changed {
                        rig: id.to_owned(),
                        from,
                        to,
                        fields: diff(before, after).into_iter().map(|m| m.field).collect(),
                    });
                }
            }
            (None, None) => {}
        }
    }
    changes
}

#[cfg(test)]
mod tests {
    use super::*;

    const RIG: &str = r#"
version = 1

[[rig]]
id = "R-TEST"
os_image = "lab-linux-1"
kernel = "6.8.0"
power_plan = "performance"
room_sensor = "room-1"

[rig.cpu]
model = "Test CPU"
microcode = "0x1"

[[rig.memory]]
slot = "B"
size_gib = 32
speed_mts = 6000
part = "P2"

[[rig.memory]]
slot = "A"
size_gib = 32
speed_mts = 6000
part = "P1"

[[rig.nvme]]
slot = "M2_1"
model = "Test NVMe"
firmware = "1.0"
attachment = "cpu"
pcie = "gen4x4"

[rig.bios]
version = "1.0"
memory_profile = "XMP"
power_limits = "default"
smt = true
resizable_bar = true
other = { c_states = "on" }
"#;

    fn rig() -> Rig {
        Inventory::parse(RIG).unwrap().rigs.remove(0)
    }

    #[test]
    fn the_canonical_encoding_is_tagged_sorted_and_length_prefixed() {
        let text = String::from_utf8(rig().canonical()).unwrap();
        assert_eq!(
            text,
            "m11:\
             s4:biosm6:s14:memory_profiles3:XMPs5:otherk1:s8:c_statess2:ons12:power_limitss7:defaults13:resizable_barts3:smtts7:versions3:1.0\
             s3:cpum2:s9:microcodes3:0x1s5:models8:Test CPU\
             s3:gpuk0:\
             s2:ids6:R-TEST\
             s6:kernels5:6.8.0\
             s6:memoryk2:s1:Am4:s4:parts2:P1s8:size_gibi32;s4:slots1:As9:speed_mtsi6000;\
             s1:Bm4:s4:parts2:P2s8:size_gibi32;s4:slots1:Bs9:speed_mtsi6000;\
             s3:nick0:\
             s4:nvmek1:s4:M2_1m5:s10:attachments3:cpus8:firmwares3:1.0s5:models9:Test NVMes4:pcies6:gen4x4s4:slots4:M2_1\
             s8:os_images11:lab-linux-1\
             s10:power_plans11:performance\
             s11:room_sensors6:room-1"
        );
    }

    #[test]
    fn list_order_and_formatting_do_not_change_the_fingerprint() {
        let reordered = RIG.replacen("slot = \"B\"", "slot = \"TMP\"", 1);
        let reordered = reordered.replacen("slot = \"A\"", "slot = \"B\"", 1);
        let reordered = reordered.replacen("slot = \"TMP\"", "slot = \"A\"", 1);
        let reordered = reordered
            .replacen("part = \"P2\"", "part = \"TMP\"", 1)
            .replacen("part = \"P1\"", "part = \"P2\"", 1)
            .replacen("part = \"TMP\"", "part = \"P1\"", 1);
        let crlf = RIG.replace('\n', "\r\n");
        let commented = RIG.replace("kernel = ", "# a comment\nkernel   =   ");
        for text in [reordered, crlf, commented] {
            let other = Inventory::parse(&text).unwrap().rigs.remove(0);
            assert_eq!(other.fingerprint(), rig().fingerprint(), "{text}");
        }
    }

    #[test]
    fn an_optional_field_counts_when_present() {
        let without = RIG.replace("microcode = \"0x1\"\n", "");
        let other = Inventory::parse(&without).unwrap().rigs.remove(0);
        assert_ne!(other.fingerprint(), rig().fingerprint());
        assert_eq!(
            diff(&rig(), &other),
            [Mismatch {
                field: "cpu.microcode".to_owned(),
                recorded: Some("0x1".to_owned()),
                observed: None,
            }]
        );
    }

    #[test]
    fn identical_rigs_have_no_mismatch() {
        assert_eq!(diff(&rig(), &rig()), []);
    }

    #[test]
    fn invalid_records_are_named() {
        for (from, to, expected) in [
            ("id = \"R-TEST\"", "id = \"ada\"", "a rig ID is `R-`"),
            (
                "slot = \"B\"",
                "slot = \"A\"",
                "`memory`: the slot `A` appears twice",
            ),
            (
                "size_gib = 32",
                "size_gib = 0",
                "`memory[B]`: `size_gib` and `speed_mts`",
            ),
            (
                "kernel = \"6.8.0\"",
                "kernel = \" 6.8.0\"",
                "`kernel` is ` 6.8.0`",
            ),
            ("part = \"P1\"", "part = \"\"", "`memory[A].part` is ``"),
            (
                "pcie = \"gen4x4\"",
                "pcie = \"gen4x3\"",
                "`gen4x3` is not a PCIe link",
            ),
            (
                "attachment = \"cpu\"",
                "attachment = \"cpu0\"",
                "unknown variant `cpu0`",
            ),
            ("version = 1", "version = 2", "`version` is 2"),
            (
                "smt = true",
                "smt = true\nhyperthreads = 2",
                "unknown field `hyperthreads`",
            ),
        ] {
            let text = RIG.replacen(from, to, 1);
            assert_ne!(text, RIG, "`{from}` not found");
            let error = Inventory::parse(&text).unwrap_err().to_string();
            assert!(error.contains(expected), "{to}: {error}");
        }
        let twice = format!("{RIG}{}", &RIG[RIG.find("[[rig]]").unwrap()..]);
        let error = Inventory::parse(&twice).unwrap_err().to_string();
        assert!(
            error.contains("rig `R-TEST`: the ID is used twice"),
            "{error}"
        );
    }

    #[test]
    fn pcie_links_are_generation_and_width() {
        assert_eq!("gen5x16".parse(), Ok(PcieLink { gen_: 5, width: 16 }));
        assert_eq!(PcieLink { gen_: 3, width: 4 }.to_string(), "gen3x4");
        for bad in [
            "gen0x16", "gen4x3", "gen4", "4x16", "gen4x16x", "gen+4x16", "genx16",
        ] {
            assert!(bad.parse::<PcieLink>().is_err(), "{bad}");
        }
    }

    #[test]
    fn rig_ids_follow_the_lab_names() {
        for good in ["R-ADA", "R-ULTRA-A", "R-PI5", "R-AM5-4D"] {
            assert!(is_rig_id(good), "{good}");
        }
        for bad in ["", "R-", "R-ada", "ADA", "R--A", "R-A-", "R-A B"] {
            assert!(!is_rig_id(bad), "{bad}");
        }
    }
}
