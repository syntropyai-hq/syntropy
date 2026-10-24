// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compile flags per unit and target, the kernel object cache keys, and the contraction check
//! (RFC-0040 §10.2 rules 2, 3, 6 and 7; ADR-0623 decision 1; ADR-0400 decisions 1 and 7).
//!
//! The keys follow ADR-0623 decision 1:
//! - `SourceKey = BLAKE3(version tag ‖ toolchain_id ‖ target ‖ normalised flags ‖ source)`;
//! - `ObjectKey = BLAKE3(SourceKey ‖ include hashes ‖ instantiation list)`.
//!
//! Each field is framed by its length, so no two inputs hash the same by moving bytes across a
//! field boundary. A bit-exact-class unit's flags include `--fmad=false`, so the flag is part of
//! its `SourceKey` and so of its `ObjectKey` (rule 7). The instantiation list is part of the
//! `ObjectKey`, so adding an instantiation misses the cache (rule 2).
//!
//! The cache store, the include manifest and `cargo xtask kcache` are QUAL-E16-T04's, and the
//! build script that calls them is COMP-E9-T02's. This module fixes what goes into the keys.

use std::collections::BTreeSet;
use std::fmt;

use super::units::Unit;

/// The cache's version tag, the first field of every `SourceKey`.
pub const KEY_VERSION: &str = "syntropy-kcache-v1";

/// The CUDA targets of RFC-0040 §7.1: one cubin per `sm_*` target and PTX for `compute_120`.
pub const TARGETS: &[&str] = &[
    "sm_75",
    "sm_80",
    "sm_86",
    "sm_89",
    "sm_90",
    "sm_100",
    "sm_120",
    "sm_121",
    "compute_120",
];

/// Flags no kernel compile may carry: fast math, flush-to-zero, approximate division or square
/// root (RFC-0040 §10.2 rule 6; RFC-0042 §4.1).
pub const FORBIDDEN_FLAGS: &[&str] = &[
    "--use_fast_math",
    "-use_fast_math",
    "--ftz=true",
    "-ftz=true",
    "--prec-div=false",
    "-prec-div=false",
    "--prec-sqrt=false",
    "-prec-sqrt=false",
];

/// The flags that turn contraction off, and keep subnormals and exact division and square
/// root, for a bit-exact-class unit (RFC-0040 §10.2 rule 7).
pub const BIT_EXACT_FLAGS: &[&str] = &[
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
];

/// The `nvcc` flags that compile `unit` for `target`, in order.
#[must_use]
pub fn nvcc_flags(unit: &Unit, target: &str) -> Vec<String> {
    let output = if target.starts_with("compute_") {
        "--ptx"
    } else {
        "--cubin"
    };
    let mut flags = vec![
        output.to_owned(),
        format!("--gpu-architecture={target}"),
        "--std=c++17".to_owned(),
        "-O3".to_owned(),
        "--include-path=csrc".to_owned(),
    ];
    if unit.bit_exact {
        flags.extend(BIT_EXACT_FLAGS.iter().map(|f| (*f).to_owned()));
    }
    flags
}

/// The flags as the key hashes them: each trimmed, empty ones dropped, in order. The order is
/// kept because a later `nvcc` flag overrides an earlier one.
#[must_use]
pub fn normalised_flags(flags: &[String]) -> Vec<String> {
    flags
        .iter()
        .map(|f| f.trim().to_owned())
        .filter(|f| !f.is_empty())
        .collect()
}

/// A 32-byte BLAKE3 key.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key(pub [u8; 32]);

impl fmt::Debug for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in &self.0[..8] {
            write!(f, "{b:02x}")?;
        }
        f.write_str("…")
    }
}

fn field(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

/// `SourceKey`: the version tag, the toolchain's full version banner, the target, the
/// normalised flags and the unit's source (ADR-0623 decision 1).
#[must_use]
pub fn source_key(toolchain_id: &str, target: &str, flags: &[String], source: &[u8]) -> Key {
    let mut h = blake3::Hasher::new();
    field(&mut h, KEY_VERSION.as_bytes());
    field(&mut h, toolchain_id.as_bytes());
    field(&mut h, target.as_bytes());
    let flags = normalised_flags(flags);
    h.update(&(flags.len() as u64).to_le_bytes());
    for flag in &flags {
        field(&mut h, flag.as_bytes());
    }
    field(&mut h, source);
    Key(*h.finalize().as_bytes())
}

/// `ObjectKey`: the `SourceKey`, the hashes of the headers the unit includes, in the order the
/// include manifest lists them, and the unit's instantiation list (ADR-0623 decision 1).
#[must_use]
pub fn object_key(source: Key, include_hashes: &[Key], instantiation: &[u8]) -> Key {
    let mut h = blake3::Hasher::new();
    h.update(&source.0);
    h.update(&(include_hashes.len() as u64).to_le_bytes());
    for include in include_hashes {
        h.update(&include.0);
    }
    field(&mut h, instantiation);
    Key(*h.finalize().as_bytes())
}

/// BLAKE3 of a file's bytes, for an include hash.
#[must_use]
pub fn content_hash(bytes: &[u8]) -> Key {
    Key(*blake3::hash(bytes).as_bytes())
}

/// What one unit's compile reads: its source, the hashes of its includes, its instantiation
/// list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnitInputs<'a> {
    /// The unit.
    pub unit: &'a Unit,
    /// `csrc/<unit>.cu`.
    pub source: Vec<u8>,
    /// The hashes of the headers it includes.
    pub includes: Vec<Key>,
    /// `csrc/instantiate/<unit>.inc`.
    pub instantiation: Vec<u8>,
}

/// The record of one compiled object: what the kernel build keeps per unit and target, and what
/// the contraction check reads (RFC-0042 §6.4: "the kernel build records the flag per
/// translation unit").
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuildRecord<'a> {
    /// The unit.
    pub unit: &'a Unit,
    /// The target.
    pub target: String,
    /// The flags it was compiled with.
    pub flags: Vec<String>,
    /// Its `ObjectKey`.
    pub object_key: Key,
}

impl BuildRecord<'_> {
    /// The record of compiling `inputs` for `target` with `flags`.
    #[must_use]
    pub fn new<'a>(
        inputs: &UnitInputs<'a>,
        toolchain_id: &str,
        target: &str,
        flags: Vec<String>,
    ) -> BuildRecord<'a> {
        let source = source_key(toolchain_id, target, &flags, &inputs.source);
        BuildRecord {
            unit: inputs.unit,
            target: target.to_owned(),
            object_key: object_key(source, &inputs.includes, &inputs.instantiation),
            flags,
        }
    }
}

/// A compile the contraction check refuses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FlagViolation {
    /// A forbidden flag in any unit's compile (RFC-0040 §10.2 rule 6).
    FastMath {
        /// The unit.
        unit: &'static str,
        /// The target.
        target: String,
        /// The flag.
        flag: String,
    },
    /// A bit-exact-class unit compiled without contraction off: its last `--fmad` flag is not
    /// `--fmad=false` (RFC-0040 §10.2 rule 7; RFC-0060 §4.2 rule 4).
    ContractionOn {
        /// The unit.
        unit: &'static str,
        /// The target.
        target: String,
    },
}

impl fmt::Display for FlagViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FlagViolation::FastMath { unit, target, flag } => {
                write!(f, "{unit} ({target}): forbidden flag `{flag}`")
            }
            FlagViolation::ContractionOn { unit, target } => write!(
                f,
                "{unit} ({target}): bit-exact-class unit built without `--fmad=false`"
            ),
        }
    }
}

/// The contraction and fast-math check over build records: what L0 runs over the kernel
/// build's records, failing on any violation (RFC-0060 §4.2 rule 4).
#[must_use]
pub fn check_flags(records: &[BuildRecord<'_>]) -> Vec<FlagViolation> {
    let mut out = Vec::new();
    for r in records {
        let flags = normalised_flags(&r.flags);
        for flag in &flags {
            if FORBIDDEN_FLAGS.contains(&flag.as_str()) {
                out.push(FlagViolation::FastMath {
                    unit: r.unit.name,
                    target: r.target.clone(),
                    flag: flag.clone(),
                });
            }
        }
        if r.unit.bit_exact {
            // nvcc contracts by default; the last `--fmad` flag wins.
            let last = flags
                .iter()
                .rev()
                .find(|f| f.starts_with("--fmad") || f.starts_with("-fmad"));
            if !matches!(
                last.map(String::as_str),
                Some("--fmad=false" | "-fmad=false")
            ) {
                out.push(FlagViolation::ContractionOn {
                    unit: r.unit.name,
                    target: r.target.clone(),
                });
            }
        }
    }
    out
}

/// What a build would compile against a cache that holds `cached`: the per-unit, per-target
/// hits and misses (RFC-0040 §10.2 rules 1 to 3).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Objects found in the cache.
    pub hits: usize,
    /// (unit, target) pairs that would compile.
    pub misses: Vec<(&'static str, String)>,
}

/// The keys of every unit × target and how many `cached` already holds.
#[must_use]
pub fn plan(
    units: &[UnitInputs<'_>],
    targets: &[&str],
    toolchain_id: &str,
    cached: &BTreeSet<Key>,
) -> (Vec<Key>, CacheStats) {
    let mut keys = Vec::new();
    let mut stats = CacheStats::default();
    for inputs in units {
        for target in targets {
            let flags = nvcc_flags(inputs.unit, target);
            let record = BuildRecord::new(inputs, toolchain_id, target, flags);
            if cached.contains(&record.object_key) {
                stats.hits += 1;
            } else {
                stats.misses.push((inputs.unit.name, (*target).to_owned()));
            }
            keys.push(record.object_key);
        }
    }
    (keys, stats)
}
