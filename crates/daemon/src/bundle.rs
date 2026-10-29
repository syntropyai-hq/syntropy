// SPDX-License-Identifier: MIT OR Apache-2.0

//! The diagnostic bundle: `syntropy doctor --bundle`, alias `syntropy debug dump`
//! (RFC-0053 §6; ADR-0510 decision 4; PROD-E9-T04).
//!
//! [`build`] writes one `tar.zst` in memory from the [`Sources`] it is given. Nothing is uploaded,
//! and nothing is sent anywhere: the caller writes the file ([`Bundle::write_new`]) and prints
//! [`Bundle::finish_message`]. The command and its flags are PROD-E6-T08's, and the API route,
//! `POST /v1/syntropy/debug/dump` with `admin` scope, needs PROD-E5-T01's scopes; both call this.
//!
//! # What goes in, and how it is kept safe
//!
//! - **An allow-list of record types.** [`Sources`] has one field per record type the bundle
//!   takes. A record type not in it cannot reach a bundle, so a new one is excluded until someone
//!   adds it deliberately.
//! - **An allow-list of fields.** Each record is serialised, then projected through its shape
//!   (the private `shape` module): a field or enum variant not on the list is left out, and its
//!   name, never its value, is listed in `MANIFEST.txt`. Configuration values in the cluster state are replaced by
//!   `[redacted]`, since a cluster key may be secret-typed; `config.txt` shows them with
//!   `config show`'s own redaction of secret-typed keys.
//! - **No content.** Log records keep no content field: flight-recorder records are rendered with
//!   content arguments as `[content]`, and crash records hold only content-free lines
//!   ([`crate::crash::LogTail`]). Prompts, outputs, token IDs, KV pages, weights, API keys, Hub
//!   tokens, the node's secret key, invite tokens, TLS private keys and stored Responses have no
//!   record type here.
//! - **Addresses.** Every string kept, and every map key, passes the address redactor (the
//!   private `redact` module): IP addresses wherever they occur in a string, inside longer
//!   tokens too, and URLs become stable per-bundle placeholders unless `--with-addresses` is
//!   given, and every place one was replaced is listed in `MANIFEST.txt`. A URL's userinfo is a
//!   credential, so it is removed even with `--with-addresses`. Records are serialised
//!   human-readable (the private `json` module), so an address type reaches the redactor as text.
//! - **Size.** At most [`DEFAULT_MAX_SIZE`] (200 MiB) unless `--max-size` says otherwise. The
//!   flight recorder is truncated first, oldest entries first, then the cluster log; a bundle that
//!   still does not fit is [`BundleError::TooLarge`]. The cap is checked against the archive
//!   before compression, which bounds the file.
//!
//! # The files
//!
//! ```text
//! MANIFEST.txt            every other file with its size and SHA-256, the redactions, what was
//!                         left out, truncated or not collected
//! build.json              version, build ID, protocol majors, support tier
//! config.txt              the resolved configuration with origins, secrets redacted
//! cluster-state.json      the whole-state view of ADR-0009
//! cluster-log.jsonl       the last 10,000 log entries, one per line
//! flight-recorder.jsonl   spans and log records, no content, one per line
//! doctor.json             the doctor report
//! crashes/<name>.json     the crash records
//! ```
//!
//! Not collected yet, and listed as such: OS facts and the resource graph and capability profiles
//! (the probe's export), committed plans' explanations and `PredictedCost`, runner residency
//! and cache statistics, and the metrics history. Each joins when its source exists, through a
//! new field of [`Sources`] and a shape. `--cluster`, which nests every member's bundle, needs the
//! node-query channel.

mod json;
mod redact;
mod shape;
pub mod tar;

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::io;
use std::path::Path;

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use syntropy_api::version::VersionInfo;
use syntropy_cluster::log::{ClusterState, LogEntry};
use syntropy_config::ResolvedConfig;
use syntropy_obs::log::Callsite;
use syntropy_obs::trace::FlightRecorder;
use syntropy_obs::trace::flight::Entry;

use crate::crash::CrashStore;
use crate::doctor::Report;
use crate::logging::{self, Context};

use redact::Redactor;
use shape::{Omitted, Shape, project};
use tar::TarFile;

/// The default size cap: 200 MiB (RFC-0053 §6).
pub const DEFAULT_MAX_SIZE: u64 = 200 << 20;

/// How many of the most recent cluster log entries a bundle holds.
pub const CLUSTER_LOG_ENTRIES: usize = 10_000;

/// The manifest's name.
pub const MANIFEST: &str = "MANIFEST.txt";

/// The zstd level.
const ZSTD_LEVEL: i32 = 3;

/// Room kept under the cap for `MANIFEST.txt` and the archive's headers.
const MANIFEST_ROOM: u64 = 256 << 10;

/// What a bundle is built from: one field per record type on the allow-list. A part that is
/// `None` or empty is listed in `MANIFEST.txt` as not collected.
#[derive(Clone, Copy, Default)]
pub struct Sources<'a> {
    /// The build (`GET /v1/syntropy/version`'s body).
    pub build: Option<&'a VersionInfo>,
    /// The resolved configuration, shown as `config show` shows it.
    pub config: Option<&'a ResolvedConfig>,
    /// The cluster state.
    pub cluster_state: Option<&'a ClusterState>,
    /// The cluster log, oldest first; the last [`CLUSTER_LOG_ENTRIES`] are taken.
    pub cluster_log: &'a [LogEntry],
    /// The flight recorder.
    pub flight: Option<&'a FlightRecorder>,
    /// The build's hot-thread callsites, to render the flight recorder's log records. A record
    /// whose callsite is not here keeps its callsite number and level only.
    pub callsites: &'a [Callsite],
    /// The doctor report.
    pub doctor: Option<&'a Report>,
    /// The crash records.
    pub crashes: Option<&'a CrashStore>,
}

impl std::fmt::Debug for Sources<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sources")
            .field("build", &self.build.is_some())
            .field("config", &self.config.is_some())
            .field("cluster_state", &self.cluster_state.is_some())
            .field("cluster_log", &self.cluster_log.len())
            .field("flight", &self.flight.map(FlightRecorder::len))
            .field("doctor", &self.doctor.is_some())
            .field("crashes", &self.crashes.map(CrashStore::dir))
            .finish()
    }
}

/// How a bundle is built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BundleOptions {
    /// `--with-addresses`: keep addresses instead of replacing them.
    pub with_addresses: bool,
    /// `--max-size`, in bytes.
    pub max_size: u64,
    /// When the bundle is made, in milliseconds since the Unix epoch, from the daemon's `Clock`.
    pub created_unix_ms: u64,
}

impl BundleOptions {
    /// The defaults at `created_unix_ms`: addresses replaced, [`DEFAULT_MAX_SIZE`].
    #[must_use]
    pub fn at(created_unix_ms: u64) -> BundleOptions {
        BundleOptions {
            with_addresses: false,
            max_size: DEFAULT_MAX_SIZE,
            created_unix_ms,
        }
    }
}

/// Why a bundle could not be built.
#[non_exhaustive]
#[derive(Debug)]
pub enum BundleError {
    /// It does not fit the size cap even after truncation.
    TooLarge {
        /// Its size, in bytes, before compression after truncation, or after it.
        size: u64,
        /// The cap.
        cap: u64,
    },
    /// A record could not be serialised.
    Encode {
        /// The file it was for.
        file: &'static str,
        /// What failed.
        message: String,
    },
    /// The crash records could not be listed.
    Crashes(io::Error),
    /// Compressing failed.
    Compress(io::Error),
}

impl std::fmt::Display for BundleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BundleError::TooLarge { size, cap } => write!(
                f,
                "the bundle needs {size} bytes even with the flight recorder and the cluster log \
                 truncated, over the {cap}-byte cap; raise it with --max-size"
            ),
            BundleError::Encode { file, message } => write!(f, "cannot write {file}: {message}"),
            BundleError::Crashes(e) => write!(f, "cannot list the crash records: {e}"),
            BundleError::Compress(e) => write!(f, "cannot compress the bundle: {e}"),
        }
    }
}

impl std::error::Error for BundleError {}

/// A built bundle: the `tar.zst` bytes.
#[derive(Clone, Debug)]
pub struct Bundle {
    bytes: Vec<u8>,
    sha256: [u8; 32],
    manifest: String,
}

impl Bundle {
    /// The `tar.zst` file's bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Its size in bytes.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.bytes.len() as u64
    }

    /// Its SHA-256, as hex.
    #[must_use]
    pub fn sha256_hex(&self) -> String {
        hex(&self.sha256)
    }

    /// `MANIFEST.txt`'s text.
    #[must_use]
    pub fn manifest(&self) -> &str {
        &self.manifest
    }

    /// Writes the bundle to a new owner-only file at `path`. An existing file is never
    /// overwritten.
    ///
    /// # Errors
    ///
    /// [`io::ErrorKind::AlreadyExists`] when `path` exists, and the OS error otherwise.
    pub fn write_new(&self, path: &Path) -> io::Result<()> {
        syntropy_io::os::create_private_file(path, &self.bytes)
    }

    /// What the command prints when it is done: the path, size and SHA-256, and a reminder to
    /// read `MANIFEST.txt` before sharing (RFC-0053 §6).
    #[must_use]
    pub fn finish_message(&self, path: &Path) -> String {
        format!(
            "wrote {} ({} bytes, SHA-256 {})\nread {MANIFEST} inside it before you share it: it \
             lists every file, what was redacted and what was left out",
            path.display(),
            self.size(),
            self.sha256_hex()
        )
    }
}

/// One file being assembled, as lines when it is truncated line by line.
struct Part {
    name: String,
    lines: Vec<String>,
    /// Joined with newlines and ended with one when true; written as the single line otherwise.
    jsonl: bool,
}

impl Part {
    fn text(name: impl Into<String>, text: String) -> Part {
        Part {
            name: name.into(),
            lines: vec![text],
            jsonl: false,
        }
    }

    fn bytes(&self) -> Vec<u8> {
        if self.jsonl {
            let mut out = String::new();
            for line in &self.lines {
                out.push_str(line);
                out.push('\n');
            }
            out.into_bytes()
        } else {
            self.lines.concat().into_bytes()
        }
    }

    fn len(&self) -> u64 {
        if self.jsonl {
            self.lines.iter().map(|l| l.len() as u64 + 1).sum()
        } else {
            self.lines.iter().map(|l| l.len() as u64).sum()
        }
    }
}

/// A JSON part: `value` through `shape`, pretty-printed.
fn json_part(
    name: &'static str,
    value: Value,
    shape: &Shape,
    redactor: &mut Redactor,
    omitted: &mut Omitted,
) -> Result<Part, BundleError> {
    let kept = project(value, shape, name, "", redactor, omitted);
    let text = serde_json::to_string_pretty(&kept).map_err(|e| BundleError::Encode {
        file: name,
        message: e.to_string(),
    })?;
    Ok(Part::text(name, text + "\n"))
}

/// The size of a ustar archive of files of these lengths.
fn archive_len(lens: impl Iterator<Item = u64>) -> u64 {
    lens.map(|len| 512 + len.div_ceil(512) * 512).sum::<u64>() + 1024
}

/// `value` as JSON through the bundle's human-readable serializer ([`json`]): a map key that is
/// not a string, such as a `NodeId`, is written as text, and an address type as its text form.
fn to_json<T: Serialize + ?Sized>(value: &T, file: &'static str) -> Result<Value, BundleError> {
    json::to_value(value).map_err(|e| BundleError::Encode {
        file,
        message: e.to_string(),
    })
}

/// One cluster log entry as the bundle writes it, before projection.
fn log_entry_value(entry: &LogEntry) -> Result<Value, BundleError> {
    let mut object = serde_json::Map::new();
    object.insert("index".into(), Value::from(entry.index));
    object.insert("term".into(), Value::from(entry.term));
    object.insert("at_ms".into(), Value::from(entry.at_ms));
    object.insert("cause".into(), to_json(&entry.cause, "cluster-log.jsonl")?);
    object.insert("event".into(), to_json(&entry.event, "cluster-log.jsonl")?);
    Ok(Value::Object(object))
}

/// One flight-recorder entry as the bundle writes it, before projection. A log record's
/// arguments are kept only through its callsite, which says which are content; content shows as
/// `[content]` in the message and is not a field.
fn flight_value(entry: &Entry, callsites: &[Callsite]) -> Value {
    use serde_json::json;
    match entry {
        Entry::Span(s) => json!({"span": {
            "trace_id": hex(&s.trace_id),
            "span_id": hex(&s.span_id),
            "parent_id": hex(&s.parent_id),
            "start_ns": s.start.0,
            "dur_ns": s.dur,
            "name": s.name.0,
            "attrs": &s.attrs[..usize::from(s.attr_count.min(4))],
        }}),
        Entry::Log(record) => match logging::from_record(record, callsites, Context::default()) {
            Some(event) => {
                let fields: serde_json::Map<String, Value> = event
                    .fields
                    .iter()
                    .filter(|f| !f.content)
                    .map(|f| (f.name.to_string(), Value::String(f.value.to_string())))
                    .collect();
                json!({"log": {
                    "at_ns": record.at.0,
                    "level": record.level.name(),
                    "callsite": record.callsite.0,
                    "target": event.target,
                    "message": event.message,
                    "fields": fields,
                }})
            }
            None => json!({"log": {
                "at_ns": record.at.0,
                "level": record.level.name(),
                "callsite": record.callsite.0,
            }}),
        },
    }
}

/// Builds the bundle from `sources` (module docs).
///
/// # Errors
///
/// - [`BundleError::TooLarge`] when it does not fit `options.max_size` after truncation.
/// - [`BundleError::Encode`] when a record cannot be serialised.
/// - [`BundleError::Crashes`] when the crash directory cannot be listed.
/// - [`BundleError::Compress`] when compression fails.
pub fn build(sources: &Sources<'_>, options: &BundleOptions) -> Result<Bundle, BundleError> {
    let mut redactor = Redactor::new(options.with_addresses);
    let mut omitted = Omitted::default();
    let mut parts: Vec<Part> = Vec::new();
    let mut not_collected: Vec<&str> = vec![
        "OS facts, the resource graph and capability profiles: the probe's export",
        "committed plans' explanations and PredictedCost",
        "runner residency digests and cache statistics",
        "the metrics history",
    ];
    let mut unreadable_crashes = Vec::new();

    match sources.build {
        Some(build) => {
            let value = serde_json::to_value(build).map_err(|e| BundleError::Encode {
                file: "build.json",
                message: e.to_string(),
            })?;
            parts.push(json_part(
                "build.json",
                value,
                &shape::BUILD,
                &mut redactor,
                &mut omitted,
            )?);
        }
        None => not_collected.push("build.json: no build information was given"),
    }
    match sources.config {
        Some(config) => {
            let text = syntropy_config::command::show_resolved(config);
            let text = redactor.scan(&text, "config.txt: (text)");
            omitted
                .secrets
                .insert("config.txt: every secret-typed key (config show)".to_owned());
            parts.push(Part::text("config.txt", text));
        }
        None => not_collected.push("config.txt: no configuration was given"),
    }
    match sources.cluster_state {
        Some(state) => {
            let value = to_json(state, "cluster-state.json")?;
            parts.push(json_part(
                "cluster-state.json",
                value,
                &shape::CLUSTER_STATE,
                &mut redactor,
                &mut omitted,
            )?);
        }
        None => not_collected.push("cluster-state.json: this node has no cluster state"),
    }
    let log = sources.cluster_log;
    let log = &log[log.len().saturating_sub(CLUSTER_LOG_ENTRIES)..];
    let mut lines = Vec::with_capacity(log.len());
    for entry in log {
        let value = log_entry_value(entry)?;
        let kept = project(
            value,
            &shape::CLUSTER_LOG_ENTRY,
            "cluster-log.jsonl",
            "",
            &mut redactor,
            &mut omitted,
        );
        lines.push(kept.to_string());
    }
    let log_total = lines.len();
    parts.push(Part {
        name: "cluster-log.jsonl".into(),
        lines,
        jsonl: true,
    });
    let mut flight_total = 0;
    match sources.flight {
        Some(flight) => {
            let mut lines = Vec::with_capacity(flight.len());
            for entry in flight.entries() {
                let kept = project(
                    flight_value(entry, sources.callsites),
                    &shape::FLIGHT_ENTRY,
                    "flight-recorder.jsonl",
                    "",
                    &mut redactor,
                    &mut omitted,
                );
                lines.push(kept.to_string());
            }
            flight_total = lines.len();
            parts.push(Part {
                name: "flight-recorder.jsonl".into(),
                lines,
                jsonl: true,
            });
        }
        None => not_collected.push("flight-recorder.jsonl: no flight recorder was given"),
    }
    match sources.doctor {
        Some(report) => {
            let value = serde_json::to_value(report).map_err(|e| BundleError::Encode {
                file: "doctor.json",
                message: e.to_string(),
            })?;
            parts.push(json_part(
                "doctor.json",
                value,
                &shape::DOCTOR_REPORT,
                &mut redactor,
                &mut omitted,
            )?);
        }
        None => not_collected.push("doctor.json: no doctor report was given"),
    }
    if let Some(store) = sources.crashes {
        let listing = store.list().map_err(BundleError::Crashes)?;
        unreadable_crashes = listing.unreadable;
        for (name, record) in listing.records.into_iter().rev() {
            let value = serde_json::to_value(&record).map_err(|e| BundleError::Encode {
                file: "crashes/",
                message: e.to_string(),
            })?;
            let file = format!("crashes/{name}");
            let kept = project(
                value,
                &shape::CRASH_RECORD,
                &file,
                "",
                &mut redactor,
                &mut omitted,
            );
            let text = serde_json::to_string_pretty(&kept).map_err(|e| BundleError::Encode {
                file: "crashes/",
                message: e.to_string(),
            })?;
            parts.push(Part::text(file, text + "\n"));
        }
    }

    // The size cap: the flight recorder first, then the cluster log, oldest entries first.
    let budget = options.max_size.saturating_sub(MANIFEST_ROOM);
    let mut truncated = Vec::new();
    for name in ["flight-recorder.jsonl", "cluster-log.jsonl"] {
        let total = archive_len(parts.iter().map(Part::len));
        if total <= budget {
            break;
        }
        let Some(part) = parts.iter_mut().find(|p| p.name == name) else {
            continue;
        };
        let mut over = total - budget;
        let mut drop = 0;
        while drop < part.lines.len() && over > 0 {
            over = over.saturating_sub(part.lines[drop].len() as u64 + 1);
            drop += 1;
        }
        // Whole 512-byte blocks: one more line may be needed after rounding.
        part.lines.drain(..drop);
        while archive_len(parts.iter().map(Part::len)) > budget {
            let part = parts
                .iter_mut()
                .find(|p| p.name == name)
                .expect("found above");
            if part.lines.is_empty() {
                break;
            }
            part.lines.remove(0);
            drop += 1;
        }
        let of = if name == "flight-recorder.jsonl" {
            flight_total
        } else {
            log_total
        };
        truncated.push(format!(
            "{name}: the oldest {drop} of {of} entries left out"
        ));
    }
    let content_len = archive_len(parts.iter().map(Part::len));
    if content_len > budget {
        return Err(BundleError::TooLarge {
            size: content_len,
            cap: options.max_size,
        });
    }

    let files: Vec<TarFile> = parts
        .iter()
        .map(|p| TarFile {
            name: p.name.clone(),
            bytes: p.bytes(),
        })
        .collect();
    let manifest = manifest_text(
        options,
        &files,
        &redactor,
        &omitted,
        &truncated,
        &not_collected,
        &unreadable_crashes,
    );
    let mut all = Vec::with_capacity(files.len() + 1);
    all.push(TarFile {
        name: MANIFEST.into(),
        bytes: manifest.clone().into_bytes(),
    });
    all.extend(files);
    let archive = tar::write(&all, options.created_unix_ms / 1000);
    let bytes = zstd::encode_all(&archive[..], ZSTD_LEVEL).map_err(BundleError::Compress)?;
    if bytes.len() as u64 > options.max_size {
        return Err(BundleError::TooLarge {
            size: bytes.len() as u64,
            cap: options.max_size,
        });
    }
    let sha256 = Sha256::digest(&bytes).into();
    Ok(Bundle {
        bytes,
        sha256,
        manifest,
    })
}

fn manifest_text(
    options: &BundleOptions,
    files: &[TarFile],
    redactor: &Redactor,
    omitted: &Omitted,
    truncated: &[String],
    not_collected: &[&str],
    unreadable_crashes: &[String],
) -> String {
    let mut out = String::new();
    let created = logging::format::utc(i128::from(options.created_unix_ms) * 1_000);
    let _ = writeln!(out, "Syntropy diagnostic bundle (RFC-0053 §6)");
    let _ = writeln!(out, "created: {created}");
    let _ = writeln!(out, "size cap: {} bytes", options.max_size);
    if redactor.keeps() {
        let _ = writeln!(out, "addresses: kept (--with-addresses)");
    } else {
        let (addrs, urls) = redactor.counts();
        let _ = writeln!(
            out,
            "addresses: replaced by placeholders ({addrs} addresses as <addr-N>, {urls} URLs as \
             <url-N>; the same address has the same placeholder throughout this bundle)"
        );
    }
    let _ = writeln!(out, "\nfiles (SHA-256, bytes, path):");
    for file in files {
        let _ = writeln!(
            out,
            "{}  {}  {}",
            sha256_hex(&file.bytes),
            file.bytes.len(),
            file.name
        );
    }
    let section = |out: &mut String, title: &str, items: &mut dyn Iterator<Item = String>| {
        let _ = writeln!(out, "\n{title}:");
        let mut any = false;
        for item in items {
            let _ = writeln!(out, "  {item}");
            any = true;
        }
        if !any {
            let _ = writeln!(out, "  none");
        }
    };
    section(
        &mut out,
        "addresses replaced in (file: field)",
        &mut redactor.fields.iter().cloned(),
    );
    section(
        &mut out,
        "secrets replaced by [redacted] in (file: field)",
        &mut omitted.secrets.iter().cloned().chain(
            redactor
                .credentials
                .iter()
                .map(|at| format!("{at} (URL userinfo)")),
        ),
    );
    section(
        &mut out,
        "left out, not on the allow-list (file: field)",
        &mut omitted.fields.iter().cloned(),
    );
    section(
        &mut out,
        "truncated for the size cap",
        &mut truncated.iter().cloned(),
    );
    section(
        &mut out,
        "crash records that could not be read, left out",
        &mut unreadable_crashes.iter().cloned(),
    );
    section(
        &mut out,
        "not collected",
        &mut not_collected.iter().map(|s| (*s).to_owned()),
    );
    let _ = writeln!(
        out,
        "\nnever included, whatever the flags: prompts, outputs, token IDs, KV pages, weights, API \
         keys, Hub tokens, the node's secret key, invite tokens, TLS private keys, stored Responses"
    );
    out
}

/// Checks a bundle: it decompresses, its archive reads, `MANIFEST.txt` lists every other file
/// exactly once, and each file's size and SHA-256 match. Returns the files' names.
///
/// # Errors
///
/// What failed, naming the file.
pub fn verify(bundle: &[u8]) -> Result<Vec<String>, String> {
    let archive = zstd::decode_all(bundle).map_err(|e| format!("not zstd: {e}"))?;
    let files = tar::read(&archive).map_err(str::to_owned)?;
    let manifest = files
        .iter()
        .find(|f| f.name == MANIFEST)
        .ok_or("no MANIFEST.txt")?;
    let text = std::str::from_utf8(&manifest.bytes).map_err(|_| "MANIFEST.txt is not UTF-8")?;
    let listed: Vec<(&str, u64, &str)> = text
        .split("\nfiles (SHA-256, bytes, path):\n")
        .nth(1)
        .ok_or("MANIFEST.txt has no file list")?
        .lines()
        .take_while(|l| !l.is_empty())
        .map(|l| {
            let mut words = l.splitn(3, "  ");
            let (Some(sha), Some(len), Some(name)) = (words.next(), words.next(), words.next())
            else {
                return Err(format!("a malformed MANIFEST.txt line: {l}"));
            };
            let len = len
                .parse()
                .map_err(|_| format!("a malformed size in MANIFEST.txt: {l}"))?;
            Ok((sha, len, name))
        })
        .collect::<Result<_, _>>()?;
    let mut seen = BTreeSet::new();
    for (sha, len, name) in &listed {
        if !seen.insert(*name) {
            return Err(format!("{name} is listed twice"));
        }
        let file = files
            .iter()
            .find(|f| f.name == *name)
            .ok_or_else(|| format!("{name} is listed but not in the bundle"))?;
        if file.bytes.len() as u64 != *len {
            return Err(format!("{name}'s size does not match MANIFEST.txt"));
        }
        if sha256_hex(&file.bytes) != *sha {
            return Err(format!("{name}'s SHA-256 does not match MANIFEST.txt"));
        }
    }
    for file in &files {
        if file.name != MANIFEST && !seen.contains(file.name.as_str()) {
            return Err(format!("{} is in the bundle but not listed", file.name));
        }
    }
    Ok(files.into_iter().map(|f| f.name).collect())
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests;
