// SPDX-License-Identifier: MIT OR Apache-2.0

//! The report file: one JSON document per run, which the tester reads and submits by hand
//! (ADR-0510). [`Report::check`] is its schema test, and `syntropy-netprobe check <file>` runs it
//! together with the identifier scan of [`crate::scan`].
//!
//! # Format, version 1
//!
//! Every object refuses unknown keys, and every string is one of the enum values listed here, or
//! the probe's version (three numbers joined by dots). So the report has no free text in which an
//! SSID, a host name or an address could appear, and [`crate::scan`] checks the text too. Every
//! number is bounded by what it measures (see "Bounds" below). Times are
//! milliseconds (`_ms`) or microseconds (`_us`) from the moment the probe started connecting.
//!
//! - `format`: `"syntropy-netprobe-report"`. `version`: `1`. `probe_version`: the binary's version.
//! - `outcome`: `"completed"` (the planned duration ran), `"in_progress"` (a checkpoint written
//!   during the run) or `"interrupted"` (the run stopped early).
//! - `settings`: `setting` (`"lan"`, or `"off_lan"` for a test join with one machine moved to
//!   another network, as the tester declared it), `relays` (`"public"`, `"custom"` or
//!   `"disabled"`), `direct_paths`, `period_ms` (one echo request per period), `payload_bytes`
//!   (each request's size), `timeout_ms` (an echo later than this counts as lost), `duration_s`
//!   (planned) and `elapsed_ms`.
//! - `ends`: `local` and `remote` (`null` until the peer's first echo), each with `os`, `medium`
//!   (`"ethernet"`, `"wifi"`, `"thunderbolt_ip"`, `"overlay"`, `"loopback"` or `"unknown"`,
//!   RFC-0030 §7.2's classes) and `wifi` (`null` unless the medium is Wi-Fi: `generation`, one of
//!   `"wifi4"`, `"wifi5"`, `"wifi6"`, `"wifi7"`, `"older"`, `"unknown"`, and `band`, one of
//!   `"2.4ghz"`, `"5ghz"`, `"6ghz"`, `"unknown"`). Wi-Fi 6E is `"wifi6"` in `"6ghz"`. The values are
//!   the last ones seen; `trace.changes` has their history.
//! - `join`: `connect_ms` (from the start to an open connection), `path_at_connect` (`"direct"` or
//!   `"relay"`), `first_direct_ms` (when an echo request first left on a direct path, or `null`)
//!   and `relayed_rtt_us` (an RTT summary of the echoes sent on a relayed path, or `null`).
//! - `hops`: one entry per combination of `local_medium`, `remote_medium` and `path` seen, with
//!   `beats` (echo requests sent), `lost`, `rtt_us` (a summary, or `null` if every one was lost)
//!   and `histogram`. An RTT summary has `samples`, `min`, `p50`, `p99`, `p999` and `max`, each a
//!   nearest-rank percentile in µs. The histogram has one `{ "le_us", "count" }` bucket per bound
//!   of [`crate::stats::BUCKETS_US`], counting the RTTs at or below that bound and above the one
//!   before.
//! - `stalls`: `min_gap_ms` and `log`. A stall is a gap of at least `min_gap_ms` (the period plus
//!   1 s) between two echoes that arrive, between the connection and the first echo, or between
//!   the last echo and the end of the measurement: one period after the last request, or the
//!   moment an `in_progress` report was written. Each entry has `start_ms` (the echo before the
//!   gap), `gap_ms`, the `local_medium` and `path` of the first request in the gap, and `open`
//!   (true when the measurement ended inside it).
//! - `timings`: what RFC-0030 §5.3's failure detector would have done with these gaps, under the
//!   LAN and WAN timings of RFC-0030 §5.4, for a cluster of these two machines (see
//!   [`crate::detector`]). Each of `lan` and `wan` has `period_ms`, `suspect_after_ms`,
//!   `fail_after_ms`, `would_suspect`, `would_fail` and `events`, one per gap that reached
//!   `suspect_after_ms`: `{ "at_ms", "silence_ms", "outcome" }`, with `outcome` `"suspect"` or
//!   `"fail"`.
//!
//! ## Bounds
//!
//! [`Report::check`] holds every number to its physical range, so that a number cannot carry an
//! IPv4 address read as an integer. Every address from 1.0.0.0 up is at least 2^24 (16,777,216).
//!
//! - **RTTs** (`rtt_us`, the RTT summaries) are at most `timeout_ms`, which is at most 10 s
//!   ([`MAX_TIMEOUT_MS`]), so at most 10,000,000 µs: under 2^24.
//! - **Settings** are the probe's own values: `period_ms` 10 to 1,000, `payload_bytes` the
//!   version's 128, `timeout_ms` 1 to 10,000, `duration_s` up to 7 days. The detector timings and
//!   `stalls.min_gap_ms` must equal the values they are derived from, and the histogram bounds the
//!   fixed ones.
//! - **Counts** (`beats`, `lost`, `samples`, histogram counts, the detector's counts) must add up
//!   to the trace, so none exceeds the number of requests the trace holds.
//! - **Times.** `connect_ms` is at most 120 s and `trace.start_ms` at most 120 s later.
//!   `elapsed_ms` is at most the time the trace's requests take plus the timeout and 60 s. Every
//!   other time is at most `elapsed_ms`. A run longer than 4.66 hours has times above 2^24 ms, so
//!   such a time can equal some address read as an integer. But it is then a plausible time in
//!   this run, bounded by the trace's length, and not a field that can carry anything else.
//!
//! `check`'s output names a field and a rule, never a value from the report: a parse error is
//! given as the field's path, with any key the format does not have shown as `?`.
//!
//! ## The trace section
//!
//! `trace` holds every echo request, so a run can be replayed through an impairment bridge or a
//! simulator (DIST-E4-T12, DIST-E14-T16):
//!
//! - `period_ms` and `timeout_ms`, as in `settings`, and `start_ms`, when request 0 left: once the
//!   connection was open and the local end found.
//! - `rtt_us`: one element per request, in the order sent. Request `i` was due `i` periods after
//!   request 0. The element is the request's round-trip time in µs,
//!   or `null` if its echo did not come back within `timeout_ms`, if no connection was open when it
//!   was due, or, in an `in_progress` report, if it had not come back yet.
//! - `changes`: what changed, and from which request on: `{ "beat": i, "change": c }`, with `c`
//!   one of `{ "path": p }` (`"direct"`, `"relay"`, or `"none"` while disconnected), `{ "local":
//!   end }` and `{ "remote": end }`, where `end` has the form of an entry of `ends`. Request 0
//!   carries the starting path and local end, and the remote end appears with its first echo.

use serde::{Deserialize, Serialize};

use crate::stats::{BUCKETS_US, RttSummary};

/// The value of `format`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Format {
    /// The only format.
    #[serde(rename = "syntropy-netprobe-report")]
    Report,
}

/// The current `version`.
pub const VERSION: u32 = 1;

/// One run's report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Report {
    /// Always [`Format::Report`].
    pub format: Format,
    /// [`VERSION`].
    pub version: u32,
    /// The binary's version: digits and dots.
    pub probe_version: String,
    /// How the run ended.
    pub outcome: Outcome,
    /// What the run was asked to do.
    pub settings: Settings,
    /// The two machines.
    pub ends: Ends,
    /// The first connection.
    pub join: Join,
    /// RTT and loss per medium and path.
    pub hops: Vec<Hop>,
    /// Gaps between echoes.
    pub stalls: Stalls,
    /// The failure detector's verdicts on the gaps.
    pub timings: Timings,
    /// Every request.
    pub trace: Trace,
}

/// How a run ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// The planned duration ran.
    Completed,
    /// A checkpoint written during the run.
    InProgress,
    /// The run stopped before the planned duration.
    Interrupted,
}

/// Where the tester ran the probe, as they declared it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Setting {
    /// Both machines on the home network.
    Lan,
    /// A test join: one machine moved to another network.
    OffLan,
}

/// The relays the probe could use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelayChoice {
    /// iroh's public relays (internal and alpha builds only, RFC-0030 §6.4).
    Public,
    /// A relay URL the tester gave. The URL itself is not recorded.
    Custom,
    /// No relay.
    Disabled,
}

/// The run's settings.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    /// LAN or off-LAN.
    pub setting: Setting,
    /// The relays allowed.
    pub relays: RelayChoice,
    /// Whether direct paths were allowed.
    pub direct_paths: bool,
    /// One echo request per period.
    pub period_ms: u32,
    /// The size of each request and echo.
    pub payload_bytes: u32,
    /// An echo later than this counts as lost.
    pub timeout_ms: u32,
    /// The planned duration.
    pub duration_s: u64,
    /// The time the run has taken so far.
    pub elapsed_ms: u64,
}

/// The operating system, from the build target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Os {
    /// Linux.
    Linux,
    /// Windows.
    Windows,
    /// macOS.
    Macos,
    /// Android.
    Android,
    /// iOS.
    Ios,
    /// Anything else.
    Other,
}

impl Os {
    /// The operating system this binary was built for.
    pub fn current() -> Self {
        match std::env::consts::OS {
            "linux" => Self::Linux,
            "windows" => Self::Windows,
            "macos" => Self::Macos,
            "android" => Self::Android,
            "ios" => Self::Ios,
            _ => Self::Other,
        }
    }
}

/// The medium of the interface a path leaves from (RFC-0030 §7.2's classes, with `Unknown` when
/// the platform tools could not tell).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Medium {
    /// Wired Ethernet.
    Ethernet,
    /// Wi-Fi.
    Wifi,
    /// IP over a Thunderbolt or USB4 cable.
    ThunderboltIp,
    /// A VPN or overlay interface.
    Overlay,
    /// Both ends on one host.
    Loopback,
    /// Not found.
    Unknown,
}

/// A Wi-Fi generation, from the PHY mode the platform reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WifiGeneration {
    /// 802.11n.
    Wifi4,
    /// 802.11ac.
    Wifi5,
    /// 802.11ax (Wi-Fi 6E is this generation in the 6 GHz band).
    Wifi6,
    /// 802.11be.
    Wifi7,
    /// 802.11a, b or g.
    Older,
    /// Not found.
    Unknown,
}

/// A Wi-Fi band.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum WifiBand {
    /// 2.4 GHz.
    #[serde(rename = "2.4ghz")]
    Ghz2_4,
    /// 5 GHz.
    #[serde(rename = "5ghz")]
    Ghz5,
    /// 6 GHz.
    #[serde(rename = "6ghz")]
    Ghz6,
    /// Not found.
    #[serde(rename = "unknown")]
    Unknown,
}

/// The Wi-Fi link: generation and band only, never the network's name or the access point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Wifi {
    /// The generation.
    pub generation: WifiGeneration,
    /// The band.
    pub band: WifiBand,
}

/// One machine's side of a hop.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndInfo {
    /// Its operating system.
    pub os: Os,
    /// The medium its path leaves from.
    pub medium: Medium,
    /// The Wi-Fi link, when the medium is Wi-Fi.
    pub wifi: Option<Wifi>,
}

/// The two machines.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ends {
    /// The machine that wrote the report.
    pub local: EndInfo,
    /// The machine that echoed, once it has.
    pub remote: Option<EndInfo>,
}

/// The kind of path a request left on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Path {
    /// A direct path.
    Direct,
    /// Through a relay.
    Relay,
    /// No connection was open.
    None,
}

/// The first connection.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Join {
    /// From the start to an open connection.
    pub connect_ms: u64,
    /// The path the connection opened on.
    pub path_at_connect: Path,
    /// When a request first left on a direct path.
    pub first_direct_ms: Option<u64>,
    /// The RTTs of the echoes sent on a relayed path.
    pub relayed_rtt_us: Option<RttSummary>,
}

/// One histogram bucket.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bucket {
    /// The bucket's upper bound, inclusive.
    pub le_us: u32,
    /// The RTTs in the bucket.
    pub count: u64,
}

/// RTT and loss for one combination of media and path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hop {
    /// The local medium.
    pub local_medium: Medium,
    /// The remote medium, once known.
    pub remote_medium: Option<Medium>,
    /// The path.
    pub path: Path,
    /// Requests sent.
    pub beats: u64,
    /// Requests whose echo did not come back in time.
    pub lost: u64,
    /// The RTTs of the others.
    pub rtt_us: Option<RttSummary>,
    /// Their histogram.
    pub histogram: Vec<Bucket>,
}

/// One stall.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stall {
    /// The echo before the gap, or the start of the run.
    pub start_ms: u64,
    /// The gap.
    pub gap_ms: u64,
    /// The local medium when the gap began.
    pub local_medium: Medium,
    /// The path when the gap began.
    pub path: Path,
    /// Whether the run ended inside the gap.
    pub open: bool,
}

/// The stall log.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stalls {
    /// The shortest gap that counts.
    pub min_gap_ms: u64,
    /// The stalls, in order.
    pub log: Vec<Stall>,
}

/// What the detector would have done about one gap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// `Suspect` raised, then cleared by the next echo.
    Suspect,
    /// The peer declared failed.
    Fail,
}

/// One gap that reached `suspect_after_ms`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorEvent {
    /// The echo before the gap.
    pub at_ms: u64,
    /// The gap.
    pub silence_ms: u64,
    /// What the detector would have done.
    pub outcome: Verdict,
}

/// The detector's verdicts under one set of timings.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Detector {
    /// The heartbeat period.
    pub period_ms: u64,
    /// The silence that raises `Suspect`.
    pub suspect_after_ms: u64,
    /// The silence that fails the peer.
    pub fail_after_ms: u64,
    /// Gaps that would have raised `Suspect`, failures included.
    pub would_suspect: u64,
    /// Gaps that would have failed the peer.
    pub would_fail: u64,
    /// Each such gap.
    pub events: Vec<DetectorEvent>,
}

/// The detector under RFC-0030 §5.4's LAN and WAN timings.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Timings {
    /// LAN timings.
    pub lan: Detector,
    /// WAN timings.
    pub wan: Detector,
}

/// What changed from one request on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ChangeKind {
    /// The path.
    Path(Path),
    /// The local end.
    Local(EndInfo),
    /// The remote end.
    Remote(EndInfo),
}

/// One change in the trace.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Change {
    /// The first request it applies to.
    pub beat: u64,
    /// What changed.
    pub change: ChangeKind,
}

/// Every request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Trace {
    /// As in [`Settings::period_ms`].
    pub period_ms: u32,
    /// As in [`Settings::timeout_ms`].
    pub timeout_ms: u32,
    /// When request 0 left.
    pub start_ms: u64,
    /// Each request's RTT, or `null` if lost.
    pub rtt_us: Vec<Option<u32>>,
    /// What changed, in order.
    pub changes: Vec<Change>,
}

impl Report {
    /// The schema test: the invariants a well-formed report keeps beyond its types. Each problem
    /// found is a line in the result; an empty result is a pass.
    pub fn check(&self) -> Vec<String> {
        let mut problems = Vec::new();
        let mut fail = |problem: &str| problems.push(problem.to_owned());
        let settings = &self.settings;
        let trace = &self.trace;
        let traced = trace.rtt_us.len() as u64;
        let max_rtt_us = u64::from(settings.timeout_ms) * 1_000;

        if self.version != VERSION {
            fail("version: not 1");
        }
        let parts: Vec<&str> = self.probe_version.split('.').collect();
        if parts.len() != 3
            || parts.iter().any(|part| {
                part.is_empty() || part.len() > 3 || !part.bytes().all(|b| b.is_ascii_digit())
            })
        {
            fail("probe_version: not three numbers joined by dots");
        }

        // Settings: the values the probe itself can write.
        if !(MIN_PERIOD_MS..=MAX_PERIOD_MS).contains(&settings.period_ms) {
            fail("settings.period_ms: outside 10 to 1000");
        }
        if settings.payload_bytes as usize != crate::wire::PAYLOAD_BYTES {
            fail("settings.payload_bytes: not the version's request size");
        }
        if !(1..=MAX_TIMEOUT_MS).contains(&settings.timeout_ms) {
            fail("settings.timeout_ms: outside 1 to 10000");
        }
        if !(1..=MAX_DURATION_S).contains(&settings.duration_s) {
            fail("settings.duration_s: outside 1 s to 7 days");
        }
        if !settings.direct_paths && settings.relays == RelayChoice::Disabled {
            fail("settings: no relay and no direct paths");
        }

        // Times: the connection opens within the connect bound, request 0 leaves soon after, and
        // the run ends within a margin of its last request. Every other time is inside the run.
        let join = &self.join;
        if join.connect_ms > MAX_CONNECT_MS {
            fail("join.connect_ms: longer than a connection attempt");
        }
        if traced > 0
            && !(join.connect_ms..=join.connect_ms.saturating_add(MAX_START_MS))
                .contains(&trace.start_ms)
        {
            fail("trace.start_ms: not soon after the connection opened");
        }
        let period = u64::from(settings.period_ms);
        if traced.saturating_sub(1) * period > settings.duration_s.saturating_mul(1_000) {
            fail("trace.rtt_us: more requests than the duration holds");
        }
        let latest_end = trace
            .start_ms
            .saturating_add(traced * period)
            .saturating_add(u64::from(settings.timeout_ms) + END_MARGIN_MS);
        if settings.elapsed_ms > latest_end {
            fail("settings.elapsed_ms: later than the last request allows");
        }
        let elapsed = settings.elapsed_ms;
        let inside = |ms: u64| ms <= elapsed;
        if !inside(join.connect_ms) || join.first_direct_ms.is_some_and(|ms| !inside(ms)) {
            fail("join: a time after the end of the run");
        }
        if trace.period_ms != settings.period_ms || trace.timeout_ms != settings.timeout_ms {
            fail("trace: period or timeout differs from settings");
        }
        if trace
            .rtt_us
            .iter()
            .flatten()
            .any(|&rtt| u64::from(rtt) > max_rtt_us)
        {
            fail("trace.rtt_us: an RTT longer than the timeout");
        }

        for (index, end) in std::iter::once(&self.ends.local)
            .chain(&self.ends.remote)
            .enumerate()
        {
            check_end(
                if index == 0 {
                    "ends.local"
                } else {
                    "ends.remote"
                },
                end,
                &mut fail,
            );
        }
        if let Some(rtt) = &join.relayed_rtt_us {
            check_summary("join.relayed_rtt_us", rtt, traced, max_rtt_us, &mut fail);
        }

        // Hops add up to the trace.
        let beats: u64 = self.hops.iter().map(|hop| hop.beats.min(traced + 1)).sum();
        let lost: u64 = self.hops.iter().map(|hop| hop.lost.min(traced + 1)).sum();
        let traced_lost = trace.rtt_us.iter().filter(|rtt| rtt.is_none()).count() as u64;
        if beats != traced || lost != traced_lost {
            fail("hops: requests or losses differ from the trace");
        }
        let mut seen = std::collections::BTreeSet::new();
        for (index, hop) in self.hops.iter().enumerate() {
            let at = format!("hops[{index}]");
            let mut fail_at = |rule: &str| fail(&format!("{at}: {rule}"));
            if !seen.insert((hop.local_medium, hop.remote_medium, hop.path)) {
                fail_at("a second entry for the same media and path");
            }
            if hop.beats > traced || hop.lost > hop.beats {
                fail_at("more requests or losses than the trace has");
            }
            if hop.path == Path::None && hop.lost != hop.beats {
                fail_at("an echo with no connection");
            }
            let bounds: Vec<u32> = hop.histogram.iter().map(|bucket| bucket.le_us).collect();
            if bounds != BUCKETS_US {
                fail_at("histogram bounds are not the fixed ones");
            }
            let counted = hop
                .histogram
                .iter()
                .map(|bucket| bucket.count.min(traced + 1))
                .sum::<u64>();
            match &hop.rtt_us {
                Some(rtt) => {
                    check_summary(&format!("{at}.rtt_us"), rtt, traced, max_rtt_us, &mut fail);
                    if rtt.samples != hop.beats.saturating_sub(hop.lost) || counted != rtt.samples {
                        fail(&format!("{at}: samples, histogram and losses disagree"));
                    }
                }
                None => {
                    if hop.lost != hop.beats || counted != 0 {
                        fail(&format!("{at}: RTTs counted but no summary"));
                    }
                }
            }
        }

        // Changes are in order and inside the trace.
        let mut last = None;
        for (index, change) in trace.changes.iter().enumerate() {
            if change.beat >= traced.max(1) || last.is_some_and(|last| change.beat < last) {
                fail(&format!(
                    "trace.changes[{index}]: out of order or past the trace"
                ));
            }
            last = Some(change.beat);
            if let ChangeKind::Local(end) | ChangeKind::Remote(end) = &change.change {
                check_end(&format!("trace.changes[{index}]"), end, &mut fail);
            }
        }

        if self.stalls.min_gap_ms != period + 1_000 {
            fail("stalls.min_gap_ms: not the period plus 1 s");
        }
        for (index, stall) in self.stalls.log.iter().enumerate() {
            if stall.gap_ms < self.stalls.min_gap_ms
                || !inside(stall.start_ms)
                || !inside(stall.gap_ms)
            {
                fail(&format!(
                    "stalls.log[{index}]: under the minimum or outside the run"
                ));
            }
        }
        for (name, detector, timing) in [
            ("timings.lan", &self.timings.lan, crate::detector::LAN),
            ("timings.wan", &self.timings.wan, crate::detector::WAN),
        ] {
            check_detector(name, detector, timing, elapsed, &mut fail);
        }
        problems
    }
}

/// The shortest period the probe runs at.
pub const MIN_PERIOD_MS: u32 = 10;
/// The longest: RFC-0030 §5.4's LAN heartbeat period.
pub const MAX_PERIOD_MS: u32 = 1_000;
/// The longest echo timeout. An RTT is at most this, so at most 10,000,000 µs, under 2^24: no RTT
/// field can hold an IPv4 address read as an integer (any address from 1.0.0.0 up is at least
/// 2^24).
pub const MAX_TIMEOUT_MS: u32 = 10_000;
/// The longest run: 7 days.
pub const MAX_DURATION_S: u64 = 7 * 86_400;
/// The longest a first connection can take: the probe gives up after 60 s, with margin.
pub const MAX_CONNECT_MS: u64 = 120_000;
/// From the connection to request 0: finding the local end, whose tools time out after 15 s each.
pub const MAX_START_MS: u64 = 120_000;
/// After the last request: waiting for its echo, with margin.
pub const END_MARGIN_MS: u64 = 60_000;

fn check_end(at: &str, end: &EndInfo, fail: &mut impl FnMut(&str)) {
    if end.wifi.is_some() != (end.medium == Medium::Wifi) {
        fail(&format!(
            "{at}: Wi-Fi details on a medium that is not Wi-Fi, or none on Wi-Fi"
        ));
    }
}

fn check_summary(
    at: &str,
    rtt: &RttSummary,
    traced: u64,
    max_rtt_us: u64,
    fail: &mut impl FnMut(&str),
) {
    if rtt.samples == 0
        || rtt.samples > traced
        || !(rtt.min <= rtt.p50 && rtt.p50 <= rtt.p99 && rtt.p99 <= rtt.p999 && rtt.p999 <= rtt.max)
        || u64::from(rtt.max) > max_rtt_us
    {
        fail(&format!(
            "{at}: not an ordered RTT summary within the timeout and the trace"
        ));
    }
}

fn check_detector(
    at: &str,
    detector: &Detector,
    timing: crate::detector::Timing,
    elapsed_ms: u64,
    fail: &mut impl FnMut(&str),
) {
    if (
        detector.period_ms,
        detector.suspect_after_ms,
        detector.fail_after_ms,
    ) != (
        timing.period_ms,
        timing.suspect_after_ms,
        timing.fail_after_ms,
    ) {
        fail(&format!("{at}: not RFC-0030 §5.4's timings"));
    }
    let suspected = detector.events.len() as u64;
    let failed = detector
        .events
        .iter()
        .filter(|event| event.outcome == Verdict::Fail)
        .count() as u64;
    if detector.would_suspect != suspected || detector.would_fail != failed {
        fail(&format!("{at}: counts differ from the events"));
    }
    for (index, event) in detector.events.iter().enumerate() {
        let expected = if event.silence_ms >= detector.fail_after_ms {
            Verdict::Fail
        } else {
            Verdict::Suspect
        };
        if event.silence_ms < detector.suspect_after_ms
            || event.outcome != expected
            || event.at_ms > elapsed_ms
            || event.silence_ms > elapsed_ms
        {
            fail(&format!(
                "{at}.events[{index}]: does not fit the timings or the run"
            ));
        }
    }
}

/// Every object key the format has, so that an error can name a field without repeating a key
/// that the format does not have.
const KEYS: &[&str] = &[
    "format",
    "version",
    "probe_version",
    "outcome",
    "settings",
    "ends",
    "join",
    "hops",
    "stalls",
    "timings",
    "trace",
    "setting",
    "relays",
    "direct_paths",
    "period_ms",
    "payload_bytes",
    "timeout_ms",
    "duration_s",
    "elapsed_ms",
    "local",
    "remote",
    "os",
    "medium",
    "wifi",
    "generation",
    "band",
    "connect_ms",
    "path_at_connect",
    "first_direct_ms",
    "relayed_rtt_us",
    "samples",
    "min",
    "p50",
    "p99",
    "p999",
    "max",
    "local_medium",
    "remote_medium",
    "path",
    "beats",
    "lost",
    "rtt_us",
    "histogram",
    "le_us",
    "count",
    "min_gap_ms",
    "log",
    "start_ms",
    "gap_ms",
    "open",
    "lan",
    "wan",
    "suspect_after_ms",
    "fail_after_ms",
    "would_suspect",
    "would_fail",
    "events",
    "at_ms",
    "silence_ms",
    "changes",
    "beat",
    "change",
];

/// The `check` command: the identifier scan of `text`, then the format and [`Report::check`].
/// Each problem is a line that names a place in the report and a rule, and never repeats
/// anything the report holds: a parse error is given as the field's path, with keys the format
/// does not have shown as `?`, and a fixed kind of error.
pub fn check_text(text: &str) -> Result<Report, Vec<String>> {
    let mut problems: Vec<String> = crate::scan::scan(text)
        .into_iter()
        .map(|finding| format!("identifier: {:?} at byte {}", finding.kind, finding.offset))
        .collect();
    match serde_json::from_str::<Report>(text) {
        Ok(report) => {
            problems.extend(report.check());
            if problems.is_empty() {
                return Ok(report);
            }
        }
        Err(error) => problems.push(describe_parse_error(text, &error)),
    }
    Err(problems)
}

/// A parse error as its place and kind, without serde's message, which quotes the input.
fn describe_parse_error(text: &str, error: &serde_json::Error) -> String {
    use serde_json::error::Category;
    let kind = match error.classify() {
        Category::Io => "cannot be read",
        Category::Eof => "the file ends early",
        Category::Syntax => "not well-formed JSON",
        Category::Data => {
            // serde's data errors start with one of these fixed phrases; what follows quotes the
            // input, so only the phrase is used.
            let message = error.to_string();
            [
                ("invalid type", "a value of the wrong type"),
                ("invalid value", "a value out of range"),
                ("invalid length", "a list of the wrong length"),
                ("unknown variant", "a value the field does not allow"),
                ("unknown field", "a field the format does not have"),
                ("missing field", "a field is missing"),
                ("duplicate field", "a field given twice"),
            ]
            .into_iter()
            .find(|(phrase, _)| message.starts_with(phrase))
            .map_or("a value the format does not accept", |(_, kind)| kind)
        }
    };
    let offset = text
        .split_inclusive('\n')
        .take(error.line().saturating_sub(1))
        .map(str::len)
        .sum::<usize>()
        + error.column().saturating_sub(1);
    format!("format: {kind}, at {}", path_at(text, offset))
}

/// The JSON path at byte `offset` of `text`, from the root: keys the format has, `?` for any
/// other key, and array indices.
fn path_at(text: &str, offset: usize) -> String {
    enum Frame {
        Object {
            key: Option<String>,
            expecting_key: bool,
        },
        Array {
            index: usize,
        },
    }
    let bytes = text.as_bytes();
    let mut stack: Vec<Frame> = Vec::new();
    let mut at = 0;
    while at < bytes.len() && at < offset {
        match bytes[at] {
            b'"' => {
                let start = at + 1;
                at += 1;
                while at < bytes.len() && bytes[at] != b'"' {
                    if bytes[at] == b'\\' {
                        at += 1;
                    }
                    at += 1;
                }
                if let Some(Frame::Object { key, expecting_key }) = stack.last_mut()
                    && *expecting_key
                {
                    *key = text.get(start..at.min(bytes.len())).map(str::to_owned);
                    *expecting_key = false;
                }
            }
            b'{' => stack.push(Frame::Object {
                key: None,
                expecting_key: true,
            }),
            b'[' => stack.push(Frame::Array { index: 0 }),
            b'}' | b']' => {
                stack.pop();
            }
            b',' => match stack.last_mut() {
                Some(Frame::Object { expecting_key, .. }) => *expecting_key = true,
                Some(Frame::Array { index }) => *index += 1,
                None => {}
            },
            _ => {}
        }
        at += 1;
    }
    let mut path = String::new();
    for frame in &stack {
        match frame {
            Frame::Object { key: Some(key), .. } => {
                if !path.is_empty() {
                    path.push('.');
                }
                path.push_str(if KEYS.contains(&key.as_str()) {
                    key
                } else {
                    "?"
                });
            }
            Frame::Object { key: None, .. } => {}
            Frame::Array { index } => path.push_str(&format!("[{index}]")),
        }
    }
    if path.is_empty() {
        "the top level".to_owned()
    } else {
        path
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use serde_json::Value;

    use super::*;
    use crate::probe::{Recorder, RunOptions};

    const S: u64 = 1_000_000;

    /// A report with every optional part present: both ends (one on Wi-Fi), a relayed then direct
    /// path, a lost run of requests (a stall and a LAN suspicion), and changes of each kind.
    fn rich() -> Report {
        let wifi = EndInfo {
            os: Os::Macos,
            medium: Medium::Wifi,
            wifi: Some(Wifi {
                generation: WifiGeneration::Wifi6,
                band: WifiBand::Ghz5,
            }),
        };
        let wired = EndInfo {
            os: Os::Linux,
            medium: Medium::Ethernet,
            wifi: None,
        };
        let mut options = RunOptions::new(Setting::OffLan, RelayChoice::Public, true);
        options.duration = Duration::from_secs(60);
        let mut recorder = Recorder::new(options, 200_000, Path::Relay);
        for seq in 0..12u64 {
            let at = (seq + 1) * S;
            let path = if seq < 3 { Path::Relay } else { Path::Direct };
            let local = if seq < 8 { wifi } else { wired };
            recorder.sent(at, path, local);
            if !(5..=8).contains(&seq) {
                recorder.echoed(seq, at + 2_000, if seq < 9 { wired } else { wifi });
            }
        }
        let report = recorder.report(12 * S + 500_000, wired, Outcome::Completed);
        assert_eq!(report.check(), Vec::<String>::new());
        assert!(!report.stalls.log.is_empty() && !report.timings.lan.events.is_empty());
        report
    }

    /// The JSON pointer of every leaf of `value` that `pick` selects, and of every object.
    fn pointers(value: &Value, at: String, out: &mut Vec<(String, bool)>) {
        match value {
            Value::Object(map) => {
                out.push((at.clone(), true));
                for (key, item) in map {
                    pointers(item, format!("{at}/{key}"), out);
                }
            }
            Value::Array(items) => {
                for (index, item) in items.iter().enumerate() {
                    pointers(item, format!("{at}/{index}"), out);
                }
            }
            _ => out.push((at, false)),
        }
    }

    /// `check_text` refuses `text`, and its output repeats none of `plants`.
    fn refused_quietly(text: &str, plants: &[&str], what: &str) {
        let problems = check_text(text).expect_err(what);
        let shown = problems.join("\n");
        for plant in plants {
            assert!(
                !shown.contains(plant),
                "{what}: the output repeats {plant}: {shown}"
            );
        }
    }

    #[test]
    fn the_keys_list_covers_the_format() {
        let mut leaves = Vec::new();
        pointers(
            &serde_json::to_value(rich()).unwrap(),
            String::new(),
            &mut leaves,
        );
        for (pointer, _) in leaves {
            for key in pointer
                .split('/')
                .filter(|part| part.parse::<usize>().is_err())
            {
                assert!(
                    key.is_empty() || KEYS.contains(&key),
                    "{key} is not in KEYS"
                );
            }
        }
    }

    #[test]
    fn a_planted_identifier_is_refused_and_never_repeated() {
        // 0xC0A80001 is 192.168.0.1 read as an integer.
        const AS_NUMBER: u64 = 0xC0A8_0001;
        let plants = [
            "3232235521",
            "192.168.0.1",
            "192.0.2.77",
            "02:00:00:aa:bb:cc",
            "home-net",
        ];
        let base = serde_json::to_value(rich()).unwrap();
        let mut places = Vec::new();
        pointers(&base, String::new(), &mut places);
        let mut planted = 0;
        for (pointer, is_object) in &places {
            let mut variants = Vec::new();
            if *is_object {
                // A key the format does not have, holding an address.
                let mut value = base.clone();
                let target = if pointer.is_empty() {
                    &mut value
                } else {
                    value.pointer_mut(pointer).unwrap()
                };
                target
                    .as_object_mut()
                    .unwrap()
                    .insert("home-net".to_owned(), Value::from("192.0.2.77"));
                variants.push(value);
            } else {
                for plant in [
                    Value::from(AS_NUMBER),
                    Value::from("192.0.2.77"),
                    Value::from("02:00:00:aa:bb:cc"),
                    Value::from("home-net"),
                ] {
                    let mut value = base.clone();
                    *value.pointer_mut(pointer).unwrap() = plant;
                    variants.push(value);
                }
            }
            for value in variants {
                let text = serde_json::to_string_pretty(&value).unwrap();
                refused_quietly(&text, &plants, pointer);
                planted += 1;
            }
        }
        assert!(planted > 1_000, "{planted} plants");

        // Text that is not well-formed, with an address inside it.
        let text = serde_json::to_string_pretty(&base).unwrap();
        refused_quietly(
            &text.replacen("\"lan\"", "\"192.0.2.77", 1),
            &plants,
            "an unterminated string",
        );
        refused_quietly(&text[..text.len() / 2], &plants, "a truncated file");
    }

    #[test]
    fn a_parse_error_names_the_field_by_its_path() {
        let text =
            serde_json::to_string_pretty(&rich())
                .unwrap()
                .replacen("\"wifi6\"", "\"home-net\"", 1);
        assert_eq!(
            check_text(&text).unwrap_err(),
            ["format: a value the field does not allow, at ends.remote.wifi.generation"]
        );
        let text = serde_json::to_string_pretty(&rich()).unwrap().replacen(
            "\"beats\"",
            "\"home-net\": 1, \"beats\"",
            1,
        );
        assert_eq!(
            check_text(&text).unwrap_err(),
            ["format: a field the format does not have, at hops[0].?"]
        );
    }
}
