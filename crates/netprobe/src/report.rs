// SPDX-License-Identifier: MIT OR Apache-2.0

//! The report file: one JSON document per run, which the tester reads and submits by hand
//! (ADR-0510). [`Report::check`] is its schema test, and `syntropy-netprobe check <file>` runs it
//! together with the identifier scan of [`crate::scan`].
//!
//! # Format, version 1
//!
//! Every object refuses unknown keys, and every string is one of the enum values listed here, or
//! the probe's version (digits and dots). So the report has no free text in which an SSID, a host
//! name or an address could appear, and [`crate::scan`] checks the numbers too. Times are
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
        let mut fail = |problem: String| problems.push(problem);
        if self.version != VERSION {
            fail(format!(
                "version is {}, and this probe reads {VERSION}",
                self.version
            ));
        }
        if self.probe_version.is_empty()
            || self.probe_version.len() > 32
            || !self
                .probe_version
                .chars()
                .all(|c| c.is_ascii_digit() || c == '.')
        {
            fail("probe_version is not digits and dots".to_owned());
        }
        let settings = &self.settings;
        if !(10..=10_000).contains(&settings.period_ms) {
            fail(format!(
                "period_ms {} is outside 10..=10000",
                settings.period_ms
            ));
        }
        if settings.payload_bytes as usize > crate::wire::MAX_PAYLOAD {
            fail(format!(
                "payload_bytes {} exceeds {}",
                settings.payload_bytes,
                crate::wire::MAX_PAYLOAD
            ));
        }
        if !settings.direct_paths && settings.relays == RelayChoice::Disabled {
            fail("no relay and no direct paths".to_owned());
        }
        if self.trace.period_ms != settings.period_ms
            || self.trace.timeout_ms != settings.timeout_ms
        {
            fail("trace's period or timeout differs from settings'".to_owned());
        }
        if !self.trace.rtt_us.is_empty() && self.trace.start_ms < self.join.connect_ms {
            fail("request 0 left before the connection opened".to_owned());
        }
        for end in std::iter::once(&self.ends.local).chain(&self.ends.remote) {
            check_end(end, &mut fail);
        }
        if let Some(rtt) = &self.join.relayed_rtt_us {
            check_summary("join.relayed_rtt_us", rtt, &mut fail);
        }

        // Hops add up to the trace.
        let beats: u64 = self.hops.iter().map(|hop| hop.beats).sum();
        let lost: u64 = self.hops.iter().map(|hop| hop.lost).sum();
        let traced = self.trace.rtt_us.len() as u64;
        let traced_lost = self.trace.rtt_us.iter().filter(|rtt| rtt.is_none()).count() as u64;
        if beats != traced || lost != traced_lost {
            fail(format!(
                "hops count {beats} requests and {lost} lost; the trace has {traced} and \
                 {traced_lost}"
            ));
        }
        let mut seen = std::collections::BTreeSet::new();
        for hop in &self.hops {
            let key = (hop.local_medium, hop.remote_medium, hop.path);
            if !seen.insert(key) {
                fail(format!("two hops for {key:?}"));
            }
            if hop.lost > hop.beats {
                fail(format!("{key:?}: more lost than sent"));
            }
            if hop.path == Path::None && hop.lost != hop.beats {
                fail(format!("{key:?}: an echo with no connection"));
            }
            let bounds: Vec<u32> = hop.histogram.iter().map(|bucket| bucket.le_us).collect();
            if bounds != BUCKETS_US {
                fail(format!("{key:?}: histogram bounds are not the fixed ones"));
            }
            let counted: u64 = hop.histogram.iter().map(|bucket| bucket.count).sum();
            match &hop.rtt_us {
                Some(rtt) => {
                    check_summary(&format!("{key:?}"), rtt, &mut fail);
                    if rtt.samples != hop.beats - hop.lost.min(hop.beats) || counted != rtt.samples
                    {
                        fail(format!("{key:?}: samples, histogram and losses disagree"));
                    }
                }
                None => {
                    if hop.lost != hop.beats || counted != 0 {
                        fail(format!("{key:?}: RTTs counted but no summary"));
                    }
                }
            }
        }

        // Changes are in order and inside the trace.
        let mut last = None;
        for change in &self.trace.changes {
            if change.beat >= traced.max(1) || last.is_some_and(|last| change.beat < last) {
                fail(format!("change at request {} is out of order", change.beat));
            }
            last = Some(change.beat);
            if let ChangeKind::Local(end) | ChangeKind::Remote(end) = &change.change {
                check_end(end, &mut fail);
            }
        }

        for stall in &self.stalls.log {
            if stall.gap_ms < self.stalls.min_gap_ms {
                fail(format!(
                    "a stall of {} ms is under the minimum",
                    stall.gap_ms
                ));
            }
        }
        for detector in [&self.timings.lan, &self.timings.wan] {
            check_detector(detector, &mut fail);
        }
        problems
    }
}

fn check_end(end: &EndInfo, fail: &mut impl FnMut(String)) {
    if end.wifi.is_some() != (end.medium == Medium::Wifi) {
        fail(format!(
            "{:?} has Wi-Fi details {:?}; only a Wi-Fi medium has them",
            end.medium, end.wifi
        ));
    }
}

fn check_summary(what: &str, rtt: &RttSummary, fail: &mut impl FnMut(String)) {
    if rtt.samples == 0
        || !(rtt.min <= rtt.p50 && rtt.p50 <= rtt.p99 && rtt.p99 <= rtt.p999 && rtt.p999 <= rtt.max)
    {
        fail(format!("{what}: the RTT summary is not ordered: {rtt:?}"));
    }
}

fn check_detector(detector: &Detector, fail: &mut impl FnMut(String)) {
    let suspected = detector.events.len() as u64;
    let failed = detector
        .events
        .iter()
        .filter(|event| event.outcome == Verdict::Fail)
        .count() as u64;
    if detector.would_suspect != suspected || detector.would_fail != failed {
        fail("a detector's counts differ from its events".to_owned());
    }
    for event in &detector.events {
        let expected = if event.silence_ms >= detector.fail_after_ms {
            Verdict::Fail
        } else {
            Verdict::Suspect
        };
        if event.silence_ms < detector.suspect_after_ms || event.outcome != expected {
            fail(format!(
                "a detector event does not fit its timings: {event:?}"
            ));
        }
    }
}
