// SPDX-License-Identifier: MIT OR Apache-2.0

//! The probe itself: a listener that echoes, and a run that sends one echo request per period,
//! times the echoes, and writes what it saw into a [`Report`].
//!
//! [`Recorder`] holds a run's observations and builds the report from them. It is plain data with
//! times passed in, so its rules are tested without a network. [`run`] and [`serve`] drive it over
//! a real [`Endpoint`].

use std::collections::BTreeMap;
use std::net::IpAddr;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use syntropy_net::endpoint::{Connection, Endpoint, EndpointError, NodeAddr, PathKind};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::detector::{self, LAN, WAN};
use crate::platform::Detect;
use crate::report::{
    Change, ChangeKind, EndInfo, Ends, Format, Hop, Join, Outcome, Path, RelayChoice, Report,
    Setting, Settings, Stall, Stalls, Timings, Trace, VERSION,
};
use crate::stats::{RttSummary, histogram};
use crate::wire::{ALPN, Message, PAYLOAD_BYTES};

/// What a run is asked to do.
#[derive(Clone, Debug)]
pub struct RunOptions {
    /// How long to send echo requests.
    pub duration: Duration,
    /// One request per period; at most RFC-0030 §5.4's LAN period of 1 s.
    pub period: Duration,
    /// An echo later than this counts as lost.
    pub timeout: Duration,
    /// LAN or off-LAN, as the tester declares it.
    pub setting: Setting,
    /// The relays the endpoint was bound with.
    pub relays: RelayChoice,
    /// Whether the endpoint has direct paths.
    pub direct_paths: bool,
    /// How often to look at the local interface again.
    pub detect_every: Duration,
    /// How often to hand a checkpoint report to the caller.
    pub checkpoint_every: Duration,
    /// How long to wait between attempts to reconnect.
    pub reconnect_every: Duration,
    /// How long one connection attempt may take.
    pub connect_timeout: Duration,
}

impl RunOptions {
    /// A 24-hour run at RFC-0030 §5.4's LAN heartbeat period, with a 10 s echo timeout.
    pub fn new(setting: Setting, relays: RelayChoice, direct_paths: bool) -> Self {
        Self {
            duration: Duration::from_secs(24 * 3_600),
            period: Duration::from_secs(1),
            timeout: Duration::from_secs(10),
            setting,
            relays,
            direct_paths,
            detect_every: Duration::from_secs(60),
            checkpoint_every: Duration::from_secs(60),
            reconnect_every: Duration::from_secs(5),
            connect_timeout: Duration::from_secs(60),
        }
    }
}

/// One echo request.
#[derive(Clone, Copy, Debug)]
struct Beat {
    sent_us: u64,
    path: Path,
    local: EndInfo,
    remote: Option<EndInfo>,
    rtt_us: Option<u32>,
}

/// A run's observations. Times are µs from the start of the run.
#[derive(Clone, Debug)]
pub struct Recorder {
    options: RunOptions,
    connect_us: u64,
    path_at_connect: Path,
    beats: Vec<Beat>,
    arrivals_us: Vec<u64>,
    changes: Vec<Change>,
    remote: Option<EndInfo>,
}

fn millis(us: u64) -> u64 {
    us / 1_000
}

fn as_u32(value: u128) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

impl Recorder {
    /// A run whose first connection opened at `connect_us`, on `path`.
    pub fn new(options: RunOptions, connect_us: u64, path: Path) -> Self {
        Self {
            options,
            connect_us,
            path_at_connect: path,
            beats: Vec::new(),
            arrivals_us: Vec::new(),
            changes: Vec::new(),
            remote: None,
        }
    }

    /// Records a request sent at `at_us` on `path` (`Path::None` when no connection was open)
    /// from `local`, and returns its sequence number.
    pub fn sent(&mut self, at_us: u64, path: Path, local: EndInfo) -> u64 {
        let seq = self.beats.len() as u64;
        let last = self.beats.last();
        if last.is_none_or(|beat| beat.path != path) {
            self.changes.push(Change {
                beat: seq,
                change: ChangeKind::Path(path),
            });
        }
        if last.is_none_or(|beat| beat.local != local) {
            self.changes.push(Change {
                beat: seq,
                change: ChangeKind::Local(local),
            });
        }
        self.beats.push(Beat {
            sent_us: at_us,
            path,
            local,
            remote: self.remote,
            rtt_us: None,
        });
        seq
    }

    /// Records the echo of request `seq`, arriving at `at_us` from `remote`. An echo later than
    /// the timeout does not count as an RTT, but it shows the peer was heard.
    pub fn echoed(&mut self, seq: u64, at_us: u64, remote: EndInfo) {
        let Some(beat) = usize::try_from(seq)
            .ok()
            .and_then(|index| self.beats.get_mut(index))
        else {
            return;
        };
        self.arrivals_us.push(at_us);
        if self.remote != Some(remote) {
            self.remote = Some(remote);
            self.changes.push(Change {
                beat: seq,
                change: ChangeKind::Remote(remote),
            });
        }
        beat.remote = Some(remote);
        let rtt = at_us.saturating_sub(beat.sent_us);
        if beat.rtt_us.is_none() && rtt <= self.options.timeout.as_micros() as u64 {
            beat.rtt_us = Some(as_u32(u128::from(rtt)));
        }
    }

    /// Whether every request sent on a connection has its RTT.
    fn all_answered(&self) -> bool {
        self.beats
            .iter()
            .all(|beat| beat.path == Path::None || beat.rtt_us.is_some())
    }

    /// The report at `now_us`.
    pub fn report(&self, now_us: u64, local: EndInfo, outcome: Outcome) -> Report {
        let options = &self.options;
        let period_ms = as_u32(options.period.as_millis());
        let timeout_ms = as_u32(options.timeout.as_millis());

        // Silences: from the connection, between echoes, and from the last echo to the end of
        // the measurement, which is when the request after the last one would have been due (or
        // now, if that is earlier). Waiting for late echoes after the last request is not silence.
        let mut points = Vec::with_capacity(self.arrivals_us.len() + 1);
        points.push(self.connect_us);
        let mut arrivals = self.arrivals_us.clone();
        arrivals.sort_unstable();
        points.extend(arrivals.into_iter().filter(|&at| at >= self.connect_us));
        let mut silences: Vec<(u64, u64, bool)> = points
            .windows(2)
            .map(|pair| (pair[0], pair[1] - pair[0], false))
            .collect();
        let last = *points.last().unwrap_or(&self.connect_us);
        let end = self.beats.last().map_or(now_us, |beat| {
            now_us.min(beat.sent_us + options.period.as_micros() as u64)
        });
        silences.push((last, end.saturating_sub(last), true));
        let silences_ms: Vec<(u64, u64)> = silences
            .iter()
            .map(|&(start, gap, _)| (millis(start), millis(gap)))
            .collect();

        let min_gap_ms = u64::from(period_ms) + 1_000;
        let log = silences
            .iter()
            .filter(|&&(_, gap, _)| millis(gap) >= min_gap_ms)
            .map(|&(start, gap, open)| {
                let first = self
                    .beats
                    .iter()
                    .find(|beat| beat.sent_us > start)
                    .or(self.beats.last());
                Stall {
                    start_ms: millis(start),
                    gap_ms: millis(gap),
                    local_medium: first.map_or(local.medium, |beat| beat.local.medium),
                    path: first.map_or(Path::None, |beat| beat.path),
                    open,
                }
            })
            .collect();

        let mut groups: BTreeMap<_, (u64, u64, Vec<u32>)> = BTreeMap::new();
        for beat in &self.beats {
            let key = (
                beat.local.medium,
                beat.remote.map(|remote| remote.medium),
                beat.path,
            );
            let group = groups.entry(key).or_default();
            group.0 += 1;
            match beat.rtt_us {
                Some(rtt) => group.2.push(rtt),
                None => group.1 += 1,
            }
        }
        let hops = groups
            .into_iter()
            .map(
                |((local_medium, remote_medium, path), (beats, lost, mut rtts))| Hop {
                    local_medium,
                    remote_medium,
                    path,
                    beats,
                    lost,
                    histogram: histogram(&rtts),
                    rtt_us: RttSummary::of(&mut rtts),
                },
            )
            .collect();

        let mut relayed: Vec<u32> = self
            .beats
            .iter()
            .filter(|beat| beat.path == Path::Relay)
            .filter_map(|beat| beat.rtt_us)
            .collect();
        let mut changes = self.changes.clone();
        changes.sort_by_key(|change| change.beat);

        Report {
            format: Format::Report,
            version: VERSION,
            probe_version: env!("CARGO_PKG_VERSION").to_owned(),
            outcome,
            settings: Settings {
                setting: options.setting,
                relays: options.relays,
                direct_paths: options.direct_paths,
                period_ms,
                payload_bytes: PAYLOAD_BYTES as u32,
                timeout_ms,
                duration_s: options.duration.as_secs(),
                elapsed_ms: millis(now_us),
            },
            ends: Ends {
                local,
                remote: self.remote,
            },
            join: Join {
                connect_ms: millis(self.connect_us),
                path_at_connect: self.path_at_connect,
                first_direct_ms: self
                    .beats
                    .iter()
                    .find(|beat| beat.path == Path::Direct)
                    .map(|beat| millis(beat.sent_us)),
                relayed_rtt_us: RttSummary::of(&mut relayed),
            },
            hops,
            stalls: Stalls { min_gap_ms, log },
            timings: Timings {
                lan: detector::evaluate(LAN, &silences_ms),
                wan: detector::evaluate(WAN, &silences_ms),
            },
            trace: Trace {
                period_ms,
                timeout_ms,
                start_ms: self.beats.first().map_or(0, |beat| millis(beat.sent_us)),
                rtt_us: self.beats.iter().map(|beat| beat.rtt_us).collect(),
                changes,
            },
        }
    }
}

/// The report's name for a path kind.
fn path_of(kind: Option<PathKind>) -> Option<Path> {
    kind.map(|kind| match kind {
        PathKind::Direct => Path::Direct,
        PathKind::Relay => Path::Relay,
    })
}

/// The local end, kept current by a background task that asks `detect` again every `every`,
/// for the peer address in `peer`.
struct LocalEnd {
    end: Arc<Mutex<EndInfo>>,
    peer: Arc<Mutex<Option<IpAddr>>>,
    task: JoinHandle<()>,
}

impl LocalEnd {
    async fn start(detect: Arc<dyn Detect>, peer: Option<IpAddr>, every: Duration) -> Self {
        let first = {
            let detect = detect.clone();
            tokio::task::spawn_blocking(move || detect.local_end(peer))
                .await
                .expect("the detection thread does not panic")
        };
        let end = Arc::new(Mutex::new(first));
        let peer = Arc::new(Mutex::new(peer));
        let task = {
            let (end, peer) = (end.clone(), peer.clone());
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(every).await;
                    let target = *peer.lock().unwrap_or_else(PoisonError::into_inner);
                    let detect = detect.clone();
                    if let Ok(found) =
                        tokio::task::spawn_blocking(move || detect.local_end(target)).await
                    {
                        *end.lock().unwrap_or_else(PoisonError::into_inner) = found;
                    }
                }
            })
        };
        Self { end, peer, task }
    }

    fn get(&self) -> EndInfo {
        *self.end.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn set_peer(&self, peer: Option<IpAddr>) {
        *self.peer.lock().unwrap_or_else(PoisonError::into_inner) = peer;
    }
}

impl Drop for LocalEnd {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// Echoes every request on every connection `endpoint` accepts, until the endpoint closes. Each
/// echo carries this machine's end, as `detect` finds it every `detect_every`.
pub async fn serve(
    endpoint: Endpoint,
    detect: Arc<dyn Detect>,
    detect_every: Duration,
) -> Result<(), EndpointError> {
    loop {
        match endpoint.accept().await {
            Ok(conn) => {
                tokio::spawn(echo(conn, detect.clone(), detect_every));
            }
            Err(EndpointError::Closed) => return Ok(()),
            // One failed handshake does not stop the listener.
            Err(_) => {}
        }
    }
}

async fn echo(conn: Connection, detect: Arc<dyn Detect>, detect_every: Duration) {
    let local = LocalEnd::start(
        detect,
        conn.remote_direct_addr().map(|addr| addr.ip()),
        detect_every,
    )
    .await;
    while let Ok(data) = conn.read_datagram().await {
        if let Some(Message::Request { seq }) = Message::decode(&data) {
            local.set_peer(conn.remote_direct_addr().map(|addr| addr.ip()));
            let reply = Message::Echo {
                seq,
                end: local.get(),
            };
            // A lost echo is what the probe measures; a closed connection ends the loop above.
            let _ = conn.send_datagram(&reply.encode());
        }
    }
}

/// What the tasks of a run tell its main loop.
enum Event {
    Echo { seq: u64, at: Instant, end: EndInfo },
    Closed { id: u64 },
    Connected { conn: Connection },
    ConnectFailed,
}

/// Reads echoes from `conn` until it closes.
fn read_echoes(conn: Connection, id: u64, events: mpsc::UnboundedSender<Event>) -> JoinHandle<()> {
    tokio::spawn(async move {
        while let Ok(data) = conn.read_datagram().await {
            if let Some(Message::Echo { seq, end }) = Message::decode(&data) {
                let at = Instant::now();
                if events.send(Event::Echo { seq, at, end }).is_err() {
                    return;
                }
            }
        }
        let _ = events.send(Event::Closed { id });
    })
}

async fn connect(
    endpoint: &Endpoint,
    peer: NodeAddr,
    within: Duration,
) -> Result<Connection, EndpointError> {
    tokio::time::timeout(within, endpoint.connect(peer, ALPN))
        .await
        .unwrap_or_else(|_| {
            Err(EndpointError::Connect(format!(
                "no connection within {} s",
                within.as_secs()
            )))
        })
}

/// Runs the probe against `peer` and returns the final report. `checkpoint` receives an
/// `in_progress` report every [`RunOptions::checkpoint_every`]. A lost connection is retried every
/// [`RunOptions::reconnect_every`] while requests keep counting as lost; only a failure of the first
/// connection is an error.
pub async fn run(
    endpoint: &Endpoint,
    peer: NodeAddr,
    options: RunOptions,
    detect: Arc<dyn Detect>,
    mut checkpoint: impl FnMut(&Report),
) -> Result<Report, EndpointError> {
    let start = Instant::now();
    let us = |at: Instant| at.saturating_duration_since(start).as_micros() as u64;

    let first = connect(endpoint, peer.clone(), options.connect_timeout).await?;
    let mut path = path_of(first.path_kind()).unwrap_or(Path::Relay);
    let mut recorder = Recorder::new(options.clone(), us(Instant::now()), path);
    let peer_ip = |conn: &Connection| conn.remote_direct_addr().map(|addr| addr.ip());
    let local = LocalEnd::start(detect, peer_ip(&first), options.detect_every).await;

    let (events, mut inbox) = mpsc::unbounded_channel();
    let mut conn_id = 0;
    let mut reader = read_echoes(first.clone(), conn_id, events.clone());
    let mut conn = Some(first);
    let mut connecting = false;
    let mut next_attempt = Instant::now();

    let beats_start = Instant::now();
    let mut ticker = tokio::time::interval(options.period);
    let mut last_checkpoint = Instant::now();
    loop {
        tokio::select! {
            tick = ticker.tick() => {
                if tick.into_std().saturating_duration_since(beats_start) >= options.duration {
                    break;
                }
                let now = Instant::now();
                match &conn {
                    Some(open) => {
                        path = path_of(open.path_kind()).unwrap_or(path);
                        local.set_peer(peer_ip(open));
                        let seq = recorder.sent(us(now), path, local.get());
                        if let Err(EndpointError::ConnectionLost(_)) =
                            open.send_datagram(&Message::Request { seq }.encode())
                        {
                            conn = None;
                        }
                    }
                    None => {
                        recorder.sent(us(now), Path::None, local.get());
                        if !connecting && now >= next_attempt {
                            connecting = true;
                            let (endpoint, peer, events) =
                                (endpoint.clone(), peer.clone(), events.clone());
                            let within = options.connect_timeout;
                            tokio::spawn(async move {
                                let event = match connect(&endpoint, peer, within).await {
                                    Ok(conn) => Event::Connected { conn },
                                    Err(_) => Event::ConnectFailed,
                                };
                                let _ = events.send(event);
                            });
                        }
                    }
                }
                if now.duration_since(last_checkpoint) >= options.checkpoint_every {
                    last_checkpoint = now;
                    checkpoint(&recorder.report(us(now), local.get(), Outcome::InProgress));
                }
            }
            Some(event) = inbox.recv() => match event {
                Event::Echo { seq, at, end } => recorder.echoed(seq, us(at), end),
                Event::Closed { id } if id == conn_id => conn = None,
                Event::Closed { .. } => {}
                Event::Connected { conn: opened } => {
                    connecting = false;
                    conn_id += 1;
                    reader.abort();
                    reader = read_echoes(opened.clone(), conn_id, events.clone());
                    conn = Some(opened);
                }
                Event::ConnectFailed => {
                    connecting = false;
                    next_attempt = Instant::now() + options.reconnect_every;
                }
            },
        }
    }

    // The last requests' echoes may still be on their way.
    let drain_until = tokio::time::Instant::now() + options.timeout;
    while !recorder.all_answered() {
        match tokio::time::timeout_at(drain_until, inbox.recv()).await {
            Ok(Some(Event::Echo { seq, at, end })) => recorder.echoed(seq, us(at), end),
            Ok(Some(_)) => {}
            Ok(None) | Err(_) => break,
        }
    }
    reader.abort();
    if let Some(open) = conn {
        open.close();
    }
    Ok(recorder.report(us(Instant::now()), local.get(), Outcome::Completed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::{Medium, Os, Verdict, Wifi, WifiBand, WifiGeneration};

    const WIRED: EndInfo = EndInfo {
        os: Os::Linux,
        medium: Medium::Ethernet,
        wifi: None,
    };
    const WIFI: EndInfo = EndInfo {
        os: Os::Macos,
        medium: Medium::Wifi,
        wifi: Some(Wifi {
            generation: WifiGeneration::Wifi6,
            band: WifiBand::Ghz5,
        }),
    };

    fn options() -> RunOptions {
        let mut options = RunOptions::new(Setting::Lan, RelayChoice::Public, true);
        options.duration = Duration::from_secs(60);
        options
    }

    const S: u64 = 1_000_000;

    #[test]
    fn a_recorded_run_reports_hops_stalls_verdicts_and_its_trace() {
        // Connected at 0.2 s on a relay; one request per second from 1 s. Requests 0-2 are
        // relayed and echoed after 30 ms; the path then moves to direct. Requests 3-9 are echoed
        // after 2 ms, except 5-7, which are lost (a gap of 4 s between echoes); the local machine
        // moves from Wi-Fi to Ethernet at request 8; 10 is still out when the run ends.
        let mut recorder = Recorder::new(options(), 200_000, Path::Relay);
        for seq in 0..11u64 {
            let at = (seq + 1) * S;
            let path = if seq < 3 { Path::Relay } else { Path::Direct };
            let local = if seq < 8 { WIFI } else { WIRED };
            assert_eq!(recorder.sent(at, path, local), seq);
            let rtt = if seq < 3 { 30_000 } else { 2_000 };
            if !(5..=7).contains(&seq) && seq != 10 {
                recorder.echoed(seq, at + rtt, WIRED);
            }
        }
        let report = recorder.report(11 * S + 500_000, WIRED, Outcome::Completed);
        assert_eq!(report.check(), Vec::<String>::new());

        assert_eq!(report.join.connect_ms, 200);
        assert_eq!(report.join.path_at_connect, Path::Relay);
        assert_eq!(report.join.first_direct_ms, Some(4_000));
        let relayed = report.join.relayed_rtt_us.unwrap();
        assert_eq!((relayed.samples, relayed.p50), (3, 30_000));

        let hops: Vec<_> = report
            .hops
            .iter()
            .map(|hop| (hop.local_medium, hop.path, hop.beats, hop.lost))
            .collect();
        assert_eq!(
            hops,
            [
                (Medium::Ethernet, Path::Direct, 3, 1),
                (Medium::Wifi, Path::Direct, 5, 3),
                (Medium::Wifi, Path::Relay, 3, 0),
            ]
        );
        assert_eq!(
            report.trace.rtt_us,
            [
                Some(30_000),
                Some(30_000),
                Some(30_000),
                Some(2_000),
                Some(2_000),
                None,
                None,
                None,
                Some(2_000),
                Some(2_000),
                None,
            ]
        );
        let changes: Vec<(u64, ChangeKind)> = report
            .trace
            .changes
            .iter()
            .map(|change| (change.beat, change.change))
            .collect();
        assert_eq!(
            changes,
            [
                (0, ChangeKind::Path(Path::Relay)),
                (0, ChangeKind::Local(WIFI)),
                (0, ChangeKind::Remote(WIRED)),
                (3, ChangeKind::Path(Path::Direct)),
                (8, ChangeKind::Local(WIRED)),
            ]
        );

        // The echo of request 4 arrived at 5.002 s and the next at 9.002 s: a 4 s gap, a stall,
        // and a LAN suspicion that the next echo clears. WAN timings raise nothing.
        let stalls: Vec<(u64, u64, bool)> = report
            .stalls
            .log
            .iter()
            .map(|stall| (stall.start_ms, stall.gap_ms, stall.open))
            .collect();
        // The run ended 1.498 s after the last echo: shorter than a stall.
        assert_eq!(stalls, [(5_002, 4_000, false)]);
        assert_eq!(report.stalls.log[0].local_medium, Medium::Wifi);
        let lan: Vec<(u64, Verdict)> = report
            .timings
            .lan
            .events
            .iter()
            .map(|event| (event.silence_ms, event.outcome))
            .collect();
        assert_eq!(lan, [(4_000, Verdict::Suspect)]);
        assert_eq!(report.timings.wan.would_suspect, 0);
        assert_eq!(report.ends.remote, Some(WIRED));
    }

    #[test]
    fn waiting_after_the_last_request_is_not_silence() {
        // Every request answered; the report is taken 10 s after the last one, as when the run
        // waits out a lost echo. The measurement ended one period after the last request.
        let mut recorder = Recorder::new(options(), 0, Path::Direct);
        for seq in 0..3u64 {
            let at = (seq + 1) * S;
            recorder.sent(at, Path::Direct, WIRED);
            recorder.echoed(seq, at + 1_000, WIRED);
        }
        let report = recorder.report(13 * S, WIRED, Outcome::Completed);
        assert_eq!(report.check(), Vec::<String>::new());
        assert_eq!(report.stalls.log, []);
        assert_eq!(report.timings.lan.would_suspect, 0);
        // A checkpoint just after a request measures to now.
        let early = recorder.report(3 * S + 2_000, WIRED, Outcome::InProgress);
        assert_eq!(early.stalls.log, []);
    }

    #[test]
    fn requests_while_disconnected_are_lost_and_late_echoes_still_count_as_heard() {
        let mut options = options();
        options.timeout = Duration::from_secs(2);
        let mut recorder = Recorder::new(options, 0, Path::Direct);
        recorder.sent(S, Path::Direct, WIRED);
        recorder.echoed(0, S + 1_000, WIRED);
        recorder.sent(2 * S, Path::Direct, WIRED);
        // Request 1's echo comes after 2.5 s: past the timeout, so lost, but heard.
        recorder.sent(3 * S, Path::None, WIRED);
        recorder.sent(4 * S, Path::None, WIRED);
        recorder.echoed(1, 4 * S + 500_000, WIRED);
        recorder.sent(5 * S, Path::Direct, WIRED);
        recorder.echoed(4, 5 * S + 1_000, WIRED);
        let report = recorder.report(5 * S + 2_000, WIRED, Outcome::Interrupted);
        assert_eq!(report.check(), Vec::<String>::new());
        assert_eq!(
            report.trace.rtt_us,
            [Some(1_000), None, None, None, Some(1_000)]
        );
        let none = report
            .hops
            .iter()
            .find(|hop| hop.path == Path::None)
            .unwrap();
        assert_eq!((none.beats, none.lost, none.rtt_us), (2, 2, None));
        // Heard at 1.001 s, 4.5 s and 5.001 s: the longest silence is 3.499 s.
        assert_eq!(report.timings.lan.would_suspect, 1);
        assert_eq!(report.timings.lan.events[0].silence_ms, 3_499);
    }
}
