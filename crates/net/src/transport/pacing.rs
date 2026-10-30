// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bulk pacing: bulk transfers yield to the data plane (RFC-0031 §7.5; RFC-0001 §8.4 rule 6;
//! ADR-0011 item 8; DIST-E7-T13). Sans I/O: the caller passes the time in.
//!
//! A [`BulkPacer`] paces the bulk sends on one edge, `BlobData` and the data frames of `Bulk`
//! and `Replay` channels, with a [`TokenBucket`], and sets the bucket's rate by AIMD on the edge's
//! live data channel:
//! - **Every window** of [`WINDOW`] (250 ms), it takes the p99 of the α samples the data channel
//!   gave in the window, and compares it with `net.bulk_p99_ratio` (1.2) times the edge's idle
//!   p99, which RFC-0010 records.
//! - **Above it,** the rate halves. **Otherwise** it grows by [`INCREASE`] (5 %) of the edge's β,
//!   up to β. A window with no sample, when no data channel is live, counts as within it.
//! - **The rate never drops below** `net.bulk_floor` (5 % of β), so a transfer always progresses.
//! - **It starts at the floor** and rises from there, so a transfer that starts beside a live
//!   ring does not first take the whole link.
//! - **After a halving from rate R, growth stops [`CEILING_MARGIN`] steps below R.** The ceiling
//!   rises one step after [`REPROBE_AFTER`] windows within the bound, and the wait doubles after
//!   each breach that follows a rise. Without the ceiling the rate walks back into the saturation
//!   it just left every few seconds, and each probe window puts enough samples over the bound
//!   that the data channel's p99 over a long transfer exceeds it; a heavier data load still
//!   lowers the ceiling through further halvings (RFC-0031 §7.5, note of 2026-10-23).
//! - **[`RESET_AFTER`] windows in a row with no sample** (256 s), once the data channel has gone
//!   away or gone idle, reset the ceiling to β and the re-probe wait to [`REPROBE_AFTER`]: with
//!   nothing to protect, an old breach no longer holds the rate down. A shorter pause, such as
//!   one between bursts of decode, keeps them. The ring that comes back saturates the link where
//!   the last breach found, and a reset after a short pause would let the next burst's first
//!   window break the bound.
//!
//! Rates and tokens are whole bytes and nanoseconds, so a run paces the same on every host.
//!
//! **In the real backends** (DIST-E7-T25), the QUIC backend and the TCP fallback each keep a
//! [`PacedEdges`]: one [`BulkPacer`] per peer edge, set with [`PacedEdges::pace`] from
//! `net.bulk_p99_ratio`, `net.bulk_floor`, the edge's β and its idle p99
//! ([`PacingConfig::new`]). Every channel to that peer shares the edge:
//! - `try_send` on a `Bulk` or `Replay` channel takes the frame's bytes from the pacer, after the
//!   channel's credit check. Without the tokens it returns `WouldBlock`, and
//!   `Channel::paced_until` says when they will be there.
//! - Each α sample the core of one of the peer's other channels takes goes to the pacer, in the
//!   channel's `net` task, as the frame that gave it lands.
//! - The pacer reads one [`Clock`], the one `pace` was given: the dispatcher's sends and the
//!   `net` task's samples are on one timeline, and a test can drive it by hand.
//! - **Resends are not paced.** The gate is `try_send`, as RFC-0031 §7.5 and DIST-E7-T25 place
//!   it, so a frame's bytes are charged once, when it is first sent. A resend after an ack
//!   timeout or a NACK (§7.3) goes from the channel core in the `net` task, unpaced. On QUIC and
//!   TCP the transport repairs packet loss below the channel, so a resend is rare, and a batched
//!   receiver's credit on idle (§7.1, note of 2026-10-31) keeps a sparse paced sender from
//!   resending at every ack timeout.
//! - The edge's state sits behind a mutex that only bulk sends, α samples and the owner's calls
//!   take, each for one pacer step. A bulk send never waits on it: if the `net` task holds it,
//!   the send returns `WouldBlock` and may go again at once. An edge that is not paced is one
//!   atomic load on the send.
//! - **No reset when the last data channel closes** (weighed, and declined). A relink closes
//!   every channel of the old epoch and opens the new epoch's at once, and a ring between two
//!   requests closes nothing, so the ring that comes back is usually the one the last breach
//!   measured. A reset on close would let the first window after it climb into the saturation that
//!   breach found, as a reset after a short pause does in the simulator (RFC-0031 §7.5, note of
//!   2026-10-23). The reset after [`RESET_AFTER`] windows with no sample still runs, and an owner
//!   that knows an edge has no data plane left can call [`PacedEdges::pace`] again, which starts
//!   a new pacer.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};

use syntropy_core::clock::Clock;
use syntropy_core::policy::Fraction;
use syntropy_core::{BytesPerSec, MonoTime, Nanos};

/// The window over which the data channel's p99 is taken (RFC-0031 §7.5).
pub const WINDOW: Nanos = Nanos(250_000_000);

/// The share of β the rate grows by after a window within the bound (RFC-0031 §7.5).
pub const INCREASE: f64 = 0.05;

/// Steps of β the ceiling sits below the rate that broke the bound: 2, so 10 % of β \[target\].
pub const CEILING_MARGIN: u64 = 2;

/// Windows within the bound before the ceiling first rises one step again: 64, 16 s \[target\].
/// Each breach after a rise doubles it, up to [`REPROBE_MAX`].
pub const REPROBE_AFTER: u32 = 64;

/// The longest wait between re-probes: 1,024 windows, 256 s \[target\].
pub const REPROBE_MAX: u32 = 1_024;

/// Windows in a row with no α sample before the ceiling and the re-probe wait reset: the longest
/// re-probe wait, 1,024 windows, 256 s \[target\]. A pause no longer than that keeps what the
/// last breach found, as the re-probes do.
pub const RESET_AFTER: u32 = REPROBE_MAX;

/// The most α samples one window keeps; the data channel gives about one per frame, far fewer.
pub const MAX_SAMPLES: usize = 1 << 16;

const NANOS_PER_SEC: u128 = 1_000_000_000;

/// A token bucket in whole bytes: it fills at `rate` bytes a second up to `burst` bytes, and a
/// send takes its size. A send larger than the burst goes once the bucket is full, and leaves it
/// in debt, so the long-run rate is exact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenBucket {
    /// Bytes a second.
    rate: u64,
    /// The most tokens, in bytes.
    burst: u64,
    /// Tokens, in byte-nanoseconds per second: bytes × 10⁹. Negative is debt.
    tokens: i128,
    /// When the tokens were last brought up to date.
    at: MonoTime,
}

impl TokenBucket {
    /// A full bucket filling at `rate` bytes a second, holding at most `burst` bytes.
    pub fn new(rate: u64, burst: u64, now: MonoTime) -> TokenBucket {
        TokenBucket {
            rate,
            burst,
            tokens: i128::from(burst) * NANOS_PER_SEC as i128,
            at: now,
        }
    }

    fn cap(&self) -> i128 {
        i128::from(self.burst) * NANOS_PER_SEC as i128
    }

    /// Brings the tokens up to `now`.
    fn refill(&mut self, now: MonoTime) {
        let elapsed = now.saturating_duration_since(self.at).0;
        self.at = self.at.max(now);
        let added = i128::from(self.rate) * i128::from(elapsed);
        self.tokens = self.tokens.saturating_add(added).min(self.cap());
    }

    /// Bytes a second.
    pub fn rate(&self) -> u64 {
        self.rate
    }

    /// Changes the rate from `now`; the tokens up to `now` filled at the old one.
    pub fn set_rate(&mut self, rate: u64, now: MonoTime) {
        self.refill(now);
        self.rate = rate;
    }

    /// The tokens a send of `bytes` needs before it goes: all of them, or the whole burst for a
    /// send larger than it.
    fn needed(&self, bytes: u64) -> i128 {
        i128::from(bytes.min(self.burst)) * NANOS_PER_SEC as i128
    }

    /// Takes `bytes` if the bucket has them now; otherwise says when it will.
    pub fn try_take(&mut self, bytes: u64, now: MonoTime) -> Result<(), MonoTime> {
        self.refill(now);
        if self.tokens >= self.needed(bytes) {
            self.tokens -= i128::from(bytes) * NANOS_PER_SEC as i128;
            Ok(())
        } else {
            Err(self.ready_at(bytes, now))
        }
    }

    /// When a send of `bytes` can go, at the current rate: `now` if it can now.
    pub fn ready_at(&self, bytes: u64, now: MonoTime) -> MonoTime {
        let elapsed = now.saturating_duration_since(self.at).0;
        let tokens = self
            .tokens
            .saturating_add(i128::from(self.rate) * i128::from(elapsed))
            .min(self.cap());
        let deficit = self.needed(bytes) - tokens;
        if deficit <= 0 {
            return now;
        }
        if self.rate == 0 {
            return MonoTime(u64::MAX);
        }
        // Whole nanoseconds, rounded up, so the tokens are there when it comes.
        let wait = deficit.unsigned_abs().div_ceil(u128::from(self.rate));
        now.saturating_add(Nanos(u64::try_from(wait).unwrap_or(u64::MAX)))
    }
}

/// What a [`BulkPacer`] paces against.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PacingConfig {
    /// The edge's β: its goodput.
    pub beta: BytesPerSec,
    /// The data channel's α p99 on the idle edge, as RFC-0010 records it.
    pub idle_p99: Nanos,
    /// `net.bulk_p99_ratio`: the multiple of the idle p99 the data channel may reach (1.2).
    pub p99_ratio: f64,
    /// `net.bulk_floor`: the share of β bulk always keeps (5 %).
    pub floor: Fraction,
    /// The bucket's burst, in bytes: what may go at once.
    pub burst: u64,
}

impl PacingConfig {
    /// An edge's config from `net.bulk_p99_ratio` and `net.bulk_floor` in `net`, the edge's β
    /// and idle p99 from its link profile (RFC-0010), and the bucket's burst.
    pub fn new(
        net: &crate::config::NetSection,
        beta: BytesPerSec,
        idle_p99: Nanos,
        burst: u64,
    ) -> PacingConfig {
        PacingConfig {
            beta,
            idle_p99,
            p99_ratio: net.bulk_p99_ratio,
            floor: net.bulk_floor,
            burst,
        }
    }
}

/// What a [`BulkPacer`] did, for telemetry and tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PacingStats {
    /// Windows closed.
    pub windows: u64,
    /// Windows whose p99 was above the bound, each a halving.
    pub halvings: u64,
    /// Windows within it, each an increase.
    pub increases: u64,
    /// The last closed window's p99, if it had samples.
    pub last_p99: Option<Nanos>,
    /// Bytes the bucket let through.
    pub bytes: u64,
}

/// The pacer of one edge's bulk sends (see the [module](self) docs).
#[derive(Clone, Debug)]
pub struct BulkPacer {
    beta: u64,
    floor: u64,
    step: u64,
    /// The p99 above which the rate halves: `p99_ratio × idle_p99`.
    bound: u64,
    bucket: TokenBucket,
    /// The most growth may reach: β, or [`CEILING_MARGIN`] steps below the rate that last broke
    /// the bound.
    ceiling: u64,
    /// Windows within the bound since the ceiling last moved.
    quiet: u32,
    /// Windows within the bound before the next re-probe.
    reprobe_after: u32,
    /// Whether the ceiling has risen since the last breach.
    probed: bool,
    /// Windows in a row with no α sample.
    empty: u32,
    window_start: MonoTime,
    samples: Vec<u64>,
    stats: PacingStats,
}

impl BulkPacer {
    /// A pacer at the floor, its first window starting `now`.
    pub fn new(config: PacingConfig, now: MonoTime) -> BulkPacer {
        let beta = rate(config.beta.0);
        let floor = rate(config.beta.0 * f64::from(config.floor.get())).clamp(1, beta.max(1));
        let step = rate(config.beta.0 * INCREASE).max(1);
        let bound = (config.idle_p99.0 as f64 * config.p99_ratio).ceil();
        BulkPacer {
            beta,
            floor,
            step,
            bound: if bound.is_finite() && bound >= 0.0 {
                bound as u64
            } else {
                u64::MAX
            },
            bucket: TokenBucket::new(floor, config.burst, now),
            ceiling: beta.max(floor),
            quiet: 0,
            reprobe_after: REPROBE_AFTER,
            probed: false,
            empty: 0,
            window_start: now,
            samples: Vec::new(),
            stats: PacingStats::default(),
        }
    }

    /// The rate now, in bytes a second.
    pub fn rate(&self) -> u64 {
        self.bucket.rate()
    }

    /// The floor and β, in bytes a second.
    pub fn floor(&self) -> u64 {
        self.floor
    }

    pub fn beta(&self) -> u64 {
        self.beta
    }

    /// The most growth may reach now.
    pub fn ceiling(&self) -> u64 {
        self.ceiling
    }

    /// The p99 the data channel may reach before the rate halves.
    pub fn bound(&self) -> Nanos {
        Nanos(self.bound)
    }

    pub fn stats(&self) -> PacingStats {
        self.stats
    }

    /// An α sample of the edge's live data channel.
    pub fn on_alpha(&mut self, sample: Nanos, now: MonoTime) {
        self.on_tick(now);
        if self.samples.len() < MAX_SAMPLES {
            self.samples.push(sample.0);
        }
    }

    /// Closes every window that has ended by `now`, setting the rate from each.
    pub fn on_tick(&mut self, now: MonoTime) {
        while now.saturating_duration_since(self.window_start) >= WINDOW {
            let end = self.window_start.saturating_add(WINDOW);
            self.close_window(end);
            self.window_start = end;
        }
    }

    fn close_window(&mut self, end: MonoTime) {
        let p99 = p99(&mut self.samples);
        self.samples.clear();
        self.stats.windows += 1;
        self.stats.last_p99 = p99.map(Nanos);
        let rate = self.rate();
        self.empty = if p99.is_none() {
            self.empty.saturating_add(1)
        } else {
            0
        };
        if self.empty == RESET_AFTER {
            // The data channel has gone away or gone idle: nothing to protect, so the last
            // breach no longer bounds the rate.
            self.ceiling = self.beta.max(self.floor);
            self.reprobe_after = REPROBE_AFTER;
            self.quiet = 0;
            self.probed = false;
        }
        let next = if p99.is_some_and(|p99| p99 > self.bound) {
            self.stats.halvings += 1;
            self.ceiling = rate
                .saturating_sub(CEILING_MARGIN * self.step)
                .max(self.floor);
            if self.probed {
                self.reprobe_after = self.reprobe_after.saturating_mul(2).min(REPROBE_MAX);
            }
            self.quiet = 0;
            self.probed = false;
            (rate / 2).max(self.floor)
        } else {
            self.stats.increases += 1;
            self.quiet = self.quiet.saturating_add(1);
            if self.ceiling < self.beta && self.quiet >= self.reprobe_after {
                self.ceiling = self.ceiling.saturating_add(self.step).min(self.beta);
                self.quiet = 0;
                self.probed = true;
            }
            rate.saturating_add(self.step).min(self.ceiling)
        };
        self.bucket.set_rate(next, end);
    }

    /// Takes `bytes` of bulk send now, or says when they can go.
    pub fn try_send(&mut self, bytes: u64, now: MonoTime) -> Result<(), MonoTime> {
        self.on_tick(now);
        self.bucket.try_take(bytes, now)?;
        self.stats.bytes += bytes;
        Ok(())
    }

    /// When the current window closes, and the rate is next set.
    pub fn window_end(&self) -> MonoTime {
        self.window_start.saturating_add(WINDOW)
    }

    /// When a send of `bytes` can go at the current rate: `now` if it can now.
    pub fn ready_at(&self, bytes: u64, now: MonoTime) -> MonoTime {
        self.bucket.ready_at(bytes, now)
    }

    /// When a send of `bytes` can go, or the current window closes, whichever is first.
    pub fn next_deadline(&self, bytes: u64, now: MonoTime) -> MonoTime {
        let window_end = self.window_start.saturating_add(WINDOW);
        self.bucket.ready_at(bytes, now).min(window_end)
    }
}

/// The bulk pacing of a real backend's peer edges (see the [module](self) docs): at most one
/// [`BulkPacer`] for each peer, by the peer's endpoint key, shared by every channel to it.
#[derive(Debug, Default)]
pub struct PacedEdges {
    edges: Mutex<HashMap<[u8; 32], Arc<Edge>>>,
}

/// One peer edge: its pacer, if it is paced, and the clock the pacer reads.
#[derive(Debug, Default)]
pub(crate) struct Edge {
    /// Whether `state` holds a pacer, so an edge that is not paced costs a send one load.
    paced: AtomicBool,
    state: Mutex<Option<Paced>>,
}

struct Paced {
    pacer: BulkPacer,
    clock: Arc<dyn Clock>,
}

impl std::fmt::Debug for Paced {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Paced")
            .field("pacer", &self.pacer)
            .finish_non_exhaustive()
    }
}

/// Locks `mutex` past a panic of another holder: each pacer step leaves it consistent.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl PacedEdges {
    /// The edge to `peer`, made unpaced if it has none yet. Channels hold it for their lives.
    pub(crate) fn edge(&self, peer: &[u8; 32]) -> Arc<Edge> {
        Arc::clone(lock(&self.edges).entry(*peer).or_default())
    }

    /// Paces the bulk sends to `peer` from now on with a new pacer of `config`, at its floor,
    /// reading `clock`: every `Bulk` and `Replay` channel to the peer, open or opened later. A
    /// pacer the edge had is replaced.
    pub fn pace(&self, peer: &[u8; 32], config: PacingConfig, clock: Arc<dyn Clock>) {
        let edge = self.edge(peer);
        let pacer = BulkPacer::new(config, clock.now());
        *lock(&edge.state) = Some(Paced { pacer, clock });
        edge.paced.store(true, Ordering::Release);
    }

    /// Stops pacing the bulk sends to `peer`: they are no longer held.
    pub fn unpace(&self, peer: &[u8; 32]) {
        let mut edges = lock(&self.edges);
        if let Some(edge) = edges.get(peer) {
            edge.paced.store(false, Ordering::Release);
            *lock(&edge.state) = None;
            // An edge no channel holds goes; one a channel holds stays, so pacing it again
            // reaches that channel.
            if Arc::strong_count(edge) == 1 {
                edges.remove(peer);
            }
        }
    }

    /// A copy of the pacer of the bulk sends to `peer`, its windows closed up to its clock's
    /// time, if the edge is paced.
    pub fn pacer(&self, peer: &[u8; 32]) -> Option<BulkPacer> {
        let edge = lock(&self.edges).get(peer).cloned()?;
        let mut state = lock(&edge.state);
        let paced = state.as_mut()?;
        let now = paced.clock.now();
        paced.pacer.on_tick(now);
        Some(paced.pacer.clone())
    }
}

impl Edge {
    /// Takes `bytes` of bulk send from the edge's pacer: `Ok` if the edge is not paced or the
    /// tokens are there; otherwise when to try again, on the pacer's clock. While the `net` task
    /// holds the edge, that is a time already past: try again at once.
    pub(crate) fn try_send(&self, bytes: u64) -> Result<(), MonoTime> {
        if !self.paced.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            // Held for one pacer step.
            Err(TryLockError::WouldBlock) => return Err(MonoTime(0)),
        };
        let Some(paced) = state.as_mut() else {
            return Ok(());
        };
        let now = paced.clock.now();
        paced.pacer.try_send(bytes, now)
    }

    /// An α sample of one of the edge's data channels, taken now.
    pub(crate) fn on_alpha(&self, sample: Nanos) {
        if !self.paced.load(Ordering::Acquire) {
            return;
        }
        if let Some(paced) = lock(&self.state).as_mut() {
            let now = paced.clock.now();
            paced.pacer.on_alpha(sample, now);
        }
    }
}

/// Bytes a second, rounded, from a rate in bytes a second.
fn rate(bytes_per_sec: f64) -> u64 {
    if bytes_per_sec.is_finite() && bytes_per_sec > 0.0 {
        bytes_per_sec.round() as u64
    } else {
        0
    }
}

/// The nearest-rank p99 of `samples`, which it sorts; `None` if there are none.
fn p99(samples: &mut [u64]) -> Option<u64> {
    if samples.is_empty() {
        return None;
    }
    samples.sort_unstable();
    let rank = (samples.len() * 99).div_ceil(100);
    Some(samples[rank.max(1) - 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    const MB: u64 = 1_000_000;

    fn at_ms(ms: u64) -> MonoTime {
        MonoTime(ms * 1_000_000)
    }

    fn config() -> PacingConfig {
        PacingConfig {
            beta: BytesPerSec(100.0 * MB as f64),
            idle_p99: Nanos(1_000_000),
            p99_ratio: 1.2,
            floor: Fraction::new(0.05).unwrap(),
            burst: 64 * 1024,
        }
    }

    #[test]
    fn the_bucket_passes_its_rate_and_no_more() {
        let mut bucket = TokenBucket::new(1_000, 100, MonoTime(0));
        assert_eq!(bucket.try_take(100, MonoTime(0)), Ok(()), "a full burst");
        // 100 bytes at 1,000 B/s take 100 ms to come back.
        assert_eq!(bucket.try_take(100, MonoTime(0)), Err(at_ms(100)));
        assert_eq!(
            bucket.try_take(100, MonoTime(at_ms(100).0 - 1)),
            Err(at_ms(100))
        );
        assert_eq!(bucket.try_take(100, at_ms(100)), Ok(()));
        // It never holds more than the burst.
        assert_eq!(bucket.try_take(100, at_ms(10_000)), Ok(()));
        assert!(bucket.try_take(1, at_ms(10_000)).is_err());
        // A send larger than the burst goes when the bucket is full, into debt.
        let mut bucket = TokenBucket::new(1_000, 100, MonoTime(0));
        assert_eq!(bucket.try_take(300, MonoTime(0)), Ok(()));
        // 200 bytes of debt, then 100 for the next: 300 ms.
        assert_eq!(bucket.ready_at(100, MonoTime(0)), at_ms(300));
    }

    #[test]
    fn the_rate_changes_from_the_moment_it_is_set() {
        let mut bucket = TokenBucket::new(1_000, 1_000, MonoTime(0));
        bucket.try_take(1_000, MonoTime(0)).unwrap();
        // 100 ms at 1,000 B/s gives 100 bytes, then 100 ms at 9,000 B/s gives 900.
        bucket.set_rate(9_000, at_ms(100));
        assert_eq!(bucket.try_take(1_000, at_ms(200)), Ok(()));
    }

    #[test]
    fn p99_is_the_nearest_rank() {
        assert_eq!(p99(&mut []), None);
        assert_eq!(p99(&mut [5]), Some(5));
        let mut hundred: Vec<u64> = (1..=100).rev().collect();
        assert_eq!(p99(&mut hundred), Some(99));
        let mut two_hundred: Vec<u64> = (1..=200).collect();
        assert_eq!(p99(&mut two_hundred), Some(198));
    }

    /// RFC-0031 §7.5: a window above 1.2 × idle p99 halves the rate, one within it adds 5 % of β,
    /// the rate stays between the floor and β, and it starts at the floor.
    #[test]
    fn aimd_halves_above_the_bound_and_adds_five_percent_of_beta_within_it() {
        let mut pacer = BulkPacer::new(config(), MonoTime(0));
        let (floor, step, beta) = (5 * MB, 5 * MB, 100 * MB);
        assert_eq!(
            (pacer.rate(), pacer.floor(), pacer.beta()),
            (floor, floor, beta)
        );
        assert_eq!(pacer.bound(), Nanos(1_200_000));
        // Quiet windows: +5 MB/s each, up to β.
        for window in 1..=30u64 {
            pacer.on_alpha(Nanos(1_000_000), MonoTime(window * WINDOW.0 - 1));
            pacer.on_tick(MonoTime(window * WINDOW.0));
            assert_eq!(
                pacer.rate(),
                (floor + window * step).min(beta),
                "window {window}"
            );
        }
        // One window whose p99 is above 1.2 ms halves it; at 1.2 ms exactly it does not.
        let t = 30 * WINDOW.0;
        for i in 0..100 {
            let sample = if i < 2 { 1_300_000 } else { 1_000_000 };
            pacer.on_alpha(Nanos(sample), MonoTime(t + i));
        }
        pacer.on_tick(MonoTime(t + WINDOW.0));
        assert_eq!(pacer.rate(), beta / 2);
        for i in 0..100 {
            let sample = if i < 2 { 1_200_000 } else { 1_000_000 };
            pacer.on_alpha(Nanos(sample), MonoTime(t + WINDOW.0 + i));
        }
        pacer.on_tick(MonoTime(t + 2 * WINDOW.0));
        assert_eq!(pacer.rate(), beta / 2 + step);
        // Halving never goes below the floor.
        for window in 3..=20u64 {
            pacer.on_alpha(Nanos(10_000_000), MonoTime(t + window * WINDOW.0 - 1));
        }
        pacer.on_tick(MonoTime(t + 20 * WINDOW.0));
        assert_eq!(pacer.rate(), floor);
        let stats = pacer.stats();
        assert_eq!(stats.windows, 50);
        assert_eq!(stats.halvings, 1 + 18);
    }

    /// After a halving from R, growth stops two steps below R; the ceiling rises one step after
    /// 64 quiet windows, and a breach after that rise doubles the wait to 128.
    #[test]
    fn the_ceiling_holds_the_rate_below_the_last_breach_and_reprobes_rarely() {
        let mut pacer = BulkPacer::new(config(), MonoTime(0));
        let step = 5 * MB;
        let mut window = 0u64;
        let mut close = |pacer: &mut BulkPacer, sample: u64| {
            window += 1;
            pacer.on_alpha(Nanos(sample), MonoTime(window * WINDOW.0 - 1));
            pacer.on_tick(MonoTime(window * WINDOW.0));
        };
        // Climb to 80 MB/s, then breach there.
        while pacer.rate() < 80 * MB {
            close(&mut pacer, 1_000_000);
        }
        close(&mut pacer, 2_000_000);
        assert_eq!(
            (pacer.rate(), pacer.ceiling()),
            (40 * MB, 80 * MB - 2 * step)
        );
        // It climbs back to the ceiling and stays there.
        for _ in 0..63 {
            close(&mut pacer, 1_000_000);
        }
        assert_eq!(pacer.rate(), 70 * MB);
        assert_eq!(pacer.ceiling(), 70 * MB);
        // The 64th quiet window raises the ceiling one step.
        close(&mut pacer, 1_000_000);
        assert_eq!(pacer.ceiling(), 75 * MB);
        close(&mut pacer, 1_000_000);
        assert_eq!(pacer.rate(), 75 * MB);
        // A breach after that rise: the ceiling drops, and the next rise waits 128 windows.
        close(&mut pacer, 2_000_000);
        assert_eq!(pacer.ceiling(), 75 * MB - 2 * step);
        for _ in 0..127 {
            close(&mut pacer, 1_000_000);
        }
        assert_eq!(pacer.ceiling(), 65 * MB);
        close(&mut pacer, 1_000_000);
        assert_eq!(pacer.ceiling(), 70 * MB);
    }

    /// 1,024 windows in a row with no sample reset the ceiling to β and the re-probe wait to 64,
    /// so after breaches the rate climbs back to β a step a window; 1,023 do not.
    #[test]
    fn a_sustained_absence_of_the_data_channel_resets_the_ceiling() {
        let mut pacer = BulkPacer::new(config(), MonoTime(0));
        let step = 5 * MB;
        let mut window = 0u64;
        let mut close = |pacer: &mut BulkPacer, sample: Option<u64>| {
            window += 1;
            if let Some(sample) = sample {
                pacer.on_alpha(Nanos(sample), MonoTime(window * WINDOW.0 - 1));
            }
            pacer.on_tick(MonoTime(window * WINDOW.0));
        };
        // Climb to 60 MB/s and breach: the ceiling is 50 MB/s. Then rise and breach four times,
        // which doubles the re-probe wait to its longest.
        while pacer.rate() < 60 * MB {
            close(&mut pacer, Some(1_000_000));
        }
        close(&mut pacer, Some(2_000_000));
        for _ in 0..4 {
            let ceiling = pacer.ceiling();
            while pacer.ceiling() == ceiling {
                close(&mut pacer, Some(1_000_000));
            }
            close(&mut pacer, Some(2_000_000));
        }
        assert_eq!(pacer.reprobe_after, REPROBE_MAX);
        let ceiling = pacer.ceiling();
        assert_eq!(ceiling, 30 * MB);
        // The data channel goes away: 1,023 windows with no sample keep the ceiling, and the
        // rate waits at it.
        for _ in 0..RESET_AFTER - 1 {
            close(&mut pacer, None);
        }
        assert_eq!(pacer.ceiling(), ceiling);
        assert_eq!(pacer.rate(), ceiling);
        // The 1,024th resets it, and the rate climbs past the old ceiling a step a window, to β.
        close(&mut pacer, None);
        assert_eq!(pacer.ceiling(), 100 * MB);
        assert_eq!(pacer.reprobe_after, REPROBE_AFTER);
        while pacer.rate() < 100 * MB {
            let before = pacer.rate();
            close(&mut pacer, None);
            assert_eq!(pacer.rate(), (before + step).min(100 * MB));
        }
        // A breach after that starts the re-probe wait at 64 again, not at its doubled value.
        close(&mut pacer, Some(2_000_000));
        assert_eq!(pacer.ceiling(), 90 * MB);
        for _ in 0..63 {
            close(&mut pacer, Some(1_000_000));
        }
        assert_eq!(pacer.ceiling(), 90 * MB);
        close(&mut pacer, Some(1_000_000));
        assert_eq!(pacer.ceiling(), 95 * MB);
    }

    /// A data channel that pauses for fewer than 1,024 windows between bursts keeps the ceiling,
    /// so on a model edge that saturates above 57 MB/s the p99 over a 4,000-window run stays
    /// within 1.2 × idle, for each pattern of bursts and pauses here.
    #[test]
    fn short_pauses_in_the_data_channel_keep_the_ceiling() {
        let bound = 1_200_000;
        for (on, off) in [(1, 0), (8, 1), (20, 4), (40, 2), (40, 63), (40, 200)] {
            let mut pacer = BulkPacer::new(config(), MonoTime(0));
            let mut samples = Vec::new();
            for window in 1..=4_000u64 {
                if (window - 1) % (on + off) < on {
                    // 125 samples a window, as a data channel sending every 2 ms gives.
                    let alpha = if pacer.rate() > 57 * MB {
                        2_000_000
                    } else {
                        1_000_000
                    };
                    for i in 0..125 {
                        pacer.on_alpha(Nanos(alpha), MonoTime((window - 1) * WINDOW.0 + i));
                    }
                    samples.extend([alpha; 125]);
                }
                pacer.on_tick(MonoTime(window * WINDOW.0));
            }
            let breaches = pacer.stats().halvings;
            let p99 = p99(&mut samples).unwrap();
            assert!(
                p99 <= bound,
                "{on} on, {off} off: p99 {p99} ns, {breaches} halvings"
            );
            assert!(breaches <= 8, "{on} on, {off} off: {breaches} halvings");
        }
    }

    /// A window with no sample, when no data channel is live, counts as within the bound; and
    /// windows that pass without a tick are each closed in turn.
    #[test]
    fn an_empty_window_grows_and_missed_windows_are_each_closed() {
        let mut pacer = BulkPacer::new(config(), MonoTime(0));
        pacer.on_tick(MonoTime(4 * WINDOW.0));
        assert_eq!(pacer.stats().windows, 4);
        assert_eq!(pacer.rate(), 25 * MB);
    }

    /// The pacer lets bulk through at its rate: over a second at the floor, 5 MB plus the burst.
    #[test]
    fn the_pacer_paces_at_its_rate() {
        let mut pacer = BulkPacer::new(
            PacingConfig {
                // A bound no sample reaches, but samples in every window, keep it at the floor.
                idle_p99: Nanos(0),
                ..config()
            },
            MonoTime(0),
        );
        let mut now = MonoTime(0);
        let frame = 16 * 1024;
        while now < at_ms(1_000) {
            pacer.on_alpha(Nanos(1), now);
            match pacer.try_send(frame, now) {
                Ok(()) => {}
                Err(when) => now = pacer.next_deadline(frame, now).max(when.min(at_ms(1_000))),
            }
        }
        assert_eq!(pacer.rate(), 5 * MB);
        let sent = pacer.stats().bytes;
        assert!(sent <= 5 * MB + 64 * 1024 + frame, "{sent}");
        assert!(sent >= 5 * MB - frame, "{sent}");
    }
}
