// SPDX-License-Identifier: MIT OR Apache-2.0

//! The half of [`super`] that exists only with the `fault-injection` feature: the plan, its
//! triggers, the check every fault point calls, and the counters.
//!
//! `cargo xtask check-deps` rule 10 looks for this module's path in the core binary's symbol
//! table, so every feature-only item lives here.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fmt;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use syntropy_core::{Clock, DeviceId, Entropy, LinkId, MonoTime, NodeId, SequenceId};

use super::{Ctx, FaultEffect};

/// The environment variable that names a fault plan's TOML file, read at process start
/// (RFC-0060 §9.2).
pub const PLAN_ENV: &str = "SYNTROPY_FAULT_PLAN";

/// The fault rules a process runs with (RFC-0060 §9.2).
///
/// In TOML, each rule is a `[[rules]]` table. A filter is a sub-table, and a node is its 32-byte
/// ID:
///
/// ```toml
/// [[rules]]
/// point = "io.ring.submit"
/// when = { nth = 3 }
/// effect = { error = "Device" }
///
/// [[rules]]
/// point = "net.channel.send"
/// when = { probability = { p = 0.25, seed = 7 } }
/// effect = { delay = 5000000 }
///
/// [rules.filter.device]
/// node = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
/// ordinal = 0
/// ```
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FaultPlan {
    /// The rules, in order: a hit gets the effect of the first rule that matches it and fires.
    #[serde(default)]
    pub rules: Vec<FaultRule>,
}

/// One rule of a [`FaultPlan`] (RFC-0060 §9.2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FaultRule {
    /// The fault point's name, such as `io.ring.submit`.
    pub point: String,
    /// When a matching hit fires.
    pub when: Trigger,
    /// Which hits match, by where they were reached. `None` matches every hit of the point.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<CtxFilter>,
    /// What a fire does.
    pub effect: FaultEffect,
}

/// When a rule's matching hits fire (RFC-0060 §9.2). A matching hit is one at the rule's point
/// that passes its filter. Each rule counts its own from 1, whether or not an earlier rule fired
/// on the hit.
///
/// In TOML: `when = "always"`, `when = { nth = 3 }`,
/// `when = { probability = { p = 0.25, seed = 7 } }`, `when = { after = 1000000 }` and
/// `when = { window = { from = 1000000, to = 2000000 } }`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Trigger {
    /// Every matching hit.
    Always,
    /// Only the n-th matching hit, so at most once. n starts at 1.
    Nth(u64),
    /// Each matching hit with probability `p`. The k-th matching hit fires when a keyed hash of
    /// (`seed`, the point, k), mapped to [0, 1), is below `p`, so the draws hold no state and a
    /// seed replays them.
    Probability {
        /// The chance of firing, in [0, 1].
        p: f64,
        /// The seed of the draws.
        seed: u64,
    },
    /// Every matching hit at or after this instant of the installed clock.
    After(MonoTime),
    /// Every matching hit from `from` up to, but not including, `to`, on the installed clock.
    Window {
        /// The first instant that fires.
        from: MonoTime,
        /// The first instant after the window.
        to: MonoTime,
    },
}

impl Trigger {
    /// A [`Trigger::Probability`] whose seed is drawn from `entropy`, such as the simulator's
    /// seeded source, so the run's seed decides the fires (RFC-0060 §2.1, rule T1).
    pub fn probability(p: f64, entropy: &dyn Entropy) -> Trigger {
        Trigger::Probability {
            p,
            seed: entropy.u64(),
        }
    }
}

/// Which hits a rule matches, by the [`Ctx`] they were reached in (RFC-0060 §9.2). Each field
/// that is set must equal the context's, and a context without that field does not match.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CtxFilter {
    /// The node.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node: Option<NodeId>,
    /// The device.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<DeviceId>,
    /// The link.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link: Option<LinkId>,
    /// The sequence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequence: Option<SequenceId>,
}

impl CtxFilter {
    /// Whether a hit reached in `ctx` passes the filter.
    #[must_use]
    pub fn matches(&self, ctx: &Ctx) -> bool {
        fn field<T: PartialEq>(want: Option<T>, have: Option<T>) -> bool {
            want.is_none_or(|want| have == Some(want))
        }
        field(self.node, ctx.node)
            && field(self.device, ctx.device)
            && field(self.link, ctx.link)
            && field(self.sequence, ctx.sequence)
    }
}

/// Why a fault plan could not be read or installed.
#[derive(Debug)]
pub enum PlanError {
    /// The plan file could not be read.
    Read {
        /// The file.
        path: PathBuf,
        /// The I/O error.
        error: std::io::Error,
    },
    /// The plan is not valid TOML of the plan's shape.
    Parse(String),
    /// A rule is malformed.
    Rule {
        /// The rule's position in the plan, from 0.
        index: usize,
        /// What is wrong with it.
        reason: String,
    },
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlanError::Read { path, error } => {
                write!(f, "cannot read fault plan {}: {error}", path.display())
            }
            PlanError::Parse(error) => write!(f, "fault plan is not valid: {error}"),
            PlanError::Rule { index, reason } => write!(f, "fault plan rule {index}: {reason}"),
        }
    }
}

impl std::error::Error for PlanError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PlanError::Read { error, .. } => Some(error),
            _ => None,
        }
    }
}

impl FaultPlan {
    /// Parses a plan from TOML and validates it.
    ///
    /// # Errors
    ///
    /// [`PlanError::Parse`] for TOML that is not a plan, and [`PlanError::Rule`] for a rule
    /// [`FaultPlan::validate`] rejects.
    pub fn from_toml(text: &str) -> Result<FaultPlan, PlanError> {
        let plan: FaultPlan =
            toml::from_str(text).map_err(|error| PlanError::Parse(error.to_string()))?;
        plan.validate()?;
        Ok(plan)
    }

    /// Checks every rule: a point name that is not empty, `Nth` from 1, a `Probability` in
    /// [0, 1] and a `Window` that ends after it starts.
    ///
    /// # Errors
    ///
    /// [`PlanError::Rule`] for the first rule that breaks one of these.
    pub fn validate(&self) -> Result<(), PlanError> {
        for (index, rule) in self.rules.iter().enumerate() {
            let reason = if rule.point.is_empty() {
                Some("the point name is empty".to_owned())
            } else {
                match rule.when {
                    Trigger::Nth(0) => Some("`nth` counts hits from 1, not 0".to_owned()),
                    Trigger::Probability { p, .. } if !(0.0..=1.0).contains(&p) => {
                        Some(format!("probability {p} is not in [0, 1]"))
                    }
                    Trigger::Window { from, to } if from >= to => Some(format!(
                        "the window from {} ns to {} ns is empty",
                        from.0, to.0
                    )),
                    _ => None,
                }
            };
            if let Some(reason) = reason {
                return Err(PlanError::Rule { index, reason });
            }
        }
        Ok(())
    }
}

/// How often a fault point was reached and fired while a plan was installed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PointCounts {
    /// Hits: each time the point was reached.
    pub hits: u64,
    /// Fires: each hit that fired a rule.
    pub fires: u64,
}

struct Installed {
    plan: FaultPlan,
    clock: Arc<dyn Clock>,
    /// Matching hits so far, per rule.
    matched: Vec<u64>,
}

struct State {
    installed: Option<Installed>,
    /// Bumped by every `install` and `clear`, which releases the threads a `Hang` holds.
    generation: u64,
    counts: BTreeMap<&'static str, PointCounts>,
}

static STATE: Mutex<State> = Mutex::new(State {
    installed: None,
    generation: 0,
    counts: BTreeMap::new(),
});
static RELEASE: Condvar = Condvar::new();

/// The state, even after a thread panicked while it held the lock: an injected `Panic` never
/// holds it, and the state stays consistent at every point a panic could leave it.
fn state() -> MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Installs `plan`, which replaces any plan installed before, with `clock` as the time its
/// `After` and `Window` rules read. Counters carry on until [`reset_counters`].
///
/// # Errors
///
/// [`PlanError::Rule`] when [`FaultPlan::validate`] rejects the plan; the plan installed before
/// stays.
pub fn install(plan: FaultPlan, clock: Arc<dyn Clock>) -> Result<(), PlanError> {
    plan.validate()?;
    let mut state = state();
    state.installed = Some(Installed {
        matched: vec![0; plan.rules.len()],
        plan,
        clock,
    });
    state.generation += 1;
    RELEASE.notify_all();
    Ok(())
}

/// Removes the installed plan, so no point fires, and releases every thread a `Hang` holds.
pub fn clear() {
    let mut state = state();
    state.installed = None;
    state.generation += 1;
    RELEASE.notify_all();
}

/// Reads a plan from the TOML file at `path` and installs it.
///
/// # Errors
///
/// [`PlanError::Read`] when the file cannot be read, and the errors of [`FaultPlan::from_toml`].
pub fn load_from_path(path: &Path, clock: Arc<dyn Clock>) -> Result<(), PlanError> {
    let text = std::fs::read_to_string(path).map_err(|error| PlanError::Read {
        path: path.to_owned(),
        error,
    })?;
    install(FaultPlan::from_toml(&text)?, clock)
}

/// At process start: installs the plan whose file [`PLAN_ENV`] names. Returns whether it named
/// one; unset or empty, it names none and nothing changes.
///
/// # Errors
///
/// The errors of [`load_from_path`].
pub fn load_from_env(clock: Arc<dyn Clock>) -> Result<bool, PlanError> {
    load_from_var(std::env::var_os(PLAN_ENV), clock)
}

/// [`load_from_env`] on the variable's value, so tests need not set the process environment.
pub(super) fn load_from_var(
    value: Option<OsString>,
    clock: Arc<dyn Clock>,
) -> Result<bool, PlanError> {
    match value {
        Some(path) if !path.is_empty() => {
            load_from_path(Path::new(&path), clock)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// Every point reached while a plan was installed, by name, with its hits and fires since the
/// last [`reset_counters`].
#[must_use]
pub fn counters() -> BTreeMap<&'static str, PointCounts> {
    state().counts.clone()
}

/// Every point that fired since the last [`reset_counters`], by name.
#[must_use]
pub fn fired() -> Vec<&'static str> {
    state()
        .counts
        .iter()
        .filter(|(_, counts)| counts.fires > 0)
        .map(|(point, _)| *point)
        .collect()
}

/// Sets every counter back to zero.
pub fn reset_counters() {
    state().counts.clear();
}

thread_local! {
    static HERE: Cell<Ctx> = const {
        Cell::new(Ctx {
            node: None,
            device: None,
            link: None,
            sequence: None,
        })
    };
}

impl Ctx {
    /// The context this thread is working in: the one the innermost live [`Ctx::enter`] set, or
    /// the empty context.
    #[must_use]
    pub fn here() -> Ctx {
        HERE.with(Cell::get)
    }

    /// Makes `self` this thread's context until the guard drops, which restores the one before.
    pub fn enter(self) -> CtxGuard {
        CtxGuard {
            before: HERE.with(|here| here.replace(self)),
            not_send: PhantomData,
        }
    }
}

/// Restores the thread's context from before [`Ctx::enter`] when it drops. It stays on the thread
/// that made it, so it cannot restore another thread's context.
#[must_use = "the context is restored when the guard drops"]
#[derive(Debug)]
pub struct CtxGuard {
    before: Ctx,
    not_send: PhantomData<*const ()>,
}

impl Drop for CtxGuard {
    fn drop(&mut self) {
        HERE.with(|here| here.set(self.before));
    }
}

/// splitmix64's finaliser: a bijection of `u64` that mixes every input bit into every output bit.
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// FNV-1a of `bytes`: a fixed hash of the point name, the same on every host and build.
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The draw for the `hit`-th matching hit of a `Probability` rule at `point`: a value in [0, 1)
/// from the top 53 bits of a keyed hash.
pub(super) fn draw(seed: u64, point: &str, hit: u64) -> f64 {
    let bits = mix(mix(seed ^ fnv1a(point.as_bytes())).wrapping_add(hit));
    (bits >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Whether the `hit`-th matching hit of a rule with trigger `when` at `point` fires at `now`.
fn fires(when: Trigger, point: &str, hit: u64, now: impl FnOnce() -> MonoTime) -> bool {
    match when {
        Trigger::Always => true,
        Trigger::Nth(n) => hit == n,
        Trigger::Probability { p, seed } => draw(seed, point, hit) < p,
        Trigger::After(at) => now() >= at,
        Trigger::Window { from, to } => (from..to).contains(&now()),
    }
}

/// The fault point `point`, reached in `ctx`. `fault_point!` calls it; nothing else needs to.
///
/// With no plan installed it returns [`FaultEffect::None`] and counts nothing. Otherwise it counts
/// the hit, and every rule whose point and filter match counts it as that rule's next matching
/// hit. The first of those rules whose trigger fires gives the effect, and the fire is counted.
/// Then the effect is applied:
/// - `Delay` sleeps the thread for its duration, then returns;
/// - `Panic` panics, naming the point;
/// - `Hang` blocks the thread until the plan is cleared or replaced, then returns;
/// - `Error`, `Corrupt` and `None` return at once, for the call site to act on.
///
/// The installed clock is read at most once per hit, and the lock on the plan is never held while
/// an effect is applied.
#[inline(never)]
pub fn check(point: &'static str, ctx: &Ctx) -> FaultEffect {
    let (effect, generation) = {
        let mut state = state();
        let State {
            installed,
            generation,
            counts,
        } = &mut *state;
        let Some(Installed {
            plan,
            clock,
            matched,
        }) = installed.as_mut()
        else {
            return FaultEffect::None;
        };
        let mut now = None;
        let mut effect = None;
        for (rule, matched) in plan.rules.iter().zip(matched.iter_mut()) {
            if rule.point != point || !rule.filter.is_none_or(|filter| filter.matches(ctx)) {
                continue;
            }
            *matched += 1;
            if effect.is_none()
                && fires(rule.when, point, *matched, || {
                    *now.get_or_insert_with(|| clock.now())
                })
            {
                effect = Some(rule.effect);
            }
        }
        let counts = counts.entry(point).or_default();
        counts.hits += 1;
        if effect.is_some() {
            counts.fires += 1;
        }
        (effect.unwrap_or_default(), *generation)
    };
    match effect {
        FaultEffect::Delay(nanos) => std::thread::sleep(Duration::from_nanos(nanos.0)),
        FaultEffect::Panic => panic!("fault point `{point}` fired: an injected panic"),
        FaultEffect::Hang => {
            let mut state = state();
            while state.generation == generation {
                state = RELEASE.wait(state).unwrap_or_else(PoisonError::into_inner);
            }
        }
        FaultEffect::None | FaultEffect::Error(_) | FaultEffect::Corrupt { .. } => {}
    }
    effect
}
