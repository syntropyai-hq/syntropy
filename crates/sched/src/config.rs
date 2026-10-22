// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `sched.*` configuration section (RFC-0012 §12; RFC-0054 §5.1).
//!
//! Every key is instance scope, as RFC-0012 §12 says of the `sched.*` keys. A node's files
//! cannot set one outside a `[policy."<model>"]` table (RFC-0054 §2.2 Rule 3). The names and
//! defaults are RFC-0012 §12's. RFC-0012 gives no reload classes or types, so this section
//! chooses them:
//!
//! - times are [`Duration`]s, because RFC-0054 §3 never accepts a bare integer where a unit is
//!   meant, so `sched.spin_us = "50us"`;
//! - counts are `u32`;
//! - a key whose RFC-0012 default depends on the instance or the power source has a computed
//!   default and is an `Option`, `None` until the scheduler applies the rule;
//! - a key the head reads each step is `Hot` (RFC-0054 §9.1 lists queue limits as hot); a key
//!   that changes the lane credits or the captured buckets is `Instance`.
//!
//! [`SchedSection::from_policy`] builds the section from the `sched` table of an instance's
//! policy and rejects every invalid value with its RFC-0054 §8 code.

use serde::Deserialize;
use syntropy_config::{
    Code, ConfigSection, DefaultSpec, Duration, Value, ValueError, ValueErrorKind,
};

/// Scheduler settings for one instance (RFC-0012 §12).
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "sched", scope = Instance, owner = "RFC-0012")]
pub struct SchedSection {
    /// How long a dispatcher spins before it parks (ADR-0012 item 5; RFC-0012 §3.3).
    #[config(reload = Hot, default = computed("50us on mains power, 10us on battery"))]
    pub spin_us: Option<Duration>,
    /// How far ahead of a predicted frame a parked stage is woken (RFC-0012 §3.3).
    #[config(reload = Hot, default = "300us")]
    pub prewake_us: Duration,
    /// The cap on a prefill chunk's time while an interactive sequence decodes (RFC-0012 §4.3).
    #[config(
        reload = Hot,
        default = computed("0.5 x the active sequence's predicted t_tok, at least 64 tokens of chunk")
    )]
    pub decode_stall_budget: Option<Duration>,
    /// Prefill-lane chunks in flight per instance (RFC-0012 §4.3).
    #[config(reload = Instance, default = computed("D, the instance's stage count"), min = 1)]
    pub prefill_inflight: Option<u32>,
    /// Throughput mode: decode steps per prefill step, so one prefill step per this many
    /// decode steps (RFC-0012 §6.1).
    #[config(reload = Hot, default = computed("G, the instance's group count"), min = 1)]
    pub prefill_share: Option<u32>,
    /// The head's admission queue length (RFC-0012 §10.1).
    #[config(reload = Hot, default = 64)]
    pub max_queued: u32,
    /// Latency mode: the most sequences decoding in one step on dense and RAM-resident MoE
    /// instances; 1 forces serial decoding. NVMe-tiered MoE instances always use 1
    /// (RFC-0012 §4.6). Default 4 [target]. The step uses the `Batch { rows }` bucket at 2 or 4
    /// rows, so 4 is the largest value.
    #[config(reload = Instance, default = 4, min = 1, max = 4)]
    pub latency_decode_rows: u32,
    /// The longest scheduler-initiated pause (RFC-0012 §10.3).
    #[config(reload = Hot, default = "30s")]
    pub max_pause: Duration,
}

/// `sched.spin_us` on mains power when no policy sets it: 50 µs [target] (ADR-0012 item 5).
pub const SPIN_MAINS: Duration = Duration::from_nanos(50_000);
/// `sched.spin_us` on battery when no policy sets it: 10 µs [target] (ADR-0012 item 5).
pub const SPIN_BATTERY: Duration = Duration::from_nanos(10_000);

impl SchedSection {
    /// The spin budget: the policy's value, else the default for the power source.
    #[must_use]
    pub fn spin(&self, on_battery: bool) -> Duration {
        self.spin_us
            .unwrap_or(if on_battery { SPIN_BATTERY } else { SPIN_MAINS })
    }

    /// Builds the section from `values`, the `sched` table of an instance's policy (RFC-0054
    /// §5.4). A key the table leaves out takes its default; a computed default stays `None`.
    ///
    /// # Errors
    ///
    /// Every rejected key, each with its RFC-0054 §8 code: an unknown key, a value of the
    /// wrong type, bad unit syntax, or a value outside the key's range. `values` that is not
    /// a table is one `wrong_type` rejection for `sched`.
    pub fn from_policy(values: &Value) -> Result<Self, Vec<Rejection>> {
        let Value::Table(table) = values else {
            return Err(vec![Rejection::value(
                "sched",
                ValueError::new(ValueErrorKind::WrongType, "`sched` must be a table"),
            )]);
        };
        let mut rejected = Vec::new();
        for name in table.keys() {
            if Self::SCHEMA.key(name).is_none() {
                rejected.push(Rejection {
                    key: format!("sched.{name}"),
                    code: Code::UnknownKey,
                    message: format!("`sched.{name}` is not a scheduler key"),
                });
            }
        }
        let mut get = Fields {
            table: values,
            rejected: &mut rejected,
        };
        let spin_us = get.optional("spin_us", duration);
        let prewake_us = get.required("prewake_us", duration);
        let decode_stall_budget = get.optional("decode_stall_budget", duration);
        let prefill_inflight = get.optional("prefill_inflight", count);
        let prefill_share = get.optional("prefill_share", count);
        let max_queued = get.required("max_queued", count);
        let latency_decode_rows = get.required("latency_decode_rows", count);
        let max_pause = get.required("max_pause", duration);
        let (
            Some(spin_us),
            Some(prewake_us),
            Some(decode_stall_budget),
            Some(prefill_inflight),
            Some(prefill_share),
            Some(max_queued),
            Some(latency_decode_rows),
            Some(max_pause),
        ) = (
            spin_us,
            prewake_us,
            decode_stall_budget,
            prefill_inflight,
            prefill_share,
            max_queued,
            latency_decode_rows,
            max_pause,
        )
        else {
            return Err(rejected);
        };
        let section = Self {
            spin_us,
            prewake_us,
            decode_stall_budget,
            prefill_inflight,
            prefill_share,
            max_queued,
            latency_decode_rows,
            max_pause,
        };
        if rejected.is_empty() {
            rejected.extend(
                section
                    .check_ranges()
                    .into_iter()
                    .map(|e| Rejection::value(e.key, e.error)),
            );
        }
        if rejected.is_empty() {
            Ok(section)
        } else {
            Err(rejected)
        }
    }
}

/// A rejected `sched.*` value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rejection {
    /// The key's full path.
    pub key: String,
    /// The RFC-0054 §8 error code: `wrong_type`, `unknown_key`, `out_of_range` or `bad_unit`,
    /// since no `sched.*` key is a selector, regex or secret.
    pub code: Code,
    /// What is wrong and what is accepted.
    pub message: String,
}

impl Rejection {
    fn value(key: impl Into<String>, error: ValueError) -> Self {
        Self {
            key: key.into(),
            code: Code::from(error.kind()),
            message: error.message().to_owned(),
        }
    }
}

/// Reads the section's fields from the policy table, collecting rejections. Each read gives
/// `None` for a rejected value; an optional key's read gives `Some(None)` when it is absent.
struct Fields<'a> {
    table: &'a Value,
    rejected: &'a mut Vec<Rejection>,
}

impl Fields<'_> {
    fn parse<T>(
        &mut self,
        name: &str,
        parse: fn(&Value) -> Result<T, ValueError>,
    ) -> Option<Option<T>> {
        let Some(value) = self.table.get(name) else {
            return Some(None);
        };
        match parse(value) {
            Ok(v) => Some(Some(v)),
            Err(e) => {
                self.rejected
                    .push(Rejection::value(format!("sched.{name}"), e));
                None
            }
        }
    }

    /// A key with a computed default: `None` inside when the table leaves it out.
    fn optional<T>(
        &mut self,
        name: &str,
        parse: fn(&Value) -> Result<T, ValueError>,
    ) -> Option<Option<T>> {
        self.parse(name, parse)
    }

    /// A key with a fixed default, which the table may leave out.
    fn required<T: serde::de::DeserializeOwned>(
        &mut self,
        name: &str,
        parse: fn(&Value) -> Result<T, ValueError>,
    ) -> Option<T> {
        match self.parse(name, parse)? {
            Some(v) => Some(v),
            None => {
                let key = SchedSection::SCHEMA
                    .key(name)
                    .expect("every field is a key of the schema");
                match key.default {
                    DefaultSpec::Value(v) => Some(
                        v.deserialize()
                            .expect("`SchedSection::check_defaults` keeps every default valid"),
                    ),
                    DefaultSpec::Unset | DefaultSpec::Computed(_) => {
                        unreachable!("a required field has a fixed default")
                    }
                }
            }
        }
    }
}

/// A duration in RFC-0054 §3's syntax.
fn duration(value: &Value) -> Result<Duration, ValueError> {
    match value {
        Value::String(s) => s.parse(),
        _ => Err(ValueError::new(
            ValueErrorKind::WrongType,
            "expected a duration string, such as \"50us\" or \"30s\"",
        )),
    }
}

/// A non-negative count.
fn count(value: &Value) -> Result<u32, ValueError> {
    match value {
        Value::Integer(i) => u32::try_from(*i).map_err(|_| {
            ValueError::new(
                ValueErrorKind::OutOfRange,
                format!("{i} is outside 0..={}", u32::MAX),
            )
        }),
        _ => Err(ValueError::new(
            ValueErrorKind::WrongType,
            "expected a whole number",
        )),
    }
}

#[cfg(test)]
mod tests;
