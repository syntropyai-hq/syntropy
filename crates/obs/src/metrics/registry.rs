// SPDX-License-Identifier: MIT OR Apache-2.0

//! The metric registry (RFC-0053 §2.1): metrics are registered and their series bound to label
//! values at `Warming`, before any hot thread records.
//!
//! A series is one metric with one set of label values. Binding resolves the labels once and
//! returns a handle, a [`CounterIx`], [`HistIx`] or [`GaugeIx`], that a hot thread records
//! through, so recording never hashes or compares a label. [`Registry::freeze`] fixes the
//! series into a [`Layout`], from which every thread's cells and the drain are built.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::desc::{LabelKey, MetricDesc, MetricId, MetricKind, Unit};
use super::histogram::HistScale;

/// A series' number in a [`Layout`]: one metric with one set of label values. Telemetry records
/// carry it, and a [`TelemetryRecord::Series`](crate::telemetry::TelemetryRecord::Series)
/// announces what it stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SeriesId(pub u32);

/// A counter series' handle, bound at registration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CounterIx(pub(crate) u32);

/// A histogram series' handle, bound at registration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HistIx(pub(crate) u32);

/// A gauge series' handle, bound at registration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GaugeIx(pub(crate) u32);

/// One bound series: its metric and its label values, in the order of the metric's labels.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SeriesInfo {
    /// The series.
    pub id: SeriesId,
    /// Its metric.
    pub metric: MetricId,
    /// Its label values.
    pub labels: Vec<String>,
}

/// `obs`'s own metric: records a full ring made the writer drop, per ring.
pub const RING_DROPPED: MetricDesc = MetricDesc {
    id: MetricId(0),
    name: "syntropy_obs_ring_dropped_records_total",
    kind: MetricKind::Counter,
    unit: Unit::Count,
    labels: &[LabelKey("ring")],
    help: "Records a full shared-memory ring made its writer drop.",
    owner: "obs",
};

/// Why a registration or a binding was refused.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RegistryError {
    /// Another metric has this number.
    DuplicateId(MetricId),
    /// Another metric has this name.
    DuplicateName(&'static str),
    /// No metric has this number.
    Unknown(MetricId),
    /// The metric is of another kind than the binding asked for.
    WrongKind {
        /// The metric.
        metric: MetricId,
        /// Its kind.
        kind: MetricKind,
        /// The kind asked for.
        asked: MetricKind,
    },
    /// The binding gave another number of label values than the metric has labels.
    LabelCount {
        /// The metric.
        metric: MetricId,
        /// Its labels.
        expected: usize,
        /// The values given.
        given: usize,
    },
    /// A histogram's unit has no bucket range: only times and sizes do.
    NoScale(MetricId),
    /// More series than a `u32` counts.
    TooManySeries,
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegistryError::DuplicateId(id) => write!(f, "metric {} is registered twice", id.0),
            RegistryError::DuplicateName(name) => write!(f, "metric {name} is registered twice"),
            RegistryError::Unknown(id) => write!(f, "metric {} is not registered", id.0),
            RegistryError::WrongKind {
                metric,
                kind,
                asked,
            } => write!(f, "metric {} is a {kind:?}, not a {asked:?}", metric.0),
            RegistryError::LabelCount {
                metric,
                expected,
                given,
            } => write!(
                f,
                "metric {} has {expected} labels, and {given} values were given",
                metric.0
            ),
            RegistryError::NoScale(id) => {
                write!(f, "histogram {} is neither of times nor of sizes", id.0)
            }
            RegistryError::TooManySeries => f.write_str("too many series"),
        }
    }
}

impl std::error::Error for RegistryError {}

/// The metrics and series of one runner, while they are being registered (`Warming`).
#[derive(Debug)]
pub struct Registry {
    descs: BTreeMap<MetricId, &'static MetricDesc>,
    names: BTreeMap<&'static str, MetricId>,
    bound: BTreeMap<(MetricId, Vec<String>), SeriesId>,
    layout: Layout,
}

impl Registry {
    /// An empty registry, with `obs`'s own [`RING_DROPPED`] registered and bound for the
    /// `telemetry` and `step` rings.
    #[must_use]
    pub fn new() -> Registry {
        let mut registry = Registry {
            descs: BTreeMap::new(),
            names: BTreeMap::new(),
            bound: BTreeMap::new(),
            layout: Layout::default(),
        };
        // An empty registry has neither the number, the name nor a series yet.
        let bind = |registry: &mut Registry, ring: &str| {
            registry
                .counter(RING_DROPPED.id, &[ring])
                .unwrap_or(CounterIx(0))
        };
        let _ = registry.register(&RING_DROPPED);
        registry.layout.telemetry_drops = bind(&mut registry, "telemetry");
        registry.layout.step_drops = bind(&mut registry, "step");
        registry
    }

    /// Registers `desc`.
    ///
    /// # Errors
    ///
    /// [`RegistryError::DuplicateId`] or [`RegistryError::DuplicateName`] when another metric has
    /// its number or its name.
    pub fn register(&mut self, desc: &'static MetricDesc) -> Result<(), RegistryError> {
        if self.descs.contains_key(&desc.id) {
            return Err(RegistryError::DuplicateId(desc.id));
        }
        if self.names.contains_key(desc.name) {
            return Err(RegistryError::DuplicateName(desc.name));
        }
        self.descs.insert(desc.id, desc);
        self.names.insert(desc.name, desc.id);
        Ok(())
    }

    /// The handle of counter `metric` with label values `labels`, one per label in order. The
    /// same values give the same handle.
    ///
    /// # Errors
    ///
    /// [`RegistryError::Unknown`], [`RegistryError::WrongKind`] or [`RegistryError::LabelCount`].
    pub fn counter(
        &mut self,
        metric: MetricId,
        labels: &[&str],
    ) -> Result<CounterIx, RegistryError> {
        let (_, slot) = self.bind(metric, labels, MetricKind::Counter)?;
        Ok(CounterIx(slot))
    }

    /// The handle of histogram `metric` with label values `labels`.
    ///
    /// # Errors
    ///
    /// As for [`Registry::counter`], and [`RegistryError::NoScale`] for a histogram of a unit
    /// with no bucket range.
    pub fn histogram(
        &mut self,
        metric: MetricId,
        labels: &[&str],
    ) -> Result<HistIx, RegistryError> {
        let (_, slot) = self.bind(metric, labels, MetricKind::Histogram)?;
        Ok(HistIx(slot))
    }

    /// The handle of gauge `metric` with label values `labels`. Only one thread may set a gauge
    /// series ([`LocalMetrics::set`](super::LocalMetrics::set)).
    ///
    /// # Errors
    ///
    /// As for [`Registry::counter`].
    pub fn gauge(&mut self, metric: MetricId, labels: &[&str]) -> Result<GaugeIx, RegistryError> {
        let (_, slot) = self.bind(metric, labels, MetricKind::Gauge)?;
        Ok(GaugeIx(slot))
    }

    /// Fixes the series. Every thread's cells and the drain are built from the result.
    #[must_use]
    pub fn freeze(self) -> Arc<Layout> {
        Arc::new(self.layout)
    }

    /// Binds a series of `kind`; its series ID and its slot among the series of that kind.
    fn bind(
        &mut self,
        metric: MetricId,
        labels: &[&str],
        kind: MetricKind,
    ) -> Result<(SeriesId, u32), RegistryError> {
        let desc = *self
            .descs
            .get(&metric)
            .ok_or(RegistryError::Unknown(metric))?;
        if desc.kind != kind {
            return Err(RegistryError::WrongKind {
                metric,
                kind: desc.kind,
                asked: kind,
            });
        }
        if labels.len() != desc.labels.len() {
            return Err(RegistryError::LabelCount {
                metric,
                expected: desc.labels.len(),
                given: labels.len(),
            });
        }
        let scale = match kind {
            MetricKind::Histogram => {
                Some(HistScale::of(desc.unit).ok_or(RegistryError::NoScale(metric))?)
            }
            _ => None,
        };
        let key = (
            metric,
            labels
                .iter()
                .map(|value| (*value).to_owned())
                .collect::<Vec<_>>(),
        );
        if let Some(&id) = self.bound.get(&key) {
            return Ok((id, self.layout.slot_of(id)));
        }
        let layout = &mut self.layout;
        let id =
            SeriesId(u32::try_from(layout.series.len()).map_err(|_| RegistryError::TooManySeries)?);
        let slot = match scale {
            Some(scale) => push(&mut layout.histograms, (id, scale))?,
            None if kind == MetricKind::Counter => push(&mut layout.counters, id)?,
            None => push(&mut layout.gauges, id)?,
        };
        layout.series.push(SeriesInfo {
            id,
            metric,
            labels: key.1.clone(),
        });
        layout.slots.push(slot);
        self.bound.insert(key, id);
        Ok((id, slot))
    }
}

impl Default for Registry {
    fn default() -> Registry {
        Registry::new()
    }
}

/// Appends `item`; its index.
fn push<T>(list: &mut Vec<T>, item: T) -> Result<u32, RegistryError> {
    let at = u32::try_from(list.len()).map_err(|_| RegistryError::TooManySeries)?;
    list.push(item);
    Ok(at)
}

/// The series of one runner, fixed at the end of `Warming`.
#[derive(Debug, Default)]
pub struct Layout {
    series: Vec<SeriesInfo>,
    /// Each series' slot among the series of its kind.
    slots: Vec<u32>,
    counters: Vec<SeriesId>,
    histograms: Vec<(SeriesId, HistScale)>,
    gauges: Vec<SeriesId>,
    telemetry_drops: CounterIx,
    step_drops: CounterIx,
}

impl Layout {
    /// Every series, by `SeriesId`.
    #[must_use]
    pub fn series(&self) -> &[SeriesInfo] {
        &self.series
    }

    /// The counter series, by slot.
    #[must_use]
    pub fn counters(&self) -> &[SeriesId] {
        &self.counters
    }

    /// The histogram series and their scales, by slot.
    #[must_use]
    pub fn histograms(&self) -> &[(SeriesId, HistScale)] {
        &self.histograms
    }

    /// The gauge series, by slot.
    #[must_use]
    pub fn gauges(&self) -> &[SeriesId] {
        &self.gauges
    }

    /// The series counter `counter` records.
    #[must_use]
    pub fn counter_series(&self, counter: CounterIx) -> SeriesId {
        self.counters[counter.0 as usize]
    }

    /// The series histogram `hist` records.
    #[must_use]
    pub fn histogram_series(&self, hist: HistIx) -> SeriesId {
        self.histograms[hist.0 as usize].0
    }

    /// The series gauge `gauge` records.
    #[must_use]
    pub fn gauge_series(&self, gauge: GaugeIx) -> SeriesId {
        self.gauges[gauge.0 as usize]
    }

    /// The counter of records the `telemetry` ring's writer, the drain, dropped.
    #[must_use]
    pub fn telemetry_drops(&self) -> CounterIx {
        self.telemetry_drops
    }

    /// The counter of records a dispatcher's step ring dropped; each dispatcher counts its own.
    #[must_use]
    pub fn step_drops(&self) -> CounterIx {
        self.step_drops
    }

    fn slot_of(&self, id: SeriesId) -> u32 {
        self.slots[id.0 as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STEP: MetricDesc = MetricDesc {
        id: MetricId(10),
        name: "syntropy_exec_step_seconds",
        kind: MetricKind::Histogram,
        unit: Unit::Nanos,
        labels: &[LabelKey::STAGE],
        help: "Stage step time.",
        owner: "exec",
    };

    const TOKENS: MetricDesc = MetricDesc {
        id: MetricId(11),
        name: "syntropy_exec_tokens_total",
        kind: MetricKind::Counter,
        unit: Unit::Count,
        labels: &[LabelKey::STAGE, LabelKey::DEVICE],
        help: "Tokens processed.",
        owner: "exec",
    };

    #[test]
    fn binding_resolves_labels_once_and_reuses_the_series() {
        let mut registry = Registry::new();
        registry.register(&STEP).unwrap();
        registry.register(&TOKENS).unwrap();
        let a = registry.counter(TOKENS.id, &["0", "cuda0"]).unwrap();
        let b = registry.counter(TOKENS.id, &["1", "cuda0"]).unwrap();
        assert_ne!(a, b);
        assert_eq!(registry.counter(TOKENS.id, &["0", "cuda0"]).unwrap(), a);
        let h = registry.histogram(STEP.id, &["0"]).unwrap();
        let layout = registry.freeze();
        // obs's two drop series, then the three bound here.
        assert_eq!(layout.series().len(), 5);
        assert_eq!(layout.counters().len(), 4);
        assert_eq!(layout.histograms(), &[(SeriesId(4), HistScale::Nanos)]);
        assert_eq!(h, HistIx(0));
        assert_eq!(layout.series()[2].labels, ["0", "cuda0"]);
        assert_eq!(layout.series()[0].labels, ["telemetry"]);
    }

    #[test]
    fn bad_registrations_and_bindings_are_refused() {
        let mut registry = Registry::new();
        registry.register(&TOKENS).unwrap();
        assert_eq!(
            registry.register(&TOKENS),
            Err(RegistryError::DuplicateId(TOKENS.id))
        );
        static RENAMED: MetricDesc = MetricDesc {
            id: MetricId(99),
            ..TOKENS
        };
        assert_eq!(
            registry.register(&RENAMED),
            Err(RegistryError::DuplicateName(TOKENS.name))
        );
        assert_eq!(
            registry.counter(MetricId(5), &[]),
            Err(RegistryError::Unknown(MetricId(5)))
        );
        assert!(matches!(
            registry.gauge(TOKENS.id, &["0", "x"]),
            Err(RegistryError::WrongKind { .. })
        ));
        assert!(matches!(
            registry.counter(TOKENS.id, &["0"]),
            Err(RegistryError::LabelCount {
                expected: 2,
                given: 1,
                ..
            })
        ));
        static COUNTS: MetricDesc = MetricDesc {
            id: MetricId(12),
            name: "syntropy_exec_window_tokens",
            kind: MetricKind::Histogram,
            unit: Unit::Count,
            labels: &[],
            help: "",
            owner: "exec",
        };
        registry.register(&COUNTS).unwrap();
        assert_eq!(
            registry.histogram(COUNTS.id, &[]),
            Err(RegistryError::NoScale(COUNTS.id))
        );
    }
}
