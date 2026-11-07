// SPDX-License-Identifier: MIT OR Apache-2.0

//! Graph-capture rules every provider shares (RFC-0040 §4.4; RFC-0012 §7; COMP-E4-T05).
//!
//! The provider records and replays graphs ([`KernelProvider::begin_capture`],
//! [`end_capture`](crate::KernelProvider::end_capture), [`replay`](crate::KernelProvider::replay)). What is
//! decided the same way on every backend lives here, as plain data and pure functions:
//!
//! - [`segments`]: a step whose ops include one with `OpCaps::capture_safe = false` becomes a
//!   list of captured segments and direct launches, the capture ending before that op and a new
//!   one starting after it (RFC-0040 §4.4 rule 5). Each part is one submission.
//! - [`GraphChange::invalidates`]: which changes drop a captured graph (RFC-0040 §4.4,
//!   "Invalidation"; RFC-0012 §7.2).
//! - [`capture_degrade`]: `CaptureFailed`, `(Kernels, 7)`, means the segment runs as per-op
//!   launches and the runner reports [`DegradeFlags::CAPTURE_OFF`] (RFC-0040 §13; RFC-0012
//!   §7.3). The `graph_capture` kill switch raises the same code, so both take one path.
//! - [`GraphSet`]: the instances of a stage keyed by (bucket, parity, variant), with a newly
//!   instantiated executable swapped in only between steps (RFC-0012 §7.1).
//! - [`instances`]: RFC-0040 §4.4's count of instances per stage.
//!
//! [`KernelProvider::begin_capture`]: crate::KernelProvider::begin_capture

use std::collections::BTreeMap;
use std::ops::Range;

use syntropy_core::ErrorClass;
use syntropy_core::runner::DegradeFlags;

use crate::error::{Error, codes};
use crate::op::ShapeClass;
use crate::step::Parity;

/// One part of a step: a captured segment of consecutive ops, or one op launched directly
/// because it cannot be captured (RFC-0040 §4.4 rule 5).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Segment {
    /// Ops `range` of the step, captured together and replayed as one graph.
    Captured(Range<usize>),
    /// Op `index`, launched directly between two segments.
    Direct(usize),
}

/// The parts of a step whose ops have the `capture_safe` flags given, in order: each run of
/// capture-safe ops is one [`Segment::Captured`], and each other op is a [`Segment::Direct`].
/// An empty step has no parts. The step costs one submission per part, so a step without a
/// capture-unsafe op is one submission.
#[must_use]
pub fn segments(capture_safe: impl IntoIterator<Item = bool>) -> Vec<Segment> {
    let mut out = Vec::new();
    let mut start = None;
    let mut n = 0;
    for (i, safe) in capture_safe.into_iter().enumerate() {
        n = i + 1;
        match (safe, start) {
            (true, None) => start = Some(i),
            (true, Some(_)) => {}
            (false, s) => {
                if let Some(s) = s {
                    out.push(Segment::Captured(s..i));
                }
                start = None;
                out.push(Segment::Direct(i));
            }
        }
    }
    if let Some(s) = start {
        out.push(Segment::Captured(s..n));
    }
    out
}

/// A change to the plan, the device or the data a captured graph reads (RFC-0040 §4.4,
/// "Invalidation"; RFC-0012 §7.2).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GraphChange {
    /// A new plan epoch: addresses may move (RFC-0001 §5.5).
    NewEpoch,
    /// The provider's health changed.
    HealthChanged,
    /// A `PlanDelta` changed the stage's bucket set.
    BucketSet,
    /// A `PlanDelta` changed a scratch size.
    ScratchSize,
    /// Expert-cache contents changed.
    ExpertCacheContents,
    /// Slot-table words changed (through the table-delta ring).
    SlotTableWords,
    /// KV page contents changed.
    KvPageContents,
    /// A new step descriptor was written.
    StepDescriptor,
    /// The expert cache shrank by unmapping tail segments; the remaining slot addresses stay
    /// valid (RFC-0001 §9.3).
    ExpertCacheTailShrink,
}

impl GraphChange {
    /// Whether the change drops the device's captured graphs. Only a new epoch, a health
    /// change, a bucket-set change or a scratch-size change does; contents that the graph reads
    /// through fixed addresses change freely (RFC-0040 §4.4).
    #[must_use]
    pub fn invalidates(self) -> bool {
        match self {
            Self::NewEpoch | Self::HealthChanged | Self::BucketSet | Self::ScratchSize => true,
            Self::ExpertCacheContents
            | Self::SlotTableWords
            | Self::KvPageContents
            | Self::StepDescriptor
            | Self::ExpertCacheTailShrink => false,
        }
    }
}

/// `CaptureFailed`, `(Kernels, 7)`, class `Unsupported` (RFC-0040 §13), with `message`.
#[must_use]
pub fn capture_failed(message: impl Into<String>) -> Error {
    Error::new(ErrorClass::Unsupported, codes::CAPTURE_FAILED, message)
}

/// What the executor does with an error from `begin_capture` or `end_capture`: for
/// `CaptureFailed`, the segment falls back to per-op launches and the runner is
/// `Ready { degraded }` with [`DegradeFlags::CAPTURE_OFF`] (RFC-0040 §13; RFC-0012 §7.3).
/// `None` for any other error: an `Unsupported` provider without capture (the CPU) runs its op
/// list as its normal path, undegraded, and other errors are not capture fallbacks.
#[must_use]
pub fn capture_degrade(e: &Error) -> Option<DegradeFlags> {
    (e.code == codes::CAPTURE_FAILED).then_some(DegradeFlags::CAPTURE_OFF)
}

/// Which instance of a (bucket, parity) a graph is (RFC-0040 §3.7, §3.8, §4.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GraphVariant {
    /// The steady-state graph.
    Plain,
    /// The profiled variant, with timestamp nodes (§3.7).
    Profiled,
    /// The parity variant, with parity taps, when `debug.parity_api` is set (§3.8).
    Parity,
}

/// The key of one captured graph instance in a stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GraphKey {
    /// The bucket.
    pub bucket: ShapeClass,
    /// The descriptor mirror the instance reads.
    pub parity: Parity,
    /// Which variant.
    pub variant: GraphVariant,
}

/// RFC-0040 §4.4's instance count for a stage: `buckets × 2` parities, plus one profiled
/// variant per profiled bucket, plus the parity variant when `debug.parity_api` is set.
#[must_use]
pub fn instances(buckets: usize, profiled: usize, parity_api: bool) -> usize {
    buckets * 2 + profiled + usize::from(parity_api)
}

/// A stage's graph instances, keyed by [`GraphKey`].
///
/// A newly instantiated graph is [`stage`](Self::stage)d from any point, and becomes live only
/// at [`between_steps`](Self::between_steps), which the device's dispatcher calls between two
/// steps; the graph it replaces is handed back for the provider to release. So a replay never
/// sees a half-swapped set, and the dispatcher stays the device's one submitter (RFC-0012
/// §7.1).
#[derive(Debug)]
pub struct GraphSet<G> {
    live: BTreeMap<GraphKey, G>,
    staged: Vec<(GraphKey, G)>,
}

impl<G> Default for GraphSet<G> {
    fn default() -> Self {
        Self {
            live: BTreeMap::new(),
            staged: Vec::new(),
        }
    }
}

impl<G> GraphSet<G> {
    /// An empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The live graph for `key`.
    #[must_use]
    pub fn get(&self, key: &GraphKey) -> Option<&G> {
        self.live.get(key)
    }

    /// Live instances.
    #[must_use]
    pub fn len(&self) -> usize {
        self.live.len()
    }

    /// Whether no instance is live.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.live.is_empty()
    }

    /// Queues `g` to become the live graph for `key` at the next
    /// [`between_steps`](Self::between_steps). A later `stage` of the same key wins.
    pub fn stage(&mut self, key: GraphKey, g: G) {
        self.staged.push((key, g));
    }

    /// Whether graphs are waiting to be swapped in.
    #[must_use]
    pub fn has_staged(&self) -> bool {
        !self.staged.is_empty()
    }

    /// Swaps the staged graphs in. Returns the graphs they replace, and any staged graph a
    /// later one of the same key superseded, for the provider to release.
    pub fn between_steps(&mut self) -> Vec<G> {
        let mut retired = Vec::new();
        for (key, g) in self.staged.drain(..) {
            if let Some(old) = self.live.insert(key, g) {
                retired.push(old);
            }
        }
        retired
    }

    /// Drops every graph, live and staged, after a change that [`GraphChange::invalidates`].
    /// Returns them for the provider to release.
    pub fn invalidate(&mut self) -> Vec<G> {
        let mut out: Vec<G> = std::mem::take(&mut self.live).into_values().collect();
        out.extend(self.staged.drain(..).map(|(_, g)| g));
        out
    }

    /// Applies `change`: drops every graph when it [`invalidates`](GraphChange::invalidates)
    /// them, and returns what was dropped.
    pub fn apply(&mut self, change: GraphChange) -> Vec<G> {
        if change.invalidates() {
            self.invalidate()
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests;
