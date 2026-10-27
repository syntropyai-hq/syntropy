// SPDX-License-Identifier: MIT OR Apache-2.0

//! The statistics ring from a dispatcher to the housekeeping thread (RFC-0001 §8.3, "hot
//! threads → housekeeping"; RFC-0023 §5.1).
//!
//! One ring per dispatcher. It is lossy: a record that finds the ring full is dropped and
//! counted, and the step never waits for housekeeping (RFC-0001 §8.3, §8.4). Its first record
//! kind is [`RouteRecord`]: one pass's `RouteSlot` list for one MoE layer, copied from the
//! doorbell after the pass. The step loop's other statistics deltas join it as further
//! [`StatRecord`] kinds (PLC-E7-T10), so a dispatcher has one statistics ring, not several.
//!
//! **No allocation per step.** The ring is [`syntropy_spsc`]'s, sized when it is made, at
//! `Warming`. A routing record's list lives in a buffer with room for `MAX_UNION` entries, and
//! the ring owns `capacity` such buffers from the start. Housekeeping hands each buffer back
//! over a second SPSC ring once it has read the record, and the producer refills it, so a push
//! copies at most `MAX_UNION` entries into memory that already exists. When every buffer is in
//! flight, the ring is full and the record is dropped.
//!
//! **Order.** Records arrive in the order the dispatcher pushed them, so the routing records of
//! one dispatcher arrive in step order.

#![forbid(unsafe_code)]

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use syntropy_core::LayerIx;
use syntropy_kernels::moe::RouteSlot;
use syntropy_spsc::{CapacityError, Consumer, PopError, Producer, PushError, ring};

/// One pass's routing for one MoE layer, as the router wrote it into the layer's doorbell
/// (RFC-0023 §2, §5.1; RFC-0060 §12.4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouteRecord {
    /// The step that ran the pass.
    pub step: u64,
    /// The doorbell's `seq`: the low 32 bits of the step number the router wrote, a payload
    /// check (RFC-0023 §6). A well-formed record has `seq == step as u32`.
    pub seq: u32,
    /// The MoE layer.
    pub layer: LayerIx,
    /// The pass's token count: its active rows.
    pub tokens: u32,
    /// The doorbell's `route` entries: each distinct expert of the pass once, with the number
    /// of the pass's tokens that selected it.
    pub route: Vec<RouteSlot>,
}

/// A record in the statistics ring. Further kinds come with the step loop's statistics
/// (PLC-E7-T10).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StatRecord {
    /// One pass's routing for one MoE layer.
    Route(RouteRecord),
}

/// Why a record was dropped. Every drop is also counted in [`StatsProducer::dropped`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatDrop {
    /// The ring is full: housekeeping has not caught up.
    Full,
    /// Housekeeping's end is gone.
    Closed,
    /// More route entries than the epoch's `MAX_UNION`: not a well-formed doorbell.
    TooManyRoutes {
        /// The entries offered.
        count: usize,
        /// `MAX_UNION`.
        max: u16,
    },
}

impl fmt::Display for StatDrop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Full => f.write_str("the statistics ring is full"),
            Self::Closed => f.write_str("the statistics ring's consumer is gone"),
            Self::TooManyRoutes { count, max } => {
                write!(f, "{count} route entries; MAX_UNION is {max}")
            }
        }
    }
}

impl std::error::Error for StatDrop {}

/// Makes a statistics ring of `capacity` records, a power of two, whose routing records hold up
/// to `max_union` entries each (the epoch's `MAX_UNION`, [`super::sizing`]). Every buffer is
/// allocated here.
///
/// # Errors
///
/// [`CapacityError`] when `capacity` is not a power of two.
pub fn stats_ring(
    capacity: usize,
    max_union: u16,
) -> Result<(StatsProducer, StatsConsumer), CapacityError> {
    let (records_tx, records_rx) = ring::<StatRecord>(capacity)?;
    let (mut free_tx, free_rx) = ring::<Vec<RouteSlot>>(capacity)?;
    for _ in 0..capacity {
        free_tx
            .push(Vec::with_capacity(usize::from(max_union)))
            .expect("the free ring holds `capacity` buffers");
    }
    let dropped = Arc::new(AtomicU64::new(0));
    Ok((
        StatsProducer {
            records: records_tx,
            free: free_rx,
            spare: None,
            max_union,
            dropped: Arc::clone(&dropped),
        },
        StatsConsumer {
            records: records_rx,
            free: free_tx,
            dropped,
        },
    ))
}

/// The dispatcher's end of a statistics ring.
pub struct StatsProducer {
    records: Producer<StatRecord>,
    free: Consumer<Vec<RouteSlot>>,
    /// A buffer taken back from a record the full ring refused, used before the free ring.
    spare: Option<Vec<RouteSlot>>,
    max_union: u16,
    dropped: Arc<AtomicU64>,
}

impl StatsProducer {
    /// Pushes one pass's routing for one MoE layer, copying `route` into a buffer the ring
    /// owns. Never blocks and never allocates.
    ///
    /// # Errors
    ///
    /// [`StatDrop`] when the record was dropped; the drop is counted.
    pub fn push_route(
        &mut self,
        step: u64,
        seq: u32,
        layer: LayerIx,
        tokens: u32,
        route: &[RouteSlot],
    ) -> Result<(), StatDrop> {
        if route.len() > usize::from(self.max_union) {
            return Err(self.drop(StatDrop::TooManyRoutes {
                count: route.len(),
                max: self.max_union,
            }));
        }
        let mut buf = match self.spare.take() {
            Some(buf) => buf,
            None => match self.free.pop() {
                Ok(buf) => buf,
                // Every buffer is in a record housekeeping has not read yet.
                Err(PopError::Empty) => return Err(self.drop(StatDrop::Full)),
                Err(PopError::Closed) => return Err(self.drop(StatDrop::Closed)),
            },
        };
        buf.clear();
        buf.extend_from_slice(route);
        let record = StatRecord::Route(RouteRecord {
            step,
            seq,
            layer,
            tokens,
            route: buf,
        });
        match self.records.push(record) {
            Ok(()) => Ok(()),
            Err(PushError::Full(StatRecord::Route(r))) => {
                self.spare = Some(r.route);
                Err(self.drop(StatDrop::Full))
            }
            Err(PushError::Closed(_)) => Err(self.drop(StatDrop::Closed)),
        }
    }

    /// Records dropped since the ring was made.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    fn drop(&self, why: StatDrop) -> StatDrop {
        self.dropped.fetch_add(1, Ordering::Relaxed);
        why
    }
}

impl fmt::Debug for StatsProducer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StatsProducer")
            .field("max_union", &self.max_union)
            .field("dropped", &self.dropped())
            .finish_non_exhaustive()
    }
}

/// Housekeeping's end of a statistics ring.
pub struct StatsConsumer {
    records: Consumer<StatRecord>,
    free: Producer<Vec<RouteSlot>>,
    dropped: Arc<AtomicU64>,
}

impl StatsConsumer {
    /// Hands every record in the ring to `f`, oldest first, and gives each routing record's
    /// buffer back to the producer. Returns how many records it read.
    pub fn drain(&mut self, mut f: impl FnMut(&StatRecord)) -> usize {
        let mut n = 0;
        while let Ok(record) = self.records.pop() {
            f(&record);
            n += 1;
            match record {
                // The free ring holds every buffer, so it is never full. A closed one means
                // the producer is gone, and the buffer is freed here.
                StatRecord::Route(r) => {
                    let _ = self.free.push(r.route);
                }
            }
        }
        n
    }

    /// Records the producer has dropped since the ring was made.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

impl fmt::Debug for StatsConsumer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StatsConsumer")
            .field("dropped", &self.dropped())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(expert: u16, tokens: u16) -> RouteSlot {
        RouteSlot {
            expert,
            tokens,
            cache_slot: u32::from(expert),
        }
    }

    fn routes(rx: &mut StatsConsumer) -> Vec<RouteRecord> {
        let mut out = Vec::new();
        rx.drain(|r| match r {
            StatRecord::Route(r) => out.push(r.clone()),
        });
        out
    }

    #[test]
    fn records_arrive_in_push_order() {
        let (mut tx, mut rx) = stats_ring(4, 8).unwrap();
        tx.push_route(7, 7, LayerIx(1), 1, &[slot(3, 1), slot(9, 1)])
            .unwrap();
        tx.push_route(7, 7, LayerIx(2), 1, &[slot(5, 1)]).unwrap();
        let got = routes(&mut rx);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].layer, LayerIx(1));
        assert_eq!(got[0].route, vec![slot(3, 1), slot(9, 1)]);
        assert_eq!(got[1].route, vec![slot(5, 1)]);
        assert_eq!(rx.dropped(), 0);
    }

    #[test]
    fn a_full_ring_drops_and_counts_then_recovers() {
        let (mut tx, mut rx) = stats_ring(2, 4).unwrap();
        for step in 0..2 {
            tx.push_route(step, step as u32, LayerIx(0), 1, &[slot(1, 1)])
                .unwrap();
        }
        assert_eq!(
            tx.push_route(2, 2, LayerIx(0), 1, &[slot(1, 1)]),
            Err(StatDrop::Full)
        );
        assert_eq!((tx.dropped(), rx.dropped()), (1, 1));
        assert_eq!(routes(&mut rx).len(), 2);
        // The buffers came back: the ring takes records again.
        for step in 3..5 {
            tx.push_route(step, step as u32, LayerIx(0), 1, &[slot(2, 1)])
                .unwrap();
        }
        let got = routes(&mut rx);
        assert_eq!(got.iter().map(|r| r.step).collect::<Vec<_>>(), vec![3, 4]);
        assert_eq!(rx.dropped(), 1);
    }

    #[test]
    fn a_list_longer_than_max_union_is_dropped() {
        let (mut tx, mut rx) = stats_ring(2, 1).unwrap();
        assert_eq!(
            tx.push_route(0, 0, LayerIx(0), 2, &[slot(1, 1), slot(2, 1)]),
            Err(StatDrop::TooManyRoutes { count: 2, max: 1 })
        );
        assert_eq!(rx.dropped(), 1);
        assert_eq!(routes(&mut rx).len(), 0);
    }

    #[test]
    fn closing_either_end() {
        let (mut tx, rx) = stats_ring(2, 1).unwrap();
        drop(rx);
        assert_eq!(
            tx.push_route(0, 0, LayerIx(0), 1, &[slot(1, 1)]),
            Err(StatDrop::Closed)
        );
        let (mut tx, mut rx) = stats_ring(2, 1).unwrap();
        tx.push_route(0, 0, LayerIx(0), 1, &[slot(1, 1)]).unwrap();
        drop(tx);
        // What was pushed before the producer went is still read.
        assert_eq!(routes(&mut rx).len(), 1);
        assert_eq!(routes(&mut rx).len(), 0);
    }

    #[test]
    fn capacity_must_be_a_power_of_two() {
        assert!(stats_ring(3, 1).is_err());
    }
}
