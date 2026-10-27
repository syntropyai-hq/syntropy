// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `telemetry` ring's records (RFC-0053 §1): what the drain writes and the daemon folds.
//!
//! One slot holds one record, the postcard encoding of one [`TelemetryRecord`]. The decoder is
//! total: any input gives a record or a [`RecordError`], never a panic, and a declared sequence
//! length reserves no memory before its elements decode, since the record comes from the runner,
//! which the daemon treats as possibly hostile (RFC-0001 §3.5).
//!
//! Postcard writes a variant as its index and a struct's fields in order, with no names, so a
//! change here changes the ring's record schema, which `ipc_protocol` covers (RFC-0001 §3.5).
//! New variants go at the end: RFC-0053's `Spans` (PROD-E8-T07) and `Notice` come with the tasks
//! that define their payloads.

use std::fmt;

use postcard::de_flavors::{Flavor, Slice};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use syntropy_core::MonoTime;

use crate::metrics::{SeriesId, ThreadKind};

/// One `telemetry` ring record (RFC-0053 §1). Owner: RFC-0053.
///
/// Records name series by [`SeriesId`], a metric with its label values bound, where RFC-0053 §1's
/// listing names a `MetricId`; [`TelemetryRecord::Series`] tells the daemon what each ID stands
/// for, before any record uses it.
// `Histograms` holds up to four deltas inline, far more than the other variants. Records are
// built by the drain every 250 ms, never on a hot thread, so the size costs nothing, and the
// variants keep RFC-0053 §1's inline lists.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TelemetryRecord {
    /// Counter deltas from one thread since the last drain.
    Counters {
        /// When the drain read them.
        at: MonoTime,
        /// The thread whose cells they are.
        thread: ThreadKind,
        /// Each changed series and how much it grew.
        deltas: SmallVec<[(SeriesId, u64); 16]>,
    },
    /// Histogram deltas from one thread since the last drain.
    Histograms {
        /// When the drain read them.
        at: MonoTime,
        /// The thread whose cells they are.
        thread: ThreadKind,
        /// Each changed series and its bucket deltas.
        deltas: SmallVec<[(SeriesId, HistDelta); 4]>,
    },
    /// Gauges whose value changed since the last drain.
    Gauges {
        /// When the drain read them.
        at: MonoTime,
        /// Each changed series and its value.
        values: SmallVec<[(SeriesId, f64); 16]>,
    },
    /// What series IDs stand for: each one's metric and label values. Sent before the first
    /// record that uses an ID.
    Series {
        /// The series.
        entries: SmallVec<[SeriesEntry; 8]>,
    },
}

/// One histogram's change since the last drain. A large change may come in several deltas for
/// one series, which the daemon adds up.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HistDelta {
    /// How much the sum of recorded values grew.
    pub sum: u64,
    /// Each changed bucket and how much its count grew.
    pub buckets: SmallVec<[(u8, u64); 8]>,
}

/// What one series ID stands for.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SeriesEntry {
    /// The series.
    pub id: SeriesId,
    /// Its metric's number, which the daemon finds in the build's metric catalogue.
    pub metric: crate::metrics::MetricId,
    /// Its label values, in the order of the metric's labels.
    pub labels: Vec<String>,
}

/// Why a record did not encode or decode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecordError {
    /// A field did not serialize, or the bytes are not a record: postcard's reason.
    Malformed(postcard::Error),
    /// The record ends before the bytes do.
    Trailing {
        /// The bytes after the record.
        unread: usize,
    },
}

impl fmt::Display for RecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RecordError::Malformed(error) => {
                write!(f, "the telemetry record is malformed: {error}")
            }
            RecordError::Trailing { unread } => {
                write!(f, "the telemetry record has {unread} bytes after it")
            }
        }
    }
}

impl std::error::Error for RecordError {}

/// Encodes `record` into `out`, replacing its contents and keeping its capacity.
///
/// # Errors
///
/// [`RecordError::Malformed`] if a field does not serialize.
pub fn encode(record: &TelemetryRecord, out: &mut Vec<u8>) -> Result<(), RecordError> {
    out.clear();
    let buffer = std::mem::take(out);
    *out = postcard::to_extend(record, buffer).map_err(RecordError::Malformed)?;
    Ok(())
}

/// Decodes one record.
///
/// # Errors
///
/// [`RecordError::Malformed`] for bytes that are not a record, and [`RecordError::Trailing`] for
/// bytes left after one.
pub fn decode(bytes: &[u8]) -> Result<TelemetryRecord, RecordError> {
    let mut deserializer = postcard::Deserializer::from_flavor(NoPrealloc(Slice::new(bytes)));
    let record = TelemetryRecord::deserialize(&mut deserializer).map_err(RecordError::Malformed)?;
    let rest = deserializer.finalize().map_err(RecordError::Malformed)?;
    if !rest.is_empty() {
        return Err(RecordError::Trailing { unread: rest.len() });
    }
    Ok(record)
}

/// Postcard's slice flavor with a size hint of 0, so a declared length reserves nothing before
/// its elements decode (the same flavor as `ipc`'s command channel, which `obs` cannot name).
struct NoPrealloc<'de>(Slice<'de>);

impl<'de> Flavor<'de> for NoPrealloc<'de> {
    type Remainder = &'de [u8];
    type Source = &'de [u8];

    fn pop(&mut self) -> postcard::Result<u8> {
        self.0.pop()
    }

    fn size_hint(&self) -> Option<usize> {
        Some(0)
    }

    fn try_take_n(&mut self, ct: usize) -> postcard::Result<&'de [u8]> {
        self.0.try_take_n(ct)
    }

    fn finalize(self) -> postcard::Result<&'de [u8]> {
        self.0.finalize()
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use smallvec::smallvec;

    use super::*;
    use crate::metrics::MetricId;

    fn records() -> Vec<TelemetryRecord> {
        vec![
            TelemetryRecord::Counters {
                at: MonoTime(5),
                thread: ThreadKind::Dispatcher(1),
                deltas: smallvec![(SeriesId(3), 7), (SeriesId(9), u64::MAX)],
            },
            TelemetryRecord::Histograms {
                at: MonoTime(6),
                thread: ThreadKind::CpuPool(2),
                deltas: smallvec![(
                    SeriesId(4),
                    HistDelta {
                        sum: 12_345,
                        buckets: smallvec![(0, 1), (255, 2)],
                    }
                )],
            },
            TelemetryRecord::Gauges {
                at: MonoTime(7),
                values: smallvec![(SeriesId(5), 0.25), (SeriesId(6), f64::NAN)],
            },
            TelemetryRecord::Series {
                entries: smallvec![SeriesEntry {
                    id: SeriesId(3),
                    metric: MetricId(11),
                    labels: vec!["0".to_owned(), "cuda0".to_owned()],
                }],
            },
        ]
    }

    #[test]
    fn records_round_trip() {
        let mut out = Vec::new();
        for record in records() {
            encode(&record, &mut out).unwrap();
            let back = decode(&out).unwrap();
            // NaN is not equal to itself; compare the encodings instead.
            let mut again = Vec::new();
            encode(&back, &mut again).unwrap();
            assert_eq!(again, out, "{record:?}");
        }
    }

    #[test]
    fn bad_bytes_are_refused() {
        let mut out = Vec::new();
        encode(&records()[0], &mut out).unwrap();
        out.push(0);
        assert_eq!(decode(&out), Err(RecordError::Trailing { unread: 1 }));
        assert!(matches!(decode(&[0xff]), Err(RecordError::Malformed(_))));
        // Series with 2^32 - 1 entries declared and none present.
        assert!(matches!(
            decode(&[3, 0xff, 0xff, 0xff, 0xff, 0x0f]),
            Err(RecordError::Malformed(_))
        ));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4096))]

        /// Decoding is total over arbitrary bytes.
        #[test]
        fn decoding_is_total(bytes in prop::collection::vec(any::<u8>(), 0..512)) {
            let _ = decode(&bytes);
        }
    }
}
