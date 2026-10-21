// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shape signatures and row buckets (RFC-0040 §3.2; RFC-0012 §2).

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// The bucket class of a shape (RFC-0040 §3.2; the buckets of RFC-0012 §2). Each class fixes
/// the bucket's row count, [`ShapeClass::rows`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ShapeClass {
    /// One row per sequence.
    Decode,
    /// A verify window or tree of `window` rows: `γ_max + 1`, or the tree size.
    Verify {
        /// Rows in the window.
        window: u16,
    },
    /// Concurrent decode rows (RFC-0012 §2).
    Batch {
        /// Rows in the batch.
        rows: u32,
    },
    /// A prefill chunk at execute-in-place sites.
    Prefill {
        /// Tokens in the chunk.
        chunk: u32,
    },
    /// A prefill chunk with `HostPinned` groups streamed over PCIe (RFC-0012 §2 rule 4).
    PrefillStream {
        /// Tokens in the chunk.
        chunk: u32,
    },
}

impl ShapeClass {
    /// The bucket: the most token rows an op prepared for this class handles.
    #[must_use]
    pub fn rows(self) -> u32 {
        match self {
            Self::Decode => 1,
            Self::Verify { window } => u32::from(window),
            Self::Batch { rows } => rows,
            Self::Prefill { chunk } | Self::PrefillStream { chunk } => chunk,
        }
    }
}

/// The prefill ladder of RFC-0012 §2: the `Prefill` buckets below the plan's `prefill_chunk`
/// are the ones of these sizes, in tokens.
pub const PREFILL_LADDER: [u32; 4] = [64, 256, 1024, 4096];

/// A stage's bucket set (RFC-0012 §2; RFC-0040 §3.2; ADR-0105), in this order: `Decode`;
/// `Verify { window: t_max }` when speculation is on; the `Batch` buckets, ascending; the
/// prefill ladder entries strictly below `prefill_chunk`, then `Prefill { prefill_chunk }`;
/// and `PrefillStream { prefill_chunk }` when a group streams at that chunk.
///
/// `t_max` is `γ_max + 1` or the tree size. `batch_rows` are the `Batch` buckets the
/// instance's mode allows (RFC-0012 §2's table); duplicates appear once.
#[must_use]
pub fn bucket_set(
    prefill_chunk: u32,
    t_max: Option<u16>,
    batch_rows: &[u32],
    prefill_stream: bool,
) -> Vec<ShapeClass> {
    let mut set = vec![ShapeClass::Decode];
    if let Some(window) = t_max {
        set.push(ShapeClass::Verify { window });
    }
    let mut batch: SmallVec<[u32; 8]> = batch_rows.iter().copied().collect();
    batch.sort_unstable();
    batch.dedup();
    set.extend(batch.into_iter().map(|rows| ShapeClass::Batch { rows }));
    set.extend(
        PREFILL_LADDER
            .iter()
            .copied()
            .filter(|&chunk| chunk < prefill_chunk)
            .map(|chunk| ShapeClass::Prefill { chunk }),
    );
    set.push(ShapeClass::Prefill {
        chunk: prefill_chunk,
    });
    if prefill_stream {
        set.push(ShapeClass::PrefillStream {
            chunk: prefill_chunk,
        });
    }
    set
}

/// The shape an op is queried, prepared and benchmarked at (RFC-0040 §3.2).
///
/// Rows are bucketed and the active count is dynamic: an op prepared for this shape handles
/// any `rows_active ≤ rows`, read from the step descriptor (RFC-0040 §3.5).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ShapeSig {
    /// The bucket class.
    pub class: ShapeClass,
    /// The bucket: the most token rows the prepared op handles.
    pub rows: u32,
    /// Op-specific static dims, in the order [`OpKind::dims`](super::OpKind::dims) names.
    pub dims: SmallVec<[u32; 6]>,
}

impl ShapeSig {
    /// A shape in `class`'s bucket, with `rows` set to [`ShapeClass::rows`].
    #[must_use]
    pub fn new(class: ShapeClass, dims: &[u32]) -> Self {
        Self {
            class,
            rows: class.rows(),
            dims: SmallVec::from_slice(dims),
        }
    }

    /// Whether an op prepared for this shape handles a step with `rows_active` rows: every
    /// count up to the bucket's `rows`, zero included, and none above it.
    #[must_use]
    pub fn admits(&self, rows_active: u32) -> bool {
        rows_active <= self.rows
    }
}
