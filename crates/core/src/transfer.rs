// SPDX-License-Identifier: MIT OR Apache-2.0

//! What a byte transfer names: a range of a file on a volume ([`ExtentRef`], whose file is a
//! [`FileIx`]), a range of host memory ([`HostRegion`]), resident bytes in some address space
//! ([`BufferView`], in a [`ViewKind`]), and the point in a device's execution by which a
//! prefetch stops being useful ([`StepLayer`]).
//!
//! RFC-0020 §3 defines `ExtentRef`, `StepLayer` and `BufferView` among the types RFC-0020 to
//! RFC-0023 share, and RFC-0021 §2 defines `FileIx` and `HostRegion`. They live in `core`
//! because the crates that use them sit side by side or a layer apart: `syntropy-io`'s requests
//! carry `ExtentRef` and `StepLayer`, `syntropy-tier` (a layer above `io`) round-trips
//! `ExtentRef` through serde in its `ArenaPlan`, both `io` and `syntropy-kernels` (RFC-0040
//! §2.1) register `HostRegion`s, and `syntropy-kernels` (layer 1) binds the `BufferView`s that
//! `syntropy-tier` (layer 2) hands out as op weights (RFC-0040 §3.5; RFC-0001 §4.1, §4.2 rules
//! 1 and 2).
//!
//! All six are plain data. Naming memory is harmless here; the obligation sits with whoever
//! asks an engine or a provider to touch it, through the `unsafe` calls `syntropy-io` defines
//! (RFC-0021 §2).

use serde::{Deserialize, Serialize};

use crate::ids::DeviceId;
use crate::units::Bytes;

/// A file an I/O engine has opened, as its `open` returned it (RFC-0021 §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FileIx(pub u16);

/// A byte range of one file on a volume (RFC-0020 §3).
///
/// `offset` and `len` are multiples of the volume's direct-I/O alignment
/// ([`DioAlign`](crate::resource::DioAlign)), as RFC-0021 §6 requires of every request.
/// RFC-0020 §3 writes `file` and `len` as bare integers; here they are [`FileIx`] and
/// [`Bytes`], because interfaces never take bare integers for sizes (RFC-0001 §5.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExtentRef {
    /// The file, as the I/O thread opened it.
    pub file: FileIx,
    /// Offset of the first byte in the file.
    pub offset: u64,
    /// Length of the range.
    pub len: Bytes,
}

/// A point in a device's execution: the step number and the layer within it (RFC-0020 §3).
///
/// Consumers publish their progress as one `AtomicU64` per device, `(step << 16) | layer`
/// ([`StepLayer::encode`]). A `StepLayer` has passed when the published word is greater than
/// or equal to its encoding ([`StepLayer::has_passed`]). The derived order, step first and then
/// layer, is the order of the encodings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StepLayer {
    /// The step number. The encoding keeps its low 48 bits.
    pub step: u64,
    /// The layer within the step.
    pub layer: u16,
}

impl StepLayer {
    /// Steps at or above this value do not fit the 48 bits the encoding gives them.
    pub const STEP_LIMIT: u64 = 1 << 48;

    /// The progress-word encoding, `(step << 16) | layer` (RFC-0020 §3).
    ///
    /// `step` must be below [`StepLayer::STEP_LIMIT`]; debug builds assert it.
    #[must_use]
    pub const fn encode(self) -> u64 {
        debug_assert!(self.step < Self::STEP_LIMIT);
        (self.step << 16) | self.layer as u64
    }

    /// Whether a consumer whose published progress word is `progress` has reached this point.
    #[must_use]
    pub const fn has_passed(self, progress: u64) -> bool {
        progress >= self.encode()
    }
}

/// A range of host memory to register with an I/O engine as a buffer (RFC-0021 §2, §4.1), or
/// with a kernel provider (RFC-0040 §2.1).
///
/// RFC-0021 §2 and RFC-0040 §2.1 name this type without fields; this is the smallest shape
/// registration needs. It is plain data: the caller of the `unsafe` registration call vouches
/// for the memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HostRegion {
    /// Address of the first byte, aligned per RFC-0021 §6.
    pub addr: u64,
    /// Length of the range.
    pub len: Bytes,
}

/// What the hot path and the I/O engine use to address resident bytes (RFC-0020 §3). The tier
/// manager's `view` returns it, and op launches bind it as weights (RFC-0040 §3.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BufferView {
    /// Which address space `addr` is in.
    pub kind: ViewKind,
    /// The start address.
    pub addr: u64,
    /// The length.
    pub len: Bytes,
}

/// The address space of a [`BufferView`] (RFC-0020 §3).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ViewKind {
    /// Host memory.
    Host,
    /// One device's memory.
    Device(DeviceId),
    /// Unified memory, addressable by host and GPU alike.
    Unified,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_layer_encoding() {
        let at = StepLayer { step: 3, layer: 7 };
        assert_eq!(at.encode(), (3 << 16) | 7);
        assert!(!at.has_passed(0));
        assert!(!at.has_passed(StepLayer { step: 3, layer: 6 }.encode()));
        assert!(at.has_passed(at.encode()));
        assert!(at.has_passed(StepLayer { step: 3, layer: 8 }.encode()));
        assert!(at.has_passed(StepLayer { step: 4, layer: 0 }.encode()));

        let last = StepLayer {
            step: StepLayer::STEP_LIMIT - 1,
            layer: u16::MAX,
        };
        assert_eq!(last.encode(), u64::MAX);
    }

    #[test]
    fn step_layer_order_matches_encoding_order() {
        let points = [
            StepLayer { step: 0, layer: 0 },
            StepLayer {
                step: 0,
                layer: u16::MAX,
            },
            StepLayer { step: 1, layer: 0 },
            StepLayer { step: 1, layer: 1 },
            StepLayer {
                step: 1 << 40,
                layer: 2,
            },
            StepLayer {
                step: StepLayer::STEP_LIMIT - 1,
                layer: 0,
            },
        ];
        for a in points {
            for b in points {
                assert_eq!(a.cmp(&b), a.encode().cmp(&b.encode()), "{a:?} vs {b:?}");
            }
        }
    }
}
