// SPDX-License-Identifier: MIT OR Apache-2.0

//! `OpKey`, its fusion bits, and the re-export of `core`'s `LayoutVersion` (RFC-0001 §5.7;
//! RFC-0040 §3.1, §5.4).

use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use syntropy_core::QuantScheme;

use super::kind::OpKind;

/// The weight byte layout an op reads (RFC-0040 §3.1, §5.4): [`LayoutVersion::SOURCE`] for
/// the source format's own layout, or a provider's repack ID. It is `core`'s type, because a
/// `PackKey` on the command channel holds one (RFC-0001 §18.28; COMP-E10-T11); this re-export
/// keeps `syntropy_kernels::LayoutVersion` and its `SOURCE` constant where they were.
pub use syntropy_core::LayoutVersion;

bitflags! {
    /// The fusions an [`OpKey`]'s `variant` bitmask names (RFC-0040 §3.1). Bits are assigned
    /// in the order RFC-0040 §3.1 lists the fusions. A bit is part of the key, so its position
    /// never changes once released.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
    #[serde(transparent)]
    pub struct Fusions: u16 {
        /// Bias add in the epilogue.
        const FUSE_BIAS = 1 << 0;
        /// Residual add in the epilogue.
        const FUSE_RESIDUAL = 1 << 1;
        /// Activation in the epilogue.
        const FUSE_ACT = 1 << 2;
        /// RoPE applied in the op.
        const FUSE_ROPE = 1 << 3;
        /// Input normalisation in the prologue.
        const FUSE_NORM_IN = 1 << 4;
        /// Output scaling in the epilogue.
        const FUSE_SCALE_OUT = 1 << 5;
    }
}

/// One op as a provider registers, prepares and certifies it (RFC-0001 §5.7; RFC-0040 §3.1).
///
/// Tile sizes and other tuning choices are private to a provider and chosen in `prepare`;
/// they are never part of the key.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct OpKey {
    /// The op kind.
    pub op: OpKind,
    /// In order: the weight scheme, the activation dtype and the output dtype.
    pub schemes: SmallVec<[QuantScheme; 3]>,
    /// The weight byte layout the op reads.
    pub layout: LayoutVersion,
    /// A bitmask of [`Fusions`].
    pub variant: u16,
}

impl OpKey {
    /// The fusions `variant` names, or `None` when it sets a bit no fusion owns.
    #[must_use]
    pub fn fusions(&self) -> Option<Fusions> {
        Fusions::from_bits(self.variant)
    }
}
