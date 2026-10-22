// SPDX-License-Identifier: MIT OR Apache-2.0

//! The op registry and the op types the [`KernelProvider`](crate::KernelProvider) signature
//! names (RFC-0001 §5.7; RFC-0040 §3.1, §3.2, §5.3).
//!
//! - [`kind`] holds the closed [`OpKind`] list of RFC-0040 §3.1, each kind's dims signature
//!   and its profiling class.
//! - [`key`] holds [`OpKey`], its [`Fusions`] bits and [`LayoutVersion`].
//! - [`shape`] holds [`ShapeSig`], [`ShapeClass`] and RFC-0012 §2's bucket set.
//! - [`caps`] holds [`OpCaps`] and [`PreparedOp`], with the workspace check of RFC-0040 §5.3.
//! - [`registry`] holds [`OpRegistry`].
//!
//! The argument types are COMP-E1-T05's; [`OpArgs`] is a placeholder until then, declared
//! here so that the trait has its final signature.

use std::marker::PhantomData;

pub mod caps;
pub mod key;
pub mod kind;
pub mod registry;
pub mod shape;

pub use caps::{OpCaps, PreparedOp};
pub use key::{Fusions, LayoutVersion, OpKey};
pub use kind::{
    ActKind, AttnKind, GatedAct, IndexKind, OpKind, PoolKind, RopeKind, RouteNorm, ScanKind,
};
pub use registry::{OpRegistry, RegistryError};
pub use shape::{PREFILL_LADDER, ShapeClass, ShapeSig, bucket_set};

/// The buffers and constants one launch binds (RFC-0040 §3.5). Placeholder for
/// COMP-E1-T05.
#[non_exhaustive]
#[derive(Debug)]
pub struct OpArgs<'a> {
    _borrows: PhantomData<&'a ()>,
}

impl OpArgs<'_> {
    /// The placeholder's one value: no bindings. A provider that replays a recorded launch
    /// passes it, until COMP-E1-T05 gives launches their real arguments.
    #[must_use]
    pub fn none() -> Self {
        Self {
            _borrows: PhantomData,
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
