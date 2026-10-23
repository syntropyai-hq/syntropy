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
//! - [`args`] holds what a launch binds: [`OpArgs`], [`TensorArg`] and [`Scalar`]
//!   (RFC-0040 §3.5).

pub mod args;
pub mod caps;
pub mod key;
pub mod kind;
pub mod registry;
pub mod shape;

pub use args::{OpArgs, Scalar, TensorArg};
pub use caps::{OpCaps, PreparedOp};
pub use key::{Fusions, LayoutVersion, OpKey};
pub use kind::{
    ActKind, AttnKind, GatedAct, IndexKind, OpKind, PoolKind, RopeKind, RouteNorm, ScanKind,
};
pub use registry::{OpRegistry, RegistryError};
pub use shape::{PREFILL_LADDER, ShapeClass, ShapeSig, bucket_set};

#[cfg(test)]
pub(crate) mod tests;
