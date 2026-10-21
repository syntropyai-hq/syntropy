// SPDX-License-Identifier: MIT OR Apache-2.0

//! The op types the [`KernelProvider`](crate::KernelProvider) signature names, as
//! placeholders.
//!
//! RFC-0040 §3 defines them, and later tasks build them: the op registry, `OpKey`,
//! `LayoutVersion`, `ShapeSig` and `OpCaps` are COMP-E1-T02's work, and the argument types
//! are COMP-E1-T05's. Each is declared here, empty, so that the trait has its final
//! signature now. None can be built outside this crate, so no code comes to depend on the
//! placeholder; those tasks give each type its fields.

use std::marker::PhantomData;

/// Every op a provider implements: op kind, accepted quant schemes and dtypes, layouts and
/// shape constraints (RFC-0001 §5.7, RFC-0040 §3). Placeholder for COMP-E1-T02.
#[non_exhaustive]
#[derive(Debug)]
pub struct OpRegistry {}

/// An op kind with its quant schemes, weight layout and variant (RFC-0001 §5.7,
/// RFC-0040 §3.1). Placeholder for COMP-E1-T02.
#[non_exhaustive]
#[derive(Debug)]
pub struct OpKey {}

/// The version of a provider's repacked weight layout (RFC-0001 §5.6, RFC-0040 §5.4).
/// Placeholder for COMP-E1-T02.
#[non_exhaustive]
#[derive(Debug)]
pub struct LayoutVersion {}

/// The shape signature an op is queried, prepared and benchmarked at (RFC-0040 §3.2).
/// Placeholder for COMP-E1-T02.
#[non_exhaustive]
#[derive(Debug)]
pub struct ShapeSig {}

/// Whether an op runs on a device at given shapes, its workspace and the provider's cost
/// estimate (RFC-0001 §5.7, RFC-0040 §3). Placeholder for COMP-E1-T02.
#[non_exhaustive]
#[derive(Debug)]
pub struct OpCaps {}

/// An op made ready to launch on one device at one shape (RFC-0001 §5.7). RFC-0001 names
/// the type without fields. Placeholder for COMP-E1-T02.
#[non_exhaustive]
#[derive(Debug)]
pub struct PreparedOp {}

/// The buffers and constants one launch binds (RFC-0040 §3.5). Placeholder for
/// COMP-E1-T05.
#[non_exhaustive]
#[derive(Debug)]
pub struct OpArgs<'a> {
    _borrows: PhantomData<&'a ()>,
}
