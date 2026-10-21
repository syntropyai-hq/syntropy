// SPDX-License-Identifier: MIT OR Apache-2.0

//! Identifiers, units, plan types, policy objects and the pure functions over them.
//!
//! Layer 0 in RFC-0001 §4.1, designed in RFC-0001 §5.

#![forbid(unsafe_code)]

pub mod dtype;
pub mod fidelity;
pub mod numerics;
pub mod quant;

pub use dtype::{DType, GgmlType};
pub use numerics::{AccumKind, ActQuant, Numerics};
pub use quant::{CtScheme, DTypeClass, ModelClassError, QuantScheme, ScaleGranularity};
