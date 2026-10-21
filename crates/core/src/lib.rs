// SPDX-License-Identifier: MIT OR Apache-2.0

//! Identifiers, units, plan types, policy objects and the pure functions over them.
//!
//! Layer 0 in RFC-0001 §4.1, designed in RFC-0001 §5.

#![forbid(unsafe_code)]

pub mod dtype;
pub mod error;
pub mod fidelity;
pub mod ids;
pub mod measured;
pub mod numerics;
pub mod quant;
pub mod resource;
pub mod units;

pub use dtype::{DType, GgmlType};
pub use error::ErrorClass;
pub use ids::{
    Blake3, DeviceId, Endpoint, GraphVersion, LinkId, LinkKind, NetMedium, NodeId, RdmaKind, TierId,
};
pub use measured::{DerivedFrom, MeasureSource, Measured};
pub use numerics::{AccumKind, ActQuant, Numerics};
pub use quant::{CtScheme, DTypeClass, ModelClassError, QuantScheme, ScaleGranularity};
pub use units::{Bytes, BytesPerSec, FlopsPerSec, Nanos};
