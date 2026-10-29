// SPDX-License-Identifier: MIT OR Apache-2.0

//! Resource graph records (RFC-0001 §5.2) and RFC-0010 §1's additions to them: topology
//! records from enumeration, capability records from measurement, graph assembly records,
//! and the ordinal rules. Owner: RFC-0010.
//!
//! Every field RFC-0010 adds has a serde default, so a record written before a field existed
//! still decodes, and a decoder ignores fields it does not know.

pub mod caps;
pub mod device;
pub mod graph;
pub mod iface;
pub mod inventory;
pub mod link;
pub mod node;
pub mod ordinal;
pub mod tier;

pub use caps::*;
pub use device::*;
pub use graph::*;
pub use iface::*;
pub use inventory::*;
pub use link::*;
pub use node::*;
pub use tier::*;
