// SPDX-License-Identifier: MIT OR Apache-2.0

//! The node's local TLS certificate authority (RFC-0050 §11.1.1; ADR-0505 item 6; PROD-E5-T03).
//!
//! TLS on a LAN works only if clients can trust the node's certificate without a public CA, so
//! each node runs a small CA of its own.
//! - **One CA per node, never shared.** An ECDSA P-256 key in the state directory's `tls`
//!   directory, generated on this node and never sent anywhere: no code here talks to another
//!   node or to the coordinator.
//! - **Bound to the node.** A node-key signature over the CA certificate's exact bytes
//!   ([`CaBinding`], RFC-0034 R5) lets a client that knows the node key check that the
//!   certificate, name constraints and all, is the one the node minted. The node holds its own
//!   stored CA to the same check, and to the constraints it mints, when it loads it.
//! - **Constrained.** Critical X.509 name constraints limit the CA to the node's DNS names and to
//!   the private subnets it serves ([`CaScope`]), so a leaked key cannot impersonate public
//!   sites, nor a private host outside those subnets. Inside them it can: the CA covers each
//!   served /24 or /64 whole. A changed scope replaces the CA and records it for `doctor`
//!   SEC-006.
//! - **Leaves.** 30-day certificates for the node's current names and addresses
//!   ([`LocalCa::issue_leaf`]), served through [`LeafResolver`], which swaps in a new leaf
//!   without dropping a connection.
//!
//! The listener that serves these leaves, and the `GET /v1/syntropy/tls/ca` route that returns
//! [`LocalCa::certificate_pem`], are PROD-E5-T06's. The OS keystore for the CA key is
//! PROD-E5-T10's (M3), and so is the fingerprint's six-word form. The state directory and the
//! node key come from the caller: DIST-E2-T01 creates them.

mod ca;
mod files;
mod leaf;
mod scope;
#[cfg(test)]
mod tests;

pub use ca::{
    CA_FILE, CA_VALIDITY, CaBinding, CaOpen, CaReplacement, LocalCa, REPLACED_FILE, ReplaceReason,
    TlsError,
};
pub use leaf::{LEAF_VALIDITY, Leaf, LeafNames, LeafResolver, REISSUE_BEFORE};
pub use scope::{CaScope, LeftOut, ScopeError, Subnet, is_local};
