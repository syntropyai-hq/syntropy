// SPDX-License-Identifier: MIT OR Apache-2.0

//! The cluster event log: entries and their hash chain, the base events, `ClusterState` and
//! the pure `apply` (RFC-0032 §6.1–§6.3, RFC-0001 §5.10, ADR-0009 item 1). Owner: RFC-0032.
//!
//! - **Entries** ([`LogEntry`]) carry the event twice: decoded, and as the CBOR bytes the
//!   coordinator encoded once with the control envelope ([`syntropy_net::ctl::envelope`]).
//!   The bytes are persisted, relayed and hashed verbatim, never re-encoded: a node of an older
//!   minor drops fields it does not know when it decodes, so only the received bytes give every
//!   node the same chain.
//! - **The chain** ([`chain`]) links each entry to the one before it, from a [`genesis`] value
//!   over the cluster key's public half. Two nodes agree on the log up to an index exactly when
//!   their chain values there match. It detects divergence; it is not a security mechanism,
//!   which `validate()` and keyholder signatures are (DIST-E9-T12, DIST-E9-T16).
//! - **The state** ([`ClusterState`]) is the fold of the log by [`apply`], which is pure and
//!   total, and [`state_hash`] is the same for the same log on every platform.
//!
//! This module has the base events and the shell of the state they fold into. `decide()` and
//! `ClusterCommand` are DIST-E9-T02's, `validate()` DIST-E9-T12's, and the rest of RFC-0032
//! §6.5's catalogue joins through DIST-E9-T15's owner hooks.

mod entry;
mod event;
pub mod payload;
mod state;

pub use entry::{CommandId, EntryHeader, LogEntry, ReceiveError, chain, genesis};
pub use event::ClusterEvent;
pub use state::{
    AllocState, ClusterConfig, ClusterState, ClusterValue, CommittedPlan, CoordinatorRecord,
    GraphState, InstanceRecord, MemberRecord, MemberStatus, ModelRecord, Routable, SessionRecord,
    apply, state_hash,
};
