// SPDX-License-Identifier: MIT OR Apache-2.0

//! Discovery, invites, SWIM membership, the coordinator lease, the event log, the pure `apply()`,
//! snapshots and re-placement triggers.
//!
//! Layer 3 in RFC-0001 §4.1, designed in RFC-0030. So far it holds [`pairing`]: the
//! commit-then-reveal fields of LAN pairing and the SAS both sides derive from them; and
//! [`config`]: the configuration sections of RFC-0030, RFC-0032, RFC-0033 and RFC-0034.

#![forbid(unsafe_code)]

pub mod config;
pub mod pairing;
