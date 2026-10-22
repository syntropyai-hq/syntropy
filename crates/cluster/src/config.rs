// SPDX-License-Identifier: MIT OR Apache-2.0

//! The configuration sections of RFC-0030 §9, RFC-0032 §9, RFC-0033 §12 (with RFC-0033 §8.2's
//! `runner.*`) and RFC-0034 §14 (RFC-0054 §5.1). RFC-0031 §11's sections are `net`'s
//! ([`syntropy_net::config`]).
//!
//! The names, scopes and defaults are the RFCs'. The RFCs give no reload classes and, for
//! several keys, no value syntax, so these sections choose them by the rules
//! [`syntropy_net::config`] states: `Node` for a value read once when the daemon builds a
//! long-lived component, `Instance` for a value that changes an instance's plan or runtime
//! layout, `Hot` otherwise; [`Duration`](syntropy_config::Duration)s and
//! [`ByteSize`](syntropy_config::ByteSize)s for times and sizes; `computed` defaults for the
//! RFCs' conditional ones; and small tables for compound values ("3 failures in 10 min").
//! RFC-0054 §5.1's note of 2026-10-22 records these choices.
//!
//! A key inside another RFC's prefix is declared by a shared section: RFC-0030's `net.*` keys,
//! RFC-0033's `replan.degraded_min_gain` (RFC-0011 owns `replan.*`), and RFC-0034's `rdma.encrypt_untrusted`,
//! `invite.require_confirm` and `models.*` keys. [`SECTIONS`] lists this crate's sections, and
//! [`syntropy_net::config::SECTIONS`] lists `net`'s, for the binary's registry.

pub mod log;
pub mod membership;
pub mod recovery;
pub mod security;

use syntropy_config::{ConfigSection, SectionSchema};
pub use syntropy_net::config::FEATURE_LEVEL_V1;

/// Every section this crate declares, in RFC order.
pub const SECTIONS: &[&SectionSchema] = &[
    // RFC-0030 §9.
    &membership::ClusterSection::SCHEMA,
    &membership::DiscoverySection::SCHEMA,
    &membership::NetMembershipSection::SCHEMA,
    &membership::SwimSection::SCHEMA,
    &membership::MembershipSection::SCHEMA,
    &membership::PowerSection::SCHEMA,
    &membership::InviteSection::SCHEMA,
    &membership::PairingSection::SCHEMA,
    // RFC-0032 §9.
    &log::CoordinatorSection::SCHEMA,
    &log::LogSection::SCHEMA,
    &log::OrphanSection::SCHEMA,
    &log::RingSection::SCHEMA,
    &log::TpSection::SCHEMA,
    &log::ActionsSection::SCHEMA,
    // RFC-0033 §12 and §8.2.
    &recovery::RecoverySection::SCHEMA,
    &recovery::ReplanRecoverySection::SCHEMA,
    &recovery::StragglerSection::SCHEMA,
    &recovery::RunnerSection::SCHEMA,
    &recovery::StandbySection::SCHEMA,
    // RFC-0034 §14.
    &security::SecuritySection::SCHEMA,
    &security::RdmaSecuritySection::SCHEMA,
    &security::InviteSecuritySection::SCHEMA,
    &security::IntegritySection::SCHEMA,
    &security::ModelsSecuritySection::SCHEMA,
];
