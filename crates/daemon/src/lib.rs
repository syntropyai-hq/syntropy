// SPDX-License-Identifier: MIT OR Apache-2.0

//! The node daemon: supervisor, device ledger, and the wiring of cluster, store, api and probe
//! orchestration.
//!
//! Layer 5 in RFC-0001 §4.1, designed in RFC-0001 §3. Implemented so far:
//! - the host-change ledger of RFC-0052 §11.1 ([`host_changes`]), which `setup`, `service`,
//!   `doctor --fix`, `uninstall` and `trust` share;
//! - the `syntropy doctor` framework of RFC-0053 §5.1 ([`doctor`]);
//! - `syntropy trust` on a client machine, which installs a node's local CA into the trust
//!   stores (RFC-0050 §11.1.1, [`trust`]). It lives here, beside the ledger it records in, so
//!   the CLI (layer 6) calls it.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod doctor;
pub mod host_changes;
pub mod trust;
