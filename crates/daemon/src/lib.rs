// SPDX-License-Identifier: MIT OR Apache-2.0

//! The node daemon: supervisor, device ledger, and the wiring of cluster, store, api and probe
//! orchestration.
//!
//! Layer 5 in RFC-0001 §4.1, designed in RFC-0001 §3. Implemented so far: the host-change ledger
//! of RFC-0052 §11.1 ([`host_changes`]), which `setup`, `service`, `doctor --fix` and `uninstall`
//! share.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod host_changes;
