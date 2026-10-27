// SPDX-License-Identifier: MIT OR Apache-2.0

//! The node daemon: supervisor, device ledger, and the wiring of cluster, store, api and probe
//! orchestration.
//!
//! Layer 5 in RFC-0001 §4.1, designed in RFC-0001 §3. Implemented so far: the host-change ledger
//! of RFC-0052 §11.1 ([`host_changes`]), which `setup`, `service`, `doctor --fix` and `uninstall`
//! share, the `syntropy doctor` framework of RFC-0053 §5.1 ([`doctor`]), and the `ctl` runtime
//! with the API server on it (RFC-0001 §8.1, RFC-0050 §1.1; [`ctl`]), logging: the `tracing`
//! subscriber, the formats, repeat suppression, the content gate and the sinks of RFC-0053 §4
//! ([`logging`]), the forwarding of a runner's `StateChanged` as `ReportRunner` up to its seam
//! (RFC-0001 §3.5; [`report`]), and the runner supervisor: its restart policy as a protocol core
//! ([`supervisor`]) and the shell that feeds it a runner's channel and counters
//! ([`runner_link`]; RFC-0001 §3.4, RFC-0033 §8.2).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod ctl;
pub mod doctor;
pub mod host_changes;
pub mod logging;
pub mod report;
pub mod runner_link;
pub mod supervisor;
