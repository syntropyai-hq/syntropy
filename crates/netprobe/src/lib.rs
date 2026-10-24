// SPDX-License-Identifier: MIT OR Apache-2.0

//! The field network probe (DIST-E4-T15): a small standalone binary that alpha testers run
//! between two of their own machines, on the daemon's iroh version, to measure what RFC-0030 §5.4's
//! membership timings and RFC-0030 §6.4's off-LAN joins meet in real homes.
//!
//! One machine runs `syntropy-netprobe listen` and prints a ticket. The other runs
//! `syntropy-netprobe run <ticket>`, sends an echo request every second for 24 hours by default,
//! and writes one report file, which the tester reads and submits by hand (ADR-0510: nothing is
//! uploaded). The report records:
//!
//! - hop RTT histograms (p50, p99, p99.9) and a stall log, per medium and path;
//! - the suspicions and failures RFC-0030 §5.3's detector would have raised under §5.4's LAN and
//!   WAN timings ([`detector`]);
//! - for a test join with one machine moved to another network (`--off-lan`), whether the path was
//!   direct or relayed, the join time and the relayed RTT;
//! - the Wi-Fi generation and band of each end ([`platform`]);
//! - a trace of every request, for replay ([`report`] documents the format).
//!
//! It records no IP address, MAC address or SSID: [`report`]'s schema has no field that could hold
//! one, and [`scan`] checks the text before the report is written and when it is checked.
//!
//! `net` is the only crate that links iroh (RFC-0001 §4.2 rule 6), so this crate reaches it through
//! `syntropy_net::endpoint`. It is layer 2 in RFC-0001 §4.1.

#![forbid(unsafe_code)]

pub mod detector;
pub mod platform;
pub mod probe;
pub mod report;
pub mod scan;
pub mod stats;
pub mod ticket;
pub mod wire;
