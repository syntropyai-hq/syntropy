// SPDX-License-Identifier: MIT OR Apache-2.0

//! Span and metric naming, counters and histograms for hot threads, the telemetry ring codec, fault
//! points and exporters.
//!
//! Layer 0 in RFC-0001 §4.1, designed in RFC-0053. So far it holds the `CostTerm` key space
//! (RFC-0053 §2.2) and the fault points of RFC-0060 §9.2 ([`faults`], [`fault_point!`]).

#![forbid(unsafe_code)]

pub mod cost_term;
pub mod faults;

pub use cost_term::CostTerm;
