// SPDX-License-Identifier: MIT OR Apache-2.0

//! The placement solver: filter-then-score front end, the DP with greedy in-node placement,
//! per-request routing and plan diffing.
//!
//! Layer 2 in RFC-0001 §4.1, designed in RFC-0011. So far it holds the collective cost model
//! ([`collective`], RFC-0032 §5.3–§5.5), which prices all-reduces for tensor-parallel placement
//! nodes. Like the rest of the solver it is pure: no I/O, clocks or threads (RFC-0001 §4.2 rule 5).

#![forbid(unsafe_code)]

pub mod collective;
