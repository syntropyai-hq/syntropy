// SPDX-License-Identifier: MIT OR Apache-2.0

//! Benchmark scenarios, the result schema and the regression gates (RFC-0061).
//!
//! A dev-only crate in RFC-0001 §4.1. So far it holds [`schema`], the benchmark key, the
//! result record, its JSON form and the comparability rule of RFC-0061 §2; and [`rigs`], the lab
//! inventory of RFC-0061 §6.3 and the `RigFingerprint` computed from it. The `syntropy-bench`
//! binary runs the checks on them that `lint` calls.

#![forbid(unsafe_code)]

pub mod rigs;
pub mod schema;
