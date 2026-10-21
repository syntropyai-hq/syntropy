// SPDX-License-Identifier: MIT OR Apache-2.0

//! Benchmark scenarios, the result schema and the regression gates (RFC-0061).
//!
//! A dev-only crate in RFC-0001 §4.1. So far it holds [`schema`]: the benchmark key, the
//! result record, its JSON form and the comparability rule of RFC-0061 §2.

#![forbid(unsafe_code)]

pub mod schema;
