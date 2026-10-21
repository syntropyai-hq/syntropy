// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the benchmark result schema (QUAL-E12-T01; RFC-0061 §2.1, §2.2).
//!
//! Properties, named per RFC-0060 §3.1:
//! - P-bench-1: every result survives a JSON round trip unchanged, and its JSON passes the
//!   committed schema file (`round_trip`).
//! - P-bench-2: two results whose keys differ only in `build` are comparable, and a difference
//!   in any other key field makes them incomparable (`comparability`).

mod checker;
mod comparability;
mod round_trip;
mod schema_file;
mod support;
mod validation;
