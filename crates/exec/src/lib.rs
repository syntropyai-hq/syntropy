// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lowers the graph to per-stage step programs: per-device dispatchers, the CPU compute pool,
//! doorbells and graph capture.
//!
//! Stub: layer 3 in RFC-0001 §4.1, designed in RFC-0012. Nothing is implemented yet.

#![forbid(unsafe_code)]
