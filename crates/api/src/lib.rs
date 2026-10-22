// SPDX-License-Identifier: MIT OR Apache-2.0

//! The HTTP API surfaces, ingress routing, sequence channels and dashboard state endpoints.
//!
//! Layer 4 in RFC-0001 §4.1, designed in RFC-0050. So far it holds [`tls`]: the node's local
//! TLS CA and the leaves it serves (RFC-0050 §11.1.1).

#![forbid(unsafe_code)]

pub mod tls;
