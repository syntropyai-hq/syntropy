// SPDX-License-Identifier: MIT OR Apache-2.0

//! The HTTP API surfaces, ingress routing, sequence channels and dashboard state endpoints.
//!
//! Layer 4 in RFC-0001 §4.1, designed in RFC-0050. So far it holds the HTTP server and its route
//! table (RFC-0050 §1.1–§1.2, PROD-E2-T01), the bind policy of RFC-0050 §11.1 (PROD-E5-T02), the
//! node's local TLS CA (PROD-E5-T03) and the `Figure` type that `doctor`'s evidence uses
//! (PROD-E9-T01):
//!
//! - [`server`]: the `api.bind` listeners and the local socket, served with HTTP/1.1 and
//!   cleartext HTTP/2 on the daemon's `ctl` runtime;
//! - [`routes`]: the route table, with `/healthz`, `/readyz` and `/v1/syntropy/version`;
//! - [`version`]: what `/v1/syntropy/version` reports;
//! - [`config`]: the `[api]` section's listener keys, `api.bind`, `api.tls`,
//!   `api.allow_plaintext_lan` and `api.local_socket` (RFC-0054 §5.2);
//! - [`bind`]: each bind address's class (loopback, LAN or public) and [`bind::plan_binds`],
//!   which the daemon runs at start to choose each listener's transport, or to refuse to start;
//! - [`rules`]: the validation rules for those keys, which the daemon passes to
//!   `syntropy_config::load_and_validate`;
//! - [`tls`]: the node's local TLS CA and the leaves it serves (RFC-0050 §11.1.1);
//! - [`figure`]: [`figure::Figure`], how every predicted or measured number reaches a client with
//!   its kind and basis (RFC-0001 §5.16).
//!
//! Binds restricted to overlay interfaces, which may omit TLS, need RFC-0030's interface classes
//! and come with PROD-E5-T09.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod bind;
pub mod config;
pub mod figure;
pub mod routes;
pub mod rules;
pub mod server;
pub mod tls;
pub mod version;
