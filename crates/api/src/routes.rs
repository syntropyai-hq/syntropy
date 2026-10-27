// SPDX-License-Identifier: MIT OR Apache-2.0

//! The route table of RFC-0050 §1.2, and the routes served so far: `GET /healthz`,
//! `GET /readyz` and `GET /v1/syntropy/version`.
//!
//! [`ROUTES`] maps a method and an exact path to a handler. A path with no route gets 404
//! `unsupported_endpoint`; a path whose route takes another method gets 405 with an `Allow`
//! header. The tasks that build the other routes of §1.2 add them to the table. The error bodies
//! carry the native shape's `class`, `code` and `message`; the full mapping of RFC-0050 §12,
//! with each vendor shape's body, is PROD-E2-T05's.

use std::convert::Infallible;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt as _, Full};
use hyper::body::{Bytes, Incoming};
use hyper::header::{self, HeaderValue};
use hyper::{Method, Request, Response, StatusCode};
use serde::Serialize;

use crate::version::VersionInfo;

/// A response body. Every route so far answers with one JSON document; the type leaves room for
/// the streaming routes.
pub type Body = BoxBody<Bytes, Infallible>;

/// What a handler returns.
pub type Reply<'a> = Pin<Box<dyn Future<Output = Response<Body>> + Send + 'a>>;

/// A route's handler.
pub type Handler = for<'a> fn(&'a ApiState, Request<Incoming>) -> Reply<'a>;

/// One row of the route table.
#[derive(Clone, Debug)]
pub struct Route {
    /// The method it answers.
    pub method: Method,
    /// The exact path, without a query.
    pub path: &'static str,
    /// The handler.
    pub handler: Handler,
}

/// The routes served, in RFC-0050 §1.2's order.
pub const ROUTES: &[Route] = &[
    Route {
        method: Method::GET,
        path: "/v1/syntropy/version",
        handler: version,
    },
    Route {
        method: Method::GET,
        path: "/healthz",
        handler: healthz,
    },
    Route {
        method: Method::GET,
        path: "/readyz",
        handler: readyz,
    },
];

/// What the daemon tells the API about the node, for `/readyz`.
pub trait NodeStatus: Send + Sync + 'static {
    /// Whether the daemon holds a cluster state replica, or runs solo and needs none (RFC-0050
    /// §1.2). Readiness waits for nothing else: no model need be loaded.
    fn has_state_replica(&self) -> bool;
}

/// What every handler can read.
pub struct ApiState {
    /// The body of `GET /v1/syntropy/version`.
    pub version: VersionInfo,
    /// The node, as the daemon reports it.
    pub node: Arc<dyn NodeStatus>,
}

impl std::fmt::Debug for ApiState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiState")
            .field("version", &self.version)
            .field("has_state_replica", &self.node.has_state_replica())
            .finish()
    }
}

/// Answers `request` from the route table.
pub async fn dispatch(state: &ApiState, request: Request<Incoming>) -> Response<Body> {
    let path = request.uri().path();
    let on_path = || ROUTES.iter().filter(move |route| route.path == path);
    if let Some(route) = on_path().find(|route| route.method == request.method()) {
        return (route.handler)(state, request).await;
    }
    let allow: Vec<&str> = on_path().map(|route| route.method.as_str()).collect();
    if allow.is_empty() {
        let message = format!("no endpoint at {} {path}", request.method());
        return error(
            StatusCode::NOT_FOUND,
            "Unsupported",
            "unsupported_endpoint",
            &message,
        );
    }
    let allow = allow.join(", ");
    let message = format!("{path} takes {allow}, not {}", request.method());
    let mut response = error(
        StatusCode::METHOD_NOT_ALLOWED,
        "InvalidRequest",
        "method_not_allowed",
        &message,
    );
    if let Ok(value) = HeaderValue::from_str(&allow) {
        response.headers_mut().insert(header::ALLOW, value);
    }
    response
}

/// A JSON response with `status`.
pub fn json<T: Serialize>(status: StatusCode, body: &T) -> Response<Body> {
    // The bodies are plain structs and maps, which always serialise.
    let bytes = serde_json::to_vec(body).unwrap_or_default();
    let mut response = Response::new(Full::new(Bytes::from(bytes)).boxed());
    *response.status_mut() = status;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}

/// An error in the native shape's form: `{"error": {"class", "code", "message"}}`.
fn error(status: StatusCode, class: &str, code: &str, message: &str) -> Response<Body> {
    json(
        status,
        &serde_json::json!({ "error": { "class": class, "code": code, "message": message } }),
    )
}

fn version(state: &ApiState, _: Request<Incoming>) -> Reply<'_> {
    Box::pin(async move { json(StatusCode::OK, &state.version) })
}

/// Liveness: the daemon answers.
fn healthz(_: &ApiState, _: Request<Incoming>) -> Reply<'_> {
    Box::pin(async { json(StatusCode::OK, &serde_json::json!({ "status": "ok" })) })
}

/// Readiness: 200 once the daemon holds a cluster state replica or runs solo, 503 before. It
/// never waits for a model (RFC-0050 §1.2).
fn readyz(state: &ApiState, _: Request<Incoming>) -> Reply<'_> {
    Box::pin(async move {
        if state.node.has_state_replica() {
            json(StatusCode::OK, &serde_json::json!({ "ready": true }))
        } else {
            json(
                StatusCode::SERVICE_UNAVAILABLE,
                &serde_json::json!({
                    "ready": false,
                    "reason": "no cluster state replica yet",
                }),
            )
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_route_is_unique_and_absolute() {
        for (i, route) in ROUTES.iter().enumerate() {
            assert!(route.path.starts_with('/'), "{}", route.path);
            assert!(!route.path.contains('?'), "{}", route.path);
            for other in &ROUTES[..i] {
                assert!(
                    (other.method.clone(), other.path) != (route.method.clone(), route.path),
                    "{} {} twice",
                    route.method,
                    route.path
                );
            }
        }
    }

    #[test]
    fn the_m0_routes_are_in_the_table() {
        let table: Vec<(Method, &str)> =
            ROUTES.iter().map(|r| (r.method.clone(), r.path)).collect();
        for path in ["/healthz", "/readyz", "/v1/syntropy/version"] {
            assert!(table.contains(&(Method::GET, path)), "{path}");
        }
    }
}
