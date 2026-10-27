// SPDX-License-Identifier: MIT OR Apache-2.0

//! PROD-E2-T01's acceptance tests for the HTTP server (RFC-0050 §1.1–§1.2), on every OS the
//! hosted jobs run:
//!
//! - the `api.bind` schema default is `["127.0.0.1:7433", "[::1]:7433"]` (RFC-0054 §5.2);
//! - `/healthz`, `/readyz` and `/v1/syntropy/version` over HTTP/1.1 on both loopback families,
//!   and over cleartext HTTP/2 with prior knowledge;
//! - `/readyz` is 200 with a state replica and no model loaded, and 503 without a replica;
//! - version reports the build, the protocol majors and the support tier;
//! - the route table's 404 and 405;
//! - the local socket serves the same routes to this user, with its endpoint owner-only;
//! - listeners this build does not serve yet are refused at start.
//!
//! A peer of another user is refused by the local socket's listener,
//! `syntropy_ipc::cmd::AsyncListener`, whose other-user test (`syntropy-ipc`'s `other_user`)
//! connects from a second OS account on each hosted job.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use http_body_util::{BodyExt as _, Empty};
use hyper::body::Bytes;
use hyper::{Method, Request, StatusCode, Version};
use hyper_util::rt::{TokioExecutor, TokioIo};
use syntropy_api::bind::{BindClass, Transport};
use syntropy_api::config::{ApiSection, TlsSource};
use syntropy_api::routes::{ApiState, NodeStatus};
use syntropy_api::server::{ServeError, Server, local_endpoint};
use syntropy_api::version::{Build, ProtocolMajors, SupportTier, VersionInfo};
use syntropy_config::{ConfigSection, DefaultSpec, Registry};
use syntropy_core::build::{BuildId, BuildProfile, TargetTriple};
use syntropy_core::ids::Blake3;
use syntropy_ipc::cmd::{Endpoint, connect_async};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpStream;
use tokio::sync::oneshot;

/// A node whose replica the test sets.
struct Node(AtomicBool);

impl NodeStatus for Node {
    fn has_state_replica(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

fn version() -> VersionInfo {
    VersionInfo {
        build: Build {
            build_id: BuildId {
                version: "0.0.0-test".parse().unwrap(),
                commit: [0x1c; 20],
                profile: BuildProfile::Dev,
                target: TargetTriple("x86_64-unknown-linux-gnu".into()),
                toolchain: Blake3([0x70; 32]),
            },
            features: Vec::new(),
            id: [0x5a; 32],
        },
        protocols: ProtocolMajors::current(),
        support_tier: SupportTier::Tier1,
    }
}

fn state(node: Arc<Node>) -> ApiState {
    ApiState {
        version: version(),
        node,
    }
}

fn api(bind: &[&str]) -> ApiSection {
    ApiSection {
        bind: bind.iter().map(|addr| addr.parse().unwrap()).collect(),
        tls: TlsSource::Auto,
        allow_plaintext_lan: false,
        local_socket: true,
    }
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap()
}

/// A served API, and the signal that stops it.
struct Running {
    addrs: Vec<SocketAddr>,
    stop: oneshot::Sender<()>,
    task: tokio::task::JoinHandle<()>,
}

impl Running {
    async fn stop(self) {
        let _ = self.stop.send(());
        self.task.await.unwrap();
    }
}

fn serve(server: Server) -> Running {
    let addrs = server.local_addrs();
    let (stop, stopped) = oneshot::channel::<()>();
    let task = tokio::spawn(server.serve(async {
        let _ = stopped.await;
    }));
    Running { addrs, stop, task }
}

/// One answer: status, protocol version, headers and the JSON body.
struct Answer {
    status: StatusCode,
    version: Version,
    allow: Option<String>,
    content_type: Option<String>,
    body: serde_json::Value,
}

/// Sends `method path` over `stream`, with HTTP/2 prior knowledge when `h2`.
async fn request<S>(stream: S, method: Method, path: &str, h2: bool) -> Answer
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let request = Request::builder()
        .method(method)
        .uri(path)
        .header(hyper::header::HOST, "localhost")
        .body(Empty::<Bytes>::new())
        .unwrap();
    let response = if h2 {
        let (mut sender, connection) =
            hyper::client::conn::http2::handshake(TokioExecutor::new(), TokioIo::new(stream))
                .await
                .unwrap();
        tokio::spawn(connection);
        sender.send_request(request).await.unwrap()
    } else {
        let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
            .await
            .unwrap();
        tokio::spawn(connection);
        sender.send_request(request).await.unwrap()
    };
    let header = |name| {
        response
            .headers()
            .get(name)
            .map(|v: &hyper::header::HeaderValue| v.to_str().unwrap().to_owned())
    };
    let (status, version) = (response.status(), response.version());
    let (allow, content_type) = (
        header(hyper::header::ALLOW),
        header(hyper::header::CONTENT_TYPE),
    );
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    Answer {
        status,
        version,
        allow,
        content_type,
        body: serde_json::from_slice(&bytes).unwrap(),
    }
}

async fn get(addr: SocketAddr, path: &str, h2: bool) -> Answer {
    request(
        TcpStream::connect(addr).await.unwrap(),
        Method::GET,
        path,
        h2,
    )
    .await
}

/// Whether this host has IPv6 loopback: some containers do not.
fn has_ipv6_loopback() -> bool {
    std::net::TcpListener::bind("[::1]:0").is_ok()
}

#[test]
fn the_api_bind_schema_default_is_both_loopback_addresses() {
    static REGISTRY: Registry = Registry::new(&[&ApiSection::SCHEMA]);
    let key = REGISTRY.key("api.bind").expect("api.bind");
    let DefaultSpec::Value(default) = key.default else {
        panic!("api.bind has no fixed default: {:?}", key.default)
    };
    assert_eq!(default.to_string(), r#"["127.0.0.1:7433", "[::1]:7433"]"#);
    let local = REGISTRY.key("api.local_socket").expect("api.local_socket");
    let DefaultSpec::Value(default) = local.default else {
        panic!("api.local_socket has no fixed default: {:?}", local.default)
    };
    assert_eq!(default.to_string(), "true");
    assert_eq!(local.reload, syntropy_config::Reload::Node);
}

#[test]
fn the_routes_answer_on_both_loopback_families_over_http1_and_h2c() {
    let rt = runtime();
    rt.block_on(async {
        let ipv6 = has_ipv6_loopback();
        if !ipv6 {
            eprintln!("no IPv6 loopback on this host: `[::1]` is not checked");
        }
        let binds: &[&str] = if ipv6 {
            &["127.0.0.1:0", "[::1]:0"]
        } else {
            &["127.0.0.1:0"]
        };
        let node = Arc::new(Node(AtomicBool::new(true)));
        let server = Server::bind(&api(binds), None, false, state(node)).unwrap();
        let running = serve(server);
        assert_eq!(running.addrs.len(), binds.len());
        for &addr in &running.addrs {
            for h2 in [false, true] {
                let expected = if h2 {
                    Version::HTTP_2
                } else {
                    Version::HTTP_11
                };
                let health = get(addr, "/healthz", h2).await;
                assert_eq!(health.status, StatusCode::OK, "{addr} h2={h2}");
                assert_eq!(health.version, expected, "{addr} h2={h2}");
                assert_eq!(health.content_type.as_deref(), Some("application/json"));
                assert_eq!(health.body, serde_json::json!({ "status": "ok" }));

                let ready = get(addr, "/readyz", h2).await;
                assert_eq!(ready.status, StatusCode::OK, "{addr} h2={h2}");
                assert_eq!(ready.body, serde_json::json!({ "ready": true }));

                let version = get(addr, "/v1/syntropy/version", h2).await;
                assert_eq!(version.status, StatusCode::OK, "{addr} h2={h2}");
                assert_eq!(version.body, serde_json::to_value(self::version()).unwrap());
            }
        }
        running.stop().await;
    });
}

#[test]
fn readyz_needs_a_state_replica_and_no_model() {
    let rt = runtime();
    rt.block_on(async {
        // No model is loaded, and none exists: readiness waits only for the replica.
        let node = Arc::new(Node(AtomicBool::new(false)));
        let server =
            Server::bind(&api(&["127.0.0.1:0"]), None, false, state(node.clone())).unwrap();
        let running = serve(server);
        let addr = running.addrs[0];
        let before = get(addr, "/readyz", false).await;
        assert_eq!(before.status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(before.body["ready"], false);
        // Liveness does not wait for it.
        assert_eq!(get(addr, "/healthz", false).await.status, StatusCode::OK);
        node.0.store(true, Ordering::SeqCst);
        let after = get(addr, "/readyz", false).await;
        assert_eq!(after.status, StatusCode::OK);
        assert_eq!(after.body, serde_json::json!({ "ready": true }));
        running.stop().await;
    });
}

#[test]
fn version_reports_the_build_the_protocol_majors_and_the_support_tier() {
    let rt = runtime();
    rt.block_on(async {
        let node = Arc::new(Node(AtomicBool::new(true)));
        let server = Server::bind(&api(&["127.0.0.1:0"]), None, false, state(node)).unwrap();
        let running = serve(server);
        let answer = get(running.addrs[0], "/v1/syntropy/version", false).await;
        assert_eq!(answer.status, StatusCode::OK);
        let body = answer.body;
        assert_eq!(body["build"]["version"], "0.0.0-test");
        assert_eq!(body["build"]["id"], "5a".repeat(32));
        assert_eq!(body["build"]["commit"], "1c".repeat(20));
        assert_eq!(body["build"]["profile"], "dev");
        assert_eq!(
            body["protocols"]["wire"],
            u64::from(syntropy_net::transport::stream::PROTO_MAJOR)
        );
        assert_eq!(
            body["protocols"]["ipc"],
            u64::from(syntropy_ipc::cmd::PROTOCOL.major)
        );
        assert_eq!(body["support_tier"], "tier1");
        running.stop().await;
    });
}

#[test]
fn the_route_table_answers_404_and_405() {
    let rt = runtime();
    rt.block_on(async {
        let node = Arc::new(Node(AtomicBool::new(true)));
        let server = Server::bind(&api(&["127.0.0.1:0"]), None, false, state(node)).unwrap();
        let running = serve(server);
        let addr = running.addrs[0];
        let missing = get(addr, "/v1/no/such/route", false).await;
        assert_eq!(missing.status, StatusCode::NOT_FOUND);
        assert_eq!(missing.body["error"]["code"], "unsupported_endpoint");
        assert_eq!(missing.body["error"]["class"], "Unsupported");
        let stream = TcpStream::connect(addr).await.unwrap();
        let wrong = request(stream, Method::POST, "/healthz", false).await;
        assert_eq!(wrong.status, StatusCode::METHOD_NOT_ALLOWED);
        assert_eq!(wrong.allow.as_deref(), Some("GET"));
        assert_eq!(wrong.body["error"]["code"], "method_not_allowed");
        // A query does not change the route.
        assert_eq!(
            get(addr, "/healthz?probe=1", false).await.status,
            StatusCode::OK
        );
        running.stop().await;
    });
}

/// A fresh local-socket endpoint for one test.
fn local_test_endpoint(tag: &str) -> (Endpoint, Option<PathBuf>) {
    let name = format!("syntropy-api-test-{tag}-{}", std::process::id());
    if cfg!(windows) {
        (Endpoint::Pipe(name), None)
    } else {
        let dir = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&dir);
        (Endpoint::Socket(dir.join("api.sock")), Some(dir))
    }
}

#[test]
fn the_local_socket_serves_the_same_routes_to_this_user() {
    let rt = runtime();
    rt.block_on(async {
        let (endpoint, dir) = local_test_endpoint("local");
        let node = Arc::new(Node(AtomicBool::new(true)));
        let server =
            Server::bind(&api(&["127.0.0.1:0"]), Some(&endpoint), false, state(node)).unwrap();
        assert_eq!(server.local_socket(), Some(&endpoint));
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let Endpoint::Socket(path) = &endpoint else {
                unreachable!()
            };
            let socket = std::fs::symlink_metadata(path).unwrap();
            assert_eq!(socket.mode() & 0o777, 0o600, "the socket's mode");
            let parent = std::fs::symlink_metadata(path.parent().unwrap()).unwrap();
            assert_eq!(parent.mode() & 0o777, 0o700, "its directory's mode");
        }
        let running = serve(server);
        for h2 in [false, true] {
            for path in ["/healthz", "/readyz", "/v1/syntropy/version"] {
                let stream = connect_async(&endpoint).await.unwrap();
                let answer = request(stream, Method::GET, path, h2).await;
                assert_eq!(answer.status, StatusCode::OK, "{path} h2={h2}");
            }
        }
        let stream = connect_async(&endpoint).await.unwrap();
        let version = request(stream, Method::GET, "/v1/syntropy/version", false).await;
        assert_eq!(version.body, serde_json::to_value(self::version()).unwrap());
        running.stop().await;
        if let Some(dir) = dir {
            // The socket goes with the server.
            assert!(!dir.join("api.sock").exists());
            let _ = std::fs::remove_dir(dir);
        }
    });
}

#[test]
fn the_local_endpoint_follows_the_runtime_location() {
    let dir = std::env::temp_dir().join("syntropy-runtime");
    assert_eq!(
        local_endpoint(&syntropy_config::Runtime::Dir(dir.clone())).unwrap(),
        Endpoint::Socket(dir.join("api.sock"))
    );
    let pipes = local_endpoint(&syntropy_config::Runtime::NamedPipes);
    if cfg!(windows) {
        let Endpoint::Pipe(name) = pipes.unwrap() else {
            panic!("a socket on Windows")
        };
        assert!(name.starts_with("syntropy-S-1-"), "{name}");
        assert!(name.ends_with("-api"), "{name}");
    } else {
        assert_eq!(pipes.unwrap_err().kind(), std::io::ErrorKind::Unsupported);
    }
}

#[test]
fn listeners_this_build_does_not_serve_yet_are_refused() {
    let rt = runtime();
    let _context = rt.enter();
    let node = Arc::new(Node(AtomicBool::new(true)));
    // A LAN bind with a key: the plan gives it TLS, which PROD-E5-T06 serves.
    let error = Server::bind(
        &api(&["192.168.1.10:7433"]),
        None,
        true,
        state(node.clone()),
    )
    .unwrap_err();
    let ServeError::NotServedYet { listener } = &error else {
        panic!("{error}")
    };
    assert_eq!(listener.class, BindClass::Lan);
    assert!(matches!(listener.transport, Transport::Tls(_)));
    assert!(
        error
            .to_string()
            .contains("does not serve TLS listeners yet"),
        "{error}"
    );
    // Without a key, the bind policy refuses first, naming the command.
    let error = Server::bind(
        &api(&["192.168.1.10:7433"]),
        None,
        false,
        state(node.clone()),
    )
    .unwrap_err();
    assert!(matches!(error, ServeError::Refused(_)), "{error}");
    assert!(
        error.to_string().contains("syntropy api-key create"),
        "{error}"
    );
    // Plain HTTP on a LAN address, with the opt-in: it needs keys on every route, not built yet.
    let mut plain = api(&["192.168.1.10:7433"]);
    plain.tls = TlsSource::None;
    plain.allow_plaintext_lan = true;
    let error = Server::bind(&plain, None, true, state(node)).unwrap_err();
    assert!(
        matches!(
            &error,
            ServeError::NotServedYet { listener } if listener.transport == Transport::Plain
        ),
        "{error}"
    );
}

#[test]
fn a_taken_port_is_a_bind_error_naming_the_address() {
    let rt = runtime();
    let _context = rt.enter();
    let taken = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = taken.local_addr().unwrap().to_string();
    let node = Arc::new(Node(AtomicBool::new(true)));
    let error = Server::bind(&api(&[&addr]), None, false, state(node)).unwrap_err();
    assert!(matches!(error, ServeError::Bind { .. }), "{error}");
    assert!(error.to_string().contains(&addr), "{error}");
}
