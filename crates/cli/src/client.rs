// SPDX-License-Identifier: MIT OR Apache-2.0

//! The typed native-API client (RFC-0051 §1; ADR-0503 decisions 3 and 5).
//!
//! **Reaching a daemon.** [`targets`] lists where to look, in order:
//! 1. the local socket, `<runtime dir>/api.sock` or the named pipe `syntropy-<user SID>-api`,
//!    which needs no API key because the daemon checks the OS peer credentials;
//! 2. loopback HTTP, `http://127.0.0.1:7433`.
//!
//! `SYNTROPY_HOST` names a remote daemon instead (RFC-0054 §2.4). When it is set, the client
//! reaches that daemon alone, sending `SYNTROPY_API_KEY` as a bearer token, and never falls back
//! to a local daemon, so a command never acts on another node than the one asked for. Only
//! `http://` is served: the TLS client for `https://` comes with the API's TLS listeners
//! (PROD-E5-T06) and `syntropy trust`.
//!
//! The client speaks HTTP/1.1 over each target through hyper, on a single-threaded tokio
//! runtime the client owns, and reads and writes the `syntropy-api` types the server uses, so the
//! CLI and the server cannot disagree about a field.

use std::fmt;
use std::time::Duration;

use http_body_util::{BodyExt as _, Empty};
use hyper::body::Bytes;
use hyper::{Request, StatusCode};
use hyper_util::rt::TokioIo;
use serde::de::DeserializeOwned;
use syntropy_api::routes::ErrorBody;
use syntropy_api::version::VersionInfo;
use syntropy_config::{Env, Mode, Os, Secret, locate};
use syntropy_ipc::cmd::{Endpoint, connect_async};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpStream;

use crate::error::{CliError, Exit};

/// The loopback address a local daemon serves by default (RFC-0054 §5.2, `api.bind`).
pub const LOOPBACK: &str = "127.0.0.1:7433";

/// The port `SYNTROPY_HOST` means when it names none.
pub const DEFAULT_PORT: u16 = 7433;

/// How long a connection may take to open \[target\]: a chosen budget. A local target answers or
/// refuses at once; this bounds a remote one that does neither.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);

/// The largest answer the client reads \[target\]: a chosen budget, ample for every read route, so
/// a wrong or hostile listener cannot fill memory.
pub const MAX_RESPONSE: usize = 4 << 20;

/// How long a request may take once connected \[target\]: a chosen budget for the read routes.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// One place a daemon may answer.
#[derive(Clone)]
pub enum Target {
    /// The local socket.
    LocalSocket(Endpoint),
    /// HTTP over TCP.
    Http {
        /// The address to connect to, `host:port`.
        addr: String,
        /// Whether `SYNTROPY_HOST` named it.
        remote: bool,
        /// `SYNTROPY_API_KEY`, sent only to a remote daemon.
        api_key: Option<Secret<String>>,
    },
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Target::LocalSocket(endpoint) => write!(f, "the local socket ({endpoint})"),
            Target::Http {
                addr, remote: true, ..
            } => write!(f, "http://{addr} (SYNTROPY_HOST)"),
            Target::Http { addr, .. } => write!(f, "http://{addr}"),
        }
    }
}

/// Never prints the key.
impl fmt::Debug for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl Target {
    /// Whether `SYNTROPY_HOST` named it.
    #[must_use]
    pub fn is_remote(&self) -> bool {
        matches!(self, Target::Http { remote: true, .. })
    }
}

/// Where to look for a daemon, in order, for `os` and `env` (see the module docs).
///
/// # Errors
///
/// When `SYNTROPY_HOST` is not an `http://host[:port]` URL.
pub fn targets(os: Os, env: &Env) -> Result<Vec<Target>, CliError> {
    if let Some(host) = env.host() {
        let addr = parse_host(&host.to_string_lossy())?;
        return Ok(vec![Target::Http {
            addr,
            remote: true,
            api_key: env.api_key(),
        }]);
    }
    let mut found = Vec::with_capacity(2);
    let runtime = locate(os, Mode::User, env, None).dirs.runtime;
    if let Some(endpoint) = runtime.and_then(|r| syntropy_api::server::local_endpoint(&r).ok()) {
        found.push(Target::LocalSocket(endpoint));
    }
    found.push(Target::Http {
        addr: LOOPBACK.to_owned(),
        remote: false,
        api_key: None,
    });
    Ok(found)
}

/// `SYNTROPY_HOST`'s `host:port`, from `http://host[:port][/]`.
fn parse_host(text: &str) -> Result<String, CliError> {
    let bad = |why: &str| {
        CliError::cli(
            Exit::Invalid,
            "bad_host",
            format!("SYNTROPY_HOST=`{text}` {why}"),
        )
        .hint("export SYNTROPY_HOST=http://127.0.0.1:7433")
    };
    if text.starts_with("https://") {
        return Err(no_tls(format!(
            "SYNTROPY_HOST=`{text}` asks for TLS, and this build's client speaks plain HTTP only"
        )));
    }
    let Some(rest) = text.strip_prefix("http://") else {
        return Err(bad("is not an http:// URL"));
    };
    let authority = rest.strip_suffix('/').unwrap_or(rest);
    if authority.is_empty() || authority.contains(['/', '?', '#', '@']) {
        return Err(bad("must be http://<host>[:<port>] with no path"));
    }
    // `[v6]` or `[v6]:port`; otherwise `host` or `host:port`.
    let port = match authority.strip_prefix('[') {
        Some(rest) => match rest.split_once(']') {
            Some((_, "")) => None,
            Some((_, after)) => match after.strip_prefix(':') {
                Some(port) => Some(port),
                None => return Err(bad("must be http://<host>[:<port>] with no path")),
            },
            None => return Err(bad("has an IPv6 address without its closing `]`")),
        },
        None => authority.split_once(':').map(|(_, port)| port),
    };
    if port.is_some_and(|port| port.parse::<u16>().is_err()) {
        return Err(bad("has a port that is not a number from 0 to 65535"));
    }
    let host = match port {
        Some(port) => &authority[..authority.len() - port.len() - 1],
        None => authority,
    };
    // Plain HTTP only to this machine: no daemon serves plain HTTP beyond loopback (RFC-0050
    // §11.1), so anything that answers there is not one, and `SYNTROPY_API_KEY` would cross the
    // network in the clear.
    if !is_loopback_host(host) {
        return Err(no_tls(format!(
            "SYNTROPY_HOST=`{text}` is plain HTTP to a host beyond this machine, which would send \
             SYNTROPY_API_KEY in the clear; a daemon serves such hosts only over TLS"
        )));
    }
    match port {
        Some(_) => Ok(authority.to_owned()),
        None => Ok(format!("{authority}:{DEFAULT_PORT}")),
    }
}

/// Whether `host`, from a URL, is this machine: `localhost`, `127.0.0.0/8` or `[::1]`.
fn is_loopback_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    let ip = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host);
    ip.parse::<std::net::IpAddr>()
        .is_ok_and(|ip| ip.to_canonical().is_loopback())
}

/// The exit-5 error for a daemon that only TLS can reach, which this build's client lacks.
fn no_tls(message: String) -> CliError {
    CliError::cli(Exit::Invalid, "not_built", message)
        .detail("the TLS client comes with the API's TLS listeners and `syntropy trust`")
        .hint(
            "forward the daemon's port to this machine, as `ssh -L 7433:127.0.0.1:7433 <host>`, \
             then export SYNTROPY_HOST=http://127.0.0.1:7433",
        )
}

/// A daemon the client reached, and the runtime it talks to it on.
pub struct Client {
    runtime: tokio::runtime::Runtime,
    target: Target,
}

impl fmt::Debug for Client {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Client")
            .field("target", &self.target)
            .finish_non_exhaustive()
    }
}

/// A connected byte stream to a target.
trait Io: AsyncRead + AsyncWrite + Unpin + Send + 'static {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send + 'static> Io for T {}

/// Opens a connection to `target`.
async fn open(target: &Target) -> std::io::Result<Box<dyn Io>> {
    match target {
        Target::LocalSocket(endpoint) => {
            let stream = tokio::time::timeout(CONNECT_TIMEOUT, connect_async(endpoint))
                .await
                .map_err(|_| std::io::ErrorKind::TimedOut)??;
            Ok(Box::new(stream))
        }
        Target::Http { addr, .. } => {
            let stream = tokio::time::timeout(CONNECT_TIMEOUT, TcpStream::connect(addr.as_str()))
                .await
                .map_err(|_| std::io::ErrorKind::TimedOut)??;
            stream.set_nodelay(true)?;
            Ok(Box::new(stream))
        }
    }
}

impl Client {
    /// Reaches the first of `targets` that accepts a connection.
    ///
    /// # Errors
    ///
    /// Exit 3 when none does, listing each one tried.
    pub fn reach(targets: Vec<Target>) -> Result<Client, CliError> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .enable_time()
            .build()
            .map_err(|error| {
                CliError::cli(
                    Exit::Failure,
                    "runtime",
                    format!("cannot start the client: {error}"),
                )
            })?;
        let mut tried = Vec::with_capacity(targets.len());
        for target in targets {
            match runtime.block_on(open(&target)) {
                Ok(_) => return Ok(Client { runtime, target }),
                Err(error) => tried.push((target, error)),
            }
        }
        Err(no_daemon(&tried))
    }

    /// Where the client reached the daemon.
    #[must_use]
    pub fn target(&self) -> &Target {
        &self.target
    }

    /// `GET path`, read as `T`.
    ///
    /// # Errors
    ///
    /// The daemon's error, mapped to its exit code; or a failed or malformed exchange.
    pub fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T, CliError> {
        let (status, body) = self.runtime.block_on(async {
            tokio::time::timeout(REQUEST_TIMEOUT, exchange(&self.target, path))
                .await
                .unwrap_or_else(|_| {
                    Err(CliError::cli(
                        Exit::Timeout,
                        "timeout",
                        format!("the daemon at {} did not answer `GET {path}`", self.target),
                    ))
                })
        })?;
        if status.is_success() {
            return serde_json::from_slice(&body).map_err(|error| {
                CliError::cli(
                    Exit::Failure,
                    "bad_response",
                    format!(
                        "the daemon's answer to `GET {path}` is not the expected type: {error}"
                    ),
                )
            });
        }
        match serde_json::from_slice::<ErrorBody>(&body) {
            Ok(body) => Err(CliError::api(status.as_u16(), &body.error)),
            Err(_) => Err(CliError::cli(
                Exit::Failure,
                "bad_response",
                format!("the daemon answered `GET {path}` with {status} and no error body"),
            )),
        }
    }

    /// `GET /v1/syntropy/version`.
    ///
    /// # Errors
    ///
    /// As [`Client::get`].
    pub fn version(&self) -> Result<VersionInfo, CliError> {
        self.get("/v1/syntropy/version")
    }
}

/// One HTTP/1.1 `GET` to `target`: its status and body.
async fn exchange(target: &Target, path: &str) -> Result<(StatusCode, Bytes), CliError> {
    let failed = |error: &dyn fmt::Display| {
        CliError::cli(
            Exit::Failure,
            "exchange",
            format!("`GET {path}` to the daemon at {target} failed: {error}"),
        )
    };
    let stream = open(target).await.map_err(|error| failed(&error))?;
    let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
        .await
        .map_err(|error| failed(&error))?;
    tokio::spawn(connection);
    let host = match target {
        Target::LocalSocket(_) => "localhost",
        Target::Http { addr, .. } => addr.as_str(),
    };
    let mut request = Request::get(path).header(hyper::header::HOST, host);
    if let Target::Http {
        remote: true,
        api_key: Some(key),
        ..
    } = target
    {
        request = request.header(
            hyper::header::AUTHORIZATION,
            format!("Bearer {}", key.expose_for("the remote daemon's API key")),
        );
    }
    let request = request
        .body(Empty::<Bytes>::new())
        .map_err(|error| failed(&error))?;
    let response = sender
        .send_request(request)
        .await
        .map_err(|error| failed(&error))?;
    let status = response.status();
    let body = http_body_util::Limited::new(response.into_body(), MAX_RESPONSE)
        .collect()
        .await
        .map_err(|error| {
            if error.is::<http_body_util::LengthLimitError>() {
                CliError::cli(
                    Exit::Failure,
                    "bad_response",
                    format!(
                        "the daemon's answer to `GET {path}` is over the client's {} limit",
                        crate::output::iec_bytes(MAX_RESPONSE as u64)
                    ),
                )
            } else {
                failed(&error)
            }
        })?
        .to_bytes();
    Ok((status, body))
}

/// The exit-3 error after trying `tried`.
fn no_daemon(tried: &[(Target, std::io::Error)]) -> CliError {
    let remote = tried.iter().any(|(target, _)| target.is_remote());
    let mut error = CliError::cli(
        Exit::NoDaemon,
        "no_daemon",
        "no Syntropy daemon is reachable",
    );
    for (target, why) in tried {
        error = error.detail(format!("tried {target}: {why}"));
    }
    if remote {
        error.hint("check that the daemon at SYNTROPY_HOST runs, or unset SYNTROPY_HOST")
    } else {
        error.hint("syntropy serve")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> Env {
        Env::from_pairs(Os::Linux, pairs.iter().copied())
    }

    #[test]
    fn the_local_socket_comes_first_then_loopback() {
        let found = targets(Os::Linux, &env(&[("XDG_RUNTIME_DIR", "/run/user/1000")])).unwrap();
        assert_eq!(found.len(), 2, "{found:?}");
        assert!(
            matches!(&found[0], Target::LocalSocket(Endpoint::Socket(path))
                if path == std::path::Path::new("/run/user/1000/syntropy/api.sock")),
            "{found:?}"
        );
        assert!(
            matches!(&found[1], Target::Http { addr, remote: false, api_key: None }
                if addr == LOOPBACK),
            "{found:?}"
        );
        // With no runtime directory there is no local socket to try.
        let found = targets(Os::Linux, &env(&[])).unwrap();
        assert_eq!(found.len(), 1);
        assert!(!found[0].is_remote());
    }

    #[test]
    fn syntropy_host_replaces_every_local_target_and_carries_the_key() {
        let found = targets(
            Os::Linux,
            &env(&[
                ("XDG_RUNTIME_DIR", "/run/user/1000"),
                ("SYNTROPY_HOST", "http://127.0.0.2:7433"),
                ("SYNTROPY_API_KEY", "sk-test"),
            ]),
        )
        .unwrap();
        assert_eq!(found.len(), 1);
        let Target::Http {
            addr,
            remote,
            api_key,
        } = &found[0]
        else {
            panic!("{found:?}")
        };
        assert_eq!((addr.as_str(), *remote), ("127.0.0.2:7433", true));
        assert_eq!(
            api_key.as_ref().map(|k| k.expose_for("test").as_str()),
            Some("sk-test")
        );
        // The key never shows in the target's text.
        assert!(!format!("{:?} {}", found[0], found[0]).contains("sk-test"));
    }

    #[test]
    fn syntropy_host_takes_http_urls_with_an_optional_port() {
        for (text, addr) in [
            ("http://localhost:8080", "localhost:8080"),
            ("http://LOCALHOST", "LOCALHOST:7433"),
            ("http://localhost/", "localhost:7433"),
            ("http://127.0.0.1", "127.0.0.1:7433"),
            ("http://127.5.6.7:9000", "127.5.6.7:9000"),
            ("http://[::1]:7433", "[::1]:7433"),
            ("http://[::1]", "[::1]:7433"),
            ("http://[::ffff:127.0.0.1]", "[::ffff:127.0.0.1]:7433"),
        ] {
            assert_eq!(parse_host(text).unwrap(), addr, "{text}");
        }
        // Plain HTTP beyond this machine would send the key in the clear: refused, as TLS is.
        for text in [
            "http://10.0.0.2:7433",
            "http://box.tailnet",
            "http://[fd00::2]",
            "http://192.168.1.10:7433",
            "http://localhost.example.com",
            "http://128.0.0.1",
        ] {
            let error = parse_host(text).unwrap_err();
            assert_eq!(
                (error.exit, error.code.as_str()),
                (Exit::Invalid, "not_built"),
                "{text}"
            );
            assert!(error.message.contains("in the clear"), "{text}: {error}");
            assert!(
                error.hints[0].contains("ssh -L 7433:127.0.0.1:7433"),
                "{error}"
            );
        }
        for text in [
            "box:7433",
            "ftp://box",
            "http://",
            "http://box/v1",
            "http://box:port",
        ] {
            let error = parse_host(text).unwrap_err();
            assert_eq!(
                (error.exit, error.code.as_str()),
                (Exit::Invalid, "bad_host"),
                "{text}"
            );
        }
        let tls = parse_host("https://box:7433").unwrap_err();
        assert_eq!((tls.exit, tls.code.as_str()), (Exit::Invalid, "not_built"));
    }

    #[test]
    fn no_daemon_lists_what_was_tried_and_exits_3() {
        let closed = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = closed.local_addr().unwrap().to_string();
        drop(closed);
        let error = Client::reach(vec![Target::Http {
            addr: addr.clone(),
            remote: false,
            api_key: None,
        }])
        .unwrap_err();
        assert_eq!(error.exit, Exit::NoDaemon);
        assert!(
            error.details[0].starts_with(&format!("tried http://{addr}:")),
            "{error}"
        );
        assert_eq!(error.hints, ["syntropy serve"]);
    }
}
