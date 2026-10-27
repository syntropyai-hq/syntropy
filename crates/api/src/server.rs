// SPDX-License-Identifier: MIT OR Apache-2.0

//! The HTTP server of RFC-0050 §1.1: its listeners and how each connection is served.
//!
//! - **`api.bind` listeners.** [`Server::bind`] runs [`plan_binds`] and opens one TCP listener per
//!   address. Each serves HTTP/1.1 and cleartext HTTP/2 with prior knowledge, told apart by the
//!   HTTP/2 preface. `TCP_NODELAY` is set on every accepted socket, so a streamed delta leaves the
//!   host when it is written.
//! - **The local socket** (`api.local_socket`): [`syntropy_ipc::cmd::AsyncListener`], a Unix
//!   socket of mode 0600 in a 0700 directory, or a named pipe whose DACL grants this user alone.
//!   Every connection's peer is checked with the OS peer-credential call and refused unless it is
//!   the daemon's own user. The same routes are served on both.
//!
//! - **A missing address family.** A loopback listener whose address family the host lacks, such
//!   as `[::1]` in a container without IPv6 or under `ipv6.disable=1`, is skipped and reported in
//!   [`Server::skipped`], as long as another `api.bind` listener bound. The default `api.bind`
//!   names both families, so such a host still starts on `127.0.0.1`. If none bound, the start
//!   fails.
//! - **Timeouts.** Each connection gets a timer, and HTTP/1 requests must deliver their headers
//!   within [`HEADER_READ_TIMEOUT`].
//!
//! Every request carries a [`Via`] extension: the listener and peer it came from, which the
//! authentication of RFC-0050 §11 reads.
//!
//! **Not served yet.** Listeners beyond loopback need API keys (PROD-E5-T01) and TLS
//! (PROD-E5-T06). Until those exist, [`Server::bind`] refuses to start a plan that holds one,
//! with [`ServeError::NotServedYet`], rather than serve it without the protection RFC-0050 §11.1
//! requires.

use std::convert::Infallible;
use std::fmt;
use std::future::Future;
use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use hyper::Request;
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper_util::rt::{TokioExecutor, TokioIo, TokioTimer};
use hyper_util::server::conn::auto;
use syntropy_ipc::cmd::{AcceptError, AsyncListener, Endpoint, Peer};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinSet;

use crate::bind::{BindClass, Listener, StartRefusal, Transport, plan_binds};
use crate::config::ApiSection;
use crate::routes::{ApiState, dispatch};

/// Where a request came from, as a request extension.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Via {
    /// A TCP listener from `api.bind`.
    Tcp {
        /// The listener's address.
        local: SocketAddr,
        /// The client's address.
        remote: SocketAddr,
        /// The listener's class: loopback, until PROD-E5-T06 serves the others.
        class: BindClass,
    },
    /// The local socket, with the peer the OS reported: always the daemon's own user.
    LocalSocket(Peer),
}

/// Why the server did not start.
#[derive(Debug)]
pub enum ServeError {
    /// The bind policy refuses the configuration (RFC-0050 §11.1).
    Refused(StartRefusal),
    /// A listener the plan holds that this build cannot serve yet: TLS (PROD-E5-T06), or plain
    /// HTTP beyond loopback, which needs API keys on every route (PROD-E5-T01).
    NotServedYet {
        /// The planned listener.
        listener: Listener,
    },
    /// An address could not be bound.
    Bind {
        /// The address from `api.bind`.
        addr: SocketAddr,
        /// The error.
        error: io::Error,
    },
    /// The local socket could not be created.
    LocalSocket {
        /// Its endpoint.
        endpoint: Endpoint,
        /// The error.
        error: io::Error,
    },
}

impl fmt::Display for ServeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ServeError::Refused(refusal) => refusal.fmt(f),
            ServeError::NotServedYet { listener } => {
                let what = match listener.transport {
                    Transport::Tls(_) => "TLS listeners",
                    Transport::Plain => "plain-HTTP listeners beyond loopback",
                };
                write!(
                    f,
                    "refusing to start: `api.bind` holds `{}`, and this build does not serve \
                     {what} yet; bind a loopback address such as `127.0.0.1:7433`",
                    listener.addr
                )
            }
            ServeError::Bind { addr, error } => {
                write!(f, "cannot listen on `{addr}` from `api.bind`: {error}")
            }
            ServeError::LocalSocket { endpoint, error } => {
                write!(
                    f,
                    "cannot create the API's local socket at {endpoint}: {error}"
                )
            }
        }
    }
}

impl std::error::Error for ServeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ServeError::Refused(refusal) => Some(refusal),
            ServeError::Bind { error, .. } | ServeError::LocalSocket { error, .. } => Some(error),
            ServeError::NotServedYet { .. } => None,
        }
    }
}

/// The local socket's endpoint in the runtime location `runtime` (RFC-0054 §1):
/// `<runtime dir>/api.sock`, or the named pipe `syntropy-<user SID>-api` on Windows.
///
/// # Errors
///
/// When this user's SID cannot be read, on Windows.
pub fn local_endpoint(runtime: &syntropy_config::Runtime) -> io::Result<Endpoint> {
    match runtime {
        syntropy_config::Runtime::Dir(dir) => Ok(Endpoint::Socket(dir.join("api.sock"))),
        #[cfg(windows)]
        syntropy_config::Runtime::NamedPipes => syntropy_ipc::cmd::user_pipe("api"),
        #[cfg(not(windows))]
        syntropy_config::Runtime::NamedPipes => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "named pipes are a Windows runtime location",
        )),
    }
}

/// A loopback listener from `api.bind` left out because the host lacks its address family.
#[derive(Debug)]
pub struct Skipped {
    /// The address from `api.bind`.
    pub addr: SocketAddr,
    /// The bind error: address not available, or address family not supported.
    pub error: io::Error,
}

impl fmt::Display for Skipped {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "`api.bind` listener `{}` is not served: this host lacks its address family ({}); \
             the other listeners serve",
            self.addr, self.error
        )
    }
}

/// `EAFNOSUPPORT`, the OS's "address family not supported", which has no `io::ErrorKind` of its
/// own: 97 on Linux, 47 on Apple platforms, `WSAEAFNOSUPPORT` (10047) on Windows.
#[cfg(target_os = "linux")]
const EAFNOSUPPORT: Option<i32> = Some(97);
#[cfg(target_vendor = "apple")]
const EAFNOSUPPORT: Option<i32> = Some(47);
#[cfg(windows)]
const EAFNOSUPPORT: Option<i32> = Some(10047);
#[cfg(not(any(target_os = "linux", target_vendor = "apple", windows)))]
const EAFNOSUPPORT: Option<i32> = None;

/// Whether a bind failed because the host lacks the address's family or address.
fn family_unavailable(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::AddrNotAvailable
        || (error.raw_os_error().is_some() && error.raw_os_error() == EAFNOSUPPORT)
}

/// The listeners that bound, each with its plan, and those skipped.
type Bound<T> = (Vec<(T, Listener)>, Vec<Skipped>);

/// Binds each planned listener with `bind`. A loopback listener whose family is unavailable is
/// skipped while another binds; any other failure, or no listener bound, fails the start.
fn bind_listeners<T>(
    plan: Vec<Listener>,
    mut bind: impl FnMut(SocketAddr) -> io::Result<T>,
) -> Result<Bound<T>, ServeError> {
    let mut bound = Vec::with_capacity(plan.len());
    let mut skipped = Vec::new();
    for listener in plan {
        let addr = listener.addr;
        match bind(addr) {
            Ok(socket) => bound.push((socket, listener)),
            Err(error) if listener.class == BindClass::Loopback && family_unavailable(&error) => {
                skipped.push(Skipped { addr, error });
            }
            Err(error) => return Err(ServeError::Bind { addr, error }),
        }
    }
    if bound.is_empty() {
        if let Some(Skipped { addr, error }) = skipped.into_iter().next() {
            return Err(ServeError::Bind { addr, error });
        }
        return Ok((bound, Vec::new()));
    }
    Ok((bound, skipped))
}

/// The API's bound listeners, ready to serve.
#[derive(Debug)]
pub struct Server {
    tcp: Vec<(TcpListener, Listener)>,
    skipped: Vec<Skipped>,
    local: Option<AsyncListener>,
    state: Arc<ApiState>,
}

impl Server {
    /// Binds the listeners `api` asks for, and the local socket at `local` when it is given
    /// (the daemon passes [`local_endpoint`] when `api.local_socket` is on). `has_api_key` says
    /// whether at least one API key exists.
    ///
    /// # Errors
    ///
    /// [`ServeError`]: the bind policy's refusal, a listener this build does not serve yet, or a
    /// failure to bind. A loopback listener whose address family the host lacks is not a failure
    /// while another listener binds: it is in [`Server::skipped`].
    ///
    /// # Panics
    ///
    /// Outside a tokio runtime with I/O enabled.
    pub fn bind(
        api: &ApiSection,
        local: Option<&Endpoint>,
        has_api_key: bool,
        state: ApiState,
    ) -> Result<Server, ServeError> {
        let plan = plan_binds(api, has_api_key).map_err(ServeError::Refused)?;
        if let Some(listener) = plan
            .iter()
            .find(|l| l.class != BindClass::Loopback || l.transport != Transport::Plain)
        {
            return Err(ServeError::NotServedYet {
                listener: listener.clone(),
            });
        }
        let (tcp, skipped) = bind_listeners(plan, |addr| {
            let socket = std::net::TcpListener::bind(addr)?;
            socket.set_nonblocking(true)?;
            TcpListener::from_std(socket)
        })?;
        let local = local
            .map(|endpoint| {
                AsyncListener::bind(endpoint).map_err(|error| ServeError::LocalSocket {
                    endpoint: endpoint.clone(),
                    error,
                })
            })
            .transpose()?;
        Ok(Server {
            tcp,
            skipped,
            local,
            state: Arc::new(state),
        })
    }

    /// The bound TCP addresses, in `api.bind`'s order. A port of 0 in `api.bind` shows here as
    /// the port the OS chose.
    pub fn local_addrs(&self) -> Vec<SocketAddr> {
        self.tcp
            .iter()
            .filter_map(|(socket, _)| socket.local_addr().ok())
            .collect()
    }

    /// The loopback listeners left out because the host lacks their address family. The daemon
    /// reports each as a warning at start.
    pub fn skipped(&self) -> &[Skipped] {
        &self.skipped
    }

    /// The local socket's endpoint, when it is served.
    pub fn local_socket(&self) -> Option<&Endpoint> {
        self.local.as_ref().map(AsyncListener::endpoint)
    }

    /// Serves every listener until `shutdown` completes, then stops accepting and closes every
    /// connection.
    pub async fn serve(self, shutdown: impl Future<Output = ()>) {
        let mut tasks = JoinSet::new();
        for (socket, listener) in self.tcp {
            tasks.spawn(serve_tcp(socket, listener.class, self.state.clone()));
        }
        if let Some(local) = self.local {
            tasks.spawn(serve_local(local, self.state.clone()));
        }
        shutdown.await;
        tasks.shutdown().await;
    }
}

/// How long an HTTP/1 client has to send a request's headers before the connection is closed,
/// a chosen budget \[target\]: long for a local client, and short enough that an idle or stalled
/// connection does not hold its task. HTTP/2 sets no such limit here.
pub const HEADER_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// How long an accept loop waits after an error that is not one connection's, such as running
/// out of file descriptors, before it tries again [target].
const ACCEPT_BACKOFF: Duration = Duration::from_millis(50);

/// Whether an accept error belongs to one connection only, so the next accept may go ahead.
fn per_connection(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::ConnectionAborted
            | io::ErrorKind::ConnectionReset
            | io::ErrorKind::Interrupted
            | io::ErrorKind::WouldBlock
    )
}

/// Accepts one TCP connection and sets `TCP_NODELAY` on it (RFC-0050 §1.1).
async fn accept_tcp(socket: &TcpListener) -> io::Result<(TcpStream, SocketAddr)> {
    let (stream, remote) = socket.accept().await?;
    stream.set_nodelay(true)?;
    Ok((stream, remote))
}

async fn serve_tcp(socket: TcpListener, class: BindClass, state: Arc<ApiState>) {
    let mut connections = JoinSet::new();
    let Ok(local) = socket.local_addr() else {
        return;
    };
    loop {
        // Finished connections leave the set as the loop goes.
        while connections.try_join_next().is_some() {}
        match accept_tcp(&socket).await {
            Ok((stream, remote)) => {
                let via = Via::Tcp {
                    local,
                    remote,
                    class,
                };
                connections.spawn(serve_connection(stream, via, state.clone()));
            }
            Err(error) if per_connection(&error) => {}
            Err(_) => tokio::time::sleep(ACCEPT_BACKOFF).await,
        }
    }
}

async fn serve_local(mut socket: AsyncListener, state: Arc<ApiState>) {
    let mut connections = JoinSet::new();
    loop {
        while connections.try_join_next().is_some() {}
        match socket.accept().await {
            Ok((stream, peer)) => {
                connections.spawn(serve_connection(
                    stream,
                    Via::LocalSocket(peer),
                    state.clone(),
                ));
            }
            // Another user's connection, already closed.
            Err(AcceptError::Refused(_)) => {}
            Err(AcceptError::Io(error)) if per_connection(&error) => {}
            Err(AcceptError::Io(_)) => tokio::time::sleep(ACCEPT_BACKOFF).await,
        }
    }
}

/// Serves one connection: HTTP/1.1, or HTTP/2 when it opens with the HTTP/2 preface.
///
/// Cleartext HTTP/2 is allowed on loopback and the local socket only (RFC-0050 §1.1). Those are the
/// only listeners served today. When plain listeners beyond loopback arrive
/// (`api.allow_plaintext_lan`, PROD-E5-T01), pass the listener's class here and serve those with
/// `http1_only()`.
async fn serve_connection<S>(stream: S, via: Via, state: Arc<ApiState>)
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let service = service_fn(move |mut request: Request<Incoming>| {
        let state = state.clone();
        request.extensions_mut().insert(via.clone());
        async move { Ok::<_, Infallible>(dispatch(&state, request).await) }
    });
    let mut builder = auto::Builder::new(TokioExecutor::new());
    builder
        .http1()
        .timer(TokioTimer::new())
        .header_read_timeout(HEADER_READ_TIMEOUT);
    builder.http2().timer(TokioTimer::new());
    // A connection that fails, such as a client that leaves mid-request, ends here.
    let _ = builder
        .serve_connection(TokioIo::new(stream), service)
        .await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_accepted_tcp_socket_has_nodelay() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .build()
            .unwrap();
        rt.block_on(async {
            let socket = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = socket.local_addr().unwrap();
            let client = tokio::spawn(TcpStream::connect(addr));
            let (stream, remote) = accept_tcp(&socket).await.unwrap();
            let client = client.await.unwrap().unwrap();
            assert!(stream.nodelay().unwrap());
            assert_eq!(remote, client.local_addr().unwrap());
        });
    }

    fn plan(addrs: &[&str]) -> Vec<Listener> {
        addrs
            .iter()
            .map(|addr| Listener {
                addr: addr.parse().unwrap(),
                class: BindClass::Loopback,
                transport: Transport::Plain,
            })
            .collect()
    }

    fn unavailable() -> io::Error {
        io::ErrorKind::AddrNotAvailable.into()
    }

    #[test]
    fn a_loopback_family_the_host_lacks_is_skipped_while_another_binds() {
        let v6: SocketAddr = "[::1]:7433".parse().unwrap();
        let (bound, skipped) = bind_listeners(plan(&["127.0.0.1:7433", "[::1]:7433"]), |addr| {
            if addr == v6 {
                Err(unavailable())
            } else {
                Ok(addr)
            }
        })
        .unwrap();
        assert_eq!(bound.len(), 1);
        assert_eq!(bound[0].0, "127.0.0.1:7433".parse::<SocketAddr>().unwrap());
        assert_eq!(skipped.len(), 1);
        assert_eq!(skipped[0].addr, v6);
        assert!(skipped[0].to_string().contains("lacks its address family"));
        // The family-not-supported code counts too.
        if let Some(code) = EAFNOSUPPORT {
            let (bound, skipped) =
                bind_listeners(plan(&["[::1]:7433", "127.0.0.1:7433"]), |addr| {
                    if addr == v6 {
                        Err(io::Error::from_raw_os_error(code))
                    } else {
                        Ok(addr)
                    }
                })
                .unwrap();
            assert_eq!((bound.len(), skipped.len()), (1, 1));
        }
    }

    #[test]
    fn no_listener_bound_or_another_error_fails_the_start() {
        // Every family missing: the first one's error.
        let error =
            bind_listeners(plan(&["[::1]:7433"]), |_| Err::<(), _>(unavailable())).unwrap_err();
        assert!(
            matches!(&error, ServeError::Bind { addr, .. } if addr.port() == 7433),
            "{error}"
        );
        // A taken port is no missing family, even on `[::1]`.
        let error = bind_listeners(plan(&["127.0.0.1:7433", "[::1]:7433"]), |addr| {
            if addr.is_ipv6() {
                Err(io::ErrorKind::AddrInUse.into())
            } else {
                Ok(())
            }
        })
        .unwrap_err();
        assert!(matches!(error, ServeError::Bind { .. }), "{error}");
        // No `api.bind` listener at all is not a failure here: the local socket may serve alone.
        let (bound, skipped) = bind_listeners(Vec::new(), |_| Ok(())).unwrap();
        assert!(bound.is_empty() && skipped.is_empty());
    }

    #[test]
    fn per_connection_errors_are_told_apart() {
        assert!(per_connection(&io::ErrorKind::ConnectionAborted.into()));
        assert!(!per_connection(&io::Error::other("too many open files")));
    }
}
