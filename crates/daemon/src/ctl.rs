// SPDX-License-Identifier: MIT OR Apache-2.0

//! The daemon's `ctl` runtime and the API server on it (RFC-0001 §8.1; RFC-0050 §1.1).
//!
//! `ctl` is a multi-thread tokio runtime of `daemon.ctl_threads` workers. It runs the API server,
//! and later the cluster, the supervisor's IPC and the ingress sequence channels; bulk transfers
//! run on `bulk`, so a download cannot starve it. [`start_api`] binds the API's listeners on it
//! and serves them until [`Api::stop`].

use std::io;
use std::net::SocketAddr;

use syntropy_api::config::ApiSection;
use syntropy_api::routes::ApiState;
use syntropy_api::server::{ServeError, Server, local_endpoint};
use syntropy_config::Runtime as RuntimeLocation;
use syntropy_ipc::cmd::Endpoint;
use tokio::runtime::Runtime;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

/// `daemon.ctl_threads`' default for a host with `physical_cores` cores: `clamp(cores / 8, 2, 4)`
/// (RFC-0054 §5.2).
#[must_use]
pub fn default_threads(physical_cores: usize) -> usize {
    (physical_cores / 8).clamp(2, 4)
}

/// Builds the `ctl` runtime with `threads` workers, each named `syntropy-ctl`, with I/O and timers.
///
/// # Errors
///
/// When the OS refuses the threads or the I/O driver.
pub fn runtime(threads: usize) -> io::Result<Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(threads.max(1))
        .thread_name("syntropy-ctl")
        .enable_io()
        .enable_time()
        .build()
}

/// Why the API did not start.
#[derive(Debug)]
pub enum ApiStartError {
    /// The local socket's endpoint could not be named, such as a Windows user SID that cannot
    /// be read.
    LocalEndpoint(io::Error),
    /// The server refused to start, or could not bind.
    Serve(ServeError),
}

impl std::fmt::Display for ApiStartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiStartError::LocalEndpoint(error) => {
                write!(f, "cannot name the API's local socket: {error}")
            }
            ApiStartError::Serve(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for ApiStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ApiStartError::LocalEndpoint(error) => Some(error),
            ApiStartError::Serve(error) => Some(error),
        }
    }
}

/// The API, serving on the `ctl` runtime.
#[derive(Debug)]
pub struct Api {
    addrs: Vec<SocketAddr>,
    local: Option<Endpoint>,
    stop: oneshot::Sender<()>,
    task: JoinHandle<()>,
}

impl Api {
    /// The TCP addresses it listens on, in `api.bind`'s order.
    pub fn addrs(&self) -> &[SocketAddr] {
        &self.addrs
    }

    /// The local socket's endpoint, when `api.local_socket` is on.
    pub fn local_socket(&self) -> Option<&Endpoint> {
        self.local.as_ref()
    }

    /// Stops accepting, closes every connection and waits for the server to finish.
    pub async fn stop(self) {
        let _ = self.stop.send(());
        let _ = self.task.await;
    }
}

/// Binds the API's listeners on `ctl` and starts serving them: every `api.bind` address, and the
/// local socket in `runtime_dir` when `api.local_socket` is on. `has_api_key` says whether at
/// least one API key exists (RFC-0050 §11.1).
///
/// # Errors
///
/// [`ApiStartError`]: the server's refusal or bind failure, or a local socket that cannot be
/// named. A local socket needs a runtime location; with none (`runtime_dir` is `None`) the API
/// serves without it.
pub fn start_api(
    ctl: &Runtime,
    api: &ApiSection,
    runtime_dir: Option<&RuntimeLocation>,
    has_api_key: bool,
    state: ApiState,
) -> Result<Api, ApiStartError> {
    let local = match (api.local_socket, runtime_dir) {
        (true, Some(location)) => {
            Some(local_endpoint(location).map_err(ApiStartError::LocalEndpoint)?)
        }
        _ => None,
    };
    let server = {
        let _context = ctl.enter();
        Server::bind(api, local.as_ref(), has_api_key, state).map_err(ApiStartError::Serve)?
    };
    let addrs = server.local_addrs();
    let (stop, stopped) = oneshot::channel();
    let task = ctl.spawn(server.serve(async {
        let _ = stopped.await;
    }));
    Ok(Api {
        addrs,
        local,
        stop,
        task,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_thread_count_follows_rfc_0054() {
        assert_eq!(default_threads(1), 2);
        assert_eq!(default_threads(16), 2);
        assert_eq!(default_threads(24), 3);
        assert_eq!(default_threads(32), 4);
        assert_eq!(default_threads(128), 4);
    }
}
