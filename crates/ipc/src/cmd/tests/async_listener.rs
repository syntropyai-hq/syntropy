// SPDX-License-Identifier: MIT OR Apache-2.0

//! The async listener, the API's local socket on tokio (PROD-E2-T01; RFC-0050 §1.1).
//!
//! - A connection of this user is accepted, and the peer the OS reported comes with it.
//! - On Linux and Apple platforms the directory has mode 0700 and the socket 0600, and the socket
//!   goes with the listener. A stale socket, as a killed daemon leaves, is taken over; a live one,
//!   a regular file or a link is not.
//! - On Windows the pipe name must be free: no second listener, async or not, can take it.
//!
//! A peer of another user is refused in `other_user`, beside the blocking listener's case.

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::super::{AsyncListener, Endpoint, Peer, connect_async};
use super::{endpoint, remove_dir};

/// A runtime like the one the API serves on, with one thread.
fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .enable_time()
        .build()
        .unwrap()
}

/// Checks `peer` is this process: its user, and its process where the OS reports one.
fn assert_is_this_process(peer: &Peer) {
    #[cfg(any(target_os = "linux", target_vendor = "apple"))]
    {
        let Peer::Unix { uid, pid } = peer else {
            panic!("a Windows peer on a Unix socket: {peer:?}")
        };
        // SAFETY: no arguments, and it cannot fail.
        assert_eq!(*uid, unsafe { libc::geteuid() });
        if cfg!(target_os = "linux") {
            assert_eq!(*pid, Some(std::process::id()));
        }
    }
    #[cfg(windows)]
    {
        let Peer::Windows { pid, sid } = peer else {
            panic!("a Unix peer on a named pipe: {peer:?}")
        };
        let ours = crate::shm::UserSid::current().unwrap().text().unwrap();
        assert_eq!(sid.as_deref(), Some(ours.as_str()));
        assert_eq!(*pid, std::process::id());
    }
}

#[test]
fn a_connection_of_this_user_is_accepted_with_its_peer() {
    runtime().block_on(async {
        let endpoint = endpoint();
        let mut listener = AsyncListener::bind(&endpoint).expect("bind");
        assert_eq!(listener.endpoint(), &endpoint);
        let client = tokio::spawn({
            let endpoint = endpoint.clone();
            async move {
                let mut stream = connect_async(&endpoint).await.expect("connect");
                stream.write_all(b"ping").await.unwrap();
                let mut answer = [0u8; 4];
                stream.read_exact(&mut answer).await.unwrap();
                answer
            }
        });
        let (mut stream, peer) = listener.accept().await.expect("accept");
        assert_is_this_process(&peer);
        let mut request = [0u8; 4];
        stream.read_exact(&mut request).await.unwrap();
        assert_eq!(&request, b"ping");
        stream.write_all(b"pong").await.unwrap();
        assert_eq!(&client.await.unwrap(), b"pong");
        drop(listener);
        remove_dir(&endpoint);
    });
}

#[cfg(any(target_os = "linux", target_vendor = "apple"))]
#[test]
fn the_directory_is_0700_the_socket_0600_and_both_go_with_the_listener() {
    use std::os::unix::fs::MetadataExt;

    let rt = runtime();
    let _context = rt.enter();
    let endpoint = endpoint();
    let Endpoint::Socket(path) = &endpoint else {
        unreachable!()
    };
    let listener = AsyncListener::bind(&endpoint).expect("bind");
    let dir = std::fs::symlink_metadata(path.parent().unwrap()).unwrap();
    assert_eq!(dir.mode() & 0o777, 0o700);
    let socket = std::fs::symlink_metadata(path).unwrap();
    assert_eq!(socket.mode() & 0o777, 0o600);
    // The name is taken while the listener lives.
    assert!(AsyncListener::bind(&endpoint).is_err());
    drop(listener);
    assert!(!path.exists());
    remove_dir(&endpoint);
}

#[cfg(any(target_os = "linux", target_vendor = "apple"))]
#[test]
fn bind_takes_over_a_stale_socket_and_nothing_else() {
    let rt = runtime();
    let _context = rt.enter();
    super::unix::stale_socket_cases(&|endpoint| {
        AsyncListener::bind(endpoint).map(|l| Box::new(l) as Box<dyn std::any::Any>)
    });
}

#[cfg(windows)]
#[test]
fn a_pipe_name_is_taken_by_one_listener_only() {
    let rt = runtime();
    let _context = rt.enter();
    let endpoint = endpoint();
    let first = AsyncListener::bind(&endpoint).expect("bind");
    // `FILE_FLAG_FIRST_PIPE_INSTANCE`: neither kind of listener can take the name.
    assert!(AsyncListener::bind(&endpoint).is_err());
    assert!(super::super::Listener::bind(&endpoint).is_err());
    drop(first);
    // A blocking listener that holds the name keeps the async one out too.
    let _blocking = super::super::Listener::bind(&endpoint).expect("bind");
    assert!(AsyncListener::bind(&endpoint).is_err());
}

#[cfg(windows)]
#[test]
fn the_api_pipe_names_this_user() {
    let ours = crate::shm::UserSid::current().unwrap().text().unwrap();
    assert_eq!(
        super::super::user_pipe("api").unwrap(),
        Endpoint::Pipe(format!("syntropy-{ours}-api"))
    );
}
