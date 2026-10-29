// SPDX-License-Identifier: MIT OR Apache-2.0

//! The channel's endpoint on each OS, and the peer check (ADR-0015 item 4; RFC-0034 §6.4).

use std::fmt;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::str::FromStr;

/// Where the daemon listens and a client connects.
///
/// Its text form, `socket:<path>` or `pipe:<name>`, can go on a command line. A socket path that
/// is not UTF-8 has no text form that parses back.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Endpoint {
    /// A Unix-domain socket at this path, on Linux and Apple platforms. Its directory is the
    /// user's alone: [`Listener::bind`] creates it with mode 0700, or checks that it has that mode
    /// and owner.
    Socket(PathBuf),
    /// The named pipe `\\.\pipe\<name>`, on Windows. The name holds no backslash.
    Pipe(String),
}

impl fmt::Display for Endpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Endpoint::Socket(path) => write!(f, "socket:{}", path.display()),
            Endpoint::Pipe(name) => write!(f, "pipe:{name}"),
        }
    }
}

/// Why an endpoint's text form did not parse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EndpointParseError(String);

impl fmt::Display for EndpointParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "`{}` is not an IPC endpoint: expected socket:<path> or pipe:<name>",
            self.0
        )
    }
}

impl std::error::Error for EndpointParseError {}

impl FromStr for Endpoint {
    type Err = EndpointParseError;

    fn from_str(text: &str) -> Result<Endpoint, EndpointParseError> {
        let error = || EndpointParseError(text.to_owned());
        if let Some(path) = text.strip_prefix("socket:") {
            if path.is_empty() {
                return Err(error());
            }
            Ok(Endpoint::Socket(PathBuf::from(path)))
        } else if let Some(name) = text.strip_prefix("pipe:") {
            if name.is_empty() || name.contains('\\') {
                return Err(error());
            }
            Ok(Endpoint::Pipe(name.to_owned()))
        } else {
            Err(error())
        }
    }
}

/// Who is at the other end of a connection, as the OS reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Peer {
    /// Linux and Apple platforms: the peer's effective user ID, and on Linux its process ID.
    Unix {
        /// The user ID.
        uid: u32,
        /// The process ID, where the OS gives it.
        pid: Option<u32>,
    },
    /// Windows: the client's process ID, and its token's user SID (`S-1-5-…`) when it could be
    /// read.
    Windows {
        /// The process ID.
        pid: u32,
        /// The user SID.
        sid: Option<String>,
    },
}

impl fmt::Display for Peer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Peer::Unix {
                uid,
                pid: Some(pid),
            } => write!(f, "user {uid}, process {pid}"),
            Peer::Unix { uid, pid: None } => write!(f, "user {uid}"),
            Peer::Windows {
                pid,
                sid: Some(sid),
            } => write!(f, "user {sid}, process {pid}"),
            Peer::Windows { pid, sid: None } => {
                write!(f, "process {pid}, whose user could not be read")
            }
        }
    }
}

/// Why no connection was accepted.
#[derive(Debug)]
pub enum AcceptError {
    /// An OS call failed. A connection that was open is closed.
    Io(io::Error),
    /// The peer is another user, or its user could not be read. Its connection is closed.
    Refused(Peer),
}

impl fmt::Display for AcceptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AcceptError::Io(error) => write!(f, "accepting an IPC connection failed: {error}"),
            AcceptError::Refused(peer) => write!(
                f,
                "refused an IPC connection from {peer}: only this daemon's user may connect"
            ),
        }
    }
}

impl std::error::Error for AcceptError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AcceptError::Io(error) => Some(error),
            AcceptError::Refused(_) => None,
        }
    }
}

/// A connected, checked byte stream: a Unix-domain socket, or a named pipe on Windows.
#[derive(Debug)]
pub struct Stream(imp::Stream);

impl Stream {
    /// Wraps the OS's stream: a Unix-domain socket, or a pipe's handle on Windows.
    #[cfg(any(target_os = "linux", target_vendor = "apple"))]
    pub(super) fn from_os(stream: std::os::unix::net::UnixStream) -> Stream {
        Stream(stream)
    }

    /// Wraps the OS's stream: a Unix-domain socket, or a pipe's handle on Windows.
    #[cfg(windows)]
    pub(super) fn from_os(stream: std::fs::File) -> Stream {
        Stream(stream)
    }

    /// The pipe's handle.
    #[cfg(windows)]
    pub(super) fn into_os(self) -> std::fs::File {
        self.0
    }
}

impl Read for Stream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }
}

impl Write for Stream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
}

/// The daemon's listening endpoint. Every connection it accepts has passed the peer check.
#[derive(Debug)]
pub struct Listener {
    inner: imp::Listener,
    endpoint: Endpoint,
}

impl Listener {
    /// Listens on `endpoint`.
    ///
    /// A socket's directory is created with mode 0700, or, if it exists, must be a directory (not
    /// a link) owned by this user with no permissions for group or others. A socket file already
    /// at the path is taken over only when it is stale: a socket, not a link, owned by this user,
    /// that no process accepts on, as a daemon killed without its cleanup leaves. A live socket
    /// keeps the bind failing with `AddrInUse`. A pipe is created with a protected DACL that grants this user alone, and
    /// must be the pipe's first instance, so no other process holds the name.
    pub fn bind(endpoint: &Endpoint) -> io::Result<Listener> {
        Ok(Listener {
            inner: imp::Listener::bind(endpoint)?,
            endpoint: endpoint.clone(),
        })
    }

    /// The endpoint it listens on.
    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    /// Waits for the next connection, and checks its peer is this process's user.
    pub fn accept(&mut self) -> Result<Stream, AcceptError> {
        self.inner.accept().map(Stream)
    }

    /// A listener on a socket whose protection the caller set up, so a test can reach the peer
    /// check from another user's process.
    #[cfg(all(test, any(target_os = "linux", target_vendor = "apple")))]
    pub(super) fn from_std(socket: std::os::unix::net::UnixListener, path: PathBuf) -> Listener {
        Listener {
            inner: imp::Listener::from_std(socket, path.clone()),
            endpoint: Endpoint::Socket(path),
        }
    }

    /// A listener on a pipe whose DACL also lets every authenticated user open it, so a test
    /// can reach the peer check from another user's process.
    #[cfg(all(test, windows))]
    pub(super) fn open_to_authenticated_users(endpoint: &Endpoint) -> io::Result<Listener> {
        Ok(Listener {
            inner: imp::Listener::open_to_authenticated_users(endpoint)?,
            endpoint: endpoint.clone(),
        })
    }
}

/// Connects to the daemon at `endpoint`.
pub fn connect(endpoint: &Endpoint) -> io::Result<Stream> {
    imp::connect(endpoint).map(Stream)
}

/// The async endpoint and stream, on tokio, with the `async` feature. Not under loom: the loom
/// suite builds this crate with `--cfg loom`, which tokio does not build under.
#[cfg(all(feature = "async", not(loom)))]
mod aio {
    use std::io;
    #[cfg(all(test, any(target_os = "linux", target_vendor = "apple")))]
    use std::path::PathBuf;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

    use super::imp::aio as imp;
    use super::{AcceptError, Endpoint, Peer};

    /// The daemon's listening endpoint for async servers on tokio, such as the API's local socket
    /// (RFC-0050 §1.1). The endpoint is protected as [`Listener`](super::Listener)'s is, and a
    /// Unix socket is also given mode 0600. Every connection it accepts has passed the same peer
    /// check.
    #[derive(Debug)]
    pub struct AsyncListener {
        inner: imp::AsyncListener,
        endpoint: Endpoint,
    }

    impl AsyncListener {
        /// Listens on `endpoint`, with [`Listener::bind`](super::Listener::bind)'s rules for the
        /// directory, the socket and the pipe's first instance.
        ///
        /// # Panics
        ///
        /// Outside a tokio runtime with I/O enabled.
        pub fn bind(endpoint: &Endpoint) -> io::Result<AsyncListener> {
            Ok(AsyncListener {
                inner: imp::AsyncListener::bind(endpoint)?,
                endpoint: endpoint.clone(),
            })
        }

        /// The endpoint it listens on.
        pub fn endpoint(&self) -> &Endpoint {
            &self.endpoint
        }

        /// Waits for the next connection, checks its peer is this process's user, and returns it
        /// with the peer the OS reported. A refused peer's connection is closed.
        pub async fn accept(&mut self) -> Result<(AsyncStream, Peer), AcceptError> {
            self.inner
                .accept()
                .await
                .map(|(stream, peer)| (AsyncStream(stream), peer))
        }

        /// An async listener on a socket whose protection the caller set up, so a test can reach
        /// the peer check from another user's process.
        #[cfg(all(test, any(target_os = "linux", target_vendor = "apple")))]
        pub(in crate::cmd) fn from_std(
            socket: std::os::unix::net::UnixListener,
            path: PathBuf,
        ) -> io::Result<AsyncListener> {
            Ok(AsyncListener {
                inner: imp::AsyncListener::from_std(socket, path.clone())?,
                endpoint: Endpoint::Socket(path),
            })
        }

        /// An async listener on a pipe whose DACL also lets every authenticated user open it, so a
        /// test can reach the peer check from another user's process.
        #[cfg(all(test, windows))]
        pub(in crate::cmd) fn open_to_authenticated_users(
            endpoint: &Endpoint,
        ) -> io::Result<AsyncListener> {
            Ok(AsyncListener {
                inner: imp::AsyncListener::open_to_authenticated_users(endpoint)?,
                endpoint: endpoint.clone(),
            })
        }
    }

    /// A connected byte stream for tokio: a Unix-domain socket, or a named pipe on Windows. A
    /// stream from [`AsyncListener::accept`] has passed the peer check.
    #[derive(Debug)]
    pub struct AsyncStream(imp::AsyncStream);

    /// Connects to the daemon at `endpoint`, for a client on tokio.
    ///
    /// # Panics
    ///
    /// Outside a tokio runtime with I/O enabled.
    pub async fn connect_async(endpoint: &Endpoint) -> io::Result<AsyncStream> {
        imp::connect_async(endpoint).await.map(AsyncStream)
    }

    /// The named pipe of this user's endpoint for `role`, `syntropy-<user SID>-<role>`, as
    /// RFC-0054 §1 names the runtime endpoints on Windows: `api` is the API's local socket
    /// (RFC-0050 §1.1).
    #[cfg(windows)]
    pub fn user_pipe(role: &str) -> io::Result<Endpoint> {
        imp::user_pipe_name(role).map(Endpoint::Pipe)
    }

    impl AsyncRead for AsyncStream {
        fn poll_read(
            self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &mut ReadBuf<'_>,
        ) -> Poll<io::Result<()>> {
            Pin::new(&mut self.get_mut().0).poll_read(cx, buf)
        }
    }

    impl AsyncWrite for AsyncStream {
        fn poll_write(
            self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<io::Result<usize>> {
            Pin::new(&mut self.get_mut().0).poll_write(cx, buf)
        }

        fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Pin::new(&mut self.get_mut().0).poll_flush(cx)
        }

        fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Pin::new(&mut self.get_mut().0).poll_shutdown(cx)
        }
    }
}

#[cfg(all(windows, feature = "async", not(loom)))]
pub use aio::user_pipe;
#[cfg(all(feature = "async", not(loom)))]
pub use aio::{AsyncListener, AsyncStream, connect_async};

/// The error for an endpoint of another OS's kind.
fn unsupported(endpoint: &Endpoint) -> io::Error {
    io::Error::new(
        io::ErrorKind::Unsupported,
        format!("{endpoint}: this OS has no endpoint of this kind"),
    )
}

#[cfg(any(target_os = "linux", target_vendor = "apple"))]
mod imp {
    use std::fs;
    use std::io;
    use std::os::fd::{AsFd, BorrowedFd};
    use std::os::unix::fs::{DirBuilderExt, FileTypeExt, MetadataExt};
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::path::{Path, PathBuf};

    use super::{AcceptError, Endpoint, Peer, unsupported};

    pub(super) type Stream = UnixStream;

    #[derive(Debug)]
    pub(super) struct Listener {
        socket: UnixListener,
        path: PathBuf,
    }

    /// This process's effective user ID.
    fn euid() -> u32 {
        // SAFETY: no arguments, and it cannot fail.
        unsafe { libc::geteuid() }
    }

    /// Creates `dir` with mode 0700, or checks that an existing `dir` is a directory, not a link,
    /// owned by this user, with no permissions for group or others.
    fn owner_only_dir(dir: &Path) -> io::Result<()> {
        match fs::DirBuilder::new().mode(0o700).create(dir) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
        let meta = fs::symlink_metadata(dir)?;
        let uid = euid();
        if !meta.is_dir() || meta.uid() != uid || meta.mode() & 0o077 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!(
                    "{}: an IPC socket's directory must be a directory owned by user {uid} with \
                     mode 0700, and this is {} with mode {:o}, owned by user {}",
                    dir.display(),
                    if meta.is_dir() {
                        "a directory"
                    } else {
                        "not a directory"
                    },
                    meta.mode() & 0o7777,
                    meta.uid()
                ),
            ));
        }
        Ok(())
    }

    /// The socket path of `endpoint`, after making its directory owner-only.
    fn prepare(endpoint: &Endpoint) -> io::Result<&Path> {
        let Endpoint::Socket(path) = endpoint else {
            return Err(unsupported(endpoint));
        };
        let dir = path
            .parent()
            .filter(|dir| !dir.as_os_str().is_empty())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("{endpoint}: the socket path has no directory"),
                )
            })?;
        owner_only_dir(dir)?;
        Ok(path)
    }

    /// Binds a socket at `path`, in a directory [`prepare`] made owner-only.
    ///
    /// A daemon killed without its cleanup (SIGKILL, the OOM killer, a crash) leaves its socket
    /// file behind, and binding over it fails with `AddrInUse`. So a socket file at `path` that
    /// refuses connections is stale, and is removed before a second bind. It is removed only when
    /// it is a socket owned by this user, read without following a link. A socket that accepts the
    /// connection belongs to a live listener, and the bind fails. The directory admits this user
    /// alone, so no other user can race the check.
    fn bind_socket(path: &Path) -> io::Result<UnixListener> {
        let in_use = match UnixListener::bind(path) {
            Err(error) if error.kind() == io::ErrorKind::AddrInUse => error,
            other => return other,
        };
        match UnixStream::connect(path) {
            Err(error) if error.kind() == io::ErrorKind::ConnectionRefused => {}
            // A live listener, or a path that cannot be checked: the bind stays refused.
            _ => return Err(in_use),
        }
        let meta = fs::symlink_metadata(path)?;
        if !meta.file_type().is_socket() || meta.uid() != euid() {
            return Err(io::Error::new(
                io::ErrorKind::AddrInUse,
                format!(
                    "{}: the path is taken by something other than a stale socket of user {}, so \
                     it is left in place",
                    path.display(),
                    euid()
                ),
            ));
        }
        fs::remove_file(path)?;
        UnixListener::bind(path)
    }

    impl Listener {
        pub(super) fn bind(endpoint: &Endpoint) -> io::Result<Listener> {
            let path = prepare(endpoint)?;
            Ok(Listener {
                socket: bind_socket(path)?,
                path: path.to_path_buf(),
            })
        }

        #[cfg(test)]
        pub(super) fn from_std(socket: UnixListener, path: PathBuf) -> Listener {
            Listener { socket, path }
        }

        pub(super) fn accept(&self) -> Result<UnixStream, AcceptError> {
            let (stream, _) = self.socket.accept().map_err(AcceptError::Io)?;
            check(stream.as_fd()).map(|_| stream)
        }
    }

    /// Reads the peer's credentials on `fd`, a connected socket, and checks it is this user.
    fn check(fd: BorrowedFd<'_>) -> Result<Peer, AcceptError> {
        let peer = peer(fd).map_err(AcceptError::Io)?;
        match peer {
            Peer::Unix { uid, .. } if uid == euid() => Ok(peer),
            _ => Err(AcceptError::Refused(peer)),
        }
    }

    impl Drop for Listener {
        fn drop(&mut self) {
            // Best effort: the name goes with the listener. A failure leaves a dead socket, which
            // the next bind reports.
            let _ = fs::remove_file(&self.path);
        }
    }

    /// The peer's credentials: `SO_PEERCRED`.
    #[cfg(target_os = "linux")]
    fn peer(fd: BorrowedFd<'_>) -> io::Result<Peer> {
        use std::os::fd::AsRawFd;

        let mut cred = libc::ucred {
            pid: 0,
            uid: 0,
            gid: 0,
        };
        let size = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
        let mut len = size;
        // SAFETY: a live socket, and an output buffer of `len` bytes that outlives the call.
        let result = unsafe {
            libc::getsockopt(
                fd.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_PEERCRED,
                (&raw mut cred).cast(),
                &mut len,
            )
        };
        if result == -1 {
            return Err(io::Error::last_os_error());
        }
        if len != size {
            return Err(io::Error::other(format!(
                "SO_PEERCRED returned {len} bytes, not {size}"
            )));
        }
        Ok(Peer::Unix {
            uid: cred.uid,
            pid: u32::try_from(cred.pid).ok().filter(|&pid| pid != 0),
        })
    }

    /// The peer's credentials: `getpeereid`.
    #[cfg(target_vendor = "apple")]
    fn peer(fd: BorrowedFd<'_>) -> io::Result<Peer> {
        use std::os::fd::AsRawFd;

        let mut uid: libc::uid_t = 0;
        let mut gid: libc::gid_t = 0;
        // SAFETY: a live socket and two output pointers that outlive the call.
        if unsafe { libc::getpeereid(fd.as_raw_fd(), &mut uid, &mut gid) } == -1 {
            return Err(io::Error::last_os_error());
        }
        Ok(Peer::Unix { uid, pid: None })
    }

    pub(super) fn connect(endpoint: &Endpoint) -> io::Result<UnixStream> {
        let Endpoint::Socket(path) = endpoint else {
            return Err(unsupported(endpoint));
        };
        UnixStream::connect(path)
    }

    /// The async listener, on tokio, with the `async` feature; not under loom, which tokio does
    /// not build under.
    #[cfg(all(feature = "async", not(loom)))]
    pub(in super::super) mod aio {
        use std::fs;
        use std::io;
        use std::os::fd::AsFd;
        use std::os::unix::fs::PermissionsExt;
        use std::path::PathBuf;

        use super::super::{AcceptError, Endpoint, Peer, unsupported};
        use super::{bind_socket, check, prepare};

        pub(in crate::cmd::os) type AsyncStream = tokio::net::UnixStream;

        #[derive(Debug)]
        pub(in crate::cmd::os) struct AsyncListener {
            socket: tokio::net::UnixListener,
            path: PathBuf,
        }

        impl AsyncListener {
            pub(in crate::cmd::os) fn bind(endpoint: &Endpoint) -> io::Result<AsyncListener> {
                let path = prepare(endpoint)?;
                let socket = bind_socket(path)?;
                // From here the name is ours, so a failure removes it.
                let listen = || {
                    // The socket is owner-only too (RFC-0050 §1.1), though the directory already
                    // keeps every other user out.
                    fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
                    socket.set_nonblocking(true)?;
                    tokio::net::UnixListener::from_std(socket)
                };
                match listen() {
                    Ok(socket) => Ok(AsyncListener {
                        socket,
                        path: path.to_path_buf(),
                    }),
                    Err(error) => {
                        let _ = fs::remove_file(path);
                        Err(error)
                    }
                }
            }

            #[cfg(test)]
            pub(in crate::cmd::os) fn from_std(
                socket: std::os::unix::net::UnixListener,
                path: PathBuf,
            ) -> io::Result<AsyncListener> {
                socket.set_nonblocking(true)?;
                Ok(AsyncListener {
                    socket: tokio::net::UnixListener::from_std(socket)?,
                    path,
                })
            }

            pub(in crate::cmd::os) async fn accept(
                &mut self,
            ) -> Result<(AsyncStream, Peer), AcceptError> {
                let (stream, _) = self.socket.accept().await.map_err(AcceptError::Io)?;
                let peer = check(stream.as_fd())?;
                Ok((stream, peer))
            }
        }

        impl Drop for AsyncListener {
            fn drop(&mut self) {
                // Best effort, as for `Listener`.
                let _ = fs::remove_file(&self.path);
            }
        }

        pub(in crate::cmd::os) async fn connect_async(
            endpoint: &Endpoint,
        ) -> io::Result<AsyncStream> {
            let Endpoint::Socket(path) = endpoint else {
                return Err(unsupported(endpoint));
            };
            tokio::net::UnixStream::connect(path).await
        }
    }
}

#[cfg(windows)]
mod imp {
    use std::fs::File;
    use std::io;
    use std::os::windows::io::{AsHandle, AsRawHandle, BorrowedHandle, FromRawHandle, OwnedHandle};

    use windows_sys::Win32::Foundation::{
        ERROR_PIPE_BUSY, ERROR_PIPE_CONNECTED, GENERIC_READ, GENERIC_WRITE, INVALID_HANDLE_VALUE,
    };
    use windows_sys::Win32::Security::{EqualSid, SECURITY_ATTRIBUTES};
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FILE_FLAG_FIRST_PIPE_INSTANCE, OPEN_EXISTING, PIPE_ACCESS_DUPLEX,
        SECURITY_IDENTIFICATION, SECURITY_SQOS_PRESENT,
    };
    use windows_sys::Win32::System::Pipes::{
        ConnectNamedPipe, CreateNamedPipeW, GetNamedPipeClientProcessId, PIPE_READMODE_BYTE,
        PIPE_REJECT_REMOTE_CLIENTS, PIPE_TYPE_BYTE, PIPE_UNLIMITED_INSTANCES, PIPE_WAIT,
        WaitNamedPipeW,
    };
    use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};

    use super::{AcceptError, Endpoint, Peer, unsupported};
    use crate::shm::{OwnerOnly, UserSid, wide};

    pub(super) type Stream = File;

    /// Each pipe instance's buffers, in bytes, each way.
    const BUFFER: u32 = 64 << 10;

    /// How many times a client waits for a busy pipe before giving up.
    const BUSY_ATTEMPTS: u32 = 8;

    /// How long each wait for a busy pipe lasts, in milliseconds.
    const BUSY_WAIT_MS: u32 = 500;

    pub(super) struct Listener {
        /// `\\.\pipe\<name>`, NUL-terminated.
        path: Vec<u16>,
        /// The instance the next client connects to.
        pending: OwnedHandle,
        /// This process's user, which every peer must be.
        user: UserSid,
        /// The DACL each instance is created with.
        descriptor: OwnerOnly,
    }

    impl std::fmt::Debug for Listener {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Listener")
                .field("path", &String::from_utf16_lossy(&self.path))
                .finish_non_exhaustive()
        }
    }

    fn pipe_path(endpoint: &Endpoint) -> io::Result<Vec<u16>> {
        let Endpoint::Pipe(name) = endpoint else {
            return Err(unsupported(endpoint));
        };
        if name.is_empty() || name.contains('\\') || name.contains('\0') {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{endpoint}: a pipe name is not empty and holds no backslash or NUL"),
            ));
        }
        Ok(wide(&format!(r"\\.\pipe\{name}")))
    }

    /// One instance of the pipe at `path`, owner-only. `first` makes the call fail if any other
    /// instance of the name exists, so no other process can have taken the name first.
    fn instance(path: &[u16], descriptor: &OwnerOnly, first: bool) -> io::Result<OwnedHandle> {
        let attributes = SECURITY_ATTRIBUTES {
            nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: descriptor.descriptor(),
            bInheritHandle: 0,
        };
        let only = if first {
            FILE_FLAG_FIRST_PIPE_INSTANCE
        } else {
            0
        };
        let open_mode = PIPE_ACCESS_DUPLEX | only;
        // SAFETY: a NUL-terminated path, and security attributes whose descriptor lives for the
        // call.
        let handle = unsafe {
            CreateNamedPipeW(
                path.as_ptr(),
                open_mode,
                PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
                PIPE_UNLIMITED_INSTANCES,
                BUFFER,
                BUFFER,
                0,
                &attributes,
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: the handle `CreateNamedPipeW` just returned, owned only here.
        Ok(unsafe { OwnedHandle::from_raw_handle(handle.cast()) })
    }

    impl Listener {
        pub(super) fn bind(endpoint: &Endpoint) -> io::Result<Listener> {
            Listener::with_descriptor(endpoint, OwnerOnly::new()?)
        }

        #[cfg(test)]
        pub(super) fn open_to_authenticated_users(endpoint: &Endpoint) -> io::Result<Listener> {
            Listener::with_descriptor(endpoint, OwnerOnly::open_to_authenticated_users()?)
        }

        /// Listens on `endpoint`, creating each instance with `descriptor`.
        fn with_descriptor(endpoint: &Endpoint, descriptor: OwnerOnly) -> io::Result<Listener> {
            let path = pipe_path(endpoint)?;
            let user = UserSid::current()?;
            let pending = instance(&path, &descriptor, true)?;
            Ok(Listener {
                path,
                pending,
                user,
                descriptor,
            })
        }

        pub(super) fn accept(&mut self) -> Result<File, AcceptError> {
            // SAFETY: a live pipe instance, opened without overlapped I/O, so the call blocks
            // until a client connects.
            let connected =
                unsafe { ConnectNamedPipe(self.pending.as_raw_handle(), std::ptr::null_mut()) };
            // `ERROR_PIPE_CONNECTED`: a client connected between the instance's creation and
            // this call.
            let failed = (connected == 0)
                .then(io::Error::last_os_error)
                .filter(|error| error.raw_os_error() != Some(ERROR_PIPE_CONNECTED as i32));
            // A new instance replaces this one either way: one whose client failed, such as a
            // client that left before the call, cannot take another.
            let next = instance(&self.path, &self.descriptor, false).map_err(AcceptError::Io)?;
            let client = std::mem::replace(&mut self.pending, next);
            if let Some(error) = failed {
                return Err(AcceptError::Io(error));
            }
            check_peer(client.as_handle(), &self.user)?;
            Ok(File::from(client))
        }
    }

    /// Checks the client of `pipe` runs as `user`: its process's token user SID.
    fn check_peer(pipe: BorrowedHandle<'_>, user: &UserSid) -> Result<Peer, AcceptError> {
        let mut pid = 0u32;
        // SAFETY: a live, connected pipe instance and an output pointer.
        if unsafe { GetNamedPipeClientProcessId(pipe.as_raw_handle(), &mut pid) } == 0 {
            return Err(AcceptError::Io(io::Error::last_os_error()));
        }
        let unknown = || AcceptError::Refused(Peer::Windows { pid, sid: None });
        // SAFETY: a query right and a process ID; the call fails for a process this user may
        // not query, which another user's process is.
        let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
        if process.is_null() {
            return Err(unknown());
        }
        // SAFETY: the handle `OpenProcess` just returned, owned only here.
        let process = unsafe { OwnedHandle::from_raw_handle(process.cast()) };
        let Ok(sid) = UserSid::of_process(process.as_raw_handle()) else {
            return Err(unknown());
        };
        let peer = Peer::Windows {
            pid,
            sid: sid.text().ok(),
        };
        // SAFETY: two valid SIDs, each alive for the call.
        if unsafe { EqualSid(sid.sid(), user.sid()) } != 0 {
            Ok(peer)
        } else {
            Err(AcceptError::Refused(peer))
        }
    }

    pub(super) fn connect(endpoint: &Endpoint) -> io::Result<File> {
        let path = pipe_path(endpoint)?;
        for _ in 0..BUSY_ATTEMPTS {
            // SAFETY: a NUL-terminated path, no security attributes and no template. The
            // identification level keeps the daemon from impersonating the client.
            let handle = unsafe {
                CreateFileW(
                    path.as_ptr(),
                    GENERIC_READ | GENERIC_WRITE,
                    0,
                    std::ptr::null(),
                    OPEN_EXISTING,
                    SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION,
                    std::ptr::null_mut(),
                )
            };
            if handle != INVALID_HANDLE_VALUE {
                // SAFETY: the handle `CreateFileW` just returned, owned only here.
                let handle = unsafe { OwnedHandle::from_raw_handle(handle.cast()) };
                return Ok(File::from(handle));
            }
            let error = io::Error::last_os_error();
            if error.raw_os_error() != Some(ERROR_PIPE_BUSY as i32) {
                return Err(error);
            }
            // SAFETY: a NUL-terminated path. A timeout or error here only means another try.
            unsafe { WaitNamedPipeW(path.as_ptr(), BUSY_WAIT_MS) };
        }
        Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "the daemon's pipe stayed busy",
        ))
    }

    /// The async listener, on tokio, with the `async` feature; not under loom, which tokio does
    /// not build under.
    #[cfg(all(feature = "async", not(loom)))]
    pub(in super::super) mod aio {
        use std::ffi::OsString;
        use std::io;
        use std::os::windows::io::AsHandle;
        use std::pin::Pin;
        use std::task::{Context, Poll};

        use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
        use tokio::net::windows::named_pipe::{
            ClientOptions, NamedPipeClient, NamedPipeServer, PipeMode, ServerOptions,
        };
        use windows_sys::Win32::Foundation::ERROR_PIPE_BUSY;
        use windows_sys::Win32::Security::SECURITY_ATTRIBUTES;
        use windows_sys::Win32::Storage::FileSystem::{
            SECURITY_IDENTIFICATION, SECURITY_SQOS_PRESENT,
        };

        use super::super::{AcceptError, Endpoint, Peer};
        use super::{BUFFER, BUSY_ATTEMPTS, BUSY_WAIT_MS, check_peer, pipe_path};
        use crate::shm::{OwnerOnly, UserSid};

        /// A connected pipe for tokio: the daemon's end, or a client's.
        #[derive(Debug)]
        pub(in crate::cmd::os) enum AsyncStream {
            Server(NamedPipeServer),
            Client(NamedPipeClient),
        }

        impl AsyncRead for AsyncStream {
            fn poll_read(
                self: Pin<&mut Self>,
                cx: &mut Context<'_>,
                buf: &mut ReadBuf<'_>,
            ) -> Poll<io::Result<()>> {
                match self.get_mut() {
                    AsyncStream::Server(pipe) => Pin::new(pipe).poll_read(cx, buf),
                    AsyncStream::Client(pipe) => Pin::new(pipe).poll_read(cx, buf),
                }
            }
        }

        impl AsyncWrite for AsyncStream {
            fn poll_write(
                self: Pin<&mut Self>,
                cx: &mut Context<'_>,
                buf: &[u8],
            ) -> Poll<io::Result<usize>> {
                match self.get_mut() {
                    AsyncStream::Server(pipe) => Pin::new(pipe).poll_write(cx, buf),
                    AsyncStream::Client(pipe) => Pin::new(pipe).poll_write(cx, buf),
                }
            }

            fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
                match self.get_mut() {
                    AsyncStream::Server(pipe) => Pin::new(pipe).poll_flush(cx),
                    AsyncStream::Client(pipe) => Pin::new(pipe).poll_flush(cx),
                }
            }

            fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
                match self.get_mut() {
                    AsyncStream::Server(pipe) => Pin::new(pipe).poll_shutdown(cx),
                    AsyncStream::Client(pipe) => Pin::new(pipe).poll_shutdown(cx),
                }
            }
        }

        pub(in crate::cmd::os) struct AsyncListener {
            /// `\\.\pipe\<name>`.
            path: OsString,
            /// The instance the next client connects to.
            pending: NamedPipeServer,
            /// This process's user, which every peer must be.
            user: UserSid,
            /// The DACL each instance is created with.
            descriptor: OwnerOnly,
        }

        impl std::fmt::Debug for AsyncListener {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("AsyncListener")
                    .field("path", &self.path)
                    .finish_non_exhaustive()
            }
        }

        /// The text form of a pipe path from [`pipe_path`], without its NUL.
        fn pipe_os_path(endpoint: &Endpoint) -> io::Result<OsString> {
            let mut path = pipe_path(endpoint)?;
            path.pop();
            Ok(String::from_utf16_lossy(&path).into())
        }

        /// One instance of the pipe at `path` for tokio, with the same flags and DACL as
        /// [`instance`]'s, and overlapped I/O.
        fn async_instance(
            path: &OsString,
            descriptor: &OwnerOnly,
            first: bool,
        ) -> io::Result<NamedPipeServer> {
            let mut attributes = SECURITY_ATTRIBUTES {
                nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
                lpSecurityDescriptor: descriptor.descriptor(),
                bInheritHandle: 0,
            };
            let mut options = ServerOptions::new();
            options
                .first_pipe_instance(first)
                .reject_remote_clients(true)
                .pipe_mode(PipeMode::Byte)
                .in_buffer_size(BUFFER)
                .out_buffer_size(BUFFER);
            // SAFETY: the attributes are a valid `SECURITY_ATTRIBUTES`, and they and the descriptor
            // they point to live for the call.
            unsafe {
                options.create_with_security_attributes_raw(path, (&raw mut attributes).cast())
            }
        }

        impl AsyncListener {
            pub(in crate::cmd::os) fn bind(endpoint: &Endpoint) -> io::Result<AsyncListener> {
                AsyncListener::with_descriptor(endpoint, OwnerOnly::new()?)
            }

            #[cfg(test)]
            pub(in crate::cmd::os) fn open_to_authenticated_users(
                endpoint: &Endpoint,
            ) -> io::Result<AsyncListener> {
                AsyncListener::with_descriptor(endpoint, OwnerOnly::open_to_authenticated_users()?)
            }

            fn with_descriptor(
                endpoint: &Endpoint,
                descriptor: OwnerOnly,
            ) -> io::Result<AsyncListener> {
                let path = pipe_os_path(endpoint)?;
                let user = UserSid::current()?;
                let pending = async_instance(&path, &descriptor, true)?;
                Ok(AsyncListener {
                    path,
                    pending,
                    user,
                    descriptor,
                })
            }

            pub(in crate::cmd::os) async fn accept(
                &mut self,
            ) -> Result<(AsyncStream, Peer), AcceptError> {
                // A client that connected before this call counts as connected.
                let connected = self.pending.connect().await;
                // A new instance replaces this one either way, as in `Listener::accept`.
                let next =
                    async_instance(&self.path, &self.descriptor, false).map_err(AcceptError::Io)?;
                let client = std::mem::replace(&mut self.pending, next);
                connected.map_err(AcceptError::Io)?;
                match check_peer(client.as_handle(), &self.user) {
                    Ok(peer) => Ok((AsyncStream::Server(client), peer)),
                    Err(refusal) => {
                        // Cut the refused client off now. Dropping the instance alone closes its
                        // handle only once the runtime has processed the cancellation of the
                        // read tokio keeps pending on it, which a runtime that has stopped
                        // polling never does.
                        let _ = client.disconnect();
                        Err(refusal)
                    }
                }
            }
        }

        pub(in crate::cmd::os) async fn connect_async(
            endpoint: &Endpoint,
        ) -> io::Result<AsyncStream> {
            let path = pipe_os_path(endpoint)?;
            let mut options = ClientOptions::new();
            // The identification level keeps the daemon from impersonating the client.
            options.security_qos_flags(SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION);
            for _ in 0..BUSY_ATTEMPTS {
                match options.open(&path) {
                    Ok(client) => return Ok(AsyncStream::Client(client)),
                    Err(error) if error.raw_os_error() == Some(ERROR_PIPE_BUSY as i32) => {
                        tokio::time::sleep(std::time::Duration::from_millis(u64::from(
                            BUSY_WAIT_MS,
                        )))
                        .await;
                    }
                    Err(error) => return Err(error),
                }
            }
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "the daemon's pipe stayed busy",
            ))
        }

        /// The pipe name of this user's endpoint for `role`: `syntropy-<user SID>-<role>`.
        pub(in crate::cmd::os) fn user_pipe_name(role: &str) -> io::Result<String> {
            Ok(format!("syntropy-{}-{role}", UserSid::current()?.text()?))
        }
    }
}
