// SPDX-License-Identifier: MIT OR Apache-2.0

//! The command channel of a runner the daemon spawns: a connected pair whose runner end the child
//! inherits (RFC-0001 §3.4; ADR-0015 item 1; DIST-E1-T04).
//!
//! The daemon builds the pair before it spawns the runner, so nothing listens and nothing can
//! connect in between:
//!
//! | OS | Pair | The child's end |
//! |---|---|---|
//! | Linux, Apple platforms | `socketpair(AF_UNIX, SOCK_STREAM)` | its descriptor, close-on-exec in the daemon; a hook clears the flag in the runner's child alone, after the fork and before the exec, as [`Region::configure_spawn`](crate::shm::Region::configure_spawn) does for the region |
//! | Windows | a named pipe of a random name, created as its first instance with the owner-only DACL, which the daemon connects to itself, then closes for new clients | its handle, made inheritable until the spawn |
//!
//! The runner is told its end on its command line, `__runner --ipc <handle>`, in
//! [`ChannelHandle`]'s text form, and takes it with [`inherited`].
//!
//! **Inheritance windows.** Linux has none. On Apple platforms the pair is made close-on-exec
//! just after its creation, not with it, so a process the daemon forks from another thread in that
//! instant inherits both ends. On Windows, `std::process::Command` passes every inheritable handle
//! to the child, and offers no list of the handles to pass on stable Rust, so between
//! [`spawn_pair`] and the drop of its [`ChildEnd`], a process the daemon spawns from another
//! thread also inherits the runner's end.
//!
//! A process that inherited an end holds it open until it exits. If it holds the daemon's end, the
//! runner does not read end-of-file when the daemon's process ends, and finds the daemon lost by
//! its still heartbeat instead, within `runner.daemon_silence` (RFC-0052 §6.5). If it holds the
//! runner's end, the daemon sees the runner's exit through the process, not end-of-file.

use std::fmt;
use std::io;
use std::process::Command;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};

use syntropy_core::Entropy;

use super::os::Stream;

/// The runner's end of its command channel, as the runner's command line names it: an inherited
/// descriptor on Linux and Apple platforms, `fd:<n>`, or an inherited handle on Windows,
/// `handle:<n>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChannelHandle {
    /// A descriptor (Linux, Apple platforms).
    Fd(i32),
    /// A handle's value (Windows).
    Handle(u64),
}

impl fmt::Display for ChannelHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChannelHandle::Fd(fd) => write!(f, "fd:{fd}"),
            ChannelHandle::Handle(handle) => write!(f, "handle:{handle}"),
        }
    }
}

/// Why a channel handle's text form did not parse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelHandleParseError(String);

impl fmt::Display for ChannelHandleParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "`{}` is not a command-channel handle: expected fd:<n> or handle:<n>",
            self.0
        )
    }
}

impl std::error::Error for ChannelHandleParseError {}

impl FromStr for ChannelHandle {
    type Err = ChannelHandleParseError;

    fn from_str(text: &str) -> Result<ChannelHandle, ChannelHandleParseError> {
        let bad = || ChannelHandleParseError(text.to_owned());
        if let Some(fd) = text.strip_prefix("fd:") {
            return match fd.parse() {
                Ok(fd) if fd >= 0 => Ok(ChannelHandle::Fd(fd)),
                _ => Err(bad()),
            };
        }
        match text.strip_prefix("handle:").map(str::parse) {
            Some(Ok(handle)) => Ok(ChannelHandle::Handle(handle)),
            _ => Err(bad()),
        }
    }
}

/// The child's end of a pair from [`spawn_pair`], held by the daemon until the spawn.
///
/// **Keep it until `command` has spawned, then drop it**, or drop it when the spawn failed. On
/// Linux and Apple platforms the hook [`spawn_pair`] adds to `command` names the end's descriptor
/// by number, so spawning `command` after the drop would clear close-on-exec on whatever
/// descriptor reused the number. Until the drop the daemon holds a copy of the runner's end, so it
/// would not see end-of-file when the runner exits.
#[derive(Debug)]
pub struct ChildEnd(imp::ChildEnd);

impl ChildEnd {
    /// The handle the child finds its end at, for its command line.
    #[must_use]
    pub fn handle(&self) -> ChannelHandle {
        self.0.handle()
    }
}

/// A connected pair in this process: two [`Stream`]s, either of which reads what the other
/// writes. On Windows the pipe's random name is drawn from `entropy`.
///
/// # Errors
///
/// The OS error.
pub fn pair(entropy: &dyn Entropy) -> io::Result<(Stream, Stream)> {
    imp::pair(entropy)
}

/// Builds the command channel of the runner `command` is about to start: returns the daemon's end,
/// and the child's end, whose [`handle`](ChildEnd::handle) goes on `command`'s command line. On
/// Linux and Apple platforms it also adds a hook to `command` that leaves the child's descriptor
/// open across the exec in the child alone.
///
/// # Errors
///
/// The OS error.
pub fn spawn_pair(command: &mut Command, entropy: &dyn Entropy) -> io::Result<(Stream, ChildEnd)> {
    imp::spawn_pair(command, entropy).map(|(daemon, child)| (daemon, ChildEnd(child)))
}

/// Whether a process has taken its inherited end, which it may do once.
static TAKEN: AtomicBool = AtomicBool::new(false);

/// The runner's side: takes the command channel `handle` names, which this process inherited from
/// the daemon that spawned it.
///
/// The handle must name an open socket (Linux, Apple platforms) or pipe (Windows) above the
/// standard streams, **as it was inherited**: not close-on-exec, or on Windows inheritable. Every
/// descriptor and handle the standard library and this workspace's code open is close-on-exec, or
/// not inheritable, from its creation, and a taken channel is made so too, so this refuses any
/// channel opened that way. A process takes its channel once: a second call fails.
///
/// What the check cannot see is left to the caller: a descriptor opened without the flag by code
/// outside the workspace, such as a C library, or one the process inherited for another purpose,
/// such as a service manager's activation socket, also passes. The runner calls this once, at
/// start-up, with the handle its command line names, before anything else could take such a
/// descriptor.
///
/// # Errors
///
/// [`io::ErrorKind::InvalidInput`] for a handle of another OS's kind, a standard stream, a
/// descriptor or handle that is not a socket or a pipe, or one that was not inherited;
/// [`io::ErrorKind::AlreadyExists`] when this process has already taken its channel; otherwise the
/// OS error.
pub fn inherited(handle: &ChannelHandle) -> io::Result<Stream> {
    imp::check(handle)?;
    if TAKEN.swap(true, Ordering::AcqRel) {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "this process has already taken its inherited command channel",
        ));
    }
    imp::take(handle)
}

fn invalid(handle: &ChannelHandle, why: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("the inherited command channel {handle}: {why}"),
    )
}

#[cfg(any(target_os = "linux", target_vendor = "apple"))]
mod imp {
    use std::io;
    use std::mem::MaybeUninit;
    use std::os::fd::{AsRawFd, FromRawFd, RawFd};
    use std::os::unix::net::UnixStream;
    use std::os::unix::process::CommandExt;
    use std::process::Command;

    use syntropy_core::Entropy;

    use super::{ChannelHandle, invalid};
    use crate::cmd::os::Stream;

    #[derive(Debug)]
    pub(super) struct ChildEnd(UnixStream);

    impl ChildEnd {
        pub(super) fn handle(&self) -> ChannelHandle {
            ChannelHandle::Fd(self.0.as_raw_fd())
        }
    }

    pub(super) fn pair(_entropy: &dyn Entropy) -> io::Result<(Stream, Stream)> {
        let (a, b) = UnixStream::pair()?;
        Ok((Stream::from_os(a), Stream::from_os(b)))
    }

    pub(super) fn spawn_pair(
        command: &mut Command,
        _entropy: &dyn Entropy,
    ) -> io::Result<(Stream, ChildEnd)> {
        // Close-on-exec from creation on Linux (`SOCK_CLOEXEC`); just after it on Apple platforms
        // (module docs).
        let (daemon, child) = UnixStream::pair()?;
        let fd = child.as_raw_fd();
        // SAFETY: the hook runs in the child between `fork` and `exec`, where only
        // async-signal-safe calls are allowed. It makes one `fcntl` call, which is
        // async-signal-safe, builds its error without allocating, and touches nothing else. `fd`
        // is above the standard streams, which the child sets up before the hook runs. The hook
        // holds only the number: `ChildEnd` keeps the descriptor open, and its docs require it to
        // live until `command` has spawned.
        unsafe {
            command.pre_exec(move || {
                if libc::fcntl(fd, libc::F_SETFD, 0) == -1 {
                    Err(io::Error::last_os_error())
                } else {
                    Ok(())
                }
            });
        }
        Ok((Stream::from_os(daemon), ChildEnd(child)))
    }

    pub(super) fn check(handle: &ChannelHandle) -> io::Result<()> {
        let ChannelHandle::Fd(fd) = *handle else {
            return Err(invalid(handle, "this OS passes a descriptor, fd:<n>"));
        };
        if fd <= libc::STDERR_FILENO {
            return Err(invalid(
                handle,
                "a standard stream is not a command channel",
            ));
        }
        let mut stat = MaybeUninit::<libc::stat>::uninit();
        // SAFETY: `fstat` writes the record on success and reads only the descriptor's number; an
        // unopened number fails with EBADF.
        if unsafe { libc::fstat(fd, stat.as_mut_ptr()) } == -1 {
            let error = io::Error::last_os_error();
            return Err(io::Error::new(
                error.kind(),
                format!("the inherited command channel {handle}: {error}"),
            ));
        }
        // SAFETY: `fstat` succeeded, so it wrote the record.
        let mode = unsafe { stat.assume_init() }.st_mode;
        if mode & libc::S_IFMT != libc::S_IFSOCK {
            return Err(invalid(handle, "the descriptor is not a socket"));
        }
        // SAFETY: reads an open descriptor's flags.
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
        if flags == -1 {
            return Err(io::Error::last_os_error());
        }
        if flags & libc::FD_CLOEXEC != 0 {
            return Err(invalid(
                handle,
                "the descriptor is close-on-exec, so this process opened it; it was not inherited",
            ));
        }
        Ok(())
    }

    pub(super) fn take(handle: &ChannelHandle) -> io::Result<Stream> {
        let ChannelHandle::Fd(fd) = *handle else {
            unreachable!("`check` accepted only a descriptor");
        };
        let fd: RawFd = fd;
        // SAFETY: `check` found an open socket above the standard streams that is not
        // close-on-exec. Everything the standard library and this workspace open is close-on-exec
        // from its creation, so the descriptor crossed the exec, and the runner takes it at
        // start-up, before any other code could claim an inherited descriptor (the precondition
        // `inherited`'s docs state). It is made close-on-exec below, so it is never taken again.
        let stream = unsafe { UnixStream::from_raw_fd(fd) };
        // SAFETY: an owned, open descriptor and a flag value.
        if unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) } == -1 {
            return Err(io::Error::last_os_error());
        }
        Ok(Stream::from_os(stream))
    }
}

#[cfg(windows)]
mod imp {
    use std::fs::File;
    use std::io;
    use std::os::windows::io::{AsRawHandle, FromRawHandle};
    use std::process::Command;

    use syntropy_core::Entropy;
    use windows_sys::Win32::Foundation::{
        GetHandleInformation, HANDLE, HANDLE_FLAG_INHERIT, SetHandleInformation,
    };
    use windows_sys::Win32::Storage::FileSystem::{FILE_TYPE_PIPE, GetFileType};

    use super::{ChannelHandle, invalid};
    use crate::cmd::os::{Endpoint, Listener, Stream, connect};

    #[derive(Debug)]
    pub(super) struct ChildEnd(File);

    impl ChildEnd {
        pub(super) fn handle(&self) -> ChannelHandle {
            ChannelHandle::Handle(self.0.as_raw_handle() as u64)
        }
    }

    pub(super) fn pair(entropy: &dyn Entropy) -> io::Result<(Stream, Stream)> {
        let mut bytes = [0u8; 16];
        entropy.fill(&mut bytes);
        let name: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        let endpoint = Endpoint::Pipe(format!("syntropy-runner-{name}"));
        // The first instance, so no other process holds the name, with the owner-only DACL.
        let mut listener = Listener::bind(&endpoint)?;
        let client = connect(&endpoint)?;
        let server = listener.accept().map_err(|error| match error {
            crate::cmd::AcceptError::Io(error) => error,
            refused @ crate::cmd::AcceptError::Refused(_) => {
                io::Error::new(io::ErrorKind::PermissionDenied, refused.to_string())
            }
        })?;
        // Dropping the listener closes its next instance, so no client can connect by the name.
        drop(listener);
        Ok((server, client))
    }

    pub(super) fn spawn_pair(
        _command: &mut Command,
        entropy: &dyn Entropy,
    ) -> io::Result<(Stream, ChildEnd)> {
        let (daemon, child) = pair(entropy)?;
        let child = child.into_os();
        set_inherit(child.as_raw_handle() as HANDLE, true)?;
        Ok((daemon, ChildEnd(child)))
    }

    fn set_inherit(handle: HANDLE, inherit: bool) -> io::Result<()> {
        let flag = if inherit { HANDLE_FLAG_INHERIT } else { 0 };
        // SAFETY: an open handle this process owns, and a flag mask and value.
        if unsafe { SetHandleInformation(handle, HANDLE_FLAG_INHERIT, flag) } == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    pub(super) fn check(handle: &ChannelHandle) -> io::Result<()> {
        let ChannelHandle::Handle(value) = *handle else {
            return Err(invalid(handle, "this OS passes a handle, handle:<n>"));
        };
        if value == 0 {
            return Err(invalid(handle, "a null handle is not a command channel"));
        }
        // SAFETY: `GetFileType` only reads the handle's value; a value that is not an open
        // handle gives `FILE_TYPE_UNKNOWN` with an error.
        if unsafe { GetFileType(value as HANDLE) } != FILE_TYPE_PIPE {
            return Err(invalid(handle, "the handle is not a pipe"));
        }
        let mut flags = 0u32;
        // SAFETY: an open handle, as `GetFileType` found, and an output pointer.
        if unsafe { GetHandleInformation(value as HANDLE, &mut flags) } == 0 {
            return Err(io::Error::last_os_error());
        }
        if flags & HANDLE_FLAG_INHERIT == 0 {
            return Err(invalid(
                handle,
                "the handle is not inheritable, so this process opened it; it was not inherited",
            ));
        }
        Ok(())
    }

    pub(super) fn take(handle: &ChannelHandle) -> io::Result<Stream> {
        let ChannelHandle::Handle(value) = *handle else {
            unreachable!("`check` accepted only a handle");
        };
        // SAFETY: `check` found an open, inheritable pipe handle. Every handle the process opens
        // is not inheritable from its creation, so this one came from the daemon, and nothing in
        // the process owns it. It is made not inheritable below, so it is never taken again.
        let file = unsafe { File::from_raw_handle(value as HANDLE) };
        set_inherit(value as HANDLE, false)?;
        Ok(Stream::from_os(file))
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};

    use super::*;

    /// Bytes that differ on every call, in every test and process, so tests running in parallel
    /// never draw the same pipe name: a pipe must be its name's first instance.
    struct Counter;

    static DRAWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    impl Entropy for Counter {
        fn fill(&self, buf: &mut [u8]) {
            let n = DRAWS.fetch_add(1, Ordering::Relaxed);
            let mut x =
                (u64::from(std::process::id()) << 32 | n).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            for byte in buf {
                x ^= x >> 29;
                x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
                *byte = (x >> 56) as u8;
            }
        }
    }

    #[test]
    fn a_channel_handle_round_trips_through_its_text_form() {
        for handle in [ChannelHandle::Fd(7), ChannelHandle::Handle(0x1f4)] {
            assert_eq!(handle.to_string().parse::<ChannelHandle>(), Ok(handle));
        }
        assert_eq!("fd:7".parse(), Ok(ChannelHandle::Fd(7)));
        for bad in [
            "",
            "fd:",
            "fd:-1",
            "fd:x",
            "handle:",
            "handle:-3",
            "socket:/tmp/a",
            "7",
        ] {
            assert!(bad.parse::<ChannelHandle>().is_err(), "{bad:?} parsed");
        }
    }

    #[test]
    fn a_pair_carries_bytes_both_ways() {
        let (mut a, mut b) = pair(&Counter).unwrap();
        a.write_all(b"ping").unwrap();
        let mut got = [0; 4];
        b.read_exact(&mut got).unwrap();
        assert_eq!(&got, b"ping");
        b.write_all(b"pong").unwrap();
        a.read_exact(&mut got).unwrap();
        assert_eq!(&got, b"pong");
        drop(b);
        assert_eq!(
            a.read(&mut got).unwrap(),
            0,
            "end-of-file once the other end closes"
        );
    }

    /// A standard stream, a handle of the other OS's kind, and a descriptor that is not a socket
    /// are refused before anything is taken.
    #[test]
    fn inherited_refuses_what_is_not_a_channel() {
        let refused = |handle: ChannelHandle| {
            inherited(&handle).expect_err("refused").kind() == io::ErrorKind::InvalidInput
        };
        if cfg!(windows) {
            assert!(refused(ChannelHandle::Fd(5)));
            assert!(refused(ChannelHandle::Handle(0)));
        } else {
            assert!(refused(ChannelHandle::Fd(0)));
            assert!(refused(ChannelHandle::Fd(2)));
            assert!(refused(ChannelHandle::Handle(5)));
            #[cfg(unix)]
            {
                use std::os::fd::AsRawFd;
                let file = std::fs::File::open(std::env::current_exe().unwrap()).unwrap();
                assert!(refused(ChannelHandle::Fd(file.as_raw_fd())));
            }
        }
    }

    /// A socket or pipe this process opened itself is refused, so `inherited` never takes a second
    /// owner's channel: the standard library opens everything close-on-exec, or not inheritable.
    #[test]
    fn inherited_refuses_a_channel_this_process_owns() {
        #[cfg(unix)]
        let (_owned, handle) = {
            use std::os::fd::AsRawFd;
            let (a, b) = std::os::unix::net::UnixStream::pair().unwrap();
            let fd = a.as_raw_fd();
            ((a, b), ChannelHandle::Fd(fd))
        };
        #[cfg(windows)]
        let (_owned, handle) = {
            use std::os::windows::io::AsRawHandle;
            let (a, b) = pair(&Counter).unwrap();
            let a = a.into_os();
            let raw = a.as_raw_handle() as u64;
            ((a, b), ChannelHandle::Handle(raw))
        };
        let error = inherited(&handle).expect_err("refused");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("not inherited"), "{error}");
    }
}
