// SPDX-License-Identifier: MIT OR Apache-2.0

//! The command channel between the daemon and a runner (RFC-0001 §3.5; ADR-0015 items 1 and 4;
//! RFC-0034 §6.4).
//!
//! **Frames.** Each message travels as one frame: a 4-byte little-endian length, then that many
//! bytes of postcard encoding one [`Envelope`]. A frame holds at most [`MAX_FRAME`] bytes after
//! its length. [`decode_frame`] is the one decoder. [`Channel::recv`] calls it on every frame, and
//! the `ipc_msgs` fuzz target (QUAL-E10-T03) wraps it. It is total: any input gives an envelope or
//! a [`FrameError`], never a panic.
//!
//! **Request IDs.** Each side numbers the messages it sends from 1. A reply names the message it
//! answers in [`Envelope::reply_to`], as `Pong` names its `Ping`.
//!
//! **Messages.** [`DaemonMsg`] goes from daemon to runner, and [`RunnerMsg`] from runner to
//! daemon. They hold the RFC-0001 §3.5 messages whose field types exist at M0a. The tasks that
//! own the other messages' types add them (§3.5, note of 2026-10-23), and each addition bumps
//! [`PROTOCOL`].
//!
//! **Endpoints and peers.** The daemon [listens](Listener::bind) on an [`Endpoint`] only its user
//! can reach, and [checks](Listener::accept) each connecting peer's OS credentials:
//!
//! | OS | Endpoint | Owner-only | Peer check |
//! |---|---|---|---|
//! | Linux | Unix-domain socket | in a directory of mode 0700, owned by the user | `SO_PEERCRED`: the peer's user ID is ours |
//! | Apple platforms | Unix-domain socket | as on Linux | `getpeereid`: the peer's user ID is ours |
//! | Windows | named pipe; remote clients rejected | a protected DACL with one entry, for the user | the client process's token user SID is ours |
//!
//! A peer that fails the check, or whose credentials cannot be read, is refused and its connection
//! closed. A runner only connects; it never listens (RFC-0034 §6.4).
//!
//! [`AsyncListener`] is the same endpoint and check for servers on tokio: the API's local socket,
//! on the daemon's `ctl` runtime (RFC-0050 §1.1), which also gives its Unix socket mode 0600.
//! [`connect_async`] is its client. Both come with the `async` feature, which is off by default,
//! so a crate that only speaks the command channel, such as the runner, does not link tokio
//! through `ipc`. Neither exists under `--cfg loom`, which tokio does not build under.
//!
//! **Handshake.** A spawned runner is the daemon's own build (ADR-0015 item 1). It sends `Hello`
//! first, and [`DaemonChannel::handshake`] checks the build ID, the protocol and the process ID,
//! and kills the runner on any mismatch. The daemon then sends `Init`, and
//! [`RunnerChannel::handshake`] checks that its build ID is the runner's own. Adopting a runner of
//! another build, which needs an equal protocol instead, is RFC-0052 §6.5's (DIST-E1-T08).
//!
//! **A spawned runner's channel.** The daemon spawns a runner with its end of a connected pair
//! already open: a socket pair, or on Windows a pipe the daemon connected to itself
//! ([`spawn_pair`]). The runner finds its end at the [`ChannelHandle`] its command line names,
//! and takes it with [`inherited`] (RFC-0001 §3.4; DIST-E1-T04).
//!
//! **In-process transport.** A channel runs over a [`Transport`]: the OS [`Stream`] above, or
//! [`memory::MemoryStream`], a connected pair in one process, for a runner the daemon hosts on its
//! own threads (`RunnerHost::InProcess`, RFC-0001 §3.1; ADR-0015 item 7; DIST-E1-T11). The framing
//! and the encoding are the same over both. [`memory_channel`] gives a connected daemon end and
//! runner end, and [`DaemonChannel::handshake_in_process`] checks the in-process runner's `Hello`
//! against this process's ID.
//!
//! **Fault plans (test builds only).** With the `fault-injection` feature, a test daemon can push
//! a fault plan to a runner (RFC-0060 §9.2; DIST-E1-T13): `DaemonFrame::SetFaultPlan`, answered
//! by `RunnerFrame::FaultPlanSet`, at the fixed wire index `TEST_VARIANT` (2²⁰) above every
//! production variant. A build without the feature has neither type, and refuses such a frame
//! as an unknown variant:
//!
#![cfg_attr(not(feature = "fault-injection"), doc = "```compile_fail")]
#![cfg_attr(feature = "fault-injection", doc = "```")]
//! use syntropy_ipc::cmd::DaemonFrame;
//! ```

use std::fmt;
use std::io::{self, Read, Write};
use std::marker::PhantomData;
use std::process::{Child, ExitStatus};

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::shm::IpcProtocol;

mod frame;
pub mod memory;
mod msg;
mod os;
mod spawn;

pub(crate) use frame::NoPrealloc;
pub use frame::{EncodeError, FrameError, HEADER, MAX_FRAME, decode_frame, encode_frame};
pub use msg::{DaemonMsg, Envelope, Hello, Init, RequestId, RunnerMsg};
pub use os::{AcceptError, Endpoint, EndpointParseError, Listener, Peer, Stream, connect};
pub use spawn::{ChannelHandle, ChannelHandleParseError, ChildEnd, inherited, pair, spawn_pair};
// The async half, on tokio, with the `async` feature: not under loom, which tokio does not build
// under.
#[cfg(all(windows, feature = "async", not(loom)))]
pub use os::user_pipe;
#[cfg(all(feature = "async", not(loom)))]
pub use os::{AsyncListener, AsyncStream, connect_async};
// The test-only fault-plan message (RFC-0060 §9.2; DIST-E1-T13), with `fault-injection` only.
#[cfg(feature = "fault-injection")]
mod fault;
#[cfg(feature = "fault-injection")]
pub use fault::{
    DaemonFrame, FaultDaemonChannel, FaultRunnerChannel, RunnerFrame, TEST_VARIANT,
    install_fault_plan,
};

/// The IPC protocol this build speaks (RFC-0001 §3.5): the command messages and every ring record
/// schema. A change to either bumps it (RFC-0001 §5, note of 2026-10-22). QUAL-E17-T15's schema
/// fingerprint and golden vectors check that it does.
///
/// 1.0 was DIST-E1-T03's message set. 1.1 appends `Drain`, `Shutdown` and `MemoryPressure` to
/// [`DaemonMsg`] and `SequenceEnded` to [`RunnerMsg`] (DIST-E1-T15): a compatible change, because
/// every earlier message keeps its variant index and layout. 2.0 changes `StateChanged`'s `state`
/// from a `u32` discriminant to the whole `RunnerState` (DIST-E1-T05): the variant keeps its index,
/// but a 1.x decoder cannot read its new layout, so the change is incompatible and bumps the major.
/// `Hello` keeps its layout, so a daemon of any build still reads a runner's `Hello` and refuses a
/// mismatch by its protocol. 3.0 adds the live-sequence count to the shared-memory header
/// (DIST-E1-T06): the header grows by a line, the ring table moves, and the region's
/// `LAYOUT_VERSION` goes to 2, so a 2.x peer cannot read the region. That is incompatible, so the
/// major moves. 4.0 appends `Spans` to the `telemetry` ring's `TelemetryRecord` (PROD-E8-T07;
/// RFC-0001 §18.49). Every earlier variant keeps its index, but a 4.0 runner writes `Spans` into
/// the ring whenever it records spans, and a 3.x daemon cannot decode one. A change is compatible
/// only when every older reader can read everything a newer writer sends it, so this one moves
/// the major. Adoption needs equal versions anyway (RFC-0001 §3.5), so the bump changes what the
/// version claims, not who may connect.
/// 5.0 appends `FetchShard` and `PackCommitted` to [`RunnerMsg`] (COMP-E10-T11; RFC-0001
/// §18.50). Every earlier message keeps its index and layout, but a 5.0 runner sends them whenever
/// it fetches a shard or commits a pack, and a 4.x daemon cannot decode either, so by the same rule
/// the major moves.
pub const PROTOCOL: IpcProtocol = IpcProtocol { major: 5, minor: 0 };

/// The byte stream a command channel runs over: an OS [`Stream`] between processes, or a
/// [`memory::MemoryStream`] between threads of one process (RFC-0001 §3.1, `RunnerHost::InProcess`;
/// DIST-E1-T11). A channel frames and encodes the same way over either, so the two are
/// interchangeable.
pub trait Transport: Read + Write + Send + fmt::Debug {}

impl Transport for Stream {}

impl Transport for memory::MemoryStream {}

/// One end of a command channel, which sends `S` and receives `R` over a [`Transport`], by
/// default an OS [`Stream`].
///
/// After an error from [`recv`](Channel::recv) other than [`FrameError::Malformed`] or
/// [`FrameError::Trailing`], the stream is no longer at a frame boundary, and the channel should
/// be dropped.
pub struct Channel<S, R, T: Transport = Stream> {
    stream: T,
    sent: u64,
    types: PhantomData<fn(S) -> R>,
}

/// The daemon's end of a channel to a runner.
pub type DaemonChannel<T = Stream> = Channel<DaemonMsg, RunnerMsg, T>;

/// A runner's end of its channel to the daemon.
pub type RunnerChannel<T = Stream> = Channel<RunnerMsg, DaemonMsg, T>;

/// A connected daemon end and runner end in one process, over a [`memory::pair`]: the channel
/// of a runner the daemon hosts in-process.
#[must_use]
pub fn memory_channel() -> (
    DaemonChannel<memory::MemoryStream>,
    RunnerChannel<memory::MemoryStream>,
) {
    let (daemon, runner) = memory::pair();
    (Channel::new(daemon), Channel::new(runner))
}

impl<S, R, T: Transport> fmt::Debug for Channel<S, R, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Channel")
            .field("stream", &self.stream)
            .field("sent", &self.sent)
            .finish()
    }
}

impl<S: Serialize, R: DeserializeOwned, T: Transport> Channel<S, R, T> {
    /// A channel over a connected stream.
    pub fn new(stream: T) -> Channel<S, R, T> {
        Channel {
            stream,
            sent: 0,
            types: PhantomData,
        }
    }

    /// Sends `msg`, and returns its number.
    pub fn send(&mut self, msg: &S) -> Result<RequestId, SendError> {
        self.send_envelope(None, msg)
    }

    /// Sends `msg` as the answer to the other side's message `to`, and returns its number.
    pub fn reply(&mut self, to: RequestId, msg: &S) -> Result<RequestId, SendError> {
        self.send_envelope(Some(to), msg)
    }

    fn send_envelope(
        &mut self,
        reply_to: Option<RequestId>,
        msg: &S,
    ) -> Result<RequestId, SendError> {
        let id = RequestId(self.sent + 1);
        let frame = encode_frame(&Envelope { id, reply_to, msg }).map_err(SendError::Encode)?;
        self.stream.write_all(&frame).map_err(SendError::Io)?;
        self.sent += 1;
        Ok(id)
    }

    /// Waits for the next message.
    pub fn recv(&mut self) -> Result<Envelope<R>, RecvError> {
        let mut frame = vec![0; HEADER];
        let mut filled = 0;
        while filled < HEADER {
            match self.stream.read(&mut frame[filled..]) {
                Ok(0) if filled == 0 => return Err(RecvError::Closed),
                Ok(0) => return Err(RecvError::Io(io::ErrorKind::UnexpectedEof.into())),
                Ok(n) => filled += n,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(RecvError::Io(error)),
            }
        }
        let len = u32::from_le_bytes([frame[0], frame[1], frame[2], frame[3]]);
        if len as usize > MAX_FRAME {
            return Err(RecvError::Frame(FrameError::TooLarge { len }));
        }
        // The payload grows as it arrives, so a peer that declares more than it sends costs only
        // what it sent.
        let read = (&mut self.stream)
            .take(u64::from(len))
            .read_to_end(&mut frame)
            .map_err(RecvError::Io)?;
        if read < len as usize {
            return Err(RecvError::Io(io::ErrorKind::UnexpectedEof.into()));
        }
        let (envelope, _) = decode_frame(&frame).map_err(RecvError::Frame)?;
        Ok(envelope)
    }

    /// The stream under the channel.
    pub fn stream(&self) -> &T {
        &self.stream
    }

    /// The stream under the channel, to change its mode, such as a memory stream's
    /// non-blocking reads.
    pub fn stream_mut(&mut self) -> &mut T {
        &mut self.stream
    }
}

impl<T: Transport> Channel<DaemonMsg, RunnerMsg, T> {
    /// The daemon's side of a spawned runner's handshake: waits for the runner's `Hello`, and
    /// checks that the runner is this build (`build_id`), speaks [`PROTOCOL`] and is the process
    /// `runner`. On any failure it kills and reaps `runner` (ADR-0015 item 1).
    ///
    /// The daemon sends `Init` next.
    pub fn handshake(&mut self, runner: &mut Child, build_id: &[u8; 32]) -> Result<Hello, Refused> {
        match self.check_hello(runner.id(), build_id) {
            Ok(hello) => Ok(hello),
            Err(cause) => {
                let exit = runner.kill().and_then(|()| runner.wait());
                Err(Refused { cause, exit })
            }
        }
    }

    /// The daemon's side of the handshake with a runner it hosts in-process
    /// (`RunnerHost::InProcess`): as [`handshake`](Self::handshake), but the runner shares the
    /// daemon's process, so its `Hello` must name this process's ID, and on a failure there is no
    /// process to kill. The caller ends the runner's thread.
    ///
    /// # Errors
    ///
    /// [`HandshakeError`] as for [`handshake`](Self::handshake).
    pub fn handshake_in_process(&mut self, build_id: &[u8; 32]) -> Result<Hello, HandshakeError> {
        self.check_hello(std::process::id(), build_id)
    }

    fn check_hello(&mut self, pid: u32, build_id: &[u8; 32]) -> Result<Hello, HandshakeError> {
        let envelope = self.recv().map_err(HandshakeError::Recv)?;
        let RunnerMsg::Hello(hello) = envelope.msg else {
            return Err(HandshakeError::Unexpected { expected: "Hello" });
        };
        if hello.build_id != *build_id {
            return Err(HandshakeError::Build {
                theirs: hello.build_id,
                ours: *build_id,
            });
        }
        if hello.ipc_protocol != PROTOCOL {
            return Err(HandshakeError::Protocol {
                theirs: hello.ipc_protocol,
                ours: PROTOCOL,
            });
        }
        if hello.pid != pid {
            return Err(HandshakeError::Pid {
                claimed: hello.pid,
                spawned: pid,
            });
        }
        Ok(hello)
    }
}

impl<T: Transport> Channel<RunnerMsg, DaemonMsg, T> {
    /// The runner's side of the handshake: sends `hello`, waits for `Init`, and checks that `Init`
    /// comes from a daemon of the runner's own build. On an error the runner exits.
    pub fn handshake(&mut self, hello: &Hello) -> Result<Init, HandshakeError> {
        self.send(&RunnerMsg::Hello(*hello))
            .map_err(HandshakeError::Send)?;
        let envelope = self.recv().map_err(HandshakeError::Recv)?;
        let DaemonMsg::Init(init) = envelope.msg else {
            return Err(HandshakeError::Unexpected { expected: "Init" });
        };
        if init.build_id != hello.build_id {
            return Err(HandshakeError::Build {
                theirs: init.build_id,
                ours: hello.build_id,
            });
        }
        Ok(init)
    }
}

/// Why a message was not sent.
#[derive(Debug)]
pub enum SendError {
    /// It could not be framed.
    Encode(EncodeError),
    /// The stream refused the frame.
    Io(io::Error),
}

impl fmt::Display for SendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SendError::Encode(error) => error.fmt(f),
            SendError::Io(error) => write!(f, "the command channel failed: {error}"),
        }
    }
}

impl std::error::Error for SendError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SendError::Encode(error) => Some(error),
            SendError::Io(error) => Some(error),
        }
    }
}

/// Why no message was received.
#[derive(Debug)]
pub enum RecvError {
    /// The other side closed the channel between frames.
    Closed,
    /// The stream failed, or ended inside a frame.
    Io(io::Error),
    /// A frame arrived that did not decode.
    Frame(FrameError),
}

impl fmt::Display for RecvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RecvError::Closed => f.write_str("the other side closed the command channel"),
            RecvError::Io(error) => write!(f, "the command channel failed: {error}"),
            RecvError::Frame(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for RecvError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RecvError::Closed => None,
            RecvError::Io(error) => Some(error),
            RecvError::Frame(error) => Some(error),
        }
    }
}

/// Why a handshake failed.
#[derive(Debug)]
pub enum HandshakeError {
    /// The handshake message could not be sent.
    Send(SendError),
    /// The other side's handshake message did not arrive.
    Recv(RecvError),
    /// The other side sent something else first.
    Unexpected {
        /// The message the handshake needed.
        expected: &'static str,
    },
    /// The other side is another build.
    Build {
        /// Its build ID.
        theirs: [u8; 32],
        /// Ours.
        ours: [u8; 32],
    },
    /// The runner speaks another protocol.
    Protocol {
        /// Its protocol.
        theirs: IpcProtocol,
        /// Ours.
        ours: IpcProtocol,
    },
    /// The runner's `Hello` names a process other than the one the daemon spawned.
    Pid {
        /// The process ID in `Hello`.
        claimed: u32,
        /// The spawned runner's.
        spawned: u32,
    },
}

/// A build ID's first 8 bytes in hex, enough to tell two builds apart in a message.
struct Short<'a>(&'a [u8; 32]);

impl fmt::Display for Short<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0[..8]
            .iter()
            .try_for_each(|byte| write!(f, "{byte:02x}"))
    }
}

impl fmt::Display for HandshakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HandshakeError::Send(error) => write!(f, "handshake: {error}"),
            HandshakeError::Recv(error) => write!(f, "handshake: {error}"),
            HandshakeError::Unexpected { expected } => {
                write!(f, "handshake: the first message was not {expected}")
            }
            HandshakeError::Build { theirs, ours } => write!(
                f,
                "handshake: the other side is build {}…, and this is build {}…",
                Short(theirs),
                Short(ours)
            ),
            HandshakeError::Protocol { theirs, ours } => write!(
                f,
                "handshake: the runner speaks ipc_protocol {}.{}, and this build speaks {}.{}",
                theirs.major, theirs.minor, ours.major, ours.minor
            ),
            HandshakeError::Pid { claimed, spawned } => write!(
                f,
                "handshake: the Hello names process {claimed}, and the runner spawned is {spawned}"
            ),
        }
    }
}

impl std::error::Error for HandshakeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            HandshakeError::Send(error) => Some(error),
            HandshakeError::Recv(error) => Some(error),
            _ => None,
        }
    }
}

/// A spawned runner the daemon refused at the handshake, and killed.
#[derive(Debug)]
pub struct Refused {
    /// Why it was refused.
    pub cause: HandshakeError,
    /// How it ended: its exit status after the kill, or why it could not be killed or reaped.
    pub exit: io::Result<ExitStatus>,
}

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.exit {
            Ok(status) => write!(f, "{}; the runner was killed ({status})", self.cause),
            Err(error) => write!(f, "{}; killing the runner failed: {error}", self.cause),
        }
    }
}

impl std::error::Error for Refused {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

#[cfg(test)]
mod tests;
