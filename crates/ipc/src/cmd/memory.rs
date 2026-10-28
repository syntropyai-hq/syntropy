// SPDX-License-Identifier: MIT OR Apache-2.0

//! The command channel's in-memory transport: a connected pair of byte streams in one process
//! (RFC-0001 §3.1; ADR-0015 item 7; DIST-E1-T11).
//!
//! On a platform that cannot spawn a runner process, and in the simulator, the runner runs on its
//! own threads in the daemon's process. Its command channel is a [`Channel`](super::Channel) over
//! a [`MemoryStream`] instead of an OS [`Stream`](super::Stream). The framing and the encoding
//! are the same, so the two transports are interchangeable: a frame one writes is the frame the
//! other carries.
//!
//! - [`pair`] gives the two connected ends. What one end writes, the other reads, in order.
//! - Dropping an end closes it: the other end reads end-of-file once it has read what was
//!   written before the drop, which [`Channel::recv`](super::Channel::recv) reports as
//!   [`RecvError::Closed`](super::RecvError::Closed). A write to a closed peer fails with
//!   `BrokenPipe`.
//! - **The buffer is unbounded.** A write never blocks or fails for space. The command channel
//!   carries control messages only, never per-token traffic (RFC-0001 §3.5), so what is queued is
//!   bounded by the protocol's own message rate. The simulator drives both ends from one thread
//!   and drains each end as it steps, so it needs no backpressure here. Backpressure on bulk data
//!   belongs to the shared-memory rings, which are fixed-size (RFC-0001 §3.5, §8.3). No RFC asks
//!   for a bound on this channel.
//! - A read blocks until bytes or end-of-file arrive. [`MemoryStream::set_nonblocking`] makes it
//!   return `WouldBlock` instead, for a caller that drives both ends from one thread, as the
//!   simulator does (RFC-0060 §2.3). A [`Channel`](super::Channel) writes each frame in one write
//!   call, which this stream takes whole, so a reader never sees half a frame.

use std::collections::VecDeque;
use std::fmt;
use std::io::{self, Read, Write};
use std::sync::{Arc, Condvar, Mutex, PoisonError};

/// The bytes travelling one way, and whether their writer has gone.
#[derive(Default)]
struct Pipe {
    bytes: VecDeque<u8>,
    closed: bool,
}

/// One direction: a pipe and the condition its reader waits on.
#[derive(Default)]
struct Half {
    pipe: Mutex<Pipe>,
    ready: Condvar,
}

/// One end of an in-memory command-channel transport. See the [module docs](self).
pub struct MemoryStream {
    /// What this end reads.
    inbound: Arc<Half>,
    /// What this end writes.
    outbound: Arc<Half>,
    nonblocking: bool,
}

/// Two connected [`MemoryStream`]s.
#[must_use]
pub fn pair() -> (MemoryStream, MemoryStream) {
    let a_to_b = Arc::new(Half::default());
    let b_to_a = Arc::new(Half::default());
    (
        MemoryStream {
            inbound: Arc::clone(&b_to_a),
            outbound: Arc::clone(&a_to_b),
            nonblocking: false,
        },
        MemoryStream {
            inbound: a_to_b,
            outbound: b_to_a,
            nonblocking: false,
        },
    )
}

impl MemoryStream {
    /// Makes a read with nothing to read return `WouldBlock` rather than wait.
    pub fn set_nonblocking(&mut self, nonblocking: bool) {
        self.nonblocking = nonblocking;
    }

    /// The bytes written by the other end and not yet read.
    #[must_use]
    pub fn available(&self) -> usize {
        self.inbound
            .pipe
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .bytes
            .len()
    }
}

impl Read for MemoryStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let mut pipe = self
            .inbound
            .pipe
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        loop {
            if !pipe.bytes.is_empty() {
                let n = buf.len().min(pipe.bytes.len());
                for (slot, byte) in buf.iter_mut().zip(pipe.bytes.drain(..n)) {
                    *slot = byte;
                }
                return Ok(n);
            }
            if pipe.closed {
                return Ok(0);
            }
            if self.nonblocking {
                return Err(io::ErrorKind::WouldBlock.into());
            }
            pipe = self
                .inbound
                .ready
                .wait(pipe)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }
}

impl Write for MemoryStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut pipe = self
            .outbound
            .pipe
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if pipe.closed {
            return Err(io::ErrorKind::BrokenPipe.into());
        }
        pipe.bytes.extend(buf);
        self.outbound.ready.notify_all();
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Drop for MemoryStream {
    /// Closes both directions: the other end reads end-of-file after what was written, and its
    /// writes fail.
    fn drop(&mut self) {
        for half in [&self.outbound, &self.inbound] {
            half.pipe
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .closed = true;
            half.ready.notify_all();
        }
    }
}

impl fmt::Debug for MemoryStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryStream")
            .field("available", &self.available())
            .field("nonblocking", &self.nonblocking)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;

    #[test]
    fn bytes_cross_in_order_and_a_drop_is_end_of_file() {
        let (mut a, mut b) = pair();
        a.write_all(b"hello ").unwrap();
        a.write_all(b"world").unwrap();
        let mut got = [0u8; 11];
        b.read_exact(&mut got).unwrap();
        assert_eq!(&got, b"hello world");
        b.write_all(b"back").unwrap();
        drop(b);
        // What was written before the drop still arrives, then end-of-file.
        let mut back = Vec::new();
        a.read_to_end(&mut back).unwrap();
        assert_eq!(back, b"back");
        assert_eq!(a.write(b"x").unwrap_err().kind(), io::ErrorKind::BrokenPipe);
    }

    #[test]
    fn a_blocking_read_waits_for_the_writer_and_a_nonblocking_one_does_not() {
        let (mut a, mut b) = pair();
        let reader = thread::spawn(move || {
            let mut buf = [0u8; 3];
            b.read_exact(&mut buf).unwrap();
            buf
        });
        a.write_all(b"abc").unwrap();
        assert_eq!(&reader.join().unwrap(), b"abc");
        let (_c, mut d) = pair();
        d.set_nonblocking(true);
        assert_eq!(
            d.read(&mut [0u8; 1]).unwrap_err().kind(),
            io::ErrorKind::WouldBlock
        );
        assert_eq!(d.available(), 0);
    }
}
