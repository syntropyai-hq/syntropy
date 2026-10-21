// SPDX-License-Identifier: MIT OR Apache-2.0

//! The preallocated buffers the I/O thread passes to its engine (RFC-0021 §2, §3.1).
//!
//! [`SubmitBatch`] and [`CompletionBuf`] are allocated once per I/O thread at `Warming`, with a
//! fixed capacity. After that nothing in them allocates: a push into a full buffer returns the
//! item instead of growing (RFC-0021 goal 6; ADR-0012 items 4 and 6).
//!
//! [`SubmitBatch::push`] is also the one check between a request and an engine that safe code
//! cannot get around: it refuses an operation whose extent is longer than its destination
//! (RFC-0021 §2).

use std::fmt;

use crate::request::{IoOp, ReadCompletion};

/// A push into a full [`SubmitBatch`] or [`CompletionBuf`]. Carries the item back.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Full<T>(pub T);

impl<T> fmt::Display for Full<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("preallocated I/O buffer is full")
    }
}

impl<T: fmt::Debug> std::error::Error for Full<T> {}

/// Why [`SubmitBatch::push`] refused an operation. Carries the operation back, unsubmitted.
#[derive(Debug, PartialEq, Eq, Hash)]
pub enum PushError {
    /// The batch is full. The operation can be pushed again once the engine accepts some.
    Full(IoOp),
    /// The operation's extent is longer than its destination's
    /// [`len`](crate::IoDest::len), so the engine would transfer past the memory the
    /// destination vouches for. A bug in the producer, of class `Internal`, like
    /// `IO_MISALIGNED` (RFC-0021 §16).
    ExceedsDest(IoOp),
}

impl PushError {
    /// The refused operation.
    #[must_use]
    pub fn into_op(self) -> IoOp {
        match self {
            PushError::Full(op) | PushError::ExceedsDest(op) => op,
        }
    }
}

impl fmt::Display for PushError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PushError::Full(_) => f.write_str("preallocated I/O buffer is full"),
            PushError::ExceedsDest(op) => write!(
                f,
                "extent of {} bytes exceeds its destination of {} bytes",
                op.extent().len.0,
                op.dest().len().0
            ),
        }
    }
}

impl std::error::Error for PushError {}

/// Operations waiting for [`IoEngine::submit`](crate::IoEngine::submit), in submission order.
///
/// The I/O thread pushes chunks highest priority first (RFC-0021 §3.2). The engine accepts a
/// prefix and removes it with [`SubmitBatch::consume`]; the rest stay for the next call. Every
/// operation in a batch fits its destination: `push` checks it.
#[derive(Debug)]
pub struct SubmitBatch {
    ops: Vec<IoOp>,
    cap: usize,
}

impl SubmitBatch {
    /// Allocates a batch that holds up to `cap` operations. The only allocation the batch makes.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            ops: Vec::with_capacity(cap),
            cap,
        }
    }

    /// The most operations the batch holds.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Operations waiting.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether no operation is waiting.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Whether the batch holds [`SubmitBatch::capacity`] operations.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.ops.len() >= self.cap
    }

    /// Appends an operation. Never allocates.
    ///
    /// Refuses, and returns, an operation whose extent is longer than its destination
    /// ([`PushError::ExceedsDest`]), and any operation once the batch is full
    /// ([`PushError::Full`]).
    pub fn push(&mut self, op: IoOp) -> Result<(), PushError> {
        if op.extent().len > op.dest().len() {
            return Err(PushError::ExceedsDest(op));
        }
        if self.is_full() {
            return Err(PushError::Full(op));
        }
        self.ops.push(op);
        Ok(())
    }

    /// The waiting operations, first to submit first.
    #[must_use]
    pub fn pending(&self) -> &[IoOp] {
        &self.ops
    }

    /// Removes the first `n` operations: the engine calls it for the prefix it accepted. Keeps
    /// the order of the rest and never allocates.
    ///
    /// # Panics
    ///
    /// If `n` exceeds [`SubmitBatch::len`].
    pub fn consume(&mut self, n: usize) {
        assert!(
            n <= self.ops.len(),
            "consume({n}) on a batch of {}",
            self.ops.len()
        );
        self.ops.drain(..n);
    }

    /// Removes every waiting operation.
    pub fn clear(&mut self) {
        self.ops.clear();
    }
}

/// Completions [`IoEngine::reap`](crate::IoEngine::reap) appends, in the order the engine
/// reaped them.
#[derive(Debug)]
pub struct CompletionBuf {
    done: Vec<ReadCompletion>,
    cap: usize,
}

impl CompletionBuf {
    /// Allocates a buffer that holds up to `cap` completions. The only allocation the buffer
    /// makes.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            done: Vec::with_capacity(cap),
            cap,
        }
    }

    /// The most completions the buffer holds.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Completions held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.done.len()
    }

    /// Whether the buffer holds no completion.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.done.is_empty()
    }

    /// Whether the buffer holds [`CompletionBuf::capacity`] completions.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.done.len() >= self.cap
    }

    /// How many more completions fit: the most a `reap` call may append.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.cap - self.done.len()
    }

    /// Appends a completion, or returns it in [`Full`] when the buffer is full. Never
    /// allocates.
    pub fn push(&mut self, c: ReadCompletion) -> Result<(), Full<ReadCompletion>> {
        if self.is_full() {
            return Err(Full(c));
        }
        self.done.push(c);
        Ok(())
    }

    /// The completions held, oldest first.
    #[must_use]
    pub fn as_slice(&self) -> &[ReadCompletion] {
        &self.done
    }

    /// Removes and yields every completion, oldest first. Never allocates. Dropping the iterator
    /// early still removes them all.
    pub fn drain(&mut self) -> impl Iterator<Item = ReadCompletion> {
        self.done.drain(..)
    }

    /// Removes every completion.
    pub fn clear(&mut self) {
        self.done.clear();
    }
}

#[cfg(test)]
mod tests {
    use syntropy_core::{Bytes, ExtentRef, FileIx, Nanos};

    use super::*;
    use crate::engine::BufIndex;
    use crate::error::{IoError, IoErrorCode};
    use crate::request::{IoDest, IoPriority, IoTicket, ReadRequest};

    /// A read of `extent_len` bytes into a destination of `dest_len` bytes.
    fn read_into(ticket: u64, extent_len: u64, dest_len: u64) -> IoOp {
        IoOp::Read(ReadRequest {
            ticket: IoTicket(ticket),
            extent: ExtentRef {
                file: FileIx(0),
                offset: ticket * 4096,
                len: Bytes(extent_len),
            },
            // SAFETY: no engine in this test process registers buffers, so the destination
            // names no memory, and these tests submit nothing.
            dest: unsafe { IoDest::fixed(BufIndex(0), ticket * 4096, Bytes(dest_len)) },
            priority: IoPriority::Demand,
            useful_until: None,
        })
    }

    fn read(ticket: u64) -> IoOp {
        read_into(ticket, 4096, 4096)
    }

    fn done(ticket: u64) -> ReadCompletion {
        ReadCompletion {
            ticket: IoTicket(ticket),
            result: Ok(Bytes(4096)),
            latency: Nanos(1),
        }
    }

    #[test]
    fn submit_batch_rejects_a_push_when_full() {
        let mut b = SubmitBatch::with_capacity(3);
        assert!(b.is_empty());
        for t in 0..3 {
            assert_eq!(b.push(read(t)), Ok(()));
        }
        assert!(b.is_full());
        assert_eq!(b.push(read(3)), Err(PushError::Full(read(3))));
        assert_eq!(b.len(), 3);
        assert_eq!(b.capacity(), 3);
    }

    #[test]
    fn submit_batch_refuses_an_extent_longer_than_its_destination() {
        let mut b = SubmitBatch::with_capacity(4);
        // A 1 GiB extent into a 4 KiB destination is refused and handed back unsubmitted.
        assert_eq!(
            b.push(read_into(0, 1 << 30, 4096)),
            Err(PushError::ExceedsDest(read_into(0, 1 << 30, 4096)))
        );
        assert_eq!(
            b.push(read_into(1, 4097, 4096)).map_err(PushError::into_op),
            Err(read_into(1, 4097, 4096))
        );
        assert!(b.is_empty());
        // An extent that fills its destination, or less, fits.
        b.push(read_into(2, 4096, 4096)).unwrap();
        b.push(read_into(3, 512, 4096)).unwrap();
        assert_eq!(b.len(), 2);
        // The length check comes first, so a full batch still reports the producer's bug.
        b.push(read(4)).unwrap();
        b.push(read(5)).unwrap();
        assert!(matches!(
            b.push(read_into(6, 8192, 4096)),
            Err(PushError::ExceedsDest(_))
        ));
    }

    #[test]
    fn push_errors_describe_themselves() {
        let too_long: Box<dyn std::error::Error> =
            Box::new(PushError::ExceedsDest(read_into(0, 8192, 4096)));
        assert_eq!(
            too_long.to_string(),
            "extent of 8192 bytes exceeds its destination of 4096 bytes"
        );
        let full: Box<dyn std::error::Error> = Box::new(PushError::Full(read(0)));
        assert_eq!(full.to_string(), "preallocated I/O buffer is full");
    }

    #[test]
    fn submit_batch_consume_keeps_the_rest_in_order() {
        let mut b = SubmitBatch::with_capacity(4);
        for t in 0..4 {
            b.push(read(t)).unwrap();
        }
        b.consume(0);
        assert_eq!(b.len(), 4);
        b.consume(2);
        let left: Vec<_> = b.pending().iter().map(IoOp::ticket).collect();
        assert_eq!(left, [IoTicket(2), IoTicket(3)]);
        // Room freed by consume is usable again, up to the same capacity.
        b.push(read(4)).unwrap();
        b.push(read(5)).unwrap();
        assert!(b.push(read(6)).is_err());
        b.clear();
        assert!(b.is_empty());
    }

    #[test]
    #[should_panic(expected = "consume(2) on a batch of 1")]
    fn submit_batch_consume_past_the_end_panics() {
        let mut b = SubmitBatch::with_capacity(2);
        b.push(read(0)).unwrap();
        b.consume(2);
    }

    #[test]
    fn zero_capacity_buffers_are_always_full() {
        let mut b = SubmitBatch::with_capacity(0);
        assert!(b.is_full());
        assert!(b.push(read(0)).is_err());
        let mut c = CompletionBuf::with_capacity(0);
        assert_eq!(c.remaining(), 0);
        assert!(c.push(done(0)).is_err());
    }

    #[test]
    fn completion_buf_rejects_a_push_when_full_and_drains_in_order() {
        let mut c = CompletionBuf::with_capacity(2);
        assert_eq!(c.remaining(), 2);
        c.push(done(1)).unwrap();
        let failed = ReadCompletion {
            ticket: IoTicket(2),
            result: Err(IoError::new(IoErrorCode::ShortRead)),
            latency: Nanos(5),
        };
        c.push(failed).unwrap();
        assert_eq!(c.remaining(), 0);
        assert_eq!(c.push(done(3)), Err(Full(done(3))));
        assert_eq!(c.as_slice(), [done(1), failed]);

        let drained: Vec<_> = c.drain().collect();
        assert_eq!(drained, [done(1), failed]);
        assert!(c.is_empty());
        assert_eq!(c.remaining(), 2);
    }

    #[test]
    fn dropping_drain_early_still_empties_the_buffer() {
        let mut c = CompletionBuf::with_capacity(3);
        for t in 0..3 {
            c.push(done(t)).unwrap();
        }
        assert_eq!(c.drain().next(), Some(done(0)));
        assert!(c.is_empty());
    }

    #[test]
    fn full_reports_itself_as_an_error() {
        let e: Box<dyn std::error::Error> = Box::new(Full(done(0)));
        assert_eq!(e.to_string(), "preallocated I/O buffer is full");
    }
}
