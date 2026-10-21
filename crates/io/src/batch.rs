// SPDX-License-Identifier: MIT OR Apache-2.0

//! The preallocated buffers the I/O thread passes to its engine (RFC-0021 §2, §3.1).
//!
//! [`SubmitBatch`] and [`CompletionBuf`] are allocated once per I/O thread at `Warming`, with a
//! fixed capacity. After that nothing in them allocates: a push into a full buffer returns
//! [`Full`] with the item instead of growing (RFC-0021 goal 6; ADR-0012 items 4 and 6).

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

/// Operations waiting for [`IoEngine::submit`](crate::IoEngine::submit), in submission order.
///
/// The I/O thread pushes chunks highest priority first (RFC-0021 §3.2). The engine accepts a
/// prefix and removes it with [`SubmitBatch::consume`]; the rest stay for the next call.
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

    /// Appends an operation, or returns it in [`Full`] when the batch is full. Never allocates.
    pub fn push(&mut self, op: IoOp) -> Result<(), Full<IoOp>> {
        if self.is_full() {
            return Err(Full(op));
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
    use super::*;
    use crate::engine::BufIndex;
    use crate::engine::FileIx;
    use crate::error::{IoError, IoErrorCode};
    use crate::request::{ExtentRef, IoDest, IoPriority, IoTicket, ReadRequest};
    use crate::units::{Bytes, Nanos};

    fn read(ticket: u64) -> IoOp {
        IoOp::Read(ReadRequest {
            ticket: IoTicket(ticket),
            extent: ExtentRef {
                file: FileIx(0),
                offset: ticket * 4096,
                len: Bytes(4096),
            },
            // SAFETY: these tests never submit a batch to an engine, so no memory is accessed
            // through this destination.
            dest: unsafe { IoDest::fixed(BufIndex(0), ticket * 4096) },
            priority: IoPriority::Demand,
            useful_until: None,
        })
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
        assert_eq!(b.push(read(3)), Err(Full(read(3))));
        assert_eq!(b.len(), 3);
        assert_eq!(b.capacity(), 3);
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
