// SPDX-License-Identifier: MIT OR Apache-2.0

//! The shared ring's tests, over heap memory laid out as a shared segment would be. Like the heap
//! ring's, they run inside `model`, a plain call normally and loom's model checker under
//! `--cfg loom`. loom tracks the index atomics, not the plain slot bytes.

use std::alloc::{self, Layout};

use super::*;
use crate::sync::thread;

/// Runs `test` once, or under loom with a preemption bound of 3 unless `LOOM_MAX_PREEMPTIONS`
/// sets one.
fn model(test: impl Fn() + Sync + Send + 'static) {
    #[cfg(loom)]
    {
        let mut builder = loom::model::Builder::new();
        if builder.preemption_bound.is_none() {
            builder.preemption_bound = Some(3);
        }
        builder.check(test);
    }
    #[cfg(not(loom))]
    test();
}

/// Ring memory on the heap, aligned as a mapped segment is, with the header written.
struct Memory {
    base: NonNull<u8>,
    layout: Layout,
}

// SAFETY: the memory is a plain heap block; the ends that use it carry their own `Send`.
unsafe impl Send for Memory {}
// SAFETY: as above; shared access goes through the ends.
unsafe impl Sync for Memory {}

impl Memory {
    fn new(ring: SharedLayout) -> Memory {
        let layout = Layout::from_size_align(ring.bytes(), SharedLayout::ALIGN).unwrap();
        // SAFETY: the layout has a non-zero size, since a ring has a header.
        let base = NonNull::new(unsafe { alloc::alloc_zeroed(layout) }).unwrap();
        // SAFETY: the block is `ring.bytes()` long, 128-aligned, and nothing uses it yet.
        unsafe { ring.init(base) };
        Memory { base, layout }
    }

    fn header(&self) -> &Header {
        // SAFETY: `new` wrote the header, and it lives as long as `self`.
        unsafe { self.base.cast::<Header>().as_ref() }
    }

    fn producer(&self, ring: SharedLayout) -> SharedProducer<'_> {
        // SAFETY: the memory holds a ring of this layout for `self`'s lifetime, and each test makes
        // one producer at a time.
        unsafe { ring.producer(self.base) }
    }

    fn consumer(&self, ring: SharedLayout) -> SharedConsumer<'_> {
        // SAFETY: as for `producer`.
        unsafe { ring.consumer(self.base) }
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        // SAFETY: the header was written by `new` and no end outlives `self`; the block came from
        // `alloc_zeroed` with this layout.
        unsafe {
            std::ptr::drop_in_place(self.base.cast::<Header>().as_ptr());
            alloc::dealloc(self.base.as_ptr(), self.layout);
        }
    }
}

fn pop(consumer: &mut SharedConsumer<'_>) -> Result<Vec<u8>, SharedPopError> {
    let mut buffer = [0u8; 64];
    let len = consumer.pop_into(&mut buffer)?;
    Ok(buffer[..len].to_vec())
}

#[test]
fn records_come_out_in_the_order_they_went_in() {
    model(|| {
        let ring = SharedLayout::new(4, 16).unwrap();
        let memory = Memory::new(ring);
        let mut tx = memory.producer(ring);
        let mut rx = memory.consumer(ring);
        assert_eq!(pop(&mut rx), Err(SharedPopError::Empty));
        tx.push(b"one").unwrap();
        tx.push(b"").unwrap();
        tx.push(b"three").unwrap();
        assert_eq!(pop(&mut rx).unwrap(), b"one");
        assert_eq!(pop(&mut rx).unwrap(), b"");
        assert_eq!(pop(&mut rx).unwrap(), b"three");
        assert_eq!(pop(&mut rx), Err(SharedPopError::Empty));
    });
}

#[test]
fn a_full_ring_refuses_and_a_long_record_never_fits() {
    model(|| {
        let ring = SharedLayout::new(2, 8).unwrap();
        let memory = Memory::new(ring);
        let mut tx = memory.producer(ring);
        let mut rx = memory.consumer(ring);
        assert_eq!(
            tx.push(&[0; 9]),
            Err(SharedPushError::TooLong { len: 9, max: 8 })
        );
        tx.push(&[1; 8]).unwrap();
        tx.push(&[2; 8]).unwrap();
        assert_eq!(tx.push(&[3]), Err(SharedPushError::Full));
        assert_eq!(pop(&mut rx).unwrap(), [1; 8]);
        tx.push(&[3]).unwrap();
        assert_eq!(pop(&mut rx).unwrap(), [2; 8]);
        assert_eq!(pop(&mut rx).unwrap(), [3]);
    });
}

/// A record longer than the caller's buffer stays in the ring for a larger buffer.
#[test]
fn a_record_too_long_for_the_buffer_stays_in_the_ring() {
    model(|| {
        let ring = SharedLayout::new(2, 32).unwrap();
        let memory = Memory::new(ring);
        let mut tx = memory.producer(ring);
        let mut rx = memory.consumer(ring);
        tx.push(&[7; 20]).unwrap();
        let mut small = [0u8; 10];
        assert_eq!(
            rx.pop_into(&mut small),
            Err(SharedPopError::BufferTooSmall { len: 20 })
        );
        assert_eq!(pop(&mut rx).unwrap(), [7; 20]);
    });
}

#[test]
fn indices_wrap_around_the_slots() {
    model(|| {
        let ring = SharedLayout::new(4, 8).unwrap();
        let memory = Memory::new(ring);
        let mut tx = memory.producer(ring);
        let mut rx = memory.consumer(ring);
        for round in 0u8..10 {
            for at in 0..4 {
                tx.push(&[round, at]).unwrap();
            }
            for at in 0..4 {
                assert_eq!(pop(&mut rx).unwrap(), [round, at]);
            }
        }
    });
}

#[test]
fn the_indices_wrap_past_u64_max() {
    model(|| {
        let ring = SharedLayout::new(4, 8).unwrap();
        let memory = Memory::new(ring);
        let start = u64::MAX - 5;
        memory.header().head.store(start, Ordering::Relaxed);
        memory.header().tail.store(start, Ordering::Relaxed);
        let mut tx = memory.producer(ring);
        let mut rx = memory.consumer(ring);
        for value in 0u8..12 {
            tx.push(&[value]).unwrap();
            assert_eq!(pop(&mut rx).unwrap(), [value]);
        }
    });
}

#[test]
fn a_closed_producer_leaves_its_records_to_pop_first() {
    model(|| {
        let ring = SharedLayout::new(4, 8).unwrap();
        let memory = Memory::new(ring);
        let mut tx = memory.producer(ring);
        let mut rx = memory.consumer(ring);
        tx.push(b"a").unwrap();
        drop(tx);
        assert!(rx.is_closed());
        assert_eq!(pop(&mut rx).unwrap(), b"a");
        assert_eq!(pop(&mut rx), Err(SharedPopError::Closed));
    });
}

#[test]
fn a_closed_consumer_refuses_pushes_until_a_consumer_reattaches() {
    model(|| {
        let ring = SharedLayout::new(4, 8).unwrap();
        let memory = Memory::new(ring);
        let mut tx = memory.producer(ring);
        let mut rx = memory.consumer(ring);
        tx.push(b"kept").unwrap();
        drop(rx);
        assert!(tx.is_closed());
        assert_eq!(tx.push(b"x"), Err(SharedPushError::Closed));
        // A new consumer, as a readopting daemon would make, picks up where the last one stopped.
        rx = memory.consumer(ring);
        assert!(!tx.is_closed());
        assert_eq!(pop(&mut rx).unwrap(), b"kept");
        tx.push(b"y").unwrap();
        assert_eq!(pop(&mut rx).unwrap(), b"y");
    });
}

/// A peer that writes indices or lengths outside the protocol is reported, and nothing is read or
/// written out of range.
#[test]
fn corrupt_indices_and_lengths_are_refused() {
    model(|| {
        let ring = SharedLayout::new(4, 8).unwrap();
        let memory = Memory::new(ring);
        let tx = memory.producer(ring);
        let mut rx = memory.consumer(ring);
        // The producer's index runs ahead of the consumer's by more than the ring holds.
        memory.header().tail.store(9, Ordering::Release);
        assert_eq!(pop(&mut rx), Err(SharedPopError::Corrupt));
        memory.header().tail.store(1, Ordering::Release);
        // A length above the slot size.
        // SAFETY: slot 0 is the first slot, past the header, and nothing else uses it now.
        unsafe {
            memory
                .base
                .add(header_bytes())
                .cast::<u32>()
                .write(9u32.to_le())
        };
        assert_eq!(pop(&mut rx), Err(SharedPopError::Corrupt));
        // The consumer's index runs behind the producer's by more than the ring holds, as a new
        // producer finds it.
        drop(tx);
        memory.header().head.store(0, Ordering::Release);
        memory.header().tail.store(10, Ordering::Release);
        let mut tx = memory.producer(ring);
        assert_eq!(tx.push(b"x"), Err(SharedPushError::Corrupt));
    });
}

#[test]
fn layouts_are_checked_and_sized() {
    for slots in [0, 3, SharedLayout::MAX_SLOTS * 2] {
        assert_eq!(SharedLayout::new(slots, 8), Err(LayoutError::Slots(slots)));
    }
    for bytes in [0, SharedLayout::MAX_SLOT_BYTES + 1] {
        assert_eq!(
            SharedLayout::new(4, bytes),
            Err(LayoutError::SlotBytes(bytes))
        );
    }
    let ring = SharedLayout::new(4, 13).unwrap();
    // Each slot is 8 bytes of length plus 13 bytes rounded up to 16.
    assert_eq!(
        ring.bytes(),
        (header_bytes() + 4 * 24).next_multiple_of(128)
    );
    assert_eq!(ring.bytes() % SharedLayout::ALIGN, 0);
}

/// Across two threads, records cross in order through a ring smaller than the run.
#[test]
fn records_cross_threads_in_order() {
    model(|| {
        let ring = SharedLayout::new(2, 8).unwrap();
        let memory = std::sync::Arc::new(Memory::new(ring));
        let shared = memory.clone();
        let producer = thread::spawn(move || {
            let mut tx = shared.producer(ring);
            for value in 0u8..3 {
                loop {
                    match tx.push(&[value]) {
                        Ok(()) => break,
                        Err(SharedPushError::Full) => thread::yield_now(),
                        Err(other) => panic!("{other}"),
                    }
                }
            }
        });
        let mut rx = memory.consumer(ring);
        let mut seen = Vec::new();
        loop {
            match pop(&mut rx) {
                Ok(record) => seen.push(record[0]),
                Err(SharedPopError::Empty) => thread::yield_now(),
                Err(SharedPopError::Closed) => break,
                Err(other) => panic!("{other}"),
            }
        }
        assert_eq!(seen, [0, 1, 2]);
        producer.join().unwrap();
    });
}

#[test]
fn errors_name_what_happened() {
    assert_eq!(
        SharedPushError::TooLong { len: 9, max: 8 }.to_string(),
        "a 9-byte record does not fit the shared ring's 8-byte slots"
    );
    assert_eq!(
        SharedPopError::BufferTooSmall { len: 20 }.to_string(),
        "the next record is 20 bytes, more than the buffer holds"
    );
    assert_eq!(
        LayoutError::Slots(3).to_string(),
        "a shared ring's slot count is a power of two from 1 to 16777216, not 3"
    );
}
