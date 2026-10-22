// SPDX-License-Identifier: MIT OR Apache-2.0

//! The ring's push, pop, wrap and close tests (RFC-0001 §8.3). Each runs inside [`model`]: a
//! plain call normally, and loom's model checker under `RUSTFLAGS="--cfg loom"`, which explores
//! the interleavings of the tests that use two threads. The loom models of RFC-0060 §3.4 are
//! QUAL-E3-T09's.

use std::sync::Arc as StdArc;

use super::*;
use crate::sync::thread;

/// Runs `test` once, or under loom's model checker with `--cfg loom`. A test that sets no
/// preemption bound, through `LOOM_MAX_PREEMPTIONS`, gets 3, which keeps each model to seconds.
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

/// Pushes `value`, yielding while the ring is full.
fn push_spinning<T>(producer: &mut Producer<T>, mut value: T) {
    loop {
        match producer.push(value) {
            Ok(()) => return,
            Err(PushError::Full(back)) => {
                value = back;
                thread::yield_now();
            }
            Err(PushError::Closed(_)) => panic!("the consumer closed the ring"),
        }
    }
}

/// Pops the next value, yielding while the ring is empty; `None` once it is closed.
fn pop_spinning<T>(consumer: &mut Consumer<T>) -> Option<T> {
    loop {
        match consumer.pop() {
            Ok(value) => return Some(value),
            Err(PopError::Empty) => thread::yield_now(),
            Err(PopError::Closed) => return None,
        }
    }
}

#[test]
fn values_come_out_in_the_order_they_went_in() {
    model(|| {
        let (mut tx, mut rx) = ring::<u32>(4).unwrap();
        assert_eq!(rx.pop(), Err(PopError::Empty));
        for value in [10, 11, 12] {
            tx.push(value).unwrap();
        }
        assert_eq!(rx.len(), 3);
        assert_eq!(rx.pop(), Ok(10));
        assert_eq!(rx.pop(), Ok(11));
        tx.push(13).unwrap();
        assert_eq!(rx.pop(), Ok(12));
        assert_eq!(rx.pop(), Ok(13));
        assert_eq!(rx.pop(), Err(PopError::Empty));
        assert!(rx.is_empty());
    });
}

#[test]
fn a_full_ring_refuses_the_push_and_hands_the_value_back() {
    model(|| {
        let (mut tx, mut rx) = ring::<String>(2).unwrap();
        tx.push("a".to_owned()).unwrap();
        tx.push("b".to_owned()).unwrap();
        let refused = tx.push("c".to_owned()).unwrap_err();
        assert!(matches!(refused, PushError::Full(_)));
        assert_eq!(refused.into_inner(), "c");
        assert_eq!(rx.pop().as_deref(), Ok("a"));
        tx.push("c".to_owned()).unwrap();
        assert_eq!(rx.pop().as_deref(), Ok("b"));
        assert_eq!(rx.pop().as_deref(), Ok("c"));
    });
}

/// The slots are reused as the indices go round, and the indices wrap past `usize::MAX`.
#[test]
fn indices_wrap_around_the_slots_and_past_usize_max() {
    model(|| {
        for start in [0, usize::MAX - 5] {
            let (mut tx, mut rx) = starting_at::<usize>(4, Waker::default(), start).unwrap();
            for round in 0..12 {
                // Fill the ring, then empty it, so each round crosses every slot.
                for at in 0..4 {
                    tx.push(round * 4 + at).unwrap();
                }
                assert!(matches!(tx.push(0), Err(PushError::Full(0))));
                for at in 0..4 {
                    assert_eq!(rx.pop(), Ok(round * 4 + at), "start {start}");
                }
                assert_eq!(rx.pop(), Err(PopError::Empty));
            }
        }
    });
}

#[test]
fn a_closed_producer_leaves_its_values_to_pop_first() {
    model(|| {
        let (mut tx, mut rx) = ring::<u8>(4).unwrap();
        tx.push(1).unwrap();
        tx.push(2).unwrap();
        drop(tx);
        assert_eq!(rx.pop(), Ok(1));
        assert_eq!(rx.pop(), Ok(2));
        assert_eq!(rx.pop(), Err(PopError::Closed));
        assert_eq!(rx.pop(), Err(PopError::Closed));
    });
}

#[test]
fn a_closed_consumer_refuses_pushes() {
    model(|| {
        let (mut tx, rx) = ring::<u8>(4).unwrap();
        assert!(!tx.is_closed());
        drop(rx);
        assert!(tx.is_closed());
        assert_eq!(tx.push(9).unwrap_err(), PushError::Closed(9));
    });
}

#[test]
fn unread_values_are_dropped_with_the_ring() {
    model(|| {
        let token = StdArc::new(());
        let (mut tx, mut rx) = ring::<StdArc<()>>(4).unwrap();
        for _ in 0..3 {
            tx.push(token.clone()).unwrap();
        }
        drop(rx.pop().unwrap());
        assert_eq!(StdArc::strong_count(&token), 3);
        drop(tx);
        drop(rx);
        assert_eq!(StdArc::strong_count(&token), 1);
    });
}

#[test]
fn a_capacity_is_a_power_of_two() {
    model(|| {
        for capacity in [0, 3, 6, 1000] {
            let error = ring::<u8>(capacity).unwrap_err();
            assert_eq!(error, CapacityError { capacity });
        }
        assert_eq!(ring::<u8>(usize::MAX).unwrap_err().capacity, usize::MAX);
        for capacity in [1, 2, 64] {
            let (tx, rx) = ring::<u8>(capacity).unwrap();
            assert_eq!((tx.capacity(), rx.capacity()), (capacity, capacity));
        }
        assert_eq!(
            CapacityError { capacity: 3 }.to_string(),
            format!("a ring's capacity is a power of two from 1 to {MAX_CAPACITY}, not 3")
        );
    });
}

/// Across two threads, through a ring smaller than the run, every value arrives once and in
/// order. Under loom this explores the interleavings of the full, empty and wrap cases.
#[test]
fn values_cross_threads_in_order() {
    model(|| {
        let (mut tx, mut rx) = ring::<u32>(2).unwrap();
        let producer = thread::spawn(move || {
            for value in 0..3 {
                push_spinning(&mut tx, value);
            }
        });
        for expected in 0..3 {
            assert_eq!(pop_spinning(&mut rx), Some(expected));
        }
        producer.join().unwrap();
        assert_eq!(rx.pop(), Err(PopError::Closed));
    });
}

/// A consumer that sees the producer's close also sees every value pushed before it.
#[test]
fn the_close_follows_the_last_push_across_threads() {
    model(|| {
        let (mut tx, mut rx) = ring::<u32>(4).unwrap();
        let producer = thread::spawn(move || {
            tx.push(1).unwrap();
            tx.push(2).unwrap();
        });
        let mut seen = Vec::new();
        while let Some(value) = pop_spinning(&mut rx) {
            seen.push(value);
        }
        assert_eq!(seen, [1, 2]);
        producer.join().unwrap();
    });
}

/// Values pushed while the consumer is being dropped are dropped exactly once, by whichever end
/// is last.
#[test]
fn values_left_in_flight_are_dropped_once() {
    model(|| {
        let token = StdArc::new(());
        let (mut tx, rx) = ring::<StdArc<()>>(2).unwrap();
        let sent = token.clone();
        let producer = thread::spawn(move || {
            // The consumer may be gone already, in which case the value comes back.
            let _ = tx.push(sent);
        });
        drop(rx);
        producer.join().unwrap();
        assert_eq!(StdArc::strong_count(&token), 1);
    });
}

#[test]
fn errors_name_what_happened() {
    assert_eq!(PushError::Full(1).to_string(), "the ring is full");
    assert_eq!(
        PushError::Closed(1).to_string(),
        "the ring's consumer is gone"
    );
    assert_eq!(format!("{:?}", PushError::Full(1)), "Full(..)");
    assert_eq!(PopError::Empty.to_string(), "the ring is empty");
    assert_eq!(
        PopError::Closed.to_string(),
        "the ring is empty and its producer is gone"
    );
}
