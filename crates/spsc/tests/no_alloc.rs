// SPDX-License-Identifier: MIT OR Apache-2.0

//! The ring allocates nothing in steady state (ADR-0012 item 4; RFC-0060 §3.5): every slot exists
//! from the start, and pushing, popping, waiting and waking allocate on neither thread.
//!
//! `CountingAlloc` counts each thread's allocations, so each thread checks its own count.

#![cfg(not(loom))]

use std::thread;
use std::time::Duration;

use syntropy_spsc::{PushError, ring};
use syntropy_testkit::guards::{CountingAlloc, alloc_count_this_thread};

#[global_allocator]
static ALLOCATOR: CountingAlloc = CountingAlloc;

/// The allocator counts at all: without that, a zero below would prove nothing.
#[test]
fn the_allocator_counts() {
    let before = alloc_count_this_thread();
    let boxed = Box::new(1_u64);
    assert_eq!(alloc_count_this_thread() - before, 1);
    drop(boxed);
}

#[test]
fn pushing_and_popping_on_one_thread_allocates_nothing() {
    let (mut tx, mut rx) = ring::<[u64; 4]>(64).unwrap();
    let before = alloc_count_this_thread();
    for value in 0..100_000u64 {
        tx.push([value; 4]).unwrap();
        if value % 3 == 0 {
            let _ = tx.push([value; 4]);
            assert_eq!(rx.pop().map(|v| v[0]), Ok(value));
        }
        assert_eq!(rx.pop().map(|v| v[0]), Ok(value));
    }
    assert_eq!(alloc_count_this_thread() - before, 0);
}

/// Two threads, with the consumer parking and the producer waking it. Each thread counts from
/// after its first round, once the consumer's thread has waited and the threads' own setup is
/// done.
#[test]
fn a_parked_consumer_and_its_producer_allocate_nothing() {
    const ROUNDS: u64 = 2_000;
    let (mut tx, mut rx) = ring::<u64>(8).unwrap();
    let consumer = thread::spawn(move || {
        assert_eq!(rx.pop_wait(Duration::ZERO), Some(0));
        let before = alloc_count_this_thread();
        for expected in 1..ROUNDS {
            assert_eq!(rx.pop_wait(Duration::ZERO), Some(expected));
        }
        assert_eq!(rx.pop_wait(Duration::ZERO), None);
        alloc_count_this_thread() - before
    });
    let mut push = |value: u64| loop {
        match tx.push(value) {
            Ok(()) => break,
            Err(PushError::Full(_)) => std::hint::spin_loop(),
            Err(PushError::Closed(_)) => panic!("the consumer closed the ring"),
        }
    };
    push(0);
    let before = alloc_count_this_thread();
    for value in 1..ROUNDS {
        // Every eighth value waits a little, so the consumer runs dry and parks.
        if value % 8 == 0 {
            for _ in 0..2_000 {
                std::hint::spin_loop();
            }
        }
        push(value);
    }
    let producer_allocations = alloc_count_this_thread() - before;
    drop(tx);
    assert_eq!(producer_allocations, 0, "the producer allocated");
    assert_eq!(consumer.join().unwrap(), 0, "the consumer allocated");
}
