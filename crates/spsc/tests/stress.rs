// SPDX-License-Identifier: MIT OR Apache-2.0

//! The ring under load across two threads: every value arrives exactly once and in order
//! (DIST-E1-T01). The consumer waits with `pop_wait`, so the runs also cross the park and wake
//! path whenever the consumer catches up with the producer.
//!
//! The 10⁸-message run is `#[ignore]`d, because it takes seconds in a release build and much
//! longer in a debug one:
//!
//! ```text
//! cargo test -p syntropy-spsc --release --test stress -- --ignored
//! ```

#![cfg(not(loom))]

use std::thread;

use syntropy_spsc::{PushError, SPIN_DEFAULT, ring};

/// Sends `0..count` through a ring of `capacity` slots and checks what arrives.
fn send_in_order(count: u64, capacity: usize) {
    let (mut tx, mut rx) = ring::<u64>(capacity).unwrap();
    let producer = thread::spawn(move || {
        for value in 0..count {
            let mut value = value;
            loop {
                match tx.push(value) {
                    Ok(()) => break,
                    Err(PushError::Full(back)) => {
                        value = back;
                        std::hint::spin_loop();
                    }
                    Err(PushError::Closed(_)) => panic!("the consumer closed the ring"),
                }
            }
        }
    });
    let mut expected = 0u64;
    while let Some(value) = rx.pop_wait(SPIN_DEFAULT) {
        if value != expected {
            panic!("message {expected} arrived as {value}: lost or reordered");
        }
        expected += 1;
    }
    producer.join().unwrap();
    assert_eq!(expected, count, "messages lost at the end of the run");
}

#[test]
fn a_million_messages_arrive_once_and_in_order() {
    send_in_order(1_000_000, 1024);
}

/// A two-slot ring is full or empty almost all the time, so most pushes and pops reload the
/// other end's index.
#[test]
fn a_two_slot_ring_keeps_a_million_messages_in_order() {
    send_in_order(1_000_000, 2);
}

#[test]
#[ignore = "10⁸ messages: run with --release --ignored"]
fn a_hundred_million_messages_arrive_once_and_in_order() {
    send_in_order(100_000_000, 1024);
}
