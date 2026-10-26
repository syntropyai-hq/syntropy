// SPDX-License-Identifier: MIT OR Apache-2.0

//! The statistics ring allocates nothing in steady state (RFC-0001 invariant 9; RFC-0023 §5.1):
//! every buffer exists from the start, and pushing routing records, dropping them on a full
//! ring and draining them reuse it.

use syntropy_core::LayerIx;
use syntropy_kernels::moe::RouteSlot;
use syntropy_testkit::guards::{CountingAlloc, alloc_count_this_thread};
use syntropy_tier::experts::{StatDrop, stats_ring};

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
fn pushing_dropping_and_draining_allocate_nothing() {
    const MAX_UNION: u16 = 32;
    let route: Vec<RouteSlot> = (0..MAX_UNION)
        .map(|e| RouteSlot {
            expert: e,
            tokens: 1,
            cache_slot: 0,
        })
        .collect();
    let (mut tx, mut rx) = stats_ring(4, MAX_UNION).unwrap();
    let mut seen = 0u64;
    let before = alloc_count_this_thread();
    for step in 0..10_000u64 {
        let n = (step % u64::from(MAX_UNION + 1)) as usize;
        for layer in 0..6u16 {
            match tx.push_route(step, step as u32, LayerIx(layer), 1, &route[..n]) {
                Ok(()) | Err(StatDrop::Full) => {}
                Err(e) => panic!("{e}"),
            }
        }
        if step % 3 == 0 {
            seen += rx.drain(|_| {}) as u64;
        }
    }
    assert_eq!(alloc_count_this_thread() - before, 0);
    // Both paths ran: records were read, and the full ring dropped some.
    assert!(seen > 0);
    assert!(rx.dropped() > 0);
}
