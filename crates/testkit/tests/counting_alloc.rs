// SPDX-License-Identifier: MIT OR Apache-2.0

//! QUAL-E3-T25's acceptance tests, in a test binary that installs [`CountingAlloc`] as its
//! global allocator, as RFC-0060 §3.5 has every test binary that counts do.
//!
//! Each measured window holds only the code under test: no formatting, no assertion and no
//! synchronisation that might allocate. Threads hand off through atomics, which never allocate.

use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::thread;

use syntropy_testkit::guards::{CountingAlloc, alloc_count_this_thread};

#[global_allocator]
static ALLOCATOR: CountingAlloc = CountingAlloc;

/// Makes `n` heap allocations: each `Box::new` allocates once, and dropping frees it.
fn allocate(n: u64) {
    for i in 0..n {
        drop(black_box(Box::new(i)));
    }
}

/// Waits until `flag` reads `value`.
fn wait_for(flag: &AtomicU8, value: u8) {
    while flag.load(Ordering::Acquire) != value {
        thread::yield_now();
    }
}

#[test]
fn n_allocations_on_one_thread_read_n_more_there_and_none_on_another() {
    const N: u64 = 1000;
    // 0: start; 1: the watcher has read its count; 2: the allocator has finished.
    let stage = Arc::new(AtomicU8::new(0));

    let watcher = {
        let stage = Arc::clone(&stage);
        thread::spawn(move || {
            let before = alloc_count_this_thread();
            stage.store(1, Ordering::Release);
            wait_for(&stage, 2);
            alloc_count_this_thread() - before
        })
    };
    let allocator = {
        let stage = Arc::clone(&stage);
        thread::spawn(move || {
            wait_for(&stage, 1);
            let before = alloc_count_this_thread();
            allocate(N);
            let after = alloc_count_this_thread();
            stage.store(2, Ordering::Release);
            after - before
        })
    };

    let on_the_allocating_thread = allocator.join().unwrap();
    let on_the_other_thread = watcher.join().unwrap();
    assert_eq!(on_the_allocating_thread, N);
    assert_eq!(on_the_other_thread, 0);
}

#[test]
// The plant is a `Vec::push`, the call the criterion names; `vec![i]` would make the same
// allocation without it.
#[allow(clippy::vec_init_then_push)]
fn a_planted_vec_push_inside_a_measured_loop_is_counted() {
    const ROUNDS: u64 = 64;
    let mut scratch: Vec<u64> = Vec::with_capacity(ROUNDS as usize);

    // The loop as it should be: it writes only into storage reserved before it.
    let before = alloc_count_this_thread();
    for i in 0..ROUNDS {
        scratch.push(black_box(i));
    }
    let clean = alloc_count_this_thread() - before;

    // The same loop with a planted `Vec::push` into a new vector each round. One push into an
    // empty `Vec` allocates exactly once and never grows, so the count rests on `alloc` alone.
    scratch.clear();
    let before = alloc_count_this_thread();
    for i in 0..ROUNDS {
        scratch.push(black_box(i));
        let mut planted: Vec<u64> = Vec::new();
        planted.push(black_box(i));
        black_box(&planted);
    }
    let with_the_plant = alloc_count_this_thread() - before;

    assert_eq!(clean, 0, "the loop allocates without the plant");
    assert_eq!(
        with_the_plant, ROUNDS,
        "each planted push should count once ({with_the_plant} allocations in {ROUNDS} rounds)"
    );
}

#[test]
fn zeroed_allocations_and_growth_count_too() {
    let before = alloc_count_this_thread();
    let zeroed = black_box(vec![0_u8; 256]); // alloc_zeroed
    let after_zeroed = alloc_count_this_thread();
    let mut growing = black_box(Vec::<u8>::with_capacity(1)); // alloc
    growing.reserve_exact(4096); // realloc
    let after_growth = alloc_count_this_thread();
    drop((zeroed, growing));
    let after_drop = alloc_count_this_thread();

    assert_eq!(after_zeroed - before, 1);
    assert_eq!(after_growth - after_zeroed, 2);
    assert_eq!(after_drop, after_growth, "freeing memory is not counted");
}
