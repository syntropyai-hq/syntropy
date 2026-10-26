// SPDX-License-Identifier: MIT OR Apache-2.0

//! The loom model of the region's two local rings (RFC-0060 §3.4; QUAL-E3-T09), built with
//! `RUSTFLAGS="--cfg loom"`.
//!
//! Three threads share one region through two mappings of its segment, as the daemon and the
//! runner do. The runner pushes `local_tokens` and then pops `local_sequences`, one daemon thread
//! pops `local_tokens`, and the other pushes `local_sequences`. Each ring has one slot, so the
//! second record through it takes the slot the first left. Each thread makes a fixed number of
//! attempts that never wait, so loom runs every interleaving within a preemption bound of 3 to its
//! end. Spinning would not: in loom 0.7 a yielding thread hands over without spending a
//! preemption, so two threads spinning on the third can pass control between them forever, and
//! the run fails at loom's branch limit (RFC-0060 §3.4, note of 2026-10-23). The main thread then
//! finishes both rings through new mappings, as a process that reattaches does, and every record
//! must arrive once and in order. loom also checks, on each ring's shadow cells, that a slot's
//! write is ordered before its read and its read before the next write. The planted relaxed store
//! is found in the ring's own crate (`syntropy_spsc::shared`'s tests), on the same ring type.

use std::sync::Arc;

use loom::thread;
use syntropy_spsc::shared::{SharedPopError, SharedPushError};

use super::*;

const PROTOCOL: IpcProtocol = IpcProtocol { major: 1, minor: 0 };
const BUILD: [u8; 32] = [7; 32];

/// The token frames the runner sends and the sequence controls the daemon sends, as one byte each.
const TOKENS: [u8; 2] = [0, 1];
const SEQUENCES: [u8; 2] = [10, 11];

/// Segment names for the model: this process's ID and a counter. Only Apple platforms and Windows
/// name a segment, and each run of the model makes a new one.
struct CountingEntropy(AtomicU64);

impl Entropy for CountingEntropy {
    fn fill(&self, buf: &mut [u8]) {
        let n = self.0.fetch_add(1, Ordering::Relaxed);
        let bytes = (u64::from(std::process::id()) << 32 | n).to_le_bytes();
        for (at, byte) in buf.iter_mut().enumerate() {
            *byte = bytes[at % bytes.len()];
        }
    }
}

static ENTROPY: CountingEntropy = CountingEntropy(AtomicU64::new(0));

/// Runs `test` under loom with a preemption bound of 3, unless `LOOM_MAX_PREEMPTIONS` sets one.
fn model(test: impl Fn() + Sync + Send + 'static) {
    let mut builder = loom::model::Builder::new();
    if builder.preemption_bound.is_none() {
        builder.preemption_bound = Some(3);
    }
    builder.check(test);
}

/// The two local rings, one slot each.
fn layout() -> ShmLayout {
    let ring = |role| RingSpec {
        role,
        slots: 1,
        slot_bytes: 8,
    };
    ShmLayout {
        rings: vec![ring(RingRole::LocalTokens), ring(RingRole::LocalSequences)],
    }
}

/// Pushes `records` in order until the ring refuses one, and returns how many went.
fn push_some(tx: &mut SharedProducer<'_>, records: &[u8]) -> usize {
    let mut sent = 0;
    for record in records {
        match tx.push(&[*record]) {
            Ok(()) => sent += 1,
            Err(SharedPushError::Full | SharedPushError::Closed) => break,
            Err(other) => panic!("{other}"),
        }
    }
    sent
}

/// Pops one record, if the ring has one.
fn pop_one(rx: &mut SharedConsumer<'_>) -> Option<u8> {
    let mut record = [0u8; 8];
    match rx.pop_into(&mut record) {
        Ok(len) => {
            assert_eq!(len, 1);
            Some(record[0])
        }
        Err(SharedPopError::Empty | SharedPopError::Closed) => None,
        Err(other) => panic!("{other}"),
    }
}

/// Makes `attempts` pops, and returns the records they read.
fn pop_some(rx: &mut SharedConsumer<'_>, attempts: usize) -> Vec<u8> {
    (0..attempts).filter_map(|_| pop_one(rx)).collect()
}

/// Finishes one ring in one thread: reads what it still holds, then pushes each of `rest` and
/// reads it back. The caller takes `rx` before `tx`, so `tx` finds a consumer.
fn finish(mut rx: SharedConsumer<'_>, mut tx: SharedProducer<'_>, rest: &[u8], seen: &mut Vec<u8>) {
    while let Some(record) = pop_one(&mut rx) {
        seen.push(record);
    }
    for record in rest {
        tx.push(&[*record]).unwrap();
        seen.push(pop_one(&mut rx).unwrap());
    }
}

/// Tokens go runner to daemon and sequence controls daemon to runner, each once and in order,
/// across three threads and three mappings of the region.
#[test]
fn the_local_rings_carry_records_both_ways_across_three_threads() {
    model(|| {
        let daemon = Arc::new(Region::create(&layout(), BUILD, PROTOCOL, &ENTROPY).unwrap());
        let handle = daemon.handle();
        let runner = Region::open(&handle, Side::Runner, &layout(), PROTOCOL).unwrap();
        let runner = thread::spawn(move || {
            let mut tokens = runner.producer(RingRole::LocalTokens).unwrap();
            let sent = push_some(&mut tokens, &TOKENS);
            drop(tokens);
            let mut sequences = runner.consumer(RingRole::LocalSequences).unwrap();
            (sent, pop_some(&mut sequences, 2))
        });
        let tokens_side = Arc::clone(&daemon);
        let tokens = thread::spawn(move || {
            let mut tokens = tokens_side.consumer(RingRole::LocalTokens).unwrap();
            pop_some(&mut tokens, 2)
        });
        let mut sequences = daemon.producer(RingRole::LocalSequences).unwrap();
        let sequences_sent = push_some(&mut sequences, &SEQUENCES);
        drop(sequences);
        let mut tokens_seen = tokens.join().unwrap();
        let (tokens_sent, mut sequences_seen) = runner.join().unwrap();

        // The rest of the run, through ends taken from new mappings, as a process that reattaches
        // takes them (RFC-0052 §6.5). Each picks up from the indices in the ring's header.
        let runner_again = Region::open(&handle, Side::Runner, &layout(), PROTOCOL).unwrap();
        let daemon_again = Region::open(&handle, Side::Daemon, &layout(), PROTOCOL).unwrap();
        let rx = daemon_again.consumer(RingRole::LocalTokens).unwrap();
        let tx = runner_again.producer(RingRole::LocalTokens).unwrap();
        finish(rx, tx, &TOKENS[tokens_sent..], &mut tokens_seen);
        let rx = runner_again.consumer(RingRole::LocalSequences).unwrap();
        let tx = daemon_again.producer(RingRole::LocalSequences).unwrap();
        finish(rx, tx, &SEQUENCES[sequences_sent..], &mut sequences_seen);
        assert_eq!(tokens_seen, TOKENS);
        assert_eq!(sequences_seen, SEQUENCES);
    });
}
