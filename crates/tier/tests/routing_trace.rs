// SPDX-License-Identifier: MIT OR Apache-2.0

//! TIER-E8-T01's round trip (RFC-0060 §12.4; RFC-0023 §5.1): seeded synthetic `RouteSlot`
//! lists pushed through the statistics ring, written by the recorder on the housekeeping side,
//! read back by the reader as the same lists in step order.

use std::thread;

use syntropy_core::LayerIx;
use syntropy_kernels::moe::RouteSlot;
use syntropy_kernels::moe::mirror::MOE_MISS;
use syntropy_tier::experts::{
    MoeRouting, RouteRecord, StatDrop, StatRecord, TraceReader, TraceWriter, doorbell_layout,
    stats_ring,
};

/// A small seeded generator (SplitMix64), so the test needs no randomness from outside.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

const MOE: MoeRouting = MoeRouting {
    experts: 64,
    top_k: 4,
    predict_extra: 2,
};
const MOE_LAYERS: [u16; 3] = [1, 2, 5];

/// One pass of `tokens` tokens: each token picks `k` distinct experts, and the pass lists each
/// distinct expert once with its token count, in the order first selected, as the router does.
fn synthetic_pass(rng: &mut Rng, tokens: u32) -> Vec<RouteSlot> {
    let mut route: Vec<RouteSlot> = Vec::new();
    for _ in 0..tokens {
        let mut picked: Vec<u16> = Vec::new();
        while picked.len() < usize::from(MOE.top_k) {
            let e = rng.below(u64::from(MOE.experts)) as u16;
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        for e in picked {
            match route.iter_mut().find(|s| s.expert == e) {
                Some(s) => s.tokens += 1,
                None => route.push(RouteSlot {
                    expert: e,
                    tokens: 1,
                    cache_slot: rng.below(u64::from(MOE_MISS)) as u32,
                }),
            }
        }
    }
    route
}

/// Seeded passes, in step order: decode steps of one token and a few prefill and verify passes,
/// each with one list per MoE layer.
fn synthetic_run(seed: u64, steps: u64) -> Vec<RouteRecord> {
    let mut rng = Rng(seed);
    let mut out = Vec::new();
    for step in 0..steps {
        let tokens = match rng.below(10) {
            0 => 16,
            1 => 4,
            _ => 1,
        };
        for layer in MOE_LAYERS {
            let route = synthetic_pass(&mut rng, tokens);
            out.push(RouteRecord {
                step,
                seq: step as u32,
                layer: LayerIx(layer),
                tokens,
                route,
            });
        }
    }
    out
}

/// What the trace keeps of a record: `cache_slot` belongs to the recording run.
fn as_traced(r: &RouteRecord) -> RouteRecord {
    let mut r = r.clone();
    for s in &mut r.route {
        s.cache_slot = MOE_MISS;
    }
    r
}

fn read_trace(bytes: &[u8]) -> Vec<RouteRecord> {
    TraceReader::new(bytes)
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap()
}

/// The ring is sized for the plan's largest pass: 16 tokens × top-4 = 64 = E.
fn max_union() -> u16 {
    doorbell_layout(MOE, 16, 1).unwrap().max_union()
}

#[test]
fn routing_round_trips_through_the_ring_the_recorder_and_the_reader() {
    for seed in [1, 2, 0xdead_beef] {
        let run = synthetic_run(seed, 200);
        let (mut tx, mut rx) = stats_ring(16, max_union()).unwrap();
        let mut writer = TraceWriter::new(Vec::new()).unwrap();
        let mut record = |rec: &StatRecord| match rec {
            StatRecord::Route(r) => writer.record(r).unwrap(),
            _ => unreachable!("only routing records are pushed"),
        };
        // The dispatcher pushes each step's layers; housekeeping drains every few steps.
        for (i, r) in run.iter().enumerate() {
            tx.push_route(r.step, r.seq, r.layer, r.tokens, &r.route)
                .unwrap();
            if i % 7 == 6 {
                rx.drain(&mut record);
            }
        }
        rx.drain(&mut record);
        assert_eq!(rx.dropped(), 0);

        let got = read_trace(&writer.finish().unwrap());
        let want: Vec<_> = run.iter().map(as_traced).collect();
        assert_eq!(got, want, "seed {seed}");
        assert!(got.windows(2).all(|w| w[0].step <= w[1].step));
    }
}

/// The same with the dispatcher and housekeeping on their own threads. The producer retries a
/// record the full ring refused, which a real dispatcher never does, so that the trace is
/// complete and comparable; the drops are counted all the same.
#[test]
fn routing_round_trips_across_threads() {
    let run = synthetic_run(42, 2_000);
    let (mut tx, mut rx) = stats_ring(8, max_union()).unwrap();
    let pushed = run.clone();
    let dispatcher = thread::spawn(move || {
        for r in &pushed {
            loop {
                match tx.push_route(r.step, r.seq, r.layer, r.tokens, &r.route) {
                    Ok(()) => break,
                    Err(StatDrop::Full) => thread::yield_now(),
                    Err(e) => panic!("{e}"),
                }
            }
        }
    });
    let mut writer = TraceWriter::new(Vec::new()).unwrap();
    let mut written = 0;
    while written < run.len() {
        written += rx.drain(|rec| match rec {
            StatRecord::Route(r) => writer.record(r).unwrap(),
            _ => unreachable!("only routing records are pushed"),
        });
        thread::yield_now();
    }
    dispatcher.join().unwrap();

    let got = read_trace(&writer.finish().unwrap());
    let want: Vec<_> = run.iter().map(as_traced).collect();
    assert_eq!(got, want);
}
