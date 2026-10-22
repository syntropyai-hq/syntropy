// SPDX-License-Identifier: MIT OR Apache-2.0

//! PLC-E7-T17's third criterion. The simulator runs a one-stage instance on one OS thread
//! through `Dispatcher::poll_once` and `ComputePool::run_inline`, and two runs with the same seed
//! give identical step records and tokens (RFC-0060 §2.3; RFC-0012 §3.1, §3.2; ADR-0600
//! decisions 3 and 5).
//!
//! The stage is head and tail at once. Each step runs three MoE-like layers. For each layer:
//!
//! 1. The device runs a segment and writes the layer's route doorbell (`Rd[l]`).
//! 2. The dispatcher services the doorbell, and hands the layer's CPU work to the compute pool.
//! 3. The pool runs it when the virtual timeline reaches it. It transforms the activation and
//!    raises `Q[l]`.
//! 4. The device, which waits on `Q[l]` device-side, then runs the next segment.
//!
//! After the last layer the device sets `O[p]`, and the tail samples a token from the
//! activation. Segment, pool and client times are drawn from the run's seeded entropy. So a seed
//! fixes every interleaving, and only the computation decides the tokens.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;
use std::thread::{self, ThreadId};

use syntropy_core::{Entropy, MonoTime, Nanos, TickRate};
use syntropy_exec::{
    ComputePool, Dispatcher, PollOutcome, PoolProgram, Serviced, StepDevice, StepProgress,
    StepScheduler, WakeSource,
};
use syntropy_kernels::Error;
use syntropy_spsc::{Arm, Consumer, PopError, Producer, PushError};
use syntropy_testkit::sim::{
    Checker, Env, Model, RunOutcome, Schedule, SimEntropy, SimTicks, SimWorld, TraceEvent,
};

/// MoE-like layers per step.
const LAYERS: usize = 3;
/// The vocabulary the tail samples from.
const VOCAB: u64 = 32_000;

/// A draw from `[low, low + span)` nanoseconds.
fn draw(entropy: &SimEntropy, low: u64, span: u64) -> Nanos {
    Nanos(low + entropy.u64() % span)
}

/// One step of the layers' transform: a 64-bit mix, so every token depends on every layer.
fn mix(x: u64, layer: usize) -> u64 {
    let x = (x ^ (layer as u64 + 1)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^ (x >> 29)
}

/// Host-mapped memory that the device, the dispatcher and the pool share: the flags each writes,
/// and the step's activation.
#[derive(Default)]
struct Memory {
    /// `Rd[l]`: the device wrote layer `l`'s route.
    route: [bool; LAYERS],
    /// `O[p]`: the step is done.
    done: bool,
    /// The activation the layers transform.
    hidden: u64,
    /// The thread each piece of work ran on.
    threads: Vec<ThreadId>,
}

type Shared = Rc<RefCell<Memory>>;

/// A sequence the client opens: its first input token, and how many tokens it wants.
#[derive(Clone, Copy, Debug)]
struct Open {
    seq: u32,
    prompt: u32,
    tokens: u32,
}

/// A token frame, tail to client.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TokenFrame {
    seq: u32,
    pos: u32,
    token: u32,
}

/// One step: one position of one sequence.
#[derive(Debug)]
struct Step {
    number: u64,
    seq: u32,
    pos: u32,
    input: u32,
    launched: MonoTime,
    done: MonoTime,
    token: u32,
}

/// The step record this test compares across runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StepRecord {
    number: u64,
    seq: u32,
    pos: u32,
    token: u32,
    launched: MonoTime,
    done: MonoTime,
    sent: MonoTime,
}

/// A live sequence at the scheduler.
struct Sequence {
    seq: u32,
    next_input: u32,
    pos: u32,
    left: u32,
}

/// The scheduler side: composes one step at a time, round-robin over open sequences, and samples
/// at the tail.
struct SimScheduler {
    inbound: Consumer<Open>,
    outbound: Producer<TokenFrame>,
    sequences: VecDeque<Sequence>,
    next_number: u64,
    records: Vec<StepRecord>,
    mem: Shared,
}

impl StepScheduler for SimScheduler {
    type Step = Step;

    fn ingest(&mut self, _now: MonoTime) -> bool {
        let mut arrived = false;
        while let Ok(open) = self.inbound.pop() {
            self.sequences.push_back(Sequence {
                seq: open.seq,
                next_input: open.prompt,
                pos: 0,
                left: open.tokens,
            });
            arrived = true;
        }
        arrived
    }

    fn next_step(&mut self, now: MonoTime) -> Option<Step> {
        let sequence = self.sequences.pop_front()?;
        let step = Step {
            number: self.next_number,
            seq: sequence.seq,
            pos: sequence.pos,
            input: sequence.next_input,
            launched: now,
            done: now,
            token: 0,
        };
        self.next_number += 1;
        self.sequences.push_front(sequence);
        Some(step)
    }

    fn finish(&mut self, step: &mut Step, now: MonoTime) {
        step.token = u32::try_from(self.mem.borrow().hidden % VOCAB).unwrap();
        step.done = now;
        let mut sequence = self.sequences.pop_front().expect("the step's sequence");
        assert_eq!(sequence.seq, step.seq);
        sequence.next_input = step.token;
        sequence.pos += 1;
        sequence.left -= 1;
        if sequence.left > 0 {
            self.sequences.push_back(sequence);
        }
    }

    fn try_send(&mut self, step: &mut Step, _now: MonoTime) -> Result<bool, Error> {
        let frame = TokenFrame {
            seq: step.seq,
            pos: step.pos,
            token: step.token,
        };
        match self.outbound.push(frame) {
            Ok(()) => Ok(true),
            Err(PushError::Full(_)) => Ok(false),
            Err(PushError::Closed(_)) => unreachable!("the client holds its end"),
        }
    }

    fn sent(&mut self, step: Step, now: MonoTime) {
        self.records.push(StepRecord {
            number: step.number,
            seq: step.seq,
            pos: step.pos,
            token: step.token,
            launched: step.launched,
            done: step.done,
            sent: now,
        });
    }

    fn next_wake(&self, _now: MonoTime) -> Option<MonoTime> {
        None
    }

    fn arm_inbound(&mut self) -> Arm {
        self.inbound.arm_thread()
    }

    fn disarm_inbound(&mut self) {
        self.inbound.disarm();
    }

    fn closed(&self) -> bool {
        false
    }
}

/// One doorbell's CPU work: layer `layer`'s transform of the activation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PoolWork {
    layer: usize,
}

/// What the device's virtual timeline does next.
#[derive(Clone, Copy, Debug)]
enum DeviceEvent {
    Route(usize),
    Done,
}

/// The device side. Its timeline is a list of requests that the world schedules after each
/// pass.
struct SimDevice {
    mem: Shared,
    entropy: Arc<SimEntropy>,
    /// The next layer whose route doorbell the dispatcher services.
    layer: usize,
    expect: MonoTime,
    timeline: Vec<(Nanos, DeviceEvent)>,
    pool_work: Vec<(Nanos, PoolWork)>,
}

impl SimDevice {
    /// The pool has raised `Q[layer]`. The device, waiting on it device-side, runs the next
    /// segment.
    fn pool_flag(&mut self, layer: usize, now: MonoTime) {
        let segment = draw(&self.entropy, 200_000, 1_800_000);
        let next = if layer + 1 < LAYERS {
            DeviceEvent::Route(layer + 1)
        } else {
            DeviceEvent::Done
        };
        self.expect = now.saturating_add(segment);
        self.timeline.push((segment, next));
    }
}

impl StepDevice<Step> for SimDevice {
    fn launch(&mut self, step: &mut Step, now: MonoTime) -> Result<(), Error> {
        let mut mem = self.mem.borrow_mut();
        mem.route = [false; LAYERS];
        mem.done = false;
        mem.hidden = u64::from(step.input) << 20 | u64::from(step.pos);
        mem.threads.push(thread::current().id());
        drop(mem);
        self.layer = 0;
        let segment = draw(&self.entropy, 200_000, 1_800_000);
        self.expect = now.saturating_add(segment);
        self.timeline.push((segment, DeviceEvent::Route(0)));
        Ok(())
    }

    fn service(&mut self, _step: &mut Step, now: MonoTime) -> Result<Serviced, Error> {
        let mut mem = self.mem.borrow_mut();
        let mut fired = 0;
        // Doorbells in layer order: a later one cannot fire before an earlier one.
        while self.layer < LAYERS && mem.route[self.layer] {
            // The consumer reset, then the host work: the layer's CPU work to the pool.
            mem.route[self.layer] = false;
            let cost = draw(&self.entropy, 100_000, 900_000);
            self.pool_work.push((cost, PoolWork { layer: self.layer }));
            self.expect = now.saturating_add(cost);
            self.layer += 1;
            fired += 1;
        }
        let progress = if mem.done {
            StepProgress::Done
        } else {
            StepProgress::Pending {
                expect: self.expect,
                wake: WakeSource::Spin,
            }
        };
        Ok(Serviced { fired, progress })
    }
}

/// The pool's program: a layer's transform of the shared activation.
struct SimProgram {
    mem: Shared,
}

impl PoolProgram for SimProgram {
    type Work = PoolWork;

    fn run(&mut self, work: &PoolWork) -> Result<(), Error> {
        let mut mem = self.mem.borrow_mut();
        mem.hidden = mix(mem.hidden, work.layer);
        mem.threads.push(thread::current().id());
        Ok(())
    }
}

/// The world's events.
#[derive(Clone, Copy, Debug)]
enum Event {
    /// The client opens a sequence.
    Open(Open),
    /// The device's timeline reaches a doorbell write or the end of the step.
    Device(DeviceEvent),
    /// The pool's timeline reaches a work item.
    Pool(PoolWork),
    /// The client reads its token frames.
    Drain,
}

impl TraceEvent for Event {
    fn trace_bytes(&self, out: &mut Vec<u8>) {
        match *self {
            Event::Open(open) => {
                out.push(0);
                for field in [open.seq, open.prompt, open.tokens] {
                    out.extend_from_slice(&field.to_le_bytes());
                }
            }
            Event::Device(DeviceEvent::Route(layer)) => out.extend_from_slice(&[1, layer as u8]),
            Event::Device(DeviceEvent::Done) => out.push(2),
            Event::Pool(work) => out.extend_from_slice(&[3, work.layer as u8]),
            Event::Drain => out.push(4),
        }
    }
}

/// The one-stage instance and its client.
struct OneStage {
    dispatcher: Dispatcher<SimScheduler, SimDevice, SimTicks>,
    pool: ComputePool<SimProgram>,
    entropy: Arc<SimEntropy>,
    mem: Shared,
    client: Producer<Open>,
    from_tail: Consumer<TokenFrame>,
    tokens: Vec<TokenFrame>,
    drain_pending: bool,
    /// Passes that returned `Idle` with a step in flight: criterion 2, which must stay 0.
    idle_in_flight: u32,
}

impl OneStage {
    fn new(env: Env) -> OneStage {
        let mem = Shared::default();
        let (client, inbound) = syntropy_spsc::ring(8).unwrap();
        // One slot, so a slow client holds finished steps on a full ring.
        let (outbound, from_tail) = syntropy_spsc::ring(1).unwrap();
        let scheduler = SimScheduler {
            inbound,
            outbound,
            sequences: VecDeque::new(),
            next_number: 0,
            records: Vec::new(),
            mem: Rc::clone(&mem),
        };
        let device = SimDevice {
            mem: Rc::clone(&mem),
            entropy: Arc::clone(&env.entropy),
            layer: 0,
            expect: MonoTime(0),
            timeline: Vec::new(),
            pool_work: Vec::new(),
        };
        let ticks = SimTicks::new(env.clock.clone(), TickRate::NANOS);
        OneStage {
            dispatcher: Dispatcher::new(scheduler, device, ticks),
            pool: ComputePool::new(SimProgram {
                mem: Rc::clone(&mem),
            }),
            entropy: env.entropy,
            mem,
            client,
            from_tail,
            tokens: Vec::new(),
            drain_pending: false,
            idle_in_flight: 0,
        }
    }

    /// Polls the dispatcher until it stops making progress, then schedules what the pass set
    /// going on the device's and the pool's timelines.
    fn drive(&mut self, now: MonoTime, schedule: &mut Schedule<'_, Event>) {
        loop {
            match self.dispatcher.poll_once(now) {
                PollOutcome::Progress => {}
                PollOutcome::Waiting { .. } => break,
                PollOutcome::Idle { next_wake } => {
                    if self.dispatcher.in_flight() {
                        self.idle_in_flight += 1;
                    }
                    assert_eq!(next_wake, None, "this stage sets no timer");
                    break;
                }
                PollOutcome::Fault(error) => panic!("fault: {error}"),
            }
        }
        let device = self.dispatcher.device_mut();
        for (after, event) in device.timeline.drain(..) {
            schedule.after(after, Event::Device(event));
        }
        for (after, work) in device.pool_work.drain(..) {
            schedule.after(after, Event::Pool(work));
        }
        if !self.drain_pending && !self.from_tail.is_empty() {
            self.drain_pending = true;
            // A client that reads up to 20 ms late, longer than a step [estimate: 3 layers of
            // 0.2-2 ms segments and 0.1-1 ms pool items], so finished steps wait on the ring.
            schedule.after(draw(&self.entropy, 0, 20_000_000), Event::Drain);
        }
    }
}

impl Model for OneStage {
    type Event = Event;

    fn handle(&mut self, now: MonoTime, event: Event, schedule: &mut Schedule<'_, Event>) {
        match event {
            Event::Open(open) => self.client.push(open).unwrap(),
            Event::Device(DeviceEvent::Route(layer)) => self.mem.borrow_mut().route[layer] = true,
            Event::Device(DeviceEvent::Done) => self.mem.borrow_mut().done = true,
            Event::Pool(work) => {
                self.pool.run_inline(&work).unwrap();
                self.dispatcher.device_mut().pool_flag(work.layer, now);
            }
            Event::Drain => {
                self.drain_pending = false;
                loop {
                    match self.from_tail.pop() {
                        Ok(frame) => self.tokens.push(frame),
                        Err(PopError::Empty) => break,
                        Err(PopError::Closed) => unreachable!("the stage holds its end"),
                    }
                }
            }
        }
        self.drive(now, schedule);
    }
}

/// Criterion 2, checked after every event: no pass returned `Idle` with a step in flight.
struct IdleOnlyBetweenSteps;

impl Checker<OneStage> for IdleOnlyBetweenSteps {
    fn name(&self) -> &str {
        "idle-only-between-steps"
    }

    fn after_event(&mut self, model: &OneStage, _now: MonoTime) -> Result<(), String> {
        match model.idle_in_flight {
            0 => Ok(()),
            n => Err(format!("{n} passes returned Idle with a step in flight")),
        }
    }
}

/// What a run produced.
#[derive(Debug, PartialEq, Eq)]
struct Run {
    records: Vec<StepRecord>,
    tokens: Vec<TokenFrame>,
    trace: syntropy_core::Blake3,
    polls: u64,
    overruns: u64,
    longest: Nanos,
    pool_items: u64,
}

const SEQUENCES: u32 = 3;
const TOKENS_EACH: u32 = 4;

/// Runs the one-stage instance for `seed`: three sequences of four tokens, opened at random
/// times in the first 5 ms, with one virtual second to finish.
fn run(seed: u64) -> Run {
    let mut world = SimWorld::new(seed, OneStage::new);
    world.add_checker(IdleOnlyBetweenSteps);
    for seq in 0..SEQUENCES {
        let at = MonoTime(world.model().entropy.u64() % 5_000_000);
        let open = Open {
            seq,
            prompt: 100 + seq,
            tokens: TOKENS_EACH,
        };
        world.schedule(at, Event::Open(open));
    }
    let end = MonoTime(1_000_000_000);
    match world.run_until(end) {
        RunOutcome::Reached { .. } => {}
        RunOutcome::Violated(violation) => panic!("{violation}"),
    }
    // Every piece of work ran on this, the test's, thread.
    let me = thread::current().id();
    let mem = world.model().mem.borrow();
    assert!(
        mem.threads.iter().all(|id| *id == me),
        "work left the thread"
    );
    assert!(!mem.threads.is_empty());
    drop(mem);
    let model = world.model();
    let stats = model.dispatcher.stats();
    Run {
        records: model.dispatcher.scheduler().records.clone(),
        tokens: model.tokens.clone(),
        trace: world.trace_hash(),
        polls: stats.polls,
        overruns: stats.overruns,
        longest: stats.longest,
        pool_items: model.pool.ran(),
    }
}

/// The tokens of one sequence, in order.
fn stream(run: &Run, seq: u32) -> Vec<u32> {
    run.tokens
        .iter()
        .filter(|frame| frame.seq == seq)
        .map(|frame| frame.token)
        .collect()
}

#[test]
fn one_stage_runs_on_one_thread_and_a_seed_replays_it_exactly() {
    let first = run(7);
    let total = (SEQUENCES * TOKENS_EACH) as usize;
    assert_eq!(first.records.len(), total, "every step ran");
    assert_eq!(first.tokens.len(), total, "every token reached the client");
    assert_eq!(
        first.pool_items,
        (total * LAYERS) as u64,
        "run_inline per layer"
    );
    for seq in 0..SEQUENCES {
        let positions: Vec<u32> = first
            .tokens
            .iter()
            .filter(|frame| frame.seq == seq)
            .map(|frame| frame.pos)
            .collect();
        assert_eq!(positions, [0, 1, 2, 3], "sequence {seq} in order");
    }
    // A slow client held at least one finished step on the full outbound ring.
    assert!(
        first.records.iter().any(|record| record.sent > record.done),
        "no credit stall exercised"
    );
    assert!(first.polls > first.records.len() as u64);
    // Every pass takes no virtual time here, because `SimTicks` moves only between events, so
    // the watchdog cannot fire in the simulator: a pass that spun or parked would hang this one
    // thread instead. `dispatch::tests::the_watchdog_counts_a_pass_over_its_budget` covers it.
    assert_eq!((first.overruns, first.longest), (0, Nanos(0)));

    // The same seed: the same step records, tokens and trace.
    assert_eq!(run(7), first);

    // Another seed: other timings, but the same tokens per sequence, which only the
    // computation decides.
    let other = run(8);
    assert_ne!(other.records, first.records);
    assert_ne!(other.trace, first.trace);
    for seq in 0..SEQUENCES {
        assert_eq!(stream(&other, seq), stream(&first, seq));
    }
}
