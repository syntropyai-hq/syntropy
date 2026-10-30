// SPDX-License-Identifier: MIT OR Apache-2.0

//! COMP-E3-T02's third criterion, on PLC-E7-T17's dispatcher loop: with a 10 ms CPU segment per
//! MoE layer, no step waits for a timer, and a thread parks inside a step only on an armed wake
//! (RFC-0012 §3.3; RFC-0040 §4.2, the CPU row).
//!
//! The CPU provider's stream holds each step's CPU work. For each of three MoE layers:
//! 1. a stand-in kernel runs the layer's CPU segment, 10 ms;
//! 2. the stream raises the layer's doorbell `Q[l]` with a release store;
//! 3. `arm_wake` wakes the dispatcher if it announced it is parking.
//!
//! A pool thread runs the stream, and the dispatcher, on the test's thread, runs
//! `Dispatcher::run`. Its step services the three doorbells in order. Each pending doorbell is
//! predicted 100 ms out, far past the real 10 ms, so a park that a wake did not end would show
//! as a timed-out park of about 100 ms. Every park must instead end with the wake.

#![allow(
    clippy::result_large_err,
    reason = "core::Error is RFC-0001 §5.12's struct, 344 B inline [measured: size_of, debug, x86-64 Linux, ce2044ae] (COMP-E13-T15)"
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

use syntropy_core::{DeviceClockSync, DeviceId, MonoTime, Nanos, NodeId, TickRate, TickSource};
use syntropy_exec::{
    Dispatcher, LoopConfig, Serviced, StepDevice, StepProgress, StepScheduler, WakeSource,
    WakeWaiter,
};
use syntropy_kernels::{
    CapturedGraph, DeviceCtx, Error, HostFlag, HostWaker, KernelProvider, OpArgs, OpKey, OpKind,
    OpenOptions, PowerHint, ShapeClass, ShapeSig, Stream, StreamPlan, StreamPriority,
};
use syntropy_kernels_cpu::{CpuProvider, CpuWaker, Doorbells, KillSwitches, Parked, Run};
use syntropy_spsc::{Arm, Consumer, PopError, Producer};

const LAYERS: u32 = 3;
const STEPS: u32 = 3;

/// The CPU segment of one MoE layer: 10 ms of the calling thread.
fn moe_cpu_segment(_: &OpArgs<'_>) -> Result<(), Error> {
    thread::sleep(Duration::from_millis(10));
    Ok(())
}

/// A clock that moves 1 µs at every read, so the loop's spin budget passes without a wall clock.
#[derive(Clone, Default)]
struct Stepping(Arc<AtomicU64>);

impl TickSource for Stepping {
    fn ticks(&self) -> u64 {
        self.0.fetch_add(1_000, Ordering::SeqCst) + 1_000
    }

    fn ticks_to_nanos(&self, ticks: u64) -> Nanos {
        TickRate::NANOS.ticks_to_nanos(ticks)
    }

    fn now(&self) -> MonoTime {
        MonoTime(self.0.load(Ordering::SeqCst))
    }

    fn device_to_ticks(&self, device_ts: u64, sync: &DeviceClockSync) -> u64 {
        sync.to_ticks(device_ts, TickRate::NANOS.hz())
    }
}

/// Steps come in on a ring; a step is done once its frame leaves on the other.
struct Steps {
    inbound: Consumer<u32>,
    outbound: Producer<u32>,
    ready: Vec<u32>,
    closed: bool,
}

impl StepScheduler for Steps {
    type Step = u32;

    fn ingest(&mut self, _now: MonoTime) -> bool {
        let mut arrived = false;
        loop {
            match self.inbound.pop() {
                Ok(step) => {
                    self.ready.push(step);
                    arrived = true;
                }
                Err(PopError::Empty) => return arrived,
                Err(PopError::Closed) => {
                    self.closed = true;
                    return arrived;
                }
            }
        }
    }

    fn next_step(&mut self, _now: MonoTime) -> Option<u32> {
        (!self.ready.is_empty()).then(|| self.ready.remove(0))
    }

    fn finish(&mut self, _step: &mut u32, _now: MonoTime) {}

    fn try_send(&mut self, step: &mut u32, _now: MonoTime) -> Result<bool, Error> {
        Ok(self.outbound.push(*step).is_ok())
    }

    fn sent(&mut self, _step: u32, _now: MonoTime) {}

    fn abandoned(&mut self, step: u32, error: Error, _now: MonoTime) {
        panic!("step {step} abandoned: {error}");
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
        self.closed && self.ready.is_empty()
    }
}

/// The dispatcher's view of the CPU device: launching a step replays the CPU op list, and
/// servicing it reads the layers' doorbells in order, with the consumer's reset.
struct CpuStage<'a> {
    provider: &'a CpuProvider,
    ctx: &'a DeviceCtx,
    stream: &'a Stream,
    graph: &'a CapturedGraph,
    bells: Doorbells,
    q: Vec<HostFlag>,
    waker: HostWaker,
    next_layer: u32,
}

impl StepDevice<u32> for CpuStage<'_> {
    fn launch(&mut self, _step: &mut u32, _now: MonoTime) -> Result<(), Error> {
        self.next_layer = 0;
        self.provider.replay(self.ctx, self.stream, self.graph)
    }

    fn service(&mut self, _step: &mut u32, now: MonoTime) -> Result<Serviced, Error> {
        let mut fired = 0;
        while self.next_layer < LAYERS {
            let flag = self.bells.flag(self.q[self.next_layer as usize]).unwrap();
            if flag.load(Ordering::Acquire) != 1 {
                break;
            }
            flag.store(0, Ordering::Release);
            self.next_layer += 1;
            fired += 1;
        }
        let progress = if self.next_layer == LAYERS {
            StepProgress::Done
        } else {
            StepProgress::Pending {
                // Far past the real 10 ms: only the wake can end a park on time.
                expect: now.saturating_add(Nanos(100_000_000)),
                wake: WakeSource::Device(self.waker),
            }
        };
        Ok(Serviced { fired, progress })
    }
}

/// The loop's waiter over the CPU provider's wake, counting how each park ended.
struct CpuWait {
    waker: Arc<CpuWaker>,
    arms: u32,
    woken: u32,
    timed_out: u32,
}

impl WakeWaiter for CpuWait {
    fn arm(&mut self, _waker: &HostWaker) -> bool {
        self.waker.arm();
        self.arms += 1;
        true
    }

    fn park(&mut self, _waker: &HostWaker, timeout: Duration) {
        match self.waker.park(timeout) {
            Parked::Woken => self.woken += 1,
            Parked::TimedOut => self.timed_out += 1,
        }
    }

    fn disarm(&mut self, _waker: &HostWaker) {
        self.waker.disarm();
    }
}

#[test]
fn a_10ms_cpu_segment_per_layer_wakes_the_parked_dispatcher_every_time() {
    let provider =
        CpuProvider::new(KillSwitches::NONE).with_kernel(OpKind::MoeGrouped, moe_cpu_segment);
    let options = OpenOptions {
        streams: StreamPlan {
            compute: 1,
            copy_in: 0,
            copy_out: 0,
            collective: 0,
        },
        doorbell_slots: 8,
        capture: true,
        deterministic: true,
        power: PowerHint::Mains,
    };
    let dev = DeviceId {
        node: NodeId([9; 32]),
        ordinal: 0,
    };
    let ctx = provider.open(dev, &options).unwrap();
    let stream = provider.stream(&ctx, StreamPriority::Compute).unwrap();
    let bells = provider.doorbells(&ctx).unwrap();
    let q: Vec<HostFlag> = (0..LAYERS).map(|l| bells.flag_at(l)).collect();
    let cpu_waker = Arc::new(CpuWaker::new());
    let waker = provider
        .register_waker(&ctx, Arc::clone(&cpu_waker))
        .unwrap();
    let key = OpKey {
        op: OpKind::MoeGrouped,
        schemes: smallvec::smallvec![],
        layout: syntropy_kernels::LayoutVersion::SOURCE,
        variant: 0,
    };
    let shape = ShapeSig {
        class: ShapeClass::Decode,
        rows: 1,
        dims: smallvec::smallvec![],
    };
    let op = provider.prepare(&ctx, &key, &shape).unwrap();
    provider.begin_capture(&ctx, &stream).unwrap();
    for &flag in &q {
        provider
            .launch(&ctx, &stream, &op, &OpArgs::none())
            .unwrap();
        provider.signal_host(&ctx, &stream, flag, 1).unwrap();
        provider.arm_wake(&ctx, &stream, &waker).unwrap();
    }
    let graph = provider.end_capture(&ctx, &stream).unwrap();

    // The pool thread: runs the stream as its items come, spinning between them.
    let mut runner = provider.take_runner(&ctx, &stream).unwrap().unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let pool_stop = Arc::clone(&stop);
    let pool = thread::spawn(move || {
        while !pool_stop.load(Ordering::Acquire) {
            if runner.run().unwrap() == Run::Idle {
                std::hint::spin_loop();
            }
        }
    });

    let (mut to_stage, inbound) = syntropy_spsc::ring(8).unwrap();
    let (outbound, mut from_stage) = syntropy_spsc::ring(8).unwrap();
    for step in 0..STEPS {
        to_stage.push(step).unwrap();
    }
    drop(to_stage);
    let steps = Steps {
        inbound,
        outbound,
        ready: Vec::new(),
        closed: false,
    };
    let device = CpuStage {
        provider: &provider,
        ctx: &ctx,
        stream: &stream,
        graph: &graph,
        bells,
        q,
        waker,
        next_layer: 0,
    };
    let mut wait = CpuWait {
        waker: cpu_waker,
        arms: 0,
        woken: 0,
        timed_out: 0,
    };
    let mut dispatcher = Dispatcher::new(steps, device, Stepping::default());
    dispatcher.run(&LoopConfig::MAINS, &mut wait).unwrap();
    stop.store(true, Ordering::Release);
    pool.join().unwrap();

    let done: Vec<u32> = std::iter::from_fn(|| from_stage.pop().ok()).collect();
    eprintln!(
        "{STEPS} steps of {LAYERS} 10 ms layers: {} arms, {} parks woken, {} timed out",
        wait.arms, wait.woken, wait.timed_out
    );
    assert_eq!(done, [0, 1, 2], "every step ran");
    assert_eq!(wait.timed_out, 0, "no park waited for its timer");
    assert!(wait.woken >= 1, "the dispatcher parked inside a step");
    // The loop arms the device wake only for a pending doorbell (`Waiting`), never between
    // steps, and parks at most once per arm.
    assert!(wait.woken <= wait.arms);
    assert!(
        wait.arms <= STEPS * LAYERS,
        "armed only for pending doorbells: {} arms",
        wait.arms
    );
}
