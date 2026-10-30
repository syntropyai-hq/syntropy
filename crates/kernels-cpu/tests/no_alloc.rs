// SPDX-License-Identifier: MIT OR Apache-2.0

//! COMP-E3-T02's second criterion: `CountingAlloc` records zero allocations over 10,000
//! steady-state steps (RFC-0040 §2.4; RFC-0060 §3.5).
//!
//! A steady-state step is what the dispatcher does each step after `Warming`. It replays the
//! captured op list and the stream runs it. The list holds:
//! - a copy between host buffers;
//! - a doorbell signal, then a wait on it with the consumer's reset;
//! - a stand-in kernel launch;
//! - an event record;
//! - a device-triggered wake.
//!
//! Then the dispatcher polls the event and checks the device's health. The device's thread and
//! a runner on it both count their own allocations.

#![allow(
    clippy::result_large_err,
    reason = "core::Error is RFC-0001 §5.12's struct, 344 B inline [measured: size_of, debug, x86-64 Linux, ce2044ae] (COMP-E13-T15)"
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use smallvec::smallvec;
use syntropy_core::{Bytes, DType, DeviceId, HostRegion, NodeId, QuantScheme};
use syntropy_kernels::{
    BufferSlice, DeviceHealth, Error, KernelProvider, LayoutVersion, OpArgs, OpKey, OpKind,
    OpenOptions, PowerHint, ShapeClass, ShapeSig, StreamPlan, StreamPriority, WaitCmp,
};
use syntropy_kernels_cpu::{CpuProvider, CpuWaker, KillSwitches, Run};
use syntropy_testkit::guards::{CountingAlloc, alloc_count_this_thread};

#[global_allocator]
static ALLOCATOR: CountingAlloc = CountingAlloc;

static KERNEL_RUNS: AtomicU64 = AtomicU64::new(0);

fn stand_in(_: &OpArgs<'_>) -> Result<(), Error> {
    KERNEL_RUNS.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

/// The allocator counts at all: without that, a zero below would prove nothing.
#[test]
fn the_allocator_counts() {
    let before = alloc_count_this_thread();
    let boxed = Box::new(1_u64);
    assert_eq!(alloc_count_this_thread() - before, 1);
    drop(boxed);
}

#[test]
fn ten_thousand_steady_state_steps_allocate_nothing() {
    const STEPS: u64 = 10_000;
    let provider = CpuProvider::new(KillSwitches::NONE).with_kernel(OpKind::Gemv, stand_in);
    let options = OpenOptions {
        streams: StreamPlan {
            compute: 1,
            copy_in: 0,
            copy_out: 0,
            collective: 0,
        },
        doorbell_slots: 4,
        capture: true,
        deterministic: true,
        power: PowerHint::Mains,
    };
    let dev = DeviceId {
        node: NodeId([5; 32]),
        ordinal: 0,
    };

    // Warming: open, buffers, the stream, the waker, the prepared op, the capture.
    let ctx = provider.open(dev, &options).unwrap();
    let s = provider.stream(&ctx, StreamPriority::Compute).unwrap();
    let mut src: Vec<u8> = (0..=255).collect();
    let mut dst = vec![0u8; 256];
    let a = provider
        .import(
            &ctx,
            HostRegion {
                addr: src.as_mut_ptr() as u64,
                len: Bytes(256),
            },
        )
        .unwrap();
    let b = provider
        .import(
            &ctx,
            HostRegion {
                addr: dst.as_mut_ptr() as u64,
                len: Bytes(256),
            },
        )
        .unwrap();
    let bells = provider.doorbells(&ctx).unwrap();
    let flag = bells.flag_at(0);
    let e = provider.event(&ctx).unwrap();
    let waker = Arc::new(CpuWaker::new());
    let host_waker = provider.register_waker(&ctx, Arc::clone(&waker)).unwrap();
    let key = OpKey {
        op: OpKind::Gemv,
        schemes: smallvec![QuantScheme::Plain(DType::F32)],
        layout: LayoutVersion::SOURCE,
        variant: 0,
    };
    let shape = ShapeSig {
        class: ShapeClass::Decode,
        rows: 1,
        dims: smallvec![256],
    };
    let op = provider.prepare(&ctx, &key, &shape).unwrap();
    let whole = |buffer| BufferSlice {
        buffer,
        offset: Bytes(0),
        bytes: Bytes(256),
    };
    provider.begin_capture(&ctx, &s).unwrap();
    provider
        .copy(&ctx, &s, whole(a.handle), whole(b.handle))
        .unwrap();
    provider.signal_host(&ctx, &s, flag, 1).unwrap();
    provider
        .wait_host(&ctx, &s, flag, 1, WaitCmp::Eq, Some(0))
        .unwrap();
    provider.launch(&ctx, &s, &op, &OpArgs::none()).unwrap();
    provider.record(&ctx, &s, &e).unwrap();
    provider.arm_wake(&ctx, &s, &host_waker).unwrap();
    let graph = provider.end_capture(&ctx, &s).unwrap();
    // One step outside the count, so first-use work (the thread handle) is done.
    provider.replay(&ctx, &s, &graph).unwrap();
    assert_eq!(provider.run(&ctx, &s).unwrap(), Run::Idle);

    let runs = KERNEL_RUNS.load(Ordering::Relaxed);
    let before = alloc_count_this_thread();
    for _ in 0..STEPS {
        provider.replay(&ctx, &s, &graph).unwrap();
        assert!(!provider.poll(&ctx, &e).unwrap());
        assert_eq!(provider.run(&ctx, &s).unwrap(), Run::Idle);
        assert!(provider.poll(&ctx, &e).unwrap());
        assert_eq!(provider.health(&ctx), DeviceHealth::Ok);
    }
    let allocations = alloc_count_this_thread() - before;
    assert_eq!(
        allocations, 0,
        "{allocations} allocations over {STEPS} steps"
    );
    assert_eq!(KERNEL_RUNS.load(Ordering::Relaxed) - runs, STEPS);
    assert_eq!(dst, src);
    assert_eq!(
        bells.flag(flag).unwrap().load(Ordering::Acquire),
        0,
        "reset every step"
    );
}

/// The same steps with the stream's runner on its own thread, as a CPU pool thread runs it:
/// the runner's thread allocates nothing either.
#[test]
fn a_runner_on_its_own_thread_allocates_nothing() {
    const STEPS: u64 = 10_000;
    let provider = CpuProvider::new(KillSwitches::NONE).with_kernel(OpKind::Gemv, stand_in);
    let options = OpenOptions {
        streams: StreamPlan {
            compute: 1,
            copy_in: 0,
            copy_out: 0,
            collective: 0,
        },
        doorbell_slots: 4,
        capture: true,
        deterministic: true,
        power: PowerHint::Mains,
    };
    let dev = DeviceId {
        node: NodeId([5; 32]),
        ordinal: 0,
    };
    let ctx = provider.open(dev, &options).unwrap();
    let s = provider.stream(&ctx, StreamPriority::Compute).unwrap();
    let bells = provider.doorbells(&ctx).unwrap();
    let (step_flag, done_flag) = (bells.flag_at(0), bells.flag_at(1));
    provider.begin_capture(&ctx, &s).unwrap();
    // Each step waits for the dispatcher's go, then raises its done flag.
    provider
        .wait_host(&ctx, &s, step_flag, 1, WaitCmp::Eq, Some(0))
        .unwrap();
    provider.signal_host(&ctx, &s, done_flag, 1).unwrap();
    let graph = provider.end_capture(&ctx, &s).unwrap();
    let mut runner = provider.take_runner(&ctx, &s).unwrap().unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let halt = Arc::clone(&stop);
    let pool = std::thread::spawn(move || {
        // Counted from the runner's first idle pass, once its thread's own setup is done.
        let mut counted = None;
        while !halt.load(Ordering::Acquire) {
            if runner.run().unwrap() == Run::Idle && counted.is_none() {
                counted = Some(alloc_count_this_thread());
            }
            std::hint::spin_loop();
        }
        alloc_count_this_thread() - counted.unwrap_or_else(alloc_count_this_thread)
    });
    let go = bells.flag(step_flag).unwrap();
    let done = bells.flag(done_flag).unwrap();
    let before = alloc_count_this_thread();
    for _ in 0..STEPS {
        provider.replay(&ctx, &s, &graph).unwrap();
        go.store(1, Ordering::Release);
        while done.swap(0, Ordering::AcqRel) != 1 {
            std::hint::spin_loop();
        }
    }
    let device_thread = alloc_count_this_thread() - before;
    stop.store(true, Ordering::Release);
    let runner_thread = pool.join().unwrap();
    assert_eq!(
        (device_thread, runner_thread),
        (0, 0),
        "(device thread, runner thread)"
    );
}
