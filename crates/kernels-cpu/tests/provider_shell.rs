// SPDX-License-Identifier: MIT OR Apache-2.0

//! COMP-E3-T02's first criterion: the CPU provider's shell holds to RFC-0040 §2's contract over
//! real host buffers. No CPU kernel exists yet, so the launches run stand-in kernels that
//! [`CpuProvider::with_kernel`] adds.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::thread;
use std::time::Duration;

use smallvec::smallvec;
use syntropy_core::{Bytes, DType, DeviceId, ErrorClass, HostRegion, NodeId, QuantScheme};
use syntropy_kernels::{
    AllocRequest, BufferSlice, DeviceClass, DeviceCtx, DeviceFeatures, DeviceHealth, Error,
    ErrorCode, KernelProvider, LayoutVersion, MemKind, MemoryDomain, OpArgs, OpKey, OpKind,
    OpenOptions, PowerHint, ShapeClass, ShapeSig, Stream, StreamPlan, StreamPriority, WaitCmp,
    unsupported,
};
use syntropy_kernels_cpu::{CpuProvider, CpuWaker, KillSwitches, Parked, Run, STREAM_CAPACITY};

fn options(doorbells: u32) -> OpenOptions {
    OpenOptions {
        streams: StreamPlan {
            compute: 1,
            copy_in: 1,
            copy_out: 0,
            collective: 0,
        },
        doorbell_slots: doorbells,
        capture: true,
        deterministic: true,
        power: PowerHint::Mains,
    }
}

fn device(ordinal: u16) -> DeviceId {
    DeviceId {
        node: NodeId([3; 32]),
        ordinal,
    }
}

fn open(provider: &CpuProvider) -> DeviceCtx {
    provider.open(device(0), &options(8)).unwrap()
}

fn key(op: OpKind) -> OpKey {
    OpKey {
        op,
        schemes: smallvec![QuantScheme::Plain(DType::F32)],
        layout: LayoutVersion::SOURCE,
        variant: 0,
    }
}

fn shape() -> ShapeSig {
    ShapeSig {
        class: ShapeClass::Decode,
        rows: 1,
        dims: smallvec![64],
    }
}

/// A host buffer the test owns, imported into the device.
fn import(provider: &CpuProvider, ctx: &DeviceCtx, bytes: &mut [u8]) -> u64 {
    let region = HostRegion {
        addr: bytes.as_mut_ptr() as u64,
        len: Bytes(bytes.len() as u64),
    };
    provider.import(ctx, region).unwrap().handle
}

fn slice(buffer: u64, offset: u64, bytes: u64) -> BufferSlice {
    BufferSlice {
        buffer,
        offset: Bytes(offset),
        bytes: Bytes(bytes),
    }
}

#[test]
fn one_device_per_numa_domain_with_the_detected_class() {
    let provider = CpuProvider::new(KillSwitches::NONE);
    assert_eq!(provider.id().0, "cpu");
    assert_eq!(provider.domain(), MemoryDomain::Host);
    let devices = provider.enumerate().unwrap();
    assert!(!devices.is_empty());
    let detected = syntropy_kernels_cpu::detect(&KillSwitches::NONE).unwrap();
    for (i, dev) in devices.iter().enumerate() {
        assert_eq!(dev.local as usize, i);
        assert_eq!(dev.class, DeviceClass::Cpu { isa: detected.isa });
        assert!(dev.numa.is_some());
        assert!(dev.features.contains(
            DeviceFeatures::GRAPH_CAPTURE
                | DeviceFeatures::STREAM_MEM_OPS
                | DeviceFeatures::HOST_REGISTER
                | DeviceFeatures::HOST_IMPORT
        ));
    }
    let mut nodes: Vec<u16> = devices.iter().filter_map(|d| d.numa).collect();
    nodes.dedup();
    assert_eq!(nodes.len(), devices.len(), "one device per node");
    let missing = provider
        .open(device(devices.len() as u16), &options(1))
        .unwrap_err();
    assert_eq!(missing.class, ErrorClass::Unsupported);
}

#[test]
fn host_memory_is_allocated_registered_and_imported_as_it_is() {
    let provider = CpuProvider::new(KillSwitches::NONE);
    let ctx = open(&provider);
    let buf = provider
        .alloc(
            &ctx,
            AllocRequest {
                bytes: Bytes(4096),
                kind: MemKind::HostMapped,
                align: Bytes(0),
                reserve_only: false,
            },
        )
        .unwrap();
    assert_eq!(buf.addr % 64, 0, "at least cache-line aligned");
    assert_eq!((buf.bytes, buf.kind), (Bytes(4096), MemKind::HostMapped));
    for kind in [MemKind::Device, MemKind::Unified] {
        let request = AllocRequest {
            bytes: Bytes(64),
            kind,
            align: Bytes(64),
            reserve_only: false,
        };
        assert_eq!(provider.alloc(&ctx, request).unwrap_err(), unsupported());
    }
    let reserve = AllocRequest {
        bytes: Bytes(64),
        kind: MemKind::HostMapped,
        align: Bytes(64),
        reserve_only: true,
    };
    assert_eq!(provider.alloc(&ctx, reserve).unwrap_err(), unsupported());
    let empty = AllocRequest {
        bytes: Bytes(0),
        kind: MemKind::HostMapped,
        align: Bytes(64),
        reserve_only: false,
    };
    let error = provider.alloc(&ctx, empty).unwrap_err();
    assert_eq!(
        (error.class, error.code),
        (ErrorClass::Capacity, ErrorCode::ALLOC_FAILED)
    );

    let mut owned = vec![7u8; 128];
    let region = HostRegion {
        addr: owned.as_mut_ptr() as u64,
        len: Bytes(128),
    };
    let imported = provider.import(&ctx, region).unwrap();
    assert_eq!((imported.addr, imported.bytes), (region.addr, region.len));
    let registration = provider.register_host(&ctx, region).unwrap();
    assert_eq!(registration.region, region);
    provider.free(&ctx, imported);
    provider.free(&ctx, buf);
    assert_eq!(owned, vec![7u8; 128], "freeing an import leaves the memory");
}

#[test]
fn a_copy_runs_in_stream_order_over_host_buffers() {
    let provider = CpuProvider::new(KillSwitches::NONE);
    let ctx = open(&provider);
    let s = provider.stream(&ctx, StreamPriority::CopyIn).unwrap();
    let mut src: Vec<u8> = (0..64).collect();
    let mut dst = vec![0u8; 64];
    let (a, b) = (
        import(&provider, &ctx, &mut src),
        import(&provider, &ctx, &mut dst),
    );
    provider
        .copy(&ctx, &s, slice(a, 16, 32), slice(b, 0, 32))
        .unwrap();
    assert_eq!(provider.run(&ctx, &s).unwrap(), Run::Idle);
    assert_eq!(&dst[..32], &src[16..48]);
    assert!(dst[32..].iter().all(|&x| x == 0));
    // Out of bounds and mismatched lengths are refused at push.
    assert!(
        provider
            .copy(&ctx, &s, slice(a, 60, 8), slice(b, 0, 8))
            .is_err()
    );
    assert!(
        provider
            .copy(&ctx, &s, slice(a, 0, 8), slice(b, 0, 4))
            .is_err()
    );
}

/// A buffer freed while a copy into it is still queued is not deallocated until the streams
/// have drained, so the copy cannot land in memory a later `alloc` hands out (review of #1829,
/// B1).
#[test]
fn a_buffer_freed_under_a_queued_copy_outlives_the_copy() {
    let provider = CpuProvider::new(KillSwitches::NONE);
    let ctx = open(&provider);
    let s = provider.stream(&ctx, StreamPriority::Compute).unwrap();
    let mut src = vec![0xABu8; 4096];
    let a = import(&provider, &ctx, &mut src);
    let request = AllocRequest {
        bytes: Bytes(4096),
        kind: MemKind::HostMapped,
        align: Bytes(64),
        reserve_only: false,
    };
    let dst = provider.alloc(&ctx, request).unwrap();
    provider
        .copy(&ctx, &s, slice(a, 0, 4096), slice(dst.handle, 0, 4096))
        .unwrap();
    provider.free(&ctx, dst);
    assert_eq!(
        provider.deferred_frees(&ctx).unwrap(),
        1,
        "the copy is still queued"
    );
    let next = provider.alloc(&ctx, request).unwrap();
    assert_eq!(provider.run(&ctx, &s).unwrap(), Run::Idle);
    let mut seen = vec![0u8; 4096];
    let probe = import(&provider, &ctx, &mut seen);
    let t = provider.stream(&ctx, StreamPriority::CopyIn).unwrap();
    provider
        .copy(&ctx, &t, slice(next.handle, 0, 4096), slice(probe, 0, 4096))
        .unwrap();
    provider.synchronize(&ctx, &t).unwrap();
    assert!(
        seen.iter().all(|&b| b == 0),
        "the queued copy landed in the next buffer"
    );
    assert_eq!(
        provider.deferred_frees(&ctx).unwrap(),
        0,
        "released once drained"
    );
}

/// A captured copy names its buffers, not their addresses, so a buffer freed after the capture
/// fails the replay rather than being written (review of #1829).
#[test]
fn a_replay_after_its_buffer_is_freed_fails_instead_of_writing() {
    let provider = CpuProvider::new(KillSwitches::NONE);
    let ctx = open(&provider);
    let s = provider.stream(&ctx, StreamPriority::Compute).unwrap();
    let mut src = vec![0xCDu8; 256];
    let a = import(&provider, &ctx, &mut src);
    let request = AllocRequest {
        bytes: Bytes(256),
        kind: MemKind::HostMapped,
        align: Bytes(64),
        reserve_only: false,
    };
    let dst = provider.alloc(&ctx, request).unwrap();
    provider.begin_capture(&ctx, &s).unwrap();
    provider
        .copy(&ctx, &s, slice(a, 0, 256), slice(dst.handle, 0, 256))
        .unwrap();
    let graph = provider.end_capture(&ctx, &s).unwrap();
    provider.free(&ctx, dst);
    assert_eq!(
        provider.deferred_frees(&ctx).unwrap(),
        0,
        "nothing was queued"
    );
    let error = provider.replay(&ctx, &s, &graph).unwrap_err();
    assert_eq!(error.class, ErrorClass::Internal);
    assert!(error.message.contains("no buffer"), "{error}");
    assert_eq!(
        provider.run(&ctx, &s).unwrap(),
        Run::Idle,
        "nothing was queued"
    );
}

#[test]
fn events_are_atomics_that_streams_record_and_wait_on() {
    let provider = CpuProvider::new(KillSwitches::NONE);
    let ctx = open(&provider);
    let (s1, s2) = (
        provider.stream(&ctx, StreamPriority::Compute).unwrap(),
        provider.stream(&ctx, StreamPriority::CopyIn).unwrap(),
    );
    let e = provider.event(&ctx).unwrap();
    provider.record(&ctx, &s1, &e).unwrap();
    provider.wait(&ctx, &s2, &e).unwrap();
    assert!(!provider.poll(&ctx, &e).unwrap(), "recorded, not reached");
    assert_eq!(provider.run(&ctx, &s2).unwrap(), Run::WaitingForEvent);
    assert_eq!(provider.run(&ctx, &s1).unwrap(), Run::Idle);
    assert!(provider.poll(&ctx, &e).unwrap());
    assert_eq!(provider.run(&ctx, &s2).unwrap(), Run::Idle);
}

#[test]
fn a_doorbell_wait_passes_on_one_then_resets_it() {
    let provider = CpuProvider::new(KillSwitches::NONE);
    let ctx = open(&provider);
    let s = provider.stream(&ctx, StreamPriority::Compute).unwrap();
    let bells = provider.doorbells(&ctx).unwrap();
    let (incoming, outgoing) = (bells.flag_at(3), bells.flag_at(4));
    provider
        .wait_host(&ctx, &s, incoming, 1, WaitCmp::Eq, Some(0))
        .unwrap();
    provider.signal_host(&ctx, &s, outgoing, 1).unwrap();
    assert_eq!(
        provider.run(&ctx, &s).unwrap(),
        Run::WaitingForFlag(incoming)
    );
    assert_eq!(bells.flag(outgoing).unwrap().load(Ordering::Acquire), 0);
    bells.flag(incoming).unwrap().store(1, Ordering::Release);
    assert_eq!(provider.run(&ctx, &s).unwrap(), Run::Idle);
    assert_eq!(
        bells.flag(incoming).unwrap().load(Ordering::Acquire),
        0,
        "consumer reset"
    );
    assert_eq!(bells.flag(outgoing).unwrap().load(Ordering::Acquire), 1);
    // A flag the device does not have is refused.
    let past = syntropy_kernels::HostFlag { page: 9, slot: 0 };
    assert!(provider.signal_host(&ctx, &s, past, 1).is_err());
}

#[test]
fn capture_records_an_op_list_that_replays_afresh() {
    let provider = CpuProvider::new(KillSwitches::NONE);
    let ctx = open(&provider);
    let s = provider.stream(&ctx, StreamPriority::Compute).unwrap();
    let bells = provider.doorbells(&ctx).unwrap();
    let flag = bells.flag_at(0);
    let e = provider.event(&ctx).unwrap();
    provider.begin_capture(&ctx, &s).unwrap();
    assert!(
        provider.begin_capture(&ctx, &s).is_err(),
        "one capture at a time"
    );
    provider.signal_host(&ctx, &s, flag, 1).unwrap();
    provider.record(&ctx, &s, &e).unwrap();
    let graph = provider.end_capture(&ctx, &s).unwrap();
    // Capturing ran nothing and queued nothing.
    assert_eq!(provider.run(&ctx, &s).unwrap(), Run::Idle);
    assert_eq!(bells.flag(flag).unwrap().load(Ordering::Acquire), 0);

    for _ in 0..3 {
        provider.replay(&ctx, &s, &graph).unwrap();
        assert!(
            !provider.poll(&ctx, &e).unwrap(),
            "each replay records afresh"
        );
        assert_eq!(provider.run(&ctx, &s).unwrap(), Run::Idle);
        assert!(provider.poll(&ctx, &e).unwrap());
        assert_eq!(bells.flag(flag).unwrap().swap(0, Ordering::AcqRel), 1);
    }
}

static STAND_IN_RUNS: AtomicU32 = AtomicU32::new(0);

fn stand_in(_: &OpArgs<'_>) -> Result<(), Error> {
    STAND_IN_RUNS.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

fn failing(_: &OpArgs<'_>) -> Result<(), Error> {
    Err(Error {
        class: ErrorClass::Device,
        code: ErrorCode::DEVICE_STICKY,
        message: "the stand-in failed".to_owned(),
    })
}

#[test]
fn a_prepared_kernel_runs_in_order_and_an_unknown_op_is_unsupported() {
    let provider = CpuProvider::new(KillSwitches::NONE).with_kernel(OpKind::Gemv, stand_in);
    let ctx = open(&provider);
    let s = provider.stream(&ctx, StreamPriority::Compute).unwrap();
    assert_eq!(
        provider
            .prepare(&ctx, &key(OpKind::Gemm), &shape())
            .unwrap_err(),
        unsupported(),
        "no kernel task has filled the table yet"
    );
    let op = provider
        .prepare(&ctx, &key(OpKind::Gemv), &shape())
        .unwrap();
    let before = STAND_IN_RUNS.load(Ordering::Relaxed);
    provider.launch(&ctx, &s, &op, &OpArgs::none()).unwrap();
    provider.launch(&ctx, &s, &op, &OpArgs::none()).unwrap();
    assert_eq!(
        STAND_IN_RUNS.load(Ordering::Relaxed),
        before,
        "queued, not run"
    );
    assert_eq!(provider.run(&ctx, &s).unwrap(), Run::Idle);
    assert_eq!(STAND_IN_RUNS.load(Ordering::Relaxed), before + 2);
}

#[test]
fn a_failed_kernel_makes_the_device_sticky() {
    let provider = CpuProvider::new(KillSwitches::NONE).with_kernel(OpKind::Gemm, failing);
    let ctx = open(&provider);
    let s = provider.stream(&ctx, StreamPriority::Compute).unwrap();
    let op = provider
        .prepare(&ctx, &key(OpKind::Gemm), &shape())
        .unwrap();
    assert_eq!(provider.health(&ctx), DeviceHealth::Ok);
    provider.launch(&ctx, &s, &op, &OpArgs::none()).unwrap();
    assert!(provider.run(&ctx, &s).is_err());
    assert_eq!(
        provider.health(&ctx),
        DeviceHealth::Sticky {
            code: ErrorCode::DEVICE_STICKY
        }
    );
    let later = provider.signal_host(&ctx, &s, syntropy_kernels::HostFlag { page: 0, slot: 0 }, 1);
    assert_eq!(later.unwrap_err().class, ErrorClass::Device);
}

/// RFC-0040 §4.2, the CPU row: `arm_wake` wakes the waiter after the doorbell's release store,
/// and only a waiter that announced it is parking.
#[test]
fn arm_wake_wakes_a_parked_waiter_after_the_doorbell() {
    let provider = CpuProvider::new(KillSwitches::NONE);
    let ctx = open(&provider);
    let s = provider.stream(&ctx, StreamPriority::Compute).unwrap();
    let bells = provider.doorbells(&ctx).unwrap();
    let flag = bells.flag_at(1);
    let waker = Arc::new(CpuWaker::new());
    let host_waker = provider.register_waker(&ctx, Arc::clone(&waker)).unwrap();
    let (w, b) = (Arc::clone(&waker), bells.clone());
    let waiter = thread::spawn(move || {
        w.arm();
        let outcome = if b.flag(flag).unwrap().load(Ordering::Acquire) == 1 {
            Parked::Woken
        } else {
            w.park(Duration::from_secs(10))
        };
        w.disarm();
        (outcome, b.flag(flag).unwrap().load(Ordering::Acquire))
    });
    thread::sleep(Duration::from_millis(20));
    provider.signal_host(&ctx, &s, flag, 1).unwrap();
    provider.arm_wake(&ctx, &s, &host_waker).unwrap();
    assert_eq!(provider.run(&ctx, &s).unwrap(), Run::Idle);
    assert_eq!(
        waiter.join().unwrap(),
        (Parked::Woken, 1),
        "woken, and the doorbell is up"
    );
}

/// A pool thread can take a stream's runner and run it, while the device's thread pushes and
/// `synchronize` waits for it to finish.
#[test]
fn a_taken_runner_runs_the_stream_on_another_thread() {
    let provider = CpuProvider::new(KillSwitches::NONE);
    let ctx = open(&provider);
    let s = provider.stream(&ctx, StreamPriority::Compute).unwrap();
    let bells = provider.doorbells(&ctx).unwrap();
    let mut runner = provider.take_runner(&ctx, &s).unwrap().unwrap();
    assert!(provider.take_runner(&ctx, &s).unwrap().is_none());
    assert!(provider.run(&ctx, &s).is_err(), "another thread runs it");
    for i in 0..4 {
        provider.signal_host(&ctx, &s, bells.flag_at(i), 1).unwrap();
    }
    let pool = thread::spawn(move || runner.drain(None).unwrap());
    provider.synchronize(&ctx, &s).unwrap();
    pool.join().unwrap();
    for i in 0..4 {
        assert_eq!(
            bells
                .flag(bells.flag_at(i))
                .unwrap()
                .load(Ordering::Acquire),
            1
        );
    }
}

/// `synchronize` drains a stream on the calling thread, spinning through a doorbell wait that
/// another thread satisfies.
#[test]
fn synchronize_drains_the_stream_here() {
    let provider = CpuProvider::new(KillSwitches::NONE);
    let ctx = open(&provider);
    let s = provider.stream(&ctx, StreamPriority::Compute).unwrap();
    let bells = provider.doorbells(&ctx).unwrap();
    let flag = bells.flag_at(2);
    provider
        .wait_host(&ctx, &s, flag, 1, WaitCmp::Geq, None)
        .unwrap();
    let b = bells.clone();
    let host = thread::spawn(move || {
        thread::sleep(Duration::from_millis(10));
        b.flag(flag).unwrap().store(3, Ordering::Release);
    });
    provider.synchronize(&ctx, &s).unwrap();
    host.join().unwrap();
    assert_eq!(
        bells.flag(flag).unwrap().load(Ordering::Acquire),
        3,
        "Geq with no reset"
    );
}

#[test]
fn a_full_ring_is_refused_never_grown() {
    let provider = CpuProvider::new(KillSwitches::NONE);
    let ctx = open(&provider);
    let s: Stream = provider.stream(&ctx, StreamPriority::Compute).unwrap();
    let flag = provider.doorbells(&ctx).unwrap().flag_at(0);
    for _ in 0..STREAM_CAPACITY {
        provider.signal_host(&ctx, &s, flag, 1).unwrap();
    }
    let full = provider.signal_host(&ctx, &s, flag, 1).unwrap_err();
    assert!(full.message.contains("full"), "{full}");
    assert_eq!(provider.run(&ctx, &s).unwrap(), Run::Idle);
    provider.signal_host(&ctx, &s, flag, 1).unwrap();
}
