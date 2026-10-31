// SPDX-License-Identifier: MIT OR Apache-2.0

//! The stream tests that need a CUDA device. They run with `--ignored` on a host with the NVIDIA
//! driver, and pass with a note on a host without one.

use std::ffi::{c_uint, c_void};

use cudarc::driver::sys::{CUdeviceptr, CUstream};
use syntropy_core::{DeviceId, NodeId};
use syntropy_kernels::{DeviceCtx, KernelProvider, OpenOptions, PowerHint, StreamPriority};

use super::*;
use crate::driver::{PrimaryContext, Status, sym};
use crate::{CudaDevice, CudaProvider};

/// The memory calls these tests need, until the provider's own allocation (COMP-E4-T02).
struct Mem {
    alloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> Status,
    free: unsafe extern "C" fn(CUdeviceptr) -> Status,
    host_alloc: unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> Status,
    host_free: unsafe extern "C" fn(*mut c_void) -> Status,
    memset32: unsafe extern "C" fn(CUdeviceptr, c_uint, usize, CUstream) -> Status,
}

/// An open device, or `None` with a note when the host has no CUDA device.
fn device() -> Option<(CudaProvider, DeviceCtx)> {
    let p = CudaProvider::new();
    let devices = p.enumerate().unwrap();
    if devices.is_empty() {
        eprintln!("skipped: no CUDA device on this host");
        return None;
    }
    eprintln!(
        "device: {} ({:?}), {:?}",
        devices[0].name,
        devices[0].class,
        p.build().driver
    );
    let opts = OpenOptions {
        streams: StreamPlan {
            compute: 1,
            copy_in: 1,
            copy_out: 1,
            collective: 0,
        },
        doorbell_slots: 8,
        capture: true,
        deterministic: true,
        power: PowerHint::Mains,
    };
    let id = DeviceId {
        node: NodeId([0; 32]),
        ordinal: 0,
    };
    let ctx = p.open(id, &opts).unwrap();
    Some((p, ctx))
}

fn cuda(ctx: &DeviceCtx) -> &CudaDevice<PrimaryContext> {
    ctx.downcast_ref().unwrap()
}

fn mem(ctx: &DeviceCtx) -> Mem {
    let lib = cuda(ctx)._context.api().lib();
    Mem {
        alloc: sym(lib, "cuMemAlloc_v2").unwrap(),
        free: sym(lib, "cuMemFree_v2").unwrap(),
        host_alloc: sym(lib, "cuMemHostAlloc").unwrap(),
        host_free: sym(lib, "cuMemFreeHost").unwrap(),
        memset32: sym(lib, "cuMemsetD32Async").unwrap(),
    }
}

/// A device allocation, freed on drop.
struct DevBuf<'a>(&'a Mem, CUdeviceptr);

impl<'a> DevBuf<'a> {
    fn new(m: &'a Mem, bytes: usize) -> Self {
        let mut p: CUdeviceptr = 0;
        // SAFETY: writes one address; the device's context is current on this thread.
        assert_eq!(unsafe { (m.alloc)(&raw mut p, bytes) }, 0);
        Self(m, p)
    }
}

impl Drop for DevBuf<'_> {
    fn drop(&mut self) {
        // SAFETY: allocated by `cuMemAlloc_v2`, freed once, after the streams were drained.
        unsafe { (self.0.free)(self.1) };
    }
}

/// Pinned host memory of `len` `u32`s, mapped for the device through unified addressing.
struct HostBuf<'a>(&'a Mem, *mut u32, usize);

impl<'a> HostBuf<'a> {
    fn new(m: &'a Mem, len: usize) -> Self {
        let mut p: *mut c_void = std::ptr::null_mut();
        // SAFETY: writes one pointer; the device's context is current on this thread.
        assert_eq!(unsafe { (m.host_alloc)(&raw mut p, len * 4, 0) }, 0);
        // SAFETY: the allocation holds `len` `u32`s; zeroing makes them initialised.
        unsafe { std::ptr::write_bytes(p.cast::<u32>(), 0, len) };
        Self(m, p.cast(), len)
    }

    fn addr(&self) -> u64 {
        self.1 as u64
    }

    fn slice(&self) -> &[u32] {
        // SAFETY: `len` initialised `u32`s, read only after the streams that write them are
        // drained.
        unsafe { std::slice::from_raw_parts(self.1, self.2) }
    }

    fn slice_mut(&mut self) -> &mut [u32] {
        // SAFETY: as `slice`, written only while no copy reads it.
        unsafe { std::slice::from_raw_parts_mut(self.1, self.2) }
    }
}

impl Drop for HostBuf<'_> {
    fn drop(&mut self) {
        // SAFETY: allocated by `cuMemHostAlloc`, freed once, after the streams were drained.
        unsafe { (self.0.host_free)(self.1.cast()) };
    }
}

/// The plan's streams have the priorities of RFC-0040 §4.1 on a real device, and the range is
/// what the driver reports.
#[test]
#[ignore = "needs the NVIDIA driver and a GPU"]
fn the_plan_has_compute_highest_on_a_real_device() {
    let Some((_p, ctx)) = device() else { return };
    let d = &cuda(&ctx).streams;
    eprintln!("priority range (least, greatest): {:?}", d.range);
    eprintln!("plan: {:?}", d.priorities());
    eprintln!("cuMemcpyBatchAsync_v2: {}", d.batched());
    let p = d.priorities();
    assert_eq!(p[0], (StreamPriority::Compute, d.range.1));
    assert!(p[1..].iter().all(|&(_, v)| v >= p[0].1));
}

/// Events order work across the compute and copy streams (COMP-E4-T04's stress test).
///
/// Each round, the copy-in stream fills `a` with the round number, the compute stream copies
/// `a` to `b`, and the copy-out stream copies `b` into the round's host slot. `a` and `b` are
/// reused every round, so every hand-off needs its event: copy-in waits until compute has read
/// `a`, compute waits until copy-in has filled `a` and copy-out has read `b`, and copy-out waits
/// until compute has written `b`. Without any one of those waits, a slot would hold another
/// round's value or a mix.
#[test]
#[ignore = "needs the NVIDIA driver and a GPU"]
fn events_order_work_across_the_compute_and_copy_streams() {
    let Some((p, ctx)) = device() else { return };
    let m = mem(&ctx);
    let d = &cuda(&ctx).streams;
    const N: usize = 1 << 20; // u32s per buffer: 4 MiB
    const ROUNDS: usize = 32;
    let a = DevBuf::new(&m, N * 4);
    let b = DevBuf::new(&m, N * 4);
    let mut host = HostBuf::new(&m, N * ROUNDS);
    host.slice_mut().fill(u32::MAX);

    let copy_in = p.stream(&ctx, StreamPriority::CopyIn).unwrap();
    let compute = p.stream(&ctx, StreamPriority::Compute).unwrap();
    let copy_out = p.stream(&ctx, StreamPriority::CopyOut).unwrap();
    let raw_in = d.raw_stream(&copy_in).unwrap();
    let [filled, a_read, b_written, b_read] = [(); 4].map(|()| p.event(&ctx).unwrap());

    for round in 0..ROUNDS {
        let value = u32::try_from(round).unwrap() + 1;
        // Copy-in: refill `a` once compute has read it.
        if round > 0 {
            p.wait(&ctx, &copy_in, &a_read).unwrap();
        }
        // SAFETY: `a` holds `N` u32s; `raw_in` is a live stream of the device.
        let s = unsafe {
            (m.memset32)(
                a.1,
                value,
                N,
                std::ptr::with_exposed_provenance_mut(raw_in as usize),
            )
        };
        assert_eq!(s, 0);
        p.record(&ctx, &copy_in, &filled).unwrap();
        // Compute: `a` to `b`, once `a` is filled and copy-out has read `b`.
        p.wait(&ctx, &compute, &filled).unwrap();
        if round > 0 {
            p.wait(&ctx, &compute, &b_read).unwrap();
        }
        // SAFETY: both buffers hold `N * 4` bytes and outlive the streams' drain below.
        unsafe { d.copy(&compute, b.1, a.1, (N * 4) as u64) }.unwrap();
        p.record(&ctx, &compute, &a_read).unwrap();
        p.record(&ctx, &compute, &b_written).unwrap();
        // Copy-out: `b` to the round's host slot.
        p.wait(&ctx, &copy_out, &b_written).unwrap();
        let slot = host.addr() + (round * N * 4) as u64;
        // SAFETY: the slot is `N * 4` bytes of pinned host memory inside `host`.
        unsafe { d.copy(&copy_out, slot, b.1, (N * 4) as u64) }.unwrap();
        p.record(&ctx, &copy_out, &b_read).unwrap();
    }
    // `poll` never blocks: spin on it until copy-out's last record completes.
    let t = std::time::Instant::now();
    while !p.poll(&ctx, &b_read).unwrap() {
        assert!(t.elapsed().as_secs() < 30, "the last round never completed");
        std::hint::spin_loop();
    }
    for s in [&copy_in, &compute, &copy_out] {
        p.synchronize(&ctx, s).unwrap();
    }
    let bad: Vec<(usize, u32)> = host
        .slice()
        .chunks(N)
        .enumerate()
        .filter_map(|(round, slot)| {
            let want = u32::try_from(round).unwrap() + 1;
            slot.iter().find(|&&v| v != want).map(|&v| (round, v))
        })
        .collect();
    assert!(bad.is_empty(), "rounds with a wrong value: {bad:?}");
    eprintln!(
        "{ROUNDS} rounds of {} MiB each ordered correctly",
        (N * 4) >> 20
    );
}

/// `copy_batch` moves the same bytes as a loop of copies, and `elapsed` measures device time
/// between two timing events.
#[test]
#[ignore = "needs the NVIDIA driver and a GPU"]
fn copy_batch_moves_the_same_bytes_as_a_loop_of_copies() {
    let Some((p, ctx)) = device() else { return };
    let m = mem(&ctx);
    let d = &cuda(&ctx).streams;
    const N: usize = 1 << 18; // u32s
    // 300 pairs, more than two chunks, of varied lengths and offsets, some empty.
    let pairs: Vec<(usize, usize, usize)> = (0..300)
        .map(|i| {
            let len = if i % 37 == 5 { 0 } else { 1 + (i * 97) % 600 };
            (i * 800, (i * 613) % (N - 700), len)
        })
        .collect();
    let mut src = HostBuf::new(&m, N);
    for (i, v) in src.slice_mut().iter_mut().enumerate() {
        *v = u32::try_from(i).unwrap().wrapping_mul(2_654_435_761);
    }
    let (looped, batched) = (DevBuf::new(&m, N * 4), DevBuf::new(&m, N * 4));
    let (out_loop, out_batch) = (HostBuf::new(&m, N), HostBuf::new(&m, N));
    let s = p.stream(&ctx, StreamPriority::CopyIn).unwrap();
    let (t0, t1) = (p.timing_event(&ctx).unwrap(), p.timing_event(&ctx).unwrap());
    let at = |base: u64, i: usize| base + (i * 4) as u64;
    for &(dst, from, len) in &pairs {
        // SAFETY: every range lies inside its `N`-u32 buffer.
        unsafe {
            d.copy(
                &s,
                at(looped.1, dst),
                at(src.addr(), from),
                (len * 4) as u64,
            )
        }
        .unwrap();
    }
    p.record(&ctx, &s, &t0).unwrap();
    let batch = pairs
        .iter()
        .map(|&(dst, from, len)| Ok((at(batched.1, dst), at(src.addr(), from), (len * 4) as u64)));
    // SAFETY: as above.
    unsafe { d.copy_batch(&s, batch) }.unwrap();
    p.record(&ctx, &s, &t1).unwrap();
    // SAFETY: as above; whole buffers back to the host.
    unsafe {
        d.copy(&s, out_loop.addr(), looped.1, (N * 4) as u64)
            .unwrap();
        d.copy(&s, out_batch.addr(), batched.1, (N * 4) as u64)
            .unwrap();
    }
    p.synchronize(&ctx, &s).unwrap();
    eprintln!(
        "batched: {}; {} pairs; copy_batch took {:?} of device time",
        d.batched(),
        pairs.len(),
        p.elapsed(&ctx, &t0, &t1)
    );
    assert!(p.elapsed(&ctx, &t0, &t1).is_some());
    assert_eq!(out_loop.slice(), out_batch.slice());
    // And the bytes are the source's, not two equal wrong answers.
    let &(dst, from, len) = pairs.iter().find(|p| p.2 > 0).unwrap();
    assert_eq!(
        &out_batch.slice()[dst..dst + len],
        &src.slice()[from..from + len]
    );
}
