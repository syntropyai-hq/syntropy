// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::atomic::AtomicI32;
use std::sync::{Arc, Mutex};

use syntropy_core::{DeviceId, NodeId};
use syntropy_kernels::{KernelProvider, OpenOptions};

use super::*;
use crate::driver::RawDevice;

/// One memory call a [`Recorder`] saw.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Call {
    DeviceAlloc(u64),
    DeviceFree(u64),
    HostAlloc(u64),
    HostFree(u64),
    Register {
        host: u64,
        bytes: u64,
    },
    Unregister(u64),
    DevicePointer(u64),
    /// The primary context was released (from [`MemFake`]'s context).
    ContextReleased,
}

impl Call {
    /// Whether the call takes memory from the driver.
    fn takes(&self) -> bool {
        matches!(
            self,
            Call::DeviceAlloc(_) | Call::HostAlloc(_) | Call::Register { .. }
        )
    }
}

/// A stand-in for the driver's memory calls: it logs each call, hands out addresses, and fails
/// the call named in `fail`.
#[derive(Clone, Default)]
pub(crate) struct Recorder {
    pub(crate) log: Arc<Mutex<Vec<Call>>>,
    pub(crate) fail: Arc<Mutex<Option<&'static str>>>,
    next: Arc<Mutex<u64>>,
}

/// Device slabs come back aligned to 256 bytes and no more, as the driver guarantees.
const DEVICE_BASE: u64 = 0x7000_0000_0100;
/// Host-mapped pages: the device sees them at a different address, to tell the two apart.
const HOST_BASE: u64 = 0x5000_0000_0000;
const MAPPED_OFFSET: u64 = 0x1000_0000_0000;

impl Recorder {
    fn call(&self, call: Call, name: &'static str) -> Result<(), Fault> {
        self.log.lock().unwrap().push(call);
        if *self.fail.lock().unwrap() == Some(name) {
            return Err(Fault {
                call: name,
                error: "CUDA_ERROR_OUT_OF_MEMORY".into(),
                no_device: false,
            });
        }
        Ok(())
    }

    fn calls(&self) -> Vec<Call> {
        self.log.lock().unwrap().clone()
    }

    fn fail(&self, name: &'static str) {
        *self.fail.lock().unwrap() = Some(name);
    }

    fn bump(&self, bytes: u64) -> u64 {
        let mut next = self.next.lock().unwrap();
        let at = *next;
        // Keep every allocation 1 MiB apart and 256-byte aligned only.
        *next += bytes.div_ceil(1 << 20) * (1 << 20) + (1 << 20);
        at
    }
}

impl MemoryOps for Recorder {
    fn device_alloc(&self, bytes: u64) -> Result<u64, Fault> {
        self.call(Call::DeviceAlloc(bytes), "cuMemAlloc_v2")?;
        Ok(DEVICE_BASE + self.bump(bytes))
    }
    fn device_free(&self, addr: u64) -> Result<(), Fault> {
        self.call(Call::DeviceFree(addr), "cuMemFree_v2")
    }
    fn host_alloc(&self, bytes: u64) -> Result<u64, Fault> {
        self.call(Call::HostAlloc(bytes), "cuMemHostAlloc")?;
        Ok(HOST_BASE + self.bump(bytes))
    }
    fn host_free(&self, host: u64) -> Result<(), Fault> {
        self.call(Call::HostFree(host), "cuMemFreeHost")
    }
    fn host_register(&self, host: u64, bytes: u64) -> Result<(), Fault> {
        self.call(Call::Register { host, bytes }, "cuMemHostRegister_v2")
    }
    fn host_unregister(&self, host: u64) -> Result<(), Fault> {
        self.call(Call::Unregister(host), "cuMemHostUnregister")
    }
    fn device_pointer(&self, host: u64) -> Result<u64, Fault> {
        self.call(Call::DevicePointer(host), "cuMemHostGetDevicePointer_v2")?;
        Ok(host + MAPPED_OFFSET)
    }
}

/// A driver with one sm_89 device whose memory calls go to a shared [`Recorder`].
#[derive(Clone, Default)]
struct MemFake {
    rec: Recorder,
    retained: Arc<AtomicI32>,
    /// Devices beyond the first.
    more: i32,
}

struct MemFakeContext(Recorder, Arc<AtomicI32>);

impl Drop for MemFakeContext {
    fn drop(&mut self) {
        self.1.fetch_sub(1, Ordering::SeqCst);
        self.0.log.lock().unwrap().push(Call::ContextReleased);
    }
}

impl Driver for MemFake {
    type Context = MemFakeContext;
    fn present(&self) -> bool {
        true
    }
    fn init(&self) -> Result<(), Fault> {
        Ok(())
    }
    fn version(&self) -> Result<i32, Fault> {
        Ok(13_000)
    }
    fn devices(&self) -> Result<Vec<RawDevice>, Fault> {
        Ok((0..=self.more)
            .map(|ordinal| RawDevice {
                ordinal,
                name: format!("GPU {ordinal}"),
                uuid: [7 + ordinal as u8; 16],
                cc: (8, 9),
                pci: (0, 1 + ordinal as u32, 0),
                total: 16 << 30,
                free: 15 << 30,
                integrated: false,
                vmm: true,
                host_register: true,
                peer: false,
                numa: None,
            })
            .collect())
    }
    fn retain(&self, _: i32) -> Result<MemFakeContext, Fault> {
        self.retained.fetch_add(1, Ordering::SeqCst);
        Ok(MemFakeContext(self.rec.clone(), self.retained.clone()))
    }
}

impl MemoryDriver for MemFake {
    fn memory(&self, _: &MemFakeContext) -> Result<Arc<dyn MemoryOps>, Fault> {
        Ok(Arc::new(self.rec.clone()))
    }
}

fn opts() -> OpenOptions {
    OpenOptions {
        streams: syntropy_kernels::StreamPlan {
            compute: 1,
            copy_in: 1,
            copy_out: 1,
            collective: 0,
        },
        doorbell_slots: 8,
        capture: true,
        deterministic: true,
        power: syntropy_kernels::PowerHint::Mains,
    }
}

fn dev0() -> DeviceId {
    DeviceId {
        node: NodeId([0; 32]),
        ordinal: 0,
    }
}

/// A provider over a fresh [`MemFake`], its recorder, the events its hook saw, and an open
/// device.
fn setup() -> (
    CudaProvider<MemFake>,
    Recorder,
    Arc<Mutex<Vec<MemEvent>>>,
    DeviceCtx,
) {
    let fake = MemFake::default();
    let rec = fake.rec.clone();
    let events = Arc::new(Mutex::new(Vec::new()));
    let seen = events.clone();
    let p = CudaProvider::with_driver(fake)
        .with_alloc_hook(Arc::new(move |e: &MemEvent| seen.lock().unwrap().push(*e)));
    let ctx = p.open(dev0(), &opts()).unwrap();
    (p, rec, events, ctx)
}

fn req(kind: MemKind, bytes: u64, align: u64) -> AllocRequest {
    AllocRequest {
        bytes: Bytes(bytes),
        kind,
        align: Bytes(align),
        reserve_only: false,
    }
}

fn region(addr: u64, len: u64) -> HostRegion {
    HostRegion {
        addr,
        len: Bytes(len),
    }
}

/// A host address for registrations: page-aligned, never dereferenced by the recorder.
const ARENA: u64 = 0x6000_0000_0000;
const MIB: u64 = 1 << 20;

#[test]
fn padded_and_round_up() {
    assert_eq!(padded(1000, 256, DEVICE_ALIGN), Some(1000));
    assert_eq!(padded(1000, 0, DEVICE_ALIGN), Some(1000));
    assert_eq!(padded(1000, 4096, DEVICE_ALIGN), Some(1000 + 4096 - 256));
    assert_eq!(padded(0, 256, DEVICE_ALIGN), None);
    assert_eq!(padded(1000, 384, DEVICE_ALIGN), None);
    assert_eq!(padded(u64::MAX, 1 << 20, DEVICE_ALIGN), None);
    assert_eq!(round_up(0x100, 0x1000), Some(0x1000));
    assert_eq!(round_up(0x1000, 0x1000), Some(0x1000));
    assert_eq!(round_up(u64::MAX - 10, 0x1000), None);
}

/// A `Device` request is one fixed slab from `cuMemAlloc_v2`; an alignment above the driver's
/// 256 bytes pads the slab, and the buffer starts at the first aligned byte inside it.
#[test]
fn device_alloc_is_one_aligned_slab() {
    let (p, rec, _, ctx) = setup();
    let small = p.alloc(&ctx, req(MemKind::Device, 1000, 256)).unwrap();
    assert_eq!(small.kind, MemKind::Device);
    assert_eq!(small.bytes, Bytes(1000));
    assert_eq!(small.addr % 256, 0);

    let big = p
        .alloc(&ctx, req(MemKind::Device, 3 * MIB, 2 * MIB))
        .unwrap();
    assert_eq!(big.addr % (2 * MIB), 0, "{:#x}", big.addr);
    let Call::DeviceAlloc(taken) = rec.calls()[1] else {
        panic!("{:?}", rec.calls())
    };
    assert_eq!(taken, 3 * MIB + 2 * MIB - 256);
    // The aligned buffer ends inside the slab the driver gave.
    let base = p.resolve(&ctx, slice(&big, 0, 1)).unwrap().device;
    assert_eq!(base, big.addr);
    assert_eq!(
        rec.calls()
            .iter()
            .filter(|c| matches!(c, Call::DeviceAlloc(_)))
            .count(),
        2
    );
    assert_ne!(small.handle, big.handle);
    let s = p.memory_stats(&ctx).unwrap();
    assert_eq!((s.device_buffers, s.device_bytes), (2, 1000 + 3 * MIB));
}

/// A driver address aligned less than the driver documents is caught: the padded slab
/// cannot hold the buffer at its aligned start, so the slab goes back and `alloc` fails.
#[test]
fn a_slab_too_small_for_its_aligned_start_is_returned() {
    let (p, rec, events, ctx) = setup();
    // The next slab starts 128 bytes past a 256-byte boundary.
    *rec.next.lock().unwrap() = 0xF80;
    let e = p.alloc(&ctx, req(MemKind::Device, MIB, 4096)).unwrap_err();
    assert_eq!(e.code, codes::ALLOC_FAILED);
    assert!(
        matches!(rec.calls()[..], [Call::DeviceAlloc(_), Call::DeviceFree(_)]),
        "{:?}",
        rec.calls()
    );
    assert!(events.lock().unwrap().is_empty());
}

/// `HostMapped` doorbell pages come from `cuMemHostAlloc` in whole pages, and the buffer's
/// address is the device's mapping of them.
#[test]
fn host_mapped_alloc_gives_device_mapped_pages() {
    let (p, rec, _, ctx) = setup();
    let page = p.alloc(&ctx, req(MemKind::HostMapped, 64, 64)).unwrap();
    assert_eq!(page.kind, MemKind::HostMapped);
    assert_eq!(page.bytes, Bytes(64));
    let calls = rec.calls();
    let Call::HostAlloc(taken) = calls[0] else {
        panic!("{calls:?}")
    };
    assert_eq!(taken, host_page());
    let Call::DevicePointer(host) = calls[1] else {
        panic!("{calls:?}")
    };
    assert_eq!(page.addr, host + MAPPED_OFFSET);
    let r = p.resolve(&ctx, slice(&page, 8, 4)).unwrap();
    assert_eq!(r.host, Some(host + 8));
    assert_eq!(r.device, page.addr + 8);
    assert_eq!(p.memory_stats(&ctx).unwrap().host_buffers, 1);
}

fn slice(buf: &DeviceBuffer, offset: u64, bytes: u64) -> BufferSlice {
    BufferSlice {
        buffer: buf.handle,
        offset: Bytes(offset),
        bytes: Bytes(bytes),
    }
}

/// A request the provider cannot meet fails before any driver call: `AllocFailed` for a bad
/// size, `Internal` for an alignment that is not a power of two (a caller's error), and
/// `Unsupported` for kinds and modes later tasks add.
#[test]
fn bad_requests_fail_without_a_driver_call() {
    let (p, rec, _, ctx) = setup();
    for bad in [
        req(MemKind::Device, 0, 256),
        req(MemKind::Device, u64::MAX, 1 << 20),
        req(MemKind::HostMapped, 0, 64),
    ] {
        let e = p.alloc(&ctx, bad).unwrap_err();
        assert_eq!(e.code, codes::ALLOC_FAILED, "{bad:?}");
        assert_eq!(e.class, ErrorClass::Capacity);
    }
    for bad in [
        req(MemKind::Device, 4096, 384),
        req(MemKind::HostMapped, 4096, 3),
    ] {
        let e = p.alloc(&ctx, bad).unwrap_err();
        assert_eq!(e.class, ErrorClass::Internal, "{bad:?}");
    }
    let reserve = AllocRequest {
        reserve_only: true,
        ..req(MemKind::Device, MIB, 256)
    };
    assert_eq!(p.alloc(&ctx, reserve), Err(unsupported()));
    assert_eq!(
        p.alloc(&ctx, req(MemKind::Unified, MIB, 256)),
        Err(unsupported())
    );
    assert_eq!(rec.calls(), []);
}

/// Driver failures map to RFC-0040 §13's codes: `AllocFailed` for `alloc` and `import`,
/// `RegisterFailed` for `register_host`, both class `Capacity`, naming the failed call.
#[test]
fn driver_failures_map_to_their_codes() {
    let (p, rec, events, ctx) = setup();
    rec.fail("cuMemAlloc_v2");
    let e = p.alloc(&ctx, req(MemKind::Device, MIB, 256)).unwrap_err();
    assert_eq!(
        (e.code, e.class),
        (codes::ALLOC_FAILED, ErrorClass::Capacity)
    );
    assert!(e.message.contains("cuMemAlloc_v2"), "{e}");
    assert!(e.message.contains("CUDA_ERROR_OUT_OF_MEMORY"), "{e}");

    rec.fail("cuMemHostRegister_v2");
    let e = p.register_host(&ctx, region(ARENA, MIB)).unwrap_err();
    assert_eq!(
        (e.code, e.class),
        (codes::REGISTER_FAILED, ErrorClass::Capacity)
    );
    let e = p.import(&ctx, region(ARENA, MIB)).unwrap_err();
    assert_eq!(
        (e.code, e.class),
        (codes::ALLOC_FAILED, ErrorClass::Capacity)
    );
    // A failed registration holds nothing, so the range can be registered once the driver
    // allows it.
    *rec.fail.lock().unwrap() = None;
    p.register_host(&ctx, region(ARENA, MIB)).unwrap();
    assert_eq!(
        events.lock().unwrap().len(),
        1,
        "only the success is an event"
    );
}

/// A failure after the driver has given memory hands it back: no path leaks.
#[test]
fn a_failure_after_the_driver_gave_memory_releases_it() {
    let (p, rec, _, ctx) = setup();
    rec.fail("cuMemHostGetDevicePointer_v2");
    assert!(p.alloc(&ctx, req(MemKind::HostMapped, 4096, 4096)).is_err());
    let calls = rec.calls();
    let Call::HostAlloc(_) = calls[0] else {
        panic!("{calls:?}")
    };
    assert!(
        matches!(calls[2], Call::HostFree(h) if h == HOST_BASE),
        "{calls:?}"
    );

    rec.log.lock().unwrap().clear();
    assert!(p.import(&ctx, region(ARENA, MIB)).is_err());
    assert_eq!(rec.calls().last(), Some(&Call::Unregister(ARENA)));
    assert_eq!(p.memory_stats(&ctx).unwrap(), MemStats::default());
}

/// `free` returns each buffer to the driver once. A second `free` of the same handle, a handle
/// of another device, and a registration's handle are ignored.
#[test]
fn free_releases_once_and_ignores_what_it_does_not_hold() {
    let (p, rec, _, ctx) = setup();
    let buf = p.alloc(&ctx, req(MemKind::Device, MIB, 256)).unwrap();
    let again = DeviceBuffer { ..buf };
    let reg = p.register_host(&ctx, region(ARENA, MIB)).unwrap();
    p.free(&ctx, buf);
    p.free(&ctx, again);
    p.free(
        &ctx,
        DeviceBuffer {
            addr: 0,
            bytes: Bytes(MIB),
            kind: MemKind::Device,
            handle: reg.handle,
        },
    );
    let frees: Vec<_> = rec
        .calls()
        .into_iter()
        .filter(|c| matches!(c, Call::DeviceFree(_) | Call::Unregister(_)))
        .collect();
    assert_eq!(frees, [Call::DeviceFree(DEVICE_BASE)]);
    assert_eq!(p.memory_stats(&ctx).unwrap().registrations, 1);

    // Another device's buffer: handles are unique across devices.
    let (q, rec_q, _, ctx_q) = setup();
    let theirs = q.alloc(&ctx_q, req(MemKind::Device, MIB, 256)).unwrap();
    let mine = p.alloc(&ctx, req(MemKind::Device, MIB, 256)).unwrap();
    assert_ne!(theirs.handle, mine.handle);
    p.free(
        &ctx,
        DeviceBuffer {
            addr: theirs.addr,
            bytes: theirs.bytes,
            kind: theirs.kind,
            handle: theirs.handle,
        },
    );
    assert_eq!(p.memory_stats(&ctx).unwrap().device_buffers, 1);
    assert!(
        !rec_q
            .calls()
            .iter()
            .any(|c| matches!(c, Call::DeviceFree(_)))
    );
    q.free(&ctx_q, theirs);
}

/// `register_host` takes whole pages and refuses a range that overlaps one it holds, before
/// any driver call, so no page, and no DMA source, belongs to two registrations
/// (RFC-0020 §6.1). Adjacent units are fine.
#[test]
fn registrations_are_whole_pages_and_never_overlap() {
    let (p, rec, _, ctx) = setup();
    for bad in [
        region(0, MIB),
        region(ARENA, 0),
        region(ARENA + 8, MIB),
        region(ARENA, MIB + 8),
        region(u64::MAX - 4095, 8192),
    ] {
        let e = p.register_host(&ctx, bad).unwrap_err();
        assert_eq!(e.class, ErrorClass::Internal, "{bad:?}");
    }
    assert_eq!(rec.calls(), []);

    let a = p.register_host(&ctx, region(ARENA, 16 * MIB)).unwrap();
    let b = p
        .register_host(&ctx, region(ARENA + 16 * MIB, 16 * MIB))
        .unwrap();
    rec.log.lock().unwrap().clear();
    for overlapping in [
        region(ARENA, 16 * MIB),
        region(ARENA + host_page(), host_page()),
        region(ARENA - host_page(), 2 * host_page()),
        region(ARENA + 15 * MIB, 2 * MIB),
        region(ARENA - MIB, 64 * MIB),
    ] {
        let e = p.register_host(&ctx, overlapping).unwrap_err();
        assert_eq!(e.class, ErrorClass::Internal, "{overlapping:?}");
        assert!(e.message.contains("overlaps"), "{e}");
        assert!(p.import(&ctx, overlapping).is_err());
    }
    assert_eq!(rec.calls(), []);
    // A neighbour on either side is a unit of its own.
    p.register_host(&ctx, region(ARENA - MIB, MIB)).unwrap();
    p.register_host(&ctx, region(ARENA + 32 * MIB, MIB))
        .unwrap();

    // A DMA source inside one unit names it; one that crosses two units names none.
    assert_eq!(p.registration_for(&ctx, ARENA, 16 * MIB), Some(a.handle));
    assert_eq!(
        p.registration_for(&ctx, ARENA + 20 * MIB, 4096),
        Some(b.handle)
    );
    assert_eq!(p.registration_for(&ctx, ARENA + 15 * MIB, 2 * MIB), None);
    assert_eq!(p.registration_for(&ctx, ARENA + 64 * MIB, 4096), None);
    assert_eq!(p.registration_for(&ctx, u64::MAX, 2), None);
}

/// A shrinking arena unregisters whole units: each once, after which the range can be
/// registered again. A second unregister of the same unit is refused and calls nothing.
#[test]
fn unregister_releases_a_unit_once() {
    let (p, rec, _, ctx) = setup();
    let a = p.register_host(&ctx, region(ARENA, MIB)).unwrap();
    let copy = Registration {
        region: a.region,
        handle: a.handle,
    };
    p.unregister_host(&ctx, a).unwrap();
    let e = p.unregister_host(&ctx, copy).unwrap_err();
    assert_eq!(e.class, ErrorClass::Internal);
    assert_eq!(
        rec.calls()
            .iter()
            .filter(|c| matches!(c, Call::Unregister(_)))
            .count(),
        1
    );
    assert_eq!(p.registration_for(&ctx, ARENA, 4096), None);
    p.register_host(&ctx, region(ARENA, MIB)).unwrap();

    // A buffer handle is not a registration.
    let buf = p.alloc(&ctx, req(MemKind::Device, MIB, 256)).unwrap();
    let e = p
        .unregister_host(
            &ctx,
            Registration {
                region: region(ARENA, MIB),
                handle: buf.handle,
            },
        )
        .unwrap_err();
    assert_eq!(e.class, ErrorClass::Internal);
}

/// `import` registers the tier manager's range device-mapped and returns the device's address
/// of it, without copying; `free` unregisters it.
#[test]
fn import_maps_the_tier_managers_memory() {
    let (p, rec, _, ctx) = setup();
    let buf = p.import(&ctx, region(ARENA, 8 * MIB)).unwrap();
    assert_eq!(buf.kind, MemKind::HostMapped);
    assert_eq!(buf.addr, ARENA + MAPPED_OFFSET);
    assert_eq!(buf.bytes, Bytes(8 * MIB));
    assert_eq!(
        rec.calls(),
        [
            Call::Register {
                host: ARENA,
                bytes: 8 * MIB,
            },
            Call::DevicePointer(ARENA)
        ]
    );
    assert_eq!(
        p.resolve(&ctx, slice(&buf, MIB, 4096)),
        Some(Resolved {
            device: ARENA + MAPPED_OFFSET + MIB,
            host: Some(ARENA + MIB)
        })
    );
    assert_eq!(p.registration_for(&ctx, ARENA, 8 * MIB), Some(buf.handle));
    p.free(&ctx, buf);
    assert_eq!(rec.calls().last(), Some(&Call::Unregister(ARENA)));
    assert_eq!(p.registration_for(&ctx, ARENA, 4096), None);
}

/// RFC-0020 §6 registers each unit once per device in `dma_devices`. Registrations are
/// portable, so the driver registers a range once: the second device shares it with a handle
/// of its own, one device's release or close leaves the other's in place, and the driver
/// unregisters the range after the last release. A range that overlaps another device's
/// without being exactly it is refused.
#[test]
fn devices_share_one_driver_registration_per_range() {
    let fake = MemFake {
        more: 1,
        ..MemFake::default()
    };
    let rec = fake.rec.clone();
    let p = CudaProvider::with_driver(fake);
    let a = p.open(dev0(), &opts()).unwrap();
    let b = p
        .open(
            DeviceId {
                ordinal: 1,
                ..dev0()
            },
            &opts(),
        )
        .unwrap();
    let registers = |rec: &Recorder| {
        rec.calls()
            .iter()
            .filter(|c| matches!(c, Call::Register { .. }))
            .count()
    };
    let unregisters = |rec: &Recorder| {
        rec.calls()
            .iter()
            .filter(|c| matches!(c, Call::Unregister(_)))
            .count()
    };
    let ra = p.register_host(&a, region(ARENA, 16 * MIB)).unwrap();
    let rb = p.register_host(&b, region(ARENA, 16 * MIB)).unwrap();
    assert_ne!(ra.handle, rb.handle);
    assert_eq!(registers(&rec), 1);
    assert_eq!(p.host_ranges.users(ARENA), 2);
    // Each device sees its own registration.
    assert_eq!(p.registration_for(&a, ARENA, MIB), Some(ra.handle));
    assert_eq!(p.registration_for(&b, ARENA, MIB), Some(rb.handle));
    // A partial overlap with the other device's unit is refused before the driver.
    let e = p
        .register_host(&b, region(ARENA + 8 * MIB, 16 * MIB))
        .unwrap_err();
    assert_eq!(e.class, ErrorClass::Internal, "{e}");
    assert!(p.import(&b, region(ARENA - MIB, 2 * MIB)).is_err());
    assert_eq!(registers(&rec), 1);

    // A's release keeps B's registration; B's release unregisters.
    p.unregister_host(&a, ra).unwrap();
    assert_eq!(unregisters(&rec), 0);
    assert_eq!(p.host_ranges.users(ARENA), 1);
    p.unregister_host(&b, rb).unwrap();
    assert_eq!(unregisters(&rec), 1);
    assert_eq!(p.host_ranges.users(ARENA), 0);

    // An import on both devices: each reads its own device pointer; A's close leaves B's.
    let ia = p.import(&a, region(ARENA, 8 * MIB)).unwrap();
    let ib = p.import(&b, region(ARENA, 8 * MIB)).unwrap();
    assert_eq!(registers(&rec), 2);
    assert_eq!(
        rec.calls()
            .iter()
            .filter(|c| matches!(c, Call::DevicePointer(h) if *h == ARENA))
            .count(),
        2
    );
    drop(a);
    assert_eq!(unregisters(&rec), 1);
    assert_eq!(p.host_ranges.users(ARENA), 1);
    assert!(p.resolve(&b, slice(&ib, 0, MIB)).is_some());
    p.free(&b, ib);
    assert_eq!(unregisters(&rec), 2);
    let _ = ia;
}

#[test]
fn the_page_size_comes_from_the_auxiliary_vector() {
    let w = std::mem::size_of::<usize>();
    let pair = |t: usize, v: usize| [t.to_ne_bytes(), v.to_ne_bytes()].concat();
    let auxv = [pair(3, 0x40), pair(6, 65536), pair(0, 0)].concat();
    assert_eq!(auxv.len(), 6 * w);
    assert_eq!(auxv_page_size(&auxv), Some(65536));
    assert_eq!(auxv_page_size(&pair(6, 4096)), Some(4096));
    assert_eq!(auxv_page_size(&pair(6, 3000)), None);
    assert_eq!(auxv_page_size(&pair(3, 4096)), None);
    assert_eq!(auxv_page_size(&[]), None);
    let page = host_page();
    assert!(page >= MIN_HOST_PAGE && page.is_power_of_two(), "{page}");
}

/// A `HostMapped` buffer aligned above the page starts inside the pages the driver gave, and
/// its device address is the device's mapping of that same byte.
#[test]
fn host_mapped_alignment_moves_host_and_device_addresses_together() {
    let (p, rec, _, ctx) = setup();
    // The next pages start one page past a 64 KiB boundary.
    *rec.next.lock().unwrap() = host_page();
    let buf = p
        .alloc(&ctx, req(MemKind::HostMapped, 4096, 64 << 10))
        .unwrap();
    let base = HOST_BASE + host_page();
    let host = p.resolve(&ctx, slice(&buf, 0, 1)).unwrap().host.unwrap();
    assert_eq!(host % (64 << 10), 0);
    assert!(host > base);
    assert_eq!(buf.addr, host + MAPPED_OFFSET);
    let Call::HostAlloc(taken) = rec.calls()[0] else {
        panic!("{:?}", rec.calls())
    };
    assert!(host + 4096 <= base + taken);
}

/// `resolve` refuses a slice past the buffer's end, a registration, and a handle it does not
/// hold.
#[test]
fn resolve_checks_bounds_and_handles() {
    let (p, _, _, ctx) = setup();
    let buf = p.alloc(&ctx, req(MemKind::Device, 4096, 256)).unwrap();
    assert_eq!(
        p.resolve(&ctx, slice(&buf, 4000, 96)),
        Some(Resolved {
            device: buf.addr + 4000,
            host: None
        })
    );
    assert_eq!(p.resolve(&ctx, slice(&buf, 4000, 97)), None);
    assert_eq!(p.resolve(&ctx, slice(&buf, u64::MAX, 2)), None);
    let reg = p.register_host(&ctx, region(ARENA, MIB)).unwrap();
    let as_buffer = BufferSlice {
        buffer: reg.handle,
        offset: Bytes(0),
        bytes: Bytes(1),
    };
    assert_eq!(p.resolve(&ctx, as_buffer), None);
    assert_eq!(
        p.resolve(
            &ctx,
            BufferSlice {
                buffer: u64::MAX,
                ..as_buffer
            }
        ),
        None
    );
}

/// Closing a device releases everything the tier manager did not, each once, before the
/// primary context is released.
#[test]
fn closing_the_device_releases_what_is_left_before_the_context() {
    let (p, rec, events, ctx) = setup();
    p.alloc(&ctx, req(MemKind::Device, MIB, 256)).unwrap();
    p.alloc(&ctx, req(MemKind::HostMapped, 4096, 4096)).unwrap();
    p.register_host(&ctx, region(ARENA, MIB)).unwrap();
    p.import(&ctx, region(ARENA + MIB, MIB)).unwrap();
    let freed = p.alloc(&ctx, req(MemKind::Device, MIB, 256)).unwrap();
    p.free(&ctx, freed);
    rec.log.lock().unwrap().clear();
    drop(ctx);
    let calls = rec.calls();
    assert_eq!(calls.len(), 5, "{calls:?}");
    assert_eq!(calls.last(), Some(&Call::ContextReleased));
    assert_eq!(
        calls
            .iter()
            .filter(|c| matches!(c, Call::DeviceFree(_)))
            .count(),
        1
    );
    assert_eq!(
        calls
            .iter()
            .filter(|c| matches!(c, Call::HostFree(_)))
            .count(),
        1
    );
    assert_eq!(
        calls
            .iter()
            .filter(|c| matches!(c, Call::Unregister(_)))
            .count(),
        2
    );
    let events = events.lock().unwrap();
    let taken = events
        .iter()
        .filter(|e| !matches!(e, MemEvent::Release { .. }))
        .count();
    let released = events
        .iter()
        .filter(|e| matches!(e, MemEvent::Release { .. }))
        .count();
    assert_eq!((taken, released), (5, 5));
    drop(p);
}

/// No provider allocation outside the tier manager's requests (COMP-E4-T02; RFC-0001 §9.1):
/// enumeration, `open` and every method other than `alloc`, `register_host` and `import` take
/// nothing from the driver, and each request takes exactly one allocation, which the hook sees.
#[test]
fn only_the_tier_managers_requests_allocate() {
    let (p, rec, events, ctx) = setup();
    p.enumerate().unwrap();
    let _ = p.build();
    let _ = p.memory_info(&ctx);
    let _ = p.stream(&ctx, syntropy_kernels::StreamPriority::Compute);
    let _ = p.event(&ctx);
    let _ = p.health(&ctx);
    let other = p.open(dev0(), &opts()).unwrap();
    drop(other);
    assert!(
        !rec.calls().iter().any(Call::takes),
        "allocation outside a request: {:?}",
        rec.calls()
    );

    let requests = 4;
    let a = p.alloc(&ctx, req(MemKind::Device, 3 * MIB, 4096)).unwrap();
    let b = p.alloc(&ctx, req(MemKind::HostMapped, 4096, 4096)).unwrap();
    let r = p.register_host(&ctx, region(ARENA, 4 * MIB)).unwrap();
    let i = p.import(&ctx, region(ARENA + 4 * MIB, MIB)).unwrap();
    let taken: Vec<Call> = rec.calls().into_iter().filter(Call::takes).collect();
    assert_eq!(taken.len(), requests, "{taken:?}");
    let hooked: Vec<MemEvent> = events.lock().unwrap().clone();
    assert_eq!(hooked.len(), requests);
    assert_eq!(
        hooked
            .iter()
            .map(|e| match *e {
                MemEvent::Alloc { handle, .. }
                | MemEvent::Register { handle, .. }
                | MemEvent::Import { handle, .. }
                | MemEvent::Release { handle } => handle,
            })
            .collect::<Vec<_>>(),
        [a.handle, b.handle, r.handle, i.handle]
    );
    // The hook reports what was taken from the driver, padding included.
    assert_eq!(
        hooked[0],
        MemEvent::Alloc {
            kind: MemKind::Device,
            bytes: 3 * MIB + 4096 - 256,
            handle: a.handle
        }
    );
    assert_eq!(taken[0], Call::DeviceAlloc(3 * MIB + 4096 - 256));
}

/// A context from another provider is refused, not misread.
#[test]
fn a_foreign_context_is_refused() {
    let (p, rec, _, _ctx) = setup();
    let foreign = DeviceCtx::new(7u32);
    let e = p
        .alloc(&foreign, req(MemKind::Device, MIB, 256))
        .unwrap_err();
    assert_eq!(e.class, ErrorClass::Internal);
    assert!(p.register_host(&foreign, region(ARENA, MIB)).is_err());
    assert!(p.memory_stats(&foreign).is_err());
    assert_eq!(rec.calls(), []);
}

// ---- On a CUDA device -----------------------------------------------------------------------

/// The driver's synchronous copy and set calls, for checking what the device sees.
struct Probe {
    ctx: PrimaryContext,
    htod: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> Status,
    dtoh: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> Status,
    memset32: unsafe extern "C" fn(CUdeviceptr, c_uint, usize) -> Status,
    info: unsafe extern "C" fn(*mut usize, *mut usize) -> Status,
}

impl Probe {
    fn new(ctx: &DeviceCtx) -> Self {
        let dev = ctx.downcast_ref::<CudaDevice<PrimaryContext>>().unwrap();
        let ctx = dev._context.share().unwrap();
        let api = Arc::clone(ctx.api());
        // SAFETY: the signatures are the CUDA 13 driver API's (`cuda.h`).
        unsafe {
            Self {
                htod: api.symbol("cuMemcpyHtoD_v2").unwrap(),
                dtoh: api.symbol("cuMemcpyDtoH_v2").unwrap(),
                memset32: api.symbol("cuMemsetD32_v2").unwrap(),
                info: api.symbol("cuMemGetInfo_v2").unwrap(),
                ctx,
            }
        }
    }

    fn write(&self, dst: u64, src: &[u8]) {
        self.ctx.make_current().unwrap();
        // SAFETY: `src` is live for the synchronous copy; `dst` is device memory of at least
        // `src.len()` bytes, which the caller allocated.
        let s = unsafe { (self.htod)(dst, src.as_ptr().cast(), src.len()) };
        self.ctx.api().check("cuMemcpyHtoD_v2", s).unwrap();
    }

    fn read(&self, src: u64, len: usize) -> Vec<u8> {
        self.ctx.make_current().unwrap();
        let mut out = vec![0u8; len];
        // SAFETY: `out` holds `len` bytes for the synchronous copy; `src` is device memory of
        // at least that many.
        let s = unsafe { (self.dtoh)(out.as_mut_ptr().cast(), src, len) };
        self.ctx.api().check("cuMemcpyDtoH_v2", s).unwrap();
        out
    }

    fn set32(&self, dst: u64, value: u32, count: usize) {
        self.ctx.make_current().unwrap();
        // SAFETY: `dst` is the device address of at least `4 * count` bytes of pinned host
        // memory, for which `cuMemsetD32_v2` is synchronous with the host.
        let s = unsafe { (self.memset32)(dst, value, count) };
        self.ctx.api().check("cuMemsetD32_v2", s).unwrap();
    }

    fn free_bytes(&self) -> u64 {
        self.ctx.make_current().unwrap();
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: writes two sizes through pointers to live locals.
        let s = unsafe { (self.info)(&raw mut free, &raw mut total) };
        self.ctx.api().check("cuMemGetInfo_v2", s).unwrap();
        free as u64
    }
}

/// Page-aligned host memory the test owns, as the tier manager's arena would be.
struct HostArena {
    ptr: *mut u8,
    layout: std::alloc::Layout,
}

impl HostArena {
    fn new(bytes: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(bytes, 2 << 20).unwrap();
        // SAFETY: the layout has a non-zero size.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null());
        Self { ptr, layout }
    }
    fn addr(&self) -> u64 {
        self.ptr as u64
    }
    fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: the allocation is `layout.size()` bytes, live while `self` is, and the test
        // touches it from one thread between synchronous driver calls.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.layout.size()) }
    }
}

impl Drop for HostArena {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this layout, freed once.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) };
    }
}

/// The arena build of RFC-0020 §4.1's fixed-slab path and §4.2's pinned host arena, through
/// this provider on a real CUDA device: one `Device` slab per VRAM purpose, the doorbell page,
/// the host arena registered in §6.1 units and an import, with the bytes checked from the
/// device's side; then a shrink by whole units, and teardown with nothing left. Runs with
/// `--ignored`; on a host without a CUDA device it returns after saying so.
#[test]
#[ignore = "needs the NVIDIA driver and a GPU"]
fn arenas_on_a_cuda_device() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let seen = events.clone();
    let p = CudaProvider::new()
        .with_alloc_hook(Arc::new(move |e: &MemEvent| seen.lock().unwrap().push(*e)));
    if p.enumerate().unwrap().is_empty() {
        eprintln!("no CUDA device: skipped");
        return;
    }
    eprintln!("build: {:?}", p.build());
    let ctx = p.open(dev0(), &opts()).unwrap();
    let probe = Probe::new(&ctx);
    let free_before = probe.free_bytes();

    // VRAM: one fixed slab per purpose (RFC-0020 §4.1, fallback row).
    let purposes = [
        ("Weights", 64 * MIB, 2 * MIB),
        ("Kv", 32 * MIB, 64 << 10),
        ("ExpertStaging", 16 * MIB, 4096),
        ("Scratch", 8 * MIB, 256),
        ("Graphs", MIB, 256),
    ];
    let mut slabs = Vec::new();
    for (i, &(name, bytes, align)) in purposes.iter().enumerate() {
        let b = p.alloc(&ctx, req(MemKind::Device, bytes, align)).unwrap();
        assert_eq!(b.addr % align, 0, "{name} at {:#x}", b.addr);
        let pattern: Vec<u8> = (0..bytes).map(|j| (j as u8) ^ (i as u8 * 37)).collect();
        probe.write(b.addr, &pattern);
        slabs.push((b, pattern));
    }
    for (i, (a, _)) in slabs.iter().enumerate() {
        for (b, _) in &slabs[i + 1..] {
            let disjoint = a.addr + a.bytes.0 <= b.addr || b.addr + b.bytes.0 <= a.addr;
            assert!(disjoint, "{a:?} overlaps {b:?}");
        }
    }
    for (b, pattern) in &slabs {
        assert_eq!(&probe.read(b.addr, pattern.len()), pattern, "{b:?}");
    }

    // The doorbell page: zeroed, and the host and the device see each other's writes.
    let bell = p.alloc(&ctx, req(MemKind::HostMapped, 4096, 4096)).unwrap();
    let host = p
        .resolve(&ctx, slice(&bell, 0, 4096))
        .unwrap()
        .host
        .unwrap();
    assert_eq!(host % 4096, 0);
    // SAFETY: `host` is the page `alloc` returned, live until `free` below.
    let flags = unsafe { std::slice::from_raw_parts_mut(host as *mut u32, 1024) };
    assert!(flags.iter().all(|&f| f == 0), "doorbell page not zeroed");
    flags[3] = 0x0D00_BE11;
    assert_eq!(probe.read(bell.addr + 12, 4), 0x0D00_BE11_u32.to_ne_bytes());
    probe.set32(bell.addr + 64, 1, 1);
    // SAFETY: as above; a volatile read sees the device's write.
    assert_eq!(unsafe { std::ptr::read_volatile(&raw const flags[16]) }, 1);

    // The pinned host arena, registered in §6.1 units of 16 MiB.
    let mut arena = HostArena::new(64 * MIB as usize);
    let unit = 16 * MIB;
    let mut units = Vec::new();
    for k in 0..4 {
        units.push(
            p.register_host(&ctx, region(arena.addr() + k * unit, unit))
                .unwrap(),
        );
    }
    let e = p
        .register_host(&ctx, region(arena.addr() + 8 * MIB, unit))
        .unwrap_err();
    assert_eq!(e.class, ErrorClass::Internal, "{e}");
    assert_eq!(
        p.registration_for(&ctx, arena.addr() + unit, unit),
        Some(units[1].handle)
    );
    assert_eq!(
        p.registration_for(&ctx, arena.addr() + unit - 4096, 8192),
        None
    );
    // A DMA source inside one unit copies to the device.
    let src = &mut arena.bytes()[(unit + 4096) as usize..(unit + 4096 + MIB) as usize];
    src.iter_mut()
        .enumerate()
        .for_each(|(j, b)| *b = (j % 251) as u8);
    let staging = &slabs[2].0;
    probe.write(staging.addr, src);
    assert_eq!(probe.read(staging.addr, MIB as usize), src);

    // Shrink by whole units, then grow back.
    let last = units.pop().unwrap();
    let third = units.pop().unwrap();
    p.unregister_host(&ctx, last).unwrap();
    p.unregister_host(&ctx, third).unwrap();
    units.push(
        p.register_host(&ctx, region(arena.addr() + 2 * unit, 2 * unit))
            .unwrap(),
    );

    // An import: the device writes the tier manager's memory in place.
    let mut tier = HostArena::new(8 * MIB as usize);
    let imported = p.import(&ctx, region(tier.addr(), 8 * MIB)).unwrap();
    probe.set32(imported.addr + MIB, 0xABCD_EF01, 4);
    assert_eq!(
        &tier.bytes()[MIB as usize..MIB as usize + 16],
        0xABCD_EF01u32.to_ne_bytes().repeat(4).as_slice()
    );

    // Errors from the driver.
    let e = p
        .alloc(&ctx, req(MemKind::Device, 1 << 50, 256))
        .unwrap_err();
    assert_eq!(
        (e.code, e.class),
        (codes::ALLOC_FAILED, ErrorClass::Capacity)
    );
    eprintln!("1 PiB: {e}");
    let stats = p.memory_stats(&ctx).unwrap();
    eprintln!("{stats:?}; device free {free_before} before");
    assert_eq!(stats.device_buffers, 5);
    assert_eq!(stats.registrations, 3);
    assert_eq!(stats.host_buffers, 2);

    // Teardown: free everything; then a device closed with live memory frees it.
    for (b, _) in slabs {
        p.free(&ctx, b);
    }
    p.free(&ctx, bell);
    p.free(&ctx, imported);
    for u in units {
        p.unregister_host(&ctx, u).unwrap();
    }
    assert_eq!(p.memory_stats(&ctx).unwrap(), MemStats::default());
    let free_after = probe.free_bytes();
    eprintln!("device free {free_before} before, {free_after} after");
    p.alloc(&ctx, req(MemKind::Device, 32 * MIB, 256)).unwrap();
    p.register_host(&ctx, region(arena.addr(), unit)).unwrap();
    drop(probe);
    drop(ctx);
    let events = events.lock().unwrap();
    let released = events
        .iter()
        .filter(|e| matches!(e, MemEvent::Release { .. }))
        .count();
    assert_eq!(released * 2, events.len(), "{events:?}");
    // The arenas outlive the device, as the tier manager's do: dropped last.
    drop(tier);
    drop(arena);
}
