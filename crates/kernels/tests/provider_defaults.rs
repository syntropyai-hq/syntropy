// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `KernelProvider` trait's default bodies, over a provider that implements only the
//! required methods. It is an integration test so that its `unsafe fn` implementations sit
//! outside `kernels`' sources, where RFC-0001 §4.2 rule 8 allows `unsafe` only on the trait's
//! method declarations.

use std::cell::RefCell;

use syntropy_core::resource::Backend;
use syntropy_core::{Bytes, DeviceId, ErrorClass, HostRegion};

use syntropy_kernels::*;

/// A provider that implements only the required methods, to exercise the default bodies.
/// `copy` records each source offset and fails on offset 2.
struct Minimal {
    ops: OpRegistry,
}

thread_local! {
    static COPIES: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

fn not_needed<T>() -> Result<T, Error> {
    Err(Error {
        class: ErrorClass::Internal,
        code: ErrorCode(0),
        message: String::from("not used by this test"),
    })
}

impl KernelProvider for Minimal {
    fn id(&self) -> ProviderId {
        ProviderId("minimal")
    }
    fn backend(&self) -> Backend {
        Backend::Cpu
    }
    fn build(&self) -> ProviderBuild {
        ProviderBuild {
            version: 1,
            module: None,
            toolchain: Default::default(),
            driver: None,
        }
    }
    fn domain(&self) -> MemoryDomain {
        MemoryDomain::Host
    }
    fn enumerate(&self) -> Result<Vec<ProviderDevice>, Error> {
        Ok(Vec::new())
    }
    fn open(&self, _dev: DeviceId, _opts: &OpenOptions) -> Result<DeviceCtx, Error> {
        Ok(DeviceCtx::new(()))
    }
    fn ops(&self) -> &OpRegistry {
        &self.ops
    }
    fn query(&self, _: &DeviceCtx, _: &OpKey, _: &ShapeSig) -> Option<OpCaps> {
        None
    }
    fn alloc(&self, _: &DeviceCtx, _: AllocRequest) -> Result<DeviceBuffer, Error> {
        not_needed()
    }
    fn free(&self, _: &DeviceCtx, _: DeviceBuffer) {}
    unsafe fn register_host(&self, _: &DeviceCtx, _: HostRegion) -> Result<Registration, Error> {
        not_needed()
    }
    unsafe fn import(&self, _: &DeviceCtx, _: HostRegion) -> Result<DeviceBuffer, Error> {
        not_needed()
    }
    fn repack(&self, _: &OpKey, _: &[u8], _: &mut [u8]) -> Option<Result<LayoutVersion, Error>> {
        None
    }
    fn repacked_len(&self, _: &OpKey, _: Bytes) -> Option<Bytes> {
        None
    }
    fn stream(&self, _: &DeviceCtx, _: StreamPriority) -> Result<Stream, Error> {
        Ok(Stream { handle: 1 })
    }
    fn event(&self, _: &DeviceCtx) -> Result<Event, Error> {
        Ok(Event { handle: 1 })
    }
    fn record(&self, _: &DeviceCtx, _: &Stream, _: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn wait(&self, _: &DeviceCtx, _: &Stream, _: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn signal_host(&self, _: &DeviceCtx, _: &Stream, _: HostFlag, _: u32) -> Result<(), Error> {
        Ok(())
    }
    fn poll(&self, _: &DeviceCtx, _: &Event) -> Result<bool, Error> {
        Ok(true)
    }
    fn synchronize(&self, _: &DeviceCtx, _: &Stream) -> Result<(), Error> {
        Ok(())
    }
    fn copy(
        &self,
        _: &DeviceCtx,
        _: &Stream,
        src: BufferSlice,
        _: BufferSlice,
    ) -> Result<(), Error> {
        COPIES.with(|c| c.borrow_mut().push(src.offset.0));
        if src.offset.0 == 2 {
            not_needed()
        } else {
            Ok(())
        }
    }
    fn prepare(&self, _: &DeviceCtx, _: &OpKey, _: &ShapeSig) -> Result<PreparedOp, Error> {
        not_needed()
    }
    fn launch(
        &self,
        _: &DeviceCtx,
        _: &Stream,
        _: &PreparedOp,
        _: &OpArgs<'_>,
    ) -> Result<(), Error> {
        not_needed()
    }
    fn begin_capture(&self, _: &DeviceCtx, _: &Stream) -> Result<(), Error> {
        Err(unsupported())
    }
    fn end_capture(&self, _: &DeviceCtx, _: &Stream) -> Result<CapturedGraph, Error> {
        Err(unsupported())
    }
    fn replay(&self, _: &DeviceCtx, _: &Stream, _: &CapturedGraph) -> Result<(), Error> {
        Err(unsupported())
    }
    fn health(&self, _: &DeviceCtx) -> DeviceHealth {
        DeviceHealth::Ok
    }
}

fn setup() -> (Minimal, DeviceCtx, Stream) {
    let p = Minimal {
        ops: OpRegistry::new(),
    };
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
    let dev = DeviceId {
        node: syntropy_core::NodeId([0; 32]),
        ordinal: 0,
    };
    let ctx = p.open(dev, &opts).expect("opens");
    let s = p.stream(&ctx, StreamPriority::Compute).expect("stream");
    (p, ctx, s)
}

fn slice(offset: u64) -> BufferSlice {
    BufferSlice {
        buffer: 1,
        offset: Bytes(offset),
        bytes: Bytes(4),
    }
}

/// Each default body returns what RFC-0040 §2.1 specifies.
#[test]
fn default_bodies() {
    let (p, ctx, s) = setup();
    let reservation = DeviceBuffer {
        addr: 0,
        bytes: Bytes(1 << 20),
        kind: MemKind::Device,
        handle: 9,
    };
    let seg = Segment {
        reservation: 9,
        offset: Bytes(0),
        bytes: Bytes(1 << 20),
        handle: 10,
    };
    let flag = HostFlag { page: 0, slot: 3 };
    let e = p.event(&ctx).expect("event");

    assert_eq!(p.memory_info(&ctx), Err(unsupported()));
    assert_eq!(
        p.map_segment(&ctx, &reservation, Bytes(0), Bytes(1 << 20)),
        Err(unsupported())
    );
    assert_eq!(p.unmap_segment(&ctx, seg), Err(unsupported()));
    assert!(p.host_registrar(&ctx).is_none());
    assert_eq!(p.set_resident(&ctx, &[&reservation], true), Ok(()));
    assert_eq!(
        p.wait_host(&ctx, &s, flag, 1, WaitCmp::Eq, Some(0)),
        Err(unsupported())
    );
    let waker = HostWaker {
        kind: crate::stream::WakeKind::Futex,
        handle: 0x1000,
    };
    assert_eq!(p.arm_wake(&ctx, &s, &waker), Err(unsupported()));
    assert_eq!(p.elapsed(&ctx, &e, &e), None);
    assert!(p.collectives().is_none());

    let err = unsupported();
    assert_eq!(err.class, ErrorClass::Unsupported);
    assert_eq!(err.code, ErrorCode(1));
}

/// The default `copy_batch` issues one `copy` per pair, in order, and stops at the first
/// error.
#[test]
fn copy_batch_default_loops_in_order() {
    let (p, ctx, s) = setup();
    COPIES.with(|c| c.borrow_mut().clear());
    let ok = [(slice(0), slice(8)), (slice(1), slice(9))];
    assert_eq!(p.copy_batch(&ctx, &s, &ok), Ok(()));
    COPIES.with(|c| assert_eq!(*c.borrow(), vec![0, 1]));

    COPIES.with(|c| c.borrow_mut().clear());
    let failing = [
        (slice(0), slice(8)),
        (slice(2), slice(9)),
        (slice(3), slice(10)),
    ];
    assert!(p.copy_batch(&ctx, &s, &failing).is_err());
    COPIES.with(|c| assert_eq!(*c.borrow(), vec![0, 2]));
}
