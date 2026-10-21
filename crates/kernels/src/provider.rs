// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `KernelProvider` trait (RFC-0001 §5.7, RFC-0040 §2.1; ADR-0002 item 2).

use syntropy_core::resource::Backend;
use syntropy_core::{Bytes, DeviceId, Nanos};

use crate::device::{
    DeviceCtx, DeviceHealth, DeviceMemoryInfo, MemoryDomain, OpenOptions, ProviderBuild,
    ProviderDevice, ProviderId,
};
use crate::error::{Error, unsupported};
use crate::memory::{
    AllocRequest, BufferSlice, DeviceBuffer, HostRegion, HostRegistrar, Registration, Segment,
};
use crate::op::{LayoutVersion, OpArgs, OpCaps, OpKey, OpRegistry, PreparedOp, ShapeSig};
use crate::stream::{CapturedGraph, Event, HostFlag, HostWaker, Stream, StreamPriority, WaitCmp};

/// A source of kernels for one backend (ADR-0002). Native providers cover the hot ops on
/// Tier 1 backends; the ggml provider covers every op ggml supports and is the correctness
/// reference for native kernels. Owner: RFC-0040.
///
/// RFC-0001 §5.7 fixes the original methods. RFC-0040 §2.1 adds `build`, `domain`,
/// `memory_info`, `map_segment`, `unmap_segment`, `host_registrar`, `set_resident`,
/// `wait_host`, `arm_wake`, `poll`, `elapsed`, `synchronize` and `copy_batch`. Every
/// addition except `build`, `domain`, `poll` and `synchronize` has a default body. (RFC-0001
/// §5.7's comment says all but `poll` and `synchronize`; RFC-0040 §2.1, which §5.7 defers
/// to, also leaves `build` and `domain` without one. Issue #1525, item 1, tracks the fix.)
///
/// A provider that implements every required method compiles:
///
/// ```
/// # use syntropy_core::{Bytes, DeviceId, resource::Backend};
/// # use syntropy_kernels::*;
/// # macro_rules! provider { ($($extra:item)*) => {
/// # struct P;
/// # impl KernelProvider for P {
/// #     fn id(&self) -> ProviderId { todo!() }
/// #     fn backend(&self) -> Backend { todo!() }
/// #     fn build(&self) -> ProviderBuild { todo!() }
/// #     fn domain(&self) -> MemoryDomain { todo!() }
/// #     fn enumerate(&self) -> Result<Vec<ProviderDevice>, Error> { todo!() }
/// #     fn open(&self, _: DeviceId, _: &OpenOptions) -> Result<DeviceCtx, Error> { todo!() }
/// #     fn ops(&self) -> &OpRegistry { todo!() }
/// #     fn query(&self, _: &DeviceCtx, _: &OpKey, _: &ShapeSig) -> Option<OpCaps> { todo!() }
/// #     fn alloc(&self, _: &DeviceCtx, _: AllocRequest) -> Result<DeviceBuffer, Error> { todo!() }
/// #     fn free(&self, _: &DeviceCtx, _: DeviceBuffer) {}
/// #     fn register_host(&self, _: &DeviceCtx, _: HostRegion) -> Result<Registration, Error> { todo!() }
/// #     fn import(&self, _: &DeviceCtx, _: HostRegion) -> Result<DeviceBuffer, Error> { todo!() }
/// #     fn repack(&self, _: &OpKey, _: &[u8], _: &mut [u8]) -> Option<Result<LayoutVersion, Error>> { None }
/// #     fn repacked_len(&self, _: &OpKey, _: Bytes) -> Option<Bytes> { None }
/// #     fn stream(&self, _: &DeviceCtx, _: StreamPriority) -> Result<Stream, Error> { todo!() }
/// #     fn event(&self, _: &DeviceCtx) -> Result<Event, Error> { todo!() }
/// #     fn record(&self, _: &DeviceCtx, _: &Stream, _: &Event) -> Result<(), Error> { todo!() }
/// #     fn wait(&self, _: &DeviceCtx, _: &Stream, _: &Event) -> Result<(), Error> { todo!() }
/// #     fn signal_host(&self, _: &DeviceCtx, _: &Stream, _: HostFlag, _: u32) -> Result<(), Error> { todo!() }
/// #     fn synchronize(&self, _: &DeviceCtx, _: &Stream) -> Result<(), Error> { todo!() }
/// #     fn copy(&self, _: &DeviceCtx, _: &Stream, _: BufferSlice, _: BufferSlice) -> Result<(), Error> { todo!() }
/// #     fn prepare(&self, _: &DeviceCtx, _: &OpKey, _: &ShapeSig) -> Result<PreparedOp, Error> { todo!() }
/// #     fn launch(&self, _: &DeviceCtx, _: &Stream, _: &PreparedOp, _: &OpArgs<'_>) -> Result<(), Error> { todo!() }
/// #     fn begin_capture(&self, _: &DeviceCtx, _: &Stream) -> Result<(), Error> { todo!() }
/// #     fn end_capture(&self, _: &DeviceCtx, _: &Stream) -> Result<CapturedGraph, Error> { todo!() }
/// #     fn replay(&self, _: &DeviceCtx, _: &Stream, _: &CapturedGraph) -> Result<(), Error> { todo!() }
/// #     fn health(&self, _: &DeviceCtx) -> DeviceHealth { todo!() }
/// #     $($extra)*
/// # }
/// # } }
/// provider! {
///     fn poll(&self, _: &DeviceCtx, _: &Event) -> Result<bool, Error> { Ok(true) }
/// }
/// ```
///
/// The same provider without `poll`, which has no default body, does not:
///
/// ```compile_fail
/// # use syntropy_core::{Bytes, DeviceId, resource::Backend};
/// # use syntropy_kernels::*;
/// # macro_rules! provider { ($($extra:item)*) => {
/// # struct P;
/// # impl KernelProvider for P {
/// #     fn id(&self) -> ProviderId { todo!() }
/// #     fn backend(&self) -> Backend { todo!() }
/// #     fn build(&self) -> ProviderBuild { todo!() }
/// #     fn domain(&self) -> MemoryDomain { todo!() }
/// #     fn enumerate(&self) -> Result<Vec<ProviderDevice>, Error> { todo!() }
/// #     fn open(&self, _: DeviceId, _: &OpenOptions) -> Result<DeviceCtx, Error> { todo!() }
/// #     fn ops(&self) -> &OpRegistry { todo!() }
/// #     fn query(&self, _: &DeviceCtx, _: &OpKey, _: &ShapeSig) -> Option<OpCaps> { todo!() }
/// #     fn alloc(&self, _: &DeviceCtx, _: AllocRequest) -> Result<DeviceBuffer, Error> { todo!() }
/// #     fn free(&self, _: &DeviceCtx, _: DeviceBuffer) {}
/// #     fn register_host(&self, _: &DeviceCtx, _: HostRegion) -> Result<Registration, Error> { todo!() }
/// #     fn import(&self, _: &DeviceCtx, _: HostRegion) -> Result<DeviceBuffer, Error> { todo!() }
/// #     fn repack(&self, _: &OpKey, _: &[u8], _: &mut [u8]) -> Option<Result<LayoutVersion, Error>> { None }
/// #     fn repacked_len(&self, _: &OpKey, _: Bytes) -> Option<Bytes> { None }
/// #     fn stream(&self, _: &DeviceCtx, _: StreamPriority) -> Result<Stream, Error> { todo!() }
/// #     fn event(&self, _: &DeviceCtx) -> Result<Event, Error> { todo!() }
/// #     fn record(&self, _: &DeviceCtx, _: &Stream, _: &Event) -> Result<(), Error> { todo!() }
/// #     fn wait(&self, _: &DeviceCtx, _: &Stream, _: &Event) -> Result<(), Error> { todo!() }
/// #     fn signal_host(&self, _: &DeviceCtx, _: &Stream, _: HostFlag, _: u32) -> Result<(), Error> { todo!() }
/// #     fn synchronize(&self, _: &DeviceCtx, _: &Stream) -> Result<(), Error> { todo!() }
/// #     fn copy(&self, _: &DeviceCtx, _: &Stream, _: BufferSlice, _: BufferSlice) -> Result<(), Error> { todo!() }
/// #     fn prepare(&self, _: &DeviceCtx, _: &OpKey, _: &ShapeSig) -> Result<PreparedOp, Error> { todo!() }
/// #     fn launch(&self, _: &DeviceCtx, _: &Stream, _: &PreparedOp, _: &OpArgs<'_>) -> Result<(), Error> { todo!() }
/// #     fn begin_capture(&self, _: &DeviceCtx, _: &Stream) -> Result<(), Error> { todo!() }
/// #     fn end_capture(&self, _: &DeviceCtx, _: &Stream) -> Result<CapturedGraph, Error> { todo!() }
/// #     fn replay(&self, _: &DeviceCtx, _: &Stream, _: &CapturedGraph) -> Result<(), Error> { todo!() }
/// #     fn health(&self, _: &DeviceCtx) -> DeviceHealth { todo!() }
/// #     $($extra)*
/// # }
/// # } }
/// provider! {}
/// ```
///
/// **Threading (RFC-0040 §2.4, ADR-0012).** `enumerate`, `ops`, `repack`, `repacked_len`,
/// `build` and `domain` may run on any thread. Every other method takes `&DeviceCtx`, which
/// is neither `Send` nor `Sync`, so it runs only on the device's dispatcher thread (or a CPU
/// pool thread for the `cpu` provider). After `Warming`, a step may call only `replay`,
/// `launch` (on providers without capture), `record`, `wait`, `poll`, `signal_host`,
/// `wait_host`, `copy`, `copy_batch` and `health`; each must be allocation-free and
/// lock-free and make no synchronising driver call.
pub trait KernelProvider: Send + Sync {
    /// The provider's name, e.g. `"cuda"`, `"metal"`, `"cpu"`, `"ggml-vulkan"`.
    fn id(&self) -> ProviderId;
    /// The provider family.
    fn backend(&self) -> Backend;
    /// Build identity: provider version, kernel-module BLAKE3, compiler versions, and the
    /// driver or runtime version seen at load. Part of every selection-cache key
    /// (RFC-0040 §6.5).
    fn build(&self) -> ProviderBuild;
    /// The memory domain the provider belongs to (RFC-0040 §2.3). Selection never mixes
    /// domains on a device.
    fn domain(&self) -> MemoryDomain;

    /// Devices this provider can drive on this host, with their class and features.
    fn enumerate(&self) -> Result<Vec<ProviderDevice>, Error>;
    /// Creates the device context. Called once per device, on its dispatcher thread.
    fn open(&self, dev: DeviceId, opts: &OpenOptions) -> Result<DeviceCtx, Error>;
    /// Free and budgeted memory now (DXGI budget, `VK_EXT_memory_budget`, `cuMemGetInfo`,
    /// `recommendedMaxWorkingSetSize`). Never called on a hot path.
    fn memory_info(&self, _ctx: &DeviceCtx) -> Result<DeviceMemoryInfo, Error> {
        Err(unsupported())
    }

    /// Every op this provider implements: op kind, accepted quant schemes and dtypes,
    /// layouts, shape constraints.
    fn ops(&self) -> &OpRegistry;
    /// Whether `op` runs on this device at these shapes, its workspace size, and the
    /// provider's own cost estimate. The executor still microbenchmarks before choosing.
    fn query(&self, ctx: &DeviceCtx, op: &OpKey, shape: &ShapeSig) -> Option<OpCaps>;

    /// Allocates device, host-mapped or unified memory. Called only by the tier manager
    /// (RFC-0001 §9.1); the executor never allocates. `reserve_only` reserves address space
    /// to be backed later with [`map_segment`](Self::map_segment).
    fn alloc(&self, ctx: &DeviceCtx, req: AllocRequest) -> Result<DeviceBuffer, Error>;
    /// Releases a buffer from [`alloc`](Self::alloc) or [`import`](Self::import).
    fn free(&self, ctx: &DeviceCtx, buf: DeviceBuffer);
    /// Backs `bytes` at `offset` of a `reserve_only` allocation with a physical segment.
    /// Addresses inside the reservation never move (elastic expert cache, RFC-0001 §9.3;
    /// RFC-0020 §4.1). May synchronise, so it runs only between steps.
    fn map_segment(
        &self,
        _ctx: &DeviceCtx,
        _reservation: &DeviceBuffer,
        _offset: Bytes,
        _bytes: Bytes,
    ) -> Result<Segment, Error> {
        Err(unsupported())
    }
    /// Releases a segment from [`map_segment`](Self::map_segment). May synchronise, so it
    /// runs only between steps.
    fn unmap_segment(&self, _ctx: &DeviceCtx, _seg: Segment) -> Result<(), Error> {
        Err(unsupported())
    }
    /// Makes a region of the tier manager's pinned arena DMA-visible to this device.
    fn register_host(&self, ctx: &DeviceCtx, region: HostRegion) -> Result<Registration, Error>;
    /// A `Send` capability that registers and unregisters host ranges from the
    /// housekeeping thread, so progressive pinning (RFC-0020 §6) never blocks the
    /// dispatcher. `None` means the provider registers on its dispatcher thread during
    /// `Loading`.
    fn host_registrar(&self, _ctx: &DeviceCtx) -> Option<Box<dyn HostRegistrar>> {
        None
    }
    /// Wraps memory the tier manager owns, without copying (unified tiers, no-copy buffers).
    fn import(&self, ctx: &DeviceCtx, region: HostRegion) -> Result<DeviceBuffer, Error>;
    /// Residency hint for backends with explicit residency (Metal residency sets, Vulkan
    /// priority). The default does nothing: on CUDA and HIP every mapped allocation is
    /// resident. May synchronise, so it runs only between steps.
    fn set_resident(
        &self,
        _ctx: &DeviceCtx,
        _bufs: &[&DeviceBuffer],
        _resident: bool,
    ) -> Result<(), Error> {
        Ok(())
    }
    /// Repacks source bytes into this provider's layout for `op`. `None` means the source
    /// layout is used as is. Runs during `Packing`.
    fn repack(
        &self,
        op: &OpKey,
        src: &[u8],
        dst: &mut [u8],
    ) -> Option<Result<LayoutVersion, Error>>;
    /// The size of `op`'s repacked form of `src` bytes; `None` when `repack` keeps the
    /// source layout.
    fn repacked_len(&self, op: &OpKey, src: Bytes) -> Option<Bytes>;

    /// Creates a stream of the given priority.
    fn stream(&self, ctx: &DeviceCtx, prio: StreamPriority) -> Result<Stream, Error>;
    /// Creates an event.
    fn event(&self, ctx: &DeviceCtx) -> Result<Event, Error>;
    /// Records `e` at the current end of `s`.
    fn record(&self, ctx: &DeviceCtx, s: &Stream, e: &Event) -> Result<(), Error>;
    /// Device-side wait: `s` waits for `e` without a host round trip.
    fn wait(&self, ctx: &DeviceCtx, s: &Stream, e: &Event) -> Result<(), Error>;
    /// The device writes `value` to the host-mapped `flag` when `s` reaches this point: the
    /// doorbell the host spins on instead of a synchronising driver call (RFC-0001 §8.2).
    /// On host-mapped flags, under RFC-0040 §4.2's handshake, the producer only ever writes
    /// 1. Metal shared events differ: they carry per-step values and are never reset.
    fn signal_host(
        &self,
        ctx: &DeviceCtx,
        s: &Stream,
        flag: HostFlag,
        value: u32,
    ) -> Result<(), Error>;
    /// Device-side wait until the host-mapped `flag` satisfies `cmp` against `value`: the
    /// other half of the doorbell (RFC-0001 §8.2).
    ///
    /// `reset`, when set, is written to the flag by the device right after the wait passes.
    /// It is the consumer reset of RFC-0040 §4.2's handshake: the consumer waits for 1, then
    /// writes 0, and only after that produces anything that can lead to the flag's next set.
    /// In captured regions `value` and `reset` are constants.
    ///
    /// `Unsupported`, the default, means the executor splits the step at this point into two
    /// submissions (RFC-0040 §4.2).
    fn wait_host(
        &self,
        _ctx: &DeviceCtx,
        _s: &Stream,
        _flag: HostFlag,
        _value: u32,
        _cmp: WaitCmp,
        _reset: Option<u32>,
    ) -> Result<(), Error> {
        Err(unsupported())
    }
    /// Device-triggered wake for a host thread parked inside a step (RFC-0012 §3.3): after
    /// the work enqueued so far on `s` (normally a doorbell write), signal `waker` without
    /// the stream waiting for the signal. Captured like any other node. `Unsupported`, the
    /// default, means the waiter spins (RFC-0040 §4.2).
    fn arm_wake(&self, _ctx: &DeviceCtx, _s: &Stream, _waker: &HostWaker) -> Result<(), Error> {
        Err(unsupported())
    }
    /// Non-blocking completion poll. Dispatchers use it instead of synchronising.
    fn poll(&self, ctx: &DeviceCtx, e: &Event) -> Result<bool, Error>;
    /// Device time between two completed events, for microbenchmarks and telemetry.
    fn elapsed(&self, _ctx: &DeviceCtx, _start: &Event, _end: &Event) -> Option<Nanos> {
        None
    }
    /// Blocking drain of a stream. Allowed only outside `Ready`.
    fn synchronize(&self, ctx: &DeviceCtx, s: &Stream) -> Result<(), Error>;
    /// Copies `src` to `dst` in stream order.
    fn copy(
        &self,
        ctx: &DeviceCtx,
        s: &Stream,
        src: BufferSlice,
        dst: BufferSlice,
    ) -> Result<(), Error>;
    /// Many copies in one submission (`cudaMemcpyBatchAsync` on CUDA 13). The default issues
    /// one [`copy`](Self::copy) per pair, in order, and stops at the first error.
    fn copy_batch(
        &self,
        ctx: &DeviceCtx,
        s: &Stream,
        pairs: &[(BufferSlice, BufferSlice)],
    ) -> Result<(), Error> {
        for (src, dst) in pairs {
            self.copy(ctx, s, *src, *dst)?;
        }
        Ok(())
    }

    /// Makes `op` ready to launch at `shape` on this device.
    fn prepare(&self, ctx: &DeviceCtx, op: &OpKey, shape: &ShapeSig) -> Result<PreparedOp, Error>;
    /// Enqueues a prepared op on `s`.
    fn launch(
        &self,
        ctx: &DeviceCtx,
        s: &Stream,
        op: &PreparedOp,
        args: &OpArgs<'_>,
    ) -> Result<(), Error>;
    /// Starts recording `s`'s launches into a replayable graph (CUDA graphs, Metal indirect
    /// command buffers). Providers without capture return `Unsupported`, and the executor
    /// replays the launches itself.
    fn begin_capture(&self, ctx: &DeviceCtx, s: &Stream) -> Result<(), Error>;
    /// Ends the recording [`begin_capture`](Self::begin_capture) started.
    fn end_capture(&self, ctx: &DeviceCtx, s: &Stream) -> Result<CapturedGraph, Error>;
    /// Enqueues a captured graph on `s`.
    fn replay(&self, ctx: &DeviceCtx, s: &Stream, g: &CapturedGraph) -> Result<(), Error>;

    /// Intra-node collectives (all-reduce and all-gather over NVLink or PCIe P2P) for
    /// `IntraNodeTp`. Cross-node collectives are built by RFC-0032 from `Transport` plus
    /// `copy`.
    fn collectives(&self) -> Option<&dyn CollectiveOps> {
        None
    }
    /// Checks for sticky device errors. A sticky error fails the runner with class `Device`.
    fn health(&self, ctx: &DeviceCtx) -> DeviceHealth;
}

/// Intra-node collectives for `IntraNodeTp` (RFC-0040 §7.6), on CUDA and HIP.
///
/// RFC-0040 §7.6 gives its methods (`join`, `all_reduce`, `all_gather`) and their
/// rendezvous and group types; COMP-E4-T09 adds them with the first implementation. Until
/// then the trait is empty, and no provider returns one from
/// [`KernelProvider::collectives`].
pub trait CollectiveOps {}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use syntropy_core::ErrorClass;

    use super::*;
    use crate::device::{PowerHint, StreamPlan};
    use crate::error::ErrorCode;
    use crate::memory::MemKind;

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
        fn register_host(&self, _: &DeviceCtx, _: HostRegion) -> Result<Registration, Error> {
            not_needed()
        }
        fn import(&self, _: &DeviceCtx, _: HostRegion) -> Result<DeviceBuffer, Error> {
            not_needed()
        }
        fn repack(
            &self,
            _: &OpKey,
            _: &[u8],
            _: &mut [u8],
        ) -> Option<Result<LayoutVersion, Error>> {
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
        let p = Minimal { ops: OpRegistry {} };
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
}
