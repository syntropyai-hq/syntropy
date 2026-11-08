// SPDX-License-Identifier: MIT OR Apache-2.0

//! A test provider library: a `cdylib` built with `kernels::abi::export!`, which the loader's
//! boundary tests open (COMP-E9-T04). `build.rs` links it with the single-export version script
//! (Linux), the `.def` file (Windows) or the exported-symbols list (macOS).
//!
//! The provider is a small in-memory one. Its behaviour is steered through the core's `config`
//! callback, which the thunks reach on the caller's thread:
//!
//! - `fixture.panic_in`: the name of a `KernelProvider` method that panics when called;
//! - `fixture.devices`: how many devices `enumerate` lists (default 1);
//! - `fixture.long_error`: `alloc` fails with a message of that many bytes.
//!
//! A stream whose handle is [`ECHO`] makes the calls that carry structured arguments (`launch`,
//! `copy_batch`, `wait_host`, `signal_host`, `arm_wake`) fail with the arguments they received,
//! as text, so a test can compare them with what it sent.

#![allow(unsafe_code)]
#![allow(
    clippy::result_large_err,
    reason = "core::Error is RFC-0001 §5.12's struct, 344 B inline [measured: size_of, debug, x86-64 Linux, ce2044ae] (COMP-E13-T15)"
)]

use std::sync::atomic::{AtomicU64, Ordering};

use smallvec::smallvec;
use syntropy_core::resource::{Backend, CpuIsa, DeviceClass, DeviceFeatures, DeviceKind};
use syntropy_core::{Blake3, Bytes, DType, DeviceId, ErrorClass, Nanos, Numerics, QuantScheme};
use syntropy_kernels::abi::guest::HostApi;
use syntropy_kernels::abi::guest::Level;
use syntropy_kernels::{
    AllocRequest, BufferSlice, CapturedGraph, DeviceBuffer, DeviceCtx, DeviceHealth,
    DeviceMemoryInfo, Error, ErrorCode, Event, HostFlag, HostRegFlags, HostRegion, HostRegistrar,
    HostWaker, KernelProvider, LayoutVersion, MemKind, MemoryDomain, OpArgs, OpCaps, OpKey, OpKind,
    OpRegistry, OpenOptions, PreparedOp, ProviderBuild, ProviderDevice, ProviderId, Registration,
    Segment, ShapeSig, Stream, StreamPriority, WaitCmp,
};

/// The build ID the test provider states.
const BUILD_ID: [u8; 32] = [7; 32];

/// The stream handle that makes structured-argument calls echo what they received.
const ECHO: u64 = 0xEC40;

/// Metric IDs the provider sends when its release work runs, so the tests can tell that
/// `free`, `close_ctx` and a registrar's destructor ran, sticky provider or not.
const FREED: u32 = 101;
const CONTEXT_CLOSED: u32 = 102;
const REGISTRAR_CLOSED: u32 = 103;

struct TestProvider {
    host: HostApi,
    ops: OpRegistry,
    next: AtomicU64,
}

/// What `open` keeps for a device. Dropping it is `close_ctx`'s provider-side work.
struct Opened {
    device: DeviceId,
    host: HostApi,
}

impl Drop for Opened {
    fn drop(&mut self) {
        if self.host.config("fixture.panic_in").as_deref() == Some("close_ctx") {
            panic!("injected panic in close_ctx");
        }
        self.host.metric(CONTEXT_CLOSED, 1);
    }
}

fn op_key(op: OpKind, layout: u16) -> OpKey {
    OpKey {
        op,
        schemes: smallvec![
            QuantScheme::Plain(DType::F32),
            QuantScheme::Plain(DType::F32),
            QuantScheme::Plain(DType::F32)
        ],
        layout: LayoutVersion(layout),
        variant: 0,
    }
}

impl TestProvider {
    fn new(host: &HostApi) -> Self {
        let mut ops = OpRegistry::new();
        ops.register(op_key(OpKind::Gemv, 0))
            .expect("a fresh registry");
        ops.register(op_key(OpKind::Gemm, 1))
            .expect("a fresh registry");
        Self {
            host: *host,
            ops,
            next: AtomicU64::new(1),
        }
    }

    /// Panics if the core's config names `method`.
    fn hook(&self, method: &str) {
        if self.host.config("fixture.panic_in").as_deref() == Some(method) {
            panic!("injected panic in {method}");
        }
    }

    fn handle(&self) -> u64 {
        self.next.fetch_add(1, Ordering::Relaxed)
    }
}

fn echo(what: &str, text: String) -> Error {
    Error::new(
        ErrorClass::InvalidRequest,
        ErrorCode::new(syntropy_core::Domain::Kernels, 0x0E),
        format!("{what}: {text}"),
    )
}

struct Registrar {
    host: HostApi,
}

impl Drop for Registrar {
    fn drop(&mut self) {
        if self.host.config("fixture.panic_in").as_deref() == Some("registrar_close") {
            panic!("injected panic in registrar_close");
        }
        self.host.metric(REGISTRAR_CLOSED, 1);
    }
}

impl HostRegistrar for Registrar {
    fn register(&self, region: HostRegion, flags: HostRegFlags) -> Result<Registration, Error> {
        if self.host.config("fixture.panic_in").as_deref() == Some("registrar_register") {
            panic!("injected panic in registrar_register");
        }
        Ok(Registration {
            region,
            handle: 1000 + u64::from(flags.bits()),
        })
    }

    fn unregister(&self, reg: Registration) -> Result<(), Error> {
        if self.host.config("fixture.panic_in").as_deref() == Some("registrar_unregister") {
            panic!("injected panic in registrar_unregister");
        }
        if reg.handle < 1000 {
            return Err(echo("unregister", format!("{reg:?}")));
        }
        Ok(())
    }
}

impl KernelProvider for TestProvider {
    fn id(&self) -> ProviderId {
        ProviderId::new("test-host")
    }

    fn backend(&self) -> Backend {
        Backend::Cpu
    }

    fn build(&self) -> ProviderBuild {
        self.hook("build");
        ProviderBuild {
            version: 1,
            module: Some(Blake3(BUILD_ID)),
            toolchain: smallvec![("rustc".into(), "test".to_owned())],
            driver: None,
        }
    }

    fn domain(&self) -> MemoryDomain {
        MemoryDomain::Host
    }

    fn enumerate(&self) -> Result<Vec<ProviderDevice>, Error> {
        self.hook("enumerate");
        let count: u32 = self
            .host
            .config("fixture.devices")
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        Ok((0..count)
            .map(|local| ProviderDevice {
                local,
                uuid: Some([local as u8; 16]),
                pci: None,
                kind: DeviceKind::Cpu,
                name: format!("test device {local}"),
                class: DeviceClass::Cpu { isa: CpuIsa::X86V3 },
                features: DeviceFeatures::empty(),
                memory: DeviceMemoryInfo {
                    total: Bytes(1 << 30),
                    free: Bytes(1 << 29),
                    budget: None,
                    host_visible_device_local: None,
                    max_alloc: Bytes(1 << 28),
                    unified: true,
                },
                numa: Some(0),
            })
            .collect())
    }

    fn open(&self, dev: DeviceId, opts: &OpenOptions) -> Result<DeviceCtx, Error> {
        self.hook("open");
        self.host.log(
            Level::Info,
            "test-provider",
            &format!("open {dev:?} {opts:?}"),
        );
        self.host.metric(1, 1);
        Ok(DeviceCtx::new(Opened {
            device: dev,
            host: self.host,
        }))
    }

    fn memory_info(&self, ctx: &DeviceCtx) -> Result<DeviceMemoryInfo, Error> {
        self.hook("memory_info");
        let opened = ctx
            .downcast_ref::<Opened>()
            .expect("a context this provider opened");
        Ok(DeviceMemoryInfo {
            total: Bytes(1 << 30),
            free: Bytes(u64::from(opened.device.ordinal)),
            budget: Some(Bytes(900)),
            host_visible_device_local: None,
            max_alloc: Bytes(1 << 28),
            unified: false,
        })
    }

    fn ops(&self) -> &OpRegistry {
        self.hook("ops");
        &self.ops
    }

    fn query(&self, _: &DeviceCtx, op: &OpKey, shape: &ShapeSig) -> Option<OpCaps> {
        self.hook("query");
        if op.variant != 0 {
            return None;
        }
        Some(OpCaps {
            workspace: Bytes(u64::from(shape.rows) * 16),
            est: Some(Nanos(1500)),
            layout: op.layout,
            capture_safe: true,
            deterministic: true,
            numerics: Numerics {
                accumulate: syntropy_core::AccumKind::F32,
                act_in: syntropy_core::ActQuant::None,
                out: DType::F32,
                softmax_f32: false,
                state_f32: false,
            },
            kernel_hash: Blake3([3; 32]),
            source_key: Blake3([4; 32]),
        })
    }

    fn alloc(&self, _: &DeviceCtx, req: AllocRequest) -> Result<DeviceBuffer, Error> {
        self.hook("alloc");
        if let Some(len) = self
            .host
            .config("fixture.long_error")
            .and_then(|v| v.parse::<usize>().ok())
        {
            return Err(Error::new(
                ErrorClass::Capacity,
                syntropy_kernels::error::codes::ALLOC_FAILED,
                "x".repeat(len),
            ));
        }
        if req.bytes.0 == 0 {
            return Err(Error::new(
                ErrorClass::Capacity,
                syntropy_kernels::error::codes::ALLOC_FAILED,
                "alloc of zero bytes".to_owned(),
            ));
        }
        let handle = self.handle();
        Ok(DeviceBuffer {
            addr: 0x1000 * handle,
            bytes: req.bytes,
            kind: req.kind,
            handle,
        })
    }

    fn free(&self, _: &DeviceCtx, _: DeviceBuffer) {
        self.hook("free");
        self.host.metric(FREED, 1);
    }

    fn map_segment(
        &self,
        _: &DeviceCtx,
        reservation: &DeviceBuffer,
        offset: Bytes,
        bytes: Bytes,
    ) -> Result<Segment, Error> {
        self.hook("map_segment");
        Ok(Segment {
            reservation: reservation.handle,
            offset,
            bytes,
            handle: self.handle(),
        })
    }

    fn unmap_segment(&self, _: &DeviceCtx, seg: Segment) -> Result<(), Error> {
        self.hook("unmap_segment");
        if seg.bytes.0 == 0 {
            return Err(echo("unmap_segment", format!("{seg:?}")));
        }
        Ok(())
    }

    fn register_host(&self, _: &DeviceCtx, region: HostRegion) -> Result<Registration, Error> {
        self.hook("register_host");
        Ok(Registration {
            region,
            handle: self.handle(),
        })
    }

    fn host_registrar(&self, _: &DeviceCtx) -> Option<Box<dyn HostRegistrar>> {
        self.hook("host_registrar");
        Some(Box::new(Registrar { host: self.host }))
    }

    fn import(&self, _: &DeviceCtx, region: HostRegion) -> Result<DeviceBuffer, Error> {
        self.hook("import");
        Ok(DeviceBuffer {
            addr: region.addr,
            bytes: region.len,
            kind: MemKind::HostMapped,
            handle: self.handle(),
        })
    }

    fn set_resident(
        &self,
        _: &DeviceCtx,
        bufs: &[&DeviceBuffer],
        resident: bool,
    ) -> Result<(), Error> {
        self.hook("set_resident");
        if bufs.len() == 2 {
            return Err(echo("set_resident", format!("{bufs:?} {resident}")));
        }
        Ok(())
    }

    fn repack(
        &self,
        op: &OpKey,
        src: &[u8],
        dst: &mut [u8],
    ) -> Option<Result<LayoutVersion, Error>> {
        self.hook("repack");
        if op.layout != LayoutVersion(1) {
            return None;
        }
        for (d, s) in dst.iter_mut().zip(src.iter().rev()) {
            *d = *s;
        }
        Some(Ok(LayoutVersion(9)))
    }

    fn repacked_len(&self, op: &OpKey, src: Bytes) -> Option<Bytes> {
        self.hook("repacked_len");
        (op.layout == LayoutVersion(1)).then_some(Bytes(src.0 + 8))
    }

    fn stream(&self, _: &DeviceCtx, prio: StreamPriority) -> Result<Stream, Error> {
        self.hook("stream");
        let handle = match prio {
            StreamPriority::Compute => ECHO,
            _ => 100 + self.handle(),
        };
        Ok(Stream { handle })
    }

    fn event(&self, _: &DeviceCtx) -> Result<Event, Error> {
        self.hook("event");
        Ok(Event {
            handle: self.handle(),
        })
    }

    fn record(&self, _: &DeviceCtx, _: &Stream, _: &Event) -> Result<(), Error> {
        self.hook("record");
        Ok(())
    }

    fn wait(&self, _: &DeviceCtx, _: &Stream, _: &Event) -> Result<(), Error> {
        self.hook("wait");
        Ok(())
    }

    fn signal_host(
        &self,
        _: &DeviceCtx,
        s: &Stream,
        flag: HostFlag,
        value: u32,
    ) -> Result<(), Error> {
        self.hook("signal_host");
        if s.handle == ECHO {
            return Err(echo("signal_host", format!("{flag:?} {value}")));
        }
        Ok(())
    }

    fn wait_host(
        &self,
        _: &DeviceCtx,
        s: &Stream,
        flag: HostFlag,
        value: u32,
        cmp: WaitCmp,
        reset: Option<u32>,
    ) -> Result<(), Error> {
        self.hook("wait_host");
        if s.handle == ECHO {
            return Err(echo(
                "wait_host",
                format!("{flag:?} {value} {cmp:?} {reset:?}"),
            ));
        }
        Ok(())
    }

    fn arm_wake(&self, _: &DeviceCtx, s: &Stream, waker: &HostWaker) -> Result<(), Error> {
        self.hook("arm_wake");
        if s.handle == ECHO {
            return Err(echo("arm_wake", format!("{waker:?}")));
        }
        Ok(())
    }

    fn poll(&self, _: &DeviceCtx, e: &Event) -> Result<bool, Error> {
        self.hook("poll");
        Ok(e.handle.is_multiple_of(2))
    }

    fn elapsed(&self, _: &DeviceCtx, start: &Event, end: &Event) -> Option<Nanos> {
        self.hook("elapsed");
        end.handle.checked_sub(start.handle).map(Nanos)
    }

    fn synchronize(&self, _: &DeviceCtx, _: &Stream) -> Result<(), Error> {
        self.hook("synchronize");
        Ok(())
    }

    fn copy(
        &self,
        _: &DeviceCtx,
        _: &Stream,
        src: BufferSlice,
        dst: BufferSlice,
    ) -> Result<(), Error> {
        self.hook("copy");
        if src.bytes != dst.bytes {
            return Err(echo("copy", format!("{src:?} {dst:?}")));
        }
        Ok(())
    }

    fn copy_batch(
        &self,
        _: &DeviceCtx,
        s: &Stream,
        pairs: &[(BufferSlice, BufferSlice)],
    ) -> Result<(), Error> {
        self.hook("copy_batch");
        if s.handle == ECHO {
            return Err(echo("copy_batch", format!("{pairs:?}")));
        }
        Ok(())
    }

    fn prepare(&self, _: &DeviceCtx, op: &OpKey, shape: &ShapeSig) -> Result<PreparedOp, Error> {
        self.hook("prepare");
        let mut prepared = PreparedOp::new(Bytes(u64::from(shape.rows) * 16));
        prepared.handle = u64::from(op.layout.0) + 1;
        Ok(prepared)
    }

    fn launch(
        &self,
        _: &DeviceCtx,
        s: &Stream,
        op: &PreparedOp,
        args: &OpArgs<'_>,
    ) -> Result<(), Error> {
        self.hook("launch");
        if s.handle == ECHO {
            return Err(echo("launch", format!("{op:?} {args:?}")));
        }
        Ok(())
    }

    fn begin_capture(&self, _: &DeviceCtx, _: &Stream) -> Result<(), Error> {
        self.hook("begin_capture");
        Ok(())
    }

    fn end_capture(&self, _: &DeviceCtx, _: &Stream) -> Result<CapturedGraph, Error> {
        self.hook("end_capture");
        Ok(CapturedGraph {
            handle: self.handle(),
        })
    }

    fn replay(&self, _: &DeviceCtx, _: &Stream, _: &CapturedGraph) -> Result<(), Error> {
        self.hook("replay");
        Ok(())
    }

    fn health(&self, _: &DeviceCtx) -> DeviceHealth {
        self.hook("health");
        DeviceHealth::Ok
    }
}

syntropy_kernels::abi::export! {
    build_id: BUILD_ID,
    providers: [TestProvider = TestProvider::new],
}
