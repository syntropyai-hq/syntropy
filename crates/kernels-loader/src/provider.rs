// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`ModuleProvider`]: a `KernelProvider` that calls through a module's function table
//! (RFC-0040 §10.4).

use std::ffi::{CStr, c_void};
use std::sync::{Arc, Mutex, PoisonError};

use syntropy_core::resource::Backend;
use syntropy_core::{Bytes, DeviceId, ErrorClass, Nanos};
use syntropy_kernels::abi::status::error_of_status;
use syntropy_kernels::abi::types::{
    SynAllocRequest, SynBuffer, SynDeviceId, SynHealth, SynHostRegion, SynMemoryInfo, SynOpArgs,
    SynOpenOptions, SynOutBuf, SynPreparedOp, SynRegistrarVtbl, SynRegistration, SynScalar,
    SynSegment, SynSlice, SynSlicePair, SynTensorArg,
};
use syntropy_kernels::abi::wire::{
    BuildWire, CapsWire, backend_of_code, cmp_code, domain_of_code, from_json, internal,
    mem_kind_code, priority_code, to_json, wake_code,
};
use syntropy_kernels::abi::{SynProviderVtbl, SynStatus};
use syntropy_kernels::{
    AllocRequest, BufferSlice, CapturedGraph, DeviceBuffer, DeviceCtx, DeviceHealth,
    DeviceMemoryInfo, Error, ErrorCode, Event, HostFlag, HostRegFlags, HostRegion, HostRegistrar,
    HostWaker, KernelProvider, LayoutVersion, MemoryDomain, OpArgs, OpCaps, OpKey, OpRegistry,
    OpenOptions, PreparedOp, ProviderBuild, ProviderDevice, ProviderId, Registration, Segment,
    ShapeSig, Stream, StreamPriority, WaitCmp,
};

use crate::module::ModuleError;
use crate::sys::RawLibrary;

/// A module's table and the library that keeps its code alive.
#[derive(Clone)]
struct Table {
    vt: SynProviderVtbl,
    /// Keeps the library mapped for as long as any proxy, context or registrar exists. A library
    /// is never unloaded, so this is documentation as much as ownership.
    _lib: Arc<RawLibrary>,
}

// SAFETY: a table is plain function pointers and the module's own state pointer. The module's
// thunks take `&self`-shaped access to that state (the provider is `Send + Sync` by the
// `KernelProvider` bound), and the library outlives the table.
unsafe impl Send for Table {}
// SAFETY: as above.
unsafe impl Sync for Table {}

/// How many bytes the first attempt to read a variable-size result offers.
const FIRST_BUFFER: usize = 4096;

impl Table {
    /// Turns a call's status into a result. The message of a failure is fetched at once, on the
    /// calling thread, because the module keeps it per thread.
    fn check(&self, status: SynStatus, ctx: u64) -> Result<(), Error> {
        if status.is_ok() {
            return Ok(());
        }
        let message = self.last_error(ctx);
        let error = error_of_status(status, message)
            .unwrap_or_else(|| internal("a failed status with no class".into()));
        Err(error)
    }

    fn last_error(&self, ctx: u64) -> String {
        let mut buf = vec![0u8; 512];
        let mut len = 0usize;
        let status = (self.vt.last_error)(self.vt.slf, ctx, buf.as_mut_ptr(), buf.len(), &mut len);
        if !status.is_ok() {
            return "the module could not report its error".to_owned();
        }
        if len > buf.len() {
            buf.resize(len, 0);
            let status =
                (self.vt.last_error)(self.vt.slf, ctx, buf.as_mut_ptr(), buf.len(), &mut len);
            if !status.is_ok() {
                return "the module could not report its error".to_owned();
            }
        }
        buf.truncate(len.min(buf.len()));
        String::from_utf8_lossy(&buf).into_owned()
    }

    /// Calls a function that writes a variable-size result into a buffer the caller supplies,
    /// and returns the bytes; the call is repeated once with a buffer of the size it reported.
    fn fetch(
        &self,
        ctx: u64,
        call: impl Fn(*mut SynOutBuf) -> SynStatus,
    ) -> Result<Vec<u8>, Error> {
        let mut buf = vec![0u8; FIRST_BUFFER];
        for _ in 0..2 {
            let mut out = SynOutBuf {
                ptr: buf.as_mut_ptr(),
                cap: buf.len(),
                len: 0,
            };
            self.check(call(&raw mut out), ctx)?;
            if out.len <= buf.len() {
                buf.truncate(out.len);
                return Ok(buf);
            }
            buf = vec![0u8; out.len];
        }
        Err(internal(
            "the module's result grew between two calls".into(),
        ))
    }
}

/// The module's side of a `DeviceCtx`: the handle `open` returned. Dropping it closes the
/// context in the module, on the thread that opened it (a `DeviceCtx` is not `Send`).
struct HostCtx {
    handle: u64,
    table: Table,
    fault: Arc<Fault>,
}

impl Drop for HostCtx {
    fn drop(&mut self) {
        let status = (self.table.vt.close_ctx)(self.table.vt.slf, self.handle);
        if let Err(error) = self.table.check(status, self.handle) {
            self.fault.record(&error);
        }
    }
}

/// The first `Internal` error a call produced, for the methods that cannot return one.
#[derive(Default)]
struct Fault(Mutex<Option<Error>>);

impl Fault {
    fn record(&self, error: &Error) {
        if error.class == ErrorClass::Internal {
            self.0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get_or_insert_with(|| error.clone());
        }
    }

    fn get(&self) -> Option<Error> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

/// A [`KernelProvider`] backed by a provider library's function table.
pub struct ModuleProvider {
    table: Table,
    id: ProviderId,
    backend: Backend,
    domain: MemoryDomain,
    build: ProviderBuild,
    ops: OpRegistry,
    fault: Arc<Fault>,
}

impl std::fmt::Debug for ModuleProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModuleProvider")
            .field("id", &self.id)
            .field("backend", &self.backend)
            .field("domain", &self.domain)
            .finish_non_exhaustive()
    }
}

fn abi_error(message: String) -> ModuleError {
    ModuleError(Error {
        class: ErrorClass::Unsupported,
        code: ErrorCode(4),
        message,
    })
}

impl ModuleProvider {
    /// Wraps `vt`, reading the provider's identity, build and op registry.
    ///
    /// The provider ID is interned (it is a `&'static str` in `ProviderId`), as are the toolchain
    /// keys of its build ([`syntropy_kernels::abi::wire::intern`]): each distinct text is
    /// allocated once for the process, however many times a module is loaded.
    pub(crate) fn new(vt: SynProviderVtbl, lib: Arc<RawLibrary>) -> Result<Self, ModuleError> {
        if vt.id.is_null() {
            return Err(abi_error("a provider table has no ID".into()));
        }
        // SAFETY: the ABI contract makes `id` a NUL-terminated string the module keeps alive.
        let id = unsafe { CStr::from_ptr(vt.id) }
            .to_string_lossy()
            .into_owned();
        let id = ProviderId(syntropy_kernels::abi::wire::intern(&id));
        let backend = backend_of_code(vt.backend).ok_or_else(|| {
            abi_error(format!(
                "provider `{}` has backend code {}",
                id.0, vt.backend
            ))
        })?;
        let domain = domain_of_code(vt.domain).ok_or_else(|| {
            abi_error(format!("provider `{}` has domain code {}", id.0, vt.domain))
        })?;
        let table = Table { vt, _lib: lib };
        let fault = Arc::new(Fault::default());
        let read = |what: &str, result: Result<Vec<u8>, Error>| {
            result.map_err(|e| {
                ModuleError(Error {
                    message: format!("provider `{}`, reading {what}: {}", id.0, e.message),
                    ..e
                })
            })
        };
        let build: BuildWire = from_json(&read(
            "its build",
            table.fetch(0, |out| (table.vt.build)(table.vt.slf, out)),
        )?)
        .map_err(ModuleError)?;
        let keys: Vec<OpKey> = from_json(&read(
            "its ops",
            table.fetch(0, |out| (table.vt.ops)(table.vt.slf, out)),
        )?)
        .map_err(ModuleError)?;
        let mut ops = OpRegistry::default();
        for key in keys {
            ops.register(key)
                .map_err(|e| abi_error(format!("provider `{}` registers a bad op: {e:?}", id.0)))?;
        }
        Ok(Self {
            table,
            id,
            backend,
            domain,
            build: build.into_build(),
            ops,
            fault,
        })
    }

    /// The first `Internal` error the module reported through a method whose signature cannot
    /// carry one (`free`, `query`, `repack`, `repacked_len`, `elapsed`, `host_registrar`, and the
    /// closing of a context). `None` if there was none.
    #[must_use]
    pub fn fault(&self) -> Option<Error> {
        self.fault.get()
    }

    fn vt(&self) -> &SynProviderVtbl {
        &self.table.vt
    }

    /// Checks `status` and records an `Internal` failure.
    fn done(&self, status: SynStatus, ctx: u64) -> Result<(), Error> {
        self.table
            .check(status, ctx)
            .inspect_err(|e| self.fault.record(e))
    }

    fn handle(ctx: &DeviceCtx) -> Result<u64, Error> {
        ctx.downcast_ref::<HostCtx>()
            .map(|c| c.handle)
            .ok_or_else(|| internal("the DeviceCtx was not opened by this module provider".into()))
    }

    fn ctx_of(&self, ctx: &DeviceCtx) -> Result<u64, Error> {
        Self::handle(ctx).inspect_err(|e| self.fault.record(e))
    }
}

fn unknown(what: &str) -> Error {
    internal(format!("a {what} the ABI has no code for"))
}

impl KernelProvider for ModuleProvider {
    fn id(&self) -> ProviderId {
        self.id
    }

    fn backend(&self) -> Backend {
        self.backend
    }

    fn build(&self) -> ProviderBuild {
        self.build.clone()
    }

    fn domain(&self) -> MemoryDomain {
        self.domain
    }

    fn enumerate(&self) -> Result<Vec<ProviderDevice>, Error> {
        let bytes = self
            .table
            .fetch(0, |out| (self.vt().enumerate)(self.vt().slf, out))
            .inspect_err(|e| self.fault.record(e))?;
        from_json(&bytes)
    }

    fn open(&self, dev: DeviceId, opts: &OpenOptions) -> Result<DeviceCtx, Error> {
        let dev = SynDeviceId::from(dev);
        let opts = SynOpenOptions::from(opts);
        let mut handle = 0u64;
        let status = (self.vt().open)(self.vt().slf, &dev, &opts, &mut handle);
        self.done(status, 0)?;
        Ok(DeviceCtx::new(HostCtx {
            handle,
            table: self.table.clone(),
            fault: Arc::clone(&self.fault),
        }))
    }

    fn memory_info(&self, ctx: &DeviceCtx) -> Result<DeviceMemoryInfo, Error> {
        let ctx = self.ctx_of(ctx)?;
        let mut info = SynMemoryInfo::default();
        let status = (self.vt().memory_info)(self.vt().slf, ctx, &mut info);
        self.done(status, ctx)?;
        Ok((&info).into())
    }

    fn ops(&self) -> &OpRegistry {
        &self.ops
    }

    fn query(&self, ctx: &DeviceCtx, op: &OpKey, shape: &ShapeSig) -> Option<OpCaps> {
        let run = || -> Result<Option<OpCaps>, Error> {
            let ctx = self.ctx_of(ctx)?;
            let (op, shape) = (to_json(op)?, to_json(shape)?);
            let bytes = self
                .table
                .fetch(ctx, |out| {
                    (self.vt().query)(
                        self.vt().slf,
                        ctx,
                        op.as_ptr(),
                        op.len(),
                        shape.as_ptr(),
                        shape.len(),
                        out,
                    )
                })
                .inspect_err(|e| self.fault.record(e))?;
            if bytes.is_empty() {
                return Ok(None);
            }
            from_json::<CapsWire>(&bytes)?.into_caps().map(Some)
        };
        run().inspect_err(|e| self.fault.record(e)).ok().flatten()
    }

    fn alloc(&self, ctx: &DeviceCtx, req: AllocRequest) -> Result<DeviceBuffer, Error> {
        let ctx = self.ctx_of(ctx)?;
        mem_kind_code(req.kind).ok_or_else(|| unknown("memory kind"))?;
        let req = SynAllocRequest::from(&req);
        let mut out = SynBuffer {
            addr: 0,
            bytes: 0,
            handle: 0,
            kind: 0,
            reserved: 0,
        };
        let status = (self.vt().alloc)(self.vt().slf, ctx, &req, &mut out);
        self.done(status, ctx)?;
        out.to_buffer().ok_or_else(|| unknown("memory kind"))
    }

    fn free(&self, ctx: &DeviceCtx, buf: DeviceBuffer) {
        let Ok(ctx) = self.ctx_of(ctx) else { return };
        let Some(wire) = SynBuffer::of(&buf) else {
            self.fault.record(&unknown("memory kind"));
            return;
        };
        let status = (self.vt().free)(self.vt().slf, ctx, &wire);
        let _ = self.done(status, ctx);
    }

    fn map_segment(
        &self,
        ctx: &DeviceCtx,
        reservation: &DeviceBuffer,
        offset: Bytes,
        bytes: Bytes,
    ) -> Result<Segment, Error> {
        let ctx = self.ctx_of(ctx)?;
        let wire = SynBuffer::of(reservation).ok_or_else(|| unknown("memory kind"))?;
        let mut out = SynSegment {
            reservation: 0,
            offset: 0,
            bytes: 0,
            handle: 0,
        };
        let status =
            (self.vt().map_segment)(self.vt().slf, ctx, &wire, offset.0, bytes.0, &mut out);
        self.done(status, ctx)?;
        Ok((&out).into())
    }

    fn unmap_segment(&self, ctx: &DeviceCtx, seg: Segment) -> Result<(), Error> {
        let ctx = self.ctx_of(ctx)?;
        let wire = SynSegment::from(&seg);
        let status = (self.vt().unmap_segment)(self.vt().slf, ctx, &wire);
        self.done(status, ctx)
    }

    fn register_host(&self, ctx: &DeviceCtx, region: HostRegion) -> Result<Registration, Error> {
        let ctx = self.ctx_of(ctx)?;
        let mut out = SynRegistration {
            region: SynHostRegion { addr: 0, len: 0 },
            handle: 0,
        };
        let status = (self.vt().register_host)(self.vt().slf, ctx, region.into(), &mut out);
        self.done(status, ctx)?;
        Ok((&out).into())
    }

    fn host_registrar(&self, ctx: &DeviceCtx) -> Option<Box<dyn HostRegistrar>> {
        let ctx = self.ctx_of(ctx).ok()?;
        // The module overwrites these placeholders when it has a registrar; `struct_size` stays 0
        // when it has none, and then they are never called.
        extern "C" fn unset_register(
            _: *mut c_void,
            _: SynHostRegion,
            _: u32,
            _: *mut SynRegistration,
        ) -> SynStatus {
            SynStatus::OK
        }
        extern "C" fn unset_unregister(_: *mut c_void, _: *const SynRegistration) -> SynStatus {
            SynStatus::OK
        }
        extern "C" fn unset_close(_: *mut c_void) -> SynStatus {
            SynStatus::OK
        }
        let mut vtbl = SynRegistrarVtbl {
            struct_size: 0,
            register: unset_register,
            unregister: unset_unregister,
            close: unset_close,
            slf: std::ptr::null_mut(),
        };
        let status = (self.vt().host_registrar)(self.vt().slf, ctx, &mut vtbl);
        self.done(status, ctx).ok()?;
        if vtbl.struct_size == 0 {
            return None;
        }
        if (vtbl.struct_size as usize) < size_of::<SynRegistrarVtbl>() {
            self.fault.record(&internal(
                "a registrar table is smaller than ABI 1's".into(),
            ));
            return None;
        }
        Some(Box::new(ModuleRegistrar {
            vtbl,
            table: self.table.clone(),
            fault: Arc::clone(&self.fault),
        }))
    }

    fn import(&self, ctx: &DeviceCtx, region: HostRegion) -> Result<DeviceBuffer, Error> {
        let ctx = self.ctx_of(ctx)?;
        let mut out = SynBuffer {
            addr: 0,
            bytes: 0,
            handle: 0,
            kind: 0,
            reserved: 0,
        };
        let status = (self.vt().import)(self.vt().slf, ctx, region.into(), &mut out);
        self.done(status, ctx)?;
        out.to_buffer().ok_or_else(|| unknown("memory kind"))
    }

    fn set_resident(
        &self,
        ctx: &DeviceCtx,
        bufs: &[&DeviceBuffer],
        resident: bool,
    ) -> Result<(), Error> {
        let ctx = self.ctx_of(ctx)?;
        let wire = bufs
            .iter()
            .map(|b| SynBuffer::of(b).ok_or_else(|| unknown("memory kind")))
            .collect::<Result<Vec<_>, _>>()?;
        let status = (self.vt().set_resident)(
            self.vt().slf,
            ctx,
            wire.as_ptr(),
            wire.len(),
            u32::from(resident),
        );
        self.done(status, ctx)
    }

    fn repack(
        &self,
        op: &OpKey,
        src: &[u8],
        dst: &mut [u8],
    ) -> Option<Result<LayoutVersion, Error>> {
        let op = match to_json(op) {
            Ok(op) => op,
            Err(e) => return Some(Err(e)),
        };
        let (mut present, mut layout) = (0u32, 0u32);
        let status = (self.vt().repack)(
            self.vt().slf,
            op.as_ptr(),
            op.len(),
            src.as_ptr(),
            src.len(),
            dst.as_mut_ptr(),
            dst.len(),
            &mut present,
            &mut layout,
        );
        if let Err(e) = self.done(status, 0) {
            return Some(Err(e));
        }
        (present != 0).then(|| {
            u16::try_from(layout)
                .map(LayoutVersion)
                .map_err(|_| unknown("layout version"))
        })
    }

    fn repacked_len(&self, op: &OpKey, src: Bytes) -> Option<Bytes> {
        let op = to_json(op).ok()?;
        let (mut present, mut out) = (0u32, 0u64);
        let status = (self.vt().repacked_len)(
            self.vt().slf,
            op.as_ptr(),
            op.len(),
            src.0,
            &mut present,
            &mut out,
        );
        self.done(status, 0).ok()?;
        (present != 0).then_some(Bytes(out))
    }

    fn stream(&self, ctx: &DeviceCtx, prio: StreamPriority) -> Result<Stream, Error> {
        let ctx = self.ctx_of(ctx)?;
        let prio = priority_code(prio).ok_or_else(|| unknown("stream priority"))?;
        let mut handle = 0u64;
        let status = (self.vt().stream)(self.vt().slf, ctx, prio, &mut handle);
        self.done(status, ctx)?;
        Ok(Stream { handle })
    }

    fn event(&self, ctx: &DeviceCtx) -> Result<Event, Error> {
        let ctx = self.ctx_of(ctx)?;
        let mut handle = 0u64;
        let status = (self.vt().event)(self.vt().slf, ctx, &mut handle);
        self.done(status, ctx)?;
        Ok(Event { handle })
    }

    fn record(&self, ctx: &DeviceCtx, s: &Stream, e: &Event) -> Result<(), Error> {
        let ctx = self.ctx_of(ctx)?;
        self.done(
            (self.vt().record)(self.vt().slf, ctx, s.handle, e.handle),
            ctx,
        )
    }

    fn wait(&self, ctx: &DeviceCtx, s: &Stream, e: &Event) -> Result<(), Error> {
        let ctx = self.ctx_of(ctx)?;
        self.done(
            (self.vt().wait)(self.vt().slf, ctx, s.handle, e.handle),
            ctx,
        )
    }

    fn signal_host(
        &self,
        ctx: &DeviceCtx,
        s: &Stream,
        flag: HostFlag,
        value: u32,
    ) -> Result<(), Error> {
        let ctx = self.ctx_of(ctx)?;
        let flag = u32::from(flag.page) << 16 | u32::from(flag.slot);
        self.done(
            (self.vt().signal_host)(self.vt().slf, ctx, s.handle, flag, value),
            ctx,
        )
    }

    fn wait_host(
        &self,
        ctx: &DeviceCtx,
        s: &Stream,
        flag: HostFlag,
        value: u32,
        cmp: WaitCmp,
        reset: Option<u32>,
    ) -> Result<(), Error> {
        let ctx = self.ctx_of(ctx)?;
        let flag = u32::from(flag.page) << 16 | u32::from(flag.slot);
        self.done(
            (self.vt().wait_host)(
                self.vt().slf,
                ctx,
                s.handle,
                flag,
                value,
                cmp_code(cmp),
                u32::from(reset.is_some()),
                reset.unwrap_or(0),
            ),
            ctx,
        )
    }

    fn arm_wake(&self, ctx: &DeviceCtx, s: &Stream, waker: &HostWaker) -> Result<(), Error> {
        let ctx = self.ctx_of(ctx)?;
        let kind = wake_code(waker.kind).ok_or_else(|| unknown("wake kind"))?;
        self.done(
            (self.vt().arm_wake)(self.vt().slf, ctx, s.handle, kind, waker.handle),
            ctx,
        )
    }

    fn poll(&self, ctx: &DeviceCtx, e: &Event) -> Result<bool, Error> {
        let ctx = self.ctx_of(ctx)?;
        let mut done = 0u32;
        self.done(
            (self.vt().poll)(self.vt().slf, ctx, e.handle, &mut done),
            ctx,
        )?;
        Ok(done != 0)
    }

    fn elapsed(&self, ctx: &DeviceCtx, start: &Event, end: &Event) -> Option<Nanos> {
        let ctx = self.ctx_of(ctx).ok()?;
        let (mut present, mut nanos) = (0u32, 0u64);
        let status = (self.vt().elapsed)(
            self.vt().slf,
            ctx,
            start.handle,
            end.handle,
            &mut present,
            &mut nanos,
        );
        self.done(status, ctx).ok()?;
        (present != 0).then_some(Nanos(nanos))
    }

    fn synchronize(&self, ctx: &DeviceCtx, s: &Stream) -> Result<(), Error> {
        let ctx = self.ctx_of(ctx)?;
        self.done((self.vt().synchronize)(self.vt().slf, ctx, s.handle), ctx)
    }

    fn copy(
        &self,
        ctx: &DeviceCtx,
        s: &Stream,
        src: BufferSlice,
        dst: BufferSlice,
    ) -> Result<(), Error> {
        let ctx = self.ctx_of(ctx)?;
        let (src, dst) = (SynSlice::from(src), SynSlice::from(dst));
        self.done(
            (self.vt().copy)(self.vt().slf, ctx, s.handle, &src, &dst),
            ctx,
        )
    }

    fn copy_batch(
        &self,
        ctx: &DeviceCtx,
        s: &Stream,
        pairs: &[(BufferSlice, BufferSlice)],
    ) -> Result<(), Error> {
        let ctx = self.ctx_of(ctx)?;
        let wire: Vec<SynSlicePair> = pairs
            .iter()
            .map(|&(src, dst)| SynSlicePair {
                src: src.into(),
                dst: dst.into(),
            })
            .collect();
        self.done(
            (self.vt().copy_batch)(self.vt().slf, ctx, s.handle, wire.as_ptr(), wire.len()),
            ctx,
        )
    }

    fn prepare(&self, ctx: &DeviceCtx, op: &OpKey, shape: &ShapeSig) -> Result<PreparedOp, Error> {
        let ctx = self.ctx_of(ctx)?;
        let (op, shape) = (to_json(op)?, to_json(shape)?);
        let mut out = SynPreparedOp {
            workspace: 0,
            handle: 0,
        };
        let status = (self.vt().prepare)(
            self.vt().slf,
            ctx,
            op.as_ptr(),
            op.len(),
            shape.as_ptr(),
            shape.len(),
            &mut out,
        );
        self.done(status, ctx)?;
        let mut prepared = PreparedOp::new(Bytes(out.workspace));
        prepared.handle = out.handle;
        Ok(prepared)
    }

    fn launch(
        &self,
        ctx: &DeviceCtx,
        s: &Stream,
        op: &PreparedOp,
        args: &OpArgs<'_>,
    ) -> Result<(), Error> {
        let ctx = self.ctx_of(ctx)?;
        let tensors = |list: &[syntropy_kernels::TensorArg]| {
            list.iter()
                .map(|t| SynTensorArg::of(t).ok_or_else(|| unknown("dtype")))
                .collect::<Result<Vec<_>, _>>()
        };
        let (inputs, outputs) = (tensors(args.inputs)?, tensors(args.outputs)?);
        let weights = args
            .weights
            .iter()
            .map(|w| {
                syntropy_kernels::abi::SynBufferView::of(w).ok_or_else(|| unknown("buffer view"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let statics = args
            .statics
            .iter()
            .map(|v| SynScalar::of(v).ok_or_else(|| unknown("scalar")))
            .collect::<Result<Vec<_>, _>>()?;
        let wire = SynOpArgs {
            struct_size: size_of::<SynOpArgs>() as u32,
            has_workspace: u32::from(args.workspace.is_some()),
            inputs: inputs.as_ptr(),
            n_inputs: inputs.len(),
            outputs: outputs.as_ptr(),
            n_outputs: outputs.len(),
            weights: weights.as_ptr(),
            n_weights: weights.len(),
            step: args.step.into(),
            workspace: args.workspace.map_or(
                SynSlice {
                    buffer: 0,
                    offset: 0,
                    bytes: 0,
                },
                SynSlice::from,
            ),
            statics: statics.as_ptr(),
            n_statics: statics.len(),
        };
        let prepared = SynPreparedOp {
            workspace: op.workspace.0,
            handle: op.handle,
        };
        self.done(
            (self.vt().launch)(self.vt().slf, ctx, s.handle, &prepared, &wire),
            ctx,
        )
    }

    fn begin_capture(&self, ctx: &DeviceCtx, s: &Stream) -> Result<(), Error> {
        let ctx = self.ctx_of(ctx)?;
        self.done((self.vt().begin_capture)(self.vt().slf, ctx, s.handle), ctx)
    }

    fn end_capture(&self, ctx: &DeviceCtx, s: &Stream) -> Result<CapturedGraph, Error> {
        let ctx = self.ctx_of(ctx)?;
        let mut handle = 0u64;
        self.done(
            (self.vt().end_capture)(self.vt().slf, ctx, s.handle, &mut handle),
            ctx,
        )?;
        Ok(CapturedGraph { handle })
    }

    fn replay(&self, ctx: &DeviceCtx, s: &Stream, g: &CapturedGraph) -> Result<(), Error> {
        let ctx = self.ctx_of(ctx)?;
        self.done(
            (self.vt().replay)(self.vt().slf, ctx, s.handle, g.handle),
            ctx,
        )
    }

    fn health(&self, ctx: &DeviceCtx) -> DeviceHealth {
        let sticky = |e: &Error| DeviceHealth::Sticky { code: e.code };
        let ctx = match self.ctx_of(ctx) {
            Ok(ctx) => ctx,
            Err(e) => return sticky(&e),
        };
        let mut out = SynHealth { kind: 0, detail: 0 };
        let status = (self.vt().health)(self.vt().slf, ctx, &mut out);
        if let Err(e) = self.done(status, ctx) {
            return sticky(&e);
        }
        out.to_health().unwrap_or_else(|| {
            let e = unknown("health");
            self.fault.record(&e);
            sticky(&e)
        })
    }
}

/// A module's `HostRegistrar`: its own small table, usable from another thread.
struct ModuleRegistrar {
    vtbl: SynRegistrarVtbl,
    table: Table,
    fault: Arc<Fault>,
}

// SAFETY: a `HostRegistrar` is `Send` by its trait. The state pointer belongs to the module's
// registrar, which the module built to be used from the thread that holds it; `ModuleRegistrar`
// is not `Sync`, so calls never overlap.
unsafe impl Send for ModuleRegistrar {}

impl ModuleRegistrar {
    fn check(&self, status: SynStatus) -> Result<(), Error> {
        self.table
            .check(status, 0)
            .inspect_err(|e| self.fault.record(e))
    }
}

impl HostRegistrar for ModuleRegistrar {
    fn register(&self, region: HostRegion, flags: HostRegFlags) -> Result<Registration, Error> {
        let mut out = SynRegistration {
            region: SynHostRegion { addr: 0, len: 0 },
            handle: 0,
        };
        let status = (self.vtbl.register)(
            self.vtbl.slf,
            region.into(),
            u32::from(flags.bits()),
            &mut out,
        );
        self.check(status)?;
        Ok((&out).into())
    }

    fn unregister(&self, reg: Registration) -> Result<(), Error> {
        let wire = SynRegistration::from(&reg);
        self.check((self.vtbl.unregister)(self.vtbl.slf, &wire))
    }
}

impl Drop for ModuleRegistrar {
    fn drop(&mut self) {
        let status = (self.vtbl.close)(self.vtbl.slf);
        let _ = self.check(status);
    }
}
