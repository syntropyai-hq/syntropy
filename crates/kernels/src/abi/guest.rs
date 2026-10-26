// SPDX-License-Identifier: MIT OR Apache-2.0

//! The module side of the boundary, as safe code (RFC-0040 §10.4).
//!
//! [`export!`](crate::abi::export) expands the `extern "C"` thunks in the provider's crate. A
//! thunk turns its raw pointers into references, which is the only `unsafe` there is, and calls
//! one function of this module. Everything else lives here, where `unsafe` is forbidden:
//!
//! - [`call`] runs a body under `catch_unwind`. A caught panic makes the provider
//!   [`Sticky`](DeviceHealth::Sticky) with [`ErrorCode::PROVIDER_PANICKED`] and is reported as
//!   `Internal`; later calls other than `health`, `free` and `close_ctx` fail with
//!   [`ErrorCode::DEVICE_STICKY`], so the runner exits and restarts cleanly (RFC-0001 §3.4).
//! - Failures keep their message in a per-thread store, which `last_error` reads.
//! - Device contexts live in a per-thread table, keyed by the handle `open` returned. A
//!   `DeviceCtx` is not `Send`, so a context handle used on another thread finds no entry and
//!   fails with `Internal`: the thread-affinity rule of RFC-0040 §2.4 is enforced, not assumed.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU16, AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};
use std::thread::{self, ThreadId};

use smallvec::SmallVec;

use super::status::{SynStatus, status_of_error};
use super::types::{
    SynAllocRequest, SynBuffer, SynDeviceId, SynHealth, SynHostApi, SynHostRegion, SynMemoryInfo,
    SynOpArgs, SynOpenOptions, SynPreparedOp, SynRegistration, SynSegment, SynSlice, SynSlicePair,
    SynTensorArg,
};
use super::wire::{
    BuildWire, CapsWire, backend_code, cmp_of_code, domain_code, from_json, internal, panicked,
    priority_of_code, to_json, wake_of_code,
};
use crate::device::{DeviceCtx, DeviceHealth, ProviderDevice};
use crate::error::{Error, ErrorCode};
use crate::memory::{BufferSlice, HostRegFlags, HostRegistrar};
use crate::op::{OpArgs, OpKey, Scalar, ShapeSig, TensorArg};
use crate::provider::KernelProvider;
use crate::stream::{CapturedGraph, Event, HostFlag, HostWaker, Stream};
use syntropy_core::{BufferView, Bytes};

/// The result of a thunk body.
pub type R = Result<(), Error>;

/// The core's services, as the module sees them: a safe wrapper over [`SynHostApi`]. A provider's
/// constructor receives it, and may keep it.
#[derive(Clone, Copy)]
pub struct HostApi {
    log: extern "C" fn(u8, *const c_char, *const c_char, usize),
    metric: extern "C" fn(u32, u64),
    config: extern "C" fn(*const c_char) -> *const c_char,
    read_cstr: fn(*const c_char) -> Option<String>,
}

/// A log level of [`HostApi::log`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    /// Errors.
    Error = 0,
    /// Warnings.
    Warn = 1,
    /// Information.
    Info = 2,
    /// Debugging detail.
    Debug = 3,
    /// Tracing detail.
    Trace = 4,
}

impl HostApi {
    /// Wraps `raw`. `read_cstr` copies a NUL-terminated string out of memory the core owns; the
    /// thunks' expansion supplies it, because reading it is `unsafe`.
    #[must_use]
    pub fn new(raw: &SynHostApi, read_cstr: fn(*const c_char) -> Option<String>) -> Self {
        Self {
            log: raw.log,
            metric: raw.metric,
            config: raw.config,
            read_cstr,
        }
    }

    /// Logs `msg` under `target`.
    pub fn log(&self, level: Level, target: &str, msg: &str) {
        let Ok(target) = CString::new(target) else {
            return;
        };
        (self.log)(level as u8, target.as_ptr(), msg.as_ptr().cast(), msg.len());
    }

    /// Adds `value` to the metric `id`.
    pub fn metric(&self, id: u32, value: u64) {
        (self.metric)(id, value);
    }

    /// The value of the config `key` (`features.disable`, `kernels.*`), if the core sets one.
    #[must_use]
    pub fn config(&self, key: &str) -> Option<String> {
        let key = CString::new(key).ok()?;
        (self.read_cstr)((self.config)(key.as_ptr()))
    }
}

/// One provider inside a module: the provider, its sticky state and its per-thread error store.
pub struct Instance<P> {
    /// The provider.
    pub provider: P,
    /// What the core gave the module.
    pub host: HostApi,
    id: CString,
    /// 0 while healthy, else the code of the panic that made the provider sticky.
    sticky: AtomicU16,
    errors: Mutex<HashMap<ThreadId, String>>,
}

impl<P: KernelProvider> Instance<P> {
    /// Wraps `provider`.
    ///
    /// # Panics
    ///
    /// If the provider's ID holds a NUL byte, a bug in the provider.
    #[must_use]
    pub fn new(provider: P, host: HostApi) -> Self {
        let id = CString::new(provider.id().0).expect("a provider ID holds no NUL byte");
        Self {
            provider,
            host,
            id,
            sticky: AtomicU16::new(0),
            errors: Mutex::new(HashMap::new()),
        }
    }

    /// The ID, NUL-terminated, for [`SynProviderVtbl::id`](super::SynProviderVtbl::id). It lives
    /// as long as the instance.
    #[must_use]
    pub fn id_ptr(&self) -> *const c_char {
        self.id.as_ptr()
    }

    /// The identity codes of the vtable: `(backend, domain)`.
    ///
    /// # Errors
    ///
    /// `Internal` if the provider names a backend or domain the ABI has no code for.
    pub fn identity(&self) -> Result<(u32, u32), Error> {
        let backend = backend_code(self.provider.backend())
            .ok_or_else(|| internal("a backend the ABI has no code for".into()))?;
        let domain = domain_code(self.provider.domain())
            .ok_or_else(|| internal("a memory domain the ABI has no code for".into()))?;
        Ok((backend, domain))
    }

    /// The panic code that made the provider sticky, if one did.
    #[must_use]
    pub fn sticky(&self) -> Option<ErrorCode> {
        match self.sticky.load(Ordering::Acquire) {
            0 => None,
            code => Some(ErrorCode(code)),
        }
    }

    fn set_message(&self, message: String) {
        self.errors
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(thread::current().id(), message);
    }

    fn message(&self) -> String {
        self.errors
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&thread::current().id())
            .cloned()
            .unwrap_or_default()
    }
}

/// The calls that still run on a sticky provider: the runner needs `health` to see the state,
/// and `free` and `close_ctx` release what the module still holds.
const ALLOWED_WHEN_STICKY: [&str; 3] = ["health", "free", "close_ctx"];

fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic with a payload that is not text".to_owned())
}

/// Runs the thunk `name`'s body `f` under the guard of RFC-0040 §10.4. `inst` is `None` when the
/// thunk's `self` pointer was null.
pub fn call<'a, P, F>(inst: Option<&'a Instance<P>>, name: &'static str, f: F) -> SynStatus
where
    P: KernelProvider,
    F: FnOnce(&'a Instance<P>) -> R,
{
    let Some(inst) = inst else {
        return status_of_error(&internal(format!("`{name}` was called with a null self")));
    };
    if let Some(code) = inst.sticky()
        && !ALLOWED_WHEN_STICKY.contains(&name)
    {
        let error = Error {
            class: syntropy_core::ErrorClass::Device,
            code: ErrorCode::DEVICE_STICKY,
            message: format!(
                "`{name}` refused: the provider is sticky after code {}",
                code.0
            ),
        };
        inst.set_message(error.message.clone());
        return status_of_error(&error);
    }
    let error = match catch_unwind(AssertUnwindSafe(|| f(inst))) {
        Ok(Ok(())) => return SynStatus::OK,
        Ok(Err(error)) => error,
        Err(payload) => {
            inst.sticky
                .store(ErrorCode::PROVIDER_PANICKED.0, Ordering::Release);
            panicked(format!(
                "the provider panicked in `{name}`: {}",
                panic_text(&*payload)
            ))
        }
    };
    inst.set_message(error.message.clone());
    status_of_error(&error)
}

/// The `last_error` thunk's body: copies the calling thread's last message into `buf` and
/// reports its whole length.
pub fn last_error<P: KernelProvider>(inst: &Instance<P>, buf: &mut [u8], len: &mut usize) {
    let message = inst.message();
    *len = message.len();
    let n = message.len().min(buf.len());
    buf[..n].copy_from_slice(&message.as_bytes()[..n]);
}

/// A caller-supplied output buffer ([`SynOutBuf`](super::SynOutBuf)).
pub struct Out<'a> {
    /// The caller's buffer.
    pub buf: &'a mut [u8],
    /// Where to report the bytes needed.
    pub len: &'a mut usize,
}

impl Out<'_> {
    /// Reports `data.len()` as the bytes needed, and writes `data` if it fits.
    pub fn put(&mut self, data: &[u8]) {
        *self.len = data.len();
        if data.len() <= self.buf.len() {
            self.buf[..data.len()].copy_from_slice(data);
        }
    }
}

thread_local! {
    static CONTEXTS: RefCell<HashMap<u64, DeviceCtx>> = RefCell::new(HashMap::new());
}

static NEXT_CONTEXT: AtomicU64 = AtomicU64::new(1);

fn with_ctx<T>(ctx: u64, f: impl FnOnce(&DeviceCtx) -> Result<T, Error>) -> Result<T, Error> {
    CONTEXTS.with(|table| {
        let table = table.borrow();
        let ctx = table.get(&ctx).ok_or_else(|| {
            internal(format!(
                "context {ctx} is not open on this thread: a context is used on the thread \
                 that opened it"
            ))
        })?;
        f(ctx)
    })
}

fn unknown(what: &str) -> Error {
    internal(format!("a {what} the ABI has no code for"))
}

/// `build`.
pub fn build<P: KernelProvider>(i: &Instance<P>, mut out: Out<'_>) -> R {
    out.put(&to_json(&BuildWire::of(&i.provider.build()))?);
    Ok(())
}

/// `enumerate`.
pub fn enumerate<P: KernelProvider>(i: &Instance<P>, mut out: Out<'_>) -> R {
    let devices: Vec<ProviderDevice> = i.provider.enumerate()?;
    out.put(&to_json(&devices)?);
    Ok(())
}

/// `open`.
pub fn open<P: KernelProvider>(
    i: &Instance<P>,
    dev: &SynDeviceId,
    opts: &SynOpenOptions,
    ctx: &mut u64,
) -> R {
    let dev = dev
        .to_device_id()
        .ok_or_else(|| unknown("device ordinal"))?;
    let opts = opts.to_options().ok_or_else(|| unknown("power hint"))?;
    let opened = i.provider.open(dev, &opts)?;
    let handle = NEXT_CONTEXT.fetch_add(1, Ordering::Relaxed);
    CONTEXTS.with(|table| table.borrow_mut().insert(handle, opened));
    *ctx = handle;
    Ok(())
}

/// `close_ctx`.
pub fn close_ctx<P: KernelProvider>(_: &Instance<P>, ctx: u64) -> R {
    let removed = CONTEXTS.with(|table| table.borrow_mut().remove(&ctx));
    removed.map(drop).ok_or_else(|| {
        internal(format!(
            "context {ctx} is not open on this thread, so it was not closed"
        ))
    })
}

/// `memory_info`.
pub fn memory_info<P: KernelProvider>(i: &Instance<P>, ctx: u64, out: &mut SynMemoryInfo) -> R {
    *out = (&with_ctx(ctx, |c| i.provider.memory_info(c))?).into();
    Ok(())
}

/// `ops`.
pub fn ops<P: KernelProvider>(i: &Instance<P>, mut out: Out<'_>) -> R {
    let keys: Vec<&OpKey> = i.provider.ops().iter().collect();
    out.put(&to_json(&keys)?);
    Ok(())
}

/// `query`.
pub fn query<P: KernelProvider>(
    i: &Instance<P>,
    ctx: u64,
    op: &[u8],
    shape: &[u8],
    mut out: Out<'_>,
) -> R {
    let op: OpKey = from_json(op)?;
    let shape: ShapeSig = from_json(shape)?;
    match with_ctx(ctx, |c| Ok(i.provider.query(c, &op, &shape)))? {
        Some(caps) => out.put(&to_json(&CapsWire::of(&caps)?)?),
        None => out.put(&[]),
    }
    Ok(())
}

/// `alloc`.
pub fn alloc<P: KernelProvider>(
    i: &Instance<P>,
    ctx: u64,
    req: &SynAllocRequest,
    out: &mut SynBuffer,
) -> R {
    let req = req.to_request().ok_or_else(|| unknown("memory kind"))?;
    let buf = with_ctx(ctx, |c| i.provider.alloc(c, req))?;
    *out = SynBuffer::of(&buf).ok_or_else(|| unknown("memory kind"))?;
    Ok(())
}

/// `free`.
pub fn free<P: KernelProvider>(i: &Instance<P>, ctx: u64, buf: &SynBuffer) -> R {
    let buf = buf.to_buffer().ok_or_else(|| unknown("memory kind"))?;
    with_ctx(ctx, |c| {
        i.provider.free(c, buf);
        Ok(())
    })
}

/// `map_segment`.
pub fn map_segment<P: KernelProvider>(
    i: &Instance<P>,
    ctx: u64,
    reservation: &SynBuffer,
    offset: u64,
    bytes: u64,
    out: &mut SynSegment,
) -> R {
    let reservation = reservation
        .to_buffer()
        .ok_or_else(|| unknown("memory kind"))?;
    let seg = with_ctx(ctx, |c| {
        i.provider
            .map_segment(c, &reservation, Bytes(offset), Bytes(bytes))
    })?;
    *out = (&seg).into();
    Ok(())
}

/// `unmap_segment`.
pub fn unmap_segment<P: KernelProvider>(i: &Instance<P>, ctx: u64, seg: &SynSegment) -> R {
    with_ctx(ctx, |c| i.provider.unmap_segment(c, seg.into()))
}

/// `register_host`.
pub fn register_host<P: KernelProvider>(
    i: &Instance<P>,
    ctx: u64,
    region: SynHostRegion,
    out: &mut SynRegistration,
) -> R {
    let reg = with_ctx(ctx, |c| i.provider.register_host(c, region.into()))?;
    *out = (&reg).into();
    Ok(())
}

/// `import`.
pub fn import<P: KernelProvider>(
    i: &Instance<P>,
    ctx: u64,
    region: SynHostRegion,
    out: &mut SynBuffer,
) -> R {
    let buf = with_ctx(ctx, |c| i.provider.import(c, region.into()))?;
    *out = SynBuffer::of(&buf).ok_or_else(|| unknown("memory kind"))?;
    Ok(())
}

/// `set_resident`.
pub fn set_resident<P: KernelProvider>(
    i: &Instance<P>,
    ctx: u64,
    bufs: &[SynBuffer],
    resident: bool,
) -> R {
    let owned = bufs
        .iter()
        .map(|b| b.to_buffer().ok_or_else(|| unknown("memory kind")))
        .collect::<Result<SmallVec<[_; 16]>, _>>()?;
    let refs: SmallVec<[&_; 16]> = owned.iter().collect();
    with_ctx(ctx, |c| i.provider.set_resident(c, &refs, resident))
}

/// `repack`.
pub fn repack<P: KernelProvider>(
    i: &Instance<P>,
    op: &[u8],
    src: &[u8],
    dst: &mut [u8],
    present: &mut u32,
    layout: &mut u32,
) -> R {
    let op: OpKey = from_json(op)?;
    match i.provider.repack(&op, src, dst) {
        None => *present = 0,
        Some(result) => {
            *layout = u32::from(result?.0);
            *present = 1;
        }
    }
    Ok(())
}

/// `repacked_len`.
pub fn repacked_len<P: KernelProvider>(
    i: &Instance<P>,
    op: &[u8],
    src: u64,
    present: &mut u32,
    out: &mut u64,
) -> R {
    let op: OpKey = from_json(op)?;
    match i.provider.repacked_len(&op, Bytes(src)) {
        None => *present = 0,
        Some(len) => {
            *out = len.0;
            *present = 1;
        }
    }
    Ok(())
}

/// `stream`.
pub fn stream<P: KernelProvider>(i: &Instance<P>, ctx: u64, prio: u32, out: &mut u64) -> R {
    let prio = priority_of_code(prio).ok_or_else(|| unknown("stream priority"))?;
    *out = with_ctx(ctx, |c| i.provider.stream(c, prio))?.handle;
    Ok(())
}

/// `event`.
pub fn event<P: KernelProvider>(i: &Instance<P>, ctx: u64, out: &mut u64) -> R {
    *out = with_ctx(ctx, |c| i.provider.event(c))?.handle;
    Ok(())
}

/// `record`.
pub fn record<P: KernelProvider>(i: &Instance<P>, ctx: u64, s: u64, e: u64) -> R {
    with_ctx(ctx, |c| {
        i.provider
            .record(c, &Stream { handle: s }, &Event { handle: e })
    })
}

/// `wait`.
pub fn wait<P: KernelProvider>(i: &Instance<P>, ctx: u64, s: u64, e: u64) -> R {
    with_ctx(ctx, |c| {
        i.provider
            .wait(c, &Stream { handle: s }, &Event { handle: e })
    })
}

fn host_flag(flag: u32) -> HostFlag {
    // The two halves are 16 bits each by construction.
    HostFlag {
        page: (flag >> 16) as u16,
        slot: (flag & 0xFFFF) as u16,
    }
}

/// `signal_host`.
pub fn signal_host<P: KernelProvider>(
    i: &Instance<P>,
    ctx: u64,
    s: u64,
    flag: u32,
    value: u32,
) -> R {
    with_ctx(ctx, |c| {
        i.provider
            .signal_host(c, &Stream { handle: s }, host_flag(flag), value)
    })
}

/// `wait_host`.
pub fn wait_host<P: KernelProvider>(
    i: &Instance<P>,
    ctx: u64,
    s: u64,
    (flag, value, cmp): (u32, u32, u32),
    reset: Option<u32>,
) -> R {
    let cmp = cmp_of_code(cmp).ok_or_else(|| unknown("comparison"))?;
    with_ctx(ctx, |c| {
        i.provider
            .wait_host(c, &Stream { handle: s }, host_flag(flag), value, cmp, reset)
    })
}

/// `arm_wake`.
pub fn arm_wake<P: KernelProvider>(i: &Instance<P>, ctx: u64, s: u64, kind: u32, handle: u64) -> R {
    let kind = wake_of_code(kind).ok_or_else(|| unknown("wake kind"))?;
    with_ctx(ctx, |c| {
        i.provider
            .arm_wake(c, &Stream { handle: s }, &HostWaker { kind, handle })
    })
}

/// `poll`.
pub fn poll<P: KernelProvider>(i: &Instance<P>, ctx: u64, e: u64, done: &mut u32) -> R {
    *done = u32::from(with_ctx(ctx, |c| i.provider.poll(c, &Event { handle: e }))?);
    Ok(())
}

/// `elapsed`.
pub fn elapsed<P: KernelProvider>(
    i: &Instance<P>,
    ctx: u64,
    start: u64,
    end: u64,
    present: &mut u32,
    nanos: &mut u64,
) -> R {
    let result = with_ctx(ctx, |c| {
        Ok(i.provider
            .elapsed(c, &Event { handle: start }, &Event { handle: end }))
    })?;
    *present = u32::from(result.is_some());
    *nanos = result.map_or(0, |n| n.0);
    Ok(())
}

/// `synchronize`.
pub fn synchronize<P: KernelProvider>(i: &Instance<P>, ctx: u64, s: u64) -> R {
    with_ctx(ctx, |c| i.provider.synchronize(c, &Stream { handle: s }))
}

/// `copy`.
pub fn copy<P: KernelProvider>(
    i: &Instance<P>,
    ctx: u64,
    s: u64,
    src: SynSlice,
    dst: SynSlice,
) -> R {
    with_ctx(ctx, |c| {
        i.provider
            .copy(c, &Stream { handle: s }, src.into(), dst.into())
    })
}

/// `copy_batch`.
pub fn copy_batch<P: KernelProvider>(
    i: &Instance<P>,
    ctx: u64,
    s: u64,
    pairs: &[SynSlicePair],
) -> R {
    let pairs: SmallVec<[(BufferSlice, BufferSlice); 16]> =
        pairs.iter().map(|p| (p.src.into(), p.dst.into())).collect();
    with_ctx(ctx, |c| {
        i.provider.copy_batch(c, &Stream { handle: s }, &pairs)
    })
}

/// `prepare`.
pub fn prepare<P: KernelProvider>(
    i: &Instance<P>,
    ctx: u64,
    op: &[u8],
    shape: &[u8],
    out: &mut SynPreparedOp,
) -> R {
    let op: OpKey = from_json(op)?;
    let shape: ShapeSig = from_json(shape)?;
    let prepared = with_ctx(ctx, |c| i.provider.prepare(c, &op, &shape))?;
    *out = SynPreparedOp {
        workspace: prepared.workspace.0,
        handle: prepared.handle,
    };
    Ok(())
}

/// The slices of a [`SynOpArgs`], as the thunk read them.
pub struct ArgSlices<'a> {
    /// `inputs`.
    pub inputs: &'a [SynTensorArg],
    /// `outputs`.
    pub outputs: &'a [SynTensorArg],
    /// `weights`.
    pub weights: &'a [super::types::SynBufferView],
    /// `statics`.
    pub statics: &'a [super::types::SynScalar],
}

/// `launch`.
pub fn launch<P: KernelProvider>(
    i: &Instance<P>,
    ctx: u64,
    s: u64,
    prepared: &SynPreparedOp,
    args: &SynOpArgs,
    slices: &ArgSlices<'_>,
) -> R {
    let tensors = |list: &[SynTensorArg]| {
        list.iter()
            .map(|t| t.to_tensor().ok_or_else(|| unknown("dtype")))
            .collect::<Result<SmallVec<[TensorArg; 8]>, _>>()
    };
    let inputs = tensors(slices.inputs)?;
    let outputs = tensors(slices.outputs)?;
    let weights = slices
        .weights
        .iter()
        .map(|w| w.to_view().ok_or_else(|| unknown("buffer view")))
        .collect::<Result<SmallVec<[BufferView; 8]>, _>>()?;
    let statics = slices
        .statics
        .iter()
        .map(|v| v.to_scalar().ok_or_else(|| unknown("scalar")))
        .collect::<Result<SmallVec<[Scalar; 8]>, _>>()?;
    let op_args = OpArgs {
        inputs: &inputs,
        outputs: &outputs,
        weights: &weights,
        step: args.step.into(),
        workspace: (args.has_workspace != 0).then(|| args.workspace.into()),
        statics: &statics,
    };
    let mut prepared_op = crate::op::PreparedOp::new(Bytes(prepared.workspace));
    prepared_op.handle = prepared.handle;
    with_ctx(ctx, |c| {
        i.provider
            .launch(c, &Stream { handle: s }, &prepared_op, &op_args)
    })
}

/// `begin_capture`.
pub fn begin_capture<P: KernelProvider>(i: &Instance<P>, ctx: u64, s: u64) -> R {
    with_ctx(ctx, |c| i.provider.begin_capture(c, &Stream { handle: s }))
}

/// `end_capture`.
pub fn end_capture<P: KernelProvider>(i: &Instance<P>, ctx: u64, s: u64, out: &mut u64) -> R {
    *out = with_ctx(ctx, |c| i.provider.end_capture(c, &Stream { handle: s }))?.handle;
    Ok(())
}

/// `replay`.
pub fn replay<P: KernelProvider>(i: &Instance<P>, ctx: u64, s: u64, g: u64) -> R {
    with_ctx(ctx, |c| {
        i.provider
            .replay(c, &Stream { handle: s }, &CapturedGraph { handle: g })
    })
}

/// `health`. A sticky provider reports `Sticky` without calling the provider.
pub fn health<P: KernelProvider>(i: &Instance<P>, ctx: u64, out: &mut SynHealth) -> R {
    let health = match i.sticky() {
        Some(code) => DeviceHealth::Sticky { code },
        None => with_ctx(ctx, |c| Ok(i.provider.health(c)))?,
    };
    *out = SynHealth::of(health).ok_or_else(|| unknown("health"))?;
    Ok(())
}

/// A registrar and the instance it reports through.
pub struct RegistrarState<P: 'static> {
    /// The provider instance, for the sticky guard and the error store.
    pub inst: &'static Instance<P>,
    /// The provider's registrar.
    pub registrar: Box<dyn HostRegistrar>,
}

/// Registrar `register`.
pub fn registrar_register<P: KernelProvider>(
    st: &RegistrarState<P>,
    region: SynHostRegion,
    flags: u32,
    out: &mut SynRegistration,
) -> R {
    let flags = HostRegFlags::from_bits(u8::try_from(flags).map_err(|_| unknown("flag"))?)
        .ok_or_else(|| unknown("flag"))?;
    *out = (&st.registrar.register(region.into(), flags)?).into();
    Ok(())
}

/// Registrar `unregister`.
pub fn registrar_unregister<P: KernelProvider>(st: &RegistrarState<P>, reg: &SynRegistration) -> R {
    st.registrar.unregister(reg.into())
}

/// `host_registrar`: the state to hand out, or `None`.
pub fn host_registrar<P: KernelProvider>(
    i: &'static Instance<P>,
    ctx: u64,
) -> Result<Option<RegistrarState<P>>, Error> {
    let registrar = with_ctx(ctx, |c| Ok(i.provider.host_registrar(c)))?;
    Ok(registrar.map(|registrar| RegistrarState { inst: i, registrar }))
}
