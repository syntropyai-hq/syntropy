// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `#[repr(C)]` tables and records of the provider-library ABI (RFC-0040 §10.4).
//!
//! Field order is the ABI. A minor version may only append fields to a struct and entries to a
//! vtable, and `struct_size` tells each side what the other built. Every vtable function takes
//! the vtable's `self` pointer first.

use std::ffi::{c_char, c_void};

use super::status::SynStatus;

/// The identity both sides state (RFC-0040 §10.4).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SynHeader {
    /// [`MAGIC`](super::MAGIC).
    pub magic: [u8; 8],
    /// [`ABI_MAJOR`](super::ABI_MAJOR) as the builder knew it.
    pub abi_major: u32,
    /// [`ABI_MINOR`](super::ABI_MINOR) as the builder knew it.
    pub abi_minor: u32,
    /// The build ID of the binary that built the table (ADR-0507 rule 3).
    pub build_id: [u8; 32],
    /// Size of the struct this header starts, as the builder built it.
    pub struct_size: u32,
}

impl SynHeader {
    /// A header for a struct of `struct_size` bytes with this build's ABI version.
    #[must_use]
    pub const fn new(build_id: [u8; 32], struct_size: usize) -> Self {
        Self {
            magic: super::MAGIC,
            abi_major: super::ABI_MAJOR,
            abi_minor: super::ABI_MINOR,
            build_id,
            // A table is a few hundred bytes; the cast cannot truncate.
            struct_size: struct_size as u32,
        }
    }
}

/// What the core offers a module (RFC-0040 §10.4).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SynHostApi {
    /// The core's identity, checked by the module too.
    pub h: SynHeader,
    /// Logs `len` bytes of UTF-8 `msg` at `level` (0 error … 4 trace) under the NUL-terminated
    /// `target`.
    pub log: extern "C" fn(level: u8, target: *const c_char, msg: *const c_char, len: usize),
    /// Adds `value` to the metric `id`.
    pub metric: extern "C" fn(id: u32, value: u64),
    /// The value of the NUL-terminated config `key` (`features.disable`, `kernels.*`), as a
    /// NUL-terminated string the core keeps alive, or null when the key has no value.
    pub config: extern "C" fn(key: *const c_char) -> *const c_char,
}

/// A device identity: `DeviceId` as plain bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SynDeviceId {
    /// The node ID.
    pub node: [u8; 32],
    /// The position in the node's device order.
    pub ordinal: u32,
}

/// `OpenOptions`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SynOpenOptions {
    /// Streams per priority: compute, copy-in, copy-out, collective.
    pub streams: [u8; 4],
    /// Doorbell slots.
    pub doorbell_slots: u32,
    /// 1 when graph capture is wanted.
    pub capture: u8,
    /// 1 when deterministic kernels are wanted.
    pub deterministic: u8,
    /// 0 for mains power, 1 for battery.
    pub power: u8,
    /// Zero.
    pub reserved: u8,
}

/// `DeviceMemoryInfo`. `flags` bit 0 says `budget` is set, bit 1 says `host_visible` is set,
/// bit 2 says the memory is unified.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SynMemoryInfo {
    /// Total bytes.
    pub total: u64,
    /// Free bytes.
    pub free: u64,
    /// The OS budget, when `flags` bit 0 is set.
    pub budget: u64,
    /// Host-visible device-local bytes, when `flags` bit 1 is set.
    pub host_visible: u64,
    /// The largest single allocation.
    pub max_alloc: u64,
    /// The flag bits above.
    pub flags: u32,
    /// Zero.
    pub reserved: u32,
}

/// `AllocRequest`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SynAllocRequest {
    /// Bytes wanted.
    pub bytes: u64,
    /// Alignment in bytes.
    pub align: u64,
    /// `MemKind`: 0 device, 1 host-mapped, 2 unified.
    pub kind: u32,
    /// 1 to reserve address space only.
    pub reserve_only: u32,
}

/// `DeviceBuffer`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SynBuffer {
    /// The device address.
    pub addr: u64,
    /// The length.
    pub bytes: u64,
    /// The provider's handle.
    pub handle: u64,
    /// `MemKind` code.
    pub kind: u32,
    /// Zero.
    pub reserved: u32,
}

/// `BufferSlice`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SynSlice {
    /// The buffer handle.
    pub buffer: u64,
    /// Offset in bytes.
    pub offset: u64,
    /// Length in bytes.
    pub bytes: u64,
}

/// A `(src, dst)` pair for `copy_batch`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SynSlicePair {
    /// The source.
    pub src: SynSlice,
    /// The destination.
    pub dst: SynSlice,
}

/// `Segment`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SynSegment {
    /// The reservation's handle.
    pub reservation: u64,
    /// Offset in the reservation.
    pub offset: u64,
    /// Length.
    pub bytes: u64,
    /// The provider's handle.
    pub handle: u64,
}

/// `HostRegion`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SynHostRegion {
    /// Address of the first byte.
    pub addr: u64,
    /// Length.
    pub len: u64,
}

/// `Registration`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SynRegistration {
    /// The registered region.
    pub region: SynHostRegion,
    /// The provider's handle.
    pub handle: u64,
}

/// `DeviceHealth`: `kind` 0 is `Ok`, 1 is `Throttled` (`detail` is the reason: 0 thermal, 1
/// power, 2 other), 2 is `Sticky` (`detail` is the `ErrorCode`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SynHealth {
    /// The variant.
    pub kind: u32,
    /// The variant's payload.
    pub detail: u32,
}

/// `PreparedOp`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SynPreparedOp {
    /// Workspace bytes.
    pub workspace: u64,
    /// The provider's handle.
    pub handle: u64,
}

/// `TensorArg`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SynTensorArg {
    /// The bytes it covers.
    pub buf: SynSlice,
    /// `DType` code ([`wire::dtype_code`](super::wire::dtype_code)).
    pub dtype: u32,
    /// Zero.
    pub reserved: u32,
    /// Extents.
    pub dims: [u32; 4],
    /// Strides in elements.
    pub strides: [u32; 4],
}

/// `BufferView`: `kind` 0 is host, 1 is one device's memory (`node` and `ordinal`), 2 is
/// unified.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SynBufferView {
    /// The address space.
    pub kind: u32,
    /// The device's ordinal, for kind 1.
    pub ordinal: u32,
    /// The device's node, for kind 1.
    pub node: [u8; 32],
    /// The start address.
    pub addr: u64,
    /// The length.
    pub len: u64,
}

/// `Scalar`: `tag` 0 is `F32`, 1 is `I32`, 2 is `U32`; `bits` holds the value's bits.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SynScalar {
    /// The variant.
    pub tag: u32,
    /// The value's bits.
    pub bits: u32,
}

/// `OpArgs`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SynOpArgs {
    /// Size of this struct as its builder built it.
    pub struct_size: u32,
    /// 1 when `workspace` is set.
    pub has_workspace: u32,
    /// Input tensors.
    pub inputs: *const SynTensorArg,
    /// Their count.
    pub n_inputs: usize,
    /// Output tensors.
    pub outputs: *const SynTensorArg,
    /// Their count.
    pub n_outputs: usize,
    /// Weight views.
    pub weights: *const SynBufferView,
    /// Their count.
    pub n_weights: usize,
    /// The step descriptor's slice.
    pub step: SynSlice,
    /// The workspace, when `has_workspace` is 1.
    pub workspace: SynSlice,
    /// Static scalars.
    pub statics: *const SynScalar,
    /// Their count.
    pub n_statics: usize,
}

/// A buffer the caller supplies for data the callee produces. The callee sets `len` to the bytes
/// it needs and writes them only if `len <= cap`; the caller retries with a larger buffer
/// otherwise. A `len` of 0 on return means "no value".
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SynOutBuf {
    /// Where to write.
    pub ptr: *mut u8,
    /// Capacity in bytes.
    pub cap: usize,
    /// Bytes needed, set by the callee.
    pub len: usize,
}

/// A host-registrar's own small vtable: a `HostRegistrar` is `Send`, so it does not share the
/// provider's context-bound calls (RFC-0040 §10.4).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SynRegistrarVtbl {
    /// Size of this struct as the module built it; 0 means the provider has no registrar.
    pub struct_size: u32,
    /// `register(region, flags)`.
    pub register: extern "C" fn(
        slf: *mut c_void,
        region: SynHostRegion,
        flags: u32,
        out: *mut SynRegistration,
    ) -> SynStatus,
    /// `unregister(reg)`.
    pub unregister: extern "C" fn(slf: *mut c_void, reg: *const SynRegistration) -> SynStatus,
    /// Frees the registrar. Called once, by the side that holds it last. A destructor that
    /// panics makes the provider sticky, and the status says so.
    pub close: extern "C" fn(slf: *mut c_void) -> SynStatus,
    /// The registrar's state.
    pub slf: *mut c_void,
}

/// One provider's function table: one entry per `KernelProvider` method, in declaration order
/// (RFC-0040 §10.4). `ctx` is a handle from `open`; it is valid only on the thread that opened
/// it.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SynProviderVtbl {
    /// Size of this table as the module built it.
    pub struct_size: u32,
    /// `ProviderId`, NUL-terminated, kept alive by the module.
    pub id: *const c_char,
    /// `Backend` code ([`wire::backend_code`](super::wire::backend_code)).
    pub backend: u32,
    /// `MemoryDomain` code ([`wire::domain_code`](super::wire::domain_code)).
    pub domain: u32,
    /// `build()`: `ProviderBuild` as JSON.
    pub build: extern "C" fn(slf: *mut c_void, out: *mut SynOutBuf) -> SynStatus,
    /// `enumerate()`: a JSON array of `ProviderDevice`.
    pub enumerate: extern "C" fn(slf: *mut c_void, out: *mut SynOutBuf) -> SynStatus,
    /// `open(dev, opts)`: sets `ctx`.
    pub open: extern "C" fn(
        slf: *mut c_void,
        dev: *const SynDeviceId,
        opts: *const SynOpenOptions,
        ctx: *mut u64,
    ) -> SynStatus,
    /// Drops the context `ctx` (the core's `DeviceCtx` was dropped).
    pub close_ctx: extern "C" fn(slf: *mut c_void, ctx: u64) -> SynStatus,
    /// `memory_info(ctx)`.
    pub memory_info:
        extern "C" fn(slf: *mut c_void, ctx: u64, out: *mut SynMemoryInfo) -> SynStatus,
    /// `ops()`: a JSON array of `OpKey`.
    pub ops: extern "C" fn(slf: *mut c_void, out: *mut SynOutBuf) -> SynStatus,
    /// `query(ctx, op, shape)`: `OpCaps` as JSON, with `len` 0 for `None`. `op` and `shape` are
    /// JSON.
    pub query: extern "C" fn(
        slf: *mut c_void,
        ctx: u64,
        op: *const u8,
        op_len: usize,
        shape: *const u8,
        shape_len: usize,
        out: *mut SynOutBuf,
    ) -> SynStatus,
    /// `alloc(ctx, req)`.
    pub alloc: extern "C" fn(
        slf: *mut c_void,
        ctx: u64,
        req: *const SynAllocRequest,
        out: *mut SynBuffer,
    ) -> SynStatus,
    /// `free(ctx, buf)`.
    pub free: extern "C" fn(slf: *mut c_void, ctx: u64, buf: *const SynBuffer) -> SynStatus,
    /// `map_segment(ctx, reservation, offset, bytes)`.
    pub map_segment: extern "C" fn(
        slf: *mut c_void,
        ctx: u64,
        reservation: *const SynBuffer,
        offset: u64,
        bytes: u64,
        out: *mut SynSegment,
    ) -> SynStatus,
    /// `unmap_segment(ctx, seg)`.
    pub unmap_segment:
        extern "C" fn(slf: *mut c_void, ctx: u64, seg: *const SynSegment) -> SynStatus,
    /// `register_host(ctx, region)`.
    pub register_host: extern "C" fn(
        slf: *mut c_void,
        ctx: u64,
        region: SynHostRegion,
        out: *mut SynRegistration,
    ) -> SynStatus,
    /// `host_registrar(ctx)`: `struct_size` 0 in `out` means `None`.
    pub host_registrar:
        extern "C" fn(slf: *mut c_void, ctx: u64, out: *mut SynRegistrarVtbl) -> SynStatus,
    /// `import(ctx, region)`.
    pub import: extern "C" fn(
        slf: *mut c_void,
        ctx: u64,
        region: SynHostRegion,
        out: *mut SynBuffer,
    ) -> SynStatus,
    /// `set_resident(ctx, bufs, resident)`.
    pub set_resident: extern "C" fn(
        slf: *mut c_void,
        ctx: u64,
        bufs: *const SynBuffer,
        n: usize,
        resident: u32,
    ) -> SynStatus,
    /// `repack(op, src, dst)`: `present` is 0 for `None`, else `layout` is the layout version.
    pub repack: extern "C" fn(
        slf: *mut c_void,
        op: *const u8,
        op_len: usize,
        src: *const u8,
        src_len: usize,
        dst: *mut u8,
        dst_len: usize,
        present: *mut u32,
        layout: *mut u32,
    ) -> SynStatus,
    /// `repacked_len(op, src)`: `present` is 0 for `None`.
    pub repacked_len: extern "C" fn(
        slf: *mut c_void,
        op: *const u8,
        op_len: usize,
        src: u64,
        present: *mut u32,
        out: *mut u64,
    ) -> SynStatus,
    /// `stream(ctx, prio)`: `prio` is the `StreamPriority` code.
    pub stream: extern "C" fn(slf: *mut c_void, ctx: u64, prio: u32, out: *mut u64) -> SynStatus,
    /// `event(ctx)`.
    pub event: extern "C" fn(slf: *mut c_void, ctx: u64, out: *mut u64) -> SynStatus,
    /// `record(ctx, s, e)`.
    pub record: extern "C" fn(slf: *mut c_void, ctx: u64, s: u64, e: u64) -> SynStatus,
    /// `wait(ctx, s, e)`.
    pub wait: extern "C" fn(slf: *mut c_void, ctx: u64, s: u64, e: u64) -> SynStatus,
    /// `signal_host(ctx, s, flag, value)`: `flag` is `page << 16 | slot`.
    pub signal_host:
        extern "C" fn(slf: *mut c_void, ctx: u64, s: u64, flag: u32, value: u32) -> SynStatus,
    /// `wait_host(ctx, s, flag, value, cmp, reset)`: `cmp` is 0 for `Eq`, 1 for `Geq`; `reset` is
    /// read only when `has_reset` is 1.
    pub wait_host: extern "C" fn(
        slf: *mut c_void,
        ctx: u64,
        s: u64,
        flag: u32,
        value: u32,
        cmp: u32,
        has_reset: u32,
        reset: u32,
    ) -> SynStatus,
    /// `arm_wake(ctx, s, waker)`: `kind` is the `WakeKind` code.
    pub arm_wake:
        extern "C" fn(slf: *mut c_void, ctx: u64, s: u64, kind: u32, handle: u64) -> SynStatus,
    /// `poll(ctx, e)`: `done` is 0 or 1.
    pub poll: extern "C" fn(slf: *mut c_void, ctx: u64, e: u64, done: *mut u32) -> SynStatus,
    /// `elapsed(ctx, start, end)`: `present` is 0 for `None`.
    pub elapsed: extern "C" fn(
        slf: *mut c_void,
        ctx: u64,
        start: u64,
        end: u64,
        present: *mut u32,
        nanos: *mut u64,
    ) -> SynStatus,
    /// `synchronize(ctx, s)`.
    pub synchronize: extern "C" fn(slf: *mut c_void, ctx: u64, s: u64) -> SynStatus,
    /// `copy(ctx, s, src, dst)`.
    pub copy: extern "C" fn(
        slf: *mut c_void,
        ctx: u64,
        s: u64,
        src: *const SynSlice,
        dst: *const SynSlice,
    ) -> SynStatus,
    /// `copy_batch(ctx, s, pairs)`.
    pub copy_batch: extern "C" fn(
        slf: *mut c_void,
        ctx: u64,
        s: u64,
        pairs: *const SynSlicePair,
        n: usize,
    ) -> SynStatus,
    /// `prepare(ctx, op, shape)`: `op` and `shape` are JSON.
    pub prepare: extern "C" fn(
        slf: *mut c_void,
        ctx: u64,
        op: *const u8,
        op_len: usize,
        shape: *const u8,
        shape_len: usize,
        out: *mut SynPreparedOp,
    ) -> SynStatus,
    /// `launch(ctx, s, op, args)`.
    pub launch: extern "C" fn(
        slf: *mut c_void,
        ctx: u64,
        s: u64,
        op: *const SynPreparedOp,
        args: *const SynOpArgs,
    ) -> SynStatus,
    /// `begin_capture(ctx, s)`.
    pub begin_capture: extern "C" fn(slf: *mut c_void, ctx: u64, s: u64) -> SynStatus,
    /// `end_capture(ctx, s)`.
    pub end_capture: extern "C" fn(slf: *mut c_void, ctx: u64, s: u64, out: *mut u64) -> SynStatus,
    /// `replay(ctx, s, g)`.
    pub replay: extern "C" fn(slf: *mut c_void, ctx: u64, s: u64, g: u64) -> SynStatus,
    /// `health(ctx)`. A provider whose thunk caught a panic reports `Sticky`.
    pub health: extern "C" fn(slf: *mut c_void, ctx: u64, out: *mut SynHealth) -> SynStatus,
    /// The message of the last failure on the calling thread, written into `buf` (`cap` bytes);
    /// `len` is the bytes of the whole message.
    pub last_error: extern "C" fn(
        slf: *mut c_void,
        ctx: u64,
        buf: *mut u8,
        cap: usize,
        len: *mut usize,
    ) -> SynStatus,
    /// The provider's state.
    pub slf: *mut c_void,
}

/// What `syntropy_provider_entry_v1` returns.
#[repr(C)]
pub struct SynModule {
    /// The module's identity.
    pub h: SynHeader,
    /// The number of providers.
    pub n_providers: u32,
    /// One table per provider.
    pub providers: *const SynProviderVtbl,
    /// Called once before the library is unloaded.
    pub shutdown: extern "C" fn() -> SynStatus,
}
