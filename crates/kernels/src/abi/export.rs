// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `export!` macro: the module side of the ABI, expanded in a provider's own crate.

/// Emits the `extern "C"` thunks of one or more providers and the single exported entry symbol,
/// `syntropy_provider_entry_v1` (RFC-0040 §10.4; ADR-0400 decision 4).
///
/// Call it once, at the root of a provider-library `cdylib`:
///
/// ```ignore
/// syntropy_kernels::abi::export! {
///     // The build ID of this build, shared with the core (ADR-0507 rule 3).
///     build_id: BUILD_ID,
///     // `Type = constructor`, where the constructor is `fn(&HostApi) -> Type`.
///     providers: [CudaProvider = CudaProvider::new, GgmlCuda = GgmlCuda::new],
/// }
/// ```
///
/// What it emits, and the rules the thunks keep:
///
/// - Every thunk runs its body under `catch_unwind` ([`guest::call`](crate::abi::guest::call)).
///   A caught panic returns `Internal` ([`ErrorCode::PROVIDER_PANICKED`](crate::ErrorCode)) and
///   makes the provider report `DeviceHealth::Sticky`.
/// - Only plain `#[repr(C)]` records, pointers with lengths and `u64` handles cross. The module
///   allocates nothing the core frees: results go into buffers the core supplies.
/// - A context handle works on the thread that opened it only.
/// - The entry symbol checks the core's header (magic, ABI major, struct size). It does not
///   compare build IDs: the core's load check does, and it can then relax that one check for a
///   developer build (RFC-0040 §10.4 check 5).
///
/// The macro holds all the `unsafe` the module side needs: each raw pointer becomes a reference
/// in one of the helper functions at the top of the expansion, and every block states why the
/// ABI contract makes it valid. The crate that calls the macro must allow `unsafe_code`
/// (RFC-0001 §4.2 rule 8 lets every `kernels-*` crate). The link step must export the entry
/// symbol and nothing else: a version script on Linux and a `.def` file on Windows (see the
/// crate `syntropy-kernels-loader`'s `build.rs`, which does it for the test provider).
#[macro_export]
macro_rules! __abi_export {
    (
        build_id: $build_id:expr,
        providers: [ $( $ty:ty = $ctor:expr ),+ $(,)? ] $(,)?
    ) => {
        #[allow(unsafe_code, clippy::missing_safety_doc, clippy::too_many_arguments)]
        const _: () = {
            use ::std::ffi::{c_char, c_void};
            use $crate::abi::guest::{self as g, Instance};
            use $crate::abi::types::{
                SynAllocRequest, SynBuffer, SynDeviceId, SynHeader, SynHealth, SynHostApi,
                SynHostRegion, SynMemoryInfo, SynModule, SynOpArgs, SynOpenOptions, SynOutBuf,
                SynPreparedOp, SynProviderVtbl, SynRegistrarVtbl, SynRegistration, SynSegment,
                SynSlice, SynSlicePair,
            };
            use $crate::abi::{SynStatus, status_of_error};
            use $crate::{Error, KernelProvider};

            fn null_pointer() -> Error {
                $crate::abi::wire::internal("a null pointer crossed the provider boundary".into())
            }

            fn rd<'a, T>(p: *const T) -> Result<&'a T, Error> {
                if p.is_null() {
                    return Err(null_pointer());
                }
                // SAFETY: the ABI contract (RFC-0040 §10.4) makes the core pass a pointer to a
                // live `T` that it does not mutate for the duration of the call; null is checked.
                Ok(unsafe { &*p })
            }

            fn wr<'a, T>(p: *mut T) -> Result<&'a mut T, Error> {
                if p.is_null() {
                    return Err(null_pointer());
                }
                // SAFETY: the ABI contract makes the core pass a pointer to a live `T` that
                // nothing else accesses during the call; null is checked.
                Ok(unsafe { &mut *p })
            }

            fn items<'a, T>(p: *const T, n: usize) -> Result<&'a [T], Error> {
                if n == 0 {
                    return Ok(&[]);
                }
                if p.is_null() {
                    return Err(null_pointer());
                }
                // SAFETY: the ABI contract makes the core pass `n` initialised `T`s at `p` that
                // it does not mutate during the call; null and the empty case are handled.
                Ok(unsafe { ::std::slice::from_raw_parts(p, n) })
            }

            fn items_mut<'a>(p: *mut u8, n: usize) -> Result<&'a mut [u8], Error> {
                if n == 0 {
                    return Ok(&mut []);
                }
                if p.is_null() {
                    return Err(null_pointer());
                }
                // SAFETY: the ABI contract makes the core pass `n` writable bytes at `p` that
                // nothing else accesses during the call; null and the empty case are handled.
                Ok(unsafe { ::std::slice::from_raw_parts_mut(p, n) })
            }

            fn outbuf<'a>(p: *mut SynOutBuf) -> Result<g::Out<'a>, Error> {
                let o = wr(p)?;
                let buf = items_mut(o.ptr, o.cap)?;
                Ok(g::Out { buf, len: &mut o.len })
            }

            fn inst<'a, P>(p: *mut c_void) -> Option<&'a Instance<P>> {
                if p.is_null() {
                    return None;
                }
                // SAFETY: `p` is the `slf` of a vtable built in `build_module`, which points at
                // an `Instance<P>` leaked with `Box::leak`, so it is valid for `'static`. The
                // thunk for provider type `P` is only ever in that provider's vtable.
                Some(unsafe { &*p.cast::<Instance<P>>() })
            }

            fn cstr(p: *const c_char) -> Option<String> {
                if p.is_null() {
                    return None;
                }
                // SAFETY: the ABI contract makes `config` return a NUL-terminated string that
                // the core keeps alive for the call.
                let text = unsafe { ::std::ffi::CStr::from_ptr(p) };
                Some(text.to_string_lossy().into_owned())
            }

            // Thunks. Each one is `g::call` around one function of `$crate::abi::guest`.

            extern "C" fn build<P: KernelProvider>(slf: *mut c_void, out: *mut SynOutBuf) -> SynStatus {
                g::call(inst::<P>(slf), "build", |i| g::build(i, outbuf(out)?))
            }

            extern "C" fn enumerate<P: KernelProvider>(slf: *mut c_void, out: *mut SynOutBuf) -> SynStatus {
                g::call(inst::<P>(slf), "enumerate", |i| g::enumerate(i, outbuf(out)?))
            }

            extern "C" fn open<P: KernelProvider>(
                slf: *mut c_void,
                dev: *const SynDeviceId,
                opts: *const SynOpenOptions,
                ctx: *mut u64,
            ) -> SynStatus {
                g::call(inst::<P>(slf), "open", |i| g::open(i, rd(dev)?, rd(opts)?, wr(ctx)?))
            }

            extern "C" fn close_ctx<P: KernelProvider>(slf: *mut c_void, ctx: u64) -> SynStatus {
                g::call(inst::<P>(slf), "close_ctx", |i| g::close_ctx(i, ctx))
            }

            extern "C" fn memory_info<P: KernelProvider>(slf: *mut c_void, ctx: u64, out: *mut SynMemoryInfo) -> SynStatus {
                g::call(inst::<P>(slf), "memory_info", |i| g::memory_info(i, ctx, wr(out)?))
            }

            extern "C" fn ops<P: KernelProvider>(slf: *mut c_void, out: *mut SynOutBuf) -> SynStatus {
                g::call(inst::<P>(slf), "ops", |i| g::ops(i, outbuf(out)?))
            }

            extern "C" fn query<P: KernelProvider>(
                slf: *mut c_void,
                ctx: u64,
                op: *const u8,
                op_len: usize,
                shape: *const u8,
                shape_len: usize,
                out: *mut SynOutBuf,
            ) -> SynStatus {
                g::call(inst::<P>(slf), "query", |i| {
                    g::query(i, ctx, items(op, op_len)?, items(shape, shape_len)?, outbuf(out)?)
                })
            }

            extern "C" fn alloc<P: KernelProvider>(slf: *mut c_void, ctx: u64, req: *const SynAllocRequest, out: *mut SynBuffer) -> SynStatus {
                g::call(inst::<P>(slf), "alloc", |i| g::alloc(i, ctx, rd(req)?, wr(out)?))
            }

            extern "C" fn free<P: KernelProvider>(slf: *mut c_void, ctx: u64, buf: *const SynBuffer) -> SynStatus {
                g::call(inst::<P>(slf), "free", |i| g::free(i, ctx, rd(buf)?))
            }

            extern "C" fn map_segment<P: KernelProvider>(
                slf: *mut c_void,
                ctx: u64,
                reservation: *const SynBuffer,
                offset: u64,
                bytes: u64,
                out: *mut SynSegment,
            ) -> SynStatus {
                g::call(inst::<P>(slf), "map_segment", |i| {
                    g::map_segment(i, ctx, rd(reservation)?, offset, bytes, wr(out)?)
                })
            }

            extern "C" fn unmap_segment<P: KernelProvider>(slf: *mut c_void, ctx: u64, seg: *const SynSegment) -> SynStatus {
                g::call(inst::<P>(slf), "unmap_segment", |i| g::unmap_segment(i, ctx, rd(seg)?))
            }

            extern "C" fn register_host<P: KernelProvider>(slf: *mut c_void, ctx: u64, region: SynHostRegion, out: *mut SynRegistration) -> SynStatus {
                g::call(inst::<P>(slf), "register_host", |i| g::register_host(i, ctx, region, wr(out)?))
            }

            extern "C" fn host_registrar<P: KernelProvider + 'static>(slf: *mut c_void, ctx: u64, out: *mut SynRegistrarVtbl) -> SynStatus {
                g::call(inst::<P>(slf), "host_registrar", |i| {
                    let out = wr(out)?;
                    out.struct_size = 0;
                    if let Some(state) = g::host_registrar(i, ctx)? {
                        out.struct_size = ::std::mem::size_of::<SynRegistrarVtbl>() as u32;
                        out.register = registrar_register::<P>;
                        out.unregister = registrar_unregister::<P>;
                        out.close = registrar_close::<P>;
                        out.slf = Box::into_raw(Box::new(state)).cast::<c_void>();
                    }
                    Ok(())
                })
            }

            fn registrar<'a, P>(slf: *mut c_void) -> Option<&'a g::RegistrarState<P>> {
                if slf.is_null() {
                    return None;
                }
                // SAFETY: `slf` came from `Box::into_raw` in `host_registrar` and is freed only
                // by `registrar_close`, which the core calls once, after its last use.
                Some(unsafe { &*slf.cast::<g::RegistrarState<P>>() })
            }

            extern "C" fn registrar_register<P: KernelProvider + 'static>(slf: *mut c_void, region: SynHostRegion, flags: u32, out: *mut SynRegistration) -> SynStatus {
                match registrar::<P>(slf) {
                    Some(st) => g::call(Some(st.inst), "registrar_register", |_| {
                        g::registrar_register(st, region, flags, wr(out)?)
                    }),
                    None => status_of_error(&null_pointer()),
                }
            }

            extern "C" fn registrar_unregister<P: KernelProvider + 'static>(slf: *mut c_void, reg: *const SynRegistration) -> SynStatus {
                match registrar::<P>(slf) {
                    Some(st) => g::call(Some(st.inst), "registrar_unregister", |_| {
                        g::registrar_unregister(st, rd(reg)?)
                    }),
                    None => status_of_error(&null_pointer()),
                }
            }

            extern "C" fn registrar_close<P: KernelProvider + 'static>(slf: *mut c_void) {
                if slf.is_null() {
                    return;
                }
                // The registrar's destructor is provider code, so it runs under the guard too.
                let _ = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
                    // SAFETY: `slf` came from `Box::into_raw` in `host_registrar`, and the core
                    // calls `close` once, so this is the only owner.
                    drop(unsafe { Box::from_raw(slf.cast::<g::RegistrarState<P>>()) });
                }));
            }

            extern "C" fn import<P: KernelProvider>(slf: *mut c_void, ctx: u64, region: SynHostRegion, out: *mut SynBuffer) -> SynStatus {
                g::call(inst::<P>(slf), "import", |i| g::import(i, ctx, region, wr(out)?))
            }

            extern "C" fn set_resident<P: KernelProvider>(slf: *mut c_void, ctx: u64, bufs: *const SynBuffer, n: usize, resident: u32) -> SynStatus {
                g::call(inst::<P>(slf), "set_resident", |i| {
                    g::set_resident(i, ctx, items(bufs, n)?, resident != 0)
                })
            }

            extern "C" fn repack<P: KernelProvider>(
                slf: *mut c_void,
                op: *const u8,
                op_len: usize,
                src: *const u8,
                src_len: usize,
                dst: *mut u8,
                dst_len: usize,
                present: *mut u32,
                layout: *mut u32,
            ) -> SynStatus {
                g::call(inst::<P>(slf), "repack", |i| {
                    g::repack(
                        i,
                        items(op, op_len)?,
                        items(src, src_len)?,
                        items_mut(dst, dst_len)?,
                        wr(present)?,
                        wr(layout)?,
                    )
                })
            }

            extern "C" fn repacked_len<P: KernelProvider>(slf: *mut c_void, op: *const u8, op_len: usize, src: u64, present: *mut u32, out: *mut u64) -> SynStatus {
                g::call(inst::<P>(slf), "repacked_len", |i| {
                    g::repacked_len(i, items(op, op_len)?, src, wr(present)?, wr(out)?)
                })
            }

            extern "C" fn stream<P: KernelProvider>(slf: *mut c_void, ctx: u64, prio: u32, out: *mut u64) -> SynStatus {
                g::call(inst::<P>(slf), "stream", |i| g::stream(i, ctx, prio, wr(out)?))
            }

            extern "C" fn event<P: KernelProvider>(slf: *mut c_void, ctx: u64, out: *mut u64) -> SynStatus {
                g::call(inst::<P>(slf), "event", |i| g::event(i, ctx, wr(out)?))
            }

            extern "C" fn record<P: KernelProvider>(slf: *mut c_void, ctx: u64, s: u64, e: u64) -> SynStatus {
                g::call(inst::<P>(slf), "record", |i| g::record(i, ctx, s, e))
            }

            extern "C" fn wait<P: KernelProvider>(slf: *mut c_void, ctx: u64, s: u64, e: u64) -> SynStatus {
                g::call(inst::<P>(slf), "wait", |i| g::wait(i, ctx, s, e))
            }

            extern "C" fn signal_host<P: KernelProvider>(slf: *mut c_void, ctx: u64, s: u64, flag: u32, value: u32) -> SynStatus {
                g::call(inst::<P>(slf), "signal_host", |i| g::signal_host(i, ctx, s, flag, value))
            }

            extern "C" fn wait_host<P: KernelProvider>(
                slf: *mut c_void,
                ctx: u64,
                s: u64,
                flag: u32,
                value: u32,
                cmp: u32,
                has_reset: u32,
                reset: u32,
            ) -> SynStatus {
                g::call(inst::<P>(slf), "wait_host", |i| {
                    g::wait_host(i, ctx, s, (flag, value, cmp), (has_reset != 0).then_some(reset))
                })
            }

            extern "C" fn arm_wake<P: KernelProvider>(slf: *mut c_void, ctx: u64, s: u64, kind: u32, handle: u64) -> SynStatus {
                g::call(inst::<P>(slf), "arm_wake", |i| g::arm_wake(i, ctx, s, kind, handle))
            }

            extern "C" fn poll<P: KernelProvider>(slf: *mut c_void, ctx: u64, e: u64, done: *mut u32) -> SynStatus {
                g::call(inst::<P>(slf), "poll", |i| g::poll(i, ctx, e, wr(done)?))
            }

            extern "C" fn elapsed<P: KernelProvider>(slf: *mut c_void, ctx: u64, start: u64, end: u64, present: *mut u32, nanos: *mut u64) -> SynStatus {
                g::call(inst::<P>(slf), "elapsed", |i| {
                    g::elapsed(i, ctx, start, end, wr(present)?, wr(nanos)?)
                })
            }

            extern "C" fn synchronize<P: KernelProvider>(slf: *mut c_void, ctx: u64, s: u64) -> SynStatus {
                g::call(inst::<P>(slf), "synchronize", |i| g::synchronize(i, ctx, s))
            }

            extern "C" fn copy<P: KernelProvider>(slf: *mut c_void, ctx: u64, s: u64, src: *const SynSlice, dst: *const SynSlice) -> SynStatus {
                g::call(inst::<P>(slf), "copy", |i| g::copy(i, ctx, s, *rd(src)?, *rd(dst)?))
            }

            extern "C" fn copy_batch<P: KernelProvider>(slf: *mut c_void, ctx: u64, s: u64, pairs: *const SynSlicePair, n: usize) -> SynStatus {
                g::call(inst::<P>(slf), "copy_batch", |i| g::copy_batch(i, ctx, s, items(pairs, n)?))
            }

            extern "C" fn prepare<P: KernelProvider>(
                slf: *mut c_void,
                ctx: u64,
                op: *const u8,
                op_len: usize,
                shape: *const u8,
                shape_len: usize,
                out: *mut SynPreparedOp,
            ) -> SynStatus {
                g::call(inst::<P>(slf), "prepare", |i| {
                    g::prepare(i, ctx, items(op, op_len)?, items(shape, shape_len)?, wr(out)?)
                })
            }

            extern "C" fn launch<P: KernelProvider>(slf: *mut c_void, ctx: u64, s: u64, op: *const SynPreparedOp, args: *const SynOpArgs) -> SynStatus {
                g::call(inst::<P>(slf), "launch", |i| {
                    let args = rd(args)?;
                    let slices = g::ArgSlices {
                        inputs: items(args.inputs, args.n_inputs)?,
                        outputs: items(args.outputs, args.n_outputs)?,
                        weights: items(args.weights, args.n_weights)?,
                        statics: items(args.statics, args.n_statics)?,
                    };
                    g::launch(i, ctx, s, rd(op)?, args, &slices)
                })
            }

            extern "C" fn begin_capture<P: KernelProvider>(slf: *mut c_void, ctx: u64, s: u64) -> SynStatus {
                g::call(inst::<P>(slf), "begin_capture", |i| g::begin_capture(i, ctx, s))
            }

            extern "C" fn end_capture<P: KernelProvider>(slf: *mut c_void, ctx: u64, s: u64, out: *mut u64) -> SynStatus {
                g::call(inst::<P>(slf), "end_capture", |i| g::end_capture(i, ctx, s, wr(out)?))
            }

            extern "C" fn replay<P: KernelProvider>(slf: *mut c_void, ctx: u64, s: u64, graph: u64) -> SynStatus {
                g::call(inst::<P>(slf), "replay", |i| g::replay(i, ctx, s, graph))
            }

            extern "C" fn health<P: KernelProvider>(slf: *mut c_void, ctx: u64, out: *mut SynHealth) -> SynStatus {
                g::call(inst::<P>(slf), "health", |i| g::health(i, ctx, wr(out)?))
            }

            extern "C" fn last_error<P: KernelProvider>(slf: *mut c_void, _ctx: u64, buf: *mut u8, cap: usize, len: *mut usize) -> SynStatus {
                let Some(i) = inst::<P>(slf) else {
                    return status_of_error(&null_pointer());
                };
                let run = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| -> Result<(), Error> {
                    g::last_error(i, items_mut(buf, cap)?, wr(len)?);
                    Ok(())
                }));
                match run {
                    Ok(Ok(())) => SynStatus::OK,
                    Ok(Err(e)) => status_of_error(&e),
                    Err(_) => status_of_error(&$crate::abi::wire::panicked("the provider panicked in `last_error`".into())),
                }
            }

            fn vtbl<P: KernelProvider + 'static>(inst: &'static Instance<P>) -> Result<SynProviderVtbl, Error> {
                let (backend, domain) = inst.identity()?;
                Ok(SynProviderVtbl {
                    struct_size: ::std::mem::size_of::<SynProviderVtbl>() as u32,
                    id: inst.id_ptr(),
                    backend,
                    domain,
                    build: build::<P>,
                    enumerate: enumerate::<P>,
                    open: open::<P>,
                    close_ctx: close_ctx::<P>,
                    memory_info: memory_info::<P>,
                    ops: ops::<P>,
                    query: query::<P>,
                    alloc: alloc::<P>,
                    free: free::<P>,
                    map_segment: map_segment::<P>,
                    unmap_segment: unmap_segment::<P>,
                    register_host: register_host::<P>,
                    host_registrar: host_registrar::<P>,
                    import: import::<P>,
                    set_resident: set_resident::<P>,
                    repack: repack::<P>,
                    repacked_len: repacked_len::<P>,
                    stream: stream::<P>,
                    event: event::<P>,
                    record: record::<P>,
                    wait: wait::<P>,
                    signal_host: signal_host::<P>,
                    wait_host: wait_host::<P>,
                    arm_wake: arm_wake::<P>,
                    poll: poll::<P>,
                    elapsed: elapsed::<P>,
                    synchronize: synchronize::<P>,
                    copy: copy::<P>,
                    copy_batch: copy_batch::<P>,
                    prepare: prepare::<P>,
                    launch: launch::<P>,
                    begin_capture: begin_capture::<P>,
                    end_capture: end_capture::<P>,
                    replay: replay::<P>,
                    health: health::<P>,
                    last_error: last_error::<P>,
                    slf: ::std::ptr::from_ref(inst).cast_mut().cast::<c_void>(),
                })
            }

            extern "C" fn shutdown() -> SynStatus {
                // Providers are leaked for the life of the process, so there is nothing to
                // release; the core calls this once before it unloads the library.
                SynStatus::OK
            }

            fn build_module(host: *const SynHostApi) -> Result<usize, Error> {
                // The header is the prefix of every table, so it is read before the rest.
                let header = rd(host.cast::<SynHeader>())?;
                if header.magic != $crate::abi::MAGIC
                    || header.abi_major != $crate::abi::ABI_MAJOR
                    || (header.struct_size as usize) < ::std::mem::size_of::<SynHostApi>()
                {
                    return Err($crate::abi::wire::internal(
                        "the core's host table is not ABI 1 or is too small".into(),
                    ));
                }
                let api = g::HostApi::new(rd(host)?, cstr);
                let mut tables = Vec::new();
                $(
                    let instance: &'static Instance<$ty> =
                        Box::leak(Box::new(Instance::new(($ctor)(&api), api)));
                    tables.push(vtbl::<$ty>(instance)?);
                )+
                let tables: &'static [SynProviderVtbl] = Box::leak(tables.into_boxed_slice());
                let module: &'static SynModule = Box::leak(Box::new(SynModule {
                    h: SynHeader::new($build_id, ::std::mem::size_of::<SynModule>()),
                    n_providers: tables.len() as u32,
                    providers: tables.as_ptr(),
                    shutdown,
                }));
                Ok(::std::ptr::from_ref(module) as usize)
            }

            /// The one symbol the library exports.
            #[unsafe(no_mangle)]
            pub extern "C" fn syntropy_provider_entry_v1(host: *const SynHostApi) -> *const SynModule {
                static MODULE: ::std::sync::OnceLock<usize> = ::std::sync::OnceLock::new();
                let built = ::std::panic::catch_unwind(|| {
                    if let Some(addr) = MODULE.get() {
                        return Some(*addr);
                    }
                    let addr = build_module(host).ok()?;
                    Some(*MODULE.get_or_init(|| addr))
                });
                match built {
                    Ok(Some(addr)) => addr as *const SynModule,
                    _ => ::std::ptr::null(),
                }
            }
        };
    };
}
