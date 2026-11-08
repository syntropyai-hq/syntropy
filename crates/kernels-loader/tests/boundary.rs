// SPDX-License-Identifier: MIT OR Apache-2.0

//! The loader's boundary tests: every call crosses the C ABI into the test provider library and
//! back (COMP-E9-T04; RFC-0040 §10.4).

mod common;

use std::path::Path;

use smallvec::smallvec;
use syntropy_core::resource::{Backend, DeviceKind};
use syntropy_core::{BufferView, Bytes, DType, DeviceId, ErrorClass, NodeId, ViewKind};
use syntropy_kernels::abi::{ABI_MAJOR, ABI_MINOR, MAGIC};
use syntropy_kernels::{
    AllocRequest, BufferSlice, DeviceBuffer, DeviceCtx, DeviceHealth, ErrorCode, HostFlag,
    HostRegFlags, HostRegion, HostWaker, KernelProvider, LayoutVersion, MemKind, MemoryDomain,
    OpArgs, OpKey, OpKind, OpenOptions, PowerHint, PreparedOp, Scalar, ShapeClass, ShapeSig,
    Stream, StreamPlan, StreamPriority, TensorArg, WaitCmp, WakeKind,
};
use syntropy_kernels_loader::{Module, ModuleProvider};

fn load(path: &Path) -> ModuleProvider {
    common::install_services();
    Module::load(path)
        .expect("the test provider loads")
        .into_providers()
        .remove(0)
}

fn shared() -> ModuleProvider {
    load(&common::provider_library())
}

fn device() -> DeviceId {
    DeviceId {
        node: NodeId([5; 32]),
        ordinal: 3,
    }
}

fn options() -> OpenOptions {
    OpenOptions {
        streams: StreamPlan {
            compute: 2,
            copy_in: 1,
            copy_out: 1,
            collective: 0,
        },
        doorbell_slots: 64,
        capture: true,
        deterministic: true,
        power: PowerHint::Battery,
    }
}

fn open(p: &ModuleProvider) -> DeviceCtx {
    p.open(device(), &options()).expect("open")
}

fn key(layout: u16, variant: u16) -> OpKey {
    OpKey {
        op: if layout == 0 {
            OpKind::Gemv
        } else {
            OpKind::Gemm
        },
        schemes: smallvec![
            syntropy_core::QuantScheme::Plain(DType::F32),
            syntropy_core::QuantScheme::Plain(DType::F32),
            syntropy_core::QuantScheme::Plain(DType::F32)
        ],
        layout: LayoutVersion(layout),
        variant,
    }
}

fn slice(buffer: u64, offset: u64, bytes: u64) -> BufferSlice {
    BufferSlice {
        buffer,
        offset: Bytes(offset),
        bytes: Bytes(bytes),
    }
}

#[test]
fn the_module_states_its_identity() {
    common::install_services();
    let module = Module::load(&common::provider_library()).expect("load");
    let header = module.header();
    assert_eq!(header.magic, MAGIC);
    assert_eq!((header.abi_major, header.abi_minor), (ABI_MAJOR, ABI_MINOR));
    assert_eq!(header.build_id, common::MODULE_BUILD_ID);
    assert_eq!(module.providers().len(), 1);
    let p = &module.providers()[0];
    assert_eq!(p.id().0, "test-host");
    assert_eq!(p.backend(), Backend::Cpu);
    assert_eq!(p.domain(), MemoryDomain::Host);
    assert_eq!(p.build().version, 1);
    assert_eq!(
        p.build().toolchain.as_slice(),
        [("rustc".into(), "test".to_owned())]
    );
    let ops: Vec<_> = p.ops().iter().cloned().collect();
    assert_eq!(ops.len(), 2);
    assert!(p.ops().contains(&key(0, 0)) && p.ops().contains(&key(1, 0)));
}

#[test]
fn enumerate_and_open_round_trip_and_the_core_services_are_reached() {
    let p = shared();
    let devices = p.enumerate().expect("enumerate");
    assert_eq!(devices.len(), 1);
    assert_eq!(devices[0].kind, DeviceKind::Cpu);
    assert_eq!(devices[0].name, "test device 0");

    let (logs, metrics) = (
        common::LOG_LINES.load(std::sync::atomic::Ordering::Relaxed),
        common::METRIC_TOTAL.load(std::sync::atomic::Ordering::Relaxed),
    );
    let ctx = open(&p);
    assert!(common::LOG_LINES.load(std::sync::atomic::Ordering::Relaxed) > logs);
    assert!(common::METRIC_TOTAL.load(std::sync::atomic::Ordering::Relaxed) > metrics);
    // The provider saw the device ID and options it was given.
    let info = p.memory_info(&ctx).expect("memory_info");
    assert_eq!(info.free, Bytes(3));
    assert_eq!(info.budget, Some(Bytes(900)));
    assert!(!info.unified);
    assert!(p.fault().is_none());
}

#[test]
fn a_result_larger_than_the_first_buffer_is_fetched_whole() {
    let p = shared();
    common::set("devices", "200");
    let devices = p.enumerate().expect("enumerate");
    common::clear("devices");
    assert_eq!(devices.len(), 200);
    assert_eq!(devices[199].name, "test device 199");
}

#[test]
fn a_long_error_message_is_fetched_whole_and_keeps_its_class_and_code() {
    let p = shared();
    let ctx = open(&p);
    common::set("long_error", "3000");
    let error = p
        .alloc(
            &ctx,
            AllocRequest {
                bytes: Bytes(64),
                kind: MemKind::Device,
                align: Bytes(16),
                reserve_only: false,
            },
        )
        .expect_err("the provider fails");
    common::clear("long_error");
    assert_eq!(error.class, ErrorClass::Capacity);
    assert_eq!(error.code, syntropy_kernels::error::codes::ALLOC_FAILED);
    assert_eq!(error.message, "x".repeat(3000));
}

#[test]
fn memory_calls_round_trip() {
    let p = shared();
    let ctx = open(&p);
    let req = AllocRequest {
        bytes: Bytes(4096),
        kind: MemKind::Unified,
        align: Bytes(256),
        reserve_only: true,
    };
    let buf = p.alloc(&ctx, req).expect("alloc");
    assert_eq!((buf.bytes, buf.kind), (Bytes(4096), MemKind::Unified));
    assert_eq!(buf.addr, 0x1000 * buf.handle);

    let seg = p
        .map_segment(&ctx, &buf, Bytes(512), Bytes(1024))
        .expect("map_segment");
    assert_eq!(
        (seg.reservation, seg.offset, seg.bytes),
        (buf.handle, Bytes(512), Bytes(1024))
    );
    p.unmap_segment(&ctx, seg).expect("unmap_segment");

    let region = HostRegion {
        addr: 0x7000,
        len: Bytes(8192),
    };
    assert_eq!(
        p.register_host(&ctx, region).expect("register").region,
        region
    );
    let imported = p.import(&ctx, region).expect("import");
    assert_eq!((imported.addr, imported.bytes), (0x7000, Bytes(8192)));
    assert_eq!(imported.kind, MemKind::HostMapped);

    p.set_resident(&ctx, &[&buf], true).expect("set_resident");
    p.free(&ctx, buf);
    assert!(p.fault().is_none());

    let zero = AllocRequest {
        bytes: Bytes(0),
        ..req
    };
    let error = p.alloc(&ctx, zero).expect_err("zero bytes");
    assert_eq!(error.message, "alloc of zero bytes");
}

#[test]
fn query_prepare_and_repack_round_trip() {
    let p = shared();
    let ctx = open(&p);
    let shape = ShapeSig::new(ShapeClass::Batch { rows: 8 }, &[64, 128]);
    let caps = p.query(&ctx, &key(0, 0), &shape).expect("a candidate");
    assert_eq!(caps.workspace, Bytes(8 * 16));
    assert!(caps.capture_safe && caps.deterministic);
    assert_eq!(caps.kernel_hash.0, [3; 32]);
    assert!(p.query(&ctx, &key(0, 5), &shape).is_none());
    assert!(p.fault().is_none());

    let prepared = p.prepare(&ctx, &key(1, 0), &shape).expect("prepare");
    assert_eq!((prepared.workspace, prepared.handle), (Bytes(128), 2));

    let mut dst = [0u8; 4];
    let layout = p.repack(&key(1, 0), &[1, 2, 3, 4], &mut dst);
    assert_eq!(layout.expect("present").expect("ok"), LayoutVersion(9));
    assert_eq!(dst, [4, 3, 2, 1]);
    assert!(p.repack(&key(0, 0), &[1], &mut [0]).is_none());
    assert_eq!(p.repacked_len(&key(1, 0), Bytes(100)), Some(Bytes(108)));
    assert_eq!(p.repacked_len(&key(0, 0), Bytes(100)), None);
}

#[test]
fn streams_events_and_graphs_round_trip() {
    let p = shared();
    let ctx = open(&p);
    let s = p.stream(&ctx, StreamPriority::CopyIn).expect("stream");
    let (e1, e2) = (p.event(&ctx).expect("e1"), p.event(&ctx).expect("e2"));
    p.record(&ctx, &s, &e1).expect("record");
    p.wait(&ctx, &s, &e2).expect("wait");
    assert_eq!(
        p.poll(&ctx, &e1).expect("poll"),
        e1.handle.is_multiple_of(2)
    );
    assert_eq!(
        p.elapsed(&ctx, &e1, &e2).map(|n| n.0),
        e2.handle.checked_sub(e1.handle)
    );
    assert_eq!(p.elapsed(&ctx, &e2, &e1), None);
    p.synchronize(&ctx, &s).expect("synchronize");
    p.copy(&ctx, &s, slice(1, 0, 8), slice(2, 0, 8))
        .expect("copy");
    let error = p
        .copy(&ctx, &s, slice(1, 0, 8), slice(2, 0, 16))
        .expect_err("sizes differ");
    assert!(error.message.contains("BufferSlice"), "{error}");
    p.begin_capture(&ctx, &s).expect("begin");
    let graph = p.end_capture(&ctx, &s).expect("end");
    p.replay(&ctx, &s, &graph).expect("replay");
    assert_eq!(p.health(&ctx), DeviceHealth::Ok);
    assert!(p.fault().is_none());
}

/// The stream that makes structured calls echo what they received.
fn echo_stream(p: &ModuleProvider, ctx: &DeviceCtx) -> Stream {
    p.stream(ctx, StreamPriority::Compute).expect("stream")
}

#[test]
fn structured_arguments_arrive_as_sent() {
    let p = shared();
    let ctx = open(&p);
    let s = echo_stream(&p, &ctx);

    let inputs = [TensorArg::packed(slice(1, 0, 64), DType::F16, [1, 2, 4, 4]).unwrap()];
    let outputs = [
        TensorArg::packed(slice(2, 16, 128), DType::BF16, [2, 2, 2, 2]).unwrap(),
        TensorArg::packed(slice(3, 0, 32), DType::I32, [1, 1, 1, 8]).unwrap(),
    ];
    let weights = [
        BufferView {
            kind: ViewKind::Host,
            addr: 0x100,
            len: Bytes(32),
        },
        BufferView {
            kind: ViewKind::Device(device()),
            addr: 0x200,
            len: Bytes(64),
        },
        BufferView {
            kind: ViewKind::Unified,
            addr: 0x300,
            len: Bytes(8),
        },
    ];
    let statics = [Scalar::F32(0.5), Scalar::I32(-3), Scalar::U32(u32::MAX)];
    let args = OpArgs {
        inputs: &inputs,
        outputs: &outputs,
        weights: &weights,
        step: slice(9, 4, 128),
        workspace: Some(slice(10, 0, 256)),
        statics: &statics,
    };
    let mut prepared = PreparedOp::new(Bytes(256));
    prepared.handle = 77;
    let error = p.launch(&ctx, &s, &prepared, &args).expect_err("echo");
    assert_eq!(error.message, format!("launch: {prepared:?} {args:?}"));

    let none = OpArgs {
        workspace: None,
        ..OpArgs::none()
    };
    let error = p.launch(&ctx, &s, &prepared, &none).expect_err("echo");
    assert_eq!(error.message, format!("launch: {prepared:?} {none:?}"));

    let pairs = [
        (slice(1, 0, 8), slice(2, 8, 8)),
        (slice(3, 1, 2), slice(4, 3, 2)),
    ];
    let error = p.copy_batch(&ctx, &s, &pairs).expect_err("echo");
    assert_eq!(error.message, format!("copy_batch: {pairs:?}"));

    let flag = HostFlag { page: 513, slot: 7 };
    let error = p.signal_host(&ctx, &s, flag, 99).expect_err("echo");
    assert_eq!(error.message, format!("signal_host: {flag:?} 99"));
    for (cmp, reset) in [(WaitCmp::Eq, None), (WaitCmp::Geq, Some(0))] {
        let error = p
            .wait_host(&ctx, &s, flag, 5, cmp, reset)
            .expect_err("echo");
        assert_eq!(
            error.message,
            format!("wait_host: {flag:?} 5 {cmp:?} {reset:?}")
        );
    }
    let waker = HostWaker {
        kind: WakeKind::TimelineValue,
        handle: 0xABCD,
    };
    let error = p.arm_wake(&ctx, &s, &waker).expect_err("echo");
    assert_eq!(error.message, format!("arm_wake: {waker:?}"));

    let a = p
        .alloc(
            &ctx,
            AllocRequest {
                bytes: Bytes(8),
                kind: MemKind::Device,
                align: Bytes(8),
                reserve_only: false,
            },
        )
        .unwrap();
    let b = DeviceBuffer {
        handle: 99,
        ..DeviceBuffer {
            addr: 1,
            bytes: Bytes(2),
            kind: MemKind::Device,
            handle: 0,
        }
    };
    let error = p.set_resident(&ctx, &[&a, &b], false).expect_err("echo");
    assert_eq!(error.message, format!("set_resident: {:?} false", [&a, &b]));
}

#[test]
fn the_host_registrar_is_its_own_table_and_works_from_another_thread() {
    let p = shared();
    let ctx = open(&p);
    let registrar = p.host_registrar(&ctx).expect("a registrar");
    let region = HostRegion {
        addr: 0x9000,
        len: Bytes(4096),
    };
    let handle = std::thread::scope(|scope| {
        scope
            .spawn(move || {
                let reg = registrar
                    .register(region, HostRegFlags::DMA | HostRegFlags::MAPPED)
                    .expect("register");
                let handle = reg.handle;
                registrar.unregister(reg).expect("unregister");
                let bad = syntropy_kernels::Registration { region, handle: 1 };
                let error = registrar.unregister(bad).expect_err("bad handle");
                assert!(error.message.starts_with("unregister:"), "{error}");
                handle
            })
            .join()
            .expect("the registrar thread")
    });
    assert_eq!(handle, 1003);
}

/// Lets a test carry a context to another thread, which safe Rust forbids.
struct Smuggled(DeviceCtx);
// SAFETY: this is the point of the test: the context is moved to a thread that did not open it,
// to show the module refuses it. The context is only handed to the proxy, never dereferenced.
unsafe impl Send for Smuggled {}

#[test]
fn a_context_works_only_on_the_thread_that_opened_it() {
    let p = shared();
    let ctx = open(&p);
    p.event(&ctx).expect("the opening thread");
    let smuggled = Smuggled(ctx);
    let (error, smuggled) = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let smuggled = smuggled;
                let error = p.event(&smuggled.0).expect_err("another thread");
                (error, smuggled)
            })
            .join()
            .expect("the other thread")
    });
    assert_eq!(error.class, ErrorClass::Internal);
    assert!(error.message.contains("thread that opened it"), "{error}");
    // The refusal is a caller bug, not a panic: the provider stays healthy.
    assert_eq!(p.health(&smuggled.0), DeviceHealth::Ok);
    // Dropping the context on the thread that opened it closes it.
    drop(smuggled);
    assert_eq!(
        p.fault().expect("the refusal is recorded").class,
        ErrorClass::Internal
    );
}

#[test]
fn a_context_from_another_provider_is_refused() {
    let p = shared();
    let foreign = DeviceCtx::new(42u32);
    let error = p.event(&foreign).expect_err("foreign");
    assert_eq!(error.class, ErrorClass::Internal);
    assert!(error.message.contains("not opened by this module provider"));
}

#[test]
fn a_missing_or_relative_path_is_module_missing() {
    common::install_services();
    for path in [
        "/definitely/not/here/libnothing.so",
        "relative/libnothing.so",
    ] {
        let error = Module::load(Path::new(path)).expect_err("cannot open").0;
        assert_eq!(error.class, ErrorClass::Unsupported);
        assert_eq!(
            error.code,
            ErrorCode::new(syntropy_core::Domain::Kernels, 2),
            "{error}"
        );
    }
}

#[test]
fn a_library_without_the_entry_symbol_is_module_abi() {
    common::install_services();
    let dir = common::TempDir::new("empty");
    let source = dir.0.join("empty.rs");
    std::fs::write(
        &source,
        "#[unsafe(no_mangle)] pub extern \"C\" fn other() {}\n",
    )
    .unwrap();
    let lib = dir.0.join(format!(
        "{}empty{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    ));
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
    let status = std::process::Command::new(rustc)
        .args(["--edition", "2024", "--crate-type", "cdylib", "-o"])
        .arg(&lib)
        .arg(&source)
        .status()
        .expect("run rustc");
    assert!(status.success(), "rustc failed to build the empty library");
    let error = Module::load(&lib).expect_err("no entry symbol").0;
    assert_eq!(error.class, ErrorClass::Unsupported);
    assert_eq!(
        error.code,
        ErrorCode::new(syntropy_core::Domain::Kernels, 4),
        "{error}"
    );
    assert!(
        error.message.contains("syntropy_provider_entry_v1"),
        "{error}"
    );
}
