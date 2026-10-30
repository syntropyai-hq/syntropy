// SPDX-License-Identifier: MIT OR Apache-2.0

//! A panic injected in each thunk (COMP-E9-T04; RFC-0040 §10.4): the provider becomes `Sticky`
//! and the loader returns `Internal` to its caller.
//!
//! Each case loads its own copy of the test provider library, so a provider made sticky in one
//! case does not affect another. After the injected call, every case checks the three effects
//! of RFC-0040 §10.4 and §13 code 16:
//!
//! 1. the call reports `Internal` (code 16, `ProviderPanicked`), or, where the method's signature
//!    cannot carry an error, `ModuleProvider::fault` holds it;
//! 2. `health` reports `Sticky` with that code;
//! 3. every later call other than `health`, `free` and `close_ctx` fails as `DeviceSticky` (8).
//!
//! The `last_error` thunk calls no provider code, so no provider panic can reach it; its body
//! runs under the same guard.

mod common;

use smallvec::smallvec;
use syntropy_core::{Bytes, DType, DeviceId, ErrorClass, NodeId, QuantScheme};
use syntropy_kernels::{
    AllocRequest, BufferSlice, DeviceBuffer, DeviceCtx, DeviceHealth, Error, ErrorCode, Event,
    HostFlag, HostRegFlags, HostRegion, HostRegistrar, HostWaker, KernelProvider, LayoutVersion,
    MemKind, OpArgs, OpKey, OpKind, OpenOptions, PowerHint, PreparedOp, ShapeClass, ShapeSig,
    Stream, StreamPlan, StreamPriority, WaitCmp, WakeKind,
};
use syntropy_kernels_loader::{Module, ModuleProvider};

const PANICKED: ErrorCode = ErrorCode::PROVIDER_PANICKED;

fn options() -> OpenOptions {
    OpenOptions {
        streams: StreamPlan {
            compute: 1,
            copy_in: 1,
            copy_out: 1,
            collective: 0,
        },
        doorbell_slots: 8,
        capture: false,
        deterministic: false,
        power: PowerHint::Mains,
    }
}

fn device() -> DeviceId {
    DeviceId {
        node: NodeId([1; 32]),
        ordinal: 0,
    }
}

fn key() -> OpKey {
    OpKey {
        op: OpKind::Gemv,
        schemes: smallvec![QuantScheme::Plain(DType::F32); 3],
        layout: LayoutVersion(0),
        variant: 0,
    }
}

fn slice(buffer: u64) -> BufferSlice {
    BufferSlice {
        buffer,
        offset: Bytes(0),
        bytes: Bytes(8),
    }
}

/// What the case holds before the panic is armed.
struct Fixture {
    p: ModuleProvider,
    ctx: DeviceCtx,
    /// A second context, kept open to read `health` after the first call fails.
    witness: DeviceCtx,
    stream: Stream,
    e1: Event,
    e2: Event,
    buf: DeviceBuffer,
    registrar: Box<dyn HostRegistrar>,
}

/// How a method reports the panic.
enum Reported {
    /// An `Err` the call returned.
    Error(Error),
    /// Nothing the call returned; the provider's `fault` has it.
    Fault,
}

fn expect_internal(name: &str, error: &Error) {
    assert_eq!(error.class, ErrorClass::Internal, "{name}: {error}");
    assert_eq!(error.code, PANICKED, "{name}: {error}");
    assert!(
        error.message.contains(&format!("panicked in `{name}`")),
        "{name}: {error}"
    );
}

fn fixture(dir: &common::TempDir) -> Fixture {
    let p = Module::load(&common::private_copy(dir))
        .expect("load")
        .into_providers()
        .remove(0);
    let ctx = p.open(device(), &options()).expect("open");
    let witness = p.open(device(), &options()).expect("open");
    let stream = p.stream(&ctx, StreamPriority::CopyIn).expect("stream");
    let (e1, e2) = (p.event(&ctx).expect("event"), p.event(&ctx).expect("event"));
    let buf = p
        .alloc(
            &ctx,
            AllocRequest {
                bytes: Bytes(64),
                kind: MemKind::Device,
                align: Bytes(16),
                reserve_only: false,
            },
        )
        .expect("alloc");
    let registrar = p.host_registrar(&ctx).expect("a registrar");
    Fixture {
        p,
        ctx,
        witness,
        stream,
        e1,
        e2,
        buf,
        registrar,
    }
}

/// Arms `name`, runs `call`, disarms, and checks the effects.
fn inject(name: &str, call: impl FnOnce(&Fixture) -> Option<Result<(), Error>>) {
    common::install_services();
    let dir = common::TempDir::new(name);
    let fx = fixture(&dir);
    common::set("panic_in", name);
    let outcome = call(&fx);
    common::clear("panic_in");
    let reported = match outcome {
        Some(Err(error)) => Reported::Error(error),
        Some(Ok(())) => panic!("{name}: the call succeeded after an injected panic"),
        None => Reported::Fault,
    };
    match reported {
        Reported::Error(error) => expect_internal(name, &error),
        Reported::Fault => expect_internal(name, &fx.p.fault().expect("a recorded fault")),
    }
    assert_eq!(
        fx.p.health(&fx.witness),
        DeviceHealth::Sticky { code: PANICKED },
        "{name}: health"
    );
    let later =
        fx.p.event(&fx.witness)
            .expect_err("a sticky provider refuses calls");
    assert_eq!(later.class, ErrorClass::Device, "{name}: {later}");
    assert_eq!(later.code, ErrorCode::DEVICE_STICKY, "{name}: {later}");
}

macro_rules! cases {
    ($($test:ident: $name:literal, |$fx:ident| $body:expr;)+) => {
        $(
            #[test]
            fn $test() {
                inject($name, |$fx| $body);
            }
        )+
    };
}

cases! {
    enumerate: "enumerate", |fx| Some(fx.p.enumerate().map(drop));
    open: "open", |fx| Some(fx.p.open(device(), &options()).map(drop));
    memory_info: "memory_info", |fx| Some(fx.p.memory_info(&fx.ctx).map(drop));
    alloc: "alloc", |fx| Some(fx.p.alloc(&fx.ctx, AllocRequest {
        bytes: Bytes(8),
        kind: MemKind::Device,
        align: Bytes(8),
        reserve_only: false,
    }).map(drop));
    map_segment: "map_segment", |fx| Some(fx.p.map_segment(&fx.ctx, &fx.buf, Bytes(0), Bytes(8)).map(drop));
    unmap_segment: "unmap_segment", |fx| {
        let seg = fx.p.map_segment(&fx.ctx, &fx.buf, Bytes(0), Bytes(8)).expect("map_segment");
        Some(fx.p.unmap_segment(&fx.ctx, seg))
    };
    register_host: "register_host", |fx| Some(fx.p.register_host(&fx.ctx, HostRegion { addr: 1, len: Bytes(8) }).map(drop));
    import: "import", |fx| Some(fx.p.import(&fx.ctx, HostRegion { addr: 1, len: Bytes(8) }).map(drop));
    set_resident: "set_resident", |fx| Some(fx.p.set_resident(&fx.ctx, &[&fx.buf], true));
    stream: "stream", |fx| Some(fx.p.stream(&fx.ctx, StreamPriority::Low).map(drop));
    event: "event", |fx| Some(fx.p.event(&fx.ctx).map(drop));
    record: "record", |fx| Some(fx.p.record(&fx.ctx, &fx.stream, &fx.e1));
    wait: "wait", |fx| Some(fx.p.wait(&fx.ctx, &fx.stream, &fx.e1));
    signal_host: "signal_host", |fx| Some(fx.p.signal_host(&fx.ctx, &fx.stream, HostFlag { page: 0, slot: 1 }, 1));
    wait_host: "wait_host", |fx| Some(fx.p.wait_host(&fx.ctx, &fx.stream, HostFlag { page: 0, slot: 1 }, 1, WaitCmp::Eq, None));
    arm_wake: "arm_wake", |fx| Some(fx.p.arm_wake(&fx.ctx, &fx.stream, &HostWaker { kind: WakeKind::Futex, handle: 1 }));
    poll: "poll", |fx| Some(fx.p.poll(&fx.ctx, &fx.e1).map(drop));
    synchronize: "synchronize", |fx| Some(fx.p.synchronize(&fx.ctx, &fx.stream));
    copy: "copy", |fx| Some(fx.p.copy(&fx.ctx, &fx.stream, slice(1), slice(2)));
    copy_batch: "copy_batch", |fx| Some(fx.p.copy_batch(&fx.ctx, &fx.stream, &[(slice(1), slice(2))]));
    prepare: "prepare", |fx| Some(fx.p.prepare(&fx.ctx, &key(), &ShapeSig::new(ShapeClass::Decode, &[8])).map(drop));
    launch: "launch", |fx| Some(fx.p.launch(&fx.ctx, &fx.stream, &PreparedOp::new(Bytes(0)), &OpArgs::none()));
    begin_capture: "begin_capture", |fx| Some(fx.p.begin_capture(&fx.ctx, &fx.stream));
    end_capture: "end_capture", |fx| Some(fx.p.end_capture(&fx.ctx, &fx.stream).map(drop));
    replay: "replay", |fx| Some(fx.p.replay(&fx.ctx, &fx.stream, &syntropy_kernels::CapturedGraph { handle: 1 }));
    registrar_register: "registrar_register", |fx| Some(fx.registrar.register(HostRegion { addr: 1, len: Bytes(8) }, HostRegFlags::DMA).map(drop));
    registrar_unregister: "registrar_unregister", |fx| {
        let reg = fx.registrar.register(HostRegion { addr: 1, len: Bytes(8) }, HostRegFlags::DMA).expect("register");
        Some(fx.registrar.unregister(reg))
    };
}

// The methods whose signature cannot carry an error report through `fault`.

macro_rules! fault_cases {
    ($($test:ident: $name:literal, |$fx:ident| $body:expr;)+) => {
        $(
            #[test]
            fn $test() {
                inject($name, |$fx| {
                    $body;
                    None
                });
            }
        )+
    };
}

fault_cases! {
    free: "free", |fx| fx.p.free(&fx.ctx, DeviceBuffer { addr: fx.buf.addr, bytes: fx.buf.bytes, kind: fx.buf.kind, handle: fx.buf.handle });
    query: "query", |fx| { let _ = fx.p.query(&fx.ctx, &key(), &ShapeSig::new(ShapeClass::Decode, &[8])); };
    repack: "repack", |fx| { let _ = fx.p.repack(&key(), &[1], &mut [0]); };
    repacked_len: "repacked_len", |fx| { let _ = fx.p.repacked_len(&key(), Bytes(8)); };
    elapsed: "elapsed", |fx| { let _ = fx.p.elapsed(&fx.ctx, &fx.e1, &fx.e2); };
    host_registrar: "host_registrar", |fx| { let _ = fx.p.host_registrar(&fx.ctx); };
}

#[test]
fn health_reports_sticky_when_its_own_thunk_panics() {
    common::install_services();
    let dir = common::TempDir::new("health");
    let fx = fixture(&dir);
    common::set("panic_in", "health");
    let health = fx.p.health(&fx.ctx);
    common::clear("panic_in");
    assert_eq!(health, DeviceHealth::Sticky { code: PANICKED });
    assert_eq!(
        fx.p.health(&fx.witness),
        DeviceHealth::Sticky { code: PANICKED }
    );
}

#[test]
fn dropping_a_context_whose_destructor_panics_makes_the_provider_sticky() {
    common::install_services();
    let dir = common::TempDir::new("close_ctx");
    let fx = fixture(&dir);
    common::set("panic_in", "close_ctx");
    drop(fx.ctx);
    common::clear("panic_in");
    expect_internal("close_ctx", &fx.p.fault().expect("a recorded fault"));
    assert_eq!(
        fx.p.health(&fx.witness),
        DeviceHealth::Sticky { code: PANICKED }
    );
}

#[test]
fn a_panic_while_the_module_reads_its_build_or_ops_fails_the_load_as_internal() {
    common::install_services();
    for name in ["build", "ops"] {
        let dir = common::TempDir::new(name);
        let copy = common::private_copy(&dir);
        common::set("panic_in", name);
        let error = Module::load(&copy).expect_err("the load fails").0;
        common::clear("panic_in");
        expect_internal(name, &error);
    }
}

#[test]
fn a_sticky_provider_still_frees_what_it_holds() {
    common::install_services();
    let dir = common::TempDir::new("free-after");
    let fx = fixture(&dir);
    common::set("panic_in", "event");
    let _ = fx.p.event(&fx.ctx);
    common::clear("panic_in");
    let Fixture { p, ctx, buf, .. } = fx;
    let before = common::metric(common::FREED);
    p.free(&ctx, buf);
    // `free` reached the provider on the sticky module, which recorded it, and did not fail.
    assert_eq!(
        common::metric(common::FREED),
        before + 1,
        "free did not run"
    );
    assert_eq!(p.fault().expect("the panic").code, PANICKED);
    assert_eq!(p.fault().expect("the panic").class, ErrorClass::Internal);
}

/// The review of #1881, B2: a sticky provider still closes a context. Dropping the context runs
/// the module's `close_ctx`, which the provider records.
#[test]
fn a_sticky_provider_still_closes_a_context() {
    common::install_services();
    let dir = common::TempDir::new("close-after");
    let fx = fixture(&dir);
    common::set("panic_in", "event");
    let _ = fx.p.event(&fx.ctx);
    common::clear("panic_in");
    assert!(fx.p.fault().is_some(), "the provider is sticky");
    let Fixture { p, ctx, .. } = fx;
    let before = common::metric(common::CONTEXT_CLOSED);
    drop(ctx);
    assert_eq!(
        common::metric(common::CONTEXT_CLOSED),
        before + 1,
        "close_ctx did not run"
    );
    drop(p);
}

/// The review of #1881, B2: a sticky provider still runs a registrar's destructor.
#[test]
fn a_sticky_provider_still_closes_a_registrar() {
    common::install_services();
    let dir = common::TempDir::new("registrar-after");
    let fx = fixture(&dir);
    common::set("panic_in", "event");
    let _ = fx.p.event(&fx.ctx);
    common::clear("panic_in");
    assert!(fx.p.fault().is_some(), "the provider is sticky");
    let Fixture { p, registrar, .. } = fx;
    let before = common::metric(common::REGISTRAR_CLOSED);
    drop(registrar);
    assert_eq!(
        common::metric(common::REGISTRAR_CLOSED),
        before + 1,
        "the registrar's destructor did not run"
    );
    drop(p);
}

#[test]
fn dropping_a_registrar_whose_destructor_panics_makes_the_provider_sticky() {
    common::install_services();
    let dir = common::TempDir::new("registrar_close");
    let Fixture {
        p,
        witness,
        registrar,
        ..
    } = fixture(&dir);
    common::set("panic_in", "registrar_close");
    drop(registrar);
    common::clear("panic_in");
    expect_internal("registrar_close", &p.fault().expect("a recorded fault"));
    assert_eq!(p.health(&witness), DeviceHealth::Sticky { code: PANICKED });
    let later = p
        .event(&witness)
        .expect_err("a sticky provider refuses calls");
    assert_eq!(later.code, ErrorCode::DEVICE_STICKY);
}
