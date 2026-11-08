// SPDX-License-Identifier: MIT OR Apache-2.0

//! Capture over the fake driver: the provider's `begin_capture`, `end_capture`, `replay`,
//! `signal_host` and `wait_host` as the driver sees them, with no GPU. The same step on a real
//! device is `gpu_tests`.

use syntropy_core::runner::DegradeFlags;
use syntropy_core::{Bytes, DeviceId, ErrorClass, NodeId};
use syntropy_kernels::capture::capture_degrade;
use syntropy_kernels::error::codes;
use syntropy_kernels::{
    AllocRequest, BufferSlice, DeviceBuffer, DeviceCtx, HostFlag, KernelProvider, MemKind,
    OpenOptions, PowerHint, Stream, StreamPlan, StreamPriority, WaitCmp,
};

use crate::CudaProvider;
use crate::stream::fake::Call;
use crate::tests::{Fake, fake, gpu};

const MIB: u64 = 1 << 20;

fn opts(capture: bool) -> OpenOptions {
    OpenOptions {
        streams: StreamPlan {
            compute: 1,
            copy_in: 1,
            copy_out: 0,
            collective: 0,
        },
        doorbell_slots: 8,
        capture,
        deterministic: false,
        power: PowerHint::Mains,
    }
}

fn dev0() -> DeviceId {
    DeviceId {
        node: NodeId([0; 32]),
        ordinal: 0,
    }
}

/// A device with the buffers of the synthetic step: the descriptor mirror (`HostMapped`), the
/// device's descriptor copy and output, and one doorbell page with the flags `F` (host → device)
/// and `DC` (device → host), each on its own 64-byte line.
struct Rig {
    p: CudaProvider<Fake>,
    ctx: DeviceCtx,
    compute: Stream,
    copy: Stream,
    mirror: DeviceBuffer,
    desc: DeviceBuffer,
    out: DeviceBuffer,
    bell: DeviceBuffer,
    f: HostFlag,
    dc: HostFlag,
}

fn alloc(p: &CudaProvider<Fake>, ctx: &DeviceCtx, kind: MemKind, bytes: u64) -> DeviceBuffer {
    p.alloc(
        ctx,
        AllocRequest {
            bytes: Bytes(bytes),
            kind,
            align: Bytes(256),
            reserve_only: false,
        },
    )
    .unwrap()
}

fn whole(b: &DeviceBuffer) -> BufferSlice {
    BufferSlice {
        buffer: b.handle,
        offset: Bytes(0),
        bytes: b.bytes,
    }
}

fn rig(capture: bool) -> Rig {
    let p = fake(13_050, vec![gpu(0, (8, 9))]);
    p.driver.streams.lock().unwrap().free = 10 << 30;
    let ctx = p.open(dev0(), &opts(capture)).unwrap();
    let compute = p.stream(&ctx, StreamPriority::Compute).unwrap();
    let copy = p.stream(&ctx, StreamPriority::CopyIn).unwrap();
    let mirror = alloc(&p, &ctx, MemKind::HostMapped, 4096);
    let desc = alloc(&p, &ctx, MemKind::Device, 4096);
    let out = alloc(&p, &ctx, MemKind::Device, 4096);
    let bell = alloc(&p, &ctx, MemKind::HostMapped, 4096);
    let f = p.host_flag(&ctx, &bell, 0).unwrap();
    let dc = p.host_flag(&ctx, &bell, 64).unwrap();
    Rig {
        p,
        ctx,
        compute,
        copy,
        mirror,
        desc,
        out,
        bell,
        f,
        dc,
    }
}

impl Rig {
    /// The synthetic decode step on `s`: the descriptor copy, the wait for `F` with its reset,
    /// the step's work (a copy), and the `DC` signal (RFC-0040 §4.2, §4.4).
    fn step(&self, s: &Stream) {
        let p = &self.p;
        p.copy(&self.ctx, s, whole(&self.mirror), whole(&self.desc))
            .unwrap();
        p.wait_host(&self.ctx, s, self.f, 1, WaitCmp::Eq, Some(0))
            .unwrap();
        p.copy(&self.ctx, s, whole(&self.desc), whole(&self.out))
            .unwrap();
        p.signal_host(&self.ctx, s, self.dc, 1).unwrap();
    }

    fn calls(&self) -> Vec<Call> {
        self.p.driver.streams.lock().unwrap().log.clone()
    }

    fn submissions(&self, s: &Stream) -> u64 {
        self.p.submissions(&self.ctx, s).unwrap()
    }

    fn flag_dev(&self, flag: HostFlag) -> u64 {
        self.p
            .device(&self.ctx)
            .unwrap()
            .memory
            .flag_addrs(flag)
            .unwrap()
            .0
    }
}

/// The step's nodes, as the driver records them: what a replay runs.
fn nodes(r: &Rig) -> Vec<Call> {
    let st = r.p.driver.streams.lock().unwrap();
    assert_eq!(st.execs.len(), 1);
    st.execs.values().next().unwrap().clone()
}

/// AC1, on the fake: a captured step replays as one compute-stream submission, whatever the
/// step holds; copy-stream submissions are counted apart.
#[test]
fn a_captured_step_is_one_compute_submission_per_replay() {
    let r = rig(true);
    r.p.begin_capture(&r.ctx, &r.compute).unwrap();
    r.step(&r.compute);
    let g = r.p.end_capture(&r.ctx, &r.compute).unwrap();
    // Capturing submits nothing; the upload is one submission, made once at `Warming`.
    assert_eq!(r.submissions(&r.compute), 1);

    let (f, dc) = (r.flag_dev(r.f), r.flag_dev(r.dc));
    let raw =
        r.p.device(&r.ctx)
            .unwrap()
            .streams
            .raw_stream(&r.compute)
            .unwrap();
    let n = nodes(&r);
    assert!(matches!(n[0], Call::Copy { bytes: 4096, .. }), "{n:?}");
    assert_eq!(
        n[1..3],
        [
            Call::WaitValue {
                s: raw,
                addr: f,
                value: 1,
                eq: true
            },
            Call::WriteValue {
                s: raw,
                addr: f,
                value: 0
            },
        ]
    );
    assert!(matches!(n[3], Call::Copy { .. }));
    assert_eq!(
        n[4],
        Call::WriteValue {
            s: raw,
            addr: dc,
            value: 1
        }
    );
    assert_eq!(n.len(), 5);

    for step in 1..=10 {
        let before = r.submissions(&r.compute);
        r.p.replay(&r.ctx, &r.compute, &g).unwrap();
        assert_eq!(r.submissions(&r.compute) - before, 1, "step {step}");
        // A copied miss on the copy stream is the copy stream's submission, not compute's.
        r.p.copy(&r.ctx, &r.copy, whole(&r.mirror), whole(&r.out))
            .unwrap();
        assert_eq!(r.submissions(&r.copy), step);
    }
    assert_eq!(r.submissions(&r.compute), 11);
    let by_stream = r.p.device(&r.ctx).unwrap().streams.submissions_by_stream();
    assert_eq!(
        by_stream,
        [(StreamPriority::Compute, 11), (StreamPriority::CopyIn, 10)]
    );
}

/// AC3, on the fake: with the kill switch, `begin_capture` is `CaptureFailed`, the runner is
/// degraded with `CaptureOff`, and the step's operations issued directly are exactly the
/// captured graph's nodes, in order. On a GPU, `gpu_tests` compares the bytes.
#[test]
fn the_kill_switch_issues_the_graphs_nodes_as_direct_launches() {
    let captured = rig(true);
    captured
        .p
        .begin_capture(&captured.ctx, &captured.compute)
        .unwrap();
    captured.step(&captured.compute);
    captured
        .p
        .end_capture(&captured.ctx, &captured.compute)
        .unwrap();
    let graph = nodes(&captured);

    let eager = rig(false);
    let e = eager
        .p
        .begin_capture(&eager.ctx, &eager.compute)
        .unwrap_err();
    assert_eq!(
        (e.code, e.class),
        (codes::CAPTURE_FAILED, ErrorClass::Unsupported),
        "{e}"
    );
    assert_eq!(capture_degrade(&e), Some(DegradeFlags::CAPTURE_OFF));
    let before = eager.calls().len();
    eager.step(&eager.compute);
    let direct = eager.calls()[before..].to_vec();

    // Both fakes hand out the same handles and addresses for the same calls in the same order.
    assert_eq!(direct, graph);
    // Per-op launches: one submission per operation.
    assert_eq!(eager.submissions(&eager.compute), graph.len() as u64);
    assert!(
        !eager
            .calls()
            .iter()
            .any(|c| matches!(c, Call::BeginCapture(_) | Call::Instantiate(..))),
        "the kill switch reached the driver's capture calls"
    );
}

#[test]
fn a_refused_capture_or_instantiation_is_capture_failed_and_leaks_nothing() {
    let r = rig(true);
    r.p.driver.streams.lock().unwrap().fail = Some(("cuStreamBeginCapture_v2", 900));
    let e = r.p.begin_capture(&r.ctx, &r.compute).unwrap_err();
    assert_eq!(e.code, codes::CAPTURE_FAILED, "{e}");
    // The stream is not left in capture mode: the step can run eagerly.
    r.step(&r.compute);
    assert_eq!(r.submissions(&r.compute), 5);

    for call in [
        "cuStreamEndCapture",
        "cuGraphInstantiateWithFlags",
        "cuGraphUpload",
    ] {
        r.p.begin_capture(&r.ctx, &r.compute).unwrap();
        r.step(&r.compute);
        r.p.driver.streams.lock().unwrap().fail = Some((call, 900));
        let e = r.p.end_capture(&r.ctx, &r.compute).unwrap_err();
        assert_eq!(
            (e.code, e.class),
            (codes::CAPTURE_FAILED, ErrorClass::Unsupported),
            "{call}: {e}"
        );
        assert_eq!(capture_degrade(&e), Some(DegradeFlags::CAPTURE_OFF));
        let st = r.p.driver.streams.lock().unwrap();
        assert!(st.capturing.is_empty(), "{call}: still capturing");
        assert!(st.graphs.is_empty(), "{call}: graph leaked");
        assert!(st.execs.is_empty(), "{call}: executable leaked");
    }
    assert_eq!(r.p.graph_stats(&r.ctx).unwrap().instances, 0);
}

#[test]
fn a_sticky_error_during_instantiation_is_device_sticky() {
    let r = rig(true);
    r.p.begin_capture(&r.ctx, &r.compute).unwrap();
    r.p.driver.streams.lock().unwrap().fail = Some(("cuGraphInstantiateWithFlags", 700));
    let e = r.p.end_capture(&r.ctx, &r.compute).unwrap_err();
    assert_eq!(
        (e.code, e.class),
        (codes::DEVICE_STICKY, ErrorClass::Device)
    );
    assert_eq!(capture_degrade(&e), None);
}

/// AC4, on the fake: each instance's device memory is the fall in free memory across its
/// instantiation and upload; releasing one gives it back.
#[test]
fn graph_stats_report_each_instance_and_its_memory() {
    let r = rig(true);
    {
        let mut st = r.p.driver.streams.lock().unwrap();
        st.instantiate_bytes = 3 * MIB;
        st.upload_bytes = MIB;
    }
    let mut graphs = Vec::new();
    for _ in 0..2 {
        r.p.begin_capture(&r.ctx, &r.compute).unwrap();
        r.step(&r.compute);
        graphs.push(r.p.end_capture(&r.ctx, &r.compute).unwrap());
    }
    let stats = r.p.graph_stats(&r.ctx).unwrap();
    assert_eq!(stats.instances, 2);
    assert_eq!(stats.per_instance, [4 * MIB, 4 * MIB]);
    assert_eq!(stats.device_bytes, 8 * MIB);

    r.p.release_graph(&r.ctx, graphs.remove(0)).unwrap();
    let stats = r.p.graph_stats(&r.ctx).unwrap();
    assert_eq!((stats.instances, stats.device_bytes), (1, 4 * MIB));
    // A released graph cannot be replayed.
    let stale = syntropy_kernels::CapturedGraph {
        handle: graphs[0].handle - 1,
    };
    assert!(r.p.replay(&r.ctx, &r.compute, &stale).is_err());
    r.p.replay(&r.ctx, &r.compute, &graphs[0]).unwrap();
}

/// RFC-0012 §7.1: a helper thread instantiates; the dispatcher adopts the executable between
/// steps and replays it.
#[test]
fn a_helper_thread_instantiates_and_the_dispatcher_adopts() {
    let r = rig(true);
    r.p.driver.streams.lock().unwrap().instantiate_bytes = 2 * MIB;
    r.p.begin_capture(&r.ctx, &r.compute).unwrap();
    r.step(&r.compute);
    let pending = r.p.end_capture_pending(&r.ctx, &r.compute).unwrap();
    let inst = r.p.instantiator(&r.ctx).unwrap();
    let ready = std::thread::spawn(move || inst.instantiate(pending))
        .join()
        .unwrap()
        .unwrap();
    assert_eq!(ready.bytes(), 2 * MIB);
    assert!(r.p.driver.streams.lock().unwrap().graphs.is_empty());
    let g = r.p.adopt(&r.ctx, &r.compute, ready).unwrap();
    let before = r.submissions(&r.compute);
    r.p.replay(&r.ctx, &r.compute, &g).unwrap();
    assert_eq!(r.submissions(&r.compute) - before, 1);

    // A ready graph dropped before adoption destroys its executable.
    r.p.begin_capture(&r.ctx, &r.compute).unwrap();
    let pending = r.p.end_capture_pending(&r.ctx, &r.compute).unwrap();
    drop(
        r.p.instantiator(&r.ctx)
            .unwrap()
            .instantiate(pending)
            .unwrap(),
    );
    assert_eq!(r.p.driver.streams.lock().unwrap().execs.len(), 1);

    // Another device's instantiator refuses the graph, which is then destroyed.
    let other = rig(true);
    r.p.begin_capture(&r.ctx, &r.compute).unwrap();
    let pending = r.p.end_capture_pending(&r.ctx, &r.compute).unwrap();
    let e = other
        .p
        .instantiator(&other.ctx)
        .unwrap()
        .instantiate(pending)
        .unwrap_err();
    assert_eq!(e.class, ErrorClass::Internal, "{e}");
    assert!(r.p.driver.streams.lock().unwrap().graphs.is_empty());
}

#[test]
fn capture_calls_out_of_order_are_refused() {
    let r = rig(true);
    assert!(r.p.end_capture(&r.ctx, &r.compute).is_err());
    r.p.begin_capture(&r.ctx, &r.compute).unwrap();
    let e = r.p.begin_capture(&r.ctx, &r.compute).unwrap_err();
    assert_eq!(e.class, ErrorClass::Internal, "{e}");
    r.p.end_capture(&r.ctx, &r.compute).unwrap();
    let made_up = Stream { handle: 0xdead };
    assert!(r.p.begin_capture(&r.ctx, &made_up).is_err());
    let foreign = syntropy_kernels::CapturedGraph { handle: 1 << 40 };
    assert!(r.p.replay(&r.ctx, &r.compute, &foreign).is_err());
}

#[test]
fn a_flag_outside_the_doorbell_pages_is_refused_before_the_driver() {
    let r = rig(true);
    let before = r.calls().len();
    let off_page = HostFlag {
        page: r.f.page,
        slot: 1024,
    };
    let no_page = HostFlag { page: 999, slot: 0 };
    for flag in [off_page, no_page] {
        assert!(r.p.signal_host(&r.ctx, &r.compute, flag, 1).is_err());
        assert!(
            r.p.wait_host(&r.ctx, &r.compute, flag, 1, WaitCmp::Eq, Some(0))
                .is_err()
        );
    }
    assert_eq!(r.calls().len(), before);
    assert_eq!(r.submissions(&r.compute), 0);
}

/// Doorbell pages are numbered in allocation order and never renumbered (`memory` docs).
#[test]
fn doorbell_flags_name_slots_of_host_mapped_pages() {
    let r = rig(true);
    // The rig's descriptor mirror is page 0 and the doorbell buffer page 1.
    assert_eq!(r.f, HostFlag { page: 1, slot: 0 });
    assert_eq!(r.dc, HostFlag { page: 1, slot: 16 });
    let big = alloc(&r.p, &r.ctx, MemKind::HostMapped, 3 * 4096);
    assert_eq!(
        r.p.host_flag(&r.ctx, &big, 2 * 4096 + 8),
        Some(HostFlag { page: 4, slot: 2 })
    );
    assert_eq!(r.p.host_flag(&r.ctx, &big, 3), None, "unaligned");
    assert_eq!(r.p.host_flag(&r.ctx, &big, 3 * 4096), None, "past the end");
    assert_eq!(r.p.host_flag(&r.ctx, &r.desc, 0), None, "device memory");
    let m = &r.p.device(&r.ctx).unwrap().memory;
    let (dev, host) = m.flag_addrs(HostFlag { page: 4, slot: 2 }).unwrap();
    let base = r.p.resolve(&r.ctx, whole(&big)).unwrap();
    assert_eq!((dev, host), (base.device + 8200, base.host.unwrap() + 8200));
    // Freeing a buffer removes its pages and renames no other flag.
    let bell = DeviceBuffer {
        addr: r.bell.addr,
        bytes: r.bell.bytes,
        kind: r.bell.kind,
        handle: r.bell.handle,
    };
    r.p.free(&r.ctx, bell);
    assert_eq!(m.flag_addrs(r.f), None);
    assert!(m.flag_addrs(HostFlag { page: 4, slot: 2 }).is_some());
    let next = alloc(&r.p, &r.ctx, MemKind::HostMapped, 4096);
    assert_eq!(
        r.p.host_flag(&r.ctx, &next, 0),
        Some(HostFlag { page: 5, slot: 0 })
    );
}

/// Closing the device drains its streams, destroys its executables while the context is still
/// retained, and only then releases the context.
#[test]
fn closing_a_device_destroys_its_graphs_before_the_context() {
    let r = rig(true);
    r.p.begin_capture(&r.ctx, &r.compute).unwrap();
    r.step(&r.compute);
    r.p.end_capture(&r.ctx, &r.compute).unwrap();
    let shared = r.p.driver.streams.clone();
    let retained = r.p.driver.retained.clone();
    let Rig { p, ctx, .. } = r;
    drop(ctx);
    let st = shared.lock().unwrap();
    assert!(st.execs.is_empty());
    let last_stream = st
        .log
        .iter()
        .rposition(|c| matches!(c, Call::DestroyStream(..)))
        .unwrap();
    let exec = st
        .log
        .iter()
        .position(|c| matches!(c, Call::DestroyExec(..)))
        .unwrap();
    assert!(exec > last_stream, "{:?}", st.log);
    assert!(matches!(st.log[exec], Call::DestroyExec(_, true)));
    assert_eq!(retained.load(std::sync::atomic::Ordering::SeqCst), 0);
    drop(p);
}

/// ADR-0404 item 7: every capture this crate begins is thread-local; no other mode is named
/// outside the tests (the GPU test uses global mode once, as the control).
#[test]
fn capture_is_thread_local_and_no_other_mode_is_used() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let system = std::fs::read_to_string(dir.join("capture/system.rs")).unwrap();
    assert!(system.contains("CU_STREAM_CAPTURE_MODE_THREAD_LOCAL"));
    let mut files = vec![dir.join("lib.rs"), dir.join("capture.rs")];
    for sub in ["capture", "stream", "memory"] {
        for e in std::fs::read_dir(dir.join(sub)).unwrap() {
            files.push(e.unwrap().path());
        }
    }
    for path in files {
        if path.to_string_lossy().contains("tests") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        for mode in ["MODE_GLOBAL", "MODE_RELAXED"] {
            let needle = ["CU_STREAM_CAPTURE_", mode].concat();
            assert!(!text.contains(&needle), "{} names {needle}", path.display());
        }
    }
}
