// SPDX-License-Identifier: MIT OR Apache-2.0

use syntropy_core::{DeviceId, NodeId};
use syntropy_kernels::{DeviceClass, DeviceFeatures, KernelProvider, OpenOptions};

use super::*;

/// A driver of a chosen version and device set, which counts its retains and releases.
#[derive(Default)]
struct Fake {
    present: bool,
    version: i32,
    init: Option<Fault>,
    devices: Vec<RawDevice>,
    retained: std::sync::Arc<std::sync::atomic::AtomicI32>,
    streams: stream::fake::Shared,
    /// At each memory free, from [`FreeWatch`]: the streams and events still live, and the
    /// primary context's retain count.
    frees: std::sync::Arc<std::sync::Mutex<Vec<(usize, usize, i32)>>>,
}

struct FakeContext(std::sync::Arc<std::sync::atomic::AtomicI32>);

impl Drop for FakeContext {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

impl Driver for Fake {
    type Context = FakeContext;
    fn present(&self) -> bool {
        self.present
    }
    fn init(&self) -> Result<(), Fault> {
        self.init.clone().map_or(Ok(()), Err)
    }
    fn version(&self) -> Result<i32, Fault> {
        Ok(self.version)
    }
    fn devices(&self) -> Result<Vec<RawDevice>, Fault> {
        Ok(self.devices.clone())
    }
    fn retain(&self, _ordinal: i32) -> Result<FakeContext, Fault> {
        self.retained
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(FakeContext(self.retained.clone()))
    }
}

impl StreamDriver for Fake {
    fn stream_api(
        &self,
        _: &FakeContext,
    ) -> Result<Box<dyn stream::StreamApi>, stream::StreamFault> {
        self.streams.lock().unwrap().retained = Some(self.retained.clone());
        Ok(Box::new(stream::fake::FakeStreams(self.streams.clone())))
    }
}

/// `memory`'s recorder, which hands out addresses, behind a [`FreeWatch`].
impl crate::memory::MemoryDriver for Fake {
    fn memory(
        &self,
        _: &FakeContext,
    ) -> Result<std::sync::Arc<dyn crate::memory::MemoryOps>, Fault> {
        Ok(std::sync::Arc::new(FreeWatch {
            rec: crate::memory::tests::Recorder::default(),
            streams: self.streams.clone(),
            retained: self.retained.clone(),
            frees: self.frees.clone(),
        }))
    }
}

/// The recorder's memory calls, noting at each free how many streams and events are live and
/// whether the primary context is still retained, so a test sees where the frees fall between
/// the streams' destruction and the context's release.
struct FreeWatch {
    rec: crate::memory::tests::Recorder,
    streams: stream::fake::Shared,
    retained: std::sync::Arc<std::sync::atomic::AtomicI32>,
    frees: std::sync::Arc<std::sync::Mutex<Vec<(usize, usize, i32)>>>,
}

impl FreeWatch {
    fn note(&self) {
        let (streams, events) = self.streams.lock().unwrap().live();
        let retained = self.retained.load(std::sync::atomic::Ordering::SeqCst);
        self.frees.lock().unwrap().push((streams, events, retained));
    }
}

impl crate::memory::MemoryOps for FreeWatch {
    fn device_alloc(&self, bytes: u64) -> Result<u64, Fault> {
        self.rec.device_alloc(bytes)
    }
    fn device_free(&self, addr: u64) -> Result<(), Fault> {
        self.note();
        self.rec.device_free(addr)
    }
    fn host_alloc(&self, bytes: u64) -> Result<u64, Fault> {
        self.rec.host_alloc(bytes)
    }
    fn host_free(&self, host: u64) -> Result<(), Fault> {
        self.note();
        self.rec.host_free(host)
    }
    fn host_register(&self, host: u64, bytes: u64) -> Result<(), Fault> {
        self.rec.host_register(host, bytes)
    }
    fn host_unregister(&self, host: u64) -> Result<(), Fault> {
        self.rec.host_unregister(host)
    }
    fn device_pointer(&self, host: u64) -> Result<u64, Fault> {
        self.rec.device_pointer(host)
    }
}

fn gpu(ordinal: i32, cc: (i32, i32)) -> RawDevice {
    RawDevice {
        ordinal,
        name: format!("GPU {ordinal}"),
        uuid: [u8::try_from(ordinal).unwrap(); 16],
        cc,
        pci: (0, 1 + u32::try_from(ordinal).unwrap(), 0),
        total: 24 << 30,
        free: 20 << 30,
        integrated: false,
        vmm: true,
        host_register: true,
        peer: false,
        numa: None,
    }
}

fn fake(version: i32, devices: Vec<RawDevice>) -> CudaProvider<Fake> {
    CudaProvider::with_driver(Fake {
        present: true,
        version,
        devices,
        ..Fake::default()
    })
}

#[test]
fn no_driver_library_means_no_device() {
    let p = CudaProvider::with_driver(Fake::default());
    assert_eq!(p.enumerate().unwrap(), Vec::new());
    assert_eq!(p.build().driver, None);
}

#[test]
fn a_driver_that_reports_no_device_gives_none() {
    let p = CudaProvider::with_driver(Fake {
        present: true,
        version: 13_050,
        init: Some(Fault {
            call: "cuInit",
            error: "CUDA_ERROR_NO_DEVICE".into(),
            no_device: true,
        }),
        ..Fake::default()
    });
    assert_eq!(p.enumerate().unwrap(), Vec::new());
}

#[test]
fn a_driver_below_580_is_skipped_with_driver_too_old() {
    // CUDA 12.9 is the newest API before driver branch 580.
    for version in [12_090, 12_000, 11_080] {
        let err = fake(version, vec![gpu(0, (8, 9))]).enumerate().unwrap_err();
        assert_eq!(err.code, codes::DRIVER_TOO_OLD, "{version}");
        assert_eq!(
            err.code,
            syntropy_core::ErrorCode::new(syntropy_core::Domain::Kernels, 5)
        );
        assert_eq!(err.class, ErrorClass::Unsupported);
    }
    for version in [13_000, 13_040, 13_050] {
        assert_eq!(
            fake(version, vec![gpu(0, (8, 9))])
                .enumerate()
                .unwrap()
                .len(),
            1
        );
    }
}

#[test]
fn each_device_has_the_sm_of_its_compute_capability() {
    let ccs = [
        (7, 5),
        (8, 0),
        (8, 6),
        (8, 9),
        (9, 0),
        (10, 0),
        (12, 0),
        (12, 1),
    ];
    let devices = ccs
        .iter()
        .enumerate()
        .map(|(i, &cc)| gpu(i32::try_from(i).unwrap(), cc));
    let records = fake(13_050, devices.collect()).enumerate().unwrap();
    let sms: Vec<_> = records.iter().map(|r| r.class).collect();
    let want: Vec<_> = [75, 80, 86, 89, 90, 100, 120, 121]
        .into_iter()
        .map(|sm| DeviceClass::Cuda { sm })
        .collect();
    assert_eq!(sms, want);
}

#[test]
fn a_device_below_sm_75_is_not_claimed_and_local_indices_stay_dense() {
    let records = fake(13_050, vec![gpu(0, (6, 1)), gpu(1, (8, 6)), gpu(2, (7, 0))])
        .enumerate()
        .unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].local, 0);
    assert_eq!(records[0].name, "GPU 1");
}

#[test]
fn feature_bits_follow_the_architecture_and_the_attributes() {
    let bits = |cc, f: fn(&mut RawDevice)| {
        let mut d = gpu(0, cc);
        f(&mut d);
        fake(13_050, vec![d]).enumerate().unwrap()[0].features
    };
    let ampere = bits((8, 6), |_| {});
    assert!(ampere.contains(DeviceFeatures::BF16_MMA | DeviceFeatures::INT8_TC));
    assert!(!ampere.intersects(DeviceFeatures::FP8_TC | DeviceFeatures::FP4_TC));
    assert!(bits((8, 9), |_| {}).contains(DeviceFeatures::FP8_TC));
    assert!(!bits((8, 9), |_| {}).contains(DeviceFeatures::FP4_TC));
    assert!(bits((12, 0), |_| {}).contains(DeviceFeatures::FP8_TC | DeviceFeatures::FP4_TC));
    assert!(!bits((7, 5), |_| {}).contains(DeviceFeatures::BF16_MMA));
    // The memory bits are the attributes, one by one.
    assert!(ampere.contains(DeviceFeatures::VMM | DeviceFeatures::HOST_REGISTER));
    assert!(!bits((8, 6), |d| d.vmm = false).contains(DeviceFeatures::VMM));
    assert!(!bits((8, 6), |d| d.host_register = false).contains(DeviceFeatures::HOST_REGISTER));
    assert!(!ampere.contains(DeviceFeatures::P2P));
    assert!(bits((8, 6), |d| d.peer = true).contains(DeviceFeatures::P2P));
    assert!(bits((8, 7), |d| d.integrated = true).contains(DeviceFeatures::UNIFIED));
}

#[test]
fn stream_mem_ops_is_never_set_at_enumerate() {
    for cc in [(7, 5), (8, 6), (8, 9), (12, 0)] {
        let r = fake(13_050, vec![gpu(0, cc)]).enumerate().unwrap();
        assert!(!r[0].features.contains(DeviceFeatures::STREAM_MEM_OPS));
    }
}

/// RFC-0040 §7.1: the stream memory operations are decided by COMP-E4-T14's functional probe,
/// never by the deprecated attribute, which reads 0 on WDDM even where the operations work
/// (OQ-KB-13). No source file of this crate names it.
#[test]
fn no_code_path_reads_the_stream_mem_ops_attribute() {
    let needle = ["CAN_USE_STREAM", "_MEM_OPS"].concat();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut seen = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "rs") {
            seen += 1;
            let text = std::fs::read_to_string(&path).unwrap();
            assert!(!text.contains(&needle), "{} names it", path.display());
        }
    }
    assert!(seen >= 3, "only {seen} source files read");
}

#[test]
fn the_record_carries_identity_memory_and_kind() {
    let mut d = gpu(3, (8, 6));
    d.numa = Some(1);
    let r = fake(13_050, vec![d.clone()]).enumerate().unwrap().remove(0);
    assert_eq!(r.uuid, Some(d.uuid));
    let pci = r.pci.unwrap();
    assert_eq!(
        (pci.domain, pci.bus, pci.device, pci.function),
        (0, 4, 0, 0)
    );
    assert_eq!(r.kind, syntropy_core::resource::DeviceKind::DiscreteGpu);
    assert_eq!(r.memory.total, Bytes(24 << 30));
    assert_eq!(r.memory.free, Bytes(20 << 30));
    assert_eq!(r.numa, Some(1));
    let mut i = gpu(0, (8, 7));
    i.integrated = true;
    let r = fake(13_050, vec![i]).enumerate().unwrap().remove(0);
    assert_eq!(r.kind, syntropy_core::resource::DeviceKind::IntegratedGpu);
    assert!(r.memory.unified);
}

#[test]
fn open_retains_the_primary_context_and_drop_releases_it() {
    let p = fake(13_050, vec![gpu(0, (6, 1)), gpu(1, (8, 6))]);
    let retained = p.driver.retained.clone();
    let opts = OpenOptions {
        streams: syntropy_kernels::StreamPlan {
            compute: 1,
            copy_in: 1,
            copy_out: 1,
            collective: 0,
        },
        doorbell_slots: 8,
        capture: true,
        deterministic: false,
        power: syntropy_kernels::PowerHint::Mains,
    };
    let id = |ordinal| DeviceId {
        node: NodeId([0; 32]),
        ordinal,
    };
    let ctx = p.open(id(0), &opts).unwrap();
    // Local index 0 is the driver's ordinal 1: ordinal 0 is below sm_75.
    assert_eq!(
        ctx.downcast_ref::<CudaDevice<FakeContext>>()
            .unwrap()
            .ordinal,
        1
    );
    assert_eq!(retained.load(std::sync::atomic::Ordering::SeqCst), 1);
    drop(ctx);
    assert_eq!(retained.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert!(p.open(id(1), &opts).is_err());
}

/// `open` creates the stream plan in the retained context, and closing the device destroys every
/// stream and event while the context is still retained, then releases it (RFC-0040 §4.1, §4.3).
#[test]
fn open_creates_the_plan_and_drop_destroys_it_before_the_context() {
    use stream::fake::Call;
    let p = fake(13_050, vec![gpu(0, (8, 9))]);
    p.driver.streams.lock().unwrap().range = (0, -5);
    let opts = OpenOptions {
        streams: syntropy_kernels::StreamPlan {
            compute: 1,
            copy_in: 1,
            copy_out: 1,
            collective: 1,
        },
        doorbell_slots: 8,
        capture: true,
        deterministic: false,
        power: syntropy_kernels::PowerHint::Mains,
    };
    let id = DeviceId {
        node: NodeId([0; 32]),
        ordinal: 0,
    };
    let ctx = p.open(id, &opts).unwrap();
    let log = || p.driver.streams.lock().unwrap().log.clone();
    assert_eq!(
        log(),
        [-5, -3, -2, -4].map(Call::CreateStream),
        "compute, copy-in, copy-out, collective"
    );
    let s = p.stream(&ctx, StreamPriority::Compute).unwrap();
    let e = p.event(&ctx).unwrap();
    let t = p.timing_event(&ctx).unwrap();
    p.record(&ctx, &s, &e).unwrap();
    p.wait(&ctx, &s, &t).unwrap();
    assert!(!p.poll(&ctx, &e).unwrap());
    assert_eq!(
        p.elapsed(&ctx, &e, &t),
        None,
        "`event`'s events have no timing"
    );
    drop(ctx);
    let retained = p.driver.retained.load(std::sync::atomic::Ordering::SeqCst);
    assert_eq!(retained, 0);
    let destroyed: Vec<bool> = log()
        .into_iter()
        .filter_map(|c| match c {
            Call::DestroyStream(_, held) | Call::DestroyEvent(_, held) => Some(held),
            _ => None,
        })
        .collect();
    assert_eq!(
        destroyed, [true; 6],
        "4 streams and 2 events, all before the release"
    );
    assert_eq!(p.driver.streams.lock().unwrap().live(), (0, 0));
}

/// Closing a device drains and destroys its streams, then frees the buffers the tier manager
/// left allocated, then releases the primary context (`CudaDevice`'s field order): no copy still
/// queued touches freed memory, and nothing is freed in a released context.
#[test]
fn closing_a_device_destroys_its_streams_before_freeing_its_memory() {
    use syntropy_kernels::{AllocRequest, MemKind};
    let p = fake(13_050, vec![gpu(0, (8, 9))]);
    let opts = OpenOptions {
        streams: syntropy_kernels::StreamPlan {
            compute: 1,
            copy_in: 1,
            copy_out: 0,
            collective: 0,
        },
        doorbell_slots: 8,
        capture: true,
        deterministic: false,
        power: syntropy_kernels::PowerHint::Mains,
    };
    let id = DeviceId {
        node: NodeId([0; 32]),
        ordinal: 0,
    };
    let ctx = p.open(id, &opts).unwrap();
    for kind in [MemKind::Device, MemKind::HostMapped] {
        let req = AllocRequest {
            bytes: Bytes(4096),
            kind,
            align: Bytes(256),
            reserve_only: false,
        };
        p.alloc(&ctx, req).unwrap();
    }
    p.event(&ctx).unwrap();
    assert_eq!(p.driver.streams.lock().unwrap().live(), (2, 1));
    p.driver.streams.lock().unwrap().log.clear();
    drop(ctx);
    assert_eq!(
        *p.driver.frees.lock().unwrap(),
        [(0, 0, 1); 2],
        "both buffers freed after every stream and event was destroyed, before the release"
    );
    let log = p.driver.streams.lock().unwrap().log.clone();
    let syncs = log
        .iter()
        .position(|c| matches!(c, stream::fake::Call::Sync(_)));
    let destroys = log.iter().position(|c| {
        matches!(
            c,
            stream::fake::Call::DestroyStream(..) | stream::fake::Call::DestroyEvent(..)
        )
    });
    assert!(
        syncs.unwrap() < destroys.unwrap(),
        "drained, then destroyed"
    );
    let drained = log
        .iter()
        .filter(|c| matches!(c, stream::fake::Call::Sync(_)))
        .count();
    assert_eq!(drained, 2, "every stream drained");
    assert_eq!(
        p.driver.retained.load(std::sync::atomic::Ordering::SeqCst),
        0
    );
}

/// `copy` and `copy_batch` take their addresses from the device's memory table: a slice of a
/// buffer from `alloc` reaches the driver at the buffer's address plus its offset. A slice of
/// another device's buffer, a freed buffer, a range past a buffer's end or a length mismatch is
/// refused without reaching the driver, and a refused pair in a batch enqueues none of it.
#[test]
fn copies_resolve_slices_through_the_devices_memory() {
    use stream::fake::Call;
    use syntropy_kernels::{AllocRequest, MemKind};
    let p = fake(13_050, vec![gpu(0, (8, 9)), gpu(1, (8, 9))]);
    let opts = OpenOptions {
        streams: syntropy_kernels::StreamPlan {
            compute: 1,
            copy_in: 0,
            copy_out: 0,
            collective: 0,
        },
        doorbell_slots: 8,
        capture: true,
        deterministic: false,
        power: syntropy_kernels::PowerHint::Mains,
    };
    let id = |ordinal| DeviceId {
        node: NodeId([0; 32]),
        ordinal,
    };
    let ctx = p.open(id(0), &opts).unwrap();
    let other = p.open(id(1), &opts).unwrap();
    let s = p.stream(&ctx, StreamPriority::Compute).unwrap();
    let alloc = |ctx: &DeviceCtx, kind| {
        p.alloc(
            ctx,
            AllocRequest {
                bytes: Bytes(4096),
                kind,
                align: Bytes(256),
                reserve_only: false,
            },
        )
        .unwrap()
    };
    let a = alloc(&ctx, MemKind::Device);
    let b = alloc(&ctx, MemKind::HostMapped);
    let elsewhere = alloc(&other, MemKind::Device);
    let slice = |buf: &DeviceBuffer, offset, bytes| BufferSlice {
        buffer: buf.handle,
        offset: Bytes(offset),
        bytes: Bytes(bytes),
    };
    let log = || std::mem::take(&mut p.driver.streams.lock().unwrap().log);
    log();

    p.copy(&ctx, &s, slice(&a, 64, 128), slice(&b, 1024, 128))
        .unwrap();
    let raw = p
        .driver
        .streams
        .lock()
        .unwrap()
        .streams
        .keys()
        .copied()
        .next()
        .unwrap();
    assert_eq!(
        log(),
        [Call::Copy {
            s: raw,
            dst: b.addr + 1024,
            src: a.addr + 64,
            bytes: 128,
        }]
    );
    let pairs = [
        (slice(&a, 0, 4096), slice(&b, 0, 4096)),
        (slice(&b, 4000, 96), slice(&a, 0, 96)),
    ];
    p.copy_batch(&ctx, &s, &pairs).unwrap();
    assert_eq!(
        log(),
        [
            Call::Copy {
                s: raw,
                dst: b.addr,
                src: a.addr,
                bytes: 4096,
            },
            Call::Copy {
                s: raw,
                dst: a.addr,
                src: b.addr + 4000,
                bytes: 96,
            },
        ],
        "the fake driver has no batched entry point, so the loop path runs"
    );

    let refused = |src, dst| {
        let err = p.copy(&ctx, &s, src, dst).unwrap_err();
        assert_eq!(err.class, ErrorClass::Internal);
        let batch = [(slice(&a, 0, 8), slice(&a, 8, 8)), (src, dst)];
        assert!(p.copy_batch(&ctx, &s, &batch).is_err());
        assert_eq!(log(), [], "nothing reaches the driver");
    };
    refused(slice(&elsewhere, 0, 64), slice(&a, 0, 64));
    refused(slice(&a, 0, 64), slice(&elsewhere, 0, 64));
    refused(slice(&a, 4033, 64), slice(&b, 0, 64));
    refused(slice(&a, 0, 64), slice(&b, 0, 32));
    let freed = slice(&b, 0, 64);
    p.free(&ctx, b);
    refused(slice(&a, 0, 64), freed);
    refused(freed, slice(&a, 0, 64));
    // The other device's buffer is refused by this device and accepted by its own.
    let s1 = p.stream(&other, StreamPriority::Compute).unwrap();
    p.copy(
        &other,
        &s1,
        slice(&elsewhere, 0, 64),
        slice(&elsewhere, 64, 64),
    )
    .unwrap();
    assert_eq!(log().len(), 1);
}

#[test]
fn the_build_records_the_driver_version() {
    assert_eq!(
        fake(13_050, vec![]).build().driver.as_deref(),
        Some("CUDA 13.5")
    );
}

/// On a host with the NVIDIA driver, every CUDA device enumerates with the `sm` that
/// `nvidia-smi` reports for it (RFC-0040 §7.1; COMP-E4-T01). Needs a GPU, so it runs with
/// `--ignored`; on a host without the driver it checks that enumeration gives no device.
#[test]
#[ignore = "needs the NVIDIA driver and a GPU"]
fn the_real_driver_enumerates_with_the_right_sm() {
    let p = CudaProvider::new();
    let records = p.enumerate().unwrap();
    eprintln!("build: {:?}", p.build());
    for r in &records {
        eprintln!("{r:?}");
    }
    let smi = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap,uuid", "--format=csv,noheader"])
        .output();
    let Ok(smi) = smi else {
        assert!(records.is_empty(), "devices but no nvidia-smi");
        return;
    };
    let text = String::from_utf8_lossy(&smi.stdout);
    let mut want: Vec<(u16, String)> = text
        .lines()
        .filter_map(|l| {
            let (cc, uuid) = l.split_once(", ")?;
            let (major, minor) = cc.trim().split_once('.')?;
            let sm = major.parse::<u16>().ok()? * 10 + minor.parse::<u16>().ok()?;
            Some((sm, uuid.trim().trim_start_matches("GPU-").replace('-', "")))
        })
        .filter(|(sm, _)| *sm >= describe::MIN_SM)
        .collect();
    let mut got: Vec<(u16, String)> = records
        .iter()
        .map(|r| {
            let DeviceClass::Cuda { sm } = r.class else {
                panic!("not a CUDA class: {:?}", r.class)
            };
            let uuid = r.uuid.unwrap().iter().map(|b| format!("{b:02x}")).collect();
            (sm, uuid)
        })
        .collect();
    want.sort();
    got.sort();
    assert!(
        !got.is_empty(),
        "nvidia-smi lists GPUs but enumerate found none"
    );
    assert_eq!(got, want);
}
