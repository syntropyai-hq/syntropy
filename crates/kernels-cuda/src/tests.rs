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

/// Until the memory module issues buffers, a slice names no buffer of the device, and `copy`
/// refuses it without reaching the driver.
#[test]
fn a_copy_of_an_unknown_buffer_is_refused() {
    let p = fake(13_050, vec![gpu(0, (8, 9))]);
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
    let id = DeviceId {
        node: NodeId([0; 32]),
        ordinal: 0,
    };
    let ctx = p.open(id, &opts).unwrap();
    let s = p.stream(&ctx, StreamPriority::Compute).unwrap();
    let slice = |buffer| BufferSlice {
        buffer,
        offset: Bytes(0),
        bytes: Bytes(64),
    };
    p.driver.streams.lock().unwrap().log.clear();
    let err = p.copy(&ctx, &s, slice(1), slice(2)).unwrap_err();
    assert_eq!(err.class, ErrorClass::Internal);
    let short = BufferSlice {
        bytes: Bytes(8),
        ..slice(2)
    };
    assert!(p.copy(&ctx, &s, slice(1), short).is_err());
    assert!(p.copy_batch(&ctx, &s, &[(slice(1), slice(2))]).is_err());
    assert!(p.driver.streams.lock().unwrap().log.is_empty());
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
