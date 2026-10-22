// SPDX-License-Identifier: MIT OR Apache-2.0

//! The spec database (QUAL-E12-T11; RFC-0061 §5.3). The acceptance criteria this file checks:
//! - the source lint fails a record without a URL;
//! - R-ADA's record produces `W_peak` values matching the research's verified figures;
//! - a fixture probe at 40% of spec marks the run `dirty: misconfigured`.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use syntropy_bench::rigs::{self, Inventory, Rig};
use syntropy_bench::schema::EnvCapture;
use syntropy_bench::specs::{self, DataPath, Peak, Probe, SpecDb};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn database() -> SpecDb {
    let (db, problems) = SpecDb::load(&root().join(specs::DIR));
    assert_eq!(problems, Vec::new(), "the repository's spec database");
    db
}

fn r_ada() -> Rig {
    let text = include_str!("data/r-ada.toml");
    let inventory = Inventory::parse(text).expect("the fixture is a valid inventory");
    inventory.rig("R-ADA").expect("R-ADA").clone()
}

fn peak(peaks: &[Peak], path: &DataPath) -> f64 {
    peaks
        .iter()
        .find(|p| &p.path == path)
        .unwrap_or_else(|| panic!("no peak for {path}"))
        .gb_per_s()
}

/// Within 0.1%: the research rounds the PCIe rates, which follow from 16.0 GT/s × 128/130.
fn close(got: f64, want: f64) -> bool {
    (got - want).abs() <= 0.001 * want
}

#[test]
fn the_repository_database_passes_the_source_lint() {
    let db = database();
    assert_eq!(db.records.len(), 5);
}

/// The figures of `plan/research/hardware-and-performance-model.md` §2 for R-ADA's parts:
/// RTX 4090 1008 GB/s (§2.1, [S10]); DDR5-6000 in 2 channels 96.0 GB/s (§2.1, [S22]); PCIe 4.0
/// x16 31.5 GB/s and x4 7.88 GB/s (§2.3, [S28]); 990 PRO 7450 MB/s sequential read (§2.4,
/// [S32]).
#[test]
fn r_adas_record_gives_the_researchs_verified_peaks() {
    let peaks = specs::peaks(&r_ada(), &database()).expect("every R-ADA part has a record");
    assert_eq!(peaks.len(), 4);
    let vram = peak(&peaks, &DataPath::Vram(String::from("PCIEX16_1")));
    assert!(close(vram, 1008.0), "{vram}");
    let dram = peak(&peaks, &DataPath::HostByCpu);
    assert!(close(dram, 96.0), "{dram}");
    // A GPU reading host memory is capped by its link, the narrowest element of that path.
    let by_gpu = peaks
        .iter()
        .find(|p| p.path == DataPath::HostByGpu(String::from("PCIEX16_1")))
        .expect("host by GPU");
    assert!(close(by_gpu.gb_per_s(), 31.5), "{}", by_gpu.gb_per_s());
    assert!(close(by_gpu.chain[0].gb_per_s, 96.0));
    assert!(
        by_gpu
            .narrowest()
            .expect("an element")
            .name
            .starts_with("PCIe gen4x16")
    );
    // A drive is capped by itself, not by its 7.88 GB/s link.
    let nvme = peaks
        .iter()
        .find(|p| p.path == DataPath::Nvme(String::from("M2_1")))
        .expect("nvme");
    assert!(close(nvme.gb_per_s(), 7.45), "{}", nvme.gb_per_s());
    assert!(
        close(nvme.chain[1].gb_per_s, 7.88),
        "{}",
        nvme.chain[1].gb_per_s
    );
    // Every element names its sources, each a vendor or standards document.
    for peak in &peaks {
        for element in &peak.chain {
            assert!(!element.sources.is_empty());
            assert!(element.sources.iter().all(|s| s.starts_with("https://")));
        }
    }
}

fn env() -> EnvCapture {
    EnvCapture {
        rig_id: String::from("R-ADA"),
        checks: Vec::new(),
        dirty: None,
        driver: Some(String::from("580.65.06")),
        firmware: BTreeMap::new(),
        kernel: String::from("6.8.0-fixture"),
        microcode: None,
        gpu_clock_mhz: None,
    }
}

#[test]
fn a_probe_at_40_percent_of_spec_marks_the_run_dirty_misconfigured() {
    let peaks = specs::peaks(&r_ada(), &database()).expect("peaks");
    let vram = DataPath::Vram(String::from("PCIEX16_1"));
    let probe = |gb_per_s: f64| {
        vec![
            Probe {
                path: vram.clone(),
                gb_per_s,
            },
            Probe {
                path: DataPath::HostByCpu,
                gb_per_s: 90.0,
            },
        ]
    };
    let mut env40 = env();
    let found = specs::mark_misconfigured(&mut env40, &peaks, &probe(0.40 * 1008.0));
    assert_eq!(env40.dirty.as_deref(), Some("misconfigured"));
    assert_eq!(
        found.misconfigured,
        ["vram PCIEX16_1: probed 403.20 GB/s, 40% of its spec 1008.00 GB/s, below 50%"]
    );
    assert!(found.unchecked.is_empty());
    let mut env60 = env();
    assert_eq!(
        specs::mark_misconfigured(&mut env60, &peaks, &probe(0.60 * 1008.0)),
        specs::Sanity::default()
    );
    assert_eq!(env60.dirty, None);
    // A run already dirty for another reason keeps it.
    let mut inventory = env();
    inventory.dirty = Some(String::from("inventory"));
    specs::mark_misconfigured(&mut inventory, &peaks, &probe(0.40 * 1008.0));
    assert_eq!(inventory.dirty.as_deref(), Some("inventory"));
}

#[test]
fn dram_counts_fitted_channels_at_the_slowest_speed_and_a_fabric_limit_caps_it() {
    let db = database();
    let mut rig = r_ada();
    rig.memory.truncate(1);
    let peaks = specs::peaks(&rig, &db).expect("peaks");
    assert!(close(peak(&peaks, &DataPath::HostByCpu), 48.0));
    let mut rig = r_ada();
    rig.memory[1].speed_mts = 4800;
    let peaks = specs::peaks(&rig, &db).expect("peaks");
    assert!(close(peak(&peaks, &DataPath::HostByCpu), 76.8));

    // A CPU with a documented fabric limit narrower than its DRAM.
    let dir = TempDir::new("fabric");
    for (name, record) in &db.records {
        let text = fs::read_to_string(root().join(specs::DIR).join(name)).expect("record");
        let text = if matches!(record, specs::Record::Cpu { .. }) {
            format!(
                "{text}\n[fabric_read_gb_per_s]\nvalue = 64.0\n\
                 source = \"https://example.com/fabric\"\n\
                 where = \"fixture\"\nchecked = \"2026-10-22\"\n"
            )
        } else {
            text
        };
        fs::write(dir.0.join(name), text).expect("write");
    }
    let (fabric_db, problems) = SpecDb::load(&dir.0);
    assert_eq!(problems, Vec::new());
    let peaks = specs::peaks(&r_ada(), &fabric_db).expect("peaks");
    let by_cpu = peaks
        .iter()
        .find(|p| p.path == DataPath::HostByCpu)
        .expect("cpu");
    assert!(close(by_cpu.gb_per_s(), 64.0));
    assert!(by_cpu.narrowest().expect("element").name.contains("fabric"));
}

#[test]
fn a_part_without_a_record_is_reported() {
    let mut rig = r_ada();
    rig.gpu[0].model = String::from("Unknown GPU");
    let problems = specs::peaks(&rig, &database()).unwrap_err();
    assert!(
        problems
            .iter()
            .any(|p| p.contains("`Unknown GPU` in PCIEX16_1 has no `gpu` record")),
        "{problems:?}"
    );
}

/// A chipset-attached drive has no peak until the chipset uplink has a record, and only that
/// drive: the rig's other paths keep theirs, and a probe of the drive is unchecked, which never
/// marks the run dirty.
#[test]
fn a_chipset_drive_has_no_peak_and_its_probe_is_unchecked() {
    let mut rig = r_ada();
    rig.nvme[0].attachment = rigs::Attachment::Chipset;
    let peaks = specs::peaks(&rig, &database()).expect("the other paths still have peaks");
    assert_eq!(peaks.len(), 3);
    assert!(close(peak(&peaks, &DataPath::HostByCpu), 96.0));
    let drive = DataPath::Nvme(String::from("M2_1"));
    assert!(peaks.iter().all(|p| p.path != drive));
    let without = specs::without_peak(&rig);
    assert_eq!(without.len(), 1);
    assert_eq!(without[0].0, drive);
    assert!(without[0].1.contains("chipset-attached"));
    let mut run = env();
    let found = specs::mark_misconfigured(
        &mut run,
        &peaks,
        &[Probe {
            path: drive,
            gb_per_s: 0.5,
        }],
    );
    assert!(found.misconfigured.is_empty());
    assert_eq!(found.unchecked.len(), 1);
    assert_eq!(run.dirty, None);
}

/// Every rig the real inventory holds resolves its peaks. The inventory is empty until the rigs
/// come online (QUAL-E15-T02 adds R-ADA), and the test then covers each.
#[test]
fn every_inventory_rig_has_its_spec_records() {
    let text = fs::read_to_string(root().join(rigs::PATH)).expect("inventory");
    let inventory = Inventory::parse(&text.replace("\r\n", "\n")).expect("valid");
    let db = database();
    for rig in &inventory.rigs {
        if let Err(problems) = specs::peaks(rig, &db) {
            panic!("{}: {problems:#?}", rig.id);
        }
    }
}

/// `syntropy-bench specs check`, the `lint` step, fails a record without a source URL.
#[test]
fn the_lint_step_fails_a_record_without_a_url() {
    let dir = TempDir::new("lint");
    let good = "kind = \"gpu\"\nmodels = [\"Fixture GPU\"]\n\n[memory_gb_per_s]\nvalue = 100.0\n\
                source = \"https://example.com/gpu.pdf\"\nwhere = \"p. 1\"\n\
                checked = \"2026-10-22\"\n";
    fs::write(dir.0.join("gpu-fixture.toml"), good).expect("write");
    let run = || {
        Command::new(env!("CARGO_BIN_EXE_syntropy-bench"))
            .args(["specs", "check", "--dir"])
            .arg(&dir.0)
            .output()
            .expect("runs")
    };
    let ok = run();
    assert!(
        ok.status.success(),
        "{}",
        String::from_utf8_lossy(&ok.stderr)
    );
    fs::write(
        dir.0.join("gpu-fixture.toml"),
        good.replace("source = \"https://example.com/gpu.pdf\"\n", ""),
    )
    .expect("write");
    let failed = run();
    assert_eq!(failed.status.code(), Some(1));
    let err = String::from_utf8_lossy(&failed.stderr);
    assert!(
        err.contains("gpu-fixture.toml") && err.contains("source"),
        "{err}"
    );
}

/// A directory under the system temp directory, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> TempDir {
        let path =
            std::env::temp_dir().join(format!("syntropy-specs-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("temp dir");
        TempDir(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
