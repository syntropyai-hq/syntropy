// SPDX-License-Identifier: MIT OR Apache-2.0

//! PLC-E1-T04: IOKit's `gpu-core-count` fills `DeviceInfoExt::gpu_cores`, from the recorded
//! `IOAccelerator` fixture in `tests/fixtures/iokit/`; and, on macOS, the live registry reads
//! as the same records.

use syntropy_core::resource::DeviceKind;
use syntropy_kernels_loader::gpu::iokit::{Accelerator, apple_gpus, parse_fixture, render_fixture};

fn fixture(name: &str) -> Vec<Accelerator> {
    let path = format!(
        "{}/tests/fixtures/iokit/{name}.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    parse_fixture(&text).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// The criterion: the recorded M4 Pro's `gpu-core-count` of 16 fills `gpu_cores`, on one
/// integrated, unified-memory GPU named by its model, with no PCI record.
#[test]
fn recorded_gpu_core_count_fills_gpu_cores() {
    let accelerators = fixture("m4-pro");
    assert_eq!(
        accelerators,
        vec![Accelerator {
            class: "AGXAcceleratorG16X".into(),
            model: Some("Apple M4 Pro".into()),
            gpu_core_count: Some(16),
        }]
    );
    let gpus = apple_gpus(&accelerators);
    assert_eq!(gpus.len(), 1);
    let gpu = &gpus[0];
    assert_eq!(gpu.ext.gpu_cores, Some(16));
    assert_eq!(gpu.kind, DeviceKind::IntegratedGpu);
    assert!(gpu.unified);
    assert_eq!(gpu.name, "Apple M4 Pro");
    assert_eq!(gpu.ext.pci, None);
    assert_eq!(gpu.vram_total, None);
    let candidate = gpu.candidate();
    assert!(candidate.unified && candidate.pci.is_none());
}

/// A missing count, or one outside `u16`, leaves `gpu_cores` unknown rather than guessing it.
#[test]
fn a_missing_or_impossible_count_is_unknown() {
    let accelerators = parse_fixture(
        "accelerator class=AGXAcceleratorG13G\n\
         accelerator class=AGXAcceleratorG13G;model=Apple M1;gpu-core-count=70000\n",
    )
    .unwrap();
    let gpus = apple_gpus(&accelerators);
    assert_eq!(gpus[0].ext.gpu_cores, None);
    assert_eq!(gpus[0].name, "AGXAcceleratorG13G");
    assert_eq!(gpus[1].ext.gpu_cores, None);
}

/// Fixture text round-trips, and a malformed line is refused with its line number.
#[test]
fn fixture_text_round_trips_and_refuses_bad_lines() {
    let accelerators = fixture("m4-pro");
    assert_eq!(
        parse_fixture(&render_fixture(&accelerators)),
        Ok(accelerators)
    );
    for bad in [
        "gpu class=X",
        "accelerator model=Apple M4",
        "accelerator class=X;gpu-core-count=many",
        "accelerator class=X;colour=blue",
        "accelerator class",
    ] {
        let error = parse_fixture(bad).unwrap_err();
        assert!(error.starts_with("line 1:"), "{bad}: {error}");
    }
}

/// On a Mac, the live registry has at least one accelerator, each with a class, and the
/// mapping never fails on what it reads. On Apple silicon each has a model and a core count. In a
/// virtual machine (`kern.hv_vmm_present` 1, as on the hosted macOS runner) the accelerator is
/// paravirtual, `AppleParavirtGPU`, with neither, and its `gpu_cores` stays unknown. Set
/// `SYNTROPY_RECORD_IOKIT=1` to print the registry as fixture text.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn live_registry_reads_on_macos() {
    let live = syntropy_kernels_loader::gpu::iokit::live_accelerators().unwrap();
    if std::env::var_os("SYNTROPY_RECORD_IOKIT").is_some() {
        print!("{}", render_fixture(&live));
    }
    let vm = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "kern.hv_vmm_present"])
        .output()
        .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).trim() == "1");
    assert!(!live.is_empty());
    let gpus = apple_gpus(&live);
    for (a, gpu) in live.iter().zip(&gpus) {
        assert!(!a.class.is_empty());
        if vm {
            assert!(gpu.unified && !gpu.name.is_empty(), "{a:?}");
        } else {
            assert!(a.model.is_some(), "{a:?}");
            assert!(a.gpu_core_count.is_some_and(|n| n > 0), "{a:?}");
            assert!(gpu.ext.gpu_cores.is_some(), "{a:?}");
        }
    }
}

/// The hosted macOS runner's virtual machine, recorded: a paravirtual accelerator with no model
/// and no core count maps to a GPU named by its class, with `gpu_cores` unknown, and no error.
#[test]
fn a_paravirtual_accelerator_has_no_core_count() {
    let accelerators = fixture("paravirt");
    assert_eq!(
        accelerators,
        vec![Accelerator {
            class: "AppleParavirtGPU".into(),
            model: None,
            gpu_core_count: None,
        }]
    );
    let gpus = apple_gpus(&accelerators);
    assert_eq!(gpus.len(), 1);
    assert_eq!(gpus[0].name, "AppleParavirtGPU");
    assert_eq!(gpus[0].ext.gpu_cores, None);
    assert!(gpus[0].unified);
}
