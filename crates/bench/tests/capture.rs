// SPDX-License-Identifier: MIT OR Apache-2.0

//! `bench/lab/capture-inventory.sh` (QUAL-E15-T02), run against stubbed tools.
//!
//! The script reads a rig with `dmidecode`, `nvidia-smi`, `lscpu`, `lspci`, `ethtool` and sysfs.
//! Each test builds a stand-in rig in a temporary directory: stub tools first on `PATH`, and a
//! `SYNTROPY_SYSROOT` tree for `/sys` and `/proc`. Every value below is **fixture data**, made
//! up for the test. None of it was read from the lab's R-ADA, whose record only the
//! maintainer's run on the rig can produce.
//!
//! The output must parse as an inventory (`rigs check`'s own parser) and list the parts that
//! QUAL-E15-T02 names: two 64 GiB modules at 6000 MT/s, a 24 GB `sm_89`-class GPU, a
//! CPU-attached Gen4 NVMe, and 10 GbE and 2.5 GbE NICs.

#[cfg(unix)]
mod on_unix {
    use std::fs;
    use std::os::unix::fs::{PermissionsExt, symlink};
    use std::path::{Path, PathBuf};
    use std::process::{Command, Output};

    use syntropy_bench::rigs::{Attachment, Inventory};

    const SCRIPT: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../bench/lab/capture-inventory.sh"
    );

    const DMIDECODE_MEMORY: &str = "Memory Device\n\tSize: 64 GB\n\tLocator: FIXTURE_DIMM_A2\n\tConfigured Memory Speed: 6000 MT/s\n\tPart Number: FIXTURE-PART-64G\n\nMemory Device\n\tSize: No Module Installed\n\tLocator: FIXTURE_DIMM_A1\n\tConfigured Memory Speed: Unknown\n\tPart Number: Not Specified\n\nMemory Device\n\tSize: 64 GB\n\tLocator: FIXTURE_DIMM_B2\n\tConfigured Memory Speed: 6000 MT/s\n\tPart Number: FIXTURE-PART-64G\n\n";

    /// A stand-in rig in a fresh temporary directory, removed on drop.
    struct Fake {
        dir: PathBuf,
    }

    impl Fake {
        fn new(name: &str) -> Fake {
            let dir = std::env::temp_dir()
                .join(format!("syntropy-capture-{}-{name}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(dir.join("bin")).unwrap();
            let fake = Fake { dir };
            fake.tool("uname", "echo 0.0.0-fixture");
            fake.tool(
                "lscpu",
                "echo 'Model name:            Fixture Desktop CPU AVX2-only'",
            );
            fake.file("dmidecode-memory.txt", DMIDECODE_MEMORY);
            fake.tool(
                "dmidecode",
                &format!(
                    "case \"$*\" in *bios-version*) echo FIXTURE-BIOS-1 ;; *) cat {} ;; esac",
                    fake.dir.join("dmidecode-memory.txt").display()
                ),
            );
            fake.tool(
                "nvidia-smi",
                "echo '00000000:01:00.0, Fixture Ada-class 24GB, 00.00.00.00.01, 000.00, 8.9, 4, 5, 16'",
            );
            fake.tool(
                "lspci",
                "case \"$*\" in\n  *0000:02:00.0*) echo '02:00.0 \"Ethernet\" \"Vendor\" \"Fixture 10GbE\"' ;;\n  *) echo '03:00.0 \"Ethernet\" \"Vendor\" \"Fixture 2.5GbE\"' ;;\nesac",
            );
            fake.tool("ethtool", "echo 'firmware-version: fixture-fw'");
            fake.file("proc/cpuinfo", "processor\t: 0\nmicrocode\t: 0xfffe\n");
            fake.file(
                "sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
                "performance\n",
            );
            fake.file("sys/devices/system/cpu/smt/active", "1\n");
            fake.nvme("nvme0", "02:00.0");
            fake.nic("eth0", "0000:02:00.0", "16.0 GT/s PCIe", "4");
            fake.nic("eth1", "0000:03:00.0", "8.0 GT/s PCIe", "1");
            fake
        }

        fn file(&self, rel: &str, text: &str) {
            let path = self.dir.join(rel);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, text).unwrap();
        }

        fn tool(&self, name: &str, body: &str) {
            let path = self.dir.join("bin").join(name);
            fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
            fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        }

        fn nvme(&self, name: &str, _addr: &str) {
            let base = format!("sys/class/nvme/{name}");
            self.file(&format!("{base}/model"), "Fixture NVMe 2TB\n");
            self.file(&format!("{base}/firmware_rev"), "FIXFW01\n");
            self.file(
                &format!("{base}/device/current_link_speed"),
                "16.0 GT/s PCIe\n",
            );
            self.file(&format!("{base}/device/current_link_width"), "4\n");
        }

        /// A NIC whose `device` link points at a PCI directory named by its address.
        fn nic(&self, name: &str, addr: &str, speed: &str, width: &str) {
            let pci = self.dir.join("sys/devices/pci").join(addr);
            fs::create_dir_all(&pci).unwrap();
            fs::write(pci.join("current_link_speed"), format!("{speed}\n")).unwrap();
            fs::write(pci.join("current_link_width"), format!("{width}\n")).unwrap();
            let net = self.dir.join("sys/class/net").join(name);
            fs::create_dir_all(&net).unwrap();
            symlink(&pci, net.join("device")).unwrap();
        }

        fn run(&self, args: &[&str]) -> Output {
            let path = format!(
                "{}:{}",
                self.dir.join("bin").display(),
                std::env::var("PATH").unwrap()
            );
            Command::new("bash")
                .arg(SCRIPT)
                .args(args)
                .env("PATH", path)
                .env("SYNTROPY_SYSROOT", &self.dir)
                .output()
                .expect("bash runs the script")
        }
    }

    impl Drop for Fake {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    const GOOD: &[&str] = &[
        "--id",
        "R-FIXTURE",
        "--os-image",
        "fixture-image-1",
        "--room-sensor",
        "fixture-sensor-1",
        "--memory-profile",
        "FIXTURE-PROFILE",
        "--power-limits",
        "PL1=1W,PL2=2W",
        "--resizable-bar",
        "true",
        "--attachment",
        "nvme0=cpu",
        "--expect-compute-cap",
        "8.9",
    ];

    fn stdout(output: &Output) -> String {
        String::from_utf8(output.stdout.clone()).unwrap()
    }

    fn stderr(output: &Output) -> String {
        String::from_utf8(output.stderr.clone()).unwrap()
    }

    fn captured(fake: &Fake) -> Inventory {
        let output = fake.run(GOOD);
        assert!(
            output.status.success(),
            "script failed: {}",
            stderr(&output)
        );
        let text = format!("version = 1\n\n{}", stdout(&output));
        let inventory = Inventory::parse(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
        assert_eq!(inventory.rigs.len(), 1);
        inventory
    }

    #[test]
    fn bash_is_available() {
        // A missing bash must fail the suite, not skip it: every other test would pass vacuously.
        let status = Command::new("bash").arg("-c").arg("exit 0").status();
        assert!(
            status.is_ok_and(|s| s.success()),
            "bash is required on unix"
        );
    }

    #[test]
    fn the_record_parses_and_lists_the_named_parts() {
        let rig = captured(&Fake::new("parts")).rigs.remove(0);
        assert_eq!(rig.id, "R-FIXTURE");
        assert_eq!(rig.power_plan, "performance");
        assert_eq!(rig.cpu.microcode.as_deref(), Some("0xfffe"));
        assert_eq!(rig.memory.len(), 2);
        assert!(
            rig.memory
                .iter()
                .all(|m| m.size_gib == 64 && m.speed_mts == 6000)
        );
        assert_eq!(rig.gpu.len(), 1);
        assert_eq!(rig.gpu[0].pcie.as_ref().unwrap().to_string(), "gen4x16");
        assert_eq!(rig.nvme.len(), 1);
        assert_eq!(rig.nvme[0].attachment, Attachment::Cpu);
        assert_eq!(rig.nvme[0].pcie.as_ref().unwrap().to_string(), "gen4x4");
        let links: Vec<String> = rig
            .nic
            .iter()
            .map(|n| n.pcie.as_ref().unwrap().to_string())
            .collect();
        assert_eq!(links, ["gen4x4", "gen3x1"]);
        let bios = rig.bios.unwrap();
        assert!(bios.smt && bios.resizable_bar);
        assert_eq!(bios.memory_profile, "FIXTURE-PROFILE");
    }

    #[test]
    fn a_gen4_device_in_a_gen3_slot_is_recorded_at_gen3() {
        let fake = Fake::new("gen3slot");
        // The drive is rated for Gen4 but trained at Gen3 x4.
        fake.file(
            "sys/class/nvme/nvme0/device/max_link_speed",
            "16.0 GT/s PCIe\n",
        );
        fake.file(
            "sys/class/nvme/nvme0/device/current_link_speed",
            "8.0 GT/s PCIe\n",
        );
        let rig = captured(&fake).rigs.remove(0);
        assert_eq!(rig.nvme[0].pcie.as_ref().unwrap().to_string(), "gen3x4");
    }

    #[test]
    fn a_cpu_without_microcode_omits_the_key() {
        let fake = Fake::new("nomicrocode");
        fake.file("proc/cpuinfo", "processor\t: 0\n");
        assert_eq!(captured(&fake).rigs[0].cpu.microcode, None);
    }

    #[test]
    fn the_spare_gpu_record_differs_only_by_its_own_fields() {
        let fake = Fake::new("spare");
        let main = captured(&fake).rigs.remove(0);
        fake.tool(
            "nvidia-smi",
            "echo '00000000:01:00.0, Fixture Ada-class 24GB, 00.00.00.00.02, 000.00, 8.9, 4, 5, 16'",
        );
        let mut args = GOOD.to_vec();
        args[1] = "R-FIXTURE-SPARE";
        let output = fake.run(&args);
        assert!(output.status.success(), "{}", stderr(&output));
        let text = format!("version = 1\n{}", stdout(&output));
        let spare = Inventory::parse(&text).unwrap().rigs.remove(0);
        assert_eq!(spare.id, "R-FIXTURE-SPARE");
        assert_ne!(spare.gpu[0].vbios, main.gpu[0].vbios);
        assert_eq!(spare.memory, main.memory);
    }

    fn refused(fake: &Fake, args: &[&str], needle: &str) {
        let output = fake.run(args);
        assert!(!output.status.success(), "expected a refusal for {needle}");
        assert!(stdout(&output).is_empty(), "a refusal prints no record");
        assert!(
            stderr(&output).contains(needle),
            "stderr: {}",
            stderr(&output)
        );
    }

    #[test]
    fn a_value_the_rig_cannot_report_stops_the_script() {
        let fake = Fake::new("missing");
        fs::remove_file(
            fake.dir
                .join("sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        )
        .unwrap();
        refused(&fake, GOOD, "scaling_governor");
    }

    #[test]
    fn a_missing_nvme_attachment_is_asked_for_not_guessed() {
        let args: Vec<&str> = GOOD
            .iter()
            .copied()
            .filter(|a| *a != "--attachment" && *a != "nvme0=cpu")
            .collect();
        refused(&Fake::new("attach"), &args, "--attachment nvme0=cpu");
    }

    #[test]
    fn a_missing_flag_is_named() {
        let args = &GOOD[..GOOD.len() - 4];
        let args: Vec<&str> = args
            .iter()
            .copied()
            .filter(|a| *a != "--resizable-bar" && *a != "true")
            .collect();
        refused(&Fake::new("flag"), &args, "--resizable-bar is required");
    }

    #[test]
    fn the_wrong_gpu_class_is_refused() {
        let mut args = GOOD.to_vec();
        let last = args.len() - 1;
        args[last] = "12.0";
        refused(&Fake::new("class"), &args, "expected 12.0");
    }

    #[test]
    fn an_unquotable_value_is_refused() {
        let mut args = GOOD.to_vec();
        args[5] = "bad\"sensor";
        refused(&Fake::new("quote"), &args, "quote or backslash");
    }

    #[test]
    fn a_populated_machine_without_modules_is_refused() {
        let fake = Fake::new("nodimm");
        fake.tool(
            "dmidecode",
            "case \"$*\" in *bios-version*) echo B ;; *) echo 'Memory Device' ;; esac",
        );
        refused(&fake, GOOD, "no populated memory module");
    }

    #[test]
    fn a_colon_inside_a_value_is_kept() {
        let fake = Fake::new("colon");
        fake.tool("ethtool", "echo 'firmware-version: 2017:888d'");
        fake.tool(
            "lscpu",
            "echo 'Model name:            Fixture CPU: Rev 2 @ 3:1'",
        );
        let memory = DMIDECODE_MEMORY.replace("FIXTURE-PART-64G", "FIXTURE:PART:64G");
        fake.file("dmidecode-memory.txt", &memory);
        let rig = captured(&fake).rigs.remove(0);
        assert!(rig.nic.iter().all(|n| n.firmware == "2017:888d"));
        assert_eq!(rig.cpu.model, "Fixture CPU: Rev 2 @ 3:1");
        assert!(rig.memory.iter().all(|m| m.part == "FIXTURE:PART:64G"));
    }

    #[test]
    fn an_empty_configured_memory_speed_is_refused() {
        let fake = Fake::new("emptyspeed");
        let memory = DMIDECODE_MEMORY.replacen(
            "Configured Memory Speed: 6000 MT/s",
            "Configured Memory Speed:",
            1,
        );
        fake.file("dmidecode-memory.txt", &memory);
        refused(&fake, GOOD, "reports no configured speed");
    }

    #[test]
    fn an_unreadable_memory_speed_is_refused_on_its_own() {
        let fake = Fake::new("badspeed");
        let memory = DMIDECODE_MEMORY.replacen("6000 MT/s", "Unknown", 1);
        fake.file("dmidecode-memory.txt", &memory);
        refused(&fake, GOOD, "not a whole number of MT/s");
    }

    #[test]
    fn the_older_clock_speed_label_is_accepted_when_unambiguous() {
        let fake = Fake::new("clocklabel");
        let memory = DMIDECODE_MEMORY.replace("Configured Memory Speed", "Configured Clock Speed");
        fake.file("dmidecode-memory.txt", &memory);
        assert!(
            captured(&fake).rigs[0]
                .memory
                .iter()
                .all(|m| m.speed_mts == 6000)
        );

        // Both labels, naming different speeds: ambiguous.
        let both = DMIDECODE_MEMORY.replace(
            "\tConfigured Memory Speed: 6000 MT/s",
            "\tConfigured Memory Speed: 6000 MT/s\n\tConfigured Clock Speed: 4800 MT/s",
        );
        fake.file("dmidecode-memory.txt", &both);
        refused(&fake, GOOD, "configured speeds disagree");
    }

    #[test]
    fn a_network_device_that_is_not_a_pci_function_is_refused() {
        let fake = Fake::new("virtio");
        let virtio = fake.dir.join("sys/devices/virtio0");
        fs::create_dir_all(&virtio).unwrap();
        let net = fake.dir.join("sys/class/net/eth9");
        fs::create_dir_all(&net).unwrap();
        symlink(&virtio, net.join("device")).unwrap();
        refused(&fake, GOOD, "not a PCI function");
    }

    #[test]
    fn two_network_devices_on_one_pci_function_are_refused() {
        let fake = Fake::new("dupslot");
        let net = fake.dir.join("sys/class/net/eth2");
        fs::create_dir_all(&net).unwrap();
        symlink(
            fake.dir.join("sys/devices/pci/0000:02:00.0"),
            net.join("device"),
        )
        .unwrap();
        refused(&fake, GOOD, "already recorded");
    }

    #[test]
    fn the_script_is_executable_and_strict() {
        let mode = fs::metadata(SCRIPT).unwrap().permissions().mode();
        assert!(mode & 0o111 != 0);
        let text = fs::read_to_string(SCRIPT).unwrap();
        assert!(text.starts_with("#!/usr/bin/env bash\n# SPDX-License-Identifier"));
        assert!(text.contains("set -euo pipefail"));
        let _ = Path::new(SCRIPT);
    }
}

/// Windows has no bash. The suite says so, rather than passing without checking anything.
#[cfg(windows)]
#[test]
fn skipped_on_windows_which_has_no_bash() {
    eprintln!("SKIP: capture-inventory.sh is a bash script for Linux rigs; not run on Windows");
}
