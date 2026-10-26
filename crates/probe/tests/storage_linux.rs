// SPDX-License-Identifier: MIT OR Apache-2.0

//! PLC-E1-T06 acceptance criteria on Linux: `DirectIoSupport` on ext4, xfs and tmpfs volumes, and
//! a chipset-attached NVMe flagged `shared_lane` (RFC-0010 §2.1 E3, §2.2).
//!
//! **Every tree here is synthetic.** It is written by hand from the sysfs layout, not recorded
//! from a machine: the vendor and device IDs, drive names, firmware strings and link figures are
//! made up, and the AM5 tree follows a chipset's published shape, not a captured board. The
//! `statfs` and `statx` answers come from a shim, because a system call cannot be recorded in a
//! tree. A real recording of an AM5 board is still outstanding; until it exists the AM5 criterion
//! is met on a synthetic tree only.

use std::collections::BTreeMap;
use std::io;
use std::path::Path;

use syntropy_core::resource::{
    DirectIoSupport, FsKind, PcieUpstream, StorageBus, TempSource, VolumeInfo,
};
use syntropy_io::os::volume::{DioReport, FsStat, StatX, VolumeCalls};
use syntropy_probe::storage::linux_volume;
use syntropy_probe::{EnumError, HostFs};

/// An in-memory `/sys` and `/proc`.
#[derive(Default)]
struct Tree(BTreeMap<String, String>);

impl Tree {
    fn put(&mut self, path: &str, text: &str) -> &mut Self {
        self.0.insert(path.to_owned(), format!("{text}\n"));
        self
    }

    /// One PCI function directory with the attributes the walk reads.
    fn pci(&mut self, dir: &str, class: &str, max: (&str, u8), cur: (&str, u8)) -> &mut Self {
        self.put(&format!("{dir}/class"), class)
            .put(&format!("{dir}/vendor"), "0xffee")
            .put(&format!("{dir}/device"), "0x0001")
            .put(&format!("{dir}/subsystem_vendor"), "0xffee")
            .put(&format!("{dir}/subsystem_device"), "0x0002")
            .put(
                &format!("{dir}/max_link_speed"),
                &format!("{} GT/s PCIe", max.0),
            )
            .put(&format!("{dir}/max_link_width"), &max.1.to_string())
            .put(
                &format!("{dir}/current_link_speed"),
                &format!("{} GT/s PCIe", cur.0),
            )
            .put(&format!("{dir}/current_link_width"), &cur.1.to_string())
            .put(&format!("{dir}/numa_node"), "-1")
    }

    /// A synthetic NVMe namespace `nvme0n1` with a partition `nvme0n1p2` at `259:2`.
    fn nvme_disk(&mut self, address: &str, with_hwmon: bool) -> &mut Self {
        self.put(
            "sys/dev/block/259:2/uevent",
            "MAJOR=259\nMINOR=2\nDEVNAME=nvme0n1p2",
        )
        .put("sys/block/nvme0n1/queue/logical_block_size", "512")
        .put("sys/block/nvme0n1/queue/physical_block_size", "4096")
        .put("sys/block/nvme0n1/nvme0n1p2/partition", "2")
        .put("sys/class/nvme/nvme0/model", "SYNTHETIC NVMe 2TB")
        .put("sys/class/nvme/nvme0/firmware_rev", "SYN0001")
        .put("sys/class/nvme/nvme0/address", address);
        if with_hwmon {
            self.put("sys/class/nvme/nvme0/hwmon1/temp1_input", "41000");
        }
        self
    }

    fn mount(&mut self, point: &str, major_minor: &str, fstype: &str, source: &str) -> &mut Self {
        self.put(
            "proc/self/mountinfo",
            &format!(
                "20 1 8:1 / / rw - ext4 /dev/synthetic rw\n\
                 36 20 {major_minor} / {point} rw,relatime shared:1 - {fstype} {source} rw"
            ),
        )
    }
}

impl HostFs for Tree {
    fn read(&self, path: &str) -> io::Result<String> {
        self.0
            .get(path)
            .cloned()
            .ok_or_else(|| io::ErrorKind::NotFound.into())
    }

    fn exists(&self, path: &str) -> bool {
        let dir = format!("{path}/");
        self.0.keys().any(|k| k == path || k.starts_with(&dir))
    }

    fn list(&self, path: &str) -> io::Result<Vec<String>> {
        let dir = format!("{path}/");
        let mut names: Vec<String> = self
            .0
            .keys()
            .filter_map(|k| k.strip_prefix(&dir))
            .map(|rest| rest.split('/').next().unwrap_or(rest).to_owned())
            .collect();
        names.dedup();
        if names.is_empty() {
            return Err(io::ErrorKind::NotFound.into());
        }
        Ok(names)
    }
}

/// The answers `statfs` and `statx` would give.
struct Shim {
    fs_type: u64,
    free: u64,
    dev: (u32, u32),
    dio: DioReport,
}

impl VolumeCalls for Shim {
    fn statfs(&self, _: &Path) -> io::Result<FsStat> {
        Ok(FsStat {
            fs_type: self.fs_type,
            free: self.free,
        })
    }

    fn statx(&self, _: &Path) -> io::Result<StatX> {
        Ok(StatX {
            dev_major: self.dev.0,
            dev_minor: self.dev.1,
            dio: self.dio,
        })
    }
}

const EXT4: u64 = 0xEF53;
const XFS: u64 = 0x5846_5342;
const TMPFS: u64 = 0x0102_1994;
const STORE: &str = "/srv/store/models";

fn run(tree: &Tree, shim: &Shim) -> Result<VolumeInfo, EnumError> {
    linux_volume(tree, shim, Path::new(STORE))
}

/// A CPU root port with an NVMe controller directly below it.
fn cpu_attached() -> Tree {
    let mut t = Tree::default();
    t.mount("/srv/store", "259:2", "ext4", "/dev/nvme0n1p2")
        .nvme_disk("0000:01:00.0", true)
        .pci(
            "sys/devices/pci0000:00/0000:00:01.1",
            "0x060400",
            ("16.0", 4),
            ("16.0", 4),
        )
        .pci(
            "sys/devices/pci0000:00/0000:00:01.1/0000:01:00.0",
            "0x010802",
            ("16.0", 4),
            ("16.0", 4),
        );
    t
}

#[test]
fn ext4_on_a_cpu_attached_nvme() {
    let shim = Shim {
        fs_type: EXT4,
        free: 500 << 30,
        dev: (259, 2),
        dio: DioReport::Aligned(512, 4096),
    };
    let v = run(&cpu_attached(), &shim).unwrap();
    assert_eq!(v.fs, FsKind::Ext4);
    assert_eq!(v.mount, Path::new("/srv/store"));
    assert_eq!(v.bus, StorageBus::Nvme);
    assert_eq!(
        v.direct_io,
        DirectIoSupport::Supported {
            mem_align: 512,
            offset_align: 4096
        }
    );
    assert_eq!((v.logical_block, v.physical_block), (512, 4096));
    assert_eq!(v.free.0, 500 << 30);
    assert_eq!(
        (v.model.as_str(), v.firmware.as_str()),
        ("SYNTHETIC NVMe 2TB", "SYN0001")
    );
    assert_eq!(v.temperature_source, TempSource::Hwmon);
    assert!(!v.shared_lane);
    let pci = v.pci.unwrap();
    assert_eq!(pci.upstream, PcieUpstream::CpuRootPort);
    assert_eq!((pci.link_cur.r#gen, pci.link_cur.width), (4, 4));
    assert_eq!(pci.bdf.bus, 1);
}

#[test]
fn xfs_on_a_sata_partition() {
    let mut t = Tree::default();
    t.mount("/srv/store", "8:17", "xfs", "/dev/sdb1")
        .put(
            "sys/dev/block/8:17/uevent",
            "MAJOR=8\nMINOR=17\nDEVNAME=sdb1",
        )
        .put("sys/block/sdb/queue/logical_block_size", "512")
        .put("sys/block/sdb/queue/physical_block_size", "512")
        .put("sys/block/sdb/sdb1/partition", "1")
        .put("sys/block/sdb/device/vendor", "ATA")
        .put("sys/block/sdb/device/model", "SYNTHETIC SATA SSD")
        .put("sys/block/sdb/device/rev", "S001");
    let shim = Shim {
        fs_type: XFS,
        free: 1 << 40,
        dev: (8, 17),
        dio: DioReport::Aligned(512, 512),
    };
    let v = run(&t, &shim).unwrap();
    assert_eq!(v.fs, FsKind::Xfs);
    assert_eq!(v.bus, StorageBus::Sata);
    assert_eq!(
        v.direct_io,
        DirectIoSupport::Supported {
            mem_align: 512,
            offset_align: 512
        }
    );
    assert_eq!(v.pci, None);
    assert_eq!(v.temperature_source, TempSource::None);
    assert_eq!(v.model, "SYNTHETIC SATA SSD");
}

#[test]
fn tmpfs_has_no_block_device_and_refuses_direct_io() {
    let mut t = Tree::default();
    t.mount("/srv/store", "0:50", "tmpfs", "tmpfs");
    // Even if a kernel reported an alignment, a memory-backed volume is `Unsupported`.
    for dio in [DioReport::NotReported, DioReport::Aligned(4096, 4096)] {
        let shim = Shim {
            fs_type: TMPFS,
            free: 8 << 30,
            dev: (0, 50),
            dio,
        };
        let v = run(&t, &shim).unwrap();
        assert_eq!(v.fs, FsKind::Other);
        assert_eq!(v.bus, StorageBus::Virtual);
        assert_eq!(v.direct_io, DirectIoSupport::Unsupported);
        assert_eq!((v.logical_block, v.physical_block), (0, 0));
    }
}

#[test]
fn an_old_kernel_falls_back_to_the_logical_block_size() {
    // Before Linux 6.1 `STATX_DIOALIGN` is not reported (RFC-0021 §4.2): ext4 on a 512-byte
    // logical-block drive is `Supported` at 512.
    let shim = Shim {
        fs_type: EXT4,
        free: 1,
        dev: (259, 2),
        dio: DioReport::NotReported,
    };
    assert_eq!(
        run(&cpu_attached(), &shim).unwrap().direct_io,
        DirectIoSupport::Supported {
            mem_align: 512,
            offset_align: 512
        }
    );
}

#[test]
fn a_reported_zero_is_unsupported() {
    let shim = Shim {
        fs_type: EXT4,
        free: 1,
        dev: (259, 2),
        dio: DioReport::Zero,
    };
    assert_eq!(
        run(&cpu_attached(), &shim).unwrap().direct_io,
        DirectIoSupport::Unsupported
    );
}

#[test]
fn an_unreported_alignment_without_a_block_device_is_unsupported() {
    // A network or overlay volume: no disk to take a block size from.
    let mut t = Tree::default();
    t.mount("/srv/store", "0:60", "overlay", "overlay");
    let shim = Shim {
        fs_type: 0x794c_7630,
        free: 1,
        dev: (0, 60),
        dio: DioReport::NotReported,
    };
    assert_eq!(
        run(&t, &shim).unwrap().direct_io,
        DirectIoSupport::Unsupported
    );
}

/// A synthetic AM5-shaped tree: the NVMe sits behind a chipset whose upstream port is a Gen4 x4
/// link shared with a SATA controller, a USB controller and a NIC.
fn am5_chipset() -> Tree {
    let root = "sys/devices/pci0000:00/0000:00:02.1";
    let up = format!("{root}/0000:02:00.0");
    let mut t = Tree::default();
    t.mount("/srv/store", "259:2", "ext4", "/dev/nvme0n1p2")
        .nvme_disk("0000:04:00.0", false)
        .pci(root, "0x060400", ("16.0", 4), ("16.0", 4))
        .pci(&up, "0x060400", ("16.0", 4), ("16.0", 4));
    for (port, bus, class, max) in [
        ("0000:03:08.0", "04:00.0", "0x010802", ("16.0", 4)), // the NVMe drive
        ("0000:03:09.0", "05:00.0", "0x010601", ("8.0", 1)),  // SATA
        ("0000:03:0a.0", "06:00.0", "0x0c0330", ("8.0", 1)),  // USB
        ("0000:03:0b.0", "07:00.0", "0x020000", ("5.0", 1)),  // NIC
    ] {
        let port_dir = format!("{up}/{port}");
        t.pci(&port_dir, "0x060400", max, max);
        t.pci(&format!("{port_dir}/0000:{bus}"), class, max, max);
    }
    t
}

#[test]
fn a_chipset_attached_nvme_is_flagged_shared_lane() {
    let shim = Shim {
        fs_type: EXT4,
        free: 1 << 40,
        dev: (259, 2),
        dio: DioReport::Aligned(512, 4096),
    };
    let v = run(&am5_chipset(), &shim).unwrap();
    assert!(v.shared_lane);
    assert_eq!(v.pci.unwrap().upstream, PcieUpstream::Chipset);
    // No hwmon directory in this tree: the source is None, not guessed.
    assert_eq!(v.temperature_source, TempSource::None);
}

#[test]
fn an_uplink_wide_enough_for_its_children_is_not_shared() {
    // The same chipset shape, but the uplink is Gen5 x4, wider than the sum of its children.
    let mut t = am5_chipset();
    let up = "sys/devices/pci0000:00/0000:00:02.1/0000:02:00.0";
    t.put(&format!("{up}/max_link_speed"), "32.0 GT/s PCIe");
    let shim = Shim {
        fs_type: EXT4,
        free: 1,
        dev: (259, 2),
        dio: DioReport::Aligned(512, 4096),
    };
    let v = run(&t, &shim).unwrap();
    assert!(!v.shared_lane);
    assert_eq!(v.pci.unwrap().upstream, PcieUpstream::Switch);
}

#[test]
fn a_bridge_with_only_the_drive_below_it_is_not_shared() {
    // A switch whose only endpoint is the NVMe drive: no SATA, USB or NIC below it.
    let root = "sys/devices/pci0000:00/0000:00:01.1";
    let mut t = Tree::default();
    t.mount("/srv/store", "259:2", "ext4", "/dev/nvme0n1p2")
        .nvme_disk("0000:03:00.0", true)
        .pci(root, "0x060400", ("16.0", 4), ("16.0", 4))
        .pci(
            &format!("{root}/0000:01:00.0"),
            "0x060400",
            ("8.0", 1),
            ("8.0", 1),
        )
        .pci(
            &format!("{root}/0000:01:00.0/0000:03:00.0"),
            "0x010802",
            ("16.0", 4),
            ("8.0", 1),
        );
    let shim = Shim {
        fs_type: EXT4,
        free: 1,
        dev: (259, 2),
        dio: DioReport::Aligned(512, 4096),
    };
    let v = run(&t, &shim).unwrap();
    assert!(!v.shared_lane);
    assert_eq!(v.pci.unwrap().upstream, PcieUpstream::Switch);
}

#[test]
fn dm_crypt_over_a_partition_is_marked_encrypted() {
    let mut t = cpu_attached();
    t.mount("/srv/store", "253:0", "ext4", "/dev/mapper/synthetic")
        .put(
            "sys/dev/block/253:0/uevent",
            "MAJOR=253\nMINOR=0\nDEVNAME=dm-0",
        )
        .put("sys/block/dm-0/dm/uuid", "CRYPT-LUKS2-0000-synthetic")
        .put("sys/block/dm-0/slaves/nvme0n1p2/.keep", "");
    let shim = Shim {
        fs_type: EXT4,
        free: 1,
        dev: (253, 0),
        dio: DioReport::Aligned(512, 4096),
    };
    let v = run(&t, &shim).unwrap();
    assert_eq!(v.encrypted, Some(true));
    assert_eq!(v.bus, StorageBus::Nvme);
}

#[test]
fn the_deepest_mount_holds_the_path() {
    let mut t = cpu_attached();
    t.put(
        "proc/self/mountinfo",
        "20 1 8:1 / / rw - ext4 /dev/synthetic rw\n\
         36 20 259:2 / /srv rw - ext4 /dev/nvme0n1p2 rw\n\
         37 36 259:2 / /srv/store\\040data rw - ext4 /dev/nvme0n1p2 rw\n\
         38 36 259:2 / /srv/stor rw - ext4 /dev/nvme0n1p2 rw",
    );
    let shim = Shim {
        fs_type: EXT4,
        free: 1,
        dev: (259, 2),
        dio: DioReport::Aligned(512, 512),
    };
    // `/srv/stor` is a string prefix of the path but not a path prefix.
    assert_eq!(run(&t, &shim).unwrap().mount, Path::new("/srv"));
}

#[test]
fn a_missing_block_queue_file_is_reported_not_guessed() {
    let mut t = cpu_attached();
    t.0.remove("sys/block/nvme0n1/queue/logical_block_size");
    let shim = Shim {
        fs_type: EXT4,
        free: 1,
        dev: (259, 2),
        dio: DioReport::Aligned(512, 512),
    };
    assert!(matches!(run(&t, &shim), Err(EnumError::Missing { .. })));
}

/// The hosted Linux runner: the live `statfs` and `statx` calls answer for the temp directory. A
/// container's volume may have no block device, so only the system calls are held to account.
#[cfg(target_os = "linux")]
#[test]
fn live_linux_calls_answer() {
    use syntropy_io::os::volume::LiveVolumeCalls;
    use syntropy_probe::DirFs;

    let dir = std::env::temp_dir();
    match linux_volume(&DirFs::new("/"), &LiveVolumeCalls, &dir) {
        Ok(v) => {
            eprintln!("{v:?}");
            assert!(v.free.0 > 0);
        }
        Err(EnumError::Os { call, message, .. }) => panic!("{call} failed: {message}"),
        Err(other) => eprintln!("volume sysfs not readable here: {other:?}"),
    }
}
