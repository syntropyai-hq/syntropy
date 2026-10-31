// SPDX-License-Identifier: MIT OR Apache-2.0

//! E3 storage-volume enumeration (RFC-0010 §2.1 E3, §2.2 "NVMe volume" and "Chipset-shared
//! lane"; PLC-E1-T06): the `VolumeInfo` of the volume holding a store directory.
//!
//! - [`linux`]: `statfs` and `statx(STATX_DIOALIGN)` through [`VolumeCalls`](syntropy_io::os::volume::VolumeCalls), then `/sys` and
//!   `/proc/self/mountinfo` through a [`HostFs`](crate::HostFs) for the block queue, the drive, the controller's
//!   PCI path and the temperature source.
//! - [`windows`] and [`macos`]: classify what `syntropy_io::os::volume` reports.
//!
//! The system calls that need `unsafe` are in `syntropy_io::os::volume` (RFC-0001 §4.2 rule 8,
//! OQ-PRC-26); this module has none.

pub mod linux;
pub mod macos;
mod pci;
pub mod windows;

pub use linux::volume as linux_volume;
pub use macos::{classify as classify_macos, live as macos_live};
pub use windows::{classify as classify_windows, live as windows_live};
