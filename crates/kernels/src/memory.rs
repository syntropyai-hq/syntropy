// SPDX-License-Identifier: MIT OR Apache-2.0

//! Memory a provider allocates, maps, registers or imports for the tier manager
//! (RFC-0001 §5.7; RFC-0040 §2.2, §5.1, §5.2; RFC-0020 §4.1, §6).
//!
//! `kernels` uses no `unsafe` code beyond declaring three `unsafe` methods (see the crate
//! docs), so nothing here holds a raw pointer. Device and host addresses are `u64` values and provider objects are `u64`
//! handles, both meaningful only to the provider that issued them. The provider crates
//! (`kernels-*`), which may use `unsafe`, turn them back into driver objects.

use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use syntropy_core::Bytes;

use crate::error::Error;

/// What `alloc` is asked for (RFC-0001 §5.7; provider semantics in RFC-0040 §5.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AllocRequest {
    /// Size of the allocation.
    pub bytes: Bytes,
    /// Where the memory lives.
    pub kind: MemKind,
    /// Required alignment.
    pub align: Bytes,
    /// Reserve address space only, to be backed later with `map_segment` (CUDA and HIP VMM).
    pub reserve_only: bool,
}

/// The kind of memory an [`AllocRequest`] asks for (RFC-0001 §5.7, RFC-0040 §5.1).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemKind {
    /// Device-local memory.
    Device,
    /// Host memory the device reads and writes through a mapping.
    HostMapped,
    /// Memory both the host and the device access at full rate.
    Unified,
}

/// A buffer a provider allocated or imported (RFC-0001 §5.7). The tier manager owns its
/// lifetime and returns it through `free`. RFC-0001 names the type without fields; this is
/// the smallest shape the trait needs.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct DeviceBuffer {
    /// The device address of the first byte; 0 for a reservation with nothing mapped yet on
    /// backends without address reservation (RFC-0040 §5.1).
    pub addr: u64,
    /// Size of the buffer or reservation.
    pub bytes: Bytes,
    /// Where the memory lives.
    pub kind: MemKind,
    /// The provider's handle for the buffer.
    pub handle: u64,
}

/// A byte range of a [`DeviceBuffer`], the unit `copy` moves (RFC-0001 §5.7). RFC-0001 names
/// the type without fields; the argument types of RFC-0040 §3.5 that use it are
/// COMP-E1-T05's work.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferSlice {
    /// The buffer's [`DeviceBuffer::handle`].
    pub buffer: u64,
    /// Offset of the first byte in the buffer.
    pub offset: Bytes,
    /// Length of the range.
    pub bytes: Bytes,
}

/// A physical segment backing part of a `reserve_only` allocation (RFC-0040 §2.2). Addresses
/// inside the reservation never move (RFC-0020 §4.1). Not `Clone`: `unmap_segment` takes it
/// by value, so each segment is released once.
///
/// A segment can be built and moved:
///
/// ```
/// use syntropy_core::Bytes;
/// use syntropy_kernels::Segment;
/// let seg = Segment { reservation: 1, offset: Bytes(0), bytes: Bytes(4096), handle: 2 };
/// let moved = seg;
/// # let _ = moved;
/// ```
///
/// but not copied:
///
/// ```compile_fail
/// use syntropy_core::Bytes;
/// use syntropy_kernels::Segment;
/// let seg = Segment { reservation: 1, offset: Bytes(0), bytes: Bytes(4096), handle: 2 };
/// let copy = seg.clone();
/// # let _ = (seg, copy);
/// ```
#[derive(Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Segment {
    /// The reservation's [`DeviceBuffer::handle`].
    pub reservation: u64,
    /// Offset of the segment in the reservation.
    pub offset: Bytes,
    /// Size of the segment, rounded to the driver's granularity.
    pub bytes: Bytes,
    /// The provider's handle for the physical allocation.
    pub handle: u64,
}

/// The range of host memory `register_host` and `import` take (RFC-0001 §5.7): `core`'s
/// type, which the I/O engine registers too (RFC-0021 §2).
pub use syntropy_core::HostRegion;

/// A host range registered with a device (RFC-0001 §5.7; RFC-0020 §6). It lives until it is
/// passed back to [`HostRegistrar::unregister`]. RFC-0001 names the type without fields;
/// this is the smallest shape the trait needs.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Registration {
    /// The registered range.
    pub region: HostRegion,
    /// The provider's handle for the registration.
    pub handle: u64,
}

bitflags! {
    /// How a [`HostRegistrar`] registers a range (RFC-0040 §2.2; RFC-0020 §6). A provider-level
    /// set, because a layer-1 trait cannot name `syntropy-tier`'s `Registrations`: the tier
    /// manager maps its `GPU_DMA` and `GPU_MAPPED` onto these and handles `IO_FIXED` and
    /// `RDMA_MR` itself.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
    #[serde(transparent)]
    pub struct HostRegFlags: u8 {
        /// DMA-visible to the device.
        const DMA = 1 << 0;
        /// Mapped into the device's address space.
        const MAPPED = 1 << 1;
    }
}

/// A `Send` capability that registers and unregisters host ranges from the housekeeping
/// thread, so progressive pinning never blocks the dispatcher (RFC-0040 §2.1, §2.2;
/// RFC-0020 §6).
///
/// Registrations made through it are valid for every stream of the device. Units follow
/// RFC-0020 §6.1: they are aligned to extent, slot and elastic-segment boundaries. Until the
/// M1 spike settles RFC-0020 open question 9, every DMA registration completes before
/// `Ready`, and none is issued while steps run.
pub trait HostRegistrar: Send {
    /// Registers `region` with the device.
    ///
    /// # Safety
    ///
    /// As for [`KernelProvider::register_host`](crate::KernelProvider::register_host):
    /// `region` must be host memory the caller owns, valid for reads and writes over its whole
    /// length. It must stay so until [`unregister`](Self::unregister) releases the
    /// registration and every operation queued on the range has finished.
    #[allow(unsafe_code)]
    unsafe fn register(
        &self,
        region: HostRegion,
        flags: HostRegFlags,
    ) -> Result<Registration, Error>;
    /// Releases a registration.
    fn unregister(&self, reg: Registration) -> Result<(), Error>;
}
