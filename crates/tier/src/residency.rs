// SPDX-License-Identifier: MIT OR Apache-2.0

//! The residency map, tensor handles, leases and step fences (RFC-0020 §3, §7). This task
//! defines [`HomeState`], [`StepFence`] and [`BufferView`]; `ResidencyMap` comes in
//! TIER-E1-T04 and leases in TIER-E1-T05.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use syntropy_core::{Bytes, DeviceId};
use syntropy_io::IoTicket;

use crate::arena::PinClass;

/// Where a group's home copy is in its life (RFC-0020 §3, §7.2). RFC-0020 adds it to RFC-0001
/// §5.5's `Residency` as a `#[serde(default)]` field, so the default is [`HomeState::Absent`]:
/// a record that predates the field claims no bytes are resident.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum HomeState {
    /// The arena is allocated; the bytes are not read yet. The default.
    #[default]
    Absent,
    /// A read is in flight.
    Loading {
        /// The read's ticket. `syntropy-io` defines `IoTicket` without serde, so it is
        /// serialized as its `u64`.
        #[serde(with = "io_ticket")]
        ticket: IoTicket,
    },
    /// Resident and at its planned pin class.
    Resident,
    /// Resident but below its planned pin class (pinning not done, or yielded). Costed as
    /// `HostPageable`; never a DMA source while in this state (ADR-0007 item 4).
    Underpinned {
        /// The pin class the bytes have now.
        have: PinClass,
    },
    /// The home is `Nvme`: an extent map, read on demand or by prefetch.
    Extent,
}

/// Serde for [`IoTicket`] as its `u64`, since `syntropy-io` takes no serde dependency.
mod io_ticket {
    use serde::{Deserialize, Deserializer, Serializer};
    use syntropy_io::IoTicket;

    pub(super) fn serialize<S: Serializer>(t: &IoTicket, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(t.0)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<IoTicket, D::Error> {
        u64::deserialize(d).map(IoTicket)
    }
}

/// A completed-step count on one device (RFC-0020 §3). A slot freed at step `s` may be
/// overwritten once the device's completed-step counter reaches `s`. The host keeps the
/// counter: the dispatcher increments it each time it consumes the step-done flag `O[p]`
/// (RFC-0012 §3.2), in step order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StepFence {
    /// The device's ordinal.
    pub device: u16,
    /// The step the fence waits for.
    pub step: u64,
}

/// What the hot path and the I/O engine use to address resident bytes (RFC-0020 §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BufferView {
    /// Which address space `addr` is in.
    pub kind: ViewKind,
    /// The start address.
    pub addr: u64,
    /// The length.
    pub len: Bytes,
}

/// The address space of a [`BufferView`] (RFC-0020 §3).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ViewKind {
    /// Host memory.
    Host,
    /// One device's memory.
    Device(DeviceId),
    /// Unified memory, addressable by host and GPU alike.
    Unified,
}
