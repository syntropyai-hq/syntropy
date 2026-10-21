// SPDX-License-Identifier: MIT OR Apache-2.0

//! The residency map, tensor handles, leases and step fences (RFC-0020 §3, §7). This task
//! defines [`StepFence`] and [`BufferView`]; `ResidencyMap` comes in TIER-E1-T04 and leases in
//! TIER-E1-T05.
//
// TODO(#1491): add `HomeState` (RFC-0020 §3) once `IoTicket` lands in `syntropy-io`
// (TIER-E5-T01). Its `Loading { ticket: IoTicket }` variant needs that type, and it must be
// re-exported from `syntropy-io`, never defined twice.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use syntropy_core::{Bytes, DeviceId};

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
