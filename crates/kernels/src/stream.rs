// SPDX-License-Identifier: MIT OR Apache-2.0

//! Streams, events, doorbell flags, host wakes and captured graphs (RFC-0040 §2.2, §4).
//!
//! Provider objects are `u64` handles, meaningful only to the provider that issued them, as
//! in [`memory`](crate::memory).

use serde::{Deserialize, Serialize};

/// A stream's priority (RFC-0040 §2.2, §4.1). CUDA and HIP map these to stream priorities,
/// compute highest.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StreamPriority {
    /// The step's compute work. All of a device's steps run on one compute stream, in order.
    Compute,
    /// Host to device: copied misses, prefetch, KV restore.
    CopyIn,
    /// Device to host: KV spill, the activation hand-off to the wire.
    CopyOut,
    /// Intra-node collectives (`IntraNodeTp`).
    Collective,
    /// Background work.
    Low,
}

/// An in-order work queue on one device (RFC-0001 §5.7, RFC-0040 §4.1). RFC-0001 names the
/// type without fields.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Stream {
    /// The provider's handle for the stream.
    pub handle: u64,
}

/// A point in a stream that other streams and the host can wait for (RFC-0001 §5.7,
/// RFC-0040 §4.3). RFC-0001 names the type without fields.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Event {
    /// The provider's handle for the event.
    pub handle: u64,
}

/// A replayable recording of a stream's launches (RFC-0001 §5.7, RFC-0040 §4.4). RFC-0001
/// names the type without fields.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct CapturedGraph {
    /// The provider's handle for the instantiated graph.
    pub handle: u64,
}

/// How `wait_host` compares a flag with its value (RFC-0040 §2.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WaitCmp {
    /// The flag equals the value. The doorbell handshake waits for `Eq` 1 (RFC-0040 §4.2).
    Eq,
    /// The flag is greater than or equal to the value.
    Geq,
}

/// A 4-byte slot in a doorbell page (RFC-0040 §2.2, §4.2): pinned, host-mapped and
/// registered with the device at `open`. The host sees it at a host address and the device
/// at a device address. Each flag has one producer and one consumer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct HostFlag {
    /// The doorbell page.
    pub page: u16,
    /// The slot in the page.
    pub slot: u16,
}

/// A wake target for a host thread parked inside a step (RFC-0040 §2.2, §4.2; RFC-0012
/// §3.3): a futex word, eventfd or Windows event that the waiter announced. Created at
/// `Warming`; its address is fixed for the epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HostWaker {
    /// What `handle` is.
    pub kind: WakeKind,
    /// The futex address, eventfd, event handle or timeline value.
    pub handle: u64,
}

/// What kind of wake target a [`HostWaker`] is (RFC-0040 §2.2).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WakeKind {
    /// A futex word.
    Futex,
    /// A Linux eventfd.
    EventFd,
    /// A Windows event.
    WinEvent,
    /// A timeline-semaphore value.
    TimelineValue,
}
