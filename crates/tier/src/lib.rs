// SPDX-License-Identifier: MIT OR Apache-2.0

//! The tier manager: arenas, pinned budgets, the `ResidencyMap`, the expert cache, prefetch and the
//! KV store.
//!
//! Layer 2 in RFC-0001 §4.1, designed in RFC-0020. This crate holds the RFC-0020 §1 modules and
//! the RFC-0020 §3 types (TIER-E1-T01); the behaviour comes in later TIER tasks.
//!
//! # `unsafe`
//!
//! RFC-0020 §1 confines `unsafe` to the [`arena`] and [`os`] modules (RFC-0001 §4.2 rule 8).
//! The crate root denies it, every other module forbids it with `#![forbid(unsafe_code)]`, and
//! only [`arena`] and [`os`] may lift the deny with a scoped `allow` once they hold `unsafe`
//! code. A `forbid` at the root could not be lifted in those two modules, which is why the root
//! uses `deny`. Every `unsafe` block and impl carries a `// SAFETY:` comment that states its
//! invariant (ADR-0001 item 4).

#![deny(unsafe_code)]
#![deny(clippy::undocumented_unsafe_blocks)]

pub mod arena;
pub mod budget;
pub mod eip;
pub mod experts;
pub mod kv;
pub mod manager;
pub mod os;
pub mod pin;
pub mod pressure;
pub mod residency;

pub use arena::{
    ArenaPlan, ArenaRef, ArenaSpec, PinClass, Purpose, Registrations, SegmentSpec, SlotClass,
    SlotGeometry, TierBudgetLine,
};
pub use manager::TierCommand;
pub use os::{
    CgroupMem, CommitCharge, CommitOutcome, HostRange, HugePages, LockOutcome, MemSnapshot,
    OsMemError, OsMemory, PageableReason, PinLimits, Psi, ReclaimStalls,
};
pub use pin::{LockRange, PinProgress, PinState, UnitSlot, lock_ranges, slot_layout};
pub use pressure::PressureLevel;
pub use residency::{HomeState, StepFence};
/// RFC-0020 §3's supporting types that `syntropy-core` defines, because `syntropy-io` carries
/// them a layer below this crate (RFC-0020 §3, revised 2026-10-21), or `syntropy-kernels` binds
/// them (`BufferView`, RFC-0040 §3.5). Re-exported, never redefined.
pub use syntropy_core::{BufferView, ExtentRef, FileIx, HostRegion, StepLayer, ViewKind};
/// The I/O ticket [`HomeState::Loading`] records, from `syntropy-io` (RFC-0021 §2).
pub use syntropy_io::IoTicket;
