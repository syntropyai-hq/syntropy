// SPDX-License-Identifier: MIT OR Apache-2.0

//! `KvStore`, its page classes and pools, block tables and KV byte counters (RFC-0022 §1, §3,
//! §4; ADR-0205 items 1 and 2; TIER-E6-T02).
//!
//! - [`types`]: RFC-0022 §4's types and RFC-0001 §5.9's [`KvHandle`].
//! - [`pool`]: one class's pool on one tier and the page state machine of RFC-0022 §3.
//! - [`store`]: [`KvStore`] with `open`, `append`, `truncate`, `fork` and `release`, the
//!   per-class KV byte counters and their stats-ring records.
//!
//! This task builds the `Full`, `Windowed` and `State` classes in `Vram`. The spill pools and
//! tier moves are TIER-E6-T04's, `Compressed` TIER-E6-T16's, `Aux` TIER-E6-T08's, speculative
//! positions TIER-E6-T09's and state snapshots TIER-E6-T18's. The `Kv` error domain arrives with
//! TIER-E6-T12, so the store returns [`KvError`], which names RFC-0022 §14's code where one
//! applies, as the OS adapter's `OsMemError` does for the tier codes.

#![forbid(unsafe_code)]

pub mod pool;
pub mod store;
pub mod types;

use std::fmt;

use syntropy_core::{ErrorClass, StageIx};

pub use pool::PageState;
pub use store::{
    ClassSpec, KvBytes, KvBytesRecord, KvStore, KvStoreConfig, LAYER_ALIGN, ring_pages,
};
pub use types::{
    ClassTable, KvChunkDesc, KvExport, KvHandle, KvHeadroom, KvResidency, KvWriteSlots, MoveTicket,
    PAGE_TOKENS, PageClass, PageCopy, PageId, PageRef, PageSpan, PageTable, PageTableId,
    SessionSource, TailId,
};

/// Why a [`KvStore`] operation failed.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvError {
    /// A class's pool has no free page (`KV_POOL_FULL`, code 1, `Capacity`; RFC-0022 §14). The
    /// caller pauses the sequence (RFC-0012 §10.3).
    PoolFull {
        /// The class.
        class: PageClass,
        /// The tier's ordinal.
        tier: u8,
    },
    /// Every sequence slot is in use (`Capacity`): admission let in more sequences than the
    /// plan's maximum.
    NoSlot,
    /// A position that does not follow the sequence, or a cut the store cannot honour
    /// (`Internal`: a caller bug, or a window ring too short for the cut).
    BadPosition {
        /// The sequence's committed positions.
        tokens: u32,
        /// The position asked for.
        at: u32,
    },
    /// The page is moving between tiers: it is neither read, written nor released until the
    /// move completes (RFC-0022 §14, `Transient`).
    Moving {
        /// The page.
        page: PageId,
    },
    /// A page in a state the operation does not accept: a double free, a write after sealing,
    /// a shared page written (`Internal`).
    BadState {
        /// The page.
        page: PageId,
        /// Its state.
        state: PageState,
    },
    /// A page id outside its pool (`Internal`).
    BadPage {
        /// The page.
        page: PageId,
    },
    /// A handle whose slot is not open (`Internal`).
    BadTable {
        /// The slot.
        table: PageTableId,
    },
    /// A handle for another stage (`Internal`).
    WrongStage {
        /// The stage asked for.
        stage: StageIx,
    },
    /// A page shared by more holders than a reference count holds (`Internal`).
    TooManyRefs {
        /// The page.
        page: PageId,
    },
    /// Something a later task builds (`Unsupported`).
    Unsupported(&'static str),
}

impl KvError {
    /// The error class (RFC-0001 §5.12).
    #[must_use]
    pub fn class(&self) -> ErrorClass {
        match self {
            KvError::PoolFull { .. } | KvError::NoSlot => ErrorClass::Capacity,
            KvError::Moving { .. } => ErrorClass::Transient,
            KvError::Unsupported(_) => ErrorClass::Unsupported,
            KvError::BadPosition { .. }
            | KvError::BadState { .. }
            | KvError::BadPage { .. }
            | KvError::BadTable { .. }
            | KvError::WrongStage { .. }
            | KvError::TooManyRefs { .. } => ErrorClass::Internal,
        }
    }

    /// RFC-0022 §14's code, where the error has one.
    #[must_use]
    pub fn kv_code(&self) -> Option<u16> {
        match self {
            KvError::PoolFull { .. } => Some(1),
            _ => None,
        }
    }
}

impl fmt::Display for KvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KvError::PoolFull { class, tier } => {
                write!(f, "KV_POOL_FULL: no free {class:?} page on tier {tier}")
            }
            KvError::NoSlot => f.write_str("every KV sequence slot is in use"),
            KvError::BadPosition { tokens, at } => write!(
                f,
                "position {at} cannot be appended or cut at in a sequence of {tokens}"
            ),
            KvError::Moving { page } => write!(f, "KV page {page:?} is moving between tiers"),
            KvError::BadState { page, state } => {
                write!(
                    f,
                    "KV page {page:?} is {state:?}, which this operation refuses"
                )
            }
            KvError::BadPage { page } => write!(f, "KV page {page:?} is outside its pool"),
            KvError::BadTable { table } => write!(f, "KV sequence slot {} is not open", table.0),
            KvError::WrongStage { stage } => {
                write!(f, "this KV store is not stage {}'s", stage.0)
            }
            KvError::TooManyRefs { page } => write!(f, "KV page {page:?} has too many holders"),
            KvError::Unsupported(what) => f.write_str(what),
        }
    }
}

impl std::error::Error for KvError {}

#[cfg(test)]
mod tests;
