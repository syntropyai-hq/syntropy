// SPDX-License-Identifier: MIT OR Apache-2.0

//! The host's side of the per-parity step flags (RFC-0040 §3.5, §4.2 rule 5; ADR-0404 items 1
//! and 6).
//!
//! Each mirror parity `p` has a descriptor-consumed flag `DC[p]` and a step-done flag `O[p]`,
//! each with the device as producer and the host as consumer. The device writes the constant
//! 1; the host waits for 1, then writes 0. The host's next submission of parity `p` is what
//! leads to the flags' next set, so it consumes both from `p`'s previous use before it writes
//! `p`'s mirror again or resubmits `p`. Both flags start at 1, as if a previous use had just
//! finished, and are re-armed that way after each capture and after any abandoned step.

use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};

use super::desc::{StepDescError, StepDescLayout, StepDescriptor};

/// One of the two mirrors, and the graph instance that reads it (RFC-0040 §3.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Parity {
    /// Mirror 0.
    Even,
    /// Mirror 1.
    Odd,
}

impl Parity {
    /// Both parities, by index.
    pub const BOTH: [Parity; 2] = [Parity::Even, Parity::Odd];

    /// 0 or 1: the mirror's index.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Even => 0,
            Self::Odd => 1,
        }
    }

    /// The other parity.
    #[must_use]
    pub const fn other(self) -> Self {
        match self {
            Self::Even => Self::Odd,
            Self::Odd => Self::Even,
        }
    }
}

/// Why the handshake refused a host action. Every refusal leaves the state unchanged.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HandshakeError {
    /// `DC[p]` from the parity's previous use is not consumed yet.
    DescriptorNotConsumed(Parity),
    /// `O[p]` from the parity's previous use is not consumed yet.
    StepNotDone(Parity),
    /// The parity's mirror was not written since its last submission.
    NotWritten(Parity),
    /// Nothing is owed on that flag: it was consumed already, and the parity was not
    /// resubmitted since.
    NothingPending(Parity),
    /// Writing the descriptor failed.
    Descriptor(StepDescError),
}

impl fmt::Display for HandshakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DescriptorNotConsumed(p) => {
                write!(f, "DC[{}] of the previous use is not consumed", p.index())
            }
            Self::StepNotDone(p) => {
                write!(f, "O[{}] of the previous use is not consumed", p.index())
            }
            Self::NotWritten(p) => write!(f, "mirror {} not written since submit", p.index()),
            Self::NothingPending(p) => {
                write!(f, "no set is owed on this flag of parity {}", p.index())
            }
            Self::Descriptor(e) => write!(f, "step descriptor: {e}"),
        }
    }
}

impl std::error::Error for HandshakeError {}

impl From<StepDescError> for HandshakeError {
    fn from(e: StepDescError) -> Self {
        Self::Descriptor(e)
    }
}

/// One parity's host-side state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Slot {
    /// A set of `DC[p]` is owed and not yet consumed.
    dc_pending: bool,
    /// A set of `O[p]` is owed and not yet consumed.
    o_pending: bool,
    /// The mirror holds a descriptor written since the last submission.
    written: bool,
}

impl Slot {
    /// The state at start, after re-arming and after a submission: both flags owed to the
    /// host, the mirror not yet written for the next use.
    const ARMED: Slot = Slot {
        dc_pending: true,
        o_pending: true,
        written: false,
    };
}

/// The host's record of both parities' flags, which gates every mirror write and every
/// submission (RFC-0040 §4.2 rule 5).
///
/// The flags themselves are 4-byte host-mapped slots (`AtomicU32` here), passed to each call
/// that touches one. The host consumes a flag with an acquire load that sees 1, then a release
/// store of 0.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParityHandshake {
    slots: [Slot; 2],
}

impl Default for ParityHandshake {
    fn default() -> Self {
        Self::new()
    }
}

impl ParityHandshake {
    /// The state matching flags that start at 1: both parities owe their `DC` and `O`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            slots: [Slot::ARMED; 2],
        }
    }

    /// Re-arms after a capture or an abandoned step: stores 1 to every flag and owes each of
    /// them to the host again (RFC-0040 §4.2). `dc` and `o` are indexed by
    /// [`Parity::index`].
    pub fn rearm(&mut self, dc: [&AtomicU32; 2], o: [&AtomicU32; 2]) {
        for flag in dc.into_iter().chain(o) {
            flag.store(1, Ordering::Release);
        }
        self.slots = [Slot::ARMED; 2];
    }

    fn consume(pending: &mut bool, p: Parity, flag: &AtomicU32) -> Result<bool, HandshakeError> {
        if !*pending {
            return Err(HandshakeError::NothingPending(p));
        }
        if flag.load(Ordering::Acquire) != 1 {
            return Ok(false);
        }
        flag.store(0, Ordering::Release);
        *pending = false;
        Ok(true)
    }

    /// Consumes `DC[p]` if the device has raised it: returns whether it had.
    ///
    /// # Errors
    ///
    /// [`HandshakeError::NothingPending`] when no set of `DC[p]` is owed.
    pub fn consume_dc(&mut self, p: Parity, dc: &AtomicU32) -> Result<bool, HandshakeError> {
        Self::consume(&mut self.slots[p.index()].dc_pending, p, dc)
    }

    /// Consumes `O[p]` if the device has raised it: returns whether it had.
    ///
    /// # Errors
    ///
    /// [`HandshakeError::NothingPending`] when no set of `O[p]` is owed.
    pub fn consume_o(&mut self, p: Parity, o: &AtomicU32) -> Result<bool, HandshakeError> {
        Self::consume(&mut self.slots[p.index()].o_pending, p, o)
    }

    /// Whether parity `p`'s previous use is fully consumed, so its mirror may be written and
    /// the parity submitted.
    #[must_use]
    pub fn is_free(&self, p: Parity) -> bool {
        let s = self.slots[p.index()];
        !s.dc_pending && !s.o_pending
    }

    fn check_free(&self, p: Parity) -> Result<(), HandshakeError> {
        let s = self.slots[p.index()];
        if s.dc_pending {
            return Err(HandshakeError::DescriptorNotConsumed(p));
        }
        if s.o_pending {
            return Err(HandshakeError::StepNotDone(p));
        }
        Ok(())
    }

    /// Writes `d` into parity `p`'s mirror.
    ///
    /// # Errors
    ///
    /// [`HandshakeError::DescriptorNotConsumed`] or [`HandshakeError::StepNotDone`] while
    /// `p`'s previous use owes a flag, and [`HandshakeError::Descriptor`] when the descriptor
    /// does not fit `layout` or `mirror`. The mirror is unchanged on error.
    pub fn write(
        &mut self,
        p: Parity,
        d: &StepDescriptor,
        layout: &StepDescLayout,
        mirror: &mut [u8],
    ) -> Result<(), HandshakeError> {
        self.check_free(p)?;
        d.write(layout, mirror)?;
        self.slots[p.index()].written = true;
        Ok(())
    }

    /// Records the submission of parity `p`'s graph instance, which owes one set of `DC[p]`
    /// and one of `O[p]`. The caller replays the instance right after.
    ///
    /// # Errors
    ///
    /// [`HandshakeError::DescriptorNotConsumed`] or [`HandshakeError::StepNotDone`] while
    /// `p`'s previous use owes a flag, and [`HandshakeError::NotWritten`] when the mirror was
    /// not written since `p`'s last submission.
    pub fn submit(&mut self, p: Parity) -> Result<(), HandshakeError> {
        self.check_free(p)?;
        let slot = &mut self.slots[p.index()];
        if !slot.written {
            return Err(HandshakeError::NotWritten(p));
        }
        *slot = Slot::ARMED;
        Ok(())
    }
}
