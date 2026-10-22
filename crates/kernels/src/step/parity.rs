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
    /// Two of the flags given to [`ParityHandshake::new`] are the same slot.
    AliasedFlags,
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
            Self::AliasedFlags => write!(f, "two step flags share one slot"),
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

/// The four step-level flags of one graph's two parities (RFC-0040 §4.2 rule 5): 4-byte
/// host-mapped slots, `AtomicU32` here, each on its own 64-byte line. Both arrays are indexed
/// by [`Parity::index`].
///
/// The fields are named, and [`ParityHandshake::new`] binds them once and refuses two fields
/// that are the same slot, so no later call can pass `O[p]` where `DC[p]` belongs.
#[derive(Clone, Copy, Debug)]
pub struct ParityFlags<'a> {
    /// The descriptor-consumed flags `DC[0]` and `DC[1]`.
    pub dc: [&'a AtomicU32; 2],
    /// The step-done flags `O[0]` and `O[1]`.
    pub o: [&'a AtomicU32; 2],
}

impl ParityFlags<'_> {
    fn all(&self) -> [&AtomicU32; 4] {
        [self.dc[0], self.dc[1], self.o[0], self.o[1]]
    }
}

/// The host's record of both parities' flags, which gates every mirror write and every
/// submission (RFC-0040 §4.2 rule 5).
///
/// It holds the flags it was built with. The host consumes a flag with an acquire load that
/// sees 1, then a release store of 0.
///
/// There is one handshake per set of flags, and it is not `Clone`: a copy would share the
/// flags but keep its own record, so it could rewrite and resubmit a parity the original has
/// in flight. It can be built and moved:
///
/// ```
/// use std::sync::atomic::AtomicU32;
/// use syntropy_kernels::{ParityFlags, ParityHandshake};
/// let f = [AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)];
/// let flags = ParityFlags { dc: [&f[0], &f[1]], o: [&f[2], &f[3]] };
/// let hs = ParityHandshake::new(flags).unwrap();
/// let moved = hs;
/// # let _ = moved;
/// ```
///
/// but not copied:
///
/// ```compile_fail
/// use std::sync::atomic::AtomicU32;
/// use syntropy_kernels::{ParityFlags, ParityHandshake};
/// let f = [AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)];
/// let flags = ParityFlags { dc: [&f[0], &f[1]], o: [&f[2], &f[3]] };
/// let hs = ParityHandshake::new(flags).unwrap();
/// let copy = hs.clone();
/// # let _ = (hs, copy);
/// ```
#[derive(Debug)]
pub struct ParityHandshake<'a> {
    flags: ParityFlags<'a>,
    slots: [Slot; 2],
}

impl<'a> ParityHandshake<'a> {
    /// Binds `flags` and arms them: stores 1 to all four, so both parities owe their `DC` and
    /// `O` as if a previous use had just finished (RFC-0040 §4.2 rule 5: both start at 1).
    ///
    /// # Errors
    ///
    /// [`HandshakeError::AliasedFlags`] when two of the four are the same slot. Nothing is
    /// stored then.
    pub fn new(flags: ParityFlags<'a>) -> Result<Self, HandshakeError> {
        let all = flags.all();
        for (i, a) in all.iter().enumerate() {
            if all[i + 1..].iter().any(|b| std::ptr::eq(*a, *b)) {
                return Err(HandshakeError::AliasedFlags);
            }
        }
        let mut hs = Self {
            flags,
            slots: [Slot::ARMED; 2],
        };
        hs.rearm();
        Ok(hs)
    }

    /// Re-arms after a capture or an abandoned step: stores 1 to every flag and owes each of
    /// them to the host again (RFC-0040 §4.2).
    pub fn rearm(&mut self) {
        for flag in self.flags.all() {
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
    pub fn consume_dc(&mut self, p: Parity) -> Result<bool, HandshakeError> {
        let flag = self.flags.dc[p.index()];
        Self::consume(&mut self.slots[p.index()].dc_pending, p, flag)
    }

    /// Consumes `O[p]` if the device has raised it: returns whether it had.
    ///
    /// # Errors
    ///
    /// [`HandshakeError::NothingPending`] when no set of `O[p]` is owed.
    pub fn consume_o(&mut self, p: Parity) -> Result<bool, HandshakeError> {
        let flag = self.flags.o[p.index()];
        Self::consume(&mut self.slots[p.index()].o_pending, p, flag)
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
