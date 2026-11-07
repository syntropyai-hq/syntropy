// SPDX-License-Identifier: MIT OR Apache-2.0

//! One class's page pool on one tier, and the page state machine of RFC-0022 §3:
//!
//! ```text
//! Free ──alloc──▶ Writable ──page full──▶ Sealed ──move──▶ Moving ──completion──▶ Sealed
//! Writable/Sealed ──last reference dropped──▶ (retired) ──step fence──▶ Free
//! ```
//!
//! A sealed page is immutable while anything else may see it. The one way back to `Writable` is
//! [`Pool::reopen`]: a page whose only reference is the caller's, not moving, being rewritten
//! after a truncation or by a window ring that reuses its oldest slot. A shared page is never
//! written; the store copies it first.
//!
//! Every vector is sized when the pool is made, so no operation allocates.

#![forbid(unsafe_code)]

use super::KvError;
use super::types::{MoveTicket, PageClass, PageId};

/// The state of one page (RFC-0022 §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageState {
    /// On the free stack.
    Free,
    /// A sequence's tail page: positions are being appended.
    Writable,
    /// Immutable; may be shared, moved or saved.
    Sealed,
    /// Being copied to tier `to`; readable by no one until the completion.
    Moving {
        /// The destination tier's ordinal.
        to: u8,
        /// The move.
        ticket: MoveTicket,
    },
    /// The last reference was dropped; free after the next step fence (RFC-0020 §7.4).
    Retired,
}

/// The pages of one class on one tier.
#[derive(Debug)]
pub struct Pool {
    class: PageClass,
    tier: u8,
    states: Vec<PageState>,
    refs: Vec<u16>,
    /// The free stack; the top is the next page handed out.
    free: Vec<u32>,
    /// Pages retired since the last fence.
    retired: Vec<u32>,
}

impl Pool {
    /// A pool of `capacity` free pages.
    pub fn new(class: PageClass, tier: u8, capacity: u32) -> Self {
        let n = capacity as usize;
        Pool {
            class,
            tier,
            states: vec![PageState::Free; n],
            refs: vec![0; n],
            // Pages come out in index order: the stack's top is page 0.
            free: (0..capacity).rev().collect(),
            retired: Vec::with_capacity(n),
        }
    }

    /// Pages in the pool.
    pub fn capacity(&self) -> u32 {
        self.states.len() as u32
    }

    /// Pages on the free stack.
    pub fn free_pages(&self) -> u32 {
        self.free.len() as u32
    }

    /// The state of `page`.
    pub fn state(&self, page: PageId) -> PageState {
        self.states[page.index as usize]
    }

    /// References held on `page`.
    pub fn refs(&self, page: PageId) -> u16 {
        self.refs[page.index as usize]
    }

    fn id(&self, index: u32) -> PageId {
        PageId {
            tier: self.tier,
            index,
        }
    }

    fn check(&self, page: PageId) -> Result<usize, KvError> {
        let i = page.index as usize;
        if page.tier != self.tier || i >= self.states.len() {
            return Err(KvError::BadPage { page });
        }
        Ok(i)
    }

    /// Takes a free page: `Free → Writable`, one reference.
    ///
    /// # Errors
    ///
    /// [`KvError::PoolFull`] when the free stack is empty.
    pub fn alloc(&mut self) -> Result<PageId, KvError> {
        let Some(index) = self.free.pop() else {
            return Err(KvError::PoolFull {
                class: self.class,
                tier: self.tier,
            });
        };
        let i = index as usize;
        debug_assert_eq!(self.states[i], PageState::Free);
        self.states[i] = PageState::Writable;
        self.refs[i] = 1;
        Ok(self.id(index))
    }

    /// `Writable → Sealed`. Sealing a sealed page does nothing.
    ///
    /// # Errors
    ///
    /// [`KvError::BadState`] for a page that is neither.
    pub fn seal(&mut self, page: PageId) -> Result<(), KvError> {
        let i = self.check(page)?;
        match self.states[i] {
            PageState::Writable => {
                self.states[i] = PageState::Sealed;
                Ok(())
            }
            PageState::Sealed => Ok(()),
            state => Err(KvError::BadState { page, state }),
        }
    }

    /// One more reference, for a fork or a prefix. Only a sealed page is shared.
    ///
    /// # Errors
    ///
    /// [`KvError::BadState`] for a page that is not `Sealed`, or `Moving` (whose copy at the old
    /// tier stays valid).
    pub fn retain(&mut self, page: PageId) -> Result<(), KvError> {
        let i = self.check(page)?;
        match self.states[i] {
            PageState::Sealed | PageState::Moving { .. } => {
                self.refs[i] = self.refs[i]
                    .checked_add(1)
                    .ok_or(KvError::TooManyRefs { page })?;
                Ok(())
            }
            state => Err(KvError::BadState { page, state }),
        }
    }

    /// Drops one reference. The last one retires the page, which becomes free at the next
    /// [`fence`](Pool::fence).
    ///
    /// # Errors
    ///
    /// [`KvError::BadState`] for a page with no reference: a double free.
    pub fn release(&mut self, page: PageId) -> Result<(), KvError> {
        let i = self.check(page)?;
        match self.states[i] {
            PageState::Writable | PageState::Sealed => {}
            // A move in flight finishes first; the store never releases a moving page.
            state => return Err(KvError::BadState { page, state }),
        }
        self.refs[i] -= 1;
        if self.refs[i] == 0 {
            self.states[i] = PageState::Retired;
            self.retired.push(page.index);
        }
        Ok(())
    }

    /// `Sealed → Writable` for a page whose only reference is the caller's and which is not
    /// moving: the page is about to be rewritten (truncation, a window ring's reuse).
    ///
    /// # Errors
    ///
    /// [`KvError::BadState`] for a shared or moving page.
    pub fn reopen(&mut self, page: PageId) -> Result<(), KvError> {
        let i = self.check(page)?;
        match (self.states[i], self.refs[i]) {
            (PageState::Writable, 1) => Ok(()),
            (PageState::Sealed, 1) => {
                self.states[i] = PageState::Writable;
                Ok(())
            }
            (state, _) => Err(KvError::BadState { page, state }),
        }
    }

    /// Checks that `page` may be written: `Writable` and held once.
    ///
    /// # Errors
    ///
    /// [`KvError::BadState`] otherwise.
    pub fn check_write(&self, page: PageId) -> Result<(), KvError> {
        let i = self.check(page)?;
        match (self.states[i], self.refs[i]) {
            (PageState::Writable, 1) => Ok(()),
            (state, _) => Err(KvError::BadState { page, state }),
        }
    }

    /// Checks that `page` may be read: `Writable` or `Sealed`, never `Moving` (RFC-0022 §14).
    ///
    /// # Errors
    ///
    /// [`KvError::Moving`] for a moving page, [`KvError::BadState`] for a free or retired one.
    pub fn check_read(&self, page: PageId) -> Result<(), KvError> {
        let i = self.check(page)?;
        match self.states[i] {
            PageState::Writable | PageState::Sealed => Ok(()),
            PageState::Moving { .. } => Err(KvError::Moving { page }),
            state => Err(KvError::BadState { page, state }),
        }
    }

    /// `Sealed → Moving { to, ticket }`.
    ///
    /// # Errors
    ///
    /// [`KvError::BadState`] for a page that is not `Sealed`.
    pub fn begin_move(&mut self, page: PageId, to: u8, ticket: MoveTicket) -> Result<(), KvError> {
        let i = self.check(page)?;
        match self.states[i] {
            PageState::Sealed => {
                self.states[i] = PageState::Moving { to, ticket };
                Ok(())
            }
            state => Err(KvError::BadState { page, state }),
        }
    }

    /// `Moving → Sealed` when `ticket` completes. Moving into another tier's pool, and freeing
    /// the old copy after its fence, come with the spill pools (TIER-E6-T04); here the page
    /// stays in this pool.
    ///
    /// # Errors
    ///
    /// [`KvError::BadState`] for a page that is not moving under `ticket`.
    pub fn finish_move(&mut self, page: PageId, ticket: MoveTicket) -> Result<(), KvError> {
        let i = self.check(page)?;
        match self.states[i] {
            PageState::Moving { ticket: t, .. } if t == ticket => {
                self.states[i] = PageState::Sealed;
                Ok(())
            }
            state => Err(KvError::BadState { page, state }),
        }
    }

    /// The step fence (RFC-0020 §7.4): every page retired since the last fence goes back on the
    /// free stack, in retirement order. Returns how many.
    pub fn fence(&mut self) -> u32 {
        let n = self.retired.len() as u32;
        for index in self.retired.drain(..) {
            self.states[index as usize] = PageState::Free;
            self.free.push(index);
        }
        n
    }

    /// Pages that are neither free nor retired.
    pub fn live(&self) -> u32 {
        self.capacity() - self.free_pages() - self.retired.len() as u32
    }
}
