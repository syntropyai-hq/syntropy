// SPDX-License-Identifier: MIT OR Apache-2.0

//! The live-range registry every OS adapter shares (RFC-0020 §4.2, note of 2026-10-22).
//!
//! Every range an adapter hands out is recorded with a reservation id, carried in
//! [`HostRange::os_handle`]. Ids come from one process-wide counter, so a handle from one
//! adapter never matches a range another adapter reserves at the same address. `commit`, `lock`
//! and `unlock` run their OS call inside [`Registry::with_pages`], which holds the registry's
//! lock shared; `release` runs inside [`Registry::remove_with`], which holds it exclusively. A
//! range is therefore never released while another call acts on it, and a stale or forged
//! handle is refused with [`OsMemError::UnknownRange`].

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{HostRange, OsMemError, check_sub};

/// The next reservation id, process-wide; 0 is never handed out.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// One live reservation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Live {
    len: u64,
    id: u64,
}

/// How [`Registry::with_pages`] turns a byte range into whole pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Round {
    /// Every page the range touches: commit and lock must cover each byte asked for.
    Outward,
    /// Only pages wholly inside the range: unlock must not release a page a neighbouring
    /// locked sub-range shares, because locks are not counted per caller.
    Inward,
}

/// The live reservations of one adapter, by base address.
#[derive(Debug, Default)]
pub(crate) struct Registry {
    live: RwLock<BTreeMap<usize, Live>>,
}

impl Registry {
    /// Records a new reservation and returns its id.
    pub(crate) fn insert(&self, base: usize, len: u64) -> u64 {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        self.live
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .insert(base, Live { len, id });
        id
    }

    fn is_live(live: &BTreeMap<usize, Live>, r: &HostRange) -> bool {
        live.get(&(r.base as usize))
            == Some(&Live {
                len: r.len.0,
                id: r.os_handle,
            })
    }

    /// Runs `f` on the address and length of `sub` widened or narrowed to whole pages of
    /// `page` bytes, as `round` says, with the registry's shared lock held throughout.
    /// `UnknownRange` when `r` is not live with the same base, length and id.
    pub(crate) fn with_pages<T>(
        &self,
        r: &HostRange,
        sub: &Range<u64>,
        page: u64,
        round: Round,
        f: impl FnOnce(*mut u8, usize) -> Result<T, OsMemError>,
    ) -> Result<T, OsMemError> {
        check_sub(r.len, sub)?;
        let live = self.live.read().unwrap_or_else(|p| p.into_inner());
        if !Self::is_live(&live, r) {
            return Err(OsMemError::UnknownRange);
        }
        let (start, end) = match round {
            Round::Outward => (
                sub.start - sub.start % page,
                sub.end.div_ceil(page).saturating_mul(page).min(r.len.0),
            ),
            Round::Inward => (
                sub.start.div_ceil(page).saturating_mul(page),
                sub.end - sub.end % page,
            ),
        };
        let len = end.saturating_sub(start);
        // An Inward start past the end gives len 0; the address is then never used.
        let addr = r.base.wrapping_add(start.min(r.len.0) as usize);
        let out = f(addr, len as usize);
        drop(live);
        out
    }

    /// Removes `r` and runs `release` on it, with the exclusive lock held, so no call that
    /// found the range live is still acting on it. Does nothing for a range that is not live.
    pub(crate) fn remove_with(&self, r: &HostRange, release: impl FnOnce()) {
        let mut live = self.live.write().unwrap_or_else(|p| p.into_inner());
        if !Self::is_live(&live, r) {
            return;
        }
        live.remove(&(r.base as usize));
        release();
        drop(live);
    }

    /// Takes every live reservation, as (base, length), for the adapter's `Drop`.
    pub(crate) fn drain(&mut self) -> Vec<(usize, u64)> {
        std::mem::take(self.live.get_mut().unwrap_or_else(|p| p.into_inner()))
            .into_iter()
            .map(|(base, l)| (base, l.len))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use syntropy_core::Bytes;

    use super::*;

    fn range(base: usize, len: u64, id: u64) -> HostRange {
        HostRange {
            base: base as *mut u8,
            len: Bytes(len),
            numa: None,
            os_handle: id,
        }
    }

    #[test]
    fn rounding_and_identity() {
        let reg = Registry::default();
        let id = reg.insert(0x10000, 0x4000);
        let r = range(0x10000, 0x4000, id);
        let got = reg.with_pages(&r, &(0x100..0x1100), 0x1000, Round::Outward, |a, l| {
            Ok((a as usize, l))
        });
        assert_eq!(got, Ok((0x10000, 0x2000)));
        let got = reg.with_pages(&r, &(0x100..0x3100), 0x1000, Round::Inward, |a, l| {
            Ok((a as usize, l))
        });
        assert_eq!(got, Ok((0x11000, 0x2000)));
        let got = reg.with_pages(&r, &(0x100..0x200), 0x1000, Round::Inward, |_, l| Ok(l));
        assert_eq!(got, Ok(0));
        let forged = range(0x10000, 0x4000, id + 1000);
        assert_eq!(
            reg.with_pages(&forged, &(0..1), 0x1000, Round::Outward, |_, _| Ok(())),
            Err(OsMemError::UnknownRange)
        );
        let mut released = 0;
        reg.remove_with(&forged, || released += 1);
        reg.remove_with(&r, || released += 1);
        reg.remove_with(&r, || released += 1);
        assert_eq!(released, 1);
    }

    #[test]
    fn ids_are_unique_across_registries() {
        let (a, b) = (Registry::default(), Registry::default());
        assert_ne!(a.insert(0x1000, 1), b.insert(0x1000, 1));
    }
}
