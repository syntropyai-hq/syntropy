// SPDX-License-Identifier: MIT OR Apache-2.0

//! A CPU device's doorbell flags (RFC-0040 §4.2, the CPU row).
//!
//! `open` allocates `OpenOptions::doorbell_slots` flags. Each is a 4-byte atomic on a 64-byte
//! line of its own, so a store to one flag shares its line with nothing, as RFC-0040 §4.2
//! requires. A [`HostFlag`] names a flag by page and slot, 64 slots to a 4 KiB page. The CPU and
//! the host see the same memory, so the flag's host address and its device address are one.
//!
//! [`Doorbells`] is `Send` and `Sync`, and cheap to clone. The device's stream runners write and
//! wait on its flags, and so do the host threads on the other side of each doorbell: a GPU
//! dispatcher, or a test standing in for one.

use std::sync::Arc;
use std::sync::atomic::AtomicU32;

use syntropy_kernels::HostFlag;

/// Flags in one 4 KiB doorbell page, one per 64-byte line.
pub const FLAGS_PER_PAGE: u32 = 64;

/// One flag on a 64-byte line of its own.
#[repr(C, align(64))]
#[derive(Debug, Default)]
struct Line {
    flag: AtomicU32,
}

/// A device's doorbell flags.
#[derive(Clone, Debug)]
pub struct Doorbells {
    lines: Arc<[Line]>,
}

impl Doorbells {
    /// `slots` flags, all 0, rounded up to whole pages.
    pub(crate) fn new(slots: u32) -> Doorbells {
        let pages = slots.div_ceil(FLAGS_PER_PAGE).max(1);
        let lines = (0..pages * FLAGS_PER_PAGE)
            .map(|_| Line::default())
            .collect();
        Doorbells { lines }
    }

    /// How many flags there are.
    #[must_use]
    pub fn len(&self) -> u32 {
        u32::try_from(self.lines.len()).unwrap_or(u32::MAX)
    }

    /// Whether there are none. A device always has at least one page.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// The flag at index `index`, counting slots across pages.
    #[must_use]
    pub fn flag_at(&self, index: u32) -> HostFlag {
        HostFlag {
            page: u16::try_from(index / FLAGS_PER_PAGE).unwrap_or(u16::MAX),
            slot: u16::try_from(index % FLAGS_PER_PAGE).unwrap_or(u16::MAX),
        }
    }

    /// The index of `flag`, if the device has it.
    #[must_use]
    pub fn index(&self, flag: HostFlag) -> Option<u32> {
        if u32::from(flag.slot) >= FLAGS_PER_PAGE {
            return None;
        }
        let index = u32::from(flag.page) * FLAGS_PER_PAGE + u32::from(flag.slot);
        (index < self.len()).then_some(index)
    }

    /// The atomic behind `flag`, for a host thread on the other side of the doorbell. Stores
    /// that raise a flag are release stores, and loads that wait for one are acquire loads.
    #[must_use]
    pub fn flag(&self, flag: HostFlag) -> Option<&AtomicU32> {
        self.index(flag).map(|i| self.at(i))
    }

    /// The atomic at `index`, which [`Doorbells::index`] checked.
    pub(crate) fn at(&self, index: u32) -> &AtomicU32 {
        &self.lines[index as usize].flag
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::*;

    #[test]
    fn flags_sit_on_lines_of_their_own_in_whole_pages() {
        assert_eq!(std::mem::size_of::<Line>(), 64);
        assert_eq!(std::mem::align_of::<Line>(), 64);
        let bells = Doorbells::new(70);
        assert_eq!(bells.len(), 128, "two pages");
        let last = bells.flag_at(127);
        assert_eq!(last, HostFlag { page: 1, slot: 63 });
        assert_eq!(bells.index(last), Some(127));
        assert_eq!(bells.index(HostFlag { page: 2, slot: 0 }), None);
        assert_eq!(bells.index(HostFlag { page: 0, slot: 64 }), None);
        let a = bells.flag(bells.flag_at(0)).unwrap() as *const AtomicU32 as usize;
        let b = bells.flag(bells.flag_at(1)).unwrap() as *const AtomicU32 as usize;
        assert_eq!(b - a, 64);
        assert_eq!(a % 64, 0);
        bells.at(5).store(1, Ordering::Release);
        assert_eq!(bells.clone().at(5).load(Ordering::Acquire), 1);
        assert_eq!(Doorbells::new(0).len(), 64, "at least one page");
    }
}
