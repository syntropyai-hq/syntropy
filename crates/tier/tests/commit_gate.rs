// SPDX-License-Identifier: MIT OR Apache-2.0

//! No commit charge is taken before the gate allows it (TIER-E2-T03; RFC-0020 §4.2 Windows
//! row, §6): a scripted committer over a fake OS that charges commit the way Windows does,
//! only on `commit`, never on `reserve`. OS-independent.

use std::ops::Range;
use std::sync::Mutex;
use std::time::Instant;

use smallvec::SmallVec;
use syntropy_core::Bytes;
use syntropy_tier::os::{commit_if, commit_term};
use syntropy_tier::{
    CommitOutcome, HostRange, HugePages, MemSnapshot, OsMemError, OsMemory, PinLimits,
};

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

/// A fake Windows: a commit limit, a charge other programs move by script, and this process's
/// commits on top.
struct ScriptedOs {
    limit: u64,
    state: Mutex<State>,
}

struct State {
    /// The charge of other programs at each snapshot, in order; the last value repeats.
    others: Vec<u64>,
    snapshots: usize,
    /// Bytes this process has committed.
    committed: u64,
    /// Every call, in order.
    log: Vec<&'static str>,
}

impl ScriptedOs {
    fn new(limit: u64, others: Vec<u64>) -> Self {
        Self {
            limit,
            state: Mutex::new(State {
                others,
                snapshots: 0,
                committed: 0,
                log: Vec::new(),
            }),
        }
    }

    fn committed(&self) -> u64 {
        self.state.lock().unwrap().committed
    }
}

impl OsMemory for ScriptedOs {
    fn snapshot(&self) -> MemSnapshot {
        let mut s = self.state.lock().unwrap();
        let i = s.snapshots.min(s.others.len() - 1);
        s.snapshots += 1;
        s.log.push("snapshot");
        let charge = s.others[i] + s.committed;
        // Built as `commit_headroom` would on Windows, independent of the host OS.
        let headroom = Bytes(self.limit.saturating_sub(charge));
        MemSnapshot {
            total: Bytes(64 * GIB),
            available: Bytes(32 * GIB),
            numa_free: SmallVec::new(),
            numa_available: SmallVec::new(),
            cgroup: None,
            commit_headroom: Some(headroom),
            own_locked: Bytes(0),
            own_registered: Bytes(0),
            psi: None,
            direct_reclaim_stalls: None,
            os_level: None,
            at: Instant::now(),
        }
    }
    fn limits(&self) -> PinLimits {
        PinLimits::default()
    }
    fn reserve(
        &self,
        bytes: Bytes,
        numa: Option<u16>,
        _: HugePages,
    ) -> Result<HostRange, OsMemError> {
        self.state.lock().unwrap().log.push("reserve");
        Ok(HostRange {
            base: std::ptr::null_mut(),
            len: bytes,
            numa,
            os_handle: 1,
        })
    }
    fn commit(&self, _: &HostRange, sub: Range<u64>) -> Result<(), OsMemError> {
        let mut s = self.state.lock().unwrap();
        s.log.push("commit");
        s.committed += sub.end - sub.start;
        Ok(())
    }
    fn lock(&self, _: &HostRange, _: Range<u64>) -> Result<(), OsMemError> {
        Ok(())
    }
    fn unlock(&self, _: &HostRange, _: Range<u64>) -> Result<(), OsMemError> {
        Ok(())
    }
    fn release(&self, _: HostRange) {}
}

/// Reserving a whole arena takes no charge; each pin step commits only when the commit term
/// allows it, and a refused step leaves the charge where it was.
#[test]
fn no_commit_charge_before_the_gate_allows_it() {
    let headroom = Bytes(4 * GIB);
    let step = 1 << 30;
    // Commit limit 24 GiB. Other programs hold 12 GiB, then spike to 19 GiB for three
    // readings, then fall back to 10 GiB.
    let os = ScriptedOs::new(
        24 * GIB,
        vec![12 * GIB, 12 * GIB, 19 * GIB, 19 * GIB, 19 * GIB, 10 * GIB],
    );
    let r = os.reserve(Bytes(8 * step), None, HugePages::Off).unwrap();
    assert_eq!(os.committed(), 0, "reserve took a commit charge");

    let mut outcomes = Vec::new();
    let mut charge_before = Vec::new();
    let mut offset = 0;
    while offset < r.len.0 && outcomes.len() < 32 {
        charge_before.push(os.committed());
        let out = commit_if(&os, &r, offset..offset + step, |s, len| {
            commit_term(s, len, headroom)
        })
        .unwrap();
        if out == CommitOutcome::Committed {
            offset += step;
        }
        outcomes.push(out);
    }
    use CommitOutcome::{Committed as C, Deferred as D};
    // A step fits while others + own + 1 GiB + 4 GiB ≤ 24 GiB, that is others + own ≤ 19 GiB.
    // Readings 1–2: 12 + 0 and 12 + 1. Readings 3–5: 19 + 2 > 19, deferred. Then 10 + 2 up to
    // 10 + 7: the six remaining steps.
    assert_eq!(outcomes, [C, C, D, D, D, C, C, C, C, C, C]);
    // A deferred step never moved the charge.
    for (i, o) in outcomes.iter().enumerate() {
        if *o == D {
            let after = charge_before.get(i + 1).copied().unwrap_or(os.committed());
            assert_eq!(after, charge_before[i], "step {i} charged while deferred");
        }
    }
    assert_eq!(os.committed(), 8 * step);
    let log = os.state.lock().unwrap().log.clone();
    // Every commit follows the snapshot the gate judged it on.
    for (i, call) in log.iter().enumerate() {
        if *call == "commit" {
            assert_eq!(log[i - 1], "snapshot");
        }
    }
}

/// The commit term itself: `commit_headroom − len ≥ headroom`, and no term without commit
/// accounting.
#[test]
fn commit_term_cases() {
    let mut s = ScriptedOs::new(10 * GIB, vec![0]).snapshot();
    s.commit_headroom = Some(Bytes(6 * GIB));
    assert!(commit_term(&s, Bytes(2 * GIB), Bytes(4 * GIB)));
    assert!(!commit_term(&s, Bytes(2 * GIB + 1), Bytes(4 * GIB)));
    assert!(!commit_term(&s, Bytes(7 * GIB), Bytes(0)));
    s.commit_headroom = None;
    assert!(commit_term(&s, Bytes(u64::MAX), Bytes(u64::MAX)));
}

/// A sub-range outside the range is refused before the gate runs.
#[test]
fn commit_if_checks_the_range_first() {
    let os = ScriptedOs::new(GIB, vec![0]);
    let r = os.reserve(Bytes(MIB), None, HugePages::Off).unwrap();
    let mut ran = false;
    let out = commit_if(&os, &r, 0..MIB + 1, |_, _| {
        ran = true;
        true
    });
    assert!(matches!(out, Err(OsMemError::OutOfRange { .. })));
    assert!(!ran);
    assert_eq!(os.committed(), 0);
}
