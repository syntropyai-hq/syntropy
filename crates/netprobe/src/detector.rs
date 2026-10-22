// SPDX-License-Identifier: MIT OR Apache-2.0

//! What RFC-0030 §5.3's failure detector would have done with a run's silences, under RFC-0030
//! §5.4's LAN and WAN timings.
//!
//! The probe is two machines, so the detector is the one of a two-member cluster. A member
//! raises `Suspect` once its peer has been silent for `swim.miss_suspect` periods: 3 periods of
//! 1 s on LAN, 4 of 2 s on WAN. With one possible observer (K = 1), the suspicion timeout is
//! always `t_max`: 4 s on LAN, 20 s on WAN. So a silence of 3 s would raise `Suspect` on LAN and
//! one of 7 s would fail the peer; on WAN, 8 s and 28 s. The probe's echo requests go out every
//! period of its own (1 s by default, at most the LAN period), and a silence is the gap between
//! two echoes that arrive.
//!
//! What this leaves out: indirect probes, which need a third member; local health (§5.3.1),
//! which depends on the node's own scheduling; and the coordinator's own probe before it records
//! `NodeFailed`, which a two-member cluster's coordinator skips when it already holds the peer
//! `Suspect`.

use crate::report::{Detector, DetectorEvent, Verdict};

/// One set of timings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Timing {
    /// The heartbeat period (`swim.period`, `swim.wan_period`).
    pub period_ms: u64,
    /// Silence that raises `Suspect`: `swim.miss_suspect` (or `swim.wan_miss_suspect`) periods.
    pub suspect_after_ms: u64,
    /// Silence that fails the peer: the above plus `t_max`.
    pub fail_after_ms: u64,
}

/// RFC-0030 §5.4's LAN timings: 1 s period, `Suspect` after 3 periods, `t_max` 4 s.
pub const LAN: Timing = Timing {
    period_ms: 1_000,
    suspect_after_ms: 3_000,
    fail_after_ms: 7_000,
};

/// RFC-0030 §5.4's WAN timings: 2 s period, `Suspect` after 4 periods, `t_max` 20 s.
pub const WAN: Timing = Timing {
    period_ms: 2_000,
    suspect_after_ms: 8_000,
    fail_after_ms: 28_000,
};

/// The verdicts on `silences`, each `(start_ms, gap_ms)`.
pub fn evaluate(timing: Timing, silences: &[(u64, u64)]) -> Detector {
    let events: Vec<DetectorEvent> = silences
        .iter()
        .filter(|&&(_, gap)| gap >= timing.suspect_after_ms)
        .map(|&(at_ms, silence_ms)| DetectorEvent {
            at_ms,
            silence_ms,
            outcome: if silence_ms >= timing.fail_after_ms {
                Verdict::Fail
            } else {
                Verdict::Suspect
            },
        })
        .collect();
    Detector {
        period_ms: timing.period_ms,
        suspect_after_ms: timing.suspect_after_ms,
        fail_after_ms: timing.fail_after_ms,
        would_suspect: events.len() as u64,
        would_fail: events
            .iter()
            .filter(|event| event.outcome == Verdict::Fail)
            .count() as u64,
        events,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lan_and_wan_timings_judge_the_same_silences_differently() {
        // Gaps of 1 s (normal), 2.9 s, 3 s, 6.9 s, 7 s and 30 s.
        let silences = [
            (0, 1_000),
            (1_000, 2_900),
            (3_900, 3_000),
            (6_900, 6_900),
            (13_800, 7_000),
            (20_800, 30_000),
        ];
        let lan = evaluate(LAN, &silences);
        let verdicts: Vec<(u64, Verdict)> = lan
            .events
            .iter()
            .map(|event| (event.silence_ms, event.outcome))
            .collect();
        assert_eq!(
            verdicts,
            [
                (3_000, Verdict::Suspect),
                (6_900, Verdict::Suspect),
                (7_000, Verdict::Fail),
                (30_000, Verdict::Fail),
            ]
        );
        assert_eq!((lan.would_suspect, lan.would_fail), (4, 2));

        let wan = evaluate(WAN, &silences);
        assert_eq!((wan.would_suspect, wan.would_fail), (1, 1));
        assert_eq!(wan.events[0].at_ms, 20_800);
    }
}
