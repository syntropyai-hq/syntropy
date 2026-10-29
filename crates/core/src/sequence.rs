// SPDX-License-Identifier: MIT OR Apache-2.0

//! The sequence types of RFC-0001 §5.8 and RFC-0032 §3.6: what ingress sends to open and control
//! a sequence, and what the tail sends back (DIST-E1-T14).
//!
//! - **Control.** [`SequenceControl`], RFC-0033 §11's `Reconcile` among its variants, and
//!   [`SequenceOpen`], with the types an open carries:
//!   [`SamplingParams`], [`SequenceLimits`], [`PrefixRef`], [`PrefixScope`], [`ResumePoint`],
//!   [`Priority`] and the rest. [`SequenceMeta`] is an open without its prompt tokens, which
//!   travels downstream on the sequence's first frame.
//! - **Tokens.** [`TokenFrame`] and [`TokenEntry`], with [`FinishReason`], [`LogprobBlock`] and
//!   a draft's [`DraftNode`]s.
//! - **Tracing.** The one [`TraceContext`], 26 bytes on the wire (RFC-0053 §3.2).
//!
//! The types live in `core` because the head's and tail's step loops in `exec` and `sched` read
//! them from the M0 rings, and RFC-0001 §4.2 rule 3 keeps those crates off `net`. `net` encodes
//! them on the wire (RFC-0031 §6.3, §6.5, §6.6) without defining them, and keeps its zero-copy
//! views under names of their own (RFC-0001 §5.8 and §5.13, notes of 2026-10-22).
//!
//! They are plain data with serde. A field RFC-0032 §3.6 or a later amendment added has a
//! default, so an encoding written without it still decodes.

use serde::{Deserialize, Serialize};

mod open;
#[cfg(test)]
mod serde_tests;
mod token;
mod trace;

pub use open::{
    CancelReason, GrammarRef, PoolingMode, PrefixKey, PrefixRef, PrefixScope, ResumeCause,
    ResumePoint, SamplingParams, SequenceControl, SequenceLimits, SequenceMeta, SequenceOpen,
    TailId, TokenSlice,
};
pub use token::{DraftNode, FinishReason, LogprobBlock, TokenEntry, TokenFrame};
pub use trace::TraceContext;

/// A sequence's scheduling priority (RFC-0001 §5.8, `SequenceOpen::priority`; RFC-0012 §6.3).
/// RFC-0050 maps each API's fields onto it, and RFC-0054 §7 resolves a request's. Default:
/// `Normal`, which RFC-0054 §7 gives every request that asks for nothing else. Owner: RFC-0012.
///
/// In latency mode an `Interactive` sequence's decode step goes first, and an instance holds at
/// most one at a time (RFC-0054 §7).
///
/// - **Serde form.** The variant names as written (`"Interactive"`, `"Normal"`,
///   `"Background"`), as [`Fidelity`](crate::Fidelity) uses its own. The API's wire form is
///   lowercase (`"priority": "background"`, RFC-0050 §3.4 and §13.1), and the API front ends map
///   between the two.
/// - **Order.** The derived `Ord` is service order, `Interactive < Normal < Background`: what
///   is served first sorts first. So the highest priority in a group, by which RFC-0012 §6.3
///   orders throughput-mode groups, is the group's `min()`, not its `max()`.
#[non_exhaustive]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum Priority {
    /// The one sequence whose step goes first.
    Interactive,
    /// The default.
    #[default]
    Normal,
    /// Runs when the others leave room, such as OpenAI's `service_tier = "flex"`.
    Background,
}

/// How a sequence ended on a runner, which the runner reports in `SequenceEnded` (RFC-0001 §3.5,
/// note of 2026-10-23; DIST-E1-T15). Over the command channel it is postcard-encoded, so a variant
/// is only ever appended. Owner: RFC-0001.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SequenceOutcome {
    /// It finished.
    Finished,
    /// It failed, with the error's class (RFC-0001 §5.12).
    Failed(crate::ErrorClass),
    /// It moved to the plan epoch `to` (RFC-0033 §5), where it continues.
    Migrated {
        /// The epoch it continues on.
        to: crate::PlanEpoch,
    },
}

#[cfg(test)]
mod tests {
    use super::Priority;

    /// The derived order is service order, so the highest priority in a group is its `min()`.
    #[test]
    fn priority_orders_by_service() {
        assert!(Priority::Interactive < Priority::Normal);
        assert!(Priority::Normal < Priority::Background);
        let group = [
            Priority::Background,
            Priority::Interactive,
            Priority::Normal,
        ];
        assert_eq!(group.iter().min(), Some(&Priority::Interactive));
    }

    /// `Normal` is the default, and the variant names are the serialised form, so they never
    /// change.
    #[test]
    fn priority_defaults_to_normal_and_keeps_its_names() {
        assert_eq!(Priority::default(), Priority::Normal);
        for (priority, name) in [
            (Priority::Interactive, "\"Interactive\""),
            (Priority::Normal, "\"Normal\""),
            (Priority::Background, "\"Background\""),
        ] {
            assert_eq!(serde_json::to_string(&priority).unwrap(), name);
            assert_eq!(serde_json::from_str::<Priority>(name).unwrap(), priority);
        }
    }
}
