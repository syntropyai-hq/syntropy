// SPDX-License-Identifier: MIT OR Apache-2.0

//! The sequence types of RFC-0001 §5.8.
//!
//! So far it holds [`Priority`], which PROD-E1-T10's request resolution needs first. DIST-E1-T14
//! adds the rest of §5.8 beside it: `SequenceControl`, `SequenceOpen`, `SequenceMeta`,
//! `SamplingParams`, `SequenceLimits`, `TokenFrame`, `TokenEntry` and the others.

use serde::{Deserialize, Serialize};

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
