// SPDX-License-Identifier: MIT OR Apache-2.0

//! The sequence types of RFC-0001 §5.8.
//!
//! So far it holds [`Priority`], which PROD-E1-T10's request resolution needs first. DIST-E1-T14
//! adds the rest of §5.8 beside it: `SequenceControl`, `SequenceOpen`, `SequenceMeta`,
//! `SamplingParams`, `SequenceLimits`, `TokenFrame`, `TokenEntry` and the others.

use serde::{Deserialize, Serialize};

/// A sequence's scheduling priority (RFC-0001 §5.8, `SequenceOpen::priority`; RFC-0012 §6.3).
/// RFC-0050 maps each API's fields onto it, and RFC-0054 §7 resolves a request's. Default:
/// `Normal`. Owner: RFC-0012.
///
/// In latency mode an `Interactive` sequence's decode step goes first, and an instance holds at
/// most one at a time (RFC-0054 §7).
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
