// SPDX-License-Identifier: MIT OR Apache-2.0

//! A shard fetch's priority (RFC-0041 §7), which a runner's `FetchShard` carries to the store
//! (RFC-0001 §3.5, note of 2026-10-23; DIST-E1-T15).

use serde::{Deserialize, Serialize};

/// How urgently the store fetches a shard, ahead of other fetches on the node (RFC-0041 §7).
/// RFC-0031 §6.10's note of 2026-10-23 maps each to the blob-plane class of its chunk requests.
/// COMP-E12-T04's fetch planner implements it. Default: `Normal`, a load. Owner: RFC-0041.
///
/// The derived `Ord` is service order, `Urgent < Normal < Background`: what is fetched first sorts
/// first, as [`Priority`](crate::Priority) orders sequences.
#[non_exhaustive]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum FetchPriority {
    /// Recovery after a node loss (RFC-0033 §6.3). Its chunk requests use the `replacement` class.
    Urgent,
    /// A load: the runner is `Fetching`. Its chunk requests use the `demand` class.
    #[default]
    Normal,
    /// Pre-staging a sibling or a replica. Its chunk requests use the `background` class.
    Background,
}
