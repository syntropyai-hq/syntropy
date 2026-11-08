// SPDX-License-Identifier: MIT OR Apache-2.0

//! `MAX_UNION` and `MAX_PRED` for an epoch's doorbell layers (RFC-0023 §2, §9, §11).
//!
//! ```text
//! MAX_UNION = min(E, P_max · k)
//! MAX_PRED  = d_max · min(E, P_max · (k + predict_extra))
//! ```
//!
//! `E` is a MoE layer's routed expert count and `k` its top-k. `P_max` is the most tokens in
//! one pass of the plan: the largest `rows` of its buckets (RFC-0012 §2), prefill buckets
//! included. It is not `T_max`, which only sizes the verify bucket. `d_max` is the plan's
//! lookahead (`PrefetchWindow::lookahead_layers`, RFC-0023 §9), and `predict_extra` is
//! `experts.predict_extra` at the epoch's start (RFC-0023 §15). Both values are fixed for the
//! epoch: a plan delta may lower what feeds them, never raise it (RFC-0012 §7.1, "Pre-sizing").
//!
//! [`doorbell_layout`] evaluates the formulas from explicit inputs. [`largest_pass`] and
//! [`lookahead`] read `P_max` and `d_max` from a plan's fields, and [`plan_layout`] combines the
//! three.

#![forbid(unsafe_code)]

use std::fmt;

use syntropy_core::plan::{BatchingLimits, PlacementPlan, PrefetchWindow, SpeculativeConfig};
use syntropy_kernels::moe::DoorbellLayout;

/// A MoE layer's routing shape: what the formulas need from the manifest (RFC-0023 §2). For a
/// model whose MoE layers differ, the caller passes the largest `experts` and `top_k` of the
/// stage's layers, since one layout serves every doorbell layer of the epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MoeRouting {
    /// `E`: the layer's routed experts. An expert index is a `u16` (`ExpertIx`).
    pub experts: u16,
    /// `k`: experts selected per token.
    pub top_k: u16,
    /// `experts.predict_extra` at the epoch's start: extra predicted experts per lookahead
    /// layer (RFC-0023 §15; default `⌈k/2⌉`).
    pub predict_extra: u16,
}

/// Why the doorbell could not be sized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SizingError {
    /// `d_max` above 255: a `PredSlot`'s `ahead` is a `u8`.
    LookaheadTooDeep {
        /// The lookahead asked for.
        lookahead: u16,
    },
    /// `P_max` above 65,535: a `RouteSlot`'s `tokens` is a `u16`, so one expert's count in
    /// such a pass could wrap.
    PassTooLarge {
        /// `P_max` as given.
        largest_pass: u32,
    },
    /// `MAX_PRED` above 65,535: the header's `n_pred` is a `u16`.
    PredTooLarge {
        /// `MAX_PRED` as computed.
        max_pred: u64,
    },
}

impl fmt::Display for SizingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::LookaheadTooDeep { lookahead } => {
                write!(
                    f,
                    "lookahead of {lookahead} layers; a prediction's `ahead` holds at most 255"
                )
            }
            Self::PassTooLarge { largest_pass } => {
                write!(
                    f,
                    "a pass of {largest_pass} tokens; a route entry counts at most 65535"
                )
            }
            Self::PredTooLarge { max_pred } => {
                write!(
                    f,
                    "MAX_PRED would be {max_pred}; a doorbell holds at most 65535"
                )
            }
        }
    }
}

impl std::error::Error for SizingError {}

/// The doorbell layout for `P_max = largest_pass` tokens and `d_max = lookahead` layers
/// (RFC-0023 §2): `MAX_UNION = min(E, P_max · k)` and
/// `MAX_PRED = d_max · min(E, P_max · (k + predict_extra))`.
///
/// # Errors
///
/// [`SizingError`] when `P_max`, `d_max` or `MAX_PRED` does not fit the doorbell's fields.
/// `MAX_UNION` always fits, since it is at most `E`.
pub fn doorbell_layout(
    moe: MoeRouting,
    largest_pass: u32,
    lookahead: u16,
) -> Result<DoorbellLayout, SizingError> {
    if lookahead > u16::from(u8::MAX) {
        return Err(SizingError::LookaheadTooDeep { lookahead });
    }
    if largest_pass > u32::from(u16::MAX) {
        return Err(SizingError::PassTooLarge { largest_pass });
    }
    let e = u64::from(moe.experts);
    let p = u64::from(largest_pass);
    let max_union = e.min(p * u64::from(moe.top_k));
    let per_layer = e.min(p * (u64::from(moe.top_k) + u64::from(moe.predict_extra)));
    let max_pred = u64::from(lookahead) * per_layer;
    let max_pred = u16::try_from(max_pred).map_err(|_| SizingError::PredTooLarge { max_pred })?;
    let max_union = u16::try_from(max_union).expect("MAX_UNION is at most E, a u16");
    Ok(DoorbellLayout::new(max_union, max_pred))
}

/// `P_max`: the most tokens in one pass of any bucket the plan's batching and speculation
/// imply (RFC-0012 §2), prefill buckets included:
///
/// - `Decode`: 1 row;
/// - `Verify`: `T_max = max_window + 1`, or the draft tree's nodes plus the root when a tree
///   is larger;
/// - `Batch`: up to `micro_batch × T`, with `T = T_max` when speculation is on and 1 otherwise;
/// - `Prefill` and `PrefillStream`: every rung of `prefill_ladder` and `prefill_chunk` itself.
///
/// The latency-mode `Batch { rows }` buckets of RFC-0012 §4.6 come from `sched`'s
/// configuration, not from the plan. A caller that captures them passes the larger of this and
/// their rows to [`doorbell_layout`].
#[must_use]
pub fn largest_pass(batching: &BatchingLimits, speculative: Option<&SpeculativeConfig>) -> u32 {
    let t_max = speculative.map_or(1, |s| {
        let chain = u32::from(s.max_window) + 1;
        let tree = s.tree.as_ref().map_or(0, |t| t.parents.len() as u32 + 1);
        chain.max(tree)
    });
    let batch = u32::from(batching.micro_batch).saturating_mul(t_max);
    let prefill = batching
        .prefill_ladder
        .iter()
        .copied()
        .chain([batching.prefill_chunk])
        .max()
        .unwrap_or(0);
    1.max(t_max).max(batch).max(prefill)
}

/// `d_max`: the deepest `lookahead_layers` of the plan's prefetch windows (RFC-0023 §9), or 0
/// when the plan has none, so the doorbell holds no predictions.
#[must_use]
pub fn lookahead(prefetch: &[PrefetchWindow]) -> u16 {
    prefetch
        .iter()
        .map(|w| w.lookahead_layers)
        .max()
        .unwrap_or(0)
}

/// [`doorbell_layout`] for a committed plan: `P_max` from [`largest_pass`] and `d_max` from
/// [`lookahead`].
///
/// # Errors
///
/// As [`doorbell_layout`].
pub fn plan_layout(plan: &PlacementPlan, moe: MoeRouting) -> Result<DoorbellLayout, SizingError> {
    doorbell_layout(
        moe,
        largest_pass(&plan.batching, plan.speculative.as_ref()),
        lookahead(&plan.prefetch),
    )
}

#[cfg(test)]
mod tests {
    use smallvec::smallvec;
    use syntropy_core::plan::{Acceptance, DraftSource, TreeShape};
    use syntropy_core::units::Bytes;
    use syntropy_core::{DeviceId, NodeId, TierId};

    use super::*;

    /// Qwen3-30B-A3B-class routing: 128 experts, top-8, `predict_extra = ⌈8/2⌉ = 4`.
    const QWEN: MoeRouting = MoeRouting {
        experts: 128,
        top_k: 8,
        predict_extra: 4,
    };

    fn batching(chunk: u32, ladder: &[u32], micro_batch: u16) -> BatchingLimits {
        BatchingLimits {
            max_sequences: 4,
            prefill_chunk: chunk,
            micro_batch,
            prefill_ladder: ladder.iter().copied().collect(),
        }
    }

    fn spec(max_window: u16, tree_nodes: usize) -> SpeculativeConfig {
        SpeculativeConfig {
            source: DraftSource::Mtp { heads: 1 },
            max_window,
            tree: (tree_nodes > 0).then(|| TreeShape {
                parents: vec![0; tree_nodes],
            }),
            acceptance: Acceptance::Exact,
        }
    }

    fn window(lookahead_layers: u16) -> PrefetchWindow {
        PrefetchWindow {
            from: TierId {
                node: NodeId([0; 32]),
                ordinal: 2,
            },
            to: TierId {
                node: NodeId([0; 32]),
                ordinal: 1,
            },
            max_inflight: Bytes(1 << 20),
            lookahead_layers,
        }
    }

    #[test]
    fn decode_only_pass_selects_k_experts() {
        // P_max = 1: MAX_UNION = min(128, 1 · 8) = 8; MAX_PRED = 1 · min(128, 1 · 12) = 12.
        let l = doorbell_layout(QWEN, 1, 1).unwrap();
        assert_eq!((l.max_union(), l.max_pred()), (8, 12));
        // d_max = 2 doubles MAX_PRED and leaves MAX_UNION.
        let l = doorbell_layout(QWEN, 1, 2).unwrap();
        assert_eq!((l.max_union(), l.max_pred()), (8, 24));
        // No lookahead: no predictions.
        let l = doorbell_layout(QWEN, 1, 0).unwrap();
        assert_eq!((l.max_union(), l.max_pred()), (8, 0));
    }

    #[test]
    fn a_large_pass_is_capped_at_the_expert_count() {
        // P_max = 12: 12 · 8 = 96 < 128, and 12 · 12 = 144 is capped at 128.
        let l = doorbell_layout(QWEN, 12, 2).unwrap();
        assert_eq!((l.max_union(), l.max_pred()), (96, 256));
        // A 4,096-token prefill chunk touches at most every expert.
        let l = doorbell_layout(QWEN, 4096, 2).unwrap();
        assert_eq!((l.max_union(), l.max_pred()), (128, 256));
    }

    #[test]
    fn the_formulas_hold_over_their_inputs() {
        for experts in [1u16, 8, 64, 128, 256, 384] {
            for top_k in [1u16, 2, 4, 8] {
                for predict_extra in [0u16, 1, 4] {
                    for p in [1u32, 2, 3, 4, 8, 64, 1024, 8192] {
                        for d in [0u16, 1, 2, 3] {
                            let moe = MoeRouting {
                                experts,
                                top_k,
                                predict_extra,
                            };
                            let l = doorbell_layout(moe, p, d).unwrap();
                            let e = u64::from(experts);
                            let union = e.min(u64::from(p) * u64::from(top_k));
                            let pred = u64::from(d)
                                * e.min(u64::from(p) * u64::from(top_k + predict_extra));
                            assert_eq!(u64::from(l.max_union()), union);
                            assert_eq!(u64::from(l.max_pred()), pred);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn fields_that_do_not_fit_are_refused() {
        assert_eq!(
            doorbell_layout(QWEN, 1, 256),
            Err(SizingError::LookaheadTooDeep { lookahead: 256 })
        );
        let wide = MoeRouting {
            experts: u16::MAX,
            top_k: 8,
            predict_extra: 4,
        };
        assert_eq!(
            doorbell_layout(wide, 65_535, 2),
            Err(SizingError::PredTooLarge {
                max_pred: 2 * u64::from(u16::MAX)
            })
        );
        assert_eq!(
            doorbell_layout(wide, 65_535, 1).unwrap().max_union(),
            u16::MAX
        );
    }

    /// A `RouteSlot` counts a pass's tokens in a `u16`: 65,535 fits, 65,536 would wrap.
    #[test]
    fn a_pass_above_a_route_entrys_count_is_refused() {
        let l = doorbell_layout(QWEN, 65_535, 2).unwrap();
        assert_eq!((l.max_union(), l.max_pred()), (128, 256));
        assert_eq!(
            doorbell_layout(QWEN, 65_536, 2),
            Err(SizingError::PassTooLarge {
                largest_pass: 65_536
            })
        );
        let mut plan_batching = batching(65_536, &[64], 1);
        assert_eq!(largest_pass(&plan_batching, None), 65_536);
        plan_batching.prefill_chunk = 65_535;
        assert_eq!(largest_pass(&plan_batching, None), 65_535);
    }

    #[test]
    fn largest_pass_is_decode_without_buckets() {
        assert_eq!(largest_pass(&BatchingLimits::default(), None), 1);
    }

    #[test]
    fn largest_pass_includes_prefill_buckets() {
        // The chunk is the top rung.
        assert_eq!(
            largest_pass(&batching(2048, &[64, 256, 1024], 1), None),
            2048
        );
        // A ladder rung above the chunk would be a malformed plan, but it still bounds a pass.
        assert_eq!(largest_pass(&batching(512, &[64, 4096], 1), None), 4096);
    }

    #[test]
    fn largest_pass_includes_verify_and_batch_buckets() {
        // γ_max = 3: T_max = 4, above a 2-token chunk.
        assert_eq!(largest_pass(&batching(2, &[], 1), Some(&spec(3, 0))), 4);
        // A draft tree of 9 nodes plus its root beats the chain.
        assert_eq!(largest_pass(&batching(2, &[], 1), Some(&spec(3, 9))), 10);
        // Throughput mode: 16 sequences × T_max 4 = 64 rows.
        assert_eq!(largest_pass(&batching(32, &[], 16), Some(&spec(3, 0))), 64);
        // Without speculation T = 1: 16 rows.
        assert_eq!(largest_pass(&batching(8, &[], 16), None), 16);
    }

    #[test]
    fn lookahead_is_the_deepest_window() {
        assert_eq!(lookahead(&[]), 0);
        assert_eq!(lookahead(&[window(1), window(2), window(1)]), 2);
    }

    #[test]
    fn plan_layout_reads_the_plan() {
        use syntropy_core::plan::{PlanEpoch, PredictedCost};
        use syntropy_core::policy::{ExactnessPolicy, Objective};
        use syntropy_core::{Blake3, GraphVersion, InstanceId, ModelId};

        let mut plan = PlacementPlan {
            epoch: PlanEpoch::default(),
            instance: InstanceId(1),
            model: ModelId(Blake3([1; 32])),
            graph: GraphVersion(1),
            policy: Blake3([7; 32]),
            objective: Objective::default(),
            exactness: ExactnessPolicy::default(),
            stages: Vec::new(),
            tensors: Vec::new(),
            expert_cache: Vec::new(),
            kv: Vec::new(),
            prefetch: vec![window(1), window(2)],
            scratch: vec![(
                DeviceId {
                    node: NodeId([0; 32]),
                    ordinal: 0,
                },
                Bytes(1),
            )],
            speculative: Some(spec(1, 0)),
            batching: batching(4, &[], 1),
            predicted: PredictedCost::default(),
            assumptions: Vec::new(),
            ext: syntropy_core::plan::PlacementPlanExt::default(),
        };
        // P_max = max(T_max 2, prefill 4) = 4: MAX_UNION = 32, MAX_PRED = 2 · min(128, 48) = 96.
        let l = plan_layout(&plan, QWEN).unwrap();
        assert_eq!((l.max_union(), l.max_pred()), (32, 96));
        assert_eq!(l, doorbell_layout(QWEN, 4, 2).unwrap());

        // A 256-token prefill bucket raises P_max past E / k.
        plan.batching.prefill_ladder = smallvec![64, 256];
        let l = plan_layout(&plan, QWEN).unwrap();
        assert_eq!((l.max_union(), l.max_pred()), (128, 256));
    }
}
