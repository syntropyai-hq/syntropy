// SPDX-License-Identifier: MIT OR Apache-2.0
// @generated from the specification in `src/moe/spec.rs` by the `syntropy-kernels`
// tests. Do not edit: change the specification and run `SYNTROPY_BLESS_GENERATED=1
// cargo test -p syntropy-kernels moe::`.

//! The split rule in Rust (RFC-0023 §6; RFC-0040 §3.4; ADR-0404 item 4), printed from
//! the same specification as `include/syntropy/moe_assign.h`.

use super::mirror::{MOE_CPU, MOE_HIT, MOE_MISS, RouteSlot};

/// Splits one MoE layer's routed experts between the three sites (RFC-0023 §6;
/// RFC-0040 §3.4): writes `out[i]` for each of the `n_route` entries of `route`.
///
/// - `MOE_HIT` when the entry's `cache_slot` is not `MOE_MISS`.
/// - A copy-slot index `c` when the expert missed, is marked in `eligible` (bit e % 32
///   of word e / 32, for e below `n_experts`), and is the c-th such expert in the walk's
///   order (more tokens first, then the lower expert index, then the earlier entry),
///   with `c < copy_slots` and `(c + 1) * expert_bytes <= b_copy`.
/// - `MOE_CPU` for every other miss.
// The printer keeps the specification's shape: indexed loops and one parameter per
// kernel argument.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub(super) fn assign_codes(
    route: &[RouteSlot],
    n_route: u32,
    eligible: &[u32],
    n_experts: u32,
    b_copy: u32,
    expert_bytes: u64,
    copy_slots: u32,
    out: &mut [u32],
) {
    for i in 0..n_route {
        let r: RouteSlot = route[i as usize];
        let e: u32 = r.expert as u32;
        if r.cache_slot != MOE_MISS {
            out[i as usize] = MOE_HIT;
        } else if (e < n_experts) && (((eligible[(e >> 5) as usize] >> (e & 31)) & 1) == 1) {
            let mut c: u32 = 0;
            for j in 0..n_route {
                let q: RouteSlot = route[j as usize];
                let f: u32 = q.expert as u32;
                if ((q.cache_slot == MOE_MISS) && ((f < n_experts) && (((eligible[(f >> 5) as usize] >> (f & 31)) & 1) == 1))) && ((q.tokens > r.tokens) || ((q.tokens == r.tokens) && ((f < e) || ((f == e) && (j < i))))) {
                    c += 1;
                }
            }
            if (c < copy_slots) && ((expert_bytes <= (b_copy as u64)) && ((((c as u64) + 1) * expert_bytes) <= (b_copy as u64))) {
                out[i as usize] = c;
            } else {
                out[i as usize] = MOE_CPU;
            }
        } else {
            out[i as usize] = MOE_CPU;
        }
    }
}
