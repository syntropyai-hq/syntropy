/* SPDX-License-Identifier: MIT OR Apache-2.0 */

/*
 * @generated from the specification in `crates/kernels/src/moe/spec.rs` by the
 * `syntropy-kernels` tests. Do not edit: change the specification and run
 * `SYNTROPY_BLESS_GENERATED=1 cargo test -p syntropy-kernels moe::`.
 *
 * The split rule in C (RFC-0023 §6; RFC-0040 §3.4; ADR-0404 item 4), printed from the
 * same specification as `crates/kernels/src/moe/split.rs`. The tests compile it as C11
 * and C++11 and run it against the Rust version.
 */

#ifndef SYNTROPY_MOE_ASSIGN_H
#define SYNTROPY_MOE_ASSIGN_H

#include <stdint.h>

#include "syntropy/moe_doorbell.h"

/// Splits one MoE layer's routed experts between the three sites (RFC-0023 §6;
/// RFC-0040 §3.4): writes `out[i]` for each of the `n_route` entries of `route`.
///
/// - `MOE_HIT` when the entry's `cache_slot` is not `MOE_MISS`.
/// - A copy-slot index `c` when the expert missed, is marked in `eligible` (bit e % 32
///   of word e / 32, for e below `n_experts`), and is the c-th such expert in the walk's
///   order (more tokens first, then the lower expert index, then the earlier entry),
///   with `c < copy_slots` and `(c + 1) * expert_bytes <= b_copy`.
/// - `MOE_CPU` for every other miss.
static inline void syn_moe_assign(
    const struct syn_route_slot *route,
    uint32_t n_route,
    const uint32_t *eligible,
    uint32_t n_experts,
    uint32_t b_copy,
    uint64_t expert_bytes,
    uint32_t copy_slots,
    uint32_t *out) {
    for (uint32_t i = 0u; i < n_route; i++) {
        const struct syn_route_slot r = route[i];
        const uint32_t e = (uint32_t)r.expert;
        if (r.cache_slot != SYN_MOE_MISS) {
            out[i] = SYN_MOE_HIT;
        } else if ((e < n_experts) && (((eligible[e >> 5u] >> (e & 31u)) & 1u) == 1u)) {
            uint32_t c = 0u;
            for (uint32_t j = 0u; j < n_route; j++) {
                const struct syn_route_slot q = route[j];
                const uint32_t f = (uint32_t)q.expert;
                if (((q.cache_slot == SYN_MOE_MISS) && ((f < n_experts) && (((eligible[f >> 5u] >> (f & 31u)) & 1u) == 1u))) && ((q.tokens > r.tokens) || ((q.tokens == r.tokens) && ((f < e) || ((f == e) && (j < i)))))) {
                    c += 1u;
                }
            }
            if ((c < copy_slots) && ((expert_bytes <= ((uint64_t)b_copy)) && (((((uint64_t)c) + 1u) * expert_bytes) <= ((uint64_t)b_copy)))) {
                out[i] = c;
            } else {
                out[i] = SYN_MOE_CPU;
            }
        } else {
            out[i] = SYN_MOE_CPU;
        }
    }
}

/* SYNTROPY_MOE_ASSIGN_H */
#endif
