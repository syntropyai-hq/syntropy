// SPDX-License-Identifier: MIT OR Apache-2.0

// @generated from the specification in `crates/kernels/src/moe/spec.rs` and the layouts
// of `crates/kernels/include/syntropy/moe_doorbell.h` by the `syntropy-kernels` tests. Do
// not edit: change the specification or the header and run
// `SYNTROPY_BLESS_GENERATED=1 cargo test -p syntropy-kernels moe::`.
//
// The split rule in Metal Shading Language (RFC-0023 §6; RFC-0040 §3.4; ADR-0404 item 4),
// printed from the same specification as `crates/kernels/src/moe/split.rs` and
// `crates/kernels/include/syntropy/moe_assign.h`. The `syntropy-kernels-metal` tests
// compile it with the Metal compiler and run it against the Rust version.

#include <metal_stdlib>

using namespace metal;

/// The version of the layouts and codes below. A change to any struct, to the order of a
/// doorbell layer's parts, or to a code bumps it.
constant uint SYN_MOE_DOORBELL_ABI_VERSION = 1u;

/// `syn_route_slot.cache_slot` of an expert that is not resident in the device's cache.
constant uint SYN_MOE_MISS = 4294967295u;

/// The `assign` code of an expert computed from its cache slot (`HIT`). Every code below
/// `SYN_MOE_CPU` is a copy-slot index (`COPY`).
constant uint SYN_MOE_HIT = 4294967295u;

/// The `assign` code of an expert the CPU pools compute in place (`CPU`).
constant uint SYN_MOE_CPU = 4294967294u;

/// The fixed header of a doorbell layer (RFC-0023 §2 `DoorbellLayer`). It is followed by
/// `syn_route_slot route[MAX_UNION]` and `syn_pred_slot pred[MAX_PRED]`.
struct syn_doorbell_header {
    /// The low 32 bits of the step number, `syn_step_desc.step`: a payload check the host
    /// asserts, not a flag (RFC-0023 §6).
    uint seq;
    /// Distinct experts selected in this pass: the valid entries of `route`.
    ushort n_route;
    /// The valid entries of `pred`.
    ushort n_pred;
};

/// One expert the router selected in this pass (RFC-0023 §2).
struct syn_route_slot {
    /// The expert's index in its layer.
    ushort expert;
    /// Tokens in the pass that selected it (at least 1).
    ushort tokens;
    /// Its slot in the device's cache, or `SYN_MOE_MISS`.
    uint cache_slot;
};

/// One predicted expert of a later layer (RFC-0023 §2, §9).
struct syn_pred_slot {
    /// j: the slot predicts layer l + j, 1 <= j <= d.
    uchar ahead;
    /// Zero.
    uchar _pad;
    /// The predicted expert's index in layer l + j.
    ushort expert;
    /// Router probability mass, summed over the pass's tokens.
    float score;
};

// The layout `moe_doorbell.h` gives each struct, checked by the Metal compiler.
static_assert(sizeof(syn_doorbell_header) == 8, "syn_doorbell_header size");
static_assert(alignof(syn_doorbell_header) == 4, "syn_doorbell_header align");
static_assert(__builtin_offsetof(syn_doorbell_header, seq) == 0, "syn_doorbell_header.seq");
static_assert(__builtin_offsetof(syn_doorbell_header, n_route) == 4, "syn_doorbell_header.n_route");
static_assert(__builtin_offsetof(syn_doorbell_header, n_pred) == 6, "syn_doorbell_header.n_pred");
static_assert(sizeof(syn_route_slot) == 8, "syn_route_slot size");
static_assert(alignof(syn_route_slot) == 4, "syn_route_slot align");
static_assert(__builtin_offsetof(syn_route_slot, expert) == 0, "syn_route_slot.expert");
static_assert(__builtin_offsetof(syn_route_slot, tokens) == 2, "syn_route_slot.tokens");
static_assert(__builtin_offsetof(syn_route_slot, cache_slot) == 4, "syn_route_slot.cache_slot");
static_assert(sizeof(syn_pred_slot) == 8, "syn_pred_slot size");
static_assert(alignof(syn_pred_slot) == 4, "syn_pred_slot align");
static_assert(__builtin_offsetof(syn_pred_slot, ahead) == 0, "syn_pred_slot.ahead");
static_assert(__builtin_offsetof(syn_pred_slot, _pad) == 1, "syn_pred_slot._pad");
static_assert(__builtin_offsetof(syn_pred_slot, expert) == 2, "syn_pred_slot.expert");
static_assert(__builtin_offsetof(syn_pred_slot, score) == 4, "syn_pred_slot.score");

/// Splits one MoE layer's routed experts between the three sites (RFC-0023 §6;
/// RFC-0040 §3.4): writes `out[i]` for each of the `n_route` entries of `route`.
///
/// - `MOE_HIT` when the entry's `cache_slot` is not `MOE_MISS`.
/// - A copy-slot index `c` when the expert missed, is marked in `eligible` (bit e % 32
///   of word e / 32, for e below `n_experts`), and is the c-th such expert in the walk's
///   order (more tokens first, then the lower expert index, then the earlier entry),
///   with `c < copy_slots` and `(c + 1) * expert_bytes <= b_copy`.
/// - `MOE_CPU` for every other miss.
///
/// This function decides entry `i` alone, reading every entry of `route`: it is the body
/// of the specification's loop over `i`.
static inline void syn_moe_assign_entry(
    device const syn_route_slot *route,
    uint n_route,
    device const uint *eligible,
    uint n_experts,
    uint b_copy,
    ulong expert_bytes,
    uint copy_slots,
    device uint *out,
    uint i) {
    const syn_route_slot r = route[i];
    const uint e = (uint)r.expert;
    if (r.cache_slot != SYN_MOE_MISS) {
        out[i] = SYN_MOE_HIT;
    } else if ((e < n_experts) && (((eligible[e >> 5u] >> (e & 31u)) & 1u) == 1u)) {
        uint c = 0u;
        for (uint j = 0u; j < n_route; j++) {
            const syn_route_slot q = route[j];
            const uint f = (uint)q.expert;
            if (((q.cache_slot == SYN_MOE_MISS) && ((f < n_experts) && (((eligible[f >> 5u] >> (f & 31u)) & 1u) == 1u))) && ((q.tokens > r.tokens) || ((q.tokens == r.tokens) && ((f < e) || ((f == e) && (j < i)))))) {
                c += 1u;
            }
        }
        if ((c < copy_slots) && ((expert_bytes <= ((ulong)b_copy)) && (((((ulong)c) + 1u) * expert_bytes) <= ((ulong)b_copy)))) {
            out[i] = c;
        } else {
            out[i] = SYN_MOE_CPU;
        }
    } else {
        out[i] = SYN_MOE_CPU;
    }
}

/// `syn_moe_assign_entry` for every route entry, one thread per entry: dispatch at least
/// `n_route` threads in one dimension. Threads at or past `n_route` write nothing.
kernel void syn_moe_assign(
    device const syn_route_slot *route [[buffer(0)]],
    constant uint &n_route [[buffer(1)]],
    device const uint *eligible [[buffer(2)]],
    constant uint &n_experts [[buffer(3)]],
    constant uint &b_copy [[buffer(4)]],
    constant ulong &expert_bytes [[buffer(5)]],
    constant uint &copy_slots [[buffer(6)]],
    device uint *out [[buffer(7)]],
    uint i [[thread_position_in_grid]]) {
    if (i < n_route) {
        syn_moe_assign_entry(route, n_route, eligible, n_experts, b_copy, expert_bytes, copy_slots, out, i);
    }
}
