/* SPDX-License-Identifier: MIT OR Apache-2.0 */

/*
 * The MoE doorbell a router op writes for one layer: its fixed header, `syn_route_slot` and
 * `syn_pred_slot` (RFC-0023 §2; RFC-0040 §3.4; ADR-0404 item 3), and the codes of
 * `kernels::moe::assign` (RFC-0040 §3.4; ADR-0404 item 4).
 *
 * This header is the one definition of these layouts. `crates/kernels/src/moe/mirror.rs` is the
 * Rust mirror generated from it, and the `syntropy-kernels` tests fail when the two differ
 * (`SYNTROPY_BLESS_GENERATED=1` rewrites the mirror). The tests also compile it as C11 and
 * C++11 when a compiler is present (always in CI). The generator reads the subset of C that
 * `step_desc.h` describes, with `uint8_t`, `uint16_t` and `float` members as well; it skips the
 * function-like macros below, one line each, which the tests check against the Rust layout
 * with a C compiler.
 *
 * A doorbell layer is `syn_doorbell_header`, then `syn_route_slot route[MAX_UNION]`, then
 * `syn_pred_slot pred[MAX_PRED]`, with no padding between them: each part's size is a multiple
 * of the next part's alignment, 4. `MAX_UNION` and `MAX_PRED` are fixed for the plan epoch
 * (RFC-0023 §2). Entries past `n_route` and `n_pred` are never read. All values are
 * little-endian.
 */

#ifndef SYNTROPY_MOE_DOORBELL_H
#define SYNTROPY_MOE_DOORBELL_H

#include <stddef.h>
#include <stdint.h>

/// The version of the layouts and codes below. A change to any struct, to the order of a
/// doorbell layer's parts, or to a code bumps it.
#define SYN_MOE_DOORBELL_ABI_VERSION 1

/// `syn_route_slot.cache_slot` of an expert that is not resident in the device's cache.
#define SYN_MOE_MISS 4294967295u

/// The `assign` code of an expert computed from its cache slot (`HIT`). Every code below
/// `SYN_MOE_CPU` is a copy-slot index (`COPY`).
#define SYN_MOE_HIT 4294967295u
/// The `assign` code of an expert the CPU pools compute in place (`CPU`).
#define SYN_MOE_CPU 4294967294u

/// Byte offset of `route` in a doorbell layer.
#define SYN_DOORBELL_ROUTE_OFFSET() (sizeof(struct syn_doorbell_header))
/// Byte offset of `pred` in a doorbell layer whose `route` holds `max_union` entries.
#define SYN_DOORBELL_PRED_OFFSET(max_union) (SYN_DOORBELL_ROUTE_OFFSET() + (size_t)(max_union) * sizeof(struct syn_route_slot))
/// Size in bytes of a doorbell layer for `max_union` and `max_pred`.
#define SYN_DOORBELL_LAYER_SIZE(max_union, max_pred) (SYN_DOORBELL_PRED_OFFSET(max_union) + (size_t)(max_pred) * sizeof(struct syn_pred_slot))

/// The fixed header of a doorbell layer (RFC-0023 §2 `DoorbellLayer`). It is followed by
/// `syn_route_slot route[MAX_UNION]` and `syn_pred_slot pred[MAX_PRED]`.
struct syn_doorbell_header {
    /// The low 32 bits of the step number, `syn_step_desc.step`: a payload check the host
    /// asserts, not a flag (RFC-0023 §6).
    uint32_t seq;
    /// Distinct experts selected in this pass: the valid entries of `route`.
    uint16_t n_route;
    /// The valid entries of `pred`.
    uint16_t n_pred;
};

/// One expert the router selected in this pass (RFC-0023 §2).
struct syn_route_slot {
    /// The expert's index in its layer.
    uint16_t expert;
    /// Tokens in the pass that selected it (at least 1).
    uint16_t tokens;
    /// Its slot in the device's cache, or `SYN_MOE_MISS`.
    uint32_t cache_slot;
};

/// One predicted expert of a later layer (RFC-0023 §2, §9).
struct syn_pred_slot {
    /// j: the slot predicts layer l + j, 1 <= j <= d.
    uint8_t ahead;
    /// Zero.
    uint8_t _pad;
    /// The predicted expert's index in layer l + j.
    uint16_t expert;
    /// Router probability mass, summed over the pass's tokens.
    float score;
};

/* SYNTROPY_MOE_DOORBELL_H */
#endif
