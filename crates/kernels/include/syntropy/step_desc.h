/* SPDX-License-Identifier: MIT OR Apache-2.0 */

/*
 * The step descriptor and the per-sequence sampler state (RFC-0040 §3.5, §3.8; ADR-0404
 * items 1 and 2).
 *
 * This header is the one definition of both layouts. `crates/kernels/src/step/mirror.rs` is the
 * Rust mirror generated from it, and the `syntropy-kernels` tests fail when the two differ
 * (`SYNTROPY_BLESS_GENERATED=1` rewrites the mirror). The tests also compile it as C11 and
 * C++11 when a compiler is present (always in CI). It is not checked as Metal Shading Language
 * or Slang: the Metal and Vulkan providers verify or port it when they land.
 *
 * The generator reads a small subset of C and refuses anything else: `#define NAME value`;
 * `#ifndef`, `#if`, `#else`, `#endif` and `#include` outside structs, with structs and
 * constants only at the include guard's level; structs whose members are `uint32_t`, `int32_t`
 * or `uint64_t`, one per line, with `SYN_ALIGNAS(n)` on the first member to raise the struct's
 * alignment; and comments that fill their lines. A `///` comment documents the item below it.
 *
 * Every value that changes from step to step is in the descriptor. A step descriptor is this
 * header followed by its arrays, at the byte offsets the header names; the host sizes the
 * whole block once at Warming from the batching limits, so the offsets are fixed for the plan
 * epoch. All values are little-endian.
 */

#ifndef SYNTROPY_STEP_DESC_H
#define SYNTROPY_STEP_DESC_H

#include <stdint.h>

#if defined(__cplusplus)
#define SYN_ALIGNAS(n) alignas(n)
#else
#define SYN_ALIGNAS(n) _Alignas(n)
#endif

/// The version of the layouts below. A change to either struct, or to the array order a
/// step descriptor follows, bumps it.
#define SYN_STEP_DESC_ABI_VERSION 1

/// The alignment of a step descriptor, and the granularity its total size is rounded to.
#define SYN_STEP_DESC_ALIGN 64

/// `syn_step_desc.flags`: this step carries the last chunk of a prefill.
#define SYN_STEP_FLAG_PREFILL_LAST (1u << 0)
/// `syn_step_desc.flags`: the rows form a token tree, and `tree_parent` holds its edges.
#define SYN_STEP_FLAG_TREE (1u << 1)

/// The fixed header of a step descriptor (RFC-0040 §3.5). It is followed, at the offsets it
/// names, by `u32 row_seq[R]`, `u32 row_pos[R]`, `u32 row_token[R]`, `i32 tree_parent[R]`,
/// `u32 row_kv_write_from[R]`, `u32 seq_kv_len[S]`, `u32 seq_page_table[S]`,
/// `u32 seq_state_slot[S]`, `syn_sampler_state seq_sampler[S]`, `u32 moe_balance[L_moe]` and
/// `u32 tap_rows[R_tap]`.
struct syn_step_desc {
    /// The step number, `ActivationFrame.step`.
    SYN_ALIGNAS(64) uint64_t step;
    /// Rows this step computes; at most the bucket's row count.
    uint32_t rows_active;
    /// Sequences the active rows belong to.
    uint32_t seqs_active;
    /// The MoE table epoch the step reads.
    uint32_t moe_epoch;
    /// `SYN_STEP_FLAG_*` bits.
    uint32_t flags;
    /// Byte offset of `row_seq`: each row's sequence index.
    uint32_t off_row_seq;
    /// Byte offset of `row_pos`: each row's position in its sequence.
    uint32_t off_row_pos;
    /// Byte offset of `row_token`: each row's input token.
    uint32_t off_row_token;
    /// Byte offset of `tree_parent`: each row's parent row in a token tree, or -1.
    uint32_t off_tree_parent;
    /// Byte offset of `seq_kv_len`: each sequence's KV length before this step.
    uint32_t off_seq_kv_len;
    /// Byte offset of `seq_page_table`: each sequence's offset into the KV store's page table.
    uint32_t off_seq_page_table;
    /// Byte offset of `seq_state_slot`: each sequence's recurrent-state slot.
    uint32_t off_seq_state_slot;
    /// Byte offset of `seq_sampler`: each sequence's sampler state.
    uint32_t off_seq_sampler;
    /// Byte offset of `moe_balance`: per MoE layer of the stage, B_copy(l) in bytes (RFC-0040
    /// §3.4; RFC-0023 §8).
    uint32_t off_moe_balance;
    /// Byte offset of `row_kv_write_from`: per row, the first position whose KV this step
    /// writes (RFC-0033 replay); `row_pos` in normal steps.
    uint32_t off_row_kv_write_from;
    /// The table-delta ring position `ApplyTableDeltas` applies up to (RFC-0040 §3.4).
    uint64_t delta_head;
    /// Device-visible address of this step's inbound `ActivationFrame` payload.
    uint64_t in_payload;
    /// Device-visible address of the registered send buffer for the outbound frame.
    uint64_t out_payload;
    /// Tapped rows this step (RFC-0040 §3.8); 0 in service steps.
    uint32_t rows_tap;
    /// Reference ids per tapped row, K.
    uint32_t tap_k;
    /// Byte offset of `tap_rows`: the tapped row indices.
    uint32_t off_tap_rows;
    /// The doorbell slot the reference ids' arrival raises.
    uint32_t tap_flag;
    /// Pinned buffer of reference token ids, `rows_tap` x `tap_k` (`TapMode::Gather`).
    uint64_t tap_ids;
    /// Result block: per tapped row, K log-probs, logsumexp, argmax and top-2 margin.
    uint64_t tap_out;
    /// Shared-memory region for the F32 logits of tapped rows (`TapMode::FullLocal`), or 0.
    uint64_t tap_full;
};

/// One sequence's random-number key and counter (RFC-0040 §3.5; RFC-0012 §8.2; ADR-0107).
/// The sampling parameters RFC-0012 §8 defines (temperature, top-k, top-p, penalties, the
/// grammar state slot) are not part of this version of the layout.
struct syn_sampler_state {
    /// The client's seed, or the seed ingress generated (ADR-0107).
    uint64_t seed;
    /// High half of the key's 128-bit sequence component: the choice index when the client
    /// supplied the seed, else the `SequenceId`.
    uint64_t stream_hi;
    /// Low half of the key's 128-bit sequence component.
    uint64_t stream_lo;
    /// Draws taken so far. `SampleSeeded` and `VerifyAccept` report the value after their last
    /// draw in the step's result block.
    uint64_t rng_counter;
};

/* SYNTROPY_STEP_DESC_H */
#endif
