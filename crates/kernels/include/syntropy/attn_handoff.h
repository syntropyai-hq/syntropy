/* SPDX-License-Identifier: MIT OR Apache-2.0 */

/*
 * The `CpuSplit` attention hand-off: the per-layer `AttnDoorbell`, the flags of RFC-0040 §4.2's
 * constant-valued consumer-reset handshake, and the F32 (o, m, l) partial blocks the CPU pools
 * write and `AttnMerge` reads (RFC-0022 §5.3; RFC-0040 §4.2).
 *
 * This header is the one definition of these layouts. `crates/kernels/src/attn_handoff/mirror.rs`
 * is the Rust mirror generated from it, and `crates/kernels-metal/shaders/attn_handoff.metal` the
 * Metal version, with `static_assert`s of the same layout. The `syntropy-kernels` tests fail when
 * either differs (`SYNTROPY_BLESS_GENERATED=1` rewrites them), compile this header as C11 and
 * C++11 when a compiler is present (always in CI), and check the macros below against the Rust
 * layout; the `syntropy-kernels-metal` tests compile the Metal version. The generator reads the
 * subset of C that `step_desc.h` describes; it skips the function-like macros, one line each.
 *
 * One hand-off layer, for the layer's attention pass, is in order:
 * - `syn_attn_doorbell`, which the GPU writes after projection and RoPE;
 * - `selected[rows_max][selected_max]`, `uint32_t` host positions per query row, for sparse
 *   kinds only (`selected_max` is 0 for dense kinds);
 * - the query, `float[rows_max][n_q_heads][d_q]`;
 * - per NUMA domain d: the flag `Ra(l, d)`, which the GPU raises for domain d's pool, the flag
 *   `Qa(l, d)`, which that pool raises when its partials are written, and its partials,
 *   `float[rows_max][n_q_heads][d_v + 2]`: per row and query head, `o[d_v]`, then `m`, then `l`.
 * Each part starts on a `SYN_ATTN_LINE` boundary, so, in a layer whose base is 64-byte aligned
 * (the arena's allocation), the GPU's and each pool's writes never share a cache line. The host
 * `AttnHandoff` sub-arena holds two layers, for layers of even and odd index (double-buffered
 * across layers). Every flag is 1 when raised and 0 when its consumer has
 * reset it after its `EQ 1` wait; a side raises its flag with a release store after its last
 * write, and the waiter reads only after it sees 1 (RFC-0012 §3.2). `step` is a payload check,
 * not a flag. A side with no pages in
 * the pass writes `o = 0`, `m = -inf`, `l = 0` and still raises its flag. All values are
 * little-endian.
 */

#ifndef SYNTROPY_ATTN_HANDOFF_H
#define SYNTROPY_ATTN_HANDOFF_H

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
#define SYN_ALIGNAS(n) alignas(n)
#else
#define SYN_ALIGNAS(n) _Alignas(n)
#endif

/// The version of the layouts below. A change to either struct, or to the order or alignment of
/// a hand-off layer's parts, bumps it.
#define SYN_ATTN_HANDOFF_ABI_VERSION 1

/// The alignment of every part of a hand-off layer, and of the flags: one cache line.
#define SYN_ATTN_LINE 64

/// `x` rounded up to a whole number of `SYN_ATTN_LINE`s.
#define SYN_ATTN_ROUND(x) ((((size_t)(x)) + (size_t)SYN_ATTN_LINE - 1u) / (size_t)SYN_ATTN_LINE * (size_t)SYN_ATTN_LINE)
/// Floats in one partial block, for one row and one query head: `o[d_v]`, `m` and `l`.
#define SYN_ATTN_PARTIAL_FLOATS(d_v) ((size_t)(d_v) + 2u)
/// Byte offset of `selected` in a hand-off layer.
#define SYN_ATTN_SELECTED_OFFSET() SYN_ATTN_ROUND(sizeof(struct syn_attn_doorbell))
/// Byte offset of `selected[row][k]`; the row stride is the layer's `selected_max`, `sel`.
#define SYN_ATTN_SELECTED_AT(sel, row, k) (SYN_ATTN_SELECTED_OFFSET() + ((size_t)(row) * (size_t)(sel) + (size_t)(k)) * sizeof(uint32_t))
/// Byte offset of the query in a hand-off layer.
#define SYN_ATTN_QUERY_OFFSET(rows, sel) (SYN_ATTN_SELECTED_OFFSET() + SYN_ATTN_ROUND((size_t)(rows) * (size_t)(sel) * sizeof(uint32_t)))
/// Byte offset of `query[row][head][0]`: rows, then heads, then `d_q` floats.
#define SYN_ATTN_QUERY_AT(rows, sel, heads, d_q, row, head) (SYN_ATTN_QUERY_OFFSET(rows, sel) + ((size_t)(row) * (size_t)(heads) + (size_t)(head)) * (size_t)(d_q) * sizeof(float))
/// Byte offset of NUMA domain 0's part in a hand-off layer.
#define SYN_ATTN_DOMAINS_OFFSET(rows, sel, heads, d_q) (SYN_ATTN_QUERY_OFFSET(rows, sel) + SYN_ATTN_ROUND((size_t)(rows) * (size_t)(heads) * (size_t)(d_q) * sizeof(float)))
/// Size in bytes of one NUMA domain's part: its two flags and its partials.
#define SYN_ATTN_DOMAIN_SIZE(rows, heads, d_v) (2u * sizeof(struct syn_attn_flag) + SYN_ATTN_ROUND((size_t)(rows) * (size_t)(heads) * SYN_ATTN_PARTIAL_FLOATS(d_v) * sizeof(float)))
/// Byte offset of domain `d`'s `Ra` flag; its `Qa` flag follows it, and its partials follow that.
#define SYN_ATTN_DOMAIN_OFFSET(rows, sel, heads, d_q, d_v, d) (SYN_ATTN_DOMAINS_OFFSET(rows, sel, heads, d_q) + (size_t)(d) * SYN_ATTN_DOMAIN_SIZE(rows, heads, d_v))
/// Byte offset of domain `d`'s partial block for `row` and `head`: rows, then heads, then
/// `o[d_v]`, `m` and `l`.
#define SYN_ATTN_PARTIAL_AT(rows, sel, heads, d_q, d_v, d, row, head) (SYN_ATTN_DOMAIN_OFFSET(rows, sel, heads, d_q, d_v, d) + 2u * sizeof(struct syn_attn_flag) + ((size_t)(row) * (size_t)(heads) + (size_t)(head)) * SYN_ATTN_PARTIAL_FLOATS(d_v) * sizeof(float))
/// Size in bytes of one hand-off layer with `domains` NUMA domains.
#define SYN_ATTN_LAYER_SIZE(rows, sel, heads, d_q, d_v, domains) SYN_ATTN_DOMAIN_OFFSET(rows, sel, heads, d_q, d_v, domains)

/// The per-layer attention doorbell (RFC-0022 §5.3 `AttnDoorbell`). It sits next to the MoE
/// `syn_doorbell_header`, not inside it.
struct syn_attn_doorbell {
    /// The low 32 bits of the step number, `syn_step_desc.step`: a payload check the pool
    /// asserts, not a flag.
    uint32_t step;
    /// Query rows in this pass, at most the layer's `rows_max`.
    uint16_t rows;
    /// Zero.
    uint16_t _pad;
    /// Host pages the CPU side attends over in this pass, over all NUMA domains; each domain's
    /// pool attends the ones in its domain. 0 when there are none.
    uint32_t host_pages;
    /// Valid entries per row of `selected`; 0 for dense kinds.
    uint32_t selected;
};

/// One flag of the handshake, alone on its cache line: 1 when raised, 0 once its consumer has
/// reset it.
struct syn_attn_flag {
    /// The flag's value.
    SYN_ALIGNAS(64) uint32_t value;
};

/* SYNTROPY_ATTN_HANDOFF_H */
#endif
