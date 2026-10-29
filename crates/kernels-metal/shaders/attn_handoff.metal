// SPDX-License-Identifier: MIT OR Apache-2.0

// @generated from `crates/kernels/include/syntropy/attn_handoff.h` by the
// `syntropy-kernels` tests. Do not edit: change the header and run
// `SYNTROPY_BLESS_GENERATED=1 cargo test -p syntropy-kernels attn_handoff::`.
//
// The attention hand-off's layouts in Metal Shading Language (RFC-0022 §5.3), for the
// Metal provider. The `syntropy-kernels-metal` tests compile it with the Metal compiler,
// which checks the `static_assert`s below.

#include <metal_stdlib>

using namespace metal;

/// The version of the layouts below. A change to either struct, or to the order or alignment of
/// a hand-off layer's parts, bumps it.
constant uint SYN_ATTN_HANDOFF_ABI_VERSION = 1u;

/// The alignment of every part of a hand-off layer, and of the flags: one cache line.
constant uint SYN_ATTN_LINE = 64u;

/// The per-layer attention doorbell (RFC-0022 §5.3 `AttnDoorbell`). It sits next to the MoE
/// `syn_doorbell_header`, not inside it.
struct syn_attn_doorbell {
    /// The low 32 bits of the step number, `syn_step_desc.step`: a payload check the pool
    /// asserts, not a flag.
    uint step;
    /// Query rows in this pass, at most the layer's `rows_max`.
    ushort rows;
    /// Zero.
    ushort _pad;
    /// Host pages the CPU side attends over in this pass, over all NUMA domains; each domain's
    /// pool attends the ones in its domain. 0 when there are none.
    uint host_pages;
    /// Valid entries per row of `selected`; 0 for dense kinds.
    uint selected;
};

/// One flag of the handshake, alone on its cache line: 1 when raised, 0 once its consumer has
/// reset it.
struct syn_attn_flag {
    /// The flag's value.
    alignas(64) uint value;
};

// The layout `attn_handoff.h` gives each struct, checked by the Metal compiler.
static_assert(sizeof(syn_attn_doorbell) == 16, "syn_attn_doorbell size");
static_assert(alignof(syn_attn_doorbell) == 4, "syn_attn_doorbell align");
static_assert(__builtin_offsetof(syn_attn_doorbell, step) == 0, "syn_attn_doorbell.step");
static_assert(__builtin_offsetof(syn_attn_doorbell, rows) == 4, "syn_attn_doorbell.rows");
static_assert(__builtin_offsetof(syn_attn_doorbell, _pad) == 6, "syn_attn_doorbell._pad");
static_assert(__builtin_offsetof(syn_attn_doorbell, host_pages) == 8, "syn_attn_doorbell.host_pages");
static_assert(__builtin_offsetof(syn_attn_doorbell, selected) == 12, "syn_attn_doorbell.selected");
static_assert(sizeof(syn_attn_flag) == 64, "syn_attn_flag size");
static_assert(alignof(syn_attn_flag) == 64, "syn_attn_flag align");
static_assert(__builtin_offsetof(syn_attn_flag, value) == 0, "syn_attn_flag.value");
