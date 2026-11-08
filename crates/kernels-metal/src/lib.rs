// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `KernelProvider` for Apple GPUs, written in Metal Shading Language.
//!
//! Layer 2 in RFC-0001 §4.1, designed in RFC-0040. The provider itself is not implemented yet.
//! What exists is the MSL source of the MoE split rule, [`MOE_ASSIGN_MSL`], and of the attention
//! hand-off's layouts, [`ATTN_HANDOFF_MSL`].
//!
//! `unsafe` is allowed in this crate (RFC-0001 §4.2 rule 8). Every `unsafe` block and impl carries
//! a `// SAFETY:` comment that states its invariant (ADR-0001 item 4).

#![deny(clippy::undocumented_unsafe_blocks)]

/// The MoE split rule, `kernels::moe::assign`, in MSL (RFC-0040 §3.4; ADR-0404 item 4): the
/// doorbell's structs and codes, `syn_moe_assign_entry`, which decides one route entry, and
/// the kernel `syn_moe_assign`, one thread per route entry. `shaders/moe_assign.metal` is
/// generated from the one specification in `syntropy-kernels`, whose tests fail when it is
/// stale, and this crate's tests run it on the GPU against the Rust version.
pub const MOE_ASSIGN_MSL: &str = include_str!("../shaders/moe_assign.metal");

/// The attention hand-off's layouts in MSL (RFC-0022 §5.3): the per-layer `AttnDoorbell` and a
/// handshake flag on its own cache line, with `static_assert`s of the layout
/// `include/syntropy/attn_handoff.h` gives them in C. `shaders/attn_handoff.metal` is generated
/// from that header by the `syntropy-kernels` tests, which fail when it is stale, and this crate's
/// tests compile it with the Metal compiler.
pub const ATTN_HANDOFF_MSL: &str = include_str!("../shaders/attn_handoff.metal");
