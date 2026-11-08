// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `CpuSplit` attention hand-off between the GPU and the CPU pools: the per-layer
//! `AttnDoorbell`, the handshake's flags and the F32 (o, m, l) partial blocks, one layout for the
//! CUDA, Metal and CPU sides (RFC-0022 §5.3; RFC-0040 §4.2). COMP's split-attention kernels,
//! PLC's CPU pool and TIER's `CpuSplit` path build against it.
//!
//! - [`mirror`] holds the `#[repr(C)]` mirrors of the doorbell and a flag, [`AttnDoorbell`] and
//!   [`AttnFlag`], generated from `include/syntropy/attn_handoff.h`, their one definition. The
//!   Metal version, `crates/kernels-metal/shaders/attn_handoff.metal`, is printed from the same
//!   header by this module's tests, with the layout as `static_assert`s.
//! - [`layer`] lays a hand-off layer out for one shape ([`HandoffLayout`]) and writes and reads
//!   the doorbell, the flags and the partials in its bytes, including the empty side's encoding.
//!
//! `kernels` forbids `unsafe`, so a hand-off layer is a byte slice here; the provider crates map
//! it onto `GPU_MAPPED` host memory.

pub mod layer;
// Generated: the tests compare it with the generator's output, so rustfmt leaves it alone.
#[rustfmt::skip]
pub mod mirror;
#[cfg(test)]
mod msl;
#[cfg(test)]
mod tests;

pub use layer::{HandoffError, HandoffLayout, HandoffShape, Partial};
pub use mirror::{AttnDoorbell, AttnFlag};
