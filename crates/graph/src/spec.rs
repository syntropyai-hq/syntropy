// SPDX-License-Identifier: MIT OR Apache-2.0

//! `FamilySpec`: what a family's synthetic fixtures must contain (RFC-0043 §9.1; RFC-0060
//! §12.1).
//!
//! The fixture generator (QUAL-E7-T01) builds `nano` and `micro` checkpoints from a spec, and
//! each fixture is structure-true: it holds at least one of every layer kind, attention form,
//! MoE feature and MTP module its family has (RFC-0060 §12.1). Some features are fixed for a
//! whole checkpoint, such as QK-norm or tied embeddings, so a spec lists variants, and the
//! generator builds one fixture per variant. A variant names the architecture strings the
//! generator writes, so the family resolves its own fixtures.
//!
//! RFC-0043 §9.1 lists the spec's contents: layer pattern, attention kinds, MoE layout, MTP
//! modules, recurrent blocks and side data. The M0 families (`llama`, `qwen_moe`) have no
//! recurrent block and no side data, so those fields come with the first family that does.

use syntropy_kernels::RopeKind;

/// A family's fixture spec.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FamilySpec {
    /// The family tag.
    pub family: &'static str,
    /// The checkpoints the generator builds, one per variant.
    pub variants: &'static [VariantSpec],
}

/// One fixture checkpoint's structure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VariantSpec {
    /// The variant's name, unique within the family.
    pub name: &'static str,
    /// The GGUF `general.architecture` the generator writes.
    pub gguf_architecture: &'static str,
    /// The `config.json` `architectures` entry the generator writes.
    pub architecture: &'static str,
    /// The `config.json` `model_type` the generator writes.
    pub model_type: &'static str,
    /// The layer pattern, repeated to the fixture's depth.
    pub layers: &'static [LayerKind],
    /// The attention form of every attention layer.
    pub attention: AttentionSpec,
    /// The MoE layout of the MoE layers, if the pattern has any.
    pub moe: Option<MoeSpec>,
    /// Whether the head reads the token table.
    pub tied_embeddings: bool,
    /// MTP modules after the last layer.
    pub mtp_modules: u8,
}

/// One layer's blocks.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LayerKind {
    /// `Gqa` then `DenseFfn`.
    GqaDense,
    /// `Gqa` then `Moe`.
    GqaMoe,
}

/// The attention features a variant has.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttentionSpec {
    /// Q and K RMS-normalised per head.
    pub qk_norm: bool,
    /// A bias on the Q, K and V projections.
    pub qkv_bias: bool,
    /// The RoPE frequency rule; the generator picks its parameters.
    pub rope: RopeKind,
}

/// The MoE layout of a variant's MoE layers (RFC-0043 §9.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MoeSpec {
    /// Routed experts in the real model; a fixture's size class may hold fewer.
    pub experts: u16,
    /// Experts selected per token.
    pub top_k: u16,
    /// Shared experts.
    pub shared: u16,
    /// Whether the experts work in a latent dimension.
    pub latent: bool,
    /// Whether the router has a per-expert bias.
    pub router_bias: bool,
}
