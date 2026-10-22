// SPDX-License-Identifier: MIT OR Apache-2.0

//! Attention mixers (RFC-0043 §3.1). M0 has `Gqa`, which covers MHA and MQA.

use syntropy_core::TensorGroupId;
use syntropy_kernels::{AttnKind, Fusions, OpKind, RopeKind};

use super::{Builder, Form, NodeSpec, add, push, rms_norm, with_matmul};
use crate::ir::{AttrValue, NodeClass, OpSet, Program, TensorRole, ValueId};

/// How the Q, K and V projections are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QkvLayout {
    /// One fused projection (`TensorRole::Qkv`).
    Fused,
    /// Three projections (`Q`, `K`, `V`), with optional RMS norms on Q and K per head.
    Separate {
        /// Whether Q and K are RMS-normalised per head before RoPE.
        qk_norm: bool,
    },
}

/// The rotary embedding of a `Gqa` layer (RFC-0043 §3.1: "RoPE variants … are `OpAttrs`").
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeParams {
    /// The frequency rule.
    pub kind: RopeKind,
    /// The base, θ.
    pub theta: f32,
    /// Rotated dims per head: `head_dim` for full rotary, fewer for partial rotary, 0 for a
    /// layer without RoPE.
    pub dims: u16,
    /// The position scaling factor of `Linear`, `Yarn` and `Llama3`; 1 for `Plain`.
    pub scale: f32,
}

/// `Gqa`'s parameters (RFC-0043 §3.1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GqaParams {
    /// Query heads.
    pub heads: u16,
    /// KV heads; `heads` for MHA, 1 for MQA.
    pub kv_heads: u16,
    /// Head dim.
    pub head_dim: u16,
    /// How the projections are stored.
    pub qkv: QkvLayout,
    /// Whether the Q, K and V projections have a bias.
    pub bias: bool,
    /// The rotary embedding.
    pub rope: RopeParams,
    /// The attention scale; `None` for `1 / √head_dim`.
    pub scale: Option<f32>,
    /// The attention logit soft-cap, if any.
    pub softcap: Option<f32>,
    /// The RMS norm epsilon.
    pub eps: f32,
}

impl GqaParams {
    /// The ops both forms lower to (RFC-0040 §11.1).
    #[must_use]
    pub fn ops(&self) -> OpSet {
        with_matmul([
            OpKind::RmsNorm,
            OpKind::KvWrite(AttnKind::Gqa),
            OpKind::AttnDecode(AttnKind::Gqa),
            OpKind::AttnPrefill(AttnKind::Gqa),
            OpKind::Add,
        ])
    }

    fn q_width(&self) -> u32 {
        u32::from(self.heads) * u32::from(self.head_dim)
    }

    fn kv_width(&self) -> u32 {
        u32::from(self.kv_heads) * u32::from(self.head_dim)
    }
}

/// The attribute code of a RoPE frequency rule. `RopeKind` is `kernels`' type, and an
/// attribute holds numbers, so each kind has a fixed code; a kind this table does not know is
/// `u32::MAX`, which no consumer accepts. Whether the kind belongs in the attributes or in the
/// op key is issue #1610.
#[must_use]
pub fn rope_kind_code(kind: RopeKind) -> u32 {
    match kind {
        RopeKind::Plain => 0,
        RopeKind::Linear => 1,
        RopeKind::Yarn => 2,
        RopeKind::Llama3 => 3,
        _ => u32::MAX,
    }
}

/// Appends a `Gqa` mixer reading the layer's residual `x` (RFC-0043 §3.1):
/// `RmsNorm` → `Gemv/Gemm` (QKV) → `KvWrite(Gqa)` with RoPE folded in →
/// `AttnDecode/AttnPrefill(Gqa)` → `Gemv/Gemm` (O) → `Add`.
///
/// Every weight, the norms included, is in `group`, the layer's `DenseEveryToken` mixer group.
/// RoPE has no node of its own: `KvWrite` applies it to K as it stores K, and the attention op
/// to Q, both under the `FUSE_ROPE` hint with the RoPE attributes. Returns the post-attention
/// residual.
pub fn gqa(
    b: &mut Builder,
    p: &mut Program,
    params: &GqaParams,
    form: Form,
    group: TensorGroupId,
    x: ValueId,
) -> ValueId {
    let matmul = form.matmul();
    let bias = if params.bias {
        Fusions::FUSE_BIAS
    } else {
        Fusions::empty()
    };
    let rope = if params.rope.dims > 0 {
        Fusions::FUSE_ROPE
    } else {
        Fusions::empty()
    };
    let n = rms_norm(b, p, group, params.eps, x);

    let project = |b: &mut Builder, p: &mut Program, role, width| {
        let out = b.rows(width);
        push(
            p,
            NodeSpec {
                op: matmul,
                inputs: &[n],
                outputs: &[out],
                weights: &[(group, role)],
                class: NodeClass::Dense,
                attrs: &[],
                fuse: bias,
            },
        );
        out
    };
    // `q` feeds the attention op; `kv` feeds `KvWrite`.
    let (q, kv): (ValueId, Vec<ValueId>) = match params.qkv {
        QkvLayout::Fused => {
            let qkv = project(
                b,
                p,
                TensorRole::Qkv,
                params.q_width() + 2 * params.kv_width(),
            );
            (qkv, vec![qkv])
        }
        QkvLayout::Separate { qk_norm } => {
            let mut q = project(b, p, TensorRole::Q, params.q_width());
            let mut k = project(b, p, TensorRole::K, params.kv_width());
            let v = project(b, p, TensorRole::V, params.kv_width());
            if qk_norm {
                q = rms_norm(b, p, group, params.eps, q);
                k = rms_norm(b, p, group, params.eps, k);
            }
            (q, vec![k, v])
        }
    };

    let heads = [
        ("n_q_heads", AttrValue::U32(u32::from(params.heads))),
        ("n_kv_heads", AttrValue::U32(u32::from(params.kv_heads))),
        ("d_head", AttrValue::U32(u32::from(params.head_dim))),
    ];
    let rope_attrs = [
        (
            "rope_kind",
            AttrValue::U32(rope_kind_code(params.rope.kind)),
        ),
        ("rope_theta", AttrValue::F32(params.rope.theta)),
        ("rope_dims", AttrValue::U32(u32::from(params.rope.dims))),
        ("rope_scale", AttrValue::F32(params.rope.scale)),
    ];
    let kv_attrs: Vec<_> = heads[1..].iter().chain(&rope_attrs).copied().collect();
    push(
        p,
        NodeSpec {
            op: OpKind::KvWrite(AttnKind::Gqa),
            inputs: &kv,
            outputs: &[],
            weights: &[],
            class: NodeClass::State,
            attrs: &kv_attrs,
            fuse: rope,
        },
    );

    let mut attn_attrs: Vec<_> = heads.iter().chain(&rope_attrs).copied().collect();
    if let Some(scale) = params.scale {
        attn_attrs.push(("scale", AttrValue::F32(scale)));
    }
    if let Some(cap) = params.softcap {
        attn_attrs.push(("softcap", AttrValue::F32(cap)));
    }
    let attn = b.rows(params.q_width());
    push(
        p,
        NodeSpec {
            op: match form {
                Form::Decode => OpKind::AttnDecode(AttnKind::Gqa),
                Form::Prefill => OpKind::AttnPrefill(AttnKind::Gqa),
            },
            inputs: &[q],
            outputs: &[attn],
            weights: &[],
            class: NodeClass::State,
            attrs: &attn_attrs,
            fuse: rope,
        },
    );

    let width = super::row_width(b, x);
    let o = b.value(
        syntropy_core::DType::F32,
        &width,
        crate::ir::ValueKind::Local,
    );
    push(
        p,
        NodeSpec {
            op: matmul,
            inputs: &[attn],
            outputs: &[o],
            weights: &[(group, TensorRole::O)],
            class: NodeClass::Dense,
            attrs: &[],
            fuse: Fusions::empty(),
        },
    );
    add(b, p, x, o)
}
