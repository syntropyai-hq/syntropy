// SPDX-License-Identifier: MIT OR Apache-2.0

//! Attention mixers (RFC-0043 §3.1). M0 has `Gqa`, which covers MHA and MQA.

use syntropy_core::{DType, TensorGroupId};
use syntropy_kernels::{AttnKind, Fusions, OpKind, RopeKind};

use super::{Builder, Form, NodeSpec, add, push, rms_norm, with_matmul};
use crate::ir::{AttrValue, Dim, NodeClass, OpSet, Program, TensorRole, ValueId, ValueKind};

/// How the Q, K and V projections are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QkvLayout {
    /// One fused projection (`TensorRole::Qkv`).
    Fused,
    /// Three projections (`Q`, `K`, `V`), with optional RMS norms on Q and K per head. Each
    /// projection's output is typed `[Rows, heads, head_dim]`, so a norm over its last dim is
    /// per head.
    Separate {
        /// Whether Q and K are RMS-normalised per head before RoPE.
        qk_norm: bool,
    },
}

/// The rotary embedding of a `Gqa` layer (RFC-0043 §3.1: "RoPE variants … are `OpAttrs`",
/// computed once into tables at `prepare`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeParams {
    /// The frequency rule and its inputs.
    pub scaling: RopeScaling,
    /// The base, θ.
    pub theta: f32,
    /// Rotated dims per head: `head_dim` for full rotary, fewer for partial rotary, 0 for a
    /// layer without RoPE.
    pub dims: u16,
}

/// A RoPE frequency rule with every input its table needs.
///
/// Only the rules whose inputs the plan defines are here: unscaled frequencies (the base and
/// the rotated dims) and linear position scaling (one factor). RFC-0043 §3.1 and §5 name
/// YaRN and Llama-3-style scaling, and RFC-0022 §7.1 hashes "scaling type and factors,
/// original context length", but no RFC lists their parameters. They come with the family
/// that needs them (COMP-E16-T03) once the plan names them, so this block never emits a rule
/// whose inputs are missing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RopeScaling {
    /// Unscaled frequencies.
    Plain,
    /// Positions divided by `factor`.
    Linear {
        /// The position scaling factor.
        factor: f32,
    },
}

impl RopeScaling {
    /// The `kernels` frequency rule this is.
    #[must_use]
    pub fn kind(&self) -> RopeKind {
        match self {
            Self::Plain => RopeKind::Plain,
            Self::Linear { .. } => RopeKind::Linear,
        }
    }
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

/// The `rope_kind` attribute code of a RoPE frequency rule: an attribute holds numbers, so
/// each rule has a fixed code. The match is exhaustive, so a rule added to [`RopeScaling`]
/// needs a code here before it compiles. Folding RoPE into `KvWrite` and attention puts the
/// rule in attributes instead of an `OpKey`, which is issue #1610's question.
#[must_use]
pub fn rope_kind_code(scaling: &RopeScaling) -> u32 {
    match scaling {
        RopeScaling::Plain => 0,
        RopeScaling::Linear { .. } => 1,
    }
}

/// Appends a `Gqa` mixer reading the layer's residual `x` (RFC-0043 §3.1):
/// `RmsNorm` → `Gemv/Gemm` (QKV) → `KvWrite(Gqa)` with RoPE folded in →
/// `AttnDecode/AttnPrefill(Gqa)` → `Gemv/Gemm` (O) → `Add`.
///
/// Every weight, the norms included, is in `group`, the layer's `DenseEveryToken` mixer group.
/// RoPE has no node of its own: `KvWrite` applies it to K as it stores K, as RFC-0040 §3.1
/// allows, and the attention op applies it to Q, which no RFC states yet. Both carry the
/// `FUSE_ROPE` hint with the RoPE attributes. With separate projections, Q, K and V are typed
/// `[Rows, heads, head_dim]`, and the Q/K norms normalise over that last dim, so per head.
/// Returns the post-attention residual.
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

    let project = |b: &mut Builder, p: &mut Program, role, dims: &[Dim]| {
        let out = b.value(DType::F32, dims, ValueKind::Local);
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
            let width = params.q_width() + 2 * params.kv_width();
            let qkv = project(b, p, TensorRole::Qkv, &[Dim::Rows, Dim::Const(width)]);
            (qkv, vec![qkv])
        }
        QkvLayout::Separate { qk_norm } => {
            let d_head = Dim::Const(u32::from(params.head_dim));
            let q_dims = [Dim::Rows, Dim::Const(u32::from(params.heads)), d_head];
            let kv_dims = [Dim::Rows, Dim::Const(u32::from(params.kv_heads)), d_head];
            let mut q = project(b, p, TensorRole::Q, &q_dims);
            let mut k = project(b, p, TensorRole::K, &kv_dims);
            let v = project(b, p, TensorRole::V, &kv_dims);
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
    let mut rope_attrs = vec![
        (
            "rope_kind",
            AttrValue::U32(rope_kind_code(&params.rope.scaling)),
        ),
        ("rope_theta", AttrValue::F32(params.rope.theta)),
        ("rope_dims", AttrValue::U32(u32::from(params.rope.dims))),
    ];
    match params.rope.scaling {
        RopeScaling::Plain => {}
        RopeScaling::Linear { factor } => rope_attrs.push(("rope_factor", AttrValue::F32(factor))),
    }
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
    let o = b.value(DType::F32, &width, ValueKind::Local);
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
