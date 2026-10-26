// SPDX-License-Identifier: MIT OR Apache-2.0

//! Feed-forward blocks (RFC-0043 §3.2): `DenseFfn`, and `Moe` with softmax top-k routing and
//! no shared expert, the M0 subset.

use syntropy_core::{DType, LayerIx, TensorGroupId};
use syntropy_kernels::{Fusions, GatedAct, OpKind, RouteNorm};

use super::{Builder, Form, NodeSpec, RoutingKnown, add, push, rms_norm, row_width, with_matmul};
use crate::ir::{
    AttrValue, Dim, GroupSel, Node, NodeClass, OpAttrs, OpSet, Program, TensorRole, ValueId,
    ValueKind, WeightRef,
};

/// `DenseFfn`'s parameters (RFC-0043 §3.2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseFfnParams {
    /// The intermediate dim.
    pub inter: u32,
    /// The gated activation.
    pub act: GatedAct,
    /// The RMS norm epsilon.
    pub eps: f32,
}

impl DenseFfnParams {
    /// The ops both forms lower to (RFC-0040 §11.1).
    #[must_use]
    pub fn ops(&self) -> OpSet {
        with_matmul([
            OpKind::RmsNorm,
            OpKind::GateUpAct { act: self.act },
            OpKind::Add,
        ])
    }
}

/// Appends a `DenseFfn` reading the residual `x` (RFC-0043 §3.2):
/// `RmsNorm` → `GateUpAct` → `Gemv/Gemm` (down) → `Add`.
///
/// Every weight, the pre-FFN norm included, is in `group`, the layer's `DenseEveryToken` FFN
/// group. `GateUpAct` keeps its own kind in both forms: it is the fused gate and up product
/// with the activation, for any row count. Returns the layer's new residual.
pub fn dense_ffn(
    b: &mut Builder,
    p: &mut Program,
    params: &DenseFfnParams,
    form: Form,
    group: TensorGroupId,
    x: ValueId,
) -> ValueId {
    let n = rms_norm(b, p, group, params.eps, x);
    let h = b.rows(params.inter);
    push(
        p,
        NodeSpec {
            op: OpKind::GateUpAct { act: params.act },
            inputs: &[n],
            outputs: &[h],
            weights: &[(group, TensorRole::GateUp)],
            class: NodeClass::Dense,
            attrs: &[("inter", AttrValue::U32(params.inter))],
            fuse: Fusions::empty(),
        },
    );
    let width = row_width(b, x);
    let d = b.value(DType::F32, &width, ValueKind::Local);
    push(
        p,
        NodeSpec {
            op: form.matmul(),
            inputs: &[h],
            outputs: &[d],
            weights: &[(group, TensorRole::Down)],
            class: NodeClass::Dense,
            attrs: &[],
            fuse: Fusions::empty(),
        },
    );
    add(b, p, x, d)
}

/// `Moe`'s parameters for M0 (RFC-0043 §3.2): softmax routing over `experts`, top-`top_k`,
/// no shared expert, no router bias, no group-limited top-k and no latent projections.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoeParams {
    /// Routed experts, E.
    pub experts: u16,
    /// Experts per token.
    pub top_k: u16,
    /// Each expert's intermediate dim.
    pub expert_inter: u32,
    /// Whether the top-k weights are renormalised to sum to 1.
    pub renormalize: bool,
    /// The RMS norm epsilon.
    pub eps: f32,
}

impl MoeParams {
    /// The ops both forms lower to (RFC-0040 §11.1). The MoE ops are the same in both forms.
    #[must_use]
    pub fn ops(&self) -> OpSet {
        [
            OpKind::RmsNorm,
            OpKind::MoeRoute {
                norm: RouteNorm::Softmax,
            },
            OpKind::MoeGrouped,
            OpKind::MoeCombine,
            OpKind::Add,
        ]
        .into_iter()
        .collect()
    }
}

/// Appends a `Moe` FFN reading the post-attention residual `x` of layer `layer` (RFC-0043
/// §3.2): `RmsNorm` → `MoeRoute` → `MoeGrouped` over `GroupSel::Experts` → `MoeCombine` →
/// `Add`. M0 has no shared expert, so the row's "shared expert as `DenseFfn` nodes" step is
/// absent.
///
/// `router` is the layer's `Router` group, which also holds the pre-FFN norm (RFC-0041 §3.1).
/// The expert groups are chosen per step and never named. The router reads the
/// post-attention residual of its own layer, so routing is [`RoutingKnown::AtRouter`] (§3.4),
/// which the function returns with the layer's new residual. `form` changes nothing here:
/// the MoE ops serve every row count.
pub fn moe(
    b: &mut Builder,
    p: &mut Program,
    params: &MoeParams,
    form: Form,
    router: TensorGroupId,
    layer: LayerIx,
    x: ValueId,
) -> (ValueId, RoutingKnown) {
    let _ = form;
    let n = rms_norm(b, p, router, params.eps, x);
    let k = u32::from(params.top_k);
    let ids = b.value(DType::I32, &[Dim::Rows, Dim::Const(k)], ValueKind::Local);
    let weights = b.value(DType::F32, &[Dim::Rows, Dim::Const(k)], ValueKind::Local);
    let moe_attrs = [
        ("n_experts", AttrValue::U32(u32::from(params.experts))),
        ("top_k", AttrValue::U32(k)),
    ];
    let mut route_attrs: Vec<_> = moe_attrs.to_vec();
    route_attrs.push(("renormalize", AttrValue::Bool(params.renormalize)));
    push(
        p,
        NodeSpec {
            op: OpKind::MoeRoute {
                norm: RouteNorm::Softmax,
            },
            inputs: &[n],
            outputs: &[ids, weights],
            weights: &[(router, TensorRole::Router)],
            class: NodeClass::Router,
            attrs: &route_attrs,
            fuse: Fusions::empty(),
        },
    );

    let width = row_width(b, x);
    let mut expert_dims = vec![Dim::Rows, Dim::Const(k)];
    expert_dims.extend(width.iter().skip(1).copied());
    let per_expert = b.value(DType::F32, &expert_dims, ValueKind::Local);
    let mut grouped_attrs: Vec<_> = moe_attrs.to_vec();
    grouped_attrs.push(("inter", AttrValue::U32(params.expert_inter)));
    p.nodes.push(Node {
        op: OpKind::MoeGrouped,
        inputs: [n, ids].into_iter().collect(),
        outputs: [per_expert].into_iter().collect(),
        // An expert group holds the expert's gate, up and down matrices (RFC-0041 §3.1).
        weights: [TensorRole::GateUp, TensorRole::Down]
            .into_iter()
            .map(|role| WeightRef {
                group: GroupSel::Experts { layer },
                role,
            })
            .collect(),
        attrs: OpAttrs {
            entries: grouped_attrs.into_iter().collect(),
        },
        class: NodeClass::Expert,
        fuse: Default::default(),
    });

    let combined = b.value(DType::F32, &width, ValueKind::Local);
    push(
        p,
        NodeSpec {
            op: OpKind::MoeCombine,
            inputs: &[per_expert, weights],
            outputs: &[combined],
            weights: &[],
            class: NodeClass::Elementwise,
            attrs: &[
                ("top_k", AttrValue::U32(k)),
                ("n_shared", AttrValue::U32(0)),
            ],
            fuse: Fusions::empty(),
        },
    );
    (add(b, p, x, combined), RoutingKnown::AtRouter)
}
