// SPDX-License-Identifier: MIT OR Apache-2.0

//! Embeddings and the head (RFC-0043 §3.3): `Embed` and `LmHead`, the M0 subset.

use syntropy_core::{DType, TensorGroupId};
use syntropy_kernels::{Fusions, OpKind};

use super::{Builder, NodeSpec, push, rms_norm};
use crate::ir::{AttrValue, Dim, NodeClass, OpSet, Program, TensorRole, ValueId, ValueKind};

/// `Embed`'s parameters (RFC-0043 §3.3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmbedParams {
    /// The hidden size: the width of a table row.
    pub hidden: u32,
    /// The vocabulary size: the table's rows.
    pub vocab: u32,
    /// A scale applied to the gathered rows (√hidden in Gemma-class models), or `None`.
    pub scale: Option<f32>,
}

impl EmbedParams {
    /// The ops it lowers to (RFC-0040 §11.1).
    #[must_use]
    pub fn ops(&self) -> OpSet {
        let mut ops = OpSet::from([OpKind::RowGather]);
        if self.scale.is_some() {
            ops.insert(OpKind::Scale);
        }
        ops
    }
}

/// Appends an `Embed` to the prologue (RFC-0043 §3.3): `RowGather` of the token IDs `tokens`
/// from `table`, the `RowLookup` token-embedding group, then the optional `Scale`. Returns the
/// rows, which the family marks as the residual stream.
pub fn embed(
    b: &mut Builder,
    p: &mut Program,
    params: &EmbedParams,
    table: TensorGroupId,
    tokens: ValueId,
) -> ValueId {
    let rows = b.rows(params.hidden);
    push(
        p,
        NodeSpec {
            op: OpKind::RowGather,
            inputs: &[tokens],
            outputs: &[rows],
            weights: &[(table, TensorRole::Embed)],
            class: NodeClass::Lookup,
            attrs: &[
                ("table_rows", AttrValue::U32(params.vocab)),
                ("width", AttrValue::U32(params.hidden)),
            ],
            fuse: Fusions::empty(),
        },
    );
    let Some(scale) = params.scale else {
        return rows;
    };
    let scaled = b.rows(params.hidden);
    push(
        p,
        NodeSpec {
            op: OpKind::Scale,
            inputs: &[rows],
            outputs: &[scaled],
            weights: &[],
            class: NodeClass::Elementwise,
            attrs: &[("scale", AttrValue::F32(scale))],
            fuse: Fusions::empty(),
        },
    );
    scaled
}

/// `LmHead`'s parameters (RFC-0043 §3.3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LmHeadParams {
    /// The vocabulary size: the logits' width.
    pub vocab: u32,
    /// The final logit soft-cap, if any.
    pub softcap: Option<f32>,
    /// The final RMS norm's epsilon.
    pub eps: f32,
}

impl LmHeadParams {
    /// The ops it lowers to (RFC-0040 §11.1).
    #[must_use]
    pub fn ops(&self) -> OpSet {
        OpSet::from([OpKind::RmsNorm, OpKind::Logits])
    }
}

/// Appends an `LmHead` to the epilogue (RFC-0043 §3.3): the final norm → `Logits` → the
/// optional logit soft-cap.
///
/// Both read `head`, the `Head` group, which holds the final norm because the norm feeds the
/// head (RFC-0041 §3.1). No op in RFC-0040 §3.1 computes a soft-cap, so it is a `softcap`
/// attribute of `Logits`, applied in its epilogue. Returns the logits, `[Rows, vocab]` in
/// F32.
pub fn lm_head(
    b: &mut Builder,
    p: &mut Program,
    params: &LmHeadParams,
    head: TensorGroupId,
    x: ValueId,
) -> ValueId {
    let n = rms_norm(b, p, head, params.eps, x);
    let logits = b.value(
        DType::F32,
        &[Dim::Rows, Dim::Const(params.vocab)],
        ValueKind::Local,
    );
    let mut attrs = vec![("n_vocab", AttrValue::U32(params.vocab))];
    if let Some(cap) = params.softcap {
        attrs.push(("softcap", AttrValue::F32(cap)));
    }
    push(
        p,
        NodeSpec {
            op: OpKind::Logits,
            inputs: &[n],
            outputs: &[logits],
            weights: &[(head, TensorRole::Head)],
            class: NodeClass::Head,
            attrs: &attrs,
            fuse: Fusions::empty(),
        },
    );
    logits
}
