// SPDX-License-Identifier: MIT OR Apache-2.0

//! The block library of RFC-0043 §3: each block appends IR nodes to a program from a parameter
//! struct that the family's hyperparameter parser fills. Blocks are the only place an
//! attention, scan or FFN variant is described; families compose them.
//!
//! The M0 blocks are here: [`gqa`] (§3.1), [`dense_ffn`] and [`moe`] with softmax top-k
//! routing and no shared expert (§3.2), and [`embed`] and [`lm_head`] (§3.3). Each lowers to
//! exactly the ops of its table row, and each has an `ops` function that lists them, so a
//! family's `required_ops` is the union over its blocks (RFC-0040 §11.1).
//!
//! **Forms.** A row such as `Gemv/Gemm` or `AttnDecode/AttnPrefill` names one op for few rows
//! and another for many. A block builds one [`Form`] at a time. Which form a shape class uses
//! (decode, verify windows, batch buckets, prefill chunks) is the segment builder's choice
//! (§6.2, COMP-E15-T07), and a family's required op set covers both forms (RFC-0040 §11.1).
//!
//! **Groups.** A block names the census groups its weights live in, which the caller passes:
//! the census lives in `manifest`, which does not group tensors yet. Norms read the group of
//! the block they feed (RFC-0041 §3.1).

use smallvec::SmallVec;
use syntropy_core::{DType, TensorGroupId};
use syntropy_kernels::{Fusions, OpKind};

use crate::ir::{
    AttrValue, Dim, FuseHints, GroupSel, Node, NodeClass, OpAttrs, OpSet, Program, TensorRole,
    Value, ValueId, ValueKind, ValueType, WeightRef,
};

mod attention;
mod embed;
mod ffn;

pub use attention::{GqaParams, QkvLayout, RopeParams, gqa};
pub use embed::{EmbedParams, LmHeadParams, embed, lm_head};
pub use ffn::{DenseFfnParams, MoeParams, dense_ffn, moe};

/// Which of a row's two ops a block emits where the row names both (RFC-0043 §3.1, §3.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Form {
    /// Few rows: `Gemv` and `AttnDecode`.
    Decode,
    /// Many rows: `Gemm` and `AttnPrefill`.
    Prefill,
}

impl Form {
    /// Both forms, decode first.
    pub const ALL: [Form; 2] = [Form::Decode, Form::Prefill];

    /// The matrix product of this form.
    #[must_use]
    pub fn matmul(self) -> OpKind {
        match self {
            Self::Decode => OpKind::Gemv,
            Self::Prefill => OpKind::Gemm,
        }
    }
}

/// When a MoE layer's routing is known, which RFC-0023 §9 uses for prefetch (RFC-0043 §3.4;
/// RFC-0023 §2 `RoutingKnown`). Owner: RFC-0023.
///
/// The M0 `Moe` block declares `AtRouter`. `PreAttention`, `PreGate { lookahead }` and
/// `TokenHash` come with the blocks that need them (COMP-E15-T05).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RoutingKnown {
    /// Known when the layer's router runs: it reads the post-attention residual of its own
    /// layer.
    AtRouter,
}

/// The value table of the graph being built, which blocks add values and nodes to.
///
/// Values get [`ValueKind::Local`]; the family marks the values that cross layers (residual
/// streams, side data) with [`Builder::set_kind`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Builder {
    /// Every value so far, indexed by [`ValueId`]: `ModelGraph::values` once the graph is
    /// built.
    pub values: Vec<Value>,
}

impl Builder {
    /// An empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A new value of type `dtype` and `dims`.
    pub fn value(&mut self, dtype: DType, dims: &[Dim], kind: ValueKind) -> ValueId {
        let id = ValueId(u32::try_from(self.values.len()).unwrap_or(u32::MAX));
        self.values.push(Value {
            ty: ValueType {
                dtype,
                dims: SmallVec::from_slice(dims),
            },
            kind,
        });
        id
    }

    /// A new local F32 value of `[Rows, width]`: the activations every M0 block passes on.
    pub fn rows(&mut self, width: u32) -> ValueId {
        self.value(
            DType::F32,
            &[Dim::Rows, Dim::Const(width)],
            ValueKind::Local,
        )
    }

    /// Changes `v`'s kind, if `v` is a value of this builder.
    pub fn set_kind(&mut self, v: ValueId, kind: ValueKind) {
        if let Some(value) = usize::try_from(v.0)
            .ok()
            .and_then(|i| self.values.get_mut(i))
        {
            value.kind = kind;
        }
    }
}

/// What a block passes to [`push`] for one node.
struct NodeSpec<'a> {
    op: OpKind,
    inputs: &'a [ValueId],
    outputs: &'a [ValueId],
    weights: &'a [(TensorGroupId, TensorRole)],
    class: NodeClass,
    attrs: &'a [(&'static str, AttrValue)],
    fuse: Fusions,
}

/// Appends a node to `p`.
fn push(p: &mut Program, spec: NodeSpec<'_>) {
    p.nodes.push(Node {
        op: spec.op,
        inputs: SmallVec::from_slice(spec.inputs),
        outputs: SmallVec::from_slice(spec.outputs),
        weights: spec
            .weights
            .iter()
            .map(|&(group, role)| WeightRef {
                group: GroupSel::Fixed(group),
                role,
            })
            .collect(),
        attrs: OpAttrs {
            entries: SmallVec::from_slice(spec.attrs),
        },
        class: spec.class,
        fuse: FuseHints { allowed: spec.fuse },
    });
}

/// An `RmsNorm` reading the norm weight of `group`.
fn rms_norm(
    b: &mut Builder,
    p: &mut Program,
    group: TensorGroupId,
    eps: f32,
    x: ValueId,
) -> ValueId {
    let width = row_width(b, x);
    let out = b.value(DType::F32, &width, ValueKind::Local);
    push(
        p,
        NodeSpec {
            op: OpKind::RmsNorm,
            inputs: &[x],
            outputs: &[out],
            weights: &[(group, TensorRole::Norm)],
            class: NodeClass::Elementwise,
            attrs: &[("eps", AttrValue::F32(eps))],
            fuse: Fusions::empty(),
        },
    );
    out
}

/// A residual `Add` of `x` and `y`.
fn add(b: &mut Builder, p: &mut Program, x: ValueId, y: ValueId) -> ValueId {
    let width = row_width(b, x);
    let out = b.value(DType::F32, &width, ValueKind::Local);
    push(
        p,
        NodeSpec {
            op: OpKind::Add,
            inputs: &[x, y],
            outputs: &[out],
            weights: &[],
            class: NodeClass::Elementwise,
            attrs: &[],
            fuse: Fusions::empty(),
        },
    );
    out
}

/// The dims of `x`, or `[Rows]` if `x` is not a value of this builder.
fn row_width(b: &Builder, x: ValueId) -> SmallVec<[Dim; 4]> {
    usize::try_from(x.0)
        .ok()
        .and_then(|i| b.values.get(i))
        .map_or_else(|| SmallVec::from_slice(&[Dim::Rows]), |v| v.ty.dims.clone())
}

/// The ops of both forms of a block whose rows are `common` plus a matrix product.
fn with_matmul(common: impl IntoIterator<Item = OpKind>) -> OpSet {
    let mut ops: OpSet = common.into_iter().collect();
    ops.extend(Form::ALL.map(Form::matmul));
    ops
}

#[cfg(test)]
mod tests;
