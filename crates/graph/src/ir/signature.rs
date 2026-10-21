// SPDX-License-Identifier: MIT OR Apache-2.0

//! The value signature of each [`OpKind`]: how many values a node of that kind reads and
//! defines, and the dtype class of each. Rule 2 of RFC-0043 §2.2 checks every node against it.
//!
//! RFC-0043 §2.2 checks node types against "its `OpKind`'s signature in the registry"
//! (RFC-0040 §3.1). The registry in `kernels` holds each kind's static dims
//! ([`OpKind::dims`]) but no value signature yet, so this module holds one, built from each
//! op's description in RFC-0040 §3.1. Weights are not values: a node reads them through
//! [`Node::weights`](super::Node::weights). The table moves next to the reference semantics
//! (`kernels::reference`) when that task lands and may be refined there.

use syntropy_core::DType;
use syntropy_kernels::OpKind;

/// The dtype class a value at one position of a signature must have.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Port {
    /// Activations: F32, F16, BF16 or F64.
    Float,
    /// Token IDs, expert IDs or other indices: I16, I32 or I64.
    Ids,
    /// Any dtype.
    Any,
}

impl Port {
    /// Whether a value of `dtype` fits this port.
    #[must_use]
    pub fn admits(self, dtype: DType) -> bool {
        match self {
            Self::Float => matches!(dtype, DType::F32 | DType::F16 | DType::BF16 | DType::F64),
            Self::Ids => matches!(dtype, DType::I16 | DType::I32 | DType::I64),
            Self::Any => true,
        }
    }
}

/// One side of a [`Signature`]: the inputs or the outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Ports {
    /// The port of each position. With `variadic`, the last port repeats.
    pub ports: &'static [Port],
    /// The fewest values: positions from `min` on are optional.
    pub min: u8,
    /// Whether any number of values beyond `ports.len()` may follow, each of the last port.
    pub variadic: bool,
}

impl Ports {
    /// Exactly these ports.
    const fn exactly(ports: &'static [Port]) -> Self {
        Self {
            ports,
            min: ports.len() as u8,
            variadic: false,
        }
    }

    /// These ports, of which only the first `min` are required.
    const fn at_least(ports: &'static [Port], min: u8) -> Self {
        Self {
            ports,
            min,
            variadic: false,
        }
    }

    /// At least `min` values, each of `port`.
    const fn many(port: &'static [Port], min: u8) -> Self {
        Self {
            ports: port,
            min,
            variadic: true,
        }
    }

    /// Whether `n` values fit.
    #[must_use]
    pub fn admits_count(&self, n: usize) -> bool {
        n >= usize::from(self.min) && (self.variadic || n <= self.ports.len())
    }

    /// The port of position `i`, if there is one.
    #[must_use]
    pub fn port(&self, i: usize) -> Option<Port> {
        match self.ports.get(i) {
            Some(&port) => Some(port),
            None if self.variadic => self.ports.last().copied(),
            None => None,
        }
    }
}

/// The value signature of one [`OpKind`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Signature {
    /// The values it reads.
    pub inputs: Ports,
    /// The values it defines.
    pub outputs: Ports,
    /// Whether every input and output has the dims of the first input: elementwise ops, norms,
    /// RoPE, conversions and hyper-connection mixing.
    pub same_dims: bool,
}

const F: Port = Port::Float;
const IDS: Port = Port::Ids;
const ANY: Port = Port::Any;

const fn sig(inputs: Ports, outputs: Ports) -> Signature {
    Signature {
        inputs,
        outputs,
        same_dims: false,
    }
}

const fn same(inputs: Ports, outputs: Ports) -> Signature {
    Signature {
        inputs,
        outputs,
        same_dims: true,
    }
}

/// The value signature of `op`, or `None` for a kind this table does not know yet. Rule 2
/// rejects a node whose kind has none, so a kind added to [`OpKind`] fails closed here until
/// it has a row.
///
/// Inputs are activations and indices only; weights come through `Node::weights`. KV and
/// recurrent state are not values (RFC-0043 §2.3): `KvWrite` defines nothing, and attention
/// and scans read the state through the step descriptor. The optional `Ids` input of the
/// attention kinds is the selection of a preceding `AttnIndex`.
#[must_use]
pub fn signature(op: &OpKind) -> Option<Signature> {
    let one_f = Ports::exactly(&[F]);
    Some(match op {
        // ---- Weight-bound linear algebra ----
        OpKind::Gemv | OpKind::Gemm | OpKind::GateUpAct { .. } => sig(one_f, one_f),
        // Router logits to top-k expert IDs and their weights.
        OpKind::MoeRoute { .. } => sig(one_f, Ports::exactly(&[IDS, F])),
        // Rows and the selected expert IDs to per-expert outputs.
        OpKind::MoeGrouped => sig(Ports::exactly(&[F, IDS]), one_f),
        // Per-expert outputs and their weights, then any shared-expert outputs.
        OpKind::MoeCombine => sig(Ports::many(&[F], 2), one_f),
        // ---- Attention and sequence mixers ----
        OpKind::AttnDecode(_) | OpKind::AttnPrefill(_) => sig(Ports::at_least(&[F, IDS], 1), one_f),
        // `(o, m, l)` unnormalised (RFC-0040 §3.6).
        OpKind::AttnPartial(_) => sig(Ports::at_least(&[F, IDS], 1), Ports::exactly(&[F, F, F])),
        // The partials' `(o, m, l)` triples.
        OpKind::AttnMerge => sig(Ports::many(&[F], 3), one_f),
        OpKind::AttnIndex(_) => sig(one_f, Ports::exactly(&[IDS])),
        // K and V, or the latent and RoPE key, into the KV pages.
        OpKind::KvWrite(_) => sig(Ports::at_least(&[F, F], 1), Ports::exactly(&[])),
        // Projected inputs: q, k, v, gates, step sizes.
        OpKind::ScanDecode(_) | OpKind::ScanPrefill(_) => sig(Ports::many(&[F], 1), one_f),
        OpKind::CausalConv1d => same(one_f, one_f),
        // ---- Normalisation and elementwise ----
        OpKind::RmsNorm | OpKind::LayerNorm | OpKind::Rope(_) | OpKind::Act(_) | OpKind::Scale => {
            same(one_f, one_f)
        }
        OpKind::Add | OpKind::Mul => same(Ports::exactly(&[F, F]), one_f),
        OpKind::HyperMix => same(Ports::many(&[F], 1), Ports::many(&[F], 1)),
        // ---- Tables, head and sampling ----
        OpKind::RowGather => sig(Ports::exactly(&[IDS]), one_f),
        // Embedding rows, then the encoder's rows.
        OpKind::EmbedInject => sig(Ports::exactly(&[F, F]), one_f),
        OpKind::Pool(_) | OpKind::Logits | OpKind::LogitsProcess | OpKind::TopKTopP => {
            sig(one_f, one_f)
        }
        OpKind::SampleSeeded | OpKind::Argmax => sig(one_f, Ports::exactly(&[IDS])),
        // Draft IDs and the target's logits or IDs; the accepted IDs.
        OpKind::VerifyAccept => sig(Ports::many(&[ANY], 1), Ports::exactly(&[IDS])),
        // ---- Data movement ----
        OpKind::Convert => same(Ports::exactly(&[ANY]), Ports::exactly(&[ANY])),
        OpKind::GatherRows | OpKind::ScatterRows => {
            sig(Ports::exactly(&[ANY, IDS]), Ports::exactly(&[ANY]))
        }
        OpKind::PartialReduce => sig(Ports::many(&[F], 1), one_f),
        // KV movement, table deltas and timing act on state the step owns, not on values.
        OpKind::KvPack | OpKind::KvUnpack | OpKind::ApplyTableDeltas | OpKind::Timestamp => {
            sig(Ports::many(&[ANY], 0), Ports::many(&[ANY], 0))
        }
        // Parity taps write to a tap buffer; `HiddenTap` copies a residual stream.
        OpKind::TapGather | OpKind::HiddenTap => sig(Ports::many(&[F], 1), Ports::many(&[ANY], 0)),
        _ => return None,
    })
}
