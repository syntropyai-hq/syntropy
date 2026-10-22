// SPDX-License-Identifier: MIT OR Apache-2.0

//! The contract between the reference interpreter and what drives it (RFC-0060 §4.1, RFC-0043
//! §1).
//!
//! `graph::reference` interprets the IR to define what each family computes. It is production
//! code: the runner calls it for the warm-up canary's digest (RFC-0060 §5.8). What it needs from
//! its driver is defined here, so `testkit`'s `RefExecutor` (a dev-only crate, RFC-0001 §4.2 rule
//! 9) and the runner build against one definition:
//! - [`RefOptions`]: the numeric mode and device ([`RefMode`], [`RefDevice`], from
//!   `kernels::reference`), and whether declared activation quantisation is applied;
//! - [`WeightSource`]: a model's weights, as F32;
//! - [`TapSpec`] and [`RefSink`]: which hidden states to emit, and where logits and hidden
//!   states go, as [`HiddenBlock`]s.
//!
//! [`Interpreter`] holds only its options so far. The interpreter itself is COMP-E15-T03's.

use std::collections::BTreeSet;
use std::fmt;

use syntropy_core::{ExpertIx, LayerIx};
pub use syntropy_kernels::reference::{RefDevice, RefMode};

use crate::ir::WeightRef;

/// How a reference run computes (RFC-0060 §4.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RefOptions {
    /// `F32Fixed` or `F64`.
    pub mode: RefMode,
    /// Where `F32Fixed` runs.
    pub device: RefDevice,
    /// Whether the IR's declared activation quantisation is applied. Off for the
    /// model-definition oracle (RFC-0060 §7.2 check 9), so the graph is compared with the
    /// publisher's F32 forward pass.
    pub activation_quant: bool,
}

/// One weight tensor: a [`WeightRef`] (a census group and a role), and for a layer's routed
/// experts (`GroupSel::Experts`) the expert.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WeightKey {
    /// The group and role.
    pub weight: WeightRef,
    /// The expert, for an expert group; `None` otherwise.
    pub expert: Option<ExpertIx>,
}

/// A model's weights, as the interpreter reads them (RFC-0060 §4.1).
///
/// A source returns F32 values, dequantised with RFC-0042's reference dequantisers, so the
/// interpreter never sees a block format. The interpreter reads one layer's weights at a time,
/// which keeps resident memory to one layer of F32 weights.
pub trait WeightSource {
    /// Writes the F32 values of `key`'s tensor into `out`, replacing what it held.
    fn read_f32(&mut self, key: &WeightKey, out: &mut Vec<f32>) -> Result<(), WeightError>;
}

/// Why a [`WeightSource`] could not supply a tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WeightError {
    /// The source has no such tensor.
    Missing(WeightKey),
    /// The tensor exists but could not be read or dequantised.
    Unreadable {
        /// The tensor.
        key: WeightKey,
        /// What went wrong.
        reason: String,
    },
}

impl fmt::Display for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeightError::Missing(key) => write!(f, "no weight {key:?}"),
            WeightError::Unreadable { key, reason } => {
                write!(f, "weight {key:?} could not be read: {reason}")
            }
        }
    }
}

impl std::error::Error for WeightError {}

/// Which layers' hidden states a run emits through [`RefSink::hidden`]. Logits are always
/// emitted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TapSpec {
    layers: BTreeSet<LayerIx>,
}

impl TapSpec {
    /// Taps no hidden states.
    pub fn none() -> TapSpec {
        TapSpec::default()
    }

    /// Taps the hidden states of `layers`.
    pub fn layers(layers: impl IntoIterator<Item = LayerIx>) -> TapSpec {
        TapSpec {
            layers: layers.into_iter().collect(),
        }
    }

    /// Whether `layer`'s hidden states are emitted.
    pub fn taps(&self, layer: LayerIx) -> bool {
        self.layers.contains(&layer)
    }

    /// The tapped layers, in order.
    pub fn tapped(&self) -> impl Iterator<Item = LayerIx> + '_ {
        self.layers.iter().copied()
    }
}

/// Hidden states: `rows` rows of `width` F32 values each, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct HiddenBlock {
    rows: usize,
    width: usize,
    data: Vec<f32>,
}

impl HiddenBlock {
    /// The block over `data`, or `None` if `data` does not hold exactly `rows × width` values.
    pub fn new(rows: usize, width: usize, data: Vec<f32>) -> Option<HiddenBlock> {
        (rows.checked_mul(width)? == data.len()).then_some(HiddenBlock { rows, width, data })
    }

    /// The number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The values per row.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Row `row`, or `None` past the last.
    pub fn row(&self, row: usize) -> Option<&[f32]> {
        (row < self.rows).then(|| &self.data[row * self.width..(row + 1) * self.width])
    }

    /// Every value, row-major.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

/// Where a reference run's outputs go (RFC-0060 §4.1).
pub trait RefSink {
    /// The full-vocabulary F32 logits at position `pos` of sequence `seq`.
    fn logits(&mut self, seq: u32, pos: u32, logits: &[f32]);
    /// The hidden states of a tapped `layer` for sequence `seq`.
    fn hidden(&mut self, seq: u32, layer: LayerIx, rows: &HiddenBlock);
}

/// The reference interpreter (RFC-0043 §1). It holds only its [`RefOptions`] so far; the
/// interpretation of the IR is COMP-E15-T03's.
#[derive(Debug)]
pub struct Interpreter {
    options: RefOptions,
}

impl Interpreter {
    /// An interpreter that runs with `options`.
    pub fn new(options: RefOptions) -> Interpreter {
        Interpreter { options }
    }

    /// The options it runs with.
    pub fn options(&self) -> &RefOptions {
        &self.options
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tap_spec_taps_exactly_its_layers() {
        let taps = TapSpec::layers([LayerIx(3), LayerIx(0), LayerIx(3)]);
        assert!(taps.taps(LayerIx(0)) && taps.taps(LayerIx(3)));
        assert!(!taps.taps(LayerIx(1)));
        assert_eq!(taps.tapped().collect::<Vec<_>>(), [LayerIx(0), LayerIx(3)]);
        assert_eq!(TapSpec::none().tapped().count(), 0);
    }

    #[test]
    fn a_hidden_block_holds_exactly_rows_times_width_values() {
        let block = HiddenBlock::new(2, 3, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!((block.rows(), block.width()), (2, 3));
        assert_eq!(block.row(1), Some(&[3.0, 4.0, 5.0][..]));
        assert_eq!(block.row(2), None);
        assert_eq!(HiddenBlock::new(2, 3, vec![0.0; 5]), None);
        assert_eq!(HiddenBlock::new(usize::MAX, 2, Vec::new()), None);
        assert_eq!(
            HiddenBlock::new(0, 7, Vec::new()).map(|b| b.rows()),
            Some(0)
        );
    }

    #[test]
    fn the_interpreter_keeps_its_options() {
        let options = RefOptions {
            mode: RefMode::F64,
            device: RefDevice::Gpu(1),
            activation_quant: false,
        };
        assert_eq!(*Interpreter::new(options).options(), options);
    }
}
