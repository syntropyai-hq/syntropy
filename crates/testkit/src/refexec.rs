// SPDX-License-Identifier: MIT OR Apache-2.0

//! The reference executor's side of RFC-0060 §4.1. `RefExecutor` drives `graph::reference` with
//! weights, numeric mode, evaluation order and taps; it is QUAL-E4-T01's.
//!
//! This module holds the stub implementations of the interpreter's contract
//! (`syntropy_graph::reference`), which tests use until real sources exist:
//! - [`InMemoryWeights`]: a [`WeightSource`] over F32 tensors held in memory;
//! - [`RecordingSink`]: a [`RefSink`] that records every logits row and hidden block it receives.

use syntropy_core::{DetHashMap, LayerIx};
use syntropy_graph::reference::{HiddenBlock, RefSink, WeightError, WeightKey, WeightSource};

/// A [`WeightSource`] over F32 tensors held in memory: a fixture's weights, already dequantised.
#[derive(Clone, Debug, Default)]
pub struct InMemoryWeights {
    tensors: DetHashMap<WeightKey, Vec<f32>>,
}

impl InMemoryWeights {
    /// A source with no tensors.
    pub fn new() -> InMemoryWeights {
        InMemoryWeights::default()
    }

    /// Adds `key`'s values, replacing any it held.
    pub fn insert(&mut self, key: WeightKey, values: Vec<f32>) {
        self.tensors.insert(key, values);
    }
}

impl WeightSource for InMemoryWeights {
    fn read_f32(&mut self, key: &WeightKey, out: &mut Vec<f32>) -> Result<(), WeightError> {
        let values = self.tensors.get(key).ok_or(WeightError::Missing(*key))?;
        out.clear();
        out.extend_from_slice(values);
        Ok(())
    }
}

/// One logits row a [`RecordingSink`] received.
#[derive(Clone, Debug, PartialEq)]
pub struct LogitsRow {
    /// The sequence.
    pub seq: u32,
    /// The position.
    pub pos: u32,
    /// The logits.
    pub logits: Vec<f32>,
}

/// One hidden block a [`RecordingSink`] received.
#[derive(Clone, Debug, PartialEq)]
pub struct HiddenRows {
    /// The sequence.
    pub seq: u32,
    /// The layer.
    pub layer: LayerIx,
    /// The rows.
    pub rows: HiddenBlock,
}

/// A [`RefSink`] that records what it receives, in order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RecordingSink {
    /// Every logits row, in the order received.
    pub logits: Vec<LogitsRow>,
    /// Every hidden block, in the order received.
    pub hidden: Vec<HiddenRows>,
}

impl RefSink for RecordingSink {
    fn logits(&mut self, seq: u32, pos: u32, logits: &[f32]) {
        self.logits.push(LogitsRow {
            seq,
            pos,
            logits: logits.to_vec(),
        });
    }

    fn hidden(&mut self, seq: u32, layer: LayerIx, rows: &HiddenBlock) {
        self.hidden.push(HiddenRows {
            seq,
            layer,
            rows: rows.clone(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syntropy_core::{ExpertIx, TensorGroupId};
    use syntropy_graph::ir::{GroupSel, TensorRole, WeightRef};
    use syntropy_graph::reference::{Interpreter, RefDevice, RefMode, RefOptions};

    fn key(group: u32, role: TensorRole) -> WeightKey {
        WeightKey {
            weight: WeightRef {
                group: GroupSel::Fixed(TensorGroupId(group)),
                role,
            },
            expert: None,
        }
    }

    #[test]
    fn in_memory_weights_return_what_they_hold() {
        let mut weights = InMemoryWeights::new();
        let q = key(0, TensorRole::Q);
        weights.insert(q, vec![0.5, -1.0, 2.0]);
        let expert = WeightKey {
            weight: WeightRef {
                group: GroupSel::Experts { layer: LayerIx(1) },
                role: TensorRole::Up,
            },
            expert: Some(ExpertIx(7)),
        };
        weights.insert(expert, vec![3.0]);

        let mut out = vec![9.0; 10];
        weights.read_f32(&q, &mut out).unwrap();
        assert_eq!(out, [0.5, -1.0, 2.0]);
        weights.read_f32(&expert, &mut out).unwrap();
        assert_eq!(out, [3.0]);
        // Another expert of the same group is a different tensor.
        let other = WeightKey {
            expert: Some(ExpertIx(8)),
            ..expert
        };
        assert_eq!(
            weights.read_f32(&other, &mut out),
            Err(WeightError::Missing(other))
        );
        assert_eq!(
            weights.read_f32(&key(0, TensorRole::K), &mut out),
            Err(WeightError::Missing(key(0, TensorRole::K)))
        );
    }

    #[test]
    fn a_recording_sink_keeps_everything_in_order() {
        let mut sink = RecordingSink::default();
        let block = HiddenBlock::new(1, 2, vec![0.25, 0.75]).unwrap();
        sink.logits(0, 0, &[1.0, 2.0]);
        sink.hidden(0, LayerIx(2), &block);
        sink.logits(0, 1, &[3.0, 4.0]);
        assert_eq!(
            sink.logits,
            [
                LogitsRow {
                    seq: 0,
                    pos: 0,
                    logits: vec![1.0, 2.0]
                },
                LogitsRow {
                    seq: 0,
                    pos: 1,
                    logits: vec![3.0, 4.0]
                },
            ]
        );
        assert_eq!(
            sink.hidden,
            [HiddenRows {
                seq: 0,
                layer: LayerIx(2),
                rows: block
            }]
        );
    }

    #[test]
    fn options_with_activation_quantisation_off_reach_the_interpreter_unchanged() {
        // The model-definition oracle's options (RFC-0060 §7.2 check 9).
        let options = RefOptions {
            mode: RefMode::F32Fixed,
            device: RefDevice::Cpu,
            activation_quant: false,
        };
        let interpreter = Interpreter::new(options);
        assert_eq!(*interpreter.options(), options);
        assert!(!interpreter.options().activation_quant);
    }

    #[test]
    fn the_stubs_are_usable_as_trait_objects() {
        // The interpreter takes its weight source and sink as trait objects (RFC-0060 §4.1).
        let mut weights: Box<dyn WeightSource> = Box::new(InMemoryWeights::new());
        let mut out = Vec::new();
        assert!(weights.read_f32(&key(1, TensorRole::O), &mut out).is_err());
        let mut recording = RecordingSink::default();
        let sink: &mut dyn RefSink = &mut recording;
        sink.logits(3, 4, &[5.0]);
        assert_eq!(recording.logits.len(), 1);
    }
}
