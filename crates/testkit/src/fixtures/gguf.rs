// SPDX-License-Identifier: MIT OR Apache-2.0

//! The GGUF v3 writer (RFC-0041 §2.1's layout): metadata in a fixed order, then the tensor-info
//! table, then each tensor at the aligned end of the one before, as the reader requires.

use syntropy_graph::spec::VariantSpec;
use syntropy_kernels::RopeKind;

use super::model::Original;
use super::tokenizer;

/// The data alignment: GGUF's default, so the file states none.
const ALIGNMENT: usize = 32;
/// ggml's type ID for F32.
const GGML_F32: u32 = 0;

/// A metadata value.
enum Value {
    U32(u32),
    I32s(Vec<i32>),
    F32(f32),
    Bool(bool),
    Str(String),
    Strs(Vec<String>),
}

#[derive(Default)]
struct W(Vec<u8>);

impl W {
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn str(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.0.extend_from_slice(s.as_bytes());
    }
    fn value(&mut self, v: &Value) {
        match v {
            Value::U32(x) => {
                self.u32(4);
                self.u32(*x);
            }
            Value::F32(x) => {
                self.u32(6);
                self.0.extend_from_slice(&x.to_le_bytes());
            }
            Value::Bool(x) => {
                self.u32(7);
                self.0.push(u8::from(*x));
            }
            Value::Str(s) => {
                self.u32(8);
                self.str(s);
            }
            Value::I32s(xs) => {
                self.u32(9);
                self.u32(5);
                self.u64(xs.len() as u64);
                for x in xs {
                    self.0.extend_from_slice(&x.to_le_bytes());
                }
            }
            Value::Strs(xs) => {
                self.u32(9);
                self.u32(8);
                self.u64(xs.len() as u64);
                for x in xs {
                    self.str(x);
                }
            }
        }
    }
    fn pad(&mut self) {
        while !self.0.len().is_multiple_of(ALIGNMENT) {
            self.0.push(0);
        }
    }
}

/// Whether GGUF metadata can state `kind`. GGUF stores Llama-3-style scaling as a frequency
/// table and YaRN under keys the M0 dense family does not read (RFC-0043 §5.5), so variants
/// with those rules get no GGUF file.
pub fn states(kind: RopeKind) -> bool {
    matches!(kind, RopeKind::Plain | RopeKind::Linear)
}

/// The metadata, in the order it is written.
fn metadata(v: &VariantSpec, o: &Original, name: &str) -> Vec<(String, Value)> {
    let a = v.gguf_architecture;
    let d = o.dims;
    let k = |s: &str| format!("{a}.{s}");
    let mut kv = vec![
        ("general.architecture".into(), Value::Str(a.into())),
        ("general.name".into(), Value::Str(name.into())),
        (k("block_count"), Value::U32(d.layers)),
        (k("context_length"), Value::U32(d.max_ctx)),
        (k("embedding_length"), Value::U32(d.hidden)),
        (k("feed_forward_length"), Value::U32(d.d_ffn)),
        (k("attention.head_count"), Value::U32(d.heads)),
        (k("attention.head_count_kv"), Value::U32(d.kv_heads)),
        (k("attention.key_length"), Value::U32(d.d_head)),
        (k("attention.value_length"), Value::U32(d.d_head)),
        (k("attention.layer_norm_rms_epsilon"), Value::F32(1e-6)),
        (k("rope.freq_base"), Value::F32(o.rope.theta)),
        (k("rope.dimension_count"), Value::U32(d.d_head)),
        (k("vocab_size"), Value::U32(d.vocab)),
    ];
    if o.rope.kind == RopeKind::Linear {
        kv.push((k("rope.scaling.type"), Value::Str("linear".into())));
        kv.push((k("rope.scaling.factor"), Value::F32(o.rope.factor)));
    }
    let types = (0..d.vocab)
        .map(|id| if id < 256 { 1 } else { 3 })
        .collect();
    kv.extend([
        ("tokenizer.ggml.model".into(), Value::Str("gpt2".into())),
        (
            "tokenizer.ggml.tokens".into(),
            Value::Strs(tokenizer::tokens()),
        ),
        ("tokenizer.ggml.token_type".into(), Value::I32s(types)),
        ("tokenizer.ggml.merges".into(), Value::Strs(Vec::new())),
        (
            "tokenizer.ggml.bos_token_id".into(),
            Value::U32(tokenizer::BOS),
        ),
        (
            "tokenizer.ggml.eos_token_id".into(),
            Value::U32(tokenizer::EOS),
        ),
        (
            "tokenizer.ggml.padding_token_id".into(),
            Value::U32(tokenizer::PAD),
        ),
        ("tokenizer.ggml.add_bos_token".into(), Value::Bool(true)),
    ]);
    kv
}

/// The GGUF file of `o`, every tensor F32.
pub fn write(v: &VariantSpec, o: &Original, name: &str) -> Vec<u8> {
    let kv = metadata(v, o, name);
    let mut w = W::default();
    w.0.extend_from_slice(b"GGUF");
    w.u32(3);
    w.u64(o.tensors.len() as u64);
    w.u64(kv.len() as u64);
    for (key, value) in &kv {
        w.str(key);
        w.value(value);
    }
    let mut offset = 0u64;
    for t in &o.tensors {
        w.str(&t.gguf);
        w.u32(t.shape.len() as u32);
        for ne in t.shape.iter().rev() {
            w.u64(*ne);
        }
        w.u32(GGML_F32);
        w.u64(offset);
        let bytes = 4 * t.data.len() as u64;
        offset = (offset + bytes).next_multiple_of(ALIGNMENT as u64);
    }
    w.pad();
    for t in &o.tensors {
        for x in t.gguf_data() {
            w.0.extend_from_slice(&x.to_le_bytes());
        }
        w.pad();
    }
    w.0
}
