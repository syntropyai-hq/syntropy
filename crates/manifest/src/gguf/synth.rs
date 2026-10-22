// SPDX-License-Identifier: MIT OR Apache-2.0

//! The synthetic GGUF generator the reader's tests use (test builds only).
//!
//! [`Synth`] writes a GGUF file from a list of key-value pairs and tensor specs, laying the
//! tensors out as ggml does: in order, each at the aligned end of the one before, with the
//! data section at the aligned end of the tensor-info table. Tensor bytes are a fixed
//! pattern, so files are deterministic. [`W`] is the byte writer underneath, which the
//! malformed-input tests use directly to write headers no valid generator would.
//!
//! Every fixture under `tests/data/gguf/` comes from here; none is cut from a real checkpoint
//! or downloaded (RFC-0041 "Testing and validation": seed corpora are project-synthetic).

use syntropy_core::GgmlType;

use super::{ALIGNMENT_KEY, Array, DEFAULT_ALIGNMENT, Value, block_geometry, ggml_type_id, pad};

/// A little-endian byte writer for GGUF fields.
#[derive(Default)]
pub(crate) struct W(pub Vec<u8>);

impl W {
    pub fn u8(&mut self, v: u8) -> &mut Self {
        self.0.push(v);
        self
    }
    pub fn u16(&mut self, v: u16) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn bytes(&mut self, b: &[u8]) -> &mut Self {
        self.0.extend_from_slice(b);
        self
    }
    /// A GGUF string: `u64` length, then the bytes.
    pub fn str(&mut self, s: impl AsRef<[u8]>) -> &mut Self {
        let s = s.as_ref();
        self.u64(s.len() as u64).bytes(s)
    }
    /// The magic, the version and the two counts.
    pub fn preamble(&mut self, version: u32, n_tensors: u64, n_kv: u64) -> &mut Self {
        self.bytes(b"GGUF").u32(version).u64(n_tensors).u64(n_kv)
    }
    /// A value's type ID and body.
    pub fn typed(&mut self, v: &Value) -> &mut Self {
        self.u32(v.value_type().id()).value(v)
    }
    /// A value's body.
    pub fn value(&mut self, v: &Value) -> &mut Self {
        match v {
            Value::U8(x) => self.u8(*x),
            Value::I8(x) => self.bytes(&x.to_le_bytes()),
            Value::U16(x) => self.u16(*x),
            Value::I16(x) => self.bytes(&x.to_le_bytes()),
            Value::U32(x) => self.u32(*x),
            Value::I32(x) => self.bytes(&x.to_le_bytes()),
            Value::F32(x) => self.bytes(&x.to_le_bytes()),
            Value::Bool(x) => self.u8(u8::from(*x)),
            Value::Str(s) => self.str(s),
            Value::U64(x) => self.u64(*x),
            Value::I64(x) => self.bytes(&x.to_le_bytes()),
            Value::F64(x) => self.bytes(&x.to_le_bytes()),
            Value::Array(a) => {
                self.u32(a.element_type().id()).u64(a.len() as u64);
                match a {
                    Array::U8(v) => {
                        for x in v {
                            self.u8(*x);
                        }
                    }
                    Array::I8(v) => {
                        for x in v {
                            self.bytes(&x.to_le_bytes());
                        }
                    }
                    Array::U16(v) => {
                        for x in v {
                            self.u16(*x);
                        }
                    }
                    Array::I16(v) => {
                        for x in v {
                            self.bytes(&x.to_le_bytes());
                        }
                    }
                    Array::U32(v) => {
                        for x in v {
                            self.u32(*x);
                        }
                    }
                    Array::I32(v) => {
                        for x in v {
                            self.bytes(&x.to_le_bytes());
                        }
                    }
                    Array::F32(v) => {
                        for x in v {
                            self.bytes(&x.to_le_bytes());
                        }
                    }
                    Array::Bool(v) => {
                        for x in v {
                            self.u8(u8::from(*x));
                        }
                    }
                    Array::Str(v) => {
                        for x in v.iter() {
                            self.str(x);
                        }
                    }
                    Array::U64(v) => {
                        for x in v {
                            self.u64(*x);
                        }
                    }
                    Array::I64(v) => {
                        for x in v {
                            self.bytes(&x.to_le_bytes());
                        }
                    }
                    Array::F64(v) => {
                        for x in v {
                            self.bytes(&x.to_le_bytes());
                        }
                    }
                }
                self
            }
        }
    }
    /// One tensor-info entry.
    pub fn tensor_info(&mut self, name: &str, ne: &[u64], type_id: u32, offset: u64) -> &mut Self {
        self.str(name).u32(ne.len() as u32);
        for d in ne {
            self.u64(*d);
        }
        self.u32(type_id).u64(offset)
    }
}

/// One tensor to generate.
#[derive(Clone, Debug)]
pub(crate) struct TensorSpec {
    pub name: String,
    pub ne: Vec<u64>,
    pub ggml_type: GgmlType,
}

impl TensorSpec {
    /// Its byte length.
    pub fn bytes(&self) -> u64 {
        let (block, block_bytes) = block_geometry(self.ggml_type).unwrap();
        self.ne.iter().product::<u64>() / block * block_bytes
    }
}

/// A synthetic GGUF file.
#[derive(Clone, Debug)]
pub(crate) struct Synth {
    pub version: u32,
    pub kvs: Vec<(String, Value)>,
    pub tensors: Vec<TensorSpec>,
}

impl Synth {
    pub fn new(version: u32) -> Self {
        Self {
            version,
            kvs: Vec::new(),
            tensors: Vec::new(),
        }
    }

    pub fn kv(mut self, key: &str, v: Value) -> Self {
        self.kvs.push((key.to_owned(), v));
        self
    }

    pub fn tensor(mut self, name: &str, ne: &[u64], t: GgmlType) -> Self {
        self.tensors.push(TensorSpec {
            name: name.to_owned(),
            ne: ne.to_vec(),
            ggml_type: t,
        });
        self
    }

    /// The alignment the file declares, or the default.
    pub fn alignment(&self) -> u64 {
        self.kvs
            .iter()
            .find_map(|(k, v)| match (k.as_str(), v) {
                (ALIGNMENT_KEY, Value::U32(a)) => Some(u64::from(*a)),
                _ => None,
            })
            .unwrap_or(DEFAULT_ALIGNMENT)
    }

    /// Each tensor's (data offset, bytes), laid out as ggml lays them.
    pub fn layout(&self) -> Vec<(u64, u64)> {
        let a = self.alignment();
        let mut at = 0;
        self.tensors
            .iter()
            .map(|t| {
                let here = at;
                at = pad(here + t.bytes(), a).unwrap();
                (here, t.bytes())
            })
            .collect()
    }

    /// The header alone: preamble, key-value pairs and tensor infos.
    pub fn header(&self) -> Vec<u8> {
        let mut w = W::default();
        w.preamble(
            self.version,
            self.tensors.len() as u64,
            self.kvs.len() as u64,
        );
        for (k, v) in &self.kvs {
            w.str(k).typed(v);
        }
        for (t, (offset, _)) in self.tensors.iter().zip(self.layout()) {
            w.tensor_info(&t.name, &t.ne, ggml_type_id(t.ggml_type).unwrap(), offset);
        }
        w.0
    }

    /// The whole file: header, padding to the alignment, then each tensor's bytes (a fixed
    /// pattern) with padding between them.
    pub fn build(&self) -> Vec<u8> {
        let mut out = self.header();
        let data_start = pad(out.len() as u64, self.alignment()).unwrap();
        out.resize(data_start as usize, 0);
        for (offset, bytes) in self.layout() {
            out.resize((data_start + offset) as usize, 0);
            out.extend((0..bytes).map(|i| (i.wrapping_mul(31).wrapping_add(7) % 251) as u8));
        }
        out
    }
}
