// SPDX-License-Identifier: MIT OR Apache-2.0

//! The F32 original of one fixture variant: its dimensions and its tensors, each named as GGUF
//! and as a safetensors checkpoint names it (RFC-0060 §12.1).
//!
//! Neither naming is in [`FamilySpec`](syntropy_graph::spec::FamilySpec): the spec says what a
//! checkpoint holds, and the names are the formats' own. The census of the generated files
//! checks every name against the family's classification rules.

use syntropy_graph::spec::{LayerKind, VariantSpec};
use syntropy_kernels::RopeKind;

use super::normal::Normals;
use super::{FixtureError, SizeClass, tokenizer};

/// A fixture's dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    /// Decoder layers.
    pub layers: u32,
    /// The residual width.
    pub hidden: u32,
    /// Query heads.
    pub heads: u32,
    /// KV heads.
    pub kv_heads: u32,
    /// The head dim.
    pub d_head: u32,
    /// The dense FFN's intermediate width.
    pub d_ffn: u32,
    /// The vocabulary: the byte-level tokenizer's.
    pub vocab: u32,
    /// The context length the metadata states.
    pub max_ctx: u32,
}

impl Dims {
    /// The `Nano` dimensions (RFC-0060 §12.1: ≤ 8 layers, hidden 64–256, byte-level vocab): the
    /// layer pattern repeated to at least two layers.
    pub fn nano(v: &VariantSpec) -> Result<Self, FixtureError> {
        let pattern = u32::try_from(v.layers.len()).unwrap_or(u32::MAX);
        if pattern == 0 {
            return Err(FixtureError::Unsupported(format!(
                "variant `{}` has an empty layer pattern",
                v.name
            )));
        }
        let layers = pattern * 2_u32.div_ceil(pattern);
        if layers > 8 {
            return Err(FixtureError::Unsupported(format!(
                "variant `{}`'s layer pattern needs {layers} layers, past Nano's 8",
                v.name
            )));
        }
        Ok(Self {
            layers,
            hidden: 64,
            heads: 4,
            kv_heads: 2,
            d_head: 16,
            d_ffn: 128,
            vocab: tokenizer::VOCAB,
            max_ctx: 4096,
        })
    }

    /// The dimensions for `size`.
    pub fn of(v: &VariantSpec, size: SizeClass) -> Result<Self, FixtureError> {
        match size {
            SizeClass::Nano => Self::nano(v),
            other => Err(FixtureError::Unsupported(format!(
                "size class {other:?} is not generated yet"
            ))),
        }
    }
}

/// The RoPE settings the generator picks for each rule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rope {
    /// The rule.
    pub kind: RopeKind,
    /// θ.
    pub theta: f32,
    /// The scaling factor, for every rule but plain.
    pub factor: f32,
    /// The pre-scaling context, for Llama-3-style and YaRN scaling.
    pub original_ctx: u32,
}

impl Rope {
    /// The settings for `kind`.
    pub fn of(kind: RopeKind) -> Self {
        Self {
            kind,
            theta: 10_000.0,
            factor: match kind {
                RopeKind::Plain => 1.0,
                RopeKind::Linear => 2.0,
                _ => 4.0,
            },
            original_ctx: 1024,
        }
    }
}

/// How a tensor's values are drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Init {
    /// Zero-mean normal scaled by 1/√fan-in, where fan-in is the innermost dim.
    FanIn,
    /// Standard normal: the token table, whose rows are looked up, not multiplied.
    Unit,
    /// All ones: norm gains, which QUAL-E7-T02's calibration tunes.
    Ones,
}

/// One tensor of the original.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    /// Its GGUF name.
    pub gguf: String,
    /// Its safetensors name.
    pub hf: String,
    /// Its shape, outermost first, as safetensors writes it; GGUF writes it reversed.
    pub shape: Vec<u64>,
    /// How its values were drawn.
    pub init: Init,
    /// Q and K rows of a `llama` GGUF file are stored with each head's halves interleaved, as
    /// that format's RoPE pairs them: the number of heads, for such a tensor.
    pub gguf_rope_heads: Option<u32>,
    /// Its values, row-major in `shape`.
    pub data: Vec<f32>,
}

impl Tensor {
    /// The values in GGUF's order: rows permuted for a `llama` Q or K projection, as is.
    pub fn gguf_data(&self) -> Vec<f32> {
        let Some(heads) = self.gguf_rope_heads else {
            return self.data.clone();
        };
        let (rows, cols) = (self.shape[0] as usize, self.shape[1] as usize);
        let d = rows / heads as usize;
        let mut out = vec![0.0; self.data.len()];
        for h in 0..heads as usize {
            for i in 0..d / 2 {
                for j in 0..2 {
                    let to = h * d + 2 * i + j;
                    let from = h * d + j * (d / 2) + i;
                    out[to * cols..(to + 1) * cols]
                        .copy_from_slice(&self.data[from * cols..(from + 1) * cols]);
                }
            }
        }
        out
    }
}

/// The original: its dimensions, RoPE settings and tensors in drawing order.
#[derive(Clone, Debug, PartialEq)]
pub struct Original {
    /// The dimensions.
    pub dims: Dims,
    /// The RoPE settings.
    pub rope: Rope,
    /// Every tensor, in the order its values were drawn and GGUF lists them.
    pub tensors: Vec<Tensor>,
}

struct Builder<'a> {
    normals: &'a mut Normals,
    tensors: Vec<Tensor>,
}

impl Builder<'_> {
    fn push(&mut self, gguf: String, hf: String, shape: &[u64], init: Init) -> &mut Tensor {
        let n: u64 = shape.iter().product();
        let scale = match init {
            Init::FanIn => 1.0 / (shape[shape.len() - 1] as f64).sqrt(),
            Init::Unit => 1.0,
            Init::Ones => 0.0,
        };
        let data = (0..n)
            .map(|_| match init {
                Init::Ones => 1.0,
                _ => (self.normals.draw() * scale) as f32,
            })
            .collect();
        self.tensors.push(Tensor {
            gguf,
            hf,
            shape: shape.to_vec(),
            init,
            gguf_rope_heads: None,
            data,
        });
        self.tensors.last_mut().expect("just pushed")
    }
}

/// Draws the original of `v` at `dims` from `normals`.
pub fn build(v: &VariantSpec, dims: Dims, normals: &mut Normals) -> Result<Original, FixtureError> {
    let Dims {
        layers,
        hidden,
        heads,
        kv_heads,
        d_head,
        d_ffn,
        vocab,
        ..
    } = dims;
    let (h, q, kv, f, voc) = (
        u64::from(hidden),
        u64::from(heads * d_head),
        u64::from(kv_heads * d_head),
        u64::from(d_ffn),
        u64::from(vocab),
    );
    if !matches!(
        v.attention.rope,
        RopeKind::Plain | RopeKind::Linear | RopeKind::Llama3 | RopeKind::Yarn
    ) {
        return Err(FixtureError::Unsupported(format!(
            "the RoPE rule {:?} is not generated yet",
            v.attention.rope
        )));
    }
    let permute = v.gguf_architecture == "llama";
    let mut b = Builder {
        normals,
        tensors: Vec::new(),
    };
    b.push(
        "token_embd.weight".into(),
        "model.embed_tokens.weight".into(),
        &[voc, h],
        Init::Unit,
    );
    for l in 0..layers {
        let kind = v.layers[l as usize % v.layers.len()];
        let g = |n: &str| format!("blk.{l}.{n}");
        let m = |n: &str| format!("model.layers.{l}.{n}");
        b.push(
            g("attn_norm.weight"),
            m("input_layernorm.weight"),
            &[h],
            Init::Ones,
        );
        b.push(
            g("attn_q.weight"),
            m("self_attn.q_proj.weight"),
            &[q, h],
            Init::FanIn,
        )
        .gguf_rope_heads = permute.then_some(heads);
        b.push(
            g("attn_k.weight"),
            m("self_attn.k_proj.weight"),
            &[kv, h],
            Init::FanIn,
        )
        .gguf_rope_heads = permute.then_some(kv_heads);
        b.push(
            g("attn_v.weight"),
            m("self_attn.v_proj.weight"),
            &[kv, h],
            Init::FanIn,
        );
        if v.attention.qkv_bias {
            for (gn, hn, w) in [
                ("attn_q.bias", "self_attn.q_proj.bias", q),
                ("attn_k.bias", "self_attn.k_proj.bias", kv),
                ("attn_v.bias", "self_attn.v_proj.bias", kv),
            ] {
                if permute {
                    return Err(FixtureError::Unsupported(
                        "a Q or K bias on a `llama` GGUF architecture".into(),
                    ));
                }
                // A bias has no fan-in of its own: it is drawn at the scale of its weight's
                // outputs, 1/√hidden.
                let t = b.push(g(gn), m(hn), &[w], Init::Unit);
                let scale = 1.0 / (h as f64).sqrt();
                for x in &mut t.data {
                    *x = (f64::from(*x) * scale) as f32;
                }
            }
        }
        if v.attention.qk_norm {
            b.push(
                g("attn_q_norm.weight"),
                m("self_attn.q_norm.weight"),
                &[u64::from(d_head)],
                Init::Ones,
            );
            b.push(
                g("attn_k_norm.weight"),
                m("self_attn.k_norm.weight"),
                &[u64::from(d_head)],
                Init::Ones,
            );
        }
        b.push(
            g("attn_output.weight"),
            m("self_attn.o_proj.weight"),
            &[h, q],
            Init::FanIn,
        );
        b.push(
            g("ffn_norm.weight"),
            m("post_attention_layernorm.weight"),
            &[h],
            Init::Ones,
        );
        match kind {
            LayerKind::GqaDense => {
                b.push(
                    g("ffn_gate.weight"),
                    m("mlp.gate_proj.weight"),
                    &[f, h],
                    Init::FanIn,
                );
                b.push(
                    g("ffn_up.weight"),
                    m("mlp.up_proj.weight"),
                    &[f, h],
                    Init::FanIn,
                );
                b.push(
                    g("ffn_down.weight"),
                    m("mlp.down_proj.weight"),
                    &[h, f],
                    Init::FanIn,
                );
            }
            other => {
                return Err(FixtureError::Unsupported(format!(
                    "layer kind {other:?} is not generated yet"
                )));
            }
        }
    }
    b.push(
        "output_norm.weight".into(),
        "model.norm.weight".into(),
        &[h],
        Init::Ones,
    );
    if !v.tied_embeddings {
        b.push(
            "output.weight".into(),
            "lm_head.weight".into(),
            &[voc, h],
            Init::FanIn,
        );
    }
    if v.mtp_modules > 0 {
        return Err(FixtureError::Unsupported(
            "MTP modules are not generated yet".into(),
        ));
    }
    Ok(Original {
        dims,
        rope: Rope::of(v.attention.rope),
        tensors: b.tensors,
    })
}
