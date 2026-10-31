// SPDX-License-Identifier: MIT OR Apache-2.0

//! The safetensors writer and `config.json` (RFC-0041 §2.2's layout): an 8-byte header length,
//! the JSON header padded with spaces to 8 bytes, then the tensors in name order, back to back.

use std::fmt::Write;

use syntropy_graph::spec::VariantSpec;
use syntropy_kernels::RopeKind;

use super::model::Original;
use super::tokenizer::{self, json_str};

/// The safetensors file of `o`, every tensor F32.
pub fn write(o: &Original) -> Vec<u8> {
    let mut order: Vec<_> = o.tensors.iter().collect();
    order.sort_by(|a, b| a.hf.cmp(&b.hf));
    let mut header = String::from(r#"{"__metadata__": {"format": "pt"}"#);
    let mut offset = 0u64;
    for t in &order {
        let end = offset + 4 * t.data.len() as u64;
        let shape: Vec<String> = t.shape.iter().map(u64::to_string).collect();
        write!(
            header,
            r#", {}: {{"dtype": "F32", "shape": [{}], "data_offsets": [{offset}, {end}]}}"#,
            json_str(&t.hf),
            shape.join(", ")
        )
        .expect("a String");
        offset = end;
    }
    header.push('}');
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut out = Vec::with_capacity(8 + header.len() + offset as usize);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for t in order {
        for x in &t.data {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    out
}

/// `config.json` for `v` at `o`'s dimensions.
pub fn config_json(v: &VariantSpec, o: &Original) -> String {
    let d = o.dims;
    let r = o.rope;
    let rope = match r.kind {
        RopeKind::Plain => "null".to_owned(),
        RopeKind::Linear => format!(r#"{{"rope_type": "linear", "factor": {:?}}}"#, r.factor),
        RopeKind::Llama3 => format!(
            r#"{{"rope_type": "llama3", "factor": {:?}, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": {}}}"#,
            r.factor, r.original_ctx
        ),
        RopeKind::Yarn => format!(
            r#"{{"rope_type": "yarn", "factor": {:?}, "original_max_position_embeddings": {}, "beta_fast": 32.0, "beta_slow": 1.0}}"#,
            r.factor, r.original_ctx
        ),
        other => unreachable!("`model::build` refuses the RoPE rule {other:?}"),
    };
    format!(
        concat!(
            "{{\n",
            "  \"architectures\": [{arch}],\n",
            "  \"model_type\": {model_type},\n",
            "  \"num_hidden_layers\": {layers},\n",
            "  \"hidden_size\": {hidden},\n",
            "  \"intermediate_size\": {ffn},\n",
            "  \"num_attention_heads\": {heads},\n",
            "  \"num_key_value_heads\": {kv_heads},\n",
            "  \"head_dim\": {d_head},\n",
            "  \"vocab_size\": {vocab},\n",
            "  \"max_position_embeddings\": {ctx},\n",
            "  \"rms_norm_eps\": 1e-06,\n",
            "  \"rope_theta\": {theta:?},\n",
            "  \"rope_scaling\": {rope},\n",
            "  \"hidden_act\": \"silu\",\n",
            "  \"attention_bias\": false,\n",
            "  \"tie_word_embeddings\": {tied},\n",
            "  \"bos_token_id\": {bos},\n",
            "  \"eos_token_id\": {eos},\n",
            "  \"torch_dtype\": \"float32\"\n",
            "}}\n"
        ),
        arch = json_str(v.architecture),
        model_type = json_str(v.model_type),
        layers = d.layers,
        hidden = d.hidden,
        ffn = d.d_ffn,
        heads = d.heads,
        kv_heads = d.kv_heads,
        d_head = d.d_head,
        vocab = d.vocab,
        ctx = d.max_ctx,
        theta = r.theta,
        rope = rope,
        tied = v.tied_embeddings,
        bos = tokenizer::BOS,
        eos = tokenizer::EOS,
    )
}
