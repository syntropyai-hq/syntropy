// SPDX-License-Identifier: MIT OR Apache-2.0

//! The model-class rule of RFC-0042 §1.2 on synthetic censuses.
//!
//! The censuses list batch-1 `read_per_token` bytes per tensor group by RFC-0041 §3.1's
//! rule: every byte of dense, head and router groups, one row of the token embedding, and
//! `bytes × top_k / n_experts` for each routed expert. Their shapes are synthetic test
//! inputs, not measurements of any published checkpoint.

use syntropy_core::{CtScheme, DType, DTypeClass, GgmlType, QuantScheme};

const BF16: QuantScheme = QuantScheme::Plain(DType::BF16);
const F32: QuantScheme = QuantScheme::Plain(DType::F32);
/// Symmetric INT4 in groups of 32 along K, BF16 group scales.
const INT4_G32: QuantScheme = QuantScheme::CompressedTensors(CtScheme::PackedInt {
    bits: 4,
    group: 32,
    symmetric: true,
    zero_point: false,
});

/// A Kimi-K2-Thinking-like shape: MLA attention, one dense layer, then MoE layers with
/// 384 routed experts (top 8) and one shared expert.
struct MoeShape {
    layers: u64,
    dense_layers: u64,
    hidden: u64,
    dense_inter: u64,
    expert_inter: u64,
    n_experts: u64,
    top_k: u64,
    n_shared: u64,
    vocab: u64,
    heads: u64,
    q_lora: u64,
    kv_lora: u64,
    qk_nope: u64,
    qk_rope: u64,
    v_head: u64,
}

const KIMI_K2_LIKE: MoeShape = MoeShape {
    layers: 61,
    dense_layers: 1,
    hidden: 7168,
    dense_inter: 18432,
    expert_inter: 2048,
    n_experts: 384,
    top_k: 8,
    n_shared: 1,
    vocab: 163_840,
    heads: 64,
    q_lora: 1536,
    kv_lora: 512,
    qk_nope: 128,
    qk_rope: 64,
    v_head: 128,
};

/// Bytes of one routed expert's three projections in symmetric INT4 with BF16 scales per
/// 32 weights: half a byte per weight plus 2 bytes per 32 weights.
fn int4_g32_bytes(weights: u64) -> u64 {
    weights / 2 + weights / 32 * 2
}

/// One `(scheme, read_per_token bytes)` pair per tensor group, with routed experts stored
/// in `expert_scheme` and everything else (the hot set) in BF16, apart from F32 router
/// biases.
fn census(shape: &MoeShape, expert_scheme: QuantScheme) -> Vec<(QuantScheme, u64)> {
    let h = shape.hidden;
    let qk_head = shape.qk_nope + shape.qk_rope;
    let attn_weights = h * shape.q_lora
        + shape.q_lora * shape.heads * qk_head
        + h * (shape.kv_lora + shape.qk_rope)
        + shape.kv_lora * shape.heads * (shape.qk_nope + shape.v_head)
        + shape.heads * shape.v_head * h;
    let norm_weights = 2 * h + shape.q_lora + shape.kv_lora;
    let expert_weights = 3 * h * shape.expert_inter;
    let expert_bytes = if expert_scheme == INT4_G32 {
        int4_g32_bytes(expert_weights)
    } else {
        expert_weights * 2
    };

    let mut out = vec![
        (BF16, h * 2),               // token embedding: one row per token
        (BF16, shape.vocab * h * 2), // output head
        (BF16, h * 2),               // final norm
    ];
    for layer in 0..shape.layers {
        out.push((BF16, attn_weights * 2));
        out.push((BF16, norm_weights * 2));
        if layer < shape.dense_layers {
            out.push((BF16, 3 * h * shape.dense_inter * 2));
            continue;
        }
        out.push((BF16, shape.n_experts * h * 2)); // router weight
        out.push((F32, shape.n_experts * 4)); // router bias
        out.push((BF16, shape.n_shared * expert_weights * 2)); // shared expert
        for _expert in 0..shape.n_experts {
            out.push((expert_scheme, expert_bytes * shape.top_k / shape.n_experts));
        }
    }
    out
}

fn share_of(census: &[(QuantScheme, u64)], scheme: QuantScheme) -> f64 {
    let total: u64 = census.iter().map(|&(_, b)| b).sum();
    let mine: u64 = census
        .iter()
        .filter(|&&(s, _)| s == scheme)
        .map(|&(_, b)| b)
        .sum();
    mine as f64 / total as f64
}

#[test]
fn kimi_k2_thinking_like_census_is_bits4() {
    let census = census(&KIMI_K2_LIKE, INT4_G32);

    // The census exercises the rule: both the INT4 experts and the BF16 hot set carry at
    // least 10 % of the bytes, so the rule has to pick the lower-precision one, and no
    // single expert group comes near 10 %, so the threshold must apply to the scheme's
    // total rather than to each group.
    let total: u64 = census.iter().map(|&(_, b)| b).sum();
    let int4 = share_of(&census, INT4_G32);
    let bf16 = share_of(&census, BF16);
    assert!(
        int4 >= 0.10 && bf16 >= 0.10,
        "int4 {int4:.3}, bf16 {bf16:.3}"
    );
    let largest_expert_group = census
        .iter()
        .filter(|&&(s, _)| s == INT4_G32)
        .map(|&(_, b)| b)
        .max()
        .unwrap();
    assert!(largest_expert_group * 10 < total);

    assert_eq!(DTypeClass::of_model(census), Ok(DTypeClass::Bits4));
}

#[test]
fn same_shape_with_bf16_experts_is_float16() {
    let census = census(&KIMI_K2_LIKE, BF16);
    assert_eq!(DTypeClass::of_model(census), Ok(DTypeClass::Float16));
}

#[test]
fn gguf_q4_k_bulk_with_q6_k_head_is_bits4() {
    // RFC-0042 §1.2's second example, on a small dense shape.
    let (layers, hidden, inter, vocab) = (32_u64, 4096_u64, 14336_u64, 128_256_u64);
    let q4_k = |weights: u64| weights / 256 * 144; // 144 bytes per 256 weights (§2.1)
    let q6_k = |weights: u64| weights / 256 * 210; // 210 bytes per 256 weights (§2.1)
    let mut census = vec![
        (QuantScheme::Ggml(GgmlType::Q4_K), q4_k(hidden)), // one embedding row
        (QuantScheme::Ggml(GgmlType::Q6_K), q6_k(vocab * hidden)), // output head
        (QuantScheme::Ggml(GgmlType::F32), hidden * 4),    // final norm
    ];
    for _ in 0..layers {
        census.push((QuantScheme::Ggml(GgmlType::Q4_K), q4_k(4 * hidden * hidden)));
        census.push((QuantScheme::Ggml(GgmlType::Q4_K), q4_k(3 * hidden * inter)));
        census.push((QuantScheme::Ggml(GgmlType::F32), 2 * hidden * 4));
    }
    assert_eq!(DTypeClass::of_model(census), Ok(DTypeClass::Bits4));
}
