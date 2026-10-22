// SPDX-License-Identifier: MIT OR Apache-2.0

//! A test vocabulary and a sampler that applies masks to random logits.

use std::sync::Arc;

use syntropy_text::constraint::{
    ConstraintEngine, OutputConstraint, TokenId, TokenMask, VocabToken, Vocabulary,
};

/// Multi-byte text tokens beside the 256 single bytes, so masks are computed over tokens that
/// span JSON syntax, keys, numbers, escapes and UTF-8 characters, not only over bytes.
const FRAGMENTS: &[&str] = &[
    "{\"",
    "\": ",
    "\", \"",
    "\"}",
    "\"]",
    "[\"",
    ", ",
    ": ",
    "{}",
    "[]",
    "true",
    "false",
    "null",
    "name",
    "type",
    "value",
    "id",
    "query",
    "location",
    "unit",
    "\"name\"",
    "\"id\": ",
    "12",
    "00",
    "-1",
    "0.5",
    ".25",
    "e+3",
    "E-2",
    "\\\"",
    "\\n",
    "\\u00e9",
    "\\\\",
    "abc",
    "xyz",
    "the ",
    " of",
    "2026-",
    "10-22",
    "T12:",
    "00:00",
    "Z\"",
    "+02:00",
    "@example.com",
    "https://",
    ".org",
    "é",
    "日本",
    "ü\"",
    "550e8400-",
    "e29b-",
    "192.168.",
    "::1",
    "fe80::",
];

/// The name of the end-of-sequence token.
pub const EOS: &str = "<|end|>";

/// The test vocabulary: every byte, [`FRAGMENTS`], two special tokens, and [`EOS`] last.
pub fn vocabulary() -> Arc<Vocabulary> {
    let mut tokens: Vec<VocabToken> = (0..=255u8).map(|b| VocabToken::Text(vec![b])).collect();
    tokens.extend(
        FRAGMENTS
            .iter()
            .map(|f| VocabToken::Text(f.as_bytes().to_vec())),
    );
    tokens.push(VocabToken::Special("<|tool|>".to_owned()));
    tokens.push(VocabToken::Special("<|user|>".to_owned()));
    tokens.push(VocabToken::Special(EOS.to_owned()));
    let eos = (tokens.len() - 1) as TokenId;
    Arc::new(Vocabulary::new(tokens, eos).expect("a valid test vocabulary"))
}

/// A small deterministic generator, so every case replays from its seed.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    /// splitmix64.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// One sampled output.
pub struct Sampled {
    /// The text the chosen tokens stand for, without the end token.
    pub bytes: Vec<u8>,
}

/// The random walk's length before the sampler steers toward an end.
const SOFT_BUDGET: usize = 48;

/// When steering, the single-byte tokens taken first, best first after the end token: closing a
/// string (or opening one, which closes at once), then an email's `@`, then closing an array or
/// object, then a comma. A structure then gains at most one more member before it closes, so the
/// walk ends.
const STEER: &[u8] = b"\"@]},";

/// The steering rank of `token`: lower is taken first; `None` for a token outside the list.
fn rank(vocabulary: &Vocabulary, token: TokenId) -> Option<usize> {
    if token == vocabulary.eos() {
        return Some(0);
    }
    match vocabulary.token(token) {
        Some(VocabToken::Text(bytes)) if bytes.len() == 1 => {
            STEER.iter().position(|b| *b == bytes[0]).map(|p| p + 1)
        }
        _ => None,
    }
}

/// Whether the number the output ends in has grown long enough to stop: more than 12 characters,
/// or any digit of exponent. JSON numbers are unbounded, and a long random exponent would overflow
/// the `f64` the test parses into. The longest token adds two digits, so an exponent stays at two
/// digits or fewer.
fn long_number(bytes: &[u8]) -> bool {
    let run: Vec<u8> = bytes
        .iter()
        .rev()
        .take_while(|b| b.is_ascii_digit() || b"eE+-.".contains(b))
        .copied()
        .collect();
    let exponent_digits = run.iter().take_while(|b| b.is_ascii_digit()).count();
    let has_exponent = run.iter().any(|b| *b == b'e' || *b == b'E');
    run.len() > 12 || (has_exponent && exponent_digits >= 1)
}

/// Samples under `constraint`: at every step, a random logit for every token, the mask applied
/// (forbidden tokens to −∞), and a draw from the softmax. After [`SOFT_BUDGET`] steps, or once a
/// number grows long, it steers instead: of the allowed tokens it takes the best ranked one in
/// [`STEER`], and draws as before when none is allowed. Also checks the engine's contract at every
/// step: the mask is never empty, and it allows the end token exactly when the state is
/// accepting.
pub fn sample<E: ConstraintEngine>(
    engine: &E,
    compiled: &E::Compiled,
    seed: u64,
    limit: usize,
) -> Result<Sampled, String> {
    let vocabulary = engine.vocabulary();
    let eos = vocabulary.eos();
    let mut state = engine.new_state(compiled).map_err(|e| e.to_string())?;
    let mut mask = TokenMask::new(vocabulary.len());
    let mut rng = Rng::new(seed);
    let mut bytes = Vec::new();
    let mut logits = vec![0.0f64; vocabulary.len()];
    for step in 0..limit {
        engine
            .mask_into(&mut state, &mut mask)
            .map_err(|e| e.to_string())?;
        if mask.count() == 0 {
            return Err(format!("an empty mask at step {step}"));
        }
        let accepting = engine.is_accepting(&mut state).map_err(|e| e.to_string())?;
        if accepting != mask.allows(eos) {
            return Err(format!(
                "step {step}: accepting is {accepting}, but the mask allows the end token: {}",
                mask.allows(eos)
            ));
        }
        for (token, logit) in logits.iter_mut().enumerate() {
            let random = rng.next_f64() * 4.0;
            *logit = if mask.allows(token as TokenId) {
                random
            } else {
                f64::NEG_INFINITY
            };
        }
        let steering = step >= SOFT_BUDGET || long_number(&bytes);
        let steered = steering
            .then(|| {
                mask.allowed()
                    .filter_map(|t| rank(vocabulary, t).map(|r| (r, t)))
                    .min()
                    .map(|(_, t)| t)
            })
            .flatten();
        let token = steered.unwrap_or_else(|| draw(&logits, &mut rng));
        if !mask.allows(token) {
            return Err(format!("drew forbidden token {token} at step {step}"));
        }
        if token == eos {
            return Ok(Sampled { bytes });
        }
        engine
            .advance(&mut state, token)
            .map_err(|e| e.to_string())?;
        match vocabulary.token(token) {
            Some(VocabToken::Text(text)) => bytes.extend_from_slice(text),
            other => return Err(format!("step {step}: drew {other:?}, which is not text")),
        }
    }
    Err(format!(
        "no end after {limit} tokens: {:?}",
        String::from_utf8_lossy(&bytes)
    ))
}

/// A draw from the softmax of `logits`.
fn draw(logits: &[f64], rng: &mut Rng) -> TokenId {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let mut target = rng.next_f64() * weights.iter().sum::<f64>();
    for (token, weight) in weights.iter().enumerate() {
        if *weight > 0.0 {
            if target < *weight {
                return token as TokenId;
            }
            target -= weight;
        }
    }
    // Rounding: the last token with weight.
    weights
        .iter()
        .rposition(|w| *w > 0.0)
        .expect("an allowed token") as TokenId
}

/// `constraint` compiled for the test vocabulary, then sampled from `seed`, as text.
pub fn sample_text<E: ConstraintEngine>(
    engine: &E,
    constraint: &OutputConstraint,
    seed: u64,
) -> Result<String, String> {
    let output = engine.compile(constraint).map_err(|e| e.to_string())?;
    let sampled = sample(engine, &output.compiled, seed, 4000)?;
    String::from_utf8(sampled.bytes).map_err(|e| format!("not UTF-8: {e}"))
}
