// SPDX-License-Identifier: MIT OR Apache-2.0

//! The literal-tokenization fuzz target of RFC-0043 §8.6 (COMP-E17-T13), run as a test so the
//! unit suite runs it in CI: there is no libFuzzer job yet.
//!
//! An input picks a tokenizer shape, a content position and a list of pieces, each either a
//! needle (an added token's text or a role header) or raw bytes read as lossy UTF-8. The target
//! renders the content into a conversation, encodes it, and asserts that no control ID came from
//! the content and that the prompt decodes back to its text. RFC-0060 §10.2's harness checks the
//! run: no panic, bounded allocation and time, and only allowed rejection classes. Inputs come
//! from proptest's deterministic runner, so a failure reproduces.

#[path = "literal/support.rs"]
#[allow(
    dead_code,
    reason = "shared with the `literal` tests, which use all of it"
)]
mod support;

use proptest::prelude::*;
use proptest::test_runner::{Config, TestRunner};
use syntropy_core::ErrorClass;
use syntropy_testkit::fuzz::{Accepted, Properties};
use syntropy_testkit::guards::CountingAlloc;
use syntropy_text::tokenizer::injection::{self, ContentPosition};
use syntropy_text::tokenizer::{SpanEncoder, SpecialTokens};

use crate::support::{FAMILIES, concat, decode, encoder, normalized, render, template_only};

#[global_allocator]
static ALLOCATOR: CountingAlloc = CountingAlloc;

/// One tokenizer shape, ready before the harness starts counting.
struct Target {
    encoder: SpanEncoder,
    needles: Vec<String>,
}

/// Decodes an input into its content and position, then checks one encoding.
fn run(targets: &[Target], input: &[u8]) -> Result<Accepted, ErrorClass> {
    let Some((&head, mut rest)) = input.split_first() else {
        return Ok(Accepted::plain());
    };
    let family = usize::from(head & 1);
    let position = ContentPosition::ALL[usize::from(head >> 1) % ContentPosition::ALL.len()];
    let target = &targets[family];
    let mut content = String::new();
    while let Some((&tag, tail)) = rest.split_first() {
        if tag & 0x80 != 0 {
            let needle = &target.needles[usize::from(tag & 0x7f) % target.needles.len()];
            content.push_str(needle);
            rest = tail;
        } else {
            let len = usize::from(tag).min(tail.len());
            content.push_str(&String::from_utf8_lossy(&tail[..len]));
            rest = &tail[len..];
        }
    }

    let encoder = &target.encoder;
    let spans = render(FAMILIES[family].layout, position, &content);
    let ids = encoder
        .encode(&spans, SpecialTokens::Literal)
        .map_err(|e| e.class())?;
    let frame = encoder
        .encode(&template_only(&spans), SpecialTokens::Literal)
        .map_err(|e| e.class())?;
    let controls = |ids: &[u32]| -> Vec<u32> {
        ids.iter()
            .copied()
            .filter(|&id| encoder.is_control(id))
            .collect()
    };
    assert_eq!(
        controls(&ids),
        controls(&frame),
        "content {content:?} made a control ID"
    );
    let tokenizer = encoder.tokenizer();
    assert_eq!(
        decode(tokenizer, &ids),
        normalized(tokenizer, &concat(&spans))
    );
    Ok(Accepted::plain())
}

#[test]
fn literal_tokenization_fuzz_target_runs_clean() {
    let targets: Vec<Target> = FAMILIES
        .iter()
        .map(|family| {
            let encoder = encoder(family.json);
            let needles = injection::needles(&encoder, family.markers, family.role_headers);
            Target { encoder, needles }
        })
        .collect();
    let properties = Properties::default();
    let mut runner = TestRunner::new_with_rng(
        Config {
            cases: 2_000,
            ..Config::default()
        },
        proptest::test_runner::TestRng::deterministic_rng(
            proptest::test_runner::RngAlgorithm::ChaCha,
        ),
    );
    runner
        .run(&prop::collection::vec(any::<u8>(), 0..512), |input| {
            prop_assert_eq!(properties.check(&input, |i| run(&targets, i)), Ok(()));
            Ok(())
        })
        .expect("every input keeps RFC-0060 §10.2's properties and the literal invariants");
}
