// SPDX-License-Identifier: MIT OR Apache-2.0

//! The token-ID range check at ingress (PROD-E2-T20; RFC-0050 §2.3, §12.1).
//!
//! The pipeline runs against `syntropy-testkit`'s fake runner, as in `tests/pipeline.rs`. The
//! fake runner's control log counts every `SequenceOpen` it received, and the stages count the
//! routing and admission calls, so a refusal is shown to happen before routing and before any
//! `SequenceOpen`. The fields a `GenerateRequest` carries at M0, the native `tokens` and
//! `logit_bias`, are fuzzed through the pipeline; the fields whose endpoints come later are
//! fuzzed through the validator itself, which those endpoints call.

#![cfg(any(target_os = "linux", target_vendor = "apple", windows))]

use std::sync::atomic::{AtomicUsize, Ordering};

use proptest::prelude::*;
use syntropy_api::pipeline::{
    ApiError, ApiShape, ClientMeta, ClientWatch, Detokenize, GenerateRequest, Ingress, Input,
    ModelInfo, ModelRef, Route, SamplingRequest, SequenceIdMinter, Stages, StopMatch, StopStep,
    StreamMode, TailChannels, TokenField, TokenIdOutOfRange, check_token_ids, vocab_limit,
};
use syntropy_core::{ErrorClass, InstanceId, NodeId, PlanEpoch, SequenceControl};
use syntropy_testkit::sim::fake_runner::{FakeRegion, FakeRunner, Script};
use syntropy_testkit::sim::{SimClock, SimEntropy};

const NODE: NodeId = NodeId([9; 32]);

/// Stages over a model whose `vocab_limit` is `limit`. Stage 5 passes native tokens through,
/// after `bos` when it is set, and renders a text prompt as its bytes' values plus `shift`.
struct Gate {
    limit: u32,
    bos: Option<u32>,
    shift: u32,
    routes: AtomicUsize,
    admits: AtomicUsize,
}

impl Gate {
    fn new(limit: u32) -> Gate {
        Gate {
            limit,
            bos: None,
            shift: 0,
            routes: AtomicUsize::new(0),
            admits: AtomicUsize::new(0),
        }
    }
}

impl Stages for Gate {
    fn resolve_model(&self, _: &ModelRef) -> Result<ModelInfo, ApiError> {
        Ok(ModelInfo {
            name: "m".to_owned(),
            context_length: 1 << 20,
            vocab_limit: self.limit,
        })
    }

    fn tokenize(&self, _: &ModelInfo, input: &Input, _: bool) -> Result<Vec<u32>, ApiError> {
        let mut out: Vec<u32> = self.bos.into_iter().collect();
        match input {
            Input::Tokens(tokens) => out.extend(tokens),
            Input::Prompt { prompt, .. } => {
                out.extend(prompt.bytes().map(|b| u32::from(b) + self.shift));
            }
            Input::Messages { .. } => {
                return Err(ApiError::InvalidRequest("no messages here".to_owned()));
            }
        }
        Ok(out)
    }

    fn route(&self, _: &ModelInfo, _: &[u32], _: u64) -> Result<Route, ApiError> {
        self.routes.fetch_add(1, Ordering::Relaxed);
        Ok(Route {
            instance: InstanceId(1),
            epoch: PlanEpoch { term: 1, seq: 1 },
            prefill_chunk: 0,
            initial_window: None,
            tail: TailChannels::default(),
        })
    }

    fn admit(&self, _: &Route, _: u32, _: u32) -> Result<(), ApiError> {
        self.admits.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn detokenizer(&self, _: &ModelInfo) -> Box<dyn Detokenize> {
        Box::new(Silent)
    }

    fn stop_matcher(&self, _: &[String]) -> Box<dyn StopMatch> {
        Box::new(Silent)
    }
}

struct Silent;

impl Detokenize for Silent {
    fn push(&mut self, _: u32) -> String {
        String::new()
    }

    fn finish(&mut self) -> String {
        String::new()
    }
}

impl StopMatch for Silent {
    fn push(&mut self, text: &str) -> StopStep {
        StopStep {
            release: text.to_owned(),
            matched: None,
        }
    }

    fn finish(&mut self) -> String {
        String::new()
    }
}

fn request(input: Input, logit_bias: Vec<(u32, f32)>) -> GenerateRequest {
    GenerateRequest {
        shape: ApiShape::Native,
        model: ModelRef("m".to_owned()),
        input,
        sampling: SamplingRequest {
            seed: Some(1),
            logit_bias,
            max_tokens: Some(4),
            ..SamplingRequest::default()
        },
        n: 1,
        logprobs: None,
        stream: StreamMode::Off,
        deadline: None,
        raw_special_tokens: false,
        client: ClientMeta::default(),
        trace: false,
    }
}

/// Runs `body` with an ingress over `stages` and a fake runner, and returns the `SequenceOpen`s
/// that reached the runner, in order. A generation `body` starts and drops is cancelled and
/// released; those controls are left out.
fn with_ingress(stages: &Gate, body: impl FnOnce(&Ingress<'_>)) -> Vec<SequenceControl> {
    let region = FakeRegion::new().unwrap();
    let minter = SequenceIdMinter::new(NODE, 1);
    let clock = SimClock::new();
    let handle = clock.handle();
    let entropy = SimEntropy::new(3);
    let ingress = Ingress::new(region.daemon(), &minter, stages, &handle, &entropy).unwrap();
    let mut runner = FakeRunner::new(region.runner(), Script::new()).unwrap();
    body(&ingress);
    runner.step().unwrap();
    runner
        .log()
        .iter()
        .filter(|c| matches!(c, SequenceControl::Open(_)))
        .cloned()
        .collect()
}

/// The IDs a fuzz case puts in a field: the limit, one past it, and `u32::MAX`, those that are
/// not below the limit.
fn bad_ids(limit: u32) -> Vec<u32> {
    let mut ids = vec![limit, limit.saturating_add(1), u32::MAX];
    ids.dedup();
    ids
}

/// The limits the cases run at: the smallest model, the fake vocabulary, a real one (151,936,
/// the Qwen3 embedding rows), and the largest a `u32` allows.
const LIMITS: [u32; 4] = [1, 10, 151_936, u32::MAX - 1];

/// Acceptance 1, through the pipeline: an ID at `vocab_limit`, one past it or at `u32::MAX`, at
/// the start, the middle or the end of `tokens` or of `logit_bias`, is refused with
/// `token_id_out_of_range` naming the field, the first offending position and the limit. Nothing
/// is routed or admitted and no `SequenceOpen` reaches the runner.
#[test]
fn out_of_range_ids_are_refused_before_any_open() {
    for limit in LIMITS {
        let gate = Gate::new(limit);
        let ok = limit - 1;
        let log = with_ingress(&gate, |ingress| {
            for bad in bad_ids(limit) {
                for at in [0usize, 2, 4] {
                    // A later bad ID too, so the error must name the first.
                    let mut ids = vec![ok, 0, ok, 0, ok, bad];
                    ids[at] = bad;
                    let want = |field| {
                        ApiError::TokenIdOutOfRange(TokenIdOutOfRange {
                            field,
                            position: at as u64,
                            id: bad,
                            vocab_limit: limit,
                        })
                    };

                    let tokens = request(Input::Tokens(ids.clone()), vec![]);
                    let got = ingress.start(&tokens, ClientWatch::default()).unwrap_err();
                    assert_eq!(got, want(TokenField::Tokens), "limit {limit}");

                    let bias = ids.iter().map(|&id| (id, 1.0)).collect();
                    let biased = request(Input::Tokens(vec![0]), bias);
                    let got = ingress.start(&biased, ClientWatch::default()).unwrap_err();
                    assert_eq!(got, want(TokenField::LogitBias), "limit {limit}");
                }
            }
        });
        assert!(log.is_empty(), "limit {limit}: {log:?}");
        assert_eq!(gate.routes.load(Ordering::Relaxed), 0);
        assert_eq!(gate.admits.load(Ordering::Relaxed), 0);
    }
}

/// Acceptance: valid IDs pass unchanged. IDs up to `vocab_limit - 1` in `tokens` and
/// `logit_bias` reach the `SequenceOpen` as the client sent them, nothing clamped or dropped.
#[test]
fn valid_ids_pass_unchanged() {
    for limit in LIMITS {
        let gate = Gate::new(limit);
        let top = limit - 1;
        let tokens = vec![0, top, top / 2, top];
        let bias = vec![(top, -2.0), (0, 3.0), (top, 1.0)];
        let log = with_ingress(&gate, |ingress| {
            ingress
                .start(
                    &request(Input::Tokens(tokens.clone()), bias.clone()),
                    ClientWatch::default(),
                )
                .unwrap();
        });
        let [SequenceControl::Open(open)] = log.as_slice() else {
            panic!("limit {limit}: {log:?}");
        };
        assert_eq!(open.prompt.as_slice(), tokens.as_slice());
        assert_eq!(open.sampling.logit_bias.as_slice(), bias.as_slice());
        assert_eq!(gate.routes.load(Ordering::Relaxed), 1);
    }
}

/// `tokens` is checked before `logit_bias`, and both before the rendered prompt; positions are
/// the client's, not the rendered prompt's, so a BOS stage 5 adds does not shift them.
#[test]
fn fields_are_checked_in_order_at_the_client_s_positions() {
    let mut gate = Gate::new(10);
    gate.bos = Some(1);
    with_ingress(&gate, |ingress| {
        let both = request(Input::Tokens(vec![3, 10]), vec![(11, 0.0)]);
        assert_eq!(
            ingress.start(&both, ClientWatch::default()).unwrap_err(),
            ApiError::TokenIdOutOfRange(TokenIdOutOfRange {
                field: TokenField::Tokens,
                position: 1,
                id: 10,
                vocab_limit: 10,
            })
        );
    });
}

/// A rendered prompt with an ID out of range, which only a tokenizer that disagrees with the
/// manifest produces, is refused too, and never routed.
#[test]
fn a_rendered_prompt_out_of_range_is_refused() {
    let mut gate = Gate::new(200);
    gate.shift = 100;
    let log = with_ingress(&gate, |ingress| {
        let text = Input::Prompt {
            prompt: "a!".to_owned(),
            suffix: None,
        };
        // 'a' + 100 = 197 passes; '!' + 100 = 133 passes; so this one is fine...
        ingress
            .start(&request(text, vec![]), ClientWatch::default())
            .unwrap();
        // ...and 'z' + 100 = 222 is not.
        let text = Input::Prompt {
            prompt: "az".to_owned(),
            suffix: None,
        };
        assert_eq!(
            ingress
                .start(&request(text, vec![]), ClientWatch::default())
                .unwrap_err(),
            ApiError::TokenIdOutOfRange(TokenIdOutOfRange {
                field: TokenField::RenderedPrompt,
                position: 1,
                id: 222,
                vocab_limit: 200,
            })
        );
    });
    assert_eq!(log.len(), 1, "{log:?}");
    assert_eq!(gate.routes.load(Ordering::Relaxed), 1);
}

/// RFC-0050 §12.1: the error is `InvalidRequest`, code `token_id_out_of_range` (number 33 in
/// §12's registry), and its message names the field, the position and the limit.
#[test]
fn the_error_s_class_code_and_message() {
    let e = TokenIdOutOfRange {
        field: TokenField::LogitBias,
        position: 7,
        id: 151_936,
        vocab_limit: 151_936,
    };
    assert_eq!(e.class(), ErrorClass::InvalidRequest);
    assert_eq!(TokenIdOutOfRange::CODE, "token_id_out_of_range");
    assert_eq!(TokenIdOutOfRange::NUMBER, 33);
    assert_eq!(
        e.to_string(),
        "logit_bias[7] is token ID 151936, not below the model's vocab_limit of 151936"
    );
    assert_eq!(
        ApiError::from(e).to_string(),
        "token ID out of range: logit_bias[7] is token ID 151936, not below the model's \
         vocab_limit of 151936"
    );
}

/// Every field the row names, through the one validator: the three bad IDs at each position are
/// refused naming that field and position; IDs below the limit pass; an empty field passes.
#[test]
fn the_validator_covers_every_field() {
    let fields = [
        (TokenField::Tokens, "tokens"),
        (TokenField::TeacherTokens, "teacher.tokens"),
        (TokenField::LogitBias, "logit_bias"),
        (TokenField::StopTokenIds, "stop_token_ids"),
        (TokenField::CompletionsPrompt, "prompt"),
        (TokenField::EmbeddingsInput, "input"),
        (TokenField::GrammarTokens, "constraint"),
        (TokenField::RenderedPrompt, "rendered prompt"),
    ];
    for (field, name) in fields {
        assert_eq!(field.name(), name);
        for limit in LIMITS {
            assert_eq!(check_token_ids(field, [], limit), Ok(()));
            assert_eq!(check_token_ids(field, [0, limit - 1], limit), Ok(()));
            for bad in bad_ids(limit) {
                for at in 0..3 {
                    let mut ids = [0, limit - 1, 0];
                    ids[at] = bad;
                    assert_eq!(
                        check_token_ids(field, ids, limit),
                        Err(TokenIdOutOfRange {
                            field,
                            position: at as u64,
                            id: bad,
                            vocab_limit: limit,
                        })
                    );
                }
            }
        }
    }
    // A limit of 0 refuses every ID.
    assert!(check_token_ids(TokenField::Tokens, [0], 0).is_err());
}

/// `vocab_limit` is the smallest of the three table sizes, whichever it is.
#[test]
fn vocab_limit_is_the_smallest_table() {
    assert_eq!(vocab_limit(151_936, 151_936, 151_936), 151_936);
    assert_eq!(vocab_limit(151_669, 151_936, 151_936), 151_669);
    assert_eq!(vocab_limit(151_936, 151_669, 151_936), 151_669);
    assert_eq!(vocab_limit(151_936, 151_936, 151_669), 151_669);
    assert_eq!(vocab_limit(0, u32::MAX, u32::MAX), 0);
}

proptest! {
    /// Any IDs: the validator refuses exactly when some ID is not below the limit, and then names
    /// the first such ID and its position.
    #[test]
    fn the_validator_names_the_first_offending_id(
        limit in prop_oneof![1u32..300, Just(u32::MAX - 1), Just(u32::MAX)],
        ids in proptest::collection::vec(
            prop_oneof![0u32..300, Just(u32::MAX - 1), Just(u32::MAX)],
            0..40,
        ),
    ) {
        let first = ids.iter().position(|&id| id >= limit);
        let got = check_token_ids(TokenField::StopTokenIds, ids.iter().copied(), limit);
        match first {
            None => prop_assert_eq!(got, Ok(())),
            Some(at) => prop_assert_eq!(got, Err(TokenIdOutOfRange {
                field: TokenField::StopTokenIds,
                position: at as u64,
                id: ids[at],
                vocab_limit: limit,
            })),
        }
    }
}
