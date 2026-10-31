// SPDX-License-Identifier: MIT OR Apache-2.0

//! The ingress pipeline against `syntropy-testkit`'s fake runner (PROD-E2-T02; QUAL-E3-T18).
//!
//! Each test runs the pipeline on the daemon side of a real region and the fake runner on its
//! runner side, both stepped by hand on one thread: nothing waits on time. The stages are fakes:
//! a token's text is `VOCAB[token]`, and stop strings are found by a naive search with hold-back.

#![cfg(any(target_os = "linux", target_vendor = "apple", windows))]

use syntropy_api::pipeline::{
    ApiError, ApiShape, ClientGone, ClientMeta, ClientWatch, Detokenize, EventSink, GenEvent,
    GenerateRequest, Ingress, Input, ModelInfo, ModelRef, Outcome, Progress, Route,
    SamplingRequest, SequenceIdMinter, Stages, StopMatch, StopReason, StopStep, StreamMode,
    TailChannels, Usage,
};
use syntropy_core::{
    CancelReason, FinishReason, InstanceId, Nanos, NodeId, PlanEpoch, SequenceControl,
};
use syntropy_testkit::sim::fake_runner::{self, FakeRegion, FakeRunner, Script, SeqState, Stream};
use syntropy_testkit::sim::{SimClock, SimEntropy};

const NODE: NodeId = NodeId([6; 32]);
const CONTEXT: u32 = 64;
/// The fake model's `vocab_limit`: one past the last token of [`VOCAB`].
const VOCAB_LIMIT: u32 = VOCAB.len() as u32;

/// Token `t`'s text.
const VOCAB: [&str; 10] = [
    "<0>", "Hel", "lo", " wo", "rld", "!", " and", " more", "e", "STOP",
];

struct FakeStages;

impl Stages for FakeStages {
    fn resolve_model(&self, model: &ModelRef) -> Result<ModelInfo, ApiError> {
        match model.0.as_str() {
            "m" => Ok(ModelInfo {
                name: "m".to_owned(),
                context_length: CONTEXT,
                vocab_limit: VOCAB_LIMIT,
            }),
            other => Err(ApiError::ModelNotFound(other.to_owned())),
        }
    }

    fn tokenize(&self, _: &ModelInfo, input: &Input, _: bool) -> Result<Vec<u32>, ApiError> {
        match input {
            Input::Tokens(tokens) => Ok(tokens.clone()),
            _ => Err(ApiError::InvalidRequest("tokens only".to_owned())),
        }
    }

    fn route(&self, _: &ModelInfo, _: &[u32], _: u64) -> Result<Route, ApiError> {
        Ok(Route {
            instance: InstanceId(3),
            epoch: PlanEpoch { term: 1, seq: 1 },
            prefill_chunk: 0,
            initial_window: None,
            tail: TailChannels::default(),
        })
    }

    fn admit(&self, _: &Route, _: u32, _: u32) -> Result<(), ApiError> {
        Ok(())
    }

    fn detokenizer(&self, _: &ModelInfo) -> Box<dyn Detokenize> {
        Box::new(Vocab)
    }

    fn stop_matcher(&self, stops: &[String]) -> Box<dyn StopMatch> {
        Box::new(NaiveStops {
            stops: stops.to_vec(),
            held: String::new(),
        })
    }
}

struct Vocab;

impl Detokenize for Vocab {
    fn push(&mut self, token: u32) -> String {
        VOCAB[token as usize].to_owned()
    }

    fn finish(&mut self) -> String {
        String::new()
    }
}

/// Finds the first stop string in the held text; holds back the longest suffix that is a proper
/// prefix of a stop string.
struct NaiveStops {
    stops: Vec<String>,
    held: String,
}

impl StopMatch for NaiveStops {
    fn push(&mut self, text: &str) -> StopStep {
        self.held.push_str(text);
        let found = self
            .stops
            .iter()
            .filter_map(|stop| self.held.find(stop.as_str()).map(|at| (at, stop)))
            .min();
        if let Some((at, stop)) = found {
            let stop = stop.clone();
            let release = self.held[..at].to_owned();
            self.held.clear();
            return StopStep {
                release,
                matched: Some(stop),
            };
        }
        let keep = (1..=self.held.len())
            .rev()
            .find(|&n| {
                let tail = &self.held[self.held.len() - n..];
                self.stops.iter().any(|stop| stop.starts_with(tail))
            })
            .unwrap_or(0);
        let release = self.held[..self.held.len() - keep].to_owned();
        self.held.drain(..self.held.len() - keep);
        StopStep {
            release,
            matched: None,
        }
    }

    fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

/// The encoder: records every event; with `fail_after`, the write after that many fails.
#[derive(Default)]
struct Sink {
    events: Vec<GenEvent>,
    fail_after: Option<usize>,
    refused: usize,
}

impl EventSink for Sink {
    fn emit(&mut self, event: GenEvent) -> Result<(), ClientGone> {
        if self.fail_after.is_some_and(|n| self.events.len() >= n) {
            self.refused += 1;
            return Err(ClientGone);
        }
        self.events.push(event);
        Ok(())
    }
}

impl Sink {
    fn text(&self) -> String {
        self.events
            .iter()
            .filter_map(|event| match event {
                GenEvent::TextDelta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn finished(&self) -> Option<(&StopReason, &Usage)> {
        self.events.iter().find_map(|event| match event {
            GenEvent::Finished { reason, usage, .. } => Some((reason, usage)),
            _ => None,
        })
    }
}

fn request(prompt: Vec<u32>, stop: &[&str]) -> GenerateRequest {
    GenerateRequest {
        shape: ApiShape::Native,
        model: ModelRef("m".to_owned()),
        input: Input::Tokens(prompt),
        sampling: SamplingRequest {
            seed: Some(42),
            stop: stop.iter().map(|s| (*s).to_owned()).collect(),
            max_tokens: Some(16),
            ..SamplingRequest::default()
        },
        n: 1,
        logprobs: None,
        stream: StreamMode::Sse {
            include_usage: true,
        },
        deadline: None,
        raw_special_tokens: false,
        client: ClientMeta::default(),
        trace: false,
    }
}

/// The region, the node's minter, the clock and the entropy every test's ingress borrows.
struct World {
    region: FakeRegion,
    minter: SequenceIdMinter,
    clock: SimClock,
    entropy: SimEntropy,
}

impl World {
    fn new() -> World {
        World {
            region: FakeRegion::new().unwrap(),
            minter: SequenceIdMinter::new(NODE, 1),
            clock: SimClock::new(),
            entropy: SimEntropy::new(7),
        }
    }
}

/// Acceptance 1: a disconnect sends `Cancel` before the next encoder write. The client goes away
/// while a token is on the ring: the next poll writes nothing, and the runner receives `Cancel`
/// and `Release` and emits nothing more.
#[test]
fn a_disconnect_sends_cancel_before_the_next_encoder_write() {
    let world = World::new();
    let handle = world.clock.handle();
    let ingress = Ingress::new(
        world.region.daemon(),
        &world.minter,
        &FakeStages,
        &handle,
        &world.entropy,
    )
    .unwrap();
    let mut runner = FakeRunner::new(
        world.region.runner(),
        Script::new().otherwise(Stream::new().tokens([1, 2, 3, 4, 5])),
    )
    .unwrap();
    let watch = ClientWatch::default();
    let mut generation = ingress
        .start(&request(vec![9, 9], &[]), watch.clone())
        .unwrap();
    let seq = generation.seq();
    let mut sink = Sink::default();

    runner.step().unwrap();
    assert_eq!(generation.poll(&mut sink), Progress::Pending);
    assert_eq!(sink.text(), "Hel");
    let written = sink.events.len();

    runner.step().unwrap(); // token 2 is on the ring
    watch.client_gone();
    assert_eq!(
        generation.poll(&mut sink),
        Progress::Ended(Outcome::ClientGone)
    );
    assert_eq!(
        sink.events.len(),
        written,
        "nothing written after the disconnect"
    );

    runner.run_until_idle(10).unwrap();
    let log = runner.log();
    assert_eq!(
        &log[1..],
        &[
            SequenceControl::Cancel {
                seq,
                reason: CancelReason::ClientGone
            },
            SequenceControl::Release { seq, persist: None },
        ]
    );
    assert_eq!(runner.state(seq), None, "released");
    assert!(runner.anomalies().is_empty(), "{:?}", runner.anomalies());
    assert_eq!(ingress.transcript(seq), None);
}

/// A write that fails is a disconnect too: `Cancel` and `Release` follow, and the pipeline tries
/// no other write.
#[test]
fn a_failed_write_cancels_and_writes_nothing_more() {
    let world = World::new();
    let handle = world.clock.handle();
    let ingress = Ingress::new(
        world.region.daemon(),
        &world.minter,
        &FakeStages,
        &handle,
        &world.entropy,
    )
    .unwrap();
    let mut runner = FakeRunner::new(
        world.region.runner(),
        Script::new().otherwise(Stream::new().tokens([1, 2, 3, 4, 5])),
    )
    .unwrap();
    let mut generation = ingress
        .start(&request(vec![9], &[]), ClientWatch::default())
        .unwrap();
    let seq = generation.seq();
    // `Started` and the first delta are written; the second delta's write fails.
    let mut sink = Sink {
        fail_after: Some(2),
        ..Sink::default()
    };
    runner.step().unwrap();
    runner.step().unwrap();
    runner.step().unwrap();
    assert_eq!(
        generation.poll(&mut sink),
        Progress::Ended(Outcome::ClientGone)
    );
    assert_eq!(sink.refused, 1, "one failed write, no retries");
    runner.step().unwrap();
    assert!(matches!(
        runner.log().last(),
        Some(SequenceControl::Release { .. })
    ));
    assert_eq!(runner.state(seq), None);
}

/// Dropping a generation that has not ended cancels and releases it, as a disconnect does.
#[test]
fn dropping_a_live_generation_cancels_it() {
    let world = World::new();
    let handle = world.clock.handle();
    let ingress = Ingress::new(
        world.region.daemon(),
        &world.minter,
        &FakeStages,
        &handle,
        &world.entropy,
    )
    .unwrap();
    let mut runner = FakeRunner::new(
        world.region.runner(),
        Script::new().otherwise(Stream::new().tokens([1, 2, 3])),
    )
    .unwrap();
    let generation = ingress
        .start(&request(vec![9], &[]), ClientWatch::default())
        .unwrap();
    let seq = generation.seq();
    drop(generation);
    runner.step().unwrap();
    assert_eq!(runner.log().len(), 3, "{:?}", runner.log());
    assert_eq!(runner.state(seq), None);
    assert!(ingress.live().is_empty());
}

/// Acceptance 2: the transcript holds the prompt and the emitted tokens until release, and is
/// gone after it.
#[test]
fn the_transcript_holds_prompt_and_emitted_tokens_until_release() {
    let world = World::new();
    let handle = world.clock.handle();
    let ingress = Ingress::new(
        world.region.daemon(),
        &world.minter,
        &FakeStages,
        &handle,
        &world.entropy,
    )
    .unwrap();
    let mut runner = FakeRunner::new(
        world.region.runner(),
        Script::new().otherwise(
            Stream::new()
                .tokens([1, 2])
                .window([3, 4])
                .tokens([5])
                .finish(FinishReason::Stop),
        ),
    )
    .unwrap();
    let mut generation = ingress
        .start(&request(vec![8, 9, 8], &[]), ClientWatch::default())
        .unwrap();
    let seq = generation.seq();
    let mut sink = Sink::default();

    let transcript = ingress.transcript(seq).unwrap();
    assert_eq!(transcript.prompt, [8, 9, 8]);
    assert!(transcript.emitted.is_empty());

    for emitted in [vec![1], vec![1, 2], vec![1, 2, 3, 4]] {
        runner.step().unwrap();
        assert_eq!(generation.poll(&mut sink), Progress::Pending);
        let transcript = ingress.transcript(seq).unwrap();
        assert_eq!(transcript.prompt, [8, 9, 8]);
        assert_eq!(transcript.emitted, emitted);
        assert_eq!(transcript.rng_counter, emitted.len() as u64);
    }

    runner.step().unwrap(); // token 5 with the finish
    assert_eq!(
        generation.poll(&mut sink),
        Progress::Ended(Outcome::Finished(StopReason::EndOfTurn))
    );
    assert_eq!(ingress.transcript(seq), None, "dropped at release");
    assert_eq!(sink.text(), "Hello world!");
    assert_eq!(
        sink.finished().map(|(_, usage)| *usage),
        Some(Usage {
            prompt_tokens: 3,
            completion_tokens: 5,
            cached_tokens: 0
        })
    );
    runner.step().unwrap();
    assert!(matches!(
        runner.log(),
        [SequenceControl::Open(_), SequenceControl::Release { .. }]
    ));
}

/// Acceptance 3: tokens after a matched stop are discarded and not counted. The stop string
/// " world" spans tokens 3 and 4 of a five-token window; tokens 5 and 6 in the same window, and a
/// frame already in flight, are discarded. Usage counts tokens 1 to 4, the transcript held them,
/// and the runner gets `Cancel` then `Release`.
#[test]
fn tokens_after_a_matched_stop_are_discarded_and_not_counted() {
    let world = World::new();
    let handle = world.clock.handle();
    let ingress = Ingress::new(
        world.region.daemon(),
        &world.minter,
        &FakeStages,
        &handle,
        &world.entropy,
    )
    .unwrap();
    let mut runner = FakeRunner::new(
        world.region.runner(),
        Script::new().otherwise(Stream::new().window([1, 2, 3, 4, 5, 6]).tokens([7, 7])),
    )
    .unwrap();
    let mut generation = ingress
        .start(&request(vec![9], &[" world"]), ClientWatch::default())
        .unwrap();
    let seq = generation.seq();
    let mut sink = Sink::default();

    runner.step().unwrap(); // the window
    runner.step().unwrap(); // a frame in flight behind it
    let progress = generation.poll(&mut sink);
    assert_eq!(
        progress,
        Progress::Ended(Outcome::Finished(StopReason::StopSequence(
            " world".to_owned()
        )))
    );
    assert_eq!(sink.text(), "Hello", "the text before the stop string");
    assert_eq!(
        sink.finished().map(|(_, usage)| usage.completion_tokens),
        Some(4),
        "tokens 5 and 6 are not counted"
    );
    assert_eq!(generation.completion_tokens(), 4);
    assert!(
        !sink
            .events
            .iter()
            .skip_while(|event| !matches!(event, GenEvent::Finished { .. }))
            .skip(1)
            .any(|_| true),
        "nothing after the final event"
    );

    runner.run_until_idle(10).unwrap();
    assert_eq!(
        &runner.log()[1..],
        &[
            SequenceControl::Cancel {
                seq,
                reason: CancelReason::StopString
            },
            SequenceControl::Release { seq, persist: None },
        ]
    );
    // The frame in flight was taken with the window and discarded with it: it is not a stray,
    // and polling again changes nothing.
    assert_eq!(generation.poll(&mut sink), progress);
    assert_eq!(ingress.stray_entries(), 0);
}

/// With `return_tokens` (PROD-E4-T13), each entry's counted tokens follow its text in one
/// `Tokens` event, and a matched stop string cuts them where it cuts the count: the tokens in the
/// events add up to the usage. Off, no `Tokens` event is sent.
#[test]
fn returned_tokens_add_up_to_the_usage_even_past_a_stop() {
    for on in [true, false] {
        let world = World::new();
        let handle = world.clock.handle();
        let ingress = Ingress::new(
            world.region.daemon(),
            &world.minter,
            &FakeStages,
            &handle,
            &world.entropy,
        )
        .unwrap();
        let mut runner = FakeRunner::new(
            world.region.runner(),
            Script::new().otherwise(Stream::new().tokens([1, 2]).window([3, 4, 5, 6])),
        )
        .unwrap();
        let mut generation = ingress
            .start(&request(vec![9], &[" world"]), ClientWatch::default())
            .unwrap();
        generation.return_tokens(on);
        let mut sink = Sink::default();
        runner.run_until_idle(10).unwrap();
        while generation.poll(&mut sink) == Progress::Pending {
            runner.step().unwrap();
        }
        let returned: Vec<Vec<u32>> = sink
            .events
            .iter()
            .filter_map(|event| match event {
                GenEvent::Tokens { tokens, .. } => Some(tokens.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            sink.finished().map(|(_, usage)| usage.completion_tokens),
            Some(4)
        );
        if on {
            assert_eq!(returned, [vec![1], vec![2], vec![3, 4]]);
            // Each `Tokens` event comes after the text its tokens released.
            let last_text = sink
                .events
                .iter()
                .rposition(|e| matches!(e, GenEvent::TextDelta { .. }))
                .unwrap();
            let last_tokens = sink
                .events
                .iter()
                .rposition(|e| matches!(e, GenEvent::Tokens { .. }))
                .unwrap();
            assert!(last_text < last_tokens);
        } else {
            assert!(returned.is_empty());
        }
        assert_eq!(sink.text(), "Hello");
    }
}

/// A stop string that ends in the middle of a token: the token is counted, its text after the
/// stop is not delivered, and the stop matcher's hold-back keeps a partial match out of the
/// stream until it fails.
#[test]
fn a_stop_inside_a_token_counts_that_token() {
    let world = World::new();
    let handle = world.clock.handle();
    let ingress = Ingress::new(
        world.region.daemon(),
        &world.minter,
        &FakeStages,
        &handle,
        &world.entropy,
    )
    .unwrap();
    let mut runner = FakeRunner::new(
        world.region.runner(),
        Script::new().otherwise(Stream::new().tokens([1, 2, 3, 4, 5, 6])),
    )
    .unwrap();
    let mut generation = ingress
        .start(
            &request(vec![9], &["lo w", "rld!x"]),
            ClientWatch::default(),
        )
        .unwrap();
    let mut sink = Sink::default();
    runner.step().unwrap();
    runner.step().unwrap();
    assert_eq!(generation.poll(&mut sink), Progress::Pending);
    assert_eq!(
        sink.text(),
        "Hel",
        "\"lo\" is held back: it may start \"lo w\""
    );
    runner.step().unwrap();
    assert_eq!(
        generation.poll(&mut sink),
        Progress::Ended(Outcome::Finished(StopReason::StopSequence(
            "lo w".to_owned()
        )))
    );
    assert_eq!(sink.text(), "Hel");
    assert_eq!(generation.completion_tokens(), 3);
}

/// Tokens in flight after a sequence ends are counted as strays, never delivered.
#[test]
fn entries_after_the_end_are_strays() {
    let world = World::new();
    let handle = world.clock.handle();
    let ingress = Ingress::new(
        world.region.daemon(),
        &world.minter,
        &FakeStages,
        &handle,
        &world.entropy,
    )
    .unwrap();
    let mut runner = FakeRunner::new(
        world.region.runner(),
        Script::new()
            .in_order(Stream::new().tokens([1, 2, 3, 4]))
            .in_order(Stream::new().tokens([5]).finish(FinishReason::Length)),
    )
    .unwrap();
    let watch = ClientWatch::default();
    let mut first = ingress
        .start(&request(vec![9], &[]), watch.clone())
        .unwrap();
    let mut second = ingress
        .start(&request(vec![9], &[]), ClientWatch::default())
        .unwrap();
    let mut sink = Sink::default();
    runner.step().unwrap();
    watch.client_gone();
    assert_eq!(first.poll(&mut sink), Progress::Ended(Outcome::ClientGone));
    // The frame the runner wrote before it read the Cancel still carries the first sequence's
    // token; the runner now reads the Cancel and the Release.
    let mut other = Sink::default();
    runner.step().unwrap();
    assert_eq!(
        second.poll(&mut other),
        Progress::Ended(Outcome::Finished(StopReason::Length))
    );
    assert_eq!(other.text(), "!");
    assert_eq!(ingress.stray_entries(), 1, "the first sequence's token 1");
    assert_eq!(runner.state(first.seq()), None, "released");
}

/// Stages 3 to 9 refuse before any `SequenceOpen` is sent.
#[test]
fn refusals_send_no_open() {
    let world = World::new();
    let handle = world.clock.handle();
    let ingress = Ingress::new(
        world.region.daemon(),
        &world.minter,
        &FakeStages,
        &handle,
        &world.entropy,
    )
    .unwrap();
    let mut runner = FakeRunner::new(world.region.runner(), Script::new()).unwrap();

    let mut long = request(vec![1; 60], &[]);
    long.sampling.max_tokens = Some(5);
    assert_eq!(
        ingress.start(&long, ClientWatch::default()).unwrap_err(),
        ApiError::ContextLengthExceeded {
            need: 65,
            limit: CONTEXT
        }
    );
    let mut unknown = request(vec![1], &[]);
    unknown.model = ModelRef("x".to_owned());
    assert_eq!(
        ingress.start(&unknown, ClientWatch::default()).unwrap_err(),
        ApiError::ModelNotFound("x".to_owned())
    );
    let mut many = request(vec![1], &[]);
    many.n = 2;
    assert!(matches!(
        ingress.start(&many, ClientWatch::default()),
        Err(ApiError::Unsupported(_))
    ));
    let mut zero = request(vec![1], &[]);
    zero.sampling.max_tokens = Some(0);
    assert!(matches!(
        ingress.start(&zero, ClientWatch::default()),
        Err(ApiError::InvalidRequest(_))
    ));
    runner.step().unwrap();
    assert!(runner.log().is_empty());
    assert!(ingress.live().is_empty());
}

/// Stage 9's `SequenceOpen` carries the prompt, the request's sampling and limits, the route's
/// choices and the principal; without a client seed it carries one ingress drew, and no choice
/// index. `max_tokens` defaults to the rest of the context.
#[test]
fn the_open_carries_the_request() {
    let world = World::new();
    let handle = world.clock.handle();
    let ingress = Ingress::new(
        world.region.daemon(),
        &world.minter,
        &FakeStages,
        &handle,
        &world.entropy,
    )
    .unwrap();
    let mut runner = FakeRunner::new(world.region.runner(), Script::new()).unwrap();

    let mut asked = request(vec![4, 5, 6], &["x"]);
    asked.sampling.temperature = Some(0.0);
    asked.sampling.logit_bias = vec![(5, -1.5)];
    asked.client.principal = Some([1; 16]);
    asked.deadline = Some(Nanos(5_000_000_000));
    let seeded = ingress.start(&asked, ClientWatch::default()).unwrap();
    let mut unseeded = request(vec![4], &[]);
    unseeded.sampling.seed = None;
    unseeded.sampling.max_tokens = None;
    let drawn = ingress.start(&unseeded, ClientWatch::default()).unwrap();
    runner.step().unwrap();

    let opens: Vec<_> = runner
        .log()
        .iter()
        .map(|control| match control {
            SequenceControl::Open(open) => open.clone(),
            other => panic!("not an open: {other:?}"),
        })
        .collect();
    let (first, second) = (&opens[0], &opens[1]);
    assert_eq!(first.seq, seeded.seq());
    assert_eq!(first.ingress, NODE);
    assert_eq!(first.prompt.as_slice(), [4, 5, 6]);
    assert_eq!(first.sampling.seed, 42);
    assert_eq!(first.choice_index, Some(0));
    assert_eq!(first.sampling.temperature, 0.0);
    assert_eq!(first.sampling.logit_bias.as_slice(), [(5, -1.5)]);
    assert_eq!(first.limits.max_tokens, 16);
    assert_eq!(first.limits.deadline, Some(Nanos(5_000_000_000)));
    assert_eq!(first.prefix_scope.principal, Some([1; 16]));
    assert_ne!(first.seq, second.seq);
    assert_eq!(second.sampling.seed, drawn.seed());
    assert_eq!(second.choice_index, None);
    assert_eq!(second.limits.max_tokens, CONTEXT - 1);
}

/// [`FakeStages`] with the route's tail on other nodes, its channels to the ingress and to the
/// head with these capabilities.
struct RemoteTail(TailChannels);

impl Stages for RemoteTail {
    fn resolve_model(&self, model: &ModelRef) -> Result<ModelInfo, ApiError> {
        FakeStages.resolve_model(model)
    }

    fn tokenize(&self, model: &ModelInfo, input: &Input, raw: bool) -> Result<Vec<u32>, ApiError> {
        FakeStages.tokenize(model, input, raw)
    }

    fn route(&self, model: &ModelInfo, prompt: &[u32], need: u64) -> Result<Route, ApiError> {
        Ok(Route {
            tail: self.0,
            ..FakeStages.route(model, prompt, need)?
        })
    }

    fn admit(&self, route: &Route, prompt_tokens: u32, max_tokens: u32) -> Result<(), ApiError> {
        FakeStages.admit(route, prompt_tokens, max_tokens)
    }

    fn detokenizer(&self, model: &ModelInfo) -> Box<dyn Detokenize> {
        FakeStages.detokenizer(model)
    }

    fn stop_matcher(&self, stops: &[String]) -> Box<dyn StopMatch> {
        FakeStages.stop_matcher(stops)
    }
}

/// DIST-E8-T18: stage 9 asks the tail for the chosen tokens' logprobs alone (`Some(0)`) only
/// when the tail's channels to the ingress and to the head both negotiated capability bit 16
/// `CHOSEN_LOGPROBS`, or do not cross `net`. With the bit missing at the ingress only, or at the
/// head only, it asks for one alternative (RFC-0050 §3.1). Other requests go as they came.
#[test]
fn the_open_asks_for_chosen_logprobs_only_where_both_channels_carry_them() {
    let bit = syntropy_net::frame::capability::CHOSEN_LOGPROBS;
    let remote = |to_ingress, to_head| TailChannels {
        to_ingress: Some(to_ingress),
        to_head: Some(to_head),
    };
    let cases = [
        ("in process", TailChannels::default(), Some(0), Some(0)),
        ("both channels", remote(bit, bit), Some(0), Some(0)),
        (
            "missing at the ingress",
            remote(!bit, bit),
            Some(0),
            Some(1),
        ),
        ("missing at the head", remote(bit, !bit), Some(0), Some(1)),
        ("alternatives", remote(0, 0), Some(3), Some(3)),
        ("no logprobs", remote(0, 0), None, None),
    ];
    for (name, tail, requested, asked) in cases {
        let world = World::new();
        let handle = world.clock.handle();
        let stages = RemoteTail(tail);
        let ingress = Ingress::new(
            world.region.daemon(),
            &world.minter,
            &stages,
            &handle,
            &world.entropy,
        )
        .unwrap();
        let mut runner = FakeRunner::new(world.region.runner(), Script::new()).unwrap();
        let mut req = request(vec![4], &[]);
        req.logprobs = requested;
        let _generation = ingress.start(&req, ClientWatch::default()).unwrap();
        runner.step().unwrap();
        match runner.log() {
            [SequenceControl::Open(open)] => {
                assert_eq!(open.sampling.logprobs, asked, "{name}");
            }
            other => panic!("{name}: {other:?}"),
        }
    }
}

/// `Started` comes first, with the route and the prompt length; `Finished` carries the times on
/// the pipeline's clock.
#[test]
fn started_and_finished_carry_the_route_and_the_times() {
    let mut world = World::new();
    let handle = world.clock.handle();
    let ingress = Ingress::new(
        world.region.daemon(),
        &world.minter,
        &FakeStages,
        &handle,
        &world.entropy,
    )
    .unwrap();
    let mut runner = FakeRunner::new(
        world.region.runner(),
        Script::new().otherwise(Stream::new().tokens([1, 2]).finish(FinishReason::Length)),
    )
    .unwrap();
    let mut generation = ingress
        .start(&request(vec![9, 9], &[]), ClientWatch::default())
        .unwrap();
    let mut sink = Sink::default();
    world.clock.advance(Nanos(1_000));
    runner.step().unwrap();
    generation.poll(&mut sink);
    world.clock.advance(Nanos(500));
    runner.step().unwrap();
    generation.poll(&mut sink);

    assert_eq!(
        sink.events[0],
        GenEvent::Started {
            seq: generation.seq(),
            instance: InstanceId(3),
            epoch: PlanEpoch { term: 1, seq: 1 },
            prompt_tokens: 2,
            cached_tokens: 0,
        }
    );
    match sink.events.last() {
        Some(GenEvent::Finished {
            reason, timings, ..
        }) => {
            assert_eq!(*reason, StopReason::Length);
            assert_eq!(timings.ttft, Some(Nanos(1_000)));
            assert_eq!(timings.total, Nanos(1_500));
        }
        other => panic!("{other:?}"),
    }
}

/// A `Cancel` that finds `local_sequences` full waits and goes first once there is room; an
/// `Open` that finds it full fails with `Capacity`. The generation ends with its `Cancel` and
/// final `Release` both waiting, and the owner's `flush` sends them, in order.
#[test]
fn controls_wait_for_room_and_the_final_release_goes_out_on_flush() {
    let region = FakeRegion::with_layout(&fake_runner::layout(2, 4096)).unwrap();
    let minter = SequenceIdMinter::new(NODE, 1);
    let clock = SimClock::new();
    let handle = clock.handle();
    let entropy = SimEntropy::new(1);
    let ingress = Ingress::new(region.daemon(), &minter, &FakeStages, &handle, &entropy).unwrap();
    let mut runner = FakeRunner::new(region.runner(), Script::new()).unwrap();

    let watch = ClientWatch::default();
    let mut first = ingress
        .start(&request(vec![1], &[]), watch.clone())
        .unwrap();
    let second = ingress
        .start(&request(vec![1], &[]), ClientWatch::default())
        .unwrap();
    assert!(matches!(
        ingress.start(&request(vec![1], &[]), ClientWatch::default()),
        Err(ApiError::Capacity(_))
    ));
    watch.client_gone();
    let mut sink = Sink::default();
    assert_eq!(first.poll(&mut sink), Progress::Ended(Outcome::ClientGone));
    assert_eq!(ingress.waiting_controls(), 2, "Cancel and Release wait");

    runner.step().unwrap();
    ingress.flush().unwrap();
    assert_eq!(ingress.waiting_controls(), 0);
    runner.step().unwrap();
    assert_eq!(
        &runner.log()[2..],
        &[
            SequenceControl::Cancel {
                seq: first.seq(),
                reason: CancelReason::ClientGone
            },
            SequenceControl::Release {
                seq: first.seq(),
                persist: None
            },
        ],
        "the final Release went out"
    );
    assert_eq!(runner.state(first.seq()), None);
    assert_eq!(runner.state(second.seq()), Some(SeqState::Running));
    assert!(runner.anomalies().iter().all(|anomaly| matches!(
        anomaly,
        syntropy_testkit::sim::fake_runner::Anomaly::Unscripted(_)
    )));
}

/// Two ingresses on one node, one per runner, share the node's minter, so they never mint the same
/// `SequenceId` (the review of #1917, B1).
#[test]
fn two_ingresses_on_one_node_mint_distinct_ids() {
    let minter = SequenceIdMinter::new(NODE, 1);
    let (first_region, second_region) = (FakeRegion::new().unwrap(), FakeRegion::new().unwrap());
    let clock = SimClock::new();
    let handle = clock.handle();
    let entropy = SimEntropy::new(3);
    let first = Ingress::new(
        first_region.daemon(),
        &minter,
        &FakeStages,
        &handle,
        &entropy,
    )
    .unwrap();
    let second = Ingress::new(
        second_region.daemon(),
        &minter,
        &FakeStages,
        &handle,
        &entropy,
    )
    .unwrap();
    let mut seqs = Vec::new();
    let mut live = Vec::new();
    for _ in 0..3 {
        for ingress in [&first, &second] {
            let generation = ingress
                .start(&request(vec![1], &[]), ClientWatch::default())
                .unwrap();
            seqs.push(generation.seq());
            live.push(generation);
        }
    }
    let ns: Vec<u32> = seqs.iter().map(|seq| seq.n).collect();
    assert_eq!(ns, [0, 1, 2, 3, 4, 5]);
    let mut unique = seqs.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), seqs.len(), "{seqs:?}");
    assert!(
        seqs.iter()
            .all(|seq| seq.boot == 1 && seq.ingress == NODE.short())
    );
}

/// The client goes away between two writes of one `poll`: the write after it is never made, and
/// `Cancel` and `Release` follow (the review of #1917, N2). The sink sets the watch itself after
/// its second write, as the HTTP layer would from another task.
#[test]
fn a_disconnect_between_two_writes_of_one_poll_stops_the_next_write() {
    struct GoneAfter {
        events: Vec<GenEvent>,
        watch: ClientWatch,
        after: usize,
    }
    impl EventSink for GoneAfter {
        fn emit(&mut self, event: GenEvent) -> Result<(), ClientGone> {
            self.events.push(event);
            if self.events.len() == self.after {
                self.watch.client_gone();
            }
            Ok(())
        }
    }

    let world = World::new();
    let handle = world.clock.handle();
    let ingress = Ingress::new(
        world.region.daemon(),
        &world.minter,
        &FakeStages,
        &handle,
        &world.entropy,
    )
    .unwrap();
    let mut runner = FakeRunner::new(
        world.region.runner(),
        Script::new().otherwise(Stream::new().window([1, 2, 3]).tokens([4])),
    )
    .unwrap();
    let watch = ClientWatch::default();
    let mut generation = ingress
        .start(&request(vec![9], &[]), watch.clone())
        .unwrap();
    let seq = generation.seq();
    runner.step().unwrap(); // three tokens in one entry: three writes in one poll, after Started
    let mut sink = GoneAfter {
        events: Vec::new(),
        watch,
        after: 2,
    };
    assert_eq!(
        generation.poll(&mut sink),
        Progress::Ended(Outcome::ClientGone)
    );
    assert_eq!(
        sink.events.len(),
        2,
        "Started and \"Hel\" only: {:?}",
        sink.events
    );
    assert!(matches!(
        &sink.events[1],
        GenEvent::TextDelta { text, .. } if text == "Hel"
    ));
    runner.step().unwrap();
    assert_eq!(
        &runner.log()[1..],
        &[
            SequenceControl::Cancel {
                seq,
                reason: CancelReason::ClientGone
            },
            SequenceControl::Release { seq, persist: None },
        ]
    );
}

// ---- Trace context through the pipeline (PROD-E8-T07; RFC-0053 §3.2) ----------------------------

const TRACE_ID: &str = "4bf92f3577b34da6a3ce929d0e0e4736";
const CALLER_SPAN: &str = "00f067aa0ba902b7";

fn traceparent(flags: &str) -> String {
    format!("00-{TRACE_ID}-{CALLER_SPAN}-{flags}")
}

/// What one request does to trace context: the `TraceContext` its `SequenceOpen` carried to the
/// runner, the `traceparent` its response returns, and its root span.
struct Traced {
    open: syntropy_core::sequence::TraceContext,
    response: Option<String>,
    root_span: [u8; 8],
}

fn run_traced(incoming: Option<&str>, trace: bool, honour: bool) -> Traced {
    let world = World::new();
    let handle = world.clock.handle();
    let ingress = Ingress::new(
        world.region.daemon(),
        &world.minter,
        &FakeStages,
        &handle,
        &world.entropy,
    )
    .unwrap();
    ingress.set_honor_remote_sampled(honour);
    let mut runner = FakeRunner::new(
        world.region.runner(),
        Script::new().otherwise(Stream::new().tokens([1])),
    )
    .unwrap();
    let mut req = request(vec![9], &[]);
    req.client.traceparent = incoming.map(str::to_owned);
    req.trace = trace;
    let generation = ingress.start(&req, ClientWatch::default()).unwrap();
    runner.step().unwrap();
    let open = match &runner.log()[0] {
        SequenceControl::Open(open) => open.trace.expect("SequenceOpen carries the trace"),
        other => panic!("{other:?}"),
    };
    Traced {
        open,
        response: generation.response_traceparent(),
        root_span: generation.trace().root_span,
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A response to a request without `traceparent` carries none; nor does one whose header does
/// not parse. A request that carried one gets one back.
#[test]
fn a_response_returns_traceparent_only_when_the_request_carried_one() {
    for (incoming, trace, honour) in [(None, false, false), (None, true, true)] {
        assert_eq!(run_traced(incoming, trace, honour).response, None);
    }
    assert_eq!(
        run_traced(Some("not a traceparent"), false, false).response,
        None
    );
    assert!(
        run_traced(Some(&traceparent("01")), false, false)
            .response
            .is_some()
    );
}

/// A returned `traceparent` has bit 0 set, carries bit 1 exactly when the incoming one did, and
/// never Syntropy's full-trace bit, whatever the request's full-trace state. It continues the
/// trace under ingress's root span, which the `SequenceOpen` names as the head's parent.
#[test]
fn a_returned_traceparent_sets_bit_0_echoes_bit_1_and_never_the_full_trace_bit() {
    for (flags, want) in [("00", "01"), ("01", "01"), ("02", "03"), ("03", "03")] {
        for (trace, honour) in [(false, false), (true, false), (false, true), (true, true)] {
            let t = run_traced(Some(&traceparent(flags)), trace, honour);
            let back = t.response.unwrap();
            assert_eq!(
                back,
                format!("00-{TRACE_ID}-{}-{want}", hex(&t.root_span)),
                "incoming {flags}, trace {trace}, honour {honour}"
            );
            assert_eq!(hex(&t.open.trace_id), TRACE_ID);
            assert_eq!(t.open.span_id, t.root_span);
        }
    }
}

/// An incoming sampled `traceparent` does not force full tracing unless the key is set; `02`
/// never does, key set or not; `03` behaves as `01`; `trace: true` always does. What the head
/// receives is Syntropy's flags alone.
#[test]
fn only_trace_true_or_an_honoured_bit_0_gives_the_head_a_full_trace() {
    use syntropy_core::sequence::TraceContext;
    let full = TraceContext::FULL_TRACE | TraceContext::SAMPLED;
    let flags =
        |incoming: Option<&str>, trace, honour| run_traced(incoming, trace, honour).open.flags;
    assert_eq!(flags(Some(&traceparent("01")), false, false), 0);
    assert_eq!(flags(Some(&traceparent("01")), false, true), full);
    assert_eq!(flags(Some(&traceparent("02")), false, false), 0);
    assert_eq!(flags(Some(&traceparent("02")), false, true), 0);
    assert_eq!(
        flags(Some(&traceparent("03")), false, false),
        flags(Some(&traceparent("01")), false, false)
    );
    assert_eq!(
        flags(Some(&traceparent("03")), false, true),
        flags(Some(&traceparent("01")), false, true)
    );
    assert_eq!(flags(None, true, false), full);
    assert_eq!(flags(None, false, true), 0);
}
