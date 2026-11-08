// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CLI client's contract with native generate (PROD-E4-T02 acceptance 2; RFC-0051 §1,
//! ADR-0503 decisions 2 and 3): the client is built on the same `syntropy-api` types the server
//! parses and writes and the OpenAPI document is generated from, so the three cannot disagree.
//!
//! - **Same types.** `Client::generate` takes `syntropy_api::native::wire::GenerateBody` and
//!   yields `EventLine`s, and `Client::generate_once` yields a `GenerateResponse`; the
//!   function-pointer coercions below fail to compile if the client ever takes its own copies.
//! - **Same document.** Against a real server and ingress over the fake runner, the body the
//!   client sends and every event and response it reads back validate against the served
//!   document's schemas for those types.

#![cfg(any(target_os = "linux", target_vendor = "apple", windows))]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{Value, json};
use syntropy_api::config::{ApiSection, TlsSource};
use syntropy_api::native::wire::{Event, EventLine, GenerateBody, GenerateResponse, Sampling};
use syntropy_api::native::{GenerateQueue, drive, generate_queue};
use syntropy_api::openapi;
use syntropy_api::pipeline::{
    ApiError, Detokenize, Ingress, Input, ModelInfo, ModelRef, Route, SequenceIdMinter, Stages,
    StopMatch, StopStep, TailChannels, Tokenized,
};
use syntropy_api::routes::{ApiState, NodeStatus};
use syntropy_api::server::Server;
use syntropy_cli::client::{Client, Target};
use syntropy_cli::error::CliError;
use syntropy_core::{FinishReason, InstanceId, NodeId, PlanEpoch};
use syntropy_testkit::sim::fake_runner::{FakeRegion, FakeRunner, Script, Stream};
use syntropy_testkit::sim::{SimClock, SimEntropy};

mod common;

/// Stages over one model: token input only, each token detokenized to `t<id> `.
struct Model;

impl Stages for Model {
    fn resolve_model(&self, model: &ModelRef) -> Result<ModelInfo, ApiError> {
        Ok(ModelInfo {
            name: model.0.clone(),
            context_length: 4096,
            vocab_limit: 1000,
        })
    }

    fn tokenize(&self, _: &ModelInfo, input: &Input, _: bool) -> Result<Tokenized, ApiError> {
        match input {
            Input::Tokens(tokens) => Ok(tokens.clone().into()),
            _ => Err(ApiError::Unsupported("tokens only".to_owned())),
        }
    }

    fn route(&self, _: &ModelInfo, _: &[u32], _: u64) -> Result<Route, ApiError> {
        Ok(Route {
            instance: InstanceId(1),
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
        Box::new(Words)
    }

    fn stop_matcher(&self, _: &[String]) -> Box<dyn StopMatch> {
        Box::new(Words)
    }
}

struct Words;

impl Detokenize for Words {
    fn push(&mut self, token: u32) -> String {
        format!("t{token} ")
    }

    fn finish(&mut self) -> String {
        String::new()
    }
}

impl StopMatch for Words {
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

struct Node(GenerateQueue);

impl NodeStatus for Node {
    fn has_state_replica(&self) -> bool {
        true
    }

    fn generate_queue(&self) -> Option<&GenerateQueue> {
        Some(&self.0)
    }
}

/// Serves the API with an ingress over a fake runner that answers every sequence with tokens
/// 4, 5 and 6, runs `client` with a CLI client reaching it over loopback HTTP, and stops it all.
fn with_daemon<T>(client: impl FnOnce(&Client) -> T) -> T {
    let region = FakeRegion::new().unwrap();
    let script = Script::new().otherwise(
        Stream::new()
            .tokens([4, 5])
            .tokens([6])
            .finish(FinishReason::Length),
    );
    let (stop_driver, stop_runner) = (AtomicBool::new(false), AtomicBool::new(false));
    let (queue, starts) = generate_queue();
    std::thread::scope(|scope| {
        let runner = scope.spawn(|| {
            let mut runner = FakeRunner::new(region.runner(), script).unwrap();
            runner.serve(&stop_runner).unwrap();
        });
        let (region, stop_driver) = (&region, &stop_driver);
        let driver = scope.spawn(move || {
            let minter = SequenceIdMinter::new(NodeId([3; 32]), 1);
            let clock = SimClock::new();
            let handle = clock.handle();
            let entropy = SimEntropy::new(5);
            let ingress =
                Ingress::new(region.daemon(), &minter, &Model, &handle, &entropy).unwrap();
            drive(&ingress, &starts, stop_driver);
        });
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let api = ApiSection {
            bind: vec!["127.0.0.1:0".parse().unwrap()],
            tls: TlsSource::Auto,
            allow_plaintext_lan: false,
            local_socket: false,
        };
        let state = ApiState {
            version: common::test_version(),
            node: Arc::new(Node(queue)),
        };
        let server = {
            let _context = runtime.enter();
            Server::bind(&api, None, false, state).unwrap()
        };
        let addr = server.local_addrs()[0];
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        runtime.spawn(server.serve(async {
            let _ = stopped.await;
        }));
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let target = Target::Http {
                addr: addr.to_string(),
                remote: false,
                api_key: None,
            };
            client(&Client::reach(vec![target]).unwrap())
        }));
        let _ = stop.send(());
        runtime.shutdown_timeout(std::time::Duration::from_secs(5));
        stop_driver.store(true, Ordering::Release);
        driver.join().unwrap();
        stop_runner.store(true, Ordering::Release);
        runner.join().unwrap();
        out.unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    })
}

/// Checks `value` against the schema the document names `name`.
fn conforms(name: &str, value: &Value) {
    let schema = json!({ "$ref": format!("#/components/schemas/{name}") });
    openapi::validate(openapi::document(), &schema, value)
        .unwrap_or_else(|e| panic!("{name}: {e}\n{value}"));
}

/// Compile-time half: the client's generate methods take and return the API's own wire types.
#[test]
fn the_client_s_generate_methods_are_typed_on_the_api_s_wire_types() {
    type OnEvent<'a> = &'a mut dyn FnMut(EventLine);
    let _: fn(&Client, &GenerateBody, OnEvent<'_>) -> Result<(), CliError> = Client::generate;
    let _: fn(&Client, &GenerateBody) -> Result<GenerateResponse, CliError> = Client::generate_once;
}

/// Run-time half: what the client sends and reads matches the served document, and the server
/// serves what it sends.
#[test]
fn what_the_client_sends_and_reads_matches_the_document() {
    let body = GenerateBody {
        model: "m".to_owned(),
        tokens: Some(vec![1, 2, 3]),
        sampling: Sampling {
            temperature: Some(0.0),
            seed: Some(7),
            max_tokens: Some(3),
            stop_token_ids: vec![999],
            ..Sampling::default()
        },
        return_tokens: true,
        trace: true,
        ..GenerateBody::default()
    };
    let (events, response) = with_daemon(|client| {
        let mut events = Vec::new();
        client
            .generate(&body, &mut |event| events.push(event))
            .unwrap();
        (events, client.generate_once(&body).unwrap())
    });

    // The request, as the client writes it, in both modes.
    for stream in [true, false] {
        let sent = GenerateBody {
            stream: Some(stream),
            ..body.clone()
        };
        conforms("GenerateBody", &serde_json::to_value(&sent).unwrap());
    }

    // The stream: typed events that match the document, numbered from 0.
    assert!(matches!(
        events.first().map(|e| &e.event),
        Some(Event::Started { .. })
    ));
    assert!(matches!(
        events.last().map(|e| &e.event),
        Some(Event::Done { .. })
    ));
    let Event::Started { sequence, .. } = &events[0].event else {
        unreachable!()
    };
    let mut tokens: Vec<u32> = Vec::new();
    for (n, event) in events.iter().enumerate() {
        assert_eq!(event.id, format!("{sequence}:{n}"));
        conforms("EventLine", &serde_json::to_value(event).unwrap());
        if let Event::Text {
            tokens: Some(t), ..
        } = &event.event
        {
            tokens.extend(t);
        }
    }
    assert_eq!(tokens, [4, 5, 6]);

    // The buffered response.
    conforms(
        "GenerateResponse",
        &serde_json::to_value(&response).unwrap(),
    );
    assert_eq!(response.choices[0].text, "t4 t5 t6 ");
    assert_eq!(response.choices[0].tokens.as_deref(), Some(&[4, 5, 6][..]));
    assert_eq!(response.usage.completion_tokens, 3);
    assert_eq!(response.trace_id.as_ref().map(String::len), Some(32));
}

/// A refusal before the stream reaches the client as the daemon's own error.
#[test]
fn a_refused_request_is_the_daemon_s_error() {
    let body = GenerateBody {
        model: "m".to_owned(),
        tokens: Some(vec![1]),
        sampling: Sampling {
            stop_token_ids: vec![1000],
            ..Sampling::default()
        },
        ..GenerateBody::default()
    };
    let (streamed, once) = with_daemon(|client| {
        (
            client.generate(&body, &mut |_| {}).unwrap_err(),
            client.generate_once(&body).unwrap_err(),
        )
    });
    for error in [streamed, once] {
        assert_eq!(error.domain, "api");
        assert_eq!(error.code, "token_id_out_of_range");
        assert!(
            error.message.contains("stop_token_ids[0]"),
            "{}",
            error.message
        );
    }
}
