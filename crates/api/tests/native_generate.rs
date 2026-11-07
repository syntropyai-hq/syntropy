// SPDX-License-Identifier: MIT OR Apache-2.0

//! PROD-E4-T13's acceptance tests: `POST /v1/syntropy/generate` for M0 benchmarking, over HTTP
//! on a loopback listener, against `syntropy-testkit`'s fake runner (RFC-0050 §13.1, §13.6).
//!
//! Each test runs the real server, the real ingress pipeline driven by
//! `syntropy_api::native::drive` on a thread of its own, and the fake runner serving the rings on
//! another. The fake runner's control log shows every `SequenceOpen`, and the stages count the
//! routing calls, so a refusal is shown to come before routing and before any `SequenceOpen`.
//!
//! - A token-input request streams NDJSON events whose tokens add up to the response's usage.
//! - With `debug.bench_api` unset, the route answers 404 as an unknown path does. Its loopback
//!   half (a wildcard or LAN listener gets 404 too) is the unit test
//!   `the_gate_opens_on_loopback_with_bench_api_only`, since this build refuses to bind those.
//! - IDs at `vocab_limit`, `vocab_limit + 1` and `u32::MAX` in `tokens` get 400
//!   `token_id_out_of_range` before any `SequenceOpen`, through PROD-E2-T20's validator.

#![cfg(any(target_os = "linux", target_vendor = "apple", windows))]

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use http_body_util::{BodyExt as _, Full};
use hyper::body::Bytes;
use hyper::{Method, Request, StatusCode};
use hyper_util::rt::TokioIo;
use serde_json::{Value, json};
use syntropy_api::config::{ApiSection, TlsSource};
use syntropy_api::native::{GenerateQueue, StartReceiver, drive, generate_queue};
use syntropy_api::pipeline::{
    ApiError, Detokenize, Ingress, Input, ModelInfo, ModelRef, Route, SequenceIdMinter, Stages,
    StopMatch, StopStep, TailChannels,
};
use syntropy_api::routes::{ApiState, NodeStatus};
use syntropy_api::server::Server;
use syntropy_api::version::{Build, ProtocolMajors, SupportTier, VersionInfo};
use syntropy_config::DebugSection;
use syntropy_core::build::{BuildId, BuildProfile, TargetTriple};
use syntropy_core::ids::Blake3;
use syntropy_core::{FinishReason, InstanceId, NodeId, PlanEpoch, SequenceControl};
use syntropy_testkit::sim::fake_runner::{FakeRegion, FakeRunner, Script, Stream};
use syntropy_testkit::sim::{SimClock, SimEntropy};
use tokio::net::TcpStream;

const NODE: NodeId = NodeId([9; 32]);
const PATH: &str = "/v1/syntropy/generate?format=ndjson";

/// Stages over one model whose `vocab_limit` is `limit`. Native tokens pass through; each emitted
/// token detokenizes to `t<id> `.
struct Model {
    limit: u32,
    routes: AtomicUsize,
}

impl Stages for Model {
    fn resolve_model(&self, model: &ModelRef) -> Result<ModelInfo, ApiError> {
        if model.0 != "m" {
            return Err(ApiError::ModelNotFound(model.0.clone()));
        }
        Ok(ModelInfo {
            name: "m".to_owned(),
            context_length: 1 << 16,
            vocab_limit: self.limit,
        })
    }

    fn tokenize(&self, _: &ModelInfo, input: &Input, _: bool) -> Result<Vec<u32>, ApiError> {
        match input {
            Input::Tokens(tokens) => Ok(tokens.clone()),
            _ => Err(ApiError::InvalidRequest("tokens only".to_owned())),
        }
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

/// The node: its debug gates, and the queue to the ingress driver.
struct Node {
    gates: DebugSection,
    queue: GenerateQueue,
}

impl NodeStatus for Node {
    fn has_state_replica(&self) -> bool {
        true
    }

    fn debug_gates(&self) -> DebugSection {
        self.gates
    }

    fn generate_queue(&self) -> Option<&GenerateQueue> {
        Some(&self.queue)
    }
}

fn version() -> VersionInfo {
    VersionInfo {
        build: Build {
            build_id: BuildId {
                version: "0.0.0-test".parse().unwrap(),
                commit: [0x1c; 20],
                profile: BuildProfile::Dev,
                target: TargetTriple("x86_64-unknown-linux-gnu".into()),
                toolchain: Blake3([0x70; 32]),
            },
            features: Vec::new(),
            id: [0x5a; 32],
        },
        protocols: ProtocolMajors::current(),
        support_tier: SupportTier::Tier1,
    }
}

const BENCH_ON: DebugSection = DebugSection {
    parity_api: false,
    bench_api: true,
    telemetry_api: false,
};

/// What a run left behind: the controls the runner received, and the routing calls.
struct Ran {
    log: Vec<SequenceControl>,
    routes: usize,
}

impl Ran {
    fn opens(&self) -> Vec<&SequenceControl> {
        self.log
            .iter()
            .filter(|c| matches!(c, SequenceControl::Open(_)))
            .collect()
    }
}

/// Serves the API on `127.0.0.1` with `gates`, an ingress over a model of `vocab_limit` `limit`
/// and a fake runner playing `script`, and runs `client` against the address. Returns what the
/// runner received and what `client` returned.
fn with_api<F, Fut, T>(gates: DebugSection, limit: u32, script: Script, client: F) -> (Ran, T)
where
    F: FnOnce(SocketAddr) -> Fut,
    Fut: std::future::Future<Output = T>,
{
    let region = FakeRegion::new().unwrap();
    let model = Model {
        limit,
        routes: AtomicUsize::new(0),
    };
    let (stop_driver, stop_runner) = (AtomicBool::new(false), AtomicBool::new(false));
    let (queue, starts) = generate_queue();
    let (log, out) = std::thread::scope(|scope| {
        let runner = scope.spawn(|| {
            let mut runner = FakeRunner::new(region.runner(), script).unwrap();
            runner.serve(&stop_runner).unwrap();
            // Whatever the driver sent before it stopped.
            runner.receive().unwrap();
            runner.log().to_vec()
        });
        let (region, model, stop_driver) = (&region, &model, &stop_driver);
        let driver = scope.spawn(move || driving(region, model, &starts, stop_driver));
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let out = runtime.block_on(async {
            let api = ApiSection {
                bind: vec!["127.0.0.1:0".parse().unwrap()],
                tls: TlsSource::Auto,
                allow_plaintext_lan: false,
                local_socket: false,
            };
            let state = ApiState {
                version: version(),
                node: Arc::new(Node { gates, queue }),
            };
            let server = Server::bind(&api, None, false, state).unwrap();
            let addr = server.local_addrs()[0];
            let (stopped, on_stop) = tokio::sync::oneshot::channel::<()>();
            let task = tokio::spawn(server.serve(async {
                let _ = on_stop.await;
            }));
            let out = client(addr).await;
            let _ = stopped.send(());
            task.await.unwrap();
            out
        });
        // The driver stops first, cancelling and releasing anything still live, then the runner
        // reads every control it sent.
        stop_driver.store(true, Ordering::Release);
        driver.join().unwrap();
        stop_runner.store(true, Ordering::Release);
        (runner.join().unwrap(), out)
    });
    let ran = Ran {
        log,
        routes: model.routes.load(Ordering::Relaxed),
    };
    (ran, out)
}

fn driving(region: &FakeRegion, model: &Model, starts: &StartReceiver, stop: &AtomicBool) {
    let minter = SequenceIdMinter::new(NODE, 1);
    let clock = SimClock::new();
    let handle = clock.handle();
    let entropy = SimEntropy::new(3);
    let ingress = Ingress::new(region.daemon(), &minter, model, &handle, &entropy).unwrap();
    drive(&ingress, starts, stop);
}

/// One answer: status, content type and the body's bytes.
struct Answer {
    status: StatusCode,
    content_type: Option<String>,
    body: Bytes,
}

impl Answer {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap()
    }

    fn lines(&self) -> Vec<Value> {
        std::str::from_utf8(&self.body)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }
}

async fn post(addr: SocketAddr, path: &str, body: Vec<u8>) -> Answer {
    let stream = TcpStream::connect(addr).await.unwrap();
    let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
        .await
        .unwrap();
    tokio::spawn(connection);
    let request = Request::builder()
        .method(Method::POST)
        .uri(path)
        .header(hyper::header::HOST, "localhost")
        .body(Full::new(Bytes::from(body)))
        .unwrap();
    let response = sender.send_request(request).await.unwrap();
    let status = response.status();
    let content_type = response
        .headers()
        .get(hyper::header::CONTENT_TYPE)
        .map(|v| v.to_str().unwrap().to_owned());
    let body = response.into_body().collect().await.unwrap().to_bytes();
    Answer {
        status,
        content_type,
        body,
    }
}

fn body(value: &Value) -> Vec<u8> {
    serde_json::to_vec(value).unwrap()
}

/// Acceptance 1: a token-input request streams NDJSON events whose token count equals the
/// response's usage, with the request's `max_tokens`, `temperature` and `seed` in the
/// `SequenceOpen`. One step carries three tokens, as a verify window does.
#[test]
fn a_token_request_streams_ndjson_whose_tokens_equal_usage() {
    let script = Script::new().in_order(
        Stream::new()
            .tokens([5, 6])
            .window([7, 8, 9])
            .tokens([10])
            .finish(FinishReason::Length),
    );
    let (ran, answer) = with_api(BENCH_ON, 100, script, |addr| async move {
        let request = json!({
            "model": "m",
            "tokens": [1, 2, 3],
            "sampling": { "temperature": 0.0, "seed": 7, "max_tokens": 6 },
            "stream": true,
            "return_tokens": true,
        });
        post(addr, PATH, body(&request)).await
    });
    assert_eq!(answer.status, StatusCode::OK);
    assert_eq!(answer.content_type.as_deref(), Some("application/x-ndjson"));
    let lines = answer.lines();
    let events: Vec<&str> = lines.iter().map(|l| l["event"].as_str().unwrap()).collect();
    assert_eq!(events.first(), Some(&"started"));
    assert_eq!(events.last(), Some(&"done"));
    let sequence = lines[0]["sequence"].as_str().unwrap();
    for (n, line) in lines.iter().enumerate() {
        assert_eq!(line["id"], format!("{sequence}:{n}"));
    }
    let tokens: Vec<u64> = lines
        .iter()
        .filter(|l| l["event"] == "text")
        .flat_map(|l| {
            l["tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t.as_u64().unwrap())
        })
        .collect();
    let text: String = lines
        .iter()
        .filter(|l| l["event"] == "text")
        .map(|l| l["text"].as_str().unwrap())
        .collect();
    let done = lines.last().unwrap();
    assert_eq!(tokens, [5, 6, 7, 8, 9, 10]);
    assert_eq!(done["usage"]["completion_tokens"], tokens.len());
    assert_eq!(done["usage"]["prompt_tokens"], 3);
    assert_eq!(done["finish_reason"], "length");
    assert_eq!(text, "t5 t6 t7 t8 t9 t10 ");
    assert_eq!(lines[0]["prompt_tokens"], 3);

    let opens = ran.opens();
    let [SequenceControl::Open(open)] = opens.as_slice() else {
        panic!("{:?}", ran.log);
    };
    assert_eq!(open.prompt.as_slice(), [1, 2, 3]);
    assert_eq!(open.limits.max_tokens, 6);
    assert_eq!(open.sampling.temperature, 0.0);
    assert_eq!(open.sampling.seed, 7);
    // The sequence ended: released, with nothing cancelled.
    assert!(
        ran.log
            .iter()
            .any(|c| matches!(c, SequenceControl::Release { .. }))
    );
}

/// Without `return_tokens`, each token's text is its own `text` event, and `done` still counts
/// every token.
#[test]
fn without_return_tokens_text_events_carry_text_only() {
    let script = Script::new().in_order(Stream::new().tokens([4, 5]).finish(FinishReason::Stop));
    let (_, answer) = with_api(BENCH_ON, 100, script, |addr| async move {
        let request = json!({ "model": "m", "tokens": [1], "sampling": { "max_tokens": 8 } });
        post(addr, PATH, body(&request)).await
    });
    let lines = answer.lines();
    let texts: Vec<&Value> = lines.iter().filter(|l| l["event"] == "text").collect();
    assert_eq!(texts.len(), 2);
    assert!(texts.iter().all(|l| l.get("tokens").is_none()));
    let done = lines.last().unwrap();
    assert_eq!(done["finish_reason"], "stop");
    assert_eq!(done["usage"]["completion_tokens"], 2);
}

/// Acceptance 2: with `debug.bench_api` unset, the route answers 404 `unsupported_endpoint`, as
/// a path with no route does, whatever the other keys are, and nothing reaches the runner.
#[test]
fn the_route_is_404_while_bench_api_is_unset() {
    let others = DebugSection {
        parity_api: true,
        bench_api: false,
        telemetry_api: true,
    };
    for gates in [DebugSection::OFF, others] {
        let (ran, answers) = with_api(gates, 100, Script::new(), |addr| async move {
            let request = json!({ "model": "m", "tokens": [1], "sampling": { "max_tokens": 1 } });
            [
                post(addr, PATH, body(&request)).await,
                post(addr, "/v1/syntropy/nothing", body(&request)).await,
            ]
        });
        let (gated, unknown) = (answers[0].json(), answers[1].json());
        assert_eq!(answers[0].status, StatusCode::NOT_FOUND);
        assert_eq!(answers[1].status, StatusCode::NOT_FOUND);
        assert_eq!(gated["error"]["code"], "unsupported_endpoint");
        assert_eq!(gated["error"]["class"], unknown["error"]["class"]);
        assert_eq!(
            gated["error"]["message"],
            "no endpoint at POST /v1/syntropy/generate"
        );
        assert!(ran.log.is_empty(), "{:?}", ran.log);
        assert_eq!(ran.routes, 0);
    }
}

/// Acceptance 3: IDs at `vocab_limit`, `vocab_limit + 1` and `u32::MAX`, at the start, middle
/// and end of `tokens`, with a later bad ID too, get 400 `token_id_out_of_range` naming the field,
/// the first offending position and the limit, before routing and before any `SequenceOpen`.
#[test]
fn out_of_range_token_ids_get_400_before_any_open() {
    for limit in [1, 10, 151_936, u32::MAX - 1] {
        let (ran, (answers, cases)) = with_api(BENCH_ON, limit, Script::new(), |addr| async move {
            let (mut answers, mut cases) = (Vec::new(), Vec::new());
            let mut bad = vec![limit, limit.saturating_add(1), u32::MAX];
            bad.dedup();
            let ok = limit - 1;
            for bad in bad {
                for at in [0usize, 2, 4] {
                    let mut tokens = vec![ok, 0, ok, 0, ok, bad];
                    tokens[at] = bad;
                    let request =
                        json!({ "model": "m", "tokens": tokens, "sampling": { "max_tokens": 1 } });
                    answers.push(post(addr, PATH, body(&request)).await);
                    cases.push((bad, at));
                }
            }
            (answers, cases)
        });
        for (answer, (bad, at)) in answers.iter().zip(&cases) {
            assert_eq!(answer.status, StatusCode::BAD_REQUEST, "limit {limit}");
            let error = &answer.json()["error"];
            assert_eq!(error["code"], "token_id_out_of_range");
            assert_eq!(error["class"], "InvalidRequest");
            let message = error["message"].as_str().unwrap();
            for part in [
                &format!("tokens[{at}] is token ID {bad},"),
                &format!("vocab_limit of {limit}"),
            ] {
                assert!(message.contains(part), "{message:?} lacks {part:?}");
            }
        }
        assert_eq!(answers.len(), cases.len());
        assert!(ran.opens().is_empty(), "limit {limit}: {:?}", ran.log);
        assert_eq!(ran.routes, 0, "limit {limit}");
    }
}

/// Requests the M0 endpoint does not take are refused before routing.
#[test]
fn requests_outside_the_m0_subset_are_refused() {
    let (ran, answers) = with_api(BENCH_ON, 100, Script::new(), |addr| async move {
        let mut answers = Vec::new();
        let ok = json!({ "model": "m", "tokens": [1], "sampling": { "max_tokens": 1 } });
        // SSE, the default without `format=ndjson`, is PROD-E4-T02's.
        answers.push(post(addr, "/v1/syntropy/generate", body(&ok)).await);
        answers.push(post(addr, PATH, body(&json!({ "model": "m", "prompt": "hi" }))).await);
        answers.push(
            post(
                addr,
                PATH,
                body(&json!({ "model": "m", "tokens": [1], "n": 2 })),
            )
            .await,
        );
        answers.push(post(addr, PATH, b"{".to_vec()).await);
        answers.push(post(addr, PATH, body(&json!({ "model": "x", "tokens": [1] }))).await);
        answers.push(post(addr, PATH, vec![b' '; syntropy_api::native::MAX_BODY + 1]).await);
        answers
    });
    let got: Vec<(StatusCode, String)> = answers
        .iter()
        .map(|a| {
            (
                a.status,
                a.json()["error"]["code"].as_str().unwrap().to_owned(),
            )
        })
        .collect();
    let want = [
        (StatusCode::BAD_REQUEST, "unsupported_feature"),
        (StatusCode::BAD_REQUEST, "unsupported_feature"),
        (StatusCode::BAD_REQUEST, "invalid_request"),
        (StatusCode::BAD_REQUEST, "invalid_request"),
        (StatusCode::NOT_FOUND, "model_not_found"),
        (StatusCode::PAYLOAD_TOO_LARGE, "request_too_large"),
    ];
    let want: Vec<(StatusCode, String)> = want.iter().map(|(s, c)| (*s, (*c).to_owned())).collect();
    assert_eq!(got, want);
    assert!(ran.log.is_empty(), "{:?}", ran.log);
}

/// A client that leaves mid-stream gets its sequence cancelled and released (RFC-0050 §5.3).
#[test]
fn a_client_that_leaves_mid_stream_cancels_its_sequence() {
    // The stream never finishes, so only the client's leaving ends it.
    let script = Script::new().in_order(Stream::new().tokens([4]));
    let (ran, ()) = with_api(BENCH_ON, 100, script, |addr| async move {
        let stream = TcpStream::connect(addr).await.unwrap();
        let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
            .await
            .unwrap();
        let connection = tokio::spawn(connection);
        let request = json!({ "model": "m", "tokens": [1], "sampling": { "max_tokens": 50 } });
        let request = Request::builder()
            .method(Method::POST)
            .uri(PATH)
            .header(hyper::header::HOST, "localhost")
            .body(Full::new(Bytes::from(body(&request))))
            .unwrap();
        let response = sender.send_request(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let mut body = response.into_body();
        // The `started` line arrives, then the client goes.
        let frame = body.frame().await.unwrap().unwrap();
        assert!(frame.data_ref().is_some());
        drop(body);
        drop(sender);
        connection.abort();
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    });
    assert!(
        ran.log
            .iter()
            .any(|c| matches!(c, SequenceControl::Cancel { .. })),
        "{:?}",
        ran.log
    );
    assert!(
        ran.log
            .iter()
            .any(|c| matches!(c, SequenceControl::Release { .. }))
    );
}
