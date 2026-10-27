// SPDX-License-Identifier: MIT OR Apache-2.0

//! DIST-E1-T14's criteria for the types themselves: a serde round trip of every type; a
//! `SequenceMeta` built from a `SequenceOpen` keeps every field but the prompt tokens, and records
//! their count; a `SequenceOpen` encoded without an addition decodes with the addition's default.
//! Then DIST-E8-T05's additions: `SequenceControl::Reconcile`, and the prompt's two serde forms,
//! an array in JSON and a byte string of little-endian `u32`s in a binary format.

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use smallvec::smallvec;

use super::*;
use crate::{
    ApproxKnobs, Blake3, ExactnessPolicy, ForceMode, InstanceId, LogitTap, Nanos, NodeId,
    ParityTapSpec, PlanEpoch, SequenceId, SessionId, TapMode, TapRow, TapValues, TeacherStreamId,
};

fn round_trip<T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug>(value: &T) {
    let json = serde_json::to_string(value).unwrap();
    let back: T = serde_json::from_str(&json).unwrap();
    assert_eq!(&back, value, "{json}");
}

/// The same through postcard, the IPC's encoding (RFC-0001 §3.5), which is binary and not
/// self-describing, so the prompt takes its byte-string form and no field may be skipped.
fn binary_round_trip<T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug>(value: &T) {
    let bytes = postcard::to_allocvec(value).unwrap();
    let back: T = postcard::from_bytes(&bytes).unwrap();
    assert_eq!(&back, value);
}

fn seq() -> SequenceId {
    SequenceId {
        ingress: 0x0102_0304_0506_0708,
        boot: 3,
        n: 42,
    }
}

fn trace() -> TraceContext {
    TraceContext {
        version: 0,
        trace_id: [0x11; 16],
        span_id: [0x22; 8],
        flags: TraceContext::SAMPLED | TraceContext::FULL_TRACE,
    }
}

fn parity() -> ParityTapSpec {
    ParityTapSpec {
        stream: TeacherStreamId(7),
        mode: ForceMode::Decode { prefill_len: 16 },
        tap: TapMode::Gather { k: 8, tail: 2 },
    }
}

/// A `SequenceOpen` with every field set to a value other than its default.
fn full_open() -> SequenceOpen {
    SequenceOpen {
        seq: seq(),
        session: Some(SessionId([0x33; 16])),
        ingress: NodeId([0x44; 32]),
        prompt: TokenSlice(vec![1, 2, 3, 4, 5]),
        prefix: Some(PrefixRef::Cached {
            last_full: PrefixKey(Blake3([0x55; 32])),
            tokens: 128,
            tail: Some(TailId(3)),
            epoch: PlanEpoch::default(),
        }),
        sampling: SamplingParams {
            seed: 0xDEAD_BEEF,
            temperature: 0.7,
            top_p: 0.95,
            top_k: 40,
            min_p: 0.05,
            presence_penalty: 0.5,
            frequency_penalty: 0.25,
            repetition_penalty: 1.1,
            logit_bias: smallvec![(17, -1.5), (23, 2.0)],
            logprobs: Some(5),
            constraint: Some(GrammarRef(Blake3([0x66; 32]))),
            grammar_inline: Some(vec![0xAB; 9]),
            reasoning_budget: Some(512),
        },
        limits: SequenceLimits {
            max_tokens: 256,
            stop_tokens: smallvec![2, 32_000],
            deadline: Some(Nanos(30_000_000_000)),
            embed: Some(PoolingMode::Mean),
        },
        exactness: ExactnessPolicy::Approximate {
            knobs: ApproxKnobs::all(),
            params: Default::default(),
        },
        priority: Priority::Interactive,
        choice_index: Some(1),
        prefill_chunk: 512,
        initial_window: Some(4),
        resume: Some(ResumePoint {
            emitted: 3,
            rng_counter: 1 << 40,
            cause: ResumeCause::Recovery,
        }),
        prefix_scope: PrefixScope {
            principal: Some([0x77; 16]),
            shared_until: Some(64),
        },
        session_pk: Some([0x88; 32]),
        trace: Some(trace()),
        parity: Some(parity()),
    }
}

fn full_entry() -> TokenEntry {
    TokenEntry {
        seq: seq(),
        tokens: smallvec![10, 11, 12],
        first_position: 99,
        logprobs: Some(LogprobBlock {
            k: 1,
            chosen: vec![-0.5, f32::NEG_INFINITY, -2.0],
            top_ids: vec![7, 8, 9],
            top_logprobs: vec![-1.0, -3.0, f32::NAN],
        }),
        finish: Some(FinishReason::Error {
            domain: 3,
            code: 0x0301,
        }),
        approx: ApproxKnobs::all(),
        draft: smallvec![20, 21],
        draft_tree: Some(smallvec![
            DraftNode {
                parent: None,
                depth: 1,
                flags: DraftNode::PROMPT_LOOKUP,
            },
            DraftNode {
                parent: Some(0),
                depth: 2,
                flags: 0,
            },
        ]),
        rng_counter: u64::MAX - 1,
        tap: Some(LogitTap {
            k: 0,
            tail: 0,
            rows: vec![TapRow {
                pos: 4,
                argmax: 9,
                margin: 0.25,
                lse: 12.5,
                lse_complement: 0.0,
                values: TapValues::FullLocal { shm_offset: 4096 },
            }],
        }),
    }
}

#[test]
fn every_type_round_trips_through_serde() {
    let open = full_open();
    round_trip(&open);
    round_trip(&open.meta());
    round_trip(&SequenceControl::Open(open.clone()));
    for reason in [
        CancelReason::ClientGone,
        CancelReason::ClientRequest,
        CancelReason::StopString,
        CancelReason::Deadline,
        CancelReason::Error { domain: 4, code: 9 },
    ] {
        round_trip(&SequenceControl::Cancel { seq: seq(), reason });
    }
    round_trip(&SequenceControl::Pause { seq: seq() });
    round_trip(&SequenceControl::Resume { seq: seq() });
    round_trip(&SequenceControl::Release {
        seq: seq(),
        persist: Some(SessionId([9; 16])),
    });
    for recent in [
        smallvec![],
        smallvec![7, 8, 9],
        smallvec![1, 2, 3, 4, 5, 6, 7, 8],
    ] {
        round_trip(&SequenceControl::Reconcile {
            seq: seq(),
            emitted: 3,
            rng_counter: 1 << 40,
            recent,
        });
    }
    round_trip(&PrefixRef::Session {
        session: SessionId([5; 16]),
        tokens: 77,
        blob: Blake3([6; 32]),
    });
    for cause in [
        ResumeCause::Migration,
        ResumeCause::Recovery,
        ResumeCause::Rebalance,
    ] {
        round_trip(&ResumePoint {
            emitted: 1,
            rng_counter: 2,
            cause,
        });
    }
    for mode in [PoolingMode::LastToken, PoolingMode::Mean, PoolingMode::Cls] {
        round_trip(&mode);
    }
    round_trip(&SamplingParams::default());
    round_trip(&SequenceLimits::default());
    round_trip(&PrefixScope::default());
    round_trip(&trace());

    let entry = full_entry();
    // NaN never equals itself, so the logprobs are checked by bit pattern below.
    let mut comparable = entry.clone();
    comparable.logprobs = None;
    round_trip(&comparable);
    for finish in [
        FinishReason::Stop,
        FinishReason::Length,
        FinishReason::TokenStop,
        FinishReason::Cancelled,
    ] {
        round_trip(&TokenEntry {
            finish: Some(finish),
            ..comparable.clone()
        });
    }
    let frame = TokenFrame {
        instance: InstanceId(5),
        epoch: PlanEpoch::default(),
        step: 1 << 33,
        entries: smallvec![comparable.clone(), comparable],
    };
    round_trip(&frame);

    // Non-finite logprobs keep their bit patterns, as a parity tap's floats do: in JSON as the
    // strings "nan" and "-inf".
    let json = serde_json::to_string(&entry).unwrap();
    assert!(
        json.contains("\"-inf\"") && json.contains("\"nan\""),
        "{json}"
    );
    let back: TokenEntry = serde_json::from_str(&json).unwrap();
    let (sent, got) = (
        entry.logprobs.as_ref().unwrap(),
        back.logprobs.as_ref().unwrap(),
    );
    assert_eq!(got.k, sent.k);
    assert_eq!(got.top_ids, sent.top_ids);
    for (a, b) in sent
        .chosen
        .iter()
        .chain(&sent.top_logprobs)
        .zip(got.chosen.iter().chain(&got.top_logprobs))
    {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert!(sent.is_consistent());
}

/// `SequenceMeta` is the open without its prompt: compared as JSON objects, the two agree on
/// every field once the prompt and the prompt length are taken out, so a field added to one and
/// not the other fails here.
#[test]
fn sequence_meta_keeps_every_field_but_the_prompt_tokens() {
    let open = full_open();
    let meta = open.meta();
    assert_eq!(meta.prompt_len, 5);
    let Value::Object(mut open_fields) = serde_json::to_value(&open).unwrap() else {
        unreachable!()
    };
    let Value::Object(mut meta_fields) = serde_json::to_value(&meta).unwrap() else {
        unreachable!()
    };
    assert_eq!(
        open_fields.remove("prompt"),
        Some(serde_json::json!([1, 2, 3, 4, 5]))
    );
    assert_eq!(meta_fields.remove("prompt_len"), Some(serde_json::json!(5)));
    assert_eq!(open_fields, meta_fields);
}

/// An open encoded before an addition existed, without that key, decodes with the addition's
/// default, at the top level and inside `sampling` and `limits`.
#[test]
fn an_open_without_an_addition_decodes_with_its_default() {
    let full = serde_json::to_value(full_open()).unwrap();
    let defaults = SequenceOpen {
        session: None,
        prefix: None,
        exactness: ExactnessPolicy::default(),
        priority: Priority::default(),
        choice_index: None,
        prefill_chunk: 0,
        initial_window: None,
        resume: None,
        prefix_scope: PrefixScope::default(),
        session_pk: None,
        trace: None,
        parity: None,
        ..full_open()
    };
    let top = [
        "session",
        "prefix",
        "exactness",
        "priority",
        "choice_index",
        "prefill_chunk",
        "initial_window",
        "resume",
        "prefix_scope",
        "session_pk",
        "trace",
        "parity",
    ];
    for key in top {
        let mut older = full.clone();
        older.as_object_mut().unwrap().remove(key).unwrap();
        let decoded: SequenceOpen = serde_json::from_value(older).unwrap();
        let back = serde_json::to_value(&decoded).unwrap();
        assert_eq!(
            back[key],
            serde_json::to_value(&defaults).unwrap()[key],
            "{key}"
        );
        let mut rest = back.clone();
        rest.as_object_mut().unwrap().remove(key);
        let mut expected = full.clone();
        expected.as_object_mut().unwrap().remove(key);
        assert_eq!(rest, expected, "{key}: another field changed");
    }
    let nested = [
        ("sampling", "logit_bias"),
        ("sampling", "logprobs"),
        ("sampling", "constraint"),
        ("sampling", "grammar_inline"),
        ("sampling", "reasoning_budget"),
        ("limits", "stop_tokens"),
        ("limits", "deadline"),
        ("limits", "embed"),
    ];
    let default_sampling = serde_json::to_value(SamplingParams::default()).unwrap();
    let default_limits = serde_json::to_value(SequenceLimits::default()).unwrap();
    for (outer, key) in nested {
        let mut older = full.clone();
        older[outer].as_object_mut().unwrap().remove(key).unwrap();
        let decoded: SequenceOpen = serde_json::from_value(older).unwrap();
        let back = serde_json::to_value(&decoded).unwrap();
        let default = if outer == "sampling" {
            &default_sampling
        } else {
            &default_limits
        };
        assert_eq!(back[outer][key], default[key], "{outer}.{key}");
    }
    // The same holds for the metadata, which travels downstream on the first frame.
    let meta = serde_json::to_value(full_open().meta()).unwrap();
    for key in top {
        let mut older = meta.clone();
        older.as_object_mut().unwrap().remove(key).unwrap();
        let decoded: SequenceMeta = serde_json::from_value(older).unwrap();
        assert_eq!(
            serde_json::to_value(&decoded).unwrap()[key],
            serde_json::to_value(defaults.meta()).unwrap()[key],
            "meta {key}"
        );
    }
}

#[test]
fn a_trace_context_reads_its_two_bits() {
    let mut context = trace();
    assert!(context.sampled() && context.full_trace());
    context.flags = TraceContext::SAMPLED;
    assert!(context.sampled() && !context.full_trace());
    context.flags = 0;
    assert!(!context.sampled() && !context.full_trace());
}

#[test]
fn a_logprob_block_checks_its_lengths() {
    let mut block = LogprobBlock {
        k: 2,
        chosen: vec![-1.0, -2.0],
        top_ids: vec![1, 2, 3, 4],
        top_logprobs: vec![-1.0, -1.5, -2.0, -2.5],
    };
    assert!(block.is_consistent());
    block.top_ids.pop();
    assert!(!block.is_consistent());
    // `k = 0` is the chosen tokens' logprobs alone, which the in-process path carries.
    let chosen_only = LogprobBlock {
        k: 0,
        chosen: vec![-0.1],
        top_ids: vec![],
        top_logprobs: vec![],
    };
    assert!(chosen_only.is_consistent());
    round_trip(&chosen_only);
    let mut stray = chosen_only.clone();
    stray.top_ids.push(3);
    stray.top_logprobs.push(-2.0);
    assert!(!stray.is_consistent());
}

#[test]
fn a_token_slice_reads_as_its_tokens() {
    let prompt = TokenSlice::from(vec![5, 6, 7]);
    assert_eq!(prompt.len(), 3);
    assert_eq!(prompt.as_slice(), &[5, 6, 7]);
    assert_eq!(serde_json::to_string(&prompt).unwrap(), "[5,6,7]");
}

/// The open, its metadata and every control variant round-trip through postcard too: the IPC
/// carries them in M0 (RFC-0001 §3.5), and there the prompt is a byte string.
#[test]
fn every_control_round_trips_through_a_binary_format() {
    let open = full_open();
    binary_round_trip(&open);
    binary_round_trip(&open.meta());
    binary_round_trip(&SequenceControl::Open(open));
    for control in [
        SequenceControl::Cancel {
            seq: seq(),
            reason: CancelReason::Error { domain: 4, code: 9 },
        },
        SequenceControl::Pause { seq: seq() },
        SequenceControl::Resume { seq: seq() },
        SequenceControl::Release {
            seq: seq(),
            persist: None,
        },
        SequenceControl::Reconcile {
            seq: seq(),
            emitted: 0,
            rng_counter: u64::MAX,
            recent: smallvec![u32::MAX, 0],
        },
    ] {
        binary_round_trip(&control);
    }
}

/// A binary format gets the prompt as a byte string of little-endian `u32`s, 4 bytes a token
/// whatever the IDs (RFC-0031 §6.6), and JSON keeps the array. The canonical encoding is binary,
/// so a digest over a prompt would cover those bytes; no digest covers one today.
#[test]
fn a_token_slice_is_a_byte_string_in_a_binary_format() {
    let prompt = TokenSlice::from(vec![1, 0x0102_0304, u32::MAX]);
    let mut expected = vec![0x08];
    expected.extend_from_slice(&12u64.to_le_bytes());
    expected.extend_from_slice(&[1, 0, 0, 0, 4, 3, 2, 1, 0xff, 0xff, 0xff, 0xff]);
    assert_eq!(crate::canon::to_bytes(&prompt).unwrap(), expected);

    // postcard writes a byte string as its varint length and the bytes.
    let bytes = postcard::to_allocvec(&prompt).unwrap();
    assert_eq!(bytes, [&[12][..], &expected[9..]].concat());
    assert_eq!(postcard::from_bytes::<TokenSlice>(&bytes).unwrap(), prompt);
    assert_eq!(
        serde_json::to_string(&prompt).unwrap(),
        "[1,16909060,4294967295]"
    );

    let empty = TokenSlice::default();
    assert_eq!(postcard::to_allocvec(&empty).unwrap(), [0]);
    assert_eq!(postcard::from_bytes::<TokenSlice>(&[0]).unwrap(), empty);
}

/// A byte string that is not a whole number of token IDs is refused, not truncated.
#[test]
fn a_prompt_byte_string_of_a_partial_token_is_refused() {
    for len in [1u8, 2, 3, 5, 7] {
        let mut bytes = vec![len];
        bytes.extend(std::iter::repeat_n(0xaa, usize::from(len)));
        assert!(
            postcard::from_bytes::<TokenSlice>(&bytes).is_err(),
            "{len} bytes"
        );
    }
    let mut whole = vec![8];
    whole.extend([1, 0, 0, 0, 2, 0, 0, 0]);
    assert_eq!(
        postcard::from_bytes::<TokenSlice>(&whole).unwrap(),
        TokenSlice::from(vec![1, 2])
    );
}
