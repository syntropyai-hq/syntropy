// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `SeqControl` codec (DIST-E8-T05; RFC-0031 §6.6, RFC-0032 §3.6) against golden frames in
//! `tests/golden/seq_control/` and its three criteria:
//! - a 128K-token prompt fits the head's sequence slot;
//! - unknown CBOR fields are ignored by older decoders, in both directions between versions;
//! - `choice_index` 0 and 1 survive encoding a `SequenceOpen` and the `SequenceMeta` built from
//!   it.
//!
//! The goldens are the small bodies, written from RFC-0031 §6.6's table and RFC-0033 §11's
//! fields: the header and extension by [`spec`], and the CBOR body by hand as a
//! `ciborium::Value`, field by field, so they do not come from the codec's own encoder.
//! `SYNTROPY_BLESS_GOLDEN=1` rewrites the files, as it does the other goldens.

use std::collections::BTreeSet;
use std::path::PathBuf;

use ciborium::Value;
use proptest::prelude::*;
use serde::Deserialize;
use syntropy_core::{
    ApproxKnobs, Blake3, CancelReason, ErrorClass, ExactnessPolicy, ForceMode, GrammarRef,
    InstanceId, Nanos, NodeId, ParityTapSpec, PlanEpoch, PoolingMode, PrefixKey, PrefixRef,
    PrefixScope, Priority, ResumeCause, ResumePoint, SamplingParams, SequenceControl, SequenceId,
    SequenceLimits, SequenceMeta, SequenceOpen, SessionId, TailId, TapMode, TeacherStreamId,
    TokenSlice, TraceContext,
};
use syntropy_net::ctl::envelope::{self, BodyLimit, DecodeError};
use syntropy_net::frame::{
    Flags, Frame, FrameError, FrameSpec, FrameType, INLINE_ALLOWANCE, MAX_RECENT, OPEN_ALLOWANCE,
    SMALL_BODY, SeqControlError, SeqControlKind, SeqControlLimits, SeqControlSpec, WIRE_MINOR,
    code, decode_seq_meta, encode_seq_control, encode_seq_meta, encode_to_vec, open_body_limit,
    sequence_slot_size,
};

use super::{INSTANCE, REFUSAL_CLASSES, base, epoch, golden_dir, spec};

/// The context of the small instance most tests use: a 4K-token slot.
const SMALL_CONTEXT: u32 = 4096;

/// 128K tokens: criterion 1's prompt.
const LONG_CONTEXT: u32 = 128 << 10;

fn seq(n: u32) -> SequenceId {
    SequenceId {
        ingress: 0x0102_0304_0506_0708,
        boot: 3,
        n,
    }
}

fn spec_for(chan_seq: u64) -> SeqControlSpec {
    SeqControlSpec {
        wire_minor: WIRE_MINOR,
        flags: Flags::default(),
        instance: InstanceId(INSTANCE),
        epoch: epoch(),
        chan_seq,
    }
}

fn small_slot() -> usize {
    sequence_slot_size(SMALL_CONTEXT)
}

/// An open with every field at its default but the ones RFC-0001 §5.8 requires.
fn plain_open(n: u32, prompt: Vec<u32>) -> SequenceOpen {
    SequenceOpen {
        seq: seq(n),
        session: None,
        ingress: NodeId([0x44; 32]),
        prompt: TokenSlice(prompt),
        prefix: None,
        sampling: SamplingParams::default(),
        limits: SequenceLimits {
            max_tokens: 64,
            ..SequenceLimits::default()
        },
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
    }
}

/// An open with every field set to a value other than its default, `resume` included, so it
/// travels as a `ResumeOpen`.
fn full_open() -> SequenceOpen {
    SequenceOpen {
        seq: seq(42),
        session: Some(SessionId([0x33; 16])),
        ingress: NodeId([0x44; 32]),
        prompt: TokenSlice(vec![1, 2, 3, 70_000, u32::MAX]),
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
            logit_bias: vec![(17, -1.5), (23, 2.0)].into(),
            logprobs: Some(5),
            constraint: Some(GrammarRef(Blake3([0x66; 32]))),
            grammar_inline: Some(vec![0xAB; 9]),
            reasoning_budget: Some(512),
        },
        limits: SequenceLimits {
            max_tokens: 256,
            stop_tokens: vec![2, 32_000].into(),
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
        trace: Some(TraceContext {
            version: 0,
            trace_id: [0x11; 16],
            span_id: [0x22; 8],
            flags: TraceContext::SAMPLED,
        }),
        parity: Some(ParityTapSpec {
            stream: TeacherStreamId(7),
            mode: ForceMode::Decode { prefill_len: 16 },
            tap: TapMode::Gather { k: 8, tail: 2 },
        }),
    }
}

/// `control` encoded for a slot of `slot_size` bytes, then decoded with the same slot.
fn round_trip(control: &SequenceControl, slot_size: usize) -> SequenceControl {
    let bytes = encode_seq_control(control, &spec_for(1), slot_size).unwrap();
    let frame = Frame::decode(&bytes).unwrap();
    assert_eq!(frame.header.spec.ftype, FrameType::SeqControl);
    frame
        .seq_control(SeqControlLimits::slot(slot_size))
        .unwrap()
}

/// The extension's `kind` and `body_enc` bytes of an encoded frame.
fn ext_bytes(bytes: &[u8]) -> (u8, u8) {
    (bytes[64], bytes[65])
}

/// A `SeqControl` frame with the given extension bytes and payload, built from the tables.
fn raw_frame(kind: u8, body_enc: u8, payload: &[u8]) -> Vec<u8> {
    let mut ext = spec::ext(8);
    ext[0] = kind;
    ext[1] = body_enc;
    spec::frame(&base(0x03, 0, 0, 1), &ext, payload)
}

/// `value` as CBOR bytes.
fn cbor(value: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    ciborium::into_writer(value, &mut out).unwrap();
    out
}

fn text(s: &str) -> Value {
    Value::Text(s.to_owned())
}

fn int(n: u64) -> Value {
    Value::Integer(n.into())
}

fn map(entries: Vec<(&str, Value)>) -> Value {
    Value::Map(entries.into_iter().map(|(k, v)| (text(k), v)).collect())
}

/// A `SequenceId` as CBOR: its three fields by name.
fn seq_value(seq: SequenceId) -> Value {
    map(vec![
        ("ingress", int(seq.ingress)),
        ("boot", int(seq.boot.into())),
        ("n", int(seq.n.into())),
    ])
}

/// One golden frame: the bytes from the tables and the message they hold.
struct Golden {
    name: &'static str,
    about: &'static str,
    bytes: Vec<u8>,
    control: SequenceControl,
}

fn goldens() -> Vec<Golden> {
    let s = seq(7);
    let golden = |name, about, kind: u8, body: Value, control| Golden {
        name,
        about,
        bytes: raw_frame(kind, 0, &cbor(&body)),
        control,
    };
    vec![
        golden(
            "seq-control-cancel",
            "Cancel, kind 2: { seq, reason: ClientGone }",
            2,
            map(vec![("seq", seq_value(s)), ("reason", text("ClientGone"))]),
            SequenceControl::Cancel {
                seq: s,
                reason: CancelReason::ClientGone,
            },
        ),
        golden(
            "seq-control-cancel-error",
            "Cancel, kind 2: { seq, reason: Error { domain 4, code 9 } }",
            2,
            map(vec![
                ("seq", seq_value(s)),
                (
                    "reason",
                    map(vec![(
                        "Error",
                        map(vec![("domain", int(4)), ("code", int(9))]),
                    )]),
                ),
            ]),
            SequenceControl::Cancel {
                seq: s,
                reason: CancelReason::Error { domain: 4, code: 9 },
            },
        ),
        golden(
            "seq-control-pause",
            "Pause, kind 3: { seq }",
            3,
            map(vec![("seq", seq_value(s))]),
            SequenceControl::Pause { seq: s },
        ),
        golden(
            "seq-control-resume",
            "Resume, kind 4: { seq }",
            4,
            map(vec![("seq", seq_value(s))]),
            SequenceControl::Resume { seq: s },
        ),
        golden(
            "seq-control-release-persist",
            "Release, kind 5: { seq, persist: a 16-byte SessionId }",
            5,
            map(vec![
                ("seq", seq_value(s)),
                (
                    "persist",
                    Value::Array((0..16).map(|_| int(0x5a)).collect()),
                ),
            ]),
            SequenceControl::Release {
                seq: s,
                persist: Some(SessionId([0x5a; 16])),
            },
        ),
        golden(
            "seq-control-reconcile",
            "Reconcile, kind 7: { seq, emitted 12, rng_counter 2^40 + 5, recent [9, 10, 70000] }",
            7,
            map(vec![
                ("seq", seq_value(s)),
                ("emitted", int(12)),
                ("rng_counter", int((1 << 40) + 5)),
                ("recent", Value::Array(vec![int(9), int(10), int(70_000)])),
            ]),
            SequenceControl::Reconcile {
                seq: s,
                emitted: 12,
                rng_counter: (1 << 40) + 5,
                recent: vec![9, 10, 70_000].into(),
            },
        ),
    ]
}

fn seq_control_dir() -> PathBuf {
    golden_dir().join("seq_control")
}

fn read_golden(name: &str) -> Vec<u8> {
    let path = seq_control_dir().join(format!("{name}.hex"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    spec::from_hex(&text)
}

#[test]
fn seq_control_golden_files_match_the_spec_tables() {
    let goldens = goldens();
    if std::env::var_os("SYNTROPY_BLESS_GOLDEN").is_some() {
        std::fs::create_dir_all(seq_control_dir()).unwrap();
        for golden in &goldens {
            let text = spec::to_hex(golden.name, golden.about, &golden.bytes);
            std::fs::write(seq_control_dir().join(format!("{}.hex", golden.name)), text).unwrap();
        }
    }
    for golden in &goldens {
        assert_eq!(
            read_golden(golden.name),
            golden.bytes,
            "{} differs from the tables",
            golden.name
        );
    }
    let listed: BTreeSet<String> = goldens.iter().map(|g| format!("{}.hex", g.name)).collect();
    let on_disk: BTreeSet<String> = std::fs::read_dir(seq_control_dir())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.ends_with(".hex"))
        .collect();
    assert_eq!(on_disk, listed);
}

/// Each golden decodes to its message, and the codec writes that message as the same bytes.
#[test]
fn seq_control_goldens_round_trip() {
    for golden in goldens() {
        let name = golden.name;
        let bytes = read_golden(name);
        let frame = Frame::decode(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        let decoded = frame
            .seq_control(SeqControlLimits::slot(small_slot()))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(decoded, golden.control, "{name}");
        assert_eq!(
            encode_seq_control(&golden.control, &spec_for(1), small_slot()),
            Ok(bytes),
            "{name}"
        );
    }
}

/// Every variant round-trips, each under its kind, with an uncompressed body.
#[test]
fn every_variant_round_trips_under_its_kind() {
    let resumed = full_open();
    let mut fresh = full_open();
    fresh.resume = None;
    let mut controls = vec![
        (SequenceControl::Open(fresh), SeqControlKind::Open),
        (SequenceControl::Open(resumed), SeqControlKind::ResumeOpen),
        (
            SequenceControl::Open(plain_open(1, vec![])),
            SeqControlKind::Open,
        ),
        (
            SequenceControl::Pause { seq: seq(1) },
            SeqControlKind::Pause,
        ),
        (
            SequenceControl::Resume { seq: seq(1) },
            SeqControlKind::Resume,
        ),
        (
            SequenceControl::Release {
                seq: seq(1),
                persist: None,
            },
            SeqControlKind::Release,
        ),
        (
            SequenceControl::Reconcile {
                seq: seq(1),
                emitted: 0,
                rng_counter: u64::MAX,
                recent: vec![].into(),
            },
            SeqControlKind::Reconcile,
        ),
        (
            SequenceControl::Reconcile {
                seq: seq(1),
                emitted: 8,
                rng_counter: 3,
                recent: (1..=8).collect::<Vec<u32>>().into(),
            },
            SeqControlKind::Reconcile,
        ),
    ];
    for reason in [
        CancelReason::ClientGone,
        CancelReason::ClientRequest,
        CancelReason::StopString,
        CancelReason::Deadline,
        CancelReason::Error {
            domain: 1,
            code: u16::MAX,
        },
    ] {
        controls.push((
            SequenceControl::Cancel {
                seq: seq(1),
                reason,
            },
            SeqControlKind::Cancel,
        ));
    }
    for (control, kind) in controls {
        let bytes = encode_seq_control(&control, &spec_for(9), small_slot()).unwrap();
        assert_eq!(ext_bytes(&bytes), (kind.code(), 0), "{control:?}");
        assert_eq!(SeqControlKind::of(&control), Some(kind));
        let frame = Frame::decode(&bytes).unwrap();
        assert_eq!(frame.seq_control_kind().unwrap().0, kind);
        assert_eq!(frame.header.spec.chan_seq, 9);
        assert_eq!(frame.header.spec.step, 0);
        assert_eq!(
            frame.seq_control(SeqControlLimits::slot(small_slot())),
            Ok(control)
        );
    }
}

/// Criterion 1: an `Open` whose prompt is 128K tokens, with every other field set, fits the
/// slot [`sequence_slot_size`] gives an instance of that context, uncompressed. The prompt takes
/// 4 bytes a token whatever its IDs, so the slot is within the IPC ring's 1 MiB slot limit too,
/// for the `local_sequences` ring that carries the same frames (RFC-0031 §4).
#[test]
fn a_128k_token_prompt_fits_the_heads_sequence_slot() {
    let tokens = LONG_CONTEXT as usize;
    // IDs across a 2^18-token vocabulary, most of them above 2^16, where a CBOR integer takes
    // 5 bytes.
    let prompt: Vec<u32> = (0..tokens as u32)
        .map(|i| i.wrapping_mul(2_654_435_761) % (1 << 18))
        .collect();
    let mut open = full_open();
    open.resume = None;
    open.prompt = TokenSlice(prompt);
    let control = SequenceControl::Open(open);

    let slot = sequence_slot_size(LONG_CONTEXT);
    assert_eq!(
        slot,
        (64 + 64 + 4 * tokens + OPEN_ALLOWANCE).next_multiple_of(64)
    );
    assert!(
        slot <= 1 << 20,
        "a {slot}-byte slot is over the IPC ring's 1 MiB"
    );

    let bytes = encode_seq_control(&control, &spec_for(1), slot).unwrap();
    assert_eq!(
        ext_bytes(&bytes),
        (SeqControlKind::Open.code(), 0),
        "plain CBOR"
    );
    assert!(
        bytes.len() <= slot,
        "{} bytes in a {slot}-byte slot",
        bytes.len()
    );
    let payload = bytes.len() - 128;
    assert!(
        payload >= 4 * tokens && payload - 4 * tokens < OPEN_ALLOWANCE,
        "a {payload}-byte body for {tokens} tokens"
    );
    let frame = Frame::decode(&bytes).unwrap();
    assert_eq!(frame.seq_control(SeqControlLimits::slot(slot)), Ok(control));
}

/// RFC-0031 §6.6: the prompt in `Open` is a CBOR byte string of little-endian `u32` token IDs.
#[test]
fn the_prompt_is_a_byte_string_of_little_endian_u32s() {
    let open = plain_open(1, vec![1, 0x0102_0304, u32::MAX]);
    let bytes =
        encode_seq_control(&SequenceControl::Open(open), &spec_for(1), small_slot()).unwrap();
    let body: Value = ciborium::from_reader(&bytes[128..]).unwrap();
    let prompt = body
        .as_map()
        .unwrap()
        .iter()
        .find(|(key, _)| key.as_text() == Some("prompt"))
        .map(|(_, value)| value.clone());
    assert_eq!(
        prompt,
        Some(Value::Bytes(vec![
            1, 0, 0, 0, 4, 3, 2, 1, 0xff, 0xff, 0xff, 0xff
        ]))
    );
}

/// `value`'s map with `extra` entries added, recursively at `path` (a list of keys).
fn with_unknown(value: &Value, path: &[&str], extra: &[(&str, Value)]) -> Value {
    let Value::Map(entries) = value else {
        panic!("not a map at {path:?}");
    };
    let mut entries = entries.clone();
    match path.split_first() {
        None => {
            for (key, value) in extra {
                entries.push((text(key), value.clone()));
            }
        }
        Some((key, rest)) => {
            let at = entries
                .iter()
                .position(|(k, _)| k.as_text() == Some(*key))
                .unwrap_or_else(|| panic!("no {key}"));
            entries[at].1 = with_unknown(&entries[at].1, rest, extra);
        }
    }
    Value::Map(entries)
}

/// Fields a later version might add: a scalar, a nested map and an array.
fn future_fields() -> Vec<(&'static str, Value)> {
    vec![
        ("future_scalar", int(7)),
        (
            "future_map",
            map(vec![("a", Value::Bool(true)), ("b", Value::Null)]),
        ),
        ("future_list", Value::Array(vec![text("x"), int(1)])),
    ]
}

/// Criterion 2, newer to older: this build reads a body from a later version that added
/// fields, at the top level and inside nested structs, and ignores them.
#[test]
fn unknown_fields_from_a_newer_sender_are_ignored() {
    let open = full_open();
    let current: Value =
        ciborium::from_reader(envelope::to_cbor(&open).unwrap().as_slice()).unwrap();
    let mut newer = with_unknown(&current, &[], &future_fields());
    for path in [
        &["sampling"][..],
        &["limits"],
        &["resume"],
        &["prefix_scope"],
    ] {
        newer = with_unknown(&newer, path, &future_fields());
    }
    let frame = raw_frame(SeqControlKind::ResumeOpen.code(), 0, &cbor(&newer));
    assert_eq!(
        Frame::decode(&frame)
            .unwrap()
            .seq_control(SeqControlLimits::slot(small_slot())),
        Ok(SequenceControl::Open(open.clone()))
    );

    let reconcile = map(vec![
        ("seq", seq_value(seq(2))),
        ("emitted", int(4)),
        ("rng_counter", int(99)),
        ("recent", Value::Array(vec![int(1), int(2)])),
    ]);
    let frame = raw_frame(
        7,
        0,
        &cbor(&with_unknown(&reconcile, &[], &future_fields())),
    );
    assert_eq!(
        Frame::decode(&frame)
            .unwrap()
            .seq_control(SeqControlLimits::slot(small_slot())),
        Ok(SequenceControl::Reconcile {
            seq: seq(2),
            emitted: 4,
            rng_counter: 99,
            recent: vec![1, 2].into(),
        })
    );

    let meta = open.meta();
    let current: Value = ciborium::from_reader(encode_seq_meta(&meta).unwrap().as_slice()).unwrap();
    let newer = with_unknown(&current, &[], &future_fields());
    assert_eq!(decode_seq_meta(&cbor(&newer)), Ok(meta));
}

/// RFC-0001 §5.8's `SequenceOpen`, before RFC-0032 §3.6 and the amendments added fields: what
/// an older build's decoder knows.
#[derive(Debug, PartialEq, Deserialize)]
struct OlderOpen {
    seq: SequenceId,
    #[serde(default)]
    session: Option<SessionId>,
    ingress: NodeId,
    prompt: TokenSlice,
    #[serde(default)]
    prefix: Option<PrefixRef>,
    sampling: OlderSampling,
    limits: OlderLimits,
    #[serde(default)]
    exactness: ExactnessPolicy,
    #[serde(default)]
    priority: Priority,
}

/// `SamplingParams` before amendment A3 and `logprobs`.
#[derive(Debug, PartialEq, Deserialize)]
struct OlderSampling {
    seed: u64,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    min_p: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    repetition_penalty: f32,
    #[serde(default)]
    logit_bias: Vec<(u32, f32)>,
}

/// `SequenceLimits` before amendment A5's `embed`.
#[derive(Debug, PartialEq, Deserialize)]
struct OlderLimits {
    max_tokens: u32,
    #[serde(default)]
    stop_tokens: Vec<u32>,
    #[serde(default)]
    deadline: Option<Nanos>,
}

/// `SequenceMeta` as an older build knows it: without the router's choices, `choice_index`,
/// `session_pk`, `trace` and `parity`.
#[derive(Debug, PartialEq, Deserialize)]
struct OlderMeta {
    seq: SequenceId,
    ingress: NodeId,
    prompt_len: u32,
    sampling: OlderSampling,
    limits: OlderLimits,
    #[serde(default)]
    resume: Option<ResumePoint>,
    #[serde(default)]
    prefix: Option<PrefixRef>,
}

/// Criterion 2, older reading newer: a decoder that predates RFC-0032 §3.6's additions reads
/// this build's `Open` body and `SEQ_META`, skipping every field it does not know.
#[test]
fn an_older_decoder_ignores_the_fields_this_build_added() {
    let open = full_open();
    let body = envelope::to_cbor(&open).unwrap();
    let older: OlderOpen = envelope::from_cbor(&body).unwrap();
    assert_eq!(older.seq, open.seq);
    assert_eq!(older.session, open.session);
    assert_eq!(older.ingress, open.ingress);
    assert_eq!(older.prompt, open.prompt);
    assert_eq!(older.prefix, open.prefix);
    assert_eq!(older.exactness, open.exactness);
    assert_eq!(older.priority, open.priority);
    assert_eq!(older.sampling.seed, open.sampling.seed);
    assert_eq!(older.sampling.top_k, open.sampling.top_k);
    assert_eq!(older.sampling.logit_bias, open.sampling.logit_bias.to_vec());
    assert_eq!(older.limits.max_tokens, open.limits.max_tokens);
    assert_eq!(older.limits.stop_tokens, open.limits.stop_tokens.to_vec());
    assert_eq!(older.limits.deadline, open.limits.deadline);

    let meta = open.meta();
    let older: OlderMeta = envelope::from_cbor(&encode_seq_meta(&meta).unwrap()).unwrap();
    assert_eq!(older.seq, meta.seq);
    assert_eq!(older.prompt_len, 5);
    assert_eq!(older.resume, meta.resume);
    assert_eq!(older.prefix, meta.prefix);
}

/// Criterion 3: a client-seeded `n = 2` request opens two sequences with one seed and
/// `choice_index` 0 and 1. Each index survives the `Open` (and the `ResumeOpen` that recovery
/// sends), the `SequenceMeta` the head builds from the decoded open, and that meta's `SEQ_META`
/// encoding.
#[test]
fn choice_index_0_and_1_survive_the_open_and_its_meta() {
    for (n, index) in [(10, 0u16), (11, 1)] {
        let mut open = plain_open(n, vec![5, 6, 7]);
        open.sampling.seed = 0x5EED;
        open.choice_index = Some(index);
        for resume in [
            None,
            Some(ResumePoint {
                emitted: 2,
                rng_counter: 9,
                cause: ResumeCause::Recovery,
            }),
        ] {
            open.resume = resume;
            let SequenceControl::Open(decoded) =
                round_trip(&SequenceControl::Open(open.clone()), small_slot())
            else {
                panic!("not an open");
            };
            assert_eq!(decoded.choice_index, Some(index));
            assert_eq!(decoded.sampling.seed, 0x5EED);
            let meta = decoded.meta();
            assert_eq!(meta.choice_index, Some(index));
            let carried: SequenceMeta = decode_seq_meta(&encode_seq_meta(&meta).unwrap()).unwrap();
            assert_eq!(carried.choice_index, Some(index));
            assert_eq!(carried, meta);
        }
    }
}

/// `SequenceMeta` with RFC-0032 §3.6's `prefix` and `parity` set round-trips through `SEQ_META`.
#[test]
fn seq_meta_carries_prefix_and_parity() {
    let meta = full_open().meta();
    assert!(meta.prefix.is_some() && meta.parity.is_some());
    let bytes = encode_seq_meta(&meta).unwrap();
    assert_eq!(decode_seq_meta(&bytes), Ok(meta));
    assert!(matches!(
        decode_seq_meta(&bytes[..bytes.len() - 1]),
        Err(DecodeError::Malformed(_) | DecodeError::Truncated { .. })
    ));
}

/// Kind 1 is an `Open` without `resume` and kind 6 one with it; a frame whose kind and body
/// disagree is refused.
#[test]
fn an_open_and_a_resume_open_must_agree_with_resume() {
    let resumed = full_open();
    let mut fresh = full_open();
    fresh.resume = None;
    for (kind, open) in [
        (SeqControlKind::Open, &resumed),
        (SeqControlKind::ResumeOpen, &fresh),
    ] {
        let frame = raw_frame(kind.code(), 0, &envelope::to_cbor(open).unwrap());
        let error = Frame::decode(&frame)
            .unwrap()
            .seq_control(SeqControlLimits::slot(small_slot()))
            .unwrap_err();
        assert_eq!(error, SeqControlError::ResumeMismatch { kind });
        assert_eq!(error.class(), ErrorClass::InvalidRequest);
    }
}

/// Bytes that look random to zstd, for a body that compression cannot shrink.
fn noise(len: usize) -> Vec<u8> {
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 56) as u8
        })
        .collect()
}

/// A body too large for the slot as plain CBOR is compressed, and decodes; one too large even
/// compressed is refused with a `Capacity` error before anything is sent.
#[test]
fn compression_is_used_only_when_plain_cbor_does_not_fit() {
    let slot = small_slot();
    let mut open = plain_open(3, vec![1; 100]);
    open.sampling.grammar_inline = Some(vec![b'g'; 2 * slot]);
    let control = SequenceControl::Open(open.clone());
    let bytes = encode_seq_control(&control, &spec_for(1), slot).unwrap();
    assert_eq!(ext_bytes(&bytes), (SeqControlKind::Open.code(), 1), "zstd");
    assert!(bytes.len() <= slot);
    let frame = Frame::decode(&bytes).unwrap();
    assert_eq!(
        frame.seq_control_kind(),
        Ok((SeqControlKind::Open, envelope::Encoding::ZstdCbor))
    );
    assert_eq!(frame.seq_control(SeqControlLimits::slot(slot)), Ok(control));

    // Decompression stops at the reader's limit.
    let tight = SeqControlLimits {
        slot_size: slot,
        body: BodyLimit::new(slot).unwrap(),
    };
    assert!(matches!(
        frame.seq_control(tight),
        Err(SeqControlError::Body(DecodeError::TooLarge { .. }))
    ));

    open.sampling.grammar_inline = Some(noise(2 * slot));
    let error = encode_seq_control(&SequenceControl::Open(open), &spec_for(1), slot).unwrap_err();
    let SeqControlError::DoesNotFit { len, slot_size } = error else {
        panic!("{error:?}");
    };
    assert!(len > slot, "{len} bytes compressed");
    assert_eq!(slot_size, slot);
    assert_eq!(error.class(), ErrorClass::Capacity);
}

/// A `Reconcile` body as bytes, its `recent` array declaring `zeros` zero tokens: review-net's
/// shape of frame, built straight as CBOR so the test never holds the array as values.
fn reconcile_with_zeros(zeros: u32) -> Vec<u8> {
    let mut body = vec![0xa4];
    for (key, value) in [
        ("seq", seq_value(seq(1))),
        ("emitted", int(4)),
        ("rng_counter", int(99)),
    ] {
        body.extend(cbor(&text(key)));
        body.extend(cbor(&value));
    }
    body.extend(cbor(&text("recent")));
    body.push(0x9a);
    body.extend(zeros.to_be_bytes());
    body.resize(body.len() + zeros as usize, 0);
    body
}

/// B1 of #1809's review: a 2 KB zstd frame whose `Reconcile` declares millions of tokens is
/// refused once decompression passes the small kinds' 1 KiB ceiling, before any array is built.
/// A body over the ceiling is refused plain too, and an encoder never writes one.
#[test]
fn a_small_frame_cannot_declare_a_large_body_for_a_small_kind() {
    let body = reconcile_with_zeros(6_000_000);
    let compressed = zstd::bulk::compress(&body, 3).unwrap();
    let frame = raw_frame(SeqControlKind::Reconcile.code(), 1, &compressed);
    assert!(frame.len() < 4096, "a {}-byte frame", frame.len());
    let error = Frame::decode(&frame)
        .unwrap()
        .seq_control(SeqControlLimits::slot(sequence_slot_size(LONG_CONTEXT)))
        .unwrap_err();
    let SeqControlError::Body(DecodeError::TooLarge { limit, .. }) = error else {
        panic!("{error:?}");
    };
    assert_eq!(limit, SMALL_BODY);
    assert_eq!(error.class(), ErrorClass::InvalidRequest);

    let plain = raw_frame(7, 0, &reconcile_with_zeros(2000));
    let error = Frame::decode(&plain)
        .unwrap()
        .seq_control(SeqControlLimits::slot(small_slot()))
        .unwrap_err();
    assert!(
        matches!(error, SeqControlError::Body(DecodeError::TooLarge { .. })),
        "{error:?}"
    );
}

/// N1 of #1809's review: an `Open` whose body is over [`open_body_limit`] of the receiver's slot
/// is refused by the encoder before it compresses, and, built by hand and compressed into a small
/// frame, by the decoder once decompression passes the limit.
#[test]
fn an_open_over_its_declared_size_is_refused_both_ways() {
    let slot = small_slot();
    let open = plain_open(5, vec![0; 1_000_000]);
    let cbor = envelope::to_cbor(&open).unwrap();
    assert!(cbor.len() > open_body_limit(slot));
    assert_eq!(
        encode_seq_control(&SequenceControl::Open(open), &spec_for(1), slot),
        Err(SeqControlError::BodyTooLarge {
            len: cbor.len(),
            limit: open_body_limit(slot),
        })
    );
    let frame = raw_frame(
        SeqControlKind::Open.code(),
        1,
        &zstd::bulk::compress(&cbor, 3).unwrap(),
    );
    assert!(frame.len() < slot, "a {}-byte frame", frame.len());
    let error = Frame::decode(&frame)
        .unwrap()
        .seq_control(SeqControlLimits::slot(slot))
        .unwrap_err();
    let SeqControlError::Body(DecodeError::TooLarge { limit, .. }) = error else {
        panic!("{error:?}");
    };
    assert_eq!(limit, open_body_limit(slot));

    // Within the ceiling, an open larger than the slot still compresses and decodes.
    let mut open = plain_open(6, vec![1; 100]);
    open.sampling.grammar_inline = Some(vec![b'g'; INLINE_ALLOWANCE / 2]);
    let control = SequenceControl::Open(open);
    let bytes = encode_seq_control(&control, &spec_for(1), slot).unwrap();
    assert_eq!(ext_bytes(&bytes).1, 1, "zstd");
    assert_eq!(
        Frame::decode(&bytes)
            .unwrap()
            .seq_control(SeqControlLimits::slot(slot)),
        Ok(control)
    );
}

/// The small kinds' ceiling holds on the encoder too: a `Release` or a `Reconcile` never needs
/// more than [`SMALL_BODY`], and every legitimate one is far under it.
#[test]
fn every_small_kind_fits_its_ceiling() {
    for control in [
        SequenceControl::Cancel {
            seq: seq(1),
            reason: CancelReason::Error {
                domain: u16::MAX,
                code: u16::MAX,
            },
        },
        SequenceControl::Release {
            seq: seq(u32::MAX),
            persist: Some(SessionId([0xff; 16])),
        },
        SequenceControl::Reconcile {
            seq: seq(u32::MAX),
            emitted: u32::MAX,
            rng_counter: u64::MAX,
            recent: vec![u32::MAX; MAX_RECENT].into(),
        },
    ] {
        let bytes = encode_seq_control(&control, &spec_for(1), small_slot()).unwrap();
        assert!(
            bytes.len() - 128 < SMALL_BODY / 4,
            "{} bytes",
            bytes.len() - 128
        );
    }
}

/// A frame larger than the receiver's slot is refused, with RFC-0031 §12's `FrameTooLarge`.
#[test]
fn a_frame_over_the_receivers_slot_is_refused() {
    let control = SequenceControl::Open(plain_open(4, vec![3; 1000]));
    let bytes = encode_seq_control(&control, &spec_for(1), small_slot()).unwrap();
    let frame = Frame::decode(&bytes).unwrap();
    let error = frame
        .seq_control(SeqControlLimits::slot(bytes.len() - 1))
        .unwrap_err();
    assert_eq!(
        error,
        SeqControlError::OverSlot {
            len: bytes.len(),
            slot_size: bytes.len() - 1
        }
    );
    assert_eq!(error.class(), ErrorClass::InvalidRequest);
    assert_eq!(error.code(), Some(code::FRAME_TOO_LARGE));
    assert_eq!(
        frame.seq_control(SeqControlLimits::slot(bytes.len())),
        Ok(control)
    );
}

/// A kind or body encoding this build does not know is refused as `Unsupported`.
#[test]
fn unknown_kinds_and_encodings_are_refused() {
    let body = cbor(&map(vec![("seq", seq_value(seq(1)))]));
    for kind in [0u8, 8, 255] {
        let frame = raw_frame(kind, 0, &body);
        let error = Frame::decode(&frame)
            .unwrap()
            .seq_control(SeqControlLimits::slot(small_slot()))
            .unwrap_err();
        assert_eq!(error, SeqControlError::UnknownKind(kind));
        assert_eq!(error.class(), ErrorClass::Unsupported);
    }
    let frame = raw_frame(3, 2, &body);
    let error = Frame::decode(&frame)
        .unwrap()
        .seq_control(SeqControlLimits::slot(small_slot()))
        .unwrap_err();
    assert_eq!(
        error,
        SeqControlError::Body(DecodeError::UnknownEncoding(2))
    );
    assert_eq!(error.class(), ErrorClass::Unsupported);
}

/// A `Reconcile` carries at most 8 recent tokens, both ways.
#[test]
fn a_reconcile_with_more_than_eight_recent_tokens_is_refused() {
    let control = SequenceControl::Reconcile {
        seq: seq(1),
        emitted: 9,
        rng_counter: 1,
        recent: (0..9).collect::<Vec<u32>>().into(),
    };
    assert_eq!(
        encode_seq_control(&control, &spec_for(1), small_slot()),
        Err(SeqControlError::TooManyRecent { len: 9 })
    );
    let body = map(vec![
        ("seq", seq_value(seq(1))),
        ("emitted", int(9)),
        ("rng_counter", int(1)),
        (
            "recent",
            Value::Array((0..=MAX_RECENT as u64).map(int).collect()),
        ),
    ]);
    let error = Frame::decode(&raw_frame(7, 0, &cbor(&body)))
        .unwrap()
        .seq_control(SeqControlLimits::slot(small_slot()))
        .unwrap_err();
    assert_eq!(error, SeqControlError::TooManyRecent { len: 9 });
    assert_eq!(error.class(), ErrorClass::InvalidRequest);
}

/// A frame of another type, a sealed one, or one with a body of the wrong shape is refused,
/// each with a class RFC-0060 §10.2 allows.
#[test]
fn other_frames_and_bad_bodies_are_refused() {
    let token = spec::frame(&base(0x02, 0, 0, 1), &spec::ext(16), &[]);
    assert_eq!(
        Frame::decode(&token)
            .unwrap()
            .seq_control(SeqControlLimits::slot(small_slot())),
        Err(SeqControlError::Frame(FrameError::WrongType {
            expected: FrameType::SeqControl,
            found: FrameType::Token,
        }))
    );
    let sealed = encode_to_vec(
        &FrameSpec {
            wire_minor: WIRE_MINOR,
            ftype: FrameType::SeqControl,
            flags: Flags(Flags::ENCRYPTED),
            instance: InstanceId(INSTANCE),
            epoch: epoch(),
            step: 0,
            chan_seq: 1,
        },
        &spec::ext(8),
        &[0; 32],
    )
    .unwrap();
    assert_eq!(
        Frame::decode(&sealed)
            .unwrap()
            .seq_control(SeqControlLimits::slot(small_slot())),
        Err(SeqControlError::Frame(FrameError::Sealed(
            FrameType::SeqControl
        )))
    );
    let no_ext = spec::frame(&base(0x03, 0, 0, 1), &[], &[0xa0]);
    let error = Frame::decode(&no_ext)
        .unwrap()
        .seq_control(SeqControlLimits::slot(small_slot()))
        .unwrap_err();
    assert!(
        matches!(
            error,
            SeqControlError::Frame(FrameError::ExtTooShort { .. })
        ),
        "{error:?}"
    );
    // A prompt byte string that is not a whole number of tokens: the open's 4-byte prompt,
    // header 0x44, shortened to 3 bytes in place.
    let mut short_prompt = envelope::to_cbor(&plain_open(1, vec![1])).unwrap();
    let at = short_prompt
        .windows(5)
        .position(|w| w == [0x44, 1, 0, 0, 0])
        .unwrap();
    short_prompt[at] = 0x43;
    short_prompt.remove(at + 1);
    for (kind, body) in [
        // A Cancel without its reason.
        (2u8, cbor(&map(vec![("seq", seq_value(seq(1)))]))),
        // Not a map at all.
        (4, cbor(&int(5))),
        (1, short_prompt),
    ] {
        let error = Frame::decode(&raw_frame(kind, 0, &body))
            .unwrap()
            .seq_control(SeqControlLimits::slot(small_slot()))
            .unwrap_err();
        assert!(
            matches!(error, SeqControlError::Body(DecodeError::Mismatch(_))),
            "kind {kind}: {error:?}"
        );
        assert!(REFUSAL_CLASSES.contains(&error.class()), "{error}");
    }
}

proptest! {
    /// Decoding is total: any extension bytes and any payload under a valid header give a
    /// message or a refusal of an allowed class, never a panic.
    #[test]
    fn decoding_any_body_never_panics(
        kind in any::<u8>(),
        body_enc in 0u8..3,
        payload in proptest::collection::vec(any::<u8>(), 0..512),
    ) {
        let frame = raw_frame(kind, body_enc, &payload);
        let decoded = Frame::decode(&frame).unwrap();
        if let Err(error) = decoded.seq_control(SeqControlLimits::slot(small_slot())) {
            prop_assert!(
                REFUSAL_CLASSES.contains(&error.class()),
                "{error}: {:?}",
                error.class()
            );
        }
    }

    /// The same for a golden's body with one byte changed: the CRCs are recomputed, so the
    /// change reaches the body decoder.
    #[test]
    fn a_changed_golden_body_is_read_or_refused(
        which in 0usize..6,
        at in any::<prop::sample::Index>(),
        byte in any::<u8>(),
    ) {
        let golden = &goldens()[which];
        let payload_start = 128;
        let mut payload = golden.bytes[payload_start..].to_vec();
        let at = at.index(payload.len());
        payload[at] = byte;
        let frame = raw_frame(golden.bytes[64], 0, &payload);
        if let Err(error) = Frame::decode(&frame)
            .unwrap()
            .seq_control(SeqControlLimits::slot(small_slot()))
        {
            prop_assert!(REFUSAL_CLASSES.contains(&error.class()), "{error}");
        }
    }
}
