// SPDX-License-Identifier: MIT OR Apache-2.0

//! RFC-0001 §5.12's `Error` (COMP-E13-T15).
//!
//! - Every type round-trips through serde, in the IPC's postcard encoding and in JSON, from a
//!   buffer that is dropped before the value is used. [`decode`] asks for `DeserializeOwned`
//!   and no `'static` input, so a type that borrowed from its input would not compile here
//!   (RFC-0001 §18.32).
//! - An `Error` keeps its context and its source chain through a round trip, and
//!   `std::error::Error::source` walks the same chain.
//! - A decode refuses a source chain deeper than `MAX_SOURCE_DEPTH`, including a forged
//!   million-deep frame, without exhausting the stack.

use std::borrow::Cow;
use std::error::Error as _;

use proptest::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;
use syntropy_core::error::codes;
use syntropy_core::{
    Domain, Endpoint, Error, ErrorClass, ErrorCode, ErrorScope, LinkId, LinkKind, MAX_SOURCE_DEPTH,
    Nanos, NodeId, RetryHint, SOURCES_DROPPED, SequenceId, StageIx,
};

/// Decodes `bytes` and drops them before returning the value, so the value cannot borrow them.
fn decode<T: DeserializeOwned>(bytes: Vec<u8>, json: bool) -> T {
    let value = if json {
        serde_json::from_slice(&bytes).expect("json decode")
    } else {
        postcard::from_bytes(&bytes).expect("postcard decode")
    };
    drop(bytes);
    value
}

/// Encodes `value` both ways and checks each decodes to an equal value.
fn round_trip<T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug>(value: &T) {
    let postcard = postcard::to_allocvec(value).expect("postcard encode");
    assert_eq!(&decode::<T>(postcard, false), value);
    let json = serde_json::to_vec(value).expect("json encode");
    assert_eq!(&decode::<T>(json, true), value);
}

const CLASSES: [ErrorClass; 13] = [
    ErrorClass::InvalidRequest,
    ErrorClass::Unsupported,
    ErrorClass::Capacity,
    ErrorClass::Transient,
    ErrorClass::Link,
    ErrorClass::Device,
    ErrorClass::Integrity,
    ErrorClass::Correctness,
    ErrorClass::Drift,
    ErrorClass::Timeout,
    ErrorClass::Auth,
    ErrorClass::Cancelled,
    ErrorClass::Internal,
];

fn class() -> impl Strategy<Value = ErrorClass> {
    (0..CLASSES.len()).prop_map(|i| CLASSES[i])
}

/// A published code, or any number in any domain.
fn code() -> impl Strategy<Value = ErrorCode> {
    let published: Vec<ErrorCode> = codes::all().map(|info| info.code).collect();
    let domains = [
        Domain::Cluster,
        Domain::Net,
        Domain::Dist,
        Domain::Recovery,
        Domain::Security,
        Domain::Kernels,
        Domain::Exec,
        Domain::Tier,
    ];
    prop_oneof![
        (0..published.len()).prop_map(move |i| published[i]),
        (0..domains.len(), any::<u16>()).prop_map(move |(d, n)| ErrorCode::new(domains[d], n)),
    ]
}

fn node() -> impl Strategy<Value = NodeId> {
    any::<[u8; 32]>().prop_map(NodeId)
}

fn scope() -> impl Strategy<Value = ErrorScope> {
    prop_oneof![
        any::<(u64, u32, u32)>().prop_map(|(ingress, boot, n)| ErrorScope::Sequence(SequenceId {
            ingress,
            boot,
            n
        })),
        any::<u16>().prop_map(|s| ErrorScope::Stage(StageIx(s))),
        Just(ErrorScope::Runner),
        node().prop_map(ErrorScope::Node),
        (node(), node(), any::<u16>(), any::<u8>()).prop_map(|(a, b, ordinal, path)| {
            ErrorScope::Link(LinkId {
                src: Endpoint::Nic { node: a, ordinal },
                dst: Endpoint::Nic { node: b, ordinal },
                kind: LinkKind::Network(Default::default()),
                path,
            })
        }),
        Just(ErrorScope::Instance),
        Just(ErrorScope::Cluster),
    ]
}

fn retry() -> impl Strategy<Value = RetryHint> {
    prop_oneof![
        Just(RetryHint::Never),
        any::<u64>().prop_map(|n| RetryHint::After(Nanos(n))),
        Just(RetryHint::OnNewPlan),
        Just(RetryHint::Backoff),
    ]
}

/// A context key: a literal, as code writes it, or an owned string, as a decode gives it.
fn key() -> impl Strategy<Value = Cow<'static, str>> {
    prop_oneof![
        prop_oneof![Just("op"), Just("device"), Just("tier"), Just("path")].prop_map(Cow::Borrowed),
        "[a-z_]{0,12}".prop_map(Cow::Owned),
    ]
}

/// One error without a source.
fn leaf() -> impl Strategy<Value = Error> {
    (
        class(),
        code(),
        ".{0,40}",
        scope(),
        retry(),
        prop::collection::vec((key(), ".{0,20}"), 0..6),
    )
        .prop_map(|(class, code, message, scope, retry, context)| {
            let mut e = Error::new(class, code, message)
                .with_scope(scope)
                .with_retry(retry);
            for (k, v) in context {
                e = e.with_context(k, v);
            }
            e
        })
}

/// An error with a source chain up to three deep.
fn error() -> impl Strategy<Value = Error> {
    prop::collection::vec(leaf(), 1..4).prop_map(|mut chain| {
        let mut top = chain.pop().unwrap();
        while let Some(outer) = chain.pop() {
            top = outer.with_source(top);
        }
        top
    })
}

proptest! {
    #[test]
    fn every_error_type_round_trips_from_a_dropped_buffer(
        e in error(),
        s in scope(),
        r in retry(),
        c in code(),
        k in class(),
    ) {
        round_trip(&e);
        round_trip(&s);
        round_trip(&r);
        round_trip(&c);
        round_trip(&c.domain);
        round_trip(&k);
    }

    /// The decoded error has the same context, in order, and the same chain, which
    /// `std::error::Error::source` walks too.
    #[test]
    fn an_error_keeps_its_context_and_source_chain(e in error()) {
        let back: Error = decode(postcard::to_allocvec(&e).unwrap(), false);
        prop_assert_eq!(back.chain().count(), e.chain().count());
        for (b, o) in back.chain().zip(e.chain()) {
            prop_assert_eq!(&b.context, &o.context);
            prop_assert_eq!(&b.message, &o.message);
            prop_assert_eq!(b.code, o.code);
        }
        let mut via_std = 1;
        let mut cur: &dyn std::error::Error = &back;
        while let Some(next) = cur.source() {
            via_std += 1;
            cur = next;
        }
        prop_assert_eq!(via_std, e.chain().count());
    }
}

#[test]
fn context_and_source_by_example() {
    let device = Error::new(
        ErrorClass::Device,
        codes::kernels::DEVICE_STICKY,
        "illegal address",
    )
    .with_context("device", "gpu0");
    let step = Error::new(
        ErrorClass::Internal,
        codes::exec::RING_CLOSED,
        "step 41 abandoned",
    )
    .with_scope(ErrorScope::Stage(StageIx(2)))
    .with_retry(RetryHint::OnNewPlan)
    .with_context("step", String::from("41"))
    .with_source(device.clone());
    assert_eq!(step.context("step"), Some("41"));
    assert_eq!(step.context("device"), None);
    assert_eq!(step.chain().nth(1), Some(&device));
    assert_eq!(
        step.source().map(|s| s.to_string()),
        Some(device.to_string())
    );
    assert_eq!(
        device.to_string(),
        "illegal address (Device, Kernels 0x0008)"
    );
    // A literal key and the owned key a decode gives are equal.
    let json = serde_json::to_vec(&step).unwrap();
    let back: Error = decode(json, true);
    assert!(matches!(back.context[0].0, Cow::Owned(_)));
    assert_eq!(back, step);
    // `new`'s defaults.
    let plain = Error::new(ErrorClass::Internal, codes::exec::STEP_STALLED, "stuck");
    assert_eq!(
        (
            plain.scope,
            plain.retry,
            plain.context.len(),
            plain.source.is_none()
        ),
        (ErrorScope::Runner, RetryHint::Never, 0, true)
    );
}

/// A chain of `n` errors: the outermost and `n − 1` sources, linked through the field rather
/// than [`Error::with_source`], so it can exceed the bound a decode refuses.
fn chain_of(n: usize) -> Error {
    let mut e = Error::new(ErrorClass::Internal, codes::exec::STEP_STALLED, "0");
    for i in 1..n {
        let mut outer = Error::new(
            ErrorClass::Device,
            codes::kernels::DEVICE_STICKY,
            i.to_string(),
        );
        outer.source = Some(Box::new(e));
        e = outer;
    }
    e
}

/// A decode accepts a chain of `MAX_SOURCE_DEPTH` sources below the outermost error and
/// refuses one more, in both encodings.
#[test]
fn a_decode_bounds_the_source_chain() {
    assert_eq!(MAX_SOURCE_DEPTH, 16);
    let deepest = chain_of(MAX_SOURCE_DEPTH + 1);
    round_trip(&deepest);
    let too_deep = chain_of(MAX_SOURCE_DEPTH + 2);
    let postcard = postcard::to_allocvec(&too_deep).unwrap();
    assert!(postcard::from_bytes::<Error>(&postcard).is_err());
    let json = serde_json::to_vec(&too_deep).unwrap();
    let err = serde_json::from_slice::<Error>(&json).unwrap_err();
    assert!(err.to_string().contains("deeper than 16"), "{err}");
}

/// A forged frame nesting a million sources is refused at depth 17, without recursing a
/// million deep. The bytes are built by hand: an error's postcard encoding ends with its
/// `source`, `00` for none, so each wrapping repeats the fields and writes `01` instead.
#[test]
fn a_forged_deep_chain_is_refused_without_exhausting_the_stack() {
    let leaf = postcard::to_allocvec(&Error::new(
        ErrorClass::Internal,
        codes::exec::RING_CLOSED,
        "x",
    ))
    .unwrap();
    assert_eq!(leaf.last(), Some(&0));
    let mut wrap = leaf[..leaf.len() - 1].to_vec();
    wrap.push(1);
    let depth = 1_000_000;
    let mut forged = Vec::with_capacity(wrap.len() * depth + leaf.len());
    for _ in 0..depth {
        forged.extend_from_slice(&wrap);
    }
    forged.extend_from_slice(&leaf);
    assert!(postcard::from_bytes::<Error>(&forged).is_err());
    // The same construction at the limit decodes.
    let mut ok = wrap.repeat(MAX_SOURCE_DEPTH);
    ok.extend_from_slice(&leaf);
    let decoded: Error = postcard::from_bytes(&ok).unwrap();
    assert_eq!(decoded.chain().count(), MAX_SOURCE_DEPTH + 1);
}

/// The JSON form decodes as the derived form did: an absent `source` is none, an unknown field
/// is skipped, and a repeated field is refused.
#[test]
fn the_json_form_keeps_the_derived_rules() {
    let e = Error::new(ErrorClass::Internal, codes::exec::RING_CLOSED, "gone");
    let mut value = serde_json::to_value(&e).unwrap();
    let object = value.as_object_mut().unwrap();
    object.remove("source");
    object.insert("later_field".into(), serde_json::json!([1, 2]));
    let back: Error = serde_json::from_value(value).unwrap();
    assert_eq!(back, e);
    let repeated = r#"{"class":"Link","class":"Link","code":{"domain":"Exec","code":1537},"message":"","scope":"Runner","retry":"Never","context":[]}"#;
    let err = serde_json::from_str::<Error>(repeated).unwrap_err();
    assert!(err.to_string().contains("duplicate field `class`"), "{err}");
}

/// `with_source` keeps a chain within `MAX_SOURCE_DEPTH` sources, so what a sender builds a
/// receiver accepts. Past it the deepest sources go, and the deepest kept records how many,
/// across repeated cuts.
#[test]
fn with_source_keeps_the_chain_within_the_bound() {
    let mut e = Error::new(ErrorClass::Internal, codes::exec::STEP_STALLED, "root");
    for i in 1..=40 {
        e = Error::new(
            ErrorClass::Device,
            codes::kernels::DEVICE_STICKY,
            i.to_string(),
        )
        .with_source(e);
        assert!(e.chain().count() <= MAX_SOURCE_DEPTH + 1);
    }
    // 41 errors were linked; 17 remain, so 24 were dropped.
    assert_eq!(e.chain().count(), MAX_SOURCE_DEPTH + 1);
    let deepest = e.chain().last().unwrap();
    assert_eq!(deepest.context(SOURCES_DROPPED), Some("24"));
    assert_eq!(e.message, "40");
    assert_eq!(deepest.message, "24");
    // Exactly one marker, and a receiver accepts the chain.
    assert_eq!(
        e.chain()
            .filter(|x| x.context(SOURCES_DROPPED).is_some())
            .count(),
        1
    );
    round_trip(&e);
    // Wrapping an over-deep chain built through the field cuts it too. A million errors, so a
    // tail dropped by recursion would overflow the test thread's stack: the cut unlinks it one
    // error at a time.
    let wrapped = Error::new(ErrorClass::Internal, codes::exec::STEP_STALLED, "top")
        .with_source(chain_of(1_000_000));
    assert_eq!(wrapped.chain().count(), MAX_SOURCE_DEPTH + 1);
    assert_eq!(
        wrapped.chain().last().unwrap().context(SOURCES_DROPPED),
        Some("999984")
    );
    // A chain within the bound is untouched.
    let short = chain_of(MAX_SOURCE_DEPTH);
    let wrapped = Error::new(ErrorClass::Internal, codes::exec::STEP_STALLED, "top")
        .with_source(short.clone());
    assert_eq!(wrapped.source.as_deref(), Some(&short));
}

/// A dropped count is untrusted: a chain a receiver accepted can carry any `sources_dropped`.
/// Wrapping one whose markers hold `u64::MAX` saturates the count rather than overflowing.
#[test]
fn a_forged_dropped_count_saturates() {
    for marked in [MAX_SOURCE_DEPTH, MAX_SOURCE_DEPTH + 1] {
        // A 17-error chain, the most a decode accepts, with a `u64::MAX` marker on the deepest
        // error kept (the 16th) or on the one cut (the 17th).
        let mut chain = chain_of(MAX_SOURCE_DEPTH + 1);
        let mut e = &mut chain;
        for _ in 1..marked {
            e = e.source.as_deref_mut().unwrap();
        }
        e.context
            .push((Cow::Borrowed(SOURCES_DROPPED), u64::MAX.to_string()));
        let bytes = postcard::to_allocvec(&chain).unwrap();
        let chain: Error = decode(bytes, false);
        let wrapped =
            Error::new(ErrorClass::Internal, codes::exec::STEP_STALLED, "top").with_source(chain);
        assert_eq!(wrapped.chain().count(), MAX_SOURCE_DEPTH + 1);
        let deepest = wrapped.chain().last().unwrap();
        assert_eq!(
            deepest.context(SOURCES_DROPPED),
            Some("18446744073709551615")
        );
        assert_eq!(
            deepest
                .context
                .iter()
                .filter(|(k, _)| k == SOURCES_DROPPED)
                .count(),
            1
        );
    }
}
