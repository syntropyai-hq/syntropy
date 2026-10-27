// SPDX-License-Identifier: MIT OR Apache-2.0

//! RFC-0001 §5.12's `Error` (COMP-E13-T15).
//!
//! - Every type round-trips through serde, in the IPC's postcard encoding and in JSON, from a
//!   buffer that is dropped before the value is used. [`decode`] asks for `DeserializeOwned`
//!   and no `'static` input, so a type that borrowed from its input would not compile here
//!   (RFC-0001 §18.32).
//! - An `Error` keeps its context and its source chain through a round trip, and
//!   `std::error::Error::source` walks the same chain.

use std::borrow::Cow;
use std::error::Error as _;

use proptest::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;
use syntropy_core::error::codes;
use syntropy_core::{
    Domain, Endpoint, Error, ErrorClass, ErrorCode, ErrorScope, LinkId, LinkKind, Nanos, NodeId,
    RetryHint, SequenceId, StageIx,
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
        ErrorClass::Device,
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
