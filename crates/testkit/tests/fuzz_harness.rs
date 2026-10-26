// SPDX-License-Identifier: MIT OR Apache-2.0

//! The fuzz harness of RFC-0060 §10.2 (QUAL-E10-T01) reports each property a fixture target
//! breaks: a planted panic, an over-allocation and an `Internal`-class rejection. It runs with
//! `CountingAlloc` as this binary's global allocator, as a fuzz target does.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Duration;

use syntropy_core::ErrorClass;
use syntropy_testkit::fuzz::fixture::{self, DATA, DECLARE, PLANTED_BYTES};
use syntropy_testkit::fuzz::{Accepted, Properties, Violation};
use syntropy_testkit::guards::CountingAlloc;

#[global_allocator]
static ALLOCATOR: CountingAlloc = CountingAlloc;

fn record(tag: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = vec![tag, payload.len() as u8];
    out.extend_from_slice(payload);
    out
}

#[test]
fn valid_inputs_and_allowed_rejections_pass() {
    let properties = Properties::default();
    let mut input = record(DATA, b"hello");
    input.extend(record(DATA, &[]));
    for input in [Vec::new(), input] {
        assert_eq!(properties.check(&input, fixture::parse), Ok(()));
    }
    // A truncated record is Integrity and an unknown tag InvalidRequest: both allowed.
    for input in [vec![DATA, 9, 1], vec![0x7f, 0], vec![DATA]] {
        assert_eq!(
            properties.check(&input, fixture::parse),
            Ok(()),
            "{input:?}"
        );
    }
}

#[test]
fn the_planted_panic_is_reported() {
    let violations = Properties::default()
        .check(&record(fixture::PLANTED_PANIC, &[]), fixture::parse_planted)
        .unwrap_err();
    assert!(
        matches!(&violations[..], [Violation::Panic { message }] if message.contains("planted panic")),
        "{violations:?}"
    );
}

#[test]
fn the_planted_over_allocation_is_reported() {
    let input = record(fixture::PLANTED_OVER_ALLOCATION, &[]);
    let violations = Properties::default()
        .check(&input, fixture::parse_planted)
        .unwrap_err();
    match &violations[..] {
        [Violation::OverAllocation { bytes, bound }] => {
            assert!(*bytes >= PLANTED_BYTES as u64, "{bytes}");
            assert_eq!(*bound, 4 * input.len() as u64 + (1 << 20));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn the_planted_internal_rejection_is_reported() {
    let violations = Properties::default()
        .check(
            &record(fixture::PLANTED_INTERNAL, &[]),
            fixture::parse_planted,
        )
        .unwrap_err();
    assert_eq!(
        violations,
        [Violation::Rejection {
            class: ErrorClass::Internal
        }]
    );
}

/// A size the format declares raises the bound: the same 8 MiB that fails undeclared passes
/// when a record declares it, bookkeeping included.
#[test]
fn a_declared_size_raises_the_bound() {
    let input = record(DECLARE, &(PLANTED_BYTES as u32).to_le_bytes());
    assert_eq!(Properties::default().check(&input, fixture::parse), Ok(()));
    // The clean parser treats a planted tag as unknown, so it passes as InvalidRequest.
    let planted = record(fixture::PLANTED_OVER_ALLOCATION, &[]);
    assert_eq!(
        Properties::default().check(&planted, fixture::parse),
        Ok(())
    );
}

#[test]
fn a_slow_parse_is_reported() {
    let properties = Properties {
        time_limit: Duration::from_millis(10),
        ..Properties::default()
    };
    let violations = properties
        .check(b"", |_| {
            std::thread::sleep(Duration::from_millis(50));
            Ok(Accepted::plain())
        })
        .unwrap_err();
    assert!(
        matches!(&violations[..], [Violation::TooSlow { .. }]),
        "{violations:?}"
    );
}

/// `assert` is what a fuzz target calls: it panics with every violation, so libFuzzer records
/// the input as a crash.
#[test]
fn assert_panics_with_the_violation() {
    let input = record(fixture::PLANTED_INTERNAL, &[]);
    let panic = catch_unwind(AssertUnwindSafe(|| {
        Properties::default().assert(&input, fixture::parse_planted)
    }))
    .unwrap_err();
    let message = panic.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(
        message.contains("RFC-0060 §10.2") && message.contains("Internal"),
        "{message}"
    );
}
