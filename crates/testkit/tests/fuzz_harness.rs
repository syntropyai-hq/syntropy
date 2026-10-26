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

/// A size the format declares, checked against the input, raises the bound: a parse that holds
/// that much beyond the default allowance passes when it declares it and fails when it does not.
#[test]
fn a_declared_size_raises_the_bound() {
    let input = vec![0u8; 1 << 20];
    let default = Properties::default().allocation_bound(input.len(), None) as usize;
    let holds = |declared: Option<u64>| {
        move |input: &[u8]| {
            let held = vec![1u8; default + input.len() / 2];
            std::hint::black_box(&held);
            Ok(Accepted { declared })
        }
    };
    let half = (input.len() / 2) as u64 + 4096;
    assert_eq!(
        Properties::default().check(&input, holds(Some(half))),
        Ok(())
    );
    let undeclared = Properties::default()
        .check(&input, holds(None))
        .unwrap_err();
    assert!(
        matches!(&undeclared[..], [Violation::OverAllocation { .. }]),
        "{undeclared:?}"
    );
}

/// A declared size larger than the input is reported: the parser trusted a size field it never
/// checked (review of #1886, B2).
#[test]
fn a_declared_size_larger_than_the_input_is_reported() {
    let violations = Properties::default()
        .check(b"x", |_| Ok(Accepted::declaring(u64::MAX)))
        .unwrap_err();
    assert_eq!(
        violations,
        [Violation::UncheckedDeclaredSize {
            declared: u64::MAX,
            len: 1
        }]
    );
}

/// The correct fixture parser keeps every property on inputs that once made it over-allocate:
/// two declarations, a declaration then an unknown tag, and a declaration then a lone byte
/// (review of #1886, B1). A declaration larger than what follows is rejected before it allocates.
#[test]
fn the_correct_parser_checks_declarations_before_allocating() {
    let size = |n: u32| n.to_le_bytes();
    let mut two = record(DECLARE, &size(64 << 20));
    two.extend(record(DECLARE, &size(64 << 20)));
    let mut unknown = record(DECLARE, &size(64 << 20));
    unknown.extend([0x7f, 0]);
    let mut lone = record(DECLARE, &size(64 << 20));
    lone.push(DATA);
    for input in [two, unknown, lone] {
        assert_eq!(
            Properties::default().check(&input, fixture::parse),
            Ok(()),
            "{input:?}"
        );
        assert_eq!(
            fixture::parse(&input),
            Err(ErrorClass::Integrity),
            "{input:?}"
        );
    }
    let mut fits = record(DECLARE, &size(3));
    fits.extend(record(DATA, b"a"));
    assert_eq!(fixture::parse(&fits), Ok(Accepted::declaring(3)));
    let mut second = record(DECLARE, &size(0));
    second.extend(record(DECLARE, &size(0)));
    assert_eq!(fixture::parse(&second), Err(ErrorClass::InvalidRequest));
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
