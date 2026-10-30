// SPDX-License-Identifier: MIT OR Apache-2.0

//! The boundary's domain numbers (RFC-0040 §10.4, note of 2026-10-30): every `core::Domain` has
//! one, the numbers never collide, `Kernels` keeps 1, and a number this build does not know
//! decodes as nothing. `core`'s `tests/error_codes.rs` checks that `Domain::ALL` holds every
//! variant, so a domain added to `core` fails here until it has a number.

use std::collections::BTreeSet;

use syntropy_core::{Domain, ErrorCode};
use syntropy_kernels::DeviceHealth;
use syntropy_kernels::abi::{DOMAIN_CODES, DOMAIN_KERNELS, SynHealth, domain_code, domain_of_code};

#[test]
fn every_domain_has_a_boundary_number() {
    for domain in Domain::ALL {
        let code = domain_code(domain).unwrap_or_else(|| panic!("{domain:?} has no number"));
        assert_eq!(domain_of_code(code), Some(domain));
    }
    assert_eq!(DOMAIN_CODES.len(), Domain::ALL.len());
}

#[test]
fn numbers_are_distinct_and_kernels_keeps_one() {
    let numbers: BTreeSet<u32> = DOMAIN_CODES.iter().map(|&(_, n)| n).collect();
    assert_eq!(numbers.len(), DOMAIN_CODES.len());
    assert!(!numbers.contains(&0), "0 is the number of no domain");
    assert_eq!(DOMAIN_KERNELS, 1);
    assert_eq!(domain_code(Domain::Kernels), Some(1));
    // The published table (RFC-0040 §10.4, note of 2026-10-30): numbers never move.
    assert_eq!(
        DOMAIN_CODES,
        [
            (Domain::Kernels, 1),
            (Domain::Cluster, 2),
            (Domain::Net, 3),
            (Domain::Dist, 4),
            (Domain::Recovery, 5),
            (Domain::Security, 6),
            (Domain::Exec, 7),
        ]
    );
}

#[test]
fn an_unknown_number_has_no_domain() {
    for n in [0, 8, 99, u32::MAX] {
        assert_eq!(domain_of_code(n), None, "{n}");
    }
}

/// `Sticky` carries a code of any domain across the boundary unchanged: a `Net` code is a
/// `Net` code on the other side (COMP-E13-T15).
#[test]
fn a_sticky_health_keeps_its_domain() {
    for domain in Domain::ALL {
        let health = DeviceHealth::Sticky {
            code: ErrorCode::new(domain, 0x0042),
        };
        let wire = SynHealth::of(health).expect("every domain has a number");
        assert_eq!(wire.kind, 2);
        assert_eq!(Some(wire.domain), domain_code(domain));
        assert_eq!(wire.to_health(), Some(health));
    }
}

/// A `Sticky` health with a domain number this build does not know decodes as `None`, which
/// the core's proxy reports as an unknown health.
#[test]
fn a_sticky_health_of_an_unknown_domain_is_none() {
    let wire = SynHealth {
        kind: 2,
        domain: 99,
        detail: 0x0042,
    };
    assert_eq!(wire.to_health(), None);
}

#[test]
fn the_health_record_is_twelve_bytes() {
    assert_eq!(std::mem::size_of::<SynHealth>(), 12);
}
