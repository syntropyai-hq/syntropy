// SPDX-License-Identifier: MIT OR Apache-2.0

//! P-bench-2: the comparability rule of RFC-0061 §2.1.

use proptest::prelude::*;
use proptest::sample::select;
use syntropy_bench::schema::{BenchKey, CacheStart, KeyField};

use crate::support::{
    bench_key, bench_result, build_id, copy_field, other_key, sample_key, sample_result,
};

/// Every key field except `build`.
fn non_build_field() -> impl Strategy<Value = KeyField> {
    select(
        KeyField::ALL
            .iter()
            .copied()
            .filter(|f| *f != KeyField::Build)
            .collect::<Vec<_>>(),
    )
}

proptest! {
    /// P-bench-2: keys that differ only in `build` are comparable.
    #[test]
    fn p_bench_2_results_differing_only_in_build_are_comparable(
        mut a in bench_result(),
        mut b in bench_result(),
        build in build_id(),
    ) {
        // `b` keeps its own measurements; only its key is tied to `a`'s.
        b.key = BenchKey { build, ..a.key.clone() };
        prop_assert!(a.comparable_with(&b));
        prop_assert!(b.comparable_with(&a));
        let differences = a.key.differences(&b.key);
        prop_assert!(differences.is_empty() || differences == vec![KeyField::Build]);

        a.key.build = b.key.build.clone();
        prop_assert!(a.key.differences(&b.key).is_empty());
        prop_assert!(a.comparable_with(&b));
    }

    /// P-bench-2: a difference in any other key field makes the results incomparable, with or
    /// without a different build.
    #[test]
    fn p_bench_2_any_other_key_difference_makes_results_incomparable(
        a in bench_result(),
        donor in bench_key(),
        field in non_build_field(),
        also_build in any::<bool>(),
    ) {
        // Take the field from the random donor, or, where it happens to match, from whichever of
        // the two fixed keys differs: they differ in every field, so one of them does here.
        let mut b = a.clone();
        for source in [donor.clone(), sample_key(), other_key()] {
            copy_field(&mut b.key, &source, field);
            if b.key != a.key {
                break;
            }
        }
        prop_assert_ne!(&b.key, &a.key);
        if also_build {
            copy_field(&mut b.key, &donor, KeyField::Build);
        }
        prop_assert!(!a.comparable_with(&b));
        prop_assert!(!b.comparable_with(&a));
        let differences = a.key.differences(&b.key);
        prop_assert!(differences.contains(&field));
        prop_assert!(differences.iter().all(|f| *f == field || *f == KeyField::Build));
    }
}

/// The criterion names `expert_cache`: learned against cold-cache never compares.
#[test]
fn expert_cache_difference_makes_results_incomparable() {
    let a = sample_result();
    let mut b = sample_result();
    assert_eq!(a.key.expert_cache, CacheStart::Learned);
    b.key.expert_cache = CacheStart::Cold;
    assert!(!a.comparable_with(&b));
    assert_eq!(a.key.differences(&b.key), vec![KeyField::ExpertCache]);

    b.key.build = other_key().build;
    assert!(!a.comparable_with(&b));
    assert_eq!(
        a.key.differences(&b.key),
        vec![KeyField::ExpertCache, KeyField::Build]
    );
}

/// `differences` sees every field: two keys that differ everywhere report all 17.
#[test]
fn every_key_field_is_reported() {
    assert_eq!(
        sample_key().differences(&other_key()),
        KeyField::ALL.to_vec()
    );
    for field in KeyField::ALL {
        let mut b = sample_key();
        copy_field(&mut b, &other_key(), field);
        assert_eq!(sample_key().differences(&b), vec![field], "{field}");
        assert_eq!(
            sample_key().comparable_with(&b),
            field == KeyField::Build,
            "{field}"
        );
    }
}

/// `KeyField::ALL` names exactly the key's JSON fields.
#[test]
fn key_field_names_match_json() {
    let json = serde_json::to_value(sample_key()).expect("serialise key");
    let mut names: Vec<&str> = json
        .as_object()
        .expect("object")
        .keys()
        .map(String::as_str)
        .collect();
    let mut expected: Vec<&str> = KeyField::ALL.iter().map(|f| f.name()).collect();
    names.sort_unstable();
    expected.sort_unstable();
    assert_eq!(names, expected);
}
