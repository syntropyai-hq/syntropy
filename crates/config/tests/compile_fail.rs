// SPDX-License-Identifier: MIT OR Apache-2.0

//! The schema rules that fail the build (PROD-E1-T01; RFC-0054 §2.3 steps 9–10, §5.1, §10.3).
//! Each case in `tests/ui/` must fail to compile with the error in its `.stderr` file.

#[test]
fn schema_rules_fail_the_build() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/*.rs");
}
