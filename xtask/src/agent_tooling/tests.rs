// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

const DIGEST_A: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const DIGEST_B: &str = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

fn lock() -> Vec<Tool> {
    parse(&format!(
        "[[tool]]\nname = \"fixture-server\"\nkind = \"toolserver\"\nversion = \"4.7.2\"\n\
         digest = \"{DIGEST_A}\"\nabout = \"a fixture\"\n\n\
         [[tool]]\nname = \"fixture-plugin\"\nkind = \"plugin\"\nabout = \"a fixture\"\n\
         planned = \"QUAL-E1-T24\"\n"
    ))
    .unwrap()
}

fn observed(version: &str, digest: &str) -> BTreeMap<String, Pin> {
    parse_observed(&format!(
        "[[tool]]\nname = \"fixture-server\"\nversion = \"{version}\"\ndigest = \"{digest}\"\n"
    ))
    .unwrap()
}

fn commit(message: &str) -> Commit {
    Commit {
        sha: "0123456789abcdef".to_owned(),
        author_name: "a".to_owned(),
        author_email: "a@example.invalid".to_owned(),
        message: message.to_owned(),
    }
}

#[test]
fn the_workspace_lock_is_well_formed() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../ci/agent-tooling.lock");
    parse(&std::fs::read_to_string(path).unwrap()).unwrap();
}

#[test]
fn a_tool_is_pinned_or_planned_never_half() {
    assert_eq!(lock().len(), 2);
    let half = "[[tool]]\nname = \"x\"\nkind = \"plugin\"\nversion = \"1\"\nabout = \"a\"\n";
    assert!(
        parse(half)
            .unwrap_err()
            .contains("both `version` and `digest`")
    );
    let bad = "[[tool]]\nname = \"x\"\nkind = \"nope\"\nversion = \"1\"\ndigest = \"sha256:abc\"\nabout = \"a\"\n";
    let error = parse(bad).unwrap_err();
    assert!(error.contains("`kind` must be one of"), "{error}");
    assert!(error.contains("`digest` must be"), "{error}");
    let twice = format!(
        "{0}{0}",
        "[[tool]]\nname = \"x\"\nkind = \"plugin\"\nabout = \"a\"\nplanned = \"T\"\n"
    );
    assert!(parse(&twice).unwrap_err().contains("listed twice"));
    assert!(
        parse("[[tool]]\nname = \"x\"\nbogus = 1\n")
            .unwrap_err()
            .contains("unknown key")
    );
}

#[test]
fn matching_tooling_may_start() {
    assert_eq!(
        verify(&lock(), &observed("4.7.2", DIGEST_A)),
        Vec::<String>::new()
    );
}

#[test]
fn a_run_with_another_digest_or_version_is_refused() {
    let problems = verify(&lock(), &observed("4.7.2", DIGEST_B));
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("has digest"), "{problems:?}");
    let problems = verify(&lock(), &observed("4.8.0", DIGEST_A));
    assert!(
        problems[0].contains("is version 4.8.0; the lock pins 4.7.2"),
        "{problems:?}"
    );
}

#[test]
fn unlisted_unpinned_and_missing_tools_are_refused() {
    let extra = parse_observed(&format!(
        "[[tool]]\nname = \"fixture-server\"\nversion = \"4.7.2\"\ndigest = \"{DIGEST_A}\"\n\
         [[tool]]\nname = \"fixture-plugin\"\nversion = \"1\"\ndigest = \"{DIGEST_A}\"\n\
         [[tool]]\nname = \"stranger\"\nversion = \"1\"\ndigest = \"{DIGEST_A}\"\n"
    ))
    .unwrap();
    let problems = verify(&lock(), &extra);
    assert_eq!(problems.len(), 2, "{problems:?}");
    assert!(problems.iter().any(|p| p.contains("`stranger` is not in")));
    assert!(
        problems
            .iter()
            .any(|p| p.contains("`fixture-plugin` is in") && p.contains("not pinned"))
    );
    let none = parse_observed("").unwrap();
    assert_eq!(
        verify(&lock(), &none),
        ["`fixture-server` is pinned but the run does not have it"]
    );
}

#[test]
fn the_provenance_lists_the_pinned_tools() {
    let record: serde_json::Value = serde_json::from_str(&provenance(&lock())).unwrap();
    assert_eq!(
        record,
        serde_json::json!({"agent_tooling": [{
            "name": "fixture-server", "kind": "toolserver", "version": "4.7.2", "digest": DIGEST_A
        }]})
    );
}

#[test]
fn a_commit_that_names_a_tool_with_its_version_fails() {
    for message in [
        "chore: bump fixture-server 4.7.2",
        "chore: Fixture-Server@4.7.2 is out",
        "chore: fixture-server v4.7.2.",
        "fixture-server-4.7.2",
        &format!("chore: pin {DIGEST_A}"),
        // The phrasings a bump writes, including a version the base lock does not hold yet.
        "chore(deps): bump fixture-server from 4.7.1 to 4.7.2",
        "chore(deps): bump fixture-server to 4.7.2",
        "update fixture-server (4.7.2)",
        "chore(deps): bump fixture-server from 4.7.2 to 4.8.0",
        "fix: fixture-server 4.7.20 is another thing",
        "chore: fixture-server version 4.7.2",
        "chore: fixture-server: pin to 4.7.2",
        // A planned tool's first pin.
        "chore: pin fixture-plugin to 1.0.0",
        // The tool named on one line and its pinned version on another.
        "chore: update fixture-server\n\nIt is now 4.7.2.",
    ] {
        let problems = commit_problems(&lock(), &[commit(message)]);
        assert_eq!(problems.len(), 1, "{message}: {problems:?}");
        assert!(problems[0].contains("commit 0123456789ab"), "{problems:?}");
    }
}

#[test]
fn a_plain_version_number_or_a_bare_name_passes() {
    for message in [
        "chore: bump serde to 4.7.2",
        "fix: handle 4.7.2 as a release tag",
        "feat: ask fixture-server for its tools",
        "fix: fixture-servers 4.7.2",
        "fix: myfixture-server 4.7.2",
        "fix: fixture-server-ext 4.7.2",
        "docs: fixture-server follows RFC-0062 §12.5",
        "fix: fixture-server\n\nSee 4.7.20, which is another thing.",
    ] {
        assert_eq!(
            commit_problems(&lock(), &[commit(message)]),
            Vec::<String>::new(),
            "{message}"
        );
    }
}

#[test]
fn a_v_prefixed_pin_is_matched_with_or_without_the_v() {
    let lock = parse(&format!(
        "[[tool]]\nname = \"fixture-server\"\nkind = \"toolserver\"\nversion = \"v1.2.3\"\n\
         digest = \"{DIGEST_A}\"\nabout = \"a\"\n"
    ))
    .unwrap();
    for message in [
        "bump fixture-server v1.2.3",
        "bump fixture-server 1.2.3",
        "fixture-server@v1.2.3",
    ] {
        assert_eq!(
            commit_problems(&lock, &[commit(message)]).len(),
            1,
            "{message}"
        );
    }
    // The pinned version is matched whole: on a line of its own, `v1.2.30` is another version.
    assert!(
        commit_problems(
            &lock,
            &[commit("bump fixture-server\n\nv1.2.30 is another thing")]
        )
        .is_empty()
    );
}

#[test]
fn the_pr_title_and_body_are_checked_like_a_commit_message() {
    let pr = |title: &str, body: &str| PullRequest {
        title: title.to_owned(),
        body: body.to_owned(),
        labels: Vec::new(),
        commits: 1,
    };
    assert_eq!(
        pull_request_problems(&lock(), &pr("chore(deps): bump fixture-server 4.7.2", "")).len(),
        1
    );
    let problems = pull_request_problems(
        &lock(),
        &pr("chore: tidy", "Moves fixture-server to 4.7.3."),
    );
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].starts_with("the PR body"), "{problems:?}");
    assert!(pull_request_problems(&lock(), &pr("chore: tidy", "No tooling here.")).is_empty());
}

#[test]
fn a_bare_hex_digest_in_a_message_fails() {
    let hex = DIGEST_A.strip_prefix("sha256:").unwrap();
    assert_eq!(
        commit_problems(&lock(), &[commit(&format!("pin {hex}"))]).len(),
        1
    );
}

#[test]
fn a_provenance_record_needs_the_observed_tooling() {
    let context = Context {
        root: std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".."),
        args: crate::cli::Args::from_pairs(&[("--provenance", "unused")]),
    };
    let error = record_step(&context).unwrap_err();
    assert!(error.contains("needs `--observed`"), "{error}");
}
