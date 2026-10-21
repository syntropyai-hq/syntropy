// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests over the fixture payloads in `xtask/fixtures/contrib/`. Their identities are samples:
//! `alice` and `bob` are the fixture roster's maintainers, `project-review-agent` its listed
//! agent, `project-agent[bot]` a GitHub App account, and `carol` a contributor.

use super::*;

macro_rules! fixture {
    ($name:literal) => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/contrib/",
            $name
        ))
    };
}

fn roster() -> Roster {
    Roster::parse(fixture!("roster.toml")).unwrap()
}

fn scopes() -> Vec<String> {
    ["core", "obs", "xtask", "quality", "plan"]
        .map(String::from)
        .into()
}

/// `body` as `gh api --include` records a single, last page.
fn recorded(body: &str) -> String {
    format!("HTTP/2.0 200 OK\r\nContent-Type: application/json; charset=utf-8\r\n\r\n{body}\n")
}

fn commits(text: &str) -> Vec<Commit> {
    parse_commits(text).unwrap()
}

/// The `pr-title` problems for an event, commits and label-events fixture.
fn title_problems(event: &str, commit_list: &str, label_events: &str) -> Vec<String> {
    pr_title(
        &parse_pull_request(event).unwrap(),
        &commits(commit_list),
        &parse_label_events(label_events).unwrap(),
        &roster(),
        &scopes(),
        "0.0.0",
    )
}

// ---- dco ----

#[test]
fn signed_off_commits_pass_dco() {
    assert_eq!(
        dco(&commits(fixture!("commits-signed-off.json"))),
        Vec::<String>::new()
    );
}

#[test]
fn a_missing_or_mismatched_sign_off_fails_dco() {
    let problems = dco(&commits(fixture!("commits-missing-sign-off.json")));
    assert_eq!(
        problems,
        [
            "commit dddddddddddd by Carol Contributor <carol@example.org> has no `Signed-off-by:` line",
            "commit eeeeeeeeeeee by project-agent[bot] <1234+project-agent[bot]@users.noreply.github.com> \
             has no `Signed-off-by:` line",
            "commit ffffffffffff is signed off by Alice Maintainer <alice@example.org>, which does not \
             match its author Carol Contributor <carol@example.org>",
        ]
    );
}

#[test]
fn an_agent_authored_commit_goes_through_the_same_check() {
    // The signed-off fixture's second commit is agent-authored and passes; the missing-sign-off
    // fixture's second is agent-authored and fails with the same message a person's commit gets.
    let signed = commits(fixture!("commits-signed-off.json"));
    let agent = &signed[1];
    assert_eq!(agent.author_name, "project-agent[bot]");
    assert_eq!(dco(std::slice::from_ref(agent)), Vec::<String>::new());

    let unsigned = commits(fixture!("commits-missing-sign-off.json"));
    let person = dco(std::slice::from_ref(&unsigned[0]));
    let machine = dco(std::slice::from_ref(&unsigned[1]));
    assert_eq!(person.len(), 1);
    assert_eq!(machine.len(), 1);
    assert!(person[0].ends_with("has no `Signed-off-by:` line"));
    assert!(machine[0].ends_with("has no `Signed-off-by:` line"));
}

#[test]
fn sign_off_matching_ignores_case_and_spacing() {
    let commit = Commit {
        sha: "1".repeat(40),
        author_name: "Ada Lovelace".into(),
        author_email: "Ada@Example.org".into(),
        message: "x\n\n  signed-off-by:   ada lovelace   <ada@example.ORG>  \n".into(),
    };
    assert_eq!(dco(&[commit]), Vec::<String>::new());
}

#[test]
fn a_sign_off_must_name_both_author_name_and_email() {
    let base = Commit {
        sha: "2".repeat(40),
        author_name: "Ada Lovelace".into(),
        author_email: "ada@example.org".into(),
        message: String::new(),
    };
    for message in [
        "Signed-off-by: Ada Lovelace <other@example.org>",
        "Signed-off-by: Someone Else <ada@example.org>",
        "Signed-off-by: <ada@example.org>",
        "Signed-off-by: Ada Lovelace",
        "Acked-by: Ada Lovelace <ada@example.org>",
    ] {
        let commit = Commit {
            message: format!("x\n\n{message}\n"),
            ..base.clone()
        };
        assert_eq!(dco(&[commit]).len(), 1, "{message}");
    }
}

#[test]
fn no_commits_fails_dco() {
    assert_eq!(dco(&[]), ["no commits to check; the commits file is empty"]);
    assert_eq!(
        dco(&commits("[]")),
        ["no commits to check; the commits file is empty"]
    );
}

#[test]
fn paginated_commits_are_read_as_one_list() {
    let one = fixture!("commits-signed-off.json");
    let two_pages = format!("{one}\n{one}");
    assert_eq!(commits(&two_pages).len(), 6);
}

#[test]
fn the_commits_file_must_hold_every_commit_of_the_pr() {
    let pr = parse_pull_request(fixture!("event-plain.json")).unwrap();
    assert_eq!(pr.commits, 3);
    let all = commits(fixture!("commits-signed-off.json"));
    assert_eq!(check_commit_count(&pr, &all), Ok(()));
    // One page short, as an unpaginated fetch of a longer PR would be.
    let error = check_commit_count(&pr, &all[..2]).unwrap_err();
    assert!(
        error.contains("holds 2 commits, but the PR has 3"),
        "{error}"
    );
    // The API never lists more than 250, so a PR that size cannot be shown complete.
    let big = PullRequest {
        commits: API_COMMIT_LIMIT,
        ..pr
    };
    let error = check_commit_count(&big, &all).unwrap_err();
    assert!(error.contains("lists at most 250"), "{error}");
}

#[test]
fn an_event_without_a_commit_count_is_refused() {
    let error =
        parse_pull_request(r#"{"title": "docs: x", "body": null, "labels": []}"#).unwrap_err();
    assert!(
        error.contains("`commits`, the PR's commit count, is missing"),
        "{error}"
    );
}

#[test]
fn malformed_commit_payloads_are_refused() {
    for (text, wanted) in [
        ("{}", "expected a JSON array, got an object"),
        (
            "[{\"sha\": \"a\"}]",
            "commits[0]: `commit.author.name` is missing",
        ),
        (
            "[{\"sha\": 1}]",
            "commits[0]: `sha` is a number, not a string",
        ),
        ("[", "not JSON"),
    ] {
        let error = parse_commits(text).unwrap_err();
        assert!(error.contains(wanted), "{text}: {error}");
    }
}

// ---- pr-title: the title ----

#[test]
fn valid_titles_parse() {
    let scopes = scopes();
    for (title, kind, scope, breaking) in [
        ("feat(core): add the widget", "feat", Some("core"), false),
        (
            "build(quality): add the dco check",
            "build",
            Some("quality"),
            false,
        ),
        ("docs: fix a typo", "docs", None, false),
        ("feat!: x", "feat", None, true),
        (
            "refactor(xtask)!: rename the jobs",
            "refactor",
            Some("xtask"),
            true,
        ),
        (
            "fix(core): handle `a: b` in a subject",
            "fix",
            Some("core"),
            false,
        ),
    ] {
        assert_eq!(
            parse_title(title, &scopes),
            Ok(Title {
                kind,
                scope,
                breaking
            }),
            "{title}"
        );
    }
}

#[test]
fn invalid_titles_are_refused_with_a_reason() {
    let scopes = scopes();
    for (title, wanted) in [
        ("add the widget", "has no `: ` after the type"),
        ("feat(core):add the widget", "has no `: ` after the type"),
        ("feat(core):  add", "empty or starts with a space"),
        ("feat(core): ", "empty or starts with a space"),
        ("feat(core): add ", "ends with a space"),
        ("feature(core): add", "`feature` is not an allowed type"),
        ("Feat(core): add", "`Feat` is not an allowed type"),
        ("feat(widgets): add", "`widgets` is not an allowed scope"),
        ("feat(core: add", "does not close its `(scope)`"),
        ("feat(): add", "`` is not an allowed scope"),
        ("Revert \"feat(core): add\"", "is not an allowed type"),
    ] {
        let error = parse_title(title, &scopes).unwrap_err();
        assert!(error.contains(wanted), "{title}: {error}");
    }
}

#[test]
fn the_type_error_lists_the_allowed_types() {
    let error = parse_title("wip: x", &scopes()).unwrap_err();
    assert!(error.ends_with(&TYPES.join(", ")), "{error}");
}

#[test]
fn breaking_footers_are_found_as_the_release_tooling_finds_them() {
    // The parser's footer: the token at the start of a line, then `:` or ` #`.
    assert!(has_breaking_footer(
        "x\n\nBREAKING CHANGE: the format changes"
    ));
    assert!(has_breaking_footer("x\n\nBREAKING CHANGE #42"));
    assert!(has_breaking_footer("x\n\nBREAKING-CHANGE #42"));
    assert!(has_breaking_footer("x\n\nBREAKING CHANGE #1"));
    // Spaces before a separator, which errs towards a marker.
    assert!(has_breaking_footer("x\n\nBREAKING CHANGE  : spaced"));
    assert!(has_breaking_footer("x\n\nBREAKING-CHANGE   #1"));
    // release-please's extra rule: `BREAKING-CHANGE:` anywhere.
    assert!(has_breaking_footer(
        "a `BREAKING-CHANGE:` written inline still counts"
    ));
    // Erring towards a marker: indented or listed tokens.
    assert!(has_breaking_footer("x\n\n  BREAKING CHANGE: indented"));
    assert!(has_breaking_footer(
        "x\n\n* BREAKING CHANGE: in a squash list"
    ));
    assert!(has_breaking_footer("x\n\n- BREAKING CHANGE: in a list"));
    // Not markers: the spaced token inline, lower case, or no separator.
    assert!(!has_breaking_footer(
        "a `BREAKING CHANGE:` footer is described here"
    ));
    assert!(!has_breaking_footer(
        "breaking change: lower case is not a footer"
    ));
    assert!(!has_breaking_footer("breaking-change footer in the body"));
    assert!(!has_breaking_footer("BREAKING CHANGES are listed below"));
}

#[test]
fn a_breaking_header_in_a_commit_override_needs_the_label() {
    let event = r#"{"pull_request": {"title": "feat(core): add the widget", "labels": [], "commits": 3,
        "body": "What this changes.\n\nBEGIN_COMMIT_OVERRIDE\nfeat(core)!: drop the widget\nEND_COMMIT_OVERRIDE\n"}}"#;
    let problems = title_problems(
        event,
        fixture!("commits-signed-off.json"),
        fixture!("label-events-none.http"),
    );
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("a `!` header in the PR body's commit override"),
        "{problems:?}"
    );
    // Outside an override section, a `!` line in the body is prose.
    let event = r#"{"pull_request": {"title": "feat(core): add the widget", "labels": [], "commits": 3,
        "body": "An example title:\nfeat(core)!: drop the widget\n"}}"#;
    let problems = title_problems(
        event,
        fixture!("commits-signed-off.json"),
        fixture!("label-events-none.http"),
    );
    assert_eq!(problems, Vec::<String>::new());
}

// ---- pr-title: the label ----

#[test]
fn a_plain_title_passes_without_any_label() {
    let problems = title_problems(
        fixture!("event-plain.json"),
        fixture!("commits-signed-off.json"),
        fixture!("label-events-none.http"),
    );
    assert_eq!(problems, Vec::<String>::new());
}

#[test]
fn a_breaking_title_without_the_label_fails() {
    let problems = title_problems(
        fixture!("event-breaking-unlabelled.json"),
        fixture!("commits-signed-off.json"),
        fixture!("label-events-none.http"),
    );
    assert_eq!(
        problems,
        [
            "the PR is marked breaking (`!` in the title), which needs `semver-major-approved` \
          applied by a maintainer: the PR does not carry `semver-major-approved`"
        ]
    );
}

#[test]
fn the_same_title_with_a_maintainer_applied_label_passes() {
    let problems = title_problems(
        fixture!("event-breaking-labelled.json"),
        fixture!("commits-signed-off.json"),
        fixture!("label-events-maintainer.http"),
    );
    assert_eq!(problems, Vec::<String>::new());
}

#[test]
fn a_label_applied_by_anyone_but_a_maintainer_does_not_count() {
    for (events, wanted) in [
        (
            fixture!("label-events-agent.http"),
            "applied by `project-review-agent`, an agent identity",
        ),
        (
            fixture!("label-events-app.http"),
            "applied by `project-agent[bot]` through a GitHub App",
        ),
        (
            fixture!("label-events-contributor.http"),
            "applied by `carol`, who is not a maintainer",
        ),
        (
            fixture!("label-events-reapplied-by-agent.http"),
            "applied by `project-review-agent`, an agent identity",
        ),
        (
            fixture!("label-events-none.http"),
            "has no event applying `semver-major-approved`",
        ),
    ] {
        let problems = title_problems(
            fixture!("event-breaking-labelled.json"),
            fixture!("commits-signed-off.json"),
            events,
        );
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(problems[0].contains(wanted), "{problems:?}");
    }
}

#[test]
fn a_breaking_commit_subject_needs_the_label() {
    // The review's probe: a plain, unlabelled PR whose commit subject carries `!`.
    let problems = title_problems(
        fixture!("event-plain.json"),
        fixture!("commits-one-breaking-subject.json"),
        fixture!("label-events-none.http"),
    );
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("marked breaking (`!` in the subject of commit 111111111111)"),
        "{problems:?}"
    );
    // The same subject among several commits is a marker too.
    let mut several = commits(fixture!("commits-signed-off.json"));
    several.extend(commits(fixture!("commits-one-breaking-subject.json")));
    let problems = pr_title(
        &parse_pull_request(fixture!("event-plain.json")).unwrap(),
        &several,
        &parse_label_events(fixture!("label-events-none.http")).unwrap(),
        &roster(),
        &scopes(),
        "0.0.0",
    );
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("`!` in the subject of commit 111111111111"));
    // With a maintainer's label, it passes.
    let problems = title_problems(
        fixture!("event-breaking-labelled.json"),
        fixture!("commits-one-breaking-subject.json"),
        fixture!("label-events-maintainer.http"),
    );
    assert_eq!(problems, Vec::<String>::new());
}

#[test]
fn a_single_commit_subject_must_be_a_valid_title() {
    let problems = title_problems(
        fixture!("event-one-commit.json"),
        fixture!("commits-one-bad-subject.json"),
        fixture!("label-events-none.http"),
    );
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].starts_with(
            "the PR's only commit, 222222222222, lands on `main` under its own subject"
        ),
        "{problems:?}"
    );
    // With more than one commit the PR title is what lands, so a loose subject passes.
    let mut several = commits(fixture!("commits-signed-off.json"));
    several.extend(commits(fixture!("commits-one-bad-subject.json")));
    let problems = pr_title(
        &parse_pull_request(fixture!("event-plain.json")).unwrap(),
        &several,
        &[],
        &roster(),
        &scopes(),
        "0.0.0",
    );
    assert_eq!(problems, Vec::<String>::new());
}

#[test]
fn a_truncated_label_history_fails_closed() {
    // The review's second probe: the reapplied-by-agent history cut to its first page, which
    // ends on the maintainer's `labeled`, as an unpaginated fetch would leave it.
    let error = parse_label_events(fixture!("label-events-truncated.http")).unwrap_err();
    assert!(error.contains("links to a next one"), "{error}");
    // The full history, over two recorded pages, reads every event.
    let events = parse_label_events(fixture!("label-events-reapplied-by-agent.http")).unwrap();
    assert_eq!(events.len(), 3);
}

#[test]
fn a_label_history_must_be_recorded_with_its_headers() {
    let error = parse_label_events("[]").unwrap_err();
    assert!(
        error.contains("record the list with `gh api --include --paginate`"),
        "{error}"
    );
}

#[test]
fn recorded_pages_read_crlf_headers_and_every_page() {
    let two_pages = "HTTP/2.0 200 OK\r\nLink: <https://x/?page=2>; rel=\"next\"\r\n\r\n[1, 2]\n\
                     HTTP/2.0 200 OK\r\nLink: <https://x/?page=1>; rel=\"prev\"\r\n\r\n[3]\n";
    let items = list_items(two_pages, "list", true).unwrap();
    assert_eq!(items.len(), 3);
    let failed = "HTTP/2.0 404 Not Found\r\n\r\n{\"message\": \"Not Found\"}\n";
    let error = list_items(failed, "list", true).unwrap_err();
    assert!(error.contains("page 1 has HTTP status 404"), "{error}");
    let cut = "HTTP/2.0 200 OK\r\nContent-Type: application/json\r\n";
    let error = list_items(cut, "list", true).unwrap_err();
    assert!(error.contains("ends inside its headers"), "{error}");
}

#[test]
fn a_recorded_commits_list_must_be_complete_too() {
    let body = fixture!("commits-signed-off.json");
    assert_eq!(commits(&recorded(body)).len(), 3);
    let truncated =
        format!("HTTP/2.0 200 OK\r\nLink: <https://x/?page=2>; rel=\"next\"\r\n\r\n{body}\n");
    let error = parse_commits(&truncated).unwrap_err();
    assert!(error.contains("links to a next one"), "{error}");
}

#[test]
fn an_app_account_without_the_app_field_is_still_an_agent() {
    let events = parse_label_events(&recorded(
        r#"[{"event": "labeled", "label": {"name": "semver-major-approved"},
             "actor": {"login": "renovate[bot]"}, "created_at": "2026-10-21T09:00:00Z"}]"#,
    ))
    .unwrap();
    let labels = vec![BREAKING_LABEL.to_owned()];
    let why = applied_by_maintainer(BREAKING_LABEL, &labels, &events, &roster()).unwrap_err();
    assert!(why.contains("an agent identity"), "{why}");
}

#[test]
fn a_breaking_footer_needs_the_label_too() {
    // In the body, with a maintainer's label: passes.
    let problems = title_problems(
        fixture!("event-footer-labelled.json"),
        fixture!("commits-signed-off.json"),
        fixture!("label-events-maintainer.http"),
    );
    assert_eq!(problems, Vec::<String>::new());
    // In the body, with an agent's label: fails and names the footer.
    let problems = title_problems(
        fixture!("event-footer-labelled.json"),
        fixture!("commits-signed-off.json"),
        fixture!("label-events-agent.http"),
    );
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("a `BREAKING CHANGE:` footer in the PR body"));
    // In a commit message, with no label: fails and names the commit.
    let commit = r#"[{"sha": "9999999999999999999999999999999999999999", "commit": {
        "author": {"name": "Carol", "email": "carol@example.org"},
        "message": "feat(core): x\n\nBREAKING CHANGE: y\n\nSigned-off-by: Carol <carol@example.org>"}}]"#;
    let problems = title_problems(
        fixture!("event-plain.json"),
        commit,
        fixture!("label-events-none.http"),
    );
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("a `BREAKING CHANGE:` footer in commit 999999999999"),
        "{problems:?}"
    );
}

#[test]
fn a_malformed_breaking_title_still_needs_the_label() {
    let event =
        r#"{"pull_request": {"title": "wip!: x", "body": null, "labels": [], "commits": 3}}"#;
    let problems = title_problems(
        event,
        fixture!("commits-signed-off.json"),
        fixture!("label-events-none.http"),
    );
    assert_eq!(problems.len(), 2, "{problems:?}");
    assert!(problems[0].contains("`wip` is not an allowed type"));
    assert!(problems[1].contains("marked breaking"));
}

#[test]
fn the_label_history_is_replayed_in_time_order() {
    // Out of order on input: the removal is the latest event, so the label does not count.
    let events = parse_label_events(&recorded(
        r#"[
        {"event": "unlabeled", "label": {"name": "semver-major-approved"},
         "actor": {"login": "alice"}, "created_at": "2026-10-21T10:00:00Z"},
        {"event": "labeled", "label": {"name": "semver-major-approved"},
         "actor": {"login": "alice"}, "created_at": "2026-10-21T09:00:00Z"}
    ]"#,
    ))
    .unwrap();
    let labels = vec![BREAKING_LABEL.to_owned()];
    let why = applied_by_maintainer(BREAKING_LABEL, &labels, &events, &roster()).unwrap_err();
    assert!(
        why.contains("ends with `semver-major-approved` removed by `alice`"),
        "{why}"
    );
}

#[test]
fn other_labels_and_events_do_not_count() {
    let events = parse_label_events(&recorded(
        r#"[{"event": "labeled", "label": {"name": "maintainer-approval"},
             "actor": {"login": "alice"}, "created_at": "2026-10-21T09:00:00Z"},
            {"event": "assigned", "actor": {"login": "alice"}, "created_at": "2026-10-21T09:00:00Z"}]"#,
    ))
    .unwrap();
    assert_eq!(events.len(), 1);
    let labels = vec![BREAKING_LABEL.to_owned()];
    assert!(applied_by_maintainer(BREAKING_LABEL, &labels, &events, &roster()).is_err());
}

#[test]
fn a_pull_request_object_reads_like_an_event() {
    let object =
        r#"{"title": "docs(plan): x", "body": "b", "labels": [{"name": "l"}], "commits": 1}"#;
    assert_eq!(
        parse_pull_request(object).unwrap(),
        PullRequest {
            title: "docs(plan): x".into(),
            body: "b".into(),
            labels: vec!["l".into()],
            commits: 1,
        }
    );
    let event = fixture!("event-breaking-labelled.json");
    let pr = parse_pull_request(event).unwrap();
    assert_eq!(pr.title, "feat!: x");
    assert_eq!(pr.labels, [BREAKING_LABEL]);
}

#[test]
fn the_real_workspace_scopes_include_crates_areas_and_others() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let scopes = super::scopes(&root).unwrap();
    for scope in [
        "core",
        "obs",
        "kernels-cuda",
        "xtask",
        "quality",
        "plan",
        "deps",
    ] {
        assert!(
            scopes.iter().any(|s| s == scope),
            "{scope} missing from {scopes:?}"
        );
    }
    // Every title already on main in this form uses one of these scopes.
    for title in [
        "build(quality): add the xtask framework with ci and test job registries",
        "feat(core): add the fidelity metrics and the KL tightness rule",
        "docs(plan): sync tasks to the project board, and add claim and lane rules",
        "docs(product): record 7433 as the default API port in RFC-0054",
        "feat(obs): add the CostTerm key enum",
        "test(core): catch the fidelity mutants the #1492 review found",
        "docs: add the Syntropy design plan and project front matter",
    ] {
        assert!(parse_title(title, &scopes).is_ok(), "{title}");
    }
}

// ---- pr-title: markers release-please finds in commit bodies ----

#[test]
fn a_breaking_type_footer_needs_the_label() {
    // The review's probe (a): `feat!: …` as a footer, which release-please splits out as a
    // breaking commit of its own.
    let problems = title_problems(
        fixture!("event-plain.json"),
        fixture!("commits-footer-breaking-type.json"),
        fixture!("label-events-none.http"),
    );
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("a `!` header in the body of commit 333333333333"),
        "{problems:?}"
    );
}

#[test]
fn a_nested_breaking_commit_needs_the_label() {
    // The review's probe (b): a `BEGIN_NESTED_COMMIT` block holding `feat!: …`.
    let problems = title_problems(
        fixture!("event-plain.json"),
        fixture!("commits-nested-breaking.json"),
        fixture!("label-events-none.http"),
    );
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("a `!` header in the body of commit 444444444444"),
        "{problems:?}"
    );
    // release-please cuts the block out wherever the token stands, even mid-line.
    assert!(has_breaking_header(
        "feat(core): x\n\nSee BEGIN_NESTED_COMMITfeat!: y END_NESTED_COMMIT\n"
    ));
}

#[test]
fn breaking_headers_follow_the_parser_grammar() {
    for line in [
        "feat!: drop the widget",
        "feat(core)!: drop the widget",
        "fix(net-io)!: change the frame",
        "  feat!: indented",
        "* feat!: in a squash list",
        "Note!: any word counts, which errs towards a marker",
        "feat2!: a digit in the type",
        "a-b!: a hyphen in the type",
        "a_b!: an underscore in the type",
        "fé!: a non-ASCII type",
        "`feat!: x` quoted, which the parser reads as a type with a backtick",
        "feat! #42 as a footer with the hash separator",
    ] {
        assert!(is_breaking_header(line), "{line}");
    }
    for line in [
        "feat(core): not breaking",
        "Note: plain prose",
        "This is great!: a space before the bang",
        "| `feat!: x` in a table",
        "feat(core!: unclosed scope",
        "!: no word",
        "feat (core)!: a space before the scope",
        "feat\t!: a tab before the bang",
    ] {
        assert!(!is_breaking_header(line), "{line}");
    }
}

#[test]
fn a_major_release_as_needs_the_label() {
    // `Release-As: 1.0.0` from 0.0.0 is a major release (ADR-0621 decision 3).
    let problems = title_problems(
        fixture!("event-plain.json"),
        fixture!("commits-release-as-major.json"),
        fixture!("label-events-none.http"),
    );
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains(
            "a `Release-As: 1.0.0` footer in commit 555555555555, a major release from 0.0.0"
        ),
        "{problems:?}"
    );
    // With a maintainer's label, it passes.
    let problems = title_problems(
        fixture!("event-breaking-labelled.json"),
        fixture!("commits-release-as-major.json"),
        fixture!("label-events-maintainer.http"),
    );
    assert_eq!(problems, Vec::<String>::new());
}

#[test]
fn release_as_is_a_marker_only_for_a_new_major_version() {
    let pr = parse_pull_request(fixture!("event-plain.json")).unwrap();
    let with = |message: &str, version: &str| {
        let commit = Commit {
            sha: "6".repeat(40),
            author_name: "Carol".into(),
            author_email: "carol@example.org".into(),
            message: format!(
                "feat(core): x\n\n{message}\nSigned-off-by: Carol <carol@example.org>"
            ),
        };
        pr_title(&pr, &[commit], &[], &roster(), &scopes(), version)
    };
    // A minor or patch pin is not a major release; the token is read without regard to case.
    assert_eq!(with("Release-As: 0.2.0", "0.0.0"), Vec::<String>::new());
    assert_eq!(with("release-as: 1.9.0", "1.4.0"), Vec::<String>::new());
    assert_eq!(with("RELEASE-AS: v2.0.0", "1.4.0").len(), 1);
    assert_eq!(with("release-as: 1.0.0-rc.1", "0.0.0").len(), 1);
    // A value that is not a version fails closed.
    assert_eq!(with("Release-As: next", "0.0.0").len(), 1);
    // The parser's other footer separator, and spaces before a separator.
    assert_eq!(with("Release-As #2.0.0", "1.4.0").len(), 1);
    assert_eq!(with("release-as : 2.0.0", "1.4.0").len(), 1);
    assert_eq!(with("Release-As #1.5.0", "1.4.0"), Vec::<String>::new());
    // Not a footer: the token mid-line, or a longer token.
    assert_eq!(
        with("We discussed Release-As: 9.0.0 here", "0.0.0"),
        Vec::<String>::new()
    );
    assert_eq!(with("Release-Asap: 9.0.0", "0.0.0"), Vec::<String>::new());
    // In the PR body, as release-please could read it from an override.
    let body_pr = PullRequest {
        body: "Release-As: 3.0.0\n".into(),
        ..pr
    };
    let commit = commits(fixture!("commits-one-breaking-subject.json"))
        .into_iter()
        .map(|c| Commit {
            message: "feat(core): x\n".into(),
            ..c
        })
        .collect::<Vec<_>>();
    let problems = pr_title(&body_pr, &commit, &[], &roster(), &scopes(), "0.0.0");
    assert!(
        problems
            .iter()
            .any(|p| p.contains("a `Release-As: 3.0.0` footer in the PR body")),
        "{problems:?}"
    );
}

#[test]
fn versions_parse_as_release_please_writes_them() {
    assert_eq!(major("0.0.0"), Some(0));
    assert_eq!(major("v1.2.3"), Some(1));
    assert_eq!(major("2.0.0-rc.1"), Some(2));
    assert_eq!(major("1.2"), None);
    assert_eq!(major("next"), None);
    let pr = parse_pull_request(fixture!("event-plain.json")).unwrap();
    let problems = pr_title(&pr, &[], &[], &roster(), &scopes(), "unknown");
    assert_eq!(
        problems,
        ["version.txt holds `unknown`, which is not a version"]
    );
}

#[test]
fn text_spliced_around_a_nested_commit_is_read_as_release_please_joins_it() {
    // `splitMessages()` joins what follows a block onto the outer message, so `BREAKING` before
    // a block and ` CHANGE: …` after it become one footer.
    let problems = title_problems(
        fixture!("event-plain.json"),
        fixture!("commits-nested-splice.json"),
        fixture!("label-events-none.http"),
    );
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("a `BREAKING CHANGE:` footer in commit 777777777777"),
        "{problems:?}"
    );
    // The same splice builds a `Release-As` footer.
    let spliced = "feat(core): x\n\nRelease\
                   BEGIN_NESTED_COMMIT\nfix(core): y\nEND_NESTED_COMMIT-As: 9.0.0\n";
    assert_eq!(release_as_values(spliced), ["9.0.0"]);
    // And the messages are the original, the rebuilt outer one, and each block.
    let messages = release_messages("a\nBEGIN_NESTED_COMMITb\nEND_NESTED_COMMITc");
    assert_eq!(
        messages,
        ["a\nBEGIN_NESTED_COMMITb\nEND_NESTED_COMMITc", "a\nc", "b\n"]
    );
    assert_eq!(release_messages("no nested commit"), ["no nested commit"]);
}
