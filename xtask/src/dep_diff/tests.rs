// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

fn roster() -> Roster {
    Roster::parse("maintainers = [\"alice\", \"bob\"]\nagents = [\"project-review-agent\"]")
        .unwrap()
}

/// A `cargo metadata --no-deps` document with one package and `deps` as
/// `(name, kind, source)`; a `None` source is a path dependency.
fn metadata(package: &str, deps: &[(&str, Option<&str>, Option<&str>)]) -> String {
    let deps: Vec<Value> = deps
        .iter()
        .map(|(name, kind, source)| {
            serde_json::json!({ "name": name, "kind": kind, "source": source, "rename": null })
        })
        .collect();
    serde_json::json!({ "packages": [{ "name": package, "dependencies": deps }] }).to_string()
}

#[test]
fn declared_dependencies_are_external_crates_by_name_and_source() {
    let text = metadata(
        "syntropy-x",
        &[
            ("serde", None, Some(CRATES_IO)),
            ("serde", Some("dev"), Some(CRATES_IO)),
            ("proptest", Some("dev"), Some(CRATES_IO)),
            ("syntropy-core", None, None),
            (
                "cc",
                Some("build"),
                Some("git+https://example.invalid/cc#abc"),
            ),
        ],
    );
    let crates = declared(&text).unwrap();
    let shown: Vec<(String, Vec<String>)> = crates
        .iter()
        .map(|(krate, users)| (krate.name.clone(), users.clone()))
        .collect();
    assert_eq!(
        shown,
        [
            ("cc".to_owned(), vec!["syntropy-x (build)".to_owned()]),
            ("proptest".to_owned(), vec!["syntropy-x (dev)".to_owned()]),
            (
                "serde".to_owned(),
                vec![
                    "syntropy-x (normal)".to_owned(),
                    "syntropy-x (dev)".to_owned()
                ]
            ),
        ]
    );
    assert!(declared("{}").is_err());
}

#[test]
fn only_a_name_or_source_the_base_lacks_is_new() {
    let base = declared(&metadata("a", &[("serde", None, Some(CRATES_IO))])).unwrap();
    // The same crate in another member, at another version requirement, is not new.
    let head = declared(&metadata("b", &[("serde", Some("dev"), Some(CRATES_IO))])).unwrap();
    let none = BTreeSet::new();
    assert!(compare(&base, &head, &none, &none).is_empty());
    // A new name is; so is the same name from a git fork.
    let head = declared(&metadata(
        "a",
        &[
            (
                "serde",
                None,
                Some("git+https://example.invalid/serde-fork#1"),
            ),
            ("tokio", None, Some(CRATES_IO)),
        ],
    ))
    .unwrap();
    let new = compare(&base, &head, &none, &none);
    let names: Vec<&str> = new.crates.keys().map(|k| k.name.as_str()).collect();
    assert_eq!(names, ["serde", "tokio"]);
    // A vendored component is new by its upstream.
    let base_v: BTreeSet<String> = ["https://example.invalid/a".to_owned()].into();
    let head_v: BTreeSet<String> = [
        "https://example.invalid/a".to_owned(),
        "https://example.invalid/b".to_owned(),
    ]
    .into();
    let new = compare(&base, &base, &base_v, &head_v);
    assert_eq!(new.vendored, ["https://example.invalid/b"]);
    assert_eq!(
        new.lines(),
        ["new vendored component from https://example.invalid/b"]
    );
}

fn review(login: &str, state: &str, commit: &str) -> Review {
    Review {
        login: login.to_owned(),
        state: state.to_owned(),
        commit: commit.to_owned(),
    }
}

#[test]
fn an_approval_counts_from_a_maintainer_on_the_head_commit_only() {
    let head = "h";
    let approved = |reviews: &[Review]| approving_review(reviews, head, &roster());
    assert_eq!(
        approved(&[review("alice", "APPROVED", "h")]),
        Some("alice".into())
    );
    // On an earlier commit, by a contributor, or by an agent.
    assert_eq!(approved(&[review("alice", "APPROVED", "old")]), None);
    assert_eq!(approved(&[review("carol", "APPROVED", "h")]), None);
    assert_eq!(
        approved(&[review("project-review-agent", "APPROVED", "h")]),
        None
    );
    // A later comment leaves the approval standing; a later change request or a dismissal ends it.
    assert_eq!(
        approved(&[
            review("alice", "APPROVED", "h"),
            review("alice", "COMMENTED", "h")
        ]),
        Some("alice".into())
    );
    assert_eq!(
        approved(&[
            review("alice", "APPROVED", "h"),
            review("alice", "CHANGES_REQUESTED", "h")
        ]),
        None
    );
    assert_eq!(approved(&[review("alice", "DISMISSED", "h")]), None);
    // Another maintainer's standing approval still counts.
    assert_eq!(
        approved(&[
            review("alice", "CHANGES_REQUESTED", "h"),
            review("bob", "APPROVED", "h")
        ]),
        Some("bob".into())
    );
}

#[test]
fn reviews_are_read_from_every_page() {
    let text = r#"[{"user":{"login":"alice"},"state":"COMMENTED","commit_id":"a"}]
[{"user":{"login":"bob"},"state":"APPROVED","commit_id":"b"},{"user":{"login":"carol"},"state":"PENDING"}]"#;
    let reviews = parse_reviews(text).unwrap();
    assert_eq!(
        reviews,
        [
            review("alice", "COMMENTED", "a"),
            review("bob", "APPROVED", "b"),
            review("carol", "PENDING", "")
        ]
    );
    assert!(parse_reviews(r#"{"message":"Not Found"}"#).is_err());
    assert!(parse_reviews(r#"[{"state":"APPROVED"}]"#).is_err());
}

#[test]
fn the_registry_line_names_the_first_publish_and_its_publisher() {
    let crate_json = r#"{"crate":{"created_at":"2020-01-02T03:04:05Z"},"versions":[
        {"num":"2.0.0","created_at":"2024-01-01T00:00:00Z","published_by":{"login":"later"}},
        {"num":"0.1.0","created_at":"2020-01-02T03:04:05Z","published_by":{"login":"first"}}]}"#;
    let owners = r#"{"users":[{"login":"first"},{"login":"github:org:team"}]}"#;
    assert_eq!(
        describe("x", crate_json, owners).unwrap(),
        "`x` on crates.io: first published 2020-01-02, by `first` (version 0.1.0); owners \
         `first`, `github:org:team`"
    );
    let unrecorded = r#"{"crate":{"created_at":"2016-01-01T00:00:00Z"},"versions":[
        {"num":"0.1.0","created_at":"2016-01-01T00:00:00Z","published_by":null}]}"#;
    assert_eq!(
        describe("y", unrecorded, r#"{"users":[]}"#).unwrap(),
        "`y` on crates.io: first published 2016-01-01, no publisher recorded; owners none listed"
    );
    assert!(describe("z", "{}", "{}").is_err());
}

/// Answers held in memory.
struct Answers(Vec<(&'static str, Lookup)>);

impl Registry for Answers {
    fn lookup(&mut self, name: &str) -> Result<Lookup, String> {
        let at = self
            .0
            .iter()
            .position(|(n, _)| *n == name)
            .ok_or_else(|| format!("unexpected lookup of `{name}`"))?;
        Ok(self.0.remove(at).1)
    }
}

fn new_crates(crates: &[(&str, &str)]) -> NewDependencies {
    NewDependencies {
        crates: crates
            .iter()
            .map(|(name, source)| {
                (
                    Crate {
                        name: (*name).to_owned(),
                        source: (*source).to_owned(),
                    },
                    vec!["a (normal)".to_owned()],
                )
            })
            .collect(),
        vendored: Vec::new(),
        sources: Vec::new(),
    }
}

#[test]
fn the_registry_check_fails_on_a_name_that_does_not_resolve_and_other_registries() {
    let found = Lookup::Found {
        crate_json: r#"{"crate":{"created_at":"2020-01-01T00:00:00Z"},"versions":[]}"#.into(),
        owners_json: r#"{"users":[]}"#.into(),
    };
    let mut answers = Answers(vec![("real", found), ("invented", Lookup::Missing)]);
    let problems = check_registry(
        &new_crates(&[
            ("real", CRATES_IO),
            ("invented", CRATES_IO),
            ("forked", "git+https://example.invalid/forked#1"),
            ("elsewhere", "registry+https://example.invalid/index"),
            ("bad/name", CRATES_IO),
        ]),
        &mut answers,
    );
    assert_eq!(
        problems,
        [
            "`bad/name` is not a crate name crates.io accepts",
            "`elsewhere` comes from `registry+https://example.invalid/index`, which is not \
             crates.io; the registry check covers crates.io only",
            "`invented` does not resolve on crates.io: no crate of that name exists, so the name \
             is misspelt, invented or not yet published (ADR-0624 decision 5)",
        ]
    );
}

#[test]
fn crate_names_are_checked_before_they_reach_a_url() {
    for good in ["serde", "serde_json", "proc-macro2", "a"] {
        assert!(valid_name(good), "{good}");
    }
    for bad in ["", "1abc", "a/b", "a?b", "../x", "a b", &"x".repeat(65)] {
        assert!(!valid_name(bad), "{bad}");
    }
}

#[test]
fn source_settings_are_the_patch_replace_and_source_tables() {
    let manifest = "\
[workspace]
members = [\"crates/*\"]

[dependencies]
patch = \"1\" # a crate named patch, not a table

[patch.crates-io]
serde = { git = \"https://example.invalid/serde\" } # a fork

[replace]
\"foo:0.1.0\" = { path = \"../foo\" }
";
    let settings: Vec<String> = source_settings(manifest, MANIFEST_TABLES)
        .into_iter()
        .collect();
    assert_eq!(
        settings,
        [
            "[patch.crates-io]",
            "[patch.crates-io] serde = { git = \"https://example.invalid/serde\" }",
            "[replace]",
            "[replace] \"foo:0.1.0\" = { path = \"../foo\" }",
        ]
    );
    // Dotted keys at the top of a config file count too; other tables do not.
    let config = "source.crates-io.replace-with = \"mirror\"\n[build]\nrustflags = []\n\
                  [registries.mine]\nindex = \"https://example.invalid/index\"\n";
    let settings: Vec<String> = source_settings(config, CONFIG_TABLES).into_iter().collect();
    assert_eq!(
        settings,
        [
            "[registries.mine]",
            "[registries.mine] index = \"https://example.invalid/index\"",
            "source.crates-io.replace-with = \"mirror\"",
        ]
    );
    assert!(source_settings("[build]\njobs = 4\n", CONFIG_TABLES).is_empty());
}

#[test]
fn configs_are_found_at_any_depth_except_in_the_head_inside_the_base() {
    let root =
        std::env::temp_dir().join(format!("syntropy-dep-diff-configs-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    for dir in [".cargo", "crates/a/.cargo", "pr/.cargo", "target/.cargo"] {
        std::fs::create_dir_all(root.join(dir)).unwrap();
        std::fs::write(root.join(dir).join("config.toml"), "").unwrap();
    }
    let found = cargo_configs(&root, Some(&root.join("pr"))).unwrap();
    assert_eq!(found, [".cargo/config.toml", "crates/a/.cargo/config.toml"]);
    let found = cargo_configs(&root, None).unwrap();
    assert_eq!(
        found,
        [
            ".cargo/config.toml",
            "crates/a/.cargo/config.toml",
            "pr/.cargo/config.toml"
        ]
    );
    std::fs::remove_dir_all(&root).unwrap();
}
