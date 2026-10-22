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

fn comment(login: &str, body: &str, edited: bool) -> Comment {
    Comment {
        login: login.to_owned(),
        body: body.to_owned(),
        created_at: "2026-10-21T10:00:00Z".to_owned(),
        updated_at: if edited {
            "2026-10-21T10:05:00Z"
        } else {
            "2026-10-21T10:00:00Z"
        }
        .to_owned(),
        via_app: false,
    }
}

const HEAD: &str = "1111111111111111111111111111111111111111";
const OLD: &str = "0000000000000000000000000000000000000000";

#[test]
fn a_comment_approves_only_the_commit_it_names_from_a_maintainer_unedited() {
    let approve = |sha: &str| format!("{APPROVAL} {sha}");
    let decide = |comments: &[Comment]| approving_comment(comments, HEAD, &roster());
    assert_eq!(
        decide(&[comment("alice", &approve(HEAD), false)]),
        Ok("alice".into())
    );
    // Among other text, on its own line.
    let body = format!("Checked the publisher.\n  {}  \nThanks.", approve(HEAD));
    assert_eq!(decide(&[comment("bob", &body, false)]), Ok("bob".into()));
    // Stale, edited, a contributor's, an agent's, through an App.
    let stale = decide(&[comment("alice", &approve(OLD), false)]).unwrap_err();
    assert!(stale.contains("not the head commit"), "{stale}");
    let edited = decide(&[comment("alice", &approve(HEAD), true)]).unwrap_err();
    assert!(edited.contains("edited theirs"), "{edited}");
    let carol = decide(&[comment("carol", &approve(HEAD), false)]).unwrap_err();
    assert!(
        carol.contains("`carol`, who posted one, is not a maintainer"),
        "{carol}"
    );
    let agent = decide(&[comment("project-review-agent", &approve(HEAD), false)]).unwrap_err();
    assert!(agent.contains("is an agent identity"), "{agent}");
    let mut via_app = comment("alice", &approve(HEAD), false);
    via_app.via_app = true;
    assert!(
        decide(&[via_app])
            .unwrap_err()
            .contains("through a GitHub App")
    );
    // A refused comment does not hide a good one.
    assert_eq!(
        decide(&[
            comment("carol", &approve(HEAD), false),
            comment("alice", &approve(HEAD), false)
        ]),
        Ok("alice".into())
    );
    // No approval at all names the form to use.
    assert_eq!(
        decide(&[comment("alice", "Looks good.", false)]).unwrap_err(),
        format!("no maintainer has commented `{APPROVAL} {HEAD}`")
    );
}

#[test]
fn only_the_exact_line_form_names_a_commit() {
    // Letters too, so an upper-case copy differs.
    let sha = "0123456789abcdef0123456789abcdef01234567";
    assert_eq!(approved_shas(&format!("{APPROVAL} {sha}")), [sha]);
    for not_it in [
        format!("LGTM, {APPROVAL} {sha}"),
        format!("{APPROVAL} {}", &sha[..7]),
        format!("{APPROVAL} {}", sha.to_uppercase()),
        format!("{APPROVAL}  {sha}"),
        format!("{APPROVAL} {sha} and more"),
        format!("{APPROVAL}:{sha}"),
        format!("dep-diff approves {sha}"),
    ] {
        assert!(
            approved_shas(&not_it).is_empty(),
            "`{not_it}` was read as an approval"
        );
    }
}

#[test]
fn comments_are_read_from_every_page() {
    let text = r#"[{"user":{"login":"alice"},"body":"a","created_at":"t","updated_at":"t","performed_via_github_app":null}]
[{"user":{"login":"bob"},"body":null,"created_at":"t","updated_at":"u","performed_via_github_app":{"slug":"x"}}]"#;
    let comments = parse_comments(text).unwrap();
    assert_eq!(comments.len(), 2);
    assert_eq!(comments[1].body, "");
    assert!(comments[1].via_app);
    assert!(parse_comments(r#"{"message":"Not Found"}"#).is_err());
    assert!(parse_comments(r#"[{"body":"x"}]"#).is_err());
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
fn configs_are_found_at_any_depth_except_in_the_head_inside_the_base_and_target() {
    let root =
        std::env::temp_dir().join(format!("syntropy-dep-diff-configs-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    for dir in [
        ".cargo",
        "crates/a/.cargo",
        "pr/.cargo",
        "target/.cargo",
        "tests/fixture/.cargo",
    ] {
        std::fs::create_dir_all(root.join(dir)).unwrap();
        std::fs::write(root.join(dir).join("config.toml"), "").unwrap();
    }
    // A fixture workspace of its own is read too: telling it apart would rest on its text.
    std::fs::write(
        root.join("tests/fixture/Cargo.toml"),
        "[workspace]\nmembers = []\n",
    )
    .unwrap();
    let found = cargo_configs(&root, Some(&root.join("pr"))).unwrap();
    assert_eq!(
        found.configs,
        [
            ".cargo/config.toml",
            "crates/a/.cargo/config.toml",
            "tests/fixture/.cargo/config.toml"
        ]
    );
    let found = cargo_configs(&root, None).unwrap();
    assert_eq!(
        found.configs,
        [
            ".cargo/config.toml",
            "crates/a/.cargo/config.toml",
            "pr/.cargo/config.toml",
            "tests/fixture/.cargo/config.toml"
        ]
    );
    assert!(found.links.is_empty());
    // A symbolic link named `.cargo`, or to a directory, is reported, not followed.
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(root.join("pr/.cargo"), root.join("crates/.cargo")).unwrap();
        std::os::unix::fs::symlink(root.join("pr"), root.join("crates/b")).unwrap();
        let found = cargo_configs(&root, Some(&root.join("pr"))).unwrap();
        assert_eq!(found.links, ["crates/.cargo", "crates/b"]);
    }
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn a_key_spelt_so_the_check_cannot_read_it_counts_as_a_setting() {
    for text in [
        "['patch'.crates-io]\nserde = { git = \"x\" }\n",
        "[\"pa\\u0074ch\".crates-io]\nserde = { git = \"x\" }\n",
        "'source'.crates-io.replace-with = \"m\"\n",
        "[pa\\tch]\nx = 1\n",
    ] {
        assert!(
            !source_settings(text, CONFIG_TABLES).is_empty(),
            "{text} was not counted"
        );
    }
    // A multi-line string could hide a header, so the file is compared whole.
    let hidden = "[patch.crates-io]\nnote = \"\"\"\n[build]\n\"\"\"\nserde = { git = \"x\" }\n";
    let settings: Vec<String> = source_settings(hidden, CONFIG_TABLES).into_iter().collect();
    assert_eq!(settings.len(), 1, "{settings:?}");
    assert!(settings[0].starts_with("the whole file, which holds a multi-line string"));
    let changed = hidden.replace("\"x\"", "\"y\"");
    assert_ne!(
        source_settings(&changed, CONFIG_TABLES),
        source_settings(hidden, CONFIG_TABLES)
    );
    // A configuration that uses `include`, or holds a multi-line string, cannot be read with
    // certainty, in any spelling of the key.
    for text in [
        "[[include]]\npath = \"more.toml\"\n",
        "include = [\"more.toml\"]\n",
        "'include' = [\"more.toml\"]\n",
        "[alias]\nx = \"\"\"\n[build]\n\"\"\"\n",
    ] {
        assert!(unreadable(text).is_some(), "{text} was read as certain");
    }
    assert_eq!(
        unreadable("[build]\njobs = 4\n[alias]\nb = \"build\"\n"),
        None
    );
}
