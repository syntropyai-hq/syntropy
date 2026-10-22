// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

fn roster() -> Roster {
    Roster::parse("maintainers = [\"alice\", \"bob\"]\nagents = [\"project-review-agent\"]")
        .unwrap()
}

/// A `cargo metadata --no-deps` document for a workspace at `/w`, with one package and `deps` as
/// `(name, kind, source)`. A `None` source is a path dependency on a member at `/w/<name>`,
/// which the document lists too.
fn metadata(package: &str, deps: &[(&str, Option<&str>, Option<&str>)]) -> String {
    let mut packages = Vec::new();
    let deps: Vec<Value> = deps
        .iter()
        .map(|(name, kind, source)| match source {
            Some(source) => {
                serde_json::json!({ "name": name, "kind": kind, "source": source, "rename": null })
            }
            None => {
                packages.push(member(name, vec![]));
                serde_json::json!({ "name": name, "kind": kind, "source": null, "rename": null,
                                    "path": format!("/w/{name}") })
            }
        })
        .collect();
    packages.insert(0, member(package, deps));
    serde_json::json!({ "workspace_root": "/w", "packages": packages }).to_string()
}

/// What a `cargo metadata --no-deps` document's members declare.
fn declared(metadata: &str) -> Result<BTreeMap<Crate, Vec<String>>, String> {
    workspace(metadata).map(|workspace| workspace.crates)
}

/// A path dependency named `name` on the directory `path`, as `cargo metadata` lists it.
fn path_dependency(name: &str, path: &str) -> Value {
    serde_json::json!({ "name": name, "kind": null, "source": null, "path": path })
}

/// A workspace at `/w` whose member `a` depends on the non-member path crate `v` at
/// `/w/vendor/v`, and whose member `b` declares nothing.
fn with_path_crate() -> Workspace {
    workspace(
        &serde_json::json!({
            "workspace_root": "/w",
            "packages": [member("a", vec![path_dependency("v", "/w/vendor/v")]), member("b", vec![])],
        })
        .to_string(),
    )
    .unwrap()
}

/// Each crate of `workspace`, as `<name> <source> <- <users>`.
fn shown(workspace: &Workspace) -> Vec<String> {
    workspace
        .crates
        .iter()
        .map(|(krate, users)| format!("{} {} <- {}", krate.name, krate.source, users.join("; ")))
        .collect()
}

/// A member at `/w/<name>`, as `cargo metadata` lists it.
fn member(name: &str, dependencies: Vec<Value>) -> Value {
    serde_json::json!({
        "name": name,
        "manifest_path": format!("/w/{name}/Cargo.toml"),
        "dependencies": dependencies,
    })
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
fn a_path_dependency_outside_the_members_is_external() {
    let path_dep = |path: &str| serde_json::json!({ "name": "fixture-v", "kind": null, "source": null, "path": path });
    let document = |path: &str| {
        serde_json::json!({
            "workspace_root": "/w",
            "packages": [member("a", vec![path_dep(path)])],
        })
        .to_string()
    };
    for (path, dir) in [
        // A directory the workspace excludes.
        ("/w/vendor/fixture-v", "vendor/fixture-v"),
        // Beside the workspace root, and further out.
        ("/fixture-v", "../fixture-v"),
        ("/elsewhere/fixture-v", "../elsewhere/fixture-v"),
    ] {
        let crates = declared(&document(path)).unwrap();
        let sources: Vec<&str> = crates.keys().map(|k| k.source.as_str()).collect();
        assert_eq!(sources, [format!("path+{dir}")], "{path}");
    }
    // The same directory in two checkouts compares equal, so it is new only once.
    let at = |root: &str| {
        serde_json::json!({
            "workspace_root": root,
            "packages": [{
                "name": "a",
                "manifest_path": format!("{root}/a/Cargo.toml"),
                "dependencies": [path_dep(&format!("{root}/vendor/fixture-v"))],
            }],
        })
        .to_string()
    };
    let base = declared(&at("/checkout/base")).unwrap();
    let head = declared(&at("/checkout/head")).unwrap();
    let none = BTreeSet::new();
    assert!(compare(&base, &head, &none, &none).is_empty());
    let new = compare(&BTreeMap::new(), &head, &none, &none);
    assert_eq!(
        new.lines(),
        [
            "new crate `fixture-v` from the directory `vendor/fixture-v`, which is not a workspace \
          member, in a (normal)"
        ]
    );
    // A path dependency with no path, or a document without a root, fails closed.
    let no_path = serde_json::json!({
        "workspace_root": "/w",
        "packages": [member("a", vec![serde_json::json!({ "name": "x", "source": null })])],
    })
    .to_string();
    assert!(
        declared(&no_path)
            .unwrap_err()
            .contains("neither a source nor a path")
    );
    let no_root = serde_json::json!({ "packages": [] }).to_string();
    assert!(declared(&no_root).is_err());
}

#[test]
fn a_dependency_of_one_platform_names_it() {
    let dependency = |target: Value| {
        serde_json::json!({ "name": "windows-sys", "kind": null, "source": CRATES_IO,
                            "target": target })
    };
    let text = serde_json::json!({
        "workspace_root": "/w",
        "packages": [member("a", vec![
            dependency(Value::Null),
            dependency(serde_json::json!("cfg(windows)")),
            dependency(serde_json::json!("x86_64-pc-windows-gnu")),
        ])],
    })
    .to_string();
    let crates = declared(&text).unwrap();
    assert_eq!(
        crates.values().collect::<Vec<_>>(),
        [&vec![
            "a (normal)".to_owned(),
            "a (normal, cfg(windows))".to_owned(),
            "a (normal, x86_64-pc-windows-gnu)".to_owned(),
        ]]
    );
}

#[test]
fn a_non_member_path_crate_counts_what_its_manifest_declares() {
    let manifests = BTreeMap::from([
        (
            PathBuf::from("/w/vendor/v/Cargo.toml"),
            r#"[package]
name = "v"

[dependencies]
serde = "1"
alias = { package = "itoa", version = "1", optional = true }
member-b = { path = "../../b" }
inner = { path = "./../inner" }

[dev-dependencies]
proptest = "1"

[build-dependencies]
cc = { git = "https://example.invalid/cc", rev = "abc" }

[target.'cfg(unix)'.dependencies]
libc = { version = "0.2", registry = "mirror" }
"#,
        ),
        (
            PathBuf::from("/w/vendor/inner/Cargo.toml"),
            r#"[package]
name = "inner"

[dependencies]
foo = { git = "https://example.invalid/foo", branch = "dev" }
v = { path = "../v" }
"#,
        ),
    ]);
    let mut workspace = with_path_crate();
    let mut read = Vec::new();
    workspace
        .read_path_crates(|manifest| {
            read.push(manifest.to_path_buf());
            manifests
                .get(manifest)
                .map(|text| Manifest::Text(text.to_string()))
                .ok_or_else(|| format!("{} is not a fixture", manifest.display()))
        })
        .unwrap();
    // Each manifest is read once, though `inner` depends on `v` again; the path dependency on
    // the member `b` is a workspace edge.
    assert_eq!(read.len(), 2, "{read:?}");
    assert_eq!(
        shown(&workspace),
        [
            "cc git+https://example.invalid/cc?rev=abc <- v at vendor/v (build)",
            "foo git+https://example.invalid/foo?branch=dev <- inner at vendor/inner (normal)",
            "inner path+vendor/inner <- v at vendor/v (normal)",
            &format!("itoa {CRATES_IO} <- v at vendor/v (normal)"),
            "libc registry+mirror <- v at vendor/v (normal, cfg(unix))",
            &format!("proptest {CRATES_IO} <- v at vendor/v (dev)"),
            &format!("serde {CRATES_IO} <- v at vendor/v (normal)"),
            "v path+vendor/v <- a (normal); inner at vendor/inner (normal)",
        ]
    );
    assert!(
        workspace.unfollowed.is_empty(),
        "{:?}",
        workspace.unfollowed
    );

    // So a dependency added inside an already-declared directory is new.
    let mut base = with_path_crate();
    base.read_path_crates(|_| {
        Ok(Manifest::Text(
            "[package]\nname = \"v\"\n[dependencies]\nserde = \"1\"\n".into(),
        ))
    })
    .unwrap();
    let mut head = with_path_crate();
    head.read_path_crates(|_| {
        Ok(Manifest::Text(
            "[package]\nname = \"v\"\n[dependencies]\nserde = \"1.0.200\"\nproptest = \"1\"\n"
                .into(),
        ))
    })
    .unwrap();
    let none = BTreeSet::new();
    assert_eq!(
        compare(&base.crates, &head.crates, &none, &none).lines(),
        ["new crate `proptest` from crates.io, in v at vendor/v (normal)"]
    );
}

#[test]
fn what_the_check_cannot_follow_counts_on_every_pr() {
    // A path crate outside the workspace root is not read.
    let mut outside = workspace(
        &serde_json::json!({
            "workspace_root": "/w",
            "packages": [member("a", vec![path_dependency("v", "/elsewhere/v")])],
        })
        .to_string(),
    )
    .unwrap();
    outside
        .read_path_crates(|manifest| panic!("read {}", manifest.display()))
        .unwrap();
    assert_eq!(
        outside.unfollowed,
        [
            "`../elsewhere/v`, a path dependency of a (normal), compared on every PR because it \
             is outside the workspace root, where this check does not read"
        ]
    );
    // Nor is a dependency inherited from a workspace the check does not read.
    let mut inherited = with_path_crate();
    inherited
        .read_path_crates(|_| {
            Ok(Manifest::Text(
                "[package]\nname = \"v\"\n[dependencies]\nserde = { workspace = true }\n".into(),
            ))
        })
        .unwrap();
    assert_eq!(
        inherited.unfollowed,
        [
            "`vendor/v/Cargo.toml`, compared on every PR because its dependency `serde` sets \
             `workspace = true`, and this check does not read the workspace it inherits from"
        ]
    );
    // A manifest that is missing or that cargo could not read fails the check.
    let error = with_path_crate()
        .read_path_crates(|_| Err("No such file or directory".into()))
        .unwrap_err();
    assert_eq!(
        error,
        "`vendor/v/Cargo.toml`, a path dependency that is not a member: No such file or directory"
    );
    for bad in [
        "not = = toml",
        "[package]\nversion = \"1\"\n",
        "[package]\nname = \"v\"\n[dependencies]\nx = 1\n",
        "[package]\nname = \"v\"\n[dependencies]\nx = { path = 1 }\n",
        "dependencies = 1\n[package]\nname = \"v\"\n",
        "[package]\nname = \"v\"\n[target]\nx = 1\n",
        // What would be printed must be printable, and a name one cargo accepts.
        "[package]\nname = \"v x\"\n",
        "[package]\nname = \"v\"\n[dependencies]\n\"a\\nb\" = \"1\"\n",
        "[package]\nname = \"v\"\n[dependencies]\nx = { package = \"a/b\", version = \"1\" }\n",
        "[package]\nname = \"v\"\n[dependencies]\nx = { path = \"a\\nb\" }\n",
        "[package]\nname = \"v\"\n[target.\"cfg(unix)\\n::error::x\".dependencies]\nx = \"1\"\n",
        // A git source that could spell an approved one cargo reads otherwise.
        "[package]\nname = \"v\"\n[dependencies]\nx = { git = \"https://e.invalid/x?branch=main\" }\n",
        "[package]\nname = \"v\"\n[dependencies]\nx = { git = \"https://e.invalid/x#abc\" }\n",
        "[package]\nname = \"v\"\n[dependencies]\nx = { git = \"https://e.invalid/x\", branch = \"a\", rev = \"b\" }\n",
    ] {
        let result = with_path_crate().read_path_crates(|_| Ok(Manifest::Text(bad.into())));
        assert!(
            result
                .as_ref()
                .is_err_and(|e| e.starts_with("`vendor/v/Cargo.toml`: ")),
            "{bad:?}: {result:?}"
        );
    }
}

#[test]
fn a_manifest_error_names_a_line_and_column_never_the_text() {
    // As if a link had led the check to a file of secrets: the parser's own message would
    // quote the line.
    let error = with_path_crate()
        .read_path_crates(|_| Ok(Manifest::Text("# x\nSECRET_TOKEN=hunter2-abc123\n".into())))
        .unwrap_err();
    assert_eq!(
        error,
        "`vendor/v/Cargo.toml`: not a manifest cargo can read: a TOML error at line 2, column 14"
    );
    assert!(!error.contains("hunter2"), "{error}");
    assert_eq!(line_and_column("ab\ncd", 4), (2, 2));
    assert_eq!(line_and_column("ab", 9), (1, 3));
}

#[test]
fn a_manifest_behind_a_link_is_not_read_and_counts_on_every_pr() {
    let mut linked = with_path_crate();
    linked.read_path_crates(|_| Ok(Manifest::Linked)).unwrap();
    assert_eq!(
        linked.unfollowed,
        [
            "`vendor/v/Cargo.toml`, compared on every PR because a symbolic link is on its path, \
             and this check does not read through one"
        ]
    );
}

/// A fresh directory for one test, under the system's temporary directory.
#[cfg(unix)]
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("syntropy-dep-diff-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[cfg(unix)]
#[test]
fn read_manifest_refuses_a_link_to_a_file_or_a_directory() {
    use std::os::unix::fs::symlink;
    let top = scratch("links");
    let (root, outside) = (top.join("root"), top.join("outside"));
    std::fs::create_dir_all(outside.join("crate")).unwrap();
    std::fs::write(outside.join("secret.toml"), "SECRET_TOKEN=hunter2-abc123\n").unwrap();
    std::fs::write(
        outside.join("crate/Cargo.toml"),
        "[package]\nname = \"smuggled\"\n",
    )
    .unwrap();
    // A plain crate is read.
    std::fs::create_dir_all(root.join("vendor/plain")).unwrap();
    std::fs::write(
        root.join("vendor/plain/Cargo.toml"),
        "[package]\nname = \"p\"\n",
    )
    .unwrap();
    assert_eq!(
        read_manifest(&root, &root.join("vendor/plain/Cargo.toml")).unwrap(),
        Manifest::Text("[package]\nname = \"p\"\n".into())
    );
    // A manifest that is a link to a file outside the root.
    std::fs::create_dir_all(root.join("vendor/file")).unwrap();
    symlink(
        outside.join("secret.toml"),
        root.join("vendor/file/Cargo.toml"),
    )
    .unwrap();
    // A crate directory that is a link to one outside the root.
    symlink(outside.join("crate"), root.join("vendor/dir")).unwrap();
    // A link further up, and one that stays inside the root.
    symlink(root.join("vendor"), root.join("also")).unwrap();
    for manifest in [
        "vendor/file/Cargo.toml",
        "vendor/dir/Cargo.toml",
        "also/plain/Cargo.toml",
    ] {
        assert_eq!(
            read_manifest(&root, &root.join(manifest)).unwrap(),
            Manifest::Linked,
            "{manifest}"
        );
    }
    // A missing manifest is an error naming its path only.
    assert_eq!(
        read_manifest(&root, &root.join("vendor/none/Cargo.toml")).unwrap_err(),
        "`vendor/none`: No such file or directory (os error 2)"
    );
    std::fs::remove_dir_all(&top).unwrap();
}

#[test]
fn what_is_printed_has_its_control_characters_escaped() {
    assert_eq!(printable("plain `text`"), "plain `text`");
    assert_eq!(printable("a\nb\r\u{1b}[0m"), "a\\nb\\r\\u{1b}[0m");
    // A setting line read from a PR's file cannot start a line of its own in the log.
    let new = NewDependencies {
        sources: vec!["Cargo.toml: [patch.crates-io] x = \"1\"\r::error::forged".to_owned()],
        ..NewDependencies::default()
    };
    assert_eq!(
        new.lines(),
        [
            "new dependency source setting in Cargo.toml: [patch.crates-io] x = \"1\"\\r::error::forged"
        ]
    );
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
