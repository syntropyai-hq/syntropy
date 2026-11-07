// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::PathBuf;

use super::*;
use crate::cli::{self, Args, Command as Cli};
use crate::registry;

/// The options of `cargo xtask ci changes` with `options`, parsed as the command line is.
fn args(options: &[(&str, &str)]) -> Args {
    let mut line = vec!["ci".to_owned(), "changes".to_owned()];
    for (name, value) in options {
        line.push((*name).to_owned());
        line.push((*value).to_owned());
    }
    let Ok(Cli::Run { args, .. }) = cli::parse(&line, registry::JOBS, registry::SUITES) else {
        panic!("`{line:?}` did not parse");
    };
    args
}

/// A directory under the system temp directory, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    /// A new directory: tests run at once, so each gets its own even under one name.
    fn new(name: &str) -> Scratch {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "syntropy-xtask-changes-{name}-{}-{n}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    fn write(&self, path: &str, text: &str) {
        let full = self.0.join(path);
        std::fs::create_dir_all(full.parent().unwrap()).unwrap();
        std::fs::write(full, text).unwrap();
    }

    fn remove(&self, path: &str) {
        std::fs::remove_file(self.0.join(path)).unwrap();
    }

    fn git(&self, args: &[&str]) -> String {
        let output = Command::new("git")
            .current_dir(&self.0)
            .args([
                "-c",
                "user.name=fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "init.defaultBranch=main",
                "-c",
                "core.autocrlf=false",
            ])
            .args(args)
            .output()
            .unwrap();
        assert!(output.status.success(), "git {args:?}: {output:?}");
        String::from_utf8(output.stdout).unwrap().trim().to_owned()
    }

    /// Commits everything and returns the commit's SHA.
    fn commit(&self, message: &str) -> String {
        self.git(&["add", "-A"]);
        self.git(&["commit", "--quiet", "--allow-empty", "-m", message]);
        self.git(&["rev-parse", "HEAD"])
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A workspace whose members declare their edges in each way a manifest can:
/// - `net` on `core` by path;
/// - `kernels-cuda` on `kernels` through `workspace = true`;
/// - `bench` on `net` as a renamed dev-dependency;
/// - `xtask` on `core` under a `[target]` table;
/// - `kernels` on nothing, so its closure is itself and `kernels-cuda`.
fn write_workspace(tree: &Scratch) {
    tree.write(
        "Cargo.toml",
        r#"[workspace]
members = ["crates/core", "crates/net", "crates/kernels", "crates/kernels-cuda", "crates/bench", "xtask"]

[workspace.dependencies]
syntropy-kernels = { path = "crates/kernels" }
serde = "1"
"#,
    );
    tree.write(
        "crates/core/Cargo.toml",
        "[package]\nname = \"syntropy-core\"\n[dependencies]\nserde.workspace = true\n",
    );
    tree.write(
        "crates/net/Cargo.toml",
        "[package]\nname = \"syntropy-net\"\n[dependencies]\nsyntropy-core = { path = \"../core\" }\n",
    );
    tree.write(
        "crates/kernels/Cargo.toml",
        "[package]\nname = \"syntropy-kernels\"\n",
    );
    tree.write(
        "crates/kernels-cuda/Cargo.toml",
        "[package]\nname = \"syntropy-kernels-cuda\"\n[dependencies]\nsyntropy-kernels.workspace = true\n",
    );
    tree.write(
        "crates/bench/Cargo.toml",
        "[package]\nname = \"syntropy-bench\"\n[dev-dependencies]\nwire = { package = \"syntropy-net\", path = \"../net\" }\n",
    );
    tree.write(
        "xtask/Cargo.toml",
        "[package]\nname = \"syntropy-xtask\"\n[target.'cfg(unix)'.dependencies]\nsyntropy-core = { path = \"../crates/core\" }\n",
    );
}

const RULES: &str = r#"
[paths]
workspace = ["Cargo.lock", "xtask/**"]
no_crate = ["docs/**", "**/*.md"]

[inputs]
"numerics/**" = ["kernels"]

[hardware]
status_creators = []

[job.lint]
runner = "hosted"
when = "always"

[job.hosted-test]
runner = "hosted"
when = "closure"

[job.hosted-l3]
runner = "hosted"
when = "closure"
crates = ["net"]

[job.hosted-parity-nano]
runner = "hosted"
when = "closure"
planned = "QUAL-E5-T06"

[job.hw-cuda-ada]
runner = "lab"
when = "direct"
paths = ["crates/kernels-*/**", "crates/kernels/**"]
lock_deps = ["cuda-bindings"]
planned = "QUAL-E2-T02"
"#;

fn rules() -> Config {
    Config::parse(RULES).expect("the test rules parse")
}

fn workspace() -> (Scratch, Workspace) {
    let tree = Scratch::new("workspace");
    write_workspace(&tree);
    let workspace = Workspace::read(&Files(&tree.0)).expect("the workspace reads");
    (tree, workspace)
}

fn names(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

fn paths(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

#[test]
fn globs_match_whole_segments_and_any_depth() {
    let glob = |text: &str| Glob::parse(text).unwrap();
    assert!(glob("docs/**").matches("docs/a/b.md"));
    assert!(glob("**/*.md").matches("README.md"));
    assert!(glob("**/*.md").matches("a/b/c.md"));
    assert!(!glob("**/*.md").matches("a/b.mdx"));
    assert!(glob("crates/kernels-*/**").matches("crates/kernels-cuda/src/lib.rs"));
    assert!(!glob("crates/kernels-*/**").matches("crates/kernels/src/lib.rs"));
    assert!(!glob("crates/kernels/**").matches("crates/kernels-cuda/src/lib.rs"));
    assert!(glob("ci/images/cuda*/**").matches("ci/images/cuda-12/Dockerfile"));
    assert!(glob("rust-toolchain.toml").matches("rust-toolchain.toml"));
    assert!(!glob("rust-toolchain.toml").matches("x/rust-toolchain.toml"));
    // `*` stays inside one segment.
    assert!(!glob("crates/*").matches("crates/core/src/lib.rs"));
    for bad in [
        "", "/docs/**", "docs/", "a//b", "a/**b", "a/[ab]", "../x", "a\\b",
    ] {
        assert!(Glob::parse(bad).is_err(), "`{bad}` parsed");
    }
}

#[test]
fn the_rules_reject_what_they_cannot_mean() {
    let problems = |job: &str| {
        let text = format!(
            "[paths]\nworkspace = []\nno_crate = []\n[hardware]\nstatus_creators = []\n{job}"
        );
        Config::parse(&text).unwrap_err()
    };
    let one = |job: &str, want: &str| {
        let found = problems(job);
        assert!(
            found.iter().any(|p| p.contains(want)),
            "{job}: wanted `{want}` in {found:?}"
        );
    };
    one(
        "[job.x]\nrunner = \"cloud\"\nwhen = \"always\"\n",
        "is not `hosted` or `lab`",
    );
    one(
        "[job.x]\nrunner = \"hosted\"\nwhen = \"sometimes\"\n",
        "is not `always`",
    );
    one(
        "[job.x]\nrunner = \"hosted\"\nwhen = \"direct\"\n",
        "a direct rule needs `paths` or `lock_deps`",
    );
    one(
        "[job.x]\nrunner = \"hosted\"\nwhen = \"direct\"\ncrates = [\"core\"]\npaths = [\"a\"]\n",
        "`crates` goes with",
    );
    one(
        "[job.x]\nrunner = \"hosted\"\nwhen = \"closure\"\npaths = [\"a\"]\n",
        "`paths` and `lock_deps` go with",
    );
    one(
        "[job.hw-x]\nrunner = \"lab\"\nwhen = \"closure\"\n",
        "a lab job uses a direct rule",
    );
    one(
        "[job.cuda]\nrunner = \"lab\"\nwhen = \"direct\"\npaths = [\"a\"]\n",
        "a lab job is named `hw-<rig>`",
    );
    one(
        "[job.x]\nrunner = \"hosted\"\nwhen = \"always\"\nplanned = \"soon\"\n",
        "is not a task ID",
    );
    one(
        "[job.x]\nrunner = \"hosted\"\nwhen = \"always\"\npaused = \"M1a\"\n",
        "is not an ADR ID",
    );
    one(
        "[job.x]\nrunner = \"hosted\"\nwhen = \"always\"\nplanned = \"QUAL-E5-T06\"\npaused = \"ADR-0626\"\n",
        "only a hosted job with a suite can be paused",
    );
    one(
        "[job.hw-x]\nrunner = \"lab\"\nwhen = \"direct\"\npaths = [\"a\"]\npaused = \"ADR-0626\"\n",
        "only a hosted job with a suite can be paused",
    );
    one(
        "[job.Lint]\nrunner = \"hosted\"\nwhen = \"always\"\n",
        "lowercase words",
    );
    one(
        "[job.x]\nrunner = \"hosted\"\nwhen = \"always\"\nextra = 1\n",
        "unknown field",
    );
    one(
        "[job.x]\nrunner = \"hosted\"\nwhen = \"direct\"\npaths = [\"/abs\"]\n",
        "a relative path",
    );
    // Several task IDs of one epic are one reference.
    assert!(is_task_ref("QUAL-E3-T09/T10"));
    assert!(!is_task_ref("QUAL-E3-T09/E4-T10"));
}

#[test]
fn the_workspace_reads_every_kind_of_edge() {
    let (_tree, workspace) = workspace();
    assert_eq!(
        workspace.all(),
        names(&["bench", "core", "kernels", "kernels-cuda", "net", "xtask"])
    );
    assert_eq!(
        workspace.closure(&names(&["core"])),
        names(&["bench", "core", "net", "xtask"])
    );
    assert_eq!(
        workspace.closure(&names(&["kernels"])),
        names(&["kernels", "kernels-cuda"])
    );
    assert_eq!(workspace.closure(&names(&["bench"])), names(&["bench"]));
    // The deepest directory wins, and a prefix of a name is not its directory.
    assert_eq!(
        workspace.crate_of("crates/kernels-cuda/src/lib.rs"),
        Some("kernels-cuda")
    );
    assert_eq!(
        workspace.crate_of("crates/kernels/src/lib.rs"),
        Some("kernels")
    );
    assert_eq!(workspace.crate_of("crates/kernelsx/lib.rs"), None);
    assert_eq!(workspace.crate_of("crates/core"), None);
}

#[test]
fn a_member_glob_or_a_missing_manifest_is_refused() {
    let tree = Scratch::new("glob");
    tree.write("Cargo.toml", "[workspace]\nmembers = [\"crates/*\"]\n");
    let error = Workspace::read(&Files(&tree.0)).unwrap_err();
    assert!(error.contains("is a glob"), "{error}");
    tree.write("Cargo.toml", "[workspace]\nmembers = [\"crates/gone\"]\n");
    let error = Workspace::read(&Files(&tree.0)).unwrap_err();
    assert!(error.contains("has no crates/gone/Cargo.toml"), "{error}");
}

#[test]
fn each_path_falls_in_the_first_class_that_takes_it() {
    let (_tree, workspace) = workspace();
    let config = rules();
    let class = |path: &str| classify(&config, &workspace, path);
    // `xtask/**` is a workspace path before it is the `xtask` crate.
    assert_eq!(
        class("xtask/src/main.rs"),
        Class::Workspace("xtask/**".into())
    );
    assert_eq!(class("crates/net/src/lib.rs"), Class::Crate("net".into()));
    // A crate's own markdown is the crate's, before `**/*.md`.
    assert_eq!(class("crates/core/README.md"), Class::Crate("core".into()));
    assert_eq!(
        class("numerics/tolerances-v1.toml"),
        Class::Input("numerics/**".into(), names(&["kernels"]))
    );
    assert_eq!(class("docs/HOW_IT_WORKS.md"), Class::NoCrate);
    assert_eq!(class("AGENTS.md"), Class::NoCrate);
    assert_eq!(class(".gitattributes"), Class::Unclassified);
}

#[test]
fn the_rules_pick_the_jobs() {
    let (_tree, workspace) = workspace();
    let config = rules();
    let run = |changed: &[&str], lock: &[&str]| {
        let found = affected(&config, &workspace, &paths(changed), &names(lock));
        let planned: Vec<String> = found.planned.iter().map(|(n, _)| n.clone()).collect();
        (found.jobs, found.hw_required, planned)
    };
    let list = |items: &[&str]| paths(items);
    // Docs only: `lint` alone.
    assert_eq!(
        run(&["docs/a.md", "README.md"], &[]),
        (list(&["lint"]), list(&[]), list(&[]))
    );
    // `core` only: every hosted job, through the closure, and no hardware job.
    assert_eq!(
        run(&["crates/core/src/lib.rs"], &[]),
        (
            list(&["hosted-l3", "hosted-test", "lint"]),
            list(&[]),
            list(&["hosted-parity-nano"])
        )
    );
    // `bench` only: its closure holds no crate `hosted-l3` names.
    assert_eq!(
        run(&["crates/bench/src/lib.rs"], &[]),
        (
            list(&["hosted-test", "lint"]),
            list(&[]),
            list(&["hosted-parity-nano"])
        )
    );
    // A kernel provider: the direct hardware rule, planned.
    assert_eq!(
        run(&["crates/kernels-cuda/src/lib.rs"], &[]),
        (
            list(&["hosted-test", "lint"]),
            list(&["hw-cuda-ada"]),
            list(&["hosted-parity-nano", "hw-cuda-ada"])
        )
    );
    // An input of `kernels` changes `kernels` and its closure, but no path the direct rule
    // names.
    assert_eq!(run(&["numerics/x.toml"], &[]).1, list(&[]));
    // A lock dependency triggers the hardware job; any `Cargo.lock` change runs every crate.
    assert_eq!(
        run(&["Cargo.lock"], &["cuda-bindings"]),
        (
            list(&["hosted-l3", "hosted-test", "lint"]),
            list(&["hw-cuda-ada"]),
            list(&["hosted-parity-nano", "hw-cuda-ada"])
        )
    );
    assert_eq!(run(&["Cargo.lock"], &["serde"]).1, list(&[]));
    // A path no rule knows fails closed: every crate.
    assert_eq!(
        run(&[".gitattributes"], &[]).0,
        list(&["hosted-l3", "hosted-test", "lint"])
    );
}

#[test]
fn a_paused_job_is_reported_but_never_listed() {
    let (_tree, workspace) = workspace();
    let config = Config::parse(&format!(
        "{RULES}\n[job.hosted-test-macos-windows]\nrunner = \"hosted\"\nwhen = \"closure\"\n\
         paused = \"ADR-0626\"\n"
    ))
    .expect("the rules parse");
    let found = affected(
        &config,
        &workspace,
        &paths(&["crates/core/src/lib.rs"]),
        &names(&[]),
    );
    assert_eq!(found.jobs, paths(&["hosted-l3", "hosted-test", "lint"]));
    assert_eq!(
        found.paused,
        [(
            "hosted-test-macos-windows".to_owned(),
            "ADR-0626".to_owned()
        )]
    );
    assert!(
        report(&found, &config).contains(
            &"hosted-test-macos-windows: would run (the closure holds 4 crates), but it is \
              paused: ADR-0626"
                .to_owned()
        ),
        "{:?}",
        report(&found, &config)
    );
    // Nor does a path no rule knows, which runs every other hosted job.
    let found = affected(
        &config,
        &workspace,
        &paths(&[".gitattributes"]),
        &names(&[]),
    );
    assert!(!found.jobs.contains(&"hosted-test-macos-windows".to_owned()));
}

#[test]
fn a_lock_change_names_each_package_whose_entry_differs() {
    let lock = |entries: &[(&str, &str)]| {
        entries
            .iter()
            .map(|(name, version)| {
                format!("[[package]]\nname = \"{name}\"\nversion = \"{version}\"\n")
            })
            .collect::<String>()
    };
    let base = lock(&[("a", "1.0.0"), ("b", "1.0.0"), ("c", "1.0.0")]);
    let head = lock(&[("a", "1.0.0"), ("b", "1.1.0"), ("d", "1.0.0")]);
    assert_eq!(
        lock_changes(Some(&base), Some(&head)).unwrap(),
        names(&["b", "c", "d"])
    );
    assert_eq!(
        lock_changes(None, Some(&head)).unwrap(),
        names(&["a", "b", "d"])
    );
    assert!(lock_changes(Some("not toml ["), Some(&head)).is_err());
}

#[test]
fn the_workflow_jobs_are_the_keys_under_jobs() {
    let workflow = "name: pr\non:\n  pull_request:\njobs:\n  changes:\n    runs-on: x\n  # a note\n  lint:\n    needs: changes\n    steps:\n      - run: y\npermissions:\n  contents: read\n";
    assert_eq!(workflow_jobs(workflow), ["changes", "lint"]);
}

/// A golden suite and workflow for [`RULES`] on [`write_workspace`]'s crates that passes.
fn passing_goldens(config: &str) -> String {
    format!(
        r#"jobs_toml = "{}"

[[golden]]
name = "docs"
source = "constructed"
paths = ["docs/a.md", "README.md"]
jobs = ["lint"]
hw_required = []

[[golden]]
name = "core"
source = "constructed"
paths = ["crates/core/src/lib.rs"]
jobs = ["lint", "hosted-test", "hosted-l3"]
hw_required = []
planned = ["hosted-parity-nano"]

[[golden]]
name = "kernels"
source = "constructed"
paths = ["crates/kernels/src/lib.rs", "numerics/t.toml"]
jobs = ["lint", "hosted-test"]
hw_required = ["hw-cuda-ada"]
planned = ["hosted-parity-nano", "hw-cuda-ada"]

[[golden]]
name = "kernels-cuda"
source = "constructed"
paths = ["crates/kernels-cuda/src/lib.rs"]
jobs = ["lint", "hosted-test"]
hw_required = ["hw-cuda-ada"]
planned = ["hosted-parity-nano", "hw-cuda-ada"]

[[golden]]
name = "lock and xtask"
source = "constructed"
paths = ["Cargo.lock", "xtask/src/main.rs", ".gitattributes"]
lock = ["cuda-bindings"]
jobs = ["lint", "hosted-test", "hosted-l3"]
hw_required = ["hw-cuda-ada"]
planned = ["hosted-parity-nano", "hw-cuda-ada"]
"#,
        digest(config)
    )
}

const WORKFLOW_TEXT: &str =
    "jobs:\n  changes:\n  lint:\n  hosted-test:\n  hosted-l3:\n  hosted-parity-nano:\n  ci-gate:\n";

#[test]
fn the_golden_check_passes_a_complete_suite() {
    let (_tree, workspace) = workspace();
    let problems = check(RULES, &passing_goldens(RULES), &workspace, WORKFLOW_TEXT);
    assert_eq!(problems, Vec::<String>::new());
}

#[test]
fn editing_the_rules_without_the_goldens_fails() {
    let (_tree, workspace) = workspace();
    let goldens = passing_goldens(RULES);
    // Even a comment changes the digest.
    let edited = format!("{RULES}# a note\n");
    let problems = check(&edited, &goldens, &workspace, WORKFLOW_TEXT);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("changed since ci/jobs-goldens.toml"),
        "{problems:?}"
    );
    // A line-end change alone does not.
    let crlf = RULES.replace('\n', "\r\n");
    assert_eq!(digest(&crlf), digest(RULES));
}

#[test]
fn each_part_of_a_rule_needs_a_golden_that_depends_on_it_alone() {
    let (_tree, workspace) = workspace();
    // One golden meets both hardware patterns, so either could be dropped unseen.
    let goldens = passing_goldens(RULES).replace(
        "paths = [\"crates/kernels-cuda/src/lib.rs\"]",
        "paths = [\"crates/kernels-cuda/src/lib.rs\", \"crates/kernels/src/x.rs\"]",
    );
    let goldens = goldens.replace(
        "paths = [\"crates/kernels/src/lib.rs\", \"numerics/t.toml\"]",
        "paths = [\"numerics/t.toml\"]",
    );
    let goldens = goldens.replace(
        "name = \"kernels\"\nsource = \"constructed\"\npaths = [\"numerics/t.toml\"]\njobs = [\"lint\", \"hosted-test\"]\nhw_required = [\"hw-cuda-ada\"]\nplanned = [\"hosted-parity-nano\", \"hw-cuda-ada\"]",
        "name = \"kernels\"\nsource = \"constructed\"\npaths = [\"numerics/t.toml\"]\njobs = [\"lint\", \"hosted-test\"]\nhw_required = []\nplanned = [\"hosted-parity-nano\"]",
    );
    assert_eq!(
        check(RULES, &goldens, &workspace, WORKFLOW_TEXT),
        [
            "ci/jobs-goldens.toml: no golden meets `crates/kernels-*/**` alone among \
             `hw-cuda-ada`'s patterns, so dropping it would go unseen",
            "ci/jobs-goldens.toml: no golden meets `crates/kernels/**` alone among \
             `hw-cuda-ada`'s patterns, so dropping it would go unseen",
        ]
    );
    // A lock dependency changed beside a hardware path is not pinned either.
    let goldens = passing_goldens(RULES).replace(
        "paths = [\"Cargo.lock\", \"xtask/src/main.rs\", \".gitattributes\"]",
        "paths = [\"Cargo.lock\", \"xtask/src/main.rs\", \".gitattributes\", \"crates/kernels/src/y.rs\"]",
    );
    assert_eq!(
        check(RULES, &goldens, &workspace, WORKFLOW_TEXT),
        [
            "ci/jobs-goldens.toml: no golden changes `cuda-bindings` alone among `hw-cuda-ada`'s \
             lock dependencies and paths, so dropping it would go unseen"
        ]
    );
    // Of a closure rule's crates, one whose own closure reaches another listed crate is
    // redundant and needs no golden; `core`'s closure holds `net`. `net` alone needs one.
    let rules = RULES.replace("crates = [\"net\"]", "crates = [\"net\", \"core\"]");
    let goldens = passing_goldens(&rules);
    assert_eq!(
        check(&rules, &goldens, &workspace, WORKFLOW_TEXT),
        [
            "ci/jobs-goldens.toml: no golden's closure includes `net` alone among `hosted-l3`'s \
             crates, so dropping it would go unseen"
        ]
    );
    let goldens = format!(
        "{goldens}\n[[golden]]\nname = \"net\"\nsource = \"constructed\"\n\
         paths = [\"crates/net/src/lib.rs\"]\njobs = [\"lint\", \"hosted-test\", \"hosted-l3\"]\n\
         hw_required = []\nplanned = [\"hosted-parity-nano\"]\n"
    );
    assert_eq!(
        check(&rules, &goldens, &workspace, WORKFLOW_TEXT),
        Vec::<String>::new()
    );
}

#[test]
fn the_golden_check_names_each_gap() {
    let (_tree, workspace) = workspace();
    // A new pattern with no golden, and a golden whose sets the rules no longer give.
    let rules = RULES.replace(
        "paths = [\"crates/kernels-*/**\", \"crates/kernels/**\"]",
        "paths = [\"crates/kernels-*/**\", \"crates/kernels/**\", \"third_party/ggml/**\"]",
    );
    let goldens = passing_goldens(&rules).replace(
        "jobs = [\"lint\"]\nhw_required = []",
        "jobs = [\"lint\", \"hosted-test\"]\nhw_required = []",
    );
    let problems = check(&rules, &goldens, &workspace, WORKFLOW_TEXT);
    assert_eq!(
        problems,
        [
            "ci/jobs-goldens.toml: golden `docs`: jobs is [\"lint\"], expected [\"hosted-test\", \
             \"lint\"]",
            "ci/jobs-goldens.toml: no golden meets `third_party/ggml/**` alone among \
             `hw-cuda-ada`'s patterns, so dropping it would go unseen",
        ]
    );
    // A workflow job with no rule, and a rule for a job the workflow lacks.
    let workflow = WORKFLOW_TEXT.replace("  hosted-l3:\n", "  sim-pr:\n");
    let problems = check(RULES, &passing_goldens(RULES), &workspace, &workflow);
    assert_eq!(
        problems,
        [
            "ci/jobs.toml: hosted job `hosted-l3` is not a job of .github/workflows/pr.yml",
            ".github/workflows/pr.yml: job `sim-pr` has no rule in ci/jobs.toml, so `changes` \
             never lists it",
        ]
    );
    // No golden fails closed, and a crate the rules name does not exist.
    let rules = RULES.replace("crates = [\"net\"]", "crates = [\"nett\"]");
    let goldens = passing_goldens(&rules).replace(", \".gitattributes\"", "");
    let problems = check(&rules, &goldens, &workspace, WORKFLOW_TEXT);
    assert!(
        problems
            .iter()
            .any(|p| p.contains("names `nett`, which is not a workspace crate")),
        "{problems:?}"
    );
    assert!(
        problems
            .iter()
            .any(|p| p.contains("no golden has a path no rule knows")),
        "{problems:?}"
    );
}

/// `changes` in a repository: the base checkout's rules decide, even when the head changes the
/// rules to list nothing, and the PR's own manifests add the edges they declare.
#[test]
fn the_step_applies_the_base_rules_to_the_heads_diff() {
    let tree = Scratch::new("step");
    tree.git(&["init", "--quiet"]);
    write_workspace(&tree);
    tree.write(CONFIG, RULES);
    let base = tree.commit("base");
    // The head edits the rules to list no job at all, moves a file within `kernels-cuda`, and
    // makes `kernels` depend on `net`, so a `net` change now reaches `kernels` too.
    tree.write(
        CONFIG,
        "[paths]\nworkspace = []\nno_crate = [\"**\"]\n[hardware]\nstatus_creators = []\n",
    );
    tree.write("crates/kernels-cuda/src/old.rs", "");
    let moved_from = tree.commit("add a file to move");
    tree.remove("crates/kernels-cuda/src/old.rs");
    tree.write("crates/kernels-cuda/src/new.rs", "");
    tree.write(
        "crates/kernels/Cargo.toml",
        "[package]\nname = \"syntropy-kernels\"\n[dependencies]\nsyntropy-net = { path = \"../net\" }\n",
    );
    let head = tree.commit("head");
    // Check the base out, as `pr.yml` does, so its rules are on disk.
    tree.git(&["checkout", "--quiet", "--detach", &base]);
    let changed = changed_paths(&tree.0, &moved_from, &head).unwrap();
    assert_eq!(
        changed,
        [
            "crates/kernels-cuda/src/new.rs",
            "crates/kernels-cuda/src/old.rs",
            "crates/kernels/Cargo.toml"
        ]
    );
    let output = tree.0.join("github-output");
    let context = Context {
        root: tree.0.clone(),
        args: args(&[
            ("--base", base.as_str()),
            ("--head", head.as_str()),
            ("--github-output", output.to_str().unwrap()),
        ]),
    };
    step(&context).expect("changes runs");
    let written = std::fs::read_to_string(&output).unwrap();
    // The base's rules decide: the head's `no_crate = ["**"]` would have listed `lint` alone.
    // Under them the head's edit to `ci/jobs.toml` is a path no rule knows, so it counts as every
    // crate, and `hosted-l3` runs too.
    assert_eq!(
        written,
        "jobs=[\"hosted-l3\",\"hosted-test\",\"lint\"]\nhw_required=[\"hw-cuda-ada\"]\n"
    );
    // The union of the graphs: the head's new edge puts `kernels` in a `net` change's closure.
    let graph = Workspace::read(&Git {
        root: &tree.0,
        rev: &base,
    })
    .unwrap()
    .union(
        Workspace::read(&Git {
            root: &tree.0,
            rev: &head,
        })
        .unwrap(),
    );
    assert!(graph.closure(&names(&["net"])).contains("kernels"));
    // An unknown revision is refused by name.
    let context = Context {
        root: tree.0.clone(),
        args: args(&[("--base", "no-such-rev"), ("--head", head.as_str())]),
    };
    assert_eq!(
        step(&context).unwrap_err(),
        "`no-such-rev` is not a commit in this repository"
    );
}

#[test]
fn a_missing_file_at_a_revision_reads_as_none() {
    let tree = Scratch::new("git-read");
    tree.git(&["init", "--quiet"]);
    tree.write("a.txt", "one\n");
    let rev = tree.commit("one");
    let git = Git {
        root: &tree.0,
        rev: &rev,
    };
    assert_eq!(git.read("a.txt").unwrap().as_deref(), Some("one\n"));
    assert_eq!(git.read("b.txt").unwrap(), None);
}

/// The checkout's rules, goldens and workflow agree: what `lint`'s `jobs` step checks.
#[test]
fn the_checkout_passes() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let context = Context {
        root: root.to_owned(),
        args: Args::default(),
    };
    if let Err(problems) = goldens_step(&context) {
        panic!("{problems}");
    }
}
