// SPDX-License-Identifier: MIT OR Apache-2.0

//! End-to-end tests of `syntropy-bench scenarios check`, the `lint` step (QUAL-E13-T01;
//! RFC-0061 §4.1, §4.2).
//!
//! Each test builds a small repository in a temporary directory, with git history where the
//! version rule needs it, and runs the binary the way `lint` runs it. The acceptance criteria:
//! - the check fails if any SC row is unmapped, or if a scenario changes without a
//!   `scenario_version` bump;
//! - the SC table renders from the map.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use syntropy_bench::{sc_map, scenario};

const VISION: &str = "# Vision\n\n| SC-1 | a | b |\n| SC-2 | c | d |\n";
const EPICS: &str = "| ID | Task | Milestone | Depends on | Acceptance criteria | Implements |\n\
                     |---|---|---|---|---|---|\n\
                     | QUAL-E13-T02 | the SC-1 and SC-2 scenarios | M0b | none | reported nightly \
                     | RFC-0061 §4.2 |\n";
const EPICS_PATH: &str = "plan/implementation/epics/QUAL.md";
const RFC_HEAD: &str = "# RFC-0061\n\n#### 4.2 Success criteria\n\n\
                        <!-- BEGIN GENERATED: sc-table -->\n";
const RFC_TAIL: &str = "<!-- END GENERATED -->\n\n#### 6.2 Rigs (ADR-0610)\n\n\
                        | **R-ADA** (M0) | a desktop |\n\n#### 6.3 Inventory\n";

fn row(n: u32) -> String {
    format!(
        "[[sc]]\nid = \"SC-{n}\"\nscenarios = \"`decode.single.*`\"\n\
         patterns = [\"decode.single.*\"]\nrigs = \"R-ADA\"\nmeasured = \"`t_tok`\"\n\
         pass = \"≤ 1.05×\"\nadded_owners = [\"QUAL-E13-T02\"]\n\n"
    )
}

const SCENARIO: &str = "scenario = \"decode.single.llama8b.q4km\"\nscenario_version = 1\n\
                        model = \"llama-3.1-8b\"\nquant = \"q4km\"\nrigs = [\"R-ADA\"]\n\
                        topology = \"single\"\ninterconnect = \"none\"\n\
                        objective = \"latency\"\ngrid = \"standard\"\n";
const SCENARIO_PATH: &str = "bench/scenarios/decode.single.llama8b.q4km.toml";

/// A throwaway repository under the system temp directory, removed on drop.
struct Repo {
    root: PathBuf,
}

impl Repo {
    fn new(name: &str) -> Repo {
        let root =
            std::env::temp_dir().join(format!("syntropy-bench-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("temp dir");
        let repo = Repo { root };
        repo.git(&["init", "--quiet", "--initial-branch=main"]);
        repo
    }

    /// A repository whose map covers `mapped` and whose RFC table is rendered from it.
    fn with_map(name: &str, mapped: &[u32]) -> Repo {
        let repo = Repo::new(name);
        repo.write("plan/00-vision-and-scope.md", VISION);
        repo.write(EPICS_PATH, EPICS);
        let map: String = mapped.iter().map(|&n| row(n)).collect();
        repo.write(sc_map::SC_MAP, &map);
        let map = sc_map::parse(&map).expect("test map parses");
        let tasks = sc_map::tasks(&repo.root.join(sc_map::EPICS));
        let table = sc_map::render(&map, &tasks, &[]);
        repo.write(sc_map::RFC_0061, &format!("{RFC_HEAD}{table}{RFC_TAIL}"));
        repo
    }

    fn write(&self, rel: &str, text: &str) {
        let path = self.root.join(rel);
        fs::create_dir_all(path.parent().expect("has a parent")).expect("mkdir");
        fs::write(path, text).expect("write");
    }

    fn git(&self, args: &[&str]) {
        let status = Command::new("git")
            .args([
                "-c",
                "user.name=bench test",
                "-c",
                "user.email=bench-test@example.invalid",
                "-c",
                "commit.gpgsign=false",
            ])
            .args(args)
            .current_dir(&self.root)
            .status()
            .expect("git runs");
        assert!(status.success(), "git {args:?} failed");
    }

    /// Commits everything and points `origin/main` at the commit, as a fetched base would be.
    fn commit_as_base(&self) {
        self.git(&["add", "--all"]);
        self.git(&["commit", "--quiet", "--message", "base"]);
        self.git(&["update-ref", "refs/remotes/origin/main", "HEAD"]);
    }

    /// Runs `syntropy-bench scenarios <args> --root <root>`.
    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_syntropy-bench"))
            .arg("scenarios")
            .args(args)
            .arg("--root")
            .arg(&self.root)
            .output()
            .expect("syntropy-bench runs")
    }

    fn rfc(&self) -> String {
        fs::read_to_string(self.root.join(sc_map::RFC_0061)).expect("rfc")
    }
}

impl Drop for Repo {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn a_complete_map_with_a_current_table_passes() {
    let repo = Repo::with_map("complete", &[1, 2]);
    repo.write(SCENARIO_PATH, SCENARIO);
    repo.commit_as_base();
    // The scenario file now exists, so the table must show it.
    let render = repo.run(&["render", "--write"]);
    assert!(render.status.success(), "{}", stderr(&render));
    assert!(repo.rfc().contains("| 1 scenario; QUAL-E13-T02 |"));
    let output = repo.run(&["check"]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    assert!(stderr(&output).contains("1 scenario files, 2 SC rows, 0 problems"));
}

#[test]
fn an_unmapped_sc_row_fails() {
    let repo = Repo::with_map("unmapped", &[1]);
    repo.commit_as_base();
    let output = repo.run(&["check"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("SC-2 is a success criterion in plan/00-vision-and-scope.md §7"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn a_scenario_changed_without_a_version_bump_fails() {
    let repo = Repo::with_map("nobump", &[1, 2]);
    repo.write(SCENARIO_PATH, SCENARIO);
    let _ = repo.run(&["render", "--write"]);
    repo.commit_as_base();
    // Change the grid but keep `scenario_version = 1`.
    let changed =
        SCENARIO.replace("grid = \"standard\"\n", "") + "prefix = \"warm\"\ngrid = \"standard\"\n";
    repo.write(SCENARIO_PATH, &changed);
    let output = repo.run(&["check"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output)
            .contains("changed without a `scenario_version` bump: it was 1 at the base commit"),
        "{}",
        stderr(&output)
    );
    // The same change with a bump passes.
    repo.write(
        SCENARIO_PATH,
        &changed.replace("scenario_version = 1", "scenario_version = 2"),
    );
    let output = repo.run(&["check"]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
}

#[test]
fn a_new_scenario_needs_no_bump_and_an_explicit_base_works() {
    let repo = Repo::with_map("newfile", &[1, 2]);
    repo.commit_as_base();
    repo.write(SCENARIO_PATH, SCENARIO);
    let _ = repo.run(&["render", "--write"]);
    let output = repo.run(&["check", "--base", "main"]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
}

#[test]
fn a_missing_base_fails_closed_and_says_how_to_fix_it() {
    let repo = Repo::with_map("nobase", &[1, 2]);
    // Committed, but no `origin/main`: a shallow CI checkout looks like this.
    repo.git(&["add", "--all"]);
    repo.git(&["commit", "--quiet", "--message", "base"]);
    let output = repo.run(&["check"]);
    assert_eq!(output.status.code(), Some(1));
    let err = stderr(&output);
    assert!(
        err.contains("the scenario_version check has no base commit"),
        "{err}"
    );
    assert!(
        err.contains("git fetch origin main") && err.contains("--base <ref>"),
        "{err}"
    );
    let output = repo.run(&["check", "--base", "no-such-ref"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("`no-such-ref` is not a commit here"));
}

#[test]
fn a_stale_table_fails_until_rendered() {
    let repo = Repo::with_map("stale", &[1, 2]);
    repo.commit_as_base();
    let map = fs::read_to_string(repo.root.join(sc_map::SC_MAP)).expect("map");
    repo.write(sc_map::SC_MAP, &map.replace("≤ 1.05×", "≤ 1.10×"));
    let output = repo.run(&["check"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("section is stale"),
        "{}",
        stderr(&output)
    );
    let render = repo.run(&["render", "--write"]);
    assert!(render.status.success());
    assert!(repo.rfc().contains("≤ 1.10×"));
    let output = repo.run(&["check"]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
}

#[test]
fn a_bad_scenario_file_is_reported_with_its_path() {
    let repo = Repo::with_map("badfile", &[1, 2]);
    repo.commit_as_base();
    repo.write(
        "bench/scenarios/decode.single.other.q4km.toml",
        &SCENARIO.replace("[\"R-ADA\"]", "[\"R-NOWHERE\"]"),
    );
    let output = repo.run(&["check"]);
    assert_eq!(output.status.code(), Some(1));
    let err = stderr(&output);
    assert!(
        err.contains("the file name must be `decode.single.llama8b.q4km.toml`"),
        "{err}"
    );
    assert!(
        err.contains("rig `R-NOWHERE` is not a rig of RFC-0061 §6.2"),
        "{err}"
    );
}

#[test]
fn a_scenario_removed_and_re_added_needs_a_higher_version_than_its_last() {
    let repo = Repo::with_map("readd", &[1, 2]);
    let v3 = SCENARIO.replace("scenario_version = 1", "scenario_version = 3");
    repo.write(SCENARIO_PATH, &v3);
    let _ = repo.run(&["render", "--write"]);
    repo.git(&["add", "--all"]);
    repo.git(&["commit", "--quiet", "--message", "add at version 3"]);
    // Changed once before it is removed: its last version is 5, not the 3 it was added at.
    let v5 = format!(
        "{}objective = \"latency\"\n",
        v3.replace("scenario_version = 3", "scenario_version = 5")
    )
    .replace("objective = \"latency\"\ngrid", "grid");
    repo.write(SCENARIO_PATH, &v5);
    repo.git(&[
        "commit",
        "--quiet",
        "--all",
        "--message",
        "change to version 5",
    ]);
    repo.git(&["rm", "--quiet", SCENARIO_PATH]);
    let _ = repo.run(&["render", "--write"]);
    repo.commit_as_base();

    // Re-added with a different definition at a version it already had: refused.
    let changed = format!("{v3}prefix = \"warm\"\n");
    repo.write(
        SCENARIO_PATH,
        &changed.replace("scenario_version = 3", "scenario_version = 1"),
    );
    let _ = repo.run(&["render", "--write"]);
    let output = repo.run(&["check"]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    let err = stderr(&output);
    assert!(
        err.contains("it was 5 at commit ")
            && err.contains("its last version before it was removed, and is 1 now; raise it to 6"),
        "{err}"
    );

    // Above the version it was added at, but below its last one: still refused.
    repo.write(
        SCENARIO_PATH,
        &changed.replace("scenario_version = 3", "scenario_version = 4"),
    );
    let output = repo.run(&["check"]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("is 4 now; raise it to 6"),
        "{}",
        stderr(&output)
    );

    // The last definition re-added unchanged keeps its version.
    repo.write(SCENARIO_PATH, &v5);
    let output = repo.run(&["check"]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));

    // A changed definition above the last version passes.
    repo.write(
        SCENARIO_PATH,
        &changed.replace("scenario_version = 3", "scenario_version = 6"),
    );
    let output = repo.run(&["check"]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
}

/// An epic edit that makes a task cite a criterion makes the table stale, and the check names
/// the row and the task until the table is rendered again or the map says why the task is not
/// an owner.
#[test]
fn a_new_sc_citation_fails_until_the_table_or_the_map_follows() {
    let repo = Repo::with_map("citation", &[1, 2]);
    repo.commit_as_base();
    let ok = repo.run(&["check"]);
    assert_eq!(ok.status.code(), Some(0), "{}", stderr(&ok));
    let cited = format!(
        "{EPICS}| QUAL-E13-T09 | a microbenchmark | M1a | none | reported against SC-2 | \
         RFC-0061 §9 |\n"
    );
    repo.write(EPICS_PATH, &cited);
    let output = repo.run(&["check"]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    let err = stderr(&output);
    assert!(
        err.contains(
            "SC-2: `QUAL-E13-T09` is an owner now (its acceptance criteria name SC-2), but the \
             table doesn't list it"
        ),
        "{err}"
    );
    assert!(err.contains("section is stale"), "{err}");

    // Either the table follows...
    let render = repo.run(&["render", "--write"]);
    assert!(render.status.success());
    assert!(repo.rfc().contains("pending; QUAL-E13-T02, QUAL-E13-T09 |"));
    let output = repo.run(&["check"]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));

    // ...or the map says why the citation is not ownership, and the table drops it again.
    let map = fs::read_to_string(repo.root.join(sc_map::SC_MAP)).expect("map");
    repo.write(
        sc_map::SC_MAP,
        &format!(
            "{map}[[not_owner]]\ntask = \"QUAL-E13-T09\"\nsc = [\"SC-2\"]\n\
             reason = \"it reports SC-2 for context only\"\n"
        ),
    );
    let output = repo.run(&["check"]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("the table lists `QUAL-E13-T09`, which is no longer an owner"),
        "{}",
        stderr(&output)
    );
    let _ = repo.run(&["render", "--write"]);
    let output = repo.run(&["check"]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));

    // An exclusion whose citation is gone is stale.
    repo.write(EPICS_PATH, EPICS);
    let output = repo.run(&["check"]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("not_owner `QUAL-E13-T09` is not a task"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn a_new_file_in_a_shallow_clone_fails_closed() {
    let origin = Repo::with_map("shallow-origin", &[1, 2]);
    origin.commit_as_base();
    origin.write("README", "second commit\n");
    origin.commit_as_base();
    let clone = Repo {
        root: std::env::temp_dir().join(format!(
            "syntropy-bench-{}-shallow-clone",
            std::process::id()
        )),
    };
    let _ = fs::remove_dir_all(&clone.root);
    let url = format!("file://{}", origin.root.display());
    let status = Command::new("git")
        .args(["clone", "--quiet", "--depth", "1", &url])
        .arg(&clone.root)
        .status()
        .expect("git runs");
    assert!(status.success());
    clone.write(SCENARIO_PATH, SCENARIO);
    let _ = clone.run(&["render", "--write"]);
    let output = clone.run(&["check", "--base", "HEAD"]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    let err = stderr(&output);
    assert!(err.contains("this clone's history is shallow"), "{err}");
    assert!(err.contains("git fetch --unshallow"), "{err}");
}

#[cfg(unix)]
#[test]
fn an_unreadable_scenario_file_is_reported() {
    let repo = Repo::with_map("unreadable", &[1, 2]);
    repo.commit_as_base();
    let path = repo
        .root
        .join("bench/scenarios/decode.single.gone.q4km.toml");
    fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    std::os::unix::fs::symlink("no-such-target.toml", &path).expect("symlink");
    let output = repo.run(&["check"]);
    assert_eq!(output.status.code(), Some(1));
    let err = stderr(&output);
    assert!(
        err.contains(
            "bench/scenarios/decode.single.gone.q4km.toml: cannot read, so its \
             `scenario_version` cannot be checked"
        ),
        "{err}"
    );
}

/// A Windows checkout with no `.gitattributes` has CRLF line endings in every text file, while
/// the committed blobs keep LF. `check` reads through that, the version rule compares the file
/// with its blob by content, `render` prints LF, and `render --write` keeps the RFC's CRLF.
#[test]
fn a_crlf_checkout_checks_and_renders() {
    let repo = Repo::with_map("crlf", &[1, 2]);
    repo.write(SCENARIO_PATH, SCENARIO);
    let _ = repo.run(&["render", "--write"]);
    repo.commit_as_base();
    for rel in [
        "plan/00-vision-and-scope.md",
        EPICS_PATH,
        sc_map::SC_MAP,
        sc_map::RFC_0061,
        SCENARIO_PATH,
    ] {
        let text = fs::read_to_string(repo.root.join(rel)).expect(rel);
        repo.write(rel, &text.replace('\n', "\r\n"));
    }
    let output = repo.run(&["check"]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));

    // A map edit on the CRLF checkout makes the table stale, and rendering keeps CRLF.
    let map = fs::read_to_string(repo.root.join(sc_map::SC_MAP)).expect("map");
    repo.write(sc_map::SC_MAP, &map.replace("≤ 1.05×", "≤ 1.10×"));
    let output = repo.run(&["check"]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("section is stale"),
        "{}",
        stderr(&output)
    );
    let render = repo.run(&["render", "--write"]);
    assert!(render.status.success(), "{}", stderr(&render));
    let rfc = repo.rfc();
    assert!(rfc.contains("≤ 1.10×"));
    assert!(
        !rfc.replace("\r\n", "").contains('\n'),
        "render --write left a bare LF in a CRLF file"
    );
    let output = repo.run(&["check"]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let printed = repo.run(&["render"]);
    assert!(!String::from_utf8_lossy(&printed.stdout).contains('\r'));
}

/// The checked-in map covers every success criterion, its owners are real tasks, and
/// RFC-0061 §4.2's table is current. `lint` checks the same on every change; this keeps
/// `cargo test` honest too.
#[test]
fn the_repository_map_is_complete_and_the_table_current() {
    assert_repository_current(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."));
}

/// The same check on a copy of the repository's inputs with CRLF line endings, as a Windows
/// checkout with no `.gitattributes` has them.
#[test]
fn the_repository_map_is_current_on_a_crlf_checkout() {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let copy =
        std::env::temp_dir().join(format!("syntropy-bench-{}-repo-crlf", std::process::id()));
    let _ = fs::remove_dir_all(&copy);
    let crlf = |rel: &Path| {
        let text = fs::read_to_string(source.join(rel))
            .expect("read")
            .replace("\r\n", "\n");
        let to = copy.join(rel);
        fs::create_dir_all(to.parent().expect("parent")).expect("mkdir");
        fs::write(&to, text.replace('\n', "\r\n")).expect("write");
    };
    for rel in [sc_map::SC_MAP, sc_map::VISION, sc_map::RFC_0061] {
        crlf(Path::new(rel));
    }
    for dir in [sc_map::EPICS, scenario::SCENARIO_DIR] {
        let Ok(entries) = fs::read_dir(source.join(dir)) else {
            continue;
        };
        for entry in entries.flatten() {
            crlf(&Path::new(dir).join(entry.file_name()));
        }
    }
    let rfc = fs::read_to_string(copy.join(sc_map::RFC_0061)).expect("rfc");
    assert!(rfc.contains("\r\n"), "the copy has CRLF line endings");
    assert_repository_current(&copy);
    let _ = fs::remove_dir_all(&copy);
}

fn assert_repository_current(root: &Path) {
    let read = |rel: &str| sc_map::to_lf(&fs::read_to_string(root.join(rel)).expect(rel));
    let map = sc_map::parse(&read(sc_map::SC_MAP)).expect("map parses");
    let vision = sc_map::vision_ids(&read(sc_map::VISION));
    assert_eq!(vision, (1..=32).collect::<Vec<u32>>());
    let tasks = sc_map::tasks(&root.join(sc_map::EPICS));
    assert_eq!(sc_map::check(&map, &vision, &tasks), Vec::<String>::new());
    let rfc = read(sc_map::RFC_0061);
    let rigs = scenario::rig_ids(&rfc);
    for rig in ["R-ADA", "R-BW", "R-ULTRA-A", "R-TB5", "R-PAIR10", "S2"] {
        assert!(rigs.contains(rig), "{rig} missing from {rigs:?}");
    }
    let (scenarios, problems) = scenario::load(root, &rigs);
    assert_eq!(problems, Vec::new());
    let ids: Vec<String> = scenarios.iter().map(|s| s.id.0.clone()).collect();
    assert_eq!(
        sc_map::section(&rfc),
        Some(sc_map::render(&map, &tasks, &ids)),
        "RFC-0061 §4.2 is stale: run `syntropy-bench scenarios render --write`"
    );
}
