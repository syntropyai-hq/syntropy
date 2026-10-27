// SPDX-License-Identifier: MIT OR Apache-2.0

//! PROD-E1-T12's acceptance tests, through the binary: `config set`, `unset` and `edit`
//! (RFC-0054 §10.2; RFC-0051 §2.2).
//!
//! - Golden edits: on a file with comments and odd spacing, each edit changes only the lines of
//!   its key, and the file's other bytes stay as they were.
//! - Validate before write: an edit that would make the configuration invalid changes nothing
//!   and exits 5, with RFC-0054 §8's diagnostics on stderr.
//! - `edit` refuses to keep an invalid file without `--force`, and never asks without a
//!   terminal. The editor is a fake: this test binary itself, run as `EDITOR` with the test
//!   `fake_editor`, which writes the text the test gives it.
//! - A canary secret never reaches the output: not as a value written, not in a refused one.
//!
//! Each command runs [`common::isolated`], with the file under test named by `--config`, which
//! replaces the system, drop-in and user files (RFC-0054 §1, order 4). One test writes the user
//! file under the isolated home.

mod common;

use std::path::{Path, PathBuf};

use common::{code, finish, isolated};

/// The canary. No output may contain it.
const CANARY: &str = "cnry-Lw83Qx1v";

/// A file as a person writes it.
const FILE: &str = "\
# My node.   Edited by hand.

[api]   # the API
  # where to listen
  bind   =   [\"127.0.0.1:7499\"]   # loopback only
  tls = \"auto\"

# discovery, tuned
[discovery]
port=7999
";

/// The fake editor's text, in the child only.
const FAKE_TEXT: &str = "FAKE_EDITOR_TEXT";
/// Set in the child to make the fake editor fail.
const FAKE_FAIL: &str = "FAKE_EDITOR_FAIL";

/// Not a test of its own: the editor `EDITOR` names in these tests. Run as
/// `<this binary> --exact fake_editor <file>`, it writes `FAKE_EDITOR_TEXT` to the file, or
/// exits 3 with `FAKE_EDITOR_FAIL`. Run as a test, with neither set, it does nothing.
#[test]
fn fake_editor() {
    if std::env::var_os(FAKE_FAIL).is_some() {
        std::process::exit(3);
    }
    if let Ok(text) = std::env::var(FAKE_TEXT) {
        let file = std::env::args().next_back().expect("the file to edit");
        std::fs::write(file, text).unwrap();
    }
}

/// `EDITOR` for the fake editor.
fn fake_editor_command() -> String {
    format!(
        "{} --exact fake_editor --quiet",
        std::env::current_exe().unwrap().display()
    )
}

/// A test's directory, with `config.toml` holding `text`.
fn setup(tag: &str, text: &str) -> (PathBuf, PathBuf) {
    let dir = common::temp(&format!("write-{tag}"));
    let file = dir.join("config.toml");
    std::fs::write(&file, text).unwrap();
    (dir, file)
}

fn run(args: &[&str], home: &Path) -> std::process::Output {
    finish(&mut isolated(args, home))
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).unwrap()
}

#[test]
fn set_changes_only_the_target_line() {
    let (dir, file) = setup("set", FILE);
    let path = file.to_str().unwrap();
    let out = run(
        &["config", "set", "--config", path, "discovery.port", "8000"],
        &dir,
    );
    assert_eq!(code(&out), 0, "{}", text(&out.stderr));
    assert_eq!(
        std::fs::read_to_string(&file).unwrap(),
        FILE.replace("port=7999", "port=8000")
    );
    assert_eq!(
        text(&out.stdout),
        format!("set `discovery.port` in {path}\n")
    );
    assert!(text(&out.stderr).contains("`syntropy config reload` is not available"));

    // An array, replacing the value between its spacing and its comment.
    let out = run(
        &[
            "config",
            "set",
            "--config",
            path,
            "api.bind",
            r#"["0.0.0.0:7499"]"#,
        ],
        &dir,
    );
    assert_eq!(code(&out), 0, "{}", text(&out.stderr));
    let want = FILE.replace("port=7999", "port=8000").replace(
        r#"  bind   =   ["127.0.0.1:7499"]   # loopback only"#,
        r#"  bind   =   ["0.0.0.0:7499"]   # loopback only"#,
    );
    assert_eq!(std::fs::read_to_string(&file).unwrap(), want);

    // A new key in an existing table, indented like its neighbours.
    let out = run(
        &[
            "config",
            "set",
            "--config",
            path,
            "api.local_socket",
            "false",
        ],
        &dir,
    );
    assert_eq!(code(&out), 0, "{}", text(&out.stderr));
    let want = want.replace(
        "  tls = \"auto\"\n",
        "  tls = \"auto\"\n  local_socket = false\n",
    );
    assert_eq!(std::fs::read_to_string(&file).unwrap(), want);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn unset_removes_only_the_key_s_line() {
    let (dir, file) = setup("unset", FILE);
    let path = file.to_str().unwrap();
    let out = run(&["config", "unset", "--config", path, "api.bind"], &dir);
    assert_eq!(code(&out), 0, "{}", text(&out.stderr));
    assert_eq!(
        std::fs::read_to_string(&file).unwrap(),
        FILE.replace("  bind   =   [\"127.0.0.1:7499\"]   # loopback only\n", "")
    );
    assert_eq!(
        text(&out.stdout),
        format!("removed `api.bind` from {path}\n")
    );
    // Again: nothing to remove, nothing written.
    let before = std::fs::read_to_string(&file).unwrap();
    let out = run(&["config", "unset", "--config", path, "api.bind"], &dir);
    assert_eq!(code(&out), 0);
    assert_eq!(
        text(&out.stdout),
        format!("`api.bind` is not set in {path}; nothing changed\n")
    );
    assert_eq!(std::fs::read_to_string(&file).unwrap(), before);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn an_invalid_set_changes_nothing_and_exits_5() {
    let (dir, file) = setup("invalid-set", FILE);
    let path = file.to_str().unwrap();
    for (key, value) in [
        // A wrong type.
        ("discovery.port", "\"seventy\""),
        // A value outside the key's choices.
        ("api.tls", "sometimes"),
        // An unknown key, holding the canary: the value is never shown.
        ("api.bnd", CANARY),
    ] {
        let out = run(&["config", "set", "--config", path, key, value], &dir);
        assert_eq!(code(&out), 5, "{key}: {}", text(&out.stderr));
        let stderr = text(&out.stderr);
        assert!(stderr.contains("error[config/"), "{key}: {stderr}");
        assert!(stderr.contains("was not changed"), "{key}: {stderr}");
        assert!(!stderr.contains(CANARY) && !text(&out.stdout).contains(CANARY));
        assert_eq!(std::fs::read_to_string(&file).unwrap(), FILE, "{key}");
    }
    // Not a key at all, and a key that is a table.
    for key in ["api..bind", "api"] {
        let out = run(&["config", "set", "--config", path, key, "1"], &dir);
        assert_eq!(code(&out), 5, "{key}: {}", text(&out.stderr));
        assert_eq!(std::fs::read_to_string(&file).unwrap(), FILE);
    }
    // `--cluster` needs the cluster configuration endpoint.
    let out = run(&["config", "set", "--cluster", "fidelity", "exact"], &dir);
    assert_eq!(code(&out), 5);
    assert!(text(&out.stderr).contains("not available in this build yet"));
    std::fs::remove_dir_all(dir).unwrap();
}

/// Without `--config`, the user file under the home: created when absent.
#[cfg(target_os = "linux")]
#[test]
fn set_writes_the_user_file_by_default() {
    let dir = common::temp("write-user");
    let out = run(&["config", "set", "api.tls", "none"], &dir);
    assert_eq!(code(&out), 0, "{}", text(&out.stderr));
    let user = dir.join("home/.config/syntropy/config.toml");
    assert_eq!(
        std::fs::read_to_string(&user).unwrap(),
        "[api]\ntls = \"none\"\n"
    );
    {
        use std::os::unix::fs::PermissionsExt as _;
        let mode = std::fs::metadata(&user).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "a new config file is private");
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(unix)]
#[test]
fn a_symbolic_link_is_never_followed() {
    let (dir, real) = setup("link", FILE);
    let link = dir.join("link.toml");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    let path = link.to_str().unwrap();
    for args in [
        vec!["config", "set", "--config", path, "discovery.port", "1"],
        vec!["config", "unset", "--config", path, "discovery.port"],
    ] {
        let out = run(&args, &dir);
        assert_eq!(code(&out), 1, "{args:?}: {}", text(&out.stderr));
        assert!(
            text(&out.stderr).contains("symbolic link"),
            "{}",
            text(&out.stderr)
        );
        assert_eq!(std::fs::read_to_string(&real).unwrap(), FILE);
    }
    std::fs::remove_dir_all(dir).unwrap();
}

/// `edit` through the fake editor.
fn edit(dir: &Path, path: &str, write: Option<&str>, extra: &[&str]) -> std::process::Output {
    let mut args = vec!["config", "edit", "--config", path];
    args.extend_from_slice(extra);
    let mut command = isolated(&args, dir);
    command.env("EDITOR", fake_editor_command());
    match write {
        Some(text) => command.env(FAKE_TEXT, text),
        None => command.env(FAKE_FAIL, "1"),
    };
    finish(&mut command)
}

#[test]
fn edit_saves_a_valid_file() {
    let (dir, file) = setup("edit-ok", FILE);
    let path = file.to_str().unwrap();
    let new = FILE.replace("port=7999", "port=7000");
    let out = edit(&dir, path, Some(&new), &[]);
    assert_eq!(code(&out), 0, "{}", text(&out.stderr));
    assert_eq!(std::fs::read_to_string(&file).unwrap(), new);
    // The fake editor is a test binary and prints libtest's lines on the stdout it inherits, so
    // stdout is checked for the command's own line.
    assert!(
        text(&out.stdout).contains(&format!("saved {path}\n")),
        "{}",
        text(&out.stdout)
    );
    // No copy is left behind.
    assert!(
        names(&dir)
            .iter()
            .all(|n| !n.contains("syntropy-edit") && !n.ends_with(".rejected")),
        "{:?}",
        names(&dir)
    );
    // The editor changed nothing.
    let out = edit(&dir, path, Some(&new), &[]);
    assert_eq!(code(&out), 0);
    assert!(text(&out.stdout).contains(&format!("no change to {path}\n")));
    std::fs::remove_dir_all(dir).unwrap();
}

/// The directory's entries.
fn names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

#[test]
fn edit_refuses_to_keep_an_invalid_file_without_force() {
    let (dir, file) = setup("edit-bad", FILE);
    let path = file.to_str().unwrap();
    let bad = format!("{FILE}bogus_key = \"{CANARY}\"\nport_typo = 1\n");
    // No terminal: no question, exit 5, the file unchanged, the edit kept beside it.
    let out = edit(&dir, path, Some(&bad), &[]);
    let stderr = text(&out.stderr);
    assert_eq!(code(&out), 5, "{stderr}");
    assert!(!stderr.contains("[y/N]"), "{stderr}");
    assert!(stderr.contains("error[config/"), "{stderr}");
    assert!(stderr.contains("was not changed"), "{stderr}");
    assert!(
        stderr.contains("hint: syntropy config edit --force"),
        "{stderr}"
    );
    assert!(!stderr.contains(CANARY) && !text(&out.stdout).contains(CANARY));
    assert_eq!(std::fs::read_to_string(&file).unwrap(), FILE);
    let rejected = dir.join("config.toml.rejected");
    assert_eq!(std::fs::read_to_string(&rejected).unwrap(), bad);
    assert!(stderr.contains("config.toml.rejected"), "{stderr}");

    // With `--force` it is kept, and the command says it is invalid.
    let out = edit(&dir, path, Some(&bad), &["--force"]);
    assert_eq!(code(&out), 0, "{}", text(&out.stderr));
    assert_eq!(std::fs::read_to_string(&file).unwrap(), bad);
    assert!(
        text(&out.stdout).contains("which is invalid"),
        "{}",
        text(&out.stdout)
    );
    assert!(!text(&out.stderr).contains(CANARY));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_failed_editor_changes_nothing() {
    let (dir, file) = setup("edit-fail", FILE);
    let path = file.to_str().unwrap();
    let out = edit(&dir, path, None, &[]);
    assert_eq!(code(&out), 1, "{}", text(&out.stderr));
    assert!(
        text(&out.stderr).contains("the editor exited with"),
        "{}",
        text(&out.stderr)
    );
    assert_eq!(std::fs::read_to_string(&file).unwrap(), FILE);
    assert!(!dir.join("config.toml.rejected").exists());
    std::fs::remove_dir_all(dir).unwrap();
}

/// Without a terminal and without `EDITOR` or `VISUAL`, the default editor cannot run: exit 2,
/// and the file is not touched.
#[test]
fn edit_without_a_terminal_or_an_editor_exits_2() {
    let (dir, file) = setup("edit-none", FILE);
    let path = file.to_str().unwrap();
    let out = run(&["config", "edit", "--config", path], &dir);
    assert_eq!(code(&out), 2, "{}", text(&out.stderr));
    assert!(
        text(&out.stderr).contains("no terminal"),
        "{}",
        text(&out.stderr)
    );
    assert_eq!(std::fs::read_to_string(&file).unwrap(), FILE);
    assert_eq!(
        names(&dir)
            .iter()
            .filter(|n| n.contains("syntropy-edit"))
            .count(),
        0
    );
    std::fs::remove_dir_all(dir).unwrap();
}

/// A canary in the variables the CLI reads as secrets never reaches `config show`, `check` or
/// `path`. The binary's own sections hold no secret-typed key yet, so the config crate's
/// `tests/secrets.rs` checks the secret forms against a test registry.
#[test]
fn the_secret_variables_never_reach_config_output() {
    let (dir, file) = setup("canary-env", FILE);
    let path = file.to_str().unwrap();
    for args in [
        vec!["config", "show", "--resolved", "--config", path],
        vec!["config", "check", "--config", path],
        vec!["config", "path", "--config", path],
    ] {
        let out = finish(
            isolated(&args, &dir)
                .env("SYNTROPY_API_KEY", CANARY)
                .env("SYNTROPY_HF_TOKEN", CANARY),
        );
        assert_eq!(code(&out), 0, "{args:?}: {}", text(&out.stderr));
        assert!(
            !text(&out.stdout).contains(CANARY) && !text(&out.stderr).contains(CANARY),
            "{args:?}"
        );
    }
    std::fs::remove_dir_all(dir).unwrap();
}
