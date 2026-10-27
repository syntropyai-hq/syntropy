// SPDX-License-Identifier: MIT OR Apache-2.0

//! PROD-E6-T23's acceptance tests, through the binary: `config show --resolved`, `config check`
//! and `config path` (RFC-0054 §1, §2.2, §2.4, §8; RFC-0051 §2.2).
//!
//! Each command runs with no daemon, with every `SYNTROPY_*` variable of the test's own
//! environment removed, and with the home, XDG, `%LOCALAPPDATA%`, `%PROGRAMDATA%` and `TMPDIR`
//! locations moved into the test's directory, so nothing on this machine's own configuration
//! reaches it. `show` and `check` read their file through `--config`, which replaces the system,
//! drop-in and user files (RFC-0054 §1, order 4).

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use common::{code, finish, syntropy};

/// A fresh directory for one test.
fn temp(tag: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("syntropy-cli-config-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `syntropy` with `args`, isolated in `home` as the module docs say.
fn isolated(args: &[&str], home: &Path) -> Command {
    let mut command = syntropy(args);
    for (name, _) in std::env::vars_os() {
        if name.to_string_lossy().starts_with("SYNTROPY_") {
            command.env_remove(name);
        }
    }
    for var in [
        "HOME",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "TMPDIR",
        "XDG_RUNTIME_DIR",
    ] {
        command.env(var, home.join(var.to_ascii_lowercase()));
    }
    for var in ["XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_DATA_HOME"] {
        command.env_remove(var);
    }
    command
}

/// The text of a stream.
fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).unwrap()
}

/// `line` with the space before its `#` comment collapsed to two, so a golden line does not
/// depend on the longest key of every section.
fn normalized(line: &str) -> String {
    match line.split_once("  # ") {
        Some((left, comment)) => format!("{}  # {comment}", left.trim_end()),
        None => line.to_owned(),
    }
}

/// The golden file's lines.
fn golden() -> Vec<String> {
    include_str!("golden/config_show_resolved.txt")
        .lines()
        .map(str::to_owned)
        .collect()
}

/// `config show --resolved` names each value's layer: the file and line, the environment
/// variable, the flag, or the default (RFC-0054 §2.2; RFC-0001 §11.2).
#[test]
fn show_resolved_names_the_file_line_variable_or_flag_of_each_value() {
    let home = temp("show");
    let file = home.join("config.toml");
    std::fs::write(&file, "[api]\nbind = [\"127.0.0.1:7499\"]\n").unwrap();
    let file_arg = file.to_str().unwrap();
    let output = finish(
        isolated(
            &[
                "config",
                "show",
                "--resolved",
                "--config",
                file_arg,
                "--set",
                "api.allow_plaintext_lan=true",
            ],
            &home,
        )
        .env("SYNTROPY_DISCOVERY__PORT", "7999"),
    );
    assert_eq!(code(&output), 0, "{}", text(&output.stderr));
    assert_eq!(text(&output.stderr), "", "no warning expected");
    let shown: Vec<String> = text(&output.stdout)
        .lines()
        .map(|line| normalized(&line.replace(file_arg, "<file>")))
        .collect();
    // Every golden line appears, in the golden file's order.
    let mut at = 0;
    for want in golden() {
        let found = shown[at..]
            .iter()
            .position(|line| *line == want)
            .unwrap_or_else(|| panic!("missing or out of order: {want}\n{}", shown.join("\n")));
        at += found + 1;
    }
    // Every key the schema knows is listed, each once.
    let keys = shown.iter().filter(|line| !line.starts_with('#')).count();
    assert!(keys > 100, "{keys} keys");
    let _ = std::fs::remove_dir_all(home);
}

/// An unknown `SYNTROPY_*` variable is a warning on stderr; the result still goes to stdout
/// (RFC-0054 §2.4).
#[test]
fn an_unknown_syntropy_variable_is_a_warning_on_stderr() {
    let home = temp("unknown-var");
    let file = home.join("config.toml");
    std::fs::write(&file, "").unwrap();
    for sub in ["show", "check"] {
        let mut args = vec!["config", sub, "--config", file.to_str().unwrap()];
        if sub == "show" {
            args.push("--resolved");
        }
        let output = finish(isolated(&args, &home).env("SYNTROPY_NOT_A_KEY", "1"));
        assert_eq!(code(&output), 0, "{sub}: {}", text(&output.stderr));
        let stderr = text(&output.stderr);
        assert!(
            stderr.starts_with(
                "warning[config/0003]: unknown variable: `SYNTROPY_NOT_A_KEY` names no config key"
            ),
            "{sub}: {stderr}"
        );
        assert!(
            stderr.contains("--> env SYNTROPY_NOT_A_KEY"),
            "{sub}: {stderr}"
        );
        assert!(
            !text(&output.stdout).contains("SYNTROPY_NOT_A_KEY"),
            "{sub}"
        );
    }
    let _ = std::fs::remove_dir_all(home);
}

/// `config check` needs no daemon: 0 on a valid layer.
#[test]
fn check_exits_0_on_a_valid_layer() {
    let home = temp("check-ok");
    let file = home.join("config.toml");
    std::fs::write(&file, "[api]\nbind = [\"127.0.0.1:7433\"]\n").unwrap();
    let output = finish(
        isolated(
            &["config", "check", "--config", file.to_str().unwrap()],
            &home,
        )
        .env("SYNTROPY_HOST", "http://127.0.0.1:1"),
    );
    assert_eq!(code(&output), 0, "{}", text(&output.stderr));
    assert_eq!(
        text(&output.stdout),
        "config check: ok (1 file read, 0 warnings)\n"
    );
    assert_eq!(text(&output.stderr), "");
    let _ = std::fs::remove_dir_all(home);
}

/// `config check` exits 5 on an invalid layer and lists every error, from the file, the
/// environment and the flags together (RFC-0054 §8).
#[test]
fn check_exits_5_listing_every_error() {
    let home = temp("check-bad");
    let file = home.join("config.toml");
    std::fs::write(&file, "[api]\nbnd = 1\n[discovery]\nport = \"seventy\"\n").unwrap();
    let output = finish(
        isolated(
            &[
                "config",
                "check",
                "--config",
                file.to_str().unwrap(),
                "--set",
                "swim.period=often",
            ],
            &home,
        )
        .env("SYNTROPY_API__LOCAL_SOCKET", "maybe"),
    );
    assert_eq!(code(&output), 5);
    let stderr = text(&output.stderr);
    let errors: Vec<&str> = stderr.lines().filter(|l| l.starts_with("error[")).collect();
    assert_eq!(errors.len(), 4, "{stderr}");
    for want in [
        "`api.bnd`",
        "discovery.port",
        "api.local_socket",
        "swim.period",
    ] {
        assert!(stderr.contains(want), "{want} missing: {stderr}");
    }
    assert!(
        stderr.contains(&format!("--> {}:2", file.display())),
        "{stderr}"
    );
    assert!(
        stderr.contains("--> env SYNTROPY_API__LOCAL_SOCKET"),
        "{stderr}"
    );
    assert!(stderr.contains("--> flag --set swim.period"), "{stderr}");
    assert_eq!(text(&output.stdout), "config check: 4 errors, 0 warnings\n");
    let _ = std::fs::remove_dir_all(home);
}

/// `config path` prints RFC-0054 §1's four orders and the state, store and runtime locations,
/// as the OS this runs on lays them out.
#[test]
fn path_prints_the_four_orders_and_the_three_locations() {
    let home = temp("path");
    let output = finish(&mut isolated(&["config", "path"], &home));
    assert_eq!(code(&output), 0, "{}", text(&output.stderr));
    let stdout = text(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    for prefix in [
        "config files, in merge order (later wins):",
        "  1  system file    ",
        "  2  drop-ins       ",
        "  3  user file      ",
        "  4  explicit file  (none; `--config` or SYNTROPY_CONFIG replaces 1-3)",
        "directories:",
        "  state    ",
        "  store    ",
        "  runtime  ",
    ] {
        assert!(
            lines.iter().any(|l| l.starts_with(prefix)),
            "{prefix}\n{stdout}"
        );
    }
    let home_dir = home.join("home");
    let (system, user, state, runtime) = if cfg!(windows) {
        let local = home.join("localappdata").join("Syntropy");
        (
            home.join("programdata")
                .join("Syntropy")
                .join("config.toml"),
            local.join("config.toml"),
            local.join("state"),
            r"\\.\pipe\syntropy-<user SID>-*".to_owned(),
        )
    } else if cfg!(target_os = "macos") {
        let support = home_dir
            .join("Library")
            .join("Application Support")
            .join("Syntropy");
        (
            PathBuf::from("/Library/Application Support/Syntropy/config.toml"),
            support.join("config.toml"),
            support.join("state"),
            home.join("tmpdir").join("syntropy").display().to_string(),
        )
    } else {
        (
            PathBuf::from("/etc/syntropy/config.toml"),
            home_dir
                .join(".config")
                .join("syntropy")
                .join("config.toml"),
            home_dir.join(".local").join("state").join("syntropy"),
            home.join("xdg_runtime_dir")
                .join("syntropy")
                .display()
                .to_string(),
        )
    };
    for want in [
        format!("  1  system file    {}", system.display()),
        format!("  3  user file      {}  (absent)", user.display()),
        format!("  state    {}", state.display()),
        format!("  runtime  {runtime}"),
    ] {
        assert!(
            lines.iter().any(|l| l.starts_with(&want)),
            "{want}\n{stdout}"
        );
    }

    // With `--config`, order 4 names the file and says it replaces 1–3.
    let file = home.join("explicit.toml");
    std::fs::write(&file, "").unwrap();
    let output = finish(&mut isolated(
        &["config", "path", "--config", file.to_str().unwrap()],
        &home,
    ));
    assert_eq!(code(&output), 0);
    let stdout = text(&output.stdout);
    assert!(
        stdout.contains(&format!(
            "  4  explicit file  {}  (from --config; found)",
            file.display()
        )),
        "{stdout}"
    );
    assert!(
        stdout.contains("(replaced by the explicit file)"),
        "{stdout}"
    );
    let _ = std::fs::remove_dir_all(home);
}

/// The paths of a missing `--config` file still print, with a warning; `show` and `check`
/// refuse it with exit 5, as they do a malformed `--set`, `--json`, and `show` without
/// `--resolved`.
#[test]
fn bad_arguments_exit_5_and_path_still_prints() {
    let home = temp("bad-args");
    let missing = home.join("missing.toml");
    let missing = missing.to_str().unwrap();
    let path = finish(&mut isolated(
        &["config", "path", "--config", missing],
        &home,
    ));
    assert_eq!(code(&path), 0);
    assert!(
        text(&path.stderr).starts_with("warning: "),
        "{}",
        text(&path.stderr)
    );
    assert!(text(&path.stdout).contains("(from --config; absent)"));

    let file = home.join("config.toml");
    std::fs::write(&file, "").unwrap();
    let file = file.to_str().unwrap();
    for (args, code_name) in [
        (
            vec!["config", "check", "--config", missing],
            "error[config/path_unusable]",
        ),
        (
            vec!["config", "show", "--resolved", "--config", missing],
            "error[config/path_unusable]",
        ),
        (
            vec!["config", "check", "--config", file, "--set", "no-equals"],
            "error[config/syntax]",
        ),
        (
            vec!["config", "show", "--config", file],
            "error[cli/not_built]",
        ),
        (
            vec!["config", "check", "--config", file, "--json"],
            "error[cli/not_built]",
        ),
    ] {
        let output = finish(&mut isolated(&args, &home));
        assert_eq!(code(&output), 5, "{args:?}: {}", text(&output.stderr));
        assert!(
            text(&output.stderr).starts_with(code_name),
            "{args:?}: {}",
            text(&output.stderr)
        );
    }
    let _ = std::fs::remove_dir_all(home);
}

/// One case: the arguments, an environment variable to set, and the exit code.
type Case<'a> = (Vec<&'a str>, Option<(&'a str, &'a str)>, i32);

/// Text from the configuration's sources never reaches the terminal raw: the diagnostics quote a
/// file's value, an environment value, a variable's name and a flag's value, and each line is
/// escaped on the way out (B1 of the review of #1921).
#[test]
fn diagnostics_never_put_a_source_s_escape_sequences_on_the_terminal() {
    let home = temp("escapes");
    let hostile = home.join("hostile.toml");
    // A TOML basic string's `\u001b` and `\u0007` decode to a real ESC and BEL.
    std::fs::write(
        &hostile,
        "[api]\ntls = \"\\u001b]0;PWNED\\u0007\\u001b[2J\"\n",
    )
    .unwrap();
    let empty = home.join("empty.toml");
    std::fs::write(&empty, "").unwrap();
    let (hostile, empty) = (hostile.to_str().unwrap(), empty.to_str().unwrap());
    let cases: [Case<'_>; 4] = [
        // A file's value.
        (vec!["config", "check", "--config", hostile], None, 5),
        // An environment value.
        (
            vec!["config", "check", "--config", empty],
            Some(("SYNTROPY_API__TLS", "\u{1b}]0;PWNED\u{7}\u{1b}[2J")),
            5,
        ),
        // An environment variable's name.
        (
            vec!["config", "check", "--config", empty],
            Some(("SYNTROPY_X\u{1b}[31m", "1")),
            0,
        ),
        // A flag's value, on `show`.
        (
            vec![
                "config",
                "show",
                "--resolved",
                "--config",
                empty,
                "--set",
                "api.tls=\u{1b}[2J",
            ],
            None,
            5,
        ),
    ];
    for (args, env, want) in cases {
        let mut command = isolated(&args, &home);
        if let Some((name, value)) = env {
            command.env(name, value);
        }
        let output = finish(&mut command);
        let stderr = text(&output.stderr);
        assert_eq!(code(&output), want, "{args:?} {env:?}: {stderr}");
        assert!(
            !stderr.contains('\u{1b}') && !stderr.contains('\u{7}'),
            "{args:?} {env:?}: raw control characters on stderr: {stderr:?}"
        );
        assert!(stderr.contains("\\u{1b}"), "{args:?} {env:?}: {stderr}");
        let stdout = text(&output.stdout);
        assert!(
            !stdout.contains('\u{1b}') && !stdout.contains('\u{7}'),
            "{stdout:?}"
        );
    }
    let _ = std::fs::remove_dir_all(home);
}

/// Paths `config path` and `show --resolved` echo come from `--config`, `SYNTROPY_CONFIG` and the
/// home variables, so their control characters are escaped on stdout too (N1 of the review of
/// #1921).
#[test]
fn echoed_paths_never_put_escape_sequences_on_the_terminal() {
    let home = temp("path-escapes");
    let odd = home.join("ev\u{1b}]0;x\u{7}il\u{1b}[2J.toml");
    let odd = odd.to_str().unwrap();
    let clean = |output: &std::process::Output| {
        let (stdout, stderr) = (text(&output.stdout), text(&output.stderr));
        assert!(
            !stdout.contains('\u{1b}') && !stdout.contains('\u{7}'),
            "raw control characters on stdout: {stdout:?}"
        );
        assert!(
            !stderr.contains('\u{1b}') && !stderr.contains('\u{7}'),
            "raw control characters on stderr: {stderr:?}"
        );
        stdout
    };
    // From `--config`, and from `SYNTROPY_CONFIG`: the file need not exist.
    let output = finish(&mut isolated(&["config", "path", "--config", odd], &home));
    assert_eq!(code(&output), 0);
    assert!(clean(&output).contains("ev\\u{1b}]0;x\\u{7}il\\u{1b}[2J.toml"));
    let output = finish(isolated(&["config", "path"], &home).env("SYNTROPY_CONFIG", odd));
    assert_eq!(code(&output), 0);
    let stdout = clean(&output);
    assert!(
        stdout.contains("(from SYNTROPY_CONFIG; absent)"),
        "{stdout}"
    );
    assert!(stdout.contains("\\u{1b}"), "{stdout}");
    // From the home variables, which every location under them repeats.
    let var = if cfg!(windows) {
        "LOCALAPPDATA"
    } else {
        "HOME"
    };
    let output = finish(isolated(&["config", "path"], &home).env(var, home.join("h\u{1b}[31m")));
    assert_eq!(code(&output), 0);
    assert!(clean(&output).contains("h\\u{1b}[31m"));
    // `show --resolved` names the file in each origin. Only Unix lets a file name hold ESC.
    if cfg!(unix) {
        std::fs::write(odd, "[api]\nbind = [\"127.0.0.1:7433\"]\n").unwrap();
        let output = finish(&mut isolated(
            &["config", "show", "--resolved", "--config", odd],
            &home,
        ));
        assert_eq!(code(&output), 0, "{}", text(&output.stderr));
        assert!(clean(&output).contains("il\\u{1b}[2J.toml:2"));
    }
    let _ = std::fs::remove_dir_all(home);
}
