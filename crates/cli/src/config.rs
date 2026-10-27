// SPDX-License-Identifier: MIT OR Apache-2.0

//! `syntropy config show --resolved`, `config check` and `config path` (RFC-0051 §2.1; RFC-0054
//! §1, §2.2, §2.4; RFC-0001 §11.2; PROD-E6-T23), and `config set`, `unset` and `edit` (RFC-0054
//! §10.2; PROD-E1-T12).
//!
//! Each calls `syntropy-config`'s `command` or `write` functions over [`REGISTRY`], the binary's
//! section registry, with the process environment and the `--config` and `--set` flags. None
//! needs a daemon (RFC-0051 §1).
//!
//! A write edits one file: the user file, the system file under `--system`, or the explicit
//! file `--config` or `SYNTROPY_CONFIG` names. It changes only the lines of the key it sets,
//! validates the whole layer as it would be with the new text, RFC-0054 §8's phases 1–5 with no
//! environment overrides or flags, and writes nothing if the result is invalid. The file is
//! replaced atomically, keeping its permissions, and a symbolic link at its path is refused.
//! Nothing a write prints holds the value it wrote, which can be a secret. A running daemon
//! applies the change at its next start: `config reload` is PROD-E1-T11's.
//!
//! Output follows RFC-0051 §2.2. The command's result goes to stdout: the resolved values,
//! `check`'s summary line, or the paths. Every diagnostic goes to stderr, in RFC-0054 §8's form,
//! as the configuration's own renderer writes it: an error, a warning such as an unknown
//! `SYNTROPY_*` variable, and its source lines. The renderer quotes the configuration's sources,
//! so each line is escaped on the way out ([`syntropy_api::printable`]). An invalid configuration
//! exits 5.
//!
//! The files are those a user's daemon reads (`Mode::User`). `--json` is refused until the API
//! has a configuration type to print (`/v1/syntropy/config`, RFC-0054 §10.2).

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use syntropy_api::config::ApiSection;
use syntropy_api::rules::API_RULES;
use syntropy_config::{
    Checked, ConfigSection, DebugSection, Env, Flags, HostEnvironment, Locations, Mode, Os,
    Problem, Registry, Scan, SectionSchema, command, load_and_validate, locate, write,
};

use crate::args::{
    ConfigCheckArgs, ConfigEditArgs, ConfigPathArgs, ConfigSetArgs, ConfigShowArgs, ConfigSource,
    ConfigTarget, ConfigUnsetArgs,
};
use crate::error::{CliError, Exit};
use crate::output::{Output, Terminal};
use crate::prompt;

/// `syntropy-net`'s sections.
const NET: &[&SectionSchema] = syntropy_net::config::SECTIONS;
/// `syntropy-cluster`'s sections.
const CLUSTER: &[&SectionSchema] = syntropy_cluster::config::SECTIONS;
/// The sections of crates that declare one each: the API's listeners, the debug endpoints and
/// the scheduler's instance keys.
const SINGLES: [&SectionSchema; 3] = [
    &ApiSection::SCHEMA,
    &DebugSection::SCHEMA,
    &syntropy_sched::SchedSection::SCHEMA,
];

/// How many sections the binary links.
const COUNT: usize = NET.len() + CLUSTER.len() + SINGLES.len();

/// Every section the binary links, in one array: `net`'s, then `cluster`'s, then the others.
const ALL: [&SectionSchema; COUNT] = {
    let mut all = [SINGLES[0]; COUNT];
    let mut i = 0;
    while i < NET.len() {
        all[i] = NET[i];
        i += 1;
    }
    let mut j = 0;
    while j < CLUSTER.len() {
        all[NET.len() + j] = CLUSTER[j];
        j += 1;
    }
    let mut k = 0;
    while k < SINGLES.len() {
        all[NET.len() + CLUSTER.len() + k] = SINGLES[k];
        k += 1;
    }
    all
};

/// The binary's section registry: every configuration section of the crates it links. A
/// `static`, so two sections that claim one prefix fail the build. A crate that adds a section
/// lists it in its own `SECTIONS`, or here.
pub static REGISTRY: Registry = Registry::new(&ALL);

/// `config show`: with `--resolved`, every key's effective value and where it came from.
pub fn show(args: &ConfigShowArgs, output: Output) -> Result<Exit, CliError> {
    refuse_json(output, "show")?;
    if !args.resolved {
        return Err(CliError::not_built("config show")
            .detail("only the resolved view is built")
            .hint("syntropy config show --resolved"));
    }
    let checked = load(&args.source)?;
    let diagnostics = checked.render(&REGISTRY);
    eprint!("{}", printable_lines(&diagnostics));
    match &checked.outcome {
        Ok(config) => {
            // The values only: stderr already carries each warning in RFC-0054 §8's form, so
            // the one-line copies `show_resolved` appends come out of stdout.
            let mut values = command::show_resolved(config);
            for warning in config.warnings() {
                values = values.replacen(&format!("\nwarning: {warning}\n"), "", 1);
            }
            // Values are TOML-quoted, but each line's origin echoes a path or a variable's name
            // from the environment or the command line.
            print!("{}", printable_lines(&values));
            Ok(Exit::Success)
        }
        Err(_) => Err(CliError::cli(
            Exit::Invalid,
            "invalid_config",
            format!(
                "the configuration has {}, so nothing is shown",
                errors(checked.errors().len())
            ),
        )
        .hint("syntropy config check")),
    }
}

/// `config check`: every error and warning of RFC-0054 §8's phases 1–5, without a daemon.
pub fn check(args: &ConfigCheckArgs, output: Output) -> Result<Exit, CliError> {
    refuse_json(output, "check")?;
    let checked = load(&args.source)?;
    let diagnostics = checked.render(&REGISTRY);
    let report = command::check(&checked, &REGISTRY);
    eprint!("{}", printable_lines(&diagnostics));
    // The report is the diagnostics, then one summary line.
    // `command::check`'s text is `Checked::render` followed by the summary, and the render is
    // pure over the same `Checked`, so the prefix always matches
    // (`check_s_report_starts_with_its_diagnostics`). Were it ever not to, the whole report goes
    // to stdout, escaped like the rest.
    let summary = report
        .text
        .strip_prefix(&diagnostics)
        .unwrap_or(&report.text);
    print!("{}", printable_lines(summary));
    Ok(if report.exit == command::EXIT_OK {
        Exit::Success
    } else {
        Exit::Invalid
    })
}

/// `config path`: RFC-0054 §1's four orders, whether each file exists, and the state, store and
/// runtime locations.
pub fn path(args: &ConfigPathArgs, output: Output) -> Result<Exit, CliError> {
    refuse_json(output, "path")?;
    let (_, _, locations) = locations(args.config.as_deref());
    let scan = match locations.scan() {
        Ok(scan) => scan,
        // An explicit file that does not exist: the paths still print, with it absent.
        Err(problem) => {
            eprintln!(
                "warning: {}",
                syntropy_api::printable::printable(&problem.to_string())
            );
            Scan {
                system_file: None,
                dropins: Vec::new(),
                user_file: None,
                explicit: None,
            }
        }
    };
    // Every path comes from the environment or `--config`: `SYNTROPY_CONFIG`, `HOME`, the XDG
    // and Windows variables, `TMPDIR`.
    print!("{}", printable_lines(&command::path(&locations, &scan)));
    Ok(Exit::Success)
}

/// `config set <key> <value>`: the key set in the target file, the rest of the file unchanged.
pub fn set(args: &ConfigSetArgs, output: Output) -> Result<Exit, CliError> {
    refuse_json(output, "set")?;
    if args.cluster {
        return Err(cluster_not_built("set"));
    }
    let value = if args.value == "-" {
        value_from_stdin(&args.key)?
    } else {
        if holds_secret(&REGISTRY, &args.key) && !names_a_source(&args.value) {
            eprintln!(
                "warning: `{}` is a secret, and its value is on the command line, where shell \
                 history and the process list can see it; pass `-` to read it from stdin, or set \
                 `{{ file = \"…\" }}` or `{{ env = \"…\" }}`",
                printable(&args.key)
            );
        }
        args.value.clone()
    };
    let target = Target::of(&args.target)?;
    let old = target.read()?;
    let new = write::set(&old, &args.key, &value).map_err(|e| edit_failed(&target, &e))?;
    target.save(&new, false)?;
    println!("set `{}` in {}", printable(&args.key), target.shown());
    reload_note();
    Ok(Exit::Success)
}

/// `config unset <key>`: the key's line removed from the target file, its comments kept.
pub fn unset(args: &ConfigUnsetArgs, output: Output) -> Result<Exit, CliError> {
    refuse_json(output, "unset")?;
    if args.cluster {
        return Err(cluster_not_built("unset"));
    }
    let target = Target::of(&args.target)?;
    let old = target.read()?;
    let Some(new) = write::unset(&old, &args.key).map_err(|e| edit_failed(&target, &e))? else {
        println!(
            "`{}` is not set in {}; nothing changed",
            printable(&args.key),
            target.shown()
        );
        return Ok(Exit::Success);
    };
    target.save(&new, false)?;
    println!("removed `{}` from {}", printable(&args.key), target.shown());
    reload_note();
    Ok(Exit::Success)
}

/// `config set <key> -`: the value from stdin, without one trailing line break. Piped in, it is
/// read whole. On a terminal it is read as one line with echo off, after a prompt on stderr
/// ([`read_hidden_line`]), so a secret never shows on the screen.
fn value_from_stdin(key: &str) -> Result<String, CliError> {
    use std::io::{IsTerminal as _, Read as _};
    let failed = |e: std::io::Error| {
        CliError::cli(
            Exit::Failure,
            "read_failed",
            format!("cannot read the value from stdin: {e}"),
        )
    };
    let stdin = std::io::stdin();
    let mut value = String::new();
    if stdin.is_terminal() {
        value = read_hidden_line(&format!("value for `{}`: ", printable(key))).map_err(|e| {
            CliError::cli(
                Exit::Failure,
                "read_failed",
                format!("cannot read the value from the terminal with echo off: {e}"),
            )
            .hint("pipe the value in instead, as `printf '%s' … | syntropy config set <key> -`")
        })?;
    } else {
        stdin.lock().read_to_string(&mut value).map_err(failed)?;
    }
    if let Some(line) = value.strip_suffix('\n') {
        value.truncate(line.strip_suffix('\r').unwrap_or(line).len());
    }
    if value.is_empty() {
        return Err(CliError::cli(
            Exit::Invalid,
            "no_value",
            "`-` reads the value from stdin, and stdin was empty",
        ));
    }
    Ok(value)
}

/// The Unix half of [`read_hidden_line`]: `sh` saves the terminal's state, turns echo off, prints
/// the prompt (`$1`), reads one line and writes it to its stdout, a pipe to this process. Traps
/// put the terminal back on exit and on an interrupt, so a Ctrl-C never leaves it without echo.
#[cfg(unix)]
const HIDDEN_READ: &str = r#"s=$(stty -g) || exit 3
trap 'stty "$s"' EXIT
trap 'stty "$s"; exit 130' INT TERM HUP
stty -echo || exit 3
printf '%s' "$1" >&2
IFS= read -r v || [ -n "$v" ] || exit 4
printf '%s' "$v"
"#;

/// The Windows half of [`read_hidden_line`]: PowerShell's `Read-Host -AsSecureString`, which shows
/// `*` for each character typed, written to its stdout, a pipe to this process, as UTF-8. The
/// prompt comes in `HIDDEN_READ_PROMPT`: `-Command` would join an argument into the script.
#[cfg(windows)]
const HIDDEN_READ: &str = "[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false; \
    [Console]::Error.Write($env:HIDDEN_READ_PROMPT); $s = Read-Host -AsSecureString; \
    $p = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($s); \
    try { [Console]::Out.Write([Runtime.InteropServices.Marshal]::PtrToStringBSTR($p)) } \
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($p) }";

/// One line from the terminal on stdin, typed with echo off, after `prompt` on stderr.
///
/// This crate has no `unsafe` (RFC-0001 §4.2), and turning echo off is a terminal-mode call, so
/// the read runs in a short child: `sh` with `stty` on Unix, PowerShell on Windows. The child
/// shares this process's terminal, and hands the line back through a pipe.
fn read_hidden_line(prompt: &str) -> std::io::Result<String> {
    use std::process::{Command, Stdio};
    #[cfg(unix)]
    let mut command = {
        let mut command = Command::new("/bin/sh");
        command.args(["-c", HIDDEN_READ, "sh", prompt]);
        command
    };
    #[cfg(windows)]
    let mut command = {
        let mut command = Command::new("powershell.exe");
        command
            .args(["-NoProfile", "-Command", HIDDEN_READ])
            .env("HIDDEN_READ_PROMPT", prompt);
        command
    };
    #[cfg(not(any(unix, windows)))]
    return Err(std::io::Error::other(
        "no way to turn the terminal's echo off here",
    ));
    let output = command
        .stdin(Stdio::inherit())
        .stderr(Stdio::inherit())
        .stdout(Stdio::piped())
        .output()?;
    // Enter was not echoed either.
    eprintln!();
    match output.status.code() {
        Some(0) => String::from_utf8(output.stdout)
            .map_err(|_| std::io::Error::other("the value is not UTF-8")),
        Some(3) => Err(std::io::Error::other(
            "the terminal's echo could not be turned off",
        )),
        Some(4) => Ok(String::new()),
        _ => Err(std::io::Error::other(format!(
            "the read was interrupted ({})",
            output.status
        ))),
    }
}

/// Whether `path` is a secret-typed key of `registry`, or a key under one, such as an entry of
/// `obs.otlp.headers`.
fn holds_secret(registry: &Registry, path: &str) -> bool {
    registry.keys().any(|key| {
        key.holds_secret
            && (path == key.path
                || path
                    .strip_prefix(key.path)
                    .is_some_and(|rest| rest.starts_with('.')))
    })
}

/// Whether `value` is a secret's file or variable form, which holds no secret itself.
fn names_a_source(value: &str) -> bool {
    write::parse_value(value)
        .as_inline_table()
        .is_some_and(|table| {
            table.len() == 1 && (table.contains_key("file") || table.contains_key("env"))
        })
}

/// `config edit`: the target file in `$VISUAL` or `$EDITOR`, validated when the editor exits.
///
/// The editor works on a copy beside the file (mode `0600` on Unix). A valid result replaces the
/// file. An invalid one is kept only with `--force`. Otherwise, on a terminal, the editor can
/// open again; without one, or when the answer is no, the file stays as it was and the edit is
/// kept beside it as `<file>.rejected`, and the command exits 5. No question is asked without a
/// terminal (RFC-0051 §2.2).
pub fn edit(args: &ConfigEditArgs, output: Output) -> Result<Exit, CliError> {
    refuse_json(output, "edit")?;
    let target = Target::of(&args.target)?;
    let old = target.read()?;
    let terminal = Terminal::detect();
    let (program, editor_args) = editor(terminal)?;
    let working = target.working_copy(&old)?;
    let result = edit_until_done(
        &target,
        &old,
        &working,
        &program,
        &editor_args,
        args,
        terminal,
    );
    if working.exists() {
        let _ = std::fs::remove_file(&working);
    }
    result
}

fn edit_until_done(
    target: &Target,
    old: &str,
    working: &Path,
    program: &OsString,
    editor_args: &[OsString],
    args: &ConfigEditArgs,
    terminal: Terminal,
) -> Result<Exit, CliError> {
    loop {
        let status = std::process::Command::new(program)
            .args(editor_args)
            .arg(working)
            .status()
            .map_err(|e| {
                CliError::cli(
                    Exit::Failure,
                    "editor_failed",
                    format!(
                        "cannot start the editor `{}`: {e}",
                        printable(&program.to_string_lossy())
                    ),
                )
                .hint("set EDITOR or VISUAL to your editor")
            })?;
        if !status.success() {
            return Err(CliError::cli(
                Exit::Failure,
                "editor_failed",
                format!(
                    "the editor exited with {status}, so {} was not changed",
                    target.shown()
                ),
            ));
        }
        let text = std::fs::read_to_string(working).map_err(|e| {
            CliError::cli(
                Exit::Failure,
                "editor_failed",
                format!("cannot read the edited copy: {e}"),
            )
        })?;
        if text == old {
            println!("no change to {}", target.shown());
            return Ok(Exit::Success);
        }
        let checked = target.check(&text);
        eprint!("{}", printable_lines(&checked.render(&REGISTRY)));
        if checked.outcome.is_ok() || args.force {
            target.write(&text)?;
            if checked.outcome.is_ok() {
                println!("saved {}", target.shown());
            } else {
                println!(
                    "saved {}, which is invalid ({}), because of --force",
                    target.shown(),
                    errors(checked.errors().len())
                );
            }
            reload_note();
            return Ok(Exit::Success);
        }
        let again = terminal.can_ask()
            && prompt::confirm(
                terminal,
                "The edited file is invalid. Edit it again?",
                "--force",
            )?;
        if again {
            continue;
        }
        let rejected = target.rejected_path();
        let kept = std::fs::rename(working, &rejected).is_ok();
        let mut error = CliError::cli(
            Exit::Invalid,
            "invalid_config",
            format!(
                "the edited file has {}, so {} was not changed",
                errors(checked.errors().len()),
                target.shown()
            ),
        );
        if kept {
            error = error.detail(format!(
                "your edit is kept in {}",
                printable(&rejected.display().to_string())
            ));
        }
        return Err(error.hint("syntropy config edit --force"));
    }
}

/// The editor: `$VISUAL`, then `$EDITOR`, as a program and its arguments. A value that names an
/// existing file is the program, spaces and all; any other is split on whitespace. Without
/// either, `vi` (`notepad` on Windows), which needs a terminal: without one the command exits 2.
fn editor(terminal: Terminal) -> Result<(OsString, Vec<OsString>), CliError> {
    for var in ["VISUAL", "EDITOR"] {
        let Some(value) = std::env::var_os(var).filter(|v| !v.is_empty()) else {
            continue;
        };
        if Path::new(&value).is_file() {
            return Ok((value, Vec::new()));
        }
        let text = value.to_string_lossy().into_owned();
        let mut words = text.split_whitespace().map(OsString::from);
        if let Some(program) = words.next() {
            return Ok((program, words.collect()));
        }
    }
    if !(terminal.stdin && terminal.stdout) || terminal.dumb {
        return Err(CliError::cli(
            Exit::Usage,
            "needs_terminal",
            "`config edit` opens an editor, and there is no terminal for the default one",
        )
        .hint("set EDITOR to an editor that needs no terminal, or use `syntropy config set`"));
    }
    let default = if cfg!(windows) { "notepad" } else { "vi" };
    Ok((OsString::from(default), Vec::new()))
}

/// The file a write changes, and the layer it is read in.
struct Target {
    os: Os,
    path: PathBuf,
    /// The layer's files, with `path` among them whether it exists or not.
    scan: Scan,
}

impl Target {
    /// The file `args` names: the explicit file, the system file, or the user file.
    fn of(args: &ConfigTarget) -> Result<Target, CliError> {
        let (os, _, locations) = locations(args.config.as_deref());
        if !args.system
            && let Some(explicit) = &locations.explicit
        {
            let path = explicit.path.clone();
            let scan = Scan {
                system_file: None,
                dropins: Vec::new(),
                user_file: None,
                explicit: Some(path.clone()),
            };
            return Ok(Target { os, path, scan });
        }
        // The layer without the explicit file, as a daemon reads it when none is named.
        let layered = Locations {
            explicit: None,
            ..locations.clone()
        };
        let mut scan = layered.scan().map_err(invalid)?;
        let path = if args.system {
            if locations.explicit.is_some() {
                eprintln!(
                    "warning: SYNTROPY_CONFIG names the file a daemon reads, in place of the \
                     system file; this change applies where SYNTROPY_CONFIG is not set"
                );
            }
            scan.system_file = Some(locations.system_file.clone());
            locations.system_file.clone()
        } else {
            let path = locations.user_file.clone().ok_or_else(|| {
                CliError::cli(
                    Exit::Invalid,
                    "no_user_file",
                    "there is no user config file here: the environment gives no home directory",
                )
                .hint("name the file with --config, or write the system file with --system")
            })?;
            scan.user_file = Some(path.clone());
            path
        };
        Ok(Target { os, path, scan })
    }

    /// The path, safe to print.
    fn shown(&self) -> String {
        printable(&self.path.display().to_string())
    }

    /// The file's text, or nothing when it does not exist yet. A symbolic link is refused.
    fn read(&self) -> Result<String, CliError> {
        let exists = write::replaceable(&self.path).map_err(|e| self.write_failed(&e))?;
        if !exists {
            return Ok(String::new());
        }
        std::fs::read_to_string(&self.path).map_err(|e| {
            CliError::cli(
                Exit::Failure,
                "read_failed",
                format!("cannot read {}: {e}", self.shown()),
            )
        })
    }

    /// The layer validated with `text` as this file.
    fn check(&self, text: &str) -> Checked {
        // No environment overrides or flags: what is checked is the file layer a daemon reads.
        let env = Env::from_pairs(
            self.os,
            std::env::vars_os()
                .filter(|(name, _)| !name.to_string_lossy().starts_with("SYNTROPY_")),
        );
        write::check_edit(
            &REGISTRY,
            self.os,
            &self.scan,
            &env,
            &self.path,
            text,
            API_RULES,
            &HostEnvironment,
        )
    }

    /// Validates `text` and writes it, or refuses: the diagnostics go to stderr either way.
    fn save(&self, text: &str, force: bool) -> Result<(), CliError> {
        let checked = self.check(text);
        eprint!("{}", printable_lines(&checked.render(&REGISTRY)));
        if checked.outcome.is_err() && !force {
            return Err(CliError::cli(
                Exit::Invalid,
                "invalid_config",
                format!(
                    "the change would leave the configuration with {}, so {} was not changed",
                    errors(checked.errors().len()),
                    self.shown()
                ),
            ));
        }
        self.write(text)
    }

    fn write(&self, text: &str) -> Result<(), CliError> {
        write::write_atomic(&self.path, text).map_err(|e| self.write_failed(&e))
    }

    fn write_failed(&self, error: &std::io::Error) -> CliError {
        CliError::cli(
            Exit::Failure,
            "write_failed",
            format!(
                "cannot write {}: {}",
                self.shown(),
                printable(&error.to_string())
            ),
        )
    }

    /// A private copy of `text` beside the file, for the editor.
    fn working_copy(&self, text: &str) -> Result<PathBuf, CliError> {
        let dir = self.path.parent().unwrap_or(Path::new("."));
        let name = self.path.file_name().map_or_else(
            || "config.toml".into(),
            |n| n.to_string_lossy().into_owned(),
        );
        let working = dir.join(format!(".{name}.syntropy-edit-{}.toml", std::process::id()));
        let failed = |e: std::io::Error| {
            CliError::cli(
                Exit::Failure,
                "write_failed",
                format!(
                    "cannot make a copy to edit in {}: {e}",
                    printable(&dir.display().to_string())
                ),
            )
        };
        std::fs::create_dir_all(dir).map_err(failed)?;
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        let mut file = options.open(&working).map_err(failed)?;
        std::io::Write::write_all(&mut file, text.as_bytes()).map_err(failed)?;
        Ok(working)
    }

    /// Where a refused edit is kept: `<file>.rejected` beside the file.
    fn rejected_path(&self) -> PathBuf {
        let mut name = self.path.file_name().unwrap_or_default().to_os_string();
        name.push(".rejected");
        self.path.with_file_name(name)
    }
}

fn edit_failed(target: &Target, error: &write::EditError) -> CliError {
    CliError::cli(
        Exit::Invalid,
        "edit_failed",
        format!("{}: {}", target.shown(), printable(&error.to_string())),
    )
}

/// `--cluster` submits `SetClusterConfig`, which needs the cluster configuration endpoint.
fn cluster_not_built(command: &str) -> CliError {
    CliError::not_built(&format!("config {command} --cluster"))
        .detail("cluster values are submitted to the coordinator as `SetClusterConfig`")
}

/// What a write cannot do yet: reload the daemon (PROD-E1-T11).
fn reload_note() {
    eprintln!(
        "note: a running daemon applies this at its next start; `syntropy config reload` is not \
         available in this build yet"
    );
}

fn printable(text: &str) -> String {
    syntropy_api::printable::printable(text).into_owned()
}

/// The OS, its environment and where a user's daemon looks, with `config` as the explicit file.
fn locations(config: Option<&Path>) -> (Os, Env, Locations) {
    let os = Os::current();
    let env = Env::from_process(os);
    let locations = locate(os, Mode::User, &env, config);
    (os, env, locations)
}

/// Reads and checks the configuration `source` names: RFC-0054 §8's phases 1–5. Computed
/// defaults stay uncomputed, since computing them needs the probe, and `show` says so per key.
fn load(source: &ConfigSource) -> Result<syntropy_config::Checked, CliError> {
    let (os, env, locations) = locations(source.config.as_deref());
    let mut flags = Flags::new();
    for arg in &source.set {
        flags.set(arg).map_err(invalid)?;
    }
    let scan = locations.scan().map_err(invalid)?;
    Ok(load_and_validate(
        &REGISTRY,
        os,
        &scan,
        &env,
        &flags,
        |_| None,
        API_RULES,
        &HostEnvironment,
    ))
}

/// A problem before the configuration resolves, such as a `--set` without `=` or a missing
/// `--config` file: exit 5, in the configuration's own words.
fn invalid(problem: Problem) -> CliError {
    let code = problem.code.map_or("invalid", |code| code.name());
    CliError {
        exit: Exit::Invalid,
        domain: "config",
        code: code.to_owned(),
        message: problem.to_string(),
        details: Vec::new(),
        hints: Vec::new(),
    }
}

/// `--json` has no API type to print for these commands yet.
fn refuse_json(output: Output, command: &str) -> Result<(), CliError> {
    if output.json {
        return Err(CliError::cli(
            Exit::Invalid,
            "not_built",
            format!(
                "`syntropy config {command} --json` is not available in this build yet: the API's \
                 configuration type comes with `/v1/syntropy/config` (RFC-0054 §10.2)"
            ),
        ));
    }
    Ok(())
}

/// `text` safe to print on a terminal, a line at a time: each line through
/// `syntropy_api::printable`, with its newline kept. The renderer quotes the configuration's own
/// sources, a file's or the environment's values and the variables' names, and none of them is
/// text this binary wrote (RFC-0051 §2.2): a TOML `\u001b` escape, an environment value or a
/// variable's name can hold ESC or BEL.
fn printable_lines(text: &str) -> String {
    let mut safe = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        let (body, end) = match line.strip_suffix('\n') {
            Some(body) => (body, "\n"),
            None => (line, ""),
        };
        safe.push_str(&syntropy_api::printable::printable(body));
        safe.push_str(end);
    }
    safe
}

/// "1 error" or "n errors".
fn errors(n: usize) -> String {
    format!("{n} error{}", if n == 1 { "" } else { "s" })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A section with a secret, for [`holds_secret`].
    #[derive(Debug, serde::Deserialize, syntropy_config::ConfigSection)]
    #[config(section = "vault", scope = Node, owner = "RFC-0054")]
    #[allow(dead_code)]
    struct Vault {
        /// A secret.
        #[config(reload = Hot, default = {})]
        headers: std::collections::BTreeMap<String, syntropy_config::SecretSource>,
        /// Not a secret.
        #[config(reload = Hot, default = 1)]
        retries: u32,
    }

    /// `config set` warns for a secret on the command line: a secret key or a key under one, and
    /// not for its file or variable form (review of #1927, N3).
    #[test]
    fn a_secret_on_the_command_line_is_recognised() {
        static VAULT: Registry = Registry::new(&[&Vault::SCHEMA]);
        assert!(holds_secret(&VAULT, "vault.headers"));
        assert!(holds_secret(&VAULT, "vault.headers.authorization"));
        assert!(!holds_secret(&VAULT, "vault.headersx"));
        assert!(!holds_secret(&VAULT, "vault.retries"));
        assert!(names_a_source(r#"{ file = "/run/secrets/x" }"#));
        assert!(names_a_source(r#"{ env = "X" }"#));
        assert!(!names_a_source("Bearer x"));
        assert!(!names_a_source(r#"{ file = "/x", env = "Y" }"#));
    }

    #[test]
    fn the_registry_holds_every_section_and_every_default_fits() {
        assert_eq!(REGISTRY.check(), vec![]);
        assert_eq!(REGISTRY.check_defaults(), vec![]);
        for prefix in [
            "api.bind",
            "debug.parity_api",
            "net.listen_port",
            "discovery.port",
            "swim.period",
            "quic.congestion",
        ] {
            assert!(REGISTRY.key(prefix).is_some(), "{prefix}");
        }
        assert_eq!(COUNT, NET.len() + CLUSTER.len() + 3);
    }

    /// `check` prints its summary as the report after the diagnostics, which holds while
    /// `command::check`'s text starts with `Checked::render`'s.
    #[test]
    fn check_s_report_starts_with_its_diagnostics() {
        let dir = std::env::temp_dir().join(format!("syntropy-cli-prefix-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (name, text) in [("ok.toml", ""), ("bad.toml", "[api]\nbnd = 1\n")] {
            let file = dir.join(name);
            std::fs::write(&file, text).unwrap();
            let checked = load(&ConfigSource {
                config: Some(file),
                set: vec!["discovery.port=x".to_owned()],
            })
            .unwrap();
            let diagnostics = checked.render(&REGISTRY);
            let report = command::check(&checked, &REGISTRY);
            assert!(report.text.starts_with(&diagnostics), "{name}");
            assert!(report.text.len() > diagnostics.len(), "{name}: no summary");
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The registry is the linked crates' lists, then the three single sections, in that order,
    /// by prefix and count.
    #[test]
    fn the_registry_is_net_then_cluster_then_the_singles() {
        let got: Vec<&str> = REGISTRY.sections().iter().map(|s| s.prefix).collect();
        let want: Vec<&str> = NET
            .iter()
            .chain(CLUSTER)
            .chain(SINGLES.iter())
            .map(|s| s.prefix)
            .collect();
        assert_eq!(got, want);
        assert_eq!(
            got.len(),
            32,
            "the sections the workspace declares on 2026-10-27"
        );
    }

    /// The drift guard: every configuration section a workspace crate `cli` depends on declares,
    /// outside its tests, is in the registry, and the registry holds nothing else. A crate that
    /// adds a section, or a new dependency of `cli` that has one, fails this test until the
    /// registry lists it. The crates come from `cli`'s own manifest, so the guard follows it; a
    /// section in a crate `cli` reaches only through another needs a direct edge to be listed.
    #[test]
    fn every_section_the_linked_crates_declare_is_in_the_registry() {
        let cli = Path::new(env!("CARGO_MANIFEST_DIR"));
        let manifest = std::fs::read_to_string(cli.join("Cargo.toml")).unwrap();
        let crates: Vec<&str> = manifest
            .lines()
            .filter_map(|line| line.split_once("path = \"../")?.1.split_once('"'))
            .map(|(name, _)| name)
            .collect();
        assert!(
            crates.contains(&"cluster") && crates.contains(&"api"),
            "{crates:?}"
        );
        let mut declared = std::collections::BTreeSet::new();
        for name in &crates {
            declared_sections(&cli.join("..").join(name).join("src"), &mut declared);
        }
        let registered: std::collections::BTreeSet<&str> =
            REGISTRY.sections().iter().map(|s| s.type_name).collect();
        let declared: std::collections::BTreeSet<&str> =
            declared.iter().map(String::as_str).collect();
        assert_eq!(
            declared, registered,
            "the linked crates' sections and the registry differ"
        );
    }

    /// The structs under a `#[config(section = …)]` attribute in the `.rs` files below `dir`,
    /// leaving out test modules (from `#[cfg(test)]` on), test files and doc examples.
    fn declared_sections(dir: &Path, out: &mut std::collections::BTreeSet<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.map(Result::unwrap) {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if path.is_dir() {
                if name != "tests" {
                    declared_sections(&path, out);
                }
                continue;
            }
            if !name.ends_with(".rs") || name == "tests.rs" {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            let code = text.split("#[cfg(test)]").next().unwrap_or("");
            let mut pending = false;
            for line in code.lines().map(str::trim) {
                if line.starts_with("#[config(section") {
                    pending = true;
                } else if pending && let Some(rest) = line.strip_prefix("pub struct ") {
                    let ident: String = rest
                        .chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .collect();
                    out.insert(ident);
                    pending = false;
                }
            }
        }
    }
}
