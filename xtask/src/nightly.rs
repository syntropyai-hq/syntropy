// SPDX-License-Identifier: MIT OR Apache-2.0

//! The nightly toolchain that only tests use (RFC-0060 §3.4; QUAL-E3-T10), and what the suites
//! that run on it share: [`crate::miri`] and [`crate::sanitizers`].
//!
//! ADR-0001 item 1 pins one stable release for every build, in `rust-toolchain.toml`, and forbids
//! nightly compilers for release builds only. Miri and the sanitizers need unstable compiler
//! flags, so their suites run a dated nightly pinned in `ci/nightly-toolchain.toml`:
//! - each suite installs that toolchain and its components with rustup when any is missing, the
//!   way the licence steps need cargo-deny, so a local run and a CI run use the same compiler;
//! - each runs cargo through `rustup run <channel>`, which overrides `rust-toolchain.toml` for
//!   that command alone, and drops `RUSTC` and `RUSTDOC`, which could name the stable tools;
//! - each builds into a target directory of its own under the usual one, so a nightly artefact
//!   never lands where a stable build looks, and the stable build is not rebuilt;
//! - each streams cargo's output as it comes and stops a run still going at [`STEP_LIMIT`], so a
//!   hung test fails the step with its output and exit status, not the job at its time limit.
//!
//! `cargo xtask build --profile release` checks the other side: the compiler its build recorded
//! is the stable release (`crate::build`).

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// The pin, relative to the workspace root.
pub const PIN_FILE: &str = "ci/nightly-toolchain.toml";

/// The components every suite needs: Miri, and the standard library's source, which Miri and
/// `-Zbuild-std` build from.
const NEEDED: &[&str] = &["miri", "rust-src"];

/// The pinned nightly.
#[derive(Debug, PartialEq, Eq)]
pub struct Pin {
    /// A dated nightly, such as `nightly-2026-10-10`.
    pub channel: String,
    /// The components to install with it.
    pub components: Vec<String>,
}

/// Whether `channel` is a dated nightly, `nightly-YYYY-MM-DD`. An undated `nightly` would change
/// under the suites from one day to the next.
pub fn is_dated_nightly(channel: &str) -> bool {
    let Some(date) = channel.strip_prefix("nightly-") else {
        return false;
    };
    let parts: Vec<&str> = date.split('-').collect();
    parts.len() == 3
        && [4, 2, 2]
            .iter()
            .zip(&parts)
            .all(|(len, part)| part.len() == *len && part.chars().all(|c| c.is_ascii_digit()))
}

/// Reads the pin from the text of `ci/nightly-toolchain.toml`.
pub fn parse(text: &str) -> Result<Pin, String> {
    let value: toml::Table = text
        .parse()
        .map_err(|e| format!("{PIN_FILE} is not TOML: {e}"))?;
    let table = value
        .get("toolchain")
        .and_then(toml::Value::as_table)
        .ok_or_else(|| format!("{PIN_FILE} has no [toolchain] table"))?;
    let channel = table
        .get("channel")
        .and_then(toml::Value::as_str)
        .ok_or_else(|| format!("{PIN_FILE} has no `channel` in [toolchain]"))?;
    if !is_dated_nightly(channel) {
        return Err(format!(
            "{PIN_FILE} pins `{channel}`; the suites need a dated nightly, `nightly-YYYY-MM-DD`"
        ));
    }
    let components: Vec<String> = table
        .get("components")
        .and_then(toml::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(toml::Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    let missing: Vec<&str> = NEEDED
        .iter()
        .copied()
        .filter(|needed| !components.iter().any(|c| c == needed))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "{PIN_FILE} must list the components {}",
            missing.join(" and ")
        ));
    }
    Ok(Pin {
        channel: channel.to_owned(),
        components,
    })
}

/// The pin of the workspace at `root`.
pub fn read(root: &Path) -> Result<Pin, String> {
    let text = std::fs::read_to_string(root.join(PIN_FILE))
        .map_err(|e| format!("cannot read {PIN_FILE}: {e}"))?;
    parse(&text)
}

/// Whether `rustup component list --installed` printed every component of `wanted`. A component
/// is listed by its name, such as `rust-src`, or with the host after it, such as
/// `miri-x86_64-unknown-linux-gnu`.
pub fn has_components(listed: &str, wanted: &[String]) -> bool {
    wanted.iter().all(|component| {
        listed.lines().map(str::trim).any(|line| {
            line == component
                || line
                    .strip_prefix(component.as_str())
                    .is_some_and(|rest| rest.starts_with('-'))
        })
    })
}

/// Installs the pinned toolchain and its components, unless rustup lists them all already.
pub fn ensure(pin: &Pin) -> Result<(), String> {
    let listed = Command::new("rustup")
        .args(["component", "list", "--installed", "--toolchain"])
        .arg(&pin.channel)
        .output();
    match listed {
        Ok(output)
            if output.status.success()
                && has_components(&String::from_utf8_lossy(&output.stdout), &pin.components) =>
        {
            eprintln!(
                "nightly: {} with {} is installed",
                pin.channel,
                pin.components.join(", ")
            );
            return Ok(());
        }
        Err(e) => {
            return Err(format!(
                "cannot start rustup ({e}); the suites run the nightly pinned in {PIN_FILE} \
                 through rustup"
            ));
        }
        Ok(_) => {}
    }
    let mut install = Command::new("rustup");
    install.args(["toolchain", "install", &pin.channel, "--profile", "minimal"]);
    for component in &pin.components {
        install.args(["--component", component]);
    }
    install.arg("--no-self-update");
    eprintln!(
        "nightly: rustup toolchain install {} --profile minimal --component {} --no-self-update",
        pin.channel,
        pin.components.join(" --component ")
    );
    let status = install
        .status()
        .map_err(|e| format!("cannot start rustup: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "rustup could not install {} ({status})",
            pin.channel
        ))
    }
}

/// `rustup run <channel> <tool>`, without the variables that could point cargo at other tools.
pub fn command(pin: &Pin, tool: &str) -> Command {
    let mut command = Command::new("rustup");
    command
        .args(["run", &pin.channel, tool])
        .env_remove("RUSTC")
        .env_remove("RUSTDOC")
        .env_remove("RUSTUP_TOOLCHAIN");
    command
}

/// The host triple of the pinned compiler, from `rustc -vV`.
pub fn host(pin: &Pin) -> Result<String, String> {
    let output = command(pin, "rustc")
        .arg("-vV")
        .output()
        .map_err(|e| format!("cannot start rustup: {e}"))?;
    if !output.status.success() {
        return Err(format!("`rustc -vV` on {} failed", pin.channel));
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .map(|host| host.trim().to_owned())
        .ok_or_else(|| format!("`rustc -vV` on {} printed no host", pin.channel))
}

/// The compiler flags a suite's cargo run sets: the caller's, from `CARGO_ENCODED_RUSTFLAGS` when
/// set, which cargo then reads instead, else `RUSTFLAGS`, with `extra` added. Returns the
/// variable and its value.
pub fn flags(var: impl Fn(&str) -> Option<String>, extra: &[&str]) -> (&'static str, String) {
    match var("CARGO_ENCODED_RUSTFLAGS") {
        Some(encoded) if !encoded.is_empty() => {
            let mut value = encoded;
            for flag in extra {
                value.push('\u{1f}');
                value.push_str(flag);
            }
            ("CARGO_ENCODED_RUSTFLAGS", value)
        }
        _ => {
            let mut parts: Vec<String> = var("RUSTFLAGS")
                .filter(|flags| !flags.trim().is_empty())
                .into_iter()
                .collect();
            parts.extend(extra.iter().map(|flag| (*flag).to_owned()));
            ("RUSTFLAGS", parts.join(" "))
        }
    }
}

/// `<name>/` under the caller's target directory: `CARGO_TARGET_DIR`, else `target/` in `root`.
pub fn target_dir(root: &Path, var: impl Fn(&str) -> Option<String>, name: &str) -> PathBuf {
    var("CARGO_TARGET_DIR")
        .map_or_else(|| root.join("target"), PathBuf::from)
        .join(name)
}

/// How long one suite step may run before it is stopped: 20 minutes \[target\], chosen to leave
/// `hosted-l3`'s 30-minute job limit time for the checkout, the cache and the verdict. A step
/// that hangs then fails with everything it printed, rather than being killed with the job.
pub const STEP_LIMIT: Duration = Duration::from_secs(20 * 60);

/// When a step that starts now must have finished.
pub fn deadline() -> Instant {
    Instant::now() + STEP_LIMIT
}

/// What a finished child process printed.
pub struct Output {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
}

/// Copies `from` to `to` line by line as it arrives, and returns all of it.
fn tee(from: impl Read, mut to: impl Write) -> String {
    let mut reader = BufReader::new(from);
    let mut all = Vec::new();
    let mut line = Vec::new();
    while matches!(reader.read_until(b'\n', &mut line), Ok(n) if n > 0) {
        // A closed terminal must not stop the capture the verdict reads.
        let _ = to.write_all(&line).and_then(|()| to.flush());
        all.append(&mut line);
    }
    String::from_utf8_lossy(&all).into_owned()
}

/// How long the output pipes may stay open after the run has ended and its group was stopped.
const PIPE_GRACE: Duration = Duration::from_secs(5);

/// On Unix, stops every process left in the group of the child whose ID is `id`: the test
/// binaries cargo started and any symbolizer a sanitizer started. Each holds the output pipes
/// open. Elsewhere it does nothing.
fn stop_group(id: u32) {
    #[cfg(unix)]
    {
        let group = format!("-{id}");
        let _ = Command::new("kill")
            .args(["-KILL", "--", &group])
            .stderr(Stdio::null())
            .status();
    }
    #[cfg(not(unix))]
    let _ = id;
}

/// Runs `command`, echoing what it prints as it prints it, so a log shows how far it got even if
/// it never ends. A run still going at `deadline` is stopped and is an error. Whether it ended or
/// was stopped, any process it started that is still running is stopped too, so a leftover
/// process neither outlives the step nor holds its output open. A run that fails prints its exit
/// status.
pub fn run(mut command: Command, deadline: Instant) -> Result<Output, String> {
    let program = command.get_program().to_string_lossy().into_owned();
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut command, 0);
    let mut child = command
        .spawn()
        .map_err(|e| format!("cannot start {program}: {e}"))?;
    let id = child.id();
    let (sender, received) = mpsc::channel();
    let out = child.stdout.take().expect("stdout is piped");
    let err = child.stderr.take().expect("stderr is piped");
    let to_out = sender.clone();
    std::thread::spawn(move || to_out.send((true, tee(out, std::io::stdout()))));
    std::thread::spawn(move || sender.send((false, tee(err, std::io::stderr()))));
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if Instant::now() >= deadline => {
                stop_group(id);
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(100)),
            Err(e) => return Err(format!("cannot wait for {program}: {e}")),
        }
    };
    stop_group(id);
    // The pipes close once the last process holding them has gone. One that left the group can
    // still hold them, so the wait for them is bounded too.
    let until = deadline.max(Instant::now()) + PIPE_GRACE;
    let (mut stdout, mut stderr) = (String::new(), String::new());
    for _ in 0..2 {
        match received.recv_timeout(until.saturating_duration_since(Instant::now())) {
            Ok((true, text)) => stdout = text,
            Ok((false, text)) => stderr = text,
            Err(_) => {
                return Err(format!(
                    "{program} ended, but a process it started still held its output open \
                     after the step limit; its output up to then is above"
                ));
            }
        }
    }
    let Some(status) = status else {
        return Err(format!(
            "{program} was still running at the step limit of {} min [target] and was \
             stopped; its output up to then is above",
            STEP_LIMIT.as_secs() / 60
        ));
    };
    if !status.success() {
        eprintln!("nightly: {program} exited with {status}");
    }
    Ok(Output {
        success: status.success(),
        stdout,
        stderr,
    })
}

/// The verdict of a planted step: the run must fail, and its output must hold every marker of
/// the checker's report.
pub fn reported(output: &Output, checker: &str, markers: &[&str]) -> Result<(), String> {
    let text = format!("{}\n{}", output.stdout, output.stderr);
    let missing: Vec<&str> = markers
        .iter()
        .copied()
        .filter(|marker| !text.contains(marker))
        .collect();
    match (output.success, missing.as_slice()) {
        (false, []) => Ok(()),
        (true, _) => Err(format!(
            "the planted out-of-bounds read passed: {checker} did not report it"
        )),
        (false, missing) => Err(format!(
            "the planted run failed, but not with {checker}'s report: {} not found",
            missing.join(", ")
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vars<'a>(set: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            set.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_string())
        }
    }

    #[test]
    fn the_workspace_pins_a_dated_nightly_with_miri_and_the_source() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let pin = read(root).unwrap();
        assert!(is_dated_nightly(&pin.channel), "{}", pin.channel);
        for needed in NEEDED {
            assert!(pin.components.iter().any(|c| c == needed), "{needed}");
        }
    }

    #[test]
    fn the_test_nightly_is_not_the_release_toolchain() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let release = std::fs::read_to_string(root.join("rust-toolchain.toml")).unwrap();
        let release = crate::toolchain::channel(&release).unwrap();
        assert!(!release.contains("nightly"), "{release}");
        assert_ne!(read(root).unwrap().channel, release);
    }

    #[test]
    fn only_a_dated_nightly_with_both_components_is_a_pin() {
        let good = "[toolchain]\nchannel = \"nightly-2026-10-10\"\n\
                    components = [\"miri\", \"rust-src\"]\n";
        assert_eq!(
            parse(good).unwrap(),
            Pin {
                channel: "nightly-2026-10-10".to_owned(),
                components: vec!["miri".to_owned(), "rust-src".to_owned()],
            }
        );
        for channel in ["nightly", "1.99.1", "nightly-2026-1-10", "beta-2026-10-10"] {
            let text = good.replace("nightly-2026-10-10", channel);
            assert!(parse(&text).is_err(), "{channel}");
        }
        let no_miri = good.replace("\"miri\", ", "");
        assert!(parse(&no_miri).unwrap_err().contains("miri"));
        assert!(parse("[other]\n").is_err());
    }

    #[test]
    fn components_match_with_or_without_the_host() {
        let listed = "cargo-x86_64-unknown-linux-gnu\nmiri-x86_64-unknown-linux-gnu\nrust-src\n";
        let want = |names: &[&str]| names.iter().map(|n| (*n).to_owned()).collect::<Vec<_>>();
        assert!(has_components(listed, &want(&["miri", "rust-src"])));
        assert!(!has_components(listed, &want(&["miri", "llvm-tools"])));
        // `rust` is not a prefix match for `rust-src`'s line without the dash rule.
        assert!(!has_components("rustc-x\n", &want(&["rust-src"])));
        assert!(!has_components(
            "miri-preview-x\n",
            &want(&["miri-preview-y"])
        ));
    }

    #[test]
    fn flags_keep_the_callers_and_add_the_suites() {
        assert_eq!(
            flags(vars(&[]), &["-Zsanitizer=address"]),
            ("RUSTFLAGS", "-Zsanitizer=address".to_owned())
        );
        assert_eq!(
            flags(vars(&[("RUSTFLAGS", "-D warnings")]), &["--cfg", "x"]),
            ("RUSTFLAGS", "-D warnings --cfg x".to_owned())
        );
        assert_eq!(
            flags(
                vars(&[
                    ("CARGO_ENCODED_RUSTFLAGS", "-D\u{1f}warnings"),
                    ("RUSTFLAGS", "-O")
                ]),
                &["--cfg", "x"]
            ),
            (
                "CARGO_ENCODED_RUSTFLAGS",
                "-D\u{1f}warnings\u{1f}--cfg\u{1f}x".to_owned()
            )
        );
        assert_eq!(flags(vars(&[]), &[]), ("RUSTFLAGS", String::new()));
    }

    #[test]
    fn each_suite_builds_under_the_callers_target_directory() {
        assert_eq!(
            target_dir(Path::new("/w"), vars(&[]), "miri"),
            PathBuf::from("/w/target/miri")
        );
        assert_eq!(
            target_dir(Path::new("/w"), vars(&[("CARGO_TARGET_DIR", "/c")]), "asan"),
            PathBuf::from("/c/asan")
        );
    }

    #[cfg(unix)]
    fn sh(script: &str) -> Command {
        let mut command = Command::new("sh");
        command.args(["-c", script]);
        command
    }

    #[cfg(unix)]
    #[test]
    fn a_run_captures_both_streams_and_its_verdict() {
        let far = Instant::now() + Duration::from_secs(60);
        let output = run(sh("echo out; echo err >&2"), far).unwrap();
        assert!(output.success);
        assert_eq!(output.stdout, "out\n");
        assert_eq!(output.stderr, "err\n");
        assert!(!run(sh("exit 3"), far).unwrap().success);
    }

    /// The hang the step limit is for: a child that never ends, and a grandchild that holds the
    /// pipes. Both are stopped, and what was printed before survives in the error's log.
    #[cfg(unix)]
    #[test]
    fn a_run_past_its_deadline_is_stopped_with_what_it_started() {
        let started = Instant::now();
        let deadline = started + Duration::from_secs(1);
        let error = run(sh("echo started; sleep 60 & wait"), deadline)
            .err()
            .expect("a run past its deadline is an error");
        assert!(error.contains("step limit"), "{error}");
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "the run took {:?}: the sleep it started was not stopped",
            started.elapsed()
        );
    }

    /// A run that ends while a process it started keeps running: the leftover is stopped, so the
    /// run returns at once, not when the leftover ends or at its deadline.
    #[cfg(unix)]
    #[test]
    fn a_process_left_running_after_the_run_ends_is_stopped() {
        for limit in [2, 60] {
            let started = Instant::now();
            let output = run(
                sh("echo hi; sleep 30 &"),
                started + Duration::from_secs(limit),
            )
            .unwrap();
            assert!(output.success);
            assert_eq!(output.stdout, "hi\n");
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "the run took {:?} with a {limit} s deadline",
                started.elapsed()
            );
        }
    }

    #[test]
    fn a_planted_run_passes_only_when_it_fails_with_the_report() {
        let output = |success: bool, stderr: &str| Output {
            success,
            stdout: String::new(),
            stderr: stderr.to_owned(),
        };
        let markers = ["Undefined Behavior", "out-of-bounds"];
        assert!(
            reported(
                &output(false, "error: Undefined Behavior: out-of-bounds read"),
                "Miri",
                &markers
            )
            .is_ok()
        );
        assert!(
            reported(&output(true, ""), "Miri", &markers)
                .unwrap_err()
                .contains("did not report")
        );
        assert!(
            reported(&output(false, "error: could not compile"), "Miri", &markers)
                .unwrap_err()
                .contains("not with Miri's report")
        );
    }
}
