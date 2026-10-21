// SPDX-License-Identifier: MIT OR Apache-2.0

//! One writer process per store root (RFC-0041 §4.1), tested with real processes.
//!
//! The tests re-run this test binary as child processes. One child holds a root open for
//! writing; another tries to open the same root and reports what happened. The children's entry
//! points are ignored tests, so a plain `cargo test` only runs the parents, which start the
//! children with `--ignored --exact`.

mod common;

use std::env;
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdout, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use common::TempDir;
use syntropy_store::{StoreError, WritableRoot};

/// The root a child opens.
const ROOT_ENV: &str = "SYNTROPY_STORE_TEST_ROOT";
/// Marks a child's report in its output, which also carries the test harness's own lines.
const MARK: &str = "store-child:";

/// A command that runs one child entry point of this binary on `root`.
fn child(entry: &str, root: &Path) -> Command {
    let mut cmd = Command::new(env::current_exe().unwrap());
    cmd.args([
        "--ignored",
        "--exact",
        entry,
        "--nocapture",
        "--test-threads=1",
    ])
    .env(ROOT_ENV, root)
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::inherit());
    cmd
}

/// Reads a child's output up to its next report and returns the report.
fn next_report(out: &mut BufReader<ChildStdout>) -> String {
    let mut line = String::new();
    loop {
        line.clear();
        let n = out.read_line(&mut line).unwrap();
        assert!(n > 0, "the child exited without reporting");
        if let Some(at) = line.find(MARK) {
            return line[at + MARK.len()..].trim().to_owned();
        }
    }
}

/// Starts a child that opens `root` for writing and holds it, and waits until it holds it.
fn spawn_holder(root: &Path) -> (Child, BufReader<ChildStdout>) {
    let mut holder = child("child_hold_writer", root).spawn().unwrap();
    let mut out = BufReader::new(holder.stdout.take().unwrap());
    assert_eq!(next_report(&mut out), format!("holding {}", holder.id()));
    (holder, out)
}

/// Runs a child that tries to open `root` for writing, and returns its report.
fn run_contender(root: &Path) -> String {
    let mut contender = child("child_try_writer", root).spawn().unwrap();
    drop(contender.stdin.take());
    let mut out = BufReader::new(contender.stdout.take().unwrap());
    let report = next_report(&mut out);
    // Drain the harness's last lines so the child never writes to a closed pipe.
    io::copy(&mut out, &mut io::sink()).unwrap();
    assert!(contender.wait().unwrap().success());
    report
}

/// Runs contenders until one opens `root` or `limit` has passed, and returns the last report.
/// Used only after a holder is killed: Windows documents that it frees a dead process's file
/// locks when the system gets to it, not at once. Every earlier attempt must be a refusal.
fn run_contender_until_opened(root: &Path, limit: Duration) -> String {
    let start = Instant::now();
    loop {
        let report = run_contender(root);
        if report == "opened" || start.elapsed() > limit {
            return report;
        }
        assert!(report.starts_with("refused: "), "{report}");
        thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn a_second_writer_process_is_refused_while_the_first_holds_the_root() {
    let tmp = TempDir::new("lock");
    let root = tmp.join("models");
    let (mut holder, mut holder_out) = spawn_holder(&root);

    let resolved = fs::canonicalize(&root).unwrap();
    assert_eq!(
        run_contender(&root),
        format!(
            "refused: store root {} is already open for writing by another process (pid {}); \
             a store root has one writer process at a time",
            resolved.display(),
            holder.id()
        )
    );

    // Closing the holder's stdin makes it drop the root, which unlocks it, and exit. The next
    // writer must get in on its first attempt.
    drop(holder.stdin.take());
    io::copy(&mut holder_out, &mut io::sink()).unwrap();
    assert!(holder.wait().unwrap().success());
    assert_eq!(run_contender(&root), "opened");
}

#[test]
fn a_killed_writer_leaves_the_root_unlocked() {
    let tmp = TempDir::new("kill");
    let root = tmp.join("models");
    let (mut holder, _holder_out) = spawn_holder(&root);
    assert!(run_contender(&root).starts_with("refused: "));

    // SIGKILL on Unix, TerminateProcess on Windows: the holder runs no destructor, so only the
    // OS can release its lock.
    holder.kill().unwrap();
    holder.wait().unwrap();
    assert_eq!(
        run_contender_until_opened(&root, Duration::from_secs(10)),
        "opened"
    );
}

fn report(msg: &str) {
    let mut out = io::stdout().lock();
    writeln!(out, "{MARK} {msg}").unwrap();
    out.flush().unwrap();
}

/// Child: opens the root for writing, reports, then holds it until stdin closes or the parent
/// kills this process.
#[test]
#[ignore = "child process of the writer-lock tests"]
fn child_hold_writer() {
    let Some(root) = env::var_os(ROOT_ENV) else {
        return;
    };
    let held = WritableRoot::open(&root).expect("the holder opens the root");
    report(&format!("holding {}", std::process::id()));
    let mut line = String::new();
    io::stdin().read_line(&mut line).unwrap();
    drop(held);
}

/// Child: tries to open the root for writing and reports the outcome.
#[test]
#[ignore = "child process of the writer-lock tests"]
fn child_try_writer() {
    let Some(root) = env::var_os(ROOT_ENV) else {
        return;
    };
    match WritableRoot::open(&root) {
        Ok(_) => report("opened"),
        Err(e @ StoreError::WriterLocked { .. }) => report(&format!("refused: {e}")),
        Err(e) => report(&format!("error: {e}")),
    }
}
