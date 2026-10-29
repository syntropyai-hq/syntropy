// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of crash records (PROD-E9-T04; RFC-0053 §6).

use std::sync::atomic::AtomicUsize;

use syntropy_core::MonoTime;
use syntropy_obs::log::Level;

use super::*;
use crate::logging::Field;

/// A clock at a fixed wall time whose monotonic time steps 1 ms per reading.
pub(crate) struct FixedClock {
    pub(crate) unix_ms: u64,
    mono: AtomicU64,
}

impl FixedClock {
    pub(crate) fn at(unix_ms: u64) -> FixedClock {
        FixedClock {
            unix_ms,
            mono: AtomicU64::new(0),
        }
    }
}

impl Clock for FixedClock {
    fn now(&self) -> MonoTime {
        MonoTime(self.mono.fetch_add(1_000_000, Ordering::Relaxed))
    }

    fn unix_ms(&self) -> u64 {
        self.unix_ms
    }
}

/// A fresh state directory, removed on drop.
pub(crate) struct StateDir(pub(crate) PathBuf);

impl StateDir {
    pub(crate) fn new(tag: &str) -> StateDir {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "syntropy-daemon-crash-{tag}-{}-{n}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        StateDir(dir)
    }
}

impl Drop for StateDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub(crate) fn record(at_unix_ms: u64, cause: CrashCause) -> CrashRecord {
    CrashRecord {
        version: CRASH_RECORD_VERSION,
        at_unix_ms,
        build_id: "ab".repeat(32),
        process: CrashProcess::Runner {
            pid: 4242,
            runner: "instance 3 stage 0".into(),
        },
        cause,
        last_logs: vec!["a line".into()],
    }
}

#[test]
fn records_are_kept_owner_only_newest_first_and_pruned_to_twenty() {
    let state = StateDir::new("prune");
    let store = CrashStore::new(&state.0);
    assert_eq!(store.list().unwrap(), CrashListing::default());
    let mut paths = Vec::new();
    for i in 0..25u64 {
        let cause = CrashCause::Signal {
            signal: 9,
            name: Some("SIGKILL".into()),
        };
        paths.push(store.write(&record(1_800_000_000_000 + i, cause)).unwrap());
    }
    assert!(syntropy_io::os::is_owner_only(store.dir()).unwrap());
    assert!(syntropy_io::os::is_owner_only(paths.last().unwrap()).unwrap());
    let CrashListing {
        records,
        unreadable,
    } = store.list().unwrap();
    assert!(unreadable.is_empty());
    assert_eq!(records.len(), CRASHES_KEPT);
    let times: Vec<u64> = records.iter().map(|(_, r)| r.at_unix_ms).collect();
    let expected: Vec<u64> = (5..25).rev().map(|i| 1_800_000_000_000 + i).collect();
    assert_eq!(times, expected, "newest first, the oldest five pruned");
    assert!(!paths[0].exists());

    // A file that is not a record is named, not used.
    fs::write(
        store.dir().join("99999999999999999999-1-0.json"),
        b"{not json",
    )
    .unwrap();
    let CrashListing {
        records,
        unreadable,
    } = store.list().unwrap();
    assert_eq!(records.len(), 20);
    assert_eq!(unreadable, ["99999999999999999999-1-0.json"]);
}

/// A real process killed by a signal: its exit status gives a `Signal` cause, and the record
/// written for it lists back. Spawning runner processes is DIST-E1-T04's; this kills a child the
/// test spawns itself.
#[cfg(unix)]
#[test]
fn a_process_killed_by_a_signal_leaves_a_crash_record() {
    let state = StateDir::new("signal");
    let store = CrashStore::new(&state.0);
    let mut child = std::process::Command::new("sleep")
        .arg("30")
        .spawn()
        .unwrap();
    child.kill().unwrap();
    let status = child.wait().unwrap();
    let cause = CrashCause::from_exit(&status).expect("a kill is a crash");
    assert_eq!(
        cause,
        CrashCause::Signal {
            signal: 9,
            name: Some("SIGKILL".into())
        }
    );
    let mut rec = record(1_800_000_000_000, cause);
    rec.process = CrashProcess::Runner {
        pid: child.id(),
        runner: "sleep".into(),
    };
    store.write(&rec).unwrap();
    let records = store.list().unwrap().records;
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].1, rec);

    // And a bundle built afterwards holds it.
    let sources = crate::bundle::Sources {
        crashes: Some(&store),
        ..crate::bundle::Sources::default()
    };
    let bundle = crate::bundle::build(
        &sources,
        &crate::bundle::BundleOptions::at(1_800_000_000_500),
    )
    .unwrap();
    let names = crate::bundle::verify(bundle.bytes()).unwrap();
    let file = format!("crashes/{}", records[0].0);
    assert!(names.contains(&file), "{names:?}");
    let archive = zstd::decode_all(bundle.bytes()).unwrap();
    let entry = crate::bundle::tar::read(&archive)
        .unwrap()
        .into_iter()
        .find(|f| f.name == file)
        .unwrap();
    assert!(
        String::from_utf8(entry.bytes)
            .unwrap()
            .contains("\"SIGKILL\"")
    );

    // A normal exit, with any code, is not a crash.
    let ok = std::process::Command::new("sh")
        .args(["-c", "exit 3"])
        .status()
        .unwrap();
    assert_eq!(CrashCause::from_exit(&ok), None);
}

#[cfg(windows)]
#[test]
fn an_exception_status_is_a_crash_and_an_exit_code_is_not() {
    use std::os::windows::process::ExitStatusExt as _;
    let crashed = ExitStatus::from_raw(0xC000_0005);
    assert_eq!(
        CrashCause::from_exit(&crashed),
        Some(CrashCause::Exception { code: 0xC000_0005 })
    );
    assert_eq!(CrashCause::from_exit(&ExitStatus::from_raw(1)), None);
}

#[test]
fn the_log_tail_keeps_two_hundred_lines_and_no_content() {
    let tail = LogTail::new();
    let wall = WallClock::new(1_800_000_000_000, MonoTime(0));
    for i in 0..250u64 {
        let event = LogEvent::new(MonoTime(i), Level::Info, "syntropy_daemon::test", "step")
            .with(Field::new("n", crate::logging::Value::U64(i)))
            .with(Field::content(
                "prompt",
                crate::logging::Value::Str("PROMPT-CANARY".into()),
            ));
        tail.push(&event, &wall);
    }
    let lines = tail.lines();
    assert_eq!(lines.len(), TAIL_LINES);
    assert!(lines[0].contains("n=50"), "{}", lines[0]);
    assert!(lines.iter().all(|l| !l.contains("PROMPT-CANARY")));
}

/// Set in the child process of the panic test: the state directory to record into.
const PANIC_CHILD: &str = "SYNTROPY_CRASH_TEST_PANIC_STATE";

/// The child half of the panic test: does nothing unless [`PANIC_CHILD`] is set. A panic hook is
/// process-wide, so the hook is installed only in a process of its own, where no other test's
/// panic can reach it.
#[test]
fn panic_child() {
    let Some(dir) = std::env::var_os(PANIC_CHILD) else {
        return;
    };
    let tail = Arc::new(LogTail::new());
    tail.push(
        &LogEvent::new(
            MonoTime(0),
            Level::Warn,
            "syntropy_daemon::test",
            "before the panic",
        )
        .with(Field::content(
            "output",
            crate::logging::Value::Str("OUTPUT-CANARY".into()),
        )),
        &WallClock::new(1_800_000_000_000, MonoTime(0)),
    );
    install_panic_hook(PanicRecorder {
        store: CrashStore::new(Path::new(&dir)),
        build_id: "cd".repeat(32),
        process: CrashProcess::Daemon {
            pid: std::process::id(),
        },
        clock: Arc::new(FixedClock::at(1_800_000_000_123)),
        tail,
    });
    panic!("the test's panic {}", 42);
}

/// A panic writes a record through the hook, and the hook it replaced still runs: the test
/// harness's, which prints the message.
#[test]
fn a_panic_is_recorded_by_the_hook_and_still_reaches_the_previous_one() {
    let state = StateDir::new("panic");
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "crash::tests::panic_child",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(PANIC_CHILD, &state.0)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "the child did not panic");
    assert!(
        stderr.contains("the test's panic 42"),
        "the previous hook did not print:\n{stderr}"
    );

    let CrashListing {
        records,
        unreadable,
    } = CrashStore::new(&state.0).list().unwrap();
    assert!(unreadable.is_empty());
    assert_eq!(records.len(), 1);
    let record = &records[0].1;
    match &record.cause {
        CrashCause::Panic {
            message,
            location,
            backtrace,
        } => {
            assert_eq!(message, "the test's panic 42");
            assert!(location.as_deref().unwrap().contains("tests.rs"));
            assert!(!backtrace.is_empty());
        }
        other => panic!("expected a panic, got {other:?}"),
    }
    assert_eq!(record.build_id, "cd".repeat(32));
    assert_eq!(record.at_unix_ms, 1_800_000_000_123);
    assert_eq!(record.last_logs.len(), 1);
    assert!(record.last_logs[0].contains("before the panic"));
    assert!(!record.last_logs[0].contains("OUTPUT-CANARY"));
}
