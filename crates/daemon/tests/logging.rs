// SPDX-License-Identifier: MIT OR Apache-2.0

//! The daemon's logging end to end (PROD-E8-T04; RFC-0053 §4).
//!
//! Records come in through the `tracing` subscriber and through runner `LogRecord`s, with time
//! from a `SimClock`, so repeat windows are exact. Sinks are real where the test can read them
//! back (stderr through a captured writer, the rotating file, journald's protocol over a socket
//! the test binds) and, for unified logging and the event log, what each would be handed,
//! through the same rendering the real sink uses. `io`'s own tests call the two OS facilities.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use syntropy_config::logging::{LogFile, LogFormat, LogLevel, LoggingSection};
use syntropy_core::{MonoTime, Nanos};
use syntropy_daemon::logging::sink::{
    FILE_COUNT, Rendered, RotatingFile, Sink, SinkPlan, Situation, plan,
};
use syntropy_daemon::logging::{
    CONTENT_WARNING, Context, LogEvent, LogSubscriber, Logger, from_record,
};
use syntropy_obs::log::{Callsite, CallsiteId, Level, LogRecord};
use syntropy_testkit::sim::SimClock;

/// A prompt, an output and a token ID no other text contains.
const PROMPT: &str = "CANARY-PROMPT-6f1e9c";
const OUTPUT: &str = "CANARY-OUTPUT-2b7d40";
const TOKEN: u64 = 987_654_321_123;

/// stderr, captured.
#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<u8>>>);

impl Write for Captured {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Captured {
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.0.lock().unwrap()).into_owned()
    }
}

fn section(level: LogLevel, include_content: bool) -> LoggingSection {
    LoggingSection {
        level,
        filter: String::new(),
        format: LogFormat::Auto,
        file: LogFile::Auto,
        include_content,
    }
}

/// A fresh directory for one test.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("syntropy-logging-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// A runner callsite with a content argument: a token ID.
const SAMPLED: Callsite = Callsite {
    id: CallsiteId(3),
    level: Level::Debug,
    target: "syntropy_sched::sampler",
    format: "sampled token {token} at position {position}",
    fields: &["token", "position"],
    content: 0b01,
    key_fields: 0,
    file: "crates/sched/src/sampler.rs",
    line: 10,
};

const CALLSITES: &[Callsite] = &[SAMPLED];

/// Every record the canary test sends: `tracing` events with content fields at `warn` and
/// `debug`, and a runner record whose content argument is a token ID.
fn send_canaries(logger: &Arc<Logger>, clock: &mut SimClock) {
    tracing::subscriber::with_default(LogSubscriber::new(Arc::clone(logger)), || {
        tracing::warn!(
            content.prompt = PROMPT,
            content.output = OUTPUT,
            model = "m",
            "request failed"
        );
        tracing::debug!(
            content.prompt = PROMPT,
            content.tokens = ?[TOKEN, 1, 2],
            tokens = 3u64,
            "request summary"
        );
    });
    clock.advance(Nanos(1_000));
    let record = LogRecord {
        callsite: SAMPLED.id,
        level: Level::Debug,
        at: MonoTime(2_000),
        args: [TOKEN, 17, 0, 0],
    };
    logger.log(from_record(&record, CALLSITES, Context::default()).unwrap());
    logger.flush();
}

/// The sinks of the canary test: stderr as text and as JSON (captured), the file as text and
/// as JSON, journald, unified logging and the event log (what each is handed).
struct AllSinks {
    stderr: Captured,
    stderr_json: Captured,
    files: [PathBuf; 2],
}

fn all_sinks(dir: &Path) -> (Vec<Sink>, AllSinks) {
    let (stderr, stderr_json) = (Captured::default(), Captured::default());
    let files = [dir.join("text.log"), dir.join("json.log")];
    let sinks = vec![
        Sink::Stderr {
            out: Box::new(stderr.clone()),
            json: false,
            colour: true,
        },
        Sink::Stderr {
            out: Box::new(stderr_json.clone()),
            json: true,
            colour: false,
        },
        Sink::File(RotatingFile::open(files[0].clone(), false, 1 << 20, 5).unwrap()),
        Sink::File(RotatingFile::open(files[1].clone(), true, 1 << 20, 5).unwrap()),
        Sink::Memory {
            plan: SinkPlan::Journald,
            kept: Vec::new(),
        },
        Sink::Memory {
            plan: SinkPlan::UnifiedLog {
                subsystem: "org.syntropy".to_owned(),
            },
            kept: Vec::new(),
        },
        Sink::Memory {
            plan: SinkPlan::EventLog,
            kept: Vec::new(),
        },
    ];
    (
        sinks,
        AllSinks {
            stderr,
            stderr_json,
            files,
        },
    )
}

/// Everything every sink was handed, as text.
fn everything(logger: &Logger, sinks: &AllSinks) -> String {
    let mut all = sinks.stderr.text() + &sinks.stderr_json.text();
    for file in &sinks.files {
        all += &fs::read_to_string(file).unwrap();
    }
    logger.with_sinks(|open| {
        for sink in open.iter() {
            if let Sink::Memory { kept, .. } = sink {
                for rendered in kept {
                    all += &match rendered {
                        Rendered::Line(line) | Rendered::EventLog(line) => line.clone(),
                        Rendered::UnifiedLog { line, .. } => line.clone(),
                        Rendered::Journal(bytes) => String::from_utf8_lossy(bytes).into_owned(),
                        Rendered::Skip => String::new(),
                    };
                }
            }
        }
    });
    all
}

/// The criterion "the canary test finds no prompt, output or token ID in any sink": with
/// `logging.include_content` off, at `trace` so every record is written, no sink is handed a
/// canary.
#[test]
fn no_sink_receives_content_without_include_content() {
    let dir = scratch("canary-off");
    let mut clock = SimClock::starting_at(MonoTime(0), 1_793_125_815_000);
    let (sinks, read) = all_sinks(&dir);
    let logger = Arc::new(Logger::new(
        Arc::new(clock.handle()),
        &section(LogLevel::Trace, false),
        sinks,
    ));
    send_canaries(&logger, &mut clock);
    let all = everything(&logger, &read);
    assert!(
        all.contains("request failed") && all.contains("sampled token [content]"),
        "{all}"
    );
    for canary in [PROMPT, OUTPUT, &TOKEN.to_string()] {
        assert!(!all.contains(canary), "{canary} reached a sink:\n{all}");
    }
}

/// The positive control: with `logging.include_content` on, the same records carry the canaries
/// at `debug`, but still not at `warn`, and the logger warns that content is on. So the test above
/// would see a canary if one leaked.
#[test]
fn content_appears_only_at_debug_with_include_content() {
    let dir = scratch("canary-on");
    let mut clock = SimClock::starting_at(MonoTime(0), 0);
    let (sinks, read) = all_sinks(&dir);
    let logger = Arc::new(Logger::new(
        Arc::new(clock.handle()),
        &section(LogLevel::Trace, true),
        sinks,
    ));
    send_canaries(&logger, &mut clock);
    let text = read.stderr.text();
    let warn_line = text.lines().find(|l| l.contains("request failed")).unwrap();
    assert!(!warn_line.contains(PROMPT), "content at warn: {warn_line}");
    let debug_line = text
        .lines()
        .find(|l| l.contains("request summary"))
        .unwrap();
    assert!(debug_line.contains(PROMPT) && debug_line.contains(&TOKEN.to_string()));
    assert!(text.contains(&TOKEN.to_string()));
    assert_eq!(text.matches(CONTENT_WARNING).count(), 1);
    logger.apply(&section(LogLevel::Trace, true));
    assert_eq!(
        read.stderr.text().matches(CONTENT_WARNING).count(),
        2,
        "the warning repeats on reload"
    );
    let _ = everything(&logger, &read);
}

/// The criterion "1,000 identical warnings in 10 s become one record with `repeated`", through
/// the `tracing` subscriber and the real pipeline: the first is written at once; the 999 repeats
/// become one record with `repeated=999` when the window closes on the logger's clock.
#[test]
fn a_thousand_identical_warnings_in_ten_seconds_become_one_record() {
    let mut clock = SimClock::starting_at(MonoTime(0), 0);
    let stderr = Captured::default();
    let logger = Arc::new(Logger::new(
        Arc::new(clock.handle()),
        &section(LogLevel::Info, false),
        vec![Sink::Stderr {
            out: Box::new(stderr.clone()),
            json: true,
            colour: false,
        }],
    ));
    let subscriber = LogSubscriber::new(Arc::clone(&logger));
    tracing::subscriber::with_default(subscriber, || {
        for _ in 0..1_000 {
            tracing::warn!(link = "eth0", "link flapped");
            clock.advance(Nanos(9_000_000));
            logger.tick();
        }
    });
    // 1,000 × 9 ms = 9 s: the window is still open.
    let lines: Vec<serde_json::Value> = stderr
        .text()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 1, "{}", stderr.text());
    assert!(lines[0].get("repeated").is_none());
    clock.advance(Nanos(1_000_000_000));
    logger.tick();
    let lines: Vec<serde_json::Value> = stderr
        .text()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[1]["repeated"], 999);
    assert_eq!(lines[1]["message"], "link flapped");
    assert_eq!(lines[1]["link"], "eth0");
}

/// RFC-0053 §4.2's table.
#[test]
fn the_sinks_follow_the_situation() {
    let state = Path::new("/state");
    let auto = state.join("logs").join("daemon.log");
    let default = section(LogLevel::Info, false);
    assert_eq!(
        plan(Situation::Foreground, &default, state, true, "org.syntropy"),
        [SinkPlan::Stderr {
            json: false,
            colour: true
        }]
    );
    assert_eq!(
        plan(Situation::Systemd, &default, state, false, "org.syntropy"),
        [SinkPlan::Journald]
    );
    assert_eq!(
        plan(Situation::Launchd, &default, state, false, "org.syntropy"),
        [
            SinkPlan::UnifiedLog {
                subsystem: "org.syntropy".to_owned()
            },
            SinkPlan::File {
                path: auto.clone(),
                json: false
            }
        ]
    );
    assert_eq!(
        plan(
            Situation::WindowsService,
            &default,
            state,
            false,
            "org.syntropy"
        ),
        [
            SinkPlan::EventLog,
            SinkPlan::File {
                path: auto,
                json: false
            }
        ]
    );
    let mut json = default.clone();
    json.format = LogFormat::Json;
    json.file = LogFile::Path(PathBuf::from("/var/log/s.log"));
    assert_eq!(
        plan(Situation::Systemd, &json, state, false, "x"),
        [
            SinkPlan::Journald,
            SinkPlan::File {
                path: PathBuf::from("/var/log/s.log"),
                json: true
            }
        ]
    );
    let mut off = default;
    off.file = LogFile::Off;
    assert_eq!(
        plan(Situation::Launchd, &off, state, false, "x"),
        [SinkPlan::UnifiedLog {
            subsystem: "x".to_owned()
        }]
    );
    let env = |pairs: &'static [(&'static str, &'static str)]| {
        move |name: &str| {
            pairs
                .iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| (*v).to_owned())
        }
    };
    assert_eq!(
        Situation::from_env(env(&[("JOURNAL_STREAM", "8:123")])),
        Situation::Systemd
    );
    assert_eq!(
        Situation::from_env(env(&[("XPC_SERVICE_NAME", "org.syntropy.daemon")])),
        Situation::Launchd
    );
    assert_eq!(
        Situation::from_env(env(&[("XPC_SERVICE_NAME", "0")])),
        Situation::Foreground
    );
    assert_eq!(Situation::from_env(env(&[])), Situation::Foreground);
}

/// The rotating file keeps `daemon.log` and four older files, oldest dropped.
#[test]
fn the_file_rotates_and_keeps_five() {
    let dir = scratch("rotate");
    let path = dir.join("logs").join("daemon.log");
    let mut file = RotatingFile::open(path.clone(), false, 100, FILE_COUNT).unwrap();
    for n in 0..40 {
        file.write(&format!("line {n:02} {}\n", "x".repeat(40)))
            .unwrap();
    }
    drop(file);
    let mut names: Vec<String> = fs::read_dir(dir.join("logs"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "daemon.log",
            "daemon.log.1",
            "daemon.log.2",
            "daemon.log.3",
            "daemon.log.4"
        ]
    );
    assert!(fs::read_to_string(&path).unwrap().contains("line 39"));
    assert!(
        fs::read_to_string(dir.join("logs").join("daemon.log.4"))
            .unwrap()
            .contains("line 31")
    );
    for name in &names {
        assert!(fs::metadata(dir.join("logs").join(name)).unwrap().len() <= 100);
    }
}

/// The per-OS sink test for journald (Unix): the sink sends the native protocol to a socket the
/// test binds in journald's place, and the test reads the fields back. Reading the real journal
/// is the ignored test below.
#[cfg(unix)]
#[test]
fn journald_receives_structured_fields() {
    let dir = scratch("journal");
    let socket = dir.join("journal.sock");
    let server = std::os::unix::net::UnixDatagram::bind(&socket).unwrap();
    let clock = SimClock::starting_at(MonoTime(0), 0);
    let logger = Logger::new(
        Arc::new(clock.handle()),
        &section(LogLevel::Info, false),
        vec![Sink::Journald {
            socket: socket.clone(),
        }],
    );
    logger.log(
        LogEvent::new(MonoTime(0), Level::Error, "syntropy_net::link", "link down").with(
            syntropy_daemon::logging::Field::new(
                "link",
                syntropy_daemon::logging::Value::Str("eth0".to_owned()),
            ),
        ),
    );
    let mut buf = vec![0; 4096];
    let len = server.recv(&mut buf).unwrap();
    let text = String::from_utf8_lossy(&buf[..len]);
    for field in [
        "MESSAGE=link down\n",
        "PRIORITY=3\n",
        "SYSLOG_IDENTIFIER=syntropy\n",
        "SYNTROPY_TARGET=syntropy_net::link\n",
        "SYNTROPY_LINK=eth0\n",
    ] {
        assert!(text.contains(field), "{field:?} not in {text:?}");
    }
}

/// Writes a unique record to the real journal and reads it back with `journalctl`.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "needs a running journald and journalctl"]
fn the_real_journal_receives_records() {
    let marker = format!("syntropy-journald-{}", std::process::id());
    let payload = format!("MESSAGE={marker}\nPRIORITY=4\nSYSLOG_IDENTIFIER=syntropy\n");
    syntropy_daemon::logging::sink::send_journal(
        Path::new(syntropy_daemon::logging::sink::JOURNAL_SOCKET),
        payload.as_bytes(),
    )
    .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(500));
    let out = std::process::Command::new("journalctl")
        .args([
            "--no-pager",
            "-o",
            "cat",
            "-t",
            "syntropy",
            "--since",
            "-1min",
        ])
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&out.stdout).contains(&marker));
}

/// The per-OS sink test for unified logging (macOS) and the event log (Windows): the planned
/// sink opens on its OS and takes records of every level; the event log keeps warnings and
/// errors only. What the OS keeps is read back by `io`'s ignored tests.
#[test]
fn the_os_sinks_open_and_take_records() {
    let clock = SimClock::starting_at(MonoTime(0), 0);
    let plans = [
        SinkPlan::UnifiedLog {
            subsystem: "org.syntropy.test".to_owned(),
        },
        SinkPlan::EventLog,
    ];
    let sinks: Vec<Sink> = plans.iter().map(|p| p.open().unwrap()).collect();
    #[cfg(target_os = "macos")]
    assert!(
        matches!(sinks[0], Sink::UnifiedLog { .. }),
        "{:?}",
        sinks[0]
    );
    #[cfg(windows)]
    assert!(matches!(sinks[1], Sink::EventLog(_)), "{:?}", sinks[1]);
    let logger = Logger::new(
        Arc::new(clock.handle()),
        &section(LogLevel::Trace, false),
        sinks,
    );
    for level in [Level::Error, Level::Warn, Level::Info, Level::Debug] {
        logger.log(LogEvent::new(
            MonoTime(0),
            level,
            "syntropy_daemon::test",
            format!("os sink test at {}", level.name()),
        ));
    }
    logger.flush();
    // Where the OS lacks the facility, the sink kept what it would have been handed.
    logger.with_sinks(|sinks| {
        if let Sink::Memory { kept, .. } = &sinks[1] {
            let lines = kept
                .iter()
                .filter(|r| matches!(r, Rendered::EventLog(_)))
                .count();
            assert_eq!(lines, 2, "the event log takes warnings and errors only");
        }
    });
}

/// `logging.level` and `logging.filter` choose what is written.
#[test]
fn the_level_and_filter_apply() {
    let clock = SimClock::starting_at(MonoTime(0), 0);
    let stderr = Captured::default();
    let mut filtered = section(LogLevel::Warn, false);
    filtered.filter = "syntropy_net=debug, syntropy_net::quic=error, bogus".to_owned();
    let logger = Logger::new(
        Arc::new(clock.handle()),
        &filtered,
        vec![Sink::Stderr {
            out: Box::new(stderr.clone()),
            json: false,
            colour: false,
        }],
    );
    let log = |level, target: &'static str| {
        logger.log(LogEvent::new(
            MonoTime(0),
            level,
            target,
            format!("{target} {level:?}"),
        ));
    };
    log(Level::Info, "syntropy_api");
    log(Level::Warn, "syntropy_api");
    log(Level::Debug, "syntropy_net::link");
    log(Level::Warn, "syntropy_net::quic");
    log(Level::Error, "syntropy_net::quic");
    let text = stderr.text();
    assert!(!text.contains("syntropy_api Info"));
    assert!(text.contains("syntropy_api Warn"));
    assert!(text.contains("syntropy_net::link Debug"));
    assert!(!text.contains("syntropy_net::quic Warn"));
    assert!(text.contains("syntropy_net::quic Error"));
}
