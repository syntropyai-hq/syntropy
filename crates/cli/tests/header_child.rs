// SPDX-License-Identifier: MIT OR Apache-2.0

//! The header child (DIST-E17-T16; RFC-0034 §10.4): `daemon::header_child::parse_in_child`
//! re-executes the `syntropy` binary as `__header`, which confines itself and validates one
//! file's header. A crash, a limit or a refusal costs the child and reports the file invalid;
//! the daemon's process is untouched.

#![cfg(any(target_os = "linux", target_vendor = "apple", windows))]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use syntropy_core::{Clock, MonoTime, Nanos};
use syntropy_daemon::header_child::{HeaderChildError, HeaderKind, HeaderLimits, ParsedHeader};
use syntropy_manifest::gguf::{GgufHeader, Limits as GgufLimits};
use syntropy_manifest::range::FileSource;
use syntropy_manifest::range::replay::Recording;
use syntropy_manifest::safetensors::read_header_default;

/// The `syntropy` binary cargo built for this test.
fn binary() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_syntropy"))
}

/// The daemon's `parse_in_child`, run again while the program is busy. A stand-in script this
/// test just wrote can be refused with `ETXTBSY` for a moment: another test thread may have
/// forked while the script's write descriptor was open, and that child holds a copy until it
/// execs.
fn parse_in_child(
    program: &Path,
    kind: HeaderKind,
    path: &Path,
    limits: &HeaderLimits,
    clock: &dyn Clock,
) -> Result<ParsedHeader, HeaderChildError> {
    let mut tries = 0;
    loop {
        match syntropy_daemon::header_child::parse_in_child(program, kind, path, limits, clock) {
            Err(HeaderChildError::Io(e))
                if e.kind() == std::io::ErrorKind::ExecutableFileBusy && tries < 100 =>
            {
                tries += 1;
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            other => return other,
        }
    }
}

/// A fresh directory of this test's own.
fn scratch(name: &str) -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "syntropy-header-child-{name}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// A real monotonic clock for the wall-clock limit.
struct Wall(std::time::Instant);

impl Clock for Wall {
    fn now(&self) -> MonoTime {
        MonoTime(self.0.elapsed().as_nanos() as u64)
    }
    fn unix_ms(&self) -> u64 {
        0
    }
}

fn wall() -> Wall {
    Wall(std::time::Instant::now())
}

/// A safetensors file of one F32 tensor, with `metadata_bytes` of `__metadata__` padding.
fn safetensors(dir: &Path, metadata_bytes: usize) -> PathBuf {
    let pad = "x".repeat(metadata_bytes);
    let json = format!(
        r#"{{"__metadata__":{{"pad":"{pad}"}},"t":{{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}}}"#
    );
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(json.as_bytes());
    bytes.extend_from_slice(&[0u8; 8]);
    let path = dir.join("model.safetensors");
    fs::write(&path, bytes).unwrap();
    path
}

/// A small GGUF v3 file, rebuilt from `manifest`'s synthetic recording of its header.
fn gguf(dir: &Path) -> PathBuf {
    let text = include_str!("../../manifest/tests/data/gguf/tiny-v3.gguf.hubrec");
    let recording = Recording::parse(text).unwrap();
    let mut bytes = vec![0u8; recording.length as usize];
    for (first, body) in &recording.responses {
        bytes[*first as usize..*first as usize + body.len()].copy_from_slice(body);
    }
    let path = dir.join("model.gguf");
    fs::write(&path, bytes).unwrap();
    path
}

fn parse(
    kind: HeaderKind,
    path: &Path,
    limits: &HeaderLimits,
) -> Result<ParsedHeader, HeaderChildError> {
    parse_in_child(binary(), kind, path, limits, &wall())
}

fn invalid_reason(result: Result<ParsedHeader, HeaderChildError>) -> String {
    match result {
        Err(HeaderChildError::Invalid { reason, .. }) => reason,
        other => panic!("expected an invalid file, got {other:?}"),
    }
}

/// A valid header comes back as the daemon's own reader parses it.
#[test]
fn a_valid_header_comes_back() {
    let dir = scratch("valid");
    let path = safetensors(&dir, 16);
    let in_process = read_header_default(FileSource::open(&path).unwrap()).unwrap();
    let parsed = parse(HeaderKind::Safetensors, &path, &HeaderLimits::default()).unwrap();
    assert_eq!(parsed, ParsedHeader::Safetensors(in_process));

    let path = gguf(&dir);
    let in_process =
        GgufHeader::read(FileSource::open(&path).unwrap(), &GgufLimits::DEFAULT).unwrap();
    let parsed = parse(HeaderKind::Gguf, &path, &HeaderLimits::default()).unwrap();
    assert_eq!(parsed, ParsedHeader::Gguf(in_process));
    fs::remove_dir_all(&dir).ok();
}

/// A header the reader refuses makes the file invalid, with the child's reason.
#[test]
fn a_bad_header_is_invalid() {
    let dir = scratch("bad");
    let path = dir.join("model.safetensors");
    fs::write(&path, [0xffu8; 64]).unwrap();
    let reason = invalid_reason(parse(
        HeaderKind::Safetensors,
        &path,
        &HeaderLimits::default(),
    ));
    assert!(reason.contains("exited with code 8"), "{reason}");
    fs::remove_dir_all(&dir).ok();
}

/// A program that stands in for a child whose parser crashes or hangs: a shell script.
#[cfg(unix)]
fn script(dir: &Path, body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let path = dir.join("child.sh");
    fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    path
}

/// A parser crash kills only the child: the daemon reports the file invalid and carries on.
#[cfg(unix)]
#[test]
fn a_crash_kills_only_the_child() {
    let dir = scratch("crash");
    let path = safetensors(&dir, 16);
    let crashing = script(&dir, "kill -SEGV $$");
    let reason = invalid_reason(parse_in_child(
        &crashing,
        HeaderKind::Safetensors,
        &path,
        &HeaderLimits::default(),
        &wall(),
    ));
    assert!(reason.contains("signal 11"), "{reason}");
    // This process, the daemon's stand-in, still parses the next file.
    assert!(parse(HeaderKind::Safetensors, &path, &HeaderLimits::default()).is_ok());
    fs::remove_dir_all(&dir).ok();
}

/// A child past the wall-clock limit is killed, and the file is invalid.
#[cfg(unix)]
#[test]
fn a_child_past_the_time_limit_is_killed() {
    let dir = scratch("slow");
    let path = safetensors(&dir, 16);
    let slow = script(&dir, "exec sleep 30");
    let start = std::time::Instant::now();
    let reason = invalid_reason(parse_in_child(
        &slow,
        HeaderKind::Safetensors,
        &path,
        &HeaderLimits {
            wall: Nanos(300_000_000),
            ..HeaderLimits::default()
        },
        &wall(),
    ));
    assert!(reason.contains("ran past"), "{reason}");
    assert!(start.elapsed() < std::time::Duration::from_secs(10));
    fs::remove_dir_all(&dir).ok();
}

/// On Linux a confined child cannot open a socket, so a connection to a listening address
/// fails; the same connection from this process succeeds.
#[cfg(target_os = "linux")]
#[test]
fn a_confined_child_cannot_connect() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    assert!(
        std::net::TcpStream::connect(addr).is_ok(),
        "the control connection"
    );
    let dir = scratch("net");
    let path = safetensors(&dir, 16);
    let output = std::process::Command::new(binary())
        .args(["__header", "--kind", "safetensors"])
        .arg("--path")
        .arg(&path)
        .args([
            "--memory",
            "1073741824",
            "--cpu-secs",
            "30",
            "--open-files",
            "32",
            "--max-header-bytes",
            "268435456",
        ])
        .args(["--check-network", &addr.to_string()])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    // The seccomp filter's EACCES on socket(), not some other failure.
    assert!(stderr.contains("PermissionDenied"), "{stderr}");
    fs::remove_dir_all(&dir).ok();
}

/// A header that needs more memory than the limit is refused, while a small one parses under
/// the same limit. Linux only: there the address-space limit is enforced and tested here.
#[cfg(target_os = "linux")]
#[test]
fn a_header_over_the_memory_limit_is_refused() {
    let limits = HeaderLimits {
        memory: 64 << 20,
        ..HeaderLimits::default()
    };
    let dir = scratch("mem");
    let small = safetensors(&dir, 16);
    assert!(parse(HeaderKind::Safetensors, &small, &limits).is_ok());
    let big_dir = scratch("membig");
    let big = safetensors(&big_dir, 96 << 20);
    let reason = invalid_reason(parse(HeaderKind::Safetensors, &big, &limits));
    assert!(
        reason.contains("signal 6") || reason.contains("memory allocation"),
        "{reason}"
    );
    // The same header parses with room for it.
    assert!(parse(HeaderKind::Safetensors, &big, &HeaderLimits::default()).is_ok());
    fs::remove_dir_all(&dir).ok();
    fs::remove_dir_all(&big_dir).ok();
}

/// A stand-in child that writes `reply` to stdout and exits with `code`.
#[cfg(unix)]
fn replying(dir: &Path, reply: &[u8], code: i32) -> PathBuf {
    let file = dir.join("reply.bin");
    fs::write(&file, reply).unwrap();
    script(dir, &format!("cat '{}'\nexit {code}", file.display()))
}

/// The daemon checks what a child sends: no length, a length that is not the file's, and a
/// header cut short are refused, and a confinement failure is the platform's, not the file's.
#[cfg(unix)]
#[test]
fn a_childs_reply_is_checked() {
    let dir = scratch("reply");
    let path = gguf(&dir);
    let file = fs::read(&path).unwrap();
    let len = file.len() as u64;
    let header = GgufHeader::read(FileSource::open(&path).unwrap(), &GgufLimits::DEFAULT)
        .unwrap()
        .header_bytes as usize;
    let run = |child: &Path| {
        parse_in_child(
            child,
            HeaderKind::Gguf,
            &path,
            &HeaderLimits::default(),
            &wall(),
        )
    };
    // The true reply passes, through a stand-in child too.
    let mut good = len.to_le_bytes().to_vec();
    good.extend_from_slice(&file[..header]);
    assert!(run(&replying(&dir, &good, 0)).is_ok());
    // Nothing at all.
    let reason = invalid_reason(run(&replying(&dir, b"", 0)));
    assert!(reason.contains("no length"), "{reason}");
    // Another file's length: the file changed between the daemon's look and the child's read.
    let mut other = (len + 1).to_le_bytes().to_vec();
    other.extend_from_slice(&file[..header]);
    let reason = invalid_reason(run(&replying(&dir, &other, 0)));
    assert!(reason.contains("changed"), "{reason}");
    // The header cut short: the daemon's reader would need bytes the child did not vet.
    let mut short = len.to_le_bytes().to_vec();
    short.extend_from_slice(&file[..header - 16]);
    let reason = invalid_reason(run(&replying(&dir, &short, 0)));
    assert!(!reason.is_empty());
    // A child that could not confine itself.
    match run(&script(&dir, "echo 'no seccomp here' >&2; exit 5")) {
        Err(HeaderChildError::Confinement(m)) => assert!(m.contains("no seccomp"), "{m}"),
        other => panic!("expected a confinement error, got {other:?}"),
    }
    fs::remove_dir_all(&dir).ok();
}

/// The child runs with an empty environment: a variable of the daemon's does not reach it.
#[cfg(unix)]
#[test]
fn the_child_sees_no_environment() {
    let dir = scratch("env");
    let path = safetensors(&dir, 16);
    // The stand-in child prints its environment as its reason. `/bin/sh` sets `PWD` itself;
    // nothing of the test's environment, such as `HOME` or `PATH`, may be there.
    assert!(std::env::var_os("HOME").is_some() && std::env::var_os("PATH").is_some());
    let child = script(&dir, "env >&2; exit 8");
    let reason = invalid_reason(parse_in_child(
        &child,
        HeaderKind::Safetensors,
        &path,
        &HeaderLimits::default(),
        &wall(),
    ));
    assert!(
        !reason.contains("HOME=") && !reason.contains("PATH="),
        "{reason}"
    );
    fs::remove_dir_all(&dir).ok();
}
