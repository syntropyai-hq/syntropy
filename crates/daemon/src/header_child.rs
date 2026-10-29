// SPDX-License-Identifier: MIT OR Apache-2.0

//! The header of a newly added model file, parsed in a confined child (RFC-0034 §10.4, open
//! question 3; DIST-E17-T16).
//!
//! [`parse_in_child`] re-executes this binary as `syntropy __header` for one file. The child
//! confines itself (memory, CPU time, open files; on Linux, no new sockets), validates the
//! header with `manifest`'s reader, and writes the header's bytes to its stdout. The daemon waits
//! for it under a wall-clock limit on its [`Clock`], and parses only those vetted bytes, with the
//! same reader. A child that crashes, is killed by a limit, runs past the wall-clock limit or
//! refuses the header means the file is invalid ([`HeaderChildError::Invalid`]): a parser bug
//! costs the child, not the daemon.

use std::fmt;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::Duration;

use syntropy_core::{Clock, Nanos};
use syntropy_manifest::gguf::{GgufHeader, Limits as GgufLimits};
use syntropy_manifest::range::{RangeError, RangeSource, check_bounds};
use syntropy_manifest::safetensors::{SafetensorsHeader, parse_header};

/// A header's format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeaderKind {
    /// GGUF.
    Gguf,
    /// safetensors.
    Safetensors,
}

impl HeaderKind {
    fn arg(self) -> &'static str {
        match self {
            HeaderKind::Gguf => "gguf",
            HeaderKind::Safetensors => "safetensors",
        }
    }
}

/// The limits a header child runs under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeaderLimits {
    /// Memory: address space on Linux, data segment on macOS, process memory on Windows. Default
    /// 1 GiB \[target\]: the readers cap a header at 256 MiB (RFC-0034 §10.4), read once into
    /// memory, with room for the binary and the parse.
    pub memory: u64,
    /// CPU time. Default 30 s \[target\].
    pub cpu_secs: u64,
    /// Open files. Default 32 \[target\]: the file, stdio, and what the runtime opens.
    pub open_files: u64,
    /// Wall-clock time, on the daemon's [`Clock`]. Default 60 s \[target\].
    pub wall: Nanos,
}

impl Default for HeaderLimits {
    fn default() -> Self {
        Self {
            memory: 1 << 30,
            cpu_secs: 30,
            open_files: 32,
            wall: Nanos(60_000_000_000),
        }
    }
}

/// A header, as the daemon parsed it from the child's vetted bytes.
#[derive(Clone, Debug, PartialEq)]
pub enum ParsedHeader {
    /// A GGUF header.
    Gguf(GgufHeader),
    /// A safetensors header.
    Safetensors(SafetensorsHeader),
}

/// Why no header came back.
#[non_exhaustive]
#[derive(Debug)]
pub enum HeaderChildError {
    /// The file's length could not be read, or the child could not be spawned or waited for.
    Io(io::Error),
    /// The file is invalid: the child refused its header, crashed, hit a limit or ran past the
    /// wall-clock limit, or its bytes did not parse.
    Invalid {
        /// The file.
        path: PathBuf,
        /// Why, for the report.
        reason: String,
    },
}

impl fmt::Display for HeaderChildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HeaderChildError::Io(e) => write!(f, "the header child failed: {e}"),
            HeaderChildError::Invalid { path, reason } => {
                write!(f, "{} is invalid: {reason}", path.display())
            }
        }
    }
}

impl std::error::Error for HeaderChildError {}

/// The most stdout the daemon reads from a child: the readers' 256 MiB header cap, plus one
/// byte, so a longer reply is refused.
const MAX_REPLY: u64 = (256 << 20) + 1;

/// The most stderr the daemon keeps, for the report.
const MAX_STDERR: u64 = 64 << 10;

/// How often the daemon checks on a running child.
const POLL: Duration = Duration::from_millis(10);

/// Parses the `kind` header of `path` in a child: `program __header …` (module docs).
///
/// # Errors
///
/// [`HeaderChildError::Invalid`] when the file is invalid, and [`HeaderChildError::Io`] when the
/// child could not be run at all.
pub fn parse_in_child(
    program: &Path,
    kind: HeaderKind,
    path: &Path,
    limits: &HeaderLimits,
    clock: &dyn Clock,
) -> Result<ParsedHeader, HeaderChildError> {
    let invalid = |reason: String| HeaderChildError::Invalid {
        path: path.to_owned(),
        reason,
    };
    let file_len = fs::metadata(path).map_err(HeaderChildError::Io)?.len();
    let mut child = Command::new(program)
        .arg("__header")
        .args(["--kind", kind.arg()])
        .arg("--path")
        .arg(path)
        .args(["--memory", &limits.memory.to_string()])
        .args(["--cpu-secs", &limits.cpu_secs.to_string()])
        .args(["--open-files", &limits.open_files.to_string()])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(HeaderChildError::Io)?;
    let stdout = drain(child.stdout.take(), MAX_REPLY);
    let stderr = drain(child.stderr.take(), MAX_STDERR);
    let deadline = clock.now().saturating_add(limits.wall);
    let status = loop {
        match child.try_wait().map_err(HeaderChildError::Io)? {
            Some(status) => break Some(status),
            None if clock.now() > deadline => {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            None => thread::sleep(POLL),
        }
    };
    let bytes = stdout.join().unwrap_or_default();
    let message = String::from_utf8_lossy(&stderr.join().unwrap_or_default())
        .trim()
        .to_owned();
    let Some(status) = status else {
        return Err(invalid(format!(
            "the header child ran past the {} s limit and was killed",
            limits.wall.0 / 1_000_000_000
        )));
    };
    if !status.success() {
        return Err(invalid(format!(
            "the header child {}{}",
            describe(status),
            if message.is_empty() {
                String::new()
            } else {
                format!(": {message}")
            }
        )));
    }
    if bytes.len() as u64 >= MAX_REPLY {
        return Err(invalid("the header child's reply is too long".into()));
    }
    parse_vetted(kind, bytes, file_len).map_err(invalid)
}

/// Reads up to `max` bytes of `pipe` on a thread of its own, so a full pipe never stalls the
/// child.
fn drain<R: Read + Send + 'static>(pipe: Option<R>, max: u64) -> thread::JoinHandle<Vec<u8>> {
    thread::spawn(move || {
        let mut out = Vec::new();
        if let Some(pipe) = pipe {
            let _ = pipe.take(max).read_to_end(&mut out);
        }
        out
    })
}

/// How a child ended, for the report.
fn describe(status: ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return format!("was killed by signal {signal}");
        }
    }
    match status.code() {
        Some(code) => format!("exited with code {code}"),
        None => format!("ended with {status}"),
    }
}

/// Parses the header bytes a child vetted, of a file `file_len` bytes long.
fn parse_vetted(kind: HeaderKind, bytes: Vec<u8>, file_len: u64) -> Result<ParsedHeader, String> {
    match kind {
        HeaderKind::Gguf => {
            let vetted = bytes.len() as u64;
            let header = GgufHeader::read(Prefix { bytes, file_len }, &GgufLimits::DEFAULT)
                .map_err(|e| e.to_string())?;
            // The reader reads ahead past the header; it must still end where the child's bytes
            // end, so nothing past them shaped it.
            if header.header_bytes != vetted {
                return Err(format!(
                    "the header ends at byte {}, not at the {vetted} the child vetted",
                    header.header_bytes
                ));
            }
            Ok(ParsedHeader::Gguf(header))
        }
        HeaderKind::Safetensors => {
            let (len, json) = bytes
                .split_first_chunk::<8>()
                .ok_or("the reply is shorter than a safetensors length")?;
            if u64::from_le_bytes(*len) != json.len() as u64 {
                return Err("the reply's length does not match its header".into());
            }
            parse_header(json, file_len)
                .map(ParsedHeader::Safetensors)
                .map_err(|e| e.to_string())
        }
    }
}

/// The first bytes of a file of `file_len` bytes, as a source: within them, the bytes; past
/// them, zeros. The GGUF reader reads ahead in large steps before it knows where the header ends;
/// [`parse_vetted`] then checks that the header ends where the vetted bytes do.
struct Prefix {
    bytes: Vec<u8>,
    file_len: u64,
}

impl RangeSource for Prefix {
    fn len(&self) -> u64 {
        self.file_len
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), RangeError> {
        check_bounds(offset, buf.len() as u64, self.file_len)?;
        buf.fill(0);
        let start = usize::try_from(offset).unwrap_or(usize::MAX);
        if let Some(have) = self.bytes.get(start..) {
            let n = have.len().min(buf.len());
            buf[..n].copy_from_slice(&have[..n]);
        }
        Ok(())
    }
}
