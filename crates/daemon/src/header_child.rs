// SPDX-License-Identifier: MIT OR Apache-2.0

//! The header of a newly added model file, parsed in a confined child (RFC-0034 §10.4, open
//! question 3; DIST-E17-T16).
//!
//! [`parse_in_child`] re-executes this binary as `syntropy __header` for one file, with an empty
//! environment. The child confines itself (memory, CPU time, open files; on Linux, no new sockets
//! and no inherited descriptors), validates the header with `manifest`'s reader under
//! [`HeaderLimits::max_header_bytes`], and writes the file's length and the header's bytes to its
//! stdout. The daemon waits for it under a wall-clock limit on its [`Clock`], checks the length
//! against its own reading of the file, and parses only those vetted bytes, with the same reader
//! and cap. A child that crashes, is killed by a limit, runs past the wall-clock limit or refuses
//! the header means the file is invalid ([`HeaderChildError::Invalid`]); a child that cannot
//! confine itself is [`HeaderChildError::Confinement`], the platform's failure, not the file's.
//!
//! What the child contains: a parser crash, and a header past a limit. What it does not: the
//! daemon's own parse of the vetted bytes holds the reply, the reader's growing prefix and its
//! copies at once, about 3 to 5 times the header's length \[estimate: one reply buffer, a prefix
//! grown by doubling, and the reader's copies\], bounded by the cap. This function blocks for up
//! to the wall-clock limit; call it from a blocking thread.

use std::fmt;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;
use std::time::Duration;

use syntropy_core::{Clock, Nanos};
use syntropy_manifest::gguf::{GgufHeader, Limits as GgufLimits};
use syntropy_manifest::range::Prefix;
use syntropy_manifest::safetensors::{SafetensorsHeader, read_header};

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
    /// Memory: address space on Linux, and set but not confirmed enforced as the data segment on
    /// macOS and process memory on Windows. Default 1 GiB \[target\]. At its peak the child
    /// holds about 4 to 5 copies of the header \[estimate: the reader's prefix, the bytes it
    /// sends and their re-validation\], so a header in the top fifth of
    /// [`max_header_bytes`](Self::max_header_bytes) can be refused by this limit. That refusal
    /// fails safe: the file is reported invalid.
    pub memory: u64,
    /// CPU time (user time on Windows). Default 30 s \[target\].
    pub cpu_secs: u64,
    /// Open files, on Linux and macOS; Windows has no such limit. Default 32 \[target\]: the
    /// file and stdio.
    pub open_files: u64,
    /// The longest header accepted, by the child and by the daemon's own parse. Default 256 MiB,
    /// RFC-0034 §10.4's header limit \[target\].
    pub max_header_bytes: u64,
    /// Wall-clock time, on the daemon's [`Clock`]. Default 60 s \[target\].
    pub wall: Nanos,
}

impl Default for HeaderLimits {
    fn default() -> Self {
        Self {
            memory: 1 << 30,
            cpu_secs: 30,
            open_files: 32,
            max_header_bytes: 256 << 20,
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
    /// The child could not confine itself on this platform: no file was read.
    Confinement(String),
    /// The file is invalid: the child refused its header, crashed, hit a limit or ran past the
    /// wall-clock limit, its reply did not parse, or the file changed while it was read.
    Invalid {
        /// The file.
        path: PathBuf,
        /// Why, for the report: control characters escaped, at most [`MAX_REASON`] bytes.
        reason: String,
    },
}

impl fmt::Display for HeaderChildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HeaderChildError::Io(e) => write!(f, "the header child failed: {e}"),
            HeaderChildError::Confinement(m) => {
                write!(f, "the header child could not be confined: {m}")
            }
            HeaderChildError::Invalid { path, reason } => {
                write!(f, "{} is invalid: {reason}", path.display())
            }
        }
    }
}

impl std::error::Error for HeaderChildError {}

/// The longest reason kept from the child's stderr.
pub const MAX_REASON: usize = 1024;

/// The child's exit code when it cannot confine itself (`cli`'s `Exit::Invalid`).
const EXIT_CONFINEMENT: i32 = 5;

/// The most stderr the daemon reads.
const MAX_STDERR: u64 = 64 << 10;

/// How often the daemon checks on a running child.
const POLL: Duration = Duration::from_millis(10);

/// Parses the `kind` header of `path` in a child: `program __header …` (module docs).
///
/// # Errors
///
/// [`HeaderChildError::Invalid`] when the file is invalid, [`HeaderChildError::Confinement`]
/// when the child cannot be confined, and [`HeaderChildError::Io`] when it could not be run.
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
    let mut command = Command::new(program);
    command
        .arg("__header")
        .args(["--kind", kind.arg()])
        .args(["--memory", &limits.memory.to_string()])
        .args(["--cpu-secs", &limits.cpu_secs.to_string()])
        .args(["--open-files", &limits.open_files.to_string()])
        .args(["--max-header-bytes", &limits.max_header_bytes.to_string()])
        .arg("--path")
        .arg(path)
        // The child sees none of the daemon's environment, such as an API key.
        .env_clear()
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    if let Some(root) = std::env::var_os("SystemRoot") {
        command.env("SystemRoot", root);
    }
    let mut child = command.spawn().map_err(HeaderChildError::Io)?;
    // 8 bytes of length, the header at the cap, and one byte more, so a longer reply is refused.
    let max_reply = limits.max_header_bytes.saturating_add(9);
    let stdout = drain(child.stdout.take(), max_reply);
    let stderr = drain(child.stderr.take(), MAX_STDERR);
    let status = wait(&mut child, clock, limits.wall);
    let bytes = stdout.join().unwrap_or_default();
    let message = reason_text(&stderr.join().unwrap_or_default());
    let status = match status {
        Err(e) => return Err(HeaderChildError::Io(e)),
        Ok(None) => {
            return Err(invalid(format!(
                "the header child ran past the {} ms limit and was killed",
                limits.wall.0 / 1_000_000
            )));
        }
        Ok(Some(status)) => status,
    };
    if status.code() == Some(EXIT_CONFINEMENT) {
        return Err(HeaderChildError::Confinement(message));
    }
    if !status.success() {
        let tail = if message.is_empty() {
            String::new()
        } else {
            format!(": {message}")
        };
        return Err(invalid(format!(
            "the header child {}{tail}",
            describe(status)
        )));
    }
    if bytes.len() as u64 >= max_reply {
        return Err(invalid("the header child's reply is too long".into()));
    }
    let Some((len, _)) = bytes.split_first_chunk::<8>() else {
        return Err(invalid("the header child's reply has no length".into()));
    };
    let seen = u64::from_le_bytes(*len);
    if seen != file_len {
        return Err(invalid(format!(
            "the file changed while its header was read: {file_len} bytes, then {seen}"
        )));
    }
    // The header's bytes, moved down in place rather than copied.
    let mut header = bytes;
    header.drain(..8);
    parse_vetted(kind, header, file_len, limits.max_header_bytes).map_err(invalid)
}

/// Waits for `child` until `wall` passes on `clock`, then kills it. `Ok(None)` when it was killed
/// for time. The child is reaped on every path.
fn wait(child: &mut Child, clock: &dyn Clock, wall: Nanos) -> io::Result<Option<ExitStatus>> {
    let deadline = clock.now().saturating_add(wall);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(Some(status)),
            Ok(None) if clock.now() > deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Ok(None);
            }
            Ok(None) => thread::sleep(POLL),
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(e);
            }
        }
    }
}

/// The child's stderr as a reason: control characters escaped, at most [`MAX_REASON`] bytes, cut
/// only between characters.
fn reason_text(bytes: &[u8]) -> String {
    let mut out = String::new();
    for c in String::from_utf8_lossy(bytes).trim().chars() {
        let piece: String = if c.is_control() {
            c.escape_default().collect()
        } else {
            c.into()
        };
        if out.len() + piece.len() > MAX_REASON {
            break;
        }
        out.push_str(&piece);
    }
    out
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

/// Parses the header bytes a child vetted, of a file `file_len` bytes long, under `cap`.
fn parse_vetted(
    kind: HeaderKind,
    bytes: Vec<u8>,
    file_len: u64,
    cap: u64,
) -> Result<ParsedHeader, String> {
    let prefix = Prefix::new(bytes, file_len);
    let held = prefix.held();
    match kind {
        HeaderKind::Gguf => {
            let limits = GgufLimits {
                header_bytes: cap,
                ..GgufLimits::DEFAULT
            };
            let header = GgufHeader::read(&prefix, &limits).map_err(|e| e.to_string())?;
            // The reader reads ahead past the header; it must still end where the vetted bytes
            // end, so nothing past them shaped it.
            if header.header_bytes != held {
                return Err(format!(
                    "the header ends at byte {}, not at the {held} the child vetted",
                    header.header_bytes
                ));
            }
            Ok(ParsedHeader::Gguf(header))
        }
        HeaderKind::Safetensors => {
            let header = read_header(&prefix, cap).map_err(|e| e.to_string())?;
            if header.data_start != held {
                return Err(format!(
                    "the header ends at byte {}, not at the {held} the child vetted",
                    header.data_start
                ));
            }
            Ok(ParsedHeader::Safetensors(header))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reason_is_escaped_and_capped() {
        assert_eq!(reason_text(b"bad \x1b[31mname\n"), "bad \\u{1b}[31mname");
        assert_eq!(reason_text(&[b'x'; 5000]).len(), MAX_REASON);
        // A character that would cross the cap is left out whole, never split. A hostile file
        // chooses the text: a parser's error quotes its names.
        for c in ['\u{e9}', '\u{1f600}'] {
            for before in MAX_REASON - 4..=MAX_REASON {
                let mut text = "x".repeat(before);
                text.push(c);
                text.push_str("tail");
                let reason = reason_text(text.as_bytes());
                assert!(reason.len() <= MAX_REASON, "{before} {c}");
                assert!(text.starts_with(&reason), "{before} {c}");
            }
        }
        // An escape that would cross the cap is left out whole too.
        let mut text = "x".repeat(MAX_REASON - 2);
        text.push('\x1b');
        assert_eq!(reason_text(text.as_bytes()), "x".repeat(MAX_REASON - 2));
    }

    #[test]
    fn vetted_bytes_must_end_at_the_header() {
        let json = br#"{"t":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut file = (json.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(json);
        let header_len = file.len();
        file.extend_from_slice(&[0u8; 8]);
        let len = file.len() as u64;
        let cap = 1 << 20;
        // The header's bytes parse.
        assert!(
            parse_vetted(
                HeaderKind::Safetensors,
                file[..header_len].to_vec(),
                len,
                cap
            )
            .is_ok()
        );
        // Short of the header: refused.
        assert!(
            parse_vetted(
                HeaderKind::Safetensors,
                file[..header_len - 4].to_vec(),
                len,
                cap
            )
            .is_err()
        );
        // Past the header: refused, since the header ends before the bytes do.
        assert!(parse_vetted(HeaderKind::Safetensors, file.clone(), len, cap).is_err());
        // Nothing: refused.
        assert!(parse_vetted(HeaderKind::Gguf, Vec::new(), len, cap).is_err());
    }
}
