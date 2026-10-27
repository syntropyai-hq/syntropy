// SPDX-License-Identifier: MIT OR Apache-2.0

//! Output conventions (RFC-0051 §2.2): where output goes, colour, `--json`, units and labels.
//!
//! - Results go to stdout; progress, warnings and errors go to stderr.
//! - Colour only on a terminal, and never with `NO_COLOR`, `--color never` or `TERM=dumb`.
//! - `--json` prints exactly one JSON document, the native API's response type unchanged; a
//!   streaming command prints JSON Lines ([`JsonLines`]).
//! - Bytes in IEC units with three significant figures, or exact with `--bytes`; rates in SI;
//!   durations human-readable.
//! - A predicted figure carries "estimate" or `≈`; a measured one carries nothing.

use std::fmt::Write as _;
use std::io::{self, Write};
use std::time::Duration;

use serde::Serialize;
use syntropy_api::figure::FigureKind;

use crate::args::{ColorWhen, Global};

/// What the process's streams and environment allow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Terminal {
    /// Whether stdin is a terminal.
    pub stdin: bool,
    /// Whether stdout is a terminal.
    pub stdout: bool,
    /// Whether stderr is a terminal.
    pub stderr: bool,
    /// Whether `NO_COLOR` is set to a non-empty value.
    pub no_color: bool,
    /// Whether `TERM` is `dumb`.
    pub dumb: bool,
}

impl Terminal {
    /// The process's own streams and environment.
    #[must_use]
    pub fn detect() -> Terminal {
        use std::io::IsTerminal as _;
        Terminal {
            stdin: io::stdin().is_terminal(),
            stdout: io::stdout().is_terminal(),
            stderr: io::stderr().is_terminal(),
            no_color: std::env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty()),
            dumb: std::env::var_os("TERM").is_some_and(|v| v == "dumb"),
        }
    }

    /// Whether a question may be asked: stdin and stderr are terminals and `TERM` is not `dumb`.
    #[must_use]
    pub fn can_ask(self) -> bool {
        self.stdin && self.stderr && !self.dumb
    }
}

/// How a command prints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Output {
    /// `--json`.
    pub json: bool,
    /// Colour on stdout.
    pub color: bool,
    /// Rich output on stdout: aligned tables and live progress. Plain text otherwise.
    pub rich: bool,
    /// `--bytes`.
    pub exact_bytes: bool,
}

impl Output {
    /// The output the global options and `terminal` give.
    #[must_use]
    pub fn new(global: &Global, terminal: Terminal) -> Output {
        let color = match global.color {
            ColorWhen::Always => true,
            ColorWhen::Never => false,
            ColorWhen::Auto => terminal.stdout && !terminal.no_color && !terminal.dumb,
        };
        Output {
            json: global.json,
            color: color && !global.json,
            rich: terminal.stdout && !terminal.dumb && !global.json,
            exact_bytes: global.bytes,
        }
    }

    /// A byte count, as `--bytes` asks: exact, or IEC with three significant figures.
    #[must_use]
    pub fn bytes(self, n: u64) -> String {
        if self.exact_bytes {
            format!("{n} B")
        } else {
            iec_bytes(n)
        }
    }
}

/// `n` bytes in IEC units with three significant figures: `41.0 GiB`, `512 B`.
#[must_use]
pub fn iec_bytes(n: u64) -> String {
    const UNITS: [&str; 7] = ["KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB"];
    if n < 1024 {
        return format!("{n} B");
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "a display value with three significant figures"
    )]
    let mut value = n as f64 / 1024.0;
    let mut unit = 0;
    while value >= 1023.5 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{} {}", three_figures(value), UNITS[unit])
}

/// A rate in SI units: `12.3 tok/s`, `3.20 GB/s` for bytes per second.
#[must_use]
pub fn rate(value: f64, unit: &str) -> String {
    format!("{} {unit}", three_figures(value))
}

/// A duration, human-readable: `412 ms`, `1.20 s`, `3 min 5 s`.
#[must_use]
pub fn duration(d: Duration) -> String {
    let secs = d.as_secs_f64();
    if secs < 1e-3 {
        format!("{} µs", three_figures(secs * 1e6))
    } else if secs < 1.0 {
        format!("{} ms", three_figures(secs * 1e3))
    } else if secs < 60.0 {
        format!("{} s", three_figures(secs))
    } else {
        let whole = d.as_secs();
        format!("{} min {} s", whole / 60, whole % 60)
    }
}

/// A figure's label as RFC-0051 §2.2 prints it: `≈value estimate`, `value (target)`, or the
/// value alone when it is measured.
#[must_use]
pub fn labelled(value: &str, kind: FigureKind) -> String {
    match kind {
        FigureKind::Measured => value.to_owned(),
        FigureKind::Estimate => format!("≈{value} estimate"),
        FigureKind::Target => format!("{value} (target)"),
    }
}

/// `text` from a daemon, safe to print on a terminal: every control character, C0 (with ESC,
/// which starts CSI and OSC sequences), DEL and C1, is written as a `\u{…}` escape, so a
/// daemon's answer cannot move the cursor, retitle the window or forge other output. `--json`
/// output is left as the API sent it, for programs: `serde_json` escapes the C0 characters, ESC
/// among them, and nothing else.
#[must_use]
pub fn sanitize(text: &str) -> std::borrow::Cow<'_, str> {
    if !text.chars().any(char::is_control) {
        return std::borrow::Cow::Borrowed(text);
    }
    let mut safe = String::with_capacity(text.len() + 8);
    for c in text.chars() {
        if c.is_control() {
            let _ = write!(safe, "\\u{{{:x}}}", u32::from(c));
        } else {
            safe.push(c);
        }
    }
    std::borrow::Cow::Owned(safe)
}

/// `value` with three significant figures, as `41.0`, `512`, `3.20`.
fn three_figures(value: f64) -> String {
    if value >= 99.95 {
        format!("{value:.0}")
    } else if value >= 9.995 {
        format!("{value:.1}")
    } else {
        format!("{value:.2}")
    }
}

/// Prints `value` as one JSON document on stdout, with a newline (RFC-0051 §2.2).
///
/// # Errors
///
/// When stdout cannot be written.
pub fn print_json<T: Serialize>(value: &T) -> io::Result<()> {
    let mut out = io::stdout().lock();
    serde_json::to_writer(&mut out, value)?;
    writeln!(out)
}

/// A JSON Lines stream on a writer: one document per line, flushed as each is written, for the
/// streaming commands under `--json`.
pub struct JsonLines<W: Write> {
    out: W,
}

impl<W: Write> JsonLines<W> {
    /// A stream on `out`.
    pub fn new(out: W) -> JsonLines<W> {
        JsonLines { out }
    }

    /// Writes one document and a newline, then flushes, so a reader sees it at once.
    ///
    /// # Errors
    ///
    /// When `out` cannot be written.
    pub fn send<T: Serialize>(&mut self, value: &T) -> io::Result<()> {
        serde_json::to_writer(&mut self.out, value)?;
        self.out.write_all(b"\n")?;
        self.out.flush()
    }

    /// The writer back.
    pub fn into_inner(self) -> W {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn global(json: bool, color: ColorWhen) -> Global {
        Global {
            json,
            color,
            bytes: false,
        }
    }

    const TTY: Terminal = Terminal {
        stdin: true,
        stdout: true,
        stderr: true,
        no_color: false,
        dumb: false,
    };

    #[test]
    fn colour_needs_a_terminal_and_yields_to_no_color_dumb_and_never() {
        assert!(Output::new(&global(false, ColorWhen::Auto), TTY).color);
        let piped = Terminal {
            stdout: false,
            ..TTY
        };
        assert!(!Output::new(&global(false, ColorWhen::Auto), piped).color);
        let no_color = Terminal {
            no_color: true,
            ..TTY
        };
        assert!(!Output::new(&global(false, ColorWhen::Auto), no_color).color);
        let dumb = Terminal { dumb: true, ..TTY };
        assert!(!Output::new(&global(false, ColorWhen::Auto), dumb).color);
        assert!(!Output::new(&global(false, ColorWhen::Auto), dumb).rich);
        assert!(!Output::new(&global(false, ColorWhen::Never), TTY).color);
        assert!(Output::new(&global(false, ColorWhen::Always), piped).color);
        // JSON is never coloured or decorated.
        let json = Output::new(&global(true, ColorWhen::Always), TTY);
        assert!(!json.color && !json.rich && json.json);
    }

    #[test]
    fn a_question_needs_a_terminal_on_stdin_and_stderr_and_no_dumb_term() {
        assert!(TTY.can_ask());
        assert!(
            !Terminal {
                stdin: false,
                ..TTY
            }
            .can_ask()
        );
        assert!(
            !Terminal {
                stderr: false,
                ..TTY
            }
            .can_ask()
        );
        assert!(!Terminal { dumb: true, ..TTY }.can_ask());
    }

    #[test]
    fn bytes_are_iec_with_three_figures_or_exact() {
        assert_eq!(iec_bytes(0), "0 B");
        assert_eq!(iec_bytes(1023), "1023 B");
        assert_eq!(iec_bytes(1024), "1.00 KiB");
        assert_eq!(iec_bytes(1536), "1.50 KiB");
        assert_eq!(iec_bytes(41 << 30), "41.0 GiB");
        assert_eq!(iec_bytes(404 << 30), "404 GiB");
        assert_eq!(iec_bytes((1 << 40) - 1), "1.00 TiB");
        assert_eq!(iec_bytes(u64::MAX), "16.0 EiB");
        let exact = Output {
            json: false,
            color: false,
            rich: false,
            exact_bytes: true,
        };
        assert_eq!(exact.bytes(41 << 30), "44023414784 B");
    }

    #[test]
    fn rates_durations_and_labels_read_as_the_rfc_writes_them() {
        assert_eq!(rate(12.34, "tok/s"), "12.3 tok/s");
        assert_eq!(rate(3.2, "GB/s"), "3.20 GB/s");
        assert_eq!(duration(Duration::from_millis(412)), "412 ms");
        assert_eq!(duration(Duration::from_millis(1200)), "1.20 s");
        assert_eq!(duration(Duration::from_micros(85)), "85.0 µs");
        assert_eq!(duration(Duration::from_secs(185)), "3 min 5 s");
        assert_eq!(
            labelled("4.4 tok/s", FigureKind::Estimate),
            "≈4.4 tok/s estimate"
        );
        assert_eq!(labelled("41.0 GiB", FigureKind::Measured), "41.0 GiB");
        assert_eq!(labelled("50 ms", FigureKind::Target), "50 ms (target)");
    }

    #[test]
    fn control_characters_from_a_daemon_are_escaped() {
        assert_eq!(sanitize("plain text, ≈ ünïcode"), "plain text, ≈ ünïcode");
        assert_eq!(sanitize("a\u{1b}[2Jb"), "a\\u{1b}[2Jb");
        assert_eq!(sanitize("x\u{1b}]0;title\u{7}y"), "x\\u{1b}]0;title\\u{7}y");
        assert_eq!(sanitize("c1\u{9b}31m"), "c1\\u{9b}31m");
        assert_eq!(sanitize("line\nbreak\r"), "line\\u{a}break\\u{d}");
        assert_eq!(sanitize("del\u{7f}"), "del\\u{7f}");
    }

    #[test]
    fn json_lines_writes_one_document_per_line() {
        let mut lines = JsonLines::new(Vec::new());
        lines.send(&serde_json::json!({ "a": 1 })).unwrap();
        lines.send(&serde_json::json!({ "b": [2, 3] })).unwrap();
        assert_eq!(
            String::from_utf8(lines.into_inner()).unwrap(),
            "{\"a\":1}\n{\"b\":[2,3]}\n"
        );
    }
}
