// SPDX-License-Identifier: MIT OR Apache-2.0

//! Config errors and warnings in RFC-0054 §8's format, with the source line and a caret:
//!
//! ```text
//! error[config/0003]: unknown key `resource.ram_headroon`
//!   --> /home/u/.config/syntropy/config.toml:14:1
//!    |
//! 14 | ram_headroon = "16GiB"
//!    | ^^^^^^^^^^^^ did you mean `ram_headroom`?
//! ```
//!
//! **A secret's value is never printed** (RFC-0054 §3, §8 note of 2026-10-22, §10.1). The
//! renderer decides from the file's text which key a line sets, never from a parsed value,
//! because the likeliest leak is a phase-1 syntax error such as an unquoted `api_key = sk-…`,
//! which parses as nothing:
//! - a line that sets a secret-typed key shows `key = [redacted]`, with the caret on the key, so
//!   neither the value nor its length shows;
//! - a line inside a multi-line string or array that a secret-typed key opened is withheld whole;
//! - a line the renderer cannot attribute to a key is withheld whole too, when a secret-typed
//!   key could live in the table it sits in.
//!
//! A key is secret-typed when its type (`KeySchema::ty`) names `Secret`, which covers
//! `Secret<…>`, `SecretSource` and tables or arrays of them; every key under such a key is
//! secret too.

use std::fmt::Write as _;
use std::path::Path;

use crate::code::Code;
use crate::registry::Registry;
use crate::value::REDACTED;

/// Whether a report stops the load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    /// The configuration is rejected.
    Error,
    /// The configuration loads, and the report is printed.
    Warning,
}

impl Severity {
    fn word(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
        }
    }
}

/// Where a reported problem is.
#[derive(Clone, Copy, Debug)]
pub(crate) enum At<'a> {
    /// Nowhere in particular.
    Nowhere,
    /// A place in a file, with the file's text.
    File {
        path: &'a Path,
        text: &'a str,
        /// 1-based.
        line: u32,
        /// 1-based, in characters.
        column: u32,
        /// The caret's width, in characters; at least 1.
        width: u32,
    },
    /// Somewhere with no source line, such as `env SYNTROPY_LOGGING__LEVEL` or
    /// `flag --set logging.level`.
    Elsewhere(&'a str),
}

/// One error or warning to render.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Report<'a> {
    pub severity: Severity,
    /// The `Config` code, or `None` for a warning §8 gives no code.
    pub code: Option<Code>,
    /// What is wrong. Never a value.
    pub message: &'a str,
    pub at: At<'a>,
    /// A hint, such as "did you mean `ram_headroom`?".
    pub help: Option<&'a str>,
}

/// Renders `report` in §8's format, withholding any secret on its source line.
pub(crate) fn render(report: &Report<'_>, registry: &Registry) -> String {
    let mut out = String::new();
    let _ = match report.code {
        Some(code) => writeln!(
            out,
            "{}[{code}]: {}",
            report.severity.word(),
            report.message
        ),
        None => writeln!(out, "{}: {}", report.severity.word(), report.message),
    };
    match report.at {
        At::Nowhere => {
            if let Some(help) = report.help {
                let _ = writeln!(out, "  = help: {help}");
            }
        }
        At::Elsewhere(place) => {
            let _ = writeln!(out, "  --> {place}");
            if let Some(help) = report.help {
                let _ = writeln!(out, "  = help: {help}");
            }
        }
        At::File {
            path,
            text,
            line,
            column,
            width,
        } => {
            let shown = shown_line(registry, report.code, text, line, column, width);
            let gutter = line.to_string().len();
            let pad = " ".repeat(gutter);
            let _ = writeln!(out, "{pad}--> {}:{line}:{}", path.display(), shown.column);
            let _ = writeln!(out, "{pad} |");
            let _ = writeln!(out, "{line} | {}", shown.text);
            let caret = format!(
                "{}{}",
                " ".repeat(shown.column.saturating_sub(1) as usize),
                "^".repeat(shown.width.max(1) as usize)
            );
            match report.help {
                Some(help) => {
                    let _ = writeln!(out, "{pad} | {caret} {help}");
                }
                None => {
                    let _ = writeln!(out, "{pad} | {caret}");
                }
            }
        }
    }
    out
}

/// A source line as it is printed, and where the caret goes on it.
#[derive(Debug, PartialEq, Eq)]
struct Shown {
    text: String,
    column: u32,
    width: u32,
}

/// The line `line` of `text` as the renderer may show it, with the caret moved where needed.
///
/// The scan fails closed: wherever it cannot tell which key a line sets, or the line has more on
/// it than one key and value, the line is withheld if any secret-typed key could live there.
/// Comments are never printed, since a commented-out line may hold an old secret.
fn shown_line(
    registry: &Registry,
    code: Option<Code>,
    text: &str,
    line: u32,
    column: u32,
    width: u32,
) -> Shown {
    let raw = text
        .lines()
        .nth(line.saturating_sub(1) as usize)
        .unwrap_or("");
    let chars: Vec<char> = raw.chars().collect();
    // The line up to any comment, lexed from `state` and `depth`.
    let uncommented = |state: Lex, depth: i64| -> String {
        let end = lex(&chars, state, depth).comment.unwrap_or(chars.len());
        let text: String = chars[..end].iter().collect();
        text.trim_end().to_owned()
    };
    let shown = |text: String| Shown {
        text: expand_tabs(&text),
        column: display_column(raw, column),
        width: width.max(1),
    };
    let withheld = || Shown {
        text: format!("{REDACTED} (this line may hold a secret, so it is not shown)"),
        column: 1,
        width: REDACTED.len() as u32,
    };
    let could_hold = |table: &Option<String>| match table {
        Some(table) => could_hold_secret(registry, table),
        None => any_secret(registry),
    };
    // A key name is fixed without its value, and a mistyped secret key is the likeliest way a
    // secret reaches an error, so an unknown-key error never prints a value.
    let unknown_key = code == Some(Code::UnknownKey);
    match owner(text, line) {
        Owner::Blank => shown(String::new()),
        Owner::Header => shown(uncommented(Lex::Code, 0)),
        Owner::Key {
            path,
            key_column,
            key_width,
            value_column,
        } => {
            let value = &chars[(value_column as usize - 1).min(chars.len())..];
            let redact = unknown_key
                || could_hold_secret(registry, &path)
                || (has_trailer(value) && any_secret(registry));
            if !redact {
                return shown(uncommented(Lex::Code, 0));
            }
            // The key stays; everything from the value on is withheld, so the caret marks the key.
            let head: String = chars[..(value_column as usize - 1).min(chars.len())]
                .iter()
                .collect();
            Shown {
                text: format!("{}{REDACTED}", expand_tabs(&head)),
                column: display_column(raw, key_column),
                width: key_width.max(1),
            }
        }
        Owner::Continuation { path, table, open } => {
            // As a backstop, in case the scan misjudged where a value ends: the line is lexed both
            // from the state the scan carried and as if it were code, and it is withheld where a
            // secret could live if either finds an `=` outside strings, at any depth, or if it
            // reads as `key = value` for a key that could hold a secret.
            let equals = lex(&chars, open.state, open.depth).equals_any.is_some()
                || lex(&chars, Lex::Code, 0).equals_any.is_some();
            let looks_secret = key_value(raw).is_some_and(|kv| {
                could_hold_secret(registry, &kv.key)
                    || table
                        .as_deref()
                        .is_none_or(|t| could_hold_secret(registry, &join(t, &kv.key)))
            });
            if unknown_key
                || could_hold_secret(registry, &path)
                || looks_secret
                || (equals && could_hold(&table))
            {
                withheld()
            } else {
                shown(uncommented(open.state, open.depth))
            }
        }
        Owner::Unknown { table } => {
            if unknown_key || could_hold(&table) {
                withheld()
            } else {
                shown(uncommented(Lex::Code, 0))
            }
        }
    }
}

/// The column and width of the key that line `line` of `text` sets, for a caret when the
/// problem gives none; `None` if the line sets no key.
pub(crate) fn key_span(text: &str, line: u32) -> Option<(u32, u32)> {
    match owner(text, line) {
        Owner::Key {
            key_column,
            key_width,
            ..
        } => Some((key_column, key_width.max(1))),
        _ => None,
    }
}

/// Which key a line belongs to, as far as the file's text tells.
#[derive(Debug, PartialEq, Eq)]
enum Owner {
    /// An empty line.
    Blank,
    /// A `[table]` or `[[array]]` header.
    Header,
    /// A `key = value` line for the full path `path`. Columns are 1-based, in characters.
    Key {
        path: String,
        key_column: u32,
        key_width: u32,
        value_column: u32,
    },
    /// A line inside a multi-line string, array or inline table that `path` opened, in table
    /// `table`, starting in lexer state `open`.
    Continuation {
        path: String,
        table: Option<String>,
        open: Open,
    },
    /// Anything else, such as a malformed line or a comment, in table `table`; `None` when an
    /// earlier malformed header or an unreadable value left the table unknown.
    Unknown { table: Option<String> },
}

/// Lexer states that can span a line end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lex {
    /// Outside any string.
    Code,
    /// Inside a one-line basic string; it cannot span a line.
    Basic,
    /// Inside a one-line literal string; it cannot span a line.
    Literal,
    /// Inside a multi-line basic string.
    MultiBasic,
    /// Inside a multi-line literal string.
    MultiLiteral,
}

/// Where a lexed fragment ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Lexed {
    state: Lex,
    /// Bracket depth: `[` and `{` open, `]` and `}` close.
    depth: i64,
    /// The character index of the first `=` met outside strings at depth 0.
    equals: Option<usize>,
    /// The character index of the first `]` met outside strings at depth 0.
    close: Option<usize>,
    /// The character index of the first `=` met outside strings at any depth.
    equals_any: Option<usize>,
    /// The index just past the first string or bracketed value: where the lexer is first back
    /// outside strings at depth 0 after leaving it.
    settled: Option<usize>,
    /// The character index of the `#` that starts a comment.
    comment: Option<usize>,
}

/// Lexes `chars` from `state` and `depth`, honouring TOML's quoting and backslash escapes, and
/// stopping at a comment.
fn lex(chars: &[char], state: Lex, depth: i64) -> Lexed {
    let mut out = Lexed {
        state,
        depth,
        equals: None,
        close: None,
        equals_any: None,
        settled: None,
        comment: None,
    };
    let triple = |i: usize, q: char| {
        chars
            .get(i..i + 3)
            .is_some_and(|w| w.iter().all(|&c| c == q))
    };
    // Whether the lexer has been inside a string or bracket since it started.
    let mut left = false;
    let mut i = 0;
    while i < chars.len() {
        if out.state != Lex::Code || out.depth != 0 {
            left = true;
        } else if left && out.settled.is_none() {
            out.settled = Some(i);
        }
        let c = chars[i];
        match out.state {
            Lex::Code => match c {
                '#' => {
                    out.comment = Some(i);
                    break;
                }
                '"' if triple(i, '"') => {
                    out.state = Lex::MultiBasic;
                    i += 3;
                    continue;
                }
                '\'' if triple(i, '\'') => {
                    out.state = Lex::MultiLiteral;
                    i += 3;
                    continue;
                }
                '"' => out.state = Lex::Basic,
                '\'' => out.state = Lex::Literal,
                '[' | '{' => out.depth += 1,
                ']' | '}' => {
                    if c == ']' && out.depth == 0 && out.close.is_none() {
                        out.close = Some(i);
                    }
                    out.depth -= 1;
                }
                '=' => {
                    if out.depth == 0 && out.equals.is_none() {
                        out.equals = Some(i);
                    }
                    if out.equals_any.is_none() {
                        out.equals_any = Some(i);
                    }
                }
                _ => {}
            },
            Lex::Basic => match c {
                '\\' => i += 1,
                '"' => out.state = Lex::Code,
                _ => {}
            },
            Lex::Literal => {
                if c == '\'' {
                    out.state = Lex::Code;
                }
            }
            Lex::MultiBasic => {
                if c == '\\' {
                    i += 2;
                    continue;
                }
                if triple(i, '"') {
                    // Up to two more quotes belong to the content: four quotes end with one.
                    i += 3;
                    while i < chars.len() && chars[i] == '"' {
                        i += 1;
                    }
                    out.state = Lex::Code;
                    continue;
                }
            }
            Lex::MultiLiteral => {
                if triple(i, '\'') {
                    i += 3;
                    while i < chars.len() && chars[i] == '\'' {
                        i += 1;
                    }
                    out.state = Lex::Code;
                    continue;
                }
            }
        }
        i += 1;
    }
    if left && out.settled.is_none() && out.state == Lex::Code && out.depth == 0 {
        out.settled = Some(out.comment.unwrap_or(chars.len()));
    }
    out
}

/// Whether a value that ends on its line is followed by more than whitespace and a comment, as
/// in the malformed `endpoint = "x" authorization = "tok"`. A value still open at the line's end
/// has no trailer here: the next lines continue it.
fn has_trailer(value: &[char]) -> bool {
    let end = match value.first() {
        None => return false,
        Some('"' | '\'' | '[' | '{') => match lex(value, Lex::Code, 0).settled {
            Some(end) => end,
            None => return false,
        },
        Some(_) => value
            .iter()
            .position(|c| c.is_whitespace() || *c == '#')
            .unwrap_or(value.len()),
    };
    let rest: String = value[end..].iter().collect();
    let rest = rest.trim_start();
    !(rest.is_empty() || rest.starts_with('#'))
}

/// What a value still open at a line's end needs from the next lines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Open {
    state: Lex,
    depth: i64,
}

impl Open {
    /// The value's state after `chars`: still open, closed (`Ok(None)`), or unreadable (`Err`).
    fn after(state: Lex, depth: i64, chars: &[char]) -> Result<Option<Open>, ()> {
        let lexed = lex(chars, state, depth);
        match lexed.state {
            // A one-line string still open at the line's end is not TOML.
            Lex::Basic | Lex::Literal => Err(()),
            _ if lexed.depth < 0 => Err(()),
            Lex::Code if lexed.depth == 0 => Ok(None),
            state => Ok(Some(Open {
                state,
                depth: lexed.depth,
            })),
        }
    }
}

/// Attributes line `line` (1-based) of `text` to a key by scanning the file from the top.
fn owner(text: &str, line: u32) -> Owner {
    // The current table, `None` once a malformed header or unreadable value makes it unknown.
    let mut table: Option<String> = Some(String::new());
    // The key whose value is open across lines, and the lexer state it is in.
    let mut open: Option<(String, Open)> = None;
    for (index, raw) in text.lines().enumerate() {
        let current = index as u32 + 1 == line;
        let chars: Vec<char> = raw.chars().collect();
        if let Some((path, state)) = open.take() {
            if current {
                return Owner::Continuation {
                    path,
                    table,
                    open: state,
                };
            }
            match Open::after(state.state, state.depth, &chars) {
                Ok(still) => open = still.map(|s| (path, s)),
                Err(()) => table = None,
            }
            continue;
        }
        let trimmed = raw.trim_start();
        if trimmed.is_empty() {
            if current {
                return Owner::Blank;
            }
            continue;
        }
        if trimmed.starts_with('#') {
            if current {
                return Owner::Unknown { table };
            }
            continue;
        }
        if trimmed.starts_with('[') {
            let header = header_path(trimmed);
            if current {
                return match header {
                    Some(_) => Owner::Header,
                    None => Owner::Unknown { table: None },
                };
            }
            table = header;
            continue;
        }
        let parsed = key_value(raw);
        if current {
            return match (parsed, &table) {
                (Some(kv), Some(t)) => Owner::Key {
                    path: join(t, &kv.key),
                    key_column: kv.key_column,
                    key_width: kv.key_width,
                    value_column: kv.value_column,
                },
                _ => Owner::Unknown { table },
            };
        }
        match (parsed, &table) {
            (Some(kv), Some(t)) => {
                let value = &chars[kv.value_column as usize - 1..];
                match Open::after(Lex::Code, 0, value) {
                    Ok(still) => open = still.map(|s| (join(t, &kv.key), s)),
                    Err(()) => table = None,
                }
            }
            // A line that is neither a key nor a header leaves the scan unsure of what follows.
            (None, _) => table = None,
            (Some(_), None) => {}
        }
    }
    Owner::Unknown { table }
}

/// `table.key`, or `key` at the top level.
fn join(table: &str, key: &str) -> String {
    if table.is_empty() {
        key.to_owned()
    } else {
        format!("{table}.{key}")
    }
}

/// The dotted path of a `[a.b]` or `[[a.b]]` header, with quoted segments unquoted; `None` if
/// it is not a well-formed header. The closing bracket is found outside quotes.
fn header_path(trimmed: &str) -> Option<String> {
    let chars: Vec<char> = trimmed.chars().collect();
    let array = chars.get(1) == Some(&'[');
    let inner = &chars[if array { 2 } else { 1 }..];
    let close = lex(inner, Lex::Code, 0).close?;
    let keys: String = inner[..close].iter().collect();
    let mut rest = &inner[close + 1..];
    if array {
        rest = rest.strip_prefix(&[']'][..])?;
    }
    let after: String = rest.iter().collect();
    let after = after.trim();
    if !(after.is_empty() || after.starts_with('#')) {
        return None;
    }
    dotted_key(&keys)
}

/// A parsed `key = value` line.
struct KeyValue {
    /// The dotted key, quoted segments unquoted.
    key: String,
    key_column: u32,
    key_width: u32,
    /// Where the value starts, 1-based in characters.
    value_column: u32,
}

/// Splits a `key = value` line at the first `=` outside quotes.
fn key_value(raw: &str) -> Option<KeyValue> {
    let chars: Vec<char> = raw.chars().collect();
    let start = chars.iter().position(|c| !c.is_whitespace())?;
    let eq = start + lex(&chars[start..], Lex::Code, 0).equals?;
    let key_text: String = chars[start..eq].iter().collect();
    let key_text = key_text.trim_end();
    let key = dotted_key(key_text)?;
    let mut value = eq + 1;
    while value < chars.len() && chars[value].is_whitespace() {
        value += 1;
    }
    Some(KeyValue {
        key,
        key_column: start as u32 + 1,
        key_width: key_text.chars().count() as u32,
        value_column: value as u32 + 1,
    })
}

/// Stands for a `.` inside a quoted key segment, so a path splits back into its segments: a
/// selector such as `"qwen3.5*"` stays one segment. No registered key holds it.
const QUOTED_DOT: char = '\u{1f}';

/// A dotted key, `a.b`, `a."b.c"` or `'a'.b`, as one path with quotes removed and basic-string
/// escapes decoded, each segment's own dots written as [`QUOTED_DOT`]. `None` if it is not a key.
fn dotted_key(text: &str) -> Option<String> {
    let chars: Vec<char> = text.trim().chars().collect();
    let mut segments = Vec::new();
    let mut i = 0;
    loop {
        while i < chars.len() && chars[i].is_whitespace() {
            i += 1;
        }
        let mut segment = String::new();
        match chars.get(i)? {
            '"' => {
                i += 1;
                loop {
                    match *chars.get(i)? {
                        '"' => break,
                        '\\' => {
                            let (decoded, used) = unescape(&chars[i + 1..])?;
                            segment.push(decoded);
                            i += used;
                        }
                        c => segment.push(c),
                    }
                    i += 1;
                }
                i += 1;
            }
            '\'' => {
                i += 1;
                loop {
                    match *chars.get(i)? {
                        '\'' => break,
                        c => segment.push(c),
                    }
                    i += 1;
                }
                i += 1;
            }
            _ => {
                while let Some(&c) = chars.get(i) {
                    if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                        segment.push(c);
                        i += 1;
                    } else {
                        break;
                    }
                }
                if segment.is_empty() {
                    return None;
                }
            }
        }
        segments.push(segment.replace('.', &QUOTED_DOT.to_string()));
        while i < chars.len() && chars[i].is_whitespace() {
            i += 1;
        }
        match chars.get(i) {
            None => return Some(segments.join(".")),
            Some('.') => i += 1,
            Some(_) => return None,
        }
    }
}

/// Decodes the escape after a backslash in a TOML basic string: `\b \t \n \f \r \" \\ \e`,
/// `\xHH`, `\uXXXX` and `\UXXXXXXXX`. Returns the character and how many characters the escape
/// took after the backslash; `None` for anything else, which leaves the key unknown, so the scan
/// fails closed.
fn unescape(after: &[char]) -> Option<(char, usize)> {
    let hex = |len: usize| -> Option<(char, usize)> {
        let digits: String = after.get(1..1 + len)?.iter().collect();
        if !digits.chars().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }
        let code = u32::from_str_radix(&digits, 16).ok()?;
        Some((char::from_u32(code)?, 1 + len))
    };
    Some(match after.first()? {
        'b' => ('\u{8}', 1),
        't' => ('\t', 1),
        'n' => ('\n', 1),
        'f' => ('\u{c}', 1),
        'r' => ('\r', 1),
        '"' => ('"', 1),
        '\\' => ('\\', 1),
        'e' => ('\u{1b}', 1),
        'x' => return hex(2),
        'u' => return hex(4),
        'U' => return hex(8),
        _ => return None,
    })
}

/// Whether the key's type holds a secret anywhere inside it: [`crate::KeySchema::holds_secret`].
fn secret_typed(key: &crate::schema::KeySchema) -> bool {
    key.holds_secret
}

/// Whether any key of the registry is secret-typed.
fn any_secret(registry: &Registry) -> bool {
    registry.keys().any(secret_typed)
}

/// Whether a secret-typed key sits at `path`, under it or above it, so a value set at `path`
/// could hold a secret: the key itself, a table that contains it (an inline table such as
/// `obs = { otlp = { headers = … } }`), or an entry of it. An empty path is the top level.
fn could_hold_secret(registry: &Registry, path: &str) -> bool {
    let within = |inner: &str, outer: &str| {
        outer.is_empty()
            || inner == outer
            || inner
                .strip_prefix(outer)
                .is_some_and(|rest| rest.starts_with('.'))
    };
    let near = |held: &str, at: &str| within(held, at) || within(at, held);
    // A line under `policy` sets keys by their path relative to a `[policy."<selector>"]` table.
    // A line that sets `policy` or a selector's table whole has the empty relative path, so it
    // is withheld wherever any key holds a secret. A line deeper in a table is matched by its
    // exact relative path against every key that holds a secret, of any scope, since a node key
    // misplaced in a policy table is reported there too.
    let segments: Vec<&str> = path.split('.').collect();
    let policy = segments.first() == Some(&"policy");
    // A quoted segment holding dots names no key: only a policy selector may hold one. Anywhere
    // else, a reader that joins segments with `.` could take it for a secret's path, as
    // `"node.token"` or `"policy.x".token` spell one, so the line is withheld as the top level
    // is.
    let stray = segments
        .iter()
        .enumerate()
        .any(|(i, segment)| segment.contains(QUOTED_DOT) && !(policy && i == 1));
    let path = if stray { "" } else { path };
    let relative: Option<String> = (policy && !stray).then(|| {
        segments
            .get(2..)
            .map(|rest| rest.join("."))
            .unwrap_or_default()
    });
    let hit = |held: &str| {
        near(held, path)
            || relative
                .as_deref()
                .is_some_and(|relative| near(held, relative))
    };
    // A deprecated key has no schema of its own: it may be a secret's old name, and its value is
    // not needed to fix a rename, so every deprecated path counts as holding a secret.
    registry
        .keys()
        .filter(|key| secret_typed(key))
        .any(|key| hit(key.path))
        || registry.deprecations().any(|d| hit(d.path))
}

/// A line with each tab shown as four spaces.
fn expand_tabs(text: &str) -> String {
    text.replace('\t', "    ")
}

/// A 1-based character column, moved for the tabs before it.
fn display_column(raw: &str, column: u32) -> u32 {
    let before = raw.chars().take(column.saturating_sub(1) as usize);
    before.map(|c| if c == '\t' { 4 } else { 1 }).sum::<u32>() + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attributes_lines_to_keys() {
        let text = "\
config_version = 1
[api]
bind = [\"0.0.0.0:7433\"]
[obs.otlp.headers]
authorization = \"Bearer x\"
'x.y' = 1
[[policy.\"deepseek-v3*\".pin]]
layers = \"0..20\"
multi = \"\"\"
line one
\"\"\"
list = [
  \"a\",
]
garbage line
";
        let key = |line| match owner(text, line) {
            Owner::Key { path, .. } => path,
            other => panic!("line {line}: {other:?}"),
        };
        assert_eq!(key(1), "config_version");
        assert_eq!(owner(text, 2), Owner::Header);
        assert_eq!(key(3), "api.bind");
        assert_eq!(key(5), "obs.otlp.headers.authorization");
        // A quoted segment keeps its dot, as one segment.
        assert_eq!(key(6), "obs.otlp.headers.x\u{1f}y");
        assert_eq!(key(8), "policy.deepseek-v3*.pin.layers");
        assert_eq!(key(9), "policy.deepseek-v3*.pin.multi");
        let pin = Some("policy.deepseek-v3*.pin".to_owned());
        assert!(matches!(
            owner(text, 10),
            Owner::Continuation { path, table, open }
                if path == "policy.deepseek-v3*.pin.multi"
                    && table == pin
                    && open.state == Lex::MultiBasic
        ));
        assert!(matches!(
            owner(text, 13),
            Owner::Continuation { path, table, open }
                if path == "policy.deepseek-v3*.pin.list" && table == pin && open.depth == 1
        ));
        assert_eq!(
            owner(text, 15),
            Owner::Unknown {
                table: Some("policy.deepseek-v3*.pin".into())
            }
        );
        // A malformed header leaves the table unknown for every line after it.
        let text = "[resource]\n[api]x\nbind = 1\n";
        assert_eq!(owner(text, 2), Owner::Unknown { table: None });
        assert_eq!(owner(text, 3), Owner::Unknown { table: None });
        // A `]` inside a quoted header key does not end the header.
        let text = "[policy.\"a]b\"]\nx = 1\n";
        assert_eq!(owner(text, 1), Owner::Header);
        assert!(matches!(owner(text, 2), Owner::Key { path, .. } if path == "policy.a]b.x"));
    }

    /// A secret below a key's path is withheld wherever that key sets it.
    #[test]
    fn a_table_holding_a_secret_is_withheld() {
        let cases = [
            // An inline table at the top level that holds `obs.otlp.headers`.
            (
                "obs = { otlp = { headers = { authorization = \"tok-1101\" } } }\n",
                1,
            ),
            // The same, from inside `[obs]`.
            (
                "[obs]\notlp = { headers = { authorization = \"tok-1101\" } }\n",
                2,
            ),
            // An inline table spread over lines (TOML 1.1): the second line continues `obs`.
            (
                "obs = { otlp = {\n  headers = { authorization = \"tok-1101\" } } }\n",
                2,
            ),
        ];
        for (text, line) in cases {
            let out = error(Code::WrongType, "bad", file(text, line, 1, 1), None);
            assert!(!out.contains("tok-1101"), "{text:?}:\n{out}");
            assert!(out.contains(REDACTED), "{text:?}:\n{out}");
        }
    }

    /// Escaped quotes do not end a string early, so later lines keep their owners.
    #[test]
    fn escapes_do_not_misplace_the_scan() {
        // `\"]` inside a string is not the array's end; line 2 still continues `note`.
        let text = "note = [\"x\\\"]\",\n  \"y\"]\napi_key = sk-x-7781\n";
        assert!(matches!(owner(text, 2), Owner::Continuation { path, .. } if path == "note"));
        let out = error(Code::Syntax, "bad", file(text, 3, 11, 9), None);
        assert!(out.contains("3 | api_key = [redacted]"), "{out}");
        assert!(!out.contains("7781"), "{out}");
        // `\"""` inside a multi-line basic string does not close it: line 2 is still the
        // secret's content.
        let text = "api_key = \"\"\"a\\\"\"\"\nsecret-3301\n\"\"\"\n";
        assert!(matches!(owner(text, 2), Owner::Continuation { path, .. } if path == "api_key"));
        let out = error(Code::WrongType, "bad", file(text, 2, 1, 6), None);
        assert!(!out.contains("3301"), "{out}");
    }

    /// A malformed header fails closed: its line, and the lines after it, are withheld.
    #[test]
    fn a_malformed_header_fails_closed() {
        let text = "[resource]\n[obs.otlp.headers] authorization = tok-4401\n";
        let out = error(Code::Syntax, "bad", file(text, 2, 20, 1), None);
        assert!(!out.contains("tok-4401"), "{out}");
        let text = "[resource]\n[obs.otlp.headers]x\nauthorization = tok-4402\n";
        let out = error(Code::Syntax, "bad", file(text, 3, 17, 8), None);
        assert!(!out.contains("tok-4402"), "{out}");
    }

    use std::collections::BTreeMap;

    use serde::Deserialize;

    use syntropy_core::policy::Reserve;

    use crate::ConfigSection;
    use crate::value::SecretSource;

    /// API settings with a secret.
    #[derive(Deserialize, crate::ConfigSection)]
    #[config(section = "", scope = Node, owner = "RFC-0050")]
    #[allow(dead_code)]
    struct Top {
        /// A secret at the top level.
        #[config(reload = Hot, default = unset)]
        api_key: Option<SecretSource>,
    }

    /// Exporter headers, each a secret.
    #[derive(Deserialize, crate::ConfigSection)]
    #[config(section = "obs.otlp", scope = Node, owner = "RFC-0053")]
    #[allow(dead_code)]
    struct Otlp {
        /// Headers, each a secret.
        #[config(reload = Hot, default = {})]
        headers: BTreeMap<String, SecretSource>,
    }

    /// Resource settings, with no secret.
    #[derive(Deserialize, crate::ConfigSection)]
    #[config(section = "resource", scope = Node, owner = "RFC-0020")]
    #[allow(dead_code)]
    struct Resource {
        /// RAM kept back.
        #[config(reload = Hot, default = "6GiB")]
        #[serde(with = "crate::form")]
        ram_headroom: Reserve,
    }

    static REGISTRY: Registry = Registry::new(&[&Top::SCHEMA, &Otlp::SCHEMA, &Resource::SCHEMA]);

    fn file(text: &str, line: u32, column: u32, width: u32) -> At<'_> {
        At::File {
            path: Path::new("/home/u/.config/syntropy/config.toml"),
            text,
            line,
            column,
            width,
        }
    }

    fn error<'a>(code: Code, message: &'a str, at: At<'a>, help: Option<&'a str>) -> String {
        let report = Report {
            severity: Severity::Error,
            code: Some(code),
            message,
            at,
            help,
        };
        render(&report, &REGISTRY)
    }

    #[test]
    fn renders_the_rfc_example() {
        let text = "\n\n\n\n\n\n\n\n\n\n\n\n[resource]\nram_headroon = \"16GiB\"\n";
        let out = error(
            Code::UnknownKey,
            "unknown key `resource.ram_headroon`",
            file(text, 14, 1, 12),
            Some("did you mean `ram_headroom`?"),
        );
        assert_eq!(
            out,
            "\
error[config/0003]: unknown key `resource.ram_headroon`
  --> /home/u/.config/syntropy/config.toml:14:1
   |
14 | ram_headroon = [redacted]
   | ^^^^^^^^^^^^ did you mean `ram_headroom`?
"
        );
    }

    /// A key written with escapes is the same key, so a secret under it is still withheld.
    #[test]
    fn an_escaped_key_is_still_the_secret_key() {
        let text = "[obs.otlp]\n\"\\u0068eaders\" = { authorization = \"tok-6601\" }\n";
        assert!(matches!(owner(text, 2), Owner::Key { path, .. } if path == "obs.otlp.headers"));
        let out = error(Code::WrongType, "bad", file(text, 2, 1, 1), None);
        assert!(!out.contains("tok-6601"), "{out}");
        // An escape the scan cannot decode leaves the key unknown, so the line is withheld.
        let text = "[obs.otlp]\n\"\\q\" = \"tok-6602\"\n";
        let out = error(Code::WrongType, "bad", file(text, 2, 1, 1), None);
        assert!(!out.contains("tok-6602"), "{out}");
    }

    /// A quoted key holding dots, anywhere but a policy selector, is withheld wherever a key
    /// holds a secret: a reader that joins segments with `.` could take it for a secret's path.
    /// A selector's own dots leave a policy line shown.
    #[test]
    fn a_quoted_key_that_spells_a_secret_path_is_withheld() {
        for text in [
            "\"obs.otlp.headers\" = { authorization = \"tok-8801\" }\n",
            "[obs]\n'otlp.headers' = { authorization = \"tok-8801\" }\n",
            "[policy.x]\n\"obs.otlp.headers\" = { authorization = \"tok-8801\" }\n",
            "\"policy.x\".api_key = \"tok-8801\"\n",
            "[\"policy.x\"]\napi_key = \"tok-8801\"\n",
            "[policy.\"a.b\".\"c.d\"]\nx = \"tok-8801\"\n",
        ] {
            let line = text.lines().count() as u32;
            let out = error(Code::ScopeMismatch, "bad", file(text, line, 1, 1), None);
            assert!(!out.contains("tok-8801"), "{out}");
        }
        let text = "[policy.\"qwen3.5*\"]\nobjective = \"fast\"\n";
        let out = error(Code::WrongType, "bad", file(text, 2, 13, 6), None);
        assert!(out.contains("2 | objective = \"fast\""), "{out}");
    }

    /// A continuation line that reads as a secret's `key = value` is withheld even when the
    /// scan puts it inside a string, in case the scan misjudged where the string ends.
    #[test]
    fn a_continuation_that_reads_as_a_secret_is_withheld() {
        let text =
            "[obs.otlp]\nnote = \"\"\"\nheaders = { authorization = \"tok-9901\" }\n\"\"\"\n";
        assert!(
            matches!(owner(text, 3), Owner::Continuation { path, .. } if path == "obs.otlp.note")
        );
        let out = error(Code::WrongType, "bad", file(text, 3, 1, 1), None);
        assert!(!out.contains("tok-9901"), "{out}");
    }

    /// A deprecated key's value is withheld: it may be a secret's old name.
    #[test]
    fn a_deprecated_key_is_withheld() {
        static OLD: [crate::Deprecated; 1] = [crate::Deprecated {
            path: "old_api_key",
            replacement: Some("api_key"),
            since: "0.3",
        }];
        /// The old name of the secret.
        #[derive(Deserialize, crate::ConfigSection)]
        #[config(section = "", scope = Node, owner = "RFC-0050", deprecated = &OLD)]
        #[allow(dead_code)]
        struct Renamed {
            /// Unrelated.
            #[config(reload = Hot, default = 1)]
            other: u32,
        }
        static WITH_OLD: Registry = Registry::new(&[&Top::SCHEMA, &Renamed::SCHEMA]);
        let text = "old_api_key = \"tok-5151\"\n";
        let report = Report {
            severity: Severity::Warning,
            code: None,
            message: "deprecated",
            at: file(text, 1, 1, 11),
            help: None,
        };
        let out = render(&report, &WITH_OLD);
        assert!(!out.contains("tok-5151"), "{out}");
        assert!(out.contains("1 | old_api_key = [redacted]"), "{out}");
    }

    /// A policy table set whole is withheld wherever a key holds a secret, a selector's own dots
    /// included; a line deeper in a table is matched by its exact relative path, so a policy line
    /// that holds no secret is still shown.
    #[test]
    fn a_policy_table_set_whole_is_withheld() {
        /// An instance key that holds a secret.
        #[derive(Deserialize, crate::ConfigSection)]
        #[config(section = "", scope = Instance, owner = "RFC-0054")]
        #[allow(dead_code)]
        struct InstanceSecret {
            /// A secret per model.
            #[config(reload = Instance, default = unset)]
            model_token: Option<SecretSource>,
        }
        static WITH_INSTANCE: Registry =
            Registry::new(&[&InstanceSecret::SCHEMA, &Resource::SCHEMA]);
        let text = "[policy]\n\"qwen3.5*\" = { model_token = \"tok-3131\" }\n";
        let report = Report {
            severity: Severity::Error,
            code: Some(Code::WrongType),
            message: "bad",
            at: file(text, 2, 1, 1),
            help: None,
        };
        let out = render(&report, &WITH_INSTANCE);
        assert!(!out.contains("tok-3131"), "{out}");
        // With only a node secret registered, a policy line's value is shown.
        let text = "[policy.\"qwen3*\"]\nobjective = \"fast\"\n";
        let out = error(Code::WrongType, "bad", file(text, 2, 13, 6), None);
        assert!(out.contains("2 | objective = \"fast\""), "{out}");
    }

    /// More than one key and value on a line fails closed.
    #[test]
    fn content_after_a_value_is_withheld() {
        let text = "[obs.otlp]\nendpoint = \"x\" headers.authorization = \"tok-7701\"\n";
        let out = error(Code::Syntax, "expected newline", file(text, 2, 16, 1), None);
        assert!(!out.contains("tok-7701"), "{out}");
        assert!(out.contains("2 | endpoint = [redacted]"), "{out}");
    }

    /// Comments are never printed: a commented-out line may hold an old secret.
    #[test]
    fn comments_are_not_printed() {
        let text = "[resource]\nram_headroom = \"16XB\" # was api_key = tok-8801\n";
        let out = error(Code::BadUnit, "bad", file(text, 2, 16, 6), None);
        assert!(out.contains("2 | ram_headroom = \"16XB\"\n"), "{out}");
        assert!(!out.contains("tok-8801"), "{out}");
    }

    #[test]
    fn a_phase_1_error_on_a_secret_never_prints_its_value() {
        // An unquoted token is a syntax error before anything parses; the caret would point
        // into the value.
        let text = "config_version = 1\napi_key = sk-live-4417\n";
        let out = error(
            Code::Syntax,
            "string values must be quoted",
            file(text, 2, 11, 12),
            None,
        );
        assert_eq!(
            out,
            "\
error[config/0001]: string values must be quoted
 --> /home/u/.config/syntropy/config.toml:2:1
  |
2 | api_key = [redacted]
  | ^^^^^^^
"
        );
        assert!(!out.contains("sk-live") && !out.contains("4417"));
    }

    #[test]
    fn a_secret_inside_a_secret_table_is_withheld() {
        let text = "[obs.otlp.headers]\nauthorization = Bearer tok-9931\n";
        let out = error(Code::Syntax, "invalid string", file(text, 2, 17, 6), None);
        assert!(out.contains("2 | authorization = [redacted]"), "{out}");
        assert!(
            !out.contains("tok-9931") && !out.contains("Bearer"),
            "{out}"
        );
    }

    #[test]
    fn a_continuation_of_a_secret_is_withheld() {
        let text = "api_key = \"\"\"\nsecret-multi-2231\n\"\"\"\n";
        let out = error(
            Code::WrongType,
            "a secret must be a string",
            file(text, 2, 1, 6),
            None,
        );
        assert!(out.contains("2 | [redacted]"), "{out}");
        assert!(
            !out.contains("secret-multi") && !out.contains("2231"),
            "{out}"
        );
    }

    #[test]
    fn an_unattributable_line_is_withheld_where_a_secret_could_live() {
        let text = "[obs.otlp.headers]\nauthorization Bearer tok-5521\n";
        let out = error(Code::Syntax, "expected `=`", file(text, 2, 15, 1), None);
        assert!(out.contains("2 | [redacted]"), "{out}");
        assert!(!out.contains("tok-5521"), "{out}");

        // No secret can live in `[resource]`, so the line is shown.
        let text = "[resource]\nram_headroom \"16GiB\"\n";
        let out = error(Code::Syntax, "expected `=`", file(text, 2, 14, 1), None);
        assert!(out.contains("2 | ram_headroom \"16GiB\""), "{out}");
        assert!(out.contains("  |              ^"), "{out}");
    }

    #[test]
    fn a_non_secret_value_is_shown() {
        let text = "[resource]\nram_headroom = \"16XB\"\n";
        let out = error(
            Code::BadUnit,
            "`16XB` is not a size",
            file(text, 2, 16, 6),
            None,
        );
        assert!(out.contains("2 | ram_headroom = \"16XB\""), "{out}");
        assert!(out.contains("  |                ^^^^^^"), "{out}");
    }

    #[test]
    fn renders_other_places_and_warnings() {
        let out = error(
            Code::BadUnit,
            "`16XB` is not a size",
            At::Elsewhere("env SYNTROPY_RESOURCE__RAM_HEADROOM"),
            Some("write a number and a unit, such as \"16GiB\""),
        );
        assert_eq!(
            out,
            "\
error[config/0005]: `16XB` is not a size
  --> env SYNTROPY_RESOURCE__RAM_HEADROOM
  = help: write a number and a unit, such as \"16GiB\"
"
        );
        let warning = Report {
            severity: Severity::Warning,
            code: Some(Code::UnknownKey),
            message: "unknown variable `SYNTROPY_LOGGIN__LEVEL`",
            at: At::Nowhere,
            help: None,
        };
        assert_eq!(
            render(&warning, &REGISTRY),
            "warning[config/0003]: unknown variable `SYNTROPY_LOGGIN__LEVEL`\n"
        );
    }

    #[test]
    fn tabs_keep_the_caret_aligned() {
        let text = "[resource]\n\tram_headroom = \"16XB\"\n";
        let out = error(Code::BadUnit, "bad", file(text, 2, 17, 6), None);
        assert!(out.contains("2 |     ram_headroom = \"16XB\""), "{out}");
        assert!(out.contains("  |                    ^^^^^^"), "{out}");
    }

    #[test]
    fn finds_the_value_column() {
        let kv = key_value("  api_key = sk-abc").unwrap();
        assert_eq!(kv.key, "api_key");
        assert_eq!((kv.key_column, kv.key_width, kv.value_column), (3, 7, 13));
        let kv = key_value(r#""a=b".c = 1"#).unwrap();
        assert_eq!(kv.key, "a=b.c");
        assert!(key_value("no equals here").is_none());
        assert!(key_value("bad key! = 1").is_none());
    }
}
