// SPDX-License-Identifier: MIT OR Apache-2.0

//! Writing a config file: `config set`, `unset` and `edit` (RFC-0054 §10.2; PROD-E1-T12).
//!
//! - [`set`] and [`unset`] edit a file's text with `toml_edit`, so comments, ordering, spacing and
//!   line endings survive and only the lines that hold the key change. A file whose lines end in
//!   `\r\n` keeps them. Each checks its own result twice, or the edit is refused: the text must
//!   be the old text byte for byte outside the key's own lines, and it must parse to the old file
//!   with exactly that key changed.
//! - [`check_edit`] validates the configuration layer as it would be with the edited text in
//!   place of one file: RFC-0054 §8's phases 1–5, before anything is written.
//! - [`write_atomic`] writes a temporary file beside the target, flushes it to disk, gives it the
//!   target's permissions and renames it over the target. It refuses a target that is a
//!   symbolic link, and the rename never follows one.
//!
//! No message here quotes a value, the file's or the new one: either can be a secret.
//!
//! Only a [`crate::Secret`] is zeroed when it is dropped (RFC-0054 §10.1). The other copies of
//! a secret are plain strings that are not: a file's text, the text an edit makes, `set`'s value,
//! and the loader's environment snapshot and flag values.

use std::fmt;
use std::fs;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};

use toml_edit::{DocumentMut, InlineTable, Item, Table, Value as EditValue};

use crate::discovery::{Os, Scan};
use crate::env::Env;
use crate::registry::Registry;
use crate::resolve::{ConfigFile, Flags, Layers, ResolvedConfig};
use crate::validate::{Checked, Environment, Rule, validate};

/// Why an edit could not be made. Its message names the key and the file, never a value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EditError(String);

impl EditError {
    fn new(message: impl Into<String>) -> EditError {
        EditError(message.into())
    }
}

impl fmt::Display for EditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EditError {}

/// The segments of a dotted key: bare keys (`A-Z a-z 0-9 _ -`) or double-quoted ones, such as
/// `policy."qwen3*".ram_headroom`. A quoted segment holds no `"` or `\`.
pub fn key_segments(path: &str) -> Result<Vec<String>, EditError> {
    let bad = || {
        EditError::new(format!(
            "`{path}` is not a key: write dotted bare keys, quoting a segment with other \
             characters, as `policy.\"qwen3*\".ram_headroom`"
        ))
    };
    let mut segments = Vec::new();
    let mut rest = path;
    loop {
        let (segment, after) = if let Some(quoted) = rest.strip_prefix('"') {
            let end = quoted.find('"').ok_or_else(bad)?;
            let segment = &quoted[..end];
            if segment.contains('\\') {
                return Err(bad());
            }
            (segment, &quoted[end + 1..])
        } else {
            let end = rest.find('.').unwrap_or(rest.len());
            let segment = &rest[..end];
            let bare = !segment.is_empty()
                && segment
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
            if !bare {
                return Err(bad());
            }
            (segment, &rest[end..])
        };
        segments.push(segment.to_owned());
        if after.is_empty() {
            return Ok(segments);
        }
        rest = after.strip_prefix('.').ok_or_else(bad)?;
    }
}

/// The value `text` gives on the command line: a TOML value when it is one (`8080`, `true`,
/// `"auto"`, `["a", "b"]`, `{ file = "/run/secrets/otlp" }`), and otherwise the text itself as a
/// string, so `auto` and `127.0.0.1:7499` need no quotes. A number meant as a string is quoted:
/// `'"8080"'`.
pub fn parse_value(text: &str) -> EditValue {
    let as_toml = format!("v = {text}")
        .parse::<DocumentMut>()
        .ok()
        .filter(|doc| doc.len() == 1)
        .and_then(|doc| doc.get("v").and_then(Item::as_value).cloned());
    let mut value = as_toml.unwrap_or_else(|| basic_string(text));
    value.decor_mut().clear();
    value
}

/// `text` as a one-line basic string, `"…"` with escapes, so a value with a newline still takes
/// one line of the file.
fn basic_string(text: &str) -> EditValue {
    let mut quoted = String::with_capacity(text.len() + 2);
    quoted.push('"');
    for c in text.chars() {
        match c {
            '"' => quoted.push_str("\\\""),
            '\\' => quoted.push_str("\\\\"),
            '\n' => quoted.push_str("\\n"),
            '\t' => quoted.push_str("\\t"),
            '\r' => quoted.push_str("\\r"),
            c if c.is_control() => quoted.push_str(&format!("\\u{:04X}", u32::from(c))),
            c => quoted.push(c),
        }
    }
    quoted.push('"');
    format!("v = {quoted}")
        .parse::<DocumentMut>()
        .ok()
        .and_then(|doc| doc.get("v").and_then(Item::as_value).cloned())
        .expect("an escaped basic string parses")
}

/// `text` with `path` set to `value` ([`parse_value`]). An existing value is replaced in place,
/// keeping its line's spacing and comment. A new key goes after the last key of the deepest table
/// the file already has for it, indented like that key; a key under no table of the file goes in
/// a new table at the end.
pub fn set(text: &str, path: &str, value: &str) -> Result<String, EditError> {
    let segments = key_segments(path)?;
    let ending = line_ending(text)?;
    let mut doc = parse_document(text)?;
    let new = parse_value(value);
    set_in_table(doc.as_table_mut(), &segments, new.clone(), path, true)?;
    let edited = ending.apply(&doc.to_string());
    let mut expected = toml_table(text)?;
    expected_set(&mut expected, &segments, toml_of(&new)?);
    confirm(text, &edited, &segments, &expected, path)?;
    Ok(edited)
}

/// `text` without `path`, or `None` when the file does not set it. The key's line goes; the
/// comment lines above it stay, on the next key or header.
pub fn unset(text: &str, path: &str) -> Result<Option<String>, EditError> {
    let segments = key_segments(path)?;
    let ending = line_ending(text)?;
    let mut doc = parse_document(text)?;
    let removed = unset_in_doc(&mut doc, &segments, path)?;
    if !removed {
        return Ok(None);
    }
    let edited = ending.apply(&doc.to_string());
    let mut expected = toml_table(text)?;
    expected_unset(&mut expected, &segments);
    confirm(text, &edited, &segments, &expected, path)?;
    Ok(Some(edited))
}

fn parse_document(text: &str) -> Result<DocumentMut, EditError> {
    // toml_edit's error quotes the line, which can hold a secret: it is not passed on.
    text.parse::<DocumentMut>().map_err(|_| {
        EditError::new(
            "the file is not valid TOML, so it cannot be edited; `syntropy config check` shows \
             where",
        )
    })
}

fn toml_table(text: &str) -> Result<toml::Table, EditError> {
    text.parse::<toml::Table>()
        .map_err(|_| EditError::new("the file is not valid TOML, so it cannot be edited"))
}

/// The plain TOML value of an edited one.
fn toml_of(value: &EditValue) -> Result<toml::Value, EditError> {
    format!("v = {value}")
        .parse::<toml::Table>()
        .ok()
        .and_then(|mut table| table.remove("v"))
        .ok_or_else(|| EditError::new("the new value is not a TOML value"))
}

/// How a file's lines end. `toml_edit` writes `\n`, so an edit puts the file's own ending back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LineEnding {
    Lf,
    CrLf,
}

impl LineEnding {
    /// `text`, written by `toml_edit`, with every line break this ending.
    fn apply(self, text: &str) -> String {
        let lf = text.replace("\r\n", "\n");
        match self {
            LineEnding::Lf => lf,
            LineEnding::CrLf => lf.replace('\n', "\r\n"),
        }
    }
}

/// The ending of `text`'s lines: `\r\n` when every line break is one, `\n` otherwise. A file that
/// mixes the two is refused, since an edit could not keep both.
fn line_ending(text: &str) -> Result<LineEnding, EditError> {
    let breaks = text.matches('\n').count();
    let crlf = text.matches("\r\n").count();
    match (breaks, crlf) {
        (_, 0) => Ok(LineEnding::Lf),
        (b, c) if b == c => Ok(LineEnding::CrLf),
        _ => Err(EditError::new(
            "the file mixes `\\r\\n` and `\\n` line endings, which an edit would not keep; edit it \
             by hand (`syntropy config edit`)",
        )),
    }
}

/// The lines of `text`, each with its line break, so a changed ending is a changed line.
fn lines(text: &str) -> Vec<&str> {
    text.split_inclusive('\n').collect()
}

/// The old file's lines an edit of `segments` may change: those of the key's value, from the line
/// it starts on, which is the key's own line, to the line it ends on. For a key the file does not
/// set, those of the inline table it goes in, if it goes in one. `None` when there are none, as
/// for a key added under a `[table]`, which only adds lines.
fn key_lines(text: &str, segments: &[String]) -> Option<(usize, usize)> {
    /// Where the walk is: an item of a table, or a value inside an inline table.
    #[derive(Clone, Copy)]
    enum At<'a> {
        Item(&'a Item),
        Value(&'a EditValue),
    }
    let doc = toml_edit::Document::parse(text).ok()?;
    let mut at = At::Item(doc.as_item());
    for segment in segments {
        let next = match at {
            At::Item(Item::Table(table)) => table.get(segment).map(At::Item),
            At::Item(Item::Value(value)) | At::Value(value) => value
                .as_inline_table()
                .and_then(|inline| inline.get(segment))
                .map(At::Value),
            At::Item(_) => None,
        };
        match next {
            Some(next) => at = next,
            // The key is not there: an inline table on its path is what changes.
            None => break,
        }
    }
    if let At::Item(Item::Table(_) | Item::ArrayOfTables(_) | Item::None) = at {
        return None;
    }
    let span = match at {
        At::Item(item) => item.span()?,
        At::Value(value) => value.span()?,
    };
    Some(line_range(text, span))
}

/// The first and last line a byte range of `text` touches, numbered from 0.
fn line_range(text: &str, span: std::ops::Range<usize>) -> (usize, usize) {
    let line_of = |at: usize| text[..at.min(text.len())].matches('\n').count();
    (
        line_of(span.start),
        line_of(span.end.saturating_sub(1).max(span.start)),
    )
}

/// The edit's own check, twice. First, `edited` is `old` byte for byte apart from one run of
/// lines, and every old line in that run is one of the key's own lines ([`key_lines`]): none
/// when the key is new. Line endings count. Second, `edited` parses to `expected`, the old file
/// with only the key changed. Either failing means the editor misplaced something, and the
/// message says so.
fn confirm(
    old: &str,
    edited: &str,
    segments: &[String],
    expected: &toml::Table,
    path: &str,
) -> Result<(), EditError> {
    let refused = || {
        EditError::new(format!(
            "editing `{path}` here would change more than the key, so the file was left as it is; \
             edit it by hand (`syntropy config edit`)"
        ))
    };
    let (a, b) = (lines(old), lines(edited));
    let prefix = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let suffix = a[prefix..]
        .iter()
        .rev()
        .zip(b[prefix..].iter().rev())
        .take_while(|(x, y)| x == y)
        .count();
    let changed = prefix..a.len() - suffix;
    if !changed.is_empty() {
        match key_lines(old, segments) {
            Some((first, last)) if changed.start >= first && changed.end <= last + 1 => {}
            _ => return Err(refused()),
        }
    }
    let mut expected = expected.clone();
    prune(&mut expected);
    let edited = edited.parse::<toml::Table>().map(|mut table| {
        prune(&mut table);
        table
    });
    match edited {
        Ok(table) if table == expected => Ok(()),
        _ => Err(refused()),
    }
}

/// Drops empty tables, at any depth: an empty table sets no key, and removing a table's last
/// dotted key leaves none in the file.
fn prune(table: &mut toml::Table) {
    for (_, value) in table.iter_mut() {
        if let toml::Value::Table(inner) = value {
            prune(inner);
        }
    }
    table.retain(|_, value| !matches!(value, toml::Value::Table(inner) if inner.is_empty()));
}

fn expected_set(table: &mut toml::Table, segments: &[String], value: toml::Value) {
    let (last, parents) = segments.split_last().expect("a key has a segment");
    let mut current = table;
    for segment in parents {
        let next = current
            .entry(segment.clone())
            .or_insert_with(|| toml::Value::Table(toml::Table::new()));
        match next {
            toml::Value::Table(inner) => current = inner,
            // `set_in_table` refuses a value in the way; the check then has nothing to match.
            _ => return,
        }
    }
    current.insert(last.clone(), value);
}

fn expected_unset(table: &mut toml::Table, segments: &[String]) {
    let (last, parents) = segments.split_last().expect("a key has a segment");
    let mut current = table;
    for segment in parents {
        match current.get_mut(segment) {
            Some(toml::Value::Table(inner)) => current = inner,
            _ => return,
        }
    }
    current.remove(last);
}

fn not_a_table(path: &str, at: &[String]) -> EditError {
    EditError::new(format!(
        "`{path}` cannot be set: `{}` holds a value, not a table",
        at.join(".")
    ))
}

/// Sets `segments` in `table`. `root` is whether `table` is the document's root, where a new
/// table is a `[header]` table; inside a table, a missing one is a dotted key.
fn set_in_table(
    table: &mut Table,
    segments: &[String],
    new: EditValue,
    path: &str,
    root: bool,
) -> Result<(), EditError> {
    let (first, rest) = segments.split_first().expect("a key has a segment");
    if rest.is_empty() {
        return match table.get_mut(first) {
            Some(Item::Value(old)) => {
                replace(old, new);
                Ok(())
            }
            Some(Item::Table(_) | Item::ArrayOfTables(_)) => Err(EditError::new(format!(
                "`{path}` is a table in the file; set one of its keys instead"
            ))),
            Some(Item::None) | None => {
                let indent = last_indent(table);
                table.insert(first, Item::Value(new));
                if let Some(mut key) = table.key_mut(first) {
                    key.leaf_decor_mut().set_prefix(indent);
                }
                Ok(())
            }
        };
    }
    match table.get_mut(first) {
        Some(Item::Table(inner)) => set_in_table(inner, rest, new, path, false),
        Some(Item::Value(EditValue::InlineTable(inner))) => set_in_inline(inner, rest, new, path),
        Some(Item::Value(_)) => Err(not_a_table(path, &segments[..1])),
        Some(Item::ArrayOfTables(_)) => Err(EditError::new(format!(
            "`{path}` is inside an array of tables, which `config set` cannot address; use \
             `syntropy config edit`"
        ))),
        Some(Item::None) | None => {
            let mut inner = Table::new();
            if root {
                // A new `[a.b]` table at the end of the file; its parents stay implicit.
                inner.set_implicit(true);
                if rest.len() == 1 {
                    inner.set_implicit(false);
                }
            } else {
                // Inside an existing table: `b.c = …` on a line of its own.
                inner.set_dotted(true);
            }
            // A dotted line prints its leaf key's prefix first, so the indentation goes there.
            let indent = if root {
                String::new()
            } else {
                last_indent(table)
            };
            set_in_new_table(&mut inner, rest, new, root, &indent);
            table.insert(first, Item::Table(inner));
            Ok(())
        }
    }
}

/// Fills a table this edit created: header tables down to the key's own table at the root, or
/// dotted ones inside a table.
fn set_in_new_table(
    table: &mut Table,
    segments: &[String],
    new: EditValue,
    root: bool,
    indent: &str,
) {
    let (first, rest) = segments.split_first().expect("a key has a segment");
    if rest.is_empty() {
        table.insert(first, Item::Value(new));
        if !root && let Some(mut key) = table.key_mut(first) {
            key.leaf_decor_mut().set_prefix(indent);
        }
        return;
    }
    let mut inner = Table::new();
    if root {
        inner.set_implicit(rest.len() > 1);
    } else {
        inner.set_dotted(true);
    }
    set_in_new_table(&mut inner, rest, new, root, indent);
    table.insert(first, Item::Table(inner));
}

fn set_in_inline(
    table: &mut InlineTable,
    segments: &[String],
    new: EditValue,
    path: &str,
) -> Result<(), EditError> {
    let (first, rest) = segments.split_first().expect("a key has a segment");
    if rest.is_empty() {
        if let Some(old) = table.get_mut(first) {
            replace(old, new);
            return Ok(());
        }
        // The new entry goes last: it takes the last entry's spacing before `}`.
        let mut new = new;
        let tail = table.iter_mut().last().map(|(_, value)| {
            let suffix = value.decor().suffix().cloned();
            value.decor_mut().set_suffix("");
            suffix
        });
        new.decor_mut().set_prefix(" ");
        if let Some(Some(suffix)) = tail {
            new.decor_mut().set_suffix(suffix);
        }
        table.insert(first, new);
        return Ok(());
    }
    match table.get_mut(first) {
        Some(EditValue::InlineTable(inner)) => set_in_inline(inner, rest, new, path),
        Some(_) => Err(not_a_table(path, &segments[..1])),
        None => {
            let mut inner = InlineTable::new();
            set_in_inline(&mut inner, rest, new, path)?;
            inner.fmt();
            set_in_inline(table, &segments[..1], EditValue::InlineTable(inner), path)
        }
    }
}

/// Replaces `old` with `new`, keeping the spacing around `old` and its line's comment.
fn replace(old: &mut EditValue, mut new: EditValue) {
    *new.decor_mut() = old.decor().clone();
    *old = new;
}

/// The indentation of the last key in `table`, for a key added after it.
fn last_indent(table: &Table) -> String {
    table
        .iter()
        .filter_map(|(key, _)| table.key(key))
        .last()
        .and_then(|key| key.leaf_decor().prefix().and_then(|p| p.as_str()))
        .map(|prefix| {
            let line = prefix.rsplit('\n').next().unwrap_or(prefix);
            line.chars()
                .take_while(|c| *c == ' ' || *c == '\t')
                .collect()
        })
        .unwrap_or_default()
}

/// Removes `segments`; whether the file set the key.
fn unset_in_doc(doc: &mut DocumentMut, segments: &[String], path: &str) -> Result<bool, EditError> {
    let (last, parents) = segments.split_last().expect("a key has a segment");
    // The comments above the removed key, and where they go: the next key of its table, or the
    // next header in the file, or the file's end.
    let comments;
    let position;
    {
        let mut table = doc.as_table_mut();
        for (i, segment) in parents.iter().enumerate() {
            match table.get_mut(segment) {
                Some(Item::Table(inner)) => table = inner,
                Some(Item::Value(EditValue::InlineTable(inline))) => {
                    return unset_in_inline(inline, &segments[i + 1..], path);
                }
                Some(Item::Value(_)) => return Err(not_a_table(path, &segments[..=i])),
                Some(Item::ArrayOfTables(_)) => {
                    return Err(EditError::new(format!(
                        "`{path}` is inside an array of tables, which `config unset` cannot \
                         address; use `syntropy config edit`"
                    )));
                }
                Some(Item::None) | None => return Ok(false),
            }
        }
        match table.get(last) {
            Some(Item::Value(_)) => {}
            Some(Item::Table(_) | Item::ArrayOfTables(_)) => {
                return Err(EditError::new(format!(
                    "`{path}` is a table in the file; unset its keys one at a time, or use \
                     `syntropy config edit`"
                )));
            }
            Some(Item::None) | None => return Ok(false),
        }
        let keys: Vec<String> = table.iter().map(|(key, _)| key.to_owned()).collect();
        let next = keys
            .iter()
            .position(|key| key == last)
            .and_then(|at| keys.get(at + 1))
            .cloned();
        let (key, _) = table.remove_entry(last).expect("the key is there");
        let above = key
            .leaf_decor()
            .prefix()
            .and_then(|p| p.as_str())
            .map(comment_lines)
            .unwrap_or_default();
        if above.is_empty() {
            return Ok(true);
        }
        match next.and_then(|next| table.key_mut(&next)) {
            Some(mut next) => {
                let old = next
                    .leaf_decor()
                    .prefix()
                    .and_then(|p| p.as_str())
                    .unwrap_or("")
                    .to_owned();
                next.leaf_decor_mut().set_prefix(format!("{above}{old}"));
                return Ok(true);
            }
            None => {
                comments = above;
                position = table.position();
            }
        }
    }
    // The next header after the key's table, in file order.
    let mut headers = Vec::new();
    collect_headers(doc.as_table(), &mut Vec::new(), &mut headers);
    let next = headers
        .into_iter()
        .filter(|(at, _)| position.is_none_or(|p| *at > p))
        .min_by_key(|(at, _)| *at);
    match next {
        Some((_, header)) => {
            let table = header_mut(doc.as_table_mut(), &header).expect("collected above");
            let old = table
                .decor()
                .prefix()
                .and_then(|p| p.as_str())
                .unwrap_or("")
                .to_owned();
            table.decor_mut().set_prefix(format!("{comments}{old}"));
        }
        None => {
            let old = doc.trailing().as_str().unwrap_or("").to_owned();
            doc.set_trailing(format!("{comments}{old}"));
        }
    }
    Ok(true)
}

fn unset_in_inline(
    table: &mut InlineTable,
    segments: &[String],
    path: &str,
) -> Result<bool, EditError> {
    let (first, rest) = segments.split_first().expect("a key has a segment");
    if rest.is_empty() {
        let was_last = table.iter().last().is_some_and(|(key, _)| key == first);
        let Some((_, removed)) = table.remove_entry(first) else {
            return Ok(false);
        };
        // The entry that is now last takes the spacing before `}`.
        if was_last
            && let Some((_, value)) = table.iter_mut().last()
            && let Some(suffix) = removed.decor().suffix()
        {
            value.decor_mut().set_suffix(suffix.clone());
        }
        return Ok(true);
    }
    match table.get_mut(first) {
        Some(EditValue::InlineTable(inner)) => unset_in_inline(inner, rest, path),
        Some(_) => Err(not_a_table(path, &segments[..1])),
        None => Ok(false),
    }
}

/// The whole comment lines in a key's prefix: everything up to and including its last newline.
/// The rest is the key's own indentation, which goes with it.
fn comment_lines(prefix: &str) -> String {
    match prefix.rfind('\n') {
        Some(end) if prefix[..end].contains('#') => prefix[..=end].to_owned(),
        _ => String::new(),
    }
}

/// Every header table under `table`, with its position in the file and its path.
fn collect_headers(table: &Table, path: &mut Vec<String>, out: &mut Vec<(isize, Vec<String>)>) {
    for (key, item) in table.iter() {
        if let Item::Table(inner) = item {
            path.push(key.to_owned());
            if !inner.is_implicit()
                && !inner.is_dotted()
                && let Some(at) = inner.position()
            {
                out.push((at, path.clone()));
            }
            collect_headers(inner, path, out);
            path.pop();
        }
    }
}

fn header_mut<'a>(table: &'a mut Table, path: &[String]) -> Option<&'a mut Table> {
    let (first, rest) = path.split_first()?;
    let inner = table.get_mut(first)?.as_table_mut()?;
    if rest.is_empty() {
        Some(inner)
    } else {
        header_mut(inner, rest)
    }
}

/// Resolves and validates the configuration layer as it would be with `text` as the file
/// `target`: RFC-0054 §8's phases 1–5, as [`crate::load_and_validate`] runs them, with no flags.
/// `scan` lists the layer's files with `target` among them, whether it exists yet or not; every
/// other file is read from disk.
#[allow(clippy::too_many_arguments)]
pub fn check_edit(
    registry: &Registry,
    os: Os,
    scan: &Scan,
    env: &Env,
    target: &Path,
    text: &str,
    rules: &[Rule],
    environment: &dyn Environment,
) -> Checked {
    let mut files = Vec::new();
    let mut unreadable = Vec::new();
    for path in scan.files() {
        if path == target {
            files.push(ConfigFile::new(path, text));
        } else {
            match ConfigFile::read_one(&path) {
                Ok(file) => files.push(file),
                Err(problem) => unreadable.push(problem),
            }
        }
    }
    let flags = Flags::new();
    let layers = Layers {
        os,
        files: &files,
        env,
        flags: &flags,
    };
    let outcome = ResolvedConfig::resolve_with(registry, layers, |_| None)
        .and_then(|config| validate(registry, config, rules, environment));
    let outcome = if unreadable.is_empty() {
        outcome
    } else {
        let (mut errors, warnings) = match outcome {
            Ok(config) => (Vec::new(), config.warnings().to_vec()),
            Err(rejected) => (rejected.errors, rejected.warnings),
        };
        unreadable.append(&mut errors);
        Err(crate::resolve::Rejected {
            errors: unreadable,
            warnings,
        })
    };
    Checked { files, outcome }
}

/// Removes a temporary file that was not renamed into place. On Windows a read-only file cannot
/// be removed, so the attribute is cleared first: the file holds the new text, which can be a
/// secret, and must not stay beside the config.
fn discard(path: &Path) {
    #[cfg(windows)]
    if let Ok(meta) = fs::metadata(path) {
        let mut permissions = meta.permissions();
        if permissions.readonly() {
            #[allow(
                clippy::permissions_set_readonly_false,
                reason = "Windows only, where it clears the read-only attribute"
            )]
            permissions.set_readonly(false);
            let _ = fs::set_permissions(path, permissions);
        }
    }
    let _ = fs::remove_file(path);
}

/// Whether a write may replace `path`: `Ok(true)` for a regular file, `Ok(false)` when nothing is
/// there yet, and an error for a symbolic link or anything that is not a regular file.
pub fn replaceable(path: &Path) -> io::Result<bool> {
    existing(path).map(|meta| meta.is_some())
}

/// The file a write replaces: `path`, unless it is a symbolic link, which is refused. `None`
/// when it does not exist yet.
fn existing(path: &Path) -> io::Result<Option<fs::Metadata>> {
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_symlink() => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "`{}` is a symbolic link; edit the file it points to, or name that file with \
                 --config",
                path.display()
            ),
        )),
        Ok(meta) if !meta.is_file() => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("`{}` is not a regular file", path.display()),
        )),
        // Refused before anything is written: on Windows the rename over a read-only file fails,
        // and someone marked it so it would not change.
        Ok(meta) if meta.permissions().readonly() => Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!(
                "`{}` is read-only; make it writable to change it",
                path.display()
            ),
        )),
        Ok(meta) => Ok(Some(meta)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// Replaces the file at `path` with `text`, atomically: a temporary file in the same directory,
/// flushed to disk, with the old file's permissions (`0600` for a new one on Unix, since a config
/// file can hold a secret), renamed over `path`, then the directory flushed on Unix. A target
/// that is a symbolic link is refused; the rename replaces the directory entry itself, so a link
/// that appears between the check and the rename is replaced, never followed. A read-only
/// target is refused too. The directory is created if it does not exist.
///
/// On Windows the permissions copied are the read-only attribute only: the new file takes the
/// directory's inherited ACL, not the old file's. Under `%LOCALAPPDATA%` that is the user's own;
/// a file given a narrower ACL by hand loses it.
pub fn write_atomic(path: &Path, text: &str) -> io::Result<()> {
    let old = existing(path)?;
    let dir = match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.to_path_buf(),
        _ => PathBuf::from("."),
    };
    fs::create_dir_all(&dir)?;
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "the path names no file"))?;
    let mut temp = None;
    for attempt in 0..100u32 {
        let candidate = dir.join(format!(
            ".{}.syntropy-write-{}-{attempt}",
            name.to_string_lossy(),
            std::process::id()
        ));
        let mut options = fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::{OpenOptionsExt as _, PermissionsExt as _};
            let mode = old
                .as_ref()
                .map_or(0o600, |meta| meta.permissions().mode() & 0o7777);
            options.mode(mode);
        }
        match options.open(&candidate) {
            Ok(file) => {
                temp = Some((candidate, file));
                break;
            }
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    let (temp_path, mut file) = temp.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::AlreadyExists,
            "no free name for the temporary file",
        )
    })?;
    let written = (|| {
        file.write_all(text.as_bytes())?;
        file.sync_all()?;
        if let Some(meta) = &old {
            // The mode was set at creation, past the umask; this sets it exactly, and on Windows
            // keeps the read-only flag.
            fs::set_permissions(&temp_path, meta.permissions())?;
        }
        drop(file);
        fs::rename(&temp_path, path)
    })();
    if let Err(e) = written {
        discard(&temp_path);
        return Err(e);
    }
    #[cfg(unix)]
    {
        fs::File::open(&dir)?.sync_all()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
