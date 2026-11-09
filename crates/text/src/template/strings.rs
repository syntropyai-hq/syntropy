// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Python string methods and the string filters model templates call, made marker-aware
//! (`markup.rs`): each takes a marked string apart, works on its plain text, and marks the
//! result again, so content stays content.
//!
//! Methods: `startswith`, `endswith`, `split`, `strip`, `lstrip`, `rstrip`, `upper`, `lower` and
//! `replace`, with Python's arguments. Filters replaced: `trim`, `length` and `count`, `upper`,
//! `lower` and `replace`. Whitespace is Python's `str.isspace`. A method not listed here is an
//! error, never a silent empty value.

use minijinja::value::Value;
use minijinja::{Error, ErrorKind, State};

use super::markup::Marked;

/// Python's `str.isspace` for one character: Unicode whitespace plus U+001C–U+001F.
fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

fn markup_error(e: super::TemplateError) -> Error {
    Error::new(ErrorKind::InvalidOperation, e.to_string())
}

/// The marked string a value holds.
pub(super) fn marked(v: &Value) -> Result<Marked, Error> {
    let s = v
        .as_str()
        .ok_or_else(|| Error::new(ErrorKind::InvalidOperation, "expected a string"))?;
    Marked::parse(s).map_err(markup_error)
}

/// The plain text of a string argument (an argument's own marking does not matter).
fn plain(v: &Value) -> Result<String, Error> {
    Ok(marked(v)?.text)
}

/// The characters `chars` names, or Python whitespace when it is absent or none.
fn strip_set(chars: Option<&Value>) -> Result<Option<Vec<char>>, Error> {
    match chars {
        None => Ok(None),
        Some(v) if v.is_none() || v.is_undefined() => Ok(None),
        Some(v) => Ok(Some(plain(v)?.chars().collect())),
    }
}

/// `strip`, `lstrip` and `rstrip`.
fn strip(m: &Marked, set: Option<&[char]>, left: bool, right: bool) -> Value {
    let hit = |c: char| set.map_or_else(|| py_space(c), |s| s.contains(&c));
    let text = &m.text;
    let start = if left {
        text.len() - text.trim_start_matches(hit).len()
    } else {
        0
    };
    let end = if right {
        text.trim_end_matches(hit).len().max(start)
    } else {
        text.len()
    };
    Value::from(m.emit(start..end))
}

/// Python's `str.split(sep=None, maxsplit=-1)`.
fn split(m: &Marked, sep: Option<&str>, maxsplit: i64) -> Result<Value, Error> {
    let text = &m.text;
    let limit = usize::try_from(maxsplit).ok();
    let mut out = Vec::new();
    match sep {
        Some("") => return Err(Error::new(ErrorKind::InvalidOperation, "empty separator")),
        Some(sep) => {
            let mut start = 0;
            let mut splits = 0;
            while limit.is_none_or(|l| splits < l) {
                let Some(i) = text[start..].find(sep) else {
                    break;
                };
                out.push(Value::from(m.emit(start..start + i)));
                start += i + sep.len();
                splits += 1;
            }
            out.push(Value::from(m.emit(start..text.len())));
        }
        None => {
            // Runs of whitespace separate; leading and trailing whitespace gives no empty piece.
            let mut pieces = Vec::new();
            let mut i = 0;
            let bytes_at = |i: usize| text[i..].chars().next();
            while i < text.len() {
                while let Some(c) = bytes_at(i).filter(|&c| py_space(c)) {
                    i += c.len_utf8();
                }
                if i >= text.len() {
                    break;
                }
                if limit.is_some_and(|l| pieces.len() == l) {
                    // Python keeps the rest as it is, trailing whitespace included.
                    pieces.push(i..text.len());
                    break;
                }
                let start = i;
                while let Some(c) = bytes_at(i).filter(|&c| !py_space(c)) {
                    i += c.len_utf8();
                }
                pieces.push(start..i);
            }
            out.extend(pieces.into_iter().map(|r| Value::from(m.emit(r))));
        }
    }
    Ok(Value::from(out))
}

/// Applies a per-character text map to each run, keeping each run's kind.
fn map_case(m: &Marked, f: fn(&str) -> String) -> Value {
    let mut out = Marked::default();
    let mut i = 0;
    while i < m.text.len() {
        let kind = m.content[i];
        let mut j = i;
        while j < m.text.len() && m.content[j] == kind {
            j += 1;
        }
        let mapped = f(&m.text[i..j]);
        out.content.extend(std::iter::repeat_n(kind, mapped.len()));
        out.text.push_str(&mapped);
        i = j;
    }
    Value::from(out.emit_all())
}

/// `replace(old, new, count=-1)`. The replacement is text the template wrote.
fn replace(m: &Marked, old: &str, new: &str, count: i64) -> Value {
    let limit = usize::try_from(count).ok();
    let new = Marked::template(new);
    let mut out = Marked::default();
    let mut start = 0;
    let mut done = 0usize;
    if old.is_empty() {
        // Python inserts `new` at each character boundary, the end included, up to `count`.
        let bounds = m.text.char_indices().map(|(i, _)| i).chain([m.text.len()]);
        for b in bounds {
            if limit.is_none_or(|l| done < l) {
                out.push(&new);
                done += 1;
            }
            let end = m.text[b..].chars().next().map_or(b, |c| b + c.len_utf8());
            out.push(&slice(m, b..end));
        }
        return Value::from(out.emit_all());
    }
    while limit.is_none_or(|l| done < l) {
        let Some(i) = m.text[start..].find(old) else {
            break;
        };
        out.push(&slice(m, start..start + i));
        out.push(&new);
        start += i + old.len();
        done += 1;
    }
    out.push(&slice(m, start..m.text.len()));
    Value::from(out.emit_all())
}

fn slice(m: &Marked, r: std::ops::Range<usize>) -> Marked {
    Marked {
        text: m.text[r.clone()].to_owned(),
        content: m.content[r].to_vec(),
    }
}

fn arg(args: &[Value], i: usize) -> Option<&Value> {
    args.get(i).filter(|v| !v.is_undefined())
}

fn int_arg(args: &[Value], i: usize, default: i64) -> Result<i64, Error> {
    match arg(args, i) {
        None => Ok(default),
        Some(v) if v.is_none() => Ok(default),
        Some(v) => i64::try_from(v.clone())
            .map_err(|_| Error::new(ErrorKind::InvalidOperation, "expected an integer")),
    }
}

fn too_many(args: &[Value], max: usize) -> Result<(), Error> {
    if args.len() > max {
        Err(Error::from(ErrorKind::TooManyArguments))
    } else {
        Ok(())
    }
}

/// `prefix` as Python takes it: one string or a sequence of strings.
fn affixes(v: &Value) -> Result<Vec<String>, Error> {
    if v.as_str().is_some() {
        return Ok(vec![plain(v)?]);
    }
    v.try_iter()?.map(|x| plain(&x)).collect()
}

/// The unknown-method callback: string methods only.
pub(super) fn method(
    _state: &State,
    value: &Value,
    name: &str,
    args: &[Value],
) -> Result<Value, Error> {
    if value.as_str().is_none() {
        return Err(Error::from(ErrorKind::UnknownMethod));
    }
    let m = marked(value)?;
    let need = |i: usize| arg(args, i).ok_or_else(|| Error::from(ErrorKind::MissingArgument));
    match name {
        "startswith" | "endswith" => {
            too_many(args, 1)?;
            let set = affixes(need(0)?)?;
            let hit = if name == "startswith" {
                set.iter().any(|p| m.text.starts_with(p.as_str()))
            } else {
                set.iter().any(|p| m.text.ends_with(p.as_str()))
            };
            Ok(Value::from(hit))
        }
        "split" => {
            too_many(args, 2)?;
            let sep = match arg(args, 0) {
                Some(v) if !v.is_none() => Some(plain(v)?),
                _ => None,
            };
            split(&m, sep.as_deref(), int_arg(args, 1, -1)?)
        }
        "strip" | "lstrip" | "rstrip" => {
            too_many(args, 1)?;
            let set = strip_set(arg(args, 0))?;
            Ok(strip(
                &m,
                set.as_deref(),
                name != "rstrip",
                name != "lstrip",
            ))
        }
        "upper" => {
            too_many(args, 0)?;
            Ok(map_case(&m, str::to_uppercase))
        }
        "lower" => {
            too_many(args, 0)?;
            Ok(map_case(&m, str::to_lowercase))
        }
        "replace" => {
            too_many(args, 3)?;
            let (old, new) = (plain(need(0)?)?, plain(need(1)?)?);
            Ok(replace(&m, &old, &new, int_arg(args, 2, -1)?))
        }
        _ => Err(Error::from(ErrorKind::UnknownMethod)),
    }
}

/// The `trim` filter: `strip`, with optional characters.
pub(super) fn trim_filter(value: &Value, chars: Option<Value>) -> Result<Value, Error> {
    let m = marked(value)?;
    let set = strip_set(chars.as_ref())?;
    Ok(strip(&m, set.as_deref(), true, true))
}

/// The `length` and `count` filters: a string's length in characters, without markers.
pub(super) fn length_filter(value: &Value) -> Result<Value, Error> {
    if value.as_str().is_some() {
        return Ok(Value::from(marked(value)?.text.chars().count()));
    }
    value.len().map(Value::from).ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidOperation,
            format!("cannot calculate length of value of type {}", value.kind()),
        )
    })
}

/// The `upper` filter.
pub(super) fn upper_filter(value: &Value) -> Result<Value, Error> {
    Ok(map_case(&marked(value)?, str::to_uppercase))
}

/// The `lower` filter.
pub(super) fn lower_filter(value: &Value) -> Result<Value, Error> {
    Ok(map_case(&marked(value)?, str::to_lowercase))
}

/// The `replace` filter.
pub(super) fn replace_filter(
    value: &Value,
    old: &Value,
    new: &Value,
    count: Option<i64>,
) -> Result<Value, Error> {
    Ok(replace(
        &marked(value)?,
        &plain(old)?,
        &plain(new)?,
        count.unwrap_or(-1),
    ))
}

#[cfg(test)]
mod tests {
    use super::super::markup::mark;
    use super::*;

    fn call(value: &str, name: &str, args: &[Value]) -> Result<Value, Error> {
        let env = minijinja::Environment::new();
        let state = env.empty_state();
        method(&state, &Value::from(value), name, args)
    }

    fn texts(v: &Value) -> Vec<(String, Vec<bool>)> {
        v.try_iter()
            .unwrap()
            .map(|x| {
                let m = marked(&x).unwrap();
                (m.text, m.content)
            })
            .collect()
    }

    #[test]
    fn split_matches_python_and_keeps_kinds() {
        let s = format!("x{}", mark("a</think>b"));
        let v = call(&s, "split", &[Value::from("</think>")]).unwrap();
        let parts = texts(&v);
        assert_eq!(parts[0].0, "xa");
        assert_eq!(parts[0].1, [false, true]);
        assert_eq!(parts[1].0, "b");
        assert_eq!(parts[1].1, [true]);
        let v = call("  a  b ", "split", &[]).unwrap();
        assert_eq!(
            texts(&v).into_iter().map(|p| p.0).collect::<Vec<_>>(),
            ["a", "b"]
        );
        let v = call("a,b,c", "split", &[Value::from(","), Value::from(1)]).unwrap();
        assert_eq!(
            texts(&v).into_iter().map(|p| p.0).collect::<Vec<_>>(),
            ["a", "b,c"]
        );
        let v = call(" a b  c ", "split", &[Value::from(()), Value::from(1)]).unwrap();
        assert_eq!(
            texts(&v).into_iter().map(|p| p.0).collect::<Vec<_>>(),
            ["a", "b  c "]
        );
        assert!(call("a", "split", &[Value::from("")]).is_err());
    }

    #[test]
    fn strip_family_matches_python() {
        let v = call(&mark("\n\nhi\n"), "strip", &[Value::from("\n")]).unwrap();
        assert_eq!(marked(&v).unwrap().text, "hi");
        let v = call(" \u{1f}hi ", "lstrip", &[]).unwrap();
        assert_eq!(marked(&v).unwrap().text, "hi ");
        let v = call("xxhixx", "rstrip", &[Value::from("x")]).unwrap();
        assert_eq!(marked(&v).unwrap().text, "xxhi");
        let v = call("\n\n", "strip", &[]).unwrap();
        assert_eq!(marked(&v).unwrap().text, "");
    }

    #[test]
    fn startswith_reads_plain_text() {
        let s = mark("<tool_response>x</tool_response>");
        assert!(
            call(&s, "startswith", &[Value::from("<tool_response>")])
                .unwrap()
                .is_true()
        );
        assert!(
            call(&s, "endswith", &[Value::from("</tool_response>")])
                .unwrap()
                .is_true()
        );
        let both = Value::from(vec![Value::from("q"), Value::from("<tool")]);
        assert!(call(&s, "startswith", &[both]).unwrap().is_true());
    }

    #[test]
    fn replace_marks_the_replacement_as_template_text() {
        let v = call(
            &mark("a-b"),
            "replace",
            &[Value::from("-"), Value::from("+")],
        )
        .unwrap();
        let m = marked(&v).unwrap();
        assert_eq!(m.text, "a+b");
        assert_eq!(m.content, [true, false, true]);
        let v = call("ab", "replace", &[Value::from(""), Value::from("|")]).unwrap();
        assert_eq!(marked(&v).unwrap().text, "|a|b|");
        let v = call(
            "ab",
            "replace",
            &[Value::from(""), Value::from("|"), Value::from(2)],
        )
        .unwrap();
        assert_eq!(marked(&v).unwrap().text, "|a|b");
        let v = call("", "replace", &[Value::from(""), Value::from("|")]).unwrap();
        assert_eq!(marked(&v).unwrap().text, "|");
    }

    #[test]
    fn unknown_methods_and_non_strings_are_errors() {
        assert_eq!(
            call("a", "format", &[]).unwrap_err().kind(),
            ErrorKind::UnknownMethod
        );
        let env = minijinja::Environment::new();
        let state = env.empty_state();
        assert_eq!(
            method(&state, &Value::from(3), "upper", &[])
                .unwrap_err()
                .kind(),
            ErrorKind::UnknownMethod
        );
    }
}
