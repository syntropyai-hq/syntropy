// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `tojson` filter as the reference engine defines it for chat templates: Python's
//! `json.dumps(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False)`, with
//! those four arguments. It does not escape HTML characters, as the generic Jinja filter does.
//!
//! Strings inside the value are read without their markers. When any of them held content, the
//! whole JSON text is content: a request value's JSON is a value from the request.

use minijinja::value::{Kwargs, Value};
use minijinja::{Error, ErrorKind};
use serde_json::Value as Json;

use super::markup::Marked;

/// How `json.dumps` lays out its output.
struct Style {
    ascii: bool,
    indent: Option<String>,
    item_sep: String,
    key_sep: String,
    sort_keys: bool,
}

/// The filter.
pub(super) fn tojson(value: &Value, kwargs: Kwargs) -> Result<Value, Error> {
    let ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let indent: Option<Value> = kwargs.get("indent")?;
    let separators: Option<Value> = kwargs.get("separators")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    kwargs.assert_all_used()?;

    let indent = match indent {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => match v.as_str() {
            Some(s) => Some(s.to_owned()),
            None => {
                let n = i64::try_from(v)
                    .map_err(|_| Error::new(ErrorKind::InvalidOperation, "indent: an integer"))?;
                Some(" ".repeat(usize::try_from(n).unwrap_or(0).min(64)))
            }
        },
    };
    let (item_sep, key_sep) = match separators {
        Some(v) if !v.is_none() => {
            let parts: Vec<Value> = v.try_iter()?.collect();
            match parts.as_slice() {
                [a, b] => (
                    Marked::parse(a.as_str().unwrap_or_default())
                        .map_err(super::strings_error)?
                        .text,
                    Marked::parse(b.as_str().unwrap_or_default())
                        .map_err(super::strings_error)?
                        .text,
                ),
                _ => {
                    return Err(Error::new(
                        ErrorKind::InvalidOperation,
                        "separators: a pair of strings",
                    ));
                }
            }
        }
        // Python's defaults: `', '` without an indent, `','` with one.
        _ => (
            if indent.is_some() { "," } else { ", " }.to_owned(),
            ": ".to_owned(),
        ),
    };
    let style = Style {
        ascii: ascii.unwrap_or(false),
        indent,
        item_sep,
        key_sep,
        sort_keys: sort_keys.unwrap_or(false),
    };

    let json = serde_json::to_value(value)
        .map_err(|e| Error::new(ErrorKind::BadSerialization, e.to_string()))?;
    let mut had_content = false;
    let mut out = String::new();
    write(&json, &style, 0, &mut out, &mut had_content)?;
    if had_content {
        Ok(Value::from(super::markup::mark(&out)))
    } else {
        Ok(Value::from(out))
    }
}

fn unmark(s: &str, had_content: &mut bool) -> Result<String, Error> {
    let m = Marked::parse(s).map_err(super::strings_error)?;
    *had_content |= m.has_content();
    Ok(m.text)
}

fn newline(style: &Style, depth: usize, out: &mut String) {
    if let Some(ind) = &style.indent {
        out.push('\n');
        for _ in 0..depth {
            out.push_str(ind);
        }
    }
}

fn write(
    v: &Json,
    style: &Style,
    depth: usize,
    out: &mut String,
    had_content: &mut bool,
) -> Result<(), Error> {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else {
                out.push_str(&py_float(n.as_f64().unwrap_or(f64::NAN)));
            }
        }
        Json::String(s) => string(&unmark(s, had_content)?, style.ascii, out),
        Json::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return Ok(());
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(&style.item_sep);
                }
                newline(style, depth + 1, out);
                write(item, style, depth + 1, out, had_content)?;
            }
            newline(style, depth, out);
            out.push(']');
        }
        Json::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return Ok(());
            }
            let mut entries: Vec<(String, &Json)> = map
                .iter()
                .map(|(k, v)| Ok((unmark(k, had_content)?, v)))
                .collect::<Result<_, Error>>()?;
            if style.sort_keys {
                entries.sort_by(|a, b| a.0.cmp(&b.0));
            }
            out.push('{');
            for (i, (k, v)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push_str(&style.item_sep);
                }
                newline(style, depth + 1, out);
                string(k, style.ascii, out);
                out.push_str(&style.key_sep);
                write(v, style, depth + 1, out, had_content)?;
            }
            newline(style, depth, out);
            out.push('}');
        }
    }
    Ok(())
}

/// A JSON string as Python writes it.
fn string(s: &str, ascii: bool, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (ascii && (c as u32) > 0x7e) => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A float as Python's `repr` writes it: the shortest digits that round-trip, in exponent
/// form below 1e-4 and from 1e16.
fn py_float(f: f64) -> String {
    if f.is_nan() {
        return "NaN".to_owned();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity" } else { "-Infinity" }.to_owned();
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0" } else { "0.0" }.to_owned();
    }
    let exp = f.abs().log10().floor() as i32;
    if !(-4..16).contains(&exp) {
        // `{:e}` gives `1.5e-7`; Python writes `1.5e-07`, and `1e+16`.
        let s = format!("{f:e}");
        let (mantissa, e) = s.split_once('e').unwrap_or((&s, "0"));
        let e: i32 = e.parse().unwrap_or(0);
        let sign = if e < 0 { '-' } else { '+' };
        return format!("{mantissa}e{sign}{:02}", e.abs());
    }
    let s = format!("{f}");
    if s.contains('.') { s } else { format!("{s}.0") }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_print_as_python_repr() {
        for (f, want) in [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1.5e-7, "1.5e-07"),
            (1e16, "1e+16"),
            (123456.789, "123456.789"),
            (-0.0, "-0.0"),
        ] {
            assert_eq!(py_float(f), want);
        }
    }

    #[test]
    fn strings_escape_as_python() {
        let mut out = String::new();
        string("a\"\\\n\u{1}é<>&'", false, &mut out);
        assert_eq!(out, "\"a\\\"\\\\\\n\\u0001é<>&'\"");
        let mut out = String::new();
        string("é😀", true, &mut out);
        assert_eq!(out, "\"\\u00e9\\ud83d\\ude00\"");
    }
}
