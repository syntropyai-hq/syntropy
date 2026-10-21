// SPDX-License-Identifier: MIT OR Apache-2.0

//! The two pieces of TOML the licence checks read, without a parser crate:
//!
//! - [`tables`] reads a file of `[[name]]` tables whose keys hold basic strings or booleans,
//!   with `#` comments and blank lines: the shape of `third_party/licences.toml`. Anything else
//!   is an error that names the line, so every file it accepts is valid TOML with the same
//!   meaning.
//! - [`string_array`] finds one `key = ["...", ...]` array in one `[section]` of a file it does
//!   not otherwise parse, such as `allow` in the `[licenses]` table of `deny.toml`.

/// One value of a [`Table`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    Str(String),
    Bool(bool),
}

/// One `[[name]]` table: its keys in file order, and the line its header is on.
#[derive(Debug, PartialEq, Eq)]
pub struct Table {
    pub line: usize,
    pub entries: Vec<(String, Value)>,
}

impl Table {
    /// The string value of `key`, if the table has one.
    pub fn str(&self, key: &str) -> Option<&str> {
        self.entries.iter().find_map(|(k, v)| match v {
            Value::Str(s) if k == key => Some(s.as_str()),
            _ => None,
        })
    }
}

/// Reads every `[[name]]` table in `text`. Keys outside a table, other headers, repeated keys
/// and values other than basic strings and `true`/`false` are errors, each starting with its line
/// number.
pub fn tables(text: &str, name: &str) -> Result<Vec<Table>, String> {
    let header = format!("[[{name}]]");
    let mut tables: Vec<Table> = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let number = index + 1;
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') {
            if line != header {
                return Err(format!(
                    "{number}: unexpected header `{line}`; this file holds only `{header}` tables"
                ));
            }
            tables.push(Table {
                line: number,
                entries: Vec::new(),
            });
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| format!("{number}: expected `key = \"value\"`"))?;
        let key = key.trim();
        if key.is_empty()
            || !key
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return Err(format!("{number}: `{key}` is not a bare key"));
        }
        let Some(table) = tables.last_mut() else {
            return Err(format!("{number}: `{key}` is outside a `{header}` table"));
        };
        if table.entries.iter().any(|(k, _)| k == key) {
            return Err(format!("{number}: `{key}` is given twice in one table"));
        }
        let value = value.trim();
        let value = match value {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            _ => {
                let (s, rest) = string(value).map_err(|e| format!("{number}: {e}"))?;
                if !rest.trim().is_empty() {
                    return Err(format!(
                        "{number}: unexpected `{}` after the value",
                        rest.trim()
                    ));
                }
                Value::Str(s)
            }
        };
        table.entries.push((key.to_owned(), value));
    }
    Ok(tables)
}

/// The strings of `key = [...]` in `[section]` of `text`, or `None` if the section has no such
/// key. The array may span lines and end with a comma. Only that one array is parsed.
pub fn string_array(text: &str, section: &str, key: &str) -> Result<Option<Vec<String>>, String> {
    let header = format!("[{section}]");
    let mut in_section = false;
    let mut lines = text.lines().enumerate();
    while let Some((index, raw)) = lines.next() {
        let number = index + 1;
        let line = strip_comment(raw).trim();
        if line.starts_with('[') {
            in_section = line == header;
            continue;
        }
        if !in_section {
            continue;
        }
        let Some((k, value)) = line.split_once('=') else {
            continue;
        };
        if k.trim() != key {
            continue;
        }
        let mut value = value.trim().to_owned();
        if !value.starts_with('[') {
            return Err(format!("{number}: `{key}` in `{header}` is not an array"));
        }
        while !value.ends_with(']') {
            let Some((_, more)) = lines.next() else {
                return Err(format!("{number}: the `{key}` array has no closing `]`"));
            };
            value.push(' ');
            value.push_str(strip_comment(more).trim());
        }
        let mut items = Vec::new();
        let mut rest = value[1..value.len() - 1].trim();
        while !rest.is_empty() {
            let (item, after) = string(rest).map_err(|e| format!("{number}: {e}"))?;
            items.push(item);
            rest = after.trim_start();
            rest = match rest.strip_prefix(',') {
                Some(after_comma) => after_comma.trim_start(),
                None if rest.is_empty() => rest,
                None => return Err(format!("{number}: expected `,` between array items")),
            };
        }
        return Ok(Some(items));
    }
    Ok(None)
}

/// `line` up to a `#` that is not inside a basic string.
pub fn strip_comment(line: &str) -> &str {
    let mut in_string = false;
    let mut escaped = false;
    for (at, c) in line.char_indices() {
        match c {
            _ if escaped => escaped = false,
            '\\' if in_string => escaped = true,
            '"' => in_string = !in_string,
            '#' if !in_string => return &line[..at],
            _ => {}
        }
    }
    line
}

/// A basic string at the start of `text`, and the text after it. Supports the escapes `\"`,
/// `\\`, `\n` and `\t`.
fn string(text: &str) -> Result<(String, &str), String> {
    let Some(body) = text.strip_prefix('"') else {
        return Err(format!("expected a double-quoted string, found `{text}`"));
    };
    let mut out = String::new();
    let mut chars = body.char_indices();
    while let Some((at, c)) = chars.next() {
        match c {
            '"' => return Ok((out, &body[at + 1..])),
            '\\' => match chars.next() {
                Some((_, '"')) => out.push('"'),
                Some((_, '\\')) => out.push('\\'),
                Some((_, 'n')) => out.push('\n'),
                Some((_, 't')) => out.push('\t'),
                Some((_, other)) => return Err(format!("unsupported escape `\\{other}`")),
                None => break,
            },
            c => out.push(c),
        }
    }
    Err("a string has no closing `\"`".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_tables_of_strings_and_booleans() {
        let text = "\
# A comment.

[[file]]
path = \"third_party/a.c\"   # trailing comment
upstream = \"https://example.org/a#readme\"
flag = true

[[file]]
path = \"third_party/b \\\"q\\\".c\"
";
        let read = tables(text, "file").unwrap();
        assert_eq!(read.len(), 2);
        assert_eq!(read[0].line, 3);
        assert_eq!(read[0].str("path"), Some("third_party/a.c"));
        assert_eq!(
            read[0].str("upstream"),
            Some("https://example.org/a#readme")
        );
        assert_eq!(read[0].entries[2], ("flag".to_owned(), Value::Bool(true)));
        assert_eq!(read[1].str("path"), Some("third_party/b \"q\".c"));
        assert_eq!(read[1].str("upstream"), None);
    }

    #[test]
    fn an_empty_file_has_no_tables() {
        assert_eq!(tables("# only comments\n\n", "file").unwrap(), Vec::new());
    }

    #[test]
    fn rejects_what_it_does_not_read() {
        for (text, error) in [
            ("path = \"a\"\n", "1: `path` is outside a `[[file]]` table"),
            ("[file]\n", "1: unexpected header `[file]`"),
            ("[[other]]\n", "1: unexpected header `[[other]]`"),
            (
                "[[file]]\npath = 'a'\n",
                "2: expected a double-quoted string",
            ),
            (
                "[[file]]\npath = [\"a\"]\n",
                "2: expected a double-quoted string",
            ),
            (
                "[[file]]\npath = \"a\" x\n",
                "2: unexpected `x` after the value",
            ),
            (
                "[[file]]\npath = \"a\npath\"\n",
                "2: a string has no closing",
            ),
            (
                "[[file]]\npath = \"a\"\npath = \"b\"\n",
                "3: `path` is given twice",
            ),
            (
                "[[file]]\nthe path = \"a\"\n",
                "2: `the path` is not a bare key",
            ),
            ("[[file]]\npath\n", "2: expected `key = \"value\"`"),
        ] {
            let got = tables(text, "file").unwrap_err();
            assert!(got.starts_with(error), "{text:?}: {got}");
        }
    }

    #[test]
    fn finds_one_array_in_one_section() {
        let text = "\
[graph]
allow = [\"not this one\"]

[licenses]
# allow = [\"commented out\"]
allow = [
    \"MIT\",
    \"Apache-2.0 WITH LLVM-exception\", # a comment, with a ] in it
    # a whole-line comment
    \"CDLA-Permissive-2.0\",
]
exceptions = [{ allow = [\"MPL-2.0\"], crate = \"x\" }]

[bans]
allow = []
";
        assert_eq!(
            string_array(text, "licenses", "allow").unwrap(),
            Some(vec![
                "MIT".to_owned(),
                "Apache-2.0 WITH LLVM-exception".to_owned(),
                "CDLA-Permissive-2.0".to_owned(),
            ])
        );
        assert_eq!(string_array(text, "bans", "allow").unwrap(), Some(vec![]));
        assert_eq!(string_array(text, "sources", "allow").unwrap(), None);
        assert_eq!(
            string_array("[licenses]\nallow = [\"MIT\"]", "licenses", "allow").unwrap(),
            Some(vec!["MIT".to_owned()])
        );
    }

    #[test]
    fn a_bad_array_is_an_error() {
        for text in [
            "[licenses]\nallow = \"MIT\"\n",
            "[licenses]\nallow = [\"MIT\"\n",
            "[licenses]\nallow = [\"MIT\" \"ISC\"]\n",
            "[licenses]\nallow = [MIT]\n",
        ] {
            assert!(string_array(text, "licenses", "allow").is_err(), "{text:?}");
        }
    }

    #[test]
    fn comments_stop_outside_strings_only() {
        assert_eq!(strip_comment("a = \"x#y\" # z"), "a = \"x#y\" ");
        assert_eq!(strip_comment("a = \"x\\\"#y\""), "a = \"x\\\"#y\"");
        assert_eq!(strip_comment("# all"), "");
    }
}
