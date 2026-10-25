// SPDX-License-Identifier: MIT OR Apache-2.0

//! A tag scanner for the HTML and SVG the docs site ships (PROD-E13-T01). It finds every start
//! and end tag with its attributes, and the raw text of `<style>` and `<script>` elements, which
//! is all the link check and the origin check need. It builds no tree.

use std::ops::Range;

use super::markdown::{decode_entities, escape};

/// One tag.
#[derive(Debug, PartialEq, Eq)]
pub struct Tag {
    /// The name in lower case, such as `img` or `feimage`.
    pub name: String,
    /// `</name>`.
    pub closing: bool,
    pub attrs: Vec<Attr>,
}

/// One attribute of a tag.
#[derive(Debug, PartialEq, Eq)]
pub struct Attr {
    /// The name in lower case, such as `href` or `xlink:href`.
    pub name: String,
    /// The value with character references decoded; `None` for a bare attribute.
    pub value: Option<String>,
    /// Where the value is written in the scanned text, without its quotes.
    pub span: Option<Range<usize>>,
}

/// What a scan found.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Scan {
    pub tags: Vec<Tag>,
    /// The raw text of each `<style>` and `<script>` element, with its element name.
    pub raw: Vec<(String, String)>,
}

impl Tag {
    pub fn attr(&self, name: &str) -> Option<&Attr> {
        self.attrs.iter().find(|a| a.name == name)
    }
}

/// Scans `text` for tags. Comments, declarations, processing instructions and CDATA sections are
/// skipped; a `<` that starts none of these is text.
pub fn scan(text: &str) -> Scan {
    let bytes = text.as_bytes();
    let mut scan = Scan::default();
    let mut i = 0;
    while let Some(offset) = text[i..].find('<') {
        let at = i + offset;
        let rest = &text[at..];
        if rest.starts_with("<!--") {
            i = rest.find("-->").map_or(text.len(), |end| at + end + 3);
            continue;
        }
        if rest.starts_with("<![CDATA[") {
            i = rest.find("]]>").map_or(text.len(), |end| at + end + 3);
            continue;
        }
        if rest.starts_with("<!") || rest.starts_with("<?") {
            i = rest.find('>').map_or(text.len(), |end| at + end + 1);
            continue;
        }
        let closing = rest.starts_with("</");
        let name_start = at + if closing { 2 } else { 1 };
        let name_len = text[name_start..]
            .bytes()
            .take_while(|b| b.is_ascii_alphanumeric() || matches!(b, b':' | b'_' | b'-'))
            .count();
        if name_len == 0 || !bytes[name_start].is_ascii_alphabetic() {
            i = at + 1;
            continue;
        }
        let name = text[name_start..name_start + name_len].to_ascii_lowercase();
        let (attrs, end, self_closing) = attributes(text, name_start + name_len);
        i = end;
        if !closing && !self_closing && matches!(name.as_str(), "style" | "script") {
            let close = format!("</{name}");
            let content_end = text[i..]
                .to_ascii_lowercase()
                .find(&close)
                .map_or(text.len(), |n| i + n);
            scan.raw
                .push((name.clone(), text[i..content_end].to_owned()));
            i = content_end;
        }
        scan.tags.push(Tag {
            name,
            closing,
            attrs,
        });
    }
    scan
}

/// Reads attributes from `from` to the end of the tag. Returns them, the index after `>`, and
/// whether the tag ends with `/>`.
fn attributes(text: &str, from: usize) -> (Vec<Attr>, usize, bool) {
    let bytes = text.as_bytes();
    let mut attrs = Vec::new();
    let mut i = from;
    let mut self_closing = false;
    loop {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            return (attrs, i, self_closing);
        }
        match bytes[i] {
            b'>' => return (attrs, i + 1, self_closing),
            b'/' => {
                self_closing = true;
                i += 1;
                continue;
            }
            _ => self_closing = false,
        }
        let start = i;
        while i < bytes.len()
            && !bytes[i].is_ascii_whitespace()
            && !matches!(bytes[i], b'=' | b'>' | b'/')
        {
            i += 1;
        }
        let name = text[start..i].to_ascii_lowercase();
        let mut j = i;
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        if j < bytes.len() && bytes[j] == b'=' {
            j += 1;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            let (span, next) = match bytes.get(j) {
                Some(&quote @ (b'"' | b'\'')) => {
                    let end = text[j + 1..]
                        .find(quote as char)
                        .map_or(text.len(), |n| j + 1 + n);
                    (j + 1..end, (end + 1).min(text.len()))
                }
                _ => {
                    let mut end = j;
                    while end < bytes.len()
                        && !bytes[end].is_ascii_whitespace()
                        && bytes[end] != b'>'
                    {
                        end += 1;
                    }
                    (j..end, end)
                }
            };
            attrs.push(Attr {
                name,
                value: Some(decode_entities(&text[span.clone()])),
                span: Some(span),
            });
            i = next;
        } else {
            if !name.is_empty() {
                attrs.push(Attr {
                    name,
                    value: None,
                    span: None,
                });
            }
            if i == start {
                i += 1;
            }
        }
    }
}

/// Replaces attribute values: each span of `text` with its new value, escaped for an attribute.
/// A value whose span was single-quoted or unquoted is escaped the same way, which is valid in
/// either.
pub fn rewrite(text: &str, mut edits: Vec<(Range<usize>, String)>) -> String {
    edits.sort_by_key(|(span, _)| span.start);
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    for (span, value) in edits {
        out.push_str(&text[at..span.start]);
        out.push_str(&escape(&value).replace('\'', "&#39;"));
        at = span.end;
    }
    out.push_str(&text[at..]);
    out
}
