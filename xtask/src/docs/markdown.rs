// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Markdown renderer of the docs site (PROD-E13-T01): the subset of CommonMark and GitHub's
//! extensions that the pages under `docs/` use, rendered the way GitHub renders them, so a page
//! reads the same on the forge and on the site.
//!
//! Blocks: ATX and setext headings, paragraphs, fenced and indented code, bullet and ordered
//! lists (nested, tight or loose), block quotes, thematic breaks, pipe tables, HTML blocks (passed
//! through) and HTML comments (dropped). Inlines: backslash escapes, code spans, emphasis and
//! strong emphasis, inline and reference links and images, autolinks in angle brackets and bare
//! `http(s)://` URLs, raw inline HTML and hard line breaks.
//!
//! Each heading gets the `id` GitHub gives it ([`slug`]), so a link written against the forge's
//! rendering, such as `HOW_IT_WORKS.md#the-joint-placement`, finds the same anchor on the site.
//! Link targets are written as the source gives them; the site build resolves and checks them
//! afterwards, in the HTML, together with the links of raw HTML blocks.

use std::collections::BTreeMap;

/// A rendered Markdown document.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Document {
    /// The body HTML.
    pub html: String,
    /// The headings, in document order.
    pub headings: Vec<Heading>,
}

/// One heading of a document.
#[derive(Debug, PartialEq, Eq)]
pub struct Heading {
    pub level: u8,
    /// The heading's text, with markup removed.
    pub text: String,
    /// The anchor: GitHub's slug of the text, made unique within the document.
    pub id: String,
}

/// Renders `source` to HTML.
pub fn render(source: &str) -> Document {
    let lines: Vec<String> = source
        .lines()
        .map(|line| expand_tabs(line.trim_end_matches('\r')))
        .collect();
    let (lines, refs) = take_reference_definitions(lines);
    let mut renderer = Renderer {
        refs,
        seen: BTreeMap::new(),
        headings: Vec::new(),
    };
    let mut html = String::new();
    renderer.blocks(&lines, false, &mut html);
    Document {
        html,
        headings: renderer.headings,
    }
}

/// GitHub's anchor for a heading's text: lower case, every character dropped that is not a letter,
/// a digit, `-`, `_` or a space, and each space turned into `-`. A repeated slug gets `-1`, `-2`
/// and so on, which the renderer adds.
pub fn slug(text: &str) -> String {
    text.chars()
        .flat_map(char::to_lowercase)
        .filter(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | ' '))
        .map(|c| if c == ' ' { '-' } else { c })
        .collect()
}

/// `text` with tags removed and the common character references decoded.
pub fn plain(html: &str) -> String {
    let mut text = String::new();
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => in_tag = true,
            '>' if in_tag => in_tag = false,
            _ if !in_tag => text.push(c),
            _ => {}
        }
    }
    decode_entities(&text)
}

/// Decodes `&amp;`, `&lt;`, `&gt;`, `&quot;`, `&apos;`, `&nbsp;` and numeric references. Others are
/// kept as written.
pub fn decode_entities(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        rest = &rest[at..];
        let decoded = rest.find(';').filter(|end| *end <= 12).and_then(|end| {
            let name = &rest[1..end];
            let c = match name {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" => Some('\''),
                "nbsp" => Some('\u{a0}'),
                _ => name
                    .strip_prefix("#x")
                    .or_else(|| name.strip_prefix("#X"))
                    .map(|hex| u32::from_str_radix(hex, 16))
                    .or_else(|| name.strip_prefix('#').map(str::parse::<u32>))
                    .and_then(Result::ok)
                    .and_then(char::from_u32),
            };
            c.map(|c| (c, end))
        });
        match decoded {
            Some((c, end)) => {
                out.push(c);
                rest = &rest[end + 1..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// Escapes text for an HTML text node or a double-quoted attribute value.
pub fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

fn expand_tabs(line: &str) -> String {
    if !line.contains('\t') {
        return line.to_owned();
    }
    let mut out = String::new();
    let mut column = 0;
    for c in line.chars() {
        if c == '\t' {
            let width = 4 - column % 4;
            out.extend(std::iter::repeat_n(' ', width));
            column += width;
        } else {
            out.push(c);
            column += 1;
        }
    }
    out
}

/// A link reference definition's target: the destination and its title.
type Reference = (String, Option<String>);

/// Removes the link reference definitions (`[label]: <dest> "title"`) outside fenced code, and
/// returns them by normalised label.
fn take_reference_definitions(lines: Vec<String>) -> (Vec<String>, BTreeMap<String, Reference>) {
    let mut refs = BTreeMap::new();
    let mut kept = Vec::with_capacity(lines.len());
    let mut fence: Option<(char, usize)> = None;
    let mut previous_blank = true;
    for line in lines {
        if let Some((c, n)) = fence {
            if is_closing_fence(&line, c, n) {
                fence = None;
            }
            kept.push(line);
            continue;
        }
        if let Some(open) = fence_open(&line) {
            fence = Some((open.marker, open.length));
            previous_blank = false;
            kept.push(line);
            continue;
        }
        if previous_blank
            && indent(&line) < 4
            && let Some((label, reference)) = reference_definition(line.trim())
        {
            refs.entry(label).or_insert(reference);
            continue;
        }
        previous_blank = line.trim().is_empty();
        kept.push(line);
    }
    (kept, refs)
}

fn reference_definition(line: &str) -> Option<(String, Reference)> {
    let rest = line.strip_prefix('[')?;
    let close = rest.find("]:")?;
    let label = normalise_label(&rest[..close]);
    if label.is_empty() || rest[..close].contains(['[', ']']) {
        return None;
    }
    let rest = rest[close + 2..].trim();
    let (dest, rest) = if let Some(inner) = rest.strip_prefix('<') {
        let end = inner.find('>')?;
        (&inner[..end], inner[end + 1..].trim())
    } else {
        let end = rest.find(' ').unwrap_or(rest.len());
        (&rest[..end], rest[end..].trim())
    };
    if dest.is_empty() {
        return None;
    }
    let title = if rest.is_empty() {
        None
    } else {
        let quoted = (rest.len() >= 2)
            .then(|| (rest.as_bytes()[0], rest.as_bytes()[rest.len() - 1]))
            .filter(|(a, b)| matches!((a, b), (b'"', b'"') | (b'\'', b'\'') | (b'(', b')')));
        quoted?;
        Some(rest[1..rest.len() - 1].to_owned())
    };
    Some((label, (unescape(dest), title)))
}

fn normalise_label(label: &str) -> String {
    label
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Removes backslash escapes of ASCII punctuation.
fn unescape(text: &str) -> String {
    let mut out = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\'
            && let Some(next) = chars.peek().copied()
            && next.is_ascii_punctuation()
        {
            out.push(next);
            chars.next();
            continue;
        }
        out.push(c);
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Block structure.

fn indent(line: &str) -> usize {
    line.len() - line.trim_start_matches(' ').len()
}

fn is_blank(line: &str) -> bool {
    line.trim().is_empty()
}

struct Fence {
    indent: usize,
    marker: char,
    length: usize,
    info: String,
}

fn fence_open(line: &str) -> Option<Fence> {
    let indent = indent(line);
    if indent > 3 {
        return None;
    }
    let rest = &line[indent..];
    let marker = rest.chars().next().filter(|c| matches!(c, '`' | '~'))?;
    let length = rest.len() - rest.trim_start_matches(marker).len();
    if length < 3 {
        return None;
    }
    let info = rest[length..].trim();
    if marker == '`' && info.contains('`') {
        return None;
    }
    Some(Fence {
        indent,
        marker,
        length,
        info: info.split_whitespace().next().unwrap_or("").to_owned(),
    })
}

fn is_closing_fence(line: &str, marker: char, length: usize) -> bool {
    let indent = indent(line);
    let rest = line[indent..].trim_end();
    indent <= 3
        && rest.len() >= length
        && rest.chars().all(|c| c == marker)
        && rest.starts_with(marker)
}

fn atx_heading(line: &str) -> Option<(u8, &str)> {
    let indent = indent(line);
    if indent > 3 {
        return None;
    }
    let rest = &line[indent..];
    let level = rest.len() - rest.trim_start_matches('#').len();
    if !(1..=6).contains(&level) {
        return None;
    }
    let text = &rest[level..];
    if !text.is_empty() && !text.starts_with(' ') {
        return None;
    }
    let mut text = text.trim();
    // A closing sequence of `#`s, preceded by a space or alone, is not part of the text.
    let without = text.trim_end_matches('#');
    if without.is_empty() || without.ends_with(' ') {
        text = without.trim_end();
    }
    Some((level as u8, text))
}

fn is_thematic_break(line: &str) -> bool {
    if indent(line) > 3 {
        return false;
    }
    let chars: Vec<char> = line.chars().filter(|c| *c != ' ').collect();
    chars.len() >= 3 && matches!(chars[0], '-' | '*' | '_') && chars.iter().all(|c| *c == chars[0])
}

fn setext_level(line: &str) -> Option<u8> {
    if indent(line) > 3 {
        return None;
    }
    let text = line.trim();
    if !text.is_empty() && text.chars().all(|c| c == '=') {
        Some(1)
    } else if !text.is_empty() && text.chars().all(|c| c == '-') {
        Some(2)
    } else {
        None
    }
}

fn blockquote_content(line: &str) -> Option<&str> {
    let indent = indent(line);
    if indent > 3 {
        return None;
    }
    let rest = line[indent..].strip_prefix('>')?;
    Some(rest.strip_prefix(' ').unwrap_or(rest))
}

/// The block-level tag names that start an HTML block (CommonMark's kind 6), and `picture`.
const BLOCK_TAGS: &[&str] = &[
    "address",
    "article",
    "aside",
    "blockquote",
    "details",
    "dialog",
    "dd",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "picture",
    "section",
    "summary",
    "table",
    "tbody",
    "td",
    "th",
    "thead",
    "tr",
    "ul",
];

fn html_block_start(line: &str) -> bool {
    if indent(line) > 3 {
        return false;
    }
    let rest = line.trim_start();
    let Some(tag) = rest.strip_prefix("</").or_else(|| rest.strip_prefix('<')) else {
        return false;
    };
    let name: String = tag
        .chars()
        .take_while(char::is_ascii_alphanumeric)
        .collect::<String>()
        .to_ascii_lowercase();
    let after = tag[name.len()..].chars().next();
    BLOCK_TAGS.contains(&name.as_str()) && matches!(after, None | Some(' ' | '>' | '/'))
}

fn comment_start(line: &str) -> bool {
    indent(line) <= 3 && line.trim_start().starts_with("<!--")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ListKind {
    Bullet(char),
    Ordered(char),
}

struct Marker {
    kind: ListKind,
    start: u64,
    /// The column where the item's content starts.
    content: usize,
    /// Whether the marker is followed by nothing.
    empty: bool,
}

fn list_marker(line: &str) -> Option<Marker> {
    let indent = indent(line);
    if indent > 3 {
        return None;
    }
    let rest = &line[indent..];
    let (kind, start, width) = match rest.chars().next()? {
        c @ ('-' | '*' | '+') => (ListKind::Bullet(c), 1, 1),
        _ => {
            let digits = rest.len() - rest.trim_start_matches(|c: char| c.is_ascii_digit()).len();
            if digits == 0 || digits > 9 {
                return None;
            }
            let delimiter = rest[digits..].chars().next()?;
            if !matches!(delimiter, '.' | ')') {
                return None;
            }
            let start = rest[..digits].parse().ok()?;
            (ListKind::Ordered(delimiter), start, digits + 1)
        }
    };
    let after = &rest[width..];
    if after.trim().is_empty() {
        return Some(Marker {
            kind,
            start,
            content: indent + width + 1,
            empty: true,
        });
    }
    let spaces = indent_of(after);
    if spaces == 0 {
        return None;
    }
    let spaces = if spaces > 4 { 1 } else { spaces };
    Some(Marker {
        kind,
        start,
        content: indent + width + spaces,
        empty: false,
    })
}

fn indent_of(text: &str) -> usize {
    text.len() - text.trim_start_matches(' ').len()
}

fn table_delimiter(line: &str) -> Option<Vec<Option<&'static str>>> {
    let cells = split_cells(line);
    if cells.is_empty() || !line.contains('-') {
        return None;
    }
    cells
        .iter()
        .map(|cell| {
            let cell = cell.trim();
            let left = cell.starts_with(':');
            let right = cell.ends_with(':');
            let dashes = cell.trim_matches(':');
            if dashes.is_empty() || !dashes.chars().all(|c| c == '-') {
                return None;
            }
            Some(match (left, right) {
                (true, true) => Some("center"),
                (true, false) => Some("left"),
                (false, true) => Some("right"),
                (false, false) => None,
            })
        })
        .collect()
}

/// A table row's cells, split on `|` that is neither escaped nor inside a code span. `\|` stays
/// in the cell as `|`.
fn split_cells(line: &str) -> Vec<String> {
    let mut text = line.trim();
    if let Some(rest) = text.strip_prefix('|') {
        text = rest;
    }
    if text.ends_with('|') && !text.ends_with("\\|") {
        text = &text[..text.len() - 1];
    }
    let mut cells = Vec::new();
    let mut cell = String::new();
    let mut chars = text.chars().peekable();
    let mut code: Option<usize> = None;
    while let Some(c) = chars.next() {
        match c {
            '\\' if chars.peek() == Some(&'|') => {
                cell.push('|');
                chars.next();
            }
            '`' => {
                let mut run = 1;
                while chars.peek() == Some(&'`') {
                    chars.next();
                    run += 1;
                }
                code = match code {
                    Some(open) if open == run => None,
                    None => Some(run),
                    other => other,
                };
                cell.extend(std::iter::repeat_n('`', run));
            }
            '|' if code.is_none() => cells.push(std::mem::take(&mut cell)),
            _ => cell.push(c),
        }
    }
    cells.push(cell);
    cells
}

struct Renderer {
    refs: BTreeMap<String, Reference>,
    /// How many times each slug has been given, to make repeats unique.
    seen: BTreeMap<String, usize>,
    headings: Vec<Heading>,
}

impl Renderer {
    /// Whether `line` ends a paragraph by starting another block.
    fn interrupts(line: &str) -> bool {
        is_blank(line)
            || atx_heading(line).is_some()
            || fence_open(line).is_some()
            || is_thematic_break(line)
            || blockquote_content(line).is_some()
            || html_block_start(line)
            || comment_start(line)
            || list_marker(line).is_some_and(|m| {
                !m.empty && (matches!(m.kind, ListKind::Bullet(_)) || m.start == 1)
            })
    }

    fn heading(&mut self, level: u8, text: &str, out: &mut String) {
        let inner = self.inline(text);
        let text = plain(&inner).trim().to_owned();
        let base = slug(&text);
        let count = self.seen.entry(base.clone()).or_insert(0);
        let id = if *count == 0 {
            base.clone()
        } else {
            format!("{base}-{count}")
        };
        *count += 1;
        // A generated id can itself be taken by a later plain heading; GitHub has the same
        // behaviour, so the slug is kept as GitHub gives it.
        out.push_str(&format!(
            "<h{level} id=\"{}\">{inner}</h{level}>\n",
            escape(&id)
        ));
        self.headings.push(Heading { level, text, id });
    }

    /// Renders a sequence of block lines. In a `tight` list item, paragraphs have no `<p>`.
    fn blocks(&mut self, lines: &[String], tight: bool, out: &mut String) {
        let mut i = 0;
        while i < lines.len() {
            let line = &lines[i];
            if is_blank(line) {
                i += 1;
                continue;
            }
            if let Some(fence) = fence_open(line) {
                i += 1;
                let mut code = String::new();
                while i < lines.len() && !is_closing_fence(&lines[i], fence.marker, fence.length) {
                    let content = &lines[i];
                    let strip = indent(content).min(fence.indent);
                    code.push_str(&content[strip..]);
                    code.push('\n');
                    i += 1;
                }
                i += 1;
                let class = if fence.info.is_empty() {
                    String::new()
                } else {
                    format!(" class=\"language-{}\"", escape(&unescape(&fence.info)))
                };
                out.push_str(&format!(
                    "<pre><code{class}>{}</code></pre>\n",
                    escape(&code)
                ));
                continue;
            }
            if indent(line) >= 4 {
                let mut code = Vec::new();
                while i < lines.len() && (indent(&lines[i]) >= 4 || is_blank(&lines[i])) {
                    code.push(lines[i].get(4..).unwrap_or(""));
                    i += 1;
                }
                while code.last().is_some_and(|l| l.trim().is_empty()) {
                    code.pop();
                }
                let mut text = code.join("\n");
                text.push('\n');
                out.push_str(&format!("<pre><code>{}</code></pre>\n", escape(&text)));
                continue;
            }
            if let Some((level, text)) = atx_heading(line) {
                self.heading(level, text, out);
                i += 1;
                continue;
            }
            if comment_start(line) {
                while i < lines.len() && !lines[i].contains("-->") {
                    i += 1;
                }
                i += 1;
                continue;
            }
            if html_block_start(line) {
                while i < lines.len() && !is_blank(&lines[i]) {
                    out.push_str(&lines[i]);
                    out.push('\n');
                    i += 1;
                }
                continue;
            }
            if is_thematic_break(line) {
                out.push_str("<hr>\n");
                i += 1;
                continue;
            }
            if blockquote_content(line).is_some() {
                let mut inner = Vec::new();
                while i < lines.len() {
                    if let Some(content) = blockquote_content(&lines[i]) {
                        inner.push(content.to_owned());
                    } else if !is_blank(&lines[i])
                        && !Self::interrupts(&lines[i])
                        && inner.last().is_some_and(|l: &String| !is_blank(l))
                    {
                        inner.push(lines[i].clone());
                    } else {
                        break;
                    }
                    i += 1;
                }
                out.push_str("<blockquote>\n");
                self.blocks(&inner, false, out);
                out.push_str("</blockquote>\n");
                continue;
            }
            if let Some(marker) = list_marker(line) {
                i = self.list(lines, i, marker, out);
                continue;
            }
            if line.contains('|')
                && let Some(next) = lines.get(i + 1)
                && let Some(aligns) = table_delimiter(next)
                && split_cells(line).len() == aligns.len()
            {
                i = self.table(lines, i, &aligns, out);
                continue;
            }
            // A paragraph, or a setext heading.
            let mut text = vec![line.trim_start().to_owned()];
            i += 1;
            let mut level = None;
            while i < lines.len() {
                if let Some(l) = setext_level(&lines[i]) {
                    level = Some(l);
                    i += 1;
                    break;
                }
                if Self::interrupts(&lines[i]) {
                    break;
                }
                text.push(lines[i].trim_start().to_owned());
                i += 1;
            }
            let joined = text.join("\n");
            match level {
                Some(level) => self.heading(level, joined.trim(), out),
                None => {
                    let inner = self.inline(joined.trim_end());
                    if tight {
                        out.push_str(&inner);
                        out.push('\n');
                    } else {
                        out.push_str(&format!("<p>{inner}</p>\n"));
                    }
                }
            }
        }
    }

    /// Renders the list that starts at `lines[start]`, and returns the index after it.
    fn list(&mut self, lines: &[String], start: usize, first: Marker, out: &mut String) -> usize {
        let mut items: Vec<Vec<String>> = Vec::new();
        let mut loose = false;
        let mut i = start;
        let mut marker = first;
        let kind = marker.kind;
        let number = marker.start;
        loop {
            let width = marker.content;
            let head = &lines[i];
            let mut item = vec![if marker.empty {
                String::new()
            } else {
                head.get(width..).unwrap_or("").to_owned()
            }];
            i += 1;
            let mut blanks = 0;
            let mut in_fence: Option<(char, usize)> = None;
            if let Some(f) = fence_open(&item[0]) {
                in_fence = Some((f.marker, f.length));
            }
            while i < lines.len() {
                let line = &lines[i];
                if is_blank(line) {
                    blanks += 1;
                    i += 1;
                    continue;
                }
                if indent(line) >= width {
                    if blanks > 0 && in_fence.is_none() && item.iter().any(|l| !is_blank(l)) {
                        // A blank line between two blocks of one item makes the list loose,
                        // unless the next block is a nested list (its own looseness).
                        let content = &line[width..];
                        if list_marker(content).is_none() || indent(content) > 0 {
                            loose = true;
                        }
                    }
                    item.extend(std::iter::repeat_n(String::new(), blanks));
                    blanks = 0;
                    let content = line[width..].to_owned();
                    match in_fence {
                        Some((c, n)) if is_closing_fence(&content, c, n) => in_fence = None,
                        None => {
                            if let Some(f) = fence_open(&content) {
                                in_fence = Some((f.marker, f.length));
                            }
                        }
                        _ => {}
                    }
                    item.push(content);
                    i += 1;
                    continue;
                }
                if blanks == 0
                    && in_fence.is_none()
                    && !Self::interrupts(line)
                    && list_marker(line).is_none()
                {
                    // A lazy continuation line of the item's paragraph.
                    item.push(line.trim_start().to_owned());
                    i += 1;
                    continue;
                }
                break;
            }
            items.push(item);
            match lines.get(i).and_then(|l| list_marker(l)) {
                Some(next) if next.kind == kind && !is_thematic_break(&lines[i]) => {
                    if blanks > 0 {
                        loose = true;
                    }
                    marker = next;
                }
                _ => break,
            }
        }
        let (open, close) = match kind {
            ListKind::Bullet(_) => ("<ul>".to_owned(), "</ul>"),
            ListKind::Ordered(_) if number != 1 => (format!("<ol start=\"{number}\">"), "</ol>"),
            ListKind::Ordered(_) => ("<ol>".to_owned(), "</ol>"),
        };
        out.push_str(&open);
        out.push('\n');
        for item in &items {
            out.push_str("<li>");
            let mut inner = String::new();
            self.blocks(item, !loose, &mut inner);
            if !loose {
                out.push_str(inner.trim_end_matches('\n'));
            } else {
                out.push('\n');
                out.push_str(&inner);
            }
            out.push_str("</li>\n");
        }
        out.push_str(close);
        out.push('\n');
        i
    }

    /// Renders the pipe table whose header is `lines[start]`, and returns the index after it.
    fn table(
        &mut self,
        lines: &[String],
        start: usize,
        aligns: &[Option<&'static str>],
        out: &mut String,
    ) -> usize {
        let cell = |this: &mut Self, tag: &str, index: usize, text: &str| {
            let align = aligns
                .get(index)
                .copied()
                .flatten()
                .map_or(String::new(), |a| format!(" align=\"{a}\""));
            format!("<{tag}{align}>{}</{tag}>", this.inline(text.trim()))
        };
        out.push_str("<table>\n<thead>\n<tr>\n");
        for (index, text) in split_cells(&lines[start]).iter().enumerate() {
            out.push_str(&cell(self, "th", index, text));
            out.push('\n');
        }
        out.push_str("</tr>\n</thead>\n");
        let mut i = start + 2;
        let mut body = false;
        while i < lines.len() && !Self::interrupts(&lines[i]) {
            if !body {
                out.push_str("<tbody>\n");
                body = true;
            }
            let cells = split_cells(&lines[i]);
            out.push_str("<tr>\n");
            for index in 0..aligns.len() {
                let text = cells.get(index).map_or("", String::as_str);
                out.push_str(&cell(self, "td", index, text));
                out.push('\n');
            }
            out.push_str("</tr>\n");
            i += 1;
        }
        if body {
            out.push_str("</tbody>\n");
        }
        out.push_str("</table>\n");
        i
    }

    // -----------------------------------------------------------------------------------------
    // Inlines.

    fn inline(&self, text: &str) -> String {
        let chars: Vec<char> = text.chars().collect();
        let mut nodes: Vec<Node> = Vec::new();
        let mut buffer = String::new();
        let flush = |nodes: &mut Vec<Node>, buffer: &mut String| {
            if !buffer.is_empty() {
                nodes.push(Node::Html(std::mem::take(buffer)));
            }
        };
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            match c {
                '\\' => match chars.get(i + 1) {
                    Some('\n') => {
                        buffer.push_str("<br>\n");
                        i += 2;
                    }
                    Some(next) if next.is_ascii_punctuation() => {
                        buffer.push_str(&escape(&next.to_string()));
                        i += 2;
                    }
                    _ => {
                        buffer.push('\\');
                        i += 1;
                    }
                },
                '`' => {
                    let run = count_run(&chars, i, '`');
                    match find_closing_backticks(&chars, i + run, run) {
                        Some(end) => {
                            let content: String = chars[i + run..end]
                                .iter()
                                .map(|c| if *c == '\n' { ' ' } else { *c })
                                .collect();
                            let content = if content.len() >= 2
                                && content.starts_with(' ')
                                && content.ends_with(' ')
                                && !content.trim().is_empty()
                            {
                                &content[1..content.len() - 1]
                            } else {
                                content.as_str()
                            };
                            buffer.push_str(&format!("<code>{}</code>", escape(content)));
                            i = end + run;
                        }
                        None => {
                            buffer.extend(std::iter::repeat_n('`', run));
                            i += run;
                        }
                    }
                }
                '*' | '_' => {
                    let run = count_run(&chars, i, c);
                    let before = if i == 0 { None } else { Some(chars[i - 1]) };
                    let after = chars.get(i + run).copied();
                    let left = flanking(after, before);
                    let right = flanking(before, after);
                    let punct = |c: Option<char>| c.is_some_and(is_punctuation);
                    let (open, close) = if c == '*' {
                        (left, right)
                    } else {
                        (
                            left && (!right || punct(before)),
                            right && (!left || punct(after)),
                        )
                    };
                    flush(&mut nodes, &mut buffer);
                    nodes.push(Node::Delim {
                        ch: c,
                        count: run,
                        original: run,
                        open,
                        close,
                    });
                    i += run;
                }
                '!' if chars.get(i + 1) == Some(&'[') => match self.link(&chars, i + 1, true) {
                    Some((html, end)) => {
                        buffer.push_str(&html);
                        i = end;
                    }
                    None => {
                        buffer.push('!');
                        i += 1;
                    }
                },
                '[' => match self.link(&chars, i, false) {
                    Some((html, end)) => {
                        buffer.push_str(&html);
                        i = end;
                    }
                    None => {
                        buffer.push('[');
                        i += 1;
                    }
                },
                '<' => match angle(&chars, i) {
                    Some((html, end)) => {
                        buffer.push_str(&html);
                        i = end;
                    }
                    None => {
                        buffer.push_str("&lt;");
                        i += 1;
                    }
                },
                '&' => {
                    let rest: String = chars[i..chars.len().min(i + 34)].iter().collect();
                    match entity_length(&rest) {
                        Some(n) => {
                            buffer.push_str(&rest[..n]);
                            i += rest[..n].chars().count();
                        }
                        None => {
                            buffer.push_str("&amp;");
                            i += 1;
                        }
                    }
                }
                '>' => {
                    buffer.push_str("&gt;");
                    i += 1;
                }
                '"' => {
                    buffer.push_str("&quot;");
                    i += 1;
                }
                '\n' => {
                    let kept = buffer.trim_end_matches(' ').len();
                    let hard = buffer.len() - kept >= 2;
                    buffer.truncate(kept);
                    buffer.push_str(if hard { "<br>\n" } else { "\n" });
                    i += 1;
                }
                'h' if (i == 0 || matches!(chars[i - 1], ' ' | '\n' | '(' | '*' | '_' | '~'))
                    && starts_with(&chars, i, "http") =>
                {
                    match bare_url(&chars, i) {
                        Some(end) => {
                            let url: String = chars[i..end].iter().collect();
                            buffer.push_str(&format!("<a href=\"{0}\">{0}</a>", escape(&url)));
                            i = end;
                        }
                        None => {
                            buffer.push('h');
                            i += 1;
                        }
                    }
                }
                _ => {
                    buffer.push(c);
                    i += 1;
                }
            }
        }
        flush(&mut nodes, &mut buffer);
        emphasis(nodes)
    }

    /// Parses a link or image whose `[` is at `chars[open]`. Returns its HTML and the index after
    /// it.
    fn link(&self, chars: &[char], open: usize, image: bool) -> Option<(String, usize)> {
        let close = closing_bracket(chars, open)?;
        let text: String = chars[open + 1..close].iter().collect();
        let (dest, title, end) = if chars.get(close + 1) == Some(&'(') {
            inline_destination(chars, close + 2)?
        } else {
            let (label, end) = if chars.get(close + 1) == Some(&'[') {
                let label_close = chars[close + 2..].iter().position(|c| *c == ']')? + close + 2;
                let label: String = chars[close + 2..label_close].iter().collect();
                let label = if label.trim().is_empty() {
                    text.clone()
                } else {
                    label
                };
                (label, label_close + 1)
            } else {
                (text.clone(), close + 1)
            };
            let (dest, title) = self.refs.get(&normalise_label(&label))?.clone();
            (dest, title, end)
        };
        let title = title.map_or(String::new(), |t| format!(" title=\"{}\"", escape(&t)));
        let html = if image {
            format!(
                "<img src=\"{}\" alt=\"{}\"{title}>",
                escape(&dest),
                escape(&plain(&self.inline(&text)))
            )
        } else {
            format!(
                "<a href=\"{}\"{title}>{}</a>",
                escape(&dest),
                self.inline(&text)
            )
        };
        Some((html, end))
    }
}

/// An inline node before emphasis is resolved: rendered HTML, or a run of `*` or `_`.
enum Node {
    Html(String),
    Delim {
        ch: char,
        count: usize,
        original: usize,
        open: bool,
        close: bool,
    },
}

/// CommonMark's emphasis algorithm over the delimiter runs, then the nodes joined.
fn emphasis(mut nodes: Vec<Node>) -> String {
    let mut i = 0;
    while i < nodes.len() {
        let (ch, closer_count, closer_original, closer_open) = match nodes[i] {
            Node::Delim {
                ch,
                count,
                original,
                open,
                close: true,
            } if count > 0 => (ch, count, original, open),
            _ => {
                i += 1;
                continue;
            }
        };
        let mut found = None;
        for j in (0..i).rev() {
            if let Node::Delim {
                ch: c,
                count,
                original,
                open: true,
                close,
            } = nodes[j]
                && c == ch
                && count > 0
            {
                let both = close || closer_open;
                if both
                    && (original + closer_original) % 3 == 0
                    && !(original % 3 == 0 && closer_original % 3 == 0)
                {
                    continue;
                }
                found = Some((j, count));
                break;
            }
        }
        let Some((j, opener_count)) = found else {
            if !closer_open && let Node::Delim { open, close, .. } = &mut nodes[i] {
                *open = false;
                *close = false;
            }
            i += 1;
            continue;
        };
        let used = if opener_count >= 2 && closer_count >= 2 {
            2
        } else {
            1
        };
        let tag = if used == 2 { "strong" } else { "em" };
        for (index, reduce) in [(j, used), (i, used)] {
            if let Node::Delim { count, .. } = &mut nodes[index] {
                *count -= reduce;
            }
        }
        // Delimiters between the two can no longer match anything across this pair.
        for node in &mut nodes[j + 1..i] {
            if let Node::Delim { open, close, .. } = node {
                *open = false;
                *close = false;
            }
        }
        nodes.insert(i, Node::Html(format!("</{tag}>")));
        nodes.insert(j + 1, Node::Html(format!("<{tag}>")));
        i += 2;
    }
    let mut out = String::new();
    for node in nodes {
        match node {
            Node::Html(html) => out.push_str(&html),
            Node::Delim { ch, count, .. } => out.extend(std::iter::repeat_n(ch, count)),
        }
    }
    out
}

fn is_punctuation(c: char) -> bool {
    c.is_ascii_punctuation() || (!c.is_alphanumeric() && !c.is_whitespace() && !c.is_control())
}

/// Whether a delimiter run with `near` on the side it opens towards and `far` on the other side
/// is flanking on that side (CommonMark's left- and right-flanking rules).
fn flanking(near: Option<char>, far: Option<char>) -> bool {
    let white = |c: Option<char>| c.is_none_or(char::is_whitespace);
    let punct = |c: Option<char>| c.is_some_and(is_punctuation);
    !white(near) && (!punct(near) || white(far) || punct(far))
}

fn count_run(chars: &[char], at: usize, c: char) -> usize {
    chars[at..].iter().take_while(|x| **x == c).count()
}

fn starts_with(chars: &[char], at: usize, prefix: &str) -> bool {
    prefix
        .chars()
        .enumerate()
        .all(|(k, p)| chars.get(at + k) == Some(&p))
}

fn find_closing_backticks(chars: &[char], from: usize, run: usize) -> Option<usize> {
    let mut i = from;
    while i < chars.len() {
        if chars[i] == '`' {
            let n = count_run(chars, i, '`');
            if n == run {
                return Some(i);
            }
            i += n;
        } else {
            i += 1;
        }
    }
    None
}

/// The `]` that closes the `[` at `open`, skipping escapes, code spans and nested brackets.
fn closing_bracket(chars: &[char], open: usize) -> Option<usize> {
    let mut depth = 0;
    let mut i = open;
    while i < chars.len() {
        match chars[i] {
            '\\' => i += 1,
            '`' => {
                let run = count_run(chars, i, '`');
                if let Some(end) = find_closing_backticks(chars, i + run, run) {
                    i = end + run;
                    continue;
                }
                i += run;
                continue;
            }
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Parses `dest "title")` from `chars[from]`, the character after `(`.
fn inline_destination(chars: &[char], from: usize) -> Option<(String, Option<String>, usize)> {
    let mut i = from;
    let skip = |i: &mut usize| {
        while chars.get(*i).is_some_and(|c| c.is_whitespace()) {
            *i += 1;
        }
    };
    skip(&mut i);
    let mut dest = String::new();
    if chars.get(i) == Some(&'<') {
        i += 1;
        while let Some(c) = chars.get(i).copied() {
            match c {
                '>' => break,
                '\n' | '<' => return None,
                '\\' if chars.get(i + 1).is_some_and(char::is_ascii_punctuation) => {
                    dest.push(chars[i + 1]);
                    i += 2;
                    continue;
                }
                _ => dest.push(c),
            }
            i += 1;
        }
        if chars.get(i) != Some(&'>') {
            return None;
        }
        i += 1;
    } else {
        let mut depth = 0usize;
        while let Some(c) = chars.get(i).copied() {
            match c {
                '\\' if chars.get(i + 1).is_some_and(char::is_ascii_punctuation) => {
                    dest.push(chars[i + 1]);
                    i += 2;
                    continue;
                }
                '(' => depth += 1,
                ')' if depth == 0 => break,
                ')' => depth -= 1,
                c if c.is_whitespace() || c.is_control() => break,
                _ => {}
            }
            dest.push(c);
            i += 1;
        }
    }
    skip(&mut i);
    let mut title = None;
    if let Some(quote) = chars
        .get(i)
        .copied()
        .filter(|c| matches!(c, '"' | '\'' | '('))
    {
        let end_quote = if quote == '(' { ')' } else { quote };
        let mut text = String::new();
        i += 1;
        while let Some(c) = chars.get(i).copied() {
            if c == end_quote {
                break;
            }
            if c == '\\' && chars.get(i + 1).is_some_and(char::is_ascii_punctuation) {
                text.push(chars[i + 1]);
                i += 2;
                continue;
            }
            text.push(c);
            i += 1;
        }
        if chars.get(i) != Some(&end_quote) {
            return None;
        }
        i += 1;
        title = Some(text);
        skip(&mut i);
    }
    if chars.get(i) != Some(&')') {
        return None;
    }
    Some((dest, title, i + 1))
}

/// An autolink (`<https://…>`, `<name@host>`) or raw inline HTML starting at the `<` at `at`.
fn angle(chars: &[char], at: usize) -> Option<(String, usize)> {
    let end = chars[at..].iter().position(|c| *c == '>')? + at;
    let inner: String = chars[at + 1..end].iter().collect();
    if inner.starts_with("!--") {
        let text: String = chars[at..].iter().collect();
        let close = text.find("-->")?;
        let comment = &text[..close + 3];
        return Some((comment.to_owned(), at + comment.chars().count()));
    }
    let scheme_end = inner.find(':');
    let is_uri = scheme_end.is_some_and(|n| {
        (2..=32).contains(&n)
            && inner[..n]
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '.' | '-'))
            && inner
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic())
    }) && !inner.contains(|c: char| c.is_whitespace() || c == '<');
    if is_uri {
        return Some((format!("<a href=\"{0}\">{0}</a>", escape(&inner)), end + 1));
    }
    let is_email = inner.contains('@')
        && !inner.contains(|c: char| c.is_whitespace() || c == '<')
        && inner.split('@').count() == 2;
    if is_email {
        return Some((
            format!("<a href=\"mailto:{0}\">{0}</a>", escape(&inner)),
            end + 1,
        ));
    }
    // Raw HTML: an open tag, a closing tag, or a declaration.
    let name = inner.strip_prefix('/').unwrap_or(&inner);
    let starts_tag = name.chars().next().is_some_and(|c| c.is_ascii_alphabetic());
    if starts_tag {
        let raw: String = chars[at..=end].iter().collect();
        return Some((raw, end + 1));
    }
    None
}

/// The length of a character reference at the start of `text`, if it is one.
fn entity_length(text: &str) -> Option<usize> {
    let end = text.find(';')?;
    let name = &text[1..end];
    let ok = if let Some(hex) = name.strip_prefix("#x").or_else(|| name.strip_prefix("#X")) {
        (1..=6).contains(&hex.len()) && hex.chars().all(|c| c.is_ascii_hexdigit())
    } else if let Some(digits) = name.strip_prefix('#') {
        (1..=7).contains(&digits.len()) && digits.chars().all(|c| c.is_ascii_digit())
    } else {
        (2..=32).contains(&name.len())
            && name.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
            && name.chars().all(|c| c.is_ascii_alphanumeric())
    };
    ok.then_some(end + 1)
}

/// The end of a bare `http://` or `https://` URL starting at `at`, as GitHub autolinks it.
fn bare_url(chars: &[char], at: usize) -> Option<usize> {
    let scheme = if starts_with(chars, at, "https://") {
        8
    } else if starts_with(chars, at, "http://") {
        7
    } else {
        return None;
    };
    let mut end = at + scheme;
    while end < chars.len() && !chars[end].is_whitespace() && chars[end] != '<' {
        end += 1;
    }
    // Trailing punctuation ends the sentence, not the URL; so does an unbalanced `)`.
    loop {
        let last = chars[end - 1];
        if matches!(
            last,
            '?' | '!' | '.' | ',' | ':' | '*' | '_' | '~' | '\'' | '"'
        ) {
            end -= 1;
        } else if last == ')' {
            let opens = chars[at..end].iter().filter(|c| **c == '(').count();
            let closes = chars[at..end].iter().filter(|c| **c == ')').count();
            if closes > opens {
                end -= 1;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    (end > at + scheme).then_some(end)
}
