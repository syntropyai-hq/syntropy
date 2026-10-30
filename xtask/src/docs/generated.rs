// SPDX-License-Identifier: MIT OR Apache-2.0

//! The generated-reference hooks of the docs site (PROD-E13-T01), which later tasks fill: the
//! configuration keys, the error codes, the metric catalogue and the OpenAPI document of
//! PROD-E13-T07, and the pages PROD-E13-T06 and PROD-E13-T08 generate.
//!
//! A page under `docs/` holds a generated section between two marker lines, the same markers the
//! plan tools use:
//!
//! ```text
//! <!-- BEGIN GENERATED: config-keys -->
//! …whatever the `config-keys` generator returns…
//! <!-- END GENERATED -->
//! ```
//!
//! A generator is an entry of [`GENERATORS`]: a name, the task that owns it, and a function that
//! returns the section's Markdown from the checkout. The `docs` job's `generated` step fails
//! when a section names no generator, or when its text differs from what the generator returns
//! now, so a page cannot drift from the code it documents. `cargo xtask docs-generate` rewrites
//! every section. A task that adds a generator also adds its source paths to the `docs` rule in
//! `ci/jobs.toml`, so a change to those sources runs the job.

use std::path::Path;

/// A generator of one section.
pub struct Generator {
    /// The name in the markers: lower-case words joined by `-`.
    pub name: &'static str,
    /// The task that adds it, such as `PROD-E13-T07`.
    pub task: &'static str,
    /// Returns the section's Markdown, from the checkout at the root it is given.
    pub run: fn(&Path) -> Result<String, String>,
}

/// The generators. Empty until PROD-E13-T07 adds the first.
pub const GENERATORS: &[Generator] = &[];

const BEGIN: &str = "<!-- BEGIN GENERATED: ";
const BEGIN_END: &str = " -->";
const END: &str = "<!-- END GENERATED -->";

/// One generated section of a page.
#[derive(Debug, PartialEq, Eq)]
pub struct Section {
    pub name: String,
    /// The line of its `BEGIN` marker, from 1.
    pub line: usize,
    /// The byte range of its content: from after the `BEGIN` line to the start of the `END` line.
    pub content: std::ops::Range<usize>,
}

/// The generated sections of `text`, outside fenced code. A marker that is not paired fails.
pub fn sections(text: &str) -> Result<Vec<Section>, String> {
    let mut sections = Vec::new();
    let mut open: Option<(String, usize, usize)> = None;
    let mut fence: Option<String> = None;
    let mut offset = 0;
    for (index, line) in text.split_inclusive('\n').enumerate() {
        let number = index + 1;
        let start = offset;
        offset += line.len();
        let trimmed = line.trim();
        if let Some(marker) = &fence {
            if trimmed.starts_with(marker.as_str()) && trimmed.chars().all(|c| c == '`' || c == '~')
            {
                fence = None;
            }
            continue;
        }
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            let c = trimmed.chars().next().unwrap_or('`');
            fence = Some(trimmed.chars().take_while(|x| *x == c).collect());
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix(BEGIN) {
            let Some(name) = rest.strip_suffix(BEGIN_END) else {
                return Err(format!(
                    "line {number}: a BEGIN GENERATED marker must end with ` -->`"
                ));
            };
            if let Some((other, line, _)) = &open {
                return Err(format!(
                    "line {number}: section `{name}` starts inside section `{other}` (line {line})"
                ));
            }
            let valid = !name.is_empty()
                && name.split('-').all(|w| {
                    !w.is_empty()
                        && w.chars()
                            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
                });
            if !valid {
                return Err(format!(
                    "line {number}: `{name}` is not a generator name: lower-case words joined by `-`"
                ));
            }
            open = Some((name.to_owned(), number, offset));
        } else if trimmed == END {
            let Some((name, line, content_start)) = open.take() else {
                return Err(format!("line {number}: END GENERATED with no section open"));
            };
            sections.push(Section {
                name,
                line,
                content: content_start..start,
            });
        } else if trimmed.contains("GENERATED:") && trimmed.starts_with("<!--") {
            return Err(format!(
                "line {number}: a malformed generated-section marker"
            ));
        }
    }
    match open {
        Some((name, line, _)) => Err(format!(
            "line {line}: section `{name}` has no END GENERATED"
        )),
        None => Ok(sections),
    }
}

/// The section text a generator's output becomes: ending with exactly one newline.
fn normalise(output: &str) -> String {
    let mut text = output.trim_end_matches('\n').to_owned();
    if !text.is_empty() {
        text.push('\n');
    }
    text
}

/// What one page's sections should hold: `Ok` with the page as it should read, or every problem.
pub fn regenerate(
    root: &Path,
    page: &str,
    text: &str,
    generators: &[Generator],
) -> Result<String, Vec<String>> {
    let sections = sections(text).map_err(|e| vec![format!("{page}: {e}")])?;
    let mut problems = Vec::new();
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    for section in sections {
        let Some(generator) = generators.iter().find(|g| g.name == section.name) else {
            let known: Vec<&str> = generators.iter().map(|g| g.name).collect();
            problems.push(format!(
                "{page}:{}: no generator is named `{}` (known: {})",
                section.line,
                section.name,
                if known.is_empty() {
                    "none yet".to_owned()
                } else {
                    known.join(", ")
                }
            ));
            continue;
        };
        match (generator.run)(root) {
            Ok(output) => {
                out.push_str(&text[at..section.content.start]);
                out.push_str(&normalise(&output));
                at = section.content.end;
            }
            Err(error) => problems.push(format!(
                "{page}:{}: generator `{}` ({}) failed: {error}",
                section.line, generator.name, generator.task
            )),
        }
    }
    out.push_str(&text[at..]);
    if problems.is_empty() {
        Ok(out)
    } else {
        Err(problems)
    }
}
