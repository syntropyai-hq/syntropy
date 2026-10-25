// SPDX-License-Identifier: MIT OR Apache-2.0

//! The static origin check of the built docs site (PROD-E13-T01, ADR-0510): no file of the site
//! makes the browser request anything from another origin, and the site has no scripts, fonts or
//! analytics.
//!
//! This reads the files; it does not load them in a browser. It fails on:
//! - a file type the site does not ship: only HTML, CSS and images are allowed, so no script,
//!   font or other file can be served;
//! - any script: a `<script>` element, an `on…` event attribute or a `javascript:` URL;
//! - an element that loads another document or plug-in: `iframe`, `frame`, `object`, `embed`,
//!   `applet`, `portal`, and `base`, which would change where every relative URL points;
//! - a refresh `<meta>`, which navigates without a click;
//! - a URL that the browser fetches by itself and that is absolute or scheme-relative: `src`,
//!   `srcset`, `poster`, `data`, `action`, `formaction`, `background`, `ping`, `manifest`, and
//!   `href` on anything but a link a person follows (`<a>`, `<area>`), such as `<link>`, SVG's
//!   `<image>`, `<use>` and `<feImage>`;
//! - a CSS `url()` that is absolute or scheme-relative, any `@import` and any `@font-face`, in a
//!   stylesheet, a `<style>` element or a `style` attribute;
//! - an HTML page without the site's content security policy, which tells the browser to refuse
//!   any request to another origin even if this check missed one.
//!
//! A link a person follows to another origin, such as a cited RFC on the forge, is navigation,
//! not a request the page makes, and is allowed.
//!
//! The headless-browser check that the acceptance criterion names, a real page load that records
//! every request, is not this check. It needs a browser on the runner, which needs a maintainer's
//! approval; until then this check and the policy are the evidence.

use std::path::Path;

use super::html::{self, Scan};

/// The content security policy every page carries in a `<meta>`.
pub const CSP: &str = "default-src 'none'; img-src 'self'; style-src 'self'; base-uri 'none'; \
                       form-action 'none'";

/// The file extensions the site ships.
pub const SHIPPED: &[&str] = &["html", "css", "svg", "png", "jpg", "jpeg", "gif", "webp"];

/// Elements that load another document or plug-in, or change how URLs resolve.
const EMBEDDING: &[&str] = &[
    "iframe", "frame", "frameset", "object", "embed", "applet", "portal", "base",
];

/// Attributes whose URL the browser fetches without a person following a link.
const FETCHED: &[&str] = &[
    "src",
    "srcset",
    "poster",
    "data",
    "action",
    "formaction",
    "background",
    "ping",
    "manifest",
    "longdesc",
    "codebase",
    "archive",
    "imagesrcset",
];

/// Every problem with the file at `relative` (its path in the site) whose bytes are `bytes`.
pub fn check_file(relative: &str, bytes: &[u8]) -> Vec<String> {
    let extension = Path::new(relative)
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    if !SHIPPED.contains(&extension.as_str()) {
        return vec![format!(
            "{relative}: the site ships only HTML, CSS and images ({}), so no script or font \
             can be served",
            SHIPPED.join(", ")
        )];
    }
    let text = || String::from_utf8_lossy(bytes).into_owned();
    match extension.as_str() {
        "html" => {
            let text = text();
            let mut problems = check_markup(relative, &html::scan(&text));
            let csp = html::scan(&text)
                .tags
                .iter()
                .filter(|t| {
                    t.name == "meta"
                        && t.attr("http-equiv")
                            .and_then(|a| a.value.as_deref())
                            .is_some_and(|v| v.eq_ignore_ascii_case("content-security-policy"))
                })
                .map(|t| t.attr("content").and_then(|a| a.value.clone()))
                .collect::<Vec<_>>();
            if csp != [Some(CSP.to_owned())] {
                problems.push(format!(
                    "{relative}: the page must carry exactly one content security policy, \
                     `{CSP}`"
                ));
            }
            problems
        }
        "svg" => check_markup(relative, &html::scan(&text())),
        "css" => check_css(relative, "the stylesheet", &text()),
        _ => Vec::new(),
    }
}

/// Every problem with the tags of an HTML page or SVG image.
pub fn check_markup(relative: &str, scan: &Scan) -> Vec<String> {
    let mut problems = Vec::new();
    for (element, content) in &scan.raw {
        if element == "style" {
            problems.extend(check_css(relative, "a <style> element", content));
        }
    }
    for tag in scan.tags.iter().filter(|t| !t.closing) {
        let name = tag.name.as_str();
        if name == "script" {
            problems.push(format!(
                "{relative}: a <script> element; the site has no scripts"
            ));
        }
        if EMBEDDING.contains(&name) {
            problems.push(format!(
                "{relative}: a <{name}> element, which loads another document or changes where \
                 URLs point"
            ));
        }
        if name == "meta"
            && tag
                .attr("http-equiv")
                .and_then(|a| a.value.as_deref())
                .is_some_and(|v| v.eq_ignore_ascii_case("refresh"))
        {
            problems.push(format!(
                "{relative}: a refresh <meta>, which navigates by itself"
            ));
        }
        for attr in &tag.attrs {
            let value = attr.value.as_deref().unwrap_or("");
            if attr.name.starts_with("on") {
                problems.push(format!(
                    "{relative}: <{name}> has an event handler, `{}`; the site has no scripts",
                    attr.name
                ));
            }
            if value
                .trim_start()
                .to_ascii_lowercase()
                .starts_with("javascript:")
            {
                problems.push(format!(
                    "{relative}: <{name} {}> is a `javascript:` URL; the site has no scripts",
                    attr.name
                ));
            }
            if attr.name == "style" {
                problems.extend(check_css(relative, "a style attribute", value));
            }
            let link = matches!(name, "a" | "area");
            let fetched = FETCHED.contains(&attr.name.as_str())
                || (!link && matches!(attr.name.as_str(), "href" | "xlink:href"));
            if !fetched {
                continue;
            }
            let urls: Vec<&str> = if attr.name.ends_with("srcset") {
                value
                    .split(',')
                    .filter_map(|candidate| candidate.split_whitespace().next())
                    .collect()
            } else {
                vec![value]
            };
            for url in urls.into_iter().filter(|url| is_remote(url)) {
                problems.push(format!(
                    "{relative}: <{name} {}=\"{url}\"> fetches from another origin",
                    attr.name
                ));
            }
        }
    }
    problems
}

/// Every problem with a stylesheet's text.
pub fn check_css(relative: &str, place: &str, css: &str) -> Vec<String> {
    let mut problems = Vec::new();
    let lower = css.to_ascii_lowercase();
    if lower.contains("@import") {
        problems.push(format!(
            "{relative}: {place} has an @import; the site's CSS is one file"
        ));
    }
    if lower.contains("@font-face") {
        problems.push(format!(
            "{relative}: {place} declares a font; the site uses the reader's system fonts"
        ));
    }
    let mut rest = lower.as_str();
    while let Some(at) = rest.find("url(") {
        let after = &rest[at + 4..];
        let end = after.find(')').unwrap_or(after.len());
        let url = after[..end].trim().trim_matches(['"', '\'']);
        if is_remote(url) {
            problems.push(format!(
                "{relative}: {place} has url({url}), which fetches from another origin"
            ));
        }
        rest = &after[end..];
    }
    problems
}

/// Whether a URL leaves the page's origin: it has a scheme other than `data:`, or it is
/// scheme-relative (`//host`, or `/\host` and `\\host`, which browsers read the same way).
pub fn is_remote(url: &str) -> bool {
    let url: String = url
        .trim()
        .chars()
        .filter(|c| !matches!(c, '\t' | '\n' | '\r'))
        .collect();
    let lower = url.to_ascii_lowercase();
    if lower.starts_with("//") || lower.starts_with("/\\") || lower.starts_with("\\\\") {
        return true;
    }
    match lower.find(':') {
        Some(colon) => {
            let scheme = &lower[..colon];
            let is_scheme = !scheme.is_empty()
                && scheme
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_alphabetic())
                && scheme
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'));
            let before_path = !lower[..colon].contains(['/', '?', '#']);
            is_scheme && before_path && scheme != "data"
        }
        None => false,
    }
}
