// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask ci docs`: the docs site's build and checks (PROD-E13-T01; RFC-0051 §4 rule 8,
//! ADR-0510).
//!
//! The site's sources are the Markdown pages and images under `docs/`, with `docs/site.toml`
//! naming the site, the repository and the navigation. The build needs nothing but this binary
//! and the checkout: no network, no site generator, no package manager. It writes a static site,
//! by default to `target/docs-site/`:
//! - each `docs/<path>.md` becomes `<path>.html`, rendered by [`markdown`] the way the forge
//!   renders it, inside one page template with the site's navigation;
//! - each image is copied as it is, and the site's one stylesheet, `style.css`, is written beside
//!   `index.html`;
//! - the site is English only (RFC-0051 §4 rule 8): every page is `lang="en"`, and a translated
//!   page, such as `README.de.md`, is refused.
//!
//! The job's steps:
//! - `generated`: every generated section of a page matches its generator ([`generated`], the
//!   hooks later tasks fill);
//! - `build`: renders the site, and fails on any link it cannot resolve. A link to a page or
//!   image of the site must name a file that exists, with that exact case; a link with a
//!   fragment must name an anchor the target page has (a heading's id or an `id` or `name`
//!   attribute). A link out of `docs/` to another file of the repository is checked the same
//!   way, anchors included for Markdown files, and is rewritten to that file on the forge, since
//!   the site does not carry it. A site-absolute link (`/x`) is refused, so the site works under
//!   any path;
//! - `origins`: the built site makes no request to another origin and has no scripts, fonts or
//!   analytics ([`origins`]), a static check of the files.
//!
//! `pr.yml`'s `docs` job runs it when a change touches `docs/`, `plan/`, a top-level Markdown
//! file, which the pages link to, or this generator (`ci/jobs.toml`).

pub mod generated;
mod html;
mod markdown;
mod origins;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::run::Context;

/// The site's sources, relative to the workspace root.
pub const SOURCE: &str = "docs";

/// The site's configuration, relative to the workspace root.
pub const CONFIG: &str = "docs/site.toml";

/// Where the build writes the site unless `--out` says otherwise, relative to the root.
pub const DEFAULT_OUT: &str = "target/docs-site";

/// The site's one stylesheet, at the site's root.
pub const STYLE: &str = "style.css";

/// The image types a page may use, copied into the site as they are.
const IMAGES: &[&str] = &["svg", "png", "jpg", "jpeg", "gif", "webp"];

/// `docs/site.toml`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SiteConfig {
    /// The site's name, in every page's title and header.
    title: String,
    /// The repository on the forge, `https://host/owner/name`, where links out of `docs/` go.
    repository: String,
    /// The branch those links name.
    branch: String,
    /// The pages in the navigation, in order, as paths under `docs/`.
    nav: Vec<String>,
}

impl SiteConfig {
    fn read(root: &Path) -> Result<SiteConfig, Vec<String>> {
        let text = std::fs::read_to_string(root.join(CONFIG))
            .map_err(|e| vec![format!("{CONFIG}: {e}")])?;
        let config: SiteConfig =
            toml::from_str(&text).map_err(|e| vec![format!("{CONFIG}: {e}")])?;
        let mut problems = Vec::new();
        let host_path = config.repository.strip_prefix("https://");
        if host_path.is_none_or(|rest| rest.ends_with('/') || rest.split('/').count() != 3) {
            problems.push(format!(
                "{CONFIG}: `repository` is `https://<host>/<owner>/<name>` with no trailing `/`"
            ));
        }
        if config.branch.is_empty() || config.branch.contains(['/', ' ', '#', '?']) {
            problems.push(format!("{CONFIG}: `branch` is one branch name"));
        }
        if !config.nav.iter().any(|p| p == "index.md") {
            problems.push(format!(
                "{CONFIG}: `nav` must list the home page, `index.md`"
            ));
        }
        if problems.is_empty() {
            Ok(config)
        } else {
            Err(problems)
        }
    }
}

/// A file of the built site: its path in the site and its bytes.
pub type SiteFile = (String, Vec<u8>);

/// One rendered page before its links are resolved.
struct Page {
    /// The source path under `docs/`, such as `HOW_IT_WORKS.md`.
    source: String,
    title: String,
    body: String,
    /// Every anchor the page has.
    ids: BTreeSet<String>,
}

/// Where a link points, once resolved against the page that holds it.
#[derive(Debug, PartialEq, Eq)]
enum Target {
    /// Another origin, or a scheme such as `mailto:`: navigation the site does not check.
    External,
    /// A page of the site, by its source path under `docs/`.
    Page {
        source: String,
        fragment: Option<String>,
    },
    /// An image or other file of the site, by its path under `docs/`.
    Asset { path: String },
    /// A file of the repository outside `docs/`, by its path from the root.
    Repository {
        path: String,
        fragment: Option<String>,
    },
}

/// Resolves `url`, written in the page at `page` (a path under `docs/`).
fn resolve(page: &str, url: &str) -> Result<Target, String> {
    let url = url.trim();
    if url.is_empty() {
        return Err("an empty link".to_owned());
    }
    if let Some(fragment) = url.strip_prefix('#') {
        return Ok(Target::Page {
            source: page.to_owned(),
            fragment: Some(percent_decode(fragment)),
        });
    }
    if origins::is_remote(url) {
        if url
            .trim_start()
            .to_ascii_lowercase()
            .starts_with("javascript:")
        {
            return Err("a `javascript:` URL".to_owned());
        }
        return Ok(Target::External);
    }
    if url.starts_with("data:") {
        return Ok(Target::External);
    }
    if url.starts_with('/') {
        return Err(
            "a site-absolute link, which breaks when the site is not served at the root; use a \
             relative path"
                .to_owned(),
        );
    }
    let (path, fragment) = match url.split_once('#') {
        Some((path, fragment)) => (path, Some(percent_decode(fragment))),
        None => (url, None),
    };
    if path.contains('?') {
        return Err("a link with a query string, which a static site cannot serve".to_owned());
    }
    let mut parts: Vec<String> = format!("{SOURCE}/{page}")
        .split('/')
        .map(str::to_owned)
        .collect();
    parts.pop();
    for segment in percent_decode(path).split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                if parts.pop().is_none() {
                    return Err("a link above the repository's root".to_owned());
                }
            }
            other => parts.push(other.to_owned()),
        }
    }
    let joined = parts.join("/");
    match joined.strip_prefix(&format!("{SOURCE}/")) {
        Some(inside) if inside.ends_with(".md") => Ok(Target::Page {
            source: inside.to_owned(),
            fragment,
        }),
        Some(inside) => {
            if fragment.is_some() {
                return Err("a fragment on a link to an image".to_owned());
            }
            Ok(Target::Asset {
                path: inside.to_owned(),
            })
        }
        None if joined.is_empty() || joined == SOURCE => {
            Err("a link to a directory of the site; link to its page".to_owned())
        }
        None => Ok(Target::Repository {
            path: joined,
            fragment,
        }),
    }
}

fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && let Some(hex) = text.get(i + 1..i + 3)
            && let Ok(byte) = u8::from_str_radix(hex, 16)
        {
            out.push(byte);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Whether `relative` (`/`-separated) names a file or directory under `root` with exactly that
/// case, as a case-sensitive file system and the forge see it. `Some(true)` for a directory.
fn exists_exact(root: &Path, relative: &str) -> Option<bool> {
    let mut path = root.to_path_buf();
    for part in relative.split('/').filter(|p| !p.is_empty()) {
        let found = std::fs::read_dir(&path)
            .ok()?
            .filter_map(Result::ok)
            .any(|entry| entry.file_name().to_str() == Some(part));
        if !found {
            return None;
        }
        path.push(part);
    }
    Some(path.is_dir())
}

/// Every anchor of an HTML fragment: each `id`, and each `name` of an `<a>`.
fn anchors(html: &str) -> BTreeSet<String> {
    html::scan(html)
        .tags
        .iter()
        .filter(|t| !t.closing)
        .flat_map(|t| {
            t.attrs.iter().filter_map(move |a| {
                let anchor = a.name == "id" || (a.name == "name" && t.name == "a");
                anchor.then(|| a.value.clone()).flatten()
            })
        })
        .collect()
}

/// Whether a page's file name marks it as a translation, such as `README.de.md` or
/// `README.pt-BR.md`.
fn is_translation(name: &str) -> bool {
    let Some(stem) = name.strip_suffix(".md") else {
        return false;
    };
    let Some((_, tag)) = stem.rsplit_once('.') else {
        return false;
    };
    let (language, region) = tag.split_once('-').unwrap_or((tag, ""));
    (2..=3).contains(&language.len())
        && language.chars().all(|c| c.is_ascii_lowercase())
        && (region.is_empty()
            || ((2..=4).contains(&region.len()) && region.chars().all(|c| c.is_ascii_alphabetic())))
}

/// Every file under `dir`, as `/`-separated paths relative to it, sorted.
fn walk(dir: &Path) -> Result<Vec<String>, String> {
    let mut files = Vec::new();
    let mut stack = vec![(dir.to_path_buf(), String::new())];
    while let Some((path, prefix)) = stack.pop() {
        let entries = std::fs::read_dir(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("{}: {e}", path.display()))?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|n| format!("{}: a file name that is not UTF-8: {n:?}", path.display()))?;
            let relative = format!("{prefix}{name}");
            let kind = entry.file_type().map_err(|e| format!("{relative}: {e}"))?;
            if kind.is_dir() {
                stack.push((entry.path(), format!("{relative}/")));
            } else {
                files.push(relative);
            }
        }
    }
    files.sort();
    Ok(files)
}

/// The prefix that leads from the directory of `path` back to the site's root.
fn up(path: &str) -> String {
    "../".repeat(path.matches('/').count())
}

/// Builds the site from the checkout at `root`. Returns every file of the site, or every
/// problem found.
pub fn build(root: &Path) -> Result<Vec<SiteFile>, Vec<String>> {
    let config = SiteConfig::read(root)?;
    let source = root.join(SOURCE);
    let files = walk(&source).map_err(|e| vec![e])?;
    let mut problems = Vec::new();
    let mut pages: Vec<Page> = Vec::new();
    let mut assets: BTreeSet<String> = BTreeSet::new();
    for file in &files {
        let shown = format!("{SOURCE}/{file}");
        if format!("{SOURCE}/{file}") == CONFIG {
            continue;
        }
        let name = file.rsplit('/').next().unwrap_or(file);
        if let Some(stem) = file.strip_suffix(".md") {
            if is_translation(name) {
                problems.push(format!(
                    "{shown}: a translated page; the site is English only (RFC-0051 §4 rule 8)"
                ));
                continue;
            }
            let text = match std::fs::read_to_string(source.join(file)) {
                Ok(text) => text,
                Err(e) => {
                    problems.push(format!("{shown}: {e}"));
                    continue;
                }
            };
            let document = markdown::render(&text);
            let title = document.headings.iter().find(|h| h.level == 1).map_or_else(
                || stem.rsplit('/').next().unwrap_or(stem).to_owned(),
                |h| h.text.clone(),
            );
            pages.push(Page {
                source: file.clone(),
                title,
                ids: anchors(&document.html),
                body: document.html,
            });
            continue;
        }
        let extension = name.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase());
        if extension.is_some_and(|e| IMAGES.contains(&e.as_str())) {
            assets.insert(file.clone());
        } else {
            problems.push(format!(
                "{shown}: the site's sources are Markdown pages and images ({}); move other \
                 files out of {SOURCE}/",
                IMAGES.join(", ")
            ));
        }
    }
    if !pages.iter().any(|p| p.source == "index.md") {
        problems.push(format!("{SOURCE}/index.md: the home page is missing"));
    }
    let titles: BTreeMap<&str, &str> = pages
        .iter()
        .map(|p| (p.source.as_str(), p.title.as_str()))
        .collect();
    for entry in &config.nav {
        if !titles.contains_key(entry.as_str()) {
            problems.push(format!(
                "{CONFIG}: `nav` lists `{entry}`, which is not a page"
            ));
        }
    }
    let ids: BTreeMap<&str, &BTreeSet<String>> =
        pages.iter().map(|p| (p.source.as_str(), &p.ids)).collect();

    let mut repository_anchors: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut out: Vec<SiteFile> = Vec::new();
    for page in &pages {
        let shown = format!("{SOURCE}/{}", page.source);
        let scan = html::scan(&page.body);
        let mut edits = Vec::new();
        // The body is rendered from the page's own text, so scanning it again finds exactly its
        // tags; the spans index into `page.body`.
        for tag in scan.tags.iter().filter(|t| !t.closing) {
            for attr in &tag.attrs {
                if !matches!(attr.name.as_str(), "href" | "src" | "srcset" | "poster") {
                    continue;
                }
                let (Some(value), Some(span)) = (&attr.value, &attr.span) else {
                    continue;
                };
                let urls: Vec<&str> = if attr.name == "srcset" {
                    value
                        .split(',')
                        .filter_map(|c| c.split_whitespace().next())
                        .collect()
                } else {
                    vec![value.as_str()]
                };
                for url in urls {
                    let place = format!("{shown}: <{} {}=\"{url}\">", tag.name, attr.name);
                    match resolve(&page.source, url) {
                        Err(why) => problems.push(format!("{place}: {why}")),
                        Ok(Target::External) => {}
                        Ok(Target::Page { source, fragment }) => {
                            let Some(target) = ids.get(source.as_str()) else {
                                problems.push(format!(
                                    "{place}: no page {SOURCE}/{source} (names are case-sensitive)"
                                ));
                                continue;
                            };
                            if let Some(fragment) = &fragment
                                && !target.contains(fragment)
                            {
                                problems.push(format!(
                                    "{place}: {SOURCE}/{source} has no anchor `#{fragment}`"
                                ));
                            }
                            if attr.name == "href" && !url.starts_with('#') {
                                let html = |p: &str| {
                                    format!("{}.html", p.strip_suffix(".md").unwrap_or(p))
                                };
                                let rewritten = match url.split_once('#') {
                                    Some((path, rest)) => format!("{}#{rest}", html(path)),
                                    None => html(url),
                                };
                                edits.push((span.clone(), rewritten));
                            }
                        }
                        Ok(Target::Asset { path }) => {
                            if !assets.contains(&path) {
                                problems.push(format!(
                                    "{place}: no image {SOURCE}/{path} in the site (names are \
                                     case-sensitive)"
                                ));
                            }
                        }
                        Ok(Target::Repository { path, fragment }) => {
                            let Some(is_dir) = exists_exact(root, &path) else {
                                problems.push(format!(
                                    "{place}: no file {path} in the repository (names are \
                                     case-sensitive)"
                                ));
                                continue;
                            };
                            if let Some(fragment) = &fragment {
                                if path.ends_with(".md") && !is_dir {
                                    let found = repository_anchors
                                        .entry(path.clone())
                                        .or_insert_with(|| {
                                            std::fs::read_to_string(root.join(&path))
                                                .map(|t| anchors(&markdown::render(&t).html))
                                                .unwrap_or_default()
                                        });
                                    if !found.contains(fragment) {
                                        problems.push(format!(
                                            "{place}: {path} has no anchor `#{fragment}`"
                                        ));
                                    }
                                } else if !is_line_anchor(fragment) {
                                    problems.push(format!(
                                        "{place}: the anchor `#{fragment}` cannot be checked; \
                                         link to a Markdown file's heading or a line (`#L12`)"
                                    ));
                                }
                            }
                            if attr.name != "href" {
                                problems.push(format!(
                                    "{place}: an image outside {SOURCE}/, which the site does not \
                                     carry"
                                ));
                                continue;
                            }
                            let kind = if is_dir { "tree" } else { "blob" };
                            let mut rewritten =
                                format!("{}/{kind}/{}/{path}", config.repository, config.branch);
                            if let Some(fragment) = fragment {
                                rewritten.push('#');
                                rewritten.push_str(&fragment);
                            }
                            edits.push((span.clone(), rewritten));
                        }
                    }
                }
            }
        }
        let body = html::rewrite(&page.body, edits);
        let html_path = format!("{}.html", page.source.trim_end_matches(".md"));
        let document = template(&config, &titles, page, &body);
        out.push((html_path, document.into_bytes()));
    }
    for asset in &assets {
        match std::fs::read(source.join(asset)) {
            Ok(bytes) => out.push((asset.clone(), bytes)),
            Err(e) => problems.push(format!("{SOURCE}/{asset}: {e}")),
        }
    }
    out.push((STYLE.to_owned(), CSS.as_bytes().to_vec()));
    out.sort();
    if problems.is_empty() {
        Ok(out)
    } else {
        Err(problems)
    }
}

/// `L12` or `L12-L20`: a line anchor the forge gives every text file.
fn is_line_anchor(fragment: &str) -> bool {
    let line = |s: &str| {
        s.strip_prefix('L')
            .is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()))
    };
    match fragment.split_once('-') {
        Some((a, b)) => line(a) && line(b),
        None => line(fragment),
    }
}

/// The page around a rendered body.
fn template(config: &SiteConfig, titles: &BTreeMap<&str, &str>, page: &Page, body: &str) -> String {
    let up = up(&page.source);
    let esc = markdown::escape;
    let title = if page.source == "index.md" {
        esc(&config.title)
    } else {
        format!("{} · {}", esc(&page.title), esc(&config.title))
    };
    let mut nav = String::new();
    for entry in &config.nav {
        let label = titles.get(entry.as_str()).copied().unwrap_or(entry);
        let current = if *entry == page.source {
            " aria-current=\"page\""
        } else {
            ""
        };
        nav.push_str(&format!(
            "<li><a href=\"{up}{}.html\"{current}>{}</a></li>\n",
            esc(entry.trim_end_matches(".md")),
            esc(label)
        ));
    }
    format!(
        "<!doctype html>\n\
         <html lang=\"en\">\n\
         <head>\n\
         <meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <meta http-equiv=\"Content-Security-Policy\" content=\"{csp}\">\n\
         <meta name=\"referrer\" content=\"no-referrer\">\n\
         <title>{title}</title>\n\
         <link rel=\"stylesheet\" href=\"{up}{STYLE}\">\n\
         </head>\n\
         <body>\n\
         <header>\n\
         <nav aria-label=\"Site\">\n\
         <ul>\n\
         {nav}\
         </ul>\n\
         </nav>\n\
         </header>\n\
         <main>\n\
         {body}\
         </main>\n\
         </body>\n\
         </html>\n",
        csp = esc(origins::CSP),
    )
}

/// The site's stylesheet: the reader's system fonts, light and dark colours, and nothing loaded.
const CSS: &str = "\
:root {
  color-scheme: light dark;
  --text: #1f2328;
  --muted: #59636e;
  --background: #ffffff;
  --surface: #f6f8fa;
  --border: #d1d9e0;
  --link: #0969da;
}
@media (prefers-color-scheme: dark) {
  :root {
    --text: #e6edf3;
    --muted: #9198a1;
    --background: #0d1117;
    --surface: #151b23;
    --border: #3d444d;
    --link: #4493f8;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--background);
  color: var(--text);
  font-family: system-ui, -apple-system, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif;
  line-height: 1.6;
}
header { border-bottom: 1px solid var(--border); background: var(--surface); }
nav ul {
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem 1.25rem;
  list-style: none;
  margin: 0 auto;
  max-width: 50rem;
  padding: 0.75rem 1rem;
}
nav a[aria-current=page] { font-weight: 600; color: var(--text); }
main { margin: 0 auto; max-width: 50rem; padding: 1rem 1rem 4rem; }
a { color: var(--link); }
h1, h2, h3, h4 { line-height: 1.25; margin: 1.5em 0 0.5em; }
h1 { border-bottom: 1px solid var(--border); padding-bottom: 0.3em; }
img { max-width: 100%; height: auto; }
code, pre {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, \"Liberation Mono\", monospace;
  font-size: 0.9em;
}
code { background: var(--surface); border-radius: 4px; padding: 0.1em 0.3em; }
pre { background: var(--surface); border-radius: 6px; overflow-x: auto; padding: 0.75rem 1rem; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; display: block; overflow-x: auto; margin: 1rem 0; }
th, td { border: 1px solid var(--border); padding: 0.4rem 0.75rem; vertical-align: top; }
th { background: var(--surface); }
blockquote { border-left: 4px solid var(--border); color: var(--muted); margin: 1rem 0; padding: 0 1rem; }
p[align=center] { text-align: center; }
";

/// Writes the site to `out`, replacing an earlier build there. A directory that holds something
/// other than a site build is left alone.
fn write(out: &Path, files: &[SiteFile]) -> Result<(), String> {
    if out.exists() {
        let is_site = out.join("index.html").is_file() && out.join(STYLE).is_file();
        let empty = std::fs::read_dir(out)
            .map(|mut d| d.next().is_none())
            .unwrap_or(false);
        if !is_site && !empty {
            return Err(format!(
                "{}: not empty and not an earlier site build; choose another --out",
                out.display()
            ));
        }
        std::fs::remove_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    }
    for (path, bytes) in files {
        let target = out.join(path);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        std::fs::write(&target, bytes).map_err(|e| format!("{}: {e}", target.display()))?;
    }
    Ok(())
}

fn out_dir(context: &Context) -> PathBuf {
    match context.args.get("--out") {
        Some(out) => {
            let out = PathBuf::from(out);
            if out.is_absolute() {
                out
            } else {
                context.root.join(out)
            }
        }
        None => context.root.join(DEFAULT_OUT),
    }
}

/// Every page under `docs/`, by its path there, with its text.
fn pages(root: &Path) -> Result<Vec<(String, String)>, Vec<String>> {
    let files = walk(&root.join(SOURCE)).map_err(|e| vec![e])?;
    let mut pages = Vec::new();
    let mut problems = Vec::new();
    for file in files.into_iter().filter(|f| f.ends_with(".md")) {
        match std::fs::read_to_string(root.join(SOURCE).join(&file)) {
            Ok(text) => pages.push((file, text)),
            Err(e) => problems.push(format!("{SOURCE}/{file}: {e}")),
        }
    }
    if problems.is_empty() {
        Ok(pages)
    } else {
        Err(problems)
    }
}

/// Every problem with the generated sections of the pages at `root`, against `generators`.
pub fn check_generated(root: &Path, generators: &[generated::Generator]) -> Vec<String> {
    let pages = match pages(root) {
        Ok(pages) => pages,
        Err(problems) => return problems,
    };
    let mut problems = Vec::new();
    for (file, text) in pages {
        let shown = format!("{SOURCE}/{file}");
        match generated::regenerate(root, &shown, &text, generators) {
            Ok(expected) if expected == text => {}
            Ok(_) => problems.push(format!(
                "{shown}: a generated section differs from its generator; run `cargo xtask \
                 docs-generate` and commit the result"
            )),
            Err(found) => problems.extend(found),
        }
    }
    problems
}

/// Rewrites every generated section at `root`. Returns the pages it changed.
pub fn write_generated(
    root: &Path,
    generators: &[generated::Generator],
) -> Result<Vec<String>, Vec<String>> {
    let mut changed = Vec::new();
    let mut problems = Vec::new();
    for (file, text) in pages(root)? {
        let shown = format!("{SOURCE}/{file}");
        match generated::regenerate(root, &shown, &text, generators) {
            Ok(expected) if expected == text => {}
            Ok(expected) => match std::fs::write(root.join(SOURCE).join(&file), expected) {
                Ok(()) => changed.push(shown),
                Err(e) => problems.push(format!("{shown}: {e}")),
            },
            Err(found) => problems.extend(found),
        }
    }
    if problems.is_empty() {
        Ok(changed)
    } else {
        Err(problems)
    }
}

/// Every problem the origin check finds in the site built at `out`.
pub fn check_site(out: &Path) -> Vec<String> {
    let files = match walk(out) {
        Ok(files) => files,
        Err(e) => {
            return vec![format!("{e}; run the `build` step first")];
        }
    };
    let mut problems = Vec::new();
    for file in files {
        match std::fs::read(out.join(&file)) {
            Ok(bytes) => problems.extend(origins::check_file(&file, &bytes)),
            Err(e) => problems.push(format!("{file}: {e}")),
        }
    }
    problems
}

fn verdict(problems: Vec<String>) -> Result<(), String> {
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

/// The `generated` step of the `docs` job.
pub fn generated_step(context: &Context) -> Result<(), String> {
    verdict(check_generated(&context.root, generated::GENERATORS))
}

/// The `build` step of the `docs` job: builds the site, checking every link, and writes it.
pub fn build_step(context: &Context) -> Result<(), String> {
    let out = out_dir(context);
    let files = match build(&context.root) {
        Ok(files) => files,
        Err(problems) => {
            // Remove an earlier build, so the `origins` step cannot pass on a stale site.
            let mut message = problems.join("\n");
            if let Err(error) = write(&out, &[]) {
                message.push('\n');
                message.push_str(&error);
            }
            return Err(message);
        }
    };
    write(&out, &files)?;
    let pages = files.iter().filter(|(p, _)| p.ends_with(".html")).count();
    eprintln!(
        "docs: built {pages} pages, {} files, into {}",
        files.len(),
        out.display()
    );
    Ok(())
}

/// The `origins` step of the `docs` job: the static check of the site the `build` step wrote.
pub fn origins_step(context: &Context) -> Result<(), String> {
    verdict(check_site(&out_dir(context)))
}

#[cfg(test)]
mod tests;
