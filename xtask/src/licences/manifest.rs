// SPDX-License-Identifier: MIT OR Apache-2.0

//! `third_party/licences.toml`, the licence manifest of vendored non-Rust code, and
//! `cargo xtask third-party`, which checks it (ADR-0018 §3, RFC-0062 §11.3, §12.2).
//!
//! The manifest lists every vendored third-party file on its own, as one `[[file]]` table:
//!
//! ```toml
//! [[file]]
//! path = "third_party/ggml/src/ggml.c"   # one file, never a directory
//! upstream = "https://github.com/ggml-org/ggml"
//! version = "<release or commit>"
//! license = "MIT"                        # an SPDX expression the allow list meets
//! notice = "third_party/ggml/LICENSE"    # the file with its copyright and licence text
//! ```
//!
//! The check fails when a file under `third_party/` or a `crates/kernels-*/vendor/` tree has no
//! entry of its own; when an entry names a directory, a missing file or a path outside those
//! trees; when a key is missing, empty or unknown; when an entry's licence is not met by the
//! allow list in `deny.toml`; and when its notice file is missing. The manifest itself and the
//! lock files beside it in `third_party/` (such as `ggml.lock`, RFC-0062 §11.3) are Syntropy's
//! own files and need no entry.

use std::collections::BTreeMap;
use std::io::{self, Write};
use std::path::Path;

use super::expr::Expr;
use super::toml::{self, Value};
use super::{Policy, THIRD_PARTY, count, file_inside, is_vendored, relative};

/// Where the manifest lives, relative to the workspace root.
pub const PATH: &str = "third_party/licences.toml";

/// The keys of an entry. Every one is required.
pub const KEYS: &[&str] = &["path", "upstream", "version", "license", "notice"];

/// One vendored file.
#[derive(Debug, PartialEq, Eq)]
pub struct Entry {
    /// The line of its `[[file]]` header.
    pub line: usize,
    /// Root-relative, with `/` separators.
    pub path: String,
    pub upstream: String,
    pub version: String,
    pub license: String,
    pub notice: String,
}

/// The entries of the manifest under `root`, in file order, or the problems that stop it being
/// read. A missing manifest has no entries.
pub fn load(root: &Path) -> Result<Vec<Entry>, Vec<String>> {
    let path = root.join(PATH);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(vec![format!("cannot read {PATH}: {error}")]),
    };
    let tables = toml::tables(&text, "file").map_err(|error| vec![format!("{PATH}:{error}")])?;
    let mut entries = Vec::new();
    let mut problems = Vec::new();
    for table in tables {
        let at = format!("{PATH}:{}", table.line);
        for (key, value) in &table.entries {
            if !KEYS.contains(&key.as_str()) {
                problems.push(format!(
                    "{at}: unknown key `{key}`; an entry has {}",
                    KEYS.join(", ")
                ));
            } else if !matches!(value, Value::Str(_)) {
                problems.push(format!("{at}: `{key}` must be a string"));
            }
        }
        let mut get = |key: &str| match table.str(key) {
            Some(value) if !value.trim().is_empty() => value.to_owned(),
            Some(_) => {
                problems.push(format!("{at}: `{key}` is empty"));
                String::new()
            }
            None => {
                problems.push(format!("{at}: `{key}` is missing"));
                String::new()
            }
        };
        let entry = Entry {
            line: table.line,
            path: get("path"),
            upstream: get("upstream"),
            version: get("version"),
            license: get("license"),
            notice: get("notice"),
        };
        entries.push(entry);
    }
    if problems.is_empty() {
        Ok(entries)
    } else {
        Err(problems)
    }
}

/// Whether the root-relative `rel` is one of Syntropy's own files in `third_party/`: the
/// manifest, and the lock files directly beside it.
fn own_file(rel: &str) -> bool {
    rel == PATH
        || rel
            .strip_prefix("third_party/")
            .is_some_and(|name| !name.contains('/') && name.ends_with(".lock"))
}

/// Every vendored file under `root` that needs an entry, root-relative and sorted.
pub fn vendored_files(root: &Path) -> io::Result<Vec<String>> {
    let mut trees = Vec::new();
    if root.join(THIRD_PARTY).is_dir() {
        trees.push(root.join(THIRD_PARTY));
    }
    let crates = root.join("crates");
    if crates.is_dir() {
        for entry in std::fs::read_dir(&crates)? {
            let vendor = entry?.path().join("vendor");
            if vendor.is_dir() && is_vendored(&relative(root, &vendor)) {
                trees.push(vendor);
            }
        }
    }
    let mut files = Vec::new();
    while let Some(dir) = trees.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                trees.push(entry.path());
            } else {
                let rel = relative(root, &entry.path());
                if !own_file(&rel) {
                    files.push(rel);
                }
            }
        }
    }
    files.sort();
    Ok(files)
}

/// Every problem with the manifest under `root` against `policy`, and how many vendored files
/// it covers.
pub fn check(root: &Path, policy: &Policy) -> (Vec<String>, usize) {
    let entries = match load(root) {
        Ok(entries) => entries,
        Err(problems) => return (problems, 0),
    };
    let files = match vendored_files(root) {
        Ok(files) => files,
        Err(error) => return (vec![format!("cannot list the vendored trees: {error}")], 0),
    };
    let mut problems = Vec::new();
    let mut listed: BTreeMap<&str, usize> = BTreeMap::new();
    let mut directories = Vec::new();
    for entry in &entries {
        let at = format!("{PATH}:{}", entry.line);
        if let Some(first) = listed.insert(entry.path.as_str(), entry.line) {
            problems.push(format!(
                "{at}: `{}` is listed again; its entry is at line {first}",
                entry.path
            ));
        }
        problems.extend(path_problem(root, &entry.path).map(|p| format!("{at}: {p}")));
        if root.join(&entry.path).is_dir() {
            directories.push(entry.path.as_str());
        }
        match Expr::parse(&entry.license) {
            Ok(expr) if expr.allowed_by(&policy.allow) => {}
            Ok(expr) if expr.mentions("MPL-2.0") => problems.push(format!(
                "{at}: `{}` is under `{expr}`; MPL-2.0 is allowed only for unmodified \
                 dependencies, through a per-crate exception in deny.toml (ADR-0018 §3), so \
                 vendored MPL-2.0 code needs a maintainer's decision",
                entry.path
            )),
            Ok(expr) => problems.push(format!(
                "{at}: `{}` is under `{expr}`, which the allow list in deny.toml does not meet \
                 (ADR-0018 §3)",
                entry.path
            )),
            Err(error) => problems.push(format!(
                "{at}: `{}` has the licence `{}`, which is not an SPDX expression: {error}",
                entry.path, entry.license
            )),
        }
        if !entry.notice.is_empty()
            && let Some(problem) = notice_problem(root, &entry.notice)
        {
            problems.push(format!(
                "{at}: the notice file of `{}`: {problem}",
                entry.path
            ));
        }
    }
    let mut covered = 0;
    for file in &files {
        if listed.contains_key(file.as_str()) {
            covered += 1;
            continue;
        }
        let under = directories
            .iter()
            .find(|dir| file.starts_with(&format!("{}/", dir.trim_end_matches('/'))));
        problems.push(match under {
            Some(dir) => format!(
                "{file}: not listed in {PATH}; `{dir}` is listed as a directory, but each vendored \
                 file needs its own entry (ADR-0018 §3)"
            ),
            None => format!("{file}: not listed in {PATH} (RFC-0062 §12.2)"),
        });
    }
    (problems, covered)
}

/// What is wrong with an entry's `path`, if anything.
fn path_problem(root: &Path, path: &str) -> Option<String> {
    if path.is_empty() {
        return None;
    }
    if path.starts_with('/')
        || path.contains('\\')
        || path.split('/').any(|p| p == ".." || p == ".")
    {
        return Some(format!(
            "`{path}` is not a plain root-relative path with `/` separators"
        ));
    }
    if !is_vendored(path) || own_file(path) {
        return Some(format!(
            "`{path}` is not a vendored file: entries name files under `third_party/` or a \
             `crates/kernels-*/vendor/` tree"
        ));
    }
    let full = root.join(path);
    if full.is_dir() {
        return Some(format!(
            "`{path}` is a directory; list each vendored file in it individually (ADR-0018 §3)"
        ));
    }
    if !full.is_file() {
        return Some(format!("`{path}` does not exist"));
    }
    None
}

/// What is wrong with an entry's `notice`, if anything. The notices carry its text, so it must
/// be a vendored file inside `root`, symlinks followed, and never a path that reaches elsewhere.
fn notice_problem(root: &Path, notice: &str) -> Option<String> {
    if let Err(error) = file_inside(root, notice) {
        return Some(error);
    }
    if !is_vendored(notice) || own_file(notice) {
        return Some(format!(
            "`{notice}` is not a vendored file: a notice file sits under `third_party/` or a \
             `crates/kernels-*/vendor/` tree"
        ));
    }
    None
}

/// Checks the manifest under `root` against the policy of `checkout` and writes the findings and
/// the verdict to `log`. Returns whether the check passed.
pub fn run(checkout: &Path, root: &Path, log: &mut dyn Write) -> io::Result<bool> {
    let policy = match Policy::load(checkout) {
        Ok(policy) => policy,
        Err(error) => {
            writeln!(log, "third-party: FAIL: {error}")?;
            return Ok(false);
        }
    };
    let (problems, covered) = check(root, &policy);
    for problem in &problems {
        writeln!(log, "third-party: {problem}")?;
    }
    if problems.is_empty() {
        writeln!(
            log,
            "third-party: PASS: {}, each listed in {PATH} with an allowed licence (ADR-0018 §3)",
            count(covered, "vendored file")
        )?;
    } else {
        writeln!(
            log,
            "third-party: FAIL: {} with {PATH} (ADR-0018 §3, RFC-0062 §12.2)",
            count(problems.len(), "problem")
        )?;
    }
    log.flush()?;
    Ok(problems.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn own_files_are_the_manifest_and_top_level_locks() {
        assert!(own_file("third_party/licences.toml"));
        assert!(own_file("third_party/ggml.lock"));
        assert!(!own_file("third_party/ggml/ggml.lock"));
        assert!(!own_file("third_party/ggml/src/ggml.c"));
        assert!(!own_file("crates/kernels-cuda/vendor/x.lock"));
    }

    #[test]
    fn paths_must_be_plain_and_vendored() {
        let root = Path::new("/nonexistent-syntropy-root");
        for (path, expected) in [
            ("/third_party/a.c", "is not a plain root-relative path"),
            ("third_party/../a.c", "is not a plain root-relative path"),
            ("third_party\\a.c", "is not a plain root-relative path"),
            ("crates/core/src/lib.rs", "is not a vendored file"),
            ("third_party/licences.toml", "is not a vendored file"),
            ("third_party/a.c", "does not exist"),
        ] {
            let found = path_problem(root, path).unwrap_or_default();
            assert!(found.contains(expected), "{path}: {found}");
        }
    }

    #[test]
    fn a_notice_must_be_a_vendored_file_inside_the_root() {
        let dir =
            std::env::temp_dir().join(format!("syntropy-xtask-notice-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("root/third_party/a")).unwrap();
        std::fs::create_dir_all(dir.join("root/crates/core")).unwrap();
        std::fs::write(dir.join("root/third_party/a/LICENSE"), "MIT\n").unwrap();
        std::fs::write(dir.join("root/crates/core/lib.rs"), "\n").unwrap();
        std::fs::write(dir.join("outside.txt"), "secret\n").unwrap();
        let root = dir.join("root");
        assert_eq!(notice_problem(&root, "third_party/a/LICENSE"), None);
        for (notice, expected) in [
            ("../outside.txt", "is not a plain relative path"),
            ("/etc/passwd", "is not a plain relative path"),
            (
                "third_party/../../outside.txt",
                "is not a plain relative path",
            ),
            ("crates/core/lib.rs", "is not a vendored file"),
            ("third_party/a/MISSING", "does not exist"),
        ] {
            let found = notice_problem(&root, notice).unwrap_or_default();
            assert!(found.contains(expected), "{notice}: {found}");
        }
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(dir.join("outside.txt"), root.join("third_party/a/NOTICE"))
                .unwrap();
            let found = notice_problem(&root, "third_party/a/NOTICE").unwrap_or_default();
            assert!(found.contains("resolves outside"), "{found}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn every_key_is_known_and_present() {
        let dir =
            std::env::temp_dir().join(format!("syntropy-xtask-manifest-{}", std::process::id()));
        std::fs::create_dir_all(dir.join(THIRD_PARTY)).unwrap();
        std::fs::write(
            dir.join(PATH),
            "[[file]]\npath = \"third_party/a.c\"\nupstream = \"\"\nlicence = \"MIT\"\nversion = true\n",
        )
        .unwrap();
        let problems = load(&dir).unwrap_err();
        std::fs::remove_dir_all(&dir).unwrap();
        let joined = problems.join("\n");
        for expected in [
            "unknown key `licence`",
            "`version` must be a string",
            "`upstream` is empty",
            "`license` is missing",
            "`notice` is missing",
        ] {
            assert!(joined.contains(expected), "{expected}: {joined}");
        }
    }

    #[test]
    fn a_missing_manifest_has_no_entries() {
        assert_eq!(
            load(Path::new("/nonexistent-syntropy-root")),
            Ok(Vec::new())
        );
    }
}
