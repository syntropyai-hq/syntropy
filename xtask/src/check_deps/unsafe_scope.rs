// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rule 8's module-scoped exception (RFC-0001 §4.2 rule 8, note 2026-10-22).
//!
//! A crate in [`policy::UNSAFE_MODULES`] may hold exactly one `unsafe impl` of one trait, in one
//! file, under its one `#[allow(unsafe_code)]`. For such a crate the check:
//!
//! - requires `#![deny(unsafe_code)]` (or `forbid`) among each production crate root's leading
//!   inner attributes, so the compiler refuses `unsafe` everywhere the `allow` does not reach;
//! - reads every `.rs` file under the directory of each production target's root (the library's
//!   and every bin's), with comments and literals blanked, so a mention in prose never counts;
//! - refuses any `unsafe` token, and any lint attribute naming `unsafe_code`, outside the one
//!   allowed item, apart from the root's own leading level;
//! - requires the one `#[allow(unsafe_code)]` to sit on `unsafe impl <path ending in the trait>
//!   for …`, and every `unsafe` in that file to lie within that item;
//! - refuses every way to compile code it does not read: a `path` in an attribute (`#[path]`,
//!   also inside `cfg_attr`), an attribute built from a macro metavariable, which could become
//!   `#[path]`, the identifier `include` anywhere, which covers `include!` under any name or
//!   path, and a build script, whose output only `include!` could bring in.
//!
//! A macro exported by another crate can still emit `#[path]` or `include!` from tokens that
//! name neither; that is the same limit as `unsafe` from another crate's macro (issue #1527).
//!
//! With those refused, the compiler can reach a module only through `mod` declarations, which
//! resolve to `.rs` files under the directory of the file that declares them, so the files read
//! are every file the crate's production targets compile. A text check covers every `cfg` at
//! once, where the compiler's own list of the files it read covers one configuration.
//!
//! The text checks do not depend on the compiler, so a stray `allow` or an `unsafe` the compiler
//! would also refuse is reported here by its file and line.

use std::path::{Path, PathBuf};

use super::crate_roots::{denies_unsafe_code, leading_attributes, shown_path};
use super::graph::Graph;
use super::policy::UnsafeModule;
use super::{Finding, Rule};

/// The rule-8 findings for `module`'s crate, member `at` of `graph`.
pub fn check(graph: &Graph, at: usize, module: &UnsafeModule) -> Vec<Finding> {
    let name = graph.short(at);
    let package = graph.package(at);
    let finding = |message: String| Finding::new(Rule::Numbered(8), message);
    let mut findings = Vec::new();
    // Each production crate root must deny `unsafe_code`.
    let mut roots = Vec::new();
    for target in &package.targets {
        if !target.kind.iter().any(|kind| production(kind)) {
            continue;
        }
        let shown = shown_path(&graph.root, &target.src_path);
        match std::fs::read_to_string(&target.src_path) {
            Ok(source) if denies_unsafe_code(&source) => roots.push(target.src_path.clone()),
            Ok(_) => findings.push(finding(format!(
                "`{name}` does not set `#![deny(unsafe_code)]` or `#![forbid(unsafe_code)]` in \
                 {shown} ({} target `{}`); its `unsafe` exception needs one",
                target.kind.join(", "),
                target.name
            ))),
            Err(error) => findings.push(finding(format!(
                "cannot read {shown}, the crate root of `{name}`: {error}"
            ))),
        }
    }
    // A build script's output can reach the crate only through `include!`, which is refused
    // below; the script is refused too, so generated code has no way in at all.
    for target in &package.targets {
        if target.kind.iter().any(|kind| kind == "custom-build") {
            findings.push(finding(format!(
                "`{name}` has a build script, {}; its `unsafe` exception allows none, because \
                 the check cannot read the code a build script generates",
                shown_path(&graph.root, &target.src_path)
            )));
        }
    }
    // The crate's directory, the parent of the one that holds its library root.
    let Some(crate_dir) = package
        .targets
        .iter()
        .find(|target| target.kind.iter().any(|kind| kind == "lib"))
        .and_then(|lib| lib.src_path.parent())
        .and_then(Path::parent)
        .map(Path::to_path_buf)
    else {
        return findings;
    };
    let allowed = crate_dir.join(module.file);
    // The directory of every production target's root, without one that another one holds.
    let mut dirs: Vec<PathBuf> = package
        .targets
        .iter()
        .filter(|target| target.kind.iter().any(|kind| production(kind)))
        .filter_map(|target| target.src_path.parent().map(Path::to_path_buf))
        .collect();
    dirs.sort();
    dirs.dedup();
    let held = |dir: &PathBuf| {
        dirs.iter()
            .any(|other| other != dir && dir.starts_with(other))
    };
    let dirs: Vec<PathBuf> = dirs.iter().filter(|dir| !held(dir)).cloned().collect();
    let mut files = Vec::new();
    for dir in &dirs {
        if let Err(error) = rust_files(dir, &mut files) {
            findings.push(finding(format!(
                "cannot list {}: {error}",
                shown_path(&graph.root, dir)
            )));
        }
    }
    files.sort();
    files.dedup();
    for file in files {
        let shown = shown_path(&graph.root, &file);
        let source = match std::fs::read_to_string(&file) {
            Ok(source) => source,
            Err(error) => {
                findings.push(finding(format!("cannot read {shown}: {error}")));
                continue;
            }
        };
        let problems = if file == allowed {
            check_allowed_file(&source, module)
        } else {
            let level_end = roots.contains(&file).then(|| leading_attributes(&source).1);
            check_other_file(&source, level_end)
        };
        for problem in problems {
            findings.push(finding(format!("`{name}`: {shown}:{problem}")));
        }
    }
    findings
}

/// Whether a target of `kind` is built into the crate itself, as a library or a binary.
fn production(kind: &str) -> bool {
    matches!(kind, "lib" | "rlib" | "bin" | "proc-macro")
}

/// Every `.rs` file under `dir`, recursively.
fn rust_files(dir: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            rust_files(&path, files)?;
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
    Ok(())
}

/// The problems in a file other than the allowed one, each as `<line>: <what>`. `level_end` is
/// where a crate root's leading attributes end; a lint level for `unsafe_code` before it is the
/// root's own.
fn check_other_file(source: &str, level_end: Option<usize>) -> Vec<String> {
    let code = code_only(source);
    let mut problems = unread_code(&code);
    for (at, ident) in idents(&code) {
        match ident {
            "unsafe" => problems.push(format!(
                "{}: uses `unsafe` outside the one allowed impl",
                line_of(&code, at)
            )),
            "unsafe_code" if level_end.is_none_or(|end| at >= end) => problems.push(format!(
                "{}: sets a lint level for `unsafe_code`; only the crate root's leading \
                 `#![deny(unsafe_code)]` and the one `#[allow(unsafe_code)]` on the allowed impl \
                 may",
                line_of(&code, at)
            )),
            _ => {}
        }
    }
    problems
}

/// The problems in the file that holds the allowed impl.
fn check_allowed_file(source: &str, module: &UnsafeModule) -> Vec<String> {
    let code = code_only(source);
    let mentions: Vec<usize> = idents(&code)
        .filter(|(_, ident)| *ident == "unsafe_code")
        .map(|(at, _)| at)
        .collect();
    let unsafes: Vec<usize> = idents(&code)
        .filter(|(_, ident)| *ident == "unsafe")
        .map(|(at, _)| at)
        .collect();
    let mut problems = unread_code(&code);
    if mentions.len() != 1 {
        problems.push(format!(
            "{}: names `unsafe_code` {} times; exactly one `#[allow(unsafe_code)]`, on \
             `unsafe impl {} for …`, is allowed",
            mentions.first().map_or(1, |&at| line_of(&code, at)),
            mentions.len(),
            module.implements
        ));
    }
    // The span the `allow` covers: from its `#` to the end of the impl's body.
    let span = mentions
        .first()
        .and_then(|&at| allowed_item(&code, at, module.implements));
    if let (Some(&at), None) = (mentions.first(), span) {
        problems.push(format!(
            "{}: the `unsafe_code` lint is set here, but only `#[allow(unsafe_code)]` directly on \
             `unsafe impl {} for …` is allowed",
            line_of(&code, at),
            module.implements
        ));
    }
    for at in unsafes {
        if !span.is_some_and(|(start, end)| (start..end).contains(&at)) {
            problems.push(format!(
                "{}: uses `unsafe` outside the allowed `unsafe impl {}`",
                line_of(&code, at),
                module.implements
            ));
        }
    }
    problems
}

/// Every way `code` could compile a file this check does not read, each as `<line>: <what>`:
/// a `path` set in an attribute, an attribute built from a macro's metavariable, which could
/// carry a `path`, and the `include` macro under any name. `include` is refused as an
/// identifier anywhere, because a macro can call it through a metavariable (`$m!(…)`) with only
/// the bare name in its input.
fn unread_code(code: &str) -> Vec<String> {
    let attributes = attribute_spans(code);
    let in_attribute = |at: usize| {
        attributes
            .iter()
            .any(|&(start, end)| (start..end).contains(&at))
    };
    let mut problems = Vec::new();
    let mut found = |at: usize, what: &str| {
        problems.push(format!(
            "{}: {what}; `unsafe` is denied, not forbidden, here, so every file must be read",
            line_of(code, at)
        ));
    };
    for (at, ident) in idents(code) {
        let after = skip_space(code, at + ident.len());
        match ident {
            "include" => found(
                at,
                "names `include`, the macro that compiles a file this check does not read",
            ),
            "path"
                if code[after..].starts_with('=')
                    && !code[after..].starts_with("==")
                    && in_attribute(at) =>
            {
                found(
                    at,
                    "sets a module `path` in an attribute, which can load a file this check does \
                     not read",
                );
            }
            _ => {}
        }
    }
    // An attribute that a metavariable fills in, `#[$m]`, `#[cfg_attr(x, $m)]` or `# $t`, can
    // become `#[path = …]` without the text above.
    for (at, _) in code.match_indices('#') {
        let mut next = skip_space(code, at + 1);
        if code[next..].starts_with('!') {
            next = skip_space(code, next + 1);
        }
        let built = code[next..].starts_with('$')
            || attributes
                .iter()
                .find(|&&(start, _)| start == at)
                .is_some_and(|&(start, end)| code[start..end].contains('$'));
        if built {
            found(
                at,
                "builds an attribute from a macro metavariable, which could set a module `path`",
            );
        }
    }
    problems.sort_by_key(|problem| {
        problem
            .split(':')
            .next()
            .and_then(|line| line.parse::<usize>().ok())
    });
    problems
}

/// The span of every attribute in `code`, `#[…]` or `#![…]`, from `#` to its closing `]`.
fn attribute_spans(code: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    for (at, _) in code.match_indices('#') {
        let mut open = skip_space(code, at + 1);
        if code[open..].starts_with('!') {
            open = skip_space(code, open + 1);
        }
        if !code[open..].starts_with('[') {
            continue;
        }
        let mut depth = 0usize;
        for (offset, c) in code[open..].char_indices() {
            match c {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        spans.push((at, open + offset + 1));
                        break;
                    }
                }
                _ => {}
            }
        }
    }
    spans
}

/// When the `unsafe_code` at `at` is inside an outer `#[allow(unsafe_code)]` on `unsafe impl
/// <path ending in trait> for …`, the span from the attribute's `#` to the impl's closing `}`.
fn allowed_item(code: &str, at: usize, implements: &str) -> Option<(usize, usize)> {
    // The attribute: `#[` … `]` around `at`, an outer one, saying exactly `allow(unsafe_code)`.
    let open = code[..at].rfind('[')?;
    let hash = code[..open].trim_end();
    if !hash.ends_with('#') {
        return None;
    }
    let start = hash.len() - 1;
    let close = open + code[open..].find(']')?;
    let body: String = code[open + 1..close].split_whitespace().collect();
    if body != "allow(unsafe_code)" {
        return None;
    }
    // The item it is on: other outer attributes may come first.
    let mut rest = close + 1;
    loop {
        let next = skip_space(code, rest);
        if code[next..].starts_with("#[") {
            rest = next + code[next..].find(']')? + 1;
        } else {
            rest = next;
            break;
        }
    }
    let mut words = idents(&code[rest..]);
    let (unsafe_at, first) = words.next()?;
    let (impl_at, second) = words.next()?;
    if first != "unsafe" || second != "impl" || !code[rest..rest + unsafe_at].trim().is_empty() {
        return None;
    }
    // `impl<…> path for`: the path's last segment must be the trait.
    let after_impl = rest + impl_at + "impl".len();
    let header_end = after_impl + code[after_impl..].find('{')?;
    let header = &code[after_impl..header_end];
    let (path, _) = header.split_once(" for ")?;
    let path = path.trim();
    let path = match path.strip_prefix('<') {
        Some(generics) => generics.split_once('>')?.1.trim(),
        None => path,
    };
    if path.rsplit("::").next()? != implements {
        return None;
    }
    let end = matching_brace(code, header_end)?;
    Some((start, end + 1))
}

/// The index of the `}` that closes the `{` at `open`.
fn matching_brace(code: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (at, c) in code[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + at);
                }
            }
            _ => {}
        }
    }
    None
}

fn skip_space(code: &str, from: usize) -> usize {
    from + (code[from..].len() - code[from..].trim_start().len())
}

/// The 1-based line of byte offset `at`.
fn line_of(code: &str, at: usize) -> usize {
    code[..at].matches('\n').count() + 1
}

/// Every ASCII identifier or keyword in `code`, with its byte offset.
fn idents(code: &str) -> impl Iterator<Item = (usize, &str)> {
    let bytes = code.as_bytes();
    let mut at = 0;
    std::iter::from_fn(move || {
        while at < bytes.len() {
            let start = at;
            if bytes[at].is_ascii_alphabetic() || bytes[at] == b'_' {
                while at < bytes.len() && (bytes[at].is_ascii_alphanumeric() || bytes[at] == b'_') {
                    at += 1;
                }
                return Some((start, &code[start..at]));
            }
            at += 1;
        }
        None
    })
}

/// `source` with every comment and every string, character and byte literal replaced by spaces,
/// byte for byte, so offsets and line numbers stay the same. Lifetimes and labels are kept.
pub fn code_only(source: &str) -> String {
    let chars: Vec<(usize, char)> = source.char_indices().collect();
    let mut out = String::with_capacity(source.len());
    let blank = |out: &mut String, c: char| {
        if c == '\n' {
            out.push('\n');
        } else {
            out.extend(std::iter::repeat_n(' ', c.len_utf8()));
        }
    };
    let ident = |c: char| c.is_alphanumeric() || c == '_';
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i].1;
        let next = chars.get(i + 1).map(|&(_, c)| c);
        let previous = i.checked_sub(1).map(|p| chars[p].1);
        if c == '/' && next == Some('/') {
            while i < chars.len() && chars[i].1 != '\n' {
                blank(&mut out, chars[i].1);
                i += 1;
            }
        } else if c == '/' && next == Some('*') {
            let mut depth = 0usize;
            while i < chars.len() {
                let (c, next) = (chars[i].1, chars.get(i + 1).map(|&(_, c)| c));
                if c == '/' && next == Some('*') {
                    depth += 1;
                    blank(&mut out, c);
                    blank(&mut out, '*');
                    i += 2;
                } else if c == '*' && next == Some('/') {
                    depth -= 1;
                    blank(&mut out, c);
                    blank(&mut out, '/');
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    blank(&mut out, c);
                    i += 1;
                }
            }
        } else if c == 'r'
            && previous.is_none_or(|p| !ident(p) || matches!(p, 'b' | 'c'))
            && raw_string_hashes(&chars, i + 1).is_some()
        {
            // r"…", r#"…"#, and br/cr forms: blank through the matching quote and hashes.
            let hashes = raw_string_hashes(&chars, i + 1).unwrap_or(0);
            let mut j = i + 1 + hashes + 1;
            while j < chars.len() {
                if chars[j].1 == '"'
                    && (1..=hashes).all(|h| chars.get(j + h).is_some_and(|&(_, c)| c == '#'))
                {
                    j += 1 + hashes;
                    break;
                }
                j += 1;
            }
            for &(_, c) in &chars[i..j.min(chars.len())] {
                blank(&mut out, c);
            }
            i = j;
        } else if c == '"' {
            blank(&mut out, c);
            i += 1;
            while i < chars.len() {
                let c = chars[i].1;
                blank(&mut out, c);
                i += 1;
                if c == '\\' {
                    if let Some(&(_, escaped)) = chars.get(i) {
                        blank(&mut out, escaped);
                        i += 1;
                    }
                } else if c == '"' {
                    break;
                }
            }
        } else if c == '\''
            && (next == Some('\\') || chars.get(i + 2).is_some_and(|&(_, c)| c == '\''))
        {
            // A character literal: `'x'` or an escape such as `'\n'` or `'\u{1F600}'`.
            blank(&mut out, c);
            i += 1;
            while i < chars.len() {
                let c = chars[i].1;
                blank(&mut out, c);
                i += 1;
                if c == '\\' {
                    if let Some(&(_, escaped)) = chars.get(i) {
                        blank(&mut out, escaped);
                        i += 1;
                    }
                } else if c == '\'' || c == '\n' {
                    break;
                }
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

/// When the characters from `from` are `#`… then `"`, the number of `#`s: a raw string's opening.
fn raw_string_hashes(chars: &[(usize, char)], from: usize) -> Option<usize> {
    let mut hashes = 0;
    while chars.get(from + hashes).is_some_and(|&(_, c)| c == '#') {
        hashes += 1;
    }
    chars
        .get(from + hashes)
        .is_some_and(|&(_, c)| c == '"')
        .then_some(hashes)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TESTKIT: UnsafeModule = UnsafeModule {
        krate: "testkit",
        file: "src/guards.rs",
        implements: "GlobalAlloc",
    };

    /// A guards file of the allowed shape: the impl, with `unsafe fn`s and `unsafe` blocks inside.
    const ALLOWED: &str = r#"//! The guards. `unsafe` in a comment, and "unsafe_code" in a string, do not count.

use std::alloc::{GlobalAlloc, Layout, System};

pub struct CountingAlloc;

// SAFETY: every method delegates to `System`.
#[allow(unsafe_code)]
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: as above.
        unsafe { System.dealloc(ptr, layout) }
    }
}

pub fn alloc_count_this_thread() -> u64 {
    let label = 'outer: loop { break 'outer 0 };
    label
}
"#;

    #[test]
    fn the_allowed_shape_passes() {
        assert_eq!(check_allowed_file(ALLOWED, &TESTKIT), Vec::<String>::new());
    }

    #[test]
    fn a_path_qualified_trait_and_other_attributes_pass() {
        let source = "#[allow(unsafe_code)]\n#[doc(hidden)]\nunsafe impl std::alloc::GlobalAlloc for A {\n    unsafe fn alloc(&self, l: L) -> *mut u8 { unsafe { S.alloc(l) } }\n}\n";
        assert_eq!(check_allowed_file(source, &TESTKIT), Vec::<String>::new());
    }

    #[test]
    fn unsafe_outside_the_allowed_impl_fails() {
        let source = format!(
            "{ALLOWED}\npub fn sneaky() {{ unsafe {{ core::hint::unreachable_unchecked() }} }}\n"
        );
        let problems = check_allowed_file(&source, &TESTKIT);
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(
            problems[0].contains("uses `unsafe` outside the allowed"),
            "{problems:?}"
        );
    }

    #[test]
    fn a_second_allow_or_a_module_wide_allow_fails() {
        let second = format!("{ALLOWED}\n#[allow(unsafe_code)]\npub fn f() {{}}\n");
        assert!(
            check_allowed_file(&second, &TESTKIT)
                .iter()
                .any(|p| p.contains("names `unsafe_code` 2 times")),
            "{:?}",
            check_allowed_file(&second, &TESTKIT)
        );
        let inner = ALLOWED.replace("#[allow(unsafe_code)]", "#![allow(unsafe_code)]");
        let problems = check_allowed_file(&inner, &TESTKIT);
        assert!(
            problems
                .iter()
                .any(|p| p.contains("only `#[allow(unsafe_code)]` directly on")),
            "{problems:?}"
        );
    }

    #[test]
    fn an_allow_on_another_item_fails() {
        for source in [
            "#[allow(unsafe_code)]\npub fn f() { unsafe { g() } }\n",
            "#[allow(unsafe_code)]\nunsafe impl Send for A {}\n",
            "#[allow(unsafe_code, dead_code)]\nunsafe impl GlobalAlloc for A {}\n",
            "#[allow(unsafe_code)]\nunsafe impl NotGlobalAlloc for A {}\n",
        ] {
            let problems = check_allowed_file(source, &TESTKIT);
            assert!(
                problems
                    .iter()
                    .any(|p| p.contains("only `#[allow(unsafe_code)]` directly on")),
                "{source}: {problems:?}"
            );
        }
    }

    #[test]
    fn a_guards_file_without_the_impl_and_without_unsafe_is_refused_for_its_missing_allow() {
        // No `allow` at all is no `unsafe` at all: still exactly one is required, so the file
        // and the policy entry do not drift apart.
        let problems = check_allowed_file("pub fn f() {}\n", &TESTKIT);
        assert!(
            problems
                .iter()
                .any(|p| p.contains("names `unsafe_code` 0 times")),
            "{problems:?}"
        );
    }

    #[test]
    fn other_files_may_not_use_unsafe_or_set_its_level() {
        let problems = check_other_file("fn f() {\n    unsafe { g() }\n}\n", None);
        assert_eq!(problems, ["2: uses `unsafe` outside the one allowed impl"]);
        let problems = check_other_file("#[allow(unsafe_code)]\nfn f() {}\n", None);
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(problems[0].starts_with("1: sets a lint level for `unsafe_code`"));
        let problems = check_other_file("#![cfg_attr(test, allow(unsafe_code))]\n", None);
        assert_eq!(problems.len(), 1, "{problems:?}");
    }

    #[test]
    fn the_root_may_set_its_own_level_only_in_its_leading_attributes() {
        let root = "//! Docs about `unsafe`.\n\n#![deny(unsafe_code)]\n\npub mod guards;\n";
        let (_, end) = leading_attributes(root);
        assert_eq!(check_other_file(root, Some(end)), Vec::<String>::new());
        let late = "#![deny(unsafe_code)]\npub mod guards;\n#[allow(unsafe_code)]\nfn f() {}\n";
        let (_, end) = leading_attributes(late);
        assert_eq!(check_other_file(late, Some(end)).len(), 1);
    }

    #[test]
    fn comments_and_literals_are_blanked_byte_for_byte() {
        let source = "a // unsafe\nb /* unsafe /* nested */ unsafe */ c \"unsafe\\\" still\" d 'u' e '\\'' f r#\"unsafe \"# g b\"unsafe\" h 'a i é\n";
        let code = code_only(source);
        assert_eq!(code.len(), source.len());
        assert_eq!(code.matches('\n').count(), source.matches('\n').count());
        let words: Vec<&str> = idents(&code).map(|(_, w)| w).collect();
        assert_eq!(
            words,
            ["a", "b", "c", "d", "e", "f", "g", "b", "h", "a", "i"]
        );
    }

    #[test]
    fn a_module_path_in_any_attribute_is_refused() {
        for source in [
            "#[path = \"../outside.rs\"]\npub mod probe;\n",
            "#[cfg_attr(unix, path = \"unix.rs\")]\nmod probe;\n",
            "#![cfg_attr(all(), path=\"x.rs\")]\n",
            "mod a {\n    # [ path = \"b.rs\" ]\n    mod b;\n}\n",
        ] {
            let problems = check_other_file(source, None);
            assert_eq!(problems.len(), 1, "{source}: {problems:?}");
            assert!(
                problems[0].contains("sets a module `path` in an attribute"),
                "{source}: {problems:?}"
            );
        }
        // In a macro body, the `path` and the metavariable are each refused.
        let problems = check_other_file(
            "macro_rules! m { ($p:literal) => { #[path = $p] mod x; } }\n",
            None,
        );
        assert_eq!(problems.len(), 2, "{problems:?}");
        // The allowed file is held to the same rule.
        let problems =
            check_allowed_file(&format!("{ALLOWED}#[path = \"x.rs\"]\nmod x;\n"), &TESTKIT);
        assert_eq!(problems.len(), 1, "{problems:?}");
    }

    #[test]
    fn include_by_any_name_is_refused() {
        for source in [
            "include!(\"probe.inc\");\n",
            "core::include ! (\"x.rs\");\n",
            "include!(concat!(env!(\"OUT_DIR\"), \"/gen.rs\"));\n",
            "macro_rules! m { () => { include!(\"x\") } }\n",
            "use core::include as inc;\n",
            "pub use std::{include, vec};\n",
            // A macro that calls whatever it is given, given the bare name.
            "m!(include);\n",
            "let include = true;\n",
        ] {
            let problems = check_other_file(source, None);
            assert_eq!(problems.len(), 1, "{source}: {problems:?}");
            assert!(
                problems[0].contains("names `include`"),
                "{source}: {problems:?}"
            );
        }
    }

    #[test]
    fn an_attribute_built_from_a_metavariable_is_refused() {
        for source in [
            "macro_rules! m { ($p:ident $v:literal) => { #[$p = $v] mod x; } }\n",
            "macro_rules! m { ($a:meta) => { #[cfg_attr(all(), $a)] mod x; } }\n",
            "macro_rules! m { ($t:tt) => { # $t mod x; } }\n",
            "macro_rules! m { ($($a:meta)*) => { $(#[$a])* mod x; } }\n",
            "macro_rules! m { ($t:tt) => { #! $t } }\n",
        ] {
            let problems = check_other_file(source, None);
            assert_eq!(problems.len(), 1, "{source}: {problems:?}");
            assert!(
                problems[0].contains("builds an attribute from a macro metavariable"),
                "{source}: {problems:?}"
            );
        }
    }

    #[test]
    fn ordinary_uses_of_the_words_path_and_include_pass() {
        let source = "\
            #![doc = include_str!(\"../README.md\")]\n\
            const DATA: &[u8] = include_bytes!(\"data.bin\");\n\
            #[doc = \"path = x and include!(y)\"]\n\
            pub fn includes(path: &str) -> bool { let n = path.len(); n != 0 }\n\
            pub struct Filter { pub included: bool, pub path: String }\n\
            fn f() { let path = 1; if path == 1 {} }\n\
            macro_rules! twice { ($e:expr) => { $e + $e } }\n\
            use std::path::Path;\n";
        assert_eq!(check_other_file(source, None), Vec::<String>::new());
    }

    #[test]
    fn identifiers_containing_unsafe_are_not_the_keyword() {
        assert_eq!(
            check_other_file("fn not_unsafe() {}\nlet unsafely = 1;\n", None),
            Vec::<String>::new()
        );
    }
}
