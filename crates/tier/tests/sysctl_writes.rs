// SPDX-License-Identifier: MIT OR Apache-2.0

//! RFC-0020 §9.1: no code path but `syntropy tune` writes `iogpu.wired_limit_mb`; the probe and
//! the tier manager only read it.
//!
//! The Testing row checks that with a syscall trace on the macOS rig. This test stands in for it
//! wherever the trace cannot run. It is a static check of the sources of every workspace crate:
//! - every `sysctlbyname` call passes a null new value and a new length of 0, so it only reads;
//! - the only production code that names a write of the sysctl, the `-w` command and the
//!   `name=value` argument, is `tier::tune::HostOp::set_wired_limit_argv`, which `tune` hands to
//!   the caller's approved runner.
//!
//! A static check sees only what the sources spell out, so it is evidence, not the trace.

use std::path::{Path, PathBuf};

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs")
            && path.file_name().is_some_and(|n| n != "tests.rs")
        {
            out.push(path);
        }
    }
}

/// The production sources: every crate's `src/`, without its unit-test modules (`tests.rs`).
fn production_sources() -> Vec<PathBuf> {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut files = Vec::new();
    for entry in std::fs::read_dir(&crates).unwrap().flatten() {
        rust_sources(&entry.path().join("src"), &mut files);
    }
    files.sort();
    assert!(files.len() > 100, "found only {} sources", files.len());
    files
}

/// `code` (already through [`code_only`]) with every `#[cfg(test)]` module left out: a
/// `mod x;` declaration up to its `;`, and a `mod x { … }` up to its matching brace. Test modules
/// are no code path in the sense of RFC-0020 §9.1 (the daemon's host-change ledger, for one, names
/// the sysctl only in its tests' fixtures); everything around them is still scanned, and a
/// `#[cfg(test)]` function or constant is not a module, so it stays.
fn without_test_modules(code: &str) -> String {
    let b = code.as_bytes();
    let mut out = String::with_capacity(code.len());
    let mut i = 0;
    while let Some(found) = code[i..].find("#[cfg(test)]") {
        let attr = i + found;
        let after = attr + "#[cfg(test)]".len();
        let item = code[after..].trim_start();
        let item_at = code.len() - item.len();
        let is_mod = ["mod ", "pub mod ", "pub(crate) mod ", "pub(super) mod "]
            .iter()
            .any(|p| item.starts_with(p));
        if !is_mod {
            out.push_str(&code[i..after]);
            i = after;
            continue;
        }
        out.push_str(&code[i..attr]);
        // The item ends at the first `;` or at the brace matching its first `{`.
        let mut j = item_at;
        let mut depth = 0usize;
        while j < b.len() {
            match b[j] {
                b'"' => {
                    j += 1;
                    while j < b.len() && b[j] != b'"' {
                        j += if b[j] == b'\\' { 2 } else { 1 };
                    }
                }
                b';' if depth == 0 => {
                    j += 1;
                    break;
                }
                b'{' => depth += 1,
                b'}' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        j += 1;
                        break;
                    }
                }
                _ => {}
            }
            j += 1;
        }
        i = j.min(b.len());
    }
    out.push_str(&code[i..]);
    out
}

/// `text` with every comment blanked out, string and char literals kept, so a call or a name
/// that only a comment mentions is not counted. Raw strings (`r"…"`, `r#"…"#`) are kept too.
fn code_only(text: &str) -> String {
    let b = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
            i += 2;
            while i < b.len() && !(b[i] == b'*' && b.get(i + 1) == Some(&b'/')) {
                i += 1;
            }
            i += 2;
        } else if b[i] == b'r' && matches!(b.get(i + 1), Some(b'"') | Some(b'#')) {
            let hashes = b[i + 1..].iter().take_while(|c| **c == b'#').count();
            let start = i;
            i += 1 + hashes + 1;
            let close: Vec<u8> = std::iter::once(b'"')
                .chain(std::iter::repeat_n(b'#', hashes))
                .collect();
            while i < b.len() && !b[i..].starts_with(&close) {
                i += 1;
            }
            i = (i + close.len()).min(b.len());
            out.push_str(&text[start..i]);
        } else if b[i] == b'"' {
            let start = i;
            i += 1;
            while i < b.len() && b[i] != b'"' {
                i += if b[i] == b'\\' { 2 } else { 1 };
            }
            i = (i + 1).min(b.len());
            out.push_str(&text[start..i]);
        } else if b[i] == b'\'' && (b.get(i + 2) == Some(&b'\'') || b.get(i + 1) == Some(&b'\\')) {
            // A char literal such as '"' or '\\'', so its quote is not read as a string.
            let start = i;
            i += if b.get(i + 1) == Some(&b'\\') { 4 } else { 3 };
            i = i.min(b.len());
            out.push_str(&text[start..i]);
        } else {
            let len = text[i..].chars().next().map_or(1, char::len_utf8);
            out.push_str(&text[i..i + len]);
            i += len;
        }
    }
    out
}

/// The arguments of the call whose opening parenthesis is at `open`, split at top-level commas.
fn call_args(text: &str, open: usize) -> Vec<String> {
    let mut depth = 0;
    let mut args = vec![String::new()];
    for c in text[open..].chars() {
        match c {
            '(' => {
                depth += 1;
                if depth == 1 {
                    continue;
                }
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            ',' if depth == 1 => {
                args.push(String::new());
                continue;
            }
            _ => {}
        }
        if let Some(last) = args.last_mut() {
            last.push(c);
        }
    }
    // A trailing comma leaves an empty last argument.
    args.into_iter()
        .map(|a| a.trim().to_owned())
        .filter(|a| !a.is_empty())
        .collect()
}

#[test]
fn every_sysctlbyname_call_only_reads() {
    let mut calls = 0;
    for path in production_sources() {
        let text = without_test_modules(&code_only(&std::fs::read_to_string(&path).unwrap()));
        for (at, _) in text.match_indices("sysctlbyname(") {
            let before = &text[..at];
            // The declaration, not a call.
            if before.trim_end().ends_with("fn") {
                continue;
            }
            let args = call_args(&text, at + "sysctlbyname".len());
            assert_eq!(args.len(), 5, "{}: {args:?}", path.display());
            assert!(
                args[3].contains("null"),
                "{} writes a sysctl: new value {}",
                path.display(),
                args[3]
            );
            assert_eq!(args[4], "0", "{}: new length {}", path.display(), args[4]);
            calls += 1;
        }
    }
    assert!(calls >= 1, "no sysctlbyname call found: the scan is broken");
}

#[test]
fn only_tune_spells_out_the_write() {
    let mut writers = Vec::new();
    for path in production_sources() {
        let text = std::fs::read_to_string(&path).unwrap();
        if text.contains("/usr/sbin/sysctl") || text.contains("WIRED_LIMIT_SYSCTL}=") {
            writers.push(path);
        }
    }
    let tune = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tune.rs");
    assert_eq!(
        writers
            .iter()
            .map(|p| p.canonicalize().unwrap())
            .collect::<Vec<_>>(),
        [tune.canonicalize().unwrap()]
    );
}

/// The sysctl write forms in `text`, the code of the source file `shown` (see
/// [`no_other_write_form_appears`]).
fn write_forms(shown: &str, text: &str) -> Vec<String> {
    let mut found = Vec::new();
    if text.contains("Command::new(\"sysctl\")") {
        found.push(format!("{shown}: runs the sysctl tool"));
    }
    if text.contains("\"iogpu.wired_limit_mb") && !READERS.contains(&shown) {
        found.push(format!("{shown}: names iogpu.wired_limit_mb"));
    }
    // `libc::sysctl` named at all, not only called: a `use libc::sysctl;` (or `as s`) makes every
    // later bare `sysctl(…)` the MIB-form call (review of #1822, N1).
    let word = |c: char| c.is_alphanumeric() || c == '_';
    for (at, m) in text.match_indices("libc::sysctl") {
        if !text[at + m.len()..].chars().next().is_some_and(word) {
            found.push(format!("{shown}: names libc::sysctl"));
        }
    }
    // `sysctl` inside a `libc::{…}` import.
    for (at, _) in text.match_indices("libc::{") {
        let list = &text[at..text[at..].find('}').map_or(text.len(), |end| at + end)];
        if list.split(|c: char| !word(c)).any(|name| name == "sysctl") {
            found.push(format!("{shown}: imports libc::sysctl"));
        }
    }
    if text.contains("sysctlnametomib(") {
        found.push(format!("{shown}: a sysctlnametomib(…) call"));
    }
    // A raw-pointer declaration of the MIB-form `sysctl`: its calls would read as a bare
    // `sysctl(…)`, which no other rule sees. `sysctlbyname` declarations are fine, because
    // `every_sysctlbyname_call_only_reads` checks every call however it is declared.
    for (at, _) in text.match_indices("fn sysctl(") {
        let args = call_args(text, at + "fn sysctl(".len() - 1).join(",");
        if args.contains("*mut") || args.contains("*const") {
            found.push(format!("{shown}: an FFI declaration of fn sysctl(…)"));
        }
    }
    found
}

/// Where production code may name `iogpu.wired_limit_mb` to read it (RFC-0020 §9.1: the probe
/// reads the limit). Every read reaches the kernel through `syntropy_io::os::macos_host`, whose
/// `sysctlbyname` calls `every_sysctlbyname_call_only_reads` checks.
const READERS: &[&str] = &["probe/src/macos/mod.rs"];

/// Other ways to write the sysctl that a source could spell out (review of #1822, N1): the
/// `sysctl` tool found on `PATH`, the MIB-form `libc::sysctl()` and `sysctlnametomib()` calls,
/// an FFI declaration of the MIB-form `sysctl` that would bypass `libc`, and the sysctl's
/// name outside the read sites in [`READERS`]. None may appear outside `tier/src/tune.rs`.
/// Comments are not counted. Rust functions named `sysctl` that take a name, such as `io`'s
/// read-only wrapper and the probe's `MacApi::sysctl`, end in `libc::sysctlbyname`, which
/// `every_sysctlbyname_call_only_reads` checks, so they are not counted either.
#[test]
fn no_other_write_form_appears() {
    let tune = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/tune.rs")
        .canonicalize()
        .unwrap();
    let crates = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .canonicalize()
        .unwrap();
    let mut found = Vec::new();
    for path in production_sources() {
        let path = path.canonicalize().unwrap();
        if path == tune {
            continue;
        }
        let shown = path
            .strip_prefix(&crates)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        let text = without_test_modules(&code_only(&std::fs::read_to_string(&path).unwrap()));
        found.extend(write_forms(&shown, &text));
    }
    assert!(
        found.is_empty(),
        "sysctl write forms outside tune: {found:#?}"
    );
}

/// The scan's own rules catch each write form, and skip what only a comment says.
#[test]
fn the_scan_catches_each_write_form() {
    let text = code_only(
        "// libc::sysctl(mib, 2, old, oldlen, new, newlen) in a comment\n\
         let s = \"http://x\"; let n = 1; /* sysctlnametomib( */\n",
    );
    assert!(!text.contains("libc::sysctl(") && !text.contains("sysctlnametomib("));
    assert!(text.contains("\"http://x\"") && text.contains("let n = 1;"));
    let ffi = code_only("extern \"C\" { fn sysctl(name: *mut c_int, len: u32); }");
    let at = ffi.find("fn sysctl(").unwrap();
    assert!(
        call_args(&ffi, at + "fn sysctl".len())
            .join(",")
            .contains("*mut")
    );
}

/// Each rule finds its planted form in code and ignores it in a comment (review of #1822, N2).
#[test]
fn each_rule_finds_its_planted_form() {
    for (plant, expect) in [
        (
            "libc::sysctl(mib.as_ptr(), 2, old, len, new, n)",
            "names libc::sysctl",
        ),
        ("use libc::sysctl;", "names libc::sysctl"),
        ("use libc::sysctl as s;", "names libc::sysctl"),
        ("use libc::{c_int, sysctl};", "imports libc::sysctl"),
        (
            "libc::sysctlnametomib(name, mib, len)",
            "a sysctlnametomib(…) call",
        ),
        (
            "unsafe extern \"C\" { fn sysctl(name: *mut c_int, n: u32) -> c_int; }",
            "an FFI declaration",
        ),
        ("Command::new(\"sysctl\")", "runs the sysctl tool"),
        (
            "let n = \"iogpu.wired_limit_mb\";",
            "names iogpu.wired_limit_mb",
        ),
    ] {
        let found = write_forms("x/src/lib.rs", &code_only(plant));
        assert!(
            found.iter().any(|f| f.contains(expect)),
            "{plant}: {found:?}"
        );
        let commented = write_forms("x/src/lib.rs", &code_only(&format!("// {plant}")));
        assert!(commented.is_empty(), "{plant} in a comment: {commented:?}");
    }
    // Neither the read-only sysctlbyname nor a name that merely starts with sysctl counts.
    assert!(
        write_forms(
            "x/src/lib.rs",
            "libc::sysctlbyname(n, o, l, ptr::null_mut(), 0)"
        )
        .is_empty()
    );
    assert!(
        write_forms(
            "probe/src/macos/mod.rs",
            "const W: &str = \"iogpu.wired_limit_mb\";"
        )
        .is_empty()
    );
}

/// A file's test modules are left out of the scan, and nothing else is: code after a `mod tests;`
/// declaration, after a closed inline test module and after a `#[cfg(test)]` helper module is
/// still scanned, and a `#[cfg(test)]` item that is not a module stays (review of #1822, B1).
#[test]
fn only_test_modules_are_left_out() {
    let file = "#[cfg(test)]\nmod tests;\nfn a() { libc::sysctl(m) }\n\
                #[cfg(test)]\nmod inline {\n    const N: &str = \"iogpu.wired_limit_mb}\";\n    fn f() { if x { y } }\n}\n\
                fn b() { libc::sysctl(n) }\n\
                #[cfg(test)]\npub(crate) mod build { fn g() {} }\nfn c() { libc::sysctl(o) }\n\
                #[cfg(test)]\nfn helper() {}\n";
    let kept = without_test_modules(&code_only(file));
    for still in ["fn a()", "fn b()", "fn c()", "fn helper()"] {
        assert!(kept.contains(still), "{still} dropped: {kept}");
    }
    assert!(
        !kept.contains("iogpu") && !kept.contains("fn f()") && !kept.contains("fn g()"),
        "{kept}"
    );
    assert_eq!(write_forms("x/src/lib.rs", &kept).len(), 3, "{kept}");
    // A test module only a comment mentions starts no skip.
    let commented = without_test_modules(&code_only(
        "// #[cfg(test)] mod t {\nfn d() { libc::sysctl(p) }\n",
    ));
    assert!(commented.contains("fn d()"), "{commented}");
}
