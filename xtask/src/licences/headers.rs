// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask spdx`: every Syntropy-authored source file starts with
//! `SPDX-License-Identifier: MIT OR Apache-2.0` (ADR-0018 §1, RFC-0062 §12.2).
//!
//! The check covers Rust (build scripts included), CUDA, HIP, Metal Shading Language, Slang, the
//! C and C++ sources and headers kernels share, and CMake and shell build scripts. The header is
//! the first line, or the second after a `#!` line, written as a `//`, `#` or `/* */` comment.
//! Vendored third-party code, which keeps its own licence, is skipped: `third_party/` and every
//! `crates/kernels-*/vendor/`, which `cargo xtask third-party` checks instead. So are build
//! output (`target/`) and `.git/`.
//!
//! The files checked are the ones git tracks when the root is the top of a git work tree, as in
//! CI's checkout, so an untracked build or vendoring output in a local tree does not fail the
//! check. Anywhere else, such as a test fixture, the check walks the tree and checks every file.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use super::{count, is_vendored, relative};

/// The header every covered file carries.
pub const HEADER: &str = "SPDX-License-Identifier: MIT OR Apache-2.0";

/// File extensions the check covers: Rust; CUDA; HIP; Metal; Slang; C and C++; CMake and shell.
const EXTENSIONS: &[&str] = &[
    "rs", "cu", "cuh", "hip", "metal", "slang", "c", "cc", "cpp", "cxx", "h", "hh", "hpp", "hxx",
    "cmake", "sh", "bash",
];

/// File names the check covers whatever their extension.
const NAMES: &[&str] = &["CMakeLists.txt"];

/// Directory names never entered, at any depth.
const SKIPPED_DIRS: &[&str] = &[".git", "target"];

/// Whether the check covers the file at `path`.
pub fn covered(path: &Path) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    NAMES.contains(&name)
        || path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| EXTENSIONS.contains(&e))
}

/// What is wrong with the start of `text`, if anything.
pub fn problem(text: &str) -> Option<String> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut lines = text.lines();
    let mut first = lines.next().unwrap_or("");
    // A shebang, but not a Rust inner attribute such as `#![forbid(unsafe_code)]`.
    if first.starts_with("#!") && !first.starts_with("#![") {
        first = lines.next().unwrap_or("");
    }
    let line = first.trim();
    let body = ["//", "#", "/*"]
        .iter()
        .find_map(|marker| line.strip_prefix(marker))
        .map(|rest| rest.trim().trim_end_matches("*/").trim());
    match body {
        Some(body) if body == HEADER => None,
        Some(body) if body.starts_with("SPDX-License-Identifier:") => Some(format!(
            "has `{body}`; Syntropy code is `MIT OR Apache-2.0`"
        )),
        _ if text.contains("SPDX-License-Identifier:") => {
            Some("the SPDX line is not a comment on the first line".to_owned())
        }
        _ => Some("has no SPDX line".to_owned()),
    }
}

/// Every covered file under `root` that is not vendored, sorted: the files git tracks when `root`
/// is the top of a git work tree, else every file in the tree. Unreadable directories are
/// errors, so nothing is skipped quietly.
pub fn files(root: &Path) -> io::Result<Vec<PathBuf>> {
    if let Some(tracked) = tracked(root) {
        let mut found: Vec<PathBuf> = tracked
            .iter()
            .filter(|rel| !is_vendored(rel))
            .filter(|rel| !rel.split('/').any(|part| SKIPPED_DIRS.contains(&part)))
            .map(|rel| {
                rel.split('/')
                    .fold(root.to_path_buf(), |path, part| path.join(part))
            })
            // A tracked file deleted in the work tree is not part of the next commit.
            .filter(|path| covered(path) && path.is_file())
            .collect();
        found.sort();
        return Ok(found);
    }
    walk(root)
}

/// The files git tracks, `/`-separated and relative to `root`, when `root` is the top of a git
/// work tree; `None` when it is not, or git cannot say.
fn tracked(root: &Path) -> Option<Vec<String>> {
    let git = |args: &[&str]| {
        Command::new("git")
            .arg("-C")
            .arg(root)
            .args(args)
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| output.stdout)
    };
    let top = String::from_utf8(git(&["rev-parse", "--show-toplevel"])?).ok()?;
    let top = std::fs::canonicalize(top.trim_end_matches(['\n', '\r'])).ok()?;
    if top != std::fs::canonicalize(root).ok()? {
        return None;
    }
    let listed = String::from_utf8(git(&["ls-files", "-z"])?).ok()?;
    Some(
        listed
            .split('\0')
            .filter(|rel| !rel.is_empty())
            .map(str::to_owned)
            .collect(),
    )
}

/// Every covered file in the tree under `root` that is not vendored, sorted.
fn walk(root: &Path) -> io::Result<Vec<PathBuf>> {
    let mut found = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            let kind = entry.file_type()?;
            let name = entry.file_name();
            if kind.is_dir() {
                let skipped = name.to_str().is_some_and(|n| SKIPPED_DIRS.contains(&n));
                if !skipped && !is_vendored(&relative(root, &path)) {
                    pending.push(path);
                }
            } else if covered(&path) && !is_vendored(&relative(root, &path)) {
                found.push(path);
            }
        }
    }
    found.sort();
    Ok(found)
}

/// Checks every covered file under `root` and writes the findings and the verdict to `log`.
/// Returns whether every file passed.
pub fn run(root: &Path, log: &mut dyn Write) -> io::Result<bool> {
    let files = match files(root) {
        Ok(files) => files,
        Err(error) => {
            writeln!(log, "spdx: FAIL: cannot list {}: {error}", root.display())?;
            return Ok(false);
        }
    };
    let mut failed = 0;
    for path in &files {
        let shown = relative(root, path);
        let found = match std::fs::read(path) {
            Ok(bytes) => problem(&String::from_utf8_lossy(&bytes)),
            Err(error) => Some(format!("cannot be read: {error}")),
        };
        if let Some(found) = found {
            failed += 1;
            writeln!(log, "spdx: {shown}: {found}")?;
        }
    }
    if failed == 0 {
        writeln!(
            log,
            "spdx: PASS: {} start with `{HEADER}` (ADR-0018 §1)",
            count(files.len(), "source file")
        )?;
    } else {
        writeln!(
            log,
            "spdx: FAIL: {failed} of {} lack `{HEADER}` on their first line, or on the line \
             after a `#!` line (ADR-0018 §1)",
            count(files.len(), "source file")
        )?;
    }
    log.flush()?;
    Ok(failed == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_source_type_of_adr_0018_is_covered() {
        for path in [
            "crates/core/src/lib.rs",
            "crates/kernels-cuda/build.rs",
            "crates/kernels-cuda/kernels/gemm.cu",
            "crates/kernels-cuda/kernels/tile.cuh",
            "crates/kernels-hip/kernels/gemm.hip",
            "crates/kernels-metal/shaders/gemm.metal",
            "crates/kernels-vulkan/shaders/gemm.slang",
            "crates/kernels-cuda/kernels/common.h",
            "crates/kernels-cpu/native/pack.cpp",
            "crates/kernels-ggml/CMakeLists.txt",
            "crates/kernels-ggml/cmake/flags.cmake",
            "packaging/linux/install.sh",
        ] {
            assert!(covered(Path::new(path)), "{path}");
        }
        for path in [
            "README.md",
            "Cargo.toml",
            "deny.toml",
            "data.json",
            "tool.py",
        ] {
            assert!(!covered(Path::new(path)), "{path}");
        }
    }

    #[test]
    fn accepts_the_header_in_each_comment_form() {
        for text in [
            "// SPDX-License-Identifier: MIT OR Apache-2.0\nfn main() {}\n",
            "# SPDX-License-Identifier: MIT OR Apache-2.0\n",
            "/* SPDX-License-Identifier: MIT OR Apache-2.0 */\n",
            "#!/bin/sh\n# SPDX-License-Identifier: MIT OR Apache-2.0\n",
            "\u{feff}// SPDX-License-Identifier: MIT OR Apache-2.0\n",
            "//   SPDX-License-Identifier: MIT OR Apache-2.0   \r\n",
        ] {
            assert_eq!(problem(text), None, "{text:?}");
        }
    }

    #[test]
    fn rejects_a_missing_late_or_different_header() {
        for (text, expected) in [
            ("fn main() {}\n", "has no SPDX line"),
            ("", "has no SPDX line"),
            (
                "//! Docs.\n// SPDX-License-Identifier: MIT OR Apache-2.0\n",
                "the SPDX line is not a comment on the first line",
            ),
            (
                "#![forbid(unsafe_code)]\n// SPDX-License-Identifier: MIT OR Apache-2.0\n",
                "the SPDX line is not a comment on the first line",
            ),
            (
                "// SPDX-License-Identifier: MIT\n",
                "has `SPDX-License-Identifier: MIT`",
            ),
            (
                "// SPDX-License-Identifier: GPL-2.0-only\n",
                "has `SPDX-License-Identifier: GPL-2.0-only`",
            ),
            (
                "SPDX-License-Identifier: MIT OR Apache-2.0\n",
                "the SPDX line is not a comment on the first line",
            ),
        ] {
            let found = problem(text).unwrap_or_default();
            assert!(found.starts_with(expected), "{text:?}: {found}");
        }
    }
}
