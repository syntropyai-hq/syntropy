// SPDX-License-Identifier: MIT OR Apache-2.0

//! `kgen`: the generated parts of the CUDA provider's source tree (RFC-0040 §1, §10.2 rules 1,
//! 2 and 7; ADR-0400 decisions 1 and 7).
//!
//! - [`units`]: the translation units, one per (op family, quant-scheme group).
//! - [`support`]: `support.toml`, the provider's ops and the instantiations each needs.
//! - [`inc`]: each unit's explicit instantiation list, `csrc/instantiate/<unit>.inc`.
//! - [`cgen`]: the Rust mirrors of `csrc/params.h`, in `src/params.rs`.
//! - [`key`]: compile flags, contraction off for bit-exact-class units, and the kernel object
//!   cache keys those flags and lists are part of.
//!
//! `cargo xtask kgen` runs the `syntropy-kgen` binary, which calls [`generate`] and writes the
//! files, or with `--check` fails when any is stale. The output depends only on the inputs, so
//! two runs on the same tree write the same bytes.

pub mod cgen;
pub mod inc;
pub mod key;
pub mod support;
pub mod units;

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Every file `kgen` writes, as (path relative to the crate root, contents), in path order.
///
/// # Errors
///
/// A message when `support.toml` or `csrc/params.h` is invalid, or a unit's source is missing.
pub fn generate(root: &Path) -> Result<Vec<(String, String)>, String> {
    let read = |rel: &str| {
        std::fs::read_to_string(root.join(rel))
            .map(|t| t.replace("\r\n", "\n"))
            .map_err(|e| format!("{rel}: {e}"))
    };
    let support = support::parse(&read(support::SUPPORT_PATH)?).map_err(|e| e.to_string())?;
    let header = cgen::parse(&read(cgen::HEADER_PATH)?)?;
    let mut out = Vec::new();
    for unit in units::UNITS {
        if !root.join(unit.source_path()).is_file() {
            return Err(format!(
                "{}: missing; every unit in src/kgen/units.rs has a source",
                unit.source_path()
            ));
        }
        out.push((
            unit.inc_path(),
            inc::render(unit, &support.by_unit[unit.name]),
        ));
    }
    out.push((cgen::MIRROR_PATH.to_owned(), cgen::generate(&header)));
    out.sort();
    Ok(out)
}

/// Files under `csrc/` that look generated or like units but that no unit owns: a `.cu`
/// outside the unit table would build with no instantiation list, and a `.inc` no unit
/// includes is stale.
#[must_use]
pub fn orphans(root: &Path) -> Vec<String> {
    let owned: BTreeSet<String> = units::UNITS
        .iter()
        .flat_map(|u| [u.source_path(), u.inc_path()])
        .collect();
    let mut found = Vec::new();
    walk(&root.join("csrc"), &mut found);
    found
        .into_iter()
        .filter_map(|p| {
            let rel = p
                .strip_prefix(root)
                .ok()?
                .to_string_lossy()
                .replace('\\', "/");
            let ext = p.extension()?.to_str()?;
            (matches!(ext, "cu" | "inc") && !owned.contains(&rel)).then_some(rel)
        })
        .collect()
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.filter_map(Result::ok).map(|e| e.path()).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            walk(&path, out);
        } else {
            out.push(path);
        }
    }
}

/// Runs the generator over the crate at `root`: writes every stale file, or with `check` only
/// reports them. Returns the report, and whether everything was current (or was written).
///
/// # Errors
///
/// A message when the inputs are invalid or a file cannot be written.
pub fn run(root: &Path, check: bool) -> Result<(bool, String), String> {
    let files = generate(root)?;
    let mut report = String::new();
    let mut ok = true;
    for (rel, contents) in &files {
        let path = root.join(rel);
        let current = std::fs::read_to_string(&path)
            .ok()
            .map(|t| t.replace("\r\n", "\n"));
        if current.as_deref() == Some(contents.as_str()) {
            continue;
        }
        if check {
            ok = false;
            let _ = writeln!(report, "kgen: stale: {rel}");
        } else {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir).map_err(|e| format!("{rel}: {e}"))?;
            }
            std::fs::write(&path, contents).map_err(|e| format!("{rel}: {e}"))?;
            let _ = writeln!(report, "kgen: wrote {rel}");
        }
    }
    for orphan in orphans(root) {
        ok = false;
        let _ = writeln!(
            report,
            "kgen: {orphan}: no unit in src/kgen/units.rs owns it; add the unit or remove the file"
        );
    }
    let verdict = match (ok, check) {
        (true, true) => "PASS: every generated file is current",
        (true, false) => "PASS: generated files are current",
        (false, true) => "FAIL: run `cargo xtask kgen` and commit the result",
        (false, false) => "FAIL",
    };
    let _ = writeln!(
        report,
        "kgen: {verdict} ({} files, RFC-0040 §10.2 rule 2)",
        files.len()
    );
    Ok((ok, report))
}

#[cfg(test)]
mod tests;
