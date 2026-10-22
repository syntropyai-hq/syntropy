// SPDX-License-Identifier: MIT OR Apache-2.0

//! The first step of `cargo xtask ci lint`: it prints the cargo and rustc that run the job, and
//! fails unless both are the release `rust-toolchain.toml` pins (ADR-0001 item 1, RFC-0062 §4.2).
//!
//! rustup's cargo reads `rust-toolchain.toml` and runs the pinned release. A cargo installed
//! another way, such as a system package, ignores the file and builds with whatever `rustc` is
//! on the path, so its lint verdict is not CI's. This step names that case instead of letting
//! the checks after it pass or fail on another compiler.

use std::path::Path;
use std::process::Command;

use crate::registry::cargo;
use crate::run::Context;

/// The pinned channel: the `channel` value of `rust-toolchain.toml`'s `[toolchain]` table.
pub fn channel(text: &str) -> Option<String> {
    let mut in_toolchain = false;
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.starts_with('[') {
            in_toolchain = line == "[toolchain]";
        } else if in_toolchain
            && let Some((key, value)) = line.split_once('=')
            && key.trim() == "channel"
        {
            return Some(value.trim().trim_matches('"').to_owned());
        }
    }
    None
}

/// The version in a `cargo -V` or `rustc -V` line, such as `1.99.1` in
/// `rustc 1.99.1 (58cf8f9c3 2026-10-13)`.
pub fn version(line: &str) -> Option<&str> {
    line.split_whitespace().nth(1)
}

/// Whether `version` is the release `channel` names. A channel that is not a release number,
/// such as `stable` or a dated nightly, pins nothing this step can compare, so it matches.
pub fn matches(channel: &str, version: &str) -> bool {
    let is_release = !channel.is_empty()
        && channel
            .split('.')
            .all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()));
    if !is_release {
        return true;
    }
    // `1.99` pins any `1.99.x`; `1.99.1` pins exactly that release.
    version == channel
        || version
            .strip_prefix(channel)
            .is_some_and(|rest| rest.starts_with('.'))
}

/// The first line `program -V` prints, or why it printed none.
fn version_line(program: &std::ffi::OsStr) -> Result<String, String> {
    let shown = program.to_string_lossy();
    let output = Command::new(program)
        .arg("-V")
        .output()
        .map_err(|e| format!("could not start `{shown} -V`: {e}"))?;
    if !output.status.success() {
        return Err(format!("`{shown} -V` exited with {}", output.status));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(stdout.lines().next().unwrap_or("").trim().to_owned())
}

/// `cargo xtask ci lint`'s `toolchain` step.
pub fn step(context: &Context) -> Result<(), String> {
    let file = context.root.join("rust-toolchain.toml");
    let text = std::fs::read_to_string(&file)
        .map_err(|e| format!("{}: {e}", shown(&context.root, &file)))?;
    let pinned = channel(&text).ok_or_else(|| {
        "rust-toolchain.toml has no `channel` in its `[toolchain]` table".to_owned()
    })?;
    // cargo runs the `rustc` that `RUSTC` names, else the one on the path, which rustup's
    // cargo points at the pinned toolchain.
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let mut problems = Vec::new();
    for (tool, program) in [("cargo", cargo()), ("rustc", rustc)] {
        let line = version_line(&program)?;
        eprintln!("toolchain: {line}");
        match version(&line) {
            Some(found) if matches(&pinned, found) => {}
            found => problems.push(format!(
                "{tool} is {}, but rust-toolchain.toml pins {pinned}; only rustup's {tool} \
                 follows the pin, so put rustup's directory (`~/.cargo/bin`) first on the path",
                found.unwrap_or("of an unknown version")
            )),
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

fn shown(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_channel_comes_from_the_toolchain_table() {
        let text =
            "# pinned\n[toolchain]\nchannel = \"1.99.1\" # the release\nprofile = \"minimal\"\n";
        assert_eq!(channel(text).as_deref(), Some("1.99.1"));
        assert_eq!(channel("[other]\nchannel = \"1.0.0\"\n"), None);
        assert_eq!(channel("[toolchain]\nprofile = \"minimal\"\n"), None);
    }

    #[test]
    fn this_workspace_pins_a_release() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let text = std::fs::read_to_string(root.join("rust-toolchain.toml")).unwrap();
        let pinned = channel(&text).expect("a channel");
        assert!(
            pinned
                .split('.')
                .all(|p| p.chars().all(|c| c.is_ascii_digit())),
            "{pinned}"
        );
    }

    #[test]
    fn versions_are_read_from_the_version_line() {
        assert_eq!(
            version("cargo 1.99.1 (5f94df478 2026-08-27)"),
            Some("1.99.1")
        );
        assert_eq!(version("cargo 1.93.1 (Homebrew)"), Some("1.93.1"));
        assert_eq!(version("rustc"), None);
    }

    #[test]
    fn only_the_pinned_release_matches() {
        assert!(matches("1.99.1", "1.99.1"));
        assert!(!matches("1.99.1", "1.93.1"));
        assert!(!matches("1.99.1", "1.99.10"));
        assert!(matches("1.99", "1.99.4"));
        assert!(!matches("1.99", "1.990.0"));
        // A channel that is not a release number pins nothing to compare.
        assert!(matches("stable", "1.93.1"));
        assert!(matches("nightly-2026-10-01", "1.101.0-nightly"));
    }
}
