// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask actionlint`: the pinned actionlint over every workflow in `.github/workflows/`
//! (RFC-0062 §12.4; QUAL-E1-T09).
//!
//! actionlint checks what the security lint (`cargo xtask workflows`) does not: workflow syntax,
//! expression types, the inputs and outputs of actions and jobs, and untrusted input interpolated
//! into a `run:` script.
//!
//! The version is pinned, because a new release can change what a check rejects. A missing or
//! different actionlint fails the step with how to install the pinned one. CI installs it with
//! `.github/actions/install-actionlint`, which checks the release archive against a pinned
//! SHA-256 before unpacking it. The integrations with shellcheck and pyflakes are off
//! (`-shellcheck= -pyflakes=`): actionlint uses whatever versions of those a host has, so with
//! them on the same workflow could pass on one machine and fail on another.
//!
//! actionlint reads a `.github/actionlint.yaml` from the checkout, which a PR could add to
//! silence it. The step passes its own configuration with `-config-file` instead, which takes
//! that file's place: [`CONFIG`], no self-hosted runner labels.

use std::io::{self, Write};
use std::path::Path;
use std::process::Command;

use crate::workflows;

/// The actionlint release `lint` runs. CI installs exactly this version.
pub const VERSION: &str = "1.7.12";

/// The configuration the step passes: no self-hosted runner labels. `lab.yml`'s labels join it
/// when that workflow exists.
pub const CONFIG: &str = "self-hosted-runner:\n  labels: []\n";

/// How to install the pinned actionlint. Only the release binary prints the bare version this
/// step checks; a `go install` build prints `v` before it.
pub fn install_hint() -> String {
    format!(
        "install actionlint {VERSION}: the release binary from \
         https://github.com/rhysd/actionlint/releases/tag/v{VERSION}, checked against the \
         release's checksums file"
    )
}

/// The version in actionlint's `-version` output, whose first line is the bare version.
fn parse_version(stdout: &str) -> Option<String> {
    let version = stdout.lines().next()?.trim();
    let valid = !version.is_empty()
        && version
            .split('.')
            .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()));
    valid.then(|| version.to_owned())
}

/// The installed actionlint's version, or why it cannot be read.
fn installed_version() -> Result<String, String> {
    let output = Command::new("actionlint")
        .arg("-version")
        .output()
        .map_err(|_| "actionlint is not installed".to_owned())?;
    if !output.status.success() {
        return Err("`actionlint -version` failed".to_owned());
    }
    parse_version(&String::from_utf8_lossy(&output.stdout))
        .ok_or_else(|| "`actionlint -version` printed no version".to_owned())
}

/// Runs actionlint on the workflows under `root`, and reports to `log`. Returns whether it
/// passed.
pub fn run(root: &Path, log: &mut dyn Write) -> io::Result<bool> {
    match installed_version() {
        Ok(version) if version == VERSION => {}
        Ok(version) => {
            writeln!(
                log,
                "actionlint: FAIL: lint pins actionlint {VERSION}, and {version} is installed; {}",
                install_hint()
            )?;
            return Ok(false);
        }
        Err(error) => {
            writeln!(log, "actionlint: FAIL: {error}; {}", install_hint())?;
            return Ok(false);
        }
    }
    let files = workflows::workflow_files(root)?;
    if files.is_empty() {
        writeln!(
            log,
            "actionlint: FAIL: no workflow under {}",
            root.join(".github/workflows").display()
        )?;
        return Ok(false);
    }
    writeln!(
        log,
        "actionlint: actionlint {VERSION}, shellcheck and pyflakes off, its own config, on {} \
         workflows",
        files.len()
    )?;
    let config =
        std::env::temp_dir().join(format!("syntropy-actionlint-{}.yaml", std::process::id()));
    std::fs::write(&config, CONFIG)?;
    let output = Command::new("actionlint")
        .args(["-no-color", "-shellcheck=", "-pyflakes=", "-config-file"])
        .arg(&config)
        .args(&files)
        .current_dir(root)
        .output();
    let _ = std::fs::remove_file(&config);
    let output = output?;
    log.write_all(&output.stdout)?;
    log.write_all(&output.stderr)?;
    if output.status.success() {
        writeln!(log, "actionlint: PASS")?;
        Ok(true)
    } else {
        writeln!(
            log,
            "actionlint: FAIL: actionlint reported the problems above ({})",
            output.status
        )?;
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_version_line() {
        assert_eq!(
            parse_version("1.7.12\ninstalled by downloading from release page\n"),
            Some("1.7.12".to_owned())
        );
        assert_eq!(parse_version("v1.7.12\n"), None);
        assert_eq!(parse_version(""), None);
        assert_eq!(parse_version("1..2\n"), None);
    }

    /// The CI action installs the version this module pins.
    #[test]
    fn the_install_action_pins_the_same_version() {
        let action = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../.github/actions/install-actionlint/action.yml");
        let text = std::fs::read_to_string(&action).expect("the install action exists");
        let pinned: Vec<&str> = text
            .lines()
            .filter_map(|line| line.trim().strip_prefix("VERSION: "))
            .collect();
        assert_eq!(pinned, [VERSION], "{}", action.display());
    }
}
