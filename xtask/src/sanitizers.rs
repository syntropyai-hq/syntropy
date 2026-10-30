// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask test asan` and `cargo xtask test tsan`: the `unsafe` crates and the CPU provider
//! built with AddressSanitizer or ThreadSanitizer, and their tests run (RFC-0060 §3.4;
//! QUAL-E3-T10), on the pinned test nightly ([`crate::nightly`]).
//!
//! - **What runs.** The library and integration tests of [`PACKAGES`]: the crates RFC-0001 §4.2
//!   rule 8 lets hold `unsafe`, and `kernels-cpu`, the CPU provider. The other `kernels-*` crates
//!   are empty stubs today; the GPU ones need their vendor toolchains, which the hosted runner
//!   lacks, so each joins when it has code that builds there. Doctests are left out: rustdoc's
//!   sanitizer flags are a separate, unstable set.
//! - **How.** `-Zsanitizer=<kind>` with `-Zbuild-std`, so the standard library is instrumented
//!   too, which ThreadSanitizer needs to see its synchronisation; that takes an explicit
//!   `--target`, the host's. Each kind builds into `asan/` or `tsan/` under the target directory.
//!   A report makes the instrumented test binary fail, and the step with it.
//! - **What each kind leaves out.** [`SKIPS`] lists the tests that cannot pass under any
//!   sanitizer, or need what `hosted-l3` lacks, each with its reason; the suite prints each.
//!   AddressSanitizer runs with `detect_leaks=0`: leak checking
//!   is not in RFC-0060 §3.4, and `net`'s frame tests leak their fixtures on purpose to borrow
//!   them for `'static`. ThreadSanitizer reads [`TSAN_SUPPRESSIONS`], which says why each entry
//!   is there. The caller's `ASAN_OPTIONS` and `TSAN_OPTIONS` are kept, with these added.
//! - **The planted bug** (`asan` only). Last, `tier` is built with `--cfg syntropy_planted` into
//!   `asan/planted/`, and the step passes only when AddressSanitizer reports the read past the
//!   arena's end as a heap buffer overflow.

use std::path::Path;

use crate::nightly::{self, Pin};
use crate::run::Context;

/// The crates the sanitizers build, with their tests.
pub const PACKAGES: &[&str] = &[
    "syntropy-io",
    "syntropy-ipc",
    "syntropy-kernels-cpu",
    "syntropy-net",
    "syntropy-spsc",
    "syntropy-tier",
];

/// Tests left out of both sanitizer runs, by name, with why.
pub const SKIPS: &[(&str, &str)] = &[
    (
        "a_lock_within_memlock_is_taken_and_released",
        "the sanitizer runtimes replace mlock and munlock with no-ops, so the locked size it \
         checks in /proc/self/status never moves",
    ),
    (
        "cmd::tests::other_user::",
        "connects from a second OS account, which `hosted-test` creates and `hosted-l3` does \
         not; `hosted-test` runs it on every OS",
    ),
];

/// ThreadSanitizer's suppressions, relative to the workspace root.
pub const TSAN_SUPPRESSIONS: &str = "ci/tsan-suppressions.txt";

/// A sanitizer the suites build with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sanitizer {
    Address,
    Thread,
}

impl Sanitizer {
    /// The value of `-Zsanitizer=`.
    pub fn flag(self) -> &'static str {
        match self {
            Sanitizer::Address => "-Zsanitizer=address",
            Sanitizer::Thread => "-Zsanitizer=thread",
        }
    }

    /// The suite's name, and its directory under the target directory.
    pub fn name(self) -> &'static str {
        match self {
            Sanitizer::Address => "asan",
            Sanitizer::Thread => "tsan",
        }
    }

    /// The runtime's options variable.
    pub fn options_var(self) -> &'static str {
        match self {
            Sanitizer::Address => "ASAN_OPTIONS",
            Sanitizer::Thread => "TSAN_OPTIONS",
        }
    }

    /// The runtime options the suite adds, for a workspace at `root`.
    pub fn options(self, root: &Path) -> String {
        match self {
            Sanitizer::Address => "detect_leaks=0".to_owned(),
            Sanitizer::Thread => {
                format!("suppressions={}", root.join(TSAN_SUPPRESSIONS).display())
            }
        }
    }
}

/// The caller's runtime options, if any, with `ours` after them, so ours win.
pub fn combined(callers: Option<String>, ours: &str) -> String {
    match callers {
        Some(callers) if !callers.trim().is_empty() => format!("{}:{ours}", callers.trim()),
        _ => ours.to_owned(),
    }
}

/// The arguments after `cargo` that build and test `packages` for `host`.
pub fn args(host: &str, packages: &[&str]) -> Vec<String> {
    let mut args: Vec<String> = ["test", "--locked", "-Zbuild-std", "--target", host]
        .iter()
        .map(|arg| (*arg).to_owned())
        .collect();
    for package in packages {
        args.extend(["--package".to_owned(), (*package).to_owned()]);
    }
    args.extend(
        [
            "--lib",
            "--tests",
            "--no-fail-fast",
            "--",
            "--color",
            "never",
        ]
        .map(str::to_owned),
    );
    for (test, _) in SKIPS {
        args.extend(["--skip".to_owned(), (*test).to_owned()]);
    }
    args
}

fn cargo(
    pin: &Pin,
    root: &Path,
    sanitizer: Sanitizer,
    target: &str,
    cfg: &[&str],
) -> std::process::Command {
    let var = |name: &str| std::env::var(name).ok();
    let mut extra = vec![sanitizer.flag()];
    extra.extend(cfg);
    let (flags, value) = nightly::flags(var, &extra);
    let mut command = nightly::command(pin, "cargo");
    let options = combined(
        std::env::var(sanitizer.options_var()).ok(),
        &sanitizer.options(root),
    );
    command
        .env(flags, value)
        .env(sanitizer.options_var(), options)
        .env("CARGO_TARGET_DIR", nightly::target_dir(root, var, target))
        .env("CARGO_TERM_COLOR", "never")
        .current_dir(root);
    command
}

fn run(context: &Context, sanitizer: Sanitizer) -> Result<(), String> {
    let pin = nightly::read(&context.root)?;
    let host = nightly::host(&pin)?;
    let args = args(&host, PACKAGES);
    eprintln!(
        "{}: {} rustup run {} cargo {}",
        sanitizer.name(),
        sanitizer.flag(),
        pin.channel,
        args.join(" ")
    );
    for (test, why) in SKIPS {
        eprintln!("{}: notice: {test} does not run: {why}", sanitizer.name());
    }
    let mut command = cargo(&pin, &context.root, sanitizer, sanitizer.name(), &[]);
    command.args(&args);
    if nightly::run(command, nightly::deadline())?.success {
        Ok(())
    } else {
        Err(format!(
            "a test failed, or {} reported an error, in a build with {}",
            sanitizer.name(),
            sanitizer.flag()
        ))
    }
}

/// The `toolchain` step: installs the pinned nightly if it is missing.
pub fn toolchain_step(context: &Context) -> Result<(), String> {
    nightly::ensure(&nightly::read(&context.root)?)
}

/// `asan`'s main step.
pub fn asan_step(context: &Context) -> Result<(), String> {
    run(context, Sanitizer::Address)
}

/// `tsan`'s step.
pub fn tsan_step(context: &Context) -> Result<(), String> {
    run(context, Sanitizer::Thread)
}

/// `asan`'s planted step: AddressSanitizer must report the planted out-of-bounds read.
pub fn asan_planted_step(context: &Context) -> Result<(), String> {
    let pin = nightly::read(&context.root)?;
    let host = nightly::host(&pin)?;
    let (package, test) = crate::miri::PLANTED;
    let args: Vec<String> = [
        "test",
        "--locked",
        "-Zbuild-std",
        "--target",
        &host,
        "--package",
        package,
        "--lib",
        "--",
        "--color",
        "never",
        "--exact",
        test,
    ]
    .map(str::to_owned)
    .to_vec();
    eprintln!(
        "asan: planted: --cfg syntropy_planted, rustup run {} cargo {}",
        pin.channel,
        args.join(" ")
    );
    let mut command = cargo(
        &pin,
        &context.root,
        Sanitizer::Address,
        "asan/planted",
        &["--cfg", "syntropy_planted"],
    );
    command.args(&args);
    nightly::reported(
        &nightly::run(command, nightly::deadline())?,
        "AddressSanitizer",
        &["ERROR: AddressSanitizer: heap-buffer-overflow"],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check_deps::policy;

    /// The `unsafe` crates of rule 8, and the CPU provider: every `kernels-*` crate with code.
    #[test]
    fn the_crates_are_the_unsafe_ones_and_the_cpu_provider() {
        let mut want: Vec<String> = policy::UNSAFE_ALLOWED
            .iter()
            .chain(&["kernels-cpu"])
            .map(|name| format!("{}{name}", policy::PREFIX))
            .collect();
        want.sort();
        assert_eq!(PACKAGES, want);
    }

    #[test]
    fn each_kind_builds_the_standard_library_for_the_host() {
        assert_eq!(
            args("x86_64-unknown-linux-gnu", &["syntropy-spsc"]),
            [
                "test",
                "--locked",
                "-Zbuild-std",
                "--target",
                "x86_64-unknown-linux-gnu",
                "--package",
                "syntropy-spsc",
                "--lib",
                "--tests",
                "--no-fail-fast",
                "--",
                "--color",
                "never",
                "--skip",
                "a_lock_within_memlock_is_taken_and_released",
                "--skip",
                "cmd::tests::other_user::",
            ]
        );
        assert_eq!(Sanitizer::Address.flag(), "-Zsanitizer=address");
        assert_eq!(Sanitizer::Thread.flag(), "-Zsanitizer=thread");
        assert_ne!(Sanitizer::Address.name(), Sanitizer::Thread.name());
    }

    #[test]
    fn runtime_options_keep_the_callers_and_add_the_suites() {
        let root = Path::new("/w");
        assert_eq!(Sanitizer::Address.options(root), "detect_leaks=0");
        assert_eq!(
            Sanitizer::Thread.options(root),
            format!("suppressions={}", root.join(TSAN_SUPPRESSIONS).display())
        );
        assert_eq!(combined(None, "a=1"), "a=1");
        assert_eq!(combined(Some(" ".to_owned()), "a=1"), "a=1");
        assert_eq!(
            combined(Some("verbosity=1".to_owned()), "a=1"),
            "verbosity=1:a=1"
        );
    }

    /// Every suppression says why it is there, and suppresses races only.
    #[test]
    fn each_tsan_suppression_has_a_reason() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let text = std::fs::read_to_string(root.join(TSAN_SUPPRESSIONS)).unwrap();
        let mut entries = 0;
        let mut previous = "";
        for line in text.lines() {
            if !line.is_empty() && !line.starts_with('#') {
                entries += 1;
                assert!(line.starts_with("race:"), "{line}");
                assert!(previous.starts_with('#'), "{line} has no reason above it");
            }
            previous = line;
        }
        assert!(entries > 0);
    }
}
