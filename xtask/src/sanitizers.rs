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
    command
        .env(flags, value)
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
    let mut command = cargo(&pin, &context.root, sanitizer, sanitizer.name(), &[]);
    command.args(&args);
    if nightly::run(command)?.success {
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
        &nightly::run(command)?,
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
            ]
        );
        assert_eq!(Sanitizer::Address.flag(), "-Zsanitizer=address");
        assert_eq!(Sanitizer::Thread.flag(), "-Zsanitizer=thread");
        assert_ne!(Sanitizer::Address.name(), Sanitizer::Thread.name());
    }
}
