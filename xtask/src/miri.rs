// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask test miri`: Miri over the unit tests RFC-0060 §3.4 names (QUAL-E3-T10), on the
//! pinned test nightly ([`crate::nightly`]).
//!
//! - **What runs.** The unit tests of `tier` but its file-backed `tune` module, of `ipc`'s
//!   shared-memory region, which runs on a heap allocation in place of the OS segment under
//!   `cfg(miri)` (`ipc::shm::os`), and of `net`'s frame codecs and transport core. `ipc`'s
//!   command channel, and `net`'s QUIC and TCP transports and its endpoint, interface and crypto
//!   modules, call the OS or run an async runtime over sockets, which Miri cannot do, so those
//!   runs are filtered to [`IPC_FILTERS`] and [`NET_FILTERS`]. Each [`Package`] lists what it
//!   skips and why, and the suite prints it. A unit test that needs the OS elsewhere is marked
//!   `#[cfg_attr(miri, ignore = "<why>")]` where it is written.
//! - **How.** `cargo miri test --lib` for each crate, with Miri's default isolation, into
//!   `miri/` under the target directory. The caller's `MIRIFLAGS` pass through.
//! - **Named tests.** The tests of [`REQUIRED`] must pass, so a run that left a crate out, or
//!   filtered its tests away, cannot pass by running none.
//! - **The planted bug.** Last, the suite builds `tier` with `--cfg syntropy_planted` into
//!   `miri/planted/` and runs the one test that reads a slot past its arena's end. The step
//!   passes only when Miri reports the read as undefined behaviour, so a Miri that stopped seeing
//!   out-of-bounds reads fails the suite.

use std::path::Path;

use crate::nightly::{self, Pin};
use crate::run::Context;

/// A crate Miri runs, and which of its unit tests.
pub struct Package {
    pub name: &'static str,
    /// Test name filters, as libtest takes them; none runs every unit test.
    pub filters: &'static [&'static str],
    /// Test paths left out, as `--skip` takes them, each with why. The suite prints each.
    pub skips: &'static [(&'static str, &'static str)],
}

/// `ipc`'s module that runs under Miri: the shared-memory region. Its command channel (`cmd`)
/// is sockets, process spawning and account checks, all calls into the OS.
pub const IPC_FILTERS: &[&str] = &["shm::"];

/// `net`'s modules that run under Miri: the frame codecs, and the transport core.
pub const NET_FILTERS: &[&str] = &["frame::", "transport::"];

/// The crates and filters of RFC-0060 §3.4.
pub const PACKAGES: &[Package] = &[
    Package {
        name: "syntropy-tier",
        filters: &[],
        skips: &[(
            "tune::",
            "`tune` forbids `unsafe`, and its tests write and read files, which Miri's isolation \
             refuses",
        )],
    },
    Package {
        name: "syntropy-ipc",
        filters: IPC_FILTERS,
        skips: &[(
            "shm::records::",
            "`records` has no `unsafe`, and its proptest reads the working directory, which \
             Miri's isolation refuses",
        )],
    },
    Package {
        name: "syntropy-net",
        filters: NET_FILTERS,
        skips: &[
            (
                "transport::quic::",
                "runs a tokio runtime over real sockets",
            ),
            ("transport::tcp::", "runs a tokio runtime over real sockets"),
            (
                "transport::pacing::",
                "`pacing` has no `unsafe`, and one of its tests feeds the pacer three million \
                 samples, hours under Miri",
            ),
        ],
    },
];

/// Tests that must pass, by crate and test path: two that run through `ipc`'s stand-in segment,
/// one of `net`'s codecs, one of its slot pools across threads, and one of `tier`'s.
pub const REQUIRED: &[(&str, &str)] = &[
    (
        "syntropy-ipc",
        "shm::tests::each_side_takes_its_own_end_of_each_ring_once",
    ),
    (
        "syntropy-ipc",
        "shm::tests::a_garbage_header_or_table_is_refused",
    ),
    (
        "syntropy-net",
        "frame::header::tests::a_short_buffer_or_bad_magic_is_refused",
    ),
    (
        "syntropy-net",
        "transport::buffer::tests::concurrent_takers_never_share_a_slot",
    ),
    (
        "syntropy-tier",
        "os::registry::tests::rounding_and_identity",
    ),
];

/// The planted test, in `tier`.
pub const PLANTED: (&str, &str) = (
    "syntropy-tier",
    "arena::planted::a_slot_read_past_the_arena_end_is_reported",
);

/// What Miri prints for the planted read.
const MARKERS: &[&str] = &[
    "Undefined Behavior: memory access failed",
    "beyond the end of the allocation",
];

/// The arguments of one crate's run, after `cargo`.
pub fn args(package: &Package) -> Vec<String> {
    let mut args: Vec<String> = [
        "miri",
        "test",
        "--locked",
        "--lib",
        "--package",
        package.name,
    ]
    .iter()
    .map(|arg| (*arg).to_owned())
    .collect();
    args.extend(["--", "--color", "never"].map(str::to_owned));
    args.extend(package.filters.iter().map(|filter| (*filter).to_owned()));
    for (skip, _) in package.skips {
        args.extend(["--skip".to_owned(), (*skip).to_owned()]);
    }
    args
}

/// The tests of [`REQUIRED`] for `package` that `stdout` does not show passing.
pub fn missing(package: &str, stdout: &str) -> Vec<&'static str> {
    let passed = crate::loom::passed(stdout);
    REQUIRED
        .iter()
        .filter(|(name, _)| *name == package)
        .map(|(_, test)| *test)
        .filter(|test| !passed.contains(test))
        .collect()
}

fn cargo(pin: &Pin, root: &Path, target: &str, extra_flags: &[&str]) -> std::process::Command {
    let var = |name: &str| std::env::var(name).ok();
    let mut command = nightly::command(pin, "cargo");
    let (flags, value) = nightly::flags(var, extra_flags);
    if !value.is_empty() {
        command.env(flags, value);
    }
    command
        .env("CARGO_TARGET_DIR", nightly::target_dir(root, var, target))
        .env("CARGO_TERM_COLOR", "never")
        .current_dir(root);
    command
}

/// The `toolchain` step: installs the pinned nightly if it is missing.
pub fn toolchain_step(context: &Context) -> Result<(), String> {
    nightly::ensure(&nightly::read(&context.root)?)
}

/// The `miri` step: each crate's tests under Miri, then the check that [`REQUIRED`] passed.
pub fn suite_step(context: &Context) -> Result<(), String> {
    let pin = nightly::read(&context.root)?;
    let deadline = nightly::deadline();
    let mut problems = Vec::new();
    for package in PACKAGES {
        let args = args(package);
        for (skip, why) in package.skips {
            eprintln!(
                "miri: notice: {}'s {skip} tests do not run: {why}",
                package.name
            );
        }
        eprintln!("miri: rustup run {} cargo {}", pin.channel, args.join(" "));
        let mut command = cargo(&pin, &context.root, "miri", &[]);
        command.args(&args);
        let output = nightly::run(command, deadline)?;
        if !output.success {
            problems.push(format!("{}'s tests failed under Miri", package.name));
        }
        let missing = missing(package.name, &output.stdout);
        if !missing.is_empty() {
            problems.push(format!(
                "these tests of {} did not pass under Miri: {}",
                package.name,
                missing.join(", ")
            ));
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

/// The `planted` step: Miri must report the planted out-of-bounds read.
pub fn planted_step(context: &Context) -> Result<(), String> {
    let pin = nightly::read(&context.root)?;
    let (package, test) = PLANTED;
    let args = [
        "miri",
        "test",
        "--locked",
        "--lib",
        "--package",
        package,
        "--",
        "--color",
        "never",
        "--exact",
        test,
    ];
    eprintln!(
        "miri: planted: --cfg syntropy_planted, rustup run {} cargo {}",
        pin.channel,
        args.join(" ")
    );
    let mut command = cargo(
        &pin,
        &context.root,
        "miri/planted",
        &["--cfg", "syntropy_planted"],
    );
    command.args(args);
    nightly::reported(
        &nightly::run(command, nightly::deadline())?,
        "Miri",
        MARKERS,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_crates_are_rfc_0060_section_3_4s() {
        let names: Vec<&str> = PACKAGES.iter().map(|p| p.name).collect();
        assert_eq!(names, ["syntropy-tier", "syntropy-ipc", "syntropy-net"]);
        for (package, _) in REQUIRED.iter().chain([&PLANTED]) {
            assert!(names.contains(package), "{package}");
        }
    }

    #[test]
    fn nets_run_is_filtered_to_its_codecs_and_transport_core() {
        let net = PACKAGES.iter().find(|p| p.name == "syntropy-net").unwrap();
        assert_eq!(
            args(net),
            [
                "miri",
                "test",
                "--locked",
                "--lib",
                "--package",
                "syntropy-net",
                "--",
                "--color",
                "never",
                "frame::",
                "transport::",
                "--skip",
                "transport::quic::",
                "--skip",
                "transport::tcp::",
                "--skip",
                "transport::pacing::",
            ]
        );
    }

    /// Filters and skips narrow a crate's run, but never past a test it must pass.
    #[test]
    fn no_filter_or_skip_drops_a_required_test() {
        for (name, test) in REQUIRED {
            let package = PACKAGES.iter().find(|p| p.name == *name).unwrap();
            assert!(
                (package.filters.is_empty() || package.filters.iter().any(|f| test.starts_with(f)))
                    && !package.skips.iter().any(|(s, _)| test.starts_with(s)),
                "{test} is filtered out"
            );
        }
        let ipc = PACKAGES.iter().find(|p| p.name == "syntropy-ipc").unwrap();
        assert_eq!(
            &args(ipc)[6..],
            [
                "--",
                "--color",
                "never",
                "shm::",
                "--skip",
                "shm::records::"
            ]
        );
        let tier = PACKAGES.iter().find(|p| p.name == "syntropy-tier").unwrap();
        assert_eq!(
            &args(tier)[6..],
            ["--", "--color", "never", "--skip", "tune::"]
        );
    }

    #[test]
    fn a_run_without_the_named_tests_is_caught() {
        let ipc: Vec<&str> = REQUIRED
            .iter()
            .filter(|(p, _)| *p == "syntropy-ipc")
            .map(|(_, t)| *t)
            .collect();
        let all: String = ipc.iter().map(|t| format!("test {t} ... ok\n")).collect();
        assert!(missing("syntropy-ipc", &all).is_empty());
        let ignored = format!("test {} ... ignored\ntest {} ... ok\n", ipc[0], ipc[1]);
        assert_eq!(missing("syntropy-ipc", &ignored), [ipc[0]]);
        assert!(missing("syntropy-spsc", "").is_empty());
    }
}
