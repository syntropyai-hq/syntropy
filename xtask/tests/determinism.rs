// SPDX-License-Identifier: MIT OR Apache-2.0

//! The determinism lints fail `lint` on every banned item (QUAL-E3-T03; RFC-0060 §2.1 rule T1,
//! §2.4 rule T3).
//!
//! The test writes a fixture workspace under cargo's temporary target directory and runs clippy
//! on it as the `clippy` step of `cargo xtask ci lint` does (`--all-targets -- -D warnings`). It
//! adds `--offline`, with a copy of the workspace's `Cargo.lock`, `--keep-going`, so one failing
//! crate does not hide the other, and `--color never`, so the reports read the same in CI. The fixture has two crates with the same code:
//!
//! - `exec`, a covered crate, with `ci/clippy-determinism.toml` copied as its `clippy.toml`, as
//!   every covered crate has it. It calls each item the config bans in `src/banned.rs`, one per
//!   line, and implements a stand-in `TickSource` in `src/tick.rs` that reads the clock under a
//!   scoped `#[allow]`;
//! - `io`, a crate the lints do not cover, with no `clippy.toml`. It also holds stand-ins for the
//!   production sources, `syntropy_io::os::SystemClock`, `OsEntropy` and `SystemTicks`, in their
//!   shape, which `exec` depends on it for, and names itself `syntropy_io` so the shared
//!   `banned.rs` resolves in it too. `exec` builds each source in every way code outside `io` can ([`SOURCES`]), and
//!   each must be refused.
//!
//! Every banned item must be reported on its own line of `exec`, and nothing in `io` or in the
//! `TickSource` implementation. The fixture depends on `rand` and `getrandom` with their generator
//! features on, and on `tokio` with its `time` feature, so every path in the config resolves there: a path that names no item would leave
//! its line unreported and fail the test. The `determinism` step's own checks (that the copies stay
//! identical, and that only listed files, and in `exec` only the `TickSource` implementation, lift
//! a lint) are unit tests in `src/determinism.rs`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Each banned path in `ci/clippy-determinism.toml`, and one line of code that uses it.
const USES: &[(&str, &str)] = &[
    (
        "std::collections::HashMap",
        "pub fn hash_map() -> usize { std::collections::HashMap::<u8, u8>::new().len() }",
    ),
    (
        "std::collections::HashSet",
        "pub fn hash_set() -> usize { std::collections::HashSet::<u8>::new().len() }",
    ),
    (
        "std::hash::RandomState",
        "pub fn random_state() -> std::hash::RandomState { std::hash::RandomState::new() }",
    ),
    (
        "rand::rngs::ThreadRng",
        "pub fn thread_rng_type(_: rand::rngs::ThreadRng) {}",
    ),
    (
        "getrandom::SysRng",
        "pub fn sys_rng_type(_: getrandom::SysRng) {}",
    ),
    (
        "std::time::Instant::now",
        "pub fn instant_now() -> std::time::Instant { std::time::Instant::now() }",
    ),
    (
        "std::time::Instant::elapsed",
        "pub fn instant_elapsed(t: std::time::Instant) -> std::time::Duration { t.elapsed() }",
    ),
    (
        "std::time::SystemTime::now",
        "pub fn system_now() -> std::time::SystemTime { std::time::SystemTime::now() }",
    ),
    (
        "std::time::SystemTime::elapsed",
        "pub fn system_elapsed(t: std::time::SystemTime) -> bool { t.elapsed().is_ok() }",
    ),
    (
        "tokio::time::Instant::now",
        "pub fn tokio_instant_now() -> tokio::time::Instant { tokio::time::Instant::now() }",
    ),
    (
        "tokio::time::Instant::elapsed",
        "pub fn tokio_instant_elapsed(t: tokio::time::Instant) -> std::time::Duration { t.elapsed() }",
    ),
    ("rand::rng", "pub fn rng() { drop(rand::rng()); }"),
    (
        "rand::make_rng",
        "pub fn make_rng() -> rand::rngs::StdRng { rand::make_rng() }",
    ),
    ("rand::random", "pub fn random() -> u64 { rand::random() }"),
    (
        "rand::random_iter",
        "pub fn random_iter() -> Option<u8> { rand::random_iter::<u8>().next() }",
    ),
    (
        "rand::random_range",
        "pub fn random_range() -> u8 { rand::random_range(0..10) }",
    ),
    (
        "rand::random_bool",
        "pub fn random_bool() -> bool { rand::random_bool(0.5) }",
    ),
    (
        "rand::random_ratio",
        "pub fn random_ratio() -> bool { rand::random_ratio(1, 2) }",
    ),
    (
        "rand::fill",
        "pub fn fill() -> [u8; 4] { let mut b = [0u8; 4]; rand::fill(&mut b[..]); b }",
    ),
    (
        "getrandom::fill",
        "pub fn getrandom_fill() -> bool { getrandom::fill(&mut [0u8; 4]).is_ok() }",
    ),
    (
        "getrandom::fill_uninit",
        "pub fn getrandom_fill_uninit() -> bool { getrandom::fill_uninit(&mut [std::mem::MaybeUninit::<u8>::uninit(); 4]).is_ok() }",
    ),
    (
        "getrandom::u32",
        "pub fn getrandom_u32() -> bool { getrandom::u32().is_ok() }",
    ),
    (
        "getrandom::u64",
        "pub fn getrandom_u64() -> bool { getrandom::u64().is_ok() }",
    ),
    (
        "syntropy_io::os::SystemClock",
        "pub fn system_clock() { let _clock = syntropy_io::os::SystemClock::new(); }",
    ),
    (
        "syntropy_io::os::OsEntropy",
        "pub fn os_entropy() { let _entropy = syntropy_io::os::OsEntropy::new(); }",
    ),
    (
        "syntropy_io::os::SystemTicks",
        "pub fn system_ticks() { let _ticks = syntropy_io::os::SystemTicks::new(); }",
    ),
];

/// The other ways code outside `io` could build a production source, each of which must be
/// refused as well: `Default` through the type, `Default` into a binding that names it, and
/// `new()` after importing it. A bare value, `syntropy_io::os::SystemClock`, does not compile,
/// because the type is `non_exhaustive` (#1765).
const SOURCES: &[(&str, &str)] = &[
    (
        "syntropy_io::os::SystemClock",
        "pub fn clock_by_default() { let _c = <syntropy_io::os::SystemClock as Default>::default(); }",
    ),
    (
        "syntropy_io::os::SystemClock",
        "pub fn clock_by_inference() { let _c: syntropy_io::os::SystemClock = Default::default(); }",
    ),
    (
        "syntropy_io::os::SystemClock",
        "pub fn clock_by_import() { use syntropy_io::os::SystemClock; let _c = SystemClock::new(); }",
    ),
    (
        "syntropy_io::os::OsEntropy",
        "pub fn entropy_by_default() { let _e = <syntropy_io::os::OsEntropy as Default>::default(); }",
    ),
    (
        "syntropy_io::os::OsEntropy",
        "pub fn entropy_by_inference() { let _e: syntropy_io::os::OsEntropy = Default::default(); }",
    ),
    (
        "syntropy_io::os::OsEntropy",
        "pub fn entropy_by_import() { use syntropy_io::os::OsEntropy; let _e = OsEntropy::new(); }",
    ),
    (
        "syntropy_io::os::SystemTicks",
        "pub fn ticks_by_default() { let _t = <syntropy_io::os::SystemTicks as Default>::default(); }",
    ),
    (
        "syntropy_io::os::SystemTicks",
        "pub fn ticks_by_inference() { let _t: syntropy_io::os::SystemTicks = Default::default(); }",
    ),
    (
        "syntropy_io::os::SystemTicks",
        "pub fn ticks_by_import() { use syntropy_io::os::SystemTicks; let _t = SystemTicks::new(); }",
    ),
];

/// Stand-ins for `syntropy_io::os`'s production sources, in their shape: `non_exhaustive`, so no
/// other crate writes one as a bare value, `new()`, and `Default`.
const OS: &str = "#[non_exhaustive]\n#[derive(Default)]\npub struct SystemClock;\n\n\
                  impl SystemClock {\n    pub const fn new() -> Self {\n        SystemClock\n    }\n}\n\n\
                  #[non_exhaustive]\n#[derive(Default)]\npub struct OsEntropy;\n\n\
                  impl OsEntropy {\n    pub const fn new() -> Self {\n        OsEntropy\n    }\n}\n\n\
                  #[non_exhaustive]\n#[derive(Default)]\npub struct SystemTicks;\n\n\
                  impl SystemTicks {\n    pub const fn new() -> Self {\n        SystemTicks\n    }\n}\n";

/// A stand-in for `syntropy_core::TickSource` (QUAL-E3-T17) and a production implementation of
/// it, which may read the clock: the one place in `exec` that rule T1 allows to.
const TICK: &str = r#"pub trait TickSource {
    fn ticks(&self) -> u64;
}

pub struct OsTicks {
    start: std::time::Instant,
}

impl OsTicks {
    #[allow(clippy::disallowed_methods, reason = "the TickSource implementation reads the clock")]
    pub fn start() -> OsTicks {
        OsTicks { start: std::time::Instant::now() }
    }
}

impl TickSource for OsTicks {
    #[allow(clippy::disallowed_methods, reason = "the TickSource implementation reads the clock")]
    fn ticks(&self) -> u64 {
        u64::try_from(self.start.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }
}
"#;

const DEPENDENCIES: &str = r#"[dependencies]
getrandom = { version = "0.4.3", features = ["sys_rng"] }
rand = { version = "0.10.3", default-features = false, features = ["std", "std_rng", "sys_rng", "thread_rng"] }
tokio = { version = "1.54.1", default-features = false, features = ["time"] }
"#;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask sits in the workspace root")
        .to_path_buf()
}

fn write(path: &Path, text: &str) {
    fs::create_dir_all(path.parent().expect("a parent directory")).expect("create a directory");
    fs::write(path, text).expect("write a fixture file");
}

/// Writes the fixture workspace and returns its root.
fn fixture() -> PathBuf {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("determinism");
    let repo = repo();
    write(
        &root.join("Cargo.toml"),
        "[workspace]\nresolver = \"3\"\nmembers = [\"crates/exec\", \"crates/io\"]\n",
    );
    fs::copy(repo.join("Cargo.lock"), root.join("Cargo.lock")).expect("copy the lock");
    let banned: String = USES
        .iter()
        .chain(SOURCES)
        .map(|(_, code)| format!("{code}\n"))
        .collect();
    for name in ["exec", "io"] {
        let dir = root.join("crates").join(name);
        // `exec` reaches the production sources through `io`; `io` holds them.
        let (own, lib) = match name {
            "exec" => (
                "syntropy-io = { path = \"../io\" }\n",
                "pub mod banned;\npub mod tick;\n",
            ),
            _ => (
                "",
                "extern crate self as syntropy_io;\npub mod banned;\npub mod os;\npub mod tick;\n",
            ),
        };
        write(
            &dir.join("Cargo.toml"),
            &format!(
                "[package]\nname = \"syntropy-{name}\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\
                 publish = false\n\n{DEPENDENCIES}{own}"
            ),
        );
        write(&dir.join("src/lib.rs"), lib);
        if name == "io" {
            write(&dir.join("src/os.rs"), OS);
        }
        write(&dir.join("src/banned.rs"), &banned);
        write(&dir.join("src/tick.rs"), TICK);
    }
    let config = fs::read_to_string(repo.join("ci/clippy-determinism.toml")).expect("the config");
    write(&root.join("crates/exec/clippy.toml"), &config);
    let _ = fs::remove_file(root.join("crates/io/clippy.toml"));
    root
}

/// The `path = "..."` entries of the config.
fn config_paths() -> Vec<String> {
    let config = fs::read_to_string(repo().join("ci/clippy-determinism.toml")).expect("the config");
    config
        .lines()
        .filter_map(|line| line.split_once("path = \""))
        .filter_map(|(_, rest)| rest.split_once('"'))
        .map(|(path, _)| path.to_owned())
        .collect()
}

#[test]
fn the_fixture_uses_every_banned_path_once() {
    let mut used: Vec<&str> = USES.iter().map(|(path, _)| *path).collect();
    let mut listed = config_paths();
    used.sort_unstable();
    listed.sort_unstable();
    assert_eq!(
        used, listed,
        "USES and ci/clippy-determinism.toml list different paths"
    );
}

#[test]
fn every_banned_item_fails_clippy_in_a_covered_crate_only() {
    let root = fixture();
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    // Plain text: CI sets `CARGO_TERM_COLOR=always` (`.github/actions/setup-rust`), and coloured
    // lines would not match the reports below.
    let output = Command::new(cargo)
        .args([
            "clippy",
            "--offline",
            "--keep-going",
            "--all-targets",
            "--color",
            "never",
        ])
        .args(["--message-format=short", "--", "-D", "warnings"])
        .current_dir(&root)
        .env("CARGO_TARGET_DIR", root.join("target"))
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .expect("cargo runs");
    // Clippy prints paths with the host's separator, `\` on Windows; the reports below are
    // matched with `/`.
    let stderr = String::from_utf8_lossy(&output.stderr).replace('\\', "/");
    assert!(!output.status.success(), "clippy passed:\n{stderr}");

    let reports: Vec<&str> = stderr
        .lines()
        .filter(|line| line.contains("error: use of a disallowed"))
        .collect();
    for (index, (path, code)) in USES.iter().chain(SOURCES).enumerate() {
        let at = format!("crates/exec/src/banned.rs:{}:", index + 1);
        let named = format!("`{path}`");
        assert!(
            reports
                .iter()
                .any(|line| line.starts_with(&at) && line.ends_with(&named)),
            "no report of {path} on line {} (`{code}`):\n{stderr}",
            index + 1
        );
    }
    let elsewhere: Vec<&&str> = reports
        .iter()
        .filter(|line| !line.starts_with("crates/exec/src/banned.rs:"))
        .collect();
    assert!(
        elsewhere.is_empty(),
        "reports outside exec's banned items (io is not covered; the TickSource \
         implementation lifts the lint):\n{elsewhere:#?}"
    );
    let other_errors: Vec<&str> = stderr
        .lines()
        .filter(|line| line.contains("error") && !line.contains("use of a disallowed"))
        .filter(|line| !line.starts_with("error: could not compile `syntropy-exec`"))
        .collect();
    assert!(other_errors.is_empty(), "other errors:\n{stderr}");
}
