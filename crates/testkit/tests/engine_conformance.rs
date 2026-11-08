// SPDX-License-Identifier: MIT OR Apache-2.0

//! The engine conformance suite (TIER-E5-T04; RFC-0021 Testing, "Every engine returns the right
//! bytes"): `io`'s conformance harness (TIER-E5-T33) over the nano fixture files of the M0
//! families, against every engine this host has.
//!
//! The fixtures are generated here from a fixed seed (QUAL-E7-T01), so the suite needs no fixture
//! store and reads the same bytes on every run. Each weight file, the F32 safetensors original
//! and the F32 GGUF where the variant has one, goes through every engine: the harness reads
//! random aligned extents into fixed and raw destinations and compares them with a buffered read,
//! then checks writes on a scratch file. Any mismatch fails the test.
//!
//! "Every engine this host has": `PreadPool` on Linux; `MacNoCache` and `MacMapped` on macOS;
//! `PreadPool`, `IocpEngine` and `IoRingEngine` on Windows. An engine the OS refuses to build
//! with `IO_ENGINE_UNAVAILABLE`, as a Windows without IoRing does, is skipped and named on
//! standard error; any other construction error fails. At least one engine must run.

#![cfg(any(
    all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ),
    target_os = "macos",
    windows
))]

use std::path::{Path, PathBuf};

use syntropy_graph::families::{llama, qwen_moe};
use syntropy_io::conformance::{self, Options};
use syntropy_io::{EngineKind, IoEngine, IoErrorCode};
use syntropy_testkit::fixtures::{QuantClass, SizeClass, generate};

/// The seed the fixtures are generated from.
const SEED: u64 = 0x7E5_E504;

/// A directory for one run, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "syntropy-engine-conformance-{tag}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The nano fixtures' weight files for the M0 families, generated under `dir`.
fn weight_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for spec in [llama::fixture_spec(), qwen_moe::fixture_spec()] {
        let fixtures = generate(spec, SizeClass::Nano, SEED, &[QuantClass::F32], dir)
            .unwrap_or_else(|e| panic!("{}: {e}", spec.family));
        for f in fixtures {
            files.extend(f.safetensors);
            files.extend(f.gguf.into_iter().map(|(_, path)| path));
        }
    }
    assert!(!files.is_empty(), "the generator wrote no weight files");
    files
}

/// `<fixture>/<file>`, for the log.
fn shown(file: &Path) -> String {
    let name = |p: Option<&Path>| {
        p.and_then(Path::file_name)
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
    };
    format!("{}/{}", name(file.parent()), name(Some(file)))
}

/// Runs the harness over `file` with the engine `make` builds. `None` when the OS reports the
/// engine unavailable.
fn run<E, F>(name: &str, make: F, file: &Path, scratch: &Path) -> Option<EngineKind>
where
    E: IoEngine + 'static,
    F: Fn() -> Result<E, syntropy_io::IoError> + Send + Clone + 'static,
{
    match make() {
        Ok(engine) => drop(engine),
        Err(e) if e.code == IoErrorCode::EngineUnavailable => {
            eprintln!("{name}: not available on this host ({e}); skipped");
            return None;
        }
        Err(e) => panic!("{name}: could not be built: {e}"),
    }
    let seed = conformance::seed_from_env();
    let report = conformance::check(move || make().unwrap(), file, scratch, &Options::new(seed))
        .unwrap_or_else(|failure| panic!("{name} on {}: {failure}", shown(file)));
    assert!(report.reads > 0, "{name}: {report:?}");
    eprintln!(
        "{name} on {}: passed ({:?}, seed {:#x}, {} reads, {} writes, {}-byte unit)",
        shown(file),
        report.engine,
        report.seed,
        report.reads,
        report.writes,
        report.unit
    );
    Some(report.engine)
}

/// Every engine this host has, over every file: returns the engines that ran.
fn every_engine(file: &Path, scratch: &Path) -> Vec<EngineKind> {
    let mut ran = Vec::new();
    #[cfg(any(target_os = "linux", windows))]
    ran.extend(run(
        "PreadPool",
        || syntropy_io::pool::PreadPool::new(4),
        file,
        scratch,
    ));
    #[cfg(target_os = "macos")]
    {
        ran.extend(run(
            "MacNoCache",
            || syntropy_io::pool::MacNoCache::new(4),
            file,
            scratch,
        ));
        ran.extend(run(
            "MacMapped",
            || syntropy_io::pool::MacMapped::new(4),
            file,
            scratch,
        ));
    }
    #[cfg(windows)]
    {
        use syntropy_io::windows::{IoRingEngine, IoRingOptions, IocpEngine, IocpOptions};
        ran.extend(run(
            "IocpEngine",
            || IocpEngine::new(IocpOptions::default()),
            file,
            scratch,
        ));
        ran.extend(run(
            "IoRingEngine",
            || IoRingEngine::new(IoRingOptions::default()),
            file,
            scratch,
        ));
    }
    ran
}

#[test]
fn every_engine_reads_the_nano_fixtures_right() {
    let dir = Scratch::new("files");
    let files = weight_files(&dir.0);
    let scratch = dir.0.join("conformance.scratch");
    for file in &files {
        let ran = every_engine(file, &scratch);
        assert!(
            !ran.is_empty(),
            "no engine ran on {}: this host has none",
            file.display()
        );
    }
}
