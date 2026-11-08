// SPDX-License-Identifier: MIT OR Apache-2.0

//! The engine conformance suite (TIER-E5-T04; RFC-0021 Testing, "Every engine returns the right
//! bytes"): `io`'s conformance harness (TIER-E5-T33) over the nano fixture files of the M0
//! families, against every engine this host has.
//!
//! The fixtures are generated here from a fixed seed (QUAL-E7-T01), byte-identical on every
//! platform, so the suite uses no fixture store (QUAL-E7-T05's store is not exercised) and reads
//! the same bytes on every run. Each weight file, the F32 safetensors original and the F32 GGUF
//! where the variant has one, goes through every engine: the harness reads random aligned extents
//! into fixed and raw destinations and compares them with a buffered read, then checks writes on
//! a scratch file. Any mismatch fails the test. The config and tokenizer files are left out: they
//! are smaller than the harness's 16 KiB unit, which it refuses as too small.
//!
//! "Every engine this host has" is a fixed set per OS, and every engine in it must build and
//! pass: `PreadPool` on Linux; `MacNoCache` and `MacMapped` on macOS; `PreadPool` and
//! `IocpEngine` on Windows, and `IoRingEngine` where the OS has IoRing. IoRing is the only engine
//! an OS may lack, so it alone is skipped, and only when `ioring_support` says the OS has no ring;
//! `IO_ENGINE_UNAVAILABLE` from any other engine is a failed construction (a thread, a port or a
//! mapping that could not be made), not an absent engine, and fails the test. On a target outside
//! the cfg below, which has none of these engines, the file compiles to nothing.

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
use syntropy_io::{EngineKind, IoEngine};
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

/// Runs the harness over `file` with the engine `make` builds, which must build.
fn run<E, F>(name: &str, make: F, file: &Path, scratch: &Path) -> EngineKind
where
    E: IoEngine + 'static,
    F: Fn() -> Result<E, syntropy_io::IoError> + Send + Clone + 'static,
{
    if let Err(e) = make() {
        panic!("{name}: could not be built on this host, which must have it: {e}");
    }
    let seed = conformance::seed_from_env();
    let report = conformance::check(move || make().unwrap(), file, scratch, &Options::new(seed))
        .unwrap_or_else(|failure| panic!("{name} on {}: {failure}", shown(file)));
    assert!(report.reads > 0, "{name}: {report:?}");
    let len = std::fs::metadata(file).map_or(0, |m| m.len());
    eprintln!(
        "{name} on {} ({len} bytes): passed ({:?}, seed {:#x}, {} reads, {} writes, {}-byte unit)",
        shown(file),
        report.engine,
        report.seed,
        report.reads,
        report.writes,
        report.unit
    );
    report.engine
}

/// Every engine this host has, over `file`: returns the engines that ran.
#[allow(
    clippy::vec_init_then_push,
    reason = "each push is under its own OS's cfg, so no one `vec!` literal fits every target"
)]
fn every_engine(file: &Path, scratch: &Path) -> Vec<EngineKind> {
    let mut ran = Vec::new();
    #[cfg(any(target_os = "linux", windows))]
    ran.push(run(
        "PreadPool",
        || syntropy_io::pool::PreadPool::new(4),
        file,
        scratch,
    ));
    #[cfg(target_os = "macos")]
    {
        ran.push(run(
            "MacNoCache",
            || syntropy_io::pool::MacNoCache::new(4),
            file,
            scratch,
        ));
        ran.push(run(
            "MacMapped",
            || syntropy_io::pool::MacMapped::new(4),
            file,
            scratch,
        ));
    }
    #[cfg(windows)]
    {
        use syntropy_io::windows::{
            IoRingEngine, IoRingOptions, IocpEngine, IocpOptions, ioring_support,
        };
        ran.push(run(
            "IocpEngine",
            || IocpEngine::new(IocpOptions::default()),
            file,
            scratch,
        ));
        match ioring_support() {
            Ok(_) => ran.push(run(
                "IoRingEngine",
                || IoRingEngine::new(IoRingOptions::default()),
                file,
                scratch,
            )),
            Err(e) => eprintln!("IoRingEngine: this Windows has no IoRing ({e}); skipped"),
        }
    }
    ran
}

/// The engines that must run on this OS, whatever the host.
fn required() -> &'static [EngineKind] {
    if cfg!(target_os = "macos") {
        &[EngineKind::MacNoCache, EngineKind::MacMapped]
    } else if cfg!(windows) {
        &[EngineKind::PreadPool, EngineKind::Iocp]
    } else {
        &[EngineKind::PreadPool]
    }
}

#[test]
fn every_engine_reads_the_nano_fixtures_right() {
    let dir = Scratch::new("files");
    let files = weight_files(&dir.0);
    let scratch = dir.0.join("conformance.scratch");
    for file in &files {
        let ran = every_engine(file, &scratch);
        for kind in required() {
            assert!(
                ran.contains(kind),
                "{kind:?} did not run on {}: ran {ran:?}",
                shown(file)
            );
        }
    }
}
