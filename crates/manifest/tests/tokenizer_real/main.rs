// SPDX-License-Identifier: MIT OR Apache-2.0

//! `TokenizerSpec` on the M0 families' real repositories (COMP-E17-T14; RFC-0043 §8.1).
//!
//! `tests/data/tokenizer/real-m0.json` names, per M0 checkpoint, its tokenizer side files by
//! their `testdata/fixtures.toml` names, the Hub repository and revision they must be pinned
//! to, and the special-token IDs the spec must hold. Those IDs were read from the files by hand,
//! apart from this crate (the file's `about`).
//!
//! - [`m0_specs_extract_from_the_fixture_store`] reads the files from the fixture cache that
//!   `cargo xtask fixtures fetch` fills, checks each one's BLAKE3 again, extracts the spec and
//!   compares it with the expected IDs. It is `#[ignore]`d because it needs the cache: no CI
//!   job fetches fixtures yet.
//! - The other tests run everywhere. They check that the expectations match the manifest
//!   (names, revisions, files, imported), that every tokenizer in the manifest is covered or
//!   listed as pending, and the failure paths on a synthetic cache: a missing file, a revision
//!   mismatch, corrupted bytes, an unimported fixture and the licence gate.
//!
//! Gated or non-OSI fixtures, the Llama-family files among them, are read only on a runner with
//! the lab credential (`SYNTROPY_FIXTURE_LAB=1`; RFC-0060 §12.3).

mod store;

use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};
use syntropy_manifest::tokenizer::{
    CHAT_TEMPLATE_FILE, GENERATION_CONFIG_JSON, TOKENIZER_CONFIG_JSON, TOKENIZER_JSON,
    TemplateSource, TokenizerSources, TokenizerSpec,
};

use crate::store::{Fixture, LoadError};

/// The fixture cache `cargo xtask fixtures fetch --cache` filled.
const CACHE_VAR: &str = "SYNTROPY_FIXTURE_CACHE";
/// Set to `1` on a runner that holds the lab credential.
const LAB_VAR: &str = "SYNTROPY_FIXTURE_LAB";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Expectations {
    #[allow(dead_code, reason = "documentation for readers of the file")]
    about: String,
    checkpoints: Vec<Checkpoint>,
    pending: Vec<Pending>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    family: String,
    hub: String,
    revision: String,
    files: Files,
    expected: Expected,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Files {
    tokenizer_json: String,
    tokenizer_config: Option<String>,
    generation_config: Option<String>,
    chat_template_file: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Expected {
    bos: Option<u32>,
    eos: Option<u32>,
    end_of_turn: Vec<u32>,
    pad: Option<u32>,
    unk: Option<u32>,
    fim_prefix: Option<u32>,
    fim_middle: Option<u32>,
    fim_suffix: Option<u32>,
    add_bos: bool,
    add_eos: bool,
    chat_template_sha256: Option<String>,
    tokenizer_id: String,
}

/// A tokenizer in the manifest whose expected IDs are not recorded yet, and why.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Pending {
    tokenizer_json: String,
    #[allow(dead_code, reason = "documentation for readers of the file")]
    reason: String,
}

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn manifest() -> Vec<Fixture> {
    let path = repo().join("testdata/fixtures.toml");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    store::parse_manifest(&text).unwrap()
}

fn expectations() -> Expectations {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/tokenizer/real-m0.json");
    let text = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_slice(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// A checkpoint's side files, each its manifest entry.
struct Resolved<'a> {
    tokenizer_json: &'a Fixture,
    tokenizer_config: Option<&'a Fixture>,
    generation_config: Option<&'a Fixture>,
    chat_template_file: Option<&'a Fixture>,
}

impl Resolved<'_> {
    fn all(&self) -> impl Iterator<Item = &Fixture> {
        [
            Some(self.tokenizer_json),
            self.tokenizer_config,
            self.generation_config,
            self.chat_template_file,
        ]
        .into_iter()
        .flatten()
    }
}

/// Why a checkpoint's files do not resolve.
#[derive(Debug, PartialEq, Eq)]
enum ResolveError {
    Load(LoadError),
    /// The fixture comes from another repository or revision than the checkpoint pins.
    Revision {
        name: String,
        detail: String,
    },
    /// The fixture is another file than its role names.
    File {
        name: String,
        detail: String,
    },
}

/// Looks each of `cp`'s side files up in `manifest`, and checks that it is the file its role
/// names, from the repository and revision `cp` pins.
fn resolve<'a>(cp: &Checkpoint, manifest: &'a [Fixture]) -> Result<Resolved<'a>, ResolveError> {
    let one = |name: &str, file: &str| -> Result<&'a Fixture, ResolveError> {
        let f = store::find(manifest, name).map_err(ResolveError::Load)?;
        if f.hub != cp.hub || f.revision != cp.revision {
            return Err(ResolveError::Revision {
                name: name.to_owned(),
                detail: format!(
                    "the manifest has {}@{}; the expectations pin {}@{}",
                    f.hub, f.revision, cp.hub, cp.revision
                ),
            });
        }
        if f.file != file {
            return Err(ResolveError::File {
                name: name.to_owned(),
                detail: format!("the manifest's file is {:?}, not {file:?}", f.file),
            });
        }
        Ok(f)
    };
    let opt = |name: &Option<String>, file: &str| name.as_deref().map(|n| one(n, file)).transpose();
    Ok(Resolved {
        tokenizer_json: one(&cp.files.tokenizer_json, TOKENIZER_JSON)?,
        tokenizer_config: opt(&cp.files.tokenizer_config, TOKENIZER_CONFIG_JSON)?,
        generation_config: opt(&cp.files.generation_config, GENERATION_CONFIG_JSON)?,
        chat_template_file: opt(&cp.files.chat_template_file, CHAT_TEMPLATE_FILE)?,
    })
}

/// The spec of `r`'s files, read from `cache`.
fn load_spec(r: &Resolved<'_>, cache: &Path, lab: bool) -> Result<TokenizerSpec, String> {
    let load = |f: Option<&Fixture>| {
        f.map(|f| store::load(f, cache, lab))
            .transpose()
            .map_err(|e| e.to_string())
    };
    let tokenizer_json = load(Some(r.tokenizer_json))?.expect("tokenizer.json is required");
    let tokenizer_config = load(r.tokenizer_config)?;
    let generation_config = load(r.generation_config)?;
    let chat_template_file = load(r.chat_template_file)?;
    TokenizerSpec::from_sources(TokenizerSources {
        tokenizer_json: &tokenizer_json,
        tokenizer_config: tokenizer_config.as_deref(),
        generation_config: generation_config.as_deref(),
        chat_template_file: chat_template_file.as_deref(),
    })
    .map_err(|e| e.to_string())
}

/// Every way `spec` differs from `want`, one line each.
fn mismatches(want: &Expected, spec: &TokenizerSpec) -> Vec<String> {
    let s = spec.specials();
    let mut out = Vec::new();
    let mut check = |what: &str, got: String, want: String| {
        if got != want {
            out.push(format!("{what}: got {got}, expected {want}"));
        }
    };
    check("bos", format!("{:?}", s.bos), format!("{:?}", want.bos));
    check("eos", format!("{:?}", s.eos), format!("{:?}", want.eos));
    check(
        "end_of_turn",
        format!("{:?}", s.end_of_turn),
        format!("{:?}", want.end_of_turn),
    );
    check("pad", format!("{:?}", s.pad), format!("{:?}", want.pad));
    check("unk", format!("{:?}", s.unk), format!("{:?}", want.unk));
    check(
        "fim_prefix",
        format!("{:?}", s.fim_prefix),
        format!("{:?}", want.fim_prefix),
    );
    check(
        "fim_middle",
        format!("{:?}", s.fim_middle),
        format!("{:?}", want.fim_middle),
    );
    check(
        "fim_suffix",
        format!("{:?}", s.fim_suffix),
        format!("{:?}", want.fim_suffix),
    );
    check(
        "add_bos",
        spec.add_bos().to_string(),
        want.add_bos.to_string(),
    );
    check(
        "add_eos",
        spec.add_eos().to_string(),
        want.add_eos.to_string(),
    );
    let template = spec
        .chat_template()
        .map(|t| hex(&Sha256::digest(t.as_bytes())));
    check(
        "chat_template_sha256",
        format!("{template:?}"),
        format!("{:?}", want.chat_template_sha256),
    );
    let source = if want.chat_template_sha256.is_some() {
        TemplateSource::Shipped
    } else {
        TemplateSource::None
    };
    check(
        "template_source",
        format!("{:?}", spec.template_source()),
        format!("{source:?}"),
    );
    check(
        "tokenizer_id",
        spec.id().to_string(),
        want.tokenizer_id.clone(),
    );
    out
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ---- The real repositories ---------------------------------------------------------------------

/// The criterion: each M0 checkpoint's spec extracts from its recorded files, read from the
/// fixture store, and BOS, EOS, the end-of-turn set, pad and FIM equal the expected IDs.
#[test]
#[ignore = "fixture store: `cargo xtask fixtures fetch` each tokenizer fixture in \
            tests/data/tokenizer/real-m0.json, set SYNTROPY_FIXTURE_CACHE to the cache, and run \
            with --ignored"]
fn m0_specs_extract_from_the_fixture_store() {
    let cache = std::env::var_os(CACHE_VAR)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {CACHE_VAR} to the fixture cache"));
    let lab = std::env::var(LAB_VAR).as_deref() == Ok("1");
    let manifest = manifest();
    let mut checked = 0;
    for cp in expectations().checkpoints {
        let r = resolve(&cp, &manifest).unwrap_or_else(|e| panic!("{}: {e:?}", cp.hub));
        if !lab && r.all().any(Fixture::needs_lab) {
            eprintln!("{}: lab only, skipped ({LAB_VAR} is not 1)", cp.hub);
            continue;
        }
        let spec = load_spec(&r, &cache, lab).unwrap_or_else(|e| panic!("{}: {e}", cp.hub));
        let tokenizer_json = store::load(r.tokenizer_json, &cache, lab).unwrap();
        assert_eq!(
            spec.model(),
            tokenizer_json,
            "{}: model is not the shipped file",
            cp.hub
        );
        let bad = mismatches(&cp.expected, &spec);
        assert!(
            bad.is_empty(),
            "{} ({}):\n{}",
            cp.hub,
            cp.family,
            bad.join("\n")
        );
        eprintln!("{} ({}): ok, TokenizerId {}", cp.hub, cp.family, spec.id());
        checked += 1;
    }
    assert!(checked > 0, "no checkpoint was checked");
}

/// The expectations name files the manifest holds, pinned to the same repository and revision,
/// imported, and each the file its role names. Each M0 family has a checkpoint.
#[test]
fn the_expectations_match_the_fixture_manifest() {
    let manifest = manifest();
    let exp = expectations();
    for cp in &exp.checkpoints {
        let r = resolve(cp, &manifest).unwrap_or_else(|e| panic!("{}: {e:?}", cp.hub));
        for f in r.all() {
            assert!(
                !f.blake3.is_empty() && f.bytes > 0,
                "{}: not imported",
                f.name
            );
        }
        assert_eq!(
            cp.revision.len(),
            40,
            "{}: revision is not a full commit",
            cp.hub
        );
        assert_eq!(
            cp.expected.tokenizer_id.len(),
            64,
            "{}: tokenizer_id",
            cp.hub
        );
    }
    for family in ["llama", "qwen_moe"] {
        assert!(
            exp.checkpoints.iter().any(|c| c.family == family),
            "no checkpoint for the M0 family {family}"
        );
    }
}

/// Every `tokenizer.json` in `manifest` has expected IDs in `exp`, or is listed as pending while
/// it is recorded but not imported; every pending entry is in `manifest`.
fn coverage(manifest: &[Fixture], exp: &Expectations) -> Result<(), String> {
    for f in manifest.iter().filter(|f| f.file == TOKENIZER_JSON) {
        let covered = exp
            .checkpoints
            .iter()
            .any(|c| c.files.tokenizer_json == f.name);
        let pending = exp.pending.iter().any(|p| p.tokenizer_json == f.name);
        match (covered, pending) {
            (true, false) => {}
            (false, true) if f.blake3.is_empty() => {}
            (false, true) => {
                return Err(format!(
                    "{} is imported now: record its expected IDs in real-m0.json",
                    f.name
                ));
            }
            (true, true) => return Err(format!("{} is both covered and pending", f.name)),
            (false, false) => {
                return Err(format!("{} has no expected IDs in real-m0.json", f.name));
            }
        }
    }
    for p in &exp.pending {
        store::find(manifest, &p.tokenizer_json).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Every tokenizer in the manifest is covered or pending. Importing a pending one, or recording
/// a new one, fails this test until its expected IDs are added.
#[test]
fn every_manifest_tokenizer_is_covered_or_pending() {
    coverage(&manifest(), &expectations()).unwrap();

    let manifest = manifest();
    let mut exp = expectations();
    let mut imported = manifest.clone();
    let llama = imported
        .iter_mut()
        .find(|f| f.name == "llama-3.2-1b/tokenizer.json")
        .unwrap();
    llama.blake3 = "0".repeat(64);
    assert!(
        coverage(&imported, &exp)
            .unwrap_err()
            .contains("is imported now")
    );
    let pending = exp.pending.remove(0);
    assert!(
        coverage(&manifest, &exp)
            .unwrap_err()
            .contains("has no expected IDs")
    );
    exp.pending.push(pending);
    exp.pending.push(Pending {
        tokenizer_json: "qwen3-0.6b/tokenizer.json".to_owned(),
        reason: String::new(),
    });
    assert!(
        coverage(&manifest, &exp)
            .unwrap_err()
            .contains("both covered and pending")
    );
    exp.pending.pop();
    exp.pending.push(Pending {
        tokenizer_json: "gone/tokenizer.json".to_owned(),
        reason: String::new(),
    });
    assert!(
        coverage(&manifest, &exp)
            .unwrap_err()
            .contains("no such fixture")
    );
}

// ---- Failure paths, on a synthetic cache -------------------------------------------------------

const SYNTH_HUB: &str = "test/qwen-moe-like";
const SYNTH_REV: &str = "0123456789abcdef0123456789abcdef01234567";

/// A fresh cache directory holding the synthetic `qwen-moe-like` side files under their
/// BLAKE3, and a manifest that names them.
fn synthetic(test: &str) -> (PathBuf, Vec<Fixture>, Checkpoint) {
    let cache = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("tokenizer-real-{test}"));
    let _ = std::fs::remove_dir_all(&cache);
    std::fs::create_dir_all(&cache).unwrap();
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/tokenizer/qwen-moe-like");
    let mut manifest = Vec::new();
    for file in [
        TOKENIZER_JSON,
        TOKENIZER_CONFIG_JSON,
        GENERATION_CONFIG_JSON,
    ] {
        let bytes = std::fs::read(src.join(file)).unwrap();
        let hash = blake3::hash(&bytes).to_hex().to_string();
        std::fs::write(cache.join(&hash), &bytes).unwrap();
        manifest.push(Fixture {
            name: format!("synth/{file}"),
            blake3: hash,
            bytes: bytes.len() as u64,
            hub: SYNTH_HUB.to_owned(),
            revision: SYNTH_REV.to_owned(),
            file: file.to_owned(),
            licence: "apache-2.0".to_owned(),
            gated: false,
        });
    }
    let cp = Checkpoint {
        family: "qwen_moe".to_owned(),
        hub: SYNTH_HUB.to_owned(),
        revision: SYNTH_REV.to_owned(),
        files: Files {
            tokenizer_json: format!("synth/{TOKENIZER_JSON}"),
            tokenizer_config: Some(format!("synth/{TOKENIZER_CONFIG_JSON}")),
            generation_config: Some(format!("synth/{GENERATION_CONFIG_JSON}")),
            chat_template_file: None,
        },
        // The synthetic fixture's IDs, as COMP-E17-T01's `qwen_moe_like_spec_extracts` pins
        // them; the template hash and the ID are filled from the spec by the caller.
        expected: Expected {
            bos: None,
            eos: Some(11),
            end_of_turn: vec![9, 11],
            pad: Some(9),
            unk: None,
            fim_prefix: Some(12),
            fim_middle: Some(13),
            fim_suffix: Some(14),
            add_bos: false,
            add_eos: false,
            chat_template_sha256: None,
            tokenizer_id: String::new(),
        },
    };
    (cache, manifest, cp)
}

/// The whole path, resolve, load, extract and compare, runs on the synthetic cache, and the
/// comparison reports each field that differs.
#[test]
fn the_check_runs_end_to_end_on_a_synthetic_cache() {
    let (cache, manifest, mut cp) = synthetic("end-to-end");
    let r = resolve(&cp, &manifest).unwrap();
    let spec = load_spec(&r, &cache, false).unwrap();
    cp.expected.chat_template_sha256 = spec
        .chat_template()
        .map(|t| hex(&Sha256::digest(t.as_bytes())));
    cp.expected.tokenizer_id = spec.id().to_string();
    assert_eq!(mismatches(&cp.expected, &spec), Vec::<String>::new());

    // Each expected field changed alone is reported, and only that field (the template's hash
    // also changes the template source it implies).
    type Change = fn(&mut Expected);
    let changes: [(&[&str], Change); 12] = [
        (&["bos"], |e| e.bos = Some(1)),
        (&["eos"], |e| e.eos = Some(9)),
        (&["end_of_turn"], |e| e.end_of_turn = vec![11]),
        (&["pad"], |e| e.pad = Some(11)),
        (&["unk"], |e| e.unk = Some(0)),
        (&["fim_prefix"], |e| e.fim_prefix = None),
        (&["fim_middle"], |e| e.fim_middle = None),
        (&["fim_suffix"], |e| e.fim_suffix = Some(12)),
        (&["add_bos"], |e| e.add_bos = true),
        (&["add_eos"], |e| e.add_eos = true),
        (&["chat_template_sha256", "template_source"], |e| {
            e.chat_template_sha256 = None;
        }),
        (&["tokenizer_id"], |e| e.tokenizer_id = "0".repeat(64)),
    ];
    for (want, change) in changes {
        let mut wrong = cp.expected.clone();
        change(&mut wrong);
        let bad = mismatches(&wrong, &spec);
        let fields: Vec<&str> = bad.iter().map(|l| l.split(':').next().unwrap()).collect();
        assert_eq!(fields, want, "{bad:?}");
    }
}

/// A side file the cache does not hold, or the manifest does not name, is an error that names
/// it; nothing is extracted from the files that are there.
#[test]
fn a_missing_tokenizer_file_is_an_error() {
    let (cache, manifest, cp) = synthetic("missing");
    std::fs::remove_file(cache.join(&manifest[0].blake3)).unwrap();
    let r = resolve(&cp, &manifest).unwrap();
    let e = load_spec(&r, &cache, false).unwrap_err();
    assert!(e.contains("synth/tokenizer.json: not in the cache"), "{e}");

    let mut absent = cp.clone();
    absent.files.generation_config = Some("synth/nope.json".to_owned());
    assert_eq!(
        resolve(&absent, &manifest).err(),
        Some(ResolveError::Load(LoadError::Unknown(
            "synth/nope.json".to_owned()
        )))
    );
}

/// A fixture from another revision or repository than the checkpoint pins, or another file
/// than its role names, does not resolve.
#[test]
fn a_revision_mismatch_is_refused() {
    let (_, mut manifest, cp) = synthetic("revision");
    manifest[1].revision = "fedcba9876543210fedcba9876543210fedcba98".to_owned();
    assert!(matches!(
        resolve(&cp, &manifest),
        Err(ResolveError::Revision { ref name, .. }) if name == "synth/tokenizer_config.json"
    ));
    manifest[1].revision = SYNTH_REV.to_owned();
    manifest[2].hub = "test/another".to_owned();
    assert!(matches!(
        resolve(&cp, &manifest),
        Err(ResolveError::Revision { ref name, .. }) if name == "synth/generation_config.json"
    ));
    manifest[2].hub = SYNTH_HUB.to_owned();
    manifest[0].file = "vocab.json".to_owned();
    assert!(matches!(
        resolve(&cp, &manifest),
        Err(ResolveError::File { ref name, .. }) if name == "synth/tokenizer.json"
    ));
}

/// Cached bytes that are not the manifest's are never used, whether the content or the length
/// differs.
#[test]
fn corrupted_cache_bytes_are_refused() {
    let (cache, mut manifest, _) = synthetic("corrupt");
    let f = manifest[2].clone();
    // The same length, one byte changed: only the hash tells.
    let mut bytes = std::fs::read(cache.join(&f.blake3)).unwrap();
    let last = bytes.len() - 2;
    bytes[last] ^= 0x01;
    std::fs::write(cache.join(&f.blake3), &bytes).unwrap();
    assert!(matches!(
        store::load(&f, &cache, false),
        Err(LoadError::Integrity { .. })
    ));
    manifest[1].bytes += 1;
    assert!(matches!(
        store::load(&manifest[1], &cache, false),
        Err(LoadError::Integrity { .. })
    ));
}

/// A tokenizer whose model type the engine cannot load fails extraction as an `Integrity`
/// error naming `tokenizer.json`, even when its bytes are the manifest's.
#[test]
fn an_unknown_tokenizer_type_is_refused() {
    let (cache, mut manifest, cp) = synthetic("unknown-type");
    let text = std::fs::read_to_string(cache.join(&manifest[0].blake3)).unwrap();
    let changed = text.replacen("\"type\": \"BPE\"", "\"type\": \"Mystery\"", 1);
    assert_ne!(text, changed);
    let hash = blake3::hash(changed.as_bytes()).to_hex().to_string();
    std::fs::write(cache.join(&hash), &changed).unwrap();
    manifest[0].blake3 = hash;
    manifest[0].bytes = changed.len() as u64;
    let r = resolve(&cp, &manifest).unwrap();
    let e = load_spec(&r, &cache, false).unwrap_err();
    assert!(
        e.starts_with("tokenizer.json: model type \"Mystery\""),
        "{e}"
    );
}

/// The licence rule: a gated fixture, or one under a licence outside the OSI list, is read only
/// with the lab credential, even when its bytes are in the cache. An unimported fixture is
/// never read.
#[test]
fn the_licence_gate_holds_back_lab_only_files() {
    let (cache, manifest, _) = synthetic("licence");
    let open = &manifest[2];
    assert!(store::load(open, &cache, false).is_ok());

    let mut gated = open.clone();
    gated.gated = true;
    let mut llama = open.clone();
    llama.licence = "llama3.2".to_owned();
    for f in [&gated, &llama] {
        assert_eq!(
            store::load(f, &cache, false),
            Err(LoadError::LabOnly(f.name.clone()))
        );
        assert!(store::load(f, &cache, true).is_ok());
    }
    let mut unimported = llama.clone();
    unimported.blake3 = String::new();
    unimported.bytes = 0;
    assert_eq!(
        store::load(&unimported, &cache, true),
        Err(LoadError::NotImported(unimported.name.clone()))
    );

    // The real Llama-family tokenizer is lab only by the manifest's own fields.
    let manifest = self::manifest();
    let real = store::find(&manifest, "llama-3.2-1b/tokenizer.json").unwrap();
    assert!(real.needs_lab());
    let open_real = store::find(&manifest, "qwen3-0.6b/tokenizer.json").unwrap();
    assert!(!open_real.needs_lab());
}

/// The scanner reads the checked-in manifest's fields, and refuses what it does not know.
#[test]
fn the_manifest_scanner_reads_fixtures_toml() {
    let manifest = manifest();
    let f = store::find(&manifest, "qwen3-0.6b/tokenizer.json").unwrap();
    assert_eq!(f.hub, "Qwen/Qwen3-0.6B");
    assert_eq!(f.revision, "c1899de289a04d12100db370d81485cdf75e47ca");
    assert_eq!(f.file, TOKENIZER_JSON);
    assert_eq!(f.bytes, 11_422_654);
    assert_eq!((f.licence.as_str(), f.gated), ("apache-2.0", false));

    let other = "[[reference]]\nmodel = \"x\"\n[[fixture]]\nname = \"a\"\n";
    assert_eq!(store::parse_manifest(other).unwrap().len(), 1);
    for bad in [
        "name = \"a\"\n",
        "[[fixture]]\nnaem = \"a\"\n",
        "[[fixture]]\nbytes = many\n",
        "[[fixture]]\ngated = yes\n",
        "[[fixture]]\nsource = \"x\"\n",
        "[[fixture]]\nname = \"a\\\"b\"\n",
    ] {
        assert!(store::parse_manifest(bad).is_err(), "{bad}");
    }
}
