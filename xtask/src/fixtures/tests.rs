// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::{Path, PathBuf};

use super::cache::Cache;
use super::manifest::{self, Manifest};
use super::store::{self, Error, Key, LocalStore, Prefix, Store};
use super::{Action, Args, run};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "syntropy-xtask-fixtures-{name}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn hex(data: &[u8]) -> String {
    blake3::hash(data).to_hex().to_string()
}

/// Publishes `data` into `store` and returns its key.
fn put(dir: &Path, store: &LocalStore, data: &[u8]) -> Key {
    let file = dir.join("in.bin");
    let licence = dir.join("in.licence");
    fs::write(&file, data).unwrap();
    fs::write(&licence, "MIT").unwrap();
    let (hash, bytes) = store::publish(store, Prefix::Fixtures, &file, &licence).unwrap();
    assert_eq!(
        (hash.as_str(), bytes),
        (hex(data).as_str(), data.len() as u64)
    );
    Key::new(Prefix::Fixtures, &hash).unwrap()
}

#[test]
fn a_tampered_object_is_integrity_and_leaves_nothing_cached() {
    let dir = scratch("tamper");
    let store = LocalStore::new(dir.join("store"));
    let key = put(&dir, &store, b"the real bytes");
    fs::write(
        dir.join("store/fixtures").join(key.hex()),
        b"the evil bytes",
    )
    .unwrap();
    let cache = Cache::open(dir.join("cache"), 1 << 20).unwrap();
    let err = cache.fetch(&store, &key, 14).unwrap_err();
    assert!(matches!(err, Error::Integrity { .. }), "{err}");
    assert!(err.to_string().starts_with("Integrity"));
    assert_eq!(cache.size().unwrap(), 0);
    assert_eq!(fs::read_dir(dir.join("cache")).unwrap().count(), 0);
    assert!(matches!(
        store::verify(&store, &key, 14),
        Err(Error::Integrity { .. })
    ));
}

#[test]
fn a_wrong_length_is_integrity_even_when_the_prefix_hashes() {
    let dir = scratch("length");
    let store = LocalStore::new(dir.join("store"));
    let key = put(&dir, &store, b"abcdef");
    let cache = Cache::open(dir.join("cache"), 1 << 20).unwrap();
    assert!(matches!(
        cache.fetch(&store, &key, 5),
        Err(Error::Integrity { .. })
    ));
    assert!(matches!(
        cache.fetch(&store, &key, 7),
        Err(Error::Integrity { .. })
    ));
    assert_eq!(cache.size().unwrap(), 0);
}

#[test]
fn a_cached_file_that_changed_is_dropped_and_fetched_again() {
    let dir = scratch("rot");
    let store = LocalStore::new(dir.join("store"));
    let key = put(&dir, &store, b"payload");
    let cache = Cache::open(dir.join("cache"), 1 << 20).unwrap();
    let path = cache.fetch(&store, &key, 7).unwrap();
    fs::write(&path, b"garbage").unwrap();
    let path = cache.fetch(&store, &key, 7).unwrap();
    assert_eq!(fs::read(path).unwrap(), b"payload");
}

#[test]
fn the_cache_never_exceeds_its_cap_and_evicts_least_recently_used_first() {
    let dir = scratch("cap");
    let store = LocalStore::new(dir.join("store"));
    let cap = 100u64;
    let cache = Cache::open(dir.join("cache"), cap).unwrap();
    let objects: Vec<Vec<u8>> = (0u8..6).map(|i| vec![i; 30 + usize::from(i) * 5]).collect();
    let keys: Vec<Key> = objects.iter().map(|o| put(&dir, &store, o)).collect();
    // A fixed access pattern, with repeats so recency matters.
    for &i in &[0usize, 1, 2, 0, 3, 4, 0, 5, 1, 2, 5, 3] {
        cache
            .fetch(&store, &keys[i], objects[i].len() as u64)
            .unwrap();
        assert!(cache.size().unwrap() <= cap, "after object {i}");
    }
    // The last fetch was object 3, then 5 before it: both are still cached.
    let names: Vec<String> = fs::read_dir(dir.join("cache"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    assert!(names.contains(&keys[3].hex().to_owned()));
    assert!(names.contains(&keys[5].hex().to_owned()));
    assert!(
        !names.contains(&keys[4].hex().to_owned()),
        "object 4 was the least recently used"
    );
}

#[test]
fn an_object_larger_than_the_cap_is_refused_before_download() {
    let dir = scratch("toolarge");
    let store = LocalStore::new(dir.join("store"));
    let key = put(&dir, &store, &[7u8; 64]);
    let cache = Cache::open(dir.join("cache"), 63).unwrap();
    assert_eq!(
        cache.fetch(&store, &key, 64),
        Err(Error::TooLarge { bytes: 64, cap: 63 })
    );
}

#[test]
fn a_missing_object_is_not_found() {
    let dir = scratch("missing");
    let store = LocalStore::new(dir.join("store"));
    let key = Key::new(Prefix::Fixtures, &hex(b"x")).unwrap();
    let cache = Cache::open(dir.join("cache"), 10).unwrap();
    assert!(matches!(
        cache.fetch(&store, &key, 1),
        Err(Error::NotFound(_))
    ));
}

#[test]
fn publishing_without_a_licence_file_is_refused_and_writes_nothing() {
    let dir = scratch("licence");
    let store = LocalStore::new(dir.join("store"));
    let file = dir.join("f.bin");
    fs::write(&file, b"data").unwrap();
    let missing = dir.join("none.licence");
    let empty = dir.join("empty.licence");
    fs::write(&empty, "").unwrap();
    for licence in [&missing, &empty] {
        let err = store::publish(&store, Prefix::Fixtures, &file, licence).unwrap_err();
        assert!(matches!(err, Error::LicenceMissing(_)), "{err}");
    }
    let key = Key::new(Prefix::Fixtures, &hex(b"data")).unwrap();
    assert!(!store.exists(&key));
    assert!(!store.exists(&key.licence()));
}

#[test]
fn publishing_puts_the_licence_beside_the_object() {
    let dir = scratch("beside");
    let store = LocalStore::new(dir.join("store"));
    let key = put(&dir, &store, b"data");
    assert!(store.exists(&key));
    assert!(store.exists(&key.licence()));
    assert_eq!(
        key.licence().to_string(),
        format!("fixtures/{}.licence", hex(b"data"))
    );
}

#[test]
fn keys_that_are_not_a_blake3_are_refused() {
    for bad in [
        "",
        "../etc/passwd",
        &"A".repeat(64),
        &"a".repeat(63),
        &"g".repeat(64),
    ] {
        assert!(
            matches!(Key::new(Prefix::Refs, bad), Err(Error::InvalidKey(_))),
            "{bad}"
        );
    }
    assert!(Key::new(Prefix::Refs, &"a".repeat(64)).is_ok());
}

const MANIFEST: &str = r#"# header
[[fixture]]
name    = "nano/dense"
blake3  = ""                 # content address
bytes   = 0
source  = { hub = "h", revision = "r", file = "f" }
licence = "apache-2.0"
gated   = false
used_by = ["parity:smoke"]

[[fixture]]
bytes   = 0
name    = "other"
blake3  = ""
source  = { hub = "h", revision = "r", file = "g" }
licence = "mit"
gated   = false

[[reference]]
model = "nano/dense"
"#;

#[test]
fn fill_sets_blake3_and_bytes_of_one_fixture_and_nothing_else() {
    let hash = hex(b"x");
    let out = manifest::fill(MANIFEST, "other", &hash, 42).unwrap();
    let parsed = Manifest::parse(&out).unwrap();
    assert_eq!(
        (parsed.fixtures[1].blake3.as_str(), parsed.fixtures[1].bytes),
        (hash.as_str(), 42)
    );
    assert_eq!(
        (parsed.fixtures[0].blake3.as_str(), parsed.fixtures[0].bytes),
        ("", 0)
    );
    // Only those two lines differ.
    let changed = MANIFEST
        .lines()
        .zip(out.lines())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(changed, 2);
    let out = manifest::fill(MANIFEST, "nano/dense", &hash, 7).unwrap();
    assert!(out.contains(&format!("blake3  = \"{hash}\"  # content address")));
    assert!(manifest::fill(MANIFEST, "absent", &hash, 1).is_err());
}

#[test]
fn the_manifest_rejects_duplicates_and_unknown_fields() {
    let twice = format!(
        "{MANIFEST}\n[[fixture]]\nname = \"other\"\nsource = {{}}\nlicence = \"x\"\ngated = false\n"
    );
    assert!(Manifest::parse(&twice).unwrap_err().contains("twice"));
    let extra = MANIFEST.replace(
        "gated   = false\nused_by",
        "gated   = false\nfrob = 1\nused_by",
    );
    assert!(Manifest::parse(&extra).is_err());
}

#[test]
fn import_then_fetch_then_verify_end_to_end() {
    let dir = scratch("e2e");
    let manifest_path = dir.join("fixtures.toml");
    fs::write(&manifest_path, MANIFEST).unwrap();
    let data = dir.join("model.bin");
    let licence = dir.join("LICENSE");
    fs::write(&data, b"weights").unwrap();
    fs::write(&licence, "Apache-2.0").unwrap();
    let base = Args {
        manifest: Some(manifest_path.display().to_string()),
        store: Some(dir.join("store").display().to_string()),
        cache: Some(dir.join("cache").display().to_string()),
        cap: Some("1000".into()),
        ..Args::default()
    };
    let mut out = Vec::new();
    let import = Args {
        name: Some("nano/dense".into()),
        file: Some(data.display().to_string()),
        licence_file: Some(licence.display().to_string()),
        ..base.clone()
    };
    run(Action::Import, &import, &dir, &mut out).unwrap();
    let only = Args {
        name: Some("nano/dense".into()),
        ..base.clone()
    };
    run(Action::Fetch, &only, &dir, &mut out).unwrap();
    run(Action::Verify, &only, &dir, &mut out).unwrap();
    assert!(dir.join("cache").join(hex(b"weights")).is_file());
    // `other` was never imported, so it has no hash and the whole-manifest verify refuses it.
    assert!(run(Action::Verify, &base, &dir, &mut out).is_err());
    // Tamper with the store: both fetch of a fresh cache and verify abort.
    fs::write(dir.join("store/fixtures").join(hex(b"weights")), b"weightz").unwrap();
    assert!(
        run(Action::Verify, &only, &dir, &mut out)
            .unwrap_err()
            .contains("Integrity")
    );
    let fresh = Args {
        cache: Some(dir.join("cache2").display().to_string()),
        ..only
    };
    assert!(
        run(Action::Fetch, &fresh, &dir, &mut out)
            .unwrap_err()
            .contains("Integrity")
    );
}
