// SPDX-License-Identifier: MIT OR Apache-2.0

//! QUAL-E7-T15: one suite over `&dyn Store`, run on the local backend and on the S3 backend
//! against the in-process server ([`super::fake_s3`]); the access rule end to end; and the
//! `LocalStore::put`, `exists` and `import` fixes.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use super::fake_s3::FakeS3;
use super::manifest::Manifest;
use super::store::{self, Access, Error, Key, LocalStore, Prefix, Store};
use super::{Action, Args, open_store, run_with_env};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("syntropy-xtask-s3-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn hex(data: &[u8]) -> String {
    blake3::hash(data).to_hex().to_string()
}

fn key(data: &[u8]) -> Key {
    Key::new(Prefix::Fixtures, &hex(data)).unwrap()
}

fn read_all(store: &dyn Store, k: &Key) -> Result<Vec<u8>, Error> {
    let mut out = Vec::new();
    store.open(k)?.read_to_end(&mut out)?;
    Ok(out)
}

/// The contract every backend keeps, written once (QUAL-E7-T15 criterion 3).
fn conformance(store: &dyn Store, dir: &Path) {
    let data = b"conformance object".as_slice();
    let k = key(data);
    // Absent: `exists` is false and `open` is `NotFound`.
    assert!(!store.exists(&k).unwrap());
    assert!(matches!(
        store.open(&k).map(|_| ()),
        Err(Error::NotFound(_))
    ));
    // A write is readable whole, and `verify` passes.
    store.put(&k, &mut &data[..], Access::Public).unwrap();
    assert!(store.exists(&k).unwrap());
    assert_eq!(read_all(store, &k).unwrap(), data);
    store::verify(store, &k, data.len() as u64).unwrap();
    // Writing the same object again is harmless.
    store.put(&k, &mut &data[..], Access::Public).unwrap();
    assert_eq!(read_all(store, &k).unwrap(), data);
    // Bytes that do not hash to the key are refused, and nothing is stored under it.
    let other = key(b"another object");
    let err = store
        .put(&other, &mut &b"not those bytes"[..], Access::Private)
        .unwrap_err();
    assert!(matches!(err, Error::Integrity { .. }), "{err}");
    assert!(!store.exists(&other).unwrap());
    // A licence file is named by its object, so any bytes go under it.
    store
        .put(&k.licence(), &mut &b"MIT"[..], Access::Public)
        .unwrap();
    assert_eq!(read_all(store, &k.licence()).unwrap(), b"MIT");
    // `verify` of a wrong length is `Integrity`.
    assert!(matches!(
        store::verify(store, &k, 3),
        Err(Error::Integrity { .. })
    ));
    // `publish` puts the licence beside the object.
    let file = dir.join("pub.bin");
    let licence = dir.join("pub.licence");
    fs::write(&file, b"published").unwrap();
    fs::write(&licence, "Apache-2.0").unwrap();
    let (hash, bytes) =
        store::publish(store, Prefix::Fixtures, &file, &licence, Access::Private).unwrap();
    assert_eq!((hash, bytes), (hex(b"published"), 9));
    assert!(store.exists(&key(b"published")).unwrap());
    assert!(store.exists(&key(b"published").licence()).unwrap());
}

#[test]
fn the_local_backend_passes_the_store_suite() {
    let dir = scratch("suite-local");
    conformance(&LocalStore::new(dir.join("store")), &dir);
}

#[test]
fn the_s3_backend_passes_the_store_suite() {
    let dir = scratch("suite-s3");
    let s3 = FakeS3::start();
    let store = open_store(&s3.spec(), &FakeS3::env(true)).unwrap();
    conformance(store.as_ref(), &dir);
    assert!(s3.len() >= 4);
}

/// Criterion 4: a failed request is an error, never "absent".
#[test]
fn exists_reports_a_failed_request_as_an_error() {
    let s3 = FakeS3::start();
    let store = open_store(&s3.spec(), &FakeS3::env(true)).unwrap();
    let k = key(b"anything");
    assert!(!store.exists(&k).unwrap());
    for status in [500, 503] {
        s3.fail_with(status);
        let err = store.exists(&k).unwrap_err();
        assert!(matches!(err, Error::Remote(_)), "{status}: {err}");
    }
    s3.fail_with(403);
    assert!(matches!(store.exists(&k), Err(Error::Denied(_))));
    s3.fail_with(0);
    // Nothing listening: the request never completes.
    let closed = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = closed.local_addr().unwrap().port();
    drop(closed);
    let gone = open_store(&format!("s3+http://127.0.0.1:{port}/b"), &FakeS3::env(true)).unwrap();
    assert!(matches!(gone.exists(&k), Err(Error::Remote(_))));
    // The local backend: a path it cannot stat is an error too.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let dir = scratch("exists-local");
        let store = LocalStore::new(dir.join("store"));
        let locked = dir.join("store/fixtures");
        fs::create_dir_all(&locked).unwrap();
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).unwrap();
        let r = store.exists(&k);
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o755)).unwrap();
        // Root can stat anything; only an unprivileged run sees the error.
        if fs::read_dir(&locked).is_ok() && r.is_ok() {
            eprintln!("running with privileges that ignore the mode: the local leg is skipped");
        } else {
            assert!(r.is_err(), "{r:?}");
        }
    }
}

const SOURCE: &str = "source  = { hub = \"Qwen/Qwen3-0.6B\", revision = \
                      \"c1899de289a04d12100db370d81485cdf75e47ca\", file = \"f\" }";

/// A manifest with one fixture of each access class.
fn manifest() -> String {
    let entry = |name: &str, source: &str, licence: &str, gated: bool| {
        format!(
            "[[fixture]]\nname    = \"{name}\"\nblake3  = \"\"\nbytes   = 0\n{source}\n\
             licence = \"{licence}\"\ngated   = {gated}\ntier    = \"lab\"\n\n"
        )
    };
    let llama = "source  = { hub = \"meta-llama/Llama-3.2-1B\", revision = \
                 \"4e20de362430cd3b72f300e6b0f18e50e7166e08\", file = \"f\" }";
    [
        entry(
            "synthetic",
            "source  = { synthetic = \"nano llama3 0x5eed\" }",
            "mit",
            false,
        ),
        entry("permissive", SOURCE, "apache-2.0", false),
        entry("gated", SOURCE, "apache-2.0", true),
        entry("llama", llama, "llama3.2", false),
        entry("copyleft", SOURCE, "mpl-2.0", false),
    ]
    .concat()
}

/// Criteria 1 and 2, and the lab credential half of criterion 7, against the server: after a
/// lab import, anonymous reads of the synthetic and the permissive fixture and their licence
/// files succeed; anonymous reads of the gated, Llama-family and copyleft fixtures fail; the lab
/// credential reads all of them.
#[test]
fn anonymous_reads_follow_the_access_rule() {
    let dir = scratch("access");
    let s3 = FakeS3::start();
    let manifest_path = dir.join("fixtures.toml");
    fs::write(&manifest_path, manifest()).unwrap();
    let lab_env = FakeS3::env(true);
    let mut out = Vec::new();
    let mut keys = Vec::new();
    for name in ["synthetic", "permissive", "gated", "llama", "copyleft"] {
        let file = dir.join(format!("{name}.bin"));
        let licence = dir.join(format!("{name}.licence"));
        fs::write(&file, format!("{name} bytes")).unwrap();
        fs::write(&licence, format!("{name} licence text")).unwrap();
        let args = Args {
            manifest: Some(manifest_path.display().to_string()),
            store: Some(s3.spec()),
            name: Some(name.into()),
            file: Some(file.display().to_string()),
            licence_file: Some(licence.display().to_string()),
            ..Args::default()
        };
        run_with_env(Action::Import, &args, &dir, &mut out, &lab_env).unwrap();
        keys.push((name, key(format!("{name} bytes").as_bytes())));
    }
    let anon = open_store(&s3.spec(), &FakeS3::env(false)).unwrap();
    let lab = open_store(&s3.spec(), &lab_env).unwrap();
    for (name, k) in &keys {
        let public = matches!(*name, "synthetic" | "permissive");
        assert_eq!(s3.stored(&k.to_string()), Some(public), "{name}");
        assert_eq!(
            s3.stored(&k.licence().to_string()),
            Some(public),
            "{name} licence"
        );
        let read = read_all(anon.as_ref(), k);
        let licence = read_all(anon.as_ref(), &k.licence());
        if public {
            assert_eq!(read.unwrap(), format!("{name} bytes").as_bytes());
            assert_eq!(licence.unwrap(), format!("{name} licence text").as_bytes());
        } else {
            assert!(matches!(read, Err(Error::Denied(_))), "{name}: {read:?}");
            assert!(matches!(licence, Err(Error::Denied(_))), "{name}");
        }
        // The lab credential reads every object.
        assert_eq!(
            read_all(lab.as_ref(), k).unwrap(),
            format!("{name} bytes").as_bytes()
        );
    }
    // Anonymous clients cannot write, and an anonymous `exists` on a private object is an
    // error, not `false`.
    let w = anon.put(&key(b"x"), &mut &b"x"[..], Access::Public);
    assert!(matches!(w, Err(Error::Denied(_))), "{w:?}");
    assert!(matches!(anon.exists(&keys[2].1), Err(Error::Denied(_))));
    // The manifest recorded every import.
    let m = Manifest::parse(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
    assert!(
        m.fixtures
            .iter()
            .all(|f| !f.blake3.is_empty() && f.bytes > 0)
    );
    let log = String::from_utf8(out).unwrap();
    assert!(log.contains("gated:") && log.contains("Private"), "{log}");
}

/// Criterion 5: `LocalStore::put` writes through a unique temporary name and hashes while it
/// copies. Writers of one key never share a temporary file, and a refused write leaves none.
#[test]
fn local_put_uses_unique_temporary_names_and_hashes_while_copying() {
    let dir = scratch("local-put");
    let root = dir.join("store");
    let data = vec![7u8; 1 << 20];
    let k = key(&data);
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| {
                LocalStore::new(root.clone())
                    .put(&k, &mut &data[..], Access::Private)
                    .unwrap();
            });
        }
    });
    let store = LocalStore::new(root.clone());
    assert_eq!(read_all(&store, &k).unwrap(), data);
    let bad = store.put(&key(b"expected"), &mut &b"other"[..], Access::Private);
    assert!(matches!(bad, Err(Error::Integrity { .. })), "{bad:?}");
    let left: Vec<_> = fs::read_dir(root.join("fixtures"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(left, [k.hex().to_owned()], "temporary files left behind");
    let a = store::temp_beside(&root.join("x"));
    let b = store::temp_beside(&root.join("x"));
    assert_ne!(a, b);
}

/// Criterion 6: `import` validates the manifest entry before it publishes anything, and
/// writes the manifest atomically.
#[test]
fn import_validates_before_publishing_and_writes_the_manifest_atomically() {
    let dir = scratch("import");
    let store_dir = dir.join("store");
    let manifest_path = dir.join("fixtures.toml");
    let file = dir.join("w.bin");
    let licence = dir.join("w.licence");
    fs::write(&file, b"weights").unwrap();
    fs::write(&licence, "Apache-2.0").unwrap();
    let args = |name: &str| Args {
        manifest: Some(manifest_path.display().to_string()),
        store: Some(store_dir.display().to_string()),
        name: Some(name.into()),
        file: Some(file.display().to_string()),
        licence_file: Some(licence.display().to_string()),
        ..Args::default()
    };
    let env = |_: &str| None;
    let mut out = Vec::new();
    // No such entry: refused, nothing published.
    fs::write(&manifest_path, manifest()).unwrap();
    let e = run_with_env(Action::Import, &args("absent"), &dir, &mut out, &env).unwrap_err();
    assert!(e.contains("no fixture named"), "{e}");
    assert!(
        !store_dir.exists(),
        "published before the entry was checked"
    );
    // An entry without the `blake3` and `bytes` lines to fill: refused, nothing published.
    let no_lines = manifest().replacen("blake3  = \"\"\nbytes   = 0\n", "", 1);
    fs::write(&manifest_path, &no_lines).unwrap();
    let e = run_with_env(Action::Import, &args("synthetic"), &dir, &mut out, &env).unwrap_err();
    assert!(e.contains("must have"), "{e}");
    assert!(
        !store_dir.exists(),
        "published before the entry was checked"
    );
    assert_eq!(fs::read_to_string(&manifest_path).unwrap(), no_lines);
    // A good entry: published, and the manifest replaced whole with no temporary file left.
    fs::write(&manifest_path, manifest()).unwrap();
    run_with_env(Action::Import, &args("permissive"), &dir, &mut out, &env).unwrap();
    let m = Manifest::parse(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
    let f = m.fixtures.iter().find(|f| f.name == "permissive").unwrap();
    assert_eq!((f.blake3.as_str(), f.bytes), (hex(b"weights").as_str(), 7));
    let parts: Vec<_> = fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".part"))
        .collect();
    assert!(parts.is_empty(), "{parts:?}");
}

#[test]
fn the_store_spec_selects_the_backend() {
    let env = FakeS3::env(false);
    assert!(super::s3::S3Config::parse("/tmp/store", &env).is_none());
    let c = super::s3::S3Config::parse("s3+https://s3.example.org/bucket", &env)
        .unwrap()
        .unwrap();
    assert_eq!(c.endpoint, "https://s3.example.org");
    assert_eq!(c.bucket.as_deref(), Some("bucket"));
    assert!(c.credential.is_none());
    let v = super::s3::S3Config::parse("s3+https://bucket.s3.example.org/", &env)
        .unwrap()
        .unwrap();
    assert_eq!(v.bucket, None);
    // Plain HTTP only for loopback; half a credential is refused.
    assert!(
        super::s3::S3Config::parse("s3+http://s3.example.org/b", &env)
            .unwrap()
            .is_err()
    );
    let half = |k: &str| (k == "SYNTROPY_FIXTURES_S3_ACCESS_KEY_ID").then(|| "AK".to_owned());
    assert!(
        super::s3::S3Config::parse("s3+https://h/b", &half)
            .unwrap()
            .is_err()
    );
}

/// The server checks signatures: a wrong secret is refused, so the suite above shows the
/// client signs correctly.
#[test]
fn a_request_signed_with_the_wrong_secret_is_denied() {
    let s3 = FakeS3::start();
    let wrong = |k: &str| match k {
        "SYNTROPY_FIXTURES_S3_ACCESS_KEY_ID" => Some(super::fake_s3::lab().access_key),
        "SYNTROPY_FIXTURES_S3_SECRET_ACCESS_KEY" => Some("not-the-secret".to_owned()),
        _ => None,
    };
    let store = open_store(&s3.spec(), &wrong).unwrap();
    let r = store.put(&key(b"x"), &mut &b"x"[..], Access::Public);
    assert!(matches!(r, Err(Error::Denied(_))), "{r:?}");
    assert_eq!(s3.len(), 0);
}
