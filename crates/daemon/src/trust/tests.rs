// SPDX-License-Identifier: MIT OR Apache-2.0

//! `syntropy trust` against PROD-E5-T04's criteria, on fake trust stores: install then remove
//! leaves the stores as they were; the system store is touched only with `--system`; nothing
//! changes before the fingerprint is confirmed; an un-removed trust is reverted with the rest of
//! the ledger. The real stores are exercised on each OS by `real_store_round_trip` (ignored by
//! default, run on the hosted runners).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use syntropy_api::tls::{CaScope, LocalCa};
use syntropy_net::crypto::SigningKey;

use super::*;
use crate::host_changes::revert::{Outcome, revert_all};

/// A new private directory per call.
fn temp_dir(what: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let dir = std::env::temp_dir().join(format!(
        "syntropy-trust-{what}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).unwrap();
    }
    dir
}

/// A node's CA PEM, from a real local CA, distinct per `seed`.
fn ca_pem(seed: u8) -> String {
    let dir = temp_dir("ca");
    let scope = CaScope::new(["box.local"], &["192.168.1.37".parse().unwrap()], &[])
        .unwrap()
        .0;
    let now = UNIX_EPOCH + Duration::from_secs(1_792_627_200);
    let (ca, _) = LocalCa::open(
        &dir.join("tls"),
        &scope,
        &SigningKey::from_seed(&[seed; 32]),
        now,
    )
    .unwrap();
    ca.certificate_pem().to_owned()
}

/// Trust stores in memory. A store holds certificates; the NSS database and the system anchor
/// files also name each by its label or file, so another certificate there is `Changed`.
#[derive(Debug, Default)]
struct FakeStores {
    held: BTreeMap<(String, String), String>,
    calls: Vec<String>,
    fail_add: bool,
}

fn key(anchor: &Anchor) -> (String, String) {
    let place = format!("{:?} {}", anchor.store, anchor.location.display());
    let name = match anchor.store {
        TrustStore::LinuxUserNss => anchor.label.clone(),
        TrustStore::LinuxSystemCaCertificates | TrustStore::LinuxSystemCaTrust => String::new(),
        _ => anchor.certificate.clone(),
    };
    (place, name)
}

impl StoreTool for FakeStores {
    fn observe(&self, anchor: &Anchor) -> io::Result<Observed> {
        Ok(match self.held.get(&key(anchor)) {
            None => Observed::Absent,
            Some(c) if *c == anchor.certificate => Observed::AsRecorded,
            Some(_) => Observed::Changed,
        })
    }
    fn add(&mut self, anchor: &Anchor, pem_file: &Path) -> io::Result<()> {
        self.calls.push(format!("add {:?}", anchor.store));
        if self.fail_add {
            return Err(io::Error::other("the user cancelled"));
        }
        let pem = fs::read_to_string(pem_file)?;
        assert_eq!(
            store::pem_bodies(&pem),
            std::slice::from_ref(&anchor.certificate)
        );
        self.held.insert(key(anchor), anchor.certificate.clone());
        Ok(())
    }
    fn delete(&mut self, anchor: &Anchor) -> io::Result<()> {
        self.calls.push(format!("delete {:?}", anchor.store));
        self.held.remove(&key(anchor));
        Ok(())
    }
}

struct Source(String);

impl CaSource for Source {
    fn fetch_ca(&mut self, _: &str) -> Result<String, String> {
        Ok(self.0.clone())
    }
}

#[derive(Default)]
struct Answers {
    refuse_fingerprint: bool,
    refuse_sudo: bool,
    asked: Vec<String>,
}

impl Prompt for Answers {
    fn confirm_fingerprint(&mut self, node: &str, ca: &CaCertificate) -> bool {
        self.asked
            .push(format!("fingerprint {node} {}", ca.fingerprint_words()));
        !self.refuse_fingerprint
    }
    fn notice(&mut self, text: &str) {
        self.asked.push(format!("notice {text}"));
    }
    fn confirm_sudo(&mut self, what: &str) -> bool {
        self.asked.push(format!("sudo {what}"));
        !self.refuse_sudo
    }
}

/// A Linux-like plan: the user's NSS database, and a system store `--system` may use.
fn plan(root: &Path) -> Plan {
    Plan {
        pem_dir: root.join("trust"),
        user_stores: vec![Target {
            store: TrustStore::LinuxUserNss,
            location: root.join(".pki/nssdb"),
        }],
        skipped: Vec::new(),
        system_store: Some(Target {
            store: TrustStore::LinuxSystemCaCertificates,
            location: root.join("usr/local/share/ca-certificates"),
        }),
    }
}

/// The stores' contents and the PEM directory's files with their bytes.
type Snapshot = (BTreeMap<(String, String), String>, Vec<(String, Vec<u8>)>);

/// Everything a run could change: the stores and the PEM directory's files with their bytes.
fn snapshot(stores: &FakeStores, plan: &Plan) -> Snapshot {
    let mut files: Vec<(String, Vec<u8>)> = fs::read_dir(&plan.pem_dir)
        .map(|d| {
            d.map(|e| {
                let e = e.unwrap();
                (
                    e.file_name().to_string_lossy().into_owned(),
                    fs::read(e.path()).unwrap(),
                )
            })
            .collect()
        })
        .unwrap_or_default();
    files.sort();
    (stores.held.clone(), files)
}

struct Fixture {
    root: PathBuf,
    plan: Plan,
    ledger: HostChangeLedger,
    stores: FakeStores,
    answers: Answers,
}

impl Fixture {
    fn new() -> Fixture {
        let root = temp_dir("host");
        let state = root.join("state");
        fs::create_dir(&state).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&state, fs::Permissions::from_mode(0o700)).unwrap();
        }
        Fixture {
            plan: plan(&root),
            ledger: HostChangeLedger::open(&state).unwrap(),
            stores: FakeStores::default(),
            answers: Answers::default(),
            root,
        }
    }

    fn trust(&mut self) -> Trust<'_, FakeStores, Answers> {
        Trust {
            plan: &self.plan,
            ledger: &mut self.ledger,
            tool: &mut self.stores,
            prompt: &mut self.answers,
            by_version: "0.0.0-test",
        }
    }

    fn install(&mut self, node: &str, system: bool, pem: &str) -> Result<Installed, TrustError> {
        self.trust()
            .install(node, system, &mut Source(pem.to_owned()), 1)
    }
}

/// Criterion: install then remove leaves the trust store snapshot unchanged. The PEM file is
/// written and the CA added, each recorded; `--remove` takes both away again.
#[test]
fn install_then_remove_leaves_the_stores_as_they_were() {
    let mut f = Fixture::new();
    // Something of the user's in the store and the PEM directory, which must survive.
    f.stores.held.insert(
        ("LinuxUserNss elsewhere".into(), "theirs".into()),
        "QUJD".into(),
    );
    fs::create_dir_all(&f.plan.pem_dir).unwrap();
    fs::write(f.plan.pem_dir.join("other.pem"), b"the user's own\n").unwrap();
    let before = snapshot(&f.stores, &f.plan);

    let pem = ca_pem(1);
    let installed = f.install("Box.Local", false, &pem).unwrap();
    assert_eq!(installed.node, "box.local");
    assert!(installed.complete(), "{installed:?}");
    assert_eq!(installed.stores.len(), 1);
    assert_eq!(installed.stores[0].result, StoreResult::Added);
    assert_eq!(fs::read_to_string(&installed.pem_file).unwrap(), pem);
    assert_eq!(installed.pem_file, f.plan.pem_dir.join("box.local.pem"));
    assert_ne!(snapshot(&f.stores, &f.plan), before);
    let kinds: Vec<&HostChangeKind> = f.ledger.changes().iter().map(|c| &c.kind).collect();
    assert!(matches!(kinds[0], HostChangeKind::FileWritten { .. }));
    assert!(matches!(
        kinds[1],
        HostChangeKind::TrustAnchor { store: TrustStore::LinuxUserNss, node, .. } if node == "box.local"
    ));
    assert!(
        f.ledger
            .changes()
            .iter()
            .all(|c| c.state == ChangeState::Applied)
    );
    assert!(installed.revert_script.is_file());

    let removed = f.trust().remove("box.local").unwrap();
    assert_eq!(removed.reverted.len(), 2, "{removed:?}");
    assert!(
        removed
            .reverted
            .iter()
            .all(|r| r.outcome == Outcome::Reverted),
        "{removed:?}"
    );
    assert_eq!(snapshot(&f.stores, &f.plan), before);
    // A second remove finds nothing open.
    assert!(f.trust().remove("box.local").unwrap().reverted.is_empty());
}

/// Criterion: without `--system` the Linux system store is untouched, and sudo is never asked.
/// With it, the user is asked first, and the system store gets its own anchor file.
#[test]
fn the_system_store_changes_only_with_system() {
    let mut f = Fixture::new();
    let pem = ca_pem(2);
    f.install("box.local", false, &pem).unwrap();
    assert!(
        f.stores.calls.iter().all(|c| !c.contains("LinuxSystem")),
        "{:?}",
        f.stores.calls
    );
    assert!(f.answers.asked.iter().all(|a| !a.starts_with("sudo")));
    assert!(
        f.stores
            .held
            .keys()
            .all(|(place, _)| !place.contains("LinuxSystem"))
    );

    let installed = f.install("box.local", true, &pem).unwrap();
    assert!(f.answers.asked.iter().any(|a| a.starts_with("sudo")));
    let system = installed
        .stores
        .iter()
        .find(|s| s.store == TrustStore::LinuxSystemCaCertificates)
        .unwrap();
    assert_eq!(system.result, StoreResult::Added);
    let file = system.location.file_name().unwrap().to_string_lossy();
    assert!(
        file.starts_with("syntropy-box.local-") && file.ends_with(".crt"),
        "{file}"
    );
    // The user store was already ours.
    assert!(
        installed
            .stores
            .iter()
            .any(|s| s.store == TrustStore::LinuxUserNss && s.result == StoreResult::AlreadyOurs)
    );

    // Removing asks for sudo again, and declining changes nothing.
    f.answers.refuse_sudo = true;
    let held = f.stores.held.clone();
    assert!(matches!(
        f.trust().remove("box.local"),
        Err(TrustError::SudoDeclined)
    ));
    assert_eq!(f.stores.held, held);
    f.answers.refuse_sudo = false;
    f.trust().remove("box.local").unwrap();
    assert!(f.stores.held.is_empty());
}

/// Nothing changes on the host until the fingerprint is confirmed, or when what the node served
/// is refused, or when `--system` cannot be done.
#[test]
fn nothing_changes_before_the_user_confirms() {
    let pem = ca_pem(3);
    type Setup = Box<dyn Fn(&mut Fixture)>;
    let cases: Vec<(&str, Setup, String, bool)> = vec![
        (
            "fingerprint refused",
            Box::new(|f| f.answers.refuse_fingerprint = true),
            pem.clone(),
            false,
        ),
        (
            "not a CA",
            Box::new(|_| {}),
            "<html>404</html>".into(),
            false,
        ),
        (
            "sudo refused",
            Box::new(|f| f.answers.refuse_sudo = true),
            pem.clone(),
            true,
        ),
        (
            "no system store",
            Box::new(|f| f.plan.system_store = None),
            pem.clone(),
            true,
        ),
    ];
    for (what, setup, served, system) in cases {
        let mut f = Fixture::new();
        setup(&mut f);
        let err = f.install("box.local", system, &served).unwrap_err();
        assert!(f.stores.calls.is_empty(), "{what}: {:?}", f.stores.calls);
        assert!(f.ledger.changes().is_empty(), "{what}");
        assert!(!f.plan.pem_dir.exists(), "{what}");
        assert!(!err.to_string().is_empty());
    }
}

/// Trusting a node again with the same CA adds nothing; after the node's CA was replaced, the old
/// one is reverted first, so one CA stays trusted, not two.
#[test]
fn a_replaced_ca_supersedes_the_old_one() {
    let mut f = Fixture::new();
    let old = ca_pem(4);
    let new = ca_pem(5);
    f.install("box.local", false, &old).unwrap();
    let again = f.install("box.local", false, &old).unwrap();
    assert_eq!(again.stores[0].result, StoreResult::AlreadyOurs);
    assert!(again.superseded.is_empty());
    assert_eq!(f.ledger.changes().len(), 2, "no new entries");

    let replaced = f.install("box.local", false, &new).unwrap();
    assert_eq!(replaced.superseded.len(), 2, "{:?}", replaced.superseded);
    assert_eq!(replaced.stores[0].result, StoreResult::Added);
    assert_eq!(f.stores.held.len(), 1);
    let only = f.stores.held.values().next().unwrap();
    assert_eq!(*only, replaced.ca.base64());
    assert_eq!(fs::read_to_string(&replaced.pem_file).unwrap(), new);
}

/// A CA the store already held, from elsewhere, is left there by a remove, since `trust` did not
/// add it; a label that holds another certificate is reported and left.
#[test]
fn what_trust_did_not_add_it_does_not_remove() {
    let mut f = Fixture::new();
    let pem = ca_pem(6);
    let ca = CaCertificate::from_pem(&pem).unwrap();
    let label = format!("Syntropy CA box.local {}", &ca.fingerprint_hex()[..16]);
    let anchor = Anchor {
        store: TrustStore::LinuxUserNss,
        location: f.root.join(".pki/nssdb"),
        label,
        certificate: ca.base64(),
    };
    f.stores.held.insert(key(&anchor), ca.base64());
    let installed = f.install("box.local", false, &pem).unwrap();
    assert_eq!(installed.stores[0].result, StoreResult::AlreadyThere);
    f.trust().remove("box.local").unwrap();
    assert_eq!(f.stores.held.get(&key(&anchor)), Some(&ca.base64()));

    let mut f = Fixture::new();
    let anchor = Anchor {
        location: f.root.join(".pki/nssdb"),
        ..anchor
    };
    f.stores.held.insert(key(&anchor), "T1RIRVI=".into());
    let installed = f.install("box.local", false, &pem).unwrap();
    assert_eq!(installed.stores[0].result, StoreResult::Occupied);
    assert!(!installed.complete());
}

/// A store whose tool fails is reported, and an entry for a change that never happened is
/// closed as abandoned.
#[test]
fn a_failed_add_is_reported_and_abandoned() {
    let mut f = Fixture::new();
    f.stores.fail_add = true;
    let installed = f.install("box.local", false, &ca_pem(7)).unwrap();
    assert!(matches!(installed.stores[0].result, StoreResult::Failed(_)));
    let anchor = f
        .ledger
        .changes()
        .iter()
        .find(|c| matches!(c.kind, HostChangeKind::TrustAnchor { .. }))
        .unwrap();
    assert_eq!(anchor.state, ChangeState::Abandoned);
}

/// Criterion: uninstall reverts an un-removed trust. `revert_all` through [`TrustHost`] takes the
/// CA out of the store and the PEM file away, with the other entries.
#[test]
fn uninstall_reverts_an_unremoved_trust() {
    let mut f = Fixture::new();
    let before = snapshot(&f.stores, &f.plan);
    f.install("box.local", false, &ca_pem(8)).unwrap();
    f.install("nas.local", false, &ca_pem(9)).unwrap();
    let report = revert_all(&mut f.ledger, &mut TrustHost::new(&mut f.stores)).unwrap();
    assert_eq!(report.len(), 4);
    assert!(
        report.iter().all(|r| r.outcome == Outcome::Reverted),
        "{report:?}"
    );
    assert_eq!(snapshot(&f.stores, &f.plan), before);
}

/// `--remove` reverts one node's entries and leaves another node's.
#[test]
fn remove_touches_only_the_named_node() {
    let mut f = Fixture::new();
    let a = f.install("box.local", false, &ca_pem(10)).unwrap();
    let b = f.install("nas.local", false, &ca_pem(11)).unwrap();
    f.trust().remove("box.local").unwrap();
    assert!(!a.pem_file.exists());
    assert!(b.pem_file.exists());
    assert_eq!(f.stores.held.values().collect::<Vec<_>>(), [&b.ca.base64()]);
}

#[test]
fn the_variables_are_printed_for_each_shell() {
    let path = Path::new("/home/o'neil/.local/share/syntropy/trust/box.local.pem");
    let sh = env_lines(path, Shell::Posix);
    assert_eq!(
        sh,
        "export SSL_CERT_FILE='/home/o'\\''neil/.local/share/syntropy/trust/box.local.pem'\n\
         export REQUESTS_CA_BUNDLE='/home/o'\\''neil/.local/share/syntropy/trust/box.local.pem'\n\
         export NODE_EXTRA_CA_CERTS='/home/o'\\''neil/.local/share/syntropy/trust/box.local.pem'\n"
    );
    let ps = env_lines(
        Path::new(r"C:\Users\o'neil\trust\box.local.pem"),
        Shell::PowerShell,
    );
    assert!(ps.starts_with("$env:SSL_CERT_FILE = 'C:\\Users\\o''neil\\trust\\box.local.pem'\n"));
    assert_eq!(ps.lines().count(), 3);
}

/// `syntropy trust` leaves the OS's own confirmation to the user and says so first: on macOS and
/// Windows a notice precedes each add and each removal; the stores that ask nothing get none.
#[test]
fn the_user_is_told_before_the_os_asks() {
    let mut f = Fixture::new();
    f.plan.user_stores = vec![
        Target {
            store: TrustStore::MacosLoginKeychain,
            location: f.root.join("login.keychain-db"),
        },
        Target {
            store: TrustStore::WindowsCurrentUserRoot,
            location: PathBuf::from(WINDOWS_ROOT_STORE),
        },
        Target {
            store: TrustStore::LinuxUserNss,
            location: f.root.join(".pki/nssdb"),
        },
    ];
    f.install("box.local", false, &ca_pem(15)).unwrap();
    f.trust().remove("box.local").unwrap();
    let notices: Vec<&String> = f
        .answers
        .asked
        .iter()
        .filter(|a| a.starts_with("notice"))
        .collect();
    assert_eq!(notices.len(), 4, "{notices:?}");
    assert!(
        notices[0].contains("macOS will ask you to confirm"),
        "{notices:?}"
    );
    assert!(
        notices[1].contains("Windows will ask you to confirm installing"),
        "{notices:?}"
    );
    assert!(
        notices
            .iter()
            .any(|n| n.contains("Windows will ask you to confirm removing"))
    );
    assert_eq!(os_confirmation(TrustStore::LinuxUserNss, true), None);
    assert_eq!(os_confirmation(TrustStore::LinuxSystemCaTrust, false), None);
}

#[test]
fn node_names_become_safe_file_names() {
    assert_eq!(node_slug("Box.Local"), "box.local");
    assert_eq!(node_slug("box.local:8443"), "box.local_3a8443");
    assert_eq!(node_slug("[fd00::1]:8443"), "_5bfd00_3a_3a1_5d_3a8443");
    assert_eq!(node_slug("../etc"), "_2e._2fetc");
    assert_eq!(node_slug(".."), "_2e.");
    assert_eq!(node_slug(""), "_");
    // One to one: names that the old mapping folded together stay apart.
    let nodes = [
        "box:1", "box_1", "box 1", "box/1", "box\\1", "box_3a1", "bóx", "b_c3_b3x",
    ];
    let mut slugs: Vec<String> = nodes.iter().map(|n| node_slug(n)).collect();
    assert_eq!(slugs[0], "box_3a1");
    assert_eq!(slugs[1], "box_5f1");
    slugs.sort();
    slugs.dedup();
    assert_eq!(slugs.len(), nodes.len(), "{slugs:?}");
}

/// Windows reserves device names whatever their extension, so a node named for one gets its
/// first letter escaped, and the mapping stays one to one: `con` and the node whose name was
/// already `_63on` still get different files.
#[test]
fn windows_device_names_are_escaped() {
    let cases = [
        ("con", "_63on"),
        ("CON.local", "_63on.local"),
        ("nul", "_6eul"),
        ("aux.home.arpa", "_61ux.home.arpa"),
        ("prn", "_70rn"),
        ("com1", "_63om1"),
        ("com9.lan", "_63om9.lan"),
        ("lpt1", "_6cpt1"),
        ("lpt0", "_6cpt0"),
        // Not device names: kept as they are.
        ("console", "console"),
        ("com10", "com10"),
        ("comx", "comx"),
        ("box.con", "box.con"),
        ("nul-box", "nul-box"),
    ];
    for (node, slug) in cases {
        assert_eq!(node_slug(node), slug, "{node}");
        let file = Plan {
            pem_dir: PathBuf::from("/p"),
            user_stores: Vec::new(),
            skipped: Vec::new(),
            system_store: None,
        }
        .pem_file(node);
        let stem = file
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .split('.')
            .next()
            .unwrap()
            .to_owned();
        assert!(!windows_device_name(&stem), "{node}: {file:?}");
    }
    let nodes = ["con", "_63on", "c_6fn", "con.local", "_63on.local"];
    let mut slugs: Vec<String> = nodes.iter().map(|n| node_slug(n)).collect();
    slugs.sort();
    slugs.dedup();
    assert_eq!(slugs.len(), nodes.len(), "{slugs:?}");
}

/// Criterion: uninstall reverts an un-removed trust after the binary is gone. The generated sh
/// script, run for real against a stand-in `certutil` that keeps one file per nickname, removes
/// the CA from the NSS database and the PEM file, and leaves another nickname alone.
#[cfg(unix)]
#[test]
fn the_revert_script_removes_an_unremoved_trust() {
    use std::os::unix::fs::PermissionsExt;
    let mut f = Fixture::new();
    let installed = f.install("box.local", false, &ca_pem(12)).unwrap();
    let anchor = Anchor::of(&f.ledger.changes()[1].kind).unwrap();
    // The stand-in database: `<db>/<nickname>.pem`.
    let db = anchor.location.clone();
    fs::create_dir_all(&db).unwrap();
    fs::write(db.join(format!("{}.pem", anchor.label)), installed.ca.pem()).unwrap();
    fs::write(db.join("someone else.pem"), installed.ca.pem()).unwrap();
    let bin = f.root.join("bin");
    fs::create_dir(&bin).unwrap();
    let certutil = bin.join("certutil");
    fs::write(
        &certutil,
        "#!/bin/sh\n\
         d=''; n=''; op=''\n\
         while [ $# -gt 0 ]; do case \"$1\" in -d) d=${2#sql:}; shift;; -n) n=$2; shift;; -L) op=L;; -D) op=D;; esac; shift; done\n\
         f=\"$d/$n.pem\"\n\
         [ -f \"$f\" ] || exit 255\n\
         if [ \"$op\" = D ]; then rm -f \"$f\"; else cat \"$f\"; fi\n",
    )
    .unwrap();
    fs::set_permissions(&certutil, fs::Permissions::from_mode(0o755)).unwrap();

    let script = crate::host_changes::script::write_revert_script(&f.ledger, "v").unwrap();
    let out = std::process::Command::new("sh")
        .arg(&script)
        .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{stdout}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("removed the CA"), "{stdout}");
    assert!(
        !db.join(format!("{}.pem", anchor.label)).exists(),
        "{stdout}"
    );
    assert!(db.join("someone else.pem").exists());
    assert!(!installed.pem_file.exists(), "{stdout}");
    // A second run skips both.
    let again = std::process::Command::new("sh")
        .arg(&script)
        .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&again.stdout).contains("already reverted"));
}

/// The real stores on this OS: install a CA into the user's store and remove it again, and
/// check that the store's snapshot is unchanged and the system store untouched. Ignored by
/// default because it changes the running user's trust store; the hosted runners run it with
/// `--ignored` (PROD-E5-T04's per-OS criterion).
#[test]
#[ignore = "changes this user's real trust store; run on a throwaway CI runner"]
fn real_store_round_trip() {
    let root = temp_dir("real");
    let state = root.join("state");
    fs::create_dir(&state).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&state, fs::Permissions::from_mode(0o700)).unwrap();
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap();
    let plan = Plan::for_this_host(&home, root.join("trust"));
    println!("plan: {plan:?}");
    let mut tool = os::OsStoreTool;
    let pem = ca_pem(13);
    let ca = CaCertificate::from_pem(&pem).unwrap();
    let snap = |tool: &os::OsStoreTool| -> Vec<(String, Observed)> {
        // The user stores, by their entry for this CA; the system store by its listing below.
        plan.user_stores
            .iter()
            .map(|t| {
                let anchor = Anchor {
                    store: t.store,
                    location: t.location.clone(),
                    label: format!("Syntropy CA box.local {}", &ca.fingerprint_hex()[..16]),
                    certificate: ca.base64(),
                };
                (
                    format!("{:?}", t.store),
                    tool.observe(&anchor).unwrap_or(Observed::Changed),
                )
            })
            .collect()
    };
    let system_before = system_store_listing(&plan);
    let listing_before = user_store_listing(&plan);
    println!("store listing before:\n{listing_before}");
    let before = snap(&tool);
    println!("before: {before:?}");
    let mut ledger = HostChangeLedger::open(&state).unwrap();
    let mut answers = Answers::default();
    let installed = Trust {
        plan: &plan,
        ledger: &mut ledger,
        tool: &mut tool,
        prompt: &mut answers,
        by_version: "evidence",
    }
    .install("box.local", false, &mut Source(pem.clone()), 1)
    .unwrap();
    println!("installed: {:?}", installed.stores);
    println!("skipped: {:?}", installed.skipped);
    println!("{}", env_lines(&installed.pem_file, Shell::for_this_os()));
    assert!(installed.complete(), "{:?}", installed.stores);
    let during = snap(&tool);
    println!("during: {during:?}");
    for (store, observed) in &during {
        assert_eq!(*observed, Observed::AsRecorded, "{store}");
    }
    assert_eq!(
        system_store_listing(&plan),
        system_before,
        "the system store changed without --system"
    );
    let removed = Trust {
        plan: &plan,
        ledger: &mut ledger,
        tool: &mut tool,
        prompt: &mut answers,
        by_version: "evidence",
    }
    .remove("box.local")
    .unwrap();
    println!("removed: {:?}", removed.reverted);
    assert!(
        removed
            .reverted
            .iter()
            .all(|r| r.outcome == Outcome::Reverted)
    );
    let after = snap(&tool);
    println!("after: {after:?}");
    assert_eq!(after, before);
    let listing_after = user_store_listing(&plan);
    assert_eq!(
        listing_after, listing_before,
        "the user store's listing changed"
    );
    assert_eq!(system_store_listing(&plan), system_before);
    println!("PASS: the trust store snapshot is unchanged after install then remove");
}

/// Every entry of the user's stores, as the OS's tools list them: the NSS database's nicknames
/// and trust flags, the login keychain's certificate hashes and the user trust settings, or the
/// Windows Root store's thumbprints. The snapshot the per-OS criterion compares.
fn user_store_listing(plan: &Plan) -> String {
    let run = |program: &str, args: &[&str]| -> String {
        match std::process::Command::new(program).args(args).output() {
            Ok(o) => format!(
                "{}{}",
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr)
            ),
            Err(e) => format!("{program}: {e}"),
        }
    };
    let mut listing = String::new();
    for target in &plan.user_stores {
        let location = target.location.display().to_string();
        let mut lines: Vec<String> = match target.store {
            TrustStore::LinuxUserNss => run("certutil", &["-d", &format!("sql:{location}"), "-L"])
                .lines()
                .map(str::to_owned)
                .collect(),
            TrustStore::MacosLoginKeychain => {
                run("security", &["find-certificate", "-a", "-Z", &location])
                    .lines()
                    .filter(|l| l.starts_with("SHA-1 hash:"))
                    .map(str::to_owned)
                    .chain(
                        run("security", &["dump-trust-settings"])
                            .lines()
                            .map(str::to_owned),
                    )
                    .collect()
            }
            TrustStore::WindowsCurrentUserRoot => run(
                "powershell.exe",
                &[
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "Get-ChildItem Cert:\\CurrentUser\\Root | ForEach-Object { $_.Thumbprint }",
                ],
            )
            .lines()
            .map(str::to_owned)
            .collect(),
            _ => Vec::new(),
        };
        lines.sort();
        listing.push_str(&format!(
            "{:?} {location}: {} entries\n",
            target.store,
            lines.len()
        ));
        listing.push_str(&lines.join("\n"));
        listing.push('\n');
    }
    listing
}

/// The Linux system store's anchor directory and bundle, by name and bytes; empty elsewhere.
fn system_store_listing(plan: &Plan) -> Vec<(String, u64, Option<SystemTime>)> {
    let mut listing = Vec::new();
    for dir in plan.system_store.iter().map(|t| t.location.clone()).chain([
        PathBuf::from("/etc/ssl/certs"),
        PathBuf::from("/etc/pki/ca-trust/extracted/pem"),
    ]) {
        if let Ok(entries) = fs::read_dir(&dir) {
            for e in entries.flatten() {
                let meta = fs::symlink_metadata(e.path()).ok();
                listing.push((
                    e.path().display().to_string(),
                    meta.as_ref().map_or(0, fs::Metadata::len),
                    meta.and_then(|m| m.modified().ok()),
                ));
            }
        }
    }
    listing.sort();
    listing
}

/// The Linux system store for real, with `--system`: the anchor file is installed and the
/// system bundle rebuilt through sudo, and `--remove` takes both back to the bytes they had.
/// Ignored by default for the same reason as [`real_store_round_trip`]; it prints a skip where
/// the host has no system store.
#[test]
#[ignore = "changes this machine's system trust store through sudo; run on a throwaway CI runner"]
fn real_system_store_round_trip() {
    let root = temp_dir("real-system");
    let state = root.join("state");
    fs::create_dir(&state).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&state, fs::Permissions::from_mode(0o700)).unwrap();
    }
    let mut plan = Plan::for_this_host(&root, root.join("trust"));
    // Only the system store: the user stores are `real_store_round_trip`'s.
    plan.user_stores.clear();
    let Some(system) = plan.system_store.clone() else {
        println!("SKIP: this host has no system store Syntropy changes");
        return;
    };
    let bundles = [
        PathBuf::from("/etc/ssl/certs/ca-certificates.crt"),
        PathBuf::from("/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem"),
    ];
    let read_bundles =
        || -> Vec<Option<Vec<u8>>> { bundles.iter().map(|b| fs::read(b).ok()).collect() };
    let anchors = || -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(&system.location)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    };
    let (bundles_before, anchors_before) = (read_bundles(), anchors());
    let pem = ca_pem(14);
    let ca = CaCertificate::from_pem(&pem).unwrap();
    let mut ledger = HostChangeLedger::open(&state).unwrap();
    let mut tool = os::OsStoreTool;
    let mut answers = Answers::default();
    let installed = Trust {
        plan: &plan,
        ledger: &mut ledger,
        tool: &mut tool,
        prompt: &mut answers,
        by_version: "evidence",
    }
    .install("box.local", true, &mut Source(pem.clone()), 1)
    .unwrap();
    println!("installed: {:?}", installed.stores);
    assert!(installed.complete(), "{:?}", installed.stores);
    let bundle_now = read_bundles();
    let in_bundle = bundle_now
        .iter()
        .flatten()
        .any(|b| store::pem_bodies(&String::from_utf8_lossy(b)).contains(&ca.base64()));
    assert!(in_bundle, "the rebuilt system bundle holds the CA");
    assert_ne!(anchors(), anchors_before);
    let removed = Trust {
        plan: &plan,
        ledger: &mut ledger,
        tool: &mut tool,
        prompt: &mut answers,
        by_version: "evidence",
    }
    .remove("box.local")
    .unwrap();
    println!("removed: {:?}", removed.reverted);
    assert!(
        removed
            .reverted
            .iter()
            .all(|r| r.outcome == Outcome::Reverted)
    );
    assert_eq!(anchors(), anchors_before);
    assert_eq!(
        read_bundles(),
        bundles_before,
        "the system bundle is back to its bytes"
    );
    println!(
        "PASS: --system install then remove leaves the system store's anchors and bundle as they were"
    );
}
