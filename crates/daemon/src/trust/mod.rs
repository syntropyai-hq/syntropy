// SPDX-License-Identifier: MIT OR Apache-2.0

//! `syntropy trust <node> [--system] [--remove]` on a client machine (RFC-0050 §11.1.1,
//! RFC-0051 §2.1; ADR-0505 item 6; PROD-E5-T04).
//!
//! [`Trust::install`] installs a node's local CA:
//! 1. It fetches the CA certificate through a [`CaSource`]. The source is the node's
//!    `GET /v1/syntropy/tls/ca` route, which PROD-E5-T06 serves; the CLI (PROD-E6-T08) supplies
//!    the HTTP client. The certificate is checked before anything else ([`CaCertificate`]): one
//!    PEM block, a self-signed CA with critical name constraints.
//! 2. It shows the fingerprint, six words and hex, and asks the user to compare it with the one
//!    the node shows ([`Prompt::confirm_fingerprint`]). Nothing changes on the host before that.
//! 3. It writes the PEM file, `<pem dir>/<node>.pem`, and installs the CA into the user's trust
//!    store ([`Plan::for_this_host`]): the macOS login keychain, the Windows current-user Root
//!    store, or on Linux the user's NSS database. The Linux system store, through
//!    `update-ca-certificates` or `update-ca-trust`, is changed only with `--system`, and only
//!    after [`Prompt::confirm_sudo`].
//! 4. Every change is recorded in the host-change ledger before it is made
//!    ([`crate::host_changes`]), and the revert script is written again afterwards.
//! 5. It returns the PEM file's path, for [`env_lines`]: `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`
//!    and `NODE_EXTRA_CA_CERTS`, which clients with their own CA bundles read.
//!
//! Running it again for the same node first reverts the node's entries for any other
//! certificate, so a node whose CA was replaced (SEC-006) leaves one CA trusted, not two.
//! [`Trust::remove`] reverts every open entry for the node, and only those, under the revert
//! rules. `syntropy uninstall` reverts an un-removed trust with the rest of the ledger, through
//! [`TrustHost`] in the binary and through the generated revert scripts after it.

pub mod os;
pub mod store;

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use syntropy_api::tls::{CaCertificate, CaRefused};

use crate::host_changes::revert::{FsHost, Host, Observed, Reverted, revert_where};
use crate::host_changes::script::{ps_quote, sh_quote, write_revert_script};
use crate::host_changes::{
    ChangeState, HostChange, HostChangeKind, HostChangeLedger, LedgerError, TrustStore, fingerprint,
};
pub use store::{Anchor, StoreTool, TrustHost};

/// The environment variables `syntropy trust` prints, which clients with their own CA bundles
/// read: Python's `ssl` module and the HTTP stacks on it, Python `requests`, and Node.js.
pub const CA_BUNDLE_VARIABLES: [&str; 3] =
    ["SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "NODE_EXTRA_CA_CERTS"];

/// Where a node's CA certificate comes from: the node's `GET /v1/syntropy/tls/ca` route.
pub trait CaSource {
    /// The PEM text the node serves.
    ///
    /// # Errors
    ///
    /// Why it could not be fetched, for the user.
    fn fetch_ca(&mut self, node: &str) -> Result<String, String>;
}

/// The questions `syntropy trust` asks before it changes anything.
pub trait Prompt {
    /// Shows `ca`'s fingerprint and what it may issue for, and asks whether it matches the one
    /// the node shows. `false` stops before anything changes.
    fn confirm_fingerprint(&mut self, node: &str, ca: &CaCertificate) -> bool;

    /// Asks whether to run `sudo` for `what`, a change to the system store. `false` stops before
    /// anything changes.
    fn confirm_sudo(&mut self, what: &str) -> bool;

    /// Tells the user something before a store change, such as that the OS is about to ask them
    /// to confirm it ([`os_confirmation`]).
    fn notice(&mut self, text: &str);
}

/// What the OS asks the user before `store` changes, for [`Prompt::notice`]: macOS asks to
/// confirm a change to the user's trust settings, and Windows asks to confirm adding a root
/// to, or removing one from, the current user's Root store. `syntropy trust` never answers
/// these itself: the confirmation is the user's. `None` for the stores that ask nothing.
#[must_use]
pub fn os_confirmation(store: TrustStore, adding: bool) -> Option<&'static str> {
    match (store, adding) {
        (TrustStore::MacosLoginKeychain, true) => Some(
            "macOS will ask you to confirm adding this CA to your trust settings, with your \
             password or Touch ID",
        ),
        (TrustStore::MacosLoginKeychain, false) => {
            Some("macOS may ask you to confirm removing this CA from your trust settings")
        }
        (TrustStore::WindowsCurrentUserRoot, true) => Some(
            "Windows will ask you to confirm installing this CA in your Root store; check that \
             the thumbprint it shows belongs to this certificate, then choose Yes",
        ),
        (TrustStore::WindowsCurrentUserRoot, false) => Some(
            "Windows will ask you to confirm removing this CA from your Root store; choose Yes",
        ),
        _ => None,
    }
}

/// A trust store on this host, and where in it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Target {
    /// The store.
    pub store: TrustStore,
    /// The keychain file or NSS database directory; for a Linux system store, the directory its
    /// anchor files go in; `Cert:\CurrentUser\Root` on Windows.
    pub location: PathBuf,
}

/// Where `syntropy trust` writes on this host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    /// The directory the PEM files go in, one per node.
    pub pem_dir: PathBuf,
    /// The user's trust stores.
    pub user_stores: Vec<Target>,
    /// User stores this host lacks, each with the reason, for the report.
    pub skipped: Vec<String>,
    /// The system store `--system` changes; Linux only.
    pub system_store: Option<Target>,
}

/// The Windows store's location in a [`Target`] and the ledger.
pub const WINDOWS_ROOT_STORE: &str = r"Cert:\CurrentUser\Root";

impl Plan {
    /// The stores of this OS, for the user whose home directory is `home`, with PEM files in
    /// `pem_dir`.
    /// - macOS: the login keychain, `~/Library/Keychains/login.keychain-db`.
    /// - Windows: the current user's Root store.
    /// - Linux: the user's NSS database, `~/.pki/nssdb`, when it exists and `certutil` is
    ///   installed; otherwise it is skipped with the reason, and the PEM file and the variables
    ///   remain. The system store is `update-ca-certificates`'s when that is installed, else
    ///   `update-ca-trust`'s.
    #[must_use]
    pub fn for_this_host(home: &Path, pem_dir: PathBuf) -> Plan {
        let mut plan = Plan {
            pem_dir,
            user_stores: Vec::new(),
            skipped: Vec::new(),
            system_store: None,
        };
        if cfg!(target_os = "macos") {
            plan.user_stores.push(Target {
                store: TrustStore::MacosLoginKeychain,
                location: home.join("Library/Keychains/login.keychain-db"),
            });
        } else if cfg!(windows) {
            plan.user_stores.push(Target {
                store: TrustStore::WindowsCurrentUserRoot,
                location: PathBuf::from(WINDOWS_ROOT_STORE),
            });
        } else {
            let nss = home.join(".pki/nssdb");
            if os::find_program("certutil").is_none() {
                plan.skipped.push(
                    "the NSS database: certutil is not installed (it is in the libnss3-tools or \
                     nss-tools package)"
                        .into(),
                );
            } else if !nss.join("cert9.db").is_file() {
                plan.skipped.push(format!(
                    "the NSS database: there is none at {} yet; browsers that use it create it, \
                     and running `syntropy trust` again then adds the CA to it",
                    nss.display()
                ));
            } else {
                plan.user_stores.push(Target {
                    store: TrustStore::LinuxUserNss,
                    location: nss,
                });
            }
            let debian = TrustStore::LinuxSystemCaCertificates
                .anchor_dir()
                .expect("a system store");
            let fedora = TrustStore::LinuxSystemCaTrust
                .anchor_dir()
                .expect("a system store");
            if os::find_program("update-ca-certificates").is_some() && debian.is_dir() {
                plan.system_store = Some(Target {
                    store: TrustStore::LinuxSystemCaCertificates,
                    location: debian.to_owned(),
                });
            } else if os::find_program("update-ca-trust").is_some() && fedora.is_dir() {
                plan.system_store = Some(Target {
                    store: TrustStore::LinuxSystemCaTrust,
                    location: fedora.to_owned(),
                });
            }
        }
        plan
    }

    /// The PEM file for `node`.
    #[must_use]
    pub fn pem_file(&self, node: &str) -> PathBuf {
        self.pem_dir.join(format!("{}.pem", node_slug(node)))
    }
}

/// `node` as it is recorded: trimmed and lowercase, since host names are case-insensitive.
#[must_use]
pub fn normalize_node(node: &str) -> String {
    node.trim().to_ascii_lowercase()
}

/// `node` as part of a file name, one to one: lowercase letters, digits, `-` and `.` are kept, and
/// every other byte, `_` among them, becomes `_` and its two hex digits, so `box:1` is
/// `box_3a1` and `box_1` is `box_5f1`. A leading `.` is escaped too, so the name is never hidden
/// and never `..`. A name whose part before the first `.` is one Windows reserves for a device
/// (`con`, `prn`, `aux`, `nul`, `com0`–`com9`, `lpt0`–`lpt9`) has its first letter escaped, so
/// `con.local` is `_63on.local`, a file Windows can create. Two nodes that differ after
/// [`normalize_node`] never share a PEM file: a literal `_` is always escaped, so the mapping can
/// be read back.
#[must_use]
pub fn node_slug(node: &str) -> String {
    let normal = normalize_node(node);
    if normal.is_empty() {
        return "_".into();
    }
    let reserved = windows_device_name(normal.split('.').next().unwrap_or_default());
    let mut slug = String::with_capacity(normal.len());
    for (i, byte) in normal.bytes().enumerate() {
        let kept = (byte.is_ascii_lowercase() && !(i == 0 && reserved))
            || byte.is_ascii_digit()
            || byte == b'-'
            || (byte == b'.' && i > 0);
        if kept {
            slug.push(char::from(byte));
        } else {
            slug.push_str(&format!("_{byte:02x}"));
        }
    }
    slug
}

/// Whether `stem`, a file name's part before its first `.`, is a device name Windows reserves
/// whatever the extension: `con`, `prn`, `aux`, `nul`, and `com` or `lpt` with one digit.
fn windows_device_name(stem: &str) -> bool {
    match stem.as_bytes() {
        b"con" | b"prn" | b"aux" | b"nul" => true,
        [b'c', b'o', b'm', d] | [b'l', b'p', b't', d] => d.is_ascii_digit(),
        _ => false,
    }
}

/// The shell [`env_lines`] writes for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shell {
    /// `sh`, `bash`, `zsh`: `export NAME='value'`.
    Posix,
    /// PowerShell: `$env:NAME = 'value'`.
    PowerShell,
}

impl Shell {
    /// This OS's usual shell: PowerShell on Windows, `sh` elsewhere.
    #[must_use]
    pub fn for_this_os() -> Shell {
        if cfg!(windows) {
            Shell::PowerShell
        } else {
            Shell::Posix
        }
    }
}

/// The lines that set [`CA_BUNDLE_VARIABLES`] to `pem_file` in `shell`, one per variable, the
/// path quoted for that shell. `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` replace a client's
/// default bundle rather than add to it, so set them for the clients that talk to Syntropy;
/// `NODE_EXTRA_CA_CERTS` adds to Node.js's bundle.
#[must_use]
pub fn env_lines(pem_file: &Path, shell: Shell) -> String {
    let path = pem_file.to_string_lossy();
    CA_BUNDLE_VARIABLES
        .iter()
        .map(|name| match shell {
            Shell::Posix => format!("export {name}={}\n", sh_quote(&path)),
            Shell::PowerShell => format!("$env:{name} = {}\n", ps_quote(&path)),
        })
        .collect()
}

/// Why `syntropy trust` stopped.
#[derive(Debug)]
#[non_exhaustive]
pub enum TrustError {
    /// No node was named.
    NoNode,
    /// The CA certificate could not be fetched.
    Fetch(String),
    /// What the node served is not a constrained CA certificate.
    Refused(CaRefused),
    /// The user said the fingerprint does not match. Nothing was changed.
    NotConfirmed,
    /// `--system` on a host with no system store Syntropy changes: macOS and Windows, or a Linux
    /// without `update-ca-certificates` or `update-ca-trust`. Nothing was changed.
    NoSystemStore,
    /// The user declined `sudo`. Nothing was changed.
    SudoDeclined,
    /// The ledger could not be read or written.
    Ledger(LedgerError),
    /// The PEM directory could not be made.
    Io {
        /// The directory.
        path: PathBuf,
        /// The error.
        source: io::Error,
    },
}

impl fmt::Display for TrustError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrustError::NoNode => f.write_str("name the node to trust"),
            TrustError::Fetch(why) => write!(f, "could not fetch the node's CA certificate: {why}"),
            TrustError::Refused(why) => write!(f, "{why}; nothing was installed"),
            TrustError::NotConfirmed => {
                f.write_str("the fingerprint was not confirmed; nothing was installed")
            }
            TrustError::NoSystemStore => f.write_str(
                "--system changes the Linux system store, through update-ca-certificates or \
                 update-ca-trust, and this host has neither; nothing was installed",
            ),
            TrustError::SudoDeclined => {
                f.write_str("the system store needs sudo, which was declined; nothing was changed")
            }
            TrustError::Ledger(e) => write!(f, "host-change ledger: {e}"),
            TrustError::Io { path, source } => write!(f, "{}: {source}", path.display()),
        }
    }
}

impl std::error::Error for TrustError {}

impl From<LedgerError> for TrustError {
    fn from(e: LedgerError) -> TrustError {
        TrustError::Ledger(e)
    }
}

impl From<CaRefused> for TrustError {
    fn from(e: CaRefused) -> TrustError {
        TrustError::Refused(e)
    }
}

/// How one store fared.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StoreResult {
    /// The CA was added, and recorded.
    Added,
    /// An earlier `syntropy trust` added it, and it is still there.
    AlreadyOurs,
    /// The store already held it, from elsewhere. Left as it is and not recorded, so a revert
    /// leaves it too.
    AlreadyThere,
    /// The certificate's nickname or anchor file holds another certificate. Left as it is.
    Occupied,
    /// The store's tool failed.
    Failed(String),
}

/// One store's line in a report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoreOutcome {
    /// The store.
    pub store: TrustStore,
    /// Where in it.
    pub location: PathBuf,
    /// How it went.
    pub result: StoreResult,
}

/// What [`Trust::install`] did.
#[derive(Debug)]
pub struct Installed {
    /// The node, as recorded.
    pub node: String,
    /// The CA certificate installed.
    pub ca: CaCertificate,
    /// The PEM file, for [`env_lines`].
    pub pem_file: PathBuf,
    /// Entries for an earlier certificate of this node, reverted first.
    pub superseded: Vec<Reverted>,
    /// Each store's outcome.
    pub stores: Vec<StoreOutcome>,
    /// User stores this host lacks, with the reason.
    pub skipped: Vec<String>,
    /// The revert script, written again.
    pub revert_script: PathBuf,
}

impl Installed {
    /// Whether every store took the CA or already held it.
    #[must_use]
    pub fn complete(&self) -> bool {
        self.stores.iter().all(|s| {
            matches!(
                s.result,
                StoreResult::Added | StoreResult::AlreadyOurs | StoreResult::AlreadyThere
            )
        })
    }
}

/// What [`Trust::remove`] did.
#[derive(Debug)]
pub struct Removed {
    /// The node, as recorded.
    pub node: String,
    /// One line per entry reverted or left, newest first. Empty when nothing was recorded for the
    /// node.
    pub reverted: Vec<Reverted>,
    /// The revert script, written again.
    pub revert_script: PathBuf,
}

/// `syntropy trust` on one host: where it writes, the ledger, the stores and the user.
#[derive(Debug)]
pub struct Trust<'a, T: StoreTool, P: Prompt> {
    /// Where it writes.
    pub plan: &'a Plan,
    /// The client's host-change ledger.
    pub ledger: &'a mut HostChangeLedger,
    /// The trust stores.
    pub tool: &'a mut T,
    /// The user.
    pub prompt: &'a mut P,
    /// The Syntropy version, for the ledger and the revert script.
    pub by_version: &'a str,
}

fn is_open(change: &HostChange) -> bool {
    matches!(change.state, ChangeState::Pending | ChangeState::Applied)
}

/// Whether `change` is one of `node`'s: a trust-store entry for it, or its PEM file.
fn belongs(change: &HostChange, node: &str, pem_file: &Path) -> bool {
    match &change.kind {
        HostChangeKind::TrustAnchor { node: n, .. } => n == node,
        HostChangeKind::FileWritten { path } => path == pem_file,
        _ => false,
    }
}

/// Whether `change`, one of the node's, is for a certificate other than `ca`.
fn stale(change: &HostChange, ca: &CaCertificate, pem: &[u8]) -> bool {
    match &change.kind {
        HostChangeKind::TrustAnchor { certificate, .. } => *certificate != ca.base64(),
        HostChangeKind::FileWritten { .. } => change.fingerprint != Some(fingerprint(pem)),
        _ => false,
    }
}

impl<T: StoreTool, P: Prompt> Trust<'_, T, P> {
    /// Fetches `node`'s CA, has the user confirm its fingerprint, writes the PEM file and installs
    /// the CA into the user's stores, and into the Linux system store with `system`. See the
    /// module documentation for the steps. `at` is the time recorded in the ledger, in Unix
    /// milliseconds.
    ///
    /// # Errors
    ///
    /// [`TrustError`] for a stop before any change: no node, the fetch failed, the certificate
    /// was refused, the fingerprint was not confirmed, `--system` with no system store, `sudo`
    /// declined. After changes have begun, only a ledger or PEM-directory failure stops it; a
    /// store whose tool fails is reported in [`Installed::stores`], and the others go on.
    pub fn install(
        &mut self,
        node: &str,
        system: bool,
        source: &mut impl CaSource,
        at: u64,
    ) -> Result<Installed, TrustError> {
        let node = normalize_node(node);
        if node.is_empty() {
            return Err(TrustError::NoNode);
        }
        let system_store = match (system, &self.plan.system_store) {
            (false, _) => None,
            (true, Some(target)) => Some(target.clone()),
            (true, None) => return Err(TrustError::NoSystemStore),
        };
        let text = source.fetch_ca(&node).map_err(TrustError::Fetch)?;
        let ca = CaCertificate::from_pem(&text)?;
        if !self.prompt.confirm_fingerprint(&node, &ca) {
            return Err(TrustError::NotConfirmed);
        }
        let pem = ca.pem();
        let pem_file = self.plan.pem_file(&node);

        // An earlier certificate of this node's, in a system store, needs sudo to remove too.
        self.ledger.reload()?;
        let stale_system = self.ledger.changes().iter().any(|c| {
            is_open(c)
                && belongs(c, &node, &pem_file)
                && stale(c, &ca, pem.as_bytes())
                && Anchor::of(&c.kind).is_some_and(|a| a.store.is_system())
        });
        if (system_store.is_some() || stale_system)
            && !self.prompt.confirm_sudo(&format!(
                "to change the system trust store, which every user of this machine trusts, \
                 for node {node}"
            ))
        {
            return Err(TrustError::SudoDeclined);
        }

        let superseded_by = |c: &HostChange| {
            is_open(c) && belongs(c, &node, &pem_file) && stale(c, &ca, pem.as_bytes())
        };
        let superseded = if self.ledger.changes().iter().any(superseded_by) {
            revert_where(self.ledger, &mut TrustHost::new(self.tool), superseded_by)?
        } else {
            Vec::new()
        };

        self.write_pem(&pem_file, pem.as_bytes(), at)?;

        let label = format!(
            "Syntropy CA {} {}",
            node_slug(&node),
            &ca.fingerprint_hex()[..16]
        );
        let mut stores = Vec::new();
        for target in self.plan.user_stores.iter().chain(system_store.as_ref()) {
            let location = match target.store {
                TrustStore::LinuxSystemCaCertificates => target.location.join(format!(
                    "syntropy-{}-{}.crt",
                    node_slug(&node),
                    &ca.fingerprint_hex()[..16]
                )),
                TrustStore::LinuxSystemCaTrust => target.location.join(format!(
                    "syntropy-{}-{}.pem",
                    node_slug(&node),
                    &ca.fingerprint_hex()[..16]
                )),
                _ => target.location.clone(),
            };
            let anchor = Anchor {
                store: target.store,
                location,
                label: label.clone(),
                certificate: ca.base64(),
            };
            let result = self.add_anchor(&node, &anchor, &pem_file, at)?;
            stores.push(StoreOutcome {
                store: anchor.store,
                location: anchor.location,
                result,
            });
        }
        let revert_script = write_revert_script(self.ledger, self.by_version)?;
        Ok(Installed {
            node,
            ca,
            pem_file,
            superseded,
            stores,
            skipped: self.plan.skipped.clone(),
            revert_script,
        })
    }

    /// Writes the PEM file under the ledger, unless an open entry of ours already holds it.
    fn write_pem(&mut self, pem_file: &Path, pem: &[u8], at: u64) -> Result<(), TrustError> {
        fs::create_dir_all(&self.plan.pem_dir).map_err(|source| TrustError::Io {
            path: self.plan.pem_dir.clone(),
            source,
        })?;
        self.ledger.reload()?;
        let ours = self.ledger.changes().iter().any(|c| {
            is_open(c)
                && matches!(&c.kind, HostChangeKind::FileWritten { path } if path == pem_file)
                && c.fingerprint == Some(fingerprint(pem))
                && FsHost.observe(self.ledger, c).ok() == Some(Observed::AsRecorded)
        });
        if !ours {
            self.ledger.write_file(pem_file, pem, at, self.by_version)?;
        }
        Ok(())
    }

    /// Adds `anchor` to its store, recorded before it is made, unless it is already there.
    fn add_anchor(
        &mut self,
        node: &str,
        anchor: &Anchor,
        pem_file: &Path,
        at: u64,
    ) -> Result<StoreResult, TrustError> {
        let kind = anchor.kind(node);
        let observed = match self.tool.observe(anchor) {
            Ok(observed) => observed,
            Err(e) => return Ok(StoreResult::Failed(e.to_string())),
        };
        match observed {
            Observed::AsRecorded => {
                self.ledger.reload()?;
                let ours = self
                    .ledger
                    .changes()
                    .iter()
                    .any(|c| is_open(c) && c.kind == kind);
                return Ok(if ours {
                    StoreResult::AlreadyOurs
                } else {
                    StoreResult::AlreadyThere
                });
            }
            Observed::Changed => return Ok(StoreResult::Occupied),
            Observed::Absent | Observed::AsPrior => {}
        }
        let id = self.ledger.begin(kind, None, None, at, self.by_version)?;
        if let Some(text) = os_confirmation(anchor.store, true) {
            self.prompt.notice(text);
        }
        match self.tool.add(anchor, pem_file) {
            Ok(()) => {
                self.ledger.applied(id)?;
                Ok(StoreResult::Added)
            }
            Err(e) => {
                // A store that shows nothing of it never got it; one that does keeps the entry
                // pending, for a revert to resolve.
                if self.tool.observe(anchor).ok() == Some(Observed::Absent) {
                    self.ledger.abandoned(id)?;
                }
                Ok(StoreResult::Failed(e.to_string()))
            }
        }
    }

    /// Reverts every open entry for `node`, newest first, under the revert rules: its trust-store
    /// entries and its PEM file. Every other entry is left as it is.
    ///
    /// # Errors
    ///
    /// [`TrustError::SudoDeclined`] before anything changes if an entry is in a system store and
    /// the user declines `sudo`; [`TrustError::Ledger`] if the ledger cannot be read or saved.
    pub fn remove(&mut self, node: &str) -> Result<Removed, TrustError> {
        let node = normalize_node(node);
        if node.is_empty() {
            return Err(TrustError::NoNode);
        }
        let pem_file = self.plan.pem_file(&node);
        self.ledger.reload()?;
        let system = self.ledger.changes().iter().any(|c| {
            is_open(c)
                && belongs(c, &node, &pem_file)
                && Anchor::of(&c.kind).is_some_and(|a| a.store.is_system())
        });
        if system
            && !self.prompt.confirm_sudo(&format!(
                "to remove node {node}'s CA from the system trust store"
            ))
        {
            return Err(TrustError::SudoDeclined);
        }
        let mut told = Vec::new();
        for change in self.ledger.changes() {
            if let Some(anchor) = Anchor::of(&change.kind)
                && is_open(change)
                && belongs(change, &node, &pem_file)
                && !told.contains(&anchor.store)
                && let Some(text) = os_confirmation(anchor.store, false)
            {
                told.push(anchor.store);
                self.prompt.notice(text);
            }
        }
        let reverted = revert_where(self.ledger, &mut TrustHost::new(self.tool), |c| {
            is_open(c) && belongs(c, &node, &pem_file)
        })?;
        let revert_script = write_revert_script(self.ledger, self.by_version)?;
        Ok(Removed {
            node,
            reverted,
            revert_script,
        })
    }
}

#[cfg(test)]
mod tests;
