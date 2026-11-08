// SPDX-License-Identifier: MIT OR Apache-2.0

//! The trust stores `syntropy trust` changes, behind [`StoreTool`], and [`TrustHost`], the revert
//! host that reverts their entries along with the file-system changes.

use std::io;
use std::path::{Path, PathBuf};

use crate::host_changes::revert::{FsHost, Host, Observed};
use crate::host_changes::{HostChange, HostChangeKind, HostChangeLedger, TrustStore};

/// One certificate in one trust store: what a [`HostChangeKind::TrustAnchor`] records.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Anchor {
    /// The store.
    pub store: TrustStore,
    /// Where in it: the keychain file, the NSS database directory or the system anchor file;
    /// `Cert:\CurrentUser\Root` for the Windows store.
    pub location: PathBuf,
    /// The NSS nickname, or the name the certificate is shown under elsewhere.
    pub label: String,
    /// The certificate, DER in base64 on one line.
    pub certificate: String,
}

impl Anchor {
    /// The ledger entry for this anchor, for `node`.
    #[must_use]
    pub fn kind(&self, node: &str) -> HostChangeKind {
        HostChangeKind::TrustAnchor {
            store: self.store,
            location: self.location.clone(),
            label: self.label.clone(),
            node: node.to_owned(),
            certificate: self.certificate.clone(),
        }
    }

    /// The anchor a ledger entry records, if it records one.
    #[must_use]
    pub fn of(kind: &HostChangeKind) -> Option<Anchor> {
        match kind {
            HostChangeKind::TrustAnchor {
                store,
                location,
                label,
                certificate,
                ..
            } => Some(Anchor {
                store: *store,
                location: location.clone(),
                label: label.clone(),
                certificate: certificate.clone(),
            }),
            _ => None,
        }
    }
}

/// The operations on a trust store, through the OS's own tools ([`super::os::OsStoreTool`]) or,
/// in tests, a fake.
pub trait StoreTool {
    /// What the store shows for `anchor`: [`Observed::AsRecorded`] when it holds exactly the
    /// certificate, [`Observed::Absent`] when it does not, and [`Observed::Changed`] when the
    /// anchor's label or file holds another certificate.
    ///
    /// # Errors
    ///
    /// The tool could not be run or its output read.
    fn observe(&self, anchor: &Anchor) -> io::Result<Observed>;

    /// Adds the certificate in `pem_file`, which holds exactly `anchor.certificate`, to the store
    /// as a trusted root.
    ///
    /// # Errors
    ///
    /// The tool failed or refused.
    fn add(&mut self, anchor: &Anchor, pem_file: &Path) -> io::Result<()>;

    /// Removes exactly `anchor.certificate` from the store.
    ///
    /// # Errors
    ///
    /// The tool failed or refused.
    fn delete(&mut self, anchor: &Anchor) -> io::Result<()>;
}

/// The revert host for a client: trust-store entries through a [`StoreTool`], and everything the
/// binary reverts itself, the PEM files among them, through [`FsHost`]. `syntropy trust --remove`
/// reverts through it, and so can `syntropy uninstall`, so an un-removed trust is reverted with
/// the rest of the ledger. The other OS kinds are [`io::ErrorKind::Unsupported`], as with
/// [`FsHost`]: the generated revert script reverts them.
#[derive(Debug)]
pub struct TrustHost<'t, T: StoreTool> {
    tool: &'t mut T,
}

impl<'t, T: StoreTool> TrustHost<'t, T> {
    /// A host that changes trust stores through `tool`.
    pub fn new(tool: &'t mut T) -> TrustHost<'t, T> {
        TrustHost { tool }
    }
}

impl<T: StoreTool> Host for TrustHost<'_, T> {
    fn observe(&self, ledger: &HostChangeLedger, change: &HostChange) -> io::Result<Observed> {
        match Anchor::of(&change.kind) {
            Some(anchor) => self.tool.observe(&anchor),
            None => FsHost.observe(ledger, change),
        }
    }

    fn remove(&mut self, change: &HostChange) -> io::Result<()> {
        match Anchor::of(&change.kind) {
            Some(anchor) => self.tool.delete(&anchor),
            None => FsHost.remove(change),
        }
    }

    fn restore(&mut self, change: &HostChange, from: &Path) -> io::Result<()> {
        FsHost.restore(change, from)
    }

    fn restore_value(&mut self, change: &HostChange, value: &str) -> io::Result<()> {
        FsHost.restore_value(change, value)
    }
}

/// The base64 bodies of the PEM `CERTIFICATE` blocks in a tool's output, each on one line: what
/// an [`Anchor`] records. Carriage returns and white space inside a block are dropped.
#[must_use]
pub fn pem_bodies(text: &str) -> Vec<String> {
    let mut bodies = Vec::new();
    let mut body: Option<String> = None;
    for line in text.lines() {
        let line = line.trim();
        if line == "-----BEGIN CERTIFICATE-----" {
            body = Some(String::new());
        } else if line == "-----END CERTIFICATE-----" {
            if let Some(done) = body.take() {
                bodies.push(done);
            }
        } else if let Some(open) = body.as_mut() {
            open.push_str(line);
        }
    }
    bodies
}
