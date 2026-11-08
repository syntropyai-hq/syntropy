// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`OsStoreTool`]: the trust stores through each OS's own tools (RFC-0050 §11.1.1).
//!
//! - **macOS:** `security add-trusted-cert -r trustRoot -k <login keychain>` to add,
//!   `security find-certificate -a -Z -p` to find the certificate and its SHA-1 hash, and
//!   `security delete-certificate -Z <hash> -t` to remove it with its trust settings.
//! - **Windows:** the current user's Root store through PowerShell's `X509Store`, matching the
//!   certificate by its exact bytes.
//! - **Linux, user:** `certutil -d sql:<NSS database>`, `-A -t C,,` to add, `-L -n <nickname> -a`
//!   to read, `-D -n <nickname>` to remove.
//! - **Linux, system** (`--system` only): an anchor file installed with `install -m 0644`, then
//!   `update-ca-certificates` or `update-ca-trust extract`, each through `sudo` unless the
//!   process is already root.
//!
//! Every tool gets its arguments directly, never through a shell, and every certificate is
//! matched by its exact DER (base64), so nothing but the recorded certificate is ever removed.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use super::store::{Anchor, StoreTool, pem_bodies};
use crate::host_changes::TrustStore;
use crate::host_changes::revert::Observed;

/// The trust stores through the OS's own tools.
#[derive(Debug, Default)]
pub struct OsStoreTool;

fn failed(what: &str, output: &Output) -> io::Error {
    io::Error::other(format!(
        "{what} failed ({}): {}",
        output.status,
        String::from_utf8_lossy(&output.stderr).trim()
    ))
}

/// Runs `command` with its output captured, and fails unless it succeeds.
fn run(what: &str, command: &mut Command) -> io::Result<Output> {
    let output = command.stdin(Stdio::null()).output()?;
    if output.status.success() {
        Ok(output)
    } else {
        Err(failed(what, &output))
    }
}

/// Runs `command` on the user's terminal, so `sudo` can ask for a password, and fails unless it
/// succeeds.
fn run_on_terminal(what: &str, command: &mut Command) -> io::Result<()> {
    let status = command.status()?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other(format!("{what} failed ({status})")))
    }
}

/// A command for the system store: through `sudo` unless this process is root.
fn privileged(program: &str) -> Command {
    let root = Command::new("id")
        .arg("-u")
        .output()
        .is_ok_and(|o| o.status.success() && o.stdout.trim_ascii() == b"0");
    if root {
        Command::new(program)
    } else {
        let mut command = Command::new("sudo");
        command.arg("--").arg(program);
        command
    }
}

/// The command that rebuilds a Linux system store from its anchors.
fn rebuild(store: TrustStore) -> Command {
    let mut command = privileged(if store == TrustStore::LinuxSystemCaCertificates {
        "update-ca-certificates"
    } else {
        "update-ca-trust"
    });
    if store == TrustStore::LinuxSystemCaTrust {
        command.arg("extract");
    }
    command
}

fn nss_db(anchor: &Anchor) -> String {
    format!("sql:{}", anchor.location.display())
}

/// The SHA-1 hashes `security find-certificate -a -Z -p` prints for the certificates whose PEM
/// body is `certificate`. The hash is how `security delete-certificate` names a certificate.
fn keychain_hashes(output: &str, certificate: &str) -> Vec<String> {
    let mut hashes = Vec::new();
    let mut hash = None;
    let mut body: Option<String> = None;
    for line in output.lines() {
        let line = line.trim();
        if let Some(h) = line.strip_prefix("SHA-1 hash:") {
            hash = Some(h.trim().to_owned());
        } else if line == "-----BEGIN CERTIFICATE-----" {
            body = Some(String::new());
        } else if line == "-----END CERTIFICATE-----" {
            if body.take().as_deref() == Some(certificate)
                && let Some(h) = hash.take()
            {
                hashes.push(h);
            }
        } else if let Some(open) = body.as_mut() {
            open.push_str(line);
        }
    }
    hashes
}

fn find_in_keychain(anchor: &Anchor) -> io::Result<Vec<String>> {
    let output = Command::new("security")
        .args(["find-certificate", "-a", "-Z", "-p"])
        .arg(&anchor.location)
        .stdin(Stdio::null())
        .output()?;
    // An empty keychain is an error to `security`; it holds nothing either way.
    Ok(keychain_hashes(
        &String::from_utf8_lossy(&output.stdout),
        &anchor.certificate,
    ))
}

/// A PowerShell command over the current user's Root store, with `$b` set to the certificate.
fn root_store(anchor: &Anchor, body: &str) -> Command {
    let mut command = Command::new("powershell.exe");
    command.args([
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        &format!(
            "$ErrorActionPreference = 'Stop'; $b = '{}'; \
             $st = New-Object Security.Cryptography.X509Certificates.X509Store('Root', 'CurrentUser'); \
             try {{ {body} }} finally {{ $st.Close() }}",
            // Checked as base64 by the ledger, so it holds no quote.
            anchor.certificate
        ),
    ]);
    command
}

const MATCHING: &str =
    "@($st.Certificates | Where-Object { [Convert]::ToBase64String($_.RawData) -ceq $b })";

/// The certificate in a Linux system anchor file, as one base64 line; `Changed` for something
/// that is not a regular file.
fn read_anchor_file(path: &Path) -> io::Result<Option<Observed>> {
    match fs::symlink_metadata(path) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Some(Observed::Absent)),
        Err(e) => Err(e),
        Ok(meta) if !meta.file_type().is_file() => Ok(Some(Observed::Changed)),
        Ok(_) => Ok(None),
    }
}

/// Refuses a system-store anchor outside its store's directory: those commands run as root.
fn outside_store(anchor: &Anchor) -> io::Result<()> {
    if anchor.store.holds_anchor(&anchor.location) {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "{} is not a Syntropy anchor file in {}; not touched",
            anchor.location.display(),
            anchor
                .store
                .anchor_dir()
                .map_or_else(String::new, |d| d.display().to_string())
        )))
    }
}

/// What a Linux system anchor file holds: [`Observed::AsRecorded`] only for a regular file whose
/// one certificate is `anchor.certificate`.
fn observe_anchor_file(anchor: &Anchor) -> io::Result<Observed> {
    if let Some(observed) = read_anchor_file(&anchor.location)? {
        return Ok(observed);
    }
    let text = fs::read_to_string(&anchor.location).unwrap_or_default();
    Ok(match pem_bodies(&text).as_slice() {
        [only] if *only == anchor.certificate => Observed::AsRecorded,
        _ => Observed::Changed,
    })
}

/// Installs a system anchor file with `install`, then reads it back before anything trusts it.
///
/// `install` copies the PEM file from a directory the user owns, and runs only after the user has
/// typed a `sudo` password, so a process of the same user could replace that file in between.
/// The anchor's own directory is root's, so the copy cannot change once it is there: if it holds
/// anything but the certificate the user confirmed, `remove` deletes it and the store is not
/// rebuilt.
fn install_checked(
    anchor: &Anchor,
    install: impl FnOnce() -> io::Result<()>,
    remove: impl FnOnce() -> io::Result<()>,
) -> io::Result<()> {
    install()?;
    match observe_anchor_file(anchor) {
        Ok(Observed::AsRecorded) => Ok(()),
        held => {
            let removed = remove();
            Err(io::Error::other(format!(
                "{} did not hold the confirmed certificate after it was installed{}; {}, and the \
                 system store was not rebuilt",
                anchor.location.display(),
                held.err().map_or_else(String::new, |e| format!(" ({e})")),
                match removed {
                    Ok(()) => "it was removed".to_owned(),
                    Err(e) => format!("removing it failed too ({e})"),
                }
            )))
        }
    }
}

fn is_base64(text: &str) -> bool {
    !text.is_empty()
        && text
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '='))
}

impl StoreTool for OsStoreTool {
    fn observe(&self, anchor: &Anchor) -> io::Result<Observed> {
        if !is_base64(&anchor.certificate) {
            return Err(io::Error::other("the recorded certificate is not base64"));
        }
        match anchor.store {
            TrustStore::MacosLoginKeychain => Ok(if find_in_keychain(anchor)?.is_empty() {
                Observed::Absent
            } else {
                Observed::AsRecorded
            }),
            TrustStore::LinuxUserNss => {
                let output = Command::new("certutil")
                    .args(["-d", &nss_db(anchor), "-L", "-n", &anchor.label, "-a"])
                    .stdin(Stdio::null())
                    .output()?;
                if !output.status.success() {
                    // No certificate has this nickname, or there is no database.
                    return Ok(Observed::Absent);
                }
                let bodies = pem_bodies(&String::from_utf8_lossy(&output.stdout));
                Ok(match bodies.as_slice() {
                    [] => Observed::Absent,
                    [only] if *only == anchor.certificate => Observed::AsRecorded,
                    _ => Observed::Changed,
                })
            }
            TrustStore::LinuxSystemCaCertificates | TrustStore::LinuxSystemCaTrust => {
                observe_anchor_file(anchor)
            }
            TrustStore::WindowsCurrentUserRoot => {
                let output = run(
                    "reading the current user's Root store",
                    &mut root_store(anchor, &format!("$st.Open('ReadOnly'); {MATCHING}.Count")),
                )?;
                let count: u32 = String::from_utf8_lossy(&output.stdout)
                    .trim()
                    .parse()
                    .map_err(|_| io::Error::other("PowerShell did not print a count"))?;
                Ok(if count == 0 {
                    Observed::Absent
                } else {
                    Observed::AsRecorded
                })
            }
        }
    }

    fn add(&mut self, anchor: &Anchor, pem_file: &Path) -> io::Result<()> {
        match anchor.store {
            TrustStore::MacosLoginKeychain => run_on_terminal(
                "security add-trusted-cert",
                Command::new("security")
                    .args(["add-trusted-cert", "-r", "trustRoot", "-k"])
                    .arg(&anchor.location)
                    .arg(pem_file),
            ),
            TrustStore::LinuxUserNss => run(
                "certutil -A",
                Command::new("certutil")
                    .args([
                        "-d",
                        &nss_db(anchor),
                        "-A",
                        "-t",
                        "C,,",
                        "-n",
                        &anchor.label,
                        "-i",
                    ])
                    .arg(pem_file),
            )
            .map(drop),
            TrustStore::LinuxSystemCaCertificates | TrustStore::LinuxSystemCaTrust => {
                outside_store(anchor)?;
                install_checked(
                    anchor,
                    || {
                        run_on_terminal(
                            "installing the anchor file",
                            privileged("install")
                                .args(["-m", "0644", "--"])
                                .arg(pem_file)
                                .arg(&anchor.location),
                        )
                    },
                    || {
                        run_on_terminal(
                            "removing the anchor file",
                            privileged("rm").args(["-f", "--"]).arg(&anchor.location),
                        )
                    },
                )?;
                run_on_terminal("rebuilding the system store", &mut rebuild(anchor.store))
            }
            TrustStore::WindowsCurrentUserRoot => run_on_terminal(
                "adding to the current user's Root store",
                &mut root_store(
                    anchor,
                    "$c = New-Object Security.Cryptography.X509Certificates.X509Certificate2(,[Convert]::FromBase64String($b)); \
                     $st.Open('ReadWrite'); $st.Add($c)",
                ),
            ),
        }
    }

    fn delete(&mut self, anchor: &Anchor) -> io::Result<()> {
        match anchor.store {
            TrustStore::MacosLoginKeychain => {
                for hash in find_in_keychain(anchor)? {
                    run_on_terminal(
                        "security delete-certificate",
                        Command::new("security")
                            .args(["delete-certificate", "-Z", &hash, "-t"])
                            .arg(&anchor.location),
                    )?;
                }
                Ok(())
            }
            TrustStore::LinuxUserNss => run(
                "certutil -D",
                Command::new("certutil").args(["-d", &nss_db(anchor), "-D", "-n", &anchor.label]),
            )
            .map(drop),
            TrustStore::LinuxSystemCaCertificates | TrustStore::LinuxSystemCaTrust => {
                outside_store(anchor)?;
                run_on_terminal(
                    "removing the anchor file",
                    privileged("rm").args(["-f", "--"]).arg(&anchor.location),
                )?;
                run_on_terminal("rebuilding the system store", &mut rebuild(anchor.store))
            }
            TrustStore::WindowsCurrentUserRoot => run_on_terminal(
                "removing from the current user's Root store",
                &mut root_store(
                    anchor,
                    &format!(
                        "$st.Open('ReadWrite'); foreach ($c in {MATCHING}) {{ $st.Remove($c) }}"
                    ),
                ),
            ),
        }
    }
}

/// `name` on `PATH`, or in `/usr/sbin` or `/sbin`, which an unprivileged user's `PATH` may lack.
#[must_use]
pub fn find_program(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH").unwrap_or_default();
    std::env::split_paths(&path)
        .chain([PathBuf::from("/usr/sbin"), PathBuf::from("/sbin")])
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn system_anchor(dir: &Path) -> Anchor {
        Anchor {
            store: TrustStore::LinuxSystemCaCertificates,
            location: dir.join("syntropy-box.local-1.crt"),
            label: "box.local".into(),
            certificate: "T1VSUw==".into(),
        }
    }

    fn pem(body: &str) -> String {
        format!("-----BEGIN CERTIFICATE-----\n{body}\n-----END CERTIFICATE-----\n")
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("syntropy-os-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The PEM file was swapped between the user's confirmation and `install`: the installed
    /// anchor holds another certificate, so it is removed and the store is never rebuilt.
    #[test]
    fn a_swapped_anchor_is_removed_before_the_store_trusts_it() {
        let dir = scratch("swapped");
        let anchor = system_anchor(&dir);
        let mut removed = false;
        let result = install_checked(
            &anchor,
            || fs::write(&anchor.location, pem("QVRUQUNLRVI=")),
            || {
                removed = true;
                fs::remove_file(&anchor.location)
            },
        );
        let error = result.unwrap_err().to_string();
        assert!(
            error.contains("did not hold the confirmed certificate"),
            "{error}"
        );
        assert!(removed);
        assert!(!anchor.location.exists());

        // A second certificate appended to the confirmed one is refused the same way.
        let result = install_checked(
            &anchor,
            || fs::write(&anchor.location, pem("T1VSUw==") + &pem("QVRUQUNLRVI=")),
            || fs::remove_file(&anchor.location),
        );
        assert!(result.is_err());
        assert!(!anchor.location.exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_confirmed_anchor_is_kept() {
        let dir = scratch("kept");
        let anchor = system_anchor(&dir);
        install_checked(
            &anchor,
            || fs::write(&anchor.location, pem("T1VS\nUw==")),
            || panic!("nothing to remove"),
        )
        .unwrap();
        assert_eq!(observe_anchor_file(&anchor).unwrap(), Observed::AsRecorded);
        fs::remove_dir_all(&dir).unwrap();
    }

    /// An anchor that is a link, not a regular file, is never taken as the certificate.
    #[cfg(unix)]
    #[test]
    fn a_linked_anchor_is_changed_not_as_recorded() {
        let dir = scratch("linked");
        let anchor = system_anchor(&dir);
        let target = dir.join("elsewhere.crt");
        fs::write(&target, pem("T1VSUw==")).unwrap();
        std::os::unix::fs::symlink(&target, &anchor.location).unwrap();
        assert_eq!(observe_anchor_file(&anchor).unwrap(), Observed::Changed);
        fs::remove_dir_all(&dir).unwrap();
    }

    /// The shape `security find-certificate -a -Z -p` prints: hashes, attributes, then the PEM.
    #[test]
    fn keychain_output_gives_the_hash_of_the_matching_certificate() {
        let output = "SHA-256 hash: 11AA\nSHA-1 hash: AAAA\nkeychain: \"/k\"\nattributes:\n    \"labl\"<blob>=\"other\"\n-----BEGIN CERTIFICATE-----\nT1RIRVI=\n-----END CERTIFICATE-----\nSHA-256 hash: 22BB\nSHA-1 hash: BBBB\nkeychain: \"/k\"\n-----BEGIN CERTIFICATE-----\nT1VS\nUw==\n-----END CERTIFICATE-----\n";
        assert_eq!(keychain_hashes(output, "T1VSUw=="), ["BBBB"]);
        assert_eq!(keychain_hashes(output, "T1RIRVI="), ["AAAA"]);
        assert!(keychain_hashes(output, "Tk9ORQ==").is_empty());
    }

    #[test]
    fn pem_bodies_join_lines_and_drop_carriage_returns() {
        let text = "-----BEGIN CERTIFICATE-----\r\nAAAA\r\nBB==\r\n-----END CERTIFICATE-----\r\n";
        assert_eq!(pem_bodies(text), ["AAAABB=="]);
        assert!(pem_bodies("certutil: could not find certificate named").is_empty());
    }
}
