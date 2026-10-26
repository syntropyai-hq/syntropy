// SPDX-License-Identifier: MIT OR Apache-2.0

//! The generated revert scripts (RFC-0052 §11.1): `revert-host-changes.sh` on Unix and
//! `revert-host-changes.ps1` on Windows, written to the state directory after every change.
//!
//! A script undoes every open change in the ledger, newest first, under the same rules as
//! [`super::revert`], with nothing but the OS's standard tools, so it works after the binary is
//! deleted. It checks a written file against the ledger's `.written` copy (`cmp` on Unix,
//! `Get-FileHash` on Windows), because BLAKE3 has no standard command. It prints one line per
//! change and the number left in place, and it is safe to run twice. Changes already reverted or
//! abandoned are left out.

use std::path::{Path, PathBuf};

use super::revert::describe;
use super::{
    ChangeState, FirewallKind, HostChange, HostChangeKind, HostChangeLedger, LedgerError,
    PriorState, ServiceManager, TrustStore, replace_atomically,
};

/// The Unix script's file name in the state directory.
pub const SH_FILE: &str = "revert-host-changes.sh";

/// The Windows script's file name in the state directory.
pub const PS1_FILE: &str = "revert-host-changes.ps1";

/// `s` as one single-quoted `sh` word. A NUL cannot be in a path or a shell line, so it is
/// dropped rather than passed to the shell.
pub(crate) fn sh_quote(s: &str) -> String {
    format!("'{}'", s.replace('\0', "").replace('\'', r"'\''"))
}

/// The characters PowerShell treats as a single quote: the ASCII one and four typographic ones.
const PS_SINGLE_QUOTES: [char; 5] = ['\'', '\u{2018}', '\u{2019}', '\u{201A}', '\u{201B}'];

/// `s` as one single-quoted PowerShell string: every character PowerShell reads as a single quote
/// is doubled, so none can end the string. A NUL is dropped.
pub(crate) fn ps_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for ch in s.chars().filter(|&c| c != '\0') {
        out.push(ch);
        if PS_SINGLE_QUOTES.contains(&ch) {
            out.push(ch);
        }
    }
    out.push('\'');
    out
}

/// Starts every rendered PowerShell script; see [`render_ps1`].
const PS1_BOM: char = '\u{feff}';

/// `s` for a comment line: every control character and Unicode line or paragraph separator becomes
/// a space, so a value cannot end the comment in either shell.
fn comment(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_control() || matches!(c, '\u{2028}' | '\u{2029}') {
                ' '
            } else {
                c
            }
        })
        .collect()
}

/// The prior copy a revert restores from, only if it is the ledger's own copy for the change; any
/// other recorded path is not trusted.
fn own_prior(ledger: &HostChangeLedger, change: &HostChange) -> Option<PathBuf> {
    match &change.prior {
        Some(PriorState::SavedAt { path }) if *path == ledger.prior_copy(change.id) => {
            Some(path.clone())
        }
        _ => None,
    }
}

fn path_str(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// The changes a script reverts, newest first.
fn open_changes(ledger: &HostChangeLedger) -> impl Iterator<Item = &HostChange> {
    ledger
        .changes()
        .iter()
        .rev()
        .filter(|c| matches!(c.state, ChangeState::Pending | ChangeState::Applied))
}

/// Whether a pending change has a saved prior copy, which a revert must keep unless the host
/// shows the content from before the change ([`super::revert::decide`]).
fn pending_with_copy(change: &HostChange) -> bool {
    change.state == ChangeState::Pending && matches!(change.prior, Some(PriorState::SavedAt { .. }))
}

/// What the script says when something is there but not as recorded: an applied change was edited
/// by the user; a pending one was never made, unless it has a saved copy, when it may have partly
/// happened and is held with its copies.
fn sh_not_ours(change: &HostChange, what: &str) -> String {
    match change.state {
        ChangeState::Pending if pending_with_copy(change) => format!("hold \"{what}\""),
        ChangeState::Pending => format!("ok \"never made: {what}\""),
        _ => format!("leave \"{what}\""),
    }
}

fn ps_not_ours(change: &HostChange, what: &str) -> String {
    match change.state {
        ChangeState::Pending if pending_with_copy(change) => format!("Hold \"{what}\""),
        ChangeState::Pending => format!("Reverted \"never made: {what}\""),
        _ => format!("Leave \"{what}\""),
    }
}

/// What the script says when nothing is there: gone, unless a pending change with a saved copy,
/// which is held: the file it replaced is gone too, and the copy may be all that is left of it.
fn sh_absent(change: &HostChange, what: &str) -> String {
    if pending_with_copy(change) {
        format!("hold \"{what}\"")
    } else {
        format!("gone \"{what}\"")
    }
}

fn ps_absent(change: &HostChange, what: &str) -> String {
    if pending_with_copy(change) {
        format!("Hold \"{what}\"")
    } else {
        format!("Gone \"{what}\"")
    }
}

/// The `sh` script for `ledger`.
#[must_use]
pub fn render_sh(ledger: &HostChangeLedger, by_version: &str) -> String {
    let mut out = format!(
        "#!/bin/sh
# Reverts the host changes Syntropy recorded in {ledger}, newest first, under the revert rules of
# RFC-0052 section 11.1: undo only what still matches what was recorded, restore the prior state,
# and touch nothing else. Generated by Syntropy {version}. It needs no Syntropy binary, and
# running it twice is safe.
set -u
# Bytes, not characters: every comparison and every rewrite below works on bytes, whatever the
# files' encoding.
LC_ALL=C; export LC_ALL
cr=$(printf '\\r')
left=0
m=; cw=; cpr=
note() {{ printf 'syntropy-revert: %s\\n' \"$1\"; }}
# Records that the current change is done, so a second run skips it, and deletes its copies.
mark() {{ : > \"$m\"; rm -f -- \"$cw\" \"$cpr\"; }}
ok() {{ note \"$1\"; mark; }}
gone() {{ note \"already gone: $1\"; mark; }}
leave() {{ note \"left in place, edited since Syntropy changed it: $1\"; left=$((left + 1)); }}
# A pending change with a saved copy that matches neither the record nor the state before it may
# have partly happened: it is reported and left, with its copies, for the user.
hold() {{ note \"left in place, a pending change that may have partly happened: $1; its saved copies are kept in the state directory\"; left=$((left + 1)); }}
# Removes every line of file $1 that is exactly $2, or $2 and a carriage return, and keeps every
# other byte: the lines are found by number (grep -n, comparing bytes) and cut out with head and
# tail, which copy bytes. The new file takes $1's mode and owner (cp -p) and is renamed over it.
dropline() {{
  ns=$(grep -n -a -x -F -e \"$2\" -e \"$2$cr\" -- \"$1\" | cut -d: -f1 | sort -rn)
  t=$(mktemp \"$(dirname \"$1\")/.syntropy-revert.XXXXXX\") || return 1
  u=$(mktemp \"$(dirname \"$1\")/.syntropy-revert.XXXXXX\") || {{ rm -f -- \"$t\"; return 1; }}
  if cp -p -- \"$1\" \"$t\"; then
    for n in $ns; do
      {{ if [ \"$n\" -gt 1 ]; then head -n \"$((n - 1))\" -- \"$t\"; fi; tail -n \"+$((n + 1))\" -- \"$t\"; }} > \"$u\" && cat -- \"$u\" > \"$t\" || {{ rm -f -- \"$t\" \"$u\"; return 1; }}
    done
    rm -f -- \"$u\"
    mv -f -- \"$t\" \"$1\" && return 0
  fi
  rm -f -- \"$t\" \"$u\"
  return 1
}}
# A revert may run with more privilege than whoever could write the state, so refuse unless the
# state directory and the copies directory are private to the user running it. The copies
# directory is made only once the state directory passes, and only if nothing is in its place.
private() {{
  if [ -L \"$1\" ] || [ ! -d \"$1\" ] || [ ! -O \"$1\" ] || [ -n \"$(find \"$1\" -prune \\( -perm -0002 -o -perm -0020 \\) 2>/dev/null)\" ]; then
    note \"refusing to run: $1 is not private to this user\"; exit 1
  fi
}}
private {state}
[ -e {copies} ] || [ -L {copies} ] || mkdir -m 700 {copies}
private {copies}
# One writer at a time: the run holds the ledger's lock, as the binary does, where flock(1)
# exists. Without it (macOS has none) the script says so and runs unlocked.
lock={lock}
if command -v flock >/dev/null 2>&1; then
  exec 9>>\"$lock\" || {{ note \"refusing to run: cannot open $lock\"; exit 1; }}
  if ! flock -n 9; then
    note \"refusing to run: another Syntropy command is changing this host ($lock is locked)\"; exit 1
  fi
else
  note \"warning: flock(1) is not installed, so this script cannot take the ledger lock; run it only while no other Syntropy command is running\"
fi
",
        ledger = comment(&path_str(&ledger.state_dir().join(super::LEDGER_FILE))),
        version = comment(by_version),
        state = sh_quote(&path_str(ledger.state_dir())),
        copies = sh_quote(&path_str(&ledger.state_dir().join(super::COPIES_DIR))),
        lock = sh_quote(&path_str(&ledger.state_dir().join(super::LOCK_FILE))),
    );
    for change in open_changes(ledger) {
        out.push_str(&format!(
            "\n# change {} ({:?}): {}\nm={}; cw={}; cpr={}\nif [ -e \"$m\" ]; then\n  note \"already reverted: change {}\"\nelse\n",
            change.id,
            change.state,
            comment(&describe(&change.kind)),
            sh_quote(&path_str(&ledger.revert_marker(change.id))),
            sh_quote(&path_str(&ledger.written_copy(change.id))),
            sh_quote(&path_str(&ledger.prior_copy(change.id))),
            change.id,
        ));
        out.push_str(&sh_block(ledger, change));
        out.push_str("fi\n");
    }
    out.push_str(
        "\nif [ \"$left\" -gt 0 ]; then note \"$left change(s) left in place; see above\"; fi\nexit 0\n",
    );
    out
}

fn sh_block(ledger: &HostChangeLedger, change: &HostChange) -> String {
    match &change.kind {
        HostChangeKind::FileWritten { path } => {
            // The earlier content goes into a fresh file next to the target (mktemp, never an
            // existing path), which takes the target's mode and owner and is renamed over it.
            let undo = match own_prior(ledger, change) {
                Some(from) => format!(
                    "t=$(mktemp \"$(dirname \"$p\")/.syntropy-revert.XXXXXX\") && cp -p -- \"$p\" \"$t\" && cat -- {} > \"$t\" && mv -f -- \"$t\" \"$p\" && ok \"restored the earlier content: $p\"",
                    sh_quote(&path_str(&from))
                ),
                None => "rm -f -- \"$p\" && ok \"removed: $p\"".to_owned(),
            };
            // The content from before the change, from the ledger's own prior copy: never made if
            // pending, already undone if applied.
            let as_prior = match own_prior(ledger, change) {
                Some(_) => format!(
                    "elif [ ! -L \"$p\" ] && [ -f \"$p\" ] && [ -f \"$cpr\" ] && cmp -s \"$p\" \"$cpr\"; then\n  {}\n",
                    if change.state == ChangeState::Pending {
                        "ok \"never made: $p\""
                    } else {
                        "ok \"already back to its earlier content: $p\""
                    }
                ),
                None => String::new(),
            };
            format!(
                "p={p}; c={c}
if [ ! -e \"$p\" ] && [ ! -L \"$p\" ]; then
  {absent}
elif [ ! -L \"$p\" ] && [ -f \"$p\" ] && [ -f \"$c\" ] && cmp -s \"$p\" \"$c\"; then
  {undo}
{as_prior}else
  {other}
fi
",
                p = sh_quote(&path_str(path)),
                c = sh_quote(&path_str(&ledger.written_copy(change.id))),
                absent = sh_absent(change, "$p"),
                other = sh_not_ours(change, "$p"),
            )
        }
        HostChangeKind::Symlink { path, target } => format!(
            "p={p}; t={t}
if [ -L \"$p\" ] && [ \"$(readlink \"$p\")\" = \"$t\" ]; then
  rm -f -- \"$p\" && ok \"removed: $p\"
elif [ -e \"$p\" ] || [ -L \"$p\" ]; then
  {other}
else
  gone \"$p\"
fi
",
            p = sh_quote(&path_str(path)),
            t = sh_quote(&path_str(target)),
            other = sh_not_ours(change, "$p"),
        ),
        HostChangeKind::PathEntry {
            shell_rc: Some(rc),
            line,
        } => format!(
            "r={r}; l={l}
if [ -L \"$r\" ] || {{ [ -e \"$r\" ] && [ ! -f \"$r\" ]; }}; then
  leave \"$r\"
elif [ ! -f \"$r\" ]; then
  gone \"the PATH line in $r\"
elif ! tr -d '\\000' < \"$r\" | cmp -s - \"$r\"; then
  note \"left in place: $r holds NUL bytes, which this script does not rewrite; remove this line by hand: $l\"; left=$((left + 1))
elif grep -q -a -x -F -e \"$l\" -e \"$l$cr\" -- \"$r\"; then
  dropline \"$r\" \"$l\" && ok \"removed the PATH line from: $r\"
else
  gone \"the PATH line in $r\"
fi
",
            r = sh_quote(&path_str(rc)),
            l = sh_quote(line),
        ),
        HostChangeKind::PathEntry { shell_rc: None, .. } => {
            "ok \"nothing to revert: the PATH line was shown, not written\"\n".to_owned()
        }
        HostChangeKind::ServiceRegistered { manager, name } => {
            let n = sh_quote(name);
            match manager {
                ServiceManager::SystemdUser | ServiceManager::SystemdSystem => {
                    let user = if *manager == ServiceManager::SystemdUser {
                        " --user"
                    } else {
                        ""
                    };
                    format!(
                        "n={n}
if systemctl{user} cat -- \"$n\" >/dev/null 2>&1; then
  systemctl{user} disable --now -- \"$n\" && ok \"unregistered: $n\"
else
  gone \"$n\"
fi
"
                    )
                }
                ServiceManager::LaunchdAgent | ServiceManager::LaunchdDaemon => {
                    let domain = if *manager == ServiceManager::LaunchdAgent {
                        "gui/$(id -u)"
                    } else {
                        "system"
                    };
                    format!(
                        "n={n}
if launchctl print \"{domain}/$n\" >/dev/null 2>&1; then
  launchctl bootout \"{domain}/$n\" && ok \"unregistered: $n\"
else
  gone \"$n\"
fi
"
                    )
                }
                ServiceManager::Scm | ServiceManager::TaskScheduler => {
                    format!("note \"not on this OS: {manager:?} service \"{n}\n")
                }
            }
        }
        HostChangeKind::LingerEnabled { user } => {
            if change.prior == Some(PriorState::AlreadySet) {
                format!(
                    "ok \"kept: lingering was already on for \"{}\n",
                    sh_quote(user)
                )
            } else {
                format!(
                    "u={u}
if loginctl show-user -- \"$u\" -p Linger 2>/dev/null | grep -q '=yes$'; then
  loginctl disable-linger -- \"$u\" && ok \"disabled lingering for $u\"
else
  gone \"lingering for $u\"
fi
",
                    u = sh_quote(user)
                )
            }
        }
        HostChangeKind::FirewallRule { platform, name } => {
            let n = sh_quote(name);
            match platform {
                FirewallKind::Firewalld => format!(
                    "n={n}
if firewall-cmd --permanent --query-service=\"$n\" >/dev/null 2>&1; then
  firewall-cmd --permanent --remove-service=\"$n\" >/dev/null && firewall-cmd --reload >/dev/null && ok \"removed the firewalld service: $n\"
else
  gone \"firewalld service $n\"
fi
"
                ),
                FirewallKind::Ufw => format!(
                    "n={n}
if ufw status 2>/dev/null | grep -qF -e \"$n\"; then
  ufw delete allow \"$n\" >/dev/null && ok \"removed the ufw rule: $n\"
else
  gone \"ufw rule $n\"
fi
"
                ),
                FirewallKind::MacosApplicationFirewall => format!(
                    "n={n}; fw=/usr/libexec/ApplicationFirewall/socketfilterfw
if \"$fw\" --listapps 2>/dev/null | grep -qF -e \"$n\"; then
  \"$fw\" --remove \"$n\" >/dev/null && ok \"removed the firewall entry: $n\"
else
  gone \"firewall entry $n\"
fi
"
                ),
                FirewallKind::WindowsFirewall => {
                    format!("note \"not on this OS: Windows firewall rule \"{n}\n")
                }
            }
        }
        HostChangeKind::SystemUser { name } => format!(
            "n={n}
if id -u \"$n\" >/dev/null 2>&1; then
  case \"$(uname -s)\" in
    Darwin) dscl . -delete \"/Users/$n\" ;;
    *) userdel \"$n\" ;;
  esac && ok \"removed the system user: $n\"
else
  gone \"system user $n\"
fi
",
            n = sh_quote(name)
        ),
        HostChangeKind::BootTimeSetting { name } => match &change.prior {
            Some(PriorState::Value { value }) => format!(
                "s={s}; v={v}
sysctl -w \"$s=$v\" >/dev/null && ok \"restored $s to $v\"
",
                s = sh_quote(name),
                v = sh_quote(value)
            ),
            _ => format!(
                "ok \"nothing to restore: no earlier value of \"{}\" was recorded\"\n",
                sh_quote(name)
            ),
        },
        HostChangeKind::WindowsUserPath { entry } => {
            format!(
                "note \"not on this OS: user Path entry \"{}\n",
                sh_quote(entry)
            )
        }
        HostChangeKind::RegistryKey { hive, key } => format!(
            "note \"not on this OS: registry key \"{}\n",
            sh_quote(&format!("{hive}\\{key}"))
        ),
        HostChangeKind::TrustAnchor {
            store,
            location,
            label,
            certificate,
            ..
        } => sh_trust_anchor(change, *store, location, label, certificate),
    }
}

/// The `sh` revert of a trust-store entry. Each store is checked for exactly the recorded
/// certificate, compared as its base64 DER, before anything is removed.
fn sh_trust_anchor(
    change: &HostChange,
    store: TrustStore,
    location: &Path,
    label: &str,
    certificate: &str,
) -> String {
    let loc = sh_quote(&path_str(location));
    let n = sh_quote(label);
    let b = sh_quote(certificate);
    match store {
        TrustStore::LinuxUserNss => format!(
            "d={loc}; n={n}; b={b}
if ! command -v certutil >/dev/null 2>&1; then
  note \"left in place: certutil is not installed, so the CA $n cannot be removed from the NSS database $d\"; left=$((left + 1))
elif certutil -d \"sql:$d\" -L -n \"$n\" >/dev/null 2>&1; then
  if [ \"$(certutil -d \"sql:$d\" -L -n \"$n\" -a 2>/dev/null | sed '/^-----/d' | tr -d '\\r\\n')\" = \"$b\" ]; then
    certutil -d \"sql:$d\" -D -n \"$n\" && ok \"removed the CA $n from the NSS database $d\"
  else
    {other}
  fi
else
  {absent}
fi
",
            other = sh_not_ours(change, "the CA $n in the NSS database $d"),
            absent = sh_absent(change, "the CA $n in the NSS database $d"),
        ),
        TrustStore::MacosLoginKeychain => format!(
            "k={loc}; n={n}; b={b}
h=$(security find-certificate -a -Z -p \"$k\" 2>/dev/null | awk -v want=\"$b\" '/^SHA-1 hash:/ {{ h = $3 }} /^-----BEGIN CERTIFICATE-----/ {{ p = 1; s = \"\"; next }} /^-----END CERTIFICATE-----/ {{ p = 0; if (s == want) print h; next }} p {{ s = s $0 }}' | head -n 1)
if [ -n \"$h\" ]; then
  security delete-certificate -Z \"$h\" -t \"$k\" >/dev/null && ok \"removed the CA $n from the keychain $k\"
else
  {absent}
fi
",
            absent = sh_absent(change, "the CA in the keychain $k"),
        ),
        TrustStore::LinuxSystemCaCertificates | TrustStore::LinuxSystemCaTrust => {
            let tool = if store == TrustStore::LinuxSystemCaCertificates {
                "update-ca-certificates"
            } else {
                "update-ca-trust extract"
            };
            format!(
                "p={loc}; b={b}; s=
[ \"$(id -u)\" -eq 0 ] || s=sudo
if [ ! -e \"$p\" ] && [ ! -L \"$p\" ]; then
  {absent}
elif [ ! -L \"$p\" ] && [ -f \"$p\" ] && [ \"$(sed '/^-----/d' \"$p\" | tr -d '\\r\\n')\" = \"$b\" ]; then
  $s rm -f -- \"$p\" && $s {tool} >/dev/null && ok \"removed the system trust anchor: $p\"
else
  {other}
fi
",
                absent = sh_absent(change, "$p"),
                other = sh_not_ours(change, "$p"),
            )
        }
        TrustStore::WindowsCurrentUserRoot => {
            format!("note \"not on this OS: Windows Root store entry \"{n}\n")
        }
    }
}

/// The PowerShell script for `ledger`.
#[must_use]
pub fn render_ps1(ledger: &HostChangeLedger, by_version: &str) -> String {
    // The byte order mark makes Windows PowerShell 5.1 read the file as UTF-8. Without it, it reads
    // the ANSI code page, where the UTF-8 bytes of a character such as U+2011 decode to a curly
    // quote that ends a single-quoted string.
    let mut out = format!(
        "{bom}# Reverts the host changes Syntropy recorded in {ledger}, newest first, under the revert
# rules of RFC-0052 section 11.1: undo only what still matches what was recorded, restore the
# prior state, and touch nothing else. Generated by Syntropy {version}. It needs no Syntropy
# binary, and running it twice is safe.
$ErrorActionPreference = 'Continue'
$left = 0
$m = $null; $cw = $null; $cpr = $null
function Note([string]$t) {{ Write-Output \"syntropy-revert: $t\" }}
# Records that the current change is done, so a second run skips it, and deletes its copies.
function Mark {{
  New-Item -ItemType File -Path $script:m -Force | Out-Null
  Remove-Item -LiteralPath $script:cw, $script:cpr -Force -ErrorAction SilentlyContinue
}}
function Reverted([string]$t) {{ Note $t; Mark }}
function Gone([string]$t) {{ Note \"already gone: $t\"; Mark }}
function Leave([string]$p) {{ Note \"left in place, edited since Syntropy changed it: $p\"; $script:left++ }}
function Hold([string]$p) {{ Note \"left in place, a pending change that may have partly happened: $p; its saved copies are kept in the state directory\"; $script:left++ }}
# Writes $bytes to a new file beside $path and replaces $path with it, keeping $path's ACL and
# attributes ([IO.File]::Replace), so the file is never half written.
function Swap([string]$path, [byte[]]$bytes) {{
  $t = Join-Path ([IO.Path]::GetDirectoryName($path)) ('.syntropy-revert.' + [IO.Path]::GetRandomFileName())
  [IO.File]::WriteAllBytes($t, $bytes)
  [IO.File]::Replace($t, $path, [NullString]::Value)
}}
# Refuse a state directory or copies directory that is missing, not a directory, or a link (a
# junction or symbolic link). The copies directory is made only once the state directory passes.
function Private([string]$d) {{
  $i = Get-Item -LiteralPath $d -Force -ErrorAction SilentlyContinue
  if ($null -eq $i -or -not $i.PSIsContainer -or ($i.Attributes -band [IO.FileAttributes]::ReparsePoint)) {{
    Note \"refusing to run: $d is not a directory of its own\"; exit 1
  }}
}}
Private {state}
if (-not (Test-Path -LiteralPath {copies})) {{ New-Item -ItemType Directory -Path {copies} | Out-Null }}
Private {copies}
# One writer at a time: the run holds the ledger's lock file open with no sharing, so the binary,
# which locks the same file, waits for it or reports that this host is busy.
$lockPath = {lock}
try {{ $lockFile = [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None) }}
catch {{ Note \"refusing to run: another Syntropy command is changing this host ($lockPath is in use)\"; exit 1 }}
",
        ledger = comment(&path_str(&ledger.state_dir().join(super::LEDGER_FILE))),
        version = comment(by_version),
        bom = PS1_BOM,
        state = ps_quote(&path_str(ledger.state_dir())),
        copies = ps_quote(&path_str(&ledger.state_dir().join(super::COPIES_DIR))),
        lock = ps_quote(&path_str(&ledger.state_dir().join(super::LOCK_FILE))),
    );
    for change in open_changes(ledger) {
        out.push_str(&format!(
            "\n# change {} ({:?}): {}\n$m = {}; $cw = {}; $cpr = {}\nif (Test-Path -LiteralPath $m) {{ Note \"already reverted: change {}\" }} else {{\n",
            change.id,
            change.state,
            comment(&describe(&change.kind)),
            ps_quote(&path_str(&ledger.revert_marker(change.id))),
            ps_quote(&path_str(&ledger.written_copy(change.id))),
            ps_quote(&path_str(&ledger.prior_copy(change.id))),
            change.id,
        ));
        out.push_str(&ps_block(ledger, change));
        out.push_str("}\n");
    }
    out.push_str(
        "\nif ($left -gt 0) { Note \"$left change(s) left in place; see above\" }\n$lockFile.Dispose()\n",
    );
    out
}

fn ps_block(ledger: &HostChangeLedger, change: &HostChange) -> String {
    match &change.kind {
        HostChangeKind::FileWritten { path } => {
            let undo = match own_prior(ledger, change) {
                Some(from) => format!(
                    "Swap $p ([IO.File]::ReadAllBytes({})); Reverted \"restored the earlier content: $p\"",
                    ps_quote(&path_str(&from))
                ),
                None => "Remove-Item -LiteralPath $p -Force; Reverted \"removed: $p\"".to_owned(),
            };
            let as_prior = match own_prior(ledger, change) {
                Some(_) => format!(
                    "elseif ($plain -and (Test-Path -LiteralPath $cpr) -and ((Get-FileHash -LiteralPath $p).Hash -eq (Get-FileHash -LiteralPath $cpr).Hash)) {{ {} }}\n",
                    if change.state == ChangeState::Pending {
                        "Reverted \"never made: $p\""
                    } else {
                        "Reverted \"already back to its earlier content: $p\""
                    }
                ),
                None => String::new(),
            };
            format!(
                "$p = {p}; $c = {c}
$i = Get-Item -LiteralPath $p -Force -ErrorAction SilentlyContinue
$plain = ($null -ne $i) -and -not $i.LinkType -and -not $i.PSIsContainer
if ($null -eq $i) {{ {absent} }}
elseif ($plain -and (Test-Path -LiteralPath $c) -and ((Get-FileHash -LiteralPath $p).Hash -eq (Get-FileHash -LiteralPath $c).Hash)) {{ {undo} }}
{as_prior}else {{ {other} }}
",
                p = ps_quote(&path_str(path)),
                c = ps_quote(&path_str(&ledger.written_copy(change.id))),
                absent = ps_absent(change, "$p"),
                other = ps_not_ours(change, "$p"),
            )
        }
        HostChangeKind::Symlink { path, target } => format!(
            "$p = {p}; $t = {t}
$i = Get-Item -LiteralPath $p -Force -ErrorAction SilentlyContinue
if ($null -eq $i) {{ Gone \"$p\" }}
elseif ($i.LinkType -eq 'SymbolicLink' -and @($i.Target)[0] -eq $t) {{ Remove-Item -LiteralPath $p -Force; Reverted \"removed: $p\" }}
else {{ {other} }}
",
            p = ps_quote(&path_str(path)),
            t = ps_quote(&path_str(target)),
            other = ps_not_ours(change, "$p"),
        ),
        HostChangeKind::PathEntry {
            shell_rc: Some(rc),
            line,
        } => format!(
            "$r = {r}; $l = {l}
$i = Get-Item -LiteralPath $r -Force -ErrorAction SilentlyContinue
if ($null -eq $i) {{ Gone \"the PATH line in $r\" }}
elseif ($i.LinkType -or $i.PSIsContainer) {{ Leave $r }}
else {{
  # Bytes, not text: Latin-1 maps each byte to one character and back, so every line Syntropy did
  # not write is written back byte for byte, whatever its encoding, with its line ending.
  $b = [Text.Encoding]::GetEncoding(28591)
  $want = $b.GetString([Text.Encoding]::UTF8.GetBytes($l))
  $segs = [regex]::Split($b.GetString([IO.File]::ReadAllBytes($r)), '(?<=\\n)')
  $kept = @($segs | Where-Object {{ ($_ -replace '\\r?\\n?\\z', '') -cne $want }})
  if ($kept.Count -eq $segs.Count) {{ Gone \"the PATH line in $r\" }}
  else {{ Swap $r ($b.GetBytes(-join $kept)); Reverted \"removed the PATH line from: $r\" }}
}}
",
            r = ps_quote(&path_str(rc)),
            l = ps_quote(line),
        ),
        HostChangeKind::PathEntry { shell_rc: None, .. } => {
            "Reverted \"nothing to revert: the PATH line was shown, not written\"\n".to_owned()
        }
        HostChangeKind::WindowsUserPath { entry } => format!(
            "$e = {e}
$k = Get-Item -Path 'HKCU:\\Environment'
$cur = $k.GetValue('Path', $null, 'DoNotExpandEnvironmentNames')
$parts = @(); if ($null -ne $cur) {{ $parts = @($cur -split ';' | Where-Object {{ $_ -ne '' }}) }}
if ($parts -contains $e) {{
  $kind = $k.GetValueKind('Path')
  New-ItemProperty -Path 'HKCU:\\Environment' -Name 'Path' -Value ((@($parts | Where-Object {{ $_ -ne $e }})) -join ';') -PropertyType $kind -Force | Out-Null
  Reverted \"removed from the user Path: $e\"
}} else {{ Gone \"user Path entry $e\" }}
",
            e = ps_quote(entry),
        ),
        HostChangeKind::ServiceRegistered { manager, name } => {
            let n = ps_quote(name);
            match manager {
                ServiceManager::Scm => format!(
                    "$n = {n}
if (Get-Service -Name $n -ErrorAction SilentlyContinue) {{
  Stop-Service -Name $n -Force -ErrorAction SilentlyContinue
  & sc.exe delete $n | Out-Null
  Reverted \"unregistered: $n\"
}} else {{ Gone \"$n\" }}
"
                ),
                ServiceManager::TaskScheduler => format!(
                    "$n = {n}
if (Get-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue) {{
  Unregister-ScheduledTask -TaskName $n -Confirm:$false
  Reverted \"unregistered: $n\"
}} else {{ Gone \"$n\" }}
"
                ),
                _ => format!("Note (\"not on this OS: {manager:?} service \" + {n})\n"),
            }
        }
        HostChangeKind::FirewallRule { platform, name } => {
            let n = ps_quote(name);
            match platform {
                FirewallKind::WindowsFirewall => format!(
                    "$n = {n}
if (Get-NetFirewallRule -DisplayName $n -ErrorAction SilentlyContinue) {{
  Remove-NetFirewallRule -DisplayName $n
  Reverted \"removed the firewall rule: $n\"
}} else {{ Gone \"firewall rule $n\" }}
"
                ),
                _ => format!("Note (\"not on this OS: {platform:?} rule \" + {n})\n"),
            }
        }
        HostChangeKind::RegistryKey { hive, key } => format!(
            "$k = {k}
if (Test-Path -LiteralPath $k) {{ Remove-Item -LiteralPath $k -Recurse -Force; Reverted \"removed the registry key: $k\" }}
else {{ Gone \"registry key $k\" }}
",
            k = ps_quote(&format!("Registry::{hive}\\{key}")),
        ),
        HostChangeKind::SystemUser { name } => format!(
            "$n = {n}
if (Get-LocalUser -Name $n -ErrorAction SilentlyContinue) {{ Remove-LocalUser -Name $n; Reverted \"removed the system user: $n\" }}
else {{ Gone \"system user $n\" }}
",
            n = ps_quote(name),
        ),
        HostChangeKind::LingerEnabled { user } => format!(
            "Note (\"not on this OS: lingering for \" + {})\n",
            ps_quote(user)
        ),
        HostChangeKind::BootTimeSetting { name } => format!(
            "Note (\"not on this OS: boot-time setting \" + {})\n",
            ps_quote(name)
        ),
        HostChangeKind::TrustAnchor {
            store: TrustStore::WindowsCurrentUserRoot,
            label,
            certificate,
            ..
        } => format!(
            "$b = {b}; $n = {n}
$st = New-Object Security.Cryptography.X509Certificates.X509Store('Root', 'CurrentUser')
try {{
  $st.Open('ReadWrite')
  $hit = @($st.Certificates | Where-Object {{ [Convert]::ToBase64String($_.RawData) -ceq $b }})
  if ($hit.Count -gt 0) {{
    foreach ($c in $hit) {{ $st.Remove($c) }}
    Reverted \"removed the CA $n from the current user's Root store\"
  }} else {{ {absent} }}
}} catch {{ Note \"left in place, the Root store refused the change: $n\"; $script:left++ }}
finally {{ $st.Close() }}
",
            b = ps_quote(certificate),
            n = ps_quote(label),
            absent = ps_absent(change, "the CA $n in the current user's Root store"),
        ),
        HostChangeKind::TrustAnchor { store, label, .. } => format!(
            "Note (\"not on this OS: {store:?} entry \" + {})\n",
            ps_quote(label)
        ),
    }
}

/// Writes this OS's revert script to the state directory, atomically, and returns its path: the
/// `sh` script, owner-only and executable, on Unix, and the PowerShell script on Windows.
///
/// # Errors
///
/// [`LedgerError::Io`] if the script cannot be written, and [`LedgerError::Busy`] if another
/// writer holds the ledger's lock.
pub fn write_revert_script(
    ledger: &HostChangeLedger,
    by_version: &str,
) -> Result<PathBuf, LedgerError> {
    // Rendered from the ledger as saved, under its lock, so no writer changes it in between.
    let _lock = ledger.lock()?;
    let saved = HostChangeLedger::open(ledger.state_dir())?;
    #[cfg(windows)]
    let (name, text) = (PS1_FILE, render_ps1(&saved, by_version));
    #[cfg(not(windows))]
    let (name, text) = (SH_FILE, render_sh(&saved, by_version));
    let path = ledger.state_dir().join(name);
    replace_atomically(&path, text.as_bytes())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).map_err(
            |source| LedgerError::Io {
                path: path.clone(),
                source,
            },
        )?;
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A new state directory per call. Tests run in parallel and the clock can repeat between
    /// two calls (Windows counts in 100 ns), so a counter keeps the names apart, and
    /// `create_dir` fails rather than share a directory that already exists.
    fn temp_state() -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!(
            "syntropy-script-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        // A directory left by an earlier run with the same process ID.
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir(&dir).unwrap();
        dir
    }

    fn ledger_with(kinds: Vec<(HostChangeKind, Option<PriorState>)>) -> HostChangeLedger {
        let state = temp_state();
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        for (kind, prior) in kinds {
            let id = ledger.begin(kind, prior, None, 0, "v").unwrap();
            ledger.applied(id).unwrap();
        }
        ledger
    }

    #[test]
    fn quoting_survives_single_quotes() {
        assert_eq!(sh_quote("it's"), r"'it'\''s'");
        assert_eq!(ps_quote("it's"), "'it''s'");
    }

    /// PowerShell ends a single-quoted string at any of five quote characters, so all five are
    /// doubled. A NUL is dropped in both shells.
    #[test]
    fn powershell_quoting_doubles_every_single_quote_character() {
        for q in PS_SINGLE_QUOTES {
            let value = format!("a{q}; Remove-Item C:\\ -Recurse; {q}b");
            let quoted = ps_quote(&value);
            assert_eq!(
                quoted,
                format!("'a{q}{q}; Remove-Item C:\\ -Recurse; {q}{q}b'")
            );
        }
        assert_eq!(ps_quote("a\0b"), "'ab'");
        assert_eq!(sh_quote("a\0b"), "'ab'");
    }

    /// A value in a comment cannot end the comment line in either shell.
    #[test]
    fn comments_lose_line_breaks_and_control_characters() {
        let hostile = "x\nrm -rf ~\r\u{85}\u{2028}\u{2029}y";
        let clean = comment(hostile);
        assert!(
            !clean
                .chars()
                .any(|c| c.is_control() || matches!(c, '\u{2028}' | '\u{2029}'))
        );
        // `begin` refuses a name with a line break, so the entry goes in as a tampered or older
        // ledger could hold it: the renderer must still keep it out of the script's code.
        let mut ledger = ledger_with(vec![]);
        ledger.file.next_id = 2;
        ledger.file.changes.push(HostChange {
            id: 1,
            kind: HostChangeKind::SystemUser {
                name: hostile.into(),
            },
            state: ChangeState::Applied,
            at: 0,
            by_version: "v".into(),
            fingerprint: None,
            prior: None,
        });
        // The value appears twice: once in the comment, with its line breaks gone, and once as a
        // single-quoted literal, where line breaks are data. Nowhere else.
        for (script, literal) in [
            (render_sh(&ledger, "v"), sh_quote(hostile)),
            (render_ps1(&ledger, "v"), ps_quote(hostile)),
        ] {
            let comment_line = script
                .lines()
                .find(|l| l.starts_with("# change 1"))
                .unwrap();
            assert!(comment_line.contains(&clean), "{comment_line}");
            assert_eq!(script.matches(literal.as_str()).count(), 1, "{script}");
            assert_eq!(script.matches("rm -rf ~").count(), 2, "{script}");
        }
        // The version string is a comment too.
        for script in [render_sh(&ledger, hostile), render_ps1(&ledger, hostile)] {
            assert_eq!(script.matches("rm -rf ~").count(), 3, "{script}");
            assert!(script.contains(&format!("Generated by Syntropy {clean}.")));
        }
    }

    /// Windows PowerShell 5.1 reads a script without a byte order mark in the ANSI code page.
    #[test]
    fn the_powershell_script_starts_with_a_byte_order_mark() {
        let ledger = ledger_with(vec![]);
        assert!(render_ps1(&ledger, "v").starts_with('\u{feff}'));
        assert!(render_sh(&ledger, "v").starts_with("#!/bin/sh\n"));
    }

    /// Only the ledger's own prior copy is restored from; another recorded path gives a removal
    /// guarded by the usual check, never a copy from that path.
    #[test]
    fn a_foreign_prior_copy_is_not_rendered() {
        let ledger = ledger_with(vec![(
            HostChangeKind::FileWritten {
                path: "/u/a.conf".into(),
            },
            Some(PriorState::SavedAt {
                path: "/etc/shadow".into(),
            }),
        )]);
        for script in [render_sh(&ledger, "v"), render_ps1(&ledger, "v")] {
            assert!(!script.contains("/etc/shadow"), "{script}");
        }
    }

    #[test]
    fn the_sh_script_refuses_state_that_is_not_private() {
        let ledger = ledger_with(vec![]);
        let sh = render_sh(&ledger, "v");
        assert!(
            sh.contains("refusing to run: $1 is not private to this user"),
            "{sh}"
        );
        assert!(sh.contains("[ ! -O \"$1\" ]"), "{sh}");
        // The state directory is checked before anything is made in it, and nothing is chmod'ed.
        let state_check = sh.find("private '").unwrap();
        assert!(sh.find("mkdir -m 700").unwrap() > state_check, "{sh}");
        assert!(!sh.contains("chmod"), "{sh}");
        let ps1 = render_ps1(&ledger, "v");
        assert!(
            ps1.find("Private '").unwrap() < ps1.find("New-Item -ItemType Directory").unwrap(),
            "{ps1}"
        );
    }

    /// Runs the generated sh script for real, under a UTF-8 locale and a minimal `PATH`, and
    /// returns its output.
    #[cfg(unix)]
    fn run_sh(ledger: &HostChangeLedger) -> String {
        let script = write_revert_script(ledger, "v").unwrap();
        let out = std::process::Command::new("sh")
            .arg(&script)
            .env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
            .env("LC_ALL", "C.UTF-8")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        assert!(
            out.status.success(),
            "{stdout}\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        stdout
    }

    /// Run for real: a pending write whose file is half written is held, with the copy of what it
    /// replaced, as the binary holds it.
    #[cfg(unix)]
    #[test]
    fn the_sh_script_holds_a_half_written_pending_file() {
        use std::fs;
        let state = temp_state();
        let path = state.join("a.conf");
        fs::write(&path, b"ORIGINAL\n").unwrap();
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        let id = ledger.write_file(&path, b"NEW content\n", 0, "v").unwrap();
        ledger
            .file
            .changes
            .iter_mut()
            .for_each(|c| c.state = ChangeState::Pending);
        ledger.save().unwrap();
        fs::write(&path, b"NEW co").unwrap();
        let out = run_sh(&ledger);
        assert!(
            out.contains("a pending change that may have partly happened"),
            "{out}"
        );
        assert!(out.contains("1 change(s) left in place"), "{out}");
        assert_eq!(fs::read(ledger.prior_copy(id)).unwrap(), b"ORIGINAL\n");
        assert!(!ledger.revert_marker(id).exists());
        assert_eq!(fs::read(&path).unwrap(), b"NEW co");

        // Back to the original: never made, and only then are the copies deleted.
        fs::write(&path, b"ORIGINAL\n").unwrap();
        let out = run_sh(&ledger);
        assert!(out.contains("never made"), "{out}");
        assert!(!ledger.prior_copy(id).exists());
        assert!(ledger.revert_marker(id).exists());
    }

    /// Run for real under `LC_ALL=C.UTF-8`: the start-up line goes, and every other byte stays,
    /// whatever the file's encoding and line endings. A file with a NUL byte is left, with the line
    /// to remove by hand, because the tools the script uses do not all carry NUL bytes.
    #[cfg(unix)]
    #[test]
    fn the_sh_script_removes_start_up_lines_as_bytes() {
        use std::fs;
        let line = "export PATH=\"$HOME/.local/bin:$PATH\"";
        let state = temp_state();
        let latin = state.join("latin1-profile");
        let crlf = state.join("crlf-profile");
        let last = state.join("last-line-profile");
        let nul = state.join("nul-profile");
        fs::write(
            &latin,
            b"# caf\xe9 alias\nexport PATH=\"$HOME/.local/bin:$PATH\"\nexport B=\xe9t\xe9\n",
        )
        .unwrap();
        fs::write(
            &crlf,
            b"a=1\r\nexport PATH=\"$HOME/.local/bin:$PATH\"\r\nb=2\r\n",
        )
        .unwrap();
        fs::write(&last, b"a=1\nexport PATH=\"$HOME/.local/bin:$PATH\"").unwrap();
        let with_nul: &[u8] = b"a=1\nexport PATH=\"$HOME/.local/bin:$PATH\"\nb\0x\nc=3\n";
        fs::write(&nul, with_nul).unwrap();
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        for rc in [&latin, &crlf, &last, &nul] {
            let kind = HostChangeKind::PathEntry {
                shell_rc: Some(rc.clone()),
                line: line.into(),
            };
            let id = ledger.begin(kind, None, None, 0, "v").unwrap();
            ledger.applied(id).unwrap();
        }
        let out = run_sh(&ledger);
        assert_eq!(
            fs::read(&latin).unwrap(),
            b"# caf\xe9 alias\nexport B=\xe9t\xe9\n",
            "{out}"
        );
        assert_eq!(fs::read(&crlf).unwrap(), b"a=1\r\nb=2\r\n", "{out}");
        assert_eq!(fs::read(&last).unwrap(), b"a=1\n", "{out}");
        assert_eq!(fs::read(&nul).unwrap(), with_nul, "{out}");
        assert!(out.contains("holds NUL bytes"), "{out}");
        assert!(out.contains("1 change(s) left in place"), "{out}");
    }

    /// PowerShell holds a pending change with a saved copy too, and rewrites start-up files as
    /// bytes.
    #[test]
    fn the_powershell_script_holds_and_works_on_bytes() {
        let ledger = ledger_with(vec![(
            HostChangeKind::FileWritten {
                path: "/u/a.conf".into(),
            },
            None,
        )]);
        let pending_with_copy = HostChange {
            prior: Some(PriorState::SavedAt {
                path: ledger.prior_copy(1),
            }),
            state: ChangeState::Pending,
            ..ledger.changes()[0].clone()
        };
        assert_eq!(ps_absent(&pending_with_copy, "$p"), "Hold \"$p\"");
        assert_eq!(ps_not_ours(&pending_with_copy, "$p"), "Hold \"$p\"");
        assert_eq!(sh_absent(&pending_with_copy, "$p"), "hold \"$p\"");
        let ps1 = render_ps1(&ledger, "v");
        assert!(ps1.contains("function Hold"), "{ps1}");
        assert!(
            ps1.contains("[IO.File]::Replace($t, $path, [NullString]::Value)"),
            "{ps1}"
        );
        assert!(!ps1.contains("Set-Content"), "{ps1}");
    }

    /// Both scripts take the ledger's lock for the whole run.
    #[test]
    fn both_scripts_take_the_ledger_lock() {
        let ledger = ledger_with(vec![]);
        let sh = render_sh(&ledger, "v");
        assert!(sh.contains("flock -n 9"), "{sh}");
        assert!(sh.contains("host-changes.lock"), "{sh}");
        let ps1 = render_ps1(&ledger, "v");
        assert!(ps1.contains("[IO.FileShare]::None"), "{ps1}");
        assert!(ps1.contains("host-changes.lock"), "{ps1}");
    }

    /// Run for real: while the binary holds the ledger's lock, the sh script refuses to run, where
    /// flock(1) exists.
    #[cfg(unix)]
    #[test]
    fn the_sh_script_refuses_while_the_ledger_is_locked() {
        let has_flock = std::process::Command::new("sh")
            .args(["-c", "command -v flock"])
            .env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
            .output()
            .is_ok_and(|o| o.status.success());
        if !has_flock {
            eprintln!("flock(1) is not installed here: the script runs unlocked, and says so");
            return;
        }
        let ledger = ledger_with(vec![]);
        let script = write_revert_script(&ledger, "v").unwrap();
        let held = ledger.lock().unwrap();
        let out = std::process::Command::new("sh")
            .arg(&script)
            .env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(!out.status.success(), "{stdout}");
        assert!(
            stdout.contains("another Syntropy command is changing this host"),
            "{stdout}"
        );
        drop(held);
        let out = std::process::Command::new("sh")
            .arg(&script)
            .env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stdout)
        );
    }

    /// Run for real: a state directory others can write, and a link in place of the copies
    /// directory, both stop the script before it changes anything, and the link's target keeps
    /// its mode.
    #[cfg(unix)]
    #[test]
    fn the_sh_script_stops_on_state_that_is_not_private() {
        use std::fs;
        use std::os::unix::fs::PermissionsExt;
        let run = |ledger: &HostChangeLedger| {
            let script = write_revert_script(ledger, "v").unwrap();
            std::process::Command::new("sh")
                .arg(&script)
                .env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
                .output()
                .unwrap()
        };
        let mode = |p: &Path| fs::symlink_metadata(p).unwrap().permissions().mode() & 0o777;

        let ledger = ledger_with(vec![]);
        fs::set_permissions(ledger.state_dir(), fs::Permissions::from_mode(0o777)).unwrap();
        let out = run(&ledger);
        assert!(!out.status.success());
        assert!(String::from_utf8_lossy(&out.stdout).contains("is not private to this user"));
        assert!(!ledger.state_dir().join(super::super::COPIES_DIR).exists());

        let ledger = ledger_with(vec![]);
        let elsewhere = ledger.state_dir().join("elsewhere");
        fs::create_dir(&elsewhere).unwrap();
        fs::set_permissions(&elsewhere, fs::Permissions::from_mode(0o755)).unwrap();
        std::os::unix::fs::symlink(
            &elsewhere,
            ledger.state_dir().join(super::super::COPIES_DIR),
        )
        .unwrap();
        let out = run(&ledger);
        assert!(!out.status.success());
        assert!(String::from_utf8_lossy(&out.stdout).contains("is not private to this user"));
        assert_eq!(mode(&elsewhere), 0o755);
    }

    /// One rendering check per change kind, in both scripts: the command that reverts it, and the
    /// check that guards it.
    #[test]
    fn every_kind_renders_its_revert_in_both_scripts() {
        let cases: Vec<(HostChangeKind, Option<PriorState>, &str, &str)> = vec![
            (
                HostChangeKind::FileWritten {
                    path: "/u/a.service".into(),
                },
                None,
                "cmp -s \"$p\" \"$c\"",
                "Get-FileHash -LiteralPath $p",
            ),
            (
                HostChangeKind::Symlink {
                    path: "/u/bin/syntropy".into(),
                    target: "/u/v/1/syntropy".into(),
                },
                None,
                "[ \"$(readlink \"$p\")\" = \"$t\" ]",
                "$i.LinkType -eq 'SymbolicLink'",
            ),
            (
                HostChangeKind::PathEntry {
                    shell_rc: Some("/u/.profile".into()),
                    line: "export PATH=\"$HOME/.local/bin:$PATH\"".into(),
                },
                None,
                "dropline \"$r\" \"$l\"",
                "[IO.File]::ReadAllBytes($r)",
            ),
            (
                HostChangeKind::WindowsUserPath {
                    entry: r"C:\S\bin".into(),
                },
                None,
                "not on this OS: user Path entry",
                "-PropertyType $kind",
            ),
            (
                HostChangeKind::ServiceRegistered {
                    manager: ServiceManager::SystemdUser,
                    name: "syntropy.service".into(),
                },
                None,
                "systemctl --user disable --now -- \"$n\"",
                "not on this OS: SystemdUser service",
            ),
            (
                HostChangeKind::LingerEnabled { user: "u".into() },
                None,
                "loginctl disable-linger -- \"$u\"",
                "not on this OS: lingering for",
            ),
            (
                HostChangeKind::FirewallRule {
                    platform: FirewallKind::WindowsFirewall,
                    name: "Syntropy".into(),
                },
                None,
                "not on this OS: Windows firewall rule",
                "Remove-NetFirewallRule -DisplayName $n",
            ),
            (
                HostChangeKind::RegistryKey {
                    hive: "HKEY_CURRENT_USER".into(),
                    key: r"Software\Syntropy".into(),
                },
                None,
                "not on this OS: registry key",
                r"'Registry::HKEY_CURRENT_USER\Software\Syntropy'",
            ),
            (
                HostChangeKind::SystemUser {
                    name: "syntropy".into(),
                },
                None,
                "userdel \"$n\"",
                "Remove-LocalUser -Name $n",
            ),
            (
                HostChangeKind::BootTimeSetting {
                    name: "iogpu.wired_limit_mb".into(),
                },
                Some(PriorState::Value { value: "0".into() }),
                "sysctl -w \"$s=$v\"",
                "not on this OS: boot-time setting",
            ),
            (
                trust_anchor(TrustStore::LinuxUserNss, "/u/.pki/nssdb"),
                None,
                "certutil -d \"sql:$d\" -D -n \"$n\"",
                "not on this OS: LinuxUserNss entry",
            ),
            (
                trust_anchor(
                    TrustStore::MacosLoginKeychain,
                    "/u/Library/Keychains/login.keychain-db",
                ),
                None,
                "security delete-certificate -Z \"$h\" -t \"$k\"",
                "not on this OS: MacosLoginKeychain entry",
            ),
            (
                trust_anchor(
                    TrustStore::LinuxSystemCaCertificates,
                    "/usr/local/share/ca-certificates/syntropy-box.local-0123456789abcdef.crt",
                ),
                None,
                "$s rm -f -- \"$p\" && $s update-ca-certificates",
                "not on this OS: LinuxSystemCaCertificates entry",
            ),
            (
                trust_anchor(
                    TrustStore::LinuxSystemCaTrust,
                    "/etc/pki/ca-trust/source/anchors/syntropy-box.local-0123456789abcdef.pem",
                ),
                None,
                "$s rm -f -- \"$p\" && $s update-ca-trust extract",
                "not on this OS: LinuxSystemCaTrust entry",
            ),
            (
                trust_anchor(
                    TrustStore::WindowsCurrentUserRoot,
                    r"Cert:\CurrentUser\Root",
                ),
                None,
                "not on this OS: Windows Root store entry",
                "[Convert]::ToBase64String($_.RawData) -ceq $b",
            ),
        ];
        for (kind, prior, in_sh, in_ps1) in cases {
            let ledger = ledger_with(vec![(kind.clone(), prior)]);
            let sh = render_sh(&ledger, "v");
            let ps1 = render_ps1(&ledger, "v");
            assert!(sh.contains(in_sh), "{kind:?}: sh lacks {in_sh}\n{sh}");
            assert!(ps1.contains(in_ps1), "{kind:?}: ps1 lacks {in_ps1}\n{ps1}");
        }
    }

    fn trust_anchor(store: TrustStore, location: &str) -> HostChangeKind {
        HostChangeKind::TrustAnchor {
            store,
            location: location.into(),
            label: "Syntropy CA box.local 0123456789abcdef".into(),
            node: "box.local".into(),
            certificate: "MIIBhjCCASygAwIBAgIU+/A=".into(),
        }
    }

    /// A trust-store entry is removed only where the store holds exactly the recorded
    /// certificate: both scripts compare it as base64 DER before they remove anything, and the
    /// system store goes through sudo when the script is not root.
    #[test]
    fn trust_anchors_are_matched_by_their_certificate() {
        let ledger = ledger_with(vec![
            (
                trust_anchor(TrustStore::LinuxUserNss, "/u/.pki/nssdb"),
                None,
            ),
            (
                trust_anchor(
                    TrustStore::LinuxSystemCaCertificates,
                    "/usr/local/share/ca-certificates/syntropy-box.local-0123456789abcdef.crt",
                ),
                None,
            ),
            (
                trust_anchor(
                    TrustStore::WindowsCurrentUserRoot,
                    r"Cert:\CurrentUser\Root",
                ),
                None,
            ),
        ]);
        let sh = render_sh(&ledger, "v");
        assert!(sh.contains("b='MIIBhjCCASygAwIBAgIU+/A='"), "{sh}");
        assert!(sh.contains("tr -d '\\r\\n')\" = \"$b\""), "{sh}");
        assert!(sh.contains("[ \"$(id -u)\" -eq 0 ] || s=sudo"), "{sh}");
        let ps1 = render_ps1(&ledger, "v");
        assert!(ps1.contains("$b = 'MIIBhjCCASygAwIBAgIU+/A='"), "{ps1}");
        assert!(ps1.contains("$st.Remove($c)"), "{ps1}");
    }

    #[test]
    fn the_other_service_managers_and_firewalls_render() {
        for (manager, in_sh, in_ps1) in [
            (
                ServiceManager::SystemdSystem,
                "systemctl disable --now",
                "not on this OS",
            ),
            (
                ServiceManager::LaunchdAgent,
                "launchctl bootout \"gui/$(id -u)/$n\"",
                "not on this OS",
            ),
            (
                ServiceManager::LaunchdDaemon,
                "launchctl bootout \"system/$n\"",
                "not on this OS",
            ),
            (ServiceManager::Scm, "not on this OS", "& sc.exe delete $n"),
            (
                ServiceManager::TaskScheduler,
                "not on this OS",
                "Unregister-ScheduledTask -TaskName $n",
            ),
        ] {
            let ledger = ledger_with(vec![(
                HostChangeKind::ServiceRegistered {
                    manager,
                    name: "syntropy".into(),
                },
                None,
            )]);
            assert!(render_sh(&ledger, "v").contains(in_sh), "{manager:?}");
            assert!(render_ps1(&ledger, "v").contains(in_ps1), "{manager:?}");
        }
        for (platform, in_sh) in [
            (
                FirewallKind::Firewalld,
                "firewall-cmd --permanent --remove-service=\"$n\"",
            ),
            (FirewallKind::Ufw, "ufw delete allow \"$n\""),
            (
                FirewallKind::MacosApplicationFirewall,
                "\"$fw\" --remove \"$n\"",
            ),
        ] {
            let ledger = ledger_with(vec![(
                HostChangeKind::FirewallRule {
                    platform,
                    name: "syntropy".into(),
                },
                None,
            )]);
            assert!(render_sh(&ledger, "v").contains(in_sh), "{platform:?}");
        }
    }

    #[test]
    fn prior_state_shows_in_the_scripts() {
        let ledger = ledger_with(vec![(
            HostChangeKind::LingerEnabled { user: "u".into() },
            Some(PriorState::AlreadySet),
        )]);
        let sh = render_sh(&ledger, "v");
        assert!(sh.contains("kept: lingering was already on"), "{sh}");
        assert!(!sh.contains("disable-linger"), "{sh}");
    }

    /// Changes are reverted newest first, and closed ones are left out.
    #[test]
    fn newest_first_and_only_open_changes() {
        let state = temp_state();
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        for name in ["a", "b", "c"] {
            let id = ledger
                .begin(
                    HostChangeKind::SystemUser { name: name.into() },
                    None,
                    None,
                    0,
                    "v",
                )
                .unwrap();
            ledger.applied(id).unwrap();
        }
        ledger.reverted(2).unwrap();
        let sh = render_sh(&ledger, "v");
        let three = sh.find("# change 3").unwrap();
        let one = sh.find("# change 1").unwrap();
        assert!(three < one);
        assert!(!sh.contains("# change 2"));
    }
}
