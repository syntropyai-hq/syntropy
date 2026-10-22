// SPDX-License-Identifier: MIT OR Apache-2.0

//! The validation rules for the bind keys (RFC-0054 §8, RFC-0050 §11.1), run by
//! `syntropy_config::load_and_validate` after its built-in rules:
//!
//! - phase 4, [`API_RULES`]' `api_tls_none_allowed`: `api.tls = "none"` is rejected on a public
//!   bind and on a wildcard, which listens on every interface, a public one included; on a
//!   specific LAN address it needs `api.allow_plaintext_lan = true`. Loopback binds serve plain
//!   HTTP whatever `api.tls` says, so they are not checked;
//! - phase 5, `api_tls_files_readable`: both files of a `{ cert, key }` pair exist and can be
//!   read. A table with only one of them never gets this far: it is the wrong shape, rejected when
//!   the value is read (phase 3).
//!
//! Whether an API key exists is not configuration (keys live in `api-keys.toml`, PROD-E5-T01), so
//! the daemon checks it at start through [`crate::bind::plan_binds`].

use std::path::Path;

use syntropy_config::{Code, Findings, Origin, Phase, Problem, Rule, RuleInput};

use crate::bind::{BindClass, classify, is_wildcard};
use crate::config::{ApiSection, TlsSource};

/// The rules a binary that serves the API adds to `load_and_validate`.
pub const API_RULES: &[Rule] = &[
    Rule {
        name: "api_tls_none_allowed",
        phase: Phase::CrossField,
        check: tls_none_allowed,
    },
    Rule {
        name: "api_tls_files_readable",
        phase: Phase::Environment,
        check: tls_files_readable,
    },
];

/// The `[api]` section, and where `api.tls` was set. `None` when the binary registers no `[api]`
/// section, or when a value has the wrong type, which phase 3 has already reported.
fn section(input: &RuleInput<'_>) -> Option<(ApiSection, Option<Origin>)> {
    let api = input.config.section::<ApiSection>().ok()?;
    let origin = input.config.get("api.tls").map(|r| r.origin.clone());
    Some((api, origin))
}

fn problem(code: Code, at: Option<Origin>, key: &str, message: String, help: &str) -> Problem {
    Problem {
        code: Some(code),
        at,
        column: None,
        width: None,
        key: Some(key.into()),
        message,
        help: Some(help.into()),
    }
}

/// Phase 4: `api.tls = "none"` only where plain HTTP is allowed (RFC-0050 §11.1).
fn tls_none_allowed(input: &RuleInput<'_>, findings: &mut Findings) {
    let Some((api, origin)) = section(input) else {
        return;
    };
    if api.tls != TlsSource::None {
        return;
    }
    for addr in &api.bind {
        let refused = match classify(addr.ip()) {
            BindClass::Loopback => continue,
            BindClass::Lan if is_wildcard(addr.ip()) => problem(
                Code::InsecureListener,
                origin.clone(),
                "api.tls",
                format!(
                    "`api.tls = \"none\"` would serve plain HTTP on the wildcard bind `{addr}`, \
                     which listens on every interface, a public one included; a wildcard is \
                     always served with TLS (RFC-0050 §11.1)"
                ),
                "use `api.tls = \"auto\"` for the local CA, or, for plain HTTP, bind a specific \
                 LAN address and set `api.allow_plaintext_lan = true`",
            ),
            BindClass::Lan if api.allow_plaintext_lan => continue,
            BindClass::Lan => problem(
                Code::InsecureListener,
                origin.clone(),
                "api.tls",
                format!(
                    "`api.tls = \"none\"` would serve plain HTTP on the LAN address `{addr}`, which \
                     needs `api.allow_plaintext_lan = true` (RFC-0050 §11.1)"
                ),
                "use `api.tls = \"auto\"` for the local CA, or set `api.allow_plaintext_lan = \
                 true`, which the dashboard and doctor flag as insecure for as long as it is set",
            ),
            BindClass::Public => problem(
                Code::InsecureListener,
                origin.clone(),
                "api.tls",
                format!(
                    "`api.tls = \"none\"` would serve plain HTTP on `{addr}`, a public address, \
                     where TLS is always required (RFC-0050 §11.1)"
                ),
                "use `api.tls = \"auto\"` for the local CA, or bind to a LAN or loopback \
                 address",
            ),
        };
        findings.errors.push(refused);
    }
}

/// Why the file at `path` cannot serve as a certificate or key file, if it cannot.
///
/// It reads the host directly: `syntropy_config::Environment` checks that a path is usable or
/// can be created, and a missing certificate is never usable.
fn unreadable_file(path: &Path) -> Option<String> {
    match std::fs::metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Some("it does not exist".into()),
        Err(e) => Some(format!("it cannot be read: {e}")),
        Ok(meta) if !meta.is_file() => Some("it is not a file".into()),
        Ok(_) => std::fs::File::open(path)
            .err()
            .map(|e| format!("it cannot be read: {e}")),
    }
}

/// Phase 5: both files of a `{ cert, key }` pair exist and can be read (RFC-0054 §8).
fn tls_files_readable(input: &RuleInput<'_>, findings: &mut Findings) {
    let Some((api, origin)) = section(input) else {
        return;
    };
    let TlsSource::Files { cert, key } = &api.tls else {
        return;
    };
    for (name, path) in [("api.tls.cert", cert), ("api.tls.key", key)] {
        if let Some(why) = unreadable_file(path) {
            findings.errors.push(problem(
                Code::PathUnusable,
                origin.clone(),
                name,
                format!(
                    "`{name}` names `{}`, which cannot be used: {why}",
                    path.display()
                ),
                "a `{ cert, key }` pair needs both files: the certificate chain and its private \
                 key, in PEM",
            ));
        }
    }
}
