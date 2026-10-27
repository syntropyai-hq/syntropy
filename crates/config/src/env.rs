// SPDX-License-Identifier: MIT OR Apache-2.0

//! The environment layer (RFC-0054 §2.4).
//!
//! Every node key `a.b.c` maps to `SYNTROPY_A__B__C`: upper case, `.` becomes `__`, `_` stays.
//! Five names sit outside that pattern:
//!
//! | Variable | Meaning |
//! |---|---|
//! | `SYNTROPY_CONFIG` | the explicit file, which replaces the system file, the drop-ins and the user file |
//! | `SYNTROPY_HOME` | the state directory (`node.state_dir`) and the default store |
//! | `HF_TOKEN` | the Hub token, when none is configured (RFC-0041 §7.5) |
//! | `SYNTROPY_HOST`, `SYNTROPY_API_KEY` | read only by CLI client commands, to reach a remote daemon (RFC-0051 §1) |
//!
//! Any other `SYNTROPY_*` variable that names no key is reported as a warning, because a typo
//! in a variable is otherwise invisible. The variables in [`OTHER_KNOWN`], which other parts of
//! Syntropy read, are not.

use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::path::Path;

use crate::discovery::{CONFIG_VAR, HOME_VAR, Os};
use crate::value::Secret;

/// The prefix of every Syntropy variable.
pub const PREFIX: &str = "SYNTROPY_";
/// The Hub token's variable, the Hub's own convention.
pub const HF_TOKEN_VAR: &str = "HF_TOKEN";
/// The remote daemon a CLI client command reaches.
pub const HOST_VAR: &str = "SYNTROPY_HOST";
/// The API key a CLI client command sends to a remote daemon.
pub const API_KEY_VAR: &str = "SYNTROPY_API_KEY";

/// The variables outside the `SYNTROPY_A__B` pattern that start with `SYNTROPY_`.
pub const FIXED: [&str; 4] = [CONFIG_VAR, HOME_VAR, HOST_VAR, API_KEY_VAR];

/// `SYNTROPY_*` variables that other parts of Syntropy read, which are not config keys and so
/// draw no warning:
/// - `SYNTROPY_FAULT_PLAN`: the fault plan a fault-injection build loads at process start
///   (RFC-0060 §9.2);
/// - `SYNTROPY_KERNELS`: how a `kernels-*` crate's build gets its kernels, `prebuilt`, `build` or
///   `none` (COMP-E9-T02), which is often exported in a developer's shell.
pub const OTHER_KNOWN: [&str; 2] = ["SYNTROPY_FAULT_PLAN", "SYNTROPY_KERNELS"];

/// A snapshot of the environment. The daemon takes one at start; the environment layer does
/// not change while it runs (RFC-0054 §9.2).
///
/// On Windows variable names are case-insensitive, so the snapshot stores them upper-cased.
///
/// Its `Debug` lists the variables' names, never their values: the snapshot holds
/// `SYNTROPY_API_KEY`, `SYNTROPY_HF_TOKEN` and any `{ env = … }` secret's variable.
#[derive(Clone, Default)]
pub struct Env {
    vars: BTreeMap<OsString, OsString>,
}

impl std::fmt::Debug for Env {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Env")
            .field("names", &self.vars.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl Env {
    /// The current process's environment.
    pub fn from_process(os: Os) -> Env {
        Env::from_pairs(os, std::env::vars_os())
    }

    /// An environment of `pairs`, read with `os`'s rules for names.
    pub fn from_pairs<K: Into<OsString>, V: Into<OsString>>(
        os: Os,
        pairs: impl IntoIterator<Item = (K, V)>,
    ) -> Env {
        let vars = pairs
            .into_iter()
            .map(|(k, v)| {
                let k = k.into();
                let k = match os {
                    Os::Windows => k.to_ascii_uppercase(),
                    Os::Linux | Os::MacOs => k,
                };
                (k, v.into())
            })
            .collect();
        Env { vars }
    }

    /// The variable's value.
    pub fn get(&self, name: &str) -> Option<&OsStr> {
        self.vars.get(OsStr::new(name)).map(OsString::as_os_str)
    }

    /// The variable's value, treating an empty value as unset.
    pub fn get_nonempty(&self, name: &str) -> Option<&OsStr> {
        self.get(name).filter(|v| !v.is_empty())
    }

    /// The variable's value if it is an absolute path for `os`, as the XDG variables must be.
    pub(crate) fn absolute(&self, os: Os, name: &str) -> Option<&OsStr> {
        self.get_nonempty(name).filter(|v| match os {
            Os::Linux | Os::MacOs => v.as_encoded_bytes().first() == Some(&b'/'),
            Os::Windows => Path::new(v).is_absolute(),
        })
    }

    /// Every variable whose name starts with `SYNTROPY_`, in name order.
    pub(crate) fn syntropy_vars(&self) -> impl Iterator<Item = (&OsStr, &OsStr)> {
        self.vars
            .iter()
            .filter(|(k, _)| k.as_encoded_bytes().starts_with(PREFIX.as_bytes()))
            .map(|(k, v)| (k.as_os_str(), v.as_os_str()))
    }

    /// `HF_TOKEN`, the node's Hub token when none is configured.
    pub fn hf_token(&self) -> Option<Secret<String>> {
        self.secret(HF_TOKEN_VAR)
    }

    /// `SYNTROPY_HOST`, the remote daemon a CLI client command reaches.
    pub fn host(&self) -> Option<&OsStr> {
        self.get_nonempty(HOST_VAR)
    }

    /// `SYNTROPY_API_KEY`, the key a CLI client command sends to a remote daemon.
    pub fn api_key(&self) -> Option<Secret<String>> {
        self.secret(API_KEY_VAR)
    }

    fn secret(&self, name: &str) -> Option<Secret<String>> {
        self.get_nonempty(name)
            .and_then(OsStr::to_str)
            .map(|v| Secret::new(v.to_owned()))
    }
}

/// The variable that sets `path`: `resource.ram_headroom` is
/// `SYNTROPY_RESOURCE__RAM_HEADROOM`. The derive records the same name in
/// [`KeySchema::env`](crate::KeySchema::env) for node keys.
pub fn var_for(path: &str) -> String {
    format!("{PREFIX}{}", path.to_ascii_uppercase().replace('.', "__"))
}

/// The key path a `SYNTROPY_*` variable names, or `None` for a variable outside the pattern:
/// an empty segment (`SYNTROPY_A____B`), lower-case letters, or characters a key cannot hold.
pub fn path_for(var: &str) -> Option<String> {
    let rest = var.strip_prefix(PREFIX)?;
    if rest.is_empty() {
        return None;
    }
    let segments: Vec<&str> = rest.split("__").collect();
    let well_formed = segments.iter().all(|s| {
        !s.is_empty()
            && !s.starts_with('_')
            && !s.ends_with('_')
            && s.bytes()
                .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
    });
    well_formed.then(|| segments.join(".").to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_keys_to_variables_and_back() {
        assert_eq!(
            var_for("resource.ram_headroom"),
            "SYNTROPY_RESOURCE__RAM_HEADROOM"
        );
        assert_eq!(
            path_for("SYNTROPY_RESOURCE__RAM_HEADROOM").as_deref(),
            Some("resource.ram_headroom")
        );
        assert_eq!(
            path_for("SYNTROPY_API__AUTH__MODE").as_deref(),
            Some("api.auth.mode")
        );
        assert_eq!(path_for("SYNTROPY_OFFLINE").as_deref(), Some("offline"));
        for bad in [
            "SYNTROPY_",
            "SYNTROPY_A____B",
            "SYNTROPY_A__",
            "SYNTROPY_api__bind",
            "SYNTROPY_A-B",
            "SYNTROPY_A___B",
            "OTHER_A",
        ] {
            assert_eq!(path_for(bad), None, "{bad}");
        }
    }

    #[test]
    fn windows_names_are_case_insensitive() {
        let e = Env::from_pairs(Os::Windows, [("Syntropy_Config", "x"), ("hf_token", "t")]);
        assert_eq!(e.get("SYNTROPY_CONFIG"), Some(OsStr::new("x")));
        assert!(e.hf_token().is_some());
        let e = Env::from_pairs(Os::Linux, [("Syntropy_Config", "x")]);
        assert_eq!(e.get("SYNTROPY_CONFIG"), None);
    }

    #[test]
    fn secrets_never_print() {
        let e = Env::from_pairs(
            Os::Linux,
            [("HF_TOKEN", "hf_canary"), ("SYNTROPY_API_KEY", "k")],
        );
        let token = e.hf_token().expect("set");
        assert_eq!(format!("{token} {token:?}"), "[redacted] [redacted]");
        assert_eq!(token.expose_for("test"), "hf_canary");
        assert!(e.api_key().is_some());
        assert_eq!(e.host(), None);
    }
}
