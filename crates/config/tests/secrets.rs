// SPDX-License-Identifier: MIT OR Apache-2.0

//! A canary secret never leaves its `Secret` (PROD-E1-T12; RFC-0054 §10.1).
//!
//! One marker is planted in a secret in each form and from each layer: written inline in a
//! file, in a file a `{ file = … }` secret names, in a variable an `{ env = … }` secret names,
//! in a `SYNTROPY_*` variable, in a `--set` flag, and through `config set`. The tests then check
//! everything a load or a write produces as text:
//! - `config show --resolved`'s text, which prints `[redacted]` and the origin;
//! - every `Debug` and serde form on the load path: the files, the flags, the environment, the
//!   resolved configuration, the check's result, a section built from it, `Secret` and
//!   `SecretSource`. A log line that formats any of them with `{:?}` cannot carry the secret;
//! - the write path: `config set`'s result check, its errors, and the validation of an edit.
//!
//! The marker is still there for the code that needs it, through `expose_for`. And a secret's
//! memory is zeroed on drop.
//!
//! Logging itself (PROD-E8-T04) and the debug bundle (PROD-E9-T04) are not built. What reaches
//! them goes through these types, so these tests are the redaction they rely on; their own tasks
//! add the end-to-end checks.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use serde::Deserialize;
use syntropy_config::write::{check_edit, set};
use syntropy_config::{
    ConfigFile, ConfigSection, Env, Environment, Flags, Layers, Os, PathUse, Registry,
    ResolvedConfig, Scan, Secret, SecretSource, command, validate,
};
use zeroize::Zeroize;

/// The canary. No test output may contain it.
const CANARY: &str = "cnry-7Hq2Zp9w";

/// Keys of each secret shape, beside plain ones.
#[derive(Debug, Deserialize, ConfigSection)]
#[config(section = "vault", scope = Node, owner = "RFC-0054")]
struct Vault {
    /// A token in any of the three forms.
    #[config(reload = Hot, default = unset)]
    token: Option<SecretSource>,
    /// Headers, each a secret in any form.
    #[config(reload = Hot, default = {})]
    headers: BTreeMap<String, SecretSource>,
    /// A plain secret.
    #[config(reload = Hot, default = unset)]
    password: Option<Secret<String>>,
    /// Not a secret.
    #[config(reload = Hot, default = 1)]
    retries: u32,
}

static REGISTRY: Registry = Registry::new(&[&Vault::SCHEMA]);

/// A host where every path is usable and no file is open to others.
struct Host;

impl Environment for Host {
    fn usable(&self, _: &Path, _: PathUse) -> Result<(), String> {
        Ok(())
    }

    fn readable_by_others(&self, _: &Path) -> Option<bool> {
        Some(false)
    }
}

const FILE: &str = "/etc/syntropy/config.toml";

/// No text here holds the canary.
fn clean(what: &str, text: &str) {
    assert!(!text.contains(CANARY), "{what} holds the secret:\n{text}");
}

/// A file with the canary inline, as a `[vault]` table with comments.
fn file_text() -> String {
    format!(
        "# secrets\n[vault]\ntoken = \"{CANARY}-inline\"   # inline\nretries = 3\n\
         [vault.headers]\nauthorization = \"Bearer {CANARY}-header\"\n"
    )
}

/// A secret file and its path: the `{ file = … }` form's target.
fn secret_file() -> PathBuf {
    let path = std::env::temp_dir().join(format!("syntropy-canary-{}", std::process::id()));
    std::fs::write(&path, format!("{CANARY}-file\n")).unwrap();
    path
}

fn resolve(files: &[ConfigFile], env: &Env, flags: &Flags) -> ResolvedConfig {
    let config = ResolvedConfig::resolve(
        &REGISTRY,
        Layers {
            os: Os::Linux,
            files,
            env,
            flags,
        },
    )
    .expect("resolves");
    validate(&REGISTRY, config, &[], &Host).expect("valid")
}

/// Every layer and form at once: the file inline, a `--set` flag, a `SYNTROPY_*` variable, and
/// the `{ file = … }` and `{ env = … }` forms through the environment and the flags.
#[test]
fn config_show_prints_redacted_and_the_origin_for_every_form() {
    let secret = secret_file();
    let files = [ConfigFile::new(FILE, file_text())];
    let env = Env::from_pairs(
        Os::Linux,
        [
            ("SYNTROPY_VAULT__PASSWORD", format!("{CANARY}-envvar")),
            ("VAULT_ENV_SECRET", format!("{CANARY}-envform")),
        ],
    );
    let mut flags = Flags::new();
    flags
        .set(&format!(
            "vault.headers={{ authorization = \"{CANARY}-flag\", x-file = {{ file = {:?} }}, \
             x-env = {{ env = \"VAULT_ENV_SECRET\" }} }}",
            secret.display()
        ))
        .unwrap();
    let config = resolve(&files, &env, &flags);
    let shown = command::show_resolved(&config);
    clean("config show --resolved", &shown);
    assert!(
        shown
            .lines()
            .any(|l| l.starts_with("vault.token = [redacted]") && l.ends_with(&format!("{FILE}:3"))),
        "{shown}"
    );
    assert!(
        shown
            .lines()
            .any(|l| l.starts_with("vault.password = [redacted]")
                && l.ends_with("env SYNTROPY_VAULT__PASSWORD")),
        "{shown}"
    );
    assert!(
        shown
            .lines()
            .any(|l| l.starts_with("vault.headers = [redacted]")
                && l.ends_with("flag --set vault.headers")),
        "{shown}"
    );
    // A plain key beside them prints its value.
    assert!(shown.contains("vault.retries = 3"), "{shown}");

    // The values are there for the code that reads them, and only through `expose_for`.
    let vault: Vault = config.section().unwrap();
    let token = vault.token.as_ref().unwrap().resolve().unwrap();
    assert_eq!(token.expose_for("test"), &format!("{CANARY}-inline"));
    let headers = &vault.headers;
    let from_file = headers["x-file"].resolve().unwrap();
    assert_eq!(from_file.expose_for("test"), &format!("{CANARY}-file"));
    assert_eq!(vault.retries, 3);
    let password = vault.password.as_ref().unwrap();
    assert_eq!(password.expose_for("test"), &format!("{CANARY}-envvar"));

    // Every `Debug` and serde form on the way.
    clean("Debug of the files", &format!("{files:?}"));
    clean("Debug of the flags", &format!("{flags:?}"));
    clean("Debug of the environment", &format!("{env:?}"));
    clean("Debug of the configuration", &format!("{config:?}"));
    clean("pretty Debug of the configuration", &format!("{config:#?}"));
    clean("Debug of the section", &format!("{vault:?}"));
    clean(
        "Debug of a resolved secret",
        &format!("{token:?} {from_file:?}"),
    );
    clean(
        "Display of a resolved secret",
        &format!("{token} {password}"),
    );
    for source in vault.headers.values().chain(vault.token.as_ref()) {
        clean("Display of a source", &source.to_string());
        clean("JSON of a source", &serde_json::to_string(source).unwrap());
    }
    clean(
        "JSON of a secret",
        &serde_json::to_string(password).unwrap(),
    );
    assert!(format!("{env:?}").contains("SYNTROPY_VAULT__PASSWORD"));
    let flags = format!("{flags:?}");
    assert!(
        flags.contains("\"--set vault.headers\"") && flags.contains("[redacted]"),
        "{flags}"
    );
    std::fs::remove_file(secret).unwrap();
}

/// A check's result keeps the files' text to render problems with their lines; its `Debug`
/// shows their paths and sizes only.
#[test]
fn a_check_s_debug_never_holds_a_file_s_text() {
    let scan = Scan {
        system_file: Some(PathBuf::from(FILE)),
        dropins: Vec::new(),
        user_file: None,
        explicit: None,
    };
    let env = Env::from_pairs(Os::Linux, std::iter::empty::<(&str, &str)>());
    let checked = check_edit(
        &REGISTRY,
        Os::Linux,
        &scan,
        &env,
        Path::new(FILE),
        &file_text(),
        &[],
        &Host,
    );
    assert!(checked.outcome.is_ok(), "{:?}", checked.errors());
    let debug = format!("{checked:?}");
    clean("Debug of a check", &debug);
    assert!(debug.contains(FILE) && debug.contains("bytes"), "{debug}");
}

/// `config set` of a secret: the written text holds it, as it must, but nothing the command
/// prints does, valid or not.
#[test]
fn a_secret_written_by_config_set_appears_only_in_the_file() {
    let scan = Scan {
        system_file: None,
        dropins: Vec::new(),
        user_file: Some(PathBuf::from(FILE)),
        explicit: None,
    };
    let env = Env::from_pairs(Os::Linux, std::iter::empty::<(&str, &str)>());
    let check = |text: &str| {
        check_edit(
            &REGISTRY,
            Os::Linux,
            &scan,
            &env,
            Path::new(FILE),
            text,
            &[],
            &Host,
        )
    };

    let edited = set(
        "[vault]\nretries = 2\n",
        "vault.token",
        &format!("{CANARY}-set"),
    )
    .unwrap();
    assert!(edited.contains(CANARY));
    let checked = check(&edited);
    let config = checked.outcome.as_ref().expect("valid");
    clean("the check's diagnostics", &checked.render(&REGISTRY));
    clean("config show after set", &command::show_resolved(config));

    // Invalid: a secret set where a number goes, and a secret table of the wrong shape. The
    // write is refused, and its diagnostics withhold the value.
    // (A secret written into a key that is not secret-typed is printed like any value there:
    // RFC-0054 §8 withholds by the key's type, and the validation tests use that as a control.)
    for (key, value) in [
        ("vault.password", format!("[\"{CANARY}-array\"]")),
        ("vault.token", format!("{{ path = \"{CANARY}-shape\" }}")),
        ("vault.tokn", format!("{CANARY}-typo")),
    ] {
        let edited = set("[vault]\nretries = 2\n", key, &value).unwrap();
        let checked = check(&edited);
        assert!(checked.outcome.is_err(), "{key} should not validate");
        clean(
            &format!("diagnostics for {key}"),
            &checked.render(&REGISTRY),
        );
        clean(
            &format!("report for {key}"),
            &command::check(&checked, &REGISTRY).text,
        );
        clean(&format!("Debug for {key}"), &format!("{checked:?}"));
    }

    // An edit that cannot be made names the key, never the value or the file's.
    let error = set(
        &format!("[vault]\ntoken = \"{CANARY}-old\"\n"),
        "vault.token.inner",
        &format!("{CANARY}-new"),
    )
    .unwrap_err();
    clean("an edit error", &error.to_string());
}

/// A probe that records being zeroed.
struct Probe(Rc<Cell<bool>>);

impl Zeroize for Probe {
    fn zeroize(&mut self) {
        self.0.set(true);
    }
}

/// A secret's memory is zeroed when it is dropped.
#[test]
fn a_secret_is_zeroed_on_drop() {
    let zeroed = Rc::new(Cell::new(false));
    let secret = Secret::new(Probe(zeroed.clone()));
    assert!(!zeroed.get());
    drop(secret);
    assert!(zeroed.get(), "dropping a Secret did not zeroize its value");
}
