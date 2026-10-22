// SPDX-License-Identifier: MIT OR Apache-2.0

//! RFC-0054 §8's validation and error rendering (PROD-E1-T03): a golden snapshot per error code,
//! deprecated keys, newer files, every error in one report, and the property that no rendered
//! error or warning holds a secret-typed value (RFC-0054 §8 note of 2026-10-22).
//!
//! Set `SYNTROPY_BLESS=1` to rewrite `tests/golden/codes.txt` from the current output, then read
//! the diff.

// The sections exist for their schemas; most fields are never read.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use proptest::prelude::*;
use serde::Deserialize;
use syntropy_config::{
    ByteSize, Checked, Code, ConfigFile, ConfigSection, Deprecated, Env, Environment, Findings,
    Flags, HoldsSecret, HostEnvironment, Layers, Origin, Os, PathUse, Phase, Problem, Regex,
    Registry, ResolvedConfig, Rule, RuleInput, Secret, SecretSource, Severity, diagnostic,
    validate,
};
use syntropy_core::policy::NodeSelector;

/// Top-level node keys, with a secret.
#[derive(Deserialize, ConfigSection)]
#[config(section = "", scope = Node, owner = "RFC-0050", deprecated = TOP_DEPRECATED)]
struct Top {
    /// A secret at the top level.
    #[config(reload = Hot, default = unset)]
    api_key: Option<SecretSource>,
}

/// Keys this test's sections renamed.
const TOP_DEPRECATED: &[Deprecated] = &[
    Deprecated {
        path: "old_api_key",
        replacement: Some("api_key"),
        since: "0.3",
    },
    Deprecated {
        path: "t.old_size",
        replacement: Some("t.size"),
        since: "0.3",
    },
];

/// A struct that holds a secret, which its type's text does not show.
#[derive(Debug, Deserialize, HoldsSecret)]
struct Creds {
    user: String,
    token: SecretSource,
}

/// An `[[…]]` entry that holds a secret beside a plain field.
#[derive(Debug, Deserialize, HoldsSecret)]
struct Item {
    name: String,
    token: SecretSource,
}

/// Node keys of each value kind.
#[derive(Deserialize, ConfigSection)]
#[config(section = "t", scope = Node, owner = "RFC-0054")]
struct T {
    /// A size.
    #[config(reload = Hot, default = "1GiB", min = "1MiB")]
    size: ByteSize,
    /// A count.
    #[config(reload = Hot, default = 4, max = 8)]
    count: u32,
    /// A node.
    #[config(reload = Hot, default = "*")]
    #[serde(with = "syntropy_config::form")]
    node: NodeSelector,
    /// A pattern.
    #[config(reload = Hot, default = ".*")]
    pattern: Regex,
    /// Credentials.
    #[config(reload = Hot, default = unset)]
    creds: Option<Creds>,
    /// Listeners.
    #[config(reload = Hot, default = [])]
    bind: Vec<String>,
    /// Entries, each with a secret.
    #[config(reload = Hot, default = [])]
    items: Vec<Item>,
}

/// Exporter settings, with secret headers beside a plain key.
#[derive(Deserialize, ConfigSection)]
#[config(section = "obs.otlp", scope = Node, owner = "RFC-0053")]
struct Otlp {
    /// A port.
    #[config(reload = Hot, default = 0)]
    endpoint: u32,
    /// Headers, each a secret.
    #[config(reload = Hot, default = {})]
    headers: BTreeMap<String, SecretSource>,
}

/// The state directory.
#[derive(Deserialize, ConfigSection)]
#[config(section = "node", scope = Node, owner = "RFC-0054")]
struct NodeSection {
    /// Where state lives.
    #[config(reload = Node, default = computed("RFC-0054 §1"))]
    state_dir: Option<PathBuf>,
}

/// Instance keys at the top of a policy table.
#[derive(Deserialize, ConfigSection)]
#[config(section = "", scope = Instance, owner = "RFC-0054")]
struct Policy {
    /// The objective.
    #[config(reload = Instance, default = "latency")]
    objective: String,
    /// The fidelity.
    #[config(reload = Instance, default = "exact")]
    fidelity: String,
    /// What a request with no fidelity asks for.
    #[config(reload = Instance, default = "exact")]
    default_fidelity: String,
}

/// Approximate knobs.
#[derive(Deserialize, ConfigSection)]
#[config(section = "approximate", scope = Instance, owner = "RFC-0042")]
struct Approximate {
    /// A knob.
    #[config(reload = Instance, default = false)]
    topk_reduce: bool,
}

/// A map of plain secrets, which no `SecretSource` names: the map-of-`Secret` canary.
#[derive(Deserialize, ConfigSection)]
#[config(section = "vault", scope = Node, owner = "RFC-0054")]
struct Vault {
    /// Tokens by name, each a secret.
    #[config(reload = Hot, default = {})]
    tokens: BTreeMap<String, Secret<String>>,
}

static REGISTRY: Registry = Registry::new(&[
    &Top::SCHEMA,
    &T::SCHEMA,
    &Otlp::SCHEMA,
    &NodeSection::SCHEMA,
    &Policy::SCHEMA,
    &Approximate::SCHEMA,
    &Vault::SCHEMA,
]);

/// A host whose answers the test chooses.
#[derive(Default)]
struct FakeHost {
    unusable: Vec<PathBuf>,
    /// Paths that are regular files, so unusable where a directory is expected.
    files: Vec<PathBuf>,
    open_to_others: Vec<PathBuf>,
}

impl Environment for FakeHost {
    fn usable(&self, path: &Path, expected: PathUse) -> Result<(), String> {
        if self.unusable.iter().any(|p| p == path) {
            Err(String::from("permission denied"))
        } else if expected == PathUse::Directory && self.files.iter().any(|p| p == path) {
            Err(String::from("it is a file, not a directory"))
        } else {
            Ok(())
        }
    }

    fn readable_by_others(&self, path: &Path) -> Option<bool> {
        Some(self.open_to_others.iter().any(|p| p == path))
    }
}

const FILE: &str = "/etc/syntropy/config.toml";
const DROPIN: &str = "/etc/syntropy/config.d/50-distro.toml";

/// Loads and validates `files` with `env` and `flags` against the test registry, as
/// `load_and_validate` does after it reads the files.
fn checked(
    files: &[(&str, &str)],
    env: &[(&str, &str)],
    flags: &[&str],
    host: &dyn Environment,
    rules: &[Rule],
) -> Checked {
    let files: Vec<ConfigFile> = files
        .iter()
        .map(|(path, text)| ConfigFile::new(*path, *text))
        .collect();
    let env = Env::from_pairs(Os::Linux, env.iter().copied());
    let mut parsed = Flags::new();
    for flag in flags {
        parsed.set(flag).expect("well-formed flag");
    }
    let outcome = ResolvedConfig::resolve(
        &REGISTRY,
        Layers {
            os: Os::Linux,
            files: &files,
            env: &env,
            flags: &parsed,
        },
    )
    .and_then(|config| validate(&REGISTRY, config, rules, host));
    Checked { files, outcome }
}

fn one_file(text: &str) -> Checked {
    checked(&[(FILE, text)], &[], &[], &FakeHost::default(), &[])
}

/// Phase 5: `node.state_dir` must be a writable directory, so a regular file there is an error,
/// with the test double and with the host's own check; the check leaves no file behind
/// (RFC-0054 §8).
#[test]
fn a_state_dir_must_be_a_writable_directory() {
    let codes = |c: &Checked| -> Vec<Option<Code>> { c.errors().iter().map(|p| p.code).collect() };
    let host = FakeHost {
        files: vec![PathBuf::from("/srv/state.toml")],
        ..FakeHost::default()
    };
    let c = checked(
        &[(FILE, "[node]\nstate_dir = \"/srv/state.toml\"\n")],
        &[],
        &[],
        &host,
        &[],
    );
    assert_eq!(codes(&c), [Some(Code::PathUnusable)], "{:#?}", c.errors());
    assert!(
        c.errors()[0].message.contains("not a directory"),
        "{:#?}",
        c.errors()
    );

    let dir = std::env::temp_dir().join(format!("syntropy-state-dir-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    let file = dir.join("state.toml");
    std::fs::write(&file, "").expect("write");
    // A TOML literal string holds a path's backslashes as written.
    let state_dir = |path: &Path| format!("[node]\nstate_dir = '{}'\n", path.display());
    let on_host =
        |path: &Path| checked(&[(FILE, &state_dir(path))], &[], &[], &HostEnvironment, &[]);
    let c = on_host(&file);
    assert_eq!(codes(&c), [Some(Code::PathUnusable)], "{:#?}", c.errors());
    assert!(
        c.errors()[0].message.contains("not a directory"),
        "{:#?}",
        c.errors()
    );
    let c = on_host(&dir);
    assert!(c.outcome.is_ok(), "{:#?}", c.errors());
    let c = on_host(&dir.join("not-yet"));
    assert!(
        c.outcome.is_ok(),
        "a missing directory can be created: {:#?}",
        c.errors()
    );
    let mut left: Vec<String> = std::fs::read_dir(&dir)
        .expect("read dir")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    left.sort();
    assert_eq!(left, ["state.toml"], "the write check removes its file");
    std::fs::remove_dir_all(&dir).expect("cleanup");
}

/// A phase-4 rule as PROD-E5-T02 will write one: a non-loopback bind is refused here.
fn no_lan_bind(input: &RuleInput<'_>, findings: &mut Findings) {
    let Some(resolved) = input.config.get("t.bind") else {
        return;
    };
    let lan = resolved
        .value()
        .and_then(|v| v.as_array())
        .is_some_and(|binds| binds.iter().any(|b| b.as_str() == Some("0.0.0.0:7433")));
    if lan {
        findings.errors.push(Problem {
            code: Some(Code::InsecureListener),
            at: Some(resolved.origin.clone()),
            column: None,
            width: None,
            key: Some("t.bind".into()),
            message: String::from(
                "`t.bind` listens beyond loopback, which needs an API key and TLS",
            ),
            help: Some("create a key with `syntropy api-key create`".into()),
        });
    }
}

const BIND_RULE: Rule = Rule {
    name: "no_lan_bind",
    phase: Phase::CrossField,
    check: no_lan_bind,
};

/// A problem built by hand, for the codes that phases 1–5 never raise.
fn constructed(code: Code, message: &str) -> Problem {
    Problem {
        code: Some(code),
        at: None,
        column: None,
        width: None,
        key: None,
        message: message.to_owned(),
        help: None,
    }
}

/// One rendered example per error code, in code order.
fn every_code() -> String {
    let mut out = String::new();
    let mut section = |code: Code, rendered: String| {
        out.push_str(&format!("=== {} {}\n", code, code.name()));
        out.push_str(&rendered);
    };
    let first_error =
        |c: &Checked| diagnostic::render(&c.errors()[0], Severity::Error, &c.files, &REGISTRY);

    let c = one_file("[t]\nsize = \"1GiB\n");
    section(Code::Syntax, first_error(&c));
    let c = one_file("[t]\ncount = \"four\"\n");
    section(Code::WrongType, first_error(&c));
    let c = one_file("[t]\nsise = \"1GiB\"\n");
    section(Code::UnknownKey, first_error(&c));
    let c = one_file("[t]\ncount = 99\n");
    section(Code::OutOfRange, first_error(&c));
    let c = one_file("[t]\nsize = \"16XB\"\n");
    section(Code::BadUnit, first_error(&c));
    let c = one_file("[t]\nnode = \"id:12\"\n");
    section(Code::BadSelector, first_error(&c));
    let c = one_file("[t]\npattern = 'a)|(b'\n");
    section(Code::BadRegex, first_error(&c));
    let c = one_file("[policy.\"qwen3*\"]\napproximate.topk_reduce = true\n");
    section(Code::KnobWithoutApproximate, first_error(&c));
    let c = one_file("objective = \"throughput\"\n");
    section(Code::ScopeMismatch, first_error(&c));
    let c = checked(
        &[(FILE, "[t]\nbind = [\"0.0.0.0:7433\"]\n")],
        &[],
        &[],
        &FakeHost::default(),
        &[BIND_RULE],
    );
    section(Code::InsecureListener, first_error(&c));
    let host = FakeHost {
        unusable: vec![PathBuf::from("/var/lib/syntropy")],
        ..FakeHost::default()
    };
    let c = checked(
        &[(FILE, "[node]\nstate_dir = \"/var/lib/syntropy\"\n")],
        &[],
        &[],
        &host,
        &[],
    );
    section(Code::PathUnusable, first_error(&c));
    for (code, message) in [
        (
            Code::OverrideSplitsGroup,
            "override 2 splits group `blk.3.ffn_up_exps`: it selects the gate matrix but not the up \
             and down matrices",
        ),
        (
            Code::ClusterProposalRejected,
            "the cluster has `objective = \"latency\"`, set by node `mac-studio` at index 4127",
        ),
        (
            Code::ReloadPartiallyDeferred,
            "`io.engine` takes effect when the daemon restarts",
        ),
        (
            Code::ManifestSignerNotAllowed,
            "the conformance manifest is signed by a key `models.trusted_signers` does not list",
        ),
        (
            Code::AliasShadowsModel,
            "the alias `qwen3-8b` has the name of a registered model",
        ),
    ] {
        let severity = if code == Code::ReloadPartiallyDeferred {
            Severity::Warning
        } else {
            Severity::Error
        };
        section(
            code,
            diagnostic::render(&constructed(code, message), severity, &[], &REGISTRY),
        );
    }
    out
}

#[test]
fn every_code_matches_its_golden_snapshot() {
    let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/codes.txt");
    let actual = every_code();
    if std::env::var_os("SYNTROPY_BLESS").is_some() {
        std::fs::write(&golden, &actual).expect("write the golden file");
    }
    let expected = std::fs::read_to_string(&golden)
        .expect("tests/golden/codes.txt; set SYNTROPY_BLESS=1 to create it")
        .replace("\r\n", "\n");
    assert_eq!(actual, expected, "set SYNTROPY_BLESS=1 and read the diff");
    for code in Code::ALL {
        assert!(
            actual.contains(&format!("=== {code} ")),
            "no snapshot for {code}"
        );
    }
}

#[test]
fn an_unknown_key_suggests_the_closest() {
    let c = one_file("[t]\nsise = \"1GiB\"\n");
    assert_eq!(
        c.errors()[0].help.as_deref(),
        Some("did you mean `t.size`?")
    );
    let c = checked(
        &[],
        &[("SYNTROPY_T__SISE", "1GiB")],
        &[],
        &FakeHost::default(),
        &[],
    );
    assert_eq!(
        c.warnings()[0].help.as_deref(),
        Some("did you mean `SYNTROPY_T__SIZE`?")
    );
}

#[test]
fn a_deprecated_key_loads_with_a_warning_naming_its_replacement() {
    let c = one_file("[t]\nold_size = \"2GiB\"\n");
    let config = c.outcome.as_ref().expect("a deprecated key loads");
    let warning = &config.warnings()[0];
    assert_eq!(warning.key.as_deref(), Some("t.old_size"));
    assert!(
        warning.message.contains("use `t.size`"),
        "{}",
        warning.message
    );
    let size = config.get("t.size").unwrap();
    assert_eq!(
        size.value(),
        Some(&"2GiB".into()),
        "the value goes to the replacement"
    );
    let rendered = c.render(&REGISTRY);
    assert!(
        rendered.starts_with("warning: `t.old_size` is deprecated since 0.3"),
        "{rendered}"
    );

    // From the environment and a flag too.
    let c = checked(
        &[],
        &[("SYNTROPY_T__OLD_SIZE", "3GiB")],
        &["t.old_size=4GiB"],
        &FakeHost::default(),
        &[],
    );
    let config = c.outcome.as_ref().unwrap();
    assert_eq!(config.warnings().len(), 2);
    assert_eq!(config.get("t.size").unwrap().value(), Some(&"4GiB".into()));
}

#[test]
fn a_newer_config_version_names_both_versions() {
    let c = one_file("config_version = 2\n");
    let error = &c.errors()[0];
    assert_eq!(error.code, Some(Code::OutOfRange));
    assert!(
        error.message.contains("config_version = 2"),
        "{}",
        error.message
    );
    assert!(
        error.message.contains("reads version 1"),
        "{}",
        error.message
    );
    assert!(one_file("config_version = 1\n").outcome.is_ok());
    assert!(one_file("[t]\ncount = 2\n").outcome.is_ok(), "none means 1");
}

#[test]
fn every_error_is_reported_not_only_the_first() {
    // Errors in two files, the environment and a flag, of several codes, in one load.
    let c = checked(
        &[
            (FILE, "[t]\nsize = \"16XB\"\nsise = 1\n"),
            (DROPIN, "[t]\ncount = 99\n"),
        ],
        &[("SYNTROPY_T__NODE", "id:12")],
        &["t.pattern=a)|(b"],
        &FakeHost::default(),
        &[],
    );
    // Files in merge order, each in its table's key order, then the environment, then flags.
    let codes: Vec<Option<Code>> = c.errors().iter().map(|p| p.code).collect();
    assert_eq!(
        codes,
        [
            Some(Code::UnknownKey),
            Some(Code::BadUnit),
            Some(Code::OutOfRange),
            Some(Code::BadSelector),
            Some(Code::BadRegex),
        ],
        "{:#?}",
        c.errors()
    );
    let rendered = c.render(&REGISTRY);
    assert_eq!(rendered.matches("error[config/").count(), 5, "{rendered}");

    // Phase 4 and 5 errors are reported together once phases 1–3 pass.
    let host = FakeHost {
        unusable: vec![PathBuf::from("/srv/state")],
        ..FakeHost::default()
    };
    let c = checked(
        &[(
            FILE,
            "[node]\nstate_dir = \"/srv/state\"\n[t]\nbind = [\"0.0.0.0:7433\"]\n",
        )],
        &[],
        &[],
        &host,
        &[BIND_RULE],
    );
    let codes: Vec<Option<Code>> = c.errors().iter().map(|p| p.code).collect();
    assert_eq!(
        codes,
        [Some(Code::InsecureListener), Some(Code::PathUnusable)]
    );
}

#[test]
fn a_secret_in_a_file_others_can_read_warns() {
    let host = FakeHost {
        open_to_others: vec![PathBuf::from(FILE)],
        ..FakeHost::default()
    };
    let c = checked(&[(FILE, "api_key = \"tok\"\n")], &[], &[], &host, &[]);
    let config = c.outcome.as_ref().unwrap();
    assert_eq!(config.warnings().len(), 1, "{:?}", config.warnings());
    assert!(
        config.warnings()[0]
            .message
            .contains("its group or others can read it")
    );
    // The file form keeps the secret out of the file, so nothing is flagged.
    let c = checked(
        &[(FILE, "api_key = { file = \"/run/secrets/k\" }\n")],
        &[],
        &[],
        &host,
        &[],
    );
    assert!(c.outcome.unwrap().warnings().is_empty());
}

#[test]
fn a_knob_is_allowed_with_approximate_fidelity() {
    let ok = one_file(
        "[policy.\"qwen3*\"]\nfidelity = \"approximate\"\napproximate.topk_reduce = true\n",
    );
    assert!(ok.outcome.is_ok(), "{:?}", ok.errors());
    let ok = one_file(
        "[policy.default]\nfidelity = \"approximate\"\n[policy.\"qwen3*\"]\napproximate.topk_reduce = true\n",
    );
    assert!(ok.outcome.is_ok(), "{:?}", ok.errors());
    let off = one_file("[policy.\"qwen3*\"]\napproximate.topk_reduce = false\n");
    assert!(off.outcome.is_ok(), "a knob set to false is off");
    let bad = one_file("[policy.\"qwen3*\"]\ndefault_fidelity = \"approximate\"\n");
    assert_eq!(bad.errors()[0].code, Some(Code::KnobWithoutApproximate));
}

#[test]
fn policy_tables_hold_known_instance_keys() {
    let c = one_file(
        "[policy.\"qwen3*\"]\nobjectiv = \"throughput\"\n[policy.x]\n\"t\".size = \"1GiB\"\n",
    );
    let codes: Vec<Option<Code>> = c.errors().iter().map(|p| p.code).collect();
    assert_eq!(
        codes,
        [Some(Code::UnknownKey), Some(Code::ScopeMismatch)],
        "{:#?}",
        c.errors()
    );
    assert_eq!(
        c.errors()[0].help.as_deref(),
        Some("did you mean `objective`?")
    );
    // A quoted key holding dots is one unknown key, as at the top level, not the path it spells.
    let c = one_file("[policy.x]\n\"t.size\" = \"1GiB\"\n");
    let codes: Vec<Option<Code>> = c.errors().iter().map(|p| p.code).collect();
    assert_eq!(codes, [Some(Code::UnknownKey)], "{:#?}", c.errors());
    assert_eq!(
        c.errors()[0].message,
        "unknown instance key `\"t.size\"` in `policy.x`"
    );
}

/// One load: files as (path, text), environment variables, and `--set` arguments.
type Case = (
    Vec<(&'static str, String)>,
    Vec<(String, String)>,
    Vec<String>,
);

/// The cases RFC-0054 §8's note and PROD-E1-T03's criterion name, each with the marker `{M}` in
/// a secret-typed value: (files, environment, flags).
fn secret_cases(marker: &str) -> Vec<Case> {
    let m = marker;
    let file = |text: String| vec![(FILE, text)];
    vec![
        // Phase 1: an unquoted value.
        (file(format!("api_key = {m}\n")), vec![], vec![]),
        // Phase 3: a wrong type.
        (file(format!("api_key = [\"{m}\"]\n")), vec![], vec![]),
        // A struct that holds a secret, with an error on its plain field.
        (
            file(format!("[t]\ncreds = {{ user = 5, token = \"{m}\" }}\n")),
            vec![],
            vec![],
        ),
        // An error on a sibling key in the same inline table.
        (
            file(format!(
                "[obs]\notlp = {{ endpoint = \"x\", headers = {{ authorization = \"Bearer {m}\" }} }}\n"
            )),
            vec![],
            vec![],
        ),
        // A mistyped secret key.
        (
            file(format!(
                "[obs.otlp]\nheader = {{ authorization = \"Bearer {m}\" }}\n"
            )),
            vec![],
            vec![],
        ),
        (
            file(format!(
                "[obs.otlp.header]\nauthorization = \"Bearer {m}\"\n"
            )),
            vec![],
            vec![],
        ),
        // A secret in a drop-in, after a file whose last table is not secret.
        (
            vec![
                (FILE, String::from("[obs.otlp.headers]\n")),
                (
                    DROPIN,
                    format!("[t]\ncount = 2\n[obs.otlp.headers]\nauthorization = {m}\n"),
                ),
            ],
            vec![],
            vec![],
        ),
        // A multi-line string, with the error on a continuation line.
        (
            file(format!("api_key = \"\"\"\n{m} \\q\n\"\"\"\n")),
            vec![],
            vec![],
        ),
        // A deprecated name of a secret key, with an error on the same line.
        (file(format!("old_api_key = \"{m}\" x\n")), vec![], vec![]),
        (file(format!("old_api_key = \"{m}\"\n")), vec![], vec![]),
        // A node secret misplaced in a policy table, by key and by a table set whole.
        (
            file(format!("[policy.x]\napi_key = \"{m}\"\n")),
            vec![],
            vec![],
        ),
        (
            file(format!(
                "[policy]\n\"qwen3.5*\" = {{ api_key = \"{m}\" }}\n"
            )),
            vec![],
            vec![],
        ),
        // A secret set by a quoted key holding dots, which layering joins as a dotted path.
        (
            file(format!("\"obs.otlp.headers\" = [\"{m}\"]\n")),
            vec![],
            vec![],
        ),
        (
            file(format!(
                "[obs]\n'otlp.headers' = {{ authorization = [\"{m}\"] }}\n"
            )),
            vec![],
            vec![],
        ),
        (
            file(format!("[policy.x]\n\"obs.otlp.headers\" = [\"{m}\"]\n")),
            vec![],
            vec![],
        ),
        // A map of `Secret<String>`: an entry of the wrong type, an unquoted entry, the map set
        // inline, and the map from `--set` and the environment.
        (
            file(format!("[vault.tokens]\nprod = [\"{m}\"]\n")),
            vec![],
            vec![],
        ),
        (
            file(format!("[vault.tokens]\nprod = {m}\n")),
            vec![],
            vec![],
        ),
        (
            file(format!("[vault]\ntokens = {{ prod = \"{m}\", dev = 5 }}\n")),
            vec![],
            vec![],
        ),
        (vec![], vec![], vec![format!("vault.tokens=[\"{m}\"]")]),
        (
            vec![],
            vec![(String::from("SYNTROPY_VAULT__TOKENS"), m.to_owned())],
            vec![],
        ),
        // An error on another key of the same `[[…]]` entry, as a table and inline.
        (
            file(format!("[[t.items]]\nname = 5\ntoken = \"{m}\"\n")),
            vec![],
            vec![],
        ),
        (
            file(format!("[[t.items]]\nname = 5 token = \"{m}\"\n")),
            vec![],
            vec![],
        ),
        (
            file(format!("[t]\nitems = [{{ name = 5, token = \"{m}\" }}]\n")),
            vec![],
            vec![],
        ),
        // The value of an unknown key, in a file, a policy table and a flag, and of an unknown
        // `SYNTROPY_*` variable, whose warning names the variable alone.
        (file(format!("bogus = \"{m}\"\n")), vec![], vec![]),
        (file(format!("[t]\nunknown = \"{m}\"\n")), vec![], vec![]),
        (
            file(format!("[policy.x]\nnope = \"{m}\"\n")),
            vec![],
            vec![],
        ),
        (vec![], vec![], vec![format!("nope.key={m}")]),
        (
            vec![],
            vec![(String::from("SYNTROPY_NOPE"), m.to_owned())],
            vec![],
        ),
        (
            vec![],
            vec![(
                String::from("SYNTROPY_OBS__OTLP__HEADER__AUTHORIZATION"),
                format!("Bearer {m}"),
            )],
            vec![],
        ),
        // `--set` and the environment.
        (vec![], vec![], vec![format!("api_key=[\"{m}\"]")]),
        (vec![], vec![], vec![format!("api_kye={m}")]),
        (
            vec![],
            vec![(
                String::from("SYNTROPY_OBS__OTLP__HEADERS"),
                format!("[\"{m}\"]"),
            )],
            vec![],
        ),
        (
            vec![],
            vec![(String::from("SYNTROPY_OBS__OTLP__HEADER"), m.to_owned())],
            vec![],
        ),
    ]
}

/// Every error and warning of a load of `case`, rendered.
fn render_case(case: &Case) -> String {
    let (files, env, flags) = case;
    let files: Vec<(&str, &str)> = files.iter().map(|(p, t)| (*p, t.as_str())).collect();
    let env: Vec<(&str, &str)> = env.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    let flags: Vec<&str> = flags.iter().map(String::as_str).collect();
    let c = checked(&files, &env, &flags, &FakeHost::default(), &[]);
    assert!(
        !c.errors().is_empty() || !c.warnings().is_empty(),
        "the case reports nothing: {case:?}"
    );
    c.render(&REGISTRY)
}

#[test]
fn no_rendered_problem_holds_a_secret() {
    for case in secret_cases("mk7Q2xV9pL") {
        let rendered = render_case(&case);
        assert!(!rendered.contains("mk7Q2xV9pL"), "{case:?}\n{rendered}");
    }
}

#[test]
fn a_plain_value_is_shown() {
    // The control: a value that is not a secret is printed, so the property above is not met
    // by printing nothing.
    let c = one_file("[t]\nsize = \"mk7Q2xV9pL\"\n");
    assert!(c.render(&REGISTRY).contains("mk7Q2xV9pL"));
}

proptest! {
    #[test]
    fn no_marker_reaches_any_rendered_problem(token in "[A-Za-z0-9]{6,24}") {
        let marker = format!("mk{token}");
        for case in secret_cases(&marker) {
            let rendered = render_case(&case);
            prop_assert!(!rendered.contains(&marker), "{:?}\n{}", case, rendered);
        }
    }
}

#[test]
fn the_registry_is_consistent() {
    assert_eq!(REGISTRY.check(), vec![]);
    assert_eq!(REGISTRY.check_defaults(), vec![]);
    let origin = Origin::File {
        path: PathBuf::from(FILE),
        line: 1,
    };
    // A problem renders without its file at hand, by its place alone.
    let p = Problem {
        code: Some(Code::BadUnit),
        at: Some(origin),
        column: None,
        width: None,
        key: None,
        message: String::from("bad"),
        help: None,
    };
    assert_eq!(
        diagnostic::render(&p, Severity::Error, &[], &REGISTRY),
        "error[config/0005]: bad\n  --> /etc/syntropy/config.toml:1\n"
    );
}
