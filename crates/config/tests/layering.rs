// SPDX-License-Identifier: MIT OR Apache-2.0

//! Node-key layering (PROD-E1-T02; RFC-0054 §1, §2.2 Rule 1, §2.4).
//!
//! The acceptance criteria this file checks:
//! - property tests for Rule 1 precedence and drop-in order;
//! - `show --resolved` prints file:line, env var or flag per value;
//! - unknown `SYNTROPY_*` variables produce a warning.
//!
//! The per-OS path tests are in `src/discovery.rs`, which runs each OS's rules on any host and
//! the host's own rules natively.

// The sections exist for their schemas; most fields are never read.
#![allow(dead_code)]

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};

use proptest::prelude::*;
use serde::Deserialize;
use syntropy_config::{
    ByteSize, Checked, Code, ConfigFile, ConfigSection, Dirs, Env, Flags, HoldsSecret,
    HostEnvironment, Layers, Locations, Mode, Origin, Os, Registry, ResolvedConfig, SecretSource,
    SharedConfig, Value, command, diagnostic, load, load_and_validate, order_dropins,
};

/// Keys for the layering tests.
#[derive(Debug, Deserialize, ConfigSection)]
#[config(section = "t", scope = Node, owner = "RFC-0054")]
struct TSection {
    /// First.
    #[config(reload = Hot, default = 1)]
    a: i64,
    /// Second.
    #[config(reload = Hot, default = 2)]
    b: i64,
    /// Third.
    #[config(reload = Hot, default = 3)]
    c: i64,
    /// Fourth.
    #[config(reload = Hot, default = 4)]
    d: i64,
    /// A string.
    #[config(reload = Hot, default = "x")]
    name: String,
    /// A size.
    #[config(reload = Hot, default = "1GiB", min = "1MiB")]
    size: ByteSize,
    /// A secret.
    #[config(reload = Hot, default = unset)]
    token: Option<SecretSource>,
    /// Computed from the probe.
    #[config(reload = Node, default = computed("from the probe"))]
    cores: u32,
}

/// The state directory, which `SYNTROPY_HOME` sets.
#[derive(Debug, Deserialize, ConfigSection)]
#[config(section = "node", scope = Node, owner = "RFC-0054")]
struct NodeSection {
    /// Where the node keeps its state.
    #[config(reload = Node, default = computed("RFC-0054 §1, per OS"))]
    state_dir: PathBuf,
}

/// A cluster key.
#[derive(Debug, Deserialize, ConfigSection)]
#[config(section = "", scope = Cluster, owner = "RFC-0054")]
struct ClusterKeys {
    /// Whether approximate plans are allowed.
    #[config(reload = Hot, default = false, introduced_at = 1, authority = Keyholder)]
    allow_approximate: bool,
}

/// An instance key.
#[derive(Debug, Deserialize, ConfigSection)]
#[config(section = "", scope = Instance, owner = "RFC-0054")]
struct InstanceKeys {
    /// What the plan optimizes.
    #[config(reload = Instance, default = "latency")]
    objective: String,
}

/// A request key.
#[derive(Debug, Deserialize, ConfigSection)]
#[config(section = "", scope = Request, owner = "RFC-0054")]
struct RequestKeys {
    /// The most tokens a request generates.
    #[config(reload = Hot, default = 256)]
    max_tokens: u32,
}

static REGISTRY: Registry = Registry::new(&[
    &TSection::SCHEMA,
    &NodeSection::SCHEMA,
    &ClusterKeys::SCHEMA,
    &InstanceKeys::SCHEMA,
    &RequestKeys::SCHEMA,
]);

const KEYS: [&str; 4] = ["a", "b", "c", "d"];

fn env(pairs: &[(&str, &str)]) -> Env {
    Env::from_pairs(Os::Linux, pairs.iter().map(|(k, v)| (*k, *v)))
}

fn resolve(files: &[ConfigFile], env: &Env, flags: &Flags) -> ResolvedConfig {
    ResolvedConfig::resolve(
        &REGISTRY,
        Layers {
            os: Os::Linux,
            files,
            env,
            flags,
        },
    )
    .expect("resolves")
}

fn int(config: &ResolvedConfig, path: &str) -> Option<i64> {
    config
        .get(path)
        .and_then(|r| r.value())
        .and_then(Value::as_integer)
}

#[test]
fn the_registry_is_consistent() {
    assert_eq!(REGISTRY.check(), Vec::new());
    assert_eq!(REGISTRY.check_defaults(), Vec::new());
}

/// One file of the Rule 1 property test: which keys it sets, and whether it writes them under a
/// `[t]` header or as dotted keys.
#[derive(Clone, Debug)]
struct FileSpec {
    values: [Option<i64>; 4],
    header: bool,
}

fn file_spec() -> impl Strategy<Value = FileSpec> {
    (
        proptest::array::uniform4(proptest::option::of(-1000i64..1000)),
        any::<bool>(),
    )
        .prop_map(|(values, header)| FileSpec { values, header })
}

/// The file's text, and the line each key is written on.
fn render(spec: &FileSpec) -> (String, [Option<u32>; 4]) {
    let mut text = String::from("# a comment\n");
    let mut line = 2;
    if spec.header {
        text.push_str("[t]\n");
        line += 1;
    }
    let mut lines = [None; 4];
    for (i, value) in spec.values.iter().enumerate() {
        if let Some(v) = value {
            let prefix = if spec.header { "" } else { "t." };
            text.push_str(&format!("{prefix}{} = {v}\n", KEYS[i]));
            lines[i] = Some(line);
            line += 1;
        }
    }
    (text, lines)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Rule 1: `default → config files in merge order → environment → CLI`, later layers
    /// winning key by key, and every value's origin names the layer that set it.
    #[test]
    fn rule_1_later_layers_win_key_by_key(
        files in proptest::collection::vec(file_spec(), 0..5),
        env_values in proptest::array::uniform4(proptest::option::of(-1000i64..1000)),
        flags in proptest::collection::vec((0usize..4, -1000i64..1000), 0..5),
    ) {
        let mut config_files = Vec::new();
        let mut lines = Vec::new();
        for (n, spec) in files.iter().enumerate() {
            let (text, at) = render(spec);
            config_files.push(ConfigFile::new(format!("/layer/{n}.toml"), text));
            lines.push(at);
        }
        let env_pairs: Vec<(String, String)> = env_values
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|v| (format!("SYNTROPY_T__{}", KEYS[i].to_uppercase()), v.to_string())))
            .collect();
        let env = Env::from_pairs(Os::Linux, env_pairs);
        let mut cli = Flags::new();
        for (i, v) in &flags {
            cli.set(&format!("t.{}={v}", KEYS[*i])).expect("well-formed");
        }
        let resolved = resolve(&config_files, &env, &cli);

        for (i, key) in KEYS.iter().enumerate() {
            let path = format!("t.{key}");
            let got = resolved.get(&path).expect("every node key resolves");
            let from_flag = flags.iter().rev().find(|(k, _)| *k == i).map(|(_, v)| *v);
            let from_file = (0..files.len()).rev().find_map(|n| files[n].values[i].map(|v| (n, v)));
            let (value, origin) = if let Some(v) = from_flag {
                (v, Origin::Cli { flag: format!("--set {path}") })
            } else if let Some(v) = env_values[i] {
                (v, Origin::Env { var: format!("SYNTROPY_T__{}", key.to_uppercase()) })
            } else if let Some((n, v)) = from_file {
                (v, Origin::File { path: PathBuf::from(format!("/layer/{n}.toml")), line: lines[n][i].expect("written") })
            } else {
                (i as i64 + 1, Origin::Default { computed: None })
            };
            prop_assert_eq!(got.value().and_then(Value::as_integer), Some(value), "{}", path);
            prop_assert_eq!(&got.origin, &origin, "{}", path);
        }
    }

    /// Drop-ins merge in lexical (byte) order, after the system file and before the user file,
    /// whatever order the file system lists them in.
    #[test]
    fn dropins_merge_in_lexical_order(
        names in proptest::collection::btree_set("[0-9a-z_-]{1,6}", 1..8),
        extras in proptest::collection::btree_set("[.][0-9a-z]{1,4}[.]toml|[0-9a-z]{1,4}[.](txt|toml~)", 0..3),
        shuffle in any::<proptest::sample::Index>(),
        user in any::<bool>(),
    ) {
        let dropins: Vec<String> = names.iter().map(|n| format!("{n}.toml")).collect();
        // The pure order, from a rotated listing.
        let mut listing: Vec<String> = dropins.iter().chain(&extras).cloned().collect();
        let k = shuffle.index(listing.len());
        listing.rotate_left(k);
        let ordered = order_dropins(Os::Linux, listing.iter().map(Into::into));
        let sorted: Vec<String> = dropins.iter().cloned().collect::<BTreeSet<_>>().into_iter().collect();
        prop_assert_eq!(
            ordered.iter().map(|n| n.to_string_lossy().into_owned()).collect::<Vec<_>>(),
            sorted.clone()
        );

        // On disk, created in the rotated order.
        let root = TempDir::new("dropins");
        let locations = root.locations(Mode::User);
        fs::create_dir_all(&locations.dropin_dir).expect("mkdir");
        fs::write(&locations.system_file, "t.a = -1\nt.c = -1\n").expect("write");
        for name in &listing {
            let index = sorted.iter().position(|n| n == name).map_or(-2, |i| i as i64);
            fs::write(locations.dropin_dir.join(name), format!("t.a = {index}\nt.b = {index}\n")).expect("write");
        }
        if user {
            fs::write(locations.user_file.as_ref().expect("user file"), "t.a = 1000\n").expect("write");
        }
        let scan = locations.scan().expect("scans");
        prop_assert_eq!(
            scan.dropins.iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect::<Vec<_>>(),
            sorted.clone()
        );
        let resolved = load(&REGISTRY, Os::Linux, &scan, &Env::default(), &Flags::new()).expect("loads");
        let last = sorted.len() as i64 - 1;
        prop_assert_eq!(int(&resolved, "t.a"), Some(if user { 1000 } else { last }));
        prop_assert_eq!(int(&resolved, "t.b"), Some(last));
        prop_assert_eq!(int(&resolved, "t.c"), Some(-1));
        let last_name = sorted.last().expect("one drop-in");
        prop_assert_eq!(
            &resolved.get("t.b").unwrap().origin,
            &Origin::File { path: locations.dropin_dir.join(last_name), line: 2 }
        );
    }
}

/// A directory under the system temp directory, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> TempDir {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("syntropy-config-{}-{name}-{n}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("temp dir");
        TempDir(path)
    }

    /// Locations rooted here: `sys/` for the system file and drop-ins, `user/` for the user
    /// file.
    fn locations(&self, mode: Mode) -> Locations {
        fs::create_dir_all(self.0.join("sys")).expect("mkdir");
        fs::create_dir_all(self.0.join("user")).expect("mkdir");
        Locations {
            os: Os::current(),
            mode,
            system_file: self.0.join("sys").join("config.toml"),
            dropin_dir: self.0.join("sys").join("config.d"),
            user_file: match mode {
                Mode::User => Some(self.0.join("user").join("config.toml")),
                Mode::Service => None,
            },
            explicit: None,
            dirs: Dirs {
                state: None,
                store: None,
                runtime: None,
                home: None,
            },
        }
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn an_explicit_file_replaces_the_other_three() {
    let root = TempDir::new("explicit");
    let mut locations = root.locations(Mode::User);
    fs::create_dir_all(&locations.dropin_dir).expect("mkdir");
    fs::write(&locations.system_file, "t.a = 10\n").expect("write");
    fs::write(locations.dropin_dir.join("50-x.toml"), "t.b = 20\n").expect("write");
    fs::write(locations.user_file.as_ref().unwrap(), "t.c = 30\n").expect("write");
    let explicit = root.path().join("explicit.toml");
    fs::write(&explicit, "t.d = 40\n").expect("write");
    locations.explicit = Some(syntropy_config::Explicit {
        path: explicit.clone(),
        from: Origin::Cli {
            flag: String::from("--config"),
        },
    });
    let scan = locations.scan().expect("scans");
    assert_eq!(scan.files(), vec![explicit.clone()]);
    let resolved = load(&REGISTRY, Os::Linux, &scan, &Env::default(), &Flags::new()).unwrap();
    assert_eq!(
        [
            int(&resolved, "t.a"),
            int(&resolved, "t.b"),
            int(&resolved, "t.c"),
            int(&resolved, "t.d")
        ],
        [Some(1), Some(2), Some(3), Some(40)]
    );
    assert_eq!(resolved.files(), [explicit]);

    locations.explicit.as_mut().unwrap().path = root.path().join("missing.toml");
    let problem = locations.scan().unwrap_err();
    assert!(
        problem
            .to_string()
            .starts_with("flag --config: the config file"),
        "{problem}"
    );
}

#[test]
fn a_system_service_reads_no_user_file() {
    let root = TempDir::new("service");
    let user = root.locations(Mode::User);
    fs::write(&user.system_file, "t.a = 10\n").expect("write");
    fs::write(user.user_file.as_ref().unwrap(), "t.a = 30\n").expect("write");
    let as_user = load(
        &REGISTRY,
        Os::Linux,
        &user.scan().unwrap(),
        &Env::default(),
        &Flags::new(),
    );
    assert_eq!(int(&as_user.unwrap(), "t.a"), Some(30));
    let service = root.locations(Mode::Service);
    let scan = service.scan().unwrap();
    assert_eq!(scan.files(), vec![service.system_file.clone()]);
    let as_service = load(&REGISTRY, Os::Linux, &scan, &Env::default(), &Flags::new());
    assert_eq!(int(&as_service.unwrap(), "t.a"), Some(10));
}

/// `config show --resolved` prints each value with its file and line, variable or flag.
#[test]
fn show_resolved_prints_the_origin_of_every_value() {
    let files = [
        ConfigFile::new(
            "/etc/syntropy/config.toml",
            "config_version = 1\nallow_approximate = true\n\n[t]\na = 7\ntoken = \"hunter2-canary\"\n",
        ),
        ConfigFile::new(
            "/etc/syntropy/config.d/50-distro.toml",
            "t.size = \"2GiB\"\n\n[policy.\"qwen3*\"]\nobjective = \"throughput\"\n",
        ),
    ];
    let env = env(&[
        ("SYNTROPY_T__B", "8"),
        ("SYNTROPY_T__NAME", "42"),
        ("SYNTROPY_HOME", "/srv/syntropy"),
        ("SYNTROPY_T__TYPO", "1"),
    ]);
    let mut flags = Flags::new();
    flags.set("t.c=9").unwrap();
    flags.push("--log-level", "t.d", "5");
    let resolved = resolve(&files, &env, &flags);
    let shown = command::show_resolved(&resolved);
    let expected = "\
# node keys: default < config file < environment < flags (2 files read)
node.state_dir = \"/srv/syntropy/state\"          # env SYNTROPY_HOME
t.a = 7                                         # /etc/syntropy/config.toml:5
t.b = 8                                         # env SYNTROPY_T__B
t.c = 9                                         # flag --set t.c
t.cores                                         # default, computed: from the probe
t.d = 5                                         # flag --log-level
t.name = \"42\"                                   # env SYNTROPY_T__NAME
t.size = \"2GiB\"                                 # /etc/syntropy/config.d/50-distro.toml:1
t.token = [redacted]                            # /etc/syntropy/config.toml:6

# cluster keys: the cluster's value; the default until the cluster sets one
allow_approximate = false                       # default

# proposed to the cluster by this node's file, environment or flags
allow_approximate = true                        # /etc/syntropy/config.toml:2
policy.\"qwen3*\" = { objective = \"throughput\" }  # /etc/syntropy/config.d/50-distro.toml:3

warning: env SYNTROPY_T__TYPO: unknown variable: `SYNTROPY_T__TYPO` names no config key (`t.typo`)
";
    assert_eq!(shown, expected, "\n{shown}");
    assert!(!shown.contains("hunter2"));
    assert!(!format!("{resolved:?}").contains("hunter2"));
}

/// A table of secrets, the shape RFC-0054 §5.2 gives `obs.otlp.headers`.
#[derive(Debug, Deserialize, ConfigSection)]
#[config(section = "p", scope = Node, owner = "RFC-0053")]
struct SecretMapSection {
    /// Headers sent with every export.
    #[config(reload = Hot, default = {})]
    headers: std::collections::BTreeMap<String, syntropy_config::Secret<String>>,
}

/// An exporter whose token is a secret field of a nested struct.
#[derive(Debug, Deserialize, HoldsSecret)]
struct Exporter {
    url: String,
    token: syntropy_config::Secret<String>,
}

/// One header of an array of tables, with a secret value.
#[derive(Debug, Deserialize, HoldsSecret)]
struct Header {
    name: String,
    value: syntropy_config::Secret<String>,
}

/// Secrets nested in a struct and in an array of tables.
#[derive(Debug, Deserialize, ConfigSection)]
#[config(section = "q", scope = Node, owner = "RFC-0053")]
struct NestedSecretSection {
    /// The exporter.
    #[config(reload = Hot, default = unset)]
    exporter: Option<Exporter>,
    /// Headers, one table each.
    #[config(reload = Hot, default = [])]
    headers: Vec<Header>,
}

/// Instance keys, which `[policy."<selector>"]` tables hold, one of them a secret.
#[derive(Debug, Deserialize, ConfigSection)]
#[config(section = "", scope = Instance, owner = "RFC-0054")]
struct PolicyKeys {
    /// What the plan optimizes.
    #[config(reload = Instance, default = "latency")]
    objective: String,
    /// A token the draft model's source needs.
    #[config(reload = Instance, default = unset)]
    draft_token: Option<syntropy_config::Secret<String>>,
}

/// Instance keys grouped under `approximate`, one of them a secret.
#[derive(Debug, Deserialize, ConfigSection)]
#[config(section = "approximate", scope = Instance, owner = "RFC-0054")]
struct ApproximateKeys {
    /// Lossy KV.
    #[config(reload = Instance, default = false)]
    kv_lossy: bool,
    /// A token for a knob's endpoint.
    #[config(reload = Instance, default = unset)]
    endpoint_token: Option<syntropy_config::Secret<String>>,
}

static SECRET_MAP: Registry = Registry::new(&[
    &SecretMapSection::SCHEMA,
    &NestedSecretSection::SCHEMA,
    &PolicyKeys::SCHEMA,
    &ApproximateKeys::SCHEMA,
]);

/// A key whose type holds a secret prints `[redacted]` whole, whatever its value looks like: an
/// entry named `env` or `file` in a table of secrets is a secret, not a `SecretSource`
/// (RFC-0054 §10.1, G7).
#[test]
fn a_table_of_secrets_never_prints() {
    let file = |text: &str| vec![ConfigFile::new("/c.toml", text)];
    let cases = [
        (
            file("p.headers = { authorization = \"Bearer canary-1\" }\n"),
            Env::default(),
        ),
        (file("p.headers = { env = \"canary-2\" }\n"), Env::default()),
        (
            file("p.headers = { file = \"canary-3\" }\n"),
            Env::default(),
        ),
        (file("[p.headers]\nenv = \"canary-4\"\n"), Env::default()),
        (
            Vec::new(),
            env(&[("SYNTROPY_P__HEADERS", "{ env = \"canary-5\" }")]),
        ),
    ];
    for (files, env) in cases {
        check_redacted(&files, &env, &Flags::new(), "p.headers");
        let resolved = resolve_secrets(&files, &env, &Flags::new());
        // The typed section still holds the secret, inside `Secret`.
        let section = resolved.section::<SecretMapSection>().expect("builds");
        assert_eq!(section.headers.len(), 1);
    }
    // The same map through `--set`.
    let mut flags = Flags::new();
    flags
        .set("p.headers={ env = \"canary-6\" }")
        .expect("well-formed");
    check_redacted(&[], &Env::default(), &flags, "p.headers");
}

/// A secret nested in a struct, or in an array of tables, is found by the derive's
/// `holds_secret` flag, not by the key's type name (`Option<Exporter>`, `Vec<Header>`).
#[test]
fn a_secret_nested_in_a_struct_or_an_array_of_tables_never_prints() {
    let file = |text: &str| vec![ConfigFile::new("/c.toml", text)];
    let exporter = "[q.exporter]\nurl = \"https://otlp\"\ntoken = \"canary-7\"\n";
    check_redacted(
        &file(exporter),
        &Env::default(),
        &Flags::new(),
        "q.exporter",
    );
    let env_exporter = env(&[(
        "SYNTROPY_Q__EXPORTER",
        "{ url = \"https://otlp\", token = \"canary-8\" }",
    )]);
    check_redacted(&[], &env_exporter, &Flags::new(), "q.exporter");
    let headers = "[[q.headers]]\nname = \"a\"\nvalue = \"canary-9\"\n";
    check_redacted(&file(headers), &Env::default(), &Flags::new(), "q.headers");
    let mut flags = Flags::new();
    flags
        .set("q.headers=[{ name = \"a\", value = \"canary-10\" }]")
        .expect("well-formed");
    check_redacted(&[], &Env::default(), &flags, "q.headers");
    // The typed section still holds the secret, inside `Secret`.
    let resolved = resolve_secrets(&file(headers), &Env::default(), &Flags::new());
    let section = resolved.section::<NestedSecretSection>().expect("builds");
    assert_eq!(section.headers[0].value.expose_for("test"), "canary-9");
}

/// A `policy.*` table is a proposal printed whole, so each entry is redacted by its instance
/// key's `holds_secret` flag. An entry no instance key names is an `unknown_key` error, which
/// never prints its value, since it may be a misspelt secret.
#[test]
fn a_secret_in_a_policy_table_never_prints() {
    let text = "[policy.\"qwen3*\"]\nobjective = \"throughput\"\ndraft_token = \"canary-p1\"\n\
                \n[policy.\"qwen3*\".approximate]\nkv_lossy = true\n\
                endpoint_token = \"canary-p3\"\n\n[policy.plain]\nobjective = \"latency\"\n";
    let files = vec![ConfigFile::new("/c.toml", text)];
    let resolved = resolve_secrets(&files, &Env::default(), &Flags::new());
    let shown = command::show_resolved(&resolved);
    assert!(!shown.contains("canary"), "{shown}");
    assert!(
        shown.contains(
            "policy.\"qwen3*\" = { approximate = { endpoint_token = [redacted], kv_lossy = true \
             }, draft_token = [redacted], objective = \"throughput\" }"
        ),
        "{shown}"
    );
    assert!(
        shown.contains("policy.plain = { objective = \"latency\" }"),
        "{shown}"
    );
    let proposal = |path: &str| {
        resolved
            .proposals()
            .iter()
            .find(|p| p.path == path)
            .expect(path)
    };
    assert_eq!(proposal("policy.\"qwen3*\"").value(), None);
    assert!(proposal("policy.plain").value().is_some());
    assert!(!format!("{resolved:?}").contains("canary"));

    // An entry no instance key names is rejected, and its error withholds the value.
    let text = "[policy.\"qwen3*\"]\nmystery = \"canary-p2\"\n";
    let files = vec![ConfigFile::new("/c.toml", text)];
    let layers = Layers {
        os: Os::Linux,
        files: &files,
        env: &Env::default(),
        flags: &Flags::new(),
    };
    let Err(rejected) = ResolvedConfig::resolve(&SECRET_MAP, layers) else {
        panic!("an unknown policy entry is an error");
    };
    assert_eq!(rejected.errors.len(), 1, "{:#?}", rejected.errors);
    assert_eq!(rejected.errors[0].code, Some(Code::UnknownKey));
    let rendered =
        diagnostic::render_all(&rejected.errors, &rejected.warnings, &files, &SECRET_MAP);
    assert!(rendered.contains("2 | mystery = [redacted]"), "{rendered}");
    assert!(!rendered.contains("canary"), "{rendered}");
    assert!(!format!("{rejected:?}").contains("canary"));
}

/// A secret-typed key given a value of the wrong shape: serde's own error for a map, a list or a
/// struct quotes the value (`invalid type: string "…"`). Phase 3 rejects such a value at load
/// (RFC-0054 §8), and neither the error nor its rendering carries it, from a file, `--set` or
/// the environment. `section()`'s own fallback for a value that reaches it is tested in
/// `resolve.rs`.
#[test]
fn a_wrong_shaped_secret_is_rejected_at_load_without_its_value() {
    let file = |text: &str| vec![ConfigFile::new("/c.toml", text)];
    let cases: Vec<(Vec<ConfigFile>, Env, Flags)> = vec![
        (
            file("p.headers = \"canary-s1\"\n"),
            Env::default(),
            Flags::new(),
        ),
        (
            file("q.exporter = \"canary-s2\"\n"),
            Env::default(),
            Flags::new(),
        ),
        (
            file("q.headers = [\"canary-s3\"]\n"),
            Env::default(),
            Flags::new(),
        ),
        (file("q.exporter = 7\n"), Env::default(), Flags::new()),
        (
            Vec::new(),
            env(&[("SYNTROPY_P__HEADERS", "canary-s4")]),
            Flags::new(),
        ),
        (Vec::new(), Env::default(), {
            let mut flags = Flags::new();
            flags.set("q.headers=[\"canary-s5\"]").expect("well-formed");
            flags
        }),
    ];
    for (files, env, flags) in cases {
        let layers = Layers {
            os: Os::Linux,
            files: &files,
            env: &env,
            flags: &flags,
        };
        let Err(rejected) = ResolvedConfig::resolve(&SECRET_MAP, layers) else {
            panic!("a wrong-shaped secret is rejected at load");
        };
        let codes: Vec<Option<Code>> = rejected.errors.iter().map(|p| p.code).collect();
        assert_eq!(codes, [Some(Code::WrongType)], "{:#?}", rejected.errors);
        let message = &rejected.errors[0].message;
        assert!(
            message.contains("the value is not shown, since the key holds a secret"),
            "{message}"
        );
        assert!(
            !message.contains('7') || !message.contains("integer"),
            "{message}"
        );
        let rendered =
            diagnostic::render_all(&rejected.errors, &rejected.warnings, &files, &SECRET_MAP);
        assert!(!rendered.contains("canary"), "{rendered}");
        assert!(!format!("{rejected:?}").contains("canary"));
    }
}

/// A malformed `--set` never echoes its value, whatever the separator.
#[test]
fn a_malformed_set_never_echoes_its_value() {
    let mut flags = Flags::new();
    for arg in [
        "p.headers:{ env = \"canary-11\" }",
        "q.exporter canary-12",
        "canary-13",
        "p.headers:Bearer canary-14=",
        "p.headers canary-15 = x",
    ] {
        let problem = flags.set(arg).expect_err("malformed");
        assert!(!problem.to_string().contains("canary"), "{problem}");
    }
}

fn resolve_secrets(files: &[ConfigFile], env: &Env, flags: &Flags) -> ResolvedConfig {
    ResolvedConfig::resolve(
        &SECRET_MAP,
        Layers {
            os: Os::Linux,
            files,
            env,
            flags,
        },
    )
    .expect("resolves")
}

/// `path` resolves and prints `[redacted]` in `config show`, its flag says it holds a secret,
/// and its canary appears in neither the output nor `Debug`.
fn check_redacted(files: &[ConfigFile], env: &Env, flags: &Flags, path: &str) {
    let resolved = resolve_secrets(files, env, flags);
    let shown = command::show_resolved(&resolved);
    assert!(!shown.contains("canary"), "{shown}");
    assert!(shown.contains(&format!("{path} = [redacted]")), "{shown}");
    let key = resolved.get(path).expect("a node key");
    assert!(key.is_set());
    assert!(key.key.holds_secret);
    assert_eq!(key.value(), None);
    assert!(!format!("{resolved:?}").contains("canary"));
}

#[test]
fn a_secret_in_its_file_form_prints_as_written() {
    let files = [ConfigFile::new(
        "/c.toml",
        "t.token = { file = \"/run/secrets/t\" }\n",
    )];
    let resolved = resolve(&files, &Env::default(), &Flags::new());
    assert_eq!(
        resolved.get("t.token").unwrap().shown().as_deref(),
        Some("{ file = \"/run/secrets/t\" }")
    );
}

#[test]
fn unknown_syntropy_variables_are_warnings() {
    let env = env(&[
        ("SYNTROPY_T__TYPO", "1"),
        ("SYNTROPY_NOPE", "1"),
        ("SYNTROPY_lower", "1"),
        ("SYNTROPY_MAX_TOKENS", "3"),
        // The fixed names, the other known names, a real key and a variable without the prefix
        // warn about nothing.
        ("SYNTROPY_CONFIG", ""),
        ("SYNTROPY_HOST", "http://10.0.0.2:7433"),
        ("SYNTROPY_API_KEY", "k"),
        ("SYNTROPY_FAULT_PLAN", "/tmp/plan.toml"),
        ("SYNTROPY_KERNELS", "prebuilt"),
        ("SYNTROPY_T__A", "5"),
        ("HF_TOKEN", "t"),
        ("PATH", "/bin"),
    ]);
    let resolved = resolve(&[], &env, &Flags::new());
    let warnings: Vec<String> = resolved
        .warnings()
        .iter()
        .map(ToString::to_string)
        .collect();
    assert_eq!(
        warnings,
        [
            "env SYNTROPY_MAX_TOKENS: ignored: `max_tokens` is a request key, and environment \
             variables set node keys",
            "env SYNTROPY_NOPE: unknown variable: `SYNTROPY_NOPE` names no config key (`nope`)",
            "env SYNTROPY_T__TYPO: unknown variable: `SYNTROPY_T__TYPO` names no config key \
             (`t.typo`)",
            "env SYNTROPY_lower: unknown variable: `SYNTROPY_lower` names no config key",
        ]
    );
    assert_eq!(int(&resolved, "t.a"), Some(5));
    let checked = Checked {
        files: Vec::new(),
        outcome: Ok(resolved),
    };
    let report = command::check(&checked, &REGISTRY);
    assert_eq!(report.exit, command::EXIT_OK);
    assert!(
        report
            .text
            .ends_with("config check: ok (0 files read, 4 warnings)\n"),
        "{}",
        report.text
    );
}

#[test]
fn layering_errors_name_the_file_line_variable_or_flag() {
    let files = [
        ConfigFile::new("/a.toml", "[t]\na = 1\nbogus = 2\n"),
        ConfigFile::new("/b.toml", "objective = \"throughput\"\n\n[t\n"),
        ConfigFile::new("/c.toml", "\n\n  objective = \"throughput\"\n"),
    ];
    let env = env(&[("SYNTROPY_T__A", "\u{1}")]);
    let mut flags = Flags::new();
    flags.set("t.missing=1").unwrap();
    flags.set("max_tokens=1").unwrap();
    let outcome = ResolvedConfig::resolve(
        &REGISTRY,
        Layers {
            os: Os::Linux,
            files: &files,
            env: &env,
            flags: &flags,
        },
    );
    let checked = Checked {
        files: files.to_vec(),
        outcome: outcome.clone(),
    };
    let report = command::check(&checked, &REGISTRY);
    assert_eq!(report.exit, command::EXIT_INVALID);
    let errors: Vec<String> = outcome
        .unwrap_err()
        .errors
        .iter()
        .map(ToString::to_string)
        .collect();
    // The layering errors, and the phase-3 type error for the environment's value.
    assert_eq!(errors.len(), 6, "{errors:#?}");
    assert_eq!(errors[0], "/a.toml:3: unknown key `t.bogus`");
    assert!(
        errors[1].starts_with("/b.toml:3:3: TOML syntax: "),
        "{}",
        errors[1]
    );
    assert!(
        errors[2].starts_with("/c.toml:3: `objective` is an instance key"),
        "{}",
        errors[2]
    );
    assert!(
        errors[3].starts_with("env SYNTROPY_T__A: `t.a`: invalid type"),
        "{}",
        errors[3]
    );
    assert_eq!(errors[4], "flag --set t.missing: unknown key `t.missing`");
    assert!(
        errors[5].starts_with("flag --set max_tokens: `max_tokens` is a request key"),
        "{}",
        errors[5]
    );
    assert!(
        report
            .text
            .ends_with("config check: 6 errors, 0 warnings\n"),
        "{}",
        report.text
    );
}

#[test]
fn string_typed_keys_take_bare_text() {
    let mut flags = Flags::new();
    flags.set("t.name='quoted'").unwrap();
    let resolved = resolve(&[], &env(&[("SYNTROPY_T__SIZE", "16GiB")]), &flags);
    assert_eq!(
        resolved.get("t.name").unwrap().value().cloned(),
        Some(Value::String(String::from("quoted")))
    );
    assert_eq!(
        resolved.get("t.size").unwrap().value().cloned(),
        Some(Value::String(String::from("16GiB")))
    );
    let resolved = resolve(&[], &env(&[("SYNTROPY_T__NAME", "true")]), &Flags::new());
    assert_eq!(
        resolved.get("t.name").unwrap().value().cloned(),
        Some(Value::String(String::from("true")))
    );
}

#[test]
fn syntropy_home_sets_the_state_dir_unless_its_own_variable_does() {
    let resolved = resolve(&[], &env(&[("SYNTROPY_HOME", "/srv/s")]), &Flags::new());
    let state = resolved.get("node.state_dir").unwrap();
    assert_eq!(
        state.value().cloned(),
        Some(Value::String(String::from("/srv/s/state")))
    );
    assert_eq!(
        state.origin,
        Origin::Env {
            var: String::from("SYNTROPY_HOME")
        }
    );

    let both = env(&[
        ("SYNTROPY_HOME", "/srv/s"),
        ("SYNTROPY_NODE__STATE_DIR", "/x"),
    ]);
    let state = resolve(&[], &both, &Flags::new())
        .get("node.state_dir")
        .cloned()
        .unwrap();
    assert_eq!(
        state.value().cloned(),
        Some(Value::String(String::from("/x")))
    );
    assert_eq!(
        state.origin,
        Origin::Env {
            var: String::from("SYNTROPY_NODE__STATE_DIR")
        }
    );
}

#[test]
fn a_section_builds_from_the_resolved_values() {
    let files = [ConfigFile::new("/c.toml", "t.a = 11\nt.size = \"4GiB\"\n")];
    let layers = Layers {
        os: Os::Linux,
        files: &files,
        env: &Env::default(),
        flags: &Flags::new(),
    };
    let resolved = ResolvedConfig::resolve(&REGISTRY, layers).unwrap();
    let error = resolved.section::<TSection>().unwrap_err();
    assert!(
        error
            .to_string()
            .contains("`t.cores` has a computed default"),
        "{error}"
    );

    let resolved = ResolvedConfig::resolve_with(&REGISTRY, layers, |key| {
        (key.path == "t.cores").then_some(Value::Integer(8))
    })
    .unwrap();
    assert_eq!(
        resolved.get("t.cores").unwrap().origin,
        Origin::Default {
            computed: Some("from the probe")
        }
    );
    let t = resolved.section::<TSection>().expect("builds");
    assert_eq!((t.a, t.b, t.cores), (11, 2, 8));
    assert_eq!(t.size, "4GiB".parse::<ByteSize>().unwrap());
    assert!(t.token.is_none());

    // A value below its key's minimum is rejected at load, by RFC-0054 §8's phase 3, before a
    // section is built from it.
    let files = [ConfigFile::new("/c.toml", "t.size = \"1KiB\"\n")];
    let rejected = ResolvedConfig::resolve_with(
        &REGISTRY,
        Layers {
            files: &files,
            ..layers
        },
        |_| Some(Value::Integer(8)),
    )
    .unwrap_err();
    assert_eq!(rejected.errors.len(), 1, "{rejected:?}");
    assert_eq!(rejected.errors[0].code, Some(Code::OutOfRange));
    assert_eq!(rejected.errors[0].key.as_deref(), Some("t.size"));
}

/// Readers never see a mix of two snapshots while a writer swaps them.
///
/// The readers start together behind a barrier, so every one is running before the first swap,
/// and the writer keeps swapping until each reader has done `MIN_READS` reads during the swaps.
/// The test does not depend on how the OS schedules the threads.
#[test]
fn snapshots_swap_whole() {
    const READERS: usize = 4;
    const MIN_SWAPS: i64 = 200;
    const MIN_READS: u64 = 50;
    const MAX_SWAPS: i64 = 1_000_000;
    fn snapshot(n: i64) -> ResolvedConfig {
        let text = format!("t.a = {n}\nt.b = {n}\nt.c = {n}\nt.d = {n}\n");
        resolve(
            &[ConfigFile::new("/c.toml", text)],
            &Env::default(),
            &Flags::new(),
        )
    }
    let shared = Arc::new(SharedConfig::new(snapshot(0)));
    let done = Arc::new(AtomicBool::new(false));
    let start = Arc::new(Barrier::new(READERS + 1));
    let reads: Arc<Vec<AtomicU64>> = Arc::new((0..READERS).map(|_| AtomicU64::new(0)).collect());
    let readers: Vec<_> = (0..READERS)
        .map(|i| {
            let (shared, done, start, reads) = (
                Arc::clone(&shared),
                Arc::clone(&done),
                Arc::clone(&start),
                Arc::clone(&reads),
            );
            std::thread::spawn(move || {
                start.wait();
                while !done.load(Ordering::Acquire) {
                    let config = shared.load();
                    let a = int(&config, "t.a");
                    for key in ["t.b", "t.c", "t.d"] {
                        assert_eq!(int(&config, key), a, "a reader saw a mixed snapshot");
                    }
                    reads[i].fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();
    start.wait();
    let mut n = 0;
    loop {
        n += 1;
        let old = shared.replace(snapshot(n));
        assert_eq!(int(&old, "t.a"), Some(n - 1));
        let every_reader_read = reads.iter().all(|r| r.load(Ordering::Relaxed) >= MIN_READS);
        // A reader stops only when `done` is set, or when it panics: stop swapping at once and
        // let the join below report its panic.
        let a_reader_died = readers.iter().any(std::thread::JoinHandle::is_finished);
        if (n >= MIN_SWAPS && every_reader_read) || a_reader_died {
            break;
        }
        assert!(
            n < MAX_SWAPS,
            "a reader made no progress in {MAX_SWAPS} swaps"
        );
    }
    done.store(true, Ordering::Release);
    for reader in readers {
        reader
            .join()
            .expect("a reader thread panicked: its message above says what it saw");
    }
    assert_eq!(int(&shared.load(), "t.d"), Some(n));
    assert!(n >= MIN_SWAPS, "the writer stopped after {n} swaps");
}

/// A file that cannot be read is reported beside every other file's errors, and the files that
/// can be read still render with their source lines (RFC-0054 §8: every error, not only the
/// first).
#[test]
fn an_unreadable_file_does_not_hide_the_other_files_errors() {
    let root = TempDir::new("unreadable");
    let locations = root.locations(Mode::Service);
    fs::create_dir_all(&locations.dropin_dir).expect("mkdir");
    fs::write(&locations.system_file, b"t.a = \"\xff\"\n").expect("write");
    let dropin = locations.dropin_dir.join("50-x.toml");
    fs::write(&dropin, "t.b = \"unterminated\n").expect("write");
    let scan = locations.scan().expect("scans");
    let codes = |errors: &[syntropy_config::Problem]| -> Vec<Option<Code>> {
        errors.iter().map(|p| p.code).collect()
    };
    let expected = [Some(Code::PathUnusable), Some(Code::Syntax)];

    let Err(rejected) = load(&REGISTRY, Os::Linux, &scan, &Env::default(), &Flags::new()) else {
        panic!("both files are errors");
    };
    assert_eq!(codes(&rejected.errors), expected, "{:#?}", rejected.errors);

    let checked = load_and_validate(
        &REGISTRY,
        Os::Linux,
        &scan,
        &Env::default(),
        &Flags::new(),
        |_| None,
        &[],
        &HostEnvironment,
    );
    assert_eq!(codes(checked.errors()), expected, "{:#?}", checked.errors());
    assert_eq!(
        checked.files.len(),
        1,
        "the readable drop-in is kept for rendering"
    );
    let rendered = checked.render(&REGISTRY);
    assert!(rendered.contains("error[config/0011]"), "{rendered}");
    assert!(rendered.contains("is not UTF-8"), "{rendered}");
    assert!(rendered.contains("error[config/0001]"), "{rendered}");
    assert!(
        rendered.contains(&format!("{}:1:", dropin.display())),
        "{rendered}"
    );
    assert!(rendered.contains("1 | t.b = "), "{rendered}");
}

#[test]
fn path_lists_the_layer_in_merge_order() {
    let root = TempDir::new("path");
    let locations = root.locations(Mode::User);
    fs::create_dir_all(&locations.dropin_dir).expect("mkdir");
    fs::write(locations.dropin_dir.join("50-distro.toml"), "").expect("write");
    fs::write(locations.dropin_dir.join("10-site.toml"), "").expect("write");
    fs::write(locations.user_file.as_ref().unwrap(), "").expect("write");
    let text = command::path(&locations, &locations.scan().unwrap());
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], "config files, in merge order (later wins):");
    assert!(lines[1].ends_with("(absent)"), "{text}");
    assert!(
        lines[2].ends_with("(10-site.toml, 50-distro.toml)"),
        "{text}"
    );
    assert!(lines[3].ends_with("(found)"), "{text}");
    assert!(
        lines[4].contains("(none; `--config` or SYNTROPY_CONFIG replaces 1-3)"),
        "{text}"
    );
    assert_eq!(lines[5], "directories:");
}
