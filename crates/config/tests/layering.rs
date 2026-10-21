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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use proptest::prelude::*;
use serde::Deserialize;
use syntropy_config::{
    ByteSize, ConfigFile, ConfigSection, Dirs, Env, Flags, Layers, Locations, Mode, Origin, Os,
    Registry, ResolvedConfig, SecretSource, SharedConfig, Value, command, load, order_dropins,
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
        .and_then(|r| r.value.as_ref())
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
            prop_assert_eq!(got.value.as_ref().and_then(Value::as_integer), Some(value), "{}", path);
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
        use std::sync::atomic::AtomicU64;
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
        // The fixed names, a real key and a variable without the prefix warn about nothing.
        ("SYNTROPY_CONFIG", ""),
        ("SYNTROPY_HOST", "http://10.0.0.2:7433"),
        ("SYNTROPY_API_KEY", "k"),
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
    let report = command::check(&Ok(resolved));
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
    let report = command::check(&outcome);
    assert_eq!(report.exit, command::EXIT_INVALID);
    let errors: Vec<String> = outcome
        .unwrap_err()
        .errors
        .iter()
        .map(ToString::to_string)
        .collect();
    assert_eq!(errors.len(), 5, "{errors:#?}");
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
    assert_eq!(errors[3], "flag --set t.missing: unknown key `t.missing`");
    assert!(
        errors[4].starts_with("flag --set max_tokens: `max_tokens` is a request key"),
        "{}",
        errors[4]
    );
    assert!(
        report
            .text
            .ends_with("config check: 5 errors, 0 warnings\n"),
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
        resolved.get("t.name").unwrap().value,
        Some(Value::String(String::from("quoted")))
    );
    assert_eq!(
        resolved.get("t.size").unwrap().value,
        Some(Value::String(String::from("16GiB")))
    );
    let resolved = resolve(&[], &env(&[("SYNTROPY_T__NAME", "true")]), &Flags::new());
    assert_eq!(
        resolved.get("t.name").unwrap().value,
        Some(Value::String(String::from("true")))
    );
}

#[test]
fn syntropy_home_sets_the_state_dir_unless_its_own_variable_does() {
    let resolved = resolve(&[], &env(&[("SYNTROPY_HOME", "/srv/s")]), &Flags::new());
    let state = resolved.get("node.state_dir").unwrap();
    assert_eq!(
        state.value,
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
    assert_eq!(state.value, Some(Value::String(String::from("/x"))));
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

    let files = [ConfigFile::new("/c.toml", "t.size = \"1KiB\"\n")];
    let resolved = ResolvedConfig::resolve_with(
        &REGISTRY,
        Layers {
            files: &files,
            ..layers
        },
        |_| Some(Value::Integer(8)),
    )
    .unwrap();
    assert!(resolved.section::<TSection>().is_err());
}

/// Readers never see a mix of two snapshots while a writer swaps them.
#[test]
fn snapshots_swap_whole() {
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
    let readers: Vec<_> = (0..4)
        .map(|_| {
            let shared = Arc::clone(&shared);
            let done = Arc::clone(&done);
            std::thread::spawn(move || {
                let mut seen = 0u64;
                while !done.load(Ordering::Relaxed) {
                    let config = shared.load();
                    let a = int(&config, "t.a");
                    for key in ["t.b", "t.c", "t.d"] {
                        assert_eq!(int(&config, key), a);
                    }
                    seen += 1;
                }
                seen
            })
        })
        .collect();
    for n in 1..=200 {
        let old = shared.replace(snapshot(n));
        assert_eq!(int(&old, "t.a"), Some(n - 1));
    }
    done.store(true, Ordering::Relaxed);
    for reader in readers {
        assert!(reader.join().expect("no reader saw a mixed snapshot") > 0);
    }
    assert_eq!(int(&shared.load(), "t.d"), Some(200));
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
