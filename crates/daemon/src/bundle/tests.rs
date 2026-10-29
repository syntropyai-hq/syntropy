// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the diagnostic bundle (PROD-E9-T04; RFC-0053 §6).
//!
//! The "nothing leaks" canary plants a prompt, an output, an API key, a Hub token, invite text
//! and an address in every source a bundle reads, each where the real thing would sit: content
//! fields of log records and a content argument of a hot-thread record, secret-typed
//! configuration keys, cluster configuration values and a config patch, crash records' log lines
//! and panic message, a doctor finding's evidence, and the cluster state's and log's keys. It then
//! unpacks the bundle and searches every byte of every file.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use serde::Deserialize;
use syntropy_api::version::{Build, ProtocolMajors, SupportTier};
use syntropy_cluster::log::payload::{ConfigPatch, NodeInfo, Opaque};
use syntropy_cluster::log::{ClusterEvent, ClusterValue, MemberRecord, MemberStatus, genesis};
use syntropy_config::{
    ConfigFile, ConfigSection, Env, Environment, Flags, Layers, Os, PathUse, Registry,
    SecretSource, validate,
};
use syntropy_core::policy::{ConfigValue, KeyPath};
use syntropy_core::{Blake3, BuildId, BuildProfile, MonoTime, NodeId, TargetTriple};
use syntropy_obs::log::{CallsiteId, Level, LogRecord};
use syntropy_obs::trace::span::{SpanNameId, SpanRecord};

use super::*;
use crate::crash::tests::{StateDir, record};
use crate::crash::{CrashCause, LogTail};
use crate::doctor::{Area, CheckResult, Finding, JSON_SCHEMA, Needs};
use crate::logging::Field;
use crate::logging::format::WallClock;

const PROMPT: &str = "cnry-prompt-Zq81";
const OUTPUT: &str = "cnry-output-Lw40";
const API_KEY: &str = "syk_cnryKEY0n3TwoThree";
const HUB_TOKEN: &str = "hf_cnryTOKENxx9";
const INVITE: &str = "cnry-invite-Pp27";
/// A content argument of a hot-thread record: it must not appear as a number either.
const CONTENT_ARG: u64 = 0x0C0F_FEE5_1234;
const ADDR: &str = "203.0.113.77";
/// An IPv6 address, planted inside dotted keys and after a `word:` prefix (review B1).
const ADDR6: &str = "2001:db8::77";
/// A zero-padded IPv4 address (review B1).
const ADDR_PADDED: &str = "203.000.113.088";
const URL_HOST: &str = "relay.cnry.example.org";

fn canaries() -> Vec<String> {
    vec![
        PROMPT.into(),
        OUTPUT.into(),
        API_KEY.into(),
        HUB_TOKEN.into(),
        INVITE.into(),
        CONTENT_ARG.to_string(),
    ]
}

// ---- Sources ------------------------------------------------------------------------------------

/// A section with secret-typed keys, as `[hub] token` and `[api] key` are.
#[allow(
    dead_code,
    reason = "the keys are read through the resolved configuration only"
)]
#[derive(Debug, Deserialize, ConfigSection)]
#[config(section = "bundletest", scope = Node, owner = "RFC-0054")]
struct BundleTest {
    /// A Hub token.
    #[config(reload = Hot, default = unset)]
    token: Option<SecretSource>,
    /// An API key.
    #[config(reload = Hot, default = unset)]
    key: Option<SecretSource>,
    /// A peer, which is an address.
    #[config(reload = Hot, default = unset)]
    peer: Option<String>,
}

static REGISTRY: Registry = Registry::new(&[&BundleTest::SCHEMA]);

struct Host;

impl Environment for Host {
    fn usable(&self, _: &Path, _: PathUse) -> Result<(), String> {
        Ok(())
    }

    fn readable_by_others(&self, _: &Path) -> Option<bool> {
        Some(false)
    }
}

fn config() -> ResolvedConfig {
    let text = format!(
        "[bundletest]\ntoken = \"{HUB_TOKEN}\"\nkey = \"{API_KEY}\"\npeer = \"{ADDR}:7433\"\n"
    );
    let files = [ConfigFile::new("/etc/syntropy/config.toml", text)];
    let env = Env::from_pairs(Os::Linux, std::iter::empty::<(&str, &str)>());
    let flags = Flags::new();
    let config = ResolvedConfig::resolve(
        &REGISTRY,
        Layers {
            os: Os::Linux,
            files: &files,
            env: &env,
            flags: &flags,
        },
    )
    .expect("resolves");
    validate(&REGISTRY, config, &[], &Host).expect("valid")
}

fn version() -> VersionInfo {
    VersionInfo {
        build: Build {
            build_id: BuildId {
                version: "0.0.0-test".parse().unwrap(),
                commit: [0x1c; 20],
                profile: BuildProfile::Dev,
                target: TargetTriple("x86_64-unknown-linux-gnu".into()),
                toolchain: Blake3([0x70; 32]),
            },
            features: Vec::new(),
            id: [0x5a; 32],
        },
        protocols: ProtocolMajors::current(),
        support_tier: SupportTier::Tier1,
    }
}

fn node(n: u8) -> NodeId {
    NodeId([n; 32])
}

fn cluster_state() -> ClusterState {
    let mut state = ClusterState::new([1; 32]);
    state.members.insert(
        node(2),
        MemberRecord {
            info: NodeInfo { id: node(2) },
            status: MemberStatus::Alive,
            incarnation: 3,
        },
    );
    // A key that names an endpoint, and a value holding an invite and a key.
    state.config.keys.insert(
        KeyPath(format!("p2p.peers.{ADDR}:7433")),
        ClusterValue {
            value: ConfigValue::String(format!("{INVITE} {API_KEY}")),
            set_at: (4, 1),
        },
    );
    // An IPv6 peer key, the same shape as the IPv4 one.
    state.config.keys.insert(
        KeyPath(format!("p2p.peers.{ADDR6}.port")),
        ClusterValue {
            value: ConfigValue::Integer(7433),
            set_at: (5, 1),
        },
    );
    // An instance whose policy carries an instance-scope key holding a key (review N1).
    let mut policy = syntropy_core::InstancePolicy::default();
    policy.domain.insert(
        KeyPath("solver.hint".into()),
        ConfigValue::String(API_KEY.into()),
    );
    state.instances.insert(
        syntropy_core::InstanceId(9),
        syntropy_cluster::log::InstanceRecord {
            model: syntropy_core::ModelId(Blake3([6; 32])),
            policy,
            current: None,
            retiring: Vec::new(),
            runners: BTreeMap::new(),
            routable: None,
            requested_at: 1,
        },
    );
    state
}

fn cluster_log() -> Vec<LogEntry> {
    let g = genesis(&[1; 32]);
    let first = LogEntry::append(
        &g,
        1,
        1,
        1_800_000_000_000,
        None,
        ClusterEvent::NodeAdmitted {
            info: NodeInfo { id: node(2) },
        },
    )
    .unwrap();
    let second = LogEntry::append(
        &first.chain(),
        2,
        1,
        1_800_000_000_500,
        None,
        ClusterEvent::ClusterConfigChanged {
            patch: ConfigPatch {
                set: vec![(
                    KeyPath("hub.token".into()),
                    ConfigValue::String(HUB_TOKEN.into()),
                )],
                unset: vec![
                    KeyPath(format!("relay/https://{URL_HOST}:8443/r")),
                    KeyPath(format!("peer.{ADDR_PADDED}")),
                ],
                origin: Opaque(ciborium::Value::Text(INVITE.into())),
            },
        },
    )
    .unwrap();
    vec![first, second]
}

const CALLSITES: &[Callsite] = &[Callsite {
    id: CallsiteId(1),
    level: Level::Info,
    target: "syntropy_exec::step",
    format: "step {n} sampled {tok}",
    fields: &["n", "tok"],
    content: 0b10,
    key_fields: 0,
    file: "exec/step.rs",
    line: 1,
}];

fn flight(entries: usize) -> FlightRecorder {
    let mut recorder = FlightRecorder::new(64 << 20);
    for i in 0..entries as u64 {
        recorder.push(Entry::Log(LogRecord {
            callsite: CallsiteId(1),
            level: Level::Info,
            at: MonoTime(i),
            args: [i, CONTENT_ARG, 0, 0],
        }));
        recorder.push(Entry::Span(SpanRecord {
            trace_id: [3; 16],
            span_id: [4; 8],
            parent_id: [0; 8],
            start: MonoTime(i),
            dur: 1_000,
            name: SpanNameId(1),
            attr_count: 1,
            attrs: [9, 0, 0, 0],
        }));
    }
    recorder
}

fn doctor_report() -> Report {
    Report {
        schema: JSON_SCHEMA,
        offline: true,
        results: vec![CheckResult {
            id: "STO-003",
            area: Area::Storage,
            needs: Needs::Offline,
            finding: Finding::warn("the store is on a network filesystem").with_evidence(
                "mount",
                syntropy_api::figure::Figure::measured(
                    format!("nfs://{ADDR}/models"),
                    "mount table",
                ),
            ),
        }],
    }
}

/// A crash store with one runner record by signal and one daemon panic, whose log lines carried
/// content and an address.
fn crashes(state: &StateDir) -> CrashStore {
    let store = CrashStore::new(&state.0);
    let tail = Arc::new(LogTail::new());
    let wall = WallClock::new(1_800_000_000_000, MonoTime(0));
    tail.push(
        &crate::logging::LogEvent::new(
            MonoTime(1),
            Level::Warn,
            "syntropy_net::link",
            format!("dialling {ADDR}:7433 failed; remote_addr:{ADDR6} via {ADDR_PADDED}"),
        )
        .with(Field::content(
            "prompt",
            crate::logging::Value::Str(PROMPT.into()),
        ))
        .with(Field::content(
            "output",
            crate::logging::Value::Str(OUTPUT.into()),
        )),
        &wall,
    );
    let mut killed = record(
        1_800_000_000_100,
        CrashCause::Signal {
            signal: 9,
            name: Some("SIGKILL".into()),
        },
    );
    killed.last_logs = tail.lines();
    store.write(&killed).unwrap();
    let mut panicked = record(
        1_800_000_000_200,
        CrashCause::Panic {
            message: format!("lost the link to {ADDR}"),
            location: Some("net/link.rs:1:1".into()),
            backtrace: "0: main".into(),
        },
    );
    panicked.process = crate::crash::CrashProcess::Daemon { pid: 1 };
    store.write(&panicked).unwrap();
    store
}

fn unpack(bundle: &Bundle) -> BTreeMap<String, Vec<u8>> {
    let archive = zstd::decode_all(bundle.bytes()).unwrap();
    tar::read(&archive)
        .unwrap()
        .into_iter()
        .map(|f| (f.name, f.bytes))
        .collect()
}

fn assert_absent(files: &BTreeMap<String, Vec<u8>>, needle: &str) {
    for (name, bytes) in files {
        assert!(
            !String::from_utf8_lossy(bytes).contains(needle),
            "{name} holds {needle}:\n{}",
            String::from_utf8_lossy(bytes)
        );
    }
}

fn any_holds(files: &BTreeMap<String, Vec<u8>>, needle: &str) -> bool {
    files
        .values()
        .any(|b| String::from_utf8_lossy(b).contains(needle))
}

fn build_all(options: &BundleOptions) -> (Bundle, StateDir) {
    let state = StateDir::new("bundle");
    let store = crashes(&state);
    let config = config();
    let cluster = cluster_state();
    let log = cluster_log();
    let recorder = flight(3);
    let report = doctor_report();
    let version = version();
    let sources = Sources {
        build: Some(&version),
        config: Some(&config),
        cluster_state: Some(&cluster),
        cluster_log: &log,
        flight: Some(&recorder),
        callsites: CALLSITES,
        doctor: Some(&report),
        crashes: Some(&store),
    };
    (build(&sources, options).unwrap(), state)
}

// ---- The criteria -------------------------------------------------------------------------------

/// No prompt, output, key, token or invite text in any bundle, and no address unless
/// `--with-addresses` is given.
#[test]
fn nothing_leaks_into_a_bundle() {
    let (bundle, _state) = build_all(&BundleOptions::at(1_800_000_001_000));
    let files = unpack(&bundle);
    assert_eq!(
        files.keys().map(String::as_str).collect::<Vec<_>>(),
        [
            "MANIFEST.txt",
            "build.json",
            "cluster-log.jsonl",
            "cluster-state.json",
            "config.txt",
            "crashes/00000001800000000100-4242-0.json",
            "crashes/00000001800000000200-1-1.json",
            "doctor.json",
            "flight-recorder.jsonl",
        ]
    );
    for canary in canaries() {
        assert_absent(&files, &canary);
    }
    assert_absent(&files, ADDR);
    assert_absent(&files, ADDR6);
    assert_absent(&files, ADDR_PADDED);
    assert_absent(&files, URL_HOST);
    // What was kept is still there.
    let text = |name: &str| String::from_utf8(files[name].clone()).unwrap();
    assert!(
        text("config.txt").contains("bundletest.token = [redacted]"),
        "{}",
        text("config.txt")
    );
    assert!(text("config.txt").contains("bundletest.peer = \"<addr-"));
    assert!(text("cluster-state.json").contains("p2p.peers.<addr-"));
    assert!(text("cluster-log.jsonl").contains("relay/<url-"));
    assert!(text("flight-recorder.jsonl").contains("step 0 sampled [content]"));
    assert!(text("crashes/00000001800000000100-4242-0.json").contains("dialling <addr-"));
    assert!(text("doctor.json").contains("<url-"));
    assert!(text("build.json").contains("0.0.0-test"));
}

#[test]
fn with_addresses_keeps_addresses_and_nothing_else_leaks() {
    let mut options = BundleOptions::at(1_800_000_001_000);
    options.with_addresses = true;
    let (bundle, _state) = build_all(&options);
    let files = unpack(&bundle);
    for canary in canaries() {
        assert_absent(&files, &canary);
    }
    assert!(any_holds(&files, ADDR));
    assert!(any_holds(&files, ADDR6));
    assert!(any_holds(&files, ADDR_PADDED));
    assert!(any_holds(&files, URL_HOST));
    let manifest = bundle.manifest();
    assert!(manifest.contains("addresses: kept (--with-addresses)"));
    assert!(
        manifest.contains("addresses replaced in (file: field):\n  none\n"),
        "{manifest}"
    );
}

/// Every file is listed with its size and SHA-256, and the hashes verify.
#[test]
fn the_manifest_hashes_verify() {
    let (bundle, _state) = build_all(&BundleOptions::at(1_800_000_001_000));
    let names = verify(bundle.bytes()).unwrap();
    assert_eq!(names.len(), 9);
    let files = unpack(&bundle);
    assert_eq!(files["MANIFEST.txt"], bundle.manifest().as_bytes());
    for (name, bytes) in &files {
        if name == MANIFEST {
            continue;
        }
        let line = format!("{}  {}  {name}", sha256_hex(bytes), bytes.len());
        assert!(bundle.manifest().contains(&line), "{line}");
    }
    // A changed file, or one the manifest does not list, fails.
    let tampered = |edit: &dyn Fn(&mut Vec<TarFile>)| {
        let mut entries = tar::read(&zstd::decode_all(bundle.bytes()).unwrap()).unwrap();
        edit(&mut entries);
        zstd::encode_all(&tar::write(&entries, 0)[..], 3).unwrap()
    };
    let changed = tampered(&|e| e[1].bytes.push(b' '));
    assert!(verify(&changed).unwrap_err().contains("does not match"));
    let extra = tampered(&|e| {
        e.push(TarFile {
            name: "extra.txt".into(),
            bytes: b"x".to_vec(),
        });
    });
    assert_eq!(
        verify(&extra).unwrap_err(),
        "extra.txt is in the bundle but not listed"
    );
    // The bundle's own digest is the one the command prints.
    assert_eq!(bundle.sha256_hex(), sha256_hex(bundle.bytes()));
    let message = bundle.finish_message(Path::new("/tmp/b.tar.zst"));
    assert!(message.contains(&bundle.sha256_hex()));
    assert!(message.contains("read MANIFEST.txt inside it before you share it"));
}

/// `MANIFEST.txt` lists where addresses and secrets were replaced, and what was left out.
#[test]
fn the_manifest_lists_the_redacted_fields() {
    let (bundle, _state) = build_all(&BundleOptions::at(1_800_000_001_000));
    let manifest = bundle.manifest();
    for line in [
        "  cluster-state.json: config.keys.* (key)",
        "  cluster-log.jsonl: event.ClusterConfigChanged.patch.unset",
        "  config.txt: (text)",
        "  crashes/00000001800000000100-4242-0.json: last_logs",
        "  crashes/00000001800000000200-1-1.json: cause",
        "  doctor.json: results.*.evidence",
    ] {
        assert!(
            manifest.contains(&format!("{line}\n")),
            "addresses: {line} missing:\n{manifest}"
        );
    }
    for line in [
        "  cluster-state.json: config.keys.*.value",
        "  cluster-log.jsonl: event.ClusterConfigChanged.patch.set.*.1",
        "  config.txt: every secret-typed key (config show)",
    ] {
        assert!(
            manifest.contains(&format!("{line}\n")),
            "secrets: {line} missing"
        );
    }
    // The opaque config-patch origin is left out and named.
    assert!(manifest.contains("  cluster-log.jsonl: event.ClusterConfigChanged.patch.origin\n"));
    assert!(manifest.contains("addresses: replaced by placeholders"));
    assert!(manifest.contains("never included, whatever the flags"));
}

/// The flight recorder is truncated first, oldest first, and the result fits the cap.
#[test]
fn the_size_cap_truncates_the_flight_recorder_first() {
    let recorder = flight(20_000);
    let log = cluster_log();
    let sources = Sources {
        cluster_log: &log,
        flight: Some(&recorder),
        callsites: CALLSITES,
        ..Sources::default()
    };
    let mut options = BundleOptions::at(1_800_000_001_000);
    options.max_size = MANIFEST_ROOM + (512 << 10);
    let bundle = build(&sources, &options).unwrap();
    assert!(bundle.size() <= options.max_size);
    verify(bundle.bytes()).unwrap();
    let manifest = bundle.manifest();
    assert!(
        manifest.contains("flight-recorder.jsonl: the oldest "),
        "{manifest}"
    );
    assert!(manifest.contains(" of 40000 entries left out"));
    assert!(!manifest.contains("cluster-log.jsonl: the oldest"));
    let files = unpack(&bundle);
    let flight = String::from_utf8(files["flight-recorder.jsonl"].clone()).unwrap();
    // The newest entries are the ones kept.
    assert!(flight.contains("step 19999 sampled [content]"));
    assert!(!flight.contains("\"step 0 sampled [content]\""));
    assert_eq!(
        String::from_utf8(files["cluster-log.jsonl"].clone())
            .unwrap()
            .lines()
            .count(),
        2
    );

    options.max_size = 1000;
    assert!(matches!(
        build(&sources, &options),
        Err(BundleError::TooLarge { cap: 1000, .. })
    ));
}

#[test]
fn a_missing_source_is_listed_as_not_collected() {
    let bundle = build(&Sources::default(), &BundleOptions::at(1_800_000_001_000)).unwrap();
    let manifest = bundle.manifest();
    for line in [
        "build.json: no build information was given",
        "config.txt: no configuration was given",
        "cluster-state.json: this node has no cluster state",
        "OS facts, the resource graph and capability profiles: the probe's export",
    ] {
        assert!(manifest.contains(line), "{line}");
    }
    assert_eq!(
        verify(bundle.bytes()).unwrap(),
        ["MANIFEST.txt", "cluster-log.jsonl"]
    );
}

#[test]
fn a_bundle_is_written_to_a_new_owner_only_file() {
    let state = StateDir::new("write");
    let bundle = build(&Sources::default(), &BundleOptions::at(1_800_000_001_000)).unwrap();
    let path = state.0.join("bundle.tar.zst");
    bundle.write_new(&path).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), bundle.bytes());
    assert!(syntropy_io::os::is_owner_only(&path).unwrap());
    assert_eq!(
        bundle.write_new(&path).unwrap_err().kind(),
        io::ErrorKind::AlreadyExists
    );
}

/// A map key that is not a string, such as a `NodeId`, becomes hex.
#[test]
fn non_string_map_keys_become_text() {
    let state = cluster_state();
    let value = to_json(&state, "cluster-state.json").unwrap();
    let members = value["members"].as_object().unwrap();
    assert_eq!(
        members.keys().collect::<Vec<_>>(),
        ["02".repeat(32).as_str()]
    );
}

/// The review of #1967, B2: an `IpAddr`, a `SocketAddr` and an `IpNet` in a kept subtree come
/// out as text through the bundle's serializer, so the projection replaces them. Through a
/// compact serializer they would be integer arrays no text scan finds.
#[test]
fn typed_addresses_in_a_kept_subtree_are_redacted() {
    #[derive(serde::Serialize)]
    struct Interface {
        ip: std::net::IpAddr,
        endpoint: std::net::SocketAddr,
        net: syntropy_core::resource::iface::IpNet,
    }
    let record = Interface {
        ip: "203.0.113.77".parse().unwrap(),
        endpoint: "[2001:db8::5]:9000".parse().unwrap(),
        net: syntropy_core::resource::iface::IpNet {
            addr: "198.51.100.0".parse().unwrap(),
            prefix_len: 24,
            scope_id: None,
        },
    };
    let value = to_json(&record, "cluster-state.json").unwrap();
    let mut redactor = Redactor::new(false);
    let mut omitted = Omitted::default();
    let kept = project(value, &Shape::Any, "f", "", &mut redactor, &mut omitted);
    let text = kept.to_string();
    for needle in [
        "203.0.113.77",
        "2001:db8::5",
        "198.51.100.0",
        "203,0,113,77",
        "[32,1,13,184",
    ] {
        assert!(!text.contains(needle), "{needle} in {text}");
    }
    assert_eq!(kept["ip"], "<addr-1>");
    assert_eq!(kept["endpoint"], "<addr-2>");
    assert_eq!(kept["net"]["addr"], "<addr-3>");
    assert_eq!(kept["net"]["prefix_len"], 24);
}
