// SPDX-License-Identifier: MIT OR Apache-2.0

//! The distributed areas' configuration keys in RFC-0054's schema (DIST-E18-T01).
//!
//! The acceptance criteria this file checks, through the library (the `syntropy` binary's
//! `config` subcommands are PROD-E6-T23's):
//! - `config show --resolved` lists every key with its default: a registry of every section of
//!   RFC-0030 §9, RFC-0031 §11, RFC-0032 §9, RFC-0033 §12 and RFC-0034 §14, resolved with no
//!   layers, printed by `command::show_resolved`;
//! - unknown and mis-scoped keys fail validation: RFC-0054 §8's `unknown_key` (config/0003) and
//!   `scope_mismatch` (config/0009).
//!
//! It also checks that the combined registry is consistent ([`Registry::check`]) and that every
//! key has the RFC's scope and owner.

use syntropy_cluster::config::SECTIONS as CLUSTER_SECTIONS;
use syntropy_config::{
    ConfigFile, DefaultSpec, Env, Flags, Layers, Os, Registry, Reload, ResolvedConfig, Scope,
    SectionSchema, command,
};
use syntropy_net::config::SECTIONS as NET_SECTIONS;

/// The registry a binary that loads every distributed section builds: `net`'s sections, then
/// `cluster`'s. A `static`, so two sections claiming one prefix would fail this build.
const ALL: [&SectionSchema; 29] = [
    NET_SECTIONS[0],
    NET_SECTIONS[1],
    NET_SECTIONS[2],
    NET_SECTIONS[3],
    NET_SECTIONS[4],
    CLUSTER_SECTIONS[0],
    CLUSTER_SECTIONS[1],
    CLUSTER_SECTIONS[2],
    CLUSTER_SECTIONS[3],
    CLUSTER_SECTIONS[4],
    CLUSTER_SECTIONS[5],
    CLUSTER_SECTIONS[6],
    CLUSTER_SECTIONS[7],
    CLUSTER_SECTIONS[8],
    CLUSTER_SECTIONS[9],
    CLUSTER_SECTIONS[10],
    CLUSTER_SECTIONS[11],
    CLUSTER_SECTIONS[12],
    CLUSTER_SECTIONS[13],
    CLUSTER_SECTIONS[14],
    CLUSTER_SECTIONS[15],
    CLUSTER_SECTIONS[16],
    CLUSTER_SECTIONS[17],
    CLUSTER_SECTIONS[18],
    CLUSTER_SECTIONS[19],
    CLUSTER_SECTIONS[20],
    CLUSTER_SECTIONS[21],
    CLUSTER_SECTIONS[22],
    CLUSTER_SECTIONS[23],
];
static REGISTRY: Registry = Registry::new(&ALL);

use Scope::{Cluster, Instance, Node};

/// Every key of the five RFC tables (and RFC-0033 §8.2's `runner.*`, which RFC-0033 §12 names),
/// with the RFC's scope and the default as the schema records it: a value as TOML writes it,
/// or the rule of a computed default.
const EXPECTED: &[(&str, Scope, &str, &str)] = &[
    // RFC-0030 §9.
    ("cluster.namespace", Node, "RFC-0030", "\"default\""),
    (
        "cluster.name",
        Cluster,
        "RFC-0030",
        "computed: \"<hostname>'s cluster\"",
    ),
    ("cluster.lan_admission", Cluster, "RFC-0030", "\"confirm\""),
    ("cluster.static_peers", Node, "RFC-0030", "[]"),
    ("discovery.port", Node, "RFC-0030", "7434"),
    (
        "discovery.mechanisms",
        Node,
        "RFC-0030",
        "[\"mdns\", \"v6_multicast\", \"v4_broadcast\"]",
    ),
    (
        "discovery.ignore_interfaces",
        Node,
        "RFC-0030",
        "computed: container and VM bridges, and loopback",
    ),
    ("discovery.beacon_scope", Node, "RFC-0030", "\"home\""),
    ("net.listen_port", Node, "RFC-0030", "7435"),
    ("net.relay_mode", Node, "RFC-0030", "\"auto\""),
    ("net.relays", Node, "RFC-0030", "[]"),
    ("net.bulk_on_metered", Node, "RFC-0030", "false"),
    ("swim.period", Cluster, "RFC-0030", "\"1s\""),
    ("swim.wan_period", Cluster, "RFC-0030", "\"2s\""),
    ("swim.miss_probe", Cluster, "RFC-0030", "2"),
    ("swim.miss_suspect", Cluster, "RFC-0030", "3"),
    ("swim.wan_miss_suspect", Cluster, "RFC-0030", "4"),
    ("swim.suspect_min", Cluster, "RFC-0030", "\"2s\""),
    ("swim.suspect_max", Cluster, "RFC-0030", "\"4s\""),
    ("swim.wan_suspect_min", Cluster, "RFC-0030", "\"10s\""),
    ("swim.wan_suspect_max", Cluster, "RFC-0030", "\"20s\""),
    ("swim.k_indirect", Cluster, "RFC-0030", "3"),
    ("swim.full_mesh_max", Cluster, "RFC-0030", "16"),
    ("swim.retransmit_mult", Cluster, "RFC-0030", "3"),
    ("swim.resume_detect", Node, "RFC-0030", "\"5s\""),
    (
        "membership.flap_threshold",
        Cluster,
        "RFC-0030",
        "{ count = 3, window = \"10m\" }",
    ),
    (
        "membership.flap_backoff",
        Cluster,
        "RFC-0030",
        "{ initial = \"5m\", max = \"1h\" }",
    ),
    ("membership.restart_grace", Cluster, "RFC-0030", "\"30s\""),
    (
        "power.inhibit_idle_sleep",
        Node,
        "RFC-0030",
        "computed: while_serving on mains power, never on battery",
    ),
    ("invite.default_expiry", Cluster, "RFC-0030", "\"24h\""),
    ("invite.default_uses", Cluster, "RFC-0030", "1"),
    ("invite.max_expiry", Cluster, "RFC-0030", "\"7d\""),
    (
        "pairing.window_default",
        Cluster,
        "RFC-0030",
        "{ for = \"10m\", joins = 4, include_wifi = false }",
    ),
    // RFC-0031 §11.
    ("net.max_frame", Node, "RFC-0031", "\"64MiB\""),
    ("net.blob_frame", Node, "RFC-0031", "\"1MiB\""),
    ("net.bulk_p99_ratio", Cluster, "RFC-0031", "1.2"),
    ("net.bulk_floor", Cluster, "RFC-0031", "0.05"),
    (
        "link.connect_timeout",
        Cluster,
        "RFC-0031",
        "computed: 2s on LAN links, 10s on WAN links",
    ),
    (
        "link.failover_timeout",
        Cluster,
        "RFC-0031",
        "computed: 2s on LAN links, 10s on WAN links",
    ),
    (
        "link.demote_after",
        Cluster,
        "RFC-0031",
        "{ count = 3, window = \"60s\" }",
    ),
    ("link.ack_timeout_min", Cluster, "RFC-0031", "\"200us\""),
    (
        "link.p99_weight",
        Cluster,
        "RFC-0031",
        "computed: 1.0 on Wi-Fi or relayed links, 0.25 otherwise",
    ),
    ("quic.congestion", Node, "RFC-0031", "\"cubic\""),
    ("quic.lan_initial_window", Node, "RFC-0031", "\"256KiB\""),
    ("quic.max_conn_window", Node, "RFC-0031", "\"64MiB\""),
    ("rdma.enable", Node, "RFC-0031", "\"auto\""),
    ("rdma.poller_thread", Node, "RFC-0031", "\"auto\""),
    ("rdma.gpudirect", Node, "RFC-0031", "\"auto\""),
    ("shm.enable", Node, "RFC-0031", "\"auto\""),
    // RFC-0032 §9.
    ("coordinator.lease_ms", Cluster, "RFC-0032", "\"4s\""),
    ("coordinator.wan_lease_ms", Cluster, "RFC-0032", "\"12s\""),
    (
        "coordinator.claim_stagger",
        Cluster,
        "RFC-0032",
        "\"300ms\"",
    ),
    ("coordinator.claim_window", Cluster, "RFC-0032", "\"500ms\""),
    (
        "coordinator.backoff",
        Cluster,
        "RFC-0032",
        "{ count = 3, window = \"60s\", initial = \"1s\", max = \"30s\" }",
    ),
    ("coordinator.prefer", Cluster, "RFC-0032", "[]"),
    (
        "log.snapshot_every",
        Cluster,
        "RFC-0032",
        "{ entries = 4096, interval = \"15m\" }",
    ),
    ("log.group_commit", Node, "RFC-0032", "\"100ms\""),
    ("log.lag_sync", Cluster, "RFC-0032", "256"),
    ("log.follower_queue", Node, "RFC-0032", "4096"),
    ("log.ack_timeout", Cluster, "RFC-0032", "\"500ms\""),
    ("log.command_retention", Node, "RFC-0032", "\"24h\""),
    ("orphan.adopt_deadline", Cluster, "RFC-0032", "\"10m\""),
    (
        "ring.max_inflight_steps",
        Instance,
        "RFC-0032",
        "computed: D, the stage count, in throughput mode; 1 per sequence in latency mode",
    ),
    ("ring.allow_token_relay", Cluster, "RFC-0032", "true"),
    ("tp.oneshot_max", Cluster, "RFC-0032", "\"1MiB\""),
    ("tp.bringup_timeout", Cluster, "RFC-0032", "\"30s\""),
    ("tp.local_router", Cluster, "RFC-0032", "\"verify\""),
    ("actions.tick", Node, "RFC-0032", "\"250ms\""),
    // RFC-0033 §12.
    ("recovery.history", Instance, "RFC-0033", "\"auto\""),
    (
        "recovery.history_ram",
        Node,
        "RFC-0033",
        "{ fraction = 0.05, max = \"2GiB\" }",
    ),
    ("recovery.history_disk", Node, "RFC-0033", "\"16GiB\""),
    (
        "recovery.state_checkpoint_tokens",
        Instance,
        "RFC-0033",
        "1024",
    ),
    (
        "recovery.confirm_after",
        Cluster,
        "RFC-0033",
        "computed: 2 x link.failover_timeout",
    ),
    (
        "recovery.drain_deadline",
        Instance,
        "RFC-0033",
        "computed: 60s in latency mode, 300s in throughput mode",
    ),
    ("recovery.p90_len_seed", Cluster, "RFC-0033", "512"),
    ("recovery.strategy_slack", Cluster, "RFC-0033", "0.2"),
    ("replan.degraded_min_gain", Cluster, "RFC-0033", "0.1"),
    ("straggler.factor", Cluster, "RFC-0033", "1.25"),
    ("straggler.window", Cluster, "RFC-0033", "\"30s\""),
    // RFC-0033 §8.2, the `runner.*` row of §12.
    ("runner.heartbeat", Node, "RFC-0033", "\"100ms\""),
    ("runner.liveness_timeout", Node, "RFC-0033", "\"3s\""),
    ("runner.stall_factor", Node, "RFC-0033", "20.0"),
    ("runner.stall_floor", Node, "RFC-0033", "\"10s\""),
    (
        "runner.load_stall",
        Node,
        "RFC-0033",
        "computed: 3 x (shard bytes / measured W_read) + 60s",
    ),
    (
        "runner.restart_backoff",
        Node,
        "RFC-0033",
        "{ initial = \"1s\", max = \"60s\" }",
    ),
    (
        "runner.restart_budget",
        Node,
        "RFC-0033",
        "{ count = 5, window = \"10m\" }",
    ),
    ("runner.daemon_silence", Node, "RFC-0033", "\"2s\""),
    ("runner.orphan_grace", Node, "RFC-0033", "\"30s\""),
    (
        "runner.device_reprobe",
        Node,
        "RFC-0033",
        "\"on_device_failure\"",
    ),
    ("standby", Instance, "RFC-0033", "\"none\""),
    // RFC-0034 §14.
    ("security.keystore", Node, "RFC-0034", "\"auto\""),
    ("security.cert_lifetime", Cluster, "RFC-0034", "\"180d\""),
    ("security.clock_skew", Cluster, "RFC-0034", "\"10m\""),
    ("security.rotation_window", Cluster, "RFC-0034", "\"7d\""),
    ("security.trusted_fabric", Node, "RFC-0034", "[]"),
    ("rdma.encrypt_untrusted", Node, "RFC-0034", "false"),
    ("invite.require_confirm", Cluster, "RFC-0034", "false"),
    ("integrity.verify_on_load", Node, "RFC-0034", "\"sample\""),
    (
        "models.signatures",
        Node,
        "RFC-0034",
        "\"verify_if_present\"",
    ),
    ("models.trusted_signers", Node, "RFC-0034", "[]"),
];

fn default_text(spec: DefaultSpec) -> String {
    match spec {
        DefaultSpec::Value(value) => value.to_string(),
        DefaultSpec::Unset => String::from("unset"),
        DefaultSpec::Computed(rule) => format!("computed: {rule}"),
    }
}

fn sections() -> Vec<&'static SectionSchema> {
    NET_SECTIONS
        .iter()
        .chain(CLUSTER_SECTIONS)
        .copied()
        .collect()
}

#[test]
fn the_registry_holds_every_section() {
    let registered: Vec<&str> = REGISTRY.sections().iter().map(|s| s.type_name).collect();
    let declared: Vec<&str> = sections().iter().map(|s| s.type_name).collect();
    assert_eq!(
        registered, declared,
        "the static registry lists every section once"
    );
}

#[test]
fn the_combined_registry_is_consistent() {
    assert_eq!(
        REGISTRY.check(),
        [],
        "no key declared twice, no value used as a table"
    );
    assert_eq!(
        REGISTRY.check_defaults(),
        [],
        "every default parses into its key's type"
    );
}

#[test]
fn every_rfc_key_is_registered_with_its_scope_owner_and_default() {
    for (path, scope, owner, default) in EXPECTED {
        let key = REGISTRY
            .key(path)
            .unwrap_or_else(|| panic!("`{path}` is not registered"));
        assert_eq!(key.scope, *scope, "{path}: scope");
        assert_eq!(key.owner, *owner, "{path}: owner");
        assert_eq!(default_text(key.default), *default, "{path}: default");
        if key.scope == Cluster {
            assert!(key.introduced_at.is_some(), "{path}: introduced_at");
        }
    }
    let mut registered: Vec<&str> = REGISTRY.keys().map(|k| k.path).collect();
    let mut expected: Vec<&str> = EXPECTED.iter().map(|(p, ..)| *p).collect();
    registered.sort_unstable();
    expected.sort_unstable();
    assert_eq!(registered, expected, "no key beyond the RFCs' tables");
    // RFC-0030 §9: 33; RFC-0031 §11: 16; RFC-0032 §9: 19; RFC-0033 §12 with §8.2: 22 (12, less
    // RFC-0011's three `replan.*` keys, plus 10 `runner.*`); RFC-0034 §14: 10.
    assert_eq!(REGISTRY.keys().count(), 100);
}

#[test]
fn keyholder_keys_are_keyholder() {
    use syntropy_config::Authority;
    for key in REGISTRY.keys().filter(|k| k.scope == Cluster) {
        let keyholder = ["security.", "invite.", "coordinator.", "swim."]
            .iter()
            .any(|p| key.path.starts_with(p))
            || key.path == "cluster.lan_admission";
        let want = if keyholder {
            Authority::Keyholder
        } else {
            Authority::Member
        };
        assert_eq!(key.authority, Some(want), "{}", key.path);
    }
}

#[test]
fn reload_classes_follow_the_stated_rule() {
    // The RFCs give no reload classes; these keys are the ones the sections class `Node`
    // (read once when a long-lived component is built) and `Instance` (they change an
    // instance's plan or runtime layout). Every other key is `Hot`.
    let node = [
        "cluster.namespace",
        "discovery.port",
        "discovery.mechanisms",
        "net.listen_port",
        "net.max_frame",
        "net.blob_frame",
        "quic.congestion",
        "quic.lan_initial_window",
        "quic.max_conn_window",
        "rdma.enable",
        "rdma.poller_thread",
        "rdma.gpudirect",
        "rdma.encrypt_untrusted",
        "shm.enable",
        "security.keystore",
        "security.trusted_fabric",
    ];
    let instance = [
        "ring.max_inflight_steps",
        "ring.allow_token_relay",
        "tp.oneshot_max",
        "tp.local_router",
        "recovery.history",
        "recovery.state_checkpoint_tokens",
        "recovery.drain_deadline",
        "standby",
    ];
    for key in REGISTRY.keys() {
        let want = if node.contains(&key.path) {
            Reload::Node
        } else if instance.contains(&key.path) {
            Reload::Instance
        } else {
            Reload::Hot
        };
        assert_eq!(key.reload, want, "{}", key.path);
    }
}

fn resolve(files: &[ConfigFile], env: &Env) -> Result<ResolvedConfig, syntropy_config::Rejected> {
    let flags = Flags::new();
    ResolvedConfig::resolve(
        &REGISTRY,
        Layers {
            os: Os::Linux,
            files,
            env,
            flags: &flags,
        },
    )
}

#[test]
fn show_resolved_lists_every_key_with_its_default() {
    let env = Env::from_pairs(Os::Linux, std::iter::empty::<(&str, &str)>());
    let config = resolve(&[], &env).expect("the defaults resolve");
    let text = command::show_resolved(&config);
    for (path, scope, _, default) in EXPECTED {
        if *scope == Instance {
            // Instance keys live in `[policy."<model>"]` tables and `InstancePolicy.domain`,
            // not in a node's resolved configuration (RFC-0054 §2.2 Rule 3).
            assert!(config.get(path).is_none(), "{path} is an instance key");
            continue;
        }
        let line = text
            .lines()
            .find(|l| l.split([' ', '=']).next() == Some(path))
            .unwrap_or_else(|| panic!("`{path}` is missing from:\n{text}"));
        match default.strip_prefix("computed: ") {
            Some(rule) => assert!(
                line.ends_with(&format!("# default, computed: {rule}")),
                "{path}: {line}"
            ),
            None => {
                let shown = config.get(path).and_then(|r| r.shown());
                assert!(shown.is_some(), "{path} has a value");
                assert!(line.starts_with(&format!("{path} = ")), "{path}: {line}");
                assert!(line.ends_with("# default"), "{path}: {line}");
            }
        }
    }
}

#[test]
fn defaults_print_as_the_rfcs_write_them() {
    let env = Env::from_pairs(Os::Linux, std::iter::empty::<(&str, &str)>());
    let config = resolve(&[], &env).expect("the defaults resolve");
    let shown = |path: &str| config.get(path).and_then(|r| r.shown()).unwrap();
    assert_eq!(shown("discovery.port"), "7434");
    assert_eq!(shown("net.max_frame"), "\"64MiB\"");
    assert_eq!(shown("swim.period"), "\"1s\"");
    assert_eq!(shown("security.cert_lifetime"), "\"180d\"");
    assert_eq!(shown("cluster.lan_admission"), "\"confirm\"");
}

/// RFC-0054 §8 error code 3, `unknown_key`.
const UNKNOWN_KEY: &str = "unknown key";
/// RFC-0054 §8 error code 9, `scope_mismatch`: an instance key outside a `policy.*` table.
const SCOPE_MISMATCH: &str = "is an instance key";

fn file_errors(text: &str) -> Vec<String> {
    let env = Env::from_pairs(Os::Linux, std::iter::empty::<(&str, &str)>());
    match resolve(&[ConfigFile::new("/etc/syntropy/config.toml", text)], &env) {
        Ok(_) => Vec::new(),
        Err(rejected) => rejected.errors.iter().map(|e| e.to_string()).collect(),
    }
}

#[test]
fn unknown_keys_fail_validation() {
    for (text, key) in [
        ("[swim]\nperiodd = \"1s\"\n", "swim.periodd"),
        ("[net]\nrelay_modes = \"auto\"\n", "net.relay_modes"),
        ("[rdma]\nencrypt = true\n", "rdma.encrypt"),
        ("[runner]\nheartbeats = \"1s\"\n", "runner.heartbeats"),
        (
            "[recovery]\nhistory_ramm = \"1GiB\"\n",
            "recovery.history_ramm",
        ),
        ("[nosuch]\nkey = 1\n", "nosuch"),
    ] {
        let errors = file_errors(text);
        assert_eq!(errors.len(), 1, "{text}: {errors:?}");
        assert!(errors[0].contains(UNKNOWN_KEY), "{text}: {errors:?}");
        assert!(
            errors[0].contains(&format!("`{key}`")),
            "{text}: {errors:?}"
        );
    }
}

#[test]
fn instance_keys_in_a_node_file_fail_validation() {
    for (text, key) in [
        (
            "[ring]\nmax_inflight_steps = 2\n",
            "ring.max_inflight_steps",
        ),
        ("[recovery]\nhistory = \"on\"\n", "recovery.history"),
        (
            "[recovery]\nstate_checkpoint_tokens = 512\n",
            "recovery.state_checkpoint_tokens",
        ),
        (
            "[recovery]\ndrain_deadline = \"30s\"\n",
            "recovery.drain_deadline",
        ),
        ("standby = \"replica\"\n", "standby"),
    ] {
        let errors = file_errors(text);
        assert_eq!(errors.len(), 1, "{text}: {errors:?}");
        assert!(errors[0].contains(SCOPE_MISMATCH), "{text}: {errors:?}");
        assert!(
            errors[0].contains(&format!("`{key}`")),
            "{text}: {errors:?}"
        );
    }
}

#[test]
fn node_and_cluster_keys_in_a_node_file_are_accepted() {
    // A node key sets the node's value; a cluster key becomes a proposal (RFC-0054 §2.3).
    let env = Env::from_pairs(Os::Linux, std::iter::empty::<(&str, &str)>());
    let file = ConfigFile::new(
        "/etc/syntropy/config.toml",
        "[discovery]\nport = 7500\n[swim]\nperiod = \"2s\"\n[net]\nrelay_mode = \"disabled\"\n\
         max_frame = \"32MiB\"\n",
    );
    let config = resolve(&[file], &env).expect("valid");
    assert_eq!(
        config.get("discovery.port").unwrap().shown().unwrap(),
        "7500"
    );
    assert_eq!(
        config.get("net.relay_mode").unwrap().shown().unwrap(),
        "\"disabled\""
    );
    let proposed: Vec<&str> = config.proposals().iter().map(|p| p.path.as_str()).collect();
    assert_eq!(proposed, ["swim.period"]);
}

#[test]
fn confirm_after_is_twice_the_failover_timeout() {
    use syntropy_cluster::config::recovery::confirm_after;
    use syntropy_config::Duration;
    let key = REGISTRY.key("recovery.confirm_after").unwrap();
    assert_eq!(
        key.default,
        DefaultSpec::Computed("2 x link.failover_timeout")
    );
    // With link.failover_timeout's defaults: 2s on LAN links, 10s on WAN links.
    assert_eq!(
        confirm_after(None, Duration::from_secs(2)),
        Duration::from_secs(4)
    );
    assert_eq!(
        confirm_after(None, Duration::from_secs(10)),
        Duration::from_secs(20)
    );
    // An operator who raises the failover timeout raises the window with it.
    assert_eq!(
        confirm_after(None, Duration::from_secs(5)),
        Duration::from_secs(10)
    );
    // A value the operator sets wins.
    assert_eq!(
        confirm_after(Some(Duration::from_secs(3)), Duration::from_secs(10)),
        Duration::from_secs(3)
    );
}
