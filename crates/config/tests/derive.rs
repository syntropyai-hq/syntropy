// SPDX-License-Identifier: MIT OR Apache-2.0

//! What `#[derive(ConfigSection)]` records, and the section registry over derived sections.
//! The compile-time failures are in `compile_fail.rs`.

// The sections exist for their schemas; most fields are never read.
#![allow(dead_code)]

use std::collections::BTreeMap;

use serde::Deserialize;
use syntropy_config::{
    Authority, ByteSize, ConfigSection, DefaultSpec, DefaultValue, Duration, Fraction, Rate,
    Registry, Reload, Reserve, Scope, ValueErrorKind, never,
};

/// Example node settings.
///
/// A second paragraph.
#[derive(Debug, Deserialize, ConfigSection)]
#[config(section = "example", scope = Node, owner = "RFC-0054")]
struct ExampleSection {
    /// A size.
    #[config(reload = Hot, default = "16GiB", min = "1GiB", max = "1TiB")]
    size: ByteSize,
    /// A span that may be `"never"`.
    #[config(reload = Hot, default = "2m")]
    #[serde(with = "never", default)]
    keep: Option<Duration>,
    /// A count.
    ///
    /// Owned by another RFC.
    #[config(reload = Node, default = 64, min = 1, owner = "RFC-0001")]
    threads: u32,
    /// A share.
    #[config(reload = Hot, default = 0.015625)]
    sample: Fraction,
    /// A switch.
    #[config(reload = Hot, default = false)]
    enabled: bool,
    /// Listeners.
    #[config(reload = Hot, default = ["127.0.0.1:7433", "[::1]:7433"])]
    bind: Vec<String>,
    /// Rates by use.
    #[config(reload = Hot, default = { interactive = "6tok/s", background = "1tok/s" })]
    min_tok_s: BTreeMap<String, Rate>,
    /// An optional limit.
    #[config(reload = Hot, default = unset)]
    limit: Option<u32>,
    /// Memory kept back.
    #[config(reload = Hot, default = computed("max(6GiB, 10% of RAM)"), min = "1GiB")]
    headroom: Reserve,
    /// A negative offset.
    #[config(reload = Hot, default = -3)]
    offset: i32,
}

/// Cluster settings, with keys the keyholder rule covers.
#[derive(Debug, Deserialize, ConfigSection)]
#[config(section = "", scope = Cluster, owner = "RFC-0054")]
struct ClusterRoot {
    /// Whether approximate mode is allowed.
    #[config(reload = Hot, default = true, introduced_at = 1)]
    allow_approximate: bool,
    /// A routine key.
    #[config(reload = Hot, default = 3, introduced_at = FEATURE_LEVEL_TEST)]
    routine: u32,
    /// A routine key declared keyholder anyway.
    #[config(reload = Hot, default = 3, introduced_at = 2, authority = Keyholder)]
    careful: u32,
}

const FEATURE_LEVEL_TEST: u32 = 7;

/// Settings under `security.*`.
#[derive(Debug, Deserialize, ConfigSection)]
#[config(section = "security", scope = Cluster, owner = "RFC-0034")]
struct SecuritySection {
    /// A cluster key under `security.*`.
    #[config(reload = Hot, default = "x", introduced_at = 1)]
    policy: String,
    /// A node key under `security.*`: no authority.
    #[config(reload = Hot, default = [], scope = Node)]
    trusted_fabric: Vec<String>,
}

/// Update settings: node keys, plus cluster keys forced to keyholder.
#[derive(Debug, Deserialize, ConfigSection)]
#[config(section = "update", scope = Node, owner = "RFC-0052")]
struct UpdateSection {
    /// The channel.
    #[config(reload = Hot, default = "stable")]
    channel: String,
    /// Whether updates are coordinated.
    #[config(reload = Hot, default = true, scope = Cluster, introduced_at = 1)]
    cluster_coordinated: bool,
}

/// Instance keys at the top of a `policy.*` table.
#[derive(Debug, Deserialize, ConfigSection)]
#[config(section = "", scope = Instance, owner = "RFC-0054")]
struct PolicyRoot {
    /// The objective.
    #[config(reload = Instance, default = "latency", owner = "RFC-0011")]
    objective: String,
    /// The fidelity.
    #[config(reload = Instance, default = "exact", owner = "RFC-0042", introduced_at = 1)]
    fidelity: String,
}

/// Approximate knobs.
#[derive(Debug, Deserialize, ConfigSection)]
#[config(section = "approximate", scope = Instance, owner = "RFC-0042")]
struct ApproximateSection {
    /// A knob.
    #[config(reload = Instance, default = false)]
    expert_skip: bool,
}

/// Keys inside RFC-0031's `net` table, owned by RFC-0030.
#[derive(Debug, Deserialize, ConfigSection)]
#[config(section = "net", shared, scope = Node, owner = "RFC-0030")]
struct NetMembership {
    /// The relay mode.
    #[config(reload = Hot, default = "auto")]
    relay_mode: String,
}

/// RFC-0031's `net` table.
#[derive(Debug, Deserialize, ConfigSection)]
#[config(section = "net", scope = Node, owner = "RFC-0031")]
struct NetSection {
    /// The MTU.
    #[config(reload = Node, default = 1500)]
    mtu: u32,
}

/// Defaults that do not fit their types.
#[derive(Debug, Deserialize, ConfigSection)]
#[config(section = "broken", scope = Node, owner = "RFC-0054")]
struct BrokenDefaults {
    /// Not a size.
    #[config(reload = Hot, default = "16XB")]
    size: ByteSize,
    /// Unset, but not optional.
    #[config(reload = Hot, default = unset)]
    count: u32,
    /// Below its own minimum.
    #[config(reload = Hot, default = 0, min = 1)]
    threads: u32,
    /// A bound that does not parse.
    #[config(reload = Hot, default = "1GiB", min = "lots")]
    floor: ByteSize,
}

static REGISTRY: Registry = Registry::new(&[
    &ExampleSection::SCHEMA,
    &ClusterRoot::SCHEMA,
    &SecuritySection::SCHEMA,
    &UpdateSection::SCHEMA,
    &PolicyRoot::SCHEMA,
    &ApproximateSection::SCHEMA,
    &NetMembership::SCHEMA,
    &NetSection::SCHEMA,
]);

fn key(path: &str) -> &'static syntropy_config::KeySchema {
    REGISTRY
        .key(path)
        .unwrap_or_else(|| panic!("no key {path}"))
}

#[test]
fn records_the_section() {
    let schema = ExampleSection::SCHEMA;
    assert_eq!(schema.type_name, "ExampleSection");
    assert_eq!(schema.prefix, "example");
    assert!(!schema.shared);
    assert_eq!(schema.owner, "RFC-0054");
    assert_eq!(schema.scope, Scope::Node);
    assert_eq!(schema.doc, "Example node settings.\n\nA second paragraph.");
    assert_eq!(schema.keys.len(), 10);
    const { assert!(NetMembership::SCHEMA.shared) };
    // A top-level section is shared: it claims its keys, not a subtree.
    const { assert!(ClusterRoot::SCHEMA.shared) };
}

#[test]
fn records_each_key() {
    let size = key("example.size");
    assert_eq!(size.name, "size");
    assert_eq!(size.doc, "A size.");
    assert_eq!(size.ty, "ByteSize");
    assert_eq!(size.scope, Scope::Node);
    assert_eq!(size.reload, Reload::Hot);
    assert_eq!(size.owner, "RFC-0054");
    assert_eq!(size.default, DefaultSpec::Value(DefaultValue::Str("16GiB")));
    assert_eq!(size.min, Some(DefaultValue::Str("1GiB")));
    assert_eq!(size.max, Some(DefaultValue::Str("1TiB")));
    assert_eq!(size.introduced_at, None);
    assert_eq!(size.authority, None);
    assert_eq!(size.env, Some("SYNTROPY_EXAMPLE__SIZE"));

    let threads = key("example.threads");
    assert_eq!(threads.doc, "A count.\n\nOwned by another RFC.");
    assert_eq!(threads.owner, "RFC-0001");
    assert_eq!(threads.reload, Reload::Node);
    assert_eq!(key("example.keep").ty, "Option<Duration>");
    assert_eq!(key("example.min_tok_s").ty, "BTreeMap<String, Rate>");
    assert_eq!(
        key("example.min_tok_s").env,
        Some("SYNTROPY_EXAMPLE__MIN_TOK_S")
    );
    assert_eq!(key("example.limit").default, DefaultSpec::Unset);
    assert_eq!(
        key("example.headroom").default,
        DefaultSpec::Computed("max(6GiB, 10% of RAM)")
    );
}

#[test]
fn defaults_print_as_toml() {
    let printed = |path| match key(path).default {
        DefaultSpec::Value(value) => value.to_string(),
        other => panic!("{other:?}"),
    };
    assert_eq!(printed("example.size"), "\"16GiB\"");
    assert_eq!(printed("example.threads"), "64");
    assert_eq!(printed("example.sample"), "0.015625");
    assert_eq!(printed("example.enabled"), "false");
    assert_eq!(
        printed("example.bind"),
        "[\"127.0.0.1:7433\", \"[::1]:7433\"]"
    );
    assert_eq!(
        printed("example.min_tok_s"),
        "{ interactive = \"6tok/s\", background = \"1tok/s\" }"
    );
    assert_eq!(printed("example.offset"), "-3");
}

#[test]
fn every_default_fits_its_type() {
    assert_eq!(REGISTRY.check_defaults(), vec![]);
    assert_eq!(REGISTRY.check(), vec![]);
}

#[test]
fn bad_defaults_are_reported() {
    let errors = BrokenDefaults::check_defaults();
    let found: Vec<(&str, ValueErrorKind)> =
        errors.iter().map(|e| (e.key, e.error.kind())).collect();
    assert_eq!(
        found,
        vec![
            ("broken.size", ValueErrorKind::WrongType),
            ("broken.count", ValueErrorKind::WrongType),
            ("broken.threads", ValueErrorKind::OutOfRange),
            ("broken.floor", ValueErrorKind::WrongType),
        ],
        "{errors:#?}"
    );
    assert!(errors[0].error.message().contains("16XB"), "{}", errors[0]);
    assert!(errors[1].error.message().contains("unset"), "{}", errors[1]);
}

#[test]
fn ranges_are_checked_on_values() {
    let parse = |json: &str| serde_json::from_str::<ExampleSection>(json).unwrap();
    let base = r#""keep": "never", "sample": 0.5, "enabled": true, "bind": [],
        "min_tok_s": {}, "limit": null, "headroom": "8GiB", "offset": 0"#;
    let ok = parse(&format!(r#"{{ "size": "2GiB", "threads": 4, {base} }}"#));
    assert_eq!(ok.check_ranges(), vec![]);
    assert_eq!(ok.keep, None);

    let bad = parse(&format!(r#"{{ "size": "512MiB", "threads": 0, {base} }}"#));
    let keys: Vec<&str> = bad.check_ranges().iter().map(|e| e.key).collect();
    assert_eq!(keys, vec!["example.size", "example.threads"]);

    let share_only = parse(
        r#"{ "size": "2GiB", "threads": 4, "keep": "1m", "sample": 0.5, "enabled": true,
        "bind": [], "min_tok_s": {}, "limit": 3, "headroom": "10%", "offset": 0 }"#,
    );
    let keys: Vec<&str> = share_only.check_ranges().iter().map(|e| e.key).collect();
    assert_eq!(keys, vec!["example.headroom"]);
}

#[test]
fn cluster_keys_carry_feature_level_and_authority() {
    let allow = key("allow_approximate");
    assert_eq!(allow.scope, Scope::Cluster);
    assert_eq!(allow.introduced_at, Some(1));
    assert_eq!(
        allow.authority,
        Some(Authority::Keyholder),
        "forced by name"
    );
    assert_eq!(allow.env, None);

    let routine = key("routine");
    assert_eq!(routine.introduced_at, Some(FEATURE_LEVEL_TEST));
    assert_eq!(routine.authority, Some(Authority::Member), "the default");
    assert_eq!(key("careful").authority, Some(Authority::Keyholder));

    assert_eq!(
        key("security.policy").authority,
        Some(Authority::Keyholder),
        "forced by prefix"
    );
    let fabric = key("security.trusted_fabric");
    assert_eq!(fabric.scope, Scope::Node);
    assert_eq!(fabric.authority, None);
    assert_eq!(fabric.env, Some("SYNTROPY_SECURITY__TRUSTED_FABRIC"));

    let coordinated = key("update.cluster_coordinated");
    assert_eq!(coordinated.scope, Scope::Cluster);
    assert_eq!(coordinated.authority, Some(Authority::Keyholder));
    assert_eq!(key("update.channel").authority, None);
}

#[test]
fn instance_keys_carry_authority() {
    assert_eq!(key("objective").authority, Some(Authority::Member));
    assert_eq!(key("objective").owner, "RFC-0011");
    assert_eq!(key("objective").introduced_at, None);
    assert_eq!(key("fidelity").authority, Some(Authority::Keyholder));
    assert_eq!(key("fidelity").introduced_at, Some(1));
    assert_eq!(
        key("approximate.expert_skip").authority,
        Some(Authority::Keyholder)
    );
    assert_eq!(key("approximate.expert_skip").env, None);
}

#[test]
fn registry_lookups() {
    assert_eq!(REGISTRY.sections().len(), 8);
    assert_eq!(
        REGISTRY.section_of("net.relay_mode").map(|s| s.owner),
        Some("RFC-0030")
    );
    assert_eq!(
        REGISTRY.section_of("net.mtu").map(|s| s.owner),
        Some("RFC-0031")
    );
    assert!(REGISTRY.key("net.unknown").is_none());
    assert_eq!(REGISTRY.keys().count(), 22);
}
