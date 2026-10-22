// SPDX-License-Identifier: MIT OR Apache-2.0

use serde::Deserialize;

use super::*;
use crate::{
    ConfigFile, ConfigSection, Duration, Env, Flags, Layers, Os, Registry, ResolvedConfig, Secret,
};

/// The `sched` section as a binary registers it, for the load-time comparison.
static REGISTRY: Registry = Registry::new(&[&Sched::SCHEMA]);

/// The scheduler's section as PLC-E8-T13 declares it (`crates/sched/src/config.rs` in #1656),
/// field for field, so the builder is shown against its first consumer.
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "sched", scope = Instance, owner = "RFC-0012")]
struct Sched {
    /// How long a dispatcher spins before it parks.
    #[config(reload = Hot, default = computed("50us on mains power, 10us on battery"))]
    spin_us: Option<Duration>,
    /// How far ahead of a predicted frame a parked stage is woken.
    #[config(reload = Hot, default = "300us")]
    prewake_us: Duration,
    /// The cap on a prefill chunk's time while an interactive sequence decodes.
    #[config(
        reload = Hot,
        default = computed("0.5 x the active sequence's predicted t_tok, at least 64 tokens of chunk")
    )]
    decode_stall_budget: Option<Duration>,
    /// Prefill-lane chunks in flight per instance.
    #[config(reload = Instance, default = computed("D, the instance's stage count"), min = 1)]
    prefill_inflight: Option<u32>,
    /// Throughput mode: decode steps per prefill step.
    #[config(reload = Hot, default = computed("G, the instance's group count"), min = 1)]
    prefill_share: Option<u32>,
    /// The head's admission queue length.
    #[config(reload = Hot, default = 64)]
    max_queued: u32,
    /// Latency mode: the most sequences decoding in one step.
    #[config(reload = Instance, default = 4, min = 1, max = 4)]
    latency_decode_rows: u32,
    /// The longest scheduler-initiated pause.
    #[config(reload = Hot, default = "30s")]
    max_pause: Duration,
}

/// A table from `key = value` lines.
fn table(text: &str) -> Value {
    Value::Table(toml::from_str(text).expect("test TOML parses"))
}

/// The `sched` table of a policy, from `key = value` lines.
fn sched(text: &str) -> Value {
    table(text)
}

fn build(text: &str) -> Result<InstanceSection<Sched>, InstanceSectionError> {
    InstanceSection::<Sched>::from_policy(&sched(text), |_| None)
}

/// Each problem's key and code number, sorted.
fn rejections(text: &str) -> Vec<(String, u16)> {
    match build(text) {
        Err(InstanceSectionError::Rejected(problems)) => {
            let mut got: Vec<(String, u16)> = problems
                .iter()
                .map(|p| {
                    (
                        p.key.as_deref().unwrap_or("").to_owned(),
                        p.code.map_or(0, Code::number),
                    )
                })
                .collect();
            got.sort();
            got
        }
        other => panic!("{text}: {other:?}"),
    }
}

#[test]
fn an_empty_policy_gives_the_defaults_and_leaves_computed_ones_unset() {
    let s = build("").unwrap();
    assert_eq!(
        *s.section(),
        Sched {
            spin_us: None,
            prewake_us: Duration::from_nanos(300_000),
            decode_stall_budget: None,
            prefill_inflight: None,
            prefill_share: None,
            max_queued: 64,
            latency_decode_rows: 4,
            max_pause: Duration::from_secs(30),
        }
    );
    assert_eq!(s.value("sched.max_queued"), Some(&Value::Integer(64)));
    assert_eq!(s.value("sched.spin_us"), None);
    assert_eq!(s.value("sched.nothing"), None);
}

#[test]
fn a_policy_sets_every_key_as_the_hand_written_parser_does() {
    let s = build(
        "spin_us = \"20us\"\nprewake_us = \"1ms\"\ndecode_stall_budget = \"25ms\"\n\
         prefill_inflight = 2\nprefill_share = 3\nmax_queued = 0\nlatency_decode_rows = 1\n\
         max_pause = \"2m\"\n",
    )
    .unwrap()
    .into_section();
    assert_eq!(
        s,
        Sched {
            spin_us: Some(Duration::from_nanos(20_000)),
            prewake_us: Duration::from_millis(1),
            decode_stall_budget: Some(Duration::from_millis(25)),
            prefill_inflight: Some(2),
            prefill_share: Some(3),
            max_queued: 0,
            latency_decode_rows: 1,
            max_pause: Duration::from_secs(120),
        }
    );
    // The bounds themselves are accepted.
    let edges = build(
        "latency_decode_rows = 4\nprefill_inflight = 1\nprefill_share = 1\n\
         max_queued = 4294967295\nspin_us = \"0s\"\n",
    )
    .unwrap();
    assert_eq!(edges.latency_decode_rows, 4);
    assert_eq!(edges.max_queued, u32::MAX);
}

/// The cases of the scheduler's `invalid_values_are_rejected_with_their_code` (#1656), with
/// its codes: 2 `wrong_type`, 3 `unknown_key`, 4 `out_of_range`, 5 `bad_unit`. The builder
/// uses each key's own check, as a node file's `[policy."…"]` table is checked at load
/// ([`the_builder_gives_the_codes_a_policy_table_gets_at_load`]), so all three agree.
const CASES: &[(&str, &str, u16)] = &[
    ("latency_decode_rows = 0", "sched.latency_decode_rows", 4),
    ("latency_decode_rows = 5", "sched.latency_decode_rows", 4),
    ("latency_decode_rows = -1", "sched.latency_decode_rows", 4),
    (
        "latency_decode_rows = \"4\"",
        "sched.latency_decode_rows",
        2,
    ),
    ("latency_decode_rows = 4.0", "sched.latency_decode_rows", 2),
    ("prefill_inflight = 0", "sched.prefill_inflight", 4),
    ("prefill_share = 0", "sched.prefill_share", 4),
    ("max_queued = -1", "sched.max_queued", 4),
    ("max_queued = 4294967296", "sched.max_queued", 4),
    ("max_queued = true", "sched.max_queued", 2),
    ("spin_us = 50", "sched.spin_us", 2),
    ("spin_us = \"50\"", "sched.spin_us", 5),
    ("spin_us = \"50 parsecs\"", "sched.spin_us", 5),
    ("prewake_us = \"\"", "sched.prewake_us", 5),
    (
        "decode_stall_budget = [\"25ms\"]",
        "sched.decode_stall_budget",
        2,
    ),
    ("max_pause = \"-1s\"", "sched.max_pause", 5),
    ("max_pause = \"30\"", "sched.max_pause", 5),
    ("spin = \"50us\"", "sched.spin", 3),
];

#[test]
fn invalid_values_are_rejected_with_their_codes() {
    let mut failures = Vec::new();
    for (text, key, code) in CASES {
        let got = rejections(text);
        if got != [((*key).to_owned(), *code)] {
            failures.push(format!("{text}: {got:?}, expected [({key}, {code})]"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The builder and the load-time check of a node file's `[policy."<selector>"]` table
/// (PROD-E1-T03) give one key and one code for every case: one check, two callers.
#[test]
fn the_builder_gives_the_codes_a_policy_table_gets_at_load() {
    let mut failures = Vec::new();
    for (text, _, _) in CASES {
        let files = [ConfigFile::new(
            "/c.toml",
            format!("[policy.\"m\".sched]\n{text}\n"),
        )];
        let rejected = ResolvedConfig::resolve(
            &REGISTRY,
            Layers {
                os: Os::Linux,
                files: &files,
                env: &Env::default(),
                flags: &Flags::new(),
            },
        )
        .expect_err(text);
        let mut at_load: Vec<(String, u16)> = rejected
            .errors
            .iter()
            .map(|p| {
                let key = p.key.as_deref().unwrap_or("");
                (
                    key.strip_prefix("policy.m.").unwrap_or(key).to_owned(),
                    p.code.map_or(0, Code::number),
                )
            })
            .collect();
        at_load.sort();
        let built = rejections(text);
        if at_load != built {
            failures.push(format!("{text}: at load {at_load:?}, built {built:?}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn every_problem_in_a_policy_is_reported() {
    assert_eq!(
        rejections("spin_us = 50\nmax_queued = -1\ntypo = 1\nmax_pause = \"30\"\n"),
        [
            ("sched.max_pause".to_owned(), 5),
            ("sched.max_queued".to_owned(), 4),
            ("sched.spin_us".to_owned(), 2),
            ("sched.typo".to_owned(), 3),
        ]
    );
    // An unknown key close to a real one gets the suggestion.
    match build("spin_uss = \"5us\"") {
        Err(InstanceSectionError::Rejected(problems)) => {
            assert_eq!(
                problems[0].help.as_deref(),
                Some("did you mean `sched.spin_us`?")
            );
        }
        other => panic!("{other:?}"),
    }
    // A policy whose `sched` is not a table.
    match InstanceSection::<Sched>::from_policy(&Value::Integer(1), |_| None) {
        Err(InstanceSectionError::Rejected(problems)) => {
            assert_eq!(problems.len(), 1);
            assert_eq!(problems[0].key.as_deref(), Some("sched"));
            assert_eq!(problems[0].code, Some(Code::WrongType));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn the_caller_supplies_the_computed_defaults_it_owns() {
    // The scheduler knows the power source, the stage count and the group count.
    let computed = |key: &'static KeySchema| match key.name {
        "spin_us" => Some(Value::String("10us".into())),
        "prefill_inflight" => Some(Value::Integer(3)),
        _ => None,
    };
    let s =
        InstanceSection::<Sched>::from_policy(&sched("prefill_inflight = 2"), computed).unwrap();
    assert_eq!(s.spin_us, Some(Duration::from_nanos(10_000)));
    // The policy's value wins over the computed one.
    assert_eq!(s.prefill_inflight, Some(2));
    // A computed default not supplied stays unset on an `Option` key.
    assert_eq!(s.prefill_share, None);
    // A computed value that does not fit its key is the caller's error.
    match InstanceSection::<Sched>::from_policy(&sched(""), |key| {
        (key.name == "prefill_share").then_some(Value::Integer(0))
    }) {
        Err(InstanceSectionError::Section(error)) => {
            assert!(error.message.contains("`sched.prefill_share`"), "{error}");
        }
        other => panic!("{other:?}"),
    }
}

/// A section with a computed default on a key that is not an `Option`, one key of another
/// scope, and a secret-typed key.
#[derive(Debug, Deserialize, ConfigSection)]
#[config(section = "probe", scope = Instance, owner = "RFC-0054")]
#[expect(dead_code, reason = "only the problems these keys raise are read")]
struct Probe {
    /// The instance's stage count, which the caller computes.
    #[config(reload = Instance, default = computed("the stage count"))]
    stages: u32,
    /// A node key that strayed into an instance section.
    #[config(reload = Node, default = false, scope = Node)]
    node_only: bool,
    /// A token.
    #[config(reload = Hot, default = unset)]
    token: Option<Secret<String>>,
}

#[test]
fn a_computed_default_not_supplied_for_a_required_key_is_the_callers_error() {
    match InstanceSection::<Probe>::from_policy(&Value::Table(toml::Table::new()), |_| None) {
        Err(InstanceSectionError::Section(error)) => {
            assert_eq!(error.section, "Probe");
            assert!(
                error
                    .message
                    .ends_with("computed defaults not supplied: `probe.stages`"),
                "{error}"
            );
        }
        other => panic!("{other:?}"),
    }
    let stages = |key: &'static KeySchema| (key.name == "stages").then_some(Value::Integer(4));
    let p =
        InstanceSection::<Probe>::from_policy(&Value::Table(toml::Table::new()), stages).unwrap();
    assert_eq!(p.stages, 4);
}

#[test]
fn a_key_of_another_scope_in_the_policy_is_scope_mismatch() {
    let stages = |key: &'static KeySchema| (key.name == "stages").then_some(Value::Integer(4));
    match InstanceSection::<Probe>::from_policy(&table("node_only = true"), stages) {
        Err(InstanceSectionError::Rejected(problems)) => {
            assert_eq!(problems.len(), 1);
            assert_eq!(problems[0].code, Some(Code::ScopeMismatch));
            assert_eq!(problems[0].key.as_deref(), Some("probe.node_only"));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_secret_typed_value_appears_in_no_problem() {
    const MARKER: &str = "MARKER-7f3a9c";
    let stages = |key: &'static KeySchema| (key.name == "stages").then_some(Value::Integer(4));
    for text in [
        format!("token = [\"{MARKER}\"]"),
        format!("token = {{ file = 7, env = \"{MARKER}\" }}"),
        format!("token = {{ {MARKER} = 1 }}"),
    ] {
        match InstanceSection::<Probe>::from_policy(&table(&text), stages) {
            Err(error @ InstanceSectionError::Rejected(_)) => {
                let InstanceSectionError::Rejected(problems) = &error else {
                    unreachable!()
                };
                for problem in problems {
                    assert!(
                        !format!("{problem:?}").contains(MARKER),
                        "{text}: {problem:?}"
                    );
                }
                assert!(!error.to_string().contains(MARKER), "{text}: {error}");
            }
            other => panic!("{text}: {other:?}"),
        }
    }
}

#[test]
fn changes_name_each_differing_key_with_its_reload_class() {
    let before = build("max_queued = 64\nlatency_decode_rows = 4").unwrap();
    let after = build("max_queued = 32\nlatency_decode_rows = 2").unwrap();
    assert_eq!(
        before.changes(&after),
        [
            Change {
                key: "sched.latency_decode_rows",
                reload: Reload::Instance
            },
            Change {
                key: "sched.max_queued",
                reload: Reload::Hot
            },
        ]
    );
    assert_eq!(before.reload_needed(&after), Some(Reload::Instance));
    // Setting a key to its default is no change: the effective value is the same.
    let same = build("").unwrap();
    assert!(before.changes(&same).is_empty());
    assert_eq!(before.reload_needed(&same), None);
    // Only a hot key: the reload applies without a new plan.
    let hot = build("max_pause = \"1m\"").unwrap();
    assert_eq!(same.reload_needed(&hot), Some(Reload::Hot));
}

#[test]
fn the_domain_entries_build_the_same_section_as_the_table() {
    let text = "spin_us = \"20us\"\nmax_queued = 8\nlatency_decode_rows = 2";
    let from_table = build(text).unwrap();
    let domain: BTreeMap<KeyPath, ConfigValue> = [
        ("sched.spin_us", ConfigValue::String("20us".into())),
        ("sched.max_queued", ConfigValue::Integer(8)),
        ("sched.latency_decode_rows", ConfigValue::Integer(2)),
        // Another section's key is not this section's concern.
        ("solver.context_target", ConfigValue::Integer(4096)),
    ]
    .into_iter()
    .map(|(path, value)| (KeyPath::from(path), value))
    .collect();
    let from_domain = InstanceSection::<Sched>::from_domain(&domain, |_| None).unwrap();
    assert_eq!(from_domain, from_table);
    assert!(from_table.changes(&from_domain).is_empty());
    // The same keys as one table at the prefix.
    let whole: BTreeMap<KeyPath, ConfigValue> = [(
        KeyPath::from("sched"),
        ConfigValue::Table(
            [
                ("spin_us".to_owned(), ConfigValue::String("20us".into())),
                ("max_queued".to_owned(), ConfigValue::Integer(8)),
                ("latency_decode_rows".to_owned(), ConfigValue::Integer(2)),
            ]
            .into_iter()
            .collect(),
        ),
    )]
    .into_iter()
    .collect();
    assert_eq!(
        InstanceSection::<Sched>::from_domain(&whole, |_| None).unwrap(),
        from_table
    );
    // A path below one of the section's keys is unknown; a non-table at the prefix is the wrong
    // type.
    let deep: BTreeMap<KeyPath, ConfigValue> = [(
        KeyPath::from("sched.max_queued.limit"),
        ConfigValue::Integer(1),
    )]
    .into_iter()
    .collect();
    match InstanceSection::<Sched>::from_domain(&deep, |_| None) {
        Err(InstanceSectionError::Rejected(problems)) => {
            assert_eq!(problems[0].code, Some(Code::UnknownKey));
            assert_eq!(problems[0].key.as_deref(), Some("sched.max_queued.limit"));
        }
        other => panic!("{other:?}"),
    }
    let scalar: BTreeMap<KeyPath, ConfigValue> =
        [(KeyPath::from("sched"), ConfigValue::Integer(1))]
            .into_iter()
            .collect();
    assert!(matches!(
        InstanceSection::<Sched>::from_domain(&scalar, |_| None),
        Err(InstanceSectionError::Rejected(_))
    ));
}

/// A section that owns only its own keys under a prefix another section owns.
#[derive(Debug, Deserialize, ConfigSection)]
#[config(section = "net", shared, scope = Instance, owner = "RFC-0030")]
struct SharedNet {
    /// A relay setting.
    #[config(reload = Hot, default = "auto")]
    relay_mode: String,
}

#[test]
fn a_shared_section_reads_only_its_own_keys() {
    let s = InstanceSection::<SharedNet>::from_policy(
        &table("relay_mode = \"off\"\nbulk_on_metered = true"),
        |_| None,
    )
    .unwrap();
    assert_eq!(s.relay_mode, "off");
}

/// A top-level instance key, as RFC-0033's `standby` travels in `InstancePolicy::domain` under
/// its bare path (RFC-0033 note of 2026-10-22).
#[derive(Debug, Deserialize, ConfigSection)]
#[config(section = "", scope = Instance, owner = "RFC-0033")]
struct TopLevel {
    /// Standby policy.
    #[config(reload = Instance, default = "none")]
    standby: String,
}

#[test]
fn a_top_level_section_reads_its_bare_keys_from_the_domain_map() {
    const {
        assert!(
            TopLevel::SCHEMA.shared,
            "a top-level section is always shared"
        )
    };
    let domain: BTreeMap<KeyPath, ConfigValue> = [
        ("standby", ConfigValue::String("replica".into())),
        // Other sections' keys sit beside it in the same map.
        ("recovery.history", ConfigValue::Integer(5)),
        ("sched.max_queued", ConfigValue::Integer(8)),
    ]
    .into_iter()
    .map(|(path, value)| (KeyPath::from(path), value))
    .collect();
    let s = InstanceSection::<TopLevel>::from_domain(&domain, |_| None).unwrap();
    assert_eq!(s.standby, "replica");
    assert_eq!(s.value("standby"), Some(&Value::String("replica".into())));
}

#[test]
fn a_secret_typed_value_is_not_kept_but_its_change_is_seen() {
    const MARKER: &str = "MARKER-51c0de";
    let stages = |key: &'static KeySchema| (key.name == "stages").then_some(Value::Integer(4));
    let with = |token: &str| {
        InstanceSection::<Probe>::from_policy(&table(&format!("token = \"{token}\"")), stages)
            .unwrap()
    };
    let one = with(MARKER);
    assert!(!format!("{one:?}").contains(MARKER), "{one:?}");
    assert!(format!("{one:?}").contains("[redacted]"), "{one:?}");
    assert_eq!(one.value("probe.token"), None);
    assert_eq!(one.value("probe.stages"), Some(&Value::Integer(4)));
    // The same secret again is no change; another one is.
    assert!(one.changes(&with(MARKER)).is_empty());
    assert_eq!(
        one.changes(&with("other")),
        [Change {
            key: "probe.token",
            reload: Reload::Hot
        }]
    );
}
