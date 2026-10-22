// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `sched.*` keys against RFC-0012 §12, `config show --resolved`, and rejection at load.

use syntropy_config::{
    Checked, Code, ConfigFile, ConfigSection, DefaultSpec, DefaultValue, Env, Flags,
    InstanceSection, InstanceSectionError, Layers, Os, Registry, Reload, ResolvedConfig, Scope,
    Value, command,
};

use super::*;

/// The section as a binary registers it.
static REGISTRY: Registry = Registry::new(&[&SchedSection::SCHEMA]);

/// RFC-0012 §12's `sched.*` rows: name and default, in the table's order.
const RFC_KEYS: [(&str, Expected); 8] = [
    ("spin_us", Expected::Computed),
    ("prewake_us", Expected::Str("300us")),
    ("decode_stall_budget", Expected::Computed),
    ("prefill_inflight", Expected::Computed),
    ("prefill_share", Expected::Computed),
    ("max_queued", Expected::Int(64)),
    ("latency_decode_rows", Expected::Int(4)),
    ("max_pause", Expected::Str("30s")),
];

#[derive(Clone, Copy, Debug)]
enum Expected {
    Computed,
    Str(&'static str),
    Int(i64),
}

fn parse(text: &str) -> Value {
    text.parse().expect("test TOML parses")
}

/// The `sched` table of a policy, from `key = value` lines.
fn sched(text: &str) -> Value {
    let entries: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    parse(&format!("{{ {} }}", entries.join(", ")))
}

/// The section built from `values`, the `sched` table of a policy, by `config`'s builder. The
/// computed defaults stay `None`, as the scheduler applies their rules itself.
fn from_policy(values: &Value) -> Result<SchedSection, InstanceSectionError> {
    InstanceSection::<SchedSection>::from_policy(values, |_| None)
        .map(InstanceSection::into_section)
}

/// Each rejected key with its code, in the order the builder reports them.
fn rejections_of(values: &Value) -> Vec<(String, u16)> {
    match from_policy(values) {
        Err(InstanceSectionError::Rejected(problems)) => problems
            .into_iter()
            .map(|p| {
                let key = p.key.clone().unwrap_or_else(|| panic!("{p} names no key"));
                let code = p.code.unwrap_or_else(|| panic!("{p} has no code"));
                (key.into_string(), code.number())
            })
            .collect(),
        other => panic!("expected rejections, got {other:?}"),
    }
}

fn rejections(text: &str) -> Vec<(String, u16)> {
    rejections_of(&sched(text))
}

/// The same rejections in key order: the builder reports a table's keys in path order, which
/// is not the order #1656's parser used, so the multi-key cases compare as sets.
fn sorted(mut got: Vec<(String, u16)>) -> Vec<(String, u16)> {
    got.sort_unstable();
    got
}

#[test]
fn every_rfc_key_is_registered_with_its_default_scope_and_owner() {
    let schema = &SchedSection::SCHEMA;
    assert_eq!(schema.prefix, "sched");
    let names: Vec<&str> = schema.keys.iter().map(|k| k.name).collect();
    let expected: Vec<&str> = RFC_KEYS.iter().map(|(n, _)| *n).collect();
    assert_eq!(names, expected);
    for (name, default) in RFC_KEYS {
        let key = schema.key(name).unwrap();
        assert_eq!(key.path, format!("sched.{name}"));
        assert_eq!(key.scope, Scope::Instance, "{name}: RFC-0012 §12");
        assert_eq!(key.owner, "RFC-0012", "{name}");
        match (default, key.default) {
            (Expected::Computed, DefaultSpec::Computed(rule)) => assert!(!rule.is_empty()),
            (Expected::Str(s), DefaultSpec::Value(DefaultValue::Str(v))) => assert_eq!(s, v),
            (Expected::Int(i), DefaultSpec::Value(DefaultValue::Int(v))) => assert_eq!(i, v),
            (want, got) => panic!("{name}: default {got:?}, RFC-0012 §12 says {want:?}"),
        }
    }
    assert!(SchedSection::check_defaults().is_empty());
    assert!(REGISTRY.check().is_empty());
    // The two keys that change lane credits or captured buckets need a new plan.
    for (name, reload) in [
        ("prefill_inflight", Reload::Instance),
        ("latency_decode_rows", Reload::Instance),
        ("max_queued", Reload::Hot),
    ] {
        assert_eq!(schema.key(name).unwrap().reload, reload, "{name}");
    }
}

#[test]
fn show_resolved_lists_each_key_with_its_default_and_scope() {
    let resolved = ResolvedConfig::resolve(
        &REGISTRY,
        Layers {
            os: Os::Linux,
            files: &[],
            env: &Env::default(),
            flags: &Flags::new(),
        },
    )
    .expect("resolves");
    let shown = command::show_resolved(&resolved);
    // The block's header wording belongs to `config`; match its stable prefix and the rows.
    let (_, block) = shown
        .split_once("\n# instance keys:")
        .unwrap_or_else(|| panic!("no instance block:\n{shown}"));
    let rows: Vec<&str> = block.lines().skip(1).collect();
    let expected = [
        "sched.decode_stall_budget      # default, computed: 0.5 x the active sequence's predicted t_tok, at least 64 tokens of chunk",
        "sched.latency_decode_rows = 4  # default",
        "sched.max_pause = \"30s\"        # default",
        "sched.max_queued = 64          # default",
        "sched.prefill_inflight         # default, computed: D, the instance's stage count",
        "sched.prefill_share            # default, computed: G, the instance's group count",
        "sched.prewake_us = \"300us\"     # default",
        "sched.spin_us                  # default, computed: 50us on mains power, 10us on battery",
    ];
    assert_eq!(rows, expected, "\n{shown}");
}

#[test]
fn a_node_file_cannot_set_a_sched_key_outside_a_policy_table() {
    let files = [ConfigFile::new("/c.toml", "[sched]\nmax_queued = 8\n")];
    let rejected = ResolvedConfig::resolve(
        &REGISTRY,
        Layers {
            os: Os::Linux,
            files: &files,
            env: &Env::default(),
            flags: &Flags::new(),
        },
    )
    .expect_err("an instance key in a node file");
    let checked = Checked {
        files: files.to_vec(),
        outcome: Err(rejected),
    };
    let report = command::check(&checked, &REGISTRY);
    assert_eq!(report.exit, command::EXIT_INVALID);
    assert!(
        report
            .text
            .contains("`sched.max_queued` is an instance key"),
        "{}",
        report.text
    );
}

#[test]
fn an_empty_policy_gives_the_defaults() {
    let s = from_policy(&sched("")).unwrap();
    assert_eq!(
        s,
        SchedSection {
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
    assert_eq!(s.spin(false), Duration::from_nanos(50_000));
    assert_eq!(s.spin(true), Duration::from_nanos(10_000));
}

#[test]
fn a_policy_sets_every_key() {
    let s = from_policy(&sched(
        "spin_us = \"20us\"\nprewake_us = \"1ms\"\ndecode_stall_budget = \"25ms\"\n\
         prefill_inflight = 2\nprefill_share = 3\nmax_queued = 0\nlatency_decode_rows = 1\n\
         max_pause = \"2m\"\n",
    ))
    .unwrap();
    assert_eq!(s.spin(true), Duration::from_nanos(20_000));
    assert_eq!(s.prewake_us, Duration::from_millis(1));
    assert_eq!(s.decode_stall_budget, Some(Duration::from_millis(25)));
    assert_eq!(
        (
            s.prefill_inflight,
            s.prefill_share,
            s.max_queued,
            s.latency_decode_rows
        ),
        (Some(2), Some(3), 0, 1)
    );
    assert_eq!(s.max_pause, Duration::from_secs(120));
    // The bounds themselves are accepted.
    let edges = from_policy(&sched(
        "latency_decode_rows = 4\nprefill_inflight = 1\nprefill_share = 1\n\
         max_queued = 4294967295\nspin_us = \"0s\"\n",
    ))
    .unwrap();
    assert_eq!(edges.latency_decode_rows, 4);
    assert_eq!(edges.max_queued, u32::MAX);
}

/// Each invalid value is rejected at load with its RFC-0054 §8 code: 2 `wrong_type`,
/// 3 `unknown_key`, 4 `out_of_range`, 5 `bad_unit`.
#[test]
fn invalid_values_are_rejected_with_their_code() {
    let cases: &[(&str, &str, u16)] = &[
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
    let mut failures = Vec::new();
    for (text, key, code) in cases {
        let got = rejections(text);
        if got != [((*key).to_owned(), *code)] {
            failures.push(format!("{text}: {got:?}, expected [({key}, {code})]"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn every_invalid_value_in_a_policy_is_reported() {
    let got = rejections("spin_us = 50\nmax_queued = -1\ntypo = 1\nmax_pause = \"30\"\n");
    assert_eq!(
        sorted(got),
        sorted(vec![
            ("sched.typo".to_owned(), 3),
            ("sched.spin_us".to_owned(), 2),
            ("sched.max_queued".to_owned(), 4),
            ("sched.max_pause".to_owned(), 5),
        ])
    );
    // A range error is reported beside parse errors in other keys, not instead of them.
    assert_eq!(
        sorted(rejections("prefill_inflight = 0\nmax_queued = \"x\"\n")),
        sorted(vec![
            ("sched.max_queued".to_owned(), 2),
            ("sched.prefill_inflight".to_owned(), 4),
        ])
    );
    assert_eq!(
        sorted(rejections(
            "latency_decode_rows = 5\nspin_us = \"50\"\nprefill_share = 0\nbogus = 1\n"
        )),
        sorted(vec![
            ("sched.bogus".to_owned(), 3),
            ("sched.spin_us".to_owned(), 5),
            ("sched.prefill_share".to_owned(), 4),
            ("sched.latency_decode_rows".to_owned(), 4),
        ])
    );
    assert_eq!(
        rejections_of(&Value::Integer(1)),
        [("sched".to_owned(), Code::WrongType.number())]
    );
}

/// A node file's `[policy."<selector>"]` table is checked at load (RFC-0054 §8 phases 2 and 3):
/// each invalid `sched.*` value in it is one error with its code, and valid values load.
#[test]
fn invalid_values_in_a_policy_table_are_rejected_at_load() {
    let load = |text: &str| {
        let files = [ConfigFile::new("/c.toml", text)];
        ResolvedConfig::resolve(
            &REGISTRY,
            Layers {
                os: Os::Linux,
                files: &files,
                env: &Env::default(),
                flags: &Flags::new(),
            },
        )
    };
    let rejected = load(
        "[policy.\"qwen3*\".sched]\nlatency_decode_rows = 5\nspin_us = \"50\"\n\
         max_queued = \"64\"\nspin = \"50us\"\n",
    )
    .expect_err("invalid sched values in a policy table");
    let mut got: Vec<(Option<&str>, Option<u16>)> = rejected
        .errors
        .iter()
        .map(|p| (p.key.as_deref(), p.code.map(Code::number)))
        .collect();
    got.sort_unstable();
    assert_eq!(
        got,
        [
            (Some("policy.\"qwen3*\".sched.latency_decode_rows"), Some(4)),
            (Some("policy.\"qwen3*\".sched.max_queued"), Some(2)),
            (Some("policy.\"qwen3*\".sched.spin"), Some(3)),
            (Some("policy.\"qwen3*\".sched.spin_us"), Some(5)),
        ],
        "{rejected:#?}"
    );
    load("[policy.\"qwen3*\".sched]\nlatency_decode_rows = 2\nspin_us = \"20us\"\n")
        .expect("valid sched values in a policy table load");
}
