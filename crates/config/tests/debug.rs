// SPDX-License-Identifier: MIT OR Apache-2.0

//! The debug-key gates (PROD-E1-T05; RFC-0054 §7.1, ADR-0505 item 7, RFC-0060 §6.4).
//!
//! The acceptance criteria this file checks:
//! - in every build profile the three keys default to false. Nothing in the section depends on
//!   a `cfg` or a feature, and these tests run under each profile's cargo profile;
//! - with a key set, the loopback predicate is false for non-loopback peers;
//! - building a `ResolvedConfig` with any key on emits the warning, and with every key off
//!   emits none;
//! - the manifest-signer predicate allows the CI test key only when a gate is set and every API
//!   listener is loopback, and gives `manifest_signer_not_allowed` otherwise.

use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;

use syntropy_config::{
    Code, ConfigFile, ConfigSection, DebugApi, DebugSection, DefaultSpec, DefaultValue, Env, Flags,
    HostEnvironment, Layers, ManifestSignerNotAllowed, Origin, Os, Peer, Registry, Rejected,
    Reload, ResolvedConfig, Scan, Scope, Value, command, load_and_validate,
};

static REGISTRY: Registry = Registry::new(&[&DebugSection::SCHEMA]);

/// The build's profile as the test sees it, for assertion messages.
const PROFILE: &str = if cfg!(debug_assertions) {
    "a profile with debug assertions (dev, test-release)"
} else {
    "a profile without debug assertions (chaos, release)"
};

fn resolve(files: &[ConfigFile], env: &Env, flags: &Flags) -> Result<ResolvedConfig, Rejected> {
    ResolvedConfig::resolve(
        &REGISTRY,
        Layers {
            os: Os::Linux,
            files,
            env,
            flags,
        },
    )
}

fn no_env() -> Env {
    Env::from_pairs::<&str, &str>(Os::Linux, [])
}

fn defaults() -> ResolvedConfig {
    resolve(&[], &no_env(), &Flags::new()).expect("no layers resolve")
}

/// A gate state for each subset of the three keys.
fn every_gate_state() -> Vec<DebugSection> {
    (0..8u8)
        .map(|bits| DebugSection {
            parity_api: bits & 1 != 0,
            bench_api: bits & 2 != 0,
            telemetry_api: bits & 4 != 0,
        })
        .collect()
}

fn ip(text: &str) -> IpAddr {
    text.parse().unwrap()
}

#[test]
fn every_key_defaults_to_false() {
    let schema = &DebugSection::SCHEMA;
    let paths: Vec<_> = schema.keys.iter().map(|k| k.path).collect();
    assert_eq!(
        paths,
        ["debug.parity_api", "debug.bench_api", "debug.telemetry_api"]
    );
    for key in schema.keys {
        assert_eq!(
            key.default,
            DefaultSpec::Value(DefaultValue::Bool(false)),
            "{}, in {PROFILE}",
            key.path
        );
        // RFC-0054 §5 and §9.1: node keys, reloaded hot.
        assert_eq!(key.scope, Scope::Node, "{}", key.path);
        assert_eq!(key.reload, Reload::Hot, "{}", key.path);
    }
    assert_eq!(DebugSection::SCHEMA.keys.len(), DebugApi::ALL.len());
    for api in DebugApi::ALL {
        assert!(REGISTRY.key(api.key()).is_some(), "{api}");
    }
    assert!(DebugSection::check_defaults().is_empty());
    assert_eq!(DebugSection::default(), DebugSection::OFF);

    // With no layer setting them, the resolved keys are false, from the default.
    let config = defaults();
    for api in DebugApi::ALL {
        let resolved = config.get(api.key()).expect("registered");
        assert_eq!(
            resolved.value(),
            Some(&Value::Boolean(false)),
            "{api}, in {PROFILE}"
        );
        assert_eq!(resolved.origin, Origin::Default { computed: None });
    }
    assert_eq!(DebugSection::from_config(&config), DebugSection::OFF);
    assert_eq!(config.section::<DebugSection>(), Ok(DebugSection::OFF));
    assert!(!DebugSection::from_config(&config).any());
}

#[test]
fn a_registry_without_the_section_leaves_every_gate_shut() {
    static EMPTY: Registry = Registry::new(&[]);
    let env = Env::from_pairs(Os::Linux, [("SYNTROPY_DEBUG__PARITY_API", "true")]);
    let config = ResolvedConfig::resolve(
        &EMPTY,
        Layers {
            os: Os::Linux,
            files: &[],
            env: &env,
            flags: &Flags::new(),
        },
    )
    .expect("an unknown variable is a warning");
    assert_eq!(DebugSection::from_config(&config), DebugSection::OFF);
    // The only warning is the unknown variable's.
    assert_eq!(config.warnings().len(), 1, "{:?}", config.warnings());
    assert!(config.warnings()[0].message.contains("unknown variable"));
}

#[test]
fn a_gated_endpoint_answers_only_while_its_key_is_on_and_only_to_loopback() {
    let loopback = [
        Peer::LocalSocket,
        Peer::Tcp {
            listener: ip("127.0.0.1"),
            remote: ip("127.0.0.1"),
        },
        Peer::Tcp {
            listener: ip("::1"),
            remote: ip("::1"),
        },
        Peer::Tcp {
            listener: ip("127.0.0.1"),
            remote: ip("127.0.0.53"),
        },
        // An IPv4 peer on a dual-stack socket bound to IPv6 loopback.
        Peer::Tcp {
            listener: ip("::1"),
            remote: ip("::ffff:127.0.0.1"),
        },
    ];
    let not_loopback = [
        // A LAN peer on a wildcard listener.
        Peer::Tcp {
            listener: ip("0.0.0.0"),
            remote: ip("192.168.1.20"),
        },
        // A local client on a wildcard listener: a wildcard bind is a LAN bind (ADR-0505).
        Peer::Tcp {
            listener: ip("0.0.0.0"),
            remote: ip("127.0.0.1"),
        },
        Peer::Tcp {
            listener: ip("::"),
            remote: ip("::1"),
        },
        // A LAN listener, from the host itself and from the LAN.
        Peer::Tcp {
            listener: ip("192.168.1.5"),
            remote: ip("192.168.1.5"),
        },
        Peer::Tcp {
            listener: ip("192.168.1.5"),
            remote: ip("127.0.0.1"),
        },
        // An overlay listener.
        Peer::Tcp {
            listener: ip("100.64.0.7"),
            remote: ip("100.64.0.9"),
        },
        // A loopback listener reached by an off-host peer, as on a host that routes
        // `127.0.0.0/8` from outside.
        Peer::Tcp {
            listener: ip("127.0.0.1"),
            remote: ip("10.0.0.4"),
        },
        Peer::Tcp {
            listener: ip("::1"),
            remote: ip("fe80::1"),
        },
        Peer::Tcp {
            listener: ip("::1"),
            remote: ip("::ffff:192.168.1.20"),
        },
    ];
    for gates in every_gate_state() {
        for api in DebugApi::ALL {
            for peer in &loopback {
                assert!(peer.is_loopback(), "{peer:?}");
                assert_eq!(
                    gates.allows(api, peer),
                    gates.is_on(api),
                    "{api} with {gates:?} from {peer:?}"
                );
            }
            for peer in &not_loopback {
                assert!(!peer.is_loopback(), "{peer:?}");
                assert!(
                    !gates.allows(api, peer),
                    "{api} with {gates:?} from {peer:?}"
                );
            }
        }
    }
}

#[test]
fn a_resolved_config_warns_once_for_each_key_that_is_on() {
    // Every subset of the keys, each set by a different layer.
    for gates in every_gate_state() {
        let mut text = String::new();
        let mut pairs = Vec::new();
        let mut flags = Flags::new();
        // Keys left off are written as `false`, which must not warn.
        text.push_str(&format!("[debug]\nparity_api = {}\n", gates.parity_api));
        if gates.bench_api {
            pairs.push(("SYNTROPY_DEBUG__BENCH_API", "true"));
        }
        flags
            .set(&format!("debug.telemetry_api={}", gates.telemetry_api))
            .unwrap();
        let files = [ConfigFile::new("/etc/syntropy/config.toml", text)];
        let config = resolve(&files, &Env::from_pairs(Os::Linux, pairs), &flags)
            .expect("the layers resolve");

        assert_eq!(DebugSection::from_config(&config), gates);
        assert_eq!(config.section::<DebugSection>(), Ok(gates));
        let warnings = config.warnings();
        assert_eq!(
            warnings.len(),
            gates.on().count(),
            "{gates:?}: {warnings:?}"
        );
        assert_eq!(warnings.is_empty(), !gates.any());
        for (warning, api) in warnings.iter().zip(gates.on()) {
            let at = match api {
                DebugApi::Parity => Origin::File {
                    path: PathBuf::from("/etc/syntropy/config.toml"),
                    line: 2,
                },
                DebugApi::Bench => Origin::Env {
                    var: "SYNTROPY_DEBUG__BENCH_API".into(),
                },
                DebugApi::Telemetry => Origin::Cli {
                    flag: "--set debug.telemetry_api".into(),
                },
            };
            assert_eq!(warning.at.as_ref(), Some(&at), "{api}");
            // RFC-0054 §8 gives the warning no code; it is about the key.
            assert_eq!(warning.code, None, "{api}");
            assert_eq!(warning.key.as_deref(), Some(api.key()), "{api}");
            assert!(
                warning
                    .message
                    .starts_with(&format!("`{}` is on: ", api.key())),
                "{warning}"
            );
            assert!(warning.message.contains("RFC-0054 §7.1"), "{warning}");
        }
    }

    // The whole text of one warning, as the startup log prints it.
    let env = Env::from_pairs(Os::Linux, [("SYNTROPY_DEBUG__PARITY_API", "true")]);
    let config = resolve(&[], &env, &Flags::new()).unwrap();
    assert_eq!(
        config.warnings()[0].to_string(),
        "env SYNTROPY_DEBUG__PARITY_API: `debug.parity_api` is on: `POST /v1/syntropy/parity` \
         answers on loopback listeners and the local socket. It is a test-only surface \
         (RFC-0054 §7.1); turn it off when the test is done"
    );

    // A reload builds a new snapshot, which carries the warning again, or none once the key is
    // off.
    let reloaded = resolve(&[], &env, &Flags::new()).unwrap();
    assert_eq!(reloaded.warnings(), config.warnings());
    let off = Env::from_pairs(Os::Linux, [("SYNTROPY_DEBUG__PARITY_API", "false")]);
    assert!(
        resolve(&[], &off, &Flags::new())
            .unwrap()
            .warnings()
            .is_empty()
    );
}

#[test]
fn a_value_that_is_not_a_boolean_never_opens_a_gate() {
    // Layering takes the value as written; the type check of RFC-0054 §8 phase 3 rejects it
    // (PROD-E1-T03). Either way the gate stays shut and nothing warns.
    for text in ["\"true\"", "1", "\"yes\""] {
        let files = [ConfigFile::new(
            "/etc/syntropy/config.toml",
            format!("[debug]\nparity_api = {text}\n"),
        )];
        match resolve(&files, &no_env(), &Flags::new()) {
            Err(rejected) => assert!(!rejected.errors.is_empty(), "{text}"),
            Ok(config) => {
                assert_eq!(
                    DebugSection::from_config(&config),
                    DebugSection::OFF,
                    "{text}"
                );
                assert!(config.section::<DebugSection>().is_err(), "{text}");
                assert!(config.warnings().is_empty(), "{text}");
            }
        }
    }
}

#[test]
fn the_ci_test_key_needs_a_gate_and_loopback_api_listeners() {
    let addr = |text: &str| -> SocketAddr { text.parse().unwrap() };
    // Each listener set, with the listeners in it that are not loopback.
    let listener_sets: Vec<(Vec<SocketAddr>, Vec<SocketAddr>)> = vec![
        // The local socket only: it always qualifies.
        (vec![], vec![]),
        // The schema default of `api.bind`.
        (vec![addr("127.0.0.1:7433"), addr("[::1]:7433")], vec![]),
        (vec![addr("127.0.0.2:9000")], vec![]),
        (vec![addr("[::ffff:127.0.0.1]:7433")], vec![]),
        (vec![addr("0.0.0.0:7433")], vec![addr("0.0.0.0:7433")]),
        (vec![addr("[::]:7433")], vec![addr("[::]:7433")]),
        (
            vec![addr("127.0.0.1:7433"), addr("192.168.1.5:7433")],
            vec![addr("192.168.1.5:7433")],
        ),
        (
            vec![
                addr("100.64.0.7:7433"),
                addr("[::1]:7433"),
                addr("[fe80::1]:7433"),
            ],
            vec![addr("100.64.0.7:7433"), addr("[fe80::1]:7433")],
        ),
    ];
    for gates in every_gate_state() {
        let gated = gates.parity_api || gates.bench_api;
        for (bind, non_loopback) in &listener_sets {
            let result = gates.ci_test_manifest(bind);
            if gated && non_loopback.is_empty() {
                assert_eq!(result, Ok(()), "{gates:?} with {bind:?}");
                continue;
            }
            let refused = result.expect_err("refused");
            assert_eq!(
                refused,
                ManifestSignerNotAllowed {
                    no_gate: !gated,
                    non_loopback: non_loopback.clone(),
                },
                "{gates:?} with {bind:?}"
            );
            // The log line names the error and each condition that failed.
            let line = refused.to_string();
            assert!(line.starts_with("manifest_signer_not_allowed: "), "{line}");
            assert_eq!(
                line.contains("neither `debug.parity_api` nor `debug.bench_api` is on"),
                !gated,
                "{line}"
            );
            assert_eq!(
                line.contains("these `api.bind` listeners are not loopback"),
                !non_loopback.is_empty(),
                "{line}"
            );
            for listener in non_loopback {
                assert!(line.contains(&listener.to_string()), "{line}");
            }
        }
    }
    assert_eq!(
        ManifestSignerNotAllowed::CODE,
        Code::ManifestSignerNotAllowed
    );
    assert_eq!(ManifestSignerNotAllowed::CODE.number(), 15);
    assert_eq!(
        ManifestSignerNotAllowed::CODE.name(),
        "manifest_signer_not_allowed"
    );

    // Both conditions failing, in full.
    let refused = DebugSection {
        telemetry_api: true,
        ..DebugSection::OFF
    }
    .ci_test_manifest(&[addr("127.0.0.1:7433"), addr("0.0.0.0:7433")])
    .unwrap_err();
    assert_eq!(
        refused.to_string(),
        "manifest_signer_not_allowed: a conformance manifest signed by the CI test key is \
         accepted only while `debug.parity_api` or `debug.bench_api` is on and every `api.bind` \
         listener is loopback; neither `debug.parity_api` nor `debug.bench_api` is on; these \
         `api.bind` listeners are not loopback: 0.0.0.0:7433"
    );
}

#[test]
fn config_check_prints_the_warning_without_a_code() {
    let env = Env::from_pairs(Os::Linux, [("SYNTROPY_DEBUG__BENCH_API", "true")]);
    let scan = Scan {
        system_file: None,
        dropins: vec![],
        user_file: None,
        explicit: None,
    };
    let checked = load_and_validate(
        &REGISTRY,
        Os::Linux,
        &scan,
        &env,
        &Flags::new(),
        |_| None,
        &[],
        &HostEnvironment,
    );
    let report = command::check(&checked, &REGISTRY);
    // A warning: the configuration still loads.
    assert_eq!(report.exit, command::EXIT_OK, "{}", report.text);
    assert!(
        report
            .text
            .contains("warning: `debug.bench_api` is on: `BenchControls` on"),
        "{}",
        report.text
    );
    assert!(
        report.text.contains("  --> env SYNTROPY_DEBUG__BENCH_API"),
        "{}",
        report.text
    );
}
