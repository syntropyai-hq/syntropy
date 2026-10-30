// SPDX-License-Identifier: MIT OR Apache-2.0

//! PROD-E5-T02's acceptance tests, per bind class (RFC-0050 §11.1, RFC-0054 §8): the `[api]`
//! section loaded through `syntropy-config`'s layering and validation with [`API_RULES`], and
//! the bind plan the daemon runs at start.

use std::path::{Path, PathBuf};

use syntropy_api::bind::{
    BindClass, CREATE_KEY_COMMAND, Certificate, StartRefusal, Transport, plan_binds,
};
use syntropy_api::config::{ApiSection, TlsSource};
use syntropy_api::rules::API_RULES;
use syntropy_config::{
    Code, ConfigFile, ConfigSection, Env, Flags, HostEnvironment, Layers, Os, Problem, Registry,
    ResolvedConfig, validate,
};

static REGISTRY: Registry = Registry::new(&[&ApiSection::SCHEMA]);

/// Loads `text` as the node's config file and runs phases 1–5 with [`API_RULES`]: the section,
/// or every error.
fn load(text: &str) -> Result<ApiSection, Vec<Problem>> {
    let files = [ConfigFile::new("/etc/syntropy/config.toml", text)];
    let env = Env::from_pairs(Os::Linux, std::iter::empty::<(&str, &str)>());
    let flags = Flags::new();
    let layers = Layers {
        os: Os::Linux,
        files: &files,
        env: &env,
        flags: &flags,
    };
    ResolvedConfig::resolve(&REGISTRY, layers)
        .and_then(|config| validate(&REGISTRY, config, API_RULES, &HostEnvironment))
        .map(|config| config.section::<ApiSection>().expect("a valid section"))
        .map_err(|rejected| rejected.errors)
}

fn codes(errors: &[Problem]) -> Vec<Option<Code>> {
    errors.iter().map(|p| p.code).collect()
}

/// A new directory for this test's files.
fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("syntropy-api-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir(&dir).unwrap();
    dir
}

fn toml_path(path: &Path) -> String {
    // TOML basic strings take `\` as an escape, so Windows paths go in literal strings.
    format!("'{}'", path.display())
}

#[test]
fn the_defaults_are_loopback_with_tls_auto() {
    let api = load("").unwrap();
    assert_eq!(
        api.bind,
        vec![
            "127.0.0.1:7433".parse().unwrap(),
            "[::1]:7433".parse().unwrap()
        ]
    );
    assert_eq!(api.tls, TlsSource::Auto);
    assert!(!api.allow_plaintext_lan);
    // Loopback needs no key and serves plain HTTP.
    let plan = plan_binds(&api, false).unwrap();
    assert!(
        plan.iter()
            .all(|l| l.class == BindClass::Loopback && l.transport == Transport::Plain)
    );
}

/// Criterion 1: a LAN bind without a key refuses to start and names the command.
#[test]
fn a_lan_bind_without_a_key_refuses_to_start_and_names_the_command() {
    for bind in [
        "192.168.1.10:7433",
        "0.0.0.0:7433",
        "[::]:7433",
        "8.8.8.8:7433",
    ] {
        let api = load(&format!("[api]\nbind = [\"127.0.0.1:7433\", \"{bind}\"]\n")).unwrap();
        let refusal = plan_binds(&api, false).unwrap_err();
        assert_eq!(
            refusal,
            StartRefusal::NoApiKey {
                binds: vec![bind.parse().unwrap()]
            }
        );
        let text = refusal.to_string();
        assert!(text.contains(CREATE_KEY_COMMAND), "{text}");
        assert!(
            text.contains("syntropy api-key create --name … --scope inference"),
            "{text}"
        );
        // With a key it starts.
        assert!(plan_binds(&api, true).is_ok());
    }
}

/// Criterion 2: `api.tls = "none"` on a LAN bind without the opt-in fails validation.
#[test]
fn tls_none_on_a_lan_bind_needs_the_opt_in() {
    let lan = "[api]\nbind = [\"192.168.1.10:7433\"]\ntls = \"none\"\n";
    let errors = load(lan).unwrap_err();
    assert_eq!(codes(&errors), vec![Some(Code::InsecureListener)]);
    assert_eq!(errors[0].key.as_deref(), Some("api.tls"));
    assert!(
        errors[0].message.contains("api.allow_plaintext_lan"),
        "{}",
        errors[0].message
    );
    assert!(errors[0].at.is_some(), "points at the file");

    // With the opt-in it loads, and the plan serves plain HTTP there.
    let api = load(&format!("{lan}allow_plaintext_lan = true\n")).unwrap();
    let plan = plan_binds(&api, true).unwrap();
    assert_eq!(plan[0].transport, Transport::Plain);

    // A wildcard needs a key like a LAN bind, and is refused plain without the opt-in too.
    let errors = load("[api]\nbind = [\"0.0.0.0:7433\"]\ntls = \"none\"\n").unwrap_err();
    assert_eq!(codes(&errors), vec![Some(Code::InsecureListener)]);
}

/// A wildcard listens on every interface, a public one included, so it is never served plain,
/// even with the opt-in: validation rejects it, and so does the plan.
#[test]
fn a_wildcard_bind_is_never_served_plain() {
    for wildcard in ["0.0.0.0:7433", "[::]:7433", "[::ffff:0.0.0.0]:7433"] {
        let text =
            format!("[api]\nbind = [\"{wildcard}\"]\ntls = \"none\"\nallow_plaintext_lan = true\n");
        let errors = load(&text).unwrap_err();
        assert_eq!(
            codes(&errors),
            vec![Some(Code::InsecureListener)],
            "{wildcard}"
        );
        assert!(
            errors[0].message.contains("wildcard"),
            "{}",
            errors[0].message
        );
        assert!(
            errors[0]
                .help
                .as_deref()
                .is_some_and(|h| h.contains("bind a specific LAN address")),
            "{:?}",
            errors[0].help
        );

        let api = ApiSection {
            bind: vec![wildcard.parse().unwrap()],
            tls: TlsSource::None,
            allow_plaintext_lan: true,
            local_socket: true,
        };
        let refusal = plan_binds(&api, true).unwrap_err();
        assert!(
            matches!(
                refusal,
                StartRefusal::PlainNotAllowed {
                    class: BindClass::Lan,
                    ..
                }
            ),
            "{wildcard}: {refusal:?}"
        );
        assert!(
            refusal.to_string().contains("bind a specific LAN address"),
            "{refusal}"
        );
    }
    // A specific LAN address beside it still may, with the opt-in.
    let api =
        load("[api]\nbind = [\"192.168.1.10:7433\"]\ntls = \"none\"\nallow_plaintext_lan = true\n")
            .unwrap();
    assert_eq!(
        plan_binds(&api, true).unwrap()[0].transport,
        Transport::Plain
    );
}

/// The opt-in never allows plain HTTP on a public address.
#[test]
fn tls_none_on_a_public_bind_fails_even_with_the_opt_in() {
    let text = "[api]\nbind = [\"8.8.8.8:7433\", \"100.64.0.1:7433\"]\ntls = \"none\"\n\
                allow_plaintext_lan = true\n";
    let errors = load(text).unwrap_err();
    assert_eq!(
        codes(&errors),
        vec![Some(Code::InsecureListener), Some(Code::InsecureListener)]
    );
    assert!(errors[0].message.contains("public address"));
    // The plan refuses it too, rather than serve plain HTTP.
    let api = ApiSection {
        bind: vec!["8.8.8.8:7433".parse().unwrap()],
        tls: TlsSource::None,
        allow_plaintext_lan: true,
        local_socket: true,
    };
    assert!(matches!(
        plan_binds(&api, true),
        Err(StartRefusal::PlainNotAllowed {
            class: BindClass::Public,
            ..
        })
    ));
}

/// Loopback serves plain HTTP whatever `api.tls` says, so `"none"` there is not checked.
#[test]
fn tls_none_on_loopback_only_binds_is_accepted() {
    let api = load("[api]\nbind = [\"127.0.0.1:7433\", \"[::1]:7433\"]\ntls = \"none\"\n").unwrap();
    assert_eq!(api.tls, TlsSource::None);
    assert!(
        plan_binds(&api, false)
            .unwrap()
            .iter()
            .all(|l| l.transport == Transport::Plain)
    );
}

/// Criterion 3: a `{ cert, key }` pair with a missing file fails validation.
#[test]
fn a_cert_key_pair_with_a_missing_file_fails_validation() {
    let dir = temp_dir("pair");
    let cert = dir.join("cert.pem");
    let key = dir.join("key.pem");
    std::fs::write(&cert, "-----BEGIN CERTIFICATE-----\n").unwrap();
    let text = format!(
        "[api]\nbind = [\"0.0.0.0:7433\"]\ntls = {{ cert = {}, key = {} }}\n",
        toml_path(&cert),
        toml_path(&key)
    );
    let errors = load(&text).unwrap_err();
    assert_eq!(codes(&errors), vec![Some(Code::PathUnusable)]);
    assert_eq!(errors[0].key.as_deref(), Some("api.tls.key"));
    assert!(
        errors[0].message.contains("does not exist"),
        "{}",
        errors[0].message
    );

    // A directory where a file belongs is refused too.
    std::fs::create_dir(&key).unwrap();
    let errors = load(&text).unwrap_err();
    assert!(
        errors[0].message.contains("not a file"),
        "{}",
        errors[0].message
    );
    std::fs::remove_dir(&key).unwrap();

    // Both files present: it loads, and the plan serves TLS from them.
    std::fs::write(&key, "-----BEGIN PRIVATE KEY-----\n").unwrap();
    let api = load(&text).unwrap();
    let plan = plan_binds(&api, true).unwrap();
    assert_eq!(
        plan[0].transport,
        Transport::Tls(Certificate::Files {
            cert: cert.clone(),
            key: key.clone()
        })
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Half a pair is the wrong shape, rejected when the value is read.
#[test]
fn half_a_cert_key_pair_is_the_wrong_shape() {
    let errors = load("[api]\ntls = { cert = '/etc/ssl/c.pem' }\n").unwrap_err();
    assert_eq!(codes(&errors), vec![Some(Code::WrongType)]);
    assert!(
        errors[0].message.contains("needs `key` too"),
        "{}",
        errors[0].message
    );
}

/// Criterion 4: a wildcard bind is served with TLS.
#[test]
fn a_wildcard_bind_is_served_with_tls() {
    for tls in ["", "tls = \"auto\"\n", "tls = \"local_ca\"\n"] {
        let api = load(&format!(
            "[api]\nbind = [\"0.0.0.0:7433\", \"[::]:7433\", \"127.0.0.1:7433\"]\n{tls}"
        ))
        .unwrap();
        let plan = plan_binds(&api, true).unwrap();
        assert_eq!(plan.len(), 3);
        for listener in &plan[..2] {
            assert_eq!(listener.class, BindClass::Lan, "{tls}");
            assert_eq!(
                listener.transport,
                Transport::Tls(Certificate::LocalCa),
                "{tls}"
            );
        }
        assert_eq!(plan[2].transport, Transport::Plain);
    }
}
