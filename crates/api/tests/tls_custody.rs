// SPDX-License-Identifier: MIT OR Apache-2.0

//! PROD-E5-T03's custody criterion as a review test over the source (RFC-0050 §11.1.1;
//! ADR-0505 item 6): "no code path copies a CA key to another node or requests issuance from the
//! coordinator". The CA module may reach no network, IPC, cluster or process code, and no public
//! function in it returns key material, so a key can leave it only through its own file.

use std::fs;
use std::path::Path;

/// The CA module's source files, with comments and doc comments removed.
fn tls_sources() -> Vec<(String, String)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tls");
    let mut files: Vec<(String, String)> = fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|e| e == "rs"))
        .map(|path| {
            let code: String = fs::read_to_string(&path)
                .unwrap()
                .lines()
                .map(|line| line.split("//").next().unwrap_or(""))
                .collect::<Vec<_>>()
                .join("\n");
            (
                path.file_name().unwrap().to_string_lossy().into_owned(),
                code,
            )
        })
        .collect();
    files.sort();
    files
}

#[test]
fn the_ca_module_reaches_no_network_ipc_cluster_or_process_code() {
    let sources = tls_sources();
    assert!(sources.iter().any(|(name, _)| name == "ca.rs"));
    // The test module drives TLS over in-memory buffers, never a socket, so it is held to the same
    // list.
    let forbidden = [
        "syntropy_cluster",
        "syntropy_ipc",
        "syntropy_daemon",
        "syntropy_net::ctl",
        "syntropy_net::frame",
        "syntropy_net::endpoint",
        "std::net::Tcp",
        "std::net::Udp",
        "TcpStream",
        "TcpListener",
        "UdpSocket",
        "process::Command",
        "Command::new",
        "tokio",
    ];
    for (name, code) in &sources {
        for token in forbidden {
            assert!(!code.contains(token), "{name} uses `{token}`");
        }
    }
}

#[test]
fn no_public_function_of_the_ca_module_returns_key_material() {
    let key_types = [
        "KeyPair",
        "PrivateKeyDer",
        "PrivatePkcs8KeyDer",
        "SigningKey",
        "CaFile",
    ];
    for (name, code) in tls_sources() {
        for function in code.split("pub fn ").skip(1) {
            // The signature, up to the body: its return type is what follows `->`.
            let signature = function.split('{').next().unwrap();
            let Some((_, returns)) = signature.split_once("->") else {
                continue;
            };
            for key_type in key_types {
                assert!(
                    !returns.contains(key_type),
                    "{name}: `pub fn {}` returns {key_type}",
                    signature.split('(').next().unwrap()
                );
            }
        }
    }
    // Nothing that holds a key can be serialised: only the CA file's private layout in `ca.rs`
    // is.
    let sources = tls_sources();
    for type_name in ["LocalCa", "Leaf", "LeafResolver"] {
        let declaration = format!("pub struct {type_name} {{");
        let (name, code) = sources
            .iter()
            .find(|(_, code)| code.contains(&declaration))
            .unwrap_or_else(|| panic!("no `{declaration}`"));
        let before = &code[..code.find(&declaration).unwrap()];
        // With comments removed, the attributes are the lines right above the declaration.
        let attributes = before.trim_end().rsplit("\n\n").next().unwrap();
        assert!(
            !attributes.contains("Serialize"),
            "{name}: {type_name} derives Serialize"
        );
    }
}
