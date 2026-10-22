// SPDX-License-Identifier: MIT OR Apache-2.0

//! The local CA against PROD-E5-T03's criteria (RFC-0050 §11.1.1): generation, custody and
//! replacement; the name constraints, both as written and as a TLS client enforces them; leaf
//! issuance and refusal; and rotation under open connections.

use std::fs;
use std::io::{Read, Write};
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rcgen::{
    BasicConstraints, CertificateParams, ExtendedKeyUsagePurpose, IsCa, KeyPair, KeyUsagePurpose,
};
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, ServerName};
use rustls::{ClientConfig, ClientConnection, RootCertStore, ServerConfig, ServerConnection};
use syntropy_net::crypto::SigningKey;
use x509_parser::extensions::GeneralName;

use super::*;

/// 2026-10-22T00:00:00Z.
const NOW: u64 = 1_792_627_200;

fn now() -> SystemTime {
    UNIX_EPOCH + Duration::from_secs(NOW)
}

fn ip(s: &str) -> IpAddr {
    s.parse().unwrap()
}

fn node_key(seed: u8) -> SigningKey {
    SigningKey::from_seed(&[seed; 32])
}

/// `box.local` and `llm.home.arpa`, a LAN interface at 192.168.1.37 and an overlay address.
fn scope() -> CaScope {
    CaScope::new(
        ["box", "box.local", "llm.home.arpa"],
        &[ip("192.168.1.37")],
        &[ip("100.101.102.103")],
    )
    .unwrap()
    .0
}

/// A scratch `tls` directory, removed when dropped.
struct Dir(PathBuf);

impl Dir {
    fn new(name: &str) -> Dir {
        let root =
            std::env::temp_dir().join(format!("syntropy-api-tls-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        Dir(root)
    }

    fn path(&self) -> PathBuf {
        self.0.join("tls")
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn read_json(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn provider() -> Arc<rustls::crypto::CryptoProvider> {
    Arc::new(rustls::crypto::ring::default_provider())
}

fn server_config(resolver: Arc<LeafResolver>) -> Arc<ServerConfig> {
    Arc::new(
        ServerConfig::builder_with_provider(provider())
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_no_client_auth()
            .with_cert_resolver(resolver),
    )
}

fn client_config(ca: &CertificateDer<'static>) -> Arc<ClientConfig> {
    let mut roots = RootCertStore::empty();
    roots.add(ca.clone()).unwrap();
    Arc::new(
        ClientConfig::builder_with_provider(provider())
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_root_certificates(roots)
            .with_no_client_auth(),
    )
}

/// Moves TLS records between the two ends, in memory, until neither has more to send.
fn pump(client: &mut ClientConnection, server: &mut ServerConnection) -> Result<(), rustls::Error> {
    loop {
        let mut moved = false;
        while client.wants_write() {
            let mut records = Vec::new();
            client.write_tls(&mut records).unwrap();
            let mut records = records.as_slice();
            while !records.is_empty() {
                server.read_tls(&mut records).unwrap();
                server.process_new_packets()?;
            }
            moved = true;
        }
        while server.wants_write() {
            let mut records = Vec::new();
            server.write_tls(&mut records).unwrap();
            let mut records = records.as_slice();
            while !records.is_empty() {
                client.read_tls(&mut records).unwrap();
                client.process_new_packets()?;
            }
            moved = true;
        }
        if !moved {
            return Ok(());
        }
    }
}

/// A TLS 1.3 connection to `name`, handshaken.
fn connect(
    client: &Arc<ClientConfig>,
    server: &Arc<ServerConfig>,
    name: &str,
) -> Result<(ClientConnection, ServerConnection), rustls::Error> {
    let name = ServerName::try_from(name.to_owned()).unwrap();
    let mut c = ClientConnection::new(Arc::clone(client), name)?;
    let mut s = ServerConnection::new(Arc::clone(server))?;
    pump(&mut c, &mut s)?;
    assert!(!c.is_handshaking() && !s.is_handshaking());
    Ok((c, s))
}

/// Sends `message` both ways over the connection and checks that each end reads it.
fn exchange(c: &mut ClientConnection, s: &mut ServerConnection, message: &[u8]) {
    let mut buf = vec![0u8; message.len()];
    c.writer().write_all(message).unwrap();
    pump(c, s).unwrap();
    s.reader().read_exact(&mut buf).unwrap();
    assert_eq!(buf, message);
    s.writer().write_all(message).unwrap();
    pump(c, s).unwrap();
    c.reader().read_exact(&mut buf).unwrap();
    assert_eq!(buf, message);
}

fn served_leaf(c: &ClientConnection) -> CertificateDer<'static> {
    c.peer_certificates().unwrap()[0].clone().into_owned()
}

#[test]
fn a_ca_is_created_then_loaded_with_its_node_key_binding() {
    let dir = Dir::new("create");
    let key = node_key(1);
    let (ca, opened) = LocalCa::open(&dir.path(), &scope(), &key, now()).unwrap();
    assert_eq!(opened, CaOpen::Created);
    ca.binding().verify(ca.certificate_der()).unwrap();
    assert_eq!(ca.binding().node_key(), &key.public_key());
    assert!(
        ca.certificate_pem()
            .starts_with("-----BEGIN CERTIFICATE-----")
    );
    let later = now() + Duration::from_secs(86_400);
    let (again, opened) = LocalCa::open(&dir.path(), &scope(), &key, later).unwrap();
    assert_eq!(opened, CaOpen::Loaded);
    assert_eq!(again.certificate_pem(), ca.certificate_pem());
    assert_eq!(again.fingerprint(), ca.fingerprint());
    assert_eq!(LocalCa::last_replacement(&dir.path()).unwrap(), None);
    let names: Vec<String> = fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    assert_eq!(names, [CA_FILE]);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = |p: &Path| fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&dir.path()), 0o700);
        assert_eq!(mode(&dir.path().join(CA_FILE)), 0o600);
    }
    // The binding is a node-key signature over 0x01 ‖ the CA certificate's DER: it fails for a
    // certificate that differs in one bit, and under any other node key.
    let mut other = ca.certificate_der().to_vec();
    *other.last_mut().unwrap() ^= 1;
    assert!(ca.binding().verify(&other).is_err());
    let foreign = CaBinding::sign(&node_key(2), ca.certificate_der());
    assert_ne!(foreign.node_key(), ca.binding().node_key());
    assert_eq!(CaBinding::statement(b"certificate")[0], CaBinding::VERSION);
}

/// The binding statement's bytes, pinned: a node-key Ed25519 signature in context
/// `syntropy-tls-ca-v1` over `0x01 ‖ the CA certificate's DER` (RFC-0034 §3, revised
/// 2026-10-22). The certificate here is a stand-in, a DER SEQUENCE of the 64 bytes 01 … 40: the
/// statement carries whatever bytes it is given. The expected signature was computed with
/// OpenSSL 3.6.4 over `"syntropy-tls-ca-v1" ‖ 01 ‖ 30 40 01 … 40` with the seed 00 01 … 1f, not
/// with this code.
#[test]
fn the_binding_signs_the_versioned_certificate_in_its_own_context() {
    let seed: [u8; 32] = std::array::from_fn(|i| i as u8);
    let key = SigningKey::from_seed(&seed);
    let mut certificate = vec![0x30, 0x40];
    certificate.extend(1..=64u8);
    assert_eq!(
        CaBinding::statement(&certificate),
        [&[0x01][..], &certificate].concat()
    );
    let binding = CaBinding::sign(&key, &certificate);
    assert_eq!(
        hex(binding.node_key()),
        "03a107bff3ce10be1d70dd18e74bc09967e4d6309ba50d5f1ddc8664125531b8"
    );
    assert_eq!(
        hex(binding.signature()),
        "399fc64aec80cc59d17034712ffd5e1104a55a660742f9a5f8273b475f1fea39\
         4d052a42d7c6de7138a02f63f5af78d4ac9f6bbfff8d11f18f13f06de8d07407"
    );
    binding.verify(&certificate).unwrap();
}

/// The served PEM is built from the DER the binding covers: base64 per RFC 4648 §10's vectors,
/// in one block that decodes back to the same bytes and matches what rcgen would write.
#[test]
fn the_served_pem_is_the_bound_certificate_and_nothing_else() {
    for (bytes, text) in [
        ("", ""),
        ("f", "Zg=="),
        ("fo", "Zm8="),
        ("foo", "Zm9v"),
        ("foob", "Zm9vYg=="),
        ("fooba", "Zm9vYmE="),
        ("foobar", "Zm9vYmFy"),
    ] {
        assert_eq!(ca::base64(bytes.as_bytes()), text);
    }
    let dir = Dir::new("pem");
    let (ca, _) = LocalCa::open(&dir.path(), &scope(), &node_key(1), now()).unwrap();
    let pem = ca.certificate_pem();
    let decoded: Vec<CertificateDer<'static>> = CertificateDer::pem_slice_iter(pem.as_bytes())
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(decoded, [ca.certificate_der().clone()]);
    assert!(pem.starts_with("-----BEGIN CERTIFICATE-----\n"));
    assert!(pem.ends_with("\n-----END CERTIFICATE-----\n"));
    assert!(pem.lines().all(|line| line.len() <= 64));
    let (loaded, _) = LocalCa::open(&dir.path(), &scope(), &node_key(1), now()).unwrap();
    assert_eq!(loaded.certificate_pem(), pem);
}

fn hex_bytes(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
}

/// `dir`'s CA key, read from its file the way anyone who can read the file could.
fn stored_key(dir: &Path) -> KeyPair {
    let key = hex_bytes(read_json(&dir.join(CA_FILE))["key"].as_str().unwrap());
    KeyPair::from_pkcs8_der_and_sign_algo(
        &rustls::pki_types::PrivatePkcs8KeyDer::from(&key[..]),
        &rcgen::PKCS_ECDSA_P256_SHA256,
    )
    .unwrap()
}

/// Puts `certificate` in `dir`'s CA file, and with `rebind` also a binding re-signed over it
/// by that node key, as a holder of the node key could. Returns the bytes written.
fn plant(dir: &Path, certificate: &[u8], rebind: Option<&SigningKey>) -> Vec<u8> {
    let path = dir.join(CA_FILE);
    let mut file = read_json(&path);
    file["certificate"] = hex(certificate).into();
    if let Some(key) = rebind {
        file["binding"] = hex(CaBinding::sign(key, certificate).signature()).into();
    }
    let bytes = serde_json::to_vec(&file).unwrap();
    fs::write(&path, &bytes).unwrap();
    bytes
}

/// Opens `dir`'s CA and expects it refused for `what`, with the file left as `planted`.
fn assert_refused(dir: &Path, planted: &[u8], what: &str) {
    match LocalCa::open(dir, &scope(), &node_key(1), now()) {
        Err(TlsError::Corrupt { what: refused, .. }) => assert_eq!(refused, what),
        other => panic!("expected `{what}`: {other:?}"),
    }
    assert_eq!(fs::read(dir.join(CA_FILE)).unwrap(), planted);
}

/// A certificate over the CA's key with the CA's names but other constraints.
fn re_minted(key: &KeyPair, edit: impl FnOnce(&mut CertificateParams)) -> Vec<u8> {
    let mut params = CertificateParams::default();
    params.distinguished_name = ca::common_name(&ca::ca_name(&node_key(1).public_key()));
    params.is_ca = IsCa::Ca(BasicConstraints::Constrained(0));
    params.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
    params.name_constraints = Some(ca::name_constraints(&scope()));
    edit(&mut params);
    params.self_signed(key).unwrap().der().to_vec()
}

/// Review probe NC3: a CA certificate re-minted over the same key without its name constraints
/// is refused, under the old binding and under one re-signed over it, and so is one with
/// another scope's constraints. Nothing is served from such a file.
#[test]
fn a_re_minted_certificate_over_the_ca_key_is_refused() {
    let dir = Dir::new("re-minted");
    LocalCa::open(&dir.path(), &scope(), &node_key(1), now()).unwrap();
    let key = stored_key(&dir.path());
    let unconstrained = re_minted(&key, |p| {
        p.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        p.name_constraints = None;
    });
    let planted = plant(&dir.path(), &unconstrained, None);
    assert_refused(
        &dir.path(),
        &planted,
        "the node-key binding does not verify",
    );
    let planted = plant(&dir.path(), &unconstrained, Some(&node_key(1)));
    assert_refused(
        &dir.path(),
        &planted,
        "the certificate's basic constraints are not a leaf-only CA's",
    );
    let no_constraints = re_minted(&key, |p| p.name_constraints = None);
    let planted = plant(&dir.path(), &no_constraints, Some(&node_key(1)));
    assert_refused(
        &dir.path(),
        &planted,
        "the certificate lacks an extension this node mints",
    );
    let (public, _) = CaScope::new(["example.com"], &[ip("192.168.1.37")], &[]).unwrap();
    let elsewhere = re_minted(&key, |p| {
        p.name_constraints = Some(ca::name_constraints(&public))
    });
    let planted = plant(&dir.path(), &elsewhere, Some(&node_key(1)));
    assert_refused(
        &dir.path(),
        &planted,
        "the certificate's name constraints are not its scope's",
    );
    let extra = re_minted(&key, |p| {
        p.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
    });
    let planted = plant(&dir.path(), &extra, Some(&node_key(1)));
    assert_refused(
        &dir.path(),
        &planted,
        "the certificate carries an extension this node does not mint",
    );
    let usage = re_minted(&key, |p| {
        p.key_usages.push(KeyUsagePurpose::DigitalSignature)
    });
    let planted = plant(&dir.path(), &usage, Some(&node_key(1)));
    assert_refused(
        &dir.path(),
        &planted,
        "the certificate's key usages are not the CA's",
    );
    let renamed = re_minted(&key, |p| {
        p.distinguished_name = ca::common_name("example.com")
    });
    let planted = plant(&dir.path(), &renamed, Some(&node_key(1)));
    assert_refused(
        &dir.path(),
        &planted,
        "the certificate is not named for this node's CA",
    );
}

/// Review probe: a CA certificate with one byte of its self-signature flipped is refused, under
/// the old binding and under one re-signed over it.
#[test]
fn a_certificate_with_a_broken_self_signature_is_refused() {
    let dir = Dir::new("self-signature");
    let (ca, _) = LocalCa::open(&dir.path(), &scope(), &node_key(1), now()).unwrap();
    // The certificate ends with the signature's `s`, so its last byte is a signature byte.
    let mut broken = ca.certificate_der().to_vec();
    *broken.last_mut().unwrap() ^= 0x01;
    let planted = plant(&dir.path(), &broken, None);
    assert_refused(
        &dir.path(),
        &planted,
        "the node-key binding does not verify",
    );
    let planted = plant(&dir.path(), &broken, Some(&node_key(1)));
    assert_refused(
        &dir.path(),
        &planted,
        "the certificate's self-signature does not verify",
    );
}

/// Review probe: another genuine certificate over the same CA key, as this code mints it for
/// another scope, with its own valid binding, is refused in place of the stored one: the
/// binding covers the certificate, not the key, and the certificate must carry the stored
/// scope's constraints.
#[test]
fn a_swapped_certificate_over_the_same_key_is_refused() {
    let dir = Dir::new("swapped");
    LocalCa::open(&dir.path(), &scope(), &node_key(1), now()).unwrap();
    let (other, _) = CaScope::new(["box.local"], &[ip("10.0.0.5")], &[]).unwrap();
    let swapped = LocalCa::with_key(
        &dir.path(),
        &other,
        &node_key(1),
        now(),
        stored_key(&dir.path()),
    )
    .unwrap();
    let path = dir.path().join(CA_FILE);
    let good = fs::read(&path).unwrap();
    let planted = plant(&dir.path(), swapped.certificate_der(), None);
    assert_refused(
        &dir.path(),
        &planted,
        "the node-key binding does not verify",
    );
    let mut file = read_json(&path);
    file["binding"] = hex(swapped.binding().signature()).into();
    let planted = serde_json::to_vec(&file).unwrap();
    fs::write(&path, &planted).unwrap();
    assert_refused(
        &dir.path(),
        &planted,
        "the certificate's name constraints are not its scope's",
    );
    // The stored CA, put back, still loads.
    fs::write(&path, &good).unwrap();
    let (_, opened) = LocalCa::open(&dir.path(), &scope(), &node_key(1), now()).unwrap();
    assert_eq!(opened, CaOpen::Loaded);
}

#[test]
fn the_ca_certificate_is_a_leaf_only_ca_with_critical_name_constraints() {
    let dir = Dir::new("constraints");
    let (ca, _) = LocalCa::open(&dir.path(), &scope(), &node_key(1), now()).unwrap();
    let (_, cert) = x509_parser::parse_x509_certificate(ca.certificate_der()).unwrap();
    let basic = cert.basic_constraints().unwrap().unwrap();
    assert!(basic.critical);
    assert!(basic.value.ca);
    assert_eq!(basic.value.path_len_constraint, Some(0));
    let usage = cert.key_usage().unwrap().unwrap().value;
    assert!(usage.key_cert_sign() && usage.crl_sign() && !usage.digital_signature());
    let constraints = cert.name_constraints().unwrap().unwrap();
    assert!(constraints.critical, "name constraints must be critical");
    let permitted: Vec<String> = constraints
        .value
        .permitted_subtrees
        .as_ref()
        .unwrap()
        .iter()
        .map(|s| format!("{:?}", s.base))
        .collect();
    assert_eq!(
        permitted,
        [
            format!("{:?}", GeneralName::DNSName("box.local")),
            format!("{:?}", GeneralName::DNSName("llm.home.arpa")),
            format!(
                "{:?}",
                GeneralName::IPAddress(&[100, 101, 102, 103, 255, 255, 255, 255])
            ),
            format!(
                "{:?}",
                GeneralName::IPAddress(&[192, 168, 1, 0, 255, 255, 255, 0])
            ),
        ]
    );
    // No IPv6 subnet is served, so the CA may issue for no IPv6 address at all.
    let excluded = constraints.value.excluded_subtrees.as_ref().unwrap();
    assert_eq!(excluded.len(), 1);
    assert_eq!(
        format!("{:?}", excluded[0].base),
        format!("{:?}", GeneralName::IPAddress(&[0; 32]))
    );
    let validity = cert.validity();
    let span = validity.not_after.timestamp() - validity.not_before.timestamp();
    assert_eq!(span as u64, CA_VALIDITY.as_secs() + 3600);
}

#[test]
fn a_new_scope_replaces_the_ca_and_records_it_for_sec_006() {
    let dir = Dir::new("rescope");
    let key = node_key(1);
    let (old, _) = LocalCa::open(&dir.path(), &scope(), &key, now()).unwrap();
    // The node moved to another network.
    let (moved, _) = CaScope::new(["box.local"], &[ip("10.20.30.40")], &[]).unwrap();
    let later = now() + Duration::from_secs(3600);
    let (new, opened) = LocalCa::open(&dir.path(), &moved, &key, later).unwrap();
    let CaOpen::Replaced(record) = opened else {
        panic!("not replaced: {opened:?}");
    };
    assert_eq!(record.reason, ReplaceReason::ScopeChanged);
    assert_eq!(record.replaced_at, NOW + 3600);
    assert_eq!(record.old_fingerprint, hex(&old.fingerprint()));
    assert_eq!(record.new_fingerprint, hex(&new.fingerprint()));
    assert_ne!(old.fingerprint(), new.fingerprint());
    assert_eq!(new.scope(), &moved);
    assert_eq!(
        LocalCa::last_replacement(&dir.path()).unwrap(),
        Some(record)
    );
    // The new CA is the one stored, and it issues for the new network only.
    let (loaded, opened) = LocalCa::open(&dir.path(), &moved, &key, later).unwrap();
    assert_eq!(opened, CaOpen::Loaded);
    assert_eq!(loaded.fingerprint(), new.fingerprint());
    let names = LeafNames::new(["box.local"], [ip("192.168.1.37")]);
    assert!(matches!(
        loaded.issue_leaf(&names, later),
        Err(TlsError::OutsideScope(_))
    ));
}

#[test]
fn a_new_node_key_replaces_the_ca() {
    let dir = Dir::new("rekey");
    LocalCa::open(&dir.path(), &scope(), &node_key(1), now()).unwrap();
    let (new, opened) = LocalCa::open(&dir.path(), &scope(), &node_key(2), now()).unwrap();
    let CaOpen::Replaced(record) = opened else {
        panic!("not replaced: {opened:?}");
    };
    assert_eq!(record.reason, ReplaceReason::NodeKeyChanged);
    assert_eq!(new.binding().node_key(), &node_key(2).public_key());
    new.binding().verify(new.certificate_der()).unwrap();
}

#[test]
fn a_damaged_ca_file_is_refused_and_left_as_it_is() {
    let dir = Dir::new("damaged");
    let key = node_key(1);
    LocalCa::open(&dir.path(), &scope(), &key, now()).unwrap();
    let path = dir.path().join(CA_FILE);
    let good = read_json(&path);
    let other = Dir::new("damaged-other");
    LocalCa::open(&other.path(), &scope(), &key, now()).unwrap();
    let foreign = read_json(&other.path().join(CA_FILE));
    let flip = |text: &str| {
        let mut bytes = text.as_bytes().to_vec();
        bytes[10] = if bytes[10] == b'0' { b'1' } else { b'0' };
        String::from_utf8(bytes).unwrap()
    };
    type Edit<'e> = Box<dyn Fn(&mut serde_json::Value) + 'e>;
    let edits: Vec<(&str, Edit<'_>)> = vec![
        (
            "binding",
            Box::new(|v| v["binding"] = flip(v["binding"].as_str().unwrap()).into()),
        ),
        (
            "another CA's certificate",
            Box::new(|v| v["certificate"] = foreign["certificate"].clone()),
        ),
        (
            "another CA's key",
            Box::new(|v| v["key"] = foreign["key"].clone()),
        ),
        ("odd hex", Box::new(|v| v["key"] = "abc".into())),
        (
            "the certificate as PEM text",
            Box::new(|v| v["certificate"] = "-----BEGIN CERTIFICATE-----".into()),
        ),
        ("format 2", Box::new(|v| v["format"] = 2.into())),
        (
            "a public subnet",
            Box::new(|v| v["scope"]["subnets"][0] = "8.8.8.0/24".into()),
        ),
        (
            "a single-label name",
            Box::new(|v| v["scope"]["dns_names"][0] = "box".into()),
        ),
    ];
    for (name, edit) in edits {
        let mut damaged = good.clone();
        edit(&mut damaged);
        let bytes = serde_json::to_vec(&damaged).unwrap();
        fs::write(&path, &bytes).unwrap();
        let result = LocalCa::open(&dir.path(), &scope(), &key, now());
        assert!(
            matches!(result, Err(TlsError::Corrupt { .. })),
            "{name}: {result:?}"
        );
        assert_eq!(
            fs::read(&path).unwrap(),
            bytes,
            "{name}: the file was changed"
        );
    }
    fs::write(&path, b"not json").unwrap();
    assert!(matches!(
        LocalCa::open(&dir.path(), &scope(), &key, now()),
        Err(TlsError::Corrupt { .. })
    ));
}

#[test]
fn a_leaf_carries_its_names_for_30_days_and_chains_to_the_ca() {
    let dir = Dir::new("leaf");
    let (ca, _) = LocalCa::open(&dir.path(), &scope(), &node_key(1), now()).unwrap();
    let names = LeafNames::new(
        ["box.local", "llm.home.arpa"],
        [ip("192.168.1.37"), ip("100.101.102.103")],
    );
    let leaf = ca.issue_leaf(&names, now()).unwrap();
    assert_eq!(leaf.not_after(), now() + LEAF_VALIDITY);
    assert_eq!(leaf.certified_key().cert[1], *ca.certificate_der());
    let (_, cert) = x509_parser::parse_x509_certificate(leaf.certificate_der()).unwrap();
    let sans: Vec<String> = cert
        .subject_alternative_name()
        .unwrap()
        .unwrap()
        .value
        .general_names
        .iter()
        .map(|n| format!("{n:?}"))
        .collect();
    assert_eq!(
        sans,
        [
            format!("{:?}", GeneralName::DNSName("box.local")),
            format!("{:?}", GeneralName::DNSName("llm.home.arpa")),
            format!("{:?}", GeneralName::IPAddress(&[100, 101, 102, 103])),
            format!("{:?}", GeneralName::IPAddress(&[192, 168, 1, 37])),
        ]
    );
    assert!(
        !cert
            .basic_constraints()
            .unwrap()
            .is_some_and(|b| b.value.ca)
    );
    assert!(
        cert.extended_key_usage()
            .unwrap()
            .unwrap()
            .value
            .server_auth
    );
    let validity = cert.validity();
    assert_eq!(
        validity.not_after.timestamp() as u64,
        NOW + LEAF_VALIDITY.as_secs()
    );
}

#[test]
fn a_leaf_outside_the_scope_is_refused_before_anything_is_signed() {
    let dir = Dir::new("refuse");
    let (ca, _) = LocalCa::open(&dir.path(), &scope(), &node_key(1), now()).unwrap();
    for (names, outside) in [
        (LeafNames::new(["example.com"], []), "example.com"),
        (LeafNames::new(["box.local"], [ip("8.8.8.8")]), "8.8.8.8"),
        (
            LeafNames::new(["box.local"], [ip("192.168.99.5")]),
            "192.168.99.5",
        ),
        (LeafNames::new(["x.box.local"], []), "x.box.local"),
        (LeafNames::new(["box"], []), "box"),
        (LeafNames::new([], [ip("fd00::1")]), "fd00::1"),
    ] {
        match ca.issue_leaf(&names, now()) {
            Err(TlsError::OutsideScope(name)) => assert_eq!(name, outside),
            other => panic!("{outside}: {other:?}"),
        }
    }
    assert!(matches!(
        ca.issue_leaf(&LeafNames::new([], []), now()),
        Err(TlsError::NoNames)
    ));
}

/// The criterion "a client rejects any such leaf because of the name constraints": a leaf the CA
/// signs without the scope check, for a public name, a public address, a local address outside
/// the served subnets, or an IPv6 address when none is served, fails a TLS client's handshake,
/// while an in-scope leaf from the same CA passes.
#[test]
fn a_client_rejects_a_leaf_the_name_constraints_do_not_allow() {
    let dir = Dir::new("client");
    // The client checks validity against the real clock.
    let (ca, _) = LocalCa::open(&dir.path(), &scope(), &node_key(1), SystemTime::now()).unwrap();
    let client = client_config(ca.certificate_der());
    let handshake = |names: &LeafNames, server_name: &str| {
        let leaf = ca.sign_leaf(names, SystemTime::now()).unwrap();
        let server = server_config(Arc::new(LeafResolver::new(&leaf)));
        connect(&client, &server, server_name).map(|_| ())
    };
    handshake(&LeafNames::new(["box.local"], []), "box.local").unwrap();
    handshake(
        &LeafNames::new(["box.local"], [ip("192.168.1.37")]),
        "192.168.1.37",
    )
    .unwrap();
    for (names, server_name) in [
        (LeafNames::new(["example.com"], []), "example.com"),
        (LeafNames::new(["box.local"], [ip("8.8.8.8")]), "8.8.8.8"),
        (
            LeafNames::new(["box.local"], [ip("192.168.99.5")]),
            "192.168.99.5",
        ),
        (LeafNames::new(["box.local"], [ip("fd00::1")]), "fd00::1"),
    ] {
        let error = handshake(&names, server_name).unwrap_err();
        assert!(
            matches!(error, rustls::Error::InvalidCertificate(_)),
            "{server_name}: {error:?}"
        );
    }
    // A dNSName constraint also permits the names below it (RFC 5280 §4.2.1.10), which is why the
    // scope holds no single-label name; `issue_leaf` refuses such names itself.
    handshake(&LeafNames::new(["x.box.local"], []), "x.box.local").unwrap();
}

/// A node that serves no local subnet, only public or loopback addresses, gets a CA that may
/// issue for no address at all: both address families are excluded whole. Without that, a CA
/// with no IP subtree would be unconstrained for addresses.
#[test]
fn a_ca_with_no_served_subnet_issues_for_no_address() {
    let dir = Dir::new("no-subnet");
    let (scope, left_out) = CaScope::new(["box.local"], &[ip("203.0.113.9")], &[]).unwrap();
    assert!(scope.subnets().is_empty());
    assert_eq!(left_out.addresses, [ip("203.0.113.9")]);
    let (ca, _) = LocalCa::open(&dir.path(), &scope, &node_key(1), SystemTime::now()).unwrap();
    let client = client_config(ca.certificate_der());
    let handshake = |names: &LeafNames, server_name: &str| {
        let leaf = ca.sign_leaf(names, SystemTime::now()).unwrap();
        let server = server_config(Arc::new(LeafResolver::new(&leaf)));
        connect(&client, &server, server_name).map(|_| ())
    };
    handshake(&LeafNames::new(["box.local"], []), "box.local").unwrap();
    for address in ["203.0.113.9", "192.168.1.5", "fd00::1"] {
        let names = LeafNames::new(["box.local"], [ip(address)]);
        assert!(
            matches!(
                handshake(&names, address),
                Err(rustls::Error::InvalidCertificate(_))
            ),
            "{address}"
        );
    }
}

/// The criterion "leaf rotation under 100 open streams drops none", at the TLS layer: 100
/// connections handshaken with one leaf go on exchanging data after the resolver rotates to the
/// next, and the next handshake gets the new leaf.
#[test]
fn rotating_the_leaf_drops_none_of_100_open_connections() {
    let dir = Dir::new("rotate");
    let (ca, _) = LocalCa::open(&dir.path(), &scope(), &node_key(1), SystemTime::now()).unwrap();
    let names = LeafNames::new(["box.local"], []);
    let first = ca.issue_leaf(&names, SystemTime::now()).unwrap();
    let resolver = Arc::new(LeafResolver::new(&first));
    let server = server_config(Arc::clone(&resolver));
    let client = client_config(ca.certificate_der());
    let mut open: Vec<_> = (0..100)
        .map(|_| connect(&client, &server, "box.local").unwrap())
        .collect();
    for (c, s) in &mut open {
        assert_eq!(served_leaf(c), *first.certificate_der());
        exchange(c, s, b"before the rotation");
    }
    let second = ca.issue_leaf(&names, SystemTime::now()).unwrap();
    assert_ne!(first.certificate_der(), second.certificate_der());
    resolver.rotate(&second);
    for (i, (c, s)) in open.iter_mut().enumerate() {
        exchange(c, s, format!("after the rotation, stream {i}").as_bytes());
        assert_eq!(served_leaf(c), *first.certificate_der());
    }
    // A client with a session from before the rotation may resume it, and a resumed handshake
    // shows no certificate; a client with none does a full handshake and gets the new leaf.
    let fresh = client_config(ca.certificate_der());
    let (c, _) = connect(&fresh, &server, "box.local").unwrap();
    assert_eq!(served_leaf(&c), *second.certificate_der());
}

#[test]
fn a_leaf_is_reissued_on_new_names_or_7_days_before_expiry() {
    let dir = Dir::new("reissue");
    let (ca, _) = LocalCa::open(&dir.path(), &scope(), &node_key(1), now()).unwrap();
    let names = LeafNames::new(["box.local"], [ip("192.168.1.37")]);
    let leaf = ca.issue_leaf(&names, now()).unwrap();
    assert!(!leaf.needs_reissue(&names, now()));
    let day = Duration::from_secs(86_400);
    assert!(!leaf.needs_reissue(&names, now() + 22 * day));
    assert!(leaf.needs_reissue(&names, now() + 23 * day));
    let moved = LeafNames::new(["box.local"], [ip("192.168.1.38")]);
    assert!(leaf.needs_reissue(&moved, now()));
}

#[test]
fn the_ca_key_never_prints() {
    let dir = Dir::new("print");
    let (ca, _) = LocalCa::open(&dir.path(), &scope(), &node_key(1), now()).unwrap();
    let key = read_json(&dir.path().join(CA_FILE))["key"]
        .as_str()
        .unwrap()
        .to_owned();
    let leaf = ca
        .issue_leaf(&LeafNames::new(["box.local"], []), now())
        .unwrap();
    let resolver = LeafResolver::new(&leaf);
    // The private scalar: in the PKCS#8 of a P-256 key, the 32 bytes after the ECPrivateKey's
    // version (02 01 01) and its OCTET STRING header (04 20).
    let at = key.find("0201010420").unwrap() + 10;
    let probe = &key[at..at + 64];
    for printed in [
        format!("{ca:?}"),
        format!("{leaf:?}"),
        format!("{resolver:?}"),
        format!("{:?}", ca.binding()),
    ] {
        assert!(!printed.contains(probe), "{printed}");
    }
}

/// Review probe N2: a `ca.json` widened to 0644 is narrowed to 0600 again when it is loaded.
#[cfg(unix)]
#[test]
fn a_widened_ca_file_is_narrowed_on_load() {
    use std::os::unix::fs::PermissionsExt;
    let dir = Dir::new("widened");
    LocalCa::open(&dir.path(), &scope(), &node_key(1), now()).unwrap();
    let path = dir.path().join(CA_FILE);
    fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
    let (_, opened) = LocalCa::open(&dir.path(), &scope(), &node_key(1), now()).unwrap();
    assert_eq!(opened, CaOpen::Loaded);
    assert_eq!(
        fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
}

/// Review probe N1 through `open`: a `tls` that is a link to a 0755 directory elsewhere is
/// refused, and the directory it points at is neither narrowed nor written into.
#[cfg(unix)]
#[test]
fn a_tls_directory_that_is_a_link_is_refused() {
    use std::os::unix::fs::{PermissionsExt, symlink};
    let dir = Dir::new("link");
    let elsewhere = dir.0.join("elsewhere");
    fs::create_dir_all(&elsewhere).unwrap();
    fs::set_permissions(&elsewhere, fs::Permissions::from_mode(0o755)).unwrap();
    symlink(&elsewhere, dir.path()).unwrap();
    let result = LocalCa::open(&dir.path(), &scope(), &node_key(1), now());
    assert!(matches!(result, Err(TlsError::Io { .. })), "{result:?}");
    let mode = fs::metadata(&elsewhere).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o755);
    assert_eq!(fs::read_dir(&elsewhere).unwrap().count(), 0);
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
