// SPDX-License-Identifier: MIT OR Apache-2.0

//! `syntropy-core`'s policy value types read and printed in RFC-0054 §3's forms (PROD-E1-T23;
//! OQ-PRD-27 (a)).
//!
//! The acceptance criteria this file checks:
//! - every text form in §3's table for `Fraction`, `Reserve` and the node, device, tier and
//!   layer selectors round-trips, text to `core` value to text (golden table);
//! - a value `config` rejected before is rejected with the same `Config` code (golden table,
//!   taken from `config`'s own types before they were deleted);
//! - `command::show_resolved` prints every key of those kinds as before (golden text, taken the
//!   same way).

use std::fmt::Debug;

use serde::Deserialize;
use syntropy_config::form::{Form, Shown, parse};
use syntropy_config::{
    Code, ConfigFile, ConfigSection, Env, Flags, Layers, Os, Registry, ResolvedConfig, command,
};
use syntropy_core::policy::{
    DeviceSelector, Fraction, LayerRange, NodeSelector, Reserve, TierSelector,
};

/// A key of each kind, and an optional and a list one.
#[derive(Debug, Deserialize, ConfigSection)]
#[config(section = "f", scope = Node, owner = "RFC-0054")]
struct F {
    /// A share.
    #[config(reload = Hot, default = 0.5)]
    #[serde(with = "syntropy_config::form")]
    share: Fraction,
    /// Memory kept back.
    #[config(reload = Hot, default = "1GiB")]
    #[serde(with = "syntropy_config::form")]
    reserve: Reserve,
    /// A node.
    #[config(reload = Hot, default = "*")]
    #[serde(with = "syntropy_config::form")]
    node: NodeSelector,
    /// A device.
    #[config(reload = Hot, default = "*")]
    #[serde(with = "syntropy_config::form")]
    device: DeviceSelector,
    /// A tier.
    #[config(reload = Hot, default = "vram")]
    #[serde(with = "syntropy_config::form")]
    tier: TierSelector,
    /// Some layers.
    #[config(reload = Hot, default = "..")]
    #[serde(with = "syntropy_config::form")]
    layers: LayerRange,
    /// An optional share.
    #[config(reload = Hot, default = unset)]
    #[serde(with = "syntropy_config::form", default)]
    cap: Option<Fraction>,
    /// Some nodes.
    #[config(reload = Hot, default = [])]
    #[serde(with = "syntropy_config::form")]
    nodes: Vec<NodeSelector>,
}

static REGISTRY: Registry = Registry::new(&[&F::SCHEMA]);

fn resolve(text: &str) -> Result<ResolvedConfig, syntropy_config::Rejected> {
    let files = [ConfigFile::new("/etc/syntropy/config.toml", text)];
    let env = Env::from_pairs::<&str, &str>(Os::Linux, []);
    ResolvedConfig::resolve(
        &REGISTRY,
        Layers {
            os: Os::Linux,
            files: &files,
            env: &env,
            flags: &Flags::new(),
        },
    )
}

/// `key = toml` read into section `F`.
fn read(key: &str, toml: &str) -> F {
    resolve(&format!("[f]\n{key} = {toml}\n"))
        .unwrap_or_else(|r| panic!("{key} = {toml}: {:?}", r.errors))
        .section::<F>()
        .unwrap_or_else(|e| panic!("{key} = {toml}: {e}"))
}

/// The value's `Debug` form and canonical text, checking that the text parses back to it.
fn shown<T: Form + Debug + PartialEq>(value: &T) -> (String, String) {
    let text = Shown(value).to_string();
    let back: T = parse(&text).unwrap_or_else(|e| panic!("{text}: {e}"));
    assert_eq!(&back, value, "{text} does not parse back");
    (format!("{value:?}"), text)
}

/// Each text form in RFC-0054 §3's table for the six types: the key, the TOML as written, the
/// `core` value it reads as, and the canonical text `config` prints for it.
const FORMS: &[(&str, &str, &str, &str)] = &[
    ("share", "0.2", "Fraction(0.2)", "0.2"),
    ("share", "\"20%\"", "Fraction(0.2)", "0.2"),
    ("share", "\"12.5%\"", "Fraction(0.125)", "0.125"),
    ("share", "1", "Fraction(1.0)", "1"),
    (
        "reserve",
        "\"8GiB\"",
        "Reserve { min: Bytes(8589934592), fraction: 0.0, max: None }",
        "8GiB",
    ),
    (
        "reserve",
        "\"10%\"",
        "Reserve { min: Bytes(0), fraction: 0.1, max: None }",
        "10%",
    ),
    (
        "reserve",
        "{ min = \"6GiB\", fraction = 0.10 }",
        "Reserve { min: Bytes(6442450944), fraction: 0.1, max: None }",
        "{ min = \"6GiB\", fraction = 0.1 }",
    ),
    (
        "reserve",
        "{ min = \"6GiB\", fraction = 0.10, max = \"32GiB\" }",
        "Reserve { min: Bytes(6442450944), fraction: 0.1, max: Some(Bytes(34359738368)) }",
        "{ min = \"6GiB\", fraction = 0.1, max = \"32GiB\" }",
    ),
    ("node", "\"pc\"", "Name(\"pc\")", "pc"),
    (
        "node",
        "\"id:3F9A0B1C\"",
        "IdPrefix(\"3f9a0b1c\")",
        "id:3f9a0b1c",
    ),
    (
        "node",
        "\"label:gpu=rtx4090\"",
        "Label { key: \"gpu\", value: \"rtx4090\" }",
        "label:gpu=rtx4090",
    ),
    ("node", "\"self\"", "This", "self"),
    ("node", "\"*\"", "Any", "*"),
    (
        "device",
        "\"cuda:0\"",
        "Gpu { node: None, api: Cuda, index: 0 }",
        "cuda:0",
    ),
    (
        "device",
        "\"metal:0\"",
        "Gpu { node: None, api: Metal, index: 0 }",
        "metal:0",
    ),
    (
        "device",
        "\"cpu:numa1\"",
        "Cpu { node: None, numa: 1 }",
        "cpu:numa1",
    ),
    (
        "device",
        "\"pc/cuda:1\"",
        "Gpu { node: Some(\"pc\"), api: Cuda, index: 1 }",
        "pc/cuda:1",
    ),
    ("device", "\"kind:dgpu\"", "Kind(DiscreteGpu)", "kind:dgpu"),
    (
        "tier",
        "\"vram\"",
        "TierSelector { node: None, kind: Vram, index: None }",
        "vram",
    ),
    (
        "tier",
        "\"unified\"",
        "TierSelector { node: None, kind: Unified, index: None }",
        "unified",
    ),
    (
        "tier",
        "\"ram\"",
        "TierSelector { node: None, kind: HostPinned, index: None }",
        "ram",
    ),
    (
        "tier",
        "\"ram-pageable\"",
        "TierSelector { node: None, kind: HostPageable, index: None }",
        "ram-pageable",
    ),
    (
        "tier",
        "\"nvme\"",
        "TierSelector { node: None, kind: Nvme, index: None }",
        "nvme",
    ),
    (
        "tier",
        "\"pc/nvme:1\"",
        "TierSelector { node: Some(\"pc\"), kind: Nvme, index: Some(1) }",
        "pc/nvme:1",
    ),
    (
        "layers",
        "\"0..20\"",
        "LayerRange { start: Some(LayerIx(0)), end: Some(LayerIx(20)) }",
        "0..20",
    ),
    (
        "layers",
        "\"40..\"",
        "LayerRange { start: Some(LayerIx(40)), end: None }",
        "40..",
    ),
    (
        "layers",
        "\"..8\"",
        "LayerRange { start: None, end: Some(LayerIx(8)) }",
        "..8",
    ),
    (
        "layers",
        "\"7\"",
        "LayerRange { start: Some(LayerIx(7)), end: Some(LayerIx(8)) }",
        "7",
    ),
];

#[test]
fn every_rfc_form_reads_into_the_core_type_and_prints_back() {
    for &(key, toml, value, text) in FORMS {
        let section = read(key, toml);
        let got = match key {
            "share" => shown(&section.share),
            "reserve" => shown(&section.reserve),
            "node" => shown(&section.node),
            "device" => shown(&section.device),
            "tier" => shown(&section.tier),
            "layers" => shown(&section.layers),
            _ => unreachable!(),
        };
        assert_eq!(got, (value.to_owned(), text.to_owned()), "{key} = {toml}");
        // The printed text, written back as the key's value, reads as the same value. A
        // fraction prints as a number and a reserve's table form as a table; everything else
        // is a string.
        let written = if key == "share" || text.starts_with('{') {
            text.to_owned()
        } else {
            format!("\"{text}\"")
        };
        let again = read(key, &written);
        assert_eq!(
            format!("{:?}", again),
            format!("{:?}", section),
            "{key} = {written}"
        );
    }
}

#[test]
fn an_optional_and_a_list_key_read_the_same_forms() {
    let section = read("cap", "\"20%\"");
    assert_eq!(section.cap, Fraction::new(0.2));
    let section = read("nodes", "[\"pc\", \"label:gpu=rtx4090\", \"*\"]");
    assert_eq!(
        section.nodes,
        vec![
            NodeSelector::Name("pc".into()),
            NodeSelector::Label {
                key: "gpu".into(),
                value: "rtx4090".into()
            },
            NodeSelector::Any,
        ]
    );
    let defaults = resolve("").unwrap().section::<F>().unwrap();
    assert_eq!(defaults.cap, None);
    assert!(defaults.nodes.is_empty());
    assert!(F::check_defaults().is_empty());
}

/// Values `config` rejected before PROD-E1-T23, with the code it rejected each with.
const REJECTED: &[(&str, &str, Code)] = &[
    ("share", "1.5", Code::OutOfRange),
    ("share", "\"120%\"", Code::OutOfRange),
    ("share", "\"20 %%\"", Code::BadUnit),
    ("share", "\"a fifth\"", Code::BadUnit),
    ("share", "true", Code::WrongType),
    ("reserve", "\"8\"", Code::BadUnit),
    ("reserve", "\"110%\"", Code::OutOfRange),
    (
        "reserve",
        "{ min = \"32GiB\", max = \"6GiB\" }",
        Code::OutOfRange,
    ),
    ("reserve", "{ mni = \"6GiB\" }", Code::WrongType),
    ("node", "\"id:12\"", Code::BadSelector),
    ("node", "\"-pc\"", Code::BadSelector),
    ("node", "\"label:gpu\"", Code::BadSelector),
    ("node", "3", Code::WrongType),
    ("device", "\"opencl:0\"", Code::BadSelector),
    ("device", "\"cpu:1\"", Code::BadSelector),
    ("tier", "\"disk\"", Code::BadSelector),
    ("tier", "\"pc/\"", Code::BadSelector),
    ("layers", "\"5..5\"", Code::BadSelector),
    ("layers", "\"a..b\"", Code::BadSelector),
    ("cap", "\"200%\"", Code::OutOfRange),
    ("nodes", "[\"pc\", \"id:1\"]", Code::BadSelector),
];

#[test]
fn a_rejected_value_keeps_its_code() {
    for &(key, toml, code) in REJECTED {
        let rejected = resolve(&format!("[f]\n{key} = {toml}\n"))
            .err()
            .unwrap_or_else(|| panic!("{key} = {toml} was accepted"));
        assert_eq!(
            rejected.errors[0].code,
            Some(code),
            "{key} = {toml}: {:?}",
            rejected.errors
        );
    }
}

/// `config show --resolved` for a file that sets every key, as `config`'s own types printed
/// it before PROD-E1-T23: a value prints as it was written.
const SHOWN: &str = concat!(
    "# node keys: default < config file < environment < flags (1 file read)\n",
    "f.cap = 0.25                                              # /etc/syntropy/config.toml:8\n",
    "f.device = \"pc/cuda:1\"                                    # /etc/syntropy/config.toml:5\n",
    "f.layers = \"40..\"                                         # /etc/syntropy/config.toml:7\n",
    "f.node = \"label:gpu=rtx4090\"                              # /etc/syntropy/config.toml:4\n",
    "f.nodes = [\"pc\", \"self\"]                                  # /etc/syntropy/config.toml:9\n",
    "f.reserve = { fraction = 0.1, max = \"32GiB\", min = \"6GiB\" }  # /etc/syntropy/config.toml:3\n",
    "f.share = \"20%\"                                           # /etc/syntropy/config.toml:2\n",
    "f.tier = \"ram\"                                            # /etc/syntropy/config.toml:6\n",
);

#[test]
fn show_resolved_prints_these_keys_as_before() {
    let file = "[f]\n\
                share = \"20%\"\n\
                reserve = { min = \"6GiB\", fraction = 0.10, max = \"32GiB\" }\n\
                node = \"label:gpu=rtx4090\"\n\
                device = \"pc/cuda:1\"\n\
                tier = \"ram\"\n\
                layers = \"40..\"\n\
                cap = 0.25\n\
                nodes = [\"pc\", \"self\"]\n";
    let config = resolve(file).unwrap();
    assert_eq!(command::show_resolved(&config), SHOWN);
}
