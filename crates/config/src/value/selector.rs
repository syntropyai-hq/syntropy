// SPDX-License-Identifier: MIT OR Apache-2.0

//! Node, device, tier, layer and model selectors (RFC-0054 §3).
//!
//! Selectors are parsed and printed here. Resolving one against cluster membership, the
//! resource graph or the model registry happens where those live; this module only gives each
//! selector its syntax and the checks that need no other state. The node, device and tier
//! selectors and the layer range are `syntropy-core`'s types ([`Form`]); the model selector,
//! which no policy object holds, is this crate's.

use std::fmt;
use std::str::FromStr;

use syntropy_core::policy::{DeviceApi, DeviceSelector, LayerRange, NodeSelector, TierSelector};
use syntropy_core::resource::device::DeviceKind;
use syntropy_core::resource::tier::TierKind;
use syntropy_core::{LayerIx, ModelId};

use super::form::{Form, string_form};
use super::{ValueError, serde_via_str};
use crate::schema::Scope;

/// Hex digits a node or model ID prefix must have at least, so a prefix is unlikely to match
/// two IDs by chance. RFC-0054 §3 sets 8 for nodes; models use the same floor.
pub const MIN_ID_PREFIX: usize = 8;
/// Hex digits in a full 32-byte ID.
const MAX_ID_PREFIX: usize = 64;

/// Checks a node name: ASCII letters, digits, `-`, `_` and `.`, not starting with `-` or `.`,
/// and not one of the reserved words.
fn check_node_name(name: &str, input: &str) -> Result<(), ValueError> {
    let ok = !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        && !name.starts_with(['-', '.'])
        && name != "self";
    if ok {
        Ok(())
    } else {
        Err(ValueError::bad_selector(format!(
            "`{input}`: `{name}` is not a node name; node names use letters, digits, `-`, `_` \
             and `.`"
        )))
    }
}

/// Lower-cases and checks a hex ID prefix.
fn parse_id_prefix(hex: &str, input: &str) -> Result<String, ValueError> {
    let ok = (MIN_ID_PREFIX..=MAX_ID_PREFIX).contains(&hex.len())
        && hex.bytes().all(|b| b.is_ascii_hexdigit());
    if ok {
        Ok(hex.to_ascii_lowercase())
    } else {
        Err(ValueError::bad_selector(format!(
            "`{input}`: an ID prefix is {MIN_ID_PREFIX} to {MAX_ID_PREFIX} hex digits"
        )))
    }
}

/// Rejects `"self"` in a cluster-, instance- or request-scope value: in a `policy.*` table or
/// any other value every ingress node reads, it would mean a different node on each
/// (RFC-0054 §3).
pub fn check_node_scope(selector: &NodeSelector, scope: Scope) -> Result<(), ValueError> {
    if *selector == NodeSelector::This && scope != Scope::Node {
        return Err(ValueError::bad_selector(format!(
            "`self` is allowed only in node-scope values, not in a {scope}-scope value, where it \
             would mean a different node on each reader"
        )));
    }
    Ok(())
}

/// A node selector's forms: `"pc"` (a node name), `"id:3f9a…"` (a node ID prefix, at least 8
/// hex digits), `"label:gpu=rtx4090"`, `"self"` or `"*"`. `"self"` is checked against the
/// key's scope by [`check_node_scope`].
impl Form for NodeSelector {
    string_form!(
        NodeSelector,
        "a node selector: a node name, \"id:<hex>\", \"label:<key>=<value>\", \"self\" or \"*\""
    );

    fn parse_text(s: &str) -> Result<Self, ValueError> {
        if s == "*" {
            return Ok(NodeSelector::Any);
        }
        if s == "self" {
            return Ok(NodeSelector::This);
        }
        if let Some(hex) = s.strip_prefix("id:") {
            return parse_id_prefix(hex, s).map(NodeSelector::IdPrefix);
        }
        if let Some(label) = s.strip_prefix("label:") {
            let (key, value) = label.split_once('=').ok_or_else(|| {
                ValueError::bad_selector(format!("`{s}`: write a label as `label:<key>=<value>`"))
            })?;
            let key_ok =
                !key.is_empty() && !key.contains(|c: char| c.is_whitespace() || c.is_control());
            if !key_ok || value.contains(char::is_control) {
                return Err(ValueError::bad_selector(format!(
                    "`{s}`: a label key is non-empty with no spaces, and a value has no control \
                     characters"
                )));
            }
            return Ok(NodeSelector::Label {
                key: key.to_owned(),
                value: value.to_owned(),
            });
        }
        check_node_name(s, s)?;
        Ok(NodeSelector::Name(s.to_owned()))
    }

    fn fmt_text(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeSelector::Any => f.write_str("*"),
            NodeSelector::This => f.write_str("self"),
            NodeSelector::Name(name) => f.write_str(name),
            NodeSelector::IdPrefix(hex) => write!(f, "id:{hex}"),
            NodeSelector::Label { key, value } => write!(f, "label:{key}={value}"),
            // A form `core` added after this list; it prints, but does not parse back.
            other => write!(f, "{other:?}"),
        }
    }
}

/// GPU APIs by selector name.
const DEVICE_APIS: [(DeviceApi, &str); 4] = [
    (DeviceApi::Cuda, "cuda"),
    (DeviceApi::Hip, "hip"),
    (DeviceApi::Metal, "metal"),
    (DeviceApi::Vulkan, "vulkan"),
];

/// Device kinds by selector name (RFC-0001 §5.2's `DeviceKind`).
const DEVICE_KINDS: [(DeviceKind, &str); 4] = [
    (DeviceKind::DiscreteGpu, "dgpu"),
    (DeviceKind::IntegratedGpu, "igpu"),
    (DeviceKind::Cpu, "cpu"),
    (DeviceKind::Npu, "npu"),
];

/// A device selector's forms: `"cuda:0"`, `"metal:0"`, `"cpu:numa1"`, `"<node>/cuda:1"`,
/// `"kind:dgpu"` or `"*"`. Ordinals count each API's devices in RFC-0001 §5.1 probe order.
/// Device kinds are `dgpu`, `igpu`, `cpu` and `npu`.
impl Form for DeviceSelector {
    string_form!(
        DeviceSelector,
        "a device selector such as \"cuda:0\", \"cpu:numa1\", \"<node>/cuda:1\" or \"kind:dgpu\""
    );

    fn parse_text(s: &str) -> Result<Self, ValueError> {
        let bad = || {
            ValueError::bad_selector(format!(
                "`{s}` is not a device selector; write \"cuda:0\", \"metal:0\", \"cpu:numa1\", \
                 \"<node>/cuda:1\", \"kind:dgpu\" or \"*\""
            ))
        };
        if s == "*" {
            return Ok(DeviceSelector::Any);
        }
        if let Some(kind) = s.strip_prefix("kind:") {
            return DEVICE_KINDS
                .iter()
                .find(|(_, name)| *name == kind)
                .map(|(k, _)| DeviceSelector::Kind(*k))
                .ok_or_else(bad);
        }
        let (node, device) = match s.split_once('/') {
            Some((node, device)) => {
                check_node_name(node, s)?;
                (Some(node.to_owned()), device)
            }
            None => (None, s),
        };
        let (api, ordinal) = device.split_once(':').ok_or_else(bad)?;
        let number = |digits: &str| -> Result<u16, ValueError> {
            if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                return Err(bad());
            }
            digits.parse().map_err(|_| bad())
        };
        if api == "cpu" {
            let numa = ordinal.strip_prefix("numa").ok_or_else(bad)?;
            return Ok(DeviceSelector::Cpu {
                node,
                numa: number(numa)?,
            });
        }
        let api = DEVICE_APIS
            .iter()
            .find(|(_, name)| *name == api)
            .map(|(api, _)| *api)
            .ok_or_else(bad)?;
        Ok(DeviceSelector::Gpu {
            node,
            api,
            index: number(ordinal)?,
        })
    }

    fn fmt_text(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let node_prefix = |f: &mut fmt::Formatter<'_>, node: &Option<String>| match node {
            Some(node) => write!(f, "{node}/"),
            None => Ok(()),
        };
        match self {
            DeviceSelector::Any => f.write_str("*"),
            DeviceSelector::Kind(kind) => match DEVICE_KINDS.iter().find(|(k, _)| k == kind) {
                Some((_, name)) => write!(f, "kind:{name}"),
                // A kind `core` added after this table; it prints, but does not parse back.
                None => write!(f, "kind:{kind:?}"),
            },
            DeviceSelector::Gpu { node, api, index } => {
                node_prefix(f, node)?;
                match DEVICE_APIS.iter().find(|(a, _)| a == api) {
                    Some((_, name)) => write!(f, "{name}:{index}"),
                    // An API `core` added after this table; it prints, but does not parse back.
                    None => write!(f, "{api:?}:{index}"),
                }
            }
            DeviceSelector::Cpu { node, numa } => {
                node_prefix(f, node)?;
                write!(f, "cpu:numa{numa}")
            }
            // A form `core` added after this list; it prints, but does not parse back.
            other => write!(f, "{other:?}"),
        }
    }
}

/// Tier kinds by selector name. `ram` is `HostPinned` (RFC-0054 §3).
const TIER_KINDS: [(TierKind, &str); 5] = [
    (TierKind::Vram, "vram"),
    (TierKind::Unified, "unified"),
    (TierKind::HostPinned, "ram"),
    (TierKind::HostPageable, "ram-pageable"),
    (TierKind::Nvme, "nvme"),
];

/// A tier selector's forms: `"vram"`, `"unified"`, `"ram"` (= `HostPinned`), `"ram-pageable"`,
/// `"nvme"`, optionally on a node and numbered, as in `"<node>/nvme:1"`.
impl Form for TierSelector {
    string_form!(
        TierSelector,
        "a tier selector such as \"vram\", \"ram\", \"nvme\" or \"<node>/nvme:1\""
    );

    fn parse_text(s: &str) -> Result<Self, ValueError> {
        let bad = || {
            ValueError::bad_selector(format!(
                "`{s}` is not a tier selector; write \"vram\", \"unified\", \"ram\", \
                 \"ram-pageable\" or \"nvme\", optionally as \"<node>/nvme:1\""
            ))
        };
        let (node, tier) = match s.split_once('/') {
            Some((node, tier)) => {
                check_node_name(node, s)?;
                (Some(node.to_owned()), tier)
            }
            None => (None, s),
        };
        let (kind, index) = match tier.split_once(':') {
            Some((kind, digits)) => {
                if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                    return Err(bad());
                }
                (kind, Some(digits.parse().map_err(|_| bad())?))
            }
            None => (tier, None),
        };
        let kind = TIER_KINDS
            .iter()
            .find(|(_, name)| *name == kind)
            .map(|(k, _)| *k)
            .ok_or_else(bad)?;
        Ok(TierSelector { node, kind, index })
    }

    fn fmt_text(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(node) = &self.node {
            write!(f, "{node}/")?;
        }
        match TIER_KINDS.iter().find(|(k, _)| *k == self.kind) {
            Some((_, name)) => f.write_str(name)?,
            // A kind `core` added after this table; it prints, but does not parse back.
            None => write!(f, "{:?}", self.kind)?,
        }
        if let Some(index) = self.index {
            write!(f, ":{index}")?;
        }
        Ok(())
    }
}

/// A layer range's forms: `"0..20"` (half-open, as in RFC-0001), `"40.."`, `"..8"` or a single
/// layer, `"7"`. `".."` is every layer.
impl Form for LayerRange {
    string_form!(
        LayerRange,
        "a layer range such as \"0..20\", \"40..\", \"..8\" or \"7\""
    );

    fn parse_text(s: &str) -> Result<Self, ValueError> {
        let bad = |why: &str| {
            ValueError::bad_selector(format!(
                "`{s}` is not a layer range: {why}; write \"0..20\" (half-open), \"40..\", \
                 \"..8\" or \"7\""
            ))
        };
        let bound = |digits: &str| -> Result<Option<LayerIx>, ValueError> {
            if digits.is_empty() {
                return Ok(None);
            }
            if !digits.bytes().all(|b| b.is_ascii_digit()) {
                return Err(bad("a bound is not a whole number"));
            }
            digits
                .parse()
                .map(|layer| Some(LayerIx(layer)))
                .map_err(|_| bad("a bound is above 65535"))
        };
        let range = match s.split_once("..") {
            Some((start, end)) => LayerRange {
                start: bound(start)?,
                end: bound(end)?,
            },
            None => {
                let layer = bound(s)?.ok_or_else(|| bad("it is empty"))?;
                let end = layer
                    .0
                    .checked_add(1)
                    .ok_or_else(|| bad("the layer is above 65534"))?;
                LayerRange {
                    start: Some(layer),
                    end: Some(LayerIx(end)),
                }
            }
        };
        if let (Some(start), Some(end)) = (range.start, range.end)
            && start >= end
        {
            return Err(bad("it is empty"));
        }
        Ok(range)
    }

    fn fmt_text(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.start, self.end) {
            (Some(start), Some(end)) if end.0.checked_sub(start.0) == Some(1) => {
                write!(f, "{}", start.0)
            }
            (start, end) => {
                if let Some(start) = start {
                    write!(f, "{}", start.0)?;
                }
                f.write_str("..")?;
                if let Some(end) = end {
                    write!(f, "{}", end.0)?;
                }
                Ok(())
            }
        }
    }
}

/// Selects models: an exact name, a glob (`*` and `?`), `"id:<blake3 hex prefix>"` or
/// `"default"` (RFC-0054 §3), matched against resolved model names and IDs (RFC-0050 §4).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ModelSelector {
    /// `[policy.default]`: every model.
    Default,
    /// Models whose ID, in hex, starts with these digits, lower case.
    IdPrefix(String),
    /// Names matching a glob, where `*` is any run of characters and `?` is one.
    Glob(String),
    /// One exact name.
    Name(String),
}

impl ModelSelector {
    /// Whether a model with this name and ID matches.
    pub fn matches(&self, name: &str, id: &ModelId) -> bool {
        match self {
            ModelSelector::Default => true,
            ModelSelector::Name(exact) => exact == name,
            ModelSelector::Glob(glob) => glob_matches(glob, name),
            ModelSelector::IdPrefix(hex) => {
                let digits: String = id.0.0.iter().map(|b| format!("{b:02x}")).collect();
                digits.starts_with(hex.as_str())
            }
        }
    }

    /// For a glob, the text before its first wildcard. RFC-0054 §2.2 Rule 3 prefers the glob
    /// with the longest literal prefix.
    pub fn literal_prefix(&self) -> Option<&str> {
        match self {
            ModelSelector::Glob(glob) => glob.split(['*', '?']).next(),
            _ => None,
        }
    }
}

/// Glob matching with `*` (any run) and `?` (one character), in linear space.
fn glob_matches(glob: &str, name: &str) -> bool {
    let glob: Vec<char> = glob.chars().collect();
    let name: Vec<char> = name.chars().collect();
    let (mut g, mut n) = (0, 0);
    // The position after the last `*`, and the name position it was tried at.
    let mut star: Option<(usize, usize)> = None;
    while n < name.len() {
        if g < glob.len() && (glob[g] == '?' || glob[g] == name[n]) {
            g += 1;
            n += 1;
        } else if g < glob.len() && glob[g] == '*' {
            star = Some((g + 1, n));
            g += 1;
        } else if let Some((after, tried)) = star {
            g = after;
            n = tried + 1;
            star = Some((after, tried + 1));
        } else {
            return false;
        }
    }
    glob[g..].iter().all(|&c| c == '*')
}

impl fmt::Display for ModelSelector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelSelector::Default => f.write_str("default"),
            ModelSelector::IdPrefix(hex) => write!(f, "id:{hex}"),
            ModelSelector::Glob(text) | ModelSelector::Name(text) => f.write_str(text),
        }
    }
}

impl FromStr for ModelSelector {
    type Err = ValueError;

    fn from_str(s: &str) -> Result<Self, ValueError> {
        if s == "default" {
            return Ok(ModelSelector::Default);
        }
        if let Some(hex) = s.strip_prefix("id:") {
            return parse_id_prefix(hex, s).map(ModelSelector::IdPrefix);
        }
        if s.is_empty() || s.contains(|c: char| c.is_whitespace() || c.is_control()) {
            return Err(ValueError::bad_selector(format!(
                "`{s}` is not a model selector: names and globs are non-empty, with no spaces"
            )));
        }
        if s.contains(['*', '?']) {
            Ok(ModelSelector::Glob(s.to_owned()))
        } else {
            Ok(ModelSelector::Name(s.to_owned()))
        }
    }
}

serde_via_str!(
    ModelSelector,
    "a model selector: a name, a glob, \"id:<hex>\" or \"default\""
);

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use syntropy_core::Blake3;

    use super::*;
    use crate::value::form::{Shown, parse};

    /// A selector printed and parsed back, the way `config show` and a file do.
    fn back<T: Form>(value: &T) -> Result<T, ValueError> {
        parse(&Shown(value).to_string())
    }

    /// Reads JSON the way a key with `serde(with = "syntropy_config::form")` does.
    fn json<T: Form>(text: &str) -> Option<T> {
        let mut de = serde_json::Deserializer::from_str(text);
        T::deserialize_form(&mut de).ok()
    }

    /// Writes JSON the way such a key does.
    fn to_json<T: Form>(value: &T) -> String {
        let mut out = Vec::new();
        value
            .serialize_form(&mut serde_json::Serializer::new(&mut out))
            .unwrap();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn node_selectors_parse() {
        assert_eq!(parse("pc"), Ok(NodeSelector::Name("pc".into())));
        assert_eq!(parse("*"), Ok(NodeSelector::Any));
        assert_eq!(parse("self"), Ok(NodeSelector::This));
        assert_eq!(
            parse("id:3F9A0B1C2D"),
            Ok(NodeSelector::IdPrefix("3f9a0b1c2d".into()))
        );
        assert_eq!(
            parse("label:gpu=rtx4090"),
            Ok(NodeSelector::Label {
                key: "gpu".into(),
                value: "rtx4090".into()
            })
        );
        for s in [
            "",
            "id:3f9a",
            "id:xyz12345",
            "label:gpu",
            "label:=x",
            "a b",
            "-pc",
            "a:b",
        ] {
            assert!(parse::<NodeSelector>(s).is_err(), "{s:?} should not parse");
        }
    }

    #[test]
    fn self_is_node_scope_only() {
        assert!(check_node_scope(&NodeSelector::This, Scope::Node).is_ok());
        assert!(check_node_scope(&NodeSelector::This, Scope::Cluster).is_err());
        assert!(check_node_scope(&NodeSelector::This, Scope::Instance).is_err());
        assert!(check_node_scope(&NodeSelector::Any, Scope::Cluster).is_ok());
    }

    #[test]
    fn device_selectors_parse() {
        assert_eq!(
            parse("cuda:0"),
            Ok(DeviceSelector::Gpu {
                node: None,
                api: DeviceApi::Cuda,
                index: 0
            })
        );
        assert_eq!(
            parse("pc/cuda:1"),
            Ok(DeviceSelector::Gpu {
                node: Some("pc".into()),
                api: DeviceApi::Cuda,
                index: 1
            })
        );
        assert_eq!(
            parse("cpu:numa1"),
            Ok(DeviceSelector::Cpu {
                node: None,
                numa: 1
            })
        );
        assert_eq!(
            parse("kind:dgpu"),
            Ok(DeviceSelector::Kind(DeviceKind::DiscreteGpu))
        );
        assert_eq!(parse("*"), Ok(DeviceSelector::Any));
        for s in [
            "",
            "cuda",
            "cuda:",
            "cuda:x",
            "cpu:1",
            "opencl:0",
            "kind:gpu",
            "a b/cuda:0",
        ] {
            assert!(
                parse::<DeviceSelector>(s).is_err(),
                "{s:?} should not parse"
            );
        }
    }

    #[test]
    fn tier_selectors_parse() {
        let kind = |kind| TierSelector {
            node: None,
            kind,
            index: None,
        };
        assert_eq!(parse("ram"), Ok(kind(TierKind::HostPinned)));
        assert_eq!(parse("ram-pageable"), Ok(kind(TierKind::HostPageable)));
        assert_eq!(
            parse("pc/nvme:1"),
            Ok(TierSelector {
                node: Some("pc".into()),
                kind: TierKind::Nvme,
                index: Some(1)
            })
        );
        for s in ["", "disk", "nvme:", "pc/", "/nvme"] {
            assert!(parse::<TierSelector>(s).is_err(), "{s:?} should not parse");
        }
    }

    #[test]
    fn layer_ranges_parse() {
        let range = |start: Option<u16>, end: Option<u16>| LayerRange {
            start: start.map(LayerIx),
            end: end.map(LayerIx),
        };
        assert_eq!(parse("0..20"), Ok(range(Some(0), Some(20))));
        assert_eq!(parse("40.."), Ok(range(Some(40), None)));
        assert_eq!(parse("..8"), Ok(range(None, Some(8))));
        assert_eq!(parse("7"), Ok(range(Some(7), Some(8))));
        assert_eq!(parse(".."), Ok(range(None, None)));
        for s in ["", "5..5", "20..10", "a..b", "1...2", "-1..2", "70000"] {
            assert!(parse::<LayerRange>(s).is_err(), "{s:?} should not parse");
        }
    }

    #[test]
    fn model_selectors_parse_and_match() {
        let id = ModelId(Blake3([0xab; 32]));
        let parse = |s: &str| s.parse::<ModelSelector>().unwrap();
        assert_eq!(parse("default"), ModelSelector::Default);
        assert!(parse("qwen3-235b-a22b*").matches("qwen3-235b-a22b-instruct", &id));
        assert!(!parse("qwen3-235b-a22b*").matches("qwen3-30b", &id));
        assert!(parse("deepseek-v?").matches("deepseek-v3", &id));
        assert!(!parse("deepseek-v?").matches("deepseek-v30", &id));
        assert!(parse("*-instruct").matches("a-b-instruct", &id));
        assert!(parse("exact").matches("exact", &id));
        assert!(!parse("exact").matches("exact2", &id));
        assert!(parse("id:ABABABAB").matches("any", &id));
        assert!(!parse("id:abababac").matches("any", &id));
        assert_eq!(parse("qwen3*").literal_prefix(), Some("qwen3"));
        for s in ["", "a b", "id:ab"] {
            assert!(
                s.parse::<ModelSelector>().is_err(),
                "{s:?} should not parse"
            );
        }
    }

    fn node_name() -> impl Strategy<Value = String> {
        "[a-z0-9][a-z0-9._-]{0,15}".prop_filter("reserved", |s| s != "self")
    }

    fn node_selector() -> impl Strategy<Value = NodeSelector> {
        prop_oneof![
            Just(NodeSelector::Any),
            Just(NodeSelector::This),
            node_name().prop_map(NodeSelector::Name),
            "[0-9a-f]{8,64}".prop_map(NodeSelector::IdPrefix),
            ("[a-z][a-z0-9./_-]{0,10}", "[a-z0-9=._-]{0,10}")
                .prop_map(|(key, value)| NodeSelector::Label { key, value }),
        ]
    }

    fn device_selector() -> impl Strategy<Value = DeviceSelector> {
        let api = prop_oneof![
            Just(DeviceApi::Cuda),
            Just(DeviceApi::Hip),
            Just(DeviceApi::Metal),
            Just(DeviceApi::Vulkan),
        ];
        let kind = proptest::sample::select(DEVICE_KINDS.map(|(k, _)| k).to_vec());
        prop_oneof![
            Just(DeviceSelector::Any),
            kind.prop_map(DeviceSelector::Kind),
            (proptest::option::of(node_name()), api, any::<u16>())
                .prop_map(|(node, api, index)| DeviceSelector::Gpu { node, api, index }),
            (proptest::option::of(node_name()), any::<u16>())
                .prop_map(|(node, numa)| DeviceSelector::Cpu { node, numa }),
        ]
    }

    fn tier_selector() -> impl Strategy<Value = TierSelector> {
        let kind = proptest::sample::select(TIER_KINDS.map(|(k, _)| k).to_vec());
        (
            proptest::option::of(node_name()),
            kind,
            proptest::option::of(any::<u16>()),
        )
            .prop_map(|(node, kind, index)| TierSelector { node, kind, index })
    }

    fn layer_range() -> impl Strategy<Value = LayerRange> {
        (
            proptest::option::of(any::<u16>()),
            proptest::option::of(any::<u16>()),
        )
            .prop_filter("non-empty", |(s, e)| match (s, e) {
                (Some(s), Some(e)) => s < e,
                _ => true,
            })
            .prop_map(|(start, end)| LayerRange {
                start: start.map(LayerIx),
                end: end.map(LayerIx),
            })
    }

    fn model_selector() -> impl Strategy<Value = ModelSelector> {
        prop_oneof![
            Just(ModelSelector::Default),
            "[0-9a-f]{8,64}".prop_map(ModelSelector::IdPrefix),
            "[a-z0-9._/-]{0,6}[*?][a-z0-9._*?-]{0,6}".prop_map(ModelSelector::Glob),
            "[a-z0-9][a-z0-9._/:-]{0,20}"
                .prop_filter("reserved", |s| s != "default" && !s.starts_with("id:"))
                .prop_map(ModelSelector::Name),
        ]
    }

    proptest! {
        #[test]
        fn node_selectors_parse_back(sel in node_selector()) {
            prop_assert_eq!(back(&sel), Ok(sel.clone()));
            prop_assert_eq!(json::<NodeSelector>(&to_json(&sel)), Some(sel));
        }

        #[test]
        fn device_selectors_parse_back(sel in device_selector()) {
            prop_assert_eq!(back(&sel), Ok(sel.clone()));
            prop_assert_eq!(json::<DeviceSelector>(&to_json(&sel)), Some(sel));
        }

        #[test]
        fn tier_selectors_parse_back(sel in tier_selector()) {
            prop_assert_eq!(back(&sel), Ok(sel.clone()));
            prop_assert_eq!(json::<TierSelector>(&to_json(&sel)), Some(sel));
        }

        #[test]
        fn layer_ranges_parse_back(range in layer_range()) {
            prop_assert_eq!(back(&range), Ok(range));
            prop_assert_eq!(json::<LayerRange>(&to_json(&range)), Some(range));
        }

        #[test]
        fn model_selectors_parse_back(sel in model_selector()) {
            prop_assert_eq!(sel.to_string().parse::<ModelSelector>(), Ok(sel.clone()));
        }
    }
}
