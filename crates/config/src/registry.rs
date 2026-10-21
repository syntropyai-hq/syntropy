// SPDX-License-Identifier: MIT OR Apache-2.0

//! The section registry: every section a binary loads, with no prefix claimed twice
//! (RFC-0054 §5.1).
//!
//! Each section *claims* what it owns:
//! - a section with a prefix owns its subtree, `prefix.*`;
//! - a shared section (`#[config(shared)]`) and a top-level section own only their own keys,
//!   each by its full path, so RFC-0030 can own `net.relay_mode` inside RFC-0031's `net.*`.
//!
//! [`Registry::new`] is a `const fn` that panics when two sections make the same claim. Build
//! the registry in a `static` or `const` item, and the panic becomes a compile error, so two
//! sections with one prefix fail the build (RFC-0054 §5.1, §10.3):
//!
//! ```
//! use serde::Deserialize;
//! use syntropy_config::{ConfigSection, Registry};
//!
//! /// Probe settings.
//! #[derive(Deserialize, ConfigSection)]
//! #[config(section = "probe", scope = Node, owner = "RFC-0010")]
//! struct ProbeSection {
//!     /// How long the probe waits for a model.
//!     #[config(reload = Node, default = "7s")]
//!     model_wait: syntropy_config::Duration,
//! }
//!
//! static REGISTRY: Registry = Registry::new(&[&ProbeSection::SCHEMA]);
//! assert!(REGISTRY.key("probe.model_wait").is_some());
//! ```
//!
//! Key-level checks that would be slow to run at compile time over a full registry, such as two
//! sections declaring one full key path, are in [`Registry::check`], which a test of the crate
//! that builds the registry runs.

use std::fmt;

use crate::schema::{KeyError, KeySchema, SectionSchema};

/// Every section a binary loads. Construct it with [`Registry::new`] in a `static` or `const`
/// item.
#[derive(Clone, Copy, Debug)]
pub struct Registry {
    sections: &'static [&'static SectionSchema],
}

impl Registry {
    /// A registry of `sections`.
    ///
    /// # Panics
    ///
    /// If two sections claim one prefix or one key, or a section's prefix or key is malformed.
    /// In a `static` or `const` initializer that panic is a compile error.
    pub const fn new(sections: &'static [&'static SectionSchema]) -> Self {
        let mut i = 0;
        while i < sections.len() {
            let mut j = i + 1;
            while j < sections.len() {
                check_pair(sections[i], sections[j]);
                j += 1;
            }
            i += 1;
        }
        Self { sections }
    }

    /// The sections, in registration order.
    pub const fn sections(&self) -> &'static [&'static SectionSchema] {
        self.sections
    }

    /// Every key of every section, in registration order.
    pub fn keys(&self) -> impl Iterator<Item = &'static KeySchema> + use<> {
        self.sections.iter().flat_map(|s| s.keys.iter())
    }

    /// The key with this full path.
    pub fn key(&self, path: &str) -> Option<&'static KeySchema> {
        self.keys().find(|k| k.path == path)
    }

    /// The section a key belongs to.
    pub fn section_of(&self, path: &str) -> Option<&'static SectionSchema> {
        self.sections
            .iter()
            .copied()
            .find(|s| s.keys.iter().any(|k| k.path == path))
    }

    /// Checks the registry's keys as a whole: no full path is declared twice, and no key is
    /// both a value and a table that other keys live in (TOML cannot hold both). Returns every
    /// problem found.
    pub fn check(&self) -> Vec<RegistryError> {
        let mut errors = Vec::new();
        let keys: Vec<(&SectionSchema, &KeySchema)> = self
            .sections
            .iter()
            .flat_map(|s| s.keys.iter().map(move |k| (*s, k)))
            .collect();
        for (i, (section_a, a)) in keys.iter().enumerate() {
            for (section_b, b) in &keys[i + 1..] {
                if a.path == b.path {
                    errors.push(RegistryError::DuplicateKey {
                        key: a.path,
                        first: section_a.type_name,
                        second: section_b.type_name,
                    });
                } else if let Some((value, table)) = value_and_table(a.path, b.path) {
                    errors.push(RegistryError::ValueIsTable { value, table });
                }
            }
            for section in self.sections {
                if a.path == section.prefix {
                    errors.push(RegistryError::ValueIsTable {
                        value: a.path,
                        table: section.prefix,
                    });
                }
            }
        }
        errors
    }

    /// Runs every section's default check ([`crate::ConfigSection::check_defaults`]).
    pub fn check_defaults(&self) -> Vec<KeyError> {
        self.sections
            .iter()
            .flat_map(|s| (s.check_defaults)())
            .collect()
    }
}

/// If one path is a strict prefix of the other, by whole segments, the shorter is a value that
/// the longer treats as a table.
fn value_and_table(a: &'static str, b: &'static str) -> Option<(&'static str, &'static str)> {
    let under = |short: &str, long: &str| {
        long.len() > short.len() && long.starts_with(short) && long.as_bytes()[short.len()] == b'.'
    };
    if under(a, b) {
        Some((a, b))
    } else if under(b, a) {
        Some((b, a))
    } else {
        None
    }
}

/// A problem with a registry's keys as a whole.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegistryError {
    /// Two sections declare one full key path.
    DuplicateKey {
        /// The path.
        key: &'static str,
        /// The first section's type.
        first: &'static str,
        /// The second section's type.
        second: &'static str,
    },
    /// A key is a value, and another key or section treats it as a table.
    ValueIsTable {
        /// The key holding a value.
        value: &'static str,
        /// The key or section prefix under it.
        table: &'static str,
    },
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegistryError::DuplicateKey { key, first, second } => {
                write!(
                    f,
                    "config key `{key}` is declared by both `{first}` and `{second}`"
                )
            }
            RegistryError::ValueIsTable { value, table } => write!(
                f,
                "config key `{value}` holds a value, but `{table}` treats it as a table; TOML \
                 cannot hold both"
            ),
        }
    }
}

impl std::error::Error for RegistryError {}

/// Panics if `a` and `b` make the same claim.
const fn check_pair(a: &SectionSchema, b: &SectionSchema) {
    let a_owns_subtree = !a.shared && !a.prefix.is_empty();
    let b_owns_subtree = !b.shared && !b.prefix.is_empty();
    if a_owns_subtree && b_owns_subtree && str_eq(a.prefix, b.prefix) {
        let mut msg = Msg::new();
        msg.push("config sections `")
            .push(a.type_name)
            .push("` and `")
            .push(b.type_name)
            .push("` both claim the prefix `")
            .push(a.prefix)
            .push(".*`; every prefix has one owner (RFC-0054 §5.1)");
        msg.panic();
    }
    // A shared or top-level section claims each key by its full path.
    let a_claims_keys = !a_owns_subtree;
    let b_claims_keys = !b_owns_subtree;
    if a_claims_keys && b_claims_keys {
        let mut i = 0;
        while i < a.keys.len() {
            let mut j = 0;
            while j < b.keys.len() {
                if str_eq(a.keys[i].path, b.keys[j].path) {
                    let mut msg = Msg::new();
                    msg.push("config sections `")
                        .push(a.type_name)
                        .push("` and `")
                        .push(b.type_name)
                        .push("` both claim the key `")
                        .push(a.keys[i].path)
                        .push("`; every key has one owner (RFC-0054 §5.1)");
                    msg.panic();
                }
                j += 1;
            }
            i += 1;
        }
    }
}

const fn str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// A panic message built at compile time. Longer messages are cut off.
struct Msg {
    buf: [u8; 512],
    len: usize,
}

impl Msg {
    const fn new() -> Self {
        Self {
            buf: [0; 512],
            len: 0,
        }
    }

    const fn push(&mut self, s: &str) -> &mut Self {
        let bytes = s.as_bytes();
        let mut i = 0;
        // Stop at the buffer's end, and never split a multi-byte character.
        while i < bytes.len() && self.len < self.buf.len() {
            let width = match bytes[i] {
                b if b < 0x80 => 1,
                b if b >= 0xF0 => 4,
                b if b >= 0xE0 => 3,
                _ => 2,
            };
            if self.len + width > self.buf.len() {
                break;
            }
            let mut k = 0;
            while k < width {
                self.buf[self.len] = bytes[i + k];
                self.len += 1;
                k += 1;
            }
            i += width;
        }
        self
    }

    const fn panic(&self) -> ! {
        let (text, _) = self.buf.split_at(self.len);
        match core::str::from_utf8(text) {
            Ok(text) => panic!("{}", text),
            Err(_) => panic!("two config sections make the same claim (RFC-0054 §5.1)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::panic;

    use super::*;
    use crate::schema::{DefaultSpec, Reload, Scope};

    const fn key(path: &'static str) -> KeySchema {
        KeySchema {
            name: path,
            path,
            doc: "A key.",
            ty: "bool",
            scope: Scope::Node,
            reload: Reload::Hot,
            owner: "RFC-0054",
            default: DefaultSpec::Unset,
            min: None,
            max: None,
            introduced_at: None,
            authority: None,
            env: None,
            holds_secret: false,
        }
    }

    const fn section(
        type_name: &'static str,
        prefix: &'static str,
        shared: bool,
        keys: &'static [KeySchema],
    ) -> SectionSchema {
        SectionSchema {
            type_name,
            prefix,
            shared,
            owner: "RFC-0054",
            scope: Scope::Node,
            doc: "A section.",
            keys,
            check_defaults: Vec::new,
        }
    }

    static NET: SectionSchema = section("Net", "net", false, &[key("net.mtu")]);
    static NET_TOO: SectionSchema = section("NetToo", "net", false, &[key("net.other")]);
    static NET_SHARED: SectionSchema = section("NetShared", "net", true, &[key("net.relay_mode")]);
    static ROOT_A: SectionSchema = section("RootA", "", true, &[key("allow_approximate")]);
    static ROOT_B: SectionSchema = section("RootB", "", true, &[key("network_scope")]);
    static ROOT_DUP: SectionSchema = section("RootDup", "", true, &[key("network_scope")]);
    static API: SectionSchema = section("Api", "api", false, &[key("api.provenance")]);
    static API_PROV: SectionSchema = section(
        "ApiProvenance",
        "api.provenance",
        false,
        &[key("api.provenance.retention")],
    );
    static NET_DUP_KEY: SectionSchema = section("NetDupKey", "net", true, &[key("net.mtu")]);

    fn message(sections: &[&'static SectionSchema]) -> Option<String> {
        let sections: &'static [&'static SectionSchema] = Box::leak(sections.to_vec().into());
        let result = panic::catch_unwind(|| Registry::new(sections));
        result.err().map(|payload| {
            payload
                .downcast_ref::<String>()
                .cloned()
                .unwrap_or_else(|| payload.downcast_ref::<&str>().unwrap().to_string())
        })
    }

    #[test]
    fn accepts_distinct_and_shared_claims() {
        assert_eq!(message(&[&NET, &NET_SHARED, &ROOT_A, &ROOT_B]), None);
    }

    #[test]
    fn rejects_two_owners_of_one_prefix() {
        let msg = message(&[&NET, &NET_TOO]).unwrap();
        assert_eq!(
            msg,
            "config sections `Net` and `NetToo` both claim the prefix `net.*`; every prefix has \
             one owner (RFC-0054 §5.1)"
        );
    }

    #[test]
    fn rejects_two_owners_of_one_top_level_key() {
        let msg = message(&[&ROOT_A, &ROOT_B, &ROOT_DUP]).unwrap();
        assert!(msg.contains("both claim the key `network_scope`"), "{msg}");
    }

    #[test]
    fn check_finds_key_level_conflicts() {
        static REGISTRY: Registry = Registry::new(&[&NET, &NET_DUP_KEY, &API, &API_PROV]);
        let errors = REGISTRY.check();
        assert_eq!(
            errors,
            vec![
                RegistryError::DuplicateKey {
                    key: "net.mtu",
                    first: "Net",
                    second: "NetDupKey"
                },
                RegistryError::ValueIsTable {
                    value: "api.provenance",
                    table: "api.provenance.retention"
                },
                RegistryError::ValueIsTable {
                    value: "api.provenance",
                    table: "api.provenance"
                },
            ]
        );
        assert_eq!(REGISTRY.key("net.mtu").map(|k| k.path), Some("net.mtu"));
        assert_eq!(
            REGISTRY.section_of("api.provenance").map(|s| s.type_name),
            Some("Api")
        );
        assert!(REGISTRY.key("net").is_none());
    }

    #[test]
    fn long_messages_are_cut_at_a_character() {
        let mut msg = Msg::new();
        let mut i = 0;
        // 200 three-byte characters overflow the 512-byte buffer; 170 of them fit.
        while i < 200 {
            msg.push("€");
            i += 1;
        }
        assert!(core::str::from_utf8(&msg.buf[..msg.len]).is_ok());
        assert_eq!(msg.len, 510);
    }
}
