// SPDX-License-Identifier: MIT OR Apache-2.0

//! Layered configuration loading, schema, policy resolution and per-request override validation.
//!
//! Layer 0 in RFC-0001 §4.1, designed in RFC-0054. This crate holds, so far:
//!
//! - the schema as code (RFC-0054 §10.3): `#[derive(ConfigSection)]`, which each crate uses to
//!   declare its section, and the [`SectionSchema`] and [`KeySchema`] it records;
//! - the section [`Registry`], whose construction fails the build when two sections claim one
//!   prefix (§5.1);
//! - the value types of §3: [`ByteSize`], [`Duration`], [`Rate`], the model selector,
//!   [`Regex`] and [`Secret`], and, for `syntropy-core`'s `Fraction`, `Reserve` and node,
//!   device, tier and layer selectors, their forms ([`form`]; OQ-PRD-27 (a));
//! - node-key layering (§1, §2.2 Rule 1, §2.4): the config files per OS ([`locate`],
//!   [`Locations::scan`]), the environment ([`Env`]) and command-line ([`Flags`]) layers, and
//!   [`ResolvedConfig`], an immutable snapshot whose every value carries its [`Origin`], held
//!   for readers by [`SharedConfig`];
//! - the text of `config show --resolved`, `config check` and `config path`
//!   ([`command`]);
//! - the debug-key gates of §7.1 ([`DebugSection`]): the three `debug.*` keys, the loopback
//!   predicate the API applies to the debug endpoints, the warning a [`ResolvedConfig`] carries
//!   while any key is on, and the predicate for a manifest signed by the CI test key;
//! - the validation of RFC-0054 §8 (PROD-E1-T03): [`load_and_validate`] runs phases 1–5 and keeps
//!   every error, each a [`Problem`] with its [`Code`], and [`diagnostic::render`] prints them in
//!   §8's format without ever showing a secret;
//! - an instance's typed section from its policy (§5.4, PROD-E1-T24): [`InstanceSection`] builds
//!   a domain crate's section from the `<prefix>` table of the policy, or from
//!   `InstancePolicy::domain`, with the computed defaults the caller supplies, checks it as a
//!   policy table is checked at load, and lists what changed between two builds with each key's
//!   reload class;
//! - per-request overrides (§7, PROD-E1-T10): [`RequestOverrides`] and [`resolve_request`], which
//!   resolves a request against one candidate instance's policy, enforcing D3: approximate only
//!   when the request, the instance and the cluster all opt in;
//! - writing a config file (§10.2, PROD-E1-T12): [`write::set`] and [`write::unset`] change only
//!   the lines of one key, [`write::check_edit`] validates the layer with the edited text before
//!   anything is written, and [`write::write_atomic`] replaces the file by a rename, keeping its
//!   permissions and refusing a symbolic link.
//!
//! The cluster and instance scopes, and reload, come in later tasks.
//!
//! # Loading a node's configuration
//!
//! ```
//! use std::path::Path;
//! use syntropy_config::{
//!     BUILTIN_RULES, Env, Flags, HostEnvironment, Mode, Os, Registry, command, load_and_validate,
//!     locate,
//! };
//!
//! // A binary registers every section it loads.
//! static REGISTRY: Registry = Registry::new(&[]);
//!
//! let os = Os::current();
//! let env = Env::from_pairs(os, [("SYNTROPY_TYPO", "1")]);
//! // `--config`'s value, if given, is the last argument. It replaces the other three files.
//! let locations = locate(os, Mode::User, &env, Some(Path::new("/no/such/config.toml")));
//! let problem = locations.scan().unwrap_err();
//! assert!(problem.to_string().contains("does not exist"));
//!
//! # // A doctest must not read this host's own config files, so it scans nothing.
//! # let scan = syntropy_config::Scan { system_file: None, dropins: vec![], user_file: None, explicit: None };
//! // let scan = locate(os, Mode::User, &env, None).scan()?;
//! let checked = load_and_validate(
//!     &REGISTRY, os, &scan, &env, &Flags::new(), |_| None, &[], &HostEnvironment,
//! );
//! let report = command::check(&checked, &REGISTRY);
//! assert_eq!(report.exit, command::EXIT_OK);
//! assert!(report.text.contains("warning[config/0003]: unknown variable: `SYNTROPY_TYPO`"));
//! assert!(report.text.contains("  --> env SYNTROPY_TYPO"));
//! # let _ = BUILTIN_RULES;
//! ```
//!
//! # Declaring a section
//!
//! ```
//! use serde::Deserialize;
//! use syntropy_config::{ConfigSection, Duration};
//! use syntropy_core::policy::Reserve;
//!
//! /// Resource settings.
//! #[derive(Deserialize, ConfigSection)]
//! #[config(section = "resource", scope = Node, owner = "RFC-0020")]
//! struct ResourceSection {
//!     /// RAM never pinned or allocated by Syntropy.
//!     #[config(reload = Hot, default = computed("by mode: max(6GiB, 10% of RAM) for personal"),
//!              min = "1GiB")]
//!     #[serde(with = "syntropy_config::form")]
//!     ram_headroom: Reserve,
//!     /// How long a probe waits.
//!     #[config(reload = Node, default = "7s")]
//!     wait: Duration,
//! }
//! ```
//!
//! Struct attributes, all in `#[config(...)]`:
//!
//! | Attribute | Required | Meaning |
//! |---|---|---|
//! | `section = "a.b"` | yes | the table the keys live in; `""` for top-level keys |
//! | `scope = Node \| Cluster \| Instance \| Request` | yes | the keys' scope unless a key overrides it |
//! | `owner = "RFC-NNNN"` | yes | the owning RFC unless a key overrides it |
//! | `shared` | no | own only these keys, inside a table another section owns |
//! | `deprecated = CONST` | no | a `&[Deprecated]` of keys this section renamed or removed (RFC-0054 §8); [`Registry::check`] validates the records |
//!
//! Key attributes, on each field:
//!
//! | Attribute | Required | Meaning |
//! |---|---|---|
//! | doc comment | yes | the key's description |
//! | `reload = Hot \| Instance \| Node` | yes | the reload class (RFC-0054 §9.1) |
//! | `default = …` | yes | a value (`"16GiB"`, `64`, `0.5`, `false`, `["a", "b"]`, `{ k = "v" }`), `unset` for an `Option` key, or `computed("rule")` |
//! | `min = …`, `max = …` | no | a range, in the type's [`InRange::Bound`] |
//! | `scope`, `owner` | no | override the section's |
//! | `introduced_at = <u32>` | cluster keys | the feature level that introduced the key (RFC-0032 §6.4) |
//! | `authority = Member \| Keyholder` | no | cluster and instance keys only; default `Member`, forced to `Keyholder` by RFC-0054 §2.3 step 10 |
//!
//! A missing doc comment, owner, scope, default or reload class is a compile error, as is a
//! cluster key without `introduced_at`, and `authority = Member` on a key RFC-0054 §2.3 step 10
//! reserves for keyholders: cluster keys under `security.*`, `invite.*`, `coordinator.*`,
//! `swim.*` and `update.*`, `cluster.lan_admission`, `cluster_aliases`, `network_scope` and
//! `allow_approximate`, and the instance keys `fidelity`, `default_fidelity`, `pin`,
//! `tensor_override` and `approximate.*`. So is a key of `syntropy-core`'s `Fraction`,
//! `Reserve`, `NodeSelector`, `DeviceSelector`, `TierSelector` or `LayerRange`, or an `Option`
//! or `Vec` of one, without `#[serde(with = "syntropy_config::form")]`: their own serde form is
//! the record encoding, which reads none of §3's text forms.

#![forbid(unsafe_code)]

// Lets the derive's `::syntropy_config::…` paths resolve inside this crate's own tests.
extern crate self as syntropy_config;

pub mod chars;
mod code;
pub mod command;
mod debug;
mod default;
pub mod diagnostic;
mod discovery;
mod env;
mod holds_secret;
mod instance;
mod origin;
mod registry;
mod render;
mod request;
mod resolve;
mod schema;
mod text;
mod validate;
pub mod value;
pub mod write;

pub use code::{Code, suggest};
pub use debug::{DebugApi, DebugSection, ManifestSignerNotAllowed, Peer, is_loopback};
pub use diagnostic::{Problem, Severity};
pub use discovery::{
    CONFIG_FLAG, CONFIG_VAR, Dirs, Explicit, HOME_VAR, Locations, Mode, Os, Runtime, Scan, locate,
    order_dropins,
};
pub use env::{API_KEY_VAR, Env, HF_TOKEN_VAR, HOST_VAR, OTHER_KNOWN, path_for, var_for};
pub use holds_secret::HoldsSecret;
pub use instance::{Change, InstanceSection, InstanceSectionError};
pub use origin::Origin;
pub use registry::{Registry, RegistryError};
pub use request::{
    BenchControls, EffectiveRequestPolicy, RequestError, RequestOverrides, RequestWarning,
    SessionBinding, knob_names, priority_for_service_tier, resolve_request,
};
pub use resolve::{
    ConfigFile, Flags, Layers, Proposal, Rejected, Resolved, ResolvedConfig, SectionError,
    SharedConfig, load,
};
pub use schema::{
    Authority, ConfigSection, DefaultSpec, DefaultValue, Deprecated, KeyError, KeySchema, Reload,
    Scope, SectionSchema,
};
/// Declares a configuration section. See the [crate documentation](crate) for the attributes.
pub use syntropy_config_derive::ConfigSection;
/// Implements [`HoldsSecret`] for a struct or enum used as a config value.
pub use syntropy_config_derive::HoldsSecret;
/// A TOML value, as the layers hold it.
pub use toml::Value;
pub use validate::{
    BUILTIN_RULES, CONFIG_VERSION, Checked, Environment, Findings, HostEnvironment, PathUse, Phase,
    Rule, RuleInput, config_version_problem, load_and_validate, validate,
};
pub use value::{
    ByteSize, Duration, InRange, ModelSelector, Rate, Regex, Secret, SecretSource, ValueError,
    ValueErrorKind, check_node_scope, form, never,
};

/// Items the derive's generated code calls. Not a public interface.
#[doc(hidden)]
pub mod __private {
    use serde::de::DeserializeOwned;

    use crate::default::UnsetDeserializer;
    use crate::schema::{DefaultSpec, DefaultValue, KeyError, KeySchema};
    use crate::value::{InRange, ValueError, ValueErrorKind, capture_kind, uncaptured_kind};

    /// The TOML value type the generated `check` functions take, so a crate that declares a
    /// section needs no `toml` dependency of its own.
    pub use toml;

    /// How a field with `#[serde(with = …)]` or `deserialize_with` reads a TOML value.
    pub type FromToml<T> = fn(toml::Value) -> Result<T, toml::de::Error>;

    /// What a default written in an attribute deserializes from.
    pub type DefaultDeserializer = crate::default::ValueDeserializer<'static>;

    /// How a field with `#[serde(with = …)]` or `deserialize_with` reads an attribute value.
    pub type FromDefault<T> = fn(DefaultDeserializer) -> Result<T, serde::de::value::Error>;

    /// Checks that `value` deserializes into `T`, keeping the kind of the error.
    pub fn check_value<T: DeserializeOwned>(value: &toml::Value) -> Result<T, ValueError> {
        check_value_by(value, T::deserialize)
    }

    /// A key's comparison of two values, for `KeySchema::same`: each is read through `.0`,
    /// the field's deserializer. The derive calls `(&Same::<T>(..)).same(a, b)` with both
    /// traits below in scope. Method lookup takes [`SameByValue`] when `T` implements
    /// `PartialEq`, and otherwise falls back to [`SameAsWritten`], which needs one more
    /// auto-reference.
    pub struct Same<T>(pub FromToml<T>);

    impl<T: DeserializeOwned> Same<T> {
        /// Reads through `T`'s own `Deserialize`.
        pub fn deserialized() -> Self {
            Same(T::deserialize)
        }
    }

    /// The comparison for a field type with `PartialEq`.
    pub trait SameByValue {
        /// Whether `a` and `b` read as equal values; as written when either does not read.
        fn same(&self, a: &toml::Value, b: &toml::Value) -> bool;
    }

    impl<T: PartialEq> SameByValue for Same<T> {
        fn same(&self, a: &toml::Value, b: &toml::Value) -> bool {
            match ((self.0)(a.clone()), (self.0)(b.clone())) {
                (Ok(read_a), Ok(read_b)) => read_a == read_b,
                _ => a == b,
            }
        }
    }

    /// The comparison for a field type without `PartialEq`, such as a `Secret`.
    pub trait SameAsWritten {
        /// Whether `a` and `b` are written alike.
        fn same(&self, a: &toml::Value, b: &toml::Value) -> bool;
    }

    impl<T> SameAsWritten for &Same<T> {
        fn same(&self, a: &toml::Value, b: &toml::Value) -> bool {
            a == b
        }
    }

    /// Checks that `value` deserializes through `de`, the field's `serde(with)` path. A
    /// failure keeps the kind this crate's types recorded; one they did not record is
    /// `out_of_range` for an integer outside the field's integer type and `wrong_type`
    /// otherwise.
    pub fn check_value_by<T>(value: &toml::Value, de: FromToml<T>) -> Result<T, ValueError> {
        let (result, kind) = capture_kind(|| de(value.clone()));
        result.map_err(|e| {
            let message = e.message();
            ValueError::new(
                kind.unwrap_or_else(|| uncaptured_kind(message)),
                message.to_owned(),
            )
        })
    }

    /// Checks that `value` deserializes into `T` and lies in `key`'s range.
    pub fn check_value_in_range<T: DeserializeOwned + InRange>(
        value: &toml::Value,
        key: &'static KeySchema,
    ) -> Result<(), ValueError> {
        in_range(&check_value::<T>(value)?, key)
    }

    /// Checks that `value` deserializes through `de` and lies in `key`'s range.
    pub fn check_value_by_in_range<T: InRange>(
        value: &toml::Value,
        key: &'static KeySchema,
        de: FromToml<T>,
    ) -> Result<(), ValueError> {
        in_range(&check_value_by(value, de)?, key)
    }

    /// Checks a key's default deserializes through `de`, the field's `serde(with)` path. An
    /// unset default is not checked: the field's `serde(default)` supplies its absence.
    pub fn check_default_by<T>(
        key: &'static KeySchema,
        errors: &mut Vec<KeyError>,
        de: FromDefault<T>,
    ) {
        if let Err(error) = default_by(key, de) {
            errors.push(KeyError {
                key: key.path,
                error,
            });
        }
    }

    /// Checks a key's default deserializes through `de` and lies in the key's range.
    pub fn check_default_by_in_range<T: InRange>(
        key: &'static KeySchema,
        errors: &mut Vec<KeyError>,
        de: FromDefault<T>,
    ) {
        let checked = default_by(key, de).and_then(|value| match value {
            Some(value) => in_range(&value, key),
            None => bounds::<T>(key).map(|_| ()),
        });
        if let Err(error) = checked {
            errors.push(KeyError {
                key: key.path,
                error,
            });
        }
    }

    /// The key's fixed default read through `de`; `None` for an unset or computed default.
    fn default_by<T>(key: &'static KeySchema, de: FromDefault<T>) -> Result<Option<T>, ValueError> {
        match &key.default {
            DefaultSpec::Value(value) => de(crate::default::ValueDeserializer::new(value))
                .map(Some)
                .map_err(|e| {
                    ValueError::new(
                        ValueErrorKind::WrongType,
                        format!("the default does not fit `{}`: {e}", key.ty),
                    )
                }),
            DefaultSpec::Unset | DefaultSpec::Computed(_) => Ok(None),
        }
    }

    /// Checks a key's default deserializes into `T`.
    pub fn check_default<T: DeserializeOwned>(key: &'static KeySchema, errors: &mut Vec<KeyError>) {
        if let Err(error) = default_of::<T>(key) {
            errors.push(KeyError {
                key: key.path,
                error,
            });
        }
    }

    /// Checks a key's default deserializes into `T` and lies in the key's range.
    pub fn check_default_in_range<T: DeserializeOwned + InRange>(
        key: &'static KeySchema,
        errors: &mut Vec<KeyError>,
    ) {
        let checked = default_of::<T>(key).and_then(|value| match value {
            Some(value) => in_range(&value, key),
            None => bounds::<T>(key).map(|_| ()),
        });
        if let Err(error) = checked {
            errors.push(KeyError {
                key: key.path,
                error,
            });
        }
    }

    /// Checks a value against its key's range.
    pub fn check_range<T: InRange>(value: &T, key: &'static KeySchema, errors: &mut Vec<KeyError>) {
        if let Err(error) = in_range(value, key) {
            errors.push(KeyError {
                key: key.path,
                error,
            });
        }
    }

    /// The key's fixed default as a `T`, `None` for a computed default.
    fn default_of<T: DeserializeOwned>(key: &KeySchema) -> Result<Option<T>, ValueError> {
        let describe = |e: ValueError| {
            ValueError::new(
                e.kind(),
                format!("the default does not fit `{}`: {}", key.ty, e.message()),
            )
        };
        match key.default {
            DefaultSpec::Value(value) => value.deserialize::<T>().map(Some).map_err(describe),
            DefaultSpec::Unset => T::deserialize(UnsetDeserializer).map(Some).map_err(|e| {
                describe(ValueError::new(
                    crate::ValueErrorKind::WrongType,
                    e.to_string(),
                ))
            }),
            DefaultSpec::Computed(_) => Ok(None),
        }
    }

    type Bounds<T> = (Option<<T as InRange>::Bound>, Option<<T as InRange>::Bound>);

    fn bounds<T: InRange>(key: &KeySchema) -> Result<Bounds<T>, ValueError> {
        let bound = |value: Option<DefaultValue>, which: &str| {
            value
                .map(|v| v.deserialize::<T::Bound>())
                .transpose()
                .map_err(|e| {
                    ValueError::new(
                        e.kind(),
                        format!("the {which} bound does not parse: {}", e.message()),
                    )
                })
        };
        Ok((bound(key.min, "min")?, bound(key.max, "max")?))
    }

    fn in_range<T: InRange>(value: &T, key: &KeySchema) -> Result<(), ValueError> {
        let (min, max) = bounds::<T>(key)?;
        value.check_range(min.as_ref(), max.as_ref())
    }
}
