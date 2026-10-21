// SPDX-License-Identifier: MIT OR Apache-2.0

//! Layered configuration loading, schema, policy resolution and per-request override validation.
//!
//! Layer 0 in RFC-0001 §4.1, designed in RFC-0054. This crate holds, so far (PROD-E1-T01):
//!
//! - the schema as code (RFC-0054 §10.3): `#[derive(ConfigSection)]`, which each crate uses to
//!   declare its section, and the [`SectionSchema`] and [`KeySchema`] it records;
//! - the section [`Registry`], whose construction fails the build when two sections claim one
//!   prefix (§5.1);
//! - the value types of §3: [`ByteSize`], [`Duration`], [`Fraction`], [`Reserve`], [`Rate`],
//!   the node, device, tier, layer and model selectors, [`Regex`] and [`Secret`].
//!
//! Loading the layers, validation and reload come in later tasks.
//!
//! # Declaring a section
//!
//! ```
//! use serde::Deserialize;
//! use syntropy_config::{ConfigSection, Duration, Reserve};
//!
//! /// Resource settings.
//! #[derive(Deserialize, ConfigSection)]
//! #[config(section = "resource", scope = Node, owner = "RFC-0020")]
//! struct ResourceSection {
//!     /// RAM never pinned or allocated by Syntropy.
//!     #[config(reload = Hot, default = computed("by mode: max(6GiB, 10% of RAM) for personal"),
//!              min = "1GiB")]
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
//! `tensor_override` and `approximate.*`.

#![forbid(unsafe_code)]

// Lets the derive's `::syntropy_config::…` paths resolve inside this crate's own tests.
extern crate self as syntropy_config;

mod default;
mod registry;
mod schema;
pub mod value;

pub use registry::{Registry, RegistryError};
pub use schema::{
    Authority, ConfigSection, DefaultSpec, DefaultValue, KeyError, KeySchema, Reload, Scope,
    SectionSchema,
};
/// Declares a configuration section. See the [crate documentation](crate) for the attributes.
pub use syntropy_config_derive::ConfigSection;
pub use value::{
    ByteSize, DeviceApi, DeviceSelector, Duration, Fraction, InRange, LayerRange, ModelSelector,
    NodeSelector, Rate, Regex, Reserve, Secret, SecretSource, TierSelector, ValueError,
    ValueErrorKind, never,
};

/// Items the derive's generated code calls. Not a public interface.
#[doc(hidden)]
pub mod __private {
    use serde::de::DeserializeOwned;

    use crate::default::UnsetDeserializer;
    use crate::schema::{DefaultSpec, DefaultValue, KeyError, KeySchema};
    use crate::value::{InRange, ValueError};

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
