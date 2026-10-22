// SPDX-License-Identifier: MIT OR Apache-2.0

//! RFC-0054 §3's forms for `syntropy-core`'s policy value types (OQ-PRD-27 (a)).
//!
//! `core` holds [`Fraction`], [`Reserve`], [`NodeSelector`], [`DeviceSelector`],
//! [`TierSelector`] and [`LayerRange`] as plain data, with a serde form for records such as
//! `InstancePolicy`. `config` reads them in the forms §3 accepts, validates them, and prints
//! each in one canonical form that reads back:
//! - [`Form`] is implemented for each of the six;
//! - a key holding one reads it with `#[serde(with = "syntropy_config::form")]`, which also
//!   serves `Option<T>` and `Vec<T>` of them. The `ConfigSection` derive refuses a key of one
//!   of these types without it, so no key reads the record form by mistake;
//! - [`parse`] reads the text form, and [`Shown`] prints the canonical one.
//!
//! ```
//! use serde::Deserialize;
//! use syntropy_config::ConfigSection;
//! use syntropy_core::policy::{Fraction, Reserve};
//!
//! /// Example settings.
//! #[derive(Deserialize, ConfigSection)]
//! #[config(section = "example", scope = Node, owner = "RFC-0054")]
//! struct ExampleSection {
//!     /// RAM kept back.
//!     #[config(reload = Hot, default = "10%")]
//!     #[serde(with = "syntropy_config::form")]
//!     headroom: Reserve,
//!     /// A share.
//!     #[config(reload = Hot, default = "20%")]
//!     #[serde(with = "syntropy_config::form")]
//!     share: Fraction,
//! }
//!
//! assert!(ExampleSection::check_defaults().is_empty());
//! let share: Fraction = syntropy_config::form::parse("20%").unwrap();
//! assert_eq!(syntropy_config::form::Shown(&share).to_string(), "0.2");
//! ```

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
pub use syntropy_core::policy::{
    DeviceSelector, Fraction, LayerRange, NodeSelector, Reserve, TierSelector,
};

use super::ValueError;

/// A `core` value type `config` reads and prints in RFC-0054 §3's forms.
pub trait Form: Sized {
    /// Reads one of the §3 forms the value can take in TOML: a string, and for some types a
    /// number or a table.
    fn deserialize_form<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error>;

    /// Writes the canonical form, which [`Form::deserialize_form`] reads back.
    fn serialize_form<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error>;

    /// Parses the text form, as a string in TOML holds it.
    fn parse_text(text: &str) -> Result<Self, ValueError>;

    /// Writes the canonical form as text, which [`Form::parse_text`] reads back.
    fn fmt_text(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

/// Parses `text` into `T`, as a string in TOML holds it.
pub fn parse<T: Form>(text: &str) -> Result<T, ValueError> {
    T::parse_text(text)
}

/// Prints a value in its canonical form.
pub struct Shown<'a, T>(pub &'a T);

impl<T: Form> fmt::Display for Shown<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt_text(f)
    }
}

/// A key field `#[serde(with = "syntropy_config::form")]` serves: one of the six types, or an
/// `Option` or `Vec` of one.
pub trait Field: Sized {
    /// Reads the field.
    fn deserialize_field<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error>;
    /// Writes the field.
    fn serialize_field<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error>;
}

/// `serde(with)`'s reader: one of §3's forms, into `core`'s type.
pub fn deserialize<'de, D: Deserializer<'de>, T: Field>(deserializer: D) -> Result<T, D::Error> {
    T::deserialize_field(deserializer)
}

/// `serde(with)`'s writer: the canonical form.
pub fn serialize<S: Serializer, T: Field>(value: &T, serializer: S) -> Result<S::Ok, S::Error> {
    value.serialize_field(serializer)
}

/// A value read in its §3 form, for the `Option` and `Vec` fields.
struct Read<T>(T);

impl<'de, T: Form> Deserialize<'de> for Read<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        T::deserialize_form(deserializer).map(Read)
    }
}

/// A value written in its canonical form, for the `Option` and `Vec` fields.
struct Write<'a, T>(&'a T);

impl<T: Form> Serialize for Write<'_, T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize_form(serializer)
    }
}

macro_rules! fields {
    ($($ty:ty),*) => {$(
        impl Field for $ty {
            fn deserialize_field<'de, D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                <$ty as Form>::deserialize_form(d)
            }
            fn serialize_field<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                self.serialize_form(s)
            }
        }

        impl Field for Option<$ty> {
            fn deserialize_field<'de, D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                Option::<Read<$ty>>::deserialize(d).map(|value| value.map(|read| read.0))
            }
            fn serialize_field<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                match self {
                    Some(value) => s.serialize_some(&Write(value)),
                    None => s.serialize_none(),
                }
            }
        }

        impl Field for Vec<$ty> {
            fn deserialize_field<'de, D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                Vec::<Read<$ty>>::deserialize(d)
                    .map(|values| values.into_iter().map(|read| read.0).collect())
            }
            fn serialize_field<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.collect_seq(self.iter().map(Write))
            }
        }
    )*};
}

fields!(
    Fraction,
    Reserve,
    NodeSelector,
    DeviceSelector,
    TierSelector,
    LayerRange
);

/// Implements [`Form`]'s serde methods, inside its `impl`, for a type whose only form is a
/// string.
macro_rules! string_form {
    ($ty:ty, $expecting:literal) => {
        fn deserialize_form<'de, D: serde::Deserializer<'de>>(d: D) -> Result<$ty, D::Error> {
            struct V;
            impl serde::de::Visitor<'_> for V {
                type Value = $ty;
                fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    f.write_str($expecting)
                }
                fn visit_str<E: serde::de::Error>(self, s: &str) -> Result<$ty, E> {
                    <$ty as $crate::value::form::Form>::parse_text(s)
                        .map_err($crate::value::serde_error)
                }
            }
            d.deserialize_str(V)
        }

        fn serialize_form<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            s.collect_str(&$crate::value::form::Shown(self))
        }
    };
}
pub(crate) use string_form;
