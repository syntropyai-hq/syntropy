// SPDX-License-Identifier: MIT OR Apache-2.0

//! `MetaView`: one typed view over GGUF key-values and over `config.json` plus
//! `generation_config.json` (RFC-0043 §5.1).
//!
//! A family names each value once as a [`Key`]: its GGUF key and its `config.json` path. A
//! GGUF key may start with `{arch}`, which reads as the file's `general.architecture`, as
//! GGUF prefixes hyperparameters by architecture (`llama.block_count`). The accessors are the
//! same for both sources, so `parse_hparams` is written once per family. A value of the wrong
//! type, a required value that is missing and a value outside the family's range are all
//! `HyperParamInvalid` (code 3), naming the key as the source spells it.

use std::fmt::Display;
use std::ops::RangeInclusive;

use syntropy_manifest::config::{ConfigFile, HfConfig};
use syntropy_manifest::gguf::{self, MetaError, ValueRef};

use super::FamilyError;

/// The placeholder a GGUF key starts with to read under the file's architecture.
const ARCH: &str = "{arch}";

/// One hyperparameter's name in each source. `None` where a source does not carry it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Key {
    gguf: Option<&'static str>,
    hf: Option<(ConfigFile, &'static str)>,
}

impl Key {
    /// A GGUF key and a `config.json` path.
    #[must_use]
    pub const fn new(gguf: &'static str, config: &'static str) -> Self {
        Self {
            gguf: Some(gguf),
            hf: Some((ConfigFile::Config, config)),
        }
    }

    /// A GGUF key and a `generation_config.json` path.
    #[must_use]
    pub const fn generation(gguf: &'static str, generation: &'static str) -> Self {
        Self {
            gguf: Some(gguf),
            hf: Some((ConfigFile::Generation, generation)),
        }
    }

    /// A key only GGUF carries.
    #[must_use]
    pub const fn gguf(gguf: &'static str) -> Self {
        Self {
            gguf: Some(gguf),
            hf: None,
        }
    }

    /// A key only `config.json` carries.
    #[must_use]
    pub const fn config(config: &'static str) -> Self {
        Self {
            gguf: None,
            hf: Some((ConfigFile::Config, config)),
        }
    }
}

/// A checkpoint's metadata, from either source (RFC-0043 §5.1).
#[derive(Clone, Copy, Debug)]
pub enum MetaView<'a> {
    /// A GGUF file's key-values.
    Gguf(&'a gguf::MetaView),
    /// A safetensors checkpoint's `config.json` and `generation_config.json`.
    Config(&'a HfConfig),
}

impl MetaView<'_> {
    /// The architecture names a family is resolved by: GGUF `general.architecture`, or
    /// `config.json`'s `architectures` entries and then its `model_type`. Empty when the
    /// metadata names none.
    ///
    /// # Errors
    ///
    /// `HyperParamInvalid` when a key holds the wrong type.
    pub fn architectures(&self) -> Result<Vec<&str>, FamilyError> {
        match self {
            Self::Gguf(m) => Ok(m
                .architecture()
                .map_err(|e| meta_err("general.architecture", &e))?
                .into_iter()
                .collect()),
            Self::Config(c) => c
                .architectures()
                .map_err(|e| meta_err(format!("{}: {}", ConfigFile::Config.name(), e.key), &e)),
        }
    }

    /// The key's name in this source, as error messages give it; `None` when the source does
    /// not carry it.
    ///
    /// # Errors
    ///
    /// `HyperParamInvalid` when a GGUF key needs `{arch}` and the file names none.
    pub fn key_name(&self, key: Key) -> Result<Option<String>, FamilyError> {
        match self {
            Self::Gguf(m) => {
                let Some(k) = key.gguf else { return Ok(None) };
                let Some(rest) = k.strip_prefix(ARCH) else {
                    return Ok(Some(k.to_owned()));
                };
                let arch = m
                    .architecture()
                    .map_err(|e| meta_err("general.architecture", &e))?
                    .ok_or_else(|| {
                        FamilyError::invalid(k, "the file has no `general.architecture`")
                    })?;
                Ok(Some(format!("{arch}{rest}")))
            }
            Self::Config(_) => Ok(key
                .hf
                .map(|(file, path)| format!("{}: {path}", file.name()))),
        }
    }

    /// Reads `key` with `gguf_read` or `hf_read`, mapping a type error to `HyperParamInvalid`.
    fn read<T>(
        &self,
        key: Key,
        gguf_read: impl FnOnce(&gguf::MetaView, &str) -> Result<Option<T>, MetaError>,
        hf_read: impl FnOnce(&HfConfig, ConfigFile, &str) -> Result<Option<T>, MetaError>,
    ) -> Result<Option<T>, FamilyError> {
        let Some(name) = self.key_name(key)? else {
            return Ok(None);
        };
        let got = match (self, key.hf) {
            (Self::Gguf(m), _) => gguf_read(m, &name),
            (Self::Config(c), Some((file, path))) => hf_read(c, file, path),
            (Self::Config(_), None) => Ok(None),
        };
        got.map_err(|e| meta_err(name, &e))
    }

    /// `key` as an unsigned integer.
    ///
    /// # Errors
    ///
    /// `HyperParamInvalid` for a value of another type or a negative one.
    pub fn u64(&self, key: Key) -> Result<Option<u64>, FamilyError> {
        self.read(key, |m, k| m.u64(k), |c, f, p| c.u64(f, p))
    }

    /// `key` as a float: GGUF F32 or F64, or any JSON number.
    ///
    /// # Errors
    ///
    /// `HyperParamInvalid` for a value of another type.
    pub fn f64(&self, key: Key) -> Result<Option<f64>, FamilyError> {
        self.read(key, |m, k| m.f64(k), |c, f, p| c.f64(f, p))
    }

    /// `key` as a bool.
    ///
    /// # Errors
    ///
    /// `HyperParamInvalid` for a value of another type.
    pub fn bool(&self, key: Key) -> Result<Option<bool>, FamilyError> {
        self.read(key, |m, k| m.bool(k), |c, f, p| c.bool(f, p))
    }

    /// `key` as a string.
    ///
    /// # Errors
    ///
    /// `HyperParamInvalid` for a value of another type, or a GGUF string that is not UTF-8.
    pub fn str(&self, key: Key) -> Result<Option<String>, FamilyError> {
        self.read(
            key,
            |m, k| m.str(k).map(|s| s.map(str::to_owned)),
            |c, f, p| c.str(f, p).map(|s| s.map(str::to_owned)),
        )
    }

    /// `key` as a list of unsigned integers: a GGUF integer array, or a JSON integer or list
    /// of integers.
    ///
    /// # Errors
    ///
    /// `HyperParamInvalid` for a value or element of another type, or a negative one.
    pub fn u64s(&self, key: Key) -> Result<Option<Vec<u64>>, FamilyError> {
        self.read(key, gguf_u64s, |c, f, p| c.u64s(f, p))
    }

    /// `key` as an unsigned integer that must be present and in `range`.
    ///
    /// # Errors
    ///
    /// `HyperParamInvalid` when it is missing, of another type, or outside `range`.
    pub fn require_u64(&self, key: Key, range: RangeInclusive<u64>) -> Result<u64, FamilyError> {
        let v = self.u64(key)?;
        self.in_range(key, v, &range)
    }

    /// `key` as an unsigned integer in `range`, or `default` when it is absent.
    ///
    /// # Errors
    ///
    /// `HyperParamInvalid` when it is of another type or outside `range`.
    pub fn u64_or(
        &self,
        key: Key,
        default: u64,
        range: RangeInclusive<u64>,
    ) -> Result<u64, FamilyError> {
        let v = self.u64(key)?.or(Some(default));
        self.in_range(key, v, &range)
    }

    /// `key` as a float that must be present and in `range`. NaN is never in range.
    ///
    /// # Errors
    ///
    /// `HyperParamInvalid` when it is missing, of another type, or outside `range`.
    pub fn require_f64(&self, key: Key, range: RangeInclusive<f64>) -> Result<f64, FamilyError> {
        let v = self.f64(key)?;
        self.in_range(key, v, &range)
    }

    fn in_range<T: PartialOrd + Display + Copy>(
        &self,
        key: Key,
        value: Option<T>,
        range: &RangeInclusive<T>,
    ) -> Result<T, FamilyError> {
        let name = || {
            self.key_name(key)
                .ok()
                .flatten()
                .unwrap_or_else(|| "a key this source does not carry".to_owned())
        };
        match value {
            None => Err(FamilyError::invalid(name(), "required, and missing")),
            Some(v) if range.contains(&v) => Ok(v),
            Some(v) => Err(FamilyError::invalid(
                name(),
                format!("{v} is outside {}..={}", range.start(), range.end()),
            )),
        }
    }
}

fn meta_err(key: impl Into<String>, e: &MetaError) -> FamilyError {
    FamilyError::invalid(key, format!("holds {}, expected {}", e.found, e.expected))
}

fn gguf_u64s(m: &gguf::MetaView, key: &str) -> Result<Option<Vec<u64>>, MetaError> {
    const EXPECTED: &str = "an array of non-negative integers";
    let Some(array) = m.array(key)? else {
        return Ok(None);
    };
    let err = |found: String| MetaError {
        key: key.to_owned(),
        expected: EXPECTED,
        found,
    };
    array
        .iter()
        .map(|v| {
            let n: i128 = match v {
                ValueRef::U8(x) => x.into(),
                ValueRef::I8(x) => x.into(),
                ValueRef::U16(x) => x.into(),
                ValueRef::I16(x) => x.into(),
                ValueRef::U32(x) => x.into(),
                ValueRef::I32(x) => x.into(),
                ValueRef::U64(x) => x.into(),
                ValueRef::I64(x) => x.into(),
                other => return Err(err(format!("an element of type {:?}", other.value_type()))),
            };
            u64::try_from(n).map_err(|_| err(n.to_string()))
        })
        .collect::<Result<_, _>>()
        .map(Some)
}
