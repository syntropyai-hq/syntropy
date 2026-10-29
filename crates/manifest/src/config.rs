// SPDX-License-Identifier: MIT OR Apache-2.0

//! `config.json` and `generation_config.json` of a safetensors checkpoint, read as metadata
//! (RFC-0041 §2.2 "Side files"; RFC-0043 §5.1 `MetaView`).
//!
//! [`HfConfig`] holds both files, parsed with the strict JSON reader the safetensors header
//! uses, so a repeated key fails instead of silently keeping one value. Values are read by
//! dotted path (`rope_scaling.factor`) with typed accessors that mirror
//! [`gguf::MetaView`](crate::gguf::MetaView)'s, and fail with the same [`MetaError`]. A JSON
//! `null` reads as absent: checkpoints write `"rope_scaling": null` for "no scaling".
//!
//! Which keys a family reads, and their ranges, are the family's (RFC-0043 §5.2); this module
//! only reads them.

use std::fmt;

use syntropy_core::ErrorClass;

use crate::gguf::MetaError;
use crate::safetensors::json::Json;
use crate::tokenizer::GENERATION_CONFIG_JSON;

/// The configuration side file's name (RFC-0041 §2.2).
pub const CONFIG_JSON: &str = "config.json";

/// Which of the two files a key is read from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConfigFile {
    /// `config.json`.
    Config,
    /// `generation_config.json`.
    Generation,
}

impl ConfigFile {
    /// The file's name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Config => CONFIG_JSON,
            Self::Generation => GENERATION_CONFIG_JSON,
        }
    }
}

/// A configuration file that is not a JSON object. Class `Integrity`; RFC-0041 §10 and
/// RFC-0043 §10 name no code for it yet, as for [`TokenizerSpecError`](crate::tokenizer::TokenizerSpecError).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigError {
    /// The file.
    pub file: &'static str,
    /// What is wrong.
    pub reason: String,
}

impl ConfigError {
    /// Its class (RFC-0001 §10).
    #[must_use]
    pub fn class(&self) -> ErrorClass {
        ErrorClass::Integrity
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.file, self.reason)
    }
}

impl std::error::Error for ConfigError {}

/// `config.json`, and `generation_config.json` when the checkpoint ships one.
#[derive(Clone, Debug, PartialEq)]
pub struct HfConfig {
    config: Vec<(String, Json)>,
    generation: Option<Vec<(String, Json)>>,
}

impl HfConfig {
    /// Parses both files.
    ///
    /// # Errors
    ///
    /// [`ConfigError`] when either file is not valid JSON, repeats a key, or is not an object.
    pub fn parse(config: &[u8], generation: Option<&[u8]>) -> Result<Self, ConfigError> {
        Ok(Self {
            config: object(ConfigFile::Config, config)?,
            generation: generation
                .map(|g| object(ConfigFile::Generation, g))
                .transpose()?,
        })
    }

    /// Whether `generation_config.json` was supplied.
    #[must_use]
    pub fn has_generation(&self) -> bool {
        self.generation.is_some()
    }

    /// The value at `path` in `file`, or `None` when a segment is missing, a parent is not an
    /// object, or the value is `null`.
    fn get(&self, file: ConfigFile, path: &str) -> Option<&Json> {
        let mut object = match file {
            ConfigFile::Config => &self.config,
            ConfigFile::Generation => self.generation.as_ref()?,
        };
        let mut segments = path.split('.').peekable();
        while let Some(segment) = segments.next() {
            let value = object.iter().find(|(k, _)| k == segment).map(|(_, v)| v)?;
            if segments.peek().is_none() {
                return (!matches!(value, Json::Null)).then_some(value);
            }
            match value {
                Json::Object(inner) => object = inner,
                _ => return None,
            }
        }
        None
    }

    /// Whether `path` holds a value other than `null`.
    #[must_use]
    pub fn contains(&self, file: ConfigFile, path: &str) -> bool {
        self.get(file, path).is_some()
    }

    fn err(path: &str, expected: &'static str, found: &Json) -> MetaError {
        MetaError {
            key: path.to_owned(),
            expected,
            found: found.kind().to_owned(),
        }
    }

    /// `path` as an unsigned integer.
    ///
    /// # Errors
    ///
    /// [`MetaError`] for any other value, a negative or fractional number included.
    pub fn u64(&self, file: ConfigFile, path: &str) -> Result<Option<u64>, MetaError> {
        match self.get(file, path) {
            None => Ok(None),
            Some(Json::Uint(n)) => Ok(Some(*n)),
            Some(v) => Err(Self::err(path, "a non-negative integer", v)),
        }
    }

    /// `path` as a float: any JSON number.
    ///
    /// # Errors
    ///
    /// [`MetaError`] for any other value.
    pub fn f64(&self, file: ConfigFile, path: &str) -> Result<Option<f64>, MetaError> {
        match self.get(file, path) {
            None => Ok(None),
            Some(Json::Uint(n)) => Ok(Some(*n as f64)),
            Some(Json::OtherNumber(x)) => Ok(Some(*x)),
            Some(v) => Err(Self::err(path, "a number", v)),
        }
    }

    /// `path` as a bool.
    ///
    /// # Errors
    ///
    /// [`MetaError`] for any other value.
    pub fn bool(&self, file: ConfigFile, path: &str) -> Result<Option<bool>, MetaError> {
        match self.get(file, path) {
            None => Ok(None),
            Some(Json::Bool(b)) => Ok(Some(*b)),
            Some(v) => Err(Self::err(path, "a bool", v)),
        }
    }

    /// `path` as a string.
    ///
    /// # Errors
    ///
    /// [`MetaError`] for any other value.
    pub fn str(&self, file: ConfigFile, path: &str) -> Result<Option<&str>, MetaError> {
        match self.get(file, path) {
            None => Ok(None),
            Some(Json::Str(s)) => Ok(Some(s)),
            Some(v) => Err(Self::err(path, "a string", v)),
        }
    }

    /// `path` as a list of unsigned integers. A single integer reads as a list of one, as
    /// `generation_config.json`'s `eos_token_id` may be either.
    ///
    /// # Errors
    ///
    /// [`MetaError`] for any other value or element.
    pub fn u64s(&self, file: ConfigFile, path: &str) -> Result<Option<Vec<u64>>, MetaError> {
        const EXPECTED: &str = "a list of non-negative integers";
        match self.get(file, path) {
            None => Ok(None),
            Some(Json::Uint(n)) => Ok(Some(vec![*n])),
            Some(Json::Array(items)) => items
                .iter()
                .map(|v| match v {
                    Json::Uint(n) => Ok(*n),
                    other => Err(Self::err(path, EXPECTED, other)),
                })
                .collect::<Result<_, _>>()
                .map(Some),
            Some(v) => Err(Self::err(path, EXPECTED, v)),
        }
    }

    /// `path` as a list of strings.
    ///
    /// # Errors
    ///
    /// [`MetaError`] for any other value or element.
    pub fn strs(&self, file: ConfigFile, path: &str) -> Result<Option<Vec<&str>>, MetaError> {
        const EXPECTED: &str = "a list of strings";
        match self.get(file, path) {
            None => Ok(None),
            Some(Json::Array(items)) => items
                .iter()
                .map(|v| match v {
                    Json::Str(s) => Ok(s.as_str()),
                    other => Err(Self::err(path, EXPECTED, other)),
                })
                .collect::<Result<_, _>>()
                .map(Some),
            Some(v) => Err(Self::err(path, EXPECTED, v)),
        }
    }

    /// The architecture names a family is resolved by (RFC-0043 §5.1): each entry of
    /// `architectures`, then `model_type`, in that order.
    ///
    /// # Errors
    ///
    /// [`MetaError`] when either key holds the wrong type.
    pub fn architectures(&self) -> Result<Vec<&str>, MetaError> {
        let mut names = self
            .strs(ConfigFile::Config, "architectures")?
            .unwrap_or_default();
        names.extend(self.str(ConfigFile::Config, "model_type")?);
        Ok(names)
    }
}

fn object(file: ConfigFile, bytes: &[u8]) -> Result<Vec<(String, Json)>, ConfigError> {
    let reason = match Json::parse(bytes) {
        Ok(Json::Object(o)) => return Ok(o),
        Ok(other) => format!("the file is {}, not a JSON object", other.kind()),
        Err(e) => format!("not valid JSON: {e}"),
    };
    Err(ConfigError {
        file: file.name(),
        reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &[u8] = br#"{
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "num_hidden_layers": 2,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000,
        "tie_word_embeddings": false,
        "rope_scaling": {"factor": 8.0, "rope_type": "llama3"},
        "quantization_config": null
    }"#;

    #[test]
    fn reads_typed_values_by_path() {
        let c = HfConfig::parse(CONFIG, Some(br#"{"eos_token_id": [1, 2]}"#)).unwrap();
        let f = ConfigFile::Config;
        assert_eq!(c.u64(f, "num_hidden_layers").unwrap(), Some(2));
        assert_eq!(c.f64(f, "rms_norm_eps").unwrap(), Some(1e-5));
        assert_eq!(c.f64(f, "rope_theta").unwrap(), Some(500_000.0));
        assert_eq!(c.f64(f, "rope_scaling.factor").unwrap(), Some(8.0));
        assert_eq!(c.str(f, "rope_scaling.rope_type").unwrap(), Some("llama3"));
        assert_eq!(c.bool(f, "tie_word_embeddings").unwrap(), Some(false));
        assert_eq!(
            c.u64(f, "quantization_config").unwrap(),
            None,
            "null is absent"
        );
        assert!(!c.contains(f, "quantization_config"));
        assert_eq!(c.u64(f, "missing.key").unwrap(), None);
        assert_eq!(c.u64(f, "num_hidden_layers.x").unwrap(), None);
        assert_eq!(
            c.u64s(ConfigFile::Generation, "eos_token_id").unwrap(),
            Some(vec![1, 2])
        );
        assert_eq!(c.architectures().unwrap(), ["LlamaForCausalLM", "llama"]);
    }

    #[test]
    fn wrong_types_name_the_key() {
        let c = HfConfig::parse(CONFIG, None).unwrap();
        let f = ConfigFile::Config;
        let e = c.u64(f, "rms_norm_eps").unwrap_err();
        assert_eq!(e.key, "rms_norm_eps");
        assert_eq!(e.found, "a number");
        assert!(c.str(f, "num_hidden_layers").is_err());
        assert!(c.strs(f, "model_type").is_err());
        assert_eq!(
            c.u64s(ConfigFile::Generation, "eos_token_id").unwrap(),
            None
        );
        let neg = HfConfig::parse(br#"{"n": -1}"#, None).unwrap();
        assert!(neg.u64(f, "n").is_err());
        assert_eq!(neg.f64(f, "n").unwrap(), Some(-1.0));
    }

    #[test]
    fn malformed_files_fail() {
        let e = HfConfig::parse(b"[1]", None).unwrap_err();
        assert_eq!(e.file, CONFIG_JSON);
        assert!(e.reason.contains("not a JSON object"), "{e}");
        let e = HfConfig::parse(br#"{"a": 1, "a": 2}"#, None).unwrap_err();
        assert!(e.reason.contains("duplicate key"), "{e}");
        let e = HfConfig::parse(b"{}", Some(b"{")).unwrap_err();
        assert_eq!(e.file, GENERATION_CONFIG_JSON);
        assert_eq!(e.class(), ErrorClass::Integrity);
    }
}
