// SPDX-License-Identifier: MIT OR Apache-2.0

//! The reference's KV dtype per family and source, and the `kv_dtype` override (RFC-0042 §4.2;
//! RFC-0043 §5, §10; COMP-E13-T06).
//!
//! [`KvDTypeRule::reference`] is RFC-0042 §4.2's rule, which a family module's `kv_formula`
//! calls to fill [`KvFormula::dtype`](super::KvFormula::dtype). In order:
//!
//! 1. **A declared FP8 cache.** Where the checkpoint's own configuration declares an FP8 cache,
//!    every layer stores FP8 (ADR-0019's revisit case). Which configuration fields declare one
//!    is for the caller, the family module or the census, to read; the rule takes the declared
//!    dtype ([`KvDTypeRule::declared_cache`]).
//! 2. **A family requirement.** A family module may require a dtype, such as BF16 where F16
//!    would overflow (RFC-0042 open question 4) or F32, for every layer or for some layers.
//! 3. **The source.** F16 for GGUF, matching ggml's default cache type so that calibration
//!    agrees. For safetensors: BF16 when the weights are quantised below 16 bits; otherwise the
//!    checkpoint's compute dtype when it is F16, BF16 or F32, and BF16 when it is absent or
//!    anything else.
//!
//! The safetensors case for F16 and F32 checkpoints is not in RFC-0042 §4.2's text, which names
//! BF16 "for safetensors sources whose weights are BF16 or lower". Its note of 2026-10-28 records
//! this reading: the reference stores KV in its compute dtype. That reading has not been checked
//! against the publishers' code.
//!
//! **The override.** An instance's policy may set `kv_dtype` to `f16`, `bf16` or `f32`
//! ([`KvDTypeSection`], a top-level instance key: `[policy."<selector>"] kv_dtype = "bf16"`).
//! [`KvDTypeSpec::with_override`] applies it to the reference: it replaces the default dtype,
//! and keeps the layers a family requires. It changes the KV a sequence stores, so it changes
//! the KV settings that prefix and session keys hash (RFC-0022 §7.1; the hash itself is
//! TIER-E7-T01's). A dtype below the reference's is never set here: that is the `kv_lossy`
//! approximate knob (RFC-0042 §10).

use std::fmt;

use serde::Deserialize;
use syntropy_config::{ConfigSection, HoldsSecret};
use syntropy_core::{DType, LayerIx};

use super::KvDTypeSpec;

/// The container a model's weights come from, with what the rule needs to know of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvSource {
    /// A GGUF file or split.
    Gguf,
    /// A safetensors checkpoint.
    Safetensors {
        /// The checkpoint's compute dtype, as `config.json`'s `torch_dtype` (or `dtype`) gives
        /// it, if it does.
        compute: Option<DType>,
        /// Whether the weights are quantised below 16 bits (a `quantization_config` the reader
        /// maps, such as FP8, MXFP4 or 4-bit integer groups).
        quantized: bool,
    },
}

/// The inputs of RFC-0042 §4.2's rule for one model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvDTypeRule {
    /// Where the weights come from.
    pub source: KvSource,
    /// The FP8 cache dtype the checkpoint's configuration declares, if it declares one. The
    /// caller reads the fields that declare it.
    pub declared_cache: Option<DType>,
    /// A dtype the family requires for every layer, if it requires one.
    pub family: Option<DType>,
    /// Layers for which the family requires a dtype of their own.
    pub family_layers: Vec<(LayerIx, DType)>,
}

/// An input the rule refuses: the caller passed something RFC-0042 §4.2 does not allow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvDTypeError {
    /// A declared cache that is not an FP8 dtype.
    DeclaredNotFp8(DType),
    /// A family requirement that is not F16, BF16 or F32.
    FamilyNotFloat(DType),
    /// Two family requirements for one layer.
    LayerTwice(LayerIx),
}

impl fmt::Display for KvDTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeclaredNotFp8(d) => write!(f, "a declared KV cache must be FP8, not {d:?}"),
            Self::FamilyNotFloat(d) => {
                write!(f, "a family may require F16, BF16 or F32 KV, not {d:?}")
            }
            Self::LayerTwice(l) => write!(f, "layer {} has two required KV dtypes", l.0),
        }
    }
}

impl std::error::Error for KvDTypeError {}

/// The dtypes a family may require, or an override may set.
fn is_wide_float(d: DType) -> bool {
    matches!(d, DType::F16 | DType::BF16 | DType::F32)
}

impl KvDTypeRule {
    /// The rule for a source with no declared cache and no family requirement.
    #[must_use]
    pub fn for_source(source: KvSource) -> Self {
        Self {
            source,
            declared_cache: None,
            family: None,
            family_layers: Vec::new(),
        }
    }

    /// The dtype the source alone gives (step 3 of the module docs).
    #[must_use]
    pub fn source_default(source: KvSource) -> DType {
        match source {
            KvSource::Gguf => DType::F16,
            KvSource::Safetensors {
                quantized: true, ..
            } => DType::BF16,
            KvSource::Safetensors {
                compute: Some(d), ..
            } if is_wide_float(d) => d,
            KvSource::Safetensors { .. } => DType::BF16,
        }
    }

    /// The reference's KV dtype per layer (RFC-0042 §4.2), in normal form
    /// ([`KvDTypeSpec::normalized`]).
    ///
    /// # Errors
    ///
    /// [`KvDTypeError`] for a declared cache that is not FP8, a family requirement that is not
    /// F16, BF16 or F32, or two requirements for one layer.
    pub fn reference(&self) -> Result<KvDTypeSpec, KvDTypeError> {
        if let Some(d) = self.declared_cache
            && !matches!(d, DType::F8E4M3 | DType::F8E5M2)
        {
            return Err(KvDTypeError::DeclaredNotFp8(d));
        }
        if let Some(d) = self.family
            && !is_wide_float(d)
        {
            return Err(KvDTypeError::FamilyNotFloat(d));
        }
        for (i, (layer, d)) in self.family_layers.iter().enumerate() {
            if !is_wide_float(*d) {
                return Err(KvDTypeError::FamilyNotFloat(*d));
            }
            if self.family_layers[..i].iter().any(|(l, _)| l == layer) {
                return Err(KvDTypeError::LayerTwice(*layer));
            }
        }
        if let Some(d) = self.declared_cache {
            return Ok(KvDTypeSpec::uniform(d));
        }
        let spec = KvDTypeSpec {
            default: self
                .family
                .unwrap_or_else(|| Self::source_default(self.source)),
            layers: self.family_layers.clone(),
        };
        Ok(spec.normalized())
    }
}

impl KvDTypeSpec {
    /// The same dtypes in one form: exceptions sorted by layer, each layer once, and none equal
    /// to the default. Two specs that give every layer the same dtype are equal in this form,
    /// so a hash of it changes only when a layer's dtype does.
    #[must_use]
    pub fn normalized(&self) -> Self {
        let mut layers: Vec<(LayerIx, DType)> = Vec::with_capacity(self.layers.len());
        for &(layer, d) in &self.layers {
            // The first exception for a layer is the one `of` reads.
            if !layers.iter().any(|(l, _)| *l == layer) {
                layers.push((layer, d));
            }
        }
        layers.retain(|&(_, d)| d != self.default);
        layers.sort_unstable_by_key(|&(l, _)| l);
        Self {
            default: self.default,
            layers,
        }
    }

    /// The spec with the instance's `kv_dtype` override applied: the override replaces the
    /// default, and the layers the family requires keep their dtype. No override leaves the
    /// spec as it is. The result is in normal form.
    #[must_use]
    pub fn with_override(&self, kv_dtype: Option<KvDTypeOverride>) -> Self {
        match kv_dtype {
            None => self.normalized(),
            Some(o) => Self {
                default: o.dtype(),
                layers: self.layers.clone(),
            }
            .normalized(),
        }
    }
}

/// The values `kv_dtype` takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, HoldsSecret)]
#[serde(rename_all = "lowercase")]
pub enum KvDTypeOverride {
    /// F16.
    F16,
    /// BF16: the usual case, where F16 overflows on a GGUF source (RFC-0042 open question 4).
    Bf16,
    /// F32.
    F32,
}

impl KvDTypeOverride {
    /// The dtype.
    #[must_use]
    pub fn dtype(self) -> DType {
        match self {
            Self::F16 => DType::F16,
            Self::Bf16 => DType::BF16,
            Self::F32 => DType::F32,
        }
    }
}

/// The top-level `kv_dtype` instance key (RFC-0043 §10; RFC-0054 §5.4).
///
/// It is instance scope, so a node's files set it only inside a `[policy."<selector>"]` table,
/// and every ingress node builds the same instance from it (RFC-0054 §2.2 Rule 3). It is built
/// from an instance's policy by `config`'s builder, `InstanceSection::<KvDTypeSection>`, as
/// every domain section is. Unset, it is `None`: the family's reference dtype.
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "", scope = Instance, owner = "RFC-0043")]
pub struct KvDTypeSection {
    /// The model's KV dtype in exact mode, in place of the family's reference dtype: `f16`,
    /// `bf16` or `f32`. Forcing `bf16` where F16 overflows is the usual use (RFC-0042 §4.2). It
    /// changes `KvFormula::dtype`, and so prefix and session keys. A dtype below the
    /// reference's is `approximate.kv_lossy`, not this key.
    #[config(
        reload = Instance,
        default = computed("the family's reference KV dtype (RFC-0042 §4.2)")
    )]
    pub kv_dtype: Option<KvDTypeOverride>,
}

#[cfg(test)]
mod tests;
