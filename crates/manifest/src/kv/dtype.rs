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
//! [`KvDTypeRule::resolve`] applies it inside the rule, where it can tell the family's
//! requirements from the source's default:
//!
//! - it replaces only the dtype step 3 gives, the source default. A family's per-layer
//!   requirements keep their dtype, and an override that differs from a family's requirement
//!   for every layer is refused ([`KvDTypeError::OverrideOverFamily`]);
//! - it never changes a declared FP8 cache ([`KvDTypeError::OverrideOverDeclaredCache`]): that
//!   cache's rounding is part of the reference, and no test bounds what removing it does to the
//!   output (ADR-0019);
//! - it never narrows the source default ([`KvDTypeError::OverrideNarrows`]). Narrower means
//!   fewer bits per element (F32 to F16 or BF16), or, at 16 bits, F16 over a BF16 default, which
//!   loses range; whether that one should be allowed is open for the maintainers (RFC-0042 §4.2,
//!   note of 2026-10-28). BF16 over an F16 default is RFC-0043 §10's sanctioned use, "force BF16
//!   where F16 overflows", and is allowed. Below 16 bits is always the `kv_lossy` approximate
//!   knob (RFC-0042 §10), which needs `fidelity = "approximate"` (D3), so no `kv_dtype` value
//!   is below 16 bits.
//!
//! The override changes the KV a sequence stores, so it changes the KV settings that prefix and
//! session keys hash (RFC-0022 §7.1; the hash itself is TIER-E7-T01's).

use std::fmt;

use serde::Deserialize;
use syntropy_config::{ConfigSection, HoldsSecret};
use syntropy_core::{DType, ErrorClass, LayerIx};

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

/// Why the rule gives no dtype. The first three are the caller's mistake, an input RFC-0042
/// §4.2 does not allow; the last three refuse an instance's `kv_dtype` override.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvDTypeError {
    /// A declared cache that is not an FP8 dtype.
    DeclaredNotFp8(DType),
    /// A family requirement that is not F16, BF16 or F32.
    FamilyNotFloat(DType),
    /// Two family requirements for one layer.
    LayerTwice(LayerIx),
    /// The override would store KV below the reference's dtype in exact mode.
    OverrideNarrows {
        /// The dtype the override would replace: the source default.
        reference: DType,
        /// The dtype it asks for.
        requested: DType,
    },
    /// The checkpoint declares an FP8 cache, which the override may not change.
    OverrideOverDeclaredCache {
        /// The declared cache.
        declared: DType,
        /// The dtype the override asks for.
        requested: DType,
    },
    /// The family requires a dtype for every layer, and the override asks for another.
    OverrideOverFamily {
        /// The family's dtype.
        required: DType,
        /// The dtype the override asks for.
        requested: DType,
    },
}

impl KvDTypeError {
    /// A stable name for the error, for logs and API errors.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::DeclaredNotFp8(_) => "kv_declared_not_fp8",
            Self::FamilyNotFloat(_) => "kv_family_not_float",
            Self::LayerTwice(_) => "kv_layer_twice",
            Self::OverrideNarrows { .. } => "kv_dtype_narrows",
            Self::OverrideOverDeclaredCache { .. } => "kv_dtype_over_declared_cache",
            Self::OverrideOverFamily { .. } => "kv_dtype_over_family",
        }
    }

    /// Its class (RFC-0001 §10): an override's refusal is `InvalidRequest`, the policy's to fix;
    /// a bad rule input is `Internal`, a bug in the family module or the census.
    #[must_use]
    pub fn class(&self) -> ErrorClass {
        match self {
            Self::DeclaredNotFp8(_) | Self::FamilyNotFloat(_) | Self::LayerTwice(_) => {
                ErrorClass::Internal
            }
            Self::OverrideNarrows { .. }
            | Self::OverrideOverDeclaredCache { .. }
            | Self::OverrideOverFamily { .. } => ErrorClass::InvalidRequest,
        }
    }
}

/// The `kv_dtype` value that names `d`.
fn value(d: DType) -> &'static str {
    match d {
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::F32 => "f32",
        _ => "?",
    }
}

impl fmt::Display for KvDTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeclaredNotFp8(d) => write!(f, "a declared KV cache must be FP8, not {d:?}"),
            Self::FamilyNotFloat(d) => {
                write!(f, "a family may require F16, BF16 or F32 KV, not {d:?}")
            }
            Self::LayerTwice(l) => write!(f, "layer {} has two required KV dtypes", l.0),
            Self::OverrideNarrows {
                reference,
                requested,
            } => write!(
                f,
                "kv_dtype = \"{}\" would store KV below the model's reference dtype, {reference:?}, \
                 in exact mode; lower KV precision is the approximate.kv_lossy knob, which needs \
                 fidelity = \"approximate\"",
                value(*requested)
            ),
            Self::OverrideOverDeclaredCache {
                declared,
                requested,
            } => write!(
                f,
                "kv_dtype = \"{}\" cannot replace the {declared:?} KV cache the checkpoint \
                 declares: that cache is part of the model's reference",
                value(*requested)
            ),
            Self::OverrideOverFamily {
                required,
                requested,
            } => write!(
                f,
                "kv_dtype = \"{}\" cannot replace the {required:?} KV the model's family \
                 requires on every layer",
                value(*requested)
            ),
        }
    }
}

impl std::error::Error for KvDTypeError {}

/// The dtypes a family may require, or an override may set.
fn is_wide_float(d: DType) -> bool {
    matches!(d, DType::F16 | DType::BF16 | DType::F32)
}

/// Whether `requested` stores KV below `reference` (module docs): fewer bits per element, or F16
/// over BF16. BF16 over F16 is not narrower.
fn narrows(reference: DType, requested: DType) -> bool {
    match (reference, requested) {
        (DType::BF16, DType::F16) => true,
        (DType::F16, DType::BF16) => false,
        _ => requested.bits() < reference.bits(),
    }
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
        self.resolve(None)
    }

    /// The model's KV dtype per layer with an instance's `kv_dtype` override applied (module
    /// docs), in normal form. No override gives the [`reference`](Self::reference).
    ///
    /// # Errors
    ///
    /// The [`reference`](Self::reference)'s errors, and, for the override:
    /// [`KvDTypeError::OverrideOverDeclaredCache`] when the checkpoint declares a cache,
    /// [`KvDTypeError::OverrideOverFamily`] when the family requires another dtype on every
    /// layer, and [`KvDTypeError::OverrideNarrows`] when it is below the source default.
    pub fn resolve(&self, kv_dtype: Option<KvDTypeOverride>) -> Result<KvDTypeSpec, KvDTypeError> {
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
        let requested = kv_dtype.map(KvDTypeOverride::dtype);
        if let Some(declared) = self.declared_cache {
            return match requested {
                Some(requested) if requested != declared => {
                    Err(KvDTypeError::OverrideOverDeclaredCache {
                        declared,
                        requested,
                    })
                }
                _ => Ok(KvDTypeSpec::uniform(declared)),
            };
        }
        let default = match (self.family, requested) {
            (Some(required), Some(requested)) if requested != required => {
                return Err(KvDTypeError::OverrideOverFamily {
                    required,
                    requested,
                });
            }
            (Some(required), _) => required,
            (None, requested) => {
                let reference = Self::source_default(self.source);
                match requested {
                    Some(requested) if narrows(reference, requested) => {
                        return Err(KvDTypeError::OverrideNarrows {
                            reference,
                            requested,
                        });
                    }
                    Some(requested) => requested,
                    None => reference,
                }
            }
        };
        // The family's per-layer requirements, as the family gave them: an override never moves
        // one, even one equal to the source default that the normal form would fold away.
        let spec = KvDTypeSpec {
            default,
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
    /// The model's KV dtype in exact mode, in place of the source's default dtype: `f16`, `bf16`
    /// or `f32`. Forcing `bf16` where F16 overflows is the usual use (RFC-0042 §4.2). It changes
    /// `KvFormula::dtype`, and so prefix and session keys. It never replaces a family's required
    /// dtype or a declared FP8 cache, and never narrows the default: [`KvDTypeRule::resolve`]
    /// refuses those. Lower KV precision is `approximate.kv_lossy`, not this key.
    #[config(
        reload = Instance,
        default = computed("the family's reference KV dtype (RFC-0042 §4.2)")
    )]
    pub kv_dtype: Option<KvDTypeOverride>,
}

#[cfg(test)]
mod tests;
