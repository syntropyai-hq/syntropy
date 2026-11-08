// SPDX-License-Identifier: MIT OR Apache-2.0

//! Family modules and their static registry (RFC-0043 §5; ADR-0003 decisions 1 and 6).
//!
//! - [`Family`] is RFC-0043 §5.1's trait, with the methods whose types exist so far: the tag,
//!   the accepted architecture names, the version, [`parse_hparams`](Family::parse_hparams),
//!   [`classify`](Family::classify), [`kv_formula`](Family::kv_formula),
//!   [`tokenizer_spec`](Family::tokenizer_spec) and [`required_ops`](Family::required_ops).
//!   `build`, `routing_known`, `speculative_support`, `tp_sharding`, `chat_format`, `embedding`
//!   and `fixture_spec` join it with the tasks that add their types (the census graph,
//!   `RoutingKnown`, the chat formats, `FamilySpec`).
//! - [`MetaView`] reads GGUF metadata or `config.json` plus `generation_config.json` through
//!   the same accessors, so `parse_hparams` is written once per family (§5.1).
//! - [`Registry`] holds the compiled-in families; there are no plugins. [`families()`] is the
//!   registry this build serves. [`Registry::resolve`] fails closed: an architecture no family
//!   accepts is `UnknownArchitecture` (code 1), naming it and the supported families
//!   (ADR-0003 decision 6).
//! - **Tags are frozen** ([`tags`]). A tag is hashed into `ModelId` (RFC-0041 §1), so it is
//!   never renamed or reused for another module. [`Registry::check`] holds every registered
//!   family to the checked-in list, and `tests/family_tags.rs` pins the list itself.
//!
//! [`llama`] (COMP-E16-T03) and [`qwen_moe`] (COMP-E16-T04) are registered. Certification
//! gating of [`families()`] is COMP-E16-T02.

use std::fmt;

use syntropy_core::ErrorClass;
use syntropy_manifest::hparams::HyperParams;
use syntropy_manifest::kv::{KvFormula, KvSource};
use syntropy_manifest::tokenizer::{TokenizerSources, TokenizerSpec, TokenizerSpecError};

use crate::classify::Classification;
use crate::ir::OpSet;

pub mod llama;
mod meta;
pub mod qwen_moe;
pub mod tags;

pub use meta::{Key, MetaView};
pub use tags::{FROZEN_TAGS, FrozenTag};

/// One architecture family's module (RFC-0043 §5.1).
pub trait Family: Send + Sync {
    /// The frozen tag: `"llama"`, `"qwen_moe"`, … ([`tags`]).
    fn name(&self) -> &'static str;
    /// The GGUF `general.architecture` values and the `config.json` `architectures` and
    /// `model_type` values it accepts.
    fn accepts(&self) -> &'static [&'static str];
    /// Informational, shown in logs and the dashboard. Certification keys on the module's
    /// code hash (RFC-0043 §9.4), never on this number.
    fn version(&self) -> u32;
    /// Reads and range-checks the hyperparameters.
    ///
    /// # Errors
    ///
    /// [`FamilyError::HyperParamInvalid`] for a missing or out-of-range key.
    fn parse_hparams(&self, meta: &MetaView<'_>) -> Result<HyperParams, FamilyError>;
    /// Classifies one tensor from its name and its shape in the source's order (RFC-0043 §4).
    fn classify(&self, name: &str, shape: &[u64], hp: &HyperParams) -> Classification;
    /// The KV and state size per layer, with the reference's KV dtype, for weights from
    /// `source` (RFC-0022 §2; RFC-0042 §4.2's rule, `KvDTypeRule`).
    fn kv_formula(&self, hp: &HyperParams, source: KvSource) -> KvFormula;
    /// The tokenizer data of a checkpoint's tokenizer side files (RFC-0043 §8.1). The default
    /// is [`TokenizerSpec::from_sources`]; a family overrides it only to supply what its
    /// checkpoints leave out.
    ///
    /// # Errors
    ///
    /// [`TokenizerSpecError`] for a side file that cannot be read as a tokenizer source.
    fn tokenizer_spec(
        &self,
        sources: TokenizerSources<'_>,
    ) -> Result<TokenizerSpec, TokenizerSpecError> {
        TokenizerSpec::from_sources(sources)
    }
    /// The ops its graphs use (RFC-0040 §11.1).
    fn required_ops(&self, hp: &HyperParams) -> OpSet;
}

/// A failure of family resolution or hyperparameter parsing, with RFC-0043 §10's code in
/// `Domain::Graph`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FamilyError {
    /// `UnknownArchitecture`, `(Graph, 1)`, class `Unsupported`: no module accepts the
    /// architecture name.
    UnknownArchitecture {
        /// The names the metadata gives, in the order they were tried.
        names: Vec<String>,
        /// The tags of the families this build registers.
        supported: Vec<&'static str>,
    },
    /// `HyperParamInvalid`, `(Graph, 3)`, class `Unsupported`: a required key is missing, of
    /// the wrong type, or out of range.
    HyperParamInvalid {
        /// The key as the source names it, with its file for `config.json` sources.
        key: String,
        /// What is wrong.
        reason: String,
    },
}

impl FamilyError {
    /// The `Graph` code RFC-0043 §10 gives it.
    #[must_use]
    pub fn code(&self) -> u16 {
        match self {
            Self::UnknownArchitecture { .. } => 1,
            Self::HyperParamInvalid { .. } => 3,
        }
    }

    /// Its class (RFC-0001 §10).
    #[must_use]
    pub fn class(&self) -> ErrorClass {
        ErrorClass::Unsupported
    }

    /// `HyperParamInvalid` for `key`.
    #[must_use]
    pub fn invalid(key: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::HyperParamInvalid {
            key: key.into(),
            reason: reason.into(),
        }
    }
}

impl fmt::Display for FamilyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownArchitecture { names, supported } => {
                write!(f, "UnknownArchitecture (Graph 1): ")?;
                if names.is_empty() {
                    write!(f, "the metadata names no architecture")?;
                } else {
                    write!(f, "no family accepts {names:?}")?;
                }
                if supported.is_empty() {
                    write!(f, "; this build registers no family")
                } else {
                    write!(f, "; supported families: {}", supported.join(", "))
                }
            }
            Self::HyperParamInvalid { key, reason } => {
                write!(f, "HyperParamInvalid (Graph 3): `{key}`: {reason}")
            }
        }
    }
}

impl std::error::Error for FamilyError {}

/// One compiled-in family and the module that defines it.
#[derive(Clone, Copy)]
pub struct Registration {
    /// The defining module's path, from `module_path!()`, which the frozen list binds its tag
    /// to.
    pub module: &'static str,
    /// The family.
    pub family: &'static dyn Family,
}

impl fmt::Debug for Registration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Registration")
            .field("module", &self.module)
            .field("tag", &self.family.name())
            .finish()
    }
}

/// A set of compiled-in families (RFC-0043 §5.1).
#[derive(Clone, Copy, Debug)]
pub struct Registry {
    entries: &'static [Registration],
}

/// Why a registry breaks the frozen-tag rule (RFC-0043 §5.1).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegistryError {
    /// The frozen list names a tag twice.
    DuplicateListedTag(&'static str),
    /// The frozen list binds two tags to one module.
    DuplicateListedModule(&'static str),
    /// Two registrations carry the same tag.
    DuplicateTag(&'static str),
    /// A registered family's tag is not in the frozen list.
    Unlisted {
        /// The tag.
        tag: &'static str,
        /// The module registering it.
        module: &'static str,
    },
    /// A registered family carries a listed tag from a module the list does not bind it to.
    Reassigned {
        /// The tag.
        tag: &'static str,
        /// The module the list binds it to.
        listed: &'static str,
        /// The module registering it.
        module: &'static str,
    },
    /// Two families accept the same architecture name, so resolution would depend on order.
    SharedArchitecture {
        /// The name.
        name: &'static str,
        /// The two tags.
        tags: [&'static str; 2],
    },
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateListedTag(t) => write!(f, "the frozen list names `{t}` twice"),
            Self::DuplicateListedModule(m) => {
                write!(f, "the frozen list binds two tags to `{m}`")
            }
            Self::DuplicateTag(t) => write!(f, "two families are registered as `{t}`"),
            Self::Unlisted { tag, module } => write!(
                f,
                "`{module}` registers `{tag}`, which the frozen list does not name; append it"
            ),
            Self::Reassigned {
                tag,
                listed,
                module,
            } => write!(
                f,
                "`{module}` registers `{tag}`, which the frozen list binds to `{listed}`; a tag is never reassigned"
            ),
            Self::SharedArchitecture { name, tags } => write!(
                f,
                "`{}` and `{}` both accept the architecture `{name}`",
                tags[0], tags[1]
            ),
        }
    }
}

impl std::error::Error for RegistryError {}

impl Registry {
    /// A registry of `entries`.
    #[must_use]
    pub const fn new(entries: &'static [Registration]) -> Self {
        Self { entries }
    }

    /// The registrations, in order.
    #[must_use]
    pub fn entries(&self) -> &'static [Registration] {
        self.entries
    }

    /// The family registered as `tag`.
    #[must_use]
    pub fn get(&self, tag: &str) -> Option<&'static dyn Family> {
        self.entries
            .iter()
            .find(|r| r.family.name() == tag)
            .map(|r| r.family)
    }

    /// The registered tags, in order.
    #[must_use]
    pub fn tags(&self) -> Vec<&'static str> {
        self.entries.iter().map(|r| r.family.name()).collect()
    }

    /// The family that accepts the first of `names` any family accepts.
    ///
    /// # Errors
    ///
    /// [`FamilyError::UnknownArchitecture`] when none does.
    pub fn resolve_names(&self, names: &[&str]) -> Result<&'static dyn Family, FamilyError> {
        names
            .iter()
            .find_map(|name| {
                self.entries
                    .iter()
                    .find(|r| r.family.accepts().contains(name))
            })
            .map(|r| r.family)
            .ok_or_else(|| FamilyError::UnknownArchitecture {
                names: names.iter().map(|&n| n.to_owned()).collect(),
                supported: self.tags(),
            })
    }

    /// The family for a checkpoint's metadata: GGUF `general.architecture`, or `config.json`'s
    /// `architectures` entries and then its `model_type`.
    ///
    /// # Errors
    ///
    /// [`FamilyError::UnknownArchitecture`] when no family accepts any of them, or the
    /// metadata names none; [`FamilyError::HyperParamInvalid`] when a key holds the wrong
    /// type.
    pub fn resolve(&self, meta: &MetaView<'_>) -> Result<&'static dyn Family, FamilyError> {
        self.resolve_names(&meta.architectures()?)
    }

    /// Checks the registrations against the frozen list `frozen` (RFC-0043 §5.1): the list
    /// names each tag and each module once; every registered tag is listed, bound to the
    /// module that registers it; no tag is registered twice; and no architecture name is
    /// accepted by two families.
    ///
    /// # Errors
    ///
    /// The first [`RegistryError`] found.
    pub fn check(&self, frozen: &[FrozenTag]) -> Result<(), RegistryError> {
        for (i, a) in frozen.iter().enumerate() {
            for b in &frozen[i + 1..] {
                if a.tag == b.tag {
                    return Err(RegistryError::DuplicateListedTag(a.tag));
                }
                if a.module == b.module {
                    return Err(RegistryError::DuplicateListedModule(a.module));
                }
            }
        }
        for (i, r) in self.entries.iter().enumerate() {
            let tag = r.family.name();
            match frozen.iter().find(|f| f.tag == tag) {
                None => {
                    return Err(RegistryError::Unlisted {
                        tag,
                        module: r.module,
                    });
                }
                Some(f) if f.module != r.module => {
                    return Err(RegistryError::Reassigned {
                        tag,
                        listed: f.module,
                        module: r.module,
                    });
                }
                Some(_) => {}
            }
            for other in &self.entries[i + 1..] {
                if other.family.name() == tag {
                    return Err(RegistryError::DuplicateTag(tag));
                }
                if let Some(name) = r
                    .family
                    .accepts()
                    .iter()
                    .find(|n| other.family.accepts().contains(n))
                {
                    return Err(RegistryError::SharedArchitecture {
                        name,
                        tags: [tag, other.family.name()],
                    });
                }
            }
        }
        Ok(())
    }
}

/// The families this build registers: `llama` (COMP-E16-T03) and `qwen_moe` (COMP-E16-T04).
pub static REGISTRY: Registry = Registry::new(&[
    Registration {
        module: llama::MODULE,
        family: &llama::LLAMA,
    },
    Registration {
        module: qwen_moe::MODULE,
        family: &qwen_moe::QWEN_MOE,
    },
]);

/// The registry this build serves (RFC-0043 §1 `graph::families()`).
#[must_use]
pub fn families() -> &'static Registry {
    &REGISTRY
}

#[cfg(test)]
mod tests;
