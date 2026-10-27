// SPDX-License-Identifier: MIT OR Apache-2.0

//! The checked-in family tag list, frozen (RFC-0043 §5.1).
//!
//! A family's tag is hashed into `ModelId` (RFC-0041 §1). A tag is therefore never renamed,
//! never removed and never reused for a different module. A module that changes what it
//! computes for the same files keeps its tag only if the change is a bug fix certified by the
//! harness; otherwise it registers a new tag. New tags are appended.
//!
//! The list holds the modules of RFC-0043 §5.5's v1 table, each bound to the module path it
//! registers from, so their tags are fixed before any `ModelId` is computed. A listed tag may
//! have no registered module yet; [`Registry::check`](super::Registry::check) binds every
//! registered family to its entry, and `tests/family_tags.rs` pins the list, so the test fails
//! if an entry disappears, is reordered or is bound to another module.

/// One frozen tag and the module it belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrozenTag {
    /// The tag, as [`Family::name`](super::Family::name) returns it.
    pub tag: &'static str,
    /// The module path its family registers from (`module_path!()`).
    pub module: &'static str,
}

const fn tag(tag: &'static str, module: &'static str) -> FrozenTag {
    FrozenTag { tag, module }
}

/// Every family tag, in the order assigned. Append only.
pub const FROZEN_TAGS: &[FrozenTag] = &[
    tag("llama", "syntropy_graph::families::llama"),
    tag("gemma", "syntropy_graph::families::gemma"),
    tag("gptoss", "syntropy_graph::families::gptoss"),
    tag("deepseek3", "syntropy_graph::families::deepseek3"),
    tag("deepseek4", "syntropy_graph::families::deepseek4"),
    tag("kimi3", "syntropy_graph::families::kimi3"),
    tag("qwen_moe", "syntropy_graph::families::qwen_moe"),
    tag("qwen_hybrid", "syntropy_graph::families::qwen_hybrid"),
    tag("glm", "syntropy_graph::families::glm"),
    tag("bert", "syntropy_graph::families::bert"),
];
