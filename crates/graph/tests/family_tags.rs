// SPDX-License-Identifier: MIT OR Apache-2.0

//! The frozen family tag list (COMP-E16-T01; RFC-0043 §5.1). Tags are hashed into `ModelId`
//! (RFC-0041 §1), so a tag is never renamed, removed, reordered or bound to another module.

use syntropy_graph::families::{FROZEN_TAGS, families};

/// Every tag as RFC-0043 §5.5 names its module, in the order assigned, with the module path it
/// registers from. A removed, renamed or reassigned tag fails `the_tag_list_is_frozen`.
/// Adding a tag means appending it here and to `FROZEN_TAGS`; nothing else may change.
const PINNED: &[(&str, &str)] = &[
    ("llama", "syntropy_graph::families::llama"),
    ("gemma", "syntropy_graph::families::gemma"),
    ("gptoss", "syntropy_graph::families::gptoss"),
    ("deepseek3", "syntropy_graph::families::deepseek3"),
    ("deepseek4", "syntropy_graph::families::deepseek4"),
    ("kimi3", "syntropy_graph::families::kimi3"),
    ("qwen_moe", "syntropy_graph::families::qwen_moe"),
    ("qwen_hybrid", "syntropy_graph::families::qwen_hybrid"),
    ("glm", "syntropy_graph::families::glm"),
    ("bert", "syntropy_graph::families::bert"),
];

#[test]
fn the_tag_list_is_frozen() {
    for (i, &(tag, module)) in PINNED.iter().enumerate() {
        let Some(entry) = FROZEN_TAGS.get(i) else {
            panic!("frozen tag `{tag}` (entry {i}) disappeared");
        };
        assert_eq!(
            (entry.tag, entry.module),
            (tag, module),
            "frozen entry {i} was `{tag}` bound to `{module}`; tags are never renamed, removed, reordered or reassigned"
        );
    }
    assert_eq!(
        FROZEN_TAGS.len(),
        PINNED.len(),
        "a tag was appended to FROZEN_TAGS without pinning it here"
    );
}

#[test]
fn the_registry_keeps_to_the_tag_list() {
    families()
        .check(FROZEN_TAGS)
        .unwrap_or_else(|e| panic!("{e}"));
    for tag in families().tags() {
        assert!(
            PINNED.iter().any(|&(t, _)| t == tag),
            "`{tag}` is not pinned"
        );
    }
}
