// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tokenizers, chat templates, detokenization, stop sequences, tool-call and reasoning parsers, and
//! grammar token masks.
//!
//! Layer 2 in RFC-0001 §4.1, designed in RFC-0050. It holds the constraint engine so far, in
//! [`constraint`] (RFC-0050 §7), and the tokenizer engine with span-aware literal encoding, in
//! [`tokenizer`] (RFC-0043 §8.2, §8.7), and chat-template rendering into template and content
//! spans, in [`template`] (RFC-0050 §6.1, §6.4). The rest is planned.

#![forbid(unsafe_code)]

pub mod constraint;
pub mod template;
pub mod tokenizer;
