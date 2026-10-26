// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tokenizers, chat templates, detokenization, stop sequences, tool-call and reasoning parsers, and
//! grammar token masks.
//!
//! Layer 2 in RFC-0001 §4.1, designed in RFC-0050. It holds the constraint engine so far, in
//! [`constraint`] (RFC-0050 §7), and the tokenizer engine, in [`tokenizer`] (RFC-0043 §8.2). The
//! rest is planned.

#![forbid(unsafe_code)]

pub mod constraint;
pub mod tokenizer;
