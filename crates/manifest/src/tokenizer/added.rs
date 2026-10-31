// SPDX-License-Identifier: MIT OR Apache-2.0

//! The added tokens of a [`TokenizerSpec`]'s model, which `ChatFormat` resolves its markers
//! against (RFC-0043 §8.4, §8.7; COMP-E17-T04).
//!
//! An added token is one the engine splits out of the text before the model's own algorithm
//! runs, so it always encodes to its single ID. That holds whether or not `tokenizer.json`
//! marks it `special`: the Qwen3 repositories ship `<tool_call>` and `<think>` as added but
//! not special. [`AddedToken::special`] keeps the flag as shipped.

use super::{TOKENIZER_JSON, TokenizerSpec, TokenizerSpecError, get, object};
use crate::safetensors::json::Json;

/// One entry of `tokenizer.json`'s `added_tokens`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AddedToken {
    /// Its text.
    pub content: String,
    /// Its ID.
    pub id: u32,
    /// The `special` flag as shipped; false when absent.
    pub special: bool,
}

impl TokenizerSpec {
    /// The added tokens of [`model`](Self::model), in the order `tokenizer.json` lists them.
    ///
    /// # Errors
    ///
    /// [`TokenizerSpecError`] when `model` is not a JSON object, or an entry of its
    /// `added_tokens` lacks an `id` or a `content`, or has a `special` that is not a boolean.
    pub fn added_tokens(&self) -> Result<Vec<AddedToken>, TokenizerSpecError> {
        let bad = |what: String| TokenizerSpecError::new(TOKENIZER_JSON, what);
        let tok = object(TOKENIZER_JSON, self.model())?;
        let items = match get(&tok, "added_tokens") {
            None | Some(Json::Null) => return Ok(Vec::new()),
            Some(Json::Array(items)) => items,
            Some(other) => return Err(bad(format!("added_tokens is {}", other.kind()))),
        };
        items
            .iter()
            .map(|item| {
                let Json::Object(o) = item else {
                    return Err(bad(format!("an added token is {}", item.kind())));
                };
                let (Some(Json::Uint(id)), Some(Json::Str(content))) =
                    (get(o, "id"), get(o, "content"))
                else {
                    return Err(bad("an added token lacks id or content".into()));
                };
                let id = u32::try_from(*id)
                    .map_err(|_| bad(format!("added token id {id} passes u32")))?;
                let special = match get(o, "special") {
                    None | Some(Json::Null) => false,
                    Some(Json::Bool(b)) => *b,
                    Some(other) => {
                        return Err(bad(format!(
                            "added token {content:?} has special {}, not a boolean",
                            other.kind()
                        )));
                    }
                };
                Ok(AddedToken {
                    content: content.clone(),
                    id,
                    special,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{SpecialTokens, TemplateSource};

    fn spec(model: &str) -> TokenizerSpec {
        TokenizerSpec::new(
            model.as_bytes().to_vec(),
            SpecialTokens::default(),
            false,
            false,
            None,
            TemplateSource::None,
        )
    }

    #[test]
    fn added_tokens_are_read_in_order_with_their_flag() {
        let s = spec(
            r#"{"added_tokens": [
                {"id": 5, "content": "<|im_end|>", "special": true},
                {"id": 3, "content": "<think>", "special": false},
                {"id": 4, "content": "<x>"}
            ]}"#,
        );
        assert_eq!(
            s.added_tokens().unwrap(),
            [
                AddedToken {
                    content: "<|im_end|>".into(),
                    id: 5,
                    special: true
                },
                AddedToken {
                    content: "<think>".into(),
                    id: 3,
                    special: false
                },
                AddedToken {
                    content: "<x>".into(),
                    id: 4,
                    special: false
                },
            ]
        );
        assert_eq!(spec("{}").added_tokens().unwrap(), []);
        assert_eq!(
            spec(r#"{"added_tokens": null}"#).added_tokens().unwrap(),
            []
        );
    }

    #[test]
    fn malformed_added_tokens_are_refused() {
        for model in [
            "[]",
            "not json",
            r#"{"added_tokens": 1}"#,
            r#"{"added_tokens": [1]}"#,
            r#"{"added_tokens": [{"id": 1}]}"#,
            r#"{"added_tokens": [{"id": 4294967296, "content": "a"}]}"#,
            r#"{"added_tokens": [{"id": 1, "content": "a", "special": "yes"}]}"#,
        ] {
            let e = spec(model).added_tokens().unwrap_err();
            assert_eq!(e.file, TOKENIZER_JSON, "{model}");
        }
    }
}
