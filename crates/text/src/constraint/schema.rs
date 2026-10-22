// SPDX-License-Identifier: MIT OR Apache-2.0

//! RFC-0050 §7.2's JSON Schema coverage, checked before the schema reaches the engine.
//!
//! Every keyword falls in one of three sets:
//!
//! - **supported**, the §7.2 list, kept and passed on;
//! - **annotations** such as `title` and `description`, which constrain nothing and are kept,
//!   apart from `deprecated` and vendor extensions (`x-...`), which are removed silently because
//!   the engine does not know them;
//! - everything else, **unsupported**: under `strict` the first one fails the compile with
//!   `constraint_unsupported`, and otherwise each is removed and named in a warning.
//!
//! Two keywords are supported only in part. `minimum` and `maximum` are supported for integer
//! schemas only, and `$ref` only as a document-local reference (`#` or `#/...`); either kind of use
//! outside that counts as unsupported. `format` is supported for the §7.2 formats only. `oneOf` is
//! treated as `anyOf`, as §7.2 says, and `definitions`, the draft-07 name for `$defs`, counts as
//! `$defs`. The engine's own option key, `x-guidance`, is unsupported, so a request cannot change
//! how its schema compiles.

use serde_json::{Map, Value};

use super::{ConstraintError, ConstraintWarning};

/// Keywords whose value is a map of subschemas.
const SCHEMA_MAPS: &[&str] = &["properties", "$defs", "definitions"];

/// Keywords whose value is an array of subschemas.
const SCHEMA_LISTS: &[&str] = &["prefixItems", "anyOf", "oneOf", "allOf"];

/// Keywords whose value is a subschema, or a boolean schema.
const SCHEMA_ONES: &[&str] = &["items", "additionalProperties"];

/// The other supported keywords, whose values are data.
const DATA: &[&str] = &[
    "type",
    "required",
    "minItems",
    "maxItems",
    "enum",
    "const",
    "pattern",
    "minLength",
    "maxLength",
];

/// Keywords that constrain nothing.
const ANNOTATIONS: &[&str] = &[
    "$schema",
    "$id",
    "$comment",
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
];

/// Annotations removed before the engine sees the schema: the engine fails on keywords it does
/// not know, and these constrain nothing. Vendor extensions (`x-...`) go the same way.
const DROPPED: &[&str] = &["deprecated"];

/// The engine's own option key, which a request may not set.
const ENGINE_OPTIONS: &str = "x-guidance";

/// RFC-0050 §7.2's formats.
const FORMATS: &[&str] = &["date-time", "date", "time", "uuid", "email", "ipv4", "ipv6"];

/// A schema checked against the coverage, ready for the engine.
#[derive(Debug)]
pub(crate) struct Checked {
    pub schema: Value,
    pub warnings: Vec<ConstraintWarning>,
}

/// Checks `schema` against RFC-0050 §7.2's coverage. Under `strict`, the first unsupported keyword
/// is an error; otherwise each is removed and warned about.
pub(crate) fn check(schema: &Value, strict: bool) -> Result<Checked, ConstraintError> {
    match schema {
        Value::Object(_) => {}
        Value::Bool(true) => {
            return Ok(Checked {
                schema: Value::Object(Map::new()),
                warnings: Vec::new(),
            });
        }
        Value::Bool(false) => {
            return Err(ConstraintError::Invalid {
                reason: "the schema `false` accepts no value".to_owned(),
            });
        }
        other => {
            return Err(ConstraintError::Invalid {
                reason: format!("a schema is an object or a boolean, not {}", kind(other)),
            });
        }
    }
    let mut walk = Walk {
        strict,
        warnings: Vec::new(),
    };
    let schema = walk.schema(schema, "")?;
    Ok(Checked {
        schema,
        warnings: walk.warnings,
    })
}

struct Walk {
    strict: bool,
    warnings: Vec<ConstraintWarning>,
}

impl Walk {
    /// One subschema at `at`, with its unsupported keywords removed.
    fn schema(&mut self, schema: &Value, at: &str) -> Result<Value, ConstraintError> {
        let Value::Object(object) = schema else {
            return match schema {
                Value::Bool(_) => Ok(schema.clone()),
                other => Err(ConstraintError::Invalid {
                    reason: format!(
                        "the schema at {} is {}, not an object or a boolean",
                        shown(at),
                        kind(other)
                    ),
                }),
            };
        };
        let mut kept = Map::new();
        for (key, value) in object {
            let here = format!("{at}/{}", escape(key));
            let key = key.as_str();
            if let Some(reason) = unsupported_use(key, value, object) {
                self.unsupported(reason, at)?;
                continue;
            }
            if DROPPED.contains(&key) || (key.starts_with("x-") && key != ENGINE_OPTIONS) {
                continue;
            }
            let value = if SCHEMA_MAPS.contains(&key) {
                let Value::Object(members) = value else {
                    return Err(not_a(key, "an object of schemas", at));
                };
                let mut out = Map::new();
                for (name, member) in members {
                    out.insert(
                        name.clone(),
                        self.schema(member, &format!("{here}/{}", escape(name)))?,
                    );
                }
                Value::Object(out)
            } else if SCHEMA_LISTS.contains(&key) {
                let Value::Array(items) = value else {
                    return Err(not_a(key, "an array of schemas", at));
                };
                let mut out = Vec::with_capacity(items.len());
                for (index, item) in items.iter().enumerate() {
                    out.push(self.schema(item, &format!("{here}/{index}"))?);
                }
                Value::Array(out)
            } else if SCHEMA_ONES.contains(&key) {
                self.schema(value, &here)?
            } else {
                value.clone()
            };
            kept.insert(key.to_owned(), value);
        }
        Ok(Value::Object(kept))
    }

    /// An unsupported keyword at `at`: an error under `strict`, a warning otherwise.
    fn unsupported(&mut self, keyword: String, at: &str) -> Result<(), ConstraintError> {
        if self.strict {
            return Err(ConstraintError::Unsupported {
                keyword,
                at: at.to_owned(),
            });
        }
        self.warnings.push(ConstraintWarning {
            keyword,
            at: at.to_owned(),
        });
        Ok(())
    }
}

/// Why `key` is unsupported where it appears, named as a warning names it, or `None` when it is
/// supported there.
fn unsupported_use(key: &str, value: &Value, object: &Map<String, Value>) -> Option<String> {
    let known = SCHEMA_MAPS.contains(&key)
        || SCHEMA_LISTS.contains(&key)
        || SCHEMA_ONES.contains(&key)
        || DATA.contains(&key)
        || ANNOTATIONS.contains(&key);
    match key {
        "$ref" => match value.as_str() {
            Some(target) if target == "#" || target.starts_with("#/") => None,
            _ => Some("$ref".to_owned()),
        },
        "format" => match value.as_str() {
            Some(format) if FORMATS.contains(&format) => None,
            Some(format) => Some(format!("format: {format}")),
            None => Some("format".to_owned()),
        },
        ENGINE_OPTIONS => Some(key.to_owned()),
        _ if key.starts_with("x-") => None,
        "minimum" | "maximum" if integer_only(object) => None,
        "minimum" | "maximum" => Some(key.to_owned()),
        _ if known => None,
        _ => Some(key.to_owned()),
    }
}

/// Whether the schema's `type` is `integer`, alone or with `null`.
fn integer_only(object: &Map<String, Value>) -> bool {
    match object.get("type") {
        Some(Value::String(name)) => name == "integer",
        Some(Value::Array(names)) => {
            let names: Vec<&str> = names.iter().filter_map(Value::as_str).collect();
            names.contains(&"integer") && names.iter().all(|n| *n == "integer" || *n == "null")
        }
        _ => false,
    }
}

/// A JSON-pointer segment (RFC 6901): `~` as `~0`, `/` as `~1`.
fn escape(segment: &str) -> String {
    segment.replace('~', "~0").replace('/', "~1")
}

fn shown(at: &str) -> &str {
    if at.is_empty() { "/" } else { at }
}

fn not_a(key: &str, what: &str, at: &str) -> ConstraintError {
    ConstraintError::Invalid {
        reason: format!("`{key}` at {} is not {what}", shown(at)),
    }
}

fn kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn warnings(schema: Value) -> Vec<(String, String)> {
        check(&schema, false)
            .unwrap()
            .warnings
            .into_iter()
            .map(|w| (w.keyword, w.at))
            .collect()
    }

    fn strict_error(schema: Value) -> ConstraintError {
        check(&schema, true).unwrap_err()
    }

    #[test]
    fn the_supported_keywords_and_annotations_pass_strict_unchanged() {
        let schema = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Event",
            "description": "a calendar event",
            "type": "object",
            "properties": {
                "title": { "type": "string", "minLength": 1, "maxLength": 80 },
                "when": { "type": "string", "format": "date-time" },
                "tags": { "type": "array", "items": { "enum": ["a", "b"] }, "minItems": 0, "maxItems": 3 },
                "pair": { "type": "array", "prefixItems": [{ "type": "integer", "minimum": 0, "maximum": 9 }, { "const": "x" }] },
                "kind": { "oneOf": [{ "type": "null" }, { "$ref": "#/$defs/kind" }] },
                "both": { "allOf": [{ "type": "object" }, { "required": [] }] },
                "code": { "type": "string", "pattern": "^[A-Z]{3}$" },
                "count": { "type": ["integer", "null"], "minimum": 1 }
            },
            "required": ["title"],
            "additionalProperties": false,
            "$defs": { "kind": { "anyOf": [{ "const": "a" }, { "$ref": "#" }] } },
            "definitions": { "old": true },
            "examples": [{ "title": "x", "patternProperties": 1 }]
        });
        let checked = check(&schema, true).unwrap();
        assert_eq!(checked.schema, schema);
        assert_eq!(checked.warnings, []);
    }

    #[test]
    fn strict_names_the_first_unsupported_keyword_and_where_it_is() {
        let schema = json!({ "type": "object", "properties": { "a/b": { "type": "string", "patternProperties": {} } } });
        assert_eq!(
            strict_error(schema),
            ConstraintError::Unsupported {
                keyword: "patternProperties".into(),
                at: "/properties/a~1b".into()
            }
        );
        for (schema, keyword) in [
            (json!({ "not": { "type": "string" } }), "not"),
            (
                json!({ "type": "array", "uniqueItems": true }),
                "uniqueItems",
            ),
            (
                json!({ "type": "string", "format": "hostname" }),
                "format: hostname",
            ),
            (json!({ "type": "number", "minimum": 0 }), "minimum"),
            (json!({ "type": "integer", "multipleOf": 2 }), "multipleOf"),
            (json!({ "$ref": "https://example.org/schema.json" }), "$ref"),
            (json!({ "$ref": "other.json#/a" }), "$ref"),
            (json!({ "x-guidance": { "lenient": true } }), "x-guidance"),
        ] {
            match strict_error(schema) {
                ConstraintError::Unsupported { keyword: got, .. } => assert_eq!(got, keyword),
                other => panic!("{keyword}: {other:?}"),
            }
        }
    }

    #[test]
    fn without_strict_unsupported_keywords_are_removed_and_warned_about() {
        let schema = json!({
            "type": "object",
            "properties": {
                "a": { "type": "string", "format": "hostname", "maxLength": 9 },
                "b": { "$ref": "https://example.org/b.json", "description": "remote" }
            },
            "patternProperties": { "^x": {} },
            "x-guidance": { "lenient": true }
        });
        let checked = check(&schema, false).unwrap();
        assert_eq!(
            checked.schema,
            json!({
                "type": "object",
                "properties": {
                    "a": { "type": "string", "maxLength": 9 },
                    "b": { "description": "remote" }
                }
            })
        );
        assert_eq!(
            warnings(schema),
            [
                ("format: hostname".to_owned(), "/properties/a".to_owned()),
                ("$ref".to_owned(), "/properties/b".to_owned()),
                ("patternProperties".to_owned(), String::new()),
                ("x-guidance".to_owned(), String::new()),
            ]
        );
    }

    #[test]
    fn annotations_the_engine_does_not_know_are_removed_silently() {
        let schema =
            json!({ "type": "string", "deprecated": true, "x-order": 3, "description": "d" });
        let checked = check(&schema, true).unwrap();
        assert_eq!(
            checked.schema,
            json!({ "type": "string", "description": "d" })
        );
        assert_eq!(checked.warnings, []);
    }

    #[test]
    fn data_values_are_not_read_as_schemas() {
        let schema =
            json!({ "enum": [{ "not": 1 }], "const": { "if": 2 }, "default": { "then": 3 } });
        assert_eq!(check(&schema, true).unwrap().warnings, []);
    }

    #[test]
    fn malformed_schemas_are_invalid() {
        for (schema, needle) in [
            (json!(false), "accepts no value"),
            (json!(3), "not a number"),
            (
                json!({ "properties": [] }),
                "`properties` at / is not an object of schemas",
            ),
            (
                json!({ "anyOf": {} }),
                "`anyOf` at / is not an array of schemas",
            ),
            (json!({ "items": 1 }), "the schema at /items is a number"),
        ] {
            match check(&schema, false) {
                Err(ConstraintError::Invalid { reason }) => {
                    assert!(reason.contains(needle), "{needle}: {reason}");
                }
                other => panic!("{schema}: {other:?}"),
            }
        }
        assert_eq!(check(&json!(true), true).unwrap().schema, json!({}));
    }
}
