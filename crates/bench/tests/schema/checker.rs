// SPDX-License-Identifier: MIT OR Apache-2.0

//! A checker for the subset of JSON Schema (draft 2020-12) the committed schema file uses.
//!
//! It exists so the tests can hold the schema file to what `BenchResult::from_json` accepts
//! without a schema-validation dependency. A keyword outside the subset is an error, so the
//! file cannot use a constraint these tests do not enforce.

use serde_json::{Map, Value};

/// Keywords that annotate and never constrain.
const ANNOTATIONS: &[&str] = &[
    "$schema",
    "$id",
    "$defs",
    "$comment",
    "title",
    "description",
];

/// Checks `instance` against `root`, the whole schema document.
pub fn check(root: &Value, instance: &Value) -> Result<(), String> {
    check_at(root, root, instance, "$")
}

fn check_at(root: &Value, schema: &Value, instance: &Value, at: &str) -> Result<(), String> {
    let schema = match schema {
        Value::Bool(true) => return Ok(()),
        Value::Bool(false) => return Err(format!("{at}: no value is allowed here")),
        Value::Object(map) => map,
        other => return Err(format!("{at}: schema is not an object: {other}")),
    };
    for (keyword, arg) in schema {
        match keyword.as_str() {
            k if ANNOTATIONS.contains(&k) => {}
            "$ref" => {
                let target = resolve(root, arg)?;
                check_at(root, target, instance, at)?;
            }
            "type" => check_type(arg, instance, at)?,
            "enum" => {
                let options = arg.as_array().ok_or("enum must be an array")?;
                if !options.contains(instance) {
                    return Err(format!("{at}: {instance} is not one of {arg}"));
                }
            }
            "const" => {
                if arg != instance {
                    return Err(format!("{at}: {instance} is not {arg}"));
                }
            }
            "anyOf" | "oneOf" => {
                let branches = arg.as_array().ok_or("anyOf/oneOf must be an array")?;
                let passed = branches
                    .iter()
                    .filter(|b| check_at(root, b, instance, at).is_ok())
                    .count();
                let ok = if keyword == "oneOf" {
                    passed == 1
                } else {
                    passed >= 1
                };
                if !ok {
                    return Err(format!(
                        "{at}: {passed} branches of {keyword} match {instance}"
                    ));
                }
            }
            "properties" | "required" | "additionalProperties" | "propertyNames" => {
                if let Value::Object(fields) = instance {
                    check_object(root, schema, keyword, fields, at)?;
                }
            }
            "items" => {
                if let Value::Array(items) = instance {
                    for (i, item) in items.iter().enumerate() {
                        check_at(root, arg, item, &format!("{at}[{i}]"))?;
                    }
                }
            }
            "uniqueItems" => {
                if let (Value::Array(items), Value::Bool(true)) = (instance, arg) {
                    for (i, item) in items.iter().enumerate() {
                        if items[..i].contains(item) {
                            return Err(format!("{at}: {item} is repeated"));
                        }
                    }
                }
            }
            "minimum" | "maximum" => {
                if let (Some(n), Some(bound)) = (instance.as_f64(), arg.as_f64()) {
                    let ok = if keyword == "minimum" {
                        n >= bound
                    } else {
                        n <= bound
                    };
                    if !ok {
                        return Err(format!("{at}: {instance} is outside {keyword} {arg}"));
                    }
                }
            }
            "pattern" => {
                if let Value::String(s) = instance {
                    check_pattern(arg.as_str().ok_or("pattern must be a string")?, s, at)?;
                }
            }
            other => return Err(format!("{at}: unsupported keyword {other}")),
        }
    }
    Ok(())
}

/// Object keywords; each call handles one keyword so every keyword is checked once.
fn check_object(
    root: &Value,
    schema: &Map<String, Value>,
    keyword: &str,
    fields: &Map<String, Value>,
    at: &str,
) -> Result<(), String> {
    let properties = schema.get("properties").and_then(Value::as_object);
    match keyword {
        "properties" => {
            for (name, sub) in properties.into_iter().flatten() {
                if let Some(value) = fields.get(name) {
                    check_at(root, sub, value, &format!("{at}.{name}"))?;
                }
            }
        }
        "required" => {
            for name in schema["required"]
                .as_array()
                .ok_or("required must be an array")?
            {
                let name = name.as_str().ok_or("required names must be strings")?;
                if !fields.contains_key(name) {
                    return Err(format!("{at}: missing required property {name}"));
                }
            }
        }
        "additionalProperties" => {
            for (name, value) in fields {
                if properties.is_some_and(|p| p.contains_key(name)) {
                    continue;
                }
                check_at(
                    root,
                    &schema["additionalProperties"],
                    value,
                    &format!("{at}.{name}"),
                )?;
            }
        }
        "propertyNames" => {
            for name in fields.keys() {
                check_at(
                    root,
                    &schema["propertyNames"],
                    &Value::String(name.clone()),
                    &format!("{at} key {name}"),
                )?;
            }
        }
        _ => unreachable!(),
    }
    Ok(())
}

fn check_type(arg: &Value, instance: &Value, at: &str) -> Result<(), String> {
    let names: Vec<&str> = match arg {
        Value::String(s) => vec![s.as_str()],
        Value::Array(list) => list.iter().filter_map(Value::as_str).collect(),
        _ => return Err(format!("{at}: type must be a string or an array")),
    };
    let ok = names.iter().any(|name| match *name {
        "null" => instance.is_null(),
        "boolean" => instance.is_boolean(),
        "string" => instance.is_string(),
        "array" => instance.is_array(),
        "object" => instance.is_object(),
        "number" => instance.is_number(),
        // serde reads an integer field only from an integer literal, never from `1.0`.
        "integer" => instance.is_u64() || instance.is_i64(),
        _ => false,
    });
    if ok {
        Ok(())
    } else {
        Err(format!("{at}: {instance} is not of type {arg}"))
    }
}

/// Supports only `^[0-9a-f]{N}$`, the digest form.
fn check_pattern(pattern: &str, s: &str, at: &str) -> Result<(), String> {
    let len: usize = pattern
        .strip_prefix("^[0-9a-f]{")
        .and_then(|rest| rest.strip_suffix("}$"))
        .and_then(|n| n.parse().ok())
        .ok_or_else(|| format!("{at}: unsupported pattern {pattern}"))?;
    let ok = s.len() == len
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
    if ok {
        Ok(())
    } else {
        Err(format!("{at}: {s:?} does not match {pattern}"))
    }
}

fn resolve<'a>(root: &'a Value, reference: &Value) -> Result<&'a Value, String> {
    let name = reference
        .as_str()
        .and_then(|r| r.strip_prefix("#/$defs/"))
        .ok_or_else(|| format!("unsupported $ref {reference}"))?;
    root["$defs"]
        .get(name)
        .ok_or_else(|| format!("$ref to missing definition {name}"))
}
