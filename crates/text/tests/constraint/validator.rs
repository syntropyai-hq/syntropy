// SPDX-License-Identifier: MIT OR Apache-2.0

//! A JSON Schema validator for RFC-0050 §7.2's coverage, written for these tests.
//!
//! It shares no code with the engine, so the property test checks the engine against an
//! independent reading of the schema. It follows JSON Schema 2020-12 for the keywords it knows:
//! type-specific keywords apply only to values of that type, `pattern` is not anchored, `oneOf`
//! means exactly one, and `$ref` resolves as a JSON pointer into the document. Formats are checked
//! by their syntax and field ranges.

use std::net::{Ipv4Addr, Ipv6Addr};

use regex::Regex;
use serde_json::Value;

/// Whether `value` is valid under `schema`, with the first reason it is not.
pub fn validate(schema: &Value, value: &Value) -> Result<(), String> {
    Validator { root: schema }.check(schema, value, "")
}

struct Validator<'a> {
    root: &'a Value,
}

impl Validator<'_> {
    fn check(&self, schema: &Value, value: &Value, at: &str) -> Result<(), String> {
        let object = match schema {
            Value::Bool(true) => return Ok(()),
            Value::Bool(false) => return Err(format!("{at}: the schema `false`")),
            Value::Object(object) => object,
            other => return Err(format!("{at}: schema {other} is not a schema")),
        };
        if let Some(target) = object.get("$ref").and_then(Value::as_str) {
            let referred = self.resolve(target)?;
            self.check(referred, value, at)?;
        }
        if let Some(types) = object.get("type") {
            let names: Vec<&str> = match types {
                Value::String(name) => vec![name.as_str()],
                Value::Array(names) => names.iter().filter_map(Value::as_str).collect(),
                _ => return Err(format!("{at}: bad `type`")),
            };
            if !names.iter().any(|name| has_type(value, name)) {
                return Err(format!("{at}: {value} is not of type {names:?}"));
            }
        }
        if let Some(Value::Array(choices)) = object.get("enum")
            && !choices.iter().any(|choice| same(choice, value))
        {
            return Err(format!("{at}: {value} is not in the enum"));
        }
        if let Some(constant) = object.get("const")
            && !same(constant, value)
        {
            return Err(format!("{at}: {value} is not the const {constant}"));
        }
        for (key, all) in [("anyOf", false), ("oneOf", true)] {
            if let Some(Value::Array(options)) = object.get(key) {
                let passed = options
                    .iter()
                    .filter(|option| self.check(option, value, at).is_ok())
                    .count();
                let ok = if all { passed == 1 } else { passed >= 1 };
                if !ok {
                    return Err(format!(
                        "{at}: {value} passes {passed} of `{key}`'s options"
                    ));
                }
            }
        }
        if let Some(Value::Array(parts)) = object.get("allOf") {
            for part in parts {
                self.check(part, value, at)?;
            }
        }
        match value {
            Value::Object(members) => self.object(object, members, at)?,
            Value::Array(items) => self.array(object, items, at)?,
            Value::String(text) => string(object, text, at)?,
            Value::Number(_) => number(object, value, at)?,
            _ => {}
        }
        Ok(())
    }

    fn object(
        &self,
        schema: &serde_json::Map<String, Value>,
        members: &serde_json::Map<String, Value>,
        at: &str,
    ) -> Result<(), String> {
        let properties = schema.get("properties").and_then(Value::as_object);
        if let Some(Value::Array(required)) = schema.get("required") {
            for name in required.iter().filter_map(Value::as_str) {
                if !members.contains_key(name) {
                    return Err(format!("{at}: `{name}` is required"));
                }
            }
        }
        for (name, member) in members {
            let here = format!("{at}/{name}");
            match properties.and_then(|p| p.get(name)) {
                Some(property) => self.check(property, member, &here)?,
                None => {
                    if let Some(additional) = schema.get("additionalProperties") {
                        self.check(additional, member, &here)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn array(
        &self,
        schema: &serde_json::Map<String, Value>,
        items: &[Value],
        at: &str,
    ) -> Result<(), String> {
        let prefix = schema
            .get("prefixItems")
            .and_then(Value::as_array)
            .map_or(&[][..], Vec::as_slice);
        for (index, item) in items.iter().enumerate() {
            let here = format!("{at}/{index}");
            match prefix.get(index) {
                Some(part) => self.check(part, item, &here)?,
                None => {
                    if let Some(rest) = schema.get("items") {
                        self.check(rest, item, &here)?;
                    }
                }
            }
        }
        if let Some(min) = schema.get("minItems").and_then(Value::as_u64)
            && (items.len() as u64) < min
        {
            return Err(format!("{at}: {} items, under minItems {min}", items.len()));
        }
        if let Some(max) = schema.get("maxItems").and_then(Value::as_u64)
            && (items.len() as u64) > max
        {
            return Err(format!("{at}: {} items, over maxItems {max}", items.len()));
        }
        Ok(())
    }

    fn resolve(&self, target: &str) -> Result<&Value, String> {
        let pointer = target
            .strip_prefix('#')
            .ok_or_else(|| format!("`$ref` {target} is not document-local"))?;
        if pointer.is_empty() {
            return Ok(self.root);
        }
        self.root
            .pointer(pointer)
            .ok_or_else(|| format!("`$ref` {target} resolves to nothing"))
    }
}

fn has_type(value: &Value, name: &str) -> bool {
    match name {
        "null" => value.is_null(),
        "boolean" => value.is_boolean(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => {
            value.is_i64()
                || value.is_u64()
                || value
                    .as_f64()
                    .is_some_and(|f| f.is_finite() && f.fract() == 0.0)
        }
        _ => false,
    }
}

/// JSON equality, with numbers compared by value, so `1` and `1.0` are the same.
fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(a, b)| same(a, b))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| same(v, w)))
        }
        _ => a == b,
    }
}

fn string(schema: &serde_json::Map<String, Value>, text: &str, at: &str) -> Result<(), String> {
    let chars = text.chars().count() as u64;
    if let Some(min) = schema.get("minLength").and_then(Value::as_u64)
        && chars < min
    {
        return Err(format!("{at}: {chars} characters, under minLength {min}"));
    }
    if let Some(max) = schema.get("maxLength").and_then(Value::as_u64)
        && chars > max
    {
        return Err(format!("{at}: {chars} characters, over maxLength {max}"));
    }
    if let Some(pattern) = schema.get("pattern").and_then(Value::as_str) {
        let regex = Regex::new(pattern).map_err(|e| format!("{at}: bad pattern: {e}"))?;
        if !regex.is_match(text) {
            return Err(format!("{at}: {text:?} does not match {pattern}"));
        }
    }
    if let Some(format) = schema.get("format").and_then(Value::as_str)
        && !has_format(text, format)
    {
        return Err(format!("{at}: {text:?} is not a {format}"));
    }
    Ok(())
}

fn number(schema: &serde_json::Map<String, Value>, value: &Value, at: &str) -> Result<(), String> {
    let x = value.as_f64().unwrap_or(f64::NAN);
    if let Some(min) = schema.get("minimum").and_then(Value::as_f64)
        && x < min
    {
        return Err(format!("{at}: {value} is under minimum {min}"));
    }
    if let Some(max) = schema.get("maximum").and_then(Value::as_f64)
        && x > max
    {
        return Err(format!("{at}: {value} is over maximum {max}"));
    }
    Ok(())
}

fn has_format(text: &str, format: &str) -> bool {
    let matches = |pattern: &str| Regex::new(pattern).expect("a valid pattern").is_match(text);
    let date_ok = |caps: &regex::Captures<'_>| {
        let month: u32 = caps["m"].parse().unwrap_or(0);
        let day: u32 = caps["d"].parse().unwrap_or(0);
        (1..=12).contains(&month) && (1..=31).contains(&day)
    };
    let time_ok = |caps: &regex::Captures<'_>| {
        let hour: u32 = caps["h"].parse().unwrap_or(99);
        let minute: u32 = caps["mi"].parse().unwrap_or(99);
        let second: u32 = caps["s"].parse().unwrap_or(99);
        hour <= 23 && minute <= 59 && second <= 60
    };
    const DATE: &str = r"(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})";
    const TIME: &str = r"(?P<h>\d{2}):(?P<mi>\d{2}):(?P<s>\d{2})(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})";
    match format {
        "date" => Regex::new(&format!("^{DATE}$"))
            .unwrap()
            .captures(text)
            .is_some_and(|c| date_ok(&c)),
        "time" => Regex::new(&format!("^{TIME}$"))
            .unwrap()
            .captures(text)
            .is_some_and(|c| time_ok(&c)),
        "date-time" => Regex::new(&format!("^{DATE}[Tt]{TIME}$"))
            .unwrap()
            .captures(text)
            .is_some_and(|c| date_ok(&c) && time_ok(&c)),
        "uuid" => matches(
            r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
        ),
        "email" => matches(r"^[^@\s]+@[^@\s]+$"),
        "ipv4" => text.parse::<Ipv4Addr>().is_ok(),
        "ipv6" => text.parse::<Ipv6Addr>().is_ok(),
        _ => false,
    }
}

#[test]
fn the_validator_accepts_and_refuses_as_json_schema_does() {
    use serde_json::json;
    let schema = json!({
        "type": "object",
        "properties": {
            "n": { "type": "integer", "minimum": 1, "maximum": 3 },
            "s": { "type": "string", "pattern": "b", "maxLength": 3 },
            "k": { "oneOf": [{ "type": "string" }, { "type": "integer" }] },
            "t": { "$ref": "#/$defs/t" },
            "a": { "type": "array", "prefixItems": [{ "const": 1 }], "items": { "type": "string" }, "maxItems": 2 }
        },
        "required": ["n"],
        "additionalProperties": false,
        "$defs": { "t": { "type": "string", "format": "date-time" } }
    });
    let good = json!({ "n": 2.0, "s": "abc", "k": 3, "t": "2026-10-22T12:00:00Z", "a": [1, "x"] });
    assert_eq!(validate(&schema, &good), Ok(()));
    for bad in [
        json!({}),
        json!({ "n": 4 }),
        json!({ "n": 1.5 }),
        json!({ "n": 1, "s": "acd" }),
        json!({ "n": 1, "s": "abbb" }),
        json!({ "n": 1, "k": true }),
        json!({ "n": 1, "t": "2026-13-01T00:00:00Z" }),
        json!({ "n": 1, "a": [2] }),
        json!({ "n": 1, "a": [1, 2] }),
        json!({ "n": 1, "a": [1, "x", "y"] }),
        json!({ "n": 1, "extra": 0 }),
    ] {
        assert!(validate(&schema, &bad).is_err(), "{bad}");
    }
    let overlapping = json!({ "oneOf": [{ "type": "integer" }, { "type": "number" }] });
    assert!(
        validate(&overlapping, &json!(1)).is_err(),
        "oneOf is exactly one"
    );
    for (format, good, bad) in [
        ("date", "2026-02-28", "2026-2-28"),
        ("time", "23:59:60.5+02:00", "24:00:00Z"),
        ("uuid", "550e8400-e29b-41d4-a716-446655440000", "550e8400"),
        ("email", "a@example.com", "a@b@c"),
        ("ipv4", "192.168.0.1", "256.0.0.1"),
        ("ipv6", "fe80::1", "fe80:::1"),
    ] {
        assert!(has_format(good, format), "{format}: {good}");
        assert!(!has_format(bad, format), "{format}: {bad}");
    }
}
