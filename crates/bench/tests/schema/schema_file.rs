// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structural rules for the committed schema file.

use serde_json::Value;

use crate::support::SCHEMA_FILE;

fn schema() -> Value {
    serde_json::from_str(&std::fs::read_to_string(SCHEMA_FILE).expect("read schema file"))
        .expect("schema is JSON")
}

/// Calls `visit` on every subschema, depth first.
fn walk<'a>(schema: &'a Value, path: &str, visit: &mut dyn FnMut(&str, &'a Value)) {
    visit(path, schema);
    match schema {
        Value::Object(map) => {
            for (k, v) in map {
                // `enum` and `const` hold values, not schemas.
                if k != "enum" && k != "const" {
                    walk(v, &format!("{path}/{k}"), visit);
                }
            }
        }
        Value::Array(items) => {
            for (i, v) in items.iter().enumerate() {
                walk(v, &format!("{path}/{i}"), visit);
            }
        }
        _ => {}
    }
}

/// Every record is closed and lists every property as required, as `from_json` reads it.
#[test]
fn every_record_is_closed_and_fully_required() {
    let schema = schema();
    let mut records = 0;
    walk(&schema, "#", &mut |path, node| {
        let Some(properties) = node.get("properties").and_then(Value::as_object) else {
            return;
        };
        if path.ends_with("/properties") {
            return; // a property named "properties", not a record
        }
        records += 1;
        let mut required: Vec<&str> = node["required"]
            .as_array()
            .expect("required")
            .iter()
            .filter_map(Value::as_str)
            .collect();
        let mut names: Vec<&str> = properties.keys().map(String::as_str).collect();
        required.sort_unstable();
        names.sort_unstable();
        assert_eq!(required, names, "{path}: required differs from properties");
        assert_eq!(
            node["additionalProperties"],
            Value::Bool(false),
            "{path}: record is open"
        );
    });
    assert!(records > 40, "found only {records} records");
}

/// Every `$ref` resolves, and every definition is used.
#[test]
fn every_reference_resolves_and_every_definition_is_used() {
    let schema = schema();
    let defs = schema["$defs"].as_object().expect("$defs");
    let mut used = std::collections::BTreeSet::new();
    walk(&schema, "#", &mut |path, node| {
        if let Some(r) = node.get("$ref").and_then(Value::as_str) {
            let name = r
                .strip_prefix("#/$defs/")
                .unwrap_or_else(|| panic!("{path}: non-local $ref {r}"));
            assert!(defs.contains_key(name), "{path}: $ref to missing {name}");
            used.insert(name.to_owned());
        }
    });
    let unused: Vec<&String> = defs.keys().filter(|k| !used.contains(*k)).collect();
    assert!(unused.is_empty(), "unused definitions: {unused:?}");
}
