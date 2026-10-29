// SPDX-License-Identifier: MIT OR Apache-2.0

//! The allow-list: which fields of each record type a bundle copies (RFC-0053 §6).
//!
//! A record is serialised to JSON, then [`project`]ed through its [`Shape`]. A [`Shape::Fields`]
//! keeps only the fields it names, so a field a later change adds to a record type is left out
//! until someone adds it here deliberately, and an enum's variant not named by a
//! [`Shape::Variants`] is left out the same way. The names, never the values, of what was left
//! out are listed in `MANIFEST.txt`. Every string that is kept passes the address redactor, and a
//! [`Shape::Secret`] field is replaced by `[redacted]` whatever it holds.

use std::collections::BTreeSet;

use serde_json::{Map, Value};

use super::redact::Redactor;

/// What of a JSON value is kept.
#[derive(Debug)]
pub(crate) enum Shape {
    /// The whole value, every string in it passed through the address redactor. Only for plain
    /// data that can hold no secret and no content: identifiers, counts, enums.
    Any,
    /// Replaced by `"[redacted]"`, whatever it holds.
    Secret,
    /// An object: only these fields, each by its shape.
    Fields(&'static [(&'static str, Shape)]),
    /// An externally tagged enum: only these variants, each payload by its shape. A unit variant
    /// is its name as a string.
    Variants(&'static [(&'static str, Shape)]),
    /// An array, every element by the shape.
    Seq(&'static Shape),
    /// An array of fixed length, each element by its shape.
    Tuple(&'static [Shape]),
    /// An object with any keys (redacted as text), every value by the shape.
    Map(&'static Shape),
    /// `null`, or the value by the shape.
    Option(&'static Shape),
}

/// What [`project`] records besides the value: the places left out, and the secrets replaced.
#[derive(Debug, Default)]
pub(crate) struct Omitted {
    /// `file: path` of every field or variant left out, values never named.
    pub(crate) fields: BTreeSet<String>,
    /// `file: path` of every field replaced by `[redacted]`.
    pub(crate) secrets: BTreeSet<String>,
}

/// `value` through `shape`. `file` and `path` name where it is, for the records of redactions
/// and omissions; a map key or array index is written `*`.
pub(crate) fn project(
    value: Value,
    shape: &Shape,
    file: &str,
    path: &str,
    redactor: &mut Redactor,
    omitted: &mut Omitted,
) -> Value {
    let at = || format!("{file}: {}", if path.is_empty() { "." } else { path });
    let join = |name: &str| {
        if path.is_empty() {
            name.to_owned()
        } else {
            format!("{path}.{name}")
        }
    };
    match shape {
        Shape::Any => redact_all(value, &at(), redactor),
        Shape::Secret => {
            omitted.secrets.insert(at());
            Value::String("[redacted]".to_owned())
        }
        Shape::Option(inner) => match value {
            Value::Null => Value::Null,
            other => project(other, inner, file, path, redactor, omitted),
        },
        Shape::Fields(fields) => {
            let Value::Object(object) = value else {
                omitted.fields.insert(format!("{} (not an object)", at()));
                return Value::Null;
            };
            let mut out = Map::new();
            for (name, value) in object {
                match fields.iter().find(|(f, _)| *f == name) {
                    Some((_, shape)) => {
                        let kept = project(value, shape, file, &join(&name), redactor, omitted);
                        out.insert(name, kept);
                    }
                    None => {
                        omitted.fields.insert(format!("{file}: {}", join(&name)));
                    }
                }
            }
            Value::Object(out)
        }
        Shape::Variants(variants) => {
            let (name, payload) = match value {
                Value::String(name) => (name, None),
                Value::Object(object) if object.len() == 1 => {
                    let (name, payload) = object.into_iter().next().expect("one entry");
                    (name, Some(payload))
                }
                _ => {
                    omitted.fields.insert(format!("{} (not an enum)", at()));
                    return Value::Null;
                }
            };
            match variants.iter().find(|(v, _)| *v == name) {
                Some((_, shape)) => match payload {
                    None => Value::String(name),
                    Some(payload) => {
                        let kept = project(payload, shape, file, &join(&name), redactor, omitted);
                        let mut out = Map::new();
                        out.insert(name, kept);
                        Value::Object(out)
                    }
                },
                None => {
                    omitted.fields.insert(format!("{file}: {}", join(&name)));
                    Value::String("[left out]".to_owned())
                }
            }
        }
        Shape::Seq(inner) => match value {
            Value::Array(items) => Value::Array(
                items
                    .into_iter()
                    .map(|v| project(v, inner, file, &join("*"), redactor, omitted))
                    .collect(),
            ),
            _ => {
                omitted.fields.insert(format!("{} (not an array)", at()));
                Value::Null
            }
        },
        Shape::Tuple(shapes) => match value {
            Value::Array(items) if items.len() == shapes.len() => Value::Array(
                items
                    .into_iter()
                    .zip(shapes.iter())
                    .enumerate()
                    .map(|(i, (v, s))| {
                        project(v, s, file, &join(&i.to_string()), redactor, omitted)
                    })
                    .collect(),
            ),
            _ => {
                omitted.fields.insert(format!("{} (not a tuple)", at()));
                Value::Null
            }
        },
        Shape::Map(inner) => match value {
            Value::Object(object) => {
                let mut out = Map::new();
                for (key, value) in object {
                    let key = redactor.scan(&key, &format!("{file}: {} (key)", join("*")));
                    out.insert(
                        key,
                        project(value, inner, file, &join("*"), redactor, omitted),
                    );
                }
                Value::Object(out)
            }
            _ => {
                omitted.fields.insert(format!("{} (not a map)", at()));
                Value::Null
            }
        },
    }
}

/// `value` with every string, and every object key, passed through the redactor.
fn redact_all(value: Value, at: &str, redactor: &mut Redactor) -> Value {
    match value {
        Value::String(s) => Value::String(redactor.scan(&s, at)),
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .map(|v| redact_all(v, at, redactor))
                .collect(),
        ),
        Value::Object(object) => Value::Object(
            object
                .into_iter()
                .map(|(k, v)| (redactor.scan(&k, at), redact_all(v, at, redactor)))
                .collect(),
        ),
        other => other,
    }
}

// ---- The record types --------------------------------------------------------------------------

/// `api::version::VersionInfo`: the build, its protocol majors and the support tier.
pub(crate) const BUILD: Shape = Shape::Fields(&[
    ("build", Shape::Any),
    ("protocols", Shape::Any),
    ("support_tier", Shape::Any),
]);

/// A plan held in the log: its summary, and its body inline or by digest.
const COMMITTED_PLAN: Shape = Shape::Fields(&[
    (
        "body",
        Shape::Variants(&[("Inline", Shape::Any), ("ByDigest", Shape::Any)]),
    ),
    ("summary", Shape::Any),
]);

/// A member: its description (the node ID so far), status and incarnation. RFC-0030's
/// certificates and approvals and DIST-E9-T03's roles join here when they are added deliberately.
const MEMBER: Shape = Shape::Fields(&[
    ("info", Shape::Fields(&[("id", Shape::Any)])),
    ("status", Shape::Any),
    ("incarnation", Shape::Any),
]);

/// `cluster::log::ClusterState`, the whole-state view of ADR-0009. A payload the log keeps as
/// `Opaque`, a CBOR value whose shape no RFC fixes yet, is left out wherever it occurs, here and
/// in the events, since nothing says what it may hold. Configuration values are
/// replaced: a cluster key may be secret-typed, and `config.txt` shows them with secrets
/// redacted. RFC-0032 §6.2's `api_keys`, `revoked_api_keys`, `invites` and `pairing`, which
/// DIST-E9-T16 adds, are not on this list and must never be.
pub(crate) const CLUSTER_STATE: Shape = Shape::Fields(&[
    ("cluster", Shape::Any),
    ("applied", Shape::Any),
    ("chain", Shape::Any),
    ("feature_level", Shape::Any),
    ("coordinator", Shape::Any),
    ("max_term", Shape::Any),
    ("members", Shape::Map(&MEMBER)),
    ("graph", Shape::Any),
    // `workload` is left out: its parts are `Opaque` until their RFCs define them.
    (
        "models",
        Shape::Map(&Shape::Fields(&[("manifest", Shape::Any)])),
    ),
    (
        "instances",
        Shape::Map(&Shape::Fields(&[
            ("model", Shape::Any),
            ("policy", Shape::Any),
            ("current", Shape::Option(&COMMITTED_PLAN)),
            ("retiring", Shape::Seq(&COMMITTED_PLAN)),
            ("runners", Shape::Any),
            ("routable", Shape::Any),
            ("requested_at", Shape::Any),
        ])),
    ),
    (
        "sessions",
        Shape::Map(&Shape::Fields(&[
            ("instance", Shape::Any),
            ("holders", Shape::Any),
        ])),
    ),
    // `chunks` is left out: `ChunkSetDigest` is `Opaque` until RFC-0041 fixes its form.
    (
        "config",
        Shape::Fields(&[(
            "keys",
            Shape::Map(&Shape::Fields(&[
                ("value", Shape::Secret),
                ("set_at", Shape::Any),
            ])),
        )]),
    ),
    ("alloc", Shape::Any),
]);

/// `cluster::log::ClusterEvent`: the variants of the base events. A variant added later, such as
/// DIST-E9-T03's admission and invite events, is left out until it is added here.
const CLUSTER_EVENT: Shape = Shape::Variants(&[
    (
        "NodeAdmitted",
        Shape::Fields(&[("info", Shape::Fields(&[("id", Shape::Any)]))]),
    ),
    ("NodeSuspected", Shape::Any),
    ("NodeFailed", Shape::Any),
    ("NodeLeft", Shape::Any),
    ("NodeSleeping", Shape::Any),
    ("NodeRestarting", Shape::Any),
    ("NodeRevoked", Shape::Any),
    (
        "NodeInfoChanged",
        Shape::Fields(&[("info", Shape::Fields(&[("id", Shape::Any)]))]),
    ),
    ("CoordinatorElected", Shape::Any),
    ("GraphUpdated", Shape::Any),
    ("ModelRegistered", Shape::Any),
    ("InstanceRequested", Shape::Any),
    ("PlanCommitted", Shape::Any),
    ("RunnerStateChanged", Shape::Any),
    ("InstanceRoutable", Shape::Any),
    (
        "InstanceUnroutable",
        Shape::Fields(&[("instance", Shape::Any)]),
    ),
    ("PlanRetired", Shape::Any),
    ("InstanceRemoved", Shape::Any),
    (
        "WorkloadUpdated",
        Shape::Fields(&[("model", Shape::Any), ("instance", Shape::Any)]),
    ),
    (
        "ChunksAnnounced",
        Shape::Fields(&[("node", Shape::Any), ("model", Shape::Any)]),
    ),
    ("SessionPinned", Shape::Any),
    ("SessionDropped", Shape::Any),
    (
        "ClusterConfigChanged",
        Shape::Fields(&[(
            "patch",
            Shape::Fields(&[
                (
                    "set",
                    Shape::Seq(&Shape::Tuple(&[Shape::Any, Shape::Secret])),
                ),
                ("unset", Shape::Any),
            ]),
        )]),
    ),
]);

/// One cluster log entry, as [`super::log_entry_value`] writes it.
pub(crate) const CLUSTER_LOG_ENTRY: Shape = Shape::Fields(&[
    ("index", Shape::Any),
    ("term", Shape::Any),
    ("at_ms", Shape::Any),
    ("cause", Shape::Any),
    ("event", CLUSTER_EVENT),
]);

/// `doctor::Report`. Findings' evidence is text a check wrote, which the redactor scans.
pub(crate) const DOCTOR_REPORT: Shape = Shape::Fields(&[
    ("schema", Shape::Any),
    ("offline", Shape::Any),
    (
        "results",
        Shape::Seq(&Shape::Fields(&[
            ("id", Shape::Any),
            ("area", Shape::Any),
            ("needs", Shape::Any),
            ("status", Shape::Any),
            ("why", Shape::Any),
            ("summary", Shape::Any),
            ("evidence", Shape::Any),
            ("fix", Shape::Any),
        ])),
    ),
]);

/// `crash::CrashRecord`.
pub(crate) const CRASH_RECORD: Shape = Shape::Fields(&[
    ("version", Shape::Any),
    ("at_unix_ms", Shape::Any),
    ("build_id", Shape::Any),
    ("process", Shape::Any),
    ("cause", Shape::Any),
    ("last_logs", Shape::Any),
]);

/// One flight-recorder entry, as [`super::flight_value`] writes it.
pub(crate) const FLIGHT_ENTRY: Shape = Shape::Variants(&[
    (
        "span",
        Shape::Fields(&[
            ("trace_id", Shape::Any),
            ("span_id", Shape::Any),
            ("parent_id", Shape::Any),
            ("start_ns", Shape::Any),
            ("dur_ns", Shape::Any),
            ("name", Shape::Any),
            ("attrs", Shape::Any),
        ]),
    ),
    (
        "log",
        Shape::Fields(&[
            ("at_ns", Shape::Any),
            ("level", Shape::Any),
            ("callsite", Shape::Any),
            ("target", Shape::Any),
            ("message", Shape::Any),
            ("fields", Shape::Any),
        ]),
    ),
]);

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn run(value: Value, shape: &Shape) -> (Value, Omitted, Redactor) {
        let mut r = Redactor::new(false);
        let mut o = Omitted::default();
        let out = project(value, shape, "f", "", &mut r, &mut o);
        (out, o, r)
    }

    #[test]
    fn a_field_not_on_the_list_is_left_out_and_named() {
        const S: Shape = Shape::Fields(&[
            ("a", Shape::Any),
            ("b", Shape::Fields(&[("c", Shape::Any)])),
        ]);
        let (out, omitted, _) = run(
            json!({"a": 1, "b": {"c": 2, "token": "SECRET"}, "invites": ["SECRET"]}),
            &S,
        );
        assert_eq!(out, json!({"a": 1, "b": {"c": 2}}));
        assert_eq!(
            omitted.fields.iter().collect::<Vec<_>>(),
            ["f: b.token", "f: invites"]
        );
        assert!(!out.to_string().contains("SECRET"));
    }

    #[test]
    fn a_variant_not_on_the_list_is_left_out() {
        const S: Shape = Shape::Seq(&Shape::Variants(&[
            ("Known", Shape::Any),
            ("Unit", Shape::Any),
        ]));
        let (out, omitted, _) = run(
            json!([{"Known": {"x": 1}}, {"InviteIssued": {"token": "SECRET"}}, "Unit", "Other"]),
            &S,
        );
        assert_eq!(
            out,
            json!([{"Known": {"x": 1}}, "[left out]", "Unit", "[left out]"])
        );
        assert_eq!(
            omitted.fields.iter().collect::<Vec<_>>(),
            ["f: *.InviteIssued", "f: *.Other"]
        );
    }

    #[test]
    fn secrets_are_replaced_and_addresses_redacted() {
        const S: Shape = Shape::Fields(&[
            ("k", Shape::Secret),
            ("note", Shape::Any),
            ("m", Shape::Map(&Shape::Any)),
        ]);
        let (out, _, r) = run(
            json!({"k": {"String": "syk_SECRET"}, "note": "via 203.0.113.9:1",
                   "m": {"198.51.100.1": "x"}}),
            &S,
        );
        assert_eq!(
            out,
            json!({"k": "[redacted]", "note": "via <addr-1>", "m": {"<addr-2>": "x"}})
        );
        assert_eq!(
            r.fields.iter().collect::<Vec<_>>(),
            ["f: m.* (key)", "f: note"]
        );
    }
}
