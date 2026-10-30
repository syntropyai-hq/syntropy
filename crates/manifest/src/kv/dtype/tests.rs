// SPDX-License-Identifier: MIT OR Apache-2.0

//! RFC-0042 §4.2's KV dtype rule over every case, and the `kv_dtype` override (COMP-E13-T06).
//!
//! The per-family assertions, such as `llama`'s `KvFormula::dtype` per source, belong to the
//! family tasks that build each `kv_formula` (COMP-E16-T03, COMP-E16-T04).

use syntropy_config::{
    Code, ConfigSection, DefaultSpec, InstanceSection, InstanceSectionError, Registry, Reload,
    Scope, Value,
};

use super::*;

const GGUF: KvSource = KvSource::Gguf;

fn st(compute: Option<DType>, quantized: bool) -> KvSource {
    KvSource::Safetensors { compute, quantized }
}

fn rule(
    source: KvSource,
    declared: Option<DType>,
    family: Option<DType>,
    layers: &[(u16, DType)],
) -> KvDTypeRule {
    KvDTypeRule {
        source,
        declared_cache: declared,
        family,
        family_layers: layers.iter().map(|&(l, d)| (LayerIx(l), d)).collect(),
    }
}

fn uniform(d: DType) -> KvDTypeSpec {
    KvDTypeSpec::uniform(d)
}

fn spec(default: DType, layers: &[(u16, DType)]) -> KvDTypeSpec {
    KvDTypeSpec {
        default,
        layers: layers.iter().map(|&(l, d)| (LayerIx(l), d)).collect(),
    }
}

/// Every case of RFC-0042 §4.2, with the module docs' reading of the safetensors cases the RFC
/// leaves open.
#[test]
fn the_rule_gives_each_case_its_dtype() {
    use DType::{BF16, F8E4M3, F8E5M2, F16, F32};
    let cases: Vec<(&str, KvDTypeRule, KvDTypeSpec)> = vec![
        // 3. The source alone.
        (
            "GGUF: F16, ggml's default cache type",
            rule(GGUF, None, None, &[]),
            uniform(F16),
        ),
        (
            "BF16 safetensors",
            rule(st(Some(BF16), false), None, None, &[]),
            uniform(BF16),
        ),
        (
            "quantised safetensors, BF16 compute",
            rule(st(Some(BF16), true), None, None, &[]),
            uniform(BF16),
        ),
        (
            "quantised safetensors, F16 compute: lower than BF16",
            rule(st(Some(F16), true), None, None, &[]),
            uniform(BF16),
        ),
        (
            "quantised safetensors, no compute dtype",
            rule(st(None, true), None, None, &[]),
            uniform(BF16),
        ),
        (
            "F16 safetensors: its compute dtype",
            rule(st(Some(F16), false), None, None, &[]),
            uniform(F16),
        ),
        (
            "F32 safetensors: its compute dtype",
            rule(st(Some(F32), false), None, None, &[]),
            uniform(F32),
        ),
        (
            "safetensors with no compute dtype",
            rule(st(None, false), None, None, &[]),
            uniform(BF16),
        ),
        (
            "safetensors with an FP8 compute dtype and no quantisation",
            rule(st(Some(F8E4M3), false), None, None, &[]),
            uniform(BF16),
        ),
        // 2. A family requirement wins over the source.
        (
            "GGUF, family requires BF16 (F16 overflows)",
            rule(GGUF, None, Some(BF16), &[]),
            uniform(BF16),
        ),
        (
            "GGUF, family requires F32",
            rule(GGUF, None, Some(F32), &[]),
            uniform(F32),
        ),
        (
            "safetensors, family requires F32",
            rule(st(Some(BF16), false), None, Some(F32), &[]),
            uniform(F32),
        ),
        (
            "GGUF, family requires F32 on layers 0 and 5",
            rule(GGUF, None, None, &[(5, F32), (0, F32)]),
            spec(F16, &[(0, F32), (5, F32)]),
        ),
        (
            "a layer requirement equal to the default is no exception",
            rule(GGUF, None, Some(BF16), &[(2, BF16), (3, F32)]),
            spec(BF16, &[(3, F32)]),
        ),
        // 1. A declared FP8 cache wins over everything.
        (
            "declared E4M3 cache on safetensors",
            rule(st(Some(BF16), true), Some(F8E4M3), None, &[]),
            uniform(F8E4M3),
        ),
        (
            "declared E5M2 cache on GGUF",
            rule(GGUF, Some(F8E5M2), None, &[]),
            uniform(F8E5M2),
        ),
        (
            "declared cache over a family requirement",
            rule(GGUF, Some(F8E4M3), Some(F32), &[(1, F32)]),
            uniform(F8E4M3),
        ),
    ];
    for (name, rule, want) in cases {
        assert_eq!(rule.reference().as_ref(), Ok(&want), "{name}");
    }
}

#[test]
fn the_rule_refuses_inputs_rfc_0042_does_not_allow() {
    use DType::{BF16, F8E4M3, F16, I8, U8};
    assert_eq!(
        rule(GGUF, Some(BF16), None, &[]).reference(),
        Err(KvDTypeError::DeclaredNotFp8(BF16))
    );
    assert_eq!(
        rule(GGUF, None, Some(F8E4M3), &[]).reference(),
        Err(KvDTypeError::FamilyNotFloat(F8E4M3))
    );
    assert_eq!(
        rule(GGUF, None, None, &[(0, I8)]).reference(),
        Err(KvDTypeError::FamilyNotFloat(I8))
    );
    assert_eq!(
        rule(GGUF, None, None, &[(4, F16), (4, BF16)]).reference(),
        Err(KvDTypeError::LayerTwice(LayerIx(4)))
    );
    // A refused input is refused even under a declared cache: the caller's mistake shows.
    assert_eq!(
        rule(GGUF, Some(F8E4M3), Some(U8), &[]).reference(),
        Err(KvDTypeError::FamilyNotFloat(U8))
    );
    assert!(
        KvDTypeError::DeclaredNotFp8(BF16)
            .to_string()
            .contains("FP8")
    );
}

#[test]
fn for_source_is_the_source_default_everywhere() {
    for source in [GGUF, st(Some(DType::F32), false), st(None, true)] {
        assert_eq!(
            KvDTypeRule::for_source(source).reference(),
            Ok(uniform(KvDTypeRule::source_default(source)))
        );
    }
}

#[test]
fn normal_form_keeps_what_of_reads() {
    use DType::{BF16, F16, F32};
    let messy = spec(F16, &[(7, F32), (2, F16), (7, BF16), (1, BF16)]);
    let normal = messy.normalized();
    assert_eq!(normal, spec(F16, &[(1, BF16), (7, F32)]));
    for l in 0..10 {
        assert_eq!(normal.of(LayerIx(l)), messy.of(LayerIx(l)), "layer {l}");
    }
    assert_eq!(normal.normalized(), normal);
}

/// The override changes the resolved spec, and so the KV settings that prefix and session keys
/// hash (RFC-0022 §7.1). That the hash itself changes is TIER-E7-T01's test.
#[test]
fn the_override_replaces_the_source_default() {
    use DType::{BF16, F16, F32};
    let gguf = rule(GGUF, None, None, &[]);
    assert_eq!(gguf.resolve(None), gguf.reference());
    // RFC-0043 §10's use: BF16 where F16 overflows. Same width, wider range: allowed.
    let forced = gguf.resolve(Some(KvDTypeOverride::Bf16)).unwrap();
    assert_eq!(forced, uniform(BF16));
    assert_ne!(Ok(forced), gguf.reference());
    // Wider is allowed from either 16-bit default.
    assert_eq!(gguf.resolve(Some(KvDTypeOverride::F32)), Ok(uniform(F32)));
    let bf16_st = rule(st(Some(BF16), false), None, None, &[]);
    assert_eq!(
        bf16_st.resolve(Some(KvDTypeOverride::F32)),
        Ok(uniform(F32))
    );
    // The dtype the reference already has changes nothing.
    assert_eq!(gguf.resolve(Some(KvDTypeOverride::F16)), gguf.reference());
    for (o, d) in [
        (KvDTypeOverride::F16, F16),
        (KvDTypeOverride::Bf16, BF16),
        (KvDTypeOverride::F32, F32),
    ] {
        assert_eq!(o.dtype(), d);
    }
}

/// Review B1 on #1939: the override never moves a family requirement.
#[test]
fn the_override_keeps_every_family_requirement() {
    use DType::{BF16, F16, F32};
    // A family that requires BF16 on every layer, because F16 overflows: `f16` is refused, not
    // applied, and asking for the required dtype is a no-op.
    let all = rule(GGUF, None, Some(BF16), &[]);
    assert_eq!(
        all.resolve(Some(KvDTypeOverride::F16)),
        Err(KvDTypeError::OverrideOverFamily {
            required: BF16,
            requested: F16
        })
    );
    assert_eq!(
        all.resolve(Some(KvDTypeOverride::F32)),
        Err(KvDTypeError::OverrideOverFamily {
            required: BF16,
            requested: F32
        })
    );
    assert_eq!(all.resolve(Some(KvDTypeOverride::Bf16)), Ok(uniform(BF16)));
    // A per-layer requirement equal to the source default (F16 on GGUF), which the normal form
    // folds away in the reference, still holds after `bf16` moves the default.
    let layer = rule(GGUF, None, None, &[(3, F16)]);
    assert_eq!(layer.reference(), Ok(uniform(F16)));
    let resolved = layer.resolve(Some(KvDTypeOverride::Bf16)).unwrap();
    assert_eq!(resolved, spec(BF16, &[(3, F16)]));
    assert_eq!(resolved.of(LayerIx(3)), F16);
    assert_eq!(resolved.of(LayerIx(4)), BF16);
    // Other per-layer requirements stay too; one equal to the new default folds in.
    let layers = rule(GGUF, None, None, &[(0, F32), (3, BF16)]);
    assert_eq!(
        layers.resolve(Some(KvDTypeOverride::Bf16)),
        Ok(spec(BF16, &[(0, F32)]))
    );
}

/// Review B2 on #1939: exact mode never stores KV below the reference (D3).
#[test]
fn an_override_below_the_reference_is_refused() {
    use DType::{BF16, F16, F32};
    // An F32 checkpoint: either 16-bit dtype is fewer bits per element.
    let f32_st = rule(st(Some(F32), false), None, None, &[]);
    for (o, d) in [(KvDTypeOverride::F16, F16), (KvDTypeOverride::Bf16, BF16)] {
        let err = f32_st.resolve(Some(o)).unwrap_err();
        assert_eq!(
            err,
            KvDTypeError::OverrideNarrows {
                reference: F32,
                requested: d
            }
        );
        assert_eq!(err.class(), syntropy_core::ErrorClass::InvalidRequest);
        assert_eq!(err.name(), "kv_dtype_narrows");
        let text = err.to_string();
        assert!(text.contains("approximate.kv_lossy"), "{text}");
        assert!(text.contains("fidelity = \"approximate\""), "{text}");
    }
    // F16 over a BF16 reference: the same width, less range. Refused until the maintainers
    // decide otherwise (RFC-0042 §4.2, note of 2026-10-28).
    let bf16_st = rule(st(Some(BF16), false), None, None, &[]);
    assert_eq!(
        bf16_st.resolve(Some(KvDTypeOverride::F16)),
        Err(KvDTypeError::OverrideNarrows {
            reference: BF16,
            requested: F16
        })
    );
    // A quantised checkpoint's BF16 default likewise.
    assert!(matches!(
        rule(st(None, true), None, None, &[]).resolve(Some(KvDTypeOverride::F16)),
        Err(KvDTypeError::OverrideNarrows { .. })
    ));
    assert_eq!(f32_st.resolve(Some(KvDTypeOverride::F32)), Ok(uniform(F32)));
}

/// Review N1 on #1939: a declared FP8 cache is part of the reference, so no override moves it.
#[test]
fn an_override_never_changes_a_declared_cache() {
    use DType::{F8E4M3, F16, F32};
    let fp8 = rule(GGUF, Some(F8E4M3), None, &[]);
    for (o, d) in [
        (KvDTypeOverride::F16, F16),
        (KvDTypeOverride::Bf16, DType::BF16),
        (KvDTypeOverride::F32, F32),
    ] {
        let err = fp8.resolve(Some(o)).unwrap_err();
        assert_eq!(
            err,
            KvDTypeError::OverrideOverDeclaredCache {
                declared: F8E4M3,
                requested: d
            }
        );
        assert_eq!(err.class(), syntropy_core::ErrorClass::InvalidRequest);
        assert!(err.to_string().contains("declares"), "{err}");
    }
    assert_eq!(fp8.resolve(None), Ok(uniform(F8E4M3)));
    // A bad rule input is the caller's bug, refused before the override is looked at.
    let bad = rule(GGUF, Some(DType::BF16), None, &[]);
    let err = bad.resolve(Some(KvDTypeOverride::F32)).unwrap_err();
    assert_eq!(err, KvDTypeError::DeclaredNotFp8(DType::BF16));
    assert_eq!(err.class(), syntropy_core::ErrorClass::Internal);
}

static REGISTRY: Registry = Registry::new(&[&KvDTypeSection::SCHEMA]);

/// A policy table from `key = value` text, as an inline table.
fn table(text: &str) -> Value {
    format!("{{ {text} }}").parse().expect("test TOML parses")
}

fn from_policy(text: &str) -> Result<KvDTypeSection, InstanceSectionError> {
    InstanceSection::<KvDTypeSection>::from_policy(&table(text), |_| None)
        .map(InstanceSection::into_section)
}

#[test]
fn kv_dtype_is_a_top_level_instance_key_with_the_family_default() {
    let schema = &KvDTypeSection::SCHEMA;
    assert_eq!(schema.prefix, "");
    let key = schema.key("kv_dtype").unwrap();
    assert_eq!(key.path, "kv_dtype");
    assert_eq!(key.scope, Scope::Instance);
    assert_eq!(key.reload, Reload::Instance);
    assert_eq!(key.owner, "RFC-0043");
    assert!(matches!(key.default, DefaultSpec::Computed(rule) if rule.contains("RFC-0042")));
    assert!(!key.holds_secret);
    assert!(KvDTypeSection::check_defaults().is_empty());
    assert!(REGISTRY.check().is_empty());
}

#[test]
fn a_policy_sets_the_override_and_leaving_it_out_keeps_the_reference() {
    assert_eq!(from_policy("").unwrap().kv_dtype, None);
    for (text, want) in [
        ("f16", KvDTypeOverride::F16),
        ("bf16", KvDTypeOverride::Bf16),
        ("f32", KvDTypeOverride::F32),
    ] {
        let section = from_policy(&format!("kv_dtype = \"{text}\"")).unwrap();
        assert_eq!(section.kv_dtype, Some(want), "{text}");
    }
    // Another top-level key belongs to another section: this one ignores it.
    assert_eq!(from_policy("standby = \"none\"").unwrap().kv_dtype, None);
    // The resolved spec follows the policy.
    let policy = from_policy("kv_dtype = \"bf16\"").unwrap();
    assert_eq!(
        KvDTypeRule::for_source(KvSource::Gguf).resolve(policy.kv_dtype),
        Ok(KvDTypeSpec::uniform(DType::BF16))
    );
}

#[test]
fn a_dtype_below_the_wide_floats_is_refused() {
    for bad in ["\"fp8\"", "\"q8\"", "\"F16\"", "\"int8\"", "16"] {
        match from_policy(&format!("kv_dtype = {bad}")) {
            Err(InstanceSectionError::Rejected(problems)) => {
                assert_eq!(problems.len(), 1, "{bad}: {problems:?}");
                let p = &problems[0];
                assert_eq!(
                    p.key.clone().map(|k| k.into_string()).as_deref(),
                    Some("kv_dtype"),
                    "{bad}"
                );
                assert!(matches!(p.code, Some(Code::WrongType)), "{bad}: {p}");
            }
            other => panic!("{bad}: expected a rejection, got {other:?}"),
        }
    }
}
