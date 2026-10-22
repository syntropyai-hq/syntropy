// SPDX-License-Identifier: MIT OR Apache-2.0

//! COMP-E9-T01 against RFC-0040 §1 and §10.2 rules 1, 2 and 7: the units, the generated
//! instantiation lists and parameter mirrors, the layouts against a C compiler, and the cache
//! keys and the contraction check, all without a CUDA compiler.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use syntropy_core::{DType, GgmlType, QuantScheme};
use syntropy_kernels::{AttnKind, OpKind};

use super::key::{self, BuildRecord, FlagViolation, Key, UnitInputs};
use super::*;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(rel: &str) -> String {
    std::fs::read_to_string(root().join(rel))
        .unwrap()
        .replace("\r\n", "\n")
}

// ---- Units (rule 1) -----------------------------------------------------------------------

#[test]
fn units_are_unique_sorted_and_never_overlap() {
    let names: Vec<&str> = units::UNITS.iter().map(|u| u.name).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(names, sorted, "unique, in name order");
    // At most one unit per (family, group), and an `Any` unit owns its family alone.
    let mut seen = BTreeMap::new();
    for u in units::UNITS {
        assert!(
            seen.insert((u.family, u.group), u.name).is_none(),
            "{}",
            u.name
        );
    }
    for u in units::UNITS
        .iter()
        .filter(|u| u.group == units::SchemeGroup::Any)
    {
        let same: Vec<_> = units::UNITS
            .iter()
            .filter(|v| v.family == u.family)
            .collect();
        assert_eq!(same.len(), 1, "{} shares its family", u.name);
    }
}

#[test]
fn ops_map_to_their_unit_by_family_and_scheme_group() {
    let q4k = QuantScheme::Ggml(GgmlType::Q4_K);
    let q8 = QuantScheme::Ggml(GgmlType::Q8_0);
    let f16 = QuantScheme::Plain(DType::F16);
    let iq = QuantScheme::Ggml(GgmlType::IQ2_XS);
    let mx = QuantScheme::Mxfp4 { block: 32 };
    let unit = |op: OpKind, w: &QuantScheme| units::unit_for(&op, Some(w)).map(|u| u.name);
    assert_eq!(unit(OpKind::Gemv, &q4k), Some("gemv/kquant"));
    assert_eq!(unit(OpKind::Logits, &q8), Some("gemv/legacy"));
    assert_eq!(unit(OpKind::Gemv, &f16), Some("gemv/float"));
    assert_eq!(unit(OpKind::Gemv, &iq), Some("gemv/iquant"));
    assert_eq!(unit(OpKind::Gemv, &mx), Some("gemv/fp4"));
    assert_eq!(
        unit(OpKind::AttnDecode(AttnKind::Mla), &f16),
        Some("attn/decode")
    );
    assert_eq!(unit(OpKind::Convert, &q4k), Some("convert/dequant"));
    // No unit: a family or group not built yet.
    assert_eq!(unit(OpKind::Gemm, &mx), None);
    assert_eq!(unit(OpKind::Timestamp, &f16), None);
}

#[test]
fn bit_exact_units_are_rfc_0042s_classes() {
    let exact: Vec<&str> = units::UNITS
        .iter()
        .filter(|u| u.bit_exact)
        .map(|u| u.name)
        .collect();
    // Dequantisation, row copies, KV page moves, index selection (RFC-0042 §6.4).
    assert_eq!(
        exact,
        [
            "convert/dequant",
            "copy/kv_pages",
            "copy/rows",
            "sample/select"
        ]
    );
}

// ---- support.toml and the instantiation lists (rule 2) ------------------------------------

#[test]
fn support_table_builds_the_registry() {
    let support = support::parse(&read(support::SUPPORT_PATH)).unwrap();
    assert_eq!(support.registry, crate::registry());
    let listed: usize = support.by_unit.values().map(Vec::len).sum();
    assert_eq!(listed, support.registry.len());
    assert_eq!(support.by_unit.len(), units::UNITS.len());
    assert!(support.by_unit["gemv/kquant"].len() >= 2);
}

#[test]
fn support_table_errors_are_reported() {
    let gemv = r#"key = { op = "Gemv", schemes = [{ Ggml = "Q4_K" }], layout = 0, variant = 0 }"#;
    let twice = format!("[[op]]\n{gemv}\n[[op]]\n{gemv}\n");
    assert!(matches!(
        support::parse(&twice),
        Err(support::SupportError::Registry(_))
    ));
    let no_unit = r#"[[op]]
key = { op = "Timestamp", schemes = [], layout = 0, variant = 0 }"#;
    assert!(matches!(
        support::parse(no_unit),
        Err(support::SupportError::NoUnit(_))
    ));
    let tile = format!("[[op]]\n{gemv}\ntiles = [4, 2, 4]\n");
    assert!(matches!(
        support::parse(&tile),
        Err(support::SupportError::DuplicateTile(_, 4))
    ));
    let bad_fusion = r#"[[op]]
key = { op = "Gemv", schemes = [{ Ggml = "Q4_K" }], layout = 0, variant = 32768 }"#;
    assert!(matches!(
        support::parse(bad_fusion),
        Err(support::SupportError::Registry(_))
    ));
    assert!(matches!(
        support::parse("[[op]]\nkey = 1\n"),
        Err(support::SupportError::Parse(_))
    ));
}

#[test]
fn instantiation_lines_are_deterministic_and_ordered() {
    let text = "[[op]]\n\
        key = { op = \"Gemv\", schemes = [{ Ggml = \"Q6_K\" }, { Plain = \"F32\" }, { Plain = \"F32\" }], layout = 0, variant = 0 }\n\
        tiles = [8, 1]\n\
        [[op]]\n\
        key = { op = { GateUpAct = { act = \"Silu\" } }, schemes = [{ Ggml = \"Q4_K\" }], layout = 2, variant = 5 }\n";
    let support = support::parse(text).unwrap();
    let unit = units::UNITS
        .iter()
        .find(|u| u.name == "gemv/kquant")
        .unwrap();
    let rendered = inc::render(unit, &support.by_unit[unit.name]);
    let lines: Vec<&str> = rendered.lines().filter(|l| l.starts_with("SYN_")).collect();
    // `OpKey` order (`Gemv` before `GateUpAct`), then tile order; missing schemes pad with NONE.
    assert_eq!(
        lines,
        [
            "SYN_INSTANTIATE(GEMV, GGML_Q6_K, PLAIN_F32, PLAIN_F32, 0, 0x0000, 1)",
            "SYN_INSTANTIATE(GEMV, GGML_Q6_K, PLAIN_F32, PLAIN_F32, 0, 0x0000, 8)",
            "SYN_INSTANTIATE(GATEUPACT_ACT_SILU, GGML_Q4_K, NONE, NONE, 2, 0x0005, 0)",
        ]
    );
    // The same table renders the same bytes, whatever order it lists its entries in.
    let reordered = "[[op]]\n\
        key = { op = { GateUpAct = { act = \"Silu\" } }, schemes = [{ Ggml = \"Q4_K\" }], layout = 2, variant = 5 }\n\
        [[op]]\n\
        key = { op = \"Gemv\", schemes = [{ Ggml = \"Q6_K\" }, { Plain = \"F32\" }, { Plain = \"F32\" }], layout = 0, variant = 0 }\n\
        tiles = [1, 8]\n";
    let again = support::parse(reordered).unwrap();
    assert_eq!(rendered, inc::render(unit, &again.by_unit[unit.name]));
}

#[test]
fn c_identifiers_from_values() {
    assert_eq!(inc::c_ident(&OpKind::Gemv), "GEMV");
    assert_eq!(
        inc::c_ident(&OpKind::AttnDecode(AttnKind::SlidingGqa { sinks: true })),
        "ATTNDECODE_SLIDINGGQA_SINKS_TRUE"
    );
    assert_eq!(
        inc::c_ident(&QuantScheme::Nvfp4 {
            block: 16,
            scale: DType::F8E4M3
        }),
        "NVFP4_BLOCK_16_SCALE_F8E4M3"
    );
}

/// The checked-in generated files are what `kgen` makes of today's inputs, and `kgen --check`
/// agrees. `cargo xtask kgen` rewrites them.
#[test]
fn generated_files_are_current() {
    let files = generate(&root()).unwrap();
    let mut stale = Vec::new();
    for (rel, contents) in &files {
        let on_disk = std::fs::read_to_string(root().join(rel))
            .map(|t| t.replace("\r\n", "\n"))
            .unwrap_or_default();
        if &on_disk != contents {
            stale.push(rel.clone());
        }
    }
    assert!(stale.is_empty(), "stale: {stale:?}; run `cargo xtask kgen`");
    assert_eq!(files.len(), units::UNITS.len() + 1);
    assert!(orphans(&root()).is_empty(), "{:?}", orphans(&root()));
    let (ok, report) = run(&root(), true).unwrap();
    assert!(ok, "{report}");
}

/// Every unit includes its own list and nothing else's, so no unit relies on another's
/// instantiations (rule 2).
#[test]
fn each_unit_includes_its_own_list() {
    for unit in units::UNITS {
        let source = read(&unit.source_path());
        let includes: Vec<&str> = source
            .lines()
            .filter(|l| l.starts_with("#include \"instantiate/"))
            .collect();
        assert_eq!(
            includes,
            [format!("#include \"instantiate/{}.inc\"", unit.name).as_str()],
            "{}",
            unit.name
        );
        assert_eq!(
            source.contains("--fmad=false"),
            unit.bit_exact,
            "{}: the comment names the flag exactly when the unit is bit-exact",
            unit.name
        );
    }
}

/// Generating into a scratch copy writes the same bytes twice, and `--check` then passes;
/// a changed table makes `--check` fail until the files are regenerated.
#[test]
fn kgen_writes_then_checks_clean() {
    let dir = std::env::temp_dir().join(format!("syntropy-kgen-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    copy_tree(&root().join("csrc"), &dir.join("csrc"));
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::write(dir.join("support.toml"), read("support.toml")).unwrap();
    // Remove the generated files: kgen rebuilds them.
    std::fs::remove_dir_all(dir.join("csrc/instantiate")).unwrap();
    let (ok, _) = run(&dir, false).unwrap();
    assert!(ok);
    let first = generate(&dir).unwrap();
    let (ok, report) = run(&dir, true).unwrap();
    assert!(ok, "{report}");
    assert_eq!(first, generate(&root()).unwrap(), "same inputs, same bytes");
    // One more instantiation: only that unit's list goes stale.
    let mut table = read("support.toml");
    table.push_str(
        "\n[[op]]\nkey = { op = \"Gemv\", schemes = [{ Ggml = \"Q5_K\" }, { Plain = \"F32\" }, { Plain = \"F32\" }], layout = 0, variant = 0 }\ntiles = [1]\n",
    );
    std::fs::write(dir.join("support.toml"), table).unwrap();
    let (ok, report) = run(&dir, true).unwrap();
    assert!(!ok);
    let stale: Vec<&str> = report
        .lines()
        .filter_map(|l| l.strip_prefix("kgen: stale: "))
        .collect();
    assert_eq!(stale, ["csrc/instantiate/gemv/kquant.inc"]);
    // An orphan unit source fails the check.
    std::fs::write(dir.join("csrc/gemv/extra.cu"), "// SPDX\n").unwrap();
    assert_eq!(orphans(&dir), ["csrc/gemv/extra.cu"]);
    let _ = std::fs::remove_dir_all(&dir);
}

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let target = to.join(entry.file_name());
        if path.is_dir() {
            copy_tree(&path, &target);
        } else {
            std::fs::copy(&path, &target).unwrap();
        }
    }
}

// ---- The parameter header and its mirrors (§1) --------------------------------------------

#[test]
fn rust_layout_matches_the_header() {
    let header = cgen::parse(&read(cgen::HEADER_PATH)).unwrap();
    assert_eq!(header.structs.len(), crate::params::LAYOUTS.len());
    for (c, rust) in header.structs.iter().zip(crate::params::LAYOUTS) {
        let tag = rust.tag;
        assert_eq!(c.c_name, tag);
        assert_eq!((c.size, c.align), (rust.size, rust.align), "{tag}");
        let c_fields: Vec<(&str, usize)> = c
            .fields
            .iter()
            .map(|f| (f.name.as_str(), f.offset))
            .collect();
        assert_eq!(c_fields, rust.fields.to_vec(), "{tag} members");
    }
    assert_eq!(crate::params::PARAMS_ABI_VERSION, 1);
}

/// Independent of the generator: `syn_norm_params` by C's rules, written out by hand.
#[test]
fn a_layout_by_hand() {
    let norm = crate::params::LAYOUTS
        .iter()
        .find(|l| l.tag == "syn_norm_params")
        .unwrap();
    assert_eq!(
        (norm.size, norm.align, norm.fields),
        (
            40,
            8,
            &[
                ("input", 0),
                ("output", 8),
                ("weight", 16),
                ("step", 24),
                ("hidden", 32),
                ("eps", 36)
            ][..]
        )
    );
}

#[test]
fn the_generator_refuses_what_it_cannot_lay_out() {
    for bad in [
        "struct syn_a {\n    uint16_t x;\n};\n",
        "struct syn_a {\n    uint32_t x[4];\n};\n",
        "#pragma pack(1)\n",
        "struct syn_a {\n#ifdef X\n    uint32_t x;\n#endif\n};\n",
        "struct a {\n    uint32_t x;\n};\n",
        "struct syn_a {\n    uint32_t x; /* c */\n};\n",
        "#define OTHER 3\n",
        "struct syn_a {\n",
    ] {
        assert!(cgen::parse(bad).is_err(), "accepted: {bad:?}");
    }
    let h =
        cgen::parse("struct syn_a_b {\n    uint32_t x;\n    uint64_t y;\n    uint32_t z;\n};\n")
            .unwrap();
    let s = &h.structs[0];
    assert_eq!(s.rust_name(), "AB");
    assert_eq!((s.size, s.align), (24, 8));
    let offsets: Vec<usize> = s.fields.iter().map(|f| f.offset).collect();
    assert_eq!(offsets, [0, 8, 16]);
}

/// Finds a C or C++ compiler: `$CC`/`$CXX`, else `cc`/`c++`.
fn compiler(env: &str, default: &str) -> Option<String> {
    let cmd = std::env::var(env).unwrap_or_else(|_| default.to_owned());
    let found = Command::new(&cmd).arg("--version").output().ok()?;
    found.status.success().then_some(cmd)
}

/// A C and a C++ compiler, where present, compile the header with `_Static_assert`s of the
/// Rust layout: the compiler's layout equals Rust's. nvcc lays out these plain structs as the
/// host C++ compiler does. In CI on Linux and macOS both compilers must be present; elsewhere
/// a missing one is skipped with a note.
#[test]
fn c_compiler_agrees_when_present() {
    let dir = std::env::temp_dir().join(format!("syntropy-params-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let check = cgen::c_layout_check(crate::params::LAYOUTS);
    let required = std::env::var_os("CI").is_some() && !cfg!(windows);
    for (env, default, file, std) in [
        ("CC", "cc", "check.c", "-std=c11"),
        ("CXX", "c++", "check.cpp", "-std=c++11"),
    ] {
        let Some(cc) = compiler(env, default) else {
            assert!(!required, "no {default} compiler found in CI (set {env})");
            eprintln!("skipped: no {default} compiler found (set {env} to use one)");
            continue;
        };
        let src = dir.join(file);
        std::fs::write(&src, &check).unwrap();
        let out = Command::new(&cc)
            .arg(std)
            .arg("-fsyntax-only")
            .arg("-I")
            .arg(root().join("csrc"))
            .arg(&src)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{cc} rejected the layout:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Where a C++ compiler is present, every unit preprocesses and parses as C++: its includes
/// resolve under `csrc/`, and every line of its generated list is a well-formed
/// `SYN_INSTANTIATE` call. Until the kernels land the macro expands to nothing, so this checks
/// structure, not CUDA code; compiling with nvcc is COMP-E9-T02's.
#[test]
fn every_unit_parses_with_the_host_compiler_when_present() {
    let required = std::env::var_os("CI").is_some() && !cfg!(windows);
    let Some(cxx) = compiler("CXX", "c++") else {
        assert!(!required, "no c++ compiler found in CI (set CXX)");
        eprintln!("skipped: no c++ compiler found (set CXX to use one)");
        return;
    };
    for unit in units::UNITS {
        let out = Command::new(&cxx)
            .args(["-x", "c++", "-std=c++17", "-fsyntax-only", "-I"])
            .arg(root().join("csrc"))
            .arg(root().join(unit.source_path()))
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{cxx} rejected {}:\n{}",
            unit.source_path(),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

// ---- Flags, keys and the cache (rules 2, 3, 6, 7) -----------------------------------------

const TOOLCHAIN: &str = "Cuda compilation tools, release 13.0, V13.0.48";

fn inputs() -> Vec<UnitInputs<'static>> {
    let params = key::content_hash(read(cgen::HEADER_PATH).as_bytes());
    units::UNITS
        .iter()
        .map(|unit| UnitInputs {
            unit,
            source: read(&unit.source_path()).into_bytes(),
            includes: vec![params],
            instantiation: read(&unit.inc_path()).into_bytes(),
        })
        .collect()
}

fn unit(name: &str) -> &'static units::Unit {
    units::UNITS.iter().find(|u| u.name == name).unwrap()
}

#[test]
fn bit_exact_units_compile_with_contraction_off_and_others_do_not() {
    for u in units::UNITS {
        for target in key::TARGETS {
            let flags = key::nvcc_flags(u, target);
            assert_eq!(flags.contains(&"--fmad=false".to_owned()), u.bit_exact);
            assert!(
                flags
                    .iter()
                    .all(|f| !key::FORBIDDEN_FLAGS.contains(&f.as_str()))
            );
            assert!(flags.contains(&format!("--gpu-architecture={target}")));
        }
    }
    assert!(key::nvcc_flags(unit("gemv/kquant"), "compute_120").contains(&"--ptx".to_owned()));
}

/// Rule 7's check, as L0 runs it over the build records: a bit-exact-class unit built without
/// `--fmad=false` fails, and so does any unit built with fast math.
#[test]
fn the_check_fails_a_bit_exact_unit_built_without_contraction_off() {
    let all = inputs();
    let records: Vec<BuildRecord<'_>> = all
        .iter()
        .flat_map(|i| {
            key::TARGETS
                .iter()
                .map(move |t| BuildRecord::new(i, TOOLCHAIN, t, key::nvcc_flags(i.unit, t)))
        })
        .collect();
    assert_eq!(key::check_flags(&records), []);

    let dequant = all
        .iter()
        .find(|i| i.unit.name == "convert/dequant")
        .unwrap();
    let mut without: Vec<String> = key::nvcc_flags(dequant.unit, "sm_89");
    without.retain(|f| f != "--fmad=false");
    let overridden = [
        key::nvcc_flags(dequant.unit, "sm_89"),
        vec!["--fmad=true".to_owned()],
    ]
    .concat();
    let bad = [
        BuildRecord::new(dequant, TOOLCHAIN, "sm_89", without.clone()),
        BuildRecord::new(dequant, TOOLCHAIN, "sm_89", overridden),
    ];
    assert_eq!(
        key::check_flags(&bad),
        [
            FlagViolation::ContractionOn {
                unit: "convert/dequant",
                target: "sm_89".into()
            },
            FlagViolation::ContractionOn {
                unit: "convert/dequant",
                target: "sm_89".into()
            },
        ]
    );
    // The same flags on a unit outside the class pass: contraction is allowed there.
    let gemv = all.iter().find(|i| i.unit.name == "gemv/kquant").unwrap();
    assert_eq!(
        key::check_flags(&[BuildRecord::new(gemv, TOOLCHAIN, "sm_89", without)]),
        []
    );
    let fast = [
        key::nvcc_flags(gemv.unit, "sm_89"),
        vec!["--use_fast_math".to_owned()],
    ]
    .concat();
    assert_eq!(
        key::check_flags(&[BuildRecord::new(gemv, TOOLCHAIN, "sm_89", fast)]),
        [FlagViolation::FastMath {
            unit: "gemv/kquant",
            target: "sm_89".into(),
            flag: "--use_fast_math".into()
        }]
    );
    // The flag is part of the key: the same unit without it is a different object.
    let with = BuildRecord::new(
        dequant,
        TOOLCHAIN,
        "sm_89",
        key::nvcc_flags(dequant.unit, "sm_89"),
    );
    assert_ne!(with.object_key, bad[0].object_key);
}

/// Rule 1 and 3 through the keys: a rebuild with no change hits every object; changing one
/// unit's source misses exactly that unit times every target.
#[test]
fn changing_one_unit_recompiles_only_that_unit_times_targets() {
    let base = inputs();
    let (keys, cold) = key::plan(&base, key::TARGETS, TOOLCHAIN, &BTreeSet::new());
    let n = units::UNITS.len() * key::TARGETS.len();
    assert_eq!((cold.hits, cold.misses.len()), (0, n));
    let unique: BTreeSet<Key> = keys.iter().copied().collect();
    assert_eq!(unique.len(), n, "every unit × target is its own object");
    let cache = unique;

    let (_, warm) = key::plan(&base, key::TARGETS, TOOLCHAIN, &cache);
    assert_eq!((warm.hits, warm.misses.len()), (n, 0));

    let mut edited = inputs();
    let ix = edited
        .iter()
        .position(|i| i.unit.name == "attn/decode")
        .unwrap();
    edited[ix].source.extend_from_slice(b"// a kernel change\n");
    let (_, stats) = key::plan(&edited, key::TARGETS, TOOLCHAIN, &cache);
    assert_eq!(stats.hits, n - key::TARGETS.len());
    assert!(stats.misses.iter().all(|(u, _)| *u == "attn/decode"));
    let targets: Vec<&str> = stats.misses.iter().map(|(_, t)| t.as_str()).collect();
    assert_eq!(targets, key::TARGETS);
}

/// Rule 2: the instantiation list is part of the `ObjectKey`, so a new instantiation misses
/// the cache for its unit only. A shared header changes every unit that includes it, and a new
/// toolchain changes everything.
#[test]
fn the_instantiation_list_is_part_of_the_object_key() {
    let base = inputs();
    let (keys, _) = key::plan(&base, key::TARGETS, TOOLCHAIN, &BTreeSet::new());
    let cache: BTreeSet<Key> = keys.into_iter().collect();
    let n = cache.len();

    let mut edited = inputs();
    let ix = edited
        .iter()
        .position(|i| i.unit.name == "gemv/kquant")
        .unwrap();
    edited[ix].instantiation.extend_from_slice(
        b"SYN_INSTANTIATE(GEMV, GGML_Q5_K, PLAIN_F32, PLAIN_F32, 0, 0x0000, 1)\n",
    );
    let (_, stats) = key::plan(&edited, key::TARGETS, TOOLCHAIN, &cache);
    assert_eq!(stats.misses.len(), key::TARGETS.len());
    assert!(stats.misses.iter().all(|(u, _)| *u == "gemv/kquant"));

    let mut header = inputs();
    let other = key::content_hash(b"changed params.h");
    for i in &mut header {
        i.includes = vec![other];
    }
    let (_, stats) = key::plan(&header, key::TARGETS, TOOLCHAIN, &cache);
    assert_eq!((stats.hits, stats.misses.len()), (0, n));

    let (_, stats) = key::plan(
        &base,
        key::TARGETS,
        "Cuda compilation tools, release 13.1",
        &cache,
    );
    assert_eq!(stats.misses.len(), n);
}

#[test]
fn keys_frame_their_fields() {
    let flags = |f: &[&str]| f.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
    // Bytes moved across a field boundary give a different key.
    assert_ne!(
        key::source_key("ab", "c", &flags(&[]), b""),
        key::source_key("a", "bc", &flags(&[]), b"")
    );
    assert_ne!(
        key::source_key("t", "sm_89", &flags(&["-a", "b"]), b""),
        key::source_key("t", "sm_89", &flags(&["-ab"]), b"")
    );
    // Whitespace around a flag is normalised away; order is kept.
    assert_eq!(
        key::source_key("t", "sm_89", &flags(&[" -O3 ", ""]), b"x"),
        key::source_key("t", "sm_89", &flags(&["-O3"]), b"x")
    );
    assert_ne!(
        key::source_key("t", "sm_89", &flags(&["-a", "-b"]), b""),
        key::source_key("t", "sm_89", &flags(&["-b", "-a"]), b"")
    );
    let s = key::source_key("t", "sm_89", &flags(&[]), b"x");
    assert_ne!(
        key::object_key(s, &[], b"ab"),
        key::object_key(s, &[key::content_hash(b"")], b"ab")
    );
}
