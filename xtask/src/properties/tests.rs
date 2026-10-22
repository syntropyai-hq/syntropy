// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap()
}

fn property(id: &str, crate_name: &str, owner: &str) -> Property {
    Property {
        id: id.to_owned(),
        crate_name: crate_name.to_owned(),
        owner: owner.to_owned(),
        line: 1,
    }
}

fn set(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

fn file(path: &str, text: &str) -> (String, String) {
    (path.to_owned(), text.to_owned())
}

#[test]
fn the_checkout_passes() {
    let properties = parse(&read(root(), REGISTRY).unwrap()).unwrap();
    let crates = crate_names(root());
    let table = table_ids(&read(root(), RFC).unwrap());
    let epics = epics_text(root()).unwrap();
    assert_eq!(
        registry_problems(&properties, &table, &crates, &epics),
        Vec::<String>::new()
    );
    let (files, suites) = sources(root(), &crates);
    assert_eq!(
        source_problems(&properties, &files, &suites),
        Vec::<String>::new()
    );
    assert!(properties.iter().any(|p| p.id == "P-core-1"));
}

#[test]
fn ids_and_prefixes() {
    for id in ["P-core-1", "P-kv-12", "P-bench-2"] {
        assert!(valid_id(id), "{id}");
    }
    for id in [
        "P-core-0",
        "P-core-01",
        "P-Core-1",
        "P-core",
        "Q-core-1",
        "P-core-1-2",
        "P--1",
    ] {
        assert!(!valid_id(id), "{id}");
    }
    assert_eq!(prefix("P-core-1"), "p_core_1_");
    assert_eq!(prefix("P-kv-12"), "p_kv_12_");
}

#[test]
fn the_registry_reads_three_string_keys() {
    let text = "# c\n[[property]]\nid = \"P-core-1\"\ncrate = \"core\"\nowner = \"QUAL-E3-T05\"\n";
    assert_eq!(
        parse(text).unwrap(),
        [Property {
            line: 2,
            ..property("P-core-1", "core", "QUAL-E3-T05")
        }]
    );
    let extra = "[[property]]\nid = \"P-core-1\"\ncrate = \"core\"\nowner = \"Q\"\nfile = \"x\"\n";
    assert!(parse(extra).unwrap_err()[0].contains("unknown key `file`"));
    let missing = "[[property]]\nid = \"P-core-1\"\ncrate = \"core\"\n";
    assert!(parse(missing).unwrap_err()[0].contains("needs id, crate and owner"));
    let boolean = "[[property]]\nid = \"P-core-1\"\ncrate = \"core\"\nowner = true\n";
    assert!(parse(boolean).unwrap_err()[0].contains("`owner` is not a string"));
}

#[test]
fn the_table_ids_are_the_first_cells_of_rfc_0060s_rows() {
    let rfc = "| ID | Crate |\n|---|---|\n| P-core-1 | core | x |\n| P-kv-2 | tier (KV) |\n\
               text P-net-9 in prose\n";
    assert_eq!(table_ids(rfc), set(&["P-core-1", "P-kv-2"]));
    assert!(table_ids(&read(root(), RFC).unwrap()).contains("P-solver-1"));
}

#[test]
fn entries_are_checked_against_the_table_the_crates_and_the_epics() {
    let table = set(&["P-core-1", "P-tier-1"]);
    let crates = set(&["core", "tier"]);
    let epics = "| QUAL-E3-T05 | the framework |\n";
    let good = [property("P-core-1", "core", "QUAL-E3-T05")];
    assert_eq!(
        registry_problems(&good, &table, &crates, epics),
        Vec::<String>::new()
    );
    let bad = [
        property("P-core-1", "core", "QUAL-E3-T05"),
        property("P-core-1", "core", "QUAL-E3-T05"),
        property("P-core-9", "core", "QUAL-E3-T05"),
        property("P-tier-1", "teir", "QUAL-E3-T05"),
        property("P-tier-1x", "tier", "QUAL-E3-T05"),
        property("P-tier-1", "tier", "nobody"),
        property("P-tier-1", "tier", "QUAL-E9-T99"),
    ];
    let found = registry_problems(&bad, &table, &crates, epics);
    for (text, expected) in [
        ("listed twice", 3),
        ("not in RFC-0060", 1),
        ("not a crate under crates/", 1),
        ("not a property ID", 1),
        ("is not a task ID", 1),
        ("not a task in the epic files", 1),
    ] {
        let count = found.iter().filter(|line| line.contains(text)).count();
        assert_eq!(count, expected, "{text}: {found:#?}");
    }
}

#[test]
fn property_fns_are_found_by_name() {
    let text = "fn p_core_1_order() {}\n    fn p_kv_2_x(a: u8) {}\nfn p_value() {}\n\
                fn p_core_x_y() {}\nfn helper_p_core_1_() {}\nxfn p_core_3_z() {}\n";
    assert_eq!(
        property_fns(text),
        [("P-core-1".to_owned(), 1), ("P-kv-2".to_owned(), 2)]
    );
}

#[test]
fn a_property_needs_listed_tests_in_an_allowed_place_on_the_default_config() {
    let listed = [
        property("P-core-1", "core", "QUAL-E3-T05"),
        property("P-bench-2", "bench", "QUAL-E12-T01"),
    ];
    let suites = set(&["crates/bench/tests/schema"]);
    let good = [
        file("crates/core/src/plan/tests.rs", "fn p_core_1_a() {}\n"),
        file(
            "crates/bench/tests/schema/comparability.rs",
            "fn p_bench_2_a() {}\n",
        ),
        file("crates/core/tests/other.rs", "fn unrelated() {}\n"),
    ];
    assert_eq!(
        source_problems(&listed, &good, &suites),
        Vec::<String>::new()
    );

    let bad = [
        file("crates/core/tests/single.rs", "fn p_core_1_a() {}\n"),
        file(
            "crates/bench/tests/schema/comparability.rs",
            "proptest! {\n#![proptest_config(ProptestConfig::with_cases(9))]\nfn p_bench_2_a() {}\n}\n",
        ),
        file(
            "crates/tier/src/lib.rs",
            "fn p_tier_7_unlisted() {}\nfn p_core_1_elsewhere() {}\n",
        ),
    ];
    let found = source_problems(&listed, &bad, &suites);
    for text in [
        "crates/core/tests/single.rs: holds a property, but is neither under src/",
        "crates/bench/tests/schema/comparability.rs:2: sets a proptest_config",
        "crates/tier/src/lib.rs:1: a test named like property P-tier-7",
        "crates/tier/src/lib.rs:2: a test of P-core-1, which ci/properties.toml puts in `core`",
    ] {
        assert!(
            found.iter().any(|line| line.starts_with(text)),
            "{text}: {found:#?}"
        );
    }
    assert_eq!(found.len(), 4, "{found:#?}");

    let none = source_problems(&listed, &[], &suites);
    assert_eq!(none.len(), 2, "{none:#?}");
    assert!(none[0].contains("P-core-1 has no test function named `p_core_1_...`"));
}

#[test]
fn a_removed_property_needs_a_reason_in_the_pr_description() {
    let base = set(&["P-core-1", "P-core-2", "P-core-3"]);
    let head = set(&["P-core-1"]);

    let none = removal_problems(&base, &head, "");
    assert_eq!(none.len(), 2, "{none:#?}");
    assert!(none[0].contains("P-core-2 was removed, and the PR description has no line"));

    let body = "Moves two properties.\n\n- Removes P-core-2: it became P-tier-3 (OQ-PRC-27)\n\
                Removes P-core-3:   \n";
    let found = removal_problems(&base, &head, body);
    assert_eq!(found.len(), 1, "an empty reason does not count: {found:#?}");
    assert!(found[0].contains("P-core-3 was removed, and the PR description has no line"));

    let both = "Removes P-core-2: became P-tier-3\n* Removes P-core-3: became P-config-2\n";
    assert_eq!(removal_problems(&base, &head, both), Vec::<String>::new());
    assert_eq!(
        removal_problems(&head, &base, ""),
        Vec::<String>::new(),
        "adding properties needs no reason"
    );
}

#[test]
fn a_checkout_without_a_registry_has_no_properties() {
    let empty = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    assert_eq!(registry_ids(&empty).unwrap(), BTreeSet::new());
    let ids = registry_ids(root()).unwrap();
    assert!(ids.contains("P-core-1"), "{ids:?}");
}
