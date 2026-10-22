// SPDX-License-Identifier: MIT OR Apache-2.0

//! The layer table in `ci/layers.toml` (RFC-0001 §4.1, RFC-0062 §12.1).
//!
//! The file uses a subset of TOML that needs no parser crate: `[section]` headers, `key = [...]`
//! arrays of plain double-quoted strings (which may span lines, with a trailing comma), and `#`
//! comments. Every file this module accepts is valid TOML with the same meaning; anything outside
//! the subset is an error that names the line.

use std::collections::BTreeMap;
use std::path::Path;

/// Where the table lives, relative to the workspace root.
pub const PATH: &str = "ci/layers.toml";

/// Where a crate sits in the table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Place {
    /// A numbered layer of RFC-0001 §4.1.
    Layer(u8),
    /// Layer "—": dev only (rule 9).
    DevOnly,
    /// Outside the crate graph: no workspace edges either way.
    Tool,
}

/// The parsed table: each crate, by its name without the `syntropy-` prefix, and its place.
#[derive(Debug, PartialEq, Eq)]
pub struct Layers {
    places: BTreeMap<String, Place>,
}

impl Layers {
    /// Reads `ci/layers.toml` under `root`.
    pub fn load(root: &Path) -> Result<Layers, String> {
        let path = root.join(PATH);
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        Layers::parse(&text).map_err(|error| format!("{PATH}:{error}"))
    }

    /// Parses the table. An error starts with the line number it refers to.
    pub fn parse(text: &str) -> Result<Layers, String> {
        let mut places = BTreeMap::new();
        let mut layer_keys = Vec::new();
        let mut outside_keys: Vec<&str> = Vec::new();
        let mut section: Option<&str> = None;
        let mut seen_sections: Vec<&str> = Vec::new();
        let mut lines = text
            .lines()
            .enumerate()
            .map(|(index, line)| (index + 1, line));
        while let Some((number, line)) = lines.next() {
            let line = strip_comment(line).trim();
            if line.is_empty() {
                continue;
            }
            if let Some(name) = line.strip_prefix('[') {
                let name = name
                    .strip_suffix(']')
                    .ok_or_else(|| format!("{number}: a section header ends with `]`"))?
                    .trim();
                if !matches!(name, "layers" | "outside") {
                    return Err(format!(
                        "{number}: unknown section `[{name}]`; expected `[layers]` or `[outside]`"
                    ));
                }
                if seen_sections.contains(&name) {
                    return Err(format!("{number}: section `[{name}]` appears twice"));
                }
                seen_sections.push(name);
                section = Some(name);
                continue;
            }
            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| format!("{number}: expected `key = [\"name\", ...]`"))?;
            let key = key.trim();
            let mut value = value.trim().to_owned();
            // An array may span lines: read on until its closing bracket.
            while !value.ends_with(']') {
                let Some((_, more)) = lines.next() else {
                    return Err(format!(
                        "{number}: the array for `{key}` has no closing `]`"
                    ));
                };
                value.push(' ');
                value.push_str(strip_comment(more).trim());
            }
            let names = parse_array(&value).map_err(|error| format!("{number}: {error}"))?;
            let place = match (section, key) {
                (Some("layers"), key) => {
                    let layer: u8 = key
                        .parse()
                        .ok()
                        .filter(|layer: &u8| key == layer.to_string())
                        .ok_or_else(|| {
                            format!("{number}: a key in `[layers]` is a layer number, not `{key}`")
                        })?;
                    if layer as usize != layer_keys.len() {
                        return Err(format!(
                            "{number}: layer {layer} is out of order; layers run 0, 1, 2, ... \
                             without gaps"
                        ));
                    }
                    layer_keys.push(layer);
                    Place::Layer(layer)
                }
                (Some("outside"), "dev-only") => Place::DevOnly,
                (Some("outside"), "tools") => Place::Tool,
                (Some("outside"), key) => {
                    return Err(format!(
                        "{number}: unknown key `{key}` in `[outside]`; expected `dev-only` or `tools`"
                    ));
                }
                (_, key) => {
                    return Err(format!("{number}: `{key}` is outside a section"));
                }
            };
            // A repeated layer number is caught above, as out of order.
            if place == Place::DevOnly || place == Place::Tool {
                if outside_keys.contains(&key) {
                    return Err(format!("{number}: `{key}` appears twice"));
                }
                outside_keys.push(key);
            }
            for name in names {
                if let Some(previous) = places.insert(name.clone(), place) {
                    return Err(format!(
                        "{number}: `{name}` is listed twice (also as {})",
                        describe(previous)
                    ));
                }
            }
        }
        if layer_keys.is_empty() {
            return Err("1: no layers: expected a `[layers]` section with `0 = [...]`".to_owned());
        }
        Ok(Layers { places })
    }

    /// Where `name` sits, if the table lists it.
    pub fn place(&self, name: &str) -> Option<Place> {
        self.places.get(name).copied()
    }

    /// Every listed name, in order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.places.keys().map(String::as_str)
    }
}

/// How a message names a place: `layer 3`, `dev-only` or `a tool`.
pub fn describe(place: Place) -> String {
    match place {
        Place::Layer(layer) => format!("layer {layer}"),
        Place::DevOnly => "dev-only".to_owned(),
        Place::Tool => "a tool outside the crate graph".to_owned(),
    }
}

/// `line` without a trailing `#` comment. Names never contain `#`, so the first one starts it.
fn strip_comment(line: &str) -> &str {
    line.split_once('#').map_or(line, |(code, _)| code)
}

/// `["a", "b",]` into its names. Each name is a crate name: ASCII letters, digits, `-` and `_`.
fn parse_array(value: &str) -> Result<Vec<String>, String> {
    let inner = value
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .ok_or_else(|| format!("expected an array of names, not `{value}`"))?;
    let mut names = Vec::new();
    let items: Vec<&str> = inner.split(',').map(str::trim).collect();
    for (index, item) in items.iter().enumerate() {
        if item.is_empty() && index + 1 == items.len() {
            break; // A trailing comma, or an empty array.
        }
        let name = item
            .strip_prefix('"')
            .and_then(|rest| rest.strip_suffix('"'))
            .ok_or_else(|| format!("expected a double-quoted name, not `{item}`"))?;
        let valid = !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
        if !valid {
            return Err(format!("`{name}` is not a crate name"));
        }
        names.push(name.to_owned());
    }
    Ok(names)
}

/// RFC-0001, whose §4.1 holds the table this file copies.
pub const RFC: &str = "plan/rfcs/RFC-0001-system-architecture.md";

/// The table of RFC-0001 §4.1: each row's layer cell, and the backquoted crate names in its crate
/// cell. `None` when the text has no §4.1.
pub fn rfc_table(rfc: &str) -> Option<Vec<(String, Place)>> {
    let section = rfc
        .split("### 4.1 Workspace")
        .nth(1)?
        .split("### 4.2")
        .next()?;
    let mut rows = Vec::new();
    for line in section.lines().filter(|line| line.starts_with("| ")) {
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        if cells.len() < 3 {
            continue;
        }
        let place = match cells[1] {
            "—" => Place::DevOnly,
            layer => match layer.parse() {
                Ok(layer) => Place::Layer(layer),
                Err(_) => continue, // the header row
            },
        };
        for name in cells[2].split('`').skip(1).step_by(2) {
            rows.push((name.to_owned(), place));
        }
    }
    Some(rows)
}

/// Every way `layers` differs from RFC-0001 §4.1 under `root`: a crate the RFC places elsewhere
/// or not at all, and a crate the table has that the RFC lacks, `Tool` entries apart. A checkout
/// without RFC-0001, such as a test fixture, has nothing to compare.
pub fn matches_rfc_0001(layers: &Layers, root: &Path) -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(root.join(RFC)) else {
        return Vec::new();
    };
    let Some(rows) = rfc_table(&text.replace("\r\n", "\n")) else {
        return vec![format!("{RFC} has no §4.1 table to check {PATH} against")];
    };
    let mut problems = Vec::new();
    for (name, place) in &rows {
        match layers.place(name) {
            Some(found) if found == *place => {}
            Some(found) => problems.push(format!(
                "`{name}` is {} in {PATH} but {} in RFC-0001 §4.1",
                describe(found),
                describe(*place)
            )),
            None => problems.push(format!("`{name}` is in RFC-0001 §4.1 but not in {PATH}")),
        }
    }
    for name in layers.names() {
        if layers.place(name) != Some(Place::Tool) && !rows.iter().any(|(n, _)| n == name) {
            problems.push(format!("`{name}` is in {PATH} but not in RFC-0001 §4.1"));
        }
    }
    problems
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layers_of(names: &[(&str, Place)]) -> Layers {
        Layers {
            places: names
                .iter()
                .map(|(name, place)| ((*name).to_owned(), *place))
                .collect(),
        }
    }

    #[test]
    fn parses_layers_and_the_outside_rows() {
        let text = r#"
# A comment.
[layers]
0 = ["core", "obs"] # trailing comment
1 = [
    "net",
    "io", # a comment inside the array
]

[outside]
dev-only = ["testkit"]
tools = ["xtask"]
"#;
        assert_eq!(
            Layers::parse(text).unwrap(),
            layers_of(&[
                ("core", Place::Layer(0)),
                ("obs", Place::Layer(0)),
                ("net", Place::Layer(1)),
                ("io", Place::Layer(1)),
                ("testkit", Place::DevOnly),
                ("xtask", Place::Tool),
            ])
        );
    }

    #[test]
    fn errors_name_the_line() {
        for (text, expected) in [
            (
                "[layers]\n0 = [\"a\"]\n2 = [\"b\"]\n",
                "3: layer 2 is out of order",
            ),
            (
                "[layers]\nzero = [\"a\"]\n",
                "2: a key in `[layers]` is a layer number",
            ),
            (
                "[layers]\n00 = [\"a\"]\n",
                "2: a key in `[layers]` is a layer number",
            ),
            ("[layers]\n0 = [\"a\", \"a\"]\n", "2: `a` is listed twice"),
            (
                "[layers]\n0 = [\"a\"]\n[outside]\ntools = [\"a\"]\n",
                "4: `a` is listed twice (also as layer 0)",
            ),
            ("0 = [\"a\"]\n", "1: `0` is outside a section"),
            ("[layer]\n", "1: unknown section `[layer]`"),
            (
                "[layers]\n[layers]\n",
                "2: section `[layers]` appears twice",
            ),
            ("[layers]\n0 = [a]\n", "2: expected a double-quoted name"),
            ("[layers]\n0 = [\"a b\"]\n", "2: `a b` is not a crate name"),
            ("[layers]\n0 = \"a\"\n", "no closing `]`"),
            (
                "[layers]\n0 = [\"a\",\n",
                "2: the array for `0` has no closing `]`",
            ),
            (
                "[layers]\n0 [\"a\"]\n",
                "2: expected `key = [\"name\", ...]`",
            ),
            ("[layers\n", "1: a section header ends with `]`"),
            (
                "[layers]\n0 = [\"a\"]\n[outside]\nspare = [\"b\"]\n",
                "4: unknown key `spare` in `[outside]`",
            ),
            (
                "[layers]\n0 = [\"a\"]\n[outside]\ntools = [\"b\"]\ntools = [\"c\"]\n",
                "5: `tools` appears twice",
            ),
            ("# only a comment\n", "1: no layers"),
        ] {
            let error = Layers::parse(text).unwrap_err();
            assert!(error.contains(expected), "`{text}` gave `{error}`");
        }
    }

    #[test]
    fn an_empty_array_and_a_trailing_comma_are_accepted() {
        let layers = Layers::parse("[layers]\n0 = [\"a\",]\n[outside]\ntools = []\n").unwrap();
        assert_eq!(layers, layers_of(&[("a", Place::Layer(0))]));
    }

    /// The table of RFC-0001 §4.1, read from the RFC itself.
    fn rfc_0001_table(root: &Path) -> Vec<(String, Place)> {
        let rfc = std::fs::read_to_string(root.join(RFC)).expect("RFC-0001 is readable");
        rfc_table(&rfc).expect("RFC-0001 has §4.1")
    }

    #[test]
    fn check_deps_compares_the_table_with_rfc_0001() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let layers = Layers::load(root).unwrap();
        assert_eq!(matches_rfc_0001(&layers, root), Vec::<String>::new());

        let dir =
            std::env::temp_dir().join(format!("syntropy-xtask-layers-rfc-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("plan/rfcs")).unwrap();
        // No RFC: a fixture, with nothing to compare.
        let table = Layers::parse(
            "[layers]\n0 = [\"core\"]\n1 = [\"net\"]\n[outside]\ntools = [\"xtask\"]\n",
        )
        .unwrap();
        assert_eq!(matches_rfc_0001(&table, &dir), Vec::<String>::new());
        std::fs::write(
            dir.join(RFC),
            "### 4.1 Workspace\n\n| Layer | Crates |\n|---|---|\n| 0 | `core`, `obs` |\n| 2 | `net` |\n\n### 4.2 Rules\n",
        )
        .unwrap();
        let problems = matches_rfc_0001(&table, &dir);
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(
            problems,
            [
                "`obs` is in RFC-0001 §4.1 but not in ci/layers.toml",
                "`net` is layer 1 in ci/layers.toml but layer 2 in RFC-0001 §4.1",
            ]
        );
    }

    #[test]
    fn the_real_table_matches_rfc_0001_section_4_1() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let layers = Layers::load(root).unwrap();
        let rfc = rfc_0001_table(root);
        assert!(
            rfc.len() >= 30,
            "read only {} crates from RFC-0001",
            rfc.len()
        );
        for (name, place) in &rfc {
            assert_eq!(
                layers.place(name),
                Some(*place),
                "`{name}` in RFC-0001 §4.1"
            );
        }
        let tools: Vec<&str> = layers
            .names()
            .filter(|name| layers.place(name) == Some(Place::Tool))
            .collect();
        assert_eq!(tools, ["xtask"]);
        assert_eq!(layers.names().count(), rfc.len() + tools.len());
    }

    #[test]
    fn every_crate_the_rules_name_is_in_the_real_table() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let layers = Layers::load(root).unwrap();
        for name in super::super::policy::named_crates() {
            assert!(layers.place(name).is_some(), "`{name}` is not in {PATH}");
        }
    }
}
