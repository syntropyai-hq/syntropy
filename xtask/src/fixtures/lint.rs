// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `fixtures` step of `lint`: `testdata/fixtures.toml` follows the licence rule of RFC-0060
//! §12.2 and ADR-0603 decision 4 (QUAL-E7-T06).
//!
//! The step fails unless:
//! - every `nightly` fixture is ungated and under an OSI-approved licence, one of [`OSI`].
//!   Llama-family checkpoints, gated ones and anything under a licence outside the list are
//!   `lab` fixtures;
//! - every `nightly` fixture is imported: it has its `blake3` and `bytes`. Only a `lab` fixture
//!   may be recorded before its bytes are in a store, as a checkpoint's weights wait for the lab
//!   store;
//! - every fixture taken from the Hub names its repository, a full 40-hex commit as its
//!   revision, and its file, so what was confirmed on the model card is what is fetched;
//! - every fixture quantised from original weights (it has a `quant`) records the original
//!   weights it came from in `quantised_from`, as `<hub repo>@<40-hex revision>`.

use std::fs;

use super::manifest::{Fixture, Manifest, Tier};
use crate::run::Context;

/// The manifest, relative to the workspace root.
pub const MANIFEST: &str = "testdata/fixtures.toml";

/// The licences a nightly fixture may carry: OSI-approved licences, by their Hub identifiers.
/// A licence joins the list when a fixture needs it and its OSI approval is checked.
pub const OSI: &[&str] = &[
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "bsl-1.0",
    "isc",
    "mpl-2.0",
    "zlib",
];

fn is_commit(text: &str) -> bool {
    text.len() == 40 && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The problems with one fixture.
fn fixture_problems(f: &Fixture) -> Vec<String> {
    let mut out = Vec::new();
    let at = format!("{MANIFEST}: fixture `{}`", f.name);
    if f.tier == Tier::Nightly {
        if f.gated {
            out.push(format!(
                "{at} is gated, so it is a `lab` fixture, not `nightly`"
            ));
        }
        if f.blake3.is_empty() || f.bytes == 0 {
            out.push(format!(
                "{at} is `nightly` but not imported: it has no `blake3` or `bytes`"
            ));
        }
        if !OSI.contains(&f.licence.as_str()) {
            out.push(format!(
                "{at}: licence `{}` is not an OSI-approved licence a nightly fixture may carry \
                 ({})",
                f.licence,
                OSI.join(", ")
            ));
        }
    }
    if let Some(hub) = f.source.get("hub") {
        let revision = f
            .source
            .get("revision")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if hub.as_str().is_none_or(str::is_empty) {
            out.push(format!("{at}: `source.hub` is empty"));
        }
        if !is_commit(revision) {
            out.push(format!(
                "{at}: `source.revision` `{revision}` is not a 40-hex commit"
            ));
        }
        if f.source
            .get("file")
            .and_then(|v| v.as_str())
            .is_none_or(str::is_empty)
        {
            out.push(format!("{at}: `source.file` is missing"));
        }
    }
    match (&f.quant, &f.quantised_from) {
        (Some(_), None) => out.push(format!(
            "{at} is quantised but does not record `quantised_from`, the original weights"
        )),
        (_, Some(from)) => {
            let ok = from
                .rsplit_once('@')
                .is_some_and(|(repo, rev)| !repo.is_empty() && is_commit(rev));
            if !ok {
                out.push(format!(
                    "{at}: `quantised_from` `{from}` is not `<hub repo>@<40-hex revision>`"
                ));
            }
        }
        (None, None) => {}
    }
    out
}

/// Every problem in the manifest `text`.
pub fn check(text: &str) -> Vec<String> {
    match Manifest::parse(text) {
        Err(e) => vec![format!("{MANIFEST}: {e}")],
        Ok(m) => m.fixtures.iter().flat_map(fixture_problems).collect(),
    }
}

/// The `fixtures` lint step.
pub fn step(context: &Context) -> Result<(), String> {
    let path = context.root.join(MANIFEST);
    let text = fs::read_to_string(&path).map_err(|e| format!("{MANIFEST}: {e}"))?;
    let problems = check(&text);
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REV: &str = "c1899de289a04d12100db370d81485cdf75e47ca";

    /// An imported entry: it has a `blake3` and `bytes`.
    fn entry(extra: &str) -> String {
        format!(
            "[[fixture]]\nname = \"m\"\nblake3 = \"{}\"\nbytes = 7\nsource = {{ hub = \"org/model\", \
             revision = \"{REV}\", file = \"model.safetensors\" }}\nlicence = \"apache-2.0\"\n\
             gated = false\n{extra}\n",
            "ab".repeat(32)
        )
    }

    #[test]
    fn an_unimported_nightly_fixture_fails_and_an_unimported_lab_one_passes() {
        let unimported = entry("tier = \"nightly\"").replace(
            &format!("blake3 = \"{}\"\nbytes = 7\n", "ab".repeat(32)),
            "",
        );
        let problems = check(&unimported);
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(problems[0].contains("not imported"), "{problems:?}");
        let lab = unimported.replace("tier = \"nightly\"", "tier = \"lab\"");
        assert_eq!(check(&lab), Vec::<String>::new());
    }

    #[test]
    fn the_checked_in_manifest_passes() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let text = fs::read_to_string(root.join(MANIFEST)).unwrap();
        assert_eq!(check(&text), Vec::<String>::new());
    }

    #[test]
    fn an_ungated_osi_nightly_fixture_passes() {
        assert_eq!(check(&entry("tier = \"nightly\"")), Vec::<String>::new());
    }

    #[test]
    fn a_gated_nightly_fixture_fails() {
        let text = entry("tier = \"nightly\"").replace("gated = false", "gated = true");
        let problems = check(&text);
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(problems[0].contains("is gated"), "{problems:?}");
        // The same entry as a lab fixture passes.
        let lab = entry("tier = \"lab\"").replace("gated = false", "gated = true");
        assert_eq!(check(&lab), Vec::<String>::new());
    }

    #[test]
    fn a_nightly_fixture_under_a_licence_that_is_not_osi_approved_fails() {
        for licence in ["llama3.2", "gemma", "other", "cc-by-nc-4.0", ""] {
            let text = entry("tier = \"nightly\"").replace(
                "licence = \"apache-2.0\"",
                &format!("licence = \"{licence}\""),
            );
            let problems = check(&text);
            assert_eq!(problems.len(), 1, "{licence}: {problems:?}");
            assert!(problems[0].contains("not an OSI-approved"), "{problems:?}");
            let lab = text.replace("tier = \"nightly\"", "tier = \"lab\"");
            assert_eq!(check(&lab), Vec::<String>::new(), "{licence}");
        }
    }

    #[test]
    fn a_hub_fixture_needs_a_pinned_commit_and_a_file() {
        let branch = entry("tier = \"nightly\"").replace(REV, "main");
        assert!(check(&branch)[0].contains("not a 40-hex commit"));
        let no_file = entry("tier = \"nightly\"").replace(", file = \"model.safetensors\"", "");
        assert!(check(&no_file)[0].contains("`source.file` is missing"));
    }

    #[test]
    fn a_quantised_fixture_records_the_original_weights_it_came_from() {
        let missing = entry("tier = \"nightly\"\nquant = \"q4_k\"");
        assert!(check(&missing)[0].contains("does not record `quantised_from`"));
        let ok = entry(&format!(
            "tier = \"nightly\"\nquant = \"q4_k\"\nquantised_from = \"Qwen/Qwen3-0.6B@{REV}\""
        ));
        assert_eq!(check(&ok), Vec::<String>::new());
        let bad = entry("tier = \"nightly\"\nquant = \"q4_k\"\nquantised_from = \"Qwen3@main\"");
        assert!(check(&bad)[0].contains("not `<hub repo>@<40-hex revision>`"));
    }

    #[test]
    fn a_fixture_without_a_tier_does_not_parse() {
        assert!(check(&entry(""))[0].contains("tier"));
    }
}
