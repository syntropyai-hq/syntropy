// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask check-deps`: the dependency rules of RFC-0001 §4.2, checked over `cargo metadata`
//! (RFC-0062 §12.1, QUAL-E1-T03). It runs in the `lint` job.
//!
//! Each rule, and the dependency edges it reads:
//!
//! - **Layer table** (RFC-0001 §4.1): every member is named `syntropy-<name>` and is listed in
//!   `ci/layers.toml`, which lists only members.
//! - **Rule 1**: `core` depends on no workspace crate, and its only externals are `serde`,
//!   `thiserror`, `smallvec` and `bitflags`. Normal and build edges.
//! - **Rule 2**: every edge between members goes to a lower layer of `ci/layers.toml`. Every crate
//!   but `core` may use `obs` and `config`, every crate may use `core`, and tools have no workspace
//!   edges. Normal, build and dev edges: RFC-0062 §12.1 checks "every workspace edge".
//! - **Rule 3**: no compute crate (`graph`, `kernels`, every `kernels-*`, `io`, `tier`, `exec`,
//!   `sched`) reaches `net`, `cluster`, `store`, `api`, `daemon` or `tokio`. This also checks
//!   ADR-0012 §6 (no tokio in `exec`, `io` or `kernels-*`). Normal edges, transitively.
//! - **Rule 4**: `daemon` reaches no `kernels-*` crate. Normal edges, transitively.
//! - **Rule 5**: `solver` reaches no I/O crate of the workspace and no async-runtime, socket,
//!   filesystem or thread-pool crate. Normal edges, transitively.
//! - **Rule 6**: only each external system's own crate reaches its bindings ([`policy::SYSTEMS`]).
//!   Every edge kind, through external packages.
//! - **Rule 7**: the core binary, built with no default features, needs no shared library outside
//!   an allow-list of system libraries ([`policy::SYSTEM_LIBRARIES`] and the dynamic loader), so
//!   any vendor GPU library fails, named or not. As a second check, its dynamic symbol table holds
//!   no vendor runtime symbol. Linux only: QUAL-E1-T28 adds the Windows import table and the
//!   macOS load commands.
//! - **Rule 8**: every lib and bin crate root outside the `unsafe` allow-list sets
//!   `#![forbid(unsafe_code)]`.
//! - **Rule 9**: no production crate has a normal or build edge to a dev-only crate (`testkit`,
//!   `bench`).
//! - **Rule 10**: the features `cargo tree` resolves for each release crate's own build (normal and
//!   build edges) turn on no `fault-injection`, `debug-taps` or `guards` of a workspace crate. And
//!   no member's `default` feature, expanded through its `[features]` table, turns one on, its own
//!   or a member dependency's (QUAL-E3-T04).
//!
//! The graph is resolved with every feature on and for every platform, so an optional or
//! platform-specific dependency counts. Every finding starts with its rule's number, so a CI log
//! names the rule that failed. The fixture workspaces under `xtask/tests/fixtures/` break one rule
//! each.

mod crate_roots;
mod elf;
mod graph;
pub(crate) mod layers;
pub(crate) mod policy;
mod release;
mod rules;

use std::fmt;
use std::io::{self, Write};
use std::path::Path;

use graph::Graph;
use layers::Layers;
use release::Outcome;

/// Which rule a finding breaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rule {
    /// The layer table itself (RFC-0001 §4.1).
    Table,
    /// A numbered rule of RFC-0001 §4.2.
    Numbered(u8),
}

impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Rule::Table => write!(f, "layer table"),
            Rule::Numbered(number) => write!(f, "rule {number}"),
        }
    }
}

/// One violation.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Finding {
    pub rule: Rule,
    pub message: String,
}

impl Finding {
    pub fn new(rule: Rule, message: impl Into<String>) -> Finding {
        Finding {
            rule,
            message: message.into(),
        }
    }
}

impl fmt::Display for Finding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.rule, self.message)
    }
}

/// Checks the workspace of `manifest` and writes the findings and the verdict to `log`. Returns
/// whether every rule that ran passed.
pub fn run(manifest: &Path, log: &mut dyn Write) -> io::Result<bool> {
    writeln!(log, "check-deps: {}", manifest.display())?;
    log.flush()?;
    let graph = match Graph::load(manifest) {
        Ok(graph) => graph,
        Err(error) => {
            writeln!(log, "check-deps: FAIL: {error}")?;
            return Ok(false);
        }
    };
    let mut findings = Vec::new();
    let layers = match Layers::load(&graph.root) {
        Ok(layers) => Some(layers),
        Err(error) => {
            findings.push(Finding::new(
                Rule::Table,
                format!("{error}; rules 2 and 9 need the table and did not run"),
            ));
            None
        }
    };
    findings.extend(rules::check(&graph, layers.as_ref()));
    findings.extend(crate_roots::check(&graph));
    findings.extend(release::default_features(&graph));
    let mut not_run = Vec::new();
    for (number, outcome) in [
        (7, release::rule_7(manifest, &graph)),
        (10, release::rule_10(manifest, &graph)),
    ] {
        match outcome {
            Outcome::Ran(found) => findings.extend(found),
            Outcome::NotRun(reason) => not_run.push((Rule::Numbered(number), reason)),
        }
    }
    findings.sort();
    report(&findings, &not_run, log)?;
    Ok(findings.is_empty())
}

/// Writes one line per finding, then the verdict, then a line per rule that did not run.
fn report(findings: &[Finding], not_run: &[(Rule, String)], log: &mut dyn Write) -> io::Result<()> {
    for finding in findings {
        writeln!(log, "check-deps: {finding}")?;
    }
    if findings.is_empty() {
        let skipped: Vec<String> = not_run.iter().map(|(rule, _)| rule.to_string()).collect();
        if skipped.is_empty() {
            writeln!(
                log,
                "check-deps: PASS: the layer table and rules 1-10 of RFC-0001 §4.2"
            )?;
        } else {
            writeln!(
                log,
                "check-deps: PASS: the layer table and rules 1-10 of RFC-0001 §4.2, except {}",
                skipped.join(" and ")
            )?;
        }
    } else {
        let mut rules: Vec<String> = findings.iter().map(|f| f.rule.to_string()).collect();
        rules.dedup();
        writeln!(
            log,
            "check-deps: FAIL: {} {} ({})",
            findings.len(),
            if findings.len() == 1 {
                "violation"
            } else {
                "violations"
            },
            rules.join(", ")
        )?;
    }
    for (rule, reason) in not_run {
        writeln!(log, "check-deps: {rule} not run: {reason}")?;
    }
    log.flush()
}

/// `a, b and c`, each name in backquotes, with `conjunction` before the last.
fn list(names: &[&str], conjunction: &str) -> String {
    let quoted: Vec<String> = names.iter().map(|name| format!("`{name}`")).collect();
    match quoted.split_last() {
        Some((last, rest)) if !rest.is_empty() => {
            format!("{} {conjunction} {last}", rest.join(", "))
        }
        _ => quoted.concat(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reported(findings: &[Finding], not_run: &[(Rule, String)]) -> String {
        let mut log = Vec::new();
        report(findings, not_run, &mut log).unwrap();
        String::from_utf8(log).unwrap()
    }

    #[test]
    fn findings_sort_by_rule_number_then_message() {
        let mut findings = [
            Finding::new(Rule::Numbered(10), "b"),
            Finding::new(Rule::Numbered(2), "z"),
            Finding::new(Rule::Table, "t"),
            Finding::new(Rule::Numbered(2), "a"),
        ];
        findings.sort();
        let shown: Vec<String> = findings.iter().map(Finding::to_string).collect();
        assert_eq!(
            shown,
            ["layer table: t", "rule 2: a", "rule 2: z", "rule 10: b"]
        );
    }

    #[test]
    fn a_failure_names_each_finding_and_the_rules_broken() {
        let findings = [
            Finding::new(Rule::Numbered(3), "one"),
            Finding::new(Rule::Numbered(3), "two"),
            Finding::new(Rule::Numbered(8), "three"),
        ];
        assert_eq!(
            reported(&findings, &[]),
            "check-deps: rule 3: one\n\
             check-deps: rule 3: two\n\
             check-deps: rule 8: three\n\
             check-deps: FAIL: 3 violations (rule 3, rule 8)\n"
        );
        assert_eq!(
            reported(&findings[..1], &[]),
            "check-deps: rule 3: one\ncheck-deps: FAIL: 1 violation (rule 3)\n"
        );
    }

    #[test]
    fn a_pass_says_which_rules_did_not_run() {
        assert_eq!(
            reported(&[], &[]),
            "check-deps: PASS: the layer table and rules 1-10 of RFC-0001 §4.2\n"
        );
        assert_eq!(
            reported(&[], &[(Rule::Numbered(7), "not on Linux".into())]),
            "check-deps: PASS: the layer table and rules 1-10 of RFC-0001 §4.2, except rule 7\n\
             check-deps: rule 7 not run: not on Linux\n"
        );
    }

    #[test]
    fn lists_read_as_prose() {
        assert_eq!(list(&["a"], "and"), "`a`");
        assert_eq!(list(&["a", "b"], "or"), "`a` or `b`");
        assert_eq!(list(&["a", "b", "c"], "and"), "`a`, `b` and `c`");
        assert_eq!(list(&[], "and"), "");
    }
}
