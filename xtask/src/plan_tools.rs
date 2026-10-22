// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `plan` step of `lint`: the task-graph check of AGENTS.md's gates, which runs the plan
//! tools as the opt-in pre-commit hook does (`plan/implementation/tools/README.md`).
//!
//! `epics.py` parses every task row, `depcheck.py` checks the task graph, and `deps_report.py`,
//! `trace.py` and `roadmap.py` with `--check` fail when a generated section is stale. None of
//! them writes a file. They are Python 3 with the standard library only, so the step needs
//! `python3` and nothing else.
//!
//! A diff that changes only `plan/` or `docs/` runs `lint` alone (RFC-0062 §2.2, note
//! 2026-10-23), so this step is what checks a plan-only PR's task graph in CI.

use std::process::Command;

use crate::run::Context;

/// The tools' directory, relative to the workspace root.
pub const TOOLS: &str = "plan/implementation/tools";

/// Each tool with its arguments, in the hook's order.
pub const RUNS: &[(&str, &[&str])] = &[
    ("epics.py", &[]),
    ("depcheck.py", &[]),
    ("deps_report.py", &["--check"]),
    ("trace.py", &["--check"]),
    ("roadmap.py", &["--check"]),
];

/// The Python the step runs: `$PYTHON`, else `python3`.
fn python() -> String {
    std::env::var("PYTHON").unwrap_or_else(|_| "python3".to_owned())
}

/// Runs every tool, and fails with the output of each that failed.
pub fn step(context: &Context) -> Result<(), String> {
    let python = python();
    let mut problems = Vec::new();
    for (tool, args) in RUNS {
        let script = context.root.join(TOOLS).join(tool);
        let shown = format!("{python} {TOOLS}/{tool} {}", args.join(" "));
        let shown = shown.trim_end();
        // Not `-I`: isolated mode leaves the script's directory off the module path, and the
        // tools import their siblings. They are this checkout's code, as every lint step is.
        let output = Command::new(&python)
            .arg(&script)
            .args(*args)
            .current_dir(&context.root)
            .output();
        match output {
            Ok(output) if output.status.success() => {}
            Ok(output) => {
                problems.push(format!("`{shown}` failed ({}):", output.status));
                for line in String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .chain(String::from_utf8_lossy(&output.stderr).lines())
                {
                    problems.push(format!("  {line}"));
                }
            }
            Err(error) => problems.push(format!("could not start `{shown}`: {error}")),
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        problems.push(format!(
            "run the generators that failed (python3 {TOOLS}/<tool>.py) and commit the result"
        ));
        Err(problems.join("\n"))
    }
}
