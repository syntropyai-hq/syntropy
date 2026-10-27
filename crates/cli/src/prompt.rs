// SPDX-License-Identifier: MIT OR Apache-2.0

//! Questions (RFC-0051 §2.2): asked only on a terminal. Without one, a command that would ask
//! fails with exit 2 and names the flag that answers it.

use std::io::{BufRead, Write};

use crate::error::{CliError, Exit};
use crate::output::Terminal;

/// Asks `question` on stderr and reads a yes or no from stdin, or fails with exit 2 naming
/// `flag` when no question may be asked: stdin or stderr is not a terminal, or `TERM=dumb`.
///
/// # Errors
///
/// The exit-2 error above, or a failure to read stdin.
pub fn confirm(terminal: Terminal, question: &str, flag: &str) -> Result<bool, CliError> {
    if !terminal.can_ask() {
        return Err(no_terminal(question, flag));
    }
    let stdin = std::io::stdin();
    ask(&mut stdin.lock(), &mut std::io::stderr(), question)
}

/// The error for a question asked where none may be.
fn no_terminal(question: &str, flag: &str) -> CliError {
    CliError::cli(
        Exit::Usage,
        "needs_answer",
        format!("this needs an answer, and there is no terminal to ask: {question}"),
    )
    .hint(format!("pass {flag} to answer it"))
}

/// Asks on `out` and reads one line from `input`: `y` or `yes` is yes, anything else no.
fn ask(input: &mut impl BufRead, out: &mut impl Write, question: &str) -> Result<bool, CliError> {
    let failed = |error: std::io::Error| {
        CliError::cli(Exit::Failure, "terminal", format!("cannot ask: {error}"))
    };
    write!(out, "{question} [y/N] ").map_err(failed)?;
    out.flush().map_err(failed)?;
    let mut line = String::new();
    input.read_line(&mut line).map_err(failed)?;
    Ok(matches!(
        line.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TTY: Terminal = Terminal {
        stdin: true,
        stdout: true,
        stderr: true,
        no_color: false,
        dumb: false,
    };

    #[test]
    fn without_a_terminal_a_question_fails_with_exit_2_naming_the_flag() {
        for terminal in [
            Terminal {
                stdin: false,
                ..TTY
            },
            Terminal { dumb: true, ..TTY },
        ] {
            let error = confirm(terminal, "remove the model?", "--yes").unwrap_err();
            assert_eq!(error.exit, Exit::Usage);
            assert_eq!(error.hints, ["pass --yes to answer it"]);
        }
    }

    #[test]
    fn an_answer_is_read_from_one_line() {
        for (line, yes) in [
            ("y\n", true),
            ("YES\n", true),
            ("n\n", false),
            ("\n", false),
            ("", false),
        ] {
            let mut out = Vec::new();
            let answer = ask(&mut line.as_bytes(), &mut out, "go on?").unwrap();
            assert_eq!(answer, yes, "{line:?}");
            assert_eq!(out, b"go on? [y/N] ");
        }
    }
}
