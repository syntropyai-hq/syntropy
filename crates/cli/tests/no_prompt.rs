// SPDX-License-Identifier: MIT OR Apache-2.0

//! PROD-E6-T01: every command, run with stdin closed and `TERM=dumb`, never prompts (RFC-0051
//! §2.2, Testing).
//!
//! The test walks the whole command tree from `syntropy_cli::args::command()`, so a command added
//! later is covered without editing this file. Each command runs with stdin on the null device
//! (a read sees end of file at once) and `TERM=dumb`, and `SYNTROPY_HOST` aimed at a
//! closed loopback port, so no command can reach a real daemon. It must exit within the hang
//! limit, print no question, and exit with a code from RFC-0051 §2.2's table. Each runs
//! [`common::isolated`] in a directory of its own, so the `config` commands that write, and
//! `config edit`'s editor, never touch this machine's files.

mod common;

use common::{closed_addr, code, finish, isolated};

/// Every leaf of the tree as an argument list, with a value for each required argument.
fn leaves(command: &clap::Command, prefix: &[String], out: &mut Vec<Vec<String>>) {
    let subcommands: Vec<&clap::Command> = command.get_subcommands().collect();
    if subcommands.is_empty() {
        let mut args = prefix.to_vec();
        for arg in command.get_arguments().filter(|a| a.is_required_set()) {
            if let Some(long) = arg.get_long() {
                args.push(format!("--{long}"));
            }
            // A value the hidden modes parse: an endpoint, or a directory.
            args.push(if arg.get_id() == "ipc" {
                "socket:/nonexistent/cmd.sock".to_owned()
            } else {
                "/nonexistent".to_owned()
            });
        }
        out.push(args);
        return;
    }
    for sub in subcommands {
        if sub.get_name() == "help" {
            continue;
        }
        let mut next = prefix.to_vec();
        next.push(sub.get_name().to_owned());
        leaves(sub, &next, out);
    }
}

#[test]
fn every_command_with_stdin_closed_and_term_dumb_never_prompts() {
    let mut all = Vec::new();
    leaves(&syntropy_cli::args::command(), &[], &mut all);
    // The whole tree of RFC-0051 §2.1, with the hidden modes.
    assert!(all.len() > 60, "{} leaves", all.len());
    assert!(all.iter().any(|a| a[0] == "__runner") && all.iter().any(|a| a[0] == "__selftest"));
    let host = format!("http://{}", closed_addr());
    let home = common::temp("no-prompt");
    let table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 130];
    for args in &all {
        for json in [false, true] {
            let mut args: Vec<&str> = args.iter().map(String::as_str).collect();
            if json {
                args.push("--json");
            }
            let output = finish(
                isolated(&args, &home)
                    .env("TERM", "dumb")
                    .env("SYNTROPY_HOST", &host),
            );
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                !stderr.contains("[y/N]") && !stdout.contains("[y/N]"),
                "{args:?} asked a question: {stderr}"
            );
            assert!(
                table.contains(&code(&output)),
                "{args:?} exited {}",
                code(&output)
            );
            // No command prints colour under `TERM=dumb`.
            assert!(
                !stdout.contains('\u{1b}') && !stderr.contains('\u{1b}'),
                "{args:?}"
            );
        }
    }
}
