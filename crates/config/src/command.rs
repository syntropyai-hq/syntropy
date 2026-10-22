// SPDX-License-Identifier: MIT OR Apache-2.0

//! What `syntropy config show --resolved`, `config check` and `config path` print (RFC-0054
//! §2.2, §8; RFC-0051 §2.1). Each is a function of resolved data, so the CLI wires it to a
//! subcommand and the tests check the text without a binary.
//!
//! `check` and `path` work without a daemon (RFC-0051 §1).

use std::fmt::Write as _;

use crate::discovery::{Locations, Mode, Scan};
use crate::origin::Origin;
use crate::registry::Registry;
use crate::resolve::ResolvedConfig;
use crate::schema::Scope;
use crate::validate::Checked;

/// The exit status of a command that succeeded (RFC-0051 §2.2).
pub const EXIT_OK: u8 = 0;
/// The exit status for an invalid configuration (RFC-0051 §2.2, code 5).
pub const EXIT_INVALID: u8 = 5;

/// `config show --resolved`: every node and cluster key with its effective value and the layer
/// it came from, then every instance key with its schema default, then the cluster values this
/// node proposes. Each block's header names its keys' scope. One line per key:
///
/// ```text
/// api.bind = ["0.0.0.0:7433"]        # /etc/syntropy/config.toml:4
/// logging.level = "debug"            # flag --set logging.level
/// resource.ram_headroom = "16GiB"    # env SYNTROPY_RESOURCE__RAM_HEADROOM
/// node.name                          # default, computed: OS host name, lower-cased
/// ```
///
/// A key with no value (an unset `Option` key, or a computed default not yet computed) prints
/// its path alone. Secrets print redacted.
pub fn show_resolved(config: &ResolvedConfig) -> String {
    let row = |resolved: &crate::Resolved| {
        let left = match resolved.shown() {
            Some(value) => format!("{} = {value}", resolved.key.path),
            None => resolved.key.path.to_owned(),
        };
        (left, resolved.origin.to_string())
    };
    let mut node = Vec::new();
    let mut cluster = Vec::new();
    for resolved in config.iter() {
        match resolved.key.scope {
            Scope::Cluster => cluster.push(row(resolved)),
            _ => node.push(row(resolved)),
        }
    }
    let instance: Vec<(String, String)> = config.instance_defaults().map(row).collect();
    let proposals: Vec<(String, String)> = config
        .proposals()
        .iter()
        .map(|p| (format!("{} = {}", p.path, p.shown()), p.origin.to_string()))
        .collect();

    let width = node
        .iter()
        .chain(&cluster)
        .chain(&instance)
        .chain(&proposals)
        .map(|(left, _)| left.chars().count())
        .max()
        .unwrap_or(0)
        .min(56);
    let mut out = String::new();
    let files = config.files().len();
    let _ = writeln!(
        out,
        "# node keys: default < config file < environment < flags ({files} file{} read)",
        if files == 1 { "" } else { "s" }
    );
    rows(&mut out, &node, width);
    if !cluster.is_empty() {
        out.push_str(
            "\n# cluster keys: the cluster's value; the default until the cluster sets one\n",
        );
        rows(&mut out, &cluster, width);
    }
    if !instance.is_empty() {
        out.push_str(
            "\n# instance keys: the default; a model's `[policy.\"<model>\"]` table or its load sets them\n",
        );
        rows(&mut out, &instance, width);
    }
    if !proposals.is_empty() {
        out.push_str("\n# proposed to the cluster by this node's file, environment or flags\n");
        rows(&mut out, &proposals, width);
    }
    for warning in config.warnings() {
        let _ = writeln!(out, "\nwarning: {warning}");
    }
    out
}

fn rows(out: &mut String, rows: &[(String, String)], width: usize) {
    for (left, origin) in rows {
        let _ = writeln!(out, "{left:<width$}  # {origin}");
    }
}

/// What `config check` reports: the text, and the exit status.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckReport {
    /// Every error and warning in RFC-0054 §8's format, then a summary line.
    pub text: String,
    /// [`EXIT_OK`] when there is no error, else [`EXIT_INVALID`].
    pub exit: u8,
}

/// `config check`: every error and warning of one load and validation, from
/// [`crate::load_and_validate`], rendered in RFC-0054 §8's format with their source lines, and
/// never showing a secret.
pub fn check(checked: &Checked, registry: &Registry) -> CheckReport {
    let (errors, warnings) = (checked.errors(), checked.warnings());
    let files = checked
        .outcome
        .as_ref()
        .map_or(0, |config: &ResolvedConfig| config.files().len());
    let mut text = checked.render(registry);
    let plural = |n: usize, word: &str| format!("{n} {word}{}", if n == 1 { "" } else { "s" });
    if errors.is_empty() {
        let _ = writeln!(
            text,
            "config check: ok ({} read, {})",
            plural(files, "file"),
            plural(warnings.len(), "warning")
        );
    } else {
        let _ = writeln!(
            text,
            "config check: {}, {}",
            plural(errors.len(), "error"),
            plural(warnings.len(), "warning")
        );
    }
    CheckReport {
        text,
        exit: if errors.is_empty() {
            EXIT_OK
        } else {
            EXIT_INVALID
        },
    }
}

/// `config path`: the files of the config layer in merge order, whether each was found, and
/// the state, store and runtime locations.
pub fn path(locations: &Locations, scan: &Scan) -> String {
    let mut out = String::from("config files, in merge order (later wins):\n");
    let replaced = locations.explicit.is_some();
    let found = |present: bool| if present { "found" } else { "absent" };
    let status_of = |present: bool| {
        if replaced {
            "replaced by the explicit file"
        } else {
            found(present)
        }
    };
    let _ = writeln!(
        out,
        "  1  system file    {}  ({})",
        locations.system_file.display(),
        status_of(scan.system_file.is_some())
    );
    let dropins = if replaced {
        String::from("replaced by the explicit file")
    } else if scan.dropins.is_empty() {
        String::from("none")
    } else {
        let names: Vec<String> = scan
            .dropins
            .iter()
            .filter_map(|p| p.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .collect();
        names.join(", ")
    };
    let _ = writeln!(
        out,
        "  2  drop-ins       {}  ({dropins})",
        locations
            .os
            .join(locations.dropin_dir.as_os_str(), &["*.toml"])
            .display()
    );
    match (&locations.user_file, locations.mode) {
        (_, Mode::Service) => {
            let _ = writeln!(out, "  3  user file      (not read by a system service)");
        }
        (None, Mode::User) => {
            let _ = writeln!(
                out,
                "  3  user file      (no home directory in the environment)"
            );
        }
        (Some(user), Mode::User) => {
            let _ = writeln!(
                out,
                "  3  user file      {}  ({})",
                user.display(),
                status_of(scan.user_file.is_some())
            );
        }
    }
    match &locations.explicit {
        Some(explicit) => {
            let from = match &explicit.from {
                Origin::Cli { flag } => flag.clone(),
                Origin::Env { var } => var.clone(),
                other => other.to_string(),
            };
            let _ = writeln!(
                out,
                "  4  explicit file  {}  (from {from}; {})",
                explicit.path.display(),
                found(scan.explicit.is_some())
            );
        }
        None => {
            let _ = writeln!(
                out,
                "  4  explicit file  (none; `--config` or SYNTROPY_CONFIG replaces 1-3)"
            );
        }
    }
    let dirs = &locations.dirs;
    let show = |p: &Option<std::path::PathBuf>| {
        p.as_ref().map_or_else(
            || String::from("(not set: no home directory in the environment)"),
            |p| p.display().to_string(),
        )
    };
    out.push_str("directories:\n");
    let _ = writeln!(out, "  state    {}", show(&dirs.state));
    let _ = writeln!(out, "  store    {}", show(&dirs.store));
    let runtime = dirs.runtime.as_ref().map_or_else(
        || String::from("(not set: no runtime directory in the environment)"),
        ToString::to_string,
    );
    let _ = writeln!(out, "  runtime  {runtime}");
    if let Some(home) = &dirs.home {
        let _ = writeln!(
            out,
            "  (state and store from SYNTROPY_HOME={})",
            home.display()
        );
    }
    out
}
