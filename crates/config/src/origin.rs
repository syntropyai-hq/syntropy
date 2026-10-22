// SPDX-License-Identifier: MIT OR Apache-2.0

//! Where a resolved value came from (RFC-0054 §2.2, RFC-0001 §11.2).

use std::fmt;
use std::path::PathBuf;

/// The layer that set a resolved value. `syntropy config show --resolved` prints one per key.
///
/// RFC-0054 §2.2 also lists `ClusterLog`, `Explicit` and `Request`. They belong to the cluster,
/// instance and request scopes, and join this enum with the tasks that resolve those scopes.
/// Node keys, which this crate resolves today, come only from the four variants here.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Origin {
    /// The schema's default. `computed` is the rule for a default computed at start, as the
    /// schema writes it, such as `"max(6GiB, 10% of RAM)"`.
    Default {
        /// The rule of a computed default, or `None` for a fixed one.
        computed: Option<&'static str>,
    },
    /// A config file, at the line where the key is written.
    File {
        /// The file.
        path: PathBuf,
        /// The 1-based line of the key.
        line: u32,
    },
    /// An environment variable, such as `SYNTROPY_RESOURCE__RAM_HEADROOM`.
    Env {
        /// The variable's name.
        var: String,
    },
    /// A command-line flag, such as `--set resource.ram_headroom` or `--bind`.
    Cli {
        /// The flag, with the key for `--set`.
        flag: String,
    },
}

impl fmt::Display for Origin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Origin::Default { computed: None } => f.write_str("default"),
            Origin::Default {
                computed: Some(rule),
            } => write!(f, "default, computed: {rule}"),
            Origin::File { path, line } => write!(f, "{}:{line}", path.display()),
            Origin::Env { var } => write!(f, "env {var}"),
            Origin::Cli { flag } => write!(f, "flag {flag}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prints_file_line_env_var_and_flag() {
        let cases = [
            (Origin::Default { computed: None }, "default"),
            (
                Origin::Default {
                    computed: Some("max(6GiB, 10% of RAM)"),
                },
                "default, computed: max(6GiB, 10% of RAM)",
            ),
            (
                Origin::File {
                    path: PathBuf::from("/etc/syntropy/config.toml"),
                    line: 14,
                },
                "/etc/syntropy/config.toml:14",
            ),
            (
                Origin::Env {
                    var: String::from("SYNTROPY_API__BIND"),
                },
                "env SYNTROPY_API__BIND",
            ),
            (
                Origin::Cli {
                    flag: String::from("--set logging.level"),
                },
                "flag --set logging.level",
            ),
        ];
        for (origin, text) in cases {
            assert_eq!(origin.to_string(), text);
        }
    }
}
