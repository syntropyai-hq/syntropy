// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `[logging]` section (RFC-0054 §5.2; RFC-0053 §4; PROD-E8-T04).
//!
//! Node scope, every key hot. The daemon reads it to choose its log level, filter, format and
//! file, and whether content may be logged. Like the `debug.*` keys, `logging.include_content =
//! true` is accepted with a warning, which a [`ResolvedConfig`] carries in
//! [`ResolvedConfig::warnings`], so the daemon prints it at start and on every reload
//! (RFC-0053 §4.3).

use std::fmt;
use std::path::PathBuf;

use serde::Deserialize;
use serde::de::{self, Deserializer, Visitor};

use crate::diagnostic::Problem;
use crate::resolve::ResolvedConfig;
// The traits and the derives, which share the names.
use crate::{ConfigSection, HoldsSecret};

/// The node's logging (RFC-0054 §5.2). Owner: RFC-0053.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, ConfigSection)]
#[config(section = "logging", scope = Node, owner = "RFC-0053")]
pub struct LoggingSection {
    /// The most detailed level logged: `error`, `warn`, `info`, `debug` or `trace`.
    #[config(reload = Hot, default = "info")]
    pub level: LogLevel,
    /// Per-target directives that override `logging.level`, comma-separated, such as
    /// `syntropy_net=debug,syntropy_api::server=trace`.
    #[config(reload = Hot, default = "")]
    pub filter: String,
    /// `auto` (human text on a terminal, the service manager's form otherwise), `text` or
    /// `json` (JSON Lines, one object per record).
    #[config(reload = Hot, default = "auto")]
    pub format: LogFormat,
    /// The rotating log file: `auto` (`<state>/logs/daemon.log` where RFC-0053 §4.2 keeps
    /// one), `off`, or a path.
    #[config(reload = Hot, default = "auto")]
    pub file: LogFile,
    /// Logs prompts, outputs and token IDs, at `debug` only. Off by default; when on, the
    /// daemon warns at start and on every reload (RFC-0053 §4.3).
    #[config(reload = Hot, default = false)]
    pub include_content: bool,
}

/// `logging.level`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize, HoldsSecret)]
#[serde(rename_all = "snake_case")]
pub enum LogLevel {
    /// Failures only.
    Error,
    /// Failures and warnings.
    Warn,
    /// Lifecycle, plans, membership, degradations and errors: the default.
    Info,
    /// Per-request summaries too.
    Debug,
    /// Per-step detail too.
    Trace,
}

/// `logging.format`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize, HoldsSecret)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    /// Human text on a terminal, with colour; each sink's own form otherwise.
    Auto,
    /// Human text.
    Text,
    /// JSON Lines.
    Json,
}

/// `logging.file`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, HoldsSecret)]
pub enum LogFile {
    /// `<state>/logs/daemon.log` where RFC-0053 §4.2 keeps a file.
    Auto,
    /// No file.
    Off,
    /// This file.
    Path(PathBuf),
}

impl<'de> Deserialize<'de> for LogFile {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct FileVisitor;

        impl Visitor<'_> for FileVisitor {
            type Value = LogFile;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("`auto`, `off` or a path")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<LogFile, E> {
                match value {
                    "auto" => Ok(LogFile::Auto),
                    "off" => Ok(LogFile::Off),
                    "" => Err(E::custom("an empty path; use `off` for no file")),
                    path => Ok(LogFile::Path(PathBuf::from(path))),
                }
            }
        }

        deserializer.deserialize_str(FileVisitor)
    }
}

/// The key that turns content logging on.
pub const INCLUDE_CONTENT: &str = "logging.include_content";

/// The warning for `logging.include_content = true`, at the layer that set it. Every
/// [`ResolvedConfig`] carries it, so a start and each reload repeat it.
pub(crate) fn warnings(config: &ResolvedConfig) -> Vec<Problem> {
    config
        .get(INCLUDE_CONTENT)
        .filter(|resolved| resolved.value() == Some(&toml::Value::Boolean(true)))
        .map(|resolved| {
            Problem::uncoded(
                Some(resolved.origin.clone()),
                "`logging.include_content` is on: prompts, outputs and token IDs are written to \
                 the logs at `debug` (RFC-0053 §4.3). Turn it off once you no longer need them"
                    .to_owned(),
            )
            .about(INCLUDE_CONTENT)
        })
        .into_iter()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Env, Flags, Layers, Os, Registry};

    static REGISTRY: Registry = Registry::new(&[&LoggingSection::SCHEMA]);

    fn resolve(pairs: &[(&str, &str)]) -> ResolvedConfig {
        let env = Env::from_pairs(Os::Linux, pairs.iter().copied());
        let layers = Layers {
            os: Os::Linux,
            files: &[],
            env: &env,
            flags: &Flags::new(),
        };
        ResolvedConfig::resolve(&REGISTRY, layers).unwrap()
    }

    #[test]
    fn the_defaults_are_rfc_0054s() {
        assert!(LoggingSection::check_defaults().is_empty());
        let config = resolve(&[]);
        let logging: LoggingSection = config.section().unwrap();
        assert_eq!(
            logging,
            LoggingSection {
                level: LogLevel::Info,
                filter: String::new(),
                format: LogFormat::Auto,
                file: LogFile::Auto,
                include_content: false,
            }
        );
        assert!(config.warnings().is_empty());
    }

    #[test]
    fn include_content_is_accepted_with_a_warning() {
        let config = resolve(&[
            ("SYNTROPY_LOGGING__INCLUDE_CONTENT", "true"),
            ("SYNTROPY_LOGGING__FILE", "/var/log/syntropy.log"),
            ("SYNTROPY_LOGGING__FORMAT", "json"),
        ]);
        let logging: LoggingSection = config.section().unwrap();
        assert!(logging.include_content);
        assert_eq!(
            logging.file,
            LogFile::Path(PathBuf::from("/var/log/syntropy.log"))
        );
        assert_eq!(logging.format, LogFormat::Json);
        assert_eq!(config.warnings().len(), 1);
        assert!(
            config.warnings()[0]
                .message
                .starts_with("`logging.include_content` is on")
        );
    }
}
