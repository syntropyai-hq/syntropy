// SPDX-License-Identifier: MIT OR Apache-2.0

//! The command tree of RFC-0051 §2.1, with the hidden modes `__runner` (RFC-0001 §3.1) and
//! `__selftest` (RFC-0052 §7.3).
//!
//! Every command of the table is declared, so `--help` shows the whole tree from the start. A
//! command whose task has not built it takes its arguments as written, in a [`Later`], and fails
//! with "not available in this build yet" (exit 5). Its task gives it typed arguments when it
//! builds it.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};
use syntropy_ipc::cmd::ChannelHandle;
use syntropy_ipc::shm::RegionHandle;

/// `syntropy`: one binary for the daemon, its runners and every client command (ADR-0015).
#[derive(Debug, Parser)]
///
/// `-V` and `--version` are this binary's own flag, not clap's, so `--version --json` can print
/// the build as JSON (QUAL-E16-T03). Like clap's propagated flag, every command takes it. Without
/// it a command is required.
#[command(name = "syntropy", about, disable_version_flag = true)]
pub struct Cli {
    /// Print this binary's version and build, with `--json` as the native API's `Build`
    #[arg(short = 'V', long = "version", global = true)]
    pub version: bool,
    /// The options every command takes.
    #[command(flatten)]
    pub global: Global,
    /// The command.
    #[command(subcommand)]
    pub command: Option<Command>,
}

/// The options every command takes (RFC-0051 §2.2).
#[derive(Clone, Debug, Args)]
pub struct Global {
    /// Print the native API's JSON document instead of text (JSON Lines for streams)
    #[arg(long, global = true)]
    pub json: bool,
    /// When to colour the output
    #[arg(long, global = true, value_enum, default_value_t = ColorWhen::Auto)]
    pub color: ColorWhen,
    /// Print byte counts as exact integers
    #[arg(long, global = true)]
    pub bytes: bool,
}

/// `--color`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ColorWhen {
    /// Colour on a terminal, unless `NO_COLOR` is set or `TERM=dumb`.
    Auto,
    /// Always colour.
    Always,
    /// Never colour.
    Never,
}

/// The arguments of a command its task has not built yet, kept as written.
#[derive(Clone, Debug, Default, Args)]
pub struct Later {
    /// The command's arguments
    #[arg(trailing_var_arg = true, allow_hyphen_values = true, hide = true)]
    pub args: Vec<String>,
}

/// The commands of RFC-0051 §2.1.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run the node daemon in the foreground
    Serve(Later),
    /// Load a model if needed, then chat, or answer one prompt
    Run(Later),
    /// Chat with a model (`run` with no prompt)
    Chat(Later),
    /// Register and fetch a model
    Pull(Later),
    /// Models in the cluster's stores
    #[command(visible_aliases = ["models", "list"])]
    Ls(Later),
    /// Loaded instances
    Ps(Later),
    /// A model's manifest summary and census
    Show(Later),
    /// Preview a placement without loading
    Plan(Later),
    /// Drain and unload a model or instance
    Unload(Later),
    /// Unload a model, then delete its chunks and packs
    Rm(Later),
    /// Drain every runner, then stop the local daemon
    Stop(Later),
    /// Cluster membership and admission
    #[command(subcommand)]
    Cluster(ClusterCommand),
    /// Invite tokens
    #[command(subcommand)]
    Invite(InviteCommand),
    /// Run the benchmark scenarios
    Bench(Later),
    /// The node's capability profile and feature bits
    Probe(Later),
    /// Diagnostics
    Doctor(Later),
    /// Daemon and runner logs
    Logs(Later),
    /// Debug state, parity and telemetry
    #[command(subcommand)]
    Debug(DebugCommand),
    /// Configuration
    #[command(subcommand)]
    Config(ConfigCommand),
    /// API keys
    #[command(subcommand)]
    ApiKey(ApiKeyCommand),
    /// This node's identity
    #[command(subcommand)]
    Identity(IdentityCommand),
    /// Trust a node's local CA on this machine
    Trust(Later),
    /// Open the dashboard
    Dashboard(Later),
    /// First-run setup
    Setup(Later),
    /// Service manager integration
    #[command(subcommand)]
    Service(ServiceCommand),
    /// Updates
    #[command(subcommand)]
    Update(UpdateCommand),
    /// Tune the GPU memory limit
    Tune(Later),
    /// Revert every host change and remove Syntropy
    Uninstall(Later),
    /// Local provenance records
    #[command(subcommand)]
    Provenance(ProvenanceCommand),
    /// The menu-bar or tray helper
    Tray(Later),
    /// Shell completions
    Completions(Later),
    /// Build ID, protocol versions and support tier
    Version,
    /// Runner mode, which the daemon starts for each model instance (ADR-0015 decision 1)
    #[command(name = "__runner", hide = true)]
    Runner(RunnerArgs),
    /// A staged build's self-test, which the updater runs (RFC-0052 §7.3)
    #[command(name = "__selftest", hide = true)]
    Selftest(SelftestArgs),
    /// A newly added model file's header, parsed in a confined child (RFC-0034 §10.4)
    #[command(name = "__header", hide = true)]
    Header(HeaderArgs),
}

/// `syntropy __header --kind <kind> --path <file> --memory <bytes> --cpu-secs <n> --open-files
/// <n> --max-header-bytes <bytes>`: parse one file's header under the limits, write the file's
/// length and the validated header's bytes to stdout, and exit 0; exit 8 when the header is
/// refused, or 5 when the child cannot be confined (DIST-E17-T16).
#[derive(Clone, Debug, Args)]
pub struct HeaderArgs {
    /// The file's format: `gguf` or `safetensors`
    #[arg(long, value_name = "KIND")]
    pub kind: HeaderKind,
    /// The file
    #[arg(long, value_name = "PATH", allow_hyphen_values = true)]
    pub path: PathBuf,
    /// The memory limit, in bytes
    #[arg(long, value_name = "BYTES")]
    pub memory: u64,
    /// The CPU-time limit, in seconds
    #[arg(long, value_name = "SECONDS")]
    pub cpu_secs: u64,
    /// The open-files limit
    #[arg(long, value_name = "N")]
    pub open_files: u64,
    /// The longest header accepted, in bytes
    #[arg(long, value_name = "BYTES")]
    pub max_header_bytes: u64,
    /// Instead of parsing, try to connect to this address once confined, and exit 0 only if the
    /// attempt is refused: the check that the child has no network
    #[arg(long, value_name = "ADDR", hide = true)]
    pub check_network: Option<std::net::SocketAddr>,
    /// Instead of parsing, try to open this other file once confined, and exit 0 only if the
    /// attempt is refused: the check that the child can open no file but its own
    #[arg(long, value_name = "PATH", hide = true)]
    pub check_open: Option<PathBuf>,
    /// Instead of parsing, read the given file's first 8 bytes once confined, and exit 0 if that
    /// works: the check that the profile leaves the child its own file
    #[arg(long, hide = true)]
    pub check_read: bool,
    /// Instead of parsing, abort once confined: the stand-in for a parser that crashes on a
    /// hostile header under the profile
    #[arg(long, hide = true)]
    pub check_crash: bool,
    /// Instead of parsing, try to delete this file once confined, and exit 0 only if the attempt
    /// is refused: the check that the profile covers the file system, not only opens
    #[arg(long, value_name = "PATH", hide = true)]
    pub check_remove: Option<PathBuf>,
    /// Instead of parsing, try to create a UDP socket once confined, and exit 0 only if the
    /// attempt is refused: the check that the child cannot open a socket of any kind
    #[arg(long, hide = true)]
    pub check_socket: bool,
    /// Instead of parsing, exit 0 only if descriptor N is not open once confined (Linux, through
    /// `/proc/self/fd`): the check that inherited descriptors are closed
    #[arg(long, value_name = "N", hide = true)]
    pub check_fd: Option<u32>,
    /// Instead of parsing, try to send SIGTERM to this process once confined, and exit 0 only
    /// if the attempt is refused: the check that the child cannot signal the daemon
    #[arg(long, value_name = "PID", hide = true)]
    pub check_signal: Option<u32>,
    /// Instead of parsing, try to create a POSIX shared-memory object once confined, and exit 0
    /// only if the attempt is refused
    #[arg(long, hide = true)]
    pub check_shm: bool,
}

/// A header's format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum HeaderKind {
    /// GGUF, v2 or v3.
    Gguf,
    /// safetensors.
    Safetensors,
}

/// `syntropy __runner --ipc <handle> --region <handle>`, or `syntropy __runner --probe --ipc
/// <handle>`.
#[derive(Clone, Debug, Args)]
pub struct RunnerArgs {
    /// Probe mode: report the node's measurements and exit, with no region (DIST-E1-T17)
    #[arg(long, conflicts_with = "region")]
    pub probe: bool,
    /// The runner's end of the command channel, inherited from the daemon: `fd:<n>` or
    /// `handle:<n>`
    #[arg(long, value_name = "HANDLE")]
    pub ipc: ChannelHandle,
    /// The runner's shared-memory region: `fd:<n>` (inherited) or `name:<name>`
    #[arg(long, value_name = "HANDLE", required_unless_present = "probe")]
    pub region: Option<RegionHandle>,
}

/// `syntropy __selftest --root <root>`.
#[derive(Clone, Debug, Args)]
pub struct SelftestArgs {
    /// The staged build's root directory
    #[arg(long, value_name = "DIR")]
    pub root: PathBuf,
}

/// `syntropy cluster …`.
#[derive(Debug, Subcommand)]
pub enum ClusterCommand {
    /// Members, coordinator, event-log index and pending admissions
    Status(Later),
    /// Hardware per node
    Nodes(Later),
    /// Links with their transports and costs
    Graph(Later),
    /// Follow the cluster event stream
    Events(Later),
    /// Approve a pending admission
    Approve(Later),
    /// Deny a pending admission
    Deny(Later),
    /// Open the cluster for joins
    Open(Later),
    /// Revoke a node
    Revoke(Later),
    /// Remove a node
    Remove(Later),
    /// Leave the cluster
    Leave(Later),
    /// Join a cluster with an invite token
    Join(Later),
    /// Restore keyholder status from the recovery code
    Recover(Later),
    /// Found a new cluster from a surviving member
    Refound(Later),
    /// Keyholder changes
    #[command(subcommand)]
    Keyholder(KeyholderCommand),
}

/// `syntropy cluster keyholder …`.
#[derive(Debug, Subcommand)]
pub enum KeyholderCommand {
    /// Make a node a keyholder
    Promote(Later),
    /// Make a node no longer a keyholder
    Demote(Later),
}

/// `syntropy invite …`.
#[derive(Debug, Subcommand)]
pub enum InviteCommand {
    /// Create an invite token
    Create(Later),
    /// List invite tokens
    List(Later),
    /// Revoke an invite token
    Revoke(Later),
}

/// `syntropy debug …`.
#[derive(Debug, Subcommand)]
pub enum DebugCommand {
    /// Print the daemon's debug state
    State(Later),
    /// Run the parity driver against this node
    Parity(Later),
    /// Stream per-step telemetry records
    Telemetry(Later),
    /// Write a diagnostic bundle (`doctor --bundle`)
    Dump(Later),
}

/// Where `config show`, `check` and `path` read the configuration from: the files RFC-0054 §1
/// finds, or `--config`'s file in their place, then the environment and `--set` (RFC-0054 §2.2,
/// §2.4).
#[derive(Clone, Debug, Default, Args)]
pub struct ConfigSource {
    /// Read this file instead of the system, drop-in and user files (replaces `SYNTROPY_CONFIG`)
    #[arg(long, value_name = "PATH")]
    pub config: Option<PathBuf>,
    /// Set one key for this command, as `key=value`; repeatable, and a later one wins
    #[arg(long = "set", value_name = "KEY=VALUE")]
    pub set: Vec<String>,
}

/// Which file `config set`, `unset` and `edit` write (RFC-0054 §10.2): the user file, the system
/// file under `--system`, or the explicit file that `--config` or `SYNTROPY_CONFIG` names, which
/// replaces the other three when the daemon reads its configuration (RFC-0054 §1, order 4).
#[derive(Clone, Debug, Default, Args)]
pub struct ConfigTarget {
    /// Write the system file, which needs the privileges to write it
    #[arg(long, conflicts_with = "config")]
    pub system: bool,
    /// Write this file (replaces `SYNTROPY_CONFIG`)
    #[arg(long, value_name = "PATH")]
    pub config: Option<PathBuf>,
}

/// `syntropy config set <key> <value>`.
#[derive(Clone, Debug, Args)]
pub struct ConfigSetArgs {
    /// The dotted key, such as `api.tls`; quote a segment with other characters, as
    /// `policy."qwen3*".ram_headroom`
    pub key: String,
    /// The value: TOML (`8080`, `true`, `["a", "b"]`, `{ file = "/run/secrets/x" }`), or else
    /// the text as a string; quote a number meant as a string, as `'"8080"'`. `-` reads it from
    /// stdin, which keeps a secret out of shell history and the process list
    #[arg(allow_hyphen_values = true)]
    pub value: String,
    /// Submit the value to the cluster instead of writing a file
    #[arg(long, conflicts_with_all = ["system", "config"])]
    pub cluster: bool,
    /// The file to write.
    #[command(flatten)]
    pub target: ConfigTarget,
}

/// `syntropy config unset <key>`.
#[derive(Clone, Debug, Args)]
pub struct ConfigUnsetArgs {
    /// The dotted key
    pub key: String,
    /// Remove the cluster's value instead of a file's
    #[arg(long, conflicts_with_all = ["system", "config"])]
    pub cluster: bool,
    /// The file to write.
    #[command(flatten)]
    pub target: ConfigTarget,
}

/// `syntropy config edit`. The editor is `$VISUAL` or `$EDITOR`, started even without a terminal,
/// since it was named; only the default editor needs one.
#[derive(Clone, Debug, Args)]
pub struct ConfigEditArgs {
    /// Keep the edited file even when it is invalid
    #[arg(long)]
    pub force: bool,
    /// The file to edit.
    #[command(flatten)]
    pub target: ConfigTarget,
}

/// `syntropy config show`.
#[derive(Clone, Debug, Args)]
pub struct ConfigShowArgs {
    /// Print every key's effective value and the layer it came from
    #[arg(long)]
    pub resolved: bool,
    /// Where the configuration comes from.
    #[command(flatten)]
    pub source: ConfigSource,
}

/// `syntropy config check`.
#[derive(Clone, Debug, Args)]
pub struct ConfigCheckArgs {
    /// Where the configuration comes from.
    #[command(flatten)]
    pub source: ConfigSource,
}

/// `syntropy config path`.
#[derive(Clone, Debug, Args)]
pub struct ConfigPathArgs {
    /// Show this file as the explicit file, order 4 (replaces `SYNTROPY_CONFIG`)
    #[arg(long, value_name = "PATH")]
    pub config: Option<PathBuf>,
}

/// `syntropy config …`.
#[derive(Debug, Subcommand)]
pub enum ConfigCommand {
    /// Print the resolved configuration
    Show(ConfigShowArgs),
    /// Print one key
    Get(Later),
    /// Set one key in a configuration file
    Set(ConfigSetArgs),
    /// Remove one key from a configuration file
    Unset(ConfigUnsetArgs),
    /// Edit a configuration file in $EDITOR, validated on save
    Edit(ConfigEditArgs),
    /// Check the configuration, without a daemon
    Check(ConfigCheckArgs),
    /// Reload the configuration
    Reload(Later),
    /// Print the configuration files and directories
    Path(ConfigPathArgs),
    /// Print the configuration schema
    Schema(Later),
    /// Export configuration values to a file
    Export(Later),
}

/// `syntropy api-key …`.
#[derive(Debug, Subcommand)]
pub enum ApiKeyCommand {
    /// Create an API key
    Create(Later),
    /// List API keys
    List(Later),
    /// Revoke an API key
    Revoke(Later),
    /// Rotate an API key
    Rotate(Later),
}

/// `syntropy identity …`.
#[derive(Debug, Subcommand)]
pub enum IdentityCommand {
    /// This node's ID, the cluster ID and the API certificate, as fingerprints
    Show(Later),
}

/// `syntropy service …`.
#[derive(Debug, Subcommand)]
pub enum ServiceCommand {
    /// Install the service
    Install(Later),
    /// Uninstall the service
    Uninstall(Later),
    /// Start the service
    Start(Later),
    /// Stop the service
    Stop(Later),
    /// Restart the service
    Restart(Later),
    /// The service's status
    Status(Later),
}

/// `syntropy update …`.
#[derive(Debug, Subcommand)]
pub enum UpdateCommand {
    /// Check for an update
    Check(Later),
    /// Apply a staged update
    Apply(Later),
    /// Roll back to the previous version
    Rollback(Later),
    /// The update status
    Status(Later),
    /// Choose the update channel
    Channel(Later),
}

/// `syntropy provenance …`.
#[derive(Debug, Subcommand)]
pub enum ProvenanceCommand {
    /// Show provenance records
    Show(Later),
    /// Delete old provenance records
    Prune(Later),
}

/// The tree as clap sees it, for tests that walk every command.
#[must_use]
pub fn command() -> clap::Command {
    <Cli as clap::CommandFactory>::command()
}
