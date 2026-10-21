// SPDX-License-Identifier: MIT OR Apache-2.0

//! Where the daemon looks for its config files and keeps its state (RFC-0054 §1).
//!
//! [`locate`] is a pure function of the operating system, the run mode, the environment and the
//! `--config` flag, so the paths of every OS are tested on any host. [`Locations::scan`] then
//! reads the file system: which files exist, and the drop-ins in lexical order.
//!
//! The config file layer, merged in this order (later wins, key by key):
//!
//! | Order | File | Linux | macOS | Windows |
//! |---|---|---|---|---|
//! | 1 | system file | `/etc/syntropy/config.toml` | `/Library/Application Support/Syntropy/config.toml` | `%ProgramData%\Syntropy\config.toml` |
//! | 2 | system drop-ins, lexical order | `/etc/syntropy/config.d/*.toml` | `/Library/Application Support/Syntropy/config.d/*.toml` | `%ProgramData%\Syntropy\config.d\*.toml` |
//! | 3 | user file | `$XDG_CONFIG_HOME/syntropy/config.toml` (default `~/.config/…`) | `~/Library/Application Support/Syntropy/config.toml` | `%LOCALAPPDATA%\Syntropy\config.toml` |
//! | 4 | explicit file | `--config <path>` or `SYNTROPY_CONFIG` | same | same |
//!
//! An explicit file replaces orders 1–3. A system service reads orders 1–2 only.

use std::ffi::{OsStr, OsString};
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use crate::env::Env;
use crate::origin::Origin;
use crate::resolve::Problem;

/// The operating system whose path rules apply. Every Unix other than macOS takes the Linux
/// rules, which are the XDG base directory rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Os {
    /// Linux, and other Unix systems.
    Linux,
    /// macOS.
    MacOs,
    /// Windows.
    Windows,
}

impl Os {
    /// The OS this binary was built for.
    pub const fn current() -> Os {
        if cfg!(target_os = "macos") {
            Os::MacOs
        } else if cfg!(windows) {
            Os::Windows
        } else {
            Os::Linux
        }
    }

    /// Joins `parts` onto `base` with this OS's separator, so a Windows path built on a Linux
    /// host still reads `C:\ProgramData\Syntropy`.
    pub(crate) fn join(self, base: &OsStr, parts: &[&str]) -> PathBuf {
        let separator = if self == Os::Windows { "\\" } else { "/" };
        let mut path = OsString::from(base);
        for part in parts {
            let ends_with_separator = path
                .to_string_lossy()
                .ends_with(|c| c == '/' || (self == Os::Windows && c == '\\'));
            if !ends_with_separator {
                path.push(separator);
            }
            path.push(part);
        }
        PathBuf::from(path)
    }
}

/// How the daemon runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    /// As the logged-in user: every order of the config layer applies.
    User,
    /// As a system service (RFC-0052 §6). The service account has no user file, so only the
    /// system file and the drop-ins are read.
    Service,
}

/// Where the explicit file (order 4) came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Explicit {
    /// The file.
    pub path: PathBuf,
    /// `Origin::Cli { flag: "--config" }` or `Origin::Env { var: "SYNTROPY_CONFIG" }`.
    pub from: Origin,
}

/// Where the runtime endpoints live: the local API socket and the runner adoption sockets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Runtime {
    /// A directory for Unix domain sockets.
    Dir(PathBuf),
    /// Windows named pipes, `\\.\pipe\syntropy-<user SID>-*`.
    NamedPipes,
}

impl fmt::Display for Runtime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Runtime::Dir(dir) => write!(f, "{}", dir.display()),
            Runtime::NamedPipes => f.write_str(r"\\.\pipe\syntropy-<user SID>-*"),
        }
    }
}

/// The default state, store and runtime locations (RFC-0054 §1). `node.state_dir` and
/// `store.dirs` override the first two. `None` means the environment does not say where, such
/// as a user with no `HOME`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dirs {
    /// The identity key, proposal ledger, API key hashes, host-change ledger and logs.
    pub state: Option<PathBuf>,
    /// The model store: chunks, packs, session blobs.
    pub store: Option<PathBuf>,
    /// The local API socket and the runner adoption sockets.
    pub runtime: Option<Runtime>,
    /// Set when `SYNTROPY_HOME` chose `state` and `store`.
    pub home: Option<PathBuf>,
}

/// Every place the config layer and the state live, for one OS, mode and environment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Locations {
    /// The OS whose rules built these paths.
    pub os: Os,
    /// The run mode.
    pub mode: Mode,
    /// Order 1, the system file.
    pub system_file: PathBuf,
    /// Order 2, the drop-in directory beside the system file.
    pub dropin_dir: PathBuf,
    /// Order 3, the user file. `None` for a system service, or when the environment gives no
    /// home directory.
    pub user_file: Option<PathBuf>,
    /// Order 4. When set, it replaces orders 1–3.
    pub explicit: Option<Explicit>,
    /// The state, store and runtime locations.
    pub dirs: Dirs,
}

/// The name of the explicit-file variable (RFC-0054 §2.4).
pub const CONFIG_VAR: &str = "SYNTROPY_CONFIG";
/// The name of the variable that sets the state directory and the default store.
pub const HOME_VAR: &str = "SYNTROPY_HOME";
/// The flag that names the explicit file.
pub const CONFIG_FLAG: &str = "--config";

/// The config and state locations for `os`, `mode` and `env`. `config_flag` is `--config`'s
/// value, which wins over `SYNTROPY_CONFIG`.
///
/// Rules beyond RFC-0054 §1's table:
/// - an XDG variable that is empty or not absolute is ignored, as the XDG base directory
///   specification says;
/// - `%ProgramData%` defaults to `C:\ProgramData`; with no `HOME` (or `%LOCALAPPDATA%`) there is
///   no user file and no default state or store;
/// - a Windows system service keeps its state and store in `%ProgramData%\Syntropy`, the
///   per-machine layout of RFC-0052 §5.1 (RFC-0054 §1's dated correction), and a macOS
///   LaunchDaemon in `/Library/Application Support/Syntropy` (RFC-0054 §1; OQ-PRD-26, settled
///   2026-10-22);
/// - `SYNTROPY_HOME=H` puts the state in `H/state` and the store in `H/store`, the layout the
///   macOS and Windows defaults use under their `Syntropy` directory;
/// - an empty `SYNTROPY_CONFIG` or `SYNTROPY_HOME` counts as unset.
pub fn locate(os: Os, mode: Mode, env: &Env, config_flag: Option<&Path>) -> Locations {
    let home = env.get_nonempty("HOME");
    let (system_dir, user_dir) = match os {
        Os::Linux => {
            let config_home = env
                .absolute(os, "XDG_CONFIG_HOME")
                .map(OsStr::to_os_string)
                .or_else(|| home.map(|h| os.join(h, &[".config"]).into_os_string()));
            (
                PathBuf::from("/etc/syntropy"),
                config_home.map(|c| os.join(&c, &["syntropy"])),
            )
        }
        Os::MacOs => (
            PathBuf::from("/Library/Application Support/Syntropy"),
            home.map(|h| os.join(h, &["Library", "Application Support", "Syntropy"])),
        ),
        Os::Windows => {
            let program_data = env
                .get_nonempty("PROGRAMDATA")
                .unwrap_or_else(|| OsStr::new(r"C:\ProgramData"));
            (
                os.join(program_data, &["Syntropy"]),
                env.get_nonempty("LOCALAPPDATA")
                    .map(|l| os.join(l, &["Syntropy"])),
            )
        }
    };
    let system_file = os.join(system_dir.as_os_str(), &["config.toml"]);
    let dropin_dir = os.join(system_dir.as_os_str(), &["config.d"]);
    let user_file = match mode {
        Mode::User => user_dir
            .as_ref()
            .map(|d| os.join(d.as_os_str(), &["config.toml"])),
        Mode::Service => None,
    };
    let explicit = match config_flag {
        Some(path) => Some(Explicit {
            path: path.to_path_buf(),
            from: Origin::Cli {
                flag: String::from(CONFIG_FLAG),
            },
        }),
        None => env.get_nonempty(CONFIG_VAR).map(|path| Explicit {
            path: PathBuf::from(path),
            from: Origin::Env {
                var: String::from(CONFIG_VAR),
            },
        }),
    };
    Locations {
        os,
        mode,
        system_file,
        dropin_dir,
        user_file,
        explicit,
        dirs: dirs(os, mode, env, &system_dir, user_dir.as_deref()),
    }
}

fn dirs(os: Os, mode: Mode, env: &Env, system_dir: &Path, user_dir: Option<&Path>) -> Dirs {
    let home = env.get_nonempty("HOME");
    let runtime = match (os, mode) {
        (Os::Linux, Mode::Service) => Some(Runtime::Dir(PathBuf::from("/run/syntropy"))),
        (Os::Linux, Mode::User) => env
            .absolute(os, "XDG_RUNTIME_DIR")
            .map(|r| Runtime::Dir(os.join(r, &["syntropy"]))),
        (Os::MacOs, _) => env
            .get_nonempty("TMPDIR")
            .map(|t| Runtime::Dir(os.join(t, &["syntropy"]))),
        (Os::Windows, _) => Some(Runtime::NamedPipes),
    };
    if let Some(syntropy_home) = env.get_nonempty(HOME_VAR) {
        return Dirs {
            state: Some(os.join(syntropy_home, &["state"])),
            store: Some(os.join(syntropy_home, &["store"])),
            runtime,
            home: Some(PathBuf::from(syntropy_home)),
        };
    }
    let (state, store) = match (os, mode) {
        (Os::Linux, Mode::Service) => (
            Some(PathBuf::from("/var/lib/syntropy/state")),
            Some(PathBuf::from("/var/lib/syntropy/store")),
        ),
        (Os::Linux, Mode::User) => {
            let xdg = |var: &str, fallback: &[&str]| {
                env.absolute(os, var)
                    .map(OsStr::to_os_string)
                    .or_else(|| home.map(|h| os.join(h, fallback).into_os_string()))
            };
            (
                xdg("XDG_STATE_HOME", &[".local", "state"]).map(|s| os.join(&s, &["syntropy"])),
                xdg("XDG_DATA_HOME", &[".local", "share"])
                    .map(|d| os.join(&d, &["syntropy", "store"])),
            )
        }
        // A per-machine Windows service keeps state and store beside its config, in
        // `%ProgramData%\Syntropy` (RFC-0052 §5.1, where the updater reads requests, §7.1).
        // A macOS LaunchDaemon does the same in `/Library/Application Support/Syntropy`
        // (RFC-0054 §1; OQ-PRD-26, settled 2026-10-22).
        (Os::MacOs | Os::Windows, Mode::Service) => (
            Some(os.join(system_dir.as_os_str(), &["state"])),
            Some(os.join(system_dir.as_os_str(), &["store"])),
        ),
        // The user directory from `locate`: `~/Library/Application Support/Syntropy` and
        // `%LOCALAPPDATA%\Syntropy`.
        (Os::MacOs | Os::Windows, Mode::User) => (
            user_dir.map(|d| os.join(d.as_os_str(), &["state"])),
            user_dir.map(|d| os.join(d.as_os_str(), &["store"])),
        ),
    };
    Dirs {
        state,
        store,
        runtime,
        home: None,
    }
}

/// What [`Locations::scan`] found on disk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scan {
    /// The system file, if it exists. `None` when an explicit file replaces it.
    pub system_file: Option<PathBuf>,
    /// The drop-ins, in lexical order. Empty when an explicit file replaces them.
    pub dropins: Vec<PathBuf>,
    /// The user file, if it exists and this mode reads it.
    pub user_file: Option<PathBuf>,
    /// The explicit file, which must exist.
    pub explicit: Option<PathBuf>,
}

impl Scan {
    /// The files of the config layer in merge order: later files win.
    pub fn files(&self) -> Vec<PathBuf> {
        if let Some(explicit) = &self.explicit {
            return vec![explicit.clone()];
        }
        self.system_file
            .iter()
            .chain(&self.dropins)
            .chain(&self.user_file)
            .cloned()
            .collect()
    }
}

impl Locations {
    /// Finds the files of the config layer on disk. A missing system file, drop-in directory
    /// or user file is the normal case and is skipped. A missing explicit file, or a drop-in
    /// directory that exists but cannot be read, is an error.
    pub fn scan(&self) -> Result<Scan, Problem> {
        if let Some(explicit) = &self.explicit {
            if !explicit.path.is_file() {
                return Err(Problem::new(
                    Some(explicit.from.clone()),
                    format!(
                        "the config file `{}` does not exist or is not a file",
                        explicit.path.display()
                    ),
                ));
            }
            return Ok(Scan {
                system_file: None,
                dropins: Vec::new(),
                user_file: None,
                explicit: Some(explicit.path.clone()),
            });
        }
        let names = match std::fs::read_dir(&self.dropin_dir) {
            Ok(entries) => entries
                .filter_map(Result::ok)
                .filter(|entry| entry.path().is_file())
                .map(|entry| entry.file_name())
                .collect(),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Vec::new(),
            Err(e) => {
                return Err(Problem::new(
                    None,
                    format!(
                        "cannot read the drop-in directory `{}`: {e}",
                        self.dropin_dir.display()
                    ),
                ));
            }
        };
        let dropins = order_dropins(self.os, names)
            .into_iter()
            .map(|name| self.dropin_dir.join(name))
            .collect();
        Ok(Scan {
            system_file: Some(self.system_file.clone()).filter(|p| p.is_file()),
            dropins,
            user_file: self.user_file.clone().filter(|p| p.is_file()),
            explicit: None,
        })
    }
}

/// The drop-in file names that count, in merge order.
///
/// A drop-in is a `*.toml` file (the extension compared without case on Windows, as its file
/// system does). Names that start with `.` are skipped, so an editor's hidden swap or backup
/// file never loads. The order is the byte order of the name: `10-a.toml` sorts before
/// `9-b.toml`, so packagers number their files with a fixed width (`50-distro.toml`).
pub fn order_dropins(os: Os, names: impl IntoIterator<Item = OsString>) -> Vec<OsString> {
    let mut names: Vec<OsString> = names
        .into_iter()
        .filter(|name| is_dropin(os, name))
        .collect();
    names.sort_by(|a, b| a.as_encoded_bytes().cmp(b.as_encoded_bytes()));
    names.dedup();
    names
}

fn is_dropin(os: Os, name: &OsStr) -> bool {
    let name = name.to_string_lossy();
    if name.starts_with('.') {
        return false;
    }
    let Some(stem_len) = name.len().checked_sub(".toml".len()) else {
        return false;
    };
    if stem_len == 0 || !name.is_char_boundary(stem_len) {
        return false;
    }
    let extension = &name[stem_len..];
    match os {
        Os::Windows => extension.eq_ignore_ascii_case(".toml"),
        Os::Linux | Os::MacOs => extension == ".toml",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(os: Os, pairs: &[(&str, &str)]) -> Env {
        Env::from_pairs(os, pairs.iter().map(|(k, v)| (*k, *v)))
    }

    fn s(path: &Path) -> String {
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn linux_user_paths_follow_xdg() {
        let e = env(
            Os::Linux,
            &[
                ("HOME", "/home/u"),
                ("XDG_RUNTIME_DIR", "/run/user/1000"),
                ("XDG_CONFIG_HOME", "relative/ignored"),
            ],
        );
        let l = locate(Os::Linux, Mode::User, &e, None);
        assert_eq!(s(&l.system_file), "/etc/syntropy/config.toml");
        assert_eq!(s(&l.dropin_dir), "/etc/syntropy/config.d");
        assert_eq!(
            l.user_file.as_deref().map(s).as_deref(),
            Some("/home/u/.config/syntropy/config.toml")
        );
        assert_eq!(
            l.dirs.state.as_deref().map(s).as_deref(),
            Some("/home/u/.local/state/syntropy")
        );
        assert_eq!(
            l.dirs.store.as_deref().map(s).as_deref(),
            Some("/home/u/.local/share/syntropy/store")
        );
        assert_eq!(
            l.dirs.runtime,
            Some(Runtime::Dir(PathBuf::from("/run/user/1000/syntropy")))
        );
        assert_eq!(l.explicit, None);

        let e = env(
            Os::Linux,
            &[
                ("HOME", "/home/u"),
                ("XDG_CONFIG_HOME", "/cfg"),
                ("XDG_STATE_HOME", "/st/"),
                ("XDG_DATA_HOME", "/data"),
            ],
        );
        let l = locate(Os::Linux, Mode::User, &e, None);
        assert_eq!(
            l.user_file.as_deref().map(s).as_deref(),
            Some("/cfg/syntropy/config.toml")
        );
        assert_eq!(
            l.dirs.state.as_deref().map(s).as_deref(),
            Some("/st/syntropy")
        );
        assert_eq!(
            l.dirs.store.as_deref().map(s).as_deref(),
            Some("/data/syntropy/store")
        );
        assert_eq!(l.dirs.runtime, None);
    }

    #[test]
    fn linux_service_reads_no_user_file() {
        let e = env(Os::Linux, &[("HOME", "/root")]);
        let l = locate(Os::Linux, Mode::Service, &e, None);
        assert_eq!(l.user_file, None);
        assert_eq!(
            l.dirs.state.as_deref().map(s).as_deref(),
            Some("/var/lib/syntropy/state")
        );
        assert_eq!(
            l.dirs.store.as_deref().map(s).as_deref(),
            Some("/var/lib/syntropy/store")
        );
        assert_eq!(
            l.dirs.runtime,
            Some(Runtime::Dir(PathBuf::from("/run/syntropy")))
        );
    }

    #[test]
    fn macos_paths() {
        let e = env(
            Os::MacOs,
            &[("HOME", "/Users/u"), ("TMPDIR", "/var/folders/x/T/")],
        );
        let l = locate(Os::MacOs, Mode::User, &e, None);
        assert_eq!(
            s(&l.system_file),
            "/Library/Application Support/Syntropy/config.toml"
        );
        assert_eq!(
            s(&l.dropin_dir),
            "/Library/Application Support/Syntropy/config.d"
        );
        assert_eq!(
            l.user_file.as_deref().map(s).as_deref(),
            Some("/Users/u/Library/Application Support/Syntropy/config.toml")
        );
        assert_eq!(
            l.dirs.state.as_deref().map(s).as_deref(),
            Some("/Users/u/Library/Application Support/Syntropy/state")
        );
        assert_eq!(
            l.dirs.store.as_deref().map(s).as_deref(),
            Some("/Users/u/Library/Application Support/Syntropy/store")
        );
        assert_eq!(
            l.dirs.runtime,
            Some(Runtime::Dir(PathBuf::from("/var/folders/x/T/syntropy")))
        );
        let service = locate(Os::MacOs, Mode::Service, &e, None);
        assert_eq!(service.user_file, None);
        // A LaunchDaemon's state and store, as OQ-PRD-26 settled them.
        assert_eq!(
            service.dirs.state.as_deref().map(s).as_deref(),
            Some("/Library/Application Support/Syntropy/state")
        );
        assert_eq!(
            service.dirs.store.as_deref().map(s).as_deref(),
            Some("/Library/Application Support/Syntropy/store")
        );
    }

    #[test]
    fn windows_paths() {
        let e = env(
            Os::Windows,
            &[
                ("ProgramData", r"D:\PD"),
                ("LocalAppData", r"C:\Users\u\AppData\Local"),
            ],
        );
        let l = locate(Os::Windows, Mode::User, &e, None);
        assert_eq!(s(&l.system_file), r"D:\PD\Syntropy\config.toml");
        assert_eq!(s(&l.dropin_dir), r"D:\PD\Syntropy\config.d");
        assert_eq!(
            l.user_file.as_deref().map(s).as_deref(),
            Some(r"C:\Users\u\AppData\Local\Syntropy\config.toml")
        );
        assert_eq!(
            l.dirs.state.as_deref().map(s).as_deref(),
            Some(r"C:\Users\u\AppData\Local\Syntropy\state")
        );
        assert_eq!(
            l.dirs.store.as_deref().map(s).as_deref(),
            Some(r"C:\Users\u\AppData\Local\Syntropy\store")
        );
        assert_eq!(l.dirs.runtime, Some(Runtime::NamedPipes));

        // A per-machine service, as RFC-0052 §5.1 lays it out: everything under %ProgramData%.
        let e = env(
            Os::Windows,
            &[(
                "LOCALAPPDATA",
                r"C:\Windows\ServiceProfiles\Syntropy\AppData\Local",
            )],
        );
        let l = locate(Os::Windows, Mode::Service, &e, None);
        assert_eq!(s(&l.system_file), r"C:\ProgramData\Syntropy\config.toml");
        assert_eq!(l.user_file, None);
        assert_eq!(
            l.dirs.state.as_deref().map(s).as_deref(),
            Some(r"C:\ProgramData\Syntropy\state")
        );
        assert_eq!(
            l.dirs.store.as_deref().map(s).as_deref(),
            Some(r"C:\ProgramData\Syntropy\store")
        );
    }

    #[test]
    fn the_explicit_file_comes_from_the_flag_then_the_variable() {
        let e = env(Os::Linux, &[("SYNTROPY_CONFIG", "/env.toml")]);
        let l = locate(Os::Linux, Mode::User, &e, None);
        assert_eq!(
            l.explicit,
            Some(Explicit {
                path: PathBuf::from("/env.toml"),
                from: Origin::Env {
                    var: String::from("SYNTROPY_CONFIG")
                },
            })
        );
        let l = locate(Os::Linux, Mode::User, &e, Some(Path::new("/flag.toml")));
        assert_eq!(
            l.explicit.map(|x| (x.path, x.from)),
            Some((
                PathBuf::from("/flag.toml"),
                Origin::Cli {
                    flag: String::from("--config")
                }
            ))
        );
        let e = env(Os::Linux, &[("SYNTROPY_CONFIG", "")]);
        assert_eq!(locate(Os::Linux, Mode::User, &e, None).explicit, None);
    }

    #[test]
    fn syntropy_home_sets_state_and_store() {
        for (os, home, state, store) in [
            (Os::Linux, "/srv/s", "/srv/s/state", "/srv/s/store"),
            (Os::MacOs, "/srv/s/", "/srv/s/state", "/srv/s/store"),
            (Os::Windows, r"E:\S", r"E:\S\state", r"E:\S\store"),
        ] {
            let e = env(os, &[("SYNTROPY_HOME", home), ("HOME", "/home/u")]);
            let d = locate(os, Mode::User, &e, None).dirs;
            assert_eq!(d.state.as_deref().map(s).as_deref(), Some(state), "{os:?}");
            assert_eq!(d.store.as_deref().map(s).as_deref(), Some(store), "{os:?}");
            assert_eq!(d.home, Some(PathBuf::from(home)));
        }
    }

    #[test]
    fn dropins_are_toml_files_in_byte_order() {
        let names = [
            "b.toml",
            ".hidden.toml",
            "a.TOML",
            "10-x.toml",
            "9-y.toml",
            "notes.txt",
        ]
        .map(OsString::from);
        let linux: Vec<_> = order_dropins(Os::Linux, names.clone());
        assert_eq!(
            linux,
            ["10-x.toml", "9-y.toml", "b.toml"].map(OsString::from)
        );
        let windows: Vec<_> = order_dropins(Os::Windows, names);
        assert_eq!(
            windows,
            ["10-x.toml", "9-y.toml", "a.TOML", "b.toml"].map(OsString::from)
        );
        assert!(order_dropins(Os::Linux, [OsString::from(".toml")]).is_empty());
    }

    /// The paths this host's own rules give, from its real environment. The hosted runs on
    /// Linux, macOS and Windows are the evidence that each OS gets its row of RFC-0054 §1.
    #[test]
    fn this_host_gets_its_own_row() {
        let os = Os::current();
        let l = locate(os, Mode::User, &Env::from_process(os), None);
        let system = s(&l.system_file);
        if cfg!(target_os = "macos") {
            assert_eq!(system, "/Library/Application Support/Syntropy/config.toml");
        } else if cfg!(windows) {
            assert!(system.ends_with(r"\Syntropy\config.toml"), "{system}");
            assert_eq!(l.dirs.runtime, Some(Runtime::NamedPipes));
        } else {
            assert_eq!(system, "/etc/syntropy/config.toml");
        }
    }
}
