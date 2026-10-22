// SPDX-License-Identifier: MIT OR Apache-2.0

//! `syntropy tune`'s tier side (RFC-0020 §9.1; ADR-0007; TIER-E2-T13).
//!
//! On Apple silicon the GPU may wire only up to the OS's working-set limit,
//! `iogpu.wired_limit_mb`. A plan that needs more `Unified` memory than that is infeasible until
//! the user raises the limit, and ADR-0007 makes raising it an explicit opt-in. This module is the
//! part of that opt-in that the tier side owns:
//!
//! - **The number.** [`propose`] gives the smallest limit, in whole MiB, that holds the plan's
//!   `Unified` bytes. It never exceeds RAM minus `resource.ram_headroom`, so the RFC-0020 §5
//!   reserve survives tuning (§9.1 step 1).
//! - **The record.** [`Tune::apply`] writes a [`TuneRecord`] to `<state>/tune.json` *before* the
//!   limit changes (§9.1 step 3). The record holds the original value, the new value, the time and
//!   the reason, and the launchd job if the user asked for persistence (step 4).
//! - **The change.** The new value is written by a [`PrivilegedRunner`] that runs each
//!   [`HostOp`] with the administrator's approval (step 2). Syntropy holds no standing
//!   privilege, so the runner is the caller's: the CLI's authorization prompt (PROD-E6-T14), or
//!   a test's.
//! - **The re-probe.** After any change, [`Reprobe::limit_changed`] lets the probe re-read the
//!   limit and the coordinator re-plan (step 5). The probe (PLC-E1-T03) and the coordinator
//!   implement it.
//! - **The revert.** [`Tune::revert`] restores the recorded original and removes the launchd job
//!   (step 6). Uninstall calls the same revert (RFC-0052 §11.2 step 4). If the current value is
//!   no longer the one tune wrote, because someone changed it by hand, revert leaves it alone and
//!   says so ([`ValueRevert::HandEdited`]).
//!
//! Nothing in this crate writes the sysctl: the probe and the tier manager only read it
//! ([`WiredLimitSource`]; on macOS, `os::macos`). The only write is [`HostOp::SetWiredLimit`],
//! run by the caller's runner.
//!
//! Time comes from the [`Clock`] the caller gives (RFC-0060 §2.1 rule T1).

#![forbid(unsafe_code)]

use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use syntropy_core::{Bytes, Clock};

/// The sysctl that holds the GPU wired limit, in MiB; 0 means the OS default.
pub const WIRED_LIMIT_SYSCTL: &str = "iogpu.wired_limit_mb";

/// The record's file in the state directory.
pub const RECORD_FILE: &str = "tune.json";

/// The record's format version.
pub const SCHEMA: u16 = 1;

/// One MiB, the sysctl's unit.
const MIB: u64 = 1 << 20;

/// What [`propose`] found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Proposal {
    /// The plan's `Unified` bytes already fit under the current limit, so nothing changes.
    Fits,
    /// Raise the limit to `limit_mb`, the smallest whole MiB that holds the plan's `Unified`
    /// bytes.
    Raise {
        /// The proposed `iogpu.wired_limit_mb`.
        limit_mb: u64,
    },
}

/// The limit to propose for a plan that needs `plan_unified` bytes of `Unified` memory, on a
/// machine with `ram` bytes of RAM and the RFC-0020 §5 reserve `headroom`, when the OS lets the
/// GPU wire `current` bytes now (RFC-0020 §9.1 step 1). `current` is the effective limit: the
/// sysctl's value, or `recommendedMaxWorkingSetSize` while the sysctl is 0.
///
/// # Errors
///
/// [`TuneError::TooLarge`] when the plan needs more than RAM minus the reserve. No limit can
/// make it fit without giving up the reserve, which tuning never does.
pub fn propose(
    plan_unified: Bytes,
    current: Bytes,
    ram: Bytes,
    headroom: Bytes,
) -> Result<Proposal, TuneError> {
    let ceiling = Bytes(ram.0.saturating_sub(headroom.0));
    let needed_mb = plan_unified.0.div_ceil(MIB);
    if needed_mb > ceiling.0 / MIB {
        return Err(TuneError::TooLarge {
            needed: plan_unified,
            ceiling,
        });
    }
    if plan_unified <= current {
        return Ok(Proposal::Fits);
    }
    Ok(Proposal::Raise {
        limit_mb: needed_mb,
    })
}

/// Where a recorded change is in its life, as in RFC-0052 §11.1's host-change ledger.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordState {
    /// Recorded, and about to be written or being written. After a crash the write may or may
    /// not have happened, so a revert checks the current value.
    Pending,
    /// Written.
    Applied,
    /// Undone.
    Reverted,
    /// Never written: the runner failed or approval was refused.
    Abandoned,
}

/// The launchd job that re-applies the recorded value at boot, when the user asked for
/// persistence (RFC-0020 §9.1 step 4). Packaging installs it (PROD-E6-T14); revert removes it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BootJob {
    /// The job's label, as `launchctl` names it in the system domain.
    pub label: String,
    /// Its property list, under `/Library/LaunchDaemons`.
    pub plist: PathBuf,
}

/// The record of a change to the wired limit (RFC-0020 §9.1 step 3), in `<state>/tune.json`.
/// `syntropy tune --status` prints it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuneRecord {
    /// The format version, [`SCHEMA`].
    pub schema: u16,
    /// The sysctl, [`WIRED_LIMIT_SYSCTL`].
    pub sysctl: String,
    /// The value before Syntropy first changed it, which a revert restores; 0 is the OS default.
    pub original_mb: u64,
    /// The value tune wrote.
    pub new_mb: u64,
    /// When the change was recorded, in Unix milliseconds.
    pub at_unix_ms: u64,
    /// Why, as the CLI showed it: the plan that needed the limit.
    pub reason: String,
    /// The Syntropy version that recorded it.
    pub by_version: String,
    /// Where the change is.
    pub state: RecordState,
    /// The launchd job that re-applies `new_mb` at boot, if the user asked for one.
    pub boot_job: Option<BootJob>,
}

/// A change that needs the administrator's approval (RFC-0020 §9.1 step 2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostOp {
    /// Set `iogpu.wired_limit_mb` to `mb`. The command is [`HostOp::set_wired_limit_argv`].
    SetWiredLimit {
        /// The value to set; 0 restores the OS default.
        mb: u64,
    },
    /// Remove a launchd job: unload it from the system domain if it is loaded
    /// (`launchctl bootout system/<label>`), then delete its property list.
    RemoveBootJob(BootJob),
}

impl HostOp {
    /// The command that sets the wired limit to `mb`, which an approved runner executes as root:
    /// `/usr/sbin/sysctl -w iogpu.wired_limit_mb=<mb>`.
    #[must_use]
    pub fn set_wired_limit_argv(mb: u64) -> Vec<OsString> {
        vec![
            "/usr/sbin/sysctl".into(),
            "-w".into(),
            format!("{WIRED_LIMIT_SYSCTL}={mb}").into(),
        ]
    }
}

/// Runs a [`HostOp`] with the administrator's approval. The CLI's runner asks through the
/// system's authorization prompt (PROD-E6-T14), so Syntropy holds no standing privilege.
pub trait PrivilegedRunner {
    /// Runs `op`.
    ///
    /// # Errors
    ///
    /// The approval was refused, or the command failed.
    fn run(&mut self, op: &HostOp) -> io::Result<()>;
}

/// Reads the current `iogpu.wired_limit_mb`. The probe and the tier manager only ever read it.
pub trait WiredLimitSource {
    /// The sysctl's current value, in MiB; 0 is the OS default.
    ///
    /// # Errors
    ///
    /// The sysctl cannot be read, or does not exist on this host.
    fn read_mb(&self) -> io::Result<u64>;
}

/// What follows any change to the limit (RFC-0020 §9.1 step 5). The probe re-reads the limit and
/// `recommendedMaxWorkingSetSize` and reports the new `usable` (RFC-0010), and the coordinator
/// re-plans (RFC-0001 §7.2). PLC-E1-T03's probe and the coordinator implement it.
pub trait Reprobe {
    /// The limit is now `limit_mb`.
    fn limit_changed(&mut self, limit_mb: u64);
}

/// The record's file, written whole: a new file beside it, renamed over it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TuneStore {
    path: PathBuf,
}

impl TuneStore {
    /// The record in the daemon's state directory, `<state>/tune.json`.
    #[must_use]
    pub fn in_state_dir(state: &Path) -> TuneStore {
        TuneStore {
            path: state.join(RECORD_FILE),
        }
    }

    /// The record's path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The record, if there is one.
    ///
    /// # Errors
    ///
    /// The file cannot be read, or is not a record.
    pub fn load(&self) -> Result<Option<TuneRecord>, TuneError> {
        let text = match fs::read_to_string(&self.path) {
            Ok(text) => text,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(self.io_error(error)),
        };
        serde_json::from_str(&text)
            .map(Some)
            .map_err(|error| TuneError::Record {
                path: self.path.clone(),
                detail: error.to_string(),
            })
    }

    /// Writes `record`: to a new file beside the record, flushed to disk, then renamed over it,
    /// so a crash leaves the old record or the new one.
    ///
    /// # Errors
    ///
    /// The file cannot be written.
    pub fn save(&self, record: &TuneRecord) -> Result<(), TuneError> {
        let text = serde_json::to_string_pretty(record).map_err(|error| TuneError::Record {
            path: self.path.clone(),
            detail: error.to_string(),
        })?;
        let next = self.path.with_extension("json.next");
        let write = || -> io::Result<()> {
            if let Some(dir) = self.path.parent() {
                fs::create_dir_all(dir)?;
            }
            let mut file = fs::File::create(&next)?;
            file.write_all(text.as_bytes())?;
            file.write_all(b"\n")?;
            file.sync_all()?;
            fs::rename(&next, &self.path)
        };
        write().map_err(|error| self.io_error(error))
    }

    fn io_error(&self, error: io::Error) -> TuneError {
        TuneError::Record {
            path: self.path.clone(),
            detail: error.to_string(),
        }
    }
}

/// What a revert did to the value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueRevert {
    /// The value tune wrote was still set, and the original is restored.
    Restored {
        /// The value now set.
        to_mb: u64,
    },
    /// The value is already the original, so nothing was written: the write never happened
    /// (a pending record), or someone put it back by hand.
    AlreadyOriginal {
        /// The value.
        mb: u64,
    },
    /// Someone changed the value by hand since tune wrote it. Revert leaves it alone, and the
    /// CLI says so (RFC-0020 §9.1 step 6).
    HandEdited {
        /// The value now set.
        current_mb: u64,
        /// The value tune wrote.
        recorded_mb: u64,
    },
}

/// What [`Tune::revert`] did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reverted {
    /// There was no change to revert: no record, or one already reverted or abandoned.
    NothingRecorded,
    /// The record's change is undone, as far as it could be.
    Done {
        /// What happened to the value.
        value: ValueRevert,
        /// The launchd job removed, if the record listed one.
        removed_job: Option<BootJob>,
    },
}

/// Why a tune step failed.
#[derive(Debug, PartialEq, Eq)]
pub enum TuneError {
    /// The plan needs more `Unified` memory than RAM minus the reserve.
    TooLarge {
        /// The plan's `Unified` bytes.
        needed: Bytes,
        /// RAM minus `resource.ram_headroom`.
        ceiling: Bytes,
    },
    /// The limit is already `mb`, so there is nothing to apply.
    AlreadyAt {
        /// The current value.
        mb: u64,
    },
    /// The record cannot be read or written.
    Record {
        /// The record's file.
        path: PathBuf,
        /// What went wrong.
        detail: String,
    },
    /// The sysctl cannot be read.
    Read(String),
    /// The runner failed: approval refused, or the command failed.
    Run(String),
    /// After the write, the sysctl reads back something else.
    ReadBack {
        /// The value written.
        wrote_mb: u64,
        /// The value read.
        read_mb: u64,
    },
    /// A launchd job can be attached only to an applied change.
    NotApplied,
}

impl fmt::Display for TuneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TuneError::TooLarge { needed, ceiling } => write!(
                f,
                "the plan needs {} bytes of unified memory, more than RAM minus the reserve \
                 ({} bytes), which no limit can give",
                needed.0, ceiling.0
            ),
            TuneError::AlreadyAt { mb } => write!(f, "{WIRED_LIMIT_SYSCTL} is already {mb}"),
            TuneError::Record { path, detail } => {
                write!(f, "the tune record {}: {detail}", path.display())
            }
            TuneError::Read(detail) => write!(f, "cannot read {WIRED_LIMIT_SYSCTL}: {detail}"),
            TuneError::Run(detail) => write!(f, "the change was not made: {detail}"),
            TuneError::ReadBack { wrote_mb, read_mb } => write!(
                f,
                "{WIRED_LIMIT_SYSCTL} was set to {wrote_mb} but reads {read_mb}"
            ),
            TuneError::NotApplied => f.write_str("no applied change to attach a boot job to"),
        }
    }
}

impl std::error::Error for TuneError {}

/// `syntropy tune`'s tier side over its record, the sysctl's reader, the caller's privileged
/// runner, the re-probe and the clock. See the [module docs](self).
pub struct Tune<'a, S, R, P, C> {
    /// The record.
    pub store: &'a TuneStore,
    /// Reads the sysctl.
    pub source: &'a S,
    /// Runs the changes, with approval.
    pub runner: &'a mut R,
    /// Told after every change.
    pub reprobe: &'a mut P,
    /// For the record's time.
    pub clock: &'a C,
    /// The Syntropy version that records the change.
    pub version: &'a str,
}

impl<S, R, P, C> Tune<'_, S, R, P, C>
where
    S: WiredLimitSource,
    R: PrivilegedRunner,
    P: Reprobe,
    C: Clock,
{
    /// The record, for `syntropy tune --status`.
    ///
    /// # Errors
    ///
    /// The record cannot be read.
    pub fn status(&self) -> Result<Option<TuneRecord>, TuneError> {
        self.store.load()
    }

    /// Sets the limit to `limit_mb` for `reason` (RFC-0020 §9.1 steps 2, 3 and 5):
    ///
    /// 1. records the change as pending, before anything is written;
    /// 2. has the runner set the sysctl;
    /// 3. reads it back, marks the record applied, and tells the re-probe.
    ///
    /// The original kept in the record is the value before Syntropy first changed it. A second
    /// tune keeps the first original while the value is still the one Syntropy wrote. If someone
    /// changed it by hand since, their value becomes the original.
    ///
    /// # Errors
    ///
    /// - [`TuneError::AlreadyAt`] when the limit is already `limit_mb`.
    /// - [`TuneError::Run`] when approval is refused or the command fails. The record is marked
    ///   abandoned if the value did not change.
    /// - [`TuneError::ReadBack`] when the value reads back as something else. The record stays
    ///   pending, and a revert sorts it out.
    /// - A read or record failure.
    pub fn apply(&mut self, limit_mb: u64, reason: &str) -> Result<TuneRecord, TuneError> {
        let current = self.read()?;
        if current == limit_mb {
            return Err(TuneError::AlreadyAt { mb: current });
        }
        let earlier = self.store.load()?.filter(|record| {
            matches!(record.state, RecordState::Applied | RecordState::Pending)
                && record.new_mb == current
        });
        let mut record = TuneRecord {
            schema: SCHEMA,
            sysctl: WIRED_LIMIT_SYSCTL.to_owned(),
            original_mb: earlier.as_ref().map_or(current, |r| r.original_mb),
            new_mb: limit_mb,
            at_unix_ms: self.clock.unix_ms(),
            reason: reason.to_owned(),
            by_version: self.version.to_owned(),
            state: RecordState::Pending,
            boot_job: earlier.and_then(|r| r.boot_job),
        };
        self.store.save(&record)?;
        if let Err(error) = self.runner.run(&HostOp::SetWiredLimit { mb: limit_mb }) {
            if self.read()? == current {
                record.state = RecordState::Abandoned;
                self.store.save(&record)?;
            }
            return Err(TuneError::Run(error.to_string()));
        }
        let read = self.read()?;
        if read != limit_mb {
            return Err(TuneError::ReadBack {
                wrote_mb: limit_mb,
                read_mb: read,
            });
        }
        record.state = RecordState::Applied;
        self.store.save(&record)?;
        self.reprobe.limit_changed(limit_mb);
        Ok(record)
    }

    /// Lists `job` in the applied record, after packaging installed it for `--persist`
    /// (RFC-0020 §9.1 step 4), so a revert removes it.
    ///
    /// # Errors
    ///
    /// [`TuneError::NotApplied`] without an applied record, or a record failure.
    pub fn attach_boot_job(&mut self, job: BootJob) -> Result<TuneRecord, TuneError> {
        let mut record = match self.store.load()? {
            Some(record) if record.state == RecordState::Applied => record,
            _ => return Err(TuneError::NotApplied),
        };
        record.boot_job = Some(job);
        self.store.save(&record)?;
        Ok(record)
    }

    /// Undoes the recorded change (RFC-0020 §9.1 step 6). `syntropy tune --revert` and uninstall
    /// (RFC-0052 §11.2 step 4) both call it.
    ///
    /// - If the value is still the one tune wrote, the original is restored and read back.
    /// - If it is already the original, nothing is written.
    /// - If someone changed it by hand, it is left alone ([`ValueRevert::HandEdited`]).
    ///
    /// In every case the launchd job the record lists is removed, because it would re-apply
    /// tune's value at the next boot. The record is then marked reverted, and the re-probe is
    /// told if the value changed.
    ///
    /// # Errors
    ///
    /// A read, record or runner failure, or a restored value that reads back wrong. The record
    /// is left as it was, so the revert can be retried.
    pub fn revert(&mut self) -> Result<Reverted, TuneError> {
        let mut record = match self.store.load()? {
            Some(record) if matches!(record.state, RecordState::Applied | RecordState::Pending) => {
                record
            }
            _ => return Ok(Reverted::NothingRecorded),
        };
        let current = self.read()?;
        let value = if current == record.original_mb {
            ValueRevert::AlreadyOriginal { mb: current }
        } else if current == record.new_mb {
            self.run(&HostOp::SetWiredLimit {
                mb: record.original_mb,
            })?;
            let read = self.read()?;
            if read != record.original_mb {
                return Err(TuneError::ReadBack {
                    wrote_mb: record.original_mb,
                    read_mb: read,
                });
            }
            ValueRevert::Restored {
                to_mb: record.original_mb,
            }
        } else {
            ValueRevert::HandEdited {
                current_mb: current,
                recorded_mb: record.new_mb,
            }
        };
        let removed_job = record.boot_job.clone();
        if let Some(job) = &removed_job {
            self.run(&HostOp::RemoveBootJob(job.clone()))?;
        }
        record.state = RecordState::Reverted;
        self.store.save(&record)?;
        if let ValueRevert::Restored { to_mb } = value {
            self.reprobe.limit_changed(to_mb);
        }
        Ok(Reverted::Done { value, removed_job })
    }

    fn read(&self) -> Result<u64, TuneError> {
        self.source
            .read_mb()
            .map_err(|error| TuneError::Read(error.to_string()))
    }

    fn run(&mut self, op: &HostOp) -> Result<(), TuneError> {
        self.runner
            .run(op)
            .map_err(|error| TuneError::Run(error.to_string()))
    }
}

#[cfg(test)]
mod tests;
