// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use syntropy_core::MonoTime;

use super::*;

const GIB: u64 = 1 << 30;

/// The fake host's state: the sysctl's value, the ops the runner ran, and the launchd job.
struct Host {
    value: u64,
    ops: Vec<HostOp>,
    /// Refuse every op, as a denied authorization prompt does.
    deny: bool,
    job_installed: bool,
    /// The record, which the runner checks before every write: the record comes first.
    store: TuneStore,
}

type Shared = Rc<RefCell<Host>>;

/// Reads the fake host's sysctl.
struct Source(Shared);

impl WiredLimitSource for Source {
    fn read_mb(&self) -> io::Result<u64> {
        Ok(self.0.borrow().value)
    }
}

/// Runs ops on the fake host.
struct Runner(Shared);

impl PrivilegedRunner for Runner {
    fn run(&mut self, op: &HostOp) -> io::Result<()> {
        let mut host = self.0.borrow_mut();
        host.ops.push(op.clone());
        if host.deny {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "authorization was refused",
            ));
        }
        match op {
            HostOp::SetWiredLimit { mb } => {
                let record = host
                    .store
                    .load()
                    .unwrap()
                    .expect("a record before the write");
                assert!(
                    record.new_mb == *mb || record.original_mb == *mb,
                    "the record does not name {mb}: {record:?}"
                );
                host.value = *mb;
            }
            HostOp::RemoveBootJob(_) => host.job_installed = false,
        }
        Ok(())
    }
}

#[derive(Default)]
struct Reprobes(Vec<u64>);

impl Reprobe for Reprobes {
    fn limit_changed(&mut self, limit_mb: u64) {
        self.0.push(limit_mb);
    }
}

struct FixedClock;

impl Clock for FixedClock {
    fn now(&self) -> MonoTime {
        MonoTime(0)
    }

    fn unix_ms(&self) -> u64 {
        1_790_000_000_000
    }
}

/// One test's state directory, its record, and a fake host whose sysctl reads `value`.
struct Fixture {
    store: TuneStore,
    host: Shared,
    source: Source,
    runner: Runner,
    reprobe: Reprobes,
}

impl Fixture {
    fn new(name: &str, value: u64) -> Fixture {
        let dir =
            std::env::temp_dir().join(format!("syntropy-tier-tune-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let store = TuneStore::in_state_dir(&dir);
        let host = Rc::new(RefCell::new(Host {
            value,
            ops: Vec::new(),
            deny: false,
            job_installed: false,
            store: store.clone(),
        }));
        Fixture {
            store,
            source: Source(Rc::clone(&host)),
            runner: Runner(Rc::clone(&host)),
            host,
            reprobe: Reprobes::default(),
        }
    }

    fn tune(&mut self) -> Tune<'_, Source, Runner, Reprobes, FixedClock> {
        Tune {
            store: &self.store,
            source: &self.source,
            runner: &mut self.runner,
            reprobe: &mut self.reprobe,
            clock: &FixedClock,
            version: "0.0.0-test",
        }
    }

    fn value(&self) -> u64 {
        self.host.borrow().value
    }

    fn set_by_hand(&self, mb: u64) {
        self.host.borrow_mut().value = mb;
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if let Some(dir) = self.store.path().parent() {
            let _ = fs::remove_dir_all(dir);
        }
    }
}

fn job() -> BootJob {
    BootJob {
        label: "com.syntropy.wired-limit".to_owned(),
        plist: PathBuf::from("/Library/LaunchDaemons/com.syntropy.wired-limit.plist"),
    }
}

/// RFC-0020 §9.1 step 1: the smallest whole MiB that holds the plan's `Unified` bytes, never
/// above RAM minus the reserve.
#[test]
fn the_proposal_is_the_smallest_limit_that_fits_under_the_reserve() {
    let ram = Bytes(512 * GIB);
    let headroom = Bytes(32 * GIB);
    let current = Bytes(384 * GIB);
    // 380 GiB already fits under 384 GiB.
    assert_eq!(
        propose(Bytes(380 * GIB), current, ram, headroom),
        Ok(Proposal::Fits)
    );
    // 400 GiB plus one byte rounds up to the next MiB.
    assert_eq!(
        propose(Bytes(400 * GIB + 1), current, ram, headroom),
        Ok(Proposal::Raise {
            limit_mb: 400 * 1024 + 1
        })
    );
    // Exactly RAM minus the reserve fits; one byte more does not.
    assert_eq!(
        propose(Bytes(480 * GIB), current, ram, headroom),
        Ok(Proposal::Raise {
            limit_mb: 480 * 1024
        })
    );
    assert_eq!(
        propose(Bytes(480 * GIB + 1), current, ram, headroom),
        Err(TuneError::TooLarge {
            needed: Bytes(480 * GIB + 1),
            ceiling: Bytes(480 * GIB),
        })
    );
    // A reserve larger than RAM leaves no room at all.
    assert!(matches!(
        propose(Bytes(1), Bytes(0), Bytes(GIB), Bytes(2 * GIB)),
        Err(TuneError::TooLarge { .. })
    ));
}

/// The criterion's core, on a fake host: `tune` then `tune --revert` restores the recorded
/// value. The record is written before the sysctl, and the re-probe hears both changes.
#[test]
fn tune_then_revert_restores_the_recorded_value() {
    let mut f = Fixture::new("round-trip", 0);
    let record = f.tune().apply(400 * 1024, "plan p1 needs 400 GiB").unwrap();
    assert_eq!(
        (
            record.original_mb,
            record.new_mb,
            record.state,
            record.at_unix_ms
        ),
        (0, 400 * 1024, RecordState::Applied, 1_790_000_000_000)
    );
    assert_eq!(f.value(), 400 * 1024);
    assert_eq!(f.tune().status().unwrap(), Some(record));

    let reverted = f.tune().revert().unwrap();
    assert_eq!(
        reverted,
        Reverted::Done {
            value: ValueRevert::Restored { to_mb: 0 },
            removed_job: None
        }
    );
    assert_eq!(f.value(), 0);
    assert_eq!(f.reprobe.0, [400 * 1024, 0]);
    assert_eq!(
        f.tune().status().unwrap().map(|r| r.state),
        Some(RecordState::Reverted)
    );
    // A second revert has nothing to do.
    assert_eq!(f.tune().revert().unwrap(), Reverted::NothingRecorded);
    assert_eq!(f.host.borrow().ops.len(), 2);
}

/// RFC-0020 §9.1 step 6: a value changed by hand since tune wrote it is left alone, and the
/// revert says so. The launchd job is still removed, because it would re-apply tune's value at
/// boot.
#[test]
fn a_hand_edited_value_is_left_alone_and_the_job_still_goes() {
    let mut f = Fixture::new("hand-edited", 0);
    f.tune().apply(300 * 1024, "plan p2").unwrap();
    f.host.borrow_mut().job_installed = true;
    f.tune().attach_boot_job(job()).unwrap();
    f.set_by_hand(350 * 1024);

    let reverted = f.tune().revert().unwrap();
    assert_eq!(
        reverted,
        Reverted::Done {
            value: ValueRevert::HandEdited {
                current_mb: 350 * 1024,
                recorded_mb: 300 * 1024
            },
            removed_job: Some(job())
        }
    );
    assert_eq!(f.value(), 350 * 1024, "the hand-set value stays");
    assert!(!f.host.borrow().job_installed);
    assert_eq!(
        f.reprobe.0,
        [300 * 1024],
        "no re-probe: the value did not change"
    );
}

/// Uninstall calls the same revert: it restores the value and removes the launchd job the
/// record lists (RFC-0052 §11.2 step 4).
#[test]
fn uninstall_reverts_the_value_and_removes_the_boot_job() {
    let mut f = Fixture::new("uninstall", 0);
    f.tune().apply(200 * 1024, "plan p3").unwrap();
    f.host.borrow_mut().job_installed = true;
    let record = f.tune().attach_boot_job(job()).unwrap();
    assert_eq!(record.boot_job, Some(job()));

    let reverted = f.tune().revert().unwrap();
    assert_eq!(
        reverted,
        Reverted::Done {
            value: ValueRevert::Restored { to_mb: 0 },
            removed_job: Some(job())
        }
    );
    assert_eq!(f.value(), 0);
    assert!(!f.host.borrow().job_installed);
    assert_eq!(
        f.host.borrow().ops,
        [
            HostOp::SetWiredLimit { mb: 200 * 1024 },
            HostOp::SetWiredLimit { mb: 0 },
            HostOp::RemoveBootJob(job()),
        ]
    );
}

/// A second tune keeps the first original while the value is still Syntropy's, so a revert goes
/// all the way back. After a hand edit, the hand-set value is the new original.
#[test]
fn a_second_tune_keeps_the_first_original() {
    let mut f = Fixture::new("second", 0);
    f.tune().apply(100 * 1024, "plan a").unwrap();
    let record = f.tune().apply(150 * 1024, "plan b").unwrap();
    assert_eq!((record.original_mb, record.new_mb), (0, 150 * 1024));

    f.set_by_hand(120 * 1024);
    let record = f.tune().apply(160 * 1024, "plan c").unwrap();
    assert_eq!(
        (record.original_mb, record.new_mb),
        (120 * 1024, 160 * 1024)
    );
    f.tune().revert().unwrap();
    assert_eq!(f.value(), 120 * 1024);
}

/// A refused approval changes nothing. The record says the change was abandoned, the re-probe
/// hears nothing, and a revert has nothing to do.
#[test]
fn a_refused_approval_abandons_the_change() {
    let mut f = Fixture::new("denied", 0);
    f.host.borrow_mut().deny = true;
    let error = f.tune().apply(100 * 1024, "plan d").unwrap_err();
    assert!(matches!(error, TuneError::Run(_)), "{error:?}");
    assert_eq!(f.value(), 0);
    assert_eq!(
        f.tune().status().unwrap().map(|r| r.state),
        Some(RecordState::Abandoned)
    );
    assert!(f.reprobe.0.is_empty());
    f.host.borrow_mut().deny = false;
    assert_eq!(f.tune().revert().unwrap(), Reverted::NothingRecorded);
}

/// A refused re-tune leaves the live record as it was, so its revert still restores the
/// original and removes the job (review of #1822, B1 case a).
#[test]
fn a_refused_retune_keeps_the_live_record() {
    let mut f = Fixture::new("refused-retune", 0);
    f.tune().apply(400 * 1024, "plan f").unwrap();
    f.host.borrow_mut().job_installed = true;
    let live = f.tune().attach_boot_job(job()).unwrap();
    f.host.borrow_mut().deny = true;
    let error = f.tune().apply(500 * 1024, "plan g").unwrap_err();
    assert!(matches!(error, TuneError::Run(_)), "{error:?}");
    assert_eq!(f.tune().status().unwrap(), Some(live));

    f.host.borrow_mut().deny = false;
    assert_eq!(
        f.tune().revert().unwrap(),
        Reverted::Done {
            value: ValueRevert::Restored { to_mb: 0 },
            removed_job: Some(job())
        }
    );
    assert_eq!(f.value(), 0);
    assert!(!f.host.borrow().job_installed);
}

/// A re-tune after a hand edit takes the hand-set value as its original and keeps the launchd
/// job listed, so a revert restores the hand-set value and removes the job (review of #1822,
/// B1 case b).
#[test]
fn a_retune_after_a_hand_edit_keeps_the_job() {
    let mut f = Fixture::new("hand-retune", 0);
    f.tune().apply(400 * 1024, "plan h").unwrap();
    f.host.borrow_mut().job_installed = true;
    f.tune().attach_boot_job(job()).unwrap();
    f.set_by_hand(300 * 1024);
    let record = f.tune().apply(450 * 1024, "plan i").unwrap();
    assert_eq!(
        (record.original_mb, record.new_mb, record.boot_job),
        (300 * 1024, 450 * 1024, Some(job()))
    );
    assert_eq!(
        f.tune().revert().unwrap(),
        Reverted::Done {
            value: ValueRevert::Restored { to_mb: 300 * 1024 },
            removed_job: Some(job())
        }
    );
    assert!(!f.host.borrow().job_installed);
}

/// The pending record a re-tune to `new_mb` leaves when it is killed between its record and
/// the write (a crash, or Ctrl-C at the approval prompt), over the applied `live` record.
fn interrupted_retune(live: &TuneRecord, new_mb: u64) -> TuneRecord {
    TuneRecord {
        new_mb,
        prior_mb: Some(live.new_mb),
        state: RecordState::Pending,
        reason: "interrupted".to_owned(),
        ..live.clone()
    }
}

/// After an interrupted re-tune, the value is still the earlier tune's. Revert knows it from
/// the record's `prior_mb`, so it restores the original and removes the job, rather than
/// calling the value hand-edited (review of #1822).
#[test]
fn a_revert_after_an_interrupted_retune_restores_the_original() {
    let mut f = Fixture::new("interrupted", 0);
    f.tune().apply(400 * 1024, "plan j").unwrap();
    f.host.borrow_mut().job_installed = true;
    let live = f.tune().attach_boot_job(job()).unwrap();
    assert_eq!(live.prior_mb, None, "an applied record has no prior value");
    f.store
        .save(&interrupted_retune(&live, 500 * 1024))
        .unwrap();
    assert_eq!(
        f.tune().revert().unwrap(),
        Reverted::Done {
            value: ValueRevert::Restored { to_mb: 0 },
            removed_job: Some(job())
        }
    );
    assert_eq!(f.value(), 0);
    assert!(!f.host.borrow().job_installed);
}

/// A re-tune after an interrupted one still counts the value as tune's, so it keeps the first
/// original and the job, and its applied record drops `prior_mb`.
#[test]
fn a_retune_after_an_interrupted_one_keeps_the_first_original() {
    let mut f = Fixture::new("interrupted-retune", 0);
    f.tune().apply(400 * 1024, "plan k").unwrap();
    let live = f.tune().attach_boot_job(job()).unwrap();
    f.store
        .save(&interrupted_retune(&live, 500 * 1024))
        .unwrap();
    let record = f.tune().apply(450 * 1024, "plan l").unwrap();
    assert_eq!(
        (
            record.original_mb,
            record.new_mb,
            record.prior_mb,
            record.boot_job
        ),
        (0, 450 * 1024, None, Some(job()))
    );
}

/// A pending record left by a crash before the write: the value is still the original, so a
/// revert writes nothing and closes the record.
#[test]
fn a_pending_record_whose_write_never_happened_reverts_without_a_write() {
    let mut f = Fixture::new("pending", 0);
    let record = TuneRecord {
        schema: SCHEMA,
        sysctl: WIRED_LIMIT_SYSCTL.to_owned(),
        original_mb: 0,
        new_mb: 64 * 1024,
        prior_mb: None,
        at_unix_ms: 1,
        reason: "crashed".to_owned(),
        by_version: "0.0.0-test".to_owned(),
        state: RecordState::Pending,
        boot_job: None,
    };
    f.store.save(&record).unwrap();
    assert_eq!(
        f.tune().revert().unwrap(),
        Reverted::Done {
            value: ValueRevert::AlreadyOriginal { mb: 0 },
            removed_job: None
        }
    );
    assert!(f.host.borrow().ops.is_empty());
    assert!(f.reprobe.0.is_empty());
}

/// Tuning to the current value is refused, and a boot job attaches only to an applied change.
#[test]
fn no_op_tunes_and_orphan_jobs_are_refused() {
    let mut f = Fixture::new("refused", 2048);
    assert_eq!(
        f.tune().apply(2048, "same").unwrap_err(),
        TuneError::AlreadyAt { mb: 2048 }
    );
    assert_eq!(
        f.tune().attach_boot_job(job()).unwrap_err(),
        TuneError::NotApplied
    );
    assert!(f.host.borrow().ops.is_empty());
}

/// The record's file: its field names, as `--status` and the daemon read them, and a save that
/// leaves no temporary file behind.
#[test]
fn the_record_is_json_with_stable_field_names() {
    let mut f = Fixture::new("json", 0);
    f.tune().apply(128 * 1024, "plan e").unwrap();
    let text = fs::read_to_string(f.store.path()).unwrap();
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    for key in [
        "schema",
        "sysctl",
        "original_mb",
        "new_mb",
        "prior_mb",
        "at_unix_ms",
        "reason",
        "by_version",
        "state",
        "boot_job",
    ] {
        assert!(json.get(key).is_some(), "missing {key}: {text}");
    }
    assert_eq!(json["state"], "applied");
    assert_eq!(json["sysctl"], "iogpu.wired_limit_mb");
    assert!(!f.store.path().with_extension("json.next").exists());
}

/// A record written before `prior_mb` existed still loads, with no prior value.
#[test]
fn a_record_without_prior_mb_loads() {
    let f = Fixture::new("no-prior", 0);
    fs::write(
        f.store.path(),
        r#"{"schema":1,"sysctl":"iogpu.wired_limit_mb","original_mb":0,"new_mb":1024,
            "at_unix_ms":1,"reason":"r","by_version":"v","state":"applied","boot_job":null}"#,
    )
    .unwrap();
    let record = f.store.load().unwrap().unwrap();
    assert_eq!((record.new_mb, record.prior_mb), (1024, None));
}

/// The one command that writes the sysctl.
#[test]
fn the_write_command_names_the_sysctl_and_the_value() {
    let argv: Vec<String> = HostOp::set_wired_limit_argv(4096)
        .into_iter()
        .map(|arg| arg.into_string().unwrap())
        .collect();
    assert_eq!(
        argv,
        ["/usr/sbin/sysctl", "-w", "iogpu.wired_limit_mb=4096"]
    );
}
