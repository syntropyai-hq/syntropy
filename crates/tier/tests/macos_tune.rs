// SPDX-License-Identifier: MIT OR Apache-2.0

//! TIER-E2-T13's Mac integration criterion, run for real where it is safe: on a GitHub-hosted
//! macOS runner, which is a throwaway VM with passwordless `sudo`.
//!
//! The test changes the real `iogpu.wired_limit_mb` and installs a real launchd job, so it runs
//! only when `GITHUB_ACTIONS` is `true` and `RUNNER_ENVIRONMENT` is `github-hosted`. Anywhere
//! else, a developer's Mac included, it does nothing. Its runner stands in for the CLI's
//! authorization prompt (PROD-E6-T14): it runs each [`HostOp`] with `sudo -n`.
//!
//! On a hosted runner it checks the criterion end to end:
//! - `tune` then `tune --revert` restores the recorded value;
//! - the revert, which uninstall also calls, removes the launchd job the record lists;
//! - a value changed by hand since tune wrote it is left alone.
//!
//! It reports what it did, or why it skipped, as a GitHub Actions notice. The notice is written
//! straight to the standard error stream, which the test harness does not capture, so it shows
//! in the job's log.

#![cfg(target_os = "macos")]

use std::cell::RefCell;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use syntropy_core::{Clock, MonoTime};
use syntropy_tier::os::macos::{SysctlWiredLimit, sysctl_u64};
use syntropy_tier::{
    BootJob, HostOp, PrivilegedRunner, Reprobe, Reverted, Tune, TuneStore, ValueRevert,
    WiredLimitSource,
};

/// Writes a GitHub Actions notice past the test harness's capture.
fn notice(message: &str) {
    let _ = writeln!(
        io::stderr(),
        "::notice title=TIER-E2-T13 wired-limit round trip::{message}"
    );
}

fn on_a_hosted_runner() -> bool {
    std::env::var("GITHUB_ACTIONS").as_deref() == Ok("true")
        && std::env::var("RUNNER_ENVIRONMENT").as_deref() == Ok("github-hosted")
}

/// Runs `argv` as root with `sudo -n`, which never prompts.
fn sudo<S: AsRef<std::ffi::OsStr>>(argv: &[S]) -> io::Result<()> {
    let status = Command::new("/usr/bin/sudo")
        .arg("-n")
        .args(argv)
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other(format!("sudo {status}")))
    }
}

/// The CLI's approved runner, as `sudo -n`.
struct SudoRunner;

impl PrivilegedRunner for SudoRunner {
    fn run(&mut self, op: &HostOp) -> io::Result<()> {
        match op {
            HostOp::SetWiredLimit { mb } => sudo(&HostOp::set_wired_limit_argv(*mb)),
            HostOp::RemoveBootJob(job) => {
                // Not loaded is fine: the job is to be gone either way.
                let _ = sudo(&[
                    "/bin/launchctl".to_owned(),
                    "bootout".to_owned(),
                    format!("system/{}", job.label),
                ]);
                sudo(&[Path::new("/bin/rm"), Path::new("-f"), job.plist.as_path()])
            }
        }
    }
}

#[derive(Default)]
struct Reprobes(RefCell<Vec<u64>>);

impl Reprobe for Reprobes {
    fn limit_changed(&mut self, limit_mb: u64) {
        self.0.get_mut().push(limit_mb);
    }
}

struct RealClock;

impl Clock for RealClock {
    fn now(&self) -> MonoTime {
        MonoTime(0)
    }

    fn unix_ms(&self) -> u64 {
        1_790_000_000_000
    }
}

fn job_loaded(label: &str) -> bool {
    sudo(&[
        "/bin/launchctl".to_owned(),
        "print".to_owned(),
        format!("system/{label}"),
    ])
    .is_ok()
}

/// Installs a launchd job that does nothing, as `--persist` would install the real one.
fn install_job(job: &BootJob, scratch: &Path) -> io::Result<()> {
    let plist = scratch.join("job.plist");
    std::fs::write(
        &plist,
        format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
             <!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \
             \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
             <plist version=\"1.0\"><dict>\
             <key>Label</key><string>{}</string>\
             <key>ProgramArguments</key><array><string>/usr/bin/true</string></array>\
             <key>RunAtLoad</key><false/>\
             </dict></plist>\n",
            job.label
        ),
    )?;
    sudo(&[Path::new("/bin/cp"), plist.as_path(), job.plist.as_path()])?;
    sudo(&[
        Path::new("/usr/sbin/chown"),
        Path::new("root:wheel"),
        job.plist.as_path(),
    ])?;
    sudo(&[
        Path::new("/bin/chmod"),
        Path::new("644"),
        job.plist.as_path(),
    ])?;
    sudo(&[
        Path::new("/bin/launchctl"),
        Path::new("bootstrap"),
        Path::new("system"),
        job.plist.as_path(),
    ])
}

/// Puts the host back if the test fails part way: the original value, and no job.
struct Restore {
    original: u64,
    job: BootJob,
}

impl Drop for Restore {
    fn drop(&mut self) {
        let _ = SudoRunner.run(&HostOp::SetWiredLimit { mb: self.original });
        let _ = SudoRunner.run(&HostOp::RemoveBootJob(self.job.clone()));
    }
}

#[test]
fn tune_then_revert_on_a_hosted_mac() {
    if !on_a_hosted_runner() {
        return;
    }
    let source = SysctlWiredLimit;
    let original = match source.read_mb() {
        Ok(mb) => mb,
        Err(error) => {
            notice(&format!(
                "skipped: iogpu.wired_limit_mb cannot be read here: {error}"
            ));
            return;
        }
    };
    if let Err(error) = sudo(&["/usr/bin/true"]) {
        notice(&format!("skipped: no passwordless sudo: {error}"));
        return;
    }
    let ram_mb = sysctl_u64("hw.memsize").unwrap() >> 20;
    let mut target = ram_mb / 2;
    if target == original {
        target += 1;
    }
    let scratch = std::env::temp_dir().join(format!("syntropy-tune-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let job = BootJob {
        label: format!("com.syntropy.test.wired-limit.{}", std::process::id()),
        plist: PathBuf::from(format!(
            "/Library/LaunchDaemons/com.syntropy.test.wired-limit.{}.plist",
            std::process::id()
        )),
    };
    let restore = Restore {
        original,
        job: job.clone(),
    };
    let store = TuneStore::in_state_dir(&scratch);
    let mut runner = SudoRunner;
    let mut reprobe = Reprobes::default();
    let mut tune = Tune {
        store: &store,
        source: &source,
        runner: &mut runner,
        reprobe: &mut reprobe,
        clock: &RealClock,
        version: env!("CARGO_PKG_VERSION"),
    };

    // `tune`, then `--persist`'s job, then `tune --revert`.
    let record = tune.apply(target, "hosted round trip").unwrap();
    assert_eq!((record.original_mb, record.new_mb), (original, target));
    assert_eq!(source.read_mb().unwrap(), target);
    install_job(&job, &scratch).unwrap();
    assert!(job_loaded(&job.label), "the test job did not load");
    tune.attach_boot_job(job.clone()).unwrap();
    let reverted = tune.revert().unwrap();
    assert_eq!(
        reverted,
        Reverted::Done {
            value: ValueRevert::Restored { to_mb: original },
            removed_job: Some(job.clone())
        }
    );
    assert_eq!(
        source.read_mb().unwrap(),
        original,
        "the recorded value is back"
    );
    assert!(!job_loaded(&job.label), "the job is still loaded");
    assert!(!job.plist.exists(), "the job's plist is still there");

    // A value changed by hand since tune wrote it is left alone.
    tune.apply(target, "hosted hand-edit check").unwrap();
    sudo(&HostOp::set_wired_limit_argv(target + 1)).unwrap();
    let reverted = tune.revert().unwrap();
    assert_eq!(
        reverted,
        Reverted::Done {
            value: ValueRevert::HandEdited {
                current_mb: target + 1,
                recorded_mb: target
            },
            removed_job: None
        }
    );
    assert_eq!(
        source.read_mb().unwrap(),
        target + 1,
        "the hand-set value stays"
    );

    drop(restore);
    assert_eq!(source.read_mb().unwrap(), original);
    let heard = reprobe.0.into_inner();
    let _ = std::fs::remove_dir_all(&scratch);
    notice(&format!(
        "ran: iogpu.wired_limit_mb {original} -> {target} -> {original} (launchd job \
         installed and removed); hand-edited {} left alone; re-probes {heard:?}",
        target + 1
    ));
}
