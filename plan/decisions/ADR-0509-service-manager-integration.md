# ADR-0509: Service manager integration: user-level services on personal installs, system services for packages and per-machine installs
- Status: Accepted (2026-10-20); revised 2026-10-20 (see Revisions)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the product surface and operations area)
- Related: ADR-0007, ADR-0015, ADR-0020; RFC-0001 §3.4, §9.2; RFC-0052 §5, §6; RFC-0054 §1

## Context
The daemon must keep running so loaded models stay loaded and the node stays in its cluster. It must also start on boot or login, restart on failure, and drain on stop (RFC-0001 §3.4).

Each OS offers a user-level option and a system-level option, with different trade-offs:

- **Linux.**
  - *systemd user units* run with the user's files and GPU permissions. They stop at logout unless lingering is enabled. Their memlock limit comes from the user manager, which is typically small.
  - *System units* start at boot and can hold `CAP_IPC_LOCK` with an unlimited memlock limit. They need root to install, and they run under a service account with its own data directories.
- **macOS.**
  - *LaunchAgents* run in the user's session. Local Network privacy prompts reach them when they live in an app bundle (Apple TN3179).
  - *LaunchDaemons* run at boot without a user. Apple's TN3179 (revised 2026-10-06) lists any daemon started by launchd as allowed local-network access, as a separate item from programs running as root; only launchd agents are excluded. Developer forum threads report prompt bugs for daemons, so the behaviour needs a test.
  - *Automatic login* is unavailable while FileVault is on (Apple support article 102316), and FileVault is on for most Macs. A LaunchAgent on a headless Mac therefore cannot be the documented answer without telling users to turn off disk encryption.
- **Windows.**
  - *A logon task* runs as the user without admin.
  - *A Windows service* starts at boot under a service account and needs admin to install.

Two more facts matter:
- The users are owners of personal machines (vision §4): gaming PCs, laptops and desktop Macs that also run their own work. Headless boxes (a mini PC under the TV, a Mac Studio used as a server) are a significant minority.
- Pinned memory is the RAM tier (ADR-0007). On Linux the memlock limit decides how much of it Syntropy can lock (RFC-0001 §9.2).

## Options considered
| Option | For | Against |
|---|---|---|
| A. System service everywhere | always on; full privileges for pinning | admin or root for every install; on macOS either root or unanswerable prompts; models and state under a service account the user cannot easily inspect |
| B. User service everywhere | no admin; runs in the user's context and GPU session; simplest mental model | Linux needs lingering for boot start and a memlock change for large pinning; headless Macs need automatic login |
| C. User-level by default for personal installs (script, app, per-user MSI); system-level for distribution packages, `--system` scripts and per-machine MSIs | matches how each install kind is used; admin only when the user already chose an admin install | two configurations per OS to test and document |
| D. No service; the CLI starts the daemon on demand | nothing to install | the daemon stops at logout or reboot; models unload; the node leaves its cluster unpredictably |

## Decision
**Option C.**

| Install kind | Service |
|---|---|
| Linux `install.sh` | systemd user unit. `setup` asks to enable lingering, recording the prior state in the host-change ledger, and offers a memlock raise through a `limits.d` file or a user-manager drop-in, with sudo and consent |
| Linux `.deb`, `.rpm`, `install.sh --system` | systemd system unit, `syntropy` system user in `video` and `render`, `LimitMEMLOCK=infinity`, `CAP_IPC_LOCK`, `Type=notify` with a watchdog |
| macOS app, cask, script | LaunchAgent running the update guard, which execs the app bundle's executable; `AbandonProcessGroup = true` |
| macOS server option (`syntropy service install --daemon` from M1, the `.pkg` from M2) | LaunchDaemon with `UserName` set to a dedicated `_syntropy` account or one the administrator names, never root by default (revised 2026-10-20). It ships once the M1 spike (RFC-0052 §6.2) passes with FileVault on. If the spike finds prompts or no Metal device for a non-root daemon, the fallback is a root LaunchDaemon that drops privileges after start, tested in the same spike. Automatic login is withdrawn as the documented answer |
| Windows per-user (MSI, script) | logon task starting the GUI-subsystem launcher `syntropyw.exe`, which runs the daemon without a console, restarts it on failure, and handles logoff through a hidden window |
| Windows per-machine MSI | Windows service under the virtual account `NT SERVICE\Syntropy` |
| containers | none; `syntropy serve` is PID 1 |

Common rules:
- Every service drains on stop, with the stop timeout at the drain deadline plus margin.
- Every service restarts on failure but not after `syntropy stop`.
- Every service is recorded in the host-change ledger, so `uninstall` removes it.
- **Runners may outlive a daemon restart**, but only for a bounded time, and only where re-adoption is configured (RFC-0052 §6.5):
  - systemd: `KillMode=process`;
  - launchd: `AbandonProcessGroup = true`, plus `setsid()` in runners;
  - Windows: a daemon-owned job without kill-on-close, plus breakaway from the logon task's job.
  Where re-adoption is unavailable, updates drain and restart.
- **Each unit runs the update guard before the daemon** (RFC-0052 §7.4), so a crash-looping new build is rolled back by the previous build's code. On system installs the guard needs privilege the daemon lacks. The systemd unit runs it with `ExecStartPre=+`, and the Windows service's recovery action hands the switch to the `SyntropyUpdate` helper service (RFC-0052 §6.1, §6.3, §7.1).

## Consequences
- **Positive.**
  - A personal install needs no admin rights, except for optional steps the user approves (lingering, memlock).
  - Packaged and per-machine installs behave like conventional system services.
  - macOS prompts are attributed to the app and answerable.
- **Negative.**
  - On Linux user installs, declining the memlock raise leaves Syntropy with a small pinned budget. RFC-0020 then has to rely on the GPU driver's registration paths, or fall back to pageable memory, and doctor OS-001 must explain the cost clearly.
  - Two service configurations per OS double the install-matrix rows.
  - With FileVault on, a headless Mac starts nothing after a reboot until its disk is unlocked. The docs say so, and point to unlocking over SSH before boot on macOS 26 (a third-party report, to be checked in the spike).
- **Work created.**
  - RFC-0052: unit, plist and task definitions, setup prompts, ledger entries.
  - RFC-0053: doctor INS-002 and OS-001.
  - RFC-0060 and RFC-0062: install-matrix rows for both kinds on every OS.

## What would make us revisit
- Install telemetry is ruled out (ADR-0510). If issue reports show most Linux users on user installs declining lingering or memlock and then filing performance issues, make the system service the Linux default for `install.sh`.
- The headless-Mac spike, or later field reports, show a non-root LaunchDaemon losing LAN or Metal access on a macOS release. Move to the privilege-dropping root daemon on that release, and say so in the release notes.
- Windows logon tasks prove unreliable at restarting after crashes or sleep. Move per-user installs to a service that runs as the user.

## Revisions
- Revised 2026-10-20: the macOS server option is a LaunchDaemon with `UserName` set to a dedicated `_syntropy` account, offered once the M1 spike in RFC-0052 §6.2 passes with FileVault on; a root LaunchDaemon that drops privileges is the fallback. Automatic login is withdrawn as the documented headless answer, because macOS makes it unavailable while FileVault is on, so recommending it told server users to turn off disk encryption. Context and the revisit trigger are updated to match.
