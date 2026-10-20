# ADR-0508: Updates: a TUF client, side-by-side versions, self-test, atomic switch, automatic rollback
- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers
- Related: ADR-0009, ADR-0011, ADR-0015, ADR-0018, ADR-0316, ADR-0507, ADR-0510; RFC-0001 §3.4; RFC-0034 §12; RFC-0052 §7, §9; RFC-0054 §8; RFC-0062 §7, §9, §10

## Context
Syntropy runs as a long-lived service on personal machines and home clusters, so updates carry several risks.
- **Interrupting work.** An update could interrupt an agent mid-session, or force a minutes-long reload of a 400 GB model.
- **Breaking a node.** A new version that fails on this hardware could leave a machine broken while its owner is away.
- **Splitting a cluster.** Nodes that disagree on protocol majors are refused at admission (ADR-0009, ADR-0011, RFC-0062 §10).
- **Network attacks.** Updates can be attacked through compromised mirrors, replayed old metadata (rollback and freeze), or mixed artefacts.

The research recommends updating the app and backend modules separately, with a signed manifest listing module hashes per platform (`backends-and-portability.md` §5.4).

Responsibility is split three ways.
- **RFC-0034 §12** (ADR-0316) owns the keys:
  - TUF roles: root (offline, 2 of 3 hardware tokens), targets (offline release ceremony), snapshot and timestamp (online; the timestamp re-signed daily), and a delegated `nightly` targets role, path-scoped to `nightly/*` and held online in the KMS (RFC-0034 §12.1);
  - Ed25519 release and nightly keys for detached signatures, which first-run checks use before any TUF metadata is reachable.
- **RFC-0062 §7 and §9** sign in protected release environments, publish channels, promote without rebuilding, and keep previous releases published for rollback.
- **The client** is this ADR's subject.

ADR-0507 makes modules lockstep TUF targets. RFC-0001 §3.4 plans runner re-adoption by a restarted daemon in M3, which it calls "also the in-place upgrade path".

## Options considered
| Option | For | Against |
|---|---|---|
| A. Manual updates only | no updater code | security fixes do not reach users; clusters drift apart |
| B. A single Ed25519-signed manifest, verified for version monotonicity | simple; one key | no defence against freeze (an old, validly signed manifest replayed by a mirror); rotation is ad hoc |
| C. A TUF client (`tough`) over RFC-0034's roles, with detached release-key signatures for the first run; side-by-side version directories, a self-test before switching, an atomic switch, runner re-adoption and automatic rollback | standard defences against rollback, freeze, mix-and-match and mirror compromise; untrusted mirrors; a working version at all times; most updates keep models loaded | four roles to operate, and a daily timestamp re-signing that RFC-0062 must run |
| D. Platform updaters (Sparkle on macOS, MSIX on Windows) | native | two or three mechanisms; no Linux story; MSIX restricts services |
| E. Package managers only | the distribution handles trust | not available for script or direct installs; most Windows and macOS users install directly |

## Decision
1. **Option C for self-updating installs.**
   - Script, app zip and MSI installs self-update.
   - deb, rpm, Homebrew cask and winget installs report available updates and the package command, and never self-write.
   - Containers and offline bundles do not update themselves.
2. **Verification.**
   - The update client is a TUF client whose embedded root metadata is the trust anchor. It refreshes timestamp → snapshot → targets, enforcing expiry and monotonicity.
   - `install.sh`, `install.ps1` and `setup`'s first self-check verify detached release-key signatures before any TUF metadata is reachable (RFC-0034 §12.2).
   - The module loader re-checks each module's hash and build ID immediately before `dlopen`.
3. **Server-side rollback without weakening monotonicity.** Targets metadata may list `withdrawn` versions. A node running a withdrawn version rolls back locally to its previous version, or installs the channel's older target as a signed, explicit downgrade. An older version is otherwise never installed automatically.
4. **Channels** `stable` (default), `beta` and `nightly`, as RFC-0062 publishes them from M1. `stable` and `beta` are paths under the offline targets role. `nightly` is the delegated role, so a nightly key can never sign a stable or beta path; if the client's delegation support proves insufficient (spike PROD-E12-T11), nightly builds use a separate TUF repository whose root only nightly builds embed. Switching channels never downgrades.
5. **Applying an update.**
   1. Stage into a new version directory.
   2. Verify.
   3. Run the new binary's `__selftest`: config parse with the new schema, state migrations on a copy, module load in a probe process, and a CPU parity smoke test.
   4. Back up the state and config files a rollback needs.
   5. Switch the `current` link atomically.
   6. Restart the daemon. When the IPC protocol is unchanged, runners are re-adopted and loaded models stay loaded. Otherwise runners drain first.
   7. Hold a 5-minute health window.
   8. Commit, keeping the two previous versions.
   9. Refresh old-build runners at the next idle window.
   The default window is "idle": no live sequence and no request for 5 minutes, on mains power.
6. **Automatic rollback.**
   - **Triggers**, within the health window: a crash loop, readiness not reached within 60 s, a canary failure on an instance that passed before, or the cluster refusing the node.
   - **The agent** is the update guard: a copy of the last committed build that every service unit runs before the daemon. It therefore never depends on the new build starting.
   - **The window** stays armed until every instance has at least one runner refreshed onto the new build with a passing canary, so kernel regressions are caught before commit.
   - **The action**: switch back, restore the state and config backups, and mark the version bad locally. `syntropy update rollback` does the same by hand.
   - **Before restarting**, the node drains local-ingress sequences for up to `update.local_drain`. From M3 it also announces a planned restart (`Leave { reason: Restart { back_within_ms } }`, RFC-0030 §5.9), so peers suspend suspicion. Before M3 it shuts down gracefully, and peers recover its stages.
   - **System installs** default to `notify`, because the unprivileged daemon cannot write its install root. They always include a privileged helper: a root oneshot unit on Linux, a demand-start LocalSystem service on Windows. The helper performs commit, the Windows rollback switch, and `auto`-mode applies on the daemon's request, after re-verifying every target with its own TUF client. On Linux the guard runs privileged (`ExecStartPre=+`) and switches back itself. macOS `.pkg` installs update through the package or MDM (RFC-0052 §7.1).
7. **Clusters.** Minor and patch updates roll node by node, within RFC-0062 §10.2's support window. A protocol-major update is staged and self-tested on every node first, then switched together at a coordinator-chosen idle time (`update.switch_at`). A node that fails its self-test holds the switch.

## Consequences
- **Positive.**
  - Security fixes reach direct installs without user effort.
  - A bad release reverts itself, locally or from the server, and no unsigned or older version is ever installed silently.
  - Most updates do not unload models.
  - Mirrors and CDNs need no trust.
- **Negative.**
  - The project must sign TUF metadata, including a daily timestamp and a targets re-signing ceremony before each expiry, which RFC-0062 §7.1 schedules. A lapsed timestamp or targets role makes clients refuse updates until it is re-signed, which is safe but visible.
  - Two retained versions cost disk: the core plus modules, up to a few GiB on GPU hosts.
  - Re-adoption across builds needs RFC-0001's `Hello` rule and RFC-0062 §10.1's IPC row relaxed to IPC-protocol equality (RFC-0052 amendment A1).
- **Work created.**
  - RFC-0052: client, procedure, rollback and coordination.
  - RFC-0062: TUF metadata production, `withdrawn` lists, keeping previous releases published.
  - RFC-0034: custom target fields (build ID, protocol majors, provider ABI, `min_from_version`) within its targets role.
  - RFC-0060: fault-injection tests for every step.
  - RFC-0054: `update.*` keys and the config backup and restore rule.

## What would make us revisit
- Automatic rollback fires on healthy updates (false positives), for example because canaries are flaky. Tighten the health criteria before relaxing anything else.
- TUF operations prove too heavy to run: the offline ceremonies (**Maintainer action**), or the daily timestamp job in operation. Then RFC-0034 and RFC-0062 move to option B, with a long-lived `expires` field, and the client follows. The weaker freeze protection is documented.
- Platforms gain update mechanisms that support services, memory wiring and our layout. Consider delegating to them.
