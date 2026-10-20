# ADR-0621: One product SemVer from conventional commits, managed by release-please; crate versions frozen; promotion without rebuild
- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted in the quality and release engineering area)
- Related: ADR-0015, ADR-0018, ADR-0019, ADR-0020, ADR-0021, ADR-0620, ADR-0624; RFC-0001 §3.5, §16; RFC-0034 §12; RFC-0052, RFC-0062 §8, §9, §13

## Context
Syntropy ships one binary, `syntropy`, and runtime-loaded backend modules that must match it exactly (RFC-0001 §3.5 `Hello`; RFC-0052). Workspace crates are not published. Versioning must:
- drive changelogs and release notes without hand editing;
- keep tags machine-created;
- make breaking changes deliberate;
- support patch releases on older lines.

Releases must ship exactly the bits that passed RFC-0060's and RFC-0061's gates. Release-please's Rust workspace handling is reported to have trouble with inherited workspace versions and with keeping `Cargo.lock` consistent, and a release PR that changes `Cargo.toml` versions also needs a lockfile commit before `--locked` builds pass.

## Options considered
1. **Per-crate versions** with release-please's Rust workspace strategy, or a Rust-specific release tool. That gives meaningful crate versions, but crates are not published, it causes `Cargo.lock` churn on every release, and inherited versions are reported to break it.
2. **Manual versioning and tags.** Error-prone, and contradicts the project's rule that tags are never hand-created.
3. **A semantic-release-style tool that publishes on every merge.** Every merge would then release, which is incompatible with a 48-hour RC gate.
4. **Chosen:**
   - release-please with the `simple` release type over `version.txt`;
   - conventional-commit PR titles with squash merges;
   - crate versions frozen at `0.0.0`;
   - an RC pipeline on the release PR's head, behind a short merge freeze;
   - channel promotion that copies signed bits.

## Decision
1. **Version source.** `version.txt` holds the product SemVer. `BuildId` (version, commit, profile, target, toolchain hash) is constructed in the top binary crate, `cli`, when the binary is built, and is passed down at run time; `core` holds only the type, so a commit does not invalidate every crate's compilation cache. Crates are `publish = false` with `version = "0.0.0"`, so releases never touch `Cargo.toml` or `Cargo.lock`.
2. **release-please.**
   - Configuration: `release-type: simple`, `bump-minor-pre-major: true`, `bump-patch-for-minor-pre-major: false`, changelog sections for feat, fix, perf and revert.
   - It runs on `main` and on `release/vX.Y` branches, the latter for patches.
   - It runs with an installation token of the **release App**, never with `GITHUB_TOKEN`, whose events start no workflow runs. So the tag it creates starts the RC (revised 2026-10-19).
   - *Revised 2026-10-20.* The release App is a second organisation-owned GitHub App, separate from the lab dispatcher's App, which holds only Contents read so that the lab's entry point cannot write to the repository. The release App holds only Contents write and Pull requests write, is installed on this repository only, and its key lives only in the `release-please` environment, restricted to `main` and `release/*`.
   - Tags `v*` are created only by release-please. A tag ruleset rejects any other creation, update or deletion, and its only bypass actor is the release App.
   - Rulesets on `main` and `release/*` block force-pushes and deletion. No agent identity can push to them, create a tag or approve its own change.
3. **Commits.**
   - Squash merges only. The PR title is a conventional commit with a crate or area scope.
   - A breaking marker (`!` or `BREAKING CHANGE:`) requires the `semver-major-approved` label, which only a maintainer applies (**Maintainer action**). Agents never add a breaking marker without it.
   - Every commit carries a DCO sign-off, checked by one `dco` check that treats agent-authored commits like any other. There is no separate rule for agent-written code.
   - Before 1.0, breaking changes bump the minor version.
   - 1.0.0 is cut at M4 with a maintainer-approved `Release-As: 1.0.0` footer.
   - An unwanted major proposal is pinned with `Release-As: <major>.<next-minor>.0` instead of editing tags.
4. **RC and channels** (the RC runs on the tag).
   - release-please creates a draft GitHub Release with its tag created immediately (`draft`, `force-tag-creation`).
   - A maintainer applies `rc:start`, which freezes merges to `main` (except `rc:fix`), and then merges the release PR (**Maintainer action**). release-please tags the merge commit.
   - The tag push builds, attests and signs every artefact from the tag's commit, runs every RC gate on those exact artefacts, and posts `rc-gate` on that commit. `BuildId::commit`, `SOURCE_DATE_EPOCH`, the attestations and the reproducibility drill all name the tagged commit.
   - A failed RC withdraws that version, which is never reused. The fix ships as the next patch version.
   - Each long suite carries its results over to that version's RC only under its own path rule. Build, signing, parity smoke and the release-gate benchmarks always re-run.
   - Publishing follows the release ceremony's signing of the TUF targets metadata with the offline targets keys. The ceremony is a **Maintainer action**, performed by the key-holding maintainers recorded in `governance/delegations.toml`; no agent holds or handles an offline key. The snapshot is signed by an online KMS key on publish, and the timestamp is re-signed daily by a scheduled job. Key roles and custody follow RFC-0034 §12.1 and ADR-0316, and CI never holds the root or targets keys.
   - Releases publish to `beta`, and promotion to `stable` after a 7-day soak copies the same artefacts and updates the signed stable metadata. All three channels exist from M1 (RFC-0052).
   - Nightly builds publish through a delegated `nightly/*` targets role, signed by the online nightly key in the `nightly-signing` environment (RFC-0052 §7.2, RFC-0034 §12.1). The offline top-level targets role is re-signed in an owner ceremony (**Maintainer action**) scheduled on a calendar, independent of releases, with an alert at 30 days before its 90-day expiry.
5. **Patches.** Backports go to `release/vX.Y` by label, and release-please maintains the patch release PR there. Patch RCs run a reduced gate set, unless the diff touches `kernels-*`, `exec`, `net`, `cluster` or `tier`.
6. **Nightly** builds of `main` are published to the `nightly` channel through the delegated nightly role (decision 4). They have no performance gate.
7. **Security releases** (added 2026-10-20). A fix for a High or Critical vulnerability takes an expedited path:
   - it is developed in a temporary private fork under the GitHub Security Advisory, by agents working in the isolated security environment (no secrets, no signing access, no way to publish; RFC-0062 §13) and merged only with a maintainer's explicit approval;
   - its RC covers the build, signing and verification, parity smoke on the shipping set, a regression test for the vulnerability, and the suites of the crates it touches;
   - it is published to `beta` and `stable` together, with no soak, and flagged `security` in the TUF targets metadata, so installs in `notify` mode show a banner in `doctor` and the dashboard (RFC-0052);
   - it rolls forward, never back: `withdrawn` (ADR-0508) would send nodes to a version that may be just as vulnerable.

   A 24-hour `beta` soak with a maintainer's sign-off is the recorded alternative, not the default.

## Consequences
- **Positive.**
  - One number describes what a user runs, and modules match it exactly.
  - Release PRs are small, conflict-free and build with `--locked`.
  - Changelogs come from commit history.
  - Breaking changes cannot slip in unlabelled.
  - Stable users get bits that are byte-identical to what passed the gates and soaked on beta.
  - A security fix reaches users without waiting for the 7-day soak.
- **Negative.**
  - Crate versions carry no information, so publishing a crate later (an SDK, for example) needs its own versioning decision.
  - The merge freeze blocks most merges to `main` for up to 48 hours per release.
  - Patch releases through release-please on release branches need a spike to confirm behaviour, moved to M0 (RFC-0062 Open question 6).
  - A security release skips the soak, so a regression in the fix reaches `stable` directly; the reduced RC set and the roll-forward rule bound it.
  - Two Apps, each with one key to rotate.

## What would make us revisit
- The project decides to publish crates, such as a client SDK or `syntropy-testkit`. Version those crates separately with release-please's Rust strategy and explicit per-crate versions, and keep the product version as is.
- Freezes exceed 48 hours on two consecutive releases, or contributors report them as a blocker. Then cut `release/vX.Y` at RC start, so the freeze moves off `main`.
- release-please cannot manage patch releases on release branches with this configuration (the M0 spike). Use the backport workflow's fallback described in RFC-0062.
- A security release introduces a regression that a soak would have caught. Then adopt the 24-hour `beta` soak for security releases.

## Revisions
- Revised 2026-10-20: decision 2 gives release-please its own release App, because the dispatcher App now holds only Contents read, and adds rulesets on `main` and `release/*`. Decision 3 records one `dco` check for every commit, agent-authored included. Decision 7 adds the expedited security-release path, so a fix does not wait for the 7-day soak. The release-branch spike moves to M0, while the cherry-pick fallback is cheapest to adopt.
