# ADR-0622: Compatibility policy: negotiated protocol levels with an adjacent-minor window; packs rebuild; golden vectors forever
- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted in the quality and release engineering area, with the areas of RFC-0030, RFC-0031, RFC-0041 and RFC-0022)
- Related: ADR-0005, ADR-0009, ADR-0010, ADR-0011, ADR-0015; RFC-0001 §3.5, §5.6, §5.8, §5.10; RFC-0060 §12.4; RFC-0062 §10

## Context
A Syntropy cluster is a set of personal machines whose owners upgrade them at different times. Requiring every node to run the same version would make clusters fragile. The survey records clusters that failed to form because version strings and builds had to match exactly, with only a log line to explain why (`multi-device-inference.md` §13 item 24). On the other hand:
- an unconstrained compatibility promise freezes formats and slows the project;
- packs are a rebuildable local cache (ADR-0005);
- the daemon↔runner IPC is local to one host, so its compatibility only matters across a daemon upgrade (RFC-0001 §3.5, RFC-0052 A1).

The surfaces need different rules, and every rule needs a test.

## Options considered
1. **Exact version match across the cluster.** Simplest, but every upgrade becomes a whole-cluster event, which contradicts the zero-configuration home-cluster promise (D4).
2. **Full backward compatibility across all versions.** Users never notice upgrades, but every format change carries forever, and the test matrix grows without bound.
3. **Feature negotiation per message**, with no version numbers. Flexible, but hard to test exhaustively and easy to get subtly wrong.
4. **Chosen: per-surface rules.**
   - Versioned wire and control protocols (`major.minor`), negotiated per channel and at admission. The cluster runs at the minimum minor.
   - From 1.0, nodes of adjacent minors interoperate within a major.
   - IPC versioned by protocol major for runner re-adoption.
   - Packs rebuilt on layout change, with the learned profile migrated.
   - Session blobs readable within a major when the identity hash matches.
   - Golden byte vectors for every protocol version in the window, checked on every PR.

## Decision
Revised 2026-10-19: decisions 1–3 adopt RFC-0031 §10's version model and N−1/N+1 window, and RFC-0041's pack pieces.

1. **Protocol** (RFC-0031 §10, ADR-0305), for all planes:
   - one `proto_major` (in the ALPN, beacons and `Hello`), a `wire_minor` in every frame, a control `feature_level`, and a 64-bit capability set;
   - the same `proto_major` is required;
   - `wire_minor` changes are additive, and a channel runs at the lower minor;
   - control event variants are emitted only up to the cluster's minimum `feature_level`, and capability-tied frames only when both sides advertise them;
   - every event kind ever written stays decodable for the whole major;
   - each release declares `[min_supported, current]` for `wire_minor` and `feature_level`, and admission requires overlapping ranges;
   - a mismatch refuses connection or admission with an error naming both versions and the upgrade command.
2. **Support window** (RFC-0031 §10's promise, N−1/N+1).
   - Within a `proto_major`, a release interoperates with releases one minor older and one minor newer (N−1 and N+1), on all planes, as RFC-0031 §10 now states. Rolling upgrades are supported across that window, and CI tests N−1 (N+1 is the next release's N−1).
   - A release's `min_supported` values may not exceed the `current` values of the previous minor release, which CI checks.
   - Before 1.0, `proto_major` may change in a minor release, announced.
   - *Revised 2026-10-20.* Before 1.0, a release that breaks interoperation with N−1 needs sign-off. A rolling-upgrade failure against N−1 fails `rc-gate` unless a maintainer has applied the `interop-break-approved` label to the release PR, and the release notes state the break and the upgrade path. Only a maintainer may add or remove the label (RFC-0062 §12.3). From 1.0 the failure blocks outright.
   - Patch releases never change protocol versions.
3. **Packs** (RFC-0041 §5.2): three versions.
   - **The pack container version** is written in segment headers, index records and sidecars. Only the current version is read. Packs are caches, so a change rebuilds every pack of the device class, whose extents are shared.
   - **Each side record's schema version** covers the expert profile, the selection entries and the canary `refdigest`. A side record migrates across one schema change, because each has decoders for its current and previous schema.
   - **`PackKey::layout`** is the layout-table version. A change rebuilds only the affected layouts' packs.
4. **Session blobs** (RFC-0022) are readable within a major when the model-and-settings identity hash matches. Otherwise they are dropped with a notice, and the session re-prefills.
5. **Other state.**
   - Store and state-directory layouts migrate forward on start, with a metadata backup. A newer layout than the binary understands makes it refuse to start, with a clear error.
   - Node keys and admission certificates never break within a major.
   - **`ModelId` is stable across releases** for the same source files (RFC-0041 §1). It changes only with a listed family-acceptance change, named in the release notes. Golden fixture IDs are checked in, and the `compat` check fails on an unlisted change.
   - The capability-profile cache is invalidated per build.
   - Config keys are deprecated for one minor (before 1.0) or one major (after) before removal.
   - Daemon↔runner IPC: same build to spawn, same `ipc_protocol` to adopt. `Hello` carries `ipc_protocol` and the observability table digest. A new daemon re-adopts running runners within `runner.orphan_grace` when `ipc_protocol` matches and the tables match or can be translated (RFC-0052 §6.5 and A1). Otherwise the update drains and restarts them. The IPC schema fingerprint covers the IPC messages, the ring record schemas and the observability table digest. A nightly mixed-version adoption test runs through each real service manager (RFC-0062 §10.4).
6. **APIs.** The native API and the CLI's `--json` output break only in product majors after 1.0. The OpenAI- and Anthropic-shaped APIs follow upstream, with additive extension fields. Backend modules match the core version exactly.
7. **Procedure.**
   - Versions and ranges live in `ci/protocol-versions.toml`. The `compat` check re-encodes golden vectors and also compares a committed schema fingerprint per surface (serde types plus frame, TLV, capability and event-variant registries), so an added variant without a version or feature-level bump fails.
   - A versioned-surface change bumps the version, adds golden vectors under `testdata/wire/vN/`, and states the change in the PR's `Compatibility:` section.
   - The `compat` check re-encodes golden vectors and fails on a byte change without a version bump.
   - Nightly from M2, a rolling-upgrade test runs the previous release and the candidate together under load on the emulated fabric. RCs run the same on lab hardware (RFC-0060 F-20).

## Consequences
- **Positive.**
  - Home clusters upgrade one machine at a time.
  - Incompatibility is a clear refusal, never a silent failure.
  - Formats can evolve additively within a major and change at major boundaries.
  - Caches that can be rebuilt are rebuilt rather than migrated.
- **Negative.**
  - Each versioned surface carries old decoders for a whole major.
  - Golden vectors and rolling-upgrade tests add CI work.
  - Pack rebuilds after a layout change cost minutes of disk I/O per model, which users notice. Release notes must warn, and layout changes should be batched.

## What would make us revisit
- Users report that skipping a minor release during upgrades is common enough that the N±1 window strands them. Then propose widening it to two minors, which needs RFC-0031 changed with it.
- Users report pack rebuild time as a top upgrade complaint. Then add in-place layout migration for the most common layout changes.
- A security fix needs a protocol change in a patch release. That requires a deliberate exception, documented in the release notes, with a forced-upgrade notice through RFC-0052's update channel.

## Revisions
- Revised 2026-10-20: decision 2 adds a maintainer's sign-off for a pre-1.0 interop break through the `interop-break-approved` label. The rolling-upgrade test already detects the break, so the cost is one label, and mixed-version alpha clusters are not stranded silently. RFC-0062 Open question 9 is answered by it.
