# ADR-0510: No telemetry pipeline, opt-in or otherwise; observability stays on the user's machines
- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the product surface and operations area)
- Related: ADR-0000 (D4), ADR-0009, ADR-0012, ADR-0015, ADR-0020; RFC-0053 §1, §7, §8; RFC-0052 §7; RFC-0061

## Context
Syntropy is local-first: people run it to keep prompts, outputs and models on their own hardware. The assignment requires zero telemetry by default.

Telemetry would still help the project:
- crash rates per platform;
- which hardware and models people use;
- which fallbacks fire;
- which Tier 2 platforms deserve promotion. ADR-0020's revisit trigger mentions "telemetry or issue volume (opt-in only)".

Any upload path is attack surface and a trust liability:
- **Even anonymous data can be identifying.** Hardware fingerprints, model choices and timing patterns can identify a user.
- **Opt-in data is skewed.** It comes mostly from enthusiasts.
- **Code changes over time.** A "disabled by default" uploader is still code that can be enabled by mistake or by a future change.

Some outbound traffic is unavoidable or explicitly requested:
- traffic between cluster members;
- model downloads the user asks for;
- update metadata, which matters for security fixes;
- relays when invite links are in use (D4).

## Options considered
| Option | For | Against |
|---|---|---|
| A. Opt-out usage telemetry | best data | contradicts the local-first promise and the assignment |
| B. Opt-in anonymous telemetry with a clear prompt | useful data with consent | an upload path exists, whose scope can creep; data skews toward enthusiasts; consent prompts in installers are friction and often clicked through |
| C. Opt-in crash reports only | high-value data | crash payloads include stack memory and paths; the same liability as B in a narrower form |
| D. No telemetry pipeline; diagnostics are local files the user can share by hand; exporters only to endpoints the user configures | nothing leaves without an explicit act; simple to audit | the project learns less, and learns it later, from issues and shared reports |

## Decision
**Option D.**

1. Syntropy contains no code that uploads usage data, crash reports, install counts, hardware inventories or model choices to the project or any third party, in any build or configuration.
2. **What a node sends by default is fixed and documented**:
   - traffic to cluster members;
   - model origins, only for models the user pulls;
   - the update repository: static metadata and artefact GETs with no identifiers, query parameters or cookies, switchable off;
   - relays, only while invite-joined members or outstanding invites exist.
   `syntropy doctor` (PRV-001) shows this list as it applies to the node.
3. **Exporters** (Prometheus, OTLP) are off by default. They export only to endpoints the user configures, and the daemon logs the destination at start.
4. **Diagnostics are local artefacts.** Crash records stay in the state directory. `debug dump` and `bench --report` write files the user can attach to an issue after reading their manifests.
5. Adding any outbound destination requires an ADR that supersedes this one.

## Consequences
- **Positive.**
  - The privacy posture is simple to state and to audit.
  - CI verifies it with a network-namespace test of an idle daemon (RFC-0053 Testing).
  - Users and distributions can trust the binary not to phone home.
- **Negative.**
  - Platform-demand signals (ADR-0020's promotion triggers) come from issue volume, shared bench reports and community feedback, not data.
  - Crash triage depends on users sharing dumps. The dump format must make that easy and safe.
  - Update checks still reveal a node's IP address and that it runs Syntropy to whoever serves the update repository. This is documented, and switchable off.
- **Work created.**
  - RFC-0053: privacy table, doctor PRV-001, the egress test.
  - RFC-0052: header-free update fetches.
  - RFC-0061: shareable report format.
  - ADR-0020: its revisit trigger should read "issue volume and shared reports" rather than telemetry.

## What would make us revisit
- Field failures that the project cannot diagnose from shared dumps keep recurring on one platform, and users ask for an easier way to send crash data. Then consider a manual "send this crash record" command that shows the exact payload and needs confirmation each time. It is never automatic.
- A distribution or enterprise user needs centralised crash collection inside their own network. That is served by OTLP to their own collector, not by a project service.
