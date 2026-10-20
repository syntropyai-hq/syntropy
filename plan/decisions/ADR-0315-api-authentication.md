# ADR-0315: API authentication (superseded by ADR-0505)

- Status: Superseded by ADR-0505
- Date: 2026-10-19
- Deciders: maintainers (drafted by the distributed runtime area)
- Related: RFC-0001 §12 (zone Z3), RFC-0034 §11, RFC-0050 §11, ADR-0505

## Context

This ADR first defined API keys, TLS and loopback protections from the security RFC. Client-facing API authentication and bind policy then moved to the product area, RFC-0050 §11 and ADR-0505. RFC-0034 keeps the threat model and the right to tighten ADR-0505's defaults (ADR-0505, "Deciders").

## Options considered

| Option | For | Against |
|---|---|---|
| A. Keep two API-auth ADRs | — | Two sources of truth for one mechanism |
| **B. Supersede this ADR with ADR-0505, and record the threat model's requirements in RFC-0034 §11** | One mechanism, one owner; requirements stay traceable | — |

## Decision

**Option B.** ADR-0505 is the API mechanism: key format, scopes, headers, rate limits, TLS options, dashboard login. RFC-0034 §11 lists the threat model's requirements R1–R5 against it.

Four of the five are met. One is not:

> **R3.** No API key, prompt or output crosses a network link in plaintext. A non-loopback listener uses TLS, unless it is bound only to an interface that is itself encrypted end to end, such as a WireGuard-based overlay.

ADR-0505 item 2 allows plain HTTP with a warning on private addresses. RFC-0001 §12 requires TLS on every non-localhost bind. RFC-0034 asks that ADR-0505's own revisit clause be applied: require TLS on private addresses too, and invest in easy self-signed trust. It also proposes the overlay exception as an RFC-0001 §12 amendment. Resolution belongs to the product area.

## Consequences

- There is a single owner for the API mechanism.
- The R3 conflict is visible and tracked, in RFC-0034 §11 and its proposed amendments, until ADR-0505 or RFC-0001 §12 changes.

## What would make us revisit

Nothing on the API side. Changes go through ADR-0505.
