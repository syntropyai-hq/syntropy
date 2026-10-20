# ADR-0204: Pack integrity is checked at commit, after crashes and by an idle scrubber, not on every read

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the tiers area (RFC-0021))
- Related: ADR-0005, ADR-0016, ADR-0019; RFC-0001 §10.1 `Integrity`, §12 rule 3; RFC-0021 §14; RFC-0041

## Context

- Every chunk fetched from the Hub or another node is BLAKE3-verified on receipt (ADR-0016, RFC-0001 §12 rule 3). The pack built from those chunks records a BLAKE3 hash per extent in its sidecar (ADR-0005).
- After that, the bytes live on the user's NVMe. Corruption is possible (a torn write after a crash, a failing drive, a bug in the pack builder), and corrupt weights produce wrong output silently, which D3 forbids.
- Hashing is not free on the hot path. In scenario b2 the CPU's DRAM bandwidth computes in-place misses and is the bottleneck (`hardware-and-performance-model.md` §4 b2). At ≈210 expert reads per second, verifying every read means hashing ≈5.5 GB/s [estimate: b2's NVMe read rate at h = 0.9]. BLAKE3's single-core rate with SIMD is a few GB/s, so that is on the order of 1–2 cores [estimate; no measurement on the target CPUs], and, worse, it re-reads ≈5.5 GB/s from DRAM, competing directly with the miss kernels for the bottleneck resource.
- NVMe drives carry end-to-end protection in their controllers, and filesystems report media errors as `EIO`, so silent corruption between the drive and memory is rare compared with corruption written at build time or left by a crash.

## Options considered

| Option | For | Against |
|---|---|---|
| A. Hash every read | Detects any corruption before use | Milliseconds of CPU per expert miss and ≈5.5 GB/s of extra DRAM reads in b2 [estimate]; on the order of 1–2 cores at b2's read rate, plus cores at b2's read rate |
| B. Trust the pack once built | Free | A torn or corrupt pack serves wrong weights until rebuilt |
| C. Hash a random sample of hot-path reads | Bounded cost | Probabilistic; still on the hot path; a bad extent can be used many times before it is sampled |
| **D. Verify at pack commit; verify fully after an unclean shutdown; verify a sample at load (the whole dense hot set plus 1% of the rest); scrub everything in the background at idle; act on `EIO` and short reads** | Catches build-time and crash corruption before use; finds latent corruption at idle; zero hot-path cost | A latent corruption that appears after commit can be used until the scrubber or a read error finds it |

## Decision

**Option D.** As specified in RFC-0021 §14:

1. At commit, every extent is hashed from the bytes written and checked against the source tensor hash before the sidecar is renamed into place.
2. Packs that were open for writing at an unclean shutdown are marked unverified and checked in full before use.
3. At load, every extent of the dense hot set and a random 1% of the other extents are verified on the compute pool's cores before `Warming`, skipped for packs verified at commit in the same boot; target ≤ 2% of load time.
4. An idle scrubber verifies every extent at 64 MB/s, on mains power only, below 10% of the volume's sustained read rate, pausing at once for demand traffic.
5. A mismatch, or a second read failure at the same offset, marks the extent bad; it is rebuilt from verified source chunks and never served again.

## Consequences

Positive:
- No hashing on the token path.
- Build-time and crash corruption, the likely cases, are caught before any byte is used.
- Latent corruption is found in the background within one scrub cycle: a 400 GB pack at 64 MB/s is scrubbed in ≈1.7 hours of idle time [estimate].

Negative:
- A bit flip on media after commit that the drive does not report can be served until the scrubber reaches it. Exact-mode output could be wrong in that window.
- The scrubber uses disk bandwidth and some power at idle; it is off on battery.

## What would make us revisit

- Field reports or the RFC-0060 fault-injection suite show undetected corruption reaching output on a Tier 1 platform: hash demand reads of `Nvme`-homed experts on a spare core, or shorten the scrub interval.
- Packs move to storage without end-to-end protection (network filesystems, USB enclosures). Then verify every read from such volumes, flagged by the probe.
- BLAKE3 throughput on the bottleneck CPU turns out high enough that hashing every miss costs under 1% of token time on the reference rigs.
