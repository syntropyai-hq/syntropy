# ADR-0314: Plaintext data paths are limited to Thunderbolt 5 cables, operator-marked fabrics and same-user shared memory

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the distributed runtime area)
- Related: ADR-0011 (item 4), ADR-0309, RFC-0001 §12 (rule 2), RFC-0031 §5, RFC-0034 §9

## Context

RFC-0001 §12 rule 2 limits plaintext data planes to three cases:
- shared memory on one host;
- RDMA over a direct Thunderbolt 5 cable between two admitted nodes;
- RoCEv2 on interfaces the operator marks `trusted_fabric`.

Every other data path is encrypted. ADR-0011 makes RoCEv2 usable only on `trusted_fabric` interfaces. This ADR fixes the policy's details and decides whether to offer encryption for RDMA at all.

The facts:
- RDMA NICs and Thunderbolt RDMA offer no transport encryption.
- Software AEAD costs about 8 µs per 32 KiB frame per side at an assumed 4 GB/s. That is +16 µs per hop on links whose entire value is 10–25 µs hops (cost model §2.5, RFC-0031 §Performance).
- A Thunderbolt 5 cable is point-to-point and unrouted.
- RoCEv2 crosses Ethernet switches that may carry other devices.
- Shared-memory segments are protected by OS permissions.

The surveyed systems shipped plaintext data planes by default (`multi-device-inference.md` §13 item 12). The research recommends never doing so (§15.1).

## Options considered

| Option | For | Against |
|---|---|---|
| A. Encrypt everything, RDMA included | Uniform | Removes most of RDMA's latency benefit; TP over Thunderbolt might no longer pass the gate (ADR-0006) |
| B. Plaintext RDMA everywhere it exists | Fastest | RoCE over a shared switch exposes activations to any device on it |
| **C. TB5 cable: plaintext allowed (QPs paired only over the authenticated control connection; credentialed first frame; channel token; CRC). RoCE/IB: plaintext only when both ends' interfaces are in `security.trusted_fabric`; otherwise not used, or, opt-in, used with per-frame AEAD keyed from the TLS exporter. Shared memory: same host and same OS user only; otherwise QUIC over loopback. Every plaintext edge is labelled in the dashboard** | Keeps RDMA's latency where the medium confines data to admitted machines; never plaintext on a shared medium without an explicit operator assertion | The operator's `trusted_fabric` assertion can be wrong; the AEAD option adds an extra code path |

## Decision

**Option C** (RFC-0034 §9).
1. Thunderbolt 5 RDMA between admitted nodes is plaintext by default. An attacker needs physical access to the cable, and pairing is authenticated.
2. RoCEv2 and InfiniBand are used only on interfaces both ends mark `trusted_fabric`. The CLI explains the assertion when it is set.
3. `rdma.encrypt_untrusted = true` (M4) enables RDMA on unmarked interfaces with ChaCha20-Poly1305 or AES-256-GCM per frame:
   - per-direction keys exported from the pairing QUIC connection's TLS exporter;
   - `chan_seq` as the nonce;
   - the base header as AAD.

   The probe measures the cost into `α_eff`, so the solver compares it fairly with QUIC.
4. Shared-memory segments are created owner-only (mode 0600, or an owner-only DACL). An edge between nodes run by different OS users falls back to QUIC over loopback.
5. The dashboard's topology view and `syntropy doctor` label every plaintext edge with its reason.

**Revised 2026-10-19:**
- The AEAD tag is a 16-byte trailer after the last payload row, counted in `payload_len` and outside the row stride.
- The associated data is base-header bytes 0..56, which are fixed for the frame's lifetime.
- The nonce is `chan_seq`.
- A resend retransmits the stored sealed bytes unchanged, so no key and nonce pair is ever used with different associated data.

**Revised 2026-10-19, second revision:**
- **Two nonce spaces.** The 96-bit nonce is `space u8, three zero bytes, counter u64`. Data frames use space 0 with `chan_seq`. Control frames, whose `chan_seq` is 0, use space 1 with a per-direction `ctl_seq` carried in the header's `step` field. No `(key, nonce)` pair repeats.
- **Every frame is sealed** on an encrypted channel, control frames included, so credits, `RESET` and `Close` cannot be forged.
- **Associated data** is header bytes 0..56 plus the extension's plaintext first 16 bytes (RFC-0031 §6.1, RFC-0034 §9.3 agree).
- **Re-keying** flips flags bit4 `KEY_PHASE`. Receivers keep the previous key until every frame sealed under it is acknowledged, so resent sealed bytes still open.

**Revised 2026-10-20:**
1. **`trusted_fabric` is gated mechanically** (OQ-SEC-04), which brings forward the third revisit condition below. Decision 2's assertion is accepted only for an interface that is wired, has no default route, and whose ARP or NDP table holds only cluster members. Otherwise the CLI refuses unless the operator passes `--force` and types the interface name back. `doctor` flags a forced assertion, and every link probe re-checks the conditions (RFC-0034 §9.2).
2. **No flip to encrypted RoCE is expected** (OQ-NET-30). At the ≥ 4 GB/s per core RFC-0034 assumes for AES-GCM, a 28–32 KiB frame takes about 7–8 µs to seal and as long to open, about 15 µs per hop, three times the 5 µs threshold of the second revisit condition [estimate: 32 KiB / 4 GB/s per side]. Plaintext RoCE stays on `trusted_fabric` only. The M4 benchmark (DIST-E7-T14) confirms at p50 and p99.

## Consequences

- Positive:
  - RDMA keeps its latency advantage on the links where it matters, which are Mac pairs on Thunderbolt 5.
  - No shared medium carries plaintext activations without an explicit operator choice.
  - Users can see exactly which edges are plaintext.
- Negative:
  - Correctness depends on the operator's assertion for RoCE.
  - The AEAD option is one more path to test.
- Work created:
  - RFC-0030: the `TRUSTED_FABRIC` interface flag;
  - RFC-0031: the `ENCRYPTED` frame flag and AEAD sealing;
  - RFC-0051: plaintext-edge labels.

## What would make us revisit

- RDMA NICs or Thunderbolt RDMA gain hardware encryption with measured latency within 2 µs of plaintext. Make encryption the default.
- Measured software AEAD on RDMA costs less than 5 µs per hop on Tier 1 hardware. Then the default could flip to encrypted on RoCE.
- Field reports show `trusted_fabric` commonly set on shared networks. Tighten the CLI check, for example by refusing when discovery sees non-member hosts on that subnet.
