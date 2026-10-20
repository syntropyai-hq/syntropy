# ADR-0302: LAN admission needs one confirmation with a short authentication string by default

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the distributed runtime area)
- Related: ADR-0000 (D4), ADR-0009, ADR-0300, ADR-0313, RFC-0030 §6, RFC-0034

## Context

D4 requires zero-setup discovery on a LAN. Discovery is not admission, though. An admitted node receives model weights, prompt activations and generated tokens. Home LANs carry guests' phones, smart TVs and IoT devices of unknown integrity. A surveyed runtime with no authentication on its data plane says so in its README: "Never run the RPC server on an open network" (`multi-device-inference.md` §1.4). A token-gated federation leaked its token by publishing it (§1.3).

SC-24 rules out configuration files and typed addresses, and SC-25 asks that a new LAN node join within 2 heartbeats of finishing its probe.

## Options considered

| Option | For | Against |
|---|---|---|
| A. Auto-admit every node heard in the namespace | Literally zero-touch; meets SC-25 for new nodes | Any LAN device, guests included, gets prompts and activations; the namespace is not a secret |
| B. Shared passphrase typed on each device | Familiar | A typed secret, against SC-24; weak passphrases; rotation is painful |
| **C. One confirmation on any keyholder, with a 6-digit SAS shown on both devices; plus an operator-opened pairing window for multi-device setup; plus opt-in open mode for isolated labs** | One click, no typing; the user sees which device they approve; safe default | Not literally zero-touch for brand-new devices |
| D. Out-of-band QR pairing only | Strong | Needs a camera or a second app on desktops |

## Decision

**Option C** (RFC-0030 §6.2, §6.3).
1. `cluster.lan_admission = confirm` is the default.
   - A new node sends a `PairRequest` over `syntropy/join/1`, after TLS has authenticated both node keys.
   - Both devices display `SAS = BLAKE3(context ‖ cluster_id ‖ ids ‖ nonces) mod 10^6`.
   - The user approves on any keyholder.
   - The coordinator records `NodeAdmitted` with `AdmissionProof::Pairing`.
2. `syntropy cluster open [--for 10m] [--joins 4]` opens a pairing window. During it, requests arriving on non-overlay LAN interfaces from link-local or private addresses are approved automatically, up to the join limit.
3. `open` mode auto-admits any node in the namespace. It is an explicit cluster-scoped setting for isolated networks, and the dashboard flags it permanently.
4. A node's first cluster is disposable until it admits a second node, so "install on the PC, then the Mac" needs no merge logic.
5. Certified nodes rejoin with no interaction (RFC-0030 §6.5). That is the case SC-25 measures.

**Revised 2026-10-20:** `confirm` with a 6-digit SAS stays the default, and holds only with the first item below. RFC-0030 §6.2–§6.3 hold the detail.
1. **Commit-then-reveal.** `PairRequest` carries `commit = BLAKE3("syntropy-sas-commit-v1" ‖ N.id ‖ nonce_c)`, and the joiner reveals `nonce_c` in `PairReveal` only after it has received `nonce_k`. As first written, an attacker between the new node and a keyholder could grind the SAS in about 10⁶ hashes and get itself admitted while the user paired a real device. With the commitment, neither side chooses its nonce after seeing the other's, so decision 1's one-guess-per-pairing claim holds.
2. **Number matching on the approving side.** The approver sees the joiner's name, hardware summary and three codes, and picks the one the joining device shows. A wrong pick denies the request, so an approval cannot be given without comparing codes.
3. **The SAS on both sides.** The joiner shows its SAS in `setup`, `cluster status` and its dashboard; a headless joiner prints it on the console where `setup` runs.
4. **`setup`'s "set up more machines now?" prompt** makes the first node visible for pairing and pairs the next machine in `confirm` mode with number matching. It opens no pairing window, because a window approves without a SAS and on home Wi-Fi it would be open admission.
5. **Pairing windows** (decision 2) are an explicit command only: no beacon announces them, they accept wired and Thunderbolt interfaces only unless the operator includes Wi-Fi, each join is shown with its SAS and a one-click revoke, and a closing summary lists every node admitted. RFC-0034 §2 states the residual risk.
6. **Someone else's device** is admitted ingress-only when the approver says so at pairing, the same default role as an invite for someone else's device (ADR-0303).

## Consequences

- Positive: no prompt or activation reaches an unapproved device by default. Setting up several devices takes one action. Nothing is typed.
- Negative: a brand-new node needs one approval, which RFC-0030 open question 2 reconciles with SC-25's wording. The first-run experience needs a clear "join *name*?" screen (RFC-0051).
- Work created:
  - RFC-0051: pending-join notifications and the SAS display;
  - RFC-0034: certificate issuance;
  - RFC-0060: SAS binding tests.

## What would make us revisit

- Usability testing shows users approving without comparing codes in most cases, even with number matching. Consider a QR or proximity confirmation instead. The M2 pairing usability run with 5 alpha testers measures this (DIST-E5-T17).
- SC-25 is required to hold for brand-new nodes without interaction. Make pairing windows open automatically for the first N minutes after a cluster is created, and document the trade.
