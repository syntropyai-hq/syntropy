# RFC-0034: Security and trust

- Status: Draft
- Area: distributed runtime
- Depends on: RFC-0001 (§2.3, §3.2, §3.5, §5.1, §12), RFC-0020, RFC-0022, RFC-0030, RFC-0031, RFC-0032, RFC-0041, RFC-0050, RFC-0051, RFC-0052, RFC-0053, RFC-0054, RFC-0062, ADR-0000 (D4), ADR-0009, ADR-0011, ADR-0015, ADR-0016, ADR-0302, ADR-0303, ADR-0313, ADR-0314, ADR-0315 (superseded by ADR-0505), ADR-0316, ADR-0505, ADR-0507, ADR-0508, ADR-0623
- Date: 2026-10-19

## Summary

This RFC is the threat model and the security mechanisms for a Syntropy cluster: devices on a LAN plus devices joined by invite (D4), with no untrusted swarms.

- **Identity.** Every node is an Ed25519 key.
- **The cluster key** is an Ed25519 signing key, held by one or more **keyholder** nodes. It signs 192-byte **admission certificates** that bind a node ID and an admission generation to roles: member, stage host, ingress, keyholder, coordinator-eligible.
- **Authority.** Every node validates every log entry. Security events (admissions, revocations, keyholder changes, key rotations, invites, security-class configuration, cluster-wide API keys) must carry a keyholder signature. Only `COORDINATOR`-role nodes may coordinate. A coordinator therefore cannot forge membership or keys (§8.3).
- **Mutual authentication before any traffic.**
  - Every connection is TLS 1.3 with both endpoint keys authenticated.
  - Every control, bulk and sequence connection then exchanges certificates both ways before anything else.
  - Runners never hold the node key. They use per-runner keys certified by a node-signed, epoch-scoped data-plane credential, also exchanged both ways.
- **Invite tokens** carry a 128-bit secret. Only its hash is logged, and the secret is disclosed only to a rendezvous node that has proven its cluster membership. Tokens are single-use by default, expire within 24 h by default, and carry scopes.
- **Revocation** is by (node, generation), signed by a keyholder, recorded in the log and in each node's local deny list. Revoking a keyholder makes the revoking keyholder rotate the cluster key and re-issue every member's certificate. Rotation statements form a verifiable chain, so a rotation on one side of a split still heals.
- **Cluster-wide API keys** are stored in the log as hashes, scoped to instances or models, with keyholder-signed creation and monotonic revocation (§8.4).
- **Encryption** covers every IP path. The plaintext exceptions are exactly RFC-0001 §12's:
  - shared memory between same-user processes on one host;
  - RDMA over a direct Thunderbolt 5 cable between admitted nodes;
  - RoCEv2 only on operator-marked `trusted_fabric` interfaces, with an opt-in per-frame AEAD for anything else.
- **Model integrity.** Every model byte is BLAKE3-verified against content addresses before use. Packs carry per-extent hashes. Publisher signatures in the OpenSSF Model Signing format are checked when present. Parsers are fuzzed, and no model ever runs code.
- **API.** The product area owns API authentication (RFC-0050 §11, ADR-0505). This RFC states the threat model's requirements of it. One of them, TLS on every non-loopback bind except encrypted overlay interfaces, is met: plaintext on a LAN address needs the explicit `api.allow_plaintext_lan` opt-in, which stays flagged (§11).
- **Losing the only keyholder** is recoverable: an optional offline recovery code rotates the cluster key onto a surviving member, a guard stops the last keyholder from leaving by accident, and a cluster with no code can be re-founded without re-downloading models (§5.5).
- **Supply chain.** This RFC defines the signing key hierarchy: TUF root (2 of 3) and targets (1 of 2) on offline hardware tokens held by maintainers or recorded delegates, online snapshot and timestamp, an online release key for detached signatures over a digest statement, all online keys in an HSM-backed cloud KMS, and platform identities. The update client (ADR-0508) and the release pipeline (RFC-0062) use it.
- **Privacy.** Nothing leaves the machine by default except the network calls the user's own actions cause. Those are listed by `syntropy doctor`. A clustered node beacons only on networks where its cluster lives, and its beacons are blinded. Session data saved on other members is ciphertext under a key only the session's ingress holds.

## Motivation and goals / non-goals

Surveyed data planes ran with no authentication or encryption. One surveyed remote-backend RPC layer warns "Never run the RPC server on an open network". One HMAC handshake authenticated but still sent plaintext. A token-gated federation leaked its token by publishing it (`multi-device-inference.md` §1.3, §1.4, §13 item 12).

A home LAN is not a trusted network. It carries guests' phones, smart TVs and IoT devices of unknown integrity. An admitted node sees the prompts and activations it processes. So admission is the security boundary, and everything after it must be authenticated and encrypted by default.

Goals:
1. No model traffic, prompt, activation or token reaches a device that the user has not admitted (RFC-0001 §12).
2. Every IP path is encrypted. Plaintext is allowed only where the medium itself confines the data to admitted machines, and the dashboard shows each plaintext edge.
3. Losing or leaking a credential has a bounded, revocable impact.
4. Model bytes are verified before use. A hostile model file cannot execute code.
5. The binary and its downloaded modules are verifiable to their publisher.
6. No telemetry by default. The local privacy posture is explicit and testable.

Non-goals:
- Byzantine members. An admitted node is trusted with the weights and activations it handles (D4). We bound what a malicious member can do and detect corrupt bytes, but we do not prevent a member from computing wrong results.
- Public swarms (D4).
- Side channels: timing, power, GPU memory remanence.
- Physical attacks, and a compromised OS on a member.
- Confidentiality of model weights against cluster members.

## Design

### 1. Trust zones and scope

RFC-0001 §12 fixes the zones. This RFC supplies their mechanisms.

| Zone | Mechanism in this RFC |
|---|---|
| Z0 Runner | per-runner keys and data-plane credentials (§6.3); IPC peer checks (§6.4) |
| Z1 Node | node key storage (§4); state directory permissions (RFC-0030 §2.1) |
| Z2 Cluster | cluster key, certificates (§5), mutual authentication (§6), revocation and rotation (§8), encryption policy (§9) |
| Z3 API clients | mechanism in RFC-0050 §11 and ADR-0505; threat-model requirements in §11 |
| Z4 Internet | content addressing and signatures (§10, §12); relays see ciphertext only (§9) |

### 2. Threat model

**Assets.**

| Asset | Where it lives | Harm if exposed or altered |
|---|---|---|
| Prompts, outputs, session KV | ingress, stages, session holders | privacy loss |
| Activations and KV in flight | the data plane | they reveal the prompt |
| Node secret key | daemon state | impersonation of the node |
| Cluster key | keyholders | minting members |
| Invite secrets | tokens in transit | one admission |
| Hub tokens for gated models | the node that owns them | access to the user's Hub account |
| Model weights | stores, packs, tiers | licence exposure; tampering corrupts output |
| Host integrity | every node | code execution through a parser bug or a malicious module |

**Adversaries and what they can do.**

| ID | Adversary | Capability | Main defences |
|---|---|---|---|
| A1 | Another device on the LAN (guest, IoT, compromised machine) | sees broadcast and multicast; can send arbitrary packets; can sniff unencrypted Wi-Fi/LAN traffic on shared media | admission by confirmed pairing (§5, ADR-0302); TLS everywhere; signed beacons; rate-limited join ALPN |
| A2 | Internet attacker | reaches relays and any publicly exposed port | no listening service grants anything without a certificate; the join ALPN needs a valid token or a user approval |
| A3 | Relay operator | sees connection metadata (which node IDs talk, sizes, timing) | end-to-end QUIC; tokens authenticate the cluster through the relay (§7); relay mode configurable or disabled |
| A4 | Malicious model file or repository | crafted GGUF or safetensors headers, chat templates | bounds-checked, fuzzed parsers in memory-safe code; no remote code; templates sandboxed (§10) |
| A5 | Compromised or malicious member | holds a valid certificate | revocation (§8.1), spread by gossip so a coordinator cannot hide it; chunk tables committed only with origin or independent evidence (§10.1), so one member cannot poison other nodes' weights; signed elections and validated snapshots (RFC-0032 §6.7, §7); roles limit ingress-only nodes; it still sees the activations it processes (accepted, D4) |
| A6 | Supply-chain attacker | tampers with releases, modules, updates or dependencies | signed releases under an offline root; module signature checks before load; dependency auditing (§12) |
| A7 | Another local OS user on a multi-user host | reaches loopback ports; cannot read 0700 directories | IPC peer-credential checks; optional loopback API keys; key files 0600 (§4, §11) |
| A8 | Holder of a leaked invite token | can join until the token is used, expires or is revoked | single use, short expiry, scopes, admission notifications, one-click revoke (§7) |

**Trust assumptions.** The OS and the user account running Syntropy are not compromised. The OS CSPRNG is sound. Wall clocks are within 10 minutes of each other: certificates use wall-clock validity, and nothing else does.

Trust among members is tiered by certificate role (§5.2) and enforced by every node, not only the coordinator (§8.3):

| Tier | Trusted for | Not trusted for |
|---|---|---|
| Keyholders | everything, including the cluster key | — |
| Coordinator-eligible members (`COORDINATOR` role: keyholders by default, others by explicit grant) | availability decisions as coordinator: ordering the log, placement, confirming failures | any security decision: admission, revocation, keyholder changes, key rotation, security-class configuration, cluster-wide API keys, chunk-table commitment. These need signatures that every follower verifies, in log entries and in snapshots alike (RFC-0032 §6.7). A coordinator also cannot suppress a revocation, because revocations spread by gossip (§8.1) |
| Other members, the ingress-only and invite-joined defaults | the activations and weights they handle (D4); members with `STAGE_HOST` may also change cluster-scope configuration that needs no keyholder (§8.3) | claiming the coordinator role; any authority event; for members without `STAGE_HOST`, any cluster-scope configuration change |

A compromised member (A5) or the holder of a used, leaked token (A8) can therefore disrupt availability within its role, but cannot change membership, keys or security settings, and cannot become coordinator unless its certificate carries `COORDINATOR`. Model aliases, the network scope, approximate-mode permission, update policy and the exactness and pin fields of placement policy also need a keyholder (§8.3), because other users rely on them: an alias resolves before canonical names, so a member that could set one could silently redirect a model name for every client.

**Pairing windows** (residual risk). A pairing window (RFC-0030 §6.3) admits without a SAS comparison. While it is open, any device on a wired or Thunderbolt interface of the LAN that sends a `PairRequest` first is admitted as a stage host. The limits that bound this: windows are an explicit keyholder command, never opened by `setup`; they are not announced in beacons; they accept wired and Thunderbolt interfaces only unless the operator includes Wi-Fi; they close after their time or join limit; each join is shown with its SAS and a one-click revoke; and the closing summary lists every node admitted. A guest's device on Wi-Fi cannot join a default window. One on the wired LAN can, during the window, and the summary shows it.

### 3. Cryptographic primitives

| Purpose | Primitive | Library |
|---|---|---|
| Node, runner, cluster and release keys | Ed25519 (RFC 8032), strict verification (canonical encodings, no small-order keys) | a maintained Rust Ed25519 implementation |
| Transport security | TLS 1.3 inside QUIC (iroh) and over TCP (rustls); suites AES-128-GCM, AES-256-GCM, ChaCha20-Poly1305, with ChaCha20 preferred on CPUs without hardware AES | rustls with one crypto provider, pinned under RFC-0062's dependency policy |
| Content addressing, derivations, verifiers | BLAKE3 (hash, keyed hash, derive-key mode) | the official BLAKE3 crate |
| Integrity against faults (not attacks) | CRC32C (RFC-0031) | — |
| RDMA frame encryption (opt-in, §9.3) | ChaCha20-Poly1305 or AES-256-GCM with keys from the TLS exporter (RFC 8446 §7.5) | same provider |
| Saved-session encryption (§13) | X25519 key wrapping of a per-blob content key, then ChaCha20-Poly1305 or AES-256-GCM | same provider |
| Randomness | OS CSPRNG (`getrandom`) | — |
| Secret comparison | constant-time equality | — |

**Domain separation.** Every signature and derivation uses a fixed context string, so a signature or hash valid in one context is never valid in another:
- `syntropy-beacon-v1`, `syntropy-beacon-blind-v1`, `syntropy-beacon-mac-v1`, `syntropy-netkey-v1` (RFC-0030 §4.2, §4.8), `syntropy-cert-v1`, `syntropy-dpcred-v1`;
- `syntropy-sas-v1`, `syntropy-sas-commit-v1`, `syntropy-invite-id-v1`, `syntropy-invite-verifier-v1`, `syntropy-recovery-v1` (§5.5);
- `syntropy-ns-v1`, `syntropy-term-tag`;
- `syntropy-keyrotate-v1`, `syntropy-release-v1`, `syntropy-module-v1`, `syntropy-rdma-aead-v1`, `syntropy-fp-v1`;
- `syntropy-session-wrap-v1` (§13);
- `syntropy-admit-v1`, `syntropy-revoke-v1`, `syntropy-authority-v1` (the other authority events of §8.3), `syntropy-apikey-v1`.

*Note 2026-10-22 (DIST-E2-T04, from review):* strict verification also refuses a public key that is not of prime order, where `[ℓ]A ≠ identity`. Refusing small-order keys alone is not enough. A key `A + T`, with `T` a point of small order, verifies a signature made with `A`'s secret whenever `T`'s order divides the challenge `k`, so one secret could sign under up to eight public keys, and a `NodeId` would not name one key. Every honest key is `[a]B` and has prime order, so the check costs honest peers nothing. `net::crypto::verify` applies it after the canonical-encoding and small-order checks.

### 4. Keys and their storage

| Key | Generated | Held by | Storage | Lifetime |
|---|---|---|---|---|
| Node key (Ed25519) | first daemon start (RFC-0030 §2.2) | the daemon only | `identity/node.key`, 32-byte seed, mode 0600, in the 0700 state directory; wrapped by the OS keystore when `security.keystore = auto` finds one (below) | until `identity reset` |
| Cluster key (Ed25519; public half = `ClusterId`) | cluster bootstrap (RFC-0030 §6.1), or rotation (§8.2) | keyholders | `identity/cluster.key`, same protection | until rotated |
| Runner key (Ed25519) | by the daemon at each runner spawn | that runner; passed in `Init` over the inherited IPC channel (RFC-0001 §3.5) | runner memory only | the runner's life |
| API keys | `syntropy api-key create` | clients; the node stores only BLAKE3 hashes | node config, `api-keys.toml`, mode 0600 | until revoked |
| Hub tokens | user input | the node that owns them | OS keystore, or a 0600 file; never logged, never sent to another node (ADR-0016) | user-managed |
| Release root keys (public) | offline, in a key ceremony by the token custodians, each a maintainer or a recorded delegate (**Maintainer action**, §12.1) | embedded in the binary | — | rolled with releases (§12) |

**OS keystore wrapping.** With `security.keystore = auto`, seeds are stored or wrapped where the platform offers it:
- **macOS:** a Keychain generic-password item, `kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly`, readable by the signed Syntropy binary.
- **Windows:** DPAPI user-scope protection (`CryptProtectData`) of the key file.
- **Linux:** the Secret Service over D-Bus, when a session keyring is available.
- **Everywhere else:** the 0600 file. System services running as a dedicated user also use the file.

Keystore wrapping protects against offline disk theft on unencrypted disks. It does not protect against code running as the same user, which is out of scope (§2).

### 5. Admission certificates

#### 5.1 Format (192 bytes, little-endian)

```text
off  len  field
0    1    version       1
1    1    flags         reserved
2    2    roles         bit0 MEMBER, bit1 STAGE_HOST, bit2 INGRESS, bit3 KEYHOLDER, bit4 COORDINATOR
4    4    generation    u32: the node's admission generation (§8.1); revocation is by (node_id, generation)
8    32   cluster_id    the cluster public key that signed this certificate
40   32   node_id
72   8    serial        u64, random; identifies this certificate only. Revocation never uses serials (§8.1)
80   8    not_before    Unix seconds
88   8    not_after     Unix seconds
96   32   issuer        node ID of the keyholder that signed (accountability; not used for verification)
128  64   signature     Ed25519 by the cluster key over "syntropy-cert-v1" ‖ bytes 0..128
```

A fixed binary layout is used rather than X.509: it is easy to parse safely (A4), and fuzzed (§Testing).

#### 5.2 Roles

| Role | Grants | Default for |
|---|---|---|
| `MEMBER` | control-plane participation; may be routed tokens for its own requests | everyone |
| `STAGE_HOST` | may receive weight chunks and run runners (data-plane stage channels) | pairing joins; invites without `--ingress-only` |
| `INGRESS` | may open sequence channels to heads and receive token streams | everyone |
| `KEYHOLDER` | holds the cluster key; signs authority events (§8.3): admissions, revocations, keyholder changes, key rotations, invites, security-class configuration, cluster-wide API keys | the bootstrap node; explicit promotion approved by an existing keyholder, offered as in "Keyholder offers" below |
| `COORDINATOR` | may claim the coordinator role (RFC-0032 §7.3); voters and followers check it | keyholders, including the bootstrap node. Any other node only by an explicit operator grant: the checkbox at pairing approval, unticked by default, `syntropy cluster grant-coordinator <node>` on a keyholder (a re-issued certificate), or an invite with scope bit4 (ADR-0303). **Never** ingress-only nodes |

**Availability consequence of the `COORDINATOR` default**. The default cluster of two, a bootstrap keyholder and one paired node, has exactly one coordinator-eligible node, the keyholder, and it still elects.
- While that keyholder is away, nobody can coordinate. Instances whose stages all sit on the surviving node keep serving, because the data plane does not depend on the log.
- Nothing that needs a log append can happen: no plan commit, re-plan, repair, model load, admission or configuration change. That lasts until the keyholder returns.
- Granting `COORDINATOR` needs a keyholder, so the grant must be made while one is present.

**Grants and offers at pairing approval.** The default stays keyholders only, because `COORDINATOR` decides placement and so which nodes see which prompts, and the pairing flow cannot tell a one-person home from a shared one (S5). It is the stricter default, and the reversible one: loosening it later breaks no deployed cluster, and certificates minted under it need only a re-grant.
- **Who is offered.** When the approved node is a wired, mains-powered desktop, the approval screen shows two unticked options: the `COORDINATOR` grant and the keyholder offer. "Wired, mains-powered desktop" means the node reports mains power and no battery in `NodeInfo::power`, and its `PairRequest` arrived on an interface classified `Ethernet` or `ThunderboltIp` (RFC-0030 §7.2).
- **The cost of declining**, one line under each option, naming the machines: "Without this, *Studio-PC* cannot re-plan, repair or load models while *Laptop* is off" for the grant, and "Without this, no device can be admitted, removed or renewed while *Laptop* is off" for the keyholder offer.
- **Keyholder offers.** Each keyholder is another copy of the cluster key, so the offer must not nag.
  - It is made at the pairing approval above (M2), and once more when the cluster has gone 10 minutes [target] without its only keyholder while an eligible node is present (M3). There are no further prompts.
  - A permanent status line in the dashboard and `syntropy cluster status`, which never interrupts, says "one keyholder: admissions and re-planning stop while *Laptop* is off" while the cluster has one keyholder.
  - The role is never offered to laptops or battery-powered nodes, so the key stays off machines that leave the house. An operator can still promote any non-ingress-only node explicitly with `PromoteKeyholder` (RFC-0030 §8.1).
- **No survivor mode for now.** DIST's alternative, letting a non-eligible node self-elect for plans that touch only itself, is not built. Reconsider it at M3 if the alpha data shows re-plan stalls hurting users.
- **Revisit at the M2 exit** with the alpha testers' reports. If most two-node clusters are a keyholder laptop plus an always-on desktop, pre-tick the grant for wired, mains-powered nodes in clusters with a single operator (DIST-E5-T16).
- **Recovery ships with this default** (§5.5). Keyholders-only coordination plus an offer that is easy to decline leaves most clusters with one keyholder, so losing that node must not be terminal.

Enforcement:
- A node without `STAGE_HOST` gets no plan assignment: the solver filters it out (RFC-0011).
- Peers refuse to open `Activations`, `Collective` and `Replay` channels with it.
- Stores refuse it weight chunks, except chunks of models it has itself registered for local use.
- Every node rejects authority events that lack a valid keyholder signature (§8.3), whoever appended them.
- Voters and followers ignore coordinator claims, beats and appends from a node without `COORDINATOR` (RFC-0032 §7.3–§7.4).

#### 5.3 Validity and renewal

- `not_after = issue time + security.cert_lifetime` (default 180 days).
- When less than half the lifetime remains, a node requests renewal from any reachable keyholder with the `RenewCert` RPC. Renewal re-issues the certificate with a new serial and the same roles and generation, then records `CertIssued` carrying the certificate itself.
- **Renewal after expiry** (open question 2, answered). The lifetime stays 180 days. A certificate that expired less than 180 days ago may be renewed: the node proves possession of its node key over the authenticated connection, and one keyholder operator approves after seeing the node's name and six-word fingerprint (§6.5). Renewal is never silent, and a revoked generation is never renewed (§5.4 step 5). A certificate that expired 180 days or more ago is not renewed, and the node must be admitted again by pairing or invite.
- Verification tolerates 10 minutes of clock skew either way. Larger skew is reported (`ClockSkew`, RFC-0030 §10).

#### 5.4 Verification

A certificate `c` presented by peer key `k` is valid when every one of these holds:
1. `c.node_id == k`, the peer key that TLS authenticated.
2. The signature verifies under `c.cluster_id`.
3. `c.cluster_id` is the current cluster key, or is linked to it by a chain of verified rotation statements (§8.2) inside the rotation window, **and** in that case `c.node_id` is already a member.
4. `not_before − skew ≤ now ≤ not_after + skew`.
5. `c.generation` exceeds the highest revoked generation recorded for `c.node_id`, in the log and in the local deny list (§8.1).
6. The roles cover the requested use (§5.2).

`AdmissionProof::Certificate { cert }` is added to RFC-0030's `AdmissionProof`. It re-admits a node that holds a valid certificate but is unknown to the current log. The cases are a split heal (RFC-0032 §7.5) and a log restored from an older snapshot. `decide()` accepts it when the checks above pass.

#### 5.5 Losing the only keyholder: recovery code, last-keyholder guard and re-founding

Without a keyholder nobody can revoke a stolen device, admit a node or renew a certificate, and every member drops out within one certificate lifetime. Under the keyholders-only `COORDINATOR` default (§5.2) nothing can re-plan either. Three mechanisms cover it. There is no cloud escrow. All three ship at M2 with the cluster key.

**Recovery code.**
- **Offered once.** At bootstrap of a cluster created explicitly (`syntropy setup`, `syntropy cluster create`), or otherwise when the cluster admits its second node and stops being disposable (RFC-0030 §6.1), the keyholder offers an optional offline recovery code on one screen. The user may decline. A keyholder can generate a new code later; the new one replaces the old.
- **What it is.** A fresh 128-bit recovery seed, shown once as 12 words from the fingerprint word list (§6.5) with a 4-bit checksum, so a mistyped word is caught before any network traffic. The seed derives an Ed25519 recovery key: `BLAKE3 derive-key("syntropy-recovery-v1", seed)`. The seed is never stored on any machine.
- **What the cluster stores.** Only the recovery *public* key, recorded by a keyholder-signed `RecoveryKeySet { recovery_pk }` authority event (§8.3) and kept in `ClusterState` with its signed record, so snapshots carry its proof. Every member therefore knows which key may recover the cluster, and no member holds anything that unlocks it.
- **Recovering.** `syntropy cluster recover` on any surviving member asks for the 12 words, derives the recovery key and checks it against `recovery_pk`. It then shows which keyholders will be revoked and, after confirmation:
  1. generates a new cluster key and builds a recovery `RotationStatement`: as in §8.2, but `sig_old` is replaced by `sig_recovery`, the recovery key's signature over `"syntropy-keyrotate-v1" ‖ new ‖ parent_term`;
  2. records it, and every node validates it (§8.3) against the `recovery_pk` in its state;
  3. self-issues a `KEYHOLDER | COORDINATOR` certificate under the new key, revokes every previous keyholder's admission generation (§8.1), and re-issues every other member's certificate under the new key (§8.2 step 5);
  4. clears `recovery_pk`, because a used code is spent, and offers a new code at once.
- **Recovery always rotates the cluster key.** A recovery that left the old key in place would leave it on the lost or stolen device.
- **Residual risk.** Whoever holds the 12 words can take the cluster's authority. The screen that shows the code says so and asks the user to store it offline. A code found by someone else is defeated by generating a new one, which replaces `recovery_pk`.

**Last-keyholder guard.** `syntropy uninstall`, `syntropy cluster leave` and `syntropy cluster remove` refuse to remove the last non-revoked keyholder of a cluster that has other members. The user must first promote another node, or pass `--force` after a warning that names the consequences: no admissions, no revocations, no renewals, and with the default `COORDINATOR` rule no re-planning, until a recovery code is used or the cluster is re-founded. `syntropy identity reset` on the last keyholder gets the same guard, because it deletes the cluster key. RFC-0052 owns the uninstall path, and RFC-0051 the wording.

**Re-founding**, for a cluster that lost its only keyholder and has no recovery code. `syntropy cluster refound` on a surviving member leaves the old cluster locally, which needs no coordinator, and bootstraps a new cluster with a new key and `ClusterId`. The other surviving members join it by ordinary `confirm` pairing, one approval each (RFC-0030 §6.2), and keep their model stores and packs (RFC-0041), so nothing is downloaded again. The new cluster never accepts an old-cluster certificate.

**Warnings.** `syntropy doctor` and the dashboard warn while the cluster has one keyholder and no recovery code.

### 6. Mutual authentication before any traffic

#### 6.1 Every QUIC or TCP connection

1. **TLS 1.3 handshake.** Both endpoints present their Ed25519 keys: iroh's endpoint identity on QUIC, and the same key form over rustls on TCP. Each side checks that the peer key equals the key it expected: the dialled node ID or runner key, or any key on accept.
2. **ALPN gating.**

   | ALPN | Required before any other message |
   |---|---|
   | `syntropy/ctl/1`, `syntropy/blob/1` | an `AdmissionCert` from each side (control `Hello`/`HelloAck`, blob `Hello`/`HelloAck` with `cred_kind = AdmissionCert`) |
   | sequence and `Tap` channels on `syntropy/data/1` (daemon ↔ runner) | the daemon side presents its `AdmissionCert`, and the runner side its `DataPlaneCredential` (§6.3) |
   | runner-to-runner `syntropy/data/1` | a `DataPlaneCredential` from each side (§6.3) |
   | `syntropy/join/1` | nothing; only the join messages of RFC-0030 §6.7 are accepted, rate-limited |

3. Until both credentials verify, the acceptor processes nothing else from the connection and opens no streams towards it. Failure closes the connection with `CredentialInvalid` and increments the dashboard's failed-admission counter (RFC-0001 §10.1, `Auth`). There is a 5 s deadline.
4. **Revocation during a connection.** On `NodeRevoked` or a key rotation that invalidates a peer, every node closes that peer's connections within one heartbeat period. Daemons tell their runners with `PeerRevoked { node }` (IPC, §Proposed amendments), and runners close channels to it.

#### 6.2 Beacons and discovery

Beacons grant nothing. They only cause dial attempts that §6.1 authenticates (RFC-0030 §4.2). An unclustered node's beacons are signed by its node key. A clustered node's beacons are blinded and authenticated with a keyed hash under the cluster's beacon key, so only members recognise them, and it sends them only on networks where the cluster lives (RFC-0030 §4.8). Its mDNS records are blinded the same way.

The beacon key is 32 random bytes generated by the bootstrap keyholder, delivered to each member in its `JoinGrant` inside the authenticated connection, stored as `identity/beacon.key` with the protection of §4, and replaced at every cluster key rotation (§8.2). A revoked member can recognise the cluster's beacons until the next rotation. That is accepted: beacons carry nothing but a dial hint, and revocation already closes every connection to it.

#### 6.3 Data-plane credentials (runner identity)

```text
DataPlaneCredential v1 (168 bytes, little-endian)
off  len  field
0    1    version       1
1    1    flags
2    2    purposes      bitmask of ChannelPurpose (Activations, Tokens, Sequence, Collective, Replay, Bulk)
4    4    reserved
8    32   runner_key    Ed25519 public key of the runner's endpoint
40   32   node_id       the issuing node (signer)
72   8    instance
80   16   epoch_from    PlanEpoch (term, seq): valid for this epoch and later epochs of the instance
96   8    not_after     Unix seconds; default issue time + 24 h; renewed by the daemon over IPC
104  64   signature     Ed25519 by the node key over "syntropy-dpcred-v1" ‖ bytes 0..104
```

A runner's channel `Hello` or `HelloAck` credential is valid when all of these hold:
- the TLS peer key equals `runner_key`;
- the signature verifies under `node_id`;
- `node_id` is a non-revoked member with `STAGE_HOST` (or `INGRESS` for `Tokens` towards an ingress daemon);
- the plan for the frame's epoch places `node_id` at the expected stage or island position (RFC-0031 `NotANeighbour` otherwise);
- `instance` matches, epoch ≥ `epoch_from`, and the credential has not expired.

The runner checks plans and membership against the plan slices and member list its daemon gives it, which are updated over IPC.

#### 6.4 Local IPC

RFC-0001 §12 Z0 and Z1, and ADR-0015:
- The daemon's socket or pipe sits in a per-user 0700 directory, or carries an owner-only DACL.
- The daemon checks each connecting peer's OS credentials: `SO_PEERCRED` on Linux, `getpeereid` on macOS, and the client process's token SID on Windows.
- A runner accepts only its inherited IPC endpoint and never listens.
- The CLI talks to the daemon over the same socket, authorised by peer credentials (RFC-0051).

#### 6.5 Human-verifiable fingerprints

Two different checks exist, for two different jobs:

| Check | Shown as | Strength | Use |
|---|---|---|---|
| Pairing SAS (RFC-0030 §6.2, ADR-0302) | 6 digits, with number matching on the approving side | about 20 bits, enough for one interactive attempt: an attacker gets one guess per pairing, and the user sees a failed or unexpected pairing | admitting a LAN device |
| Identity fingerprint | 6 words from a 2,048-word list: `BLAKE3("syntropy-fp-v1" ‖ key)[0..66 bits]` | 66 bits, enough to resist offline second-preimage search | comparing a node ID, a cluster ID or a self-signed API certificate out of band (`syntropy identity show`, the dashboard's member list, RFC-0051's invite and admission views) |

This answers RFC-0051 open question 4. Six words is the agreed identity fingerprint, and the admission check stays the SAS, because the pairing protocol limits an attacker to one try.

**One guess per pairing needs commit-then-reveal.** In the first draft the responder sent `nonce_k` after seeing `nonce_c`. An attacker in the middle could advertise a lookalike cluster to the new node while pairing itself with the real keyholder, learn the keyholder's code, then pick the `nonce_k` it sends the new node so both codes matched, in about 10⁶ hashes. The joiner now commits to `nonce_c` in `PairRequest` and reveals it only after it has received `nonce_k` (RFC-0030 §6.2). Neither side can choose its nonce after learning the other's, so the attacker's codes match with probability 10⁻⁶ per attempt. Number matching on the approving side (RFC-0030 §6.2) means an approval cannot be given without comparing the codes.

**The word list** (open question 8, answered). The list is the BIP-39 English list, which is MIT-licensed, after a review for offensive words; the requirement is a licence on ADR-0018's allow list, not "public domain". Fingerprints are displayed as `word-word-word-word-word-word`, never as a numbered list. The list is a fixed, versioned asset of the binary. The recovery code (§5.5) uses the same list.

*Note 2026-10-22 (DIST-E2-T08, the offensive-word review):* version 1 of the list is the BIP-39 English list from `trezor/python-mnemonic` at commit `b57a5ad`, vendored in `third_party/bip39/` with its MIT licence. Upstream's file is identical to `bitcoin/bips` `bip-0039/english.txt` (SHA-256 `2f5eed53…dbda`). The vendored copy differs from it in one line: line 259 holds `cairn` in place of a word the naming rule (D5) keeps out of the repository. All 2,048 words were reviewed against one rule: replace a word whose main everyday use is to insult or demean a person (by intelligence, mental health, appearance or character, or by labelling them with a condition), or that is sexual or obscene. Keep a word whose main use is neutral, even when it names something unpleasant, such as a weapon, a crime or an illness, or has a secondary insulting sense.
- **Replaced**, each by a word that sorts between the same neighbours, so the list stays sorted and keeps BIP-39's unique four-letter prefixes: `addict` → `addition`, `crazy` → `crayon`, `dumb` → `duet`, `insane` → `inscribe`, `liar` → `lexicon`, `ugly` → `ukulele`.
- **Kept** under the rule, and noted because a reader might ask: `fat`, `lazy`, `rude`, `nasty`, `cruel`, `evil`, `naive`, `silly`, `dwarf`, `peasant`, `inmate`, `orphan`, `abuse`, `assault`, the weapon words, and `satoshi`.
- The replacements live in `syntropy-net`'s `words` module, not in the vendored file. Changing any word makes a new list version, because fingerprints and recovery codes already shown would read differently.

### 7. Invite token security

RFC-0030 §6.4 defines the format and flow (ADR-0303). Their security properties:

| Property | How it is achieved |
|---|---|
| **Unguessable** | 128-bit secret from the CSPRNG. Online guessing is rate-limited to 20 failures per hour per keyholder before a 10-minute lockout (RFC-0030 §6.7) |
| **Not recoverable from the log** | only `token_id` and `verifier`, two distinct BLAKE3 derivations, are logged; inverting them is a preimage attack |
| **Cluster authenticated to the joiner, even through a relay** | the token names the rendezvous node key; TLS proves possession; the rendezvous presents a certificate signed by a cluster key whose BLAKE3 prefix equals the token's `cluster_tag`. Forging that needs a 64-bit second preimage on BLAKE3, infeasible within a token's lifetime |
| **Secret disclosed only to the cluster** | sent only after the check above, inside the TLS session; never logged or stored |
| **Single use** | the coordinator emits `NodeAdmitted` and `InviteConsumed` atomically. Concurrent uses: one wins, the others get `TokenExhausted` |
| **Expiry** | default 24 h, cluster maximum 7 days (`invite.max_expiry`); keyholder-scope tokens ≤ 1 h |
| **Scopes** | become certificate roles at issuance (§5.2); an ingress-only token can never yield `STAGE_HOST` |
| **Revocable before use** | `RevokeInvite`; all outstanding invites are also revoked by a cluster key rotation, because their `cluster_tag` no longer matches |
| **Visible use** | every invite admission raises a dashboard and CLI notification ("*Laptop* joined using an invite created on *Studio-PC* 2 h ago") with a one-click `Revoke` |

**Residual risks.**
- A leaked *unused* token is a bearer credential until it is used, expires or is revoked. Short expiry, single use and the admission notification bound the damage. For stricter clusters, `invite.require_confirm = true` makes every invite join wait for a keyholder's approval, like a pairing.
- After a network-split heal, a token consumed only on the losing side becomes usable again until it expires (RFC-0030 §8.2). Short expiry bounds this. Keyholders that issued certificates on the losing side re-submit them, so the joined node is not lost.

### 8. Revocation and key rotation

#### 8.1 Revocation

- **Generations, not serials.** Each member has an admission generation `g` in the log (`MemberRecord::generation`), and every certificate carries it (§5.1).
  - First admission gives `g = 1`.
  - Re-admitting a node after revocation is an explicit, keyholder-signed admission that records `g + 1`.
  - Renewals keep `g`.
- **Revoking.** `syntropy cluster remove <node>` on a keyholder produces a `SignedRevocation { node, generation, issued_at, keyholder, sig }`, where `sig` is the keyholder's node-key signature (context `syntropy-revoke-v1`). The keyholder submits `Revoke` carrying it.
  - Every node verifies the signature against the keyholder's certificate as part of validating `NodeRevoked` (§8.3).
  - `apply()` sets `revoked[node] = generation`.
  - Every certificate of that node with `generation ≤ g` is invalid, whatever its serial. That includes certificates whose `CertIssued` never reached the log.
- **Local deny list.** Every node appends the *signed* revocation to `state/identity/revoked.list` when it applies it. §5.4 step 5 consults that list independently of the log, so a revocation survives a split heal and a snapshot restore. Unsigned or unverifiable entries are never written, so a coordinator cannot plant one.
- **Keeping revocations.** Keyholders keep every revocation they signed, and re-submit it after any term change or snapshot replacement (RFC-0032 §6.10).
- **Gossip, not only the log**. A coordinator could omit a `Revoke` aimed at itself, so signed revocations also travel outside the log:
  - The signing keyholder pushes each `SignedRevocation` to every member it reaches, over `Rpc::RevocationPush` (RFC-0030 §8.3).
  - Every beat carries `deny_digest`, a count and BLAKE3 over the sender's deny list (RFC-0030 §5.2). A peer whose digest differs pulls the missing entries with `Rpc::RevocationSync`.
  - Any node that receives a revocation verifies it: a keyholder certificate valid under §5.4, and a keyholder not itself revoked at or before `issued_at`. It then applies it at once: deny-list entry, connections closed, the node refused under §5.4 step 5. No log entry is needed for that.
  - A follower holding a verified revocation of its coordinator stops following it, treats the lease as expired and ignores its beats. An election follows (RFC-0032 §7.3).
  - Signed invite revocations and API-key revocations travel the same way. The rendezvous keyholder and every API-checking daemon then honour them even if the log lacks them.
- **Effects.**
  - All connections to the node close within one heartbeat (§6.1).
  - Its runners lose their channels.
  - Its stages are recovered as a node loss (RFC-0033).
  - Its chunks are no longer used as a source.
- **Re-admission.** A revoked node that is later re-paired receives a certificate at `g + 1`. Re-admitting a removed device is an explicit act.

#### 8.2 Cluster key rotation

**When it happens.**
- Revoking a keyholder records only `NodeRevoked`, because `decide()` cannot generate a key. The *revoking keyholder's daemon* then starts the rotation below.
- `syntropy cluster rotate-key` starts it manually.

```text
1. Keyholder K generates (new_sk, new_pk).
2. K builds RotationStatement { old, new, parent_term, keyholder: K, sig_old, sig_new, sig_k }:
     sig_old = Ed25519_old("syntropy-keyrotate-v1" ‖ new ‖ parent_term)
     sig_new = Ed25519_new("syntropy-keyrotate-v1" ‖ old ‖ parent_term)
     sig_k   = K's node-key signature over the statement (K must hold a KEYHOLDER certificate under `old`)
   and persists it in state/identity/rotations.list.
3. K sends new_sk only to peers that present a KEYHOLDER certificate under `old`, are not revoked in
   K's state, and appear in K's keyholder set (RPC KeyholderKeyUpdate; each recipient stores it in
   identity/cluster.key.next).
4. K submits RotateClusterKey { statement }; every node validates it (§8.3) and records
   ClusterKeyRotated { statement, accept_old_until_ms: now + security.rotation_window (7 days) },
   plus InviteRevoked for every outstanding invite.
5. K re-issues a certificate under `new` for every current member (RecordCert each, carrying the
   certificate). Members fetch theirs with RenewCert on their next contact, and keep their
   previous-key certificate until the rotation is applied in the log they follow.
6. Until accept_old_until_ms, certificates under a key on the verified chain are accepted only for
   node IDs that are already members (§5.4 step 3). After the window, they are rejected, and members
   that were offline for the whole window must be admitted again.
```

**The rotation chain.**
- Nodes keep every verified `RotationStatement` in `rotations.list`, outside the log, as they keep revocations.
- Control `Hello` messages carry the presenter's certificate plus the statements linking its key to the oldest key the presenter still holds.
- A certificate's key is accepted when a chain of statements links it to the verifier's current key in either direction, and every statement in the chain verifies.
- A chain is rejected if it contains a statement whose `keyholder` the verifier holds as revoked at or before that statement's `parent_term`. A revoked keyholder therefore cannot mint its own branch.
- **Recovery statements** (§5.5) carry `sig_recovery` in place of `sig_old`. One verifies when `sig_recovery` checks under the `recovery_pk` that the verifier's state held for the `old` key at `parent_term`, and `sig_new` checks under `new`. Its `keyholder` is the recovering node, which need not hold a KEYHOLDER certificate under `old`. In every other respect it is a link in the chain like any rotation.

**Rotation on one side of a split**:
1. The two sides can now authenticate each other: their keys share an ancestor on the chain.
2. The heal follows RFC-0032 §7.5.
3. Keyholders re-submit their `RotateClusterKey` statements as facts. The winner's log applies those whose `old` is its current key.
4. If both sides rotated from the same parent, the winner's rotation stands, and the losing branch's keyholders re-issue nothing.
5. Members holding a losing-branch certificate are re-issued under the winner's current key, while their losing-branch certificate is still accepted under the rule above.
6. The losing branch's key is retired when the window closes.

#### 8.3 Authority of control-plane events

Followers never apply an entry blindly. `validate(state, entry)` (RFC-0032 §6.3) runs on every node before `apply()`. It checks the entry's preconditions and the authority rules below, and an entry that fails is refused. A coordinator that appends a forged entry therefore splits itself off: honest followers stop accepting its appends, and elect a new coordinator.

| Event | Authority every node verifies |
|---|---|
| `NodeAdmitted` | carries a keyholder-signed `Approval { candidate, generation, roles, proof_digest, keyholder, sig }` (context `syntropy-admit-v1`); roles may not exceed what the proof allows (invite scopes, ingress-only, pairing defaults). An `Invite` proof is valid only with `InviteConsumed` for that token in the same batch, and only if no earlier `Approval` in the state names the same token ID. Consumption is derived from the signed approvals, so a coordinator that omits `InviteConsumed` cannot re-use a single-use token |
| `CertIssued` | carries the certificate itself; its signature verifies under the cluster key, and its roles and generation match the admitted record |
| `NodeRevoked` | carries a `SignedRevocation` from a non-revoked keyholder (§8.1) |
| `KeyholderChanged` | carries a promotion or demotion signed by an existing, non-revoked keyholder, plus the promoted node's own acceptance signature; a node cannot promote itself |
| `ClusterKeyRotated` | carries a `RotationStatement` that verifies (§8.2) and whose `old` is the current key; a recovery statement verifies against the state's `recovery_pk` instead (§5.5) |
| `RecoveryKeySet` | signed by a keyholder; replaces the state's `recovery_pk`. A recovery rotation clears it (§5.5) |
| `InviteCreated`, `InviteRevoked`, `PairingWindowOpened`, `PairingWindowClosed` | signed by a keyholder |
| `ClusterConfigChanged` | signed by the submitting node, which must hold `STAGE_HOST`: a patch from a member without it, such as an ingress-only or invite-joined guest, is refused. Keys whose schema `authority` is `keyholder` also need a keyholder signature: all of `security.*`, `invite.*`, `cluster.lan_admission`, `coordinator.*` and `swim.*`, and the keys that change behaviour other users rely on, which are `cluster_aliases`, `network_scope`, `allow_approximate`, `update.*`, and the exactness and pin fields of `policy.*`. RFC-0054 owns the per-key `authority` field and the full key list |
| `ApiKeyCreated`, `ApiKeyRevoked` | signed by a keyholder (§8.4) |
| `CoordinatorElected` | a `SignedElection` by the elected node's node key, which holds a valid, non-revoked `COORDINATOR` certificate; the term follows RFC-0032 §7.2's counter rule within one log, and RFC-0032 §7.5's heal order across branches |
| `ChunkIdsCommitted` | carries the `ChunkEvidence` from its `ProposeChunkIds` (§10.1; RFC-0041 §6.3): origin chunk hashes, two independent `STAGE_HOST` witness signatures, or a keyholder signature over the table and the file's SHA-256. With no valid evidence, the batch is recorded as provisional (`evidence: None`) and is never served to peers for packing |
| `NodeFailed`, `NodeSuspected`, `PlanCommitted`, runner and graph events | the appending coordinator's authority, which is availability only (§2 trust tiers) |

**Keyholder approval path.** One path serves every operator action whose event needs a keyholder signature:
- opening a pairing window;
- creating or revoking a cluster-wide API key (§8.4);
- a `SetClusterConfig` patch that touches any key with `authority = keyholder` (RFC-0054 §2.3 step 10);
- the chunk table of a model file imported locally on a non-keyholder node (§10.1 form 4).

On a keyholder node the action is signed directly, after its operator confirms. On any other node the daemon forwards it to a reachable keyholder with `Rpc::ApprovalRequest { kind, body }` (RFC-0030 §8.3):
- `kind` names the command (`OpenPairingWindow`, `CreateApiKey`, `RevokeApiKey`, `SetClusterConfig`, `ApproveImport` for a local model import's chunk table, §10.1);
- `body` is the command's canonical CBOR.

That keyholder's operator approves or rejects the request. The keyholder's signature comes back in the `Approval` reply and is attached to the command, which is then submitted as usual. Until then, the submitting node shows the action as pending. A patch that mixes member-authority and keyholder-authority keys travels this path as a whole.

*Note 2026-10-22 (#1698): when `KeyholderRequired` is raised.* An action on this path fails with `KeyholderRequired` (§15) when no keyholder can take it.
- **Who is tried.** The submitting node is not a keyholder. It tries every member with a valid, non-revoked KEYHOLDER certificate (§5.4) that its membership view (RFC-0030 §5.1) does not show as `Failed`, `Left`, `Sleeping` or `Revoked`. So a keyholder that is `Probing` or `Suspect` after a lossy moment is still tried.
- **When it is raised.** Only once none of those takes the request: each one can't be reached, or declines to queue it. It is not raised while a keyholder has the request and its operator has yet to answer; the action is pending then.
- **A rejection is something else.** An operator who rejects the request is a different outcome, reported as that rejection, never as `KeyholderRequired`. No error code names the rejection yet. That gap is open, and outside #1698.
- **Why 403, not 503.** RFC-0050 §12.1 maps `KeyholderRequired` to 403. The condition is about who may authorise the action: this node can't make the signature, and repeating the request changes nothing until a keyholder is online. A 5xx status would invite automatic retries, which would pile up as pending approvals once a keyholder returns.

**Snapshots carry the same proof**. Every authority-derived field of `ClusterState` keeps the signed record that created it, and a node that loads a snapshot re-verifies all of them before adopting it (RFC-0032 §6.7):
- members, with their `Approval`s and certificates;
- the keyholder set, as the chain of signed `KeyholderChanged` records from the bootstrap keyholder;
- revocations, as `SignedRevocation`s;
- rotations, as `RotationStatement`s;
- invites, as keyholder-signed `InviteCreated` and `InviteRevoked` records, with consumption derived from approvals;
- security-class configuration, as keyholder-signed patches;
- cluster-wide API keys, as signed records;
- chunk tables, with their evidence.

No keyholder issues or renews a certificate on the strength of state it has not verified. Before signing an invite admission, a rendezvous keyholder checks the token against the signed `InviteCreated` record itself, not only against its replica.

The signatures are Ed25519 over the event's canonical bytes plus a context string, by the signing keyholder's node key. Verification checks that key against a KEYHOLDER certificate valid under §5.4. Every check is pure: it depends only on the state and the entry, so it is deterministic across nodes.

#### 8.4 Cluster-wide API keys

RFC-0050 §11.3 and ADR-0505 own the key format (`syk_…`), the scopes and the presentation headers. This section makes keys valid on every node.

- **Creation.** `syntropy api-key create --cluster [--instance <id> | --model <name>] [--expires 90d]` runs on any node. The secret is generated and shown there once. The node submits `CreateApiKey { key_id, hash: BLAKE3(secret), scopes, instances, models, expires_ms }`, which a keyholder signs; a non-keyholder node forwards it by the keyholder approval path (§8.3, `Rpc::ApprovalRequest`), the same path as security-class configuration patches. The coordinator records `ApiKeyCreated`. The secret itself never enters the log.
- **Scope.** Optional instance and model lists narrow a key to particular models: an `inference` key scoped to `model = "qwen3-235b"` is refused on any other model. An empty list means all.
- **Checking.** Each daemon checks a presented key against its node-local keys (RFC-0050) and the log's `api_keys` set, by hash and in constant time.
- **Rotation.** Create the new key, distribute it, then revoke the old one. Overlap is allowed, so clients rotate without downtime. `syntropy api-key rotate <id>` does both steps.
- **Revocation** is monotonic. `ApiKeyRevoked { key_id }` is keyholder-signed, kept by the revoking keyholder and re-submitted after any divergence (RFC-0032 §6.10). A revoked `key_id` is never re-admitted: `apply()` refuses `ApiKeyCreated` for a revoked ID.

### 9. Encryption everywhere, and the RDMA and shared-memory policy

#### 9.1 Policy (ADR-0314)

| Path | Protection | Rule |
|---|---|---|
| Control, join, bulk (QUIC) | TLS 1.3 | always |
| Data plane over QUIC or TCP | TLS 1.3 | always |
| Relayed QUIC | end-to-end TLS 1.3; the relay forwards ciphertext | always |
| RDMA over Thunderbolt 5 | plaintext, with CRC32C and a channel token per frame | allowed: a direct point-to-point cable between two admitted nodes. QPs are paired only through the authenticated control connection (ADR-0011 item 4), and the first frame on a QP is a credentialed `Hello`. An attacker needs physical access to the cable |
| RoCEv2 / InfiniBand | plaintext | **only** on interfaces the operator lists in `security.trusted_fabric`: a direct cable, or a switch carrying only cluster members. Otherwise RDMA is not used on that interface, unless `rdma.encrypt_untrusted = true` (§9.3) |
| Shared memory | none | same host **and** same OS user only (segment mode 0600, or an owner-only DACL); otherwise the edge falls back to QUIC over loopback |
| Local IPC | none | same user, peer-credential checked (§6.4) |

**Visibility.** Every edge that carries data in plaintext is marked in the dashboard's topology view and in `syntropy doctor`, with the reason: "Thunderbolt cable" or "trusted fabric: eth2".

#### 9.2 `trusted_fabric`

- `security.trusted_fabric` is node-scoped and lists interface names or CIDRs. Setting it is an operator assertion that nobody outside the cluster can observe or inject on that segment.
- `syntropy config set security.trusted_fabric eth2` prints what the assertion means before it takes effect.
- **A mechanical gate** (open question 6, answered). The assertion is accepted only for an interface that is wired, has no default route, and whose ARP or NDP table holds only cluster members. Otherwise the command refuses, unless the operator passes `--force` and types the interface name back. `syntropy doctor` flags a forced assertion, and every link probe on the interface (RFC-0010) re-checks the three conditions; an edge that fails them stops being treated as trusted until they hold again.
- The flag appears in `IfaceInfo::flags` (RFC-0030 §7.1), so the solver can use RDMA there. An edge counts as trusted only if **both** ends mark their interface.

#### 9.3 Optional RDMA encryption

`rdma.encrypt_untrusted = true` is node-scoped and arrives in M4. It enables RDMA on untrusted interfaces with per-frame AEAD:
- **Key.** Exported from the authenticated QUIC connection used for pairing, through the TLS exporter: label `syntropy-rdma-aead-v1`, context = channel key. Each direction has its own key.
- **Nonce** (96 bits). Byte 0 is the nonce space: 0 for data frames and 1 for control frames. Bytes 1–3 are zero, and bytes 4–11 hold a u64 counter. Data frames use `chan_seq`. Control frames (`Hello`, `HelloAck`, `Credit`, `Ping`, `Close`), whose `chan_seq` is 0, use `ctl_seq`, a per-direction counter of sealed control frames that starts at 1 and is carried in the header's `step` field. So no `(key, nonce)` pair repeats. A receiver drops a control frame whose `ctl_seq` is not above the highest it has accepted.
- **AAD**. Bytes 0..56 of the base header, then the extension's first 16 bytes, which stay in plaintext so frames can be routed before decryption. Both are fixed for the frame's lifetime. This is the same rule as RFC-0031 §6.1.
- **Sealing.** Covers the payload and the extension beyond its first 16 bytes, with the `ENCRYPTED` flag set. The 16-byte tag is a trailer counted in `payload_len` (RFC-0031 §6.1). On an encrypted channel, every frame is sealed, control frames included. An unsealed frame is dropped, so an on-path attacker cannot forge credits, `RESET` or `Close`.
- **Resends and re-keying.** A resend retransmits the stored sealed bytes unchanged. A re-pair over a new QUIC connection exports new keys and flips flags bit4 `KEY_PHASE`. Retained frames keep their original phase and bytes. The receiver keeps the previous key until it has acknowledged every `chan_seq` sealed under it, and `ctl_seq` restarts at 1 under each new key.
- **Cost.** It is paid on both ends, about 8 µs per 32 KiB at 4 GB/s per side [estimate]. It is measured into the edge's `α_eff` (RFC-0010), so the solver weighs it honestly against QUIC.

### 10. Model integrity

#### 10.1 Content addressing

- **Chunks** (ADR-0016, RFC-0041) are verified against their BLAKE3 entries in the model's chunk table before they are stored or served.
  - A mismatch discards the chunk and refetches it from another source, and the store marks the source suspect for that model.
  - A node that serves mismatching chunks for 3 different models is flagged in the dashboard: "*node* served corrupt data: faulty disk or compromised". It is excluded as a chunk source cluster-wide (`SourceDistrusted`, §15) until an operator clears it.
  - It is not revoked automatically, because disk faults are far likelier than attacks.
- **The chunk table and `ModelId`** bind every tensor's content hash (RFC-0001 §5.3). All nodes verify against the same table. Its batches are committed by digest through the log (`ChunkIdsCommitted`, RFC-0041 §6), and their bodies move by digest over the blob plane.
- **Origin bytes** arrive over HTTPS, validated against the platform's root store. They are checked against the Hub's chunk-protocol hashes where offered, and against the whole-file SHA-256 once a node holds the complete file (ADR-0016). Trust on first fetch from the origin rests on TLS. That limit is stated, not hidden.
- **Who may fix a chunk table**. The first fetcher is never the sole authority. `ChunkIdsCommitted` carries `ChunkEvidence`, which `validate()` checks (§8.3), in one of three forms, with item 4 covering files that have no origin:
  1. **`OriginHashes`.** The origin publishes content-addressed chunk metadata for the file. The table records, for each `ChunkId`, the origin hashes covering its bytes. Every node that receives such a chunk, from the origin or from a peer, checks it against origin metadata it fetched itself over HTTPS, as well as against the `ChunkId`. The proposer is trusted for nothing.
  2. **`Witnesses`.** Two `STAGE_HOST` members with distinct node keys each computed the entries from bytes they read themselves, from the origin or from a local file whose SHA-256 matched the origin's published file digest. Each signs `(file SHA-256, batch digest)`. A chunk needed by a second node, with no origin metadata, is therefore read from the origin a second time by that node, not copied from the first. Such chunks cross the internet up to twice per cluster instead of once (RFC-0001 §7.1).
  3. **`KeyholderVerified`.** A keyholder read the whole file, matched its SHA-256 against the origin's published digest, computed the table, and signed `(file SHA-256, batch digest)`.
  4. **Local imports with no origin** (a file the user imports from disk, with no hub to fetch hashes from or to re-read). This uses form 3, with the operator standing in for the origin's digest:
     - **On a keyholder node**, the import is the evidence. The keyholder reads the whole file, computes its SHA-256 and the table, and signs `(file SHA-256, batch digest)` with `origin: None`.
     - **On any other node**, the importer computes the SHA-256 and the table, then asks a keyholder through `Rpc::ApprovalRequest { kind: ApproveImport, body }` (§8.3). The body names the model, file name, size, SHA-256 and batch digests. The keyholder's operator sees the importing node, the file name, size and SHA-256, and approves or rejects. On approval, the keyholder streams the file from the importer over the blob plane without storing it, checks the SHA-256 against the approved value, recomputes the table, and signs.
     - The streaming check costs one LAN read of the file. Without it, a compromised importer could show the true SHA-256 with a poisoned table.
     - Until approved, the batches stay provisional. The importer uses them itself, and no peer packs from them.
  
  Until a batch has evidence, its chunks are *provisional*. A node may use provisional chunks it read from the origin itself, but no node serves them to another node for packing. Every node verifies every chunk it receives against the committed table before writing it (RFC-0041 §6.3). A one-node cluster needs no evidence, because only the node itself consumes its chunks.

#### 10.2 Packs

- **Packs are not trusted across members.** `pack.fetch_from_peers` stays `false` after M3, so a node builds its own pack from verified source chunks and never takes another member's repacked extents. If it is ever enabled, the receiver rebuilds and compares at least 1 % of the extents plus every canary extent, and the setting is labelled "trusts members". This keeps RFC-0041 §8.5 consistent with A5 (§2): one member cannot poison another's weights through a repack (OQ-SEC-06; RFC-0041 owns the setting).
- The pack sidecar records a BLAKE3 hash per 4 KiB-aligned extent (ADR-0005), computed while the pack is written from verified source bytes.
- `integrity.verify_on_load` (default `sample`) re-hashes a random 1 % of extents, plus every extent the warm-up canary touches.
- `full` re-hashes everything. `syntropy doctor --verify` does the same on demand.
- The instance canary at `Warming` (RFC-0001 §3.3) catches gross corruption in any case.
- A mismatch deletes and rebuilds the pack (RFC-0001 §10.1, `Integrity`).

#### 10.3 Publisher signatures

When a model repository publishes a signature in the OpenSSF Model Signing (OMS) format, a Sigstore-based bundle over the model's files, the store verifies it at registration.
- `models.signatures = verify_if_present` is the default: it verifies when a signature is present and shows "signed by *identity*", "unsigned" or "signature invalid" in the CLI and dashboard. It never blocks.
- `require` refuses unsigned or invalid models. Its trusted identities or keys come from `models.trusted_signers`.
- `off` skips signature checks.

M4. Whether a Rust verifier for the format exists at that point is open question 5.

#### 10.4 Hostile model files

- **Parsers.** GGUF, safetensors and tokenizer parsers live in `syntropy-manifest`, which is `#![forbid(unsafe_code)]` (RFC-0001 §4.2 rule 8). They enforce limits:

  | Limit | Value |
  |---|---|
  | metadata key-value pairs | ≤ 1,000,000 |
  | tensors | ≤ 1,000,000 |
  | strings | ≤ 1 MiB |
  | dimensions | ≤ 8 |
  | header bytes | ≤ 256 MiB |

  Every offset and length is checked with overflow-checked arithmetic against the file size. The parsers are fuzzed continuously (RFC-0060).
- **No remote code.** Syntropy never executes code from a model repository: no Python, no "trust remote code", no native libraries. Tokenizers and chat templates are data. Templates are rendered by a sandboxed engine with no file, network or environment access, plus time and memory limits (RFC-0050).
- **Isolation** (open question 3, answered). From M1, the headers of a *newly added* model are parsed in a short-lived child process with resource limits (memory, CPU time, open files) and no network access, so a parser bug costs a child process, not the daemon. The parsed, validated header returns to the daemon over the child's IPC channel. OS sandbox profiles for that child (Linux seccomp and landlock, a macOS sandbox profile, Windows AppContainer) remain an M4 hardening item.

### 11. API exposure: threat-model requirements

The product area owns client-facing API authentication and bind policy: RFC-0050 §11 and ADR-0505 define keys, scopes, headers, rate limits, TLS options and the dashboard login. This RFC does not redefine them. It states what the threat model (§2) requires of them. ADR-0505 records that RFC-0034 may tighten its defaults.

| Requirement | Threat | Status against ADR-0505 |
|---|---|---|
| R1. Loopback listeners reject foreign `Host` headers and cross-origin state-changing requests, and the local socket checks peer credentials | browser-borne DNS rebinding and cross-site requests (A1 via the user's browser) | met (ADR-0505 item 1) |
| R2. Every non-loopback listener requires an API key; keys are stored hashed and compared in constant time | LAN devices and internet attackers (A1, A2) | met (ADR-0505 items 2–4) |
| R3. **No API key, prompt or output crosses a network link in plaintext.** A non-loopback listener uses TLS, unless it is bound only to an interface that is itself encrypted end to end: a WireGuard-based overlay such as a tailnet, classified `Overlay` (RFC-0030 §7.2) | sniffing on shared LAN media: Wi-Fi, guest devices (A1) | **met.** Non-loopback binds need a key and TLS. Overlay-only binds may skip TLS but need a key. Plain HTTP on a LAN needs the explicit `api.allow_plaintext_lan`, which is flagged permanently. RFC-0001 §12 and ADR-0505 are updated to match |
| R4. Administrative operations (invites, revocation, load and unload, configuration) need the `admin` scope wherever keys apply, and multi-user hosts are advised to require keys on loopback (`api.auth.mode = "always"`) | other local users (A7); leaked inference keys | met (ADR-0505 items 1 and 4); `doctor` gives the advice |
| R5. Self-signed TLS certificates are derived from or bound to the node identity, and the fingerprint is shown as six words (§6.5) | MITM on first contact | met in part (ADR-0505 item 6); the six-word form is §6.5 |

**R3, settled.** The requirement holds. The overlay exception keeps phone-over-tailnet working without certificate setup. `api.allow_plaintext_lan` is the only way to run plaintext on a LAN, and `doctor` and the dashboard flag it permanently.

### 12. Supply chain: the signing key hierarchy

RFC-0034 owns *which keys exist*, their custody, rotation and revocation. RFC-0062 §7 owns where the pipeline uses them. RFC-0052 and ADR-0508 own the update client: TUF metadata, read with the `tough` crate. ADR-0507 makes backend modules TUF targets.

#### 12.1 Keys

| Key | Purpose | Custody | Threshold | Lifetime and rotation |
|---|---|---|---|---|
| TUF root | signs the role keys below; the trust anchor embedded in every binary | offline hardware tokens, one each for three custodians, stored apart. ECDSA P-256 keys generated on the token in a PIV applet, with each token's attestation certificate recorded | 2 of 3 | metadata expires yearly; re-signed at least 60 days before expiry and on any custodian change; rotated by chained root versions, signed by the old and the new threshold |
| TUF targets | signs the artefact list for `stable` and `beta`: core binaries, backend modules, `catalogue.json`, `relays.json` (RFC-0030 §6.4), the config schema, with custom fields (build ID, protocol majors, provider ABI). It delegates `nightly/*` paths to the role below | offline hardware tokens (PIV, ECDSA P-256), one each for two custodians | 1 of 2 | 90-day expiry; re-signed per release **and** on a calendar every 60 days whether or not a release ships, which keeps at least 30 days in hand; expiry is monitored and alerts at 30 days |
| TUF delegated targets `nightly` | signs nightly-channel artefacts only, restricted by path to `nightly/*` | online, an HSM-backed Ed25519 key in a cloud KMS, reached only from the `nightly-signing` environment by the scheduled nightly workflow. This is the same key as the Syntropy nightly key below: one nightly key serves both purposes (RFC-0062 §7.1) | 1 | metadata expiry 7 days, re-signed by each nightly; key rotated yearly by the top-level targets role |
| TUF snapshot | binds the current targets version | online, an HSM-backed ECDSA P-256 key in the cloud KMS, reached only from the release pipeline's environment | 1 | 7-day expiry; re-signed automatically on each publish |
| TUF timestamp | binds the current snapshot; defeats freeze attacks | online, an HSM-backed ECDSA P-256 key in the cloud KMS, reached only from the `tuf-timestamp` environment | 1 | 1-day expiry, re-signed automatically every day |
| Syntropy release key (Ed25519) | detached signatures on Linux tarballs and packages, on the `SHA256SUMS`/`BLAKE3SUMS` files and on the `ConformanceManifest` (RFC-0060 §6.4), for verification without TUF: the install script, `setup`'s first self-check, users. It signs a digest statement, never the file (below) | an HSM-backed Ed25519 key in the cloud KMS, usable only by the `release-signing` environment with two approvers, each a maintainer or a recorded delegate (RFC-0062 §7.2) | 1 | rotated yearly; the current and next public keys are published as a TUF target (`signing-keys.json`) one release ahead, and embedded in each release |
| Syntropy nightly key (Ed25519) | the same, for the `nightly` channel only; also the delegated `nightly` targets key above. Detached signatures carry their own context string, so they can never be read as TUF metadata signatures, nor the reverse | the cloud KMS, `nightly-signing` environment | 1 | rotated yearly; never trusted for `stable` or `beta` |
| Linux repository keys | sign the apt and rpm repositories | the cloud KMS, or an OpenPGP subkey held only in `release-signing` if no PKCS#11 or KMS bridge can produce OpenPGP signatures (PROD-E10-T10 decides) | 1 | per distribution convention |
| Apple Developer ID and notarisation credential; Windows Authenticode via a cloud signing service | platform code signing (RFC-0062 §7.1) | as RFC-0062 §7.2 | — | as issued by the platform |
| Build provenance (keyless) | attestations for third-party verification | CI's workload identity | — | per build |

The node, cluster, runner and API keys of §4 are a separate hierarchy. No release key can admit a node, and no cluster key can sign software.

**Custody.** Every custodian is a maintainer or a recorded delegate. Each delegation is recorded in the repository and names the tokens and actions it covers, and a delegate acts only for the actions named.

- **Who holds what.** The three root tokens sit with three different custodians, stored apart, so no single person can rotate root. The two targets tokens sit with two custodians, so a targets re-sign or a revocation never waits on one person. The cloud KMS account's break-glass role is held by two custodians.
- **Tokens.** Root and targets keys are ECDSA P-256 keys generated on hardware tokens in a PIV applet, with each token's attestation certificate recorded, on firmware with no known side-channel flaw in its ECDSA implementation. ECDSA P-256 is used because `tough` verifies `ecdsa-sha2-nistp256` and PIV ECDSA is the path TUF signing tools support best. Three root tokens and two targets tokens are in use, and two blank spares are kept.
- **The 2-of-3 trade-off.** Any two root custodians can rotate root or revoke a compromised role without the third. That is the accepted cost of not waiting on one person.
- **Online keys.** The four online keys (snapshot, timestamp, release, nightly) live in a cloud KMS that keeps Ed25519 keys in an HSM, so the Ed25519 release key of §3 needs no algorithm change. CI reaches each key only by workload-identity role assumption pinned to its environment (`release-signing`, `nightly-signing`, `tuf-timestamp`, and the release pipeline's environment for snapshot). Key policies deny deletion and policy changes outside the break-glass role, and the KMS audit log raises an alert on any signing call that the expected workflows did not make. No agent run is ever given one of these environments.
- **Asynchronous signing.** Custodians never meet. Each signs root or targets metadata on their own machine, with their own token, through a signing-event pull request in the TUF repository. The ceremony tooling (QUAL-E16-T20) opens the event, collects signatures until the threshold is met, and publishes.
- **Cadence.** Root yearly, at least 60 days before its 1-year expiry, and on any custodian change. Targets per release and every 60 days. Snapshot on each publish and timestamp daily, both automated. Release and nightly keys yearly, with the next public key in `signing-keys.json` one release ahead. Every §12.3 response is rehearsed once on a staging repository before M1 (DIST-E17-T05), then in a yearly tabletop run by the custodians.
- **Who does what.** Key generation, the root and targets ceremonies, the scheduled targets re-signing and every compromise response that needs an offline token are human actions done by the custodians (**Maintainer action**). Agents write and test the ceremony runbooks, the CI signing jobs, the KMS key policies and alerts, and the expiry monitor (RFC-0062 §7), but no agent ever holds, reaches or handles an offline key.

**What the release key signs.** A KMS signing call accepts a bounded message, a few KiB, not a multi-gigabyte artefact, so the release and nightly keys never sign a file. They sign a 51-byte digest statement:

```text
statement = "syntropy-release-v1" ‖ SHA-256(file)        19 + 32 bytes
signature = Ed25519(release or nightly key, statement)     64 bytes, written as <file>.sig
```

The install script, `setup`'s self-check, the daemon's `ConformanceManifest` check and a user checking by hand all compute SHA-256 over the downloaded file, build the same statement and verify against it. The `SHA256SUMS` and `BLAKE3SUMS` files are signed the same way. Which key verifies tells the channel: a nightly-key signature is never accepted on a `stable` or `beta` artefact.

#### 12.2 Verification rules

1. **Updates and modules** are verified by the TUF client against the embedded root before they are installed, as ADR-0508 and RFC-0052 specify. The module loader then re-checks the module's hash and its build ID against the core, immediately before `dlopen`. It loads by absolute path from the user-owned modules directory only, never through `PATH`, `LD_LIBRARY_PATH` or `DYLD_*` search. A mismatch deletes the module (`ModuleSignatureInvalid`).
2. **Detached release-key signatures** are checked by `setup`'s first self-check, and by users verifying a download. Release keys are trusted only if they are embedded in the running binary or listed in a TUF-verified `signing-keys.json`. This self-check protects against corrupted mirrors and partial downloads. It does **not** protect against a compromised release host, because a hostile host can ship a binary with its own keys. The trust root of each install path is therefore stated plainly:
   - **curl script, PowerShell script:** TLS to the release host, plus the platform signatures the OS checks (Authenticode on Windows; none for a curl-downloaded macOS binary, which has no quarantine flag).
   - **Package managers** (Homebrew, winget, apt, rpm): the package manager's own repository signatures.
   - **Out-of-band check**: the release-key fingerprint, in hex and in six-word form (§6.5), is published away from the release host: in the project's source repository, in every release's notes, in the documentation and package-manager channels, and as an entry in the public transparency log that also records keyless build provenance (§12.1). A user who passes `--verify-key <fingerprint>` to the install script gets a check against that independent source (RFC-0052 §2.1). The check compares the full 256-bit fingerprint in hex, and only that. The six-word form (66 bits, §6.5) is for display and reading aloud, and is never accepted as a verification input. Substituting a key then needs the release host and the repository compromised together, and still leaves a public log entry.
   - **After the first install:** TUF's embedded root anchors every update.
3. **Platform signing** applies on top, and the OS enforces it (Gatekeeper with library validation, SmartScreen, package managers). Syntropy's own checks run on every platform, including bare tarballs.

#### 12.3 Compromise response

| Key compromised | Response |
|---|---|
| timestamp or snapshot | the root threshold signs new role keys; clients pick them up at the next refresh |
| targets | the root threshold rotates targets, and every artefact is re-signed and re-published; clients reject the old targets key after the new root version |
| release or nightly key | revoke it in `signing-keys.json`, a TUF target, so updated clients distrust it; rotate it; a maintainer publishes an advisory through the security contact (**Maintainer action**) |
| delegated `nightly` targets key (the nightly key) | both this row and the release-or-nightly row apply: the top-level targets role replaces the delegation's key and re-signs; nightly clients reject the old key from the next targets version; stable and beta are unaffected by construction |
| one root token | the holders of the remaining two tokens rotate the root, removing the lost key (**Maintainer action**) |
| two root tokens | treated as a full compromise: a new root ships only in a release a maintainer verifies by hand, announced out of band through the security contact (**Maintainer action**) |

#### 12.4 Dependencies and builds

- `cargo-deny` checks licences, advisories and bans. `cargo-audit` runs on every PR. `cargo-vet` records audited crates.
- Vendored C and C++ (ggml, HiGHS) are pinned to commit hashes.
- A CycloneDX SBOM ships with each artefact.
- Reproducible builds are tracked by RFC-0062 (ADR-0623).
- Dependency updates follow the project policy: auto-merge minor and patch versions that pass CI, hold majors.

### 13. Local privacy posture

- **No telemetry by default** (RFC-0001 §12 rule 4). Metrics and trace exporters are opt-in per node (RFC-0053).
- **Outbound connections are caused by user actions or features, and listed:**

  | Destination | When | Sends |
  |---|---|---|
  | Model hub (HTTPS) | fetching a model the user asked for; also metadata reads the user triggers before any pull: header range reads for estimates and previews, model cards and licences, gated-access status | range requests; the user's Hub token if configured for gated models |
  | iroh relays: the built-in set from the signed `relays.json` target, or the operator's own (RFC-0030 §6.4) | an invite-joined member, an outstanding invite, or a member allowed off-LAN reach at pairing, and `net.relay_mode ≠ disabled` | encrypted QUIC; the relay operator learns which node IDs communicate and from which IP addresses. Weight chunks never travel through a relay |
  | Update server | update checks (RFC-0052): daily by default, can be disabled | plain GETs of static TUF paths that name the platform and channel; the installed version is not sent, and there are no identifiers and no cookies (RFC-0052 §7.2, RFC-0053 §8) |
  | (none) publisher-signature trust root | model-signature checks (§10.3) | nothing: the Sigstore trust root is embedded in each release and refreshed only by updates |
  | Cluster members | always | per this RFC |

  `--offline` disables the first three. `syntropy doctor --network` lists every outbound destination the daemon contacted in the last 24 hours, with byte counts.
- **Content is never logged.** Prompts, outputs, KV, activations and tool arguments do not appear in logs, traces or metrics. Logs carry lengths and counts. Trace spans carry token counts and timings, never text (RFC-0053).
- **Diagnostic bundles** (`syntropy doctor --bundle`) exclude content, keys, tokens and API keys. The user reviews the bundle's file list before sharing it. Nothing is uploaded automatically.
- **Data at rest.** Two kinds of stored data hold conversation content.

  | Data | Where it is stored | Default | Who can read it |
  |---|---|---|---|
  | Persisted session KV and recurrent state (RFC-0022 §8) | each stage holder saves its own part on its own node, for RFC-0022 §8.5's retention | autosave on (RFC-0022) | from M2, every holder stores only ciphertext, under a key that only the session's ingress can unwrap (below) |
  | Stored Responses records: the transcript token lists behind `previous_response_id` (RFC-0050 §9.2) | **the ingress node only**. There is no copy on holders or any other member | `api.responses.store` (RFC-0050) | the ingress node |

  - **Per-session keys** (from M2, when session blobs first land on other nodes). Each stage saves its own layers' KV on its own node, so the key scheme lets a holder encrypt what it cannot later read:
    1. The session's ingress generates an X25519 key pair per session. The private half stays in its state directory, protected as §4 protects node keys, including OS keystore wrapping where available. The public half travels to every stage in `SequenceOpen::session_pk` (RFC-0032 §3.6).
    2. A holder that saves its part draws a fresh content key, seals the blob with it (AES-256-GCM or ChaCha20-Poly1305, §3), wraps the content key to `session_pk` with an ephemeral X25519 exchange, and writes only the ciphertext and the wrapped key (RFC-0022 owns the blob layout). It then drops the content key.
    3. For a restore, the holder sends the wrapped key to the ingress over the authenticated control connection. The ingress unwraps it and returns the content key for that blob only. The holder decrypts into its runner's memory and drops the key when the restore ends.

    A holder therefore never keeps a readable copy of the conversation at rest; it sees plaintext only while it processes, as D4 intends. Autosave stays on, so fast agent resumes keep working. The ingress's own stage, if it hosts one, follows the same scheme, so every saved blob is ciphertext and the session private keys are the only secret to protect.
  - **When the ingress is lost**, its key goes with it, and the holders' ciphertext can no longer be restored. A resume then re-prefills, and `previous_response_id` returns a clear error so the client resends its input (RFC-0050). This is accepted: the ingress is almost always the client's own machine, so when it dies the client usually dies with it (RFC-0033 §9.2).
  - **No dual-write.** Stored Responses records are never copied to session holders. A copy there would put plaintext on other members' disks, and under the per-session key it could not be read after the ingress is lost anyway. Revisit only for holders in the user's own cluster, and only if the key can be unwrapped by that cluster's other keyholders.
  - **Purge.** `syntropy sessions purge` deletes a node's sessions and asks every holder of their blobs to delete them too (RFC-0051, RFC-0022). Revoking a member does not reach into its disk, but what it holds is ciphertext it cannot read.
  - Full-disk encryption (FileVault, BitLocker, LUKS) is recommended by `doctor` when it is off.
  - The M4 option `sessions.encrypt_at_rest` is withdrawn: from M2 every saved blob is ciphertext. M4 verifies the session private keys' keystore wrapping on every Tier 1 OS instead (DIST-E17-T09).
  - **Revisit after M2's disk data:** turning `store` and `autosave` off by default on nodes that are not the session's ingress.
- **Members see what they process.** A stage host sees the activations of the sequences it computes. An ingress-only node sees only its own users' requests (§5.2). The dashboard's per-instance view lists which nodes handle a model's activations.

### 14. Configuration

| Key | Scope | Default | Notes |
|---|---|---|---|
| `security.keystore` | node | `auto` | `auto` \| `file` |
| `security.cert_lifetime` | cluster | 180 days | §5.3 |
| `security.clock_skew` | cluster | 10 min | §5.4 |
| `security.rotation_window` | cluster | 7 days | §8.2 |
| `security.trusted_fabric` | node | `[]` | interfaces or CIDRs (§9.2) |
| `rdma.encrypt_untrusted` | node | `false` | M4 (§9.3) |
| `invite.require_confirm` | cluster | `false` | §7 |
| `integrity.verify_on_load` | node | `sample` | `sample` \| `full` \| `off` |
| `models.signatures` | node | `verify_if_present` | `off` \| `verify_if_present` \| `require` |
| `models.trusted_signers` | node | `[]` | identities or keys for `require` |

### 15. Error codes

`Domain::Security`, range 0x0500–0x05FF.

| Code | Name | Class |
|---|---|---|
| 0x0501 | `CertInvalidSignature` | `Auth` |
| 0x0502 | `CertExpired` | `Auth` |
| 0x0503 | `CertRevoked` | `Auth` |
| 0x0504 | `CertWrongCluster` | `Auth` |
| 0x0505 | `RoleInsufficient` | `Auth` |
| 0x0506 | `CredentialExpired` (data plane) | `Auth` |
| 0x0507 | `KeyholderRequired` | `Auth` |
| 0x0508 | `RotationWindowClosed` | `Auth` |
| 0x0509–0x050B | reserved (API authentication codes are RFC-0050's) | — |
| 0x050C | `ModuleSignatureInvalid` | `Integrity` |
| 0x050D | `UpdateSignatureInvalid` | `Integrity` |
| 0x050E | `ModelSignatureInvalid` / `ModelUnsigned` (when required) | `Integrity` |
| 0x050F | `SourceDistrusted` | `Integrity` |
| 0x0510 | `UntrustedFabric` (RDMA requested on an unmarked interface) | `Unsupported` |
| 0x0511 | `AuthorityInvalid` (an event lacks the signature §8.3 requires) | `Auth` |
| 0x0512 | `RotationChainInvalid` | `Auth` |
| 0x0513 | `NotCoordinatorEligible` | `Auth` |

## Performance model and budgets

Security sits off the per-token critical path everywhere except transport encryption, which RFC-0031 already includes in its measured α and β. All figures are **[estimate]**.

| Cost | When | Size | Budget impact |
|---|---|---|---|
| TLS 1.3 handshake, with Ed25519 signatures and verifications | once per connection (per node pair per ALPN; per runner pair per epoch) | tens of µs of CPU plus 1 RTT (QUIC) [estimate: order of magnitude for one Ed25519 sign plus two verifies and one key exchange on a desktop core] | none on decode |
| Certificate or credential verification | once per connection or channel | one Ed25519 verify, ≈ 50 µs | none on decode |
| Per-frame AEAD on QUIC or TCP | every data-plane frame | ≥ 4 GB/s per core assumed for AES-GCM with hardware AES; the probe measures it (RFC-0031 §Performance) | inside measured α and β; ≈ 8 µs per 32 KiB frame per side |
| CRC32C | every frame | ≈ 4 µs per 32 KiB per side | RFC-0031 Z budget |
| BLAKE3 chunk verification | every received chunk | multi-GB/s per core with SIMD [estimate]; parallel across chunks | does not limit 10 GbE fetches at 1.2 GB/s |
| Pack sample verification at load | `Loading` | 1 % of the shard plus canary extents | negligible next to the load itself (SC-23) |
| Optional RDMA AEAD (§9.3) | every RDMA frame on untrusted interfaces | ≈ 8 µs per 32 KiB per side | +16 µs per hop, a large fraction of Z on TB5. This is why it is opt-in and costed by the probe |
| Revocation propagation | on `NodeRevoked` | one log append plus one beat | connections closed within ≈ 1 s on LAN [target] |

## Alternatives considered

- **A shared-secret cluster key with HMAC challenge-response.** This is the surveyed minimum bar. It is simple, but any member can admit anyone, and revoking anything requires rotating the secret on every node. Rejected in favour of a signing key with keyholders (ADR-0313).
- **X.509 certificates and a full PKI.** Standard tooling, but a large parsing surface (A4), and chain and extension semantics that this design does not need. A fixed 192-byte certificate does the job. Rejected.
- **Threshold signatures for the cluster key**, so that no single keyholder can mint certificates. This is the right shape for larger, multi-owner clusters, but it is complex, and home clusters have one owner. Deferred.
- **WireGuard or an OS-level VPN between members.** It needs root and OS integration on every platform, which defeats one-command install. Syntropy runs happily *over* such a VPN (Tailscale) but does not depend on one. Rejected as the mechanism.
- **Per-frame RDMA encryption always.** It costs about 16 µs per hop on links whose whole point is 10–25 µs hops. Physical point-to-point cables between admitted nodes do not need it. Rejected as the default; available as an opt-in.
- **Accepting plaintext API traffic on private LANs by default.** Keys and prompts could be sniffed by any device on the same Wi-Fi or switch (A1), and RFC-0001 §12 requires TLS. Rejected: plaintext on a LAN address needs the explicit `api.allow_plaintext_lan` opt-in, flagged permanently, and only encrypted overlay interfaces may skip TLS (§11, R3).
- **A custom release manifest with its own roots**, used in place of TUF. It was this RFC's first draft. ADR-0508's TUF repository already gives freeze, rollback and mix-and-match protection, so the key hierarchy is defined as TUF roles plus one release key for the cases TUF cannot reach (§12).
- **Opt-out telemetry.** Contradicts RFC-0001 §12 rule 4 and the product's local-first promise. Rejected.

## Testing and validation

| Claim | Verification |
|---|---|
| Certificate, credential, token and beacon parsers are safe | fuzz targets (RFC-0060); golden-vector tests for every signed structure; tests for context-string separation, where a signature from one context never verifies in another |
| No traffic before mutual authentication | integration tests per ALPN: a peer without a certificate, with an expired, revoked or wrong-cluster certificate, or with a valid certificate but insufficient role, gets `CredentialInvalid` before any application data is processed (asserted by an instrumented acceptor) |
| Runners never hold the node key | a process-memory test in CI: the runner's address space contains no copy of the node seed (a scan for the seed bytes in a test build) |
| Revocation takes effect within one beat and survives heals | simulated fabric: revoke during load and assert closed connections; split, revoke on one side, heal, and assert the revoked node is refused everywhere (the deny list); a re-paired node at `g + 1` is always accepted; a certificate whose `CertIssued` was lost is still covered by the generation revocation |
| Authority cannot be forged | simulator with a malicious coordinator-eligible member that appends forged `NodeRevoked`, `KeyholderChanged`, `ClusterConfigChanged { lan_admission = open }` and `ApiKeyCreated`: every honest node refuses each entry in `validate()`, writes nothing to its deny list, and elects a new coordinator; an ingress-only node's claims and appends are ignored; a claim at counter + 2 is refused |
| A compromised coordinator cannot forge through snapshots or censor revocations | simulator, replaying the verifier's scenario: a compromised `COORDINATOR` node wins an election and pushes a crafted snapshot with a KEYHOLDER-scope invite it signed itself; the keyholder refuses the snapshot and issues nothing. The same node omits a `Revoke` aimed at itself; gossip delivers the signed revocation and its followers stop following it within one beat. It omits `InviteConsumed`; the second use of the token is refused. A pairing join without the operator's grant never holds `COORDINATOR` |
| Chunk tables cannot be poisoned by one member | simulator with a member that proposes wrong `ChunkId`s: with no evidence, the batch stays provisional and no peer packs from it; a single witness signature fails `validate()`; with origin chunk metadata, a peer rejects a tampered chunk even though the table came from the attacker |
| Control frames cannot reuse nonces or be forged | golden vectors for sealed data and control frames, showing distinct nonce spaces and `ctl_seq` in `step`; a fault test that injects unsealed `Credit` and `Close` frames on an encrypted channel, which are dropped; a re-key with frames outstanding, whose resends still open under the previous key |
| Rotation | simulated: revoke a keyholder; assert new certificates for all members; assert old-key certificates are accepted only for existing members during the window and refused after it; **rotate on one side of a split, then heal**: the sides reconnect through the rotation chain and converge on the winner's key; a branch minted by a revoked keyholder is refused |
| Invite properties | property tests on `decide()`: single use, scopes become roles, expiry; MITM simulation through a hostile relay fails at the cluster-tag check; rate limits |
| The pairing SAS cannot be ground | simulator: an attacker relay between a new node and a keyholder, with an unlimited hash budget, gets both codes to match at a rate no better than 10⁻⁶ per attempt; the same attack against the first draft's exchange, kept as a negative control, succeeds; a `PairReveal` whose `nonce_c` does not match the commitment ends the pairing |
| Recovery restores authority and always rotates (§5.5) | simulator: the only keyholder is destroyed; `cluster recover` on a member with the right 12 words yields a new cluster key, a KEYHOLDER certificate on that member, re-issued certificates for every member, and the old keyholder revoked; a wrong code changes nothing; a used code cannot recover twice; a recovery statement whose `sig_recovery` does not verify is refused by every node |
| Last-keyholder guard (§5.5) | `uninstall`, `cluster leave`, `cluster remove` and `identity reset` on the last keyholder of a two-node cluster refuse without `--force`; with another keyholder present they proceed; `leave` completes with no coordinator reachable |
| Cluster-scope configuration authority (§8.3) | simulator: a `SetClusterConfig` from an ingress-only member is refused in `validate()` on every node; a stage host's patch to `cluster_aliases`, `network_scope`, `allow_approximate`, `update.*` or a `policy.*` exactness or pin field without a keyholder signature is refused; the same patch with one is applied |
| Holders keep only ciphertext (§13) | a session saved from a remote ingress leaves no plaintext canary on the holder's disk; the holder restores only after the ingress releases the key; with the ingress gone, the resume re-prefills and the holder's blob stays unreadable |
| `trusted_fabric` gate (§9.2) | on a rig, the assertion is refused on a Wi-Fi interface, on an interface with a default route, and on one whose neighbour table holds a non-member; `--force` with the typed name accepts it and `doctor` flags it |
| Encryption on every IP path | CI packet capture on QUIC and TCP data channels with a known activation pattern; assert the pattern is absent on the wire |
| Plaintext only where allowed | unit tests of the edge policy: RDMA refused on unmarked interfaces; shared memory refused across OS users |
| API requirements R1–R5 (§11) | RFC-0050's API tests cover the mechanism; this RFC adds a packet-capture check that no key or prompt crosses a non-loopback, non-overlay link in plaintext under the settled default |
| Supply chain | RFC-0052's TUF client tests (expired timestamp, rolled-back snapshot, mix-and-match targets, root rotation) plus this RFC's: a module with the right TUF hash but the wrong build ID is refused at load; a release-key signature from a key revoked in `signing-keys.json` is refused; a nightly-key signature on a stable artefact is refused |
| Privacy | a full workload in a network namespace with an egress recorder; assert no destinations beyond the §13 table, and none with `--offline`; log and trace scans for prompt canary strings |
| Security review | an adversarial review of the threat model and implementation by an independent review agent before 1.0 (M4), with findings approved by a maintainer as security-critical changes (ADR-0021); and the external audit below |

**Revision note, 2026-10-22 (DIST-E5-T14): the grinding-relay check.** Until RFC-0060's simulator exists, the check of "The pairing SAS cannot be ground" runs in a pure model of the exchange (`crates/cluster/tests/pairing/`, RFC-0030 §6.2's note of the same date). Its seeds are fixed. It runs 20 first-draft trials, each of which must match within 10⁷ hashes. Against the committed exchange it runs 10⁷ attempts with an honest reveal, of which at most 25 may match (Poisson mean 10, P(X > 25) < 10⁻⁴), and 20 attempts whose attacker grinds a cheating reveal, each of which the commitment check must deny. The default tests run a seeded prefix of each; `cargo xtask test pairing-grind` runs the full counts with release optimisation. The attacker's strategies are a fixed set, not every possible attacker. With an honest reveal it has no grinding target in either order, so its budget of 10⁷ hashes per nonce goes unused. The cheating reveal uses the budget, and only the commitment check stops it. The model bounds one attempt and leaves retries to the rate limits of RFC-0030 §6.2 and §6.7. DIST-E5-T03 repeats the check through the real exchange in the simulator.

**External audit before 1.0.** The scope is:
1. join, admission and invites: the `syntropy/join/1` ALPN, pairing with its commit-then-reveal and SAS, pairing windows, invite tokens, certificates and the recovery code (RFC-0030 §6, §5 and §7 here);
2. transport and data-plane credentials: TLS on every IP path, `DataPlaneCredential`s, RDMA and shared-memory pairing, and the optional RDMA AEAD (§6, §9);
3. the update client and its use of TUF, including `signing-keys.json` and the release-key digest statement (§12, RFC-0052, ADR-0508);
4. the local API CA (ADR-0505, RFC-0050 §11);
5. API authentication: keys, scopes, the loopback rules and the bind policy (§11, RFC-0050 §11).

It takes place after M3's feature freeze, so its findings land before the M4 release candidates. Scheduling it is QUAL's task; this list is its scope.

## Rollout / milestones

| Milestone | Lands |
|---|---|
| M0 | node key storage; content-free logging; no telemetry (the loopback API protections are RFC-0050's, M0) |
| M0 (late) | hardware tokens and KMS keys provisioned and token custodians recorded (**Maintainer action**, §12.1), so the signing-toolchain check and a dry-run ceremony can run before the first M1 release candidate |
| M1 | signing key hierarchy in place for the first public release: TUF root and targets ceremonies on hardware tokens (**Maintainer action**), KMS-held snapshot, timestamp, release and nightly keys, the release-key digest statement (§12; RFC-0062 needs it before M1); header parsing of newly added models in a resource-limited child process with no network (§10.4) |
| M2 | cluster key, certificates, keyholders, LAN pairing with commit-then-reveal and number matching (ADR-0302); blinded beacons on home networks only (RFC-0030 §4.8); the pairing-approval grant and keyholder offer (§5.2); the recovery code with its recovery rotation, the first use of §8.2's rotation statement, plus the last-keyholder guard and re-founding (§5.5); cluster-scope configuration authority (§8.3); per-session keys for holders (§13); mutual authentication on every ALPN; data-plane credentials; TLS on all IP paths; Thunderbolt 5 RDMA plaintext policy; shared-memory same-user rule; generation-based revocation with signed deny lists; authority validation of every log event on every node (§8.3); keyholder re-submission of revocations and certificates after a heal (with RFC-0032's heal); chunk verification and suspect sources |
| M3 | invite-token security (§7) with admission notifications and the own-device question; the second keyholder offer and status line (§5.2); certificate renewal, including renewal after expiry (§5.3); cluster key rotation with rotation chains; cluster-wide API keys (§8.4); `trusted_fabric` with its mechanical gate, and RoCEv2; key-rotation drills; the external audit booked |
| M4 | load-time module re-verification (§12.2) on every Tier 1 OS; OMS model-signature verification; optional RDMA AEAD; OS sandbox profiles for header parsing; session private keys verified as keystore-wrapped on every Tier 1 OS; security review; external audit findings closed |

## Open questions

1. **Default keyholders.** *Answered.* Keyholders only coordinate by default. A wired, mains-powered desktop is offered the `COORDINATOR` grant and the keyholder role, both unticked, at its pairing approval, and the keyholder role once more after 10 minutes [target] without the only keyholder; laptops and battery-powered nodes are never offered it (§5.2). A recovery code, a last-keyholder guard and re-founding cover the loss of the only keyholder (§5.5). Revisited at the M2 exit with the alpha data.
2. **Certificate lifetime.** *Answered (OQ-SEC-01).* 180 days stays. A certificate that expired less than 180 days ago may be renewed with proof of the node key and one keyholder approval showing the node's name and fingerprint; never silently, and never for a revoked generation (§5.3).
3. **Sandboxed header parsing.** *Answered (OQ-SEC-02).* From M1, newly added models' headers are parsed in a short-lived child with resource limits and no network; OS sandbox profiles stay at M4 (§10.4).
4. **Relay metadata.** *Answered.* The built-in relay set is a hosted shared relay service delivered as the signed `relays.json` target; `custom` and `disabled` remain, and weight chunks never travel through a relay (RFC-0030 §6.4, §13 here). Relays now also start when a member is allowed off-LAN reach at pairing.
7. **Signing-key custody before M1.** *Answered.* An HSM-backed cloud KMS for the four online keys, hardware tokens for the offline roles, asynchronous signing through signing-event pull requests, maintainers or recorded delegates as custodians with root tokens on three different custodians, and a digest statement as what the release key signs (§12.1). RFC-0062 open question 2 has the same answer.
8. **Fingerprint word list.** *Answered (OQ-SEC-05).* The BIP-39 English list (MIT) after an offensive-word review, displayed as `word-word-…` (§6.5).
5. **OMS verification in Rust.** Whether a maintained Rust verifier for the OpenSSF Model Signing bundle format exists by M4, or whether Syntropy implements bundle verification over existing Sigstore primitives.
6. **`trusted_fabric` UX.** *Answered (OQ-SEC-04).* A mechanical gate: wired, no default route, and a neighbour table of members only, or `--force` with the interface name typed back; `doctor` flags it and every probe re-checks it (§9.2).

## Proposed amendments to RFC-0001

1. **IPC additions (§3.5).**
   - A D → R message `PeerRevoked { node }`, so runners close channels to a revoked node (§6.1).
   - `Init`'s `credential: DataPlaneCredential` is accompanied by the runner's key seed, generated by the daemon (§4). The IPC form is `DataPlaneGrant { credential, seed }`.

   Both are additions under the extension rules, listed because §3.5 enumerates the message set.
2. **TLS for non-localhost binds (§12, Z3).** Add: "…or a bind restricted to an interface that is encrypted end to end, such as a WireGuard-based overlay". This keeps RFC-0001's intent and makes the phone-over-tailnet case work without certificates. It is the basis for this RFC's R3 and for the change it asks of ADR-0505 (§11).
3. **Glossary consequence.** The glossary says RFC-0034 decides whether the cluster key is a shared secret or a signing key. It is an **Ed25519 signing key held by keyholders**, and members hold admission certificates (ADR-0313). RFC-0001 §12's zone table already describes it this way. No text change is needed there, only in the glossary.
