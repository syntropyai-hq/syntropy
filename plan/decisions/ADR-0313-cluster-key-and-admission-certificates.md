# ADR-0313: The cluster key is an Ed25519 signing key held by keyholders; members hold fixed-format admission certificates

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the distributed runtime area)
- Related: ADR-0000 (D4), ADR-0009, ADR-0011, ADR-0302, ADR-0303, RFC-0001 §12, RFC-0030 §6, RFC-0034 §4–§8

## Context

The glossary leaves RFC-0034 to decide whether the cluster key is a shared secret or a signing key. RFC-0001 §12 requires:
- that every node hold an admission certificate signed by the cluster key;
- that admission be checked before any model traffic;
- that revocation go through `ClusterCommand::Revoke`.

There is no quorum (ADR-0009). Two-node home clusters must keep working when one node is off. Node identity is already an Ed25519 key, which is also the iroh endpoint key (ADR-0011), so TLS proves possession.

The surveyed minimum was an HMAC challenge-response over a shared cluster key, which did not encrypt traffic (`multi-device-inference.md` §1.4). A shared secret lets every member admit anyone. Revoking it means rotating the secret on every node, including the one being removed.

## Options considered

| Option | For | Against |
|---|---|---|
| A. Shared symmetric secret (HMAC) | Simplest; works offline | Every member can admit; revocation equals global rotation; leaking it is permanent admission |
| B. X.509 PKI (cluster CA, certificate chains) | Standard tools | Large parser surface; semantics the product does not need |
| **C. Ed25519 cluster key (its public half is the `ClusterId`) held by one or more keyholder nodes; 192-byte binary certificates binding node ID, roles, serial and validity; revocation by serial through the log plus a local deny list; automatic key rotation when a keyholder is revoked** | Admission is a deliberate keyholder act; ordinary members cannot mint members; revocation is per node; certificates verify without the log, which matters for split heals | Keyholders must be online to admit; rotation logic is needed |
| D. Threshold cluster key (k-of-n keyholders) | No single keyholder can mint | Complex; home clusters have one owner |

## Decision

**Option C** (RFC-0034 §4–§8).
1. **Bootstrap.** The bootstrap node generates the cluster key and is the first keyholder. Further keyholders are promoted explicitly. Promotion is offered when a second mains-powered desktop-class node pairs.
2. **Certificate format.** The certificate is fixed: 192 bytes, little-endian:

   | Field | Bytes |
   |---|---|
   | version | 1 |
   | flags | 1 |
   | roles | 2 |
   | reserved | 4 |
   | cluster_id | 32 |
   | node_id | 32 |
   | serial | 8 |
   | not_before | 8 |
   | not_after | 8 |
   | issuer | 32 |
   | Ed25519 signature over `syntropy-cert-v1` ‖ bytes 0..128 | 64 |

3. **Roles** are `MEMBER`, `STAGE_HOST`, `INGRESS` and `KEYHOLDER`.
4. **Lifetime** is 180 days by default. Renewal is automatic at half-life while the node is connected. Clock skew of 10 minutes is tolerated.
5. **Verification:**
   - the node ID equals the TLS-authenticated peer key;
   - the signature verifies under the current cluster key, or an old one within its rotation window for existing members only;
   - the validity window holds;
   - the serial is above the highest revoked serial for that node;
   - the roles suffice.
6. **Revocation.** `Revoke` records `NodeRevoked`. Every node also writes the revoked serial to a local deny list that outlives log branches. Keyholders re-submit revocations after a heal.
7. **Rotation.** Revoking a keyholder rotates the key:
   - the new key goes to the remaining keyholders over authenticated connections;
   - `ClusterKeyRotated` carries cross-signatures from both keys;
   - all members get re-issued certificates;
   - outstanding invites are revoked;
   - old certificates are honoured for existing members for 7 days.

**Revised 2026-10-19:**
1. **New role**. Role bit4 is `COORDINATOR`: eligible to coordinate, checked by voters and followers.
2. **Authority events.** Admissions, revocations, keyholder changes, key rotations, invites, pairing windows, security-class configuration and cluster-wide API keys must carry a keyholder signature. Every node verifies it in `validate()` (RFC-0034 §8.3), so a coordinator cannot forge them. Keyholder promotion needs an existing keyholder's signature, plus the promoted node's acceptance.
3. **Revocation by generation**. Certificates carry an admission generation in the former reserved field at offset 4. Revocation is a keyholder-signed `(node, generation)`, which invalidates every certificate of that node at or below it, whatever its serial. Deny-list entries come only from signed revocations.
4. **Rotation chains**. Rotations are signed statements kept outside the log. Certificates under any key linked to the current key by a verified chain are accepted, for existing members during the window. A rotation on one side of a split therefore still heals. Revoking a keyholder makes the revoking keyholder start the rotation; `decide()` does not.
5. **Cluster-wide API keys** are logged as keyholder-signed hashes with monotonic revocation (RFC-0034 §8.4).

**Revised 2026-10-19, second revision:**
1. **`COORDINATOR` default**. Keyholders hold `COORDINATOR` by default. Any other node gets it only by an explicit operator grant: the pairing-approval checkbox (unticked by default), `cluster grant-coordinator`, or an invite with scope bit4. The default two-node cluster therefore has one eligible node, the keyholder, which still elects. While the keyholder is away, nothing that needs a log append can happen, and instances wholly on the survivor keep serving (RFC-0034 §5.2).
2. **Snapshots carry their proof.** Every authority-derived field of the cluster state keeps its signed record, and a node re-verifies all of them before adopting a snapshot. No keyholder issues a certificate on unverified state, and a rendezvous keyholder checks the signed `InviteCreated` itself (RFC-0034 §8.3; RFC-0032 §6.7).
3. **Revocations spread by gossip** as well as through the log. Any node applies a verified revocation on receipt, and followers stop following a revoked coordinator (RFC-0034 §8.1).
4. **Invite consumption** is derived from signed approvals, so a coordinator that omits `InviteConsumed` cannot re-use a single-use token.
5. **Chunk tables** need origin, two-witness or keyholder evidence before commit (RFC-0034 §10.1).

**Revised 2026-10-20:**
1. **`COORDINATOR` stays keyholders-only.** Coordinator authority decides placement, and so which nodes see which prompts, and the pairing flow cannot tell a one-person home from a shared one. Keyholders-only is also the reversible default: loosening it later breaks no deployed cluster. At the pairing approval of a wired, mains-powered desktop, the approver sees the `COORDINATOR` grant and the keyholder offer, both unticked, each with one line naming what is lost without it. The default is revisited at the M2 exit with the alpha testers' reports: if most two-node clusters are a keyholder laptop plus an always-on desktop, the grant is pre-ticked for wired, mains-powered nodes in single-operator clusters. No survivor mode for now; reconsider at M3 if the reports show re-plan stalls hurting users.
2. **Keyholder offers** replace decision 1's "promotion is offered when a second mains-powered desktop-class node pairs": the offer at pairing approval above, once more after 10 minutes [target] without the only keyholder while an eligible node is present, and a permanent status line that never interrupts. Each keyholder is another copy of the cluster key, so the offer does not nag, and laptops and battery-powered nodes are never offered the role.
3. **Recovery when the only keyholder is lost.** Without it nobody could revoke a stolen device, and every member would drop out within one certificate lifetime. An optional offline recovery code, offered once at bootstrap, derives a recovery key whose public half the cluster records in a keyholder-signed event. `cluster recover` on any member signs a recovery rotation with it, so recovering always rotates the cluster key, and the lost keyholders are revoked. No cloud escrow. `uninstall`, `leave`, `remove` and `identity reset` refuse to drop the last keyholder without a promotion or `--force`, and a cluster with no code can be re-founded keeping every node's store and packs (RFC-0034 §5.5).
4. **Cluster-scope configuration.** Only `STAGE_HOST` members may change it. Aliases, `network_scope`, `allow_approximate`, `update.*`, and the exactness and pin fields of `policy.*` join the keys that need a keyholder signature, because other users rely on them and an alias resolves before canonical names (RFC-0034 §8.3; RFC-0054 lists the keys).

## Consequences

- Positive: least-privilege membership; per-node revocation; offline certificate verification; ingress-only nodes are enforceable cluster-wide.
- Negative:
  - With a single keyholder that is powered off, no new node can be admitted and, under the keyholders-only `COORDINATOR` default, nothing can re-plan. The pairing-approval offers and the second offer mitigate this; the recovery code covers a keyholder that never returns.
  - Whoever holds a cluster's recovery code can take its authority, so the code must be stored offline.
  - Members offline for a whole rotation window must re-pair.
- Work created:
  - RFC-0030: `RecordCert`, `PromoteKeyholder`, `AdmissionProof::Certificate`;
  - RFC-0051: keyholder and certificate views;
  - RFC-0060: rotation and revocation tests.

## What would make us revisit

- Multi-owner clusters, such as a shared lab, become a supported scenario. Adopt threshold keyholding.
- Field data shows admissions failing mostly because the only keyholder is asleep. Make the second keyholder automatic.
