# ADR-0316: The software signing key hierarchy: TUF roles, an online release key for detached signatures, and platform identities

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the distributed runtime area)
- Related: ADR-0014, ADR-0021, ADR-0507, ADR-0508, ADR-0623, RFC-0001 §12 (rule 3), RFC-0034 §12, RFC-0052 §7.2, RFC-0062 §7

## Context

RFC-0001 §12 rule 3 requires backend modules and updates to be verified by signature before use. Three neighbours each need part of the key hierarchy:
- ADR-0508 and RFC-0052 chose a TUF repository, read with the `tough` crate, for updates. Its root is 2 of 3 offline keys, its targets key is offline, and its snapshot and timestamp are online.
- ADR-0507 makes backend modules TUF targets.
- RFC-0062 §7 signs Linux tarballs, packages, checksum files and the `ConformanceManifest` with a "Syntropy release key" and a "nightly key" held in a cloud KMS. It also uses Apple and Windows platform identities.

All three defer to RFC-0034 for *which keys exist*, their custody, rotation and revocation.

Some verifications happen where TUF metadata is not yet reachable or does not apply:
- the install script before the binary runs;
- `setup`'s first self-check;
- the daemon checking the `ConformanceManifest` at start;
- users verifying a tarball by hand.

A first draft of this ADR defined its own release manifest and roots. That duplicated what TUF already provides: protection against freeze, rollback and mix-and-match attacks.

## Options considered

| Option | For | Against |
|---|---|---|
| A. A custom manifest with its own offline roots (first draft) | Simple | Duplicates TUF; two trust anchors to rotate |
| B. TUF only | One anchor | Install script, ConformanceManifest and manual checks cannot use TUF comfortably |
| **C. TUF roles as the anchor for updates and modules; one online Ed25519 release key (and a nightly key) for detached signatures where TUF does not reach, its public keys distributed as a TUF target and embedded per release; platform identities on top; a separate hierarchy from node and cluster keys** | One root of trust per purpose; every verification path covered; compromise responses defined per key | More keys to hold; the token custodians must be recorded before M1 |

## Decision

**Option C** (RFC-0034 §12).

1. **Keys, custody and lifetimes.**

   | Key | Custody | Threshold | Lifetime |
   |---|---|---|---|
   | TUF root | offline hardware tokens, one each for three custodians, stored apart | 2 of 3 | metadata expires yearly; chained rotation |
   | TUF targets | offline hardware tokens, one each for two custodians | 1 of 2 | 90-day expiry; re-signed per release |
   | TUF snapshot | HSM-backed key in a cloud KMS | — | 7-day expiry |
   | TUF timestamp | HSM-backed key in a cloud KMS | — | 1-day expiry, re-signed daily |
   | Release key (Ed25519) | HSM-backed key in a cloud KMS; usable only by the `release-signing` environment, whose two approvers are maintainers or recorded delegates | — | rotated yearly |
   | Nightly key (Ed25519) | KMS; nightly channel only | — | rotated yearly |
   | Linux repository keys | KMS | — | per distribution convention |
   | Apple Developer ID and notarisation credential; Windows Authenticode via a cloud signing service | as RFC-0062 §7.2 | — | as issued |
   | Keyless build provenance | CI workload identity | — | per build |

2. **Distribution of the release keys.**
   - The current and next release and nightly public keys are published as a TUF target, `signing-keys.json`, which also carries revocations.
   - They are also embedded in each release.
   - A detached signature is trusted only from an embedded or TUF-verified key.
3. **Verification.**
   - The TUF client verifies updates and modules (ADR-0508).
   - Immediately before `dlopen`, the module loader re-checks the module's hash and build ID, and loads only by absolute path from the user-owned modules directory.
   - The install script and `setup` verify detached release-key signatures before any TUF metadata exists locally.
   - Platform signing applies on top.
4. **Compromise responses.** These are defined per key in RFC-0034 §12.3:
   - online roles are rotated by the root threshold;
   - a lost targets key leads to a targets rotation and a full re-sign;
   - a release key is revoked through `signing-keys.json`;
   - the loss of one root token is absorbed by the other two.
5. **Separation.** No software-signing key can admit a node or sign an admission certificate, and no cluster or node key can sign software (RFC-0034 §4).
6. **Who holds and uses the keys** (ADR-0021). Every custodian is a maintainer or a recorded delegate, and each delegation is recorded in the repository and names the tokens and actions it covers. Key generation, the root and targets ceremonies, the scheduled re-signing and any rotation that needs an offline token are human actions done by the custodians. Agents build and test the ceremony runbooks, the KMS integration and the expiry monitor; no agent holds, reaches or handles an offline key.

**Revised 2026-10-19:**
1. **Nightly channel**. A delegated TUF targets role, `nightly`, restricted to `nightly/*` paths, is held online in the KMS `nightly-signing` environment. It is listed in, and rotated by, the offline top-level targets role. Nightly builds self-update through TUF without the offline keys. One nightly key serves both as this delegated role's key and for detached nightly signatures. A context string keeps the two kinds of signature apart (RFC-0034 §12.1).
2. **Targets expiry.** Top-level targets metadata is re-signed at a scheduled ceremony (**Maintainer action**) at least 30 days before its 90-day expiry, whether or not a release ships. Expiry is monitored.
3. **Install trust roots**. `setup`'s self-check guards against corrupted mirrors, not a compromised release host. The trust root for each install path is stated in RFC-0034 §12.2. The release-key fingerprint, in hex and six-word form, is published away from the release host: in the source repository, every release's notes, and the public transparency log that records build provenance. The install script's `--verify-key` checks against it.

**Revised 2026-10-20:** the key stores, the tokens, the signing method and the signed statement are now fixed. RFC-0034 §12.1 holds the detail.
1. **Online keys in a cloud KMS that keeps Ed25519 keys in an HSM.** Snapshot and timestamp are ECDSA P-256; the release and nightly keys are Ed25519, so RFC-0034 §3's Ed25519 release key needs no algorithm change. CI reaches each key only by workload-identity role assumption pinned to its environment. Key policies deny deletion and policy changes outside a break-glass role held by two custodians, and the KMS audit log raises an alert on any signing call the expected workflows did not make.
2. **Offline roles on hardware tokens** that generate ECDSA P-256 keys on the token in a PIV applet and attest them, on firmware with no known side-channel flaw in its ECDSA implementation. Each token's attestation certificate is recorded. Three root tokens and two targets tokens are in use, with two blank spares. ECDSA P-256 is used because `tough` verifies `ecdsa-sha2-nistp256` and PIV ECDSA is the path TUF signing tools support best.
3. **Asynchronous signing.** Each custodian signs root or targets metadata on their own machine through a signing-event pull request, so custodians never have to meet and a ceremony needs no shared location.
4. **Cadence.** Root yearly, at least 60 days before expiry, and on any custodian change; targets per release and every 60 days, which keeps at least 30 days in hand on the 90-day expiry; snapshot per publish and timestamp daily, both automated; release and nightly keys yearly, with the next public key published one release ahead.
5. **The release key signs a digest statement, never the file:** `"syntropy-release-v1" ‖ SHA-256(file)`. A KMS signing call accepts a bounded message, not a multi-gigabyte artefact, and one statement form means the install script, `setup` and users verify the same bytes.
6. **Custody rule.** Root tokens sit with three different custodians, so no single person can rotate root. Any two root custodians can rotate it together, which keeps re-signing and revocation from waiting on one person.

## Consequences

- Positive:
  - Every verification path has a defined anchor.
  - Update security inherits TUF's protections.
  - The pipeline (RFC-0062) and the client (RFC-0052) implement against one hierarchy.
- Negative:
  - Five offline hardware tokens, plus two spares, must be kept, and their custodians recorded before M1's first public release.
  - Any two of the three root custodians can rotate root without the third. That is the accepted cost of not waiting on one person.
  - Every ceremony needs its custodians available, so ceremony scheduling is on the release path.
  - Two online signing identities, release and nightly, live in the KMS.
- Work created:
  - RFC-0062: the KMS integration and ceremony runbooks;
  - RFC-0052: handling `signing-keys.json` in `setup`;
  - RFC-0060: tests for revoked-key and nightly-on-stable rejection.

## What would make us revisit

- Independent publishers ship their own modules, such as vendor providers. Add delegated TUF targets roles.
- Keyless signature verification becomes reliable offline. Retire the long-lived release key.
- A token custodian becomes unavailable for a ceremony in practice. Revisit thresholds or record a further delegate.
