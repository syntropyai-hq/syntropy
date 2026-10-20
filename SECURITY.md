# Security

## Reporting a vulnerability

Report anything that shouldn't be public until it is fixed **privately**, through GitHub's private vulnerability reporting: the repository's **Security** tab, then **Report a vulnerability** ([direct link](https://github.com/syntropyai-hq/syntropy/security/advisories/new)). A `security@` address will be listed here before the first release.

Include what you found, how to reproduce it (the commit or version, the configuration keys involved, the request or the frames sent), and what an attacker gains. Reports are triaged by the maintainers. Any work to reproduce or fix a report happens in an isolated environment with no secrets, no signing access and no way to publish.

Hardening ideas and findings that are safe to discuss in public, including design weaknesses in the plan, are welcome as an [issue](https://github.com/syntropyai-hq/syntropy/issues) or a pull request.

### What happens next

| Step | Target |
|---|---|
| Acknowledge the report | within 3 business days |
| Assess it and agree a severity | within 10 business days |
| Fix or mitigate a critical issue | within 30 days |
| Disclose, in coordination with you | within 90 days |

Critical covers remote code execution, a bypass of cluster admission and exposure of a signing key. Fixes are announced through [GitHub security advisories](https://github.com/syntropyai-hq/syntropy/security/advisories), which can assign CVE IDs, and in the release notes. You are credited in the advisory unless you ask not to be. Every security fix gets a regression test.

There is no bug bounty before 1.0.

## Supported versions

Nothing has been released yet. The first public release is planned at milestone M1a, as a preview. This section will list the supported releases from then on.

The policy covers the daemon, the CLI, the update client, the install scripts and the published backend modules. It also covers the design in [`plan/`](plan/README.md): a flaw found in a specification now is cheaper than one found in code later.

## What Syntropy will expose

The threat model and every mechanism below are specified in [RFC-0034](plan/rfcs/RFC-0034-security-and-trust.md). None of it is built yet.

A cluster is the set of machines one user has admitted. Every admitted machine is trusted with the weights, prompts and activations it handles. Public swarms of untrusted machines are out of scope, so admission is the security boundary, and everything after it is authenticated and encrypted by default.

- **LAN discovery.** Nodes announce themselves over DNS-SD on mDNS and signed multicast and broadcast beacons. A beacon grants nothing; it only prompts a connection attempt, which is then authenticated. Once a node has joined a cluster, its beacons and mDNS records are blinded with a key the cluster's members share, so only members recognise them.
- **Pairing.** A new machine on the LAN is admitted only after a person approves it on a machine that holds the cluster key, comparing a six-digit code shown on both screens. The pairing protocol commits to its nonces before revealing them, so an attacker in the middle gets one guess per pairing attempt.
- **Invite links.** An invite token carries a 128-bit secret, is single-use by default, expires after 24 hours by default (7 days at most), carries scopes and can be revoked before use. Only a hash of the secret is logged, the joiner authenticates the cluster even through a relay, and every join by invite raises a notification with a one-click revoke.
- **Cluster traffic.** Every connection is TLS 1.3 over QUIC or TCP with both ends' Ed25519 identities authenticated, followed by an exchange of admission certificates signed by the cluster key. Every node validates every entry in the cluster's event log, and admissions, revocations and key changes must carry a keyholder's signature, so the elected coordinator cannot forge membership. Revocation closes every connection to the revoked machine.
- **Encryption.** Every IP path is encrypted. The only plaintext paths are shared memory between one user's processes on one host, RDMA over a direct Thunderbolt 5 cable between admitted machines, and RoCEv2 on interfaces the operator marks `trusted_fabric`. The dashboard shows each plaintext edge.
- **The API.** On loopback, inference and read-only routes need no key by default. Admin, session and keyholder actions need the local socket or an API key, and loopback listeners reject foreign `Host` headers and cross-origin state-changing requests. Any other bind needs an API key and TLS. A bind to an encrypted overlay interface such as a tailnet may omit TLS but still needs the key. Plain HTTP on a LAN address requires an explicit `api.allow_plaintext_lan = true`, which the dashboard flags permanently. Keys are stored hashed and compared in constant time. Debug endpoints answer only on loopback, are off by default, and no key enables them remotely.
- **Models.** Every downloaded byte is checked against BLAKE3 content addresses before use, and publisher signatures in the OpenSSF Model Signing format are checked when present. Model parsers are fuzzed, and no model file ever runs code.
- **Updates and modules.** Releases are signed through a TUF repository: an offline root key held 2-of-3, offline targets keys, and online snapshot and timestamp keys. Backend modules are verified by hash, signature and build ID before they load, and an update is checked by a self-test before it is switched in.
- **Privacy.** No telemetry leaves the machine by default. Prompts, outputs, KV and activations never appear in logs, traces or metrics. The outbound connections Syntropy makes (the model hub, relays for invite-joined members, update checks) are caused by the user's own actions, `--offline` turns them off, and `syntropy doctor --network` lists every destination contacted in the last 24 hours.

Syntropy has not had an outside security audit. One is planned before 1.0.
