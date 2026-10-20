# ADR-0505: API authentication: loopback open but guarded, keys and TLS everywhere else, debug endpoints local only
- Status: Accepted (2026-10-20); revised 2026-10-20 (see Revisions)
- Date: 2026-10-19
- Deciders: maintainers
- Related: ADR-0000 (D4), ADR-0009, ADR-0011; RFC-0001 §12; RFC-0034 §6.5, §11; RFC-0050 §11, §13.6; RFC-0051; RFC-0054 §7.1

## Context
Several requirements meet in the API layer.
- **Zero configuration.** Syntropy must work with no configuration on one machine (SC-24), and many users will also want to reach it from a phone, a laptop or another PC.
- **Fixed rules.** RFC-0001 §12 zone Z3 says non-loopback binds require API keys and TLS. RFC-0034 §11 delegates client-facing API authentication to the product area and lists the requirements it must meet:
  - R1: loopback protections against browser-borne attacks;
  - R2: keys on every non-loopback listener, stored hashed and compared in constant time;
  - R3: no key, prompt or output in plaintext on any network link, with a proposed exception for encrypted overlays;
  - R4: an `admin` scope for administrative operations, and keys on loopback advised on multi-user hosts;
  - R5: self-signed certificates bound to the node identity, with six-word fingerprints.
- **Browser-borne attacks.** A loopback HTTP server without authentication is reachable by any web page through DNS rebinding and cross-site requests. A surveyed single-GPU engine closed exactly this hole with Host and Origin checks (`tiered-memory-inference.md` §19).
- **Client compatibility.** Vendor SDKs present keys as `Authorization: Bearer` (OpenAI) or `x-api-key` (Anthropic). Many clients cannot easily be configured to trust a self-signed certificate.
- **Debug endpoints.** The quality RFCs need endpoints that must never be reachable from the network: teacher-forced parity taps, a telemetry drain and benchmark controls (RFC-0060 §1.3, RFC-0061 §10.2). They must work on the signed release artefacts.

## Options considered
| Option | For | Against |
|---|---|---|
| A. Plaintext HTTP with keys on private and overlay addresses; TLS only on public addresses | easiest for LAN clients | contradicts RFC-0001 §12 and RFC-0034 R3; keys sniffable on shared Wi-Fi; "private" is a guess about the network |
| B. Keys and TLS on every non-loopback bind, with a self-signed certificate per node that clients pin | meets RFC-0001 §12 and R1–R5 | pinning is per client and per tool; many clients have no pinning option |
| B2. Keys and TLS on every non-loopback bind, with a per-node, name-constrained local CA that `syntropy trust` installs on clients; TLS omitted on encrypted overlays; plain HTTP on LAN only by explicit, permanently flagged opt-in | meets RFC-0001 §12 as updated; one trust step per client covers every certificate the node will ever reissue; the overlay case needs no certificates | a CA in the trust store is a bigger trust grant than one pinned certificate, so name constraints are mandatory; phones and runtimes with their own CA bundles still need manual steps |
| C. Mutual TLS for API clients | strong | vendor SDKs and most clients cannot present client certificates |
| D. Debug endpoints behind an `admin` key | one mechanism | a stolen admin key would expose logit taps and unbounded generation remotely; scope checks are the wrong tool for test-only surfaces |

## Decision
Option B2, as nine numbered items. It meets RFC-0034 §11's requirement R3, including its exception for encrypted overlays.

1. **Loopback** is open by default for `inference` and the read-only routes only (revised 2026-10-20; see Revisions). `admin`, `sessions`, stored-response retrieval and every action that needs a keyholder signature require the peer-credential-checked local socket or a key, even on loopback. The dashboard on loopback gets its `admin` session from a one-time login URL that `syntropy dashboard` opens (item 8). `api.cors_origins` applies to `inference` routes only and never opens `admin`. Every loopback listener keeps three protections (R1). The strict Host list applies to every loopback listener, with or without keys, because DNS rebinding is the threat there. Non-loopback listeners accept exactly the names and addresses on their local CA certificate:
   - a Host check: 421 unless the name is loopback, a bound address or in `api.server_names`;
   - an Origin check on state-changing requests: CORS is off unless `api.cors_origins` allows it;
   - CSRF tokens in the dashboard.
   The local socket checks OS peer credentials. `api.auth.mode = "always"` requires keys on loopback too, and `doctor` recommends it on hosts with several interactive users (R4).
2. **Every non-loopback bind requires an API key and TLS** (RFC-0001 §12, R2, R3). There are two exceptions, for TLS only:
   - binds restricted to encrypted overlay interfaces (RFC-0030 §7.2 `Overlay`) may omit TLS;
   - plain HTTP on a LAN address requires `api.allow_plaintext_lan = true`, which the dashboard and `doctor` flag permanently as insecure.
   The daemon refuses to start a non-loopback bind without at least one key, and the error names the command. Wildcard binds count as LAN binds.
   - *Revised 2026-10-20.* The remote-client path is documented in this order: an overlay bind first (key required, no TLS), then `syntropy trust` with its environment-variable output. From M2, LAN pairing offers a "client only" role (ingress-only membership), so a desktop that runs Syntropy calls its own loopback and needs neither a key nor a CA. Any device its approver marks as not their own, whether it arrives by invite or by LAN pairing, defaults to that role.
3. **Keys.**
   - Format: `syk_` followed by 43 base64url characters (32 random bytes), shown once.
   - Storage: BLAKE3 hashes in `api-keys.toml` (mode 0600), compared in constant time.
   - Presentation: `Authorization: Bearer` or `x-api-key`, so both vendor SDKs work.
   - Hot-reloadable.
4. **Scopes**: `inference`, `sessions`, `admin` and `dashboard`, with the route mapping in RFC-0050 §11.3. Every administrative or state-changing operation requires `admin` (R4).
5. **Failures.** 401 on a bad key. Back-off is keyed by (peer address, presented key ID: the first 8 characters after `syk_`, RFC-0050 §11.3): more than 10 failures per minute for one pair give 429 for 5 minutes. The local socket and loopback are never locked out address-wide. A listener that does not require keys ignores presented credentials and never counts them, so SDKs that insist on a key work with a dummy value. An optional per-key rate limit is available.
6. **TLS sources.**
   - **The default is a per-node local CA:**
     - an ECDSA P-256 key, bound to the node identity by a node-key signature, its fingerprint shown as six words (RFC-0034 §6.5, R5);
     - critical X.509 name constraints limit it to the node's own DNS names and to the subnets the node serves when the CA is issued (its own /24s and /64s, and its overlay /32), so a leaked CA key can impersonate neither public sites nor other private hosts (revised 2026-10-20); moving to a new network re-issues the CA, and clients re-trust it;
     - it issues 30-day leaf certificates for the node's current names and addresses, reissued without interruption.
   - **One CA per node, never a shared CA key**. No CA key is copied to another node or issued through the coordinator.
   - **`syntropy trust <node>`** installs that CA on a client machine after the user compares the fingerprint, into the user's trust store by default and the system store only with `--system`. It records the change in the client's host-change ledger and prints the environment variables for clients that use their own CA bundles.
   - **`syntropy trust --cluster <node>`** (M2) fetches every member's CA certificate through that node, each bound to its node key, which the node's admission certificate binds to the cluster key. It shows the cluster ID's six-word fingerprint once, installs every member's CA in one step, and picks up new members when it is run again.
   - **User-supplied certificate files** remain available, for example for a tailnet name.
   - TLS 1.3 only, through `rustls`.
7. **Debug endpoints** (`/v1/syntropy/parity`, `/v1/syntropy/debug/telemetry`, and bench controls on generation) are outside the scope system.
   - They answer only on loopback listeners and the local socket.
   - They answer only while their `debug.*` key is set.
   - In every other case they return 404, whatever key the caller holds.
   - They ship in every build, so the release artefacts are what gets tested.
8. **Dashboard sessions.** Off loopback, sign-in with a `dashboard` or `admin` key sets an `HttpOnly`, `Secure`, `SameSite=Strict` cookie, valid 12 hours and revoked with the key. On loopback the dashboard is read-only until `syntropy dashboard`, which proves the OS user over the local socket, opens a one-time login URL. Its token travels in the URL fragment, is good for one exchange, and is swapped for the same kind of cookie carrying every scope the local socket has.
9. **Cluster-wide keys exist** (RFC-0034 §8.4, from M2). Any node can be a request's ingress, so `syntropy api-key create --cluster` records a keyholder-signed `ApiKeyCreated` event holding the hash, scopes, model and instance lists and expiry. Every node accepts the key. Revocation is keyholder-signed and monotonic, and `rotate` overlaps old and new keys. Node-local keys remain for single machines.

## Consequences
- **Positive.**
  - No key or prompt ever crosses a shared network in plaintext from the API.
  - The first-run experience on one machine needs no key: chat clients and SDKs on loopback work unchanged, and the CLI uses the local socket.
  - A stray local process that can reach `127.0.0.1` can no longer mint invites, approve joins, create keys or change configuration.
  - Test-only endpoints cannot be reached remotely, even with stolen keys.
- **Negative.**
  - LAN TLS still has friction, and the docs state it plainly:
    - each client machine runs `syntropy trust` once per node;
    - Python and Node.js clients need an environment variable because they ignore the OS trust store;
    - iOS needs a profile and a manual full-trust switch;
    - Android apps ignore user CAs.
    The overlay route or the user's own CA is the practical path for phones.
  - A trusted local CA is a larger grant than a pinned certificate. Its name constraints and key protection are security-critical and tested.
  - Scripts that call `admin` or `sessions` routes over loopback HTTP need a key; the CLI does not, because it uses the local socket.
  - Narrow name constraints mean a node that moves to a new network re-issues its CA, and its clients run `syntropy trust` again.
  - Cluster-wide keys depend on a keyholder being reachable to sign creation and revocation.
- **Work created.**
  - RFC-0050: enforcement and error mapping.
  - RFC-0051: `api-key` commands, `identity show`, fingerprint display, dashboard login.
  - RFC-0054: `api.*` and `debug.*` keys and startup validation.
  - RFC-0060: security tests (rebinding, CSRF, brute force, refusal to start, debug endpoint reachability).

## What would make us revisit
- Field reports show users abandoning LAN access because of certificate trust. First check whether `trust --cluster` and the M2 "client only" pairing role are reaching them. Then consider an installer-registered local CA, recorded in the host-change ledger, and agree it with RFC-0034. A CA key shared between nodes stays ruled out; clients that can hold only one custom CA would get a cluster API CA held by keyholders only.
- A quality gate needs a debug endpoint driven from another machine, for example a lab driver on a separate host. Add an authenticated tunnel for that gate, never a network-reachable debug endpoint.

## Revisions
- Revised 2026-10-20: decision 1: the loopback surface is split by scope. `inference` and the read-only routes stay open on loopback; `admin`, `sessions`, stored-response retrieval and keyholder actions need the local socket or a key, and `api.cors_origins` never opens `admin`. Any local process can reach `127.0.0.1`, and the Host and Origin checks stop browsers only, so without the split a non-browser local process could mint invites and approve joins on a keyholder.
- Revised 2026-10-20: decision 2: the remote-client documentation leads with the overlay bind in M1, and from M2 LAN pairing offers a client-only (ingress-only) role, which a device its approver marks as not their own gets by default. A desktop that runs Syntropy then calls its own loopback and needs neither a key nor a CA.
- Revised 2026-10-20: decision 6: one CA per node and never a shared CA key, because a CA key held on every node turns one stolen disk into a CA that every client trusts. Name constraints narrow from all private address space to the node's DNS names and served subnets, so a node's CA cannot impersonate a router or NAS; the CA installs into the user store by default; `syntropy trust --cluster <node>` installs every member's CA in one step.
- Revised 2026-10-20: decision 8: the loopback dashboard gets its `admin` session from a one-time login URL that the CLI obtains over the local socket.
