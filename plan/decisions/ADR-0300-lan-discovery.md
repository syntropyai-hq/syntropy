# ADR-0300: LAN discovery uses three mechanisms with signed beacons and a visible version mismatch

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the distributed runtime area)
- Related: ADR-0000 (D4), ADR-0009, ADR-0011, RFC-0030 §4, RFC-0031, RFC-0034

## Context

D4 requires devices on one local network to find each other with no setup. SC-24 rules out typed addresses and configuration files.

The survey shows how discovery fails in practice (`plan/research/multi-device-inference.md` §1.1–§1.6, §13):
- One runtime's docs said mDNS while its code used UDP broadcast.
- Another limited discovery to one IPv6 multicast domain, so Tailscale, routed and cloud nodes could not join.
- Clusters silently refused to form when version strings differed, and the only explanation was a log line (item 24).
- macOS Local Network permission denials blocked discovery, and users reported one-way visibility on Thunderbolt meshes.

Home networks break multicast in several independent ways:
- access points that filter multicast between Wi-Fi and Ethernet;
- IGMP snooping without a querier;
- the Windows Public network profile;
- the macOS permission prompt;
- WSL2 in NAT mode.

IPv6 link-local addressing needs no DHCP. A direct Ethernet or Thunderbolt cable between two machines is usable the moment the link comes up, and it is often the fastest path in the cluster.

## Options considered

| Option | For | Against |
|---|---|---|
| A. DNS-SD over mDNS only | Standard; OS responders exist on macOS, Windows 10+ and most Linux desktops | Fails silently on the networks listed above; no signature on records |
| B. Custom IPv6 multicast beacons only | Works on link-local-only cables; simple | No IPv4-only networks; invisible to standard service browsers |
| C. IPv4 broadcast only | Works where multicast is filtered | Doesn't cross Wi-Fi/Ethernet bridges on some routers; IPv4-only |
| D. DHT or public rendezvous service | Crosses NATs | Public infrastructure and identity leakage; out of scope for LAN (D4) |
| **E. All of A, B and C at once, with Ed25519-signed binary beacons for B and C, a namespace hash, and the protocol major carried separately** | Each covers the others' failure modes; signed beacons prevent address spoofing; mismatches are visible | Three code paths to test; a small amount of extra LAN traffic |

## Decision

**Option E** (RFC-0030 §4).
1. Every node runs:
   - DNS-SD over mDNS, through the OS responder where one exists and a built-in responder otherwise;
   - 192-byte signed beacons to `ff12::5359:4e54` on every multicast-capable interface;
   - the same beacons by IPv4 broadcast.

   The default port is 47460/udp.
2. Beacons and TXT records carry the protocol major and the namespace hash as separate fields. A node with a different major is listed as *incompatible*, with both versions, in the CLI and dashboard. It is never silently ignored.
3. Discovery yields candidates only. Membership comes from admission (ADR-0302, ADR-0303). Every discovered address is authenticated by the QUIC handshake against the node ID before use.
4. Static peers and a member address cache are dialled whatever discovery finds, so multicast-less networks still work.
5. Diagnostics name the likely cause of silence: Local Network permission, firewall, Public profile, or WSL2 NAT.

**Revised 2026-10-20:**
1. **Clustered nodes beacon only where their cluster lives, and blinded.** An open beacon carried a persistent node ID, the cluster tag, the display name and a keyholder flag every 5 s on whatever network a laptop joined, which tracked the device across networks and pointed at the machine holding the cluster key. A clustered node now sends beacons and mDNS records only on networks where a member has been seen or the user marked home. Its beacons carry a tag keyed by a member-only beacon key and rotated every 15 minutes, plus a keyed MAC in place of the Ed25519 signature, so only members recognise them and no node ID, cluster tag, name or keyholder flag is broadcast. Elsewhere it stays silent and answers only authenticated dials. Unclustered nodes, disposable clusters and keyholders that are pairing still send decision 1's signed open beacons (RFC-0030 §4.2, §4.8). The `PAIRING_OPEN` flag is dropped, because it told every device on the LAN when admission was open. Cost: one keyed hash per beacon.
2. **Ports,** superseding the "47460/udp" of decision 1. The beacon port is UDP 7434 and the daemon's iroh port UDP and TCP 7435, beside the API's 7433. The old pair sat inside Linux's ephemeral range (32768–60999), so the fixed beacon bind could fail, and moving later would break discovery between mixed versions. IANA lists 7432–7436 as unassigned (checked 2026-10-20).

## Consequences

- Positive: discovery works on the common broken-multicast cases. Version skew is explained to the user. Point-to-point cables work with no DHCP.
- Negative: three mechanisms to keep tested on each Tier 1 OS. The beacon format is a wire contract, versioned with the protocol major.
- Work created:
  - RFC-0052: firewall rules and the macOS permission prompt;
  - RFC-0060: per-mechanism disable tests;
  - RFC-0051: the nearby list.

## What would make us revisit

- Field reports show one mechanism finding no peers that the others miss over a full release cycle. Drop it.
- macOS or Windows changes its system responder APIs or permission model so that the per-user agent cannot advertise.
- Common home software turns out to use UDP 7434–7435 (DIST-E3-T01 checks before M2), or IANA assigns them to another service.
