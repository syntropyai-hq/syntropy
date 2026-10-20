# ADR-0303: Invite tokens are compact binary records anchored on the inviter's node key, single-use by default

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the distributed runtime area)
- Related: ADR-0000 (D4), ADR-0009, ADR-0011, ADR-0313, RFC-0001 §12, RFC-0030 §6.4, RFC-0034 §7

## Context

D4 requires a device on another network to join with an invite token, over a relay, a hole-punched QUIC path or Tailscale. RFC-0001 §12 requires tokens to be single-use by default, to expire, and to carry scopes. They authorise one admission and are not a standing credential.

The joining device must authenticate the cluster even when the first packets go through a relay it does not trust. The token is pasted into chats, emails and QR codes, so it will leak sometimes. The survey describes a federation whose shared token leaked by being published (`multi-device-inference.md` §1.3).

## Options considered

| Option | For | Against |
|---|---|---|
| A. Shared cluster secret as the token | Simple | Standing credential; leaking it admits anyone forever; revocation needs rotation everywhere |
| B. Signed JWT-style bearer token | Familiar; self-describing | Long (hundreds of characters); bearer semantics alone do not authenticate the cluster to the joiner |
| C. Short typed code with a PAKE | Typeable | Needs PAKE code and a rendezvous lookup service; more moving parts |
| **D. 69-byte binary record: version, scopes, expiry, uses, cluster tag, inviter node key, 128-bit secret, CRC; base32 with prefix `syn1`; optional relay, address and alternate-inviter trailers. The coordinator stores only a verifier hash; the secret is revealed only inside a TLS session with the authenticated inviter** | About 115 characters; QR-friendly; joiner authenticates the cluster via the inviter key; single use enforced atomically by the single writer; a leaked spent token is useless | Inviting keyholder (or an alternate) must be online at join time |

## Decision

**Option D** (RFC-0030 §6.4).
1. Layout:

   | Field | Bytes |
   |---|---|
   | `version` | 1 |
   | `flags` | 1 |
   | `scopes` | 2 |
   | `expiry`, as Unix minutes | 4 |
   | `max_uses` | 1 |
   | `cluster_tag` | 8 |
   | `rendezvous`, the inviter's node ID | 32 |
   | `secret` | 16 |
   | optional trailers | variable |
   | CRC32C | 4 |

   Encoding is lowercase Crockford base32 with the prefix `syn1`, also offered as a `syntropy://join?t=` QR code.
2. `token_id` and `verifier` are distinct BLAKE3 derivations of the secret. `InviteCreated` logs only those two. The secret is never stored.
3. The joiner dials the rendezvous node ID, using hints from the token. TLS proves the peer holds that key. The peer's admission certificate must chain to a cluster key whose BLAKE3 prefix equals `cluster_tag`. Only after that check does the joiner send the secret.
4. The coordinator validates the verifier, expiry, remaining uses and scopes. It emits `NodeAdmitted` and `InviteConsumed` in one batch.
5. Defaults are `MEMBER | HOST_STAGES` scopes, 24 h expiry and 1 use, with a 7-day cluster maximum.
   - `--ingress-only` produces nodes that never receive weights or activations.
   - Keyholder invites need expiry ≤ 1 h and a confirmation.

**Revised 2026-10-19:**
- Scope bit4 `COORDINATOR` is added. Default scopes are `MEMBER | HOST_STAGES | INGRESS`, without `COORDINATOR`. `--coordinator` sets it.
- The rendezvous keyholder signs the admission `Approval`, which every node verifies (RFC-0034 §8.3).
- After a split heal, the rendezvous's journalled `Admit` is re-submitted (RFC-0032 §6.10). A single-use token therefore holds per log lineage, with at most one extra admission per side of a split.

**Revised 2026-10-20:**
1. **Whose device.** `invite create` asks whether the invite is for one of the inviting user's own devices or someone else's. Someone else's device gets `MEMBER | INGRESS`, so it never processes or stores the inviting user's prompts; the user's own device keeps decision 5's stage-host default. An invite created where the question cannot be asked is ingress-only. The admission notification shows the choice. This replaces decision 5's single default: tokens are pasted into chats and emails, and a stage-host default suited only the user's own laptop.
2. **The built-in relay set** behind the `HAS_RELAY` trailer's kind 0 is a hosted shared relay service, delivered as the signed TUF target `relays.json` (RFC-0030 §6.4). The token format does not change, and tokens keep working when the relay provider changes, because kind 0 names "the built-in set", not a URL.

## Consequences

- Positive:
  - one paste or one scan joins a device;
  - a relay cannot impersonate the cluster;
  - a leaked token is useless after use or expiry, and revocable before;
  - no public discovery service learns node addresses.
- Negative:
  - The rendezvous keyholder, or up to two alternates in the token, must be reachable when the joiner uses the token.
  - After a network-split heal, a token consumed only on the losing side becomes usable again until it expires (RFC-0030 §8.2). Short default expiry bounds this.
- Work created:
  - RFC-0051: token display, QR code and pending list;
  - RFC-0060: fuzz targets for the token parser.

## What would make us revisit

- Support data shows that joins fail mostly because the inviter is asleep. Add coordinator-held verifiers, so any keyholder can serve as rendezvous without being named in the token.
- A PAKE crate with a stable, audited API appears, and users ask for typed codes. Add a short-code form alongside this one.
