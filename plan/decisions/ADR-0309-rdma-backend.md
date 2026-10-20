# ADR-0309: One RDMA backend over SEND/RECV for Thunderbolt 5 UC and Linux RC, with Syntropy-level reliability

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the distributed runtime area)
- Related: ADR-0006, ADR-0011, ADR-0012, ADR-0307, ADR-0314, RFC-0031 §5.2, RFC-0010

## Context

ADR-0011 ranks RDMA first among network transports. It names two variants: ibverbs over RoCEv2 or InfiniBand on Linux, and RDMA over Thunderbolt 5 on macOS 26.2 and later.

Facts from Apple's TN3205, as summarised in ADR-0011, constrain the design:
- verbs API only;
- send/receive only;
- unreliable-connected (UC) queue pairs, at most 10 of them;
- 4,095 outstanding work requests;
- messages up to 16,773,120 bytes;
- matched frame counts on both sides;
- no hardware acknowledgement;
- point-to-point cables.

Linux RC queue pairs offer reliable delivery, RDMA WRITE and READ. Field reports for one surveyed runtime describe RDMA errors that needed full reboots of every node, several times a day (`multi-device-inference.md` §5.5). There is also a reported, unconfirmed limit of about 100 memory regions in the Thunderbolt RDMA driver (`hardware-and-performance-model.md` §2.5).

## Options considered

| Option | For | Against |
|---|---|---|
| A. Separate backends: one-sided RDMA WRITE on Linux, send/recv on macOS | Best possible Linux path | Two code paths, two reliability models, twice the test matrix |
| B. Use only Linux RC semantics and skip Thunderbolt 5 | Simple | Loses the one RDMA fabric consumer Macs have; TP on Mac pairs impossible (ADR-0006) |
| **C. One backend using SEND/RECV only: UC on TB5, RC on Linux; Syntropy's sequence numbers, CRC, credits, NACKs and retention (ADR-0307) provide reliability on UC and recovery after QP resets on RC** | One path, tested everywhere; matched frame counts by construction; in-process recovery | Forgoes one-sided operations on Linux for now |
| D. Wrap a third-party RDMA messaging library | Less code | Adds a dependency per platform; none covers TB5's UC-only constraints |

## Decision

**Option C** (RFC-0031 §5.2).
1. Libraries are loaded dynamically: `libibverbs.so.1` on Linux, and the system RDMA library on macOS. One binary runs on hosts without RDMA.
2. Structure:
   - one QP and two CQs per channel, at most 4 QPs per neighbour;
   - the receiver posts `recv_slots` receives;
   - credits never exceed posted receives;
   - sends use a two-entry scatter-gather list (header area, payload slot), with no copy;
   - small frames go inline where the device allows it.
3. Each buffer pool is registered as one memory region, fewer than 8 per runner. GPUDirect RDMA through `ibv_reg_dmabuf_mr` is used only after RFC-0010's device-path probe confirms it.
4. QP parameters are exchanged over the authenticated QUIC `RunnerEndpointOffer`: GID, QPN, PSN, MTU and a channel token. The first frame on a QP is a credentialed `Hello`.
5. A QP error leads to destroy, recreate and re-pair in-process, then a resend from retention. Three faults in 60 s demote the edge for the epoch (ADR-0011 item 7).
6. CQs are busy-polled by a per-NIC poller, or by the dispatcher. They are armed for events only after the idle spin budget.

**Revised 2026-10-20:** the TB5 path is first exercised by a raw-path probe on a TB5 Mac pair late in M1 (RFC-0031 open question 1). That session adds a one-day fault smoke test: a cable pull, a sleep and wake cycle, and a peer reboot. If any of them needs a reboot to recover, TB5 RDMA ships opt-in at M2 instead of waiting for the 6-month soak below, because a recovery that needs a reboot is learned far more cheaply in M1 than at M4. The same session confirms the memory-region figure behind decision 3, which RFC-0031 open question 4 now treats as not binding.

## Consequences

- Positive: one RDMA code path; Thunderbolt 5 usable despite UC semantics; failures recover without reboots by design; RDMA is opportunistic and never required.
- Negative:
  - Linux RoCE gives up some potential from one-sided writes.
  - Application-level reliability adds the CRC and ack costs measured in RFC-0031.
- Work created:
  - RFC-0060: `rdma_rxe` CI job and TB5 cable-pull test (the pull itself is a **Maintainer action** on the lab rig, ADR-0021);
  - RFC-0010: device-path probe;
  - RFC-0051: "RDMA not enabled" and "macOS build mismatch" states.

## What would make us revisit

- Apple adds RC queue pairs or one-sided operations to Thunderbolt RDMA (ADR-0011 revisit list).
- Linux RoCE clusters become common among users and benchmarks show WRITE-with-immediate saving more than 10 µs per hop.
- After 6 months of soak testing, TB5 RDMA still needs reboots to recover. Make it opt-in (ADR-0011).
