# ADR-0311: Every sending stage keeps its outbound activation history in the wire dtype, spilling to NVMe

- Status: Accepted
- Date: 2026-10-19
- Deciders: maintainers (drafted by the distributed runtime area)
- Related: ADR-0000 (D3), ADR-0010 (item 4), ADR-0019, ADR-0307, RFC-0001 §7.5 and §17, RFC-0033 §4

## Context

Recovering a lost pipeline stage without a full re-prefill needs that stage's inputs for every committed position of every live sequence. Those inputs are its predecessor's outputs.

The survey's only working example is activation-history replay. A client kept the inputs it sent to each server span and replayed them to a replacement (`multi-device-inference.md` §10.2). That client sat at the centre of a star, so it saw every boundary.

Syntropy's data plane is a ring (ADR-0010), so no single node sees every boundary. The binding texts disagree on where history lives:
- ADR-0010 item 4: "senders retain the per-sequence boundary history that RFC-0033 needs";
- RFC-0001 §7.5: "the head keeps … the activations it sent across each stage boundary", and the head is a single point of failure for replay.

Exact mode carries F32 on the wire wherever the reference holds F32 (ADR-0010 item 5). RFC-0001 §7.5's memory figure assumes F16.

## Options considered

| Option | For | Against |
|---|---|---|
| A. No history; always re-prefill | Simplest; no memory | Full prefill per recovery: about 2× the compute of replay in RFC-0033's example, worse at long contexts |
| B. Head-only history, as RFC-0001 words it | One place to look | The head sees only boundary 0. Other boundaries need every stage to send a copy of each frame to the head: D − 2 extra sends per token, prefill traffic doubled |
| C. History replicated to a standby node | Survives losing the sender | A network copy of every frame; unnecessary if no stage is a single point of failure |
| **D. Each sending stage keeps what it sent, in the wire dtype, in a `HostPageable` budget with NVMe spill; the head's "history" is the token transcript ingress already holds** | No extra network traffic; any single stage loss is replayable from its predecessor; head loss recovered from tokens; spills to disk under pressure | Memory or disk on every non-tail stage, about 20–80 % of that stage's KV in RFC-0033's examples |
| E. BF16 history to halve memory in exact mode | Smaller | Replay inputs would differ from the originals beyond device rounding; a deliberate approximation outside approximate mode (D3) |

## Decision

**Option D** (RFC-0033 §4.1).
1. History is copied from acknowledged send slots by the housekeeping thread, off the critical path. Only committed positions are kept. Verify windows are trimmed at acceptance, together with KV truncation.
2. Budgets:
   - RAM: `recovery.history_ram` = min(2 GiB, 5 % of RAM) per runner, never pinned;
   - spill: an append-only direct-I/O file capped at `recovery.history_disk` = 16 GiB.
3. History is shed early under memory pressure: it spills, and drops as a last resort. A drop advances the sequence's `from` position and sets `HISTORY_PARTIAL`. Recovery then cascades to an upstream source, ultimately the head with the transcript.
4. In throughput mode, history is `auto`: off when the predicted footprint exceeds the budgets. The dashboard says that losses then re-prefill.
5. RFC-0001 §7.5 and §17 question 2 are amended accordingly (RFC-0033 "Proposed amendments").

## Consequences

- Positive:
  - Any single stage loss resumes by replaying one hop's worth of history plus the lost layers' prefill.
  - Head loss costs only the head range's prefill.
  - The only single point of failure left is the ingress node, for its own sequences.
- Negative:
  - Memory or disk use per sequence of `context × hidden × streams × 4 B` in exact mode: 128 MiB at 4K for a 70B model per boundary.
  - Positions served from a prefix cache have no history, and their recovery cascades upstream (RFC-0033 open question 5).
- Work created:
  - RFC-0020: the history sub-arena and its place in the pressure ladder;
  - RFC-0021: the spill write path;
  - RFC-0060: replay parity tests.

## What would make us revisit

- In throughput mode, the history footprint routinely disables history on realistic household workloads, and re-prefill recoveries prove too slow. Consider per-sequence history for interactive sequences only.
- KV compression or a family's KV format makes KV smaller than history. Checkpointing KV could then become cheaper than keeping history.
