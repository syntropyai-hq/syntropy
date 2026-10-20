# ADR-0106: The head stage schedules every step; homogeneous steps; groups for throughput

- Status: Accepted (revised 2026-10-19 and 2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the placement and scheduling area)
- Related: ADR-0000 (D2), ADR-0010, ADR-0012; RFC-0001 §5.8, §6.2, §8.4; RFC-0012 §1.2, §4, §6, §10; SC-15, SC-18

## Context

A model instance is a ring of stages. Every stage must process the same steps in the same order (RFC-0001 §5.8: steps strictly increase per instance), and a sequence's next decode step cannot start until its token returns from the tail to the head (§6.2). Something must decide what each step contains: which sequences, which positions, prefill chunks or decode rows, verify windows. D2 puts single-stream latency first; throughput mode must still fill the pipeline, because with one sequence each stage is busy only t_s / t_tok of the time (cost model §3.7).

Two mixing strategies exist for prefill and decode: separate homogeneous steps, or mixed steps that carry decode rows and a slice of a prompt together. Mixed steps reduce interference but need an `AttnDecode`/`AttnPrefill` mix in one program and a captured graph per combination (ADR-0105).

## Options considered

| Option | For | Against |
|---|---|---|
| A. Each stage schedules its own work | no single scheduler | stages must agree on step contents, a message round per step on the token path |
| B. The coordinator schedules | global view | puts the coordinator on the request path, violating RFC-0001 invariant 2 |
| C. The tail schedules | it sees sampled tokens first | sequences enter at the head and prefill starts there; the head holds the prefix index |
| **D. The head schedules every step; other stages execute frames in step order; homogeneous steps; latency mode puts the active interactive sequence first and caps other prefill chunks by a stall budget; throughput mode circulates G = D groups of up to b_μ sequences** | one writer of composition per instance, at the place where sequences enter and tokens return; bounded interactive stalls; full pipeline in throughput mode; few buckets | the head must learn each stage's KV headroom (side channel); prefill chunks and decode alternate rather than mix |
| E. Mixed steps with a token budget | smooth interference in throughput mode | many more graphs; mixes attention kinds in one program |

## Decision

**Option D.**
- The head's `sched` composes every live step. Middle stages and the tail execute each sequence's live frames in step order and never reorder them; frames of different sequences may run out of global step order under the lane precedence below (RFC-0032 §3.2). The one exception is recovery: a replay source stage composes Replay steps from its own history, which run on their own channel between live steps and never reorder them (RFC-0012 §5.1, RFC-0033 §4.2).
- Latency mode: the active interactive sequence's decode or verify step always goes first; other prefill chunks run only when it is not ready, capped so `max_s t_s(c) ≤ 0.5 × predicted t_tok` (floor 64 tokens) while it decodes. Prefill chunks travel on a separate prefill lane with its own credits, and every stage serves the decode lane first (RFC-0032 §3.2); without that, FIFO queueing at the slowest stage breaks the bound. Lane precedence is decode > replay > prefill.
- Latency mode, concurrent rows (added 2026-10-20): on dense and RAM-resident MoE instances, up to 4 sequences decode in one homogeneous `Batch` step when their KV fits, and the cap above applies to other prefill chunks while any of them decodes. On instances with an `Nvme`-homed pool, one sequence decodes at a time and a second request queues at the head (RFC-0012 §4.6).
- Throughput mode: G = D groups of up to b_μ sequences, issued round-robin as their tokens return; one prefill step per G decode steps while groups wait; groups merge when half-empty.
- Admission is decided at the head against per-stage KV headroom carried on frames; ingress's check is advisory.

## Consequences

- Positive: no negotiation on the token path; a bound of one capped chunk per stage on interactive stalls, given the decode lane's precedence over the prefill lane; throughput mode keeps every stage busy with D groups.
- Negative: homogeneous steps waste a little throughput compared with mixed steps; the head is a single point of composition for its instance (it is already a single point of failure for replay, RFC-0001 §7.5).
- Work created: RFC-0031 carries per-stage KV headroom on frames; RFC-0061 tunes the stall budget and prefill share.

## What would make us revisit

- Throughput benchmarks show mixed steps beating homogeneous steps by more than 15% on aggregate tok/s at equal TTFT. Then add mixed buckets for throughput mode only.
- Interactive users report stalls during concurrent long prefills even with the cap. Then lower the cap or preempt chunks mid-stage.
- Cross-node TP islands make the head's composition latency visible (it is a few microseconds today).

## Revisions

- **Revised 2026-10-20.** Latency mode gains concurrent decode rows from M1: with `max_sequences` = 2 and no continuous batching until M3, an agent's parallel sub-agents and tool calls ran one at a time through M2. RFC-0040 §9 row 2 targets `time(rows) ≤ 1.10 × time(rows = 1)` for 2–8 rows, so the main stream pays about 10% at most on dense stages. NVMe-tiered MoE stays serial, because extra rows multiply the experts touched and a second prompt's prefill would re-read the pool on the drive the active decode needs.
