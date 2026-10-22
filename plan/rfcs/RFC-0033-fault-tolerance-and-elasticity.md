# RFC-0033: Fault tolerance and elasticity

- Status: Draft
- Area: distributed runtime
- Depends on: RFC-0001 (§3.3, §3.4, §7.2–§7.5, §8.4, §10), RFC-0011, RFC-0012, RFC-0020, RFC-0022, RFC-0030, RFC-0031, RFC-0032, RFC-0041, RFC-0050, RFC-0060, ADR-0600, ADR-0006, ADR-0009, ADR-0010, ADR-0011, ADR-0015, ADR-0016, ADR-0019, ADR-0301, ADR-0304, ADR-0307, ADR-0311, ADR-0312
- Date: 2026-10-19

## Summary

This RFC specifies what happens when something fails while Syntropy is serving, and how the cluster grows and shrinks without the user's help.

- **Replay instead of restart.** Each sending stage keeps the activations it sent across its boundary for every live sequence. The history is kept in the wire dtype, in RAM, and spills to NVMe.
  - When a stage is lost, the survivors relink in place, keeping their KV. A replacement stage, or a survivor absorbing the lost layers, rebuilds its KV by replaying its predecessor's history.
  - One recovery handshake around the ring finds, for every sequence, the nearest upstream stage that can supply the missing inputs. The head can always supply them from the ingress node's transcript, so a head loss is recovered by re-prefilling only the head's layers.
- **The same mechanism for every recovery.** Re-prefill, replay and request migration are one mechanism with different source stages. Emitted tokens are never retracted, and a counter-based sampler makes the continuation identical in distribution.
- **Planned transitions.** Re-placement, unload and drain choose per sequence between finishing, KV migration, replay and re-prefill, by predicted cost.
- **Repair plans.** After a loss, the coordinator asks the solver for a repair plan that keeps surviving shards fixed. It picks whichever of "replace" and "absorb into survivors' slower tiers" resumes sooner. A degraded result is upgraded later, under the normal hysteresis rules.
- **Smaller failures.**
  - Link faults are absorbed by the transport, with retention and failover, and never need replay.
  - Runner crashes restart in place and replay.
  - Stragglers raise drift, never kills.
- **Single points of failure, stated plainly.** The ingress node is one for the sequences it holds. The coordinator is one for *placement changes* during its 4.5–5 s election gap. Nothing else is.
- **Standby.** By default, an evictable second copy of each stage shard's source chunks sits on the node a repair would absorb that stage into, so most cold repairs read from the LAN, not the Hub. Opt-in warm standby pre-fetches and packs replacement shards, and a replica can serve as a migration target.
- **A dropped client link is not a lost ingress.** The ingress holds a paused sequence and its KV for a 60 s detach grace, and the client re-attaches.
- **Testing.** A chaos plan with 20 scenarios runs on the simulated fabric nightly and on hardware weekly.

## Motivation and goals / non-goals

Every surveyed home-cluster runtime aborts or stalls when a stage dies (`multi-device-inference.md` §10.1, §13 item 16):
- one deletes the instance after 15 s of silence and fails its requests;
- one stalls the ring until the job is restarted;
- one aborts on the first socket error;
- one requires the user to re-place the model by hand.

Mid-generation recovery existed in two forms. Activation-history replay to a replacement stage appeared only in a 2023 internet-scale swarm. Request migration by re-prefill appeared in datacenter stacks (§10.2, §10.3). Home devices make failure normal, not exceptional: laptops sleep, Wi-Fi drops, a gaming PC reboots for a game update.

Goals:
1. **SC-28.** When a non-head stage node is lost and the survivors have capacity, the sequence resumes with no client-visible error within `t_detect + t_solve + S_delta/β_fetch + t_replay`. This RFC extends the guarantee to head and tail loss whenever the ingress survives.
2. **SC-27.** A coordinator change fails no in-flight request. RFC-0032 covers the election. This RFC covers recovery that waits on it.
3. Never retract or duplicate an emitted token. In exact mode, stay inside ADR-0019's teacher-forced contract.
4. No degradation path changes output (RFC-0001 §10.2). Recovery never turns on an approximate knob.
5. Re-place automatically after every loss. Never wait for the user (`multi-device-inference.md` §15.8).
6. Bounded memory: history has a budget and degrades gracefully under memory pressure.

Non-goals:
- Surviving the loss of the ingress node for its own sequences. The client connection goes with it.
- Byzantine faults (D4). A member that computes wrong values is caught by the canary and parity checks, not tolerated.
- Bit-identical continuation across heterogeneous replacement hardware (ADR-0019).
- Losing two adjacent stages at once with no surviving upstream history *and* no ingress transcript. That case ends the sequence with a retryable error.

## Design

### 1. Failure taxonomy

| Failure | First signal | Detected by | Recovery path | Client sees |
|---|---|---|---|---|
| Link fault on one edge: QP error, cable pull, Wi-Fi roam, packet loss | channel error or ack timeout | `net` in the runner (RFC-0031 §7) | reconnect, or demote to the next transport; resend retained frames | a pause of at most `link.failover_timeout`; nothing else |
| Link down for longer than the failover timeout, node alive (indirect probe acks) | `EdgeDown` | RFC-0030 §5.3 | re-plan with another path between the same nodes (`Relink` to a new transport) | a pause |
| Runner crash: driver fault, sticky GPU error, OOM kill | process exit | supervisor (RFC-0001 §3.4) | restart in place (pack present: `Loading` + `Warming` only) or replace elsewhere, whichever resumes sooner; then replay (§4) | a pause |
| Runner hang or stall | heartbeat or step counter stops | supervisor | kill, then as a crash | a pause |
| Node loss: power, crash, network unplugged | `EdgeDown` from ring neighbours; missed beats | data plane, then SWIM (RFC-0030 §5) | repair plan (§6) and replay (§4) | a pause; continues |
| Node sleep with notice | `Leave { Sleep }` | RFC-0030 §5.9 | as node loss, starting immediately | a pause |
| Head node loss, ingress elsewhere | as node loss | as node loss | new head re-prefills its layers from the ingress transcript (§4.4) | a pause |
| Tail node loss | as node loss | as node loss | replay into the replacement tail; the sampler continues from the counter (§4.5) | a pause |
| Fast-island member loss | collective channel failure | data plane | the whole island stage is treated as lost (RFC-0032 §5.6); replay into the repaired stage | a pause |
| Ingress node loss | Tokens channel to ingress fails; `NodeFailed` | tail; coordinator | cancel the node's sequences around the ring (§9.2) | the connection is gone |
| Coordinator loss | lease expiry | every node (RFC-0032 §7) | election; in-flight sequences unaffected | nothing; new loads wait about 5 s |
| Network split | many edges down at once | SWIM | each side serves what it can; heal reconciles (RFC-0032 §7.5) | sequences on intact sides continue; others recover or end |
| Straggler: thermal throttling, a game launched, Wi-Fi degraded | step time above prediction | per-stage telemetry (§8.1) | `Drift`, then a deferred re-plan | slower tokens; no error |
| Memory pressure | OS signal | daemon (RFC-0001 §10.3) | memory ladder; history spills first (§4.1) | nothing, or slower tokens |

### 2. Single points of failure in v1

Stated explicitly, as `multi-device-inference.md` §15.2 item 9 and RFC-0001 §17 question 2 ask:

1. **The ingress node, for the sequences it holds.** It owns the client connection and the transcript. Losing it ends those sequences. Every other node's sequences are unaffected. This is inherent, because the client connection goes with it, and v1 has no replicated ingress. A client that reconnects to another node with the same session ID resumes from persisted session KV if the session was persisted and its key survives (RFC-0022, RFC-0034 §13), otherwise from scratch. A dropped client link with the ingress alive is not this case: the detach grace covers it (§9.2).
2. **The coordinator, for placement changes only, during an election.** For about 4.5–5 s after an unplanned coordinator loss (RFC-0032 §Performance), nothing can commit a plan. No new instance, no re-plan and no repair plan is possible. In-flight sequences on intact rings keep running (SC-27). If the coordinator's node also hosted a stage, recovery of that stage waits for the election (open question 4).
3. **A persisted session's holders,** until the session is re-persisted elsewhere (RFC-0022). Losing every holder costs a re-prefill on resume, not an error.
4. **A one-node instance's node.** With no other node holding its layers, the instance is unroutable until the node returns or a re-plan finds capacity. Its sequences end with a retryable `Capacity` error.

**The head is not a single point of failure for replay.**
- History is kept by every sending stage (ADR-0010 item 4, ADR-0311), not only by the head.
- The head's own inputs are token IDs, and ingress holds them.
- RFC-0001 §7.5 and §17 describe a head-only history and call the head a single point of failure. This RFC proposes amending that text ("Proposed amendments").

### 3. Detection and the stall

#### 3.1 The fast path from the data plane

SWIM confirms a node failure in about 6–8.5 s on LAN (RFC-0030 §5.4). The data plane knows sooner.

1. A runner whose channel fails past `link.failover_timeout` (2 s LAN, RFC-0031 §7.4) marks the channel `Down` and reports `EdgeDown { channel, peer }` to its daemon over IPC.
2. The daemon submits `ReportEdgeDown` and immediately starts the `Probing` step for that peer (RFC-0030 §5.3). Membership does not wait for 2 missed beats.
3. The coordinator **starts recovery** with `RecoveryStarted { instance, epoch, lost, cause }` as soon as either of these holds:
   - `NodeFailed`, `NodeSleeping` or `NodeLeft` for a node in the plan; or
   - **both** ring neighbours of a stage report `EdgeDown` towards it, or its island members do, **and** the coordinator's own probe of that node fails.

   It starts solving the repair plan at once (§6). It **commits** the plan only after `NodeFailed`, or after `recovery.confirm_after` (4 s, twice the failover timeout) of continuous unreachability, whichever comes first. A transient flap can then cost a wasted solve, but never a needless re-shard.
4. A runner crash is reported by the supervisor as `ReportRunnerFailed`, which starts recovery at once. The process is known dead.

**Detection to recovery start [target]:** about 2–3.5 s for a node loss on wired LAN (failover timeout plus one probe), and below 100 ms for a runner crash.

**`NodeFailed` [target]: ≤ 8 s after the last heartbeat on LAN** (RFC-0030 §5.4). SC-25's "detected within 3 missed heartbeats" is met by `Suspect`, which is informational. `NodeFailed` is the confirmation that lets the coordinator commit a repair plan when the data plane has not already done so, so it gets its own target. DIST-E4-T12 measures it in the simulator and on the S2 rig under load.

#### 3.2 The stall

Between the failure and the resume, the instance's ring is stalled for the sequences that cross the lost stage. In practice that is every sequence of the instance.
- **Upstream stages** hit credit exhaustion on their outbound channel. Their send attempts return `WouldBlock`, and they stop issuing steps for those sequences (RFC-0001 §8.4 rule 1). They keep their KV, their activation history and their retention windows.
- **The head** stops issuing steps for affected sequences. It keeps accepting `SequenceOpen` for the instance, but routing has already stopped sending new sequences there: `RecoveryStarted` makes the instance unroutable for new admissions in every ingress's replica.
- **Ingress** keeps client connections open. It emits the API's keep-alive (an SSE comment) every 5 s (RFC-0050), and it records the recovery event for the response metadata (RFC-0001 §10.4).
- **Deadlines.** A sequence whose request deadline passes during the stall ends with a `Timeout` error. That error is retryable, and its message tells the client a recovery was in progress.

#### 3.3 What counts as a lost stage

- A stage is lost when its runner process is gone (crash, node loss, sleep) or when it is unreachable past `recovery.confirm_after`. Its KV, its outbound history and its sampler state, if it is the tail, are gone with it.
- In a fast island, the loss of any member loses the whole island stage (RFC-0032 §5.6).
- A restarted runner on the same node is a replacement with no KV, like any other.

### 4. Activation history and replay

#### 4.1 The history store (ADR-0311)

Every stage except the tail keeps, for each live sequence, the activations it **sent** to its successor, in the wire dtype. It does this on every hop, including within a fast island's primary. This is ADR-0010 item 4, "senders retain the per-sequence boundary history".

```rust
/// Per runner. Single writer: the housekeeping thread, fed by send-completion records.
pub struct HistoryStore {
    pub budget_ram: Bytes,            // recovery.history_ram
    pub budget_disk: Bytes,           // recovery.history_disk
    seqs: HashMap<SequenceId, SeqHistory>,
}
pub struct SeqHistory {
    pub boundary: StageIx,            // this stage's outbound boundary
    pub dtype: WireDType,
    pub row_bytes: u32,               // streams × hidden × elem, unpadded
    pub from: u32,                    // first position held; > 0 when a prefix-cache hit skipped earlier positions
    pub to: u32,                      // one past the last position held
    ram: SegmentList,                 // 2 MiB segments in the HostPageable history sub-arena
    disk: Option<SpillFile>,          // 4 KiB-aligned append-only extents (§4.1.2)
}
```

##### 4.1.1 Write path

- When a frame's send is **acknowledged**, RFC-0031 `completed()` returns the slot. Before the slot is re-used, the housekeeping thread copies the payload rows into the sequence's history. A retention window already holds the slot until the ack (RFC-0031 §7.3), so this adds no slot pressure.
- The copy is off the critical path: the dispatcher has already moved on. Its cost is about 3 µs for a 32 KiB decode row [estimate: memcpy at ≥ 10 GB/s], plus about 0.1 ms per MiB of prefill.
- Only positions that are **committed** are kept. A verify window's rows are stored after the tail's acceptance comes back, and only for accepted positions plus the bonus token. The head learns the acceptance from the ring-return frame and tells its successors through the next frame's start position (implicit truncation, RFC-0032 §3.5). Every stage then truncates history at the same point at which it truncates KV.
- History for a sequence is freed on `Release`, `Cancel` or a finish.

##### 4.1.2 Budget and spill

- **RAM.** `recovery.history_ram` defaults to min(2 GiB, 5 % of RAM) per runner. History lives in `HostPageable` memory, never pinned (RFC-0020).
- **Spill.**
  - When the RAM budget is exceeded, the oldest positions of the largest histories move to an append-only spill file: `state/history/<instance>-<runner pid>.hist`, written through the I/O engine with direct I/O in 4 KiB-aligned extents (RFC-0021).
  - History is cold, sequential data, written once and read only on recovery.
  - Disk is capped by `recovery.history_disk` (default 16 GiB).
- **Memory pressure.** History is shed early. It spills in rung 3 of RFC-0001 §10.3's ladder ("prefetch staging and other caches"). If the disk cap is also exceeded, the oldest histories are **dropped**, the sequence's `from` advances, and the stage reports `Ready { degraded: HistoryPartial }`. Recovery then cascades upstream for the dropped range (§4.2). History is never kept at the price of an OOM.
- **Throughput mode** with many sequences may disable history per instance (`recovery.history = auto`). It is disabled when the predicted history footprint exceeds the RAM budget plus the disk cap. Losses then fall back to re-prefill from the head, and the dashboard says so.

#### 4.2 The recovery handshake

After the repair plan commits (§6), survivors relink in place, keeping their KV and history (RFC-0032 §4.4), and the replacement or absorbing stage reaches `Ready { degraded: Recovering }` with empty KV for the recovered layers. The head then drives one handshake per affected instance.

**Inputs.**
- For each affected sequence, the ingress daemon sends `SequenceControl::Reconcile { seq, emitted, rng_counter, recent }` to the head. `recent` holds the last up to 8 emitted tokens. It sends it when the repair `PlanCommitted` is applied **and** the head of the new epoch reports `Ready`, not on `RecoveryStarted`, whose repair head may not exist yet.
- Ingress is the authority on what the client has seen, so the target is `need = prompt_len + emitted − 1`. Every stage must hold positions `[0, need)`, and the resume step processes position `need`:
  - when `emitted > 0`, as a decode row whose input is the last emitted token;
  - when `emitted = 0`, a loss during prefill, as a one-row final `Prefill` chunk whose input is the last prompt token, so the tail samples the first token normally.
- Tokens the tail produced but ingress never received are discarded. KV-only layers truncate to `need`. State layers cannot truncate, so they restore instead (§4.2.1).

**Protocol.** Byte layout in RFC-0031 §6.12.

```text
1. QUERY  head → stage 1 → … → tail → head (ring-return channel)
          entries: (seq, resume_pos = need)
          each stage appends its REPORT per seq: have_pos (committed KV positions), history range
          [from_s, to_s) (including frames still in the retention window, sent but not yet acked),
          max_step, flags (ENDED; HAS_STATE), and for stages with State layers: state_pos (positions
          the recurrent state has absorbed) and state_restore (the highest position ≤ need it can
          restore its state to, §4.2.1)
          the stage's effective h_s = have_pos for KV-only stages, min(have_pos, state_restore) otherwise
2. PLAN   head computes, per sequence:
            lost_s   = { s : h_s < need }                      (stages that must rebuild)
            first    = min(lost_s); if empty → nothing to replay
            repeat until no h_k changes (h only decreases, so at most D rounds):
              for each stage k ≥ 1: drop_below_k = min(h_k, min_{j>k} h_j)
              for each State stage k with drop_below_k < h_k:
                h_k = k's newest restore point ≤ drop_below_k: its rollback slot, a kept checkpoint
                      (checkpoints sit at multiples of recovery.state_checkpoint_tokens, so the head
                      derives them from state_pos), else 0
            lo(u)    = min_{k>u} drop_below_k
            source   = max{ u < first : u = head, or [from_u, to_u) ⊇ [lo(u), need) }
          head sends PLAN entries (seq, source, lo(source), drop thresholds, and each State stage's
          planned restore point) around the ring
3. REPLAY source streams Replay frames for positions [lo(source), need) in prefill-chunk sized frames:
            source = head : recompute embeddings and head layers from the token transcript
            source = u>0  : read rows from u's history (RAM or spill), send as is
          each stage k on receiving Replay rows for positions P:
            compute its layers for positions p ∈ P with p ≥ drop_below_k
            (attention reads its existing KV for positions < p; writes KV only for p ≥ h_k,
             through the step descriptor's per-row kv_write_from; State layers restore to the
             PLAN's restore point and must pass the resume-point check of §4.2.1 on every row)
            append outputs for computed positions to its own history (refilling it)
            forward a Replay frame downstream if any computed position ≥ min_{j>k} h_j
          the tail computes and discards (no sampling); on the last chunk it records completion
4. DONE   tail → head: per seq, completed or failed (with reason)
          head resumes completed sequences at position `need` (decode, or the final prefill row),
          with step numbers continuing from 1 + max(max_step reported in REPORT)
```

- **The common case** is one lost middle stage s, with its predecessor holding full history. Then `source = s − 1`, the replacement computes `[0, need)`, and downstream stages compute only the few positions still in flight at the failure. Cost: history bytes over one hop, plus the lost layers' prefill over the context.
- **Cascades** happen when the predecessor's history does not cover the range: a prefix-cache hit started it late, it was dropped under pressure, or two adjacent stages were lost. The source moves upstream, ultimately to the head, which can always regenerate from tokens. The stages in between recompute outputs over the range using their existing KV. **Re-prefill is the limiting case of replay, with `source = head` and every stage computing.**
- **Ordering.** Sequences are replayed in priority order: interactive latency-mode sequences first, then by shortest `need`. Several sequences' Replay frames may share frames, one span per sequence, as in throughput mode. Replay runs on its own `Replay`-purpose channels (RFC-0031), so a sequence that finished replaying can resume decode on the Activations channel while others still replay.
- **Replay steps at a stage**:
  - Replay rows form their own steps, with the step class `Replay`.
  - A stage's dispatcher takes ready frames in the order: decode lane, then replay, then prefill lane (RFC-0032 §3.2).
  - Replay steps run between live steps and never reorder them.
  - A sequence is never on both the replay and live paths at once: it resumes live only after its DONE.
  - `Recover` frames are routed by the dispatcher to the head's `HandshakeCore` (§4.6), or to the stage's recovery handler, never into the step loop.
  - RFC-0012 and RFC-0040 own the step-class plumbing and the `kv_write_from` descriptor field.
- **Failure during replay.** If a replay source is lost mid-replay, the head re-runs the handshake. The cascade picks the next source.

#### 4.2.1 Recurrent and linear-attention state

The State layers of hybrid families (Gated-DeltaNet- and Mamba-class, ADR-0003) hold a fixed-size state per sequence (`LayerKv::Fixed`, RFC-0001 §5.3). Each position updates the whole state in place. That state can be neither truncated to an earlier position nor recomputed for one earlier position using "existing KV". Recovery therefore never truncates it. It **restores** it, then recomputes forward.

**Default: rollback slot plus periodic checkpoints**, kept by every stage that holds State layers, per live sequence:
1. **Rollback slot.** The State-class update ping-pongs between two buffers: step k reads buffer A and writes buffer B, and step k+1 reads B and writes A.
   - The buffer not written last holds the state at the start of the last step. Restoring it undoes that step with no copy.
   - For verify chains, RFC-0022's per-position snapshots (ADR-0017 item 6) serve the same purpose while the window is open.
2. **Checkpoints.** Every `recovery.state_checkpoint_tokens` committed positions (default 1,024, aligned with RFC-0022 §7.2's checkpoint interval), the stage copies the state to a host checkpoint. It keeps the two newest. Position 0, the initial zero state, is always an implicit checkpoint.
3. **What REPORT says.** The stage reports `state_pos`, the positions absorbed, and `state_restore`, the highest position ≤ `need` it can restore to. That is `need` itself when `state_pos = need`; the rollback slot's position when that lies in `[·, need]`; otherwise the newest checkpoint ≤ `need`.
4. **Restore, then recompute.** The planner uses `min(have_pos, state_restore)` as the stage's starting point, so the cascade supplies inputs for `[state_restore, need)`, from upstream history or from the head. The stage restores, then recomputes that range. Its attention layers in the same stage read their existing KV and do not rewrite it (`kv_write_from`).

**Resume-point check, which fails closed.**
- Before a stage applies any row at position p to a State layer, during recovery or in normal operation, its dispatcher checks `state_pos(seq) = p`.
- A mismatch refuses the row and reports `StatePositionMismatch` (class `Correctness`). The head re-runs the handshake for that sequence with that stage's `state_restore` set to its newest checkpoint.
- A second mismatch escalates to rebuilding that stage's State layers from position 0, with `source = head`.
- No recurrent update is ever applied twice. For example, a ring that lost stage 1 after position `need` left stage 0, stage 0 reports `state_pos = need + 1`, restores its rollback slot at `need`, and the resume step re-applies `need` exactly once.

**Cost [derivation].** Let `S_state = Σ (State layers on the stage) elems_per_seq × bytes`, from the manifest's `KvFormula` (RFC-0001 §5.3).

| Item | Cost |
|---|---|
| Device memory | `2 × S_state` per live sequence for the ping-pong pair. Where RFC-0022 already keeps `1 + T_chain` slots for verify chains, the pair is drawn from them |
| Host memory | `2 × S_state` per live sequence for checkpoints |
| Bandwidth | the ping-pong costs none, because the update writes the whole state anyway. Checkpoints copy `S_state` per 1,024 committed tokens, so `S_state / 1,024` per token amortised |
| Recompute on a checkpoint restore | at most `recovery.state_checkpoint_tokens + T_max` positions of that stage's prefill. That is bounded, unlike a re-prefill from position 0 |

**Fallback.** Where the budget cannot hold checkpoints, the stage reports `state_restore = 0`, and recovery re-prefills that stage's State layers from the transcript (`source = head`). This is correct but costs the stage's full prefill.

**Ownership.** RFC-0022 owns the State-class slots: the ping-pong pair, the checkpoint copies, `state_pos` tracking, restore, and their budget in State-class sizing. RFC-0022 §6 implements them as `max(2, 1 + T_chain)` device slots and two `HostPinned` checkpoints shared with the prefix cache's checkpoints. RFC-0022 §6 and §13 require `kv.state_checkpoint_pages × P` to equal `recovery.state_checkpoint_tokens`. This section owns the protocol.

#### 4.3 Exactness

- A replacement stage's recomputed KV comes from the same inputs on possibly different hardware. Downstream stages keep the KV they computed from the original outputs. That is the heterogeneous-execution case ADR-0019 already covers: teacher-forced agreement within tolerance, not byte identity.
- Emitted tokens are never retracted. The continuation is what the target model produces given the emitted prefix, sampled with the same seed and counter (§4.5). Replay never uses a lossy history format: history is kept in the wire dtype, which is F32 or reference-BF16 in exact mode.
- The parity harness tests "decode N tokens, kill stage s, resume" against uninterrupted decode for every conformance family (RFC-0060).

#### 4.4 Head loss

- When the head's node is lost and ingress survives, the repair plan places a new head: a replacement node, or a survivor absorbing the head's layers and embeddings.
- The new head has no sequences. Ingress re-opens each sequence on it, once the repair `PlanCommitted` is applied and the new head reports `Ready`. The re-open is `SequenceOpen { seq (unchanged), resume: ResumePoint { emitted, rng_counter, cause: Recovery } }`, using the full transcript: prompt plus emitted tokens (RFC-0032 §3.6). This takes the place of `Reconcile`.
- The handshake then runs with `source = head`. The new head recomputes its layers for `[0, need)` from tokens, and forwards only the positions that downstream stages lack, which is usually none or a few.
- Cost: the head stage's prefill over the context. No full re-prefill is needed when stage 1 onwards survived.
- The draft model (`DraftSource::DraftModel`) is hosted by the head and re-prefills its own KV from the same transcript.

#### 4.5 Tail loss and sampler continuity

- The replacement tail rebuilds KV by replay from stage D−2's history like any stage.
- Its sampler continues from the `rng_counter` that ingress reports in `Reconcile`, which came from the last `TokenEntry` ingress received (RFC-0031 §6.5). Draws are keyed by `(request_seed, SequenceId, rng_counter)`, or by the choice index in place of the `SequenceId` for a client-supplied seed (RFC-0032 §3.7). The sequence keeps its `SequenceId` and its `choice_index`, so the next token is drawn exactly as the lost tail would have drawn it, given the same logits.
- Token-level stop sets and grammar masks come from `SEQ_META`. The head re-sends it on the first resumed frame. Grammar state is recomputed by advancing the grammar over the emitted tokens (RFC-0012).
- MTP and draft heads need no history. They restart from the first resumed step's hidden state.

#### 4.6 Protocol cores (sans-I/O)

Recovery decisions are RFC-0060 §2.2 protocol cores. They sit behind `ProtocolCore`, read time only through `Clock` and randomness only through `Entropy` (RFC-0060 amendment A1), and are driven by the deterministic simulator from M2.

```rust
/// Coordinator side: when recovery starts, which repair to commit, and when (§3.1, §6).
pub struct RepairCore { pending: BTreeMap<InstanceId, RepairState> }
pub enum RepairInput {
    Applied(LogEntry),                                     // NodeFailed, EdgeStateChanged, RunnerStateChanged, ...
    EdgeDownReport { from: NodeId, peer: NodeId },
    ProbeResult { node: NodeId, reachable: bool },
    SolveDone { instance: InstanceId, candidates: Vec<(RepairStrategy, PlacementPlan, Nanos /*t_resume*/)> },
    TimerFired(RepairTimer),                               // confirm_after, re-plan back-off
}
pub enum RepairOutput {
    Submit(ClusterCommand),                                // RecoveryStarted via ReportRecovery, CommitPlan
    Probe { node: NodeId },
    StartSolve { instance: InstanceId, reason: ReplanReason, constraints: RepairConstraints },
    SetTimer { timer: RepairTimer, at: MonoTime },
}

/// Head runner side: the recovery handshake of §4.2 for one instance. Built on the pure planner
/// `plan_replay(reports, need) -> ReplayPlan`. Runs on the head's housekeeping thread; it never
/// blocks a dispatcher.
pub struct HandshakeCore { seqs: BTreeMap<SequenceId, SeqRecovery>, phase: Phase }
pub enum HandshakeInput {
    Reconcile { seq: SequenceId, emitted: u32, rng_counter: u64 },  // from ingress (§4.2)
    RecoverFrame(RecoverMsg),                                        // QUERY reports, DONE, from the ring return
    ReplayProgress { seq: SequenceId, through: u32 },
    TimerFired(HandshakeTimer),                                      // per-phase deadlines; re-run on loss
}
pub enum HandshakeOutput {
    SendRecover(RecoverMsg),                                         // QUERY or PLAN, around the ring
    StartReplay { seq: SequenceId, source: StageIx, range: Range<u32> },
    Resume { seq: SequenceId, at: u32, first_step: u64 },
    Report(RecoveryOutcome),                                         // to the daemon, which submits ReportRecovery
    SetTimer { timer: HandshakeTimer, at: MonoTime },
}
```

The choice rule of §5.1 and the strategy rule of §6.2 are pure functions that these cores call. The L6 property suites drive the public `Input` enums with random multi-stage losses, and the simulator's fault injection (RFC-0060 §9) checks the chaos-plan outcomes below.

### 5. Migration and planned transitions

A planned change retires an epoch while its runners are still alive: a re-plan, an unload, a node leaving gracefully, an upgrade drain. RFC-0001 §7.5 lists four options per sequence. This section turns them into a decision rule (ADR-0312).

#### 5.1 The per-sequence choice

For sequence q on retiring epoch e, moving to target epoch e′ (or another instance):

```text
t_finish   = remaining(q) · t_tok(e)              remaining(q) = min(limits.max_tokens − emitted, p90_len(class) − emitted)
t_kvmig    = Σ_moved stages KV_bytes(q, layers) / β_bulk(holder → new) + α        (needs old holders alive and e′ loaded)
t_replay   = H_bytes(q) / β + t_prefill(moved layers, ctx(q))                      (needs a surviving upstream source in e′)
t_reprefill= TTFT_{e′}(prompt_len + emitted)                                      (always available)
feasible   = { finish  if t_finish ≤ drain_deadline,
               kvmig   if make-before-break and β_bulk known,
               replay  if e′ shares the upstream stages that hold q's history,
               reprefill }
choose argmin over feasible of (t + penalty), with penalty(finish) = 0 and the others charged their
interruption (the client sees a pause of t), so `finish` wins whenever it fits the deadline
```

- `p90_len(class)` is the learned 90th-percentile generation length for the request's API class (chat, tool call, completion), seeded at 512 tokens. `KV_bytes` comes from the manifest's `KvFormula` (RFC-0001 §5.3). `TTFT` comes from RFC-0032 §3.8.
- The drain deadline is `Drain::deadline`, which defaults to `recovery.drain_deadline` = 60 s in latency mode and 300 s in throughput mode.

#### 5.2 KV migration

- KV migration runs only under make-before-break, because the new epoch must already be loaded.
- The old holder of each moved layer range streams RFC-0022 `KvChunk`s (one layer, one page class, one token range each), layer by layer, to the new holder. They travel as `KvChunk` frames (RFC-0031 §6.11) on a `Replay`-purpose channel at Bulk priority (RFC-0032 §4.5). The receiver checks `kv_settings_hash` and the chunk hash, then injects pages as they arrive (`KvStore::import`, RFC-0022 §10). A rejected import falls back to replay or re-prefill.
- The sequence's switch is coordinated by the old head:
  1. it stops the sequence at a step boundary;
  2. it waits until every new holder reports `KvReady { seq, to: need }`;
  3. the new head resumes at `need` on e′, with step numbers continuing.
- Recurrent and linear-attention state moves the same way, with `layer_kind = Fixed`.

#### 5.3 Re-prefill migration

- Ingress re-opens the sequence on e′ or on another instance with `SequenceOpen { resume: ResumePoint { emitted, rng_counter, cause: Migration } }` and the full transcript (RFC-0032 §3.6). The sequence keeps its `SequenceId` and seed, so its random stream continues (RFC-0032 §3.7). Ingress then cancels it on e.
- The tail of the new instance does not re-emit tokens. Its first emitted token is token `emitted + 1`, and `max_tokens` counts the emitted ones.
- This is the universal fallback, `multi-device-inference.md` §15.8 item 2. It also covers moving a sequence to a different instance, for example a replica with spare capacity. That is the datacenter "request migration" behaviour, implemented without a separate mechanism.

#### 5.4 Ending at the deadline

A sequence that has not finished or moved when the drain deadline passes ends with `finish = Error(EpochRetired)` (class `Transient`, `retry: OnNewPlan`). Ingress maps it to the API's in-stream error and recommends a retry (RFC-0050). This is the only case in which a planned transition is client-visible, and the default deadline makes it rare in latency mode.

### 6. Re-placement after a loss

#### 6.1 Repair requests

On `RecoveryStarted`, the coordinator's reaction asks the solver for a repair plan:

```rust
/// Owner: this RFC (RFC-0001 §15).
pub enum ReplanReason {
    StageLost { stages: SmallVec<[StageIx; 2]>, nodes: SmallVec<[NodeId; 2]> },
    RunnerFailed { node: NodeId, device: Option<DeviceId> },
    EdgeLost { src: NodeId, dst: NodeId },          // node alive, path gone: relink over another transport
    NodeJoined { node: NodeId },
    Drift { node: NodeId, term: DriftTerm },
    MemoryPressure { node: NodeId },
    /// Workload terms off what the plan assumed: expert hit rates, a cache tier below its hit
    /// curve, speculative acceptance (RFC-0010 §7.1, §7.3, which owns the term list:
    /// ExpertMiss | ExpertTierShortfall { tier } | SpeculativeAcceptance). The current curves
    /// arrive separately in `ReportWorkload` (RFC-0032 §6.5).
    WorkloadDrift { term: WorkloadTerm },
    /// The changed keys, so the dashboard and logs can say which change caused the re-plan
    /// (RFC-0054 amendment A4).
    Config { keys: Vec<KeyPath> },
    User,
    HealOrphans,
    DegradedExit,                                    // §7.2
}

/// Passed to RFC-0011's solver with StageLost / RunnerFailed / EdgeLost.
pub struct RepairConstraints {
    pub keep_shards: bool,                 // true: every surviving runner keeps its shard (relink only)
    pub candidates: RepairCandidates,      // which strategies to evaluate (default: all)
    pub objective: RepairObjective,        // MinTimeToResume (default) | MinSteadyStateTtok
    pub max_move_bytes: Option<Bytes>,
}
pub enum RepairStrategy {
    /// Restart the lost runner on the same node (runner crash; node alive).
    RestartInPlace,
    /// A node outside the ring (or with spare capacity) loads the lost layer range.
    Replace { node: NodeId },
    /// Survivors absorb the lost layers into their slower tiers (ExtendShard; degraded).
    Absorb { into: SmallVec<[NodeId; 2]> },
    /// Full re-solve; surviving shards may change (used when the others are infeasible).
    Resolve,
}
```

#### 6.2 Choosing the strategy

The reaction's `StartSolve` calls RFC-0011's `solve_repair(input, current, &RepairConstraints)` (RFC-0011 §12.4). It returns one `(RepairStrategy, PlacementPlan, t_resume)` candidate per feasible strategy, already filtered by `max_move_bytes`. Each repair strategy is a plan with predicted steady-state `t_tok`. RFC-0033 adds the **time to resume**:

```text
t_resume(strategy) = t_solve
                   + Σ_bytes_missing_on_target / β_fetch(best source)      (RFC-0041 source choice: local store, cluster holders, Hub)
                   + t_pack + t_load(bytes / W_read)                       (RFC-0041, SC-23)
                   + t_warm                                                (capture, canary; seconds)
                   + t_replay(Σ affected sequences)                        (§4.2; per sequence, in priority order)
```

The default rule:
1. Pick the strategy with minimal `t_resume`.
2. Among strategies within 20 % of the best, prefer the one with better steady-state `t_tok`.
3. `RestartInPlace` is preferred for a runner crash whenever the node is healthy and the pack is valid, which is the common case. A restart costs `t_load + t_warm + t_replay` with no fetch.

`Resolve` is used only if no `keep_shards` strategy is feasible. It drains the survivors under break-before-make, and its sequences re-prefill.

#### 6.3 Fetch priority

- A replacement's shard fetch runs at `BlobRequest::priority = replacement` (RFC-0031 §6.10). It preempts background distribution.
- The bulk pacing of RFC-0031 §7.5 still protects surviving rings that share a link.
- Chunk sources are chosen by RFC-0041 from `ChunksAnnounced` inventories: local store first, then cluster holders by measured β, then the Hub.

#### 6.4 Timeline

```mermaid
sequenceDiagram
  autonumber
  participant S as Stage s−1 (survivor)
  participant L as Stage s (lost)
  participant K as Coordinator
  participant R as Replacement / absorber
  participant H as Head
  participant I as Ingress
  L--xS: link silent
  S->>S: channel Down after failover timeout (2 s)
  S->>K: ReportEdgeDown (and the next stage too)
  K->>K: own probe fails → RecoveryStarted; solve repair (≤ 2 s)
  K-->>I: RecoveryStarted (log)
  I->>H: Reconcile { emitted, rng_counter } per sequence
  K->>K: NodeFailed, or confirm_after → commit repair plan (Relink, ExtendShard or new runner)
  R->>R: fetch missing chunks, pack, load, warm (canary)
  S->>R: channels for the new epoch (Relink)
  H->>R: Recover QUERY / PLAN (around the ring)
  S->>R: Replay frames from history
  R->>H: Recover DONE (via tail)
  H->>H: resume decode at `need`
```

### 7. Degraded modes

#### 7.1 Catalogue

RFC-0001 §5.11 `RunnerState::Ready { degraded: Option<Degradation> }`. The type is defined here:

```rust
/// A set: a runner may be degraded in several ways at once. Every flag is visible in the
/// dashboard and in response metadata (RFC-0001 §10.2). None changes output (D3).
pub struct Degradation { pub flags: DegradeFlags, pub detail: SmallVec<[DegradeDetail; 2]> }
bitflags! { pub struct DegradeFlags: u32 {
    const RELINKING        = 1 << 0;   // switching neighbours in place (RFC-0032 §4.4)
    const RECOVERING       = 1 << 1;   // replaying KV for some sequences (§4.2)
    const ABSORBED_STAGE   = 1 << 2;   // hosts a lost stage's layers in slower tiers (§7.2)
    const TRANSPORT_DEMOTED= 1 << 3;   // an edge runs on a lower-ranked transport (RFC-0031 §7.4)
    const PROVIDER_FALLBACK= 1 << 4;   // an op fell back to the ggml provider (RFC-0040)
    const SPECULATION_OFF  = 1 << 5;
    const CAPTURE_OFF      = 1 << 6;   // graph capture unavailable; launches replayed by the executor
    const CACHE_REDUCED    = 1 << 7;   // expert cache or KV hot window shrunk under pressure (RFC-0020)
    const HISTORY_PARTIAL  = 1 << 8;   // some history dropped under pressure (§4.1.2)
    const HISTORY_OFF      = 1 << 9;   // history disabled for this instance (§4.1.2)
}}
pub enum DegradeDetail {
    Absorbed { stages: SmallVec<[StageIx; 2]>, tiers: SmallVec<[TierKind; 2]> },
    Demoted  { edge: LinkId, from: TransportKind, to: TransportKind },
    Fallback { op: OpKind, provider: ProviderId },
    Reduced  { device: DeviceId, bytes: Bytes },
}
```

#### 7.2 Absorbing a lost stage

- `Absorb` gives one or two survivors an extra stage through `ExtendShard` (RFC-0032 §4.1). The survivor's runner keeps its own stage, unchanged, and loads the new layers into whatever tiers the repair plan assigns. Those are usually `HostPinned` with CPU execute-in-place, or `Nvme` with an expert cache for MoE layers.
- The survivor's runner now hosts two stages, which is RFC-0001's "degraded modes only" case for multiple stages per node. With its own stage adjacent to the absorbed one, the hop between them disappears and becomes a local hand-off.
- **Exit.** A degraded plan raises `RequestReplan { DegradedExit }` whenever the graph changes, for example when the lost node returns (`NodeAlive`) or a new node joins. The re-plan is adopted under the normal hysteresis and payback rules (RFC-0001 §7.2), except that the minimum-gain threshold is halved for a degraded plan (`replan.degraded_min_gain` = 10 %). Users notice degraded token rates, so the bar for leaving one is lower.

### 8. Stragglers, drift and runner supervision

#### 8.1 Stragglers

- Each stage records its step time per kind (decode, verify, prefill chunk) in the telemetry ring (RFC-0001 §13). The runner compares a 10 s EWMA with `StageAssignment::predicted_step`.
- A stage is a **straggler** when its EWMA exceeds `straggler.factor` (1.25) × predicted for `straggler.window` (30 s). Typical causes: thermal throttling, a game or video export taking the GPU, a degraded Wi-Fi hop, an NVMe drive throttling.
- The runner then reports `Drift` with `MeasureSource::Online` values (RFC-0001 §7.4, RFC-0010). The coordinator handles it as a deferred re-plan under hysteresis and payback.
- **A slow but alive stage is never killed or restarted mid-sequence** (`multi-device-inference.md` §15.8 item 4). Killing it converts slowness into a replay, which is strictly worse.
- In throughput mode, routing also deprioritises an instance with a straggling stage, through the `step_p50` field in load digests (RFC-0030 §5.6).
- **Stall versus slow.** A stage that makes *no* progress is the supervisor's business (§8.2). A stage that makes slow progress is the solver's.

#### 8.2 Runner supervision defaults

RFC-0001 §3.4 fixes the mechanism and leaves the defaults to this RFC, in RFC-0054's schema. ADR-0015 item 5 gives the starting points.

| Key | Default | Meaning |
|---|---|---|
| `runner.heartbeat` | 100 ms | interval at which the runner bumps its shared-memory heartbeat |
| `runner.liveness_timeout` | 3 s (suspend-aware clock) | no heartbeat → hung → kill |
| `runner.stall_factor` / `runner.stall_floor` | 20 × predicted p99 step / 10 s | `Ready` with live sequences and no step progress for max(floor, factor × step) → stalled → kill |
| `runner.load_stall` | 3 × (shard bytes / measured `W_read`) + 60 s | stall bound during `Fetching` to `Warming` (fetching excludes time waiting on remote sources that report progress) |
| `runner.restart_backoff` | 1 s doubling to 60 s, with jitter | per (instance, node) |
| `runner.restart_budget` | 5 failures in 10 min | then `ReportRunnerFailed`; the repair plan excludes the device |
| `runner.daemon_silence` | 2 s (suspend-aware clock) | the daemon's heartbeat counter in the shared-memory header has not advanced for this long → daemon lost → orphan mode. The daemon bumps the counter from a dedicated thread that never blocks on the `ctl` runtime, so a busy control plane does not look dead. End-of-file on the IPC command socket triggers orphan mode at once. If orphan mode was entered on a stalled heartbeat while the command socket stayed connected, and the heartbeat resumes, the runner returns to normal operation and resumes admitting, with no adoption. It reports the stall's length in a `RunnerStateChanged` note, counted by doctor RUN-006 (RFC-0052 §6.5, RFC-0053) |
| `runner.orphan_grace` | 30 s | how long an orphaned runner waits for adoption. It keeps serving remote-ingress sequences, admits no new ones, and listens on `<runtime dir>/adopt/<instance>.sock`. If no daemon adopts it in time, it drains for at most `recovery.drain_deadline` and exits (RFC-0052 §6.5) |
| `runner.device_reprobe` | on `Device` class failures | re-probe the device before restarting; if it is gone, the resource graph changes (ADR-0015 item 5) |

**Orphan rule, runner side** (RFC-0052 §6.5 owns the rule and the per-OS service settings):
- Runners install no parent-death signal and no exit watch. On Linux there is no `PR_SET_PDEATHSIG`. On macOS the runner calls `setsid()` so a launchd process-group kill misses it. On Windows a runner started under a per-user logon task breaks away from the task's job into the daemon's own job, which has no kill-on-close.
- A runner therefore outlives its daemon by at most `runner.orphan_grace + recovery.drain_deadline`.
- Where re-adoption is unavailable (containers, embedded mobile runners, a changed `ipc_protocol`, a job that forbids breakaway), the update path drains and restarts instead (RFC-0052 §7.3).
- Peers see none of this. The data plane keeps running, and a planned restart is announced with `Leave { reason: Restart }` (RFC-0030 §5.9).

A runner restart loses the runner's KV and history. Its sequences recover as for a lost stage, with `RestartInPlace` normally chosen (§6.2). The restarted runner skips `Fetching` and `Packing` because its pack is valid (RFC-0001 §3.3).

#### 8.3 Transport errors

Link faults are recovered inside the transport, without replay (RFC-0031 §7.4, ADR-0307):
- reconnect on the same backend;
- after 3 faults in 60 s, demote the edge to the next backend by measured cost;
- resend the retained frames.

Only a link that stays down past `link.failover_timeout` *and* whose node is unreachable becomes a stage loss (§3.1). A link that stays down while the node is reachable through another path becomes `ReplanReason::EdgeLost`. The repair plan then relinks the hop over another transport with every shard kept.

### 9. Coordinator and ingress failure; standby options

#### 9.1 Coordinator

- **Unplanned loss.** An election follows within about 4.5–5 s (RFC-0032 §7). During the gap:
  - in-flight sequences on intact rings continue;
  - ingress keeps routing from its replica;
  - commands are retried until a coordinator acknowledges them.

  A stage loss that coincides with the coordinator loss waits for the election before its repair plan commits. `RecoveryStarted` and the repair solve then run on the new coordinator, which rebuilds the inputs from its replica.
- **Tie-break: "hosts no stage"** (open question 4, answered; OQ-NET-25). Within a capability class, a coordinator-eligible node that hosts no stage ranks above one that does (ADR-0009 item 4, revised 2026-10-20). A coordinator that also hosts a stage makes that stage's recovery wait for an election, and the tie-break avoids it whenever the class allows. It never picks a weaker node, because it acts only inside a class. Only nodes whose certificate carries `COORDINATOR` are ranked (RFC-0034 §5.2), so with the keyholders-only default most home clusters have one or two eligible nodes and the tie-break rarely fires. In a two-node cluster both nodes host stages and nothing changes. DIST-E15-T05 measures the recovery delay it removes.
- **Planned loss.** Handover takes about 0.5 s (RFC-0032 §7.7).
- **Standby coordinator.** Every eligible node is a standby, because each holds the full replica and log. No extra configuration is needed. `coordinator.prefer` lets an operator pin the preferred node, for example an always-on mini PC that hosts no stage.

#### 9.2 Ingress

- **No replicated ingress.** The ingress is almost always the client's own machine, so when it dies the client usually dies with it. The common failure is a dropped link with both ends alive, which the detach grace below covers. A replicated transcript would be a large piece of work, and a plaintext copy of the conversation on other members, for the rare case.
- **Detach grace.** When the client's connection drops but the ingress node is alive, the ingress keeps the sequence for 60 s [target]:
  - it sends `Pause` to the head (RFC-0032 §3.6), so the ring stops issuing steps for the sequence while every stage keeps its KV, its activation history and its retention;
  - tokens already generated stay in the ingress's transcript;
  - a client that reconnects within the grace re-attaches by SSE `Last-Event-ID` or Responses `starting_after` (RFC-0050 owns the API); the ingress replays the tokens it missed and sends `Resume`;
  - when the grace ends with no re-attach, the sequence ends as cancelled and the head sends `Release` around the ring, persisting the session if it is pinned or asked for (RFC-0022);
  - a stage loss during the grace is recovered like any other (§4), since the sequence is paused, not gone.
- When the ingress node is lost, the tail sees its `Tokens` channel to that node fail. The coordinator records `NodeFailed`. In either case, the tail ends every sequence whose ingress was that node with `finish = Cancelled`, on the ring-return frame.
- The head then sends `Release` around the ring, persisting the session if it is pinned or if the request asked for persistence (RFC-0022). It frees KV and history.
- Resources are freed within `link.failover_timeout` plus one ring traversal [target: ≤ 3 s on LAN].
- **A client that reconnects** to another node continues from the persisted session if there is one and the session's key survives (RFC-0034 §13). Otherwise it starts again. Stored responses live on the ingress only, with no copy on holders, so after an ingress loss `previous_response_id` returns a clear error and the client resends its input. The API layer, RFC-0050, maps this to a retryable error on the reconnect.

#### 9.3 Standby options

`InstancePolicy` gains `standby: Standby`. The default is `None`.

```rust
pub enum Standby {
    None,
    /// Named nodes pre-fetch and pack (not load) the shards of the stages they would replace,
    /// so a Replace strategy skips Fetching and Packing. Costs disk and bulk bandwidth only.
    Warm { nodes: SmallVec<[NodeId; 2]> },
    /// A second instance of the model, solved on disjoint nodes where capacity allows; on a loss,
    /// sequences migrate to it by re-prefill (§5.3) without waiting for a repair.
    Replica,
}
```

- `Warm` changes SC-28's `S_delta/β_fetch` term to zero. In the §Performance example it cuts time to resume from about 60 s to about 20 s. Disk is cheap, so the dashboard recommends `Warm` whenever a non-ring node has free disk at least 1.2 × the largest stage shard. Pre-packing stays on non-ring nodes unless DIST-E14-T16's repair timings show it pays on ring nodes.
- `Replica` suits the household-server scenario S5, where a second instance also adds throughput.

**Second copies on the predicted absorber** (open question 1, answered). `Warm` needs a node outside the ring, and most two- and three-node home clusters have none, so without more a cold repair refetches from the Hub (≈ 220 s in the §Performance example). By default, therefore, each stage shard's source chunks get one evictable second copy:
- **Where.** On the node the repair solve would absorb that stage into, as computed for the committed plan (§6.2): the predicted absorber. If no absorber is feasible, there is no copy.
- **How.** Over the LAN, from a cluster holder, at `BlobRequest` priority `background` (RFC-0031 §6.10, §7.5), so it never delays serving. It is never fetched from the Hub for this purpose.
- **Only on spare disk.** A copy is admitted only if the target volume keeps at least 2 × `store.min_free_bytes` free after writing it, so a copy is never written only to be evicted by the next fetch's admission (RFC-0041 §4.4). It also stays within the per-node cap `store.standby_max_bytes`, where 0 turns copies off (RFC-0041 §9).
- **Evicted first.** Copies are held under a standby reference, not a shard reference. They are the first thing reclaimed when a node needs space, before covered source ranges and LRU eviction, and the first chunks dropped when the plan they were predicted for is replaced (RFC-0041 §4.4).
- **Not on battery-class nodes**, which leave and sleep.
- **Visible.** The dashboard shows the disk each node gives to second copies (RFC-0051).
- **Effect.** A cold repair then reads the lost stage's bytes from the absorber's own disk, which removes the fetch term the way `Warm` does except for packing: in the §Performance example, time to resume falls from ≈ 220 s from the Hub to ≈ 26–30 s [estimate: the ≈ 21–25 s warm figure plus ≈ 5 s of packing]. DIST-E14-T16 measures cold repair with and without copies.
- **Why not a replication factor of 2.** Two copies of every chunk would double cluster disk for the largest models, and a Kimi-K3-class model is ≈ 1.49 TB (vision §6.2). "Copy while there is room, evict first" keeps the benefit where disk allows and costs nothing where it does not.

### 10. Node join mid-session

- A node that joins (`NodeAdmitted`) or returns (`NodeAlive`) triggers a deferred background solve (RFC-0032 §8.1).
- The new plan is adopted only if it passes hysteresis and payback (RFC-0001 §7.2):
  - predicted `t_tok` improves by at least `replan.min_gain` (20 %), or 10 % from a degraded plan;
  - its moved bytes divided by measured link rates are recovered within `replan.payback_horizon`.
- **In latency mode** the adoption waits for an idle window (no live sequences) for up to `replan.max_defer` (10 min).
  - A long agent session may never go idle. When the deadline passes, each live sequence takes the per-sequence choice of §5.1. A sequence that fits `recovery.drain_deadline` finishes on the retiring epoch. The others move at step boundaries, usually by KV migration under make-before-break when memory allows, otherwise by replay or re-prefill (revised 2026-10-22, note below).
  - Persisted sessions are re-homed by RFC-0022.
- **In throughput mode** the plan is adopted by make-before-break when capacity allows. Otherwise it waits for the instance's load to fall below half its admission limit.
- A returning node that held the lost stage of a degraded plan usually still has its pack. If so, `Replace` back onto it costs only `t_load + t_warm + t_replay`, and the solver's move cost reflects that through `ChunksAnnounced`.

*Note 2026-10-22 (#1748):* the deadline bullet said live sequences "are moved at step boundaries by the per-sequence rule of §5.1, usually KV migration under make-before-break when memory allows, otherwise replay". That left out §5.1's `finish`, which wins whenever a sequence fits the drain deadline, and re-prefill, which is always feasible. RFC-0011 §12.2 now refers to this section for the deadline; before, it waited for the next idle window with no bound.

### 11. Commands, events and types added

These are added to the RFC-0032 §6.5 catalogue under RFC-0001's extension rules.

| Kind | Variant | Notes |
|---|---|---|
| Command | `ReportEdgeDemoted { edge, from, to, cause }` | from a runner's daemon after a demotion (RFC-0031 §7.4) |
| Command | `ReportRecovery { instance, epoch, outcome: RecoveryOutcome }` | from the head's daemon when the handshake ends: sequences resumed, replayed, re-prefilled, failed |
| Event | `EdgeDemoted { edge, from, to, cause }` | the solver uses the new transport's cost in the next plan |
| Event | `RecoveryStarted { instance, epoch, lost: SmallVec<[StageIx; 2]>, cause: ReplanReason }` | makes the instance unroutable for new admissions in every ingress replica; ingress sends `Reconcile` |
| Event | `RecoveryCompleted { instance, epoch, resumed: u32, failed: u32, took_ms: u32 }` | restores routability; shown in the dashboard and response metadata |
| Event | `InstanceDegraded { instance, epoch, flags: DegradeFlags }` | set and cleared as plans change |
| Sequence control | `SequenceControl::Reconcile { seq, emitted: u32, rng_counter: u64, recent: SmallVec<[u32; 8]> }` | ingress → head on the Sequence channel (RFC-0032 §3.6) |
| Data-plane message | `KvReady { seq, to }` | new holder → old head during KV migration, carried as a `Recover` entry |
| Type | `ReplanReason`, `RepairConstraints`, `RepairStrategy`, `Degradation`, `Standby` | §6.1, §7.1, §9.3 |

`RecoveryRecord`, in `InstanceRecord::recovery` (RFC-0032 §6.2): `{ started_index, cause, lost, strategy, plan_epoch, sequences_pending }`. It is rebuilt from facts after a heal. The head's daemon re-reports an in-progress recovery with `ReportRecovery`.

### 12. Configuration

| Key | Scope | Default | Notes |
|---|---|---|---|
| `recovery.history` | instance | `auto` | `on` \| `off` \| `auto` (off when the predicted footprint exceeds the budgets; §4.1.2) |
| `recovery.history_ram` | node | min(2 GiB, 5 % of RAM) per runner | |
| `recovery.history_disk` | node | 16 GiB | |
| `recovery.state_checkpoint_tokens` | instance | 1,024 | §4.2.1 |
| `recovery.confirm_after` | cluster | 4 s (2 × `link.failover_timeout`) | §3.1 |
| `recovery.drain_deadline` | instance | 60 s latency mode, 300 s throughput mode | §5.1 |
| `recovery.p90_len_seed` | cluster | 512 tokens | §5.1 |
| `recovery.strategy_slack` | cluster | 20 % | prefer better steady state within this slack of the fastest resume (§6.2) |
| `replan.min_gain` / `replan.degraded_min_gain` | cluster | 20 % / 10 % | RFC-0001 §7.2; §7.2 |
| `replan.payback_horizon` | cluster | 10 min | |
| `replan.max_defer` | cluster | 10 min | §10 |
| `straggler.factor` / `straggler.window` | cluster | 1.25 / 30 s | §8.1 |
| `runner.*` | node | §8.2 | |
| `standby` | instance | `none` | §9.3; second copies on the predicted absorber are separate and on by default, capped by `store.standby_max_bytes` (RFC-0041 §9) |

Every key above is declared, with each scope it may be set at, in its owning crate's `#[derive(ConfigSection)]` section (RFC-0054 §10.3), and cluster-scope keys carry `introduced_at`. RFC-0054's registry is generated from those sections (§5.1). So the instance-scope keys `recovery.history`, `recovery.state_checkpoint_tokens`, `recovery.drain_deadline` and `standby` appear in the registry and can be set per instance through `InstancePolicy.domain`, which the policy digest covers (RFC-0054 §4.5, §5.4).

### 13. Error codes

`Domain::Recovery`, range 0x0400–0x04FF.

| Code | Name | Class | Client sees |
|---|---|---|---|
| 0x0401 | `RecoveryInProgress` | `Transient` | keep-alives; on deadline, a retryable timeout |
| 0x0402 | `ReplayNoSource` (no history and no transcript) | `Link` | stream error, retryable |
| 0x0403 | `EpochRetired` (drain deadline passed) | `Transient` | stream error, `retry: OnNewPlan` |
| 0x0404 | `IngressLost` | `Cancelled` | — (connection gone) |
| 0x0405 | `NoRepairCapacity` (every strategy infeasible) | `Capacity` | stream error with `Retry-After`; instance unroutable |
| 0x0406 | `RestartBudgetSpent` | `Device` | — (handled by re-plan) |
| 0x0407 | `HistoryDropped` | `Capacity` | — (informational; recovery cascades) |
| 0x0408 | `KvMigrationFailed` | `Link` | falls back to replay or re-prefill; nothing visible |
| 0x0409 | `StatePositionMismatch` | `Correctness` | nothing (the handshake re-runs; §4.2.1) |

## Performance model and budgets

All figures are **[estimate]** with derivations, or **[target]**. None are measurements.

**History memory.** Per sequence and per sending boundary: `bytes = context × streams × hidden × wire bytes`.

| Model, wire dtype | Per token | 4K context | 32K | 128K | Compared with that stage's KV |
|---|---|---|---|---|---|
| Llama-3.3-70B, hidden 8192, F32 | 32 KiB | 128 MiB | 1 GiB | 4 GiB | two equal stages each hold 160 KiB/token of KV (cost model §3.6), so history is about 20 % of the stage's KV |
| DeepSeek-V3, hidden 7168, F32 | 28 KiB | 112 MiB | 896 MiB | 3.5 GiB | MLA KV is 68.6 KiB/token for the whole model, about 34 KiB per stage in two stages, so history is about 80 % of the stage's KV |

- RFC-0001 §7.5 quotes 64 MiB at 4K. That figure assumes F16, which exact mode does not use (ADR-0010). The F32 figures above are the right ones.
- Write bandwidth equals the stage's outbound frame bytes. That is negligible at decode: 32 KiB × 6.5 tok/s ≈ 210 KB/s. At prefill it equals the hop's bytes and is written to RAM.
- **Default budget.** 2 GiB of RAM per runner covers 16 concurrent 4K Llama-70B sequences, or 2 at 32K. Longer contexts spill to NVMe at sequential write speed.

**SC-28 worked example.** Example (c) as three stages (cost model §4c): RTX 4090 PC with 19.3 GB of Llama-3.3-70B Q4 (≈ 36 layers) → MacBook Pro M4 Pro with 12.0 GB (≈ 23 layers) → Strix Halo with 11.2 GB (≈ 21 layers), on 2.5 GbE. The MacBook is lost mid-decode at a 4K context.

| Term | Value | Basis |
|---|---|---|
| Detection to recovery start | ≈ 2–3.5 s | failover timeout plus one probe (§3.1) |
| Solve | ≤ 2 s, overlapped with confirmation | SC-31 |
| Commit | ≈ 6 s after the failure | `confirm_after` = 4 s after the first `EdgeDown` |
| Strategy | `Absorb` into the Strix (11.2 + 12.0 = 23.2 GB, inside its ≈ 96–100 GB GPU-usable memory): this is exactly example (c1)'s two-stage plan | solver |
| Fetch 12.0 GB: warm standby on the Strix | 0 s | §9.3 |
| Fetch: cold, from a cluster holder over 2.5 GbE | 12.0 GB / 294 MB/s ≈ 41 s | only if some survivor holds the chunks |
| Fetch: cold, from the Hub at 500 Mb/s | 12.0 GB / 62.5 MB/s ≈ 192 s | ADR-0016 fetches each chunk once per cluster, by the node that needs it, so the dead MacBook may have been the only holder. With a second copy on the Strix, the predicted absorber (§9.3), this term is 0 s and only the pack remains |
| Pack and load | ≈ 2 s load at about 6 GB/s, plus about 5 s pack when cold | SC-23 |
| Warm (capture, canary) | ≈ 2–5 s | assumption |
| Replay: history 4K × 32 KiB = 128 MiB from the 4090 stage | 0.46 s, hidden behind compute | 294 MB/s |
| Replay compute: 23/80 of 591 TFLOP on the Strix iGPU at 15 TFLOPS | ≈ 11.3 s | cost model §3.5, §4c2 |
| **Time to resume** | **≈ 21–25 s warm; ≈ 66 s from a cluster holder; ≈ 220 s from the Hub** | sum |
| Steady state after recovery | 24.1 + 129.6 ms = 153.7 ms per token, ≈ 6.5 tok/s | the MacBook added only 6.5 → 6.7 tok/s (cost model §4c1), so losing it costs almost nothing in steady state |

Re-prefill instead of replay would take the full ≈ 22 s two-stage prefill (RFC-0032 §Performance) instead of 11.3 s. Replay halves the compute part, and its advantage grows with the share of layers that survive.

**Other cases [estimate, same rig].**
- **Head loss, ingress alive:** re-prefill only the head's 36 layers. That is 36/80 × 591 TFLOP on a replacement 4090-class GPU at 100 TFLOPS ≈ 2.7 s, plus fetch and load.
- **Runner crash on the 4090 PC, restart in place:**
  - detection < 0.1 s;
  - load 19.3 GB from NVMe at ≈ 6–7 GB/s ≈ 3 s;
  - warm 2–5 s;
  - replay of 36 layers ≈ 2.7 s.

  Total ≈ 8–11 s.
- **Link flap:** a pause of at most `link.failover_timeout` (2 s), with no replay.
- **Planned KV migration of one stage** (23 layers, 4K tokens): 23 × 4 KiB per layer per token × 4,096 = 368 MiB at 294 MB/s ≈ 1.3 s pause.
- **Ingress loss cleanup:** ≤ 3 s [target].

## Alternatives considered

- **History at the head only**, as RFC-0001 §7.5 words it. To replay stage s ≥ 2, the head would need stage s−1's outputs. That means every stage forwards a copy of every frame to the head (D − 2 extra sends per token, doubling prefill traffic), or else only stage 1 can be replayed. Sender-side retention costs nothing on the wire. Rejected (ADR-0311).
- **No history; always re-prefill.** Simpler, but every recovery costs a full prefill, about twice the compute in the example and worse when the lost stage is a small share of the model. Kept as the cascade's limiting case and as the throughput-mode fallback.
- **Replicating history to a standby node** (RFC-0001 §17 question 2). This is a network copy of every frame. With sender-side retention no stage is a single point of failure for replay, so replication buys nothing. Rejected.
- **Periodic KV checkpoints** to NVMe or a peer. KV is 1.25–5× larger than history per stage in the examples, and must be re-written as it grows. History is the cheaper checkpoint. Rejected.
- **Hot standby stages** with fully loaded duplicates. These double memory, which is the scarcest resource in this design. Offered only as `Replica` for throughput deployments.
- **BF16 history to halve memory in exact mode.** Replay inputs would then differ from the originals by more than device rounding, a deliberate approximation that D3 forbids outside approximate mode. NVMe spill handles memory instead. Rejected.
- **Killing stragglers.** This turns slowness into a replay. Rejected (§8.1).

## Testing and validation

**Unit and property tests (RFC-0060).**
- *Handshake planner* (a pure function from reports to plan):
  - a source always exists, with the head as last resort;
  - the source is the nearest upstream stage whose history covers the range;
  - drop thresholds are correct;
  - no stage is asked to compute a position it does not need;
  - property-tested over random have/history vectors and random multi-stage losses.
- *History store:*
  - truncation stays consistent with KV truncation across verify windows;
  - spill and restore are byte-exact;
  - drops advance `from` and set `HISTORY_PARTIAL`.
- *Choice rule (§5.1):* property tests that `finish` is chosen whenever it fits the deadline, and that `reprefill` is always feasible.
- *Repair choice (§6.2):* deterministic on identical inputs.

**Exactness.** The parity harness decodes N tokens, kills stage s, resumes, and compares the continuation with uninterrupted decode under teacher forcing. It runs for each conformance family, hybrid State-layer families included, for s ∈ {head, middle, tail}, with F32 and BF16 wire. KL and top-1 must stay within RFC-0042's tolerances (ADR-0019).

**Chaos test plan.** The suite runs nightly on the `testkit` simulated fabric with all scenarios, and weekly on the M3 hardware rigs (scenario (c) rig and two-Mac TB5 rig) for the scenarios marked ●. Faults are injected by a chaos agent in the daemon, behind a test-only build feature: process kill, link block, latency and loss injection, clock jump, NVMe throttle (I/O engine delay), memory pressure signal, and provider fault injection.

| # | Scenario | Injection | Expected outcome | Target |
|---|---|---|---|---|
| 1 ● | middle stage node lost mid-decode, survivors have capacity; run for a dense family **and a hybrid (State-layer) family** | kill daemon and runners | resumes with no client error; time within the §Performance budget; on the hybrid family, every State layer applies each position exactly once (resume-point check counters) | SC-28 |
| 2 ● | head node lost, ingress elsewhere; dense and hybrid families | kill | head-range re-prefill from the transcript; resumes | §4.4 |
| 3 ● | tail node lost, including the case where ingress missed the last token; dense and hybrid families | kill | replay into the new tail; State stages restore from the rollback slot; sampler continues from the counter; continuation passes parity | §4.5, §4.2.1 |
| 4 | ingress node lost | kill | its sequences cancelled ring-wide; KV and history freed in ≤ 3 s; other sequences unaffected | §9.2 |
| 5 ● | coordinator lost, hosting no stage | kill | zero failed sequences | SC-27 |
| 6 | coordinator lost together with a stage | kill | election, then repair; sequences resume | §9.1 |
| 7 ● | runner crash | SIGKILL the runner | restart in place and replay; daemon unaffected | §8.2 |
| 8 | sticky GPU error | provider fault injection | `Device` class; re-probe; restart or exclude the device | §8.2 |
| 9 ● | link flap | block an edge for 1.5 s; pull a TB5 cable on the lab rig (**Maintainer action**, ADR-0021) | in-process reconnect or demotion; no replay; pause ≤ 2 s | RFC-0031 §7.4 |
| 10 | link degradation | +50 ms latency on one hop | `Drift`; deferred re-plan; no error | §8.1 |
| 11 | network split 2 \| 2 under load | partition | both sides serve what they can; heal converges; orphans adopted or drained | RFC-0032 §7.5 |
| 12 ● | laptop sleep with and without a notice | OS sleep; clock jump injection | immediate recovery with notice; no false failures from the sleeper on wake | RFC-0030 §5.8 |
| 13 | NVMe throttled 10× | I/O engine delay | `Drift`, no failure | §8.1 |
| 14 | memory pressure | OS pressure signal | ladder runs; history spills, then drops; no OOM kill | §4.1.2 |
| 15 | two non-adjacent stages lost at once | kill 2 nodes | each replays from its own predecessor | §4.2 |
| 16 | two adjacent stages lost at once | kill 2 nodes | cascade source moves upstream; resumes or ends retryably if no capacity | §4.2 |
| 17 | fast-island member lost | kill | island stage repaired and replayed | RFC-0032 §5.6 |
| 18 | replay source lost mid-replay | kill during replay | handshake re-runs with the next source | §4.2 |
| 19 | client link dropped, ingress alive | close the client's connection mid-stream; reconnect after 30 s, and in a second run after 90 s | within the 60 s grace the client re-attaches by `Last-Event-ID` and receives every token once; after it, the sequence is cancelled and its KV and history freed | §9.2 |
| 20 | cold repair from a second copy | kill a stage node whose shard exists nowhere else except the predicted absorber's second copy | the repair reads the bytes from the absorber's disk with no Hub fetch; resume time recorded against the no-copy run | §9.3 |

Each scenario asserts four things:
- emitted tokens are never duplicated or retracted (a transcript check at the client);
- every degradation flag appears and then clears;
- the recovery events appear in the response metadata;
- the event log shows `RecoveryStarted` and then `RecoveryCompleted`.

## Rollout / milestones

| Milestone | Lands |
|---|---|
| M2 | `RepairCore` as a sans-I/O protocol core in RFC-0060's simulator (Resolve strategy only); data-plane fast-path detection (`EdgeDown`); supervisor defaults (§8.2); transport recovery (RFC-0031); ingress-loss cleanup; coordinator-loss behaviour meeting SC-27. A lost stage ends its sequences with a retryable error, and the instance is re-planned automatically with `Resolve`, as the M2 baseline. Straggler drift reports |
| M3 | second copies of source chunks on the predicted absorber (§9.3); the ingress detach grace (§9.2); history store; recovery handshake (`HandshakeCore`); every repair strategy in `RepairCore`; replay for middle, head and tail stages; `Reconcile` and `ResumePoint` migration; KV migration for planned transitions; repair strategies with `Relink` and `ExtendShard` (RFC-0032); degraded modes and their exit; standby options; the chaos suite on the simulated fabric and on rigs. M3 exits on SC-28: "node kill mid-generation completes the request" (RFC-0001 §16) |
| M4 | chaos suite on every Tier 1 rig; multi-failure cascades hardened; history spill tuning; the dashboard's standby recommendation |

## Open questions

1. **Where a lost node's bytes come from.** *Answered (OQ-NET-13 closed against it).* An evictable second copy of each stage shard's source chunks on the predicted absorber, over the LAN at background priority, only on spare disk under `store.standby_max_bytes`, evicted first, never on battery-class nodes (§9.3). Pre-packing stays on non-ring nodes unless DIST-E14-T16 shows it pays on ring nodes.
2. **History in throughput mode.** At many concurrent sequences the footprint exceeds the default budget, and `auto` turns history off. Benchmark in M3 whether re-prefill on loss is acceptable for scenario S5, or whether the budget should scale.
3. **False recovery starts.** Recovery solves speculatively on `EdgeDown` and commits only on confirmation. Measure how often a flapping link causes a wasted solve, and whether that ever delays a real recovery.
4. **Coordinator co-located with a stage host.** *Answered (OQ-NET-25).* Within a capability class, "hosts no stage" ranks first, and only coordinator-eligible nodes are ranked; ADR-0009's tuple is revised (§9.1).
5. **Prefix-cache positions have no history.** Positions served by a prefix-cache hit were never sent across a boundary in this sequence, so recovery cascades towards the head for them. Should prefix-cache entries optionally carry boundary history (RFC-0022)? Benchmark agent workloads with long shared prefixes.
6. **Requirements placed on RFC-0012.** *Answered (OQ-NET-27).* The counter-based sampler is provided (ADR-0107), and grammar state replays on a replacement tail: ADR-0504's `ConstraintEngine` advances over the emitted tokens deterministically and needs no mask to do so, the compiled grammar reaches the new tail by hash in `SEQ_META`, and reasoning-budget counters derive from the transcript. Replaying 4K tokens costs about 0.2 s [estimate: at most ≈ 50 µs per token, a figure that includes the mask]. PLC-E9-T18 records it on RFC-0012's side (§4.5).

## Proposed amendments to RFC-0001

1. **§7.5, activation-history row.** Replace "the head keeps, per live sequence, the activations it sent across each stage boundary" with "each sending stage keeps, per live sequence, the activations it sent across its outbound boundary, in the wire dtype (ADR-0010 item 4); the head's own inputs are token IDs, which ingress holds". Correct the memory figure to the wire dtype: F32 at hidden 8192 is 32 KiB per token, 128 MiB at 4K tokens, not the F16 figure of 64 MiB.
2. **§7.5, unplanned stage loss paragraph.** Replace "the head's activation history is single-copy, so losing the head forces re-prefill … v1 accepts the head as a single point of failure for replay" with "losing the head forces a re-prefill of the head's layer range only, from the ingress transcript. v1's single points of failure are the ingress node, for its own sequences, and the coordinator, for placement changes during an election (RFC-0033 §2)."
3. **§17 question 2** is answered by amendments 1 and 2: no history replication to a standby is needed.
4. **`PlanDelta` relink and shard extension.** This RFC depends on RFC-0032's amendment 1 (`DeltaItem::Relink`, `DeltaItem::ExtendShard`, and `Ready → Ready` on both).
