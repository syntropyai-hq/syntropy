# ADR-0600: Deterministic multi-node simulation over sans-I/O protocol cores
- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the quality and release engineering area)
- Related: ADR-0009, ADR-0011, ADR-0012, ADR-0015, ADR-0019; RFC-0001 §2.3, §3, §7, §8; RFC-0030, RFC-0032, RFC-0033, RFC-0060 §2, §8

## Context
Syntropy's distributed behaviour is where the research found the worst failures in comparable systems:
- coordinator elections cascading from one flapping connection;
- in-flight requests failing on a coordinator change;
- instances deleted on node silence;
- rings that stall when a stage dies;
- split-brain coordinators and late-joiner catch-up bugs fixed one by one after release

(`multi-device-inference.md` §10.5, §13 items 1, 2, 16). Syntropy commits to more than those systems did:
- a coordinator change fails no request (SC-27);
- a lost stage resumes with no client-visible error (SC-28);
- membership converges within fixed heartbeat counts (SC-25);
- tokens are emitted exactly once across replay and migration (RFC-0001 §7.5).

These properties concern rare interleavings of timers, messages and failures. Real-network tests find such interleavings by luck and cannot replay them. The runtime also mixes two execution models: tokio tasks for control, network and API, and dedicated pinned threads with no async runtime for the hot path (ADR-0012). Any simulation approach must cover both.

## Options considered
1. **Real processes on real or emulated networks only** (network namespaces, netem, lab hardware). These test the real stacks. They are slow (real time), nondeterministic and costly per scenario, and a failing run cannot be replayed. They are necessary as a layer, but not sufficient.
2. **Whole-daemon simulation on a paused-time async runtime with a simulated network.** This runs the real tokio code. Determinism, however, depends on the runtime's scheduler internals and random `select!` branch order, and on there being no hidden wall-clock read anywhere in the dependency tree. The runner's hot threads are not async code at all, so they would still need separate treatment. Replay is fragile across dependency upgrades.
3. **Model-checking abstract protocol specifications** (TLA+-style). Strong for the election and log protocols. But it checks a model, not the code, and drifts from the implementation unless someone maintains both.
4. **Chosen: a single-threaded discrete-event simulator on virtual time that drives the real decision-making code.** Control-plane protocol logic is written as sans-I/O state machines. Runner dispatchers are pollable, and devices, drives, links and the store are faked behind their existing traits. A seed reproduces a run exactly.

## Decision
1. **Sans-I/O cores.** Membership, coordinator election and lease, event-log replication and catch-up, the plan-transition orchestrator, the supervisor's restart policy and the recovery decisions are each implemented as a `ProtocolCore`. A core is a deterministic state machine that takes `(now, input)` and emits outputs. It has no sockets, tasks, sleeps or clock reads. Production wraps each core in a thin tokio shell. The simulator's event queue is the other shell (RFC-0060 §2.2; RFC-0001 amendment A1).
2. **Injected time and entropy.** `Clock` and `Entropy` traits in `syntropy-core`. Decision-making code reads time and randomness only through them. Hot-thread telemetry timing is exempt because it never feeds a decision (RFC-0060 §2.1).
3. **Pollable runners.** `Dispatcher::poll_once` and `ComputePool::run_inline` make the hot path drivable from one thread. Production threads loop over the same calls (RFC-0001 amendment A3).
4. **Fakes behind existing traits:**
   - `FakeKernelProvider`, with a `Numeric` mode running deterministic CPU kernels, and a `Digest` mode that tracks only byte counts and digests of graph-IR values. Digest mode is decomposition-independent and its routing, tokens and acceptance are scripted, so full-size models fit in a simulation;
   - `FakeIo`, with an NVMe thermal and queue-depth model;
   - `SimTransport` over a fabric with α, β, jitter, tail-mixture, loss, reorder and network-split models;
   - `FakeProbe`, `FakeStore` and `FakeClient`.
5. **Event ordering.** Events are ordered by virtual time, then by a tie-break key drawn from the seeded RNG. Different seeds explore different interleavings of simultaneous events, and one seed always replays one interleaving.
6. **Oracles after every event:**
   - safety checkers: single coordinator per term, log prefix stability, epoch monotonicity, per-node capacity, single ownership, plane separation, exactly-once tokens, no approximate flags in exact runs;
   - liveness checkers for SC-25, SC-27 and SC-28;
   - in `Numeric` mode, a bit-identical token-stream oracle against a single-device run (TP/EP shapes excepted, since they reorder reductions).
7. **Shrinking and permanence.** Failing seeds are shrunk by delta debugging over faults, workload and duration. The minimal cases are committed to `testdata/sim-regressions/` and replayed on every PR forever.
8. **Determinism self-check.** Nightly, seeds run twice in separate processes, and the trace hashes must match.
9. **Real stacks stay covered.** The emulated fabric (netns + netem) and lab chaos run the same scenario files against real processes (RFC-0060 §8.9, §9).

## Consequences
- **Positive.**
  - Distributed bugs reproduce from a seed on a laptop, with the exact event index of the violation.
  - Thousands of failure interleavings run per night, in a 1-hour nightly slot on R-LABCPU, with seed counts derived from the measured cost per seed and capped at 4,096 per scenario (RFC-0060 "Performance model and budgets"; revised 2026-10-19). PR runs use a standard 4-vCPU hosted runner, capped at 64 seeds.
  - `Digest` mode tests placement, admission, fetch and recovery at real model scale (hundreds of GB across many nodes) with no real memory.
  - Sans-I/O cores are also the natural fuzzing target for protocol inputs (RFC-0060 §10).
- **Negative.**
  - The control-plane RFCs (RFC-0030, RFC-0032, RFC-0033), and the agents implementing them, must write their logic in sans-I/O style from M2, and keep their shells thin.
  - Code in the shells and the real network stacks is not covered by L6. It depends on L7 and L8.
  - The simulator's fidelity to real timing must itself be validated against lab runs (RFC-0060 "Testing and validation", item 4).
  - Determinism rules (no `RandomState` maps in simulated crates, no direct clock reads) need lint enforcement and add review friction.

## What would make us revisit
- The M2 fidelity check shows simulated step times, hop distributions or recovery timelines differing from the lab by more than 20% at p50, and recalibration from lab profiles does not close the gap.
- A class of production bugs repeatedly escapes the simulator because it lives in the shells. Then add option 2 for shell-level tests, keeping option 4 for the cores.
- Writing a protocol core sans-I/O proves to cost more than 1.5× an equivalent async implementation in agent runs and review iterations for two consecutive cores, as recorded by the implementing agents.
