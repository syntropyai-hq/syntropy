# ADR-0015: Process model: one binary, a node daemon, one runner process per model instance
- Status: Accepted
- Date: 2026-10-19
- Deciders: maintainers (drafted by the architecture area)
- Related: ADR-0001, ADR-0002, ADR-0009, ADR-0011, ADR-0012, ADR-0016, ADR-0020; RFC-0012, RFC-0030, RFC-0033, RFC-0034, RFC-0052, RFC-0053, RFC-0054

## Context
A node does two different jobs.
- **Long-lived control.** Cluster membership, possibly the coordinator role (ADR-0009), the API, discovery and the model store.
- **Fault-prone compute.** GPU contexts, pinned arenas, kernels from several providers, and the data plane.

The compute half fails in ways the control half must survive:
- **Driver faults are fatal to a process.** CUDA's sticky errors leave the context unusable until the process exits. Device-lost and hang reports on multi-GPU AMD and Intel setups, and garbage output on an Apple GPU, are among the main complaint themes for one surveyed engine (`plan/research/tiered-memory-inference.md` §18).
- **Failures cascade if one process holds both halves.** In one surveyed runtime, a node that was not the coordinator died and rejoined. That set off a broken pipe, a re-election and the teardown of the instance's runners (`multi-device-inference.md` §10.5). If the coordinator's process also holds a GPU context, any GPU fault becomes a coordinator failover and puts every in-flight request in the cluster at risk. ADR-0009 forbids that.
- **The token path cannot cross a process boundary.** A surveyed desktop runner wraps a third-party server as a subprocess. A tiering engine cannot work that way, because it must control buffers and streams inside the forward pass (`backends-and-portability.md` §2.2).

So everything a decode step touches lives in one process, and only token output and supervision cross to the daemon.

The research recommends one runner per model instance under a supervisor, with stall timeouts, crash isolation and a lifecycle state machine driven by cluster state. A split should become routable only when every stage is ready (`multi-device-inference.md` §15.2 item 4, §15.11 item 2).

Packaging constraints (`backends-and-portability.md` §5.4):
- macOS requires every nested binary to be signed and notarised.
- iOS apps cannot spawn processes or load downloaded code.

## Options considered
1. **One process for everything.** Zero IPC and the simplest deployment. A sticky GPU error or a kernel bug takes down the API, membership, the coordinator role and every other loaded model. Rejected, except for the embedded mode below.
2. **Daemon plus one runner process per loaded model instance.** Chosen.
3. **One process per device.** Finer isolation, but execute-in-place splits a single MoE layer between GPU and CPU on every token (ADR-0006), which would put IPC inside the layer. Rejected.
4. **Separate daemon and runner binaries.** Two artefacts to sign and ship, possible version skew, and PATH problems on locked-down machines. Rejected in favour of re-executing one binary.
5. **A container per runner.** GPU passthrough differs per OS, it adds heavy dependencies, and it gives no gain over an OS process. Rejected.

## Decision
1. **One binary, `syntropy`.**
   - `syntropy serve` runs the node daemon: control plane, cluster membership, discovery, API, model store and supervisor.
   - The daemon starts each runner by re-executing its own executable path with a hidden subcommand, `syntropy __runner --ipc <handle>`. The updater runs a staged build's self-test the same way, `syntropy __selftest` (RFC-0052 §7.3).
   - At the IPC handshake of a freshly spawned runner, the runner and daemon compare build IDs and refuse to pair on a mismatch. A daemon re-adopting a running runner after an update needs an equal IPC protocol version instead (decision 6, revised).
   - *Windows exception (revised 2026-10-19).* The Windows release also ships `syntropyw.exe`, a small GUI-subsystem launcher that starts the daemon without a console window, handles end-of-session notices and guards in-place updates. A console-subsystem binary cannot run as a windowless per-user service, and a GUI-subsystem binary cannot serve as a CLI. The launcher contains no inference or control-plane code, so everything that serves is still the one `syntropy` binary (RFC-0052 §6.3).
2. **A runner per loaded model instance on that node.**
   - The runner owns its stages' devices, tiers (arenas, expert caches and KV, under the budgets the daemon grants it) and data-plane endpoints.
   - It holds no cluster state beyond its own assignment and plan epoch.
   - All hot threads (ADR-0012) live in the runner.
   - The daemon is the per-node budget authority. When several runners share a device, it hands each one a lease (bytes of VRAM, bytes of pinned RAM, core set), and the runner refuses allocations beyond its lease.
3. **Lifecycle per runner instance:** `Fetching → Packing → Loading → Warming → Ready → Draining`, plus terminal `Stopped` and `Failed{cause}`.

| State | Where it runs | Exit condition |
|---|---|---|
| `Fetching` | daemon's store (ADR-0016) | every shard chunk present and BLAKE3-verified |
| `Packing` | runner, via `store::pack` and `KernelProvider::repack` (RFC-0001 §3.3), off the token path | per-device pack present for the `PackKey` (model hash, device class, shard digest, layout version) |
| `Loading` | runner process, started here | weights resident per the placement plan; arenas pinned progressively |
| `Warming` | runner | graphs captured per verify-window size, expert cache filled from the learned profile, a parity smoke test against stored reference logits passed (ADR-0019) |
| `Ready` | runner | accepts sequences |
| `Draining` | runner | in-flight sequences finished or handed to RFC-0033 migration; no new admissions |

   Every transition is a cluster event (ADR-0009). A pipeline is routable only when every one of its stages is `Ready`.
4. **IPC.**
   - Control messages are typed, length-prefixed and versioned. They travel over a Unix-domain socket on Linux, macOS and Android, or a named pipe on Windows. The endpoint lives in a per-user directory with mode 0700, or behind a same-SID ACL on Windows, and the peer is checked with the OS peer-credential call.
   - A shared-memory segment (memfd, `shm_open` or a file mapping) carries:
     - SPSC rings of sampled tokens from the tail stage to the API
     - per-thread trace and metric buffers (RFC-0053)
     - a per-runner progress counter
   - Activations between runners on the same host use the `Shm` transport (ADR-0011), not the daemon.
5. **Supervision.**
   - The daemon reads each runner's progress counter once a second.
   - While a runner has sequences in flight, no progress for `max(10 s, 20 × measured p99 step time)` is a stall. Stalls during loading are judged against `3 × expected load time` (bytes ÷ measured NVMe rate) plus 60 s.
   - A stalled or faulted runner is killed. It is marked `Failed{Stalled | DeviceLost | Crashed | OutOfMemory}` and reported to the coordinator, which triggers RFC-0033 replay or migration.
   - On `DeviceLost`, the device is re-probed before a restart. If it is gone, the resource graph changes and the solver re-plans.
   - Restarts back off exponentially from 1 s to 60 s. After 5 failures in 10 minutes the instance stays `Failed`, and the CLI and dashboard show the cause.
6. **Runners outlive the daemon only within a bound** (revised 2026-10-19; the original text, "runners never outlive the daemon", enforced with `PR_SET_PDEATHSIG`, a `NOTE_EXIT` watch and a kill-on-close job, is superseded, see Revisions). The bounded orphan rule of RFC-0052 §6.5:
   - Runners use no parent-death signal, exit watch or kill-on-close job.
   - A runner detects daemon loss by end-of-file on its IPC socket, or by the daemon's shared-memory heartbeat stalling for `runner.daemon_silence` (suspend-aware).
   - It then serves only sequences whose ingress is another node, admits nothing new, and waits on its adoption socket. A restarted daemon re-adopts it within `runner.orphan_grace` when the IPC protocol versions match and the observability tables match or translate. Otherwise it drains for at most `recovery.drain_deadline` and exits. A runner therefore outlives its daemon by at most `runner.orphan_grace + recovery.drain_deadline`.
   - Each OS's service manager is configured so that stopping the daemon does not kill its runners: `KillMode=process` with a preserved runtime directory under systemd, `AbandonProcessGroup` plus `setsid()` in runners under launchd, and a daemon-owned job without kill-on-close on Windows, with breakaway for the logon task (RFC-0052 §6.1–§6.3).
   - Where re-adoption is unavailable (containers, the embedded mobile runner, a changed IPC protocol, or a logon task whose job forbids breakaway), updates drain and restart the runners instead.
   - A normal stop (`syntropy stop`, a service stop, OS shutdown) still drains runners before the daemon exits.
7. **Embedded mode.** On iOS, and on Android when it is packaged as an app, the runner runs in-process on its own threads behind the same IPC trait, implemented in memory. Crash isolation is lost, and the node reports `isolation: none`.

## Consequences
- **Positive.**
  - A GPU fault costs one model instance on one node, never the coordinator or the API.
  - The lifecycle states give the dashboard and the router a single source of truth.
  - One signed artefact per platform.
- **Negative.**
  - Token output crosses one shared-memory ring between runner and daemon. That is a bounded cost, but it must be measured.
  - Each runner process holds its own driver context, which costs VRAM on hosts serving several models.
  - The daemon must arbitrate device budgets between runners.
- **Work created.**
  - RFC-0001: the lifecycle enum and IPC message types.
  - RFC-0012: runner internals.
  - RFC-0033: supervision, restart and replay.
  - RFC-0034: IPC authentication.
  - RFC-0052: service units, re-exec on each OS, the iOS and Android embedded build.
  - RFC-0053: the shared-memory metrics.
  - RFC-0054: timeout and back-off configuration.

## What would make us revisit
- Measured runner→API token latency through the shared-memory ring exceeds 50 µs at p99 on Tier 1 hardware.
- Per-process driver context overhead exceeds 5% of VRAM on 8 GB cards when two models are loaded. Consider one runner hosting several models on the same devices.
- GPU drivers gain fault isolation per context that survives sticky errors, which would make the process boundary unnecessary.
- Mobile embedded mode turns out to be the common deployment, which would justify a different isolation design there.

## Revisions

- **2026-10-19, RFC-0001 amendment pass.**
  - Decision 6 is superseded by RFC-0052 §6.5's bounded orphan rule. The original decision killed runners whenever the daemon exited, which made runner re-adoption across a daemon update, and so in-place upgrades that keep models loaded, impossible on every Tier 1 OS. The rule keeps the decision's purpose: no runner survives indefinitely without a supervisor.
  - Decision 1 gains the re-adoption handshake rule (same build to spawn, same IPC protocol to adopt; RFC-0062 §10.1), the hidden `__selftest` mode, and the Windows launcher exception.
  - The lifecycle table's `Packing` row already names the runner, which builds packs through `store::pack` and `KernelProvider::repack` (RFC-0041 amendment 5); no change was needed.
