# RFC-0051: CLI and dashboard
- Status: Draft
- Area: Product surface and operations
- Depends on: RFC-0001 (§3, §4.1, §9, §10), RFC-0050, RFC-0052, RFC-0053, RFC-0054, RFC-0010 (profiles), RFC-0011 (previews, explanations), RFC-0023 (hit-rate telemetry), RFC-0030 and RFC-0034 (invites, admission), RFC-0061 (bench), ADR-0015, ADR-0016, ADR-0018, ADR-0019, ADR-0503, ADR-0506
- Date: 2026-10-19

## Summary

The `syntropy` binary is also the CLI, and every daemon serves a web dashboard. Both are clients of the native API (RFC-0050 §13, ADR-0503) and nothing else, so anything they show or do can be scripted.

- **CLI.** A command tree around a few verbs people already expect (`serve`, `run`, `pull`, `ls`, `ps`) plus Syntropy's own:
  - `plan`: preview a placement before loading;
  - `cluster` and `invite`;
  - `bench`, `probe`, `doctor`, `config`, `service`, `update` and `uninstall`.
  Output is human tables on a terminal and stable JSON with `--json`. Exit codes map to error classes. Fidelity and estimates are always labelled.
- **First run.** `syntropy run` with no argument asks one question, what the machine is for (chat, or agents and coding), and computes a recommended model from this machine's (or this cluster's) measured resource graph. The default is a quick start whose download finishes within a bounded time, at 4 bits per weight or more, under a permissive licence. It shows the predicted speed as an estimate, the context the plan will hold, the download size and time, and the licence, and starts after one confirmation (SC-24: one question and one model choice).
- **Dashboard.** A small single-page app embedded in the binary and served at `/` by every daemon (ADR-0506). Its views:
  - cluster topology with measured links;
  - per-node tier usage, with the cost of yielding memory;
  - placement previews with per-node memory deltas and a fit verdict before anything loads;
  - plan explanations;
  - live tokens/s, TTFT, TPOT and expert-cache hit rates, against the cost model's prediction;
  - downloads with their sources;
  - invites and admissions;
  - settings with the origin of every value;
  - from M2, one minimal chat view (§5.12).
  It makes no external requests and loads no fonts or scripts from elsewhere.

## Motivation and goals / non-goals

The people Syntropy is for own hardware they use for other things, and they "don't want to become cluster operators" (vision §4). The research found the opposite in surveyed systems:
- source builds with many prerequisites;
- per-device command lines with hand-typed IP addresses and device indices;
- placement that only a dashboard could trigger;
- model pickers that guessed fit from free RAM (`multi-device-inference.md` §12.1, §12.3; vision §1.4).

The one surveyed engine that spread widely asked three questions with recommended defaults and nothing else (`tiered-memory-inference.md` §19).

Syntropy also has things to show that no surveyed system could:
- a prediction for every term of token time;
- the bytes each node will hold, per tier, before loading;
- why each tensor sits where it sits;
- what giving RAM back to the user will cost.

Showing them is how users learn to trust automatic placement instead of fighting it.

### Goals

- G1. From install to first token: one command, one question and one model choice with a computed default, no file, no typed address (SC-24).
- G2. Every number shown is labelled measured, estimate or target, the way the plan documents label them. Predictions say "estimate".
- G3. Fidelity is visible wherever output is (D3, ADR-0019 decision 7): the CLI banner, every dashboard model and request badge.
- G4. Users can see before loading whether a model fits, how fast it will be, what each node will hold and download, and what will be evicted.
- G5. Scriptable: `--json` everywhere, stable exit codes, no interactive prompt when stdin is not a terminal.
- G6. The dashboard works offline, on a phone-width screen, and costs the daemon almost nothing when nobody has it open.

### Non-goals

- A full chat client (vision §3.3). `syntropy run` is a terminal REPL for trying a model. From M2 the dashboard has one minimal chat view (§5.12): one conversation at a time over native sessions, with no history sync, attachments or plugins.
- Languages other than English in v1. Every human-facing CLI and dashboard string is externalised from M1 (§4 rule 8), so a translation needs no code change later.
- A desktop GUI application. The macOS menu-bar and Windows tray helpers (RFC-0052 §5.2) only show status and open the dashboard.
- Remote management of clusters the user is not a member of.
- Model hub browsing beyond the curated catalogue and Hub references (vision §3.3).

## Design

### 1. Shape of the client side

```text
syntropy-cli (crate `cli`, RFC-0001 §4.1 layer 6)
├── argument parsing (clap, derive), mode dispatch: serve | __runner | client commands
├── client: typed native-API client over the local socket, falling back to loopback HTTP
├── render: tables, progress, colours, JSON output, error rendering
├── repl: `syntropy run` interactive loop (reedline)
└── assets: the dashboard bundle (embedded, brotli and gzip precompressed)
```

The client library is the same `syntropy-api` types the server uses, so CLI and server cannot disagree about a field. The CLI never reads cluster state, the store or the state directory directly. It asks the daemon. The exceptions are commands that must work without a daemon:
- `serve`, `config check`, `config path`, `version`, `completions`, `doctor --offline`;
- the lifecycle commands of RFC-0052: `setup`, `service`, `update`, `uninstall`.

*Note, 2026-10-31 (COMP-E7-T01):* `syntropy --version`, the binary's own report, adds a line `ggml <commit>`, the commit of the static ggml copy it carries (RFC-0040 §8.1). Its `--json` form, the native API's `Build`, is unchanged.

**Reaching a daemon.** A client command tries, in order:
1. the local socket (`api.local_socket`);
2. `http://127.0.0.1:7433` (or `SYNTROPY_HOST`, for a remote daemon with `SYNTROPY_API_KEY`);
3. if neither answers and the command needs a daemon (`run`, `chat`, `pull`), starting one.

Starting a daemon follows these rules:
- **If a service is installed** (RFC-0052 §6), the CLI starts the service.
- **Otherwise** it spawns `syntropy serve` detached, as the current user, with its log in the state directory. It then prints:

```text
started a Syntropy daemon (pid 48122); it keeps running so loaded models stay loaded.
stop it with `syntropy stop`, or run it as a service with `syntropy service install`.
```

An automatically started daemon is not tied to the CLI process. Ending a `run` session must not unload a model that took minutes to load.

### 2. The CLI

#### 2.1 Command tree

| Command | What it does | API it uses | Milestone |
|---|---|---|---|
| `syntropy serve` | run the node daemon in the foreground (§2.3) | — | M0 |
| `syntropy run [MODEL] [PROMPT]` | load if needed, then chat interactively, or answer one prompt (§2.4) | generate, models, instances | M0 |
| `syntropy chat [MODEL]` | alias of `run` with no prompt (RFC-0001 §3.1 names it) | as `run` | M0 |
| `syntropy pull MODEL [--variant V] [--nodes planned\|all\|SEL] [--accept-licence]` | register and fetch; shows per-node progress and sources. A custom-licence model shows its summary and link and needs `--accept-licence` or an "I accept" on a terminal, recorded on this node; a gated model prints the Hub acceptance step (§3.2) | models/pull, models/{ref}/licence/accept | M1 |
| `syntropy ls` (`models`, `list`) `[--catalogue] [--nodes]` | models in the cluster's stores; `--catalogue` adds recommendations with estimates (§3) | models | M0 |
| `syntropy ps [--sequences]` | instances: model, fidelity, plan epoch, ring, state, tok/s, live sequences | instances, metrics/live | M1 |
| `syntropy show MODEL [--tensors] [--licence]` | manifest summary, census by class, KV bytes per token, siblings; `--tensors` lists tensor names with group and class (the names regex overrides match, RFC-0054 §6.3) | models/{ref}, tensors | M1 |
| `syntropy plan MODEL [policy flags] [--candidates N] [--explain]` | placement preview without loading (§2.5); `plan --instance ID --explain` explains a loaded plan | placement/preview, plan/explain | M1 |
| `syntropy unload MODEL\|INSTANCE` | drain and unload | instances | M1 |
| `syntropy rm MODEL [--nodes SEL]` | unload, then delete chunks and packs | models | M1 |
| `syntropy stop` | drain every runner with a deadline, then stop the local daemon (RFC-0001 §3.4) | — (signal over the local socket) | M0 |
| `syntropy cluster status\|nodes\|graph\|events\|approve ID [--code N]\|deny ID\|open [--for 10m] [--joins 4]\|revoke NODE\|remove NODE\|leave\|join TOKEN` | membership and admission (§2.6); `approve` shows the role and grant choices, and `--code` answers number matching non-interactively; `leave` and `remove` carry the last-keyholder guard | cluster, nodes | M2 (join by invite M3) |
| `syntropy cluster recover\|refound` | restore keyholder status from the offline recovery code, which always rotates the cluster key; or found a new cluster from a surviving member when the last keyholder is lost (§2.6) | cluster/recover, cluster/refound | M2 |
| `syntropy cluster keyholder promote\|demote NODE` | keyholder changes; promotion needs an existing keyholder's signature and the node's acceptance (RFC-0034 §8.3). The offer at pairing approval (M2) and the later offer (M3) use the same path | cluster | M2 (offer at pairing); M3 |
| `syntropy invite create [--own-device\|--ingress-only] [--ttl 24h] [--uses 1] [--scope …]\|list\|revoke ID` | invite tokens; asks "your own device, or someone else's?" on a terminal unless a flag answers it; without an answer the invite is ingress-only; prints the token text, the invite URL, a join command and a terminal QR code | cluster/invites | M3 |
| `syntropy bench [MODEL] [--suite quick\|standard] [--report FILE] …` | runs RFC-0061's scenarios and writes a commit-tied report; `--report` writes the redacted file alpha testers send by hand (RFC-0053 §8, from M1) | generate, metrics | M0; `--report` M1 |
| `syntropy probe [--rerun] [--json]` | the node's capability profile and feature bits (ADR-0014 decision 1) | nodes/{self} | M0 |
| `syntropy doctor [--cluster] [--offline] [--json] [--fix]` | diagnostics (RFC-0053 §5) | doctor | M1 |
| `syntropy logs [-f] [--runner INSTANCE] [--level L] [--since 10m]` | daemon and runner logs | logs stream | M1 |
| `syntropy doctor --bundle [--cluster] [--out FILE]` (alias `syntropy debug dump`) | diagnostic bundle (RFC-0053 §6) | debug/dump | M1 |
| `syntropy debug state [--instance ID]` | prints `/debug/state` (RFC-0050 §13.6) | debug/state | M0 |
| `syntropy debug parity --model M [--family F] [--mode …]` | runs RFC-0060's parity driver against this node; needs `debug.parity_api`. With `--family`, runs it for an uncertified family on one device and prints the report beside the "experimental" label (RFC-0060 §7.4; `--family` from M3) | parity | M0 |
| `syntropy debug telemetry --instance ID [--since …] [--follow]` | streams per-step records; needs `debug.telemetry_api` | debug/telemetry | M0 |
| `syntropy config show\|get\|set\|unset\|edit\|check\|reload\|path\|schema\|export` | configuration (RFC-0054 §10.2); `export --cluster` writes cluster-scope values to a file (ADR-0501) | config | M0–M2 |
| `syntropy api-key create\|list\|revoke\|rotate` | API keys (RFC-0050 §11.3); `--model` and `--instance` narrow a key; `--cluster` creates a cluster-wide key (RFC-0034 §8.4) | keys | M1; `--cluster` M2 |
| `syntropy identity show` | this node's ID, the cluster ID and the API certificate, each as a six-word fingerprint (RFC-0034 §6.5) | version, cluster | M2 |
| `syntropy trust <node> [--cluster] [--system] [--remove]` | on a client machine: fetch a node's local CA, confirm its six-word fingerprint, install it in the user's trust store (the system store only with `--system`), print the environment variables for clients with their own CA bundles (RFC-0050 §11.1.1); ledgered. `--cluster` installs every member's CA in one step after one cluster fingerprint check | tls/ca, tls/cluster-cas | M1; `--cluster` M2 |
| `syntropy dashboard [--print-url]` | open the dashboard in the default browser, signed in through a one-time login URL obtained over the local socket (RFC-0050 §11.4) | auth/login-token | M1 |
| `syntropy setup` | first-run setup run by installers: module resolution, service install, probe (RFC-0052 §3) | — | M1 |
| `syntropy service install\|uninstall\|start\|stop\|restart\|status [--user\|--system\|--daemon]` | service manager integration (RFC-0052 §6); `--daemon` installs the macOS non-root LaunchDaemon for headless Macs (RFC-0052 §6.2) | — | M1 |
| `syntropy update check\|apply\|rollback\|status\|channel NAME` | updates (RFC-0052 §7) | — | M1 |
| `syntropy tune [--persist] [--status] [--revert]` | macOS: show the GPU working-set limit, the plan that fits now and after tuning, and the proposed limit; apply it after the system's administrator prompt; `--persist` adds a ledgered boot job that re-applies it (RFC-0020 §9.1, RFC-0052 §11). Linux (M3): the same flow for the TTM `pages_limit` that caps a unified-memory iGPU's share of RAM, opt-in, written to a ledgered `modprobe.d` file and applied at the next reboot. `tune` never changes `TdrDelay` or hardware-accelerated GPU scheduling, which `doctor` only reports (RFC-0053 §5.2) | instances, placement previews | M1 (macOS); M3 (Linux TTM) |
| `syntropy uninstall [--keep-models] [--keep-config] [--keep-identity] [--force] [--yes]` | revert every host change (RFC-0052 §11); refuses to remove the last keyholder of a cluster with other members unless another is promoted first or `--force` follows the warning (§2.6) | — | M1; keyholder guard M2 |
| `syntropy provenance show\|prune [--older-than D]` | the opt-in local provenance records (RFC-0050 §10) | — | M4 |
| `syntropy tray` | menu-bar or tray helper (RFC-0052 §5.2) | events | M4 |
| `syntropy completions SHELL` | shell completions for bash, zsh, fish, PowerShell, elvish | — | M0 |
| `syntropy version [--json]` | build ID, protocol versions, support tier, backends present | version | M0 |

Hidden: `__runner` (RFC-0001 §3.1) and `__selftest` (RFC-0052 §7.3).

#### 2.2 Output conventions

- **Streams.** Results go to stdout; progress, warnings and errors go to stderr. A pipeline `syntropy run m "prompt" | jq` gets only the answer.
- **Terminal detection.** With stdout on a terminal: aligned tables, colour (off with `NO_COLOR`, `--color never`, or `TERM=dumb`), live progress bars. Otherwise: plain text, no progress bars, no prompts. A command that would ask a question fails with exit code 2 and says which flag answers it (`--yes`, `--variant`).
- **`--json`.** Every read command accepts `--json` and prints exactly one JSON document: the native API's response type, unchanged, so CLI scripts and API scripts share one schema. Streaming commands (`run`, `logs -f`, `cluster events`, `pull`) print JSON Lines with `--json`.
- **Units.** Bytes in IEC units with three significant figures (`41.0 GiB`); `--bytes` prints exact integers. Rates as `tok/s` and `GB/s` (SI, matching the research's bandwidth figures). Durations human-readable (`1.2 s`, `412 ms`). Times local, ISO 8601 in JSON.
- **Labels.** A predicted figure is printed with the word "estimate" or `≈`; a measured one without. In JSON every figure carries `"kind": "measured" | "estimate" | "target"` where the API provides it (§4).
- **Errors.** `error[<domain>/<code>]: <message>`, then `hint:` lines with the exact command that fixes the problem when one exists. For example, with illustrative figures from the vision's S1 scenario:

```text
error[api/0011]: deepseek-v3:Q4_K_M does not fit this cluster
  needs 404 GiB of RAM or NVMe on stage nodes; usable: pc 115 GiB RAM + 310 GiB NVMe
  hint: syntropy plan deepseek-v3:Q2_K   (244 GiB, ≈4.4–9.3 tok/s estimate)
```

- **Exit codes.**

| Code | Meaning | From `ErrorClass` |
|---|---|---|
| 0 | success | — |
| 1 | other failure | `Internal`, `Device`, `Link`, `Correctness` |
| 2 | usage error, or a question needed an answer and stdin is not a terminal | — |
| 3 | no daemon reachable and none could be started | — |
| 4 | not found (model, instance, session, node) | `InvalidRequest` with a not-found code |
| 5 | invalid request or configuration | `InvalidRequest`, `Unsupported` |
| 6 | does not fit, or no capacity | `Capacity` |
| 7 | authentication or permission | `Auth` |
| 8 | integrity failure | `Integrity` |
| 9 | timed out | `Timeout` |
| 10 | partial success (some nodes failed in a cluster-wide command) | — |
| 130 | interrupted (Ctrl+C) | `Cancelled` |

#### 2.3 `syntropy serve`

Foreground daemon, used by service units and containers. Dedicated flags for the keys people set on the command line (RFC-0054 §2.4): `--bind ADDR` (repeatable; replaces `api.bind`), `--config PATH`, `--models-dir DIR` (prepends to `store.dirs`), `--offline` (RFC-0041 `offline`), `--log-level L` (`logging.level`), `--state-dir DIR`, plus `--set key=value` for anything else. On start it prints one block: build and support tier, backends found, listeners and whether each requires a key (RFC-0050 §11.1), the dashboard URL, cluster membership, and any `pending` config keys.

#### 2.4 `syntropy run` and `syntropy chat`

```text
run(model?, prompt?):
    if no model:
        if this cluster has exactly one routable or available model: use it
        else:
            if picker.use is unset: ask once "What is this machine for?  [1] chat  [2] agents and coding"
                                    (Enter takes 1); store the answer in picker.use (§3.3)
            show the picker (§3.3); on a terminal, Enter accepts the recommended model
    resolve model (RFC-0050 §4.1)
    if not in any store:
        show: estimate (tok/s, TTFT at the use's prompt length, labelled estimate), the context the plan
              will hold, download size per node and its ETA at the measured origin rate, disk needed
              (the true figure, §3.1), licence identifier and class; custom licence: summary, link and
              an "I accept" step recorded on this node; gated models: say what to accept and where
        confirm (TTY) or require --yes (non-TTY); --yes never accepts a custom licence: that needs --accept-licence
        pull with progress
    ensure instance (automatic placement); show per-node phases:
        pc     Fetching  12.3/41.0 GiB  from mac-studio (LAN, 2.3 GB/s)
        mac    Packing   88%
        strix  Warming   canary
    print banner, then the REPL or the single answer
```

**Banner.** One line before the first token, always:

```text
qwen3-235b-a22b:Q4_K_M · exact · 3 nodes (pc → strix → mac) · ≈15.8 tok/s estimate · context 32K
```

A model whose publisher ships it as multimodal adds "text only in Syntropy" (RFC-0050 §3.4).

When the model or request is approximate, the banner says so in words and names every knob, and a TTY shows it in the warning colour:

```text
qwen3-235b-a22b:Q4_K_M · APPROXIMATE: topk_reduce(6), expert_skip · output may differ from the reference beyond the exactness contract
```

**REPL.** Built on `reedline`: history in the state directory, multi-line input (Alt+Enter or a line holding `"""`), Ctrl+C cancels the current generation (the sequence is cancelled around the ring) and Ctrl+D exits. Commands:

| Command | Effect |
|---|---|
| `/set temperature 0.7`, `/set top_p …`, `/set system "…"` | sampling and system prompt for the following turns |
| `/think on\|off\|low\|medium\|high\|<budget>` | reasoning controls (RFC-0050 §8.3); reasoning text prints dimmed, above the answer |
| `/fidelity exact\|approximate [knob=…]` | per-request fidelity within the model policy; the banner reprints |
| `/session new\|save\|load ID\|show` | bind turns to a persisted session (RFC-0050 §9) |
| `/stats` | the last turn's TTFT, decode tok/s, tokens, cache hits, expert-cache hit rate, recovery events |
| `/retry`, `/clear`, `/help`, `/bye` | — |

`--verbose` prints the `/stats` line after every turn. A non-interactive `run MODEL "prompt"` (or a prompt on stdin) prints only the answer to stdout. `--json` prints JSON Lines of the native generation events.

#### 2.5 `syntropy plan`

`syntropy plan MODEL [--objective latency|throughput] [--context N] [--sequences N] [--fidelity …] [--include SEL] [--exclude SEL] [--pin …] [--candidates N]` calls the placement preview and prints, per candidate:
- the predicted decode tok/s and TTFT at 1K, 4K and 32K, each labelled estimate, with the dominant term ("62% of token time: expert reads from RAM on pc");
- a per-node table of tier bytes before → after, by purpose;
- download per node with sources and an ETA;
- the fit verdict and its binding constraint;
- the instances that would be evicted;
- the inputs that are priors rather than measurements.

`--explain` adds the per-group reasons (RFC-0011's explanation records). `syntropy plan --instance ID --explain` prints the same explanation for a committed plan, next to measured per-term times.

**One informational line about approximate options** (M3). There is no named approximate preset: approximate mode is per-model knobs only (RFC-0054 §4.2). `plan` prints one line below the exact candidates, and only when some knob has a published `KnobCost` report (ADR-0019 decision 6) and the preview predicts a gain of at least 1.5× [target] with it (RFC-0011's `approx_hint`):

```text
approximate options exist: ≈X tok/s with topk_reduce(6); measured cost KL Y, top-1 Z; see docs/approximate
```

The figures are the knob's report and the preview's estimate, labelled as such. The line never appears in `run`, the banner or a prompt, and nothing is enabled by it. The dashboard's preview view shows the same line under the same rule (§5.4).

#### 2.6 `syntropy cluster` and `syntropy invite`

- `cluster status` shows members with state, the coordinator and its lease term, the event-log index, protocol versions and pending admissions. `cluster nodes` lists hardware per node. `cluster graph` lists links with transport, α p50/p99 and β. `cluster events` follows the event stream.
- `cluster join TOKEN` prints the cluster name, its members, and the cluster-scope keys this node's file would propose (RFC-0054 §2.3), then asks for confirmation.
- `cluster approve ID` / `deny ID` act on pending LAN joins (`cluster.lan_admission = confirm`, ADR-0302). Each pending node shows its name, its hardware summary, its power class and the 6-digit short authentication string (SAS) that both devices display, so the user can check they are approving the machine in front of them, with number matching on the approving side. On a terminal, a new join request on this node prints a one-line prompt with the SAS. `cluster open` opens a pairing window for setting up several devices at once (ADR-0302 decision 2).
- **The joining side shows its SAS too** (OQ-PRD-19, answered). The joiner prints its SAS in `syntropy setup` for the duration of the first join, in `cluster status` and on its own dashboard. A headless node is set up over SSH or a console, so whoever runs `setup` sees it there.
- **The pairing approval** (M2) asks only what applies to the joining node, each with its default and one line on what it costs (DIST owns the rules, RFC-0030 §6 and RFC-0034 §5.2):

  ```text
  join "studio" (Mac Studio, M5 Ultra, mains, wired)?   SAS: pick the code studio shows  [ 482 913 ] [ 107 552 ] [ 913 044 ]
    role:  (x) runs model stages   ( ) client only: uses this cluster's models, holds no stage, needs no API key or CA
    [ ] may coordinate the cluster    without it, the cluster cannot re-plan or load models while every keyholder is away
    [ ] make it a keyholder           without it, nobody can admit or revoke devices while "pc" is off
  ```

  - The coordinator grant and the keyholder offer appear only for a wired, mains-powered desktop, and both start unticked. Neither is ever offered to a laptop or a battery-powered node.
  - "Client only" makes the node an ingress-only member, the documented path for a desktop that wants to use the cluster's models (RFC-0050 §11.1). It is the default when the approver marks the device as someone else's.
  - A battery-powered node is asked instead "Reach this cluster away from home?", defaulting to yes. Yes is recorded as `NodeInfo::off_lan_reach` and turns relays on for the cluster (`net.relay_mode`, RFC-0030 §6.4), so a laptop paired at home still connects when it leaves.
- **Keyholder status.** `cluster status` prints a permanent line while the cluster has one keyholder, for example `one keyholder: admissions and re-planning stop while pc is off`, and never interrupts. From M3, the cluster offers the role once more, in the dashboard and the next interactive CLI command, when it has gone more than 10 minutes [target] without its only keyholder while an eligible node is present. A decline is remembered.
- **Recovery code** (M2). When a cluster is founded, `setup` and the dashboard offer an optional offline recovery code, shown once, with a plain warning that whoever holds it can take over the cluster and that it is not stored anywhere by Syntropy. There is no cloud escrow. `cluster recover` on any member, with the code, restores keyholder status and always rotates the cluster key (RFC-0034 §5.5). `doctor` warns while the cluster has one keyholder and no recovery code.
- **Last-keyholder guard.** `cluster leave`, `cluster remove` and `uninstall` refuse to drop the last keyholder of a cluster that has other members. The refusal names what stops working (admissions, revocation, certificate renewal, re-planning under the keyholders-only coordinator default, RFC-0034 §5.2) and offers to promote another eligible member first; `--force` proceeds after the warning. `leave` works locally even when no coordinator is reachable.
- **Re-found.** When the last keyholder is lost and no recovery code exists, `cluster refound` on a surviving member creates a new cluster, and the other survivors join it by ordinary `confirm` pairing, one approval each, keeping their stores and packs (RFC-0034 §5.5). The command first says what starting over costs.
- `invite create` asks "your own device, or someone else's?" on a terminal unless `--own-device` or `--ingress-only` answers it; someone else's device, or an invite made with no answer, is ingress-only, and the admission notice shows the choice. It prints the token text, the dashboard's invite URL and a QR code rendered in the terminal, and states the expiry and remaining uses. Text and URL are always printed beside the code, so a code that fails to scan never blocks a join (OQ-PRD-20). Token format and security belong to RFC-0030 and RFC-0034.

### 3. Fit estimates and the model picker

#### 3.1 Fit estimates

A fit estimate answers "will this model run here, how fast, and what will it cost" before any weight byte moves (G4). Its producer is RFC-0011's pure `preview()`, run on the ingress node against the live resource graph, including current reservations. That function is requested of RFC-0011; this section fixes what it must return. Every view, the API's preview and estimate endpoints and automatic placement are projections of its output. None computes numbers of its own.

**Input the product side passes.**

```rust
/// Requested of RFC-0011. Pure, runs at ingress; MILP solves run in a probe-mode child.
pub struct PreviewInput<'a> {
    pub graph: &'a ResourceGraph,                   // live, with reservations
    pub census: CensusSource<'a>,                   // Full(&ModelManifest) | Summary(&CensusSummary)
    pub policy: &'a InstancePolicy,
    pub evictable: Vec<InstanceId>,                 // idle instances the caller would unload, LRU first
    pub node_overrides: Vec<(NodeId, ResourceOverlay)>, // what-if: e.g. the headroom slider (§5.3)
    pub ttft_prompts: SmallVec<[u32; 4]>,           // prompt lengths to predict TTFT at; default [1024, 4096, 32768]
    pub candidates: u8,                             // 1..=3
    pub effort: PreviewEffort,                      // DpOnly (catalogue ordering) | Full (DP warm start + MILP)
}
pub enum CensusSource<'a> { Full(&'a ModelManifest), Summary(&'a CensusSummary) }
/// Class totals per layer, bytes per expert per layer, top-k, lookup and head bytes, KV formula,
/// hyperparameters: enough for the DP, not for per-group assignment. Built by RFC-0041 from headers
/// or carried in the catalogue (§3.2).
pub struct CensusSummary { /* owned by RFC-0041 */ }
```

**Output each view needs, field by field.**

| Field | Unit | Label | Used by |
|---|---|---|---|
| `verdict`: `Fits` \| `Degraded { why }` \| `DoesNotFit(InfeasibilityReport)` | — | — | every view; automatic placement's 400 (RFC-0050 §4.3) |
| `InfeasibilityReport`: per node and tier, bytes needed against usable, the binding constraint in words, the smallest fitting sibling variant and its estimate, the extra memory that would make it fit | bytes | estimate | does-not-fit errors, `syntropy plan`, picker |
| per candidate `decode_tok_s` at batch 1 | tokens/s | estimate, with the `PredictedCost` per `CostTerm` (RFC-0053 §2.2) it came from | picker, `plan`, preview view |
| per candidate `ttft` at each `ttft_prompts` length | ns | estimate | `plan`, preview view, banner |
| per candidate `dominant_term`: the `CostTerm`, its node and its share of token time | fraction | estimate | one-phrase reasons in the picker and `plan` |
| per candidate `jitter`, set when the ring has a `WiFi` or `Overlay` edge: the predicted stall rate and the p99 token time (RFC-0011 §14) | stalls per minute; ns | estimate | the jitter badge beside `decode_tok_s` in the picker, `plan` and the preview view |
| per candidate, per node and tier: `before` and `after` bytes, each split by the RFC-0001 §9.1 and §9.3 purposes (context reserve, scratch, dense hot, KV, expert cache, RAM-homed weights, KV spill, I/O staging, network pools, headroom kept) | bytes | before = measured, after = estimate | preview view's memory bars (§5.4), RFC-0050 §13.3 `NodeDelta` |
| per candidate, per node: bytes to fetch, split by source (local store, named cluster node, origin) | bytes | estimate | preview view, `plan`, picker download size |
| per candidate `time_to_ready`, split into fetch (bytes ÷ measured source β), pack and load (SC-23's formula with the node's measured rates) and warm | ns | estimate, with each part shown | automatic placement's `load_wait` decision and `Retry-After`; preview view ETA |
| per candidate `context_length`: what the plan holds for `max_sequences` (RFC-0054 §4.5) | tokens | estimate | picker lines, `run`'s confirmation, the banner |
| per candidate `disk_needed` per node: the true footprint, source chunks plus every pack extent, less any source ranges a committed SOURCE-layout extent lets the store punch under `store.punch_covered_sources` (default `auto`, RFC-0041 §4.5, §9) | bytes | estimate | picker's free-space filter; every picker line and `run`'s confirmation |
| per candidate `evicts`: the prefix of `evictable` this plan needs unloaded | instance IDs | — | automatic placement; preview view's eviction list |
| per candidate `transition`: make-before-break or break-before-make | — | — | preview view |
| `assumptions`: inputs that are `MeasureSource::Prior` | — | "not yet measured" | every view |

```rust
/// The product-side projection, unchanged in meaning from RFC-0011's `Preview`.
pub struct FitEstimate {
    pub verdict: FitVerdict,
    pub decode_tok_s: Figure<f32>,
    pub ttft: Vec<(u32, Figure<Nanos>)>,
    pub dominant_term: CostTermShare,
    pub jitter: Option<JitterBadge>,         // wireless or overlay hop in the ring: stall rate and p99 token time
    pub per_node: Vec<NodeDelta>,            // RFC-0050 §13.3
    pub fetch: Vec<NodeFetch>,               // per node: bytes by source, ETA
    pub time_to_ready: Figure<Nanos>,
    pub context_length: Figure<u32>,         // the planned context
    pub disk_needed: Vec<(NodeId, Figure<Bytes>)>,   // the true footprint; keeps RFC-0011's estimate label
    pub evicts: Vec<InstanceId>,
    pub assumptions: Vec<Assumption>,
}
pub struct Figure<T> { pub value: T, pub kind: FigureKind /* Measured | Estimate | Target */, pub basis: Option<String> }
```

`Figure` is how every number crosses the API to a client, so the CLI and dashboard can label it without guessing (§4).

**Census sources** for `CensusSource`, best first:

| Source | When | Precision |
|---|---|---|
| registered manifest | the model was pulled or registered before | exact census |
| Hub headers | a Hub reference not yet registered: GGUF header and tensor table, or safetensors JSON headers, fetched by HTTP range reads (ADR-0016 decision 1) | exact census; costs a few range requests |
| catalogue census summary | browsing the catalogue (§3.2) | class totals per layer: exact for sizes, prior for expert popularity |

#### 3.2 The catalogue

A curated, signed list of models the project has checked against its families and conformance suite:
- **Delivery.** `catalogue.json` is built into each release, and newer copies are published as a TUF target that the update client fetches and verifies like any other (RFC-0052 §7.2). It works offline from the embedded copy.
- **Entry contents.**
  - Hub repository and pinned revision. A quantized variant from a third-party quantizer names that quantizer and pins its revision, and the entry carries a fallback source for when that revision disappears;
  - family, maximum context, capabilities (tools, reasoning, MTP heads, embeddings), and the source model's modalities, so a multimodal model is marked "text only in Syntropy";
  - licence identifier, licence class (`permissive` for Apache-2.0/MIT-class terms, `custom` otherwise), a one-line plain summary of a custom licence's conditions with a link to its text, the gated flag, and the date of the entry's last review;
  - per variant: quant tag, its bits-per-weight class (`Bits4` and above, or `Sub4`), files, total bytes, and a census summary (layers, hidden size, vocabulary, hot-set bytes per token, routed-pool bytes, bytes per expert per layer, top-k, lookup and head bytes, KV formula);
  - an editorial `order` with a link to its public rationale.
- **What `order` is.** The project's editorial ordering of general capability within the catalogue, set by the curation process below. It is shown as "catalogue order", never as a benchmark result or a quality score. No quality number is invented or displayed.
- **Curation** (open question 1, answered). The catalogue is curated through its repository's gates, like code:
  - Agents and users propose entries, and changes to `order`, as changes to the catalogue repository, each with a public rationale.
  - The review agent checks every proposal against the rationale rule and the entry checks below, and an entry merges through the same gates and review as code (ADR-0021).
  - An entry enters only after its family and variant pass conformance on at least one Tier 1 backend.
  - Freshness: a new release in an already-supported family gets a catalogue decision within 7 days [target].
  - A quarterly sweep, run as a recurring agent task, re-checks every entry's licence, gated flag, pinned revisions and review date, and proposes removals for entries whose sources have gone.
- **Licence policy for models downloaded on the user's behalf**. The user is the licensee and Syntropy is the download tool, so:
  - the zero-configuration default recommendation comes only from `permissive` entries;
  - a `custom`-licence entry shows its summary and link and needs an explicit "I accept", recorded locally per model and node (`pull --accept-licence`, the dashboard, or `POST …/licence/accept`), before any download;
  - a gated model needs the user's own acceptance on the Hub with their own token, which Syntropy never automates (ADR-0018 decision 7).
- **Hub references.** Any Hub reference still works. The catalogue only feeds recommendations.

#### 3.3 Recommendations

```text
recommend(graph, catalogue, mode, use):
    threshold = picker.min_tok_s[mode]          // interactive: 6 tok/s; background: 1 tok/s
    context   = picker.context[use]             // chat: 8192; agent: 32768
    rate      = origin rate measured by the range reads that fetch census headers (§3.1)
    for entry in catalogue:
        for variant in entry.variants (highest bits per weight first):
            est = solver::preview(PreviewInput { census: Summary(variant.census_summary), effort: DpOnly,
                                                 policy: latency with context, ttft_prompts: [context], … })
            eta = Σ fetch bytes ÷ rate
            if est.verdict != DoesNotFit and est.decode_tok_s ≥ threshold
               and (use != agent or est.ttft[context] ≤ picker.agent_ttft_ceiling)
               and est.disk_needed fits each node's free space above store.min_free_bytes:
                candidate(entry) = (variant, est, eta); break     // best precision that meets the bar
    default_ok(c) = c.variant.bits_class ≥ Bits4 and c.entry.licence_class = permissive
                    and c.eta ≤ picker.quick_start_eta
    sort candidates by entry.order, then by est.decode_tok_s
    refine the first 3 with the full MILP preview; re-order if the refinement drops one below threshold
    return { recommended: the first candidate with default_ok,
             others: the next 4,
             faster: the fastest passing,
             bigger: the earliest-ordered entry that fits but misses the threshold or default_ok,
                     shown with its ETA ("bigger: 142 GB, about 3 h") }
```

- **The use question**. The first run asks one thing, what the machine is for: chat, or agents and coding. The answer sets `picker.use` and with it the context the picker plans for, `picker.context` (8192 for chat, 32768 for agents), and, for agents, a TTFT ceiling at that prompt length, `picker.agent_ttft_ceiling`, whose default PROD-E6-T09's study fixes before M1 ships. It never asks again; `syntropy config set picker.use` changes it. SC-24 counts "one question and one model choice".
- **The quick-start default.** The default recommendation is the best candidate whose download finishes within `picker.quick_start_eta` (20 min [target]) at the measured origin rate, at the `Bits4` class or above, under a `permissive` licence. Variants below `Bits4` appear only as `bigger` or `faster` lines labelled "heavily quantized". The best-for-this-hardware model is offered on the next line with its size and ETA, so a user who wants it waits knowingly.
- **The interactive threshold.** 6 tok/s is chosen to stay above reading speed. Roughly 250 words per minute at ≈1.3 tokens per word is ≈5.5 tok/s [estimate]. The background threshold of 1 tok/s is for agent and batch use, where nobody watches tokens arrive. Both are settings: `picker.min_tok_s`, `picker.use`, `picker.context`, `picker.agent_ttft_ceiling` and `picker.quick_start_eta`, added to RFC-0054 §5.2. PROD-E6-T09's study checks the thresholds for three uses (chat, coding agent, background) at 6, 10 and 15 tok/s, with TTFT at 32K shown, on 5–8 participants per Tier 1 OS [target] (OQ-PRD-17).
- **Crash isolation for previews**. Previews with `effort: Full` run the MILP on the ingress node. Each such solve runs in a short-lived probe-mode child process, so a solver crash cannot take down the daemon, its API or its membership. After one crash on an input, that input's digest is recorded in the state directory and later previews of it use the DP only, on every restart. This is the same guard RFC-0011 applies on the coordinator.
- **The DP for ordering**. Each variant the loop examines costs one DP table. RFC-0011 §14.1 targets ≤ 100 ms per table at 8 nodes (≈ 75 ms [estimate]).
  - Previews are pure, so the picker runs the DP previews in parallel, one per physical core.
  - The three MILP refinements run in parallel at SC-31's 2 s limit. That leaves about 1 s of the 3 s budget for ordering: up to 10 tables per core [derived: 1 s ÷ 100 ms], or 80 tables on an 8-core host.
  - The editorial catalogue (§3.2, open question 1) keeps a first run's variant count within that bound for the host classes it targets.
  - If a host has too few cores for it, the picker shows the entries ordered so far at 3 s and fills in the rest as their previews finish, rather than block.
- **What the picker shows.** Each line has the model, variant, download size and ETA, the true disk footprint, the planned context, the estimated tok/s (labelled), the licence class, and a one-phrase reason taken from the dominant cost term ("fits in VRAM", "limited by RAM bandwidth on pc", "needs NVMe: speed depends on routing locality"). A model whose publisher ships it as multimodal says "text only in Syntropy". For MoE models that need NVMe, the estimate is a range over hit rates, as in the vision's tables, and the picker says that the speed depends on routing locality learned during use.
- **Clusters.** The picker uses the whole cluster's graph, so adding a node changes the recommendation. The dashboard shows the recommendation for "this cluster now".

### 4. Presentation rules shared by CLI and dashboard

1. **Every number has a kind.** `Measured` figures show plainly. `Estimate` figures carry "estimate" or `≈` and, on hover or with `--explain`, their basis. `Target` figures appear only in docs and in performance views that compare measured values with targets (SC rows), and are labelled "target".
2. **Predicted next to measured.** Wherever a live measurement has a prediction (per-term step time, tok/s, TTFT, hit rate), both are shown, together with the ratio. A ratio outside ±15% is highlighted, because that is SC-30's bound and the edge of re-placement drift.
3. **Fidelity is never hidden.** An approximate instance or request is labelled with the word "approximate" and its knob names, in the warning colour, on every view that shows it: model lists, instance cards, request rows, the banner. Exact is shown as a neutral "exact" label, so its absence is noticeable.
4. **Degradations are named.** `Ready { degraded }` (RFC-0001 §10.2) shows the declared path in words ("ggml provider for attention on strix", "QUIC instead of RDMA on mac↔studio: RDMA hardware present but not enabled"), with the doctor check that explains it. Certification is named the same way: an instance or response whose `certification` is not `certified` (RFC-0050 §10) is labelled "exact (derived certification)", "exact (uncertified)" or "exact (CI test manifest)", and a response past its family's certified context is labelled "exact up to token N; beyond the certified context after it", never "uncertified", and each `refused` entry is shown in words, for example "prefix reuse off: not certified for this path". The model list shows, per model and backend, whether it is certified (RFC-0060 §6.4).
5. **Canonical vocabulary.** Node, stage, tier kinds, coordinator, ingress, instance, session and sequence, exactly as the glossary defines them. Support tiers are always "Tier 1/2/3", memory tiers always by kind.
6. **Privacy.** Prompts and outputs are never shown by default, in either client. `dashboard.show_prompts` enables prompt previews on the requests view for the local operator, and the setting is per node.
7. **Identity.** Nodes are shown by name. Member lists and detail views show each node's six-word identity fingerprint (RFC-0034 §6.5), and so does the API's local CA (RFC-0050 §11.1.1). Admission shows the 6-digit pairing SAS of ADR-0302 instead, because that is what the user compares between the two devices during a join.
8. **Strings are externalised; v1 is English only**. From M1, every human-facing string in the CLI (messages, prompts, table headings, hints, banners) and the dashboard comes from a message catalogue keyed by a stable ID, with English as the only v1 catalogue. Machine output (`--json`, error codes, check IDs, metric names) is never translated. A CI check fails when a human-facing string is written inline in code, so a later translation needs no code change.
9. **Approximate options are disclosed, never pushed**. The only place either client mentions approximate options for an exact model is the one informational line of §2.5 in `plan` and the preview view. Nothing in `run`, the banner, notices or the performance view suggests switching.

### 5. The dashboard

#### 5.1 Delivery and technology

- **Served by every daemon at `/`** on the API listener, with the same auth rules (RFC-0050 §11.4). Any node shows the whole cluster, because the views read the node's replica of cluster state and fetch other nodes' telemetry through it.
- **Stack** (ADR-0506): TypeScript, Preact with signals, built by Vite. Charts use uPlot (time series) and hand-written SVG (topology, memory bars, layer strips). There is no CSS framework, no web fonts (the system UI font stack with tabular numerals) and no runtime dependency fetched from anywhere.
- **Embedding.** The production bundle is precompressed (brotli and gzip) and embedded in the `cli` crate at build time. Filenames carry content hashes and are served with `Cache-Control: immutable`; `index.html` is served with `no-cache`. A build without Node.js (a plain `cargo build`) embeds a one-page placeholder that explains how to build the dashboard, so Rust contributors are never blocked. Release builds always include the real bundle (RFC-0062).
- **Types.** API types are generated from the Rust structs (RFC-0054 §10.3, ADR-0503), so a field rename breaks the dashboard's type check in CI, not a user's browser.
- **Security headers.** `Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`.
- **Size budget**: ≤ 300 KiB brotli for the initial load, ≤ 600 KiB for every route [target, chosen so that the dashboard loads in well under a second over Wi-Fi from a phone].

#### 5.2 Cluster view (home page)

- **Topology.** Nodes are drawn as cards on a deterministic circular layout, ordered by the ring of the selected instance (or by name when none is selected), so the picture does not move between visits. Each card shows:
  - name, OS and support tier, power (mains or battery %);
  - state (alive, suspected, failed, draining);
  - coordinator and ingress roles;
  - devices with backend;
  - one bar per tier kind showing used against usable.
- **Links.** Lines between cards, labelled with transport and medium (`RDMA TB5`, `QUIC 2.5GbE`, `QUIC Wi-Fi`), α p50/p99 and β. Line weight scales with log β, and links whose p99 is more than 4× their p50 are drawn dashed. Demotions (RDMA → QUIC, ADR-0011 decision 7) are shown with their cause.
- **Instance rings.** Selecting an instance overlays its ring as numbered arrows: stage index, layer range and predicted stage time on each node, hop time on each link, with the tail's token path to each active ingress drawn thinner.
- **Notices.** Pending admissions, degraded instances, pending config restarts, available updates, doctor failures, and a notice naming the proposing node on every change to an alias or a model's policy (RFC-0054 §2.3). A permanent, non-dismissable banner reads "API traffic on this LAN is unencrypted" while any node has `api.allow_plaintext_lan = true` (RFC-0050 §11.1.1). While the cluster has a single keyholder, a quiet status line says what stops while it is away (§2.6); it never interrupts.

#### 5.3 Node view: memory, devices, storage

- **Per-tier bars** with purpose segments. `Vram` follows the RFC-0001 §9.3 order: context reserve, scratch, dense hot set, KV hot window, expert cache, headroom kept, other processes. `HostPinned` shows its sub-arenas (RFC-0001 §9.1): RAM-homed weights, RAM expert cache, KV spill, I/O staging, network pools. Lines mark the pin limit, the wired limit (macOS) and `ram_headroom`. A progress band shows progressive pinning while it runs.
- **Memory-pressure panel.** Current level, the reclaim steps taken and the bytes returned, each with its predicted token-time cost (RFC-0001 §10.3), for example "yielded 8.0 GiB: RAM expert cache over NVMe; +31 ms/token on deepseek-v3 (estimate)". A headroom slider previews the cost of keeping N GiB for the user (a preview call) before writing `resource.ram_headroom`.
- **Devices.** Measured `W_eff` per tier and kernel class against the published peak, with ADR-0014 feature bits. A probe below half of the peak is flagged "possible misconfiguration" (vision principle 2) and linked to the doctor check.
- **Storage.** Store usage by model, packs, sessions and free space against `store.min_free_bytes`; the disk this node gives to second copies of shard source chunks for repair (RFC-0033 §9.3), against its cap `store.standby_max_bytes` (RFC-0041 §9) and evicted before any other chunk; sustained and burst NVMe read rates; temperature and throttle events over time.
- **Runners.** Each runner with state, uptime, restarts, `t_fixed`, and CPU spin time (ADR-0012).

#### 5.4 Placement preview

The view behind "Load a model". The user picks a model (store, catalogue or Hub reference) and variant, then sets objective, context length, sequences, fidelity (approximate knobs only after choosing "approximate", each with its RFC-0042 cost report), include and exclude nodes, and optional pins. "Preview" calls `POST /v1/syntropy/placement/preview` for up to three candidates. Each candidate column shows:
- a fit verdict, with the binding constraint in words;
- decode tok/s and TTFT at 1K/4K/32K, labelled estimates, with a per-term breakdown bar (the plan's `PredictedCost`); decode is the mean, and a candidate whose ring crosses a Wi-Fi or overlay hop carries a **jitter badge** beside it with the predicted stall rate and p99 token time, so the tail is visible next to the mean;
- **per-node memory deltas**: for every node and tier, a before/after bar split by purpose, and the change in GiB;
- downloads per node with sources (cluster node or Hub) and an ETA from measured link rates;
- the transition strategy (make-before-break or break-before-make) and any evictions;
- assumptions that are priors, flagged "not yet measured".

Below the columns, the one informational line about approximate options appears under §2.5's rule, and only there (M3). A sibling table lists the other variants with their estimates. "Load" submits `POST /v1/syntropy/instances` with the chosen policy. The committed plan is solved again at commit time, so if cluster state changed in between, the instance page shows the committed plan's difference from the preview.

#### 5.5 Plan explanation

For a committed plan (`…/plan/explain`):
- the stage table: node, layer range, mode, predicted step time, measured step time;
- a **layer strip**: one column per layer, coloured by node, with stacked marks for where its dense groups live and what share of its experts the cache holds;
- clicking a layer lists its tensor groups with home tier, compute site and the reason RFC-0011 recorded: a pin, a binding capacity constraint, or the cost term that chose it with "moving it to X would change token time by +Y ms (estimate)";
- the assumptions, degradations and the plan's predicted per-term costs against measured ones.

#### 5.6 Performance

Per instance, updated at 1 Hz:
- decode tok/s, TTFT and TPOT p50/p99, queue depth and live sequences;
- **predicted against measured per cost term**, as a stacked bar of the plan's terms beside the measured terms (RFC-0053 §2.2), with the ratio per term;
- expert cache: a heat map of hit rate by layer over time, hits against misses by resolution (GPU hit, CPU in place, PCIe copy, NVMe demand read), prefetch usefulness, and the h(c) curve the solver uses next to the observed hit rate;
- NVMe read rate, KV usage per tier, link α per hop, speculative acceptance rate and tokens per window, credit stalls.

**Smoke test.** A button sends a fixed 256-token prompt and generates 128 tokens through the native API, then shows TTFT, tok/s and fidelity. It is labelled "single run, not a benchmark" and links to `syntropy bench`.

#### 5.7 Models and downloads

Store contents per node (bytes per model, packs, sessions), with delete. Pulls in progress per node: bytes, rate, and sources per chunk range: cluster node or Hub (ADR-0016). Each model shows its licence identifier and class. A custom-licence model shows its summary, a link to the full text and an "I accept" button that records the acceptance on this node; a gated model shows what must be accepted on the Hub and whether this node's token has done so (ADR-0018 decision 7). A multimodal source model is labelled "text only in Syntropy". The catalogue tab is the picker of §3.3 for this cluster, with the same use question on first open.

#### 5.8 Settings

Two tabs, node and cluster, rendered from the generated schema (RFC-0054 §10.3):
- every effective value with its origin;
- forms typed by the schema, validated by a dry-run reload before saving;
- pending `runner` and `restart` keys as banners;
- proposal conflicts ("your file says X; the cluster has Y, set by node B") with a one-click resolution either way.
Node edits write the user file; cluster edits submit `SetClusterConfig`.

#### 5.9 Invites, admissions and API keys

- **Invites.** "Invite a device" first asks "your own device, or someone else's?"; someone else's makes it ingress-only by default. It creates a token with a TTL, use count and scope, and shows it as text, as the invite URL, as a QR code drawn in the browser by a bundled encoder, and as the join command. Outstanding invites show expiry and remaining uses, with revoke.
- **Admissions.** A "join *name*?" card for each pending node, with hardware summary, power class, number matching on the 6-digit SAS and approve and deny buttons. The card carries the choices of §2.6 that apply to the node: role (runs model stages, or client only), the coordinator grant and keyholder offer for a wired, mains-powered desktop (both unticked, each with its one-line cost), or "Reach this cluster away from home?" for a battery-powered node (default yes). On the joining node, the same card shows the SAS it expects. An "open pairing window" control mirrors `syntropy cluster open`. RFC-0030 owns the protocol, and ADR-0302 the policy.
- **Members.** Each member with its six-word fingerprint, its roles and a revoke button. RFC-0034 decides what revocation does to data the node already holds; the dialog states it. Removing the last keyholder of a cluster with other members is refused with §2.6's guard, offering to promote another member first.
- **Keyholders and recovery.** When a cluster is founded, a one-time screen offers the offline recovery code (§2.6), shows it once and asks the user to confirm they stored it. The members view shows the keyholder count and whether a recovery code exists, the status line of §2.6, "recover with code" and, when the last keyholder is lost, "start a new cluster from here" (`cluster refound`), with what that costs.
- **API keys.** List, create (shown once), revoke, with scope and last use.

#### 5.10 Data flow and refresh rates

| Data | Source | Rate |
|---|---|---|
| cluster state, plans, runner states | `GET /v1/syntropy/events` (SSE), resumed with `Last-Event-ID` | on change |
| per-instance and per-node live metrics | `GET /v1/syntropy/metrics/live` (SSE) | 1 Hz while subscribed |
| remote node telemetry (tier usage, temperature) | node-query channel through the local daemon (RFC-0050 amendment A4) | 1 Hz while a node view is open; nothing otherwise |
| static data (manifests, graph, explanations) | `GET` on demand | on navigation |

Nothing polls when no dashboard is open. The daemon computes the live aggregates only while at least one SSE subscriber exists.

#### 5.11 Visual design and accessibility

- Colour carries meaning only, and every colour-coded state also has a text label: exact (neutral), approximate (amber), degraded (orange), failed (red), measured against predicted beyond ±15% (outlined). Palettes pass a colour-vision-deficiency check, and light and dark themes follow `prefers-color-scheme`.
- Charts use tabular numerals and SI or IEC units consistent with the CLI. Every chart has a table view for screen readers and copying.
- Usable at 360 px width with a 16 px gutter and no horizontal page scroll. Topology collapses to a list on narrow screens.
- Full keyboard navigation and visible focus. Live regions announce state changes, but not every metric tick.
- **These rules are part of every dashboard task's definition of done from M1**, checked by the automated audits in Testing on each route as it lands. The M4 task is an audit of the whole dashboard, not the first time the rules apply.

#### 5.12 Chat (M2)

One minimal conversation view, so a user who installed from an app icon or the Start menu can talk to a loaded model without another client. It is not a full chat client.
- **What it does.** One conversation at a time with one loaded model: a message box, streamed replies with reasoning folded above the answer, stop, retry and "new conversation". The fidelity label and the banner of §2.4 sit above the conversation, as everywhere else (§4 rule 3).
- **How.** It is a client of the native API only: `POST /v1/syntropy/generate` with a native session (`Syntropy-Session`, RFC-0050 §9.3) per conversation, so a long conversation prefills only each new turn. It needs the `sessions` and `inference` scopes, which the dashboard's session carries on loopback after the one-time login (RFC-0050 §11.4).
- **What it leaves out.** No conversation history beyond the current session, no sync between devices or browsers, no attachments, no plugins or tools, no system-prompt library. Closing the view ends the conversation; the session's KV follows `kv.autosave` like any other session.
- **Privacy.** Conversation text exists only in the browser tab and the session's KV. It is never logged, never in a diagnostic bundle, and never shown in any other view; `dashboard.show_prompts` does not apply to it.
- **The smoke test** of §5.6 stays: a fixed prompt, labelled "single run, not a benchmark".

## Performance model and budgets

None of these touch the token path, but they decide whether the tools feel instant.

| Item | Budget [target] | Basis |
|---|---|---|
| CLI process start to first output for a client command | ≤ 50 ms | chosen budget: below the threshold where a command feels slow; the binary is large, so command dispatch must not initialise providers, iroh or HiGHS |
| `syntropy ps`, `ls`, `cluster status` round trip over the local socket | ≤ 100 ms p99 | chosen budget; the daemon answers from its in-memory replica |
| `syntropy run` time from Enter to the first streamed token, excluding model TTFT | ≤ 5 ms over the RFC-0050 ingress budget | the REPL adds only rendering |
| Picker on first run (order the catalogue, refine 3 with the MILP) | ≤ 3 s | three MILP previews in parallel at SC-31's 2 s limit, plus ≈ 1 s of DP ordering: one table per examined variant at ≤ 100 ms each (RFC-0011 §14.1 [target]), run one per physical core, so ≤ 10 × cores tables [derived]; beyond that, results fill in progressively (§3.3) |
| Dashboard initial load | ≤ 300 KiB brotli; interactive in ≤ 1 s on loopback | size budget of §5.1 |
| Daemon cost while one dashboard is open | ≤ 1% of one core | 1 Hz aggregates computed from telemetry the daemon already receives; remote telemetry at 1 Hz over the node-query channel |
| Daemon cost with no dashboard open | zero beyond normal telemetry ingestion | §5.10 |
| Topology rendering | ≤ 16 ms per frame for 16 nodes and 3 instance rings | 60 fps on a phone-class device; RFC-0062 runs a headless render benchmark |

Estimates shown to users carry the cost model's error. SC-30 commits predicted `t_tok` to within ±15% of measured on the reference rigs, and the dashboard's predicted-against-measured view is where users and developers see whether that holds.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Separate CLI binary that links the API client only | two artefacts to sign and version for no gain; ADR-0015 already chose one binary |
| CLI reading the state directory and store directly | two code paths that can disagree with the daemon; the daemon is the single reader and writer |
| Daemon tied to the CLI session (exits when `run` exits) | unloads models that took minutes to load; surprising for a long-running server |
| Dashboard in Rust compiled to WebAssembly | shares types without a generator, but bundles are larger, chart libraries are weaker, and every UI change pays Rust compile times. Types are shared through generation instead (ADR-0506) |
| Server-rendered HTML with a small hypermedia library | no JavaScript build, but the preview and explanation views are highly interactive, and charts need JavaScript anyway (ADR-0506) |
| React, Svelte or Solid | React is heavier at runtime for the same component model as Preact; Svelte and Solid are fine but have smaller pools of familiar contributors and charting examples (ADR-0506) |
| A desktop app shell for the dashboard | a second runtime to sign and update; the browser is already there |
| Ordering models by public benchmark scores | Syntropy would be importing capability claims it cannot reproduce; editorial catalogue order plus measured fit is honest about what is known |
| Showing a single tok/s figure for NVMe-dependent MoE models | it hides that speed depends on learned routing locality; a range over hit rates is what the cost model actually supports |

## Testing and validation

| Claim | Test |
|---|---|
| `--json` output is the API's schema | contract tests: every `--json` command's output deserialises into the native API type it claims, and the OpenAPI document validates it |
| Exit codes follow the table | per-command integration tests against a daemon on the `syntropy-testkit` fake runner, including each error class |
| No prompt without a terminal | tests run every command with stdin closed and with `TERM=dumb`; any prompt is a failure |
| Human output is stable and readable | golden snapshots (`insta`) of tables, banners and error messages, including the approximate banner and the fit-failure hint; reviewed when they change |
| Labels are never missing | property test over random `FitEstimate` and live-metric payloads: every rendered number in CLI and dashboard output has a kind label where the payload says estimate or target |
| Fidelity is always shown | dashboard component tests and CLI snapshots: an approximate instance renders "approximate" and its knobs on every view that lists it |
| The picker behaves | simulated resource graphs for the vision's scenarios (S1–S5): the recommendation is the earliest catalogue entry, in catalogue order, meeting the threshold; NVMe-dependent MoE shows a range; a cluster change changes the recommendation |
| Dashboard works end to end | Playwright tests against a daemon with fake runners: preview → load → explanation → performance; invite creation; admission approval; settings edit with a validation error; all at 360 px and desktop widths |
| Accessibility | automated audits (axe) in CI on every route, plus keyboard-only navigation tests |
| Budgets hold | CLI start-time benchmark in CI; bundle-size check that fails the build above the §5.1 budget; daemon CPU with one dashboard open measured in RFC-0061's soak test |
| No external requests | the Playwright run records every network request and fails on any non-same-origin URL |
| The first run asks one question and recommends a safe default | simulated graphs for S1–S5 with recorded origin rates: chat and agent answers set the planned context; the recommendation is never below `Bits4`, never `custom`-licensed, and its ETA is within `picker.quick_start_eta`; a slower, bigger model appears as the `bigger` line with its ETA; every line shows the true `disk_needed` and the planned context; the SC-24 flow counts one question and one confirmation |
| Licences are accepted, never assumed | a custom-licence entry cannot be pulled with `--yes` alone, only with `--accept-licence` or an "I accept", and the acceptance is recorded once per model and node; a gated model's pull prints the Hub step and never calls a Hub acceptance API |
| Pairing approval offers only what applies | simulator e2e: a wired mains desktop shows the coordinator grant and keyholder offer unticked, each with its cost line; a laptop sees neither and is asked "Reach this cluster away from home?" with yes preselected; "client only" admits an ingress-only member; a device marked as someone else's defaults to client only |
| Keyholder safety | `uninstall`, `cluster leave` and `cluster remove` refuse to drop the last keyholder of a multi-member cluster and succeed after a promotion or with `--force`; the recovery-code screen appears once at founding; `cluster recover` with the code rotates the cluster key; `cluster refound` creates a cluster the survivors join by `confirm` pairing with their stores intact; the single-keyholder status line is shown and never interrupts |
| Approximate options are disclosed, never pushed | with a knob carrying a `KnobCost` report and a predicted 1.6× gain, `plan` and the preview view show exactly one informational line; at 1.4×, or without a report, neither shows it; `run`, the banner and every other view never mention it |
| Strings are externalised | a CI lint fails on an inline human-facing string in the `cli` crate or the dashboard; the English catalogue covers every message ID the snapshots render |
| The chat view stays minimal and private | Playwright: a conversation streams through the native API on one session, a second turn reports `cached_tokens` for the first, "new conversation" starts a new session; the canary test finds no conversation text in logs, bundles or any other view |

## Rollout / milestones

| Milestone | Lands |
|---|---|
| M0 | `serve`, `run`/`chat` (single prompt and a basic REPL), `ls`, `stop`, `probe`, `bench` (RFC-0061 harness), `config show/check/path`, `version`, `completions`; output conventions, `--json` and exit codes; `debug state`, `debug parity` and `debug telemetry` for the M0 parity harness and benchmark driver |
| M1 | `pull` (with licence acceptance), `ps`, `show`, `plan` (one node), `unload`, `rm`, `doctor` (with `--bundle`), `logs`, `api-key`, `config set/unset/edit/reload`, `setup`, `service` (with the macOS `--daemon`), `update`, `uninstall`, `trust`, `dashboard` with the one-time login URL; the full REPL; the catalogue with its curation rules and licence classes, and the picker with the use question, the quick-start default, the `Bits4` floor and true disk figures; the dashboard with node, performance, preview (one node), explanation, models and settings views, each meeting §5.11; message catalogues for every human-facing string (§4 rule 8); the model picker driven by the resource graph (vision M1 checklist) |
| M2 | cluster view with topology and links; `cluster` commands with the pairing approval's role and grant choices, the battery-node relay question, keyholder status, `recover`, `refound` and the last-keyholder guard; the recovery-code screen; `trust --cluster`; multi-node previews with per-node memory deltas; remote telemetry; the chat view (§5.12) |
| M3 | `invite` (with the own-or-someone-else's question) and the invites and admissions views; the second keyholder offer; session commands in the REPL; approximate-knob forms with RFC-0042 cost reports and the one informational line (§2.5); Linux `tune` for the TTM `pages_limit`; the vision's M3 dashboard item complete |
| M4 | `tray`; `provenance`; accessibility audit clean; dashboard size and render budgets enforced as release gates |

## Open questions

1. **Who curates the catalogue, and how is `order` justified? (answered)** Agents and users propose entries with a public rationale; the review agent checks the rationale rule, and entries merge through the same gates and review as code; a 7-day freshness target [target] for releases in supported families; entry only after conformance on a Tier 1 backend; a licence class, gated flag and review date on each entry; a quarterly sweep; the licence policy for downloads (§3.2).
2. **Picker thresholds.** 6 tok/s interactive and 1 tok/s background are reasoned defaults, not measured preferences. PROD-E6-T09's study, run with the M1 alpha testers on their own hardware, checks three uses (chat, coding agent, background) at 6, 10 and 15 tok/s with TTFT at 32K shown, and fixes `picker.agent_ttft_ceiling` (§3.3; OQ-PRD-17).
3. **Hit-rate ranges for NVMe-dependent MoE.** The picker shows a range over hit rates from the uniform prior up to the DRAM cap. Once RFC-0023's persisted frequency profiles exist, can a profile learned on one machine predict another's hit rate well enough to show a single number? Measure across reference rigs.
4. **Join prompts on headless nodes (answered, OQ-PRD-19).** Both sides show the SAS. The approver shows the joiner's name, hardware and SAS with number matching; the joiner prints its SAS in `setup` during the first join, in `cluster status` and on its dashboard (§2.6).
5. **Terminal QR codes (decision dropped, OQ-PRD-20).** `invite create` always prints the token text and the invite URL beside the code, so a code that fails to scan never blocks a join (§2.6). PROD-E6-T12 still checks the test terminals.

## Proposed amendments to RFC-0001

- **A1. §3.1.** The client subcommand list should match §2.1 here. The RFC-0001 text names `ps`, `models`, `bench`, `run`, `pull`, `chat` and `stop`; this RFC adds `ls` (with `models` as an alias), `plan`, `unload`, `rm`, `show`, `cluster`, `invite`, `probe`, `doctor`, `logs`, `debug`, `config`, `api-key`, `identity`, `trust`, `dashboard`, `setup`, `service`, `update`, `uninstall`, `tray`, `completions` and `version`. It also adds a hidden `__selftest` mode (RFC-0052 §7.3) beside `__runner`.
- **A2. §5.** Add `Figure<T> { value, kind: Measured | Estimate | Target, basis }` as the type in which the API returns every predicted or measured number to clients (§3.1), so labelling is a property of the data and not of each view.
- **A3. §3.1.** Add the commands this RFC's 2026-10-20 revision introduces: `cluster recover`, `cluster refound`, `cluster remove`, and `provenance` (M4).
