# RFC-0054: Configuration and policy
- Status: Draft
- Area: Product surface and operations
- Depends on: RFC-0001 (§2.3, §3, §5, §7.2, §8.4, §9.2, §10.3), ADR-0000 (D2, D3, D4), ADR-0007, ADR-0008, ADR-0009, ADR-0014, ADR-0015, ADR-0019, ADR-0500, ADR-0501
- Date: 2026-10-19

## Summary

Syntropy must produce its first token with no configuration file (SC-24). It must still let an operator pin layers to a machine, cap how much RAM a laptop gives up, or opt one model into an approximate knob. RFC-0001 §11 fixes the scopes, the layering and the policy types. This RFC fills in the rest:

- **One optional TOML file per node** (ADR-0500), with a schema written once as Rust types. Every key has a type, a default (often computed from the probed hardware), a scope, a reload class and an owning RFC.
- **Four scopes** (RFC-0001 §11.1, ADR-0501):
  - *Node* keys resolve locally through `defaults < config file < env < CLI`.
  - *Cluster* keys come only from the event log. A node's file proposes them; it never overrides them.
  - *Instance* keys form the `InstancePolicy` that `EnsureInstance` carries. Their defaults for automatic placement live in cluster-wide model policy tables, so every ingress node builds the same policy.
  - *Request* keys only narrow what the instance allows.
- **Policy objects** from RFC-0001 §11.3, with fields this RFC adds: `Objective`, `ExactnessPolicy` with `ApproxKnobs` and per-knob parameters, `ResourcePolicy` (headroom, reserved cores, battery behaviour), `NetworkScope`, and `InstancePolicy` (context, keep-alive, node filters, determinism).
- **Exactness** follows RFC-0001 §11.3. A request is approximate only when it asks to be, its instance policy allows it, and the cluster allows it. The knob set is the intersection. An exact request that only an approximate plan could serve fails with an error that names the knobs.
- **Per-model overrides.** Placement pins and regex tensor overrides compile to solver constraints, never post-edits (ADR-0008 decision 8).
- **Validation** rejects unknown keys and bad values with file, line and a suggestion. Deprecated keys are accepted for a stated window. The daemon refuses to start on an invalid file.
- **Reload.** A new configuration applies atomically or not at all. Each key is `hot`, `instance` or `node` (RFC-0001 §11.2).
- **A test-only debug surface**: `debug.parity_api`, `debug.bench_api` and `debug.telemetry_api`. All three are off by default in every build, loopback-only, and loud when on, so the quality RFCs can test the signed release artefacts themselves (§7.1).

The runner never reads configuration itself. The daemon resolves everything once and hands the runner its resolved config at spawn (RFC-0001 §3.5, §11.2).

## Motivation and goals / non-goals

Three pressures shape this design.

1. **Zero configuration is a product requirement**, not a convenience (D4, SC-24). Every default must be safe and good on a personal machine that is also somebody's desktop. Defaults that depend on hardware (RAM headroom, reserved cores, VRAM headroom) are computed from the probe, not hard-coded.
2. **A cluster needs agreement on some settings.** Ingress routing runs on every node from its replica of cluster state (RFC-0001 §6.1). If two ingress nodes held different exactness policies for the same model, they would call `EnsureInstance` with different policies and create duplicate instances, or route an exact request where it must not go. Model policies and re-placement thresholds therefore need one cluster-wide value. Other settings, such as how much RAM a laptop keeps for its owner, are inherently per node.
3. **Overrides are how users already steer placement.** The research found two override forms users know: a per-module device map and regex placement per tensor (`tiered-memory-inference.md` §3.3). The solver must accept both as constraints, so that an override cannot produce a plan that breaks a per-node budget.

### Goals

- G1. No file needed before the first token. A missing file is the normal case.
- G2. One schema, written once in Rust, that generates the parser, the validator, the reference docs, a JSON Schema for editors, environment variable names and the reload classifier.
- G3. Deterministic resolution: the same inputs give the same effective configuration on every node, and every effective value can report where it came from (`syntropy config show --resolved`).
- G4. Cluster-scope values stay consistent across nodes and survive coordinator changes and split-heal (ADR-0009 decision 7).
- G5. Approximate behaviour is impossible to enable by accident: every knob is opt-in per model or per request, named, and visible (D3, ADR-0019 decisions 6–7).
- G6. A configuration change never fails an in-flight request. It may defer itself until a plan change or a restart, and it says so.
- G7. Secrets (API keys, Hub tokens) never appear in `config show`, logs, the event log in clear, or debug bundles.

### Non-goals

- Templating, includes beyond one drop-in directory, conditionals or scripting in configuration files.
- Remote configuration management for fleets. A household cluster has one source of truth, the event log, plus each node's file.
- Kubernetes custom resources or operators (RFC-0001 §1.2).
- Per-user policy inside one node. Syntropy has API keys with scopes (RFC-0050 §11), not user accounts.

## Design

### 1. File format and locations

**Format.** TOML 1.0 (ADR-0500). The first line of every file Syntropy writes is `#:schema file://<path-to-the-schema-shipped-with-this-install>`, a local path so an editor opening the file contacts no server, so editors with a TOML language server offer completion, and the file carries `config_version = 1`. Files are UTF-8; a byte-order mark is accepted and ignored.

**Where the daemon looks.** The *config file layer* is assembled from up to four files, merged in this order (later wins, field by field):

| Order | File | Linux | macOS | Windows |
|---|---|---|---|---|
| 1 | system file | `/etc/syntropy/config.toml` | `/Library/Application Support/Syntropy/config.toml` | `%ProgramData%\Syntropy\config.toml` |
| 2 | system drop-ins, lexical order | `/etc/syntropy/config.d/*.toml` | `/Library/Application Support/Syntropy/config.d/*.toml` | `%ProgramData%\Syntropy\config.d\*.toml` |
| 3 | user file | `$XDG_CONFIG_HOME/syntropy/config.toml` (default `~/.config/…`) | `~/Library/Application Support/Syntropy/config.toml` | `%LOCALAPPDATA%\Syntropy\config.toml` |
| 4 | explicit file | `--config <path>` or `SYNTROPY_CONFIG` | same | same |

- An explicit file (order 4) **replaces** orders 1–3 rather than merging over them, so `--config` gives a fully known configuration for tests and containers.
- A daemon running as a system service (RFC-0052 §6) reads orders 1–2 only; the service account has no user file.
- Drop-ins exist for packagers and administrators: a distro package can ship `config.d/50-distro.toml` without touching the user's file.
- `LOCALAPPDATA` rather than the roaming `APPDATA` on Windows, because most keys describe this machine's hardware and must not follow a user profile to another PC.

**State and data directories** (defaults; `node.state_dir` and `store.dirs` override them):

| Purpose | Linux (user) | Linux (system service) | macOS | Windows | Windows (per-machine service) |
|---|---|---|---|---|---|
| state: identity key, proposal ledger, API key hashes, host-change ledger, logs | `$XDG_STATE_HOME/syntropy` | `/var/lib/syntropy/state` | `~/Library/Application Support/Syntropy/state` | `%LOCALAPPDATA%\Syntropy\state` | `%ProgramData%\Syntropy\state` |
| model store: chunks, packs, session blobs | `$XDG_DATA_HOME/syntropy/store` | `/var/lib/syntropy/store` | `~/Library/Application Support/Syntropy/store` | `%LOCALAPPDATA%\Syntropy\store` | `%ProgramData%\Syntropy\store` |
| runtime: local API socket, runner adoption sockets | `$XDG_RUNTIME_DIR/syntropy` | `/run/syntropy` | `$TMPDIR/syntropy` (per user) | named pipes `\\.\pipe\syntropy-<sid>-*` | named pipes, with the service account's SID |

*Note 2026-10-22 (correction):* the Windows (per-machine service) column was missing. RFC-0052 §5.1 already decided it. A per-machine install runs as the service `NT SERVICE\Syntropy` (RFC-0052 §6) and keeps its state and store beside its config, in `%ProgramData%\Syntropy`, where its update service reads requests (RFC-0052 §7.1). A macOS LaunchDaemon (RFC-0052 §6.2) keeps its state and store beside the system config file, in `/Library/Application Support/Syntropy/{state,store}`. The maintainers decided this on 2026-10-22 (OQ-PRD-26, settled).

The store is deliberately not under a cache directory. Packs are safe to delete (ADR-0005), but the store also holds verified source chunks that cost a download to replace, and cache directories often sit on small volumes.

**What the file looks like.** Most users never write one. A user who wants one model in throughput mode, a laptop that keeps more RAM for itself, and a LAN API looks like this:

```toml
#:schema file:///usr/share/syntropy/schema/config-1.json   # the schema shipped with this install; no network fetch
config_version = 1

[api]
bind = ["0.0.0.0:7433"]            # non-loopback: an API key and TLS are now required; TLS comes from
                                   # the node's local CA by default (RFC-0050 §11.1)

[resource]
ram_headroom = "16GiB"             # node scope: this machine only

[policy."qwen3-235b-a22b*"]        # cluster scope: proposed to the cluster (§2.3)
objective = "throughput"
max_sequences = 6
```

### 2. Scopes, layers and precedence

#### 2.1 Scopes

RFC-0001 §11.1 fixes four scopes. Every key has exactly one, declared in the schema (§10.3), and never changes scope within a `config_version`:

| Scope | Lives in | Keys here (examples) | Who sets it |
|---|---|---|---|
| `node` | the node's config file, environment, CLI | `api.*`, `resource.*`, `store.*`, `io.*`, `net.relay_mode`, `security.trusted_fabric`, `logging.*`, `update.*`, `features.disable`, `debug.*`, `aliases` | the node's operator |
| `cluster` | the event log, through `SetClusterConfig` → `ClusterConfigChanged` (RFC-0001 §5.10) | `allow_approximate`, `network_scope`, `cluster.lan_admission` (RFC-0030), `replan.*` and `solver.*` (RFC-0011), the model policy tables `policy.*` (§6.1) | the operator of a member that holds `STAGE_HOST`; `keyholder`-authority keys also need a keyholder's signature (§2.3 step 10); the coordinator orders the patches |
| `instance` | `InstancePolicy` on `EnsureInstance` (§4.5) | objective, exactness policy and knobs, pins, regex tensor overrides, speculative mode, context length, keep-alive | the client or operator that loads the model; for automatic placement, the cluster's model policy tables (§2.2 Rule 3) |
| `request` | the API request | `fidelity`, sampling parameters, `max_tokens`, session ID, priority | the API client, within the instance's limits (§7) |

The `policy.*` tables are cluster-scoped storage for **default instance policies**. When ingress places a model automatically (RFC-0050 §4.3), it builds the `InstancePolicy` from them, so every ingress node builds the same policy and `EnsureInstance` stays idempotent. When an operator loads a model explicitly (`syntropy run --objective throughput`, `POST /v1/syntropy/instances`), the explicit values override the table for that instance.

#### 2.2 Layers and precedence

**Rule 1: node keys** resolve through `default → config file (orders 1–4 of §1) → environment → CLI`. Later layers win (RFC-0001 §11.1).

**Rule 2: cluster keys** come only from the event log: `default → cluster log`. A node's file, environment or CLI cannot override them. A value found there for a cluster key is a *proposal* that the daemon submits to the log (§2.3). The effective value is whatever the log holds, so every node computes the same answer.

**Rule 3: instance keys** resolve, most specific first: explicit `EnsureInstance` values → `policy."<model id>"` → `policy."<exact name>"` → `policy."<glob>"` (longest literal prefix wins, then table order) → `policy.default` → the schema default. A node-level `--set` cannot change an instance key, because instance keys are not node keys. An operator changes one by editing the model policy (`syntropy config set --cluster 'policy."qwen3*".objective' throughput`) or by loading the instance with explicit values.

**Rule 4: request keys** can only narrow what the instance allows (RFC-0001 §11.1). Exactness follows RFC-0001 §11.3: a request is approximate only if it asks for approximate, its instance's policy is `Approximate`, and the cluster key `allow_approximate` is true. Its knob set is the intersection of what it asks for and what the instance allows (§7).

```text
effective(key):
    match key.scope:
        node     => last of [cli, env, file(4..1), default] that sets key
        cluster  => cluster_log.get(key) or default
        instance => first of [explicit, policy[id], policy[name], policy[glob…], policy.default, default] that sets key
        request  => narrow(request.get(key), effective_instance(key))       // §7
```

Every resolved value carries an `Origin`, printed by `syntropy config show --resolved` (RFC-0001 §11.2):

```rust
pub enum Origin {
    Default { computed: Option<&'static str> },   // rule text for computed defaults, e.g. "max(6GiB, 10% of RAM)"
    File { path: PathBuf, line: u32 },
    Env { var: String },
    Cli { flag: String },
    ClusterLog { index: u64, term: u64, proposed_by: NodeId },
    Explicit { by: ClientMeta },                  // instance keys set on EnsureInstance
    Request,
}
```

#### 2.3 How a node proposes cluster values

A cluster key in a node's file, environment or CLI is a **proposal**. Every `SetClusterConfig` this node submits, whatever its source (file, environment, CLI, `config set --cluster`, the dashboard or the native API), is journalled by RFC-0032 §6.10's command journal before it is sent. The *proposal ledger* is this RFC's per-key view of that journal, kept in the state directory: the key, the BLAKE3 of the value, the source, and the `(term, index)` at which the coordinator accepted it.

1. At start and on every reload, the daemon computes the cluster keys present in its file/env/CLI layers.
2. For each key whose value hash differs from the ledger entry, it submits one `SetClusterConfig { patch }` with `ConfigPatch { set: [(key, value)], unset: [], origin: ProposalOrigin { node, layer, path } }`. The patch carries, per key, the `(index, term)` of the value this node last saw (RFC-0032 §6.10).
3. The coordinator validates the patch against the schema (§8) and appends `ClusterConfigChanged`. The proposing node records `(term, index)` in its ledger.
4. A key that has not changed on this node since its last accepted proposal is not proposed again, so two nodes with different files do not fight. The last accepted proposal wins, and `config show` on the other node reports the conflict: "your file sets `policy."qwen3*".objective = "throughput"`; the cluster has `"latency"`, set by node `mac-studio` at index 4127".
5. **Split and heal** (ADR-0009 decision 7). After a change of coordinator term or a heal that discards log entries, the journal re-submits every `SetClusterConfig` whose `CommandId` is absent from the log, whatever its source (RFC-0032 §6.10). A heal also re-proposes every ledger entry whose accepted `(term, index)` lies after the fork on the discarded branch, even when its journal entry has aged out, with `fork` set to the last entry the two logs shared (RFC-0032 §6.10, §7.5 step 2). No cluster setting is ever recorded only in the log.
6. **Conflict.** Each patch carries a per-key `base` (step 2), and a re-proposal also carries its `fork`. `decide()` applies RFC-0032 §6.10's rules:
   - a patch without `fork` is rejected if the key changed after `base`, which means a concurrent edit;
   - a re-proposal with `fork` applies only if the key has not changed in the surviving log since the fork point;
   - a `fork` the coordinator cannot confirm, because it is older than the retained log, counts as a change.

   A rejection is `ConfigConflict { key, current }`. The surviving value stands, so a stale value never overwrites a newer one. The submitter's operator sees the conflict in `config show --resolved`, in the reply to `config set --cluster` (error `cluster_proposal_rejected`) and as a dashboard notice. Each shows both values, the node that set the current one, and the command that re-applies the dropped value deliberately.
7. **Cluster-wide API keys** exist (RFC-0034 §8.4), but they are not configuration. `ApiKeyCreated` and `ApiKeyRevoked` are keyholder-signed events. Revocations are keyholder records, re-submitted permanently, so a merge never undoes them (RFC-0032 §6.10).
8. Settings changed through `syntropy config set --cluster`, the dashboard or the native API (`PATCH /v1/syntropy/cluster/config`) are not written to any node's file. They are still in the submitting node's ledger. `config show` marks them `ClusterLog { proposed_by: <node> }`.
9. **Mixed versions**.
   - Every cluster-scope key declares `introduced_at`, the cluster feature level that introduced it, in the schema (§10.3; RFC-0032 §6.4).
   - `decide()` rejects a proposal for a key above `ClusterState::feature_level` with `FeatureLevelTooLow { needed, cluster }`. The CLI and dashboard report it as `cluster_proposal_rejected`, naming the nodes that must upgrade.
   - A node applying `ClusterConfigChanged` with a key its schema does not know keeps the value as an opaque, stored entry. It passes the value on in snapshots, never interprets it, and logs one warning per key.
   - Unknown keys are therefore rejected only in files and at the CLI, as §8 says, never in the log.
10. **Authority** (RFC-0034 §8.3).
   - **Only `STAGE_HOST` members propose.** A member without `STAGE_HOST`, such as an ingress-only guest or a client-only desktop, never submits `SetClusterConfig`. Cluster keys in its file are not proposed, `config show` lists them as ignored with the reason, and the coordinator's `decide()` rejects a patch from such a member with `cluster_proposal_rejected` (RFC-0034 owns the rule).
   - Every cluster-scope key declares `authority = member | keyholder` in the schema (§10.3). The default is `member`.
   - Every key under `security.*`, `invite.*`, `coordinator.*` and `swim.*`, plus `cluster.lan_admission`, is `keyholder`; the derive macro enforces those prefixes.
   - **Keys that change behaviour other users rely on are `keyholder` too**, and the derive macro enforces them by name: `cluster_aliases`; `network_scope` (D4's boundary); `allow_approximate` (D3); every cluster key under `update.*`; and, in the `policy.*` tables, the exactness fields (`fidelity`, `default_fidelity`, every `approximate.*` knob) and the pin fields (`pin`, `tensor_override`). Routine tuning (`objective`, `keep_alive`, `replan.*`, `solver.*` and the rest) stays `member`.
   - **Aliases cannot shadow.** A `cluster_aliases` entry, or a node's `aliases` entry, whose name equals a registered model's canonical name or unique short form fails validation (`alias_shadows_model`, code 16), because aliases resolve before canonical names (RFC-0050 §4.1) and a shadowing alias would redirect every client silently.
   - **Every alias or policy change is visible.** Applying a `ClusterConfigChanged` that touches `cluster_aliases` or any `policy.*` key raises a dashboard notice and a `cluster events` line naming the proposing node and the keys (RFC-0051 §5.2).
   - A `ClusterConfigChanged` that sets a `keyholder` key carries a keyholder signature, and every node's `validate()` checks it.
   - On a keyholder node, `config set --cluster` signs the patch after the operator confirms.
   - Any other node, and any file-sourced `keyholder` key on such a node, sends the patch to a keyholder for an operator approval, as RFC-0034 §8.4 does for API keys. Until approval, `config show` reports the patch as pending.

A new cluster starts with the schema defaults plus whatever its first members propose. A node that joins an existing cluster proposes the cluster keys in its own file once; if the operator does not want that, they remove them from the file before joining. `syntropy cluster join` prints the cluster keys the node is about to propose and asks for confirmation on a TTY.

#### 2.4 Environment variables and CLI flags

- **Environment.** Every node key `a.b.c` maps to `SYNTROPY_A__B__C`: upper case, `.` becomes `__`, `_` stays. Example: `resource.ram_headroom` is `SYNTROPY_RESOURCE__RAM_HEADROOM=16GiB`. Values are parsed as TOML values, so arrays are `["a","b"]`, and a bare string without quotes is accepted for string-typed keys. Five fixed names exist outside that pattern:
  - `SYNTROPY_CONFIG`: the explicit file;
  - `SYNTROPY_HOME`: sets `node.state_dir` and the default store;
  - `HF_TOKEN`: read as the node's Hub token (RFC-0041 §7.5, stored as RFC-0034 §4 says) when none is configured, matching the Hub's own convention;
  - `SYNTROPY_HOST` and `SYNTROPY_API_KEY`: read only by CLI client commands, to reach a remote daemon (RFC-0051 §1).
- **CLI.** `syntropy serve --set key=value` is repeatable and accepts any key. A small set of keys also have dedicated flags, listed in RFC-0051 §2.3: `--bind`, `--config`, `--models-dir`, `--offline`, `--log-level`, `--state-dir`.
- Unknown `SYNTROPY_*` variables are reported as warnings at start (a typo in an environment variable is otherwise invisible). Unknown `--set` keys are errors.

*Note 2026-10-22 (PROD-E1-T02, how the layers read):*
- `SYNTROPY_HOME=H` puts the state directory at `H/state` and the default store at `H/store`. That is the layout §1's macOS and Windows defaults use under their `Syntropy` directory, and the Linux service layout under `/var/lib/syntropy`. `SYNTROPY_NODE__STATE_DIR`, when also set, wins for `node.state_dir`.
- An environment or `--set` value that does not parse as a TOML value is read as a string, so `16GiB` and `0.0.0.0:7433` need no quotes. A key whose type is a string or a path always takes the text as written, unless it is a quoted TOML string.
- An empty `SYNTROPY_CONFIG` or `SYNTROPY_HOME` counts as unset. When both `--config` and `SYNTROPY_CONFIG` are given, the flag wins, as the CLI layer wins over the environment.
- Two more `SYNTROPY_*` names draw no unknown-variable warning, because other parts of Syntropy read them and they are not config keys: `SYNTROPY_FAULT_PLAN`, the fault plan a fault-injection build loads at start (RFC-0060 §9.2), and `SYNTROPY_KERNELS`, a build setting of the `kernels-*` crates (COMP-E9-T02). Renaming either out of the `SYNTROPY_` prefix is the alternative, and that is for the maintainers.
- A system service that is given `--config` or `SYNTROPY_CONFIG` reads that file and nothing else, as any daemon does. §1's "reads orders 1–2 only" is read as: a service never reads a user file (order 3). It does not exclude order 4.
- A table-valued node key, such as `node.labels`, `aliases` or `obs.otlp.headers`, is one value. A later layer that sets it replaces it whole, as an array is replaced, and its origin is that layer's line. §1's "later wins, field by field" is read as key by key. Neither §1 nor Rule 1 says whether a table-valued key's entries merge across files. Merging them, as §6.1 merges policy tables, is the alternative, for the maintainers; it would need an origin per entry.
- A drop-in is a `*.toml` file whose name does not start with `.`, compared without case on Windows. "Lexical order" is the byte order of the file name, so `10-a.toml` sorts before `9-b.toml`.

### 3. Value types and units

Configuration values map onto the RFC-0001 §5.1 unit newtypes. A bare integer is never accepted where a unit is meant, except bytes, where an integer means bytes.

| Schema type | Accepted forms | Parsed into | Notes |
|---|---|---|---|
| `ByteSize` | `"16GiB"`, `"512MiB"`, `"1.5TiB"`, `"20GB"`, integer bytes | `Bytes` | IEC (`KiB`…`TiB`) and SI (`KB`…`TB`) both accepted and kept distinct; output always IEC |
| `Duration` | `"250ms"`, `"10s"`, `"5m"`, `"2h"`, `"7d"`, compound `"1m30s"` | `Nanos` | `"never"` where the schema allows `Option<Duration>` |
| `Fraction` | `0.2`, `"20%"` | `f32` in [0, 1] | |
| `Reserve` | `"8GiB"`, `"10%"`, `{ min = "6GiB", fraction = 0.10 }`, `{ min = "6GiB", fraction = 0.10, max = "32GiB" }` | `Reserve { min: Bytes, fraction: f32, max: Option<Bytes> }` | resolves to `clamp(max(min, fraction × capacity), .., max)` against the tier's physical capacity |
| `Rate` | `"5tok/s"`, `"2.5GB/s"` | `f32` tokens/s or `BytesPerSec` | |
| `NodeSelector` | `"pc"` (node name), `"id:3f9a…"` (node ID prefix, ≥ 8 hex), `"label:gpu=rtx4090"`, `"self"` (node scope only: rejected in `policy.*` tables and every other cluster-scope value, because it would mean a different node on each ingress), `"*"` | resolved against cluster membership at use time | an unresolvable selector is a validation warning in files and an error in requests |
| `DeviceSelector` | `"cuda:0"`, `"metal:0"`, `"cpu:numa1"`, `"<node>/cuda:1"`, `"kind:dgpu"` | `DeviceId` set | ordinals follow RFC-0001 §5.1 probe order |
| `TierSelector` | `"vram"`, `"unified"`, `"ram"` (= `HostPinned`), `"ram-pageable"`, `"nvme"`, optionally `"<node>/nvme:1"` | `TierKind`, or a `TierId` | |
| `LayerRange` | `"0..20"` (half-open, as RFC-0001), `"40.."`, `"..8"`, `"7"` | `Range<LayerIx>` | |
| `ModelSelector` | exact name, glob (`*`, `?`), `"id:<blake3 hex prefix>"`, `"default"` | matched against resolved model names and IDs (RFC-0050 §4) | |
| `Regex` | TOML literal string `'blk\.(\d+)\.ffn_.*_exps'` | Rust `regex` (linear time, no backtracking) | always anchored as a full match on the tensor name |
| `Secret` | string, or `{ file = "/path" }`, or `{ env = "VAR" }` | `Secret<String>` (§10.1) | never printed |

Durations, sizes and fractions in output (`config show`, the dashboard) are printed in the same syntax, so any printed value can be pasted back.

*Note 2026-10-21 (open, for the maintainers):* PROD-E1-T01 accepts a few forms beyond this table, so that every value prints in a form that parses back:
- `Duration` also takes `us` and `ns`, because a `Nanos` value need not be a whole millisecond.
- `ByteSize` also takes `B` and `kB`, and a fractional size rounds to the nearest byte.

Options:
- (a) keep these forms, as implemented;
- (b) accept only the listed forms and print a sub-millisecond duration as a decimal of `ms`.

Recommended default: (a).

### 4. Policy objects

RFC-0001 §11.3 defines these types in `syntropy-core`, and §5.13 assigns them to this RFC. They are plain data with invariant checks and serde defaults, so the event log, IPC and the solver can carry them. This RFC adds fields under the RFC-0001 §5 rules and renames nothing.

*Note 2026-10-22 (OQ-PRD-27):* the field types these objects hold that `config` parses (`Reserve`, `Fraction`, the selectors behind `DeviceFilter` and `NodeFilter`, tensor-override patterns, `KeyPath` and `ConfigValue`) have plain-data forms in `core`, so the objects can live there under RFC-0001 §4.2 rule 1. `config` parses text into them and validates it. Patterns are held in `core` as their source text, and `config` keeps the compiled regex.

*Note 2026-10-22 (#1672):* `config` still defines its own `Fraction`, `Reserve`, `NodeSelector`, `DeviceSelector`, `TierSelector` and `LayerRange`, which match `core`'s today. PROD-E1-T23 moves `config` onto `core`'s types and deletes the copies, so the two cannot drift.

*Note 2026-10-22 (PROD-E1-T23, as built):* `config` no longer has copies of those six types. Its keys hold `core`'s types and read §3's forms through `#[serde(with = "syntropy_config::form")]`, which also serves an `Option` or `Vec` of one. The `ConfigSection` derive rejects a key of one of these types without it, because their own serde form is the record encoding, which reads no text form. `config` keeps the text forms (`syntropy_config::form::Form`), their validation messages and `Config` codes, and the compiled regex. `Reserve::resolve` is only in `core`.

#### 4.1 `Objective`

RFC-0001 §11.3 defines the type. This RFC adds two optional fields with serde defaults, under the RFC-0001 §5 rules:

```rust
/// What the solver and scheduler optimise. Default: Latency (D2).
/// Owner: RFC-0054 (type), RFC-0011 (objective function), RFC-0012 (scheduling behaviour).
pub enum Objective {
    Latency,                                     // min t_tok, then min TTFT
    Throughput {
        per_sequence_floor_tps: f32,             // RFC-0001
        #[serde(default)] target_concurrency: Option<u16>,  // added: sequences the plan is sized for; None = solver's choice
        #[serde(default)] ttft_ceiling: Option<Nanos>,      // added: admission rejects above this predicted TTFT
    },
}
```

Instance keys: `objective = "latency" | "throughput"`, `per_sequence_floor` (a `Rate`; default `"5tok/s"` in throughput mode), `target_concurrency`, `ttft_ceiling`. In latency mode, how many other sequences may share the instance without delaying the interactive one is `max_sequences` (§5.3), not a field of `Objective`.

#### 4.2 `ExactnessPolicy`, `ApproxKnobs` and knob parameters

`ApproxKnobs` is RFC-0001 §11.3's bitflags type, with its bit positions:

| Bit | Knob | Scope of effect | What it changes |
|---|---|---|---|
| 0 | `EXPERT_SKIP` | per sequence | missed experts not computed |
| 1 | `EXPERT_DEFER` | per sequence | missed experts computed late |
| 2 | `MISS_LOW_PRECISION` | per instance (needs lower-precision expert copies in the pack and cache) | missed experts computed from a lower-precision copy |
| 3 | `TOPK_REDUCE` | per sequence | fewer routed experts than the model specifies |
| 4 | `KV_LOSSY` | per instance (KV layout) | KV stored below the model's KV dtype |
| 5 | `WIRE_COMPRESS` | per instance (hop wire dtype) | activations below the family's passed precision |
| 6 | `APPROX_ATTENTION` | per instance | attention sparsity not in the model's definition |
| 7 | `SPEC_LENIENT` | per sequence | non-exact speculative acceptance |

Each knob has parameters (how many experts `TOPK_REDUCE` keeps, which dtype `KV_LOSSY` uses). They are not bits, so this RFC adds them as a field of the `Approximate` variant. The parameter types themselves belong to RFC-0042:

```rust
pub enum ExactnessPolicy {
    Exact,
    Approximate {
        knobs: ApproxKnobs,                       // RFC-0001
        #[serde(default)] params: ApproxParams,   // added: per-knob parameters (RFC-0042); default = RFC-0042's default per knob
    },
}
pub struct ApproxParams {
    pub expert_skip: Option<ExpertSkipParams>,
    pub expert_defer: Option<ExpertDeferParams>,
    pub miss_low_precision: Option<QuantScheme>,
    pub topk_reduce: Option<TopkReduceParams>,
    pub kv_lossy: Option<KvDType>,
    pub wire_compress: Option<WireDType>,
    pub approx_attention: Option<ApproxAttnParams>,
    pub spec_lenient: Option<LenientParams>,
}
impl ApproxKnobs {
    pub const PER_SEQUENCE: ApproxKnobs;   // EXPERT_SKIP | EXPERT_DEFER | TOPK_REDUCE | SPEC_LENIENT
    pub const PER_INSTANCE: ApproxKnobs;   // MISS_LOW_PRECISION | KV_LOSSY | WIRE_COMPRESS | APPROX_ATTENTION
}
```

**No named preset** (OQ-PRD-06, answered). Approximate mode is per-model knobs only: there is no `fast` or other preset that turns on a bundle of knobs, because a preset hides which knobs are on and D3 requires each to be named. The only place an exact model's approximate options are mentioned is one informational line in `syntropy plan` and the dashboard's preview, under RFC-0051 §2.5's rule (a `KnobCost` report and a predicted gain of at least 1.5× [target]).

**`ApproxFlags`.** RFC-0001 §5.8 puts `approx: ApproxFlags` on `TokenEntry` without defining it. This RFC defines it as the same bit layout: `pub type ApproxFlags = ApproxKnobs;` (amendment A2). A `TokenEntry` reports the knobs that actually changed those tokens.

Two sets are reported to clients (RFC-0050 §10):
- the knobs **enabled** for the request: the effective set after §7's intersection;
- the knobs **applied**: the union of `TokenEntry::approx` over the sequence.
An enabled `EXPERT_SKIP` that never met a miss is enabled but not applied.
#### 4.3 `ResourcePolicy`

RFC-0001 §11.3 defines four fields: `memory_yield`, `max_pinned_fraction`, `power` and `network_scope`. This RFC adds the fields a personal machine needs (serde defaults):

```rust
/// How hard Syntropy may lean on a personal machine. Owner: RFC-0054 (type, defaults),
/// RFC-0020 (enforcement). Node keys set the node's values; an InstancePolicy's copy may only narrow them.
pub struct ResourcePolicy {
    pub memory_yield: MemoryYield,       // RFC-0001: Never | UnderPressure (default) | Aggressive
    pub max_pinned_fraction: Option<f32>, // RFC-0001: cap on HostPinned as a fraction of RAM, per-OS default (RFC-0020); None = unset, the macOS default
    pub power: PowerPolicy,              // RFC-0001: Performance | Balanced (default) | Battery
    pub network_scope: NetworkScope,     // RFC-0001: Lan | LanAndInvites (default)
    // added by this RFC:
    pub mode: UsageMode,                 // Personal (default) | Dedicated: selects the defaults below
    pub ram_headroom: Reserve,           // RAM never pinned or allocated by Syntropy
    pub vram_headroom: Reserve,          // per discrete GPU (RFC-0001 §9.3 "headroom")
    pub reserve_cores: u16,              // cores never given to dispatchers, I/O or the CPU pool
    pub min_battery_percent: u8,         // below this, Balanced acts as Battery (default 30)
    pub gpus: DeviceFilter,              // allow/deny selectors; default: every probed device
    pub macos_raise_wired_limit: bool,   // default false; ADR-0014: never silently
}
```

**Defaults by mode.** `resource.mode` selects a default set; any field can still be set on its own.

| Field | `personal` (default) | `dedicated` | Rationale |
|---|---|---|---|
| `ram_headroom` | `{ min = "6GiB", fraction = 0.10 }` | `{ min = "2GiB", fraction = 0.03 }` | a 128 GB desktop keeps 12.8 GB and gives ≈115 GB to Syntropy, the RAM figure the S1 scenario assumes (vision §4) |
| `vram_headroom` | `{ min = "1GiB", fraction = 0.06 }` | `"512MiB"` | room for a compositor and a browser on the display GPU |
| `reserve_cores` | 2 if ≥ 8 physical cores, else 1 | 0 | the user's foreground app keeps a core |
| `memory_yield` | `under-pressure` | `never` | |
| `power` | `balanced` | `performance` | |

`max_pinned_fraction` has no mode default here. Its per-OS defaults belong to RFC-0020, and `ram_headroom` and the OS pin limit cap it further (RFC-0001 §9.2).

**`memory_yield`** maps OS pressure (RFC-0001 §10.3) onto the reclaim ladder. Syntropy also raises its own signal when available RAM falls below `ram_headroom` (`warn`) or below half of it (`critical`):

| Level | `never` | `under-pressure` (default) | `aggressive` |
|---|---|---|---|
| `warn` | nothing | ladder steps 1–2 (idle-session KV spill, RAM expert cache over an `Nvme` home) | steps 1–4 |
| `critical` | nothing: the operator has accepted the OOM risk; doctor OS-006 warns when it happens | steps 1–4, then `Drift` with reduced usable capacity | steps 1–4, then a re-plan at once rather than after `replan.pressure_window` |

`never` is meant for dedicated machines. Every yield reports its predicted token-time cost (RFC-0001 §10.3), which the dashboard shows next to the reclaimed bytes (RFC-0051 §5.3).

**`power`** (`NodeInfo.power` reports the state):
- `performance`: no change on battery.
- `balanced` (default): on battery, keep current stages, cap spin-waiting at the battery budget (RFC-0001 §8.2), and mark the node so the solver's filter avoids giving it new stages at the next re-plan when an alternative exists. Below `min_battery_percent`, act as `battery`.
- `battery`: on battery, request a re-plan that moves this node's stages elsewhere, and keep serving the API as ingress.

#### 4.4 `NetworkScope` and node network settings

`NetworkScope` is RFC-0001's enum: `Lan | LanAndInvites` (default). It appears twice:
- as the **cluster** key `network_scope`, which decides whether the cluster accepts off-LAN members through invites (D4);
- in an instance's `ResourcePolicy`, where it may narrow one model to LAN members. The solver then places no stage on an invite-joined node for that instance.

Per-node network behaviour is node scope and belongs to the network RFCs:
- relays: `net.relay_mode` (RFC-0030): `auto` (default), `default`, `custom` (the user's own relay list in `net.relays`) and `disabled`. Under `auto`, relays are contacted only while the cluster has invite-joined members or open invites, or a member that answered yes to "Reach this cluster away from home?" when it was paired (asked of battery-powered nodes, default yes), so a desktop-only LAN household never contacts a relay server (RFC-0034 §13). `auto` and `default` use the built-in relay list, a signed TUF target (`relays.json`, RFC-0052 §7.2);
- interfaces Syntropy ignores: `discovery.ignore_interfaces` (RFC-0030);
- bulk transfers on metered links: `net.bulk_on_metered` (RFC-0030);
- RoCEv2 interfaces trusted for plaintext RDMA: `security.trusted_fabric` (RFC-0034, ADR-0011 decision 4).

#### 4.5 `InstancePolicy`

RFC-0001 §11.3 defines six fields. This RFC adds the rest of what a loaded model needs (serde defaults):

```rust
/// Carried by EnsureInstance. EnsureInstance is idempotent by (ModelId, InstancePolicy::digest()).
/// Owner: RFC-0054 (type and construction), RFC-0011 (how the solver uses each field).
pub struct InstancePolicy {
    pub objective: Objective,                  // RFC-0001
    pub exactness: ExactnessPolicy,            // RFC-0001: the knobs this instance allows
    pub resources: ResourceOverlay,            // RFC-0001 names ResourcePolicy here; this RFC needs an overlay of
                                               // Option fields (only explicitly set values), so the digest is
                                               // node-independent (amendment A9)
    pub pins: Vec<PlacementPin>,               // RFC-0001; syntax in §6.2
    pub tensor_overrides: Vec<TensorOverride>, // RFC-0001; syntax in §6.3
    pub speculative: SpeculativeMode,          // RFC-0001: Auto (default) | Off | Force(SpeculativeConfig)
    // added by this RFC:
    pub context: ContextPolicy,                // { context_length: Option<u32>, max_sequences: Option<u16> }; None = solver chooses
    pub nodes: NodeFilter,                     // include / exclude selectors
    pub keep_alive: KeepAlive,                 // Auto (default: max(1 h, 60 × t_reload), §5.4) | For(Nanos) | Never
    pub preload: bool,                         // load at daemon start; never evicted for another model
    pub deterministic: bool,                   // ADR-0019 decision 8 (M3)
    pub dense_on_nvme: DenseNvmePolicy,        // Never | IfOnlyOption (default) | Allow
    pub default_fidelity: Fidelity,            // RFC-0001 §11.3 (A6): what a request that states no fidelity
                                               // asks for; Exact unless set; Approximate requires an
                                               // Approximate `exactness` (§5.4, §7)
    /// Instance-scope keys that domain RFCs own (e.g. RFC-0011 `solver.context_target`, RFC-0032
    /// `ring.max_inflight_steps`, RFC-0033 `recovery.history`), keyed by path, digest-covered.
    pub domain: BTreeMap<KeyPath, ConfigValue>,
}
/// Every field optional; only values set explicitly (tables or EnsureInstance) are present.
pub struct ResourceOverlay {
    pub memory_yield: Option<MemoryYield>, pub max_pinned_fraction: Option<f32>,
    pub power: Option<PowerPolicy>, pub network_scope: Option<NetworkScope>,
    pub ram_headroom: Option<Reserve>, pub vram_headroom: Option<Reserve>,
    pub reserve_cores: Option<u16>, pub gpus: Option<DeviceFilter>,
}
impl InstancePolicy {
    /// BLAKE3 over the canonical serialisation. Two nodes with the same cluster config build
    /// the same policy for automatic placement, and so the same digest.
    pub fn digest(&self) -> Blake3;
}
```

The instance's `resources` overlay is applied on each stage node to that node's own `ResourcePolicy`, and may only narrow it. The solver checks each node against the stricter of the two: more headroom, a lower pinned fraction, a narrower network scope.

*Note 2026-10-22 (PROD-E1-T04): the canonical serialisation and who hashes it.* `core` links no hasher: RFC-0001 §4.2 rule 1 keeps the `blake3` crate out of it, and `core` holds the `Blake3` type only (RFC-0001 §5.1; the `PlanDigest` note in `core`). The digest is therefore split in two:
- `InstancePolicy::canonical_bytes() -> Vec<u8>` is the contract, in `core` (`syntropy_core::canon`): a tagged binary encoding over serde, with integers widened to 64 bits little-endian, floats by bit pattern with one NaN and -0.0 as +0.0, structs and enum variants by field and variant name with fields sorted by name, maps sorted by encoded key, and fields that serde skips left out. Golden vectors are pinned on these bytes.
- `InstancePolicy::digest_with(&self, hash: impl FnOnce(&[u8]) -> Blake3) -> Blake3` replaces `digest()`. The digest is still BLAKE3 over the canonical bytes, computed by the crate that links BLAKE3 (`config` or `api`, when it parses policies) and passes the hash in.

`ResourceOverlay`, `ApproxParams` and the optional `Objective` fields skip unset values, so only explicitly set fields enter the bytes, and a field added later as an unset `Option` leaves earlier digests unchanged. RFC-0001 §18.16 records the change.

*Note 2026-10-22 (#1672):* one function in `config` supplies the hash: BLAKE3 over `canonical_bytes()` through `digest_with`. PROD-E1-T08 builds it, with golden digests, and every caller uses it, so no two nodes hash a policy differently. The open question below still stands; if `digest()` moves into `core`, that function calls it.

Open, for the maintainers: should `digest()` live in `core` with its own hasher (a vendored or hand-written BLAKE3, or an amendment to rule 1), so that no caller can pass a different hash?

**Node independence of the digest**.
- `digest()` covers only what the policy tables and the explicit `EnsureInstance` set. It never covers node-computed defaults. That is why `resources` is an overlay of options, and why `"self"` is rejected in cluster-scope tables.
- A property test builds the policy for one model on two simulated nodes with different hardware and `resource.mode`, and requires equal digests.

**Sequence count and context, one model with RFC-0011**.
- `max_sequences` is the single key for how many sequences an instance holds KV for. RFC-0011 reads it directly; its former `solver.latency_kv_sequences` is gone. Its latency-mode default is **2**, the active sequence and one queued, as RFC-0011 sizes KV. The earlier "4" is withdrawn.
- `context_length = auto` means "what the committed plan can hold for `max_sequences`". The plan publishes that value, and ingress checks prompts against the **planned** value, not the model's maximum.
- A longer request fails with `context_length_exceeded`. The error names the planned limit, the model's maximum, and the policy keys that raise the limit at the next plan (`context_length`, or a lower `max_sequences`). Routing tries every instance's planned limit first, and there is no per-request re-plan (RFC-0050 §4.2; RFC-0011's routing follows the same rule). RFC-0011's live p90 feeds the next re-plan's target as before.

### 5. Key reference

Each crate declares its own config section (RFC-0001 §11.2), and the domain RFC behind that crate owns the section's keys and defaults. This RFC owns the schema machinery, the policy objects, the keys of the product crates, and the rule that no two sections share a prefix. §5.1 is the registry of sections. §5.2–§5.4 list the keys this RFC and the other product RFCs own. Reload classes are `hot`, `instance` and `node` (§9.1).

#### 5.1 Section registry

| Prefix | Owner | Scopes used | Notes |
|---|---|---|---|
| `config.*`, `config_version` | 0054 | node | §5.2 |
| `node.*`, `daemon.*` | 0054 (0001 for the `daemon.*` thread and memory keys) | node | §5.2 |
| `resource.*` | 0054 (type), 0020 (enforcement) | node; narrowed per instance as `resources.*` | §4.3, §5.2 |
| `api.*`, `aliases`, `cluster_aliases` | 0050; the auth keys must meet RFC-0034 §11's requirements R1–R5 | node; `cluster_aliases` cluster | §5.2, §5.3 |
| `dashboard.*`, `picker.*` | 0051 | node | §5.2 |
| `logging.*`, `obs.*` | 0053 | node | §5.2. `log.*` belongs to RFC-0032's event log |
| `update.*` | 0052 | node; `update.cluster_coordinated` and `update.switch_at` cluster | §5.2, §5.3 |
| `debug.*` | 0054 (gates); 0060, 0061 (what they enable) | node | §7.1 |
| `allow_approximate`, `network_scope` | 0054 (RFC-0001 §11.1 names them) | cluster | §5.3 |
| `policy.*` (model policy tables) | 0054 (tables); each key's owner as listed in §5.4 | cluster storage for instance keys | §5.4 |
| `probe.*` | 0010 | node | |
| `solver.*`, `replan.*` (with 0033) | 0011 | cluster; instance for `solver.context_target` | `replan.drift_threshold`, `replan.drift_window`, `replan.pressure_window`, `replan.min_gain`, `replan.payback_horizon`, `replan.max_defer` are the RFC-0001 §7.2 names |
| `sched.*`, `exec.*`, `speculative.*`, `batching.*` | 0012 (`batching.*` with 0032) | instance, node | `speculative.mode`, `.source`, `.draft_model`, `.max_window` appear in policy tables (§5.4) |
| `tier.*` | 0020 | node | |
| `io.*` | 0021 | node | |
| `kv.*` | 0022 | instance, node | `kv.prefix_cache`, `kv.autosave`, `kv.session_retention` replace the `sessions.*` names an earlier draft of this RFC used |
| `experts.*` | 0023 | instance | |
| `cluster.*`, `discovery.*`, `swim.*`, `membership.*`, `invite.*`, `pairing.*`, `power.*`, `net.relay_mode`, `net.relays`, `net.bulk_on_metered`, `net.listen_port`, `members.*` | 0030 | cluster, node | `cluster.lan_admission` (default `confirm`, ADR-0302). `invite.require_confirm` is RFC-0034's, carved out of `invite.*` |
| `link.*`, `quic.*`, `rdma.*`, `shm.*`, other `net.*` | 0031 | node | |
| `log.*`, `coordinator.*`, `ring.*`, `tp.*`, `actions.*`, `orphan.*` | 0032 | cluster; node for `log.group_commit`, `log.follower_queue`; instance for `ring.max_inflight_steps` | |
| `runner.*`, `recovery.*`, `straggler.*`, `standby` | 0033 (RFC-0001 §3.4 names the `runner.*` keys) | node; instance for `recovery.history`, `recovery.drain_deadline` and `standby`; cluster for `recovery.confirm_after` | `standby` is a top-level instance key (RFC-0033 §9.3), listed in §5.4 |
| `security.*`, `integrity.*`, `rdma.encrypt_untrusted`, `invite.require_confirm` | 0034 | node, cluster | `security.trusted_fabric` is the interface list ADR-0011 calls `trusted_fabric`. `rdma.encrypt_untrusted` and `invite.require_confirm` are carved out of RFC-0031's `rdma.*` and RFC-0030's `invite.*` |
| `kernels.*`, `features.disable` | 0040 | node | `kernels.module_dir` defaults to the running version's `modules/` directory (RFC-0052 §4.3); `<store>/modules` is a development override |
| `store.*`, `hub.*`, `p2p.*`, `pack.*`, `offline` | 0041 | node | this RFC adopts RFC-0041's defaults as final: `store.max_bytes` 90% of the volume, `store.pack_max_bytes` 25% of `store.max_bytes`, `store.min_free_bytes` 20 GiB, `store.standby_max_bytes` 10% of the node's writable volumes by total size [target] (the per-node cap on second-copy chunks for repair, RFC-0033 §9.3; 0 turns second copies off; a copy is admitted only if the volume keeps 2 × `store.min_free_bytes` free, RFC-0041 §4.4) |
| `numerics.*` | 0042 | node, instance | |
| `models.*` | 0034 (`models.signatures`, `models.trusted_signers`), 0043 (`models.experimental_family`) | node | node settings about models. Per-model policies use `policy.*`, never `models.*` |

The scopes column lists every scope a section uses. The table is generated from the domain crates' section declarations; it is not hand-written, and the copy above is a snapshot. The schema derive (§10.3) fails the build if two sections register the same prefix. A CI check fails if a key's declared scope differs from the owning RFC's key table.

*Note 2026-10-21 (open, for the maintainers): where a duplicate prefix fails.* A derive sees one struct, so it cannot see a prefix that another crate registers. PROD-E1-T01 puts the check in `syntropy_config::Registry::new`, a `const fn` that panics when two sections claim one prefix. In a `static` or `const` item that panic is a compile error; in a `let` it happens at run time.

Options:
- (a) the binary's registry, which PROD-E6-T23 builds (moved from PROD-E1-T02 on 2026-10-22, with the CLI subcommands), is a `static`, so a duplicate prefix fails that build;
- (b) a link-time check through one exported symbol per prefix, which needs `unsafe` attributes in every crate that declares a section, and RFC-0001 §4.2 rule 8 forbids those;
- (c) only the CI test of PROD-E1-T16.

Recommended default: (a), as implemented, with (c) as the cross-check the paragraph above already plans. The sentence "the schema derive fails the build" stands until the maintainers choose.

*Note 2026-10-21 (open, for the maintainers): keys carved out of another prefix.* The table carves keys out of other sections' prefixes: `net.relay_mode` (0030) sits inside RFC-0031's other `net.*` keys, and `rdma.encrypt_untrusted` (0034) inside `rdma.*`. PROD-E1-T01 expresses this as claims:
- a section owns `prefix.*` by default;
- a section marked `#[config(shared)]`, and every top-level section, owns only its own keys, each by its full path;
- two equal claims fail the build, as in the note above.

Options:
- (a) shared sections, as implemented;
- (b) the table's owner declares the carved-out keys too, with each key naming its own owner;
- (c) no carve-outs: the carved-out keys move under their owners' prefixes, which renames keys this table fixes.

Recommended default: (a). Nothing yet checks a shared section's keys against this table, so a shared section could add a key inside another owner's subtree, such as `security.x` inside RFC-0034's `security.*`. Under (a), PROD-E1-T16's CI check should compare every shared section's keys with the carve-outs this table lists.

*Note 2026-10-22 (DIST-E18-T01, as built; open, for the maintainers): the distributed areas' keys.* DIST-E18-T01 registers the keys of RFC-0030 §9, RFC-0031 §11, RFC-0032 §9, RFC-0033 §12 (with §8.2's `runner.*`) and RFC-0034 §14: 100 keys in 29 sections, in `syntropy_net::config` (RFC-0031) and `syntropy_cluster::config` (the rest). Those RFCs give names, scopes and defaults, but no reload classes and, for several keys, no value syntax. The choices below fix user-facing syntax, so they are recorded here:
- **Reload classes** (§9.1). `node` for a value read once when the daemon builds a long-lived component: `cluster.namespace`, `discovery.port`, `discovery.mechanisms`, `net.listen_port`, `net.max_frame`, `net.blob_frame`, `quic.*`, `rdma.*` (with `rdma.encrypt_untrusted`), `shm.enable`, `security.keystore` and `security.trusted_fabric`. `instance` for a value that changes an instance's plan or runtime layout: `ring.max_inflight_steps`, `ring.allow_token_relay`, `tp.oneshot_max`, `tp.local_router`, `recovery.history`, `recovery.state_checkpoint_tokens`, `recovery.drain_deadline` and `standby`. `hot` for every other key. The test `reload_classes_follow_the_stated_rule` in `crates/cluster/tests/config.rs` pins the list.
- **Feature level.** Every cluster-scope key has `introduced_at = FEATURE_LEVEL_V1`, which is 1. RFC-0032 §6.4 gives no initial level; one shared constant should replace it once `ClusterState::feature_level` exists.
- **Conditional defaults** are `computed` defaults on `Option` keys, with the RFC's rule as the text: `link.connect_timeout` and `link.failover_timeout` (LAN or WAN), `link.p99_weight`, `ring.max_inflight_steps`, `recovery.drain_deadline`, `recovery.confirm_after` (2 × `link.failover_timeout`), `runner.load_stall`, `power.inhibit_idle_sleep` (on battery), `cluster.name` and `discovery.ignore_interfaces`. An operator's value applies in every case, for example to LAN and WAN links alike.
- **Compound values** are tables with these field names: `{ count, window }` for `link.demote_after`, `membership.flap_threshold` and `runner.restart_budget`; `{ initial, max }` for `membership.flap_backoff` and `runner.restart_backoff`; `{ count, window, initial, max }` for `coordinator.backoff`; `{ entries, interval }` for `log.snapshot_every`; and `{ for, joins, include_wifi }` for `pairing.window_default`, after `syntropy cluster open`'s flags. `recovery.history_ram`, "min(2 GiB, 5 % of RAM)", is a `Reserve`: `{ fraction = 0.05, max = "2GiB" }`.
- **`coordinator.lease_ms` and `coordinator.wan_lease_ms` are `Duration`s** (`"4s"`, `"12s"`). §3 accepts no bare integer where a unit is meant, so `lease_ms = 4000`, which the name and RFC-0032's "4,000" invite, is rejected. Renaming the keys to `lease` and `wan_lease` would remove the mismatch.
- **Node names, not `NodeId`s.** `coordinator.prefer` and the nodes of `standby = { warm = { nodes = [...] } }` (RFC-0033 §9.3's `Standby::Warm { nodes: SmallVec<[NodeId; 2]> }`) are lists of node names.
- **Values the RFCs leave open.** `rdma.poller_thread`, `rdma.gpudirect` and `shm.enable` name only `auto`; each also takes `off`, as `rdma.enable` does. `runner.device_reprobe` ("on `Device` class failures") is an enum with the one value `on_device_failure`. `standby` takes `"none"`, `"replica"` or the `warm` table.
- **`replan.*` is split.** The table above gives `replan.*` to RFC-0011 "with 0033". RFC-0011's key table declares `replan.min_gain`, `replan.payback_horizon` and `replan.max_defer`, so the PLC lane registers them, and RFC-0033's shared `replan` section claims only `replan.degraded_min_gain`. RFC-0011 gives `replan.max_defer` a default of 5 min and RFC-0033 §12 gives 10 min. That conflict is open, and no default is chosen here.
- **Gaps in the table above** that these keys expose:
  - `orphan.*` (RFC-0032 §9's `orphan.adopt_deadline`) has no row. It is registered under RFC-0032.
  - RFC-0034's `invite.require_confirm` is a carve-out inside RFC-0030's `invite.*` that the table does not list, so PROD-E1-T16's carve-out check would flag its shared section.
  - The top-level instance key `standby` (RFC-0033 §9.3, §12) appears nowhere in this RFC, neither here nor in §5.4.
  - *Resolved 2026-10-22 (#1679):* the table above now lists `orphan.*` under RFC-0032, the `invite.require_confirm` carve-out under RFC-0034 (and in RFC-0030's row), and `standby` under RFC-0033; §5.4 lists `standby`.

#### 5.2 Node keys owned by the product RFCs

| Key | Type | Default | Reload | Owner |
|---|---|---|---|---|
| `config_version` | int | `1` | — | 0054 |
| `config.watch` | bool | `false`: reload only on `SIGHUP`, `syntropy config reload` or the API (RFC-0001 §11.2) | hot | 0054 |
| `node.name` | string | OS host name, lower-cased, non-alphanumerics → `-` | hot | 0054 |
| `node.labels` | table string→string | `{}` | hot | 0054 (read by 0011's filter) |
| `node.roles.ingress`, `node.roles.host_stages` | bool | `true`, `true` (coordinator eligibility and preference: RFC-0032 `coordinator.prefer`) | hot | 0054 |
| `node.state_dir` | path | §1 | node | 0054 |
| `daemon.ctl_threads` | int | `clamp(physical_cores / 8, 2, 4)` | node | 0001 |
| `daemon.bulk_threads` | int | `2` | node | 0001 |
| `daemon.max_rss` | ByteSize | `"1GiB"` | hot | 0001 |
| `resource.mode` | `personal` \| `dedicated` | `personal` | hot | 0054 |
| `resource.max_pinned_fraction` | Fraction | per OS (RFC-0020) | hot | 0020 |
| `resource.memory_yield` | `never` \| `under-pressure` \| `aggressive` | by mode (§4.3) | hot | 0020 |
| `resource.power` | `performance` \| `balanced` \| `battery` | by mode (§4.3) | hot | 0054 |
| `resource.min_battery_percent` | int | `30` | hot | 0054 |
| `resource.ram_headroom`, `resource.vram_headroom` | Reserve | by mode (§4.3) | hot | 0020 |
| `resource.reserve_cores` | int | by mode (§4.3) | node | 0020 |
| `resource.gpus.allow`, `resource.gpus.deny` | [DeviceSelector] | `["*"]`, `[]` | hot | 0010 |
| `resource.macos_raise_wired_limit` | bool | `false` | node | 0020 |
| `api.bind` | [socket addr] | `["127.0.0.1:7433", "[::1]:7433"]` | hot | 0050 |
| `api.local_socket` | bool | `true` | node | 0050 |
| `api.auth.mode` | `auto` \| `always` | `auto`: keys required on every non-loopback bind, and on loopback HTTP for `admin`, `sessions` and keyholder routes (RFC-0050 §11.1); `always` also on loopback for every route (RFC-0034 R4) | hot | 0050 |
| `api.dashboard_login_ttl` | Duration | `"2m"` [target]: how long an unused one-time dashboard login token stays valid (RFC-0050 §11.4) | hot | 0050 |
| `api.tls` | `auto` \| `local_ca` \| `{ cert = path, key = path }` \| `none` | `auto`: local CA on LAN binds, none on overlay-only binds (RFC-0050 §11.1) | hot | 0050 |
| `api.allow_plaintext_lan` | bool | `false`; `true` allows `api.tls = "none"` on LAN binds, flagged permanently by the dashboard and doctor SEC-005 (RFC-0001 §12) | hot | 0050 |
| `api.server_names`, `api.cors_origins` | [string] | `[]` | hot | 0050 |
| `api.max_body` | ByteSize | `"32MiB"` | hot | 0050 |
| `api.max_buffered_tokens` | int | `256` | hot | 0001, 0050 |
| `api.sse_keepalive` | Duration | `"15s"` | hot | 0050 |
| `api.load_wait` | Duration | `"300s"` | hot | 0050 |
| `api.queue.max`, `api.queue.max_wait` | int, Duration | `64`, `"60s"` | hot | 0050 |
| `api.auto_pull` | bool | `false` | hot | 0050 |
| `api.max_n` | int | `8` | hot | 0050 |
| `api.extensions` | `body+headers` \| `headers` \| `off` | `body+headers` | hot | 0050 |
| `api.openai.reasoning_field` | `reasoning_content` \| `reasoning` \| `inline` \| `omit` | `reasoning_content` | hot | 0050 |
| `api.responses.store`, `api.responses.retention` | bool, Duration | `true`, `"7d"`; records live on the ingress node only, and before M3 every response is stateless whatever the setting (RFC-0050 §3.2, §9.2) | hot | 0050 |
| `api.detach_grace` | Duration | `"60s"` [target]: how long a disconnected native or stored-Responses stream is held paused for re-attach (RFC-0050 §5.5; M3) | hot | 0050 |
| `api.provenance`, `api.provenance.retention` | bool, Duration or unset | `false`, unset (kept until pruned): the opt-in local provenance record (RFC-0050 §10; M4) | hot | 0050 |
| `api.rate_limit.per_key` | requests per minute, or unset | unset | hot | 0050 |
| `api.compat.local_runner` | bool | `false` (M3) | node | 0050 |
| `aliases` | table name→model ref | `{}` | hot | 0050 |
| `dashboard.enabled`, `dashboard.show_prompts` | bool | `true`, `false` | hot | 0051 |
| `picker.min_tok_s` | `{ interactive = Rate, background = Rate }` | `{ interactive = "6tok/s", background = "1tok/s" }` | hot | 0051 |
| `picker.use` | `chat` \| `agent` \| unset | unset: the first run asks once (RFC-0051 §2.4) | hot | 0051 |
| `picker.context` | `{ chat = int, agent = int }` | `{ chat = 8192, agent = 32768 }`: the context the picker plans for, by `picker.use` | hot | 0051 |
| `picker.agent_ttft_ceiling` | Duration | fixed by PROD-E6-T09's study before M1 ships: the TTFT at `picker.context.agent` above which a model is not recommended for agent use | hot | 0051 |
| `picker.quick_start_eta` | Duration | `"20m"` [target]: the longest download the default recommendation may need at the measured origin rate (RFC-0051 §3.3) | hot | 0051 |
| `logging.level` | `error` … `trace` | `info` | hot | 0053 |
| `logging.filter` | string (per-target directives) | `""` | hot | 0053 |
| `logging.format` | `auto` \| `text` \| `json` | `auto` | hot | 0053 |
| `logging.file` | `auto` \| `off` \| path | `auto` | hot | 0053 |
| `logging.include_content` | bool | `false` | hot | 0053 |
| `obs.prometheus` | bool | `false` | hot | 0053 |
| `obs.otlp.endpoint`, `obs.otlp.protocol`, `obs.otlp.headers` | URL, `grpc` \| `http`, table string→Secret | unset, `grpc`, `{}` | hot | 0053 |
| `obs.trace.step_sample` | Fraction | `0.015625` (1 step in 64) | hot | 0053 |
| `obs.trace.honor_remote_sampled` | bool | `false` | hot | 0053 |
| `obs.flight_recorder` | ByteSize | `"64MiB"` | hot | 0053 |
| `obs.step_recorder` | ByteSize | `"32MiB"` per node (RFC-0053 §2.5) | hot | 0053 |
| `update.mode` | `auto` \| `notify` \| `off` \| `package-manager` | by install method (RFC-0052 §7.1) | hot | 0052 |
| `update.channel` | `stable` \| `beta` \| `nightly` | `stable` | hot | 0052 |
| `update.window` | `idle` or `"HH:MM-HH:MM"` local time | `idle` | hot | 0052 |
| `update.require_mains` | bool | `true` | hot | 0052 |
| `update.check_interval` | Duration | `"24h"` | hot | 0052 |
| `update.mirror` | URL or unset | unset: the release host | hot | 0052 |
| `update.local_drain` | Duration | `"120s"`: how long an update waits for local-ingress sequences before restarting (RFC-0052 §7.3 step 6) | hot | 0052 |
| `update.restart_budget` | Duration | `"10s"` [target]: the planned-restart window announced to peers as `back_within_ms` (RFC-0052 §7.3); at most the cluster's `membership.restart_grace` (RFC-0030 §5.9), checked at validation | hot | 0052 |
| `debug.parity_api`, `debug.bench_api`, `debug.telemetry_api` | bool | `false` in every build; loopback only (§7.1) | hot | 0054 (gate); 0060, 0061 |

*Note 2026-10-21 (open, OQ-PRD-25):* the table above declares `api.provenance` as a bool and `api.provenance.retention` beneath it, as RFC-0050 §10 uses them. TOML cannot hold a key as both a value and a table, so no file can set both. The register lists the options and a recommended default.

#### 5.3 Cluster keys owned here

| Key | Type | Default | Reload | Owner |
|---|---|---|---|---|
| `allow_approximate` | bool | `true`: the cluster permits approximate mode where an instance policy and a request both opt in (§7). Authority `keyholder` (§2.3 step 10) | hot | 0054 |
| `network_scope` | `lan` \| `lan+invites` | `lan+invites` (D4). Authority `keyholder` | hot | 0054 (type), 0030 (behaviour) |
| `cluster_aliases` | table name→model ref | `{}`; may not shadow a registered model's name. Authority `keyholder` | hot | 0050 |
| `update.cluster_coordinated` | bool | `true`. Authority `keyholder` | hot | 0052 |
| `update.switch_at` | timestamp or unset | unset; written only by the coordinator during a coordinated update (RFC-0052 §7.5) and rejected from proposals | hot | 0052 |
| `policy.*` | the model policy tables (§5.4, §6) | `policy.default` holds the schema defaults | — (each key's own class) | 0054 |

#### 5.4 Instance keys (model policy tables)

These keys build an `InstancePolicy` (§4.5). In addition, every instance-scope key that a domain section declares, such as `solver.context_target`, `ring.max_inflight_steps` or `recovery.history`, may appear in a `policy.*` table. It travels in `InstancePolicy::domain`, so it reaches the solver through `EnsureInstance` and is covered by the digest. Each may be set in `[policy.default]` or `[policy."<selector>"]`, which live in the cluster log, or explicitly when a model is loaded. Rule 3 (§2.2) decides which applies. "Req" says what a request may do with the key (§7). "Hot" here means the change applies to new requests, or to idle-unload and preload decisions, without a new plan.

| Key | Type | Default | Reload | Req | Owner |
|---|---|---|---|---|---|
| `objective` | `latency` \| `throughput` | `latency` (D2) | instance | — | 0011 |
| `per_sequence_floor` | Rate | `"5tok/s"` (throughput mode only) | instance | — | 0011 |
| `target_concurrency` | int or unset | unset: the solver chooses | instance | — | 0011 |
| `ttft_ceiling` | Duration or unset | unset | hot | — | 0012 |
| `dense_on_nvme` | `never` \| `if-only-option` \| `allow` | `if-only-option` (ADR-0008) | instance | — | 0011 |
| `context_length` | int or `auto` | `auto`: what the committed plan can hold for `max_sequences`; the plan publishes it and ingress checks against it (§4.5) | instance | — | 0022 |
| `max_sequences` | int or `auto` | `auto`: 2 in latency mode (the active sequence and one queued, as RFC-0011 sizes KV), `target_concurrency` or the solver's choice in throughput mode (§4.5) | instance | — | 0012 |
| `keep_alive` | Duration, `auto` or `never` | `auto`: `max(1 h, 60 × t_reload)` [target], where `t_reload = t_pack + t_load` is measured per model (SC-23), so a model whose reload takes minutes is not unloaded over lunch; memory pressure still evicts through the yield ladder. `never` when `preload = true`. The constant is tuned from M1 measurements (open question 4, answered) | hot | — | 0050 |
| `preload` | bool | `false` | hot | — | 0050 |
| `nodes.include`, `nodes.exclude` | [NodeSelector] | `["*"]`, `[]` | instance | — | 0011 |
| `resources.<field>` | any `ResourcePolicy` field (§4.3) | unset: each node's own policy | instance | — | 0020 |
| `standby` | `"none"` \| `"replica"` \| `{ warm = { nodes = [...] } }`, node names (RFC-0033 §9.3) | `"none"` | instance | — | 0033 |
| `pin` | array of tables (§6.2) | `[]` | instance | — | 0011 |
| `tensor_override` | array of tables (§6.3) | `[]` | instance | — | 0011 |
| `fidelity` | `exact` \| `approximate` | `exact` (D3) | instance | narrows only (§7) | 0042 |
| `default_fidelity` | `exact` \| `approximate` | `exact` (D3): what a request that states no fidelity asks for (amendment A6, RFC-0001 §11.3); `approximate` requires `fidelity = "approximate"` (validation, §8) | instance | an explicit request fidelity wins (§7) | 0054 |
| `approximate.expert_skip`, `approximate.expert_defer`, `approximate.topk_reduce`, `approximate.spec_lenient` | parameter table or `false` | `false` | instance | a request may name a subset | 0042 |
| `approximate.miss_low_precision` | quant scheme or `false` | `false` | instance | a request may name it | 0042 |
| `approximate.kv_lossy` | KV dtype or `false` | `false` | instance | a request may name it | 0042, 0022 |
| `approximate.wire_compress` | wire dtype or `false` | `false` | instance | a request may name it | 0042, 0032 |
| `approximate.approx_attention` | parameter table or `false` | `false` | instance | a request may name it | 0042 |
| `deterministic` | bool | `false` (M3, ADR-0019 decision 8) | instance | requires an instance that has it | 0012 |
| `speculative.mode` | `auto` \| `off` \| `force` (`SpeculativeMode`, RFC-0001 §11.3) | `auto` | instance | `off` only | 0012 |
| `speculative.source`, `speculative.draft_model`, `speculative.max_window` | as RFC-0012's `SpeculativeConfig` | unset (used with `force`) | instance | — | 0012 |
| `kv.prefix_cache` | bool | `true` | instance | — | 0022 |
| `kv.autosave` | bool | `true`: persist a session's KV on release (RFC-0022 §8.1) | hot | a request may override per session (RFC-0050 §9) | 0022 |
| `sampling.temperature`, `sampling.top_p`, `sampling.top_k`, `sampling.min_p`, `sampling.repetition_penalty` | float/int | the model's own generation config, else `1.0`, `1.0`, `0` (off), `0.0`, `1.0` | hot | via the API's own fields | 0050 |
| `chat_template` | path or unset | the template in the model's metadata | hot | — | 0050 |
| `reasoning.default` | `auto` \| `on` \| `off` | `auto` (the template's default) | hot | via the API's own fields | 0050 |

An approximate knob carries its RFC-0042 parameters, for example `approximate.topk_reduce = { experts = 6 }`. Enabling any knob on a model without `fidelity = "approximate"` is a validation error. An instance policy is `Approximate` only when a person has written the word "approximate" for that model (G5), and requests still have to ask for it (§7).

### 6. Per-model policy, placement pins and tensor overrides

#### 6.1 Model selectors and merging

`[policy."<selector>"]` tables are matched against the model the request resolved to (RFC-0050 §4). Several tables may match one model; they merge field by field in specificity order (Rule 3). `[policy.default]` matches every model. Arrays (`pin`, `tensor_override`, `nodes.*`) do not merge element-wise: the most specific table that sets the array wins whole, so an override list is never a surprise union of several tables.

The ingress node builds `InstancePolicy` (§4.5) from the effective per-model keys. The function is pure over (cluster config, resolved model), so every node builds the same policy and the same digest, and `EnsureInstance` stays idempotent across ingress nodes.

#### 6.2 Placement pins

A pin is a constraint the solver must satisfy (`strength = "require"`, the default) or a preference it pays a penalty to break (`strength = "prefer"`). Pins never post-edit a solved plan (ADR-0008 decision 8).

```toml
[[policy."deepseek-v3*".pin]]          # layers 0..20 run on the PC
layers = "0..20"
node = "pc"

[[policy."deepseek-v3*".pin]]          # routers in VRAM wherever their layer runs
groups = "router"
tier = "vram"

[[policy."deepseek-v3*".pin]]          # never put a stage on the laptop
exclude = "macbook"

[[policy."deepseek-v3*".pin]]          # keep the tail on the node most requests arrive at
role = "tail"
node = "pc"
strength = "prefer"
```

```rust
/// Owner: RFC-0054 (syntax and meaning), RFC-0011 (encoding as MILP constraints).
pub struct PlacementPin { pub kind: PinKind, pub strength: PinStrength }
pub enum PinStrength { Require, Prefer { weight: f32 } }   // weight default 1.0 = one TPOT-ms
pub enum PinKind {
    /// Layers in `layers` belong to stages on a node matching `node`.
    Layers { layers: Range<LayerIx>, node: NodeSelector },
    /// Groups selected by class (and optional layer range) have their home on this tier kind,
    /// optionally on a specific node or tier.
    Tier { groups: GroupSelector, tier: TierSelector },
    /// Groups selected execute on this device at decode (ComputeSite::decode).
    Compute { groups: GroupSelector, device: DeviceSelector },
    /// The HEAD, TAIL or DRAFT role runs on a matching node.
    Role { role: StageRole, node: NodeSelector },
    /// No stage, home tier or compute site on matching nodes or devices.
    Exclude { target: ExcludeTarget },
}
/// "dense" | "expert" | "lookup" | "head" | "router" | "attention" | "ffn" | "all",
/// optionally with `layers = "a..b"` and, for experts, `experts = "0..64"`.
pub struct GroupSelector { pub class: ClassSel, pub layers: Option<Range<LayerIx>>, pub experts: Option<Range<ExpertIx>> }
```

*Note 2026-10-22 (PROD-E1-T04): pins and overrides as data in `core`.* PROD-E1-T04 holds pins and tensor overrides in `core` as the policy tables write them, for PROD-E1-T09 to compile:
- `PinKind::Layers::layers` and `GroupSelector::layers` are `LayerRange` (start and end each optional, §3), not `Range<LayerIx>`, so a table can name `"40.."` before the model's layer count is known. `GroupSelector::experts` stays `Option<Range<ExpertIx>>`.
- `ExcludeTarget`, which the struct above names but does not define, is `Node(NodeSelector) | Device(DeviceSelector)`.
- §6.3's override is `TensorOverride { pattern: String, layers: Option<LayerRange>, experts: Option<Range<ExpertIx>>, tier: TierSelector, compute: Option<DeviceSelector> }`. The TOML key `match` is the field `pattern`, because `match` is a Rust keyword. The pattern is its source text (OQ-PRD-27), and `config` keeps the compiled regex.

Before solving, the solver front end checks pins against the census and the resource graph. Selectors that match nothing, tier kinds a node does not have, and pairs of `require` pins that contradict each other are reported before any MILP runs. An infeasible pin set is an error naming the smallest conflicting subset the front end found (RFC-0011 owns the search; the API surfaces it as `placement_infeasible`, RFC-0050 §12).

#### 6.3 Regex tensor overrides

Users already know one more override form: a regular expression over tensor names with a target device or tier (`tiered-memory-inference.md` §3.3). Syntropy accepts it and compiles it to `Tier` and `Compute` pins.

```toml
[[policy."deepseek-v3*".tensor_override]]
match = 'blk\.(\d+)\.ffn_(up|gate|down)_exps\.weight'   # GGUF names
layers = "40.."                                          # optional narrowing
tier = "nvme"                                            # home tier kind
compute = "cpu"                                          # optional decode device
```

Compilation:

```text
for each override o (in file order):
    T = { t in census.tensors | o.match fully matches t.name  and  layer(t) in o.layers }
    if T is empty: warning "override matches no tensor" (error under `config check --strict`)
    G = { group(t) | t in T }
    for g in G:
        missing = g.tensors − T
        if missing ≠ ∅ and not g.is_expert_slice_of(T):
            error "override splits group {g}: it selects {T∩g} but not {missing}"
    emit Tier { groups: G, tier: o.tier } and, if set, Compute { groups: G, device: o.compute }
a later override that selects a group an earlier one already pinned replaces the earlier pin for that group
```

Two details matter:
- **Groups, not tensors.** The solver assigns tensor groups (RFC-0001 §5.3), so an override that splits a group, for example the gate matrix of an expert without its up and down matrices, is rejected rather than silently widened.
- **Fused expert tensors.** GGUF stores all experts of a layer in one tensor (`{n_embd, n_ff, n_expert}`), which the census splits into one group per expert by byte range (RFC-0041). A regex that matches the fused tensor selects every per-expert group of that layer. The optional `experts = "a..b"` key narrows it further.

Names are matched in the source format's own naming (GGUF or safetensors), because those are the names users see in model cards and in other tools. `syntropy show <model> --tensors` prints them with their group and census class (RFC-0051).

#### 6.4 Node filters

`nodes.include` and `nodes.exclude` apply before the solver's filter→score front end (ADR-0008 decision 2). `node.roles.host_stages = false` on a node excludes it from every model; `nodes.exclude` excludes it from one. A filtered-out node can still be the ingress node for the model.

### 7. Per-request overrides

Each API front end (RFC-0050 §3) extracts request-level policy into one shape-neutral struct:

```rust
/// Owner: RFC-0054 (fields and rules), RFC-0050 (how each API shape carries them).
pub struct RequestOverrides {
    pub fidelity: Option<Fidelity>,              // Exact | Approximate; None = the instance's default_fidelity (Exact unless set; D3, A6)
    pub knobs: Option<ApproxKnobs>,              // knobs the request asks for; None with Approximate = all the instance allows
    pub priority: Option<Priority>,              // Interactive | Normal | Background
    pub speculative_off: bool,
    pub deterministic: Option<bool>,
    pub session: Option<SessionBinding>,         // RFC-0050 §9
    pub persist: Option<bool>,                   // overrides kv.autosave for this session (RFC-0022)
    pub trace: bool,                             // record every step span (RFC-0053 §3.3)
    pub bench: Option<BenchControls>,            // { ignore_eos, gen_tokens }: only with debug.bench_api (§7.1)
}
```

Resolution at ingress follows RFC-0001 §11.3. A request is approximate only if three things hold: it asks for approximate, the instance policy is `Approximate`, and the cluster key `allow_approximate` is true. Its knob set is the intersection.

```text
resolve_request(req, inst: InstancePolicy, cluster) -> Result<EffectiveRequestPolicy, Error>:
    fid = req.fidelity.unwrap_or(inst.default_fidelity)                 // A6: the operator's per-model opt-in
    if req.knobs is set and fid != Approximate:
        fail InvalidRequest code=knobs_need_approximate                 // contradictory input
    wants_approx = fid == Approximate
    if wants_approx and inst.exactness is Approximate { knobs: allowed, params }
                    and cluster.allow_approximate:
        knobs = req.knobs.unwrap_or(allowed) ∩ allowed
        if knobs is empty: resolve to Exact, warning approximate_not_available
        else: return Approximate { knobs, params }                     // per-sequence part goes in SequenceOpen
    // every other case resolves to Exact
    if wants_approx: warning approximate_not_available                  // the response says exact (RFC-0050 §10)
    if inst.exactness has any PER_INSTANCE knob:                        // the plan itself is approximate
        fail InvalidRequest code=fidelity_conflict, naming the knobs    // ADR-0019 §7, RFC-0001 §11.3
    return Exact
```

**Consequences of the rule.**
- An exact request can be served by an instance whose policy allows only per-sequence knobs. The knobs are simply not used for it.
- An exact request cannot be served by an instance whose plan carries instance-wide knobs (lossy KV, compressed wire, low-precision miss copies, approximate attention). Ingress first looks for another routable instance of the model that is exact. If there is none, the request fails with `fidelity_conflict`.
- A request that asks for approximate where it is not available is served exactly, and the response says so. Exactness is never weakened by surprise, and an approximate request never fails merely because exact service is available.
- Stock clients that cannot send extension fields still reach approximate service through the header form (RFC-0050 §3.4).
- **A per-model default** (amendment A6, now in RFC-0001 §11.3; `InstancePolicy::default_fidelity`, §4.5, §5.4). An instance policy may set `default_fidelity = approximate`. A request that states no fidelity is then treated as asking for approximate, still subject to `allow_approximate`. An explicit `fidelity: exact` is served exact or fails. Adopting A6 needs ADR-0019 decision 7 ("the default is `exact`") superseded by a new ADR, and RFC-0050 G4 counts the per-model default as the operator's opt-in. Labels are unchanged: such responses still say "approximate" and name their knobs.

A request may switch speculation off but cannot force it on. `deterministic = true` requires an instance whose `InstancePolicy::deterministic` is set. If none exists, ingress requests one through `EnsureInstance` like any other policy, subject to capacity.

**Priority.** OpenAI's `service_tier = "flex"` maps to `Background`; every other value maps to `Normal`. The native API can request `Interactive` (RFC-0050 §3.4). In latency mode, `Interactive` is the sequence whose decode step goes first (RFC-0001 §6.2). At most one sequence per instance holds `Interactive` at a time; a second `Interactive` request is admitted as `Normal`, and the response says so.

#### 7.1 Test-only request controls

Three node keys gate a debug surface that the quality RFCs need on real, signed release builds. RFC-0060 §1.3 runs parity on the exact artefacts that ship, and RFC-0061 §10.2 benchmarks them:

| Key | Enables | Requested by |
|---|---|---|
| `debug.parity_api` | `POST /v1/syntropy/parity`: teacher-forced sequences with logit taps (`ParityTapSpec`, RFC-0060 §5.5) | RFC-0060 |
| `debug.bench_api` | `BenchControls` on `POST /v1/syntropy/generate`: `ignore_eos = true` and an exact `gen_tokens` | RFC-0061 §3.1 |
| `debug.telemetry_api` | `GET /v1/syntropy/debug/telemetry`: the per-step record drain (RFC-0053 §2.5) | RFC-0061 §10.2 |

Rules common to all three:
- Default `false` in every build profile, release included. They are enabled only by an explicit setting in the file, environment or CLI. No build or install sets them.
- **Loopback only.** The endpoints answer only on loopback listeners and the local socket. From any other peer they return 404, whatever key the caller holds, so their existence is not advertised off the host.
- **Visible when on.** While any of them is on, the daemon logs a warning at start and on every reload, `syntropy doctor` reports CFG-003, and the dashboard shows a banner.
- **Request validation.** `BenchControls` on a request without `debug.bench_api` fails with `InvalidRequest` (`debug_api_disabled`). It is never silently ignored, because a benchmark that silently stopped at EOS would report wrong token counts.
- **Labels.** Responses produced with `ignore_eos` carry `bench` in `syntropy.warnings`. Their fidelity label is unchanged: every token is still sampled exactly; generation just does not stop at end-of-sequence.
- **Conformance manifests signed by the CI test key** (RFC-0060 §6.4).
  - The daemon's manifest loader accepts a `ConformanceManifest` signed by the CI test key only when two conditions hold. First, `debug.parity_api` or `debug.bench_api` is `true`. Second, every `api.bind` listener is loopback; the local socket always qualifies.
  - Otherwise the loader refuses the manifest with `manifest_signer_not_allowed` (code 15) and logs which condition failed. The node then runs as if it had no valid manifest: ggml provider, single-device shapes, F32 wire, everything labelled "uncertified" (RFC-0060 §6.4).
  - The condition covers API listeners only. The cluster plane may still bind the lab network, because multi-node L5 parity needs it. No client off the host can reach such a node's API.
  - While such a manifest is loaded, a reload that would clear both gates or add a non-loopback API listener is rejected as a whole (§9), and the error names the manifest's signer. To change those keys, restart with a release- or nightly-signed manifest.
  - While such a manifest is loaded, every response's fidelity metadata carries `certification: ci_test` (RFC-0050 §10), and doctor CFG-003 names the signer.

### 8. Validation

Validation runs in six phases. Phases 1–5 run on every load and reload; phase 6 runs when a model policy is used.

| Phase | Checks | Failure |
|---|---|---|
| 1. Syntax | TOML 1.0 | error with file, line, column |
| 2. Schema | every key known for this `config_version`; correct scope for where it appears (a `request` key in a file is an error; an `instance` key outside a `policy.*` table is an error); arrays and tables where expected | unknown key → error with an edit-distance suggestion; see the version rule below |
| 3. Types and ranges | unit syntax (§3), enums, numeric ranges, regex compiles, selectors parse | error |
| 4. Cross-field | knob tables require `fidelity = "approximate"`; so does `default_fidelity = "approximate"`; a non-loopback `api.bind` needs at least one API key, and the daemon refuses to start without one, naming the command (RFC-0001 §12, RFC-0050 §11.1); `api.tls = "none"` is rejected unless every bind address is on an overlay interface or `api.allow_plaintext_lan = true`; a `{ cert, key }` pair must both exist; a `debug.*` key set to `true` is accepted with a warning (§7.1); `ram_headroom` below the RFC-0020 minimum reserve is raised to it with a warning | error, or warning where stated |
| 5. Environment | paths exist and are writable; store volumes have direct-I/O support (warning only, RFC-0021 falls back); interface globs match at least one interface (warning); config file holding a secret is readable by others (warning on Unix if mode allows group or other read) | warning, except unwritable `node.state_dir`, which is an error |
| 6. Model-time | pins and overrides against the census and resource graph (§6.2, §6.3); `context_length` above the model's maximum | `EnsureInstance` rejected with `placement_infeasible` or `InvalidRequest`; the daemon keeps running |

**Unknown, deprecated and newer keys.** Unknown keys are rejected at load, as RFC-0001 §11.2 requires.
- **Deprecated keys.** A key that a release renames or removes stays accepted, with a deprecation warning naming its replacement, for one minor release before 1.0 and one major release after (RFC-0062 §10.1). After that it is rejected as unknown. The schema records each deprecated key with its replacement, and `syntropy config migrate` rewrites files.
- **Newer files.** A file whose `config_version` is higher than the binary's is rejected, with an error that names both versions. The one place this happens in normal use is an update rollback. For that case the updater keeps a copy of every config file from before the update and restores it during rollback (RFC-0052 §7.4).
- A file with no `config_version` is treated as version 1.

**Error format.** Errors are `Error { class: InvalidRequest, code: (Config, n) }` (RFC-0001 §5.12), rendered with the source line:

```text
error[config/0003]: unknown key `resource.ram_headroon`
  --> /home/u/.config/syntropy/config.toml:14:1
   |
14 | ram_headroon = [redacted]
   | ^^^^^^^^^^^^ did you mean `ram_headroom`?
```

*Note 2026-10-22:* a rendered error or warning never contains a secret's value (G7, §3, §10.1). PROD-E1-T03 chooses the mechanism; this is the property it must hold:
- **Every line or source that holds a secret-typed value is withheld, whichever key the error is on.**
  - An error on a sibling key in the same inline table withholds the secret on that line, as with a type error on `endpoint` in `otlp = { endpoint = "x", headers = { authorization = "Bearer …" } }`.
  - So does an error on another key of the same `[[…]]` entry, or on a continuation line of a multi-line value.
- **Secret-typed** means a key whose type contains `Secret<…>` or `SecretSource` at any depth: through tables, arrays, arrays of tables and struct fields, as `obs.otlp.headers` does. Every key under such a key is secret-typed too.
- **Unknown keys.** An `unknown_key` error never prints the value. The value is not needed to fix a key name, and a mistyped secret key, such as `header` for `headers`, is the likeliest way a secret reaches an error.
- **Every source.** A value from `--set` or a `SYNTROPY_*` variable is withheld the same way. So is the value in the warning for an unknown `SYNTROPY_*` variable, which may be a misspelt secret variable.
- **Every phase.** A phase 1 syntax error has no parsed value, as with an unquoted `api_key = sk-…`, so withholding keys off the key path in the schema, not off a parsed value. Where the renderer cannot tell which key a line sets, it withholds the line whenever a secret-typed value could sit there.
- **Never the library's text.** The renderer never prints the TOML library's rendered error, which appends the source line in every phase. It renders from the error's span and message.
- **Drop-ins.** Lines are attributed one file at a time, since a table header never carries over from one file to the next.
- **Tests.** PROD-E1-T03's golden tests plant a marker token in every secret-typed value, and in the value of every unknown key and unknown `SYNTROPY_*` variable, and check that no rendered output, for any error code, contains it. The `config/0003` example above shows the unknown key's value withheld.

*Note 2026-10-22 (PROD-E1-T03):* how the validation runs, as built.
- **Phases 1–3 report together.** Phase 3 checks each value as layering assigns it, so a bad value in a file is an error even when a later layer overrides it. A file that cannot be read, or is not UTF-8, is an error beside the other files' errors, which are still resolved. Phases 4 and 5 run once phases 1–3 pass, and a load reports every error it finds, not only the first.
- **Phases 4 and 5 are frameworks.** A rule reads resolved values by path. The task that owns a key adds its rules: PROD-E5-T02 the API bind and TLS rules, and PROD-E1-T07 the `ram_headroom` floor. PROD-E1-T05 is M0a, before this framework, so its `ResolvedConfig` emits the `debug.*` warning itself, which the startup log and each reload print. Phase 5 asks an `Environment` about the host: `config` answers paths and file permissions from the standard library, and the daemon adds direct I/O and interface names (PROD-E6-T24).
- **Directory keys.** A key whose last segment is `dir` or `dirs`, or ends in `_dir` or `_dirs`, such as `node.state_dir`, names a directory. Phase 5 requires a writable directory there, or a missing one whose nearest existing ancestor is writable, so a regular file is unusable. The check writes: it creates an empty `.syntropy-write-check-<pid>` file in the directory and removes it at once, since the standard library has no check that writes nothing. `config check` therefore touches every directory a path key names.
- **Warnings without a code.** A warning that §8's table gives no code, such as a deprecated key or a secret-holding file that others can read, prints as `warning: …`.
- **The deprecation window (open, for the maintainers).** "One minor release before 1.0" can be read two ways:
  - (a) a key deprecated in 0.3 is accepted throughout 0.3.x and unknown from 0.4.0;
  - (b) it is accepted through 0.4.x too.

  Recommended default: (a), the literal reading, as built (`Deprecated::closes_in`). The release that closes a window removes the record.

*Note 2026-10-22 (PROD-E5-T02): the API listener rules, as built.* They are `syntropy_api::rules::API_RULES`, which a binary that serves the API passes to `load_and_validate`.
- **`api.tls = "none"`** (phase 4, code 10) is checked against non-loopback binds only. A loopback bind serves plain HTTP whatever `api.tls` says (RFC-0050 §11.1), so `"none"` with loopback-only binds is accepted. On a specific LAN address it needs `api.allow_plaintext_lan = true`. On a wildcard (`0.0.0.0`, `::`), which listens on every interface, and on a public address, it is rejected even with the opt-in, because RFC-0050 §11.1 allows plain HTTP "only on LAN addresses"; RFC-0050 §11.1's note of this date lists which addresses those are.
- **The `{ cert, key }` pair.** The value's type takes both files or neither, so a table with only one of them is rejected in phase 3 (code 2), and the message names the missing one. Phase 4 has nothing left to check. In phase 5, `api_tls_files_readable` requires both files to exist and be readable files, as an error (code 11). The built-in path check does not cover them: it sees only path-typed keys, accepts a missing file whose directory is writable, and warns rather than rejects.
- **The API-key check is not a configuration rule.** The phase 4 row above lists it, but keys live in `api-keys.toml` (RFC-0050 §11.3), not in a configuration layer. The daemon checks it at start through `syntropy_api::bind::plan_binds`: a non-loopback bind with no key refuses to start, and the message names `syntropy api-key create --name … --scope inference`.

**Startup.** `syntropy serve` refuses to start on any phase 1–4 error and prints every error, not only the first. `syntropy config check [--file <path>] [--strict]` runs phases 1–5 without a daemon; `--strict` turns phase 5 warnings and empty override matches into errors for CI use.

**Error codes** (domain `Config`; never reused):

| Code | Name |
|---|---|
| 1 | `syntax` |
| 2 | `wrong_type` |
| 3 | `unknown_key` |
| 4 | `out_of_range` |
| 5 | `bad_unit` |
| 6 | `bad_selector` |
| 7 | `bad_regex` |
| 8 | `knob_without_approximate` |
| 9 | `scope_mismatch` |
| 10 | `insecure_listener` |
| 11 | `path_unusable` |
| 12 | `override_splits_group` |
| 13 | `cluster_proposal_rejected` |
| 14 | `reload_partially_deferred` (informational) |
| 15 | `manifest_signer_not_allowed` |
| 16 | `alias_shadows_model` |

### 9. Hot reload

#### 9.1 Reload classes

RFC-0001 §11.2 fixes three outcomes for a changed key: it takes effect on reload (`hot`), it needs the instance to restart (`instance`), or it needs the daemon to restart (`node`). Every key has one of these classes in the schema, and §5 lists them. This section says what each means mechanically:

| Class | Takes effect | Mechanism | Examples |
|---|---|---|---|
| `hot` | on `SIGHUP`, `syntropy config reload` or `POST /v1/syntropy/config/reload` | The daemon replaces its `Arc<ResolvedConfig>` with one atomic pointer store; readers take no locks. Requests admitted afterwards see the new values, and in-flight sequences keep the policy they were admitted with. Runner-relevant keys reach runners as `ConfigChanged` (proposed IPC message, amendment A3). A hot node key that changes what the solver sees (`resource.*` headroom and fractions, `node.labels`, `resource.gpus.*`, `net.bulk_on_metered`) also submits `RequestReplan { reason: Config }` for the affected instances. | log filters, API keys and listeners, dashboard settings, `resource.memory_yield`, queue limits, keep-alive, sampling defaults, `debug.*` |
| `instance` | when the affected instances adopt a new plan | The coordinator receives `RequestReplan { reason: ReplanReason::Config }` (RFC-0001 §11.2). The new plan's epoch takes effect as RFC-0001 §7.3 says: in place through `ApplyPlanDelta` when only budgets, prefetch, speculative or batching settings changed, otherwise by drain and replace. A user-initiated change is "as requested" in RFC-0001 §7.2, so no gain or payback test applies. | objective, pins, overrides, approximate knobs, `max_sequences`, `context_length`, speculative mode |
| `node` | when the daemon restarts (and with it, its runners) | Recorded as pending; `syntropy service restart` applies it | `node.state_dir`, `store.dirs`, `api.local_socket`, `io.engine`, `features.disable`, `resource.reserve_cores`, `tier.mmap`, `daemon.*_threads` |

None of the three fails a sequence. RFC-0001 §7.5 finishes or moves live sequences across a plan change.

Headroom increases are special. When `ram_headroom` or `vram_headroom` grows, the daemon first sends `MemoryPressure { level: Warn, reclaim }`, so runners give memory back through the ladder at once, and only then requests the re-plan. A user who asks for RAM back gets it before the new plan is ready.

#### 9.2 Triggers

- `SIGHUP` on Unix, `syntropy config reload`, or `POST /v1/syntropy/config/reload` (RFC-0001 §11.2). `syntropy config set` and dashboard edits trigger a reload after they write.
- With `config.watch = true` (default `false`), the daemon also watches every file of the config layer (`notify` crate, 500 ms debounce, directory watches to catch editors that save by rename) and reloads on change.
- `ClusterConfigChanged` in the event log, for cluster keys and the `policy.*` tables.
- Environment and CLI layers do not change while the daemon runs.

#### 9.3 Algorithm

```text
reload():
    new_layers = read and parse every file in the config layer       // phase 1
    candidate  = resolve(defaults, new_layers, env, cli, cluster_log) // §2.2
    errors     = validate(candidate, phases 2..5)
    if errors has any error:
        keep the current config; report every error; return Rejected   // all or nothing
    diff = changed_keys(current, candidate)
    propose(diff ∩ cluster_scope)                                     // §2.3; effective only once logged
    replace(current, candidate)                                       // hot keys now active
    for runner in affected(diff ∩ hot):           send ConfigChanged
    for inst in affected(diff ∩ (instance ∪ solver-visible hot)):   submit RequestReplan { reason: Config }
    pending += diff ∩ node
    return Applied { hot, replan_requested, pending }
```

The reload report is printed by `syntropy config reload`, returned by the API, logged at `info` and shown on the dashboard's settings page. A pending `node` key appears as a banner until it takes effect. `syntropy config reload --dry-run` prints the same report without changing anything.

Cluster-scope changes arriving through `ClusterConfigChanged` go through the same diff and classification on every node. A change that needs a re-plan is requested once, by the coordinator, not once per node: only the coordinator submits `RequestReplan` for cluster-scope diffs.

### 10. Secrets, writes and the schema as code

#### 10.1 Secrets

- `Secret<T>` wraps every secret value. Its `Debug` and `Display` print `[redacted]`, it serialises only through an explicit `expose_for(&'static str)` call, and its memory is zeroed on drop.
- Secrets can come from the file (`obs.otlp.headers.authorization = "Bearer …"`), from a file reference (`… = { file = "/run/secrets/otlp" }`) or from an environment variable (`{ env = "MY_VAR" }`). Containers and service managers should use the file form.
- `config show` prints `[redacted]` plus the origin. Debug bundles (RFC-0053 §6) never contain secret values.
- API keys are not configuration. Node-local key hashes live in the state directory. Cluster-wide keys live in the event log's `api_keys` set as keyholder-signed `ApiKeyCreated` records holding hashes only (RFC-0034 §8.4). Both are managed by `syntropy api-key` (RFC-0050 §11.3). The Hub token stays on the node that holds it and never crosses to another node (ADR-0016 decision 8).

#### 10.2 Writing configuration

- `syntropy config set <key> <value>` edits the user file (or the system file under `--system`, which needs the privileges to write it) with `toml_edit`, so comments, ordering and formatting survive. It validates the result before writing, writes to a temporary file and renames it into place, then triggers a reload.
- `syntropy config set --cluster <key> <value>` and the dashboard's cluster settings submit `SetClusterConfig` directly and touch no file.
- Node-scope edits made in the dashboard write the same user file through the same path. There is no hidden settings database: the file is the record.
- `syntropy config unset`, `get`, `show [--resolved] [--json]`, `edit` (opens `$EDITOR`, validates on save, refuses to keep an invalid file without `--force`), `path`, `check` and `schema` complete the set (RFC-0051 §2).

#### 10.3 The schema as code

The schema is written once, as Rust types in `syntropy-config` with attributes:

```rust
#[derive(ConfigSection)]
#[config(section = "resource", scope = Node, owner = "RFC-0020")]
pub struct ResourceSection {
    /// RAM never pinned or allocated by Syntropy.
    #[config(reload = Hot, default = "by_mode(ram_headroom)", min = "1GiB")]
    pub ram_headroom: Reserve,
    // …
}
```

*Note 2026-10-21 (open, for the maintainers): the default grammar.* The derive PROD-E1-T01 builds reads the example's `default = "by_mode(ram_headroom)"` as the literal string `"by_mode(ram_headroom)"`, so a default has to say which kind it is:
- a TOML-shaped value, such as `"16GiB"`, `64`, `false`, `["a", "b"]` or `{ k = "v" }`;
- `unset`, for an `Option` key;
- `computed("rule text")`, whose text `Origin::Default { computed }` prints (§2.2).

Options:
- (a) that grammar, as implemented, under which the example reads `default = computed("by_mode(ram_headroom)")`;
- (b) a string default that does not parse as the key's type counts as a computed rule, which turns a typo in a fixed default into a silent rule;
- (c) a separate `computed = "…"` attribute.

Recommended default: (a). Section authors should not copy the example above as written.

Cluster-scope keys add two attributes: `introduced_at = <feature level>`, which is required (a cluster-scope key without it fails to compile), and `authority = Member | Keyholder`, which defaults to `Member` and is forced to `Keyholder` under the prefixes of §2.3 step 10. For example: `#[config(scope = Cluster, introduced_at = FEATURE_LEVEL_V1, authority = Keyholder, reload = Hot, default = "confirm")]` on `cluster.lan_admission`.

*Note 2026-10-21 (open, for the maintainers): authority on instance keys.* The paragraph above gives `introduced_at` and `authority` to cluster-scope keys. But §2.3 step 10 makes the exactness and pin fields of the `policy.*` tables keyholder-only, and those are instance keys (§5.4), which the tables store in the event log. PROD-E1-T01:
- lets instance keys carry `authority`, default `member` and forced to `keyholder` for `fidelity`, `default_fidelity`, `approximate.*`, `pin` and `tensor_override`;
- lets instance keys carry an optional `introduced_at`;
- rejects both attributes on node and request keys.

Options:
- (a) as implemented;
- (b) declare each `policy.*` table as one cluster key whose type carries per-field authority;
- (c) as (a), but require `introduced_at` on instance keys too, since a newer policy field in the log raises the mixed-version question of §2.3 step 9.

Recommended default: (a), leaving (c) to RFC-0032's owner.

A derive macro generates, from that one definition:
1. the parser and phase 2–4 validators;
2. the reload classifier (a key without a `reload` attribute fails to compile);
3. the environment variable names;
4. the Markdown reference page and the per-key help text for `syntropy config get --help`;
5. a JSON Schema per `config_version`, published with each release for editor completion (the `#:schema` line in §1);
6. the TypeScript types the dashboard's settings page uses (RFC-0051 §5.8).
7. for cluster-scope keys, the `introduced_at` and `authority` columns that `decide()` and `validate()` read (RFC-0032 §6.4, RFC-0034 §8.3).

A CI check fails when a key lacks a doc comment, an owner or a default.

**`RunnerConfig`.** The daemon passes runners a resolved subset in `Init`: `ResourcePolicy`, `features.disable`, `io.*`, `tier.*`, `runner.*` timings, `logging.*` and `obs.*` sampling, and the node's `security.trusted_fabric` list (RFC-0034). Runners do not read files, environment or the event log for configuration. One resolution point means a runner can never disagree with its daemon about a setting.

## Performance model and budgets

Configuration is off the token path. No hot thread reads configuration: runners receive `RunnerConfig` at `Init` and `ConfigChanged` on the `net` domain, which hands values to hot threads between steps. So the budgets below protect the request path and daemon start, not decode.

| Item | Budget | Basis |
|---|---|---|
| Resolve and validate a 300-line configuration at start | ≤ 20 ms [target] | chosen budget: under 0.2% of the SC-22 15 s probe |
| `resolve_request` plus a cached `InstancePolicy` lookup at ingress | ≤ 10 µs p99 [target] | chosen budget: 1% of the 1 ms pre-`SequenceOpen` ingress budget in RFC-0050 (performance section) |
| `InstancePolicy` construction and digest | once per (cluster config version, model), cached | recomputed only on `ClusterConfigChanged` |
| Reload visible to readers | one atomic pointer store; no reader blocks | an atomically replaced `Arc` |
| Steps stalled by a reload | zero | `hot` keys reach hot threads only between steps; `instance` changes arrive as a new plan epoch (`ApplyPlanDelta` or drain and replace) |

**The defaults are performance decisions.** `resource.ram_headroom` sets the RAM tier size, and on MoE-on-desktop plans the RAM tier sets the hit rate. For the S1 desktop with 128 GB of DDR5 and DeepSeek-V3 at Q4_K_M (394 GB routed pool, vision §1.3) [estimate, using the cost model's uniform prior `h = cache fraction`, §3.3]:

| Mode | Headroom | RAM left for Syntropy | Prior hit rate before locality |
|---|---|---|---|
| `personal` | max(6 GiB, 12.8 GB) = 12.8 GB | ≈115 GB, less the hot set's host copies and staging | ≈115 / 394 ≈ 0.29 |
| `dedicated` | max(2 GiB, 3.8 GB) = 3.8 GB | ≈124 GB | ≈124 / 394 ≈ 0.31 |

At h ≈ 0.29–0.31 the cost model puts this machine at ≈0.7 tok/s with no routing locality (vision S1), so the headroom choice moves the no-locality case by a few percent, while learned routing locality (RFC-0023) moves it by up to ≈7.6×, to the 5.3 tok/s DRAM cap. That is why `personal` keeps a generous headroom by default: it costs little speed and protects the user's desktop.

`resource.reserve_cores` takes cores from the CPU pool that executes missed experts in place. CPU execute-in-place is DRAM-bound on desktop parts (cost model §3.1), so the effect depends on how many cores saturate the memory bus. RFC-0010's probe therefore measures CPU `W_eff` with the core set the runner will really have after the reserve, and the solver sees the reserved figure, not the full-machine one.

## Alternatives considered

| Alternative | Why not |
|---|---|
| YAML, JSON5, KDL or HCL as the file format | ADR-0500. YAML's implicit typing (`no` → false, `1.10` → 1.1) and indentation errors hurt hand-edited files; JSON has no comments; KDL and HCL lack mature, comment-preserving Rust editors and are unfamiliar to Rust users |
| Every key in the cluster log, no node files | a node could not be configured while offline or before joining; per-node facts (RAM headroom, store paths) would need node-qualified keys in a shared namespace. Rejected (ADR-0501) |
| Every key node-local, no cluster scope | ingress nodes with different model policies create duplicate instances and can route an exact request to an approximate instance. Rejected (ADR-0501) |
| Instance keys as node keys, layered per node | each ingress node would build a different `InstancePolicy` for the same model, so automatic placement would create duplicate instances; and a node-level `--set objective=latency` would silently undo every per-model objective. Rejected: instance keys resolve from the cluster's policy tables (§2.2 Rule 3) |
| A settings database edited by the dashboard | hidden state that a file diff cannot show, and two sources of truth. Rejected; the dashboard writes the file |
| Per-request instance-wide knobs (start an approximate instance on demand) | one request could double a model's memory footprint. Rejected; instance-wide knobs belong to the model policy |
| Unknown keys as warnings | typos would silently do nothing, and RFC-0001 §11.2 requires rejection. Renamed keys get a deprecation window instead, and rollbacks restore the pre-update file (§8) |
| Restart runners for any change | reloading a 400 GB model to change a log level. Rejected; six reload classes instead |
| Separate config files per subsystem | users would have to know the crate structure. Rejected; one file, sections by subsystem |

## Testing and validation

| Claim | Test |
|---|---|
| Resolution is deterministic and follows Rules 1–4 | property tests (`proptest`) over random layer stacks: the same inputs give the same `ResolvedConfig` and the same `InstancePolicy::digest`; setting a key in a higher layer changes only that key; a model-specific value beats a global one at any layer; drop-in order is lexical whatever the directory listing order |
| Every printed value pastes back | round trip: `config show --json` → TOML → resolve gives the same effective config |
| Debug APIs stay off and local | integration tests: in every build profile each `debug.*` endpoint answers 404 until its key is set; with the key set, requests from a non-loopback peer still get 404 whatever API key they carry; `BenchControls` without `debug.bench_api` fails with `debug_api_disabled`; the warning, doctor CFG-003 and the dashboard banner appear while any key is on |
| The section registry holds | the schema derive rejects duplicate prefixes and scope mismatches at build time; a CI test loads every domain crate's config section and checks it against §5.1 |
| Writes preserve the user's file | golden tests: `config set` on files with comments, odd spacing and inline tables changes only the target line |
| The schema is complete | CI check: every key has a doc comment, owner, default, scope and reload class; the generated JSON Schema validates every example in the docs |
| Validation messages are useful | golden error snapshots per error code, including the edit-distance suggestion |
| Parsers do not panic or hang | `cargo-fuzz` on the TOML layer, unit parsers, selectors and regex compilation (regex size limits enforced) |
| Reload is all or nothing, and each class does what it says | a daemon under `syntropy-testkit` with fake runners: every key class triggers exactly the expected action; an invalid edit leaves the running config untouched; a hot-thread step-time histogram shows no stall across 10,000 reloads |
| Cluster scope stays consistent | multi-node simulation (RFC-0060): proposals from several nodes, conflicting proposals, coordinator change mid-proposal, network split and heal with re-proposal of discarded entries; every node ends with the same effective cluster config |
| Mixed versions and authority | RFC-0060 F-20 and the `feature_level_gate` checker: a key above the cluster's feature level is rejected with `FeatureLevelTooLow`, and older nodes carry newer keys opaquely across a split and heal; a stale re-submission gets `ConfigConflict`, and the operator sees it; a `keyholder`-authority key without a keyholder signature is refused by every node's `validate()` |
| Exact cannot become approximate by accident | property test over random (cluster config, instance policy, request) triples: the effective policy is `Approximate` only when all three opt in (RFC-0001 §11.3), its knob set is always a subset of both the request's and the instance's, and `ApproxFlags` on every `TokenEntry` is a subset of the effective set; an exact request on a plan with an instance-wide knob always fails with `fidelity_conflict`; a knob without `fidelity = "approximate"` never validates; with `default_fidelity` drawn at random as well, a request that states no fidelity is approximate only when the instance's `default_fidelity` is `approximate`, its `exactness` allows it and `allow_approximate` is true, and an explicit `fidelity: exact` is never approximate |
| Pins and overrides compile correctly | census fixtures for each v1 family (the tiny synthetic models of RFC-0060): regex selection, group-split detection, fused expert tensors, contradictory pins reported before the MILP runs |
| Computed defaults are sane | snapshot of computed defaults on each reference rig (RFC-0061) checked in review when they change |

## Rollout / milestones

| Milestone | Lands |
|---|---|
| M0 | `syntropy-config` crate with the schema derive; TOML file, environment and CLI layers for the node keys M0 uses (`api.bind`, `store.dirs`, `hub.*`, `logging.*`, `node.state_dir`); validation phases 1–4; `config show`, `check`, `path`; the three `debug.*` keys (§7.1), which the M0 parity harness (RFC-0060) and benchmark driver (RFC-0061) need |
| M1 | `ResourcePolicy` with both modes, `memory_yield` and `power`; `tier.*`, `io.*`, `runner.*`; per-model tables on one node (objective, context, keep-alive, pins, regex overrides, speculative); `RequestOverrides` with the exactness gate; the `hot`, `instance` and `node` reload classes; generated reference docs and JSON Schema; `config set/unset/edit` |
| M2 | cluster scope through the event log; proposals and the proposal ledger over RFC-0032's command journal; per-key `introduced_at` and `authority`, with proposals from `STAGE_HOST` members only, the `keyholder` list of §2.3 step 10 and the alias-shadowing rule; `NetworkScope`; `replan.*` keys; instance changes adopted in place through `ApplyPlanDelta`; `InstancePolicy` digests used by `EnsureInstance` |
| M3 | every approximate knob with its RFC-0042 parameters; `deterministic`; dashboard settings editing; split-and-heal re-proposal in the chaos suite |
| M4 | the v1 schema frozen and published; `syntropy config migrate` scaffolding for a future `config_version = 2` |

## Open questions

1. **LAN admission (settled).** The key is `cluster.lan_admission` (RFC-0030). Its default `confirm` is a 6-digit SAS pairing, confirmed on any keyholder (ADR-0302). `syntropy cluster open` gives a timed pairing window, `open` mode exists for isolated networks, and certified nodes rejoin automatically. SC-25 measures rejoin, and a new node is timed from approval. This RFC only registers the key (§5.1).
2. **Default API port 7433 (answered: OQ-PRD-13).** Decided (PROD-E1-T06): the default stays TCP 7433, and the schema default of `api.bind` is `["127.0.0.1:7433", "[::1]:7433"]` (§5.2). The sources checked, on 2026-10-21:
   - **IANA registry.** The service-name and port-number registry (`https://www.iana.org/assignments/service-names-port-numbers/service-names-port-numbers.csv`, last modified 2026-10-16) lists 7432–7436 as one unassigned row for every transport, as it did on 2026-10-20. The nearest assignments are 7431 and 7437.
   - **Software defaults.** No public list records every application's default port, so the survey reads two lists that collect reported uses. Wikipedia's "List of TCP and UDP port numbers" (revision 1381161196, 2026-10-20), which includes unofficial uses, has no entry between 7402 and 7471. The Nmap scanner's `nmap-services` table (commit `542eda7`, 2026-08-07), built from IANA data and its maintainers' own research, has no entry for 7433 on TCP or UDP. Neither list records any use of 7433.
   - **Ephemeral ranges.** 7433 sits below Linux's default range, 32768–60999 (`/proc/sys/net/ipv4/ip_local_port_range`), Windows' default dynamic range, 49152–65535 (`netsh int ipv4 show dynamicport tcp`), and macOS's, 49152–65535 (RFC-0030 §3). With default settings, an outbound connection never takes it as its local port.

   The check cannot rule out an unlisted application holding 7433, and on Windows, Hyper-V and WSL can reserve port ranges. Doctor check INS-004 (RFC-0053 §5.2, built by PROD-E9-T02) is the answer to a bind failure: it reports the bind error, on Windows whether the port sits in an excluded range (`netsh int ipv4 show excludedportrange protocol=tcp`), and the `api.bind` setting that moves the listener.
3. **Headroom defaults per OS.** macOS memory compression and Windows standby lists make "available memory" mean different things. Spike: run the RFC-0061 "desktop in use" scenario (browser, IDE, video call alongside a loaded model) on each Tier 1 OS and tune `ram_headroom` per OS if responsiveness differs.
4. **`keep_alive` default (answered: OQ-PRD-14).** `keep_alive = auto`, which is `max(1 h, 60 × t_reload)` [target], scaled by each model's measured reload cost (§5.4). PROD-E1-T14 tunes the constant from M1 measurements.
5. **Conflicting proposals.** Last accepted proposal wins today. An alternative is for the coordinator to reject a proposal that contradicts a different node's proposal made within the last N minutes. Decide after M2 user testing.
6. **Selectors by family or label (dropped until users ask, OQ-PRD-16).** Matching `[policy.*]` on architecture family or Hub tags (`family = "deepseek-v3"`) as well as names is post-1.0 and comes back only on user demand.

## Proposed amendments to RFC-0001

- **A1. §11.1 scope table.** State that the `policy.*` tables are cluster-scoped storage for default instance policies. Automatic placement builds `InstancePolicy` from them, so ingress nodes agree. An explicit `EnsureInstance` overrides them for its instance (§2.1). Today §11.1 says instance keys come only from "the client or operator that loads the model", which leaves automatic placement on first request (RFC-0001 §6.1 step 3) without a source.
- **A2. §5.8 `ApproxFlags`.** `TokenEntry::approx: ApproxFlags` is used but never defined, while §11.3 defines `ApproxKnobs`. Define `pub type ApproxFlags = ApproxKnobs;` so the bit positions are one table, which becomes part of the `TokenFrame` wire format (RFC-0031).
- **A3. §3.5 IPC.** Add `D → R ConfigChanged { patch: RunnerConfigPatch }` for `hot` keys that runners use (log filters, trace sampling, spin and power policy). Without it, §11.2's hot keys cannot reach a running runner, because runners receive configuration only at spawn.
- **A4. §5.10.** Fix `ConfigPatch` as `{ set: Vec<(KeyPath, ConfigValue)>, unset: Vec<KeyPath>, origin: ProposalOrigin }`. Let `ReplanReason::Config` carry `keys: Vec<KeyPath>`, so the dashboard and logs can say which change caused a re-plan (RFC-0033 owns `ReplanReason`).
- **A5. §5.4.** Add `PlacementPlan::policy: Blake3`, the `InstancePolicy` digest the plan was solved under, so routing and the dashboard can tell which policy produced a plan. RFC-0011 may make this addition under its ownership; it is listed here because this RFC depends on it.
- **A6. §11.3 exactness rule.** Allow an instance policy to make approximate the default for requests that do not state a fidelity: `InstancePolicy::default_fidelity: Fidelity`, default `Exact`. It is still subject to `allow_approximate`, and an explicit `exact` in a request still wins. Today a request must ask for approximate itself, so an operator who loads a model that is only usable approximately (for example a frontier MoE that needs `TOPK_REDUCE` on one PC) cannot serve stock clients that send no extension fields. D3 permits opt-in "per model or per request"; this amendment makes the per-model opt-in usable on its own. The label rules are unchanged. ADR-0019 decision 7 is revised to match.
- **A7. §11.2 reload.** Clarify that an `instance`-class change takes effect through a new plan epoch, adopted in place with `ApplyPlanDelta` when §7.3 allows. It need not be a full instance restart. Also clarify that a `hot` node key the solver reads (headroom, labels, GPU filters) requests a re-plan as part of taking effect.
- **A8. §12, zone Z3.** Add the rule of §7.1: debug endpoints (`debug.parity_api`, `debug.bench_api`, `debug.telemetry_api`) answer only on loopback listeners and the local socket, are off by default in every build, and cannot be enabled for remote callers by any key.
- **A9. §11.3 `InstancePolicy::resources`.** Change its type from `ResourcePolicy` to `ResourceOverlay` (all fields optional, only explicit values), and add `domain: BTreeMap<KeyPath, ConfigValue>` for domain instance keys. Without this, `InstancePolicy::digest` depends on each ingress node's own hardware defaults, and automatic placement creates duplicate instances.
- **A10. §11.1 scope table, the cluster row.** "Who sets it" reads: the operator of a member holding `STAGE_HOST`. Keys under `security.*`, `invite.*`, `coordinator.*`, `swim.*` and `cluster.lan_admission`, plus `cluster_aliases`, `network_scope`, `allow_approximate`, `update.*` and the exactness and pin fields of `policy.*`, need a keyholder's signature (§2.3 step 10).
- **A11. §11.3.** Approximate mode is per-model knobs only, with no named preset. The only mention of approximate options for an exact model is one informational line in `syntropy plan` and the dashboard's preview (RFC-0051 §2.5).
