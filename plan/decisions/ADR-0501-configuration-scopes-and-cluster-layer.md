# ADR-0501: Configuration scopes in practice: cluster-stored model policy tables, and files that propose cluster values
- Status: Accepted (2026-10-20); revised 2026-10-20 (see Revisions)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the product surface and operations area)
- Related: ADR-0000 (D2, D3, D4), ADR-0008, ADR-0009, ADR-0019; RFC-0001 (§5.10, §6.1, §7.2, §11), RFC-0054, RFC-0011, RFC-0030, RFC-0050

## Context
RFC-0001 §11.1 fixes four scopes:
- **node**: file, environment and CLI;
- **cluster**: the event log only;
- **instance**: the `InstancePolicy` that `EnsureInstance` carries;
- **request**: narrows the instance.

Two questions remain open, and both matter for automatic placement and for split-heal.

1. **Where do instance policies come from when nobody loaded the model explicitly?** Every ingress node routes from its own replica of cluster state (RFC-0001 §6.1). When a requested model is not loaded, ingress itself calls `EnsureInstance { model, policy }`. If two nodes built different policies for the same model, they would create duplicate instances, or one would route an exact request where only an approximate plan exists (ADR-0019).
2. **How do people set cluster keys?** RFC-0001 says cluster keys come only from the event log and a node file cannot override them. Users still expect a file on disk to mean something. ADR-0009 also requires cluster state to be reconstructible from node-reported facts after a split and heal: no event may be the only record of anything a node needs.

## Options considered
1. **Instance policies only from explicit loads.** Simple, but automatic placement on first request then has no policy source except schema defaults, so any per-model setting is lost the moment a model is unloaded and requested again. Rejected.
2. **Per-model tables in each node's file (node scope).** Works offline, but nodes disagree, which leads to duplicate instances and inconsistent routing. Rejected.
3. **Per-model tables in the cluster log, read by every ingress to build the same `InstancePolicy`; explicit loads override them for their instance.** Chosen.
4. **Cluster keys settable only through `config set --cluster` and the dashboard.** Unambiguous, but a file that cannot hold cluster settings breaks declarative setups and the first-node experience. Kept as the fallback in "What would make us revisit".
5. **Files propose cluster values, through a proposal ledger that re-proposes after a split heal.** Chosen.

## Decision
1. **Model policy tables.** `[policy.default]` and `[policy."<model selector>"]` hold default values for instance-scope keys. They are cluster-scoped storage, so every ingress node sees the same tables.
   - Ingress builds `InstancePolicy` from them purely (same inputs, same policy, same digest), and `EnsureInstance` is idempotent by (model, digest).
   - Precedence, most specific first: explicit `EnsureInstance` values, then the tables by model ID, exact name, glob, then `policy.default`, then schema defaults.
   - A node-level flag never changes an instance key.
2. **Proposals.** A cluster key or policy table found in a node's file, environment or CLI is submitted once as a `SetClusterConfig` proposal, and again only when its value changes on that node.
   - A proposal ledger in the node's state directory records what it proposed and at which `(term, index)` the proposal was accepted.
   - After a split heal discards log entries, nodes re-propose the ledger entries in the discarded range. That satisfies ADR-0009 without making the log the only record.
   - When proposals conflict, the last accepted wins, and the other node's `config show --resolved` reports the conflict.
3. **Requests** follow RFC-0001 §11.3: approximate only when request, instance policy and the cluster key `allow_approximate` all opt in, with the knob sets intersected.
4. **Who may propose** (added 2026-10-20; see Revisions). Only members holding `STAGE_HOST` submit `SetClusterConfig`; an ingress-only member, such as a guest's invited laptop, proposes nothing, and its file's cluster keys are reported as ignored. Keys that change behaviour other users rely on need a keyholder signature: `cluster_aliases`, `network_scope`, `allow_approximate`, `update.*`, and the exactness, approximate-knob and pin fields of `policy.*`. An alias may not shadow a registered canonical name or short form, and every alias or policy change raises a dashboard notice naming the proposing node. RFC-0034 owns the authority rule; RFC-0054 §2.3 lists the keys.

## Consequences
- **Positive.**
  - Automatic placement, routing and instance creation agree across nodes by construction.
  - Per-model settings survive unloads.
  - A user's file means something on every node, including a node configured while offline.
- **Negative.**
  - Changing an alias, the network scope or a model's exactness policy waits for a keyholder's approval when proposed from any other node.
  - Two nodes whose files disagree on a cluster value produce a "your file says X, the cluster has Y" state that users must understand. The CLI, the dashboard and `config show --resolved` must explain it well.
  - The proposal ledger is one more piece of state to keep correct across heals.
  - A cluster value set from the dashboard exists only in the log and snapshots, not in any file. `syntropy config export --cluster` writes it to a file for backup.
- **Work created.**
  - RFC-0054: schema, proposal ledger, resolution.
  - RFC-0030: `ClusterConfigChanged` in snapshots and catch-up.
  - RFC-0060: split-heal tests for config.
  - RFC-0051: origin display and conflict messages.
  - RFC-0001: amendment A1 of RFC-0054, which names the policy tables as the source of automatic placement's `InstancePolicy`.

## What would make us revisit
- Users find the proposal model confusing in M2 testing, for example edits that "don't stick". Then accept cluster values only through `config set --cluster` and the dashboard, and reject them in files.
- Conflicting proposals cause flapping in the field. Add coordinator-side rejection of a proposal that contradicts a recent proposal from a different node.
- An instance key turns out to need per-node values, such as a model's `keep_alive` on a node short of memory. Add a node override key for it rather than changing its scope.

## Revisions
- Revised 2026-10-20: decision 4 added: only `STAGE_HOST` members propose cluster configuration, and `cluster_aliases`, `network_scope`, `allow_approximate`, `update.*` and the exactness and pin fields of `policy.*` need a keyholder. Without it the least-trusted member, such as an ingress-only guest, could redirect model names, relax exactness or change the network scope for everyone. Aliases may not shadow a registered model name, and alias and policy changes raise a notice naming the proposer.
