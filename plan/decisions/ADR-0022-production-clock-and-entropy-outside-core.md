# ADR-0022: Production clock and entropy sources live in `io`, not `core`

- Status: Proposed
- Date: 2026-10-21
- Deciders: maintainers (drafted by an implementing agent for issue #1484)
- Related: ADR-0001 (item 4), ADR-0012, ADR-0021 (decisions 5, 9), ADR-0600 (decisions 1–2); RFC-0001 §4.1, §4.2, §5.14, §18; RFC-0030 §5.8, open question 6; RFC-0034 §3; RFC-0060 §2.1, §2.2, §2.4; RFC-0062 §12.1; QUAL-E3-T01

## Context

RFC-0001 §5.14 and RFC-0060 §2.1 place six determinism types in `syntropy-core`: `MonoTime`, `Clock`, `Entropy`, `ProtocolCore`, `Outbox` and the two production sources, `SystemClock` and `OsEntropy`. RFC-0060 §2.4 adds `DetHashMap`. QUAL-E3-T01 builds all of them in `core`, and one of its acceptance criteria is "`core` still passes its dependency allow-list (QUAL-E1-T03)".

The two production sources can't be written in `core` under RFC-0001 §4.2:

- **Rule 1** limits `core`'s external dependencies to `serde`, `thiserror`, `smallvec` and `bitflags`.
- **Rule 8** forbids `unsafe` in `core`. Every crate outside `kernels-*`, `io`, `tier`, `net` and `ipc` sets `#![forbid(unsafe_code)]`.
- **`OsEntropy`** is the OS CSPRNG: "getrandom" in RFC-0060 §2.1, and "OS CSPRNG (`getrandom`)" in RFC-0034 §3's randomness row. Rust's standard library has no public CSPRNG API. Reaching the OS source needs the `getrandom` crate or a direct FFI call.
- **`SystemClock::now`** must read a clock that keeps running through suspend: `CLOCK_BOOTTIME` on Linux and Android, `mach_continuous_time` on macOS and iOS, `QueryInterruptTime` on Windows (RFC-0060 §2.1; RFC-0030 §5.8 and open question 6). `std::time::Instant` reads a clock that stops during suspend on Linux (`CLOCK_MONOTONIC`), which RFC-0060 §2.1 rules out. Reading the suspend-aware clocks needs an `extern` declaration and an `unsafe` call, or a binding crate outside rule 1's list.
- **The wall clock** (`Clock::unix_ms`) is not a problem. `std::time::SystemTime` is safe and in the standard library.

So QUAL-E3-T01 can't meet its acceptance criteria as written. It is M0a, and `roadmap.md` lists 11 direct and 892 transitive dependents behind it (M0a–M4). This finding comes from reading the plan (unconfirmed static analysis); no code was written to test it.

Two further facts shape the options:

- **The traits are the part most crates need.** Rule T1 (RFC-0060 §2.1) makes decision-making code read time and randomness only through a `Clock` and an `Entropy` passed in. The concrete sources are built once, by the processes that wire the system together: the daemon (cluster shells, node key generation in DIST-E2-T01, the supervisor) and the runner (the scheduler and the lifecycle machine). The simulator never builds them; it uses `SimClock` and `SimEntropy` from `testkit` (QUAL-E3-T02).
- **Suspend detection needs a second clock.** RFC-0030 §5.8 has the daemon read both the suspend-aware clock and the clock that stops during suspend (`CLOCK_MONOTONIC`, `mach_absolute_time`, `QueryUnbiasedInterruptTime`), and compare them. That second reading has the same FFI needs on Windows, so it belongs beside `SystemClock`.

## Options considered

1. **Keep the traits in `core` and move the two production sources to a layer-1 crate that may already use `unsafe`.** `MonoTime`, `Clock`, `Entropy`, `ProtocolCore`, `Outbox` and `DetHashMap` stay in `core`, which stays plain data and pure functions. `SystemClock` and `OsEntropy` move. Three homes were considered:
   - **1a. `io`.** Rule 8 already allows `unsafe` there, and ADR-0001 item 4 allows it in I/O and FFI crates. `io` is the crate that already makes OS calls (io_uring, IoRing, IOCP, `preadv`, `F_NOCACHE`). The daemon, the runner and `probe` already depend on it (RFC-0001 §4.1), so no new edge is needed. Rule 3 allows compute crates to depend on `io`, and neither source needs tokio. The cost: a crate that wants only a clock pulls in the I/O engines.
   - **1b. `ipc`.** `unsafe` is allowed, and the daemon and the runner both depend on it. But its job is the daemon↔runner protocol and shared memory; clocks and randomness fit it less well than `io`.
   - **1c. A new layer-1 crate** (for example `syntropy-os`) holding only the OS clocks and the CSPRNG. It is the smallest `unsafe` surface, and a crate that needs a clock pulls in nothing else. It adds a row to RFC-0001 §4.1, a crate to rule 8's list, a crate to RFC-0062 §12.1's allow-list, and a workspace stub.

   Option 1 amends RFC-0001 §5.14 and RFC-0060 §2.1, and splits QUAL-E3-T01 into a `core` part and a sources part.
2. **Widen `core`'s allow-list** in rule 1 to add `getrandom` and a crate that wraps the suspend-aware clocks safely. This keeps all the types together. This drafting found no crate that wraps `mach_continuous_time` and `QueryInterruptTime` without `unsafe` and stays as small as rule 1's existing four. It makes every crate in the workspace depend, through `core`, on OS bindings, and adds new direct dependencies to the lowest layer.
3. **Allow `unsafe` in one `core` module** (`core::os`) for the clock calls, and add `getrandom` to rule 1. This amends rule 8 and rule 1, and makes `core` the first layer-0 crate with `unsafe`. Miri and the sanitizer jobs (QUAL-E3-T10) would then have to cover `core`.
4. **Write both sources in `core` with the standard library only.** Use `Instant` for the clock, and derive randomness from `std::collections::hash_map::RandomState`. Rejected: `Instant` stops during suspend on Linux, which breaks RFC-0030 §5.8's lease and suspicion timing, and `RandomState` is not a CSPRNG, which breaks rule T2 (key material from the OS CSPRNG). Listed only because it is the obvious shortcut.
5. **Build the sources in each binary crate** (`daemon` and `runner`). Rejected: both crates forbid `unsafe` under rule 8, the code would exist twice, and rule 6 (one crate per external system) would be broken by `getrandom` in two crates.

## Decision

Proposed: **option 1a.**

1. **`core` keeps the traits and the plain types:** `MonoTime`, `Clock`, `Entropy`, `TickSource`, `ProtocolCore`, `Outbox` and `DetHashMap`. Its external dependencies stay the four of rule 1, and it stays `#![forbid(unsafe_code)]`.
2. **`io` holds the production sources**, in one module (`syntropy_io::os` is the proposed path):
   - `SystemClock`, implementing `Clock`. `now` reads the suspend-aware clock named above for each OS. `unix_ms` reads `std::time::SystemTime`.
   - the clock that stops during suspend, as a second reading on the same type or a sibling type, so RFC-0030 §5.8's shell can compare the two without its own FFI.
   - `OsEntropy`, implementing `Entropy` over the OS CSPRNG through `getrandom`.

   Each `unsafe` block carries its written invariant (ADR-0001 item 4).
3. **Rule 6 names `io` as the one crate that depends directly on `getrandom`.** Crates that reach it transitively, such as through the TLS stack in `net`, are not affected.
4. **`getrandom` is a new direct dependency.** ADR-0021 decision 5 needs a maintainer's approval for it, and accepting this ADR gives that approval. Any binding crate the clock calls need that `io` doesn't already use (for example `libc` or `windows-sys`) also needs that approval, and the sources task lists it.

**Why 1a over the others.** It changes no rule in RFC-0001 §4.2 except to add `getrandom` to rule 6's allow-list. Rule 1 and rule 8 stay as they are, and `core` stays plain data and pure functions, as its row in §4.1 says. It puts the FFI where rule 8 and ADR-0001 item 4 already allow `unsafe`. Every crate known to build the concrete sources already depends on `io`, so it adds no dependency edge. Option 1c is cleaner on cohesion, but it widens rule 8's list and adds a crate to fix a problem that `io` already solves. It becomes the better choice if the trigger under "What would make us revisit" fires. Options 2 and 3 move OS bindings or `unsafe` into the layer every crate depends on.

### Edits on acceptance

None of these are made while this ADR is Proposed (ADR-0021 decision 9). On acceptance, the change that accepts it makes them:

1. **RFC-0001 §5.14.** Remove `SystemClock` and `OsEntropy` from the `core` code block. Add a sentence below it: the production sources are in `io` (ADR-0022), and the simulator's are in `testkit`.
2. **RFC-0001 §4.1.** Add "the production `Clock` and `Entropy` sources (`SystemClock`, `OsEntropy`, §5.14)" to `io`'s Responsibility cell.
3. **RFC-0001 §4.2 rule 6.** Add "`getrandom` only in `io`".
4. **RFC-0001 header and §18.** Extend "Builds on" to ADR-0022, and add a dated §18 entry for items 1–3 with ADR-0022 as the reason. §18 records changes when they land, so it gets no entry while this ADR is Proposed.
5. **RFC-0060 §2.1.** In the code block, mark `SystemClock` and `OsEntropy` as `// syntropy-io (ADR-0022)` instead of under the `syntropy-core` comment, and add a dated revision note. Rule T2 is unchanged: key material still comes only from `OsEntropy`.
6. **RFC-0062 §12.1, rule 6 row.** Add `getrandom` → `io` to the allow-list.
7. **`plan/README.md`.** Change "ADR-0001 to ADR-0021" to "ADR-0001 to ADR-0022", and add a row for this ADR to the foundational table.
8. **QUAL-E3-T01 is split** in `epics/QUAL.md`, as below. The plan tools are then re-run.

### The proposed task split

The task ID QUAL-E3-T24 was the next free number in the QUAL-E3 epic when this was drafted. The accepting change uses whatever number is free then.

| ID | Task | Milestone | Depends on | Acceptance criteria | Implements |
|---|---|---|---|---|---|
| QUAL-E3-T01 | `syntropy-core` determinism types: `MonoTime`, `Clock`, `Entropy`, `ProtocolCore`, `Outbox`, `DetHashMap` (fixed-key hasher) | M0a | QUAL-E1-T01 | unit tests for each; `core` still passes its dependency allow-list (QUAL-E1-T03) and stays `#![forbid(unsafe_code)]` | RFC-0060 §2.1, §2.2, §2.4; ADR-0600 (decisions 1–2); ADR-0022 (decision 1); RFC-0001 amendment A1 |
| QUAL-E3-T24 | Production clock and entropy sources in `io`: `SystemClock` over the suspend-aware OS clock (`CLOCK_BOOTTIME`, `mach_continuous_time`, `QueryInterruptTime`) with `unix_ms` from `SystemTime`; the stopped-clock reading RFC-0030 §5.8 compares against; `OsEntropy` over `getrandom` | M0b | QUAL-E3-T01 | `SystemClock::now` is monotonic across 10⁶ calls; on each Tier 1 OS a test reads the named suspend-aware clock directly and shows `now` agrees with it; `OsEntropy::fill` fills empty, 1-byte and 1 MiB buffers, and two 32-byte fills differ; every `unsafe` block has a written invariant; QUAL-E1-T03 shows `getrandom` as a direct dependency of `io` only | RFC-0060 §2.1 (rule T2); RFC-0030 §5.8; RFC-0034 §3; ADR-0022 (decisions 2–4) |

The sources task is M0b because no M0a task builds a concrete clock or entropy source. The M0a dependents of QUAL-E3-T01 (QUAL-E3-T02, QUAL-E4-T01, QUAL-E4-T14, QUAL-E5-T19) need only the traits and types. Two dependents need the concrete sources and gain QUAL-E3-T24 in their "Depends on" cell:

- **DIST-E2-T01** (node key generation, M0b), which needs `OsEntropy`;
- **QUAL-E3-T17** (`TickSource` production counter, M0b), whose `now` returns `MonoTime` on the `Clock` timeline.

The other dependents of QUAL-E3-T01 keep their edge to the `core` part only.

## Consequences

- **Positive.**
  - QUAL-E3-T01 can land at once as the `core` part, which unblocks its M0a dependents.
  - `core` keeps no OS bindings and no `unsafe`, so every crate above it builds without them.
  - The clock and CSPRNG FFI sits in a crate already on rule 8's list, which QUAL-E3-T10's ASan and TSan builds of the `unsafe` crates cover. Its Miri job names `tier`, `ipc` and `net` only, so adding `io`'s `os` module to it is a choice for that task.
- **Negative.**
  - The determinism types are split across two crates. A reader of RFC-0001 §5.14 has to follow a pointer to find the production sources.
  - A crate that builds a concrete clock and doesn't already depend on `io` takes `io`'s engines with it.
  - `getrandom` and any new clock binding are new direct dependencies, each needing a maintainer's approval.
- **Adjacent, not decided here.** QUAL-E3-T17's production `TickSource` reads the invariant TSC or CNTVCT_EL0, which may need `unsafe` or inline assembly, and `exec` and `obs` forbid `unsafe` under rule 8. The plan does not say which crate holds that implementation. This drafting did not check whether the safe standard-library intrinsics cover both counters. The `os` module in `io` is a candidate home, and the question is left to QUAL-E3-T17.

## What would make us revisit

- A crate that doesn't depend on `io` needs to build a concrete clock or entropy source, and taking `io` would break rule 3 or pull the I/O engines into a crate that has no other use for them. Then move the module to its own layer-1 crate (option 1c).
- The `os` module grows past the clocks and the CSPRNG, for example with sleep notices (RFC-0030 §5.9) or the `TickSource` counter. A module with that scope no longer belongs in the I/O crate, and option 1c applies.
- A maintained crate appears that reads all three suspend-aware clocks without `unsafe` in its user, with a dependency footprint as small as rule 1's existing four. Option 2 is then cheap enough to compare again.
