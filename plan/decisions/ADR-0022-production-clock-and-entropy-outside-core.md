# ADR-0022: Production clock and entropy sources live in `io`, not `core`

- Status: Accepted (2026-10-22): option 1a, `getrandom` approved as a direct dependency of `io` only
- Date: 2026-10-21
- Deciders: maintainers (drafted by an implementing agent for issue #1484)
- Related: ADR-0001 (item 4), ADR-0012, ADR-0021 (decisions 5, 9), ADR-0107, ADR-0600 (decisions 1–2); RFC-0001 §4.1, §4.2, §5.14, §18; RFC-0030 §5.8, open question 6; RFC-0034 §3; RFC-0060 §2.1, §2.2, §2.4; RFC-0062 §12.1; QUAL-E3-T01, QUAL-E3-T24; OQ-PRC-20; #1511; issue #1526

## Context

RFC-0001 §5.14 and RFC-0060 §2.1 place the determinism types in `syntropy-core`: `MonoTime`, `Clock`, `Entropy`, `ProtocolCore`, `Outbox`, and the two production sources `SystemClock` and `OsEntropy`. RFC-0060 §2.4 adds `DetHashMap`. QUAL-E3-T01 builds all of them in `core`, and one of its acceptance criteria is "`core` still passes its dependency allow-list (QUAL-E1-T03)".

The two production sources can't be written in `core` under RFC-0001 §4.2:

- **Rule 1** limits `core`'s external dependencies to `serde`, `thiserror`, `smallvec` and `bitflags`.
- **Rule 8** forbids `unsafe` in `core`. Every crate outside `kernels-*`, `io`, `tier`, `net` and `ipc` sets `#![forbid(unsafe_code)]`.
- **`OsEntropy`** is the OS CSPRNG: "getrandom" in RFC-0060 §2.1, and "OS CSPRNG (`getrandom`)" in RFC-0034 §3's randomness row. Rust's standard library has no stable CSPRNG API on the pinned toolchain. The independent review of this draft found that `std::random::fill` does not exist there (`error[E0425]`) and that `std::random::random` is unstable (`error[E0658]`). Reaching the OS source needs the `getrandom` crate or a direct FFI call.
- **`SystemClock::now`** must read a clock that keeps running through suspend: `CLOCK_BOOTTIME` on Linux and Android, `mach_continuous_time` on macOS and iOS, `QueryInterruptTime` on Windows (RFC-0060 §2.1; RFC-0030 §5.8 and open question 6). `std::time::Instant` reads a clock that stops during suspend on Linux (`CLOCK_MONOTONIC`), which RFC-0060 §2.1 rules out. Reading the suspend-aware clocks needs an `extern` declaration and an `unsafe` call, or a binding crate outside rule 1's list.
- **The wall clock** (`Clock::unix_ms`) is not a problem. `std::time::SystemTime` is safe and in the standard library.
- **The structs themselves must leave `core`, not only their code.** Rust's orphan rule puts `impl Clock for SystemClock` in the crate that defines `Clock` or in the one that defines `SystemClock`. `Clock` and `Entropy` stay in `core` under every option below that keeps the traits there. So the impls can live elsewhere only if the `SystemClock` and `OsEntropy` declarations move with them. A declaration in `core` with its impl in another crate is not possible.

So the two sources can't stay in `core`, where RFC-0001 §5.14 lists them. #1511 (340de4b) had already split QUAL-E3-T01 to wait for this decision. QUAL-E3-T01 keeps the traits and plain types. QUAL-E3-T24 (M0a) holds the two sources, gated on OQ-PRC-20, which asks this ADR's question. QUAL-E3-T01 is M0a, and at acceptance `roadmap.md` lists 12 direct and 898 transitive dependents behind it (M0a–M4). This finding comes from reading the plan (unconfirmed static analysis); no code was written to test it.

Three further facts shape the options:

- **The traits are the part most crates need.** Rule T1 (RFC-0060 §2.1) makes decision-making code read time and randomness only through a `Clock` and an `Entropy` passed in. The concrete sources are built once, by the processes that wire the system together, and passed down: the daemon (ingress in `api`, the cluster shells, node key generation in DIST-E2-T01, the supervisor) and the runner (the scheduler and the lifecycle machine). `api` does not depend on `io`; the daemon builds `OsEntropy` and hands `api` an `Entropy`. The simulator never builds them; it uses `SimClock` and `SimEntropy` from `testkit` (QUAL-E3-T02), which no production crate may depend on (RFC-0001 §4.2 rule 9).
- **M0a already needs both sources.** PLC-E9-T01 (M0a) requires "ingress always setting a seed", and ADR-0107 says ingress sets "the client's seed, or 64 random bits it generates". Rule T1 puts `api`'s randomness behind an `Entropy`, so the M0a OpenAI path (PROD-E2-T04, PROD-E6-T03) needs `OsEntropy`. DIST-E1-T04 (M0a) has the runner drain and exit when the daemon heartbeat stalls for `runner.daemon_silence` (2 s). That timer is the runner's lifecycle machine reading time, so it needs `SystemClock`. DIST-E1-T02 (M0a) only lays out the heartbeat field in the shared-memory header and reads no clock.
- **Suspend detection needs a second clock.** RFC-0030 §5.8 has the daemon read both the suspend-aware clock and the clock that stops during suspend (`CLOCK_MONOTONIC`, `mach_absolute_time`, `QueryUnbiasedInterruptTime`), and compare them. That second reading has the same FFI needs on Windows, so it belongs beside `SystemClock`.

## Options considered

1. **Keep the traits in `core` and move the two production sources to a layer-1 crate that may already use `unsafe`.** `MonoTime`, `Clock`, `Entropy`, `ProtocolCore`, `Outbox` and `DetHashMap` stay in `core`, which stays plain data and pure functions. `SystemClock` and `OsEntropy` move. Three homes were considered:
   - **1a. `io`.** Rule 8 already allows `unsafe` there, and ADR-0001 item 4 allows it in I/O and FFI crates. `io` is the crate that already makes OS calls (io_uring, IoRing, IOCP, `preadv`, `F_NOCACHE`). The daemon, the runner and `probe` already depend on it (RFC-0001 §4.1), so no new crate edge is needed. Rule 3 allows compute crates to depend on `io`, and neither source needs tokio. The cost: a crate that wants only a clock pulls in the I/O engines.
   - **1b. `ipc`.** `unsafe` is allowed, and the daemon and the runner both depend on it. But its job is the daemon↔runner protocol and shared memory; clocks and randomness fit it less well than `io`.
   - **1c. A new layer-1 crate** (for example `syntropy-os`) holding only the OS clocks and the CSPRNG. It is the smallest `unsafe` surface, and a crate that needs a clock pulls in nothing else. It adds a row to RFC-0001 §4.1, a crate to rule 8's list, a crate to RFC-0062 §12.1's allow-list, and a workspace stub.

   Option 1 amends RFC-0001 §5.14 and RFC-0060 §2.1, and settles the split of QUAL-E3-T01 into a `core` part and a sources part that #1511 made.
2. **Widen `core`'s allow-list** in rule 1 to add `getrandom` and a crate that wraps the suspend-aware clocks safely. This keeps all the types together. This drafting found no crate that wraps `mach_continuous_time` and `QueryInterruptTime` without `unsafe` and stays as small as rule 1's existing four. It makes every crate in the workspace depend, through `core`, on OS bindings, and adds new direct dependencies to the lowest layer.
   - **2b. Add only `getrandom` to rule 1, and move only the clocks.** `getrandom`'s API is safe and its footprint is small, so `OsEntropy` could stay in `core`, and only `SystemClock` would move to `io`. Rejected: it still widens the allow-list of the layer every crate depends on, and it splits the two production sources across two crates, so a process that wires them reaches into both. It saves nothing over 1a, because the clock FFI needs a layer-1 home anyway.
3. **Allow `unsafe` in one `core` module** (`core::os`) for the clock calls, and add `getrandom` to rule 1. This amends rule 8 and rule 1, and makes `core` the first layer-0 crate with `unsafe`. Miri and the sanitizer jobs (QUAL-E3-T10) would then have to cover `core`.
4. **Write both sources in `core` with the standard library only.** Use `Instant` for the clock, and derive randomness from `std::collections::hash_map::RandomState`. Rejected: `Instant` stops during suspend on Linux, which breaks RFC-0030 §5.8's lease and suspicion timing, and `RandomState` is not a CSPRNG, which breaks rule T2 (key material from the OS CSPRNG). Listed only because it is the obvious shortcut.
5. **Build the sources in each binary crate** (`daemon` and `runner`). Rejected: both crates forbid `unsafe` under rule 8, the code would exist twice, and rule 6 (one crate per external system) would be broken by `getrandom` in two crates.

## Decision

**Option 1a.**

1. **`core` keeps the traits and the plain types:** `MonoTime`, `Clock`, `Entropy`, `TickSource`, `ProtocolCore`, `Outbox` and `DetHashMap`. Its external dependencies stay the four of rule 1, and it stays `#![forbid(unsafe_code)]`.
2. **`io` holds the production sources**, in one module, `syntropy_io::os`:
   - `SystemClock`, implementing `Clock`. `now` reads the suspend-aware clock named above for each OS. `unix_ms` reads `std::time::SystemTime`.
   - the clock that stops during suspend, as a second reading on the same type or a sibling type, so RFC-0030 §5.8's shell can compare the two without its own FFI.
   - `OsEntropy`, implementing `Entropy` over the OS CSPRNG through `getrandom`.

   Each `unsafe` block carries its written invariant (ADR-0001 item 4).
3. **Rule 6 names `io` as the one crate that depends directly on `getrandom`.** Crates that reach it transitively, such as through the TLS stack in `net`, are not affected.
4. **`getrandom` is a new direct dependency.** ADR-0021 decision 5 needs a maintainer's approval for it, and accepting this ADR gives that approval. Any binding crate the clock calls need that `io` doesn't already use (for example `libc` or `windows-sys`) also needs that approval, and the sources task lists it.

**Why 1a over the others.** It changes no rule in RFC-0001 §4.2 except to add `getrandom` to rule 6's allow-list. Rule 1 and rule 8 stay as they are, and `core` stays plain data and pure functions, as its row in §4.1 says. It puts the FFI where rule 8 and ADR-0001 item 4 already allow `unsafe`. Every crate known to build the concrete sources (the daemon, the runner, `probe`) already depends on `io`, so it adds no crate dependency edge. Option 1c is cleaner on cohesion, but it widens rule 8's list and adds a crate to fix a problem that `io` already solves. It becomes the better choice if a trigger under "What would make us revisit" fires. Options 2, 2b and 3 move OS bindings or `unsafe` into the layer every crate depends on.

### Edits on acceptance

None of these were made while this ADR was Proposed (ADR-0021 decision 9). The change that accepted it on 2026-10-22 made them:

1. **RFC-0001 §5.14.** Remove `SystemClock` and `OsEntropy` from the `core` code block, with their doc comment ("Production sources. SystemClock reads a suspend-aware clock …", lines 1576–1579 on `main` at acceptance). That comment's two rules, a suspend-aware clock and key material only from `OsEntropy`, move with the structs to `io`'s doc comments. Add a sentence below it: the production sources are in `io` (ADR-0022), and the simulator's are in `testkit`.
2. **RFC-0001 §4.1.** Add "the production `Clock` and `Entropy` sources (`SystemClock`, `OsEntropy`, §5.14)" to `io`'s Responsibility cell.
3. **RFC-0001 §4.2 rule 6.** Add "`getrandom` only in `io`".
4. **RFC-0001 header and §18.** Extend "Builds on" to ADR-0022, and add a dated §18 entry for items 1–3 with ADR-0022 as the reason. §18 records changes when they land, so it gets no entry while this ADR is Proposed.
5. **RFC-0060 §2.1.** In the code block, mark `SystemClock` and `OsEntropy` as `// syntropy-io (ADR-0022)` instead of under the `syntropy-core` comment, and add a dated revision note. Reword the doc comment's "Clocks that stop during suspend … are not used": `now` never reads them, and `io` reads them only for RFC-0030 §5.8's comparison. Rule T2 is unchanged: key material still comes only from `OsEntropy`.
6. **RFC-0062 §12.1, rule 6 row.** Add `getrandom` → `io` to the allow-list.
7. **`plan/README.md:12`.** Change "ADR-0001 to ADR-0021" to "ADR-0001 to ADR-0022", and add a row for this ADR to the foundational table.
8. **`README.md:97`** (the repository root). Change "ADR-0001 to ADR-0021 the foundation" to "ADR-0001 to ADR-0022 the foundation".
9. **`AGENTS.md:20`.** Change "0000–0021 foundation" to "0000–0022 foundation". `AGENTS.md` is an agent instruction file, so this edit needs a maintainer's approval of its own (AGENTS.md "Gates"), separate from accepting this ADR.
10. **`epics/QUAL.md`: replace the QUAL-E3-T01 and QUAL-E3-T24 rows** with the two rows under "The task rows". Keep `QUAL-E1-T03` in both Depends cells, because both criteria name it (#1511's R9 edges), and remove QUAL-E3-T24's `gate: OQ-PRC-20`.
11. **DIST-E2-T01's Depends cell.** It reads "QUAL-E3-T01 (\`Entropy\`)" today, and becomes "QUAL-E3-T01 (\`Entropy\`); QUAL-E3-T24 (\`OsEntropy\`)".
12. **QUAL-E3-T17's Depends cell** gains "QUAL-E3-T24 (\`SystemClock\`)".
13. **PLC-E9-T01's Depends cell** gains "QUAL-E3-T24 (\`OsEntropy\` for the ingress seed)". The standard library has no stable random source, so ingress's 64 random bits (ADR-0107) need `OsEntropy`.
14. **DIST-E1-T04's Depends cell** gains "QUAL-E3-T24 (\`SystemClock\` for \`runner.daemon_silence\`)".
15. **`plan/implementation/dependency-fixes.md`.** Add a dated pass with one row for each change in items 10–14, in that file's Change / Tasks / Rule / Reason table, citing ADR-0022.
16. **Plan tools.** Re-run `epics.py`, `depcheck.py`, `deps_report.py`, `trace.py` and `roadmap.py`, and commit the regenerated sections.
17. **The rule-6 table in `xtask/src/check_deps/policy.rs`** (added by #1514, which merged first). Add a `getrandom` → `io` entry to `SYSTEMS`, so QUAL-E3-T24's "`getrandom` only in `io`" criterion is checked rather than vacuous. This is CI configuration, so it needs a maintainer's approval (AGENTS.md "Gates"). In #1514 as drafted, rule 6 follows every edge kind, dev edges included, through external packages. `getrandom` is a common transitive dependency, for example of the TLS and QUIC stack under `net` and likely of `proptest`, which `core` uses as a dev-dependency. A plain entry would then fail the real workspace. The entry therefore needs a direct-dependency-only match: a new field on `System`, or a separate check, that flags a direct normal or build dependency on `getrandom` from any member other than `io`. That matches decision 3. Confirmed at acceptance: with a plain entry, `cargo xtask check-deps` fails with 5 violations, through `proptest → rand → getrandom` from `core`, `config`, `bench`, `kernels` and `tier`.

18. **`plan/open-questions.md`: OQ-PRC-20** moves to Settled with its option (c), `io`, citing this ADR. `roadmap.py` re-runs under item 16.
19. **DIST-E1-T06 and DIST-E4-T05** each built a suspend-aware clock helper in the daemon, which forbids `unsafe`. Both now read `io`'s `SystemClock` and stopped-clock reading. DIST-E1-T06 gains "QUAL-E3-T24 (`SystemClock` and the stopped-clock reading)". DIST-E4-T05's "DIST-E1-T06 (clock helper)" edge becomes "DIST-E1-T06 (suspended-interval detector)" plus the same QUAL-E3-T24 edge.
20. **PROD-E5-T01** gains "QUAL-E3-T24 (`OsEntropy` for key generation)": API keys are 32 random bytes (RFC-0050 §11.3).
21. **RFC-0030.** Its open question 6, at line 1068 on `main` at acceptance, still says RFC-0060's `SystemClock` names `CLOCK_MONOTONIC`, with no *Answered* marker, though the register already lists it as Settled. Mark it answered, and add a note under §5.8's two-clock table that both readings come from `syntropy_io::os`.

### The task rows

#1511 had already created QUAL-E3-T24 as the sources task, and QUAL-E3-T25 exists too. So these rows replace the existing QUAL-E3-T01 and QUAL-E3-T24 rows, as accepted in `epics/QUAL.md`.

| ID | Task | Milestone | Depends on | Acceptance criteria | Implements |
|---|---|---|---|---|---|
| QUAL-E3-T01 | `syntropy-core` determinism types: `MonoTime`, `Clock`, `Entropy`, `ProtocolCore`, `Outbox`, `DetHashMap` (fixed-key hasher); the production `SystemClock` and `OsEntropy` are QUAL-E3-T24's, in `io` (ADR-0022) | M0a | QUAL-E1-T01; QUAL-E1-T03 (dependency allow-list) | unit tests for each; `core` still passes its dependency allow-list (QUAL-E1-T03) and stays `#![forbid(unsafe_code)]` | RFC-0060 §2.1, §2.2, §2.4; ADR-0600 (decisions 1–2); ADR-0022 (decision 1); RFC-0001 amendment A1 |
| QUAL-E3-T24 | Production clock and entropy sources in `io`, in `syntropy_io::os` (split from QUAL-E3-T01; ADR-0022): `SystemClock` over the suspend-aware OS clock (`CLOCK_BOOTTIME`, `mach_continuous_time`, `QueryInterruptTime`) with `unix_ms` from `SystemTime`; the stopped-clock reading RFC-0030 §5.8 compares against (`CLOCK_MONOTONIC`, `mach_absolute_time`, `QueryUnbiasedInterruptTime`); `OsEntropy` over `getrandom`. A binding crate the clock calls need that `io` does not already use (such as `libc` or `windows-sys`) is a new direct dependency, which the change names for a maintainer's approval (ADR-0022 decision 4); the module docs of `core`'s `clock` and `entropy` modules point to `syntropy_io::os` for the production sources | M0a | QUAL-E3-T01; QUAL-E1-T03 (dependency rules); QUAL-E1-T06 (hosted Windows and macOS jobs) | `SystemClock::now` is monotonic across 10⁶ calls; on each Tier 1 OS a test inside `io` reads the named suspend-aware clock through its own FFI call and shows `now` agrees with it; the stopped-clock reading exists on each Tier 1 OS and is monotonic; over 10⁶ rounds of four reads in the order `now` (n1), stopped (s1), stopped (s2), `now` (n2), `s2 − s1 ≤ n2 − n1` holds, with one tick of the coarser clock as slack; `OsEntropy::fill` fills empty, 1-byte and 1 MiB buffers, and two 32-byte fills differ; every `unsafe` block has a written invariant; `cargo xtask check-deps`, with rule 6's `getrandom` entry, shows `getrandom` as a direct dependency of `io` only | RFC-0060 §2.1 (rule T2); RFC-0001 §5.14, §4.2 rule 6; RFC-0030 §5.8; RFC-0034 §3; ADR-0022 (decisions 2–4) |

The sources task is M0a, because two M0a tasks need a production source (see Context): PLC-E9-T01 needs `OsEntropy` for the ingress seed, and DIST-E1-T04 needs `SystemClock` for the daemon-silence timer. Neither row depends on QUAL-E3-T01 today, so both edges are new. The task is small, and QUAL-E3-T01 still unblocks its other M0a dependents (QUAL-E3-T02, QUAL-E4-T01, QUAL-E4-T14, QUAL-E5-T19) at once, since those need only the traits and types. Four tasks gain QUAL-E3-T24 in their "Depends on" cell (edits 11–14), and three more under edits 19 and 20 (DIST-E1-T06, DIST-E4-T05, PROD-E5-T01):

- **PLC-E9-T01** (counter-based RNG and the ingress seed, M0a), which needs `OsEntropy`: the standard library has no stable random source, `SimEntropy` is test-only, and ingress must generate 64 random bits when the client sends no seed (ADR-0107);
- **DIST-E1-T04** (runner spawn and the daemon-silence exit, M0a), which needs `SystemClock`;
- **DIST-E2-T01** (node key generation, M0b), which needs `OsEntropy`;
- **QUAL-E3-T17** (`TickSource` production counter, M0b), whose `now` returns `MonoTime` on the `Clock` timeline.

The other dependents of QUAL-E3-T01 keep their edge to the `core` part only.

QUAL-E3-T24's criteria cover the Tier 1 OSes only. Android's `CLOCK_BOOTTIME` and iOS's `mach_continuous_time` have no covering task. Their natural homes are PROD-E10-T13 (Android, M4) and PROD-E10-T16 (iOS, M4), which build the mobile runners. Adding a criterion to each is left to the accepting change, which added them.

## Consequences

- **Positive.**
  - QUAL-E3-T01 can land at once as the `core` part, which unblocks its M0a dependents that need only the traits.
  - `core` keeps no OS bindings and no `unsafe`, so every crate above it builds without them.
  - The clock and CSPRNG FFI sits in a crate already on rule 8's list, which QUAL-E3-T10's ASan and TSan builds of the `unsafe` crates cover. Its Miri job names `tier`, `ipc` and `net` only, so adding `io`'s `os` module to it is a choice for that task.
- **Negative.**
  - The determinism types are split across two crates. A reader of RFC-0001 §5.14 has to follow a pointer to find the production sources.
  - A crate that builds a concrete clock and doesn't already depend on `io` takes `io`'s engines with it.
  - `getrandom` and any new clock binding are new direct dependencies, each needing a maintainer's approval.
  - Two M0a tasks, PLC-E9-T01 and DIST-E1-T04, gain an edge they did not have.
- **Adjacent, not decided here (#1526).** QUAL-E3-T17's production `TickSource` reads the invariant TSC or CNTVCT_EL0. The independent review of this draft found that a safe call of `core::arch::x86_64::_rdtsc()` fails on the pinned toolchain with `error[E0133]` (call to unsafe function). Reading CNTVCT_EL0 needs `asm!`, which is always `unsafe`. `exec` and `obs` forbid `unsafe` under rule 8, and the plan names no crate for that implementation. PLC-E7-T17 (M0a) also uses a production `TickSource` while QUAL-E3-T17 is M0b. Issue #1526 tracks both. The `os` module in `io` is a candidate home for the counter, and the choice is left to #1526.

## What would make us revisit

- A crate that doesn't depend on `io` needs to build a concrete clock or entropy source, and taking `io` would break rule 3 or pull the I/O engines into a crate that has no other use for them. Then move the module to its own layer-1 crate (option 1c).
- The `os` module grows past the clocks and the CSPRNG, for example with sleep notices (RFC-0030 §5.9) or the `TickSource` counter. A module with that scope no longer belongs in the I/O crate, and option 1c applies.
- A maintained crate appears that reads all three suspend-aware clocks without `unsafe` in its user, with a dependency footprint as small as rule 1's existing four. Option 2 is then cheap enough to compare again.

## Revisions

- **Proposed 2026-10-21** for issue #1484, and merged as the Proposed record (#1522).
- **Accepted 2026-10-22** by the maintainers: option 1a. Accepting it approves `getrandom` as a direct dependency of `io` only (decision 4). The change that accepted it made the edits on acceptance and the corrections below.

## Corrected on acceptance (2026-10-22)

The draft described `main` before #1511, so the accepting change corrected its text. The independent review of #1522 found most of these. Each change from the draft, and why:

1. **Status** "Proposed" → "Accepted (2026-10-22): option 1a, …", and in Decision "Proposed: option 1a" → "Option 1a": the maintainers accepted option 1a.
2. **Decision 2:** "(`syntropy_io::os` is the proposed path)" → "`syntropy_io::os`": the path is decided.
3. **Related** gains QUAL-E3-T24, OQ-PRC-20 and #1511, which hold the same question on `main`.
4. **Context:** "So QUAL-E3-T01 can't meet its acceptance criteria as written" was false on `main`, because #1511 had split the task, gated on OQ-PRC-20. The paragraph now says so. The dependent counts, "11 direct and 892 transitive", are refreshed to `roadmap.md` on `main` at acceptance: 12 and 898.
5. **Option 1:** "splits QUAL-E3-T01" → "settles the split … that #1511 made".
6. **"Edits on acceptance" lead-in:** "None of these are made while this ADR is Proposed. On acceptance, the change that accepts it makes them" → past tense.
7. **Edit 1:** the line reference "1567–1570 at drafting" → "1576–1579 on `main` at acceptance".
8. **Edit 5** gains the rewording of RFC-0060 §2.1's "are not used" sentence. Decision 2 adds a stopped-clock reading, so the old sentence contradicted it.
9. **Edit 10:** "split QUAL-E3-T01 into the two rows" would have added a second QUAL-E3-T24, which `epics.py` reports as a duplicate. It now says to replace the existing rows, keep #1511's `QUAL-E1-T03` edges and drop the `gate:` segment.
10. **Edit 17:** "lands in whichever of that PR and the accepting change merges second" → #1514 merged first. Its unconfirmed reading of #1514 is now confirmed, with the result.
11. **The optional RFC-0030 item** pointed at the wrong place. Open question 6 was already Settled in the register; what was stale is RFC-0030 itself. It became edit 21.
12. **Edits 18–20 are new:** OQ-PRC-20 to Settled, which no edit did; DIST-E1-T06 and DIST-E4-T05, which built the clock FFI in a crate that forbids `unsafe`; and PROD-E5-T01, whose API keys need `OsEntropy`.
13. **"The proposed task split"** → "The task rows". Its "QUAL-E3-T24 was the next free number" was false: #1511 created T24, and T25 exists. The table now holds the rows as accepted. Those keep the `QUAL-E1-T03` edges, add QUAL-E3-T24 → QUAL-E1-T06 for its "on each Tier 1 OS" criteria, and name the `core` module-doc pointer that the review of #1535 asked for.
14. **The list of new edges** and the mobile-OS paragraph note edits 19 and 20 and the criteria the accepting change added.
