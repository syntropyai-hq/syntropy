# ADR-0506: Dashboard stack: Preact and TypeScript, built with Vite, embedded in the binary
- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the product surface and operations area)
- Related: ADR-0001, ADR-0015, ADR-0018, ADR-0503; RFC-0051 §5, RFC-0050, RFC-0062

## Context
Every daemon serves a dashboard (RFC-0051 §5). Its views need several things:
- **Interactive views.** Placement previews with editable policy, three-way candidate comparison, plan explanations with drill-down, and topology with instance overlays.
- **Live charts** at 1 Hz from SSE: tok/s, per-term stage times, hit-rate heat maps.
- **Settings forms** generated from the configuration schema.

Constraints:
- The bundle ships inside one signed binary per platform (ADR-0015), with no runtime downloads.
- It works offline, makes no external requests, and loads quickly on a phone over Wi-Fi.
- A plain `cargo build` must work for contributors who have no JavaScript toolchain.
- The licence policy applies (ADR-0018): MIT, Apache-2.0 and similar.

## Options considered
| Option | For | Against |
|---|---|---|
| A. Preact + signals + TypeScript, Vite build, uPlot charts, hand-written SVG; types generated from Rust | ≈ 4 KiB runtime with a React-compatible component model many contributors know; fine-grained updates for live metrics; uPlot is small and fast for time series; all MIT | a Node toolchain in the release build; two languages in the repository |
| B. React | largest ecosystem | heavier runtime for the same model; nothing needed here that Preact lacks |
| C. Svelte 5 or Solid | small output, good reactivity | smaller pools of familiar contributors and fewer charting examples |
| D. Rust → WebAssembly (Leptos-class) | one language; shared types without generation | larger bundles; weaker chart libraries; every UI change pays Rust compile times; fewer front-end contributors |
| E. Server-rendered HTML with a hypermedia library | no JavaScript build; tiny | preview and explanation views are interactive enough that it turns into hand-written JavaScript anyway; charts still need JavaScript |
| F. Desktop app shell (webview or native) | OS integration | a second runtime to sign, notarise and update; the browser already exists |

## Decision
1. **Option A.** The dashboard is TypeScript with Preact and `@preact/signals`, bundled by Vite, with uPlot for time series and hand-written SVG for topology, memory bars and layer strips. There is no CSS framework and no web fonts (system UI stack, tabular numerals). Dependencies must be MIT, Apache-2.0, BSD or ISC, checked in CI alongside `cargo-deny`.
2. **Types are generated** from the `syntropy-api` and `syntropy-config` Rust structs, so API and schema changes fail the dashboard's type check in CI.
3. **Embedded and precompressed.** The release bundle is embedded in the `cli` crate with brotli and gzip variants and content-hashed names. A `cargo build` without Node embeds a placeholder page that explains how to build the dashboard. Release CI always builds the real bundle and fails if the placeholder would ship.
4. **Budgets.** At most 300 KiB brotli for the initial load and 600 KiB for all routes, enforced in CI.
5. **No external requests**, enforced by the Content-Security-Policy and by an end-to-end test that fails on any cross-origin request.

## Consequences
- **Positive.**
  - Small, fast, offline-capable UI.
  - Familiar technology for front-end contributors.
  - Rust and TypeScript cannot drift apart silently.
- **Negative.**
  - Release builds need Node and pnpm, so there is a second dependency tree to keep current (Renovate) and audit.
  - Contributors who change the API may need to rebuild the dashboard to see type errors locally.
- **Work created.**
  - RFC-0051: views.
  - RFC-0062: dashboard build job, size budget gate, licence check for npm dependencies, Renovate configuration for the `dashboard/` tree.
  - RFC-0060: Playwright and accessibility suites.

## What would make us revisit
- The bundle cannot stay within budget as views grow. Split routes further, or reconsider option E for the simple views.
- A Rust-to-WebAssembly UI stack reaches bundle sizes and charting parity close to option A. Single-language maintenance might then justify a switch.
- Users want a native desktop experience beyond the tray helper. That would be a separate ADR for a desktop shell around the same bundle.
