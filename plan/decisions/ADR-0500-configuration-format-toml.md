# ADR-0500: Configuration format: TOML, with one schema written as Rust types
- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the product surface and operations area)
- Related: ADR-0000 (D4), ADR-0001, ADR-0014; RFC-0054, RFC-0051, RFC-0052

## Context
Syntropy runs with no configuration file by default (SC-24). When a user does write one, it is to change a handful of things: a listener, how much RAM a laptop keeps, a per-model objective, a placement pin, a regex tensor override. Distro packages and administrators also need to ship settings without editing the user's file. The dashboard and `syntropy config set` must edit the same file without destroying the user's comments.

The file is read by a Rust daemon (ADR-0001), and its values include units (bytes, durations, fractions), selectors and regular expressions.

## Options considered
| Option | For | Against |
|---|---|---|
| **TOML** | Cargo's format, familiar to Rust users; explicit types; comments; `toml_edit` edits files and keeps comments and layout; JSON Schema support in editors through the `#:schema` directive | deep nesting is awkward (arrays of tables for pins); no references or includes |
| YAML | familiar from container tooling; good for nesting | implicit typing (`no` → false, `1.10` → 1.1, `0700` octal in some parsers); indentation errors are silent; comment-preserving Rust editors are weak |
| JSON / JSON5 | universal; JSON5 allows comments | JSON has no comments; JSON5 is uncommon and has no comment-preserving editor in Rust |
| KDL or HCL | expressive, readable | unfamiliar to most users; immature comment-preserving Rust tooling for KDL; HCL tooling is centred on another ecosystem |
| Environment variables and flags only | nothing to parse | per-model tables, pins and override lists do not fit; unreadable at scale |
| A settings database edited through the dashboard | rich UI | hidden state that a diff cannot show; two sources of truth |

## Decision
1. Configuration files are **TOML 1.0**, UTF-8, with `config_version = 1` and a `#:schema` line in every file Syntropy writes.
2. **One schema, written once as Rust types** in `syntropy-config` with derive attributes (scope, reload class, owner, default, range, doc). It generates the parser and validator, the reload classifier, environment variable names, the reference docs, a JSON Schema per `config_version` and the dashboard's TypeScript types. A key without a doc, owner, default, scope or reload class fails the build.
3. **Edits preserve the file.** `syntropy config set` and the dashboard edit through `toml_edit`, validate before writing, and replace the file by atomic rename.
4. **Drop-in directory** `config.d/*.toml` beside the system file, merged in lexical order, for packagers.
5. **Unknown keys are errors** (RFC-0001 §11.2). Renamed or removed keys stay accepted, with a deprecation warning, for RFC-0062 §10.1's window. A file from a newer `config_version` is rejected; an update rollback restores the pre-update copy (RFC-0052 §7.4).
6. Units are written as strings with suffixes (`"16GiB"`, `"250ms"`, `"10%"`); a bare integer means bytes only for byte-sized keys.

## Consequences
- **Positive.**
  - Users get editor completion and a precise error with file and line for every mistake.
  - Docs cannot drift from the code, because both come from the same definition.
  - Packagers have a supported place for their settings.
- **Negative.**
  - Arrays of tables (`[[policy."x".pin]]`) are verbose. The dashboard's pin editor and `syntropy config set` reduce how often users type them.
  - A derive macro is one more piece of build machinery to maintain.
- **Work created.** RFC-0054 (schema, validation, reload), RFC-0051 (`config` subcommands, settings page), RFC-0052 (drop-ins in packages; schema publication with releases).

## What would make us revisit
- Users routinely need includes, variables or conditionals across many nodes. That would point to a generated-config workflow, not a richer format.
- `toml_edit` cannot preserve formatting for constructs users actually write (seen in bug reports after M1).
- A second front end (a desktop app) needs a schema language other than Rust types. Then move the source of truth to the JSON Schema and generate Rust from it.
