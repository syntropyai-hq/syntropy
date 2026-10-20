# ADR-0431: The HF `tokenizers` crate as the tokenization engine, with GGUF metadata converted into it; `minijinja` for chat templates

- Status: Accepted (2026-10-20); revised 2026-10-20 (decision 7)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the compute and models area)
- Related: ADR-0001, ADR-0005, ADR-0018, ADR-0020; RFC-0043 §8; RFC-0050 §6; RFC-0060

## Context

- Safetensors checkpoints ship `tokenizer.json`, the publisher's own tokenizer definition, interpreted by the HF `tokenizers` library (Rust, Apache-2.0). GGUF files carry the tokenizer as metadata: a model type (`gpt2`, `llama`, `bert`, `t5`, …), tokens, merges or scores, token types, special IDs and a pre-tokenizer name.
- A token-level difference changes every downstream result: prefix-cache hits (RFC-0022), the parity corpus (RFC-0060), stop handling and tool-call parsing (RFC-0050).
- The binary must cross-compile to every ADR-0020 target, including iOS and Android, so C dependencies are a cost (ADR-0001).
- Chat templates are Jinja; RFC-0050 renders them with `minijinja` plus the helpers templates use.
- RFC-0050 asks this RFC to fix per-family tool-call and reasoning syntax; RFC-0001's crate table puts tokenization in `syntropy-text`, owned by RFC-0050.

## Options considered

| Option | For | Against |
|---|---|---|
| A. A Syntropy-owned implementation of every tokenizer type | No dependency; full control | Re-implements BPE, SentencePiece-style BPE, WordPiece and Unigram, normalisers and pre-tokenizers, with token-exact parity against publishers as an open-ended task |
| B. HF `tokenizers` for `tokenizer.json` models; Syntropy's own code for GGUF metadata tokenizers | Publisher behaviour for safetensors | Two engines; GGUF models tokenized by different code than their safetensors originals |
| **C. HF `tokenizers` for every model, pure-Rust build; GGUF tokenizer metadata converted into its models at census time (byte-level BPE from merges plus a pre-tokenizer looked up by name; SentencePiece-style BPE with merges derived from scores; WordPiece; Unigram); unknown types and pre-tokenizer names refused** | One engine; GGUF and safetensors of the same model tokenize identically, which is testable; cross-compiles | A dependency to pin and track; a table of pre-tokenizer names to maintain |

## Decision

**Option C**, plus `minijinja` for templates (as RFC-0050 already uses).

1. `tokenizers` is pinned and built with its pure-Rust regex feature.
2. `graph::tokenize` converts GGUF tokenizer metadata into a `tokenizers` model at census time and stores it serialized in `TokenizerSpec::model`, so request-time code in `syntropy-text` loads one form for every model.
3. An unknown GGUF tokenizer model or pre-tokenizer name is `Unsupported (Graph, 6)`: a guessed regex silently changes tokens.
4. Conformance: token-exact agreement with the source repository's `tokenizer.json` on the tokenizer corpus, through both the shipped-JSON and the converted-GGUF paths; byte-level round trips; template renderings matching recorded publisher renderings.
5. Template precedence: user override, then the shipped template, then the family default (only when none ships). Shipped templates are never replaced silently.
6. Each family declares its `ChatFormat` (tool-call pattern from RFC-0050 §6.2, reasoning syntax, extra stop tokens), with marker strings read from the tokenizer's special tokens and template.
7. *Revised 2026-10-20:* **Content is tokenized literally.** Rendering yields a segment map of template spans and content spans. Template spans are tokenized with special-token matching on; content spans (every message role, tool results, documents) with it off, so no special token ID can come from content and content round-trips byte for byte. Raw special tokens reach a model only through an explicit native-API opt-in, under RFC-0050's token-ID range check. Each `ChatFormat` marker resolves to a token ID where the tokenizer defines it as special, and output parsers match tool-call and reasoning markers on those IDs; text matching remains only for formats whose markers are ordinary text. Conformance adds injection and fuzz cases with every special token's text in every content position (RFC-0043 §8.6, §8.7).

## Consequences

- Positive: publisher-exact tokenization for `tokenizer.json` models, and a testable equality between GGUF and safetensors forms of the same model.
- Positive: one engine and one serialized form at request time.
- Negative: the pre-tokenizer name table must follow ggml pin bumps; new names are refused until added.
- Negative: an upstream `tokenizers` change can alter tokens; pin bumps run the tokenizer conformance suite.
- Positive (decision 7): user, tool and document content cannot forge roles, system turns or tool calls, which matters because agent clients execute parsed tool calls.
- Negative (decision 7): the renderer must keep a segment map, so a template is rendered with content placeholders rather than as one opaque string (RFC-0050).

## What would make us revisit

- The pure-Rust build of `tokenizers` is measurably slower than needed for ingress budgets in RFC-0050, or fails to build on a Tier 3 target.
- A popular GGUF tokenizer type cannot be represented in the engine's models without loss. Then add a Syntropy-owned model for that type only.

## Revisions

- Revised 2026-10-20: decision 7 added: content spans are tokenized with special-token matching off, markers are parsed on token IDs where they are special tokens, and raw special tokens need a native-API opt-in. Before this, a literal control string in a message, tool result or document became a real control token. RFC-0050 owns the renderer's segment map, the opt-in and the token-ID range check.
