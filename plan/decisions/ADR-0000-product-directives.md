# ADR-0000: Product directives

- **Status:** Accepted (2026-10-20)
- **Date:** 2026-10-19
- **Deciders:** maintainers

These are the binding product directives. Every RFC and ADR builds on them. Changing one needs a new ADR that supersedes this one.

## D1. Inside a device, the tiering model is weight placement across VRAM, RAM and NVMe

Inside one machine, Syntropy places model weights across GPU VRAM, pinned system RAM and NVMe. The reference design for this tiering has these parts:

- **Hot weights in VRAM.** Frequently used weights, especially MoE experts, sit in an adaptive VRAM cache that changes with the workload.
- **The rest in pinned RAM.** The full weight set stays resident in pinned RAM, and weights that miss the VRAM cache are computed where they already sit (execute-in-place on the CPU). They are not copied across PCIe for every token.
- **Large tables on NVMe.** Large lookup structures and cold data sit on NVMe and are read on demand.

Systems that tier only the KV cache across GPU, CPU memory and storage count as prior art for KV tiering (RFC-0022). They are not the reference design for weight placement.

*Revised 2026-10-20:* D1 is a reference design, not a fixed rule. Wherever measurement shows that streaming weights to the GPU beats execute-in-place at decode on a machine (OQ-PL-05, measured by QUAL-E13-T03 and TIER-E3-T06), the solver streams there. Which path wins depends on each machine's PCIe, DRAM and CPU figures, and the solver already chooses by measured cost terms (ADR-0006), so a fixed rule would only overrule the measurement.

## D2. Single-user latency first; multi-user throughput second

The default objective is the lowest latency for one interactive stream: a person's chat, or an agent session.

Multi-user and batched serving stays a supported second mode. The placement solver and scheduler must accept a different objective without a redesign, but release priorities and benchmarks put single-stream decode first.

## D3. Exact output by default

With default settings, output must match a reference implementation within the stated numerical tolerance for the chosen quantization.

Speed shortcuts that change output are opt-in, per model or per request, and are labelled as approximate wherever they show up (API, CLI, dashboard). Examples:
- skipping experts that miss the cache
- approximate attention
- lossy compression of activations on the wire

## D4. First release reaches LAN plus invite links

- Devices on the same local network discover each other with no setup.
- A device on another network joins with an invite token, reached over an overlay (Tailscale, or a relay or hole-punched QUIC path).
- Public swarms of untrusted machines are out of scope for the first release.

*Revised 2026-10-20:* "First release" in D4 means **1.0**. The first public release is M1a, a Linux CUDA and CPU preview of single-device tiering, with neither LAN discovery nor invite links. LAN discovery arrives at M2 and invite links at M3 (RFC-0001 §16), so D4's promise and its swarm exclusion apply to 1.0 and to every release after it.

## D5. The plan is an original design

Plan documents describe techniques on their own merits. They do not name, cite or link the third-party inference systems or system papers studied during research.

These may be named where a document depends on them, interoperates with them or measures against them:
- technologies, libraries and standards (for example CUDA, Metal, Vulkan, ggml, io_uring, QUIC, safetensors, GGUF)
- hardware vendors and their published specifications
- API shapes such as the OpenAI and Anthropic APIs

*Revised 2026-10-20:* D5's scope. The research notes are public with the rest of the plan, and a component's name identifies its system as surely as the system's own name, so:
- The `research/` notes are plan documents, and D5 covers them.
- A hardware or platform vendor's own inference runtime may be named where a document weighs it as an interoperation candidate, such as a route to an NPU.
- A component name of a system D5 covers, such as its library, binary or tool name, counts as naming that system and is not used. Such names go on the banned-terms list that the D5 naming check scans.
- Public benchmark reports are not plan documents. They may name a baseline engine, on the conditions ADR-0612 sets, because a comparison a reader cannot reproduce is worth little; plan documents stay name-free.

## D6. Implementation languages are chosen for speed on this workload

Languages are chosen first on delivered performance for this workload: kernels, I/O submission, DMA and the network data plane. Portability is the second criterion and familiarity the last.

Each component may use a different language when that is faster. For example, the system layer can use one language and each backend its own kernel language.

## D7. Use current technology and keep a fallback

The design targets the largest model and the highest tokens/s the user's hardware can physically deliver. That means adopting current hardware and software capabilities rather than designing to the lowest common denominator. Examples include:
- low-precision tensor-core formats
- modern async I/O and direct-to-GPU storage paths
- RDMA-class links
- unified-memory APUs
- modern speculative decoding and attention variants

Every such feature has a fallback path, so Syntropy still installs and runs on any supported device.

*Revised 2026-10-20:* direct-to-GPU storage is not in v1. GeForce cards get GPUDirect Storage only in compatibility mode, a CPU bounce through host memory, and DirectStorage's gain is GPU decompression, which quantised weights do not benefit from. The probe reports both capabilities only, and modern async I/O (io_uring, IoRing) carries the storage path (RFC-0021 §9).

## D8. Open source, built by AI agents, with no team ownership

*Added 2026-10-20.*

- **Open source.** Syntropy is open source from its first commit: a public repository under the ADR-0018 licence. Its plan documents are public too.
- **AI agents build it.** They implement, test, review and document the system by executing the tasks in `plan/implementation/`.
- **No teams, no team ownership.** Areas such as placement, tiers, distributed, compute, product and quality divide the codebase and the plan into domains. They are not groups of people.
- **Maintainers are the only human role.** They approve the change classes ADR-0021 lists and do the steps that need a person: holding and using offline signing keys, physical lab work, the accounts the release and signing services run under, and the security contact. Those steps are marked **Maintainer action**.
- **Gates review every change.** Automated gates and an independent review agent check each change, and a change in a listed class also needs a maintainer's approval.

ADR-0021 sets out what this means for review, capacity, contributions and risk.

## Revisions

- **Revised 2026-10-20:**
  - D1: a reference design; the solver streams wherever measurement shows streaming wins at decode on that machine.
  - D4: "first release" reads as 1.0, because the first public release is a single-device Linux preview.
  - D5: the research notes are in scope; a vendor runtime may be named as an interoperation candidate; a covered system's component names may not; benchmark reports may name baselines under ADR-0612.
  - D7: direct-to-GPU storage is not in v1; the probe reports the capability only (RFC-0021 §9).
  - D8: maintainers are the human role, and gates plus an independent review agent check every change.
