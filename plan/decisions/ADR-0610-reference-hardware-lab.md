# ADR-0610: A self-owned reference hardware lab built around the cost model's scenarios
- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the quality and release engineering area)
- Related: ADR-0004, ADR-0013, ADR-0014, ADR-0020, ADR-0021, ADR-0602, ADR-0612, ADR-0624; RFC-0060 §4.1, §9, §11; RFC-0061 §6; RFC-0062

## Context
Syntropy's targets are stated against the user's own hardware (`00-vision-and-scope.md` §7). They were derived from the cost model's worked examples:
- (a) a 70B dense model on an RTX 4090 with dual-channel DDR5;
- (b1) DeepSeek-V3-class on Ultra-class Macs over Thunderbolt 5 RDMA;
- (b2) the same model on one PC with 128 GB of RAM and NVMe;
- (c) a mixed home cluster on 2.5 GbE.

ADR-0020 requires real-hardware CI for Tier 1 on every relevant PR, and for Tier 2 weekly. It names a minimum set of hardware for this:
- Ada- and Blackwell-class GPUs;
- two Apple machines plus a TB5 RDMA pair;
- a Windows NVIDIA machine;
- AVX2 and AVX-512 hosts;
- a 10 GbE pair;
- Tier 2 AMD, Intel, GB10, Pi-class, RoCE and dual-GPU systems.

None of this is available as on-demand cloud instances in representative form. Cloud GPUs are datacenter parts on virtualised storage, and Apple unified memory, consumer DRAM channel counts, CPU-attached NVMe under thermal load and Thunderbolt RDMA do not exist in clouds at consumer scale. GitHub-hosted macOS runners are reported to expose no Metal device, so even Metal correctness needs owned machines.

## Options considered
1. **Cloud and hosted runners only.** Nothing to operate, but the measurements do not represent users, and much of the Tier 1 matrix (Metal, consumer NVMe, Thunderbolt) is unavailable.
2. **Community-contributed hardware runners.** Broad coverage. But availability, configuration drift and security of self-hosted runners on strangers' machines make it unsuitable for gating. Possible later for Tier 3 reach.
3. **A minimal lab of one rig per Tier 1 platform.** Fewer machines, but it cannot run scenarios (b1) and (c), the multi-node chaos pair, or ADR-0014's feature-versus-fallback pairs on the hardware that has each feature.
4. **Chosen: a lab of named rig classes chosen so that each rig plays several roles, plus on-demand cloud instances for coverage that needs correctness only.** It covers ADR-0020's CI hardware, the cost model's four scenarios, and RFC-0060's chaos needs. Every rig carries an inventory fingerprint and a pinned image.

## Decision
1. **The hardware test matrix** (RFC-0061 §6.2). Each rig, the property it covers, and the milestone that first needs it:

   | Rig | Property it covers | First needed |
   |---|---|---|
   | **R-ADA** | Ada-class 24 GB GPU (`sm_89`), AVX2-only desktop CPU, 128 GB of 2-channel DDR5, Gen4 NVMe, 10 GbE and RoCE NICs: scenario (a), the AVX2 CPU host, cross-load certification of `sm_80` and `sm_86` objects. It plays six roles, so a spare GPU of the same class is kept for it | M0 |
   | **R-BW** | RTX 5090 (`sm_120`, FP4/FP8), AVX-512 dual-CCD desktop CPU, 128 GB, Gen5 NVMe plus an unheatsinked Gen5 drive, 10 GbE and RoCE NICs | M0: the M0 spikes for host overhead, the host path, NVMe and CPU kernels need it from M0's third week |
   | **R-MACF** | M1- or M2-class Mac, 16 GB: the Metal floor | M0: the PR† `long`-prefill question is answered on it |
   | **R-TURING** | a Turing (`sm_75`) card in a small Linux host: the CUDA 13 floor, weekly, and the lab's only small-VRAM CUDA measurements | M0 |
   | **R-LABCPU** | an untimed many-core host, single-tenant bare metal provided on demand, under the same dispatcher and mutual-TLS JIT delivery as the rigs: CPU `F32Fixed` cross-checks on nano and micro fixtures, the publisher-reference check on PRs, the nightly simulation, the fuzz rotation, solver regression and results-site builds. Never timed. Its size is set from the measured reference throughput (RFC-0060 Open question 1), because the calibration reference runs on a GPU (decision 4) | M0 |
   | **R-WIN** | Windows 11 with an Ada- or Blackwell-class GPU | M1 |
   | **R-MBP** | M4 Pro-class MacBook Pro, 48 GB, with SME2 | M1 |
   | **R-ULTRA-A** | an Ultra-class Mac Studio with 512 GB of unified memory (M5 Ultra generation, 1.2 TB/s): scenario (b1) on one node, Metal 4 tensor ops | M1 |
   | **R-MINI** | a Thunderbolt 5 Mac pair on Apple's RDMA-capable list (R-MBP with one Mac mini, or two Mac minis): the raw-path RDMA probe that decides whether cross-node TP on Macs stays in M3 | M1 |
   | **R-AM5-4D**, **R-MID12**, **R-LAPX** | the machines most users own: a single-CCD AM5 desktop with four DIMMs on R-BW's platform; a 12 GB-class NVIDIA desktop with 32 GB of RAM; an x86 laptop | M1 |
   | **R-ULTRA-B** | a second Ultra-class Mac for the R-TB5 pair: TP = 2 and PP = 2 over Thunderbolt 5 RDMA. It joins only if the raw-path probe on R-MINI passes; otherwise cross-node TP on Macs stays simulation-only and the release notes say so | M2 |
   | **R-HALO** | Strix Halo, 128 GB: scenario (c) member and the `Unified` tier on AMD. In M2's scenario (c1) its GPU stage runs on the ggml-only `vulkan` module, labelled a Tier 2 preview; until that module exists the stage is CPU-side | M2 |
   | **R-AMD** | RDNA3 and RDNA4 cards in one Linux host, which also holds the Intel Arc card | M3 |
   | **R-ARC** | the Intel Arc card in the R-AMD host: Intel-on-Vulkan. Windows-on-Arc stays build-only | M3 |
   | **R-DUAL** | two NVIDIA cards with working peer access (current GeForce 40-series drivers disable it), such as two RTX 3090s with an NVLink bridge: intra-node TP with and without the bridge, and timed `sm_86` | M3 |
   | **R-GB10** | a GB10-class system | M3 |
   | **R-PI5** | a Raspberry Pi 5-class board | M3 |

   If the lab's Ultra-class Mac has 256 GB rather than 512 GB, scenario (b1)'s single-node case and SC-4's Mac example are measured on a model that fits, such as Qwen3-235B-A22B Q4 (142 GB), and DeepSeek-V3 on one Mac is labelled unmeasured. If no Ada-class card is in the lab at M0, R-BW is the M0 reference rig, scenario (a) is re-derived on it before the M0 exit criteria are fixed, and the release notes state the gap in timed `sm_89` coverage.
2. **Composite rigs** from the same machines:
   - the S2 cluster (R-ADA + R-MBP + R-HALO on 2.5 GbE and Wi-Fi);
   - R-PAIR10 (R-ADA + R-BW on 10 GbE);
   - R-ROCE (the same pair over RoCEv2);
   - R-TB5 (the two Ultras), when R-ULTRA-B is present.
3. **Lab infrastructure:**
   - a managed switch with data-plane VLANs, plus a per-rig quarantine VLAN that a rig's ports move to while it runs fork-derived code (revised 2026-10-19; RFC-0062 §3.3);
   - a separate admin network for PDU, switch and impairment-bridge control, which no rig port can reach, served by a small lab controller host that never runs jobs;
   - per-rig authenticated delivery of just-in-time runner configurations (mutual TLS, a per-rig certificate);
   - a Wi-Fi 6/6E access point on a 6 GHz channel; Wi-Fi scenarios stay report-only through 1.0, with no shielded enclosure;
   - a Linux impairment bridge for netem on every OS (RFC-0060 §9.3);
   - smart PDUs with per-outlet control;
   - a room-temperature sensor;
   - S3-compatible project storage for artefacts, fixtures and results, with RFC-0062's retention rules.
   - **The room.** A lockable room with power and cooling sized for every rig at full load at once, a home-like ambient of 22–26 °C [target] recorded by the room sensor, and a ≥ 1 Gbps link with outbound-only rules. Not colocation: a cold aisle would change the thermal behaviour RFC-0061 §6.1 wants measured as users run it.
4. **The untimed agent GPU pool.** Agents iterate on kernels on real GPUs that are not lab rigs.
   - The pool is kept apart from the timed lab: its own dispatcher queue, credentials and VLAN. It never runs a timed block or a gate job.
   - It holds at least one device per kernel backend under development: from M0 an NVIDIA GeForce host that boots Linux or Windows 11 with the native driver (WDDM probes need native Windows, not WSL2) and an SME2-capable M4- or M5-class Mac; from M3, when the HIP and Vulkan providers are built, an AMD GPU host and an Intel Arc host. On-demand cloud instances add data-centre architectures.
   - An agent run that has read untrusted input (ADR-0624's untrusted tier) never reaches the pool. Each lease ends with a wipe of the work directory.
   - The fixed-order F32 calibration reference on a GPU (RFC-0060 §4.1, ADR-0601 decision 7) runs on the pool or on on-demand instances, never on a timed rig.
5. **Configuration discipline.**
   - Each rig has an inventory record and fingerprint, and a pinned declarative OS image. Drivers change only by image bump.
   - A lab agent enforces cleanliness and inventory before every timed block.
   - Timed blocks run with exclusive use of the rig.
6. **No clock locking or exotic tuning.** Rigs run as users run them, with BIOS settings recorded and fixed.
7. **Physical operations.** Installing and cabling rigs, power-cut and cable-pull set-up, manual re-imaging, repairs and replacements are **Maintainer actions**. AI agents reach the lab only through the dispatcher and the lab controller (RFC-0062 §3). The target is 95% weekly availability per Tier 1 rig, and ADR-0020's 30-day demotion rule applies.
8. **Coverage on on-demand cloud instances.**
   - **AMX** is correctness-only on an on-demand instance, with no gated performance number.
   - **`sm_90` and `sm_100`**: an H100- and B200-class conformance canary runs on on-demand instances on every RC from the first M1 RC; derived certification (ADR-0602 decision 7) covers nightlies. v1 contains no kernel built only for `sm_90a` or `sm_100a`.
   - **`sm_75`** has R-TURING; GitHub's hosted GPU runner is checked in M0 as a backup.
   - **`sm_86`** is cross-load certified on R-ADA through M2 and timed on R-DUAL from M3.
9. **The off-site node for SC-26.** A cloud VM with all inbound traffic blocked at the provider's firewall, joining by invite on the nightly schedule, and from M3 a lab node on a cellular modem, so carrier-grade NAT forces the relay path. A VM with open inbound ports would test only the easy case.
10. **Refresh.** One prior generation per Tier 1 vendor stays until its platform leaves Tier 1, so regression continuity runs on hardware users still own. A new generation's rig joins once that generation is generally available, so continuity runs on old and new together. A rig is retired when its platform is demoted, or after a quarter below 95% availability.

## Consequences
- **Positive.**
  - Every SC target is measured on hardware matching the scenario it was derived from, and the typical-user machines keep a measurement on what most users own.
  - Every ADR-0020 Tier 1 and Tier 2 promise has hardware behind it, or an on-demand instance where correctness is all it needs.
  - Chaos tests include power loss and real thermal throttling, which simulation cannot fully reproduce.
  - Kernel iteration never competes with timed runs for a rig.
- **Negative.**
  - A standing lab to operate, with physical work that only maintainers can do.
  - Single points of failure: R-ADA plays six roles. Its spare GPU covers a GPU failure; the loss of the rest of the host still stalls a large part of nightly coverage until it is repaired.
  - The lab ages. GPU generations change roughly every two years, which decision 10 handles.
  - R-TB5 depends on the raw-path probe, so scenario (b1)'s two-node rows may stay simulation-only.

## What would make us revisit
- The lab cannot hold the full matrix. Drop in this order, each with the coverage it loses stated in the release notes:
  1. R-DUAL (intra-node TP stays simulation-only);
  2. the Arc card in R-AMD (Intel stays build-only);
  3. R-ULTRA-B (no TB5 pair, so cross-node TP on Macs becomes simulation-only, and ADR-0020's TB5 coverage drops to Tier 2 cadence by its own revisit clause);
  4. R-MBP (scenario c without a laptop).
- Hosted runners gain real Metal devices or consumer NVIDIA GPUs. Then move correctness jobs there and keep the lab for timed runs.
- A rig's measurements prove too noisy to gate (CV above 3% after RFC-0061's remedies). Then replace it, or demote it to report-only.
- Issue reports show many users on Xeon-class AMX workstations. Then add an AMX rig.

## Revisions
- Revised 2026-10-20: the rig list became a test matrix with the property each rig covers and the milestone that first needs it. R-BW and R-MACF move to M0, because the M0 host-path, NVMe and CPU-kernel measurements need them; R-TURING, R-MINI and three typical-user machines are added, so the CUDA floor, the TB5 RDMA probe and the hardware most users own are measured; the Arc card shares the R-AMD host; R-DUAL requires working peer access; R-LABCPU is untimed bare metal sized from the measured reference throughput; R-HALO runs scenario (c1) on the `vulkan` preview. New decisions cover the untimed agent GPU pool, on-demand coverage for architectures without a timed rig, the off-site node and refresh.
