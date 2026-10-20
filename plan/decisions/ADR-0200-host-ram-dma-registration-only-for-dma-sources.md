# ADR-0200: Host RAM tier: register for DMA only what a device copies

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the tiers area (RFC-0020))
- Related: ADR-0007, ADR-0013, ADR-0014, ADR-0020; RFC-0001 §9.1–§9.4, §11.3; RFC-0010, RFC-0011, RFC-0020, RFC-0021, RFC-0023, RFC-0052

## Context

ADR-0007 makes the hot RAM tier pinned anonymous memory with per-OS budgets and progressive pinning. Two different things hide under "pinned":

1. **Locked**: the OS will not page the memory out (`mlock`, `VirtualLock`, wiring).
2. **DMA-registered**: a GPU or NIC can copy from it directly (`cudaHostRegister`, `hipHostRegister`, RDMA memory regions, io_uring fixed buffers).

Their budgets differ, and the second is the scarce one:
- Windows lets only about half of RAM be locked for GPU access, and WDDM caps host registration for helper GPUs at 8 GiB (`tiered-memory-inference.md` §10.2).
- WSL2 caps `cudaHostAlloc` at about 1 GiB.
- io_uring fixed buffers are charged to `RLIMIT_MEMLOCK` and limited to 1 GiB each (ADR-0013), and that limit defaults to 8 MiB for most processes on current Linux distributions unless the service unit raises it.
- On Linux, whether a GPU provider's registration counts against `RLIMIT_MEMLOCK` is undocumented for current drivers.

What needs which:
- A CPU kernel executing in place reads anonymous memory at the same rate whether or not it is locked. The measured 2.2× gap (19 vs 42.8 GB/s on Windows) is between file-backed and anonymous memory, not between locked and unlocked (`tiered-memory-inference.md` §10.1).
- A GPU copy from unregistered memory goes through a driver staging buffer at about a quarter of the pinned rate (≈6 vs ≈24 GB/s on PCIe 4.0 x16, `hardware-and-performance-model.md` §2.3), so every DMA source must be registered.
- In the reference design (D1) most RAM-resident bytes are experts computed in place by the CPU. Only the copied-miss share, prefill streaming sources, staging rings, NVMe read destinations and network pools are DMA'd.

## Options considered

| Option | For | Against |
|---|---|---|
| A. Register every host byte for DMA | One class; any byte can be copied | Spends the scarce budget on bytes no device copies; on Windows and WSL2 it caps the RAM tier at the registration limit; under default memlock limits io_uring registration fails outright |
| B. Lock everything, register nothing | Simple; no driver dependence | Copied misses and prefill streaming fall to the pageable rate; violates RFC-0001 §9.4 |
| **C. Three pin classes: `DmaPinned` (locked and registered) for DMA sources and destinations; `Locked` (locked only) and `Pageable` (anonymous, unlocked) for CPU-only bytes** | The scarce budget goes where it changes a rate; CPU-only groups are not capped by DMA limits; failure to lock degrades to `Pageable` without changing bandwidth | Two host tiers for the solver to reason about; the miss balancer must know which experts are copyable |

## Decision

**Option C.**

1. `HostPinned` means **DMA-registered**: locked and registered with every provider, NIC or I/O ring that moves bytes in or out of it.
2. Groups that only the CPU computes are homed in `HostPageable`, which is **anonymous memory, never file-backed**, locked when the lock budget allows (`Locked`) and otherwise left `Pageable`.
3. The solver sees two budgets per NUMA domain over one pool of RAM: `HostPinned.usable = pin_limit` (DMA-registrable) and `HostPageable.usable = host_usable`, and must hold both rows: `bytes(HostPinned) ≤ pin_limit` and `bytes(HostPinned) + bytes(HostPageable) ≤ host_usable` (RFC-0020 §5). RFC-0010 publishes the pool; RFC-0011 adds a (CPU, `HostPageable`) execution pair and the joint row. The probe measures CPU `W_eff` over both tiers so any rate difference is measured, not assumed.
4. Copied misses and prefill streaming use only `DmaPinned` and `Resident` sources. An expert whose home is not `DmaPinned` is always an in-place miss.
4a. A RAM expert cache over an `Nvme` home is split the same way: a `DmaPinned` part for the copyable share and the sources of VRAM admission, and a `HostPageable` part for experts only the CPU computes. NVMe direct reads need no registration, so the in-place part costs no pin budget. In scenario b2 this is what lets a ≈115 GB RAM cache coexist with Windows's ≈64 GB DMA ceiling on a 128 GB machine.
5. io_uring and IoRing fixed buffers are registered only for hot destinations (staging, RAM expert-cache slots over an `Nvme` home, row caches, KV spill pages). Bulk `Loading` reads use unregistered direct I/O.
6. Registration failure is never fatal: the range stays resident as `Underpinned`, is costed as `HostPageable`, and the runner reports the registrable total it achieved.

## Consequences

Positive:
- On Windows and WSL2 the RAM tier is no longer capped by the GPU lock limit; only the DMA sources are. This answers RFC-0001 §17 item 4 for v1 without a large-page path.
- A desktop run without raised limits (no `CAP_IPC_LOCK`, default memlock) still gets full CPU execute-in-place bandwidth.
- The registration budget is spent on the copied-miss share, where ADR-0006 item 8 says PCIe helps.

Negative:
- `HostPageable` homes can be paged out by the OS under pressure when they could not be locked. RFC-0020's yield ladder keeps this rare, and rung 5 makes it deliberate.
- The solver and the balancer must track the pin class of every RAM-resident expert.
- RFC-0001 §9.1's description of `HostPageable` ("heap, or mmap of pack extents") needs the amendment proposed in RFC-0020.

## What would make us revisit

- RFC-0010's probe shows CPU `W_eff` over unlocked anonymous memory more than 5% below locked memory on a Tier 1 platform (for example through page-table or TLB effects): lock all CPU-homed groups there, within the lock budget.
- A Tier 1 platform's registration budget turns out larger than its lock budget, so the distinction stops mattering.
- Field reports show `HostPageable` homes being paged out under normal desktop use despite the yield ladder.
