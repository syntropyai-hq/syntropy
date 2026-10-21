# ADR-0405: One `DeviceClass` in `core`, with kernel-level variants, for placement, packs and certificates

- Status: Proposed
- Date: 2026-10-21
- Deciders: maintainers (drafted by the compute and models area)
- Related: ADR-0005, ADR-0014, ADR-0016, ADR-0401, ADR-0402, ADR-0410, ADR-0602, ADR-0622; RFC-0001 §4.2, §5.2, §5.6, §5.13, §5.15; RFC-0010 §1.2, §2.1; RFC-0011 §3.3, §4; RFC-0032 §5.2; RFC-0040 §2.2, §6.2, §6.5, §7.2, §7.3, §7.5; RFC-0041 §1, §8.5; RFC-0060 §6.4; RFC-0061 §6.2

## Context

`DeviceClass` is the key that decides which kernels and pack layouts apply to a device. RFC-0001 uses it in three places but never defines it, and the ownership table in RFC-0001 §5.13 has no row for it:

- `DeviceInfo::class` (§5.2, `core`): "backend plus architecture generation, e.g. CUDA sm_120, Metal Apple9, x86-64-v4 with AMX".
- `PackKey::class` (§5.6, `manifest`).
- `OpCert::class` inside `ConformanceManifest` (§5.15, `core`).

Two RFCs then define the type, with different variants.

| Variant | RFC-0010 §1.2 | RFC-0040 §2.2 | Difference |
|---|---|---|---|
| CUDA | `Cuda { sm: u16 }` | `Cuda { sm: u16 }` | none |
| HIP | `Hip { gfx: u32 }` | `Hip { gfx: u32 }` | none |
| Metal | `Metal { family: AppleGpuFamily, cores: u16 }` | `Metal { family: AppleFamily, metal4: bool }` | family type name differs; the second field is GPU core count in RFC-0010 and "Metal 4 library in use" in RFC-0040 |
| Vulkan | `Vulkan { vendor: u32, device: u32, driver: VkDriverId }` | `Vulkan { vendor: u32, coopmat: bool, subgroup: u8 }` | RFC-0010 keys on the exact GPU model and driver; RFC-0040 keys on the shader-variant axes |
| CPU | `Cpu { arch: CpuArch, level: IsaLevel }`, six levels: x86-64-v2, v3, v4; aarch64 v8.0, v8.2-dotprod, v9-sve2 | `Cpu { isa: CpuIsa }`, eight levels: `X86V2`, `X86V3`, `X86V4`, `X86V4Amx`, `Arm80`, `Arm82Dot`, `Arm86I8mm`, `Arm92Sme2` | RFC-0040 adds AMX, I8MM and SME2 levels; RFC-0010 has an SVE2 level that RFC-0040 lacks |
| NPU | `Npu { vendor: u32, model: u32 }` | absent | RFC-0040 has no NPU provider in v1 |

Neither is a refinement of the other. The Metal and Vulkan fields are on different axes, and each CPU list has a level the other lacks. The two definitions give different keys for the same device, so they cannot both be the pack and certificate key.

Where each consumer needs the type, and what it needs:

| Consumer | Where | What the key must separate |
|---|---|---|
| Inventory | `DeviceInfo::class` (RFC-0001 §5.2); filled from `ProviderDevice::class` at probe step E5 (RFC-0010 §2.1) | whatever the provider reports |
| Placement | RFC-0011 §4 filter F1 (coverage per class) and §3.3 (the per-class workspace bound); RFC-0032 §5.2 router islands of one class | devices that run the same kernels |
| Pack key | `PackKey` (RFC-0001 §5.6, RFC-0041 §1); `packs/<device-class>/seg/` (ADR-0410); `GetExtent { class, key }` between nodes (RFC-0041 §8.5); ADR-0016 decision 7 | devices that accept the same pack layout |
| Certificate key | `OpCert` (RFC-0060 §6.4, RFC-0001 §5.15); certified and derived levels (RFC-0040 §6.2) | devices that run the same certified kernel objects |
| Selection cache | `SelectionKey::class` (RFC-0040 §6.5, ADR-0401 decision 6) | as the certificate key |
| Provider | `ProviderDevice::class` (RFC-0040 §2.2); `DeviceClass::Metal { family, metal4 }` (RFC-0040 §7.2); `CpuIsa` detection (RFC-0040 §7.3) | which library or function table the provider loads |

Every consumer wants the same thing: two devices with equal classes run the same kernels and accept the same packs. That is RFC-0010's own description of the type, so the conflict is in the variants, not in the purpose.

Facts that bear on the choice:

- RFC-0001 §4.2 rule 1: `core` depends on no workspace crate. `DeviceInfo` and `ConformanceManifest` are `core` types (RFC-0001 §5.2, §5.15) and both hold a `DeviceClass`, so the type must live in `core`. `kernels` depends on `core` and can re-export it.
- `core` already holds RFC-0010's variants: `crates/core/src/resource/device.rs`, merged in #1490 (PLC-E1-T01). `core` also holds `DeviceFeatures`, whose bits match RFC-0040 §2.2 one for one.
- COMP-E1-T01 (issue #98) lists `DeviceClass` and `DeviceFeatures` among the supporting types `syntropy-kernels` defines.
- The CPU kernel families follow ADR-0402: AMX and SME2 microkernels exist, SVE2 kernels do not in v1. RFC-0040 §7.3 selects the AMX tile layout (`cpu.amx16`) on `X86V4Amx` and SME2 kernels on `Arm92Sme2`, so those levels change the pack layout and the kernel objects. An SVE2 level selects nothing different.
- The Metal provider loads `metal4.metallib` or the Metal 3.2 library by OS version and family (RFC-0040 §7.2), so `metal4` changes the kernel objects. GPU core count changes no kernel and no layout.
- Vulkan shaders come in variants over subgroup size, cooperative matrix and integer dot (RFC-0040 §7.5). A device ID and driver ID change no shader variant; the pipeline cache is already keyed by device UUID and driver version, and `SelectionKey` carries the driver.
- Certificates are issued per class on the lab rigs (RFC-0061 §6.2). A class that includes core count or the exact Vulkan device ID multiplies the classes the lab must cover, and derived certification exists only for NVIDIA targets (RFC-0060 §6.4).
- Nothing keyed by `DeviceClass` has been written yet: no pack, no released `ConformanceManifest`, no saved capability profile in the field. After packs and certificates exist, a change to the type rebuilds every pack of the affected classes (ADR-0622) and re-issues every `OpCert`.

## Options considered

| Option | For | Against |
|---|---|---|
| **A. One `DeviceClass` in `core` with RFC-0040's variants, plus RFC-0010's `Npu`; `kernels` re-exports it** | One key everywhere; satisfies §4.2 rule 1; the variants are the ones that change kernels and layouts (ADR-0402, RFC-0040 §7); the lab covers few classes | Changes the `core` type merged in #1490 (one file and its tests); RFC-0010 loses the Vulkan device and driver IDs and the Metal core count from the class, though `PciInfo`, `DriverInfo` and the device name keep them |
| B. One `DeviceClass` in `core` with RFC-0010's variants; `kernels` re-exports it | No `core` code change | Packs and certificates split by GPU core count and by Vulkan device and driver ID, which change no kernel; no AMX, I8MM or SME2 level, so the CPU provider cannot key its AMX and SME2 layouts; an SVE2 level with no kernels behind it; RFC-0040 §7.2, §7.3 and the COMP tasks change |
| C. Two types with a stated mapping: a hardware description in `core` (RFC-0010's variants, renamed, e.g. `DeviceModel`) and a kernel class for packs and certificates (RFC-0040's variants) | Keeps both sets of fields | `OpCert` lives in `core`, so both types end up in `core` anyway; the hardware fields already exist in `PciInfo`, `DriverInfo` and `name`; a second field on `DeviceInfo` and an RFC-0001 §5.2 amendment; the mapping cannot be computed from the hardware fields alone, because `metal4` and `X86V4Amx` depend on the OS and on a runtime permission |
| D. `DeviceClass` owned by `kernels`; `DeviceInfo` and `OpCert` move out of `core` | Owner and crate agree | Breaks §4.2 rule 1 as written, or moves `ResourceGraph` and `ConformanceManifest` out of `core`: a large RFC-0001 §4 and §5 amendment for one enum |
| E. Leave both definitions | No edits | Two types named `DeviceClass` in two crates; the probe must convert between them; pack and certificate keys depend on which crate wrote them |

## Decision

**Option A** (proposed).

1. **One type.** `DeviceClass` is defined once and lives in `syntropy-core`. `syntropy-kernels` re-exports it (`pub use`) and never defines its own. `ProviderDevice::class`, `DeviceInfo::class`, `PackKey::class`, `OpCert::class` and `SelectionKey::class` all use it.
2. **What a class separates.** A field belongs in `DeviceClass` only when a change in it can change the pack layout or the set of kernel objects the providers load for the device. Facts that change neither (core counts, PCI IDs, driver versions) stay in `DeviceInfoExt`, `PciInfo`, `DriverInfo` and `SelectionKey`.
3. **Variants.** RFC-0040 §2.2's variants, with two adjustments:

   ```rust
   pub enum DeviceClass {
       Cuda { sm: u16 },
       Hip { gfx: u32 },
       Metal { family: AppleGpuFamily, metal4: bool },
       Vulkan { vendor: u32, coopmat: bool, subgroup: u8 },
       Cpu { isa: CpuIsa },        // X86V2, X86V3, X86V4, X86V4Amx, Arm80, Arm82Dot, Arm86I8mm, Arm92Sme2
       Npu { vendor: u32, model: u32 },
   }
   ```

   - The Apple family type keeps the name already in `core`, `AppleGpuFamily`.
   - `Npu` stays from RFC-0010, so an enumerated NPU with `DeviceProblem::NoProvider` still has a class. No provider produces it in v1, so no pack or certificate is ever keyed by it.
   - `CpuArch`, `IsaLevel` and `VkDriverId` leave `DeviceClass`. `core` keeps them only if another type uses them.
4. **Ownership.** RFC-0040 owns `DeviceClass` and `DeviceFeatures`. RFC-0010 contributes `Npu` and the rule for merging provider records into one `DeviceInfo`.
5. **Order of work.** The `core` type changes before any pack, certificate or saved profile is written. Under this ADR, COMP-E1-T01 re-exports `core`'s `DeviceClass` and `DeviceFeatures` instead of defining them.

## Consequences

- Positive: one key for packs, certificates, selection and placement. A pack copied from a peer of the same class (ADR-0016 decision 7) is one that the receiving node's kernels accept.
- Positive: the lab certifies per kernel-relevant class, not per GPU SKU or driver build.
- Positive: `kernels` stays a thin layer over `core` types, as RFC-0001 §4.1 describes it.
- Negative: the `core` type and its tests from #1490 change before COMP-E1-T01 can rely on the final variants.
- Negative: the class now depends on the OS and on runtime permissions. A macOS upgrade that enables Metal 4, the Linux AMX permission request, or `features.disable = ["amx"]` changes the class, so the node builds new packs and looks up different certificates. This is intended: the kernels and the layout change too.
- Negative: a Vulkan device's class no longer names the GPU model. Diagnostics read it from `PciInfo` and `name`.

## Edits that follow acceptance

These are not made by this ADR. They follow in one change after a maintainer accepts it.

1. **RFC-0001 §5.13:** add a row: `DeviceClass`, `DeviceFeatures` | core | 0040 | 0010 (`Npu`, provider-record merge).
2. **RFC-0001 §5.2:** on `DeviceInfo::class`, cite "defined in RFC-0040 §2.2; ADR-0405".
3. **RFC-0001 §18:** log the amendment with the date and this ADR.
4. **RFC-0010 §1.2:** replace the `DeviceClass` block with a pointer to RFC-0040 §2.2 and ADR-0405; keep a note that `Npu` is RFC-0010's addition.
5. **RFC-0040 §2.2:** state that `DeviceClass` and `DeviceFeatures` live in `syntropy-core` and that `kernels` re-exports them; rename `AppleFamily` to `AppleGpuFamily`; add `Npu`; add decision 2's rule to the doc comment.
6. **COMP.md, COMP-E1-T01:** "`DeviceClass`, `DeviceFeatures`" becomes "re-exports of `core`'s `DeviceClass` and `DeviceFeatures` (ADR-0405)".
7. **A new task** in the placement epic (`PLC.md`, next free number in E1) to change `crates/core/src/resource/device.rs` and `crates/core/tests/resource_serde.rs` to decision 3's variants, with COMP-E1-T01 depending on it. Re-run the plan tools.
8. **`plan/glossary.md`:** add "Device class", pointing to RFC-0040 §2.2.

## Open points not decided here

- **Integer dot on Vulkan.** RFC-0040 §7.5 lists integer dot as a shader-variant axis, but the Vulkan variant has no field for it. If integer-dot shaders need a different pack layout, decision 2 puts the axis in the class. The RFC-0040 owner confirms this before COMP-E8-T06.
- **A canonical text form.** `packs/<device-class>/` (ADR-0410) needs a stable string for each class. RFC-0041 does not define one.
- **Unknown variants.** `DeviceClass` crosses the network in `GetExtent` (RFC-0041 §8.5). The cross-version rule for unknown enum variants is issue #1499 item 4 and applies here too.

## What would make us revisit

- Two devices of one class need different pack layouts or kernel objects. Add the distinguishing field under decision 2.
- A field in the class turns out to change neither, and splits packs or certificates for no reason. Remove it.
- An NPU provider lands. Its `Npu` fields are then checked against decision 2.

## Log

- Proposed 2026-10-21 for issue #1499 item 1.
