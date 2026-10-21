# ADR-0405: One `DeviceClass` in `core`, with kernel-level variants, for placement, packs and certificates

- Status: Proposed
- Date: 2026-10-21
- Deciders: maintainers (drafted by an implementing agent)
- Related: ADR-0005, ADR-0014, ADR-0016, ADR-0019, ADR-0401, ADR-0402, ADR-0410, ADR-0602, ADR-0622; RFC-0001 §4.2, §5.2, §5.6, §5.13, §5.15; RFC-0010 §1.2, §2.1; RFC-0011 §3.3, §4; RFC-0032 §5.2; RFC-0040 §2.2, §6.2, §6.5, §7.2, §7.3, §7.5; RFC-0041 §1, §8.5; RFC-0060 §6.4; RFC-0061 §6.2

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
| CPU | `Cpu { arch: CpuArch, level: IsaLevel }`, six levels: x86-64-v2, v3, v4; aarch64 v8.0, v8.2-dotprod, v9-sve2 | `Cpu { isa: CpuIsa }`, eight levels (RFC-0040 §7.3): `X86V2`, `X86V3`, `X86V4`, `X86V4Amx`, `Arm80`, `Arm82Dot`, `Arm86I8mm`, `Arm92Sme2` | RFC-0040 adds AMX, I8MM and SME2 levels; RFC-0010 has an SVE2 level that RFC-0040 lacks |
| NPU | `Npu { vendor: u32, model: u32 }` | absent | RFC-0040 has no NPU provider in v1 |

Neither is a refinement of the other. The Metal and Vulkan fields are on different axes, and each CPU list has a level the other lacks. The two definitions give different keys for the same device, so they cannot both be the pack and certificate key.

Where the type is read, and what each reader needs it to separate:

| Consumer | Where | What the key must separate |
|---|---|---|
| Inventory | `DeviceInfo::class` (RFC-0001 §5.2); filled from `ProviderDevice::class` at probe step E5 (RFC-0010 §2.1) | whatever the provider reports |
| Placement | RFC-0011 §4 filter F1 (coverage per class) and §3.3 (the per-class workspace bound); RFC-0032 §5.2 router islands of one class | devices that run the same kernels |
| Pack key | `PackKey` (RFC-0001 §5.6, RFC-0041 §1); `packs/<device-class>/seg/` (ADR-0410); `GetExtent { class, key }` between nodes (RFC-0041 §8.5); ADR-0016 decision 7; a new class seeds a new pack (OQ-MOE-14) | devices that accept the same pack layout |
| Certificate key | `OpCert` (RFC-0060 §6.4, RFC-0001 §5.15); certified and derived levels (RFC-0040 §6.2); every certification lookup by device class, including those in RFC-0022, TIER-E6-T17 and PLC-E6-T14 | devices that run the same certified kernel objects |
| Selection cache | `SelectionKey::class` (RFC-0040 §6.5, ADR-0401 decision 6) | as the certificate key |
| Provider | `ProviderDevice::class` (RFC-0040 §2.2); `DeviceClass::Metal { family, metal4 }` (RFC-0040 §7.2); `CpuIsa` detection (RFC-0040 §7.3, COMP-E3-T01) | which library or function table the provider loads |

Every consumer wants the same thing: two devices with equal classes run the same kernels and accept the same packs. That is RFC-0010's own description of the type, so the conflict is in the variants, not in the purpose.

Facts that bear on the choice:

- RFC-0001 §4.2 rule 1: `core` depends on no workspace crate. `DeviceInfo` and `ConformanceManifest` are `core` types (RFC-0001 §5.2, §5.15) and both hold a `DeviceClass`, so the type, and every type inside it, must live in `core`. `kernels` depends on `core` and can re-export it.
- `core` already holds RFC-0010's variants: `crates/core/src/resource/device.rs`, merged in #1490 (PLC-E1-T01). `core` also holds `DeviceFeatures`, whose bits match RFC-0040 §2.2 one for one.
- Open PR #1520 (PLC-E1-T02) builds `DeviceClass::Cpu { arch, level: IsaLevel }` in `crates/probe` from `/proc/cpuinfo` and asserts it in its fixture goldens.
- COMP-E1-T01 (issue #98) lists `DeviceClass` and `DeviceFeatures` among the supporting types `syntropy-kernels` defines. COMP-E3-T01 detects the `CpuIsa` levels in `kernels-cpu`.
- The CPU kernel families follow ADR-0402: AMX and SME2 microkernels exist, SVE2 kernels do not in v1. RFC-0040 §7.3 selects the AMX tile layout (`cpu.amx16`) on `X86V4Amx` and SME2 kernels on `Arm92Sme2`, so those levels change the pack layout and the kernel objects. An SVE2 level selects nothing different.
- The Metal provider loads `metal4.metallib` or the Metal 3.2 library by OS version and family (RFC-0040 §7.2), so `metal4` changes the kernel objects. GPU core count changes no kernel and no layout.
- Vulkan shaders come in variants over subgroup size {32, 64}, cooperative matrix and integer dot (RFC-0040 §7.5). Many devices report a range of subgroup sizes under subgroup size control, not one size.
- Certificates are issued per class on the lab rigs (RFC-0061 §6.2). Derived certification, which widens a certificate to a class no rig covers, exists only for NVIDIA targets, with a first-load canary and the label `exact (derived certification)` (RFC-0060 §6.4).
- Nothing keyed by `DeviceClass` has been written yet: no pack, no released `ConformanceManifest`, no saved capability profile in the field. After packs and certificates exist, a change to the type rebuilds every pack of the affected classes (ADR-0622) and re-issues every `OpCert`.

## Options considered

| Option | For | Against |
|---|---|---|
| **A. One `DeviceClass` in `core` with RFC-0040's variants, plus RFC-0010's `Npu`; `CpuIsa` moves to `core`; `kernels` re-exports both** | One key everywhere; satisfies §4.2 rule 1; the variants are the ones that change kernels and layouts (ADR-0402, RFC-0040 §7); the lab covers few classes | Changes the `core` type merged in #1490 and the probe code in #1520; the GPU core count and the Vulkan device and driver IDs need new homes outside the class (decision 4); a Vulkan certificate covers more devices than the rig it ran on (see "Exactness on Vulkan") |
| B. One `DeviceClass` in `core` with RFC-0010's variants; `kernels` re-exports it | No `core` code change; Vulkan certificates stay per device and driver | Packs and certificates split by GPU core count, which changes no kernel; no AMX, I8MM or SME2 level, so the CPU provider cannot key its AMX and SME2 layouts; an SVE2 level with no kernels behind it; RFC-0040 §7.2, §7.3 and the COMP tasks change |
| C. Two types with a stated mapping: a hardware description in `core` (RFC-0010's variants, renamed, e.g. `DeviceModel`) and a kernel class for packs and certificates (RFC-0040's variants) | Keeps both sets of fields in typed form | `OpCert` lives in `core`, so both types end up in `core` anyway; a second field on `DeviceInfo` and an RFC-0001 §5.2 amendment; the mapping cannot be computed from the hardware fields alone, because `metal4` and `X86V4Amx` depend on the OS and on a runtime permission |
| D. `DeviceClass` owned by `kernels`; `DeviceInfo` and `OpCert` move out of `core` | Owner and crate agree | Breaks §4.2 rule 1 as written, or moves `ResourceGraph` and `ConformanceManifest` out of `core`: a large RFC-0001 §4 and §5 amendment for one enum |
| E. Leave both definitions | No edits | Two types named `DeviceClass` in two crates; the probe must convert between them; pack and certificate keys depend on which crate wrote them |

## Decision

**Option A** (proposed).

1. **One type.** `DeviceClass` is defined once and lives in `syntropy-core`. `syntropy-kernels` re-exports it (`pub use`) and never defines its own. `ProviderDevice::class`, `DeviceInfo::class`, `PackKey::class`, `OpCert::class` and `SelectionKey::class` all use it. `DeviceFeatures` is treated the same way.
2. **What a class separates.** A field belongs in `DeviceClass` only when a change in it can change the pack layout or the set of kernel objects the providers load for the device. Facts that change neither stay outside the class, in the fields decision 4 lists.
3. **Variants.** RFC-0040 §2.2's variants, with these adjustments:

   ```rust
   pub enum DeviceClass {
       Cuda { sm: u16 },
       Hip { gfx: u32 },
       Metal { family: AppleGpuFamily, metal4: bool },
       Vulkan { vendor: u32, coopmat: bool, subgroup: u8 },
       Cpu { isa: CpuIsa },
       Npu { vendor: u32, model: u32 },
   }
   pub enum CpuIsa { X86V2, X86V3, X86V4, X86V4Amx, Arm80, Arm82Dot, Arm86I8mm, Arm92Sme2 }
   ```

   - **`CpuIsa`** is a `core` type: the eight-level enum of RFC-0040 §7.3, with the same requirements per level. Detection stays in `kernels-cpu` (COMP-E3-T01). The CPU device's class in the inventory is the one the CPU provider reports at E5 (RFC-0010 §2.1), as for every other device. The probe's E1 host enumeration may compute a provisional value from `/proc/cpuinfo` and the platform equivalents, and E5 replaces it, because `X86V4Amx` also depends on the Linux AMX permission and on `features.disable`.
   - **`Vulkan::subgroup`** is the subgroup size of the shader variant set the provider selects at `open`, 32 or 64. It is never a value read from the device, so two devices that load the same variant set have the same class.
   - **`AppleGpuFamily`** keeps the name already in `core`; RFC-0040's `AppleFamily` is renamed.
   - **`Npu`** stays from RFC-0010, so an enumerated NPU with `DeviceProblem::NoProvider` still has a class. No provider produces it in v1, so no pack or certificate is ever keyed by it.
4. **What happens to RFC-0010's other types and fields.**
   - `IsaLevel` is deleted. `CpuIsa` replaces it.
   - `CpuArch` stays: `NodeInfo::arch` uses it (RFC-0001 §5.2).
   - `VkDriverId` stays and moves to `DriverInfo::vk_driver: Option<VkDriverId>`.
   - The Apple GPU core count, read from IOKit `gpu-core-count` (RFC-0010 §2.1), moves to a new `DeviceInfoExt::gpu_cores: Option<u16>`. Apple GPUs have no `PciInfo`, and one chip name ships with several core counts, so nothing else holds it.
   - The Vulkan vendor and device IDs move to a new `DeviceInfoExt::api_ids: Option<(u32, u32)>`, the IDs as the graphics API reports them. `PciInfo` holds them only for PCI devices, and many integrated and mobile Vulkan GPUs have none.

   These fields serve diagnostics and the Vulkan certificate rule below. They are not part of any pack key.
5. **Ownership.** RFC-0040 owns `DeviceClass`, `CpuIsa` and `DeviceFeatures`. RFC-0010 contributes `Npu`, the fields of decision 4, and the rule for merging provider records into one `DeviceInfo`.
6. **Pack directory name.** `packs/<device-class>/` (ADR-0410) uses one text form of the class. It is a pure, one-to-one function of the class value, and it never changes between releases for an unchanged value. When a node's class changes (decision 3 and Consequences), its new packs go in a new directory. The old directory is never renamed; the store reclaims it under the pack budget like any unused pack. RFC-0041 defines the form.
7. **Order of work.** The `core` type changes before any pack, certificate or saved profile is written. See "Interim behaviour" for COMP-E1-T01.

## Exactness on Vulkan (needs a maintainer decision)

This section records a choice that D3 and ADR-0019 make the maintainers' to settle, together with the RFC-0060 §6.4 owner. This ADR does not settle it.

**The problem.** Under option A, an `OpCert` for a Vulkan class is earned on one rig GPU with one driver. Read as today's rules read it, that certificate covers every GPU of the same `(vendor, coopmat, subgroup)` as a full certificate. But the object that runs on a Vulkan device is not the SPIR-V that was certified. Each device's driver compiles the SPIR-V into its own machine code at pipeline creation. A different GPU model or driver build can change floating-point contraction or accumulation order. Decision 2's test ("the set of kernel objects the providers load") sees only the SPIR-V. Option B kept `device` and `driver` in the class and so avoided this, at the cost of a class per GPU model and driver.

**The other backends**, checked one by one:

- **CUDA:** kernels are compiled offline to SASS per `sm`. A binary JIT-compiled from PTX is already covered by the derived rule (RFC-0060 §6.4 condition 2).
- **HIP:** code objects are compiled offline per `gfx` target, the class's own field.
- **CPU:** the same machine code runs on every CPU of one `CpuIsa`.
- **Metal:** the library is compiled offline per family, but the OS finishes compiling it to GPU code at pipeline creation. Within one family and one OS release this is one compiler; an OS update can change it. This is the same problem in a milder form, and the chosen rule should state whether it covers Metal too.

**Options:**

| Option | For | Against |
|---|---|---|
| V1. A Vulkan certificate covers other devices and drivers of its class only as a derived certificate: a first-load canary that must match the reference digest, and the label `exact (derived certification)`, as RFC-0060 §6.4 does for NVIDIA. `OpCert` records the rig's device and driver so a lookup can tell full from derived. | Keeps option A's few classes; every uncertified device or driver is checked on the user's machine before exact output; reuses an existing mechanism and label | Widens derived certification beyond NVIDIA; the canary covers the warm-up sequence only, as it does for NVIDIA today; `OpCert` gains a field |
| V2. `OpCert` carries the driver identity (device IDs, `VkDriverId`, driver version) beside the class; a certificate applies only on an exact match, and anything else is uncertified | Strict: no Vulkan device runs exact without a rig certificate for its device and driver | Almost every user's Vulkan device and driver differ from the lab's, so the native Vulkan provider would rarely be selectable in exact mode |
| V3. Vulkan keeps `device` and `driver` in its class, against decision 2 | Strict, and needs no certificate rule change | Packs and selection side records split per GPU model and driver, though the pack layout is the same; ADR-0016 decision 7 sharing mostly stops for Vulkan; the same rarity as V2 |

**Recommended default: V1.** It keeps exact mode sound on every device through the canary and the label, keeps option A's pack sharing, and uses the derived-certificate path the plan already has. If the maintainers find the canary too weak a guard for a driver change, V2 is the strict fallback. Until this is decided, the native Vulkan provider must not treat a Vulkan `OpCert` as a full certificate on a device or driver other than the rig's.

## Interim behaviour for COMP-E1-T01

COMP-E1-T01 is being implemented now (issue #98). It follows its current row: `syntropy-kernels` defines `kernels::DeviceClass` (and `CpuIsa`, `AppleFamily`) per RFC-0040 §2.2, and re-exports `DeviceFeatures` from `core`. Until this ADR is decided, that leaves option E's state: two types named `DeviceClass`, one in `core` and one in `kernels`. Nothing yet stores either in a pack or certificate, so the two do not meet.

The maintainers may instead rule that COMP-E1-T01 re-exports `core`'s `DeviceClass` now, whatever variants it has. That holds under options A, B and C, and avoids the second type. The PR for this ADR asks for that ruling.

**On acceptance**, `kernels::DeviceClass`, `kernels::CpuIsa` and `kernels::AppleFamily` are deleted and replaced by `pub use` of the `core` types from decision 3. `ProviderDevice`'s shape does not change. Code that matches on the variants keeps compiling except for the `AppleFamily` → `AppleGpuFamily` rename.

## Consequences

- Positive: one key for packs, certificates, selection and placement. A pack copied from a peer of the same class (ADR-0016 decision 7) is one that the receiving node's kernels accept.
- Positive: the lab certifies per kernel-relevant class, not per GPU SKU. On Vulkan, this rests on the rule chosen in "Exactness on Vulkan"; under V1, devices other than the rig's run exact output only as derived certification.
- Positive: `kernels` stays a thin layer over `core` types, as RFC-0001 §4.1 describes it.
- Negative: the `core` type and its tests from #1490, and the probe code and fixtures from #1520, change.
- Negative: the class depends on the OS, on runtime permissions and on configuration. A macOS upgrade that enables Metal 4, the Linux AMX permission request, `features.disable = ["amx"]` or `["metal4"]` each change the class. The node then builds new packs in a new directory and looks up different certificates. This is intended, because the kernels and the layout change too.
- Negative: two cluster nodes with identical hardware but different settings or OS releases have different classes, so they do not share packs under ADR-0016 decision 7. Each builds its own.
- Negative: a Vulkan device's class no longer names the GPU model. Diagnostics read it from `PciInfo`, `DeviceInfoExt::api_ids` and `name`.

## Edits that follow acceptance

None of these is made by this ADR. They follow in one change after a maintainer accepts it.

1. **RFC-0001 §5.13:** add a row: `DeviceClass`, `CpuIsa`, `DeviceFeatures` | core | 0040 | 0010 (`Npu`, decision 4's fields, the provider-record merge).
2. **RFC-0001 §5.2:** on `DeviceInfo::class`, cite "defined in RFC-0040 §2.2; ADR-0405".
3. **RFC-0001 §18:** log the amendment with the date and this ADR.
4. **RFC-0010 §1.2:** replace the `DeviceClass` block with a pointer to RFC-0040 §2.2 and ADR-0405; keep a note that `Npu` is RFC-0010's addition; add `DeviceInfoExt::gpu_cores`, `DeviceInfoExt::api_ids` and `DriverInfo::vk_driver`. **RFC-0010 §2.1:** point the Apple GPU row's `gpu-core-count` at `gpu_cores`, and say that E5's provider value is the inventory's class.
5. **RFC-0040 §2.2:** state that `DeviceClass`, `CpuIsa` and `DeviceFeatures` live in `syntropy-core` and that `kernels` re-exports them; rename `AppleFamily` to `AppleGpuFamily`; add `Npu`; add decision 2's rule and decision 3's `subgroup` rule to the doc comment. **RFC-0040 §7.3:** "`CpuIsa` is `core`'s type; this provider detects it". **RFC-0040 §6.2** and **RFC-0060 §6.4:** the Vulkan certificate rule the maintainers choose.
6. **RFC-0041:** define the text form of decision 6 where ADR-0410's `packs/<device-class>/` is described.
7. **COMP.md:** COMP-E1-T01's "`DeviceClass`, `DeviceFeatures`" becomes "re-exports of `core`'s `DeviceClass`, `CpuIsa` and `DeviceFeatures`", and its Implements cell adds ADR-0405. COMP-E3-T01 "detects `core`'s `CpuIsa`".
8. **A new task** in `PLC.md`, at the next free number in E1 (PLC-E1-T14 today), with the Implements cell "RFC-0040 §2.2; RFC-0010 §1.2; ADR-0405". It changes `crates/core/src/resource/device.rs` and `crates/core/tests/resource_serde.rs` to decisions 3 and 4. It also changes `crates/probe` (`linux/isa.rs`, `linux/mod.rs`, `tests/linux_fixtures.rs` and the fixture goldens) from `IsaLevel` to `CpuIsa`, if #1520 has merged by then. If #1520 has not merged, PLC-E1-T02 depends on the new task instead and makes the change itself. COMP-E1-T01 depends on the new task.
9. **`plan/implementation/dependency-fixes.md`:** a dated row for the new task and its edges from COMP-E1-T01 (and PLC-E1-T02 if it applies), rule R9. Then re-run the plan tools.
10. **`plan/glossary.md`:** add "Device class", pointing to RFC-0040 §2.2.

## Open points not decided here

- **Integer dot on Vulkan.** RFC-0040 §7.5 lists integer dot as a shader-variant axis, but the Vulkan variant has no field for it. If integer-dot shaders need a different pack layout, decision 2 puts the axis in the class. The RFC-0040 owner confirms this, and decision 3's `subgroup` rule, before COMP-E8-T06.
- **Unknown variants.** `DeviceClass` crosses the network in `GetExtent` (RFC-0041 §8.5). The cross-version rule for unknown enum variants is issue #1499 item 4 and applies here too.

## What would make us revisit

- Two devices of one class need different pack layouts or kernel objects. Add the distinguishing field under decision 2.
- A field in the class turns out to change neither, and splits packs or certificates for no reason. Remove it.
- Vulkan canaries (under V1) catch a driver-caused mismatch on a user's machine. Reconsider V2 for that vendor.
- An NPU provider lands. Its `Npu` fields are then checked against decision 2.

## Revisions

- **Proposed 2026-10-21** for issue #1499 item 1. It replaces the open-question row that item asked for: the question is this ADR's decision.
- **Revised 2026-10-21** after two independent reviews of the draft: `CpuIsa` moves to `core`; the fate of `IsaLevel`, `CpuArch` and `VkDriverId` is stated; the dropped fields get homes; `Vulkan::subgroup` gets a rule; the Vulkan exactness choice is recorded for the maintainers; the pack directory name is fixed as stable; the interim for COMP-E1-T01 and the full edit list are added.
