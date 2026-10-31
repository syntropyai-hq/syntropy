// SPDX-License-Identifier: MIT OR Apache-2.0

//! `TierManager::plan_arenas` and the ledger check (TIER-E1-T02; RFC-0020 §2, §3, §12, §15).
//!
//! - Goldens: the `ArenaPlan`s of RFC-0011 §15's scenarios (a), (b2) and (c), from synthetic
//!   censuses sized to each scenario's [estimate] inputs. `SYNTROPY_BLESS_GOLDEN=1` rewrites
//!   them; read the diff before committing it.
//! - Properties over random plans: for every tier the arenas' bytes equal the plan's bytes
//!   (the module docs of `syntropy_tier::arena::plan` define them), every DMA source lands in
//!   a `DmaPinned` arena registered for the GPU that copies from it, and two calls agree byte
//!   for byte.
//! - The ledger check: a plan over its grant fails with `TIER_LEDGER_EXCEEDED`.
//! - Each refusal of inconsistent inputs.

mod fixture;

use std::collections::BTreeMap;
use std::path::PathBuf;

use fixture::*;
use proptest::prelude::*;
use syntropy_core::error::codes;
use syntropy_core::ledger::LedgerGrant;
use syntropy_core::plan::StageRole;
use syntropy_core::resource::TierKind;
use syntropy_core::{Bytes, ErrorClass, PlacementPlan, RetryHint, StageIx, TierId};
use syntropy_manifest::census::{TensorCensus, TensorClass};
use syntropy_tier::{
    ArenaPlan, ArenaPlanError, PinClass, Purpose, Registrations, SlotGeometry, TierConfig,
    TierManager,
};

// ---- Scenarios -------------------------------------------------------------------------

/// Llama-3.3-70B Q4_K_M as RFC-0011 §15.1 and §15.3 size it [estimate]: 80 layers of 0.505 GB
/// and a 1.05 GB embedding and LM head, 42.5 GB in all. Group 0 is the embedding, group
/// `1 + l` layer `l`, group 81 the head.
fn llama_70b() -> TensorCensus {
    let mut g = vec![(TensorClass::RowLookup, None, 1_050 * MB)];
    g.extend((0..80).map(|l| (TensorClass::DenseEveryToken, Some(l), 505 * MB)));
    g.push((TensorClass::Head, None, 1_050 * MB));
    census(g)
}

fn display_gpu() -> TierConfig {
    TierConfig {
        vram_headroom: Bytes(512 * MIB),
        ..TierConfig::default()
    }
}

/// Scenario (a), RFC-0011 §15.1: one RTX 4090 and 64 GB of DDR5. 18.8 GB of dense layers and
/// the head in VRAM; the rest of the dense layers (23.7 GB with the embedding) homed in
/// `HostPinned`, computed in place by the CPU and streamed to the GPU for prefill; the active
/// sequence's KV (2.68 GB) in VRAM and the queued one's reserved in `HostPinned`; 1.5 GB of
/// scratch.
fn scenario_a() -> (PlacementPlan, TensorCensus, TierConfig) {
    let mut p = plan(vec![stage(
        0,
        NODE,
        0..80,
        StageRole::HEAD | StageRole::TAIL,
    )]);
    p.tensors.push(assign(0, tier(PINNED), cpu(), false));
    for l in 0..80u32 {
        p.tensors.push(if l < 35 {
            assign(1 + l, tier(VRAM), gpu(), false)
        } else {
            assign(1 + l, tier(PINNED), cpu(), true)
        });
    }
    p.tensors.push(assign(81, tier(VRAM), gpu(), false));
    p.kv = vec![
        kv(0, tier(VRAM), 2_680 * MB, Some(tier(PINNED))),
        kv(0, tier(PINNED), 2_680 * MB, None),
    ];
    p.scratch = vec![(gpu(), Bytes(1_500 * MB))];
    (p, llama_70b(), display_gpu())
}

/// Scenario (b2), RFC-0011 §15.2: DeepSeek-V3 Q4_K_M on one RTX 4090 and 128 GB of DDR5, the
/// row with a pageable share (Windows' 64 GB pin cap). 61 layers, 58 of them MoE with 256
/// routed experts of 26.5 MB [estimate]. The 9.7 GB hot set (dense groups and head) is in
/// VRAM, the embedding in `HostPageable` for CPU lookups, and every routed expert is homed on
/// `Nvme`. Caches: 9.5 GB in VRAM, and a 104 GB RAM cache split into a 61 GB pinned part and
/// a 43 GB pageable part.
fn scenario_b2() -> (PlacementPlan, TensorCensus, TierConfig) {
    let mut g = vec![(TensorClass::RowLookup, None, 470 * MB)];
    g.extend((0..61).map(|l| (TensorClass::DenseEveryToken, Some(l), 150 * MB)));
    g.push((TensorClass::Head, None, 470 * MB));
    for l in 3..61 {
        g.extend((0..256).map(|e| (expert(l, e), Some(l), 26_500_000)));
    }
    let census = census(g);

    let mut p = plan(vec![stage(
        0,
        NODE,
        0..61,
        StageRole::HEAD | StageRole::TAIL,
    )]);
    p.tensors.push(assign(0, tier(PAGEABLE), cpu(), false));
    for i in 1..=62 {
        p.tensors.push(assign(i, tier(VRAM), gpu(), false));
    }
    for i in 63..u32::try_from(census.groups.len()).unwrap() {
        p.tensors.push(assign(i, tier(NVME), cpu(), false));
    }
    p.expert_cache = vec![
        cache(gpu(), tier(VRAM), 9_500 * MB),
        cache(cpu(), tier(PINNED), 61 * GB),
        cache(cpu(), tier(PAGEABLE), 43 * GB),
    ];
    p.kv = vec![
        kv(0, tier(VRAM), 2_250 * MB, Some(tier(PINNED))),
        kv(0, tier(PINNED), 2_250 * MB, None),
    ];
    p.scratch = vec![(gpu(), Bytes(1_500 * MB))];
    let cfg = TierConfig {
        expert_staging: Bytes(512 * MIB),
        expert_staging_host: Bytes(256 * MIB),
        ..display_gpu()
    };
    (p, census, cfg)
}

/// Scenario (c1), RFC-0011 §15.3, as the gaming PC's runner sees it: the chosen PC → Strix
/// plan. The PC's RTX 4090 holds stage 0, the embedding and 38 layers (20.2 GB), with the
/// active sequence's KV for its layers (1.27 GB) in VRAM and the queued one's in `HostPinned`.
/// Stage 1, layers 38–79 and the head, is on the Strix Halo node, whose tiers this runner
/// does not hold.
fn scenario_c() -> (PlacementPlan, TensorCensus, TierConfig) {
    let mut p = plan(vec![
        stage(0, NODE, 0..38, StageRole::HEAD),
        stage(1, OTHER, 38..80, StageRole::TAIL),
    ]);
    p.tensors.push(assign(0, tier(VRAM), gpu(), false));
    for l in 0..80u32 {
        p.tensors.push(if l < 38 {
            assign(1 + l, tier(VRAM), gpu(), false)
        } else {
            assign(1 + l, other_tier(0), gpu(), false)
        });
    }
    p.tensors.push(assign(81, other_tier(0), gpu(), false));
    p.kv = vec![
        kv(0, tier(VRAM), 1_270 * MB, Some(tier(PINNED))),
        kv(0, tier(PINNED), 1_270 * MB, None),
        kv(1, other_tier(0), 1_410 * MB, None),
    ];
    p.scratch = vec![
        (gpu(), Bytes(1_500 * MB)),
        (
            syntropy_core::DeviceId {
                node: OTHER,
                ordinal: 0,
            },
            Bytes(1_500 * MB),
        ),
    ];
    (p, llama_70b(), display_gpu())
}

fn run(p: &PlacementPlan, stages: &[u16], c: &TensorCensus, cfg: &TierConfig) -> ArenaPlan {
    let stages: Vec<StageIx> = stages.iter().map(|&s| StageIx(s)).collect();
    TierManager::plan_arenas(p, &stages, c, &profile(), cfg).expect("plan_arenas")
}

fn check_golden(name: &str, ap: &ArenaPlan) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(format!("arena_plan_{name}.json"));
    let got = serde_json::to_string_pretty(ap).unwrap() + "\n";
    if std::env::var("SYNTROPY_BLESS_GOLDEN").as_deref() == Ok("1") {
        std::fs::write(&path, &got).unwrap();
        return;
    }
    let want = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}; bless with SYNTROPY_BLESS_GOLDEN=1",
            path.display()
        )
    });
    assert!(
        got == want,
        "{} differs from plan_arenas' output:\n{got}",
        path.display()
    );
}

/// Reserved bytes per (tier, purpose) from the plan's budget lines.
fn reserved(ap: &ArenaPlan) -> BTreeMap<(TierId, Purpose), u64> {
    ap.per_tier
        .iter()
        .flat_map(|(t, line)| line.by_purpose.iter().map(move |(p, b)| ((*t, *p), b.0)))
        .collect()
}

#[test]
fn golden_scenario_a() {
    let (p, c, cfg) = scenario_a();
    let ap = run(&p, &[0], &c, &cfg);
    check_golden("a", &ap);

    // The golden's figures, by hand. VRAM: 35 layers and the head, each rounded up to 4 KiB;
    // KV; scratch. HostPinned: 45 streamed layers and the embedding; the queued KV; the load
    // staging ring, 4 slots of 64 MiB for the one GPU.
    let up = |b: u64| b.div_ceil(4096) * 4096;
    let r = reserved(&ap);
    assert_eq!(
        r[&(tier(VRAM), Purpose::Weights)],
        35 * up(505 * MB) + up(1_050 * MB)
    );
    assert_eq!(r[&(tier(VRAM), Purpose::Kv)], 2_680 * MB);
    assert_eq!(r[&(tier(VRAM), Purpose::Scratch)], 1_500 * MB);
    assert_eq!(
        r[&(tier(PINNED), Purpose::Weights)],
        45 * up(505 * MB) + up(1_050 * MB)
    );
    assert_eq!(r[&(tier(PINNED), Purpose::Kv)], 2_680 * MB);
    assert_eq!(r[&(tier(PINNED), Purpose::IoStaging)], 4 * 64 * MIB);
    assert_eq!(r.len(), 6, "{r:?}");
    let w = weights(&ap, PINNED);
    assert_eq!(w.pin, PinClass::DmaPinned);
    assert_eq!(w.dma_devices.as_slice(), &[gpu()]);
    assert_eq!(w.pin_priority, 2);
    assert_eq!(weights(&ap, VRAM).pin, PinClass::Locked);
    assert_eq!(ap.per_tier[0].1.headroom, Bytes(512 * MIB));
    assert!(ap.nvme_extents.is_empty());
}

#[test]
fn golden_scenario_b2() {
    let (p, c, cfg) = scenario_b2();
    let ap = run(&p, &[0], &c, &cfg);
    check_golden("b2", &ap);

    // RFC-0023 §4: with no profile, every layer gets the same slot count; 58 layers of 26.5 MB
    // slots in 9.5 GB give 6 each, as RFC-0023 §4 works out for (b2) [estimate], and the 10
    // slots the remainder holds go to the first 10 layers.
    let slot = 26_500_000u64.div_ceil(4096) * 4096;
    let vram = cache_arena(&ap, VRAM);
    let SlotGeometry::Classes(classes) = &vram.geometry else {
        panic!("{:?}", vram.geometry)
    };
    assert_eq!(classes.len(), 58);
    assert!(classes.iter().all(|c| c.size == Bytes(slot)));
    let counts: Vec<u32> = classes.iter().map(|c| c.count).collect();
    assert_eq!(counts[..10], [7; 10]);
    assert_eq!(counts[10..], [6; 48]);
    assert!(counts.iter().map(|&n| u64::from(n) * slot).sum::<u64>() <= 9_500 * MB);
    // 256 MiB is above 8 slots (212 MB), so the VRAM segment is `tier.vram_segment`.
    assert_eq!(vram.elastic.unwrap().segment, Bytes(256 * MIB));
    assert_eq!(vram.pin, PinClass::Locked);

    let pinned = cache_arena(&ap, PINNED);
    assert_eq!(pinned.pin, PinClass::DmaPinned);
    assert_eq!(
        pinned.registrations,
        Registrations::GPU_DMA | Registrations::IO_FIXED
    );
    assert_eq!(pinned.dma_devices.as_slice(), &[gpu()]);
    let pageable = cache_arena(&ap, PAGEABLE);
    assert_eq!(pageable.pin, PinClass::Locked);
    assert!(pageable.registrations.is_empty());
    assert!(pageable.dma_devices.is_empty());

    let r = reserved(&ap);
    assert_eq!(r[&(tier(VRAM), Purpose::ExpertStaging)], 512 * MIB);
    assert_eq!(r[&(tier(PINNED), Purpose::ExpertStaging)], 256 * MIB);
    assert_eq!(
        r[&(tier(PAGEABLE), Purpose::Weights)],
        (470 * MB).div_ceil(4096) * 4096
    );
    assert!(!r.keys().any(|(t, _)| t.ordinal == NVME), "{r:?}");
    assert!(ap.nvme_extents.is_empty());
}

#[test]
fn golden_scenario_c() {
    let (p, c, cfg) = scenario_c();
    let ap = run(&p, &[0], &c, &cfg);
    check_golden("c", &ap);

    // Only the PC's tiers, and only stage 0's groups and KV.
    let up = |b: u64| b.div_ceil(4096) * 4096;
    let r = reserved(&ap);
    assert!(ap.per_tier.iter().all(|(t, _)| t.node == NODE));
    assert_eq!(
        r[&(tier(VRAM), Purpose::Weights)],
        38 * up(505 * MB) + up(1_050 * MB)
    );
    assert_eq!(r[&(tier(VRAM), Purpose::Kv)], 1_270 * MB);
    assert_eq!(r[&(tier(PINNED), Purpose::Kv)], 1_270 * MB);
    assert!(!r.contains_key(&(tier(PINNED), Purpose::Weights)));
}

fn weights(ap: &ArenaPlan, ordinal: u16) -> &syntropy_tier::ArenaSpec {
    arena(ap, ordinal, Purpose::Weights)
}

fn cache_arena(ap: &ArenaPlan, ordinal: u16) -> &syntropy_tier::ArenaSpec {
    arena(ap, ordinal, Purpose::ExpertCache)
}

fn arena(ap: &ArenaPlan, ordinal: u16, purpose: Purpose) -> &syntropy_tier::ArenaSpec {
    ap.arenas
        .iter()
        .find(|a| a.tier == tier(ordinal) && a.purpose == purpose)
        .unwrap_or_else(|| panic!("no {purpose:?} arena on tier {ordinal}"))
}

// ---- The ledger check ------------------------------------------------------------------

fn exact_grant(ap: &ArenaPlan) -> LedgerGrant {
    LedgerGrant::new(ap.per_tier.iter().map(|(t, l)| (*t, l.reserved))).unwrap()
}

#[test]
fn ledger_at_the_grant_passes_and_one_byte_over_fails() {
    let (p, c, cfg) = scenario_a();
    let ap = run(&p, &[0], &c, &cfg);
    TierManager::check_ledger(&ap, &exact_grant(&ap)).unwrap();

    let short = LedgerGrant::new(ap.per_tier.iter().map(|(t, l)| {
        let b = if t.ordinal == PINNED {
            l.reserved.0 - 1
        } else {
            l.reserved.0
        };
        (*t, Bytes(b))
    }))
    .unwrap();
    let e = TierManager::check_ledger(&ap, &short).unwrap_err();
    assert_eq!(e.code, codes::tier::LEDGER_EXCEEDED);
    assert_eq!(e.class, ErrorClass::Capacity);
    assert_eq!(e.retry, RetryHint::OnNewPlan);
    assert_eq!(e.context("tier"), Some("1"));
    let line = &ap
        .per_tier
        .iter()
        .find(|(t, _)| t.ordinal == PINNED)
        .unwrap()
        .1;
    assert_eq!(
        e.context("reserved"),
        Some(line.reserved.0.to_string().as_str())
    );
    assert_eq!(
        e.context("granted"),
        Some((line.reserved.0 - 1).to_string().as_str())
    );
}

#[test]
fn ledger_without_a_line_for_a_tier_fails() {
    let (p, c, cfg) = scenario_a();
    let ap = run(&p, &[0], &c, &cfg);
    let missing = LedgerGrant::new(
        ap.per_tier
            .iter()
            .filter(|(t, _)| t.ordinal != VRAM)
            .map(|(t, l)| (*t, l.reserved)),
    )
    .unwrap();
    let e = TierManager::check_ledger(&ap, &missing).unwrap_err();
    assert_eq!(e.code, codes::tier::LEDGER_EXCEEDED);
    assert_eq!(e.context("tier"), Some("0"));
    assert_eq!(e.context("granted"), Some("0"));
}

#[test]
fn ledger_headroom_is_not_charged() {
    // The VRAM line has 512 MiB of headroom; a grant of exactly `reserved` still passes.
    let (p, c, cfg) = scenario_a();
    let ap = run(&p, &[0], &c, &cfg);
    assert!(ap.per_tier.iter().any(|(_, l)| l.headroom.0 > 0));
    TierManager::check_ledger(&ap, &exact_grant(&ap)).unwrap();
}

// ---- Refusals and ownership ------------------------------------------------------------

fn try_run(
    p: &PlacementPlan,
    stages: &[u16],
    c: &TensorCensus,
) -> Result<ArenaPlan, ArenaPlanError> {
    let stages: Vec<StageIx> = stages.iter().map(|&s| StageIx(s)).collect();
    TierManager::plan_arenas(p, &stages, c, &profile(), &TierConfig::default())
}

fn small() -> (PlacementPlan, TensorCensus) {
    let c = census(vec![
        (TensorClass::RowLookup, None, 10 * MB),
        (TensorClass::DenseEveryToken, Some(0), 20 * MB),
        (TensorClass::DenseEveryToken, Some(1), 20 * MB),
        (TensorClass::Head, None, 10 * MB),
    ]);
    let mut p = plan(vec![
        stage(0, NODE, 0..1, StageRole::HEAD),
        stage(1, NODE, 1..2, StageRole::TAIL),
    ]);
    p.tensors = vec![
        assign(0, tier(PAGEABLE), cpu(), false),
        assign(1, tier(PINNED), cpu(), true),
        assign(2, tier(VRAM), gpu(), false),
        assign(3, tier(VRAM), gpu(), false),
    ];
    (p, c)
}

#[test]
fn a_streamed_group_homed_on_host_pageable_is_refused() {
    let (mut p, c) = small();
    p.tensors[0] = assign(0, tier(PAGEABLE), cpu(), true);
    assert_eq!(
        try_run(&p, &[0, 1], &c),
        Err(ArenaPlanError::DmaFromPageable(
            syntropy_core::TensorGroupId(0)
        ))
    );
}

#[test]
fn stages_select_the_groups() {
    let (p, c) = small();
    // Stage 0 (HEAD) holds layer 0 and the embedding; stage 1 (TAIL) layer 1 and the head.
    let head = try_run(&p, &[0], &c).unwrap();
    let r = reserved(&head);
    let up = |b: u64| b.div_ceil(4096) * 4096;
    assert_eq!(r[&(tier(PAGEABLE), Purpose::Weights)], up(10 * MB));
    assert_eq!(r[&(tier(PINNED), Purpose::Weights)], up(20 * MB));
    assert!(!r.contains_key(&(tier(VRAM), Purpose::Weights)));
    let tail = try_run(&p, &[1], &c).unwrap();
    let r = reserved(&tail);
    let up = |b: u64| b.div_ceil(4096) * 4096;
    assert_eq!(
        r[&(tier(VRAM), Purpose::Weights)],
        up(20 * MB) + up(10 * MB)
    );
    assert!(!r.contains_key(&(tier(PINNED), Purpose::Weights)));
    assert!(!r.contains_key(&(tier(PAGEABLE), Purpose::Weights)));
}

#[test]
fn groups_homed_on_another_node_are_skipped() {
    let (mut p, c) = small();
    p.tensors[2] = assign(2, other_tier(VRAM), gpu(), false);
    p.tensors[3] = assign(3, other_tier(VRAM), gpu(), false);
    let ap = try_run(&p, &[0, 1], &c).unwrap();
    assert!(!reserved(&ap).contains_key(&(tier(VRAM), Purpose::Weights)));
}

#[test]
fn inconsistent_inputs_are_refused() {
    let (p, c) = small();
    let stages = [StageIx(0)];

    let mut caps = profile();
    caps.ext.inventory = None;
    assert_eq!(
        TierManager::plan_arenas(&p, &stages, &c, &caps, &TierConfig::default()),
        Err(ArenaPlanError::NoInventory)
    );

    let mut q = p.clone();
    q.tensors[0] = assign(0, tier(9), cpu(), false);
    assert_eq!(
        try_run(&q, &[0], &c),
        Err(ArenaPlanError::UnknownTier(tier(9)))
    );

    let mut q = p.clone();
    q.tensors.push(assign(7, tier(VRAM), gpu(), false));
    assert_eq!(
        try_run(&q, &[0], &c),
        Err(ArenaPlanError::UnknownGroup(syntropy_core::TensorGroupId(
            7
        )))
    );

    assert_eq!(
        try_run(&p, &[2], &c),
        Err(ArenaPlanError::UnknownStage(StageIx(2)))
    );

    let mut q = p.clone();
    q.kv = vec![kv(0, tier(PAGEABLE), MB, None)];
    assert_eq!(
        try_run(&q, &[0], &c),
        Err(ArenaPlanError::PurposeTier {
            purpose: Purpose::Kv,
            tier: tier(PAGEABLE),
            kind: TierKind::HostPageable,
        })
    );

    let mut q = p.clone();
    q.expert_cache = vec![cache(gpu(), tier(NVME), MB)];
    assert!(matches!(
        try_run(&q, &[0], &c),
        Err(ArenaPlanError::PurposeTier {
            purpose: Purpose::ExpertCache,
            ..
        })
    ));

    let mut q = p.clone();
    q.expert_cache = vec![cache(gpu(), tier(VRAM), MB)];
    assert_eq!(
        try_run(&q, &[0], &c),
        Err(ArenaPlanError::NoExperts(tier(VRAM)))
    );
}

#[test]
fn per_layer_budgets_set_the_slot_counts() {
    let c = census(vec![
        (expert(0, 0), Some(0), 4096),
        (expert(0, 1), Some(0), 4000),
        (expert(1, 0), Some(1), 8192),
    ]);
    let mut p = plan(vec![stage(
        0,
        NODE,
        0..2,
        StageRole::HEAD | StageRole::TAIL,
    )]);
    p.tensors = (0..3)
        .map(|g| assign(g, tier(PINNED), cpu(), false))
        .collect();
    let mut cb = cache(gpu(), tier(VRAM), 100_000);
    cb.per_layer = Some(vec![Bytes(3 * 4096), Bytes(5 * 8192)]);
    p.expert_cache = vec![cb.clone()];
    let ap = try_run(&p, &[0], &c).unwrap();
    let SlotGeometry::Classes(classes) = &cache_arena(&ap, VRAM).geometry else {
        panic!()
    };
    let got: Vec<(u64, u32)> = classes.iter().map(|c| (c.size.0, c.count)).collect();
    assert_eq!(got, [(4096, 3), (8192, 5)]);

    // Without `per_layer`: 100_000 / (4096 + 8192) = 8 each, and the 1_696 left buy nothing.
    p.expert_cache[0].per_layer = None;
    let ap = try_run(&p, &[0], &c).unwrap();
    let SlotGeometry::Classes(classes) = &cache_arena(&ap, VRAM).geometry else {
        panic!()
    };
    assert_eq!(classes.iter().map(|c| c.count).collect::<Vec<_>>(), [8, 8]);

    // One entry short, or more than the cache's bytes.
    p.expert_cache[0].per_layer = Some(vec![Bytes(4096)]);
    assert!(matches!(
        try_run(&p, &[0], &c),
        Err(ArenaPlanError::PerLayer { .. })
    ));
    p.expert_cache[0].per_layer = Some(vec![Bytes(50_000), Bytes(50_001)]);
    assert!(matches!(
        try_run(&p, &[0], &c),
        Err(ArenaPlanError::PerLayer { .. })
    ));

    // The experts are `HostPinned` and the VRAM cache copies from them.
    let w = weights(&ap, PINNED);
    assert_eq!(w.dma_devices.as_slice(), &[gpu()]);
    assert_eq!(w.registrations, Registrations::GPU_DMA);
}

// ---- Properties ------------------------------------------------------------------------

const DATA_PURPOSES: [Purpose; 4] = [
    Purpose::Weights,
    Purpose::ExpertCache,
    Purpose::Kv,
    Purpose::Scratch,
];

#[derive(Clone, Debug)]
struct Case {
    plan: PlacementPlan,
    census: TensorCensus,
    stages: Vec<StageIx>,
    cfg: TierConfig,
}

/// Plans over the fixture node with two stages (layers 0–3 HEAD, 4–7 TAIL), random groups,
/// homes, streaming, caches, KV budgets, scratch and staging sizes.
fn case() -> impl Strategy<Value = Case> {
    let group = (
        0u8..4,
        0u16..8,
        0u64..2_000_000_000,
        0u8..10,
        0u16..4,
        any::<bool>(),
    );
    let caches = prop::collection::vec((0u16..3, 0u64..20_000_000_000), 0..4);
    let kvs = prop::collection::vec((0u16..2, 0u16..2, 0u64..5_000_000_000, any::<bool>()), 0..4);
    (
        prop::collection::vec(group, 0..30),
        caches,
        kvs,
        (0u64..3_000_000_000, 0u64..1_000_000_000),
        1u8..4,
        (prop::bool::ANY, prop::bool::ANY, 0u32..5),
    )
        .prop_map(|(groups, caches, kvs, (gs, cs), sel, (xs, xh, io))| {
            let mut spec = Vec::new();
            let mut p = plan(vec![
                stage(0, NODE, 0..4, StageRole::HEAD),
                stage(1, NODE, 4..8, StageRole::TAIL),
            ]);
            for (i, (class, layer, bytes, node, home, stream)) in groups.into_iter().enumerate() {
                let (class, layer) = match class {
                    0 => (TensorClass::DenseEveryToken, Some(layer)),
                    1 => (expert(layer, u16::try_from(i).unwrap()), Some(layer)),
                    2 => (TensorClass::RowLookup, None),
                    _ => (TensorClass::Head, None),
                };
                spec.push((class, layer, bytes));
                let home = if node == 0 {
                    other_tier(home)
                } else {
                    tier(home)
                };
                let stream = stream && home == tier(PINNED);
                let decode = if home.ordinal == VRAM { gpu() } else { cpu() };
                p.tensors
                    .push(assign(u32::try_from(i).unwrap(), home, decode, stream));
            }
            for (t, bytes) in caches {
                let dev = if t == VRAM { gpu() } else { cpu() };
                p.expert_cache.push(cache(dev, tier(t), bytes));
            }
            for (s, t, bytes, spill) in kvs {
                p.kv.push(kv(s, tier(t), bytes, spill.then(|| tier(PINNED))));
            }
            p.scratch = vec![(gpu(), Bytes(gs)), (cpu(), Bytes(cs))];
            let stages = match sel {
                1 => vec![StageIx(0)],
                2 => vec![StageIx(1)],
                _ => vec![StageIx(0), StageIx(1)],
            };
            let cfg = TierConfig {
                expert_staging: Bytes(if xs { 512 * MIB } else { 0 }),
                expert_staging_host: Bytes(if xh { 256 * MIB } else { 0 }),
                io_stage_slots: io,
                ..TierConfig::default()
            };
            Case {
                plan: p,
                census: census(spec),
                stages,
                cfg,
            }
        })
}

/// Whether the runner of `stages` holds a group with this layer and class (the module docs).
fn owned(c: &Case, layer: Option<syntropy_core::LayerIx>, class: &TensorClass) -> bool {
    let st = |s: &StageIx| &c.plan.stages[usize::from(s.0)];
    match layer {
        Some(l) => c.stages.iter().any(|s| st(s).layers.contains(&l)),
        None if matches!(class, TensorClass::Head) => c
            .stages
            .iter()
            .any(|s| st(s).role.contains(StageRole::TAIL)),
        None => c
            .stages
            .iter()
            .any(|s| st(s).role.contains(StageRole::HEAD)),
    }
}

/// The plan's bytes per tier for the data purposes (module docs, "Plan bytes per tier").
fn plan_bytes(c: &Case) -> BTreeMap<TierId, u64> {
    let mut out: BTreeMap<TierId, u64> = BTreeMap::new();
    for t in &c.plan.tensors {
        let g = &c.census.groups[t.group.0 as usize];
        if t.home.node == NODE && t.home.ordinal != NVME && owned(c, g.layer, &g.class) {
            *out.entry(t.home).or_default() += g.bytes.0.div_ceil(4096) * 4096;
        }
    }
    for e in &c.plan.expert_cache {
        *out.entry(e.tier).or_default() += e.bytes.0;
    }
    for k in &c.plan.kv {
        if c.stages.contains(&k.stage) {
            *out.entry(k.tier).or_default() += k.bytes.0;
        }
    }
    *out.entry(tier(VRAM)).or_default() += c.plan.scratch[0].1.0;
    *out.entry(tier(PINNED)).or_default() += c.plan.scratch[1].1.0;
    out.retain(|_, b| *b > 0);
    out
}

/// The runner holds an expert group with bytes, so an expert cache can size its slots.
fn has_experts(c: &Case) -> bool {
    c.plan.tensors.iter().any(|t| {
        let g = &c.census.groups[t.group.0 as usize];
        t.home.node == NODE
            && matches!(g.class, TensorClass::Expert { .. })
            && g.bytes.0 > 0
            && owned(c, g.layer, &g.class)
    })
}

fn plan_case(c: &Case) -> Result<ArenaPlan, ArenaPlanError> {
    TierManager::plan_arenas(&c.plan, &c.stages, &c.census, &profile(), &c.cfg)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// Σ arena bytes = plan bytes, per tier: the data purposes' arenas sum to the plan's
    /// bytes on the tier, each budget line's `reserved` is the sum of its arenas, and staging
    /// arenas are exactly their slots.
    #[test]
    fn arena_bytes_equal_plan_bytes(c in case()) {
        let ap = match plan_case(&c) {
            Err(ArenaPlanError::NoExperts(_)) => {
                prop_assert!(!has_experts(&c));
                prop_assert!(c.plan.expert_cache.iter().any(|e| e.bytes.0 > 0));
                return Ok(());
            }
            r => r.unwrap(),
        };
        let mut data: BTreeMap<TierId, u64> = BTreeMap::new();
        let mut all: BTreeMap<TierId, u64> = BTreeMap::new();
        for a in &ap.arenas {
            prop_assert_eq!(a.bytes, a.max_bytes);
            *all.entry(a.tier).or_default() += a.max_bytes.0;
            if DATA_PURPOSES.contains(&a.purpose) {
                *data.entry(a.tier).or_default() += a.max_bytes.0;
            }
            match (&a.purpose, &a.geometry) {
                (Purpose::IoStaging, SlotGeometry::Ring { slot, count }) => {
                    prop_assert_eq!(a.max_bytes.0, slot.0 * u64::from(*count));
                }
                (Purpose::ExpertStaging, SlotGeometry::Classes(cl)) => {
                    let per = if a.tier.ordinal == VRAM { c.cfg.expert_staging } else { c.cfg.expert_staging_host };
                    let gpus = a.dma_devices.len().max(1) as u64;
                    prop_assert_eq!(a.max_bytes.0, per.0 * gpus);
                    prop_assert!(cl.iter().map(|s| s.size.0 * u64::from(s.count)).sum::<u64>() <= a.max_bytes.0);
                }
                (Purpose::ExpertCache, SlotGeometry::Classes(cl)) => {
                    prop_assert!(cl.iter().map(|s| s.size.0 * u64::from(s.count)).sum::<u64>() <= a.max_bytes.0);
                }
                (Purpose::Weights, SlotGeometry::Extents { .. })
                | (Purpose::Kv, SlotGeometry::Pages { .. })
                | (Purpose::Scratch, SlotGeometry::Bump) => {}
                other => prop_assert!(false, "unexpected arena {:?}", other),
            }
        }
        prop_assert_eq!(&data, &plan_bytes(&c));
        let lines: BTreeMap<TierId, u64> =
            ap.per_tier.iter().map(|(t, l)| (*t, l.reserved.0)).collect();
        prop_assert_eq!(&lines, &all);
        for (t, l) in &ap.per_tier {
            prop_assert_eq!(l.by_purpose.iter().map(|(_, b)| b.0).sum::<u64>(), l.reserved.0);
            prop_assert!(t.ordinal != NVME);
            let vram_headroom = if t.ordinal == VRAM { c.cfg.vram_headroom } else { Bytes(0) };
            prop_assert_eq!(l.headroom, vram_headroom);
        }
    }

    /// Every DMA source lands in a `DmaPinned` arena registered for the GPU that copies from
    /// it, `DmaPinned` means a `HostPinned` arena, and no `HostPageable` arena is registered.
    #[test]
    fn every_dma_source_is_dma_pinned(c in case()) {
        let Ok(ap) = plan_case(&c) else { return Ok(()) };
        let vram_cache = c.plan.expert_cache.iter().any(|e| e.tier == tier(VRAM));
        for t in &c.plan.tensors {
            let g = &c.census.groups[t.group.0 as usize];
            if t.home != tier(PINNED) || !owned(&c, g.layer, &g.class) {
                continue;
            }
            let copied = t.compute.batched.is_some()
                || (vram_cache && matches!(g.class, TensorClass::Expert { .. }));
            if copied {
                let w = weights(&ap, PINNED);
                prop_assert_eq!(w.pin, PinClass::DmaPinned);
                prop_assert!(w.registrations.contains(Registrations::GPU_DMA));
                prop_assert!(w.dma_devices.contains(&gpu()));
            }
        }
        let vram_kv: Vec<StageIx> = c.plan.kv.iter()
            .filter(|k| k.tier == tier(VRAM) && k.bytes.0 > 0 && c.stages.contains(&k.stage))
            .map(|k| k.stage)
            .collect();
        for a in &ap.arenas {
            let kind = [TierKind::Vram, TierKind::HostPinned, TierKind::HostPageable][usize::from(a.tier.ordinal)];
            prop_assert_eq!(a.pin == PinClass::DmaPinned, kind == TierKind::HostPinned, "{:?}", a);
            if a.registrations.contains(Registrations::GPU_DMA) {
                prop_assert!(!a.dma_devices.is_empty());
            } else {
                prop_assert!(a.dma_devices.is_empty(), "{:?}", a);
            }
            if kind != TierKind::HostPinned {
                prop_assert!(a.registrations.is_empty(), "{:?}", a);
            }
            if matches!(a.purpose, Purpose::IoStaging | Purpose::ExpertStaging) && kind == TierKind::HostPinned {
                prop_assert!(a.registrations.contains(Registrations::GPU_DMA));
            }
            if a.purpose == Purpose::Kv && kind == TierKind::HostPinned {
                prop_assert!(a.registrations.contains(Registrations::IO_FIXED));
                let shares_stage = c.plan.kv.iter().any(|k| {
                    k.tier == tier(PINNED) && c.stages.contains(&k.stage) && vram_kv.contains(&k.stage)
                });
                prop_assert_eq!(a.dma_devices.contains(&gpu()), shares_stage);
            }
        }
    }

    /// Two calls agree, and so do their serialised forms.
    #[test]
    fn plan_arenas_is_deterministic(c in case()) {
        let (a, b) = (plan_case(&c), plan_case(&c));
        prop_assert_eq!(&a, &b);
        if let (Ok(a), Ok(b)) = (a, b) {
            prop_assert_eq!(serde_json::to_vec(&a).unwrap(), serde_json::to_vec(&b).unwrap());
            // Sorted by (tier, purpose); two expert caches on one tier are two arenas.
            let keys: Vec<(TierId, Purpose)> = a.arenas.iter().map(|x| (x.tier, x.purpose)).collect();
            prop_assert!(keys.is_sorted());
        }
    }

    /// A grant of exactly the plan's bytes passes; one byte less on any tier fails with
    /// `TIER_LEDGER_EXCEEDED`.
    #[test]
    fn the_ledger_check_holds_at_the_grant(c in case(), pick in any::<prop::sample::Index>()) {
        let Ok(ap) = plan_case(&c) else { return Ok(()) };
        TierManager::check_ledger(&ap, &exact_grant(&ap)).unwrap();
        let used: Vec<usize> = (0..ap.per_tier.len()).filter(|&i| ap.per_tier[i].1.reserved.0 > 0).collect();
        if used.is_empty() {
            return Ok(());
        }
        let i = used[pick.index(used.len())];
        let grant = LedgerGrant::new(ap.per_tier.iter().enumerate().map(|(j, (t, l))| {
            (*t, Bytes(l.reserved.0 - u64::from(i == j)))
        })).unwrap();
        let e = TierManager::check_ledger(&ap, &grant).unwrap_err();
        prop_assert_eq!(e.code, codes::tier::LEDGER_EXCEEDED);
        prop_assert_eq!(e.class, ErrorClass::Capacity);
        let tier_ordinal = ap.per_tier[i].0.ordinal.to_string();
        prop_assert_eq!(e.context("tier"), Some(tier_ordinal.as_str()));
    }
}

#[test]
fn a_large_expert_sets_the_vram_segment() {
    // RFC-0020 §8.2: the segment is at least 8 slots. A 47.3 MiB expert [estimate, §8.2's
    // second example] gives 8 slots over 256 MiB; the provider's granularity rounds it up
    // further when the arena is built (TIER-E1-T07).
    let expert_bytes = 49_597_235;
    let c = census(vec![(expert(0, 0), Some(0), expert_bytes)]);
    let mut p = plan(vec![stage(
        0,
        NODE,
        0..1,
        StageRole::HEAD | StageRole::TAIL,
    )]);
    p.tensors = vec![assign(0, tier(PINNED), cpu(), false)];
    p.expert_cache = vec![cache(gpu(), tier(VRAM), 2 * GB)];
    let ap = try_run(&p, &[0], &c).unwrap();
    let slot = expert_bytes.div_ceil(4096) * 4096;
    let seg = cache_arena(&ap, VRAM).elastic.unwrap().segment;
    assert_eq!(seg, Bytes((8 * slot).div_ceil(4096) * 4096));
    assert!(seg.0 > 256 * MIB);
}

#[test]
fn host_expert_staging_only_serves_nvme_homes_and_the_pageable_cache() {
    // RAM-homed `DmaPinned` experts are copied to VRAM straight from their home, so no host
    // staging (RFC-0023 §3).
    let c = census(vec![(expert(0, 0), Some(0), 8 * MB)]);
    let mut p = plan(vec![stage(
        0,
        NODE,
        0..1,
        StageRole::HEAD | StageRole::TAIL,
    )]);
    p.tensors = vec![assign(0, tier(PINNED), cpu(), false)];
    p.expert_cache = vec![cache(gpu(), tier(VRAM), GB)];
    let cfg = TierConfig {
        expert_staging_host: Bytes(256 * MIB),
        ..TierConfig::default()
    };
    let has_host_staging = |p: &PlacementPlan| {
        TierManager::plan_arenas(p, &[StageIx(0)], &c, &profile(), &cfg)
            .unwrap()
            .arenas
            .iter()
            .any(|a| a.purpose == Purpose::ExpertStaging && a.tier == tier(PINNED))
    };
    assert!(!has_host_staging(&p));

    // A pageable part of the RAM cache stages through pinned slots.
    let mut q = p.clone();
    q.expert_cache.push(cache(cpu(), tier(PAGEABLE), GB));
    assert!(has_host_staging(&q));

    // So does an `Nvme` home.
    let mut q = p.clone();
    q.tensors = vec![assign(0, tier(NVME), cpu(), false)];
    assert!(has_host_staging(&q));
}
