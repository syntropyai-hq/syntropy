// SPDX-License-Identifier: MIT OR Apache-2.0

//! `KvStore`: one stage's page classes, their `Vram` pools, the sequence slots' block tables and
//! the KV byte counters (RFC-0022 §1, §3, §4; RFC-0001 §5.9; TIER-E6-T02).
//!
//! Everything is sized in [`KvStore::new`]: the pools, a block table per sequence slot at its
//! maximum context, the free-slot stack and each pool's retire list. `open`, `append`,
//! `truncate`, `fork`, `release`, [`note_read`](KvStore::note_read), the step
//! [`fence`](KvStore::fence) and the counter [`flush`](KvStore::flush) allocate nothing.

#![forbid(unsafe_code)]

use std::ops::Range;

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use syntropy_core::plan::KvDType;
use syntropy_core::{Bytes, LayerIx, PrefixKey, SequenceId, StageIx};
use syntropy_manifest::kv::{KvFormula, LayerKv};

use super::KvError;
use super::pool::Pool;
use super::types::{
    ClassTable, KvHandle, KvHeadroom, KvResidency, KvWriteSlots, MoveTicket, PageClass, PageCopy,
    PageId, PageRef, PageTable, PageTableId,
};
use crate::experts::StatsProducer;

/// The alignment of each layer's array in the `Kv` arena.
pub const LAYER_ALIGN: u64 = 256;

/// One page class of a store.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassSpec {
    /// The class.
    pub class: PageClass,
    /// The stage's layers in the class, in layer order.
    pub layers: Vec<LayerIx>,
    /// Bytes of one page in each layer's array, in `layers` order: `P` positions, or for `State`
    /// the layer's whole state.
    pub layer_page_bytes: Vec<Bytes>,
    /// Bytes one position adds over every layer of the class; for `State`, the bytes of one
    /// sequence's state over every layer.
    pub unit_bytes: Bytes,
    /// Pages in the class's `Vram` pool.
    pub capacity: u32,
}

/// What a [`KvStore`] is built from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvStoreConfig {
    /// The stage.
    pub stage: StageIx,
    /// The stage's layers.
    pub layers: Range<LayerIx>,
    /// The `Vram` tier's node-local ordinal.
    pub tier: u8,
    /// Positions per page, `P`: a power of two from 16 to 256 (ADR-0205 item 1).
    pub page_tokens: u32,
    /// Sequence slots: the plan's maximum live sequences.
    pub max_sequences: u32,
    /// The longest context a slot holds, in positions.
    pub max_context: u32,
    /// The storage dtype.
    pub dtype: KvDType,
    /// The classes, in the order block tables list them.
    pub classes: Vec<ClassSpec>,
}

impl KvStoreConfig {
    /// The classes of `formula`'s `layers` (RFC-0022 §3): `Full` for `PerToken { window: None }`
    /// and `Sparse` layers, one `Windowed` class per distinct window, and `State` for `Fixed`
    /// layers. `Windowed` and `State` pools are sized at their worst case for `max_sequences`
    /// (RFC-0022 §11): a window ring of `⌈w / P⌉ + 1` pages, and one state page, per sequence.
    /// The `Full` pool holds `full_pages`; sizing it from `KvBudget` is TIER-E6-T11's.
    ///
    /// # Errors
    ///
    /// [`KvError::Unsupported`] for `Compressed` layers or `Aux` caches (TIER-E6-T16,
    /// TIER-E6-T08), or a page size that is not a power of two from 16 to 256.
    #[expect(clippy::too_many_arguments, reason = "one call per stage at Loading")]
    pub fn from_formula(
        formula: &KvFormula,
        stage: StageIx,
        layers: Range<LayerIx>,
        dtype: KvDType,
        tier: u8,
        page_tokens: u32,
        max_sequences: u32,
        max_context: u32,
        full_pages: u32,
    ) -> Result<Self, KvError> {
        if !(16..=256).contains(&page_tokens) || !page_tokens.is_power_of_two() {
            return Err(KvError::Unsupported(
                "kv.page_tokens must be 16, 32, 64, 128 or 256",
            ));
        }
        if formula
            .aux
            .iter()
            .any(|a| (layers.start..layers.end).contains(&a.layer))
        {
            return Err(KvError::Unsupported("the Aux class comes with TIER-E6-T08"));
        }
        let mut classes: Vec<ClassSpec> = Vec::new();
        for l in layers.start.0..layers.end.0 {
            let layer = LayerIx(l);
            let Some(kv) = formula.layers.get(usize::from(l)) else {
                break;
            };
            let class = match *kv {
                LayerKv::PerToken { window: None, .. } | LayerKv::Sparse { .. } => PageClass::Full,
                LayerKv::PerToken {
                    window: Some(w), ..
                } => PageClass::Windowed { window: w },
                LayerKv::Fixed { .. } => PageClass::State,
                _ => {
                    return Err(KvError::Unsupported(
                        "the Compressed class comes with TIER-E6-T16",
                    ));
                }
            };
            let one = layer..LayerIx(l + 1);
            let (page, unit) = if class == PageClass::State {
                let state = formula.stored(one, 0, dtype);
                (state, state)
            } else {
                // A window caps stored tokens, so size one position and one page uncapped.
                let unit = formula.stored(one.clone(), 1, dtype);
                (Bytes(unit.0 * u64::from(page_tokens)), unit)
            };
            let spec = match classes.iter_mut().find(|c| c.class == class) {
                Some(c) => c,
                None => {
                    let capacity = match class {
                        PageClass::Full => full_pages,
                        PageClass::Windowed { window } => {
                            max_sequences.saturating_mul(ring_pages(window, page_tokens))
                        }
                        _ => max_sequences,
                    };
                    classes.push(ClassSpec {
                        class,
                        layers: Vec::new(),
                        layer_page_bytes: Vec::new(),
                        unit_bytes: Bytes(0),
                        capacity,
                    });
                    classes.last_mut().expect("just pushed")
                }
            };
            spec.layers.push(layer);
            spec.layer_page_bytes.push(page);
            spec.unit_bytes.0 += unit.0;
        }
        Ok(KvStoreConfig {
            stage,
            layers,
            tier,
            page_tokens,
            max_sequences,
            max_context,
            dtype,
            classes,
        })
    }
}

/// The pages of a window ring: `⌈w / P⌉ + 1` (RFC-0022 §3).
#[must_use]
pub fn ring_pages(window: u32, page_tokens: u32) -> u32 {
    window.div_ceil(page_tokens) + 1
}

/// The KV bytes one class read and wrote since the last [`KvStore::flush`] (RFC-0022 §5.4's
/// `B_vram`, and the appended bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvBytes {
    /// The class.
    pub class: PageClass,
    /// The tier's node-local ordinal.
    pub tier: u8,
    /// Bytes read by steps.
    pub read: u64,
    /// Bytes written by appends.
    pub written: u64,
}

/// A stats-ring record of one class's KV bytes over one step (RFC-0022 §5.4; TIER-E6-T02),
/// which [`StatsProducer::push_kv_bytes`] carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvBytesRecord {
    /// The step.
    pub step: u64,
    /// The stage.
    pub stage: StageIx,
    /// The class.
    pub class: PageClass,
    /// The tier's node-local ordinal.
    pub tier: u8,
    /// Bytes read.
    pub read: u64,
    /// Bytes written.
    pub written: u64,
}

/// A slot in a window ring with no page: its positions were truncated away.
const VACANT: PageRef = PageRef {
    page: PageId {
        tier: u8::MAX,
        index: u32::MAX,
    },
    shared: false,
};

fn vacant(r: &PageRef) -> bool {
    r.page == VACANT.page
}

/// One stage's KV store (RFC-0022 §1, §4). The stage's primary dispatcher is its only writer.
#[derive(Debug)]
pub struct KvStore {
    cfg: KvStoreConfig,
    pools: Vec<Pool>,
    /// The block tables, one per sequence slot.
    tables: Vec<PageTable>,
    in_use: Vec<bool>,
    free_slots: Vec<u32>,
    counters: Vec<KvBytes>,
    /// Per slot and class, the token page each window-ring slot holds ([`NO_PAGE`] when it holds
    /// none): a ring slot reused for a newer token page no longer holds the older one, even after
    /// a cut vacates it. Empty for other classes.
    tags: Vec<Vec<Vec<u32>>>,
}

/// A window-ring slot that holds no token page.
const NO_PAGE: u32 = u32::MAX;

impl KvStore {
    /// A store for `cfg`, with every pool, block table and list sized for it.
    #[must_use]
    pub fn new(cfg: KvStoreConfig) -> Self {
        let p = cfg.page_tokens;
        let pools = cfg
            .classes
            .iter()
            .map(|c| Pool::new(c.class, cfg.tier, c.capacity))
            .collect();
        // Each slot is built on its own: cloning a `Vec` keeps its length, not its capacity.
        let table = || PageTable {
            classes: cfg
                .classes
                .iter()
                .map(|c| {
                    let full = cfg.max_context.div_ceil(p);
                    let cap = match c.class {
                        PageClass::Windowed { window } => ring_pages(window, p).min(full.max(1)),
                        PageClass::State => 1,
                        _ => full,
                    };
                    ClassTable {
                        class: c.class,
                        pages: Vec::with_capacity(cap as usize),
                        provisional: 0,
                    }
                })
                .collect(),
        };
        let n = cfg.max_sequences;
        let counters = cfg
            .classes
            .iter()
            .map(|c| KvBytes {
                class: c.class,
                tier: cfg.tier,
                read: 0,
                written: 0,
            })
            .collect();
        let tags = || {
            cfg.classes
                .iter()
                .map(|c| match c.class {
                    PageClass::Windowed { window } => {
                        Vec::with_capacity(ring_pages(window, p) as usize)
                    }
                    _ => Vec::new(),
                })
                .collect::<Vec<_>>()
        };
        KvStore {
            pools,
            tags: (0..n).map(|_| tags()).collect(),
            tables: (0..n).map(|_| table()).collect(),
            in_use: vec![false; n as usize],
            free_slots: (0..n).rev().collect(),
            counters,
            cfg,
        }
    }

    /// The configuration.
    #[must_use]
    pub fn config(&self) -> &KvStoreConfig {
        &self.cfg
    }

    /// The block tables of slot `id`.
    #[must_use]
    pub fn table(&self, id: PageTableId) -> &PageTable {
        &self.tables[id.0 as usize]
    }

    /// The bytes of the `Kv` arena the pools need: per class, per layer, `capacity` pages, each
    /// layer's array aligned to [`LAYER_ALIGN`] (RFC-0022 §3, layer-first).
    #[must_use]
    pub fn arena_bytes(&self) -> Bytes {
        let mut end = 0u64;
        for c in &self.cfg.classes {
            for b in &c.layer_page_bytes {
                end = end.next_multiple_of(LAYER_ALIGN) + b.0 * u64::from(c.capacity);
            }
        }
        Bytes(end)
    }

    /// The byte offset in the `Kv` arena of `page` in layer `layer`'s array of `class`: the
    /// layer-first layout of RFC-0022 §3, in [`arena_bytes`](KvStore::arena_bytes)'s order.
    #[must_use]
    pub fn page_offset(&self, class: PageClass, layer: LayerIx, page: PageId) -> Option<Bytes> {
        let mut end = 0u64;
        for c in &self.cfg.classes {
            for (l, b) in c.layers.iter().zip(&c.layer_page_bytes) {
                let base = end.next_multiple_of(LAYER_ALIGN);
                if c.class == class && *l == layer {
                    return (page.index < c.capacity && page.tier == self.cfg.tier)
                        .then(|| Bytes(base + b.0 * u64::from(page.index)));
                }
                end = base + b.0 * u64::from(c.capacity);
            }
        }
        None
    }

    fn class_ix(&self, class: PageClass) -> Option<usize> {
        self.cfg.classes.iter().position(|c| c.class == class)
    }

    /// The state of `page` in `class`'s pool.
    #[must_use]
    pub fn page_state(&self, class: PageClass, page: PageId) -> Option<super::PageState> {
        let ix = self.class_ix(class)?;
        (page.index < self.pools[ix].capacity()).then(|| self.pools[ix].state(page))
    }

    /// References held on `page` in `class`'s pool.
    #[must_use]
    pub fn page_refs(&self, class: PageClass, page: PageId) -> Option<u16> {
        let ix = self.class_ix(class)?;
        (page.index < self.pools[ix].capacity()).then(|| self.pools[ix].refs(page))
    }

    /// Free pages in `class`'s pool.
    #[must_use]
    pub fn free_pages(&self, class: PageClass) -> Option<u32> {
        self.class_ix(class).map(|ix| self.pools[ix].free_pages())
    }

    /// Pages in `class`'s pool that are held (not free and not waiting for the fence).
    #[must_use]
    pub fn live_pages(&self, class: PageClass) -> Option<u32> {
        self.class_ix(class).map(|ix| self.pools[ix].live())
    }

    fn slot(&self, h: &KvHandle) -> Result<usize, KvError> {
        let s = h.pages.0 as usize;
        if s < self.in_use.len() && self.in_use[s] {
            Ok(s)
        } else {
            Err(KvError::BadTable { table: h.pages })
        }
    }

    fn residency(&self, slot: usize) -> KvResidency {
        let vram = self.tables[slot]
            .classes
            .iter()
            .map(|t| t.pages.iter().filter(|r| !vacant(r)).count() as u32)
            .sum();
        KvResidency {
            vram,
            ..KvResidency::default()
        }
    }

    /// Opens a sequence on this stage (RFC-0001 §5.9): a free slot, and its `State` page.
    ///
    /// # Errors
    ///
    /// [`KvError::NoSlot`] when every slot is in use; [`KvError::PoolFull`] when the `State`
    /// pool is empty.
    pub fn open(
        &mut self,
        seq: SequenceId,
        stage: StageIx,
        prefix: Option<PrefixKey>,
    ) -> Result<KvHandle, KvError> {
        if stage != self.cfg.stage {
            return Err(KvError::WrongStage { stage });
        }
        let Some(&slot) = self.free_slots.last() else {
            return Err(KvError::NoSlot);
        };
        for (ix, c) in self.cfg.classes.iter().enumerate() {
            if c.class == PageClass::State && self.pools[ix].free_pages() == 0 {
                return Err(KvError::PoolFull {
                    class: c.class,
                    tier: self.cfg.tier,
                });
            }
        }
        self.free_slots.pop();
        let s = slot as usize;
        self.in_use[s] = true;
        for (ix, c) in self.cfg.classes.iter().enumerate() {
            if c.class == PageClass::State {
                let page = self.pools[ix].alloc().expect("checked above");
                self.tables[s].classes[ix].pages.push(PageRef {
                    page,
                    shared: false,
                });
            }
        }
        Ok(KvHandle {
            seq,
            stage,
            layers: self.cfg.layers.clone(),
            tokens: 0,
            pages: PageTableId(slot),
            residency: self.residency(s),
            prefix,
        })
    }

    /// The new pages `append` of `positions` takes from class `ix`'s pool. Every page it would
    /// write that the slot already holds is checked first, so `append` fails before it changes
    /// anything.
    fn pages_needed(&self, slot: usize, ix: usize, positions: &Range<u32>) -> Result<u32, KvError> {
        let p = self.cfg.page_tokens;
        let table = &self.tables[slot].classes[ix];
        let pool = &self.pools[ix];
        // A held page to write: a new one when shared, none when exclusive; never a moving one.
        let held = |r: &PageRef| -> Result<u32, KvError> {
            if let super::PageState::Moving { .. } = pool.state(r.page) {
                return Err(KvError::Moving { page: r.page });
            }
            Ok(u32::from(pool.refs(r.page) > 1))
        };
        let mut n = 0;
        let ring = match table.class {
            PageClass::State => {
                return table.pages.first().map_or(Ok(0), held);
            }
            PageClass::Windowed { window } => Some(ring_pages(window, p) as usize),
            _ => None,
        };
        for t in positions.start / p..positions.end.div_ceil(p) {
            let i = ring.map_or(t as usize, |ring| t as usize % ring);
            match table.pages.get(i) {
                None => n += 1,
                Some(r) if vacant(r) => n += 1,
                Some(r) => n += held(r)?,
            }
        }
        Ok(n)
    }

    /// Makes the page at `table.pages[i]` writable: a copy for a shared page, a reopen for an
    /// exclusive sealed one. Returns the copy to issue, if any.
    fn make_writable(
        pool: &mut Pool,
        class: PageClass,
        r: &mut PageRef,
        keep_tokens: u32,
    ) -> Result<Option<PageCopy>, KvError> {
        if pool.refs(r.page) > 1 {
            let to = pool.alloc()?;
            let from = r.page;
            pool.release(from)?;
            *r = PageRef {
                page: to,
                shared: false,
            };
            Ok(
                (keep_tokens > 0 || class == PageClass::State).then_some(PageCopy {
                    class,
                    from,
                    to,
                    tokens: keep_tokens,
                }),
            )
        } else {
            pool.reopen(r.page)?;
            r.shared = false;
            Ok(None)
        }
    }

    /// Appends `positions`, which must start at `h.tokens` (RFC-0001 §5.9). Every page the
    /// positions land on is made writable: a fresh page, a reopened exclusive one, or a copy of
    /// a shared one, which the returned slots list. A page that fills is sealed.
    ///
    /// # Errors
    ///
    /// - [`KvError::BadPosition`] when `positions` does not start at `h.tokens`, runs past the
    ///   slot's maximum context, or, for a window ring, is longer than the ring can hold beside
    ///   its window.
    /// - [`KvError::PoolFull`] when a class has too few free pages; nothing is changed.
    pub fn append(
        &mut self,
        h: &mut KvHandle,
        positions: Range<u32>,
    ) -> Result<KvWriteSlots, KvError> {
        let slot = self.slot(h)?;
        let p = self.cfg.page_tokens;
        let bad = KvError::BadPosition {
            tokens: h.tokens,
            at: positions.start,
        };
        if positions.start != h.tokens || positions.end > self.cfg.max_context {
            return Err(bad);
        }
        let mut out = KvWriteSlots {
            positions: positions.clone(),
            table: h.pages,
            copies: SmallVec::new(),
        };
        if positions.is_empty() {
            return Ok(out);
        }
        for (ix, c) in self.cfg.classes.iter().enumerate() {
            if let PageClass::Windowed { window } = c.class {
                let ring = ring_pages(window, p);
                // The positions a chunk writes must not push its own first page out of the ring.
                if positions.end - positions.start > (ring - 1) * p - (positions.start % p) {
                    return Err(bad);
                }
            }
            let need = self.pages_needed(slot, ix, &positions)?;
            if need > self.pools[ix].free_pages() {
                return Err(KvError::PoolFull {
                    class: c.class,
                    tier: self.cfg.tier,
                });
            }
        }
        for ix in 0..self.cfg.classes.len() {
            let class = self.cfg.classes[ix].class;
            let unit = self.cfg.classes[ix].unit_bytes.0;
            let pool = &mut self.pools[ix];
            let table = &mut self.tables[slot].classes[ix];
            match class {
                PageClass::State => {
                    let r = &mut table.pages[0];
                    if let Some(c) = Self::make_writable(pool, class, r, 0)? {
                        out.copies.push(c);
                    }
                    pool.check_write(r.page)?;
                    self.counters[ix].written += unit;
                }
                _ => {
                    let ring = match class {
                        PageClass::Windowed { window } => Some(ring_pages(window, p) as usize),
                        _ => None,
                    };
                    for t in positions.start / p..positions.end.div_ceil(p) {
                        let page_start = t * p;
                        let page_end = page_start + p;
                        let i = ring.map_or(t as usize, |ring| t as usize % ring);
                        // Positions of this token page already in it, which a copy keeps.
                        let kept = positions.start.max(page_start) - page_start;
                        if ring.is_some() {
                            let tags = &mut self.tags[slot][ix];
                            if i == tags.len() {
                                tags.push(t);
                            } else {
                                tags[i] = t;
                            }
                        }
                        if i == table.pages.len() {
                            let page = pool.alloc()?;
                            table.pages.push(PageRef {
                                page,
                                shared: false,
                            });
                        } else if vacant(&table.pages[i]) {
                            table.pages[i] = PageRef {
                                page: pool.alloc()?,
                                shared: false,
                            };
                        } else {
                            // The sequence's tail page (`kept > 0`), or a ring slot whose
                            // positions left the window (`kept == 0`), whose bytes are not kept.
                            let r = &mut table.pages[i];
                            if let Some(c) = Self::make_writable(pool, class, r, kept)? {
                                out.copies.push(c);
                            }
                        }
                        let page = table.pages[i].page;
                        pool.check_write(page)?;
                        if positions.end >= page_end {
                            pool.seal(page)?;
                        }
                    }
                    self.counters[ix].written += unit * u64::from(positions.end - positions.start);
                }
            }
        }
        h.tokens = positions.end;
        h.residency = self.residency(slot);
        Ok(out)
    }

    /// Whether window ring `ix` of slot `slot` holds every token page a sequence of `tokens`
    /// positions needs: those of its last `window` positions.
    fn ring_holds(&self, slot: usize, ix: usize, window: u32, tokens: u32) -> bool {
        let p = self.cfg.page_tokens;
        let tags = &self.tags[slot][ix];
        let ring = ring_pages(window, p) as usize;
        (tokens.saturating_sub(window) / p..tokens.div_ceil(p))
            .all(|t| tags.get(t as usize % ring) == Some(&t))
    }

    /// Drops positions from `to` on (RFC-0001 §5.9): rejected speculative positions, cancelled
    /// chunks. Pages wholly past `to` are released. A cut inside a page leaves it in place; the
    /// next append reopens or copies it.
    ///
    /// # Errors
    ///
    /// - [`KvError::Unsupported`] when the store has a `State` class and `to` is before
    ///   `h.tokens`: rolling state back needs the snapshots of TIER-E6-T18.
    /// - [`KvError::BadPosition`] when a window ring no longer holds the positions the cut keeps
    ///   in its window: the ring of `w + T_max` is TIER-E6-T09's.
    /// - [`KvError::Moving`] when a page to release is moving.
    pub fn truncate(&mut self, h: &mut KvHandle, to: u32) -> Result<(), KvError> {
        let slot = self.slot(h)?;
        if to >= h.tokens {
            return Ok(());
        }
        self.check_cut(slot, h.tokens, to)?;
        let p = self.cfg.page_tokens;
        for ix in 0..self.cfg.classes.len() {
            let class = self.cfg.classes[ix].class;
            let pool = &mut self.pools[ix];
            let table = &mut self.tables[slot].classes[ix];
            match class {
                PageClass::Windowed { .. } => {
                    // Every slot that holds a token page at or past the cut is emptied.
                    let keep = to.div_ceil(p);
                    for (i, tag) in self.tags[slot][ix].iter_mut().enumerate() {
                        if *tag != NO_PAGE && *tag >= keep {
                            let r = &mut table.pages[i];
                            pool.release(r.page)?;
                            *r = VACANT;
                            *tag = NO_PAGE;
                        }
                    }
                }
                PageClass::State => {}
                _ => {
                    let keep = to.div_ceil(p) as usize;
                    for r in &table.pages[keep..] {
                        pool.release(r.page)?;
                    }
                    table.pages.truncate(keep);
                }
            }
        }
        h.tokens = to;
        h.residency = self.residency(slot);
        Ok(())
    }

    /// Checks that a cut of slot `slot` from `tokens` back to `to` can be honoured.
    fn check_cut(&self, slot: usize, tokens: u32, to: u32) -> Result<(), KvError> {
        let p = self.cfg.page_tokens;
        for (ix, c) in self.cfg.classes.iter().enumerate() {
            match c.class {
                PageClass::State if to < tokens => {
                    return Err(KvError::Unsupported(
                        "rolling recurrent state back needs the snapshots of TIER-E6-T18",
                    ));
                }
                PageClass::Windowed { window } if !self.ring_holds(slot, ix, window, to) => {
                    return Err(KvError::BadPosition { tokens, at: to });
                }
                _ => {}
            }
            // The pages a cut releases: token pages at or past it.
            let table = &self.tables[slot].classes[ix];
            let keep = to.div_ceil(p);
            let released: &mut dyn Iterator<Item = &PageRef> = match c.class {
                PageClass::Full => &mut table.pages.iter().skip(keep as usize),
                PageClass::Windowed { .. } => &mut table
                    .pages
                    .iter()
                    .zip(&self.tags[slot][ix])
                    .filter(move |(_, t)| **t != NO_PAGE && **t >= keep)
                    .map(|(r, _)| r),
                _ => &mut std::iter::empty(),
            };
            for r in released {
                if let super::PageState::Moving { .. } = self.pools[ix].state(r.page) {
                    return Err(KvError::Moving { page: r.page });
                }
            }
        }
        Ok(())
    }

    /// Forks `h` at `at` (RFC-0001 §5.9): a new sequence slot that shares `h`'s pages for
    /// positions before `at`, copy-on-write. A page the fork point falls inside is sealed, so
    /// neither sequence writes it again without copying it first.
    ///
    /// # Errors
    ///
    /// - [`KvError::BadPosition`] when `at` is past `h.tokens`, or a window ring no longer
    ///   holds the positions before `at` that the window needs.
    /// - [`KvError::Unsupported`] when the store has a `State` class and `at` is before
    ///   `h.tokens` (TIER-E6-T18).
    /// - [`KvError::NoSlot`] when every slot is in use.
    pub fn fork(&mut self, h: &KvHandle, at: u32) -> Result<KvHandle, KvError> {
        let src = self.slot(h)?;
        if at > h.tokens {
            return Err(KvError::BadPosition {
                tokens: h.tokens,
                at,
            });
        }
        self.check_cut(src, h.tokens, at)?;
        let Some(dst) = self.free_slots.pop() else {
            return Err(KvError::NoSlot);
        };
        let dst = dst as usize;
        self.in_use[dst] = true;
        let p = self.cfg.page_tokens;
        for ix in 0..self.cfg.classes.len() {
            let class = self.cfg.classes[ix].class;
            let (lo, hi) = if src < dst {
                let (a, b) = self.tables.split_at_mut(dst);
                (&mut a[src].classes[ix], &mut b[0].classes[ix])
            } else {
                let (a, b) = self.tables.split_at_mut(src);
                (&mut b[0].classes[ix], &mut a[dst].classes[ix])
            };
            let (from, to) = (lo, hi);
            let pool = &mut self.pools[ix];
            let shared_upto = match class {
                PageClass::State => from.pages.len(),
                _ => at.div_ceil(p) as usize,
            };
            let ring = match class {
                PageClass::Windowed { window } => Some(ring_pages(window, p) as usize),
                _ => None,
            };
            let (src_tags, dst_tags) = if src < dst {
                let (a, b) = self.tags.split_at_mut(dst);
                (&a[src][ix], &mut b[0][ix])
            } else {
                let (a, b) = self.tags.split_at_mut(src);
                (&b[0][ix], &mut a[dst][ix])
            };
            for (i, r) in from.pages.iter_mut().enumerate() {
                // A ring slot is shared if the token page it holds is before the fork point.
                let tag = ring.map(|_| src_tags[i]);
                let keep = !vacant(r)
                    && match tag {
                        Some(t) => t != NO_PAGE && (t as usize) < shared_upto,
                        None => i < shared_upto,
                    };
                if let Some(t) = tag {
                    dst_tags.push(if keep { t } else { NO_PAGE });
                }
                if keep {
                    // A moving page keeps its old copy readable until the completion; only a
                    // writable one needs sealing before it is shared.
                    if pool.state(r.page) == super::PageState::Writable {
                        pool.seal(r.page)?;
                    }
                    pool.retain(r.page)?;
                    r.shared = true;
                    to.pages.push(PageRef {
                        page: r.page,
                        shared: true,
                    });
                } else if ring.is_some() {
                    to.pages.push(VACANT);
                }
            }
        }
        Ok(KvHandle {
            seq: h.seq,
            stage: h.stage,
            layers: h.layers.clone(),
            tokens: at,
            pages: PageTableId(dst as u32),
            residency: self.residency(dst),
            prefix: h.prefix,
        })
    }

    /// Releases a sequence (RFC-0001 §5.9): every reference its slot holds is dropped, and the
    /// slot is free again. Pages whose last reference this was become free at the next
    /// [`fence`](KvStore::fence).
    ///
    /// # Errors
    ///
    /// The handle back with [`KvError::Moving`] when one of its pages is moving, so the caller
    /// releases it once the move completes.
    pub fn release(&mut self, h: KvHandle) -> Result<(), (KvHandle, KvError)> {
        let slot = match self.slot(&h) {
            Ok(s) => s,
            Err(e) => return Err((h, e)),
        };
        for (ix, t) in self.tables[slot].classes.iter().enumerate() {
            for r in t.pages.iter().filter(|r| !vacant(r)) {
                if let super::PageState::Moving { .. } = self.pools[ix].state(r.page) {
                    return Err((h, KvError::Moving { page: r.page }));
                }
            }
        }
        for ix in 0..self.cfg.classes.len() {
            let table = &mut self.tables[slot].classes[ix];
            for r in table.pages.drain(..) {
                if !vacant(&r) {
                    self.pools[ix]
                        .release(r.page)
                        .expect("a held page that is not moving releases");
                }
            }
            table.provisional = 0;
            self.tags[slot][ix].clear();
        }
        self.in_use[slot] = false;
        self.free_slots.push(slot as u32);
        Ok(())
    }

    /// Counts the KV bytes one step reads for `h` at its committed length (RFC-0022 §5.4's
    /// `B_vram`): per class, every position (`Full`), the last `w` (`Windowed`), or the state.
    ///
    /// # Errors
    ///
    /// [`KvError::Moving`] when one of its pages is moving: a step never reads such a page.
    pub fn note_read(&mut self, h: &KvHandle) -> Result<(), KvError> {
        let slot = self.slot(h)?;
        for (ix, t) in self.tables[slot].classes.iter().enumerate() {
            for r in t.pages.iter().filter(|r| !vacant(r)) {
                self.pools[ix].check_read(r.page)?;
            }
        }
        for (ix, c) in self.cfg.classes.iter().enumerate() {
            let read = match c.class {
                PageClass::State => c.unit_bytes.0,
                PageClass::Windowed { window } => c.unit_bytes.0 * u64::from(h.tokens.min(window)),
                _ => c.unit_bytes.0 * u64::from(h.tokens),
            };
            self.counters[ix].read += read;
        }
        Ok(())
    }

    /// The bytes counted since the last [`flush`](KvStore::flush), per class.
    #[must_use]
    pub fn counters(&self) -> &[KvBytes] {
        &self.counters
    }

    /// Pushes one [`KvBytesRecord`] per class for `step` to the statistics ring and zeroes the
    /// counters. Returns how many records the ring dropped (it is lossy, RFC-0001 §8.3).
    pub fn flush(&mut self, step: u64, ring: &mut StatsProducer) -> u32 {
        let mut dropped = 0;
        for c in &mut self.counters {
            let record = KvBytesRecord {
                step,
                stage: self.cfg.stage,
                class: c.class,
                tier: c.tier,
                read: c.read,
                written: c.written,
            };
            if ring.push_kv_bytes(record).is_err() {
                dropped += 1;
            }
            c.read = 0;
            c.written = 0;
        }
        dropped
    }

    /// The step fence (RFC-0020 §7.4): pages released since the last fence become free.
    /// Returns how many.
    pub fn fence(&mut self) -> u32 {
        self.pools.iter_mut().map(Pool::fence).sum()
    }

    /// Free capacity in tokens per tier (RFC-0022 §11): for `Vram`, the minimum over the `Full`
    /// classes of free pages × P. With no `Full` class, nothing limits admission here, and the
    /// figure is `u32::MAX`. The spill tiers come with TIER-E6-T04.
    #[must_use]
    pub fn headroom(&self) -> KvHeadroom {
        let vram = self
            .cfg
            .classes
            .iter()
            .zip(&self.pools)
            .filter(|(c, _)| c.class == PageClass::Full)
            .map(|(_, pool)| pool.free_pages().saturating_mul(self.cfg.page_tokens))
            .min()
            .unwrap_or(u32::MAX);
        KvHeadroom {
            tokens: [vram, 0, 0],
        }
    }

    /// Starts a tier move of a sealed page: `Sealed → Moving`. The spill and fetch paths that
    /// copy it are TIER-E6-T04's; this is the state change they make.
    ///
    /// # Errors
    ///
    /// [`KvError::BadState`] for a page that is not sealed; [`KvError::Unsupported`] for a class
    /// the store does not have.
    pub fn begin_move(
        &mut self,
        class: PageClass,
        page: PageId,
        to: u8,
        ticket: MoveTicket,
    ) -> Result<(), KvError> {
        let ix = self
            .class_ix(class)
            .ok_or(KvError::Unsupported("no such class in this store"))?;
        self.pools[ix].begin_move(page, to, ticket)
    }

    /// Completes a move: `Moving → Sealed`.
    ///
    /// # Errors
    ///
    /// [`KvError::BadState`] for a page not moving under `ticket`.
    pub fn finish_move(
        &mut self,
        class: PageClass,
        page: PageId,
        ticket: MoveTicket,
    ) -> Result<(), KvError> {
        let ix = self
            .class_ix(class)
            .ok_or(KvError::Unsupported("no such class in this store"))?;
        self.pools[ix].finish_move(page, ticket)
    }
}
