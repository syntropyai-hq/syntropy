// SPDX-License-Identifier: MIT OR Apache-2.0

//! Prefix keys: `kv_settings_hash` and the page-key chain (RFC-0022 §7.1; ADR-0206 item 1;
//! TIER-E7-T01). Owner: RFC-0022.
//!
//! A prefix-cache page is found by a key that commits to the model, to every setting that changes
//! what KV holds, to the scope it may be shared in, and to every token before it:
//!
//! ```text
//! kv_settings_hash = H("syntropy/kv-settings/v1" ‖ canon(KvSettings, normal form))
//! k_root = H("syntropy-kv-prefix-v1" ‖ ModelId ‖ kv_settings_hash ‖ shared_salt)
//! k_b    = H(k_(b−1) ‖ tokens of page b as u32 LE ‖ media)                 inside the shared span
//! k_b    = H(k_(b−1) ‖ scope_salt ‖ tokens of page b as u32 LE ‖ media)    for every later page
//! ```
//!
//! `H` is BLAKE3. `core` links no hasher (RFC-0001 §4.2 rule 1), so, as with
//! `InstancePolicy::digest_with`, the caller passes the hash function in; every byte it hashes
//! is built here.
//!
//! - **[`KvSettings`]** holds every setting RFC-0022 §7.1 lists. [`KvSettings::canonical_bytes`]
//!   writes its normal form through [`canon`](crate::canon): fields by name, maps sorted, floats
//!   by bit pattern, so the bytes depend only on the values, never on the host, the build, a
//!   `Hash` derive or a map's order. Equal settings give equal bytes on every run.
//! - **[`PrefixKeyChain`]** computes `k_root` and then one key per full page, in order.
//! - **The shared span and the scope salt.** Under [`PrefixScopeMode::Principal`], the default
//!   (`kv.prefix_scope = principal`), `shared_salt` is empty. Pages wholly inside the first
//!   `shared_until` tokens, rounded down to whole pages, carry no salt; from the first page after
//!   them, every key mixes in `scope_salt`: the principal's 16 bytes, or [`LOCAL_SCOPE_SALT`] for
//!   the local principal. Under [`PrefixScopeMode::Instance`], the opt-in that restores one cache
//!   for every client, `shared_salt` is [`INSTANCE_SHARED_SALT`] and no page carries a scope salt:
//!   prefixes are shared across principals by design (RFC-0022 §7.1), which is what the operator
//!   opts into, not a leak.
//! - **The two constants** differ from each other and sit where no principal's salt can. The
//!   instance salt is only ever in `k_root`'s input, where principal scope puts nothing; a scope
//!   salt is only ever in a page's input. The local salt is a fixed ASCII string; a principal's
//!   salt is 16 bytes ingress derives from a credential by hashing (RFC-0050), so it equals the
//!   local salt only by a 128-bit collision.
//! - **Non-text inputs** ([`MediaSpan`]). A media item enters the token stream as the family's
//!   placeholder IDs, which are the same for every image. So each page that holds any of a media
//!   item's placeholders appends, after its tokens, one fixed-width 40-byte record per such item:
//!   the span's absolute start and length (u32 LE each) and its 32-byte content hash. Records go
//!   in order of absolute start (then length, then hash), whatever order the caller lists them
//!   in. A page's token bytes are always exactly `4·P` long and each record exactly 40, so a
//!   record cannot be read as tokens: a page with media is a longer input than any text page of
//!   the same chain. A page with no media hashes exactly RFC-0022 §7.1's bytes. A different image
//!   changes the key of the first page that holds it, and of every page after.
//!
//! The constants and the media encoding fill gaps in RFC-0022 §7.1 and are recorded in its note
//! of 2026-10-28.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::canon;
use crate::dtype::DType;
use crate::ids::{Blake3, LayerIx, ModelId};
use crate::plan::KvDType;
use crate::policy::ApproxKnobs;
use crate::sequence::{PrefixKey, PrefixScope};

/// The domain string of `kv_settings_hash`.
pub const KV_SETTINGS_DOMAIN: &[u8] = b"syntropy/kv-settings/v1";

/// The domain string of `k_root` (RFC-0022 §7.1).
pub const PREFIX_DOMAIN: &[u8] = b"syntropy-kv-prefix-v1";

/// The scope salt of the local principal (`PrefixScope::principal == None`). A fixed ASCII
/// string, where a principal's salt is a credential's hash (module docs).
pub const LOCAL_SCOPE_SALT: [u8; 16] = *b"syntropy/local\0\0";

/// `shared_salt` under `kv.prefix_scope = instance`. It appears only in `k_root`'s input, never
/// in a page's, so it cannot stand for a scope salt (module docs).
pub const INSTANCE_SHARED_SALT: [u8; 16] = *b"syntropy/instanc";

/// The approximate knobs that touch KV: a lossy stored dtype, and approximate attention, whose
/// sparsity or windowing changes what is kept (RFC-0042 §10).
pub const KV_KNOBS: ApproxKnobs = ApproxKnobs::KV_LOSSY.union(ApproxKnobs::APPROX_ATTENTION);

/// `kv.prefix_scope` (RFC-0022 §7.1, §9).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixScopeMode {
    /// Pages are shared across principals only inside the shared span. The default.
    #[default]
    Principal,
    /// One cache for every client of the instance: no scope salt on any page.
    Instance,
}

/// One RoPE configuration, as the family resolves it (RFC-0022 §7.1: "base, scaling type and
/// factors, original context length").
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RopeSettings {
    /// The base, `θ`.
    pub base: f64,
    /// The scaling type as the model's configuration names it, such as `"yarn"` or `"llama3"`;
    /// `None` for no scaling.
    pub scaling: Option<String>,
    /// The scaling's factors by name, such as `factor` or `beta_fast`.
    pub factors: BTreeMap<String, f64>,
    /// The original context length the scaling extends, if it names one.
    pub original_context: Option<u32>,
}

/// Every setting that changes KV contents or layout (RFC-0022 §7.1). It is the same at every
/// stage of an instance; the stage's layer range is not in it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KvSettings {
    /// `P`, tokens per page.
    pub page_tokens: u32,
    /// The stored `KvDType` of each page class, in class order.
    pub stored: Vec<KvDType>,
    /// `KvFormula::dtype`'s default: the model's KV dtype in exact mode.
    pub dtype: DType,
    /// `KvFormula::dtype`'s per-layer exceptions.
    pub dtype_layers: Vec<(LayerIx, DType)>,
    /// `KvFormula::layout_version`.
    pub layout_version: u16,
    /// The RoPE configurations the layers use, in the order the family lists them: one for most
    /// models, two for a family with separate local and global bases.
    pub rope: Vec<RopeSettings>,
    /// Each layer's sliding window, if it has one.
    pub windows: Vec<Option<u32>>,
    /// Each layer's compression ratio, tokens per compressed entry, if it compresses.
    pub compression: Vec<Option<u16>>,
    /// The instance's approximate knobs; only [`KV_KNOBS`] count.
    pub knobs: ApproxKnobs,
    /// `kv_lossy`'s dtype, which counts only while the knob is on.
    pub kv_lossy: Option<KvDType>,
}

impl KvSettings {
    /// The same settings in one form, so that settings which store the same KV hash the same:
    /// dtype exceptions sorted, each layer once (the first, as `KvDTypeSpec::of` reads it) and
    /// none equal to the default; knobs masked to [`KV_KNOBS`]; `kv_lossy` dropped while its knob
    /// is off.
    #[must_use]
    pub fn normalized(&self) -> KvSettings {
        let mut layers: Vec<(LayerIx, DType)> = Vec::with_capacity(self.dtype_layers.len());
        for &(layer, d) in &self.dtype_layers {
            if !layers.iter().any(|(l, _)| *l == layer) {
                layers.push((layer, d));
            }
        }
        layers.retain(|&(_, d)| d != self.dtype);
        layers.sort_unstable_by_key(|&(l, _)| l);
        let knobs = self.knobs.intersection(KV_KNOBS);
        KvSettings {
            dtype_layers: layers,
            knobs,
            kv_lossy: if knobs.contains(ApproxKnobs::KV_LOSSY) {
                self.kv_lossy
            } else {
                None
            },
            ..self.clone()
        }
    }

    /// The bytes `kv_settings_hash` is computed over: [`KV_SETTINGS_DOMAIN`], then the
    /// canonical encoding of the normal form.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = KV_SETTINGS_DOMAIN.to_vec();
        // `KvSettings` holds no 128-bit integer and reports no custom error, the canonical
        // encoding's only failures.
        out.extend(canon::to_bytes(&self.normalized()).expect("KvSettings has a canonical form"));
        out
    }

    /// `kv_settings_hash`: `hash` (BLAKE3) over [`canonical_bytes`](Self::canonical_bytes).
    pub fn hash_with(&self, hash: impl FnOnce(&[u8]) -> Blake3) -> Blake3 {
        hash(&self.canonical_bytes())
    }
}

/// A non-text input's run of placeholder tokens in the sequence, with its content hash
/// (RFC-0050 §2.1's `ContentPart::Media`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MediaSpan {
    /// The token position of its first placeholder.
    pub start: u32,
    /// How many placeholder tokens it has.
    pub len: u32,
    /// Its content hash.
    pub hash: Blake3,
}

impl MediaSpan {
    /// Whether it has a placeholder in `[from, to)`.
    fn overlaps(&self, from: u64, to: u64) -> bool {
        let start = u64::from(self.start);
        self.len > 0 && start < to && start + u64::from(self.len) > from
    }
}

/// The page-key chain of one sequence (module docs).
pub struct PrefixKeyChain<H> {
    hash: H,
    page_tokens: u32,
    prev: Blake3,
    next_page: u32,
    /// Pages wholly inside the shared span; `u32::MAX` under instance scope.
    shared_pages: u32,
    scope_salt: [u8; 16],
    buf: Vec<u8>,
}

impl<H: Fn(&[u8]) -> Blake3> PrefixKeyChain<H> {
    /// The chain for a sequence of model `model`, under settings whose hash is `kv_settings`,
    /// opened with `scope` in `mode`. `page_tokens` is `P`.
    ///
    /// # Panics
    ///
    /// If `page_tokens` is 0.
    pub fn new(
        hash: H,
        model: ModelId,
        kv_settings: Blake3,
        scope: &PrefixScope,
        mode: PrefixScopeMode,
        page_tokens: u32,
    ) -> Self {
        assert!(page_tokens > 0, "a page holds at least one token");
        let mut buf = Vec::with_capacity(32 + 16 + 4 * page_tokens as usize);
        buf.extend_from_slice(PREFIX_DOMAIN);
        buf.extend_from_slice(&model.0.0);
        buf.extend_from_slice(&kv_settings.0);
        let (shared_pages, scope_salt) = match mode {
            PrefixScopeMode::Instance => {
                buf.extend_from_slice(&INSTANCE_SHARED_SALT);
                (u32::MAX, [0; 16])
            }
            PrefixScopeMode::Principal => (
                scope.shared_until.map_or(0, |until| until / page_tokens),
                scope.principal.unwrap_or(LOCAL_SCOPE_SALT),
            ),
        };
        let root = hash(&buf);
        Self {
            hash,
            page_tokens,
            prev: root,
            next_page: 0,
            shared_pages,
            scope_salt,
            buf,
        }
    }

    /// `k_root` if no page has been added, else the last page's key.
    #[must_use]
    pub fn last(&self) -> PrefixKey {
        PrefixKey(self.prev)
    }

    /// The index of the page [`push_page`](Self::push_page) keys next.
    #[must_use]
    pub fn next_page(&self) -> u32 {
        self.next_page
    }

    /// Keys the next page, whose `P` tokens are `tokens`. `media` may list every media span of
    /// the sequence, in any order; the spans that hold a placeholder in this page are mixed in by
    /// absolute start, then length, then hash.
    ///
    /// # Panics
    ///
    /// If `tokens` is not exactly `P` long: only full pages have keys.
    pub fn push_page(&mut self, tokens: &[u32], media: &[MediaSpan]) -> PrefixKey {
        assert_eq!(
            tokens.len(),
            self.page_tokens as usize,
            "only a full page has a key"
        );
        let from = u64::from(self.next_page) * u64::from(self.page_tokens);
        let to = from + u64::from(self.page_tokens);
        self.buf.clear();
        self.buf.extend_from_slice(&self.prev.0);
        if self.next_page >= self.shared_pages {
            self.buf.extend_from_slice(&self.scope_salt);
        }
        for t in tokens {
            self.buf.extend_from_slice(&t.to_le_bytes());
        }
        let mut here: SmallVec<[&MediaSpan; 4]> =
            media.iter().filter(|m| m.overlaps(from, to)).collect();
        here.sort_unstable_by_key(|m| (m.start, m.len, m.hash.0));
        for span in here {
            self.buf.extend_from_slice(&span.start.to_le_bytes());
            self.buf.extend_from_slice(&span.len.to_le_bytes());
            self.buf.extend_from_slice(&span.hash.0);
        }
        self.prev = (self.hash)(&self.buf);
        self.next_page += 1;
        PrefixKey(self.prev)
    }

    /// Keys every full page of `tokens` from the next one on, and returns their keys in order.
    /// A partial last page has no key.
    pub fn push_tokens(&mut self, tokens: &[u32], media: &[MediaSpan]) -> Vec<PrefixKey> {
        tokens
            .chunks_exact(self.page_tokens as usize)
            .map(|page| self.push_page(page, media))
            .collect()
    }
}

#[cfg(test)]
mod tests;
