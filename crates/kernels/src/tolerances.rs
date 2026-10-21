// SPDX-License-Identifier: MIT OR Apache-2.0

//! The tolerance tables of the exactness contract: `numerics/tolerances-v<N>.toml`.
//!
//! RFC-0042 §6 defines the tables and ADR-0422 how they are calibrated. A table holds:
//! - per dtype class, the end-to-end ceilings `C(d)` as a triple (mean KL, p99 KL, top-1
//!   disagreement), pooled and per corpus category (RFC-0042 §6.1, §6.2);
//! - per class and category, the design effect `deff_c`; per class, the canary ceiling
//!   `C_canary(d)` (§6.2);
//! - per (layer kind, class, statistic), the layer-local ceiling `C_ll` (§5.2);
//! - per (op class, class), the op-level `ε_op`, `ceiling_op`, `ceiling_op_max`, `tol_op` and
//!   `tol_ref` (§6.4);
//! - per family, the model-definition oracle's `τ_def` with `S_pub` and `S_ref` (§7.1).
//!
//! Every entry is [`Entry::Pending`] until calibration publishes it, so a table can list every
//! entry the schema requires before any value exists. The loader rejects a table that leaves
//! one out, names an unknown class, category, layer kind or statistic, or breaks a rule the
//! RFC states for a value (each check cites it).
//!
//! **Release stamping.** Every build embeds the newest table in `numerics/`
//! ([`RELEASE_VERSION`], [`RELEASE_TEXT`]). [`ToleranceTable::for_build`] selects it, or the
//! file named by the `numerics.tolerances` override (RFC-0042 §11), and returns the
//! [`TableStamp`] that results and responses carry: RFC-0050 §10's `FidelityReport::contract`
//! and RFC-0060 §6.4's `ConformanceManifest::ceilings`. An override marks the build unofficial.
//!
//! **Release blocker** (ADR-0019 decision 3; RFC-0042 §6.2, §11). A release build refuses to
//! start a model whose dtype class has no published end-to-end ceiling:
//! [`ToleranceTable::gate_model_start`] returns [`ToleranceMissing`], numerics code 1. The
//! release checklist (RFC-0062 §9.3) runs [`ToleranceTable::check_release`] over the classes a release
//! supports.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use syntropy_core::{DTypeClass, ErrorClass};

/// The schema this loader reads. A table states it as `schema`.
pub const SCHEMA: u16 = 1;

/// The version of the table this build embeds: the newest `numerics/tolerances-v<N>.toml`.
pub const RELEASE_VERSION: u32 = 1;

/// The text of the table this build embeds, `numerics/tolerances-v1.toml`.
pub const RELEASE_TEXT: &str = include_str!("../../../numerics/tolerances-v1.toml");

/// A corpus category (RFC-0060 §5.2; RFC-0042 §5.5).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Category {
    /// Multi-turn conversations rendered with each family's chat template.
    Chat,
    /// Source files and diffs.
    Code,
    /// Single documents of at least 32K tokens.
    Long,
    /// Tool-call JSON, function schemas and structured output.
    Tool,
    /// Text in at least eight languages.
    Multi,
    /// Repeated tokens, whitespace runs, the longest token and every special token.
    Stress,
}

impl Category {
    /// Every category, in RFC-0060 §5.2's order.
    pub const ALL: [Category; 6] = [
        Category::Chat,
        Category::Code,
        Category::Long,
        Category::Tool,
        Category::Multi,
        Category::Stress,
    ];

    /// The category's name in RFC-0060 §5.2 and in the table, such as `tool`.
    #[must_use]
    pub const fn key(self) -> &'static str {
        match self {
            Category::Chat => "chat",
            Category::Code => "code",
            Category::Long => "long",
            Category::Tool => "tool",
            Category::Multi => "multi",
            Category::Stress => "stress",
        }
    }

    fn from_key(key: &str) -> Option<Category> {
        Category::ALL.into_iter().find(|c| c.key() == key)
    }
}

/// Where an end-to-end aggregate is taken: over every position, or within one category
/// (RFC-0042 §5.3, §6.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Scope {
    /// Every scored position of the corpus.
    Pooled,
    /// The scored positions of one category.
    Category(Category),
}

impl Scope {
    /// Pooled, then each category in RFC-0060 §5.2's order.
    pub const ALL: [Scope; 7] = [
        Scope::Pooled,
        Scope::Category(Category::Chat),
        Scope::Category(Category::Code),
        Scope::Category(Category::Long),
        Scope::Category(Category::Tool),
        Scope::Category(Category::Multi),
        Scope::Category(Category::Stress),
    ];

    /// The scope's key in the table: `pooled` or the category's name.
    #[must_use]
    pub const fn key(self) -> &'static str {
        match self {
            Scope::Pooled => "pooled",
            Scope::Category(c) => c.key(),
        }
    }
}

/// The table key of each dtype class (RFC-0042 §1.2), from the highest precision to the
/// lowest.
pub const CLASS_KEYS: [(DTypeClass, &str); 5] = [
    (DTypeClass::Float16, "float16"),
    (DTypeClass::Fp8, "fp8"),
    (DTypeClass::Int8, "int8"),
    (DTypeClass::Bits4, "bits4"),
    (DTypeClass::Sub4, "sub4"),
];

/// The table key of a dtype class, such as `bits4`.
#[must_use]
pub fn class_key(class: DTypeClass) -> &'static str {
    CLASS_KEYS
        .iter()
        .find(|(c, _)| *c == class)
        .map_or("unknown", |(_, key)| key)
}

fn class_from_key(key: &str) -> Option<DTypeClass> {
    CLASS_KEYS.iter().find(|(_, k)| *k == key).map(|(c, _)| *c)
}

/// A layer kind of the layer-local gate (RFC-0042 §5.2; RFC-0060 §5.7).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LayerKind {
    /// A dense feed-forward block.
    DenseFfn,
    /// A mixture-of-experts feed-forward block.
    MoeFfn,
    /// Attention over the whole context.
    FullAttention,
    /// Attention over a sliding window.
    SlidingWindowAttention,
    /// Linear attention.
    LinearAttention,
    /// The last sampled layer's window, run on through the final norm and the LM head.
    HeadWindow,
}

impl LayerKind {
    /// Every layer kind, in RFC-0042 §5.2's order.
    pub const ALL: [LayerKind; 6] = [
        LayerKind::DenseFfn,
        LayerKind::MoeFfn,
        LayerKind::FullAttention,
        LayerKind::SlidingWindowAttention,
        LayerKind::LinearAttention,
        LayerKind::HeadWindow,
    ];

    /// The kind's key in the table, such as `moe_ffn`.
    #[must_use]
    pub const fn key(self) -> &'static str {
        match self {
            LayerKind::DenseFfn => "dense_ffn",
            LayerKind::MoeFfn => "moe_ffn",
            LayerKind::FullAttention => "full_attention",
            LayerKind::SlidingWindowAttention => "sliding_window_attention",
            LayerKind::LinearAttention => "linear_attention",
            LayerKind::HeadWindow => "head_window",
        }
    }

    /// The statistics that have a ceiling for this kind (RFC-0042 §5.2): the mean and p99
    /// `rel_l2` for every kind, and also top-1 and `KL_K` for the head window, whose window
    /// ends at the logits (RFC-0060 §5.7 item 3).
    #[must_use]
    pub const fn statistics(self) -> &'static [LayerStatistic] {
        match self {
            LayerKind::HeadWindow => &LayerStatistic::ALL,
            _ => &[LayerStatistic::RelL2Mean, LayerStatistic::RelL2P99],
        }
    }

    fn from_key(key: &str) -> Option<LayerKind> {
        LayerKind::ALL.into_iter().find(|k| k.key() == key)
    }
}

/// A statistic of the layer-local gate (RFC-0042 §5.2; RFC-0060 §5.7 item 3).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LayerStatistic {
    /// The mean over scored positions of the window output's `rel_l2`.
    RelL2Mean,
    /// The nearest-rank p99 over scored positions of the window output's `rel_l2`.
    RelL2P99,
    /// The head window's top-1 disagreement rate.
    Top1,
    /// `KL_K` of the head window's logits, in nats. RFC-0042 §5.2 and RFC-0060 §5.7 name the
    /// statistic without fixing its aggregate over positions.
    KlK,
}

impl LayerStatistic {
    /// Every statistic.
    pub const ALL: [LayerStatistic; 4] = [
        LayerStatistic::RelL2Mean,
        LayerStatistic::RelL2P99,
        LayerStatistic::Top1,
        LayerStatistic::KlK,
    ];

    /// The statistic's key in the table, such as `rel_l2_p99`.
    #[must_use]
    pub const fn key(self) -> &'static str {
        match self {
            LayerStatistic::RelL2Mean => "rel_l2_mean",
            LayerStatistic::RelL2P99 => "rel_l2_p99",
            LayerStatistic::Top1 => "top1",
            LayerStatistic::KlK => "kl_k",
        }
    }

    fn rule(self) -> Rule {
        match self {
            LayerStatistic::Top1 => Rule::Rate,
            _ => Rule::NonNegative,
        }
    }
}

/// A table entry: pending until calibration publishes it.
///
/// `Provisional` is RFC-0042's `provisional` flag (§5.2, §6.2): a published ceiling that rests
/// on dense calibration models only or, for a layer-local ceiling, on fewer than three
/// families with that layer kind. It still gates. Only ceilings carry it; the loader rejects
/// it on design effects, op-level tolerances and `τ_def`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Entry<T> {
    /// No value yet: calibration has not published it.
    Pending,
    /// A published ceiling marked `provisional`.
    Provisional(T),
    /// A published value.
    Final(T),
}

impl<T: Copy> Entry<T> {
    /// The published value, provisional or final.
    #[must_use]
    pub fn value(&self) -> Option<T> {
        match *self {
            Entry::Pending => None,
            Entry::Provisional(v) | Entry::Final(v) => Some(v),
        }
    }

    /// Whether the entry has no value yet.
    #[must_use]
    pub fn is_pending(&self) -> bool {
        matches!(self, Entry::Pending)
    }

    /// Whether the entry is a published ceiling marked `provisional`.
    #[must_use]
    pub fn is_provisional(&self) -> bool {
        matches!(self, Entry::Provisional(_))
    }
}

/// An end-to-end tolerance triple (RFC-0042 §6.1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Triple {
    /// `τ_kl_mean`: mean KL, in nats.
    pub kl_mean: f64,
    /// `τ_kl_p99`: nearest-rank p99 KL, in nats.
    pub kl_p99: f64,
    /// `τ_top1`: the top-1 disagreement rate, in [0, 1].
    pub top1: f64,
}

/// The op-level tolerances of one (op class, dtype class) (RFC-0042 §6.4).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpTolerance {
    /// `ε_op`: the p99 `rel_l2` of the F32Fixed reference with BF16-rounded inputs against
    /// F64, the floor.
    pub eps_op: f64,
    /// `ceiling_op`: the ceiling on `rel_l2` against F64.
    pub ceiling_op: f64,
    /// `ceiling_op_max`: the ceiling on `max_abs_scaled` against F64.
    pub ceiling_op_max: f64,
    /// `tol_op`: the p99 `rel_l2` limit of a candidate against ggml (RFC-0060 §4.2 rule 3).
    pub tol_op: f64,
    /// `tol_ref`: the limit against `kernels::reference`, which the `Warming` spot check uses
    /// (RFC-0040 §6.2). Equal to `tol_op`.
    pub tol_ref: f64,
}

/// Where `S_pub` comes from (RFC-0042 §7.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SPubSource {
    /// The publisher's code in F32 against the same code in F64.
    Publisher,
    /// The publisher's code cannot run in F64, so `S_ref` stands in for `S_pub`.
    Ref,
}

/// One metric of the model-definition oracle's tolerance (RFC-0042 §7.1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DefinitionalMetric {
    /// `S_pub(f)`: the metric between the publisher's code in F32 and in F64.
    pub s_pub: f64,
    /// `S_ref(f)`: the metric between `RefExecutor` in `F32Fixed` and in F64.
    pub s_ref: f64,
    /// `τ_def(f) = 2 · max(S_pub, S_ref)`.
    pub tau_def: f64,
}

/// A family's definitional tolerance (RFC-0042 §7.1; ADR-0422 decision 7), per metric of the
/// end-to-end gate plus the tapped hidden-state norms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Definitional {
    /// Where `S_pub` comes from.
    pub s_pub_source: SPubSource,
    /// Mean KL, in nats.
    pub kl_mean: DefinitionalMetric,
    /// Nearest-rank p99 KL, in nats.
    pub kl_p99: DefinitionalMetric,
    /// Top-1 disagreement at positions whose golden margin is at least `τ_tie`.
    pub top1: DefinitionalMetric,
    /// The relative difference of the tapped residual-stream norms.
    pub norm_rel: DefinitionalMetric,
}

/// A dtype class's published end-to-end ceilings, pooled and per category.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClassCeiling {
    /// The class.
    pub class: DTypeClass,
    /// The pooled triple.
    pub pooled: Triple,
    categories: [Triple; 6],
    /// The class's ceilings are marked `provisional` (RFC-0042 §6.2).
    pub provisional: bool,
}

impl ClassCeiling {
    /// The triple of one category.
    #[must_use]
    pub fn category(&self, category: Category) -> Triple {
        let ix = Category::ALL
            .iter()
            .position(|c| *c == category)
            .unwrap_or(0);
        self.categories[ix]
    }

    /// The triple of one scope.
    #[must_use]
    pub fn scope(&self, scope: Scope) -> Triple {
        match scope {
            Scope::Pooled => self.pooled,
            Scope::Category(c) => self.category(c),
        }
    }
}

/// Whether the running build is a release build: RFC-0062 §4.4's `release` profile, which
/// everything shipped is built with. `dev`, `test-release`, `chaos` and `pr` are not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BuildKind {
    /// The `release` profile.
    Release,
    /// Any other profile.
    NonRelease,
}

/// Where a build's table came from.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TableSource {
    /// The table the build embeds.
    Release,
    /// A file named by the `numerics.tolerances` override, which marks the build unofficial
    /// (RFC-0042 §11).
    Override(PathBuf),
}

/// The table version a build judges results against, as results and responses carry it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TableStamp {
    /// `N` in `tolerances-v<N>.toml`.
    pub version: u32,
    /// Where the table came from.
    pub source: TableSource,
}

impl TableStamp {
    /// Whether the table is the one the build embeds.
    #[must_use]
    pub fn is_official(&self) -> bool {
        self.source == TableSource::Release
    }

    /// The reference results and responses carry: `tolerances-v<N>` for the embedded table,
    /// and `tolerances-v<N>+unofficial` for an override.
    #[must_use]
    pub fn contract_ref(&self) -> String {
        self.to_string()
    }
}

impl fmt::Display for TableStamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tolerances-v{}", self.version)?;
        if self.is_official() {
            Ok(())
        } else {
            f.write_str("+unofficial")
        }
    }
}

/// `ToleranceMissing`, numerics code 1, class `Internal` (RFC-0042 §11): a supported dtype
/// class has no published end-to-end ceiling, so a release build does not start a model of
/// that class (ADR-0019 decision 3; RFC-0042 §6.2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToleranceMissing {
    version: u32,
    classes: Vec<DTypeClass>,
}

impl ToleranceMissing {
    /// The code in the `Numerics` domain.
    pub const CODE: u16 = 1;
    /// The code's name in RFC-0042 §11.
    pub const NAME: &'static str = "ToleranceMissing";
    /// The error class RFC-0042 §11 gives it.
    pub const CLASS: ErrorClass = ErrorClass::Internal;

    /// The table version that lacks the ceilings.
    #[must_use]
    pub fn version(&self) -> u32 {
        self.version
    }

    /// The classes without a published ceiling, from the highest precision to the lowest.
    #[must_use]
    pub fn classes(&self) -> &[DTypeClass] {
        &self.classes
    }
}

impl fmt::Display for ToleranceMissing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: tolerances-v{} has no published end-to-end ceiling for dtype class",
            Self::NAME,
            self.version
        )?;
        if self.classes.len() > 1 {
            f.write_str("es")?;
        }
        for (i, class) in self.classes.iter().enumerate() {
            let sep = if i == 0 { " " } else { ", " };
            write!(f, "{sep}{}", class_key(*class))?;
        }
        f.write_str("; a release build does not start a model of such a class")
    }
}

impl std::error::Error for ToleranceMissing {}

/// Why a table could not be loaded.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TableError {
    /// The file could not be read.
    Read {
        /// The file.
        path: PathBuf,
        /// The operating system's message.
        message: String,
    },
    /// The file name is not `tolerances-v<N>.toml` with `N ≥ 1` and no leading zero.
    FileName {
        /// The file.
        path: PathBuf,
    },
    /// The text is not TOML, or a field has the wrong type or name.
    Parse {
        /// The parser's message.
        message: String,
    },
    /// The table's `schema` is not [`SCHEMA`].
    Schema {
        /// The schema found.
        found: u16,
    },
    /// The table's `version` differs from its file name or from [`RELEASE_VERSION`].
    Version {
        /// The version the file name or the build expects.
        expected: u32,
        /// The table's `version`.
        found: u32,
    },
    /// An entry the schema requires is absent.
    Missing {
        /// The entry, such as `layer_local.bits4.head_window.kl_k`.
        entry: String,
    },
    /// A key names no dtype class, category, layer kind or statistic.
    Unknown {
        /// The key's path.
        entry: String,
    },
    /// An entry breaks a rule.
    Invalid {
        /// The entry.
        entry: String,
        /// The rule it breaks.
        reason: &'static str,
    },
}

impl fmt::Display for TableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TableError::Read { path, message } => {
                write!(f, "cannot read {}: {message}", path.display())
            }
            TableError::FileName { path } => {
                write!(f, "{} is not named tolerances-v<N>.toml", path.display())
            }
            TableError::Parse { message } => write!(f, "tolerance table: {message}"),
            TableError::Schema { found } => {
                write!(
                    f,
                    "tolerance table schema {found}; this build reads {SCHEMA}"
                )
            }
            TableError::Version { expected, found } => {
                write!(f, "tolerance table version {found}; expected {expected}")
            }
            TableError::Missing { entry } => write!(f, "tolerance table lacks {entry}"),
            TableError::Unknown { entry } => write!(f, "tolerance table: unknown key {entry}"),
            TableError::Invalid { entry, reason } => {
                write!(f, "tolerance table: {entry}: {reason}")
            }
        }
    }
}

impl std::error::Error for TableError {}

/// A loaded tolerance table. Every entry the schema requires is present, pending or published.
#[derive(Clone, Debug, PartialEq)]
pub struct ToleranceTable {
    version: u32,
    end_to_end: BTreeMap<(DTypeClass, Scope), Entry<Triple>>,
    design_effect: BTreeMap<(DTypeClass, Category), Entry<f64>>,
    canary: BTreeMap<DTypeClass, Entry<f64>>,
    layer_local: BTreeMap<(LayerKind, DTypeClass, LayerStatistic), Entry<f64>>,
    ops: BTreeMap<(String, DTypeClass), Entry<OpTolerance>>,
    definitional: BTreeMap<String, Entry<Definitional>>,
}

impl ToleranceTable {
    /// Parses and checks a table's text.
    ///
    /// # Errors
    ///
    /// [`TableError::Parse`], [`TableError::Schema`], [`TableError::Missing`],
    /// [`TableError::Unknown`] and [`TableError::Invalid`] (version 0 included).
    pub fn parse(text: &str) -> Result<ToleranceTable, TableError> {
        let head: RawHead = toml::from_str(text).map_err(parse_error)?;
        if head.schema != SCHEMA {
            return Err(TableError::Schema { found: head.schema });
        }
        let raw: RawTable = toml::from_str(text).map_err(parse_error)?;
        if raw.version == 0 {
            return Err(invalid("version", "a table version is at least 1"));
        }
        Ok(ToleranceTable {
            version: raw.version,
            end_to_end: end_to_end(raw.end_to_end)?,
            design_effect: design_effect(raw.design_effect)?,
            canary: canary(raw.canary)?,
            layer_local: layer_local(raw.layer_local)?,
            ops: ops(raw.op)?,
            definitional: definitional(raw.definitional)?,
        })
    }

    /// Reads and checks the table in `path`, whose name must be `tolerances-v<N>.toml` with
    /// `N` equal to the table's `version`.
    ///
    /// # Errors
    ///
    /// [`TableError::FileName`], [`TableError::Read`], [`TableError::Version`] when the
    /// versions differ, and every error of [`ToleranceTable::parse`].
    pub fn load(path: &Path) -> Result<ToleranceTable, TableError> {
        let expected = version_from_file_name(path).ok_or_else(|| TableError::FileName {
            path: path.to_path_buf(),
        })?;
        let text = std::fs::read_to_string(path).map_err(|e| TableError::Read {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
        let table = ToleranceTable::parse(&text)?;
        if table.version != expected {
            return Err(TableError::Version {
                expected,
                found: table.version,
            });
        }
        Ok(table)
    }

    /// The table this build embeds ([`RELEASE_TEXT`]).
    ///
    /// # Errors
    ///
    /// Every error of [`ToleranceTable::parse`], and [`TableError::Version`] when the embedded
    /// table's version is not [`RELEASE_VERSION`]. The release tests rule both out.
    pub fn release() -> Result<ToleranceTable, TableError> {
        let table = ToleranceTable::parse(RELEASE_TEXT)?;
        if table.version != RELEASE_VERSION {
            return Err(TableError::Version {
                expected: RELEASE_VERSION,
                found: table.version,
            });
        }
        Ok(table)
    }

    /// The table a build judges results against, and its stamp: the embedded table, or the
    /// file named by `numerics.tolerances` (RFC-0042 §11), which marks the build unofficial.
    ///
    /// # Errors
    ///
    /// Every error of [`ToleranceTable::release`] or [`ToleranceTable::load`].
    pub fn for_build(
        override_path: Option<&Path>,
    ) -> Result<(ToleranceTable, TableStamp), TableError> {
        let (table, source) = match override_path {
            None => (ToleranceTable::release()?, TableSource::Release),
            Some(path) => (
                ToleranceTable::load(path)?,
                TableSource::Override(path.to_path_buf()),
            ),
        };
        let stamp = TableStamp {
            version: table.version,
            source,
        };
        Ok((table, stamp))
    }

    /// `N` in `tolerances-v<N>.toml`.
    #[must_use]
    pub fn version(&self) -> u32 {
        self.version
    }

    /// The end-to-end ceiling triple of a class in a scope.
    #[must_use]
    pub fn end_to_end(&self, class: DTypeClass, scope: Scope) -> Entry<Triple> {
        self.end_to_end
            .get(&(class, scope))
            .copied()
            .unwrap_or(Entry::Pending)
    }

    /// A class's published end-to-end ceilings, or `None` while they are pending.
    #[must_use]
    pub fn class_ceiling(&self, class: DTypeClass) -> Option<ClassCeiling> {
        let pooled = self.end_to_end(class, Scope::Pooled);
        let mut categories = [pooled.value()?; 6];
        for (slot, category) in categories.iter_mut().zip(Category::ALL) {
            *slot = self.end_to_end(class, Scope::Category(category)).value()?;
        }
        Some(ClassCeiling {
            class,
            pooled: pooled.value()?,
            categories,
            provisional: pooled.is_provisional(),
        })
    }

    /// The design effect `deff_c` of a class and category.
    #[must_use]
    pub fn design_effect(&self, class: DTypeClass, category: Category) -> Entry<f64> {
        self.design_effect
            .get(&(class, category))
            .copied()
            .unwrap_or(Entry::Pending)
    }

    /// The canary ceiling `C_canary(d)`, in nats.
    #[must_use]
    pub fn canary(&self, class: DTypeClass) -> Entry<f64> {
        self.canary.get(&class).copied().unwrap_or(Entry::Pending)
    }

    /// The layer-local ceiling `C_ll(kind, class, statistic)`, or `None` when the statistic
    /// has no ceiling for that kind ([`LayerKind::statistics`]).
    #[must_use]
    pub fn layer_local(
        &self,
        kind: LayerKind,
        class: DTypeClass,
        statistic: LayerStatistic,
    ) -> Option<Entry<f64>> {
        if !kind.statistics().contains(&statistic) {
            return None;
        }
        Some(
            self.layer_local
                .get(&(kind, class, statistic))
                .copied()
                .unwrap_or(Entry::Pending),
        )
    }

    /// The op-level tolerances of an op class and dtype class, or `None` when the table has no
    /// entry for them.
    #[must_use]
    pub fn op(&self, op: &str, class: DTypeClass) -> Option<Entry<OpTolerance>> {
        self.ops.get(&(op.to_owned(), class)).copied()
    }

    /// Every op-level entry: (op class, dtype class, entry), sorted.
    pub fn ops(&self) -> impl Iterator<Item = (&str, DTypeClass, Entry<OpTolerance>)> {
        self.ops
            .iter()
            .map(|((op, class), entry)| (op.as_str(), *class, *entry))
    }

    /// A family's definitional tolerance, or `None` when the table has no entry for it.
    #[must_use]
    pub fn definitional(&self, family: &str) -> Option<Entry<Definitional>> {
        self.definitional.get(family).copied()
    }

    /// The families with a definitional entry, sorted.
    pub fn families(&self) -> impl Iterator<Item = &str> {
        self.definitional.keys().map(String::as_str)
    }

    /// The start-path check (RFC-0042 §11): the ceiling a model of `class` is judged against.
    ///
    /// Returns the class's published ceilings, or `None` while they are pending, which only a
    /// non-release build accepts.
    ///
    /// # Errors
    ///
    /// [`ToleranceMissing`] when `build` is [`BuildKind::Release`] and the class has no
    /// published end-to-end ceiling.
    pub fn gate_model_start(
        &self,
        class: DTypeClass,
        build: BuildKind,
    ) -> Result<Option<ClassCeiling>, ToleranceMissing> {
        match (self.class_ceiling(class), build) {
            (Some(ceiling), _) => Ok(Some(ceiling)),
            (None, BuildKind::NonRelease) => Ok(None),
            (None, BuildKind::Release) => Err(ToleranceMissing {
                version: self.version,
                classes: vec![class],
            }),
        }
    }

    /// The release check (ADR-0019 decision 3; RFC-0062 §9.3's release checklist): every
    /// class a release supports has a published end-to-end ceiling.
    ///
    /// # Errors
    ///
    /// [`ToleranceMissing`] listing every supported class without one.
    pub fn check_release(&self, supported: &[DTypeClass]) -> Result<(), ToleranceMissing> {
        let mut classes: Vec<DTypeClass> = supported
            .iter()
            .copied()
            .filter(|c| self.class_ceiling(*c).is_none())
            .collect();
        if classes.is_empty() {
            return Ok(());
        }
        // Highest precision first, as RFC-0042 §1.2 lists the classes.
        classes.sort_unstable_by(|a, b| b.cmp(a));
        classes.dedup();
        Err(ToleranceMissing {
            version: self.version,
            classes,
        })
    }
}

/// `N` from a file named `tolerances-v<N>.toml`, with `N ≥ 1` and no leading zero.
fn version_from_file_name(path: &Path) -> Option<u32> {
    let name = path.file_name()?.to_str()?;
    let digits = name.strip_prefix("tolerances-v")?.strip_suffix(".toml")?;
    if digits.is_empty() || digits.starts_with('0') || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

fn parse_error(e: toml::de::Error) -> TableError {
    TableError::Parse {
        message: e.to_string(),
    }
}

// ---------------------------------------------------------------------------------------------
// The file format. Field names are the table's; the module docs and the table's header say
// what each holds.
// ---------------------------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawHead {
    schema: u16,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTable {
    #[allow(dead_code, reason = "read by RawHead first")]
    schema: u16,
    version: u32,
    end_to_end: BTreeMap<String, BTreeMap<String, RawTriple>>,
    design_effect: BTreeMap<String, BTreeMap<String, RawValue>>,
    canary: BTreeMap<String, RawValue>,
    layer_local: BTreeMap<String, BTreeMap<String, BTreeMap<String, RawValue>>>,
    #[serde(default)]
    op: Vec<RawOp>,
    #[serde(default)]
    definitional: BTreeMap<String, RawDefinitional>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RawStatus {
    Pending,
    Provisional,
    Final,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTriple {
    status: RawStatus,
    kl_mean: Option<f64>,
    kl_p99: Option<f64>,
    top1: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawValue {
    status: RawStatus,
    value: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOp {
    op: String,
    class: String,
    status: RawStatus,
    eps_op: Option<f64>,
    ceiling_op: Option<f64>,
    ceiling_op_max: Option<f64>,
    tol_op: Option<f64>,
    tol_ref: Option<f64>,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RawSPubSource {
    Publisher,
    Ref,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDefinitional {
    status: RawStatus,
    s_pub_source: Option<RawSPubSource>,
    kl_mean: Option<RawDefinitionalMetric>,
    kl_p99: Option<RawDefinitionalMetric>,
    top1: Option<RawDefinitionalMetric>,
    norm_rel: Option<RawDefinitionalMetric>,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDefinitionalMetric {
    s_pub: f64,
    s_ref: f64,
    tau_def: f64,
}

// ---------------------------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------------------------

/// The range a value must lie in.
#[derive(Clone, Copy)]
enum Rule {
    /// Finite and at least 0: KL in nats, `rel_l2`, `max_abs_scaled`.
    NonNegative,
    /// A rate in [0, 1]: top-1 disagreement.
    Rate,
    /// Finite and at least 1: a design effect, which RFC-0042 §6.2 publishes as 1 when below.
    AtLeastOne,
}

fn invalid(entry: &str, reason: &'static str) -> TableError {
    TableError::Invalid {
        entry: entry.to_owned(),
        reason,
    }
}

fn check(entry: &str, v: f64, rule: Rule) -> Result<f64, TableError> {
    let ok = match rule {
        Rule::NonNegative => v.is_finite() && v >= 0.0,
        Rule::Rate => (0.0..=1.0).contains(&v),
        Rule::AtLeastOne => v.is_finite() && v >= 1.0,
    };
    if ok {
        Ok(v)
    } else {
        Err(invalid(
            entry,
            match rule {
                Rule::NonNegative => "the value must be finite and at least 0",
                Rule::Rate => "a rate must lie in [0, 1]",
                Rule::AtLeastOne => "a design effect must be finite and at least 1 (RFC-0042 §6.2)",
            },
        ))
    }
}

/// Whether `v` has at most two significant decimal digits, as a ceiling rounded up to two
/// significant digits has (RFC-0042 §5.2, §6.2, §6.4). `v` is such a ceiling when formatting
/// it to two significant digits and reading the result back gives `v` again.
pub(crate) fn two_significant_digits(v: f64) -> bool {
    v == 0.0 || format!("{v:.1e}").parse::<f64>().is_ok_and(|r| r == v)
}

fn ceiling(entry: &str, v: f64, rule: Rule) -> Result<f64, TableError> {
    let v = check(entry, v, rule)?;
    if two_significant_digits(v) {
        Ok(v)
    } else {
        Err(invalid(
            entry,
            "a ceiling is rounded up to two significant digits (RFC-0042 §6.2)",
        ))
    }
}

/// Builds an entry from its status and its value, if the entry carries one.
fn entry<T>(
    name: &str,
    status: RawStatus,
    value: Option<T>,
    provisional_allowed: bool,
) -> Result<Entry<T>, TableError> {
    match (status, value) {
        (RawStatus::Pending, None) => Ok(Entry::Pending),
        (RawStatus::Pending, Some(_)) => Err(invalid(name, "a pending entry carries no value")),
        (RawStatus::Provisional | RawStatus::Final, None) => Err(TableError::Missing {
            entry: format!("{name}: the value of a published entry"),
        }),
        (RawStatus::Provisional, Some(_)) if !provisional_allowed => Err(invalid(
            name,
            "only ceilings carry the provisional flag (RFC-0042 §5.2, §6.2)",
        )),
        (RawStatus::Provisional, Some(v)) => Ok(Entry::Provisional(v)),
        (RawStatus::Final, Some(v)) => Ok(Entry::Final(v)),
    }
}

/// Every key of `map` is a known key, and every known key is in `map`. Unknown keys are
/// reported first, so a misspelt key is named as such rather than as a missing one.
fn exact_keys<V>(
    prefix: &str,
    map: &BTreeMap<String, V>,
    known: &[&str],
) -> Result<(), TableError> {
    if let Some(key) = map.keys().find(|k| !known.contains(&k.as_str())) {
        return Err(TableError::Unknown {
            entry: format!("{prefix}.{key}"),
        });
    }
    if let Some(key) = known.iter().find(|k| !map.contains_key(**k)) {
        return Err(TableError::Missing {
            entry: format!("{prefix}.{key}"),
        });
    }
    Ok(())
}

fn class_keys() -> Vec<&'static str> {
    CLASS_KEYS.iter().map(|(_, k)| *k).collect()
}

/// The values of `map`, keyed by class, after [`exact_keys`].
fn by_class<V>(
    prefix: &str,
    map: BTreeMap<String, V>,
) -> Result<Vec<(DTypeClass, String, V)>, TableError> {
    exact_keys(prefix, &map, &class_keys())?;
    Ok(map
        .into_iter()
        .filter_map(|(key, v)| class_from_key(&key).map(|c| (c, format!("{prefix}.{key}"), v)))
        .collect())
}

fn end_to_end(
    raw: BTreeMap<String, BTreeMap<String, RawTriple>>,
) -> Result<BTreeMap<(DTypeClass, Scope), Entry<Triple>>, TableError> {
    let scopes: Vec<&str> = Scope::ALL.iter().map(|s| s.key()).collect();
    let mut out = BTreeMap::new();
    for (class, prefix, by_scope) in by_class("end_to_end", raw)? {
        exact_keys(&prefix, &by_scope, &scopes)?;
        let mut status = None;
        for scope in Scope::ALL {
            let name = format!("{prefix}.{}", scope.key());
            let Some(t) = by_scope.get(scope.key()) else {
                continue;
            };
            if *status.get_or_insert(t.status) != t.status {
                return Err(invalid(
                    &prefix,
                    "a class's end-to-end scopes share one status",
                ));
            }
            let value = match (t.kl_mean, t.kl_p99, t.top1) {
                (None, None, None) => None,
                (Some(kl_mean), Some(kl_p99), Some(top1)) => Some(Triple {
                    kl_mean: ceiling(&format!("{name}.kl_mean"), kl_mean, Rule::NonNegative)?,
                    kl_p99: ceiling(&format!("{name}.kl_p99"), kl_p99, Rule::NonNegative)?,
                    top1: ceiling(&format!("{name}.top1"), top1, Rule::Rate)?,
                }),
                _ => {
                    return Err(invalid(
                        &name,
                        "a triple has kl_mean, kl_p99 and top1, or none of them",
                    ));
                }
            };
            out.insert((class, scope), entry(&name, t.status, value, true)?);
        }
    }
    Ok(out)
}

fn design_effect(
    raw: BTreeMap<String, BTreeMap<String, RawValue>>,
) -> Result<BTreeMap<(DTypeClass, Category), Entry<f64>>, TableError> {
    let categories: Vec<&str> = Category::ALL.iter().map(|c| c.key()).collect();
    let mut out = BTreeMap::new();
    for (class, prefix, by_category) in by_class("design_effect", raw)? {
        exact_keys(&prefix, &by_category, &categories)?;
        for (key, v) in by_category {
            let Some(category) = Category::from_key(&key) else {
                continue;
            };
            let name = format!("{prefix}.{key}");
            let value = v
                .value
                .map(|x| check(&name, x, Rule::AtLeastOne))
                .transpose()?;
            out.insert((class, category), entry(&name, v.status, value, false)?);
        }
    }
    Ok(out)
}

fn canary(raw: BTreeMap<String, RawValue>) -> Result<BTreeMap<DTypeClass, Entry<f64>>, TableError> {
    let mut out = BTreeMap::new();
    for (class, name, v) in by_class("canary", raw)? {
        let value = v
            .value
            .map(|x| check(&name, x, Rule::NonNegative))
            .transpose()?;
        out.insert(class, entry(&name, v.status, value, true)?);
    }
    Ok(out)
}

type LayerLocal = BTreeMap<(LayerKind, DTypeClass, LayerStatistic), Entry<f64>>;

fn layer_local(
    raw: BTreeMap<String, BTreeMap<String, BTreeMap<String, RawValue>>>,
) -> Result<LayerLocal, TableError> {
    let kinds: Vec<&str> = LayerKind::ALL.iter().map(|k| k.key()).collect();
    let mut out = BTreeMap::new();
    for (class, prefix, by_kind) in by_class("layer_local", raw)? {
        exact_keys(&prefix, &by_kind, &kinds)?;
        for (kind_key, by_statistic) in by_kind {
            let Some(kind) = LayerKind::from_key(&kind_key) else {
                continue;
            };
            let kind_prefix = format!("{prefix}.{kind_key}");
            let statistics: Vec<&str> = kind.statistics().iter().map(|s| s.key()).collect();
            exact_keys(&kind_prefix, &by_statistic, &statistics)?;
            for statistic in kind.statistics() {
                let Some(v) = by_statistic.get(statistic.key()) else {
                    continue;
                };
                let name = format!("{kind_prefix}.{}", statistic.key());
                let value = v
                    .value
                    .map(|x| ceiling(&name, x, statistic.rule()))
                    .transpose()?;
                out.insert(
                    (kind, class, *statistic),
                    entry(&name, v.status, value, true)?,
                );
            }
        }
    }
    Ok(out)
}

fn ops(raw: Vec<RawOp>) -> Result<BTreeMap<(String, DTypeClass), Entry<OpTolerance>>, TableError> {
    let mut out = BTreeMap::new();
    for (i, o) in raw.into_iter().enumerate() {
        let name = format!("op[{i}]");
        if o.op.is_empty() || o.op.chars().any(char::is_whitespace) {
            return Err(invalid(
                &name,
                "an op class is a non-empty name without spaces",
            ));
        }
        let class = class_from_key(&o.class).ok_or_else(|| TableError::Unknown {
            entry: format!("{name}.class = {}", o.class),
        })?;
        let name = format!("op[{i}] ({}, {})", o.op, o.class);
        let fields = [
            o.eps_op,
            o.ceiling_op,
            o.ceiling_op_max,
            o.tol_op,
            o.tol_ref,
        ];
        let value = if fields.iter().all(Option::is_none) {
            None
        } else if let [
            Some(eps_op),
            Some(ceiling_op),
            Some(ceiling_op_max),
            Some(tol_op),
            Some(tol_ref),
        ] = fields
        {
            Some(op_tolerance(
                &name,
                OpTolerance {
                    eps_op,
                    ceiling_op,
                    ceiling_op_max,
                    tol_op,
                    tol_ref,
                },
            )?)
        } else {
            return Err(invalid(
                &name,
                "an op entry has all of eps_op, ceiling_op, ceiling_op_max, tol_op and tol_ref, or none",
            ));
        };
        let e = entry(&name, o.status, value, false)?;
        if out.insert((o.op, class), e).is_some() {
            return Err(invalid(
                &name,
                "a second entry for the same op class and dtype class",
            ));
        }
    }
    Ok(out)
}

/// RFC-0042 §6.4's relations between the five op-level quantities.
fn op_tolerance(name: &str, t: OpTolerance) -> Result<OpTolerance, TableError> {
    check(&format!("{name}.eps_op"), t.eps_op, Rule::NonNegative)?;
    ceiling(
        &format!("{name}.ceiling_op"),
        t.ceiling_op,
        Rule::NonNegative,
    )?;
    ceiling(
        &format!("{name}.ceiling_op_max"),
        t.ceiling_op_max,
        Rule::NonNegative,
    )?;
    check(&format!("{name}.tol_op"), t.tol_op, Rule::NonNegative)?;
    check(&format!("{name}.tol_ref"), t.tol_ref, Rule::NonNegative)?;
    if t.tol_ref != t.tol_op {
        return Err(invalid(name, "tol_ref equals tol_op (RFC-0042 §6.4)"));
    }
    if t.tol_op < t.eps_op {
        return Err(invalid(
            name,
            "tol_op = max(2 · p99 rel_l2(ggml, reference), eps_op) is at least eps_op (RFC-0042 §6.4)",
        ));
    }
    if t.ceiling_op < 2.0 * t.eps_op {
        return Err(invalid(
            name,
            "ceiling_op = max over cases of 2 · max(rel_l2(ggml, F64), eps_op) is at least 2 · eps_op (RFC-0042 §6.4)",
        ));
    }
    Ok(t)
}

/// A family name: lowercase ASCII letters, digits and `_`, starting with a letter, as
/// RFC-0043 names families (`llama`, `qwen_moe`).
fn family_name(name: &str) -> bool {
    name.starts_with(|c: char| c.is_ascii_lowercase())
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

fn definitional(
    raw: BTreeMap<String, RawDefinitional>,
) -> Result<BTreeMap<String, Entry<Definitional>>, TableError> {
    let mut out = BTreeMap::new();
    for (family, d) in raw {
        let name = format!("definitional.{family}");
        if !family_name(&family) {
            return Err(invalid(
                &name,
                "a family name is lowercase letters, digits and _, starting with a letter",
            ));
        }
        let parts = (d.s_pub_source, d.kl_mean, d.kl_p99, d.top1, d.norm_rel);
        let value = match parts {
            (None, None, None, None, None) => None,
            (Some(source), Some(kl_mean), Some(kl_p99), Some(top1), Some(norm_rel)) => {
                let source = match source {
                    RawSPubSource::Publisher => SPubSource::Publisher,
                    RawSPubSource::Ref => SPubSource::Ref,
                };
                let metric = |key: &str, m: RawDefinitionalMetric, rule: Rule| {
                    definitional_metric(&format!("{name}.{key}"), m, rule, source)
                };
                Some(Definitional {
                    s_pub_source: source,
                    kl_mean: metric("kl_mean", kl_mean, Rule::NonNegative)?,
                    kl_p99: metric("kl_p99", kl_p99, Rule::NonNegative)?,
                    top1: metric("top1", top1, Rule::Rate)?,
                    norm_rel: metric("norm_rel", norm_rel, Rule::NonNegative)?,
                })
            }
            _ => {
                return Err(invalid(
                    &name,
                    "a published entry has s_pub_source, kl_mean, kl_p99, top1 and norm_rel; a pending one has none",
                ));
            }
        };
        out.insert(family, entry(&name, d.status, value, false)?);
    }
    Ok(out)
}

/// RFC-0042 §7.1: `τ_def = 2 · max(S_pub, S_ref)`, and `S_pub = S_ref` when `S_ref` stands
/// in for it. Doubling is exact in binary floating point, so the check is exact.
fn definitional_metric(
    name: &str,
    m: RawDefinitionalMetric,
    rule: Rule,
    source: SPubSource,
) -> Result<DefinitionalMetric, TableError> {
    let s_pub = check(&format!("{name}.s_pub"), m.s_pub, rule)?;
    let s_ref = check(&format!("{name}.s_ref"), m.s_ref, rule)?;
    let tau_def = check(&format!("{name}.tau_def"), m.tau_def, Rule::NonNegative)?;
    if source == SPubSource::Ref && s_pub != s_ref {
        return Err(invalid(
            name,
            "with s_pub_source = \"ref\", S_ref stands in for S_pub, so s_pub equals s_ref (RFC-0042 §7.1)",
        ));
    }
    if tau_def != 2.0 * s_pub.max(s_ref) {
        return Err(invalid(
            name,
            "tau_def = 2 · max(s_pub, s_ref) (RFC-0042 §7.1)",
        ));
    }
    Ok(DefinitionalMetric {
        s_pub,
        s_ref,
        tau_def,
    })
}

#[cfg(test)]
mod tests;
