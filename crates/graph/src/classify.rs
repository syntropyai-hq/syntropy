// SPDX-License-Identifier: MIT OR Apache-2.0

//! Census classification: each family's name patterns, with shape checks, compiled into a
//! [`Matcher`] (RFC-0043 §4).
//!
//! - **Rules are data** ([`ClassRule`]) inside the family module, one [`RuleTable`] per source
//!   convention: GGUF names and the original safetensors names (§4.1).
//! - **Patterns** are literal text with placeholders: `{L}` a decoder layer (or, for an MTP
//!   rule, the module's layer), `{E}` a routed expert, and `{N}` any other index, such as a
//!   tower's layer, matched but not read. A placeholder matches a decimal number with no
//!   leading zero. [`Matcher::new`] refuses a pattern it could read two ways, and a rule whose
//!   placeholders do not fit its class.
//! - **Shapes** are checked against the hyperparameters after the name matches ([`ShapeDim`]),
//!   in the source's own dimension order: ggml's `ne`, innermost first, for GGUF; outermost
//!   first for safetensors. RFC-0043 §4.1's `ClassRule` lists no shape field, and §4.1 says the
//!   shape is checked; [`ClassRule::shape`] is where a rule states it. A name that matches with
//!   the wrong shape, an out-of-range layer or an out-of-range expert is
//!   [`Classification::Unknown`], never a guess, and the first rule whose pattern matches
//!   decides: a later rule is never tried.
//! - **Ignored tensors** (§4.3). A rule with [`ClassSel::Encoder`] matches a multimodal tower
//!   or projector tensor and shape-checks it like any other; v1 classifies it
//!   [`Classification::Ignored`] with [`MULTIMODAL_REASON`]. [`ClassSel::Ignored`] does the same
//!   for training-only tensors with the rule's own reason.
//! - **Planning** ([`classify_tensors`]). Classification reads names and shapes only, which the
//!   headers give, so it runs before any tensor byte is fetched. [`TensorPlan::fetch`] lists
//!   the classified tensors, the only ones a later fetch may read; ignored tensors are listed
//!   apart, as `ModelManifest::ignored` lists them (RFC-0001 §5.3).

use std::fmt;

use syntropy_core::{ExpertIx, LayerIx};
use syntropy_manifest::census::{TensorClass, TowerIx};
use syntropy_manifest::hparams::{HyperParams, N_EXPERTS};

use crate::families::Family;
use crate::ir::{Block, TensorRole};

/// The reason every v1 family gives for tower and projector tensors (RFC-0043 §4.3).
pub const MULTIMODAL_REASON: &str = "multimodal input arrives after 1.0";

/// One name pattern and what a tensor it matches is (RFC-0043 §4.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClassRule {
    /// The name, with `{L}`, `{E}` and `{N}` placeholders: `blk.{L}.attn_q.weight`,
    /// `model.layers.{L}.self_attn.q_proj.weight`.
    pub pattern: &'static str,
    /// Which block of layer `{L}`, or the prologue, epilogue, an MTP module or a tower.
    pub block: BlockSel,
    /// The tensor's role in its group.
    pub role: TensorRole,
    /// Its class.
    pub class: ClassSel,
    /// Its shape in the source's dimension order, one entry per dimension.
    pub shape: &'static [ShapeDim],
    /// The weight this scale or zero-point tensor binds to (RFC-0041 §2.2), as that weight's
    /// pattern. Read by the census builder; the engine does not use it.
    pub aux_of: Option<&'static str>,
}

/// Which part of the model a rule's tensor belongs to (RFC-0043 §4.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlockSel {
    /// A block of decoder layer `{L}`.
    Layer(Block),
    /// The prologue: embeddings and tables.
    Prologue,
    /// The epilogue: final norm and LM head.
    Epilogue,
    /// MTP module `n`, whose layer is `n_layers + n`.
    Mtp(u8),
    /// A multimodal tower or projector (RFC-0043 §2.4).
    Encoder,
}

/// The class a rule assigns (RFC-0043 §4.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClassSel {
    /// [`TensorClass::DenseEveryToken`].
    DenseEveryToken,
    /// One routed expert stored on its own (`…experts.{E}.gate_proj.weight`).
    Expert,
    /// Every routed expert of a layer in one tensor, the expert index outermost
    /// (`blk.{L}.ffn_gate_exps.weight`); the census slices it (RFC-0041 §2.1).
    ExpertsStacked,
    /// [`TensorClass::RowLookup`].
    RowLookup,
    /// [`TensorClass::Head`].
    Head,
    /// [`TensorClass::Router`].
    Router,
    /// A tower or projector tensor (RFC-0043 §2.4). `Ignored` with [`MULTIMODAL_REASON`] in v1.
    Encoder {
        /// The tower.
        tower: TowerIx,
    },
    /// A tensor the text path never reads, such as a training-only one, with the reason.
    Ignored {
        /// Why.
        reason: &'static str,
    },
}

/// One dimension of a rule's shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShapeDim {
    /// A fixed size.
    Const(u64),
    /// The hyperparameter of that name ([`HyperParams::count`]).
    Hp(&'static str),
    /// The product of two hyperparameters, such as `n_heads · d_head`.
    Mul(&'static str, &'static str),
    /// Any size.
    Any,
}

impl ShapeDim {
    fn expected(self, hp: &HyperParams) -> Option<Option<u64>> {
        Some(match self {
            Self::Const(n) => Some(n),
            Self::Hp(k) => Some(hp.count(k)?),
            Self::Mul(a, b) => Some(hp.count(a)?.checked_mul(hp.count(b)?)?),
            Self::Any => None,
        })
    }
}

/// The dimension order of a source's shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DimOrder {
    /// GGUF: ggml's `ne`, innermost (contiguous) dimension first.
    InnermostFirst,
    /// safetensors: outermost dimension first.
    OutermostFirst,
}

/// One source convention's rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuleTable {
    /// The order its shapes are written in.
    pub order: DimOrder,
    /// Its rules.
    pub rules: &'static [ClassRule],
}

/// What a tensor is (RFC-0043 §4.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Classification {
    /// A tensor the text path reads.
    Classified {
        /// Its class. For a stacked expert tensor, the class of its first slice,
        /// `Expert { layer, id: 0 }`.
        class: TensorClass,
        /// Its layer, if it has one.
        layer: Option<LayerIx>,
        /// Its role in its group.
        role: TensorRole,
        /// For a stacked expert tensor, how many experts it holds.
        stacked_experts: Option<u16>,
        /// The block it belongs to, which the census groups by (RFC-0041 §3.1).
        block: BlockSel,
    },
    /// A tensor that is never placed or fetched, with the reason (RFC-0043 §4.3).
    Ignored {
        /// Why.
        reason: &'static str,
    },
    /// No rule matches, or the matching rule's checks fail. The load is refused
    /// (`UnknownTensor`, RFC-0041 §10 code 3).
    Unknown,
}

/// Why [`Matcher::new`] refuses a rule. A module bug, caught by the family's own tests.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PatternError {
    /// The rule's pattern.
    pub pattern: &'static str,
    /// What is wrong.
    pub reason: String,
}

impl fmt::Display for PatternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "classification rule `{}`: {}", self.pattern, self.reason)
    }
}

impl std::error::Error for PatternError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Part {
    Lit(&'static str),
    Layer,
    Expert,
    Index,
}

#[derive(Clone, Debug)]
struct Compiled {
    rule: &'static ClassRule,
    order: DimOrder,
    parts: Vec<Part>,
}

#[derive(Clone, Copy, Debug, Default)]
struct Captures {
    layer: Option<u32>,
    expert: Option<u32>,
}

/// A family's rule tables, compiled once (RFC-0043 §4.1).
#[derive(Clone, Debug)]
pub struct Matcher {
    rules: Vec<Compiled>,
}

impl Matcher {
    /// Compiles `tables`.
    ///
    /// # Errors
    ///
    /// [`PatternError`] for an unknown or repeated placeholder, two placeholders with no text
    /// between them, a placeholder followed by a digit, a pattern listed twice, or
    /// placeholders that do not fit the rule's block and class.
    pub fn new(tables: &[RuleTable]) -> Result<Self, PatternError> {
        let mut rules: Vec<Compiled> = Vec::new();
        for table in tables {
            for rule in table.rules {
                if rules.iter().any(|c| c.rule.pattern == rule.pattern) {
                    return Err(err(rule, "the pattern is listed twice"));
                }
                let parts = parse(rule)?;
                check_rule(rule, table.order, &parts)?;
                rules.push(Compiled {
                    rule,
                    order: table.order,
                    parts,
                });
            }
        }
        Ok(Self { rules })
    }

    /// Classifies the tensor `name` of shape `shape`, in its source's order.
    #[must_use]
    pub fn classify(&self, name: &str, shape: &[u64], hp: &HyperParams) -> Classification {
        let Some((c, caps)) = self
            .rules
            .iter()
            .find_map(|c| matches(&c.parts, name).map(|caps| (c, caps)))
        else {
            return Classification::Unknown;
        };
        outcome(c, caps, shape, hp).unwrap_or(Classification::Unknown)
    }
}

fn err(rule: &ClassRule, reason: impl Into<String>) -> PatternError {
    PatternError {
        pattern: rule.pattern,
        reason: reason.into(),
    }
}

fn parse(rule: &'static ClassRule) -> Result<Vec<Part>, PatternError> {
    let mut parts = Vec::new();
    let mut rest = rule.pattern;
    while !rest.is_empty() {
        let Some(open) = rest.find('{') else {
            parts.push(Part::Lit(rest));
            break;
        };
        if open > 0 {
            parts.push(Part::Lit(&rest[..open]));
        }
        let close = rest[open..]
            .find('}')
            .ok_or_else(|| err(rule, "a `{` is not closed"))?;
        let part = match &rest[open + 1..open + close] {
            "L" => Part::Layer,
            "E" => Part::Expert,
            "N" => Part::Index,
            other => return Err(err(rule, format!("unknown placeholder `{{{other}}}`"))),
        };
        if matches!(parts.last(), Some(Part::Layer | Part::Expert | Part::Index)) {
            return Err(err(rule, "two placeholders with no text between them"));
        }
        if part != Part::Index && parts.contains(&part) {
            return Err(err(rule, "a placeholder appears twice"));
        }
        parts.push(part);
        rest = &rest[open + close + 1..];
    }
    for pair in parts.windows(2) {
        if let [p, Part::Lit(lit)] = pair
            && !matches!(p, Part::Lit(_))
            && lit.starts_with(|c: char| c.is_ascii_digit())
        {
            return Err(err(rule, "a placeholder is followed by a digit"));
        }
    }
    if parts
        .iter()
        .any(|p| matches!(p, Part::Lit(l) if l.contains('}')))
    {
        return Err(err(rule, "a `}` has no `{`"));
    }
    Ok(parts)
}

fn check_rule(rule: &ClassRule, order: DimOrder, parts: &[Part]) -> Result<(), PatternError> {
    let has = |p: Part| parts.contains(&p);
    let needs_layer = matches!(rule.block, BlockSel::Layer(_))
        || matches!(
            rule.class,
            ClassSel::Expert | ClassSel::ExpertsStacked | ClassSel::Router
        );
    if needs_layer && !has(Part::Layer) && !matches!(rule.block, BlockSel::Mtp(_)) {
        return Err(err(rule, "its block or class needs `{L}`"));
    }
    if has(Part::Layer) && matches!(rule.block, BlockSel::Prologue | BlockSel::Epilogue) {
        return Err(err(rule, "a prologue or epilogue tensor has no `{L}`"));
    }
    if has(Part::Layer) && rule.block == BlockSel::Encoder {
        return Err(err(rule, "a tower tensor's index is `{N}`, not `{L}`"));
    }
    if has(Part::Expert) != (rule.class == ClassSel::Expert) {
        return Err(err(
            rule,
            "`{E}` is used exactly by `ClassSel::Expert` rules",
        ));
    }
    if matches!(rule.class, ClassSel::Encoder { .. }) != (rule.block == BlockSel::Encoder) {
        return Err(err(
            rule,
            "`ClassSel::Encoder` goes with `BlockSel::Encoder`",
        ));
    }
    if rule.class == ClassSel::ExpertsStacked {
        let outer = match order {
            DimOrder::InnermostFirst => rule.shape.last(),
            DimOrder::OutermostFirst => rule.shape.first(),
        };
        if outer != Some(&ShapeDim::Hp(N_EXPERTS)) {
            return Err(err(
                rule,
                format!("a stacked expert tensor has `{N_EXPERTS}` as its outermost dimension"),
            ));
        }
    }
    Ok(())
}

/// Parses a placeholder's number: ASCII digits, no leading zero, within `u32`.
fn number(s: &str) -> Option<(u32, usize)> {
    let len = s.bytes().take_while(u8::is_ascii_digit).count();
    let digits = &s[..len];
    if digits.is_empty() || (digits.len() > 1 && digits.starts_with('0')) {
        return None;
    }
    Some((digits.parse().ok()?, len))
}

fn matches(parts: &[Part], name: &str) -> Option<Captures> {
    let mut caps = Captures::default();
    let mut rest = name;
    for part in parts {
        match part {
            Part::Lit(lit) => rest = rest.strip_prefix(lit)?,
            Part::Layer | Part::Expert | Part::Index => {
                let (n, len) = number(rest)?;
                match part {
                    Part::Layer => caps.layer = Some(n),
                    Part::Expert => caps.expert = Some(n),
                    _ => {}
                }
                rest = &rest[len..];
            }
        }
    }
    rest.is_empty().then_some(caps)
}

fn shape_ok(c: &Compiled, shape: &[u64], hp: &HyperParams) -> Option<()> {
    let rule = c.rule;
    if shape.len() != rule.shape.len() {
        return None;
    }
    for (&dim, &got) in rule.shape.iter().zip(shape) {
        if let Some(want) = dim.expected(hp)?
            && want != got
        {
            return None;
        }
    }
    Some(())
}

/// The outcome for a tensor whose name matched `c`; `None` when a check fails.
fn outcome(
    c: &Compiled,
    caps: Captures,
    shape: &[u64],
    hp: &HyperParams,
) -> Option<Classification> {
    let rule = c.rule;
    let layer = match rule.block {
        BlockSel::Layer(_) => {
            let l = u16::try_from(caps.layer?)
                .ok()
                .filter(|&l| l < hp.n_layers)?;
            Some(LayerIx(l))
        }
        BlockSel::Mtp(n) => {
            let l = hp.n_layers.checked_add(n.into())?;
            if caps.layer.is_some_and(|got| got != u32::from(l)) {
                return None;
            }
            Some(LayerIx(l))
        }
        BlockSel::Prologue | BlockSel::Epilogue | BlockSel::Encoder => None,
    };
    if let Some(e) = caps.expert
        && u64::from(e) >= hp.count(N_EXPERTS)?
    {
        return None;
    }
    shape_ok(c, shape, hp)?;
    let classified = |class, stacked_experts| Classification::Classified {
        class,
        layer,
        role: rule.role,
        stacked_experts,
        block: rule.block,
    };
    Some(match rule.class {
        ClassSel::DenseEveryToken => classified(TensorClass::DenseEveryToken, None),
        ClassSel::RowLookup => classified(TensorClass::RowLookup, None),
        ClassSel::Head => classified(TensorClass::Head, None),
        ClassSel::Router => classified(TensorClass::Router { layer: layer? }, None),
        ClassSel::Expert => classified(
            TensorClass::Expert {
                layer: layer?,
                id: ExpertIx(u16::try_from(caps.expert?).ok()?),
            },
            None,
        ),
        ClassSel::ExpertsStacked => {
            let outer = match c.order {
                DimOrder::InnermostFirst => shape.last(),
                DimOrder::OutermostFirst => shape.first(),
            };
            let n = u16::try_from(*outer?).ok().filter(|&n| n > 0)?;
            classified(
                TensorClass::Expert {
                    layer: layer?,
                    id: ExpertIx(0),
                },
                Some(n),
            )
        }
        ClassSel::Encoder { .. } => Classification::Ignored {
            reason: MULTIMODAL_REASON,
        },
        ClassSel::Ignored { reason } => Classification::Ignored { reason },
    })
}

/// A checkpoint's tensors after classification, before any tensor byte is read: the planning
/// output a fetch is driven by (RFC-0043 §4.3).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TensorPlan {
    /// The classified tensors, in input order, each with its classification. The only
    /// tensors a fetch may read.
    pub fetch: Vec<(String, Classification)>,
    /// The ignored tensors with their reasons: `ModelManifest::ignored`. Never fetched.
    pub ignored: Vec<(String, &'static str)>,
    /// The tensors no rule classifies. Any entry refuses the load (`UnknownTensor`, RFC-0041
    /// §10 code 3).
    pub unknown: Vec<String>,
}

/// Classifies every tensor of a checkpoint, from its names and shapes alone.
pub fn classify_tensors<'a>(
    family: &dyn Family,
    hp: &HyperParams,
    tensors: impl IntoIterator<Item = (&'a str, &'a [u64])>,
) -> TensorPlan {
    let mut plan = TensorPlan::default();
    for (name, shape) in tensors {
        match family.classify(name, shape, hp) {
            c @ Classification::Classified { .. } => plan.fetch.push((name.to_owned(), c)),
            Classification::Ignored { reason } => plan.ignored.push((name.to_owned(), reason)),
            Classification::Unknown => plan.unknown.push(name.to_owned()),
        }
    }
    plan
}

#[cfg(test)]
mod tests;
