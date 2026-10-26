// SPDX-License-Identifier: MIT OR Apache-2.0

//! The selection engine: which memory domain drives a device, and which provider runs each
//! op inside it (RFC-0040 §6.2, §6.4; ADR-0401 items 1, 2 and 4).
//!
//! The engine is pure data in, choice out. The executor queries the providers, runs the
//! microbenchmarks of RFC-0040 §6.3 and hands the results here, so recorded benchmark tables
//! replay through the same code (`select/tests.rs`).
//!
//! **Domain first** ([`choose_domain`], RFC-0040 §6.4's first table; ADR-0401 item 1). A
//! [`Vendor`] fixes the domains considered. NVIDIA takes `Cuda` whenever it opens, Apple takes
//! `Metal`, the CPU takes `Host`, and ggml-only devices take their ggml domain. AMD (`Hip`
//! against `Vulkan`) and Intel GPUs (`Vulkan` against `Ggml(Sycl)`) compare the predicted step
//! times when both open: the other domain must be more than 5 % faster
//! ([`DOMAIN_MARGIN_PERCENT`]) to replace `Hip` (or `Vulkan`). The predicted step time of a
//! domain is [`ProgramChoice::predicted_step`], the sum over the step program of the per-op
//! winners' costs, segment breaks included. It exists only when every op of the program was
//! decided by measurement: RFC-0040 §6.4 compares the two domains when both were benchmarked,
//! and a sum that leaves out the ops the benchmark budget skipped (§6.3) is not comparable
//! with one that covers them. When either prediction is missing, the other domain cannot show
//! the 5 % win, so the default stays ("else `Hip`").
//!
//! **Candidates** ([`admit`], RFC-0040 §6.2 conditions 1–4). A candidate is in the device's
//! domain, its `query` answered `Some`, its `Numerics` passes the RFC-0042 §4 accumulation
//! contract ([`numerics::check_caps`]) and, for a matrix product, the activation quantisation
//! rule of RFC-0042 §4.3 for the op's weight tensor ([`act::check_act`], with the tensor's
//! [`OpSlot::act`]), its layout is the layout the reader reads, and it is deterministic when
//! the runner was opened deterministic. Conditions 5 (a certificate in the
//! `ConformanceManifest`) and 6 (the runtime spot check) come with COMP-E2-T02 and the spot
//! check; until then every candidate that passes 1–4 is admitted.
//!
//! **Op per domain second** ([`decide_op`], [`select_program`]; ADR-0401 item 4). The cost of
//! a candidate is `measured + breaks · t_submit + fences · t_fence`. A capture-safe candidate
//! forces no segment break, so its `breaks` count is ignored. The incumbent is the native
//! candidate if one is admitted, else the cached choice, else the cheapest. The cheapest
//! candidate replaces the incumbent only when it costs less than 97 % of it
//! ([`OP_MARGIN_PERCENT`]). Ties go to native.
//!
//! **Determinism.** Every comparison uses integer nanoseconds, and every ordering ends in a
//! total tie-break (cost, native first, provider name, variant). The choice therefore does not
//! depend on the order the candidates are listed in, and identical inputs give identical
//! choices.

use std::cmp::Ordering;
use std::fmt;

use syntropy_core::resource::{Backend, DeviceClass, GgmlBackend};
use syntropy_core::{ErrorClass, Nanos};

use crate::device::{MemoryDomain, ProviderId};
use crate::error::{Error, ErrorCode};
use crate::numerics::act::{self, ActContext, ActViolation};
use crate::numerics::{self, NumericsViolation};
use crate::op::{LayoutVersion, OpCaps, OpKey, ShapeSig};

/// A challenger must cost less than `100 − 3` percent of the incumbent to replace it
/// (RFC-0040 §6.4: `cost(challenger) < 0.97 · cost(incumbent)`; ADR-0401 item 4).
pub const OP_MARGIN_PERCENT: u64 = 3;

/// On AMD and Intel GPUs the other domain must predict a step time below `100 − 5` percent of
/// the default domain's to be chosen (RFC-0040 §6.4; ADR-0401 item 1).
pub const DOMAIN_MARGIN_PERCENT: u64 = 5;

/// Whether `challenger` is below `(100 − margin_percent) %` of `incumbent`, in exact integer
/// arithmetic.
fn beats(challenger: Nanos, incumbent: Nanos, margin_percent: u64) -> bool {
    u128::from(challenger.0) * 100 < u128::from(incumbent.0) * u128::from(100 - margin_percent)
}

// ---- Domain choice -------------------------------------------------------------------------

/// The device families RFC-0040 §6.4's domain table has a row for.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Vendor {
    /// NVIDIA: `Cuda`, then `Vulkan`, then a ggml domain.
    Nvidia,
    /// AMD: `Hip` against `Vulkan`, by measurement.
    Amd,
    /// Apple: `Metal`.
    Apple,
    /// An Intel GPU: `Vulkan` against `Ggml(Sycl)`, by measurement.
    IntelGpu,
    /// The CPU: `Host`.
    Cpu,
    /// A device only a ggml backend drives (Adreno, Hexagon, CANN, ...).
    GgmlOnly(GgmlBackend),
}

/// The PCI vendor IDs Vulkan reports for the vendors with their own row.
const PCI_NVIDIA: u32 = 0x10DE;
const PCI_AMD: u32 = 0x1002;
const PCI_INTEL: u32 = 0x8086;

impl Vendor {
    /// The row a device of class `class` takes; `None` for a class with no row (an NPU, or a
    /// Vulkan device of another vendor, whose row the caller decides from the ggml backend
    /// that drives it).
    #[must_use]
    pub fn of_class(class: &DeviceClass) -> Option<Self> {
        match class {
            DeviceClass::Cuda { .. } => Some(Self::Nvidia),
            DeviceClass::Hip { .. } => Some(Self::Amd),
            DeviceClass::Metal { .. } => Some(Self::Apple),
            DeviceClass::Cpu { .. } => Some(Self::Cpu),
            DeviceClass::Vulkan { vendor, .. } => match *vendor {
                PCI_NVIDIA => Some(Self::Nvidia),
                PCI_AMD => Some(Self::Amd),
                PCI_INTEL => Some(Self::IntelGpu),
                _ => None,
            },
            _ => None,
        }
    }

    /// The (default, challenger) pair of a vendor whose domain is chosen by measurement.
    fn measured_pair(self) -> Option<(MemoryDomain, MemoryDomain)> {
        match self {
            Self::Amd => Some((MemoryDomain::Hip, MemoryDomain::Vulkan)),
            Self::IntelGpu => Some((MemoryDomain::Vulkan, MemoryDomain::Ggml(GgmlBackend::Sycl))),
            _ => None,
        }
    }
}

/// A domain whose provider opened the device, with the predicted step time of the program in
/// it, where it was benchmarked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DomainOption {
    /// The domain.
    pub domain: MemoryDomain,
    /// [`ProgramChoice::predicted_step`] in this domain; `None` when not benchmarked, or not
    /// benchmarked for every op.
    pub predicted: Option<Nanos>,
}

/// Why [`choose_domain`] chose a domain.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DomainReason {
    /// The vendor's rule takes this domain whenever it opens (`Cuda` on NVIDIA, `Metal`,
    /// `Host`, a ggml-only backend).
    Rule,
    /// The first domain in the vendor's order that opened (NVIDIA without `Cuda`).
    FirstThatOpens,
    /// Only one of the two measured domains opened.
    OnlyOneOpens,
    /// Both measured domains opened, but this one has no complete predicted step time, so the
    /// default domain stays (RFC-0040 §6.4, "else `Hip`").
    Unmeasured(MemoryDomain),
    /// Both measured domains opened and the step times decided.
    Measured {
        /// The default domain's predicted step time.
        default: Nanos,
        /// The other domain's predicted step time.
        challenger: Nanos,
    },
}

/// The domain chosen for a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DomainChoice {
    /// The domain.
    pub domain: MemoryDomain,
    /// Why.
    pub reason: DomainReason,
}

/// Why no domain was chosen.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DomainError {
    /// No domain the vendor's row considers opened the device.
    NoneOpens(Vendor),
}

impl fmt::Display for DomainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoneOpens(vendor) => {
                write!(
                    f,
                    "no memory domain considered for {vendor:?} opened the device"
                )
            }
        }
    }
}

impl std::error::Error for DomainError {}

/// Picks the one memory domain that drives a device (RFC-0040 §2.3, §6.4; ADR-0401 item 1).
/// `opened` lists the domains whose provider opened the device; domains the vendor's row does
/// not consider are ignored, and of a domain listed twice the first entry counts.
///
/// # Errors
///
/// [`DomainError::NoneOpens`] when no considered domain opened.
pub fn choose_domain(vendor: Vendor, opened: &[DomainOption]) -> Result<DomainChoice, DomainError> {
    let find = |domain: MemoryDomain| opened.iter().find(|o| o.domain == domain);
    let pick = |domain, reason| DomainChoice { domain, reason };

    if let Some((default, other)) = vendor.measured_pair() {
        return match (find(default), find(other)) {
            (Some(d), Some(o)) => {
                let (dt, ot) = match (d.predicted, o.predicted) {
                    (Some(dt), Some(ot)) => (dt, ot),
                    (None, _) => return Ok(pick(default, DomainReason::Unmeasured(default))),
                    (Some(_), None) => return Ok(pick(default, DomainReason::Unmeasured(other))),
                };
                let reason = DomainReason::Measured {
                    default: dt,
                    challenger: ot,
                };
                let domain = if beats(ot, dt, DOMAIN_MARGIN_PERCENT) {
                    other
                } else {
                    default
                };
                Ok(pick(domain, reason))
            }
            (Some(_), None) => Ok(pick(default, DomainReason::OnlyOneOpens)),
            (None, Some(_)) => Ok(pick(other, DomainReason::OnlyOneOpens)),
            (None, None) => Err(DomainError::NoneOpens(vendor)),
        };
    }

    let fixed = match vendor {
        Vendor::Apple => Some(MemoryDomain::Metal),
        Vendor::Cpu => Some(MemoryDomain::Host),
        Vendor::GgmlOnly(b) => Some(MemoryDomain::Ggml(b)),
        _ => None,
    };
    if let Some(domain) = fixed {
        return find(domain)
            .map(|_| pick(domain, DomainReason::Rule))
            .ok_or(DomainError::NoneOpens(vendor));
    }

    // NVIDIA: `Cuda` whenever it opens; else `Vulkan`; else a ggml domain, the lowest backend
    // first so the order of `opened` does not matter.
    if find(MemoryDomain::Cuda).is_some() {
        return Ok(pick(MemoryDomain::Cuda, DomainReason::Rule));
    }
    if find(MemoryDomain::Vulkan).is_some() {
        return Ok(pick(MemoryDomain::Vulkan, DomainReason::FirstThatOpens));
    }
    opened
        .iter()
        .filter_map(|o| match o.domain {
            MemoryDomain::Ggml(b) => Some(b),
            _ => None,
        })
        .min()
        .map(|b| pick(MemoryDomain::Ggml(b), DomainReason::FirstThatOpens))
        .ok_or(DomainError::NoneOpens(vendor))
}

// ---- Candidates ----------------------------------------------------------------------------

/// One (provider, prepared variant) offered for an (op, shape) on a device, with what its
/// `query` answered (RFC-0040 §6.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate {
    /// The provider.
    pub provider: ProviderId,
    /// The provider family: every family but ggml is native (ADR-0002).
    pub backend: Backend,
    /// The provider's memory domain.
    pub domain: MemoryDomain,
    /// The provider's prepared variant (RFC-0040 §6.5 `SelectionEntry::variant`).
    pub variant: u32,
    /// What `query` returned; `None` when it does not run the op at this shape.
    pub caps: Option<OpCaps>,
}

impl Candidate {
    /// Whether the provider is native, that is, not ggml (ADR-0002: ties go to native).
    #[must_use]
    pub fn is_native(&self) -> bool {
        !matches!(self.backend, Backend::Ggml(_))
    }

    fn id(&self) -> (ProviderId, u32) {
        (self.provider, self.variant)
    }
}

/// What every candidate for an op on a device must meet (RFC-0040 §6.2 conditions 1–4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Requirements {
    /// The device's domain, from [`choose_domain`].
    pub domain: MemoryDomain,
    /// The layout of the copy this reader reads, or [`LayoutVersion::SOURCE`] when the pack
    /// keeps the source layout. `None` in the layout pass (RFC-0040 §6.1), which benchmarks
    /// every layout.
    pub layout: Option<LayoutVersion>,
    /// The runner was opened deterministic (ADR-0019 decision 8).
    pub deterministic: bool,
}

/// Why a candidate is not admitted (RFC-0040 §6.2).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rejection {
    /// The provider belongs to another memory domain (RFC-0040 §2.3).
    OtherDomain(MemoryDomain),
    /// `query` returned `None` (condition 1).
    NoQuery,
    /// `caps.numerics` breaks the accumulation contract (condition 2, RFC-0042 §4).
    Numerics(NumericsViolation),
    /// `caps.numerics.act_in` breaks the activation quantisation rule for the op's weight
    /// tensor (condition 2, RFC-0042 §4.3).
    Activation(ActViolation),
    /// `caps.layout` is not the layout the reader reads (condition 3).
    Layout {
        /// The layout the reader reads.
        wanted: LayoutVersion,
        /// The layout the candidate reads.
        offered: LayoutVersion,
    },
    /// The runner is deterministic and the candidate is not (condition 4).
    NotDeterministic,
}

impl fmt::Display for Rejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OtherDomain(d) => write!(f, "in memory domain {d:?}, not the device's"),
            Self::NoQuery => write!(f, "query returned None"),
            Self::Numerics(v) => write!(f, "numerics: {v}"),
            Self::Activation(v) => write!(f, "activation quantisation: {v}"),
            Self::Layout { wanted, offered } => {
                write!(
                    f,
                    "reads layout {}, the reader reads {}",
                    offered.0, wanted.0
                )
            }
            Self::NotDeterministic => write!(f, "not deterministic"),
        }
    }
}

/// Holds `candidate` to RFC-0040 §6.2 conditions 1–4 for `key`, in that order, after the
/// domain check. `act` describes the op's weight tensor for the activation rule of
/// condition 2.
///
/// # Errors
///
/// The first [`Rejection`] found.
pub fn admit(
    key: &OpKey,
    req: &Requirements,
    act: &ActContext,
    candidate: &Candidate,
) -> Result<OpCaps, Rejection> {
    if candidate.domain != req.domain {
        return Err(Rejection::OtherDomain(candidate.domain));
    }
    let caps = candidate.caps.ok_or(Rejection::NoQuery)?;
    numerics::check_caps(key, &caps).map_err(Rejection::Numerics)?;
    act::check_act(key, &caps, act).map_err(Rejection::Activation)?;
    if let Some(wanted) = req.layout
        && caps.layout != wanted
    {
        return Err(Rejection::Layout {
            wanted,
            offered: caps.layout,
        });
    }
    if req.deterministic && !caps.deterministic {
        return Err(Rejection::NotDeterministic);
    }
    Ok(caps)
}

// ---- The per-op rule -----------------------------------------------------------------------

/// The per-device cost terms of RFC-0040 §6.4, measured by the probe (RFC-0040 §6.6).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CostTerms {
    /// One submission: the cost of a forced graph segment break.
    pub t_submit: Nanos,
    /// One cross-stream event: the cost of a forced fence.
    pub t_fence: Nanos,
}

/// A candidate's microbenchmark result at the plan shape (RFC-0040 §6.3), with what it
/// forces on the step program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Measurement {
    /// The median measured time.
    pub measured: Nanos,
    /// Graph segment breaks the candidate forces where it sits in the program. Only a
    /// candidate that is not `capture_safe` forces any; for one that is, this is ignored.
    pub breaks: u32,
    /// Cross-stream events it forces (unpatched ggml backends, RFC-0040 §8.4).
    pub fences: u32,
}

impl Measurement {
    /// `measured + breaks · t_submit + fences · t_fence`, saturating, with `breaks` taken as
    /// zero when `capture_safe`.
    #[must_use]
    pub fn cost(&self, capture_safe: bool, terms: &CostTerms) -> Nanos {
        let breaks = if capture_safe {
            0
        } else {
            u64::from(self.breaks)
        };
        let total = self
            .measured
            .0
            .saturating_add(breaks.saturating_mul(terms.t_submit.0))
            .saturating_add(u64::from(self.fences).saturating_mul(terms.t_fence.0));
        Nanos(total)
    }
}

/// A candidate with its measurement, if it was benchmarked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Offer {
    /// The candidate.
    pub candidate: Candidate,
    /// Its measurement; `None` when the benchmark budget ran out (RFC-0040 §6.3).
    pub measurement: Option<Measurement>,
}

/// One (op, shape) occurrence of the stage's step program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpSlot {
    /// The op.
    pub key: OpKey,
    /// Its shape.
    pub shape: ShapeSig,
    /// Its share of the predicted step time: slots are decided in decreasing order of it.
    pub predicted: Nanos,
    /// The layout this reader reads; `None` in the layout pass.
    pub layout: Option<LayoutVersion>,
    /// What RFC-0042 §4.3's activation rule needs to know about the op's weight tensor:
    /// the declared activation quantisation, the census's transcription and the
    /// certificates. The default is nothing declared, no transcription and no certificate.
    pub act: ActContext,
    /// The cached choice from the selection cache (RFC-0040 §6.5), if any.
    pub cached: Option<(ProviderId, u32)>,
    /// Every candidate offered.
    pub offers: Vec<Offer>,
}

/// How the incumbent was found (RFC-0040 §6.4).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Incumbent {
    /// The native candidate.
    Native,
    /// The cached choice; no native candidate was admitted.
    Cached,
    /// The cheapest candidate; no native or cached candidate was admitted.
    Cheapest,
}

/// Why [`decide_op`] chose a candidate.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Basis {
    /// The incumbent kept its place.
    Incumbent(Incumbent),
    /// The challenger costs less than 97 % of the incumbent, which it replaced.
    Challenger {
        /// The incumbent it replaced.
        replaced: (ProviderId, u32),
    },
    /// Not every admitted candidate was benchmarked, so the providers' own `OpCaps::est`
    /// ordered them, ties to native (RFC-0040 §6.3 "Budget").
    Estimate,
}

/// The candidate chosen for one op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpChoice {
    /// The provider.
    pub provider: ProviderId,
    /// Its prepared variant.
    pub variant: u32,
    /// Whether it is native.
    pub native: bool,
    /// Its cost, when it was chosen by measurement.
    pub cost: Option<Nanos>,
    /// Why it was chosen.
    pub basis: Basis,
}

/// No candidate passed RFC-0040 §6.2 for an op: the op is unsupported on the device, and the
/// solver re-plans its compute site (RFC-0040 §11.3; code 1 `OpUnsupported`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unsupported {
    /// The slot's index in the step program.
    pub slot: usize,
    /// The op.
    pub key: OpKey,
    /// Its shape.
    pub shape: ShapeSig,
    /// Every candidate offered and why it was not admitted.
    pub rejections: Vec<(ProviderId, u32, Rejection)>,
}

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "op {:?} at slot {} has no candidate on this device",
            self.key.op, self.slot
        )?;
        for (provider, variant, why) in &self.rejections {
            write!(f, "; {}#{variant}: {why}", provider.0)?;
        }
        Ok(())
    }
}

impl std::error::Error for Unsupported {}

impl From<Box<Unsupported>> for Error {
    /// `OpUnsupported` (RFC-0040 §13 code 1, class `Unsupported`).
    fn from(u: Box<Unsupported>) -> Self {
        Self {
            class: ErrorClass::Unsupported,
            code: ErrorCode::OP_UNSUPPORTED,
            message: u.to_string(),
        }
    }
}

/// The total tie-break shared by every ordering: native first, then provider name, then
/// variant.
fn tie_break(a: &Candidate, b: &Candidate) -> Ordering {
    b.is_native()
        .cmp(&a.is_native())
        .then_with(|| a.provider.cmp(&b.provider))
        .then_with(|| a.variant.cmp(&b.variant))
}

/// The cheapest of `pool`, ties broken by [`tie_break`].
fn cheapest<'a>(
    pool: impl Iterator<Item = (&'a Candidate, Nanos)>,
) -> Option<(&'a Candidate, Nanos)> {
    pool.min_by(|(a, ca), (b, cb)| ca.cmp(cb).then_with(|| tie_break(a, b)))
}

/// Decides one op (RFC-0040 §6.4's inner loop; ADR-0401 item 4). `slot` is the index
/// reported in [`Unsupported`].
///
/// # Errors
///
/// [`Unsupported`], boxed, when no offer is admitted.
pub fn decide_op(
    op: &OpSlot,
    slot: usize,
    domain: MemoryDomain,
    deterministic: bool,
    terms: &CostTerms,
) -> Result<OpChoice, Box<Unsupported>> {
    let req = Requirements {
        domain,
        layout: op.layout,
        deterministic,
    };
    let mut admitted = Vec::new();
    let mut rejections = Vec::new();
    for offer in &op.offers {
        let c = &offer.candidate;
        match admit(&op.key, &req, &op.act, c) {
            Ok(caps) => admitted.push((c, caps, offer.measurement)),
            Err(why) => rejections.push((c.provider, c.variant, why)),
        }
    }
    if admitted.is_empty() {
        rejections.sort_by_key(|r| (r.0, r.1));
        return Err(Box::new(Unsupported {
            slot,
            key: op.key.clone(),
            shape: op.shape.clone(),
            rejections,
        }));
    }

    let choice = |c: &Candidate, cost, basis| OpChoice {
        provider: c.provider,
        variant: c.variant,
        native: c.is_native(),
        cost,
        basis,
    };

    // Budget ran out for this op: the providers' own estimates, missing ones last, ties to
    // native.
    if admitted.iter().any(|(_, _, m)| m.is_none()) {
        let (c, ..) = admitted
            .iter()
            .min_by(|(a, ca, _), (b, cb, _)| {
                let key = |caps: &OpCaps| (caps.est.is_none(), caps.est);
                key(ca).cmp(&key(cb)).then_with(|| tie_break(a, b))
            })
            .expect("non-empty");
        return Ok(choice(c, None, Basis::Estimate));
    }

    let scored: Vec<(&Candidate, Nanos)> = admitted
        .iter()
        .map(|(c, caps, m)| {
            let m = m.expect("all measured");
            (*c, m.cost(caps.capture_safe, terms))
        })
        .collect();
    let challenger = cheapest(scored.iter().copied()).expect("non-empty");
    let (incumbent, kind) =
        if let Some(native) = cheapest(scored.iter().copied().filter(|(c, _)| c.is_native())) {
            (native, Incumbent::Native)
        } else if let Some(cached) = op
            .cached
            .and_then(|id| scored.iter().find(|(c, _)| c.id() == id).copied())
        {
            (cached, Incumbent::Cached)
        } else {
            (challenger, Incumbent::Cheapest)
        };

    if challenger.0.id() != incumbent.0.id() && beats(challenger.1, incumbent.1, OP_MARGIN_PERCENT)
    {
        Ok(choice(
            challenger.0,
            Some(challenger.1),
            Basis::Challenger {
                replaced: incumbent.0.id(),
            },
        ))
    } else {
        Ok(choice(
            incumbent.0,
            Some(incumbent.1),
            Basis::Incumbent(kind),
        ))
    }
}

/// Every op of a stage's step program decided on one device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgramChoice {
    /// One choice per slot, in program order.
    pub choices: Vec<OpChoice>,
    /// The sum of the chosen costs, segment breaks and fences included: the domain's predicted
    /// step time for [`choose_domain`]. `None` when any op was chosen by estimate, because a
    /// sum without that op's time would understate the step.
    pub predicted_step: Option<Nanos>,
    /// How many ops were chosen by estimate rather than measurement.
    pub estimated: usize,
}

/// Decides every op of `program` on a device in `domain` (RFC-0040 §6.4's loop): in decreasing
/// order of [`OpSlot::predicted`], ties in program order, stopping at the first op with no
/// candidate.
///
/// # Errors
///
/// [`Unsupported`], boxed, for the first op, in that order, with no admitted candidate.
pub fn select_program(
    program: &[OpSlot],
    domain: MemoryDomain,
    deterministic: bool,
    terms: &CostTerms,
) -> Result<ProgramChoice, Box<Unsupported>> {
    let mut order: Vec<usize> = (0..program.len()).collect();
    order.sort_by(|&a, &b| {
        program[b]
            .predicted
            .cmp(&program[a].predicted)
            .then(a.cmp(&b))
    });
    let mut choices = vec![None; program.len()];
    for i in order {
        choices[i] = Some(decide_op(&program[i], i, domain, deterministic, terms)?);
    }
    let choices: Vec<OpChoice> = choices.into_iter().map(|c| c.expect("decided")).collect();
    let predicted_step = choices.iter().try_fold(Nanos(0), |sum, c| {
        c.cost.map(|cost| Nanos(sum.0.saturating_add(cost.0)))
    });
    let estimated = choices.iter().filter(|c| c.cost.is_none()).count();
    Ok(ProgramChoice {
        choices,
        predicted_step,
        estimated,
    })
}

#[cfg(test)]
mod tests;
