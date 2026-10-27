// SPDX-License-Identifier: MIT OR Apache-2.0

//! The error classes and codes of RFC-0001 §5.12. RFC-0001 §10 defines what each class
//! triggers.
//!
//! - [`Error`], the one error every fallible interface returns, with its [`ErrorScope`],
//!   [`RetryHint`], context and source chain (COMP-E13-T15).
//! - [`ErrorClass`], which `DeviceProblem::ProbeCrashed` also records (RFC-0010 §1.2).
//! - [`ErrorCode`], the `(Domain, u16)` pair.
//! - The code tables of the domains whose RFCs publish them ([`codes`]).
//! - The map from each code to its RFC-0050 §12 API error shape ([`api`]).

use std::borrow::Cow;
use std::fmt;
use std::ops::RangeInclusive;

use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use smallvec::SmallVec;

use crate::ids::{LinkId, NodeId, StageIx};
use crate::session::SequenceId;
use crate::units::Nanos;

pub mod api;
pub mod codes;

/// What kind of failure an error is (RFC-0001 §5.12). Owner: RFC-0001.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ErrorClass {
    /// Bad client input.
    InvalidRequest,
    /// Op, dtype or feature missing on this device or build.
    Unsupported,
    /// VRAM, pinned RAM, disk, KV pool or admission limit.
    Capacity,
    /// Momentary: would-block, `EAGAIN`, one lost control message.
    Transient,
    /// Transport or network failure on one link.
    Link,
    /// Device fault or sticky driver error.
    Device,
    /// Hash mismatch, truncated or corrupt pack or chunk.
    Integrity,
    /// Parity or canary failure.
    Correctness,
    /// Measured costs no longer match the plan's prediction.
    Drift,
    /// Deadline passed or progress stalled.
    Timeout,
    /// Admission, credential or signature failure.
    Auth,
    /// Client or system cancellation.
    Cancelled,
    /// A bug.
    Internal,
}

/// The area that publishes a code (RFC-0001 §5.12, §10.4). Each domain RFC owns the codes in
/// its range. Other domains are added with the tasks that build their tables. Owner: RFC-0001.
///
/// Variants are appended, so `Ord` and the postcard index follow declaration order, not
/// [`range`](Domain::range) order: `Kernels` (0x0001–0x00FF) sorts after `Security`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Domain {
    /// Discovery, membership and invites: 0x0100–0x01FF (RFC-0030 §10).
    Cluster,
    /// Transport and wire protocol: 0x0200–0x02FF (RFC-0031 §12).
    Net,
    /// Distributed execution and the cluster event log: 0x0300–0x03FF (RFC-0032 §10).
    Dist,
    /// Fault tolerance and elasticity: 0x0400–0x04FF (RFC-0033 §13).
    Recovery,
    /// Security and trust: 0x0500–0x05FF (RFC-0034 §15).
    Security,
    /// Kernel providers and backends: 0x0001–0x00FF, so RFC-0040 §13's codes 1–17 keep their
    /// published numbers (RFC-0040 §13, note of 2026-10-28).
    Kernels,
    /// The faults `exec` detects in its dispatchers and CPU pools: 0x0600–0x06FF (RFC-0012
    /// §13, note of 2026-10-28).
    Exec,
}

impl Domain {
    /// The code range the domain's RFC publishes.
    pub const fn range(self) -> RangeInclusive<u16> {
        match self {
            Self::Cluster => 0x0100..=0x01FF,
            Self::Net => 0x0200..=0x02FF,
            Self::Dist => 0x0300..=0x03FF,
            Self::Recovery => 0x0400..=0x04FF,
            Self::Security => 0x0500..=0x05FF,
            Self::Kernels => 0x0001..=0x00FF,
            Self::Exec => 0x0600..=0x06FF,
        }
    }

    /// The RFC section that publishes the domain's codes.
    pub const fn source(self) -> &'static str {
        match self {
            Self::Cluster => "RFC-0030 §10",
            Self::Net => "RFC-0031 §12",
            Self::Dist => "RFC-0032 §10",
            Self::Recovery => "RFC-0033 §13",
            Self::Security => "RFC-0034 §15",
            Self::Kernels => "RFC-0040 §13",
            Self::Exec => "RFC-0012 §13",
        }
    }
}

/// A stable, machine-readable error code: `(Domain, u16)` (RFC-0001 §5.12). Codes are never
/// reused (RFC-0001 §10.4); [`codes::REGISTRY`] lists the published ones. Owner: RFC-0001
/// (the pair), each domain RFC (its codes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ErrorCode {
    /// The domain that publishes the code.
    pub domain: Domain,
    /// The code inside the domain, as its RFC writes it.
    pub code: u16,
}

impl ErrorCode {
    /// The code `code` of `domain`.
    pub const fn new(domain: Domain, code: u16) -> Self {
        Self { domain, code }
    }
}

/// Shows the domain and the code as its RFC writes it, e.g. `Net 0x0207`.
impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} 0x{:04X}", self.domain, self.code)
    }
}

/// What an error is about (RFC-0001 §5.12): the sequence, stage, runner, node, link,
/// instance or cluster it affects. Owner: RFC-0001.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorScope {
    /// One sequence; the rest of the instance is unaffected.
    Sequence(SequenceId),
    /// One stage of the instance.
    Stage(StageIx),
    /// The runner process that raised it.
    Runner,
    /// One node.
    Node(NodeId),
    /// One link.
    Link(LinkId),
    /// The whole instance.
    Instance,
    /// The whole cluster.
    Cluster,
}

/// Whether and when the operation may be tried again (RFC-0001 §5.12). Owner: RFC-0001.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RetryHint {
    /// Retrying cannot help.
    Never,
    /// Retry after this long.
    After(Nanos),
    /// Retry once a new plan is committed.
    OnNewPlan,
    /// Retry with the caller's back-off.
    Backoff,
}

/// The error every fallible interface returns (RFC-0001 §5.12). Owner: RFC-0001 (classes and
/// propagation), each domain RFC (the codes in its domain).
///
/// It is owned data: `context`'s keys are `Cow<'static, str>` (RFC-0001 §18.32), so code writes
/// literal keys without allocating, and decoding one needs no buffer that lives for the whole
/// program. `Error` crosses the IPC in `ErrorReport` (RFC-0001 §3.5), so its decode refuses a
/// `source` chain deeper than [`MAX_SOURCE_DEPTH`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Error {
    /// What kind of failure; RFC-0001 §10 says what each class triggers.
    pub class: ErrorClass,
    /// The stable, machine-readable code.
    pub code: ErrorCode,
    /// For people; never parsed.
    pub message: String,
    /// What the error is about.
    pub scope: ErrorScope,
    /// Whether and when to retry.
    pub retry: RetryHint,
    /// Key and value pairs for people and diagnostics, such as the op or the device.
    pub context: SmallVec<[(Cow<'static, str>, String); 4]>,
    /// The error this one wraps, if any.
    pub source: Option<Box<Error>>,
}

impl Error {
    /// An error of `class` with `code` and `message`, scoped to the runner that raises it,
    /// with no retry, context or source. The `with_*` methods set the rest.
    #[must_use]
    pub fn new(class: ErrorClass, code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            class,
            code,
            message: message.into(),
            scope: ErrorScope::Runner,
            retry: RetryHint::Never,
            context: SmallVec::new(),
            source: None,
        }
    }

    /// The error with its scope set to `scope`.
    #[must_use]
    pub fn with_scope(mut self, scope: ErrorScope) -> Self {
        self.scope = scope;
        self
    }

    /// The error with its retry hint set to `retry`.
    #[must_use]
    pub fn with_retry(mut self, retry: RetryHint) -> Self {
        self.retry = retry;
        self
    }

    /// The error with `(key, value)` appended to its context.
    #[must_use]
    pub fn with_context(
        mut self,
        key: impl Into<Cow<'static, str>>,
        value: impl Into<String>,
    ) -> Self {
        self.context.push((key.into(), value.into()));
        self
    }

    /// The error wrapping `source`. The chain stays within [`MAX_SOURCE_DEPTH`] sources, so a
    /// receiver never refuses an error its sender built: past it, the deepest sources are
    /// dropped, and the deepest one kept gains the context entry `sources_dropped` with how
    /// many went, counting any that an earlier cut dropped.
    #[must_use]
    pub fn with_source(mut self, mut source: Error) -> Self {
        // `source` is the first of this error's sources; keep it and the 15 below it.
        let mut kept = &mut source;
        for _ in 1..MAX_SOURCE_DEPTH {
            match kept.source {
                Some(ref mut next) => kept = next,
                None => break,
            }
        }
        if let Some(tail) = kept.source.take() {
            let mut dropped = 0u64;
            let mut next = Some(*tail);
            // Unlinks the tail one error at a time, so a long tail is not dropped by recursion.
            // The counts may come from a decoded chain, so they are untrusted: the sums saturate.
            while let Some(mut e) = next {
                dropped = dropped.saturating_add(1).saturating_add(dropped_count(&e));
                next = e.source.take().map(|b| *b);
            }
            dropped = dropped.saturating_add(dropped_count(kept));
            kept.context.retain(|(k, _)| k != SOURCES_DROPPED);
            kept.context
                .push((Cow::Borrowed(SOURCES_DROPPED), dropped.to_string()));
        }
        self.source = Some(Box::new(source));
        self
    }

    /// The value of the first context entry named `key`.
    #[must_use]
    pub fn context(&self, key: &str) -> Option<&str> {
        self.context
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// This error, then its source, then that one's source, and so on.
    pub fn chain(&self) -> impl Iterator<Item = &Error> {
        std::iter::successors(Some(self), |e| e.source.as_deref())
    }
}

/// The count an earlier cut recorded on `e` under [`SOURCES_DROPPED`], or 0.
fn dropped_count(e: &Error) -> u64 {
    e.context(SOURCES_DROPPED)
        .and_then(|n| n.parse::<u64>().ok())
        .unwrap_or(0)
}

/// Shows the message, then the class and the code, e.g.
/// `not supported by this provider (Unsupported, Kernels 0x0001)`.
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({:?}, {})", self.message, self.class, self.code)
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_deref().map(|e| e as _)
    }
}

/// The deepest `source` chain a decode accepts: the outermost error and 16 sources. A decoded
/// `Error` comes from another process or node (`ErrorReport`, RFC-0001 §3.5), and a nested
/// chain decodes by recursion, so a forged frame could otherwise exhaust the stack.
pub const MAX_SOURCE_DEPTH: usize = 16;

/// The context key [`Error::with_source`] records dropped sources under.
pub const SOURCES_DROPPED: &str = "sources_dropped";

/// Decodes as the derived form would, field by field, but refuses a `source` chain deeper than
/// [`MAX_SOURCE_DEPTH`].
impl<'de> Deserialize<'de> for Error {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        ErrorSeed { depth: 0 }.deserialize(deserializer)
    }
}

const FIELDS: &[&str] = &[
    "class", "code", "message", "scope", "retry", "context", "source",
];

/// Decodes one `Error` at `depth` sources below the outermost.
struct ErrorSeed {
    depth: usize,
}

impl<'de> DeserializeSeed<'de> for ErrorSeed {
    type Value = Error;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Error, D::Error> {
        if self.depth > MAX_SOURCE_DEPTH {
            return Err(de::Error::custom(format_args!(
                "an error's source chain is deeper than {MAX_SOURCE_DEPTH}"
            )));
        }
        deserializer.deserialize_struct("Error", FIELDS, ErrorVisitor { depth: self.depth })
    }
}

/// Decodes an `Error`'s `source`: `None`, or one error a level deeper.
struct SourceSeed {
    depth: usize,
}

impl<'de> DeserializeSeed<'de> for SourceSeed {
    type Value = Option<Box<Error>>;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_option(self)
    }
}

impl<'de> Visitor<'de> for SourceSeed {
    type Value = Option<Box<Error>>;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("an optional error")
    }

    fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        ErrorSeed {
            depth: self.depth + 1,
        }
        .deserialize(deserializer)
        .map(|e| Some(Box::new(e)))
    }
}

/// `Error`'s field names, for self-describing formats. An unknown field is skipped, as the
/// derived form does.
#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "lowercase")]
enum Field {
    Class,
    Code,
    Message,
    Scope,
    Retry,
    Context,
    Source,
    #[serde(other)]
    Other,
}

struct ErrorVisitor {
    depth: usize,
}

impl<'de> Visitor<'de> for ErrorVisitor {
    type Value = Error;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("struct Error")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Error, A::Error> {
        let missing = |n| de::Error::invalid_length(n, &"struct Error with 7 elements");
        Ok(Error {
            class: seq.next_element()?.ok_or_else(|| missing(0))?,
            code: seq.next_element()?.ok_or_else(|| missing(1))?,
            message: seq.next_element()?.ok_or_else(|| missing(2))?,
            scope: seq.next_element()?.ok_or_else(|| missing(3))?,
            retry: seq.next_element()?.ok_or_else(|| missing(4))?,
            context: seq.next_element()?.ok_or_else(|| missing(5))?,
            source: seq
                .next_element_seed(SourceSeed { depth: self.depth })?
                .ok_or_else(|| missing(6))?,
        })
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Error, A::Error> {
        let mut class = None;
        let mut code = None;
        let mut message = None;
        let mut scope = None;
        let mut retry = None;
        let mut context = None;
        let mut source = None;
        fn set<T, E: de::Error>(slot: &mut Option<T>, name: &'static str, v: T) -> Result<(), E> {
            if slot.replace(v).is_some() {
                return Err(E::duplicate_field(name));
            }
            Ok(())
        }
        while let Some(key) = map.next_key::<Field>()? {
            match key {
                Field::Class => set(&mut class, "class", map.next_value()?)?,
                Field::Code => set(&mut code, "code", map.next_value()?)?,
                Field::Message => set(&mut message, "message", map.next_value()?)?,
                Field::Scope => set(&mut scope, "scope", map.next_value()?)?,
                Field::Retry => set(&mut retry, "retry", map.next_value()?)?,
                Field::Context => set(&mut context, "context", map.next_value()?)?,
                Field::Source => set(
                    &mut source,
                    "source",
                    map.next_value_seed(SourceSeed { depth: self.depth })?,
                )?,
                Field::Other => {
                    map.next_value::<de::IgnoredAny>()?;
                }
            }
        }
        Ok(Error {
            class: class.ok_or_else(|| de::Error::missing_field("class"))?,
            code: code.ok_or_else(|| de::Error::missing_field("code"))?,
            message: message.ok_or_else(|| de::Error::missing_field("message"))?,
            scope: scope.ok_or_else(|| de::Error::missing_field("scope"))?,
            retry: retry.ok_or_else(|| de::Error::missing_field("retry"))?,
            context: context.ok_or_else(|| de::Error::missing_field("context"))?,
            source: source.unwrap_or(None),
        })
    }
}
