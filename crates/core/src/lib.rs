// SPDX-License-Identifier: MIT OR Apache-2.0

//! Identifiers, units, plan types, policy objects and the pure functions over them.
//!
//! Layer 0 in RFC-0001 §4.1, designed in RFC-0001 §5.

#![forbid(unsafe_code)]

pub mod budget;
pub mod build;
pub mod canon;
pub mod clock;
pub mod det_hash;
pub mod dtype;
pub mod entropy;
pub mod error;
pub mod fetch;
pub mod fidelity;
pub mod ids;
pub mod ledger;
pub mod measured;
pub mod numerics;
pub mod parity;
pub mod plan;
pub mod policy;
pub mod prefix;
pub mod pressure;
pub mod protocol;
pub mod quant;
pub mod replan;
pub mod resource;
pub mod runner;
pub mod sequence;
pub mod session;
pub mod tick;
pub mod transfer;
pub mod units;

pub use build::{BuildId, BuildProfile, SemVer, SemVerError, TargetTriple};
pub use clock::{AwakeClock, AwakeTime, Clock, Interval, MonoTime, SuspendDetector};
pub use det_hash::{DetHashMap, DetHashSet, DetHasher, DetState};
pub use dtype::{DType, GgmlType};
pub use entropy::Entropy;
pub use error::{
    Domain, Error, ErrorClass, ErrorCode, ErrorScope, MAX_SOURCE_DEPTH, RetryHint, SOURCES_DROPPED,
};
pub use fetch::FetchPriority;
pub use ids::{
    Blake3, DeviceId, Endpoint, ExpertIx, GraphVersion, InstanceId, LayerIx, LinkId, LinkKind,
    ModelId, NetMedium, NodeId, RdmaKind, StageIx, TensorGroupId, TierId,
};
pub use measured::{DerivedFrom, MeasureSource, Measured};
pub use numerics::{AccumKind, ActQuant, Numerics};
pub use parity::{ForceMode, LogitTap, ParityTapSpec, TapMode, TapRow, TapValues, TeacherStreamId};
pub use plan::{PlacementPlan, PlanDelta, PlanDigest, PlanEpoch, PlanViolation};
pub use policy::{
    ApproxFlags, ApproxKnobs, ApproxParams, ExactnessPolicy, Fidelity, InstancePolicy,
    NetworkScope, Objective, ResourceOverlay, ResourcePolicy,
};
pub use prefix::{KvSettings, MediaSpan, PrefixKeyChain, PrefixScopeMode, RopeSettings};
pub use pressure::PressureLevel;
pub use protocol::{Outbox, ProtocolCore};
pub use quant::{CtScheme, DTypeClass, ModelClassError, QuantScheme, ScaleGranularity};
pub use replan::{DriftTerm, ReplanReason, WorkloadTerm};
pub use runner::{
    Deadline, Degradation, DegradeDetail, DegradeFlags, DrainReason, RunnerState, RunnerStateKind,
    StopReason, WarmStep,
};
pub use sequence::{
    CancelReason, DraftNode, FinishReason, GrammarRef, LogprobBlock, PoolingMode, PrefixKey,
    PrefixRef, PrefixScope, Priority, ResumeCause, ResumePoint, SamplingParams, SequenceControl,
    SequenceLimits, SequenceMeta, SequenceOpen, SequenceOutcome, TailId, TokenEntry, TokenFrame,
    TokenSlice, TraceContext,
};
pub use session::{ParseSessionIdError, SequenceId, SessionId};
pub use tick::{DeviceClockSync, TickRate, TickSource};
pub use transfer::{BufferView, ExtentRef, FileIx, HostRegion, StepLayer, ViewKind};
pub use units::{Bytes, BytesPerSec, FlopsPerSec, Nanos};
