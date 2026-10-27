// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fault plans over the command channel (RFC-0060 §9.2; DIST-E1-T13): a test-only message from
//! the daemon to a runner, compiled into both ends only with the `fault-injection` feature.
//!
//! **The messages.** [`DaemonFrame`] is what a test daemon sends: any [`DaemonMsg`], or
//! [`DaemonFrame::SetFaultPlan`], a [`FaultPlan`] as TOML. [`RunnerFrame`] is what the runner
//! answers: any [`RunnerMsg`], or [`RunnerFrame::FaultPlanSet`], the install's result. The plan
//! travels as TOML, the form `FaultPlan::from_toml` reads and checks, because the plan's types
//! skip absent fields when they serialize, which postcard cannot read back.
//!
//! **The wire.** On the wire, each test message is one more variant of the production enum, at
//! the fixed index [`TEST_VARIANT`], 2²⁰, far above every production variant. Production variants
//! are numbered by their order, from 0, and new ones are appended, so this one never moves, and
//! never moves them. A frame of a production message is byte for byte the same in both builds.
//! A build without the feature has no `DaemonFrame` and no `RunnerFrame`, and its decoder reads
//! index 2²⁰ as an unknown variant: the frame is refused with [`FrameError::Malformed`]
//! (`super::FrameError::Malformed`).
//!
//! **The runner's side.** The runner (DIST-E1-T04, T05) receives on a [`FaultRunnerChannel`], and
//! hands each `SetFaultPlan` to [`install_fault_plan`] with its clock, then replies with the
//! result. An invalid plan is refused in the reply, and the plan installed before stays.

use std::fmt;
use std::sync::Arc;

use serde::de::value::U32Deserializer;
use serde::de::{
    self, DeserializeSeed, Deserializer, EnumAccess, IntoDeserializer, VariantAccess, Visitor,
};
use serde::{Deserialize, Serialize, Serializer};
use syntropy_core::Clock;
use syntropy_obs::faults::{FaultPlan, install};

use super::Channel;
use super::msg::{DaemonMsg, RunnerMsg};

/// The wire index of the test-only variants: 2²⁰, which no production variant reaches.
pub const TEST_VARIANT: u32 = 1 << 20;

/// What a test daemon sends a runner.
#[derive(Clone, Debug, PartialEq)]
pub enum DaemonFrame {
    /// A production message, encoded exactly as [`DaemonMsg`] encodes it.
    Msg(DaemonMsg),
    /// Installs this fault plan, as TOML, in the runner (RFC-0060 §9.2).
    SetFaultPlan(String),
}

/// What a runner answers a test daemon.
#[derive(Clone, Debug, PartialEq)]
pub enum RunnerFrame {
    /// A production message, encoded exactly as [`RunnerMsg`] encodes it.
    Msg(RunnerMsg),
    /// The result of a [`DaemonFrame::SetFaultPlan`]: `Ok`, or why the plan was refused. A
    /// refused plan leaves the plan installed before in place.
    FaultPlanSet(Result<(), String>),
}

/// A test daemon's end of a channel to a runner.
pub type FaultDaemonChannel = Channel<DaemonFrame, RunnerFrame>;

/// A runner's end of its channel to a test daemon.
pub type FaultRunnerChannel = Channel<RunnerFrame, DaemonFrame>;

impl DaemonFrame {
    /// The `SetFaultPlan` message for `plan`.
    ///
    /// # Errors
    ///
    /// The reason when the plan has no TOML form.
    pub fn set_fault_plan(plan: &FaultPlan) -> Result<DaemonFrame, String> {
        plan.to_toml()
            .map(DaemonFrame::SetFaultPlan)
            .map_err(|error| error.to_string())
    }
}

/// The runner's handler of [`DaemonFrame::SetFaultPlan`]: parses and checks `toml`, then installs
/// the plan with `syntropy_obs::faults::install` on `clock`. Its result is the reply. On an error
/// the plan installed before stays.
///
/// # Errors
///
/// Why the plan was refused: TOML that is not a plan, or a rule `FaultPlan::validate` rejects.
pub fn install_fault_plan(toml: &str, clock: Arc<dyn Clock>) -> Result<(), String> {
    let plan = FaultPlan::from_toml(toml).map_err(|error| error.to_string())?;
    install(plan, clock).map_err(|error| error.to_string())
}

impl Serialize for DaemonFrame {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            DaemonFrame::Msg(msg) => msg.serialize(serializer),
            DaemonFrame::SetFaultPlan(toml) => serializer.serialize_newtype_variant(
                "DaemonMsg",
                TEST_VARIANT,
                "SetFaultPlan",
                toml,
            ),
        }
    }
}

impl Serialize for RunnerFrame {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            RunnerFrame::Msg(msg) => msg.serialize(serializer),
            RunnerFrame::FaultPlanSet(result) => serializer.serialize_newtype_variant(
                "RunnerMsg",
                TEST_VARIANT,
                "FaultPlanSet",
                result,
            ),
        }
    }
}

impl<'de> Deserialize<'de> for DaemonFrame {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<DaemonFrame, D::Error> {
        deserializer.deserialize_enum("DaemonMsg", &[], DaemonVisitor)
    }
}

impl<'de> Deserialize<'de> for RunnerFrame {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<RunnerFrame, D::Error> {
        deserializer.deserialize_enum("RunnerMsg", &[], RunnerVisitor)
    }
}

/// Reads a daemon frame's variant index once: [`TEST_VARIANT`] decodes `SetFaultPlan`, and any
/// other index goes to [`DaemonMsg`]'s own decoder through [`Typed`].
struct DaemonVisitor;

impl<'de> Visitor<'de> for DaemonVisitor {
    type Value = DaemonFrame;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a daemon message")
    }

    fn visit_enum<A: EnumAccess<'de>>(self, data: A) -> Result<DaemonFrame, A::Error> {
        let (index, variant): (u32, A::Variant) = data.variant()?;
        if index == TEST_VARIANT {
            return variant.newtype_variant().map(DaemonFrame::SetFaultPlan);
        }
        DaemonMsg::deserialize(Typed { index, variant }).map(DaemonFrame::Msg)
    }
}

/// As [`DaemonVisitor`], for runner frames.
struct RunnerVisitor;

impl<'de> Visitor<'de> for RunnerVisitor {
    type Value = RunnerFrame;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a runner message")
    }

    fn visit_enum<A: EnumAccess<'de>>(self, data: A) -> Result<RunnerFrame, A::Error> {
        let (index, variant): (u32, A::Variant) = data.variant()?;
        if index == TEST_VARIANT {
            return variant.newtype_variant().map(RunnerFrame::FaultPlanSet);
        }
        RunnerMsg::deserialize(Typed { index, variant }).map(RunnerFrame::Msg)
    }
}

/// A deserializer that offers a production enum's derived decoder the variant index already read
/// and the decoder's own variant access, so the variant's fields decode exactly as they always do.
/// An index the enum lacks is its own "unknown variant" error.
struct Typed<V> {
    index: u32,
    variant: V,
}

impl<'de, Access: VariantAccess<'de>> Deserializer<'de> for Typed<Access> {
    type Error = Access::Error;

    fn deserialize_enum<Vis: Visitor<'de>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: Vis,
    ) -> Result<Vis::Value, Access::Error> {
        visitor.visit_enum(self)
    }

    fn deserialize_any<Vis: Visitor<'de>>(
        self,
        _visitor: Vis,
    ) -> Result<Vis::Value, Access::Error> {
        Err(de::Error::custom("a command message is an enum"))
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string bytes byte_buf
        option unit unit_struct newtype_struct seq tuple tuple_struct map struct identifier
        ignored_any
    }
}

impl<'de, V: VariantAccess<'de>> EnumAccess<'de> for Typed<V> {
    type Error = V::Error;
    type Variant = V;

    fn variant_seed<S: DeserializeSeed<'de>>(self, seed: S) -> Result<(S::Value, V), V::Error> {
        let index: U32Deserializer<V::Error> = self.index.into_deserializer();
        Ok((seed.deserialize(index)?, self.variant))
    }
}
