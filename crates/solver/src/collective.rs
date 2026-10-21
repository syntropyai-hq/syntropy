// SPDX-License-Identifier: MIT OR Apache-2.0

//! The collective cost model: how long one all-reduce takes, which algorithm runs it, and how
//! many messages expert parallelism sends per layer (RFC-0032 §5.3–§5.5; ADR-0310).
//!
//! Pure and I/O-free: plain arithmetic over measured inputs, so the solver can price `IntraTp`
//! placement nodes (PLC-E4-T09) and cross-node islands without the network crates.
//!
//! **The per-message cost α.** RFC-0032 §5.3's formulas take a size-independent per-message
//! cost α and add the size term S/β themselves. ADR-0310's `α_eff` (its 2026-10-19 revision) is
//! measured by RFC-0010's L2 probe at the `h_model × 4 B` partial with S/β included, so the
//! inputs here are α, and [`alpha_from_probe`] derives it as `α = α_eff,probe − S₀/β`
//! (RFC-0032 §5.3, note of 2026-10-22). At S = S₀ on a full mesh, [`one_shot`] then gives back
//! `α_eff,probe` exactly.
//!
//! **The formulas** (RFC-0032 §5.3), for N members and a partial of S bytes:
//!
//! | Algorithm | Time per all-reduce |
//! |---|---|
//! | `OneShot`, full mesh of point-to-point links | `α + S/β` |
//! | `OneShot`, the N−1 sends sharing one NIC | `α + (N−1)·S/β` |
//! | `Ring`, reduce-scatter then all-gather | `2(N−1)·α + 2(N−1)/N · S/β` |
//!
//! [`choose`] picks `OneShot` when N ≤ 4 and S ≤ `tp.oneshot_max` (1 MiB by default), and `Ring`
//! otherwise (ADR-0310 decision 1).
//!
//! **Intra-node** (RFC-0032 §5.5, note of 2026-10-22): the same formulas, with the device pair's
//! measured α and β over the provider's `CollectiveOps` on a verified peer-to-peer link
//! ([`LinkCost::peer_to_peer`]). A pair that fails the verified test copy reduces through pinned
//! host memory, so each partial is copied twice and S/β becomes `S/β_d2h + S/β_h2d`
//! ([`LinkCost::host_staged`]).
//!
//! **Expert parallelism** (RFC-0032 §5.4): at most two messages per remote owner touched per MoE
//! layer, a dispatch and a combine ([`ep_messages_per_layer`]).
//!
//! Times are computed in F64 and rounded to the nearest nanosecond. Fewer than two members need
//! no collective, so they cost nothing.

use std::fmt;

use syntropy_core::{Bytes, BytesPerSec, Nanos};

/// The most members `OneShot` runs with (ADR-0310 decision 1; RFC-0032 §5.3).
pub const ONESHOT_MAX_MEMBERS: u32 = 4;

/// The default of `tp.oneshot_max`, the largest partial `OneShot` runs with: 1 MiB
/// (RFC-0032 §5.3 and its configuration table).
pub const ONESHOT_MAX_DEFAULT: Bytes = Bytes(1 << 20);

const NANOS_PER_SEC: f64 = 1e9;

/// An all-reduce algorithm (RFC-0032 §5.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Algorithm {
    /// Each member sends its partial to every other member, and each sums all N in member order.
    OneShot,
    /// A reduce-scatter then an all-gather, in 2(N−1) steps of S/N.
    Ring,
}

/// How `OneShot`'s N−1 sends leave a member (RFC-0032 §5.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fabric {
    /// A full mesh of point-to-point links: the N−1 sends go out in parallel on separate links.
    FullMesh,
    /// The N−1 sends share one NIC, so they go out one after another.
    SharedNic,
}

/// How a partial's bytes cross between two members.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Path {
    /// One transfer at rate β: a network link, or a verified peer-to-peer copy.
    Direct { beta: f64 },
    /// Through pinned host memory: a device-to-host copy, then a host-to-device copy.
    HostStaged { d2h: f64, h2d: f64 },
}

/// The cost of moving a partial between two members: the size-independent per-message cost α
/// and the rate or rates its bytes move at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinkCost {
    alpha: Nanos,
    path: Path,
}

/// Why a [`LinkCost`] could not be built.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CostError {
    /// A rate is zero, negative, NaN or infinite.
    InvalidRate {
        /// Which rate.
        which: &'static str,
        /// Its value, in bytes per second.
        value: f64,
    },
}

impl fmt::Display for CostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CostError::InvalidRate { which, value } => {
                write!(
                    f,
                    "{which} must be a finite rate above zero, not {value} B/s"
                )
            }
        }
    }
}

impl std::error::Error for CostError {}

fn rate(which: &'static str, beta: BytesPerSec) -> Result<f64, CostError> {
    if beta.0.is_finite() && beta.0 > 0.0 {
        Ok(beta.0)
    } else {
        Err(CostError::InvalidRate {
            which,
            value: beta.0,
        })
    }
}

impl LinkCost {
    /// A network link between nodes: per-message cost `alpha` and effective rate `beta`.
    ///
    /// # Errors
    ///
    /// [`CostError::InvalidRate`] when `beta` is not a finite rate above zero.
    pub fn network(alpha: Nanos, beta: BytesPerSec) -> Result<LinkCost, CostError> {
        Ok(LinkCost {
            alpha,
            path: Path::Direct {
                beta: rate("beta", beta)?,
            },
        })
    }

    /// A verified peer-to-peer link between two devices of one node, over the provider's
    /// `CollectiveOps` (RFC-0032 §5.5): the pair's measured per-message cost and rate.
    ///
    /// # Errors
    ///
    /// [`CostError::InvalidRate`] when `beta` is not a finite rate above zero.
    pub fn peer_to_peer(alpha: Nanos, beta: BytesPerSec) -> Result<LinkCost, CostError> {
        LinkCost::network(alpha, beta)
    }

    /// Two devices of one node that reduce through pinned host memory, because their
    /// peer-to-peer test copy failed (RFC-0032 §5.5): each partial is copied to the host at
    /// `d2h` and then to the other device at `h2d`.
    ///
    /// # Errors
    ///
    /// [`CostError::InvalidRate`] when either rate is not a finite rate above zero.
    pub fn host_staged(
        alpha: Nanos,
        d2h: BytesPerSec,
        h2d: BytesPerSec,
    ) -> Result<LinkCost, CostError> {
        Ok(LinkCost {
            alpha,
            path: Path::HostStaged {
                d2h: rate("d2h", d2h)?,
                h2d: rate("h2d", h2d)?,
            },
        })
    }

    /// The size-independent per-message cost α.
    #[must_use]
    pub fn alpha(&self) -> Nanos {
        self.alpha
    }

    /// The time `bytes` take to cross, in seconds: `S/β`, or `S/β_d2h + S/β_h2d` when staged
    /// through host memory.
    fn transfer_secs(&self, bytes: f64) -> f64 {
        match self.path {
            Path::Direct { beta } => bytes / beta,
            Path::HostStaged { d2h, h2d } => bytes / d2h + bytes / h2d,
        }
    }
}

/// The size-independent per-message cost α from ADR-0310's measured `α_eff`: `α_eff,probe` is
/// the per-collective cost RFC-0010's L2 probe measures at the `probe_partial` (`h_model × 4 B`)
/// with S/β included, so `α = α_eff,probe − S₀/β` (RFC-0032 §5.3, note of 2026-10-22). A probe
/// that measured less than S₀/β gives α = 0.
///
/// # Errors
///
/// [`CostError::InvalidRate`] when `beta` is not a finite rate above zero.
pub fn alpha_from_probe(
    alpha_eff: Nanos,
    probe_partial: Bytes,
    beta: BytesPerSec,
) -> Result<Nanos, CostError> {
    let size_term = probe_partial.0 as f64 / rate("beta", beta)? * NANOS_PER_SEC;
    Ok(round(alpha_eff.0 as f64 - size_term))
}

/// The algorithm for an all-reduce of `members` members over a partial of `partial` bytes:
/// `OneShot` when `members` ≤ 4 and `partial` ≤ `oneshot_max`, `Ring` otherwise (ADR-0310
/// decision 1; RFC-0032 §5.3). `oneshot_max` is `tp.oneshot_max`, [`ONESHOT_MAX_DEFAULT`] by
/// default.
#[must_use]
pub fn choose(members: u32, partial: Bytes, oneshot_max: Bytes) -> Algorithm {
    if members <= ONESHOT_MAX_MEMBERS && partial <= oneshot_max {
        Algorithm::OneShot
    } else {
        Algorithm::Ring
    }
}

fn round(nanos: f64) -> Nanos {
    // `as` saturates: a negative value becomes 0 and an overflow becomes u64::MAX.
    Nanos(nanos.round() as u64)
}

/// `OneShot`'s time per all-reduce: `α + S/β` on a full mesh, `α + (N−1)·S/β` when the N−1 sends
/// share one NIC (RFC-0032 §5.3).
#[must_use]
pub fn one_shot(members: u32, partial: Bytes, link: &LinkCost, fabric: Fabric) -> Nanos {
    if members < 2 {
        return Nanos(0);
    }
    let sends = match fabric {
        Fabric::FullMesh => 1.0,
        Fabric::SharedNic => f64::from(members - 1),
    };
    let transfer = link.transfer_secs(partial.0 as f64) * NANOS_PER_SEC;
    round(link.alpha.0 as f64 + sends * transfer)
}

/// `Ring`'s time per all-reduce: `2(N−1)·α + 2(N−1)/N · S/β` (RFC-0032 §5.3).
#[must_use]
pub fn ring(members: u32, partial: Bytes, link: &LinkCost) -> Nanos {
    if members < 2 {
        return Nanos(0);
    }
    let n = f64::from(members);
    let steps = 2.0 * (n - 1.0);
    let transfer = link.transfer_secs(partial.0 as f64) * NANOS_PER_SEC;
    round(steps * link.alpha.0 as f64 + steps / n * transfer)
}

/// One all-reduce: the algorithm [`choose`] picks, and its time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllReduce {
    /// The algorithm that runs it.
    pub algorithm: Algorithm,
    /// Its time.
    pub time: Nanos,
}

/// The algorithm and time of one all-reduce: [`choose`], then [`one_shot`] or [`ring`]. `fabric`
/// applies only when `OneShot` runs.
#[must_use]
pub fn all_reduce(
    members: u32,
    partial: Bytes,
    link: &LinkCost,
    fabric: Fabric,
    oneshot_max: Bytes,
) -> AllReduce {
    let algorithm = choose(members, partial, oneshot_max);
    let time = match algorithm {
        Algorithm::OneShot => one_shot(members, partial, link, fabric),
        Algorithm::Ring => ring(members, partial, link),
    };
    AllReduce { algorithm, time }
}

/// Expert parallelism's messages per MoE layer: an `EpDispatch` and an `EpCombine` for each
/// remote owner with at least one selected expert, so `2 × owners_touched` at most
/// (RFC-0032 §5.4).
#[must_use]
pub fn ep_messages_per_layer(owners_touched: u32) -> u32 {
    owners_touched.saturating_mul(2)
}

#[cfg(test)]
mod tests;
