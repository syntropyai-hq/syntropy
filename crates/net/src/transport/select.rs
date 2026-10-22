// SPDX-License-Identifier: MIT OR Apache-2.0

//! Link selection at run time (RFC-0031 §3, DIST-E7-T11): connect a hop on its planned transport
//! when it can, and otherwise on the next candidate by measured cost, reported as a [`Demotion`].
//!
//! The cost function and the candidate order are `core`'s ([`link_cost`], [`candidate_order`]),
//! so the solver ranks a hop's transports the same way without depending on `net`.
//!
//! [`connect_hop`] runs in the `net` domain on whatever runtime drives it. It takes the clock as a
//! `sleep` function, so the runner's runtime and the simulator's virtual time can both run it, and
//! it tries transports of different backend types through [`AnyTransport`].

use std::future::{Future, poll_fn};
use std::pin::{Pin, pin};
use std::task::Poll;
use std::time::Duration;

use syntropy_core::resource::{Demotion, LinkCaps, TransportKind, candidate_order, link_cost};
use syntropy_core::{Bytes, LinkId, Nanos, Objective};

use super::{Channel, ChannelSpec, RemoteEndpoint, Transport};
use crate::error::Error;

/// A [`Transport`] behind a vtable, so that one hop's candidates, each a different backend type,
/// can be tried in turn. Every `Transport` is one. The boxed future is the only cost, and
/// connecting is never on a hot path.
pub trait AnyTransport: Send + Sync {
    /// See [`Transport::kind`].
    fn kind(&self) -> TransportKind;

    /// [`Transport::connect`], with its future boxed.
    fn connect_boxed<'a>(
        &'a self,
        remote: &'a RemoteEndpoint,
        spec: ChannelSpec,
    ) -> Pin<Box<dyn Future<Output = Result<Channel, Error>> + Send + 'a>>;
}

impl<T: Transport> AnyTransport for T {
    fn kind(&self) -> TransportKind {
        Transport::kind(self)
    }

    fn connect_boxed<'a>(
        &'a self,
        remote: &'a RemoteEndpoint,
        spec: ChannelSpec,
    ) -> Pin<Box<dyn Future<Output = Result<Channel, Error>> + Send + 'a>> {
        Box::pin(self.connect(remote, spec))
    }
}

/// What a hop needs to choose its transport (RFC-0031 §3).
#[derive(Clone, Copy, Debug)]
pub struct HopChoice<'a> {
    /// The hop's link in the plan (`HopPlan::link`): the edge a [`Demotion`] names.
    pub link: LinkId,
    /// The hop's transport in the plan (`HopPlan::transport`), tried first.
    pub planned: TransportKind,
    /// Every (link, transport) pair the two ends could use for the hop, with its probed costs.
    /// The fallback tries them in [`candidate_order`].
    pub candidates: &'a [LinkCaps],
    /// The decode frame size, `S_dec`: the size the candidates are ranked at and the new cost is
    /// reported at.
    pub frame: Bytes,
    /// The instance's objective, which sets λ.
    pub objective: Objective,
    /// `link.p99_weight` as configured; `None` for its default.
    pub p99_weight: Option<f64>,
    /// `link.connect_timeout`, resolved for the hop's link (2 s LAN, 10 s WAN): how long each
    /// attempt waits.
    pub connect_timeout: Duration,
}

/// A hop that [`connect_hop`] connected.
pub struct HopConnection {
    /// The open channel.
    pub channel: Channel,
    /// The transport it runs on.
    pub transport: TransportKind,
    /// `Some` when that is not the planned transport. It holds the demotion and the new
    /// transport's `cost_t` at the decode frame size, from its [`LinkCaps`] (measured, or the
    /// prior figures of one without a probe result). The runner reports both in `EdgeDemoted`
    /// with cause `DemotionCause::ConnectFailed` (RFC-0001 §3.5), and RFC-0033 §7.1's
    /// `TRANSPORT_DEMOTED` is set from the demotion (DIST-E14-T13).
    pub demoted: Option<(Demotion, Nanos)>,
}

/// Why one transport did not carry the hop.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AttemptFailure {
    /// This host has no backend for it.
    Unavailable,
    /// `link.connect_timeout` passed before it connected.
    TimedOut,
    /// Its channel spec could not be built, or the connect failed.
    Failed(Error),
}

/// No transport carried the hop: every attempt, in the order made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HopError {
    pub attempts: Vec<(TransportKind, AttemptFailure)>,
}

/// Connects one hop (RFC-0031 §3, runtime fallback).
///
/// The planned transport goes first. If this host has no backend for it, or it fails, or it does
/// not connect within `link.connect_timeout`, the other candidates follow in [`candidate_order`]:
/// the measured ones by `cost_t` at the decode frame size, then the unmeasured ones by rank. Each
/// transport kind is tried once. `transports` are this host's backends; `spec_for` builds the
/// channel spec for one of them, registering the hop's pool with it; `sleep` is the clock, a
/// future that completes after the duration.
pub async fn connect_hop<S, F>(
    choice: &HopChoice<'_>,
    transports: &[&dyn AnyTransport],
    remote: &RemoteEndpoint,
    spec_for: impl Fn(&dyn AnyTransport) -> Result<ChannelSpec, Error>,
    sleep: S,
) -> Result<HopConnection, HopError>
where
    S: Fn(Duration) -> F,
    F: Future<Output = ()>,
{
    let ranked = candidate_order(
        choice.candidates,
        choice.frame,
        &choice.objective,
        choice.p99_weight,
    );
    let mut order = vec![choice.planned];
    for caps in ranked {
        if !order.contains(&caps.transport) {
            order.push(caps.transport);
        }
    }
    let mut attempts = Vec::new();
    for kind in order {
        let Some(transport) = transports.iter().copied().find(|t| t.kind() == kind) else {
            attempts.push((kind, AttemptFailure::Unavailable));
            continue;
        };
        let spec = match spec_for(transport) {
            Ok(spec) => spec,
            Err(error) => {
                attempts.push((kind, AttemptFailure::Failed(error)));
                continue;
            }
        };
        let connect = transport.connect_boxed(remote, spec);
        match within(connect, sleep(choice.connect_timeout)).await {
            None => attempts.push((kind, AttemptFailure::TimedOut)),
            Some(Err(error)) => attempts.push((kind, AttemptFailure::Failed(error))),
            Some(Ok(channel)) => {
                let demoted = (kind != choice.planned).then(|| demotion(choice, kind));
                return Ok(HopConnection {
                    channel,
                    transport: kind,
                    demoted,
                });
            }
        }
    }
    Err(HopError { attempts })
}

/// The demotion to `to`, with `to`'s cheapest `cost_t` among the hop's candidates.
fn demotion(choice: &HopChoice<'_>, to: TransportKind) -> (Demotion, Nanos) {
    let cost = choice
        .candidates
        .iter()
        .filter(|caps| caps.transport == to)
        .map(|caps| link_cost(caps, choice.frame, &choice.objective, choice.p99_weight))
        .min()
        .unwrap_or(Nanos(u64::MAX));
    let demotion = Demotion {
        edge: choice.link,
        from: choice.planned,
        to,
    };
    (demotion, cost)
}

/// `work`'s output, or `None` if `timer` completes first. `work` is polled first, so work that is
/// ready wins a tie.
async fn within<W: Future, T: Future<Output = ()>>(work: W, timer: T) -> Option<W::Output> {
    let mut work = pin!(work);
    let mut timer = pin!(timer);
    poll_fn(|cx| {
        if let Poll::Ready(output) = work.as_mut().poll(cx) {
            return Poll::Ready(Some(output));
        }
        if timer.as_mut().poll(cx).is_ready() {
            return Poll::Ready(None);
        }
        Poll::Pending
    })
    .await
}

#[cfg(test)]
mod tests;
