// SPDX-License-Identifier: MIT OR Apache-2.0

//! `SimTransport`: RFC-0031's `Transport` and `Channel` over the simulated fabric (RFC-0060
//! §8.1, §8.3; QUAL-E8-T04).
//!
//! A [`SimNet`] holds every simulated node's channels and the [`SimFabric`] between them. Each
//! node opens channels through its own [`SimTransport`], and gets real
//! `syntropy_net::transport::Channel`s, as it would from the QUIC backend. A channel's two sides
//! are each a `ChannelCore` (DIST-E7-T02), so the credit window, acknowledgements and retention
//! are production's:
//! - **Delivery.** Every frame a core puts on the wire goes through
//!   [`SimFabric::send_reliable`] on the channel's link, one stream per channel side, so it
//!   arrives in order at the time §8.3 gives: α, S/β, per-packet overhead and jitter, and one
//!   PTO for each lost packet. [`SimNet::step`] delivers each frame at its own arrival time.
//! - **Back-pressure.** A sender out of credit gets `WouldBlock`, and the core records the wait
//!   as `credit_stall`, as in production.
//! - **Link down.** A frame on a link that goes down is dropped. A channel whose link stays down,
//!   in either direction, for the transport's failure timeout, counted from the moment it went
//!   down, fails at both ends with a `Link` error: sends return `SendError::Link`, and
//!   [`SimNet::fault`] gives the cause. [`SimNet::next_deadline`] says when that, or anything
//!   else, is next due, so a driver can step on events. The timeout is
//!   RFC-0031 §4's QUIC idle timeout, `link.failover_timeout`: [`FAILOVER_LAN`] or
//!   [`FAILOVER_WAN`]. A link back up before then loses nothing: the cores send the dropped
//!   frames again on their ack timeouts.
//!
//! The model is QUIC's, which retransmits lost packets itself, so a core never escalates its own
//! ack timeouts to a link fault here (RFC-0031 §7.3: on QUIC, Syntropy's retransmission is only
//! for CRC failures and failover). The handshake takes no virtual time: `connect` and `accept`
//! pair up as soon as both are called, and `connect` fails at once with `ConnectTimeout` when
//! there is no link up to the peer. `reconnect` is not modelled.
//!
//! Time comes from the world's [`SimClock`](super::SimClock), through a handle, and randomness
//! from one [`SimEntropy`]: a run replays from its seed.

use std::collections::BTreeMap;
use std::future::poll_fn;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Poll, Waker};

use syntropy_core::ids::LinkId;
use syntropy_core::resource::TransportKind;
use syntropy_core::{Bytes, Clock, ErrorClass, MonoTime, Nanos, NodeId};
use syntropy_net::frame::WIRE_MINOR;
use syntropy_net::transport::core::CRC_STORM_COUNT;
use syntropy_net::transport::{
    BufferPool, Channel, ChannelBackend, ChannelCore, ChannelKey, ChannelSpec, ChannelStats,
    CloseReason, CoreConfig, CreditCadence, InFrame, OutFrame, PoolId, PoolRegistration,
    RemoteEndpoint, SendConfig, SendError, SendTicket, Transmit, Transport, TransportCaps,
};
use syntropy_net::{Error, ErrorCode};

use super::fabric::Sent;
use super::{SimClockHandle, SimEntropy, SimFabric};

/// The failure timeout on a LAN: RFC-0031 §4's QUIC idle timeout, `link.failover_timeout`.
pub const FAILOVER_LAN: Nanos = Nanos(2_000_000_000);

/// The failure timeout across a WAN (RFC-0031 §4, §11).
pub const FAILOVER_WAN: Nanos = Nanos(10_000_000_000);

/// The largest frame: `net.max_frame`'s default, 64 MiB (RFC-0031 §2, §11).
pub const MAX_FRAME: u64 = 64 << 20;

/// Every simulated node's channels, and the fabric between them. Cloning gives another handle to
/// the same network.
#[derive(Clone)]
pub struct SimNet {
    state: Arc<Mutex<State>>,
}

/// One side of one channel.
struct Side {
    node: NodeId,
    key: ChannelKey,
    core: ChannelCore,
    /// The link this side sends on, and the one it hears its peer on.
    out_link: LinkId,
    in_link: LinkId,
    peer: u64,
    /// Since when a link of the channel has been down.
    down_since: Option<MonoTime>,
}

/// A frame on its way to a channel side.
struct InFlight {
    to: u64,
    sent: Sent,
    bytes: Vec<u8>,
}

/// Which end of a channel a waiting call opens: a connect from `connector`, or an accept, both on
/// `acceptor` and `key`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Open {
    acceptor: NodeId,
    key: ChannelKey,
    connector: Option<NodeId>,
}

/// A `connect` or `accept` waiting for its other half.
struct Waiting {
    spec: ChannelSpec,
    waker: Option<Waker>,
    /// Set when the pair is matched: this end's side, or why the channel did not open.
    result: Option<Result<u64, Error>>,
}

struct State {
    fabric: SimFabric,
    clock: SimClockHandle,
    entropy: SimEntropy,
    failure_timeout: Nanos,
    /// The link from one node to another.
    routes: BTreeMap<(NodeId, NodeId), LinkId>,
    pools: BTreeMap<(NodeId, PoolId), BufferPool>,
    sides: BTreeMap<u64, Side>,
    next_side: u64,
    /// Frames in flight, by arrival time and send order.
    in_flight: BTreeMap<(MonoTime, u64), InFlight>,
    order: u64,
    /// Connects and accepts waiting for their other half, in the order they were called.
    waiting: Vec<(Open, Waiting)>,
}

/// One node's transport: it opens that node's channels on a [`SimNet`].
#[derive(Clone)]
pub struct SimTransport {
    net: SimNet,
    node: NodeId,
}

/// A channel side as a `ChannelBackend`.
struct SimChannel {
    net: SimNet,
    side: u64,
}

impl SimNet {
    /// A network over `fabric`, reading the time from `clock`, drawing from `entropy`, and
    /// failing a channel whose link stays down for `failure_timeout`.
    pub fn new(
        fabric: SimFabric,
        clock: SimClockHandle,
        entropy: SimEntropy,
        failure_timeout: Nanos,
    ) -> SimNet {
        SimNet {
            state: Arc::new(Mutex::new(State {
                fabric,
                clock,
                entropy,
                failure_timeout,
                routes: BTreeMap::new(),
                pools: BTreeMap::new(),
                sides: BTreeMap::new(),
                next_side: 1,
                in_flight: BTreeMap::new(),
                order: 0,
                waiting: Vec::new(),
            })),
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .expect("no simulation thread panicked holding the network")
    }

    /// Channels from `from` to `to` go over `link`, which the fabric must hold.
    pub fn route(&self, from: NodeId, to: NodeId, link: LinkId) {
        self.lock().routes.insert((from, to), link);
    }

    /// `node`'s transport.
    pub fn transport(&self, node: NodeId) -> SimTransport {
        SimTransport {
            net: self.clone(),
            node,
        }
    }

    /// Runs `change` on the fabric: a split, a heal, a link's model. This is the only way the
    /// fabric changes once the network holds it, so here each channel side notes the clock's time
    /// if a link of its channel has just gone down, and the failure timeout counts from then,
    /// however late the next [`step`](Self::step) comes.
    pub fn with_fabric<T>(&self, change: impl FnOnce(&mut SimFabric) -> T) -> T {
        let mut state = self.lock();
        let out = change(&mut state.fabric);
        state.stamp_links();
        out
    }

    /// Moves the network to the clock's time: delivers every frame due by then, each at its own
    /// arrival time, then runs each side's timers and failure check, and puts what they send on
    /// the wire.
    pub fn step(&self) {
        self.lock().step();
    }

    /// When the next frame in flight arrives, if any.
    pub fn next_arrival(&self) -> Option<MonoTime> {
        self.lock().in_flight.keys().next().map(|(at, _)| *at)
    }

    /// When the network next has work, for a driver that steps on events: the next frame's
    /// arrival, a channel's failure timeout on a downed link, or a core's next timer
    /// (`ChannelCore::next_deadline`), whichever comes first. A time already past means now.
    /// `None` when nothing is due.
    pub fn next_deadline(&self) -> Option<MonoTime> {
        let state = self.lock();
        let arrival = state.in_flight.keys().next().map(|(at, _)| *at);
        let sides = state
            .sides
            .values()
            .filter(|side| side.core.fault().is_none());
        let timers = sides.flat_map(|side| {
            let failover = side
                .down_since
                .map(|since| since.saturating_add(state.failure_timeout));
            [failover, side.core.next_deadline()]
        });
        arrival.into_iter().chain(timers.flatten()).min()
    }

    /// Frames in flight.
    pub fn in_flight(&self) -> usize {
        self.lock().in_flight.len()
    }

    /// Why the channel `key`'s side on `node` failed, if it has.
    pub fn fault(&self, node: NodeId, key: &ChannelKey) -> Option<Error> {
        self.lock()
            .sides
            .values()
            .find(|side| side.node == node && side.key == *key)
            .and_then(|side| side.core.fault().cloned())
    }
}

impl State {
    fn now(&self) -> MonoTime {
        self.clock.now()
    }

    /// Notes when a link of each side's channel went down, now, or clears it if both are up.
    fn stamp_links(&mut self) {
        let now = self.now();
        for side in self.sides.values_mut() {
            let up = self.fabric.is_up(&side.out_link) && self.fabric.is_up(&side.in_link);
            if up {
                side.down_since = None;
            } else {
                side.down_since.get_or_insert(now);
            }
        }
    }

    fn step(&mut self) {
        let now = self.now();
        while let Some(entry) = self.in_flight.first_entry() {
            if entry.key().0 > now {
                break;
            }
            let frame = entry.remove();
            self.deliver(frame);
        }
        let ids: Vec<u64> = self.sides.keys().copied().collect();
        for id in ids {
            self.watch_links(id, now);
            if let Some(side) = self.sides.get_mut(&id) {
                side.core.on_tick(now);
            }
            self.flush(id, now);
        }
    }

    /// Lands a due frame in its side's core, unless its link went down on the way.
    fn deliver(&mut self, frame: InFlight) {
        if !self.fabric.arrives(&frame.sent) {
            return;
        }
        let at = frame.sent.at;
        let Some(side) = self.sides.get_mut(&frame.to) else {
            return;
        };
        let Some(mut slot) = side.core.next_receive_slot() else {
            // Every receive is in use: the peer broke the credit window, or more control frames
            // landed at once than the four receives kept for them. The frame is lost, as on a
            // full RDMA receive queue; the sender's resend repairs it.
            return;
        };
        let len = frame.bytes.len().min(slot.bytes_mut().len());
        slot.bytes_mut()[..len].copy_from_slice(&frame.bytes[..len]);
        side.core.on_receive(slot.freeze(), len, at);
        // Its answers, credits above all, leave when it lands.
        self.flush(frame.to, at);
    }

    /// Fails a side whose channel's link has been down for the failure timeout, counted from when
    /// it went down ([`SimNet::with_fabric`]). Each side watches both directions of its link, so
    /// both ends fail in the same step.
    fn watch_links(&mut self, id: u64, now: MonoTime) {
        let timeout = self.failure_timeout;
        let Some(side) = self.sides.get_mut(&id) else {
            return;
        };
        let Some(since) = side.down_since else {
            return;
        };
        if now.saturating_duration_since(since) < timeout || side.core.fault().is_some() {
            return;
        }
        side.core.fail(Error::uncoded(
            ErrorClass::Link,
            format!(
                "the link was down for the failure timeout, {} ms (RFC-0031 §4, §7.4)",
                timeout.0 / 1_000_000
            ),
        ));
    }

    /// Puts everything side `id` has to send on its link at `now`.
    fn flush(&mut self, id: u64, now: MonoTime) {
        let Some(side) = self.sides.get_mut(&id) else {
            return;
        };
        let (link, to) = (side.out_link, side.peer);
        let mut frames = Vec::new();
        while let Some(transmit) = side.core.poll_transmit(now) {
            frames.push(match transmit {
                Transmit::Data {
                    header,
                    ext,
                    payload,
                    ..
                } => [&header[..], ext, payload].concat(),
                Transmit::Control(bytes) => bytes.to_vec(),
            });
        }
        for bytes in frames {
            let sent = self
                .fabric
                .send_reliable(link, id, now, bytes.len() as u64, &self.entropy);
            if let Some(sent) = sent {
                self.order += 1;
                self.in_flight
                    .insert((sent.at, self.order), InFlight { to, sent, bytes });
            }
        }
    }

    /// Pairs a connect with its accept on `acceptor` and `key`, if both are waiting and neither
    /// is matched yet, and opens the channel.
    fn pair(&mut self, acceptor: NodeId, key: ChannelKey) {
        let unmatched = |open: &Open, waiting: &Waiting, connect: bool| {
            open.acceptor == acceptor
                && open.key == key
                && open.connector.is_some() == connect
                && waiting.result.is_none()
        };
        let connect = self.waiting.iter().position(|(o, w)| unmatched(o, w, true));
        let accept = self
            .waiting
            .iter()
            .position(|(o, w)| unmatched(o, w, false));
        let (Some(connect), Some(accept)) = (connect, accept) else {
            return;
        };
        let connector = self.waiting[connect].0.connector.expect("a connect");
        let (connect_spec, accept_spec) =
            (self.waiting[connect].1.spec, self.waiting[accept].1.spec);
        let (to_connector, to_acceptor) =
            match self.open(connector, connect_spec, acceptor, accept_spec) {
                Ok((c, a)) => (Ok(c), Ok(a)),
                Err(error) => (Err(error.clone()), Err(error)),
            };
        for (at, result) in [(connect, to_connector), (accept, to_acceptor)] {
            let waiting = &mut self.waiting[at].1;
            waiting.result = Some(result);
            if let Some(waker) = waiting.waker.take() {
                waker.wake();
            }
        }
    }

    /// Registers a waiting connect or accept, and pairs it if its other half is waiting.
    fn wait(&mut self, open: Open, spec: ChannelSpec) {
        self.waiting.push((
            open,
            Waiting {
                spec,
                waker: None,
                result: None,
            },
        ));
        self.pair(open.acceptor, open.key);
    }

    /// The result of the waiting call `open`, once matched, which then stops waiting; or else
    /// stores `waker` to be woken when it is.
    fn poll_open(&mut self, open: Open, waker: &Waker) -> Poll<Result<u64, Error>> {
        let Some(at) = self.waiting.iter().position(|(o, _)| *o == open) else {
            return Poll::Ready(Err(Error::uncoded(
                ErrorClass::Internal,
                "a waiting open was lost",
            )));
        };
        match self.waiting[at].1.result.take() {
            Some(result) => {
                self.waiting.remove(at);
                Poll::Ready(result)
            }
            None => {
                self.waiting[at].1.waker = Some(waker.clone());
                Poll::Pending
            }
        }
    }

    /// Both sides of a channel: each a core whose receive side is its own spec and whose send
    /// side is its peer's, as `Hello` and `HelloAck` would set them.
    fn open(
        &mut self,
        connector: NodeId,
        connect: ChannelSpec,
        acceptor: NodeId,
        accept: ChannelSpec,
    ) -> Result<(u64, u64), Error> {
        let route = |from, to| {
            self.routes
                .get(&(from, to))
                .copied()
                .filter(|link| self.fabric.is_up(link))
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::CONNECT_TIMEOUT,
                        format!("no link up from {from:?} to {to:?}"),
                    )
                })
        };
        let there = route(connector, acceptor)?;
        let back = route(acceptor, connector)?;
        let alpha = self
            .fabric
            .rtt(&there)
            .map_or(Nanos(0), |rtt| Nanos(rtt.srtt.0 / 2));
        let core = |own: ChannelSpec, peer: ChannelSpec, pool: BufferPool| {
            ChannelCore::new(CoreConfig {
                key: own.key,
                wire_minor: WIRE_MINOR,
                cadence: CreditCadence::for_purpose(own.key.purpose),
                alpha_p99: alpha,
                recv_slots: own.recv_slots,
                slot_size: u32::try_from(own.slot_size.0).unwrap_or(u32::MAX),
                pool,
                send: (peer.recv_slots > 0).then(|| SendConfig {
                    peer_recv_slots: peer.recv_slots,
                    peer_slot_size: u32::try_from(peer.slot_size.0).unwrap_or(u32::MAX),
                    retention: own.retention,
                }),
                // QUIC retransmits lost packets itself; the transport's failure timeout, not the
                // core's ack timeouts, says when the link is gone (RFC-0031 §7.3).
                ack_timeouts_to_fault: u32::MAX,
                crc_storm_count: CRC_STORM_COUNT,
            })
        };
        let pool_of = |node: NodeId, spec: &ChannelSpec| {
            self.pools
                .get(&(node, spec.pool.pool))
                .cloned()
                .ok_or_else(|| {
                    Error::uncoded(
                        ErrorClass::InvalidRequest,
                        format!(
                            "pool {:?} is not registered with {node:?}'s transport",
                            spec.pool.pool
                        ),
                    )
                })
        };
        let a = core(connect, accept, pool_of(connector, &connect)?)?;
        let b = core(accept, connect, pool_of(acceptor, &accept)?)?;
        let (a_id, b_id) = (self.next_side, self.next_side + 1);
        self.next_side += 2;
        self.sides.insert(
            a_id,
            Side {
                node: connector,
                key: connect.key,
                core: a,
                out_link: there,
                in_link: back,
                peer: b_id,
                down_since: None,
            },
        );
        self.sides.insert(
            b_id,
            Side {
                node: acceptor,
                key: accept.key,
                core: b,
                out_link: back,
                in_link: there,
                peer: a_id,
                down_since: None,
            },
        );
        Ok((a_id, b_id))
    }
}

impl SimNet {
    fn channel(&self, side: u64, key: ChannelKey) -> Channel {
        Channel::new(
            key,
            Box::new(SimChannel {
                net: self.clone(),
                side,
            }),
        )
    }
}

impl SimTransport {
    /// The node this transport opens channels for.
    pub fn node(&self) -> NodeId {
        self.node
    }
}

impl Transport for SimTransport {
    fn kind(&self) -> TransportKind {
        TransportKind::Quic
    }

    /// QUIC's: encrypted and reliable, frames copied into the receive slots, no registration.
    fn caps(&self) -> TransportCaps {
        TransportCaps {
            zero_copy_recv: false,
            device_memory_recv: false,
            max_frame: Bytes(MAX_FRAME),
            needs_registration: false,
            encrypted: true,
            reliable: true,
            max_channels_per_peer: u16::MAX,
            inline_max: 0,
        }
    }

    /// Records the pool, so channels can take their receive slots from it and frames their
    /// extension and payload.
    fn register(&self, pool: &BufferPool) -> Result<PoolRegistration, Error> {
        self.net
            .lock()
            .pools
            .insert((self.node, pool.id()), pool.clone());
        Ok(PoolRegistration {
            pool: pool.id(),
            transport: TransportKind::Quic,
            handle: 0,
        })
    }

    async fn connect(&self, remote: &RemoteEndpoint, spec: ChannelSpec) -> Result<Channel, Error> {
        let open = Open {
            acceptor: remote.node,
            key: spec.key,
            connector: Some(self.node),
        };
        self.open(open, spec).await
    }

    async fn accept(&self, spec: ChannelSpec) -> Result<Channel, Error> {
        let open = Open {
            acceptor: self.node,
            key: spec.key,
            connector: None,
        };
        self.open(open, spec).await
    }
}

impl SimTransport {
    /// Waits for the other half of `open`, then gives this end's channel.
    async fn open(&self, open: Open, spec: ChannelSpec) -> Result<Channel, Error> {
        self.net.lock().wait(open, spec);
        let side = poll_fn(|cx| self.net.lock().poll_open(open, cx.waker())).await?;
        Ok(self.net.channel(side, spec.key))
    }
}

impl SimChannel {
    /// Runs `op` on this side's core at the clock's time, then puts what it sends on the wire.
    fn with<T>(&self, op: impl FnOnce(&mut ChannelCore, MonoTime) -> T) -> T {
        let mut state = self.net.lock();
        let now = state.now();
        let side = state
            .sides
            .get_mut(&self.side)
            .expect("a channel's side lives with the net");
        let out = op(&mut side.core, now);
        state.flush(self.side, now);
        out
    }

    fn read<T>(&self, op: impl FnOnce(&ChannelCore) -> T) -> T {
        let state = self.net.lock();
        op(&state.sides[&self.side].core)
    }
}

impl ChannelBackend for SimChannel {
    fn try_send(&mut self, frame: OutFrame) -> Result<SendTicket, SendError> {
        let mut state = self.net.lock();
        let now = state.now();
        let node = state.sides[&self.side].node;
        let ext = if frame.ext.is_empty() {
            Some(syntropy_net::transport::ExtRef::empty())
        } else {
            state
                .pools
                .get(&(node, frame.ext.pool))
                .and_then(|pool| pool.retain_ext(frame.ext))
        };
        let payload = if frame.payload.is_empty() {
            Some(syntropy_net::transport::PayloadRef::empty())
        } else {
            state
                .pools
                .get(&(node, frame.payload.pool))
                .and_then(|pool| pool.retain_payload(frame.payload))
        };
        let (Some(ext), Some(payload)) = (ext, payload) else {
            return Err(SendError::Invalid);
        };
        let side = state
            .sides
            .get_mut(&self.side)
            .expect("a channel's side lives with the net");
        let ticket = side
            .core
            .send(frame.ftype, frame.step, frame.flags, ext, payload, now)?;
        state.flush(self.side, now);
        Ok(ticket)
    }

    fn try_recv(&mut self) -> Option<InFrame> {
        self.with(|core, _| core.recv())
    }

    fn completed(&mut self) -> Option<SendTicket> {
        self.with(|core, _| core.completed())
    }

    fn stats(&self) -> ChannelStats {
        self.read(ChannelCore::stats)
    }

    fn credit(&self) -> u32 {
        self.read(ChannelCore::credit)
    }

    fn nack(&mut self, seq: u64) {
        self.with(|core, _| core.nack(seq));
    }

    fn close(&mut self, reason: CloseReason) {
        self.with(|core, _| core.close(reason));
    }
}
