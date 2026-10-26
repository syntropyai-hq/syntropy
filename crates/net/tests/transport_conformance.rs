// SPDX-License-Identifier: MIT OR Apache-2.0

//! The channel-conformance suite (`net::transport::conformance`) run against an in-memory test
//! double of a transport, and against copies of the double that each break one rule of RFC-0031
//! §2 (DIST-E7-T01).
//!
//! The double is built only from `net`'s public API, as a backend in another crate would be. It
//! encodes real frames with `frame::encode`, delivers them into a receive pool of `recv_slots`
//! slots, acknowledges a frame when it lands in a slot, and grants a credit when the receiver
//! drops a frame and its slot is free again (RFC-0031 §7.1). It is test code, not a backend: a
//! lock guards each channel's state, and frames are copied.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use syntropy_core::ErrorClass;
use syntropy_core::resource::TransportKind;
use syntropy_net::Error;
use syntropy_net::frame::{self, Frame, FrameSpec, Header, WIRE_MINOR};
use syntropy_net::transport::conformance::{self, Fabric, MAX_FRAME, Rule, SEND_SLOT, SEND_SLOTS};
use syntropy_net::transport::{
    BufferPool, Channel, ChannelBackend, ChannelKey, ChannelStats, CloseReason, ExtRef, InFrame,
    OutFrame, PayloadRef, PoolId, PoolRegistration, RegistrationTable, SendError, SendTicket,
    SlotReader,
};

/// A way for the double to break one rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    /// The sending side reports another key.
    WrongKey,
    /// Tickets count up by two.
    SkipSeq,
    /// `try_send` ignores the credit limit.
    NoCreditLimit,
    /// The sending side drops its leases at once instead of at the ack.
    NoRetention,
    /// `completed` hands back a ticket as soon as the frame is sent.
    CompleteOnSend,
    /// `completed` hands back the first ticket twice.
    RepeatTicket,
    /// Two frames in flight swap places.
    Reorder,
    /// A payload byte changes on the way.
    Corrupt,
    /// Credit returns when a frame lands, not when the receiver drops it.
    CreditOnArrival,
    /// `bytes_sent` misses a byte per frame.
    MiscountStats,
    /// Frames sent again after a NACK reach the caller a second time.
    DuplicateOnNack,
    /// `try_send` still sends after `close`.
    SendAfterClose,
    /// `register` refuses a pool it already holds, instead of returning its registration.
    RefuseRepeatRegistration,
}

/// Receive slots hold the suite's largest frame, rounded up to a slot size.
const RECV_SLOT: usize = 512;
const _: () = assert!(RECV_SLOT >= MAX_FRAME);

/// One channel's state, both sides.
struct Link {
    key: ChannelKey,
    fault: Option<Fault>,
    send_pool: BufferPool,
    recv_pool: BufferPool,
    recv_slots: u64,
    // The sending side.
    next_seq: u64,
    retained: VecDeque<(u64, ExtRef, PayloadRef)>,
    completed_upto: u64,
    repeated: bool,
    closed: bool,
    /// Every frame sent, encoded, by `chan_seq − 1`: what a NACK sends again.
    log: Vec<Vec<u8>>,
    // The wire.
    in_flight: VecDeque<Vec<u8>>,
    /// The ack and the limit the sending side has heard (RFC-0031 §7.1).
    acked: u64,
    limit: u64,
    // The receiving side.
    received_upto: u64,
    arrived: VecDeque<(Header, SlotReader)>,
    /// Receive slots whose frame the caller has, until it drops it.
    outstanding: Vec<u32>,
    released: u64,
    send_stats: ChannelStats,
    recv_stats: ChannelStats,
}

impl Link {
    fn has(&self, fault: Fault) -> bool {
        self.fault == Some(fault)
    }

    fn try_send(&mut self, frame: OutFrame) -> Result<SendTicket, SendError> {
        if self.closed && !self.has(Fault::SendAfterClose) {
            return Err(SendError::Closed);
        }
        if frame.ftype.is_control() || !frame.ext.len.is_multiple_of(64) {
            return Err(SendError::Invalid);
        }
        let seq = self.next_seq;
        if seq > self.limit && !self.has(Fault::NoCreditLimit) {
            self.send_stats.credit_stalls += 1;
            return Err(SendError::WouldBlock);
        }
        let ext = self
            .send_pool
            .retain_ext(frame.ext)
            .ok_or(SendError::Invalid)?;
        let payload = self
            .send_pool
            .retain_payload(frame.payload)
            .ok_or(SendError::Invalid)?;
        let mut payload_bytes = payload.bytes().to_vec();
        if self.has(Fault::Corrupt) && seq == 2 {
            payload_bytes[0] ^= 0x5a;
        }
        let spec = FrameSpec {
            wire_minor: WIRE_MINOR,
            ftype: frame.ftype,
            flags: frame.flags,
            instance: self.key.instance,
            epoch: self.key.epoch,
            step: frame.step,
            chan_seq: seq,
        };
        let encoded = frame::encode_to_vec(&spec, ext.bytes(), &payload_bytes)
            .map_err(|_| SendError::Invalid)?;
        if encoded.len() > RECV_SLOT {
            return Err(SendError::TooLarge);
        }
        self.send_stats.frames_sent += 1;
        self.send_stats.bytes_sent += encoded.len() as u64;
        if self.has(Fault::MiscountStats) {
            self.send_stats.bytes_sent -= 1;
        }
        self.log.push(encoded.clone());
        self.in_flight.push_back(encoded);
        if !self.has(Fault::NoRetention) {
            self.retained.push_back((seq, ext, payload));
        }
        self.next_seq += if self.has(Fault::SkipSeq) { 2 } else { 1 };
        Ok(SendTicket { chan_seq: seq })
    }

    fn completed(&mut self) -> Option<SendTicket> {
        if self.has(Fault::RepeatTicket) && self.completed_upto == 1 && !self.repeated {
            self.repeated = true;
            return Some(SendTicket { chan_seq: 1 });
        }
        let acked = if self.has(Fault::CompleteOnSend) {
            self.next_seq - 1
        } else {
            self.acked
        };
        if self.completed_upto >= acked {
            return None;
        }
        self.completed_upto += 1;
        while self
            .retained
            .front()
            .is_some_and(|(seq, ..)| *seq <= self.completed_upto)
        {
            self.retained.pop_front();
        }
        Some(SendTicket {
            chan_seq: self.completed_upto,
        })
    }

    fn credit(&self) -> u32 {
        (self.limit + 1).saturating_sub(self.next_seq) as u32
    }

    fn try_recv(&mut self) -> Option<InFrame> {
        let (header, reader) = self.arrived.pop_front()?;
        let ext_len = header.ext_len;
        let ext = ExtRef::new(reader.clone(), frame::HEADER_LEN as u32, ext_len)?;
        let payload = PayloadRef::new(
            reader.clone(),
            frame::HEADER_LEN as u32 + ext_len,
            header.payload_len,
        )?;
        self.outstanding.push(reader.slot());
        self.recv_stats.frames_received += 1;
        self.recv_stats.bytes_received += header.frame_len();
        Some(InFrame {
            header,
            ext,
            payload,
        })
    }

    fn nack(&mut self, seq: u64) {
        self.recv_stats.nacks_sent += 1;
        let from = seq.saturating_sub(1) as usize;
        for encoded in self.log.iter().skip(from) {
            self.in_flight.push_back(encoded.clone());
            self.send_stats.retransmits += 1;
        }
    }

    /// Moves what can move: returned receive slots become credit, frames in flight land in free
    /// slots, and the sending side hears the new ack and limit.
    fn pump(&mut self, hold: bool) {
        let pool = self.recv_pool.clone();
        self.outstanding.retain(|&slot| pool.claim(slot).is_none());
        self.released = self.recv_stats.frames_received - self.outstanding.len() as u64;
        if hold {
            return;
        }
        if self.has(Fault::Reorder) && self.in_flight.len() >= 2 {
            self.in_flight.swap(0, 1);
        }
        while let Some(encoded) = self.in_flight.front() {
            let header = Frame::decode(encoded)
                .expect("the double encodes valid frames")
                .header;
            let seq = header.spec.chan_seq;
            if seq <= self.received_upto && !self.has(Fault::DuplicateOnNack) {
                self.in_flight.pop_front();
                continue;
            }
            let Some(mut slot) = (0..pool.slots()).find_map(|slot| pool.claim(slot)) else {
                break;
            };
            slot.bytes_mut()[..encoded.len()].copy_from_slice(encoded);
            self.in_flight.pop_front();
            self.arrived.push_back((header, slot.freeze()));
            self.received_upto = self.received_upto.max(seq);
        }
        self.acked = self.received_upto;
        self.limit = self.recv_slots
            + if self.has(Fault::CreditOnArrival) {
                self.received_upto
            } else {
                self.released
            };
    }
}

/// One side of a channel over a [`Link`].
struct End {
    link: Arc<Mutex<Link>>,
    sending: bool,
}

impl End {
    fn link(&self) -> std::sync::MutexGuard<'_, Link> {
        self.link
            .lock()
            .expect("no test thread panicked holding the link")
    }
}

impl ChannelBackend for End {
    fn try_send(&mut self, frame: OutFrame) -> Result<SendTicket, SendError> {
        assert!(self.sending, "the suite sends on the sending side only");
        self.link().try_send(frame)
    }

    fn try_recv(&mut self) -> Option<InFrame> {
        if self.sending {
            return None;
        }
        self.link().try_recv()
    }

    fn completed(&mut self) -> Option<SendTicket> {
        if !self.sending {
            return None;
        }
        self.link().completed()
    }

    fn stats(&self) -> ChannelStats {
        let link = self.link();
        if self.sending {
            link.send_stats
        } else {
            link.recv_stats
        }
    }

    fn credit(&self) -> u32 {
        if self.sending {
            self.link().credit()
        } else {
            0
        }
    }

    fn nack(&mut self, seq: u64) {
        if !self.sending {
            self.link().nack(seq);
        }
    }

    fn close(&mut self, _reason: CloseReason) {
        self.link().closed = true;
    }
}

/// The double as a [`Fabric`].
struct Double {
    fault: Option<Fault>,
    can_hold: bool,
    hold: bool,
    send_pool: BufferPool,
    links: Vec<Arc<Mutex<Link>>>,
    /// The double's transport registrations, kept as a backend keeps them (RFC-0031 §8).
    registrations: RegistrationTable,
}

impl Double {
    fn new(fault: Option<Fault>, can_hold: bool) -> Double {
        Double {
            fault,
            can_hold,
            hold: false,
            send_pool: BufferPool::heap(PoolId(1), SEND_SLOT, SEND_SLOTS).expect("a valid pool"),
            links: Vec::new(),
            registrations: RegistrationTable::new(),
        }
    }
}

impl Fabric for Double {
    fn register(&mut self, pool: &BufferPool) -> Option<Result<PoolRegistration, Error>> {
        let id = pool.id();
        if self.fault == Some(Fault::RefuseRepeatRegistration)
            && !self.registrations.registrants(id).is_empty()
        {
            return Some(Err(Error::uncoded(
                ErrorClass::InvalidRequest,
                format!("pool {id:?} is already registered"),
            )));
        }
        Some(
            self.registrations
                .register_transport(pool, TransportKind::Quic, 0),
        )
    }

    fn open(&mut self, key: ChannelKey, recv_slots: u16) -> Result<(Channel, Channel), Error> {
        let pool_id = PoolId(100 + self.links.len() as u32);
        let recv_pool = BufferPool::heap(pool_id, RECV_SLOT, u32::from(recv_slots))
            .map_err(|error| Error::uncoded(ErrorClass::Internal, error.to_string()))?;
        let link = Arc::new(Mutex::new(Link {
            key,
            fault: self.fault,
            send_pool: self.send_pool.clone(),
            recv_pool,
            recv_slots: u64::from(recv_slots),
            next_seq: 1,
            retained: VecDeque::new(),
            completed_upto: 0,
            repeated: false,
            closed: false,
            log: Vec::new(),
            in_flight: VecDeque::new(),
            acked: 0,
            limit: u64::from(recv_slots),
            received_upto: 0,
            arrived: VecDeque::new(),
            outstanding: Vec::new(),
            released: 0,
            send_stats: ChannelStats::default(),
            recv_stats: ChannelStats::default(),
        }));
        self.links.push(Arc::clone(&link));
        let sender_key = if self.fault == Some(Fault::WrongKey) {
            ChannelKey {
                lane: key.lane.wrapping_add(100),
                ..key
            }
        } else {
            key
        };
        let sender = End {
            link: Arc::clone(&link),
            sending: true,
        };
        let receiver = End {
            link,
            sending: false,
        };
        Ok((
            Channel::new(sender_key, Box::new(sender)),
            Channel::new(key, Box::new(receiver)),
        ))
    }

    fn send_pool(&self) -> &BufferPool {
        &self.send_pool
    }

    fn pump(&mut self) {
        for link in &self.links {
            link.lock()
                .expect("no test thread panicked holding the link")
                .pump(self.hold);
        }
    }

    fn hold_delivery(&mut self, hold: bool) -> bool {
        if self.can_hold {
            self.hold = hold;
        }
        self.can_hold
    }
}

#[test]
fn the_double_passes_every_rule() {
    let report = conformance::run(&mut Double::new(None, true));
    assert!(report.passed(), "{report}");
    assert!(report.skipped.is_empty(), "{report}");
    assert_eq!(report.to_string(), "13 of 13 rules passed");
}

#[test]
fn a_fabric_that_cannot_hold_delivery_skips_the_ack_rule() {
    let report = conformance::run(&mut Double::new(None, false));
    assert!(report.passed(), "{report}");
    assert_eq!(report.skipped, [Rule::CompletedAfterAck]);
    assert_eq!(
        report.to_string(),
        "12 of 12 rules passed, skipped [CompletedAfterAck]"
    );
}

#[test]
fn a_double_that_breaks_a_rule_fails_that_rule() {
    let cases = [
        (Fault::WrongKey, Rule::Key),
        (Fault::SkipSeq, Rule::ChanSeq),
        (Fault::NoCreditLimit, Rule::CreditLimit),
        (Fault::NoRetention, Rule::Retention),
        (Fault::CompleteOnSend, Rule::CompletedAfterAck),
        (Fault::RepeatTicket, Rule::CompletedInOrder),
        (Fault::Reorder, Rule::InOrder),
        (Fault::Corrupt, Rule::Intact),
        (Fault::CreditOnArrival, Rule::CreditReturn),
        (Fault::MiscountStats, Rule::Stats),
        (Fault::DuplicateOnNack, Rule::Nack),
        (Fault::SendAfterClose, Rule::Close),
        (Fault::RefuseRepeatRegistration, Rule::Registration),
    ];
    // Every rule is broken by one of the doubles.
    let mut rules: Vec<Rule> = cases.iter().map(|(_, rule)| *rule).collect();
    rules.dedup();
    assert_eq!(rules, Rule::ALL);
    for (fault, rule) in cases {
        let report = conformance::run(&mut Double::new(Some(fault), true));
        assert!(
            report.failed(rule),
            "{fault:?} should fail {rule:?}; {report}"
        );
        // The check of that rule alone reports it too.
        let alone = conformance::check(rule, &mut Double::new(Some(fault), true));
        assert!(alone.is_err(), "{fault:?} passed the {rule:?} check alone");
    }
}

#[test]
fn a_send_pool_too_small_is_reported_not_a_panic() {
    let mut double = Double::new(None, true);
    double.send_pool = BufferPool::heap(PoolId(1), 256, SEND_SLOTS).expect("a valid pool");
    let report = conformance::run(&mut double);
    assert!(report.failed(Rule::ChanSeq), "{report}");
    assert!(
        report.to_string().contains("the suite needs 512"),
        "{report}"
    );
}
