// SPDX-License-Identifier: MIT OR Apache-2.0

//! The channel-conformance suite (`net::transport::conformance`, RFC-0031 §2) over a backend
//! that is a `ChannelCore` on each side and a clean in-memory wire between them: the core keeps
//! every rule a backend must, with nothing added by the backend but the moving of bytes.

use std::sync::{Arc, Mutex, MutexGuard};

use syntropy_core::{ErrorClass, MonoTime, Nanos};
use syntropy_net::Error;
use syntropy_net::frame::WIRE_MINOR;
use syntropy_net::transport::conformance::{self, Fabric, MAX_FRAME, SEND_SLOT, SEND_SLOTS};
use syntropy_net::transport::core::{
    ACK_TIMEOUTS_TO_FAULT, CONTROL_RECEIVES, CRC_STORM_COUNT, MAX_CONTROL_FRAME,
};
use syntropy_net::transport::{
    BufferPool, Channel, ChannelBackend, ChannelCore, ChannelKey, ChannelStats, CloseReason,
    CoreConfig, CreditCadence, InFrame, OutFrame, PoolId, SendConfig, SendError, SendTicket,
};

use super::{deliver, drain};

/// Both sides of one channel and the time.
struct Pair {
    a: ChannelCore,
    b: ChannelCore,
    send_pool: BufferPool,
    now: MonoTime,
    /// Frames from the sending side while the suite holds delivery.
    held: Vec<Vec<u8>>,
}

impl Pair {
    /// One step of 50 µs: timers, then everything each side has to send, delivered unless the
    /// suite holds the sender's direction.
    fn pump(&mut self, hold: bool) {
        self.now = MonoTime(self.now.0 + 50_000);
        let now = self.now;
        self.a.on_tick(now);
        self.b.on_tick(now);
        self.held.extend(drain(&mut self.a, now));
        if !hold {
            for bytes in std::mem::take(&mut self.held) {
                deliver(&mut self.b, &bytes, now);
            }
        }
        for bytes in drain(&mut self.b, now) {
            deliver(&mut self.a, &bytes, now);
        }
    }
}

/// One side of a [`Pair`].
struct End {
    pair: Arc<Mutex<Pair>>,
    sending: bool,
}

impl End {
    fn pair(&self) -> MutexGuard<'_, Pair> {
        self.pair
            .lock()
            .expect("no test thread panicked holding the pair")
    }
}

impl ChannelBackend for End {
    fn try_send(&mut self, frame: OutFrame) -> Result<SendTicket, SendError> {
        let mut pair = self.pair();
        let ext = pair
            .send_pool
            .retain_ext(frame.ext)
            .ok_or(SendError::Invalid)?;
        let payload = pair
            .send_pool
            .retain_payload(frame.payload)
            .ok_or(SendError::Invalid)?;
        let now = pair.now;
        let core = if self.sending {
            &mut pair.a
        } else {
            &mut pair.b
        };
        core.send(frame.ftype, frame.step, frame.flags, ext, payload, now)
    }

    fn try_recv(&mut self) -> Option<InFrame> {
        let mut pair = self.pair();
        if self.sending {
            pair.a.recv()
        } else {
            pair.b.recv()
        }
    }

    fn completed(&mut self) -> Option<SendTicket> {
        let mut pair = self.pair();
        if self.sending {
            pair.a.completed()
        } else {
            pair.b.completed()
        }
    }

    fn stats(&self) -> ChannelStats {
        let pair = self.pair();
        if self.sending {
            pair.a.stats()
        } else {
            pair.b.stats()
        }
    }

    fn credit(&self) -> u32 {
        let pair = self.pair();
        if self.sending {
            pair.a.credit()
        } else {
            pair.b.credit()
        }
    }

    fn nack(&mut self, seq: u64) {
        let mut pair = self.pair();
        if self.sending {
            pair.a.nack(seq)
        } else {
            pair.b.nack(seq)
        }
    }

    fn close(&mut self, reason: CloseReason) {
        let mut pair = self.pair();
        if self.sending {
            pair.a.close(reason)
        } else {
            pair.b.close(reason)
        }
    }
}

/// The core-backed fabric.
struct Cores {
    send_pool: BufferPool,
    pairs: Vec<Arc<Mutex<Pair>>>,
    hold: bool,
}

impl Fabric for Cores {
    fn open(&mut self, key: ChannelKey, recv_slots: u16) -> Result<(Channel, Channel), Error> {
        let internal = |error: syntropy_net::transport::PoolError| {
            Error::uncoded(ErrorClass::Internal, error.to_string())
        };
        let slot_size = MAX_FRAME.next_multiple_of(64) as u32;
        let alpha = Nanos(200_000);
        let config = |recv_slots: u16, pool: BufferPool, send: Option<SendConfig>| CoreConfig {
            key,
            wire_minor: WIRE_MINOR,
            cadence: CreditCadence::EveryFrame,
            alpha_p99: alpha,
            recv_slots,
            slot_size: if recv_slots == 0 { 0 } else { slot_size },
            pool,
            send,
            ack_timeouts_to_fault: ACK_TIMEOUTS_TO_FAULT,
            crc_storm_count: CRC_STORM_COUNT,
        };
        let id = 10 + 2 * self.pairs.len() as u32;
        let a_pool = BufferPool::heap(PoolId(id), MAX_CONTROL_FRAME, u32::from(CONTROL_RECEIVES))
            .map_err(internal)?;
        let b_pool = BufferPool::heap(
            PoolId(id + 1),
            (slot_size as usize).max(MAX_CONTROL_FRAME),
            u32::from(recv_slots) + u32::from(CONTROL_RECEIVES),
        )
        .map_err(internal)?;
        let send = SendConfig {
            peer_recv_slots: recv_slots,
            peer_slot_size: slot_size,
            retention: recv_slots,
        };
        let pair = Pair {
            a: ChannelCore::new(config(0, a_pool, Some(send)))?,
            b: ChannelCore::new(config(recv_slots, b_pool, None))?,
            send_pool: self.send_pool.clone(),
            now: MonoTime(0),
            held: Vec::new(),
        };
        let pair = Arc::new(Mutex::new(pair));
        self.pairs.push(Arc::clone(&pair));
        let sender = End {
            pair: Arc::clone(&pair),
            sending: true,
        };
        let receiver = End {
            pair,
            sending: false,
        };
        Ok((
            Channel::new(key, Box::new(sender)),
            Channel::new(key, Box::new(receiver)),
        ))
    }

    fn send_pool(&self) -> &BufferPool {
        &self.send_pool
    }

    fn pump(&mut self) {
        for pair in &self.pairs {
            pair.lock()
                .expect("no test thread panicked holding the pair")
                .pump(self.hold);
        }
    }

    fn hold_delivery(&mut self, hold: bool) -> bool {
        self.hold = hold;
        true
    }
}

#[test]
fn the_core_passes_every_conformance_rule() {
    let mut fabric = Cores {
        send_pool: BufferPool::heap(PoolId(1), SEND_SLOT, SEND_SLOTS).expect("a valid pool"),
        pairs: Vec::new(),
        hold: false,
    };
    let report = conformance::run(&mut fabric);
    assert!(report.passed(), "{report}");
    assert!(report.skipped.is_empty(), "{report}");
    for pair in &fabric.pairs {
        let pair = pair.lock().unwrap();
        assert_eq!((pair.a.fault(), pair.b.fault()), (None, None));
    }
}
