// SPDX-License-Identifier: MIT OR Apache-2.0

//! The channel-conformance suite: the rules RFC-0031 §2 makes binding on every backend, as checks
//! any transport can run (DIST-E7-T01).
//!
//! A backend's tests implement [`Fabric`], which opens a connected pair of channels and moves
//! frames between them, and call [`run`]. The QUIC, TCP, shared-memory and RDMA backends run it
//! over their real links (DIST-E7-T03, T05, T06, T07, T08), and the simulator's `SimTransport`
//! over `SimFabric` (QUAL-E8-T04). Every check opens its own channel, sends real `Activation`
//! frames whose extension and payload sit in the fabric's send pool, and reports the [`Rule`] it
//! found broken.
//!
//! | Rule | What it checks |
//! |---|---|
//! | [`Rule::Key`] | both sides report the key they were opened with |
//! | [`Rule::ChanSeq`] | tickets number data frames from 1, one up per frame |
//! | [`Rule::CreditLimit`] | the initial credit is `recv_slots`; past it `try_send` returns `WouldBlock`, and keeps doing so while the receiver holds its slots |
//! | [`Rule::Retention`] | a sent frame's slot is not reused before the peer acknowledges it, and is released once `completed` hands its ticket back |
//! | [`Rule::CompletedAfterAck`] | `completed` hands back nothing while delivery is held |
//! | [`Rule::CompletedInOrder`] | `completed` hands back every ticket once, in order |
//! | [`Rule::InOrder`] | `try_recv` yields every frame once, in `chan_seq` order |
//! | [`Rule::Intact`] | a received frame's header fields, extension and payload equal what was sent |
//! | [`Rule::CreditReturn`] | credit returns when the receiver drops its frames, not before |
//! | [`Rule::Stats`] | the counters match the frames, the bytes and the refused sends |
//! | [`Rule::Nack`] | a NACK never hands the receiver a frame twice, and the channel goes on |
//! | [`Rule::Close`] | frames sent before `close` arrive, and sends after it return `Closed` |
//! | [`Rule::Registration`] | registering a pool the transport already holds returns its one registration, equal to the first (RFC-0031 §8) |
//!
//! [`Rule::CompletedAfterAck`] needs a fabric that can hold delivery ([`Fabric::hold_delivery`]),
//! and is reported as skipped otherwise; [`Rule::Retention`] then checks a weaker form.
//! [`Rule::Registration`] needs a fabric with a transport to register with
//! ([`Fabric::register`]), and is skipped otherwise. The suite
//! does not check what needs a fault injected on the wire (loss, corruption, a foreign epoch):
//! the channel core's property tests do (DIST-E7-T02, RFC-0031 Testing).

use std::fmt;

use syntropy_core::plan::WireDType;
use syntropy_core::{InstanceId, NodeId, PlanEpoch, SequenceId, StageIx};

use super::{
    BufferPool, Channel, ChannelKey, ChannelPurpose, CloseReason, InFrame, OutFrame, PoolId,
    PoolRegistration, SendError, SlotReader,
};
use crate::error::Error;
use crate::frame::{
    Activation, ActivationHeader, ActivationKind, Flags, FrameType, HEADER_LEN, RowLayout,
    SideItems, Span, Spans, TreeArea,
};

/// The receive slots the suite asks for on each channel: its credit window.
pub const RECV_SLOTS: u16 = 4;

/// The largest frame the suite sends, header included. Receive slots must hold it.
pub const MAX_FRAME: usize = 384;

/// The smallest send-pool slot the suite needs: an extension and a payload in one slot.
pub const SEND_SLOT: usize = 512;

/// The fewest send-pool slots the suite needs.
pub const SEND_SLOTS: u32 = 16;

/// How often a check pumps before it gives up waiting.
const PUMPS: usize = 1_000;

/// Frames in the longer exchanges: three credit windows.
const FRAMES: u32 = 3 * RECV_SLOTS as u32;

/// What a backend's tests give the suite: a way to open connected channels and move frames.
pub trait Fabric {
    /// Opens the channel `key` between two endpoints, with `recv_slots` receive slots of at least
    /// [`MAX_FRAME`] bytes on the receiving side, and returns the sending side and the receiving
    /// side.
    fn open(&mut self, key: ChannelKey, recv_slots: u16) -> Result<(Channel, Channel), Error>;

    /// The pool the sending side's frames take their extension and payload from: registered
    /// with the sender's transport, with at least [`SEND_SLOTS`] slots of at least
    /// [`SEND_SLOT`] bytes, all free between checks.
    fn send_pool(&self) -> &BufferPool;

    /// Lets frames, credits and acknowledgements in flight move. A simulated fabric moves what
    /// it can; a fabric over a real link waits a little.
    fn pump(&mut self);

    /// How long, in wall-clock time, a check keeps pumping for an event after its `PUMPS` are
    /// spent. A fabric over a real link on a loaded or instrumented machine needs this: its pumps
    /// sleep, so a count of them is a duration that slows with the machine. A check that passes
    /// never waits it out; the default, none, leaves a simulated fabric on its count.
    fn patience(&self) -> std::time::Duration {
        std::time::Duration::ZERO
    }

    /// Holds (`true`) or resumes (`false`) delivery from the sending side to the receiving side,
    /// frames already in flight included. Returns whether the fabric can; it does nothing when it
    /// cannot.
    fn hold_delivery(&mut self, hold: bool) -> bool {
        let _ = hold;
        false
    }

    /// Registers `pool` with the sending side's transport, as `Transport::register` does, and
    /// returns what it returned; `None` if the fabric has no transport to register with.
    fn register(&mut self, pool: &BufferPool) -> Option<Result<PoolRegistration, Error>> {
        let _ = pool;
        None
    }
}

/// The pool [`Rule::Registration`]'s check registers: an id a fabric's own pools are unlikely to
/// use.
const REGISTRATION_POOL: PoolId = PoolId(u32::MAX - 0x0C);

/// A rule of RFC-0031 §2 the suite checks; see the [module](self) table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rule {
    Key,
    ChanSeq,
    CreditLimit,
    Retention,
    CompletedAfterAck,
    CompletedInOrder,
    InOrder,
    Intact,
    CreditReturn,
    Stats,
    Nack,
    Close,
    Registration,
}

impl Rule {
    /// Every rule, in the order [`run`] checks them.
    pub const ALL: [Rule; 13] = [
        Rule::Key,
        Rule::ChanSeq,
        Rule::CreditLimit,
        Rule::Retention,
        Rule::CompletedAfterAck,
        Rule::CompletedInOrder,
        Rule::InOrder,
        Rule::Intact,
        Rule::CreditReturn,
        Rule::Stats,
        Rule::Nack,
        Rule::Close,
        Rule::Registration,
    ];

    /// The lane of the channel the rule's check opens, so no two checks share a key.
    fn lane(self) -> u8 {
        Rule::ALL
            .iter()
            .position(|rule| *rule == self)
            .expect("every rule is in ALL") as u8
    }
}

/// A rule the backend broke, and how.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
    pub rule: Rule,
    pub detail: String,
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.rule, self.detail)
    }
}

/// What [`run`] found.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    pub failures: Vec<Failure>,
    /// Rules the fabric could not check: [`Rule::CompletedAfterAck`] without
    /// [`Fabric::hold_delivery`], [`Rule::Registration`] without [`Fabric::register`].
    pub skipped: Vec<Rule>,
}

impl Report {
    /// No rule failed.
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }

    /// Whether `rule` failed.
    pub fn failed(&self, rule: Rule) -> bool {
        self.failures.iter().any(|failure| failure.rule == rule)
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let checked = Rule::ALL.len() - self.skipped.len();
        write!(
            f,
            "{} of {checked} rules passed",
            checked - self.failures.len()
        )?;
        if !self.skipped.is_empty() {
            write!(f, ", skipped {:?}", self.skipped)?;
        }
        for failure in &self.failures {
            write!(f, "\n  {failure}")?;
        }
        Ok(())
    }
}

/// Whether a check ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Passed,
    /// The fabric cannot do what the check needs.
    Skipped,
}

/// Checks every rule, each on a channel of its own.
pub fn run(fabric: &mut dyn Fabric) -> Report {
    let mut report = Report::default();
    for rule in Rule::ALL {
        match check(rule, fabric) {
            Ok(Outcome::Passed) => {}
            Ok(Outcome::Skipped) => report.skipped.push(rule),
            Err(detail) => report.failures.push(Failure { rule, detail }),
        }
    }
    report
}

/// Checks one rule on a channel of its own; the error says how it was broken.
pub fn check(rule: Rule, fabric: &mut dyn Fabric) -> Result<Outcome, String> {
    let key = key(rule);
    let checked = match rule {
        Rule::Key => check_key(fabric, key),
        Rule::ChanSeq => check_chan_seq(fabric, key),
        Rule::CreditLimit => check_credit_limit(fabric, key),
        Rule::Retention => check_retention(fabric, key),
        Rule::CompletedAfterAck => check_completed_after_ack(fabric, key),
        Rule::CreditReturn => check_credit_return(fabric, key),
        Rule::Nack => check_nack(fabric, key),
        Rule::Close => check_close(fabric, key),
        Rule::Registration => check_registration(fabric),
        Rule::CompletedInOrder | Rule::InOrder | Rule::Intact | Rule::Stats => {
            exchange(fabric, key).and_then(|exchange| exchange.check(rule))
        }
    };
    match checked {
        Ok(()) => Ok(Outcome::Passed),
        Err(Broken::Skipped) => Ok(Outcome::Skipped),
        Err(Broken::Rule(detail)) => Err(detail),
    }
}

/// How a check ended early.
enum Broken {
    Skipped,
    Rule(String),
}

type Checked = Result<(), Broken>;

fn fail<T>(detail: impl Into<String>) -> Result<T, Broken> {
    Err(Broken::Rule(detail.into()))
}

fn key(rule: Rule) -> ChannelKey {
    ChannelKey {
        instance: InstanceId(0x5359_4e46),
        epoch: PlanEpoch { term: 3, seq: 7 },
        purpose: ChannelPurpose::Activations,
        from: StageIx(0),
        to: StageIx(1),
        lane: rule.lane(),
    }
}

/// Opens the rule's channel and lets the opening handshake finish.
fn open(fabric: &mut dyn Fabric, key: ChannelKey) -> Result<(Channel, Channel), Broken> {
    let pair = fabric
        .open(key, RECV_SLOTS)
        .map_err(|error| Broken::Rule(format!("open failed: {error}")))?;
    fabric.pump();
    Ok(pair)
}

/// A frame the suite sent, as it expects to receive it.
#[derive(Clone, Debug)]
struct Expected {
    chan_seq: u64,
    step: u64,
    ext: Vec<u8>,
    payload: Vec<u8>,
}

impl Expected {
    fn frame_len(&self) -> u64 {
        (HEADER_LEN + self.ext.len() + self.payload.len()) as u64
    }
}

/// The `n`th frame: an `Activation` of 1 to 3 F32 rows of 8 elements, written into a free slot of
/// the send pool, extension first. The caller holds the slot's first reader; `None` if no slot
/// is free.
fn make_frame(
    pool: &BufferPool,
    key: &ChannelKey,
    n: u32,
) -> Result<Option<(OutFrame, Expected, SlotReader)>, Broken> {
    let Some(mut writer) = (0..pool.slots()).find_map(|slot| pool.claim(slot)) else {
        return Ok(None);
    };
    let rows = 1 + n % 3;
    let layout = RowLayout {
        dtype: WireDType::F32,
        streams: 1,
        hidden: 8,
    };
    let stride = layout.stride().expect("F32 has a width") as usize;
    let payload: Vec<u8> = (0..rows as usize * stride)
        .map(|i| (n as usize * 31 + i) as u8)
        .collect();
    let spans = [Span {
        seq: SequenceId::new(&NodeId([1; 32]), 1, n),
        first_position: n * 3,
        n_positions: rows,
        row_offset: 0,
        tree: None,
    }];
    let activation = Activation {
        header: ActivationHeader {
            from_stage: key.from,
            to_stage: key.to,
            kind: ActivationKind::Decode,
            layout,
            n_rows: rows,
            part_index: 0,
            part_count: 1,
        },
        spans: Spans::new(&spans),
        tree: TreeArea::new(&[]),
        side: SideItems::new(&[]),
        payload: &payload,
    };
    let bytes = writer.bytes_mut();
    if bytes.len() < SEND_SLOT {
        return fail(format!(
            "the send pool's slots hold {} bytes; the suite needs {SEND_SLOT}",
            bytes.len()
        ));
    }
    let ext_len = activation
        .write_ext(bytes)
        .map_err(|error| Broken::Rule(format!("the suite's frame does not encode: {error}")))?;
    bytes[ext_len..ext_len + payload.len()].copy_from_slice(&payload);
    let ext = bytes[..ext_len].to_vec();
    let reader = writer.freeze();
    let frame = OutFrame {
        ftype: FrameType::Activation,
        step: 1_000 + u64::from(n),
        flags: Flags::default(),
        ext: reader.ext(0, ext_len as u32).expect("inside the slot"),
        payload: reader
            .payload(ext_len as u32, payload.len() as u32)
            .expect("inside the slot"),
    };
    let expected = Expected {
        chan_seq: 0,
        step: frame.step,
        ext,
        payload,
    };
    Ok(Some((frame, expected, reader)))
}

/// A send attempt: what `try_send` returned, the frame, and the send-pool slot it used.
struct Attempt {
    result: Result<u64, SendError>,
    expected: Expected,
    slot: u32,
}

/// Tries to send the `n`th frame; `None` when no send slot is free. The suite drops its own
/// reader of the slot at once, so from then on only the channel's lease holds it.
fn send(pool: &BufferPool, channel: &mut Channel, n: u32) -> Result<Option<Attempt>, Broken> {
    let key = *channel.key();
    let Some((frame, mut expected, reader)) = make_frame(pool, &key, n)? else {
        return Ok(None);
    };
    let slot = reader.slot();
    let result = channel.try_send(frame).map(|ticket| ticket.chan_seq);
    if let Ok(chan_seq) = result {
        expected.chan_seq = chan_seq;
    }
    drop(reader);
    Ok(Some(Attempt {
        result,
        expected,
        slot,
    }))
}

/// Sends the `n`th frame and requires it to be accepted.
fn send_ok(pool: &BufferPool, channel: &mut Channel, n: u32) -> Result<Attempt, Broken> {
    match send(pool, channel, n)? {
        Some(attempt) if attempt.result.is_ok() => Ok(attempt),
        Some(attempt) => fail(format!(
            "try_send of frame {n} returned {:?} with credit {}",
            attempt.result,
            channel.credit()
        )),
        None => fail("no send-pool slot was free"),
    }
}

/// Pumps until `done` holds: at least [`PUMPS`] times, and then until the fabric's
/// [`patience`](Fabric::patience) has run out.
fn pump_until(fabric: &mut dyn Fabric, mut done: impl FnMut() -> bool) -> bool {
    let start = std::time::Instant::now();
    let patience = fabric.patience();
    let mut pumps = 0;
    loop {
        if done() {
            return true;
        }
        if pumps >= PUMPS && start.elapsed() >= patience {
            return false;
        }
        fabric.pump();
        pumps += 1;
    }
}

/// Receives into `frames` until it holds `count`; false if they never come.
fn receive(
    fabric: &mut dyn Fabric,
    receiver: &mut Channel,
    frames: &mut Vec<InFrame>,
    count: usize,
) -> bool {
    pump_until(fabric, || {
        while let Some(frame) = receiver.try_recv() {
            frames.push(frame);
        }
        frames.len() >= count
    })
}

/// Receives and drops whatever is left, and collects what `completed` still hands back, until
/// nothing moves for a few pumps.
fn drain(fabric: &mut dyn Fabric, sender: &mut Channel, receiver: &mut Channel) {
    let mut quiet = 0;
    for _ in 0..PUMPS {
        let mut moved = false;
        while receiver.try_recv().is_some() {
            moved = true;
        }
        while sender.completed().is_some() {
            moved = true;
        }
        quiet = if moved { 0 } else { quiet + 1 };
        if quiet == 4 {
            return;
        }
        fabric.pump();
    }
}

fn check_key(fabric: &mut dyn Fabric, key: ChannelKey) -> Checked {
    let (sender, receiver) = open(fabric, key)?;
    for (side, channel) in [("sending", &sender), ("receiving", &receiver)] {
        if *channel.key() != key {
            return fail(format!(
                "the {side} side reports key {:?}, opened as {key:?}",
                channel.key()
            ));
        }
    }
    Ok(())
}

fn check_chan_seq(fabric: &mut dyn Fabric, key: ChannelKey) -> Checked {
    let pool = fabric.send_pool().clone();
    let (mut sender, mut receiver) = open(fabric, key)?;
    // Credit is the credit-limit rule's to check; this one numbers whatever credit lets through.
    let mut seqs = Vec::new();
    for n in 0..u32::from(RECV_SLOTS) {
        match send(&pool, &mut sender, n)? {
            Some(Attempt {
                result: Ok(chan_seq),
                ..
            }) => seqs.push(chan_seq),
            Some(Attempt {
                result: Err(SendError::WouldBlock),
                ..
            }) => break,
            Some(attempt) => {
                return fail(format!(
                    "try_send of frame {n} returned {:?}",
                    attempt.result
                ));
            }
            None => return fail("no send-pool slot was free"),
        }
    }
    drain(fabric, &mut sender, &mut receiver);
    let expected: Vec<u64> = (1..=seqs.len() as u64).collect();
    if seqs.is_empty() || seqs != expected {
        return fail(format!("tickets {seqs:?}, expected {expected:?}"));
    }
    Ok(())
}

fn check_credit_limit(fabric: &mut dyn Fabric, key: ChannelKey) -> Checked {
    let pool = fabric.send_pool().clone();
    let (mut sender, mut receiver) = open(fabric, key)?;
    let window = u32::from(RECV_SLOTS);
    if sender.credit() != window {
        return fail(format!(
            "the initial credit is {}, not recv_slots = {window}",
            sender.credit()
        ));
    }
    for n in 0..window {
        send_ok(&pool, &mut sender, n)?;
        if sender.credit() != window - n - 1 {
            return fail(format!(
                "credit is {} after {} sends in a window of {window}",
                sender.credit(),
                n + 1
            ));
        }
    }
    // The receiver never takes its frames, so its slots stay full and no credit may return.
    for round in 0..8 {
        match send(&pool, &mut sender, window + round)? {
            Some(Attempt {
                result: Err(SendError::WouldBlock),
                ..
            }) => {}
            Some(attempt) => {
                return fail(format!(
                    "try_send past the window returned {:?}, not WouldBlock",
                    attempt.result
                ));
            }
            None => return fail("no send-pool slot was free"),
        }
        fabric.pump();
    }
    if sender.credit() != 0 {
        return fail(format!(
            "credit is {} while the receiver holds every slot",
            sender.credit()
        ));
    }
    drain(fabric, &mut sender, &mut receiver);
    Ok(())
}

fn check_retention(fabric: &mut dyn Fabric, key: ChannelKey) -> Checked {
    let pool = fabric.send_pool().clone();
    let (mut sender, mut receiver) = open(fabric, key)?;
    let held = fabric.hold_delivery(true);
    let slot = send_ok(&pool, &mut sender, 0)?.slot;
    let mut acked = false;
    for _ in 0..if held { 8 } else { 1 } {
        if pool.claim(slot).is_some() {
            // Without a hold the ack may have come already, and the slot is then free to reuse.
            if !held && sender.completed().is_some() {
                acked = true;
                break;
            }
            fabric.hold_delivery(false);
            return fail(
                "the payload slot was free to reuse before the peer acknowledged the frame",
            );
        }
        fabric.pump();
    }
    fabric.hold_delivery(false);
    let done = acked
        || pump_until(fabric, || {
            while receiver.try_recv().is_some() {}
            sender.completed().is_some()
        });
    if !done {
        return fail("the frame was never acknowledged");
    }
    if pool.claim(slot).is_none() {
        return fail("the payload slot was still held after completed handed back its ticket");
    }
    drain(fabric, &mut sender, &mut receiver);
    Ok(())
}

fn check_completed_after_ack(fabric: &mut dyn Fabric, key: ChannelKey) -> Checked {
    let pool = fabric.send_pool().clone();
    let (mut sender, mut receiver) = open(fabric, key)?;
    if !fabric.hold_delivery(true) {
        return Err(Broken::Skipped);
    }
    send_ok(&pool, &mut sender, 0)?;
    for _ in 0..8 {
        fabric.pump();
        if let Some(ticket) = sender.completed() {
            fabric.hold_delivery(false);
            return fail(format!(
                "completed handed back ticket {} while delivery was held",
                ticket.chan_seq
            ));
        }
    }
    fabric.hold_delivery(false);
    let mut ticket = None;
    let done = pump_until(fabric, || {
        while receiver.try_recv().is_some() {}
        ticket = sender.completed();
        ticket.is_some()
    });
    if !done {
        return fail("the frame was never acknowledged after delivery resumed");
    }
    if ticket.map(|ticket| ticket.chan_seq) != Some(1) {
        return fail(format!(
            "completed handed back {ticket:?} for the first frame"
        ));
    }
    drain(fabric, &mut sender, &mut receiver);
    Ok(())
}

fn check_credit_return(fabric: &mut dyn Fabric, key: ChannelKey) -> Checked {
    let pool = fabric.send_pool().clone();
    let (mut sender, mut receiver) = open(fabric, key)?;
    let window = u32::from(RECV_SLOTS);
    for n in 0..window {
        send_ok(&pool, &mut sender, n)?;
    }
    let mut held = Vec::new();
    if !receive(fabric, &mut receiver, &mut held, window as usize) {
        return fail(format!(
            "the receiver got {} of the {window} frames sent",
            held.len()
        ));
    }
    for _ in 0..8 {
        fabric.pump();
    }
    if sender.credit() != 0 {
        return fail(format!(
            "credit returned to {} while the receiver still held every frame",
            sender.credit()
        ));
    }
    drop(held);
    if !pump_until(fabric, || sender.credit() == window) {
        return fail(format!(
            "credit came back to {} of {window} after the receiver dropped its frames",
            sender.credit()
        ));
    }
    drain(fabric, &mut sender, &mut receiver);
    Ok(())
}

fn check_nack(fabric: &mut dyn Fabric, key: ChannelKey) -> Checked {
    let pool = fabric.send_pool().clone();
    let (mut sender, mut receiver) = open(fabric, key)?;
    let window = u32::from(RECV_SLOTS);
    for n in 0..window {
        send_ok(&pool, &mut sender, n)?;
    }
    let mut frames = Vec::new();
    if !receive(fabric, &mut receiver, &mut frames, window as usize) {
        return fail(format!(
            "the receiver got {} of {window} frames",
            frames.len()
        ));
    }
    drop(frames);
    receiver.nack(1);
    for _ in 0..16 {
        fabric.pump();
        if let Some(frame) = receiver.try_recv() {
            return fail(format!(
                "frame {} was handed to the receiver again after a NACK",
                frame.header.spec.chan_seq
            ));
        }
        while sender.completed().is_some() {}
    }
    if !pump_until(fabric, || sender.credit() > 0) {
        return fail("no credit came back after the NACK");
    }
    let next = send_ok(&pool, &mut sender, window)?.expected.chan_seq;
    let mut frames = Vec::new();
    if !receive(fabric, &mut receiver, &mut frames, 1) {
        return fail("the frame sent after the NACK never arrived");
    }
    let got = frames[0].header.spec.chan_seq;
    if got != next || next != u64::from(window) + 1 {
        return fail(format!(
            "after the NACK, ticket {next} arrived as chan_seq {got}; expected {}",
            window + 1
        ));
    }
    drop(frames);
    drain(fabric, &mut sender, &mut receiver);
    Ok(())
}

fn check_close(fabric: &mut dyn Fabric, key: ChannelKey) -> Checked {
    let pool = fabric.send_pool().clone();
    let (mut sender, mut receiver) = open(fabric, key)?;
    for n in 0..2 {
        send_ok(&pool, &mut sender, n)?;
    }
    sender.close(CloseReason::NORMAL);
    match send(&pool, &mut sender, 2)? {
        Some(Attempt {
            result: Err(SendError::Closed),
            ..
        }) => {}
        Some(attempt) => {
            return fail(format!(
                "try_send after close returned {:?}, not Closed",
                attempt.result
            ));
        }
        None => return fail("no send-pool slot was free"),
    }
    let mut frames = Vec::new();
    if !receive(fabric, &mut receiver, &mut frames, 2) {
        return fail(format!(
            "{} of the 2 frames sent before close arrived",
            frames.len()
        ));
    }
    let seqs: Vec<u64> = frames.iter().map(|f| f.header.spec.chan_seq).collect();
    if seqs != [1, 2] {
        return fail(format!("the frames sent before close arrived as {seqs:?}"));
    }
    drop(frames);
    drain(fabric, &mut sender, &mut receiver);
    Ok(())
}

/// RFC-0031 §8: each pool is registered once per transport, so registering a pool the transport
/// already holds returns its one registration, equal to the first, and is not refused. A runner
/// can then register its pools at each plan epoch without tracking which it already has.
fn check_registration(fabric: &mut dyn Fabric) -> Checked {
    let pool = BufferPool::heap(REGISTRATION_POOL, 64, 1)
        .map_err(|error| Broken::Rule(format!("a pool for the check: {error}")))?;
    let Some(first) = fabric.register(&pool) else {
        return Err(Broken::Skipped);
    };
    let first =
        first.map_err(|error| Broken::Rule(format!("registering a new pool failed: {error}")))?;
    let again = fabric
        .register(&pool)
        .unwrap_or_else(|| Err(Error::unsupported()))
        .map_err(|error| {
            Broken::Rule(format!(
                "registering the same pool again was refused: {error}"
            ))
        })?;
    if again != first {
        return fail(format!(
            "registering the same pool again gave {again:?}, not the first registration {first:?}"
        ));
    }
    Ok(())
}

/// A received frame, copied out of its slot.
#[derive(Clone, Debug)]
struct Received {
    chan_seq: u64,
    ftype: FrameType,
    flags: Flags,
    instance: InstanceId,
    epoch: PlanEpoch,
    step: u64,
    ext: Vec<u8>,
    payload: Vec<u8>,
    frame_len: u64,
}

/// [`FRAMES`] frames sent through a window of [`RECV_SLOTS`], with the receiver taking and
/// dropping each frame as it comes.
struct Exchange {
    key: ChannelKey,
    sent: Vec<Expected>,
    received: Vec<Received>,
    tickets: Vec<u64>,
    would_block: u64,
    sender: super::ChannelStats,
    receiver: super::ChannelStats,
}

fn exchange(fabric: &mut dyn Fabric, key: ChannelKey) -> Result<Exchange, Broken> {
    let pool = fabric.send_pool().clone();
    let (mut sender, mut receiver) = open(fabric, key)?;
    let mut sent = Vec::new();
    let mut received = Vec::new();
    let mut tickets = Vec::new();
    let mut would_block = 0;
    let mut next = 0;
    let start = std::time::Instant::now();
    let patience = fabric.patience();
    let mut pumps = 0;
    // The count first, then the fabric's patience, as in `pump_until`.
    while pumps < PUMPS || start.elapsed() < patience {
        pumps += 1;
        while next < FRAMES {
            let Some(attempt) = send(&pool, &mut sender, next)? else {
                break;
            };
            match attempt.result {
                Ok(_) => {
                    sent.push(attempt.expected);
                    next += 1;
                }
                Err(SendError::WouldBlock) => {
                    would_block += 1;
                    break;
                }
                Err(error) => return fail(format!("try_send of frame {next} returned {error:?}")),
            }
        }
        fabric.pump();
        collect(&mut sender, &mut receiver, &mut received, &mut tickets);
        if next == FRAMES && received.len() >= FRAMES as usize && tickets.len() >= FRAMES as usize {
            break;
        }
    }
    // A few more pumps, so a duplicate or a late ticket shows up.
    for _ in 0..4 {
        fabric.pump();
        collect(&mut sender, &mut receiver, &mut received, &mut tickets);
    }
    Ok(Exchange {
        key,
        sent,
        received,
        tickets,
        would_block,
        sender: sender.stats(),
        receiver: receiver.stats(),
    })
}

/// Takes every frame the receiver has, copying each out and dropping it, and every ticket
/// `completed` hands back.
fn collect(
    sender: &mut Channel,
    receiver: &mut Channel,
    received: &mut Vec<Received>,
    tickets: &mut Vec<u64>,
) {
    while let Some(frame) = receiver.try_recv() {
        let spec = frame.header.spec;
        received.push(Received {
            chan_seq: spec.chan_seq,
            ftype: spec.ftype,
            flags: spec.flags,
            instance: spec.instance,
            epoch: spec.epoch,
            step: spec.step,
            ext: frame.ext.bytes().to_vec(),
            payload: frame.payload.bytes().to_vec(),
            frame_len: frame.header.frame_len(),
        });
    }
    while let Some(ticket) = sender.completed() {
        tickets.push(ticket.chan_seq);
    }
}

impl Exchange {
    fn check(&self, rule: Rule) -> Checked {
        if self.sent.len() < FRAMES as usize {
            return fail(format!(
                "only {} of {FRAMES} frames could be sent",
                self.sent.len()
            ));
        }
        let all: Vec<u64> = (1..=u64::from(FRAMES)).collect();
        match rule {
            Rule::CompletedInOrder => {
                if self.tickets != all {
                    return fail(format!(
                        "completed handed back {:?}, expected {all:?}",
                        self.tickets
                    ));
                }
            }
            Rule::InOrder => {
                let seqs: Vec<u64> = self.received.iter().map(|r| r.chan_seq).collect();
                if seqs != all {
                    return fail(format!("received chan_seq {seqs:?}, expected {all:?}"));
                }
            }
            Rule::Intact => {
                for (sent, got) in self.sent.iter().zip(&self.received) {
                    let header = (got.ftype, got.flags, got.instance, got.epoch, got.step);
                    let want = (
                        FrameType::Activation,
                        Flags::default(),
                        self.key.instance,
                        self.key.epoch,
                        sent.step,
                    );
                    if header != want {
                        return fail(format!(
                            "frame {}: header (ftype, flags, instance, epoch, step) {header:?}, \
                             sent {want:?}",
                            got.chan_seq
                        ));
                    }
                    if got.ext != sent.ext {
                        return fail(format!("frame {}: the extension differs", got.chan_seq));
                    }
                    if got.payload != sent.payload {
                        return fail(format!("frame {}: the payload differs", got.chan_seq));
                    }
                }
                if self.received.len() < self.sent.len() {
                    return fail(format!(
                        "{} of {} frames arrived",
                        self.received.len(),
                        self.sent.len()
                    ));
                }
            }
            Rule::Stats => {
                let frames = self.sent.len() as u64;
                let bytes: u64 = self.sent.iter().map(Expected::frame_len).sum();
                let got_bytes: u64 = self.received.iter().map(|r| r.frame_len).sum();
                let checks = [
                    ("sender frames_sent", self.sender.frames_sent, frames),
                    ("sender bytes_sent", self.sender.bytes_sent, bytes),
                    (
                        "receiver frames_received",
                        self.receiver.frames_received,
                        frames,
                    ),
                    (
                        "receiver bytes_received",
                        self.receiver.bytes_received,
                        got_bytes,
                    ),
                    (
                        "sender credit_stalls",
                        self.sender.credit_stalls,
                        self.would_block,
                    ),
                ];
                for (name, got, want) in checks {
                    if got != want {
                        return fail(format!("{name} is {got}, expected {want}"));
                    }
                }
            }
            _ => unreachable!("not an exchange rule: {rule:?}"),
        }
        Ok(())
    }
}
