// SPDX-License-Identifier: MIT OR Apache-2.0

//! The fake runner that ingress is tested against (RFC-0050 "Performance model and budgets" and
//! "Testing and validation"; QUAL-E3-T18).
//!
//! An in-process stub of a head-and-tail runner. It serves the two M0 rings of a real `ipc`
//! [`Region`]: it reads `SequenceOpen` and `SequenceControl` records from `local_sequences` and
//! writes `TokenFrame` records to `local_tokens`, in the encoding of
//! [`syntropy_ipc::shm::records`], so the code under test speaks to it exactly as it speaks to a
//! runner. It answers each `Open` with a scripted token stream, honours `Cancel`, `Pause`,
//! `Resume` and `Release`, and records every control it receives. The full runner inside the
//! simulator is QUAL-E8-T06 (M1).
//!
//! **Setting up.** [`FakeRegion`] creates a region with the two rings and opens it on both sides
//! in this process. The code under test takes the daemon side's ends:
//! `region.daemon().producer(RingRole::LocalSequences)` and
//! `region.daemon().consumer(RingRole::LocalTokens)`. [`FakeRunner::new`] takes the runner side's.
//!
//! **Scripts.** A [`Script`] says what each opened sequence emits, as a [`Stream`] of steps. A
//! stream is found for an `Open` in this order: the stream given for its `SequenceId`
//! ([`Script::for_seq`]), then the next stream queued for opens in arrival order
//! ([`Script::in_order`]), since ingress picks the IDs, then the fallback
//! ([`Script::otherwise`]). An open with none of these is recorded as
//! [`Anomaly::Unscripted`] and emits nothing.
//!
//! **Steps.** The runner moves only when the test calls [`FakeRunner::step`], or while
//! [`FakeRunner::serve`] loops on another thread. One step:
//!
//! 1. reads every record waiting on `local_sequences`, logs it, and applies it;
//! 2. if an earlier step's frame found `local_tokens` full, tries it again, and ends there,
//!    whether it fits or not, so frames stay in step order;
//! 3. takes the next scripted step of every running sequence, in the order they were opened, and
//!    writes them as one `TokenFrame`, one entry per sequence, as the tail does (RFC-0001 §5.8).
//!
//! Controls are read before any token is written, so a `Cancel` the test pushes between two steps
//! stops the stream before its next token, with no timing involved. The tail stops emitting at a
//! `Cancel` and sends no closing entry (RFC-0032 §3.6). A frame already held for a full ring was
//! produced before the `Cancel` arrived: it is in flight, and is delivered, and ingress discards
//! it.
//!
//! **What is recorded.** [`FakeRunner::log`] holds every control received, in order, as decoded.
//! [`FakeRunner::anomalies`] holds what a correct ingress never does: an undecodable record, a
//! second `Open` for a live sequence, a control for a sequence that is not live, an unscripted
//! open. The emitted frames' `first_position` counts the prompt and the tokens emitted before;
//! `rng_counter` counts the tokens emitted, standing in for the sampler's draw counter.
//!
//! The runner holds no clock and no randomness; a run depends only on the script and on the order
//! in which records and steps arrive.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use smallvec::SmallVec;
use syntropy_core::{
    ApproxFlags, FinishReason, InstanceId, PlanEpoch, SequenceControl, SequenceId, SequenceOpen,
    TokenEntry, TokenFrame,
};
use syntropy_ipc::cmd::PROTOCOL;
use syntropy_ipc::shm::records::{self, RecordError};
use syntropy_ipc::shm::{EndError, Region, RingRole, RingSpec, ShmError, ShmLayout, Side};
use syntropy_spsc::shared::{SharedConsumer, SharedPopError, SharedProducer, SharedPushError};

use super::SimEntropy;

/// Slots per ring in [`layout`]'s default: a power of two.
pub const DEFAULT_SLOTS: u32 = 64;

/// The largest record in [`layout`]'s default, in bytes. It holds a `SequenceOpen` with a prompt of
/// about 21,000 tokens `[estimate]`: postcard writes a token ID below 2^21 in at most 3 bytes, so
/// 65,536 bytes less under 1 KiB for the other fields leave room for 21,500 IDs. An open that
/// carries a compiled grammar inline (`SamplingParams::grammar_inline`) holds fewer. A
/// `TokenFrame` is far smaller.
pub const DEFAULT_SLOT_BYTES: u32 = 64 << 10;

/// The build ID the fake region is created with.
pub const BUILD_ID: [u8; 32] = [0x5e; 32];

/// A region layout with the two rings the fake runner serves: `local_tokens` and
/// `local_sequences`, each of `slots` slots of `slot_bytes` bytes.
#[must_use]
pub fn layout(slots: u32, slot_bytes: u32) -> ShmLayout {
    ShmLayout {
        rings: vec![
            RingSpec {
                role: RingRole::LocalTokens,
                slots,
                slot_bytes,
            },
            RingSpec {
                role: RingRole::LocalSequences,
                slots,
                slot_bytes,
            },
        ],
    }
}

/// A region open on both sides in this process: the daemon's, for the code under test, and the
/// runner's, for [`FakeRunner`].
#[derive(Debug)]
pub struct FakeRegion {
    // `runner` is declared first so it is dropped, and its mapping closed, before the daemon's.
    runner: Region,
    daemon: Region,
}

impl FakeRegion {
    /// A region of [`layout`]`(`[`DEFAULT_SLOTS`]`, `[`DEFAULT_SLOT_BYTES`]`)`.
    ///
    /// # Errors
    ///
    /// As for [`FakeRegion::with_layout`].
    pub fn new() -> Result<FakeRegion, ShmError> {
        FakeRegion::with_layout(&layout(DEFAULT_SLOTS, DEFAULT_SLOT_BYTES))
    }

    /// A region of `layout`, created on the daemon's side and opened on the runner's, with this
    /// build's `ipc_protocol` and [`BUILD_ID`].
    ///
    /// # Errors
    ///
    /// The [`ShmError`] of [`Region::create`] or [`Region::open`].
    pub fn with_layout(layout: &ShmLayout) -> Result<FakeRegion, ShmError> {
        let daemon = Region::create(layout, BUILD_ID, PROTOCOL, &segment_entropy())?;
        let runner = Region::open(&daemon.handle(), Side::Runner, layout, PROTOCOL)?;
        Ok(FakeRegion { runner, daemon })
    }

    /// The daemon's side, whose `local_sequences` producer and `local_tokens` consumer the code
    /// under test takes.
    #[must_use]
    pub fn daemon(&self) -> &Region {
        &self.daemon
    }

    /// The runner's side, which [`FakeRunner::new`] takes.
    #[must_use]
    pub fn runner(&self) -> &Region {
        &self.runner
    }
}

/// The source of a segment name's random part. On Apple platforms and Windows the name is
/// global, so two test processes, or two regions in one, must not draw the same one: the seed
/// mixes this process's ID with a per-process counter. The bytes reach only the name.
fn segment_entropy() -> SimEntropy {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    SimEntropy::new(u64::from(std::process::id()) << 32 | n)
}

/// One scripted step of one sequence: its accepted tokens, and its finish reason if the
/// sequence ends there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScriptStep {
    /// Accepted tokens, in order: one for plain decode, several for a verify window.
    pub tokens: SmallVec<[u32; 8]>,
    /// Set on the sequence's last step.
    pub finish: Option<FinishReason>,
}

/// What one sequence emits, step by step.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stream {
    steps: Vec<ScriptStep>,
}

impl Stream {
    /// An empty stream: it emits nothing.
    #[must_use]
    pub fn new() -> Stream {
        Stream::default()
    }

    /// `tokens`, one per step: plain decode.
    #[must_use]
    pub fn tokens(mut self, tokens: impl IntoIterator<Item = u32>) -> Stream {
        self.steps
            .extend(tokens.into_iter().map(|token| ScriptStep {
                tokens: SmallVec::from_slice(&[token]),
                finish: None,
            }));
        self
    }

    /// `tokens` in one step, as a verify window accepts several.
    #[must_use]
    pub fn window(mut self, tokens: impl IntoIterator<Item = u32>) -> Stream {
        self.steps.push(ScriptStep {
            tokens: tokens.into_iter().collect(),
            finish: None,
        });
        self
    }

    /// Ends the stream with `reason`, on its last step, or on a step of its own with no tokens
    /// when it has none. A stream without a finish stops emitting after its last step and stays
    /// live until ingress releases it.
    #[must_use]
    pub fn finish(mut self, reason: FinishReason) -> Stream {
        match self.steps.last_mut() {
            Some(last) if last.finish.is_none() => last.finish = Some(reason),
            _ => self.steps.push(ScriptStep {
                tokens: SmallVec::new(),
                finish: Some(reason),
            }),
        }
        self
    }

    /// The scripted steps.
    #[must_use]
    pub fn steps(&self) -> &[ScriptStep] {
        &self.steps
    }

    /// Every token, in order.
    pub fn all_tokens(&self) -> impl Iterator<Item = u32> + '_ {
        self.steps
            .iter()
            .flat_map(|step| step.tokens.iter().copied())
    }
}

/// Which stream each opened sequence emits. See the module documentation for the order.
#[derive(Clone, Debug, Default)]
pub struct Script {
    by_seq: BTreeMap<SequenceId, Stream>,
    in_order: VecDeque<Stream>,
    otherwise: Option<Stream>,
}

impl Script {
    /// A script with no streams.
    #[must_use]
    pub fn new() -> Script {
        Script::default()
    }

    /// `stream` for the sequence `seq`.
    #[must_use]
    pub fn for_seq(mut self, seq: SequenceId, stream: Stream) -> Script {
        self.by_seq.insert(seq, stream);
        self
    }

    /// Queues `stream` for the next open that has no stream of its own, in arrival order.
    #[must_use]
    pub fn in_order(mut self, stream: Stream) -> Script {
        self.in_order.push_back(stream);
        self
    }

    /// `stream` for every open that nothing else scripts.
    #[must_use]
    pub fn otherwise(mut self, stream: Stream) -> Script {
        self.otherwise = Some(stream);
        self
    }

    fn take(&mut self, seq: SequenceId) -> Option<Stream> {
        self.by_seq
            .remove(&seq)
            .or_else(|| self.in_order.pop_front())
            .or_else(|| self.otherwise.clone())
    }
}

/// Where a live sequence stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SeqState {
    /// Emitting, or waiting for its script's next step.
    Running,
    /// Paused by `Pause`; emits again after `Resume`.
    Paused,
    /// Stopped by `Cancel`; emits nothing more.
    Cancelled,
    /// Emitted its scripted finish; emits nothing more.
    Finished,
}

/// Something a correct ingress never causes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Anomaly {
    /// A `local_sequences` record that did not decode.
    Undecodable(RecordError),
    /// An `Open` for a sequence that is already live.
    DuplicateOpen(SequenceId),
    /// A control other than `Open` for a sequence that was never opened or is already released.
    NotLive(SequenceId),
    /// An `Open` the script has no stream for. The sequence is live and emits nothing.
    Unscripted(SequenceId),
}

/// Why the fake runner stopped.
#[derive(Debug, PartialEq, Eq)]
pub enum FakeRunnerError {
    /// A ring end could not be taken.
    End(EndError),
    /// `local_sequences` broke the ring protocol.
    Pop(SharedPopError),
    /// `local_tokens` refused a frame for a reason other than being full.
    Push(SharedPushError),
    /// A frame did not encode.
    Encode(RecordError),
}

impl fmt::Display for FakeRunnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FakeRunnerError::End(error) => write!(f, "fake runner: {error}"),
            FakeRunnerError::Pop(error) => write!(f, "fake runner, local_sequences: {error}"),
            FakeRunnerError::Push(error) => write!(f, "fake runner, local_tokens: {error}"),
            FakeRunnerError::Encode(error) => write!(f, "fake runner: {error}"),
        }
    }
}

impl std::error::Error for FakeRunnerError {}

/// What one [`FakeRunner::step`] did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stepped {
    /// Controls read from `local_sequences`.
    pub received: usize,
    /// The frame written to `local_tokens`, if one was.
    pub sent: Option<TokenFrame>,
    /// Whether a frame is still waiting for room on `local_tokens`.
    pub held: bool,
}

/// A live sequence.
#[derive(Debug)]
struct Live {
    seq: SequenceId,
    state: SeqState,
    steps: VecDeque<ScriptStep>,
    /// The position of the next token.
    position: u32,
    /// Tokens emitted so far.
    emitted: u64,
}

/// The stub runner. See the module documentation.
pub struct FakeRunner<'r> {
    controls: SharedConsumer<'r>,
    tokens: SharedProducer<'r>,
    script: Script,
    instance: InstanceId,
    epoch: PlanEpoch,
    /// Live sequences, in the order they were opened.
    live: Vec<Live>,
    /// Frames written.
    step: u64,
    log: Vec<SequenceControl>,
    anomalies: Vec<Anomaly>,
    /// A frame, encoded, that `local_tokens` had no room for, and the frame itself.
    held: Option<(Vec<u8>, TokenFrame)>,
    /// The buffer records are read into: one slot's size.
    record: Vec<u8>,
    /// The buffer frames are encoded into.
    out: Vec<u8>,
}

impl fmt::Debug for FakeRunner<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FakeRunner")
            .field("step", &self.step)
            .field("live", &self.live)
            .field("received", &self.log.len())
            .field("anomalies", &self.anomalies)
            .finish_non_exhaustive()
    }
}

impl<'r> FakeRunner<'r> {
    /// A fake runner on `region`'s runner side, serving `script`. Its frames name instance 0 and
    /// the default plan epoch; [`FakeRunner::with_identity`] changes them.
    ///
    /// # Errors
    ///
    /// [`FakeRunnerError::End`] if `region` is not open on the runner's side, lacks either ring,
    /// or already handed out its ends.
    pub fn new(region: &'r Region, script: Script) -> Result<FakeRunner<'r>, FakeRunnerError> {
        let controls = region
            .consumer(RingRole::LocalSequences)
            .map_err(FakeRunnerError::End)?;
        let tokens = region
            .producer(RingRole::LocalTokens)
            .map_err(FakeRunnerError::End)?;
        let slot_bytes = controls.layout().slot_bytes() as usize;
        Ok(FakeRunner {
            controls,
            tokens,
            script,
            instance: InstanceId(0),
            epoch: PlanEpoch::default(),
            live: Vec::new(),
            step: 0,
            log: Vec::new(),
            anomalies: Vec::new(),
            held: None,
            record: vec![0; slot_bytes],
            out: Vec::new(),
        })
    }

    /// Names `instance` and `epoch` in every frame from now on.
    #[must_use]
    pub fn with_identity(mut self, instance: InstanceId, epoch: PlanEpoch) -> FakeRunner<'r> {
        self.instance = instance;
        self.epoch = epoch;
        self
    }

    /// Every control received, in order.
    #[must_use]
    pub fn log(&self) -> &[SequenceControl] {
        &self.log
    }

    /// What a correct ingress never causes, in the order it happened.
    #[must_use]
    pub fn anomalies(&self) -> &[Anomaly] {
        &self.anomalies
    }

    /// The state of `seq`, or `None` when it was never opened or has been released.
    #[must_use]
    pub fn state(&self, seq: SequenceId) -> Option<SeqState> {
        self.find(seq).map(|at| self.live[at].state)
    }

    /// The frames written so far; the next frame's `step`.
    #[must_use]
    pub fn steps(&self) -> u64 {
        self.step
    }

    /// Whether a step would do nothing: no running sequence has a scripted step left, and no
    /// frame is held. Records waiting on `local_sequences` are not counted.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.held.is_none()
            && !self
                .live
                .iter()
                .any(|live| live.state == SeqState::Running && !live.steps.is_empty())
    }

    /// Reads, logs and applies every record waiting on `local_sequences`; the number read.
    ///
    /// # Errors
    ///
    /// [`FakeRunnerError::Pop`] when the ring's indices or a record's length are out of range.
    pub fn receive(&mut self) -> Result<usize, FakeRunnerError> {
        let mut received = 0;
        loop {
            let len = match self.controls.pop_into(&mut self.record) {
                Ok(len) => len,
                Err(SharedPopError::Empty | SharedPopError::Closed) => return Ok(received),
                // Unreachable with a buffer of the ring's slot size; grow it and read again.
                Err(SharedPopError::BufferTooSmall { len }) => {
                    self.record.resize(len, 0);
                    continue;
                }
                Err(error) => return Err(FakeRunnerError::Pop(error)),
            };
            received += 1;
            match records::decode_control(&self.record[..len]) {
                Ok(control) => {
                    self.apply(&control);
                    self.log.push(control);
                }
                Err(error) => self.anomalies.push(Anomaly::Undecodable(error)),
            }
        }
    }

    /// One step: reads the controls, delivers a held frame, then writes the next frame. See the
    /// module documentation.
    ///
    /// # Errors
    ///
    /// [`FakeRunnerError::Pop`] as for [`FakeRunner::receive`]; [`FakeRunnerError::Push`] when
    /// `local_tokens`' consumer is gone or broke the ring protocol, or a frame is larger than a
    /// slot.
    pub fn step(&mut self) -> Result<Stepped, FakeRunnerError> {
        let received = self.receive()?;
        if let Some((record, frame)) = self.held.take() {
            if !push(&mut self.tokens, &record)? {
                self.held = Some((record, frame));
                return Ok(Stepped {
                    received,
                    sent: None,
                    held: true,
                });
            }
            return Ok(Stepped {
                received,
                sent: Some(frame),
                held: false,
            });
        }
        let Some(frame) = self.next_frame() else {
            return Ok(Stepped {
                received,
                sent: None,
                held: false,
            });
        };
        records::encode_tokens(&frame, &mut self.out).map_err(FakeRunnerError::Encode)?;
        if push(&mut self.tokens, &self.out)? {
            Ok(Stepped {
                received,
                sent: Some(frame),
                held: false,
            })
        } else {
            self.held = Some((self.out.clone(), frame));
            Ok(Stepped {
                received,
                sent: None,
                held: true,
            })
        }
    }

    /// Steps until [`FakeRunner::is_idle`] after a step, or `max_steps` steps; the steps taken.
    ///
    /// # Errors
    ///
    /// As for [`FakeRunner::step`].
    pub fn run_until_idle(&mut self, max_steps: u64) -> Result<u64, FakeRunnerError> {
        let mut taken = 0;
        while taken < max_steps {
            self.step()?;
            taken += 1;
            if self.is_idle() {
                break;
            }
        }
        Ok(taken)
    }

    /// Steps until `stop` is set, yielding the thread whenever a step does nothing. For a test
    /// whose code under test runs on its own threads; it runs on a scoped thread of its own. The
    /// order of tokens and controls is the same as when stepping by hand, but the step a control
    /// arrives at depends on thread timing: a test of cancel timing steps by hand instead.
    ///
    /// # Errors
    ///
    /// As for [`FakeRunner::step`].
    pub fn serve(&mut self, stop: &AtomicBool) -> Result<(), FakeRunnerError> {
        while !stop.load(Ordering::Acquire) {
            let stepped = self.step()?;
            if stepped.received == 0 && stepped.sent.is_none() {
                std::thread::yield_now();
            }
        }
        Ok(())
    }

    fn find(&self, seq: SequenceId) -> Option<usize> {
        self.live.iter().position(|live| live.seq == seq)
    }

    fn apply(&mut self, control: &SequenceControl) {
        let (seq, next) = match control {
            SequenceControl::Open(open) => return self.open(open),
            SequenceControl::Cancel { seq, .. } => (*seq, Some(SeqState::Cancelled)),
            SequenceControl::Pause { seq } => (*seq, Some(SeqState::Paused)),
            SequenceControl::Resume { seq } => (*seq, Some(SeqState::Running)),
            SequenceControl::Release { seq, .. } => (*seq, None),
            // `SequenceControl` is non-exhaustive; a variant added later is logged and ignored.
            _ => return,
        };
        let Some(at) = self.find(seq) else {
            self.anomalies.push(Anomaly::NotLive(seq));
            return;
        };
        let live = &mut self.live[at];
        match next {
            None => {
                self.live.remove(at);
            }
            // Pause and Resume move only a sequence still emitting.
            Some(SeqState::Paused) if live.state == SeqState::Running => {
                live.state = SeqState::Paused
            }
            Some(SeqState::Running) if live.state == SeqState::Paused => {
                live.state = SeqState::Running
            }
            Some(SeqState::Cancelled) if live.state != SeqState::Finished => {
                live.state = SeqState::Cancelled;
            }
            Some(_) => {}
        }
    }

    fn open(&mut self, open: &SequenceOpen) {
        if self.find(open.seq).is_some() {
            self.anomalies.push(Anomaly::DuplicateOpen(open.seq));
            return;
        }
        let stream = self.script.take(open.seq).unwrap_or_else(|| {
            self.anomalies.push(Anomaly::Unscripted(open.seq));
            Stream::new()
        });
        self.live.push(Live {
            seq: open.seq,
            state: SeqState::Running,
            steps: stream.steps.into(),
            position: u32::try_from(open.prompt.len()).unwrap_or(u32::MAX),
            emitted: 0,
        });
    }

    /// The next frame: the next scripted step of every running sequence, or `None` when no
    /// sequence has one.
    fn next_frame(&mut self) -> Option<TokenFrame> {
        let mut entries = SmallVec::new();
        for live in &mut self.live {
            if live.state != SeqState::Running {
                continue;
            }
            let Some(step) = live.steps.pop_front() else {
                continue;
            };
            let count = step.tokens.len();
            entries.push(TokenEntry {
                seq: live.seq,
                tokens: step.tokens,
                first_position: live.position,
                logprobs: None,
                finish: step.finish,
                approx: ApproxFlags::default(),
                draft: SmallVec::new(),
                draft_tree: None,
                rng_counter: live.emitted + count as u64,
                tap: None,
            });
            live.position = live.position.saturating_add(count as u32);
            live.emitted += count as u64;
            if step.finish.is_some() {
                live.state = SeqState::Finished;
                live.steps.clear();
            }
        }
        if entries.is_empty() {
            return None;
        }
        let frame = TokenFrame {
            instance: self.instance,
            epoch: self.epoch,
            step: self.step,
            entries,
        };
        self.step += 1;
        Some(frame)
    }
}

/// Pushes one record to `ring`; whether it fit.
fn push(ring: &mut SharedProducer<'_>, record: &[u8]) -> Result<bool, FakeRunnerError> {
    match ring.push(record) {
        Ok(()) => Ok(true),
        Err(SharedPushError::Full) => Ok(false),
        Err(error) => Err(FakeRunnerError::Push(error)),
    }
}
