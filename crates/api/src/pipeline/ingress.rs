// SPDX-License-Identifier: MIT OR Apache-2.0

//! The node's ingress over one runner's rings: the `local_sequences` producer, the
//! `local_tokens` consumer, and the live sequences with their transcripts.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use syntropy_core::{Clock, Entropy, NodeId, SequenceControl, SequenceId, TokenEntry};
use syntropy_ipc::shm::records::{decode_tokens, encode_control};
use syntropy_ipc::shm::{EndError, Region, RingRole};
use syntropy_spsc::shared::{SharedConsumer, SharedPopError, SharedProducer, SharedPushError};

use super::stages::Stages;
use super::types::ApiError;

/// The node's one source of `SequenceId`s for this boot (RFC-0001 §5.9): the node's short ID, its
/// boot counter, and a per-boot counter `n`.
///
/// One per node per boot, held by the daemon and lent to every [`Ingress`], one per runner, so two
/// ingresses on one node never mint the same ID. `n` runs from 0 to `u32::MAX`; after that the
/// minter refuses with [`ApiError::SequenceIdsExhausted`] and never wraps, since a wrapped ID
/// would name a sequence another may still hold.
#[derive(Debug)]
pub struct SequenceIdMinter {
    node: NodeId,
    boot: u32,
    /// The next `n`. A `u64`, so taking one cannot wrap before the refusal.
    next: AtomicU64,
}

impl SequenceIdMinter {
    /// The minter for `node`'s boot `boot`, starting at `n = 0`.
    #[must_use]
    pub fn new(node: NodeId, boot: u32) -> SequenceIdMinter {
        SequenceIdMinter {
            node,
            boot,
            next: AtomicU64::new(0),
        }
    }

    /// The node the IDs name.
    #[must_use]
    pub fn node(&self) -> NodeId {
        self.node
    }

    /// The next `SequenceId`.
    ///
    /// # Errors
    ///
    /// [`ApiError::SequenceIdsExhausted`] once `n = u32::MAX` has been minted. Each call adds one
    /// to a `u64`, so no realistic number of calls wraps it.
    pub fn mint(&self) -> Result<SequenceId, ApiError> {
        let n = self.next.fetch_add(1, Ordering::Relaxed);
        let n = u32::try_from(n).map_err(|_| ApiError::SequenceIdsExhausted)?;
        Ok(SequenceId::new(&self.node, self.boot, n))
    }
}

/// One live sequence's transcript (RFC-0050 §2.2; RFC-0001 §6.1 step 5): its prompt, the tokens
/// emitted and counted so far, and the latest draw counter. Ingress keeps it until it sends the
/// sequence's `Release`; it is what re-prefill recovery and migration rebuild from.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Transcript {
    /// The rendered prompt's tokens.
    pub prompt: Vec<u32>,
    /// The tokens emitted and counted, in order. Tokens after a matched stop string are not in
    /// it.
    pub emitted: Vec<u32>,
    /// The sampler's draw counter after the last entry taken (`TokenEntry::rng_counter`).
    pub rng_counter: u64,
}

/// The ingress side of one runner's rings, shared by every request it serves.
///
/// Any number of [`Generation`](super::Generation)s run over one `Ingress`, from any threads.
/// Each call that reads `local_tokens` drains it into the live sequences' queues, so frames reach
/// their sequences however the generations are polled. Entries for a sequence that is not live,
/// such as the in-flight tokens after a `Cancel` (RFC-0001 §6.2), are dropped and counted
/// ([`Ingress::stray_entries`]).
///
/// Controls are never lost. A `Cancel` or `Release` that finds `local_sequences` full waits in
/// order, and every later call to the ring sends what waits first. An `Open` that finds the ring
/// full fails its request with [`ApiError::Capacity`] instead, since it has not started. Nothing
/// here waits for room: the owner calls [`Ingress::flush`] until [`Ingress::waiting_controls`] is
/// 0, or a generation's `Release` may wait until the next request reaches the ring.
pub struct Ingress<'r> {
    pub(super) minter: &'r SequenceIdMinter,
    pub(super) stages: &'r dyn Stages,
    pub(super) clock: &'r dyn Clock,
    pub(super) entropy: &'r dyn Entropy,
    /// `obs.trace.honor_remote_sampled`: whether an incoming `traceparent`'s sampled bit forces
    /// a full trace (RFC-0053 §3.2). Off unless the daemon turns it on.
    pub(super) honor_remote_sampled: std::sync::atomic::AtomicBool,
    outbox: Mutex<Outbox<'r>>,
    inbox: Mutex<Inbox<'r>>,
}

impl std::fmt::Debug for Ingress<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ingress")
            .field("node", &self.minter.node())
            .field("live", &self.inbox().live.len())
            .finish_non_exhaustive()
    }
}

/// `local_sequences`, and the controls waiting for room on it.
struct Outbox<'r> {
    ring: SharedProducer<'r>,
    waiting: VecDeque<Vec<u8>>,
}

/// `local_tokens`, and the live sequences its entries go to.
struct Inbox<'r> {
    ring: SharedConsumer<'r>,
    buffer: Vec<u8>,
    live: BTreeMap<SequenceId, Live>,
    stray: u64,
    broken: Option<String>,
}

/// A live sequence: entries not yet taken, and its transcript.
#[derive(Default)]
struct Live {
    queue: VecDeque<TokenEntry>,
    transcript: Transcript,
}

/// Why a control could not be sent.
pub(super) enum SendError {
    /// The ring is full; for an `Open`, which is not queued.
    Full,
    /// The record does not fit a slot.
    TooLong,
    /// The runner is gone or broke the ring protocol.
    Broken(String),
}

impl<'r> Ingress<'r> {
    /// The ingress over `region`'s daemon side: it takes the region's `local_sequences` producer
    /// and `local_tokens` consumer. `minter` is the node's one [`SequenceIdMinter`], shared with
    /// every other ingress on the node. `stages` computes stages 3 to 8 and the text state of
    /// stage 10; `clock` times each request, and `entropy` draws the seeds of requests without
    /// one.
    ///
    /// # Errors
    ///
    /// The [`EndError`] when the region lacks either ring, is open on the runner's side, or
    /// already handed out either end.
    pub fn new(
        region: &'r Region,
        minter: &'r SequenceIdMinter,
        stages: &'r dyn Stages,
        clock: &'r dyn Clock,
        entropy: &'r dyn Entropy,
    ) -> Result<Ingress<'r>, EndError> {
        let controls = region.producer(RingRole::LocalSequences)?;
        let tokens = region.consumer(RingRole::LocalTokens)?;
        let slot = tokens.layout().slot_bytes() as usize;
        Ok(Ingress {
            minter,
            stages,
            clock,
            entropy,
            honor_remote_sampled: std::sync::atomic::AtomicBool::new(false),
            outbox: Mutex::new(Outbox {
                ring: controls,
                waiting: VecDeque::new(),
            }),
            inbox: Mutex::new(Inbox {
                ring: tokens,
                buffer: vec![0; slot],
                live: BTreeMap::new(),
                stray: 0,
                broken: None,
            }),
        })
    }

    /// Sets `obs.trace.honor_remote_sampled` for the requests that start from now on (a hot key,
    /// RFC-0053 §3.2): whether an incoming `traceparent`'s sampled bit forces a full trace. Off by
    /// default, so an untrusted client cannot make the node trace every step.
    pub fn set_honor_remote_sampled(&self, on: bool) {
        self.honor_remote_sampled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// The transcript of `seq` while it is live, or `None` once ingress has released it or never
    /// opened it.
    #[must_use]
    pub fn transcript(&self, seq: SequenceId) -> Option<Transcript> {
        self.inbox()
            .live
            .get(&seq)
            .map(|live| live.transcript.clone())
    }

    /// The live sequences, in `SequenceId` order.
    #[must_use]
    pub fn live(&self) -> Vec<SequenceId> {
        self.inbox().live.keys().copied().collect()
    }

    /// Entries dropped because their sequence was not live: tokens in flight after a `Cancel`
    /// or a finish.
    #[must_use]
    pub fn stray_entries(&self) -> u64 {
        self.inbox().stray
    }

    /// Controls waiting for room on `local_sequences`.
    #[must_use]
    pub fn waiting_controls(&self) -> usize {
        self.outbox().waiting.len()
    }

    /// Sends the controls that wait for room, in order, as far as the ring takes them.
    ///
    /// # Errors
    ///
    /// [`ApiError::RunnerUnavailable`] when the runner is gone or broke the ring protocol.
    pub fn flush(&self) -> Result<(), ApiError> {
        flush(&mut self.outbox()).map_err(ApiError::RunnerUnavailable)
    }

    /// Makes `seq` live with `prompt` as its transcript's prompt, so its entries are kept from
    /// now on.
    pub(super) fn register(&self, seq: SequenceId, prompt: Vec<u32>) {
        self.inbox().live.insert(
            seq,
            Live {
                queue: VecDeque::new(),
                transcript: Transcript {
                    prompt,
                    ..Transcript::default()
                },
            },
        );
    }

    /// Ends `seq`: its transcript and any entries not taken are dropped, and later entries for it
    /// are strays.
    pub(super) fn unregister(&self, seq: SequenceId) {
        self.inbox().live.remove(&seq);
    }

    /// Adds `token` to `seq`'s transcript.
    pub(super) fn record(&self, seq: SequenceId, token: u32) {
        if let Some(live) = self.inbox().live.get_mut(&seq) {
            live.transcript.emitted.push(token);
        }
    }

    /// Sets `seq`'s draw counter.
    pub(super) fn record_counter(&self, seq: SequenceId, rng_counter: u64) {
        if let Some(live) = self.inbox().live.get_mut(&seq) {
            live.transcript.rng_counter = rng_counter;
        }
    }

    /// Drains `local_tokens`, then takes `seq`'s queued entries.
    ///
    /// # Errors
    ///
    /// [`ApiError::RunnerUnavailable`] once the ring has broken its protocol or a frame failed to
    /// decode.
    pub(super) fn take(&self, seq: SequenceId) -> Result<VecDeque<TokenEntry>, ApiError> {
        let mut inbox = self.inbox();
        drain(&mut inbox);
        if let Some(why) = &inbox.broken {
            return Err(ApiError::RunnerUnavailable(why.clone()));
        }
        Ok(inbox
            .live
            .get_mut(&seq)
            .map(|live| std::mem::take(&mut live.queue))
            .unwrap_or_default())
    }

    /// Sends `control`. An `Open` is sent now or not at all; any other control waits for room
    /// when the ring is full.
    pub(super) fn send(&self, control: &SequenceControl) -> Result<(), SendError> {
        let mut record = Vec::new();
        // Every field of a control serializes; postcard fails only on types these do not hold.
        encode_control(control, &mut record)
            .map_err(|error| SendError::Broken(format!("a control did not encode: {error}")))?;
        let queueable = !matches!(control, SequenceControl::Open(_));
        let mut outbox = self.outbox();
        flush(&mut outbox).map_err(SendError::Broken)?;
        if !outbox.waiting.is_empty() {
            if queueable {
                outbox.waiting.push_back(record);
                return Ok(());
            }
            return Err(SendError::Full);
        }
        match outbox.ring.push(&record) {
            Ok(()) => Ok(()),
            Err(SharedPushError::Full) if queueable => {
                outbox.waiting.push_back(record);
                Ok(())
            }
            Err(SharedPushError::Full) => Err(SendError::Full),
            Err(SharedPushError::TooLong { .. }) => Err(SendError::TooLong),
            Err(error) => Err(SendError::Broken(error.to_string())),
        }
    }

    fn inbox(&self) -> MutexGuard<'_, Inbox<'r>> {
        // Nothing that holds the lock panics, so a poisoned lock holds consistent state.
        self.inbox.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn outbox(&self) -> MutexGuard<'_, Outbox<'r>> {
        self.outbox.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Sends what waits on `outbox`, in order, until the ring is full; the reason the ring broke, if
/// it did.
fn flush(outbox: &mut Outbox<'_>) -> Result<(), String> {
    while let Some(record) = outbox.waiting.front() {
        match outbox.ring.push(record) {
            Ok(()) => {
                outbox.waiting.pop_front();
            }
            Err(SharedPushError::Full) => return Ok(()),
            Err(error) => return Err(error.to_string()),
        }
    }
    Ok(())
}

/// Reads every frame waiting on `local_tokens` into the live sequences' queues.
fn drain(inbox: &mut Inbox<'_>) {
    if inbox.broken.is_some() {
        return;
    }
    loop {
        let len = match inbox.ring.pop_into(&mut inbox.buffer) {
            Ok(len) => len,
            Err(SharedPopError::Empty | SharedPopError::Closed) => return,
            Err(SharedPopError::BufferTooSmall { len }) => {
                inbox.buffer.resize(len, 0);
                continue;
            }
            Err(SharedPopError::Corrupt) => {
                inbox.broken = Some("local_tokens broke the ring protocol".to_owned());
                return;
            }
        };
        let frame = match decode_tokens(&inbox.buffer[..len]) {
            Ok(frame) => frame,
            Err(error) => {
                inbox.broken = Some(format!("a local_tokens frame did not decode: {error}"));
                return;
            }
        };
        for entry in frame.entries {
            match inbox.live.get_mut(&entry.seq) {
                Some(live) => live.queue.push_back(entry),
                None => inbox.stray += 1,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// IDs count up from 0 for the node and boot, and the minter refuses after `n = u32::MAX`
    /// instead of wrapping to 0.
    #[test]
    fn the_minter_counts_up_and_refuses_to_wrap() {
        let node = NodeId([2; 32]);
        let minter = SequenceIdMinter::new(node, 4);
        assert_eq!(minter.mint(), Ok(SequenceId::new(&node, 4, 0)));
        assert_eq!(minter.mint(), Ok(SequenceId::new(&node, 4, 1)));
        minter.next.store(u64::from(u32::MAX), Ordering::Relaxed);
        assert_eq!(minter.mint(), Ok(SequenceId::new(&node, 4, u32::MAX)));
        assert_eq!(minter.mint(), Err(ApiError::SequenceIdsExhausted));
        assert_eq!(minter.mint(), Err(ApiError::SequenceIdsExhausted));
    }
}
