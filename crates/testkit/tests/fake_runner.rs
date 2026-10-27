// SPDX-License-Identifier: MIT OR Apache-2.0

//! The fake runner's stub tests (QUAL-E3-T18). Each test plays ingress on the daemon side of a
//! real region: it pushes encoded controls on `local_sequences`, steps the runner by hand, and
//! pops encoded frames from `local_tokens`. Nothing waits on time; the one threaded test checks
//! order only.

#![cfg(any(target_os = "linux", target_vendor = "apple", windows))]

use std::sync::atomic::{AtomicBool, Ordering};

use syntropy_core::{
    CancelReason, FinishReason, InstanceId, NodeId, PlanEpoch, SamplingParams, SequenceControl,
    SequenceId, SequenceLimits, SequenceOpen, TokenFrame,
};
use syntropy_ipc::shm::RingRole;
use syntropy_ipc::shm::records::{decode_tokens, encode_control};
use syntropy_spsc::shared::{SharedConsumer, SharedPopError, SharedProducer};
use syntropy_testkit::sim::fake_runner::{
    self, Anomaly, FakeRegion, FakeRunner, Script, SeqState, Stream,
};

const INGRESS: NodeId = NodeId([4; 32]);

fn seq(n: u32) -> SequenceId {
    SequenceId::new(&INGRESS, 1, n)
}

fn open(n: u32, prompt: Vec<u32>) -> SequenceControl {
    SequenceControl::Open(SequenceOpen {
        seq: seq(n),
        session: None,
        ingress: INGRESS,
        prompt: prompt.into(),
        prefix: None,
        sampling: SamplingParams::default(),
        limits: SequenceLimits {
            max_tokens: 64,
            ..SequenceLimits::default()
        },
        exactness: Default::default(),
        priority: Default::default(),
        choice_index: None,
        prefill_chunk: 0,
        initial_window: None,
        resume: None,
        prefix_scope: Default::default(),
        session_pk: None,
        trace: None,
        parity: None,
    })
}

fn cancel(n: u32) -> SequenceControl {
    SequenceControl::Cancel {
        seq: seq(n),
        reason: CancelReason::ClientGone,
    }
}

fn release(n: u32) -> SequenceControl {
    SequenceControl::Release {
        seq: seq(n),
        persist: None,
    }
}

/// Ingress's ends of the region.
struct Ingress<'r> {
    controls: SharedProducer<'r>,
    tokens: SharedConsumer<'r>,
    sent: Vec<SequenceControl>,
}

impl<'r> Ingress<'r> {
    fn new(region: &'r FakeRegion) -> Ingress<'r> {
        Ingress {
            controls: region.daemon().producer(RingRole::LocalSequences).unwrap(),
            tokens: region.daemon().consumer(RingRole::LocalTokens).unwrap(),
            sent: Vec::new(),
        }
    }

    fn send(&mut self, control: SequenceControl) {
        let mut record = Vec::new();
        encode_control(&control, &mut record).unwrap();
        self.controls.push(&record).unwrap();
        self.sent.push(control);
    }

    /// Every frame waiting on `local_tokens`.
    fn frames(&mut self) -> Vec<TokenFrame> {
        let mut out = vec![0; fake_runner::DEFAULT_SLOT_BYTES as usize];
        let mut frames = Vec::new();
        loop {
            match self.tokens.pop_into(&mut out) {
                Ok(len) => frames.push(decode_tokens(&out[..len]).unwrap()),
                Err(SharedPopError::Empty) => return frames,
                Err(error) => panic!("local_tokens: {error}"),
            }
        }
    }
}

/// `seq`'s tokens across `frames`, in order.
fn tokens_of(frames: &[TokenFrame], seq: SequenceId) -> Vec<u32> {
    frames
        .iter()
        .flat_map(|frame| &frame.entries)
        .filter(|entry| entry.seq == seq)
        .flat_map(|entry| entry.tokens.iter().copied())
        .collect()
}

/// Acceptance 1: a scripted stream, single tokens then a verify window, arrives in order on
/// `local_tokens`, one frame per step with consecutive step numbers, positions after the prompt,
/// and the finish on the last entry.
#[test]
fn a_scripted_stream_arrives_in_order() {
    let region = FakeRegion::new().unwrap();
    let mut ingress = Ingress::new(&region);
    let stream = Stream::new()
        .tokens([101, 102, 103])
        .window([104, 105, 106])
        .tokens([107])
        .finish(FinishReason::Stop);
    let script = Script::new().for_seq(seq(1), stream.clone());
    let instance = InstanceId(5);
    let epoch = PlanEpoch { term: 2, seq: 3 };
    let mut runner = FakeRunner::new(region.runner(), script)
        .unwrap()
        .with_identity(instance, epoch);

    ingress.send(open(1, vec![1, 2, 3, 4]));
    let taken = runner.run_until_idle(100).unwrap();
    assert_eq!(taken, 5, "one step per scripted step");

    let frames = ingress.frames();
    assert_eq!(frames.len(), 5);
    assert_eq!(
        tokens_of(&frames, seq(1)),
        stream.all_tokens().collect::<Vec<_>>()
    );
    let mut position = 4;
    for (step, frame) in frames.iter().enumerate() {
        assert_eq!(frame.step, step as u64);
        assert_eq!((frame.instance, frame.epoch), (instance, epoch));
        assert_eq!(frame.entries.len(), 1);
        let entry = &frame.entries[0];
        assert_eq!(entry.first_position, position);
        position += entry.tokens.len() as u32;
        assert_eq!(entry.rng_counter, u64::from(position - 4));
        let last = step == frames.len() - 1;
        assert_eq!(entry.finish, last.then_some(FinishReason::Stop));
    }
    assert_eq!(runner.state(seq(1)), Some(SeqState::Finished));
    assert!(runner.anomalies().is_empty());
}

/// Two sequences share each step's frame, in the order they were opened, and opens without a
/// stream of their own take the queued streams in arrival order.
#[test]
fn concurrent_streams_share_frames_in_open_order() {
    let region = FakeRegion::new().unwrap();
    let mut ingress = Ingress::new(&region);
    let script = Script::new()
        .in_order(Stream::new().tokens([10, 11]).finish(FinishReason::Length))
        .in_order(
            Stream::new()
                .tokens([20, 21, 22])
                .finish(FinishReason::Stop),
        );
    let mut runner = FakeRunner::new(region.runner(), script).unwrap();

    ingress.send(open(7, vec![1]));
    ingress.send(open(3, vec![1]));
    runner.run_until_idle(100).unwrap();

    let frames = ingress.frames();
    assert_eq!(frames.len(), 3);
    let order: Vec<Vec<SequenceId>> = frames
        .iter()
        .map(|frame| frame.entries.iter().map(|entry| entry.seq).collect())
        .collect();
    assert_eq!(
        order,
        [vec![seq(7), seq(3)], vec![seq(7), seq(3)], vec![seq(3)]]
    );
    assert_eq!(tokens_of(&frames, seq(7)), [10, 11]);
    assert_eq!(tokens_of(&frames, seq(3)), [20, 21, 22]);
}

/// Acceptance 2: a `Cancel` pushed between two steps stops the stream before its next token,
/// and the sequence still running beside it is untouched.
#[test]
fn a_cancel_stops_the_stream_before_the_next_token() {
    let region = FakeRegion::new().unwrap();
    let mut ingress = Ingress::new(&region);
    let script = Script::new()
        .for_seq(seq(1), Stream::new().tokens(1..=10))
        .for_seq(seq(2), Stream::new().tokens(21..=24));
    let mut runner = FakeRunner::new(region.runner(), script).unwrap();

    ingress.send(open(1, vec![9]));
    ingress.send(open(2, vec![9]));
    runner.step().unwrap();
    runner.step().unwrap();
    assert_eq!(tokens_of(&ingress.frames(), seq(1)), [1, 2]);

    ingress.send(cancel(1));
    let stepped = runner.step().unwrap();
    assert_eq!(stepped.received, 1);
    let sent = stepped.sent.expect("seq 2 still emits");
    assert!(sent.entries.iter().all(|entry| entry.seq != seq(1)));
    assert_eq!(runner.state(seq(1)), Some(SeqState::Cancelled));

    runner.run_until_idle(100).unwrap();
    let after = ingress.frames();
    assert!(
        tokens_of(&after, seq(1)).is_empty(),
        "no token after the cancel"
    );
    assert_eq!(tokens_of(&after, seq(2)), [23, 24]);
    assert!(runner.is_idle());
    assert!(runner.anomalies().is_empty());
}

/// Acceptance 3: the recorded log equals what the test sent, in order, across every control
/// variant, and `Release` frees the sequence.
#[test]
fn the_log_equals_what_was_sent() {
    let region = FakeRegion::new().unwrap();
    let mut ingress = Ingress::new(&region);
    let mut runner = FakeRunner::new(
        region.runner(),
        Script::new().otherwise(Stream::new().tokens(1..=6)),
    )
    .unwrap();

    ingress.send(open(1, vec![5; 300]));
    ingress.send(open(2, vec![6]));
    runner.step().unwrap();
    ingress.send(SequenceControl::Pause { seq: seq(1) });
    runner.step().unwrap();
    assert_eq!(runner.state(seq(1)), Some(SeqState::Paused));
    ingress.send(SequenceControl::Resume { seq: seq(1) });
    ingress.send(cancel(2));
    runner.step().unwrap();
    ingress.send(release(2));
    ingress.send(release(1));
    runner.step().unwrap();

    assert_eq!(runner.log(), ingress.sent.as_slice());
    assert_eq!(runner.state(seq(1)), None);
    assert_eq!(runner.state(seq(2)), None);
    assert!(runner.anomalies().is_empty());

    // Paused for one step: seq 1 emitted at steps 0 and 2, seq 2 at steps 0 and 1.
    let frames = ingress.frames();
    assert_eq!(tokens_of(&frames, seq(1)), [1, 2]);
    assert_eq!(tokens_of(&frames, seq(2)), [1, 2]);
}

/// What a correct ingress never does is recorded, and the log still holds every control.
#[test]
fn misuse_is_recorded_as_anomalies() {
    let region = FakeRegion::new().unwrap();
    let mut ingress = Ingress::new(&region);
    let mut runner = FakeRunner::new(region.runner(), Script::new()).unwrap();

    ingress.send(open(1, vec![1]));
    ingress.send(open(1, vec![1]));
    ingress.send(cancel(9));
    ingress.controls.push(&[0xff, 0xff]).unwrap();
    runner.step().unwrap();

    assert_eq!(runner.log(), ingress.sent.as_slice());
    let anomalies = runner.anomalies();
    assert_eq!(anomalies.len(), 4, "{anomalies:?}");
    assert_eq!(anomalies[0], Anomaly::Unscripted(seq(1)));
    assert_eq!(anomalies[1], Anomaly::DuplicateOpen(seq(1)));
    assert_eq!(anomalies[2], Anomaly::NotLive(seq(9)));
    assert!(matches!(anomalies[3], Anomaly::Undecodable(_)));
    assert!(ingress.frames().is_empty());
}

/// `local_tokens` is lossless: a frame that finds the ring full is held and delivered first once
/// ingress reads, and no step's frame is lost or reordered.
#[test]
fn a_full_ring_holds_the_frame_until_there_is_room() {
    let region = FakeRegion::with_layout(&fake_runner::layout(2, 4096)).unwrap();
    let mut ingress = Ingress::new(&region);
    let mut runner = FakeRunner::new(
        region.runner(),
        Script::new().otherwise(Stream::new().tokens(1..=5).finish(FinishReason::Stop)),
    )
    .unwrap();
    ingress.send(open(1, vec![1]));

    assert!(runner.step().unwrap().sent.is_some());
    assert!(runner.step().unwrap().sent.is_some());
    let third = runner.step().unwrap();
    assert!(third.held && third.sent.is_none());
    assert!(runner.step().unwrap().held, "still full");

    let mut frames = ingress.frames();
    runner.run_until_idle(100).unwrap();
    frames.extend(ingress.frames());
    runner.run_until_idle(100).unwrap();
    frames.extend(ingress.frames());
    assert_eq!(tokens_of(&frames, seq(1)), [1, 2, 3, 4, 5]);
    let steps: Vec<u64> = frames.iter().map(|frame| frame.step).collect();
    assert_eq!(steps, [0, 1, 2, 3, 4]);
}

/// `serve` on a thread of its own delivers the same stream in the same order. Ingress reads until
/// the finish arrives or `serve` returns; no step timing is assumed, and a `serve` that fails
/// fails the test instead of leaving it waiting.
#[test]
fn serve_streams_on_its_own_thread() {
    let region = FakeRegion::new().unwrap();
    let mut ingress = Ingress::new(&region);
    let mut runner = FakeRunner::new(
        region.runner(),
        Script::new().otherwise(Stream::new().tokens(1..=200).finish(FinishReason::Length)),
    )
    .unwrap();
    let stop = AtomicBool::new(false);

    std::thread::scope(|scope| {
        let served = scope.spawn(|| runner.serve(&stop));
        ingress.send(open(1, vec![1]));
        let finished = |frames: &[TokenFrame]| {
            frames
                .last()
                .is_some_and(|frame| frame.entries[0].finish.is_some())
        };
        let mut frames = Vec::new();
        while !finished(&frames) && !served.is_finished() {
            frames.extend(ingress.frames());
            std::thread::yield_now();
        }
        // Anything written before `serve` returned.
        frames.extend(ingress.frames());
        stop.store(true, Ordering::Release);
        served.join().unwrap().unwrap();
        assert!(finished(&frames), "the stream ended before its finish");
        assert_eq!(tokens_of(&frames, seq(1)), (1..=200).collect::<Vec<u32>>());
    });
    assert_eq!(runner.log(), ingress.sent.as_slice());
}
