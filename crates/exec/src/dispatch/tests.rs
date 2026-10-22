// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

use syntropy_core::{DeviceClockSync, ErrorClass, MonoTime, Nanos, TickRate, TickSource};
use syntropy_kernels::{Error, ErrorCode, HostWaker, WakeKind};
use syntropy_spsc::{Arm, Consumer, PopError, Producer};

use super::*;

/// A clock every clone reads, which the test moves by hand. With a `step`, each read of `ticks`
/// also moves it on by `step` nanoseconds, so a loop that spins on it sees time pass without a
/// wall clock.
#[derive(Clone, Default)]
struct TestTicks {
    now: Arc<AtomicU64>,
    step: u64,
}

impl TestTicks {
    fn stepping(step: Nanos) -> TestTicks {
        TestTicks {
            now: Arc::default(),
            step: step.0,
        }
    }

    fn advance(&self, by: Nanos) {
        self.now.fetch_add(by.0, Ordering::SeqCst);
    }
}

impl TickSource for TestTicks {
    fn ticks(&self) -> u64 {
        self.now.fetch_add(self.step, Ordering::SeqCst) + self.step
    }

    fn ticks_to_nanos(&self, ticks: u64) -> Nanos {
        TickRate::NANOS.ticks_to_nanos(ticks)
    }

    fn now(&self) -> MonoTime {
        MonoTime(self.now.load(Ordering::SeqCst))
    }

    fn device_to_ticks(&self, device_ts: u64, sync: &DeviceClockSync) -> u64 {
        sync.to_ticks(device_ts, TickRate::NANOS.hz())
    }
}

/// A scheduler over one inbound ring of step ids and one outbound ring of "tokens": each step
/// `n` sends `n * 10`.
struct FakeScheduler {
    inbound: Consumer<u32>,
    outbound: Producer<u32>,
    ready: VecDeque<u32>,
    inbound_closed: bool,
    finished: Vec<u32>,
    sent: Vec<u32>,
    next_wake: Option<MonoTime>,
    /// How many times the loop armed every inbound ring and could park.
    parks: Arc<AtomicU32>,
}

impl FakeScheduler {
    fn new(inbound: Consumer<u32>, outbound: Producer<u32>) -> FakeScheduler {
        FakeScheduler {
            inbound,
            outbound,
            ready: VecDeque::new(),
            inbound_closed: false,
            finished: Vec::new(),
            sent: Vec::new(),
            next_wake: None,
            parks: Arc::default(),
        }
    }
}

impl StepScheduler for FakeScheduler {
    type Step = u32;

    fn ingest(&mut self, _now: MonoTime) -> bool {
        let mut arrived = false;
        loop {
            match self.inbound.pop() {
                Ok(step) => {
                    self.ready.push_back(step);
                    arrived = true;
                }
                Err(PopError::Empty) => return arrived,
                Err(PopError::Closed) => {
                    self.inbound_closed = true;
                    return arrived;
                }
            }
        }
    }

    fn next_step(&mut self, _now: MonoTime) -> Option<u32> {
        self.ready.pop_front()
    }

    fn finish(&mut self, step: &mut u32, _now: MonoTime) {
        self.finished.push(*step);
    }

    fn try_send(&mut self, step: &mut u32, _now: MonoTime) -> bool {
        self.outbound.push(*step * 10).is_ok()
    }

    fn sent(&mut self, step: u32, _now: MonoTime) {
        self.sent.push(step);
    }

    fn next_wake(&self, _now: MonoTime) -> Option<MonoTime> {
        self.next_wake
    }

    fn arm_inbound(&mut self) -> Arm {
        let armed = self.inbound.arm_thread();
        if armed == Arm::Waiting {
            self.parks.fetch_add(1, Ordering::SeqCst);
        }
        armed
    }

    fn disarm_inbound(&mut self) {
        self.inbound.disarm();
    }

    fn closed(&self) -> bool {
        self.inbound_closed && self.ready.is_empty()
    }
}

/// A device whose steps raise `doorbells` doorbells and then their done flag. `written` counts
/// what the device has written for the current step: 1 to `doorbells` are the doorbells, and
/// `doorbells + 1` is `O[p]`. A test, or a waiter standing in for the device, moves it.
struct FakeDevice {
    doorbells: u32,
    written: Arc<AtomicU32>,
    serviced: u32,
    /// How far after launch the next doorbell is predicted, and its wake.
    gap: Nanos,
    wake: WakeSource,
    expect: MonoTime,
    /// Steps complete at launch, with no doorbell.
    instant: bool,
    services: u32,
    fail: Option<Error>,
    /// Moves `ticks` on by this much inside each service: a call that takes that long.
    slow: Option<(TestTicks, Nanos)>,
}

impl FakeDevice {
    fn new(doorbells: u32) -> FakeDevice {
        FakeDevice {
            doorbells,
            written: Arc::default(),
            serviced: 0,
            gap: Nanos(1_000),
            wake: WakeSource::Spin,
            expect: MonoTime(0),
            instant: false,
            services: 0,
            fail: None,
            slow: None,
        }
    }
}

impl StepDevice<u32> for FakeDevice {
    fn launch(&mut self, _step: &mut u32, now: MonoTime) -> Result<(), Error> {
        self.serviced = 0;
        let written = if self.instant { self.doorbells + 1 } else { 0 };
        self.written.store(written, Ordering::SeqCst);
        self.expect = now.saturating_add(self.gap);
        Ok(())
    }

    fn service(&mut self, _step: &mut u32, _now: MonoTime) -> Result<Serviced, Error> {
        self.services += 1;
        if let Some((ticks, by)) = &self.slow {
            ticks.advance(*by);
        }
        if let Some(error) = self.fail.take() {
            return Err(error);
        }
        let written = self.written.load(Ordering::SeqCst).min(self.doorbells + 1);
        let fired = written - self.serviced;
        self.serviced = written;
        let progress = if written == self.doorbells + 1 {
            StepProgress::Done
        } else {
            StepProgress::Pending {
                expect: self.expect,
                wake: self.wake,
            }
        };
        Ok(Serviced { fired, progress })
    }
}

fn device_error() -> Error {
    Error {
        class: ErrorClass::Device,
        code: ErrorCode(9),
        message: "sticky device error".to_owned(),
    }
}

type TestDispatcher = Dispatcher<FakeScheduler, FakeDevice, TestTicks>;

/// A dispatcher with rings of `capacity`, and the test's ends of them.
fn dispatcher(
    device: FakeDevice,
    ticks: TestTicks,
    capacity: usize,
) -> (TestDispatcher, Producer<u32>, Consumer<u32>) {
    let (to_stage, inbound) = syntropy_spsc::ring(capacity).unwrap();
    let (outbound, from_stage) = syntropy_spsc::ring(capacity).unwrap();
    let scheduler = FakeScheduler::new(inbound, outbound);
    (
        Dispatcher::new(scheduler, device, ticks),
        to_stage,
        from_stage,
    )
}

/// One pass, checking the criterion "`Idle` only with no step in flight" on every pass.
fn poll(dispatcher: &mut TestDispatcher, now: MonoTime) -> PollOutcome {
    let outcome = dispatcher.poll_once(now);
    if let PollOutcome::Idle { .. } = outcome {
        assert!(!dispatcher.in_flight(), "Idle with a step in flight");
    }
    outcome
}

/// Criterion 2: a pass returns `Idle` only with no step in flight. While a step waits for a
/// doorbell it is `Waiting`, with the doorbell's prediction and wake.
#[test]
fn idle_only_with_no_step_in_flight() {
    let mut device = FakeDevice::new(1);
    device.gap = Nanos(3_000);
    let written = Arc::clone(&device.written);
    let (mut d, mut to_stage, mut from_stage) = dispatcher(device, TestTicks::default(), 4);
    d.scheduler_mut().next_wake = Some(MonoTime(900));
    let t = MonoTime(100);

    assert_eq!(
        poll(&mut d, t),
        PollOutcome::Idle {
            next_wake: Some(MonoTime(900))
        }
    );
    to_stage.push(7).unwrap();
    assert_eq!(poll(&mut d, t), PollOutcome::Progress, "ingest and launch");
    assert!(d.in_flight());
    let waiting = PollOutcome::Waiting {
        expect: MonoTime(3_100),
        wake: WakeSource::Spin,
    };
    assert_eq!(poll(&mut d, t), waiting);
    assert_eq!(poll(&mut d, t), waiting, "nothing fired: still waiting");
    written.store(1, Ordering::SeqCst);
    assert_eq!(poll(&mut d, t), PollOutcome::Progress, "the doorbell");
    assert_eq!(poll(&mut d, t), waiting);
    written.store(2, Ordering::SeqCst);
    assert_eq!(
        poll(&mut d, t),
        PollOutcome::Progress,
        "done, finished, sent"
    );
    assert!(!d.in_flight());
    assert_eq!(from_stage.pop(), Ok(70));
    assert_eq!(
        (
            d.scheduler().finished.as_slice(),
            d.scheduler().sent.as_slice()
        ),
        (&[7][..], &[7][..])
    );
    assert!(matches!(poll(&mut d, t), PollOutcome::Idle { .. }));
}

/// A step whose frames meet a full outbound ring stays in flight: `Waiting`, never `Idle`, and
/// it leaves once the ring has room.
#[test]
fn a_full_outbound_ring_keeps_the_step_in_flight() {
    let mut device = FakeDevice::new(0);
    device.instant = true;
    let (mut d, mut to_stage, mut from_stage) = dispatcher(device, TestTicks::default(), 1);
    // The outbound ring holds one value: fill it.
    to_stage.push(1).unwrap();
    while poll(&mut d, MonoTime(0)) == PollOutcome::Progress {}
    assert_eq!(d.scheduler().sent, [1]);
    to_stage.push(2).unwrap();
    assert_eq!(poll(&mut d, MonoTime(5)), PollOutcome::Progress, "launch");
    assert_eq!(poll(&mut d, MonoTime(5)), PollOutcome::Progress, "done");
    for _ in 0..3 {
        assert_eq!(
            poll(&mut d, MonoTime(5)),
            PollOutcome::Waiting {
                expect: MonoTime(5),
                wake: WakeSource::Spin
            },
            "held by the full ring"
        );
    }
    assert_eq!(from_stage.pop(), Ok(10));
    assert_eq!(poll(&mut d, MonoTime(6)), PollOutcome::Progress, "sent");
    assert_eq!(d.scheduler().sent, [1, 2]);
    assert_eq!(from_stage.pop(), Ok(20));
}

/// A device error faults the dispatcher, and every later pass returns the same fault without
/// touching the device again.
#[test]
fn a_device_error_is_a_fault_that_stays() {
    let mut device = FakeDevice::new(1);
    device.fail = Some(device_error());
    let (mut d, mut to_stage, _from_stage) = dispatcher(device, TestTicks::default(), 4);
    to_stage.push(3).unwrap();
    assert_eq!(poll(&mut d, MonoTime(0)), PollOutcome::Progress);
    assert_eq!(
        poll(&mut d, MonoTime(0)),
        PollOutcome::Fault(device_error())
    );
    assert_eq!(
        poll(&mut d, MonoTime(0)),
        PollOutcome::Fault(device_error())
    );
    assert_eq!(d.device().services, 1);
}

/// Criterion 1's watchdog: a pass is timed on the dispatcher's `TickSource`, and one over the
/// budget counts as an overrun. Here the device takes 80 µs inside a service, as a spin would.
#[test]
fn the_watchdog_counts_a_pass_over_its_budget() {
    let ticks = TestTicks::default();
    let mut device = FakeDevice::new(1);
    device.slow = Some((ticks.clone(), Nanos(80_000)));
    let (mut d, mut to_stage, _from_stage) = dispatcher(device, ticks.clone(), 4);
    assert_eq!(d.budget(), Nanos(50_000), "POLL_BUDGET");

    assert!(matches!(
        poll(&mut d, ticks.now()),
        PollOutcome::Idle { .. }
    ));
    to_stage.push(1).unwrap();
    assert_eq!(poll(&mut d, ticks.now()), PollOutcome::Progress, "launch");
    assert_eq!(
        d.stats(),
        PollStats {
            polls: 2,
            overruns: 0,
            longest: Nanos(0)
        }
    );
    assert!(matches!(
        poll(&mut d, ticks.now()),
        PollOutcome::Waiting { .. }
    ));
    assert_eq!(
        d.stats(),
        PollStats {
            polls: 3,
            overruns: 1,
            longest: Nanos(80_000)
        }
    );

    // The same pass under a larger budget is no overrun.
    let mut device = FakeDevice::new(1);
    device.slow = Some((ticks.clone(), Nanos(80_000)));
    let (d, mut to_stage, _from_stage) = dispatcher(device, ticks.clone(), 4);
    let mut d = d.with_budget(Duration::from_micros(100));
    to_stage.push(1).unwrap();
    poll(&mut d, ticks.now());
    poll(&mut d, ticks.now());
    assert_eq!(d.stats().overruns, 0);
    assert_eq!(d.stats().longest, Nanos(80_000));
}

/// Moves `ticks` on by 1 µs at every read, so the loop's 50 µs spin ends after about 50 reads.
fn stepping() -> TestTicks {
    TestTicks::stepping(Nanos(1_000))
}

/// Waits up to 10 s for `done`, checking every millisecond.
fn eventually(what: &str, done: impl Fn() -> bool) {
    for _ in 0..10_000 {
        if done() {
            return;
        }
        thread::sleep(Duration::from_millis(1));
    }
    panic!("{what} did not happen within 10 s");
}

/// The production loop between steps: it spins, parks on the inbound ring, wakes for a frame,
/// runs the step, parks again, and returns once the ring closes. It runs on a dispatcher thread
/// from `spawn_dispatcher`.
#[test]
fn the_loop_parks_between_steps_and_a_frame_wakes_it() {
    let (mut to_stage, from_net) = syntropy_spsc::ring(4).unwrap();
    let (to_net, mut from_stage) = syntropy_spsc::ring(4).unwrap();
    let rings = crate::DispatcherRings {
        from_net,
        to_net,
        io: None,
    };
    let parks = Arc::new(AtomicU32::new(0));
    let spec = crate::DispatcherSpec {
        kind: crate::DispatcherKind::Device,
        device: syntropy_core::DeviceId {
            node: syntropy_core::NodeId([1; 32]),
            ordinal: 0,
        },
        numa: None,
        core: None,
        spin: LoopConfig::MAINS.spin,
    };
    let counted = Arc::clone(&parks);
    let thread = crate::spawn_dispatcher(spec, rings, move |ctx| {
        let mut scheduler = FakeScheduler::new(ctx.rings.from_net, ctx.rings.to_net);
        scheduler.parks = counted;
        let mut device = FakeDevice::new(0);
        device.instant = true;
        let mut d = Dispatcher::new(scheduler, device, stepping());
        let result = d.run(&LoopConfig::MAINS, &mut NoDeviceWake);
        (result, d.scheduler().sent.clone())
    })
    .unwrap();

    eventually("the first park", || parks.load(Ordering::SeqCst) >= 1);
    to_stage.push(4).unwrap();
    assert_eq!(from_stage.pop_wait(Duration::ZERO), Some(40));
    eventually("a park after the step", || {
        parks.load(Ordering::SeqCst) >= 2
    });
    drop(to_stage);
    let (result, sent) = thread.join().unwrap();
    assert_eq!(result, Ok(()));
    assert_eq!(sent, [4]);
}

/// A waiter standing in for a backend's device-triggered wake: parking "fires" the device's
/// next write, as the doorbell's wake would.
#[derive(Default)]
struct FakeWaiter {
    written: Arc<AtomicU32>,
    arms: u32,
    parks: Vec<Duration>,
    disarms: u32,
}

impl WakeWaiter for FakeWaiter {
    fn arm(&mut self, waker: &HostWaker) -> bool {
        assert_eq!(waker.handle, 42);
        self.arms += 1;
        true
    }

    fn park(&mut self, _waker: &HostWaker, timeout: Duration) {
        self.parks.push(timeout);
        self.written.fetch_add(1, Ordering::SeqCst);
    }

    fn disarm(&mut self, _waker: &HostWaker) {
        self.disarms += 1;
    }
}

/// Runs one 2-doorbell step through the loop with `wake` and a predicted gap of `gap`. Returns
/// the waiter. With `device_writes`, this thread writes the device's flags, one per millisecond,
/// for a loop that spins. Without it, only the waiter's parks move the device, so a loop that
/// failed to park would never finish.
fn one_step_in_the_loop(wake: WakeSource, gap: Nanos, device_writes: bool) -> FakeWaiter {
    let (mut to_stage, inbound) = syntropy_spsc::ring(4).unwrap();
    let (outbound, mut from_stage) = syntropy_spsc::ring(4).unwrap();
    let mut device = FakeDevice::new(2);
    device.gap = gap;
    device.wake = wake;
    let written = Arc::clone(&device.written);
    let mut waiter = FakeWaiter {
        written: Arc::clone(&written),
        ..FakeWaiter::default()
    };
    let looping = thread::spawn(move || {
        let scheduler = FakeScheduler::new(inbound, outbound);
        let mut d = Dispatcher::new(scheduler, device, stepping());
        let result = d.run(&LoopConfig::MAINS, &mut waiter);
        (result, waiter)
    });
    to_stage.push(9).unwrap();
    for _ in 0..10_000 {
        if let Ok(token) = from_stage.pop() {
            assert_eq!(token, 90);
            break;
        }
        thread::sleep(Duration::from_millis(1));
        if device_writes && written.load(Ordering::SeqCst) < 3 {
            written.fetch_add(1, Ordering::SeqCst);
        }
    }
    drop(to_stage);
    let (result, waiter) = looping.join().unwrap();
    assert_eq!(result, Ok(()));
    waiter
}

fn waker() -> HostWaker {
    HostWaker {
        kind: WakeKind::Futex,
        handle: 42,
    }
}

/// Inside a step the loop parks only on an armed device-triggered wake whose predicted gap
/// exceeds `exec.inflight_park_threshold` (2 ms on mains). It parks until the margin (200 µs)
/// before the predicted doorbell, and disarms after each park.
#[test]
fn the_loop_parks_inside_a_step_only_on_an_armed_device_wake() {
    // Only the parks move the device here: one for each doorbell and one for the done flag.
    let waiter = one_step_in_the_loop(WakeSource::Device(waker()), Nanos(10_000_000), false);
    assert_eq!((waiter.arms, waiter.parks.len(), waiter.disarms), (3, 3, 3));
    for timeout in &waiter.parks {
        // 10 ms less the 200 µs margin, less the few µs the stepping clock moved.
        assert!(
            *timeout <= Duration::from_micros(9_800) && *timeout > Duration::from_micros(9_000),
            "{timeout:?}"
        );
    }

    // A gap under the threshold: the loop spins, and the waiter is never armed.
    let waiter = one_step_in_the_loop(WakeSource::Device(waker()), Nanos(1_000_000), true);
    assert_eq!((waiter.arms, waiter.parks.len()), (0, 0));

    // No device-triggered wake: the loop spins whatever the gap.
    let waiter = one_step_in_the_loop(WakeSource::Spin, Nanos(10_000_000), true);
    assert_eq!((waiter.arms, waiter.parks.len()), (0, 0));
}

/// A fault ends the loop with that fault.
#[test]
fn the_loop_returns_the_first_fault() {
    let (mut to_stage, inbound) = syntropy_spsc::ring(4).unwrap();
    let (outbound, _from_stage) = syntropy_spsc::ring(4).unwrap();
    let mut device = FakeDevice::new(1);
    device.fail = Some(device_error());
    to_stage.push(1).unwrap();
    let mut d = Dispatcher::new(FakeScheduler::new(inbound, outbound), device, stepping());
    assert_eq!(
        d.run(&LoopConfig::MAINS, &mut NoDeviceWake),
        Err(device_error())
    );
}
