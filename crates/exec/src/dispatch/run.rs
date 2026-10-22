// SPDX-License-Identifier: MIT OR Apache-2.0

//! The production dispatcher thread's loop over [`Dispatcher::poll_once`]
//! ([`Dispatcher::run`]), its waiting parameters ([`LoopConfig`]) and its device-triggered wake
//! ([`WakeWaiter`]) (RFC-0012 §3.1, §3.3; ADR-0012 item 5).
//!
//! A store from the device wakes no parked thread, so inside a step the loop parks only on a wake
//! the device signals. All times come from the dispatcher's [`TickSource`]. Only the length of a
//! park goes to the OS as a duration.

use std::time::Duration;

use syntropy_core::{MonoTime, Nanos, TickSource};
use syntropy_kernels::{Error, HostWaker};
use syntropy_spsc::{Arm, SPIN_DEFAULT, SPIN_ON_BATTERY};

use super::{Dispatcher, PollOutcome, StepDevice, StepScheduler, WakeSource};

/// The loop's waiting parameters (RFC-0012 §3.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoopConfig {
    /// How long the loop spins between steps before it parks: `sched.spin_us`.
    pub spin: Duration,
    /// The shortest predicted gap to a step's next doorbell for which the loop parks on a
    /// device-triggered wake rather than spinning: `exec.inflight_park_threshold`.
    pub inflight_park_threshold: Duration,
    /// How long before the predicted doorbell a parked loop resumes spinning if the wake has
    /// not fired: `exec.inflight_wake_margin`.
    pub inflight_wake_margin: Duration,
}

impl LoopConfig {
    /// The defaults on mains power: a 50 µs spin (ADR-0012 item 5), parking inside a step for
    /// gaps over 2 ms, and a 200 µs margin \[target\]. RFC-0012 §3.3 holds the two in-step figures
    /// as placeholders until the wake latency is measured.
    pub const MAINS: LoopConfig = LoopConfig {
        spin: SPIN_DEFAULT,
        inflight_park_threshold: Duration::from_millis(2),
        inflight_wake_margin: Duration::from_micros(200),
    };

    /// The defaults on battery: a 10 µs spin, parking inside a step for gaps over 0.5 ms, and the
    /// same margin \[target\].
    pub const BATTERY: LoopConfig = LoopConfig {
        spin: SPIN_ON_BATTERY,
        inflight_park_threshold: Duration::from_micros(500),
        inflight_wake_margin: Duration::from_micros(200),
    };
}

/// How the loop parks inside a step on a device-triggered wake (RFC-0012 §3.3).
///
/// Each backend's waiter, such as an eventfd, a futex word, a Windows event or a Vulkan timeline
/// wait, is PLC-E7-T18's. Until then [`NoDeviceWake`] arms nothing, and the loop spins.
pub trait WakeWaiter {
    /// Announces that the calling thread is about to park for `waker`, so the device's signal
    /// will wake it. Returns `false` if this waiter cannot wait on that kind of waker, and the
    /// loop then spins.
    ///
    /// A signal that arrives after `arm` returns must make the next [`WakeWaiter::park`] return
    /// at once. The loop polls once between `arm` and `park`, so a doorbell that fired before the
    /// arm is seen by that poll, and one that fires after it wakes the park.
    fn arm(&mut self, waker: &HostWaker) -> bool;

    /// Parks until `waker` is signalled or `timeout` passes. It may return early.
    fn park(&mut self, waker: &HostWaker, timeout: Duration);

    /// Ends the wait that [`WakeWaiter::arm`] started, whether or not the thread parked.
    fn disarm(&mut self, waker: &HostWaker);
}

/// A waiter that arms nothing, so the loop always spins inside a step.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoDeviceWake;

impl WakeWaiter for NoDeviceWake {
    fn arm(&mut self, _waker: &HostWaker) -> bool {
        false
    }

    fn park(&mut self, _waker: &HostWaker, _timeout: Duration) {}

    fn disarm(&mut self, _waker: &HostWaker) {}
}

/// The most `pause` hints between two passes while spinning.
const MAX_PAUSES: u32 = 64;

/// `pause` back-off: 1, 2, 4 … up to [`MAX_PAUSES`] hints between passes, reset by progress.
struct Backoff {
    pauses: u32,
}

impl Backoff {
    fn reset(&mut self) {
        self.pauses = 1;
    }

    fn spin(&mut self) {
        for _ in 0..self.pauses {
            std::hint::spin_loop();
        }
        self.pauses = (self.pauses * 2).min(MAX_PAUSES);
    }
}

impl<S, D, T> Dispatcher<S, D, T>
where
    S: StepScheduler,
    D: StepDevice<S::Step>,
    T: TickSource,
{
    /// The production dispatcher thread's loop over [`Dispatcher::poll_once`] (RFC-0012 §3.1,
    /// §3.3; ADR-0012 item 5). It handles the four outcomes this way:
    ///
    /// - **`Progress`:** poll again at once.
    /// - **`Waiting`, inside a step:** spin with `pause` back-off and poll again. It parks only
    ///   when two things hold. The predicted gap to the next doorbell must exceed
    ///   [`LoopConfig::inflight_park_threshold`], and the doorbell must carry a
    ///   [`WakeSource::Device`] wake that `waiter` can arm. After arming it polls once more, so a
    ///   doorbell that fired before the arm is not slept through. It then parks until the wake
    ///   fires or [`LoopConfig::inflight_wake_margin`] before the predicted doorbell.
    /// - **`Idle`, between steps:** spin for [`LoopConfig::spin`], then park on every inbound ring
    ///   ([`StepScheduler::arm_inbound`]) until a push, a close, or the pass's `next_wake`.
    /// - **`Fault`:** return it.
    ///
    /// It returns `Ok` once the scheduler is [`closed`](StepScheduler::closed) and nothing is in
    /// flight.
    ///
    /// # Errors
    ///
    /// The first fault a pass returns.
    pub fn run(&mut self, config: &LoopConfig, waiter: &mut impl WakeWaiter) -> Result<(), Error> {
        let spin = Nanos(u64::try_from(config.spin.as_nanos()).unwrap_or(u64::MAX));
        let mut backoff = Backoff { pauses: 1 };
        // The tick at which the current stretch of `Idle` passes began.
        let mut idle_since: Option<u64> = None;
        loop {
            let now = self.ticks.now();
            match self.poll_once(now) {
                PollOutcome::Progress => {
                    idle_since = None;
                    backoff.reset();
                }
                PollOutcome::Waiting { expect, wake } => {
                    idle_since = None;
                    let gap = gap(now, expect);
                    let parked = match wake {
                        WakeSource::Device(waker) if gap > config.inflight_park_threshold => {
                            self.park_in_step(&waker, gap, config, waiter)?
                        }
                        _ => false,
                    };
                    if !parked {
                        backoff.spin();
                    }
                }
                PollOutcome::Idle { next_wake } => {
                    if self.scheduler.closed() {
                        return Ok(());
                    }
                    let since = *idle_since.get_or_insert_with(|| self.ticks.ticks());
                    let idle = self
                        .ticks
                        .ticks_to_nanos(self.ticks.ticks().wrapping_sub(since));
                    if idle < spin {
                        backoff.spin();
                        continue;
                    }
                    self.park_between_steps(next_wake);
                    idle_since = None;
                    backoff.reset();
                }
                PollOutcome::Fault(error) => return Err(error),
            }
        }
    }

    /// Arms `waker`, polls once more so a doorbell that fired before the arm is not slept
    /// through, then parks until the wake fires or the margin before the predicted doorbell.
    /// Returns whether the waiter armed.
    fn park_in_step(
        &mut self,
        waker: &HostWaker,
        gap: Duration,
        config: &LoopConfig,
        waiter: &mut impl WakeWaiter,
    ) -> Result<bool, Error> {
        if !waiter.arm(waker) {
            return Ok(false);
        }
        let outcome = self.poll_once(self.ticks.now());
        if let PollOutcome::Waiting { .. } = outcome {
            waiter.park(waker, gap.saturating_sub(config.inflight_wake_margin));
        }
        waiter.disarm(waker);
        match outcome {
            PollOutcome::Fault(error) => Err(error),
            _ => Ok(true),
        }
    }

    /// Parks on the inbound rings until a push, a close, or `next_wake`. A ring that is not
    /// empty when armed means there is no park.
    fn park_between_steps(&mut self, next_wake: Option<MonoTime>) {
        if self.scheduler.arm_inbound() == Arm::Waiting {
            match next_wake {
                Some(at) => {
                    let wait = gap(self.ticks.now(), at);
                    if !wait.is_zero() {
                        std::thread::park_timeout(wait);
                    }
                }
                None => std::thread::park(),
            }
        }
        self.scheduler.disarm_inbound();
    }
}

/// The time from `now` to `at`, or zero if `at` has passed.
fn gap(now: MonoTime, at: MonoTime) -> Duration {
    Duration::from_nanos(at.saturating_duration_since(now).0)
}
