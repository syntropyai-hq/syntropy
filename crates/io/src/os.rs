// SPDX-License-Identifier: MIT OR Apache-2.0

//! The production clock and entropy sources (ADR-0022; RFC-0060 §2.1; RFC-0001 §5.14).
//!
//! `syntropy-core` defines the [`Clock`] and [`Entropy`] traits that decision-making code reads
//! time and randomness through (rule T1). The production sources live here, because reading the
//! OS clocks takes `unsafe` FFI and the OS CSPRNG takes `getrandom`, which RFC-0001 §4.2 rules 1
//! and 8 keep out of `core`. The simulator's sources, `SimClock` and `SimEntropy`, are in
//! `testkit`.
//!
//! - [`SystemClock::now`] reads a clock that keeps running while the host is suspended:
//!   `CLOCK_BOOTTIME` on Linux and Android, `mach_continuous_time` on macOS and iOS, and
//!   `QueryInterruptTime` on Windows. It never reads a clock that stops during sleep, so leases and
//!   failure timers see a suspend as elapsed time.
//! - [`SystemClock::awake`] reads the clock that stops while the host is suspended:
//!   `CLOCK_MONOTONIC`, `mach_absolute_time` and `QueryUnbiasedInterruptTime`. RFC-0030 §5.8's
//!   shell compares the two to detect a suspend, without an OS call of its own.
//! - [`OsEntropy`] fills buffers from the OS CSPRNG. Key material always comes from it (rule T2).
//! - [`SystemTicks`] is the production [`TickSource`] for hot-thread timing in `exec` and `obs`:
//!   the invariant TSC on x86-64, CNTVCT_EL0 on arm64, or the OS monotonic counter where neither
//!   is usable (RFC-0060 §2.1, note of 2026-10-22).
//! - The `thread` functions place hot threads: [`cpu_topology`], [`pin_current_thread`],
//!   [`raise_current_thread_priority`] and [`current_core`] (ADR-0012 item 2).
//! - [`checks`] holds OS system calls that need `unsafe`, behind safe functions that `probe`,
//!   `cluster` and `daemon` call (RFC-0001 §4.2 rule 8, note of 2026-10-22; OQ-PRC-26): so far
//!   E1's Linux AMX tile-data permission request and `io_uring_setup` check (RFC-0010 §2.2).
//! - `macos_host` and `windows_host` hold the system calls of E1 host enumeration that
//!   `probe` makes through safe functions (RFC-0010 §2.2; OQ-PRC-26): sysctl reads,
//!   `host_statistics64` and IOKit power sources on macOS; processor, memory, NUMA, power,
//!   working-set and version calls on Windows.
//! - [`log_sink`] holds the OS log facilities the daemon's log sinks write to: unified logging
//!   on macOS and the event log on Windows (RFC-0053 §4.2; PROD-E8-T04).
//!
//! Every `unsafe` block carries a `// SAFETY:` comment with its invariant (ADR-0001 item 4).

use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub mod checks;
pub mod log_sink;
#[cfg(target_os = "macos")]
pub mod macos_host;
mod thread;
#[cfg(windows)]
pub mod windows_host;

pub use thread::{
    Core, CoreId, PinError, Priority, cpu_topology, current_core, pin_current_thread,
    raise_current_thread_priority,
};

use syntropy_core::{Clock, DeviceClockSync, Entropy, MonoTime, Nanos, TickRate, TickSource};

/// The production [`Clock`]: monotonic nanoseconds that include time the host spent suspended,
/// and the wall clock from [`SystemTime`].
///
/// Built with [`SystemClock::new`] or [`Default`], never as a bare value: `non_exhaustive`
/// makes every construction outside this crate name the type, which the determinism lints'
/// type ban then reports in every crate rule T1 covers (RFC-0060 §2.1, #1763). Only a binary's
/// wiring builds it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SystemClock;

impl SystemClock {
    /// The production clock.
    #[must_use]
    pub const fn new() -> Self {
        SystemClock
    }

    /// The clock that stops while the host is suspended (RFC-0030 §5.8), on its own timeline.
    ///
    /// Compare only differences between two readings of this clock, and against differences of
    /// [`Clock::now`] taken around them. When `now` advances more than `awake` between two
    /// readings, the host was suspended for the difference.
    #[must_use]
    pub fn awake(&self) -> AwakeTime {
        AwakeTime(platform::awake_nanos())
    }

    /// One tick of the coarser of the two clocks: the slack a comparison of their differences
    /// needs, since each reading is a whole number of its clock's ticks.
    #[must_use]
    pub fn tick(&self) -> Nanos {
        Nanos(platform::tick_nanos())
    }
}

impl Clock for SystemClock {
    fn now(&self) -> MonoTime {
        MonoTime(platform::running_nanos())
    }

    fn unix_ms(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| {
                u64::try_from(since.as_millis()).unwrap_or(u64::MAX)
            })
    }
}

/// A reading of the clock that stops while the host is suspended, in nanoseconds since an
/// arbitrary origin ([`SystemClock::awake`]). It is not on [`MonoTime`]'s timeline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AwakeTime(pub u64);

impl AwakeTime {
    /// The awake time from `earlier` to `self`, or zero if `earlier` is later.
    #[must_use]
    pub fn saturating_duration_since(self, earlier: AwakeTime) -> Nanos {
        Nanos(self.0.saturating_sub(earlier.0))
    }
}

/// The production [`Entropy`]: the OS CSPRNG, through `getrandom` (RFC-0060 §2.1 rule T2;
/// RFC-0034 §3).
///
/// Built with [`OsEntropy::new`] or [`Default`], never as a bare value, for the reason
/// [`SystemClock`] gives.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OsEntropy;

impl OsEntropy {
    /// The OS random source.
    #[must_use]
    pub const fn new() -> Self {
        OsEntropy
    }
}

impl Entropy for OsEntropy {
    /// Fills `buf` from the OS CSPRNG.
    ///
    /// # Panics
    ///
    /// If the OS random source fails. Returning bytes that are not random would be worse, because
    /// key material comes from here.
    fn fill(&self, buf: &mut [u8]) {
        if let Err(error) = getrandom::fill(buf) {
            panic!("the OS random source failed: {error}");
        }
    }
}

/// The counter a [`SystemTicks`] reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TickCounter {
    /// The x86-64 time-stamp counter, used when CPUID reports it invariant: constant rate, and
    /// running in every power state but suspend.
    Tsc,
    /// The arm64 virtual counter, CNTVCT_EL0, at CNTFRQ_EL0.
    Cntvct,
    /// The OS monotonic counter, through `std::time::Instant`, in nanoseconds since the process
    /// first read it. It is used where neither hardware counter is usable.
    OsMonotonic,
}

/// The production [`TickSource`] (RFC-0060 §2.1 rule T1).
///
/// The first use of any `SystemTicks` in a process picks the counter and finds its rate:
/// - the TSC on x86-64, when CPUID reports it invariant and, on Linux and Android, the kernel's
///   current clocksource is `tsc`. A kernel that found the TSC unreliable has switched away
///   from it. Its rate is calibrated over [`TSC_CALIBRATION`];
/// - CNTVCT_EL0 on arm64, at the rate CNTFRQ_EL0 reports, or calibrated the same way when it
///   reports none;
/// - otherwise the OS monotonic counter, which already counts nanoseconds.
///
/// The calibration measures against a clock that nothing adjusts: `CLOCK_MONOTONIC_RAW` on
/// Linux and Android, `mach_absolute_time` on macOS and iOS, and `QueryPerformanceCounter`
/// (through `std::time::Instant`) on Windows.
/// - Not `CLOCK_MONOTONIC`, which `adjtimex` slews by up to 10%. Whatever slew is active during
///   the calibration would stay in the TSC's rate for the life of the process.
/// - Not [`SystemClock`], which on Windows advances only at the timer interrupt, up to 15.625 ms
///   at a time.
///
/// Each end of the window is bracketed between two reads of the reference.
///
/// That first use blocks for up to [`TSC_CALIBRATION`], so a binary calls
/// [`SystemTicks::rate`] at startup, before a hot thread's first step.
///
/// [`TickSource::now`] reads [`SystemClock`] itself, so it is on the `Clock` timeline exactly and
/// counts suspended time. Mapping ticks onto that timeline through an offset that is re-sampled
/// after every resume is QUAL-E3-T16's.
///
/// `SystemTicks` is `#[non_exhaustive]`, so outside this crate [`SystemTicks::new`] (or
/// `Default`) is the only way to build one, and a determinism lint can name that constructor. Every value shares the process's
/// calibration: reading it is one atomic load, with no lock and no allocation after the first
/// use.
///
/// Two readings on different cores can differ by the cores' counter skew, so a caller taking a
/// delta across a possible thread migration uses `saturating_sub`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct SystemTicks;

/// What the process's first use of [`SystemTicks`] found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Calibration {
    counter: TickCounter,
    rate: TickRate,
}

/// How long the first use of [`SystemTicks`] spends calibrating a counter whose rate it cannot
/// read, once per process. 20 ms keeps the rate within about 10 parts per million when each end
/// is bracketed to within 100 ns [derivation: 2 × 100 ns / 20 ms].
pub const TSC_CALIBRATION: Nanos = Nanos(20_000_000);

impl SystemTicks {
    /// A handle on the process's tick source. It does no work. The first use of any handle
    /// calibrates.
    #[must_use]
    pub const fn new() -> SystemTicks {
        SystemTicks
    }

    /// The counter this process reads.
    #[must_use]
    pub fn counter(&self) -> TickCounter {
        calibration().counter
    }

    /// The counter's rate. The first use in a process calibrates, which can take
    /// [`TSC_CALIBRATION`].
    #[must_use]
    pub fn rate(&self) -> TickRate {
        calibration().rate
    }
}

/// The process's calibration, found at first use.
fn calibration() -> &'static Calibration {
    static CALIBRATION: OnceLock<Calibration> = OnceLock::new();
    CALIBRATION.get_or_init(pick)
}

fn pick() -> Calibration {
    if let Some(found) = counter::hardware() {
        return found;
    }
    os_monotonic_base();
    Calibration {
        counter: TickCounter::OsMonotonic,
        rate: TickRate::NANOS,
    }
}

impl TickSource for SystemTicks {
    #[inline]
    fn ticks(&self) -> u64 {
        match calibration().counter {
            TickCounter::Tsc | TickCounter::Cntvct => counter::read(),
            TickCounter::OsMonotonic => os_monotonic_ticks(),
        }
    }

    #[inline]
    fn ticks_to_nanos(&self, ticks: u64) -> Nanos {
        calibration().rate.ticks_to_nanos(ticks)
    }

    fn now(&self) -> MonoTime {
        SystemClock.now()
    }

    fn device_to_ticks(&self, device_ts: u64, sync: &DeviceClockSync) -> u64 {
        sync.to_ticks(device_ts, calibration().rate.hz())
    }
}

/// The origin of [`TickCounter::OsMonotonic`]: the first reading in this process.
fn os_monotonic_base() -> Instant {
    static BASE: OnceLock<Instant> = OnceLock::new();
    *BASE.get_or_init(Instant::now)
}

fn os_monotonic_ticks() -> u64 {
    u64::try_from(os_monotonic_base().elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// How many times [`bracketed`] reads, keeping the tightest bracket.
const BRACKET_READS: usize = 16;

/// One reading of `read` taken between two reads of the calibration reference
/// ([`platform::reference_nanos`]), as `(counter, reference)`. The reference value is the
/// midpoint of the tightest of [`BRACKET_READS`] brackets.
fn bracketed(read: fn() -> u64) -> (u64, u64) {
    let mut best = (0, 0, u64::MAX);
    for _ in 0..BRACKET_READS {
        let before = platform::reference_nanos();
        let counter = read();
        let after = platform::reference_nanos();
        let spread = after.saturating_sub(before);
        if spread < best.2 {
            best = (counter, before + spread / 2, spread);
        }
    }
    (best.0, best.1)
}

/// The rate of `read`, measured against the calibration reference over [`TSC_CALIBRATION`] with
/// each end bracketed, or `None` if the counter did not advance.
fn calibrate(read: fn() -> u64) -> Option<TickRate> {
    let (first, start) = bracketed(read);
    let until = start.saturating_add(TSC_CALIBRATION.0);
    while platform::reference_nanos() < until {
        std::hint::spin_loop();
    }
    let (last, end) = bracketed(read);
    let ticks = last.checked_sub(first)?;
    let elapsed = u128::from(end.saturating_sub(start).max(1));
    let hz = u128::from(ticks) * 1_000_000_000 / elapsed;
    TickRate::from_hz(u64::try_from(hz).ok()?)
}

#[cfg(target_arch = "x86_64")]
mod counter {
    use std::arch::x86_64::{__cpuid, _rdtsc};

    use super::{Calibration, TickCounter, calibrate};

    /// The extended CPUID leaf that holds the invariant-TSC bit, and the bit (EDX bit 8).
    const POWER_LEAF: u32 = 0x8000_0007;
    const INVARIANT_TSC: u32 = 1 << 8;

    /// Whether CPUID reports an invariant TSC.
    pub(super) fn invariant_tsc() -> bool {
        let max_extended = __cpuid(0x8000_0000).eax;
        max_extended >= POWER_LEAF && __cpuid(POWER_LEAF).edx & INVARIANT_TSC != 0
    }

    /// The kernel's current clocksource, where it can be read.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub(super) fn kernel_clocksource() -> Option<String> {
        const CURRENT: &str = "/sys/devices/system/clocksource/clocksource0/current_clocksource";
        std::fs::read_to_string(CURRENT).ok()
    }

    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    pub(super) fn kernel_clocksource() -> Option<String> {
        None
    }

    /// Whether to use the TSC. CPUID must report it invariant. Where the kernel's clocksource
    /// is known, it must also be `tsc`: a kernel that found the TSC unreliable, from cores out
    /// of step or a hypervisor that does not keep it, has switched to another source. Where the
    /// clocksource cannot be read, as in some containers, CPUID's bit is all there is to go on.
    pub(super) fn use_tsc(invariant: bool, clocksource: Option<&str>) -> bool {
        invariant && clocksource.is_none_or(|source| source.trim() == "tsc")
    }

    /// The TSC, if [`use_tsc`] allows it, with its calibrated rate.
    pub(super) fn hardware() -> Option<Calibration> {
        if !use_tsc(invariant_tsc(), kernel_clocksource().as_deref()) {
            return None;
        }
        Some(Calibration {
            counter: TickCounter::Tsc,
            rate: calibrate(read)?,
        })
    }

    #[inline]
    pub(super) fn read() -> u64 {
        // SAFETY: RDTSC needs no CPU feature beyond x86-64's baseline, reads only the
        // time-stamp counter into registers, and touches no memory. `hardware` picks this
        // counter only when `use_tsc` allows it.
        unsafe { _rdtsc() }
    }
}

#[cfg(target_arch = "aarch64")]
mod counter {
    use std::arch::asm;

    use syntropy_core::TickRate;

    use super::{Calibration, TickCounter, calibrate};

    /// CNTVCT_EL0, with its rate from CNTFRQ_EL0, or calibrated if that reads 0.
    pub(super) fn hardware() -> Option<Calibration> {
        let rate = match TickRate::from_hz(frequency()) {
            Some(rate) => rate,
            None => calibrate(read)?,
        };
        Some(Calibration {
            counter: TickCounter::Cntvct,
            rate,
        })
    }

    /// CNTFRQ_EL0: the rate of the system counter that CNTVCT_EL0 reads.
    fn frequency() -> u64 {
        let hz: u64;
        // SAFETY: CNTFRQ_EL0 is readable at EL0 on every OS Syntropy supports (Linux, Android,
        // macOS, iOS and Windows enable user access to the generic timer), the instruction only
        // writes the output register, and it touches no memory or flags.
        unsafe {
            asm!("mrs {hz}, cntfrq_el0", hz = out(reg) hz, options(nomem, nostack, preserves_flags));
        }
        hz
    }

    #[inline]
    pub(super) fn read() -> u64 {
        let ticks: u64;
        // SAFETY: as in `frequency`, CNTVCT_EL0 is readable at EL0 on every supported OS. The
        // `isb` keeps the read from being taken ahead of earlier instructions. Neither
        // instruction touches memory or flags.
        unsafe {
            asm!(
                "isb",
                "mrs {ticks}, cntvct_el0",
                ticks = out(reg) ticks,
                options(nomem, nostack, preserves_flags),
            );
        }
        ticks
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod counter {
    use super::Calibration;

    /// No hardware counter on this architecture; the OS monotonic counter is used.
    pub(super) fn hardware() -> Option<Calibration> {
        None
    }

    pub(super) fn read() -> u64 {
        unreachable!("SystemTicks reads a hardware counter only when one was picked")
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
mod platform {
    use std::mem::MaybeUninit;

    /// Reads `clock` as nanoseconds.
    #[allow(clippy::useless_conversion)] // `time_t` and `c_long` are 32-bit on some targets
    fn read(clock: libc::clockid_t) -> u64 {
        let mut ts = MaybeUninit::<libc::timespec>::uninit();
        // SAFETY: `ts` points to writable memory the size of a `timespec`, valid for the call,
        // and `clock_gettime` writes only through it. `clock` is `CLOCK_BOOTTIME` (Linux 2.6.39
        // and later), `CLOCK_MONOTONIC` or `CLOCK_MONOTONIC_RAW` (Linux 2.6.28 and later), all
        // of which every supported kernel has.
        let rc = unsafe { libc::clock_gettime(clock, ts.as_mut_ptr()) };
        assert_eq!(rc, 0, "clock_gettime({clock}) failed");
        // SAFETY: `clock_gettime` returned 0, so it initialised every field of `ts`.
        let ts = unsafe { ts.assume_init() };
        to_nanos(ts.tv_sec.into(), ts.tv_nsec.into())
    }

    /// The resolution of `clock`, in nanoseconds.
    #[allow(clippy::useless_conversion)] // as in `read`
    fn resolution(clock: libc::clockid_t) -> u64 {
        let mut ts = MaybeUninit::<libc::timespec>::uninit();
        // SAFETY: as in `read`: `ts` is writable for the call, and `clock_getres` writes only
        // through it.
        let rc = unsafe { libc::clock_getres(clock, ts.as_mut_ptr()) };
        assert_eq!(rc, 0, "clock_getres({clock}) failed");
        // SAFETY: `clock_getres` returned 0, so it initialised every field of `ts`.
        let ts = unsafe { ts.assume_init() };
        to_nanos(ts.tv_sec.into(), ts.tv_nsec.into()).max(1)
    }

    fn to_nanos(sec: i64, nsec: i64) -> u64 {
        let sec = u64::try_from(sec).expect("a monotonic clock is never negative");
        let nsec = u64::try_from(nsec).expect("tv_nsec is never negative");
        sec.saturating_mul(1_000_000_000).saturating_add(nsec)
    }

    pub(super) fn running_nanos() -> u64 {
        read(libc::CLOCK_BOOTTIME)
    }

    pub(super) fn awake_nanos() -> u64 {
        read(libc::CLOCK_MONOTONIC)
    }

    /// The calibration reference: `CLOCK_MONOTONIC_RAW`, which `adjtimex` never slews.
    pub(super) fn reference_nanos() -> u64 {
        read(libc::CLOCK_MONOTONIC_RAW)
    }

    pub(super) fn tick_nanos() -> u64 {
        resolution(libc::CLOCK_BOOTTIME).max(resolution(libc::CLOCK_MONOTONIC))
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod platform {
    use std::sync::OnceLock;

    /// `mach_timebase_info_data_t`: ticks × `numer` / `denom` = nanoseconds.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Timebase {
        numer: u32,
        denom: u32,
    }

    // SAFETY: these are the declarations of `<mach/mach_time.h>`, from libSystem, which every
    // macOS and iOS process links. `mach_absolute_time` and `mach_continuous_time` take no
    // arguments, touch no memory of the caller and cannot fail, so they are declared safe.
    unsafe extern "C" {
        safe fn mach_absolute_time() -> u64;
        safe fn mach_continuous_time() -> u64;
        fn mach_timebase_info(info: *mut Timebase) -> i32;
    }

    fn timebase() -> Timebase {
        static TIMEBASE: OnceLock<Timebase> = OnceLock::new();
        *TIMEBASE.get_or_init(|| {
            let mut info = Timebase { numer: 0, denom: 0 };
            // SAFETY: `info` is a valid, writable `mach_timebase_info_data_t` for the call, and
            // `mach_timebase_info` writes only through it.
            let rc = unsafe { mach_timebase_info(&mut info) };
            assert!(
                rc == 0 && info.denom != 0,
                "mach_timebase_info failed: {rc}"
            );
            info
        })
    }

    fn to_nanos(ticks: u64) -> u64 {
        let tb = timebase();
        let nanos = u128::from(ticks) * u128::from(tb.numer) / u128::from(tb.denom);
        u64::try_from(nanos).unwrap_or(u64::MAX)
    }

    pub(super) fn running_nanos() -> u64 {
        to_nanos(mach_continuous_time())
    }

    pub(super) fn awake_nanos() -> u64 {
        to_nanos(mach_absolute_time())
    }

    /// The calibration reference: `mach_absolute_time`, which nothing slews.
    pub(super) fn reference_nanos() -> u64 {
        awake_nanos()
    }

    /// Both clocks count the same ticks: one tick, rounded up to whole nanoseconds.
    pub(super) fn tick_nanos() -> u64 {
        let tb = timebase();
        u64::from(tb.numer.div_ceil(tb.denom)).max(1)
    }
}

#[cfg(windows)]
mod platform {
    use std::sync::OnceLock;
    use std::time::Instant;

    use windows_sys::Win32::System::WindowsProgramming::{
        QueryInterruptTime, QueryUnbiasedInterruptTime,
    };

    /// Both clocks count 100 ns units.
    const UNIT: u64 = 100;

    pub(super) fn running_nanos() -> u64 {
        let mut units = 0u64;
        // SAFETY: `units` is a valid, writable `u64` for the call, and `QueryInterruptTime`
        // (Windows 10 and later) writes only through it.
        unsafe { QueryInterruptTime(&mut units) };
        units.saturating_mul(UNIT)
    }

    pub(super) fn awake_nanos() -> u64 {
        let mut units = 0u64;
        // SAFETY: `units` is a valid, writable `u64` for the call, and
        // `QueryUnbiasedInterruptTime` writes only through it.
        let ok = unsafe { QueryUnbiasedInterruptTime(&mut units) };
        assert!(ok != 0, "QueryUnbiasedInterruptTime failed");
        units.saturating_mul(UNIT)
    }

    /// Both clocks advance at the system timer interrupt, whose interval is at most 15.625 ms in
    /// the default configuration.
    pub(super) fn tick_nanos() -> u64 {
        15_625_000
    }

    /// The calibration reference: `QueryPerformanceCounter`, through `Instant`, in nanoseconds
    /// since the process first read it. Unlike the interrupt-time clocks above, it resolves well
    /// below a microsecond.
    pub(super) fn reference_nanos() -> u64 {
        static BASE: OnceLock<Instant> = OnceLock::new();
        let elapsed = BASE.get_or_init(Instant::now).elapsed();
        u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
    }
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios",
    windows
)))]
compile_error!(
    "syntropy_io::os has no suspend-aware clock for this OS; RFC-0060 §2.1 names Linux, \
     Android, macOS, iOS and Windows"
);

#[cfg(test)]
mod tests {
    use super::*;

    const CALLS: usize = 1_000_000;

    #[test]
    fn now_is_monotonic_across_a_million_calls() {
        let clock = SystemClock::new();
        let mut last = clock.now();
        for _ in 0..CALLS {
            let next = clock.now();
            assert!(next >= last, "now went back from {last:?} to {next:?}");
            last = next;
        }
    }

    #[test]
    fn awake_is_monotonic_across_a_million_calls() {
        let clock = SystemClock::new();
        let mut last = clock.awake();
        for _ in 0..CALLS {
            let next = clock.awake();
            assert!(next >= last, "awake went back from {last:?} to {next:?}");
            last = next;
        }
    }

    /// In the order `now` (n1), awake (s1), awake (s2), `now` (n2), the awake clock never covers
    /// more time than the running one, within one tick of the coarser clock.
    #[test]
    fn the_awake_clock_never_outruns_now() {
        let clock = SystemClock::new();
        let tick = clock.tick().0;
        for _ in 0..CALLS {
            let n1 = clock.now();
            let s1 = clock.awake();
            let s2 = clock.awake();
            let n2 = clock.now();
            let awake = s2.saturating_duration_since(s1).0;
            let running = n2.saturating_duration_since(n1).0;
            assert!(
                awake <= running + tick,
                "awake advanced {awake} ns while now advanced {running} ns (tick {tick} ns)"
            );
        }
    }

    #[test]
    fn unix_ms_is_the_wall_clock() {
        let before = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        let ms = u128::from(SystemClock::new().unix_ms());
        let after = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        assert!(before <= ms && ms <= after, "{before} <= {ms} <= {after}");
    }

    #[test]
    fn awake_time_differences_saturate() {
        assert_eq!(
            AwakeTime(5).saturating_duration_since(AwakeTime(2)),
            Nanos(3)
        );
        assert_eq!(
            AwakeTime(2).saturating_duration_since(AwakeTime(5)),
            Nanos(0)
        );
    }

    /// `now` agrees with the named suspend-aware clock, read here through this test's own call:
    /// a reading taken between two direct readings lies between them.
    #[test]
    fn now_agrees_with_the_os_suspend_aware_clock() {
        for _ in 0..1_000 {
            let before = direct::suspend_aware_nanos();
            let now = SystemClock::new().now().0;
            let after = direct::suspend_aware_nanos();
            assert!(
                before <= now && now <= after,
                "{before} <= {now} <= {after}"
            );
        }
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    mod direct {
        /// `CLOCK_BOOTTIME`, read directly.
        #[allow(clippy::useless_conversion)] // `time_t` is 32-bit on some targets
        pub(super) fn suspend_aware_nanos() -> u64 {
            let mut ts = std::mem::MaybeUninit::<libc::timespec>::uninit();
            // SAFETY: `ts` is writable for the call, and `clock_gettime` writes only through it.
            let rc = unsafe { libc::clock_gettime(libc::CLOCK_BOOTTIME, ts.as_mut_ptr()) };
            assert_eq!(rc, 0);
            // SAFETY: `clock_gettime` returned 0, so `ts` is initialised.
            let ts = unsafe { ts.assume_init() };
            let sec = u64::try_from(i64::from(ts.tv_sec)).unwrap();
            let nsec = u64::try_from(i64::from(ts.tv_nsec)).unwrap();
            sec * 1_000_000_000 + nsec
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    mod direct {
        #[repr(C)]
        struct Timebase {
            numer: u32,
            denom: u32,
        }

        // SAFETY: the declarations of `<mach/mach_time.h>` in libSystem, which every macOS and
        // iOS process links.
        unsafe extern "C" {
            safe fn mach_continuous_time() -> u64;
            fn mach_timebase_info(info: *mut Timebase) -> i32;
        }

        /// `mach_continuous_time`, read directly and converted with the timebase.
        pub(super) fn suspend_aware_nanos() -> u64 {
            let mut tb = Timebase { numer: 0, denom: 0 };
            // SAFETY: `tb` is writable for the call, and `mach_timebase_info` writes only
            // through it.
            assert_eq!(unsafe { mach_timebase_info(&mut tb) }, 0);
            let ticks = mach_continuous_time();
            u64::try_from(u128::from(ticks) * u128::from(tb.numer) / u128::from(tb.denom)).unwrap()
        }
    }

    #[cfg(windows)]
    mod direct {
        use windows_sys::Win32::System::WindowsProgramming::QueryInterruptTime;

        /// `QueryInterruptTime`, read directly.
        pub(super) fn suspend_aware_nanos() -> u64 {
            let mut units = 0u64;
            // SAFETY: `units` is writable for the call, and `QueryInterruptTime` writes only
            // through it.
            unsafe { QueryInterruptTime(&mut units) };
            units * 100
        }
    }

    /// The criterion "the production `now` never goes backwards over 10⁶ reads".
    #[test]
    fn system_ticks_now_never_goes_backwards() {
        let ticks = SystemTicks::new();
        let mut last = ticks.now();
        for _ in 0..CALLS {
            let next = ticks.now();
            assert!(next >= last, "now went back from {last:?} to {next:?}");
            last = next;
        }
    }

    /// `now` is on the `Clock` timeline: between two `SystemClock` readings.
    #[test]
    fn system_ticks_now_is_the_clock_timeline() {
        let ticks = SystemTicks::new();
        for _ in 0..1_000 {
            let before = SystemClock.now();
            let now = ticks.now();
            let after = SystemClock.now();
            assert!(
                before <= now && now <= after,
                "{before:?} <= {now:?} <= {after:?}"
            );
        }
    }

    /// The counter is the one the platform should have, and a duration in ticks agrees with
    /// the same duration on the calibration reference within 0.5%. Both ends are bracketed,
    /// as the calibration's are, so a coarse read does not swamp a 50 ms span. The reference is
    /// never slewed, so a host that is slewing `CLOCK_MONOTONIC` does not fail it.
    #[test]
    fn system_ticks_measure_durations_at_the_clock_rate() {
        const TICKS: SystemTicks = SystemTicks::new();
        let ticks = TICKS;
        assert_eq!(
            SystemTicks::default().rate(),
            ticks.rate(),
            "one calibration per process"
        );
        #[cfg(target_arch = "aarch64")]
        assert_eq!(ticks.counter(), TickCounter::Cntvct);
        #[cfg(target_arch = "x86_64")]
        assert!(matches!(
            ticks.counter(),
            TickCounter::Tsc | TickCounter::OsMonotonic
        ));
        fn read() -> u64 {
            SystemTicks::new().ticks()
        }
        let (first, start) = bracketed(read);
        while platform::reference_nanos() < start + 50_000_000 {
            std::hint::spin_loop();
        }
        let (last, end) = bracketed(read);
        let by_clock = end - start;
        let by_ticks = ticks.ticks_to_nanos(last.saturating_sub(first)).0;
        assert!(
            by_ticks.abs_diff(by_clock) <= by_clock / 200,
            "{:?} at {} Hz: {by_ticks} ns by ticks, {by_clock} ns by the clock",
            ticks.counter(),
            ticks.rate().hz()
        );
    }

    /// The TSC is used only when CPUID reports it invariant and the kernel's clocksource,
    /// where known, is `tsc`; and this process's choice follows that rule.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_tsc_is_used_only_where_the_kernel_uses_it() {
        use super::counter::{invariant_tsc, kernel_clocksource, use_tsc};

        assert!(use_tsc(true, Some("tsc\n")));
        assert!(use_tsc(true, Some("tsc")));
        assert!(use_tsc(true, None));
        assert!(!use_tsc(true, Some("hyperv_clocksource_tsc_page\n")));
        assert!(!use_tsc(true, Some("hpet")));
        assert!(!use_tsc(true, Some("tsc-early")));
        assert!(!use_tsc(false, Some("tsc")));
        assert!(!use_tsc(false, None));

        let chosen = SystemTicks::new().counter();
        if use_tsc(invariant_tsc(), kernel_clocksource().as_deref()) {
            assert!(
                matches!(chosen, TickCounter::Tsc | TickCounter::OsMonotonic),
                "{chosen:?}"
            );
        } else {
            assert_eq!(chosen, TickCounter::OsMonotonic);
        }
    }

    /// On Linux and Android the calibration reference is `CLOCK_MONOTONIC_RAW`: a reading
    /// taken between two direct readings lies between them.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn the_calibration_reference_is_the_raw_clock() {
        #[allow(clippy::useless_conversion)] // `time_t` is 32-bit on some targets
        fn raw() -> u64 {
            let mut ts = std::mem::MaybeUninit::<libc::timespec>::uninit();
            // SAFETY: `ts` is writable for the call, and `clock_gettime` writes only through it.
            let rc = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC_RAW, ts.as_mut_ptr()) };
            assert_eq!(rc, 0);
            // SAFETY: `clock_gettime` returned 0, so `ts` is initialised.
            let ts = unsafe { ts.assume_init() };
            let sec = u64::try_from(i64::from(ts.tv_sec)).unwrap();
            let nsec = u64::try_from(i64::from(ts.tv_nsec)).unwrap();
            sec * 1_000_000_000 + nsec
        }
        for _ in 0..1_000 {
            let before = raw();
            let reference = platform::reference_nanos();
            let after = raw();
            assert!(
                before <= reference && reference <= after,
                "{before} <= {reference} <= {after}"
            );
        }
    }

    #[test]
    fn system_ticks_map_device_timestamps_at_their_rate() {
        let ticks = SystemTicks::new();
        let hz = ticks.rate().hz();
        let sync = DeviceClockSync {
            device_ts: 1_000,
            host_ticks: 5_000_000,
            device_hz: 1_000_000_000,
        };
        // One device nanosecond later is one host nanosecond of ticks later.
        assert_eq!(
            ticks.device_to_ticks(1_000 + 1_000_000_000, &sync),
            5_000_000 + hz
        );
        assert_eq!(ticks.device_to_ticks(1_000, &sync), 5_000_000);
    }

    #[test]
    fn fill_handles_empty_one_byte_and_one_mebibyte_buffers() {
        let entropy = OsEntropy::new();
        entropy.fill(&mut []);
        let mut one = [0u8; 1];
        entropy.fill(&mut one);

        let mut big = vec![0u8; 1 << 20];
        entropy.fill(&mut big);
        // A uniform byte is zero with probability 1/256, so about 4,096 of the 1,048,576 are.
        // Far more zeros means the buffer was not filled.
        let zeros = big.iter().filter(|&&b| b == 0).count();
        assert!(zeros < 8_192, "{zeros} zero bytes in a 1 MiB fill");
    }

    #[test]
    fn two_fills_differ() {
        let entropy = OsEntropy::new();
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        entropy.fill(&mut a);
        entropy.fill(&mut b);
        assert_ne!(a, b);
    }

    #[test]
    fn u64_comes_from_the_os_source() {
        let entropy = OsEntropy::new();
        let draws: Vec<u64> = (0..8).map(|_| entropy.u64()).collect();
        assert!(draws.windows(2).any(|w| w[0] != w[1]), "{draws:?}");
    }
}
