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
//!
//! Every `unsafe` block carries a `// SAFETY:` comment with its invariant (ADR-0001 item 4).

use std::time::{SystemTime, UNIX_EPOCH};

use syntropy_core::{Clock, Entropy, MonoTime, Nanos};

/// The production [`Clock`]: monotonic nanoseconds that include time the host spent suspended,
/// and the wall clock from [`SystemTime`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SystemClock;

impl SystemClock {
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OsEntropy;

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

#[cfg(any(target_os = "linux", target_os = "android"))]
mod platform {
    use std::mem::MaybeUninit;

    /// Reads `clock` as nanoseconds.
    #[allow(clippy::useless_conversion)] // `time_t` and `c_long` are 32-bit on some targets
    fn read(clock: libc::clockid_t) -> u64 {
        let mut ts = MaybeUninit::<libc::timespec>::uninit();
        // SAFETY: `ts` points to writable memory the size of a `timespec`, valid for the call,
        // and `clock_gettime` writes only through it. `clock` is `CLOCK_BOOTTIME` (Linux 2.6.39
        // and later) or `CLOCK_MONOTONIC`, both of which every supported kernel has.
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

    /// Both clocks count the same ticks: one tick, rounded up to whole nanoseconds.
    pub(super) fn tick_nanos() -> u64 {
        let tb = timebase();
        u64::from(tb.numer.div_ceil(tb.denom)).max(1)
    }
}

#[cfg(windows)]
mod platform {
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
        let clock = SystemClock;
        let mut last = clock.now();
        for _ in 0..CALLS {
            let next = clock.now();
            assert!(next >= last, "now went back from {last:?} to {next:?}");
            last = next;
        }
    }

    #[test]
    fn awake_is_monotonic_across_a_million_calls() {
        let clock = SystemClock;
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
        let clock = SystemClock;
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
        let ms = u128::from(SystemClock.unix_ms());
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
            let now = SystemClock.now().0;
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

    #[test]
    fn fill_handles_empty_one_byte_and_one_mebibyte_buffers() {
        let entropy = OsEntropy;
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
        let entropy = OsEntropy;
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        entropy.fill(&mut a);
        entropy.fill(&mut b);
        assert_ne!(a, b);
    }

    #[test]
    fn u64_comes_from_the_os_source() {
        let entropy = OsEntropy;
        let draws: Vec<u64> = (0..8).map(|_| entropy.u64()).collect();
        assert!(draws.windows(2).any(|w| w[0] != w[1]), "{draws:?}");
    }
}
