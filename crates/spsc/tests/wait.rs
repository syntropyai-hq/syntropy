// SPDX-License-Identifier: MIT OR Apache-2.0

//! Waiting for a ring (ADR-0012 items 4 and 5): a parked consumer wakes on a push and on the
//! producer's close, and a ring's wake handle is signalled by a push only while the consumer is
//! armed, and stays signalled until reset.

#![cfg(not(loom))]

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use syntropy_spsc::{Arm, Consumer, PopError, ring};

/// Long enough that a missed wake-up shows as a failure, not a slow pass.
const PATIENCE: Duration = Duration::from_secs(10);

/// Lets the consumer thread get past its spin and park.
fn let_it_park() {
    thread::sleep(Duration::from_millis(50));
}

#[test]
fn a_parked_consumer_wakes_when_the_producer_pushes() {
    let (mut tx, mut rx) = ring::<u32>(4).unwrap();
    let (done, finished) = mpsc::channel();
    let consumer = thread::spawn(move || {
        let value = rx.pop_wait(Duration::ZERO);
        done.send(value).unwrap();
        rx
    });
    let_it_park();
    tx.push(5).unwrap();
    assert_eq!(finished.recv_timeout(PATIENCE), Ok(Some(5)));
    consumer.join().unwrap();
}

#[test]
fn a_parked_consumer_wakes_when_the_producer_closes() {
    let (tx, mut rx) = ring::<u32>(4).unwrap();
    let (done, finished) = mpsc::channel();
    let consumer = thread::spawn(move || {
        done.send(rx.pop_wait(Duration::ZERO)).unwrap();
    });
    let_it_park();
    drop(tx);
    assert_eq!(finished.recv_timeout(PATIENCE), Ok(None));
    consumer.join().unwrap();
}

#[test]
fn wait_returns_at_once_when_a_value_is_ready() {
    let (mut tx, mut rx) = ring::<u32>(4).unwrap();
    tx.push(1).unwrap();
    rx.wait(Duration::from_secs(3600));
    assert_eq!(rx.pop(), Ok(1));
}

/// The producer unparks the thread that first waited, so waiting from another thread would sleep
/// forever. It panics instead.
#[test]
fn a_consumer_waits_on_one_thread_only() {
    let (mut tx, mut rx) = ring::<u32>(4).unwrap();
    let first = thread::spawn(move || {
        rx.wait(Duration::ZERO);
        rx
    });
    let_it_park();
    tx.push(1).unwrap();
    let mut rx = first.join().unwrap();
    assert_eq!(rx.pop(), Ok(1));
    let second = thread::spawn(move || rx.wait(Duration::ZERO));
    let panic = second.join().unwrap_err();
    let message = panic.downcast_ref::<String>().unwrap();
    assert!(
        message.contains("a ring's consumer waits on the thread that first waited on it"),
        "{message}"
    );
    drop(tx);
}

/// One wait of `Consumer::arm_thread`'s loop, the way a dispatcher waits between steps: parks
/// the calling thread unless `a` or `b` has a value or has lost its producer. It may return with
/// neither ready, as `park` may, so the caller pops and calls again.
fn park_on_both(a: &mut Consumer<u32>, b: &mut Consumer<u32>) {
    if (a.arm_thread(), b.arm_thread()) == (Arm::Waiting, Arm::Waiting) {
        thread::park();
    }
    a.disarm();
    b.disarm();
}

/// A thread parked on two rings wakes for a push to either one, and for a close.
#[test]
fn a_thread_parked_on_two_rings_wakes_for_either() {
    let (mut tx_a, rx_a) = ring::<u32>(4).unwrap();
    let (mut tx_b, rx_b) = ring::<u32>(4).unwrap();
    let (done, finished) = mpsc::channel();
    let consumer = thread::spawn(move || {
        let (mut rx_a, mut rx_b) = (rx_a, rx_b);
        let mut got = Vec::new();
        loop {
            park_on_both(&mut rx_a, &mut rx_b);
            let a = rx_a.pop();
            let b = rx_b.pop();
            if let Ok(value) = a {
                got.push(value);
            }
            if let Ok(value) = b {
                got.push(value);
            }
            if a.is_ok() || b.is_ok() {
                done.send(got.clone()).unwrap();
            }
            if a == Err(PopError::Closed) {
                return got;
            }
        }
    });
    let_it_park();
    tx_b.push(2).unwrap();
    assert_eq!(finished.recv_timeout(PATIENCE), Ok(vec![2]));
    let_it_park();
    tx_a.push(1).unwrap();
    assert_eq!(finished.recv_timeout(PATIENCE), Ok(vec![2, 1]));
    let_it_park();
    drop(tx_a);
    assert_eq!(consumer.join().unwrap(), [2, 1]);
    drop(tx_b);
}

/// `disarm` ends the wait: a push after it makes no wake-up call. A ring with a wake handle
/// shows that, because a push signals the handle only while the consumer is armed.
#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_vendor = "apple",
    windows
))]
#[test]
fn a_disarmed_ring_wakes_nobody() {
    let (mut tx, mut rx) = syntropy_spsc::ring_with_wake::<u32>(4).unwrap();
    assert_eq!(rx.arm_thread(), Arm::Waiting);
    rx.disarm();
    tx.push(1).unwrap();
    assert!(!rx.wake_handle().unwrap().reset().unwrap(), "signalled");
    assert_eq!(rx.arm_thread(), Arm::Ready);
    assert_eq!(rx.pop(), Ok(1));
}

/// `arm_thread` binds the thread as `wait` does, so arming from a second thread panics.
#[test]
fn arm_thread_binds_one_thread() {
    let (tx, mut rx) = ring::<u32>(4).unwrap();
    let first = thread::spawn(move || {
        assert_eq!(rx.arm_thread(), Arm::Waiting);
        rx.disarm();
        rx
    });
    let mut rx = first.join().unwrap();
    let second = thread::spawn(move || rx.arm_thread());
    let panic = second.join().unwrap_err();
    let message = panic.downcast_ref::<String>().unwrap();
    assert!(
        message.contains("a ring's consumer waits on the thread that first waited on it"),
        "{message}"
    );
    drop(tx);
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_vendor = "apple",
    windows
))]
mod wake_handle {
    use super::*;
    use syntropy_spsc::{WakeHandle, ring_with_wake};

    /// Whether the handle is signalled, asked of the OS without waiting: `poll` for readability
    /// on Unix, which is what tokio's `AsyncFd` waits for, and a zero-timeout wait on Windows.
    #[cfg(unix)]
    fn signalled(handle: &WakeHandle) -> bool {
        use std::os::fd::AsRawFd;
        let mut poll = libc::pollfd {
            fd: handle.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: one live `pollfd` and a zero timeout, so `poll` only reads the state.
        let ready = unsafe { libc::poll(&mut poll, 1, 0) };
        assert!(ready >= 0, "{}", std::io::Error::last_os_error());
        ready == 1 && poll.revents & libc::POLLIN != 0
    }

    #[cfg(windows)]
    fn signalled(handle: &WakeHandle) -> bool {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::WAIT_OBJECT_0;
        use windows_sys::Win32::System::Threading::WaitForSingleObject;
        // SAFETY: a zero timeout only reads the live event's state.
        unsafe { WaitForSingleObject(handle.as_raw_handle().cast(), 0) == WAIT_OBJECT_0 }
    }

    /// Blocks until the handle is signalled or `timeout_ms` passes, as a runtime's wait does:
    /// `poll` for readability on Unix, `WaitForSingleObject` on Windows. Whether it was signalled.
    #[cfg(unix)]
    fn block_on(handle: &WakeHandle, timeout_ms: u32) -> bool {
        use std::os::fd::AsRawFd;
        let mut poll = libc::pollfd {
            fd: handle.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        let timeout = i32::try_from(timeout_ms).unwrap_or(i32::MAX);
        // SAFETY: one live `pollfd`; the timeout bounds the wait.
        let ready = unsafe { libc::poll(&mut poll, 1, timeout) };
        assert!(ready >= 0, "{}", std::io::Error::last_os_error());
        ready == 1
    }

    #[cfg(windows)]
    fn block_on(handle: &WakeHandle, timeout_ms: u32) -> bool {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::WAIT_OBJECT_0;
        use windows_sys::Win32::System::Threading::WaitForSingleObject;
        // SAFETY: the live event's handle; the timeout bounds the wait.
        unsafe { WaitForSingleObject(handle.as_raw_handle().cast(), timeout_ms) == WAIT_OBJECT_0 }
    }

    #[test]
    fn a_push_signals_the_handle_only_while_the_consumer_is_armed() {
        let (mut tx, mut rx) = ring_with_wake::<u32>(8).unwrap();
        let handle = rx.wake_handle().unwrap();
        assert!(!signalled(handle));
        // Not armed: a push makes no system call and leaves the handle alone.
        tx.push(1).unwrap();
        assert!(!signalled(handle));
        assert_eq!(
            rx.arm(),
            Arm::Ready,
            "a value is ready, so there is no wait"
        );
        assert_eq!(rx.pop(), Ok(1));
        // Armed on an empty ring: the next push signals, once.
        assert_eq!(rx.arm(), Arm::Waiting);
        tx.push(2).unwrap();
        assert!(signalled(rx.wake_handle().unwrap()));
        tx.push(3).unwrap();
        assert!(rx.wake_handle().unwrap().reset().unwrap());
        assert!(!signalled(rx.wake_handle().unwrap()));
        assert!(!rx.wake_handle().unwrap().reset().unwrap());
        assert_eq!(rx.pop(), Ok(2));
        assert_eq!(rx.pop(), Ok(3));
    }

    #[test]
    fn the_producers_close_signals_an_armed_consumer() {
        let (tx, mut rx) = ring_with_wake::<u32>(8).unwrap();
        assert_eq!(rx.arm(), Arm::Waiting);
        drop(tx);
        assert!(signalled(rx.wake_handle().unwrap()));
        assert_eq!(rx.arm(), Arm::Ready);
        assert_eq!(rx.pop(), Err(syntropy_spsc::PopError::Closed));
    }

    /// The consumer's loop from `Consumer::arm`'s docs, on a thread that blocks on the handle as
    /// a runtime would: every value arrives, in order. On every OS with a wake handle: an eventfd
    /// or a kqueue polled on Unix, and an event waited on on Windows.
    #[test]
    fn a_consumer_blocking_on_the_handle_gets_every_value() {
        const COUNT: u32 = 20_000;
        let (mut tx, mut rx) = ring_with_wake::<u32>(16).unwrap();
        let consumer = thread::spawn(move || {
            let mut expected = 0;
            loop {
                if rx.arm() == Arm::Waiting {
                    // The timeout bounds the wait if a wake-up were lost.
                    assert!(
                        block_on(rx.wake_handle().unwrap(), 10_000),
                        "no wake-up within 10 s"
                    );
                    rx.wake_handle().unwrap().reset().unwrap();
                }
                loop {
                    match rx.pop() {
                        Ok(value) => {
                            assert_eq!(value, expected);
                            expected += 1;
                        }
                        Err(syntropy_spsc::PopError::Closed) => return expected,
                        Err(syntropy_spsc::PopError::Empty) => break,
                    }
                }
            }
        });
        for value in 0..COUNT {
            loop {
                match tx.push(value) {
                    Ok(()) => break,
                    Err(_) => std::hint::spin_loop(),
                }
            }
        }
        drop(tx);
        assert_eq!(consumer.join().unwrap(), COUNT);
    }
}
