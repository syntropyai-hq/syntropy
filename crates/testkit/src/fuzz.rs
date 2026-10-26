// SPDX-License-Identifier: MIT OR Apache-2.0

//! The fuzz harness of RFC-0060 §10.2 (QUAL-E10-T01): the properties every fuzz target asserts,
//! checked around one input.
//!
//! A target hands each input to [`Properties::assert`] with its parser. The parser returns
//! [`Accepted`] or the [`ErrorClass`] of its rejection, and the harness checks:
//! - **no panic:** a panic is caught and reported, never propagated past the check;
//! - **bounded allocation:** the most heap the parse holds at once, on the calling thread, is at
//!   most 4 × the input's length + 1 MiB, plus the size the format declares, if it declares one
//!   ([`Accepted::declared`]). The declared size is added to the allowance rather than replacing
//!   it, because a parser that allocates exactly what its format declares still needs a little
//!   for its own structures. A declared size larger than the input is itself a violation: §10.2
//!   has declared sizes checked against the input before anything is allocated, so a parser that
//!   reports one has trusted an unchecked size field; §10.2's "total allocation" is read as that peak: the bytes live at
//!   once beyond what was live when the parse began, as [`CountingAlloc`] tracks them
//!   ([`peak_bytes_this_thread`]). A parser that frees as it goes is held to what it keeps;
//! - **bounded time:** a parse that returns after more than 1 s is reported. A parse that never
//!   returns cannot be stopped from inside its own thread: under libFuzzer, `-timeout=1` (§10.2)
//!   ends it, and a unit test needs its own deadline;
//! - **classified rejections:** `Integrity`, `InvalidRequest`, `Auth` or `Unsupported`, never
//!   `Internal` or any other class.
//!
//! [`Properties::check`] returns every property an input broke, for tests. [`Properties::assert`]
//! panics with them, so libFuzzer records the input as a crash.
//!
//! The allocation bound needs [`CountingAlloc`] as the target binary's global allocator. Without
//! it every count reads zero, so the check refuses to run rather than pass. A parse that hands
//! work to another thread is measured on the calling thread only.
//!
//! [`fixture`] is a small format whose parser carries the planted defects the harness's own tests
//! and the `fixture` fuzz target use.

use std::fmt;
use std::hint::black_box;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::{Duration, Instant};

use syntropy_core::ErrorClass;

use crate::guards::{
    CountingAlloc, alloc_count_this_thread, live_bytes_this_thread, peak_bytes_this_thread,
    reset_peak_bytes_this_thread,
};

pub mod fixture;

/// The rejection classes §10.2 allows.
pub const ALLOWED_CLASSES: [ErrorClass; 4] = [
    ErrorClass::Integrity,
    ErrorClass::InvalidRequest,
    ErrorClass::Auth,
    ErrorClass::Unsupported,
];

/// What a parser reports about an input it accepted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Accepted {
    /// The size, in bytes, the format declares for what the input describes, when the parser
    /// may allocate that much: it is added to the default allocation bound.
    /// The parser checks a declared size against the input or file length before it allocates.
    pub declared: Option<u64>,
}

impl Accepted {
    /// An accepted input that declares no size.
    pub fn plain() -> Accepted {
        Accepted::default()
    }

    /// An accepted input whose format declares `bytes`.
    pub fn declaring(bytes: u64) -> Accepted {
        Accepted {
            declared: Some(bytes),
        }
    }
}

/// A property an input broke.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Violation {
    /// The parse panicked, with this message.
    Panic { message: String },
    /// The parse held `bytes` of heap at once, over its `bound`.
    OverAllocation { bytes: u64, bound: u64 },
    /// The parse took `elapsed`, over the time limit.
    TooSlow { elapsed: Duration, limit: Duration },
    /// The parse rejected the input with a class §10.2 does not allow.
    Rejection { class: ErrorClass },
    /// The parse accepted a declared size larger than the input: §10.2 has declared sizes checked
    /// against the actual input or file length before anything is allocated.
    UncheckedDeclaredSize { declared: u64, len: u64 },
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Violation::Panic { message } => write!(f, "the parse panicked: {message}"),
            Violation::OverAllocation { bytes, bound } => write!(
                f,
                "the parse held {bytes} bytes at once, over its bound of {bound}"
            ),
            Violation::TooSlow { elapsed, limit } => write!(
                f,
                "the parse took {} ms, over its limit of {} ms",
                elapsed.as_millis(),
                limit.as_millis()
            ),
            Violation::UncheckedDeclaredSize { declared, len } => write!(
                f,
                "the parse accepted a declared size of {declared} bytes from a {len}-byte input; \
                 a declared size is checked against the input before it is allocated"
            ),
            Violation::Rejection { class } => write!(
                f,
                "the parse rejected the input as {class:?}, which is not Integrity, \
                 InvalidRequest, Auth or Unsupported"
            ),
        }
    }
}

/// The limits of §10.2. [`Properties::default`] is the section's; tests shorten the time limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Properties {
    /// The longest a parse may take.
    pub time_limit: Duration,
    /// The bytes allowed beyond 4 × the input's length.
    pub slack: u64,
}

impl Default for Properties {
    fn default() -> Properties {
        Properties {
            time_limit: Duration::from_secs(1),
            slack: 1 << 20,
        }
    }
}

impl Properties {
    /// The allocation bound for an input of `len` bytes whose format declares `declared`: 4 ×
    /// `len` + the slack, plus `declared`.
    pub fn allocation_bound(&self, len: usize, declared: Option<u64>) -> u64 {
        let default = (len as u64).saturating_mul(4).saturating_add(self.slack);
        default.saturating_add(declared.unwrap_or(0))
    }

    /// Runs `parse` on `input` and returns every property it broke.
    ///
    /// # Panics
    ///
    /// When [`CountingAlloc`] is not the binary's global allocator, since the allocation bound
    /// could not be checked.
    pub fn check<F>(&self, input: &[u8], parse: F) -> Result<(), Vec<Violation>>
    where
        F: FnOnce(&[u8]) -> Result<Accepted, ErrorClass>,
    {
        require_counting_allocator();
        let live_before = live_bytes_this_thread();
        reset_peak_bytes_this_thread();
        let start = Instant::now();
        let outcome = catch_unwind(AssertUnwindSafe(|| parse(input)));
        let elapsed = start.elapsed();
        let bytes = u64::try_from(peak_bytes_this_thread() - live_before).unwrap_or(0);

        let mut violations = Vec::new();
        let declared = match outcome {
            Ok(Ok(accepted)) => {
                // A declared size no larger than the input is one the parser checked; a larger one
                // is a size field it trusted, which is what this property exists to catch.
                match accepted.declared {
                    Some(declared) if declared > input.len() as u64 => {
                        violations.push(Violation::UncheckedDeclaredSize {
                            declared,
                            len: input.len() as u64,
                        });
                        None
                    }
                    declared => declared,
                }
            }
            Ok(Err(class)) => {
                if !ALLOWED_CLASSES.contains(&class) {
                    violations.push(Violation::Rejection { class });
                }
                None
            }
            Err(payload) => {
                violations.push(Violation::Panic {
                    message: panic_message(payload.as_ref()),
                });
                None
            }
        };
        let bound = self.allocation_bound(input.len(), declared);
        if bytes > bound {
            violations.push(Violation::OverAllocation { bytes, bound });
        }
        if elapsed > self.time_limit {
            violations.push(Violation::TooSlow {
                elapsed,
                limit: self.time_limit,
            });
        }
        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations)
        }
    }

    /// [`Properties::check`], panicking with every violation: what a fuzz target calls, so
    /// libFuzzer records the input.
    ///
    /// # Panics
    ///
    /// When `input` breaks a property, or when [`CountingAlloc`] is not the global allocator.
    pub fn assert<F>(&self, input: &[u8], parse: F)
    where
        F: FnOnce(&[u8]) -> Result<Accepted, ErrorClass>,
    {
        if let Err(violations) = self.check(input, parse) {
            let shown: Vec<String> = violations.iter().map(Violation::to_string).collect();
            panic!(
                "RFC-0060 §10.2 broken by a {}-byte input: {}",
                input.len(),
                shown.join("; ")
            );
        }
    }
}

/// Panics unless [`CountingAlloc`] counts this thread's allocations.
fn require_counting_allocator() {
    let before = alloc_count_this_thread();
    drop(black_box(Box::new(0_u8)));
    assert!(
        alloc_count_this_thread() != before,
        "the fuzz harness needs `{}` as the global allocator, or it cannot bound allocation: add \
         `#[global_allocator] static ALLOCATOR: CountingAlloc = CountingAlloc;` to the target",
        std::any::type_name::<CountingAlloc>()
    );
}

/// The message a panic carried, when it is text.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_owned()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "a panic whose payload is not text".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bound_is_four_times_the_input_and_a_mebibyte_or_the_declared_size() {
        let properties = Properties::default();
        assert_eq!(properties.allocation_bound(0, None), 1 << 20);
        assert_eq!(properties.allocation_bound(1000, None), 4000 + (1 << 20));
        assert_eq!(
            properties.allocation_bound(1000, Some(10 << 20)),
            (10 << 20) + 4000 + (1 << 20),
            "a declared size is added to the allowance"
        );
        assert_eq!(
            properties.allocation_bound(1000, Some(0)),
            4000 + (1 << 20),
            "a declared size of zero leaves it as it is"
        );
        assert_eq!(properties.allocation_bound(usize::MAX, None), u64::MAX);
    }

    #[test]
    #[should_panic(expected = "needs `syntropy_testkit::guards::CountingAlloc` as the global")]
    fn without_the_counting_allocator_the_check_refuses_to_run() {
        // This test binary has the system allocator, so nothing is counted.
        let _ = Properties::default().check(b"", |_| Ok(Accepted::plain()));
    }

    #[test]
    fn violations_read_as_sentences() {
        assert_eq!(
            Violation::OverAllocation { bytes: 9, bound: 8 }.to_string(),
            "the parse held 9 bytes at once, over its bound of 8"
        );
        assert_eq!(
            Violation::Rejection {
                class: ErrorClass::Internal
            }
            .to_string(),
            "the parse rejected the input as Internal, which is not Integrity, InvalidRequest, \
             Auth or Unsupported"
        );
    }
}
