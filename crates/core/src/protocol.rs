// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sans-I/O protocol cores (RFC-0001 §5.14 and invariant 12, RFC-0060 §2.2, ADR-0600
//! decision 1). Owner: RFC-0060.
//!
//! Membership, election and lease, the event log, plan transitions, supervisor policy and
//! recovery decisions are each a [`ProtocolCore`]: a deterministic state machine with no sockets,
//! tasks, sleeps or clock reads. A thin shell owns each core. In production the shell is a tokio
//! task that feeds the core network messages and timer expiries and carries out its outputs. In
//! the simulator the event queue is the shell. Each core's owner keeps its `Input` and `Output`
//! types public, so the simulator and the fuzzers can drive it.

use smallvec::SmallVec;

use crate::clock::MonoTime;

/// A deterministic control-plane state machine. Given the same sequence of `(now, input)` pairs,
/// a core emits the same outputs in the same order.
pub trait ProtocolCore {
    /// Inbound messages, timer expiries, local commands and peer-status changes.
    type Input;
    /// Outbound messages, timer requests, events to apply and effects for the shell.
    type Output;

    /// Handles one input at `now`, appending any outputs to `out` in the order the shell must
    /// carry them out. `now` is the only time the core sees.
    fn handle(&mut self, now: MonoTime, input: Self::Input, out: &mut Outbox<Self::Output>);

    /// The earliest time at which the core wants a timer input, if any.
    fn next_deadline(&self) -> Option<MonoTime>;
}

/// The outputs of one or more [`ProtocolCore::handle`] calls, in the order they were pushed.
///
/// Up to eight outputs are stored inline, which covers a typical step without allocating. The
/// shell drains the outbox after each call and carries out the outputs in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outbox<O> {
    items: SmallVec<[O; 8]>,
}

impl<O> Outbox<O> {
    /// An empty outbox.
    pub fn new() -> Outbox<O> {
        Outbox {
            items: SmallVec::new(),
        }
    }

    /// Appends an output after the ones already queued.
    pub fn push(&mut self, output: O) {
        self.items.push(output);
    }

    /// How many outputs are queued.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether no output is queued.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The queued outputs, oldest first.
    pub fn as_slice(&self) -> &[O] {
        &self.items
    }

    /// Removes and returns every queued output, oldest first, leaving the outbox empty.
    pub fn drain(&mut self) -> impl Iterator<Item = O> + '_ {
        self.items.drain(..)
    }
}

impl<O> Default for Outbox<O> {
    fn default() -> Outbox<O> {
        Outbox::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::Nanos;

    /// A lease that a local command grants and a timer expires: the smallest core with both a
    /// message path and a timer path.
    #[derive(Default)]
    struct LeaseCore {
        term: u64,
        expires: Option<MonoTime>,
    }

    #[derive(Debug)]
    enum LeaseInput {
        Acquire { length: Nanos },
        TimerFired,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum LeaseOutput {
        Granted { term: u64, until: MonoTime },
        SetTimer(MonoTime),
        Expired { term: u64 },
    }

    impl ProtocolCore for LeaseCore {
        type Input = LeaseInput;
        type Output = LeaseOutput;

        fn handle(&mut self, now: MonoTime, input: LeaseInput, out: &mut Outbox<LeaseOutput>) {
            match input {
                LeaseInput::Acquire { length } => {
                    self.term += 1;
                    let until = now.saturating_add(length);
                    self.expires = Some(until);
                    out.push(LeaseOutput::Granted {
                        term: self.term,
                        until,
                    });
                    out.push(LeaseOutput::SetTimer(until));
                }
                LeaseInput::TimerFired => {
                    if self.expires.is_some_and(|until| now >= until) {
                        self.expires = None;
                        out.push(LeaseOutput::Expired { term: self.term });
                    }
                }
            }
        }

        fn next_deadline(&self) -> Option<MonoTime> {
            self.expires
        }
    }

    /// A shell with no I/O: feeds each `(now, input)` and collects the outputs it drains.
    fn drive<C: ProtocolCore>(
        core: &mut C,
        script: impl IntoIterator<Item = (MonoTime, C::Input)>,
    ) -> Vec<C::Output> {
        let mut out = Outbox::new();
        let mut log = Vec::new();
        for (now, input) in script {
            core.handle(now, input, &mut out);
            log.extend(out.drain());
            assert!(out.is_empty());
        }
        log
    }

    fn script() -> Vec<(MonoTime, LeaseInput)> {
        vec![
            (
                MonoTime(100),
                LeaseInput::Acquire {
                    length: Nanos(1_000),
                },
            ),
            (MonoTime(600), LeaseInput::TimerFired),
            (MonoTime(1_100), LeaseInput::TimerFired),
            (MonoTime(1_200), LeaseInput::Acquire { length: Nanos(50) }),
        ]
    }

    #[test]
    fn a_core_runs_without_io_and_its_outputs_keep_their_order() {
        let mut core = LeaseCore::default();
        assert_eq!(core.next_deadline(), None);
        let log = drive(&mut core, script());
        assert_eq!(
            log,
            [
                LeaseOutput::Granted {
                    term: 1,
                    until: MonoTime(1_100)
                },
                LeaseOutput::SetTimer(MonoTime(1_100)),
                LeaseOutput::Expired { term: 1 },
                LeaseOutput::Granted {
                    term: 2,
                    until: MonoTime(1_250)
                },
                LeaseOutput::SetTimer(MonoTime(1_250)),
            ]
        );
        assert_eq!(core.next_deadline(), Some(MonoTime(1_250)));
    }

    #[test]
    fn the_same_script_gives_the_same_outputs() {
        let first = drive(&mut LeaseCore::default(), script());
        let second = drive(&mut LeaseCore::default(), script());
        assert_eq!(first, second);
    }

    #[test]
    fn a_core_can_be_driven_as_a_trait_object() {
        let mut core: Box<dyn ProtocolCore<Input = LeaseInput, Output = LeaseOutput>> =
            Box::new(LeaseCore::default());
        let mut out = Outbox::new();
        core.handle(
            MonoTime(5),
            LeaseInput::Acquire { length: Nanos(10) },
            &mut out,
        );
        assert_eq!(out.len(), 2);
        assert_eq!(core.next_deadline(), Some(MonoTime(15)));
    }

    #[test]
    fn the_outbox_keeps_order_past_its_inline_capacity() {
        let mut out = Outbox::default();
        assert!(out.is_empty());
        for n in 0..20u32 {
            out.push(n);
        }
        assert_eq!(out.len(), 20);
        assert_eq!(out.as_slice(), (0..20).collect::<Vec<_>>().as_slice());
        assert_eq!(out.drain().collect::<Vec<_>>(), (0..20).collect::<Vec<_>>());
        assert!(out.is_empty());
        assert_eq!(out, Outbox::new());
    }
}
