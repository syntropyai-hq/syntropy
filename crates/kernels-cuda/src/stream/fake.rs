// SPDX-License-Identifier: MIT OR Apache-2.0

//! A [`StreamApi`] that logs its calls and checks every handle it is given is live, for tests
//! without a GPU.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};

use super::{StreamApi, StreamFault};

/// One driver call, as the fake saw it.
#[derive(Clone, Debug, PartialEq)]
pub enum Call {
    CreateStream(i32),
    CreateEvent {
        timing: bool,
    },
    /// The stream, and whether the primary context was still retained.
    DestroyStream(u64, bool),
    /// The event, and whether the primary context was still retained.
    DestroyEvent(u64, bool),
    Record {
        e: u64,
        s: u64,
    },
    Wait {
        s: u64,
        e: u64,
    },
    Query(u64),
    Elapsed(u64, u64),
    Sync(u64),
    Copy {
        s: u64,
        dst: u64,
        src: u64,
        bytes: usize,
    },
    Batch {
        s: u64,
        pairs: Vec<(u64, u64, usize)>,
    },
}

/// The fake's state, shared with the test.
#[derive(Debug, Default)]
pub struct State {
    pub range: (i32, i32),
    pub batched: bool,
    pub log: Vec<Call>,
    /// The status the next call named here fails with.
    pub fail: Option<(&'static str, u32)>,
    /// What `cuEventQuery` answers.
    pub ready: bool,
    pub elapsed_ms: f32,
    pub next: u64,
    pub streams: BTreeMap<u64, i32>,
    pub events: BTreeSet<u64>,
    /// The fake driver's retain count, read when a handle is destroyed.
    pub retained: Option<Arc<AtomicI32>>,
}

impl State {
    pub fn live(&self) -> (usize, usize) {
        (self.streams.len(), self.events.len())
    }
}

/// The shared state, as the test holds it.
pub type Shared = Arc<Mutex<State>>;

pub struct FakeStreams(pub Shared);

impl FakeStreams {
    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.0.lock().unwrap()
    }
}

/// Fails `call` if the test asked for it.
fn inject(st: &mut State, call: &'static str) -> Result<(), StreamFault> {
    match st.fail {
        Some((c, status)) if c == call => {
            st.fail = None;
            Err(StreamFault {
                call,
                status,
                error: format!("CUDA error {status}"),
            })
        }
        _ => Ok(()),
    }
}

impl StreamApi for FakeStreams {
    fn priority_range(&self) -> Result<(i32, i32), StreamFault> {
        let mut st = self.state();
        inject(&mut st, "cuCtxGetStreamPriorityRange")?;
        Ok(st.range)
    }

    fn create_stream(&self, priority: i32) -> Result<u64, StreamFault> {
        let mut st = self.state();
        inject(&mut st, "cuStreamCreateWithPriority")?;
        st.next += 1;
        let h = 0x5000 + st.next;
        st.streams.insert(h, priority);
        st.log.push(Call::CreateStream(priority));
        Ok(h)
    }

    fn create_event(&self, timing: bool) -> Result<u64, StreamFault> {
        let mut st = self.state();
        inject(&mut st, "cuEventCreate")?;
        st.next += 1;
        let h = 0xE000 + st.next;
        st.events.insert(h);
        st.log.push(Call::CreateEvent { timing });
        Ok(h)
    }

    unsafe fn destroy_stream(&self, s: u64) -> Result<(), StreamFault> {
        let mut st = self.state();
        assert!(st.streams.remove(&s).is_some(), "stream {s:#x} not live");
        let held = st
            .retained
            .as_ref()
            .is_none_or(|r| r.load(Ordering::SeqCst) > 0);
        st.log.push(Call::DestroyStream(s, held));
        Ok(())
    }

    unsafe fn destroy_event(&self, e: u64) -> Result<(), StreamFault> {
        let mut st = self.state();
        assert!(st.events.remove(&e), "event {e:#x} not live");
        let held = st
            .retained
            .as_ref()
            .is_none_or(|r| r.load(Ordering::SeqCst) > 0);
        st.log.push(Call::DestroyEvent(e, held));
        Ok(())
    }

    unsafe fn record(&self, e: u64, s: u64) -> Result<(), StreamFault> {
        let mut st = self.state();
        assert!(st.events.contains(&e) && st.streams.contains_key(&s));
        inject(&mut st, "cuEventRecord")?;
        st.log.push(Call::Record { e, s });
        Ok(())
    }

    unsafe fn wait(&self, s: u64, e: u64) -> Result<(), StreamFault> {
        let mut st = self.state();
        assert!(st.events.contains(&e) && st.streams.contains_key(&s));
        inject(&mut st, "cuStreamWaitEvent")?;
        st.log.push(Call::Wait { s, e });
        Ok(())
    }

    unsafe fn query(&self, e: u64) -> Result<bool, StreamFault> {
        let mut st = self.state();
        assert!(st.events.contains(&e));
        inject(&mut st, "cuEventQuery")?;
        st.log.push(Call::Query(e));
        Ok(st.ready)
    }

    unsafe fn elapsed_ms(&self, start: u64, end: u64) -> Result<f32, StreamFault> {
        let mut st = self.state();
        assert!(st.events.contains(&start) && st.events.contains(&end));
        inject(&mut st, "cuEventElapsedTime")?;
        st.log.push(Call::Elapsed(start, end));
        Ok(st.elapsed_ms)
    }

    unsafe fn synchronize(&self, s: u64) -> Result<(), StreamFault> {
        let mut st = self.state();
        assert!(st.streams.contains_key(&s));
        inject(&mut st, "cuStreamSynchronize")?;
        st.log.push(Call::Sync(s));
        Ok(())
    }

    unsafe fn copy(&self, s: u64, dst: u64, src: u64, bytes: usize) -> Result<(), StreamFault> {
        let mut st = self.state();
        assert!(st.streams.contains_key(&s));
        assert!(bytes > 0);
        inject(&mut st, "cuMemcpyAsync")?;
        st.log.push(Call::Copy { s, dst, src, bytes });
        Ok(())
    }

    fn batched(&self) -> bool {
        self.state().batched
    }

    unsafe fn copy_batch(
        &self,
        s: u64,
        dsts: &[u64],
        srcs: &[u64],
        sizes: &[usize],
    ) -> Result<(), StreamFault> {
        let mut st = self.state();
        assert!(st.batched && st.streams.contains_key(&s));
        assert!(dsts.len() == srcs.len() && srcs.len() == sizes.len());
        assert!(!dsts.is_empty() && dsts.len() <= super::BATCH_CHUNK);
        assert!(sizes.iter().all(|&b| b > 0));
        inject(&mut st, "cuMemcpyBatchAsync_v2")?;
        let pairs = (0..dsts.len())
            .map(|i| (dsts[i], srcs[i], sizes[i]))
            .collect();
        st.log.push(Call::Batch { s, pairs });
        Ok(())
    }
}
