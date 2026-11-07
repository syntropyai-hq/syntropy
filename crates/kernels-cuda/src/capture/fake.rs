// SPDX-License-Identifier: MIT OR Apache-2.0

//! A [`CaptureApi`] over the stream fake's state, for tests without a GPU. Work enqueued on a
//! stream being captured is recorded as the graph's nodes; every handle is checked live.

use syntropy_kernels::WaitCmp;

use super::CaptureApi;
use crate::stream::StreamFault;
use crate::stream::fake::{Call, Shared, State, inject};

pub struct FakeCapture(pub Shared);

impl FakeCapture {
    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.0.lock().unwrap()
    }
}

impl CaptureApi for FakeCapture {
    unsafe fn begin_capture(&self, s: u64) -> Result<(), StreamFault> {
        let mut st = self.state();
        assert!(st.streams.contains_key(&s), "stream {s:#x} not live");
        assert!(
            !st.capturing.contains_key(&s),
            "stream {s:#x} already capturing"
        );
        inject(&mut st, "cuStreamBeginCapture_v2")?;
        st.capturing.insert(s, Vec::new());
        st.log.push(Call::BeginCapture(s));
        Ok(())
    }

    unsafe fn end_capture(&self, s: u64) -> Result<u64, StreamFault> {
        let mut st = self.state();
        let nodes = st
            .capturing
            .remove(&s)
            .unwrap_or_else(|| panic!("stream {s:#x} not capturing"));
        inject(&mut st, "cuStreamEndCapture")?;
        st.next += 1;
        let g = 0x6000 + st.next;
        st.graphs.insert(g, nodes);
        st.log.push(Call::EndCapture(s, g));
        Ok(g)
    }

    unsafe fn instantiate(&self, graph: u64) -> Result<u64, StreamFault> {
        let mut st = self.state();
        let nodes = st.graphs.get(&graph).expect("graph not live").clone();
        inject(&mut st, "cuGraphInstantiateWithFlags")?;
        st.next += 1;
        let x = 0x7000 + st.next;
        st.execs.insert(x, nodes);
        st.free -= st.instantiate_bytes;
        st.log.push(Call::Instantiate(graph, x));
        Ok(x)
    }

    unsafe fn destroy_graph(&self, graph: u64) -> Result<(), StreamFault> {
        let mut st = self.state();
        assert!(
            st.graphs.remove(&graph).is_some(),
            "graph {graph:#x} not live"
        );
        st.log.push(Call::DestroyGraph(graph));
        Ok(())
    }

    unsafe fn upload(&self, exec: u64, s: u64) -> Result<(), StreamFault> {
        let mut st = self.state();
        assert!(st.execs.contains_key(&exec) && st.streams.contains_key(&s));
        inject(&mut st, "cuGraphUpload")?;
        st.free -= st.upload_bytes;
        st.log.push(Call::Upload { exec, s });
        Ok(())
    }

    unsafe fn launch(&self, exec: u64, s: u64) -> Result<(), StreamFault> {
        let mut st = self.state();
        assert!(st.execs.contains_key(&exec) && st.streams.contains_key(&s));
        inject(&mut st, "cuGraphLaunch")?;
        st.enqueue(s, Call::Launch { exec, s });
        Ok(())
    }

    unsafe fn destroy_exec(&self, exec: u64) -> Result<(), StreamFault> {
        let mut st = self.state();
        assert!(st.execs.remove(&exec).is_some(), "exec {exec:#x} not live");
        let held = st.held();
        st.free += st.instantiate_bytes + st.upload_bytes;
        st.log.push(Call::DestroyExec(exec, held));
        Ok(())
    }

    fn free_bytes(&self) -> Result<u64, StreamFault> {
        let mut st = self.state();
        inject(&mut st, "cuMemGetInfo_v2")?;
        Ok(st.free)
    }

    unsafe fn write_value32(&self, s: u64, addr: u64, value: u32) -> Result<(), StreamFault> {
        let mut st = self.state();
        assert!(st.streams.contains_key(&s));
        assert_eq!(addr % 4, 0);
        inject(&mut st, "cuStreamWriteValue32_v2")?;
        st.enqueue(s, Call::WriteValue { s, addr, value });
        Ok(())
    }

    unsafe fn wait_value32(
        &self,
        s: u64,
        addr: u64,
        value: u32,
        cmp: WaitCmp,
    ) -> Result<(), StreamFault> {
        let mut st = self.state();
        assert!(st.streams.contains_key(&s));
        assert_eq!(addr % 4, 0);
        inject(&mut st, "cuStreamWaitValue32_v2")?;
        let eq = cmp == WaitCmp::Eq;
        st.enqueue(s, Call::WaitValue { s, addr, value, eq });
        Ok(())
    }
}
