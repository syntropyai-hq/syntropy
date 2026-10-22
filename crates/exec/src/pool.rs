// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CPU compute pool's inline path (RFC-0012 §3.1, §3.2; RFC-0060 §2.3; ADR-0600 decision 3;
//! PLC-E7-T17).
//!
//! A doorbell's CPU work item, such as an MoE layer's CPU experts, attention partials over host
//! KV pages, or a RAM-homed dense segment, runs through a [`PoolProgram`]. The program writes the
//! item's outputs and raises its completion flag (`Q[l][d]`, `Qa[l][d]` or `H[i+1]`).
//! [`ComputePool::run_inline`] runs one item on the calling thread:
//!
//! - **The simulator** calls it when its virtual timeline reaches the item, so one OS thread runs
//!   every node.
//! - **The production pool threads** run the same items through the same program as they spin on
//!   their doorbells: per-NUMA static core sets with a barrier per op (PLC-E7-T07).
//!
//! Work and outputs are identical either way.

use syntropy_kernels::Error;

/// What a compute pool runs: the CPU work of one stage's step program, given per doorbell
/// (RFC-0012 §3.2). The step program (PLC-E7-T02) and the CPU kernels (COMP-E3) supply it.
pub trait PoolProgram {
    /// One doorbell's CPU work item: RFC-0012 §3.1's `PoolWork`.
    type Work;

    /// Runs `work` on the calling thread. It writes the outputs and raises the item's completion
    /// flag.
    ///
    /// # Errors
    ///
    /// A kernel failure. The dispatcher sees it as a pool failure and faults.
    fn run(&mut self, work: &Self::Work) -> Result<(), Error>;
}

/// One stage's CPU compute pool, as far as the inline path needs.
pub struct ComputePool<P> {
    program: P,
    ran: u64,
}

impl<P: PoolProgram> ComputePool<P> {
    /// A pool that runs `program`'s work items.
    pub fn new(program: P) -> ComputePool<P> {
        ComputePool { program, ran: 0 }
    }

    /// Executes one doorbell's CPU work item on the calling thread (RFC-0012 §3.1). Production
    /// pool threads run the same items through the same program. The simulator calls this when
    /// its virtual timeline reaches the item, so one OS thread can run every node.
    ///
    /// # Errors
    ///
    /// The program's kernel failure.
    pub fn run_inline(&mut self, work: &P::Work) -> Result<(), Error> {
        self.program.run(work)?;
        self.ran += 1;
        Ok(())
    }

    /// Work items run so far.
    #[must_use]
    pub fn ran(&self) -> u64 {
        self.ran
    }

    /// The program.
    pub fn program(&self) -> &P {
        &self.program
    }

    /// The program, for a test or simulator that inspects its outputs.
    pub fn program_mut(&mut self) -> &mut P {
        &mut self.program
    }
}
