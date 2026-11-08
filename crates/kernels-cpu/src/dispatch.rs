// SPDX-License-Identifier: MIT OR Apache-2.0

//! The per-op function-pointer table for one ISA level (RFC-0040 §7.3, §2.1).
//!
//! The provider picks the level once, and `open` fills one table for it. A kernel is found in
//! the table when an op is prepared, so hot loops never test CPU features. A table only ever
//! holds kernels compiled for its own level or a level below it in the same chain, which is
//! how a kill switch that lowers the level also lowers the table.

use std::collections::BTreeMap;

use syntropy_kernels::{CpuIsa, OpKind};

use crate::launch::Kernel;
use crate::sme2::Sme2;

/// The kernels for one [`CpuIsa`] level, by op kind.
///
/// Every level has `Gemv` (COMP-E3-T03, [`crate::gemv`]): its own SIMD kernels at `X86V3`,
/// `X86V4`, `Arm82Dot` and `Arm86I8mm`, the next level down's at `X86V4Amx` and `Arm92Sme2`,
/// and the portable kernels at `X86V2` and `Arm80`. `MoeGrouped` at the `CPU` site
/// (COMP-E3-T05, [`crate::moe`]) is built on those kernels, at the same levels. The later
/// kernel tasks add their entries in
/// [`DispatchTable::for_level`].
///
/// It has no `PartialEq`: function-pointer equality is not reliable across codegen units.
/// Compare tables by [`isa`](Self::isa) and [`len`](Self::len).
#[derive(Clone, Debug)]
pub struct DispatchTable {
    isa: CpuIsa,
    ops: BTreeMap<OpKind, Kernel>,
    sme2: Option<Sme2>,
}

impl DispatchTable {
    /// The table for `isa`: what `enumerate` reports with the class and what `open` installs.
    #[must_use]
    pub fn for_level(isa: CpuIsa) -> Self {
        // Each kernel picks its own code for the level, falling back to the next level down
        // where it has none, and never above what the live host runs.
        let mut ops = BTreeMap::new();
        ops.insert(OpKind::Gemv, crate::gemv::entry::kernel(isa));
        ops.insert(OpKind::MoeGrouped, crate::moe::entry::kernel(isa));
        Self {
            isa,
            ops,
            sme2: Sme2::for_level(isa),
        }
    }

    /// Adds `kernel` for `kind`, replacing any kernel the table had for it. The kernel tasks
    /// add theirs in [`for_level`](Self::for_level). Until then a provider built with
    /// [`CpuProvider::with_kernel`](crate::CpuProvider::with_kernel) adds stand-in kernels this
    /// way, for tests.
    pub fn insert(&mut self, kind: OpKind, kernel: Kernel) {
        self.ops.insert(kind, kernel);
    }

    /// The level this table's kernels were selected for.
    #[must_use]
    pub fn isa(&self) -> CpuIsa {
        self.isa
    }

    /// The token for the SME2 routines ([`crate::sme2`]): `Some` only in an `Arm92Sme2` table on
    /// a host that reports SME2. A kill switch or a host without SME2 gives a lower level's
    /// table, which has none, so its kernels never reach the routines (ADR-0402 item 3).
    #[must_use]
    pub fn sme2(&self) -> Option<Sme2> {
        self.sme2
    }

    /// The kernel for `kind`, if this level has one.
    #[must_use]
    pub fn get(&self, kind: &OpKind) -> Option<Kernel> {
        self.ops.get(kind).copied()
    }

    /// How many op kinds have a kernel.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether no op kind has a kernel.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}
