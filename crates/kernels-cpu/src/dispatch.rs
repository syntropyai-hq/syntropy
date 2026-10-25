// SPDX-License-Identifier: MIT OR Apache-2.0

//! The per-op function-pointer table for one ISA level (RFC-0040 §7.3, §2.1).
//!
//! The provider picks the level once, and `open` fills one table for it. A kernel is found in
//! the table when an op is prepared, so hot loops never test CPU features. A table only ever
//! holds kernels compiled for its own level or a level below it in the same chain, which is
//! how a kill switch that lowers the level also lowers the table.

use std::collections::BTreeMap;

use syntropy_kernels::{CpuIsa, Error, OpArgs, OpKind};

use crate::sme2::Sme2;

/// One CPU kernel entry point. It runs the op on the calling thread over the buffers `args`
/// binds. The argument types are COMP-E1-T05's, so the signature follows [`OpArgs`].
pub type KernelFn = fn(&OpArgs<'_>) -> Result<(), Error>;

/// The kernels for one [`CpuIsa`] level, by op kind.
///
/// Today no kernel exists, so every level's table is empty. COMP-E3-T03 and the later kernel
/// tasks add their entries in [`DispatchTable::for_level`], one arm per level.
///
/// It has no `PartialEq`: function-pointer equality is not reliable across codegen units.
/// Compare tables by [`isa`](Self::isa) and [`len`](Self::len).
#[derive(Clone, Debug)]
pub struct DispatchTable {
    isa: CpuIsa,
    ops: BTreeMap<OpKind, KernelFn>,
    sme2: Option<Sme2>,
}

impl DispatchTable {
    /// The table for `isa`: what `enumerate` reports with the class and what `open` installs.
    #[must_use]
    pub fn for_level(isa: CpuIsa) -> Self {
        // The kernel tasks add a `match isa` here: each level's arm inserts that level's
        // kernels and, for an op the level has none of its own for, the next level down's.
        Self {
            isa,
            ops: BTreeMap::new(),
            sme2: Sme2::for_level(isa),
        }
    }

    /// Adds `kernel` for `kind`, replacing any kernel the table had for it. The kernel tasks
    /// add theirs in [`for_level`](Self::for_level). Until then a provider built with
    /// [`CpuProvider::with_kernel`](crate::CpuProvider::with_kernel) adds stand-in kernels this
    /// way, for tests.
    pub fn insert(&mut self, kind: OpKind, kernel: KernelFn) {
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
    pub fn get(&self, kind: &OpKind) -> Option<KernelFn> {
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
