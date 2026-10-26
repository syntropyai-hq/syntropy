// SPDX-License-Identifier: MIT OR Apache-2.0
// @generated from `include/syntropy/table_delta_ring.h` by the `syntropy-kernels` tests. Do not
// edit: change the header and run `SYNTROPY_BLESS_GENERATED=1 cargo test -p
// syntropy-kernels delta_ring::`.

//! The `#[repr(C)]` mirrors of `include/syntropy/table_delta_ring.h` (RFC-0023 §5.4), with
//! compile-time checks of the layout the generator computed from the header.

/// The version of the layouts below. A change to any struct or to the order of a ring's parts
/// bumps it.
pub const TABLE_DELTA_ABI_VERSION: u32 = 1;

/// The smallest ring capacity a plan may give a device, in entries (RFC-0023 §5.4).
pub const TABLE_DELTA_RING_MIN_CAPACITY: u32 = 4096;

/// One table change (RFC-0023 §5.4 `TableDelta`). `table` selects the table and `index` the
/// entry within it; `slot_of[layer][expert]` is the flat index `layer · experts + expert`.
#[repr(C, align(4))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TableDelta {
    /// Which table: a `slot_of` or `copy_eligible` table of an expert cache, or a KV block
    /// table (RFC-0040 §3.4; RFC-0022 §4). The ring's integration assigns the ids. `slot_addr`
    /// is fixed for the plan epoch and never changes through the ring.
    pub table: u16,
    /// Zero.
    pub _pad: u16,
    /// The entry's index within that table.
    pub index: u32,
    /// The entry's new value: a slot index or `MISS` for `slot_of`, a bitmap word for
    /// `copy_eligible`, a block-table entry for a KV block table.
    pub value: u32,
}

/// The fixed header of a table-delta ring (RFC-0023 §5.4 `TableDeltaRing`). It is followed by
/// `capacity` entries of `syn_table_delta`.
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DeltaRingHeader {
    /// The position after the last delta the dispatcher has posted, written with a release
    /// store. A step never reads it.
    pub head: u64,
    /// The position after the last delta the device has applied, echoed to the host.
    pub applied: u64,
    /// The step number whose `delta_head` produced `applied`, echoed with it.
    pub applied_step: u64,
    /// The ring's capacity in entries, a power of two.
    pub capacity: u32,
    /// Zero.
    pub _pad: u32,
}

// The layout the header declares, as C lays it out.
const _: () = {
    assert!(size_of::<TableDelta>() == 12);
    assert!(align_of::<TableDelta>() == 4);
    assert!(core::mem::offset_of!(TableDelta, table) == 0);
    assert!(core::mem::offset_of!(TableDelta, _pad) == 2);
    assert!(core::mem::offset_of!(TableDelta, index) == 4);
    assert!(core::mem::offset_of!(TableDelta, value) == 8);
    assert!(size_of::<DeltaRingHeader>() == 32);
    assert!(align_of::<DeltaRingHeader>() == 8);
    assert!(core::mem::offset_of!(DeltaRingHeader, head) == 0);
    assert!(core::mem::offset_of!(DeltaRingHeader, applied) == 8);
    assert!(core::mem::offset_of!(DeltaRingHeader, applied_step) == 16);
    assert!(core::mem::offset_of!(DeltaRingHeader, capacity) == 24);
    assert!(core::mem::offset_of!(DeltaRingHeader, _pad) == 28);
};

/// Each mirror's C tag, size, alignment and member offsets as the Rust compiler lays
/// them out, for the tests that compare them with the header and with a C compiler.
#[cfg(test)]
pub(super) const LAYOUTS: &[crate::step::cgen::MirrorLayout] = &[
    crate::step::cgen::MirrorLayout {
        tag: "syn_table_delta",
        size: size_of::<TableDelta>(),
        align: align_of::<TableDelta>(),
        fields: &[
            ("table", core::mem::offset_of!(TableDelta, table)),
            ("_pad", core::mem::offset_of!(TableDelta, _pad)),
            ("index", core::mem::offset_of!(TableDelta, index)),
            ("value", core::mem::offset_of!(TableDelta, value)),
        ],
    },
    crate::step::cgen::MirrorLayout {
        tag: "syn_delta_ring_header",
        size: size_of::<DeltaRingHeader>(),
        align: align_of::<DeltaRingHeader>(),
        fields: &[
            ("head", core::mem::offset_of!(DeltaRingHeader, head)),
            ("applied", core::mem::offset_of!(DeltaRingHeader, applied)),
            ("applied_step", core::mem::offset_of!(DeltaRingHeader, applied_step)),
            ("capacity", core::mem::offset_of!(DeltaRingHeader, capacity)),
            ("_pad", core::mem::offset_of!(DeltaRingHeader, _pad)),
        ],
    },
];
