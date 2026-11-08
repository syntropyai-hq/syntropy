// SPDX-License-Identifier: MIT OR Apache-2.0
// @generated from `include/syntropy/attn_handoff.h` by the `syntropy-kernels` tests. Do not
// edit: change the header and run `SYNTROPY_BLESS_GENERATED=1 cargo test -p
// syntropy-kernels attn_handoff::`.

//! The `#[repr(C)]` mirrors of `include/syntropy/attn_handoff.h` (RFC-0022 §5.3; RFC-0040 §4.2), with
//! compile-time checks of the layout the generator computed from the header.

/// The version of the layouts below. A change to either struct, or to the order or alignment of
/// a hand-off layer's parts, bumps it.
pub const ATTN_HANDOFF_ABI_VERSION: u32 = 1;

/// The alignment of every part of a hand-off layer, and of the flags: one cache line.
pub const ATTN_LINE: u32 = 64;

/// The per-layer attention doorbell (RFC-0022 §5.3 `AttnDoorbell`). It sits next to the MoE
/// `syn_doorbell_header`, not inside it.
#[repr(C, align(4))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct AttnDoorbell {
    /// The low 32 bits of the step number, `syn_step_desc.step`: a payload check the pool
    /// asserts, not a flag.
    pub step: u32,
    /// Query rows in this pass, at most the layer's `rows_max`.
    pub rows: u16,
    /// Zero.
    pub _pad: u16,
    /// Host pages the CPU side attends over in this pass, over all NUMA domains; each domain's
    /// pool attends the ones in its domain. 0 when there are none.
    pub host_pages: u32,
    /// Valid entries per row of `selected`; 0 for dense kinds.
    pub selected: u32,
}

/// One flag of the handshake, alone on its cache line: 1 when raised, 0 once its consumer has
/// reset it.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct AttnFlag {
    /// The flag's value.
    pub value: u32,
}

// The layout the header declares, as C lays it out.
const _: () = {
    assert!(size_of::<AttnDoorbell>() == 16);
    assert!(align_of::<AttnDoorbell>() == 4);
    assert!(core::mem::offset_of!(AttnDoorbell, step) == 0);
    assert!(core::mem::offset_of!(AttnDoorbell, rows) == 4);
    assert!(core::mem::offset_of!(AttnDoorbell, _pad) == 6);
    assert!(core::mem::offset_of!(AttnDoorbell, host_pages) == 8);
    assert!(core::mem::offset_of!(AttnDoorbell, selected) == 12);
    assert!(size_of::<AttnFlag>() == 64);
    assert!(align_of::<AttnFlag>() == 64);
    assert!(core::mem::offset_of!(AttnFlag, value) == 0);
};

/// Each mirror's C tag, size, alignment and member offsets as the Rust compiler lays
/// them out, for the tests that compare them with the header and with a C compiler.
#[cfg(test)]
pub(super) const LAYOUTS: &[crate::step::cgen::MirrorLayout] = &[
    crate::step::cgen::MirrorLayout {
        tag: "syn_attn_doorbell",
        size: size_of::<AttnDoorbell>(),
        align: align_of::<AttnDoorbell>(),
        fields: &[
            ("step", core::mem::offset_of!(AttnDoorbell, step)),
            ("rows", core::mem::offset_of!(AttnDoorbell, rows)),
            ("_pad", core::mem::offset_of!(AttnDoorbell, _pad)),
            ("host_pages", core::mem::offset_of!(AttnDoorbell, host_pages)),
            ("selected", core::mem::offset_of!(AttnDoorbell, selected)),
        ],
    },
    crate::step::cgen::MirrorLayout {
        tag: "syn_attn_flag",
        size: size_of::<AttnFlag>(),
        align: align_of::<AttnFlag>(),
        fields: &[
            ("value", core::mem::offset_of!(AttnFlag, value)),
        ],
    },
];
