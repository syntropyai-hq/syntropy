// SPDX-License-Identifier: MIT OR Apache-2.0
// @generated from `include/syntropy/moe_doorbell.h` by the `syntropy-kernels` tests. Do not
// edit: change the header and run `SYNTROPY_BLESS_GENERATED=1 cargo test -p
// syntropy-kernels moe::`.

//! The `#[repr(C)]` mirrors of `include/syntropy/moe_doorbell.h` (RFC-0023 §2; RFC-0040 §1, §3.4), with
//! compile-time checks of the layout the generator computed from the header.

/// The version of the layouts and codes below. A change to any struct, to the order of a
/// doorbell layer's parts, or to a code bumps it.
pub const MOE_DOORBELL_ABI_VERSION: u32 = 1;

/// `syn_route_slot.cache_slot` of an expert that is not resident in the device's cache.
pub const MOE_MISS: u32 = 4294967295;

/// The `assign` code of an expert computed from its cache slot (`HIT`). Every code below
/// `SYN_MOE_CPU` is a copy-slot index (`COPY`).
pub const MOE_HIT: u32 = 4294967295;

/// The `assign` code of an expert the CPU pools compute in place (`CPU`).
pub const MOE_CPU: u32 = 4294967294;

/// The fixed header of a doorbell layer (RFC-0023 §2 `DoorbellLayer`). It is followed by
/// `syn_route_slot route[MAX_UNION]` and `syn_pred_slot pred[MAX_PRED]`.
#[repr(C, align(4))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DoorbellHeader {
    /// The low 32 bits of the step number, `syn_step_desc.step`: a payload check the host
    /// asserts, not a flag (RFC-0023 §6).
    pub seq: u32,
    /// Distinct experts selected in this pass: the valid entries of `route`.
    pub n_route: u16,
    /// The valid entries of `pred`.
    pub n_pred: u16,
}

/// One expert the router selected in this pass (RFC-0023 §2).
#[repr(C, align(4))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct RouteSlot {
    /// The expert's index in its layer.
    pub expert: u16,
    /// Tokens in the pass that selected it (at least 1).
    pub tokens: u16,
    /// Its slot in the device's cache, or `SYN_MOE_MISS`.
    pub cache_slot: u32,
}

/// One predicted expert of a later layer (RFC-0023 §2, §9).
#[repr(C, align(4))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PredSlot {
    /// j: the slot predicts layer l + j, 1 <= j <= d.
    pub ahead: u8,
    /// Zero.
    pub _pad: u8,
    /// The predicted expert's index in layer l + j.
    pub expert: u16,
    /// Router probability mass, summed over the pass's tokens.
    pub score: f32,
}

// The layout the header declares, as C lays it out.
const _: () = {
    assert!(size_of::<DoorbellHeader>() == 8);
    assert!(align_of::<DoorbellHeader>() == 4);
    assert!(core::mem::offset_of!(DoorbellHeader, seq) == 0);
    assert!(core::mem::offset_of!(DoorbellHeader, n_route) == 4);
    assert!(core::mem::offset_of!(DoorbellHeader, n_pred) == 6);
    assert!(size_of::<RouteSlot>() == 8);
    assert!(align_of::<RouteSlot>() == 4);
    assert!(core::mem::offset_of!(RouteSlot, expert) == 0);
    assert!(core::mem::offset_of!(RouteSlot, tokens) == 2);
    assert!(core::mem::offset_of!(RouteSlot, cache_slot) == 4);
    assert!(size_of::<PredSlot>() == 8);
    assert!(align_of::<PredSlot>() == 4);
    assert!(core::mem::offset_of!(PredSlot, ahead) == 0);
    assert!(core::mem::offset_of!(PredSlot, _pad) == 1);
    assert!(core::mem::offset_of!(PredSlot, expert) == 2);
    assert!(core::mem::offset_of!(PredSlot, score) == 4);
};

/// Each mirror's C tag, size, alignment and member offsets as the Rust compiler lays
/// them out, for the tests that compare them with the header and with a C compiler.
#[cfg(test)]
pub(super) const LAYOUTS: &[crate::step::cgen::MirrorLayout] = &[
    crate::step::cgen::MirrorLayout {
        tag: "syn_doorbell_header",
        size: size_of::<DoorbellHeader>(),
        align: align_of::<DoorbellHeader>(),
        fields: &[
            ("seq", core::mem::offset_of!(DoorbellHeader, seq)),
            ("n_route", core::mem::offset_of!(DoorbellHeader, n_route)),
            ("n_pred", core::mem::offset_of!(DoorbellHeader, n_pred)),
        ],
    },
    crate::step::cgen::MirrorLayout {
        tag: "syn_route_slot",
        size: size_of::<RouteSlot>(),
        align: align_of::<RouteSlot>(),
        fields: &[
            ("expert", core::mem::offset_of!(RouteSlot, expert)),
            ("tokens", core::mem::offset_of!(RouteSlot, tokens)),
            ("cache_slot", core::mem::offset_of!(RouteSlot, cache_slot)),
        ],
    },
    crate::step::cgen::MirrorLayout {
        tag: "syn_pred_slot",
        size: size_of::<PredSlot>(),
        align: align_of::<PredSlot>(),
        fields: &[
            ("ahead", core::mem::offset_of!(PredSlot, ahead)),
            ("_pad", core::mem::offset_of!(PredSlot, _pad)),
            ("expert", core::mem::offset_of!(PredSlot, expert)),
            ("score", core::mem::offset_of!(PredSlot, score)),
        ],
    },
];
