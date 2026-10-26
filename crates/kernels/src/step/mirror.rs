// SPDX-License-Identifier: MIT OR Apache-2.0
// @generated from `include/syntropy/step_desc.h` by the `syntropy-kernels` tests. Do not
// edit: change the header and run `SYNTROPY_BLESS_GENERATED=1 cargo test -p
// syntropy-kernels step::`.

//! The `#[repr(C)]` mirrors of `include/syntropy/step_desc.h` (RFC-0040 §1, §3.5), with
//! compile-time checks of the layout the generator computed from the header.

/// The version of the layouts below. A change to either struct, or to the array order a
/// step descriptor follows, bumps it.
pub const STEP_DESC_ABI_VERSION: u32 = 1;

/// The alignment of a step descriptor, and the granularity its total size is rounded to.
pub const STEP_DESC_ALIGN: u32 = 64;

/// `syn_step_desc.flags`: this step carries the last chunk of a prefill.
pub const STEP_FLAG_PREFILL_LAST: u32 = 1 << 0;

/// `syn_step_desc.flags`: the rows form a token tree, and `tree_parent` holds its edges.
pub const STEP_FLAG_TREE: u32 = 1 << 1;

/// The fixed header of a step descriptor (RFC-0040 §3.5). It is followed, at the offsets it
/// names, by `u32 row_seq[R]`, `u32 row_pos[R]`, `u32 row_token[R]`, `i32 tree_parent[R]`,
/// `u32 row_kv_write_from[R]`, `u32 seq_kv_len[S]`, `u32 seq_page_table[S]`,
/// `u32 seq_state_slot[S]`, `syn_sampler_state seq_sampler[S]`, `u32 moe_balance[L_moe]` and
/// `u32 tap_rows[R_tap]`.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct StepDesc {
    /// The step number, `ActivationFrame.step`.
    pub step: u64,
    /// Rows this step computes; at most the bucket's row count.
    pub rows_active: u32,
    /// Sequences the active rows belong to.
    pub seqs_active: u32,
    /// The MoE table epoch the step reads.
    pub moe_epoch: u32,
    /// `SYN_STEP_FLAG_*` bits.
    pub flags: u32,
    /// Byte offset of `row_seq`: each row's sequence index.
    pub off_row_seq: u32,
    /// Byte offset of `row_pos`: each row's position in its sequence.
    pub off_row_pos: u32,
    /// Byte offset of `row_token`: each row's input token.
    pub off_row_token: u32,
    /// Byte offset of `tree_parent`: each row's parent row in a token tree, or -1.
    pub off_tree_parent: u32,
    /// Byte offset of `seq_kv_len`: each sequence's KV length before this step.
    pub off_seq_kv_len: u32,
    /// Byte offset of `seq_page_table`: each sequence's offset into the KV store's page table.
    pub off_seq_page_table: u32,
    /// Byte offset of `seq_state_slot`: each sequence's recurrent-state slot.
    pub off_seq_state_slot: u32,
    /// Byte offset of `seq_sampler`: each sequence's sampler state.
    pub off_seq_sampler: u32,
    /// Byte offset of `moe_balance`: per MoE layer of the stage, B_copy(l) in bytes (RFC-0040
    /// §3.4; RFC-0023 §8).
    pub off_moe_balance: u32,
    /// Byte offset of `row_kv_write_from`: per row, the first position whose KV this step
    /// writes (RFC-0033 replay); `row_pos` in normal steps.
    pub off_row_kv_write_from: u32,
    /// The table-delta ring position `ApplyTableDeltas` applies up to (RFC-0040 §3.4).
    pub delta_head: u64,
    /// Device-visible address of this step's inbound `ActivationFrame` payload.
    pub in_payload: u64,
    /// Device-visible address of the registered send buffer for the outbound frame.
    pub out_payload: u64,
    /// Tapped rows this step (RFC-0040 §3.8); 0 in service steps.
    pub rows_tap: u32,
    /// Reference ids per tapped row, K.
    pub tap_k: u32,
    /// Byte offset of `tap_rows`: the tapped row indices.
    pub off_tap_rows: u32,
    /// The doorbell slot the reference ids' arrival raises.
    pub tap_flag: u32,
    /// Pinned buffer of reference token ids, `rows_tap` x `tap_k` (`TapMode::Gather`).
    pub tap_ids: u64,
    /// Result block: per tapped row, K log-probs, logsumexp, argmax and top-2 margin.
    pub tap_out: u64,
    /// Shared-memory region for the F32 logits of tapped rows (`TapMode::FullLocal`), or 0.
    pub tap_full: u64,
}

/// One sequence's random-number key and counter (RFC-0040 §3.5; RFC-0012 §8.2; ADR-0107).
/// The sampling parameters RFC-0012 §8 defines (temperature, top-k, top-p, penalties, the
/// grammar state slot) are not part of this version of the layout.
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SamplerState {
    /// The client's seed, or the seed ingress generated (ADR-0107).
    pub seed: u64,
    /// High half of the key's 128-bit sequence component: the choice index when the client
    /// supplied the seed, else the `SequenceId`.
    pub stream_hi: u64,
    /// Low half of the key's 128-bit sequence component.
    pub stream_lo: u64,
    /// Draws taken so far. `SampleSeeded` and `VerifyAccept` report the value after their last
    /// draw in the step's result block.
    pub rng_counter: u64,
}

// The layout the header declares, as C lays it out.
const _: () = {
    assert!(size_of::<StepDesc>() == 128);
    assert!(align_of::<StepDesc>() == 64);
    assert!(core::mem::offset_of!(StepDesc, step) == 0);
    assert!(core::mem::offset_of!(StepDesc, rows_active) == 8);
    assert!(core::mem::offset_of!(StepDesc, seqs_active) == 12);
    assert!(core::mem::offset_of!(StepDesc, moe_epoch) == 16);
    assert!(core::mem::offset_of!(StepDesc, flags) == 20);
    assert!(core::mem::offset_of!(StepDesc, off_row_seq) == 24);
    assert!(core::mem::offset_of!(StepDesc, off_row_pos) == 28);
    assert!(core::mem::offset_of!(StepDesc, off_row_token) == 32);
    assert!(core::mem::offset_of!(StepDesc, off_tree_parent) == 36);
    assert!(core::mem::offset_of!(StepDesc, off_seq_kv_len) == 40);
    assert!(core::mem::offset_of!(StepDesc, off_seq_page_table) == 44);
    assert!(core::mem::offset_of!(StepDesc, off_seq_state_slot) == 48);
    assert!(core::mem::offset_of!(StepDesc, off_seq_sampler) == 52);
    assert!(core::mem::offset_of!(StepDesc, off_moe_balance) == 56);
    assert!(core::mem::offset_of!(StepDesc, off_row_kv_write_from) == 60);
    assert!(core::mem::offset_of!(StepDesc, delta_head) == 64);
    assert!(core::mem::offset_of!(StepDesc, in_payload) == 72);
    assert!(core::mem::offset_of!(StepDesc, out_payload) == 80);
    assert!(core::mem::offset_of!(StepDesc, rows_tap) == 88);
    assert!(core::mem::offset_of!(StepDesc, tap_k) == 92);
    assert!(core::mem::offset_of!(StepDesc, off_tap_rows) == 96);
    assert!(core::mem::offset_of!(StepDesc, tap_flag) == 100);
    assert!(core::mem::offset_of!(StepDesc, tap_ids) == 104);
    assert!(core::mem::offset_of!(StepDesc, tap_out) == 112);
    assert!(core::mem::offset_of!(StepDesc, tap_full) == 120);
    assert!(size_of::<SamplerState>() == 32);
    assert!(align_of::<SamplerState>() == 8);
    assert!(core::mem::offset_of!(SamplerState, seed) == 0);
    assert!(core::mem::offset_of!(SamplerState, stream_hi) == 8);
    assert!(core::mem::offset_of!(SamplerState, stream_lo) == 16);
    assert!(core::mem::offset_of!(SamplerState, rng_counter) == 24);
};

/// Each mirror's C tag, size, alignment and member offsets as the Rust compiler lays
/// them out, for the tests that compare them with the header and with a C compiler.
#[cfg(test)]
pub(super) const LAYOUTS: &[super::cgen::MirrorLayout] = &[
    super::cgen::MirrorLayout {
        tag: "syn_step_desc",
        size: size_of::<StepDesc>(),
        align: align_of::<StepDesc>(),
        fields: &[
            ("step", core::mem::offset_of!(StepDesc, step)),
            ("rows_active", core::mem::offset_of!(StepDesc, rows_active)),
            ("seqs_active", core::mem::offset_of!(StepDesc, seqs_active)),
            ("moe_epoch", core::mem::offset_of!(StepDesc, moe_epoch)),
            ("flags", core::mem::offset_of!(StepDesc, flags)),
            ("off_row_seq", core::mem::offset_of!(StepDesc, off_row_seq)),
            ("off_row_pos", core::mem::offset_of!(StepDesc, off_row_pos)),
            ("off_row_token", core::mem::offset_of!(StepDesc, off_row_token)),
            ("off_tree_parent", core::mem::offset_of!(StepDesc, off_tree_parent)),
            ("off_seq_kv_len", core::mem::offset_of!(StepDesc, off_seq_kv_len)),
            ("off_seq_page_table", core::mem::offset_of!(StepDesc, off_seq_page_table)),
            ("off_seq_state_slot", core::mem::offset_of!(StepDesc, off_seq_state_slot)),
            ("off_seq_sampler", core::mem::offset_of!(StepDesc, off_seq_sampler)),
            ("off_moe_balance", core::mem::offset_of!(StepDesc, off_moe_balance)),
            ("off_row_kv_write_from", core::mem::offset_of!(StepDesc, off_row_kv_write_from)),
            ("delta_head", core::mem::offset_of!(StepDesc, delta_head)),
            ("in_payload", core::mem::offset_of!(StepDesc, in_payload)),
            ("out_payload", core::mem::offset_of!(StepDesc, out_payload)),
            ("rows_tap", core::mem::offset_of!(StepDesc, rows_tap)),
            ("tap_k", core::mem::offset_of!(StepDesc, tap_k)),
            ("off_tap_rows", core::mem::offset_of!(StepDesc, off_tap_rows)),
            ("tap_flag", core::mem::offset_of!(StepDesc, tap_flag)),
            ("tap_ids", core::mem::offset_of!(StepDesc, tap_ids)),
            ("tap_out", core::mem::offset_of!(StepDesc, tap_out)),
            ("tap_full", core::mem::offset_of!(StepDesc, tap_full)),
        ],
    },
    super::cgen::MirrorLayout {
        tag: "syn_sampler_state",
        size: size_of::<SamplerState>(),
        align: align_of::<SamplerState>(),
        fields: &[
            ("seed", core::mem::offset_of!(SamplerState, seed)),
            ("stream_hi", core::mem::offset_of!(SamplerState, stream_hi)),
            ("stream_lo", core::mem::offset_of!(SamplerState, stream_lo)),
            ("rng_counter", core::mem::offset_of!(SamplerState, rng_counter)),
        ],
    },
];
