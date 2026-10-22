// SPDX-License-Identifier: MIT OR Apache-2.0
// @generated from `csrc/params.h` by `cargo xtask kgen`. Do not edit: change the header
// and run `cargo xtask kgen`.

//! The `#[repr(C)]` mirrors of `csrc/params.h`, the CUDA kernels' parameter blocks
//! (RFC-0040 §1, §7.1), with compile-time checks of the layout the generator computed
//! from the header.

/// Version of this header's layouts. Raised whenever a block changes.
pub const PARAMS_ABI_VERSION: u32 = 1;

/// `Gemv`, `GateUpAct` and `Logits` (RFC-0040 §7.1).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GemvParams {
    /// Device address of the weights, in the op's `OpKey.layout`.
    pub weight: u64,
    /// Device address of the activation rows.
    pub input: u64,
    /// Device address of the output rows.
    pub output: u64,
    /// Device address of the step descriptor: the row count comes from it.
    pub step: u64,
    /// Device address of the op's workspace, or 0.
    pub workspace: u64,
    /// Output features.
    pub n_out: u32,
    /// Input features.
    pub k_in: u32,
    /// Bytes between activation rows.
    pub in_stride: u32,
    /// Bytes between output rows.
    pub out_stride: u32,
}

/// `RmsNorm` and `LayerNorm`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NormParams {
    /// Device address of the input rows.
    pub input: u64,
    /// Device address of the output rows.
    pub output: u64,
    /// Device address of the per-feature weight.
    pub weight: u64,
    /// Device address of the step descriptor.
    pub step: u64,
    /// Features per row.
    pub hidden: u32,
    /// The epsilon added to the mean square.
    pub eps: f32,
}

/// `RowGather`, `GatherRows` and `ScatterRows`: bit-exact row copies.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RowsParams {
    /// Device address of the source table or rows.
    pub src: u64,
    /// Device address of the destination rows.
    pub dst: u64,
    /// Device address of the row indices.
    pub index: u64,
    /// Device address of the step descriptor.
    pub step: u64,
    /// Bytes per row.
    pub row_bytes: u32,
    /// Rows in the source table.
    pub table_rows: u32,
}

/// `Convert`: dtype conversion and standalone dequantisation, bit-exact.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ConvertParams {
    /// Device address of the source elements or blocks.
    pub src: u64,
    /// Device address of the destination elements.
    pub dst: u64,
    /// Elements to convert.
    pub n_elements: u64,
    /// The per-tensor global scale of NVFP4; 1 for other schemes.
    pub global_scale: f32,
    /// Zero.
    pub reserved: u32,
}

// The layout the header declares, as C lays it out. Generated text, so rustfmt leaves it.
#[rustfmt::skip]
const _: () = {
    assert!(size_of::<GemvParams>() == 56);
    assert!(align_of::<GemvParams>() == 8);
    assert!(core::mem::offset_of!(GemvParams, weight) == 0);
    assert!(core::mem::offset_of!(GemvParams, input) == 8);
    assert!(core::mem::offset_of!(GemvParams, output) == 16);
    assert!(core::mem::offset_of!(GemvParams, step) == 24);
    assert!(core::mem::offset_of!(GemvParams, workspace) == 32);
    assert!(core::mem::offset_of!(GemvParams, n_out) == 40);
    assert!(core::mem::offset_of!(GemvParams, k_in) == 44);
    assert!(core::mem::offset_of!(GemvParams, in_stride) == 48);
    assert!(core::mem::offset_of!(GemvParams, out_stride) == 52);
    assert!(size_of::<NormParams>() == 40);
    assert!(align_of::<NormParams>() == 8);
    assert!(core::mem::offset_of!(NormParams, input) == 0);
    assert!(core::mem::offset_of!(NormParams, output) == 8);
    assert!(core::mem::offset_of!(NormParams, weight) == 16);
    assert!(core::mem::offset_of!(NormParams, step) == 24);
    assert!(core::mem::offset_of!(NormParams, hidden) == 32);
    assert!(core::mem::offset_of!(NormParams, eps) == 36);
    assert!(size_of::<RowsParams>() == 40);
    assert!(align_of::<RowsParams>() == 8);
    assert!(core::mem::offset_of!(RowsParams, src) == 0);
    assert!(core::mem::offset_of!(RowsParams, dst) == 8);
    assert!(core::mem::offset_of!(RowsParams, index) == 16);
    assert!(core::mem::offset_of!(RowsParams, step) == 24);
    assert!(core::mem::offset_of!(RowsParams, row_bytes) == 32);
    assert!(core::mem::offset_of!(RowsParams, table_rows) == 36);
    assert!(size_of::<ConvertParams>() == 32);
    assert!(align_of::<ConvertParams>() == 8);
    assert!(core::mem::offset_of!(ConvertParams, src) == 0);
    assert!(core::mem::offset_of!(ConvertParams, dst) == 8);
    assert!(core::mem::offset_of!(ConvertParams, n_elements) == 16);
    assert!(core::mem::offset_of!(ConvertParams, global_scale) == 24);
    assert!(core::mem::offset_of!(ConvertParams, reserved) == 28);
};

/// Each mirror's C tag, size, alignment and member offsets as the Rust compiler lays
/// them out, for the tests that compare them with the header and with a C compiler.
#[rustfmt::skip]
pub const LAYOUTS: &[crate::kgen::cgen::MirrorLayout] = &[
    crate::kgen::cgen::MirrorLayout {
        tag: "syn_gemv_params",
        size: size_of::<GemvParams>(),
        align: align_of::<GemvParams>(),
        fields: &[
            ("weight", core::mem::offset_of!(GemvParams, weight)),
            ("input", core::mem::offset_of!(GemvParams, input)),
            ("output", core::mem::offset_of!(GemvParams, output)),
            ("step", core::mem::offset_of!(GemvParams, step)),
            ("workspace", core::mem::offset_of!(GemvParams, workspace)),
            ("n_out", core::mem::offset_of!(GemvParams, n_out)),
            ("k_in", core::mem::offset_of!(GemvParams, k_in)),
            ("in_stride", core::mem::offset_of!(GemvParams, in_stride)),
            ("out_stride", core::mem::offset_of!(GemvParams, out_stride)),
        ],
    },
    crate::kgen::cgen::MirrorLayout {
        tag: "syn_norm_params",
        size: size_of::<NormParams>(),
        align: align_of::<NormParams>(),
        fields: &[
            ("input", core::mem::offset_of!(NormParams, input)),
            ("output", core::mem::offset_of!(NormParams, output)),
            ("weight", core::mem::offset_of!(NormParams, weight)),
            ("step", core::mem::offset_of!(NormParams, step)),
            ("hidden", core::mem::offset_of!(NormParams, hidden)),
            ("eps", core::mem::offset_of!(NormParams, eps)),
        ],
    },
    crate::kgen::cgen::MirrorLayout {
        tag: "syn_rows_params",
        size: size_of::<RowsParams>(),
        align: align_of::<RowsParams>(),
        fields: &[
            ("src", core::mem::offset_of!(RowsParams, src)),
            ("dst", core::mem::offset_of!(RowsParams, dst)),
            ("index", core::mem::offset_of!(RowsParams, index)),
            ("step", core::mem::offset_of!(RowsParams, step)),
            ("row_bytes", core::mem::offset_of!(RowsParams, row_bytes)),
            ("table_rows", core::mem::offset_of!(RowsParams, table_rows)),
        ],
    },
    crate::kgen::cgen::MirrorLayout {
        tag: "syn_convert_params",
        size: size_of::<ConvertParams>(),
        align: align_of::<ConvertParams>(),
        fields: &[
            ("src", core::mem::offset_of!(ConvertParams, src)),
            ("dst", core::mem::offset_of!(ConvertParams, dst)),
            ("n_elements", core::mem::offset_of!(ConvertParams, n_elements)),
            ("global_scale", core::mem::offset_of!(ConvertParams, global_scale)),
            ("reserved", core::mem::offset_of!(ConvertParams, reserved)),
        ],
    },
];
