// SPDX-License-Identifier: MIT OR Apache-2.0
/*
 * The CUDA kernels' parameter blocks: the one source of truth for every kernel argument struct
 * (RFC-0040 §1). `cargo xtask kgen` generates the Rust mirrors in `src/params.rs` from this file,
 * with compile-time checks of each layout, and the crate's tests compare the Rust layout with a
 * C compiler's.
 *
 * Only the subset `src/kgen/cgen.rs` reads may appear here: `///` doc lines, block comments on
 * lines of their own, the include guard, `#include`, `#define SYN_NAME <decimal>`, and structs
 * with one `uint32_t`, `int32_t`, `uint64_t` or `float` member per line.
 *
 * Values that change every step live in the step descriptor, never here (RFC-0040 §3.5): a block
 * holds what `prepare` fixes, plus the descriptor's device address. These are the first blocks;
 * the kernel tasks add members and blocks as their kernels need them.
 */
#ifndef SYN_PARAMS_H
#define SYN_PARAMS_H

#include <stdint.h>

/// Version of this header's layouts. Raised whenever a block changes.
#define SYN_PARAMS_ABI_VERSION 1

/// `Gemv`, `GateUpAct` and `Logits` (RFC-0040 §7.1).
struct syn_gemv_params {
    /// Device address of the weights, in the op's `OpKey.layout`.
    uint64_t weight;
    /// Device address of the activation rows.
    uint64_t input;
    /// Device address of the output rows.
    uint64_t output;
    /// Device address of the step descriptor: the row count comes from it.
    uint64_t step;
    /// Device address of the op's workspace, or 0.
    uint64_t workspace;
    /// Output features.
    uint32_t n_out;
    /// Input features.
    uint32_t k_in;
    /// Bytes between activation rows.
    uint32_t in_stride;
    /// Bytes between output rows.
    uint32_t out_stride;
};

/// `RmsNorm` and `LayerNorm`.
struct syn_norm_params {
    /// Device address of the input rows.
    uint64_t input;
    /// Device address of the output rows.
    uint64_t output;
    /// Device address of the per-feature weight.
    uint64_t weight;
    /// Device address of the step descriptor.
    uint64_t step;
    /// Features per row.
    uint32_t hidden;
    /// The epsilon added to the mean square.
    float eps;
};

/// `RowGather`, `GatherRows` and `ScatterRows`: bit-exact row copies.
struct syn_rows_params {
    /// Device address of the source table or rows.
    uint64_t src;
    /// Device address of the destination rows.
    uint64_t dst;
    /// Device address of the row indices.
    uint64_t index;
    /// Device address of the step descriptor.
    uint64_t step;
    /// Bytes per row.
    uint32_t row_bytes;
    /// Rows in the source table.
    uint32_t table_rows;
};

/// `Convert`: dtype conversion and standalone dequantisation, bit-exact.
struct syn_convert_params {
    /// Device address of the source elements or blocks.
    uint64_t src;
    /// Device address of the destination elements.
    uint64_t dst;
    /// Elements to convert.
    uint64_t n_elements;
    /// The per-tensor global scale of NVFP4; 1 for other schemes.
    float global_scale;
    /// Zero.
    uint32_t reserved;
};

#endif
