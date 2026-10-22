// SPDX-License-Identifier: MIT OR Apache-2.0
// Translation unit `gemv/kquant` (RFC-0040 §10.2 rule 1): GEMV over the GGUF k-quants (`Gemv`, `GateUpAct`, `Logits`).
//
// The kernels arrive with the COMP-E5 tasks. Until then `SYN_INSTANTIATE` expands to nothing,
// so the generated list below is read but instantiates nothing. A kernel task replaces the
// macro with the explicit instantiation of its template; no unit relies on implicit
// instantiation from another (rule 2).

#include "params.h"

#define SYN_INSTANTIATE(op, weight, activation, output, layout, fusions, tile)
#include "instantiate/gemv/kquant.inc"
#undef SYN_INSTANTIATE
