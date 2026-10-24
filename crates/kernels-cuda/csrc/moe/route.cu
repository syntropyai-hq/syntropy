// SPDX-License-Identifier: MIT OR Apache-2.0
// Translation unit `moe/route` (RFC-0040 §10.2 rule 1): router top-k and expert output combine (`MoeRoute`, `MoeCombine`).
//
// The kernels arrive with the COMP-E5 tasks. Until then `SYN_INSTANTIATE` expands to nothing,
// so the generated list below is read but instantiates nothing. A kernel task replaces the
// macro with the explicit instantiation of its template; no unit relies on implicit
// instantiation from another (rule 2).

#include "params.h"

#define SYN_INSTANTIATE(op, weight, activation, output, layout, fusions, tile)
#include "instantiate/moe/route.inc"
#undef SYN_INSTANTIATE
