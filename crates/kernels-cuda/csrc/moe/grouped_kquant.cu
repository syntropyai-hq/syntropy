// SPDX-License-Identifier: MIT OR Apache-2.0
// Translation unit `moe/grouped_kquant` (RFC-0040 §10.2 rule 1): the residency-aware grouped expert FFN over k-quants (`MoeGrouped`).
//
// The kernels arrive with the COMP-E5 tasks. Until then `SYN_INSTANTIATE` expands to nothing,
// so the generated list below is read but instantiates nothing. A kernel task replaces the
// macro with the explicit instantiation of its template; no unit relies on implicit
// instantiation from another (rule 2).

#include "params.h"

#define SYN_INSTANTIATE(op, weight, activation, output, layout, fusions, tile)
#include "instantiate/moe/grouped_kquant.inc"
#undef SYN_INSTANTIATE
