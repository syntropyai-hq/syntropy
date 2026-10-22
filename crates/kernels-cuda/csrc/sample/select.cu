// SPDX-License-Identifier: MIT OR Apache-2.0
// Translation unit `sample/select` (RFC-0040 §10.2 rule 1): argmax and top-k/top-p index selection (`Argmax`, `TopKTopP`).
// Bit-exact class: compiled with `--fmad=false` (RFC-0040 §10.2 rule 7; RFC-0042 §6.4).
//
// The kernels arrive with the COMP-E5 tasks. Until then `SYN_INSTANTIATE` expands to nothing,
// so the generated list below is read but instantiates nothing. A kernel task replaces the
// macro with the explicit instantiation of its template; no unit relies on implicit
// instantiation from another (rule 2).

#include "params.h"

#define SYN_INSTANTIATE(op, weight, activation, output, layout, fusions, tile)
#include "instantiate/sample/select.inc"
#undef SYN_INSTANTIATE
