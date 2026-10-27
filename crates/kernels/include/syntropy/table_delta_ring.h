/* SPDX-License-Identifier: MIT OR Apache-2.0 */

/*
 * The table-delta ring a device's dispatcher posts slot-table, copy-eligibility and KV
 * block-table changes through, and the entry the first node of every captured step,
 * `ApplyTableDeltas`, applies (RFC-0023 §5.4; ADR-0404 item 5; RFC-0040 §3.4).
 *
 * This header is the one definition of these layouts. `crates/kernels/src/delta_ring/mirror.rs`
 * is the Rust mirror generated from it, and the `syntropy-kernels` tests fail when the two
 * differ (`SYNTROPY_BLESS_GENERATED=1` rewrites the mirror). The tests also compile it as C11
 * and C++11 when a compiler is present (always in CI). The generator reads the subset of C that
 * `step_desc.h` describes and skips the function-like macros below, one line each, which the
 * tests check against the Rust layout with a C compiler.
 *
 * A ring is `syn_delta_ring_header` followed by `capacity` entries of `syn_table_delta`, with
 * no padding between them: the header's size, 32, is a multiple of an entry's alignment, 4, so
 * entries start immediately after it. The dispatcher is the ring's only producer. `head` is the
 * position after the last posted delta, written with a release store; a step never reads it,
 * because it applies up to the `delta_head` its descriptor names (RFC-0040 §3.5). `applied` and
 * `applied_step` are echoed by the device to a host-mapped word, so the dispatcher learns at
 * which step each delta took effect. `capacity` is the number of entries, a power of two: the
 * plan sizes it to the ring's bound for the epoch, never below
 * `SYN_TABLE_DELTA_RING_MIN_CAPACITY`, and an overflow is a policy bug. Positions are
 * monotonic: the ring is indexed by position modulo `capacity`. All values are little-endian.
 */

#ifndef SYNTROPY_TABLE_DELTA_RING_H
#define SYNTROPY_TABLE_DELTA_RING_H

#include <stddef.h>
#include <stdint.h>

/// The version of the layouts below. A change to any struct or to the order of a ring's parts
/// bumps it.
#define SYN_TABLE_DELTA_ABI_VERSION 1

/// The smallest ring capacity a plan may give a device, in entries (RFC-0023 §5.4).
#define SYN_TABLE_DELTA_RING_MIN_CAPACITY 4096u

/// Byte offset of a ring's entries: the header's size.
#define SYN_DELTA_RING_ENTRIES_OFFSET() (sizeof(struct syn_delta_ring_header))
/// Size in bytes of a ring whose `capacity` is a power of two.
#define SYN_DELTA_RING_SIZE(capacity) (SYN_DELTA_RING_ENTRIES_OFFSET() + (size_t)(capacity) * sizeof(struct syn_table_delta))

/// One table change (RFC-0023 §5.4 `TableDelta`). `table` selects the table and `index` the
/// entry within it; `slot_of[layer][expert]` is the flat index `layer · experts + expert`.
struct syn_table_delta {
    /// Which table: a `slot_of` or `copy_eligible` table of an expert cache, or a KV block
    /// table (RFC-0040 §3.4; RFC-0022 §4). The ring's integration assigns the ids. `slot_addr`
    /// is fixed for the plan epoch and never changes through the ring.
    uint16_t table;
    /// Zero.
    uint16_t _pad;
    /// The entry's index within that table.
    uint32_t index;
    /// The entry's new value: a slot index or `MISS` for `slot_of`, a bitmap word for
    /// `copy_eligible`, a block-table entry for a KV block table.
    uint32_t value;
};

/// The fixed header of a table-delta ring (RFC-0023 §5.4 `TableDeltaRing`). It is followed by
/// `capacity` entries of `syn_table_delta`.
struct syn_delta_ring_header {
    /// The position after the last delta the dispatcher has posted, written with a release
    /// store. A step never reads it.
    uint64_t head;
    /// The position after the last delta the device has applied, echoed to the host.
    uint64_t applied;
    /// The step number whose `delta_head` produced `applied`, echoed with it.
    uint64_t applied_step;
    /// The ring's capacity in entries, a power of two.
    uint32_t capacity;
    /// Zero.
    uint32_t _pad;
};

/* SYNTROPY_TABLE_DELTA_RING_H */
#endif
