// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multi-part prefill frames landing by offset (RFC-0031 §8, §6.3).
//!
//! A prefill chunk larger than `max_frame` travels as parts, each a whole `Activation` frame for
//! its rows, in its own receive slot. The dispatcher issues one host-to-device copy per part, into
//! consecutive offsets of the stage's device input buffer. That copy is needed anyway, so
//! reassembly costs no extra host copy. [`Landing`] gives each part its offset: the rows of the
//! earlier parts times the row stride, the layout every part of the chunk shares.
//!
//! A channel delivers in `chan_seq` order, so the parts of a chunk arrive in order, part 0 first.
//! A part out of that order, a part count or layout that changes inside a chunk, or a payload
//! whose length is not its rows times the stride, is a peer that broke §6.3: an `InvalidRequest`
//! error, and the landing starts over at the next part 0.
//!
//! **Bounded by the input.** The ranges become a device copy's destination, which no Rust bounds
//! check guards. A part's stride comes from the peer's header, so a bound in rows would not bound
//! the bytes. A `Landing` therefore knows the stage input's capacity in bytes, and refuses a part
//! whose range would end past it. Every product of rows and stride is checked, so one that would
//! overflow is refused too.

use std::ops::Range;

use syntropy_core::ErrorClass;

use crate::error::Error;
use crate::frame::{ActivationHeader, RowLayout};

/// Where one part goes: `bytes` of the stage's input buffer, and whether it completes the chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Landed {
    /// The byte range of the stage's input the part's payload is copied to.
    pub bytes: Range<u64>,
    /// The chunk's rows this part carries.
    pub rows: Range<u64>,
    /// The part is the chunk's last.
    pub last: bool,
}

/// The landing of one chunk's parts, part by part. See the [module](self).
#[derive(Clone, Debug)]
pub struct Landing {
    /// The bytes the stage's input holds: no part's range may end past them.
    capacity_bytes: u64,
    /// The chunk being landed: its part count, layout, the next part expected and the rows the
    /// parts so far carried.
    current: Option<Chunk>,
}

#[derive(Clone, Copy, Debug)]
struct Chunk {
    count: u16,
    next: u16,
    layout: RowLayout,
    rows: u64,
}

fn broken(message: String) -> Error {
    Error::uncoded(ErrorClass::InvalidRequest, message)
}

impl Landing {
    /// No chunk under way, into a stage input of `capacity_bytes` bytes.
    pub fn new(capacity_bytes: u64) -> Landing {
        Landing {
            capacity_bytes,
            current: None,
        }
    }

    /// Lands the part `header` describes, whose payload is `payload_len` bytes.
    ///
    /// # Errors
    ///
    /// `InvalidRequest` for a part out of order, a part count or row layout that changes inside
    /// the chunk, a dtype the codec does not carry, a payload that is not its rows times the
    /// stride, or a part whose bytes would end past the input's capacity. The landing then starts
    /// over at the next part 0.
    pub fn land(&mut self, header: &ActivationHeader, payload_len: u64) -> Result<Landed, Error> {
        let landed = self.place(header, payload_len);
        if landed.is_err() {
            self.current = None;
        }
        landed
    }

    fn place(&mut self, header: &ActivationHeader, payload_len: u64) -> Result<Landed, Error> {
        let stride = header.layout.stride().ok_or_else(|| {
            broken(format!(
                "a part in {:?}, which this build does not carry",
                header.layout.dtype
            ))
        })?;
        let rows = u64::from(header.n_rows);
        if rows.checked_mul(stride) != Some(payload_len) {
            return Err(broken(format!(
                "a part of {rows} rows at a {stride}-byte stride with a {payload_len}-byte payload"
            )));
        }
        if header.part_count == 0 || header.part_index >= header.part_count {
            return Err(broken(format!(
                "part {} of {}",
                header.part_index, header.part_count
            )));
        }
        let mut chunk = match self.current {
            None if header.part_index == 0 => Chunk {
                count: header.part_count,
                next: 0,
                layout: header.layout,
                rows: 0,
            },
            None => {
                return Err(broken(format!(
                    "part {} of {} with no part 0 before it",
                    header.part_index, header.part_count
                )));
            }
            Some(chunk) => chunk,
        };
        if header.part_index != chunk.next
            || header.part_count != chunk.count
            || header.layout != chunk.layout
        {
            return Err(broken(format!(
                "part {} of {} where part {} of {} was due, or its layout changed",
                header.part_index, header.part_count, chunk.next, chunk.count
            )));
        }
        // The earlier parts' range ended within the input, so `first * stride` cannot overflow.
        let first = chunk.rows;
        let end = first + rows;
        let end_bytes = end
            .checked_mul(stride)
            .filter(|&bytes| bytes <= self.capacity_bytes)
            .ok_or_else(|| {
                broken(format!(
                    "a part ending at row {end} at a {stride}-byte stride, past an input of {} bytes",
                    self.capacity_bytes
                ))
            })?;
        chunk.rows = end;
        chunk.next += 1;
        let last = chunk.next == chunk.count;
        self.current = (!last).then_some(chunk);
        Ok(Landed {
            bytes: first * stride..end_bytes,
            rows: first..end,
            last,
        })
    }
}
