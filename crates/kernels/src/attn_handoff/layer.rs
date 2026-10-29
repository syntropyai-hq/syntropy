// SPDX-License-Identifier: MIT OR Apache-2.0

//! The host side of a `CpuSplit` hand-off layer: where its parts sit in the bytes of the host
//! `AttnHandoff` sub-arena ([`HandoffLayout`]), and how the doorbell, the flags and the partial
//! blocks are written and read in them (RFC-0022 §5.3; RFC-0040 §4.2).
//!
//! The order and alignment of the parts, and the row-major order inside each, are
//! `include/syntropy/attn_handoff.h`'s, whose macros the tests check against [`HandoffLayout`]
//! with a C compiler. All values are little-endian.
//!
//! **What the caller provides.** The bytes of a layer start on a 64-byte boundary: the cache-line
//! separation of the parts holds only then, which the arena's allocation guarantees (a `&[u8]`
//! cannot carry it). A side raises its flag with a release store after its last partial is
//! written, and a waiter reads the partials only after an acquire load sees the flag at 1
//! (RFC-0012 §3.2). These functions write bytes in program order and do neither: the provider
//! crates map the layer onto host-mapped memory and order the stores.

use std::fmt;
use std::mem::{offset_of, size_of};

use super::mirror::{ATTN_LINE, AttnDoorbell, AttnFlag};

/// Why a hand-off layer could not be laid out, written or read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandoffError {
    /// A shape whose two layers do not fit in a `usize`, with no rows, heads, `d_v`, `d_q` or
    /// domains, or with more rows than the doorbell's `u16` can count.
    BadShape(HandoffShape),
    /// The buffer is smaller than the layer.
    BufferTooSmall {
        /// The layer's size.
        need: usize,
        /// The buffer's length.
        have: usize,
    },
    /// A NUMA domain, row or head past the shape's.
    OutOfRange {
        /// The domain asked for.
        domain: u32,
        /// The row asked for.
        row: u32,
        /// The query head asked for.
        head: u32,
    },
    /// A selected position's row or index past the shape's.
    SelectedOutOfRange {
        /// The row asked for.
        row: u32,
        /// The index in the row asked for.
        k: u32,
    },
    /// A partial whose `o` does not have `d_v` values.
    WrongWidth {
        /// The shape's `d_v`.
        d_v: u32,
        /// The partial's `o` length.
        have: usize,
    },
    /// A doorbell whose `rows` or `selected` exceed the shape's, or whose positions do not
    /// fill `rows × selected`.
    BadDoorbell {
        /// The doorbell's rows.
        rows: u16,
        /// The doorbell's selected count per row.
        selected: u32,
        /// The positions given.
        positions: usize,
    },
}

impl fmt::Display for HandoffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HandoffError::BadShape(shape) => write!(f, "no hand-off layer for {shape:?}"),
            HandoffError::BufferTooSmall { need, have } => {
                write!(
                    f,
                    "a hand-off layer needs {need} bytes, the buffer has {have}"
                )
            }
            HandoffError::OutOfRange { domain, row, head } => write!(
                f,
                "domain {domain}, row {row}, head {head} is outside the hand-off layer"
            ),
            HandoffError::SelectedOutOfRange { row, k } => write!(
                f,
                "selected position {k} of row {row} is outside the hand-off layer"
            ),
            HandoffError::WrongWidth { d_v, have } => {
                write!(
                    f,
                    "a partial's o has {have} values, the layer's d_v is {d_v}"
                )
            }
            HandoffError::BadDoorbell {
                rows,
                selected,
                positions,
            } => write!(
                f,
                "a doorbell of {rows} rows and {selected} selected positions per row, with \
                 {positions} positions, does not fit the layer"
            ),
        }
    }
}

impl std::error::Error for HandoffError {}

/// The sizes a hand-off layer is laid out for, fixed for the plan epoch (RFC-0022 §5.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HandoffShape {
    /// The most query rows in one pass: 1 for decode, the verify window or batch above it. At
    /// most `u16::MAX`, the doorbell's `rows`.
    pub rows: u32,
    /// Query heads.
    pub n_q_heads: u32,
    /// The query's width per head, in floats.
    pub d_q: u32,
    /// The value width per head, `o`'s length, in floats.
    pub d_v: u32,
    /// The layer's selected positions per row for a sparse kind; 0 for dense kinds. It is the
    /// stride of the `selected` array: a doorbell may fill fewer per row.
    pub selected: u32,
    /// NUMA domains whose CPU pools attend: one flag pair and one partial block each.
    pub domains: u32,
}

/// One of a NUMA domain's two flags (RFC-0022 §5.3; RFC-0040 §4.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Flag {
    /// `Ra(l, d)`: the GPU raises it for domain d's pool once the doorbell and the query are
    /// written; the pool resets it after its wait.
    Ra,
    /// `Qa(l, d)`: domain d's pool raises it once its partials are written, even when its side is
    /// empty; the GPU resets it after its wait.
    Qa,
}

/// The byte offsets of one hand-off layer's parts for a [`HandoffShape`] (module docs, and
/// `attn_handoff.h`'s layer order).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HandoffLayout {
    shape: HandoffShape,
    query: usize,
    domains: usize,
    domain_size: usize,
    size: usize,
}

/// `x` rounded up to a whole number of cache lines, as `SYN_ATTN_ROUND` rounds it.
fn round(x: usize) -> Option<usize> {
    x.checked_next_multiple_of(ATTN_LINE as usize)
}

/// The product of `factors`, or `None` past `usize`.
fn product(factors: &[usize]) -> Option<usize> {
    factors.iter().try_fold(1usize, |p, &f| p.checked_mul(f))
}

const F32: usize = size_of::<f32>();
const U32: usize = size_of::<u32>();
const FLAG: usize = size_of::<AttnFlag>();

impl HandoffLayout {
    /// Lays a hand-off layer out for `shape`.
    ///
    /// # Errors
    ///
    /// [`HandoffError::BadShape`] when a count other than `selected` is zero, `rows` is past
    /// `u16::MAX`, or the sub-arena's two layers would not fit in a `usize`.
    pub fn new(shape: HandoffShape) -> Result<HandoffLayout, HandoffError> {
        let bad = HandoffError::BadShape(shape);
        let HandoffShape {
            rows,
            n_q_heads,
            d_q,
            d_v,
            selected,
            domains,
        } = shape;
        if [rows, n_q_heads, d_q, d_v, domains].contains(&0) || rows > u32::from(u16::MAX) {
            return Err(bad);
        }
        let [rows, heads, d_q, d_v, selected, domains] =
            [rows, n_q_heads, d_q, d_v, selected, domains].map(|n| n as usize);
        let layout = (|| {
            let selected_offset = Self::selected_offset();
            let query = selected_offset.checked_add(round(product(&[rows, selected, U32])?)?)?;
            let domains_at = query.checked_add(round(product(&[rows, heads, d_q, F32])?)?)?;
            let partials = round(product(&[rows, heads, d_v.checked_add(2)?, F32])?)?;
            let domain_size = (2 * FLAG).checked_add(partials)?;
            let size = domains_at.checked_add(domain_size.checked_mul(domains)?)?;
            // The sub-arena holds two layers.
            size.checked_mul(2)?;
            Some(HandoffLayout {
                shape,
                query,
                domains: domains_at,
                domain_size,
                size,
            })
        })();
        layout.ok_or(bad)
    }

    /// The shape it was laid out for.
    #[must_use]
    pub fn shape(&self) -> HandoffShape {
        self.shape
    }

    /// Byte offset of `selected`: the doorbell rounded up to a cache line.
    #[must_use]
    pub const fn selected_offset() -> usize {
        size_of::<AttnDoorbell>().next_multiple_of(ATTN_LINE as usize)
    }

    /// Byte offset of selected position `k` of `row`: `selected[row][k]`, with a row stride of
    /// the shape's `selected` (`SYN_ATTN_SELECTED_AT`).
    ///
    /// # Errors
    ///
    /// [`HandoffError::SelectedOutOfRange`] when `row` or `k` is past the shape's.
    pub fn selected_at(&self, row: u32, k: u32) -> Result<usize, HandoffError> {
        if row >= self.shape.rows || k >= self.shape.selected {
            return Err(HandoffError::SelectedOutOfRange { row, k });
        }
        let index = row as usize * self.shape.selected as usize + k as usize;
        Ok(Self::selected_offset() + index * U32)
    }

    /// Byte offset of the query.
    #[must_use]
    pub fn query_offset(&self) -> usize {
        self.query
    }

    /// Byte offset of the query of `row` and query head `head`: `query[row][head][0]`
    /// (`SYN_ATTN_QUERY_AT`).
    ///
    /// # Errors
    ///
    /// [`HandoffError::OutOfRange`] when `row` or `head` is past the shape's.
    pub fn query_at(&self, row: u32, head: u32) -> Result<usize, HandoffError> {
        self.check_slot(0, row, head)?;
        let index = row as usize * self.shape.n_q_heads as usize + head as usize;
        Ok(self.query + index * self.shape.d_q as usize * F32)
    }

    /// Bytes of the query: `rows × n_q_heads × d_q` floats.
    #[must_use]
    pub fn query_bytes(&self) -> usize {
        let HandoffShape {
            rows,
            n_q_heads,
            d_q,
            ..
        } = self.shape;
        rows as usize * n_q_heads as usize * d_q as usize * F32
    }

    /// Byte offset of domain `d`'s part: its `Ra` flag, then its `Qa` flag, then its partials.
    ///
    /// # Errors
    ///
    /// [`HandoffError::OutOfRange`] when `d` is past the shape's domains.
    pub fn domain_offset(&self, d: u32) -> Result<usize, HandoffError> {
        self.check_slot(d, 0, 0)?;
        Ok(self.domains + d as usize * self.domain_size)
    }

    /// Byte offset of domain `d`'s flag `flag`.
    ///
    /// # Errors
    ///
    /// [`HandoffError::OutOfRange`] when `d` is past the shape's domains.
    pub fn flag_offset(&self, d: u32, flag: Flag) -> Result<usize, HandoffError> {
        let base = self.domain_offset(d)?;
        Ok(match flag {
            Flag::Ra => base,
            Flag::Qa => base + FLAG,
        })
    }

    /// Byte offset of domain `d`'s partial block for `row` and query head `head`:
    /// `partials[row][head]`, row-major (`SYN_ATTN_PARTIAL_AT`).
    ///
    /// # Errors
    ///
    /// [`HandoffError::OutOfRange`] when `d`, `row` or `head` is past the shape's.
    pub fn partial_offset(&self, d: u32, row: u32, head: u32) -> Result<usize, HandoffError> {
        self.check_slot(d, row, head)?;
        let per = (self.shape.d_v as usize + 2) * F32;
        let index = row as usize * self.shape.n_q_heads as usize + head as usize;
        Ok(self.domains + d as usize * self.domain_size + 2 * FLAG + index * per)
    }

    /// Size in bytes of one hand-off layer.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Byte offset of layer `layer`'s half of the double-buffered sub-arena: the layers of even
    /// index use the first half, the odd ones the second.
    #[must_use]
    pub fn half_offset(&self, layer: u32) -> usize {
        (layer as usize % 2) * self.size
    }

    /// Size in bytes of the host `AttnHandoff` sub-arena: two layers. Part of `plan.scratch`.
    /// [`new`](Self::new) checked that it fits.
    #[must_use]
    pub fn arena_size(&self) -> usize {
        2 * self.size
    }

    fn check_len(&self, buf: &[u8]) -> Result<(), HandoffError> {
        if buf.len() < self.size {
            return Err(HandoffError::BufferTooSmall {
                need: self.size,
                have: buf.len(),
            });
        }
        Ok(())
    }

    fn check_slot(&self, d: u32, row: u32, head: u32) -> Result<(), HandoffError> {
        if d >= self.shape.domains || row >= self.shape.rows || head >= self.shape.n_q_heads {
            return Err(HandoffError::OutOfRange {
                domain: d,
                row,
                head,
            });
        }
        Ok(())
    }

    /// Writes `doorbell`, and its `rows × selected` host positions, given row-major, at the start
    /// of `buf`, one layer. Row r's positions go to `selected[r][0..selected]`; the row stride is
    /// the shape's `selected`, so a doorbell may fill fewer per row than the stride.
    ///
    /// # Errors
    ///
    /// [`HandoffError::BufferTooSmall`], or [`HandoffError::BadDoorbell`] when the doorbell's
    /// rows or selected count exceed the shape's or `positions` does not hold exactly
    /// `rows × selected` entries.
    pub fn write_doorbell(
        &self,
        doorbell: &AttnDoorbell,
        positions: &[u32],
        buf: &mut [u8],
    ) -> Result<(), HandoffError> {
        self.check_len(buf)?;
        let bad = HandoffError::BadDoorbell {
            rows: doorbell.rows,
            selected: doorbell.selected,
            positions: positions.len(),
        };
        if u32::from(doorbell.rows) > self.shape.rows
            || doorbell.selected > self.shape.selected
            || positions.len() != usize::from(doorbell.rows) * doorbell.selected as usize
        {
            return Err(bad);
        }
        put_u32(buf, offset_of!(AttnDoorbell, step), doorbell.step);
        put_u16(buf, offset_of!(AttnDoorbell, rows), doorbell.rows);
        put_u16(buf, offset_of!(AttnDoorbell, _pad), 0);
        put_u32(
            buf,
            offset_of!(AttnDoorbell, host_pages),
            doorbell.host_pages,
        );
        put_u32(buf, offset_of!(AttnDoorbell, selected), doorbell.selected);
        for row in 0..u32::from(doorbell.rows) {
            for k in 0..doorbell.selected {
                let pos = positions[row as usize * doorbell.selected as usize + k as usize];
                put_u32(buf, self.selected_at(row, k)?, pos);
            }
        }
        Ok(())
    }

    /// Reads the doorbell, and its `rows × selected` host positions, row-major, from one layer
    /// in `buf`.
    ///
    /// # Errors
    ///
    /// [`HandoffError::BufferTooSmall`], or [`HandoffError::BadDoorbell`] when the doorbell read
    /// exceeds the shape.
    pub fn read_doorbell(&self, buf: &[u8]) -> Result<(AttnDoorbell, Vec<u32>), HandoffError> {
        self.check_len(buf)?;
        let doorbell = AttnDoorbell {
            step: get_u32(buf, offset_of!(AttnDoorbell, step)),
            rows: get_u16(buf, offset_of!(AttnDoorbell, rows)),
            _pad: get_u16(buf, offset_of!(AttnDoorbell, _pad)),
            host_pages: get_u32(buf, offset_of!(AttnDoorbell, host_pages)),
            selected: get_u32(buf, offset_of!(AttnDoorbell, selected)),
        };
        if u32::from(doorbell.rows) > self.shape.rows || doorbell.selected > self.shape.selected {
            return Err(HandoffError::BadDoorbell {
                rows: doorbell.rows,
                selected: doorbell.selected,
                positions: 0,
            });
        }
        let mut positions =
            Vec::with_capacity(usize::from(doorbell.rows) * doorbell.selected as usize);
        for row in 0..u32::from(doorbell.rows) {
            for k in 0..doorbell.selected {
                positions.push(get_u32(buf, self.selected_at(row, k)?));
            }
        }
        Ok((doorbell, positions))
    }

    /// Writes domain `d`'s partial for `row` and query head `head`.
    ///
    /// # Errors
    ///
    /// [`HandoffError::BufferTooSmall`], [`HandoffError::OutOfRange`], or
    /// [`HandoffError::WrongWidth`] when `partial.o` does not hold `d_v` values.
    pub fn write_partial(
        &self,
        d: u32,
        row: u32,
        head: u32,
        partial: &Partial,
        buf: &mut [u8],
    ) -> Result<(), HandoffError> {
        self.check_len(buf)?;
        let at = self.partial_offset(d, row, head)?;
        if partial.o.len() != self.shape.d_v as usize {
            return Err(HandoffError::WrongWidth {
                d_v: self.shape.d_v,
                have: partial.o.len(),
            });
        }
        for (i, &v) in partial.o.iter().chain([&partial.m, &partial.l]).enumerate() {
            put_f32(buf, at + i * F32, v);
        }
        Ok(())
    }

    /// Reads domain `d`'s partial for `row` and query head `head`.
    ///
    /// # Errors
    ///
    /// [`HandoffError::BufferTooSmall`] or [`HandoffError::OutOfRange`].
    pub fn read_partial(
        &self,
        d: u32,
        row: u32,
        head: u32,
        buf: &[u8],
    ) -> Result<Partial, HandoffError> {
        self.check_len(buf)?;
        let at = self.partial_offset(d, row, head)?;
        let d_v = self.shape.d_v as usize;
        let o = (0..d_v).map(|i| get_f32(buf, at + i * F32)).collect();
        Ok(Partial {
            o,
            m: get_f32(buf, at + d_v * F32),
            l: get_f32(buf, at + (d_v + 1) * F32),
        })
    }

    /// Domain `d`'s whole side of a pass with no pages (RFC-0022 §5.3): writes the empty partial
    /// for every row and query head, then raises its `Qa` flag, as a side with nothing to attend
    /// always does, so the GPU's wait never depends on page counts. A side with pages for some
    /// rows only writes [`Partial::empty`] for the others with
    /// [`write_partial`](Self::write_partial).
    ///
    /// # Errors
    ///
    /// [`HandoffError::BufferTooSmall`] or [`HandoffError::OutOfRange`].
    pub fn write_empty_side(&self, d: u32, buf: &mut [u8]) -> Result<(), HandoffError> {
        self.check_len(buf)?;
        self.check_slot(d, 0, 0)?;
        let empty = Partial::empty(self.shape.d_v);
        for row in 0..self.shape.rows {
            for head in 0..self.shape.n_q_heads {
                self.write_partial(d, row, head, &empty, buf)?;
            }
        }
        self.set_flag(d, Flag::Qa, 1, buf)
    }

    /// Writes `value` into domain `d`'s flag `flag`: 1 to raise it, 0 when its consumer resets
    /// it after its `EQ 1` wait (RFC-0040 §4.2).
    ///
    /// # Errors
    ///
    /// [`HandoffError::BufferTooSmall`] or [`HandoffError::OutOfRange`].
    pub fn set_flag(
        &self,
        d: u32,
        flag: Flag,
        value: u32,
        buf: &mut [u8],
    ) -> Result<(), HandoffError> {
        self.check_len(buf)?;
        let at = self.flag_offset(d, flag)?;
        put_u32(buf, at + offset_of!(AttnFlag, value), value);
        Ok(())
    }

    /// Domain `d`'s flag `flag`.
    ///
    /// # Errors
    ///
    /// [`HandoffError::BufferTooSmall`] or [`HandoffError::OutOfRange`].
    pub fn flag(&self, d: u32, flag: Flag, buf: &[u8]) -> Result<u32, HandoffError> {
        self.check_len(buf)?;
        let at = self.flag_offset(d, flag)?;
        Ok(get_u32(buf, at + offset_of!(AttnFlag, value)))
    }
}

/// One F32 partial attention result for one query row and head (RFC-0022 §5.3 `AttnPartial`):
/// the unnormalised output `o`, the running maximum `m` and the softmax denominator `l`.
#[derive(Clone, Debug, PartialEq)]
pub struct Partial {
    /// The unnormalised output, `d_v` values.
    pub o: Vec<f32>,
    /// The running maximum of the scores.
    pub m: f32,
    /// The softmax denominator, scaled to `m`.
    pub l: f32,
}

impl Partial {
    /// The partial of a side with no pages: `o = 0`, `m = −∞`, `l = 0`. `AttnMerge` gives it no
    /// weight: `e^(m − m') = 0` for any finite `m'`.
    #[must_use]
    pub fn empty(d_v: u32) -> Partial {
        Partial {
            o: vec![0.0; d_v as usize],
            m: f32::NEG_INFINITY,
            l: 0.0,
        }
    }

    /// Whether this is the empty-side encoding, bit for bit: every `o` and `l` +0.0, and `m`
    /// −∞.
    #[must_use]
    pub fn is_empty_side(&self) -> bool {
        self.o.iter().all(|v| v.to_bits() == 0)
            && self.m.to_bits() == f32::NEG_INFINITY.to_bits()
            && self.l.to_bits() == 0
    }
}

fn put_u16(buf: &mut [u8], at: usize, v: u16) {
    buf[at..at + 2].copy_from_slice(&v.to_le_bytes());
}

fn put_u32(buf: &mut [u8], at: usize, v: u32) {
    buf[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn put_f32(buf: &mut [u8], at: usize, v: f32) {
    buf[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn get_u16(buf: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(buf[at..at + 2].try_into().expect("two bytes"))
}

fn get_u32(buf: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(buf[at..at + 4].try_into().expect("four bytes"))
}

fn get_f32(buf: &[u8], at: usize) -> f32 {
    f32::from_le_bytes(buf[at..at + 4].try_into().expect("four bytes"))
}
