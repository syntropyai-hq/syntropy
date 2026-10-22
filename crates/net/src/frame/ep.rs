// SPDX-License-Identifier: MIT OR Apache-2.0

//! The expert-parallel frames of RFC-0031 §6.8: `EpDispatch`, the rows a primary sends to an
//! expert owner with that owner's experts and weights per row, and `EpCombine`, the weighted sums
//! the owner returns (RFC-0032 §5.4).
//!
//! Both extensions start with 24 bytes of fields; a dispatch then has one routing row per
//! payload row ([`EpRow`]), and the extension is padded to 64:
//!
//! ```text
//! 0   8   step          equal to the base header's
//! 8   2   layer
//! 10  2   n_rows        rows (token positions) in this message
//! 12  1   dtype
//! 13  1   reserved
//! 14  2   part_index    0-based
//! 16  2   part_count    ≥ 1; a dispatch or combine larger than max_frame is split at row boundaries
//! 18  6   reserved
//! 24  …   per row (dispatch only), 4-byte aligned: row u16, k_local u8, pad u8,
//!         k_local × weight f32, k_local × expert u16, pad to 4
//! ```
//!
//! The payload is `n_rows` rows of `hidden` elements of `dtype`, at the `Activation` payload's
//! stride: a row's bytes rounded up to 64. `hidden` is the stage's, from the plan; the frame does
//! not carry it, so the reader passes it in.
//!
//! A message larger than `max_frame` travels in the fewest parts, each a whole frame for its
//! rows. A part's row limit counts its extension with its rows (RFC-0031 §6.8, §8):
//! [`EpDispatch::parts`] and [`EpCombine::parts`] split by it, and
//! [`EpDispatch::max_part_rows`] and [`EpDispatch::header_area`] size slots from the plan.

use std::ops::Range;

use syntropy_core::plan::WireDType;

use super::activation::{Part, dtype_code};
use super::collective::{
    CollectiveSpec, check_base, check_step, element_len, get16, get64, read_dtype,
};
use super::error::FrameError;
use super::header::{self, ALIGN, Frame, FrameType, HEADER_LEN};
use super::route::{EpRows, Head, route_row_len, walk};

/// The bytes of the fields before a dispatch's routing rows.
pub const EP_FIXED_LEN: usize = 24;

/// The fields both extensions carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EpHeader {
    /// The step, which the base header carries too.
    pub step: u64,
    pub layer: u16,
    /// Rows in this message.
    pub n_rows: u16,
    pub dtype: WireDType,
    pub part_index: u16,
    pub part_count: u16,
}

/// A frame's `EpDispatch`: the fields, a routing row per payload row, and the rows' hidden
/// states.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpDispatch<'a> {
    pub header: EpHeader,
    /// Elements per row: the stage's hidden size, from the plan. Not on the wire.
    pub hidden: u32,
    /// The routing row of each payload row, in payload order.
    pub rows: EpRows<'a>,
    /// `n_rows` rows at the stride of `hidden` elements of the dtype.
    pub payload: &'a [u8],
}

/// A frame's `EpCombine`: the fields, and per row the weighted sum of the owner's experts'
/// outputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpCombine<'a> {
    pub header: EpHeader,
    /// Elements per row: the stage's hidden size, from the plan. Not on the wire.
    pub hidden: u32,
    /// `n_rows` rows at the stride of `hidden` elements of the dtype.
    pub payload: &'a [u8],
}

/// The row stride: `hidden` elements of `dtype`, rounded up to 64 bytes.
pub(super) fn stride(dtype: WireDType, hidden: u32) -> Result<usize, FrameError> {
    let len = u64::from(hidden) * element_len(dtype)? as u64;
    Ok(len.next_multiple_of(ALIGN as u64) as usize)
}

/// A part's frame length: the base header, the fields and `meta` bytes of routing rows padded to
/// 64, and `rows` rows at `stride`.
fn frame_len(meta: usize, rows: usize, stride: usize) -> usize {
    HEADER_LEN + (EP_FIXED_LEN + meta).next_multiple_of(ALIGN) + rows * stride
}

fn read_header(ext: &[u8], wire: WireDType, ftype: FrameType) -> Result<EpHeader, FrameError> {
    if ext.len() < EP_FIXED_LEN {
        return Err(FrameError::ExtTooShort {
            ftype,
            len: ext.len(),
            needed: EP_FIXED_LEN,
        });
    }
    Ok(EpHeader {
        step: get64(ext, 0),
        layer: get16(ext, 8),
        n_rows: get16(ext, 10),
        dtype: read_dtype(ext[12], wire, ftype)?,
        part_index: get16(ext, 14),
        part_count: get16(ext, 16),
    })
}

/// Writes the fields into `ext`, which is zeroed.
fn write_header(h: &EpHeader, ext: &mut [u8]) -> Result<(), FrameError> {
    ext[0..8].copy_from_slice(&h.step.to_le_bytes());
    ext[8..10].copy_from_slice(&h.layer.to_le_bytes());
    ext[10..12].copy_from_slice(&h.n_rows.to_le_bytes());
    ext[12] = dtype_code(h.dtype).ok_or(FrameError::UnsupportedDType(h.dtype))?;
    ext[14..16].copy_from_slice(&h.part_index.to_le_bytes());
    ext[16..18].copy_from_slice(&h.part_count.to_le_bytes());
    Ok(())
}

/// The rules both frames share: a dtype this build carries, a hidden size, `part_index <
/// part_count`, at least one row, and a payload of `n_rows` strides. Returns the stride.
fn check_common(
    ftype: FrameType,
    h: &EpHeader,
    hidden: u32,
    payload: &[u8],
) -> Result<usize, FrameError> {
    let stride = stride(h.dtype, hidden)?;
    let bad = |field, value: u64| FrameError::BadField {
        ftype,
        field,
        value,
    };
    if hidden == 0 {
        return Err(bad("hidden", 0));
    }
    if h.part_count == 0 {
        return Err(bad("part_count", 0));
    }
    if h.part_index >= h.part_count {
        return Err(FrameError::Inconsistent {
            ftype,
            field: "part_index",
            value: h.part_index.into(),
            expected: h.part_count.into(),
        });
    }
    if h.n_rows == 0 {
        return Err(bad("n_rows", 0));
    }
    let expected = usize::from(h.n_rows) * stride;
    if payload.len() != expected {
        return Err(FrameError::Inconsistent {
            ftype,
            field: "payload_len",
            value: payload.len() as u64,
            expected: expected as u64,
        });
    }
    Ok(stride)
}

/// The most rows a part holds under `max_frame` when each row's routing takes `m_row` bytes:
/// the largest r with `64 + ⌈(24 + r · m_row) / 64⌉ · 64 + r · stride ≤ max_frame`, at most
/// `u16::MAX` (RFC-0031 §6.8).
fn max_rows(max_frame: usize, m_row: usize, stride: usize) -> u32 {
    let fits = |r: usize| frame_len(r * m_row, r, stride) <= max_frame;
    let (mut lo, mut hi) = (0usize, usize::from(u16::MAX));
    // The frame grows with every row, so the last row that fits is found by halving.
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo as u32
}

/// The rows from `start` that the next part holds under `max_frame`: as many as fit, each
/// adding its routing row and its stride, at most `u16::MAX`. `metas` gives the routing-row
/// bytes of the rows from `start`.
fn fit<I: Iterator<Item = usize>>(
    what: &'static str,
    mut metas: I,
    start: u32,
    n_rows: u32,
    stride: usize,
    max_frame: usize,
) -> Result<(u32, usize), FrameError> {
    let (mut end, mut meta) = (start, 0usize);
    while end < n_rows && end - start < u32::from(u16::MAX) {
        let next = meta + metas.next().unwrap_or(0);
        let len = frame_len(next, (end - start + 1) as usize, stride);
        if len > max_frame {
            if end == start {
                return Err(FrameError::TooLong { what, len });
            }
            break;
        }
        meta = next;
        end += 1;
    }
    Ok((end, frame_len(meta, (end - start) as usize, stride)))
}

/// The parts of a dispatch or combine, in order: see [`EpDispatch::parts`] and
/// [`EpCombine::parts`]. Its length is the part count.
#[derive(Clone, Debug)]
pub struct EpParts<'c, 'a> {
    message: Message<'c, 'a>,
    max_frame: usize,
    next: u32,
    index: u16,
    count: u16,
}

#[derive(Clone, Copy, Debug)]
enum Message<'c, 'a> {
    Dispatch(&'c EpDispatch<'a>),
    Combine(&'c EpCombine<'a>),
}

impl Message<'_, '_> {
    fn fit(&self, start: u32, max_frame: usize) -> Result<(u32, usize), FrameError> {
        match self {
            Message::Dispatch(d) => d.fit(start, max_frame),
            Message::Combine(c) => c.fit(start, max_frame),
        }
    }

    fn n_rows(&self) -> u32 {
        match self {
            Message::Dispatch(d) => d.header.n_rows.into(),
            Message::Combine(c) => c.header.n_rows.into(),
        }
    }
}

impl ExactSizeIterator for EpParts<'_, '_> {}

impl Iterator for EpParts<'_, '_> {
    type Item = Part;

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = usize::from(self.count - self.index);
        (left, Some(left))
    }

    fn next(&mut self) -> Option<Part> {
        if self.index == self.count {
            return None;
        }
        let (end, frame_len) = self.message.fit(self.next, self.max_frame).ok()?;
        let part = Part {
            index: self.index,
            count: self.count,
            rows: self.next..end,
            frame_len,
        };
        self.next = end;
        self.index += 1;
        Some(part)
    }
}

/// Counts the parts of a whole message, `part_count = 1`, under `max_frame`.
fn parts<'c, 'a>(
    message: Message<'c, 'a>,
    header: &EpHeader,
    ftype: FrameType,
    max_frame: usize,
) -> Result<EpParts<'c, 'a>, FrameError> {
    if header.part_count != 1 {
        return Err(FrameError::BadField {
            ftype,
            field: "part_count",
            value: header.part_count.into(),
        });
    }
    let mut count = 0usize;
    let mut start = 0;
    while start < message.n_rows() {
        start = message.fit(start, max_frame)?.0;
        count += 1;
    }
    let count = u16::try_from(count).map_err(|_| FrameError::TooLong {
        what: "part count",
        len: count,
    })?;
    Ok(EpParts {
        message,
        max_frame,
        next: 0,
        index: 0,
        count,
    })
}

/// Checks a part of a whole message with `n_rows` rows.
fn check_part(
    ftype: FrameType,
    header: &EpHeader,
    part: &Part,
) -> Result<Range<usize>, FrameError> {
    let rows = &part.rows;
    if header.part_count != 1
        || part.index >= part.count
        || rows.start >= rows.end
        || rows.end > u32::from(header.n_rows)
        || rows.end - rows.start > u32::from(u16::MAX)
    {
        return Err(FrameError::BadField {
            ftype,
            field: "part",
            value: part.index.into(),
        });
    }
    Ok(rows.start as usize..rows.end as usize)
}

impl<'a> EpDispatch<'a> {
    /// Reads an unsealed `EpDispatch` extension and payload on a channel whose wire dtype is
    /// `wire`, for a stage whose hidden size is `hidden`. A frame in another dtype is refused
    /// with [`FrameError::WireDTypeMismatch`]. Extension bytes past the routing rows are padding
    /// and ignored.
    pub fn parse(
        ext: &'a [u8],
        payload: &'a [u8],
        wire: WireDType,
        hidden: u32,
    ) -> Result<EpDispatch<'a>, FrameError> {
        let header = read_header(ext, wire, FrameType::EpDispatch)?;
        let meta = &ext[EP_FIXED_LEN..];
        let n = usize::from(header.n_rows);
        let (len, _) = walk(Head::Ep, meta, n, FrameType::EpDispatch).map_err(|e| match e {
            FrameError::ExtTooShort { ftype, len, needed } => FrameError::ExtTooShort {
                ftype,
                len: len + EP_FIXED_LEN,
                needed: needed + EP_FIXED_LEN,
            },
            other => other,
        })?;
        let dispatch = EpDispatch {
            header,
            hidden,
            rows: EpRows::wire(&meta[..len], n),
            payload,
        };
        dispatch.validate()?;
        Ok(dispatch)
    }

    /// Checks the rules the bytes cannot express: those both EP frames share (a dtype this build
    /// carries, `part_index < part_count`, at least one row, a payload of `n_rows` strides), and
    /// one routing row per payload row, each naming at least one expert, with as many weights as
    /// expert IDs.
    pub fn validate(&self) -> Result<(), FrameError> {
        let h = &self.header;
        check_common(FrameType::EpDispatch, h, self.hidden, self.payload)?;
        if self.rows.len() != usize::from(h.n_rows) {
            return Err(FrameError::Inconsistent {
                ftype: FrameType::EpDispatch,
                field: "n_rows",
                value: h.n_rows.into(),
                expected: self.rows.len() as u64,
            });
        }
        self.rows.check()
    }

    /// The header area of a dispatch part of `rows` rows that each route to at most `k_local`
    /// experts: the 64-byte base header and the extension, padded to 64. This is the plan's
    /// header slot size for EP rows (RFC-0031 §8).
    pub fn header_area(rows: u32, k_local: u8) -> usize {
        HEADER_LEN
            + (EP_FIXED_LEN + rows as usize * route_row_len(k_local.into())).next_multiple_of(ALIGN)
    }

    /// The most rows a dispatch part holds under `max_frame` when every row routes to `k_local`
    /// experts and carries `hidden` elements of `dtype`: the largest r with `64 + ⌈(24 + r ·
    /// m_row) / 64⌉ · 64 + r · row_bytes ≤ max_frame` (RFC-0031 §6.8), at most `u16::MAX`.
    pub fn max_part_rows(
        max_frame: usize,
        dtype: WireDType,
        hidden: u32,
        k_local: u8,
    ) -> Result<u32, FrameError> {
        Ok(max_rows(
            max_frame,
            route_row_len(k_local.into()),
            stride(dtype, hidden)?,
        ))
    }

    fn fit(&self, start: u32, max_frame: usize) -> Result<(u32, usize), FrameError> {
        let stride = stride(self.header.dtype, self.hidden)?;
        let metas = self
            .rows
            .iter()
            .skip(start as usize)
            .map(|row| route_row_len(row.k_local()));
        fit(
            "EpDispatch part",
            metas,
            start,
            self.header.n_rows.into(),
            stride,
            max_frame,
        )
    }

    /// The parts this dispatch travels in when frames are at most `max_frame` bytes, header
    /// included: the fewest, each holding as many rows as fit with its routing rows, which count
    /// against its row limit (RFC-0031 §6.8). A dispatch that fits is one part. The dispatch
    /// must be whole, `part_count = 1`.
    pub fn parts(&self, max_frame: usize) -> Result<EpParts<'_, 'a>, FrameError> {
        self.validate()?;
        parts(
            Message::Dispatch(self),
            &self.header,
            FrameType::EpDispatch,
            max_frame,
        )
    }

    /// Encodes rows `rows` as a frame with `part_index` and `part_count`.
    fn encode_window(
        &self,
        rows: Range<usize>,
        part: (u16, u16),
        spec: &CollectiveSpec,
        out: &mut [u8],
    ) -> Result<usize, FrameError> {
        let stride = stride(self.header.dtype, self.hidden)?;
        let meta = self.rows.span_len(rows.start, rows.end);
        let ext_len = (EP_FIXED_LEN + meta).next_multiple_of(ALIGN);
        let payload = &self.payload[rows.start * stride..rows.end * stride];
        let total = header::encoded_len(ext_len, payload.len());
        let frame = spec.frame(FrameType::EpDispatch, self.header.step)?;
        if out.len() < total {
            return Err(FrameError::BufferTooSmall {
                needed: total,
                available: out.len(),
            });
        }
        let header = EpHeader {
            n_rows: (rows.end - rows.start) as u16,
            part_index: part.0,
            part_count: part.1,
            ..self.header
        };
        let ext = &mut out[HEADER_LEN..HEADER_LEN + ext_len];
        ext.fill(0);
        write_header(&header, ext)?;
        self.rows
            .write(rows.start, rows.end, &mut ext[EP_FIXED_LEN..]);
        header::encode_in_place(&frame, ext_len, payload, out)
    }

    /// The frame's length, header and padding included.
    pub fn encoded_len(&self) -> Result<usize, FrameError> {
        self.validate()?;
        let stride = stride(self.header.dtype, self.hidden)?;
        let n = usize::from(self.header.n_rows);
        Ok(frame_len(self.rows.span_len(0, n), n, stride))
    }

    /// Encodes this frame, unsealed, into `out` and returns its length. It is written as one
    /// frame, whatever its size; [`EpDispatch::parts`] splits one that exceeds `max_frame`.
    pub fn encode(&self, spec: &CollectiveSpec, out: &mut [u8]) -> Result<usize, FrameError> {
        self.validate()?;
        let h = &self.header;
        self.encode_window(
            0..usize::from(h.n_rows),
            (h.part_index, h.part_count),
            spec,
            out,
        )
    }

    /// [`EpDispatch::encode`] into a new vector.
    pub fn encode_to_vec(&self, spec: &CollectiveSpec) -> Result<Vec<u8>, FrameError> {
        let mut out = vec![0; self.encoded_len()?];
        let len = self.encode(spec, &mut out)?;
        out.truncate(len);
        Ok(out)
    }

    /// Encodes one of this dispatch's [`EpDispatch::parts`], unsealed, into `out` and returns
    /// its length.
    pub fn encode_part(
        &self,
        part: &Part,
        spec: &CollectiveSpec,
        out: &mut [u8],
    ) -> Result<usize, FrameError> {
        self.validate()?;
        let rows = check_part(FrameType::EpDispatch, &self.header, part)?;
        self.encode_window(rows, (part.index, part.count), spec, out)
    }
}

impl<'a> EpCombine<'a> {
    /// Reads an unsealed `EpCombine` extension and payload on a channel whose wire dtype is
    /// `wire`, for a stage whose hidden size is `hidden`. A frame in another dtype is refused
    /// with [`FrameError::WireDTypeMismatch`]. Extension bytes past the fields are reserved and
    /// ignored.
    pub fn parse(
        ext: &'a [u8],
        payload: &'a [u8],
        wire: WireDType,
        hidden: u32,
    ) -> Result<EpCombine<'a>, FrameError> {
        let header = read_header(ext, wire, FrameType::EpCombine)?;
        let combine = EpCombine {
            header,
            hidden,
            payload,
        };
        combine.validate()?;
        Ok(combine)
    }

    /// Checks the rules the bytes cannot express: a dtype this build carries, `part_index <
    /// part_count`, at least one row, and a payload of `n_rows` strides.
    pub fn validate(&self) -> Result<(), FrameError> {
        check_common(
            FrameType::EpCombine,
            &self.header,
            self.hidden,
            self.payload,
        )
        .map(|_| ())
    }

    /// The header area of a combine part: the base header and the 24-byte fields padded to 64.
    pub const HEADER_AREA: usize = HEADER_LEN + ALIGN;

    /// The most rows a combine part holds under `max_frame` when each carries `hidden` elements
    /// of `dtype`, at most `u16::MAX`.
    pub fn max_part_rows(
        max_frame: usize,
        dtype: WireDType,
        hidden: u32,
    ) -> Result<u32, FrameError> {
        Ok(max_rows(max_frame, 0, stride(dtype, hidden)?))
    }

    fn fit(&self, start: u32, max_frame: usize) -> Result<(u32, usize), FrameError> {
        let stride = stride(self.header.dtype, self.hidden)?;
        fit(
            "EpCombine part",
            std::iter::repeat(0),
            start,
            self.header.n_rows.into(),
            stride,
            max_frame,
        )
    }

    /// The parts this combine travels in when frames are at most `max_frame` bytes, header
    /// included: the fewest, each holding as many rows as fit. The combine must be whole,
    /// `part_count = 1`.
    pub fn parts(&self, max_frame: usize) -> Result<EpParts<'_, 'a>, FrameError> {
        self.validate()?;
        parts(
            Message::Combine(self),
            &self.header,
            FrameType::EpCombine,
            max_frame,
        )
    }

    fn encode_window(
        &self,
        rows: Range<usize>,
        part: (u16, u16),
        spec: &CollectiveSpec,
        out: &mut [u8],
    ) -> Result<usize, FrameError> {
        let stride = stride(self.header.dtype, self.hidden)?;
        let payload = &self.payload[rows.start * stride..rows.end * stride];
        let total = header::encoded_len(ALIGN, payload.len());
        let frame = spec.frame(FrameType::EpCombine, self.header.step)?;
        if out.len() < total {
            return Err(FrameError::BufferTooSmall {
                needed: total,
                available: out.len(),
            });
        }
        let header = EpHeader {
            n_rows: (rows.end - rows.start) as u16,
            part_index: part.0,
            part_count: part.1,
            ..self.header
        };
        let ext = &mut out[HEADER_LEN..HEADER_LEN + ALIGN];
        ext.fill(0);
        write_header(&header, ext)?;
        header::encode_in_place(&frame, ALIGN, payload, out)
    }

    /// The frame's length, header and padding included.
    pub fn encoded_len(&self) -> Result<usize, FrameError> {
        self.validate()?;
        Ok(header::encoded_len(ALIGN, self.payload.len()))
    }

    /// Encodes this frame, unsealed, into `out` and returns its length. It is written as one
    /// frame, whatever its size; [`EpCombine::parts`] splits one that exceeds `max_frame`.
    pub fn encode(&self, spec: &CollectiveSpec, out: &mut [u8]) -> Result<usize, FrameError> {
        self.validate()?;
        let h = &self.header;
        self.encode_window(
            0..usize::from(h.n_rows),
            (h.part_index, h.part_count),
            spec,
            out,
        )
    }

    /// [`EpCombine::encode`] into a new vector.
    pub fn encode_to_vec(&self, spec: &CollectiveSpec) -> Result<Vec<u8>, FrameError> {
        let mut out = vec![0; self.encoded_len()?];
        let len = self.encode(spec, &mut out)?;
        out.truncate(len);
        Ok(out)
    }

    /// Encodes one of this combine's [`EpCombine::parts`], unsealed, into `out` and returns its
    /// length.
    pub fn encode_part(
        &self,
        part: &Part,
        spec: &CollectiveSpec,
        out: &mut [u8],
    ) -> Result<usize, FrameError> {
        self.validate()?;
        let rows = check_part(FrameType::EpCombine, &self.header, part)?;
        self.encode_window(rows, (part.index, part.count), spec, out)
    }
}

impl<'a> Frame<'a> {
    /// The `EpDispatch` of an unsealed frame, read on a channel whose wire dtype is `wire`, for
    /// a stage whose hidden size is `hidden`. The extension's step must be the base header's,
    /// and `HAS_TRACE` must be clear.
    pub fn ep_dispatch(&self, wire: WireDType, hidden: u32) -> Result<EpDispatch<'a>, FrameError> {
        check_base(self, FrameType::EpDispatch)?;
        let dispatch = EpDispatch::parse(self.ext, self.payload, wire, hidden)?;
        check_step(self, FrameType::EpDispatch, dispatch.header.step)?;
        Ok(dispatch)
    }

    /// The `EpCombine` of an unsealed frame, read on a channel whose wire dtype is `wire`, for a
    /// stage whose hidden size is `hidden`. The extension's step must be the base header's, and
    /// `HAS_TRACE` must be clear.
    pub fn ep_combine(&self, wire: WireDType, hidden: u32) -> Result<EpCombine<'a>, FrameError> {
        check_base(self, FrameType::EpCombine)?;
        let combine = EpCombine::parse(self.ext, self.payload, wire, hidden)?;
        check_step(self, FrameType::EpCombine, combine.header.step)?;
        Ok(combine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TB5: usize = 16_773_120;

    #[test]
    fn rfc_0031_section_6_8_derivation() {
        // Hidden 7168 in F32: a row is 28,672 bytes, already a multiple of 64.
        assert_eq!(stride(WireDType::F32, 7168).unwrap(), 28_672);
        assert_eq!(
            EpDispatch::max_part_rows(TB5, WireDType::F32, 7168, 1).unwrap(),
            584
        );
        assert_eq!(
            EpDispatch::max_part_rows(TB5, WireDType::F32, 7168, 8).unwrap(),
            583
        );
        // The bracketing frame lengths the derivation gives at k_local = 8.
        assert_eq!(frame_len(583 * 52, 583, 28_672), 16_746_240);
        assert_eq!(frame_len(584 * 52, 584, 28_672), 16_774_912);
        assert_eq!(EpDispatch::header_area(583, 8), 64 + 30_400);
    }
}
