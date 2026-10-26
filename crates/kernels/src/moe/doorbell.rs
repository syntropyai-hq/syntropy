// SPDX-License-Identifier: MIT OR Apache-2.0

//! A whole doorbell layer: the [`DoorbellHeader`], then `route[MAX_UNION]` and
//! `pred[MAX_PRED]`, sized per plan epoch and written into host-mapped bytes (RFC-0023 §2;
//! RFC-0040 §3.4).

use std::fmt;
use std::mem::{offset_of, size_of};

use super::mirror::{DoorbellHeader, PredSlot, RouteSlot};

/// Why a doorbell layer could not be written or read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoorbellError {
    /// The buffer is smaller than the layout.
    BufferTooSmall { need: usize, have: usize },
    /// More route entries than `MAX_UNION`.
    TooManyRoutes { count: usize, max: u16 },
    /// More predictions than `MAX_PRED`.
    TooManyPreds { count: usize, max: u16 },
}

impl fmt::Display for DoorbellError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::BufferTooSmall { need, have } => {
                write!(
                    f,
                    "doorbell buffer holds {have} bytes; the layout needs {need}"
                )
            }
            Self::TooManyRoutes { count, max } => {
                write!(f, "{count} route entries; MAX_UNION is {max}")
            }
            Self::TooManyPreds { count, max } => {
                write!(f, "{count} predictions; MAX_PRED is {max}")
            }
        }
    }
}

impl std::error::Error for DoorbellError {}

/// Where `route` and `pred` sit in a doorbell layer, and its size, for one epoch's
/// `MAX_UNION` and `MAX_PRED` (RFC-0023 §2). TIER-E8-T01 computes the two from the plan.
///
/// The parts follow each other with no padding, as `SYN_DOORBELL_PRED_OFFSET` and
/// `SYN_DOORBELL_LAYER_SIZE` in `moe_doorbell.h` compute them: every part's size is a multiple
/// of 4, the alignment of each part.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DoorbellLayout {
    max_union: u16,
    max_pred: u16,
}

impl DoorbellLayout {
    /// The layout for `max_union` route entries and `max_pred` predictions. Both are `u16`s, as
    /// the header's counts are.
    #[must_use]
    pub fn new(max_union: u16, max_pred: u16) -> Self {
        Self {
            max_union,
            max_pred,
        }
    }

    /// `MAX_UNION`.
    #[must_use]
    pub fn max_union(&self) -> u16 {
        self.max_union
    }

    /// `MAX_PRED`.
    #[must_use]
    pub fn max_pred(&self) -> u16 {
        self.max_pred
    }

    /// Byte offset of `route`: the header's size.
    #[must_use]
    pub fn route_offset(&self) -> usize {
        size_of::<DoorbellHeader>()
    }

    /// Byte offset of `pred`.
    #[must_use]
    pub fn pred_offset(&self) -> usize {
        self.route_offset() + usize::from(self.max_union) * size_of::<RouteSlot>()
    }

    /// The layer's size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.pred_offset() + usize::from(self.max_pred) * size_of::<PredSlot>()
    }

    fn check(&self, len: usize) -> Result<(), DoorbellError> {
        if len < self.size() {
            return Err(DoorbellError::BufferTooSmall {
                need: self.size(),
                have: len,
            });
        }
        Ok(())
    }
}

/// A doorbell layer's contents: what the router op writes for one MoE layer each pass
/// (RFC-0023 §2; RFC-0040 §3.4). `n_route` is `route.len()` and `n_pred` is `pred.len()`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DoorbellLayer {
    /// The low 32 bits of the step number.
    pub seq: u32,
    /// The distinct experts selected in this pass.
    pub route: Vec<RouteSlot>,
    /// Predictions for later layers.
    pub pred: Vec<PredSlot>,
}

fn put16(buf: &mut [u8], at: usize, v: u16) {
    buf[at..at + 2].copy_from_slice(&v.to_le_bytes());
}

fn put32(buf: &mut [u8], at: usize, v: u32) {
    buf[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn get16(buf: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(buf[at..at + 2].try_into().expect("2 bytes"))
}

fn get32(buf: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(buf[at..at + 4].try_into().expect("4 bytes"))
}

macro_rules! header_fields {
    ($mac:ident) => {
        $mac!(DoorbellHeader, seq, 32);
        $mac!(DoorbellHeader, n_route, 16);
        $mac!(DoorbellHeader, n_pred, 16);
    };
}

macro_rules! route_fields {
    ($mac:ident) => {
        $mac!(RouteSlot, expert, 16);
        $mac!(RouteSlot, tokens, 16);
        $mac!(RouteSlot, cache_slot, 32);
    };
}

macro_rules! pred_fields {
    ($mac:ident) => {
        $mac!(PredSlot, ahead, 8);
        $mac!(PredSlot, _pad, 8);
        $mac!(PredSlot, expert, 16);
        $mac!(PredSlot, score, f32);
    };
}

impl DoorbellHeader {
    /// Writes the members at `base`, little-endian, at the offsets the mirror gives them.
    fn write_to(&self, buf: &mut [u8], base: usize) {
        macro_rules! put {
            ($s:ident, $f:ident, 8) => {
                buf[base + offset_of!($s, $f)] = self.$f
            };
            ($s:ident, $f:ident, 16) => {
                put16(buf, base + offset_of!($s, $f), self.$f)
            };
            ($s:ident, $f:ident, 32) => {
                put32(buf, base + offset_of!($s, $f), self.$f)
            };
            ($s:ident, $f:ident, f32) => {
                put32(buf, base + offset_of!($s, $f), self.$f.to_bits())
            };
        }
        header_fields!(put);
    }

    /// Reads what [`write_to`](Self::write_to) wrote.
    fn read_from(buf: &[u8], base: usize) -> Self {
        let mut v = Self::default();
        macro_rules! get {
            ($s:ident, $f:ident, 8) => {
                v.$f = buf[base + offset_of!($s, $f)]
            };
            ($s:ident, $f:ident, 16) => {
                v.$f = get16(buf, base + offset_of!($s, $f))
            };
            ($s:ident, $f:ident, 32) => {
                v.$f = get32(buf, base + offset_of!($s, $f))
            };
            ($s:ident, $f:ident, f32) => {
                v.$f = f32::from_bits(get32(buf, base + offset_of!($s, $f)))
            };
        }
        header_fields!(get);
        v
    }
}

impl RouteSlot {
    /// Writes the members at `base`, little-endian, at the offsets the mirror gives them.
    fn write_to(&self, buf: &mut [u8], base: usize) {
        macro_rules! put {
            ($s:ident, $f:ident, 8) => {
                buf[base + offset_of!($s, $f)] = self.$f
            };
            ($s:ident, $f:ident, 16) => {
                put16(buf, base + offset_of!($s, $f), self.$f)
            };
            ($s:ident, $f:ident, 32) => {
                put32(buf, base + offset_of!($s, $f), self.$f)
            };
            ($s:ident, $f:ident, f32) => {
                put32(buf, base + offset_of!($s, $f), self.$f.to_bits())
            };
        }
        route_fields!(put);
    }

    /// Reads what [`write_to`](Self::write_to) wrote.
    fn read_from(buf: &[u8], base: usize) -> Self {
        let mut v = Self::default();
        macro_rules! get {
            ($s:ident, $f:ident, 8) => {
                v.$f = buf[base + offset_of!($s, $f)]
            };
            ($s:ident, $f:ident, 16) => {
                v.$f = get16(buf, base + offset_of!($s, $f))
            };
            ($s:ident, $f:ident, 32) => {
                v.$f = get32(buf, base + offset_of!($s, $f))
            };
            ($s:ident, $f:ident, f32) => {
                v.$f = f32::from_bits(get32(buf, base + offset_of!($s, $f)))
            };
        }
        route_fields!(get);
        v
    }
}

impl PredSlot {
    /// Writes the members at `base`, little-endian, at the offsets the mirror gives them.
    fn write_to(&self, buf: &mut [u8], base: usize) {
        macro_rules! put {
            ($s:ident, $f:ident, 8) => {
                buf[base + offset_of!($s, $f)] = self.$f
            };
            ($s:ident, $f:ident, 16) => {
                put16(buf, base + offset_of!($s, $f), self.$f)
            };
            ($s:ident, $f:ident, 32) => {
                put32(buf, base + offset_of!($s, $f), self.$f)
            };
            ($s:ident, $f:ident, f32) => {
                put32(buf, base + offset_of!($s, $f), self.$f.to_bits())
            };
        }
        pred_fields!(put);
    }

    /// Reads what [`write_to`](Self::write_to) wrote.
    fn read_from(buf: &[u8], base: usize) -> Self {
        let mut v = Self::default();
        macro_rules! get {
            ($s:ident, $f:ident, 8) => {
                v.$f = buf[base + offset_of!($s, $f)]
            };
            ($s:ident, $f:ident, 16) => {
                v.$f = get16(buf, base + offset_of!($s, $f))
            };
            ($s:ident, $f:ident, 32) => {
                v.$f = get32(buf, base + offset_of!($s, $f))
            };
            ($s:ident, $f:ident, f32) => {
                v.$f = f32::from_bits(get32(buf, base + offset_of!($s, $f)))
            };
        }
        pred_fields!(get);
        v
    }
}

impl DoorbellLayer {
    fn check(&self, layout: &DoorbellLayout) -> Result<(), DoorbellError> {
        if self.route.len() > usize::from(layout.max_union) {
            return Err(DoorbellError::TooManyRoutes {
                count: self.route.len(),
                max: layout.max_union,
            });
        }
        if self.pred.len() > usize::from(layout.max_pred) {
            return Err(DoorbellError::TooManyPreds {
                count: self.pred.len(),
                max: layout.max_pred,
            });
        }
        Ok(())
    }

    /// Writes the layer into `buf`, at least `layout.size()` bytes. Entries past the counts are
    /// left as they were: no site reads them. A prediction's score is written bit for bit, NaN
    /// payloads included.
    ///
    /// # Errors
    ///
    /// [`DoorbellError`]; `buf` is unchanged on error.
    pub fn write(&self, layout: &DoorbellLayout, buf: &mut [u8]) -> Result<(), DoorbellError> {
        layout.check(buf.len())?;
        self.check(layout)?;
        let h = DoorbellHeader {
            seq: self.seq,
            n_route: self.route.len() as u16,
            n_pred: self.pred.len() as u16,
        };
        h.write_to(buf, 0);
        for (i, r) in self.route.iter().enumerate() {
            r.write_to(buf, layout.route_offset() + i * size_of::<RouteSlot>());
        }
        for (i, p) in self.pred.iter().enumerate() {
            p.write_to(buf, layout.pred_offset() + i * size_of::<PredSlot>());
        }
        Ok(())
    }

    /// Reads a layer from `buf`, written under the same `layout`.
    ///
    /// # Errors
    ///
    /// [`DoorbellError::BufferTooSmall`], and the count errors when the header's counts exceed
    /// the layout: they are checked before any entry is read.
    pub fn read(layout: &DoorbellLayout, buf: &[u8]) -> Result<Self, DoorbellError> {
        layout.check(buf.len())?;
        let h = DoorbellHeader::read_from(buf, 0);
        if h.n_route > layout.max_union {
            return Err(DoorbellError::TooManyRoutes {
                count: usize::from(h.n_route),
                max: layout.max_union,
            });
        }
        if h.n_pred > layout.max_pred {
            return Err(DoorbellError::TooManyPreds {
                count: usize::from(h.n_pred),
                max: layout.max_pred,
            });
        }
        let route = (0..usize::from(h.n_route))
            .map(|i| RouteSlot::read_from(buf, layout.route_offset() + i * size_of::<RouteSlot>()))
            .collect();
        let pred = (0..usize::from(h.n_pred))
            .map(|i| PredSlot::read_from(buf, layout.pred_offset() + i * size_of::<PredSlot>()))
            .collect();
        Ok(Self {
            seq: h.seq,
            route,
            pred,
        })
    }
}

/// The members `write` and `read` move, per mirror in header order, for the test that checks
/// they are every member of the generated mirrors.
#[cfg(test)]
pub(super) fn serialized_fields() -> [Vec<&'static str>; 3] {
    let mut out = [Vec::new(), Vec::new(), Vec::new()];
    macro_rules! name {
        (DoorbellHeader, $f:ident, $w:tt) => {
            out[0].push(stringify!($f))
        };
        (RouteSlot, $f:ident, $w:tt) => {
            out[1].push(stringify!($f))
        };
        (PredSlot, $f:ident, $w:tt) => {
            out[2].push(stringify!($f))
        };
    }
    header_fields!(name);
    route_fields!(name);
    pred_fields!(name);
    out
}
