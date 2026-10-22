// SPDX-License-Identifier: MIT OR Apache-2.0

//! Routing rows: the per-row expert IDs and weights that `ROUTER_BCAST` broadcasts and that
//! `EpDispatch` sends to an expert owner (RFC-0031 §6.8).
//!
//! Both rows share one layout, every field naturally aligned: a 4-byte head, `k` routing weights
//! as `f32`, then `k` expert IDs as `u16`, padded to 4 bytes. Weights come before expert IDs so
//! the weights stay 4-byte aligned whatever `k` is.
//!
//! ```text
//! ROUTER_BCAST row:  k u8,   pad[3],              k × weight f32, k × expert u16, pad to 4
//! EpDispatch row:    row u16, k_local u8, pad u8, k × weight f32, k × expert u16, pad to 4
//! ```
//!
//! A row read from a frame borrows the frame's bytes, and a row given by a sender borrows its
//! slices; two compare equal when their values do, weights bit for bit.

use std::fmt;

use super::error::FrameError;
use super::header::FrameType;

/// The head before a row's weights.
pub const ROUTE_HEAD_LEN: usize = 4;

/// The most experts one row names: `k` and `k_local` are a `u8`.
pub const MAX_ROUTE_K: usize = u8::MAX as usize;

/// The bytes of a routing row with `k` experts: the head, `k` weights, `k` expert IDs, padded
/// to 4. This is RFC-0031 §6.8's `m_row` for an `EpDispatch` row: 12 B at `k_local = 1`, 52 B
/// at `k_local = 8`.
pub const fn route_row_len(k: usize) -> usize {
    ROUTE_HEAD_LEN + 4 * k + (2 * k).next_multiple_of(4)
}

/// Wire bytes, already checked when the frame was parsed, or a sender's values.
#[derive(Clone, Copy)]
enum Repr<'a, T> {
    Wire(&'a [u8]),
    Typed(&'a [T]),
}

/// A row's routing weights, in the row's expert order.
#[derive(Clone, Copy)]
pub struct Weights<'a>(Repr<'a, f32>);

/// A row's expert IDs, in the row's order.
#[derive(Clone, Copy)]
pub struct Experts<'a>(Repr<'a, u16>);

impl<'a> Weights<'a> {
    pub fn new(weights: &'a [f32]) -> Weights<'a> {
        Weights(Repr::Typed(weights))
    }

    pub fn len(&self) -> usize {
        match self.0 {
            Repr::Wire(bytes) => bytes.len() / 4,
            Repr::Typed(weights) => weights.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> Option<f32> {
        match self.0 {
            Repr::Wire(bytes) => {
                let at = index.checked_mul(4)?;
                let word = bytes.get(at..at.checked_add(4)?)?;
                Some(f32::from_le_bytes(word.try_into().expect("4 bytes")))
            }
            Repr::Typed(weights) => weights.get(index).copied(),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = f32> + 'a {
        let weights = *self;
        (0..weights.len()).filter_map(move |index| weights.get(index))
    }
}

impl<'a> Experts<'a> {
    pub fn new(experts: &'a [u16]) -> Experts<'a> {
        Experts(Repr::Typed(experts))
    }

    pub fn len(&self) -> usize {
        match self.0 {
            Repr::Wire(bytes) => bytes.len() / 2,
            Repr::Typed(experts) => experts.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> Option<u16> {
        match self.0 {
            Repr::Wire(bytes) => {
                let at = index.checked_mul(2)?;
                let half = bytes.get(at..at.checked_add(2)?)?;
                Some(u16::from_le_bytes([half[0], half[1]]))
            }
            Repr::Typed(experts) => experts.get(index).copied(),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = u16> + 'a {
        let experts = *self;
        (0..experts.len()).filter_map(move |index| experts.get(index))
    }
}

/// Weights compare bit for bit, so a NaN payload and the sign of zero survive a round trip.
impl PartialEq for Weights<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .map(f32::to_bits)
                .eq(other.iter().map(f32::to_bits))
    }
}

impl PartialEq for Experts<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl fmt::Debug for Weights<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl fmt::Debug for Experts<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// One row of a `ROUTER_BCAST` payload: the experts the router chose for the row and their
/// weights, `k` of each.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RouterRow<'a> {
    pub weights: Weights<'a>,
    pub experts: Experts<'a>,
}

/// One row's routing in an `EpDispatch`: its index among the step's rows, and the experts of
/// this owner it routes to with their weights, `k_local` of each.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpRow<'a> {
    /// The row's index among the step's rows (token positions).
    pub row: u16,
    pub weights: Weights<'a>,
    pub experts: Experts<'a>,
}

impl RouterRow<'_> {
    /// `k`: the experts the row names.
    pub fn k(&self) -> usize {
        self.experts.len()
    }
}

impl EpRow<'_> {
    /// `k_local`: the experts of this owner the row routes to.
    pub fn k_local(&self) -> usize {
        self.experts.len()
    }
}

/// The two row heads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Head {
    /// `k u8, pad[3]`.
    Router,
    /// `row u16, k_local u8, pad u8`.
    Ep,
}

impl Head {
    fn k(self, head: &[u8]) -> usize {
        match self {
            Head::Router => usize::from(head[0]),
            Head::Ep => usize::from(head[2]),
        }
    }
}

/// Rows read from a frame's bytes, or given by a sender.
#[derive(Clone, Copy)]
enum RowsRepr<'a, T> {
    /// `count` rows that [`walk`] has checked fill `bytes` exactly.
    Wire {
        bytes: &'a [u8],
        count: usize,
    },
    Typed(&'a [T]),
}

/// The rows of a `ROUTER_BCAST` payload, in row order.
#[derive(Clone, Copy)]
pub struct RouterRows<'a>(RowsRepr<'a, RouterRow<'a>>);

/// The routing rows of an `EpDispatch`, one per payload row, in payload order.
#[derive(Clone, Copy)]
pub struct EpRows<'a>(RowsRepr<'a, EpRow<'a>>);

/// The wire row at the start of `bytes`, whose head [`walk`] has checked: its head, weights and
/// expert IDs, and its length.
fn split_row(head: Head, bytes: &[u8]) -> (&[u8], &[u8], &[u8], usize) {
    let k = head.k(bytes);
    let weights = &bytes[ROUTE_HEAD_LEN..ROUTE_HEAD_LEN + 4 * k];
    let experts = &bytes[ROUTE_HEAD_LEN + 4 * k..ROUTE_HEAD_LEN + 6 * k];
    (&bytes[..ROUTE_HEAD_LEN], weights, experts, route_row_len(k))
}

/// Walks `count` rows from the start of `bytes` and returns the bytes they take and the experts
/// they name in all. A row cut short by the end of `bytes` is refused: as a short extension in
/// an `EpDispatch`, whose rows are in its extension, and as a short payload otherwise.
pub(super) fn walk(
    head: Head,
    bytes: &[u8],
    count: usize,
    ftype: FrameType,
) -> Result<(usize, u64), FrameError> {
    let mut at = 0usize;
    let mut experts = 0u64;
    for _ in 0..count {
        let needed = at + ROUTE_HEAD_LEN;
        if bytes.len() < needed {
            return Err(cut_short(ftype, bytes.len(), needed));
        }
        let k = head.k(&bytes[at..]);
        let needed = at + route_row_len(k);
        if bytes.len() < needed {
            return Err(cut_short(ftype, bytes.len(), needed));
        }
        at = needed;
        experts += k as u64;
    }
    Ok((at, experts))
}

fn cut_short(ftype: FrameType, len: usize, needed: usize) -> FrameError {
    match ftype {
        FrameType::EpDispatch => FrameError::ExtTooShort { ftype, len, needed },
        _ => FrameError::Inconsistent {
            ftype,
            field: "payload_len",
            value: len as u64,
            expected: needed as u64,
        },
    }
}

/// Writes one row at the start of `out`, which holds at least [`route_row_len`] of its `k`
/// bytes, and returns that length. The head's padding and the trailing padding are zero. The
/// row has been checked: as many weights as expert IDs.
fn write_row(
    head: [u8; ROUTE_HEAD_LEN],
    weights: Weights<'_>,
    experts: Experts<'_>,
    out: &mut [u8],
) -> usize {
    let k = experts.len();
    let len = route_row_len(k);
    let row = &mut out[..len];
    row.fill(0);
    row[..ROUTE_HEAD_LEN].copy_from_slice(&head);
    for (i, weight) in weights.iter().enumerate() {
        row[ROUTE_HEAD_LEN + 4 * i..][..4].copy_from_slice(&weight.to_le_bytes());
    }
    let at = ROUTE_HEAD_LEN + 4 * k;
    for (i, expert) in experts.iter().enumerate() {
        row[at + 2 * i..][..2].copy_from_slice(&expert.to_le_bytes());
    }
    len
}

/// Checks a sender's row: as many weights as expert IDs, at most 255.
fn check_typed(
    ftype: FrameType,
    weights: Weights<'_>,
    experts: Experts<'_>,
) -> Result<usize, FrameError> {
    let k = experts.len();
    if weights.len() != k {
        return Err(FrameError::Inconsistent {
            ftype,
            field: "weights",
            value: weights.len() as u64,
            expected: k as u64,
        });
    }
    if k > MAX_ROUTE_K {
        return Err(FrameError::BadField {
            ftype,
            field: "k",
            value: k as u64,
        });
    }
    Ok(k)
}

impl<'a> RouterRows<'a> {
    pub fn new(rows: &'a [RouterRow<'a>]) -> RouterRows<'a> {
        RouterRows(RowsRepr::Typed(rows))
    }

    /// `count` rows that [`walk`] has checked fill `bytes` exactly.
    pub(super) fn wire(bytes: &'a [u8], count: usize) -> RouterRows<'a> {
        RouterRows(RowsRepr::Wire { bytes, count })
    }

    pub fn len(&self) -> usize {
        match self.0 {
            RowsRepr::Wire { count, .. } => count,
            RowsRepr::Typed(rows) => rows.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = RouterRow<'a>> + 'a {
        let repr = self.0;
        let mut at = 0;
        let mut index = 0;
        std::iter::from_fn(move || match repr {
            RowsRepr::Wire { bytes, count } if index < count => {
                let (_, weights, experts, len) = split_row(Head::Router, &bytes[at..]);
                at += len;
                index += 1;
                Some(RouterRow {
                    weights: Weights(Repr::Wire(weights)),
                    experts: Experts(Repr::Wire(experts)),
                })
            }
            RowsRepr::Wire { .. } => None,
            RowsRepr::Typed(rows) => {
                let row = rows.get(index).copied();
                index += 1;
                row
            }
        })
    }

    /// Checks a sender's rows and returns the bytes they take and the experts they name.
    pub(super) fn measure(&self, ftype: FrameType) -> Result<(usize, u64), FrameError> {
        match self.0 {
            RowsRepr::Wire { bytes, count } => walk(Head::Router, bytes, count, ftype),
            RowsRepr::Typed(rows) => rows.iter().try_fold((0, 0), |(len, sum), row| {
                let k = check_typed(ftype, row.weights, row.experts)?;
                Ok((len + route_row_len(k), sum + k as u64))
            }),
        }
    }

    /// Writes the rows at the start of `out`, which [`RouterRows::measure`] has sized, and
    /// returns their length.
    pub(super) fn write(&self, out: &mut [u8]) -> usize {
        let mut at = 0;
        for row in self.iter() {
            let head = [row.k() as u8, 0, 0, 0];
            at += write_row(head, row.weights, row.experts, &mut out[at..]);
        }
        at
    }
}

impl<'a> EpRows<'a> {
    pub fn new(rows: &'a [EpRow<'a>]) -> EpRows<'a> {
        EpRows(RowsRepr::Typed(rows))
    }

    /// `count` rows that [`walk`] has checked start `bytes`.
    pub(super) fn wire(bytes: &'a [u8], count: usize) -> EpRows<'a> {
        EpRows(RowsRepr::Wire { bytes, count })
    }

    pub fn len(&self) -> usize {
        match self.0 {
            RowsRepr::Wire { count, .. } => count,
            RowsRepr::Typed(rows) => rows.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = EpRow<'a>> + 'a {
        let repr = self.0;
        let mut at = 0;
        let mut index = 0;
        std::iter::from_fn(move || match repr {
            RowsRepr::Wire { bytes, count } if index < count => {
                let (head, weights, experts, len) = split_row(Head::Ep, &bytes[at..]);
                at += len;
                index += 1;
                Some(EpRow {
                    row: u16::from_le_bytes([head[0], head[1]]),
                    weights: Weights(Repr::Wire(weights)),
                    experts: Experts(Repr::Wire(experts)),
                })
            }
            RowsRepr::Wire { .. } => None,
            RowsRepr::Typed(rows) => {
                let row = rows.get(index).copied();
                index += 1;
                row
            }
        })
    }

    /// The bytes rows `start..end` take on the wire. `end` is at most [`EpRows::len`].
    pub(super) fn span_len(&self, start: usize, end: usize) -> usize {
        self.iter()
            .skip(start)
            .take(end - start)
            .map(|row| route_row_len(row.k_local()))
            .sum()
    }

    /// Checks every row: `k_local` from 1 (a row goes to an owner only for an expert it holds),
    /// and for a sender's rows as many weights as expert IDs, at most 255.
    pub(super) fn check(&self) -> Result<(), FrameError> {
        for row in self.iter() {
            let k = check_typed(FrameType::EpDispatch, row.weights, row.experts)?;
            if k == 0 {
                return Err(FrameError::BadField {
                    ftype: FrameType::EpDispatch,
                    field: "k_local",
                    value: 0,
                });
            }
        }
        Ok(())
    }

    /// Writes rows `start..end` at the start of `out`, which [`EpRows::span_len`] has sized, and
    /// returns their length.
    pub(super) fn write(&self, start: usize, end: usize, out: &mut [u8]) -> usize {
        let mut at = 0;
        for row in self.iter().skip(start).take(end - start) {
            let [lo, hi] = row.row.to_le_bytes();
            let head = [lo, hi, row.k_local() as u8, 0];
            at += write_row(head, row.weights, row.experts, &mut out[at..]);
        }
        at
    }
}

impl PartialEq for RouterRows<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl PartialEq for EpRows<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl fmt::Debug for RouterRows<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl fmt::Debug for EpRows<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_lengths_follow_rfc_0031_section_6_8() {
        // m_row at k_local = 1 and 8, from the §6.8 derivation.
        assert_eq!(route_row_len(1), 12);
        assert_eq!(route_row_len(8), 52);
        assert_eq!(route_row_len(0), 4);
        assert_eq!(route_row_len(3), 4 + 12 + 8);
        for k in 0..=MAX_ROUTE_K {
            assert_eq!(route_row_len(k) % 4, 0, "k = {k}");
        }
    }

    #[test]
    fn rows_written_are_read_back() {
        let w1 = [0.5f32, -0.25, 1.0];
        let e1 = [7u16, 300, 65535];
        let w2 = [f32::from_bits(0x7FC0_0001)];
        let e2 = [2u16];
        let rows = [
            EpRow {
                row: 3,
                weights: Weights::new(&w1),
                experts: Experts::new(&e1),
            },
            EpRow {
                row: 511,
                weights: Weights::new(&w2),
                experts: Experts::new(&e2),
            },
        ];
        let typed = EpRows::new(&rows);
        typed.check().unwrap();
        let len = typed.span_len(0, 2);
        assert_eq!(len, route_row_len(3) + route_row_len(1));
        let mut out = vec![0xAA; len];
        assert_eq!(typed.write(0, 2, &mut out), len);
        let (walked, experts) = walk(Head::Ep, &out, 2, FrameType::EpDispatch).unwrap();
        assert_eq!((walked, experts), (len, 4));
        assert_eq!(EpRows::wire(&out, 2), typed);
    }
}
