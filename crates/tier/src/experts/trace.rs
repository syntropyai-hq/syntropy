// SPDX-License-Identifier: MIT OR Apache-2.0

//! Routing traces: the format, the recorder and the reader (RFC-0060 §12.4, revised
//! 2026-10-21).
//!
//! A trace holds one record per pass and MoE layer, in step order: the pass's token count and
//! its distinct experts, each with the number of the pass's tokens that selected it (the
//! `expert` and `tokens` of RFC-0023 §2's `RouteSlot`). The recorder takes `RouteSlot` lists,
//! whatever produced them: the statistics ring ([`super::stats`]), a simulator or a test. The
//! format and its reader live here, with the recorder, because no production crate depends on
//! `testkit` (RFC-0001 §4.2 rule 9); the trace-driven cache simulator and the recording runs
//! read traces through [`TraceReader`].
//!
//! # Format, version 1
//!
//! Every integer is little-endian.
//!
//! ```text
//! header   magic "SYNRTRC\0" (8 bytes) · version: u32 = 1 · reserved: u32 = 0
//! record   step: u64 · layer: u16 · n_route: u16 · tokens: u32
//!          n_route × (expert: u16 · tokens: u16)
//! ```
//!
//! A record is 16 bytes plus 4 bytes per routed expert. A one-token pass of a top-8 layer is
//! 48 bytes, so 61 such layers take 2,928 bytes per step [estimate: 61 × (16 + 8 × 4)].
//! Records follow each other to the end of the file, with `step` never decreasing.
//!
//! **What a trace leaves out.** A `RouteSlot`'s `cache_slot` belongs to the run that recorded
//! it, so the trace does not keep it, and the reader returns `MOE_MISS` there. The doorbell's
//! `seq` is not kept either: the recorder refuses a record whose `seq` is not the low 32 bits
//! of its step, so the reader returns exactly those bits. Predicted experts join the record in
//! a later version of the format (TIER-E10-T18).

#![forbid(unsafe_code)]

use std::fmt;
use std::io::{self, Read, Write};

use syntropy_core::LayerIx;
use syntropy_kernels::moe::RouteSlot;
use syntropy_kernels::moe::mirror::MOE_MISS;

use super::stats::RouteRecord;

/// The first 8 bytes of every routing trace.
pub const MAGIC: [u8; 8] = *b"SYNRTRC\0";

/// The format version this module writes and reads.
pub const VERSION: u32 = 1;

const HEADER_LEN: usize = 16;
const RECORD_HEAD_LEN: usize = 16;
const ENTRY_LEN: usize = 4;

/// Why a trace could not be written or read.
#[derive(Debug)]
pub enum TraceError {
    /// The underlying reader or writer failed.
    Io(io::Error),
    /// The input does not start with [`MAGIC`].
    BadMagic,
    /// A version this reader does not know.
    UnsupportedVersion(u32),
    /// The input ends inside the header or a record.
    Truncated,
    /// A record's step is below the one before it.
    OutOfOrder {
        /// The previous record's step.
        previous: u64,
        /// This record's step.
        step: u64,
    },
    /// A record's `seq` is not the low 32 bits of its step.
    SeqMismatch {
        /// The record's step.
        step: u64,
        /// The doorbell's `seq`.
        seq: u32,
    },
    /// More route entries than a record holds (`n_route` is a `u16`).
    TooManyRoutes(usize),
}

impl fmt::Display for TraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "routing trace I/O: {e}"),
            Self::BadMagic => f.write_str("not a routing trace"),
            Self::UnsupportedVersion(v) => {
                write!(f, "routing trace version {v}; this reader knows {VERSION}")
            }
            Self::Truncated => f.write_str("routing trace ends inside a record"),
            Self::OutOfOrder { previous, step } => {
                write!(f, "routing trace step {step} after step {previous}")
            }
            Self::SeqMismatch { step, seq } => {
                write!(f, "doorbell seq {seq} does not match step {step}")
            }
            Self::TooManyRoutes(n) => {
                write!(f, "{n} route entries; a trace record holds at most 65535")
            }
        }
    }
}

impl std::error::Error for TraceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for TraceError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// The recorder: writes a routing trace to `W`. Wrap a file in a `BufWriter`; the recorder
/// writes each record with one call.
#[derive(Debug)]
pub struct TraceWriter<W: Write> {
    out: W,
    last_step: Option<u64>,
    buf: Vec<u8>,
}

impl<W: Write> TraceWriter<W> {
    /// Starts a trace: writes the header.
    ///
    /// # Errors
    ///
    /// [`TraceError::Io`] when the header cannot be written.
    pub fn new(mut out: W) -> Result<Self, TraceError> {
        let mut header = [0u8; HEADER_LEN];
        header[..8].copy_from_slice(&MAGIC);
        header[8..12].copy_from_slice(&VERSION.to_le_bytes());
        out.write_all(&header)?;
        Ok(Self {
            out,
            last_step: None,
            buf: Vec::new(),
        })
    }

    /// Writes one pass's routing for one MoE layer: `tokens` is the pass's token count, and
    /// `route` its distinct experts with their token counts.
    ///
    /// # Errors
    ///
    /// [`TraceError::OutOfOrder`] when `step` is below the last record's,
    /// [`TraceError::TooManyRoutes`] when `route` has more than 65,535 entries, and
    /// [`TraceError::Io`]. Nothing is written then.
    pub fn write_pass(
        &mut self,
        step: u64,
        layer: LayerIx,
        tokens: u32,
        route: &[RouteSlot],
    ) -> Result<(), TraceError> {
        if let Some(previous) = self.last_step
            && step < previous
        {
            return Err(TraceError::OutOfOrder { previous, step });
        }
        let n_route =
            u16::try_from(route.len()).map_err(|_| TraceError::TooManyRoutes(route.len()))?;
        self.buf.clear();
        self.buf.extend_from_slice(&step.to_le_bytes());
        self.buf.extend_from_slice(&layer.0.to_le_bytes());
        self.buf.extend_from_slice(&n_route.to_le_bytes());
        self.buf.extend_from_slice(&tokens.to_le_bytes());
        for slot in route {
            self.buf.extend_from_slice(&slot.expert.to_le_bytes());
            self.buf.extend_from_slice(&slot.tokens.to_le_bytes());
        }
        self.out.write_all(&self.buf)?;
        self.last_step = Some(step);
        Ok(())
    }

    /// Writes a routing record from the statistics ring, after checking its doorbell `seq`
    /// against its step.
    ///
    /// # Errors
    ///
    /// [`TraceError::SeqMismatch`] when `seq` is not the low 32 bits of `step`, and the errors
    /// of [`write_pass`](Self::write_pass).
    pub fn record(&mut self, r: &RouteRecord) -> Result<(), TraceError> {
        if r.seq != r.step as u32 {
            return Err(TraceError::SeqMismatch {
                step: r.step,
                seq: r.seq,
            });
        }
        self.write_pass(r.step, r.layer, r.tokens, &r.route)
    }

    /// Flushes the trace and returns the writer.
    ///
    /// # Errors
    ///
    /// [`TraceError::Io`] when the flush fails.
    pub fn finish(mut self) -> Result<W, TraceError> {
        self.out.flush()?;
        Ok(self.out)
    }
}

/// The reader: reads a routing trace's records in the order they were written, which is step
/// order. Wrap a file in a `BufReader`.
#[derive(Debug)]
pub struct TraceReader<R: Read> {
    input: R,
    last_step: Option<u64>,
    done: bool,
}

impl<R: Read> TraceReader<R> {
    /// Opens a trace: reads and checks the header.
    ///
    /// # Errors
    ///
    /// [`TraceError::BadMagic`], [`TraceError::UnsupportedVersion`], [`TraceError::Truncated`]
    /// or [`TraceError::Io`].
    pub fn new(mut input: R) -> Result<Self, TraceError> {
        let mut header = [0u8; HEADER_LEN];
        if read_full(&mut input, &mut header)? < HEADER_LEN {
            return Err(TraceError::Truncated);
        }
        if header[..8] != MAGIC {
            return Err(TraceError::BadMagic);
        }
        let version = u32::from_le_bytes(header[8..12].try_into().expect("4 bytes"));
        if version != VERSION {
            return Err(TraceError::UnsupportedVersion(version));
        }
        Ok(Self {
            input,
            last_step: None,
            done: false,
        })
    }

    /// The next record, or `None` at the end of the trace. A record's `seq` is the low 32 bits
    /// of its step, and each `RouteSlot`'s `cache_slot` is `MOE_MISS` (the trace keeps
    /// neither).
    ///
    /// # Errors
    ///
    /// [`TraceError::Truncated`] when the input ends inside a record,
    /// [`TraceError::OutOfOrder`] when a step decreases, and [`TraceError::Io`].
    pub fn next_record(&mut self) -> Result<Option<RouteRecord>, TraceError> {
        if self.done {
            return Ok(None);
        }
        let mut head = [0u8; RECORD_HEAD_LEN];
        match read_full(&mut self.input, &mut head)? {
            0 => {
                self.done = true;
                return Ok(None);
            }
            RECORD_HEAD_LEN => {}
            _ => return Err(TraceError::Truncated),
        }
        let step = u64::from_le_bytes(head[0..8].try_into().expect("8 bytes"));
        let layer = u16::from_le_bytes(head[8..10].try_into().expect("2 bytes"));
        let n_route = u16::from_le_bytes(head[10..12].try_into().expect("2 bytes"));
        let tokens = u32::from_le_bytes(head[12..16].try_into().expect("4 bytes"));
        if let Some(previous) = self.last_step
            && step < previous
        {
            return Err(TraceError::OutOfOrder { previous, step });
        }
        let mut entries = vec![0u8; usize::from(n_route) * ENTRY_LEN];
        if read_full(&mut self.input, &mut entries)? < entries.len() {
            return Err(TraceError::Truncated);
        }
        let (entries, _) = entries.as_chunks::<ENTRY_LEN>();
        let route = entries
            .iter()
            .map(|e| RouteSlot {
                expert: u16::from_le_bytes([e[0], e[1]]),
                tokens: u16::from_le_bytes([e[2], e[3]]),
                cache_slot: MOE_MISS,
            })
            .collect();
        self.last_step = Some(step);
        Ok(Some(RouteRecord {
            step,
            seq: step as u32,
            layer: LayerIx(layer),
            tokens,
            route,
        }))
    }
}

impl<R: Read> Iterator for TraceReader<R> {
    type Item = Result<RouteRecord, TraceError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_record() {
            Ok(r) => r.map(Ok),
            Err(e) => {
                // An error ends the trace: what follows a bad record cannot be framed.
                self.done = true;
                Some(Err(e))
            }
        }
    }
}

/// Reads until `buf` is full or the input ends; returns the bytes read.
fn read_full(input: &mut impl Read, buf: &mut [u8]) -> io::Result<usize> {
    let mut n = 0;
    while n < buf.len() {
        match input.read(&mut buf[n..]) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(expert: u16, tokens: u16) -> RouteSlot {
        RouteSlot {
            expert,
            tokens,
            cache_slot: 5,
        }
    }

    fn read_all(bytes: &[u8]) -> Result<Vec<RouteRecord>, TraceError> {
        TraceReader::new(bytes)?.collect()
    }

    #[test]
    fn the_layout_is_as_documented() {
        let mut w = TraceWriter::new(Vec::new()).unwrap();
        w.write_pass(0x0102, LayerIx(3), 2, &[slot(7, 1), slot(9, 2)])
            .unwrap();
        let bytes = w.finish().unwrap();
        let mut want = b"SYNRTRC\0".to_vec();
        want.extend_from_slice(&[1, 0, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[0x02, 0x01, 0, 0, 0, 0, 0, 0]); // step
        want.extend_from_slice(&[3, 0, 2, 0]); // layer, n_route
        want.extend_from_slice(&[2, 0, 0, 0]); // tokens
        want.extend_from_slice(&[7, 0, 1, 0, 9, 0, 2, 0]); // entries
        assert_eq!(bytes, want);
    }

    #[test]
    fn an_empty_trace_reads_as_no_records() {
        let bytes = TraceWriter::new(Vec::new()).unwrap().finish().unwrap();
        assert_eq!(read_all(&bytes).unwrap(), Vec::new());
    }

    #[test]
    fn the_reader_drops_cache_slots_and_derives_seq() {
        let mut w = TraceWriter::new(Vec::new()).unwrap();
        w.write_pass(1 << 33 | 4, LayerIx(0), 1, &[slot(1, 1)])
            .unwrap();
        let r = &read_all(&w.finish().unwrap()).unwrap()[0];
        assert_eq!(r.seq, 4);
        assert_eq!(r.route[0].cache_slot, MOE_MISS);
    }

    #[test]
    fn the_recorder_refuses_bad_records() {
        let mut w = TraceWriter::new(Vec::new()).unwrap();
        w.write_pass(5, LayerIx(0), 1, &[]).unwrap();
        assert!(matches!(
            w.write_pass(4, LayerIx(0), 1, &[]),
            Err(TraceError::OutOfOrder {
                previous: 5,
                step: 4
            })
        ));
        let bad = RouteRecord {
            step: 6,
            seq: 7,
            layer: LayerIx(0),
            tokens: 1,
            route: Vec::new(),
        };
        assert!(matches!(
            w.record(&bad),
            Err(TraceError::SeqMismatch { step: 6, seq: 7 })
        ));
        let long = vec![slot(0, 1); 65_536];
        assert!(matches!(
            w.write_pass(6, LayerIx(0), 1, &long),
            Err(TraceError::TooManyRoutes(65_536))
        ));
        // Only the first record was written.
        assert_eq!(read_all(&w.finish().unwrap()).unwrap().len(), 1);
    }

    #[test]
    fn the_reader_refuses_bad_input() {
        assert!(matches!(read_all(b"SYNRTRC"), Err(TraceError::Truncated)));
        assert!(matches!(
            read_all(b"NOTATRACE\0\0\0\0\0\0\0"),
            Err(TraceError::BadMagic)
        ));
        let mut v2 = MAGIC.to_vec();
        v2.extend_from_slice(&[2, 0, 0, 0, 0, 0, 0, 0]);
        assert!(matches!(
            read_all(&v2),
            Err(TraceError::UnsupportedVersion(2))
        ));

        let mut w = TraceWriter::new(Vec::new()).unwrap();
        w.write_pass(1, LayerIx(0), 1, &[slot(1, 1), slot(2, 1)])
            .unwrap();
        let bytes = w.finish().unwrap();
        // Cut inside the entries, then inside the record's head.
        for cut in [bytes.len() - 1, HEADER_LEN + 3] {
            let got: Vec<_> = TraceReader::new(&bytes[..cut]).unwrap().collect();
            assert!(matches!(got[..], [Err(TraceError::Truncated)]), "cut {cut}");
        }

        // A step that goes back is refused on read as well.
        let mut a = TraceWriter::new(Vec::new()).unwrap();
        a.write_pass(9, LayerIx(0), 1, &[]).unwrap();
        let mut bytes = a.finish().unwrap();
        bytes.extend_from_slice(&[8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]);
        let got: Vec<_> = TraceReader::new(&bytes[..]).unwrap().collect();
        assert!(matches!(
            got[..],
            [
                Ok(_),
                Err(TraceError::OutOfOrder {
                    previous: 9,
                    step: 8
                })
            ]
        ));
    }
}
