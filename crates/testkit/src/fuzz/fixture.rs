// SPDX-License-Identifier: MIT OR Apache-2.0

//! A small record format for the fuzz harness's own tests and the `fixture` fuzz target
//! (QUAL-E10-T01). An input is a run of records, each a tag byte, a length byte and that many
//! payload bytes.
//!
//! | Tag | Record |
//! |---|---|
//! | `0x01` | data: the payload is accepted as is |
//! | `0x02` | a declared size: a 4-byte little-endian size of the data that follows the record, which the parser allocates, as a format with a size field does |
//!
//! [`parse`] is the correct parser. It checks a declared size against the bytes that follow
//! before it allocates (RFC-0060 §10.2), so a size larger than them is `Integrity`, and it
//! accepts one declaration per input: a second is `InvalidRequest`. A truncated record is
//! `Integrity`, and an unknown tag is `InvalidRequest`. [`parse_planted`] is the same parser with three
//! planted defects, each set off by its own tag, so a test can show the harness reports each:
//! `0xF0` panics, `0xF1` holds 8 MiB it never declares, and `0xF2` rejects the input as
//! `Internal`.

use std::hint::black_box;

use syntropy_core::ErrorClass;

use super::Accepted;

/// The data record's tag.
pub const DATA: u8 = 0x01;
/// The declared-size record's tag.
pub const DECLARE: u8 = 0x02;
/// The planted panic's tag.
pub const PLANTED_PANIC: u8 = 0xF0;
/// The planted over-allocation's tag.
pub const PLANTED_OVER_ALLOCATION: u8 = 0xF1;
/// The planted `Internal` rejection's tag.
pub const PLANTED_INTERNAL: u8 = 0xF2;
/// The bytes the planted over-allocation holds.
pub const PLANTED_BYTES: usize = 8 << 20;

/// Parses `input` as the fixture format.
pub fn parse(input: &[u8]) -> Result<Accepted, ErrorClass> {
    parse_with(input, false)
}

/// [`parse`] with the planted defects.
pub fn parse_planted(input: &[u8]) -> Result<Accepted, ErrorClass> {
    parse_with(input, true)
}

fn parse_with(input: &[u8], planted: bool) -> Result<Accepted, ErrorClass> {
    let mut rest = input;
    let mut declared = None;
    // Whatever a declaration allocates stays held until the parse returns, as a real parser's
    // tensor or buffer would.
    let mut held: Vec<Vec<u8>> = Vec::new();
    while let [tag, len, tail @ ..] = rest {
        let len = usize::from(*len);
        let Some((payload, next)) = (tail.len() >= len).then(|| tail.split_at(len)) else {
            return Err(ErrorClass::Integrity);
        };
        match *tag {
            DATA => {}
            DECLARE => {
                if declared.is_some() {
                    return Err(ErrorClass::InvalidRequest);
                }
                let size: [u8; 4] = payload.try_into().map_err(|_| ErrorClass::Integrity)?;
                let size = u64::from(u32::from_le_bytes(size));
                // Checked against what follows before anything is allocated.
                if size > next.len() as u64 {
                    return Err(ErrorClass::Integrity);
                }
                held.push(vec![0u8; size as usize]);
                declared = Some(size);
            }
            PLANTED_PANIC if planted => panic!("planted panic in the fixture parser"),
            PLANTED_OVER_ALLOCATION if planted => held.push(vec![1u8; PLANTED_BYTES]),
            PLANTED_INTERNAL if planted => return Err(ErrorClass::Internal),
            _ => return Err(ErrorClass::InvalidRequest),
        }
        rest = next;
    }
    if !rest.is_empty() {
        // A lone tag byte with no length.
        return Err(ErrorClass::Integrity);
    }
    black_box(&held);
    Ok(Accepted { declared })
}
