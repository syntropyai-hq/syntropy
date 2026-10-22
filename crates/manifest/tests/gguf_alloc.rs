// SPDX-License-Identifier: MIT OR Apache-2.0

//! The GGUF reader's peak heap use stays within 4× its input (COMP-E10-T02; RFC-0060's planned
//! bound for parsers on hostile input).
//!
//! The reader keeps the metadata and tensor-info bytes as the file has them, copied once, with
//! one 4-byte position per entry and a 4-byte sort index, and decodes on access. Its peak is the
//! header-read prefix (at most the input) plus those copies (at most the input) plus 8 bytes per
//! entry, against at least 13 input bytes per key-value pair and 24 per tensor entry. Each test
//! below parses the shape that stresses one path, with every entry as small as the format
//! allows, and measures the peak with `syntropy-testkit`'s counting allocator.

use syntropy_manifest::RangeError;
use syntropy_manifest::gguf::{GgufHeader, Limits, ValueRef, ValueType};
use syntropy_manifest::range::{HeaderReader, RangeSource, check_bounds};
use syntropy_testkit::guards::{
    CountingAlloc, alloc_count_this_thread, live_bytes_this_thread, peak_bytes_this_thread,
    reset_peak_bytes_this_thread,
};

#[global_allocator]
static ALLOCATOR: CountingAlloc = CountingAlloc;

struct Mem(Vec<u8>);

impl RangeSource for Mem {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), RangeError> {
        check_bounds(offset, buf.len() as u64, self.len())?;
        let at = offset as usize;
        buf.copy_from_slice(&self.0[at..at + buf.len()]);
        Ok(())
    }
}

fn preamble(n_tensors: u64, n_kv: u64) -> Vec<u8> {
    let mut b = b"GGUF".to_vec();
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&n_tensors.to_le_bytes());
    b.extend_from_slice(&n_kv.to_le_bytes());
    b
}

fn put_str(b: &mut Vec<u8>, s: &[u8]) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s);
}

/// Parses `file` and returns the header, the peak heap bytes during the parse, and the number of
/// allocations. The reader is made before the window and allocates nothing until it reads.
fn measure(file: Vec<u8>) -> (GgufHeader, u64, u64) {
    let mut reader = HeaderReader::new(Mem(file), Limits::DEFAULT.header_bytes);
    let base = live_bytes_this_thread();
    let count = alloc_count_this_thread();
    reset_peak_bytes_this_thread();
    let h = GgufHeader::read_from(&mut reader, &Limits::DEFAULT).unwrap();
    let peak = u64::try_from(peak_bytes_this_thread() - base).unwrap();
    (h, peak, alloc_count_this_thread() - count)
}

/// Three ASCII bytes, distinct for each `i` below 2²¹.
fn ascii3(i: u64) -> [u8; 3] {
    [
        (i >> 14 & 0x7f) as u8,
        (i >> 7 & 0x7f) as u8,
        (i & 0x7f) as u8,
    ]
}

fn assert_within_4x(what: &str, peak: u64, input: u64) {
    let ratio = peak as f64 / input as f64;
    eprintln!("{what}: peak {peak} bytes for {input} input bytes ({ratio:.2}x)");
    assert!(
        peak <= 4 * input,
        "{what}: peak {peak} bytes is more than 4x the {input} input bytes ({ratio:.2}x)"
    );
}

#[test]
fn a_vocabulary_of_one_byte_strings_stays_within_4x() {
    let n: u64 = 1 << 20;
    let mut b = preamble(0, 1);
    put_str(&mut b, b"tokenizer.ggml.tokens");
    b.extend_from_slice(&ValueType::Array.id().to_le_bytes());
    b.extend_from_slice(&ValueType::Str.id().to_le_bytes());
    b.extend_from_slice(&n.to_le_bytes());
    for i in 0..n {
        put_str(&mut b, &[(i % 251) as u8]);
    }
    let input = b.len() as u64;

    let (h, peak, allocations) = measure(b);
    assert_within_4x("2^20 one-byte strings", peak, input);
    // Nothing is allocated per string.
    assert!(
        allocations < 64,
        "{allocations} allocations for {n} strings"
    );

    let Some(ValueRef::Array(tokens)) = h.metadata.get("tokenizer.ggml.tokens") else {
        panic!("the vocabulary did not parse as an array");
    };
    assert_eq!(tokens.len() as u64, n);
    let s: Vec<&[u8]> = tokens.strings().unwrap().skip(300).take(1).collect();
    assert_eq!(s, [&[(300 % 251) as u8][..]]);
}

#[test]
fn the_most_key_value_pairs_with_the_smallest_entries_stay_within_4x() {
    // 2¹⁶ pairs, the limit: 3-byte ASCII keys (two ASCII bytes give only 2¹⁴ distinct keys)
    // and u8 values.
    let n: u64 = 1 << 16;
    let mut b = preamble(0, n);
    for i in 0..n {
        put_str(&mut b, &ascii3(i));
        b.extend_from_slice(&ValueType::U8.id().to_le_bytes());
        b.push(i as u8);
    }
    let input = b.len() as u64;

    let (h, peak, allocations) = measure(b);
    assert_within_4x("2^16 key-value pairs", peak, input);
    assert!(allocations < 64, "{allocations} allocations for {n} pairs");
    assert_eq!(h.metadata.len() as u64, n);
    let key = ascii3(40_000);
    assert_eq!(
        h.metadata.u64(std::str::from_utf8(&key).unwrap()).unwrap(),
        Some(u64::from(40_000u64 as u8))
    );
}

#[test]
fn the_most_tensors_with_the_smallest_entries_stay_within_4x() {
    // 2²⁰ tensors, the limit: 3-byte names (the shortest that are distinct and UTF-8), one dimension of 0 elements, so every tensor is at
    // data offset 0 and the file has no data section to inflate the input.
    let n: u64 = 1 << 20;
    let mut b = preamble(n, 0);
    for i in 0..n {
        let name = ascii3(i);
        put_str(&mut b, &name);
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // F32
        b.extend_from_slice(&0u64.to_le_bytes());
    }
    // The empty data section starts at the aligned end of the table.
    b.resize(b.len().next_multiple_of(32), 0);
    let input = b.len() as u64;

    let (h, peak, allocations) = measure(b);
    assert_within_4x("2^20 tensor-info entries", peak, input);
    assert!(
        allocations < 64,
        "{allocations} allocations for {n} tensors"
    );
    assert_eq!(h.tensor_count() as u64, n);
    let t = h.tensor(12_345).unwrap();
    assert_eq!(t.name.as_bytes(), ascii3(12_345));
    assert_eq!((t.ne(), t.bytes), (&[0u64][..], 0));
}
