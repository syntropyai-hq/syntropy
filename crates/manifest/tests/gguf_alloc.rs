// SPDX-License-Identifier: MIT OR Apache-2.0

//! The GGUF reader's memory stays proportional to its input (COMP-E10-T02; RFC-0060's planned
//! 4× bound for parsers on hostile input).
//!
//! A vocabulary is an array of up to 2²⁴ short strings. Kept as one allocation per string, it
//! costs several times its bytes in the file. The reader packs string arrays into two exactly
//! sized buffers, so the array costs at most its own input bytes, and the whole parse at most
//! the header-read prefix (1× the input) plus that. This binary counts allocations with
//! `syntropy-testkit`'s counting allocator to check it.

use syntropy_manifest::RangeError;
use syntropy_manifest::gguf::{Array, GgufHeader, Limits, Value, ValueType};
use syntropy_manifest::range::{HeaderReader, RangeSource, check_bounds};
use syntropy_testkit::guards::{CountingAlloc, alloc_count_this_thread};

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

/// A GGUF v3 file whose one key is an array of `n` one-byte strings; returns the file and the
/// array's bytes in it (element type, count and elements).
fn vocab_file(n: u64) -> (Vec<u8>, u64) {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    let key = b"tokenizer.ggml.tokens";
    b.extend_from_slice(&(key.len() as u64).to_le_bytes());
    b.extend_from_slice(key);
    b.extend_from_slice(&ValueType::Array.id().to_le_bytes());
    let array_start = b.len() as u64;
    b.extend_from_slice(&ValueType::Str.id().to_le_bytes());
    b.extend_from_slice(&n.to_le_bytes());
    for i in 0..n {
        b.extend_from_slice(&1u64.to_le_bytes());
        b.push((i % 251) as u8);
    }
    let array_bytes = b.len() as u64 - array_start;
    (b, array_bytes)
}

#[test]
fn a_string_array_costs_at_most_its_input_bytes_in_a_few_allocations() {
    let n = 1 << 20;
    let (file, array_bytes) = vocab_file(n);
    let file_len = file.len() as u64;
    // The whole file in the first request, so the count below is the parser's own.
    let mut reader =
        HeaderReader::new(Mem(file), Limits::DEFAULT.header_bytes).with_initial_read(file_len);

    let before = alloc_count_this_thread();
    let h = GgufHeader::read_from(&mut reader, &Limits::DEFAULT).unwrap();
    let allocations = alloc_count_this_thread() - before;

    let Some(Value::Array(Array::Str(tokens))) = h.metadata.get("tokenizer.ggml.tokens") else {
        panic!("the vocabulary did not parse as a string array");
    };
    assert_eq!(tokens.len() as u64, n);
    assert_eq!(tokens.get(300), Some(&[(300 % 251) as u8][..]));

    // One allocation per string would be over a million; packing takes a handful.
    assert!(
        allocations < 64,
        "{allocations} allocations for {n} strings"
    );
    // The array holds at most its own bytes in the file.
    assert!(
        tokens.heap_bytes() as u64 <= array_bytes,
        "{} heap bytes for {array_bytes} input bytes",
        tokens.heap_bytes()
    );
    // With the header-read prefix (the input itself), the parse holds at most 2× the input,
    // inside the 4× bound.
    assert!(tokens.heap_bytes() as u64 + file_len <= 4 * file_len);
}
