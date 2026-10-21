// SPDX-License-Identifier: MIT OR Apache-2.0

//! CRC32C (Castagnoli), the integrity check on every frame header and payload (RFC-0031 §6.1,
//! ADR-0307 item 2).
//!
//! The `crc32c` crate picks the hardware instructions at run time: SSE4.2 `crc32` on x86-64 and
//! the Armv8 CRC extension on aarch64, which RFC-0031 Open question 6 relies on, with a table
//! implementation elsewhere. Every path gives the same value.

/// CRC32C of `data`.
pub fn crc32c(data: &[u8]) -> u32 {
    ::crc32c::crc32c(data)
}

/// CRC32C of the bytes `crc` was computed over, followed by `data`.
pub fn crc32c_append(crc: u32, data: &[u8]) -> u32 {
    ::crc32c::crc32c_append(crc, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The standard check value of CRC-32C (iSCSI) over "123456789".
    #[test]
    fn matches_the_castagnoli_check_value() {
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
        assert_eq!(crc32c(b""), 0);
    }

    #[test]
    fn appending_equals_one_pass() {
        let data: Vec<u8> = (0..=255u8).cycle().take(1000).collect();
        for split in [0, 1, 60, 64, 999, 1000] {
            let (a, b) = data.split_at(split);
            assert_eq!(
                crc32c_append(crc32c(a), b),
                crc32c(&data),
                "split at {split}"
            );
        }
    }
}
