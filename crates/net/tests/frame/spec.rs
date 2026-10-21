// SPDX-License-Identifier: MIT OR Apache-2.0

//! An encoder written from RFC-0031's tables, independent of `syntropy_net::frame`: every field
//! is put at the offset §6.1, §6.2, §6.7 and §6.9 give it, and CRC32C is computed bit by bit. The
//! golden frames are generated with it, so the codec is checked against the tables, not
//! against itself.

/// CRC-32C (Castagnoli): reflected polynomial 0x82F63B78, initial value and final XOR all ones.
pub fn crc32c(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for byte in data {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0x82F6_3B78
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Writes `value` little-endian into `len` bytes at `at`.
pub fn put(buf: &mut [u8], at: usize, len: usize, value: u64) {
    buf[at..at + len].copy_from_slice(&value.to_le_bytes()[..len]);
}

/// `len` deterministic bytes: `seed + 31·i`, modulo 256. Stands in for credentials, ciphertext
/// and payload rows.
pub fn fill(seed: u8, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| seed.wrapping_add((i as u8).wrapping_mul(31)))
        .collect()
}

/// The base header fields a sender sets (RFC-0031 §6.1).
pub struct Base {
    pub ftype: u8,
    pub flags: u16,
    pub instance: u64,
    pub term: u64,
    pub seq: u64,
    pub step: u64,
    pub chan_seq: u64,
}

/// A whole frame: the 64-byte header, `ext` (already a multiple of 64 bytes) and `payload`.
pub fn frame(base: &Base, ext: &[u8], payload: &[u8]) -> Vec<u8> {
    assert_eq!(ext.len() % 64, 0, "the extension is padded by its builder");
    let mut header = vec![0u8; 64];
    header[0..4].copy_from_slice(b"SYNF"); // 0  magic
    header[4] = 1; //                          4  wire_minor
    header[5] = base.ftype; //                 5  ftype
    put(&mut header, 6, 2, base.flags.into()); // 6 flags
    put(&mut header, 8, 8, base.instance); // 8  instance
    put(&mut header, 16, 8, base.term); //    16 epoch_term
    put(&mut header, 24, 8, base.seq); //     24 epoch_seq
    put(&mut header, 32, 8, base.step); //    32 step
    put(&mut header, 40, 8, base.chan_seq); // 40 chan_seq
    put(&mut header, 48, 4, ext.len() as u64); // 48 ext_len
    put(&mut header, 52, 4, payload.len() as u64); // 52 payload_len
    let payload_crc = if payload.is_empty() {
        0
    } else {
        crc32c(payload)
    };
    put(&mut header, 56, 4, payload_crc.into()); // 56 payload_crc
    let mut covered = header[..60].to_vec();
    covered.extend_from_slice(ext);
    put(&mut header, 60, 4, crc32c(&covered).into()); // 60 header_crc
    let mut out = header;
    out.extend_from_slice(ext);
    out.extend_from_slice(payload);
    out
}

/// A zeroed extension of `len` bytes rounded up to a multiple of 64.
pub fn ext(len: usize) -> Vec<u8> {
    vec![0; len.next_multiple_of(64)]
}

/// The `Hello` extension of RFC-0031 §6.9; `resume_from` at offset 64 on a reconnect.
pub struct HelloExt {
    pub purpose: u8,
    pub wire_minor_max: u8,
    pub proto_major: u16,
    pub capabilities: u64,
    pub instance: u64,
    pub term: u64,
    pub seq: u64,
    pub from_stage: u16,
    pub to_stage: u16,
    pub lane: u8,
    pub wire_dtype: u8,
    pub slot_size: u32,
    pub recv_slots: u16,
    pub cred_kind: u16,
    pub channel_token: u64,
    pub resume_from: Option<u64>,
}

pub fn hello_ext(h: &HelloExt) -> Vec<u8> {
    let mut e = ext(if h.resume_from.is_some() { 72 } else { 64 });
    e[0] = h.purpose; //                        0  purpose
    e[1] = h.wire_minor_max; //                 1  wire_minor_max
    put(&mut e, 2, 2, h.proto_major.into()); // 2  proto_major; 4..8 reserved
    put(&mut e, 8, 8, h.capabilities); //       8  capabilities
    put(&mut e, 16, 8, h.instance); //         16 instance
    put(&mut e, 24, 8, h.term); //             24 epoch term
    put(&mut e, 32, 8, h.seq); //              32 epoch seq
    put(&mut e, 40, 2, h.from_stage.into()); // 40 from_stage
    put(&mut e, 42, 2, h.to_stage.into()); //  42 to_stage
    e[44] = h.lane; //                         44 lane
    e[45] = h.wire_dtype; //                   45 wire_dtype; 46..48 reserved
    put(&mut e, 48, 4, h.slot_size.into()); // 48 slot_size
    put(&mut e, 52, 2, h.recv_slots.into()); // 52 recv_slots
    put(&mut e, 54, 2, h.cred_kind.into()); // 54 cred_kind
    put(&mut e, 56, 8, h.channel_token); //    56 channel_token
    if let Some(resume_from) = h.resume_from {
        put(&mut e, 64, 8, resume_from); //    64 resume_from (reconnect)
    }
    e
}

/// The `HelloAck` extension: 32 bytes of fields, padded to 64.
#[allow(clippy::too_many_arguments)]
pub fn hello_ack_ext(
    status: u16,
    wire_minor: u8,
    capabilities: u64,
    limit_seq: u64,
    slot_size: u32,
    recv_slots: u16,
    cred_kind: u16,
) -> Vec<u8> {
    let mut e = ext(32);
    put(&mut e, 0, 2, status.into()); //        0  status
    e[2] = wire_minor; //                       2  wire_minor; 3..8 reserved
    put(&mut e, 8, 8, capabilities); //         8  capabilities
    put(&mut e, 16, 8, limit_seq); //          16 limit_seq
    put(&mut e, 24, 4, slot_size.into()); //   24 slot_size
    put(&mut e, 28, 2, recv_slots.into()); //  28 recv_slots
    put(&mut e, 30, 2, cred_kind.into()); //   30 cred_kind
    e
}

/// The `Close` extension: `code`, reserved, `last_seq` at offset 8.
pub fn close_ext(code: u16, last_seq: u64) -> Vec<u8> {
    let mut e = ext(16);
    put(&mut e, 0, 2, code.into()); //          0  code; 2..8 reserved
    put(&mut e, 8, 8, last_seq); //             8  last_seq
    e
}

/// The `Credit` extension of RFC-0031 §6.7.
pub fn credit_ext(
    ack_seq: u64,
    limit_seq: u64,
    echo_seq: u64,
    echo_delay: u32,
    flags: u16,
    nacks: &[u64],
) -> Vec<u8> {
    let mut e = ext(32 + 8 * nacks.len());
    put(&mut e, 0, 8, ack_seq); //              0  ack_seq
    put(&mut e, 8, 8, limit_seq); //            8  limit_seq
    put(&mut e, 16, 8, echo_seq); //           16 echo_seq
    put(&mut e, 24, 4, echo_delay.into()); //  24 echo_delay
    put(&mut e, 28, 2, nacks.len() as u64); // 28 n_nack
    put(&mut e, 30, 2, flags.into()); //       30 flags
    for (i, seq) in nacks.iter().enumerate() {
        put(&mut e, 32 + 8 * i, 8, *seq); //   32 nack list
    }
    e
}

/// The `Ping` extension: `origin_ns u64, reply u8`, padding.
pub fn ping_ext(origin_ns: u64, reply: u8) -> Vec<u8> {
    let mut e = ext(16);
    put(&mut e, 0, 8, origin_ns); //            0  origin_ns
    e[8] = reply; //                            8  reply
    e
}

/// A sealed frame's extension: the plaintext extension's first 16 bytes, then stand-in
/// ciphertext for the rest (RFC-0031 §6.1).
pub fn sealed_ext(plaintext: &[u8], seed: u8) -> Vec<u8> {
    let mut e = plaintext[..16].to_vec();
    e.extend(fill(seed, plaintext.len() - 16));
    e
}

/// Writes `bytes` as the golden file text: a header comment, then 16 bytes per line in hex,
/// each line with its offset and the part of the frame it starts in.
pub fn to_hex(name: &str, about: &str, bytes: &[u8]) -> String {
    let ext_len = u32::from_le_bytes(bytes[48..52].try_into().unwrap()) as usize;
    let mut out = format!(
        "# {name}: {about}\n\
         # Generated from the RFC-0031 §6 tables by crates/net/tests/frame/spec.rs.\n\
         # A change to this file needs a wire-minor bump (ADR-0307 item 1).\n"
    );
    for (row, chunk) in bytes.chunks(16).enumerate() {
        let at = row * 16;
        let part = if at < 64 {
            "header"
        } else if at < 64 + ext_len {
            "extension"
        } else {
            "payload"
        };
        let hex: Vec<String> = chunk.iter().map(|b| format!("{b:02x}")).collect();
        out.push_str(&format!("{:<47}  # {at:04x} {part}\n", hex.join(" ")));
    }
    out
}

/// Reads golden file text back into bytes: hex pairs, with `#` comments and whitespace ignored,
/// so a CRLF checkout reads the same.
pub fn from_hex(text: &str) -> Vec<u8> {
    let digits: String = text
        .lines()
        .map(|line| line.split('#').next().unwrap_or(""))
        .flat_map(|line| line.chars().filter(|c| !c.is_whitespace()))
        .collect();
    assert_eq!(digits.len() % 2, 0, "an odd number of hex digits");
    (0..digits.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&digits[i..i + 2], 16).expect("hex"))
        .collect()
}
