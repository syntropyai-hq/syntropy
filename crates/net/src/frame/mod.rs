// SPDX-License-Identifier: MIT OR Apache-2.0

//! The data-plane frame format of RFC-0031 §6 (DIST-E8-T01, DIST-E8-T02).
//!
//! A frame is a 64-byte base header, an extension padded to a multiple of 64 bytes, and a
//! payload:
//!
//! ```text
//! | base header (64 B) | extension (ext_len B) | payload (payload_len B) |
//! ```
//!
//! [`Frame::decode`] checks a frame's integrity and the rules that need no channel state, and
//! [`encode`] writes one, computing both CRC32Cs: `header_crc` over header bytes 0..60 and the
//! extension, and `payload_crc` over the payload. Every byte of a frame is under one of the two,
//! so a corrupted byte is always caught (ADR-0307 item 2). [`Control`] reads and writes the
//! control messages: `Hello`, `HelloAck`, `Close`, `Credit` and `Ping`. [`Activation`] reads and
//! writes the `Activation` frame (§6.3, §6.4), and splits a chunk larger than `max_frame` into
//! parts. [`Token`] reads and writes the `Token` frame (§6.5), and [`StepTokens`] writes a
//! step's ring-return and ingress copies.
//!
//! The typed codecs of the other frame types (`SeqControl`, `KvChunk`, …) are later tasks; at this
//! layer their extension and payload are bytes. So are a sealed frame's: the AEAD layer
//! (ADR-0314) seals and opens them, and this layer carries the result, `ctl_seq` in `step` and
//! the `KEY_PHASE` flag included.

mod activation;
mod control;
mod crc;
mod error;
mod header;
mod token;

pub use activation::{
    ACTIVATION_FIXED_LEN, Activation, ActivationHeader, ActivationKind, ActivationSpec, Part,
    Parts, RowLayout, SPAN_LEN, SeqCtrl, Side, SideItem, SideItems, Span, SpanTree, Spans,
    TLV_HEADER_LEN, TREE_NODE_LEN, TraceContext, TreeArea, TreeNode, tlv,
};
pub use control::{
    Close, Control, ControlSpec, CredKind, Credit, Hello, HelloAck, MAX_CONTROL_EXT, MAX_NACKS,
    MAX_REASON, Nacks, Ping, Purpose, code,
};
pub use crc::{crc32c, crc32c_append};
pub use error::FrameError;
pub use header::{
    AEAD_TAG_LEN, ALIGN, Flags, Frame, FrameSpec, FrameType, HEADER_LEN, Header, MAGIC,
    SEALED_EXT_PLAINTEXT, WIRE_MINOR, encode, encode_assembled, encode_in_place, encode_to_vec,
    encoded_len,
};
pub use token::{
    ENTRY_HEADER_LEN, Entries, Finish, Route, StepTokens, TOKEN_FIXED_LEN, Token, TokenCopy,
    TokenEntry, TokenSpec, token_flags,
};
