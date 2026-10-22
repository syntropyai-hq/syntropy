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
//! The frames of a tensor- or expert-parallel island (§6.8, DIST-E8-T08) are read and written
//! here too: a `StepDescriptor` is an [`Activation`] under its own frame type
//! ([`Frame::step_descriptor`], [`Frame::forward_as_step_descriptor`]); [`Collective`] carries
//! partial sums, gathered slices and `ROUTER_BCAST`'s [`RouterRows`]; [`EpDispatch`] and
//! [`EpCombine`] carry an expert owner's rows and their weighted sums, split into parts by a
//! row limit that counts the extension.
//!
//! [`EmbeddingResult`] reads and writes the `EmbeddingResult` frame (§6.14), the pooled vector of
//! an embedding sequence, splits a vector larger than `max_frame` into parts and [`assemble`]s
//! them, and is written only on a channel that negotiated [`capability::EMBEDDING`].
//!
//! The typed codecs of the other frame types (`SeqControl`, `KvChunk`, …) are later tasks; at this
//! layer their extension and payload are bytes. So are a sealed frame's: the AEAD layer
//! (ADR-0314) seals and opens them, and this layer carries the result, `ctl_seq` in `step` and
//! the `KEY_PHASE` flag included.

mod activation;
mod collective;
mod control;
mod crc;
mod embedding;
mod ep;
mod error;
mod header;
mod route;
mod token;

pub use activation::{
    ACTIVATION_FIXED_LEN, Activation, ActivationHeader, ActivationKind, ActivationSpec, Part,
    Parts, RowLayout, SPAN_LEN, SeqCtrl, Side, SideItem, SideItems, Span, SpanTree, Spans,
    TLV_HEADER_LEN, TREE_NODE_LEN, TraceContext, TreeArea, TreeNode, tlv,
};
pub use collective::{
    COLLECTIVE_FIELDS_LEN, Collective, CollectiveHeader, CollectivePayload, CollectivePhase,
    CollectiveSpec, WireAlgo,
};
pub use control::{
    Close, Control, ControlSpec, CredKind, Credit, Hello, HelloAck, MAX_CONTROL_EXT, MAX_NACKS,
    MAX_REASON, Nacks, Ping, Purpose, code,
};
pub use crc::{crc32c, crc32c_append};
pub use embedding::{
    EMBEDDING_F32, EMBEDDING_FIXED_LEN, EmbeddingParts, EmbeddingResult, EmbeddingSpec, Pooling,
    assemble, capability,
};
pub use ep::{EP_FIXED_LEN, EpCombine, EpDispatch, EpHeader, EpParts};
pub use error::FrameError;
pub use header::{
    AEAD_TAG_LEN, ALIGN, Flags, Frame, FrameSpec, FrameType, HEADER_LEN, Header, MAGIC,
    SEALED_EXT_PLAINTEXT, WIRE_MINOR, encode, encode_assembled, encode_header, encode_in_place,
    encode_to_vec, encoded_len,
};
pub use route::{
    EpRow, EpRows, Experts, MAX_ROUTE_K, ROUTE_HEAD_LEN, RouterRow, RouterRows, Weights,
    route_row_len,
};
pub use token::{
    ENTRY_HEADER_LEN, Entries, Route, StepTokens, TOKEN_FIXED_LEN, Token, TokenCopy,
    TokenEntryBytes, TokenEntryView, TokenSpec, token_flags,
};
