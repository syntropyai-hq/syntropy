// SPDX-License-Identifier: MIT OR Apache-2.0

//! Frames (RFC-0001 §3.5): a 4-byte little-endian length, then that many bytes of postcard, which
//! encode one [`Envelope`].

use std::fmt;

use postcard::de_flavors::{Flavor, Slice};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use super::msg::Envelope;

/// The bytes of a frame's length prefix.
pub const HEADER: usize = 4;

/// The most bytes a frame holds after its length prefix: 64 MiB `[target]`, the control plane's
/// largest message (RFC-0031 §9.1), so a plan that crosses the network fits the channel too.
pub const MAX_FRAME: usize = 64 << 20;

/// Why input did not decode as a frame. [`decode_frame`] returns one of these for every input
/// that is not a frame: it never panics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrameError {
    /// The input ends inside the frame. `needed` more bytes would complete its length prefix or
    /// its payload.
    Incomplete {
        /// The bytes missing.
        needed: usize,
    },
    /// The length prefix is over [`MAX_FRAME`].
    TooLarge {
        /// The length the prefix gives.
        len: u32,
    },
    /// The payload is not an envelope of the expected message type: postcard's reason.
    Malformed(postcard::Error),
    /// The envelope ends before the payload does.
    Trailing {
        /// The payload bytes after the envelope.
        unread: usize,
    },
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FrameError::Incomplete { needed } => {
                write!(f, "the frame is incomplete: {needed} more bytes needed")
            }
            FrameError::TooLarge { len } => write!(
                f,
                "the frame declares {len} bytes, over the limit of {MAX_FRAME}"
            ),
            FrameError::Malformed(error) => write!(f, "the frame's message is malformed: {error}"),
            FrameError::Trailing { unread } => {
                write!(f, "the frame has {unread} bytes after its message")
            }
        }
    }
}

impl std::error::Error for FrameError {}

/// Why a message could not be framed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EncodeError {
    /// The message encodes to more than [`MAX_FRAME`] bytes.
    TooLarge {
        /// Its encoded length.
        len: usize,
    },
    /// A field's serializer failed: postcard's reason.
    Serialize(postcard::Error),
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncodeError::TooLarge { len } => write!(
                f,
                "the message encodes to {len} bytes, over the frame limit of {MAX_FRAME}"
            ),
            EncodeError::Serialize(error) => write!(f, "the message did not encode: {error}"),
        }
    }
}

impl std::error::Error for EncodeError {}

/// Encodes `envelope` as one frame.
pub fn encode_frame<M: Serialize>(envelope: &Envelope<M>) -> Result<Vec<u8>, EncodeError> {
    let frame = postcard::to_extend(envelope, vec![0; HEADER]).map_err(EncodeError::Serialize)?;
    let len = frame.len() - HEADER;
    if len > MAX_FRAME {
        return Err(EncodeError::TooLarge { len });
    }
    let mut frame = frame;
    // `len` is at most `MAX_FRAME`, which fits a `u32`.
    frame[..HEADER].copy_from_slice(&(len as u32).to_le_bytes());
    Ok(frame)
}

/// Decodes the frame at the start of `input`: its envelope, and the bytes the frame took.
///
/// The one decoder: [`Channel::recv`](super::Channel::recv) calls it on each frame it reads, and
/// the `ipc_msgs` fuzz target (QUAL-E10-T03) calls it on arbitrary input. It is total: any input
/// gives an envelope or a [`FrameError`]. A declared sequence length reserves nothing before the
/// elements decode (see the private `NoPrealloc`), so a forged length costs no memory.
pub fn decode_frame<M: DeserializeOwned>(input: &[u8]) -> Result<(Envelope<M>, usize), FrameError> {
    let Some(header) = input.first_chunk::<HEADER>() else {
        return Err(FrameError::Incomplete {
            needed: HEADER - input.len(),
        });
    };
    let len = u32::from_le_bytes(*header);
    if len as usize > MAX_FRAME {
        return Err(FrameError::TooLarge { len });
    }
    let end = HEADER + len as usize;
    let Some(payload) = input.get(HEADER..end) else {
        return Err(FrameError::Incomplete {
            needed: end - input.len(),
        });
    };
    let mut deserializer = postcard::Deserializer::from_flavor(NoPrealloc(Slice::new(payload)));
    let envelope = Envelope::<M>::deserialize(&mut deserializer).map_err(FrameError::Malformed)?;
    let rest = deserializer.finalize().map_err(FrameError::Malformed)?;
    if !rest.is_empty() {
        return Err(FrameError::Trailing { unread: rest.len() });
    }
    Ok((envelope, end))
}

/// Postcard's slice flavor, with a size hint of 0.
///
/// Postcard passes a sequence's declared length on as its size hint when the input has at least
/// that many bytes left, and `SmallVec`'s deserializer reserves that many elements up front. An
/// element can take far more memory than the one byte the check allows for it, so a forged length
/// in a 64 MiB frame could reserve gigabytes. With a hint of 0, postcard hints nothing for any
/// non-empty sequence, and every collection grows as its elements decode.
pub(crate) struct NoPrealloc<'de>(pub(crate) Slice<'de>);

impl<'de> Flavor<'de> for NoPrealloc<'de> {
    type Remainder = &'de [u8];
    type Source = &'de [u8];

    fn pop(&mut self) -> postcard::Result<u8> {
        self.0.pop()
    }

    fn size_hint(&self) -> Option<usize> {
        Some(0)
    }

    fn try_take_n(&mut self, ct: usize) -> postcard::Result<&'de [u8]> {
        self.0.try_take_n(ct)
    }

    fn finalize(self) -> postcard::Result<&'de [u8]> {
        self.0.finalize()
    }
}
