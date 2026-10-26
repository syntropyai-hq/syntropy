// SPDX-License-Identifier: MIT OR Apache-2.0

//! The QUIC backend on iroh (RFC-0031 §4; ADR-0308; DIST-E7-T03).

pub mod congestion;
pub mod settings;
mod transport;

pub use congestion::{LAN_RTT, LanWindow, LanWindowFactory};
pub use settings::{ConnectionWindow, KEEP_ALIVE, LinkClass, QuicSettings};
pub use transport::{
    DATA_ALPN, INITIAL_ALPHA_P99, MAX_EARLY_PER_PEER, QuicConfig, QuicTransport, Role,
    WindowReport, link_class, stream_priority,
};
