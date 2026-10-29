// SPDX-License-Identifier: MIT OR Apache-2.0

//! The daemon's logging (RFC-0053 §4; PROD-E8-T04).
//!
//! - **Producing.** Async and other non-hot code logs with the `tracing` crate's events, which
//!   [`LogSubscriber`] turns into records. Runners' hot threads write `syntropy_obs::log`'s
//!   fixed `LogRecord`s into their `log` ring, which the daemon formats on receipt with
//!   [`from_record`] and the build's callsite table.
//! - **The pipeline.** [`Logger`] applies `logging.level` and `logging.filter`, removes content
//!   unless `logging.include_content` is on and the record is at `debug` or below (and warns at
//!   start and on every reload while it is on), collapses repeats within 10 s into one record
//!   with `repeated` ([`suppress`]), then writes to every sink. Time comes only from the injected
//!   `Clock` (RFC-0060 §2.1).
//! - **Formats.** Human text, JSON Lines, and journald's structured fields ([`mod@format`]), each
//!   stamped in UTC with microseconds.
//! - **Sinks.** stderr, the rotating file (`<state>/logs/daemon.log`, 16 MiB × 5), journald,
//!   unified logging and the Windows event log, chosen by how the daemon runs ([`sink::plan`],
//!   RFC-0053 §4.2's table). The two OS facilities are linked by `syntropy-io` alone.

mod event;
pub mod format;
mod pipeline;
pub mod sink;
mod subscriber;
pub mod suppress;

pub use event::{CONTENT_PREFIX, Context, Field, LogEvent, Value};
pub use pipeline::{CONTENT_WARNING, Logger, from_record};
pub use subscriber::LogSubscriber;
