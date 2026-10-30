// SPDX-License-Identifier: MIT OR Apache-2.0

//! `syntropy __header`: a newly added model file's header, parsed in a confined child
//! (RFC-0034 §10.4, open question 3; DIST-E17-T16).
//!
//! The daemon spawns this mode for each new file (`syntropy_daemon::header_child`). The child
//! confines itself first ([`syntropy_io::os::confine::confine_self`]), then opens the file and
//! validates its header with `manifest`'s reader, capped at `--max-header-bytes`. It reads the
//! header's bytes, from the start of the file to the end of the header, and validates those
//! bytes again on their own ([`syntropy_manifest::range::Prefix`]), so what it sends is what it
//! checked even if the file changed between the reads. It writes the file's length as 8
//! little-endian bytes, then the header's bytes, to stdout, and exits 0; the daemon parses those
//! vetted bytes against that length.
//!
//! Exit codes: 0 sent; 8 (`Integrity`) the header is refused; 5 (`Invalid`) the child could not
//! be confined, which is the platform's failure, not the file's.

use std::io::Write;

use syntropy_io::os::confine::{Limits, confine_self};
use syntropy_manifest::gguf::{GgufHeader, Limits as GgufLimits};
use syntropy_manifest::range::{FileSource, Prefix, RangeSource};
use syntropy_manifest::safetensors::read_header;

use crate::args::{HeaderArgs, HeaderKind};
use crate::error::{CliError, Exit};

/// Runs the header child (module docs).
///
/// # Errors
///
/// [`CliError`]: exit 5 when confinement fails, exit 8 when the file cannot be read, its header
/// is refused, or stdout fails; with `--check-network`, exit 8 when a connection succeeds.
pub fn run(args: &HeaderArgs) -> Result<(), CliError> {
    let refused = |message: String| CliError::cli(Exit::Integrity, "header", message);
    let limits = Limits {
        memory: args.memory,
        cpu_secs: args.cpu_secs,
        open_files: args.open_files,
    };
    let confined = confine_self(&limits).map_err(|e| {
        CliError::cli(
            Exit::Invalid,
            "confinement",
            format!("confinement failed: {e}"),
        )
    })?;
    if let Some(addr) = args.check_network {
        return match std::net::TcpStream::connect(addr) {
            Err(e) => {
                eprintln!("the connection failed, as it must: {:?}: {e}", e.kind());
                Ok(())
            }
            Ok(_) => Err(refused(format!(
                "a confined child connected to {addr} (network banned: {})",
                confined.network
            ))),
        };
    }
    let path = args.path.display();
    let source = FileSource::open(&args.path).map_err(|e| refused(format!("{path}: {e}")))?;
    let file_len = source.len();
    let gguf_limits = GgufLimits {
        header_bytes: args.max_header_bytes,
        ..GgufLimits::DEFAULT
    };
    let header_len = match args.kind {
        HeaderKind::Gguf => {
            GgufHeader::read(&source, &gguf_limits)
                .map_err(|e| refused(format!("{path}: {e}")))?
                .header_bytes
        }
        HeaderKind::Safetensors => {
            read_header(&source, args.max_header_bytes)
                .map_err(|e| refused(format!("{path}: {e}")))?
                .data_start
        }
    };
    let len = usize::try_from(header_len).map_err(|_| refused("the header is too long".into()))?;
    let mut bytes = vec![0u8; len];
    source
        .read_at(0, &mut bytes)
        .map_err(|e| refused(format!("{path}: {e}")))?;
    // Validate the bytes about to be sent, on their own: the file may have changed since the
    // first read.
    let prefix = Prefix::new(bytes, file_len);
    let again = match args.kind {
        HeaderKind::Gguf => GgufHeader::read(&prefix, &gguf_limits)
            .map(|h| h.header_bytes)
            .map_err(|e| e.to_string()),
        HeaderKind::Safetensors => read_header(&prefix, args.max_header_bytes)
            .map(|h| h.data_start)
            .map_err(|e| e.to_string()),
    };
    match again {
        Ok(n) if n == header_len => {}
        Ok(n) => {
            return Err(refused(format!(
                "{path} changed while its header was read: {n} bytes, then {header_len}"
            )));
        }
        Err(e) => {
            return Err(refused(format!(
                "{path} changed while its header was read: {e}"
            )));
        }
    }
    let mut out = std::io::stdout().lock();
    out.write_all(&file_len.to_le_bytes())
        .and_then(|()| out.write_all(prefix.bytes()))
        .and_then(|()| out.flush())
        .map_err(|e| refused(format!("stdout: {e}")))?;
    Ok(())
}
