// SPDX-License-Identifier: MIT OR Apache-2.0

//! `syntropy __header`: a newly added model file's header, parsed in a confined child
//! (RFC-0034 §10.4, open question 3; DIST-E17-T16).
//!
//! The daemon spawns this mode for each new file (`syntropy_daemon::header_child`). The child
//! confines itself first and opens the file as the last thing it may open
//! ([`syntropy_io::os::confine::confine_and_open`]), then validates its header with `manifest`'s
//! reader, capped at `--max-header-bytes`. Where the OS enforces no memory limit (macOS), the cap
//! is also at most `--memory` ÷ [`PEAK_COPIES`], so a header too large for the limit is refused
//! before it is read. It reads the
//! header's bytes, from the start of the file to the end of the header, and validates those
//! bytes again on their own ([`syntropy_manifest::range::Prefix`]), so what it sends is what it
//! checked even if the file changed between the reads. It writes the file's length as 8
//! little-endian bytes, then the header's bytes, to stdout, and exits 0; the daemon parses those
//! vetted bytes against that length.
//!
//! Exit codes: 0 sent; 8 (`Integrity`) the header is refused; 5 (`Invalid`) the child could not
//! be confined, which is the platform's failure, not the file's.

use std::io::{Read, Write};

use syntropy_io::os::confine::{Limits, confine_and_open};
use syntropy_manifest::gguf::{GgufHeader, Limits as GgufLimits};
use syntropy_manifest::range::{FileSource, Prefix, RangeSource};
use syntropy_manifest::safetensors::read_header;

use crate::args::{HeaderArgs, HeaderKind};
use crate::error::{CliError, Exit};

/// Copies of the header the child holds at its peak: RFC-0034 §10.4's note gives 4 to 5
/// \[estimate\]. Where the OS enforces no memory limit, the header cap is the limit over this.
pub const PEAK_COPIES: u64 = 5;

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
    let (confined, file) = confine_and_open(&limits, &args.path).map_err(|e| {
        CliError::cli(
            Exit::Invalid,
            "confinement",
            format!("confinement failed: {e}"),
        )
    })?;
    if args.check_crash {
        std::process::abort();
    }
    if args.check_socket {
        return match std::net::UdpSocket::bind("127.0.0.1:0") {
            Err(e) => {
                eprintln!("the socket failed, as it must: {:?}: {e}", e.kind());
                Ok(())
            }
            Ok(_) => Err(refused(format!(
                "a confined child opened a socket (network banned: {})",
                confined.network
            ))),
        };
    }
    if let Some(fd) = args.check_fd {
        let link = format!("/proc/self/fd/{fd}");
        return if std::fs::symlink_metadata(&link).is_err() {
            Ok(())
        } else {
            Err(refused(format!(
                "descriptor {fd} is still open in the child"
            )))
        };
    }
    if let Some(other) = &args.check_remove {
        return match std::fs::remove_file(other) {
            Err(e) => {
                eprintln!("the removal failed, as it must: {:?}: {e}", e.kind());
                Ok(())
            }
            Ok(()) => Err(refused(format!(
                "a confined child removed {} (files banned: {})",
                other.display(),
                confined.files
            ))),
        };
    }
    if let Some(other) = &args.check_open {
        return match std::fs::File::open(other) {
            Err(e) => {
                eprintln!("the open failed, as it must: {:?}: {e}", e.kind());
                Ok(())
            }
            Ok(_) => Err(refused(format!(
                "a confined child opened {} (files banned: {})",
                other.display(),
                confined.files
            ))),
        };
    }
    if args.check_read {
        let mut first = [0u8; 8];
        return (&file)
            .read_exact(&mut first)
            .map_err(|e| refused(format!("the given file could not be read: {e}")));
    }
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
    let source = FileSource::from_file(file).map_err(|e| refused(format!("{path}: {e}")))?;
    let file_len = source.len();
    let max_header_bytes = if confined.memory {
        args.max_header_bytes
    } else {
        args.max_header_bytes.min(args.memory / PEAK_COPIES)
    };
    let gguf_limits = GgufLimits {
        header_bytes: max_header_bytes,
        ..GgufLimits::DEFAULT
    };
    let header_len = match args.kind {
        HeaderKind::Gguf => {
            GgufHeader::read(&source, &gguf_limits)
                .map_err(|e| refused(format!("{path}: {e}")))?
                .header_bytes
        }
        HeaderKind::Safetensors => {
            read_header(&source, max_header_bytes)
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
        HeaderKind::Safetensors => read_header(&prefix, max_header_bytes)
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
