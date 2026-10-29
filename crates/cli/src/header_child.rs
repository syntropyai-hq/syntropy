// SPDX-License-Identifier: MIT OR Apache-2.0

//! `syntropy __header`: a newly added model file's header, parsed in a confined child
//! (RFC-0034 §10.4, open question 3; DIST-E17-T16).
//!
//! The daemon spawns this mode for each new file (`syntropy_daemon::header_child`). The child
//! confines itself first ([`syntropy_io::os::confine::confine_self`]: memory, CPU time and open
//! files, and on Linux no new sockets), then opens the file and validates its header with
//! `manifest`'s reader. On success it writes the header's bytes, from the start of the file to
//! the end of the header, to stdout, and exits 0; the daemon parses those vetted bytes. Any
//! refusal exits 1, naming why on stderr. A parser bug, or a header past a limit, costs this
//! process, not the daemon.

use std::io::Write;

use syntropy_io::os::confine::{Limits, confine_self};
use syntropy_manifest::gguf::{GgufHeader, Limits as GgufLimits};
use syntropy_manifest::range::{FileSource, RangeSource};
use syntropy_manifest::safetensors::read_header_default;

use crate::args::{HeaderArgs, HeaderKind};
use crate::error::{CliError, Exit};

/// Runs the header child (module docs).
///
/// # Errors
///
/// [`CliError`] (exit 1) when confinement fails, the file cannot be read, its header is
/// refused, or stdout fails; with `--check-network`, when a connection succeeds.
pub fn run(args: &HeaderArgs) -> Result<(), CliError> {
    let refused = |message: String| CliError::cli(Exit::Failure, "header", message);
    let limits = Limits {
        memory: args.memory,
        cpu_secs: args.cpu_secs,
        open_files: args.open_files,
    };
    let confined =
        confine_self(&limits).map_err(|e| refused(format!("confinement failed: {e}")))?;
    if let Some(addr) = args.check_network {
        return match std::net::TcpStream::connect(addr) {
            Err(e) => {
                eprintln!("the connection was refused, as it must be: {e}");
                Ok(())
            }
            Ok(_) => Err(refused(format!(
                "a confined child connected to {addr} (network banned: {})",
                confined.network
            ))),
        };
    }
    let source = FileSource::open(&args.path)
        .map_err(|e| refused(format!("{}: {e}", args.path.display())))?;
    let header_len = match args.kind {
        HeaderKind::Gguf => {
            GgufHeader::read(&source, &GgufLimits::DEFAULT)
                .map_err(|e| refused(format!("{}: {e}", args.path.display())))?
                .header_bytes
        }
        HeaderKind::Safetensors => {
            read_header_default(&source)
                .map_err(|e| refused(format!("{}: {e}", args.path.display())))?
                .data_start
        }
    };
    let len = usize::try_from(header_len).map_err(|_| refused("the header is too long".into()))?;
    let mut bytes = vec![0u8; len];
    source
        .read_at(0, &mut bytes)
        .map_err(|e| refused(format!("{}: {e}", args.path.display())))?;
    let mut out = std::io::stdout().lock();
    out.write_all(&bytes)
        .and_then(|()| out.flush())
        .map_err(|e| refused(format!("stdout: {e}")))?;
    Ok(())
}
