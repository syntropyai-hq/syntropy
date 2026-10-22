// SPDX-License-Identifier: MIT OR Apache-2.0

//! The shared-memory region between the daemon and one runner (RFC-0001 §3.5; ADR-0015 item 4).
//!
//! The daemon [creates](Region::create) the region before it spawns the runner, and passes the
//! runner a [`RegionHandle`] to [open](Region::open) it with. The region holds:
//!
//! | Part | Contents |
//! |---|---|
//! | Header | magic, layout version, ring count, size, the daemon's build ID, `ipc_protocol`; then, each on its own cache line, the runner's heartbeat, the daemon's heartbeat, the step counter and the `RunnerState` discriminant |
//! | Ring table | each ring's role, slot count, slot size and offset |
//! | Rings | one [`syntropy_spsc::shared`] ring per [`RingSpec`], fixed at creation: a ring is never resized |
//!
//! **The segment.** Each OS gives a segment that stays alive while either process holds it:
//!
//! | OS | Segment | Owner-only | Handle |
//! |---|---|---|---|
//! | Linux | `memfd_create`, sealed against growing and shrinking | `fchmod` 0600 | a descriptor only the spawned runner inherits |
//! | Apple platforms | `shm_open` with `O_EXCL` | created 0600 | its name, random, removed as soon as the runner maps the segment |
//! | Windows | a named file mapping in the session's `Local\` namespace | a protected DACL with one entry, for the user | its name, random |
//!
//! A name's random part is 128 bits from the [`Entropy`] the daemon passes to [`Region::create`],
//! so another process can neither guess the name nor take it first. Opening a segment checks it
//! is owner-only before mapping it. [`Region::unlink`] removes the segment's name, so no new
//! process can open it. On Linux a memfd never has a name, and on Windows the name goes when the
//! last handle closes, so there it does nothing. The daemon calls it once the runner has exited.
//!
//! **Sides.** A region is open on one [`Side`]: the daemon's or the runner's. A side takes the
//! producing end of each ring that leaves it and the consuming end of each ring that arrives. It
//! beats its own heartbeat and reads the other's.
//!
//! **Trust.** The two processes run as the same user, and the segment is owner-only, but each
//! treats the other as possibly hostile. [`Region::open`] copies the header's fixed fields and the
//! ring table out once, checks the copies against the layout it expects (magic, version, size,
//! protocol, rings), and keeps only the copies. The header's counters are atomics, and nothing
//! else of the header is ever borrowed. The rings read each index and length once and
//! bounds-check it (see [`syntropy_spsc::shared`]).
//!
//! The record types the rings carry belong to the rings' owners (RFC-0053 for telemetry, steps and
//! logs; RFC-0031 for `TokenFrame`; RFC-0001 §3.5 for the sequence controls). `BuildId`
//! (QUAL-E16-T03) and `RunnerState` (DIST-E1-T05) are given here as their encodings: 32 bytes and
//! a `u32` discriminant.

use std::fmt;
use std::io;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use syntropy_core::Entropy;
use syntropy_spsc::shared::{LayoutError, SharedConsumer, SharedLayout, SharedProducer};

mod os;

/// The first 8 bytes of every region.
pub const MAGIC: [u8; 8] = *b"SYNTRSHM";

/// The version of the region's own layout: the header and the ring table. A change to either
/// changes `ipc_protocol` too (RFC-0001 §3.5).
pub const LAYOUT_VERSION: u32 = 1;

/// The most rings a region holds.
pub const MAX_RINGS: usize = 256;

/// What a ring carries and which way (RFC-0001 §3.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RingRole {
    /// `TelemetryRecord`s, runner to daemon. Its only producer is the drain task (RFC-0053 §1).
    Telemetry,
    /// One dispatcher's `StepRecord`s, runner to daemon (RFC-0053 §2.5).
    Step {
        /// The dispatcher.
        dispatcher: u16,
    },
    /// `TokenFrame`s for sequences whose ingress is this node, tail runner to daemon.
    LocalTokens,
    /// `SequenceOpen` and `SequenceControl` for those sequences, daemon to head runner.
    LocalSequences,
    /// `LogRecord`s, runner to daemon (RFC-0053 §4.1).
    Log,
}

impl RingRole {
    /// Which way the ring's records go.
    #[must_use]
    pub fn direction(self) -> Direction {
        match self {
            RingRole::LocalSequences => Direction::DaemonToRunner,
            _ => Direction::RunnerToDaemon,
        }
    }

    /// The role as the ring table stores it: a kind, and the dispatcher for a step ring.
    fn tag(self) -> (u16, u16) {
        match self {
            RingRole::Telemetry => (1, 0),
            RingRole::Step { dispatcher } => (2, dispatcher),
            RingRole::LocalTokens => (3, 0),
            RingRole::LocalSequences => (4, 0),
            RingRole::Log => (5, 0),
        }
    }
}

/// Which way a ring's records go.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    /// The runner produces, the daemon consumes.
    RunnerToDaemon,
    /// The daemon produces, the runner consumes.
    DaemonToRunner,
}

/// The process a region is open in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    /// The node daemon.
    Daemon,
    /// The runner.
    Runner,
}

impl Side {
    /// Whether this side produces a ring that goes `direction`.
    fn produces(self, direction: Direction) -> bool {
        matches!(
            (self, direction),
            (Side::Runner, Direction::RunnerToDaemon) | (Side::Daemon, Direction::DaemonToRunner)
        )
    }
}

/// One ring of a region: its role, its slot count (a power of two) and the largest record a slot
/// holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RingSpec {
    /// What the ring carries.
    pub role: RingRole,
    /// The number of slots.
    pub slots: u32,
    /// The largest record a slot holds, in bytes.
    pub slot_bytes: u32,
}

/// The rings of a region, which `Init` carries to the runner (RFC-0001 §3.5). Fixed for the
/// region's life.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShmLayout {
    /// The rings, each role at most once.
    pub rings: Vec<RingSpec>,
}

/// The IPC protocol version (RFC-0001 §3.5). Its fingerprint covers the command messages and
/// every ring record schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IpcProtocol {
    /// Changed by an incompatible change.
    pub major: u16,
    /// Changed by a compatible one.
    pub minor: u16,
}

/// Why a layout was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutProblem {
    /// More than [`MAX_RINGS`] rings.
    TooManyRings(usize),
    /// A role appears twice.
    DuplicateRole(RingRole),
    /// A ring's slot count or slot size is out of range.
    Ring {
        /// The ring.
        role: RingRole,
        /// What is wrong with it.
        error: LayoutError,
    },
    /// The region would not fit the address space.
    TooLarge,
}

impl fmt::Display for LayoutProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayoutProblem::TooManyRings(count) => {
                write!(f, "a region holds at most {MAX_RINGS} rings, not {count}")
            }
            LayoutProblem::DuplicateRole(role) => write!(f, "the ring {role:?} appears twice"),
            LayoutProblem::Ring { role, error } => write!(f, "ring {role:?}: {error}"),
            LayoutProblem::TooLarge => f.write_str("the region would not fit the address space"),
        }
    }
}

/// Why a region could not be created or opened.
#[derive(Debug)]
pub enum ShmError {
    /// The layout is invalid.
    Layout(LayoutProblem),
    /// The OS refused a call.
    Os(io::Error),
    /// The segment does not start with [`MAGIC`].
    BadMagic,
    /// The segment's layout version is another build's.
    Version {
        /// The version found.
        found: u32,
    },
    /// The segment was made for another `ipc_protocol`.
    Protocol {
        /// The protocol found.
        found: IpcProtocol,
        /// The protocol expected.
        expected: IpcProtocol,
    },
    /// The segment's size is not the layout's.
    Size {
        /// The size found, in bytes.
        found: u64,
        /// The size the layout needs.
        expected: u64,
    },
    /// The segment's ring table is not the layout's.
    RingTable,
}

impl fmt::Display for ShmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShmError::Layout(problem) => write!(f, "invalid region layout: {problem}"),
            ShmError::Os(error) => write!(f, "shared-memory segment: {error}"),
            ShmError::BadMagic => f.write_str("the segment is not a runner region"),
            ShmError::Version { found } => write!(
                f,
                "the region's layout version is {found}, and this build reads {LAYOUT_VERSION}"
            ),
            ShmError::Protocol { found, expected } => write!(
                f,
                "the region is for ipc_protocol {}.{}, and this build speaks {}.{}",
                found.major, found.minor, expected.major, expected.minor
            ),
            ShmError::Size { found, expected } => write!(
                f,
                "the region is {found} bytes, and its layout needs {expected}"
            ),
            ShmError::RingTable => f.write_str("the region's rings are not the layout's"),
        }
    }
}

impl std::error::Error for ShmError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ShmError::Os(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for ShmError {
    fn from(error: io::Error) -> ShmError {
        ShmError::Os(error)
    }
}

/// Why a ring's end could not be taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndError {
    /// The layout has no such ring.
    NoSuchRing(RingRole),
    /// This side holds the other end of that ring.
    WrongSide {
        /// The ring.
        role: RingRole,
        /// The side the region is open on.
        side: Side,
    },
    /// This region already handed out that end.
    Taken(RingRole),
}

impl fmt::Display for EndError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EndError::NoSuchRing(role) => write!(f, "the region has no {role:?} ring"),
            EndError::WrongSide { role, side } => {
                write!(
                    f,
                    "the {side:?} side holds the other end of the {role:?} ring"
                )
            }
            EndError::Taken(role) => write!(f, "this region already handed out its {role:?} end"),
        }
    }
}

impl std::error::Error for EndError {}

/// What a runner opens its region with: the segment's descriptor on Linux, its name elsewhere. Its
/// text form, `fd:<n>` or `name:<name>`, goes on the runner's command line (ADR-0015 item 1).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RegionHandle {
    /// An inherited descriptor (Linux).
    Fd(i32),
    /// A name in the OS's shared-memory namespace (Apple platforms, Windows).
    Name(String),
}

impl fmt::Display for RegionHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegionHandle::Fd(fd) => write!(f, "fd:{fd}"),
            RegionHandle::Name(name) => write!(f, "name:{name}"),
        }
    }
}

/// Why a region handle's text form did not parse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HandleParseError(String);

impl fmt::Display for HandleParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "a region handle is fd:<n> or name:<name>, not {:?}",
            self.0
        )
    }
}

impl std::error::Error for HandleParseError {}

impl FromStr for RegionHandle {
    type Err = HandleParseError;

    fn from_str(text: &str) -> Result<RegionHandle, HandleParseError> {
        let bad = || HandleParseError(text.to_owned());
        if let Some(fd) = text.strip_prefix("fd:") {
            return fd.parse().map(RegionHandle::Fd).map_err(|_| bad());
        }
        match text.strip_prefix("name:") {
            Some(name) if !name.is_empty() => Ok(RegionHandle::Name(name.to_owned())),
            _ => Err(bad()),
        }
    }
}

/// The header's fields that are written once, by [`Region::create`].
#[repr(C)]
#[derive(Clone, Copy)]
struct Fixed {
    magic: [u8; 8],
    layout_version: u32,
    ring_count: u32,
    total_bytes: u64,
    build_id: [u8; 32],
    protocol_major: u16,
    protocol_minor: u16,
    reserved: u32,
}

/// A value alone on a 128-byte line.
#[repr(C, align(128))]
struct Line<T>(T);

/// The region's header, at offset 0.
#[repr(C)]
struct Header {
    fixed: Line<Fixed>,
    runner_heartbeat: Line<AtomicU64>,
    daemon_heartbeat: Line<AtomicU64>,
    step_counter: Line<AtomicU64>,
    runner_state: Line<AtomicU32>,
}

/// One row of the ring table, after the header.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TableEntry {
    kind: u16,
    dispatcher: u16,
    slots: u32,
    slot_bytes: u32,
    reserved: u32,
    offset: u64,
}

/// Where each ring of a layout sits, and the region's size.
#[derive(Debug)]
struct Placement {
    rings: Vec<Placed>,
    total: usize,
}

#[derive(Clone, Copy, Debug)]
struct Placed {
    spec: RingSpec,
    shape: SharedLayout,
    offset: usize,
}

impl Placed {
    fn entry(&self) -> TableEntry {
        let (kind, dispatcher) = self.spec.role.tag();
        TableEntry {
            kind,
            dispatcher,
            slots: self.spec.slots,
            slot_bytes: self.spec.slot_bytes,
            reserved: 0,
            offset: self.offset as u64,
        }
    }
}

const ALIGN: usize = SharedLayout::ALIGN;

/// The offset of the ring table: just after the header.
fn table_offset() -> usize {
    std::mem::size_of::<Header>().next_multiple_of(ALIGN)
}

/// Lays out `layout`: the header, the ring table, then each ring at a 128-byte boundary.
fn place(layout: &ShmLayout) -> Result<Placement, LayoutProblem> {
    if layout.rings.len() > MAX_RINGS {
        return Err(LayoutProblem::TooManyRings(layout.rings.len()));
    }
    let mut seen = std::collections::BTreeSet::new();
    let table_bytes = layout.rings.len() * std::mem::size_of::<TableEntry>();
    let mut offset = (table_offset() + table_bytes).next_multiple_of(ALIGN);
    let mut rings = Vec::with_capacity(layout.rings.len());
    for spec in &layout.rings {
        if !seen.insert(spec.role) {
            return Err(LayoutProblem::DuplicateRole(spec.role));
        }
        let shape = SharedLayout::new(spec.slots, spec.slot_bytes).map_err(|error| {
            LayoutProblem::Ring {
                role: spec.role,
                error,
            }
        })?;
        rings.push(Placed {
            spec: *spec,
            shape,
            offset,
        });
        offset = offset
            .checked_add(shape.bytes())
            .filter(|end| *end <= isize::MAX as usize)
            .ok_or(LayoutProblem::TooLarge)?;
    }
    Ok(Placement {
        rings,
        total: offset,
    })
}

/// The shared-memory region of one runner, open on one side.
pub struct Region {
    segment: os::Segment,
    side: Side,
    /// The header's fixed fields as this process wrote or checked them. They are never read from
    /// the segment again, so a peer that rewrites them afterwards changes nothing here.
    fixed: Fixed,
    placement: Placement,
    /// Per ring, whether this region has handed out this side's end.
    taken: Vec<AtomicBool>,
}

impl fmt::Debug for Region {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Region")
            .field("side", &self.side)
            .field("bytes", &self.placement.total)
            .field("rings", &self.placement.rings.len())
            .finish_non_exhaustive()
    }
}

impl Region {
    /// Creates the region for `layout` on the daemon's side: a new owner-only segment, with the
    /// header and ring table written and every ring empty.
    ///
    /// # Errors
    ///
    /// [`ShmError::Layout`] for an invalid layout, and [`ShmError::Os`] if the OS refuses the
    /// segment.
    ///
    /// `entropy` draws the random part of the segment's name on Apple platforms and Windows, so
    /// another process cannot guess it.
    pub fn create(
        layout: &ShmLayout,
        build_id: [u8; 32],
        protocol: IpcProtocol,
        entropy: &dyn Entropy,
    ) -> Result<Region, ShmError> {
        let placement = place(layout).map_err(ShmError::Layout)?;
        let segment = os::Segment::create(placement.total, entropy)?;
        let base = segment.base();
        let fixed = Fixed {
            magic: MAGIC,
            layout_version: LAYOUT_VERSION,
            ring_count: placement.rings.len() as u32,
            total_bytes: placement.total as u64,
            build_id,
            protocol_major: protocol.major,
            protocol_minor: protocol.minor,
            reserved: 0,
        };
        // SAFETY: the segment is `placement.total` bytes, new and zeroed, and mapped only here, so
        // nothing else reads it yet. The header sits at offset 0, which is page-aligned, and the
        // table and rings at the offsets `place` computed inside the segment, each 128-aligned.
        unsafe {
            base.cast::<Header>().write(Header {
                fixed: Line(fixed),
                runner_heartbeat: Line(AtomicU64::new(0)),
                daemon_heartbeat: Line(AtomicU64::new(0)),
                step_counter: Line(AtomicU64::new(0)),
                runner_state: Line(AtomicU32::new(0)),
            });
            let table = base.add(table_offset()).cast::<TableEntry>();
            for (at, placed) in placement.rings.iter().enumerate() {
                table.add(at).write(placed.entry());
                placed.shape.init(base.add(placed.offset));
            }
        }
        Ok(Region::with(segment, Side::Daemon, fixed, placement))
    }

    /// Opens a region another process created, on `side`, and checks it against `layout` and
    /// `protocol`. The build ID is the caller's to check ([`Region::build_id`]): a spawned runner
    /// must match the daemon's build, and an adopted one need not (RFC-0001 §3.5).
    ///
    /// On Apple platforms the segment's name is removed as soon as it is mapped here, so no third
    /// process can open it. A daemon that adopts the runner later gets the segment's descriptor over
    /// the adoption socket, not by name (RFC-0052 §6.5, note of 2026-10-22; DIST-E1-T08).
    ///
    /// Every field read from the segment is copied out once, with a volatile read, and only the
    /// copy is checked and kept.
    ///
    /// # Errors
    ///
    /// [`ShmError::Layout`] for an invalid layout, [`ShmError::Os`] if the segment cannot be
    /// opened, and the other variants when the region is not the one the layout describes.
    pub fn open(
        handle: &RegionHandle,
        side: Side,
        layout: &ShmLayout,
        protocol: IpcProtocol,
    ) -> Result<Region, ShmError> {
        let placement = place(layout).map_err(ShmError::Layout)?;
        let segment = os::Segment::open(handle, placement.total)?;
        let base = segment.base();
        // SAFETY: the segment is at least `placement.total` bytes, which covers the header, and its
        // start is page-aligned. The fixed fields are copied out with one volatile read, so a
        // later write by the peer cannot change what is checked below.
        let fixed = unsafe { base.cast::<Fixed>().read_volatile() };
        if fixed.magic != MAGIC {
            return Err(ShmError::BadMagic);
        }
        if fixed.layout_version != LAYOUT_VERSION {
            return Err(ShmError::Version {
                found: fixed.layout_version,
            });
        }
        let found = IpcProtocol {
            major: fixed.protocol_major,
            minor: fixed.protocol_minor,
        };
        if found != protocol {
            return Err(ShmError::Protocol {
                found,
                expected: protocol,
            });
        }
        if fixed.total_bytes != placement.total as u64 {
            return Err(ShmError::Size {
                found: fixed.total_bytes,
                expected: placement.total as u64,
            });
        }
        if fixed.ring_count as usize != placement.rings.len() {
            return Err(ShmError::RingTable);
        }
        // SAFETY: the table holds `ring_count` entries after the header, inside the segment, as
        // `place` laid them out; they were written once by the creator and are read by copy.
        let table = unsafe { base.add(table_offset()).cast::<TableEntry>() };
        for (at, placed) in placement.rings.iter().enumerate() {
            // SAFETY: as above, copied out with one volatile read.
            let entry = unsafe { table.add(at).read_volatile() };
            if entry != placed.entry() {
                return Err(ShmError::RingTable);
            }
        }
        segment.unlink_after_open()?;
        Ok(Region::with(segment, side, fixed, placement))
    }

    fn with(segment: os::Segment, side: Side, fixed: Fixed, placement: Placement) -> Region {
        let taken = placement
            .rings
            .iter()
            .map(|_| AtomicBool::new(false))
            .collect();
        Region {
            segment,
            side,
            fixed,
            placement,
            taken,
        }
    }

    /// The header, as a raw pointer: only its atomic fields are ever borrowed from it, never the
    /// whole header, because its fixed fields are plain memory the peer could write.
    fn header(&self) -> *const Header {
        self.segment.base().cast::<Header>().as_ptr()
    }

    fn runner_heartbeat(&self) -> &AtomicU64 {
        // SAFETY: the header lives at offset 0 for as long as the mapping, which `self` owns, and
        // the field is an atomic, so borrowing it alone is sound while the peer writes it too.
        unsafe { &(*self.header()).runner_heartbeat.0 }
    }

    fn daemon_heartbeat(&self) -> &AtomicU64 {
        // SAFETY: as in `runner_heartbeat`.
        unsafe { &(*self.header()).daemon_heartbeat.0 }
    }

    fn step_counter(&self) -> &AtomicU64 {
        // SAFETY: as in `runner_heartbeat`.
        unsafe { &(*self.header()).step_counter.0 }
    }

    fn state_word(&self) -> &AtomicU32 {
        // SAFETY: as in `runner_heartbeat`.
        unsafe { &(*self.header()).runner_state.0 }
    }

    /// The side this region is open on.
    #[must_use]
    pub fn side(&self) -> Side {
        self.side
    }

    /// The region's size, in bytes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.placement.total
    }

    /// The build ID of the daemon that created the region.
    #[must_use]
    pub fn build_id(&self) -> [u8; 32] {
        self.fixed.build_id
    }

    /// The region's `ipc_protocol`.
    #[must_use]
    pub fn protocol(&self) -> IpcProtocol {
        IpcProtocol {
            major: self.fixed.protocol_major,
            minor: self.fixed.protocol_minor,
        }
    }

    /// The handle another process opens this region with.
    #[must_use]
    pub fn handle(&self) -> RegionHandle {
        self.segment.handle()
    }

    /// Sets up `command`, the runner this daemon is about to spawn, to receive the region, and
    /// returns the handle for its command line (`__runner --ipc <handle>`).
    ///
    /// On Linux the region's descriptor keeps close-on-exec in this process, so no other child,
    /// spawned at the same time from another thread, inherits it. A hook clears the flag in
    /// `command`'s child alone, after the fork and before the exec. On Apple platforms and
    /// Windows the runner opens the region by name, and `command` is left as it is.
    ///
    /// # Errors
    ///
    /// None today; the `Result` leaves room for an OS that needs a call here.
    pub fn configure_spawn(&self, command: &mut std::process::Command) -> io::Result<RegionHandle> {
        self.segment.configure_spawn(command)
    }

    /// Removes the segment's name, so no new process can open it: `shm_unlink` on Apple
    /// platforms. A Linux memfd has no name, and a Windows mapping's name goes with its last
    /// handle, so there it does nothing. On Apple platforms [`Region::open`] already removes the
    /// name once the runner has mapped the segment; the daemon calls this when the runner exits,
    /// which also covers a runner that never opened it. Calling it twice is not an error.
    ///
    /// # Errors
    ///
    /// The OS error, other than "no such name".
    pub fn unlink(&self) -> io::Result<()> {
        self.segment.unlink()
    }

    /// Whether the segment is owner-only, read back from the OS: mode 0600 and owned by this
    /// user on Unix, and on Windows a protected DACL whose only entry allows this user.
    ///
    /// # Errors
    ///
    /// The OS error if the segment's permissions cannot be read.
    pub fn is_owner_only(&self) -> io::Result<bool> {
        self.segment.is_owner_only()
    }

    /// The segment's permissions as the OS reports them, for diagnostics: the mode and owner on
    /// Unix, and the DACL's text form on Windows.
    ///
    /// # Errors
    ///
    /// The OS error if the permissions cannot be read.
    pub fn permissions_text(&self) -> io::Result<String> {
        self.segment.permissions()
    }

    /// Adds one to this side's heartbeat.
    pub fn beat(&self) {
        self.own_heartbeat().fetch_add(1, Ordering::Release);
    }

    /// The other side's heartbeat.
    #[must_use]
    pub fn peer_heartbeat(&self) -> u64 {
        match self.side {
            Side::Daemon => self.runner_heartbeat().load(Ordering::Acquire),
            Side::Runner => self.daemon_heartbeat().load(Ordering::Acquire),
        }
    }

    /// This side's heartbeat.
    #[must_use]
    pub fn heartbeat(&self) -> u64 {
        self.own_heartbeat().load(Ordering::Acquire)
    }

    fn own_heartbeat(&self) -> &AtomicU64 {
        match self.side {
            Side::Daemon => self.daemon_heartbeat(),
            Side::Runner => self.runner_heartbeat(),
        }
    }

    /// The step counter, which the runner advances on every step and the daemon reads for stall
    /// detection (ADR-0015 item 5).
    #[must_use]
    pub fn steps(&self) -> u64 {
        self.step_counter().load(Ordering::Acquire)
    }

    /// Advances the step counter by one.
    pub fn step(&self) {
        self.step_counter().fetch_add(1, Ordering::Release);
    }

    /// The runner's current `RunnerState` discriminant.
    #[must_use]
    pub fn runner_state(&self) -> u32 {
        self.state_word().load(Ordering::Acquire)
    }

    /// Records the runner's current `RunnerState` discriminant.
    pub fn set_runner_state(&self, discriminant: u32) {
        self.state_word().store(discriminant, Ordering::Release);
    }

    /// The producing end of the ring `role`, which this side writes.
    ///
    /// # Errors
    ///
    /// [`EndError::NoSuchRing`] if the layout has no such ring, [`EndError::WrongSide`] if the
    /// other side produces it, and [`EndError::Taken`] if this region already handed it out.
    pub fn producer(&self, role: RingRole) -> Result<SharedProducer<'_>, EndError> {
        let placed = self.end(role, true)?;
        // SAFETY: `create` wrote this ring with this shape at this offset, the mapping lives as
        // long as `self`, and this side produces the ring, once per region, so no other producer
        // exists while the other process keeps to its side.
        Ok(unsafe {
            placed
                .shape
                .producer(self.segment.base().add(placed.offset))
        })
    }

    /// The consuming end of the ring `role`, which this side reads.
    ///
    /// # Errors
    ///
    /// As for [`Region::producer`].
    pub fn consumer(&self, role: RingRole) -> Result<SharedConsumer<'_>, EndError> {
        let placed = self.end(role, false)?;
        // SAFETY: as in `producer`, for the consuming end.
        Ok(unsafe {
            placed
                .shape
                .consumer(self.segment.base().add(placed.offset))
        })
    }

    /// The ring `role`, checked for this side's end and marked taken.
    fn end(&self, role: RingRole, producing: bool) -> Result<Placed, EndError> {
        let at = self
            .placement
            .rings
            .iter()
            .position(|placed| placed.spec.role == role)
            .ok_or(EndError::NoSuchRing(role))?;
        if self.side.produces(role.direction()) != producing {
            return Err(EndError::WrongSide {
                role,
                side: self.side,
            });
        }
        if self.taken[at].swap(true, Ordering::AcqRel) {
            return Err(EndError::Taken(role));
        }
        Ok(self.placement.rings[at])
    }
}

// The unit tests use real threads and a second process, so they build only without loom; under
// `--cfg loom` the region's loom model runs instead (QUAL-E3-T09).
#[cfg(all(test, loom))]
mod loom_model;
#[cfg(all(test, not(loom)))]
mod tests;
