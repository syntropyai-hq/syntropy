// SPDX-License-Identifier: MIT OR Apache-2.0

//! The v1 directory layout of a store root (RFC-0041 §4.1).
//!
//! [`Layout`] computes every path the layout names from the root's path. It does no I/O, so the
//! same paths serve a writable root and a read-only one. Opening a root, creating its
//! directories and taking its writer lock are in [`crate::root`].
//!
//! ```text
//! <root>/v1/
//!   blobs/<sha256>.blob                 Layout::blob
//!   blobs/<sha256>.inv                  Layout::inventory
//!   rangemap/<sha256>.map               Layout::range_map
//!   tables/<sha256>.chunks              Layout::chunk_table
//!   manifests/<model-id>.manifest       Layout::manifest
//!   manifests/<model-id>.<quant>.refdigest   Layout::ref_digest
//!   refs/hub/<owner>/<repo>/<revision>  Layout::hub_ref
//!   refs/local/<alias>                  Layout::local_ref
//!   local/<model-id>.sources            Layout::local_sources
//!   packs/<device-class>/seg/<n>.seg    Layout::segment
//!   packs/<device-class>/extents.idx    Layout::extent_index
//!   packs/<device-class>/side/<pack-key>.side      Layout::sidecar
//!   packs/<device-class>/side/<pack-key>.<kind>    Layout::side_record
//!   packs/<device-class>/build/<pack-key>.intent   Layout::build_intent
//!   staging/<sha256>/<chunk>.part       Layout::staging_part
//!   staging/sparse-probe                Layout::sparse_probe; the start-up probe (§4.2), only
//!                                       while it runs
//!   chunks/<id[0..2]>/<id>              Layout::chunk_file; a chunk in chunk-file mode (§4.2)
//!   sessions/                           Layout::sessions_dir
//!   locks/                              Layout::locks_dir
//!   locks/writer.lock                   Layout::writer_lock; the writer lock (not named by §4.1)
//!   locks/writer.pid                    Layout::writer_pid; the lock holder's process ID
//! ```
//!
//! `<sha256>` (a file's `FileDigest`) and `<model-id>` (a `ModelId`, BLAKE3) are 32-byte
//! digests, written as 64 lowercase hex digits, and so is `<id>` (a `ChunkId`, BLAKE3), whose
//! first two digits name its directory. `<n>` and `<chunk>` are decimal integers; a
//! staged part is named by the chunk's index in its file (RFC-0041 §7.3). Every other
//! component is a caller-supplied [`Name`].

use std::fmt;
use std::path::{Path, PathBuf};

/// The directory under a root that holds layout version 1.
pub const LAYOUT_DIR: &str = "v1";

/// File name of a root's writer lock, inside `locks/`.
pub const WRITER_LOCK_FILE: &str = "writer.lock";

/// File name of the record of the writer lock holder's process ID, inside `locks/`.
pub const WRITER_PID_FILE: &str = "writer.pid";

/// Longest [`Name`], in bytes. Common file systems limit a path component to 255 bytes.
pub const NAME_MAX_LEN: usize = 255;

/// One caller-supplied path component of the layout: a Hub owner, repository or revision, a
/// local alias, a quantisation name, a device class, a pack key or a side-record kind.
///
/// A name is 1 to [`NAME_MAX_LEN`] bytes of ASCII letters, digits, `.`, `_` and `-`, and:
///
/// - is neither `.` nor `..`;
/// - does not end with `.`, because Windows drops trailing periods, so that `a.` names the
///   same file as `a` and `...` names the directory it sits in;
/// - has a base name, the part before its first `.`, that is not one of Windows' reserved
///   device names in any case: `CON`, `PRN`, `AUX`, `NUL`, `COM1`–`COM9` or `LPT1`–`LPT9`.
///   Windows takes `nul`, `NUL.txt` and `Nul.tar.gz` all for the `NUL` device.
///
/// No name can hold a path separator, so no name can leave the directory it is joined to. The
/// last two rules are Microsoft's naming rules for files ("Naming Files, Paths, and
/// Namespaces"; the superscript-digit device names it also reserves are outside the character
/// set), applied on every OS so that a store written on Linux or macOS stays usable on Windows.
/// The layout's file names append fixed suffixes such as `.side` to a name, which keeps its
/// base name, so they follow the same rules. A value outside these rules, such as a Hub
/// revision containing `/`, must be encoded by its owner before it becomes a `Name`.
///
/// Names compare byte for byte, but a case-insensitive file system (the macOS and Windows
/// defaults) treats `Q4_K_M` and `q4_k_m` as one file. Owners choose encodings that do not
/// depend on case.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Name(String);

impl Name {
    /// Checks `name` against the rules on [`Name`].
    pub fn new(name: impl Into<String>) -> Result<Name, NameError> {
        let name = name.into();
        if name.is_empty() {
            return Err(NameError::Empty);
        }
        if name.len() > NAME_MAX_LEN {
            return Err(NameError::TooLong { len: name.len() });
        }
        if name == "." || name == ".." {
            return Err(NameError::DotName);
        }
        if let Some((at, ch)) = name.char_indices().find(|&(_, c)| !is_name_char(c)) {
            return Err(NameError::InvalidChar { ch, at });
        }
        if name.ends_with('.') {
            return Err(NameError::TrailingDot);
        }
        if is_reserved_device_name(&name) {
            return Err(NameError::ReservedDeviceName { name });
        }
        Ok(Name(name))
    }

    /// The 64-digit lowercase hex form of a 32-byte digest, which is always a valid name.
    pub fn from_digest(digest: &[u8; 32]) -> Name {
        Name(hex(digest))
    }

    /// The name as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for Name {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl TryFrom<&str> for Name {
    type Error = NameError;

    fn try_from(name: &str) -> Result<Name, NameError> {
        Name::new(name)
    }
}

impl TryFrom<String> for Name {
    type Error = NameError;

    fn try_from(name: String) -> Result<Name, NameError> {
        Name::new(name)
    }
}

fn is_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')
}

/// Windows' reserved device names, from Microsoft's "Naming Files, Paths, and Namespaces".
/// Its superscript-digit forms (`COM¹`, `LPT³`, …) are not ASCII, so no [`Name`] can hold them.
const RESERVED_DEVICE_NAMES: [&str; 22] = [
    "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
    "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
];

/// Whether Windows takes `name` for a device: its base name, before the first `.`, is a
/// reserved device name in any case.
fn is_reserved_device_name(name: &str) -> bool {
    let base = name.split_once('.').map_or(name, |(base, _)| base);
    RESERVED_DEVICE_NAMES
        .iter()
        .any(|reserved| reserved.eq_ignore_ascii_case(base))
}

/// Why a string is not a [`Name`], or a name is not a side-record kind.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NameError {
    /// The string is empty.
    Empty,
    /// The string is longer than [`NAME_MAX_LEN`] bytes.
    TooLong {
        /// Its length, in bytes.
        len: usize,
    },
    /// The string is `.` or `..`.
    DotName,
    /// The string holds a character outside ASCII letters, digits, `.`, `_` and `-`.
    InvalidChar {
        /// The first such character.
        ch: char,
        /// Its byte offset.
        at: usize,
    },
    /// The string ends with `.`, which Windows drops.
    TrailingDot,
    /// The string's base name, before its first `.`, is a Windows reserved device name.
    ReservedDeviceName {
        /// The string.
        name: String,
    },
    /// A side-record kind contains `.` or is `side`, so its file could be taken for another
    /// pack's sidecar or side record (see [`Layout::side_record`]).
    ReservedKind {
        /// The kind.
        kind: String,
    },
}

impl fmt::Display for NameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NameError::Empty => f.write_str("a store path name cannot be empty"),
            NameError::TooLong { len } => write!(
                f,
                "a store path name is at most {NAME_MAX_LEN} bytes; this one is {len}"
            ),
            NameError::DotName => f.write_str("a store path name cannot be `.` or `..`"),
            NameError::InvalidChar { ch, at } => write!(
                f,
                "character {ch:?} at byte {at} is not allowed in a store path name \
                 (ASCII letters, digits, `.`, `_` and `-` only)"
            ),
            NameError::TrailingDot => f.write_str(
                "a store path name cannot end with `.`, because Windows drops trailing periods",
            ),
            NameError::ReservedDeviceName { name } => write!(
                f,
                "store path name {name:?} names a Windows device (CON, PRN, AUX, NUL, \
                 COM1-COM9 or LPT1-LPT9, in any case, with or without an extension)"
            ),
            NameError::ReservedKind { kind } => write!(
                f,
                "side-record kind {kind:?} must not contain `.` and must not be `side`"
            ),
        }
    }
}

impl std::error::Error for NameError {}

/// Every path of the v1 layout under one root (RFC-0041 §4.1).
///
/// A `Layout` is plain data: building one and asking it for paths touches no file. The paths
/// are under [`Layout::base`], `<root>/v1`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    root: PathBuf,
    base: PathBuf,
}

impl Layout {
    /// The layout of the root at `root`.
    pub fn new(root: impl Into<PathBuf>) -> Layout {
        let root = root.into();
        let base = root.join(LAYOUT_DIR);
        Layout { root, base }
    }

    /// The root directory, as given to [`Layout::new`].
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// `<root>/v1`, the directory that holds every other path of the layout.
    pub fn base(&self) -> &Path {
        &self.base
    }

    /// The layout's fixed directories, parents before children: every directory of §4.1 that
    /// takes no caller-supplied name. A writable root creates them when it is opened.
    pub fn fixed_dirs(&self) -> [PathBuf; 12] {
        [
            self.base.clone(),
            self.blobs_dir(),
            self.range_map_dir(),
            self.tables_dir(),
            self.manifests_dir(),
            self.hub_refs_dir(),
            self.local_refs_dir(),
            self.local_dir(),
            self.packs_dir(),
            self.staging_dir(),
            self.sessions_dir(),
            self.locks_dir(),
        ]
    }

    /// `blobs/`: source files, sparse while partial, and their inventories.
    pub fn blobs_dir(&self) -> PathBuf {
        self.base.join("blobs")
    }

    /// `blobs/<sha256>.blob`: a source file's bytes at their original offsets (§4.2).
    pub fn blob(&self, sha256: &[u8; 32]) -> PathBuf {
        self.blobs_dir().join(format!("{}.blob", hex(sha256)))
    }

    /// `blobs/<sha256>.inv`: the chunk-table digest and the bitmap of verified chunks (§4.2).
    pub fn inventory(&self, sha256: &[u8; 32]) -> PathBuf {
        self.blobs_dir().join(format!("{}.inv", hex(sha256)))
    }

    /// `rangemap/`: range maps of punched source chunks.
    pub fn range_map_dir(&self) -> PathBuf {
        self.base.join("rangemap")
    }

    /// `rangemap/<sha256>.map`: chunks punched from the blob and held in a `SOURCE`-layout
    /// pack extent (§4.5).
    pub fn range_map(&self, sha256: &[u8; 32]) -> PathBuf {
        self.range_map_dir().join(format!("{}.map", hex(sha256)))
    }

    /// `tables/`: chunk tables.
    pub fn tables_dir(&self) -> PathBuf {
        self.base.join("tables")
    }

    /// `tables/<sha256>.chunks`: the chunk table of a file (§6.2).
    pub fn chunk_table(&self, sha256: &[u8; 32]) -> PathBuf {
        self.tables_dir().join(format!("{}.chunks", hex(sha256)))
    }

    /// `manifests/`: model manifests and canary digest records.
    pub fn manifests_dir(&self) -> PathBuf {
        self.base.join("manifests")
    }

    /// `manifests/<model-id>.manifest`: the `ModelManifest`, postcard + zstd.
    pub fn manifest(&self, model_id: &[u8; 32]) -> PathBuf {
        self.manifests_dir()
            .join(format!("{}.manifest", hex(model_id)))
    }

    /// `manifests/<model-id>.<quant>.refdigest`: the warm-up canary digest record (§5.3).
    pub fn ref_digest(&self, model_id: &[u8; 32], quant: &Name) -> PathBuf {
        self.manifests_dir()
            .join(format!("{}.{quant}.refdigest", hex(model_id)))
    }

    /// `refs/hub/`: resolved Hub references.
    pub fn hub_refs_dir(&self) -> PathBuf {
        self.base.join("refs").join("hub")
    }

    /// `refs/hub/<owner>/<repo>/<revision>`: a Hub reference's resolved commit and `ModelId`.
    pub fn hub_ref(&self, owner: &Name, repo: &Name, revision: &Name) -> PathBuf {
        self.hub_refs_dir()
            .join(owner.as_str())
            .join(repo.as_str())
            .join(revision.as_str())
    }

    /// `refs/local/`: local aliases.
    pub fn local_refs_dir(&self) -> PathBuf {
        self.base.join("refs").join("local")
    }

    /// `refs/local/<alias>`: the `ModelId` a local alias names.
    pub fn local_ref(&self, alias: &Name) -> PathBuf {
        self.local_refs_dir().join(alias.as_str())
    }

    /// `local/`: records of local models indexed in place.
    pub fn local_dir(&self) -> PathBuf {
        self.base.join("local")
    }

    /// `local/<model-id>.sources`: a local model's files, each with its path, `FileDigest`
    /// and (device, inode, size, mtime) (§4.3).
    pub fn local_sources(&self, model_id: &[u8; 32]) -> PathBuf {
        self.local_dir().join(format!("{}.sources", hex(model_id)))
    }

    /// `packs/`: one directory per device class.
    pub fn packs_dir(&self) -> PathBuf {
        self.base.join("packs")
    }

    /// `packs/<device-class>/`.
    pub fn pack_class_dir(&self, class: &Name) -> PathBuf {
        self.packs_dir().join(class.as_str())
    }

    /// `packs/<device-class>/seg/`: the class's extent segments.
    pub fn segments_dir(&self, class: &Name) -> PathBuf {
        self.pack_class_dir(class).join("seg")
    }

    /// `packs/<device-class>/seg/<n>.seg`: extent segment `n` (§5.2).
    pub fn segment(&self, class: &Name, n: u64) -> PathBuf {
        self.segments_dir(class).join(format!("{n}.seg"))
    }

    /// `packs/<device-class>/extents.idx`: `ExtentKey` → (segment, offset, length, BLAKE3)
    /// (§5.2).
    pub fn extent_index(&self, class: &Name) -> PathBuf {
        self.pack_class_dir(class).join("extents.idx")
    }

    /// `packs/<device-class>/side/`: sidecars and side records.
    pub fn side_dir(&self, class: &Name) -> PathBuf {
        self.pack_class_dir(class).join("side")
    }

    /// `packs/<device-class>/side/<pack-key>.side`: a pack's sidecar, immutable once committed
    /// (§5.2).
    pub fn sidecar(&self, class: &Name, pack_key: &Name) -> PathBuf {
        self.side_dir(class).join(format!("{pack_key}.side"))
    }

    /// `packs/<device-class>/side/<pack-key>.<kind>`: a side record, such as selection
    /// entries (`select`) or the expert profile (`xprofile`) (§5.2).
    ///
    /// # Errors
    ///
    /// [`NameError::ReservedKind`] when `kind` contains `.` or is `side`. Either would let one
    /// pack's side record share a file name with another pack's sidecar or side record, for
    /// example kind `b.side` of pack `a` and the sidecar of pack `a.b`.
    pub fn side_record(
        &self,
        class: &Name,
        pack_key: &Name,
        kind: &Name,
    ) -> Result<PathBuf, NameError> {
        if kind.as_str() == "side" || kind.as_str().contains('.') {
            return Err(NameError::ReservedKind {
                kind: kind.as_str().to_owned(),
            });
        }
        Ok(self.side_dir(class).join(format!("{pack_key}.{kind}")))
    }

    /// `packs/<device-class>/build/`: build-intent files.
    pub fn build_dir(&self, class: &Name) -> PathBuf {
        self.pack_class_dir(class).join("build")
    }

    /// `packs/<device-class>/build/<pack-key>.intent`: present only while the pack is being
    /// built (§5.2).
    pub fn build_intent(&self, class: &Name, pack_key: &Name) -> PathBuf {
        self.build_dir(class).join(format!("{pack_key}.intent"))
    }

    /// `staging/`: partial downloads.
    pub fn staging_dir(&self) -> PathBuf {
        self.base.join("staging")
    }

    /// `staging/<sha256>/`: a file's partial chunks.
    pub fn staging_file_dir(&self, sha256: &[u8; 32]) -> PathBuf {
        self.staging_dir().join(hex(sha256))
    }

    /// `staging/<sha256>/<chunk>.part`: the received prefix of chunk `index` of a file,
    /// promoted only after verification (§7.3).
    pub fn staging_part(&self, sha256: &[u8; 32], index: u64) -> PathBuf {
        self.staging_file_dir(sha256).join(format!("{index}.part"))
    }

    /// `staging/sparse-probe`: the file a writable root's start allocates sparse to learn
    /// whether its volume has sparse files (§4.2), removed again at once. Its name is not hex,
    /// so it cannot be a staged file's directory.
    pub fn sparse_probe(&self) -> PathBuf {
        self.staging_dir().join("sparse-probe")
    }

    /// `chunks/`: one file per chunk, for a root on a volume without sparse files (§4.2). Not a
    /// fixed directory: only a root in chunk-file mode creates it.
    pub fn chunks_dir(&self) -> PathBuf {
        self.base.join("chunks")
    }

    /// `chunks/<id[0..2]>/<id>`: the chunk whose `ChunkId` is `id`, in chunk-file mode (§4.2).
    pub fn chunk_file(&self, id: &[u8; 32]) -> PathBuf {
        let name = hex(id);
        self.chunks_dir().join(&name[..2]).join(name)
    }

    /// `sessions/`: persisted session KV (RFC-0022).
    pub fn sessions_dir(&self) -> PathBuf {
        self.base.join("sessions")
    }

    /// `locks/`: advisory locks: the root's writer lock and runner pack leases.
    pub fn locks_dir(&self) -> PathBuf {
        self.base.join("locks")
    }

    /// `locks/writer.lock`: the lock that admits one writer process per root.
    pub fn writer_lock(&self) -> PathBuf {
        self.locks_dir().join(WRITER_LOCK_FILE)
    }

    /// `locks/writer.pid`: the process ID of the writer lock's holder, as a diagnostic. It is
    /// a separate file because the lock file itself cannot be read on Windows while it is
    /// locked.
    pub fn writer_pid(&self) -> PathBuf {
        self.locks_dir().join(WRITER_PID_FILE)
    }
}

/// Lowercase hex of a 32-byte digest.
fn hex(digest: &[u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for &b in digest {
        out.push(char::from(DIGITS[usize::from(b >> 4)]));
        out.push(char::from(DIGITS[usize::from(b & 0x0f)]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn name(s: &str) -> Name {
        Name::new(s).unwrap()
    }

    /// The path relative to the root, with `/` separators on every OS.
    fn rel(layout: &Layout, path: &Path) -> String {
        let rel = path.strip_prefix(layout.root()).unwrap();
        rel.components()
            .map(|c| c.as_os_str().to_str().unwrap())
            .collect::<Vec<_>>()
            .join("/")
    }

    const SHA: [u8; 32] = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
        0x0f, 0x10, 0x2f, 0x3e, 0x4d, 0x5c, 0x6b, 0x7a, 0x89, 0x98, 0xa7, 0xb6, 0xc5, 0xd4, 0xe3,
        0xf2, 0xff,
    ];
    const SHA_HEX: &str = "000102030405060708090a0b0c0d0e0f102f3e4d5c6b7a8998a7b6c5d4e3f2ff";

    #[test]
    fn every_path_matches_rfc_0041_section_4_1() {
        let l = Layout::new("store-root");
        let m = [0xab; 32];
        let mh = "ab".repeat(32);
        let class = name("cuda-sm_120");
        let pack = name("pk01");
        let cases: Vec<(PathBuf, String)> = vec![
            (l.base().to_owned(), "v1".into()),
            (l.blob(&SHA), format!("v1/blobs/{SHA_HEX}.blob")),
            (l.inventory(&SHA), format!("v1/blobs/{SHA_HEX}.inv")),
            (l.range_map(&SHA), format!("v1/rangemap/{SHA_HEX}.map")),
            (l.chunk_table(&SHA), format!("v1/tables/{SHA_HEX}.chunks")),
            (l.manifest(&m), format!("v1/manifests/{mh}.manifest")),
            (
                l.ref_digest(&m, &name("Q4_K_M")),
                format!("v1/manifests/{mh}.Q4_K_M.refdigest"),
            ),
            (
                l.hub_ref(&name("acme"), &name("model-7B"), &name("main")),
                "v1/refs/hub/acme/model-7B/main".into(),
            ),
            (l.local_ref(&name("mine")), "v1/refs/local/mine".into()),
            (l.local_sources(&m), format!("v1/local/{mh}.sources")),
            (
                l.segment(&class, 7),
                "v1/packs/cuda-sm_120/seg/7.seg".into(),
            ),
            (
                l.extent_index(&class),
                "v1/packs/cuda-sm_120/extents.idx".into(),
            ),
            (
                l.sidecar(&class, &pack),
                "v1/packs/cuda-sm_120/side/pk01.side".into(),
            ),
            (
                l.side_record(&class, &pack, &name("select")).unwrap(),
                "v1/packs/cuda-sm_120/side/pk01.select".into(),
            ),
            (
                l.side_record(&class, &pack, &name("xprofile")).unwrap(),
                "v1/packs/cuda-sm_120/side/pk01.xprofile".into(),
            ),
            (
                l.build_intent(&class, &pack),
                "v1/packs/cuda-sm_120/build/pk01.intent".into(),
            ),
            (
                l.staging_part(&SHA, 12),
                format!("v1/staging/{SHA_HEX}/12.part"),
            ),
            (l.sparse_probe(), "v1/staging/sparse-probe".into()),
            (l.chunk_file(&SHA), format!("v1/chunks/00/{SHA_HEX}")),
            (l.sessions_dir(), "v1/sessions".into()),
            (l.locks_dir(), "v1/locks".into()),
            (l.writer_lock(), "v1/locks/writer.lock".into()),
            (l.writer_pid(), "v1/locks/writer.pid".into()),
        ];
        for (path, want) in cases {
            assert_eq!(rel(&l, &path), want);
        }
    }

    #[test]
    fn fixed_dirs_are_the_unparameterised_directories_parents_first() {
        let l = Layout::new("r");
        let got: Vec<String> = l.fixed_dirs().iter().map(|p| rel(&l, p)).collect();
        assert_eq!(
            got,
            [
                "v1",
                "v1/blobs",
                "v1/rangemap",
                "v1/tables",
                "v1/manifests",
                "v1/refs/hub",
                "v1/refs/local",
                "v1/local",
                "v1/packs",
                "v1/staging",
                "v1/sessions",
                "v1/locks",
            ]
        );
    }

    #[test]
    fn digest_names_are_lowercase_hex() {
        assert_eq!(Name::from_digest(&SHA).as_str(), SHA_HEX);
        assert_eq!(Name::new(SHA_HEX).unwrap(), Name::from_digest(&SHA));
    }

    #[test]
    fn names_accept_the_portable_set() {
        for ok in [
            "a",
            "Q4_K_M",
            "model-7B",
            "v1.2",
            "a..b",
            ".hidden",
            "x".repeat(255).as_str(),
            // Near the reserved device names, but not one of them.
            "CONSOLE",
            "auxiliary",
            "nul_alias",
            "con-1",
            "COM10",
            "lpt10",
            ".nul",
            "x.con",
        ] {
            assert_eq!(Name::new(ok).unwrap().as_str(), ok);
        }
    }

    #[test]
    fn names_reject_anything_that_could_leave_their_directory() {
        assert_eq!(Name::new(""), Err(NameError::Empty));
        assert_eq!(Name::new("."), Err(NameError::DotName));
        assert_eq!(Name::new(".."), Err(NameError::DotName));
        assert_eq!(
            Name::new("x".repeat(256)),
            Err(NameError::TooLong { len: 256 })
        );
        for (bad, ch, at) in [
            ("a/b", '/', 1),
            ("a\\b", '\\', 1),
            ("C:", ':', 1),
            ("a\0", '\0', 1),
            ("a b", ' ', 1),
            ("é", 'é', 0),
            ("refs/pr/1", '/', 4),
            // Windows drops trailing spaces as well as periods; a space is never allowed.
            ("a ", ' ', 1),
            ("a. ", ' ', 2),
            // The superscript-digit device names are outside the character set.
            ("COM\u{b9}", '\u{b9}', 3),
            ("lpt\u{b3}.txt", '\u{b3}', 3),
        ] {
            assert_eq!(
                Name::new(bad),
                Err(NameError::InvalidChar { ch, at }),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn names_reject_a_trailing_period() {
        for bad in ["a.", "...", "Q4_K_M.", "x..", ".hidden."] {
            assert_eq!(Name::new(bad), Err(NameError::TrailingDot), "{bad:?}");
        }
    }

    #[test]
    fn names_reject_windows_device_names_in_any_case_with_or_without_an_extension() {
        let mut reserved: Vec<String> = ["CON", "PRN", "AUX", "NUL"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        for n in 1..=9 {
            reserved.push(format!("COM{n}"));
            reserved.push(format!("LPT{n}"));
        }
        assert_eq!(reserved.len(), RESERVED_DEVICE_NAMES.len());
        for base in &reserved {
            let lower = base.to_ascii_lowercase();
            let mixed: String = base
                .chars()
                .enumerate()
                .map(|(i, c)| {
                    if i % 2 == 1 {
                        c.to_ascii_lowercase()
                    } else {
                        c
                    }
                })
                .collect();
            for spelling in [base.clone(), lower, mixed] {
                for name in [
                    spelling.clone(),
                    format!("{spelling}.txt"),
                    format!("{spelling}.tar.gz"),
                    format!("{spelling}.side"),
                ] {
                    assert_eq!(
                        Name::new(name.clone()),
                        Err(NameError::ReservedDeviceName { name: name.clone() }),
                        "{name:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn side_record_kinds_cannot_alias_a_sidecar() {
        let l = Layout::new("r");
        let class = name("cpu");
        for kind in ["side", "b.side", "x.y"] {
            assert_eq!(
                l.side_record(&class, &name("a"), &name(kind)),
                Err(NameError::ReservedKind { kind: kind.into() })
            );
        }
    }
}
