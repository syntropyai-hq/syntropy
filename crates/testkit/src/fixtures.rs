// SPDX-License-Identifier: MIT OR Apache-2.0

//! Synthetic model fixtures generated from a seed (RFC-0060 §12.1; ADR-0603 decision 1;
//! QUAL-E7-T01).
//!
//! [`generate`] builds one checkpoint per variant of a family's [`FamilySpec`]: the spec lists
//! variants because some features, such as QK-norm or tied embeddings, are fixed for a whole
//! checkpoint (`graph::spec`). Each fixture is structure-true: it holds every layer kind in its
//! variant's pattern and every attention feature the variant has. Each is written as
//!
//! - an F32 original in safetensors, with `config.json`, `tokenizer.json` and
//!   `tokenizer_config.json`;
//! - the same weights as an F32 GGUF file, when GGUF metadata can state the variant's RoPE rule
//!   ([`gguf::states`]). A `llama`-architecture GGUF stores each head's Q and K rows with the
//!   two RoPE halves interleaved, as that format pairs them, so its rows are permuted to match.
//!
//! **Weights.** Each matrix is drawn from a zero-mean normal scaled by 1/√fan-in, the token
//! table from a standard normal, and norm gains are 1. Calibrating gains, routers and the head
//! scale to the entropy target is QUAL-E7-T02's (M0b), and quantizing the original is
//! QUAL-E7-T03's (M0b), so [`QuantClass`] holds F32 alone for now.
//!
//! **Determinism.** Every byte comes from the seed, the family tag and the variant name: the
//! normals come from [`SimEntropy`](crate::sim::SimEntropy) through [`normal`], which uses
//! correctly rounded arithmetic only, and every writer emits fields in a fixed order. A seed
//! therefore gives byte-identical files on every platform; the tests hold their BLAKE3 digests.
//!
//! **The manifest.** [`Fixture::manifest`] is the census of the written safetensors checkpoint,
//! read back through `manifest`'s readers and `graph::census::build`, so a fixture whose
//! tensors the family cannot classify fails to generate.
//!
//! RFC-0060 §12.1 sketches `generate(spec, size, seed, quants) -> Fixture`. Here it also takes
//! the directory to write into and returns one fixture per variant.

pub mod gguf;
pub mod model;
pub mod normal;
pub mod safetensors;
pub mod tokenizer;

use std::fmt;
use std::path::{Path, PathBuf};

use syntropy_core::DType;
use syntropy_graph::census::{CensusError, Sources, Weights, build};
use syntropy_graph::families::MetaView;
use syntropy_graph::spec::{FamilySpec, VariantSpec};
use syntropy_manifest::census::ModelManifest;
use syntropy_manifest::config::HfConfig;
use syntropy_manifest::identity::{FileDigest, SourceFile};
use syntropy_manifest::kv::KvSource;
use syntropy_manifest::range::FileSource;
use syntropy_manifest::safetensors::read_header_default;
use syntropy_manifest::safetensors::side::SourceRole;
use syntropy_manifest::tokenizer::TokenizerSources;

use self::model::{Dims, Original};
use self::normal::Normals;

/// A fixture's size class (RFC-0060 §12.1).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SizeClass {
    /// ≤ 8 layers, hidden 64–256, ≤ 16 experts, the byte-level tokenizer. Every PR.
    Nano,
    /// The family's real head dims, expert width and tokenizer, 8–12 layers (QUAL-E7-T04).
    Micro,
    /// The full census and shapes with no weights (QUAL-E7-T07).
    FullShape,
}

/// A stored form of a fixture's GGUF weights.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuantClass {
    /// The F32 original. Quantized classes come with QUAL-E7-T03 and QUAL-E7-T14.
    F32,
}

/// One generated checkpoint.
#[derive(Clone, Debug)]
pub struct Fixture {
    /// The family tag.
    pub family: &'static str,
    /// The variant it was built from.
    pub variant: &'static str,
    /// The directory holding its files.
    pub dir: PathBuf,
    /// Its GGUF files, by quant class; empty when GGUF cannot state the variant.
    pub gguf: Vec<(QuantClass, PathBuf)>,
    /// The F32 original in safetensors.
    pub safetensors: Option<PathBuf>,
    /// `tokenizer.json`.
    pub tokenizer: PathBuf,
    /// The census of the safetensors checkpoint.
    pub manifest: ModelManifest,
}

/// Why a fixture cannot be generated.
#[derive(Debug)]
pub enum FixtureError {
    /// The spec asks for something the generator does not build yet.
    Unsupported(String),
    /// A file could not be written or read back.
    Io(String),
    /// The family refuses the generated checkpoint.
    Census {
        /// The file the census read.
        file: PathBuf,
        /// Why.
        error: CensusError,
    },
}

impl fmt::Display for FixtureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(what) => write!(f, "not generated: {what}"),
            Self::Io(what) => write!(f, "fixture I/O failed: {what}"),
            Self::Census { file, error } => {
                write!(f, "the census refuses {}: {error}", file.display())
            }
        }
    }
}

impl std::error::Error for FixtureError {}

fn io(path: &Path, e: impl fmt::Display) -> FixtureError {
    FixtureError::Io(format!("{}: {e}", path.display()))
}

fn write(path: &Path, bytes: &[u8]) -> Result<(), FixtureError> {
    std::fs::write(path, bytes).map_err(|e| io(path, e))
}

fn read(path: &Path) -> Result<Vec<u8>, FixtureError> {
    std::fs::read(path).map_err(|e| io(path, e))
}

/// The seed of one variant's draws: BLAKE3 over the seed, the family tag and the variant name,
/// so adding a variant leaves the others' bytes unchanged.
fn variant_seed(seed: u64, family: &str, variant: &str) -> u64 {
    let mut h = blake3::Hasher::new();
    h.update(b"syntropy-testkit-fixture-v1\0");
    h.update(&seed.to_le_bytes());
    h.update(family.as_bytes());
    h.update(b"\0");
    h.update(variant.as_bytes());
    let mut word = [0u8; 8];
    word.copy_from_slice(&h.finalize().as_bytes()[..8]);
    u64::from_le_bytes(word)
}

/// Builds every variant of `spec` at `size` from `seed`, writing each into
/// `out/<family>-<variant>/`.
///
/// # Errors
///
/// [`FixtureError::Unsupported`] for a size class, quant class, layer kind or feature the
/// generator does not build yet; [`FixtureError::Io`] when a file cannot be written;
/// [`FixtureError::Census`] when the family refuses the result.
pub fn generate(
    spec: &FamilySpec,
    size: SizeClass,
    seed: u64,
    quants: &[QuantClass],
    out: &Path,
) -> Result<Vec<Fixture>, FixtureError> {
    spec.variants
        .iter()
        .map(|v| generate_variant(spec.family, v, size, seed, quants, out))
        .collect()
}

/// The original of one variant, without writing anything.
///
/// # Errors
///
/// As [`generate`], less I/O and census errors.
pub fn original(
    family: &str,
    v: &VariantSpec,
    size: SizeClass,
    seed: u64,
) -> Result<Original, FixtureError> {
    let dims = Dims::of(v, size)?;
    let mut normals = Normals::new(variant_seed(seed, family, v.name));
    model::build(v, dims, &mut normals)
}

fn generate_variant(
    family: &'static str,
    v: &'static VariantSpec,
    size: SizeClass,
    seed: u64,
    quants: &[QuantClass],
    out: &Path,
) -> Result<Fixture, FixtureError> {
    let o = original(family, v, size, seed)?;
    let name = format!("{family}-{}", v.name);
    let dir = out.join(&name);
    std::fs::create_dir_all(&dir).map_err(|e| io(&dir, e))?;

    let st_path = dir.join("model.safetensors");
    let config_path = dir.join("config.json");
    let tok_path = dir.join("tokenizer.json");
    let tok_config_path = dir.join("tokenizer_config.json");
    write(&st_path, &safetensors::write(&o))?;
    write(&config_path, safetensors::config_json(v, &o).as_bytes())?;
    write(&tok_path, tokenizer::tokenizer_json().as_bytes())?;
    write(
        &tok_config_path,
        tokenizer::tokenizer_config_json().as_bytes(),
    )?;

    let mut gguf_files = Vec::new();
    if gguf::states(v.attention.rope) {
        for q in quants {
            match q {
                QuantClass::F32 => {
                    let path = dir.join("model-f32.gguf");
                    write(&path, &gguf::write(v, &o, &name))?;
                    gguf_files.push((*q, path));
                }
            }
        }
    }

    let manifest = census_safetensors(&st_path, &config_path, &tok_path, &tok_config_path)?;
    Ok(Fixture {
        family,
        variant: v.name,
        dir,
        gguf: gguf_files,
        safetensors: Some(st_path),
        tokenizer: tok_path,
        manifest,
    })
}

fn source_file(role: SourceRole, bytes: &[u8]) -> SourceFile {
    SourceFile {
        role,
        digest: FileDigest::of_bytes(bytes),
        len: bytes.len() as u64,
    }
}

/// The census of a written safetensors checkpoint, read back from disk.
///
/// # Errors
///
/// [`FixtureError::Io`] when a file cannot be read or parsed; [`FixtureError::Census`] when the
/// family refuses it.
pub fn census_safetensors(
    weights: &Path,
    config: &Path,
    tokenizer_json: &Path,
    tokenizer_config: &Path,
) -> Result<ModelManifest, FixtureError> {
    let source = FileSource::open(weights).map_err(|e| io(weights, e))?;
    let header = read_header_default(&source).map_err(|e| io(weights, e))?;
    let (weights_bytes, config_bytes) = (read(weights)?, read(config)?);
    let (tok, tok_config) = (read(tokenizer_json)?, read(tokenizer_config)?);
    let hf = HfConfig::parse(&config_bytes, None).map_err(|e| io(config, e))?;
    let files = [
        source_file(SourceRole::Weights, &weights_bytes),
        source_file(SourceRole::Config, &config_bytes),
        source_file(SourceRole::Tokenizer, &tok),
        source_file(SourceRole::TokenizerConfig, &tok_config),
    ];
    let shards = [header];
    build(&Sources {
        meta: MetaView::Config(&hf),
        weights: Weights::Safetensors(&shards),
        kv_source: KvSource::Safetensors {
            compute: Some(DType::F32),
            quantized: false,
        },
        tokenizer: Some(TokenizerSources {
            tokenizer_json: &tok,
            tokenizer_config: Some(&tok_config),
            generation_config: None,
            chat_template_file: None,
        }),
        files: &files,
    })
    .map_err(|error| FixtureError::Census {
        file: weights.to_owned(),
        error,
    })
}

/// The census of a written GGUF file, read back from disk.
///
/// # Errors
///
/// As [`census_safetensors`].
pub fn census_gguf(path: &Path) -> Result<ModelManifest, FixtureError> {
    use syntropy_manifest::gguf::{GgufHeader, Limits};
    let source = FileSource::open(path).map_err(|e| io(path, e))?;
    let header = GgufHeader::read(&source, &Limits::default()).map_err(|e| io(path, e))?;
    let bytes = read(path)?;
    let files = [source_file(SourceRole::Weights, &bytes)];
    build(&Sources {
        meta: MetaView::Gguf(&header.metadata),
        weights: Weights::Gguf(&header),
        kv_source: KvSource::Gguf,
        tokenizer: None,
        files: &files,
    })
    .map_err(|error| FixtureError::Census {
        file: path.to_owned(),
        error,
    })
}

#[cfg(test)]
mod tests;
