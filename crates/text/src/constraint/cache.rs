// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compiled constraints cached by BLAKE3(constraint, tokenizer) (RFC-0050 §7.3, ADR-0504
//! decision 1).

use std::collections::BTreeMap;

use super::{CompileOutput, ConstraintEngine, ConstraintError, OutputConstraint, Vocabulary};

/// A cache key: BLAKE3 over the constraint and the vocabulary's digest.
pub type ConstraintKey = [u8; 32];

/// The key of `constraint` compiled for `vocabulary`.
///
/// The constraint is hashed as its JSON form. A schema's objects keep their document order
/// (`serde_json`'s `preserve_order`), so two schemas that list the same properties in a different
/// order get different keys, as they should: property order is the order of the output.
pub fn key(constraint: &OutputConstraint, vocabulary: &Vocabulary) -> ConstraintKey {
    let encoded = serde_json::to_vec(constraint).expect("a constraint always serialises");
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"syntropy-text constraint v1\0");
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    hasher.update(vocabulary.digest());
    *hasher.finalize().as_bytes()
}

/// Compiled constraints by key, the least recently used dropped first when full. A failed compile
/// is not cached.
pub struct CompileCache<C> {
    capacity: usize,
    entries: BTreeMap<ConstraintKey, Entry<C>>,
    clock: u64,
    hits: u64,
    misses: u64,
}

struct Entry<C> {
    output: CompileOutput<C>,
    used: u64,
}

impl<C> CompileCache<C> {
    /// A cache holding at most `capacity` compiled constraints, at least one.
    pub fn new(capacity: usize) -> CompileCache<C> {
        CompileCache {
            capacity: capacity.max(1),
            entries: BTreeMap::new(),
            clock: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// `constraint` compiled by `engine`, from the cache when it holds it.
    pub fn get_or_compile<E>(
        &mut self,
        engine: &E,
        constraint: &OutputConstraint,
    ) -> Result<CompileOutput<C>, ConstraintError>
    where
        E: ConstraintEngine<Compiled = C>,
    {
        self.clock += 1;
        let key = key(constraint, engine.vocabulary());
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.used = self.clock;
            self.hits += 1;
            return Ok(entry.output.clone());
        }
        self.misses += 1;
        let output = engine.compile(constraint)?;
        if self.entries.len() >= self.capacity
            && let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.used)
                .map(|(key, _)| *key)
        {
            self.entries.remove(&oldest);
        }
        self.entries.insert(
            key,
            Entry {
                output: output.clone(),
                used: self.clock,
            },
        );
        Ok(output)
    }

    /// How many compiled constraints it holds.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether it holds none.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Lookups answered from the cache, and lookups that compiled.
    pub fn hits_and_misses(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }
}
