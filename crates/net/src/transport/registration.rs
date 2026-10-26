// SPDX-License-Identifier: MIT OR Apache-2.0

//! The registration tables of RFC-0031 §8: which pools are registered, with which transports and
//! kernel providers, and their handles.
//!
//! RFC-0031 §8 registers each pool once with every transport that needs it, and with the kernel
//! provider so the GPU can DMA to and from it (`KernelProvider::register_host`, RFC-0001 §5.7).
//! A [`RegistrationTable`] holds **one registration per pool per registrant**:
//! - registering the same pool again with the same registrant returns its one registration;
//! - a different pool under a `PoolId` the table already holds is refused, since a channel spec
//!   names its receive pool by id.
//!
//! Each transport keeps a table and registers its pools through it: [`Transport::register`]
//! (QUIC and TCP, which need no backend handle, give 0). A runner records a provider's
//! registration in the same table with [`Registrant::Provider`], under the provider's name, since
//! `net` cannot depend on `kernels` (RFC-0001 §4.2). The table is control-path state, behind a
//! lock no frame takes; a channel takes the pools it needs when it opens
//! ([`RegistrationTable::pools`]).
//!
//! [`Transport::register`]: super::Transport::register

use std::sync::{Mutex, MutexGuard};

use syntropy_core::ErrorClass;
use syntropy_core::resource::TransportKind;

use super::buffer::{BufferPool, PoolId, PoolRegistration};
use crate::error::Error;

/// Who a pool is registered with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Registrant {
    /// A transport backend: RDMA memory regions, shared-memory segments, or the bookkeeping
    /// QUIC and TCP keep.
    Transport(TransportKind),
    /// A kernel provider, by its `ProviderId`'s name, such as `"cuda"`: `register_host`, so the
    /// device can DMA to and from the pool's slots.
    Provider(&'static str),
}

/// One pool's registration with one registrant.
#[derive(Clone, Debug)]
pub struct Entry {
    pub pool: BufferPool,
    pub registrant: Registrant,
    /// The registrant's handle: an RDMA memory region's key, a provider's registration; 0 where
    /// the registrant needs none.
    pub handle: u64,
}

/// Pools and their registrations, one per pool per registrant. See the [module](self).
#[derive(Debug, Default)]
pub struct RegistrationTable {
    entries: Mutex<Vec<Entry>>,
}

fn lock(entries: &Mutex<Vec<Entry>>) -> MutexGuard<'_, Vec<Entry>> {
    entries
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl RegistrationTable {
    /// An empty table.
    pub fn new() -> RegistrationTable {
        RegistrationTable::default()
    }

    /// Registers `pool` with `registrant` under `handle`, or returns the registration it already
    /// has with that registrant.
    ///
    /// # Errors
    ///
    /// `InvalidRequest` when another pool holds `pool`'s id in this table.
    pub fn register(
        &self,
        pool: &BufferPool,
        registrant: Registrant,
        handle: u64,
    ) -> Result<Entry, Error> {
        let mut entries = lock(&self.entries);
        if let Some(other) = entries
            .iter()
            .find(|entry| entry.pool.id() == pool.id() && !entry.pool.same_pool(pool))
        {
            return Err(Error::uncoded(
                ErrorClass::InvalidRequest,
                format!(
                    "pool {:?} is already registered as another pool (with {:?})",
                    pool.id(),
                    other.registrant
                ),
            ));
        }
        if let Some(entry) = entries
            .iter()
            .find(|entry| entry.pool.same_pool(pool) && entry.registrant == registrant)
        {
            return Ok(entry.clone());
        }
        let entry = Entry {
            pool: pool.clone(),
            registrant,
            handle,
        };
        entries.push(entry.clone());
        Ok(entry)
    }

    /// [`RegistrationTable::register`] for a transport, as `Transport::register` returns it.
    ///
    /// # Errors
    ///
    /// As [`RegistrationTable::register`].
    pub fn register_transport(
        &self,
        pool: &BufferPool,
        kind: TransportKind,
        handle: u64,
    ) -> Result<PoolRegistration, Error> {
        let entry = self.register(pool, Registrant::Transport(kind), handle)?;
        Ok(PoolRegistration {
            pool: entry.pool.id(),
            transport: kind,
            handle: entry.handle,
        })
    }

    /// The pool registered under `id`, with any registrant.
    pub fn pool(&self, id: PoolId) -> Option<BufferPool> {
        lock(&self.entries)
            .iter()
            .find(|entry| entry.pool.id() == id)
            .map(|entry| entry.pool.clone())
    }

    /// The pool `registration` names, if it is registered here with that transport.
    pub fn resolve(&self, registration: PoolRegistration) -> Option<BufferPool> {
        lock(&self.entries)
            .iter()
            .find(|entry| {
                entry.pool.id() == registration.pool
                    && entry.registrant == Registrant::Transport(registration.transport)
            })
            .map(|entry| entry.pool.clone())
    }

    /// Every pool registered here, each once.
    pub fn pools(&self) -> Vec<BufferPool> {
        let entries = lock(&self.entries);
        let mut pools: Vec<BufferPool> = Vec::new();
        for entry in entries.iter() {
            if !pools.iter().any(|pool| pool.same_pool(&entry.pool)) {
                pools.push(entry.pool.clone());
            }
        }
        pools
    }

    /// The registrants `id` is registered with, in registration order.
    pub fn registrants(&self, id: PoolId) -> Vec<Registrant> {
        lock(&self.entries)
            .iter()
            .filter(|entry| entry.pool.id() == id)
            .map(|entry| entry.registrant)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(id: u32) -> BufferPool {
        BufferPool::heap(PoolId(id), 64, 4).expect("a pool")
    }

    #[test]
    fn one_registration_per_pool_per_registrant() {
        let table = RegistrationTable::new();
        let decode = pool(1);
        let first = table
            .register_transport(&decode, TransportKind::Quic, 0)
            .unwrap();
        // The same pool again with the same transport: the one registration, not a second.
        let again = table
            .register_transport(&decode, TransportKind::Quic, 9)
            .unwrap();
        assert_eq!(first, again);
        assert_eq!(again.handle, 0, "the first registration's handle stands");
        // Another transport and a provider each get their own.
        table
            .register_transport(&decode, TransportKind::Tcp, 0)
            .unwrap();
        table
            .register(&decode, Registrant::Provider("cuda"), 0x51)
            .unwrap();
        table
            .register(&decode, Registrant::Provider("cuda"), 0x52)
            .unwrap();
        assert_eq!(
            table.registrants(PoolId(1)),
            [
                Registrant::Transport(TransportKind::Quic),
                Registrant::Transport(TransportKind::Tcp),
                Registrant::Provider("cuda"),
            ]
        );
        assert_eq!(table.pools().len(), 1);
    }

    #[test]
    fn another_pool_under_a_registered_id_is_refused() {
        let table = RegistrationTable::new();
        table
            .register_transport(&pool(3), TransportKind::Quic, 0)
            .unwrap();
        let clash = table
            .register_transport(&pool(3), TransportKind::Quic, 0)
            .unwrap_err();
        assert_eq!(clash.class, ErrorClass::InvalidRequest);
        let other_transport = table
            .register_transport(&pool(3), TransportKind::Tcp, 0)
            .unwrap_err();
        assert_eq!(other_transport.class, ErrorClass::InvalidRequest);
    }

    #[test]
    fn a_registration_resolves_only_with_its_transport() {
        let table = RegistrationTable::new();
        let prefill = pool(4);
        let registration = table
            .register_transport(&prefill, TransportKind::Quic, 0)
            .unwrap();
        assert!(table.resolve(registration).unwrap().same_pool(&prefill));
        let as_tcp = PoolRegistration {
            transport: TransportKind::Tcp,
            ..registration
        };
        assert!(table.resolve(as_tcp).is_none());
        assert!(table.pool(PoolId(4)).unwrap().same_pool(&prefill));
        assert!(table.pool(PoolId(5)).is_none());
    }
}
