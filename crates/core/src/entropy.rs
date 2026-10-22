// SPDX-License-Identifier: MIT OR Apache-2.0

//! Injected randomness (RFC-0001 §5.14, RFC-0060 §2.1, ADR-0600 decision 2). Owner: RFC-0060.
//!
//! Decision-making code reads randomness only through an [`Entropy`] it is given (rule T1), so a
//! simulator seed replays one run exactly. Key material always comes from the production
//! `OsEntropy` (rule T2), which lives in `syntropy_io::os` (ADR-0022). The simulator's seeded
//! source lives only in `testkit`, so no production build can use it.

/// A source of random bytes, passed to the code that reads it (rule T1).
pub trait Entropy: Send + Sync {
    /// Fills `buf` with random bytes.
    fn fill(&self, buf: &mut [u8]);

    /// A random `u64`: eight bytes from [`Entropy::fill`], read little-endian, so a seeded source
    /// gives the same value on every host.
    fn u64(&self) -> u64 {
        let mut bytes = [0u8; 8];
        self.fill(&mut bytes);
        u64::from_le_bytes(bytes)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    /// Bytes 0, 1, 2, ... in order, wrapping at 255: a stand-in for a seeded stream.
    struct Counting(Mutex<u8>);

    impl Entropy for Counting {
        fn fill(&self, buf: &mut [u8]) {
            let mut next = self.0.lock().unwrap();
            for byte in buf {
                *byte = *next;
                *next = next.wrapping_add(1);
            }
        }
    }

    /// Code under rule T1 takes its entropy source as a parameter.
    fn pick(entropy: &impl Entropy, choices: &[&'static str]) -> &'static str {
        choices[(entropy.u64() % choices.len() as u64) as usize]
    }

    #[test]
    fn fill_takes_bytes_from_the_source_in_order() {
        let entropy = Counting(Mutex::new(0));
        let mut first = [0u8; 3];
        let mut second = [0u8; 2];
        entropy.fill(&mut first);
        entropy.fill(&mut second);
        assert_eq!(first, [0, 1, 2]);
        assert_eq!(second, [3, 4]);
    }

    #[test]
    fn u64_reads_eight_bytes_little_endian() {
        let entropy = Counting(Mutex::new(0));
        assert_eq!(entropy.u64(), 0x0706_0504_0302_0100);
        assert_eq!(entropy.u64(), 0x0f0e_0d0c_0b0a_0908);
    }

    #[test]
    fn the_same_stream_gives_the_same_decisions() {
        let choices = ["a", "b", "c"];
        let run = || {
            let entropy = Counting(Mutex::new(17));
            (0..5).map(|_| pick(&entropy, &choices)).collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn a_source_can_be_used_as_a_trait_object() {
        fn send_sync<T: Send + Sync + ?Sized>() {}
        send_sync::<dyn Entropy>();
        let entropy: Box<dyn Entropy> = Box::new(Counting(Mutex::new(8)));
        assert_eq!(entropy.u64(), 0x0f0e_0d0c_0b0a_0908);
    }
}
