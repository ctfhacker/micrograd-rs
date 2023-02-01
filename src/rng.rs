/// Rng seeded with `rdtsc` that is generated using Lehmer64
#[derive(Debug)]
pub struct Rng {
    /// Internal state
    value: u128,
}

impl std::default::Default for Rng {
    fn default() -> Self {
        Self::new()
    }
}

impl rand::RngCore for Rng {
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
    fn next_u64(&mut self) -> u64 {
        self.next_u64()
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        dest.iter_mut().for_each(|x| *x = self.next_u64() as u8);
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl Rng {
    /// Create a new `Lehmer64` rng seeded by `rdtsc`
    pub fn new() -> Rng {
        let mut res = Self {
            value: u128::from(unsafe { core::arch::x86_64::_rdtsc() }),
        };

        // Cycle through to create some chaos
        for _ in 0..123 {
            let _ = res.next_u64();
        }

        res
    }

    /// Get the next random number
    #[allow(clippy::cast_possible_truncation)]
    pub fn next_u64(&mut self) -> u64 {
        self.value = self.value.wrapping_mul(0xda94_2042_e4dd_58b5);
        (self.value >> 64) as u64
    }
}
