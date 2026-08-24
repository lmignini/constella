pub mod awgn;
pub mod phase;

pub use awgn::{AwgnChannel, AwgnIter};

use crate::constellation::{Constellation, ConstellationGeometry, ConstellationState};
use num_complex::Complex;
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};

pub trait ChannelExt<T>: Iterator<Item = Complex<T>> + Sized {
    /// Injects AWGN with a specified total noise variance $\sigma^2 = N_0$.
    #[inline]
    fn add_awgn<R>(self, noise_variance: T, rng: R) -> AwgnIter<Self, T, R>
    where
        T: Float,
        StandardNormal: Distribution<T>,
        R: rand_core::Rng,
    {
        AwgnIter::new(self, AwgnChannel::with_variance(noise_variance, rng))
    }

    /// Injects AWGN targeting an SNR (in dB) derived from a constellation instance.
    #[inline]
    fn add_awgn_snr<const M: usize, S: ConstellationState, G: ConstellationGeometry, R>(
        self,
        constellation: &Constellation<T, M, S, G>,
        snr_db: T,
        rng: R,
    ) -> AwgnIter<Self, T, R>
    where
        T: Float,
        StandardNormal: Distribution<T>,
        R: rand_core::Rng,
    {
        AwgnIter::new(self, AwgnChannel::from_snr_db(constellation, snr_db, rng))
    }

    /// Injects AWGN targeting an $E_b / N_0$ (in dB) derived from a constellation instance.
    #[inline]
    fn add_awgn_ebn0<const M: usize, S: ConstellationState, G: ConstellationGeometry, R>(
        self,
        constellation: &Constellation<T, M, S, G>,
        ebn0_db: T,
        rng: R,
    ) -> AwgnIter<Self, T, R>
    where
        T: Float,
        StandardNormal: Distribution<T>,
        R: rand_core::Rng,
    {
        AwgnIter::new(self, AwgnChannel::from_ebn0_db(constellation, ebn0_db, rng))
    }

    /// Injects AWGN targeting an SNR (in dB) with explicit symbol energy $E_s$.
    #[inline]
    fn add_awgn_snr_with_energy<R>(self, snr_db: T, es: T, rng: R) -> AwgnIter<Self, T, R>
    where
        T: Float,
        StandardNormal: Distribution<T>,
        R: rand_core::Rng,
    {
        AwgnIter::new(
            self,
            AwgnChannel::from_snr_db_with_symbol_energy(snr_db, es, rng),
        )
    }

    /// Injects AWGN targeting an $E_b / N_0$ (in dB) with explicit bits-per-symbol and symbol energy.
    #[inline]
    fn add_awgn_ebn0_with_energy<R>(
        self,
        ebn0_db: T,
        k_bits: usize,
        es: T,
        rng: R,
    ) -> AwgnIter<Self, T, R>
    where
        T: Float,
        StandardNormal: Distribution<T>,
        R: rand_core::Rng,
    {
        AwgnIter::new(
            self,
            AwgnChannel::from_ebn0_db_with_symbol_energy(ebn0_db, k_bits, es, rng),
        )
    }
}

// Blanket implementation for all complex iterators
impl<T, I: Iterator<Item = Complex<T>>> ChannelExt<T> for I {}
