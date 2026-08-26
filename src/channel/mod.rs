pub mod awgn;
pub mod fading;
pub mod phase;

pub use awgn::{AwgnChannel, AwgnIter};
pub use fading::{
    BlockFading, BlockRayleigh, BlockRician, EqualizationPolicy, EqualizeExt, EqualizeIter,
    EqualizeMmseIter, EqualizeZfIter, FadedSymbol, FadingChannel, FadingCsiIter, FadingExt,
    FadingIter, FadingModel, FastFading, FastRayleigh, FastRician, Mmse, RayleighChannel,
    RayleighModel, RicianChannel, RicianModel, ZeroForcing,
};
pub use phase::{NoRng, PhaseDistortion, PhaseDistortionIter};

use crate::constellation::{Constellation, ConstellationGeometry, ConstellationState};
use num_complex::Complex;
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};

// =============================================================================
// Channel Sample Trait
// =============================================================================

pub trait ChannelSample {
    type Float: Float;

    /// Applies a complex transformation to the baseband sample.
    fn map_sample(self, f: impl FnOnce(Complex<Self::Float>) -> Complex<Self::Float>) -> Self;

    /// Returns a copy of the baseband sample.
    fn sample(&self) -> Complex<Self::Float>;
}

impl<T: Float> ChannelSample for Complex<T> {
    type Float = T;

    #[inline(always)]
    fn map_sample(self, f: impl FnOnce(Complex<T>) -> Complex<T>) -> Self {
        f(self)
    }

    #[inline(always)]
    fn sample(&self) -> Complex<T> {
        *self
    }
}

impl<T: Float> ChannelSample for FadedSymbol<T> {
    type Float = T;

    #[inline(always)]
    fn map_sample(self, f: impl FnOnce(Complex<T>) -> Complex<T>) -> Self {
        Self {
            sample: f(self.sample),
            csi: self.csi,
        }
    }

    #[inline(always)]
    fn sample(&self) -> Complex<T> {
        self.sample
    }
}

// =============================================================================
// Universal Channel Impairments Trait (Works on Complex<T> AND FadedSymbol<T>)
// =============================================================================

pub trait ChannelExt<T: Float>: Iterator + Sized
where
    Self::Item: ChannelSample<Float = T>,
{
    // --- AWGN Injections ---

    /// Injects AWGN with a specified total noise variance $\sigma^2 = N_0$.
    #[inline]
    fn add_awgn<R>(self, noise_variance: T, rng: R) -> AwgnIter<Self, T, R>
    where
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
        StandardNormal: Distribution<T>,
        R: rand_core::Rng,
    {
        AwgnIter::new(self, AwgnChannel::from_ebn0_db(constellation, ebn0_db, rng))
    }

    /// Injects AWGN targeting an SNR (in dB) with explicit symbol energy $E_s$.
    #[inline]
    fn add_awgn_snr_with_energy<R>(self, snr_db: T, es: T, rng: R) -> AwgnIter<Self, T, R>
    where
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
        StandardNormal: Distribution<T>,
        R: rand_core::Rng,
    {
        AwgnIter::new(
            self,
            AwgnChannel::from_ebn0_db_with_symbol_energy(ebn0_db, k_bits, es, rng),
        )
    }

    // --- Phase & Frequency Distortions ---

    /// Injects a constant static phase rotation of $\theta$ radians.
    #[inline]
    fn add_phase_offset(self, theta: T) -> PhaseDistortionIter<Self, T, NoRng>
    where
        StandardNormal: Distribution<T>,
    {
        PhaseDistortionIter::new(self, PhaseDistortion::static_offset(theta))
    }

    /// Injects a Carrier Frequency Offset (CFO) of $\Delta\omega$ radians per symbol.
    #[inline]
    fn add_cfo(self, delta_omega: T) -> PhaseDistortionIter<Self, T, NoRng>
    where
        StandardNormal: Distribution<T>,
    {
        PhaseDistortionIter::new(self, PhaseDistortion::from_cfo(delta_omega))
    }

    /// Injects a Carrier Frequency Offset (CFO) specified in Hertz.
    #[inline]
    fn add_cfo_hz(self, freq_offset_hz: T, sample_rate_hz: T) -> PhaseDistortionIter<Self, T, NoRng>
    where
        StandardNormal: Distribution<T>,
    {
        PhaseDistortionIter::new(
            self,
            PhaseDistortion::from_cfo_hz(freq_offset_hz, sample_rate_hz),
        )
    }

    /// Injects Wiener phase noise (random walk phase jitter).
    #[inline]
    fn add_phase_noise<R>(self, phase_noise_std: T, rng: R) -> PhaseDistortionIter<Self, T, R>
    where
        StandardNormal: Distribution<T>,
        R: rand_core::Rng,
    {
        PhaseDistortionIter::new(self, PhaseDistortion::wiener(phase_noise_std, rng))
    }

    /// Injects general composite phase distortion combining static phase, CFO, and Wiener noise.
    #[inline]
    fn add_phase_distortion<R>(
        self,
        initial_phase: T,
        phase_step: T,
        phase_noise_std: T,
        rng: R,
    ) -> PhaseDistortionIter<Self, T, R>
    where
        StandardNormal: Distribution<T>,
        R: rand_core::Rng,
    {
        PhaseDistortionIter::new(
            self,
            PhaseDistortion::new(initial_phase, phase_step, phase_noise_std, rng),
        )
    }
}

impl<T: Float, I: Iterator> ChannelExt<T> for I where I::Item: ChannelSample<Float = T> {}