pub mod awgn;
pub mod phase;

pub use awgn::{AwgnChannel, AwgnIter};
pub use phase::{NoRng, PhaseDistortion, PhaseDistortionIter};

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

    /// Injects a constant static phase rotation of $\theta$ radians.
    ///
    /// Computes the rotated baseband symbol:
    /// $$r = s \cdot e^{j\theta}$$
    ///
    /// # Parameters
    /// * `theta`: Static angular rotation in radians.
    #[inline]
    fn add_phase_offset(self, theta: T) -> PhaseDistortionIter<Self, T, NoRng>
    where
        T: Float,
        StandardNormal: Distribution<T>,
    {
        PhaseDistortionIter::new(self, PhaseDistortion::static_offset(theta))
    }

    /// Injects a Carrier Frequency Offset (CFO) of $\Delta\omega$ radians per symbol.
    ///
    /// Rotates the $k$-th transmitted symbol according to a linear phase progression:
    /// $$r[k] = s[k] \cdot e^{j(k \cdot \Delta\omega)}$$
    ///
    /// # Parameters
    /// * `delta_omega`: Normalized frequency offset $\Delta\omega = 2\pi \frac{\Delta f}{f_s}$ in radians/symbol.
    #[inline]
    fn add_cfo(self, delta_omega: T) -> PhaseDistortionIter<Self, T, NoRng>
    where
        T: Float,
        StandardNormal: Distribution<T>,
    {
        PhaseDistortionIter::new(self, PhaseDistortion::from_cfo(delta_omega))
    }

    /// Injects a Carrier Frequency Offset (CFO) specified in Hertz.
    ///
    /// Computes the normalized phase step $\Delta\omega = 2\pi \frac{\Delta f}{f_s}$ and applies linear rotation:
    /// $$r[k] = s[k] \cdot e^{j\left(2\pi \frac{\Delta f}{f_s} k\right)}$$
    ///
    /// # Parameters
    /// * `freq_offset_hz`: Carrier frequency mismatch $\Delta f$ in Hertz.
    /// * `sample_rate_hz`: Baseband symbol/sampling rate $f_s$ in Hertz.
    #[inline]
    fn add_cfo_hz(self, freq_offset_hz: T, sample_rate_hz: T) -> PhaseDistortionIter<Self, T, NoRng>
    where
        T: Float,
        StandardNormal: Distribution<T>,
    {
        PhaseDistortionIter::new(
            self,
            PhaseDistortion::from_cfo_hz(freq_offset_hz, sample_rate_hz),
        )
    }

    /// Injects Wiener phase noise (random walk phase jitter).
    ///
    /// Models oscillator phase instability as discrete Brownian motion:
    /// $$\theta[k] = \theta[k-1] + \Delta\theta_k, \quad \Delta\theta_k \sim \mathcal{N}(0, \sigma_\phi^2)$$
    /// $$r[k] = s[k] \cdot e^{j\theta[k]}$$
    ///
    /// # Parameters
    /// * `phase_noise_std`: Standard deviation of the per-symbol phase step $\sigma_\phi$ in radians.
    /// * `rng`: The PRNG instance used for sampling Gaussian phase increments.
    #[inline]
    fn add_phase_noise<R>(self, phase_noise_std: T, rng: R) -> PhaseDistortionIter<Self, T, R>
    where
        T: Float,
        StandardNormal: Distribution<T>,
        R: rand_core::Rng,
    {
        PhaseDistortionIter::new(self, PhaseDistortion::wiener(phase_noise_std, rng))
    }

    /// Injects general composite phase distortion combining static phase, CFO, and Wiener noise.
    ///
    /// Computes phase progression:
    /// $$\theta[k] = \theta[k-1] + \Delta\omega + \Delta\theta_k, \quad \theta[0] = \theta_0$$
    /// $$r[k] = s[k] \cdot e^{j\theta[k]}$$
    ///
    /// # Parameters
    /// * `initial_phase`: Initial phase offset $\theta_0$ in radians.
    /// * `phase_step`: Frequency offset ramp $\Delta\omega$ in radians/symbol.
    /// * `phase_noise_std`: Wiener step standard deviation $\sigma_\phi$ in radians.
    /// * `rng`: The PRNG instance used for sampling Gaussian phase increments.
    #[inline]
    fn add_phase_distortion<R>(
        self,
        initial_phase: T,
        phase_step: T,
        phase_noise_std: T,
        rng: R,
    ) -> PhaseDistortionIter<Self, T, R>
    where
        T: Float,
        StandardNormal: Distribution<T>,
        R: rand_core::Rng,
    {
        PhaseDistortionIter::new(
            self,
            PhaseDistortion::new(initial_phase, phase_step, phase_noise_std, rng),
        )
    }
}

// Blanket implementation for all complex iterators
impl<T, I: Iterator<Item = Complex<T>>> ChannelExt<T> for I {}
