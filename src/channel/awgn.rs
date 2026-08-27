use crate::channel::ChannelSample;
use crate::{Constellation, ConstellationGeometry, ConstellationState};
use num_complex::Complex;
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};

/// An Additive White Gaussian Noise (AWGN) channel model for complex baseband signals.
///
/// Models an unquantized, circularly symmetric complex Gaussian noise channel:
/// $$r = s + n$$
/// where $s$ is the transmitted complex symbol and $n = n_I + j n_Q$ is complex zero-mean
/// white Gaussian noise with total variance $\sigma^2 = N_0 = \mathbb{E}[|n|^2]$:
/// $$n_I, n_Q \sim \mathcal{N}\left(0, \frac{\sigma^2}{2}\right)$$
///
/// # Type Parameters
/// * `T`: Floating-point scalar type (`f32`, `f64`).
/// * `R`: Pseudo-random number generator implementing [`rand_core::Rng`].
pub struct AwgnChannel<T, R> {
    sigma_axis: T,
    rng: R,
}

/// AwgnChannel constructors
impl<T, R> AwgnChannel<T, R>
where
    T: Float,
    rand_distr::StandardNormal: rand_distr::Distribution<T>,
    R: rand_core::Rng,
{
    /// Constructs an AWGN channel with an explicit total complex noise variance $\sigma^2 = N_0$.
    ///
    /// Computes the per-axis standard deviation:
    /// $$\sigma_{\text{axis}} = \sqrt{\frac{\sigma^2}{2}}$$
    ///
    /// # Parameters
    /// * `noise_variance`: Total variance $\sigma^2 = \mathbb{E}[|n|^2] = \sigma_I^2 + \sigma_Q^2$.
    /// * `rng`: The PRNG instance used for sampling Gaussian noise.
    pub fn with_variance(noise_variance: T, rng: R) -> Self {
        let two = T::one() + T::one();
        Self {
            sigma_axis: (noise_variance / two).sqrt(),
            rng,
        }
    }
    /// Constructs an AWGN channel for a given [`Constellation`] at a specified Signal-to-Noise Ratio (SNR) in dB.
    ///
    /// Automatically extracts the nominal average symbol energy $E_s$ from the constellation:
    /// $$\text{SNR}_{\text{lin}} = 10^{\frac{\text{SNR}_{\text{dB}}}{10}}$$
    /// $$N_0 = \frac{E_s}{\text{SNR}_{\text{lin}}}$$
    /// $$\sigma_{\text{axis}} = \sqrt{\frac{N_0}{2}}$$
    ///
    /// # Parameters
    /// * `constellation`: The constellation geometry defining nominal symbol energy $E_s$.
    /// * `snr_db`: Target Signal-to-Noise Ratio ($E_s / N_0$) in decibels (dB).
    /// * `rng`: The PRNG instance used for sampling Gaussian noise.
    pub fn from_snr_db<const M: usize, S: ConstellationState, G: ConstellationGeometry>(
        constellation: &Constellation<T, M, S, G>,
        snr_db: T,
        rng: R,
    ) -> Self {
        let es = constellation.energy();
        Self::from_snr_db_with_symbol_energy(snr_db, es, rng)
    }

    /// Constructs an AWGN channel for a given [`Constellation`] at a specified $E_b / N_0$ ratio in dB.
    ///
    /// Automatically determines bits per symbol $K = \log_2(M)$ and symbol energy $E_s$
    /// to scale noise power for spectral efficiency:
    /// $$\left(\frac{E_s}{N_0}\right)_{\text{lin}} = K \cdot 10^{\frac{(E_b / N_0)_{\text{dB}}}{10}}$$
    /// $$N_0 = \frac{E_s}{(E_s / N_0)_{\text{lin}}}$$
    /// $$\sigma_{\text{axis}} = \sqrt{\frac{N_0}{2}}$$
    ///
    /// # Parameters
    /// * `constellation`: The target constellation providing bits per symbol $K$ and energy $E_s$.
    /// * `ebn0_db`: Energy per bit to noise power spectral density ratio ($E_b / N_0$) in decibels (dB).
    /// * `rng`: The PRNG instance used for sampling Gaussian noise.
    pub fn from_ebn0_db<const M: usize, S: ConstellationState, G: ConstellationGeometry>(
        constellation: &Constellation<T, M, S, G>,
        ebn0_db: T,
        rng: R,
    ) -> Self {
        let k_bits = Constellation::<T, M, S, G>::BITS_PER_SYMBOL;
        Self::from_ebn0_db_with_symbol_energy(ebn0_db, k_bits, constellation.energy(), rng)
    }

    /// Constructs an AWGN channel with an explicit symbol energy $E_s$ and target SNR in dB.
    ///
    /// Useful for arbitrary continuous waveforms, unconstrained signal streams, or pre-scaled
    /// custom symbols where a formal [`Constellation`] instance is not available.
    ///
    /// # Parameters
    /// * `snr_db`: Target Signal-to-Noise Ratio ($E_s / N_0$) in decibels (dB).
    /// * `es`: Nominal average symbol energy $E_s = \mathbb{E}[|s|^2]$.
    /// * `rng`: The PRNG instance used for sampling Gaussian noise.
    pub fn from_snr_db_with_symbol_energy(snr_db: T, es: T, rng: R) -> Self {
        let two = T::one() + T::one();
        let ten = two + two + two + two + two;
        let snr_lin = ten.powf(snr_db.div(ten));
        let noise_variance = es / snr_lin;

        Self {
            sigma_axis: (noise_variance / two).sqrt(),
            rng,
        }
    }
    /// Constructs an AWGN channel with explicit bit-depth $K$, symbol energy $E_s$, and target $E_b / N_0$ in dB.
    ///
    /// Useful for coded modulation systems where FEC code rate $R_c$ alters the effective spectral
    /// efficiency ($K_{\text{eff}} = K \cdot R_c$).
    ///
    /// # Parameters
    /// * `ebn0_db`: Energy per information bit to noise spectral density ratio ($E_b / N_0$) in decibels (dB).
    /// * `k_bits`: Effective number of bits per symbol $K$.
    /// * `es`: Nominal average symbol energy $E_s = \mathbb{E}[|s|^2]$.
    /// * `rng`: The PRNG instance used for sampling Gaussian noise.
    ///
    /// # Panics
    /// Panics if `k_bits` cannot be cast to target floating-point type `T`.
    pub fn from_ebn0_db_with_symbol_energy(ebn0_db: T, k_bits: usize, es: T, rng: R) -> Self {
        let two = T::one() + T::one();
        let ten = two + two + two + two + two;

        let esn0_lin =
            T::from(k_bits).expect("Valid u32 conversion to float") * ten.powf(ebn0_db.div(ten));

        let noise_variance = es / esn0_lin;

        Self {
            sigma_axis: (noise_variance / two).sqrt(),
            rng,
        }
    }
}

impl<T, R> AwgnChannel<T, R>
where
    T: Float,
    rand_distr::StandardNormal: rand_distr::Distribution<T>,
    R: rand_core::Rng,
{
    /// Samples a single circularly symmetric complex Gaussian noise point.
    #[inline]
    fn sample_noise(&mut self) -> Complex<T> {
        let z_i: T = StandardNormal.sample(&mut self.rng);
        let z_q: T = StandardNormal.sample(&mut self.rng);

        Complex::new(z_i * self.sigma_axis, z_q * self.sigma_axis)
    }

    /// Adds complex Gaussian noise directly to a transmitted symbol.
    #[inline]
    pub fn apply_point(&mut self, point: Complex<T>) -> Complex<T> {
        point + self.sample_noise()
    }
}

pub struct AwgnIter<I, T, R> {
    iter: I,
    channel: AwgnChannel<T, R>,
}

impl<I, T, R> AwgnIter<I, T, R> {
    pub fn new(iter: I, channel: AwgnChannel<T, R>) -> AwgnIter<I, T, R>
    where
        R: rand_core::Rng,
        T: Float,
    {
        Self { iter, channel }
    }
}

impl<I, T, R> Iterator for AwgnIter<I, T, R>
where
    I: Iterator,
    I::Item: ChannelSample<Float = T>,
    T: Float,
    StandardNormal: Distribution<T>,
    R: rand_core::Rng,
{
    type Item = I::Item; // <-- Change from Complex<T> to I::Item

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let item = self.iter.next()?;
        Some(item.map_sample(|pt| self.channel.apply_point(pt)))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<I, T, R> ExactSizeIterator for AwgnIter<I, T, R>
where
    I: ExactSizeIterator,
    I::Item: ChannelSample<Float = T>,
    T: Float,
    StandardNormal: Distribution<T>,
    R: rand_core::Rng,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::ChannelExt;
    use crate::constellation::{Bpsk, Qam16, Qpsk};
    use crate::demodulation::DemodulateExt;
    use crate::modulation::ModulateExt;
    use alloc::vec;
    use alloc::vec::Vec;
    use rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_awgn_zero_mean_and_variance() {
        let rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let target_variance = 0.5f64; // N0 = 0.5, sigma_axis^2 = 0.25, sigma_axis = 0.5
        let mut channel = AwgnChannel::with_variance(target_variance, rng);

        let num_samples = 100_000;
        let mut sum_i = 0.0f64;
        let mut sum_q = 0.0f64;
        let mut sum_sq_i = 0.0f64;
        let mut sum_sq_q = 0.0f64;

        for _ in 0..num_samples {
            let noise = channel.sample_noise();
            sum_i += noise.re;
            sum_q += noise.im;
            sum_sq_i += noise.re * noise.re;
            sum_sq_q += noise.im * noise.im;
        }

        let mean_i = sum_i / (num_samples as f64);
        let mean_q = sum_q / (num_samples as f64);
        let var_i = (sum_sq_i / (num_samples as f64)) - (mean_i * mean_i);
        let var_q = (sum_sq_q / (num_samples as f64)) - (mean_q * mean_q);
        let total_var = var_i + var_q;

        // 1. Verify zero mean (within 3 standard errors: 3 * (sigma / sqrt(N)) approx 0.005)
        assert!(mean_i.abs() < 0.01, "Mean I was {mean_i}, expected ~0.0");
        assert!(mean_q.abs() < 0.01, "Mean Q was {mean_q}, expected ~0.0");

        // 2. Verify variance per axis (N0 / 2 = 0.25) and total variance (N0 = 0.5)
        assert!(
            (var_i - 0.25).abs() < 0.015,
            "Var I was {var_i}, expected ~0.25"
        );
        assert!(
            (var_q - 0.25).abs() < 0.015,
            "Var Q was {var_q}, expected ~0.25"
        );
        assert!(
            (total_var - target_variance).abs() < 0.02,
            "Total Var was {total_var}, expected ~{target_variance}"
        );
    }

    #[test]
    fn test_awgn_high_snr_zero_ber() {
        let payload = vec![0x55, 0xAA, 0x12, 0x34, 0xDE, 0xAD, 0xBE, 0xEF];
        let qpsk = Qpsk::<f32>::QPSK;
        let rng = Xoshiro256PlusPlus::seed_from_u64(42);

        // At 30 dB SNR, error probability is practically 0
        let recovered: Vec<u8> = payload
            .clone()
            .into_iter()
            .modulate(&qpsk)
            .add_awgn_snr(&qpsk, 30.0, rng)
            .demodulate_hard(&qpsk)
            .collect();

        assert_eq!(recovered, payload);
    }

    #[test]
    fn test_bpsk_ber_curve_against_theory() {
        let bpsk = Bpsk::<f64>::BPSK;
        let num_bytes = 10_000;
        let total_bits = (num_bytes * 8) as f64;

        // Generate deterministic payload
        let payload: Vec<u8> = (0..num_bytes).map(|i| (i * 137) as u8).collect();

        // 1. Test at Eb/N0 = 0 dB (Theoretical Pb = Q(sqrt(2)) ≈ 0.07865)
        let rng_0db = Xoshiro256PlusPlus::seed_from_u64(42);
        let rx_0db: Vec<u8> = payload
            .clone()
            .into_iter()
            .modulate(&bpsk)
            .add_awgn_ebn0(&bpsk, 0.0, rng_0db)
            .demodulate_hard(&bpsk)
            .collect();

        let bit_errors_0db: usize = payload
            .iter()
            .zip(rx_0db.iter())
            .map(|(&a, &b)| (a ^ b).count_ones() as usize)
            .sum();
        let ber_0db = (bit_errors_0db as f64) / total_bits;

        // Empirical BER should be within [6.5%, 9.5%]
        assert!(
            (ber_0db - 0.07865).abs() < 0.015,
            "BER at 0 dB was {ber_0db}, expected ~0.07865"
        );

        // 2. Test at Eb/N0 = 6.8 dB (Theoretical Pb = Q(sqrt(2 * 4.786)) ≈ 1.0e-3)
        let rng_7db = Xoshiro256PlusPlus::seed_from_u64(84);
        let rx_7db: Vec<u8> = payload
            .clone()
            .into_iter()
            .modulate(&bpsk)
            .add_awgn_ebn0(&bpsk, 6.8, rng_7db)
            .demodulate_hard(&bpsk)
            .collect();

        let bit_errors_7db: usize = payload
            .iter()
            .zip(rx_7db.iter())
            .map(|(&a, &b)| (a ^ b).count_ones() as usize)
            .sum();
        let ber_7db = (bit_errors_7db as f64) / total_bits;

        // Empirical BER should be within [0.0005, 0.0025]
        assert!(
            (ber_7db - 0.001).abs() < 0.0015,
            "BER at 6.8 dB was {ber_7db}, expected ~0.001"
        );
    }
    #[test]
    fn test_qam16_awgn_high_snr_zero_ber() {
        let payload: Vec<u8> = (0..256).map(|x| (x * 47) as u8).collect();
        let qam16 = Qam16::<f32>::QAM16;
        let rng = Xoshiro256PlusPlus::seed_from_u64(1234);

        // At 30 dB SNR, 16-QAM should have zero bit errors
        let recovered: Vec<u8> = payload
            .clone()
            .into_iter()
            .modulate(&qam16)
            .add_awgn_snr(&qam16, 30.0, rng)
            .demodulate_hard(&qam16)
            .collect();

        assert_eq!(recovered, payload);
    }

    #[test]
    fn test_qam16_ber_curve_against_theory() {
        let qam16 = Qam16::<f64>::QAM16;
        let num_bytes = 20_000;
        let total_bits = (num_bytes * 8) as f64;

        // Generate deterministic test payload
        let payload: Vec<u8> = (0..num_bytes).map(|i| (i * 179 + 11) as u8).collect();

        // ---------------------------------------------------------------------
        // 1. Low Eb/N0: 4.0 dB
        // Theoretical Gray-coded 16-QAM bit error probability:
        // Pb ≈ (3/4) * Q(sqrt(0.8 * Eb/N0_lin)) ≈ 0.0586 (5.86%)
        // ---------------------------------------------------------------------
        let rng_4db = Xoshiro256PlusPlus::seed_from_u64(4242);
        let rx_4db: Vec<u8> = payload
            .clone()
            .into_iter()
            .modulate(&qam16)
            .add_awgn_ebn0(&qam16, 4.0, rng_4db)
            .demodulate_hard(&qam16)
            .collect();

        let bit_errors_4db: usize = payload
            .iter()
            .zip(rx_4db.iter())
            .map(|(&a, &b)| (a ^ b).count_ones() as usize)
            .sum();
        let ber_4db = (bit_errors_4db as f64) / total_bits;

        assert!(
            (ber_4db - 0.0586).abs() < 0.010,
            "16-QAM BER at 4.0 dB was {ber_4db:.5}, expected ~0.0586"
        );

        // ---------------------------------------------------------------------
        // 2. Medium Eb/N0: 8.0 dB
        // Theoretical Gray-coded 16-QAM bit error probability:
        // Pb ≈ (3/4) * Q(sqrt(0.8 * 6.30957)) ≈ 0.00925 (0.925%)
        // ---------------------------------------------------------------------
        let rng_8db = Xoshiro256PlusPlus::seed_from_u64(8484);
        let rx_8db: Vec<u8> = payload
            .clone()
            .into_iter()
            .modulate(&qam16)
            .add_awgn_ebn0(&qam16, 8.0, rng_8db)
            .demodulate_hard(&qam16)
            .collect();

        let bit_errors_8db: usize = payload
            .iter()
            .zip(rx_8db.iter())
            .map(|(&a, &b)| (a ^ b).count_ones() as usize)
            .sum();
        let ber_8db = (bit_errors_8db as f64) / total_bits;

        assert!(
            (ber_8db - 0.00925).abs() < 0.0025,
            "16-QAM BER at 8.0 dB was {ber_8db:.5}, expected ~0.00925"
        );
    }

    /// Rational polynomial approximation of the standard Gaussian Q-function.
    fn q_function(x: f64) -> f64 {
        if x < 0.0 {
            return 1.0 - q_function(-x);
        }
        let t = 1.0 / (1.0 + 0.2316419 * x);
        let poly = t
            * (0.319381530
                + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
        let inv_sqrt_2pi = 0.3989422804014327; // 1 / sqrt(2 * PI)
        let gauss = (-0.5 * x * x).exp() * inv_sqrt_2pi;
        gauss * poly
    }

    #[test]
    fn test_bpsk_ber_curve_fine_0_1db_sweep() {
        let bpsk = Bpsk::<f64>::BPSK;
        let num_bytes = 8_000; // 64,000 bits per step
        let total_bits = (num_bytes * 8) as f64;

        // Generate deterministic pseudo-random payload
        let payload: Vec<u8> = (0..num_bytes).map(|i| (i * 157 + 31) as u8).collect();

        // Sweep from 0.0 dB to 10.0 dB in 0.1 dB increments (101 evaluation points)
        for step in 0..=100 {
            let ebn0_db = (step as f64) * 0.1;
            let ebn0_lin = 10.0f64.powf(ebn0_db / 10.0);

            // Theoretical BPSK error probability: Pb = Q(sqrt(2 * Eb/N0))
            let theoretical_ber = q_function((2.0 * ebn0_lin).sqrt());

            // Run end-to-end pipeline with unique deterministic seed per step
            let rng = Xoshiro256PlusPlus::seed_from_u64(0xBEEF_0000 + step as u64);
            let rx_bytes: Vec<u8> = payload
                .clone()
                .into_iter()
                .modulate(&bpsk)
                .add_awgn_ebn0(&bpsk, ebn0_db, rng)
                .demodulate_hard(&bpsk)
                .collect();

            // Count bit errors
            let bit_errors: usize = payload
                .iter()
                .zip(rx_bytes.iter())
                .map(|(&tx, &rx)| (tx ^ rx).count_ones() as usize)
                .sum();
            let empirical_ber = (bit_errors as f64) / total_bits;

            // Binomial standard deviation: sigma = sqrt(p * (1 - p) / N)
            let std_err = (theoretical_ber * (1.0 - theoretical_ber) / total_bits).sqrt();
            let margin = 4.0 * std_err + 0.0005; // 4-sigma envelope + small base floor

            let delta = (empirical_ber - theoretical_ber).abs();
            assert!(
                delta <= margin,
                "Failed at Eb/N0 = {ebn0_db:.1} dB: Empirical BER = {empirical_ber:.5}, \
                 Theoretical = {theoretical_ber:.5}, Delta = {delta:.5} (allowed margin = {margin:.5})"
            );
        }
    }
}
