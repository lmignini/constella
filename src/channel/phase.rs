use crate::channel::ChannelSample;
use core::convert::Infallible;
use num_complex::Complex;
use num_traits::Float;
use rand_core::TryRng;
use rand_distr::{Distribution, StandardNormal};

#[derive(Debug, Clone, Copy, Default)]
pub struct NoRng;

impl TryRng for NoRng {
    type Error = Infallible;

    #[inline(always)]
    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(0)
    }

    #[inline(always)]
    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        Ok(0)
    }

    #[inline(always)]
    fn try_fill_bytes(&mut self, _dest: &mut [u8]) -> Result<(), Self::Error> {
        Ok(())
    }
}

pub struct PhaseDistortion<T, R = NoRng> {
    phasor: Complex<T>,
    step_phasor: Complex<T>,
    phase_noise_std: T,
    sample_count: usize,
    rng: R,
}

// Deterministic constructors (default R = NoRng)
impl<T: Float> PhaseDistortion<T, NoRng> {
    /// Creates a static phase rotation of `theta` radians.
    pub fn static_offset(theta: T) -> Self {
        Self {
            phasor: Complex::new(theta.cos(), theta.sin()),
            step_phasor: Complex::new(T::one(), T::zero()),
            phase_noise_std: T::zero(),
            sample_count: 0,
            rng: NoRng,
        }
    }

    /// Creates a deterministic Carrier Frequency Offset of `delta_omega` radians per symbol.
    pub fn from_cfo(delta_omega: T) -> Self {
        Self {
            phasor: Complex::new(T::one(), T::zero()),
            step_phasor: Complex::new(delta_omega.cos(), delta_omega.sin()),
            phase_noise_std: T::zero(),
            sample_count: 0,
            rng: NoRng,
        }
    }

    /// Creates a deterministic Carrier Frequency Offset from frequency offset and symbol rate in Hz.
    pub fn from_cfo_hz(freq_offset_hz: T, sample_rate_hz: T) -> Self {
        let two_pi = T::from(2.0 * core::f64::consts::PI).unwrap();
        let delta_omega = two_pi * (freq_offset_hz / sample_rate_hz);
        Self::from_cfo(delta_omega)
    }
}

// Stochastic constructors (generic R)
impl<T: Float, R: rand_core::Rng> PhaseDistortion<T, R> {
    /// Creates pure Wiener phase noise (Brownian random walk) with step standard deviation `phase_noise_std`.
    pub fn wiener(phase_noise_std: T, rng: R) -> Self {
        Self {
            phasor: Complex::new(T::one(), T::zero()),
            step_phasor: Complex::new(T::one(), T::zero()),
            phase_noise_std,
            sample_count: 0,
            rng,
        }
    }

    /// Universal constructor for combined static phase, CFO, and Wiener phase noise.
    pub fn new(initial_phase: T, phase_step: T, phase_noise_std: T, rng: R) -> Self {
        Self {
            phasor: Complex::new(initial_phase.cos(), initial_phase.sin()),
            step_phasor: Complex::new(phase_step.cos(), phase_step.sin()),
            phase_noise_std,
            sample_count: 0,
            rng,
        }
    }
}

impl<T, R> PhaseDistortion<T, R>
where
    T: Float,
    StandardNormal: Distribution<T>,
    R: rand_core::Rng,
{
    // In src/channel/phase.rs:

    #[inline(always)]
    pub fn apply_point(&mut self, point: Complex<T>) -> Complex<T> {
        let rotated_point = point * self.phasor;

        // Fast-path: Static phase rotation requires no update or re-normalization
        if self.step_phasor.re == T::one()
            && self.step_phasor.im == T::zero()
            && self.phase_noise_std <= T::zero()
        {
            return rotated_point;
        }

        self.phasor = self.phasor * self.step_phasor;

        if self.phase_noise_std > T::zero() {
            let z: T = StandardNormal.sample(&mut self.rng);
            let d_theta = self.phase_noise_std * z;
            // Fast first-order small-angle approximation e^{j*d} ≈ (1, d) for σ < 0.1 rad.
            // Magnitude growth is absorbed by the periodic normalization below.
            let threshold = T::from(0.1).unwrap();
            let jitter_phasor = if self.phase_noise_std < threshold {
                Complex::new(T::one(), d_theta)
            } else {
                Complex::new(d_theta.cos(), d_theta.sin())
            };
            self.phasor = self.phasor * jitter_phasor;
        }

        self.sample_count = self.sample_count.wrapping_add(1);
        if self.sample_count & 0x3FF == 0 {
            let norm = (self.phasor.re * self.phasor.re + self.phasor.im * self.phasor.im).sqrt();
            self.phasor.re = self.phasor.re / norm;
            self.phasor.im = self.phasor.im / norm;
        }

        rotated_point
    }
}

pub struct PhaseDistortionIter<I, T, R = NoRng> {
    iter: I,
    distortion: PhaseDistortion<T, R>,
}

impl<I, T, R> PhaseDistortionIter<I, T, R> {
    #[inline]
    pub fn new(iter: I, distortion: PhaseDistortion<T, R>) -> Self {
        Self { iter, distortion }
    }
}

impl<I, T, R> Iterator for PhaseDistortionIter<I, T, R>
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
        Some(item.map_sample(|pt| self.distortion.apply_point(pt)))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let mut distortion = self.distortion;
        self.iter.fold(init, |acc, item| {
            f(acc, item.map_sample(|pt| distortion.apply_point(pt)))
        })
    }
}

impl<I, T, R> ExactSizeIterator for PhaseDistortionIter<I, T, R>
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
    use alloc::vec;
    use alloc::vec::Vec;
    use core::f64::consts::PI;
    use rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    const EPS_F64: f64 = 1e-12;
    const EPS_F32: f32 = 1e-6;

    #[test]
    fn test_static_phase_offset_rotation_and_energy() {
        let mut channel = PhaseDistortion::<f64>::static_offset(PI / 2.0);

        // Rotating (1.0 + 0.0j) by +pi/2 must yield (0.0 + 1.0j)
        let p0 = Complex::new(1.0, 0.0);
        let r0 = channel.apply_point(p0);

        assert!(
            (r0.re - 0.0).abs() < EPS_F64,
            "Expected re ≈ 0.0, got {}",
            r0.re
        );
        assert!(
            (r0.im - 1.0).abs() < EPS_F64,
            "Expected im ≈ 1.0, got {}",
            r0.im
        );
        assert!(
            (r0.norm_sqr() - 1.0).abs() < EPS_F64,
            "Energy must be preserved"
        );

        // Subsequent points must receive the exact same static rotation
        let p1 = Complex::new(0.0, 1.0);
        let r1 = channel.apply_point(p1);
        assert!(
            (r1.re - (-1.0)).abs() < EPS_F64,
            "Expected re ≈ -1.0, got {}",
            r1.re
        );
        assert!(
            (r1.im - 0.0).abs() < EPS_F64,
            "Expected im ≈ 0.0, got {}",
            r1.im
        );
    }

    #[test]
    fn test_cfo_phase_progression_and_periodicity() {
        // Delta omega = 2pi / 8 (8 samples per complete 360-degree rotation)
        let step = 2.0 * PI / 8.0;
        let mut channel = PhaseDistortion::<f64>::from_cfo(step);

        let unit_symbol = Complex::new(1.0, 0.0);
        let mut rotated = Vec::with_capacity(8);

        for _ in 0..8 {
            rotated.push(channel.apply_point(unit_symbol));
        }

        // Check exact coordinates around the 8-PSK unit circle
        for (k, pt) in rotated.iter().enumerate() {
            let expected_angle = (k as f64) * step;
            let expected_re = expected_angle.cos();
            let expected_im = expected_angle.sin();

            assert!(
                (pt.re - expected_re).abs() < EPS_F64,
                "Sample {k} re mismatch: got {}, expected {expected_re}",
                pt.re
            );
            assert!(
                (pt.im - expected_im).abs() < EPS_F64,
                "Sample {k} im mismatch: got {}, expected {expected_im}",
                pt.im
            );
            assert!(
                (pt.norm_sqr() - 1.0).abs() < EPS_F64,
                "Unit energy violated"
            );
        }

        // The 9th sample (index 8) must wrap back to 0 rad (1.0 + 0.0j)
        let r8 = channel.apply_point(unit_symbol);
        assert!(
            (r8.re - 1.0).abs() < EPS_F64,
            "Expected full period wrap to 1.0"
        );
        assert!(r8.im.abs() < EPS_F64, "Expected full period wrap to 0.0");
    }

    #[test]
    fn test_cfo_hz_conversion() {
        // 1 kHz offset at 8 kHz sample rate -> delta_omega = 2pi * (1000 / 8000) = pi / 4
        let channel_hz = PhaseDistortion::<f32>::from_cfo_hz(1000.0, 8000.0);
        let expected_step = (2.0 * core::f32::consts::PI) * (1.0 / 8.0);

        assert!(
            (channel_hz.step_phasor.im.atan2(channel_hz.step_phasor.re) - expected_step).abs()
                < EPS_F32,
            "Phase step mismatch: got {}, expected {expected_step}",
            (channel_hz.step_phasor.im.atan2(channel_hz.step_phasor.re))
        );
    }

    #[test]
    fn test_wiener_phase_noise_step_statistics() {
        let rng = Xoshiro256PlusPlus::seed_from_u64(0xABCD_1234_5678);
        let sigma_phi = 0.02f64; // Standard deviation of phase step in radians
        let mut channel = PhaseDistortion::wiener(sigma_phi, rng);

        let num_samples = 50_000;
        let unit_symbol = Complex::new(1.0f64, 0.0f64);

        let mut prev_pt = channel.apply_point(unit_symbol);
        let mut step_deltas = Vec::with_capacity(num_samples - 1);

        for _ in 1..num_samples {
            let curr_pt = channel.apply_point(unit_symbol);

            // Compute delta phase angle between consecutive symbols: dtheta = angle(curr * conj(prev))
            let conj_prod = curr_pt * Complex::new(prev_pt.re, -prev_pt.im);
            let d_theta = conj_prod.im.atan2(conj_prod.re);
            step_deltas.push(d_theta);

            prev_pt = curr_pt;
        }

        // Statistical validation of the step increments
        let n = step_deltas.len() as f64;
        let mean = step_deltas.iter().sum::<f64>() / n;
        let variance = step_deltas
            .iter()
            .map(|&d| (d - mean) * (d - mean))
            .sum::<f64>()
            / n;
        let std_dev = variance.sqrt();

        // 1. Mean must be zero (within 3 standard errors: 3 * (sigma / sqrt(N)) ≈ 0.00027)
        assert!(
            mean.abs() < 0.001,
            "Wiener step mean was {mean}, expected ~0.0"
        );

        // 2. Standard deviation must match configured sigma_phi
        assert!(
            (std_dev - sigma_phi).abs() < 0.001,
            "Wiener step std_dev was {std_dev}, expected ~{sigma_phi}"
        );
    }

    #[test]
    fn test_combined_phase_distortion() {
        let rng = Xoshiro256PlusPlus::seed_from_u64(999);
        let initial_phase = PI / 4.0;
        let phase_step = PI / 6.0;
        let phase_noise_std = 0.001;

        let mut channel = PhaseDistortion::new(initial_phase, phase_step, phase_noise_std, rng);

        let pt = Complex::new(1.0, 0.0);
        let r0 = channel.apply_point(pt);

        // r0 should be rotated by approx initial_phase (pi/4)
        let expected_angle = PI / 4.0;
        assert!((r0.re - expected_angle.cos()).abs() < 1e-2);
        assert!((r0.im - expected_angle.sin()).abs() < 1e-2);
    }

    #[test]
    fn test_phase_distortion_iter_exact_size_and_hints() {
        let symbols = vec![
            Complex::new(1.0f32, 1.0f32),
            Complex::new(-1.0f32, 1.0f32),
            Complex::new(-1.0f32, -1.0f32),
            Complex::new(1.0f32, -1.0f32),
        ];

        let iter = PhaseDistortionIter::new(
            symbols.clone().into_iter(),
            PhaseDistortion::static_offset(0.5f32),
        );

        assert_eq!(iter.len(), 4);
        assert_eq!(iter.size_hint(), (4, Some(4)));

        let output: Vec<Complex<f32>> = iter.collect();
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_channel_ext_fluent_static_phase() {
        use crate::channel::ChannelExt;

        let symbols = vec![Complex::new(1.0f64, 0.0f64), Complex::new(0.0f64, 1.0f64)];

        // Rotate by PI / 2 using ChannelExt
        let rotated: Vec<Complex<f64>> = symbols.into_iter().add_phase_offset(PI / 2.0).collect();

        assert!((rotated[0].re - 0.0).abs() < EPS_F64);
        assert!((rotated[0].im - 1.0).abs() < EPS_F64);
        assert!((rotated[1].re - (-1.0)).abs() < EPS_F64);
        assert!((rotated[1].im - 0.0).abs() < EPS_F64);
    }

    #[test]
    fn test_channel_ext_fluent_cfo_hz() {
        use crate::channel::ChannelExt;

        let symbols = vec![Complex::new(1.0f32, 0.0f32); 4];

        // 2.5 kHz offset at 10 kHz sample rate -> delta_omega = 2pi * (2500 / 10000) = pi / 2
        let out: Vec<Complex<f32>> = symbols.into_iter().add_cfo_hz(2500.0, 10000.0).collect();

        // 0 rad -> (1, 0)
        assert!((out[0].re - 1.0).abs() < EPS_F32);
        assert!(out[0].im.abs() < EPS_F32);

        // pi / 2 rad -> (0, 1)
        assert!(out[1].re.abs() < EPS_F32);
        assert!((out[1].im - 1.0).abs() < EPS_F32);

        // pi rad -> (-1, 0)
        assert!((out[2].re - (-1.0)).abs() < EPS_F32);
        assert!(out[2].im.abs() < EPS_F32);

        // 3pi / 2 rad -> (0, -1)
        assert!(out[3].re.abs() < EPS_F32);
        assert!((out[3].im - (-1.0)).abs() < EPS_F32);
    }

    #[test]
    fn test_channel_ext_composite_pipeline() {
        use crate::channel::ChannelExt;
        use crate::constellation::Qpsk;
        use crate::demodulation::DemodulateExt;
        use crate::modulation::ModulateExt;

        let payload = vec![0x12, 0x34, 0x56, 0x78];
        let qpsk = Qpsk::<f64>::QPSK;
        let rng_phase = Xoshiro256PlusPlus::seed_from_u64(101);
        let rng_awgn = Xoshiro256PlusPlus::seed_from_u64(202);

        // Small jitter (0.001 rad) + High SNR (35 dB) + 0 CFO
        let recovered: Vec<u8> = payload
            .clone()
            .into_iter()
            .modulate(&qpsk)
            .add_phase_noise(0.001, rng_phase)
            .add_awgn_snr(&qpsk, 35.0, rng_awgn)
            .demodulate_hard(&qpsk)
            .collect();

        assert_eq!(recovered, payload);
    }

    #[test]
    fn test_wiener_phase_noise_threshold_boundary() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xFACE_CAFE);
        let unit_symbol = Complex::new(1.0f64, 0.0f64);
        let num_samples = 50_000;

        // 1. Below threshold (sigma = 0.05 < 0.1) -> exercises small-angle fast path
        let mut fast_channel = PhaseDistortion::wiener(0.05f64, &mut rng);
        let mut prev_fast = fast_channel.apply_point(unit_symbol);
        let mut deltas_fast = Vec::with_capacity(num_samples - 1);

        for _ in 1..num_samples {
            let curr = fast_channel.apply_point(unit_symbol);
            let conj_prod = curr * Complex::new(prev_fast.re, -prev_fast.im);
            deltas_fast.push(conj_prod.im.atan2(conj_prod.re));
            prev_fast = curr;
        }

        let mean_fast = deltas_fast.iter().sum::<f64>() / (deltas_fast.len() as f64);
        let var_fast = deltas_fast
            .iter()
            .map(|&d| (d - mean_fast) * (d - mean_fast))
            .sum::<f64>()
            / (deltas_fast.len() as f64);
        let std_fast = var_fast.sqrt();

        assert!(mean_fast.abs() < 0.0015);
        assert!((std_fast - 0.05).abs() < 0.0015);

        // 2. Above threshold (sigma = 0.25 >= 0.1) -> exercises exact trig fallback
        let mut exact_channel = PhaseDistortion::wiener(0.25f64, &mut rng);
        let mut prev_exact = exact_channel.apply_point(unit_symbol);
        let mut deltas_exact = Vec::with_capacity(num_samples - 1);

        for _ in 1..num_samples {
            let curr = exact_channel.apply_point(unit_symbol);
            let conj_prod = curr * Complex::new(prev_exact.re, -prev_exact.im);
            deltas_exact.push(conj_prod.im.atan2(conj_prod.re));
            prev_exact = curr;
        }

        let mean_exact = deltas_exact.iter().sum::<f64>() / (deltas_exact.len() as f64);
        let var_exact = deltas_exact
            .iter()
            .map(|&d| (d - mean_exact) * (d - mean_exact))
            .sum::<f64>()
            / (deltas_exact.len() as f64);
        let std_exact = var_exact.sqrt();

        assert!(mean_exact.abs() < 0.005);
        assert!((std_exact - 0.25).abs() < 0.005);
    }
}
