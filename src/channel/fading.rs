use core::marker::PhantomData;
use num_complex::Complex;
use num_traits::Float;
use rand_core::Rng;
use rand_distr::{Distribution, StandardNormal};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FadedSymbol<T> {
    pub sample: Complex<T>,
    pub csi: Complex<T>,
}

// =============================================================================
// Coherence Policy (Temporal Dynamics)
// =============================================================================

pub trait CoherencePolicy<T: Copy> {
    fn update_gain<F>(&mut self, sample_fn: F) -> Complex<T>
    where
        F: FnOnce() -> Complex<T>;
}

/// Zero-sized marker for memoryless fast fading (resamples gain on every symbol).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FastFading;

impl<T: Copy + Float> CoherencePolicy<T> for FastFading {
    #[inline(always)]
    fn update_gain<F>(&mut self, sample_fn: F) -> Complex<T>
    where
        F: FnOnce() -> Complex<T>,
    {
        sample_fn()
    }
}

/// Block fading policy holding channel gain constant across `N` consecutive symbols.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockFading<T, const N: usize> {
    current_gain: Complex<T>,
    symbol_counter: usize,
}

impl<T: Copy + Float, const N: usize> BlockFading<T, N> {
    #[inline]
    pub fn new() -> Self {
        assert!(N > 0, "Block fading coherence N must be at least 1");
        Self {
            current_gain: Complex::new(T::zero(), T::zero()),
            symbol_counter: 0,
        }
    }
}

impl<T: Copy + Float, const N: usize> Default for BlockFading<T, N> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy, const N: usize> CoherencePolicy<T> for BlockFading<T, N> {
    #[inline(always)]
    fn update_gain<F>(&mut self, sample_fn: F) -> Complex<T>
    where
        F: FnOnce() -> Complex<T>,
    {
        if self.symbol_counter == 0 {
            self.current_gain = sample_fn();
        }
        self.symbol_counter = (self.symbol_counter + 1) % N;
        self.current_gain
    }
}

// =============================================================================
// Fading Statistical Distribution Models
// =============================================================================

pub trait FadingModel<T> {
    fn sample_gain<R>(&self, rng: &mut R) -> Complex<T>
    where
        T: Float,
        StandardNormal: Distribution<T>,
        R: rand_core::Rng;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayleighModel<T> {
    pub sigma: T,
}

impl<T: Float> RayleighModel<T> {
    #[inline]
    pub fn new() -> Self {
        let two = T::one() + T::one();
        Self {
            sigma: T::one() / two.sqrt(),
        }
    }

    #[inline]
    pub fn with_average_power(omega: T) -> Self {
        let two = T::one() + T::one();
        Self {
            sigma: (omega / two).sqrt(),
        }
    }
}

impl<T: Float> Default for RayleighModel<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T> FadingModel<T> for RayleighModel<T> {
    #[inline(always)]
    fn sample_gain<R>(&self, rng: &mut R) -> Complex<T>
    where
        T: Float,
        StandardNormal: Distribution<T>,
        R: rand_core::Rng,
    {
        let z_i: T = StandardNormal.sample(rng);
        let z_q: T = StandardNormal.sample(rng);
        Complex::new(z_i * self.sigma, z_q * self.sigma)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RicianModel<T> {
    pub h_los: Complex<T>,
    pub sigma_diffuse: T,
}

impl<T: Float> RicianModel<T> {
    #[inline]
    pub fn from_k_linear(k: T, los_angle: T) -> Self {
        assert!(k >= T::zero(), "Rician K-factor must be non-negative");

        let two = T::one() + T::one();
        let a_los = (k / (k + T::one())).sqrt();
        let h_los = Complex::new(a_los * los_angle.cos(), a_los * los_angle.sin());
        let sigma_diffuse = (T::one() / (two * (k + T::one()))).sqrt();

        Self {
            h_los,
            sigma_diffuse,
        }
    }

    #[inline]
    pub fn from_k_db(k_db: T, los_angle: T) -> Self {
        let two = T::one() + T::one();
        let ten = two + two + two + two + two;
        let k_lin = ten.powf(k_db / ten);
        Self::from_k_linear(k_lin, los_angle)
    }
}

impl<T> FadingModel<T> for RicianModel<T> {
    #[inline(always)]
    fn sample_gain<R>(&self, rng: &mut R) -> Complex<T>
    where
        T: Float,
        StandardNormal: Distribution<T>,
        R: rand_core::Rng,
    {
        let z_i: T = StandardNormal.sample(rng);
        let z_q: T = StandardNormal.sample(rng);
        self.h_los + Complex::new(z_i * self.sigma_diffuse, z_q * self.sigma_diffuse)
    }
}

// =============================================================================
// Unified Fading Channel Driver
// =============================================================================

pub struct FadingChannel<T, R, M, C = FastFading> {
    model: M,
    rng: R,
    coherence: C,
    _marker: PhantomData<T>,
}

impl<T, R, M, C> FadingChannel<T, R, M, C>
where
    T: Float,
    StandardNormal: Distribution<T>,
    R: rand_core::Rng,
    M: FadingModel<T>,
    C: CoherencePolicy<T>,
{
    #[inline(always)]
    pub fn apply_point(&mut self, point: Complex<T>) -> Complex<T> {
        let h = self
            .coherence
            .update_gain(|| self.model.sample_gain(&mut self.rng));
        point * h
    }

    #[inline(always)]
    pub fn apply_point_with_csi(&mut self, point: Complex<T>) -> FadedSymbol<T> {
        let h = self
            .coherence
            .update_gain(|| self.model.sample_gain(&mut self.rng));
        FadedSymbol {
            sample: point * h,
            csi: h,
        }
    }
}

// =============================================================================
// Ergonomic Type Aliases & Constructors
// =============================================================================

pub type RayleighChannel<T, R, C = FastFading> = FadingChannel<T, R, RayleighModel<T>, C>;
pub type RicianChannel<T, R, C = FastFading> = FadingChannel<T, R, RicianModel<T>, C>;

pub type FastRayleigh<T, R> = RayleighChannel<T, R, FastFading>;
pub type BlockRayleigh<T, R, const N: usize> = RayleighChannel<T, R, BlockFading<T, N>>;

pub type FastRician<T, R> = RicianChannel<T, R, FastFading>;
pub type BlockRician<T, R, const N: usize> = RicianChannel<T, R, BlockFading<T, N>>;

// --- Rayleigh Fast Fading Constructors ---
impl<T: Float, R: rand_core::Rng> RayleighChannel<T, R, FastFading> {
    #[inline]
    pub fn new(rng: R) -> Self {
        Self {
            model: RayleighModel::new(),
            rng,
            coherence: FastFading,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn with_average_power(omega: T, rng: R) -> Self {
        Self {
            model: RayleighModel::with_average_power(omega),
            rng,
            coherence: FastFading,
            _marker: PhantomData,
        }
    }
}

// --- Rayleigh Block Fading Constructors ---
impl<T: Float, R: rand_core::Rng, const N: usize> RayleighChannel<T, R, BlockFading<T, N>> {
    #[inline]
    pub fn new_block(rng: R) -> Self {
        Self {
            model: RayleighModel::new(),
            rng,
            coherence: BlockFading::new(),
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn with_average_power_block(omega: T, rng: R) -> Self {
        Self {
            model: RayleighModel::with_average_power(omega),
            rng,
            coherence: BlockFading::new(),
            _marker: PhantomData,
        }
    }
}

// --- Rician Fast Fading Constructors ---
impl<T: Float, R: rand_core::Rng> RicianChannel<T, R, FastFading> {
    #[inline]
    pub fn from_k_linear(k: T, los_angle: T, rng: R) -> Self {
        Self {
            model: RicianModel::from_k_linear(k, los_angle),
            rng,
            coherence: FastFading,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn from_k_db(k_db: T, los_angle: T, rng: R) -> Self {
        Self {
            model: RicianModel::from_k_db(k_db, los_angle),
            rng,
            coherence: FastFading,
            _marker: PhantomData,
        }
    }
}

// --- Rician Block Fading Constructors ---
impl<T: Float, R: rand_core::Rng, const N: usize> RicianChannel<T, R, BlockFading<T, N>> {
    #[inline]
    pub fn from_k_linear_block(k: T, los_angle: T, rng: R) -> Self {
        Self {
            model: RicianModel::from_k_linear(k, los_angle),
            rng,
            coherence: BlockFading::new(),
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn from_k_db_block(k_db: T, los_angle: T, rng: R) -> Self {
        Self {
            model: RicianModel::from_k_db(k_db, los_angle),
            rng,
            coherence: BlockFading::new(),
            _marker: PhantomData,
        }
    }
}
// =============================================================================
// Streaming Fading Iterators
// =============================================================================

pub struct FadingIter<I, T, R, M, C> {
    iter: I,
    channel: FadingChannel<T, R, M, C>,
}

impl<I, T, R, M, C> FadingIter<I, T, R, M, C> {
    #[inline]
    pub fn new(iter: I, channel: FadingChannel<T, R, M, C>) -> Self {
        Self { iter, channel }
    }
}

impl<I, T, R, M, C> Iterator for FadingIter<I, T, R, M, C>
where
    I: Iterator<Item = Complex<T>>,
    T: Float,
    StandardNormal: Distribution<T>,
    R: rand_core::Rng,
    M: FadingModel<T>,
    C: CoherencePolicy<T>,
{
    type Item = Complex<T>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let point = self.iter.next()?;
        Some(self.channel.apply_point(point))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<I, T, R, M, C> ExactSizeIterator for FadingIter<I, T, R, M, C>
where
    I: ExactSizeIterator<Item = Complex<T>>,
    T: Float,
    StandardNormal: Distribution<T>,
    R: rand_core::Rng,
    M: FadingModel<T>,
    C: CoherencePolicy<T>,
{
}

pub struct FadingCsiIter<I, T, R, M, C> {
    iter: I,
    channel: FadingChannel<T, R, M, C>,
}

impl<I, T, R, M, C> FadingCsiIter<I, T, R, M, C> {
    #[inline]
    pub fn new(iter: I, channel: FadingChannel<T, R, M, C>) -> Self {
        Self { iter, channel }
    }
}

impl<I, T, R, M, C> Iterator for FadingCsiIter<I, T, R, M, C>
where
    I: Iterator<Item = Complex<T>>,
    T: Float,
    StandardNormal: Distribution<T>,
    R: rand_core::Rng,
    M: FadingModel<T>,
    C: CoherencePolicy<T>,
{
    type Item = FadedSymbol<T>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let point = self.iter.next()?;
        Some(self.channel.apply_point_with_csi(point))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<I, T, R, M, C> ExactSizeIterator for FadingCsiIter<I, T, R, M, C>
where
    I: ExactSizeIterator<Item = Complex<T>>,
    T: Float,
    StandardNormal: Distribution<T>,
    R: rand_core::Rng,
    M: FadingModel<T>,
    C: CoherencePolicy<T>,
{
}

pub trait EqualizationPolicy<T> {
    fn equalize(&self, symbol: FadedSymbol<T>) -> Complex<T>;
}

pub struct ZeroForcing;

impl<T: Float> EqualizationPolicy<T> for ZeroForcing {
    fn equalize(&self, symbol: FadedSymbol<T>) -> Complex<T> {
        let denominator = symbol.csi.norm_sqr();
        (symbol.sample * symbol.csi.conj()) / denominator
    }
}

pub struct Mmse<T> {
    pub noise_var: T,
}
impl<T: Float> EqualizationPolicy<T> for Mmse<T> {
    fn equalize(&self, symbol: FadedSymbol<T>) -> Complex<T> {
        let denominator = symbol.csi.norm_sqr() + self.noise_var;
        (symbol.sample * symbol.csi.conj()) / denominator
    }
}
pub struct EqualizeIter<I, T, E = ZeroForcing> {
    iter: I,
    policy: E,
    _marker: core::marker::PhantomData<T>,
}

impl<I, T: Float, E> EqualizeIter<I, T, E> {
    fn new(iter: I, policy: E) -> EqualizeIter<I, T, E> {
        Self {
            iter,
            policy,
            _marker: PhantomData,
        }
    }
}

impl<I, T, E> Iterator for EqualizeIter<I, T, E>
where
    I: Iterator<Item = FadedSymbol<T>>,
    T: Float,
    E: EqualizationPolicy<T>,
{
    type Item = Complex<T>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let faded = self.iter.next()?;
        Some(self.policy.equalize(faded))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<I, T, E> ExactSizeIterator for EqualizeIter<I, T, E>
where
    I: ExactSizeIterator<Item = FadedSymbol<T>>,
    T: Float,
    E: EqualizationPolicy<T>,
{
}

pub type EqualizeZfIter<I, T> = EqualizeIter<I, T, ZeroForcing>;
pub type EqualizeMmseIter<I, T> = EqualizeIter<I, T, Mmse<T>>;

pub trait EqualizeExt<T: Float>: Iterator<Item = FadedSymbol<T>> + Sized {
    /// Applies zero-forcing (ZF) 1-tap equalization (r / h)
    #[inline]
    fn equalize_zf(self) -> EqualizeZfIter<Self, T> {
        EqualizeIter::new(self, ZeroForcing)
    }

    /// Applies MMSE 1-tap equalization (r * conj(h) / (|h|^2 + N0))
    #[inline]
    fn equalize_mmse(self, noise_var: T) -> EqualizeMmseIter<Self, T> {
        EqualizeIter::new(self, Mmse { noise_var })
    }
}

impl<T: Float, I: Iterator<Item = FadedSymbol<T>>> EqualizeExt<T> for I {}

// =============================================================================
// Fading Fluent Extension Trait
// =============================================================================

pub trait FadingExt<T: Float>: Iterator<Item = Complex<T>> + Sized {
    /// Injects fast Rayleigh flat fading with unit average power $\mathbb{E}[|h|^2] = 1.0$.
    #[inline]
    fn add_rayleigh<R>(self, rng: R) -> FadingIter<Self, T, R, RayleighModel<T>, FastFading>
    where
        StandardNormal: Distribution<T>,
        R: rand_core::Rng,
    {
        FadingIter::new(self, RayleighChannel::new(rng))
    }

    /// Injects fast Rayleigh flat fading with custom average power $\Omega = \mathbb{E}[|h|^2]$.
    #[inline]
    fn add_rayleigh_power<R>(
        self,
        omega: T,
        rng: R,
    ) -> FadingIter<Self, T, R, RayleighModel<T>, FastFading>
    where
        StandardNormal: Distribution<T>,
        R: rand_core::Rng,
    {
        FadingIter::new(self, RayleighChannel::with_average_power(omega, rng))
    }

    /// Injects block Rayleigh flat fading holding gain constant across `N` symbols.
    #[inline]
    fn add_rayleigh_block<const N: usize, R>(
        self,
        rng: R,
    ) -> FadingIter<Self, T, R, RayleighModel<T>, BlockFading<T, N>>
    where
        StandardNormal: Distribution<T>,
        R: rand_core::Rng,
    {
        FadingIter::new(self, RayleighChannel::new_block(rng))
    }

    /// Injects fast Rayleigh flat fading emitting [`FadedSymbol<T>`] with CSI.
    #[inline]
    fn add_rayleigh_csi<R>(self, rng: R) -> FadingCsiIter<Self, T, R, RayleighModel<T>, FastFading>
    where
        StandardNormal: Distribution<T>,
        R: rand_core::Rng,
    {
        FadingCsiIter::new(self, RayleighChannel::new(rng))
    }

    /// Injects block Rayleigh flat fading emitting [`FadedSymbol<T>`] with CSI.
    #[inline]
    fn add_rayleigh_block_csi<const N: usize, R>(
        self,
        rng: R,
    ) -> FadingCsiIter<Self, T, R, RayleighModel<T>, BlockFading<T, N>>
    where
        StandardNormal: Distribution<T>,
        R: rand_core::Rng,
    {
        FadingCsiIter::new(self, RayleighChannel::new_block(rng))
    }

    /// Injects fast Rician flat fading with $K$-factor in linear scale.
    #[inline]
    fn add_rician_k_linear<R>(
        self,
        k: T,
        los_angle: T,
        rng: R,
    ) -> FadingIter<Self, T, R, RicianModel<T>, FastFading>
    where
        StandardNormal: Distribution<T>,
        R: rand_core::Rng,
    {
        FadingIter::new(self, RicianChannel::from_k_linear(k, los_angle, rng))
    }

    /// Injects fast Rician flat fading with $K$-factor in dB.
    #[inline]
    fn add_rician_k_db<R>(
        self,
        k_db: T,
        los_angle: T,
        rng: R,
    ) -> FadingIter<Self, T, R, RicianModel<T>, FastFading>
    where
        StandardNormal: Distribution<T>,
        R: rand_core::Rng,
    {
        FadingIter::new(self, RicianChannel::from_k_db(k_db, los_angle, rng))
    }

    /// Injects block Rician flat fading with $K$-factor in dB holding gain constant across `N` symbols.
    #[inline]
    fn add_rician_block_k_db<const N: usize, R>(
        self,
        k_db: T,
        los_angle: T,
        rng: R,
    ) -> FadingIter<Self, T, R, RicianModel<T>, BlockFading<T, N>>
    where
        StandardNormal: Distribution<T>,
        R: rand_core::Rng,
    {
        FadingIter::new(self, RicianChannel::from_k_db_block(k_db, los_angle, rng))
    }

    /// Injects fast Rician flat fading emitting [`FadedSymbol<T>`] with CSI.
    #[inline]
    fn add_rician_csi_k_db<R>(
        self,
        k_db: T,
        los_angle: T,
        rng: R,
    ) -> FadingCsiIter<Self, T, R, RicianModel<T>, FastFading>
    where
        StandardNormal: Distribution<T>,
        R: rand_core::Rng,
    {
        FadingCsiIter::new(self, RicianChannel::from_k_db(k_db, los_angle, rng))
    }

    /// Injects block Rician flat fading emitting [`FadedSymbol<T>`] with CSI.
    #[inline]
    fn add_rician_block_csi_k_db<const N: usize, R>(
        self,
        k_db: T,
        los_angle: T,
        rng: R,
    ) -> FadingCsiIter<Self, T, R, RicianModel<T>, BlockFading<T, N>>
    where
        StandardNormal: Distribution<T>,
        R: rand_core::Rng,
    {
        FadingCsiIter::new(self, RicianChannel::from_k_db_block(k_db, los_angle, rng))
    }
}

impl<T: Float, I: Iterator<Item = Complex<T>>> FadingExt<T> for I {}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::constellation::Bpsk;
    use crate::demodulation::DemodulateExt;
    use crate::modulation::ModulateExt;
    use alloc::vec;
    use alloc::vec::Vec;
    use rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    const EPS_F64: f64 = 1e-12;
    const EPS_F32: f32 = 1e-6;

    // =========================================================================
    // 1. Statistical Moments & Energy Invariance Tests
    // =========================================================================

    #[test]
    fn test_rayleigh_power_and_envelope_moments() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x1234_5678_9ABC);
        let model = RayleighModel::<f64>::new();
        let num_samples = 100_000;

        let mut sum_power = 0.0f64;
        let mut sum_mag = 0.0f64;
        let mut magnitudes = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            let h = model.sample_gain(&mut rng);
            let power = h.norm_sqr();
            let mag = power.sqrt();

            sum_power += power;
            sum_mag += mag;
            magnitudes.push(mag);
        }

        let n = num_samples as f64;
        let avg_power = sum_power / n;
        let avg_mag = sum_mag / n;

        let variance_mag = magnitudes
            .iter()
            .map(|&m| (m - avg_mag) * (m - avg_mag))
            .sum::<f64>()
            / n;

        // 1. Unit Average Power: E[|h|^2] = 1.0
        assert!(
            (avg_power - 1.0).abs() < 0.015,
            "Rayleigh power was {avg_power}, expected ~1.0"
        );

        // 2. Theoretical Envelope Mean: E[|h|] = sqrt(PI) / 2 ≈ 0.886227
        let expected_mean = (core::f64::consts::PI / 4.0).sqrt();
        assert!(
            (avg_mag - expected_mean).abs() < 0.01,
            "Rayleigh mean magnitude was {avg_mag}, expected ~{expected_mean}"
        );

        // 3. Theoretical Envelope Variance: Var(|h|) = (4 - PI) / 4 ≈ 0.214602
        let expected_var = (4.0 - core::f64::consts::PI) / 4.0;
        assert!(
            (variance_mag - expected_var).abs() < 0.01,
            "Rayleigh magnitude variance was {variance_mag}, expected ~{expected_var}"
        );
    }

    #[test]
    fn test_rayleigh_custom_average_power() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xDEAD_BEEF);
        let target_omega = 4.0f64; // Total power = 4.0
        let model = RayleighModel::<f64>::with_average_power(target_omega);
        let num_samples = 100_000;

        let mut sum_power = 0.0f64;
        for _ in 0..num_samples {
            let h = model.sample_gain(&mut rng);
            sum_power += h.norm_sqr();
        }

        let avg_power = sum_power / (num_samples as f64);
        assert!(
            (avg_power - target_omega).abs() < 0.05,
            "Scaled Rayleigh power was {avg_power}, expected ~{target_omega}"
        );
    }

    #[test]
    fn test_rician_limiting_cases() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xCAFE_BABE);
        let num_samples = 100_000;

        // ---------------------------------------------------------------------
        // Case A: Very low K (-60 dB) -> Must match Rayleigh distribution
        // ---------------------------------------------------------------------
        let rician_nlos = RicianModel::<f64>::from_k_db(-60.0, 0.0);
        let mut sum_power_nlos = 0.0;
        let mut sum_mag_nlos = 0.0;

        for _ in 0..num_samples {
            let h = rician_nlos.sample_gain(&mut rng);
            sum_power_nlos += h.norm_sqr();
            sum_mag_nlos += h.norm_sqr().sqrt();
        }

        let mean_power_nlos = sum_power_nlos / (num_samples as f64);
        let mean_mag_nlos = sum_mag_nlos / (num_samples as f64);
        let expected_rayleigh_mean = (core::f64::consts::PI / 4.0).sqrt();

        assert!((mean_power_nlos - 1.0).abs() < 0.015);
        assert!((mean_mag_nlos - expected_rayleigh_mean).abs() < 0.01);

        // ---------------------------------------------------------------------
        // Case B: Very high K (+60 dB) -> Must converge to deterministic phasor
        // ---------------------------------------------------------------------
        let los_angle = core::f64::consts::PI / 3.0; // 60 degrees
        let rician_los = RicianModel::<f64>::from_k_db(60.0, los_angle);

        let mut sum_power_los = 0.0;
        let mut sum_mag_los = 0.0;

        for _ in 0..num_samples {
            let h = rician_los.sample_gain(&mut rng);
            sum_power_los += h.norm_sqr();
            sum_mag_los += h.norm_sqr().sqrt();
        }

        let mean_power_los = sum_power_los / (num_samples as f64);
        let mean_mag_los = sum_mag_los / (num_samples as f64);

        assert!((mean_power_los - 1.0).abs() < 1e-4);
        assert!((mean_mag_los - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_rician_power_preservation_across_k_factors() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5555_AAAA);
        let k_dbs = [-10.0, 0.0, 3.0, 10.0, 20.0];
        let num_samples = 50_000;

        for &k_db in &k_dbs {
            let model = RicianModel::<f64>::from_k_db(k_db, 0.785);
            let mut sum_power = 0.0f64;

            for _ in 0..num_samples {
                sum_power += model.sample_gain(&mut rng).norm_sqr();
            }

            let avg_power = sum_power / (num_samples as f64);
            assert!(
                (avg_power - 1.0).abs() < 0.02,
                "Rician power at K={k_db} dB was {avg_power}, expected ~1.0"
            );
        }
    }

    // =========================================================================
    // 2. Coherence Policy & Block Fading Mechanics
    // =========================================================================

    #[test]
    fn test_block_fading_invariance_and_transition() {
        let rng = Xoshiro256PlusPlus::seed_from_u64(42);
        // Block coherence N = 4
        let mut channel = BlockRayleigh::<f64, _, 4>::new_block(rng);
        let dummy_symbol = Complex::new(1.0f64, 0.0f64);

        let mut csi_records = Vec::with_capacity(12);
        for _ in 0..12 {
            let faded = channel.apply_point_with_csi(dummy_symbol);
            csi_records.push(faded.csi);
        }

        // Block 0: symbols 0..4 must have strictly identical channel state
        assert_eq!(csi_records[0], csi_records[1]);
        assert_eq!(csi_records[1], csi_records[2]);
        assert_eq!(csi_records[2], csi_records[3]);

        // Block 1: symbols 4..8 must be identical to each other, but different from Block 0
        assert_ne!(csi_records[3], csi_records[4]);
        assert_eq!(csi_records[4], csi_records[5]);
        assert_eq!(csi_records[5], csi_records[6]);
        assert_eq!(csi_records[6], csi_records[7]);

        // Block 2: symbols 8..12 transition check
        assert_ne!(csi_records[7], csi_records[8]);
        assert_eq!(csi_records[8], csi_records[9]);
    }

    #[test]
    fn test_fast_fading_resamples_every_symbol() {
        let rng = Xoshiro256PlusPlus::seed_from_u64(101);
        let mut channel = FastRayleigh::<f64, _>::new(rng);
        let pt = Complex::new(1.0, 0.0);

        let s0 = channel.apply_point_with_csi(pt);
        let s1 = channel.apply_point_with_csi(pt);
        let s2 = channel.apply_point_with_csi(pt);

        assert_ne!(s0.csi, s1.csi);
        assert_ne!(s1.csi, s2.csi);
    }

    // =========================================================================
    // 3. Equalization Pipeline Tests (ZF & MMSE)
    // =========================================================================

    #[test]
    fn test_ideal_zf_equalization_noiseless() {
        let rng = Xoshiro256PlusPlus::seed_from_u64(0x999);
        let symbols = vec![
            Complex::new(1.0f64, 1.0f64),
            Complex::new(-1.0f64, 1.0f64),
            Complex::new(-1.0f64, -1.0f64),
            Complex::new(1.0f64, -1.0f64),
        ];

        let equalized: Vec<Complex<f64>> = symbols
            .clone()
            .into_iter()
            .add_rayleigh_csi(rng)
            .equalize_zf()
            .collect();

        assert_eq!(equalized.len(), symbols.len());
        for (recovered, original) in equalized.iter().zip(symbols.iter()) {
            assert!(
                (recovered.re - original.re).abs() < EPS_F64,
                "ZF real mismatch: got {}, expected {}",
                recovered.re,
                original.re
            );
            assert!(
                (recovered.im - original.im).abs() < EPS_F64,
                "ZF imag mismatch: got {}, expected {}",
                recovered.im,
                original.im
            );
        }
    }

    #[test]
    fn test_mmse_equalizer_at_low_noise_approaches_zf() {
        let pt = Complex::new(2.5f32, -1.5f32);
        let csi = Complex::new(0.6f32, -0.8f32); // |h|^2 = 1.0
        let faded = FadedSymbol {
            sample: pt * csi,
            csi,
        };

        let zf_out = ZeroForcing.equalize(faded);
        let mmse_out = Mmse { noise_var: 1e-6f32 }.equalize(faded);

        assert!((zf_out.re - pt.re).abs() < EPS_F32);
        assert!((zf_out.im - pt.im).abs() < EPS_F32);
        assert!((mmse_out.re - zf_out.re).abs() < 1e-4);
        assert!((mmse_out.im - zf_out.im).abs() < 1e-4);
    }

    // =========================================================================
    // 4. Fluent Iterator API & Iterator Contracts
    // =========================================================================

    #[test]
    fn test_fading_iterators_exact_size_and_hints() {
        let symbols = vec![Complex::new(1.0f32, 0.0f32); 16];
        let rng = Xoshiro256PlusPlus::seed_from_u64(777);

        let mut iter = symbols.into_iter().add_rayleigh_block::<4, _>(rng);
        assert_eq!(iter.len(), 16);
        assert_eq!(iter.size_hint(), (16, Some(16)));

        iter.next();
        assert_eq!(iter.len(), 15);
        assert_eq!(iter.size_hint(), (15, Some(15)));
    }

    #[test]
    fn test_empty_fading_streams() {
        let empty: [Complex<f32>; 0] = [];
        let rng = Xoshiro256PlusPlus::seed_from_u64(1);

        let mut iter = empty.into_iter().add_rician_k_db(6.0, 0.0, rng);
        assert_eq!(iter.next(), None);
    }

    // =========================================================================
    // 5. Theoretical End-to-End BER Verification in Rayleigh Fading
    // =========================================================================

    #[test]
    fn test_bpsk_rayleigh_ber_curve_against_theory() {
        let bpsk = Bpsk::<f64>::BPSK;
        let num_bytes = 25_000; // 200,000 bits
        let total_bits = (num_bytes * 8) as f64;

        // Generate deterministic payload
        let payload: Vec<u8> = (0..num_bytes).map(|i| (i * 131 + 17) as u8).collect();

        // ---------------------------------------------------------------------
        // Theoretical Rayleigh BPSK Error Probability:
        // Pb = 0.5 * (1 - sqrt(gamma_b / (1 + gamma_b))), where gamma_b = Eb/N0
        // ---------------------------------------------------------------------

        // 1. Eb/N0 = 10.0 dB (gamma_b = 10.0) -> Pb ≈ 0.023269 (2.33%)
        let ebn0_10db_lin = 10.0f64.powf(10.0 / 10.0);
        let theoretical_ber_10db = 0.5 * (1.0 - (ebn0_10db_lin / (1.0 + ebn0_10db_lin)).sqrt());
        let noise_var_10db = 1.0 / ebn0_10db_lin;
        let sigma_axis_10db = (noise_var_10db / 2.0).sqrt();

        let rng_fading_10db = Xoshiro256PlusPlus::seed_from_u64(0x1010);
        let mut rng_awgn_10db = Xoshiro256PlusPlus::seed_from_u64(0x2020);

        let rx_10db: Vec<u8> = payload
            .clone()
            .into_iter()
            .modulate(bpsk)
            .add_rayleigh_csi(rng_fading_10db)
            .map(|faded| {
                let z_i: f64 = StandardNormal.sample(&mut rng_awgn_10db);
                let z_q: f64 = StandardNormal.sample(&mut rng_awgn_10db);
                let noise = Complex::new(z_i * sigma_axis_10db, z_q * sigma_axis_10db);
                FadedSymbol {
                    sample: faded.sample + noise,
                    csi: faded.csi,
                }
            })
            .equalize_zf()
            .demodulate_hard(bpsk)
            .collect();

        let bit_errors_10db: usize = payload
            .iter()
            .zip(rx_10db.iter())
            .map(|(&a, &b)| (a ^ b).count_ones() as usize)
            .sum();
        let empirical_ber_10db = (bit_errors_10db as f64) / total_bits;

        assert!(
            (empirical_ber_10db - theoretical_ber_10db).abs() < 0.0035,
            "Rayleigh BER at 10 dB was {empirical_ber_10db:.5}, expected ~{theoretical_ber_10db:.5}"
        );

        // 2. Eb/N0 = 17.0 dB (gamma_b = 50.1187) -> Pb ≈ 0.004902 (0.49%)
        let ebn0_17db_lin = 10.0f64.powf(17.0 / 10.0);
        let theoretical_ber_17db = 0.5 * (1.0 - (ebn0_17db_lin / (1.0 + ebn0_17db_lin)).sqrt());
        let noise_var_17db = 1.0 / ebn0_17db_lin;
        let sigma_axis_17db = (noise_var_17db / 2.0).sqrt();

        let rng_fading_17db = Xoshiro256PlusPlus::seed_from_u64(0x3030);
        let mut rng_awgn_17db = Xoshiro256PlusPlus::seed_from_u64(0x4040);

        let rx_17db: Vec<u8> = payload
            .clone()
            .into_iter()
            .modulate(bpsk)
            .add_rayleigh_csi(rng_fading_17db)
            .map(|faded| {
                let z_i: f64 = StandardNormal.sample(&mut rng_awgn_17db);
                let z_q: f64 = StandardNormal.sample(&mut rng_awgn_17db);
                let noise = Complex::new(z_i * sigma_axis_17db, z_q * sigma_axis_17db);
                FadedSymbol {
                    sample: faded.sample + noise,
                    csi: faded.csi,
                }
            })
            .equalize_zf()
            .demodulate_hard(bpsk)
            .collect();

        let bit_errors_17db: usize = payload
            .iter()
            .zip(rx_17db.iter())
            .map(|(&a, &b)| (a ^ b).count_ones() as usize)
            .sum();
        let empirical_ber_17db = (bit_errors_17db as f64) / total_bits;

        assert!(
            (empirical_ber_17db - theoretical_ber_17db).abs() < 0.0015,
            "Rayleigh BER at 17 dB was {empirical_ber_17db:.5}, expected ~{theoretical_ber_17db:.5}"
        );
    }
}

#[cfg(test)]
mod additional_tests {
    use super::*;

    use crate::constellation::{Bpsk, Qam16, Qpsk};
    use crate::demodulation::DemodulateExt;
    use crate::differential::DifferentialExt;
    use crate::modulation::ModulateExt;
    use alloc::vec;
    use alloc::vec::Vec;
    use rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    // =========================================================================
    // 1. Higher-Order Constellations over Fading (QPSK, 16-QAM)
    // =========================================================================

    #[test]
    fn test_qam16_noiseless_rayleigh_zf_roundtrip() {
        let payload = vec![0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];
        let qam16 = Qam16::<f64>::QAM16;
        let rng = Xoshiro256PlusPlus::seed_from_u64(0x4242);

        let recovered: Vec<u8> = payload
            .clone()
            .into_iter()
            .modulate(qam16)
            .add_rayleigh_csi(rng)
            .equalize_zf()
            .demodulate_hard(qam16)
            .collect();

        assert_eq!(recovered, payload);
    }

    #[test]
    fn test_qpsk_noiseless_rician_zf_roundtrip() {
        let payload = vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE];
        let qpsk = Qpsk::<f32>::QPSK;
        let rng = Xoshiro256PlusPlus::seed_from_u64(0x1337);

        let recovered: Vec<u8> = payload
            .clone()
            .into_iter()
            .modulate(qpsk)
            .add_rician_csi_k_db(6.0, 0.5, rng)
            .equalize_zf()
            .demodulate_hard(qpsk)
            .collect();

        assert_eq!(recovered, payload);
    }

    // =========================================================================
    // 2. MMSE Regularization in Deep Fades
    // =========================================================================

    #[test]
    fn test_mmse_regularization_in_deep_fade() {
        // In a deep fade where |h| is tiny, ZF inverts 1/h which heavily amplifies noise.
        // MMSE scales by h* / (|h|^2 + N0), keeping the output bounded.
        let pt = Complex::new(1.0f64, 0.0f64);
        let deep_fade_csi = Complex::new(0.005f64, 0.005f64); // |h|^2 = 5e-5
        let noise = Complex::new(0.1f64, -0.1f64);
        let noise_var = 0.02f64; // N0 = 0.02

        let faded = FadedSymbol {
            sample: pt * deep_fade_csi + noise,
            csi: deep_fade_csi,
        };

        let zf_out = ZeroForcing.equalize(faded);
        let mmse_out = Mmse { noise_var }.equalize(faded);

        // ZF severely amplifies noise (> 10.0 magnitude)
        assert!(
            zf_out.norm_sqr().sqrt() > 10.0,
            "ZF should amplify noise in deep fades, got {}",
            zf_out.norm_sqr().sqrt()
        );

        // MMSE suppresses the output magnitude (< 1.0)
        assert!(
            mmse_out.norm_sqr() < 1.0,
            "MMSE should attenuate noise in deep fades, got {}",
            mmse_out.norm_sqr().sqrt()
        );
    }

    // =========================================================================
    // 3. Soft-Decision Demodulation (LLRs) over Equalized Fading
    // =========================================================================

    #[test]
    fn test_soft_demod_after_equalization() {
        let bpsk = Bpsk::<f32>::BPSK;
        // Transmit bit 0 (+1.0) and bit 1 (-1.0)
        let tx_symbols = vec![Complex::new(1.0f32, 0.0f32), Complex::new(-1.0f32, 0.0f32)];
        let csi = Complex::new(0.8f32, 0.6f32); // |h|^2 = 1.0

        let faded_symbols = tx_symbols.into_iter().map(|s| FadedSymbol {
            sample: s * csi + Complex::new(0.05, 0.0), // small perturbation
            csi,
        });

        let noise_var = 0.1f32;
        let equalized_samples = faded_symbols.equalize_zf();

        let llrs: Vec<[f32; 1]> = equalized_samples.demodulate_soft(bpsk, noise_var).collect();

        assert!(llrs[0][0] > 0.0, "Expected positive LLR for bit 0");
        assert!(llrs[1][0] < 0.0, "Expected negative LLR for bit 1");
    }

    // =========================================================================
    // 4. Non-Coherent DPSK over Block Fading (Zero CSI)
    // =========================================================================

    #[test]
    fn test_dbpsk_over_block_rayleigh_no_csi() {
        let payload = vec![0xAA, 0x55, 0xF0, 0x0F, 0x12, 0x34];
        let bpsk = Bpsk::<f64>::BPSK;
        let rng = Xoshiro256PlusPlus::seed_from_u64(0x9876);
        let pilot = Complex::new(1.0f64, 0.0f64);

        // Constant phase across block N = 64 allows differential decoding without CSI
        let recovered: Vec<u8> = core::iter::once(pilot)
            .chain(
                payload
                    .clone()
                    .into_iter()
                    .modulate(bpsk)
                    .differential_encode(),
            )
            .add_rayleigh_block::<64, _>(rng)
            .differential_decode()
            .skip(1)
            .demodulate_hard(bpsk)
            .collect();

        assert_eq!(recovered, payload);
    }

    // =========================================================================
    // 5. Panic & Boundary Invariant Tests
    // =========================================================================

    #[test]
    #[should_panic(expected = "Block fading coherence N must be at least 1")]
    fn test_block_fading_zero_coherence_panics() {
        let _ = BlockFading::<f32, 0>::new();
    }

    #[test]
    #[should_panic(expected = "Rician K-factor must be non-negative")]
    fn test_rician_negative_k_panics() {
        let _ = RicianModel::<f32>::from_k_linear(-0.5, 0.0);
    }

    #[test]
    fn test_block_fading_coherence_one_valid() {
        let rng = Xoshiro256PlusPlus::seed_from_u64(1234);
        let mut channel = BlockRayleigh::<f64, _, 1>::new_block(rng);
        let pt = Complex::new(1.0, 0.0);

        let s0 = channel.apply_point_with_csi(pt);
        let s1 = channel.apply_point_with_csi(pt);

        // With N = 1, gain must resample on every symbol
        assert_ne!(s0.csi, s1.csi);
    }

    // =========================================================================
    // 6. High-SNR Channel + AWGN Pipeline with ZF Equalization
    // =========================================================================

    #[test]
    fn test_high_snr_fading_with_awgn_recovery() {
        let payload = vec![0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC];
        let qpsk = Qpsk::<f64>::QPSK;
        let rng_fading = Xoshiro256PlusPlus::seed_from_u64(101);
        let mut rng_awgn = Xoshiro256PlusPlus::seed_from_u64(202);

        // 35 dB SNR per symbol
        let snr_lin = 10.0f64.powf(35.0 / 10.0);
        let sigma_axis = (1.0 / (2.0 * snr_lin)).sqrt();

        let recovered: Vec<u8> = payload
            .clone()
            .into_iter()
            .modulate(qpsk)
            .add_rayleigh_csi(rng_fading)
            .map(|faded| {
                let z_i: f64 = StandardNormal.sample(&mut rng_awgn);
                let z_q: f64 = StandardNormal.sample(&mut rng_awgn);
                FadedSymbol {
                    sample: faded.sample + Complex::new(z_i * sigma_axis, z_q * sigma_axis),
                    csi: faded.csi,
                }
            })
            .equalize_zf()
            .demodulate_hard(qpsk)
            .collect();

        assert_eq!(recovered, payload);
    }
}
