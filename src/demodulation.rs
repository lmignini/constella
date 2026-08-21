//! Streaming digital demodulation pipelines.
//!
//! This module provides zero-allocation iterator adapters that transform baseband
//! complex symbol streams ([`Iterator<Item = Complex<T>>`]) back into symbol indices,
//! reconstructed data byte streams ([`u8`]), or soft Log-Likelihood Ratios (LLRs).
//!
//! # Usage Patterns
//!
//! ### 1. Stream Extension (Recommended)
//! ```rust
//! use constella::modulation::ModulateExt;
//! use constella::demodulation::DemodulateExt;
//! use constella::constellation::Qpsk;
//! use num_complex::Complex;
//!
//! let original_bytes = [0xAA, 0x55, 0x12, 0x34];
//! let qpsk = Qpsk::<f32>::QPSK;
//!
//! // Modulation -> Demodulation Roundtrip
//! let symbols: Vec<Complex<f32>> = original_bytes.into_iter().modulate(qpsk).collect();
//! let recovered_bytes: Vec<u8> = symbols.into_iter().demodulate_hard(qpsk).collect();
//!
//! assert_eq!(original_bytes.to_vec(), recovered_bytes);
//! ```
//!
//! ### 2. Soft Decision Demodulation (LLRs)
//! ```rust
//! use constella::demodulation::DemodulateExt;
//! use constella::constellation::Bpsk;
//! use num_complex::Complex;
//!
//! let received_samples = vec![
//!     Complex::new(0.95, 0.05),   // Close to +1.0 (bit 0)
//!     Complex::new(-1.05, -0.02), // Close to -1.0 (bit 1)
//! ];
//!
//! // Extract scalar LLRs for each bit (positive = 0, negative = 1)
//! let noise_var = 0.1f32;
//! let bit_llrs: Vec<f32> = received_samples
//!     .into_iter()
//!     .demodulate_soft_bits(Bpsk::<f32>::BPSK, noise_var)
//!     .collect();
//!
//! assert!(bit_llrs[0] > 0.0); // Favors bit 0
//! assert!(bit_llrs[1] < 0.0); // Favors bit 1
//! ```

use crate::bits::{BitOrder, BitPacker, MsbFirst};
use crate::constellation::{Constellation, Normalized};
use core::marker::PhantomData;
use num_complex::Complex;

/// Internal trait bridging constellation point distance metrics to scalar float precision.
pub trait DemodPoint<T> {
    /// Demodulates a single complex baseband sample to its nearest-neighbor symbol index.
    fn demod_hard(&self, point: Complex<T>) -> usize;
}

/// Internal trait bridging soft Log-Likelihood Ratio computations to scalar float precision.
pub trait SoftDemodPoint<T, const K: usize, O: BitOrder> {
    /// Computes soft Log-Likelihood Ratios for all $K$ bits of a received complex sample.
    fn demod_soft(&self, point: Complex<T>, noise_var: T) -> [T; K];
}

impl<const M: usize> DemodPoint<f32> for Constellation<f32, M, Normalized> {
    #[inline]
    fn demod_hard(&self, point: Complex<f32>) -> usize {
        self.demodulate_hard_point(point)
    }
}

impl<const M: usize> DemodPoint<f64> for Constellation<f64, M, Normalized> {
    #[inline]
    fn demod_hard(&self, point: Complex<f64>) -> usize {
        self.demodulate_hard_point(point)
    }
}

impl<const M: usize, const K: usize, O: BitOrder + 'static> SoftDemodPoint<f32, K, O>
    for Constellation<f32, M, Normalized>
{
    #[inline]
    fn demod_soft(&self, point: Complex<f32>, noise_var: f32) -> [f32; K] {
        self.demodulate_soft_point::<K, O>(point, noise_var)
    }
}

impl<const M: usize, const K: usize, O: BitOrder + 'static> SoftDemodPoint<f64, K, O>
    for Constellation<f64, M, Normalized>
{
    #[inline]
    fn demod_soft(&self, point: Complex<f64>, noise_var: f64) -> [f64; K] {
        self.demodulate_soft_point::<K, O>(point, noise_var)
    }
}

/// A trait connecting a normalized constellation of size `M` to hard-decision demodulation pipelines.
pub trait Demodulatable<I, T, O = MsbFirst>
where
    I: Iterator<Item = Complex<T>>,
    O: BitOrder,
    T: Copy,
{
    /// The resulting streaming byte demodulator iterator type.
    type HardByteOutput: Iterator<Item = u8>;
    /// The resulting streaming symbol index demodulator iterator type.
    type HardSymbolOutput: Iterator<Item = usize>;

    /// Demodulates an iterator of complex baseband symbols into reconstructed bytes using default [`MsbFirst`] ordering.
    fn demodulate_hard(self, iter: I) -> Self::HardByteOutput;

    /// Demodulates an iterator of complex baseband symbols into reconstructed bytes using explicit [`BitOrder`].
    fn demodulate_hard_with(self, iter: I) -> Self::HardByteOutput;

    /// Demodulates an iterator of complex baseband symbols into raw integer symbol indices.
    fn demodulate_hard_symbols(self, iter: I) -> Self::HardSymbolOutput;
}

/// A trait connecting a normalized constellation of size `M` to soft-decision LLR demodulation pipelines.
pub trait SoftDemodulatable<I, T, const K: usize, O = MsbFirst>
where
    I: Iterator<Item = Complex<T>>,
    O: BitOrder + 'static,
    T: Copy + Default,
{
    /// The resulting symbol-level LLR array iterator type.
    type SoftSymbolOutput: Iterator<Item = [T; K]>;
    /// The resulting flattened bit-level scalar LLR iterator type.
    type SoftBitOutput: Iterator<Item = T>;

    /// Demodulates baseband symbols into symbol-level LLR arrays using default [`MsbFirst`] ordering.
    fn demodulate_soft(self, iter: I, noise_var: T) -> Self::SoftSymbolOutput;

    /// Demodulates baseband symbols into symbol-level LLR arrays using explicit [`BitOrder`].
    fn demodulate_soft_with(self, iter: I, noise_var: T) -> Self::SoftSymbolOutput;

    /// Demodulates baseband symbols into a continuous stream of scalar bit LLRs using default [`MsbFirst`] ordering.
    fn demodulate_soft_bits(self, iter: I, noise_var: T) -> Self::SoftBitOutput;

    /// Demodulates baseband symbols into a continuous stream of scalar bit LLRs using explicit [`BitOrder`].
    fn demodulate_soft_bits_with(self, iter: I, noise_var: T) -> Self::SoftBitOutput;
}

macro_rules! impl_demodulatable_sizes {
    ($( ($m:expr, $k:expr) ),* $(,)?) => {
        $(
            impl<T: Copy> Constellation<T, $m, Normalized>
            where
                Self: DemodPoint<T>,
            {
                /// Demodulates an iterator of complex baseband symbols into reconstructed bytes.
                ///
                /// Uses default [`MsbFirst`] bit ordering.
                #[inline]
                pub fn demodulate_hard<I: Iterator<Item = Complex<T>>>(
                    self,
                    iter: I,
                ) -> HardByteDemodIter<I, T, $m, $k, MsbFirst> {
                    HardByteDemodIter::new(iter, self)
                }

                /// Demodulates an iterator of complex baseband symbols into reconstructed bytes using explicit [`BitOrder`].
                #[inline]
                pub fn demodulate_hard_with<I: Iterator<Item = Complex<T>>, O: BitOrder>(
                    self,
                    iter: I,
                ) -> HardByteDemodIter<I, T, $m, $k, O> {
                    HardByteDemodIter::with_order(iter, self)
                }

                /// Demodulates an iterator of complex baseband symbols into integer symbol indices ($0 \le \text{idx} < M$).
                #[inline]
                pub fn demodulate_hard_symbols<I: Iterator<Item = Complex<T>>>(
                    self,
                    iter: I,
                ) -> HardSymbolDemodIter<I, T, $m> {
                    HardSymbolDemodIter::new(iter, self)
                }
            }

            impl<T: Copy + Default> Constellation<T, $m, Normalized> {
                /// Demodulates an iterator of complex baseband symbols into symbol Log-Likelihood Ratio arrays ($[T; K]$).
                ///
                /// Uses default [`MsbFirst`] bit ordering and Max-Log LLR approximation.
                #[inline]
                pub fn demodulate_soft<I: Iterator<Item = Complex<T>>>(
                    self,
                    iter: I,
                    noise_var: T,
                ) -> SoftSymbolDemodIter<I, T, $m, $k, MsbFirst>
                where
                    Self: SoftDemodPoint<T, $k, MsbFirst>,
                {
                    SoftSymbolDemodIter::new(iter, self, noise_var)
                }

                /// Demodulates an iterator of complex baseband symbols into symbol Log-Likelihood Ratio arrays ($[T; K]$) with explicit [`BitOrder`].
                #[inline]
                pub fn demodulate_soft_with<I: Iterator<Item = Complex<T>>, O: BitOrder + 'static>(
                    self,
                    iter: I,
                    noise_var: T,
                ) -> SoftSymbolDemodIter<I, T, $m, $k, O>
                where
                    Self: SoftDemodPoint<T, $k, O>,
                {
                    SoftSymbolDemodIter::with_order(iter, self, noise_var)
                }

                /// Demodulates an iterator of complex baseband symbols into a continuous stream of scalar bit LLRs.
                #[inline]
                pub fn demodulate_soft_bits<I: Iterator<Item = Complex<T>>>(
                    self,
                    iter: I,
                    noise_var: T,
                ) -> SoftBitDemodIter<I, T, $m, $k, MsbFirst>
                where
                    Self: SoftDemodPoint<T, $k, MsbFirst>,
                {
                    SoftBitDemodIter::new(iter, self, noise_var)
                }

                /// Demodulates an iterator of complex baseband symbols into a continuous stream of scalar bit LLRs with explicit [`BitOrder`].
                #[inline]
                pub fn demodulate_soft_bits_with<I: Iterator<Item = Complex<T>>, O: BitOrder + 'static>(
                    self,
                    iter: I,
                    noise_var: T,
                ) -> SoftBitDemodIter<I, T, $m, $k, O>
                where
                    Self: SoftDemodPoint<T, $k, O>,
                {
                    SoftBitDemodIter::with_order(iter, self, noise_var)
                }
            }

            impl<I, T, O> Demodulatable<I, T, O> for Constellation<T, $m, Normalized>
            where
                I: Iterator<Item = Complex<T>>,
                O: BitOrder,
                T: Copy,
                Self: DemodPoint<T>,
            {
                type HardByteOutput = HardByteDemodIter<I, T, $m, $k, O>;
                type HardSymbolOutput = HardSymbolDemodIter<I, T, $m>;

                #[inline]
                fn demodulate_hard(self, iter: I) -> Self::HardByteOutput {
                    HardByteDemodIter::with_order(iter, self)
                }

                #[inline]
                fn demodulate_hard_with(self, iter: I) -> Self::HardByteOutput {
                    HardByteDemodIter::with_order(iter, self)
                }

                #[inline]
                fn demodulate_hard_symbols(self, iter: I) -> Self::HardSymbolOutput {
                    HardSymbolDemodIter::new(iter, self)
                }
            }

            impl<I, T, O> SoftDemodulatable<I, T, $k, O> for Constellation<T, $m, Normalized>
            where
                I: Iterator<Item = Complex<T>>,
                O: BitOrder + 'static,
                T: Copy + Default,
                Self: SoftDemodPoint<T, $k, O>,
            {
                type SoftSymbolOutput = SoftSymbolDemodIter<I, T, $m, $k, O>;
                type SoftBitOutput = SoftBitDemodIter<I, T, $m, $k, O>;

                #[inline]
                fn demodulate_soft(self, iter: I, noise_var: T) -> Self::SoftSymbolOutput {
                    SoftSymbolDemodIter::with_order(iter, self, noise_var)
                }

                #[inline]
                fn demodulate_soft_with(self, iter: I, noise_var: T) -> Self::SoftSymbolOutput {
                    SoftSymbolDemodIter::with_order(iter, self, noise_var)
                }

                #[inline]
                fn demodulate_soft_bits(self, iter: I, noise_var: T) -> Self::SoftBitOutput {
                    SoftBitDemodIter::with_order(iter, self, noise_var)
                }

                #[inline]
                fn demodulate_soft_bits_with(self, iter: I, noise_var: T) -> Self::SoftBitOutput {
                    SoftBitDemodIter::with_order(iter, self, noise_var)
                }
            }
        )*
    };
}

impl_demodulatable_sizes!(
    (2, 1),
    (4, 2),
    (8, 3),
    (16, 4),
    (32, 5),
    (64, 6),
    (128, 7),
    (256, 8),
    (512, 9),
    (1024, 10),
    (2048, 11),
    (4096, 12),
);

/// Extension trait providing fluent demodulation syntax directly on complex symbol iterators.
pub trait DemodulateExt<T: Copy>: Iterator<Item = Complex<T>> + Sized {
    /// Demodulates the symbol stream into reconstructed data bytes using default [`MsbFirst`].
    #[inline]
    fn demodulate_hard<C>(self, constellation: C) -> C::HardByteOutput
    where
        C: Demodulatable<Self, T, MsbFirst>,
    {
        constellation.demodulate_hard(self)
    }

    /// Demodulates the symbol stream into reconstructed data bytes with custom [`BitOrder`].
    #[inline]
    fn demodulate_hard_with<C, O: BitOrder>(self, constellation: C) -> C::HardByteOutput
    where
        C: Demodulatable<Self, T, O>,
    {
        constellation.demodulate_hard_with(self)
    }

    /// Demodulates the symbol stream into nearest-neighbor integer symbol indices ($0 \le \text{idx} < M$).
    #[inline]
    fn demodulate_hard_symbols<C>(self, constellation: C) -> C::HardSymbolOutput
    where
        C: Demodulatable<Self, T, MsbFirst>,
    {
        constellation.demodulate_hard_symbols(self)
    }

    /// Demodulates the symbol stream into arrays of soft Log-Likelihood Ratios ($[T; K]$).
    #[inline]
    fn demodulate_soft<C, const K: usize>(
        self,
        constellation: C,
        noise_var: T,
    ) -> C::SoftSymbolOutput
    where
        T: Default,
        C: SoftDemodulatable<Self, T, K, MsbFirst>,
    {
        constellation.demodulate_soft(self, noise_var)
    }

    /// Demodulates the symbol stream into arrays of soft Log-Likelihood Ratios ($[T; K]$) with custom [`BitOrder`].
    #[inline]
    fn demodulate_soft_with<C, const K: usize, O: BitOrder + 'static>(
        self,
        constellation: C,
        noise_var: T,
    ) -> C::SoftSymbolOutput
    where
        T: Default,
        C: SoftDemodulatable<Self, T, K, O>,
    {
        constellation.demodulate_soft_with(self, noise_var)
    }

    /// Demodulates the symbol stream into a continuous scalar stream of bit Log-Likelihood Ratios.
    #[inline]
    fn demodulate_soft_bits<C, const K: usize>(
        self,
        constellation: C,
        noise_var: T,
    ) -> C::SoftBitOutput
    where
        T: Default,
        C: SoftDemodulatable<Self, T, K, MsbFirst>,
    {
        constellation.demodulate_soft_bits(self, noise_var)
    }

    /// Demodulates the symbol stream into a continuous scalar stream of bit Log-Likelihood Ratios with custom [`BitOrder`].
    #[inline]
    fn demodulate_soft_bits_with<C, const K: usize, O: BitOrder + 'static>(
        self,
        constellation: C,
        noise_var: T,
    ) -> C::SoftBitOutput
    where
        T: Default,
        C: SoftDemodulatable<Self, T, K, O>,
    {
        constellation.demodulate_soft_bits_with(self, noise_var)
    }
}

impl<T: Copy, I: Iterator<Item = Complex<T>>> DemodulateExt<T> for I {}

/// A zero-allocation streaming iterator that performs hard nearest-neighbor slicing on baseband symbols.
pub struct HardSymbolDemodIter<I, T, const M: usize> {
    iter: I,
    constellation: Constellation<T, M, Normalized>,
}

impl<I, T, const M: usize> HardSymbolDemodIter<I, T, M> {
    /// Creates a new `HardSymbolDemodIter`.
    #[inline]
    pub fn new(iter: I, constellation: Constellation<T, M, Normalized>) -> Self {
        Self {
            iter,
            constellation,
        }
    }
}

impl<I, T, const M: usize> Iterator for HardSymbolDemodIter<I, T, M>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy,
    Constellation<T, M, Normalized>: DemodPoint<T>,
{
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let point = self.iter.next()?;
        Some(self.constellation.demod_hard(point))
    }
}

/// A zero-allocation streaming iterator that demodulates baseband complex symbols directly into reconstructed `u8` bytes.
pub struct HardByteDemodIter<I, T, const M: usize, const K: usize, O = MsbFirst> {
    packer: BitPacker<HardSymbolDemodIter<I, T, M>, K, O>,
}

impl<I, T, const M: usize, const K: usize> HardByteDemodIter<I, T, M, K, MsbFirst>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy,
    Constellation<T, M, Normalized>: DemodPoint<T>,
{
    /// Creates a new hard byte demodulation iterator using default [`MsbFirst`] ordering.
    #[inline]
    pub fn new(iter: I, constellation: Constellation<T, M, Normalized>) -> Self {
        Self::with_order(iter, constellation)
    }
}

impl<I, T, const M: usize, const K: usize, O: BitOrder> HardByteDemodIter<I, T, M, K, O>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy,
    Constellation<T, M, Normalized>: DemodPoint<T>,
{
    /// Creates a new hard byte demodulation iterator using explicit [`BitOrder`].
    #[inline]
    pub fn with_order(iter: I, constellation: Constellation<T, M, Normalized>) -> Self {
        assert_eq!(
            1 << K,
            M,
            "Constellation size M ({M}) must equal 2^K (2^{K} = {})",
            1 << K
        );
        let symbol_iter = HardSymbolDemodIter::new(iter, constellation);
        Self {
            packer: BitPacker::with_order(symbol_iter),
        }
    }
}

impl<I, T, const M: usize, const K: usize, O> Iterator for HardByteDemodIter<I, T, M, K, O>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy,
    O: BitOrder,
    Constellation<T, M, Normalized>: DemodPoint<T>,
{
    type Item = u8;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.packer.next()
    }
}

/// A zero-allocation streaming iterator that outputs arrays of $K$ bit LLRs for each received baseband symbol.
pub struct SoftSymbolDemodIter<I, T, const M: usize, const K: usize, O = MsbFirst> {
    iter: I,
    constellation: Constellation<T, M, Normalized>,
    noise_var: T,
    _marker: PhantomData<O>,
}

impl<I, T, const M: usize, const K: usize> SoftSymbolDemodIter<I, T, M, K, MsbFirst>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy,
{
    /// Creates a new soft symbol demodulation iterator using default [`MsbFirst`] ordering.
    #[inline]
    pub fn new(iter: I, constellation: Constellation<T, M, Normalized>, noise_var: T) -> Self {
        Self::with_order(iter, constellation, noise_var)
    }
}

impl<I, T, const M: usize, const K: usize, O: BitOrder> SoftSymbolDemodIter<I, T, M, K, O>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy,
{
    /// Creates a new soft symbol demodulation iterator using explicit [`BitOrder`].
    #[inline]
    pub fn with_order(
        iter: I,
        constellation: Constellation<T, M, Normalized>,
        noise_var: T,
    ) -> Self {
        assert_eq!(
            1 << K,
            M,
            "Constellation size M ({M}) must equal 2^K (2^{K} = {})",
            1 << K
        );
        Self {
            iter,
            constellation,
            noise_var,
            _marker: PhantomData,
        }
    }
}

impl<I, T, const M: usize, const K: usize, O> Iterator for SoftSymbolDemodIter<I, T, M, K, O>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy,
    O: BitOrder + 'static,
    Constellation<T, M, Normalized>: SoftDemodPoint<T, K, O>,
{
    type Item = [T; K];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let point = self.iter.next()?;
        Some(self.constellation.demod_soft(point, self.noise_var))
    }
}

/// A zero-allocation streaming iterator that outputs scalar bit LLRs one by one.
pub struct SoftBitDemodIter<I, T, const M: usize, const K: usize, O = MsbFirst> {
    symbol_iter: SoftSymbolDemodIter<I, T, M, K, O>,
    buffer: [T; K],
    index: usize,
}

impl<I, T, const M: usize, const K: usize> SoftBitDemodIter<I, T, M, K, MsbFirst>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy + Default,
{
    /// Creates a new scalar soft bit demodulation iterator using default [`MsbFirst`] ordering.
    #[inline]
    pub fn new(iter: I, constellation: Constellation<T, M, Normalized>, noise_var: T) -> Self {
        Self::with_order(iter, constellation, noise_var)
    }
}

impl<I, T, const M: usize, const K: usize, O: BitOrder> SoftBitDemodIter<I, T, M, K, O>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy + Default,
{
    /// Creates a new scalar soft bit demodulation iterator using explicit [`BitOrder`].
    #[inline]
    pub fn with_order(
        iter: I,
        constellation: Constellation<T, M, Normalized>,
        noise_var: T,
    ) -> Self {
        Self {
            symbol_iter: SoftSymbolDemodIter::with_order(iter, constellation, noise_var),
            buffer: [T::default(); K],
            index: K,
        }
    }
}

impl<I, T, const M: usize, const K: usize, O> Iterator for SoftBitDemodIter<I, T, M, K, O>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy + Default,
    O: BitOrder + 'static,
    Constellation<T, M, Normalized>: SoftDemodPoint<T, K, O>,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < K {
            let val = self.buffer[self.index];
            self.index += 1;
            Some(val)
        } else {
            self.buffer = self.symbol_iter.next()?;
            self.index = 1;
            Some(self.buffer[0])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::{LsbFirst, PadZeros};
    use crate::constellation::{Bpsk, Psk8, Qam16, Qpsk};
    use crate::modulation::ModulateExt;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn test_hard_demod_bpsk_roundtrip() {
        let original = vec![0b10110001, 0b11001010, 0b00110101];
        let bpsk = Bpsk::<f32>::BPSK;

        let symbols: Vec<Complex<f32>> = original.clone().into_iter().modulate(bpsk).collect();
        let recovered: Vec<u8> = symbols.into_iter().demodulate_hard(bpsk).collect();

        assert_eq!(recovered, original);
    }

    #[test]
    fn test_hard_demod_qpsk_roundtrip() {
        let original = vec![0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];
        let qpsk = Qpsk::<f32>::QPSK;

        let symbols: Vec<Complex<f32>> = original.clone().into_iter().modulate(qpsk).collect();
        let recovered: Vec<u8> = symbols.into_iter().demodulate_hard(qpsk).collect();

        assert_eq!(recovered, original);
    }

    #[test]
    fn test_hard_demod_8psk_spanning_roundtrip() {
        // 3 bytes = 24 bits = exactly eight 3-bit 8-PSK symbols
        let original = vec![0b101_100_11, 0b0_101_110_0, 0b11_000_111];
        let psk8 = Psk8::<f64>::m_psk();

        let symbols: Vec<Complex<f64>> = original.clone().into_iter().modulate(psk8).collect();
        let recovered: Vec<u8> = symbols.into_iter().demodulate_hard(psk8).collect();

        assert_eq!(recovered, original);
    }

    #[test]
    fn test_hard_demod_qam16_roundtrip() {
        let original = vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE];
        let qam16 = Qam16::<f32>::QAM16;

        let symbols: Vec<Complex<f32>> = original.clone().into_iter().modulate(qam16).collect();
        let recovered: Vec<u8> = symbols.into_iter().demodulate_hard(qam16).collect();

        assert_eq!(recovered, original);
    }

    #[test]
    fn test_hard_demod_lsb_roundtrip() {
        let original = vec![0xA5, 0x5A, 0x3C, 0xC3];
        let qpsk = Qpsk::<f32>::QPSK;

        let symbols: Vec<Complex<f32>> = original
            .clone()
            .into_iter()
            .modulate_with::<_, f32, LsbFirst, PadZeros>(qpsk)
            .collect();

        let recovered: Vec<u8> = symbols
            .into_iter()
            .demodulate_hard_with::<_, LsbFirst>(qpsk)
            .collect();

        assert_eq!(recovered, original);
    }

    #[test]
    fn test_noisy_channel_hard_demodulation() {
        let original = vec![0b11_01_00_10];
        let qpsk = Qpsk::<f32>::QPSK;

        // Add small perturbation noise that stays strictly within decision boundaries
        let noisy_symbols: Vec<Complex<f32>> = original
            .clone()
            .into_iter()
            .modulate(qpsk)
            .map(|s| s + Complex::new(0.05, -0.05))
            .collect();

        let recovered: Vec<u8> = noisy_symbols.into_iter().demodulate_hard(qpsk).collect();
        assert_eq!(recovered, original);
    }

    #[test]
    fn test_soft_demod_llr_signs() {
        let bpsk = Bpsk::<f32>::BPSK;
        // BPSK: point 0 is at +1.0 (bit 0), point 1 is at -1.0 (bit 1)
        let samples = vec![Complex::new(0.85, 0.0), Complex::new(-0.85, 0.0)];

        let llrs: Vec<[f32; 1]> = samples.into_iter().demodulate_soft(bpsk, 0.5f32).collect();

        assert_eq!(llrs.len(), 2);
        assert!(
            llrs[0][0] > 0.0,
            "Point near +1.0 must have positive LLR for bit 0"
        );
        assert!(
            llrs[1][0] < 0.0,
            "Point near -1.0 must have negative LLR (favoring bit 1)"
        );
    }

    #[test]
    fn test_soft_bit_streaming_flattening() {
        let original = vec![0b11_01_00_10]; // 4 QPSK symbols -> 8 bits
        let qpsk = Qpsk::<f32>::QPSK;

        let symbols: Vec<Complex<f32>> = original.into_iter().modulate(qpsk).collect();
        let bit_llrs: Vec<f32> = symbols
            .into_iter()
            .demodulate_soft_bits(qpsk, 0.5f32)
            .collect();

        assert_eq!(bit_llrs.len(), 8);

        // Map LLR sign back to hard decision (LLR >= 0 -> bit 0, LLR < 0 -> bit 1)
        let hard_bits: Vec<u8> = bit_llrs
            .into_iter()
            .map(|llr| if llr >= 0.0 { 0 } else { 1 })
            .collect();

        let expected_bits = vec![1, 1, 0, 1, 0, 0, 1, 0];
        assert_eq!(hard_bits, expected_bits);
    }

    #[test]
    fn test_hard_symbol_indices_direct() {
        let qpsk = Qpsk::<f32>::QPSK;
        let points = vec![qpsk[0], qpsk[1], qpsk[2], qpsk[3]];

        let indices: Vec<usize> = points.into_iter().demodulate_hard_symbols(qpsk).collect();

        assert_eq!(indices, vec![0, 1, 2, 3]);
    }
}
