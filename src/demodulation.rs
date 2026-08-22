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
use crate::constellation::{Constellation, ConstellationGeometry, General, Normalized, SquareQam};
use core::marker::PhantomData;
use num_complex::Complex;

pub trait DemodPoint<T> {
    fn demod_hard(&self, point: Complex<T>) -> usize;
}

pub trait SoftDemodPoint<T, const K: usize, O: BitOrder> {
    fn demod_soft(&self, point: Complex<T>, noise_var: T) -> [T; K];
}

impl<const M: usize> DemodPoint<f32> for Constellation<f32, M, Normalized, General> {
    #[inline(always)]
    fn demod_hard(&self, point: Complex<f32>) -> usize {
        self.demodulate_hard_point(point)
    }
}

impl<const M: usize> DemodPoint<f32> for Constellation<f32, M, Normalized, SquareQam<f32>> {
    #[inline(always)]
    fn demod_hard(&self, point: Complex<f32>) -> usize {
        self.demodulate_hard_point(point)
    }
}

impl<const M: usize> DemodPoint<f64> for Constellation<f64, M, Normalized, General> {
    #[inline(always)]
    fn demod_hard(&self, point: Complex<f64>) -> usize {
        self.demodulate_hard_point(point)
    }
}

impl<const M: usize> DemodPoint<f64> for Constellation<f64, M, Normalized, SquareQam<f64>> {
    #[inline(always)]
    fn demod_hard(&self, point: Complex<f64>) -> usize {
        self.demodulate_hard_point(point)
    }
}

impl<const M: usize, const K: usize, G: ConstellationGeometry, O: BitOrder + 'static>
    SoftDemodPoint<f32, K, O> for Constellation<f32, M, Normalized, G>
{
    #[inline(always)]
    fn demod_soft(&self, point: Complex<f32>, noise_var: f32) -> [f32; K] {
        self.demodulate_soft_point::<K, O>(point, noise_var)
    }
}

impl<const M: usize, const K: usize, G: ConstellationGeometry, O: BitOrder + 'static>
    SoftDemodPoint<f64, K, O> for Constellation<f64, M, Normalized, G>
{
    #[inline(always)]
    fn demod_soft(&self, point: Complex<f64>, noise_var: f64) -> [f64; K] {
        self.demodulate_soft_point::<K, O>(point, noise_var)
    }
}

pub trait Demodulatable<I, T, O = MsbFirst>
where
    I: Iterator<Item = Complex<T>>,
    O: BitOrder,
    T: Copy,
{
    type HardByteOutput: Iterator<Item = u8>;
    type HardSymbolOutput: Iterator<Item = usize>;

    fn demodulate_hard(self, iter: I) -> Self::HardByteOutput;
    fn demodulate_hard_with(self, iter: I) -> Self::HardByteOutput;
    fn demodulate_hard_symbols(self, iter: I) -> Self::HardSymbolOutput;
}

pub trait SoftDemodulatable<I, T, const K: usize, O = MsbFirst>
where
    I: Iterator<Item = Complex<T>>,
    O: BitOrder + 'static,
    T: Copy + Default,
{
    type SoftSymbolOutput: Iterator<Item = [T; K]>;
    type SoftBitOutput: Iterator<Item = T>;

    fn demodulate_soft(self, iter: I, noise_var: T) -> Self::SoftSymbolOutput;
    fn demodulate_soft_with(self, iter: I, noise_var: T) -> Self::SoftSymbolOutput;
    fn demodulate_soft_bits(self, iter: I, noise_var: T) -> Self::SoftBitOutput;
    fn demodulate_soft_bits_with(self, iter: I, noise_var: T) -> Self::SoftBitOutput;
}

macro_rules! impl_demodulatable_sizes {
    ($( ($m:expr, $k:expr) ),* $(,)?) => {
        $(
            impl<T: Copy, G: ConstellationGeometry> Constellation<T, $m, Normalized, G>
            where
                Self: DemodPoint<T>,
            {
                #[inline]
                pub fn demodulate_hard<I: Iterator<Item = Complex<T>>>(
                    self,
                    iter: I,
                ) -> HardByteDemodIter<I, T, $m, $k, G, MsbFirst> {
                    HardByteDemodIter::new(iter, self)
                }

                #[inline]
                pub fn demodulate_hard_with<I: Iterator<Item = Complex<T>>, O: BitOrder>(
                    self,
                    iter: I,
                ) -> HardByteDemodIter<I, T, $m, $k, G, O> {
                    HardByteDemodIter::with_order(iter, self)
                }

                #[inline]
                pub fn demodulate_hard_symbols<I: Iterator<Item = Complex<T>>>(
                    self,
                    iter: I,
                ) -> HardSymbolDemodIter<I, T, $m, G> {
                    HardSymbolDemodIter::new(iter, self)
                }
            }

            impl<T: Copy + Default, G: ConstellationGeometry> Constellation<T, $m, Normalized, G> {
                #[inline]
                pub fn demodulate_soft<I: Iterator<Item = Complex<T>>>(
                    self,
                    iter: I,
                    noise_var: T,
                ) -> SoftSymbolDemodIter<I, T, $m, $k, G, MsbFirst>
                where
                    Self: SoftDemodPoint<T, $k, MsbFirst>,
                {
                    SoftSymbolDemodIter::new(iter, self, noise_var)
                }

                #[inline]
                pub fn demodulate_soft_with<I: Iterator<Item = Complex<T>>, O: BitOrder + 'static>(
                    self,
                    iter: I,
                    noise_var: T,
                ) -> SoftSymbolDemodIter<I, T, $m, $k, G, O>
                where
                    Self: SoftDemodPoint<T, $k, O>,
                {
                    SoftSymbolDemodIter::with_order(iter, self, noise_var)
                }

                #[inline]
                pub fn demodulate_soft_bits<I: Iterator<Item = Complex<T>>>(
                    self,
                    iter: I,
                    noise_var: T,
                ) -> SoftBitDemodIter<I, T, $m, $k, G, MsbFirst>
                where
                    Self: SoftDemodPoint<T, $k, MsbFirst>,
                {
                    SoftBitDemodIter::new(iter, self, noise_var)
                }

                #[inline]
                pub fn demodulate_soft_bits_with<I: Iterator<Item = Complex<T>>, O: BitOrder + 'static>(
                    self,
                    iter: I,
                    noise_var: T,
                ) -> SoftBitDemodIter<I, T, $m, $k, G, O>
                where
                    Self: SoftDemodPoint<T, $k, O>,
                {
                    SoftBitDemodIter::with_order(iter, self, noise_var)
                }
            }

            impl<I, T, G, O> Demodulatable<I, T, O> for Constellation<T, $m, Normalized, G>
            where
                I: Iterator<Item = Complex<T>>,
                G: ConstellationGeometry,
                O: BitOrder,
                T: Copy,
                Self: DemodPoint<T>,
            {
                type HardByteOutput = HardByteDemodIter<I, T, $m, $k, G, O>;
                type HardSymbolOutput = HardSymbolDemodIter<I, T, $m, G>;

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

            impl<I, T, G, O> SoftDemodulatable<I, T, $k, O> for Constellation<T, $m, Normalized, G>
            where
                I: Iterator<Item = Complex<T>>,
                G: ConstellationGeometry,
                O: BitOrder + 'static,
                T: Copy + Default,
                Self: SoftDemodPoint<T, $k, O>,
            {
                type SoftSymbolOutput = SoftSymbolDemodIter<I, T, $m, $k, G, O>;
                type SoftBitOutput = SoftBitDemodIter<I, T, $m, $k, G, O>;

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

pub trait DemodulateExt<T: Copy>: Iterator<Item = Complex<T>> + Sized {
    #[inline]
    fn demodulate_hard<C>(self, constellation: C) -> C::HardByteOutput
    where
        C: Demodulatable<Self, T, MsbFirst>,
    {
        constellation.demodulate_hard(self)
    }

    #[inline]
    fn demodulate_hard_with<C, O: BitOrder>(self, constellation: C) -> C::HardByteOutput
    where
        C: Demodulatable<Self, T, O>,
    {
        constellation.demodulate_hard_with(self)
    }

    #[inline]
    fn demodulate_hard_symbols<C>(self, constellation: C) -> C::HardSymbolOutput
    where
        C: Demodulatable<Self, T, MsbFirst>,
    {
        constellation.demodulate_hard_symbols(self)
    }

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

pub struct HardSymbolDemodIter<I, T, const M: usize, G = General>
where
    G: ConstellationGeometry,
{
    iter: I,
    constellation: Constellation<T, M, Normalized, G>,
}

impl<I, T, const M: usize, G: ConstellationGeometry> HardSymbolDemodIter<I, T, M, G> {
    #[inline]
    pub fn new(iter: I, constellation: Constellation<T, M, Normalized, G>) -> Self {
        Self {
            iter,
            constellation,
        }
    }
}

impl<I, T, const M: usize, G> Iterator for HardSymbolDemodIter<I, T, M, G>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy,
    G: ConstellationGeometry,
    Constellation<T, M, Normalized, G>: DemodPoint<T>,
{
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let point = self.iter.next()?;
        Some(self.constellation.demod_hard(point))
    }
}

pub struct HardByteDemodIter<I, T, const M: usize, const K: usize, G = General, O = MsbFirst>
where
    G: ConstellationGeometry,
{
    packer: BitPacker<HardSymbolDemodIter<I, T, M, G>, K, O>,
}

impl<I, T, const M: usize, const K: usize, G: ConstellationGeometry>
    HardByteDemodIter<I, T, M, K, G, MsbFirst>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy,
    Constellation<T, M, Normalized, G>: DemodPoint<T>,
{
    #[inline]
    pub fn new(iter: I, constellation: Constellation<T, M, Normalized, G>) -> Self {
        Self::with_order(iter, constellation)
    }
}

impl<I, T, const M: usize, const K: usize, G: ConstellationGeometry, O: BitOrder>
    HardByteDemodIter<I, T, M, K, G, O>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy,
    Constellation<T, M, Normalized, G>: DemodPoint<T>,
{
    #[inline]
    pub fn with_order(iter: I, constellation: Constellation<T, M, Normalized, G>) -> Self {
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

impl<I, T, const M: usize, const K: usize, G, O> Iterator for HardByteDemodIter<I, T, M, K, G, O>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy,
    G: ConstellationGeometry,
    O: BitOrder,
    Constellation<T, M, Normalized, G>: DemodPoint<T>,
{
    type Item = u8;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.packer.next()
    }
}

pub struct SoftSymbolDemodIter<I, T, const M: usize, const K: usize, G = General, O = MsbFirst>
where
    G: ConstellationGeometry,
{
    iter: I,
    constellation: Constellation<T, M, Normalized, G>,
    noise_var: T,
    _marker: PhantomData<O>,
}

impl<I, T, const M: usize, const K: usize, G: ConstellationGeometry>
    SoftSymbolDemodIter<I, T, M, K, G, MsbFirst>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy,
{
    #[inline]
    pub fn new(iter: I, constellation: Constellation<T, M, Normalized, G>, noise_var: T) -> Self {
        Self::with_order(iter, constellation, noise_var)
    }
}

impl<I, T, const M: usize, const K: usize, G: ConstellationGeometry, O: BitOrder>
    SoftSymbolDemodIter<I, T, M, K, G, O>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy,
{
    #[inline]
    pub fn with_order(
        iter: I,
        constellation: Constellation<T, M, Normalized, G>,
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

impl<I, T, const M: usize, const K: usize, G, O> Iterator for SoftSymbolDemodIter<I, T, M, K, G, O>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy,
    G: ConstellationGeometry,
    O: BitOrder + 'static,
    Constellation<T, M, Normalized, G>: SoftDemodPoint<T, K, O>,
{
    type Item = [T; K];

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let point = self.iter.next()?;
        Some(self.constellation.demod_soft(point, self.noise_var))
    }
}

pub struct SoftBitDemodIter<I, T, const M: usize, const K: usize, G = General, O = MsbFirst>
where
    G: ConstellationGeometry,
{
    symbol_iter: SoftSymbolDemodIter<I, T, M, K, G, O>,
    buffer: [T; K],
    index: usize,
}

impl<I, T, const M: usize, const K: usize, G: ConstellationGeometry>
    SoftBitDemodIter<I, T, M, K, G, MsbFirst>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy + Default,
{
    #[inline]
    pub fn new(iter: I, constellation: Constellation<T, M, Normalized, G>, noise_var: T) -> Self {
        Self::with_order(iter, constellation, noise_var)
    }
}

impl<I, T, const M: usize, const K: usize, G: ConstellationGeometry, O: BitOrder>
    SoftBitDemodIter<I, T, M, K, G, O>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy + Default,
{
    #[inline]
    pub fn with_order(
        iter: I,
        constellation: Constellation<T, M, Normalized, G>,
        noise_var: T,
    ) -> Self {
        Self {
            symbol_iter: SoftSymbolDemodIter::with_order(iter, constellation, noise_var),
            buffer: [T::default(); K],
            index: K,
        }
    }
}

impl<I, T, const M: usize, const K: usize, G, O> Iterator for SoftBitDemodIter<I, T, M, K, G, O>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy + Default,
    G: ConstellationGeometry,
    O: BitOrder + 'static,
    Constellation<T, M, Normalized, G>: SoftDemodPoint<T, K, O>,
{
    type Item = T;

    #[inline(always)]
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
    use crate::{Qam64, Qam256, Qam4096};
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

    #[test]
    fn test_high_order_qam_roundtrips() {
        let payload: Vec<u8> = (0..120).map(|x| (x * 37) as u8).collect();

        // 64-QAM (K = 6)
        let qam64 = Qam64::<f32>::QAM64;
        let syms64: Vec<_> = payload.clone().into_iter().modulate(qam64).collect();
        let rec64: Vec<u8> = syms64.into_iter().demodulate_hard(qam64).collect();
        assert_eq!(rec64, payload);

        // 256-QAM (K = 8)
        let qam256 = Qam256::<f32>::QAM256;
        let syms256: Vec<_> = payload.clone().into_iter().modulate(qam256).collect();
        let rec256: Vec<u8> = syms256.into_iter().demodulate_hard(qam256).collect();
        assert_eq!(rec256, payload);

        // 4096-QAM (K = 12)
        let qam4096 = Qam4096::<f64>::QAM4096;
        let syms4096: Vec<_> = payload.clone().into_iter().modulate(qam4096).collect();
        let rec4096: Vec<u8> = syms4096.into_iter().demodulate_hard(qam4096).collect();
        assert_eq!(rec4096, payload);
    }

    #[test]
    fn test_custom_user_constellation_general_pipeline() {
        // Custom non-standard 4-point constellation
        let raw_points = [
            Complex::new(2.0f32, 1.0f32),
            Complex::new(-1.0f32, 3.0f32),
            Complex::new(-2.0f32, -2.0f32),
            Complex::new(3.0f32, -1.0f32),
        ];
        let custom = Constellation::<f32, 4>::from_points(raw_points).normalize();

        let data = vec![0x12, 0x34, 0xAB, 0xCD];
        let symbols: Vec<Complex<f32>> = data.clone().into_iter().modulate(custom).collect();
        let recovered: Vec<u8> = symbols.into_iter().demodulate_hard(custom).collect();

        assert_eq!(recovered, data);
    }
}
