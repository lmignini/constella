//! Streaming digital modulation pipelines.
//!
//! This module provides zero-allocation iterator adapters that transform arbitrary
//! byte streams ([`Iterator<Item = u8>`]) into baseband complex symbols ([`Complex<T>`]).
//!
//! # Usage Patterns
//!
//! ### 1. Stream Extension (Recommended)
//! ```rust
//! use constella::modulation::ModulateExt;
//! use constella::constellation::Qpsk;
//!
//! let bytes = [0xAA, 0x55];
//! let symbols: Vec<_> = bytes.into_iter().modulate(Qpsk::<f32>::QPSK).collect();
//! ```
//!
//! ### 2. Constellation Method
//! ```rust
//! use constella::constellation::Qam16;
//!
//! let bytes = [0x12, 0x34];
//! let symbols: Vec<_> = Qam16::<f32>::QAM16.modulate(bytes.into_iter()).collect();
//! ```

use crate::bits::{BitChunker, BitOrder, MsbFirst, PadZeros, Padding};
use crate::constellation::{Constellation, Normalized};
use num_complex::Complex;

/// A trait connecting a normalized constellation of size `M` to its corresponding bit-width `K`.
///
/// This abstraction enables static type inference so callers do not need to manually
/// specify the const parameter `K` ($\log_2 M$) when constructing modulation pipelines.
pub trait Modulatable<I, T, O = MsbFirst, P = PadZeros>
where
    I: Iterator<Item = u8>,
    O: BitOrder,
    P: Padding,
    T: Copy,
{
    /// The resulting streaming modulation iterator type.
    type Output: Iterator<Item = Complex<T>>;

    /// Modulates an iterator of raw bytes using default [`MsbFirst`] ordering and [`PadZeros`] padding.
    fn modulate(self, iter: I) -> Self::Output;

    /// Modulates an iterator of raw bytes using explicit [`BitOrder`] and [`Padding`] strategies.
    fn modulate_with(self, iter: I) -> Self::Output;
}

macro_rules! impl_modulatable_sizes {
    ($( ($m:expr, $k:expr) ),* $(,)?) => {
        $(
            impl<T: Copy> Constellation<T, $m, Normalized> {
                /// Modulates an iterator of raw bytes into complex constellation points.
                ///
                /// Uses default [`MsbFirst`] bit ordering and [`PadZeros`] padding.
                #[inline]
                pub fn modulate<I: Iterator<Item = u8>>(
                    self,
                    iter: I,
                ) -> ModulationIter<I, T, $m, $k, MsbFirst, PadZeros> {
                    ModulationIter::new(iter, self)
                }

                /// Modulates an iterator of raw bytes using explicit [`BitOrder`] and [`Padding`] strategies.
                #[inline]
                pub fn modulate_with<I: Iterator<Item = u8>, O: BitOrder, P: Padding>(
                    self,
                    iter: I,
                ) -> ModulationIter<I, T, $m, $k, O, P> {
                    ModulationIter::with_order_and_padding(iter, self)
                }
            }

            impl<I, T, O, P> Modulatable<I, T, O, P> for Constellation<T, $m, Normalized>
            where
                I: Iterator<Item = u8>,
                O: BitOrder,
                P: Padding,
                T: Copy,
            {
                type Output = ModulationIter<I, T, $m, $k, O, P>;

                #[inline]
                fn modulate(self, iter: I) -> Self::Output {
                    ModulationIter::with_order_and_padding(iter, self)
                }

                #[inline]
                fn modulate_with(self, iter: I) -> Self::Output {
                    ModulationIter::with_order_and_padding(iter, self)
                }
            }
        )*
    };
}

impl_modulatable_sizes!(
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

/// Extension trait providing fluent `.modulate(...)` syntax directly on byte iterators.
pub trait ModulateExt: Iterator<Item = u8> + Sized {
    /// Modulates the byte stream using the given normalized constellation with default [`MsbFirst`] and [`PadZeros`].
    #[inline]
    fn modulate<C, T>(self, constellation: C) -> C::Output
    where
        T: Copy,
        C: Modulatable<Self, T, MsbFirst, PadZeros>,
    {
        constellation.modulate(self)
    }

    /// Modulates the byte stream using custom bit order and padding policies.
    #[inline]
    fn modulate_with<C, T, O: BitOrder, P: Padding>(self, constellation: C) -> C::Output
    where
        T: Copy,
        C: Modulatable<Self, T, O, P>,
    {
        constellation.modulate_with(self)
    }
}

impl<I: Iterator<Item = u8>> ModulateExt for I {}

/// A zero-allocation streaming iterator that transforms sliced bits into baseband complex symbols.
///
/// `ModulationIter` continuously consumes chunks of $K$ bits from an inner [`BitChunker`]
/// and maps them to complex points in $O(1)$ time via a static [`Constellation`] lookup table.
///
/// # Type Parameters
///
/// * `I`: The underlying byte iterator source.
/// * `T`: Float scalar coordinate type (`f32` or `f64`).
/// * `M`: Constellation size (number of points, e.g. 2, 4, 16).
/// * `K`: Bits per constellation symbol ($K = \log_2 M$).
/// * `O`: Bit endianness strategy ([`MsbFirst`] or [`LsbFirst`]). Defaults to [`MsbFirst`].
/// * `P`: Trailing bit padding strategy ([`PadZeros`], [`DiscardRemainder`], or [`ExactOnly`]). Defaults to [`PadZeros`].
pub struct ModulationIter<I, T, const M: usize, const K: usize, O = MsbFirst, P = PadZeros> {
    bit_chunker: BitChunker<I, K, O, P>,
    constellation: Constellation<T, M, Normalized>,
}

impl<I, T, const M: usize, const K: usize> ModulationIter<I, T, M, K, MsbFirst, PadZeros>
where
    I: Iterator<Item = u8>,
    T: Copy,
{
    /// Creates a modulation iterator with default [`MsbFirst`] ordering and [`PadZeros`] padding.
    ///
    /// # Panics
    ///
    /// Panics if $2^K \neq M$.
    #[inline]
    pub fn new(iter: I, constellation: Constellation<T, M, Normalized>) -> Self {
        Self::with_order_and_padding(iter, constellation)
    }
}

impl<I, T, const M: usize, const K: usize, O: BitOrder, P: Padding> ModulationIter<I, T, M, K, O, P>
where
    I: Iterator<Item = u8>,
    T: Copy,
{
    /// Creates a modulation iterator with custom [`BitOrder`] and [`Padding`] strategies.
    ///
    /// # Panics
    ///
    /// Panics if $2^K \neq M$.
    #[inline]
    pub fn with_order_and_padding(iter: I, constellation: Constellation<T, M, Normalized>) -> Self {
        assert_eq!(
            1 << K,
            M,
            "Constellation size M ({M}) must equal 2^K (2^{K} = {})",
            1 << K
        );
        Self {
            bit_chunker: BitChunker::with_order_and_padding(iter),
            constellation,
        }
    }
}

impl<I, T, const M: usize, const K: usize, O, P> Iterator for ModulationIter<I, T, M, K, O, P>
where
    I: Iterator<Item = u8>,
    O: BitOrder,
    P: Padding,
    T: Copy,
{
    type Item = Complex<T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let symbol_index = self.bit_chunker.next()?;
        Some(self.constellation[symbol_index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::{DiscardRemainder, LsbFirst};
    use crate::constellation::{Bpsk, Psk8, Qam16, Qpsk};
    use alloc::vec::Vec;

    const EPS_F32: f32 = 1e-6;

    fn approx_eq_c32(a: Complex<f32>, b: Complex<f32>) -> bool {
        (a.re - b.re).abs() < EPS_F32 && (a.im - b.im).abs() < EPS_F32
    }

    #[test]
    fn test_bpsk_stream_modulation() {
        let data = [0b10110000]; // 8 BPSK symbols
        let symbols: Vec<Complex<f32>> = data.into_iter().modulate(Bpsk::<f32>::BPSK).collect();

        assert_eq!(symbols.len(), 8);
        let bpsk = Bpsk::<f32>::BPSK;

        assert!(approx_eq_c32(symbols[0], bpsk[1])); // bit 1
        assert!(approx_eq_c32(symbols[1], bpsk[0])); // bit 0
        assert!(approx_eq_c32(symbols[2], bpsk[1])); // bit 1
        assert!(approx_eq_c32(symbols[3], bpsk[1])); // bit 1
        assert!(approx_eq_c32(symbols[4], bpsk[0])); // bit 0
        assert!(approx_eq_c32(symbols[5], bpsk[0])); // bit 0
        assert!(approx_eq_c32(symbols[6], bpsk[0])); // bit 0
        assert!(approx_eq_c32(symbols[7], bpsk[0])); // bit 0
    }

    #[test]
    fn test_qpsk_constellation_modulate_method() {
        let data = [0b11_01_00_10];
        let qpsk = Qpsk::<f32>::QPSK;
        let symbols: Vec<Complex<f32>> = qpsk.modulate(data.into_iter()).collect();

        assert_eq!(symbols.len(), 4);
        assert!(approx_eq_c32(symbols[0], qpsk[0b11]));
        assert!(approx_eq_c32(symbols[1], qpsk[0b01]));
        assert!(approx_eq_c32(symbols[2], qpsk[0b00]));
        assert!(approx_eq_c32(symbols[3], qpsk[0b10]));
    }

    #[test]
    fn test_qam16_modulation() {
        let data = [0xA5]; // 0b1010_0101 -> 0xA (10), 0x5 (5)
        let qam16 = Qam16::<f32>::QAM16;
        let symbols: Vec<Complex<f32>> = data.into_iter().modulate(qam16).collect();

        assert_eq!(symbols.len(), 2);
        assert!(approx_eq_c32(symbols[0], qam16[0xA]));
        assert!(approx_eq_c32(symbols[1], qam16[0x5]));
    }

    #[test]
    fn test_modulate_with_lsb_and_discard_padding() {
        // 1 byte = 8 bits. With 8-PSK (K = 3), DiscardRemainder emits only 2 symbols (6 bits)
        let data = [0b1101_0110];
        let psk8 = Psk8::<f32>::m_psk();

        let symbols: Vec<Complex<f32>> = data
            .into_iter()
            .modulate_with::<_, f32, LsbFirst, DiscardRemainder>(psk8)
            .collect();

        assert_eq!(symbols.len(), 2);
        assert!(approx_eq_c32(symbols[0], psk8[0b110]));
        assert!(approx_eq_c32(symbols[1], psk8[0b010]));
    }

    #[test]
    fn test_empty_stream_modulation() {
        let data: [u8; 0] = [];
        let mut iter = data.into_iter().modulate(Qpsk::<f32>::QPSK);
        assert_eq!(iter.next(), None);
    }
}
