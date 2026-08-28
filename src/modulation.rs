use crate::bits::DirectBitCodec;
use crate::bits::{BitChunker, BitOrder, MsbFirst, PadZeros, Padding};
use crate::constellation::{Constellation, ConstellationGeometry, General, Normalized};
use core::marker::PhantomData;
use num_complex::Complex;

pub trait Modulatable<I, T, O = MsbFirst, P = PadZeros>
where
    I: Iterator<Item = u8>,
    O: BitOrder,
    P: Padding,
    T: Copy,
{
    type Output: Iterator<Item = Complex<T>>;
    fn modulate(self, iter: I) -> Self::Output;
    fn modulate_with(self, iter: I) -> Self::Output;
}

macro_rules! impl_direct_modulatable_sizes {
    ($( ($m:expr, $k:expr, $n:expr) ),* $(,)?) => {
        $(
            impl<T: Copy + Default, G: ConstellationGeometry> Constellation<T, $m, Normalized, G> {
                #[inline]
                pub fn modulate<'c, I: Iterator<Item = u8>>(
                    &'c self,
                    iter: I,
                ) -> DirectModulationIter<'c, I, T, $m, $k, $n, G, MsbFirst>
                where
                    MsbFirst: DirectBitCodec<$k, $n>,
                {
                    DirectModulationIter::new(iter, self)
                }

                #[inline]
                pub fn modulate_with<'c, I: Iterator<Item = u8>, O: BitOrder + DirectBitCodec<$k, $n>, P: Padding>(
                    &'c self,
                    iter: I,
                ) -> DirectModulationIter<'c, I, T, $m, $k, $n, G, O> {
                    DirectModulationIter::with_order(iter, self)
                }
            }

            impl<'c, I, T, G, O, P> Modulatable<I, T, O, P> for &'c Constellation<T, $m, Normalized, G>
            where
                I: Iterator<Item = u8>,
                G: ConstellationGeometry,
                O: BitOrder + DirectBitCodec<$k, $n>,
                P: Padding,
                T: Copy + Default,
            {
                type Output = DirectModulationIter<'c, I, T, $m, $k, $n, G, O>;

                #[inline]
                fn modulate(self, iter: I) -> Self::Output {
                    DirectModulationIter::with_order(iter, self)
                }

                #[inline]
                fn modulate_with(self, iter: I) -> Self::Output {
                    DirectModulationIter::with_order(iter, self)
                }
            }
        )*
    };
}
macro_rules! impl_fallback_modulatable_sizes {
    ($( ($m:expr, $k:expr) ),* $(,)?) => {
        $(
            impl<T: Copy, G: ConstellationGeometry> Constellation<T, $m, Normalized, G> {
                #[inline]
                pub fn modulate<'c, I: Iterator<Item = u8>>(
                    &'c self,
                    iter: I,
                ) -> ModulationIter<'c, I, T, $m, $k, G, MsbFirst, PadZeros> {
                    ModulationIter::new(iter, self)
                }

                #[inline]
                pub fn modulate_with<'c, I: Iterator<Item = u8>, O: BitOrder, P: Padding>(
                    &'c self,
                    iter: I,
                ) -> ModulationIter<'c, I, T, $m, $k, G, O, P> {
                    ModulationIter::with_order_and_padding(iter, self)
                }
            }

            impl<'c, I, T, G, O, P> Modulatable<I, T, O, P> for &'c Constellation<T, $m, Normalized, G>
            where
                I: Iterator<Item = u8>,
                G: ConstellationGeometry,
                O: BitOrder,
                P: Padding,
                T: Copy,
            {
                type Output = ModulationIter<'c, I, T, $m, $k, G, O, P>;

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
impl_direct_modulatable_sizes!(
    (2, 1, 8),   // BPSK
    (4, 2, 4),   // QPSK
    (16, 4, 2),  // 16-QAM
    (256, 8, 1), // 256-QAM
);

impl_fallback_modulatable_sizes!(
    (8, 3),     // 8-PSK
    (32, 5),    // 32-QAM
    (64, 6),    // 64-QAM
    (128, 7),   // 128-QAM
    (512, 9),   // 512-QAM
    (1024, 10), // 1024-QAM
    (2048, 11), // 2048-QAM
    (4096, 12), // 4096-QAM
);
pub trait ModulateExt: Iterator<Item = u8> + Sized {
    #[inline]
    fn modulate<C, T>(self, constellation: C) -> C::Output
    where
        T: Copy,
        C: Modulatable<Self, T, MsbFirst, PadZeros>,
    {
        constellation.modulate(self)
    }

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

pub struct ModulationIter<
    'c,
    I,
    T,
    const M: usize,
    const K: usize,
    G = General,
    O = MsbFirst,
    P = PadZeros,
> where
    G: ConstellationGeometry,
{
    bit_chunker: BitChunker<I, K, O, P>,
    constellation: &'c Constellation<T, M, Normalized, G>,
}

impl<'c, I, T, const M: usize, const K: usize, G: ConstellationGeometry>
    ModulationIter<'c, I, T, M, K, G, MsbFirst, PadZeros>
where
    I: Iterator<Item = u8>,
    T: Copy,
{
    #[inline]
    pub fn new(iter: I, constellation: &'c Constellation<T, M, Normalized, G>) -> Self {
        Self::with_order_and_padding(iter, constellation)
    }
}

impl<'c, I, T, const M: usize, const K: usize, G: ConstellationGeometry, O: BitOrder, P: Padding>
    ModulationIter<'c, I, T, M, K, G, O, P>
where
    I: Iterator<Item = u8>,
    T: Copy,
{
    #[inline]
    pub fn with_order_and_padding(
        iter: I,
        constellation: &'c Constellation<T, M, Normalized, G>,
    ) -> Self {
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

impl<'c, I, T, const M: usize, const K: usize, G, O, P> Iterator
    for ModulationIter<'c, I, T, M, K, G, O, P>
where
    I: Iterator<Item = u8>,
    G: ConstellationGeometry,
    O: BitOrder,
    P: Padding,
    T: Copy,
{
    type Item = Complex<T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let symbol_index = self.bit_chunker.next()?;
        Some(self.constellation[symbol_index & (M - 1)]) // M is a power of two by construction
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.bit_chunker.size_hint()
    }

    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let c = self.constellation;
        self.bit_chunker
            .fold(init, |acc, idx| f(acc, c[idx & (M - 1)]))
    }
}

pub struct DirectModulationIter<
    'c,
    I,
    T,
    const M: usize,
    const K: usize,
    const N: usize,
    G = General,
    O = MsbFirst,
> where
    I: Iterator<Item = u8>,
    G: ConstellationGeometry,
    T: Copy,
    O: BitOrder + DirectBitCodec<K, N>,
{
    iter: I,
    constellation: &'c Constellation<T, M, Normalized, G>,
    buffer: [Complex<T>; N],
    index: u8,
    _marker: PhantomData<O>,
}

impl<'c, I, T, const M: usize, const K: usize, const N: usize, G>
    DirectModulationIter<'c, I, T, M, K, N, G, MsbFirst>
where
    I: Iterator<Item = u8>,
    G: ConstellationGeometry,
    T: Copy + Default,
    MsbFirst: DirectBitCodec<K, N>,
{
    #[inline]
    pub fn new(iter: I, constellation: &'c Constellation<T, M, Normalized, G>) -> Self {
        Self::with_order(iter, constellation)
    }
}

impl<'c, I, T, const M: usize, const K: usize, const N: usize, G, O>
    DirectModulationIter<'c, I, T, M, K, N, G, O>
where
    I: Iterator<Item = u8>,
    G: ConstellationGeometry,
    T: Copy + Default,
    O: BitOrder + DirectBitCodec<K, N>,
{
    #[inline]
    pub fn with_order(iter: I, constellation: &'c Constellation<T, M, Normalized, G>) -> Self {
        assert_eq!(1 << K, M, "Constellation size M ({M}) must equal 2^K");
        assert_eq!(K * N, 8, "K * N must equal 8 bits per byte");
        Self {
            iter,
            constellation,
            buffer: [Complex::new(T::default(), T::default()); N],
            index: N as u8, // Set cursor past the end to trigger an immediate byte read
            _marker: PhantomData,
        }
    }
}

impl<'c, I, T, const M: usize, const K: usize, const N: usize, G, O> Iterator
    for DirectModulationIter<'c, I, T, M, K, N, G, O>
where
    I: Iterator<Item = u8>,
    G: ConstellationGeometry,
    T: Copy + Default,
    O: BitOrder + DirectBitCodec<K, N>,
{
    type Item = Complex<T>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if (self.index as usize) < N {
            let item = self.buffer[self.index as usize];
            self.index += 1;
            Some(item)
        } else {
            let byte = self.iter.next()?;
            let indices = O::unpack_byte(byte);

            let mut i = 0;
            while i < N {
                self.buffer[i] = self.constellation[indices[i]];
                i += 1;
            }

            self.index = 1;
            Some(self.buffer[0])
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();
        let buffered = (N).saturating_sub(self.index as usize);

        let lower_total = lower.saturating_mul(N).saturating_add(buffered);
        let upper_total = upper.and_then(|u| u.checked_mul(N)?.checked_add(buffered));

        (lower_total, upper_total)
    }

    #[inline]
    fn fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let mut acc = init;
        while (self.index as usize) < N {
            acc = f(acc, self.buffer[self.index as usize]);
            self.index += 1;
        }

        let c = self.constellation;
        self.iter.fold(acc, |mut acc, byte| {
            let indices = O::unpack_byte(byte);
            let mut i = 0;
            while i < N {
                acc = f(acc, c[indices[i] & (M - 1)]);
                i += 1;
            }
            acc
        })
    }
}

// Empty ExactSizeIterator implementation
impl<'c, I, T, const M: usize, const K: usize, const N: usize, G, O> ExactSizeIterator
    for DirectModulationIter<'c, I, T, M, K, N, G, O>
where
    I: ExactSizeIterator<Item = u8>,
    G: ConstellationGeometry,
    T: Copy + Default,
    O: BitOrder + DirectBitCodec<K, N>,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Qam64;
    use crate::bits::{DiscardRemainder, LsbFirst};
    use crate::constellation::{Bpsk, Psk8, Qam16, Qpsk};
    use alloc::vec;
    use alloc::vec::Vec;

    const EPS_F32: f32 = 1e-6;

    fn approx_eq_c32(a: Complex<f32>, b: Complex<f32>) -> bool {
        (a.re - b.re).abs() < EPS_F32 && (a.im - b.im).abs() < EPS_F32
    }

    #[test]
    fn test_bpsk_stream_modulation() {
        let data = [0b10110000]; // 8 BPSK symbols
        let symbols: Vec<Complex<f32>> = data.into_iter().modulate(&Bpsk::<f32>::BPSK).collect();

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
        let symbols: Vec<Complex<f32>> = data.into_iter().modulate(&qam16).collect();

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
            .modulate_with::<_, f32, LsbFirst, DiscardRemainder>(&psk8)
            .collect();

        assert_eq!(symbols.len(), 2);
        assert!(approx_eq_c32(symbols[0], psk8[0b110]));
        assert!(approx_eq_c32(symbols[1], psk8[0b010]));
    }

    #[test]
    fn test_empty_stream_modulation() {
        let data: [u8; 0] = [];
        let mut iter = data.into_iter().modulate(&Qpsk::<f32>::QPSK);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_8psk_non_aligned_stream_discard_padding() {
        // 2 bytes = 16 bits. For 8-PSK (K = 3): 5 symbols (15 bits) + 1 trailing bit discarded
        let data = [0b1010_1010, 0b1100_1100];
        let psk8 = Psk8::<f32>::m_psk();

        let symbols: Vec<_> = data
            .into_iter()
            .modulate_with::<_, f32, MsbFirst, DiscardRemainder>(&psk8)
            .collect();

        assert_eq!(symbols.len(), 5);
    }

    #[test]
    fn fallback_modulation_reports_size_hint() {
        let data = vec![0xAAu8; 300]; // 2400 bits, K=6 -> 400 symbols
        let it = data.into_iter().modulate(&Qam64::<f32>::QAM64);
        assert_eq!(it.size_hint(), (400, Some(400)));
    }
}
