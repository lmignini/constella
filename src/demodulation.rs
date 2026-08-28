//! Streaming digital demodulation pipelines.
use crate::RotatedPsk;
use crate::StandardPsk;
use crate::bits::{BitOrder, BitPacker, DirectBitCodec, MsbFirst};
use crate::constellation::{Constellation, ConstellationGeometry, General, Normalized, SquareQam};
use core::marker::PhantomData;
use num_complex::Complex;
use num_traits::Float;
pub trait DemodPoint<T> {
    fn demod_hard(&self, point: Complex<T>) -> usize;
}

pub trait SoftDemodPoint<T, const K: usize, O: BitOrder> {
    fn demod_soft(&self, point: Complex<T>, noise_var: T) -> [T; K];
}

// Implement SoftDemodPoint across all geometries for f32 and f64
macro_rules! impl_soft_demod_point {
    ($ty:ident) => {
        impl<const M: usize, const K: usize, O: BitOrder> SoftDemodPoint<$ty, K, O>
            for Constellation<$ty, M, Normalized, General>
        {
            #[inline(always)]
            fn demod_soft(&self, point: Complex<$ty>, inv_noise_var: $ty) -> [$ty; K] {
                self.demodulate_soft_point_scaled::<K, O>(point, inv_noise_var)
            }
        }

        impl<const M: usize, const K: usize, O: BitOrder> SoftDemodPoint<$ty, K, O>
            for Constellation<$ty, M, Normalized, StandardPsk>
        {
            #[inline(always)]
            fn demod_soft(&self, point: Complex<$ty>, inv_noise_var: $ty) -> [$ty; K] {
                self.demodulate_soft_point_scaled::<K, O>(point, inv_noise_var)
            }
        }

        impl<const M: usize, const K: usize, O: BitOrder> SoftDemodPoint<$ty, K, O>
            for Constellation<$ty, M, Normalized, RotatedPsk<$ty>>
        {
            #[inline(always)]
            fn demod_soft(&self, point: Complex<$ty>, inv_noise_var: $ty) -> [$ty; K] {
                self.demodulate_soft_point_scaled::<K, O>(point, inv_noise_var)
            }
        }

        impl<const M: usize, const K: usize, O: BitOrder> SoftDemodPoint<$ty, K, O>
            for Constellation<$ty, M, Normalized, SquareQam<$ty>>
        {
            #[inline(always)]
            fn demod_soft(&self, point: Complex<$ty>, inv_noise_var: $ty) -> [$ty; K] {
                self.demodulate_soft_point_scaled::<K, O>(point, inv_noise_var)
            }
        }
    };
}

impl_soft_demod_point!(f32);
impl_soft_demod_point!(f64);

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

impl<const M: usize> DemodPoint<f32>
    for Constellation<f32, M, Normalized, crate::constellation::StandardPsk>
{
    #[inline(always)]
    fn demod_hard(&self, point: Complex<f32>) -> usize {
        self.demodulate_hard_point(point)
    }
}

impl<const M: usize> DemodPoint<f32>
    for Constellation<f32, M, Normalized, crate::constellation::RotatedPsk<f32>>
{
    #[inline(always)]
    fn demod_hard(&self, point: Complex<f32>) -> usize {
        self.demodulate_hard_point(point)
    }
}

impl<const M: usize> DemodPoint<f64>
    for Constellation<f64, M, Normalized, crate::constellation::StandardPsk>
{
    #[inline(always)]
    fn demod_hard(&self, point: Complex<f64>) -> usize {
        self.demodulate_hard_point(point)
    }
}

impl<const M: usize> DemodPoint<f64>
    for Constellation<f64, M, Normalized, crate::constellation::RotatedPsk<f64>>
{
    #[inline(always)]
    fn demod_hard(&self, point: Complex<f64>) -> usize {
        self.demodulate_hard_point(point)
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
    O: BitOrder,
    T: Copy + Default,
{
    type SoftSymbolOutput: Iterator<Item = [T; K]>;
    type SoftBitOutput: Iterator<Item = T>;

    fn demodulate_soft(self, iter: I, noise_var: T) -> Self::SoftSymbolOutput;
    fn demodulate_soft_with(self, iter: I, noise_var: T) -> Self::SoftSymbolOutput;
    fn demodulate_soft_bits(self, iter: I, noise_var: T) -> Self::SoftBitOutput;
    fn demodulate_soft_bits_with(self, iter: I, noise_var: T) -> Self::SoftBitOutput;
}

macro_rules! impl_direct_demod_sizes {
    ($( ($m:expr, $k:expr, $n:expr) ),* $(,)?) => {
        $(
            // --- Inherent Hard Demodulation Methods ---
            impl<T: Copy, G: ConstellationGeometry> Constellation<T, $m, Normalized, G>
            where
                Self: DemodPoint<T>,
                MsbFirst: DirectBitCodec<$k, $n>,
            {
                #[inline]
                pub fn demodulate_hard<'c, I: Iterator<Item = Complex<T>>>(
                    &'c self,
                    iter: I,
                ) -> DirectByteDemodIter<'c, I, T, $m, $k, $n, G, MsbFirst> {
                    DirectByteDemodIter::new(iter, self)
                }

                #[inline]
                pub fn demodulate_hard_with<'c, I: Iterator<Item = Complex<T>>, O: BitOrder + DirectBitCodec<$k, $n>>(
                    &'c self,
                    iter: I,
                ) -> DirectByteDemodIter<'c, I, T, $m, $k, $n, G, O> {
                    DirectByteDemodIter::with_order(iter, self)
                }

                #[inline]
                pub fn demodulate_hard_symbols<'c, I: Iterator<Item = Complex<T>>>(
                    &'c self,
                    iter: I,
                ) -> HardSymbolDemodIter<'c, I, T, $m, G> {
                    HardSymbolDemodIter::new(iter, self)
                }
            }

            // --- Inherent Soft Demodulation Methods ---
            impl<T: Copy + Default + Float, G: ConstellationGeometry> Constellation<T, $m, Normalized, G> {
                #[inline]
                pub fn demodulate_soft<'c, I: Iterator<Item = Complex<T>>>(
                    &'c self,
                    iter: I,
                    noise_var: T,
                ) -> SoftSymbolDemodIter<'c, I, T, $m, $k, G, MsbFirst>
                where
                    Self: SoftDemodPoint<T, $k, MsbFirst>,
                {
                    SoftSymbolDemodIter::new(iter, self, noise_var)
                }

                #[inline]
                pub fn demodulate_soft_with<'c, I: Iterator<Item = Complex<T>>, O: BitOrder >(
                    &'c self,
                    iter: I,
                    noise_var: T,
                ) -> SoftSymbolDemodIter<'c, I, T, $m, $k, G, O>
                where
                    Self: SoftDemodPoint<T, $k, O>,
                {
                    SoftSymbolDemodIter::with_order(iter, self, noise_var)
                }

                #[inline]
                pub fn demodulate_soft_bits<'c, I: Iterator<Item = Complex<T>>>(
                    &'c self,
                    iter: I,
                    noise_var: T,
                ) -> SoftBitDemodIter<'c, I, T, $m, $k, G, MsbFirst>
                where
                    Self: SoftDemodPoint<T, $k, MsbFirst>,
                {
                    SoftBitDemodIter::new(iter, self, noise_var)
                }

                #[inline]
                pub fn demodulate_soft_bits_with<'c, I: Iterator<Item = Complex<T>>, O: BitOrder >(
                    &'c self,
                    iter: I,
                    noise_var: T,
                ) -> SoftBitDemodIter<'c, I, T, $m, $k, G, O>
                where
                    Self: SoftDemodPoint<T, $k, O>,
                {
                    SoftBitDemodIter::with_order(iter, self, noise_var)
                }
            }

            // --- Demodulatable Trait Implementation (Direct Output) ---
            impl<'c, I, T, G, O> Demodulatable<I, T, O> for &'c Constellation<T, $m, Normalized, G>
            where
                I: Iterator<Item = Complex<T>>,
                G: ConstellationGeometry,
                O: BitOrder + DirectBitCodec<$k, $n>,
                T: Copy,
                Constellation<T, $m, Normalized, G>: DemodPoint<T>,
            {
                type HardByteOutput = DirectByteDemodIter<'c, I, T, $m, $k, $n, G, O>;
                type HardSymbolOutput = HardSymbolDemodIter<'c, I, T, $m, G>;

                #[inline]
                fn demodulate_hard(self, iter: I) -> Self::HardByteOutput {
                    DirectByteDemodIter::with_order(iter, self)
                }

                #[inline]
                fn demodulate_hard_with(self, iter: I) -> Self::HardByteOutput {
                    DirectByteDemodIter::with_order(iter, self)
                }

                #[inline]
                fn demodulate_hard_symbols(self, iter: I) -> Self::HardSymbolOutput {
                    HardSymbolDemodIter::new(iter, self)
                }
            }

            // --- SoftDemodulatable Trait Implementation ---
            impl<'c, I, T, G, O> SoftDemodulatable<I, T, $k, O> for &'c Constellation<T, $m, Normalized, G>
            where
                I: Iterator<Item = Complex<T>>,
                G: ConstellationGeometry,
                O: BitOrder + 'static,
                T: Copy + Default + Float,
                Constellation<T, $m, Normalized, G>: SoftDemodPoint<T, $k, O>,
            {
                type SoftSymbolOutput = SoftSymbolDemodIter<'c, I, T, $m, $k, G, O>;
                type SoftBitOutput = SoftBitDemodIter<'c, I, T, $m, $k, G, O>;

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

macro_rules! impl_fallback_demod_sizes {
    ($( ($m:expr, $k:expr) ),* $(,)?) => {
        $(
            // --- Inherent Hard Demodulation Methods ---
            impl<T: Copy, G: ConstellationGeometry> Constellation<T, $m, Normalized, G>
            where
                Self: DemodPoint<T>,
            {
                #[inline]
                pub fn demodulate_hard<'c, I: Iterator<Item = Complex<T>>>(
                    &'c self,
                    iter: I,
                ) -> HardByteDemodIter<'c, I, T, $m, $k, G, MsbFirst> {
                    HardByteDemodIter::new(iter, self)
                }

                #[inline]
                pub fn demodulate_hard_with<'c, I: Iterator<Item = Complex<T>>, O: BitOrder>(
                    &'c self,
                    iter: I,
                ) -> HardByteDemodIter<'c, I, T, $m, $k, G, O> {
                    HardByteDemodIter::with_order(iter, self)
                }

                #[inline]
                pub fn demodulate_hard_symbols<'c, I: Iterator<Item = Complex<T>>>(
                    &'c self,
                    iter: I,
                ) -> HardSymbolDemodIter<'c, I, T, $m, G> {
                    HardSymbolDemodIter::new(iter, self)
                }
            }

            // --- Inherent Soft Demodulation Methods ---
            impl<T: Copy + Default + Float, G: ConstellationGeometry> Constellation<T, $m, Normalized, G> {
                #[inline]
                pub fn demodulate_soft<'c, I: Iterator<Item = Complex<T>>>(
                    &'c self,
                    iter: I,
                    noise_var: T,
                ) -> SoftSymbolDemodIter<'c, I, T, $m, $k, G, MsbFirst>
                where
                    Self: SoftDemodPoint<T, $k, MsbFirst>,
                {
                    SoftSymbolDemodIter::new(iter, self, noise_var)
                }

                #[inline]
                pub fn demodulate_soft_with<'c, I: Iterator<Item = Complex<T>>, O: BitOrder + 'static>(
                    &'c self,
                    iter: I,
                    noise_var: T,
                ) -> SoftSymbolDemodIter<'c, I, T, $m, $k, G, O>
                where
                    Self: SoftDemodPoint<T, $k, O>,
                {
                    SoftSymbolDemodIter::with_order(iter, self, noise_var)
                }

                #[inline]
                pub fn demodulate_soft_bits<'c, I: Iterator<Item = Complex<T>>>(
                    &'c self,
                    iter: I,
                    noise_var: T,
                ) -> SoftBitDemodIter<'c, I, T, $m, $k, G, MsbFirst>
                where
                    Self: SoftDemodPoint<T, $k, MsbFirst>,
                {
                    SoftBitDemodIter::new(iter, self, noise_var)
                }

                #[inline]
                pub fn demodulate_soft_bits_with<'c, I: Iterator<Item = Complex<T>>, O: BitOrder + 'static>(
                    &'c self,
                    iter: I,
                    noise_var: T,
                ) -> SoftBitDemodIter<'c, I, T, $m, $k, G, O>
                where
                    Self: SoftDemodPoint<T, $k, O>,
                {
                    SoftBitDemodIter::with_order(iter, self, noise_var)
                }
            }

            // --- Demodulatable Trait Implementation (Fallback Output) ---
            impl<'c, I, T, G, O> Demodulatable<I, T, O> for &'c Constellation<T, $m, Normalized, G>
            where
                I: Iterator<Item = Complex<T>>,
                G: ConstellationGeometry,
                O: BitOrder,
                T: Copy,
                Constellation<T, $m, Normalized, G>: DemodPoint<T>,
            {
                type HardByteOutput = HardByteDemodIter<'c, I, T, $m, $k, G, O>;
                type HardSymbolOutput = HardSymbolDemodIter<'c, I, T, $m, G>;

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

            // --- SoftDemodulatable Trait Implementation ---
            impl<'c, I, T, G, O> SoftDemodulatable<I, T, $k, O> for &'c Constellation<T, $m, Normalized, G>
            where
                I: Iterator<Item = Complex<T>>,
                G: ConstellationGeometry,
                O: BitOrder + 'static,
                T: Copy + Default + Float,
                Constellation<T, $m, Normalized, G>: SoftDemodPoint<T, $k, O>,
            T: Float
            {
                type SoftSymbolOutput = SoftSymbolDemodIter<'c, I, T, $m, $k, G, O>;
                type SoftBitOutput = SoftBitDemodIter<'c, I, T, $m, $k, G, O>;

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

impl_direct_demod_sizes!(
    (2, 1, 8),   // BPSK
    (4, 2, 4),   // QPSK
    (16, 4, 2),  // 16-QAM
    (256, 8, 1), // 256-QAM
);

impl_fallback_demod_sizes!(
    (8, 3),     // 8-PSK
    (32, 5),    // 32-QAM
    (64, 6),    // 64-QAM
    (128, 7),   // 128-QAM
    (512, 9),   // 512-QAM
    (1024, 10), // 1024-QAM
    (2048, 11), // 2048-QAM
    (4096, 12), // 4096-QAM
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

pub struct HardSymbolDemodIter<'c, I, T, const M: usize, G = General>
where
    G: ConstellationGeometry,
{
    iter: I,
    constellation: &'c Constellation<T, M, Normalized, G>,
}

impl<'c, I, T, const M: usize, G: ConstellationGeometry> HardSymbolDemodIter<'c, I, T, M, G> {
    #[inline]
    pub fn new(iter: I, constellation: &'c Constellation<T, M, Normalized, G>) -> Self {
        Self {
            iter,
            constellation,
        }
    }
}

impl<'c, I, T, const M: usize, G> Iterator for HardSymbolDemodIter<'c, I, T, M, G>
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

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'c, I, T, const M: usize, G> ExactSizeIterator for HardSymbolDemodIter<'c, I, T, M, G>
where
    I: ExactSizeIterator<Item = Complex<T>>,
    T: Copy,
    G: ConstellationGeometry,
    Constellation<T, M, Normalized, G>: DemodPoint<T>,
{
}

pub struct HardByteDemodIter<'c, I, T, const M: usize, const K: usize, G = General, O = MsbFirst>
where
    G: ConstellationGeometry,
{
    packer: BitPacker<HardSymbolDemodIter<'c, I, T, M, G>, K, O>,
}

impl<'c, I, T, const M: usize, const K: usize, G: ConstellationGeometry>
    HardByteDemodIter<'c, I, T, M, K, G, MsbFirst>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy,
    Constellation<T, M, Normalized, G>: DemodPoint<T>,
{
    #[inline]
    pub fn new(iter: I, constellation: &'c Constellation<T, M, Normalized, G>) -> Self {
        Self::with_order(iter, constellation)
    }
}

impl<'c, I, T, const M: usize, const K: usize, G: ConstellationGeometry, O: BitOrder>
    HardByteDemodIter<'c, I, T, M, K, G, O>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy,
    Constellation<T, M, Normalized, G>: DemodPoint<T>,
{
    #[inline]
    pub fn with_order(iter: I, constellation: &'c Constellation<T, M, Normalized, G>) -> Self {
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

impl<'c, I, T, const M: usize, const K: usize, G, O> Iterator
    for HardByteDemodIter<'c, I, T, M, K, G, O>
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

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.packer.size_hint()
    }
}

pub struct SoftSymbolDemodIter<'c, I, T, const M: usize, const K: usize, G = General, O = MsbFirst>
where
    G: ConstellationGeometry,
{
    iter: I,
    constellation: &'c Constellation<T, M, Normalized, G>,
    inv_noise_var: T,
    _marker: PhantomData<O>,
}

impl<'c, I, T, const M: usize, const K: usize, G: ConstellationGeometry>
    SoftSymbolDemodIter<'c, I, T, M, K, G, MsbFirst>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy + Float,
{
    #[inline]
    pub fn new(
        iter: I,
        constellation: &'c Constellation<T, M, Normalized, G>,
        noise_var: T,
    ) -> Self {
        Self::with_order(iter, constellation, noise_var)
    }
}

impl<'c, I, T, const M: usize, const K: usize, G: ConstellationGeometry, O: BitOrder>
    SoftSymbolDemodIter<'c, I, T, M, K, G, O>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy + Float,
{
    #[inline]
    pub fn with_order(
        iter: I,
        constellation: &'c Constellation<T, M, Normalized, G>,
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
            inv_noise_var: T::one() / noise_var,
            _marker: PhantomData,
        }
    }
}

impl<'c, I, T, const M: usize, const K: usize, G, O> Iterator
    for SoftSymbolDemodIter<'c, I, T, M, K, G, O>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy + Float,
    G: ConstellationGeometry,
    O: BitOrder + 'static,
    Constellation<T, M, Normalized, G>: SoftDemodPoint<T, K, O>,
{
    type Item = [T; K];

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let point = self.iter.next()?;
        Some(self.constellation.demod_soft(point, self.inv_noise_var))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'c, I, T, const M: usize, const K: usize, G, O> ExactSizeIterator
    for SoftSymbolDemodIter<'c, I, T, M, K, G, O>
where
    I: ExactSizeIterator<Item = Complex<T>>,
    T: Copy + Float,
    G: ConstellationGeometry,
    O: BitOrder + 'static,
    Constellation<T, M, Normalized, G>: SoftDemodPoint<T, K, O>,
{
}

pub struct SoftBitDemodIter<'c, I, T, const M: usize, const K: usize, G = General, O = MsbFirst>
where
    G: ConstellationGeometry,
{
    symbol_iter: SoftSymbolDemodIter<'c, I, T, M, K, G, O>,
    buffer: [T; K],
    index: usize,
}

impl<'c, I, T, const M: usize, const K: usize, G: ConstellationGeometry>
    SoftBitDemodIter<'c, I, T, M, K, G, MsbFirst>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy + Default + Float,
{
    #[inline]
    pub fn new(
        iter: I,
        constellation: &'c Constellation<T, M, Normalized, G>,
        noise_var: T,
    ) -> Self {
        Self::with_order(iter, constellation, noise_var)
    }
}

impl<'c, I, T, const M: usize, const K: usize, G: ConstellationGeometry, O: BitOrder>
    SoftBitDemodIter<'c, I, T, M, K, G, O>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy + Default + Float,
{
    #[inline]
    pub fn with_order(
        iter: I,
        constellation: &'c Constellation<T, M, Normalized, G>,
        noise_var: T,
    ) -> Self {
        Self {
            symbol_iter: SoftSymbolDemodIter::with_order(iter, constellation, noise_var),
            buffer: [T::default(); K],
            index: K,
        }
    }
}

impl<'c, I, T, const M: usize, const K: usize, G, O> Iterator
    for SoftBitDemodIter<'c, I, T, M, K, G, O>
where
    I: Iterator<Item = Complex<T>>,
    T: Copy + Default + Float,
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

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.symbol_iter.size_hint();
        let buffered = K.saturating_sub(self.index);

        let lower_total = lower.saturating_mul(K).saturating_add(buffered);
        let upper_total = upper.and_then(|u| u.checked_mul(K)?.checked_add(buffered));

        (lower_total, upper_total)
    }
}

impl<'c, I, T, const M: usize, const K: usize, G, O> ExactSizeIterator
    for SoftBitDemodIter<'c, I, T, M, K, G, O>
where
    I: ExactSizeIterator<Item = Complex<T>>,
    T: Copy + Default + Float,
    G: ConstellationGeometry,
    O: BitOrder + 'static,
    Constellation<T, M, Normalized, G>: SoftDemodPoint<T, K, O>,
{
}

pub struct DirectByteDemodIter<'c, I, T, const M: usize, const K: usize, const N: usize, G, O>
where
    I: Iterator<Item = Complex<T>>,
    G: ConstellationGeometry,
    T: Copy,
    O: BitOrder + DirectBitCodec<K, N>,
    Constellation<T, M, Normalized, G>: DemodPoint<T>,
{
    iter: I,
    constellation: &'c Constellation<T, M, Normalized, G>,
    _marker: PhantomData<O>,
}

impl<'c, I, T, const M: usize, const K: usize, const N: usize, G>
    DirectByteDemodIter<'c, I, T, M, K, N, G, MsbFirst>
where
    I: Iterator<Item = Complex<T>>,
    G: ConstellationGeometry,
    T: Copy,
    MsbFirst: DirectBitCodec<K, N>,
    Constellation<T, M, Normalized, G>: DemodPoint<T>,
{
    #[inline]
    pub fn new(iter: I, constellation: &'c Constellation<T, M, Normalized, G>) -> Self {
        Self::with_order(iter, constellation)
    }
}

impl<'c, I, T, const M: usize, const K: usize, const N: usize, G, O>
    DirectByteDemodIter<'c, I, T, M, K, N, G, O>
where
    I: Iterator<Item = Complex<T>>,
    G: ConstellationGeometry,
    T: Copy,
    O: BitOrder + DirectBitCodec<K, N>,
    Constellation<T, M, Normalized, G>: DemodPoint<T>,
{
    #[inline]
    pub fn with_order(iter: I, constellation: &'c Constellation<T, M, Normalized, G>) -> Self {
        assert_eq!(1 << K, M, "Constellation size M ({M}) must equal 2^K");
        assert_eq!(K * N, 8, "K * N must equal 8 bits per byte");
        Self {
            iter,
            constellation,
            _marker: PhantomData,
        }
    }
}

impl<'c, I, T, const M: usize, const K: usize, const N: usize, G, O> Iterator
    for DirectByteDemodIter<'c, I, T, M, K, N, G, O>
where
    I: Iterator<Item = Complex<T>>,
    G: ConstellationGeometry,
    T: Copy,
    O: BitOrder + DirectBitCodec<K, N>,
    Constellation<T, M, Normalized, G>: DemodPoint<T>,
{
    type Item = u8;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let first = self.iter.next()?;
        let mut symbols = [0usize; N];
        symbols[0] = self.constellation.demod_hard(first);

        let mut i = 1;
        while i < N {
            let point = self.iter.next()?;
            symbols[i] = self.constellation.demod_hard(point);
            i += 1;
        }

        Some(O::pack_symbols(symbols))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();
        (lower / N, upper.map(|u| u / N))
    }
}

impl<'c, I, T, const M: usize, const K: usize, const N: usize, G, O> ExactSizeIterator
    for DirectByteDemodIter<'c, I, T, M, K, N, G, O>
where
    I: ExactSizeIterator<Item = Complex<T>>,
    G: ConstellationGeometry,
    T: Copy,
    O: BitOrder + DirectBitCodec<K, N>,
    Constellation<T, M, Normalized, G>: DemodPoint<T>,
{
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

        let symbols: Vec<Complex<f32>> = original.clone().into_iter().modulate(&bpsk).collect();
        let recovered: Vec<u8> = symbols.into_iter().demodulate_hard(&bpsk).collect();

        assert_eq!(recovered, original);
    }

    #[test]
    fn test_hard_demod_qpsk_roundtrip() {
        let original = vec![0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];
        let qpsk = Qpsk::<f32>::QPSK;

        let symbols: Vec<Complex<f32>> = original.clone().into_iter().modulate(&qpsk).collect();
        let recovered: Vec<u8> = symbols.into_iter().demodulate_hard(&qpsk).collect();

        assert_eq!(recovered, original);
    }

    #[test]
    fn test_hard_demod_8psk_spanning_roundtrip() {
        let original = vec![0b101_100_11, 0b0_101_110_0, 0b11_000_111];
        let psk8 = Psk8::<f64>::m_psk();

        let symbols: Vec<Complex<f64>> = original.clone().into_iter().modulate(&psk8).collect();
        let recovered: Vec<u8> = symbols.into_iter().demodulate_hard(&psk8).collect();

        assert_eq!(recovered, original);
    }

    #[test]
    fn test_hard_demod_qam16_roundtrip() {
        let original = vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE];
        let qam16 = Qam16::<f32>::QAM16;

        let symbols: Vec<Complex<f32>> = original.clone().into_iter().modulate(&qam16).collect();
        let recovered: Vec<u8> = symbols.into_iter().demodulate_hard(&qam16).collect();

        assert_eq!(recovered, original);
    }

    #[test]
    fn test_hard_demod_lsb_roundtrip() {
        let original = vec![0xA5, 0x5A, 0x3C, 0xC3];
        let qpsk = Qpsk::<f32>::QPSK;

        let symbols: Vec<Complex<f32>> = original
            .clone()
            .into_iter()
            .modulate_with::<_, f32, LsbFirst, PadZeros>(&qpsk)
            .collect();

        let recovered: Vec<u8> = symbols
            .into_iter()
            .demodulate_hard_with::<_, LsbFirst>(&qpsk)
            .collect();

        assert_eq!(recovered, original);
    }

    #[test]
    fn test_noisy_channel_hard_demodulation() {
        let original = vec![0b11_01_00_10];
        let qpsk = Qpsk::<f32>::QPSK;

        let noisy_symbols: Vec<Complex<f32>> = original
            .clone()
            .into_iter()
            .modulate(&qpsk)
            .map(|s| s + Complex::new(0.05, -0.05))
            .collect();

        let recovered: Vec<u8> = noisy_symbols.into_iter().demodulate_hard(&qpsk).collect();
        assert_eq!(recovered, original);
    }

    #[test]
    fn test_soft_demod_llr_signs() {
        let bpsk = Bpsk::<f32>::BPSK;
        let samples = vec![Complex::new(0.85, 0.0), Complex::new(-0.85, 0.0)];

        let llrs: Vec<[f32; 1]> = samples.into_iter().demodulate_soft(&bpsk, 0.5f32).collect();

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
        let original = vec![0b11_01_00_10];
        let qpsk = Qpsk::<f32>::QPSK;

        let symbols: Vec<Complex<f32>> = original.into_iter().modulate(&qpsk).collect();
        let bit_llrs: Vec<f32> = symbols
            .into_iter()
            .demodulate_soft_bits(&qpsk, 0.5f32)
            .collect();

        assert_eq!(bit_llrs.len(), 8);

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

        let indices: Vec<usize> = points.into_iter().demodulate_hard_symbols(&qpsk).collect();

        assert_eq!(indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_high_order_qam_roundtrips() {
        let payload: Vec<u8> = (0..120).map(|x| (x * 37) as u8).collect();

        // 64-QAM (K = 6)
        let qam64 = Qam64::<f32>::QAM64;
        let syms64: Vec<_> = payload.clone().into_iter().modulate(&qam64).collect();
        let rec64: Vec<u8> = syms64.into_iter().demodulate_hard(&qam64).collect();
        assert_eq!(rec64, payload);

        // 256-QAM (K = 8)
        let qam256 = Qam256::<f32>::QAM256;
        let syms256: Vec<_> = payload.clone().into_iter().modulate(&qam256).collect();
        let rec256: Vec<u8> = syms256.into_iter().demodulate_hard(&qam256).collect();
        assert_eq!(rec256, payload);

        // 4096-QAM (K = 12)
        let qam4096 = Qam4096::<f64>::QAM4096;
        let syms4096: Vec<_> = payload.clone().into_iter().modulate(&qam4096).collect();
        let rec4096: Vec<u8> = syms4096.into_iter().demodulate_hard(&qam4096).collect();
        assert_eq!(rec4096, payload);
    }

    #[test]
    fn test_custom_user_constellation_general_pipeline() {
        let raw_points = [
            Complex::new(2.0f32, 1.0f32),
            Complex::new(-1.0f32, 3.0f32),
            Complex::new(-2.0f32, -2.0f32),
            Complex::new(3.0f32, -1.0f32),
        ];
        let custom = Constellation::<f32, 4>::from_points(raw_points).normalize();

        let data = vec![0x12, 0x34, 0xAB, 0xCD];
        let symbols: Vec<Complex<f32>> = data.clone().into_iter().modulate(&custom).collect();
        let recovered: Vec<u8> = symbols.into_iter().demodulate_hard(&custom).collect();

        assert_eq!(recovered, data);
    }

    #[test]
    fn fallback_demodulation_reports_size_hint() {
        let psk8 = Psk8::<f32>::m_psk();
        let symbols = vec![psk8[0]; 24]; // 24 3-bit symbols -> 9 bytes (72 bits)
        let it = symbols.into_iter().demodulate_hard(&psk8);
        assert_eq!(it.size_hint(), (9, Some(9)));
    }
}
