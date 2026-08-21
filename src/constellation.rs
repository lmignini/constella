use crate::utils::{binary_to_gray, gives_square_constellation, is_valid_constellation_size};
use core::marker::PhantomData;

use num_complex::Complex;

use crate::bits::BitOrder;
use crate::bits::MsbFirst;
use trig_const::{cos, sin, sqrt};

pub type Bpsk<T = f32, S = Normalized> = Constellation<T, 2, S>;
pub type Qpsk<T = f32, S = Normalized> = Constellation<T, 4, S>;
pub type Psk8<T = f32, S = Normalized> = Constellation<T, 8, S>;
pub type Qam16<T = f32, S = Normalized> = Constellation<T, 16, S>;
pub type Qam64<T = f32, S = Normalized> = Constellation<T, 64, S>;
pub type Qam256<T = f32, S = Normalized> = Constellation<T, 256, S>;
pub type Qam1024<T = f32, S = Normalized> = Constellation<T, 1024, S>;
pub type Qam4096<T = f32, S = Normalized> = Constellation<T, 4096, S>;

/// Generalize constellation over f32 and f64
macro_rules! impl_constellation_float {
    ($ty: ident) => {
        impl<const M: usize> Constellation<$ty, M, Unnormalized> {
            /// Return the same constellation shape, but with the points normalized to have unitary
            /// average energy.
            pub const fn normalize(self) -> Constellation<$ty, M, Normalized> {
                Constellation::<$ty, M, Normalized>::from_points_normalized(self.points)
            }
        }
        impl<const M: usize> Constellation<$ty, M, Normalized> {
            /// Construct normalized constellation from points
            pub const fn from_points_normalized(points: [Complex<$ty>; M]) -> Self {
                assert!(is_valid_constellation_size(M));

                let energy = Self::total_energy(points);

                let avg_energy = energy / M as $ty;

                assert!(
                    avg_energy > 0.0,
                    "Constellation energy must be greater than zero"
                );

                let scale_factor = (1.0 as $ty) / sqrt(avg_energy as f64) as $ty;
                let mut i = 0;
                let mut new_points = [Complex::new(0.0 as $ty, 0.0 as $ty); M];
                while i < M {
                    let point = points[i];
                    let (re, im) = (point.re, point.im);
                    new_points[i] = Complex::new(re * scale_factor, im * scale_factor);
                    i += 1;
                }
                Self {
                    points: new_points,
                    scale_factor,
                    energy: 1.0,
                    _state: core::marker::PhantomData::<Normalized>,
                }
            }
        }

        impl<const M: usize, S: ConstellationState> Constellation<$ty, M, S> {
            /// Construct constellation from points
            pub const fn from_points(
                points: [Complex<$ty>; M],
            ) -> Constellation<$ty, M, Unnormalized> {
                assert!(is_valid_constellation_size(M));

                let energy = Self::total_energy(points);

                let avg_energy = energy / M as $ty;

                assert!(
                    avg_energy > 0.0,
                    "Constellation energy must be greater than zero"
                );

                let scale_factor = (1.0 as $ty) / sqrt(avg_energy as f64) as $ty;

                Constellation {
                    points,
                    energy: avg_energy,
                    scale_factor,
                    _state: PhantomData::<Unnormalized>,
                }
            }
        }

        /// Utils
        impl<const M: usize, S: ConstellationState> Constellation<$ty, M, S> {
            pub const BITS_PER_SYMBOL: usize = M.ilog2() as usize;
            const fn total_energy(points: [Complex<$ty>; M]) -> $ty {
                let mut i = 0;

                let mut acc = 0.0 as $ty;
                while i < M {
                    let point = points[i];
                    let (re, im) = (point.re, point.im);
                    acc = acc + (re * re + im * im);

                    i += 1;
                }
                acc
            }
            /// Returns a reference to the array of constellation points
            pub const fn points(&self) -> &[Complex<$ty>; M] {
                &self.points
            }

            /// Returns the scaling factor to normalize
            pub const fn scale_factor(&self) -> $ty {
                self.scale_factor
            }

            /// Returns the average energy per symbol ($E_s$).
            pub const fn energy(&self) -> $ty {
                self.energy
            }

            const fn from_polar(r: $ty, theta: $ty) -> Complex<$ty> {
                Complex::new(
                    r * (cos(theta as f64) as $ty),
                    r * (sin(theta as f64) as $ty),
                )
            }
        }

        /// Demodulation
        impl<const M: usize> Constellation<$ty, M, Normalized> {
            /// Performs hard-decision demodulation on a single received complex sample.
            ///
            /// Determines the closest constellation point to `point` by finding the symbol index $i$
            /// that minimizes the squared Euclidean distance:
            ///
            /// $$\hat{i} = \arg\min_{i \in [0, M-1]} |r - s_i|^2$$
            ///
            /// Uses squared Euclidean distance to avoid square root computations and [`total_cmp`](core::primitive::f32::total_cmp)
            /// to provide deterministic comparisons across floating-point values.
            ///
            /// # Complexity
            ///
            /// $\mathcal{O}(M)$ nearest-neighbor search, where $M$ is the constellation size.
            ///
            /// # Arguments
            ///
            /// * `point` - The received complex baseband sample.
            ///
            /// # Returns
            ///
            /// The `usize` index of the closest constellation symbol ($0 \le \text{index} < M$).
            ///
            /// # Panics
            ///
            /// Panics if the constellation contains zero points (cannot occur for valid constellations where $M \ge 2$).
            ///
            /// # Examples
            ///
            /// ```rust
            /// use num_complex::Complex;
            /// use constella::constellation::Bpsk;
            ///
            /// let bpsk = Bpsk::<f32>::BPSK;
            ///
            /// // Point near (+1.0, 0.0) slices to symbol 0
            /// assert_eq!(bpsk.demodulate_hard_point(Complex::new(0.85, 0.05)), 0);
            ///
            /// // Point near (-1.0, 0.0) slices to symbol 1
            /// assert_eq!(bpsk.demodulate_hard_point(Complex::new(-1.20, -0.10)), 1);
            /// ```
            #[inline]
            pub fn demodulate_hard_point(&self, point: Complex<$ty>) -> usize {
                self.points
                    .iter()
                    .map(|constellation_point| (constellation_point - point).norm_sqr())
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(index, _)| index)
                    .expect("Constellation cannot be empty")
            }

            /// Computes soft Log-Likelihood Ratios (LLRs) for all $K$ bits of a received complex point.
            ///
            /// Uses the Max-Log approximation.
            ///
            /// # Parameters
            /// * `point`: The received complex sample.
            /// * `noise_var`: Channel noise variance $\sigma^2$. Pass `0.5` for unscaled Euclidean metric differences.
            #[inline]
            pub fn demodulate_soft_point<const K: usize, O: BitOrder + 'static>(
                &self,
                point: Complex<$ty>,
                noise_var: $ty,
            ) -> [$ty; K] {
                assert_eq!(1 << K, M, "K must satisfy 2^K == M");
                // 1. Calculate squared distances to all M constellation points once
                let mut dists = [0.0 as $ty; M];
                let mut i = 0;
                while i < M {
                    let diff = self.points[i] - point;
                    dists[i] = diff.re * diff.re + diff.im * diff.im;
                    i += 1;
                }

                let mut llrs = [0.0 as $ty; K];
                let scale = (1.0 as $ty) / (2.0 * noise_var);

                // 2. Compute LLR for each bit position
                let mut j = 0;
                while j < K {
                    let mut min_d0 = <$ty>::INFINITY;
                    let mut min_d1 = <$ty>::INFINITY;

                    let mut idx = 0;
                    while idx < M {
                        let d = dists[idx];

                        // Determine whether the j-th bit of symbol `idx` is 0 or 1
                        let bit = if core::any::TypeId::of::<O>()
                            == core::any::TypeId::of::<MsbFirst>()
                        {
                            (idx >> (K - 1 - j)) & 1
                        } else {
                            (idx >> j) & 1
                        };

                        if bit == 0 {
                            if d < min_d0 {
                                min_d0 = d;
                            }
                        } else {
                            if d < min_d1 {
                                min_d1 = d;
                            }
                        }
                        idx += 1;
                    }

                    llrs[j] = (min_d1 - min_d0) * scale;
                    j += 1;
                }

                llrs
            }
        }
        /// M-PSK
        impl<const M: usize> Constellation<$ty, M, Normalized> {
            /// Generates an $M$-PSK constellation with Gray coding.
            /// The generated constellation is already normalized to have unitary average energy.
            /// If you need a BPSK or QPSK, you might be interested in the provided constructors for those.
            pub const fn m_psk_with_phase(phase_offset: $ty) -> Self {
                let mut k = 0;
                let mut points = [Complex::new(0.0 as $ty, 0.0 as $ty); M];
                while k < M {
                    let theta_k = phase_offset
                        + (2.0 * core::f64::consts::PI * (k as f64) / (M as f64)) as $ty;
                    points[binary_to_gray(k)] = Self::from_polar(1.0 as $ty, theta_k);
                    k += 1;
                }
                Self::from_points_normalized(points)
            }

            /// Generates an $M$-PSK constellation with Gray coding.
            /// The generated constellation is already normalized to have unitary average energy.
            /// If you need a BPSK or QPSK, you might be interested in the provided constructors for those.
            /// Note that one of the points of this constellation sits on the x-axis at 1.0.
            pub const fn m_psk() -> Self {
                Self::m_psk_with_phase(0.0 as $ty)
            }
        }

        /// BPSK
        impl Constellation<$ty, 2, Normalized> {
            pub const BPSK: Self = Self::bpsk();
            /// Generates a BPSK constellation.
            pub const fn bpsk() -> Self {
                Self::m_psk()
            }
        }

        /// QPSK
        impl Constellation<$ty, 4, Normalized> {
            pub const QPSK: Self = Self::qpsk();
            /// Generates a QPSK constellation.
            pub const fn qpsk() -> Self {
                Self::m_psk_with_phase(core::$ty::consts::PI / 4.0)
            }
        }

        /// M-QAM using gray coding
        impl<const M: usize> Constellation<$ty, M, Normalized> {
            /// Generates a square $M$-QAM constellation with Gray coding.
            ///
            /// # Panics
            /// Panics if `M` is not a valid square constellation size (e.g., 4, 16, 64, 256).
            pub const fn m_qam() -> Self {
                assert!(gives_square_constellation(M));

                let k = Self::BITS_PER_SYMBOL;
                let k_axis = k / 2; // Is an integer since k is even
                let l = 1 << k_axis;

                let mut points = [Complex::new(0.0 as $ty, 0.0 as $ty); M];
                let mut i = 0;
                let mut q = 0;

                while i < l {
                    while q < l {
                        let i_val = 2 * (i as isize) - (l as isize) + 1;
                        let q_val = 2 * (q as isize) - (l as isize) + 1;

                        let g_i = binary_to_gray(i);
                        let g_q = binary_to_gray(q);

                        let symbol_index = (g_i << k_axis) | g_q;
                        points[symbol_index] = Complex::new(i_val as $ty, q_val as $ty);
                        q += 1;
                    }
                    q = 0;
                    i += 1;
                }
                Self::from_points_normalized(points)
            }
        }
    };
}

/// Helper macro to stamp out associated constants and Default impls for square QAM sizes
macro_rules! impl_qam_consts {
    ($ty:ident, $( ($m:expr, $const_name:ident) ),* $(,)?) => {
        $(
            impl Constellation<$ty, $m, Normalized> {
                /// Pre-evaluated constellation constant.
                pub const $const_name: Self = Self::m_qam();
            }

            impl Default for Constellation<$ty, $m, Normalized> {
                #[inline]
                fn default() -> Self {
                    Self::$const_name
                }
            }
        )*
    };
}
/// Compile time marker for normalized constellation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Normalized;

/// Compile time marker for unnormalized constellation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Unnormalized;

pub trait ConstellationState {}

impl ConstellationState for Normalized {}
impl ConstellationState for Unnormalized {}

/// A digital modulation constellation of size `M` using floating-point coordinates `T`.
///
/// Holds `M` constellation points in the complex plane along with their scaling
/// and energy metadata.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Constellation<T, const M: usize, S = Normalized>
where
    S: ConstellationState,
{
    points: [Complex<T>; M],
    // Normalization factor applied to the constellation
    scale_factor: T,
    // Average energy
    energy: T,
    _state: PhantomData<S>,
}

/// Allows indexing the constellation directly, without having to call .points() first
impl<T, const M: usize, S: ConstellationState> core::ops::Index<usize> for Constellation<T, M, S> {
    type Output = Complex<T>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.points[index]
    }
}

impl_constellation_float!(f32);

impl_constellation_float!(f64);

impl_qam_consts!(
    f32,
    (4, QAM4),
    (16, QAM16),
    (64, QAM64),
    (256, QAM256),
    (1024, QAM1024),
    (4096, QAM4096),
);

impl_qam_consts!(
    f64,
    (4, QAM4),
    (16, QAM16),
    (64, QAM64),
    (256, QAM256),
    (1024, QAM1024),
    (4096, QAM4096),
);

#[cfg(test)]
mod tests {
    use super::*;

    const EPS_F32: f32 = 1e-6;
    const EPS_F64: f64 = 1e-12;

    pub fn approx_eq_c32(a: Complex<f32>, b: Complex<f32>) -> bool {
        (a.re - b.re).abs() < EPS_F32 && (a.im - b.im).abs() < EPS_F32
    }

    fn approx_eq_c64(a: Complex<f64>, b: Complex<f64>) -> bool {
        (a.re - b.re).abs() < EPS_F64 && (a.im - b.im).abs() < EPS_F64
    }

    #[test]
    fn test_bpsk_properties() {
        const BPSK: Constellation<f32, 2> = Bpsk::<f32>::bpsk();

        assert_eq!(Constellation::<f32, 2>::BITS_PER_SYMBOL, 1);
        assert!((BPSK.energy() - 1.0).abs() < EPS_F32);

        let pts = BPSK.points();

        // Index 0: Gray(0) = 0, angle = 0 rad -> (+1, 0)
        assert!(approx_eq_c32(pts[0], Complex::new(1.0, 0.0)));

        // Index 1: Gray(1) = 1, angle = PI rad -> (-1, 0)
        assert!(approx_eq_c32(pts[1], Complex::new(-1.0, 0.0)));
    }

    #[test]
    fn test_qpsk_properties() {
        const QPSK: Constellation<f64, 4> = Qpsk::<f64>::qpsk();

        assert_eq!(Constellation::<f64, 4>::BITS_PER_SYMBOL, 2);
        assert!((QPSK.energy() - 1.0).abs() < EPS_F64);

        let pts = QPSK.points();

        for (i, p) in pts.iter().enumerate() {
            let magnitude_sq = p.re * p.re + p.im * p.im;
            assert!(
                (magnitude_sq - 1.0).abs() < EPS_F64,
                "Point {i} has unexpected magnitude: {magnitude_sq}"
            );
        }

        // Circular Gray-mapping sequence: 00 -> 01 -> 11 -> 10 -> 00
        let gray_order: [usize; 4] = [0b00, 0b01, 0b11, 0b10];
        for i in 0..4 {
            let curr = gray_order[i];
            let next = gray_order[(i + 1) % 4];
            let bit_diff = (curr ^ next).count_ones();
            assert_eq!(
                bit_diff, 1,
                "Adjacent sequence must differ by exactly 1 bit"
            );
        }

        let inv_sqrt2 = core::f64::consts::FRAC_1_SQRT_2;
        assert!(approx_eq_c64(pts[0b00], Complex::new(inv_sqrt2, inv_sqrt2)));
        assert!(approx_eq_c64(
            pts[0b01],
            Complex::new(-inv_sqrt2, inv_sqrt2)
        ));
        assert!(approx_eq_c64(
            pts[0b11],
            Complex::new(-inv_sqrt2, -inv_sqrt2)
        ));
        assert!(approx_eq_c64(
            pts[0b10],
            Complex::new(inv_sqrt2, -inv_sqrt2)
        ));
    }

    #[test]
    fn test_psk8_properties() {
        const PSK8: Constellation<f32, 8> = Psk8::<f32>::m_psk();

        assert_eq!(Constellation::<f32, 8>::BITS_PER_SYMBOL, 3);
        assert!((PSK8.energy() - 1.0).abs() < EPS_F32);

        let pts = PSK8.points();

        for (i, p) in pts.iter().enumerate() {
            let magnitude_sq = p.re * p.re + p.im * p.im;
            assert!(
                (magnitude_sq - 1.0).abs() < EPS_F32,
                "Point {i} has unexpected magnitude: {magnitude_sq}"
            );
        }

        // Verify circular 1-bit Gray code continuity
        for k in 0..8 {
            let g_curr = binary_to_gray(k);
            let g_next = binary_to_gray((k + 1) % 8);
            let bit_diff = (g_curr ^ g_next).count_ones();
            assert_eq!(
                bit_diff, 1,
                "Adjacent 8-PSK points must differ by exactly 1 bit"
            );
        }
    }

    #[test]
    fn test_qam16_properties() {
        const QAM16: Constellation<f32, 16> = Constellation::<f32, 16>::m_qam();

        assert_eq!(Constellation::<f32, 16>::BITS_PER_SYMBOL, 4);
        assert!((QAM16.energy() - 1.0).abs() < EPS_F32);

        // Theoretical unnormalized average energy for 16-QAM is 2*(16-1)/3 = 10
        let expected_scale = 1.0f32 / 10.0f32.sqrt();
        assert!((QAM16.scale_factor() - expected_scale).abs() < EPS_F32);

        let pts = QAM16.points();

        // 1. Verify uniqueness
        for i in 0..16 {
            for j in (i + 1)..16 {
                assert!(
                    !approx_eq_c32(pts[i], pts[j]),
                    "Points {i} and {j} overlap at {:?}",
                    pts[i]
                );
            }
        }

        // 2. Verify Cartesian grid alignment
        let inv_scale = 1.0 / QAM16.scale_factor();
        let valid_levels = [-3.0f32, -1.0, 1.0, 3.0];

        for (i, p) in pts.iter().enumerate() {
            let raw_i = (p.re * inv_scale).round();
            let raw_q = (p.im * inv_scale).round();

            assert!(
                valid_levels.contains(&raw_i),
                "Point {i} has invalid unscaled I-level: {raw_i}"
            );
            assert!(
                valid_levels.contains(&raw_q),
                "Point {i} has invalid unscaled Q-level: {raw_q}"
            );
        }

        // 3. Verify total unit average power
        let mut total_power = 0.0f32;
        for p in pts {
            total_power += p.re * p.re + p.im * p.im;
        }
        let avg_power = total_power / 16.0;
        assert!((avg_power - 1.0).abs() < EPS_F32);
    }

    #[test]
    fn test_unnormalized_to_normalized_typestate_transition() {
        let raw_points = [
            Complex::new(2.0f32, 2.0f32),
            Complex::new(-2.0f32, 2.0f32),
            Complex::new(-2.0f32, -2.0f32),
            Complex::new(2.0f32, -2.0f32),
        ];

        // E_avg = (8 + 8 + 8 + 8) / 4 = 8.0
        let unnorm: Constellation<f32, 4, Unnormalized> =
            Constellation::<f32, 4>::from_points(raw_points);
        assert!((unnorm.energy() - 8.0).abs() < EPS_F32);

        let expected_scale = 1.0f32 / 8.0f32.sqrt();
        assert!((unnorm.scale_factor() - expected_scale).abs() < EPS_F32);
        assert_eq!(unnorm[0], raw_points[0]);

        // Typestate transition via .normalize()
        let norm: Constellation<f32, 4, Normalized> = unnorm.normalize();
        assert!((norm.energy() - 1.0).abs() < EPS_F32);

        let inv_sqrt2 = core::f32::consts::FRAC_1_SQRT_2;
        assert!(approx_eq_c32(norm[0], Complex::new(inv_sqrt2, inv_sqrt2)));
        assert!(approx_eq_c32(norm[1], Complex::new(-inv_sqrt2, inv_sqrt2)));
        assert!(approx_eq_c32(norm[2], Complex::new(-inv_sqrt2, -inv_sqrt2)));
        assert!(approx_eq_c32(norm[3], Complex::new(inv_sqrt2, -inv_sqrt2)));
    }

    #[test]
    fn test_from_points_normalized_direct() {
        let raw_points = [
            Complex::new(10.0f32, 0.0f32),
            Complex::new(-10.0f32, 0.0f32),
        ];

        let norm: Constellation<f32, 2, Normalized> =
            Constellation::<f32, 2>::from_points_normalized(raw_points);

        assert!((norm.energy() - 1.0).abs() < EPS_F32);
        assert!((norm.scale_factor() - 0.1).abs() < EPS_F32);
        assert!(approx_eq_c32(norm[0], Complex::new(1.0, 0.0)));
        assert!(approx_eq_c32(norm[1], Complex::new(-1.0, 0.0)));
    }

    #[test]
    fn test_indexing_operator_across_states() {
        let raw_points = [Complex::new(3.0f32, 4.0f32), Complex::new(-3.0f32, -4.0f32)];

        let unnorm: Constellation<f32, 2, Unnormalized> =
            Constellation::<f32, 2>::from_points(raw_points);
        assert_eq!(unnorm[0], Complex::new(3.0, 4.0));
        assert_eq!(unnorm[1], Complex::new(-3.0, -4.0));

        let norm: Constellation<f32, 2, Normalized> = unnorm.normalize();
        assert!(approx_eq_c32(norm[0], Complex::new(0.6, 0.8)));
        assert!(approx_eq_c32(norm[1], Complex::new(-0.6, -0.8)));
    }

    #[test]
    fn test_const_context_instantiation() {
        static _BPSK_STATIC: Constellation<f32, 2> = Bpsk::<f32>::BPSK;
        static _QPSK_STATIC: Constellation<f64, 4> = Qpsk::<f64>::QPSK;
        static _QAM16_STATIC: Constellation<f32, 16> = Qam16::<f32>::QAM16;
        static _QAM64_STATIC: Constellation<f64, 64> = Qam64::<f64>::QAM64;
        static _QAM256_STATIC: Constellation<f32, 256> = Qam256::<f32>::QAM256;
        static _QAM1024_STATIC: Constellation<f64, 1024> = Qam1024::<f64>::QAM1024;
        static _QAM4096_STATIC: Constellation<f32, 4096> = Qam4096::<f32>::QAM4096;

        assert_eq!(_QAM4096_STATIC.points().len(), 4096);
        assert_eq!(Constellation::<f32, 4096>::BITS_PER_SYMBOL, 12);

        // Const typestate normalization
        const RAW: Constellation<f64, 2, Unnormalized> =
            Constellation::<f64, 2>::from_points([Complex::new(5.0, 0.0), Complex::new(-5.0, 0.0)]);
        const NORM: Constellation<f64, 2, Normalized> = RAW.normalize();

        assert!((NORM.energy() - 1.0).abs() < EPS_F64);
        assert!(approx_eq_c64(NORM[0], Complex::new(1.0, 0.0)));
    }
}
