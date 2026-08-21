use crate::utils::{binary_to_gray, gives_square_constellation, is_valid_constellation_size};

use num_complex::Complex;

use trig_const::{cos, sin, sqrt};

pub type Bpsk<T = f32> = Constellation<T, 2>;
pub type Qpsk<T = f32> = Constellation<T, 4>;
pub type Psk8<T = f32> = Constellation<T, 8>;
pub type Qam16<T = f32> = Constellation<T, 16>;
pub type Qam64<T = f32> = Constellation<T, 64>;
pub type Qam256<T = f32> = Constellation<T, 256>;
pub type Qam1024<T = f32> = Constellation<T, 1024>;
pub type Qam4096<T = f32> = Constellation<T, 4096>;

/// Generalize constellation over f32 and f64
macro_rules! impl_constellation_float {
    ($ty: ident) => {
        impl<const M: usize> Constellation<$ty, M> {
            /// Construct constellation from points
            pub const fn from_points(points: [Complex<$ty>; M]) -> Self {
                assert!(is_valid_constellation_size(M));

                let energy = Self::total_energy(points);

                let avg_energy = energy / M as $ty;

                assert!(
                    avg_energy > 0.0,
                    "Constellation energy must be greater than zero"
                );

                let scale_factor = (1.0 as $ty) / sqrt(avg_energy as f64) as $ty;

                Self {
                    points,
                    energy: avg_energy,
                    scale_factor,
                    is_normalized: false,
                }
            }
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
                    is_normalized: true,
                }
            }

            /// Return the same constellation shape, but with the points normalized to have unitary
            /// average energy.
            pub const fn normalize(self) -> Self {
                if self.is_normalized {
                    self
                } else {
                    Self::from_points_normalized(self.points)
                }
            }
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
        }

        /// Utils
        impl<const M: usize> Constellation<$ty, M> {
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

            pub const fn is_normalized(&self) -> bool {
                self.is_normalized
            }
            pub const fn bits_per_symbol() -> usize {
                M.ilog2() as usize
            }

            const fn from_polar(r: $ty, theta: $ty) -> Complex<$ty> {
                Complex::new(
                    r * (cos(theta as f64) as $ty),
                    r * (sin(theta as f64) as $ty),
                )
            }
        }

        /// M-PSK
        impl<const M: usize> Constellation<$ty, M> {
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
        impl Constellation<$ty, 2> {
            pub const BPSK: Self = Self::bpsk();
            /// Generates a BPSK constellation.
            pub const fn bpsk() -> Self {
                Self::m_psk()
            }
        }

        /// QPSK
        impl Constellation<$ty, 4> {
            pub const QPSK: Self = Self::qpsk();
            /// Generates a QPSK constellation.
            pub const fn qpsk() -> Self {
                Self::m_psk_with_phase(core::$ty::consts::PI / 4.0)
            }
        }

        /// M-QAM using gray coding
        impl<const M: usize> Constellation<$ty, M> {
            /// Generates a square $M$-QAM constellation with Gray coding.
            ///
            /// # Panics
            /// Panics if `M` is not a valid square constellation size (e.g., 4, 16, 64, 256).
            pub const fn m_qam() -> Self {
                assert!(gives_square_constellation(M));

                let k = Self::bits_per_symbol();
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
            impl Constellation<$ty, $m> {
                /// Pre-evaluated constellation constant.
                pub const $const_name: Self = Self::m_qam();
            }

            impl Default for Constellation<$ty, $m> {
                #[inline]
                fn default() -> Self {
                    Self::$const_name
                }
            }
        )*
    };
}
/// A digital modulation constellation of size `M` using floating-point coordinates `T`.
///
/// Holds `M` constellation points in the complex plane along with their scaling
/// and energy metadata.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Constellation<T, const M: usize> {
    points: [Complex<T>; M],
    // Normalization factor applied to the constellation
    scale_factor: T,
    // Average energy
    energy: T,
    is_normalized: bool,
}

/// Allows indexing the constellation directly, without having to call .points() first
impl<T, const M: usize> core::ops::Index<usize> for Constellation<T, M> {
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

    fn approx_eq_c32(a: Complex<f32>, b: Complex<f32>) -> bool {
        (a.re - b.re).abs() < EPS_F32 && (a.im - b.im).abs() < EPS_F32
    }

    fn approx_eq_c64(a: Complex<f64>, b: Complex<f64>) -> bool {
        (a.re - b.re).abs() < EPS_F64 && (a.im - b.im).abs() < EPS_F64
    }

    #[test]
    fn test_bpsk_properties() {
        const BPSK: Constellation<f32, 2> = Bpsk::<f32>::bpsk();

        assert_eq!(Constellation::<f32, 2>::bits_per_symbol(), 1);
        assert!(BPSK.is_normalized());
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

        assert_eq!(Constellation::<f64, 4>::bits_per_symbol(), 2);
        assert!(QPSK.is_normalized());
        assert!((QPSK.energy() - 1.0).abs() < EPS_F64);

        let pts = QPSK.points();

        // All points must lie on the unit circle
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

        // Point 0 (0b00) is at angle PI/4 -> (+1/sqrt(2), +1/sqrt(2))
        assert!(approx_eq_c64(pts[0b00], Complex::new(inv_sqrt2, inv_sqrt2)));

        // Point 1 (0b01) is at angle 3*PI/4 -> (-1/sqrt(2), +1/sqrt(2))
        assert!(approx_eq_c64(
            pts[0b01],
            Complex::new(-inv_sqrt2, inv_sqrt2)
        ));

        // Point 3 (0b11) is at angle 5*PI/4 -> (-1/sqrt(2), -1/sqrt(2))
        assert!(approx_eq_c64(
            pts[0b11],
            Complex::new(-inv_sqrt2, -inv_sqrt2)
        ));

        // Point 2 (0b10) is at angle 7*PI/4 -> (+1/sqrt(2), -1/sqrt(2))
        assert!(approx_eq_c64(
            pts[0b10],
            Complex::new(inv_sqrt2, -inv_sqrt2)
        ));
    }
    #[test]
    fn test_qam16_properties() {
        const QAM16: Constellation<f32, 16> = Constellation::<f32, 16>::m_qam();

        assert_eq!(Constellation::<f32, 16>::bits_per_symbol(), 4);
        assert!(QAM16.is_normalized());
        assert!((QAM16.energy() - 1.0).abs() < EPS_F32);

        // Theoretical unnormalized average energy for 16-QAM is 2*(16-1)/3 = 10
        // Expected scale factor = 1 / sqrt(10) ≈ 0.31622777
        let expected_scale = 1.0f32 / (10.0f32).sqrt();
        assert!((QAM16.scale_factor() - expected_scale).abs() < EPS_F32);

        let pts = QAM16.points();

        // 1. Verify all 16 coordinates are unique
        for i in 0..16 {
            for j in (i + 1)..16 {
                assert!(
                    !approx_eq_c32(pts[i], pts[j]),
                    "Points {i} and {j} overlap at {:?}",
                    pts[i]
                );
            }
        }

        // 2. Verify all unscaled points match the Cartesian lattice {-3, -1, 1, 3}
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

        // 3. Verify total unit average power across all points
        let mut total_power = 0.0f32;
        for p in pts {
            total_power += p.re * p.re + p.im * p.im;
        }
        let avg_power = total_power / 16.0;
        assert!((avg_power - 1.0).abs() < EPS_F32);
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
        assert_eq!(Constellation::<f32, 4096>::bits_per_symbol(), 12);
    }
}
