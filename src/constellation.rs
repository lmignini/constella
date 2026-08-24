use crate::bits::BitOrder;
use crate::bits::MsbFirst;
use crate::utils::{binary_to_gray, gives_square_constellation, is_valid_constellation_size};
use core::marker::PhantomData;
use num_complex::Complex;
use trig_const::{cos, sin, sqrt};

/// Compile-time marker for normalized constellation energy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Normalized;

/// Compile-time marker for unnormalized constellation energy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Unnormalized;

pub trait ConstellationState: Copy + Clone {}
impl ConstellationState for Normalized {}
impl ConstellationState for Unnormalized {}

/// Compile-time marker for arbitrary constellation geometry (monomorphizes to Euclidean search).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct General;

/// Compile-time marker for square QAM geometry (monomorphizes to fast $O(1)$ 1D slicing).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SquareQam<T: Copy> {
    pub scale: T,
    pub offset: T,
}

pub trait ConstellationGeometry: Copy + Clone {}
impl ConstellationGeometry for General {}
impl<T: Copy> ConstellationGeometry for SquareQam<T> {}

pub type Bpsk<T = f32, S = Normalized> = Constellation<T, 2, S, General>;
pub type Qpsk<T = f32, S = Normalized> = Constellation<T, 4, S, General>;
pub type Psk8<T = f32, S = Normalized> = Constellation<T, 8, S, General>;

pub type Qam16<T = f32, S = Normalized> = Constellation<T, 16, S, SquareQam<T>>;
pub type Qam64<T = f32, S = Normalized> = Constellation<T, 64, S, SquareQam<T>>;
pub type Qam256<T = f32, S = Normalized> = Constellation<T, 256, S, SquareQam<T>>;
pub type Qam1024<T = f32, S = Normalized> = Constellation<T, 1024, S, SquareQam<T>>;
pub type Qam4096<T = f32, S = Normalized> = Constellation<T, 4096, S, SquareQam<T>>;

/// A digital modulation constellation of size `M` using scalar precision `T`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(Rust)]
pub struct Constellation<T, const M: usize, S = Normalized, G = General>
where
    S: ConstellationState,
    G: ConstellationGeometry,
{
    points: [Complex<T>; M],
    scale_factor: T,
    energy: T,
    geometry: G,
    _state: PhantomData<S>,
}

impl<T, const M: usize, S: ConstellationState, G: ConstellationGeometry> core::ops::Index<usize>
    for Constellation<T, M, S, G>
{
    type Output = Complex<T>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.points[index]
    }
}

impl<T: Copy, const M: usize, S: ConstellationState, G: ConstellationGeometry>
    Constellation<T, M, S, G>
{
    pub const BITS_PER_SYMBOL: usize = M.ilog2() as usize;

    #[inline]
    pub const fn energy(&self) -> T {
        self.energy
    }

    #[inline]
    pub const fn scale_factor(&self) -> T {
        self.scale_factor
    }

    #[inline]
    pub const fn points(&self) -> &[Complex<T>; M] {
        &self.points
    }
}

macro_rules! impl_constellation_float {
    ($ty: ident) => {
        // Universal constructors accessible on Constellation<T, M, ...>
        impl<const M: usize, S: ConstellationState, G: ConstellationGeometry>
            Constellation<$ty, M, S, G>
        {
            /// Constructs an unnormalized general constellation from raw points.
            pub const fn from_points(
                points: [Complex<$ty>; M],
            ) -> Constellation<$ty, M, Unnormalized, General> {
                assert!(is_valid_constellation_size(M));
                let energy = Self::total_energy(points);
                let avg_energy = energy / M as $ty;
                assert!(avg_energy > 0.0, "Constellation energy must be > 0");
                let scale_factor = (1.0 as $ty) / sqrt(avg_energy as f64) as $ty;

                Constellation {
                    points,
                    energy: avg_energy,
                    scale_factor,
                    geometry: General,
                    _state: PhantomData,
                }
            }

            /// Constructs a normalized general constellation from raw points.
            pub const fn from_points_normalized(
                points: [Complex<$ty>; M],
            ) -> Constellation<$ty, M, Normalized, General> {
                assert!(is_valid_constellation_size(M));
                let energy = Self::total_energy(points);
                let avg_energy = energy / M as $ty;
                assert!(avg_energy > 0.0, "Constellation energy must be > 0");

                let scale_factor = (1.0 as $ty) / sqrt(avg_energy as f64) as $ty;
                let mut i = 0;
                let mut new_points = [Complex::new(0.0 as $ty, 0.0 as $ty); M];
                while i < M {
                    let point = points[i];
                    new_points[i] = Complex::new(point.re * scale_factor, point.im * scale_factor);
                    i += 1;
                }
                Constellation {
                    points: new_points,
                    scale_factor,
                    energy: 1.0,
                    geometry: General,
                    _state: PhantomData,
                }
            }

            const fn total_energy(points: [Complex<$ty>; M]) -> $ty {
                let mut i = 0;
                let mut acc = 0.0 as $ty;
                while i < M {
                    let point = points[i];
                    acc = acc + (point.re * point.re + point.im * point.im);
                    i += 1;
                }
                acc
            }

            const fn from_polar(r: $ty, theta: $ty) -> Complex<$ty> {
                Complex::new(
                    r * (cos(theta as f64) as $ty),
                    r * (sin(theta as f64) as $ty),
                )
            }
        }

        impl<const M: usize, G: ConstellationGeometry> Constellation<$ty, M, Unnormalized, G> {
            /// Normalizes constellation points to unit average energy at compile time.
            pub const fn normalize(self) -> Constellation<$ty, M, Normalized, G> {
                let norm = Constellation::<$ty, M, Normalized, General>::from_points_normalized(
                    self.points,
                );
                Constellation {
                    points: norm.points,
                    scale_factor: norm.scale_factor,
                    energy: norm.energy,
                    geometry: self.geometry,
                    _state: PhantomData,
                }
            }
        }

        // --- Fast-Path O(1) Slicing for Square QAM ---
        impl<const M: usize> Constellation<$ty, M, Normalized, SquareQam<$ty>> {
            #[inline(always)]
            pub fn demodulate_hard_point(&self, point: Complex<$ty>) -> usize {
                let k = Self::BITS_PER_SYMBOL;
                let k_axis = k / 2;
                let l = 1 << k_axis;

                let slice_axis = |val: $ty| -> usize {
                    let w = val * self.geometry.scale + self.geometry.offset;
                    if w <= 0.0 {
                        0
                    } else {
                        let idx = w as usize;
                        if idx >= l { l - 1 } else { idx }
                    }
                };

                let i = slice_axis(point.re);
                let q = slice_axis(point.im);
                (binary_to_gray(i) << k_axis) | binary_to_gray(q)
            }
        }

        // --- Linear O(M) Search for Arbitrary Geometry ---
        impl<const M: usize> Constellation<$ty, M, Normalized, General> {
            #[inline(always)]
            pub fn demodulate_hard_point(&self, point: Complex<$ty>) -> usize {
                let mut min_idx = 0;
                let diff0 = self.points[0] - point;
                let mut min_dist = diff0.re * diff0.re + diff0.im * diff0.im;

                let mut i = 1;
                while i < M {
                    let diff = self.points[i] - point;
                    let dist = diff.re * diff.re + diff.im * diff.im;
                    if dist < min_dist {
                        min_dist = dist;
                        min_idx = i;
                    }
                    i += 1;
                }
                min_idx
            }
        }

        // --- Soft Demodulation for Any Geometry ---
        impl<const M: usize, G: ConstellationGeometry> Constellation<$ty, M, Normalized, G> {
            #[inline]
            pub fn demodulate_soft_point<const K: usize, O: BitOrder + 'static>(
                &self,
                point: Complex<$ty>,
                noise_var: $ty,
            ) -> [$ty; K] {
                assert_eq!(1 << K, M, "K must satisfy 2^K == M");
                let mut dists = [0.0 as $ty; M];
                let mut i = 0;
                while i < M {
                    let diff = self.points[i] - point;
                    dists[i] = diff.re * diff.re + diff.im * diff.im;
                    i += 1;
                }

                let mut llrs = [0.0 as $ty; K];
                let scale = (1.0 as $ty) / (2.0 * noise_var);

                let mut j = 0;
                while j < K {
                    let mut min_d0 = <$ty>::INFINITY;
                    let mut min_d1 = <$ty>::INFINITY;
                    let mut idx = 0;
                    while idx < M {
                        let d = dists[idx];
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

        // --- Standard Constellation Generators ---
        impl<const M: usize> Constellation<$ty, M, Normalized, General> {
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

            pub const fn m_psk() -> Self {
                Self::m_psk_with_phase(0.0 as $ty)
            }
        }

        impl Constellation<$ty, 2, Normalized, General> {
            pub const BPSK: Self = Self::m_psk();
            pub const fn bpsk() -> Self {
                Self::m_psk()
            }
        }

        impl Constellation<$ty, 4, Normalized, General> {
            pub const QPSK: Self = Self::m_psk_with_phase(core::$ty::consts::PI / 4.0);
            pub const fn qpsk() -> Self {
                Self::m_psk_with_phase(core::$ty::consts::PI / 4.0)
            }
        }

        impl<const M: usize> Constellation<$ty, M, Normalized, SquareQam<$ty>> {
            pub const fn m_qam() -> Self {
                assert!(gives_square_constellation(M));
                let k = Self::BITS_PER_SYMBOL;
                let k_axis = k / 2;
                let l = 1 << k_axis;

                let mut points = [Complex::new(0.0 as $ty, 0.0 as $ty); M];
                let mut i = 0;
                let mut q = 0;

                while i < l {
                    while q < l {
                        let i_val = 2 * (i as isize) - (l as isize) + 1;
                        let q_val = 2 * (q as isize) - (l as isize) + 1;
                        let symbol_index = (binary_to_gray(i) << k_axis) | binary_to_gray(q);
                        points[symbol_index] = Complex::new(i_val as $ty, q_val as $ty);
                        q += 1;
                    }
                    q = 0;
                    i += 1;
                }
                let norm =
                    Constellation::<$ty, M, Normalized, General>::from_points_normalized(points);
                Self {
                    points: norm.points,
                    scale_factor: norm.scale_factor,
                    energy: norm.energy,
                    geometry: SquareQam {
                        scale: 1.0 / (2.0 * norm.scale_factor),
                        offset: l as $ty / 2.0,
                    },
                    _state: PhantomData,
                }
            }
        }
    };
}

macro_rules! impl_qam_consts {
    ($ty:ident, $( ($m:expr, $const_name:ident) ),* $(,)?) => {
        $(
            impl Constellation<$ty, $m, Normalized, SquareQam<$ty>> {
                pub const $const_name: Self = Self::m_qam();
            }

            impl Default for Constellation<$ty, $m, Normalized, SquareQam<$ty>> {
                #[inline]
                fn default() -> Self { Self::$const_name }
            }
        )*
    };
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
        const BPSK: Bpsk<f32> = Bpsk::<f32>::bpsk();
        assert_eq!(Bpsk::<f32>::BITS_PER_SYMBOL, 1);
        assert!((BPSK.energy() - 1.0).abs() < EPS_F32);

        let pts = BPSK.points();
        assert!(approx_eq_c32(pts[0], Complex::new(1.0, 0.0)));
        assert!(approx_eq_c32(pts[1], Complex::new(-1.0, 0.0)));
    }

    #[test]
    fn test_qpsk_properties() {
        const QPSK: Qpsk<f64> = Qpsk::<f64>::qpsk();
        assert_eq!(Qpsk::<f64>::BITS_PER_SYMBOL, 2);
        assert!((QPSK.energy() - 1.0).abs() < EPS_F64);

        let pts = QPSK.points();
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
        const PSK8: Psk8<f32> = Psk8::<f32>::m_psk();
        assert_eq!(Psk8::<f32>::BITS_PER_SYMBOL, 3);
        assert!((PSK8.energy() - 1.0).abs() < EPS_F32);
    }

    #[test]
    fn test_qam16_properties() {
        const QAM16: Qam16<f32> = Qam16::<f32>::QAM16;
        assert_eq!(Qam16::<f32>::BITS_PER_SYMBOL, 4);
        assert!((QAM16.energy() - 1.0).abs() < EPS_F32);

        let expected_scale = 1.0f32 / 10.0f32.sqrt();
        assert!((QAM16.scale_factor() - expected_scale).abs() < EPS_F32);

        let pts = QAM16.points();
        for i in 0..16 {
            for j in (i + 1)..16 {
                assert!(!approx_eq_c32(pts[i], pts[j]));
            }
        }
    }

    #[test]
    fn test_unnormalized_to_normalized_typestate_transition() {
        let raw_points = [
            Complex::new(2.0f32, 2.0f32),
            Complex::new(-2.0f32, 2.0f32),
            Complex::new(-2.0f32, -2.0f32),
            Complex::new(2.0f32, -2.0f32),
        ];

        let unnorm: Constellation<f32, 4, Unnormalized> =
            Constellation::<f32, 4>::from_points(raw_points);
        assert!((unnorm.energy() - 8.0).abs() < EPS_F32);

        let norm: Constellation<f32, 4, Normalized> = unnorm.normalize();
        assert!((norm.energy() - 1.0).abs() < EPS_F32);
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
        static _BPSK_STATIC: Bpsk<f32> = Bpsk::<f32>::BPSK;
        static _QPSK_STATIC: Qpsk<f64> = Qpsk::<f64>::QPSK;
        static _QAM16_STATIC: Qam16<f32> = Qam16::<f32>::QAM16;
        static _QAM64_STATIC: Qam64<f64> = Qam64::<f64>::QAM64;
        static _QAM256_STATIC: Qam256<f32> = Qam256::<f32>::QAM256;
        static _QAM1024_STATIC: Qam1024<f64> = Qam1024::<f64>::QAM1024;
        static _QAM4096_STATIC: Qam4096<f32> = Qam4096::<f32>::QAM4096;

        assert_eq!(_QAM4096_STATIC.points().len(), 4096);
        assert_eq!(Qam4096::<f32>::BITS_PER_SYMBOL, 12);

        const RAW: Constellation<f64, 2, Unnormalized> =
            Constellation::<f64, 2>::from_points([Complex::new(5.0, 0.0), Complex::new(-5.0, 0.0)]);
        const NORM: Constellation<f64, 2, Normalized> = RAW.normalize();

        assert!((NORM.energy() - 1.0).abs() < EPS_F64);
        assert!(approx_eq_c64(NORM[0], Complex::new(1.0, 0.0)));
    }
    #[test]
    fn test_qam_slicer_clamping_extreme_values() {
        let qam16 = Qam16::<f32>::QAM16;

        // Extremely distant points far outside the grid
        let far_top_right = Complex::new(100.0f32, 100.0f32);
        let far_bottom_left = Complex::new(-100.0f32, -100.0f32);

        let idx_tr = qam16.demodulate_hard_point(far_top_right);
        let idx_bl = qam16.demodulate_hard_point(far_bottom_left);

        assert!(idx_tr < 16);
        assert!(idx_bl < 16);
    }

    #[test]
    fn test_qam16_soft_demodulation_known_symbols() {
        let qam16 = Qam16::<f32>::QAM16;
        let noise_var = 0.25f32;

        // Symbol 0b0000 and 0b1111 exact points
        let p0 = qam16[0b0000];
        let p15 = qam16[0b1111];

        let llrs0 = qam16.demodulate_soft_point::<4, MsbFirst>(p0, noise_var);
        let llrs15 = qam16.demodulate_soft_point::<4, MsbFirst>(p15, noise_var);

        // For symbol 0b0000, all 4 bits are 0 -> all LLRs must be strictly positive
        for &llr in &llrs0 {
            assert!(llr > 0.0, "Expected positive LLR for bit 0, got {llr}");
        }

        // For symbol 0b1111, all 4 bits are 1 -> all LLRs must be strictly negative
        for &llr in &llrs15 {
            assert!(llr < 0.0, "Expected negative LLR for bit 1, got {llr}");
        }
    }
}
