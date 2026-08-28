//! Zero-allocation streaming differential phase encoding and decoding adapters.
//!
//! Differential Phase Shift Keying (DPSK) encodes digital information into the *phase transitions*
//! between consecutive symbols rather than their absolute phases:
//! $$y_k = y_{k-1} \cdot x_k$$
//! where $x_k$ is the incoming constellation symbol and $y_k$ is the transmitted cumulative symbol.
//!
//! At the receiver, non-coherent differential detection extracts the phase difference by multiplying
//! the received sample with the complex conjugate of the preceding sample:
//! $$z_k = r_k \cdot r_{k-1}^*$$
//! The resulting decision metric $z_k$ maps directly onto standard PSK decision boundaries,
//! eliminating the need for carrier phase synchronization.

use num_complex::Complex;
use num_traits::Float;

/// A streaming iterator adapter that performs differential phase encoding on complex baseband symbols.
///
/// Accumulates incoming phase steps $x_k$ via complex multiplication against the running state $y_{k-1}$:
/// $$y_k = y_{k-1} \cdot x_k$$
///
/// # Type Parameters
/// * `I`: The underlying symbol iterator yielding [`Complex<T>`].
/// * `T`: The floating-point precision type (`f32` or `f64`).
pub struct DifferentialEncoderIter<I, T> {
    iter: I,
    state: Complex<T>,
    sample_count: usize,
}

impl<I, T> DifferentialEncoderIter<I, T>
where
    T: Float,
    I: Iterator<Item = Complex<T>>,
{
    /// Creates a new differential encoder with the default reference phase of $1 + 0j$ ($0^\circ$).
    #[inline]
    pub fn new(iter: I) -> Self {
        Self {
            iter,
            state: Complex::new(T::one(), T::zero()),
            sample_count: 0,
        }
    }

    /// Creates a new differential encoder with an explicit initial reference state $y_0$.
    ///
    /// Useful for maintaining phase continuity across frame and burst boundaries.
    #[inline]
    pub fn with_initial_phase(iter: I, initial: Complex<T>) -> Self {
        Self {
            iter,
            state: initial,
            sample_count: 0,
        }
    }
}

impl<I, T> Iterator for DifferentialEncoderIter<I, T>
where
    T: Float,
    I: Iterator<Item = Complex<T>>,
{
    type Item = Complex<T>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let x_k = self.iter.next()?;
        let y_k = self.state * x_k;
        self.state = y_k;

        self.sample_count = self.sample_count.wrapping_add(1);
        if self.sample_count & 0x3FF == 0 {
            let norm = (self.state.re * self.state.re + self.state.im * self.state.im).sqrt();
            self.state.re = self.state.re / norm;
            self.state.im = self.state.im / norm;
        }

        Some(y_k)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let mut state = self.state;
        let mut sample_count = self.sample_count;

        self.iter.fold(init, |acc, x_k| {
            let y_k = state * x_k;
            state = y_k;

            sample_count = sample_count.wrapping_add(1);
            if sample_count & 0x3FF == 0 {
                let norm = (state.re * state.re + state.im * state.im).sqrt();
                state.re = state.re / norm;
                state.im = state.im / norm;
            }

            f(acc, y_k)
        })
    }
}

impl<I, T> ExactSizeIterator for DifferentialEncoderIter<I, T>
where
    T: Float,
    I: ExactSizeIterator<Item = Complex<T>>,
{
}

/// A streaming iterator adapter that performs non-coherent differential demodulation on received samples.
///
/// Computes the differential decision metric $z_k$ using delay-and-conjugate multiplication:
/// $$z_k = r_k \cdot r_{k-1}^*$$
///
/// # Type Parameters
/// * `I`: The underlying received sample iterator yielding [`Complex<T>`].
/// * `T`: The floating-point precision type (`f32` or `f64`).
pub struct DifferentialDecoderIter<I, T> {
    iter: I,
    prev_sample: Complex<T>,
}

impl<I, T> DifferentialDecoderIter<I, T>
where
    T: Float,
    I: Iterator<Item = Complex<T>>,
{
    /// Creates a new differential decoder assuming an initial reference sample of $1 + 0j$.
    #[inline]
    pub fn new(iter: I) -> Self {
        Self {
            iter,
            prev_sample: Complex::new(T::one(), T::zero()),
        }
    }

    /// Creates a new differential decoder with an explicit reference sample $r_0$ (e.g., a known pilot).
    #[inline]
    pub fn with_reference(iter: I, reference: Complex<T>) -> Self {
        Self {
            iter,
            prev_sample: reference,
        }
    }
}

impl<I, T> Iterator for DifferentialDecoderIter<I, T>
where
    T: Float,
    I: Iterator<Item = Complex<T>>,
{
    type Item = Complex<T>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let r_k = self.iter.next()?;
        let z_k = r_k * self.prev_sample.conj();
        self.prev_sample = r_k;
        Some(z_k)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let mut prev_sample = self.prev_sample;
        self.iter.fold(init, |acc, r_k| {
            let z_k = r_k * prev_sample.conj();
            prev_sample = r_k;
            f(acc, z_k)
        })
    }
}

impl<I, T> ExactSizeIterator for DifferentialDecoderIter<I, T>
where
    T: Float,
    I: ExactSizeIterator<Item = Complex<T>>,
{
}

/// Extension trait providing fluent differential encoding and decoding on complex symbol iterators.
pub trait DifferentialExt<T: Float>: Iterator<Item = Complex<T>> + Sized {
    /// Differentially encodes a baseband symbol stream starting from the reference phase $1 + 0j$.
    #[inline]
    fn differential_encode(self) -> DifferentialEncoderIter<Self, T> {
        DifferentialEncoderIter::new(self)
    }

    /// Differentially encodes a baseband symbol stream starting from an explicit initial reference state.
    #[inline]
    fn differential_encode_with_initial_phase(
        self,
        initial: Complex<T>,
    ) -> DifferentialEncoderIter<Self, T> {
        DifferentialEncoderIter::with_initial_phase(self, initial)
    }

    /// Differentially decodes a received sample stream against an initial reference sample of $1 + 0j$.
    #[inline]
    fn differential_decode(self) -> DifferentialDecoderIter<Self, T> {
        DifferentialDecoderIter::new(self)
    }

    /// Differentially decodes a received sample stream starting from an explicit reference sample.
    #[inline]
    fn differential_decode_with_reference(
        self,
        reference: Complex<T>,
    ) -> DifferentialDecoderIter<Self, T> {
        DifferentialDecoderIter::with_reference(self, reference)
    }
}

// Blanket implementation for all complex iterators
impl<T: Float, I: Iterator<Item = Complex<T>>> DifferentialExt<T> for I {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::ChannelExt;
    use crate::constellation::{Bpsk, Psk8, Qpsk};
    use crate::demodulation::DemodulateExt;
    use crate::modulation::ModulateExt;
    use alloc::vec;
    use alloc::vec::Vec;

    const EPS_F32: f32 = 1e-6;

    /// Verifies back-to-back DBPSK modulation and demodulation without channel distortion.
    ///
    /// In an ideal back-to-back loopback test:
    /// - The encoder starts accumulating from the default reference: y_0 = 1 + 0j.
    /// - The decoder starts demodulating from the default reference: r_0 = 1 + 0j.
    /// 
    /// Because no channel rotation is applied, the initial states match perfectly and
    /// the full payload recovers error-free.
    #[test]
    fn test_dbpsk_ideal_roundtrip() {
        let payload = vec![0b10110001, 0b11001010, 0b00110101];
        let bpsk = Bpsk::<f32>::BPSK;

        let recovered: Vec<u8> = payload
            .clone()
            .into_iter()
            .modulate(&bpsk)
            .differential_encode()
            .differential_decode()
            .demodulate_hard(&bpsk)
            .collect();

        assert_eq!(recovered, payload);
    }

    /// Verifies back-to-back DQPSK modulation and demodulation without channel distortion.
    #[test]
    fn test_dqpsk_ideal_roundtrip() {
        let payload = vec![0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];
        let qpsk = Qpsk::<f32>::QPSK;

        let recovered: Vec<u8> = payload
            .clone()
            .into_iter()
            .modulate(&qpsk)
            .differential_encode()
            .differential_decode()
            .demodulate_hard(&qpsk)
            .collect();

        assert_eq!(recovered, payload);
    }

    /// Verifies 3-bit spanning D-8PSK across byte boundaries.
    #[test]
    fn test_d8psk_spanning_roundtrip() {
        // 3 bytes = 24 bits = exactly eight 3-bit symbols (clean boundary)
        let payload = vec![0b101_100_11, 0b0_101_110_0, 0b11_000_111];
        let psk8 = Psk8::<f64>::m_psk();

        let recovered: Vec<u8> = payload
            .clone()
            .into_iter()
            .modulate(&psk8)
            .differential_encode()
            .differential_decode()
            .demodulate_hard(&psk8)
            .collect();

        assert_eq!(recovered, payload);
    }

    /// Verifies phase invariance over a channel with an unknown static phase offset using a pilot symbol.
    ///
    /// # The Pilot Mechanism
    /// Differential detection computes phase differences: z_k = r_k * conj(r_{k-1}).
    /// When passing through an unknown channel rotation e^{jθ}:
    /// 1. The transmitter prepends an unmodulated pilot symbol p_0 = 1 + 0j.
    /// 2. Over the air, the pilot rotates to r_0 = 1 * e^{jθ} = e^{jθ}.
    /// 3. When the decoder processes r_0, it outputs a dummy metric against its default
    ///    unrotated state (1 + 0j) and updates its internal memory to r_0 = e^{jθ}.
    /// 4. When the first true data symbol r_1 = y_1 * e^{jθ} arrives, the decoder calculates:
    /// 
    /// z_1 = r_1 * conj(r_0) = (y_1 * e^{jθ}) * conj(e^{jθ}) = y_1
    /// 
    ///    The unknown channel angle θ cancels out completely.
    /// 5. Calling `.skip(1)` drops the dummy metric z_0, leaving only valid payload symbols.
    #[test]
    fn test_dpsk_static_phase_invariance_with_pilot() {
        let payload = vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE];
        let qpsk = Qpsk::<f64>::QPSK;

        // Introduce a severe arbitrary static channel rotation (e.g. +73.5 degrees)
        let phase_offset = 73.5 * core::f64::consts::PI / 180.0;
        let pilot = Complex::new(1.0f64, 0.0f64);

        // 1. Modulate payload and differentially encode transitions
        let tx_data_symbols = payload
            .clone()
            .into_iter()
            .modulate(&qpsk)
            .differential_encode();

        // 2. Prepend the reference pilot symbol and send the burst across the channel
        let rx_channel_stream = core::iter::once(pilot)
            .chain(tx_data_symbols)
            .add_phase_offset(phase_offset);

        // 3. Differentially decode and drop the dummy metric emitted by the pilot step
        let recovered_symbols = rx_channel_stream.differential_decode().skip(1);

        // 4. Demodulate the cleaned baseband symbols back into bytes
        let recovered: Vec<u8> = recovered_symbols.demodulate_hard(&qpsk).collect();

        assert_eq!(recovered, payload);
    }

    /// Verifies static phase invariance when the receiver initializes directly with an acquired reference sample.
    ///
    /// If the receiver has already estimated the channel phase (e.g. r_0 = e^{jθ})
    /// from a preceding preamble or burst sync sequence, it can initialize the decoder
    /// directly using `.differential_decode_with_reference(r_0)` without needing `.skip(1)`.
    #[test]
    fn test_dpsk_static_phase_invariance_with_known_reference() {
        let payload = vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE];
        let qpsk = Qpsk::<f64>::QPSK;

        let phase_offset = 73.5 * core::f64::consts::PI / 180.0;

        // The reference symbol (1 + 0j) after experiencing the channel's phase rotation:
        let channel_reference = Complex::new(phase_offset.cos(), phase_offset.sin());

        let recovered: Vec<u8> = payload
            .clone()
            .into_iter()
            .modulate(&qpsk)
            .differential_encode()
            .add_phase_offset(phase_offset)
            .differential_decode_with_reference(channel_reference)
            .demodulate_hard(&qpsk)
            .collect();

        assert_eq!(recovered, payload);
    }

    /// Tests custom initial phase configuration for preserving encoder state across frame boundaries.
    #[test]
    fn test_differential_encoder_custom_initial_phase() {
        let bpsk = Bpsk::<f32>::BPSK;
        // BPSK symbols: bit 0 -> +1.0, bit 1 -> -1.0
        let symbols = vec![bpsk[1], bpsk[0]]; // Phase shifts: [-1.0, +1.0]

        // Start accumulation from -1 + 0j (instead of the default +1 + 0j)
        let initial_state = Complex::new(-1.0f32, 0.0f32);
        let encoded: Vec<Complex<f32>> = symbols
            .into_iter()
            .differential_encode_with_initial_phase(initial_state)
            .collect();

        // Step 1: y_1 = y_0 * x_1 = (-1.0) * (-1.0) = +1.0 + 0j
        // Step 2: y_2 = y_1 * x_2 = (+1.0) * (+1.0) = +1.0 + 0j
        assert!((encoded[0].re - 1.0).abs() < EPS_F32);
        assert!(encoded[0].im.abs() < EPS_F32);
        assert!((encoded[1].re - 1.0).abs() < EPS_F32);
        assert!(encoded[1].im.abs() < EPS_F32);
    }

    /// Tests custom reference sample configuration for the differential decoder.
    #[test]
    fn test_differential_decoder_custom_reference() {
        // Received sample stream containing a single sample r_1 = +1.0 + 0j
        let rx_samples = vec![Complex::new(1.0f32, 0.0f32)];
        // Known reference sample from previous time step r_0 = -1.0 + 0j
        let reference = Complex::new(-1.0f32, 0.0f32);

        let decoded: Vec<Complex<f32>> = rx_samples
            .into_iter()
            .differential_decode_with_reference(reference)
            .collect();

        // Decision metric: z_1 = r_1 * conj(r_0) = (1.0 + 0j) * (-1.0 - 0j) = -1.0 + 0j
        assert!((decoded[0].re - (-1.0)).abs() < EPS_F32);
        assert!(decoded[0].im.abs() < EPS_F32);
    }

    /// Verifies exact size hints and standard iterator contract bounds.
    #[test]
    fn test_differential_iterators_exact_size_and_hints() {
        let symbols = vec![
            Complex::new(1.0f64, 0.0f64),
            Complex::new(0.0f64, 1.0f64),
            Complex::new(-1.0f64, 0.0f64),
        ];

        let mut enc = symbols.clone().into_iter().differential_encode();
        assert_eq!(enc.len(), 3);
        assert_eq!(enc.size_hint(), (3, Some(3)));
        enc.next();
        assert_eq!(enc.len(), 2);

        let mut dec = symbols.into_iter().differential_decode();
        assert_eq!(dec.len(), 3);
        assert_eq!(dec.size_hint(), (3, Some(3)));
        dec.next();
        assert_eq!(dec.len(), 2);
    }

    /// Verifies graceful handling of empty streams.
    #[test]
    fn test_empty_stream() {
        let empty: [Complex<f32>; 0] = [];

        let mut enc = empty.into_iter().differential_encode();
        assert_eq!(enc.next(), None);

        let mut dec = empty.into_iter().differential_decode();
        assert_eq!(dec.next(), None);
    }

    #[test]
    fn test_dbpsk_awgn_high_snr_zero_ber() {
        use rand_core::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;

        let payload = vec![0x55, 0xAA, 0x12, 0x34, 0xDE, 0xAD, 0xBE, 0xEF];
        let bpsk = Bpsk::<f32>::BPSK; //
        let rng = Xoshiro256PlusPlus::seed_from_u64(42); //
        let pilot = Complex::new(1.0f32, 0.0f32);

        // At 30 dB SNR, error probability is practically 0
        let recovered: Vec<u8> = core::iter::once(pilot)
            .chain(
                payload
                    .clone()
                    .into_iter()
                    .modulate(&bpsk)
                    .differential_encode(),
            ) //[cite: 1, 14]
            .add_awgn_snr(&bpsk, 30.0, rng) //
            .differential_decode() //
            .skip(1)
            .demodulate_hard(&bpsk) //
            .collect();

        assert_eq!(recovered, payload); //
    }

    #[test]
    fn test_dqpsk_awgn_high_snr_zero_ber() {
        use rand_core::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;

        let payload = vec![0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];
        let qpsk = Qpsk::<f32>::QPSK; //
        let rng = Xoshiro256PlusPlus::seed_from_u64(1234); //
        let pilot = Complex::new(1.0f32, 0.0f32);

        // At 30 dB SNR, DQPSK should have zero bit errors
        let recovered: Vec<u8> = core::iter::once(pilot)
            .chain(
                payload
                    .clone()
                    .into_iter()
                    .modulate(&qpsk)
                    .differential_encode(),
            ) //[cite: 1, 14]
            .add_awgn_snr(&qpsk, 30.0, rng) //
            .differential_decode() //
            .skip(1)
            .demodulate_hard(&qpsk) //
            .collect();

        assert_eq!(recovered, payload); //
    }

    #[test]
    fn test_dbpsk_ber_curve_against_theory() {
        use rand_core::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;

        let bpsk = Bpsk::<f64>::BPSK; //
        let num_bytes = 20_000; // 160,000 bits per SNR evaluation point
        let total_bits = (num_bytes * 8) as f64; //
        let pilot = Complex::new(1.0f64, 0.0f64);

        // Generate deterministic payload
        let payload: Vec<u8> = (0..num_bytes).map(|i| (i * 137 + 17) as u8).collect();

        // ---------------------------------------------------------------------
        // 1. Low Eb/N0: 0.0 dB
        // Theoretical DBPSK bit error probability: Pb = 0.5 * exp(-Eb/N0_lin)
        // Pb = 0.5 * exp(-1.0) ≈ 0.18394 (18.39%)
        // ---------------------------------------------------------------------
        let rng_0db = Xoshiro256PlusPlus::seed_from_u64(4242); //
        let rx_0db: Vec<u8> = core::iter::once(pilot)
            .chain(
                payload
                    .clone()
                    .into_iter()
                    .modulate(&bpsk)
                    .differential_encode(),
            ) //[cite: 1, 14]
            .add_awgn_ebn0(&bpsk, 0.0, rng_0db) //
            .differential_decode() //
            .skip(1)
            .demodulate_hard(&bpsk) //
            .collect();

        let bit_errors_0db: usize = payload
            .iter()
            .zip(rx_0db.iter())
            .map(|(&a, &b)| (a ^ b).count_ones() as usize) //
            .sum(); //
        let ber_0db = (bit_errors_0db as f64) / total_bits; //

        assert!(
            (ber_0db - 0.18394).abs() < 0.015,
            "DBPSK BER at 0 dB was {ber_0db:.5}, expected ~0.18394"
        );

        // ---------------------------------------------------------------------
        // 2. Medium Eb/N0: 4.0 dB
        // Pb = 0.5 * exp(-2.51189) ≈ 0.04055 (4.06%)
        // ---------------------------------------------------------------------
        let rng_4db = Xoshiro256PlusPlus::seed_from_u64(8484); //
        let rx_4db: Vec<u8> = core::iter::once(pilot)
            .chain(
                payload
                    .clone()
                    .into_iter()
                    .modulate(&bpsk)
                    .differential_encode(),
            ) //[cite: 1, 14]
            .add_awgn_ebn0(&bpsk, 4.0, rng_4db) //
            .differential_decode() //
            .skip(1)
            .demodulate_hard(&bpsk) //
            .collect();

        let bit_errors_4db: usize = payload
            .iter()
            .zip(rx_4db.iter())
            .map(|(&a, &b)| (a ^ b).count_ones() as usize) //
            .sum(); //
        let ber_4db = (bit_errors_4db as f64) / total_bits; //

        assert!(
            (ber_4db - 0.04055).abs() < 0.005,
            "DBPSK BER at 4.0 dB was {ber_4db:.5}, expected ~0.04055"
        );

        // ---------------------------------------------------------------------
        // 3. High Eb/N0: 7.0 dB
        // Pb = 0.5 * exp(-5.01187) ≈ 0.00332 (0.332%)
        // ---------------------------------------------------------------------
        let rng_7db = Xoshiro256PlusPlus::seed_from_u64(126126);
        let rx_7db: Vec<u8> = core::iter::once(pilot)
            .chain(
                payload
                    .clone()
                    .into_iter()
                    .modulate(&bpsk)
                    .differential_encode(),
            ) //[cite: 1, 14]
            .add_awgn_ebn0(&bpsk, 7.0, rng_7db) //
            .differential_decode() //
            .skip(1)
            .demodulate_hard(&bpsk) //
            .collect();

        let bit_errors_7db: usize = payload
            .iter()
            .zip(rx_7db.iter())
            .map(|(&a, &b)| (a ^ b).count_ones() as usize) //
            .sum(); //
        let ber_7db = (bit_errors_7db as f64) / total_bits; //

        assert!(
            (ber_7db - 0.00332).abs() < 0.0015,
            "DBPSK BER at 7.0 dB was {ber_7db:.5}, expected ~0.00332"
        );
    }

    #[test]
    fn test_dbpsk_ber_curve_fine_0_2db_sweep() {
        use rand_core::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;

        let bpsk = Bpsk::<f64>::BPSK; //
        let num_bytes = 10_000; // 80,000 bits per step
        let total_bits = (num_bytes * 8) as f64; //
        let pilot = Complex::new(1.0f64, 0.0f64);

        // Generate deterministic pseudo-random payload
        let payload: Vec<u8> = (0..num_bytes).map(|i| (i * 157 + 31) as u8).collect(); //

        // Sweep from 0.0 dB to 8.0 dB in 0.2 dB increments (41 evaluation points)
        for step in 0..=40 {
            let ebn0_db = (step as f64) * 0.2;
            let ebn0_lin = 10.0f64.powf(ebn0_db / 10.0); //

            // Theoretical DBPSK error probability in AWGN: Pb = 0.5 * exp(-Eb/N0)
            let theoretical_ber = 0.5 * (-ebn0_lin).exp();

            // Run end-to-end differential pipeline with unique deterministic seed per step
            let rng = Xoshiro256PlusPlus::seed_from_u64(0xDB95_0000 + step as u64);
            let rx_bytes: Vec<u8> = core::iter::once(pilot)
                .chain(
                    payload
                        .clone()
                        .into_iter()
                        .modulate(&bpsk)
                        .differential_encode(),
                ) //[cite: 1, 14]
                .add_awgn_ebn0(&bpsk, ebn0_db, rng) //
                .differential_decode() //
                .skip(1)
                .demodulate_hard(&bpsk) //
                .collect();

            // Count bit errors
            let bit_errors: usize = payload
                .iter()
                .zip(rx_bytes.iter())
                .map(|(&tx, &rx)| (tx ^ rx).count_ones() as usize) //
                .sum(); //
            let empirical_ber = (bit_errors as f64) / total_bits; //

            // Binomial standard deviation: sigma = sqrt(p * (1 - p) / N)
            let std_err = (theoretical_ber * (1.0 - theoretical_ber) / total_bits).sqrt(); //
            let margin = 4.0 * std_err + 0.0005; // 4-sigma envelope + small base floor

            let delta = (empirical_ber - theoretical_ber).abs(); //
            assert!(
                delta <= margin,
                "Failed at Eb/N0 = {ebn0_db:.1} dB: Empirical BER = {empirical_ber:.5}, \
                 Theoretical = {theoretical_ber:.5}, Delta = {delta:.5} (allowed margin = {margin:.5})"
            ); //
        }
    }
}
