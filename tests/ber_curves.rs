//! BER-vs-theory validation suite for `constella`.
//!
//! Validates modem implementations across AWGN, Rayleigh, Rician, Phase/CFO,
//! Differential (DPSK), and Equalization channels against analytical reference curves.

// =============================================================================
// Module: theory
// =============================================================================

pub mod theory {
    /// Gaussian Q-function: Q(x) = 0.5 * erfc(x / sqrt(2)).
    #[inline]
    pub fn q(x: f64) -> f64 {
        0.5 * libm::erfc(x * core::f64::consts::FRAC_1_SQRT_2)
    }

    #[inline]
    pub fn db_to_lin(db: f64) -> f64 {
        10f64.powf(db / 10.0)
    }

    // --- AWGN Curves ---

    #[inline]
    pub fn ber_bpsk_awgn(ebn0: f64) -> f64 {
        q((2.0 * ebn0).sqrt())
    }

    #[inline]
    pub fn ber_qpsk_awgn(ebn0: f64) -> f64 {
        ber_bpsk_awgn(ebn0)
    }

    #[inline]
    pub fn ber_mpsk_awgn(m: usize, ebn0: f64) -> f64 {
        let k = (m as f64).log2();
        (2.0 / k) * q((2.0 * k * ebn0).sqrt() * (core::f64::consts::PI / m as f64).sin())
    }

    #[inline]
    pub fn ber_sqqam_awgn(m: usize, ebn0: f64) -> f64 {
        let k = (m as f64).log2();
        let root_m = (m as f64).sqrt();
        (4.0 / k) * (1.0 - 1.0 / root_m) * q((3.0 * k * ebn0 / (m as f64 - 1.0)).sqrt())
    }

    // --- Rayleigh Unified Identity ---

    #[inline]
    pub fn rayleigh_average(c: f64, a: f64, ebn0_avg: f64) -> f64 {
        let x = a * ebn0_avg;
        c * 0.5 * (1.0 - (x / (1.0 + x)).sqrt())
    }

    // --- Phase & Differential Curves ---

    #[inline]
    pub fn ber_bpsk_phase(ebn0: f64, theta: f64) -> f64 {
        q((2.0 * ebn0).sqrt() * theta.cos())
    }

    #[inline]
    pub fn ber_qpsk_phase(ebn0: f64, theta: f64) -> f64 {
        let arg = (2.0 * ebn0).sqrt();
        0.5 * (q(arg * (theta.cos() + theta.sin())) + q(arg * (theta.cos() - theta.sin())))
    }

    #[inline]
    pub fn ber_dbpsk_awgn(ebn0: f64) -> f64 {
        0.5 * (-ebn0).exp()
    }
}

// =============================================================================
// Module: harness
// =============================================================================

pub mod harness {
    use rand_core::{Rng, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    pub const FAST_TARGET_ERRORS: f64 = 200.0;
    pub const THOROUGH_TARGET_ERRORS: f64 = 2000.0;

    #[derive(Debug, Clone, Copy)]
    pub struct BerPoint {
        pub bits: u64,
        pub errors: u64,
    }

    impl BerPoint {
        #[inline]
        pub fn ber(&self) -> f64 {
            self.errors as f64 / self.bits as f64
        }
    }

    /// Generates a reproducible pseudo-random byte payload from a seed.
    pub fn random_payload(n_bytes: usize, seed: u64) -> Vec<u8> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut v = Vec::with_capacity(n_bytes);
        while v.len() < n_bytes {
            let word = rng.next_u64();
            let bytes = word.to_le_bytes();
            let take = (n_bytes - v.len()).min(8);
            v.extend_from_slice(&bytes[..take]);
        }
        v
    }

    /// Computes the payload byte count needed to observe `target_errors` at `p_theory`.
    pub fn payload_bytes_for(target_errors: f64, p_theory: f64, k: usize) -> usize {
        let bits_needed = target_errors / p_theory;
        let mut bytes = ((bits_needed / 8.0).ceil() as usize).max(1024);
        if k > 0 {
            let rem = bytes % k;
            if rem != 0 {
                bytes += k - rem;
            }
        }
        bytes
    }

    /// Counts bit errors between transmitted and received symbol indices.
    pub fn count_symbol_bit_errors<const K: usize>(
        tx: impl Iterator<Item = usize>,
        rx: impl Iterator<Item = usize>,
    ) -> BerPoint {
        let mut bits = 0u64;
        let mut errors = 0u64;
        let mut tx_count = 0u64;
        let mut rx_count = 0u64;
        let mut tx_iter = tx;
        let mut rx_iter = rx;

        loop {
            match (tx_iter.next(), rx_iter.next()) {
                (Some(t), Some(r)) => {
                    errors += (t ^ r).count_ones() as u64;
                    bits += K as u64;
                    tx_count += 1;
                    rx_count += 1;
                }
                (Some(_), None) => {
                    tx_count += 1;
                    for _ in tx_iter {
                        tx_count += 1;
                    }
                    break;
                }
                (None, Some(_)) => {
                    rx_count += 1;
                    for _ in rx_iter {
                        rx_count += 1;
                    }
                    break;
                }
                (None, None) => break,
            }
        }

        assert_eq!(
            tx_count, rx_count,
            "TX produced {tx_count} symbols but RX produced {rx_count} symbols (length mismatch)"
        );
        BerPoint { bits, errors }
    }

    /// Asserts that an observed error count is statistically consistent with `p_theory`.
    pub fn assert_ber_consistent(
        label: &str,
        point: BerPoint,
        p_theory: f64,
        sigmas: f64,
        rel_model_err: f64,
    ) {
        let n = point.bits as f64;
        let expected = n * p_theory;
        assert!(
            expected >= 50.0,
            "{label}: only {expected:.1} expected errors — increase payload size (rule 4)"
        );

        let stat = sigmas * (n * p_theory * (1.0 - p_theory)).sqrt();
        let model = rel_model_err * expected;
        let margin = stat + model;
        let observed = point.errors as f64;
        let delta = (observed - expected).abs();
        let sigma_dev = delta / (n * p_theory * (1.0 - p_theory)).sqrt();
        let ber = point.ber();

        assert!(
            delta <= margin,
            "{label}: observed {observed:.0} errors ({ber:.3e}), expected {expected:.1} ({p_theory:.3e}) \
             over {n:.0} bits. |Delta| = {delta:.1} > margin {margin:.1} \
             (statistical {stat:.1} + model {model:.1}) = {sigma_dev:.2} sigma"
        );
    }
}

// =============================================================================
// Module: oracle
// =============================================================================

mod oracle {
    use super::theory;
    use constella::channel::ChannelExt;
    use constella::constellation::Qpsk;
    use num_complex::Complex;
    use rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn q_function_matches_known_values() {
        for (x, expected) in [
            (0.0, 5.000_000_000e-1),
            (1.0, 1.586_552_539e-1),
            (2.0, 2.275_013_195e-2),
            (3.0, 1.349_898_032e-3),
            (4.0, 3.167_124_183e-5),
            (5.0, 2.866_515_719e-7),
            (6.0, 9.865_876_450e-10),
        ] {
            let got = theory::q(x);
            assert!(
                (got / expected - 1.0).abs() < 1e-9,
                "Q({x}) = {got:.12e}, expected {expected:.12e}"
            );
        }
    }

    #[test]
    fn qam4_curve_equals_qpsk_curve() {
        for step in 0..=30 {
            let ebn0_db = step as f64 * 0.5;
            let ebn0_lin = theory::db_to_lin(ebn0_db);
            let qpsk_ber = theory::ber_qpsk_awgn(ebn0_lin);
            let qam4_ber = theory::ber_sqqam_awgn(4, ebn0_lin);
            assert!(
                (qpsk_ber - qam4_ber).abs() < 1e-12,
                "4-QAM ({qam4_ber}) did not match QPSK ({qpsk_ber}) at {ebn0_db} dB"
            );
        }
    }

    #[test]
    fn qam16_formula_matches_approx() {
        for step in 0..=20 {
            let ebn0_lin = theory::db_to_lin(step as f64 * 0.5);
            let direct = 0.75 * theory::q((0.8 * ebn0_lin).sqrt());
            let formula = theory::ber_sqqam_awgn(16, ebn0_lin);
            assert!((direct - formula).abs() < 1e-12);
        }
    }

    #[test]
    fn rayleigh_identity_matches_numeric_integration() {
        let a = 1.0;
        let gbar = 10.0;
        let analytic = theory::rayleigh_average(1.0, a, gbar);

        let n_steps = 100_000;
        let gamma_max = 40.0 * gbar;
        let d_gamma = gamma_max / (n_steps as f64);
        let mut integral = 0.0;

        for i in 0..n_steps {
            let gamma = (i as f64 + 0.5) * d_gamma;
            let p_inst = theory::q((2.0 * a * gamma).sqrt());
            let pdf = (1.0 / gbar) * (-gamma / gbar).exp();
            integral += p_inst * pdf * d_gamma;
        }

        assert!(
            (integral - analytic).abs() < 1e-4,
            "Rayleigh numerical integral {integral} did not match analytic {analytic}"
        );
    }

    #[test]
    fn awgn_ebn0_delivers_requested_noise_variance() {
        let constellation = Qpsk::<f64>::QPSK;
        let k = 2.0;
        let ebn0_db = 6.0;
        let ebn0_lin = theory::db_to_lin(ebn0_db);
        let target_n0 = 1.0 / (k * ebn0_lin);

        let n_samples = 200_000;
        let rng = Xoshiro256PlusPlus::seed_from_u64(0xCAFE_0001);
        let zeros = vec![Complex::new(0.0f64, 0.0f64); n_samples];

        let noisy: Vec<Complex<f64>> = zeros
            .into_iter()
            .add_awgn_ebn0(&constellation, ebn0_db, rng)
            .collect();

        let sum_sq: f64 = noisy.iter().map(|s| s.norm_sqr()).sum();
        let empirical_n0 = sum_sq / (n_samples as f64);

        let std_err = target_n0 / (n_samples as f64).sqrt();
        let margin = 4.0 * std_err;

        assert!(
            (empirical_n0 - target_n0).abs() <= margin,
            "AWGN variance empirical {empirical_n0:.6}, expected {target_n0:.6}, margin {margin:.6}"
        );
    }
}

// =============================================================================
// Module: awgn
// =============================================================================

mod awgn {
    use super::{harness, theory};
    use constella::bits::ChunkBitsExt;
    use constella::channel::ChannelExt;
    use constella::constellation::{Bpsk, Psk8, Qam16, Qam64, Qam256, Qam1024, Qam4096, Qpsk};
    use constella::demodulation::DemodulateExt;
    use constella::modulation::ModulateExt;
    use rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    macro_rules! run_awgn_point {
        ($k:expr, $label:expr, $constellation:expr, $ebn0_db:expr, $p_theory:expr, $target_errors:expr, $rel_model_err:expr, $seed:expr) => {{
            let n_bytes = harness::payload_bytes_for($target_errors, $p_theory, $k);
            let payload = harness::random_payload(n_bytes, $seed ^ 0xDEAD_BEEF);
            let rng_awgn = Xoshiro256PlusPlus::seed_from_u64($seed);

            let tx = payload.clone().into_iter().chunk_bits::<$k>();
            let rx = payload
                .into_iter()
                .modulate($constellation)
                .add_awgn_ebn0($constellation, $ebn0_db, rng_awgn)
                .demodulate_hard_symbols($constellation);

            let point = harness::count_symbol_bit_errors::<$k>(tx, rx);
            harness::assert_ber_consistent($label, point, $p_theory, 5.0, $rel_model_err);
            point
        }};
    }

    #[test]
    fn ber_awgn_bpsk_fast() {
        let c = Bpsk::<f64>::BPSK;
        let mut prev_ber = 1.0;
        for step in 0..=8 {
            let ebn0_db = step as f64 * 1.0;
            let ebn0_lin = theory::db_to_lin(ebn0_db);
            let p_theory = theory::ber_bpsk_awgn(ebn0_lin);
            let pt = run_awgn_point!(
                1,
                &format!("BPSK AWGN {ebn0_db:.1} dB"),
                &c,
                ebn0_db,
                p_theory,
                harness::FAST_TARGET_ERRORS,
                0.0,
                0x1000 + step as u64
            );
            assert!(pt.ber() <= prev_ber + 0.02, "Monotonicity violated");
            prev_ber = pt.ber();
        }
    }

    #[test]
    fn ber_awgn_qpsk_fast() {
        let c = Qpsk::<f64>::QPSK;
        let mut prev_ber = 1.0;
        for step in 0..=8 {
            let ebn0_db = step as f64 * 1.0;
            let ebn0_lin = theory::db_to_lin(ebn0_db);
            let p_theory = theory::ber_qpsk_awgn(ebn0_lin);
            let pt = run_awgn_point!(
                2,
                &format!("QPSK AWGN {ebn0_db:.1} dB"),
                &c,
                ebn0_db,
                p_theory,
                harness::FAST_TARGET_ERRORS,
                0.0,
                0x2000 + step as u64
            );
            assert!(pt.ber() <= prev_ber + 0.02, "Monotonicity violated");
            prev_ber = pt.ber();
        }
    }

    #[test]
    fn ber_awgn_8psk_fast() {
        let c = Psk8::<f64>::m_psk();
        let mut prev_ber = 1.0;
        for step in 3..=11 {
            let ebn0_db = step as f64 * 1.0;
            let ebn0_lin = theory::db_to_lin(ebn0_db);
            let p_theory = theory::ber_mpsk_awgn(8, ebn0_lin);
            if p_theory > 1e-2 {
                continue;
            }
            let pt = run_awgn_point!(
                3,
                &format!("8-PSK AWGN {ebn0_db:.1} dB"),
                &c,
                ebn0_db,
                p_theory,
                harness::FAST_TARGET_ERRORS,
                0.10,
                0x3000 + step as u64
            );
            assert!(pt.ber() <= prev_ber + 0.005, "Monotonicity violated");
            prev_ber = pt.ber();
        }
    }

    #[test]
    fn ber_awgn_16qam_fast() {
        let c = Qam16::<f64>::QAM16;
        let mut prev_ber = 1.0;
        for step in 4..=12 {
            let ebn0_db = step as f64 * 1.0;
            let ebn0_lin = theory::db_to_lin(ebn0_db);
            let p_theory = theory::ber_sqqam_awgn(16, ebn0_lin);
            if p_theory > 1e-2 {
                continue;
            }
            let pt = run_awgn_point!(
                4,
                &format!("16-QAM AWGN {ebn0_db:.1} dB"),
                &c,
                ebn0_db,
                p_theory,
                harness::FAST_TARGET_ERRORS,
                0.10,
                0x4000 + step as u64
            );
            assert!(pt.ber() <= prev_ber + 0.005, "Monotonicity violated");
            prev_ber = pt.ber();
        }
    }

    #[test]
    fn ber_awgn_64qam_fast() {
        let c = Qam64::<f64>::QAM64;
        let mut prev_ber = 1.0;
        for step in 8..=15 {
            let ebn0_db = step as f64 * 1.0;
            let ebn0_lin = theory::db_to_lin(ebn0_db);
            let p_theory = theory::ber_sqqam_awgn(64, ebn0_lin);
            if p_theory > 1e-2 {
                continue;
            }
            let pt = run_awgn_point!(
                6,
                &format!("64-QAM AWGN {ebn0_db:.1} dB"),
                &c,
                ebn0_db,
                p_theory,
                harness::FAST_TARGET_ERRORS,
                0.10,
                0x5000 + step as u64
            );
            assert!(pt.ber() <= prev_ber + 0.005, "Monotonicity violated");
            prev_ber = pt.ber();
        }
    }

    #[test]
    fn ber_awgn_256qam_fast() {
        let c = Qam256::<f64>::QAM256;
        for step in 12..=18 {
            let ebn0_db = step as f64 * 1.0;
            let ebn0_lin = theory::db_to_lin(ebn0_db);
            let p_theory = theory::ber_sqqam_awgn(256, ebn0_lin);
            if p_theory > 1e-2 {
                continue;
            }
            run_awgn_point!(
                8,
                &format!("256-QAM AWGN {ebn0_db:.1} dB"),
                &c,
                ebn0_db,
                p_theory,
                harness::FAST_TARGET_ERRORS,
                0.10,
                0x6000 + step as u64
            );
        }
    }

    #[test]
    #[ignore = "slow"]
    fn ber_awgn_psk_thorough() {
        let bpsk = Bpsk::<f64>::BPSK;
        let qpsk = Qpsk::<f64>::QPSK;
        for step in 0..=18 {
            let ebn0_db = step as f64 * 0.5;
            let ebn0_lin = theory::db_to_lin(ebn0_db);
            let p_theory = theory::ber_bpsk_awgn(ebn0_lin);
            run_awgn_point!(
                1,
                &format!("BPSK Thorough {ebn0_db:.1} dB"),
                &bpsk,
                ebn0_db,
                p_theory,
                harness::THOROUGH_TARGET_ERRORS,
                0.0,
                0x7000 + step as u64
            );
            run_awgn_point!(
                2,
                &format!("QPSK Thorough {ebn0_db:.1} dB"),
                &qpsk,
                ebn0_db,
                p_theory,
                harness::THOROUGH_TARGET_ERRORS,
                0.0,
                0x8000 + step as u64
            );
        }
    }

    #[test]
    #[ignore = "slow"]
    fn ber_awgn_qam_thorough() {
        let qam1024 = Qam1024::<f64>::QAM1024;
        let qam4096 = Qam4096::<f64>::QAM4096;
        for step in 16..=22 {
            let ebn0_db = step as f64 * 1.0;
            let ebn0_lin = theory::db_to_lin(ebn0_db);
            let p_1024 = theory::ber_sqqam_awgn(1024, ebn0_lin);
            if p_1024 <= 1e-2 {
                run_awgn_point!(
                    10,
                    &format!("1024-QAM {ebn0_db:.1} dB"),
                    &qam1024,
                    ebn0_db,
                    p_1024,
                    harness::THOROUGH_TARGET_ERRORS,
                    0.10,
                    0x9000 + step as u64
                );
            }
            let p_4096 = theory::ber_sqqam_awgn(4096, ebn0_lin);
            if p_4096 <= 1e-2 {
                run_awgn_point!(
                    12,
                    &format!("4096-QAM {ebn0_db:.1} dB"),
                    &qam4096,
                    ebn0_db,
                    p_4096,
                    harness::THOROUGH_TARGET_ERRORS,
                    0.10,
                    0xA000 + step as u64
                );
            }
        }
    }
}

// =============================================================================
// Module: rayleigh
// =============================================================================

mod rayleigh {
    use super::{harness, theory};
    use constella::bits::ChunkBitsExt;
    use constella::channel::{ChannelExt, EqualizeExt, FadingExt};
    use constella::constellation::{Bpsk, Psk8, Qam16, Qpsk};
    use constella::demodulation::DemodulateExt;
    use constella::modulation::ModulateExt;
    use rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    macro_rules! run_rayleigh_point {
        ($k:expr, $label:expr, $constellation:expr, $ebn0_db:expr, $p_theory:expr, $target_errors:expr, $rel_model_err:expr, $seed:expr) => {{
            let n_bytes = harness::payload_bytes_for($target_errors, $p_theory, $k);
            let payload = harness::random_payload(n_bytes, $seed ^ 0xBEEF_CAFE);
            let rng_fade = Xoshiro256PlusPlus::seed_from_u64($seed);
            let rng_awgn = Xoshiro256PlusPlus::seed_from_u64($seed ^ 0x5555_AAAA_3333_1111);

            let tx = payload.clone().into_iter().chunk_bits::<$k>();
            let rx = payload
                .into_iter()
                .modulate($constellation)
                .add_rayleigh_csi(rng_fade)
                .add_awgn_ebn0($constellation, $ebn0_db, rng_awgn)
                .equalize_zf()
                .demodulate_hard_symbols($constellation);

            let point = harness::count_symbol_bit_errors::<$k>(tx, rx);
            harness::assert_ber_consistent($label, point, $p_theory, 5.0, $rel_model_err);
            point
        }};
    }

    #[test]
    fn ber_rayleigh_bpsk_fast() {
        let c = Bpsk::<f64>::BPSK;
        let mut prev_ber = 1.0;
        for &ebn0_db in &[0.0, 5.0, 10.0, 15.0, 20.0] {
            let ebn0_lin = theory::db_to_lin(ebn0_db);
            let p_theory = theory::rayleigh_average(1.0, 1.0, ebn0_lin);
            let pt = run_rayleigh_point!(
                1,
                &format!("BPSK Rayleigh {ebn0_db:.1} dB"),
                &c,
                ebn0_db,
                p_theory,
                harness::FAST_TARGET_ERRORS,
                0.0,
                0xB000 + (ebn0_db as u64)
            );
            assert!(pt.ber() <= prev_ber + 0.01);
            prev_ber = pt.ber();
        }
    }

    #[test]
    fn ber_rayleigh_qpsk_fast() {
        let c = Qpsk::<f64>::QPSK;
        let mut prev_ber = 1.0;
        for &ebn0_db in &[0.0, 5.0, 10.0, 15.0, 20.0] {
            let ebn0_lin = theory::db_to_lin(ebn0_db);
            let p_theory = theory::rayleigh_average(1.0, 1.0, ebn0_lin);
            let pt = run_rayleigh_point!(
                2,
                &format!("QPSK Rayleigh {ebn0_db:.1} dB"),
                &c,
                ebn0_db,
                p_theory,
                harness::FAST_TARGET_ERRORS,
                0.0,
                0xC000 + (ebn0_db as u64)
            );
            assert!(pt.ber() <= prev_ber + 0.01);
            prev_ber = pt.ber();
        }
    }

    #[test]
    fn ber_rayleigh_8psk_fast() {
        let c = Psk8::<f64>::m_psk();
        let k = 3.0;
        let c_coeff = 2.0 / k;
        let a_coeff = k * (core::f64::consts::PI / 8.0).sin().powi(2);

        for &ebn0_db in &[8.0, 12.0, 16.0, 20.0] {
            let ebn0_lin = theory::db_to_lin(ebn0_db);
            let p_theory = theory::rayleigh_average(c_coeff, a_coeff, ebn0_lin);
            if p_theory > 1e-2 {
                continue;
            }
            run_rayleigh_point!(
                3,
                &format!("8-PSK Rayleigh {ebn0_db:.1} dB"),
                &c,
                ebn0_db,
                p_theory,
                harness::FAST_TARGET_ERRORS,
                0.10,
                0xD000 + (ebn0_db as u64)
            );
        }
    }

    #[test]
    fn ber_rayleigh_16qam_fast() {
        let c = Qam16::<f64>::QAM16;
        let c_coeff = 0.75;
        let a_coeff = 0.4;

        for &ebn0_db in &[10.0, 14.0, 18.0, 22.0] {
            let ebn0_lin = theory::db_to_lin(ebn0_db);
            let p_theory = theory::rayleigh_average(c_coeff, a_coeff, ebn0_lin);
            if p_theory > 1e-2 {
                continue;
            }
            run_rayleigh_point!(
                4,
                &format!("16-QAM Rayleigh {ebn0_db:.1} dB"),
                &c,
                ebn0_db,
                p_theory,
                harness::FAST_TARGET_ERRORS,
                0.10,
                0xE000 + (ebn0_db as u64)
            );
        }
    }

    #[test]
    #[ignore = "slow"]
    fn ber_rayleigh_thorough() {
        let bpsk = Bpsk::<f64>::BPSK;
        for step in 0..=25 {
            let ebn0_db = step as f64 * 1.0;
            let ebn0_lin = theory::db_to_lin(ebn0_db);
            let p_theory = theory::rayleigh_average(1.0, 1.0, ebn0_lin);
            run_rayleigh_point!(
                1,
                &format!("BPSK Rayleigh Thorough {ebn0_db:.1} dB"),
                &bpsk,
                ebn0_db,
                p_theory,
                harness::THOROUGH_TARGET_ERRORS,
                0.0,
                0xF000 + step as u64
            );
        }
    }
}

// =============================================================================
// Module: rician
// =============================================================================

mod rician {
    use super::{harness, theory};
    use constella::bits::ChunkBitsExt;
    use constella::channel::{ChannelExt, EqualizeExt, FadingExt};
    use constella::constellation::Bpsk;
    use constella::demodulation::DemodulateExt;
    use constella::modulation::ModulateExt;
    use rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn rician_bracketed_by_rayleigh_and_awgn() {
        let bpsk = Bpsk::<f64>::BPSK;
        let ebn0_db = 10.0;
        let ebn0_lin = theory::db_to_lin(ebn0_db);

        let ber_awgn = theory::ber_bpsk_awgn(ebn0_lin);
        let ber_rayleigh = theory::rayleigh_average(1.0, 1.0, ebn0_lin);

        let k_dbs = [0.0, 3.0, 10.0, 30.0];
        let mut prev_rician_ber = ber_rayleigh + 0.01;

        for &k_db in &k_dbs {
            let n_bytes = 40_000;
            let payload = harness::random_payload(n_bytes, 0x1111 ^ (k_db as u64));
            let rng_fade = Xoshiro256PlusPlus::seed_from_u64(0x2222 + (k_db as u64));
            let rng_awgn = Xoshiro256PlusPlus::seed_from_u64(0x3333 + (k_db as u64));

            let tx = payload.clone().into_iter().chunk_bits::<1>();
            let rx = payload
                .into_iter()
                .modulate(&bpsk)
                .add_rician_csi_k_db(k_db, 0.0, rng_fade)
                .add_awgn_ebn0(&bpsk, ebn0_db, rng_awgn)
                .equalize_zf()
                .demodulate_hard_symbols(&bpsk);

            let pt = harness::count_symbol_bit_errors::<1>(tx, rx);
            let ber = pt.ber();

            assert!(
                ber >= ber_awgn - 0.005 && ber <= ber_rayleigh + 0.005,
                "Rician K={k_db} dB BER {ber} was not between AWGN ({ber_awgn}) and Rayleigh ({ber_rayleigh})"
            );
            assert!(
                ber <= prev_rician_ber + 0.002,
                "Rician BER not strictly decreasing with K: at {k_db} dB got {ber}, prev {prev_rician_ber}"
            );
            prev_rician_ber = ber;
        }
    }

    #[test]
    fn rician_k_zero_matches_rayleigh() {
        let bpsk = Bpsk::<f64>::BPSK;
        let ebn0_db = 10.0;
        let ebn0_lin = theory::db_to_lin(ebn0_db);
        let p_theory = theory::rayleigh_average(1.0, 1.0, ebn0_lin);

        let n_bytes = 50_000;
        let payload = harness::random_payload(n_bytes, 0x4444);
        let rng_fade = Xoshiro256PlusPlus::seed_from_u64(0x5555);
        let rng_awgn = Xoshiro256PlusPlus::seed_from_u64(0x6666);

        let tx = payload.clone().into_iter().chunk_bits::<1>();
        let rx = payload
            .into_iter()
            .modulate(&bpsk)
            .add_rician_csi_k_db(-60.0, 0.0, rng_fade)
            .add_awgn_ebn0(&bpsk, ebn0_db, rng_awgn)
            .equalize_zf()
            .demodulate_hard_symbols(&bpsk);

        let pt = harness::count_symbol_bit_errors::<1>(tx, rx);
        harness::assert_ber_consistent("Rician K=-60dB vs Rayleigh", pt, p_theory, 5.0, 0.0);
    }

    #[test]
    fn rician_high_k_approaches_awgn() {
        let bpsk = Bpsk::<f64>::BPSK;
        let ebn0_db = 7.0;
        let ebn0_lin = theory::db_to_lin(ebn0_db);
        let p_awgn = theory::ber_bpsk_awgn(ebn0_lin);

        let n_bytes = 100_000;
        let payload = harness::random_payload(n_bytes, 0x7777);
        let rng_fade = Xoshiro256PlusPlus::seed_from_u64(0x8888);
        let rng_awgn = Xoshiro256PlusPlus::seed_from_u64(0x9999);

        let tx = payload.clone().into_iter().chunk_bits::<1>();
        let rx = payload
            .into_iter()
            .modulate(&bpsk)
            .add_rician_csi_k_db(30.0, 0.0, rng_fade)
            .add_awgn_ebn0(&bpsk, ebn0_db, rng_awgn)
            .equalize_zf()
            .demodulate_hard_symbols(&bpsk);

        let pt = harness::count_symbol_bit_errors::<1>(tx, rx);
        assert!(
            (pt.ber() - p_awgn).abs() / p_awgn < 0.15,
            "Rician K=30dB BER {} did not match AWGN {p_awgn}",
            pt.ber()
        );
    }
}

// =============================================================================
// Module: phase
// =============================================================================

// =============================================================================
// Module: phase
// =============================================================================

mod phase {
    use super::{harness, theory};
    use constella::bits::ChunkBitsExt;
    use constella::channel::ChannelExt;
    use constella::constellation::{Bpsk, Qpsk};
    use constella::demodulation::DemodulateExt;
    use constella::modulation::ModulateExt;
    use rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn phase_static_offset_bpsk() {
        let bpsk = Bpsk::<f64>::BPSK;
        let ebn0_db = 6.0;
        let ebn0_lin = theory::db_to_lin(ebn0_db);

        let angles = [
            0.0,
            core::f64::consts::PI / 16.0,
            core::f64::consts::PI / 8.0,
            core::f64::consts::PI / 6.0,
        ];

        for (i, &theta) in angles.iter().enumerate() {
            let p_theory = theory::ber_bpsk_phase(ebn0_lin, theta);
            let n_bytes = harness::payload_bytes_for(harness::FAST_TARGET_ERRORS, p_theory, 1);
            let payload = harness::random_payload(n_bytes, 0x1010 + i as u64);
            let rng_awgn = Xoshiro256PlusPlus::seed_from_u64(0x2020 + i as u64);

            let tx = payload.clone().into_iter().chunk_bits::<1>();
            let rx = payload
                .into_iter()
                .modulate(&bpsk)
                .add_phase_offset(theta)
                .add_awgn_ebn0(&bpsk, ebn0_db, rng_awgn)
                .demodulate_hard_symbols(&bpsk);

            let pt = harness::count_symbol_bit_errors::<1>(tx, rx);
            harness::assert_ber_consistent(
                &format!("BPSK static theta={theta:.3}"),
                pt,
                p_theory,
                5.0,
                0.0,
            );
        }
    }

    #[test]
    fn phase_static_offset_qpsk() {
        let qpsk = Qpsk::<f64>::QPSK;
        let ebn0_db = 6.0;
        let ebn0_lin = theory::db_to_lin(ebn0_db);

        let angles = [
            0.0,
            core::f64::consts::PI / 16.0,
            core::f64::consts::PI / 8.0,
            core::f64::consts::PI / 6.0,
        ];

        for (i, &theta) in angles.iter().enumerate() {
            let p_theory = theory::ber_qpsk_phase(ebn0_lin, theta);
            let n_bytes = harness::payload_bytes_for(harness::FAST_TARGET_ERRORS, p_theory, 2);
            let payload = harness::random_payload(n_bytes, 0x3030 + i as u64);
            let rng_awgn = Xoshiro256PlusPlus::seed_from_u64(0x4040 + i as u64);

            let tx = payload.clone().into_iter().chunk_bits::<2>();
            let rx = payload
                .into_iter()
                .modulate(&qpsk)
                .add_phase_offset(theta)
                .add_awgn_ebn0(&qpsk, ebn0_db, rng_awgn)
                .demodulate_hard_symbols(&qpsk);

            let pt = harness::count_symbol_bit_errors::<2>(tx, rx);
            harness::assert_ber_consistent(
                &format!("QPSK static theta={theta:.3}"),
                pt,
                p_theory,
                5.0,
                0.0,
            );
        }
    }

    #[test]
    fn phase_catastrophic_offsets() {
        let bpsk = Bpsk::<f64>::BPSK;
        let qpsk = Qpsk::<f64>::QPSK;
        let payload = harness::random_payload(20_000, 0x5050);
        let ebn0_db = 10.0;

        // BPSK at pi/2: signal projects to 0 on I-axis (Q(0) = 0.5) -> BER ≈ 0.5
        let tx_b = payload.clone().into_iter().chunk_bits::<1>();
        let rx_b = payload
            .clone()
            .into_iter()
            .modulate(&bpsk)
            .add_phase_offset(core::f64::consts::PI / 2.0)
            .add_awgn_ebn0(&bpsk, ebn0_db, Xoshiro256PlusPlus::seed_from_u64(0x5151))
            .demodulate_hard_symbols(&bpsk);
        let pt_b = harness::count_symbol_bit_errors::<1>(tx_b, rx_b);
        assert!(
            (pt_b.ber() - 0.5).abs() < 0.02,
            "BPSK at pi/2 should have BER ≈ 0.5, got {}",
            pt_b.ber()
        );

        // QPSK at pi/4: one bit projects to boundary (Q(0) = 0.5), other bit clean (Q ≈ 0) -> BER ≈ 0.25
        let tx_q_pi4 = payload.clone().into_iter().chunk_bits::<2>();
        let rx_q_pi4 = payload
            .clone()
            .into_iter()
            .modulate(&qpsk)
            .add_phase_offset(core::f64::consts::PI / 4.0)
            .add_awgn_ebn0(&qpsk, ebn0_db, Xoshiro256PlusPlus::seed_from_u64(0x5252))
            .demodulate_hard_symbols(&qpsk);
        let pt_q_pi4 = harness::count_symbol_bit_errors::<2>(tx_q_pi4, rx_q_pi4);
        assert!(
            (pt_q_pi4.ber() - 0.25).abs() < 0.02,
            "QPSK at pi/4 should have BER ≈ 0.25, got {}",
            pt_q_pi4.ber()
        );

        // QPSK at pi/2: rotates into adjacent Gray-coded quadrants -> exactly 1 bit error / 2 bits -> BER ≈ 0.5
        let tx_q_pi2 = payload.clone().into_iter().chunk_bits::<2>();
        let rx_q_pi2 = payload
            .into_iter()
            .modulate(&qpsk)
            .add_phase_offset(core::f64::consts::PI / 2.0)
            .add_awgn_ebn0(&qpsk, ebn0_db, Xoshiro256PlusPlus::seed_from_u64(0x5353))
            .demodulate_hard_symbols(&qpsk);
        let pt_q_pi2 = harness::count_symbol_bit_errors::<2>(tx_q_pi2, rx_q_pi2);
        assert!(
            (pt_q_pi2.ber() - 0.5).abs() < 0.02,
            "QPSK at pi/2 should have BER ≈ 0.5, got {}",
            pt_q_pi2.ber()
        );
    }

    #[test]
    fn phase_cfo_qpsk_numeric_average() {
        let qpsk = Qpsk::<f64>::QPSK;
        let ebn0_db = 6.0;
        let ebn0_lin = theory::db_to_lin(ebn0_db);
        let delta_omega = 0.0002;

        let n_bytes = 20_000;
        let n_symbols = n_bytes * 4;
        let p_theory: f64 = (0..n_symbols)
            .map(|k| theory::ber_qpsk_phase(ebn0_lin, k as f64 * delta_omega))
            .sum::<f64>()
            / (n_symbols as f64);

        let payload = harness::random_payload(n_bytes, 0x6060);
        let rng_awgn = Xoshiro256PlusPlus::seed_from_u64(0x7070);

        let tx = payload.clone().into_iter().chunk_bits::<2>();
        let rx = payload
            .into_iter()
            .modulate(&qpsk)
            .add_cfo(delta_omega)
            .add_awgn_ebn0(&qpsk, ebn0_db, rng_awgn)
            .demodulate_hard_symbols(&qpsk);

        let pt = harness::count_symbol_bit_errors::<2>(tx, rx);
        harness::assert_ber_consistent("QPSK CFO averaged", pt, p_theory, 5.0, 0.05);
    }

    #[test]
    fn phase_wiener_noise_properties() {
        let qpsk = Qpsk::<f64>::QPSK;
        let ebn0_db = 7.0;

        // 1. sigma = 0 is bit-identical to no phase distortion
        let payload = harness::random_payload(10_000, 0x8080);
        let rx_no_phase: Vec<usize> = payload
            .clone()
            .into_iter()
            .modulate(&qpsk)
            .add_awgn_ebn0(&qpsk, ebn0_db, Xoshiro256PlusPlus::seed_from_u64(0x999))
            .demodulate_hard_symbols(&qpsk)
            .collect();

        let rx_zero_wiener: Vec<usize> = payload
            .clone()
            .into_iter()
            .modulate(&qpsk)
            .add_phase_noise(0.0, Xoshiro256PlusPlus::seed_from_u64(0xAAA))
            .add_awgn_ebn0(&qpsk, ebn0_db, Xoshiro256PlusPlus::seed_from_u64(0x999))
            .demodulate_hard_symbols(&qpsk)
            .collect();

        assert_eq!(rx_no_phase, rx_zero_wiener);

        // 2. Monotonic degradation over calibrated burst lengths (500 symbols = 125 bytes)
        let n_bursts = 160;
        let burst_bytes = 125;
        let mut prev_avg_ber = 0.0;

        for &sigma in &[0.0, 0.002, 0.006, 0.015] {
            let mut total_errors = 0u64;
            let mut total_bits = 0u64;

            for b in 0..n_bursts {
                let burst_payload = harness::random_payload(burst_bytes, 0xB000 + b as u64);
                let tx = burst_payload.clone().into_iter().chunk_bits::<2>();
                let rx = burst_payload
                    .into_iter()
                    .modulate(&qpsk)
                    .add_phase_noise(sigma, Xoshiro256PlusPlus::seed_from_u64(0xC000 + b as u64))
                    .add_awgn_ebn0(
                        &qpsk,
                        ebn0_db,
                        Xoshiro256PlusPlus::seed_from_u64(0xD000 + b as u64),
                    )
                    .demodulate_hard_symbols(&qpsk);

                let pt = harness::count_symbol_bit_errors::<2>(tx, rx);
                total_errors += pt.errors;
                total_bits += pt.bits;
            }

            let avg_ber = total_errors as f64 / total_bits as f64;
            assert!(
                avg_ber >= prev_avg_ber - 0.001,
                "Wiener noise not monotonic with sigma: at {sigma} got {avg_ber}, prev {prev_avg_ber}"
            );
            prev_avg_ber = avg_ber;
        }
    }
}

// =============================================================================
// Module: equalizer
// =============================================================================

mod equalizer {
    use super::{harness, theory};
    use constella::bits::ChunkBitsExt;
    use constella::channel::{ChannelExt, EqualizeExt, FadingExt};
    use constella::constellation::{Bpsk, Psk8, Qam16, Qpsk};
    use constella::demodulation::DemodulateExt;
    use constella::modulation::ModulateExt;
    use rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn zf_and_mmse_are_identical_for_psk() {
        let bpsk = Bpsk::<f64>::BPSK;
        let qpsk = Qpsk::<f64>::QPSK;
        let psk8 = Psk8::<f64>::m_psk();

        let n_bytes = 10_000;
        let payload = harness::random_payload(n_bytes, 0xA1B2);
        let ebn0_db = 8.0;
        let ebn0_lin = theory::db_to_lin(ebn0_db);

        // BPSK
        let noise_var_bpsk = 1.0 / ebn0_lin;
        let symbols_b: Vec<_> = payload
            .clone()
            .into_iter()
            .modulate(&bpsk)
            .add_rayleigh_csi(Xoshiro256PlusPlus::seed_from_u64(101))
            .add_awgn_ebn0(&bpsk, ebn0_db, Xoshiro256PlusPlus::seed_from_u64(202))
            .collect();

        let zf_b: Vec<usize> = symbols_b
            .clone()
            .into_iter()
            .equalize_zf()
            .demodulate_hard_symbols(&bpsk)
            .collect();
        let mmse_b: Vec<usize> = symbols_b
            .into_iter()
            .equalize_mmse(noise_var_bpsk)
            .demodulate_hard_symbols(&bpsk)
            .collect();
        assert_eq!(zf_b, mmse_b, "ZF and MMSE must be bit-identical for BPSK");

        // QPSK
        let noise_var_qpsk = 1.0 / (2.0 * ebn0_lin);
        let symbols_q: Vec<_> = payload
            .clone()
            .into_iter()
            .modulate(&qpsk)
            .add_rayleigh_csi(Xoshiro256PlusPlus::seed_from_u64(303))
            .add_awgn_ebn0(&qpsk, ebn0_db, Xoshiro256PlusPlus::seed_from_u64(404))
            .collect();

        let zf_q: Vec<usize> = symbols_q
            .clone()
            .into_iter()
            .equalize_zf()
            .demodulate_hard_symbols(&qpsk)
            .collect();
        let mmse_q: Vec<usize> = symbols_q
            .into_iter()
            .equalize_mmse(noise_var_qpsk)
            .demodulate_hard_symbols(&qpsk)
            .collect();
        assert_eq!(zf_q, mmse_q, "ZF and MMSE must be bit-identical for QPSK");

        // 8-PSK
        let noise_var_psk8 = 1.0 / (3.0 * ebn0_lin);
        let symbols_p8: Vec<_> = payload
            .into_iter()
            .modulate(&psk8)
            .add_rayleigh_csi(Xoshiro256PlusPlus::seed_from_u64(505))
            .add_awgn_ebn0(&psk8, ebn0_db, Xoshiro256PlusPlus::seed_from_u64(606))
            .collect();

        let zf_p8: Vec<usize> = symbols_p8
            .clone()
            .into_iter()
            .equalize_zf()
            .demodulate_hard_symbols(&psk8)
            .collect();
        let mmse_p8: Vec<usize> = symbols_p8
            .into_iter()
            .equalize_mmse(noise_var_psk8)
            .demodulate_hard_symbols(&psk8)
            .collect();
        assert_eq!(
            zf_p8, mmse_p8,
            "ZF and MMSE must be bit-identical for 8-PSK"
        );
    }

    #[test]
    fn zf_and_mmse_differ_for_qam() {
        let qam16 = Qam16::<f64>::QAM16;
        let ebn0_db = 10.0;
        let ebn0_lin = theory::db_to_lin(ebn0_db);
        let noise_var = 1.0 / (4.0 * ebn0_lin);

        let n_bytes = 40_000;
        let payload = harness::random_payload(n_bytes, 0x8888);

        let symbols: Vec<_> = payload
            .clone()
            .into_iter()
            .modulate(&qam16)
            .add_rayleigh_csi(Xoshiro256PlusPlus::seed_from_u64(707))
            .add_awgn_ebn0(&qam16, ebn0_db, Xoshiro256PlusPlus::seed_from_u64(808))
            .collect();

        let tx_zf = payload.clone().into_iter().chunk_bits::<4>();
        let tx_mmse = payload.into_iter().chunk_bits::<4>();

        let rx_zf = symbols
            .clone()
            .into_iter()
            .equalize_zf()
            .demodulate_hard_symbols(&qam16);

        let rx_mmse = symbols
            .into_iter()
            .equalize_mmse(noise_var)
            .demodulate_hard_symbols(&qam16);

        let pt_zf = harness::count_symbol_bit_errors::<4>(tx_zf, rx_zf);
        let pt_mmse = harness::count_symbol_bit_errors::<4>(tx_mmse, rx_mmse);

        // Amplitude information causes ZF and MMSE decisions to diverge on QAM
        assert_ne!(
            pt_zf.errors, pt_mmse.errors,
            "ZF and MMSE decisions must differ for 16-QAM"
        );
    }
}
// =============================================================================
// Module: differential
// =============================================================================

mod differential {
    use super::{harness, theory};
    use constella::bits::ChunkBitsExt;
    use constella::channel::ChannelExt;
    use constella::constellation::{Bpsk, Qpsk};
    use constella::demodulation::DemodulateExt;
    use constella::differential::DifferentialExt;
    use constella::modulation::ModulateExt;
    use num_complex::Complex;
    use rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn ber_dbpsk_fast() {
        let bpsk = Bpsk::<f64>::BPSK;
        let pilot = Complex::new(1.0f64, 0.0);
        let mut prev_ber = 1.0;

        for step in 0..=8 {
            let ebn0_db = step as f64 * 1.0;
            let ebn0_lin = theory::db_to_lin(ebn0_db);
            let p_theory = theory::ber_dbpsk_awgn(ebn0_lin);

            let n_bytes = harness::payload_bytes_for(harness::FAST_TARGET_ERRORS, p_theory, 1);
            let payload = harness::random_payload(n_bytes, 0xD0D0 + step as u64);
            let rng = Xoshiro256PlusPlus::seed_from_u64(0xE0E0 + step as u64);

            let tx = payload.clone().into_iter().chunk_bits::<1>();
            let rx = core::iter::once(pilot)
                .chain(payload.into_iter().modulate(&bpsk).differential_encode())
                .add_awgn_ebn0(&bpsk, ebn0_db, rng)
                .differential_decode()
                .skip(1)
                .demodulate_hard_symbols(&bpsk);

            let pt = harness::count_symbol_bit_errors::<1>(tx, rx);
            harness::assert_ber_consistent(
                &format!("DBPSK AWGN {ebn0_db:.1} dB"),
                pt,
                p_theory,
                5.0,
                0.0,
            );
            assert!(pt.ber() <= prev_ber + 0.01);
            prev_ber = pt.ber();
        }
    }

    #[test]
    fn differential_dqpsk_d8psk_bracketed() {
        let qpsk = Qpsk::<f64>::QPSK;
        let pilot = Complex::new(1.0f64, 0.0);
        let ebn0_db = 8.0;

        let ber_coh_8db = theory::ber_qpsk_awgn(theory::db_to_lin(8.0));
        let ber_coh_4db = theory::ber_qpsk_awgn(theory::db_to_lin(4.0));

        let payload = harness::random_payload(30_000, 0xF1F1);
        let rng = Xoshiro256PlusPlus::seed_from_u64(0xF2F2);

        let tx = payload.clone().into_iter().chunk_bits::<2>();
        let rx = core::iter::once(pilot)
            .chain(payload.into_iter().modulate(&qpsk).differential_encode())
            .add_awgn_ebn0(&qpsk, ebn0_db, rng)
            .differential_decode()
            .skip(1)
            .demodulate_hard_symbols(&qpsk);

        let pt = harness::count_symbol_bit_errors::<2>(tx, rx);
        let ber_diff = pt.ber();

        assert!(
            ber_diff > ber_coh_8db,
            "DQPSK ({ber_diff}) must be worse than coherent QPSK at same SNR ({ber_coh_8db})"
        );
        assert!(
            ber_diff < ber_coh_4db,
            "DQPSK ({ber_diff}) must be better than coherent QPSK at SNR - 4dB ({ber_coh_4db})"
        );
    }
}
