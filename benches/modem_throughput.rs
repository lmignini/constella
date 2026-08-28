use constella::channel::ChannelExt;
use constella::prelude::*;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use num_complex::Complex;
use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::time::Duration;

const PAYLOAD_SIZE: usize = 1024 * 1024; // 1 MB payload

/// 1. Constellation Modulation & Hard Demodulation Scalability
fn bench_constellations(c: &mut Criterion) {
    let mut group = c.benchmark_group("modulation_and_demodulation");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(20));
    group.warm_up_time(Duration::from_secs(5));
    let payload = vec![0x5A; PAYLOAD_SIZE];
    group.throughput(Throughput::Bytes(PAYLOAD_SIZE as u64));

    // A. BPSK (K = 1, Direct Codec)
    let bpsk = Bpsk::<f32>::BPSK;
    let mut bpsk_sym_buf = Vec::with_capacity(PAYLOAD_SIZE * 8);
    group.bench_function("bpsk/modulate", |b| {
        b.iter(|| {
            bpsk_sym_buf.clear();
            bpsk_sym_buf.extend(payload.iter().copied().modulate(&bpsk));
            std::hint::black_box(&bpsk_sym_buf);
        })
    });

    let bpsk_syms: Vec<Complex<f32>> = payload.iter().copied().modulate(&bpsk).collect();
    let mut bpsk_byte_buf = Vec::with_capacity(PAYLOAD_SIZE);
    group.bench_function("bpsk/demodulate_hard", |b| {
        b.iter(|| {
            bpsk_byte_buf.clear();
            bpsk_byte_buf.extend(bpsk_syms.iter().copied().demodulate_hard(&bpsk));
            std::hint::black_box(&bpsk_byte_buf);
        })
    });

    // B. QPSK (K = 2, Direct Codec)
    let qpsk = Qpsk::<f32>::QPSK;
    let mut qpsk_sym_buf = Vec::with_capacity(PAYLOAD_SIZE * 4);
    group.bench_function("qpsk/modulate", |b| {
        b.iter(|| {
            qpsk_sym_buf.clear();
            qpsk_sym_buf.extend(payload.iter().copied().modulate(&qpsk));
            std::hint::black_box(&qpsk_sym_buf);
        })
    });

    let qpsk_syms: Vec<Complex<f32>> = payload.iter().copied().modulate(&qpsk).collect();
    let mut qpsk_byte_buf = Vec::with_capacity(PAYLOAD_SIZE);
    group.bench_function("qpsk/demodulate_hard", |b| {
        b.iter(|| {
            qpsk_byte_buf.clear();
            qpsk_byte_buf.extend(qpsk_syms.iter().copied().demodulate_hard(&qpsk));
            std::hint::black_box(&qpsk_byte_buf);
        })
    });

    // C. 16-QAM (K = 4, Direct Codec)
    let qam16 = Qam16::<f32>::QAM16;
    let mut qam16_sym_buf = Vec::with_capacity(PAYLOAD_SIZE * 2);
    group.bench_function("qam16/modulate", |b| {
        b.iter(|| {
            qam16_sym_buf.clear();
            qam16_sym_buf.extend(payload.iter().copied().modulate(&qam16));
            std::hint::black_box(&qam16_sym_buf);
        })
    });

    let qam16_syms: Vec<Complex<f32>> = payload.iter().copied().modulate(&qam16).collect();
    let mut qam16_byte_buf = Vec::with_capacity(PAYLOAD_SIZE);
    group.bench_function("qam16/demodulate_square_o1", |b| {
        b.iter(|| {
            qam16_byte_buf.clear();
            qam16_byte_buf.extend(qam16_syms.iter().copied().demodulate_hard(&qam16));
            std::hint::black_box(&qam16_byte_buf);
        })
    });

    // D. 64-QAM (K = 6, Bit-Buffer Streaming Path)
    let qam64 = Qam64::<f32>::QAM64;
    let mut qam64_sym_buf = Vec::with_capacity((PAYLOAD_SIZE * 8).div_ceil(6));
    group.bench_function("qam64/modulate_bit_buffered", |b| {
        b.iter(|| {
            qam64_sym_buf.clear();
            qam64_sym_buf.extend(payload.iter().copied().modulate(&qam64));
            std::hint::black_box(&qam64_sym_buf);
        })
    });

    let qam64_syms: Vec<Complex<f32>> = payload.iter().copied().modulate(&qam64).collect();
    let mut qam64_byte_buf = Vec::with_capacity(PAYLOAD_SIZE);
    group.bench_function("qam64/demodulate_square_o1", |b| {
        b.iter(|| {
            qam64_byte_buf.clear();
            qam64_byte_buf.extend(qam64_syms.iter().copied().demodulate_hard(&qam64));
            std::hint::black_box(&qam64_byte_buf);
        })
    });

    // E. 256-QAM (K = 8, Direct 1:1 Byte Codec)
    let qam256 = Qam256::<f32>::QAM256;
    let mut qam256_sym_buf = Vec::with_capacity(PAYLOAD_SIZE);
    group.bench_function("qam256/modulate_1to1", |b| {
        b.iter(|| {
            qam256_sym_buf.clear();
            qam256_sym_buf.extend(payload.iter().copied().modulate(&qam256));
            std::hint::black_box(&qam256_sym_buf);
        })
    });

    let qam256_syms: Vec<Complex<f32>> = payload.iter().copied().modulate(&qam256).collect();
    let mut qam256_byte_buf = Vec::with_capacity(PAYLOAD_SIZE);
    group.bench_function("qam256/demodulate_square_o1", |b| {
        b.iter(|| {
            qam256_byte_buf.clear();
            qam256_byte_buf.extend(qam256_syms.iter().copied().demodulate_hard(&qam256));
            std::hint::black_box(&qam256_byte_buf);
        })
    });

    group.finish();
}

/// 2. Channel Impairments Overhead (AWGN, CFO, Wiener Noise)
fn bench_channel_impairments(c: &mut Criterion) {
    let mut group = c.benchmark_group("channels");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(20));
    group.warm_up_time(Duration::from_secs(5));
    let payload = vec![0xA5; PAYLOAD_SIZE];
    let qam16 = Qam16::<f32>::QAM16;
    let symbols: Vec<Complex<f32>> = payload.iter().copied().modulate(&qam16).collect();
    let sym_len = symbols.len();

    group.throughput(Throughput::Elements(sym_len as u64));

    // A. AWGN Injection
    let mut awgn_buf = Vec::with_capacity(sym_len);
    group.bench_function("awgn_snr_injection", |b| {
        b.iter(|| {
            let rng = Xoshiro256PlusPlus::seed_from_u64(42);
            awgn_buf.clear();
            awgn_buf.extend(symbols.iter().copied().add_awgn_snr(&qam16, 20.0, rng));
            std::hint::black_box(&awgn_buf);
        })
    });

    // B. Static Phase Offset (Trig multiply only)
    let mut static_phase_buf = Vec::with_capacity(sym_len);
    group.bench_function("static_phase_rotation", |b| {
        b.iter(|| {
            static_phase_buf.clear();
            static_phase_buf.extend(
                symbols
                    .iter()
                    .copied()
                    .add_phase_offset(core::f32::consts::FRAC_PI_4),
            );
            std::hint::black_box(&static_phase_buf);
        })
    });

    // C. CFO Streaming (Frequency rotation ramp)
    let mut cfo_buf = Vec::with_capacity(sym_len);
    group.bench_function("cfo_stream", |b| {
        b.iter(|| {
            cfo_buf.clear();
            cfo_buf.extend(symbols.iter().copied().add_cfo_hz(1000.0, 100_000.0));
            std::hint::black_box(&cfo_buf);
        })
    });

    // D. Wiener Phase Noise (Gaussian step PRNG + rotation)
    let mut wiener_buf = Vec::with_capacity(sym_len);
    group.bench_function("wiener_phase_noise", |b| {
        b.iter(|| {
            let rng = Xoshiro256PlusPlus::seed_from_u64(42);
            wiener_buf.clear();
            wiener_buf.extend(symbols.iter().copied().add_phase_noise(0.01, rng));
            std::hint::black_box(&wiener_buf);
        })
    });

    group.finish();
}

/// 3. End-to-End Transceiver Pipelines
fn bench_end_to_end_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end_pipeline");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(20));
    group.warm_up_time(Duration::from_secs(5));
    let payload = vec![0x3C; PAYLOAD_SIZE];
    let qam16 = Qam16::<f32>::QAM16;
    group.throughput(Throughput::Bytes(PAYLOAD_SIZE as u64));

    // Clean pipeline: Modulate -> Demodulate
    let mut clean_buf = Vec::with_capacity(PAYLOAD_SIZE);
    group.bench_function("qam16_clean_loopback", |b| {
        b.iter(|| {
            clean_buf.clear();
            clean_buf.extend(
                payload
                    .iter()
                    .copied()
                    .modulate(&qam16)
                    .demodulate_hard(&qam16),
            );
            std::hint::black_box(&clean_buf);
        })
    });

    // Impaired pipeline: Modulate -> Phase -> AWGN -> Demodulate
    let mut impaired_buf = Vec::with_capacity(PAYLOAD_SIZE);
    group.bench_function("qam16_impaired_channel_chain", |b| {
        b.iter(|| {
            let rng_phase = Xoshiro256PlusPlus::seed_from_u64(1);
            let rng_awgn = Xoshiro256PlusPlus::seed_from_u64(2);
            impaired_buf.clear();
            impaired_buf.extend(
                payload
                    .iter()
                    .copied()
                    .modulate(&qam16)
                    .add_cfo_hz(50.0, 1_000_000.0)
                    .add_phase_noise(0.001, rng_phase)
                    .add_awgn_snr(&qam16, 25.0, rng_awgn)
                    .demodulate_hard(&qam16),
            );
            std::hint::black_box(&impaired_buf);
        })
    });

    group.finish();
}

/// 4. Soft Demodulation (LLR Extraction)
fn bench_soft_demodulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("soft_demodulation");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(20));
    group.warm_up_time(Duration::from_secs(5));
    let payload = vec![0x55; 256 * 1024]; // 256 KB
    let qpsk = Qpsk::<f32>::QPSK;
    let symbols: Vec<Complex<f32>> = payload.iter().copied().modulate(&qpsk).collect();
    let sym_len = symbols.len();
    let noise_var = 0.1f32;

    group.throughput(Throughput::Elements(sym_len as u64));

    // Symbol LLR vectors: [f32; 2]
    let mut symbol_llrs_buf = Vec::with_capacity(sym_len);
    group.bench_function("qpsk_soft_symbols", |b| {
        b.iter(|| {
            symbol_llrs_buf.clear();
            symbol_llrs_buf.extend(symbols.iter().copied().demodulate_soft(&qpsk, noise_var));
            std::hint::black_box(&symbol_llrs_buf);
        })
    });

    // Flattened bitstream LLRs: f32
    let mut bit_llrs_buf = Vec::with_capacity(sym_len * 2);
    group.bench_function("qpsk_soft_bits_flattened", |b| {
        b.iter(|| {
            bit_llrs_buf.clear();
            bit_llrs_buf.extend(
                symbols
                    .iter()
                    .copied()
                    .demodulate_soft_bits(&qpsk, noise_var),
            );
            std::hint::black_box(&bit_llrs_buf);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_constellations,
    bench_channel_impairments,
    bench_end_to_end_pipeline,
    bench_soft_demodulation
);
criterion_main!(benches);
