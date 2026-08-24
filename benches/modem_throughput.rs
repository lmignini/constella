use constella::channel::ChannelExt;

use constella::prelude::*;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use num_complex::Complex;
use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

const PAYLOAD_SIZE: usize = 1024 * 1024; // 1 MB payload

/// 1. Constellation Modulation & Hard Demodulation Scalability
fn bench_constellations(c: &mut Criterion) {
    let mut group = c.benchmark_group("modulation_and_demodulation");
    let payload = vec![0x5A; PAYLOAD_SIZE];
    group.throughput(Throughput::Bytes(PAYLOAD_SIZE as u64));

    // A. BPSK (K = 1, Direct Codec)
    let bpsk = Bpsk::<f32>::BPSK;
    group.bench_function("bpsk/modulate", |b| {
        b.iter(|| {
            let symbols: Vec<Complex<f32>> = payload.iter().copied().modulate(bpsk).collect();
            std::hint::black_box(symbols);
        })
    });

    let bpsk_syms: Vec<Complex<f32>> = payload.iter().copied().modulate(bpsk).collect();
    group.bench_function("bpsk/demodulate_hard", |b| {
        b.iter(|| {
            let recovered: Vec<u8> = bpsk_syms.iter().copied().demodulate_hard(bpsk).collect();
            std::hint::black_box(recovered);
        })
    });

    // B. QPSK (K = 2, Direct Codec)
    let qpsk = Qpsk::<f32>::QPSK;
    group.bench_function("qpsk/modulate", |b| {
        b.iter(|| {
            let symbols: Vec<Complex<f32>> = payload.iter().copied().modulate(qpsk).collect();
            std::hint::black_box(symbols);
        })
    });

    let qpsk_syms: Vec<Complex<f32>> = payload.iter().copied().modulate(qpsk).collect();
    group.bench_function("qpsk/demodulate_hard", |b| {
        b.iter(|| {
            let recovered: Vec<u8> = qpsk_syms.iter().copied().demodulate_hard(qpsk).collect();
            std::hint::black_box(recovered);
        })
    });

    // C. 16-QAM (K = 4, Direct Codec)
    let qam16 = Qam16::<f32>::QAM16;
    group.bench_function("qam16/modulate", |b| {
        b.iter(|| {
            let symbols: Vec<Complex<f32>> = payload.iter().copied().modulate(qam16).collect();
            std::hint::black_box(symbols);
        })
    });

    let qam16_syms: Vec<Complex<f32>> = payload.iter().copied().modulate(qam16).collect();
    group.bench_function("qam16/demodulate_square_o1", |b| {
        b.iter(|| {
            let recovered: Vec<u8> = qam16_syms.iter().copied().demodulate_hard(qam16).collect();
            std::hint::black_box(recovered);
        })
    });

    // D. 64-QAM (K = 6, Bit-Buffer Streaming Path)
    let qam64 = Qam64::<f32>::QAM64;
    group.bench_function("qam64/modulate_bit_buffered", |b| {
        b.iter(|| {
            let symbols: Vec<Complex<f32>> = payload.iter().copied().modulate(qam64).collect();
            std::hint::black_box(symbols);
        })
    });

    let qam64_syms: Vec<Complex<f32>> = payload.iter().copied().modulate(qam64).collect();
    group.bench_function("qam64/demodulate_square_o1", |b| {
        b.iter(|| {
            let recovered: Vec<u8> = qam64_syms.iter().copied().demodulate_hard(qam64).collect();
            std::hint::black_box(recovered);
        })
    });

    // E. 256-QAM (K = 8, Direct 1:1 Byte Codec)
    let qam256 = Qam256::<f32>::QAM256;
    group.bench_function("qam256/modulate_1to1", |b| {
        b.iter(|| {
            let symbols: Vec<Complex<f32>> = payload.iter().copied().modulate(qam256).collect();
            std::hint::black_box(symbols);
        })
    });

    let qam256_syms: Vec<Complex<f32>> = payload.iter().copied().modulate(qam256).collect();
    group.bench_function("qam256/demodulate_square_o1", |b| {
        b.iter(|| {
            let recovered: Vec<u8> = qam256_syms
                .iter()
                .copied()
                .demodulate_hard(qam256)
                .collect();
            std::hint::black_box(recovered);
        })
    });

    group.finish();
}

/// 2. Channel Impairments Overhead (AWGN, CFO, Wiener Noise)
fn bench_channel_impairments(c: &mut Criterion) {
    let mut group = c.benchmark_group("channels");
    let payload = vec![0xA5; PAYLOAD_SIZE];
    let qam16 = Qam16::<f32>::QAM16;
    let symbols: Vec<Complex<f32>> = payload.iter().copied().modulate(qam16).collect();

    group.throughput(Throughput::Elements(symbols.len() as u64));

    // A. AWGN Injection
    group.bench_function("awgn_snr_injection", |b| {
        b.iter(|| {
            let rng = Xoshiro256PlusPlus::seed_from_u64(42);
            let noisy: Vec<Complex<f32>> = symbols
                .iter()
                .copied()
                .add_awgn_snr(&qam16, 20.0, rng)
                .collect();
            std::hint::black_box(noisy);
        })
    });

    // B. Static Phase Offset (Trig multiply only)
    group.bench_function("static_phase_rotation", |b| {
        b.iter(|| {
            let rotated: Vec<Complex<f32>> = symbols
                .iter()
                .copied()
                .add_phase_offset(core::f32::consts::FRAC_PI_4)
                .collect();
            std::hint::black_box(rotated);
        })
    });

    // C. CFO Streaming (Frequency rotation ramp)
    group.bench_function("cfo_stream", |b| {
        b.iter(|| {
            let rotated: Vec<Complex<f32>> = symbols
                .iter()
                .copied()
                .add_cfo_hz(1000.0, 100_000.0)
                .collect();
            std::hint::black_box(rotated);
        })
    });

    // D. Wiener Phase Noise (Gaussian step PRNG + rotation)
    group.bench_function("wiener_phase_noise", |b| {
        b.iter(|| {
            let rng = Xoshiro256PlusPlus::seed_from_u64(42);
            let noisy: Vec<Complex<f32>> =
                symbols.iter().copied().add_phase_noise(0.01, rng).collect();
            std::hint::black_box(noisy);
        })
    });

    group.finish();
}

/// 3. End-to-End Transceiver Pipelines
fn bench_end_to_end_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end_pipeline");
    let payload = vec![0x3C; PAYLOAD_SIZE];
    let qam16 = Qam16::<f32>::QAM16;
    group.throughput(Throughput::Bytes(PAYLOAD_SIZE as u64));

    // Clean pipeline: Modulate -> Demodulate
    group.bench_function("qam16_clean_loopback", |b| {
        b.iter(|| {
            let recovered: Vec<u8> = payload
                .iter()
                .copied()
                .modulate(qam16)
                .demodulate_hard(qam16)
                .collect();
            std::hint::black_box(recovered);
        })
    });

    // Impaired pipeline: Modulate -> Phase -> AWGN -> Demodulate
    group.bench_function("qam16_impaired_channel_chain", |b| {
        b.iter(|| {
            let rng_phase = Xoshiro256PlusPlus::seed_from_u64(1);
            let rng_awgn = Xoshiro256PlusPlus::seed_from_u64(2);
            let recovered: Vec<u8> = payload
                .iter()
                .copied()
                .modulate(qam16)
                .add_cfo_hz(50.0, 1_000_000.0)
                .add_phase_noise(0.001, rng_phase)
                .add_awgn_snr(&qam16, 25.0, rng_awgn)
                .demodulate_hard(qam16)
                .collect();
            std::hint::black_box(recovered);
        })
    });

    group.finish();
}

/// 4. Soft Demodulation (LLR Extraction)
fn bench_soft_demodulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("soft_demodulation");
    let payload = vec![0x55; 256 * 1024]; // 256 KB
    let qpsk = Qpsk::<f32>::QPSK;
    let symbols: Vec<Complex<f32>> = payload.iter().copied().modulate(qpsk).collect();
    let noise_var = 0.1f32;

    group.throughput(Throughput::Elements(symbols.len() as u64));

    // Symbol LLR vectors: [f32; 2]
    group.bench_function("qpsk_soft_symbols", |b| {
        b.iter(|| {
            let llrs: Vec<[f32; 2]> = symbols
                .iter()
                .copied()
                .demodulate_soft(qpsk, noise_var)
                .collect();
            std::hint::black_box(llrs);
        })
    });

    // Flattened bitstream LLRs: f32
    group.bench_function("qpsk_soft_bits_flattened", |b| {
        b.iter(|| {
            let llrs: Vec<f32> = symbols
                .iter()
                .copied()
                .demodulate_soft_bits(qpsk, noise_var)
                .collect();
            std::hint::black_box(llrs);
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
