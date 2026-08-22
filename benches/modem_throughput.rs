use constella::prelude::*;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use num_complex::Complex;

fn bench_qam16_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("qam16");
    let payload = vec![0x5A; 1024 * 1024]; // 1 MB payload
    group.throughput(Throughput::Bytes(payload.len() as u64)); //

    // Fast-path Square QAM (O(1) 1D Cartesian Slicing)
    let qam16_square = Qam16::<f32>::QAM16;

    // General Constellation with identical points (O(M) Euclidean Distance Search)
    let qam16_general = Constellation::<f32, 16>::from_points_normalized(*qam16_square.points());

    // 1. Modulation Benchmark (1 MB -> 2M Complex Symbols)
    group.bench_function("modulate_1mb", |b| {
        b.iter(|| {
            let symbols: Vec<Complex<f32>> =
                payload.iter().copied().modulate(qam16_square).collect();
            std::hint::black_box(symbols);
        })
    });

    // Pre-generate symbols for isolated demodulation benchmarking
    let symbols: Vec<Complex<f32>> = payload.iter().copied().modulate(qam16_square).collect();

    // 2. Square QAM O(1) Slicing Demodulation
    group.bench_function("demodulate_square_o1_1mb", |b| {
        b.iter(|| {
            let recovered: Vec<u8> = symbols
                .iter()
                .copied()
                .demodulate_hard(qam16_square)
                .collect();
            std::hint::black_box(recovered);
        })
    });

    // 3. General Geometry O(M) Nearest-Neighbor Search Demodulation
    group.bench_function("demodulate_general_om_1mb", |b| {
        b.iter(|| {
            let recovered: Vec<u8> = symbols
                .iter()
                .copied()
                .demodulate_hard(qam16_general)
                .collect();
            std::hint::black_box(recovered);
        })
    });

    group.finish(); //
}

criterion_group!(benches, bench_qam16_throughput);
criterion_main!(benches); //
