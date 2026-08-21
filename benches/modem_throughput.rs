use constella::prelude::*;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use num_complex::Complex;

fn bench_qam16_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("qam16");
    let payload = vec![0x5A; 1024 * 1024]; // 1 MB payload
    group.throughput(Throughput::Bytes(payload.len() as u64));

    let qam16 = Qam16::<f32>::QAM16;

    group.bench_function("modulate_1mb", |b| {
        b.iter(|| {
            let symbols: Vec<Complex<f32>> = payload.iter().copied().modulate(qam16).collect();
            std::hint::black_box(symbols);
        })
    });

    let symbols: Vec<Complex<f32>> = payload.iter().copied().modulate(qam16).collect();

    group.bench_function("demodulate_hard_1mb", |b| {
        b.iter(|| {
            let recovered: Vec<u8> = symbols.iter().copied().demodulate_hard(qam16).collect();
            std::hint::black_box(recovered);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_qam16_roundtrip);
criterion_main!(benches);
