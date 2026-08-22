use constella::prelude::*;
use num_complex::Complex;
use std::hint::black_box;

fn main() {
    let payload = vec![0x5A; 1024 * 1024]; // 1 MB payload
    let qam16 = Qam16::<f32>::QAM16;

    // Pre-modulate once so symbol allocation is excluded from profiling
    let symbols: Vec<Complex<f32>> = payload
        .iter()
        .copied()
        .modulate(qam16)
        .collect();

    // Hot loop: 1,000 iterations = 2 billion complex symbols demodulated back to bytes
    for _ in 0..1_000 {
        let recovered: Vec<u8> = symbols
            .iter()
            .copied()
            .demodulate_hard(qam16)
            .collect();

        black_box(recovered);
    }
}