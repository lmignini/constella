use constella::prelude::*;
use num_complex::Complex;
use std::hint::black_box;

fn main() {
    let payload = vec![0x5A; 1024 * 1024]; // 1 MB payload
    let qam16 = Qam16::<f32>::QAM16;

    // Hot loop: 1,000 iterations = 1 GB data modulated into 2 billion complex symbols
    for _ in 0..1_000 {
        let symbols: Vec<Complex<f32>> = payload.iter().copied().modulate(qam16).collect();

        black_box(symbols);
    }
}
