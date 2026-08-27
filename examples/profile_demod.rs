use constella::prelude::*;
use num_complex::Complex;
use std::hint::black_box;

fn main() {
    let payload = vec![0x5A; 1024 * 1024];
    let qam16 = Qam16::<f32>::QAM16;

    // Pre-modulate once outside the hot loop
    let symbols: Vec<Complex<f32>> = payload.iter().copied().modulate(&qam16).collect();

    // 1,000 iterations = 2 billion symbols demodulated and consumed on CPU
    for _ in 0..1_000 {
        let stream = symbols.iter().copied().demodulate_hard(&qam16);
        for byte in stream {
            black_box(byte);
        }
    }
}
