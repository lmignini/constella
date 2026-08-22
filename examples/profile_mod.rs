use constella::prelude::*;
use std::hint::black_box;

fn main() {
    let payload = vec![0x5A; 1024 * 1024]; // 1 MB payload allocated once upfront
    let qam16 = Qam16::<f32>::QAM16;

    // 1,000 iterations = 2 billion complex symbols streamed and consumed on CPU
    for _ in 0..1_000 {
        let stream = payload.iter().copied().modulate(qam16);
        for symbol in stream {
            black_box(symbol);
        }
    }
}