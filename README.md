# constella

A high-performance, `#![no_std]` bit modulation and demodulation library in Rust featuring compile-time const-evaluable
constellations, fast-path $O (1)$ decision slicing, and soft Log-Likelihood Ratio (LLR) calculation.

`constella` provides zero-allocation, streaming bit-slicing, modulation, and demodulation pipelines engineered
specifically for digital signal processing (DSP), software-defined radio (SDR), and resource-constrained embedded
systems.

---

## Key Highlights

- **Pure `#![no_std]` First-Class Support**: Designed from the ground up for bare-metal microcontrollers, DSP
  processors, and memory-constrained platforms with zero mandatory standard library dependencies.
- **Zero Heap Allocations**: Slicing, packing, modulation, and demodulation are implemented as streaming iterators
  operating directly on stack registers (`u64` bit buffers) and fixed-size arrays.
- **`const fn` Constellation Generation**: Standard BPSK, QPSK, 8-PSK, and Square QAM constellations (16-QAM up to
  4096-QAM) can be constructed and energy-normalized entirely at compile time.
- **Fast-Path $O (1)$ Demodulation**: Hardware-friendly 1D coordinate slicing for Square QAM constellations bypassing
  brute-force Euclidean distance search loops ($O (M) \rightarrow O (1)$).
- **Soft Decision / LLR Demodulation**: Calculate exact and Max-Log Log-Likelihood Ratios for soft-decision Forward
  Error Correction (FEC) decoders like LDPC, Turbo, and Viterbi decoders.
- **Arbitrary Custom Constellations**: Users can define irregular or experimental constellations from static arrays with
  compile-time power normalization.
- **Flexible Bit Packing & Endianness**: Full control over symbol bit streams with configurable bit-endianness
  (`MsbFirst`, `LsbFirst`) and end-of-stream padding strategies (`PadZeros`, `DiscardRemainder`, `ExactOnly`).

---

## Constellation Matrix & Performance Characteristics

| Constellation    | Bits / Symbol ($K$) | Geometry Marker | Demodulation Pipeline                       | Time Complexity |
|:-----------------|:--------------------|:----------------|:--------------------------------------------|:----------------|
| **BPSK**         | 1                   | `General`       | Direct Byte-Aligned / Nearest Neighbor      | $O(1)$          |
| **QPSK / 4-QAM** | 2                   | `General`       | Direct Byte-Aligned / Nearest Neighbor      | $O(1)$          |
| **8-PSK**        | 3                   | `General`       | Multi-Byte Bit-Streaming / Euclidean Search | $O(M)$          |
| **16-QAM**       | 4                   | `SquareQam<T>`  | Fast-Path 1D Slicing (Direct Byte-Aligned)  | $O(1)$          |
| **64-QAM**       | 6                   | `SquareQam<T>`  | Fast-Path 1D Slicing (Multi-Byte Streaming) | $O(1)$          |
| **256-QAM**      | 8                   | `SquareQam<T>`  | Fast-Path 1D Slicing (Direct Byte-Aligned)  | $O(1)$          |
| **1024-QAM**     | 10                  | `SquareQam<T>`  | Fast-Path 1D Slicing (Multi-Byte Streaming) | $O(1)$          |
| **4096-QAM**     | 12                  | `SquareQam<T>`  | Fast-Path 1D Slicing (Multi-Byte Streaming) | $O(1)$          |
| **Custom Grid**  | $\log_2(M)$         | `General`       | Minimum Euclidean Distance Search           | $O(M)$          |

---

## Installation

Add `constella` to your `Cargo.toml`:

```toml
[dependencies]
constella = "0.1.0"
num-complex = { version = "0.4", default-features = false }
```

### Feature Flags

| Feature   | Description                                                               |
|:----------|:--------------------------------------------------------------------------|
| `default` | `[]` (Pure `#![no_std]` without any dynamic heap allocation)              |
| `alloc`   | Enables heap-allocated collection conversions (e.g. `Vec`)                |
| `std`     | Enables Rust standard library integrations (implies `alloc`)              |
| `libm`    | Enables math primitives for targets lacking hardware FPUs in `#![no_std]` |

---

## Quickstart & Usage Examples

### 1. End-to-End Modulation and Demodulation Pipeline

Stream raw data bytes directly into complex baseband IQ symbols and recover them back with zero heap overhead:

```rust
use constella::prelude::*;
use num_complex::Complex;

fn main() {
    let payload = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE];
    let qam16 = Qam16::<f32>::QAM16;

    // 1. Modulate bytes into complex baseband symbols (streaming iterator)
    let symbols: Vec<Complex<f32>> = payload
        .into_iter()
        .modulate(qam16)
        .collect();

    // 2. Demodulate received baseband symbols back into bytes via fast O(1) slicing
    let recovered: Vec<u8> = symbols
        .into_iter()
        .demodulate_hard(qam16)
        .collect();

    assert_eq!(&payload[..], &recovered[..]);
}
```

---

### 2. Soft-Decision Demodulation (Log-Likelihood Ratios)

For forward error correction (FEC) decoders, extract signed LLRs indicating bit probabilities (positive values favor bit
`0`, negative values favor bit `1`):

```rust
use constella::demodulation::DemodulateExt;
use constella::constellation::Bpsk;
use num_complex::Complex;

fn main() {
    let bpsk = Bpsk::<f32>::BPSK;
    let noise_variance = 0.1f32; // Channel noise variance sigma^2

    // Received IQ baseband samples with channel noise
    let received_samples = [
        Complex::new(0.92, 0.04),   // Close to +1.0 (bit 0)
        Complex::new(-1.08, -0.05), // Close to -1.0 (bit 1)
    ];

    // Compute scalar LLR per received bit
    let bit_llrs: Vec<f32> = received_samples
        .into_iter()
        .demodulate_soft_bits(bpsk, noise_variance)
        .collect();

    assert!(bit_llrs[0] > 0.0); // High confidence for bit 0
    assert!(bit_llrs[1] < 0.0); // High confidence for bit 1
}
```

---

### 3. Custom Constellation with Compile-Time Normalization

Construct custom or non-standard constellations at global/static scope with verified unit average symbol energy
($E_{\text{avg}} = 1.0$):

```rust
use constella::constellation::{Constellation, Normalized};
use constella::modulation::ModulateExt;
use constella::demodulation::DemodulateExt;
use num_complex::Complex;

// Define custom 4-ary non-Cartesian constellation points
const CUSTOM_POINTS: [Complex<f32>; 4] = [
    Complex::new(2.0, 1.0),
    Complex::new(-1.0, 3.0),
    Complex::new(-2.0, -2.0),
    Complex::new(3.0, -1.0),
];

// Automatically compute scaling factor and normalize to unit energy at compile time
pub static CUSTOM_CONSTEL: Constellation<f32, 4, Normalized> =
    Constellation::<f32, 4>::from_points_normalized(CUSTOM_POINTS);

fn main() {
    let payload = [0x12, 0x34, 0xAB, 0xCD];

    let symbols: Vec<_> = payload.into_iter().modulate(CUSTOM_CONSTEL).collect();
    let recovered: Vec<u8> = symbols.into_iter().demodulate_hard(CUSTOM_CONSTEL).collect();

    assert_eq!(&payload[..], &recovered[..]);
}
```

---

### 4. Low-Level Zero-Allocation Bit Slicing & Packing

Directly slice byte streams into irregular symbol bit widths (e.g. 3 bits for 8-PSK, 5 bits for 32-QAM) and repack them:

```rust
use constella::bits::{ChunkBitsExt, PackBitsExt, LsbFirst, MsbFirst, PadZeros, DiscardRemainder};

fn main() {
    let data = [0b101_100_11, 0b0_101_110_0, 0b11_000_111]; // 24 bits = 8 x 3-bit chunks

    // Chunk into 3-bit symbol indices
    let chunks: Vec<usize> = data.into_iter().chunk_bits::<3>().collect();
    assert_eq!(chunks.len(), 8);

    // Pack 3-bit symbol indices back into full bytes
    let repacked: Vec<u8> = chunks.into_iter().pack_bits::<3>().collect();
    assert_eq!(&repacked[..], &data[..]);
}
```

---

## Architecture & Design Principles

```
  +-------------------------------------------------------------+
  |                        Input Data                           |
  |                (&[u8] / Iterator<Item = u8>)                |
  +-------------------------------------------------------------+
                                 |
                                 v
  +-------------------------------------------------------------+
  |              Bit Streaming & Slicing Layer                  |
  |   (BitChunker: K-bit register slicer, MSB/LSB, Zero-Alloc)  |
  +-------------------------------------------------------------+
                                 |
                                 v
  +-------------------------------------------------------------+
  |                 Constellation Mapping                       |
  |  (Constellation<T, M, S, G>: const-eval table & Gray map)   |
  +-------------------------------------------------------------+
                                 |
                                 v
  +-------------------------------------------------------------+
  |                   Modulation Adapter                        |
  |      (Yields Complex<T> Baseband In-Phase / Quadrature)     |
  +-------------------------------------------------------------+
                                 |
                     [ Channel / RF Transmission ]
                                 |
                                 v
  +-------------------------------------------------------------+
  |                    Demodulation Engine                      |
  |                                                             |
  |  * SquareQam<T>: Fast O(1) 1D Slicing & De-mapping          |
  |  * General: Linear O(M) Minimum Euclidean Distance Search   |
  |  * Soft Demodulation: Exact / Max-Log Scalar LLR Generator  |
  +-------------------------------------------------------------+
                                 |
                                 v
  +-------------------------------------------------------------+
  |                       Bit Packer                            |
  |  (BitPacker: Reconstitutes raw byte stream [u8] or LLRs)    |
  +-------------------------------------------------------------+
```

---

## Benchmarks & Performance

`constella` features dedicated SIMD-friendly loops and byte-aligned direct paths for $K \in \{1, 2, 4, 8\}$ (BPSK, QPSK,
16-QAM, 256-QAM).

To run benchmarks on your local machine:

```bash
cargo bench
```

---

## Publishing to crates.io Checklist

Before releasing a new version to [crates.io](https://crates.io):

1. Ensure documentation tests and unit tests pass on pure `no_std` targets:
   ```bash
   cargo test --no-default-features
   cargo test --all-features
   ```
2. Verify bare-metal compilation:
   ```bash
   rustup target add thumbv7em-none-eabihf
   cargo check --no-default-features --target thumbv7em-none-eabihf
   ```
3. Run packaging dry-run:
   ```bash
   cargo package
   cargo publish --dry-run
   ```

---

## License

Dual-licensed under either of:

- **Apache License, Version 2.0**
- **MIT License**

at your option.
