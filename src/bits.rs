use core::marker::PhantomData;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MsbFirst;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LsbFirst;

pub trait BitOrder {
    // Functions used in BitChunker
    fn push_byte(buffer: &mut u64, count: &mut u8, byte: u8);
    fn extract_chunk<const K: usize>(buffer: &mut u64, count: &mut u8) -> usize;
    fn finalize_pad_zeros<const K: usize>(buffer: &mut u64, count: &mut u8) -> usize;

    // Functions used in BitPacker
    fn extract_byte(buffer: &mut u64, count: &mut u8) -> u8;
    fn push_chunk<const K: usize>(buffer: &mut u64, count: &mut u8, chunk: usize);
}

impl BitOrder for MsbFirst {
    fn push_byte(buffer: &mut u64, count: &mut u8, byte: u8) {
        // Make space for 8 new bits (byte) in the buffer
        // The old bits (that should be extracted first move up the buffer, aka to the left)
        *buffer <<= 8;
        // Copy byte into the newly created spaces
        *buffer |= byte as u64;
        // Increment count by number of bits copied into buffer
        *count += 8;
    }

    fn extract_chunk<const K: usize>(buffer: &mut u64, count: &mut u8) -> usize {
        // Extract the K leftmost (MSB) bits
        let mask_for_bits_to_extract = (1u64 << K) - 1;
        *count -= K as u8;
        let chunk = (*buffer >> *count) & mask_for_bits_to_extract;

        *buffer &= (1u64 << *count) - 1; // Keep only remaining valid bits

        chunk as usize
    }

    fn finalize_pad_zeros<const K: usize>(buffer: &mut u64, count: &mut u8) -> usize {
        let missing_bits = K as u8 - *count;
        let chunk = (*buffer << missing_bits) & ((1u64 << K) - 1);
        *buffer = 0;
        *count = 0;
        chunk as usize
    }

    fn extract_byte(buffer: &mut u64, count: &mut u8) -> u8 {
        *count -= 8;
        let byte = ((*buffer >> *count) & 0xFF) as u8;
        *buffer &= (1u64 << *count) - 1;
        byte
    }

    fn push_chunk<const K: usize>(buffer: &mut u64, count: &mut u8, chunk: usize) {
        *buffer <<= K;
        *buffer |= (chunk as u64) & ((1u64 << K) - 1);
        *count += K as u8;
    }
}
impl BitOrder for LsbFirst {
    fn push_byte(buffer: &mut u64, count: &mut u8, byte: u8) {
        // Skip the first count bits (from the right) and copy byte into buffer
        *buffer |= (byte as u64) << *count;
        // Increment count by number of bits copied into buffer
        *count += 8;
    }

    fn extract_chunk<const K: usize>(buffer: &mut u64, count: &mut u8) -> usize {
        // Extract the K rightmost (LSB) bits
        let mask_for_bits_to_extract = (1u64 << K) - 1;
        let chunk = *buffer & mask_for_bits_to_extract;
        // Shift the buffer right to remove the extracted bits
        *buffer >>= K;
        *count -= K as u8;

        chunk as usize
    }

    fn finalize_pad_zeros<const K: usize>(buffer: &mut u64, count: &mut u8) -> usize {
        let chunk = *buffer & ((1u64 << K) - 1);
        *buffer = 0;
        *count = 0;
        chunk as usize
    }

    fn extract_byte(buffer: &mut u64, count: &mut u8) -> u8 {
        let byte = (*buffer & 0xFF) as u8;
        *buffer >>= 8;
        *count -= 8;

        byte
    }

    fn push_chunk<const K: usize>(buffer: &mut u64, count: &mut u8, chunk: usize) {
        let mask = (1u64 << K) - 1;
        *buffer |= ((chunk as u64) & mask) << *count;
        *count += K as u8;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PadZeros;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DiscardRemainder;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExactOnly;

pub trait Padding {
    fn finalize<const K: usize, O: BitOrder>(buffer: &mut u64, count: &mut u8) -> Option<usize>;
}

impl Padding for PadZeros {
    fn finalize<const K: usize, O: BitOrder>(buffer: &mut u64, count: &mut u8) -> Option<usize> {
        if *count == 0 {
            None
        } else {
            Some(O::finalize_pad_zeros::<K>(buffer, count))
        }
    }
}
impl Padding for DiscardRemainder {
    fn finalize<const K: usize, O: BitOrder>(buffer: &mut u64, count: &mut u8) -> Option<usize> {
        *buffer = 0;
        *count = 0;
        None
    }
}
impl Padding for ExactOnly {
    fn finalize<const K: usize, O: BitOrder>(_buffer: &mut u64, count: &mut u8) -> Option<usize> {
        if *count == 0 {
            None
        } else {
            panic!(
                "Bit stream ended with {} unaligned bits in ExactOnly mode",
                *count
            );
        }
    }
}

/// A zero-allocation streaming iterator that chunks a stream of bytes into $K$-bit symbol indices.
///
/// `BitChunker` continuously consumes bytes from an underlying [`Iterator<Item = u8>`], accumulates
/// them in an internal stack-allocated register, and extracts chunks of exactly `K` bits for modulation.
///
/// # Type Parameters
///
/// * `I`: The underlying byte iterator source.
/// * `K`: The number of bits per symbol ($\log_2 M$, where $M$ is the constellation size).
/// * `O`: The bit endianness strategy ([`MsbFirst`] or [`LsbFirst`]). Defaults to [`MsbFirst`].
/// * `P`: The end-of-stream padding policy ([`PadZeros`], [`DiscardRemainder`] and [`ExactOnly`]). Defaults to [`PadZeros`].
///
/// # Panics
///
/// * Panics at construction if `K == 0` or `K > 56` (exceeds internal buffer ingestion headroom).
pub struct BitChunker<I, const K: usize, O = MsbFirst, P = PadZeros> {
    iter: I,
    bit_buffer: u64,
    bits_in_buffer: u8,
    _marker: PhantomData<(O, P)>,
}

impl<I, const K: usize> BitChunker<I, K, MsbFirst, PadZeros>
where
    I: Iterator<Item = u8>,
{
    /// Creates a new `BitChunker` with default [`MsbFirst`] bit ordering and [`PadZeros`] padding.
    ///
    /// # Panics
    ///
    /// Panics if `K == 0` or `K > 56`.
    pub fn new(iter: I) -> Self {
        Self::with_order_and_padding(iter)
    }
}

impl<I, const K: usize, O: BitOrder, P: Padding> BitChunker<I, K, O, P>
where
    I: Iterator<Item = u8>,
{
    /// Creates a new `BitChunker` with provided bit ordering and padding.
    ///
    /// # Panics
    ///
    /// Panics if `K == 0` or `K > 56`.
    pub fn with_order_and_padding(iter: I) -> Self {
        assert!(
            K > 0 && K <= 56,
            "Chunk size K must be between 1 and 56 bits"
        );
        Self {
            iter,
            bit_buffer: 0,
            bits_in_buffer: 0,
            _marker: PhantomData::<(O, P)>,
        }
    }
}

impl<I, const K: usize, O: BitOrder, P: Padding> Iterator for BitChunker<I, K, O, P>
where
    I: Iterator<Item = u8>,
{
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // 1. Pull bytes until we have enough bits for at least one K-bit chunk
        while self.bits_in_buffer < K as u8 {
            match self.iter.next() {
                Some(byte) => O::push_byte(&mut self.bit_buffer, &mut self.bits_in_buffer, byte),
                None => break,
            }
        }

        // 2. If enough bits exist, extract and return a full symbol
        if self.bits_in_buffer >= K as u8 {
            Some(O::extract_chunk::<K>(
                &mut self.bit_buffer,
                &mut self.bits_in_buffer,
            ))
        } else {
            // 3. Otherwise, delegate leftover bits to the padding strategy
            P::finalize::<K, O>(&mut self.bit_buffer, &mut self.bits_in_buffer)
        }
    }
}

pub struct BitPacker<I, const K: usize, O = MsbFirst> {
    iter: I,
    bit_buffer: u64,
    bits_in_buffer: u8,
    _marker: PhantomData<O>,
}

impl<I, const K: usize> BitPacker<I, K, MsbFirst>
where
    I: Iterator<Item = usize>,
{
    /// Creates a new `BitPacker` with default [`MsbFirst`] bit ordering.
    ///
    /// # Panics
    ///
    /// Panics if `K == 0` or `K > 56`.
    pub fn new(iter: I) -> Self {
        Self::with_order(iter)
    }
}

impl<I, const K: usize, O: BitOrder> BitPacker<I, K, O>
where
    I: Iterator<Item = usize>,
{
    /// Creates a new `BitPacker` with provided bit ordering.
    ///
    /// # Panics
    ///
    /// Panics if `K == 0` or `K > 56`.
    pub fn with_order(iter: I) -> Self {
        assert!(
            K > 0 && K <= 56,
            "Chunk size K must be between 1 and 56 bits"
        );
        Self {
            iter,
            bit_buffer: 0,
            bits_in_buffer: 0,
            _marker: PhantomData::<O>,
        }
    }
}

impl<I, const K: usize, O: BitOrder> Iterator for BitPacker<I, K, O>
where
    I: Iterator<Item = usize>,
{
    type Item = u8;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // 1. Pull chunks until we have enough bits for at least one byte
        while self.bits_in_buffer < 8 {
            match self.iter.next() {
                Some(chunk) => {
                    O::push_chunk::<K>(&mut self.bit_buffer, &mut self.bits_in_buffer, chunk)
                }
                None => break,
            }
        }

        // 2. If enough bits exist, extract and return a full byte
        if self.bits_in_buffer >= 8 {
            Some(O::extract_byte(
                &mut self.bit_buffer,
                &mut self.bits_in_buffer,
            ))
        } else {
            // 3. Otherwise, drop trailing padding bits
            None
        }
    }
}

/// Extension trait providing fluent `.pack_bits(...)` syntax directly on symbol iterators.
pub trait ChunkBitsExt: Iterator<Item = u8> + Sized {
    /// Packs an iterator of $K$-bit symbol indices into `u8` bytes using default [`MsbFirst`] ordering.
    #[inline]
    fn chunk_bits<const K: usize>(self) -> BitChunker<Self, K, MsbFirst> {
        BitChunker::new(self)
    }

    /// Packs an iterator of $K$-bit symbol indices into `u8` bytes using explicit [`BitOrder`].
    #[inline]
    fn chunk_bits_with<const K: usize, O: BitOrder, P: Padding>(self) -> BitChunker<Self, K, O> {
        BitChunker::with_order_and_padding(self)
    }
}

// Blanket implementation for all symbol index iterators
impl<I: Iterator<Item = usize>> PackBitsExt for I {}

/// Extension trait providing fluent `.pack_bits(...)` syntax directly on symbol iterators.
pub trait PackBitsExt: Iterator<Item = usize> + Sized {
    /// Packs an iterator of $K$-bit symbol indices into `u8` bytes using default [`MsbFirst`] ordering.
    #[inline]
    fn pack_bits<const K: usize>(self) -> BitPacker<Self, K, MsbFirst> {
        BitPacker::new(self)
    }

    /// Packs an iterator of $K$-bit symbol indices into `u8` bytes using explicit [`BitOrder`].
    #[inline]
    fn pack_bits_with<const K: usize, O: BitOrder>(self) -> BitPacker<Self, K, O> {
        BitPacker::with_order(self)
    }
}

// Blanket implementation for all byte iterators
impl<I: Iterator<Item = u8>> ChunkBitsExt for I {}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    // --- Standard Aligned Slicing (K = 1, 2, 4) ---

    #[test]
    fn test_bpsk_k1_msb_and_lsb() {
        let data = [0b10110001]; // b7..b0

        // MsbFirst: reads bits 7 down to 0[cite: 10]
        let msb: Vec<usize> = BitChunker::<_, 1, MsbFirst>::new(data.into_iter()).collect();
        assert_eq!(msb, vec![1, 0, 1, 1, 0, 0, 0, 1]);

        // LsbFirst: reads bits 0 up to 7[cite: 10]
        let lsb: Vec<usize> =
            BitChunker::<_, 1, LsbFirst>::with_order_and_padding(data.into_iter()).collect();
        assert_eq!(lsb, vec![1, 0, 0, 0, 1, 1, 0, 1]);
    }

    #[test]
    fn test_qpsk_k2_msb_and_lsb() {
        let data = [0b11_01_00_10];

        // MsbFirst: [11, 01, 00, 10] -> [3, 1, 0, 2][cite: 10]
        let msb: Vec<usize> = BitChunker::<_, 2, MsbFirst>::new(data.into_iter()).collect();
        assert_eq!(msb, vec![3, 1, 0, 2]);

        // LsbFirst: [10, 00, 01, 11] -> [2, 0, 1, 3][cite: 10]
        let lsb: Vec<usize> =
            BitChunker::<_, 2, LsbFirst>::with_order_and_padding(data.into_iter()).collect();
        assert_eq!(lsb, vec![2, 0, 1, 3]);
    }

    #[test]
    fn test_qam16_k4_msb_and_lsb() {
        let data = [0xA5, 0x3C]; // [0b1010_0101, 0b0011_1100]

        let msb: Vec<usize> = BitChunker::<_, 4, MsbFirst>::new(data.into_iter()).collect();
        assert_eq!(msb, vec![0xA, 0x5, 0x3, 0xC]);

        let lsb: Vec<usize> =
            BitChunker::<_, 4, LsbFirst>::with_order_and_padding(data.into_iter()).collect();
        assert_eq!(lsb, vec![0x5, 0xA, 0xC, 0x3]);
    }

    // --- Boundary Crossing (K = 3, 5) ---

    #[test]
    fn test_8psk_k3_aligned_boundary_crossing() {
        // 3 bytes = 24 bits = exactly 8 chunks of 3 bits
        let data = [0b101_100_11, 0b0_101_110_0, 0b11_000_111];

        // MsbFirst:
        // Byte 0: [101, 100, (11_)]
        // Byte 1: [(11_0), 101, 110, (_0)]
        // Byte 2: [(0_11), 000, 111]
        let msb: Vec<usize> = BitChunker::<_, 3, MsbFirst>::new(data.into_iter()).collect();
        assert_eq!(
            msb,
            vec![0b101, 0b100, 0b110, 0b101, 0b110, 0b011, 0b000, 0b111]
        );

        // LsbFirst:
        // Byte 0: [011, 001, (10_)]
        // Byte 1: [(10_) | (010_ << 2) -> 01010 -> 010, next: 111, rem: 00]
        let lsb: Vec<usize> =
            BitChunker::<_, 3, LsbFirst>::with_order_and_padding(data.into_iter()).collect();
        assert_eq!(lsb.len(), 8);
    }

    #[test]
    fn test_k5_multi_byte_spanning() {
        // 5 bytes = 40 bits = exactly 8 chunks of 5 bits
        let data = [0xFF, 0x00, 0xAA, 0x55, 0xF0];
        let msb: Vec<usize> = BitChunker::<_, 5, MsbFirst>::new(data.into_iter()).collect();
        assert_eq!(msb.len(), 8);

        for &chunk in &msb {
            assert!(chunk < (1 << 5), "Chunk {chunk} exceeds 5-bit width");
        }
    }

    // --- Padding Strategies ---

    #[test]
    fn test_padding_pad_zeros() {
        // 1 byte = 8 bits. With K = 3: 2 full chunks (6 bits) + 2 leftover bits.
        let data = [0b1101_0110];

        // MsbFirst: chunks [110, 101], remainder 2 bits: [10].
        // PadZeros shifts [10] left by 1 to form [100] (4).[cite: 10]
        let msb: Vec<usize> =
            BitChunker::<_, 3, MsbFirst, PadZeros>::with_order_and_padding(data.into_iter())
                .collect();
        assert_eq!(msb, vec![0b110, 0b101, 0b100]);

        // LsbFirst: lowest 3 bits [110] (6), next 3 bits [010] (2), remainder 2 bits [11].
        // PadZeros takes remainder [11] directly as [011] (3).[cite: 10]
        let lsb: Vec<usize> =
            BitChunker::<_, 3, LsbFirst, PadZeros>::with_order_and_padding(data.into_iter())
                .collect();
        assert_eq!(lsb, vec![0b110, 0b010, 0b011]);
    }

    #[test]
    fn test_padding_discard_remainder() {
        // 1 byte = 8 bits. With K = 3, remaining 2 bits should be dropped.[cite: 10]
        let data = [0b1101_0110];

        let msb: Vec<usize> =
            BitChunker::<_, 3, MsbFirst, DiscardRemainder>::with_order_and_padding(
                data.into_iter(),
            )
            .collect();
        assert_eq!(msb, vec![0b110, 0b101]);

        let lsb: Vec<usize> =
            BitChunker::<_, 3, LsbFirst, DiscardRemainder>::with_order_and_padding(
                data.into_iter(),
            )
            .collect();
        assert_eq!(lsb, vec![0b110, 0b010]);
    }

    #[test]
    fn test_padding_exact_only_success() {
        // 3 bytes = 24 bits (cleanly divisible by K = 3)[cite: 10]
        let data = [0xAA, 0xBB, 0xCC];
        let msb: Vec<usize> =
            BitChunker::<_, 3, MsbFirst, ExactOnly>::with_order_and_padding(data.into_iter())
                .collect();
        assert_eq!(msb.len(), 8);
    }

    #[test]
    #[should_panic(expected = "Bit stream ended with 2 unaligned bits in ExactOnly mode")]
    fn test_padding_exact_only_panic_on_unaligned() {
        // 1 byte = 8 bits with K = 3 leaves 2 unaligned bits -> must panic[cite: 10]
        let data = [0xFF];
        let mut chunker =
            BitChunker::<_, 3, MsbFirst, ExactOnly>::with_order_and_padding(data.into_iter());

        chunker.next(); // 1st chunk (3 bits consumed)
        chunker.next(); // 2nd chunk (6 bits consumed)
        chunker.next(); // Trailing 2 bits -> triggers panic[cite: 10]
    }

    // --- Edge Cases & Constellation Compatibility ---

    #[test]
    fn test_empty_stream() {
        let data: [u8; 0] = [];
        let mut chunker = BitChunker::<_, 4, MsbFirst>::new(data.into_iter());
        assert_eq!(chunker.next(), None);
    }

    #[test]
    fn test_qam4096_k12_slicing() {
        // 3 bytes = 24 bits = exactly two 12-bit symbols
        let data = [0x12, 0x34, 0x56];

        // MsbFirst: symbol 0 = 0x123 (bits 23..12), symbol 1 = 0x456 (bits 11..0)[cite: 10]
        let msb: Vec<usize> = BitChunker::<_, 12, MsbFirst>::new(data.into_iter()).collect();
        assert_eq!(msb, vec![0x123, 0x456]);
    }

    #[test]
    #[should_panic(expected = "Chunk size K must be between 1 and 56 bits")]
    fn test_invalid_k_zero_panics() {
        let data = [0xAA];
        let _ = BitChunker::<_, 0, MsbFirst>::new(data.into_iter());
    }

    #[test]
    #[should_panic(expected = "Chunk size K must be between 1 and 56 bits")]
    fn test_invalid_k_too_large_panics() {
        let data = [0xAA];
        let _ = BitChunker::<_, 57, MsbFirst>::new(data.into_iter());
    }

    #[test]
    fn test_pack_bits_bpsk_k1() {
        // 8 bits -> 1 byte 0b10110001 (0xB1)
        let symbols = vec![1, 0, 1, 1, 0, 0, 0, 1];
        let bytes: Vec<u8> = symbols.into_iter().pack_bits::<1>().collect();
        assert_eq!(bytes, vec![0b10110001]);

        let lsb_symbols = vec![1, 0, 0, 0, 1, 1, 0, 1];
        let lsb_bytes: Vec<u8> = lsb_symbols
            .into_iter()
            .pack_bits_with::<1, LsbFirst>()
            .collect();
        assert_eq!(lsb_bytes, vec![0b10110001]);
    }

    #[test]
    fn test_pack_bits_qpsk_k2() {
        let symbols = vec![3, 1, 0, 2]; // [0b11, 0b01, 0b00, 0b10]
        let bytes: Vec<u8> = symbols.into_iter().pack_bits::<2>().collect();
        assert_eq!(bytes, vec![0b11_01_00_10]);
    }

    #[test]
    fn test_pack_bits_qam16_k4() {
        let symbols = vec![0xA, 0x5, 0x3, 0xC];
        let bytes: Vec<u8> = symbols.into_iter().pack_bits::<4>().collect();
        assert_eq!(bytes, vec![0xA5, 0x3C]);
    }

    #[test]
    fn test_chunk_and_pack_roundtrip_all_k() {
        // 120 bytes is cleanly divisible by all K in 1..=8 (LCM of 1..8 is 840 bits = 105 bytes)
        let original: Vec<u8> = (0..120).map(|x| (x * 73) as u8).collect();

        // Test symmetric and spanning bit widths in MSB order
        assert_eq!(
            original
                .clone()
                .into_iter()
                .chunk_bits::<1>()
                .pack_bits::<1>()
                .collect::<Vec<_>>(),
            original
        );
        assert_eq!(
            original
                .clone()
                .into_iter()
                .chunk_bits::<2>()
                .pack_bits::<2>()
                .collect::<Vec<_>>(),
            original
        );
        assert_eq!(
            original
                .clone()
                .into_iter()
                .chunk_bits::<3>()
                .pack_bits::<3>()
                .collect::<Vec<_>>(),
            original
        );
        assert_eq!(
            original
                .clone()
                .into_iter()
                .chunk_bits::<4>()
                .pack_bits::<4>()
                .collect::<Vec<_>>(),
            original
        );
        assert_eq!(
            original
                .clone()
                .into_iter()
                .chunk_bits::<5>()
                .pack_bits::<5>()
                .collect::<Vec<_>>(),
            original
        );
        assert_eq!(
            original
                .clone()
                .into_iter()
                .chunk_bits::<6>()
                .pack_bits::<6>()
                .collect::<Vec<_>>(),
            original
        );

        // Test LSB order roundtrip
        assert_eq!(
            original
                .clone()
                .into_iter()
                .chunk_bits_with::<3, LsbFirst, PadZeros>()
                .pack_bits_with::<3, LsbFirst>()
                .collect::<Vec<_>>(),
            original
        );
    }
}
