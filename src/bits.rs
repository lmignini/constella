mod direct;

use core::marker::PhantomData;
pub use direct::DirectBitCodec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MsbFirst;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LsbFirst;

pub trait BitOrder {
    const IS_MSB_FIRST: bool;
    // Functions used in BitChunker
    fn push_byte(buffer: &mut u64, count: &mut u8, byte: u8);
    fn extract_chunk<const K: usize>(buffer: &mut u64, count: &mut u8) -> usize;
    fn finalize_pad_zeros<const K: usize>(buffer: &mut u64, count: &mut u8) -> usize;

    // Functions used in BitPacker
    fn extract_byte(buffer: &mut u64, count: &mut u8) -> u8;
    fn push_chunk<const K: usize>(buffer: &mut u64, count: &mut u8, chunk: usize);
}

impl BitOrder for MsbFirst {
    const IS_MSB_FIRST: bool = true;
    #[inline]
    fn push_byte(buffer: &mut u64, count: &mut u8, byte: u8) {
        *buffer <<= 8;
        *buffer |= byte as u64;
        *count += 8;
    }
    #[inline]
    fn extract_chunk<const K: usize>(buffer: &mut u64, count: &mut u8) -> usize {
        let mask_for_bits_to_extract = (1u64 << K) - 1;
        *count -= K as u8;
        let chunk = (*buffer >> *count) & mask_for_bits_to_extract;
        *buffer &= (1u64 << *count) - 1;
        chunk as usize
    }
    #[inline]
    fn finalize_pad_zeros<const K: usize>(buffer: &mut u64, count: &mut u8) -> usize {
        let missing_bits = K as u8 - *count;
        let chunk = (*buffer << missing_bits) & ((1u64 << K) - 1);
        *buffer = 0;
        *count = 0;
        chunk as usize
    }
    #[inline]
    fn extract_byte(buffer: &mut u64, count: &mut u8) -> u8 {
        *count -= 8;
        let byte = ((*buffer >> *count) & 0xFF) as u8;
        *buffer &= (1u64 << *count) - 1;
        byte
    }
    #[inline]
    fn push_chunk<const K: usize>(buffer: &mut u64, count: &mut u8, chunk: usize) {
        *buffer <<= K;
        *buffer |= (chunk as u64) & ((1u64 << K) - 1);
        *count += K as u8;
    }
}

impl BitOrder for LsbFirst {
    const IS_MSB_FIRST: bool = false;

    #[inline]
    fn push_byte(buffer: &mut u64, count: &mut u8, byte: u8) {
        *buffer |= (byte as u64) << *count;
        *count += 8;
    }
    #[inline]
    fn extract_chunk<const K: usize>(buffer: &mut u64, count: &mut u8) -> usize {
        let mask_for_bits_to_extract = (1u64 << K) - 1;
        let chunk = *buffer & mask_for_bits_to_extract;
        *buffer >>= K;
        *count -= K as u8;
        chunk as usize
    }
    #[inline]
    fn finalize_pad_zeros<const K: usize>(buffer: &mut u64, count: &mut u8) -> usize {
        let chunk = *buffer & ((1u64 << K) - 1);
        *buffer = 0;
        *count = 0;
        chunk as usize
    }
    #[inline]
    fn extract_byte(buffer: &mut u64, count: &mut u8) -> u8 {
        let byte = (*buffer & 0xFF) as u8;
        *buffer >>= 8;
        *count -= 8;
        byte
    }
    #[inline]
    fn push_chunk<const K: usize>(buffer: &mut u64, count: &mut u8, chunk: usize) {
        let mask = (1u64 << K) - 1;
        *buffer |= ((chunk as u64) & mask) << *count;
        *count += K as u8;
    }
}

// =============================================================================
// Direct Bit Codecs
// =============================================================================

impl DirectBitCodec<1, 8> for MsbFirst {
    #[inline]
    fn unpack_byte(byte: u8) -> [usize; 8] {
        [
            ((byte >> 7) & 1) as usize,
            ((byte >> 6) & 1) as usize,
            ((byte >> 5) & 1) as usize,
            ((byte >> 4) & 1) as usize,
            ((byte >> 3) & 1) as usize,
            ((byte >> 2) & 1) as usize,
            ((byte >> 1) & 1) as usize,
            (byte & 1) as usize,
        ]
    }

    #[inline]
    fn pack_symbols(symbols: [usize; 8]) -> u8 {
        (((symbols[0] & 1) << 7)
            | ((symbols[1] & 1) << 6)
            | ((symbols[2] & 1) << 5)
            | ((symbols[3] & 1) << 4)
            | ((symbols[4] & 1) << 3)
            | ((symbols[5] & 1) << 2)
            | ((symbols[6] & 1) << 1)
            | (symbols[7] & 1)) as u8
    }
}

impl DirectBitCodec<1, 8> for LsbFirst {
    #[inline]
    fn unpack_byte(byte: u8) -> [usize; 8] {
        [
            (byte & 1) as usize,
            ((byte >> 1) & 1) as usize,
            ((byte >> 2) & 1) as usize,
            ((byte >> 3) & 1) as usize,
            ((byte >> 4) & 1) as usize,
            ((byte >> 5) & 1) as usize,
            ((byte >> 6) & 1) as usize,
            ((byte >> 7) & 1) as usize,
        ]
    }

    #[inline]
    fn pack_symbols(symbols: [usize; 8]) -> u8 {
        ((symbols[0] & 1)
            | ((symbols[1] & 1) << 1)
            | ((symbols[2] & 1) << 2)
            | ((symbols[3] & 1) << 3)
            | ((symbols[4] & 1) << 4)
            | ((symbols[5] & 1) << 5)
            | ((symbols[6] & 1) << 6)
            | ((symbols[7] & 1) << 7)) as u8
    }
}

impl DirectBitCodec<2, 4> for MsbFirst {
    #[inline]
    fn unpack_byte(byte: u8) -> [usize; 4] {
        [
            ((byte >> 6) & 0x03) as usize,
            ((byte >> 4) & 0x03) as usize,
            ((byte >> 2) & 0x03) as usize,
            (byte & 0x03) as usize,
        ]
    }

    #[inline]
    fn pack_symbols(symbols: [usize; 4]) -> u8 {
        (((symbols[0] & 0x03) << 6)
            | ((symbols[1] & 0x03) << 4)
            | ((symbols[2] & 0x03) << 2)
            | (symbols[3] & 0x03)) as u8
    }
}

impl DirectBitCodec<2, 4> for LsbFirst {
    #[inline]
    fn unpack_byte(byte: u8) -> [usize; 4] {
        [
            (byte & 0x03) as usize,
            ((byte >> 2) & 0x03) as usize,
            ((byte >> 4) & 0x03) as usize,
            ((byte >> 6) & 0x03) as usize,
        ]
    }

    #[inline]
    fn pack_symbols(symbols: [usize; 4]) -> u8 {
        ((symbols[0] & 0x03)
            | ((symbols[1] & 0x03) << 2)
            | ((symbols[2] & 0x03) << 4)
            | ((symbols[3] & 0x03) << 6)) as u8
    }
}

impl DirectBitCodec<4, 2> for MsbFirst {
    #[inline]
    fn unpack_byte(byte: u8) -> [usize; 2] {
        [((byte >> 4) & 0x0F) as usize, (byte & 0x0F) as usize]
    }

    #[inline]
    fn pack_symbols(symbols: [usize; 2]) -> u8 {
        (((symbols[0] & 0x0F) << 4) | (symbols[1] & 0x0F)) as u8
    }
}

impl DirectBitCodec<4, 2> for LsbFirst {
    #[inline]
    fn unpack_byte(byte: u8) -> [usize; 2] {
        [(byte & 0x0F) as usize, ((byte >> 4) & 0x0F) as usize]
    }

    #[inline]
    fn pack_symbols(symbols: [usize; 2]) -> u8 {
        (((symbols[1] & 0x0F) << 4) | (symbols[0] & 0x0F)) as u8
    }
}

impl DirectBitCodec<8, 1> for MsbFirst {
    #[inline]
    fn unpack_byte(byte: u8) -> [usize; 1] {
        [byte as usize]
    }

    #[inline]
    fn pack_symbols(symbols: [usize; 1]) -> u8 {
        symbols[0] as u8
    }
}

impl DirectBitCodec<8, 1> for LsbFirst {
    #[inline]
    fn unpack_byte(byte: u8) -> [usize; 1] {
        [byte as usize]
    }

    #[inline]
    fn pack_symbols(symbols: [usize; 1]) -> u8 {
        symbols[0] as u8
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

    #[inline]
    fn pad_extra_symbol(_rem_bits: usize) -> usize {
        0
    }
}

impl Padding for PadZeros {
    fn finalize<const K: usize, O: BitOrder>(buffer: &mut u64, count: &mut u8) -> Option<usize> {
        if *count == 0 {
            None
        } else {
            Some(O::finalize_pad_zeros::<K>(buffer, count))
        }
    }

    #[inline]
    fn pad_extra_symbol(rem_bits: usize) -> usize {
        if rem_bits > 0 { 1 } else { 0 }
    }
}

impl Padding for DiscardRemainder {
    fn finalize<const K: usize, O: BitOrder>(buffer: &mut u64, count: &mut u8) -> Option<usize> {
        *buffer = 0;
        *count = 0;
        None
    }

    #[inline]
    fn pad_extra_symbol(_rem_bits: usize) -> usize {
        0
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

    #[inline]
    fn pad_extra_symbol(_rem_bits: usize) -> usize {
        0
    }
}

/// A zero-allocation streaming iterator that chunks a stream of bytes into $K$-bit symbol indices.
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
    pub fn new(iter: I) -> Self {
        Self::with_order_and_padding(iter)
    }
}

impl<I, const K: usize, O: BitOrder, P: Padding> BitChunker<I, K, O, P>
where
    I: Iterator<Item = u8>,
{
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
        while self.bits_in_buffer < K as u8 {
            match self.iter.next() {
                Some(byte) => O::push_byte(&mut self.bit_buffer, &mut self.bits_in_buffer, byte),
                None => break,
            }
        }

        if self.bits_in_buffer >= K as u8 {
            Some(O::extract_chunk::<K>(
                &mut self.bit_buffer,
                &mut self.bits_in_buffer,
            ))
        } else {
            P::finalize::<K, O>(&mut self.bit_buffer, &mut self.bits_in_buffer)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();
        let buffered = self.bits_in_buffer as usize;

        let lower_bits = lower.saturating_mul(8).saturating_add(buffered);
        let lower_symbols = lower_bits / K + P::pad_extra_symbol(lower_bits % K);

        let upper_symbols = upper.and_then(|u| {
            let upper_bits = u.checked_mul(8)?.checked_add(buffered)?;
            Some(upper_bits / K + P::pad_extra_symbol(upper_bits % K))
        });

        (lower_symbols, upper_symbols)
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
    pub fn new(iter: I) -> Self {
        Self::with_order(iter)
    }
}

impl<I, const K: usize, O: BitOrder> BitPacker<I, K, O>
where
    I: Iterator<Item = usize>,
{
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
        while self.bits_in_buffer < 8 {
            match self.iter.next() {
                Some(chunk) => {
                    O::push_chunk::<K>(&mut self.bit_buffer, &mut self.bits_in_buffer, chunk)
                }
                None => break,
            }
        }

        if self.bits_in_buffer >= 8 {
            Some(O::extract_byte(
                &mut self.bit_buffer,
                &mut self.bits_in_buffer,
            ))
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();
        let buffered = self.bits_in_buffer as usize;

        let lower_bits = lower.saturating_mul(K).saturating_add(buffered);
        let lower_bytes = lower_bits / 8;

        let upper_bytes = upper.and_then(|u| {
            let upper_bits = u.checked_mul(K)?.checked_add(buffered)?;
            Some(upper_bits / 8)
        });

        (lower_bytes, upper_bytes)
    }
}

/// Extension trait providing fluent `.chunk_bits(...)` syntax for byte streams.
pub trait ChunkBitsExt: Iterator<Item = u8> + Sized {
    /// Chunks a stream of bytes into $K$-bit symbol indices using default [`MsbFirst`] ordering.
    #[inline]
    fn chunk_bits<const K: usize>(self) -> BitChunker<Self, K, MsbFirst> {
        BitChunker::new(self)
    }

    /// Chunks a stream of bytes into $K$-bit symbol indices using explicit [`BitOrder`] and [`Padding`].
    #[inline]
    fn chunk_bits_with<const K: usize, O: BitOrder, P: Padding>(self) -> BitChunker<Self, K, O, P> {
        BitChunker::with_order_and_padding(self)
    }
}

pub trait PackBitsExt: Iterator<Item = usize> + Sized {
    #[inline]
    fn pack_bits<const K: usize>(self) -> BitPacker<Self, K, MsbFirst> {
        BitPacker::new(self)
    }

    #[inline]
    fn pack_bits_with<const K: usize, O: BitOrder>(self) -> BitPacker<Self, K, O> {
        BitPacker::with_order(self)
    }
}

impl<I: Iterator<Item = usize>> PackBitsExt for I {}
impl<I: Iterator<Item = u8>> ChunkBitsExt for I {}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn test_bitchunker_size_hint() {
        let data = vec![0xAAu8; 10]; // 80 bits

        // K=3 with PadZeros: 80 / 3 = 26 full + 2 rem -> 27 symbols
        let chunker_pad = BitChunker::<_, 3, MsbFirst, PadZeros>::new(data.clone().into_iter());
        assert_eq!(chunker_pad.size_hint(), (27, Some(27)));

        // K=3 with DiscardRemainder: 80 / 3 = 26 symbols
        let chunker_discard =
            BitChunker::<_, 3, MsbFirst, DiscardRemainder>::with_order_and_padding(
                data.clone().into_iter(),
            );
        assert_eq!(chunker_discard.size_hint(), (26, Some(26)));
    }

    #[test]
    fn test_bitpacker_size_hint() {
        let symbols = vec![0usize; 27]; // 27 symbols with K=3 -> 81 bits -> 10 bytes
        let packer = BitPacker::<_, 3, MsbFirst>::new(symbols.into_iter());
        assert_eq!(packer.size_hint(), (10, Some(10)));
    }

    #[test]
    fn test_bpsk_k1_msb_and_lsb() {
        let data = [0b10110001];
        let msb: Vec<usize> = BitChunker::<_, 1, MsbFirst>::new(data.into_iter()).collect();
        assert_eq!(msb, vec![1, 0, 1, 1, 0, 0, 0, 1]);

        let lsb: Vec<usize> =
            BitChunker::<_, 1, LsbFirst>::with_order_and_padding(data.into_iter()).collect();
        assert_eq!(lsb, vec![1, 0, 0, 0, 1, 1, 0, 1]);
    }

    #[test]
    fn test_qpsk_k2_msb_and_lsb() {
        let data = [0b11_01_00_10];
        let msb: Vec<usize> = BitChunker::<_, 2, MsbFirst>::new(data.into_iter()).collect();
        assert_eq!(msb, vec![3, 1, 0, 2]);

        let lsb: Vec<usize> =
            BitChunker::<_, 2, LsbFirst>::with_order_and_padding(data.into_iter()).collect();
        assert_eq!(lsb, vec![2, 0, 1, 3]);
    }

    #[test]
    fn test_qam16_k4_msb_and_lsb() {
        let data = [0xA5, 0x3C];
        let msb: Vec<usize> = BitChunker::<_, 4, MsbFirst>::new(data.into_iter()).collect();
        assert_eq!(msb, vec![0xA, 0x5, 0x3, 0xC]);

        let lsb: Vec<usize> =
            BitChunker::<_, 4, LsbFirst>::with_order_and_padding(data.into_iter()).collect();
        assert_eq!(lsb, vec![0x5, 0xA, 0xC, 0x3]);
    }

    #[test]
    fn test_8psk_k3_aligned_boundary_crossing() {
        let data = [0b101_100_11, 0b0_101_110_0, 0b11_000_111];
        let msb: Vec<usize> = BitChunker::<_, 3, MsbFirst>::new(data.into_iter()).collect();
        assert_eq!(
            msb,
            vec![0b101, 0b100, 0b110, 0b101, 0b110, 0b011, 0b000, 0b111]
        );

        let lsb: Vec<usize> =
            BitChunker::<_, 3, LsbFirst>::with_order_and_padding(data.into_iter()).collect();
        assert_eq!(lsb.len(), 8);
    }

    #[test]
    fn test_k5_multi_byte_spanning() {
        let data = [0xFF, 0x00, 0xAA, 0x55, 0xF0];
        let msb: Vec<usize> = BitChunker::<_, 5, MsbFirst>::new(data.into_iter()).collect();
        assert_eq!(msb.len(), 8);
        for &chunk in &msb {
            assert!(chunk < (1 << 5), "Chunk {chunk} exceeds 5-bit width");
        }
    }

    #[test]
    fn test_padding_pad_zeros() {
        let data = [0b1101_0110];
        let msb: Vec<usize> =
            BitChunker::<_, 3, MsbFirst, PadZeros>::with_order_and_padding(data.into_iter())
                .collect();
        assert_eq!(msb, vec![0b110, 0b101, 0b100]);

        let lsb: Vec<usize> =
            BitChunker::<_, 3, LsbFirst, PadZeros>::with_order_and_padding(data.into_iter())
                .collect();
        assert_eq!(lsb, vec![0b110, 0b010, 0b011]);
    }

    #[test]
    fn test_padding_discard_remainder() {
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
        let data = [0xAA, 0xBB, 0xCC];
        let msb: Vec<usize> =
            BitChunker::<_, 3, MsbFirst, ExactOnly>::with_order_and_padding(data.into_iter())
                .collect();
        assert_eq!(msb.len(), 8);
    }

    #[test]
    #[should_panic(expected = "Bit stream ended with 2 unaligned bits in ExactOnly mode")]
    fn test_padding_exact_only_panic_on_unaligned() {
        let data = [0xFF];
        let mut chunker =
            BitChunker::<_, 3, MsbFirst, ExactOnly>::with_order_and_padding(data.into_iter());
        chunker.next();
        chunker.next();
        chunker.next();
    }

    #[test]
    fn test_empty_stream() {
        let data: [u8; 0] = [];
        let mut chunker = BitChunker::<_, 4, MsbFirst>::new(data.into_iter());
        assert_eq!(chunker.next(), None);
    }

    #[test]
    fn test_qam4096_k12_slicing() {
        let data = [0x12, 0x34, 0x56];
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
        let symbols = vec![3, 1, 0, 2];
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
        let original: Vec<u8> = (0..120).map(|x| (x * 73) as u8).collect();

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

    #[test]
    fn chunk_bits_with_honours_padding_policy() {
        let data = [0b1101_0110u8];
        let discarded: Vec<usize> = data
            .into_iter()
            .chunk_bits_with::<3, MsbFirst, DiscardRemainder>()
            .collect();
        assert_eq!(discarded, vec![0b110, 0b101]);

        let padded: Vec<usize> = data
            .into_iter()
            .chunk_bits_with::<3, MsbFirst, PadZeros>()
            .collect();
        assert_eq!(padded, vec![0b110, 0b101, 0b100]);
    }

    #[test]
    #[should_panic(expected = "unaligned bits in ExactOnly mode")]
    fn chunk_bits_with_exact_only_panics() {
        let data = [0xFFu8];
        let _: Vec<usize> = data
            .into_iter()
            .chunk_bits_with::<3, MsbFirst, ExactOnly>()
            .collect();
    }
}
