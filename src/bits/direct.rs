pub trait DirectBitCodec<const K: usize, const N: usize> {
    /// Unpacks a single raw byte into N distinct K-bit symbol indices.
    fn unpack_byte(byte: u8) -> [usize; N];

    /// Packs N distinct K-bit symbol indices into a single raw byte.
    fn pack_symbols(symbols: [usize; N]) -> u8;
}
