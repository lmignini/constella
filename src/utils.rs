pub const fn is_valid_constellation_size(m: usize) -> bool {
    m >= 2 && m.is_power_of_two()
}

pub const fn gives_square_constellation(m: usize) -> bool {
    m.is_power_of_two() && m >= 4 && m.ilog2().is_multiple_of(2)
}
pub const fn binary_to_gray(val: usize) -> usize {
    val ^ (val >> 1)
}

#[allow(dead_code)]
pub const fn gray_to_binary(mut val: usize) -> usize {
    let mut b: usize = 0;

    while val > 0 {
        b ^= val;
        val = val >> 1;
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_known_gray_mappings_2bit() {
        assert_eq!(binary_to_gray(0b00), 0b00);
        assert_eq!(binary_to_gray(0b01), 0b01);
        assert_eq!(binary_to_gray(0b10), 0b11);
        assert_eq!(binary_to_gray(0b11), 0b10);

        assert_eq!(gray_to_binary(0b00), 0b00);
        assert_eq!(gray_to_binary(0b01), 0b01);
        assert_eq!(gray_to_binary(0b11), 0b10);
        assert_eq!(gray_to_binary(0b10), 0b11);
    }

    #[test]
    fn test_known_gray_mappings_3bit() {
        let expected = [
            0b000, // 0
            0b001, // 1
            0b011, // 2
            0b010, // 3
            0b110, // 4
            0b111, // 5
            0b101, // 6
            0b100, // 7
        ];

        for (bin, &gray) in expected.iter().enumerate() {
            assert_eq!(binary_to_gray(bin), gray);
            assert_eq!(gray_to_binary(gray), bin);
        }
    }

    #[test]
    fn test_roundtrip_identity() {
        for val in 0..2048 {
            let gray = binary_to_gray(val);
            let binary = gray_to_binary(gray);
            assert_eq!(binary, val, "Roundtrip failed for binary value: {val}");
        }
    }

    #[test]
    fn test_single_bit_transition_property() {
        for val in 0..2047 {
            let g1 = binary_to_gray(val);
            let g2 = binary_to_gray(val + 1);
            let bit_diff = (g1 ^ g2).count_ones();
            assert_eq!(
                bit_diff,
                1,
                "Adjacent Gray codes for {val} and {} differ by {bit_diff} bits instead of 1",
                val + 1
            );
        }
    }

    #[test]
    fn test_const_evaluation() {
        const TEST_VAL: usize = 0b1010_1100;
        const CONST_GRAY: usize = binary_to_gray(TEST_VAL);
        const CONST_BIN: usize = gray_to_binary(CONST_GRAY);

        assert_eq!(CONST_GRAY, 0b1111_1010);
        assert_eq!(CONST_BIN, TEST_VAL);
    }
}
