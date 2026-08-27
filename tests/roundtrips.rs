use constella::prelude::*;

#[test]
fn test_all_standard_constellations_roundtrip() {
    let payload: Vec<u8> = (0..120).map(|x| (x * 43) as u8).collect();

    // BPSK
    assert_eq!(
        payload
            .iter()
            .copied()
            .modulate(&Bpsk::<f32>::BPSK)
            .demodulate_hard(&Bpsk::<f32>::BPSK)
            .collect::<Vec<_>>(),
        payload
    );

    // QPSK
    assert_eq!(
        payload
            .iter()
            .copied()
            .modulate(&Qpsk::<f32>::QPSK)
            .demodulate_hard(&Qpsk::<f32>::QPSK)
            .collect::<Vec<_>>(),
        payload
    );

    // 16-QAM
    assert_eq!(
        payload
            .iter()
            .copied()
            .modulate(&Qam16::<f32>::QAM16)
            .demodulate_hard(&Qam16::<f32>::QAM16)
            .collect::<Vec<_>>(),
        payload
    );
}

#[cfg(test)]
mod rotated_psk_tests {
    use constella::constellation::RotatedPsk;
    use constella::prelude::*;

    use std::f32::consts::{PI, TAU};

    #[test]
    fn test_rotated_psk_all_orders_sweep() {
        let offsets = [-7.0f32, -PI, 0.0, 0.5, PI, TAU, 12.56];

        for &offset in &offsets {
            // BPSK (M = 2)
            let bpsk =
                Constellation::<f32, 2, Normalized, RotatedPsk<f32>>::m_psk_with_phase(offset);
            for i in 0..2 {
                assert_eq!(bpsk.demodulate_hard_point(bpsk[i]), i);
            }

            // QPSK (M = 4)
            let qpsk =
                Constellation::<f32, 4, Normalized, RotatedPsk<f32>>::m_psk_with_phase(offset);
            for i in 0..4 {
                assert_eq!(qpsk.demodulate_hard_point(qpsk[i]), i);
            }

            // 8-PSK (M = 8)
            let psk8 =
                Constellation::<f32, 8, Normalized, RotatedPsk<f32>>::m_psk_with_phase(offset);
            for i in 0..8 {
                assert_eq!(psk8.demodulate_hard_point(psk8[i]), i);
            }

            // 16-PSK (M = 16)
            let psk16 = Constellation::<f64, 16, Normalized, RotatedPsk<f64>>::m_psk_with_phase(
                offset as f64,
            );
            for i in 0..16 {
                assert_eq!(psk16.demodulate_hard_point(psk16[i]), i);
            }
        }
    }
}
