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
