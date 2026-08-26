use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use ringbuf::HeapRb;
use ringbuf::traits::{Consumer, Producer, Split};
use std::f32::consts::PI;
use std::io::{self, BufRead};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use constella::channel::ChannelExt;
use constella::constellation::Qpsk;
use constella::demodulation::DemodulateExt;
use constella::modulation::ModulateExt;
use constella::prelude::DifferentialExt;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let host = cpal::default_host();

    // 1. Setup Audio Input (Microphone) & Output (Speakers)
    let input_device = host
        .default_input_device()
        .expect("No audio input device found");
    let output_device = host
        .default_output_device()
        .expect("No audio output device found");

    println!("Input Device:  {}", input_device.name()?);
    println!("Output Device: {}", output_device.name()?);

    let input_config = input_device.default_input_config()?;
    let output_config = output_device.default_output_config()?;

    let sample_rate = input_config.sample_rate().0;
    println!("Sample Rate:   {} Hz", sample_rate);

    // 2. Lock-free ring buffer between input DSP and output playback
    let latency_samples = (sample_rate as usize) / 4; // ~250ms buffer
    let ring_buffer = HeapRb::<f32>::new(latency_samples * 4);
    let (mut producer, mut consumer) = ring_buffer.split();

    // 3. DSP Configuration
    let constellation = Qpsk::<f32>::QPSK;
    let snr_db = 10.0f32; // Lower to ~12-15 dB for audible noise; increase to >25 dB for clear audio
    let mut _rng_fading = Xoshiro256PlusPlus::seed_from_u64(0x1234_5678);
    let mut rng_awgn = Xoshiro256PlusPlus::seed_from_u64(0x8765_4321);

    let running = Arc::new(AtomicBool::new(true));

    let input_channels = input_config.channels() as usize;
    let output_channels = output_config.channels() as usize;

    // 1. Input Audio Callback (Extract Mono Frame -> Modulate -> Demodulate)
    let input_data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
        // Extract only channel 0 (mono) per frame
        let pcm_bytes: Vec<u8> = data
            .chunks_exact(input_channels)
            .flat_map(|frame| {
                let mono_sample = frame[0].clamp(-1.0, 1.0);
                let sample_i16 = (mono_sample * 32767.0) as i16;
                sample_i16.to_le_bytes()
            })
            .collect();

        let recovered_bytes: Vec<u8> = pcm_bytes
            .into_iter()
            .modulate(constellation)
            .differential_encode()
            .add_phase_offset(PI / 2.0)
            .add_awgn_snr(&constellation, snr_db, &mut rng_awgn)
            .differential_decode() // Channel rotates phase, NO EQUALIZER
            .demodulate_hard(constellation)
            .collect();

        // Push mono samples to ring buffer
        for chunk in recovered_bytes.chunks_exact(2) {
            let sample_i16 = i16::from_le_bytes([chunk[0], chunk[1]]);
            let sample_f32 = (sample_i16 as f32) / 32767.0;
            let _ = producer.try_push(sample_f32);
        }
    };

    // 2. Output Audio Callback (Duplicate Mono Sample to All Stereo Channels)
    let output_data_fn = move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
        for frame in data.chunks_exact_mut(output_channels) {
            let mono_sample = consumer.try_pop().unwrap_or(0.0);
            for channel_slot in frame.iter_mut() {
                *channel_slot = mono_sample;
            }
        }
    };
    // 6. Start Streams
    let err_fn = |err| eprintln!("Audio stream error: {}", err);

    let input_stream =
        input_device.build_input_stream(&input_config.into(), input_data_fn, err_fn, None)?;
    let output_stream =
        output_device.build_output_stream(&output_config.into(), output_data_fn, err_fn, None)?;

    input_stream.play()?;
    output_stream.play()?;

    println!("\n--- Audio Loopback Running ---");
    println!("Speak into your microphone. You will hear the decoded audio live.");
    println!("SNR configured to: {:.1} dB", snr_db);
    println!("Press [ENTER] to stop.\n");

    let stdin = io::stdin();
    let _ = stdin.lock().read_line(&mut String::new());

    running.store(false, Ordering::SeqCst);
    Ok(())
}
