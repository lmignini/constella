#![no_std]
#![doc = include_str!("../README.md")]
#![allow(clippy::unusual_byte_groupings)]

#[cfg(any(feature = "alloc", test))]
extern crate alloc;
pub mod bits;
pub mod channel;
pub mod constellation;
pub mod demodulation;
pub mod differential;
pub mod modulation;
pub mod utils;

pub use bits::{
    BitOrder, ChunkBitsExt, DiscardRemainder, ExactOnly, LsbFirst, MsbFirst, PackBitsExt,
    PadZeros, Padding,
};
pub use channel::{AwgnChannel, ChannelExt, FadingChannel, PhaseDistortion};
pub use constellation::{
    Bpsk, Constellation, ConstellationGeometry, ConstellationState, General, Normalized, Psk8,
    Qam4, Qam16, Qam64, Qam256, Qam1024, Qam4096, Qpsk, RotatedPsk, SquareQam, StandardPsk,
    Unnormalized,
};
pub use demodulation::{
    DemodulateExt, HardByteDemodIter, HardSymbolDemodIter, SoftBitDemodIter, SoftSymbolDemodIter,
};
pub use differential::{DifferentialDecoderIter, DifferentialEncoderIter, DifferentialExt};
pub use modulation::{DirectModulationIter, ModulateExt, ModulationIter};

pub mod prelude {
    pub use crate::bits::{
        ChunkBitsExt, DiscardRemainder, ExactOnly, LsbFirst, MsbFirst, PackBitsExt, PadZeros,
    };
    pub use crate::channel::ChannelExt;
    pub use crate::constellation::{
        Bpsk, Constellation, General, Normalized, Psk8, Qam4, Qam16, Qam64, Qam256, Qam1024,
        Qam4096, Qpsk, RotatedPsk, SquareQam, StandardPsk, Unnormalized,
    };
    pub use crate::demodulation::DemodulateExt;
    pub use crate::differential::DifferentialExt;
    pub use crate::modulation::ModulateExt;
}