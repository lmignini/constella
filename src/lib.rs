#![no_std]
#![doc = include_str!("../README.md")]
#![allow(clippy::unusual_byte_groupings)]

#[cfg(any(feature = "alloc", test))]
extern crate alloc;

pub mod bits;
pub mod constellation;
pub mod demodulation;
pub mod modulation;
pub(crate) mod utils;

pub mod channel;

pub use constellation::{
    Bpsk, Constellation, ConstellationGeometry, ConstellationState, General, Normalized, Psk8,
    Qam16, Qam64, Qam256, Qam1024, Qam4096, Qpsk, SquareQam, Unnormalized,
};

pub use demodulation::DemodulateExt;
pub use modulation::ModulateExt;

pub use bits::{
    BitOrder, ChunkBitsExt, DiscardRemainder, ExactOnly, LsbFirst, MsbFirst, PackBitsExt, PadZeros,
    Padding,
};

pub use demodulation::{Demodulatable, SoftDemodulatable};
pub use modulation::Modulatable;

pub mod prelude {
    pub use crate::bits::{
        ChunkBitsExt, DiscardRemainder, ExactOnly, LsbFirst, MsbFirst, PackBitsExt, PadZeros,
    };
    pub use crate::constellation::{
        Bpsk, Constellation, General, Normalized, Psk8, Qam16, Qam64, Qam256, Qam1024, Qam4096,
        Qpsk, SquareQam, Unnormalized,
    };
    pub use crate::demodulation::DemodulateExt;
    pub use crate::modulation::ModulateExt;
}
