#![no_std]
#![doc = include_str!("../README.md")]

#[cfg(any(feature = "alloc", test))]
extern crate alloc;

pub mod bits;
pub mod constellation;
pub mod demodulation;
pub mod modulation;
pub(crate) mod utils;

// Core constellation types and typestate markers
pub use constellation::{
    Bpsk, Constellation, ConstellationState, Normalized, Psk8, Qam16, Qam64, Qam256, Qam1024,
    Qam4096, Qpsk, Unnormalized,
};

// Fluent extension traits
pub use demodulation::DemodulateExt;
pub use modulation::ModulateExt;

// Bit manipulation markers & traits
pub use bits::{
    BitOrder, ChunkBitsExt, DiscardRemainder, ExactOnly, LsbFirst, MsbFirst, PackBitsExt, PadZeros,
    Padding,
};

// Modulation & demodulation traits
pub use demodulation::{Demodulatable, SoftDemodulatable};
pub use modulation::Modulatable;

/// Common traits and types for convenient glob importing (`use constella::prelude::*;`).
pub mod prelude {
    pub use crate::bits::{
        ChunkBitsExt, DiscardRemainder, ExactOnly, LsbFirst, MsbFirst, PackBitsExt, PadZeros,
    };
    pub use crate::constellation::{
        Bpsk, Constellation, Normalized, Psk8, Qam16, Qam64, Qam256, Qam1024, Qam4096, Qpsk,
        Unnormalized,
    };
    pub use crate::demodulation::DemodulateExt;
    pub use crate::modulation::ModulateExt;
}
