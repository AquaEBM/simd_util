#![cfg(any(feature = "std_simd", feature = "core_simd_crate"))]
#![feature(portable_simd)]

#[cfg(all(feature = "core_simd_crate", not(feature = "std_simd")))]
pub mod simd {
    pub use core_simd::simd::*;
    pub use std_float::*;
}

#[cfg(feature = "std_simd")]
pub use std::simd;

use simd::{
    LaneCount, Simd, SupportedLaneCount,
    num::{SimdFloat, SimdUint},
};

pub mod math;
mod util;
pub use util::*;
