#![cfg(any(feature = "std_simd", feature = "core_simd_crate"))]
#![feature(portable_simd)]
#![cfg_attr(target_feature = "avx512f", feature(stdarch_x86_avx512))]

#[cfg(all(feature = "core_simd_crate", not(feature = "std_simd")))]
pub mod simd {
    pub use core_simd::simd::*;
    pub use std_float::*;
}

#[cfg(feature = "std_simd")]
pub use std::simd;

use simd::{
    LaneCount,
    SupportedLaneCount,
    Simd,
    num::{SimdFloat, SimdUint},
};


pub mod math;
pub mod smoothing;
mod util;
pub use util::*;