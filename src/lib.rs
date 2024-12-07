#![feature(portable_simd)]
#![cfg_attr(target_feature = "avx512f", feature(stdarch_x86_avx512))]

use cfg_if::cfg_if;

#[cfg(feature = "core_simd_crate")]
pub mod simd {
    pub use core_simd::simd::*;
    pub use std_float::*;
}

/// reeport of the `simd` module from the standard library, or the  core_simd` crate
#[cfg(feature = "std_simd")]
pub use std::simd;

cfg_if! {
    if #[cfg(any(feature = "std_simd", feature = "core_simd_crate"))] {

        pub mod math;
        pub mod smoothing;
        mod util;
        pub use util::*;

        use simd::{
            LaneCount,
            SupportedLaneCount,
            Simd,
            num::{SimdFloat, SimdUint},
        };
    }
}
