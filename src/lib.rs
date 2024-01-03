#![feature(portable_simd, stdsimd)]

cfg_if::cfg_if! {

    if #[cfg(feature = "non_std_simd")] {

        pub mod simd {
            pub use core_simd::simd::*;
            pub use std_float::*;
        }

    } else {

        pub use std::simd;

    }
}

pub mod filter;
pub mod math;
pub mod simd_util;
pub mod smoothing;
use simd::prelude::*;
