#![feature(portable_simd, stdsimd)]

cfg_if::cfg_if! {

    if #[cfg(feature = "non_std_simd")] {
    
        pub mod simd {
            pub use core_simd::simd::*;
            pub use std_float::*;
            pub use prelude::*;
        }
    
    } else {

        pub use std::simd;
        
    }
}

use simd::*;

pub mod smoothing;
pub mod simd_util;
pub mod math;
pub mod filter;