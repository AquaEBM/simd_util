#![feature(portable_simd, stdsimd)]

use core_simd::simd::*;
use std_float::*;
pub mod smoothing;
pub mod simd_util;
pub mod math;
pub mod filter;