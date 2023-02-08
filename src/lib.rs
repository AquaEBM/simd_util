#![feature(core_intrinsics)]
#![feature(portable_simd)]

pub mod dsp;
pub mod util;
pub mod gui;

pub mod parameter;

pub use std::intrinsics::{fadd_fast, fsub_fast, frem_fast, fdiv_fast, fmul_fast};

#[inline(always)]
pub fn add(a: f32, b: f32) -> f32 { unsafe { fadd_fast(a, b) } }
#[inline(always)]
pub fn mul(a: f32, b: f32) -> f32 { unsafe { fmul_fast(a, b)} }
#[inline(always)]
pub fn sub(a: f32, b: f32) -> f32 { unsafe { fsub_fast(a, b) } }
#[inline(always)]
pub fn div(a: f32, b: f32) -> f32 { unsafe { fdiv_fast(a, b) } }
#[inline(always)]
pub fn rem(a: f32, b: f32) -> f32 { unsafe { frem_fast(a, b) } }