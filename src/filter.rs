#[cfg(feature = "transfer_funcs")]
use ::num::{Float, Complex, One};

#[cfg(feature = "nih_plug")]
use nih_plug::prelude::Enum;

use super::{simd::*, smoothing, math, simd_util::FLOATS_PER_VECTOR};

pub mod svf;
pub mod one_pole;

#[derive(Default, Clone, Copy)]
/// Transposed Direct Form II integrator
pub struct Integrator<const N: usize = FLOATS_PER_VECTOR>
where
    LaneCount<N>: SupportedLaneCount
{
    s: Simd<f32, N>
}

impl<const N: usize> Integrator<N>
where
    LaneCount<N>: SupportedLaneCount
{
    #[inline]
    pub fn process(&mut self, sample: Simd<f32, N>) -> Simd<f32, N> {
        let output = sample + self.s;
        self.s = output + sample;
        output
    }

    #[inline]
    /// Set the internal (`z^-1`) state to `0.0`
    pub fn reset(&mut self) {
        self.s = Simd::splat(0.);
    }
    
    /// get current (z^-1) state
    #[inline]
    pub fn get_current(&self) -> Simd<f32, N> { self.s }
}