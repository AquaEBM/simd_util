use super::*;

pub mod svf;
pub mod one_pole;

#[cfg(feature = "nih_plug")]
use nih_plug::prelude::Enum;

#[derive(Default, Clone, Copy)]
/// Transposed Direct Form II integrator
pub struct Integrator<const N: usize>
where
    LaneCount<N>: SupportedLaneCount
{
    s: Simd<f32, N>
}

impl<const N: usize> Integrator<N>
where
    LaneCount<N>: SupportedLaneCount
{
    pub fn process(&mut self, sample: Simd<f32, N>) -> Simd<f32, N> {
        let output = sample + self.s;
        self.s = output + sample;
        output
    }

    /// Set the internal (`z^-1`) state to `0.0`
    pub fn reset(&mut self) {
        self.s = Simd::splat(0.);
    }
    
    /// get current (z^-1) state
    pub fn get_current(&self) -> Simd<f32, N> { self.s }
}