use std::ops::Deref;
use super::*;

pub mod svf;
pub mod one_pole;

#[derive(Default, Clone, Copy)]
/// Transposed Direct Form II integrator, dereference to get internal (`z^-1`) state
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
    /// Process the integrator cirucit with the given pre-gain.
    /// 
    /// `g` is usually `cutoff / (2 * sample_rate)` (unprewarped), or
    /// `tan(PI / sample_rate * cutoff)` (prewarped) in most filter types
    pub fn process(&mut self, sample: Simd<f32, N>, g: Simd<f32, N>) -> Simd<f32, N> {
        let v = g * sample;
        let output = v + self.s;
        self.s = output + v;
        output
    }

    /// Set the internal (`z^-1`) state to `0.0`
    pub fn reset(&mut self) {
        self.s = Simd::splat(0.);
    }
}

impl<const N: usize> Deref for Integrator<N>
where
    LaneCount<N>: SupportedLaneCount
{
    type Target = Simd<f32, N>;

    /// internal (`z^-1`) state
    fn deref(&self) -> &Self::Target {
        &self.s
    }
}

