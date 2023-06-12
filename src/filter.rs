use std::ops::Deref;

use super::*;

#[derive(Default)]
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
    pub fn process(&mut self, sample: Simd<f32, N>, g: Simd<f32, N>) -> Simd<f32, N> {
        let v = g * sample;
        let output = v + self.s;
        self.s = output + v;
        output
    }

    pub fn reset(&mut self) {
        self.s = Simd::splat(0.);
    }
}

impl<const N: usize> Deref for Integrator<N>
where
    LaneCount<N>: SupportedLaneCount
{
    type Target = Simd<f32, N>;

    fn deref(&self) -> &Self::Target {
        &self.s
    }
}