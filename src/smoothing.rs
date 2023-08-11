use std::ops::{Deref, DerefMut};
use super::*;
use simd_util::MAX_VECTOR_WIDTH;
use math::pow;

pub trait SIMDSmoother<T, const N: usize = MAX_VECTOR_WIDTH>
where
    LaneCount<N>: SupportedLaneCount,
    T: SimdElement
{
    fn set_target(&mut self, target: Simd<T, N>, num_samples: usize);
    fn tick(&mut self);
    fn tick_n(&mut self, n: u32);
    fn current(&self) -> &Simd<T, N>;
}

pub trait Smoother<T: SimdElement> {
    fn set_target(&mut self, target: T, num_samples: usize);
    fn tick(&mut self);
    fn tick_n(&mut self, n: u32);
    fn current(&self) -> &T;
}

impl<U: SimdElement, T: SIMDSmoother<U, 1>> Smoother<U> for T {
    fn tick(&mut self) {
        self.tick();
    }

    fn set_target(&mut self, target: U, num_samples: usize) {
        self.set_target(Simd::from_array([target]), num_samples);
    }

    fn tick_n(&mut self, n: u32) {
        self.tick_n(n);
    }

    fn current(&self) -> &U {
        &self.current()[0]
    }
}

#[derive(Clone, Copy)]
pub struct LogSmoother<const N: usize = MAX_VECTOR_WIDTH>
where
    LaneCount<N>: SupportedLaneCount
{
    factor: Simd<f32, N>,
    value: Simd<f32, N>,
}

impl<const N: usize> Default for LogSmoother<N>
where
    LaneCount<N>: SupportedLaneCount
{
    fn default() -> Self {
        Self {
            factor: Simd::splat(1.),
            value: Simd::splat(1.),
        }
    }
}

impl<const N: usize> SIMDSmoother<f32, N> for LogSmoother<N>
where
    LaneCount<N>: SupportedLaneCount
{
    fn set_target(&mut self, target: Simd<f32, N>, num_samples: usize) {
        let base = target / self.value;
        let exp = Simd::splat(1. / num_samples as f32);
        self.factor = pow(base, exp);
    }

    fn tick(&mut self) {
        self.value *= self.factor;
    }

    fn tick_n(&mut self, n: u32) {
        let n = n as i32;
        self.value *= pow(self.factor, Simd::splat(n as f32));
    }

    fn current(&self) -> &Simd<f32, N> {
        &self.value
    }
}

impl<const N: usize> Deref for LogSmoother<N>
where
    LaneCount<N>: SupportedLaneCount
{
    type Target = Simd<f32, N>;

    fn deref(&self) -> &Self::Target {
        self.current()
    }
}

impl<const N: usize> DerefMut for LogSmoother<N>
where
    LaneCount<N>: SupportedLaneCount
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

#[derive(Default, Clone, Copy)]
pub struct LinearSmoother<const N: usize = MAX_VECTOR_WIDTH>
where
    LaneCount<N>: SupportedLaneCount
{
    increment: Simd<f32, N>,
    value: Simd<f32, N>,
}

impl<const N: usize> SIMDSmoother<f32, N> for LinearSmoother<N>
where
    LaneCount<N>: SupportedLaneCount
{
    fn set_target(&mut self, target: Simd<f32, N>, num_samples: usize) {
        self.increment = (target - self.value) / Simd::splat(num_samples as f32);
    }

    fn tick(&mut self) {
        self.value += self.increment;
    }

    fn tick_n(&mut self, n: u32) {
        self.value += self.increment * Simd::splat(n as f32);
    }

    fn current(&self) -> &Simd<f32, N> {
        &self.value
    }
}

impl<const N: usize> Deref for LinearSmoother<N>
where
    LaneCount<N>: SupportedLaneCount
{
    type Target = Simd<f32, N>;

    fn deref(&self) -> &Self::Target {
        self.current()
    }
}

impl<const N: usize> DerefMut for LinearSmoother<N>
where
    LaneCount<N>: SupportedLaneCount
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}