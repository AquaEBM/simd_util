use super::{math::pow, simd::*, simd_util::{FLOATS_PER_VECTOR, Float}};

pub trait Smoother {
    type Value;

    fn set_target(&mut self, target: Self::Value, num_samples: usize);
    fn set_instantly(&mut self, value: Self::Value);
    fn tick(&mut self);
    fn tick_n(&mut self, n: u32);
    fn get_current(&self) -> Self::Value;
}

#[derive(Clone, Copy)]
pub struct LogSmoother<const N: usize = FLOATS_PER_VECTOR>
where
    LaneCount<N>: SupportedLaneCount
{
    factor: Float<N>,
    value: Float<N>,
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

impl<const N: usize> Smoother for LogSmoother<N>
where
    LaneCount<N>: SupportedLaneCount
{
    type Value = Float<N>;

    #[inline]
    fn set_target(&mut self, target: Self::Value, num_samples: usize) {
        let base = target / self.value;
        let exp = Simd::splat(1. / num_samples as f32);
        self.factor = pow(base, exp);
    }

    #[inline]
    fn set_instantly(&mut self, value: Self::Value) {
        self.value = value;
    }

    #[inline]
    fn tick(&mut self) {
        self.value *= self.factor;
    }

    #[inline]
    fn tick_n(&mut self, n: u32) {
        self.value *= pow(self.factor, Simd::splat(n as f32));
    }

    #[inline]
    fn get_current(&self) -> Self::Value {
        self.value
    }
}

#[derive(Default, Clone, Copy)]
pub struct LinearSmoother<const N: usize = FLOATS_PER_VECTOR>
where
    LaneCount<N>: SupportedLaneCount
{
    increment: Float<N>,
    value: Float<N>,
}

impl<const N: usize> Smoother for LinearSmoother<N>
where
    LaneCount<N>: SupportedLaneCount
{
    type Value = Float<N>;

    #[inline]
    fn set_target(&mut self, target: Self::Value, num_samples: usize) {
        self.increment = (target - self.value) / Simd::splat(num_samples as f32);
    }

    #[inline]
    fn set_instantly(&mut self, value: Self::Value) {
        self.value = value;
    }

    #[inline]
    fn tick(&mut self) {
        self.value += self.increment;
    }

    #[inline]
    fn tick_n(&mut self, n: u32) {
        self.value += self.increment * Simd::splat(n as f32);
    }

    #[inline]
    fn get_current(&self) -> Self::Value {
        self.value
    }
}

pub struct CachedTarget<T: Smoother> {
    smoother: T,
    target: T::Value,
}

impl<T: Smoother> Smoother for CachedTarget<T>
where
    T::Value: Clone
{
    type Value = T::Value;

    #[inline]
    fn set_target(&mut self, target: Self::Value, num_samples: usize) {
        self.target = target.clone();
        self.smoother.set_target(target, num_samples);
    }

    #[inline]
    fn set_instantly(&mut self, value: Self::Value) {
        self.smoother.set_instantly(value);
    }

    #[inline]
    fn tick(&mut self) {
        self.smoother.tick()
    }

    #[inline]
    fn tick_n(&mut self, n: u32) {
        self.smoother.tick_n(n)
    }

    #[inline]
    fn get_current(&self) -> Self::Value {
        self.smoother.get_current()
    }
}

impl<T: Smoother> CachedTarget<T>
where
    T::Value: Clone
{   
    #[inline]
    pub fn get_target(&self) -> T::Value {
        self.target.clone()
    }
}