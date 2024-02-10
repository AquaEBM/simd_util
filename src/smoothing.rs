use super::{math::pow, simd::*, simd_util::FLOATS_PER_VECTOR};

pub trait Smoother {
    type Value;

    fn set_increment(&mut self, target: Self::Value, inc: Self::Value);
    fn set_instantly(&mut self, value: Self::Value);
    fn tick(&mut self);
    fn tick_increments(&mut self, inc: Self::Value);
    fn get_current(&self) -> &Self::Value;
}

#[derive(Clone, Copy)]
pub struct LogSmoother<const N: usize = FLOATS_PER_VECTOR>
where
    LaneCount<N>: SupportedLaneCount,
{
    factor: Simd<f32, N>,
    value: Simd<f32, N>,
}

impl<const N: usize> Default for LogSmoother<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    fn default() -> Self {
        Self {
            factor: Simd::splat(1.),
            value: Simd::splat(1.),
        }
    }
}

impl<const N: usize> LogSmoother<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    pub fn get_current_mut(&mut self) -> &mut Simd<f32, N> {
        &mut self.value
    }
}

impl<const N: usize> Smoother for LogSmoother<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type Value = Simd<f32, N>;

    #[inline]
    fn set_increment(&mut self, target: Self::Value, inc: Self::Value) {
        self.factor = pow(target / self.value, inc);
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
    fn tick_increments(&mut self, inc: Self::Value) {
        self.value *= pow(self.factor, inc);
    }

    #[inline]
    fn get_current(&self) -> &Self::Value {
        &self.value
    }
}

#[derive(Default, Clone, Copy)]
pub struct LinearSmoother<const N: usize = FLOATS_PER_VECTOR>
where
    LaneCount<N>: SupportedLaneCount,
{
    increment: Simd<f32, N>,
    value: Simd<f32, N>,
}

impl<const N: usize> LinearSmoother<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    pub fn get_current_mut(&mut self) -> &mut Simd<f32, N> {
        &mut self.value
    }
}

impl<const N: usize> Smoother for LinearSmoother<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type Value = Simd<f32, N>;

    #[inline]
    fn set_increment(&mut self, target: Self::Value, inc: Simd<f32, N>) {
        self.increment = (target - self.value) * inc;
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
    fn tick_increments(&mut self, inc: Self::Value) {
        self.value += self.increment * inc;
    }

    #[inline]
    fn get_current(&self) -> &Self::Value {
        &self.value
    }
}

pub struct CachedTarget<T: Smoother> {
    smoother: T,
    target: T::Value,
}

impl<T: Smoother> Smoother for CachedTarget<T>
where
    T::Value: Clone,
{
    type Value = T::Value;

    #[inline]
    fn set_increment(&mut self, target: Self::Value, inc: Self::Value) {
        self.target = target.clone();
        self.smoother.set_increment(target, inc);
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
    fn tick_increments(&mut self, inc: Self::Value) {
        self.smoother.tick_increments(inc)
    }

    #[inline]
    fn get_current(&self) -> &Self::Value {
        self.smoother.get_current()
    }
}

impl<T: Smoother> CachedTarget<T>
where
    T::Value: Clone,
{
    #[inline]
    pub fn get_target(&self) -> T::Value {
        self.target.clone()
    }
}
