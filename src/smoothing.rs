use super::{math::pow, simd::*, FLOATS_PER_VECTOR};

pub trait Smoother {
    type Value;

    fn set_target(&mut self, target: Self::Value, dt: Self::Value);
    fn set_val_instantly(&mut self, value: Self::Value);
    fn tick1(&mut self);
    fn tick(&mut self, k: Self::Value);
    fn get_current(&self) -> &Self::Value;
}

#[derive(Clone, Copy)]
pub struct LogSmoother<const N: usize = FLOATS_PER_VECTOR>
where
    LaneCount<N>: SupportedLaneCount,
{
    pub factor: Simd<f32, N>,
    pub value: Simd<f32, N>,
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

impl<const N: usize> Smoother for LogSmoother<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type Value = Simd<f32, N>;

    #[inline]
    fn set_target(&mut self, target: Self::Value, dt: Self::Value) {
        self.factor = pow(target / self.value, dt);
    }

    #[inline]
    fn set_val_instantly(&mut self, value: Self::Value) {
        self.value = value;
    }

    #[inline]
    fn tick1(&mut self) {
        self.value *= self.factor;
    }

    #[inline]
    fn tick(&mut self, k: Self::Value) {
        self.value *= pow(self.factor, k);
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
    pub increment: Simd<f32, N>,
    pub value: Simd<f32, N>,
}

impl<const N: usize> Smoother for LinearSmoother<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type Value = Simd<f32, N>;

    #[inline]
    fn set_target(&mut self, target: Self::Value, k: Simd<f32, N>) {
        self.increment = (target - self.value) * k;
    }

    #[inline]
    fn set_val_instantly(&mut self, value: Self::Value) {
        self.value = value;
    }

    #[inline]
    fn tick1(&mut self) {
        self.value += self.increment;
    }

    #[inline]
    fn tick(&mut self, dt: Self::Value) {
        self.value += self.increment * dt;
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
    fn set_target(&mut self, target: Self::Value, dt: Self::Value) {
        self.target = target.clone();
        self.smoother.set_target(target, dt);
    }

    #[inline]
    fn set_val_instantly(&mut self, value: Self::Value) {
        self.smoother.set_val_instantly(value);
    }

    #[inline]
    fn tick1(&mut self) {
        self.smoother.tick1()
    }

    #[inline]
    fn tick(&mut self, dt: Self::Value) {
        self.smoother.tick(dt)
    }

    #[inline]
    fn get_current(&self) -> &Self::Value {
        self.smoother.get_current()
    }
}

impl<T: Smoother> CachedTarget<T> {
    #[inline]
    pub fn get_target(&self) -> &T::Value {
        &self.target
    }

    #[inline]
    pub fn inner(&self) -> &T {
        &self.smoother
    }
}

pub struct CachedPrevious<T: Smoother> {
    smoother: T,
    prev_val: T::Value,
}

impl<T: Smoother> Smoother for CachedPrevious<T>
where
    T::Value: Clone,
{
    type Value = T::Value;

    #[inline]
    fn set_target(&mut self, target: Self::Value, dt: Self::Value) {
        self.prev_val = self.smoother.get_current().clone();
        self.smoother.set_target(target, dt);
    }

    #[inline]
    fn set_val_instantly(&mut self, value: Self::Value) {
        self.smoother.set_val_instantly(value);
    }

    #[inline]
    fn tick1(&mut self) {
        self.smoother.tick1()
    }

    #[inline]
    fn tick(&mut self, dt: Self::Value) {
        self.smoother.tick(dt)
    }

    #[inline]
    fn get_current(&self) -> &Self::Value {
        self.smoother.get_current()
    }
}

impl<T: Smoother> CachedPrevious<T> {
    #[inline]
    pub fn get_previous(&self) -> &T::Value {
        &self.prev_val
    }

    #[inline]
    pub fn inner(&self) -> &T {
        &self.smoother
    }
}