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