use super::{
    math::{exp2, log2, pow},
    simd::*,
    FLOATS_PER_VECTOR,
    Float,
    TMask
};

pub trait Smoother {

    type Value;

    fn set_target(&mut self, target: Self::Value, t: Self::Value);
    fn set_target_recip(&mut self, target: Self::Value, t_recip: Self::Value);
    fn set_val_instantly(&mut self, target: Self::Value);
    fn tick(&mut self, t: Self::Value);
    fn tick1(&mut self);
    fn get_current(&self) -> Self::Value;
}

#[derive(Clone, Copy)]
pub struct LogSmoother<const N: usize = FLOATS_PER_VECTOR>
where
    LaneCount<N>: SupportedLaneCount,
{
    pub factor: Float<N>,
    pub value: Float<N>,
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
    type Value = Float<N>;

    #[inline]
    fn set_target(&mut self, target: Self::Value, t: Self::Value) {
        self.factor = exp2(log2(target / self.value) / t);
    }

    #[inline]
    fn set_target_recip(&mut self, target: Self::Value, t_recip: Self::Value) {
        self.factor = pow(target / self.value, t_recip);
    }

    #[inline]
    fn set_val_instantly(&mut self, target: Self::Value) {
        self.value = target;
        self.factor = Simd::splat(1.0);
    }

    #[inline]
    fn tick(&mut self, dt: Self::Value) {
        self.value *= pow(self.factor, dt);
    }

    #[inline]
    fn tick1(&mut self) {
        self.value *= self.factor;
    }

    #[inline]
    fn get_current(&self) -> Self::Value {
        self.value
    }
}

#[derive(Default, Clone, Copy)]
pub struct LinearSmoother<const N: usize = FLOATS_PER_VECTOR>
where
    LaneCount<N>: SupportedLaneCount,
{
    pub increment: Float<N>,
    pub value: Float<N>,
}

impl<const N: usize> Smoother for LinearSmoother<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type Value = Float<N>;

    #[inline]
    fn set_target(&mut self, target: Self::Value, t: Self::Value) {
        self.increment = (target - self.value) / t;
    }

    #[inline]
    fn set_target_recip(&mut self, target: Self::Value, t_recip: Self::Value) {
        self.increment = (target - self.value) * t_recip;
    }

    #[inline]
    fn set_val_instantly(&mut self, target: Self::Value) {
        self.increment = Simd::splat(0.0);
        self.value = target;
    }

    #[inline]
    fn tick(&mut self, t: Self::Value) {
        self.value += self.increment * t;
    }

    #[inline]
    fn tick1(&mut self) {
        self.value += self.increment;
    }

    #[inline]
    fn get_current(&self) -> Self::Value {
        self.value
    }
}

pub struct GenericSmoother<const N: usize = FLOATS_PER_VECTOR>
where
    LaneCount<N>: SupportedLaneCount,
{
    pub current: Float<N>,
    pub target: Float<N>,
}

impl<const N: usize> GenericSmoother<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    pub fn smooth_exp(&mut self, alpha: Float<N>) {
        let x = &mut self.current;
        *x = alpha.mul_add(self.target - *x, *x);
    }

    #[inline]
    pub fn set_val_instantly(&mut self, target: Float<N>, mask: &TMask<N>) {
        self.target = mask.select(target, self.target);
        self.current = mask.select(target, self.current);
    }

    #[inline]
    pub fn set_target(&mut self, target: Float<N>, mask: &TMask<N>) {
        self.target = mask.select(target, self.target);
    }
}