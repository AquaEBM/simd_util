use core::ops::{Div, SubAssign};

use super::{
    math::{exp2, log2, pow},
    simd::{cmp::SimdOrd, *},
    FLOATS_PER_VECTOR,
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
        self.factor = Self::Value::ONE;
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
    pub increment: Simd<f32, N>,
    pub value: Simd<f32, N>,
}

impl<const N: usize> Smoother for LinearSmoother<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type Value = Simd<f32, N>;

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
        self.increment = Self::Value::ZERO;
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

#[derive(Default, Clone, Copy)]
pub struct Bounded<T: Smoother> {
    pub t: T::Value,
    pub smoother: T,
}

pub trait Zero { const ZERO: Self; }
pub trait One { const ONE: Self; }

impl<const N: usize> Zero for Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    const ZERO: Self = Self::from_array([0. ; N]);
}

impl<const N: usize> One for Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    const ONE: Self = Self::from_array([1. ; N]);
}

impl<const N: usize> Zero for Simd<f64, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    const ZERO: Self = Self::from_array([0. ; N]);
}

impl<const N: usize> One for Simd<f64, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    const ONE: Self = Self::from_array([1. ; N]);
}


impl<T: Smoother> Smoother for Bounded<T>
where
    T::Value: Zero + One + SubAssign + Div<Output = T::Value> + Clone + SimdOrd,
{
    type Value = T::Value;

    #[inline]
    fn set_target(&mut self, target: Self::Value, t: Self::Value) {
        self.smoother.set_target(target, t.clone());
        self.t = t;
    }

    #[inline]
    fn set_target_recip(&mut self, target: Self::Value, t_recip: Self::Value) {
        self.smoother.set_target_recip(target, t_recip.clone());
        self.t = Self::Value::ONE / t_recip;
    }

    #[inline]
    fn set_val_instantly(&mut self, target: Self::Value) {
        self.t = Self::Value::ZERO;
        self.smoother.set_val_instantly(target);
    }

    #[inline]
    fn tick(&mut self, mut t: Self::Value) {
        t = t.clone().simd_min(self.t.clone());
        self.smoother.tick(t.clone());
        self.t -= t;
    }

    #[inline]
    fn get_current(&self) -> Self::Value {
        self.smoother.get_current()
    }
    
    fn tick1(&mut self) {
        self.smoother.tick(Self::Value::ONE)
    }
}

#[derive(Default, Clone, Copy)]
pub struct ExpSmoother<const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    alpha: Simd<f32, N>,
    target: Simd<f32, N>,
    current: Simd<f32, N>,
}

impl<const N: usize> Smoother for ExpSmoother<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type Value = Simd<f32, N>;

    fn set_target(&mut self, target: Self::Value, t: Self::Value) {
        self.target = target;
        // ~= log2(0.001)
        let w = Simd::splat(-9.965_784_285);
        self.alpha = exp2(w / t);
    }

    fn set_target_recip(&mut self, target: Self::Value, t_recip: Self::Value) {
        self.target = target;
        // ~= log2(0.001)
        let w = Simd::splat(-9.965784285);
        self.alpha = exp2(w * t_recip);
    }

    fn set_val_instantly(&mut self, target: Self::Value) {
        self.target = target;
        self.current = target;
    }

    fn tick(&mut self, t: Self::Value) {
        let x = &mut self.current;
        *x = pow(self.alpha, t).mul_add(self.target - *x, *x);
    }

    fn tick1(&mut self) {
        let x = &mut self.current;
        *x = self.alpha.mul_add(self.target - *x, *x);
    }

    fn get_current(&self) -> Self::Value {
        self.current
    }
}
