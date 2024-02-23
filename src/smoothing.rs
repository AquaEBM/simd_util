use super::{
    math::{exp2, log2, pow},
    simd::{num::SimdFloat, *},
    FLOATS_PER_VECTOR,
};

pub trait Smoother<const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    fn set_target(&mut self, target: Simd<f32, N>, t: Simd<f32, N>);
    #[inline]
    fn set_target_recip(&mut self, target: Simd<f32, N>, t_recip: Simd<f32, N>) {
        self.set_target(target, t_recip.recip());
    }
    fn set_val_instantly(&mut self, target: Simd<f32, N>);
    fn tick(&mut self, t: Simd<f32, N>);
    #[inline]
    fn tick1(&mut self) {
        self.tick(Simd::splat(1.));
    }
    fn get_current(&self) -> Simd<f32, N>;
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

impl<const N: usize> Smoother<N> for LogSmoother<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn set_target(&mut self, target: Simd<f32, N>, t: Simd<f32, N>) {
        self.factor = exp2(log2(target / self.value) / t);
    }

    #[inline]
    fn set_target_recip(&mut self, target: Simd<f32, N>, t_recip: Simd<f32, N>) {
        self.factor = pow(target / self.value, t_recip);
    }

    #[inline]
    fn set_val_instantly(&mut self, target: Simd<f32, N>) {
        self.value = target;
        self.factor = Simd::splat(1.);
    }

    #[inline]
    fn tick(&mut self, dt: Simd<f32, N>) {
        self.value *= pow(self.factor, dt);
    }

    #[inline]
    fn tick1(&mut self) {
        self.value *= self.factor;
    }

    #[inline]
    fn get_current(&self) -> Simd<f32, N> {
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

impl<const N: usize> Smoother<N> for LinearSmoother<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn set_target(&mut self, target: Simd<f32, N>, t: Simd<f32, N>) {
        self.increment = (target - self.value) / t;
    }

    #[inline]
    fn set_target_recip(&mut self, target: Simd<f32, N>, t_recip: Simd<f32, N>) {
        self.increment = (target - self.value) * t_recip;
    }

    #[inline]
    fn set_val_instantly(&mut self, target: Simd<f32, N>) {
        self.increment = Simd::splat(0.);
        self.value = target;
    }

    #[inline]
    fn tick(&mut self, t: Simd<f32, N>) {
        self.value += self.increment * t;
    }

    #[inline]
    fn tick1(&mut self) {
        self.value += self.increment;
    }

    #[inline]
    fn get_current(&self) -> Simd<f32, N> {
        self.value
    }
}

pub struct Bounded<T, const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    t: Simd<f32, N>,
    pub smoother: T,
}

impl<const N: usize, T: Smoother<N>> Smoother<N> for Bounded<T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn set_target(&mut self, target: Simd<f32, N>, t: Simd<f32, N>) {
        self.smoother.set_target(target, t);
        self.t = t;
    }

    #[inline]
    fn set_target_recip(&mut self, target: Simd<f32, N>, t_recip: Simd<f32, N>) {
        self.smoother.set_target_recip(target, t_recip);
        self.t = t_recip.recip();
    }

    #[inline]
    fn set_val_instantly(&mut self, target: Simd<f32, N>) {
        self.t = Simd::splat(0.);
        self.smoother.set_val_instantly(target);
    }

    #[inline]
    fn tick(&mut self, mut t: Simd<f32, N>) {
        t = t.simd_min(self.t);
        self.smoother.tick(t);
        self.t -= t;
    }

    #[inline]
    fn get_current(&self) -> Simd<f32, N> {
        self.smoother.get_current()
    }
}
