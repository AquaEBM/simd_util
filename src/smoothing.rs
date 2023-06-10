use super::*;

pub trait SIMDSmoother<T, const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
    T: SimdElement
{
    fn set_target(&mut self, target: Simd<T, N>, num_samples: usize);
    fn next(&mut self) -> Simd<T, N>;
}

pub trait Smoother<T: SimdElement> {
    fn set_target(&mut self, target: T, num_samples: usize);
    fn next(&mut self) -> T;
}

impl<U: SimdElement, T: SIMDSmoother<U, 1>> Smoother<U> for T {
    fn next(&mut self) -> U {
        self.next()[0]
    }

    fn set_target(&mut self, target: U, num_samples: usize) {
        self.set_target(Simd::from_array([target]), num_samples);
    }
}

pub struct LogSmoother<const N: usize>
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
        Self { factor: Simd::splat(1.), value: Simd::splat(1.) }
    }
}

impl<const N: usize> SIMDSmoother<f32, N> for LogSmoother<N>
where
    LaneCount<N>: SupportedLaneCount
{
    fn set_target(&mut self, target: Simd<f32, N>, num_samples: usize) {
        let base = target / self.value;
        let exp = 1. / num_samples as f32;
        self.factor = util::map(base, |v| v.powf(exp));
    }

    fn next(&mut self) -> Simd<f32, N> {
        self.value *= self.factor;
        self.value
    }
}

#[derive(Default)]
pub struct LinearSmoother<const N: usize>
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

    fn next(&mut self) -> Simd<f32, N> {
        self.value += self.increment;
        self.value
    }
}