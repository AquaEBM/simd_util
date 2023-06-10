use super::*;

// convenience function on simd types when specialized functions aren't
// available in the standard library, hoping autovectorization compiles this
// into an simd instruction

pub fn map<T: SimdElement, U: SimdElement, const N: usize>(v: Simd<T, N>, f: impl FnMut(T) -> U) -> Simd<U, N>
where
    LaneCount<N>: SupportedLaneCount
{
    v.to_array().map(f).into()
}

pub const fn enclosing_div(n: usize, d: usize) -> usize {
    (n + d - 1) / d
}

pub const fn const_splat<T: SimdElement, const N: usize>(item: T) -> Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount
{
    Simd::from_array([item ; N])
}