use super::*;
use std::f32::consts::LN_2;

pub fn lerp<const N: usize>(a: Simd<f32, N>, b: Simd<f32, N>, t: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount
{
    (b - a).mul_add(t, a)
}

/// results in undefined behavior if i is not in [-126 ; 126]
pub fn fexp2i<const N: usize>(i: Simd<i32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount
{
    let mantissa_bits = Simd::splat(f32::MANTISSA_DIGITS as i32 - 1);
    let max_finite_exp = Simd::splat(f32::MAX_EXP - 1);
    Simd::from_bits((i + max_finite_exp << mantissa_bits).cast())
}

/// "cheap" 2 ^ x approximation, results in undefined behavior in case of
/// NAN, inf or subnormal numbers, taylor series already works pretty well here since
/// all we need is a polynomial approximation in [-0.5, 0.5]
pub fn exp2<const N: usize>(v: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount
{
    let a = Simd::splat(1.);
    let b = Simd::splat(LN_2);
    let c = Simd::splat(LN_2 * LN_2 / 2.);
    let d = Simd::splat(LN_2 * LN_2 * LN_2 / 6.);
    let e = Simd::splat(LN_2 * LN_2 * LN_2 * LN_2 / 24.);

    let rounded = a.round();

    let int = fexp2i(unsafe { rounded.to_int_unchecked() }); // very cheap

    let x = v - rounded; // is always in [-0.5 ; 0.5]

    let y = x.mul_add(x.mul_add(x.mul_add(x.mul_add(e, d), c), b), a);
    int * y
}

/// Compute floor(log2(x)) as an Int results in undefined behavior
/// when x is NAN, inf or subnormal
pub fn ilog2f<const N: usize>(x: Simd<f32, N>) -> Simd<i32, N>
where
    LaneCount<N>: SupportedLaneCount
{
    let mantissa_bits = Simd::splat(f32::MANTISSA_DIGITS as i32 - 1);
    let max_finite_exp = Simd::splat(f32::MAX_EXP - 1);
    (x.to_bits().cast() >> mantissa_bits) - max_finite_exp
}

/// "cheap" log2 approximation, results in undefined behavior in case of
/// NAN, inf or subnormal numbers. taylor series is terrible here so
/// TODO: find a minimax polynomial and use that
pub fn log2<const N: usize>(v: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount
{
    let c = Simd::splat(-1. / 2. / LN_2);
    let d = Simd::splat(1. / 3. / LN_2);

    let mantissa_mask = Simd::splat(1 << (f32::MANTISSA_DIGITS - 1) - 1);
    let zero_exponent = Simd::splat(1f32.to_bits());

    let log_exponent = ilog2f(v).cast();
    let x = Simd::from_bits(v.to_bits() & mantissa_mask | zero_exponent) - Simd::splat(1f32); 

    let y = (x * x).mul_add(x.mul_add(d, c), x);
    log_exponent + y
}