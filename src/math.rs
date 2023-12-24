use super::*;
use std::f32::consts::LN_2;

pub fn lerp<const N: usize>(a: Simd<f32, N>, b: Simd<f32, N>, t: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount
{
    (b - a).mul_add(t, a)
}

/// Unspecified results for i not in [-126 ; 126]
pub fn fexp2i<const N: usize>(i: Simd<i32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount
{
    let mantissa_bits = Simd::splat(f32::MANTISSA_DIGITS as i32 - 1);
    let max_finite_exp = Simd::splat(f32::MAX_EXP - 1);
    Simd::from_bits((i + max_finite_exp << mantissa_bits).cast())
}

/// "cheap" 2 ^ x approximation, Unspecified results if v is
/// NAN, inf or subnormal. Taylor series already works pretty well here since
/// the polynomial approximation we need here is in the interval (-0.5, 0.5)
/// (which is small and centered at zero)
pub fn exp2<const N: usize>(v: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount
{
    let a = Simd::splat(1.);
    let b = Simd::splat(LN_2);
    let c = Simd::splat(LN_2 * LN_2 / 2.);
    let d = Simd::splat(LN_2 * LN_2 * LN_2 / 6.);
    let e = Simd::splat(LN_2 * LN_2 * LN_2 * LN_2 / 24.);
    let f = Simd::splat(LN_2 * LN_2 * LN_2 * LN_2 * LN_2 / 120.);

    let rounded = v.round();

    let int = fexp2i(unsafe { rounded.to_int_unchecked() }); // very cheap

    let x = v - rounded; // is always in [-0.5 ; 0.5]

    let y = x.mul_add(x.mul_add(x.mul_add(x.mul_add(x.mul_add(f, e), d), c), b), a);
    int * y
}

/// Compute floor(log2(x)) as an Int. Unspecified results
/// if x is NAN, inf or subnormal
pub fn ilog2f<const N: usize>(x: Simd<f32, N>) -> Simd<i32, N>
where
    LaneCount<N>: SupportedLaneCount
{
    let mantissa_bits = Simd::splat(f32::MANTISSA_DIGITS as i32 - 1);
    let max_finite_exp = Simd::splat(f32::MAX_EXP - 1);
    (x.to_bits().cast() >> mantissa_bits) - max_finite_exp
}

/// "cheap" log2 approximation. Unspecified results is v is
/// NAN, inf or subnormal.
pub fn log2<const N: usize>(v: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount
{
    let a = Simd::splat(-1819.0 / 651.0);
    let b = Simd::splat(5.0);
    let c = Simd::splat(-10.0 / 3.0);
    let d = Simd::splat(10.0 / 7.0);
    let e = Simd::splat(-1.0 / 3.0);
    let f = Simd::splat(1.0 / 31.0);

    let mantissa_mask = Simd::splat((1 << (f32::MANTISSA_DIGITS - 1)) - 1);
    let zero_exponent = Simd::splat(1f32.to_bits());

    let log_exponent = ilog2f(v).cast();
    let x = Simd::<f32, N>::from_bits(v.to_bits() & mantissa_mask | zero_exponent); 

    let y = x.mul_add(x.mul_add(x.mul_add(x.mul_add(x.mul_add(f, e), d), c), b), a);
    log_exponent + y
}

pub fn pow<const N: usize>(base: Simd<f32, N>, exp: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount
{
    exp2(log2(base) * exp)
}

pub fn flp_to_fxp<const N: usize>(x: Simd<f32, N>) -> Simd<u32, N>
where
    LaneCount<N>: SupportedLaneCount
{
    let exponent_addend = Simd::splat(u32::BITS << (f32::MANTISSA_DIGITS - 1));
    unsafe { Simd::<f32, N>::from_bits(x.to_bits() + exponent_addend).to_int_unchecked() }
}

pub fn fxp_to_flp<const N: usize>(x: Simd<u32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount
{
    let exponent_subtrahend = Simd::splat(u32::BITS << (f32::MANTISSA_DIGITS - 1));
    Simd::from_bits(x.cast() - exponent_subtrahend)
}