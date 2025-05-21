use super::*;

use simd::{num::{SimdInt, SimdFloat}, StdFloat};

const MANTISSA_BITS: u32 = f32::MANTISSA_DIGITS - 1;
const ONE_BITS: u32 = 1f32.to_bits();

#[inline]
/// lerp innit
pub fn lerp<const N: usize>(a: Simd<f32, N>, b: Simd<f32, N>, t: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    t.mul_add(b - a, a)
}

/// "Efficient" `tan(x/2)` approximation. Unspecified results if `|x| >= pi`
#[inline]
pub fn tan_half_x<const N: usize>(x: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    // constants
    let n5 = Simd::splat(0.000_066_137_57);
    let n3 = Simd::splat(-0.027_777_778);
    let n1 = Simd::splat(1.);
    let d4 = Simd::splat(0.001_984_127);
    let d2 = Simd::splat(-0.222_222_22);
    let d0 = Simd::splat(2.);

    let x2 = x * x;
    let den = x2.mul_add(x2.mul_add(d4, d2), d0);
    let xden = x / den;
    let num = x2.mul_add(x2.mul_add(n5, n3), n1);

    num * xden
}

/// Returns `2^i` as a `float`.
///
/// Unspecified results if `-126 <= i <= 127` doesn't hold.
#[inline]
pub fn fexp2i<const N: usize>(i: Simd<i32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    Simd::from_bits((i.cast() << MANTISSA_BITS) + Simd::splat(ONE_BITS))
}

/// "Efficient" `exp2` approximation. Unspecified results if `-126 <= v <= 127` doesn't hold.
///
/// # Safety
///
/// `v` must be non-NAN, finite, and in the range `[i32::MIN ; i32::MAX]`
#[inline]
pub unsafe fn exp2<const N: usize>(v: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    // Taylor series already works pretty well here since
    // the polynomial approximation we need is only evaluated in [-0.5, 0.5]
    // (which is small and centered at zero)

    // LN_2^n / n!
    // constants
    let a1 = Simd::splat(core::f32::consts::LN_2);
    let a2 = Simd::splat(0.240_226_5);
    let a3 = Simd::splat(0.055_504_11);
    let a4 = Simd::splat(0.009_618_129);
    let a5 = Simd::splat(0.001_333_355_8);

    // for some reason, v.round() optimizes badly, but this doesn't
    let rounded = map(v, f32::round_ties_even);

    let int = fexp2i(unsafe { rounded.to_int_unchecked() });

    let x = v - rounded; // is always in [-0.5 ; 0.5]

    let y = x.mul_add(x.mul_add(x.mul_add(x.mul_add(a5, a4), a3), a2), a1);
    int.mul_add(x * y, int)
}

/// Returns [`fast_exp2(semitones / 12)`](exp2)
///
/// # Safety
///
/// Same conditions as [`exp2`]
#[inline]
pub unsafe fn semitones_to_ratio<const N: usize>(semitones: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    const RATIO: f32 = 1. / 12.;
    unsafe { exp2(semitones * Simd::splat(RATIO)) }
}

/// Returns `floor(log2(x))` as an `int`. Unspecified results
/// if `x` is `NAN`, `inf` or non-positive.
#[inline]
pub fn ilog2f<const N: usize>(x: Simd<f32, N>) -> Simd<i32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    ((x.to_bits() - Simd::splat(ONE_BITS)) >> MANTISSA_BITS).cast()
}

/// "Efficient" `log2` approximation. Unspecified results if `v` is
/// `NAN`, `inf` or non-positive.
#[inline]
pub fn log2<const N: usize>(v: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    // constants
    let a = Simd::splat(-2.632_416_7);
    let b = Simd::splat(5.);
    let c = Simd::splat(-3.333_333_3);
    let d = Simd::splat(1.428_571_5);
    let e = Simd::splat(-0.333_333_34);
    let f = Simd::splat(0.032_258_064);

    let log_exponent = ilog2f(v).cast();
    let x = Simd::<f32, N>::from_bits(
        v.to_bits() & Simd::splat((1 << MANTISSA_BITS) - 1) | Simd::splat(ONE_BITS),
    );

    let y = x.mul_add(x.mul_add(x.mul_add(x.mul_add(x.mul_add(f, e), d), c), b), a);
    log_exponent + y
}

/// Returns `exp2(log2(base) * exp)`, or, approximately, `base^exp`
/// # Safety
///
/// Same conditions as [`fast_exp2`].
#[inline]
pub unsafe fn pow<const N: usize>(base: Simd<f32, N>, exp: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let exp = log2(base) * exp;
    unsafe { exp2(exp) }
}

#[inline]
pub fn flp_to_fxp<const N: usize>(x: Simd<f32, N>) -> Simd<u32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    const RATIO: f32 = (1u64 << u32::BITS) as f32;
    unsafe { (x * Simd::splat(RATIO)).to_int_unchecked() }
}

#[inline]
pub fn fxp_to_flp<const N: usize>(x: Simd<u32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    const RATIO: f32 = 1. / (1u64 << u32::BITS) as f32;
    x.cast() * Simd::splat(RATIO)
}
