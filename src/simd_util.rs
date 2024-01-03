use super::simd;
use simd::{LaneCount, Simd, SimdElement, SupportedLaneCount, Mask, f32x2};

use cfg_if::cfg_if;
use core::mem::{transmute, size_of};

#[cfg(any(target_feature = "avx512f", target_feature = "avx2"))]
use std::arch::x86_64::*;

pub const MAX_VECTOR_WIDTH: usize = {
    if cfg!(any(target_feature = "avx512f")) {
        64
    } else if cfg!(any(target_feature = "avx")) {
        32
    } else if cfg!(any(target_feature = "sse", target_feature = "neon")) {
        16
    } else {
        8
    }
};

pub const FLOATS_PER_VECTOR: usize = MAX_VECTOR_WIDTH / size_of::<f32>();

pub type Float = Simd<f32, FLOATS_PER_VECTOR>;
pub type UInt = Simd<u32, FLOATS_PER_VECTOR>;
pub type TMask = Mask<i32, FLOATS_PER_VECTOR>;

/// Convenience function on simd types when specialized functions aren't
/// available in the standard library, hoping autovectorization compiles this
/// into an simd instruction

#[inline]
pub fn map<T: SimdElement, U: SimdElement, const N: usize>(
    v: Simd<T, N>,
    f: impl FnMut(T) -> U,
) -> Simd<U, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    v.to_array().map(f).into()
}

pub const fn enclosing_div(n: usize, d: usize) -> usize {
    (n + d - 1) / d
}

#[inline]
pub const fn const_splat<T: SimdElement, const N: usize>(item: T) -> Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    Simd::from_array([item; N])
}

// We're using intrinsics for now because u32 gathers aren't in core::simd (yet?)
#[inline]
pub unsafe fn gather_select_unchecked(
    ptr: *const f32,
    index: UInt,
    enable: TMask,
    or: Float,
) -> Float {
    cfg_if! {

        if #[cfg(target_feature = "avx512f")] {

            #[cfg(feature = "non_std_simd")]
            let bitmask = transmute(enable);
            #[cfg(not(feature = "non_std_simd"))]
            let bitmask = enable.to_bitmask() as __mmask16;

            _mm512_mask_i32gather_ps(
                or.into(),
                bitmask,
                index.into(),
                ptr.cast(),
                4,
            ).into()

        } else if #[cfg(target_feature = "avx2")] {

            _mm256_mask_i32gather_ps(
                or.into(),
                ptr,
                index.into(),
                transmute(enable), // Why is this __m256, not __m256i? I don't know
                4
            ).into()

        } else {
            use simd::num::SimdUint;

            let slice = core::slice::from_raw_parts(ptr, 0);
            Simd::gather_select_unchecked(slice, enable.cast(), index.cast(), or)
        }
    }
}

#[inline]
pub unsafe fn gather_unchecked(ptr: *const f32, index: UInt) -> Float {
    cfg_if! {

        if #[cfg(target_feature = "avx512f")] {

            _mm512_i32gather_ps(index.into(), ptr.cast(), 4).into()

        } else if #[cfg(target_feature = "avx2")] {

            _mm256_i32gather_ps(ptr, index.into(), 4).into()

        } else {
            use simd::num::SimdUint;

            let slice = core::slice::from_raw_parts(ptr, 0);
            Simd::gather_select_unchecked(
                slice,
                simd::Mask::splat(true),
                index.cast(),
                Float::splat(0.)
            )
        }
    }
}

#[inline]
pub fn sum_to_stereo_sample(x: Float) -> f32x2 {
    unsafe {
        cfg_if! {

            if #[cfg(any(target_feature = "avx512f"))] {

                // FLOATS_PER_VECTOR = 16
                let [left1, right1]: [Simd<f32, { FLOATS_PER_VECTOR / 2 }> ; 2] = transmute(x);
                let [left2, right2]: [Simd<f32, { FLOATS_PER_VECTOR / 4 }> ; 2] = transmute(left1 + right1);
                let [left3, right3]: [Simd<f32, { FLOATS_PER_VECTOR / 8 }> ; 2] = transmute(left2 + right2);

                left3 + right3

            } else if #[cfg(any(target_feature = "avx"))] {

                // FLOATS_PER_VECTOR = 8
                let [left1, right1]: [Simd<f32, { FLOATS_PER_VECTOR / 2 }> ; 2] = transmute(x);
                let [left2, right2]: [Simd<f32, { FLOATS_PER_VECTOR / 4 }> ; 2] = transmute(left1 + right1);
                left2 + right2

            } else if #[cfg(any(target_feature = "sse", target_feature = "neon"))] {

                // FLOATS_PER_VECTOR = 4
                let [left, right]: [Simd<f32, { FLOATS_PER_VECTOR / 2 }> ; 2] = transmute(x);
                left + right

            } else {

                // FLOATS_PER_VECTOR = 2
                x
            }
        }
    }
}