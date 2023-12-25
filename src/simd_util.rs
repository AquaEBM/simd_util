use super::simd;
use simd::{Simd, SimdElement, LaneCount, SupportedLaneCount};

use std::mem;
use cfg_if::cfg_if;

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

pub const FLOATS_PER_VECTOR: usize = MAX_VECTOR_WIDTH / mem::size_of::<f32>();

pub type Float = Simd<f32, FLOATS_PER_VECTOR>;
pub type UInt = Simd<u32, FLOATS_PER_VECTOR>;
pub type Mask = simd::Mask<i32, FLOATS_PER_VECTOR>;

/// Convenience function on simd types when specialized functions aren't
/// available in the standard library, hoping autovectorization compiles this
/// into an simd instruction

#[inline]
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

// We're using intrinsics for now because u32 gathers aren't in core::simd (yet?)
pub unsafe fn gather_select_unchecked(ptr: *const f32, index: UInt, mask: Mask, or: Float) -> Float {

    cfg_if! {

        if #[cfg(target_feature = "avx512f")] {

            #[cfg(feature = "non_std_simd")]
            let bitmask = mem::transmute(mask);
            #[cfg(not(feature = "non_std_simd"))]
            let bitmask = mask.to_bitmask();

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
                mem::transmute(mask), // Why is this __m256, not __m256i? I don't know
                4
            ).into()

        } else {
            let slice = core::slice::from_raw_parts(ptr, 0);
            Simd::gather_select_unchecked(slice, mask.cast(), simd::SimdUint::cast(index), or)
        }
    }
}

pub unsafe fn gather_unchecked(ptr: *const f32, index: UInt) -> Float {

    cfg_if! {
    
        if #[cfg(target_feature = "avx512f")] {

            _mm512_i32gather_ps(index.into(), ptr.cast(), 4).into()
        
        } else if #[cfg(target_feature = "avx2")] {
        
            _mm256_i32gather_ps(ptr, index.into(), 4).into()
        
        } else {
            let slice = core::slice::from_raw_parts(ptr, 0);
            Simd::gather_select_unchecked(
                slice,
                simd::Mask::splat(true),
                simd::SimdUint::cast(index),
                Float::splat(0.)
            )
        }
    }
}