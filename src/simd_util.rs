use super::simd::{*, prelude::*};

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

pub type Float<const N: usize = FLOATS_PER_VECTOR> = Simd<f32, N>;
pub type UInt<const N: usize = FLOATS_PER_VECTOR> = Simd<u32, N>;
pub type TMask<const N: usize = FLOATS_PER_VECTOR> = Mask<i32, N>;

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
    pointer: *const f32,
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
                pointer.cast(),
                4,
            ).into()

        } else if #[cfg(target_feature = "avx2")] {

            _mm256_mask_i32gather_ps(
                or.into(),
                pointer,
                index.into(),
                transmute(enable), // Why is this __m256, not __m256i? I don't know
                4
            ).into()

        } else {

            let slice = core::slice::from_raw_parts(pointer, 0);
            Simd::gather_select_unchecked(slice, enable.cast(), index.cast(), or)
        }
    }
}

#[inline]
pub unsafe fn gather_unchecked(pointer: *const f32, index: UInt) -> Float {
    cfg_if! {

        if #[cfg(target_feature = "avx512f")] {

            _mm512_i32gather_ps(index.into(), pointer.cast(), 4).into()

        } else if #[cfg(target_feature = "avx2")] {

            _mm256_i32gather_ps(pointer, index.into(), 4).into()

        } else {

            let slice = core::slice::from_raw_parts(pointer, 0);
            Simd::gather_select_unchecked(
                slice,
                Mask::splat(true),
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

pub const STEREO_VOICES_PER_VECTOR: usize = FLOATS_PER_VECTOR / 2;

// Safety argument for the following two functions:
//  - both referred to types have the same size, more specifically, 2 * STEREO_VOICES_PER_VECTOR
// is always equal to FLOATS_PER_VECTOR, because it is always a multiple of 2
//  - the type of `vector` has greater alignment that of the return type
//  - the output reference's lifetime is the same as that of the input, so no unbounded lifetimes
//  - we are transmuting a vector to an array over the same scalar, so values are valid

#[inline]
pub fn as_stereo_sample_array<T: SimdElement>(
    vector: &Simd<T, FLOATS_PER_VECTOR>,
) -> &[Simd<T, 2>; STEREO_VOICES_PER_VECTOR] {
    // SAFETY: see above
    unsafe { transmute(vector) }
}

#[inline]
pub fn as_mut_stereo_sample_array<T: SimdElement>(
    vector: &mut Simd<T, FLOATS_PER_VECTOR>,
) -> &mut [Simd<T, 2>; STEREO_VOICES_PER_VECTOR] {
    // SAFETY: see above
    unsafe { transmute(vector) }
}

#[inline]
pub fn splat_stereo<T: SimdElement>(pair: Simd<T, 2>) -> Simd<T, FLOATS_PER_VECTOR> {
    const ZERO_ONE: [usize; FLOATS_PER_VECTOR] = {
        let mut array = [0; FLOATS_PER_VECTOR];
        let mut i = 1;
        while i < FLOATS_PER_VECTOR {
            array[i] = 1;
            i += 2;
        }
        array
    };

    simd_swizzle!(pair, ZERO_ONE)
}

/// return a vector where values on the left channel
/// are on the right and vice-versa
#[inline]
pub fn swap_stereo<T: SimdElement>(v: Simd<T, FLOATS_PER_VECTOR>) -> Simd<T, FLOATS_PER_VECTOR> {
    const FLIP_PAIRS: [usize; FLOATS_PER_VECTOR] = {
        let mut array = [0; FLOATS_PER_VECTOR];

        let mut i = 0;
        while i < FLOATS_PER_VECTOR {
            array[i] = i ^ 1;
            i += 1;
        }
        array
    };

    simd_swizzle!(v, FLIP_PAIRS)
}

/// triangluar panning of a vector of stereo samples, 0 <= pan <= 1
#[inline]
pub fn triangular_pan_weights(pan_norm: Float) -> Float {
    const SIGN_MASK: Float = {
        let mut array = [0.; FLOATS_PER_VECTOR];
        let mut i = 0;
        while i < FLOATS_PER_VECTOR {
            array[i] = -0.;
            i += 2;
        }
        Simd::from_array(array)
    };

    const ALT_ONE: Float = {
        let mut array = [0.; FLOATS_PER_VECTOR];
        let mut i = 0;
        while i < FLOATS_PER_VECTOR {
            array[i] = 1.;
            i += 2;
        }
        Simd::from_array(array)
    };

    Float::from_bits(pan_norm.to_bits() ^ SIGN_MASK.to_bits()) + ALT_ONE
}

#[inline]
pub fn splat_slot<T: SimdElement>(
    vector: &Simd<T, FLOATS_PER_VECTOR>,
    index: usize,
) -> Option<Simd<T, FLOATS_PER_VECTOR>> {
    let array = as_stereo_sample_array(vector);

    array.get(index).copied().map(splat_stereo)
}
